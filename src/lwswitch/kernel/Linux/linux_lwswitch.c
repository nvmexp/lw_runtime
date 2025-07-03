/*******************************************************************************
    Copyright (c) 2016-2021 LWPU Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/
#include "linux_lwswitch.h"

#include <linux/version.h>

#include "conftest.h"
#include "lwlink_errors.h"
#include "lwlink_linux.h"
#include "lwCpuUuid.h"
#include "lw-time.h"
#include "lwlink_caps.h"

#include <linux/module.h>
#include <linux/interrupt.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/poll.h>
#include <linux/sched.h>
#include <linux/time.h>
#include <linux/string.h>
#include <linux/moduleparam.h>
#include <linux/ctype.h>
#include <linux/wait.h>
#include <linux/jiffies.h>

#include "ioctl_lwswitch.h"

const static struct
{
    LwlStatus status;
    int err;
} lwswitch_status_map[] = {
    { LWL_ERR_GENERIC,                  -EIO        },
    { LWL_NO_MEM,                       -ENOMEM     },
    { LWL_BAD_ARGS,                     -EILWAL     },
    { LWL_ERR_ILWALID_STATE,            -EIO        },
    { LWL_ERR_NOT_SUPPORTED,            -EOPNOTSUPP },
    { LWL_NOT_FOUND,                    -EILWAL     },
    { LWL_ERR_STATE_IN_USE,             -EBUSY      },
    { LWL_ERR_NOT_IMPLEMENTED,          -ENOSYS     },
    { LWL_ERR_INSUFFICIENT_PERMISSIONS, -EPERM      },
    { LWL_ERR_OPERATING_SYSTEM,         -EIO        },
    { LWL_MORE_PROCESSING_REQUIRED,     -EAGAIN     },
    { LWL_SUCCESS,                       0          },
};

int
lwswitch_map_status
(
    LwlStatus status
)
{
    int err = -EIO;
    LwU32 i;
    LwU32 limit = sizeof(lwswitch_status_map) / sizeof(lwswitch_status_map[0]);

    for (i = 0; i < limit; i++)
    {
        if (lwswitch_status_map[i].status == status ||
            lwswitch_status_map[i].status == -status)
        {
            err = lwswitch_status_map[i].err;
            break;
        }
    }

    return err;
}

#if !defined(IRQF_SHARED)
#define IRQF_SHARED SA_SHIRQ
#endif

#define LW_FILE_INODE(file) (file)->f_inode

static int lwswitch_probe(struct pci_dev *, const struct pci_device_id *);
static void lwswitch_remove(struct pci_dev *);

static struct pci_device_id lwswitch_pci_table[] =
{
    {
        .vendor      = PCI_VENDOR_ID_LWIDIA,
        .device      = PCI_ANY_ID,
        .subvendor   = PCI_ANY_ID,
        .subdevice   = PCI_ANY_ID,
        .class       = (PCI_CLASS_BRIDGE_OTHER << 8),
        .class_mask  = ~0
    },
    {}
};

static struct pci_driver lwswitch_pci_driver =
{
    .name           = LWSWITCH_DRIVER_NAME,
    .id_table       = lwswitch_pci_table,
    .probe          = lwswitch_probe,
    .remove         = lwswitch_remove,
    .shutdown       = lwswitch_remove
};

//
// lwidia_lwswitch_mknod uses minor number 255 to create lwpu-lwswitchctl
// node. Hence, if LWSWITCH_CTL_MINOR is changed, then LW_LWSWITCH_CTL_MINOR
// should be updated. See lwdia-modprobe-utils.h
//
#define LWSWITCH_CTL_MINOR 255
#define LWSWITCH_MINOR_COUNT (LWSWITCH_CTL_MINOR + 1)

// 32 bit hex value - including 0x prefix. (10 chars)
#define LWSWITCH_REGKEY_VALUE_LEN 10

static char *LwSwitchRegDwords;
module_param(LwSwitchRegDwords, charp, 0);
MODULE_PARM_DESC(LwSwitchRegDwords, "LwSwitch regkey");

static char *LwSwitchBlacklist;
module_param(LwSwitchBlacklist, charp, 0);
MODULE_PARM_DESC(LwSwitchBlacklist, "LwSwitchBlacklist=uuid[,uuid...]");

//
// Locking:
//   We handle lwswitch driver locking in the OS layer. The lwswitch lib
//   layer does not have its own locking. It relies on the OS layer for
//   atomicity.
//
//   All locking is done with sleep locks. We use threaded MSI interrupts to
//   facilitate this.
//
//   When handling a request from a user context we use the interruptible
//   version to enable a quick ^C return if there is lock contention.
//
//   lwswitch.driver_mutex is used to protect driver's global state, "struct
//   LWSWITCH". The driver_mutex is taken during .probe, .remove, .open,
//   .close, and lwswitch-ctl .ioctl operations.
//
//   lwswitch_dev.device_mutex is used to protect per-device state, "struct
//   LWSWITCH_DEV", once a device is opened. The device_mutex is taken during
//   .ioctl, .poll and other background tasks.
//
//   The kernel guarantees that .close won't happen while .ioctl and .poll
//   are going on and without successful .open one can't execute any file ops.
//   This behavior guarantees correctness of the locking model.
//
//   If .close is ilwoked and holding the lock which is also used by threaded
//   tasks such as interrupt, driver will deadlock while trying to stop such
//   tasks. For example, when threaded interrupts are enabled, free_irq() calls
//   kthread_stop() to flush pending interrupt tasks. The locking model
//   makes sure that such deadlock cases don't happen.
//
// Lock ordering:
//   lwswitch.driver_mutex
//   lwswitch_dev.device_mutex
//
// Note:
//   Due to bug 2856314, lwswitch_dev.device_mutex is taken when calling
//   lwswitch_post_init_device() in lwswitch_probe().
//

// Per-chip driver state is defined in linux_lwswitch.h

// Global driver state
typedef struct
{
    LwBool initialized;
    struct cdev cdev;
    struct cdev cdev_ctl;
    dev_t devno;
    atomic_t count;
    struct mutex driver_mutex;
    struct list_head devices;
} LWSWITCH;

static LWSWITCH lwswitch = {0};

// LwSwitch event
typedef struct lwswitch_event_t
{
    wait_queue_head_t wait_q_event;
    LwBool            event_pending;
} lwswitch_event_t;

typedef struct lwswitch_file_private
{
    LWSWITCH_DEV     *lwswitch_dev;
    lwswitch_event_t file_event;
    struct
    {
        /* A duped file descriptor for fabric_mgmt capability */
        int fabric_mgmt;
    } capability_fds;
} lwswitch_file_private_t;

#define LWSWITCH_SET_FILE_PRIVATE(filp, data) ((filp)->private_data = (data))
#define LWSWITCH_GET_FILE_PRIVATE(filp) ((lwswitch_file_private_t *)(filp)->private_data)

static int lwswitch_device_open(struct inode *inode, struct file *file);
static int lwswitch_device_release(struct inode *inode, struct file *file);
static unsigned int lwswitch_device_poll(struct file *file, poll_table *wait);
static int lwswitch_device_ioctl(struct inode *inode,
                                 struct file *file,
                                 unsigned int cmd,
                                 unsigned long arg);
static long lwswitch_device_unlocked_ioctl(struct file *file,
                                           unsigned int cmd,
                                           unsigned long arg);

static int lwswitch_ctl_ioctl(struct inode *inode,
                              struct file *file,
                              unsigned int cmd,
                              unsigned long arg);
static long lwswitch_ctl_unlocked_ioctl(struct file *file,
                                        unsigned int cmd,
                                        unsigned long arg);

struct file_operations device_fops =
{
    .owner = THIS_MODULE,
#if defined(LW_FILE_OPERATIONS_HAS_IOCTL)
    .ioctl = lwswitch_device_ioctl,
#endif
    .unlocked_ioctl = lwswitch_device_unlocked_ioctl,
    .open    = lwswitch_device_open,
    .release = lwswitch_device_release,
    .poll    = lwswitch_device_poll
};

struct file_operations ctl_fops =
{
    .owner = THIS_MODULE,
#if defined(LW_FILE_OPERATIONS_HAS_IOCTL)
    .ioctl = lwswitch_ctl_ioctl,
#endif
    .unlocked_ioctl = lwswitch_ctl_unlocked_ioctl,
};

static int lwswitch_initialize_device_interrupt(LWSWITCH_DEV *lwswitch_dev);
static void lwswitch_shutdown_device_interrupt(LWSWITCH_DEV *lwswitch_dev);
static void lwswitch_load_bar_info(LWSWITCH_DEV *lwswitch_dev);
static void lwswitch_task_dispatch(LWSWITCH_DEV *lwswitch_dev);

static LwBool
lwswitch_is_device_blacklisted
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    LWSWITCH_DEVICE_FABRIC_STATE device_fabric_state = 0;
    LwlStatus status;

    status = lwswitch_lib_read_fabric_state(lwswitch_dev->lib_device, 
                                            &device_fabric_state, NULL, NULL);

    if (status != LWL_SUCCESS)
    {
        printk(KERN_INFO "%s: Failed to read fabric state, %x\n", lwswitch_dev->name, status);
        return LW_FALSE;
    }

    return device_fabric_state == LWSWITCH_DEVICE_FABRIC_STATE_BLACKLISTED;
}

static void
lwswitch_deinit_background_tasks
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    LW_ATOMIC_SET(lwswitch_dev->task_q_ready, 0);

    wake_up(&lwswitch_dev->wait_q_shutdown);

    lw_kthread_q_stop(&lwswitch_dev->task_q);
}

static int
lwswitch_init_background_tasks
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    int rc;

    rc = lw_kthread_q_init(&lwswitch_dev->task_q, lwswitch_dev->sname);
    if (rc)
    {
        printk(KERN_ERR "%s: Failed to create task queue\n", lwswitch_dev->name);
        return rc;
    }

    LW_ATOMIC_SET(lwswitch_dev->task_q_ready, 1);

    lw_kthread_q_item_init(&lwswitch_dev->task_item,
                           (lw_q_func_t) &lwswitch_task_dispatch,
                           lwswitch_dev);

    if (!lw_kthread_q_schedule_q_item(&lwswitch_dev->task_q,
                                      &lwswitch_dev->task_item))
    {
        printk(KERN_ERR "%s: Failed to schedule an item\n",lwswitch_dev->name);
        rc = -ENODEV;
        goto init_background_task_failed;
    }

    return 0;

init_background_task_failed:
    lwswitch_deinit_background_tasks(lwswitch_dev);

    return rc;
}

static LWSWITCH_DEV*
lwswitch_find_device(int minor)
{
    struct list_head *lwr;
    LWSWITCH_DEV *lwswitch_dev = NULL;

    list_for_each(lwr, &lwswitch.devices)
    {
        lwswitch_dev = list_entry(lwr, LWSWITCH_DEV, list_node);
        if (lwswitch_dev->minor == minor)
        {
            return lwswitch_dev;
        }
    }

    return NULL;
}

static int
lwswitch_find_minor(void)
{
    struct list_head *lwr;
    LWSWITCH_DEV *lwswitch_dev;
    int minor;
    int minor_in_use;

    for (minor = 0; minor < LWSWITCH_DEVICE_INSTANCE_MAX; minor++)
    {
        minor_in_use = 0;

        list_for_each(lwr, &lwswitch.devices)
        {
            lwswitch_dev = list_entry(lwr, LWSWITCH_DEV, list_node);
            if (lwswitch_dev->minor == minor)
            {
                minor_in_use = 1;
                break;
            }
        }

        if (!minor_in_use)
        {
            return minor;
        }
    }

    return LWSWITCH_DEVICE_INSTANCE_MAX;
}

static int
lwswitch_init_i2c_adapters
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    LwlStatus retval;
    LwU32 i, valid_ports_mask;
    struct i2c_adapter *adapter;
    lwswitch_i2c_adapter_entry *adapter_entry;

    if (!lwswitch_lib_is_i2c_supported(lwswitch_dev->lib_device))
    {
        return 0;
    }

    retval = lwswitch_lib_get_valid_ports_mask(lwswitch_dev->lib_device,
                                               &valid_ports_mask);
    if (retval != LWL_SUCCESS)
    {
        printk(KERN_ERR "Failed to get valid I2C ports mask.\n");
        return -ENODEV;
    }

    FOR_EACH_INDEX_IN_MASK(32, i, valid_ports_mask)
    {
        adapter = lwswitch_i2c_add_adapter(lwswitch_dev, i);
        if (adapter == NULL)
        {
            continue;
        }

        adapter_entry = lwswitch_os_malloc(sizeof(*adapter_entry));
        if (adapter_entry == NULL)
        {
            printk(KERN_ERR "Failed to create I2C adapter entry.\n");
            lwswitch_i2c_del_adapter(adapter);
            continue;
        }

        adapter_entry->adapter = adapter;

        list_add_tail(&adapter_entry->entry, &lwswitch_dev->i2c_adapter_list);
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return 0;
}

static void
lwswitch_deinit_i2c_adapters
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    lwswitch_i2c_adapter_entry *lwrr;
    lwswitch_i2c_adapter_entry *next;

    list_for_each_entry_safe(lwrr,
                             next,
                             &lwswitch_dev->i2c_adapter_list,
                             entry)
    {
        lwswitch_i2c_del_adapter(lwrr->adapter);
        list_del(&lwrr->entry);
        lwswitch_os_free(lwrr);
    }
}

static int
lwswitch_init_device
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    struct pci_dev *pci_dev = lwswitch_dev->pci_dev;
    LwlStatus retval;
    int rc;

    INIT_LIST_HEAD(&lwswitch_dev->i2c_adapter_list);

    retval = lwswitch_lib_register_device(LW_PCI_DOMAIN_NUMBER(pci_dev),
                                          LW_PCI_BUS_NUMBER(pci_dev),
                                          LW_PCI_SLOT_NUMBER(pci_dev),
                                          PCI_FUNC(pci_dev->devfn),
                                          pci_dev->device,
                                          pci_dev,
                                          lwswitch_dev->minor,
                                          &lwswitch_dev->lib_device);
    if (LWL_SUCCESS != retval)
    {
        printk(KERN_ERR "%s: Failed to register device : %d\n",
               lwswitch_dev->name,
               retval);
        return -ENODEV;
    }

    lwswitch_load_bar_info(lwswitch_dev);

    retval = lwswitch_lib_initialize_device(lwswitch_dev->lib_device);
    if (LWL_SUCCESS != retval)
    {
        printk(KERN_ERR "%s: Failed to initialize device : %d\n",
               lwswitch_dev->name,
               retval);
        rc = -ENODEV;
        goto init_device_failed;
    }

    lwswitch_lib_get_uuid(lwswitch_dev->lib_device, &lwswitch_dev->uuid);

    if (lwswitch_lib_get_bios_version(lwswitch_dev->lib_device,
                                      &lwswitch_dev->bios_ver) != LWL_SUCCESS)
    {
        lwswitch_dev->bios_ver = 0;
    }

    if (lwswitch_lib_get_physid(lwswitch_dev->lib_device,
                                &lwswitch_dev->phys_id) != LWL_SUCCESS)
    {
        lwswitch_dev->phys_id = LWSWITCH_ILWALID_PHYS_ID;
    }

    rc = lwswitch_initialize_device_interrupt(lwswitch_dev);
    if (rc)
    {
        printk(KERN_ERR "%s: Failed to initialize interrupt : %d\n",
               lwswitch_dev->name,
               rc);
        goto init_intr_failed;
    }

    if (lwswitch_is_device_blacklisted(lwswitch_dev))
    {
        printk(KERN_ERR "%s: Blacklisted lwswitch device\n", lwswitch_dev->name);
        // Keep device registered for HAL access and Fabric State updates
        return 0;
    }

    lwswitch_lib_enable_interrupts(lwswitch_dev->lib_device);

    return 0;

init_intr_failed:
    lwswitch_lib_shutdown_device(lwswitch_dev->lib_device);

init_device_failed:
    lwswitch_lib_unregister_device(lwswitch_dev->lib_device);
    lwswitch_dev->lib_device = NULL;

    return rc;
}

static int
lwswitch_post_init_device
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    int rc;
    LwlStatus retval;

    rc = lwswitch_init_i2c_adapters(lwswitch_dev);
    if (rc < 0)
    {
       return rc;
    }

    retval = lwswitch_lib_post_init_device(lwswitch_dev->lib_device);
    if (retval != LWL_SUCCESS)
    {
        return -ENODEV;
    }

    return 0;
}

static void
lwswitch_post_init_blacklisted
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    lwswitch_lib_post_init_blacklist_device(lwswitch_dev->lib_device);
}

static void
lwswitch_deinit_device
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    lwswitch_lib_disable_interrupts(lwswitch_dev->lib_device);

    lwswitch_shutdown_device_interrupt(lwswitch_dev);

    lwswitch_lib_shutdown_device(lwswitch_dev->lib_device);

    lwswitch_lib_unregister_device(lwswitch_dev->lib_device);
    lwswitch_dev->lib_device = NULL;
}

static void
lwswitch_init_file_event
(
    lwswitch_file_private_t *private
)
{
    init_waitqueue_head(&private->file_event.wait_q_event);
    private->file_event.event_pending = LW_FALSE;
}

//
// Basic device open to support IOCTL interface
//
static int
lwswitch_device_open
(
    struct inode *inode,
    struct file *file
)
{
    LWSWITCH_DEV *lwswitch_dev;
    int rc = 0;
    lwswitch_file_private_t *private = NULL;

    //
    // Get the major/minor device
    // We might want this for routing requests to multiple lwswitches
    //
    printk(KERN_INFO "lwpu-lwswitch%d: open (major=%d)\n",
           MINOR(inode->i_rdev),
           MAJOR(inode->i_rdev));

    rc = mutex_lock_interruptible(&lwswitch.driver_mutex);
    if (rc)
    {
        return rc;
    }

    lwswitch_dev = lwswitch_find_device(MINOR(inode->i_rdev));
    if (!lwswitch_dev)
    {
        rc = -ENODEV;
        goto done;
    }

    if (lwswitch_is_device_blacklisted(lwswitch_dev))
    {
        rc = -ENODEV;
        goto done;
    }

    private = lwswitch_os_malloc(sizeof(*private));
    if (private == NULL)
    {
        rc = -ENOMEM;
        goto done;
    }

    private->lwswitch_dev = lwswitch_dev;

    lwswitch_init_file_event(private);

    private->capability_fds.fabric_mgmt = -1;
    LWSWITCH_SET_FILE_PRIVATE(file, private);

    LW_ATOMIC_INC(lwswitch_dev->ref_count);

done:
    mutex_unlock(&lwswitch.driver_mutex);

    return rc;
}

//
// Basic device release to support IOCTL interface
//
static int
lwswitch_device_release
(
    struct inode *inode,
    struct file *file
)
{
    lwswitch_file_private_t *private = LWSWITCH_GET_FILE_PRIVATE(file);
    LWSWITCH_DEV *lwswitch_dev = private->lwswitch_dev;

    printk(KERN_INFO "lwpu-lwswitch%d: release (major=%d)\n",
           MINOR(inode->i_rdev),
           MAJOR(inode->i_rdev));

    mutex_lock(&lwswitch.driver_mutex);

    lwswitch_lib_remove_client_events(lwswitch_dev->lib_device, (void *)private);

    //
    // If there are no outstanding references and the device is marked
    // unusable, free it.
    //
    if (LW_ATOMIC_DEC_AND_TEST(lwswitch_dev->ref_count) &&
        lwswitch_dev->unusable)
    {
        kfree(lwswitch_dev);
    }

    if (private->capability_fds.fabric_mgmt > 0)
    {
        lwlink_cap_release(private->capability_fds.fabric_mgmt);
        private->capability_fds.fabric_mgmt = -1;
    }

    lwswitch_os_free(file->private_data);
    LWSWITCH_SET_FILE_PRIVATE(file, NULL);

    mutex_unlock(&lwswitch.driver_mutex);

    return 0;
}

static unsigned int
lwswitch_device_poll
(
    struct file *file,
    poll_table *wait
)
{
    lwswitch_file_private_t *private = LWSWITCH_GET_FILE_PRIVATE(file);
    LWSWITCH_DEV *lwswitch_dev = private->lwswitch_dev;
    int rc = 0;
    LwlStatus status;
    struct LWSWITCH_CLIENT_EVENT *client_event;

    rc = mutex_lock_interruptible(&lwswitch_dev->device_mutex);
    if (rc)
    {
        return rc;
    }

    if (lwswitch_dev->unusable)
    {
        printk(KERN_INFO "%s: a stale fd detected\n", lwswitch_dev->name);
        rc = POLLHUP;
        goto done;
    }

    status = lwswitch_lib_get_client_event(lwswitch_dev->lib_device,
                                           (void *) private, &client_event);
    if (status != LWL_SUCCESS)
    {
        printk(KERN_INFO "%s: no events registered for fd\n", lwswitch_dev->name);
        rc = POLLERR;
        goto done;
    }

    poll_wait(file, &private->file_event.wait_q_event, wait);

    if (private->file_event.event_pending)
    {
        rc = POLLPRI | POLLIN;
        private->file_event.event_pending = LW_FALSE;
    }

done:
    mutex_unlock(&lwswitch_dev->device_mutex);

    return rc;
}

typedef struct {
    void *kernel_params;                // Kernel copy of ioctl parameters
    unsigned long kernel_params_size;   // Size of ioctl params according to user
} IOCTL_STATE;

//
// Clean up any dynamically allocated memory for ioctl state
//
static void
lwswitch_ioctl_state_cleanup
(
    IOCTL_STATE *state
)
{
    kfree(state->kernel_params);
    state->kernel_params = NULL;
}

//
// Initialize buffer state for ioctl.
//
// This handles allocating memory and copying user data into kernel space.  The
// ioctl params structure only is supported. Nested data pointers are not handled.
//
// State is maintained in the IOCTL_STATE struct for use by the ioctl, _sync and
// _cleanup calls.
//
static int
lwswitch_ioctl_state_start(IOCTL_STATE *state, int cmd, unsigned long user_arg)
{
    int rc;

    state->kernel_params = NULL;
    state->kernel_params_size = _IOC_SIZE(cmd);

    if (0 == state->kernel_params_size)
    {
        return 0;
    }

    state->kernel_params = kzalloc(state->kernel_params_size, GFP_KERNEL);
    if (NULL == state->kernel_params)
    {
        rc = -ENOMEM;
        goto lwswitch_ioctl_state_start_fail;
    }

    // Copy params to kernel buffers.  Simple _IOR() ioctls can skip this step.
    if (_IOC_DIR(cmd) & _IOC_WRITE)
    {
        rc = copy_from_user(state->kernel_params,
                            (const void *)user_arg,
                            state->kernel_params_size);
        if (rc)
        {
            rc = -EFAULT;
            goto lwswitch_ioctl_state_start_fail;
        }
    }

    return 0;

lwswitch_ioctl_state_start_fail:
    lwswitch_ioctl_state_cleanup(state);
    return rc;
}

//
// Synchronize any ioctl output in the kernel buffers to the user mode buffers.
//
static int
lwswitch_ioctl_state_sync
(
    IOCTL_STATE *state,
    int cmd,
    unsigned long user_arg
)
{
    int rc;

    // Nothing to do if no buffer or write-only ioctl
    if ((0 == state->kernel_params_size) || (0 == (_IOC_DIR(cmd) & _IOC_READ)))
    {
        return 0;
    }

    // Copy params structure back to user mode
    rc = copy_to_user((void *)user_arg,
                      state->kernel_params,
                      state->kernel_params_size);
    if (rc)
    {
        rc = -EFAULT;
    }

    return rc;
}

static int
lwswitch_device_ioctl
(
    struct inode *inode,
    struct file *file,
    unsigned int cmd,
    unsigned long arg
)
{
    lwswitch_file_private_t *private = LWSWITCH_GET_FILE_PRIVATE(file);
    LWSWITCH_DEV *lwswitch_dev = private->lwswitch_dev;
    IOCTL_STATE state = {0};
    LwlStatus retval;
    int rc = 0;

    if (_IOC_TYPE(cmd) != LWSWITCH_DEV_IO_TYPE)
    {
        return -EILWAL;
    }

    rc = mutex_lock_interruptible(&lwswitch_dev->device_mutex);
    if (rc)
    {
        return rc;
    }

    if (lwswitch_dev->unusable)
    {
        printk(KERN_INFO "%s: a stale fd detected\n", lwswitch_dev->name);
        rc = -ENODEV;
        goto lwswitch_device_ioctl_exit;
    }

    if (lwswitch_is_device_blacklisted(lwswitch_dev))
    {
        printk(KERN_INFO "%s: ioctl attempted on blacklisted device\n", lwswitch_dev->name);
        rc = -ENODEV;
        goto lwswitch_device_ioctl_exit;
    }

    rc = lwswitch_ioctl_state_start(&state, cmd, arg);
    if (rc)
    {
        goto lwswitch_device_ioctl_exit;
    }

    retval = lwswitch_lib_ctrl(lwswitch_dev->lib_device,
                               _IOC_NR(cmd),
                               state.kernel_params,
                               state.kernel_params_size,
                               file->private_data);
    rc = lwswitch_map_status(retval);
    if (!rc)
    {
        rc = lwswitch_ioctl_state_sync(&state, cmd, arg);
    }

    lwswitch_ioctl_state_cleanup(&state);

lwswitch_device_ioctl_exit:
    mutex_unlock(&lwswitch_dev->device_mutex);

    return rc;
}

static long
lwswitch_device_unlocked_ioctl
(
    struct file *file,
    unsigned int cmd,
    unsigned long arg
)
{
    return lwswitch_device_ioctl(LW_FILE_INODE(file), file, cmd, arg);
}

static int
lwswitch_ctl_check_version(LWSWITCH_CHECK_VERSION_PARAMS *p)
{
    LwlStatus retval;

    p->is_compatible = 0;
    p->user.version[LWSWITCH_VERSION_STRING_LENGTH - 1] = '\0';

    retval = lwswitch_lib_check_api_version(p->user.version, p->kernel.version,
                                            LWSWITCH_VERSION_STRING_LENGTH);
    if (retval == LWL_SUCCESS)
    {
        p->is_compatible = 1;
    }
    else if (retval == -LWL_ERR_NOT_SUPPORTED)
    {
        printk(KERN_ERR "lwpu-lwswitch: Version mismatch, "
               "kernel version %s user version %s\n",
               p->kernel.version, p->user.version);
    }
    else
    {
        // An unexpected failure
        return lwswitch_map_status(retval);
    }

    return 0;
}

static void
lwswitch_ctl_get_devices(LWSWITCH_GET_DEVICES_PARAMS *p)
{
    int index = 0;
    LWSWITCH_DEV *lwswitch_dev;
    struct list_head *lwr;

    BUILD_BUG_ON(LWSWITCH_DEVICE_INSTANCE_MAX != LWSWITCH_MAX_DEVICES);

    list_for_each(lwr, &lwswitch.devices)
    {
        lwswitch_dev = list_entry(lwr, LWSWITCH_DEV, list_node);
        p->info[index].deviceInstance = lwswitch_dev->minor;
        p->info[index].pciDomain = LW_PCI_DOMAIN_NUMBER(lwswitch_dev->pci_dev);
        p->info[index].pciBus = LW_PCI_BUS_NUMBER(lwswitch_dev->pci_dev);
        p->info[index].pciDevice = LW_PCI_SLOT_NUMBER(lwswitch_dev->pci_dev);
        p->info[index].pciFunction = PCI_FUNC(lwswitch_dev->pci_dev->devfn);
        index++;
    }

    p->deviceCount = index;
}

static void
lwswitch_ctl_get_devices_v2(LWSWITCH_GET_DEVICES_V2_PARAMS *p)
{
    int index = 0;
    LWSWITCH_DEV *lwswitch_dev;
    struct list_head *lwr;

    BUILD_BUG_ON(LWSWITCH_DEVICE_INSTANCE_MAX != LWSWITCH_MAX_DEVICES);

    list_for_each(lwr, &lwswitch.devices)
    {
        lwswitch_dev = list_entry(lwr, LWSWITCH_DEV, list_node);
        p->info[index].deviceInstance = lwswitch_dev->minor;
        memcpy(&p->info[index].uuid, &lwswitch_dev->uuid, sizeof(lwswitch_dev->uuid));
        p->info[index].pciDomain = LW_PCI_DOMAIN_NUMBER(lwswitch_dev->pci_dev);
        p->info[index].pciBus = LW_PCI_BUS_NUMBER(lwswitch_dev->pci_dev);
        p->info[index].pciDevice = LW_PCI_SLOT_NUMBER(lwswitch_dev->pci_dev);
        p->info[index].pciFunction = PCI_FUNC(lwswitch_dev->pci_dev->devfn);
        p->info[index].physId = lwswitch_dev->phys_id;

        if (lwswitch_dev->lib_device != NULL)
        {
            mutex_lock(&lwswitch_dev->device_mutex);
            (void)lwswitch_lib_read_fabric_state(lwswitch_dev->lib_device,
                                                 &p->info[index].deviceState,
                                                 &p->info[index].deviceReason,
                                                 &p->info[index].driverState);
            mutex_unlock(&lwswitch_dev->device_mutex);
        }
        index++;
    }

    p->deviceCount = index;
}

#define LWSWITCH_CTL_CHECK_PARAMS(type, size) (sizeof(type) == size ? 0 : -EILWAL)

static int
lwswitch_ctl_cmd_dispatch
(
    unsigned int cmd,
    void *params,
    unsigned int param_size
)
{
    int rc;

    switch(cmd)
    {
        case CTRL_LWSWITCH_CHECK_VERSION:
            rc = LWSWITCH_CTL_CHECK_PARAMS(LWSWITCH_CHECK_VERSION_PARAMS,
                                           param_size);
            if (!rc)
            {
                rc = lwswitch_ctl_check_version(params);
            }
            break;
        case CTRL_LWSWITCH_GET_DEVICES:
            rc = LWSWITCH_CTL_CHECK_PARAMS(LWSWITCH_GET_DEVICES_PARAMS,
                                           param_size);
            if (!rc)
            {
                lwswitch_ctl_get_devices(params);
            }
            break;
        case CTRL_LWSWITCH_GET_DEVICES_V2:
            rc = LWSWITCH_CTL_CHECK_PARAMS(LWSWITCH_GET_DEVICES_V2_PARAMS,
                                           param_size);
            if (!rc)
            {
                lwswitch_ctl_get_devices_v2(params);
            }
            break;

        default:
            rc = -EILWAL;
            break;
    }

    return rc;
}

static int
lwswitch_ctl_ioctl
(
    struct inode *inode,
    struct file *file,
    unsigned int cmd,
    unsigned long arg
)
{
    int rc = 0;
    IOCTL_STATE state = {0};

    if (_IOC_TYPE(cmd) != LWSWITCH_CTL_IO_TYPE)
    {
        return -EILWAL;
    }

    rc = mutex_lock_interruptible(&lwswitch.driver_mutex);
    if (rc)
    {
        return rc;
    }

    rc = lwswitch_ioctl_state_start(&state, cmd, arg);
    if (rc)
    {
        goto lwswitch_ctl_ioctl_exit;
    }

    rc = lwswitch_ctl_cmd_dispatch(_IOC_NR(cmd),
                                   state.kernel_params,
                                   state.kernel_params_size);
    if (!rc)
    {
        rc = lwswitch_ioctl_state_sync(&state, cmd, arg);
    }

    lwswitch_ioctl_state_cleanup(&state);

lwswitch_ctl_ioctl_exit:
    mutex_unlock(&lwswitch.driver_mutex);

    return rc;
}

static long
lwswitch_ctl_unlocked_ioctl
(
    struct file *file,
    unsigned int cmd,
    unsigned long arg
)
{
    return lwswitch_ctl_ioctl(LW_FILE_INODE(file), file, cmd, arg);
}

static irqreturn_t
lwswitch_isr_pending
(
    int   irq,
    void *arg
)
{

    LWSWITCH_DEV *lwswitch_dev = (LWSWITCH_DEV *)arg;
    LwlStatus retval;

    //
    // On silicon MSI must be enabled.  Since interrupts will not be shared
    // with MSI, we can simply signal the thread.
    //
    if (lwswitch_dev->irq_mechanism == LWSWITCH_IRQ_MSI)
    {
        return IRQ_WAKE_THREAD;
    }

    if (lwswitch_dev->irq_mechanism == LWSWITCH_IRQ_PIN)
    {
        //
        // We do not take mutex in the interrupt context. The interrupt
        // check is safe to driver state.
        //
        retval = lwswitch_lib_check_interrupts(lwswitch_dev->lib_device);

        // Wake interrupt thread if there is an interrupt pending
        if (-LWL_MORE_PROCESSING_REQUIRED == retval)
        {
            lwswitch_lib_disable_interrupts(lwswitch_dev->lib_device);
            return IRQ_WAKE_THREAD;
        }

        // PCI errors are handled else where.
        if (-LWL_PCI_ERROR == retval)
        {
            return IRQ_NONE;
        }

        if (LWL_SUCCESS != retval)
        {
            pr_err("lwpu-lwswitch: unrecoverable error in ISR\n");
            LWSWITCH_OS_ASSERT(0);
        }
        return IRQ_NONE;
    }

    pr_err("lwpu-lwswitch: unsupported IRQ mechanism in ISR\n");
    LWSWITCH_OS_ASSERT(0);

    return IRQ_NONE;
}

static irqreturn_t
lwswitch_isr_thread
(
    int   irq,
    void *arg
)
{
    LWSWITCH_DEV *lwswitch_dev = (LWSWITCH_DEV *)arg;
    LwlStatus retval;

    mutex_lock(&lwswitch_dev->device_mutex);

    retval = lwswitch_lib_service_interrupts(lwswitch_dev->lib_device);

    wake_up(&lwswitch_dev->wait_q_errors);

    if (lwswitch_dev->irq_mechanism == LWSWITCH_IRQ_PIN)
    {
        lwswitch_lib_enable_interrupts(lwswitch_dev->lib_device);
    }

    mutex_unlock(&lwswitch_dev->device_mutex);

    if (WARN_ON(retval != LWL_SUCCESS))
    {
        printk(KERN_ERR "%s: Interrupts disabled to avoid a storm\n",
               lwswitch_dev->name);
    }

    return IRQ_HANDLED;
}

static void
lwswitch_task_dispatch
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    LwU64 nsec;
    LwU64 timeout;
    LwS64 rc;

    if (LW_ATOMIC_READ(lwswitch_dev->task_q_ready) == 0)
    {
        return;
    }

    mutex_lock(&lwswitch_dev->device_mutex);

    nsec = lwswitch_lib_deferred_task_dispatcher(lwswitch_dev->lib_device);

    mutex_unlock(&lwswitch_dev->device_mutex);

    timeout = usecs_to_jiffies(nsec / NSEC_PER_USEC);

    rc = wait_event_interruptible_timeout(lwswitch_dev->wait_q_shutdown,
                              (LW_ATOMIC_READ(lwswitch_dev->task_q_ready) == 0),
                              timeout);

    //
    // These background tasks should rarely, if ever, get interrupted. We use
    // the "interruptible" variant of wait_event in order to avoid contributing
    // to the system load average (/proc/loadavg), and to avoid softlockup
    // warnings that can occur if a kernel thread lingers too long in an
    // uninterruptible state. If this does get interrupted, we'd like to debug
    // and find out why, so WARN in that case.
    //
    WARN_ON(rc < 0);

    //
    // Schedule a work item only if the above actually timed out or got
    // interrupted, without the condition becoming true.
    //
    if (rc <= 0)
    {
        if (!lw_kthread_q_schedule_q_item(&lwswitch_dev->task_q,
                                          &lwswitch_dev->task_item))
        {
            printk(KERN_ERR "%s: Failed to re-schedule background task\n",
                   lwswitch_dev->name);
        }
    }
}

static int
lwswitch_probe
(
    struct pci_dev *pci_dev,
    const struct pci_device_id *id_table
)
{
    LWSWITCH_DEV *lwswitch_dev = NULL;
    int rc = 0;
    int minor;

    if (!lwswitch_lib_validate_device_id(pci_dev->device))
    {
        return -EILWAL;
    }

    printk(KERN_INFO "lwpu-lwswitch: Probing device %04x:%02x:%02x.%x, "
           "Vendor Id = 0x%x, Device Id = 0x%x, Class = 0x%x \n",
           LW_PCI_DOMAIN_NUMBER(pci_dev),
           LW_PCI_BUS_NUMBER(pci_dev),
           LW_PCI_SLOT_NUMBER(pci_dev),
           PCI_FUNC(pci_dev->devfn),
           pci_dev->vendor,
           pci_dev->device,
           pci_dev->class);

    mutex_lock(&lwswitch.driver_mutex);

    minor = lwswitch_find_minor();
    if (minor >= LWSWITCH_DEVICE_INSTANCE_MAX)
    {
        rc = -ERANGE;
        goto find_minor_failed;
    }

    lwswitch_dev = kzalloc(sizeof(*lwswitch_dev), GFP_KERNEL);
    if (NULL == lwswitch_dev)
    {
        rc = -ENOMEM;
        goto kzalloc_failed;
    }

    mutex_init(&lwswitch_dev->device_mutex);
    init_waitqueue_head(&lwswitch_dev->wait_q_errors);
    init_waitqueue_head(&lwswitch_dev->wait_q_shutdown);

    snprintf(lwswitch_dev->name, sizeof(lwswitch_dev->name),
        LWSWITCH_DRIVER_NAME "%d", minor);

    snprintf(lwswitch_dev->sname, sizeof(lwswitch_dev->sname),
        LWSWITCH_SHORT_NAME "%d", minor);

    rc = pci_enable_device(pci_dev);
    if (rc)
    {
        printk(KERN_ERR "%s: Failed to enable PCI device : %d\n",
               lwswitch_dev->name,
               rc);
        goto pci_enable_device_failed;
    }

    pci_set_master(pci_dev);

    rc = pci_request_regions(pci_dev, lwswitch_dev->name);
    if (rc)
    {
        printk(KERN_ERR "%s: Failed to request memory regions : %d\n",
               lwswitch_dev->name,
               rc);
        goto pci_request_regions_failed;
    }

    lwswitch_dev->bar0 = pci_iomap(pci_dev, 0, 0);
    if (!lwswitch_dev->bar0)
    {
        rc = -ENOMEM;
        printk(KERN_ERR "%s: Failed to map BAR0 region : %d\n",
               lwswitch_dev->name,
               rc);
        goto pci_iomap_failed;
    }

    lwswitch_dev->pci_dev = pci_dev;
    lwswitch_dev->minor = minor;

    rc = lwswitch_init_device(lwswitch_dev);
    if (rc)
    {
        printk(KERN_ERR "%s: Failed to initialize device : %d\n",
               lwswitch_dev->name,
               rc);
        goto init_device_failed;
    }

    if (lwswitch_is_device_blacklisted(lwswitch_dev))
    {
        lwswitch_post_init_blacklisted(lwswitch_dev);
        goto blacklisted;
    }

    //
    // device_mutex held here because post_init entries may call soeService_HAL()
    // with IRQs on. see bug 2856314 for more info
    //
    mutex_lock(&lwswitch_dev->device_mutex);
    rc = lwswitch_post_init_device(lwswitch_dev);
    mutex_unlock(&lwswitch_dev->device_mutex);
    if (rc)
    {
        printk(KERN_ERR "%s:Failed during device post init : %d\n",
               lwswitch_dev->name, rc);
        goto post_init_device_failed;
    }

blacklisted:
    rc = lwswitch_init_background_tasks(lwswitch_dev);
    if (rc)
    {
        printk(KERN_ERR "%s: Failed to initialize background tasks : %d\n",
               lwswitch_dev->name,
               rc);
        goto init_background_task_failed;
    }

    pci_set_drvdata(pci_dev, lwswitch_dev);

    lwswitch_procfs_device_add(lwswitch_dev);

    list_add_tail(&lwswitch_dev->list_node, &lwswitch.devices);

    LW_ATOMIC_INC(lwswitch.count);

    mutex_unlock(&lwswitch.driver_mutex);

    return 0;

init_background_task_failed:
post_init_device_failed:
    lwswitch_deinit_device(lwswitch_dev);

init_device_failed:
    pci_iounmap(pci_dev, lwswitch_dev->bar0);

pci_iomap_failed:
    pci_release_regions(pci_dev);

pci_request_regions_failed:
#ifdef CONFIG_PCI
    pci_clear_master(pci_dev);
#endif
    pci_disable_device(pci_dev);

pci_enable_device_failed:
    kfree(lwswitch_dev);

kzalloc_failed:
find_minor_failed:
    mutex_unlock(&lwswitch.driver_mutex);

    return rc;
}

void
lwswitch_remove
(
    struct pci_dev *pci_dev
)
{
    LWSWITCH_DEV *lwswitch_dev;

    mutex_lock(&lwswitch.driver_mutex);

    lwswitch_dev = pci_get_drvdata(pci_dev);

    if (lwswitch_dev == NULL)
    {
        goto done;
    }

    printk(KERN_INFO "%s: removing device %04x:%02x:%02x.%x\n",
           lwswitch_dev->name,
           LW_PCI_DOMAIN_NUMBER(pci_dev),
           LW_PCI_BUS_NUMBER(pci_dev),
           LW_PCI_SLOT_NUMBER(pci_dev),
           PCI_FUNC(pci_dev->devfn));

    //
    // Synchronize with device operations such as .ioctls/.poll, and then mark
    // the device unusable.
    //
    mutex_lock(&lwswitch_dev->device_mutex);
    lwswitch_dev->unusable = LW_TRUE;
    mutex_unlock(&lwswitch_dev->device_mutex);

    LW_ATOMIC_DEC(lwswitch.count);

    list_del(&lwswitch_dev->list_node);

    lwswitch_deinit_i2c_adapters(lwswitch_dev);

    WARN_ON(!list_empty(&lwswitch_dev->i2c_adapter_list));

    pci_set_drvdata(pci_dev, NULL);

    lwswitch_deinit_background_tasks(lwswitch_dev);

    lwswitch_deinit_device(lwswitch_dev);

    pci_iounmap(pci_dev, lwswitch_dev->bar0);

    pci_release_regions(pci_dev);

#ifdef CONFIG_PCI
    pci_clear_master(pci_dev);
#endif

    pci_disable_device(pci_dev);

    lwswitch_procfs_device_remove(lwswitch_dev);

    // Free lwswitch_dev only if it is not in use.
    if (LW_ATOMIC_READ(lwswitch_dev->ref_count) == 0)
    {
        kfree(lwswitch_dev);
    }

done:
    mutex_unlock(&lwswitch.driver_mutex);

    return;
}

static void
lwswitch_load_bar_info
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    struct pci_dev *pci_dev = lwswitch_dev->pci_dev;
    lwlink_pci_info *info;
    LwU32 bar = 0;

    lwswitch_lib_get_device_info(lwswitch_dev->lib_device, &info);

    info->bars[0].offset = LWRM_PCICFG_BAR_OFFSET(0);
    pci_read_config_dword(pci_dev, info->bars[0].offset, &bar);

    info->bars[0].busAddress = (bar & PCI_BASE_ADDRESS_MEM_MASK);
    if (LW_PCI_RESOURCE_FLAGS(pci_dev, 0) & PCI_BASE_ADDRESS_MEM_TYPE_64)
    {
        pci_read_config_dword(pci_dev, info->bars[0].offset + 4, &bar);
        info->bars[0].busAddress |= (((LwU64)bar) << 32);
    }

    info->bars[0].baseAddr = LW_PCI_RESOURCE_START(pci_dev, 0);

    info->bars[0].barSize = LW_PCI_RESOURCE_SIZE(pci_dev, 0);

    info->bars[0].pBar = lwswitch_dev->bar0;
}

static int
_lwswitch_initialize_msix_interrupt
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    // Not supported (bug 3018806)
    return -EILWAL;
}

static int
_lwswitch_initialize_msi_interrupt
(
    LWSWITCH_DEV *lwswitch_dev
)
{
#ifdef CONFIG_PCI_MSI
    struct pci_dev *pci_dev = lwswitch_dev->pci_dev;
    int rc;

    rc = pci_enable_msi(pci_dev);
    if (rc)
    {
        return rc;
    }

    return 0;
#else
    return -EILWAL;
#endif
}

static int
_lwswitch_get_irq_caps(LWSWITCH_DEV *lwswitch_dev, unsigned long *irq_caps)
{
    struct pci_dev *pci_dev;

    if (!lwswitch_dev || !irq_caps)
        return -EILWAL;

    pci_dev = lwswitch_dev->pci_dev;

    if (pci_find_capability(pci_dev, PCI_CAP_ID_MSIX))
        set_bit(LWSWITCH_IRQ_MSIX, irq_caps);

    if (pci_find_capability(pci_dev, PCI_CAP_ID_MSI))
        set_bit(LWSWITCH_IRQ_MSI, irq_caps);

    if (lwswitch_lib_use_pin_irq(lwswitch_dev->lib_device))
        set_bit(LWSWITCH_IRQ_PIN, irq_caps);

    return 0;
}

static int
lwswitch_initialize_device_interrupt
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    struct pci_dev *pci_dev = lwswitch_dev->pci_dev;
    int flags = 0;
    unsigned long irq_caps = 0;
    int rc;

    if (_lwswitch_get_irq_caps(lwswitch_dev, &irq_caps))
    {
        pr_err("%s: failed to retrieve device interrupt capabilities\n",
               lwswitch_dev->name);
        return -EILWAL;
    }

    lwswitch_dev->irq_mechanism = LWSWITCH_IRQ_NONE;

    if (test_bit(LWSWITCH_IRQ_MSIX, &irq_caps))
    {
        rc = _lwswitch_initialize_msix_interrupt(lwswitch_dev);
        if (!rc)
        {
            lwswitch_dev->irq_mechanism = LWSWITCH_IRQ_MSIX;
            pr_info("%s: using MSI-X\n", lwswitch_dev->name);
        }
    }

    if (lwswitch_dev->irq_mechanism == LWSWITCH_IRQ_NONE
        && test_bit(LWSWITCH_IRQ_MSI, &irq_caps))
    {
        rc = _lwswitch_initialize_msi_interrupt(lwswitch_dev);
        if (!rc)
        {
            lwswitch_dev->irq_mechanism = LWSWITCH_IRQ_MSI;
            pr_info("%s: using MSI\n", lwswitch_dev->name);
        }
    }

    if (lwswitch_dev->irq_mechanism == LWSWITCH_IRQ_NONE
        && test_bit(LWSWITCH_IRQ_PIN, &irq_caps))
    {
        flags |= IRQF_SHARED;
        lwswitch_dev->irq_mechanism = LWSWITCH_IRQ_PIN;
        pr_info("%s: using PCI pin\n", lwswitch_dev->name);
    }

    if (lwswitch_dev->irq_mechanism == LWSWITCH_IRQ_NONE)
    {
        pr_err("%s: No supported interrupt mechanism was found. This device supports:\n",
               lwswitch_dev->name);

        if (test_bit(LWSWITCH_IRQ_MSIX, &irq_caps))
            pr_err("%s: MSI-X\n", lwswitch_dev->name);
        if (test_bit(LWSWITCH_IRQ_MSI, &irq_caps))
            pr_err("%s: MSI\n", lwswitch_dev->name);
        if (test_bit(LWSWITCH_IRQ_PIN, &irq_caps))
             pr_err("%s: PCI Pin\n", lwswitch_dev->name);

        return -EILWAL;
    }

    rc = request_threaded_irq(pci_dev->irq,
                              lwswitch_isr_pending,
                              lwswitch_isr_thread,
                              flags, lwswitch_dev->sname,
                              lwswitch_dev);
    if (rc)
    {
#ifdef CONFIG_PCI_MSI
        if (lwswitch_dev->irq_mechanism == LWSWITCH_IRQ_MSI)
        {
            pci_disable_msi(pci_dev);
        }
#endif
        printk(KERN_ERR "%s: failed to get IRQ\n",
               lwswitch_dev->name);

        return rc;
    }

    return 0;
}

void
lwswitch_shutdown_device_interrupt
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    struct pci_dev *pci_dev = lwswitch_dev->pci_dev;

    free_irq(pci_dev->irq, lwswitch_dev);
#ifdef CONFIG_PCI_MSI
    if (lwswitch_dev->irq_mechanism == LWSWITCH_IRQ_MSI)
    {
        pci_disable_msi(pci_dev);
    }
#endif
}

static void
lwswitch_ctl_exit
(
    void
)
{
    cdev_del(&lwswitch.cdev_ctl);
}

static int
lwswitch_ctl_init
(
    int major
)
{
    int rc = 0;
    dev_t lwswitch_ctl = MKDEV(major, LWSWITCH_CTL_MINOR);

    cdev_init(&lwswitch.cdev_ctl, &ctl_fops);

    lwswitch.cdev_ctl.owner = THIS_MODULE;

    rc = cdev_add(&lwswitch.cdev_ctl, lwswitch_ctl, 1);
    if (rc < 0)
    {
        printk(KERN_ERR "lwpu-lwswitch: Unable to create cdev ctl\n");
        return rc;
    }

    return 0;
}

//
// Initialize lwswitch driver SW state.  This is lwrrently called
// from the RM as a backdoor interface, and not by the Linux device
// manager
//
int
lwswitch_init
(
    void
)
{
    int rc;

    if (lwswitch.initialized)
    {
        printk(KERN_ERR "lwpu-lwswitch: Interface already initialized\n");
        return -EBUSY;
    }

    BUILD_BUG_ON(LWSWITCH_DEVICE_INSTANCE_MAX >= LWSWITCH_MINOR_COUNT);

    mutex_init(&lwswitch.driver_mutex);

    INIT_LIST_HEAD(&lwswitch.devices);

    rc = alloc_chrdev_region(&lwswitch.devno,
                             0,
                             LWSWITCH_MINOR_COUNT,
                             LWSWITCH_DRIVER_NAME);
    if (rc < 0)
    {
        printk(KERN_ERR "lwpu-lwswitch: Unable to create cdev region\n");
        goto alloc_chrdev_region_fail;
    }

    printk(KERN_ERR, "lwpu-lwswitch: Major: %d Minor: %d\n",
           MAJOR(lwswitch.devno),
           MINOR(lwswitch.devno));

    cdev_init(&lwswitch.cdev, &device_fops);
    lwswitch.cdev.owner = THIS_MODULE;
    rc = cdev_add(&lwswitch.cdev, lwswitch.devno, LWSWITCH_DEVICE_INSTANCE_MAX);
    if (rc < 0)
    {
        printk(KERN_ERR "lwpu-lwswitch: Unable to create cdev\n");
        goto cdev_add_fail;
    }

    rc = lwswitch_procfs_init();
    if (rc < 0)
    {
        goto lwswitch_procfs_init_fail;
    }

    rc = pci_register_driver(&lwswitch_pci_driver);
    if (rc < 0)
    {
        printk(KERN_ERR "lwpu-lwswitch: Failed to register driver : %d\n", rc);
        goto pci_register_driver_fail;
    }

    rc = lwswitch_ctl_init(MAJOR(lwswitch.devno));
    if (rc < 0)
    {
        goto lwswitch_ctl_init_fail;
    }

    lwswitch.initialized = LW_TRUE;

    return 0;

lwswitch_ctl_init_fail:
    pci_unregister_driver(&lwswitch_pci_driver);

pci_register_driver_fail:
lwswitch_procfs_init_fail:
    cdev_del(&lwswitch.cdev);

cdev_add_fail:
    unregister_chrdev_region(lwswitch.devno, LWSWITCH_MINOR_COUNT);

alloc_chrdev_region_fail:

    return rc;
}

//
// Clean up driver state on exit.  Lwrrently called from RM backdoor call,
// and not by the Linux device manager.
//
void
lwswitch_exit
(
    void
)
{
    if (LW_FALSE == lwswitch.initialized)
    {
        return;
    }

    lwswitch_procfs_exit();

    lwswitch_ctl_exit();

    pci_unregister_driver(&lwswitch_pci_driver);

    cdev_del(&lwswitch.cdev);

    unregister_chrdev_region(lwswitch.devno, LWSWITCH_MINOR_COUNT);

    WARN_ON(!list_empty(&lwswitch.devices));

    lwswitch.initialized = LW_FALSE;
}

//
// Get current time in seconds.nanoseconds
// In this implementation, the time is from epoch time
// (midnight UTC of January 1, 1970)
//
LwU64
lwswitch_os_get_platform_time
(
    void
)
{
    struct timespec64 ts;

    ktime_get_raw_ts64(&ts);
    return (LwU64) timespec64_to_ns(&ts);
}

void
lwswitch_os_print
(
    const int  log_level,
    const char *fmt,
    ...
)
{
    va_list arglist;
    char   *kern_level;
    char    fmt_printk[LWSWITCH_LOG_BUFFER_SIZE];

    switch (log_level)
    {
        case LWSWITCH_DBG_LEVEL_MMIO:
            kern_level = KERN_DEBUG;
            break;
        case LWSWITCH_DBG_LEVEL_INFO:
            kern_level = KERN_INFO;
            break;
        case LWSWITCH_DBG_LEVEL_SETUP:
            kern_level = KERN_INFO;
            break;
        case LWSWITCH_DBG_LEVEL_WARN:
            kern_level = KERN_WARNING;
            break;
        case LWSWITCH_DBG_LEVEL_ERROR:
            kern_level = KERN_ERR;
            break;
        default:
            kern_level = KERN_DEFAULT;
            break;
    }

    va_start(arglist, fmt);
    snprintf(fmt_printk, sizeof(fmt_printk), "%s%s", kern_level, fmt);
    vprintk(fmt_printk, arglist);
    va_end(arglist);
}

void
lwswitch_os_override_platform
(
    void *os_handle,
    LwBool *rtlsim
)
{
    // Never run on RTL
    *rtlsim = LW_FALSE;
}

LwlStatus
lwswitch_os_read_registery_binary
(
    void *os_handle,
    const char *name,
    LwU8 *data,
    LwU32 length
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwU32
lwswitch_os_get_device_count
(
    void
)
{
    return LW_ATOMIC_READ(lwswitch.count);
}

//
// A helper to colwert a string to an unsigned int.
//
// The string should be NULL terminated.
// Only works with base16 values.
//
static int
lwswitch_os_strtouint
(
    char *str,
    unsigned int *data
)
{
    char *p;
    unsigned long long val;

    if (!str || !data)
    {
        return -EILWAL;
    }

    *data = 0;
    val = 0;
    p = str;

    while (*p != '\0')
    {
        if ((tolower(*p) == 'x') && (*str == '0') && (p == str + 1))
        {
            p++;
        }
        else if (*p >='0' && *p <= '9')
        {
            val = val * 16 + (*p - '0');
            p++;
        }
        else if (tolower(*p) >= 'a' && tolower(*p) <= 'f')
        {
            val = val * 16 + (tolower(*p) - 'a' + 10);
            p++;
        }
        else
        {
            return -EILWAL;
        }
    }

    if (val > 0xFFFFFFFF)
    {
        return -EILWAL;
    }

    *data = (unsigned int)val;

    return 0;
}

LwlStatus
lwswitch_os_read_registry_dword
(
    void *os_handle,
    const char *name,
    LwU32 *data
)
{
    char *regkey, *regkey_val_start, *regkey_val_end;
    char regkey_val[LWSWITCH_REGKEY_VALUE_LEN + 1];
    LwU32 regkey_val_len = 0;

    *data = 0;

    if (!LwSwitchRegDwords)
    {
        return -LWL_ERR_GENERIC;
    }

    regkey = strstr(LwSwitchRegDwords, name);
    if (!regkey)
    {
        return -LWL_ERR_GENERIC;
    }

    regkey = strchr(regkey, '=');
    if (!regkey)
    {
        return -LWL_ERR_GENERIC;
    }

    regkey_val_start = regkey + 1;

    regkey_val_end = strchr(regkey, ';');
    if (!regkey_val_end)
    {
        regkey_val_end = strchr(regkey, '\0');
    }

    regkey_val_len = regkey_val_end - regkey_val_start;
    if (regkey_val_len > LWSWITCH_REGKEY_VALUE_LEN || regkey_val_len == 0)
    {
        return -LWL_ERR_GENERIC;
    }

    strncpy(regkey_val, regkey_val_start, regkey_val_len);
    regkey_val[regkey_val_len] = '\0';

    if (lwswitch_os_strtouint(regkey_val, data) != 0)
    {
        return -LWL_ERR_GENERIC;
    }

    return LWL_SUCCESS;
}

static LwBool
_lwswitch_is_space(const char ch)
{
    return ((ch == ' ') || ((ch >= '\t') && (ch <= '\r')));
}

static char *
_lwswitch_remove_spaces(const char *in)
{
    unsigned int len = lwswitch_os_strlen(in) + 1;
    const char *in_ptr;
    char *out, *out_ptr;

    out = lwswitch_os_malloc(len);
    if (out == NULL)
        return NULL;

    in_ptr = in;
    out_ptr = out;

    while (*in_ptr != '\0')
    {
        if (!_lwswitch_is_space(*in_ptr))
            *out_ptr++ = *in_ptr;
        in_ptr++;
    }
    *out_ptr = '\0';

    return out;
}

/*
 * Compare given string UUID with the LwSwitchBlacklist registry parameter string and
 * return whether the UUID is in the LwSwitch blacklist
 */
LwBool
lwswitch_os_is_uuid_in_blacklist
(
    LwUuid *uuid
)
{
    char *list;
    char *ptr;
    char *token;
    LwU8 uuid_string[LWSWITCH_UUID_STRING_LENGTH];

    if (LwSwitchBlacklist == NULL)
        return LW_FALSE;

    if (lwswitch_uuid_to_string(uuid, uuid_string, LWSWITCH_UUID_STRING_LENGTH) == 0)
        return LW_FALSE;

    if ((list = _lwswitch_remove_spaces(LwSwitchBlacklist)) == NULL)
        return LW_FALSE;

    ptr = list;

    while ((token = strsep(&ptr, ",")) != NULL)
    {
        if (strcmp(token, uuid_string) == 0)
        {
            lwswitch_os_free(list);
            return LW_TRUE;
        }
    }
    lwswitch_os_free(list);
    return LW_FALSE;
}


LwlStatus
lwswitch_os_alloc_contig_memory
(
    void *os_handle,
    void **virt_addr,
    LwU32 size,
    LwBool force_dma32
)
{
    LwU32 gfp_flags;
    unsigned long lw_gfp_addr = 0;

    if (!virt_addr)
        return -LWL_BAD_ARGS;

    gfp_flags = GFP_KERNEL | (force_dma32 ? GFP_DMA32 : 0);
    LW_GET_FREE_PAGES(lw_gfp_addr, get_order(size), gfp_flags);

    if(!lw_gfp_addr)
    {
        pr_err("lwpu-lwswitch: unable to allocate kernel memory\n");
        return -LWL_NO_MEM;
    }

    *virt_addr = (void *)lw_gfp_addr;

    return LWL_SUCCESS;
}

void
lwswitch_os_free_contig_memory
(
    void *os_handle,
    void *virt_addr,
    LwU32 size
)
{
    LW_FREE_PAGES((unsigned long)virt_addr, get_order(size));
}

static inline int
_lwswitch_to_pci_dma_direction
(
    LwU32 direction
)
{
    if (direction == LWSWITCH_DMA_DIR_TO_SYSMEM)
        return PCI_DMA_FROMDEVICE;
    else if (direction == LWSWITCH_DMA_DIR_FROM_SYSMEM)
        return PCI_DMA_TODEVICE;
    else
        return PCI_DMA_BIDIRECTIONAL;
}

LwlStatus
lwswitch_os_map_dma_region
(
    void *os_handle,
    void *cpu_addr,
    LwU64 *dma_handle,
    LwU32 size,
    LwU32 direction
)
{
    int dma_dir;
    struct pci_dev *pdev = (struct pci_dev *)os_handle;

    if (!pdev || !cpu_addr || !dma_handle)
        return -LWL_BAD_ARGS;

    dma_dir = _lwswitch_to_pci_dma_direction(direction);

    *dma_handle = (LwU64)pci_map_single(pdev, cpu_addr, size, dma_dir);

    if (pci_dma_mapping_error(pdev, *dma_handle))
    {
        pr_err("lwpu-lwswitch: unable to create PCI DMA mapping\n");
        return -LWL_ERR_GENERIC;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_os_unmap_dma_region
(
    void *os_handle,
    void *cpu_addr,
    LwU64 dma_handle,
    LwU32 size,
    LwU32 direction
)
{
    int dma_dir;
    struct pci_dev *pdev = (struct pci_dev *)os_handle;

    if (!pdev || !cpu_addr)
        return -LWL_BAD_ARGS;

    dma_dir = _lwswitch_to_pci_dma_direction(direction);

    pci_unmap_single(pdev, dma_handle, size, dma_dir);

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_os_set_dma_mask
(
    void *os_handle,
    LwU32 dma_addr_width
)
{
    struct pci_dev *pdev = (struct pci_dev *)os_handle;

    if (!pdev)
        return -LWL_BAD_ARGS;

    if (pci_set_dma_mask(pdev, DMA_BIT_MASK(dma_addr_width)))
        return -LWL_ERR_GENERIC;

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_os_sync_dma_region_for_cpu
(
    void *os_handle,
    LwU64 dma_handle,
    LwU32 size,
    LwU32 direction
)
{
    int dma_dir;
    struct pci_dev *pdev = (struct pci_dev *)os_handle;

    if (!pdev)
        return -LWL_BAD_ARGS;

    dma_dir = _lwswitch_to_pci_dma_direction(direction);

    pci_dma_sync_single_for_cpu(pdev, dma_handle, size, dma_dir);

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_os_sync_dma_region_for_device
(
    void *os_handle,
    LwU64 dma_handle,
    LwU32 size,
    LwU32 direction
)
{
    int dma_dir;
    struct pci_dev *pdev = (struct pci_dev *)os_handle;

    if (!pdev)
        return -LWL_BAD_ARGS;

    dma_dir = _lwswitch_to_pci_dma_direction(direction);

    pci_dma_sync_single_for_device(pdev, dma_handle, size, dma_dir);

    return LWL_SUCCESS;
}

static inline void *
_lwswitch_os_malloc
(
    LwLength size
)
{
    void *ptr = NULL;

    if (!LW_MAY_SLEEP())
    {
        if (size <= LWSWITCH_KMALLOC_LIMIT)
        {
            ptr = kmalloc(size, LW_GFP_ATOMIC);
        }
    }
    else
    {
        if (size <= LWSWITCH_KMALLOC_LIMIT)
        {
            ptr = kmalloc(size, LW_GFP_NO_OOM);
        }

        if (ptr == NULL)
        {
            ptr = vmalloc(size);
        }
    }

    return ptr;
}

void *
lwswitch_os_malloc_trace
(
    LwLength size,
    const char *file,
    LwU32 line
)
{
#if defined(LW_MEM_LOGGER)
    void *ptr = _lwswitch_os_malloc(size);
    if (ptr)
    {
        lw_memdbg_add(ptr, size, file, line);
    }

    return ptr;
#else
    return _lwswitch_os_malloc(size);
#endif
}

static inline void
_lwswitch_os_free
(
    void *ptr
)
{
    if (!ptr)
        return;

    if (is_vmalloc_addr(ptr))
    {
        vfree(ptr);
    }
    else
    {
        kfree(ptr);
    }
}

void
lwswitch_os_free
(
    void *ptr
)
{
#if defined (LW_MEM_LOGGER)
    if (ptr == NULL)
        return;

    lw_memdbg_remove(ptr, 0, NULL, 0);

    return _lwswitch_os_free(ptr);
#else
    return _lwswitch_os_free(ptr);
#endif
}

LwLength
lwswitch_os_strlen
(
    const char *str
)
{
    return strlen(str);
}

char*
lwswitch_os_strncpy
(
    char *dest,
    const char *src,
    LwLength length
)
{
    return strncpy(dest, src, length);
}

int
lwswitch_os_strncmp
(
    const char *s1,
    const char *s2,
    LwLength length
)
{
    return strncmp(s1, s2, length);
}

void *
lwswitch_os_memset
(
    void *dest,
    int value,
    LwLength size
)
{
     return memset(dest, value, size);
}

void *
lwswitch_os_memcpy
(
    void *dest,
    const void *src,
    LwLength size
)
{
    return memcpy(dest, src, size);
}

int
lwswitch_os_memcmp
(
    const void *s1,
    const void *s2,
    LwLength size
)
{
    return memcmp(s1, s2, size);
}

LwU32
lwswitch_os_mem_read32
(
    const volatile void * address
)
{
    return (*(const volatile LwU32*)(address));
}

void
lwswitch_os_mem_write32
(
    volatile void *address,
    LwU32 data
)
{
    (*(volatile LwU32 *)(address)) = data;
}

LwU64
lwswitch_os_mem_read64
(
    const volatile void * address
)
{
    return (*(const volatile LwU64 *)(address));
}

void
lwswitch_os_mem_write64
(
    volatile void *address,
    LwU64 data
)
{
    (*(volatile LwU64 *)(address)) = data;
}

int
lwswitch_os_snprintf
(
    char *dest,
    LwLength size,
    const char *fmt,
    ...
)
{
    va_list arglist;
    int chars_written;

    va_start(arglist, fmt);
    chars_written = vsnprintf(dest, size, fmt, arglist);
    va_end(arglist);

    return chars_written;
}

int
lwswitch_os_vsnprintf
(
    char *buf,
    LwLength size,
    const char *fmt,
    va_list arglist
)
{
    return vsnprintf(buf, size, fmt, arglist);
}

void
lwswitch_os_assert_log
(
    int cond,
    const char *fmt,
    ...
)
{
    if(cond == 0x0)
    {
        if (printk_ratelimit())
        {
            va_list arglist;
            char fmt_printk[LWSWITCH_LOG_BUFFER_SIZE];

            va_start(arglist, fmt);
            vsnprintf(fmt_printk, sizeof(fmt_printk), fmt, arglist);
            va_end(arglist);
            lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR, fmt_printk);
            WARN_ON(1);
         }
         dbg_breakpoint();
    }
}

/*
 * Sleep for specified milliseconds. Yields the CPU to scheduler.
 */
void
lwswitch_os_sleep
(
    unsigned int ms
)
{
    LW_STATUS status;
    status = lw_sleep_ms(ms);

    if (status != LW_OK)
    {
        if (printk_ratelimit())
        {
            lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR, "LWSwitch: requested"
                              " sleep duration %d msec exceeded %d msec\n",
                              ms, LW_MAX_ISR_DELAY_MS);
            WARN_ON(1);
        }
    }
}

LwlStatus
lwswitch_os_acquire_fabric_mgmt_cap
(
    void *osPrivate,
    LwU64 capDescriptor
)
{
    int dup_fd = -1;
    lwswitch_file_private_t *private_data = (lwswitch_file_private_t *)osPrivate;

    if (private_data == NULL)
    {
        return -LWL_BAD_ARGS;
    }

    dup_fd = lwlink_cap_acquire((int)capDescriptor,
                                LWLINK_CAP_FABRIC_MANAGEMENT);
    if (dup_fd < 0)
    {
        return -LWL_ERR_OPERATING_SYSTEM;
    }

    private_data->capability_fds.fabric_mgmt = dup_fd;
    return LWL_SUCCESS;
}

int
lwswitch_os_is_fabric_manager
(
    void *osPrivate
)
{
    lwswitch_file_private_t *private_data = (lwswitch_file_private_t *)osPrivate;

    /* Make sure that fabric mgmt capbaility fd is valid */
    if ((private_data == NULL) ||
        (private_data->capability_fds.fabric_mgmt < 0))
    {
        return 0;
    }

    return 1;
}

int
lwswitch_os_is_admin
(
    void
)
{
    return LW_IS_SUSER();
}

#define LW_KERNEL_RELEASE    ((LINUX_VERSION_CODE >> 16) & 0x0ff)
#define LW_KERNEL_VERSION    ((LINUX_VERSION_CODE >> 8)  & 0x0ff)
#define LW_KERNEL_SUBVERSION ((LINUX_VERSION_CODE)       & 0x0ff)

LwlStatus
lwswitch_os_get_os_version
(
    LwU32 *pMajorVer,
    LwU32 *pMinorVer,
    LwU32 *pBuildNum
)
{
    if (pMajorVer)
        *pMajorVer = LW_KERNEL_RELEASE;
    if (pMinorVer)
        *pMinorVer = LW_KERNEL_VERSION;
    if (pBuildNum)
        *pBuildNum = LW_KERNEL_SUBVERSION;

    return LWL_SUCCESS;
}

/*!
 * @brief: OS specific handling to add an event.
 */
LwlStatus
lwswitch_os_add_client_event
(
    void            *osHandle,
    void            *osPrivate,
    LwU32           eventId
)
{
    return LWL_SUCCESS;
}

/*!
 * @brief: OS specific handling to remove all events corresponding to osPrivate.
 */
LwlStatus
lwswitch_os_remove_client_event
(
    void            *osHandle,
    void            *osPrivate
)
{
    return LWL_SUCCESS;
}

/*!
 * @brief: OS specific handling to notify an event.
 */
LwlStatus
lwswitch_os_notify_client_event
(
    void *osHandle,
    void *osPrivate,
    LwU32 eventId
)
{
    lwswitch_file_private_t *private_data = (lwswitch_file_private_t *)osPrivate;

    if (private_data == NULL)
    {
        return -LWL_BAD_ARGS;
    }

    private_data->file_event.event_pending = LW_TRUE;
    wake_up_interruptible(&private_data->file_event.wait_q_event);

    return LWL_SUCCESS;
}

/*!
 * @brief: Gets OS specific support for the REGISTER_EVENTS ioctl
 */
LwlStatus
lwswitch_os_get_supported_register_events_params
(
    LwBool *many_events,
    LwBool *os_descriptor
)
{
    *many_events   = LW_FALSE;
    *os_descriptor = LW_FALSE;
    return LWL_SUCCESS;
}
