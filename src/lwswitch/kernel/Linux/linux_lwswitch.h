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
#ifndef LINUX_LWSWITCH_H
#define LINUX_LWSWITCH_H

#include "lwmisc.h"
#include "lw-linux.h"
#include "lw-kthread-q.h"
#include "export_lwswitch.h"

#define LWSWITCH_SHORT_NAME "lwswi"

#define LWSWITCH_IRQ_NONE 0
#define LWSWITCH_IRQ_MSIX 1
#define LWSWITCH_IRQ_MSI  2
#define LWSWITCH_IRQ_PIN  3

#define LWSWITCH_OS_ASSERT(_cond)                                                       \
    lwswitch_os_assert_log((_cond), "LWSwitch: Assertion failed in %s() at %s:%d\n",    \
         __FUNCTION__ , __FILE__, __LINE__)

#define LWSWITCH_KMALLOC_LIMIT (128 * 1024)

#define lwswitch_os_malloc(_size)        lwswitch_os_malloc_trace(_size, __FILE__, __LINE__)

typedef struct
{
    struct list_head entry;
    struct i2c_adapter *adapter;
} lwswitch_i2c_adapter_entry;

// Per-chip driver state
typedef struct
{
    char name[sizeof(LWSWITCH_DRIVER_NAME) + 4];
    char sname[sizeof(LWSWITCH_SHORT_NAME) + 4];  /* short name */
    int minor;
    LwUuid uuid;
    struct mutex device_mutex;
    lwswitch_device *lib_device;                  /* lwswitch library device */
    wait_queue_head_t wait_q_errors;
    void *bar0;
    struct lw_kthread_q task_q;                   /* Background task queue */
    struct lw_kthread_q_item task_item;           /* Background dispatch task */
    atomic_t task_q_ready;
    wait_queue_head_t wait_q_shutdown;
    struct pci_dev *pci_dev;
    atomic_t ref_count;
    struct list_head list_node;
    LwBool unusable;
    LwU32 phys_id;
    LwU64 bios_ver;
#if defined(CONFIG_PROC_FS)
    struct proc_dir_entry *procfs_dir;
#endif
    LwU8 irq_mechanism;
    struct list_head i2c_adapter_list;
} LWSWITCH_DEV;


int lwswitch_map_status(LwlStatus status);
int lwswitch_procfs_init(void);
void lwswitch_procfs_exit(void);
int lwswitch_procfs_device_add(LWSWITCH_DEV *lwswitch_dev);
void lwswitch_procfs_device_remove(LWSWITCH_DEV *lwswitch_dev);
struct i2c_adapter *lwswitch_i2c_add_adapter(LWSWITCH_DEV *lwswitch_dev, LwU32 port);
void lwswitch_i2c_del_adapter(struct i2c_adapter *adapter);

#endif // LINUX_LWSWITCH_H
