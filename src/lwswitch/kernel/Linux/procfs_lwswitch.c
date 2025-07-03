/*******************************************************************************
    Copyright (c) 2016-2020 LWPU Corporation

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
#include "lw-procfs.h"

#include <linux/fs.h>

#if defined(CONFIG_PROC_FS)

#define LW_DEFINE_SINGLE_LWSWITCH_PROCFS_FILE(name) \
    LW_DEFINE_SINGLE_PROCFS_FILE_READ_ONLY(name, lw_system_pm_lock)

#define LWSWITCH_PROCFS_DIR "driver/lwpu-lwswitch"

static struct proc_dir_entry *lwswitch_procfs_dir;
static struct proc_dir_entry *lwswitch_permissions;
static struct proc_dir_entry *lwswitch_procfs_devices;

static int
lw_procfs_read_permissions
(
    struct seq_file *s,
    void *v
)
{
    // Restrict device node permissions - 0666. Used by lwpu-modprobe.
    seq_printf(s, "%s: %u\n", "DeviceFileMode", 438);

    return 0;
}

LW_DEFINE_SINGLE_LWSWITCH_PROCFS_FILE(permissions);

static int
lw_procfs_read_device_info
(
    struct seq_file *s,
    void *v
)
{
    LWSWITCH_DEV *lwswitch_dev = s->private;

    if (!lwswitch_dev)
    {
        LWSWITCH_OS_ASSERT(0);
        return -EFAULT;
    }

    seq_printf(s, "BIOS Version: ");

    if (lwswitch_dev->bios_ver)
    {
        seq_printf(s, "%02llx.%02llx.%02llx.%02llx.%02llx\n",
                       lwswitch_dev->bios_ver >> 32,
                       (lwswitch_dev->bios_ver >> 24) & 0xFF,
                       (lwswitch_dev->bios_ver >> 16) & 0xFF,
                       (lwswitch_dev->bios_ver >> 8) & 0xFF,
                       lwswitch_dev->bios_ver & 0xFF);
    }
    else
    {
        seq_printf(s, "N/A\n");
    }

    return 0;
}

LW_DEFINE_SINGLE_LWSWITCH_PROCFS_FILE(device_info);

void
lwswitch_procfs_device_remove
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    if (!lwswitch_dev || !lwswitch_dev->procfs_dir)
    {
        LWSWITCH_OS_ASSERT(0);
        return;
    }

    lw_procfs_unregister_all(lwswitch_dev->procfs_dir, lwswitch_dev->procfs_dir);
    lwswitch_dev->procfs_dir = NULL;
}

int
lwswitch_procfs_device_add
(
    LWSWITCH_DEV *lwswitch_dev
)
{
    struct pci_dev *pci_dev;
    struct proc_dir_entry *device_dir, *entry;
    char name[32];

    if (!lwswitch_dev || !lwswitch_dev->pci_dev)
    {
        LWSWITCH_OS_ASSERT(0);
        return -1;
    }

    pci_dev = lwswitch_dev->pci_dev;

    snprintf(name, sizeof(name), "%04x:%02x:%02x.%1x",
             LW_PCI_DOMAIN_NUMBER(pci_dev), LW_PCI_BUS_NUMBER(pci_dev),
             LW_PCI_SLOT_NUMBER(pci_dev), PCI_FUNC(pci_dev->devfn));

    device_dir = LW_CREATE_PROC_DIR(name, lwswitch_procfs_devices);
    if (!device_dir)
        return -1;

    lwswitch_dev->procfs_dir = device_dir;

    entry = LW_CREATE_PROC_FILE("information", device_dir, device_info,
                                lwswitch_dev);
    if (!entry)
        goto failed;

    return 0;

failed:
    lwswitch_procfs_device_remove(lwswitch_dev);
    return -1;
}

void
lwswitch_procfs_exit
(
    void
)
{
    if (!lwswitch_procfs_dir)
    {
        return;
    }

    lw_procfs_unregister_all(lwswitch_procfs_dir, lwswitch_procfs_dir);
    lwswitch_procfs_dir = NULL;
}

int
lwswitch_procfs_init
(
    void
)
{
    lwswitch_procfs_dir = LW_CREATE_PROC_DIR(LWSWITCH_PROCFS_DIR, NULL);
    if (!lwswitch_procfs_dir)
    {
        return -EACCES;
    }

    lwswitch_permissions = LW_CREATE_PROC_FILE("permissions",
                                               lwswitch_procfs_dir,
                                               permissions,
                                               NULL);
    if (!lwswitch_permissions)
    {
        goto cleanup;
    }

    lwswitch_procfs_devices = LW_CREATE_PROC_DIR("devices", lwswitch_procfs_dir);
    if (!lwswitch_procfs_devices)
    {
        goto cleanup;
    }

    return 0;

cleanup:

    lwswitch_procfs_exit();

    return -EACCES;
}

#else // !CONFIG_PROC_FS

int lwswitch_procfs_init(void) { return 0; }
void lwswitch_procfs_exit(void) { }
int lwswitch_procfs_device_add(LWSWITCH_DEV *lwswitch_dev) { return 0; }
void lwswitch_procfs_device_remove(LWSWITCH_DEV *lwswitch_dev) { }

#endif // CONFIG_PROC_FS
