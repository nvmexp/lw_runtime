/*******************************************************************************
    Copyright (c) 2016-2021 LWpu Corporation

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

#include "lwlink_export.h"
#include "lwmisc.h"
#include "lwlink_mods.h"
#include "../inc/common_lwswitch.h"
#include "../inc/haldef_lwswitch.h"

#include "g_lwconfig.h"
#include "export_lwswitch.h"
#include "mods_lwswitch.h"

#include "ctrl_dev_internal_lwswitch.h"
#include "ctrl_dev_lwswitch.h"

#define LWSWITCH_OS_ASSERT(_cond)                                       \
    lwswitch_os_assert_log((_cond), "LWSwitch: Assertion failed in OS layer \n")

typedef struct mods_lwswitch_memdesc_mapping
{
    void *descriptor;
    void *virt_addr;
    LWListRec entry;
} mods_lwswitch_memdesc_mapping;

typedef struct mods_lwswitch_device_handle
{
    lwswitch_device *lib_device;
    lwlink_pci_info *info;

    LwU32 pci_domain;
    LwU32 pci_bus;
    LwU32 pci_device;
    LwU32 pci_function;
    LwU32 irq;

    LWListRec memdesc_mapping_list;
    LwU32 dma_addr_width;
    void * device_mutex;
} mods_lwswitch_device_handle;

static mods_lwswitch_device_handle *lwswitch_devices[LWSWITCH_DEVICE_INSTANCE_MAX];

static LwBool s_TaskDispatcherRunning = LW_FALSE;
static LwS32  s_TaskDispatcherTid     = -1;

static LwU32 s_PrintLevel = PRI_NORMAL;

static LwlStatus lwswitch_mods_initialize_device_interrupt(mods_lwswitch_device_handle *);
static void lwswitch_mods_shutdown_device_interrupt(mods_lwswitch_device_handle *);
static void lwswitch_mods_initialize_device_pci_bus(mods_lwswitch_device_handle *);
static LwlStatus lwswitch_mods_map_device(mods_lwswitch_device_handle *dev_handle);
static void lwswitch_mods_unmap_device(mods_lwswitch_device_handle *dev_handle);

static void
lwswitch_task_dispatch
(
    void *pvArgs
)
{
    while (s_TaskDispatcherRunning)
    {
        LwU32 slot;

        // Check at least once per second
        LwU64 min_next_delay = LWSWITCH_INTERVAL_1SEC_IN_NS;

        // Handle any registered tasks on all devices
        for (slot = 0; slot < LWSWITCH_DEVICE_INSTANCE_MAX; slot++)
        {
            if (lwswitch_devices[slot])
            {
                ModsDrvSetSwitchId(slot);
                ModsDrvAcquireMutex(lwswitch_devices[slot]->device_mutex);
                LwU64 next_delay =
                    lwswitch_lib_deferred_task_dispatcher(lwswitch_devices[slot]->lib_device);
                ModsDrvReleaseMutex(lwswitch_devices[slot]->device_mutex);
                if (next_delay < min_next_delay)
                    min_next_delay = next_delay;
            }
        }
        lwswitch_os_sleep(min_next_delay/1000000ull);
    }
}

static LwBool
is_pci_device_allowed
(
    lwswitch_mods_pci_info *pAllowedDevices,
    LwU32 numAllowedDevices,
    LwU32 pci_domain,
    LwU32 pci_bus,
    LwU32 pci_device,
    LwU32 pci_function
)
{
    LwU32 lwrIdx;

    if (numAllowedDevices == 0)
        return LW_TRUE;

    for (lwrIdx = 0; lwrIdx < numAllowedDevices; lwrIdx++)
    {
        if ((pAllowedDevices[lwrIdx].domain == pci_domain) &&
            (pAllowedDevices[lwrIdx].bus == pci_bus) &&
            (pAllowedDevices[lwrIdx].device == pci_device) &&
            (pAllowedDevices[lwrIdx].function == pci_function))
        {
            return LW_TRUE;
        }
    }
    return LW_FALSE;
}

LwlStatus lwswitch_mods_lib_load
(
    lwswitch_mods_pci_info *pAllowedDevices,
    LwU32 numAllowedDevices,
    LwU32 printLevel
)
{
    LwlStatus retval = LWL_SUCCESS;
    mods_lwswitch_device_handle *dev_handle = NULL;
    LwU32 pci_domain;
    LwU32 pci_bus;
    LwU32 pci_device;
    LwU32 pci_function;
    LwU32 pci_vendor;
    LwU32 pci_device_id;
    LwU32 lwswitch_count = 0;
    int i = 0;

    memset(lwswitch_devices, 0, sizeof(lwswitch_devices));
    
    s_PrintLevel = printLevel;

    // Discover all switch devices.  In order to work with emulation it is
    // necessary to find by class code rather than vendor/device
    while (PCI_OK ==
            ModsDrvFindPciClassCode(PCI_CLASS_BRIDGE_LWSWITCH << 8, i++, &pci_domain,
                                    &pci_bus, &pci_device, &pci_function))
    {
        // Check if this is an LWPU device
        pci_vendor = ModsDrvPciRd16(pci_domain, pci_bus, pci_device,
                                        pci_function, PCI_ADDR_OFFSET_VENDOR);
        if (PCI_VENDOR_ID_LWIDIA != pci_vendor)
        {
            lwswitch_os_print(LWSWITCH_DBG_LEVEL_SETUP,
                "Bypassing non-LWPU bridge device...\n");
            continue;
        }

        // Check if this LWPU device is an lwswitch
        pci_device_id = ModsDrvPciRd16(pci_domain, pci_bus, pci_device,
                                            pci_function, PCI_ADDR_OFFSET_DEVID);

        if (lwswitch_lib_validate_device_id(pci_device_id))
        {
            lwswitch_os_print(LWSWITCH_DBG_LEVEL_SETUP,
                "Found LWSWitch device 0x%04x/0x%04x\n",
                pci_vendor, pci_device_id);
        }
        else
        {
            lwswitch_os_print(LWSWITCH_DBG_LEVEL_SETUP,
                "Bypassing non-lwswitch bridge device...\n");
            continue;
        }

        if (LW_FALSE == is_pci_device_allowed(pAllowedDevices,
                                              numAllowedDevices,
                                              pci_domain,
                                              pci_bus,
                                              pci_device,
                                              pci_function))
        {
            lwswitch_os_print(LWSWITCH_DBG_LEVEL_SETUP,
                "LwSwitch bridge device (%04d:%02d:%02d.%02d) ignored due to device filter...\n",
                pci_domain, pci_bus, pci_device, pci_function);
            continue;
        }

        lwswitch_os_print(LWSWITCH_DBG_LEVEL_SETUP,
                    "Found LWSWitch device at (%04d:%02d:%02d.%02d)\n",
                    pci_domain, pci_bus, pci_device, pci_function);

        if (lwswitch_count >= LWSWITCH_DEVICE_INSTANCE_MAX)
        {
            lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR, "Skipping LWSWitch - too many devices!\n");
            continue;
        }

        dev_handle = lwswitch_os_malloc(sizeof(*dev_handle));
        if (NULL == dev_handle)
        {
            lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
                        "Failed to create arch device handle!\n");
            continue;
        }
        dev_handle->pci_domain = pci_domain;
        dev_handle->pci_bus = pci_bus;
        dev_handle->pci_device = pci_device;
        dev_handle->pci_function = pci_function;

        retval = lwswitch_lib_register_device(pci_domain,
                                            pci_bus,
                                            pci_device,
                                            pci_function,
                                            pci_device_id,
                                            dev_handle,
                                            lwswitch_count,
                                            &dev_handle->lib_device);
        if (LWL_SUCCESS != retval)
        {
            lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
                "Failed to register LWSWITCH device : %d\n",
                retval);
            lwswitch_os_free(dev_handle);
            return retval;
        }

        lwswitch_lib_get_device_info(dev_handle->lib_device, &dev_handle->info);
        lwswitch_mods_initialize_device_pci_bus(dev_handle);
        retval = lwswitch_mods_map_device(dev_handle);
        if (LWL_SUCCESS != retval)
        {
            lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
                "Failed to map LWSWITCH device : %d - disabled\n",
                retval);
            lwswitch_lib_unregister_device(dev_handle->lib_device);
            lwswitch_os_free(dev_handle);
            return retval;
        }

        //
        // MODS queries LWSwitch driver for ARCH/IMPL values before chip_device
        // is even initialized. As these values can be platform dependent,
        // the MODS shim needs to determine the platform (fmodel, RTL,
        // emulation, silicon) at the earliest possible point.
        //
        (void)lwswitch_lib_load_platform_info(dev_handle->lib_device);

#if defined(SIM_BUILD)
        if (!dev_handle->lib_device->is_fmodel && !dev_handle->lib_device->is_rtlsim)
#endif
        {
            LwU32  dma_bits = 0;

            // SV10 doesnt have a DMA engine so no need to set the mask
            if (lwswitch_is_sv10_device_id(pci_device_id))
                dma_bits = 0;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
            else if (lwswitch_is_lr10_device_id(pci_device_id))
                dma_bits = 47;
#endif
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
            else if (lwswitch_is_ls10_device_id(pci_device_id))
                dma_bits = 64;
#endif
            else
            {
                // This is soley so that MODS will fail on new chips if we fail to update the dma
                // bit width.
                lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
                    "Unknown lwswitch pci device id : %04x\n",
                    pci_device_id);
                return -LWL_ERR_NOT_SUPPORTED;
            }

            if (dma_bits != 0)
            {
                retval = ModsDrvSetDmaMask(pci_domain,
                                           pci_bus,
                                           pci_device,
                                           pci_function,
                                           dma_bits);
                if (retval != 0)
                {
                    lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
                        "Setting the DMA mask failed on %04x:%02x:%02x.%02x, retval = %u\n",
                        pci_domain, pci_bus, pci_device, pci_function, retval);
                    return -LWL_ERR_GENERIC;
                }
            }
        }

        dev_handle->device_mutex = ModsDrvAllocMutex();
        // Record initialized device
        lwswitch_devices[lwswitch_count] = dev_handle;
        lwswitch_count++;

        // Initialize mapping between cpu_addr and mods descriptor.
        lwListInit(&dev_handle->memdesc_mapping_list);
    }

    return LWL_SUCCESS;
}

static void lwswitch_mods_initialize_device_pci_bus(mods_lwswitch_device_handle *dev_handle)
{
    LwU32 pci_data;

    // Enable the memory space and bus mastering
    pci_data = ModsDrvPciRd32(dev_handle->pci_domain,
                                dev_handle->pci_bus,
                                dev_handle->pci_device,
                                dev_handle->pci_function,
                                LW_CONFIG_PCI_LW_1);
    pci_data |= DRF_DEF(_CONFIG, _PCI_LW_1, _MEMORY_SPACE, _ENABLED)
             |  DRF_DEF(_CONFIG, _PCI_LW_1, _BUS_MASTER, _ENABLED);
    ModsDrvPciWr32(dev_handle->pci_domain,
                    dev_handle->pci_bus,
                    dev_handle->pci_device,
                    dev_handle->pci_function,
                    LW_CONFIG_PCI_LW_1,
                    pci_data);
}

static LwlStatus lwswitch_mods_map_device(mods_lwswitch_device_handle *dev_handle)
{
    lwlink_pci_info *info = dev_handle->info;

    if (PCI_OK != ModsDrvPciGetBarInfo(dev_handle->pci_domain,
                                        dev_handle->pci_bus,
                                        dev_handle->pci_device,
                                        dev_handle->pci_function,
                                        0,
                                        &info->bars[0].baseAddr,
                                        &info->bars[0].barSize))
    {
        info->bars[0].baseAddr = 0;
        info->bars[0].barSize = 0;
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR, "Pci get bar info failed!\n");
        return -LWL_PCI_ERROR;
    }

    // Determine whether GetBarInfo actually found a bar
    if ((0 == info->bars[0].baseAddr) ||
        (~0xFU == info->bars[0].baseAddr) ||
        (~(LwU64)0xFU == info->bars[0].baseAddr) ||
        (0 == info->bars[0].barSize))
    {
        info->bars[0].baseAddr = 0;
        info->bars[0].barSize = 0;
        return -LWL_PCI_ERROR;
    }

    info->bars[0].pBar = ModsDrvMapDeviceMemory(info->bars[0].baseAddr,
                                                info->bars[0].barSize,
                                                ATTRIB_UC, PROTECT_READ_WRITE);

    return LWL_SUCCESS;
}

long lwswitch_mods_service_interrupts(void *arg)
{
    LwU32 prevDevId;
    mods_lwswitch_device_handle *dev_handle = (mods_lwswitch_device_handle *)arg;

    //
    // Note that the mods driver has disabled interrupts on the device.
    //

    if (NULL == arg)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "An interrupt was fired for an LWSwitch device, but the passed device "
            "handle was NULL!");
        return -1;
    }

    prevDevId = ModsDrvSetSwitchId(dev_handle->lib_device->os_instance);

    ModsDrvAcquireMutex(dev_handle->device_mutex);

    lwswitch_os_print(LWSWITCH_DBG_LEVEL_SETUP,
        "An interrupt was fired for LWSwitch device (%04x:%02x:%02x.%x)\n",
        dev_handle->pci_domain, dev_handle->pci_bus, dev_handle->pci_device, dev_handle->pci_function);
    // Service the interrupt.
    lwswitch_lib_service_interrupts(dev_handle->lib_device);
    lwswitch_lib_enable_interrupts(dev_handle->lib_device);

    ModsDrvReleaseMutex(dev_handle->device_mutex);

    ModsDrvSetSwitchId(prevDevId);

    return 0;
}

static LwlStatus lwswitch_mods_get_irq_info
(
    const mods_lwswitch_device_handle *dev_handle,
    IrqParams *irqInfo
)
{
    LWSWITCH_GET_IRQ_INFO_PARAMS params;
    LwlStatus retval;
    LwU32     i;

    memset(irqInfo, 0, sizeof(*irqInfo));
    irqInfo->irqNumber       = dev_handle->irq;
    irqInfo->barAddr         = dev_handle->info->bars[0].baseAddr;
    irqInfo->barSize         = dev_handle->info->bars[0].barSize;
    irqInfo->irqType         = MODS_XP_IRQ_TYPE_INT;
    irqInfo->pciDev.domain   = dev_handle->pci_domain;
    irqInfo->pciDev.bus      = dev_handle->pci_bus;
    irqInfo->pciDev.device   = dev_handle->pci_device;
    irqInfo->pciDev.function = dev_handle->pci_function;

    retval = lwswitch_lib_ctrl(dev_handle->lib_device,
                    CTRL_LWSWITCH_GET_IRQ_INFO,
                    &params, sizeof(params), NULL);

    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    irqInfo->maskInfoCount   = params.maskInfoCount;
    for (i = 0; i < params.maskInfoCount; i++)
    {
        // Set the mask to AND out during servicing in order to avoid int storm.
        irqInfo->maskInfoList[i].irqPendingOffset = params.maskInfoList[i].irqPendingOffset;
        irqInfo->maskInfoList[i].irqEnabledOffset = params.maskInfoList[i].irqEnabledOffset;
        irqInfo->maskInfoList[i].irqEnableOffset  = params.maskInfoList[i].irqEnableOffset;
        irqInfo->maskInfoList[i].irqDisableOffset = params.maskInfoList[i].irqDisableOffset;
        irqInfo->maskInfoList[i].andMask          = 0;
        irqInfo->maskInfoList[i].orMask           = 0xffffffff;
        irqInfo->maskInfoList[i].maskType         = 0; // 32 bit access
    }

    return retval;
}

static LwlStatus lwswitch_mods_initialize_device_interrupt
(
    mods_lwswitch_device_handle *dev_handle
)
{
    lwswitch_device *device = dev_handle->lib_device;
    LwlStatus retval;

    if (PCI_OK != ModsDrvPciGetIRQ(dev_handle->pci_domain,
                                    dev_handle->pci_bus,
                                    dev_handle->pci_device,
                                    dev_handle->pci_function,
                                    &dev_handle->irq))
    {
        dev_handle->irq = ModsDrvPciRd32(dev_handle->pci_domain,
                             dev_handle->pci_bus,
                             dev_handle->pci_device,
                             dev_handle->pci_function,
                             LW_CONFIG_PCI_LW_15);
        dev_handle->irq = DRF_VAL(_CONFIG, _PCI_LW_15, _INTR_LINE, dev_handle->irq);
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_SETUP,
            "LWSwitch device (%04x:%02x:%02x.%x) has legacy IRQ %d assigned.\n",
            dev_handle->pci_domain,
            dev_handle->pci_bus,
            dev_handle->pci_device,
            dev_handle->pci_function,
            dev_handle->irq
            );
    }

    IrqParams irqInfo;
    retval = lwswitch_mods_get_irq_info(dev_handle, &irqInfo);

    if (retval != LWL_SUCCESS)
    {
        return -LWL_ERR_GENERIC;
    }

    irqInfo.irqType = MODS_XP_IRQ_TYPE_MSI;
    // MSI is unsupported in emulation/rtl/fmodel
    if (device->is_emulation || 
#if defined(SIM_BUILD)
        device->is_rtlsim || 
        device->is_fmodel || 
#endif
        !ModsDrvHookInt(&irqInfo, lwswitch_mods_service_interrupts, dev_handle))
    {
        irqInfo.irqType = MODS_XP_IRQ_TYPE_INT;
        if (!ModsDrvHookInt(&irqInfo, lwswitch_mods_service_interrupts, dev_handle))
        {
            return -LWL_ERR_GENERIC;
        }
    }

    lwswitch_os_print(LWSWITCH_DBG_LEVEL_SETUP,
        "LWSwitch device (%04x:%02x:%02x.%x) hooked %s.\n",
        dev_handle->pci_domain,
        dev_handle->pci_bus,
        dev_handle->pci_device,
        dev_handle->pci_function,
        (irqInfo.irqType == MODS_XP_IRQ_TYPE_MSI) ? "MSI" : "legacy interrupt");

    return retval;
}

static void lwswitch_mods_shutdown_device_interrupt
(
    mods_lwswitch_device_handle *dev_handle
)
{
    IrqParams irqInfo;
    lwswitch_mods_get_irq_info(dev_handle, &irqInfo);
    ModsDrvUnhookInt(&irqInfo, lwswitch_mods_service_interrupts, dev_handle);
}

LwlStatus lwswitch_mods_lib_unload(void)
{
    LwU32 prevDevId = ModsDrvSetSwitchId(~0U);
    LwU32 slot;

    if (s_TaskDispatcherTid != -1)
    {
        s_TaskDispatcherRunning = LW_FALSE;
        ModsDrvJoinThread(s_TaskDispatcherTid);
        s_TaskDispatcherTid = -1;
    }

    // Unload all devices
    for (slot = 0; slot < LWSWITCH_DEVICE_INSTANCE_MAX; slot++)
    {
        if (lwswitch_devices[slot])
        {
            ModsDrvSetSwitchId(slot);
            lwswitch_lib_disable_interrupts(lwswitch_devices[slot]->lib_device);
            lwswitch_mods_shutdown_device_interrupt(lwswitch_devices[slot]);
            lwswitch_lib_shutdown_device(lwswitch_devices[slot]->lib_device);
            lwswitch_mods_unmap_device(lwswitch_devices[slot]);
            lwswitch_lib_unregister_device(lwswitch_devices[slot]->lib_device);
            if (lwswitch_devices[slot]->device_mutex != NULL)
            {
                ModsDrvFreeMutex(lwswitch_devices[slot]->device_mutex);
                lwswitch_devices[slot]->device_mutex = NULL;
            }
            lwswitch_os_free(lwswitch_devices[slot]);
            lwswitch_devices[slot] = NULL;
        }
    }

    ModsDrvSetSwitchId(prevDevId);

    return LWL_SUCCESS;
}

LwlStatus lwswitch_mods_initialize_all_devices(void)
{
    LwlStatus retval = LWL_SUCCESS;
    LwU32 prevDevId = ModsDrvSetSwitchId(~0U);
    LwU32 slot;
    LwBool bAnyDeviceInitialized = LW_FALSE;

    // Initialize all devices
    for (slot = 0; slot < LWSWITCH_DEVICE_INSTANCE_MAX; slot++)
    {
        if (lwswitch_devices[slot])
        {
            ModsDrvSetSwitchId(slot);

            // Try to initialize device.  On failure disable this device and continue.
            retval = lwswitch_lib_initialize_device(lwswitch_devices[slot]->lib_device);
            if (LWL_SUCCESS != retval)
            {
                lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
                    "Failed to initialize LWSWITCH device : %d - disabled\n",
                    retval);
                goto load_init_fail;
            }

            retval = lwswitch_mods_initialize_device_interrupt(lwswitch_devices[slot]);
            if (LWL_SUCCESS != retval)
            {
                lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
                    "Failed to hook interrupts for LWSWITCH device : %d\n",
                    retval);
                goto load_init_intr_fail;
            }

            lwswitch_lib_enable_interrupts(lwswitch_devices[slot]->lib_device);

            //
            // device_mutex held here because post_init entries may call soeService_HAL()
            // with IRQs on. see bug 2856314 for more info
            //
            ModsDrvAcquireMutex(lwswitch_devices[slot]->device_mutex);
            retval = lwswitch_lib_post_init_device(lwswitch_devices[slot]->lib_device);
            ModsDrvReleaseMutex(lwswitch_devices[slot]->device_mutex);
            if (LWL_SUCCESS != retval)
            {
                lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
                    "Post initialization failed for LWSWITCH device : %d\n",
                    retval);
                goto load_init_intr_fail;
            }

            bAnyDeviceInitialized = LW_TRUE;

            continue;

load_init_intr_fail:
            lwswitch_lib_shutdown_device(lwswitch_devices[slot]->lib_device);
load_init_fail:
            lwswitch_mods_unmap_device(lwswitch_devices[slot]);
            lwswitch_lib_unregister_device(lwswitch_devices[slot]->lib_device);
            lwswitch_os_free(lwswitch_devices[slot]);
            lwswitch_devices[slot] = NULL;
            ModsDrvSetSwitchId(prevDevId);
            return retval;
        }
    }

    if (bAnyDeviceInitialized)
    {
        s_TaskDispatcherRunning = LW_TRUE;
        s_TaskDispatcherTid     = (LwS32)ModsDrvCreateThread(lwswitch_task_dispatch,
                                                             0,
                                                             0,
                                                             "LwSwitchTaskDispatcher");
    }

    ModsDrvSetSwitchId(prevDevId);

    return LWL_SUCCESS;
}

static void lwswitch_mods_unmap_device(mods_lwswitch_device_handle *dev_handle)
{
    ModsDrvUnMapDeviceMemory(dev_handle->info->bars[0].pBar);
    dev_handle->info->bars[0].pBar = NULL;
}

LwBool lwswitch_os_check_failure(void *os_handle)
{
    // FENCE detection/support is not required on MODS.
    return LW_FALSE;
}

//
// Get current time in seconds.nanoseconds
// In this implementation, the time is epoch time (midnight UTC of January 1, 1970)
//

LwU64 lwswitch_os_get_platform_time(void)
{
    return ModsDrvGetTimeNS();
}

void
lwswitch_os_print
(
    const int  log_level,
    const char *fmt,
    ...
)
{
    va_list args;
    int    pri;

    // Print at the same priorities that are used in RM for similar information
    // (i.e. lowered priorities in release, sanity, and manufacturing builds)
    switch (log_level)
    {
        case LWSWITCH_DBG_LEVEL_MMIO:
        case LWSWITCH_DBG_LEVEL_INFO:
        case LWSWITCH_DBG_LEVEL_SETUP:
            pri = (s_PrintLevel < PRI_LOW) ? PRI_NORMAL : PRI_DEBUG;
            break;
        case LWSWITCH_DBG_LEVEL_WARN:
#if !defined(DEBUG) || defined(LINUX_MFG) || defined(MACOSX_MFG) || defined(WIN_MFG) || defined(SANITY_BUILD)
            pri = (s_PrintLevel < PRI_NORMAL) ? PRI_WARN : PRI_LOW;
#else
            pri = PRI_WARN;
#endif
            break;
        case LWSWITCH_DBG_LEVEL_ERROR:
#if !defined(DEBUG) || defined(LINUX_MFG) || defined(MACOSX_MFG)|| defined(WIN_MFG) || defined(SANITY_BUILD)
            pri = (s_PrintLevel < PRI_NORMAL) ? PRI_ERR : PRI_LOW;
#elif defined(SIM_BUILD) // See bug 1610644 and 1733774
            pri = PRI_NORMAL;
#else
            pri = PRI_ERR;
#endif
            break;
        default:
            pri = PRI_ERR;
            break;
    }

    va_start(args, fmt);
    ModsDrvVPrintf(pri, fmt, args);
    va_end(args);
}

LwlStatus lwswitch_mods_ctrl(LwU32 instance, LwU32 ctrlid, void *pParams, LwU32 paramSize)
{
    mods_lwswitch_device_handle *dev_handle;
    LwlStatus retval;
    LwU32 prevDevId;

    dev_handle = lwswitch_devices[instance];
    if (NULL == dev_handle)
    {
        return -LWL_BAD_ARGS;
    }

    prevDevId = ModsDrvSetSwitchId(instance);

#if !defined(SIM_BUILD)
    ModsDrvAcquireMutex(dev_handle->device_mutex);
#endif  //!defined(SIM_BUILD)

    retval = lwswitch_lib_ctrl(dev_handle->lib_device, ctrlid,
                               pParams, paramSize, NULL);

#if !defined(SIM_BUILD)
    ModsDrvReleaseMutex(dev_handle->device_mutex);
#endif  //!defined(SIM_BUILD)

    ModsDrvSetSwitchId(prevDevId);

    return retval;
}

LwlStatus lwswitch_mods_get_device_info(LwU32 instance, LwU32 *linkMask, struct lwlink_pci_info *retPciInfo)
{
    mods_lwswitch_device_handle *dev_handle = lwswitch_devices[instance];
    LWSWITCH_GET_LWLINK_CAPS_PARAMS params;
    struct lwlink_pci_info *pciInfo;
    LwlStatus retval;
    LwU32 prevDevId;

    if (NULL == dev_handle)
    {
        return -LWL_NOT_FOUND;
    }

    prevDevId = ModsDrvSetSwitchId(instance);

    lwswitch_lib_get_device_info(dev_handle->lib_device, &pciInfo);
    *retPciInfo = *pciInfo;

    retval = lwswitch_mods_ctrl(instance,
                                CTRL_LWSWITCH_GET_LWLINK_CAPS,
                                &params,
                                sizeof(params));
    *linkMask = params.enabledLinkMask;

    ModsDrvSetSwitchId(prevDevId);

    return retval;
}

LwlStatus lwswitch_mods_get_bios_rom_version(LwU32 instance, lwswitch_mods_bios *retBiosInfo)
{
    LWSWITCH_GET_BIOS_INFO_PARAMS params;
    LwlStatus retval;

    retval = lwswitch_mods_ctrl(instance,
                                CTRL_LWSWITCH_GET_BIOS_INFO,
                                &params,
                                sizeof(params));

    retBiosInfo->version = params.version;

    return retval;
}

LwlStatus lwswitch_os_read_registry_dword(void *os_handle, const char *name, LwU32 *data)
{
    // We don't have a per-device look-up like GPU yet, just a global key
    if (ModsDrvReadRegistryDword("LwSwitch", name, (UINT32 *)data))
        return LWL_SUCCESS;

    return -LWL_ERR_GENERIC;
}

void
lwswitch_os_override_platform(void *os_handle, LwBool *rtlsim)
{
    // Differentiate between rtlsim and HW for sim mods.
    if (ModsDrvGetSimulationMode() == SIM_MODE_HARDWARE)
    {
        *rtlsim = LW_FALSE;
    }
}

static LwlStatus
_lwswitch_get_memory_descriptor
(
    mods_lwswitch_device_handle *dev_handle,
    void *virt_addr,
    void **descriptor
)
{
    mods_lwswitch_memdesc_mapping *lwrr = NULL;

    lwListForEachEntry(lwrr, &dev_handle->memdesc_mapping_list, entry)
    {
        if (lwrr->virt_addr == virt_addr)
        {
            *descriptor = lwrr->descriptor;
            return LWL_SUCCESS;
        }
    }

    return -LWL_NOT_FOUND;
}

static LwlStatus
_lwswitch_add_memory_descriptor
(
    mods_lwswitch_device_handle *dev_handle,
    void *virt_addr,
    void *descriptor
)
{
    mods_lwswitch_memdesc_mapping *memdesc;

    memdesc = (mods_lwswitch_memdesc_mapping*)
        ModsDrvAlloc(sizeof(mods_lwswitch_memdesc_mapping));
    if (!memdesc)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "%s: No mem!\n", __FUNCTION__);
        return -LWL_NO_MEM;
    }

    memdesc->descriptor = descriptor;
    memdesc->virt_addr = virt_addr;
    lwListAdd(&memdesc->entry, &dev_handle->memdesc_mapping_list);

    return LWL_SUCCESS;
}

static void
_lwswitch_free_memory_descriptor
(
    mods_lwswitch_device_handle *dev_handle,
    void *virt_addr
)
{
    mods_lwswitch_memdesc_mapping *lwrr = NULL;
    mods_lwswitch_memdesc_mapping *next = NULL;

    lwListForEachEntry_safe(lwrr, next, &dev_handle->memdesc_mapping_list, entry)
    {
        if (lwrr->virt_addr == virt_addr)
        {
            lwListDel(&lwrr->entry);
            ModsDrvFree(lwrr);
            return;
        }
    }
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
    mods_lwswitch_device_handle *dev_handle;
    void *descriptor;
    LwlStatus status;

    if (!os_handle || !virt_addr)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "%s: Bad args!\n", __FUNCTION__);
        LWSWITCH_OS_ASSERT(0);
        return -LWL_BAD_ARGS;
    }

    dev_handle = (mods_lwswitch_device_handle *)os_handle;

    descriptor = ModsDrvAllocPagesForPciDev((size_t)size, MODS_FALSE,
        dev_handle->dma_addr_width, ATTRIB_WB, dev_handle->pci_domain,
        dev_handle->pci_bus, dev_handle->pci_device, dev_handle->pci_function);

    if (!descriptor)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "%s: DMA memory allocation failed!\n",
            __FUNCTION__);
        LWSWITCH_OS_ASSERT(0);
        return -LWL_NO_MEM;
    }

    *virt_addr = ModsDrvMapPages(descriptor, 0, (size_t)size, PROTECT_READ_WRITE);

    if (!*virt_addr)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "%s: Failed to map pages and get cpu address!\n",
            __FUNCTION__);
        LWSWITCH_OS_ASSERT(0);
        goto lwswitch_os_alloc_map_pages_fail;
    }

    status = _lwswitch_add_memory_descriptor(dev_handle, *virt_addr, descriptor);
    if (status != LWL_SUCCESS)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "%s: Failed to add memory descriptor\n",
            __FUNCTION__);
        LWSWITCH_OS_ASSERT(0);
        goto lwswitch_os_alloc_add_memdesc_fail;
    }

    return LWL_SUCCESS;

lwswitch_os_alloc_add_memdesc_fail:
    ModsDrvUnMapPages(*virt_addr);

lwswitch_os_alloc_map_pages_fail:
    ModsDrvFreePages(descriptor);

    return -LWL_ERR_GENERIC;
}

void
lwswitch_os_free_contig_memory
(
    void *os_handle,
    void *virt_addr,
    LwU32 size
)
{
    mods_lwswitch_device_handle *dev_handle;
    LwlStatus status;
    void *descriptor;

    if (!os_handle || !virt_addr)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "%s: Bad args!\n",
            __FUNCTION__);
        LWSWITCH_OS_ASSERT(0);
        return;
    }

    dev_handle = (mods_lwswitch_device_handle *)os_handle;
    
    status = _lwswitch_get_memory_descriptor(dev_handle, virt_addr, &descriptor);
    if (status != LWL_SUCCESS)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "%s: Failed to get memory descriptor\n",
            __FUNCTION__);
        LWSWITCH_OS_ASSERT(0);
        return;
    }

    ModsDrvUnMapPages(virt_addr);
    ModsDrvFreePages(descriptor);

    _lwswitch_free_memory_descriptor(dev_handle, virt_addr);
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
    mods_lwswitch_device_handle *dev_handle;
    void *descriptor;
    LwlStatus status;

    if (!os_handle || !cpu_addr || !dma_handle)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "%s: Bad args!\n",
            __FUNCTION__);
        LWSWITCH_OS_ASSERT(0);
        return -LWL_BAD_ARGS;
    }

    dev_handle = (mods_lwswitch_device_handle *)os_handle;

    status = _lwswitch_get_memory_descriptor(dev_handle, cpu_addr, &descriptor);
    if (status != LWL_SUCCESS)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "%s: Failed to get memory descriptor\n",
            __FUNCTION__);
        LWSWITCH_OS_ASSERT(0);
        return status;
    }

#if !defined(SIM_BUILD)
    if (ModsDrvDmaMapMemory(dev_handle->pci_domain, dev_handle->pci_bus,
            dev_handle->pci_device, dev_handle->pci_function, descriptor))
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "%s: Failed to map DMA region!\n",
            __FUNCTION__);
        LWSWITCH_OS_ASSERT(0);
        return -LWL_ERR_GENERIC;
    }
#endif

    *dma_handle = (LwU64)ModsDrvGetMappedPhysicalAddress(dev_handle->pci_domain,
                                                         dev_handle->pci_bus,
                                                         dev_handle->pci_device,
                                                         dev_handle->pci_function,
                                                         descriptor,
                                                         0);

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
#if !defined(SIM_BUILD)
    mods_lwswitch_device_handle *dev_handle;
    void *descriptor;
    LwlStatus status;

    if (!os_handle || !cpu_addr)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "%s: Bad args!\n",
            __FUNCTION__);
        LWSWITCH_OS_ASSERT(0);
        return -LWL_BAD_ARGS;
    }

    dev_handle = (mods_lwswitch_device_handle *)os_handle;

    status = _lwswitch_get_memory_descriptor(dev_handle, cpu_addr, &descriptor);
    if (status != LWL_SUCCESS)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "%s: Failed to get memory descriptor\n",
            __FUNCTION__);
        LWSWITCH_OS_ASSERT(0);
        return status;
    }

    if (ModsDrvDmaUnmapMemory(dev_handle->pci_domain, dev_handle->pci_bus,
            dev_handle->pci_device, dev_handle->pci_function, descriptor))
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "%s: Failed to unmap DMA region!\n",
            __FUNCTION__);
        LWSWITCH_OS_ASSERT(0);
        return -LWL_ERR_GENERIC;
    }
#endif

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_os_set_dma_mask
(
    void *os_handle,
    LwU32 dma_addr_width
)
{
    mods_lwswitch_device_handle *dev_handle;
    PHYSADDR dma_base_address;
    LwU64 dma_mask;

    if (!os_handle)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "%s: Bad args!\n",
            __FUNCTION__);
        LWSWITCH_OS_ASSERT(0);
        return -LWL_BAD_ARGS;
    }

    dev_handle = (mods_lwswitch_device_handle *)os_handle;

    dma_mask = (dma_addr_width = 64) ? LW_U64_MAX :
        (1ULL << dma_addr_width) - 1ULL;

    if (ModsDrvSetupDmaBase(dev_handle->pci_domain, dev_handle->pci_bus,
            dev_handle->pci_device, dev_handle->pci_function,
            MODSDRV_PPC_TCE_BYPASS_DEFAULT, dma_mask, &dma_base_address))
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
            "%s: Failed to setup DMA base!\n",
            __FUNCTION__);
        LWSWITCH_OS_ASSERT(0);
        return -LWL_ERR_GENERIC;
    }

    dev_handle->dma_addr_width = dma_addr_width;

    return LWL_SUCCESS;
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
    LwU32 unused;

    if (ModsDrvReadRegistryBinary("LwSwitch",
                                  name,
                                  data, (UINT32*)&unused))
        return LWL_SUCCESS;

    return -LWL_ERR_NOT_SUPPORTED;
}

LwBool
lwswitch_os_is_uuid_in_blacklist
(
    LwUuid *uuid
)
{
    return LW_FALSE;
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
    // Not required for mods
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
    // Not required for mods
    return LWL_SUCCESS;
}

void *
lwswitch_os_malloc_trace
(
    LwLength size,
    const char *file,
    LwU32 line
)
{
   return ModsDrvAlloc(size);
}

void
lwswitch_os_free
(
    void *pMem
)
{
    return ModsDrvFree(pMem);
}

void *
lwswitch_os_memset
(
    void *pDest,
    int value,
    LwLength size
)
{
    ModsDrvMemSet(pDest, value, size);
    return pDest;
}

void *
lwswitch_os_memcpy
(
    void *pDest,
    const void *pSrc,
    LwLength size
)
{
    ModsDrvMemCopy(pDest, pSrc, size);
    return pDest;
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
    const volatile void *pAddress
)
{
    return ModsDrvMemRd32((const volatile void *)(pAddress));
}

void
lwswitch_os_mem_write32
(
    volatile void *pAddress,
    LwU32 data
)
{
    ModsDrvMemWr32(pAddress, data);
}

LwU64
lwswitch_os_mem_read64
(
    const volatile void *pAddress
)
{
    return ModsDrvMemRd64((const volatile void *)(pAddress));
}

void
lwswitch_os_mem_write64
(
    volatile void *pAddress,
    LwU64 data
)
{
    return ModsDrvMemWr64(pAddress, data);
}

int
lwswitch_os_snprintf
(
    char *pString,
    LwLength size,
    const char *pFormat,
    ...
)
{
    va_list args;
    int charsWritten;

    va_start(args, pFormat);
    charsWritten = vsnprintf(pString, size, pFormat, args);
    va_end(args);

    return charsWritten;
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
        va_list arglist;
        char fmtPrintk[LWSWITCH_LOG_BUFFER_SIZE];

        va_start(arglist, fmt);
        vsnprintf(fmtPrintk, sizeof(fmtPrintk), fmt, arglist);
        va_end(arglist);
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR, fmtPrintk);
        ModsDrvBreakPoint(__FILE__, __LINE__);
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
    ModsDrvSleep(ms);
}

LwlStatus
lwswitch_os_acquire_fabric_mgmt_cap
(
    void *osPrivate,
    LwU64 fd
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

int
lwswitch_os_is_fabric_manager
(
    void *osPrivate
)
{
    return 1;
}

int
lwswitch_os_is_admin
(
    void
)
{
    return 1;
}

LwlStatus
lwswitch_os_get_os_version
(
    LwU32 *pMajorVer,
    LwU32 *pMinorVer,
    LwU32 *pBuildNum
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
}

/*!
 * @brief: OS Specific handling to add an event.
 */
LwlStatus
lwswitch_os_add_client_event
(
    void            *osHandle,
    void            *osPrivate,
    LwU32           eventId
)
{
    return -LWL_ERR_NOT_IMPLEMENTED;
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
    return -LWL_ERR_NOT_IMPLEMENTED;
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
    return -LWL_ERR_NOT_IMPLEMENTED;
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
    return -LWL_ERR_NOT_IMPLEMENTED;
}
