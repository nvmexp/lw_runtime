/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file   lwlink.c
 * @brief  Lwlink stubs for Vmware
 */
#include "os-interface.h"
#include "vmware_lwswitch.h"
#include "lwswitch-vmkmgmt.h"
#include "lwlink_common.h"
#include "lwlink_proto.h"
#include "lw.h"
#include "lw-vmware.h"
#include "lw-linux.h"
#include "lw-proto.h"
#include "vmkapi.h"
#include "ioctl_lwswitch.h"

lw_vmware_state_t *lwswitch_ctl_device;
vmk_MgmtHandle    lwswitch_mgmt_handle;

// Global driver state
typedef struct
{
    LwBool initialized;
    vmk_atomic64 count;
    vmk_Semaphore driver_mutex;
    vmk_SList_Links devices;
} LWSWITCH;

static LWSWITCH lwswitch = {0};

static VMK_ReturnStatus lwswitch_register_ctl_device(void);

#if LWCFG(GLOBAL_ARCH_LWSWITCH)
/*
 * LwSwitch SVNP01
 */
#include "export_lwswitch.h"

#define LWSWITCH_OS_ASSERT(_cond)                                               \
    lwswitch_os_assert_log((_cond), "LWSwitch: Assertion failed in OS layer \n")
static char lw_error_string[LWSWITCH_LOG_BUFFER_SIZE];

LwU32 default_debug_level = LWSWITCH_DBG_LEVEL_ERROR;

extern vmk_HeapID lw_small_alloc_heap;

LwBool
lwswitch_os_check_failure
(
    void *os_handle
)
{
    return LW_FALSE;
}

LwU64
lwswitch_os_get_platform_time
(
    void
)
{
    return 0ULL;
}

void
lwswitch_os_print
(
    const int  log_level,
    const char *fmt,
    ...
)
{
    char *p = lw_error_string;
    va_list arglist;
    vmk_ByteCount chars_written = 0;
    VMK_ReturnStatus status;

    if (log_level >= default_debug_level)
    {
        va_start(arglist, fmt);
        status = vmk_StringVFormat(p, sizeof(lw_error_string), &chars_written,
                                   fmt, arglist);
        VMK_ASSERT(status == VMK_OK);
        va_end(arglist);

        if (p[chars_written-1] == '\n')
        {
            p[chars_written-1] = '\0';
        }

        vmk_LogNoLevel(VMK_LOG_URGENCY_NORMAL, "%s\n", p);
    }
}

LwlStatus
lwswitch_os_read_registry_dword
(
    void *os_handle,
    const char *name,
    LwU32 *data
)
{
    return -1;
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

LwU32
lwswitch_os_get_device_count
(
    void
)
{
    return 0;
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
lwswitch_os_alloc_contig_memory
(
    void *os_handle,
    void **virt_addr,
    LwU32 size,
    LwBool force_dma32
)
{
    void *lw_gfp_addr = NULL;
    lw_vmware_state_t *lwv = NULL;
    lw_state_t *lw = NULL;

    if (!virt_addr)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
                        "%s: virt_addr arg is NULL!\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }
    if (!os_handle)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
                        "%s: os_handle arg is NULL!\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    lwv = (lw_vmware_state_t *)os_handle;
    lw = LW_STATE_PTR(lwv);
    if (!lw)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
                "%s: lw_state is NULL!\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }
    lw->force_dma32_alloc = force_dma32;


    lw_gfp_addr = lw_mem_pool_alloc_pages(lw_calc_order(size), lw);

    if (!lw_gfp_addr)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
                          "%s: lwpu-lwswitch: unable to allocate kernel memory!\n",
                          __FUNCTION__);
        return -LWL_NO_MEM;
    }

    *virt_addr = lw_gfp_addr;

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
    lw_vmware_state_t *lwv = (lw_vmware_state_t *)os_handle;

    if (!lwv)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
                    "%s: Bad args!\n", __FUNCTION__);
        return;
    }

    lw_mem_pool_free_pages(virt_addr, lw_calc_order(size), LW_STATE_PTR(lwv));

    return;
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
    vmk_MA dma_ma = 0;

    if (!cpu_addr)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
                          "%s: Bad args!\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }
    vmk_VA2MA((vmk_VA)cpu_addr, size, &dma_ma);
    *dma_handle = (LwU64)dma_ma;

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
    // Unmapping oclwrs during lwswitch_os_free_contig_memory
    return LWL_SUCCESS;
}

LwlStatus
lwswitch_os_set_dma_mask
(
    void *os_handle,
    LwU32 dma_addr_width
)
{
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
    return -LWL_ERR_NOT_IMPLEMENTED;
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
    void *ptr = NULL;

    ptr  = vmk_HeapAllocWithTimeout(lw_small_alloc_heap, size,
                                    VMK_TIMEOUT_UNLIMITED_MS);
    return ptr;
}

void
lwswitch_os_free
(
    void *ptr
)
{
    vmk_HeapFree(lw_small_alloc_heap, (ptr));
    return;
}

LwLength
lwswitch_os_strlen
(
    const char *str
)
{
    return vmk_Strnlen(str, MAX_STRING_LENGTH);
}

char*
lwswitch_os_strncpy
(
    char *dest,
    const char *src,
    LwLength length
)
{
    return vmk_Strncpy(dest, src, length);
}

int
lwswitch_os_strncmp
(
    const char *s1,
    const char *s2,
    LwLength length
)
{
    return vmk_Strncmp(s1, s2, length);
}

void *
lwswitch_os_memset
(
    void *dest,
    int value,
    LwLength size
)
{
    return vmk_Memset(dest, (int)value, size);
}

void *
lwswitch_os_memcpy
(
    void *dest,
    const void *src,
    LwLength size
)
{
    return vmk_Memcpy(dest, src, size);
}

int
lwswitch_os_memcmp
(
    const void *s1,
    const void *s2,
    LwLength size
)
{
    return vmk_Memcmp(s1, s2, size);
}

LwU32
lwswitch_os_mem_read32
(
    const volatile void * address
)
{
    return (*(const volatile vmk_uint32*)(address));
}

void
lwswitch_os_mem_write32
(
    volatile void *address,
    LwU32 data
)
{
    (*(volatile vmk_uint32 *)(address)) = data;
}

LwU64
lwswitch_os_mem_read64
(
    const volatile void *address
)
{
    return (*(const volatile vmk_uint64 *)(address));
}

void
lwswitch_os_mem_write64
(
    volatile void *address,
    LwU64 data
)
{
    (*(volatile vmk_uint64 *)(address)) = data;
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
    vmk_ByteCount chars_written;

    va_start(arglist, fmt);
    vmk_StringVFormat(dest, size, &chars_written, fmt, arglist);
    va_end(arglist);

    return (int)chars_written;
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
    vmk_ByteCount chars_written = 0;

    vmk_StringVFormat(buf, size, &chars_written, fmt, arglist);
    return (LwS32)chars_written;
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
        char fmt_printk[LWSWITCH_LOG_BUFFER_SIZE];
        vmk_ByteCount chars_written;

        va_start(arglist, fmt);
        vmk_StringVFormat(fmt_printk, LWSWITCH_LOG_BUFFER_SIZE, &chars_written, fmt, arglist);
        va_end(arglist);
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR, fmt_printk);
#if defined(DEBUG)
#if defined(CONFIG_X86_REMOTE_DEBUG) || defined(CONFIG_KGDB)
            __asm__ __volatile__ ("int $3");
#elif defined(CONFIG_KDB)
            KDB_ENTER();
#endif // defined(CONFIG_X86_REMOTE_DEBUG) || defined(CONFIG_KGDB)
#endif // defined(DEBUG)
    }
}

void
lwswitch_os_sleep
(
    unsigned int milliSeconds
)
{
    unsigned long microSeconds;

    if (lw_in_interrupt() && (milliSeconds > LW_MAX_ISR_DELAY_MS))
    {
        return;
    }

    microSeconds = milliSeconds * 1000;
    if (!LW_MAY_SLEEP())
    {
        vmk_DelayUsecs(microSeconds);
        return;
    }
    vmk_WorldSleep(microSeconds);
    return;
}

LwlStatus
lwswitch_os_acquire_fabric_mgmt_cap
(
    void *osPrivate,
    LwU64 capDescriptor
)
{
    return LWL_SUCCESS;
}

int
lwswitch_os_is_fabric_manager
(
    void *osPrivate
)
{
    return 0;
}

int
lwswitch_os_is_admin
(
    void
)
{
    return 0;
}

LwlStatus
lwswitch_os_get_os_version
(
    LwU32 *pMajorVer,
    LwU32 *pMinorVer,
    LwU32 *pBuildNum
)
{
    if (pMajorVer)
        *pMajorVer = 2;
    if (pMinorVer)
#if VMKAPI_REVISION >= VMK_REVISION_FROM_NUMBERS(2,4,0,0)
        *pMinorVer = 4;
#else
        *pMinorVer = 3;
#endif
    if (pBuildNum)
        *pBuildNum = 0;

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
    // TODO: Implement after lwlink driver is created
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
    // TODO: Implement after lwlink driver is created
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

/*
 * lwswitch device open entry point.  Sessions are created here.
 */
static VMK_ReturnStatus lwswitch_open(
    vmk_CharDevFdAttr *vmkChardevAttr
)
{
    return VMK_OK;
}

/*
 * lwswitch device close entry point.
 */
static VMK_ReturnStatus lwswitch_close(
    vmk_CharDevFdAttr *vmkChardevAttr
)
{
    return VMK_OK;
}

/*
 * Handle lwswitch device polling events.
 */
VMK_ReturnStatus lwswitch_poll
(
    vmk_CharDevFdAttr *attr,
    vmk_PollContext pollCtx,
    unsigned *pollMask
)
{
    return VMK_OK;
}

#define LW_CTL_DEVICE_ONLY(lw)                 \
{                                              \
    if (((lw)->flags & LW_FLAG_CONTROL) == 0)  \
    {                                          \
        status = VMK_ILWALID_IOCTL;            \
        goto done;                             \
    }                                          \
}

#define LWSWITCH_CTL_CHECK_PARAMS(type, size) (sizeof(type) == size ? 0 : VMK_ILWALID_IOCTL)

static VMK_ReturnStatus lwswitch_device_ioctl(
    vmk_CharDevFdAttr *vmkChardevAttr,
    unsigned int cmd,
    vmk_uintptr_t userData,
    vmk_IoctlCallerSize callerSize,
    vmk_int32 *result)
{
    // TODO: Implement once lwlink driver is complete
    return VMK_OK;
}

/*
 * lwswitch control device open callback implementation.
 */
int lwswitch_ctl_open
(
    vmk_MgmtCookies    *lwvCookie,
    vmk_MgmtElwelope   *lwvElwelope)
{
    return VMK_OK;
}

/*
 * lwswitch control device close callback implementation.
 */
int lwswitch_ctl_close
(   vmk_MgmtCookies    *lwvCookie,
    vmk_MgmtElwelope   *lwvElwelope
)
{
    return VMK_OK;
}

int lwswitch_ctl_ioctl
(
    vmk_MgmtCookies    *lwvCookie,
    vmk_MgmtElwelope   *lwvElwelope,
    vmk_uint32         *cmd,
    vmk_MgmtVectorParm *cmdParam
)
{
    // TODO: Implement once lwlink driver is complete
    return 0;
}

static VMK_ReturnStatus  lwswitch_ctl_session_announce
(
        vmk_MgmtHandle          handle,
        vmk_uint64              handleCookie,
        vmk_MgmtSessionID       sessionId,
        vmk_uint64              *sessionCookie
)
{
    return VMK_OK;
}

/*
 * lwswitch control device session close entry point. Here, sessionCookie
 * which hold client private data will be freed.
 */
static void lwswitch_ctl_session_cleanup
(
    vmk_MgmtHandle     handle,
    vmk_uint64         handleCookie,
    vmk_MgmtSessionID  sessionId,
    vmk_AddrCookie     sessionCookie
)
{
    return;
}

/*
 * Unregister lwswitch device.
 */
VMK_ReturnStatus lwswitch_unregister_device(
    vmk_Device vmkDeviceLogical
)
{
    VMK_ReturnStatus status;
    vmk_AddrCookie   addr;
    lw_vmware_state_t *lwv;

    lw_printf(LW_DBG_INFO, "lwpu-lwswitch: %s\n", __FUNCTION__);

    status = vmk_DeviceGetRegisteringDriverData(vmkDeviceLogical, &addr);
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: failed to get registering driver data: %s\n",
                  vmk_StatusToString(status));
        return VMK_FAILURE;
    }

    lwv = addr.ptr;

    if (lwv->logical_device == NULL)
    {
        return VMK_OK;
    }

    status = vmk_DeviceUnregister(lwv->logical_device);
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: failed to unregister logical device: %s\n",
                  vmk_StatusToString(status));
        return VMK_FAILURE;
    }
    lwv->logical_device = NULL;

    // Destroy per device lock
    LW_LOCK_DESTROY(&lwv->lwswitch_state.device_mutex);

    return VMK_OK;
}

/*
 * Management interface cleanup callback.
 */
void lwswitchMgmtCleanup(vmk_uint64 handleCookie)
{
    lw_vmware_state_t *lwv = (lw_vmware_state_t *) handleCookie;
    vmk_HeapID     heap_id = vmk_ModuleGetHeapID(lwidia_module_id);

    VMK_ASSERT(lwv->usage_count == 0);
    lw_printf(LW_DBG_ERRORS, "lwpu-lwswitch: %s %s\n", __FUNCTION__,
              vmk_NameToString(&lwv->logical_name));
    vmk_HeapFree(heap_id, lwv);
}

/*
 * Gets the alias for lwswitch character device.
 */
VMK_ReturnStatus lwswitch_associate(
    vmk_AddrCookie addr,
    vmk_CharDevHandle charDevHandle
)
{
    lw_chardev_data_t *chardev_data = (lw_chardev_data_t *)addr.ptr;
    lw_vmware_state_t *lwv = chardev_data->lwv;
    VMK_ReturnStatus status;

    status = vmk_CharDeviceGetAlias(charDevHandle, &lwv->logical_name);
    if (status != VMK_OK)
    {
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,
                          "lwpu-lwswitch: failed to get device alias: %s\n",
                          vmk_StatusToString(status));
    }
    lw_printf(LW_DBG_ERRORS, "lwpu-lwswitch: %s %s\n", __FUNCTION__,
              vmk_NameToString(&lwv->logical_name));

    return VMK_OK;
}


/*
 * Disassociate callback of device driver operations.
 */
VMK_ReturnStatus lwswitch_disassociate(
   vmk_AddrCookie addr
)
{
    lw_chardev_data_t *chardev_data = (lw_chardev_data_t *)addr.ptr;
    lw_vmware_state_t *lwv = chardev_data->lwv;

    lw_printf(LW_DBG_ERRORS, "lwpu-lwswitch: %s %s\n", __FUNCTION__,
              vmk_NameToString(&lwv->logical_name));
    return VMK_OK;
}

static vmk_DeviceOps lwswitch_device_ops = {
   .removeDevice = lwswitch_unregister_device,
};

static vmk_CharDevOps lwswitch_file_ops = {
   .ioctl = lwswitch_device_ioctl,
   .open = lwswitch_open,
   .close = lwswitch_close,
   .poll = lwswitch_poll,
};

static vmk_CharDevRegOps lwswitch_chardev_ops= {
    .associate    = lwswitch_associate,
    .disassociate = lwswitch_disassociate,
    .fileOps      = &lwswitch_file_ops,
};

void lwswitch_interrupt_handler(
    void *handlerData,
    vmk_IntrCookie intrCookie
)
{
    lw_vmware_state_t *lwv = handlerData;

    LW_LOCK(&lwv->lwswitch_state.device_mutex);
    lwswitch_lib_service_interrupts(lwv->lwswitch_state.lib_device);
    LW_UNLOCK(&lwv->lwswitch_state.device_mutex);

    return;
}

/*
 * Callback to acknowledge the device interrupt only. Interrupt
 * processing should be done in lwswitch_interrupt_handler rather than here.
 */
VMK_ReturnStatus lwswitch_interrupt_acknowledge(
    void *handlerData,
    vmk_IntrCookie intrCookie
)
{
    return VMK_OK;
}

static VMK_ReturnStatus lwswitch_initialize_device_interrupt
(
    lw_vmware_state_t *lwv
)
{
    VMK_ReturnStatus status;
    LwBool intr_allocated = LW_FALSE;
    vmk_uint32 register_interrupts = 0;
    vmk_uint32 num_interrupts;
    vmk_uint32 i;

    status = vmk_PCIAllocIntrCookie(lwidia_module_id,
                                lwv->pci_device,
                                VMK_PCI_INTERRUPT_TYPE_MSI,
                                1, 1, NULL,
                                lwv->interrupt,
                                &num_interrupts);
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: failed to allocate interrupt: %s\n",
                  vmk_StatusToString(status));
        goto failed;
    }

    intr_allocated = LW_TRUE;

    {
        vmk_IntrProps intrProps = {
            .deviceName  = {
            .string               = DRIVER_NAME,
            },
            .device               = lwv->physical_device,
            .acknowledgeInterrupt = lwswitch_interrupt_acknowledge,
            .handler              = lwswitch_interrupt_handler,
            .handlerData          = (void *) lwv,
            .attrs                = VMK_INTR_ATTRS_ENTROPY_SOURCE,
        };

        for (i = 0; i < lwv->num_intrs; i++)
        {
            status = vmk_IntrRegister(lwidia_module_id,
                                      lwv->interrupt[i],
                                      &intrProps);
            if (status == VMK_OK)
            {
                register_interrupts++;
            }
            else
            {
                lw_printf(LW_DBG_ERRORS,
                          "lwpu-lwswitch: failed to register interrupt: %s\n",
                          vmk_StatusToString(status));

                goto failed;
            }
        }
    }

    for (i = 0; i < lwv->num_intrs; i++)
    {
        status = vmk_IntrEnable(lwv->interrupt[i]);
        if (status != VMK_OK)
        {
            lw_printf(LW_DBG_ERRORS,
                      "lwpu-lwswitch: failed to enable interrupt: %s\n",
                      vmk_StatusToString(status));
            goto failed;
        }
    }

    return VMK_OK;

failed:
    for (i = 0; i < lwv->num_intrs; i++)
    {
        vmk_IntrUnregister(lwidia_module_id, lwv->interrupt[i], lwv);
    }
    if (intr_allocated)
    {
        vmk_PCIFreeIntrCookie(lwidia_module_id, lwv->pci_device);
    }

    return VMK_FAILURE;
}

static VMK_ReturnStatus lwswitch_load_bar_info
(
    lw_vmware_state_t *lwv
)
{
    vmk_PCIDevice     pci_device = lwv->pci_device;
    lwswitch_device  *lib_device = lwv->lwswitch_state.lib_device;
    vmk_PCIResource pciResources[1]; // Bar0
    vmk_uint32 bar = 0;
    VMK_ReturnStatus status;
    lwlink_pci_info *info;
    
    status = vmk_PCIQueryIOResources(pci_device, 1,
                                     pciResources);
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: failed to query PCI IO resources: %s\n",
                  vmk_StatusToString(status));
        return status;
    }

    status = vmk_PCIReadConfig(lwidia_module_id, pci_device,
                               VMK_PCI_CONFIG_ACCESS_32,
                               LWRM_PCICFG_BAR_OFFSET(0), &bar);
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: Failed to read lwswitch config space BAR: %s\n",
                  vmk_StatusToString(status));
        return status;
    }

    lwswitch_lib_get_device_info(lib_device, &info);
    info->bars[0].offset = LWRM_PCICFG_BAR_OFFSET(0);
    info->bars[0].busAddress = bar;
    info->bars[0].baseAddr = LW_PCI_RESOURCE_START(&pciResources[0]);
    info->bars[0].barSize = LW_PCI_RESOURCE_SIZE(&pciResources[0]);
    info->bars[0].pBar = lwv->lwswitch_state.bar0;

    return VMK_OK;
}

void lwswitch_post_init_blacklisted
(
    lw_vmware_state_t *lwv
)
{
    LW_LOCK(&lwswitch.driver_mutex);
    lwswitch_lib_post_init_blacklist_device(lwv->lwswitch_state.lib_device);
    LW_UNLOCK(&lwswitch.driver_mutex);
}

LwBool lwswitch_is_device_blacklisted
(
    lw_vmware_state_t *lwv
)
{
    LWSWITCH_DEVICE_FABRIC_STATE device_fabric_state = 0;
    LwlStatus status;

    status = lwswitch_lib_read_fabric_state(lwv->lwswitch_state.lib_device, 
                                            &device_fabric_state, NULL, NULL);

    if (status != LWL_SUCCESS)
    {
        lw_printf(LW_DBG_ERRORS,
                  "%s: Failed to read fabric state, %x\n",
                  lwv->lwswitch_state.name, status);
        return LW_FALSE;
    }

    return device_fabric_state == LWSWITCH_DEVICE_FABRIC_STATE_BLACKLISTED;
}

VMK_ReturnStatus lwswitch_post_init_device
(
    lw_vmware_state_t *lwv
)
{
    LwlStatus retval;
    LW_LOCK(&lwv->lwswitch_state.device_mutex);
    retval = lwswitch_lib_post_init_device(lwv->lwswitch_state.lib_device);
    if (retval != LWL_SUCCESS)
    {
        LW_UNLOCK(&lwv->lwswitch_state.device_mutex);
        return VMK_FAILURE;
    }
    LW_UNLOCK(&lwv->lwswitch_state.device_mutex);
   return VMK_OK; 
}

VMK_ReturnStatus lwswitch_init_device
(
    lw_vmware_state_t *lwv
)
{
    vmk_PCIDeviceAddr *pciDevAddr = &lwv->lwswitch_state.pciDevAddr;
    vmk_PCIDeviceID   *pciDevId = &lwv->lwswitch_state.pciDevId;
    VMK_ReturnStatus status;
    LwlStatus retval;

    LW_LOCK(&lwswitch.driver_mutex);
    retval = lwswitch_lib_register_device(LW_PCI_DOMAIN_NUMBER(pciDevAddr),
                                          LW_PCI_BUS_NUMBER(pciDevAddr),
                                          LW_PCI_SLOT_NUMBER(pciDevAddr),
                                          LW_PCI_FUNC(pciDevAddr),
                                          pciDevId->deviceID,
                                          lwv, //opaque os handle
                                          0, //TODO: Fill in with minor
                                          &lwv->lwswitch_state.lib_device);

    if (LWL_SUCCESS != retval)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: Failed to register lwswitch device: %s\n",
                  lwv->lwswitch_state.name);
        LW_UNLOCK(&lwswitch.driver_mutex);
        return VMK_FAILURE;
    }

    status = lwswitch_load_bar_info(lwv);
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: Failed to load bar info\n");
        goto init_device_failed;
    }

    retval = lwswitch_lib_initialize_device(lwv->lwswitch_state.lib_device);
    if (LWL_SUCCESS != retval)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: Failed to initialize lwswitch device: %s\n",
                  lwv->lwswitch_state.name);
        goto init_device_failed;
    }

    lwswitch_lib_get_uuid(lwv->lwswitch_state.lib_device, &lwv->lwswitch_state.uuid);

    if (lwswitch_lib_get_bios_version(lwv->lwswitch_state.lib_device,
                                      &lwv->lwswitch_state.bios_ver) != LWL_SUCCESS)
    {
        lwv->lwswitch_state.bios_ver = 0;
    }

    if (lwswitch_lib_get_physid(lwv->lwswitch_state.lib_device,
                                &lwv->lwswitch_state.phys_id) != LWL_SUCCESS)
    {
        lwv->lwswitch_state.phys_id = LWSWITCH_ILWALID_PHYS_ID;
    }

    status = lwswitch_initialize_device_interrupt(lwv);
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "%s: Failed to initialize interrupt : %d\n",
                  lwv->lwswitch_state.name,
                  status);
        goto init_intr_failed;
    }

    if (lwswitch_is_device_blacklisted(lwv))
    {
        lw_printf(LW_DBG_ERRORS,
                  "%s: Blacklisted lwswitch device\n",
                  lwv->lwswitch_state.name);
        // Keep device registered for HAL access and Fabric State updates
        LW_UNLOCK(&lwswitch.driver_mutex);
        return VMK_OK;
    }

    lwswitch_lib_enable_interrupts(lwv->lwswitch_state.lib_device);

    LW_UNLOCK(&lwswitch.driver_mutex);

    return VMK_OK;

init_intr_failed:
    lwswitch_lib_shutdown_device(lwv->lwswitch_state.lib_device);

init_device_failed:
    lwswitch_lib_unregister_device(lwv->lwswitch_state.lib_device);
    lwv->lwswitch_state.lib_device = NULL;
    LW_UNLOCK(&lwswitch.driver_mutex);

    return VMK_FAILURE;
}

VMK_ReturnStatus lwswitch_register_device(
    lw_vmware_state_t *lwv,
    vmk_Device physical
)
{
    VMK_ReturnStatus status;
    LwBool created_bus_address = LW_FALSE;
    vmk_DeviceID logical_dev_id;
    int device_num = lwv->device_num;

    vmk_DeviceProps  deviceProps = {
        .registeringDriver        = lw_driver,
        .deviceID                 = &logical_dev_id,
        .deviceOps                = &lwswitch_device_ops,
        .registeringDriverData    = {
            .ptr                  = lwv
        },
        .registrationData         = {
            .ptr                  = &lwv->chardev_reg_data
        }
    };

    lwv->chardev_data.device_num            = device_num;
    lwv->chardev_data.lwv                   = lwv;
    lwv->chardev_reg_data.moduleID          = lwidia_module_id;
    lwv->chardev_reg_data.deviceOps         = &lwswitch_chardev_ops;
    lwv->chardev_reg_data.devicePrivate.ptr = &lwv->chardev_data;

    logical_dev_id.busType = lw_logical_bus;

    status = vmk_LogicalCreateBusAddress(lw_driver,
                                         physical,
                                         device_num,
                                         &logical_dev_id.busAddress,
                                         &logical_dev_id.busAddressLen);
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: failed to create logical bus address: %s\n",
                  vmk_StatusToString(status));
        goto failed;
    }
    created_bus_address = LW_TRUE;

    logical_dev_id.busIdentifier    = VMK_CHARDEV_IDENTIFIER_GRAPHICS;
    logical_dev_id.busIdentifierLen = sizeof(VMK_CHARDEV_IDENTIFIER_GRAPHICS) - 1;

    status = vmk_DeviceRegister(&deviceProps,
                                physical,
                                &(lwv->logical_device));
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: failed to register device: %s\n",
                  vmk_StatusToString(status));
        goto failed;
    }

    vmk_LogicalFreeBusAddress(lw_driver, logical_dev_id.busAddress);

    // Initialize device lock
    status = LW_LOCK_CREATE(&lwv->lwswitch_state.device_mutex, "lwswitch_device_mutex", 1);
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: failed to create lwswitch device mutex!: %s\n",
                  vmk_StatusToString(status));
        goto failed;
    }

    return VMK_OK;


failed:

    if (created_bus_address)
    {
        vmk_LogicalFreeBusAddress(lw_driver, logical_dev_id.busAddress);
    }

    return VMK_FAILURE;
}

static VMK_ReturnStatus lwswitch_register_ctl_device(void)
{
    
    VMK_ReturnStatus status;
    vmk_HeapID       heap_id = vmk_ModuleGetHeapID(lwidia_module_id);
    lw_vmware_state_t *lwv = NULL;
    lw_state_t *lw;

    /* Ensure control device number cannot overlap with regular devices */
    VMK_ASSERT_ON_COMPILE(LW_CONTROL_DEVICE_NUM > LW_MAX_DEVICES);

    lwv = vmk_HeapAlloc(heap_id, sizeof(*lwv));
    if (lwv == NULL)
    {
        return VMK_FAILURE;
    }

    lw = LW_STATE_PTR(lwv);

    os_mem_set(lwv, 0, sizeof(lw_vmware_state_t));
    lw->os_state                 = (void *)lwv;
    lwv->chardev_data.device_num = LW_CONTROL_DEVICE_NUM;
    lwv->chardev_data.lwv        = lwv;

    lwswitch_ctl_device = lwv;

    vmk_MgmtProps mgmtProps = {
        .modId             = lwidia_module_id,
        .heapId            = heap_id,
        .sig               = &lwswitchMgmtSignature,
        .cleanupFn         = lwswitchMgmtCleanup,
        .sessionAnnounceFn = lwswitch_ctl_session_announce,
        .sessionCleanupFn  = lwswitch_ctl_session_cleanup,
        .handleCookie      = (vmk_uint64) lwv,
    };

    status = vmk_MgmtInit(&mgmtProps, &lwswitch_mgmt_handle);

    if (status != VMK_OK)
    {
        vmk_HeapFree(heap_id, lwv);
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: failed to register vmkMgmt control device: %s\n",
                  vmk_StatusToString(status));
    }

    return status;
}

static void lwswitch_shutdown_device_interrupt
(
    lw_vmware_state_t *lwv
)
{
    VMK_ReturnStatus status = VMK_OK;
    vmk_uint32 i;

    for (i = 0; i < lwv->num_intrs; i++)
    {
        status = vmk_IntrDisable(lwv->interrupt[i]);
        if (status != VMK_OK)
        {
            lw_printf(LW_DBG_ERRORS,
                      "lwpu-lwswitch: failed to disable interrupt: %s\n",
                      vmk_StatusToString(status));
        }

        status = vmk_IntrUnregister(lwidia_module_id, lwv->interrupt[i], lwv);
        if (status != VMK_OK)
        {
            lw_printf(LW_DBG_ERRORS,
                      "lwpu-lwswitch: failed to unregister interrupt: %s\n",
                      vmk_StatusToString(status));
        }
    }

    status = vmk_PCIFreeIntrCookie(lwidia_module_id, lwv->pci_device);
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: failed to free interrupt: %s\n",
                  vmk_StatusToString(status));
    }
}

static void lwswitch_deinit_device
(
    lw_vmware_state_t *lwv
)
{
    LW_LOCK(&lwswitch.driver_mutex);

    lwswitch_lib_disable_interrupts(lwv->lwswitch_state.lib_device);

    lwswitch_shutdown_device_interrupt(lwv);
    LwlStatus status;
    status = lwswitch_lib_shutdown_device(lwv->lwswitch_state.lib_device);
    if (status != LWL_SUCCESS)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: Unable to shutdown lwswitch device\n");
    }

    lwswitch_lib_unregister_device(lwv->lwswitch_state.lib_device);
    lwv->lwswitch_state.lib_device = NULL;

    LW_UNLOCK(&lwswitch.driver_mutex);
}

void lwswitch_device_remove
(
    lw_vmware_state_t *lwv
)
{
    lwv->lwswitch_state.unusable = LW_TRUE;
    lwswitch_deinit_device(lwv);
}


void lwswitch_pci_device_detach
(
    lw_vmware_state_t *lwv
)
{
    lw_state_t *lw;

    VMK_ASSERT(lwv != NULL);
    lw = LW_STATE_PTR(lwv);

    if ((lw->bars)[0].size != 0)
    {
        lw_user_map_unregister((lw->bars)[0].cpu_address,
                                (lw->bars)[0].size);
    }

    // Free Bar0
    lw_kernel_map_free(lwv->lwswitch_state.bar0);
    lwv->lwswitch_state.bar0 = NULL;
}

VMK_ReturnStatus lwswitch_pci_device_attach
(
    lw_vmware_state_t *lwv
)
{
    lw_state_t *lw;
    VMK_ReturnStatus status;
    vmk_PCIResource pciResources[1]; // Bar0
    vmk_uint32 irq = 0;
    vmk_uint32 bar = 0;
    vmk_uint32 offset = 0;
    vmk_uint64 cpu_address = 0, size = 0;

    VMK_ASSERT(lwv != NULL);

    status = vmk_PCIQueryDeviceID(lwv->pci_device, &lwv->lwswitch_state.pciDevId);
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: failed to query PCI device id: %s\n",
                  vmk_StatusToString(status));
        return status;
    }

    status = vmk_PCIQueryDeviceAddr(lwv->pci_device, &lwv->lwswitch_state.pciDevAddr);
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: failed to query PCI device address: %s\n",
                  vmk_StatusToString(status));
        return status;
    }

    status = vmk_PCIQueryIOResources(lwv->pci_device, 1,
                                     pciResources);
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: failed to query PCI IO resources: %s\n",
                  vmk_StatusToString(status));
        return status;
    }

    status = vmk_PCIReadConfig(lwidia_module_id, lwv->pci_device,
                            VMK_PCI_CONFIG_ACCESS_8, PCI_INTERRUPT_LINE, &irq);
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: failed to read config space interrupt: %s\n",
                  vmk_StatusToString(status));
        return status;
    }

    if (irq == 0)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: Can't find an IRQ for LWPU device\n");
        return VMK_FAILURE;
    }

    if ((LW_PCI_RESOURCE_VALID(&pciResources[0])) &&
        (LW_PCI_RESOURCE_FLAGS(&pciResources[0]) & PCI_BASE_ADDRESS_SPACE)
            == PCI_BASE_ADDRESS_SPACE_MEMORY)
    {
        cpu_address = LW_PCI_RESOURCE_START(&pciResources[0]);
        size        = LW_PCI_RESOURCE_SIZE(&pciResources[0]);
        offset      = LWRM_PCICFG_BAR_OFFSET(0);

        status = vmk_PCIReadConfig(lwidia_module_id, lwv->pci_device,
                                    VMK_PCI_CONFIG_ACCESS_32,
                                    offset, &bar);
        if (status != VMK_OK)
        {
            lw_printf(LW_DBG_ERRORS,
                      "lwpu-lwswitch: failed to read config space BAR: %s\n",
                      vmk_StatusToString(status));
            return status;
        }
    }

    // Set up Bar0
    lwv->lwswitch_state.bar0 = lw_kernel_map_create(cpu_address, size, LW_MEMORY_UNCACHED);
    if (lwv->lwswitch_state.bar0 == NULL)
    {
        lw_printf(LW_DBG_ERRORS, "lwpu-lwswitch: BAR0 mapping failed.\n");
        return VMK_FAILURE;
    }

    // Initialize lwpu device state
    lw                        = LW_STATE_PTR(lwv);
    (lw->bars)[0].cpu_address = cpu_address;
    (lw->bars)[0].size        = size;
    lw->os_state              = (void *)lwv;
    lwv->device_num           = lw_num_devices;

    if (size != 0)
    {
        if (lw_user_map_register(cpu_address, size) != 0)
        {
            lw_printf(LW_DBG_ERRORS,
                      "lwpu-lwswitch: failed to register usermap for BAR0\n");
            lw_user_map_unregister(cpu_address, size);
            lw_kernel_map_free(lwv->lwswitch_state.bar0);
            lwv->lwswitch_state.bar0 = NULL;
            return VMK_FAILURE;
        }
    }

    lw_num_devices++;
    return status;
}

VMK_ReturnStatus lwswitch_ctl_init
(
    void
)
{
    VMK_ReturnStatus status;

    if (lwswitch.initialized)
    {
        lw_printf(LW_DBG_ERRORS, "lwpu-lwswitch: Interface already initialized\n");

        return VMK_FAILURE;
    }
    // Create lwswitch driver lock
    status = LW_LOCK_CREATE(&lwswitch.driver_mutex, "lwswitch_driver_mutex", 1);
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: Failed to create lwswitch driver mutex: %s\n",
                  vmk_StatusToString(status));
        return VMK_FAILURE;
    }

    // Register lwswitch driver control device
    status = lwswitch_register_ctl_device();
    if (status != VMK_OK)
    {
        lw_printf(LW_DBG_ERRORS,
                  "lwpu-lwswitch: Failed to register lwswitch control device.\n");
        LW_LOCK_DESTROY(&lwswitch.driver_mutex);
        return VMK_FAILURE;
    }

    return VMK_OK;

}

void lwswitch_ctl_deinit
(
    void
)
{
    LW_LOCK_DESTROY(&lwswitch.driver_mutex);
    vmk_MgmtDestroy(lwswitch_mgmt_handle);
}

#endif
