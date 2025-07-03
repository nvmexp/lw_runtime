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


#ifndef _LWSWITCH_EXPORT_H_
#define _LWSWITCH_EXPORT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "lw_stdarg.h"
#include "lwlink_common.h"
#include "ioctl_common_lwswitch.h"

#define LWSWITCH_DRIVER_NAME            "lwpu-lwswitch"

#define LWSWITCH_MAX_BARS               1

#define LWSWITCH_DEVICE_INSTANCE_MAX    64

#define PCI_CLASS_BRIDGE_LWSWITCH       0x0680

#ifndef PCI_VENDOR_ID_LWIDIA
#define PCI_VENDOR_ID_LWIDIA            0x10DE
#endif

#define PCI_ADDR_OFFSET_VENDOR          0
#define PCI_ADDR_OFFSET_DEVID           2

#define LWSWITCH_NSEC_PER_SEC           1000000000ULL

#define LWSWITCH_DBG_LEVEL_MMIO         0x0
#define LWSWITCH_DBG_LEVEL_INFO         0x1
#define LWSWITCH_DBG_LEVEL_SETUP        0x2
#define LWSWITCH_DBG_LEVEL_WARN         0x3
#define LWSWITCH_DBG_LEVEL_ERROR        0x4

#define LWSWITCH_LOG_BUFFER_SIZE         512

#define LWSWITCH_DMA_DIR_TO_SYSMEM      0
#define LWSWITCH_DMA_DIR_FROM_SYSMEM    1
#define LWSWITCH_DMA_DIR_BIDIRECTIONAL  2

#define LWSWITCH_I2C_CMD_READ               0
#define LWSWITCH_I2C_CMD_WRITE              1
#define LWSWITCH_I2C_CMD_SMBUS_READ         2
#define LWSWITCH_I2C_CMD_SMBUS_WRITE        3
#define LWSWITCH_I2C_CMD_SMBUS_QUICK_READ   4
#define LWSWITCH_I2C_CMD_SMBUS_QUICK_WRITE  5

typedef struct lwswitch_device lwswitch_device;
typedef struct LWSWITCH_CLIENT_EVENT LWSWITCH_CLIENT_EVENT;

/*
 * @Brief : The interface will check if the client's version is supported by the
 *          driver.
 *
 * @param[in] user_version        Version of the interface that the client is
 *                                compiled with.
 * @param[out] kernel_version     Version of the interface that the kernel driver
 *                                is compiled with. This information will be
 *                                filled even if the CTRL call returns
 *                                -LWL_ERR_NOT_SUPPORTED due to version mismatch.
 * @param[in] length              Version string buffer length
 *
 * @returns                       LWL_SUCCESS if the client is using compatible
 *                                interface.
 *                                -LWL_ERR_NOT_SUPPORTED if the client is using
 *                                incompatible interface.
 *                                Or, Other LWL_XXX status value.
 */
LwlStatus
lwswitch_lib_check_api_version
(
    const char *user_version,
    char *kernel_version,
    LwU32 length
);

/*
 * @Brief : Allocate a new lwswitch lib device instance.
 *
 * @Description : Creates and registers a new lwswitch device and registers
 *   with the lwlink library.  This only initializes software state,
 *   it does not initialize the hardware state.
 *
 * @param[in] pci_domain    pci domain of the device
 * @param[in] pci_bus       pci bus of the device
 * @param[in] pci_device    pci device of the device
 * @param[in] pci_func      pci function of the device
 * @param[in] device_id     pci device ID of the device
 * @param[in] os_handle     Device handle used to interact with OS layer
 * @param[in] os_instance   instance number of this device
 * @param[out] device       return device handle for interfacing with library
 *
 * @returns                 LWL_SUCCESS if the action succeeded
 *                          an LWL error code otherwise
 */
LwlStatus
lwswitch_lib_register_device
(
    LwU16 pci_domain,
    LwU8 pci_bus,
    LwU8 pci_device,
    LwU8 pci_func,
    LwU16 device_id,
    void *os_handle,
    LwU32 os_instance,
    lwswitch_device **device
);

/*
 * @Brief : Clean-up the software state for a lwswitch device.
 *
 * @Description :
 *
 * @param[in] device        device handle to destroy
 *
 * @returns                 none
 */
void
lwswitch_lib_unregister_device
(
    lwswitch_device *device
);

/*
 * @Brief : Initialize the hardware for a lwswitch device.
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 *
 * @returns                 LWL_SUCCESS if the action succeeded
 *                          -LWL_BAD_ARGS if bad arguments provided
 *                          -LWL_PCI_ERROR if bar info unable to be retrieved
 */
LwlStatus
lwswitch_lib_initialize_device
(
    lwswitch_device *device
);

/*
 * @Brief : Shutdown the hardware for a lwswitch device.
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 *
 * @returns                 LWL_SUCCESS if the action succeeded
 *                          -LWL_BAD_ARGS if bad arguments provided
 *                          -LWL_PCI_ERROR if bar info unable to be retrieved
 */
LwlStatus
lwswitch_lib_shutdown_device
(
    lwswitch_device *device
);

/*
 * @Brief Control call (ioctl) interface.
 *
 * @param[in] device        device to operate on
 * @param[in] cmd           Enumerated command to execute.
 * @param[in] params        Params structure to pass to the command.
 * @param[in] params_size   Size of the parameter structure.
 * @param[in] osPrivate     The private data structure for OS.
 *
 * @return                  LWL_SUCCESS on a successful command
 *                          -LWL_NOT_FOUND if target device unable to be found
 *                          -LWL_BAD_ARGS if an invalid cmd is provided
 *                          -LWL_BAD_ARGS if a null arg is provided
 *                          -LWL_ERR_GENERIC otherwise
 */
LwlStatus lwswitch_lib_ctrl
(
    lwswitch_device *device,
    LwU32 cmd,
    void *params,
    LwU64 size,
    void *osPrivate
);

/*
 * @Brief: Retrieve PCI information for a switch based from device instance
 *
 * @Description :
 *
 * @param[in]  lib_handle   device to query
 * @param[out] pciInfo      return pointer to lwswitch lib copy of device info
 */
void lwswitch_lib_get_device_info
(
    lwswitch_device *lib_handle,
    struct lwlink_pci_info **pciInfo
);

/*
 * @Brief: Retrieve BIOS version for an lwswitch device
 *
 * @Description: For devices with a BIOS, this retrieves the BIOS version.
 *
 * @param[in]  device  device to query
 * @param[out] version BIOS version is stored here
 *
 * @returns LWL_SUCCESS                 BIOS version was retrieved successfully
 *          -LWL_BAD_ARGS               an invalid device is provided
 *          -LWL_ERR_ILWALID_STATE      an error oclwrred reading BIOS info
 *          -LWL_ERR_NOT_SUPPORTED      device doesn't support this feature
 */

LwlStatus
lwswitch_lib_get_bios_version
(
    lwswitch_device *device,
    LwU64 *version
);


/*
 * @Brief: Retrieve whether the device supports PCI pin interrupts
 *
 * @Description: Returns whether the device can use PCI pin IRQs
 *
 *
 * @returns LW_TRUE                 device can use PCI pin IRQs
 *          LW_FALSE                device cannot use PCI pin IRQs
 */

LwlStatus
lwswitch_lib_use_pin_irq
(
    lwswitch_device *device
);


/*
 * @Brief: Load platform information (emulation, simulation etc.).
 *
 * @param[in]  lib_handle   device
 *
 * @return                  LWL_SUCCESS on a successful command
 *                          -LWL_BAD_ARGS if an invalid device is provided
 */
LwlStatus lwswitch_lib_load_platform_info
(
    lwswitch_device *lib_handle
);

/*
 * @Brief : Enable interrupts for this device
 *
 * @Description :
 *
 * @param[in] device        device to enable
 *
 * @returns                 LWL_SUCCESS
 *                          -LWL_PCI_ERROR if there was a register access error
 */
void
lwswitch_lib_enable_interrupts
(
    lwswitch_device *device
);

/*
 * @Brief : Disable interrupts for this device
 *
 * @Description :
 *
 * @param[in] device        device to enable
 *
 * @returns                 LWL_SUCCESS
 *                          -LWL_PCI_ERROR if there was a register access error
 */
void
lwswitch_lib_disable_interrupts
(
    lwswitch_device *device
);

/*
 * @Brief : Check if interrupts are pending on this device
 *
 * @Description :
 *
 * @param[in] device        device to check
 *
 * @returns                 LWL_SUCCESS if there were no errors and interrupts were handled
 *                          -LWL_BAD_ARGS if bad arguments provided
 *                          -LWL_PCI_ERROR if there was a register access error
 *                          -LWL_MORE_PROCESSING_REQUIRED no interrupts were found for this device
 */
LwlStatus
lwswitch_lib_check_interrupts
(
    lwswitch_device *device
);

/*
 * @Brief : Services interrupts for this device
 *
 * @Description :
 *
 * @param[in] device        device to service
 *
 * @returns                 LWL_SUCCESS if there were no errors and interrupts were handled
 *                          -LWL_BAD_ARGS if bad arguments provided
 *                          -LWL_PCI_ERROR if there was a register access error
 *                          -LWL_MORE_PROCESSING_REQUIRED no interrupts were found for this device
 */
LwlStatus
lwswitch_lib_service_interrupts
(
    lwswitch_device *device
);

/*
 * @Brief : Get depth of error logs
 *
 * @Description :
 *
 * @param[in]  device       device to check
 *
 * @param[out] fatal        Count of fatal errors
 * @param[out] nonfatal     Count of non-fatal errors
 *
 * @returns                 LWL_SUCCESS if there were no errors and interrupts were handled
 *                          -LWL_NOT_FOUND if bad arguments provided
 */
LwlStatus
lwswitch_lib_get_log_count
(
    lwswitch_device *device,
    LwU32 *fatal, LwU32 *nonfatal
);

/*
 * @Brief : Periodic thread-based dispatcher for kernel functions
 *
 * @Description : Its purpose is to do any background subtasks (data collection, thermal
 * monitoring, etc.  These subtasks may need to run at varying intervals, and
 * may even wish to adjust their exelwtion period based on other factors.
 * Each subtask's entry notes the last time it was exelwted and its desired
 * exelwtion period.  This function returns back to the dispatcher the desired
 * time interval before it should be called again.
 *
 * @param[in] device          The device to run background tasks on
 *
 * @returns nsec interval to wait before the next call.
 */
LwU64
lwswitch_lib_deferred_task_dispatcher
(
    lwswitch_device *device
);

/*
 * @Brief : Perform post init tasks
 *
 * @Description : Any device initialization/tests which need the device to be
 * initialized to a sane state go here.
 *
 * @param[in] device    The device to run the post-init on
 *
 * @returns             returns LwlStatus code, see lwlink_errors.h
 */
LwlStatus
lwswitch_lib_post_init_device
(
    lwswitch_device *device
);

/*
 * @Brief : Perform post init tasks for a blacklisted device
 *
 * @Description : Any initialization tasks that should be run after a
 *                blacklisted item should go here.
 *
 * @param[in] device    The device to run the post-init-blacklist on
 *
 * @returns             void
 */
void
lwswitch_lib_post_init_blacklist_device
(
    lwswitch_device *device
);

/*
 * @Brief : Get the UUID of the device
 *
 * @Description : Copies out the device's UUID into the uuid field
 *
 * @param[in] device    The device to get the UUID from
 *
 * @param[out] uuid     A pointer to a uuid struct in which the UUID is written to
 *
 * @returns             void
 */
void
lwswitch_lib_get_uuid
(
    lwswitch_device *device,
    LwUuid *uuid
);

/*
 * @Brief : Get the Physical ID of the device
 *
 * @Description : Copies out the device's Physical ID into the phys_id field
 *
 * @param[in] device    The device to get the UUID from
 *
 * @param[out] phys_id  A pointer to a LwU32 which the physical ID is written to
 *
 * @returns             LWL_SUCCESS if successful
 *                      -LWL_BAD_ARGS if bad arguments provided
 */
LwlStatus
lwswitch_lib_get_physid
(
    lwswitch_device *device,
    LwU32 *phys_id
);

/*
 * @Brief : Read the Fabric State for a lwswitch device.
 *
 * @Description : Returns the Fabric State for the device
 *
 * @param[in] device        a reference to the device
 * @param[in] *ptrs         references to the fabric state
 *
 * @returns                 LWL_SUCCESS if the action succeeded
 *                          -LWL_BAD_ARGS if bad arguments provided
 */
LwlStatus
lwswitch_lib_read_fabric_state
(
    lwswitch_device *device,
    LWSWITCH_DEVICE_FABRIC_STATE *device_fabric_state,
    LWSWITCH_DEVICE_BLACKLIST_REASON *device_blacklist_reason,
    LWSWITCH_DRIVER_FABRIC_STATE *driver_fabric_state
);

/*
 * @Brief : Validates PCI device id
 *
 * @Description : Validates PCI device id
 *
 * @param[in] device    The device id to be validated
 *
 * @returns             True if device id is valid
 */
LwBool
lwswitch_lib_validate_device_id
(
    LwU32 device_id
);

/*
 * @Brief : Gets an event if it exists in the Event list
 *
 * @Description : Gets an event if it is in the Device's Client
 *                Event list
 *
 * @param[in]  device         Device to operate on
 * @param[in]  osPrivate      The private data structure for the OS
 * @param[out] ppClientEvent  Double pointer to client event
 *
 * @returns                  LWL_SUCCESS if client event found
 *                           -LWL_BAD_ARGS if bad arguments provided
 *                           -LWL_NOT_FOUND if no client event found
 */
LwlStatus
lwswitch_lib_get_client_event
(
    lwswitch_device *device,
    void *osPrivate,
    LWSWITCH_CLIENT_EVENT **ppClientEvent
);

/*
 * @Brief : Adds a single entry into the Event list
 *
 * @Description : Adds an entry into the front of the Device's
 *                Client Event List
 *
 * @param[in] device     Device to operate on
 * @param[in] osPrivate  The private data structure for OS
 * @param[in] pParams    The parameters for the client event
 *
 * @returns              LWL_SUCCESS if event added
 *                       -LWL_BAD_ARGS if bad arguments provided
 *                       -LWL_NO_MEM if allocation fails
 */
LwlStatus
lwswitch_lib_add_client_event
(
    lwswitch_device *device,
    void *osPrivate,
    LwU32 eventId
);

/*
 * @Brief : Removes entries from the Event list
 *
 * @Description : Removes the entries associated with osPrivate
 *                from the Device's Client Event List
 *
 * @param[in] device     Device to operate on
 * @param[in] osPrivate  The private data structure for OS
 *
 * @returns              LWL_SUCCESS if event removed
 */
LwlStatus
lwswitch_lib_remove_client_events
(
    lwswitch_device *device,
    void *osPrivate
);

/*
 * @Brief : Notifies all events with a matching event Id in the Client Event list
 *
 * @Description : Notifies all events with a matching event Id in the Client Event list
 *
 * @param[in] device     Device to operate on
 * @param[in] eventId    The event ID to notify
 *
 * @returns              LWL_SUCCESS if arguments are valid
 *                       -LWL_BAD_ARGS if bad arguments provided
 */
LwlStatus
lwswitch_lib_notify_client_events
(
    lwswitch_device *device,
    LwU32 eventId
);

/*
 * @Brief : Gets a mask of valid I2C ports for the device
 *
 * @Description : Gets a mask of valid I2C ports for the device
 *
 * @param[in]  device          Device to operate on
 * @param[out] validPortsMask  A pointer to a mask of valid ports
 *
 * @returns              LWL_SUCCESS if successfuly
 *                       -LWL_BAD_ARGS if bad arguments provided
 */
LwlStatus
lwswitch_lib_get_valid_ports_mask
(
    lwswitch_device *device,
    LwU32 *validPortsMask
);

/*
 * @Brief : Returns a boolean if the I2C interface is supported for the device
 *
 * @Description : Returns a boolean if the I2C interface is supported for the device
 *
 * @param[in]  device         Device to operate on
 *
 * @returns LW_TRUE           device can use the I2C interface
 *          LW_FALSE          device cannot use the I2C interface
 */
LwBool
lwswitch_lib_is_i2c_supported
(
    lwswitch_device *device
);

/*
 * @Brief : Performs an I2C transaction
 *
 * @Description : Performs an I2C transaction
 *
 * @param[in]  device         Device to operate on
 * @param[in]  port           Port to issue I2C transaction
 * @param[in]  type           Type of I2C transaction
 * @param[in]  addr           Device address to perform I2C transaction on
 * @param[in]  command        I2C command to perform on
 * @param[in]  len            Length of the I2C transaction message
 * @param[in/out] pData       A pointer to the buffer containing the input/output data
 *
 * @returns              LWL_SUCCESS if I2C transaction completes
 *                       -LWL_BAD_ARGS if bad arguments provided
 *                       -LWL_ERR_ILWALID_STATE if something internal went wrong
 */
LwlStatus
lwswitch_lib_i2c_transfer
(
    lwswitch_device *device,
    LwU32 port,
    LwU8 type,
    LwU8 addr,
    LwU8 command,
    LwU32 len,
    LwU8 *pData
);

/*
 * Returns count of registered LwSwitch devices.
 */
LwU32
lwswitch_os_get_device_count
(
    void
);

/*
 * Get current time in nanoseconds
 * The time is since epoch time (midnight UTC of January 1, 1970)
 */
LwU64
lwswitch_os_get_platform_time
(
    void
);

#if (defined(_WIN32) || defined(_WIN64))
#define LWSWITCH_PRINT_ATTRIB(str, arg1)
#else
#define LWSWITCH_PRINT_ATTRIB(str, arg1)             \
    __attribute__ ((format (printf, (str), (arg1))))
#endif // (defined(_WIN32) || defined(_WIN64))

/*
 * printf wrapper
 */
void
LWSWITCH_PRINT_ATTRIB(2, 3)
lwswitch_os_print
(
    int         log_level,
    const char *pFormat,
    ...
);

/*
 * "Registry" interface for dword
 */
LwlStatus
lwswitch_os_read_registry_dword
(
    void *os_handle,
    const char *name,
    LwU32 *data
);

/*
 * "Registry" interface for binary data
 */
LwlStatus
lwswitch_os_read_registery_binary
(
    void *os_handle,
    const char *name,
    LwU8 *data,
    LwU32 length
);

LwBool
lwswitch_os_is_uuid_in_blacklist
(
    LwUuid *uuid
);


/*
 * Override platform/simulation settings for cases
 */
void
lwswitch_os_override_platform
(
    void *os_handle,
    LwBool *rtlsim
);

/*
 * Memory management interface
 */
LwlStatus
lwswitch_os_alloc_contig_memory
(
    void *os_handle,
    void **virt_addr,
    LwU32 size,
    LwBool force_dma32
);

void
lwswitch_os_free_contig_memory
(
    void *os_handle,
    void *virt_addr,
    LwU32 size
);

LwlStatus
lwswitch_os_map_dma_region
(
    void *os_handle,
    void *cpu_addr,
    LwU64 *dma_handle,
    LwU32 size,
    LwU32 direction
);

LwlStatus
lwswitch_os_unmap_dma_region
(
    void *os_handle,
    void *cpu_addr,
    LwU64 dma_handle,
    LwU32 size,
    LwU32 direction
);

LwlStatus
lwswitch_os_set_dma_mask
(
    void *os_handle,
    LwU32 dma_addr_width
);

LwlStatus
lwswitch_os_sync_dma_region_for_cpu
(
    void *os_handle,
    LwU64 dma_handle,
    LwU32 size,
    LwU32 direction
);

LwlStatus
lwswitch_os_sync_dma_region_for_device
(
    void *os_handle,
    LwU64 dma_handle,
    LwU32 size,
    LwU32 direction
);

void *
lwswitch_os_malloc_trace
(
    LwLength size,
    const char *file,
    LwU32 line
);

void
lwswitch_os_free
(
    void *pMem
);

LwLength
lwswitch_os_strlen
(
    const char *str
);

char*
lwswitch_os_strncpy
(
    char *pDest,
    const char *pSrc,
    LwLength length
);

int
lwswitch_os_strncmp
(
    const char *s1,
    const char *s2,
    LwLength length
);

void *
lwswitch_os_memset
(
    void *pDest,
    int value,
    LwLength size
);

void *
lwswitch_os_memcpy
(
    void *pDest,
    const void *pSrc,
    LwLength size
);

int
lwswitch_os_memcmp
(
    const void *s1,
    const void *s2,
    LwLength size
);

/*
 * Memory read / write interface
 */
LwU32
lwswitch_os_mem_read32
(
    const volatile void * pAddress
);

void
lwswitch_os_mem_write32
(
    volatile void *pAddress,
    LwU32 data
);

LwU64
lwswitch_os_mem_read64
(
    const volatile void *pAddress
);

void
lwswitch_os_mem_write64
(
    volatile void *pAddress,
    LwU64 data
);

/*
 * Interface to write formatted output to sized buffer
 */
int
lwswitch_os_snprintf
(
    char *pString,
    LwLength size,
    const char *pFormat,
    ...
);

/*
 * Interface to write formatted output to sized buffer
 */
int
lwswitch_os_vsnprintf
(
    char *buf,
    LwLength size,
    const char *fmt,
    va_list arglist
);

/*
 * Debug assert and log interface
 */
void
lwswitch_os_assert_log
(
    int cond,
    const char *pFormat,
    ...
);

/*
 * Interface to sleep for specified milliseconds. Yields the CPU to scheduler.
 */
void
lwswitch_os_sleep
(
    unsigned int ms
);

LwlStatus
lwswitch_os_acquire_fabric_mgmt_cap
(
    void *osPrivate,
    LwU64 capDescriptor
);

int
lwswitch_os_is_fabric_manager
(
    void *osPrivate
);

int
lwswitch_os_is_admin
(
    void
);

LwlStatus
lwswitch_os_get_os_version
(
    LwU32 *pMajorVer,
    LwU32 *pMinorVer,
    LwU32 *pBuildNum
);

void
lwswitch_lib_smbpbi_log_sxid
(
    lwswitch_device *device,
    LwU32           sxid,
    const char      *pFormat,
    ...
);

/*!
 * @brief: OS Specific handling to add an event.
 */
LwlStatus
lwswitch_os_add_client_event
(
    void            *osHandle,
    void            *osPrivate,
    LwU32           eventId
);

/*!
 * @brief: OS specific handling to remove all events corresponding to osPrivate.
 */
LwlStatus
lwswitch_os_remove_client_event
(
    void            *osHandle,
    void            *osPrivate
);

/*!
 * @brief: OS specific handling to notify an event.
 */
LwlStatus
lwswitch_os_notify_client_event
(
    void *osHandle,
    void *osPrivate,
    LwU32 eventId
);

/*!
 * @brief: Gets OS specific support for the REGISTER_EVENTS ioctl
 */
LwlStatus
lwswitch_os_get_supported_register_events_params
(
    LwBool *bSupportsManyEvents,
    LwBool *bUserSuppliesOsData
);

#ifdef __cplusplus
}
#endif
#endif //_LWSWITCH_EXPORT_H_
