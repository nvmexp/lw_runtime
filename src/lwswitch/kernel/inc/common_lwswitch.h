/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _COMMON_LWSWITCH_H_
#define _COMMON_LWSWITCH_H_

#ifdef INCLUDE_LWLINK_LIB
#include "lwlink.h"
#endif
#include "g_lwconfig.h"
#include "export_lwswitch.h"
#include "error_lwswitch.h"
#include "io_lwswitch.h"
#include "rom_lwswitch.h"
#include "haldef_lwswitch.h"
#include "lwctassert.h"
#include "flcn/flcnable_lwswitch.h"
#include "inforom/inforom_lwswitch.h"
#include "spi_lwswitch.h"
#include "smbpbi_lwswitch.h"
#include "bus_lwswitch.h"
#include "lwCpuUuid.h"

#define LWSWITCH_GET_BIT(v, p)       (((v) >> (p)) & 1)
#define LWSWITCH_SET_BIT(v, p)       ((v) |  LWBIT(p))
#define LWSWITCH_CLEAR_BIT(v, p)     ((v) & ~LWBIT(p))
#define LWSWITCH_MASK_BITS(n)        (~(0xFFFFFFFF << (n)))

static LW_INLINE LwBool lwswitch_test_flags(LwU32 val, LwU32 flags)
{
    return !!(val & flags);
}

static LW_INLINE void lwswitch_set_flags(LwU32 *val, LwU32 flags)
{
    *val |= flags;
}

static LW_INLINE void lwswitch_clear_flags(LwU32 *val, LwU32 flags)
{
    *val &= ~flags;
}

// Destructive operation to reverse bits in a mask
#define LWSWITCH_REVERSE_BITMASK_32(numBits, mask)  \
{                                                   \
    LwU32 i, reverse = 0;                           \
    FOR_EACH_INDEX_IN_MASK(32, i, mask)             \
    {                                               \
        reverse |= LWBIT((numBits - 1) - i);          \
    }                                               \
    FOR_EACH_INDEX_IN_MASK_END;                     \
                                                    \
    mask = reverse;                                 \
}

#define LWSWITCH_CHECK_STATUS(_d, _status)                  \
    if (_status != LWL_SUCCESS)                             \
    {                                                       \
        LWSWITCH_PRINT(_d, MMIO, "%s(%d): status=%d\n",     \
            __FUNCTION__, __LINE__,                         \
            _status);                                       \
    }

#define IS_RTLSIM(device)   (device->is_rtlsim)
#define IS_FMODEL(device)   (device->is_fmodel)
#define IS_EMULATION(device)   (device->is_emulation)

#define LWSWITCH_DEVICE_NAME                            "lwswitch"
#define LWSWITCH_LINK_NAME                              "link"

// Max size of sprintf("%d", valid_instance) compile time check
#if LWSWITCH_DEVICE_INSTANCE_MAX < 100
#define LWSWITCH_INSTANCE_LEN 2
#endif

#define LW_ARRAY_ELEMENTS(x)   ((sizeof(x)/sizeof((x)[0])))

#define LWSWITCH_DBG_LEVEL LWSWITCH_DBG_LEVEL_INFO

#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
#define LWSWITCH_PRINT(_d, _lvl, _fmt, ...)                 \
    ((LWSWITCH_DBG_LEVEL <= LWSWITCH_DBG_LEVEL_ ## _lvl) ?  \
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ ## _lvl,      \
            "%s[%-5s]: " _fmt,                              \
            ((_d == NULL) ?                                 \
                "lwswitchx" :                               \
                ((lwswitch_device *)_d)->name),             \
            #_lvl,                                          \
            ## __VA_ARGS__) :                               \
        ((void)(0))                                         \
    )
#else
    #define LWSWITCH_PRINT(_d, _lvl, _fmt, ...) ((void)0)
#endif

#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
#define lwswitch_os_malloc(_size)                           \
    lwswitch_os_malloc_trace(_size, __FILE__, __LINE__)
#else
#define lwswitch_os_malloc(_size)                           \
    lwswitch_os_malloc_trace(_size, NULL, 0)
#endif

//
// This macro should be used to check assertion statements and print Error messages.
//
#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
#define LWSWITCH_ASSERT(_cond)                                                       \
    lwswitch_os_assert_log((_cond), "LWSwitch: Assertion failed in %s() at %s:%d\n", \
         __FUNCTION__ , __FILE__, __LINE__)
#else
#define LWSWITCH_ASSERT(_cond)                                       \
    lwswitch_os_assert_log((_cond), "LWSwitch: Assertion failed \n")
#endif

#if defined(LW_MODS)
#include "modsdrv.h"

#define LWSWITCH_ASSERT_ERROR_INFO(errorCategory, errorInfo)\
    do                                                                          \
    {                                                                           \
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,                             \
            "bp @ %s:%d (reason=%d)\n", __FILE__, __LINE__, (int)errorCategory);\
        ModsDrvBreakPointWithErrorInfo((int)errorCategory, __FILE__, __LINE__, errorInfo);\
    } while (0)

#define LWSWITCH_ASSERT_INFO(errCode, errLinkMask, errSubcode) {                \
    LwU32 savedId = ModsDrvSetSwitchId(device->os_instance);                    \
    ModsDrvErrorInfo errInfo;                                                   \
    errInfo.type = MODSDRV_LWLINK_ERROR_TYPE;                                   \
    errInfo.lwlinkInfo.lwlinkMask = (errLinkMask);                              \
    errInfo.lwlinkInfo.subCode = (MODSDRV_##errSubcode);                        \
    LWSWITCH_ASSERT_ERROR_INFO((errCode), (&errInfo));                          \
    ModsDrvSetSwitchId(savedId); }

#else

#define LWSWITCH_ASSERT_ERROR_INFO(errorCategory, errorInfo) LWSWITCH_ASSERT(0x0)
#define LWSWITCH_ASSERT_INFO(errCode, errLinkMask, errSubcode) LWSWITCH_ASSERT(0x0)

#endif

//
// This macro should be used cautiously as it prints information in the release
// drivers.
//
#define LWSWITCH_PRINT_SXID(_d, _sxid, _fmt, ...)                                         \
    do                                                                                    \
    {                                                                                     \
        LWSWITCH_ASSERT(lwswitch_translate_hw_error(_sxid) != LWSWITCH_LWLINK_HW_GENERIC); \
        lwswitch_os_print(LWSWITCH_DBG_LEVEL_ERROR,                                       \
            "lwpu-%s: SXid (PCI:" LWLINK_PCI_DEV_FMT "): %05d, " _fmt,                  \
            (_d)->name, LWLINK_PCI_DEV_FMT_ARGS(&(_d)->lwlink_device->pciInfo), _sxid,    \
            ##__VA_ARGS__);                                                               \
        lwswitch_lib_smbpbi_log_sxid(_d, _sxid, _fmt, ##__VA_ARGS__);                     \
        lwswitch_inforom_bbx_add_sxid(_d, _sxid, 0, 0, 0);                                \
    } while(0)

#define LWSWITCH_DEV_CMD_DISPATCH_WITH_PRIVATE_DATA(cmd, function, type, private)\
    case cmd:                                                               \
    {                                                                       \
        if (sizeof(type) == size)                                           \
        {                                                                   \
            retval = function(device, params, private);                     \
        }                                                                   \
        else                                                                \
        {                                                                   \
            retval = -LWL_BAD_ARGS;                                         \
        }                                                                   \
        break;                                                              \
    }

#define LWSWITCH_DEV_CMD_DISPATCH_HELPER(cmd, supported, function, type) \
    case cmd:                                                            \
    {                                                                    \
        if (!supported)                                                  \
        {                                                                \
            retval = -LWL_ERR_NOT_SUPPORTED;                             \
        }                                                                \
        else if (sizeof(type) == size)                                   \
        {                                                                \
            retval = function(device, params);                           \
        }                                                                \
        else                                                             \
        {                                                                \
            retval = -LWL_BAD_ARGS;                                      \
        }                                                                \
        break;                                                           \
    }

#define LWSWITCH_DEV_CMD_DISPATCH(cmd, function, type) \
    LWSWITCH_DEV_CMD_DISPATCH_HELPER(cmd, LW_TRUE, function, type)

#if defined (LW_MODS)
#define LWSWITCH_MODS_CMDS_SUPPORTED LW_TRUE
#else
#define LWSWITCH_MODS_CMDS_SUPPORTED LW_FALSE
#endif

#if defined(DEBUG) || defined(DEVELOP) || defined(LW_MODS)
#define LWSWITCH_TEST_CMDS_SUPPORTED LW_TRUE
#else
#define LWSWITCH_TEST_CMDS_SUPPORTED LW_FALSE
#endif

#define LWSWITCH_DEV_CMD_DISPATCH_MODS(cmd, function, type)   \
    LWSWITCH_DEV_CMD_DISPATCH_HELPER(cmd, LWSWITCH_MODS_CMDS_SUPPORTED, function, type)

#define LWSWITCH_DEV_CMD_DISPATCH_TEST(cmd, function, type)   \
    LWSWITCH_DEV_CMD_DISPATCH_HELPER(cmd, LWSWITCH_TEST_CMDS_SUPPORTED, function, type)

#define LWSWITCH_MAX_NUM_LINKS 100
#if LWSWITCH_MAX_NUM_LINKS <= 100
#define LWSWITCH_LINK_INSTANCE_LEN 2
#endif

extern const lwlink_link_handlers lwswitch_link_handlers;

//
// link_info is used to store private link information
//
typedef struct
{
    char        name[sizeof(LWSWITCH_LINK_NAME) + LWSWITCH_LINK_INSTANCE_LEN];
} LINK_INFO;

typedef struct
{
    LwU32 external_fabric_mgmt;
    LwU32 txtrain_control;
    LwU32 crossbar_DBI;
    LwU32 link_DBI;
    LwU32 ac_coupled_mask;
    LwU32 ac_coupled_mask2;
    LwU32 swap_clk;
    LwU32 link_enable_mask;
    LwU32 link_enable_mask2;
    LwU32 bandwidth_shaper;
    LwU32 ssg_control;
    LwU32 skip_buffer_ready;
    LwU32 enable_pm;
    LwU32 chiplib_forced_config_link_mask;
    LwU32 chiplib_forced_config_link_mask2;
    LwU32 soe_dma_self_test;
    LwU32 soe_disable;
    LwU32 soe_enable;
    LwU32 soe_boot_core;
    LwU32 minion_cache_seeds;
    LwU32 latency_counter;
    LwU32 lwlink_speed_control;
    LwU32 inforom_bbx_periodic_flush;
    LwU32 inforom_bbx_write_periodicity;
    LwU32 inforom_bbx_write_min_duration;
    LwU32 ato_control;
    LwU32 sto_control;
    LwU32 minion_disable;
    LwU32 set_ucode_target;
    LwU32 set_simmode;
    LwU32 set_smf_settings;
    LwU32 select_uphy_tables;
    LwU32 link_training_mode;
    LwU32 i2c_access_control;
    LwU32 link_recal_settings;
    LwU32 crc_bit_error_rate_short;
    LwU32 crc_bit_error_rate_long;
} LWSWITCH_REGKEY_TYPE;

//
// Background tasks
//
typedef struct LWSWITCH_TASK
{
    struct LWSWITCH_TASK *next;
    void (*task_fn)(lwswitch_device *);
    LwU64 period_nsec;
    LwU64 last_run_nsec;
    LwU32 flags;
} LWSWITCH_TASK_TYPE;

#define LWSWITCH_TASK_TYPE_FLAGS_ALWAYS_RUN  0x1    // Run even the if not initialized

//
// PLL
//
typedef struct
{
    LwU32   src_freq_khz;
    LwU32   M;
    LwU32   N;
    LwU32   PL;
    LwU32   dist_mode;
    LwU32   refclk_div;
    LwU32   vco_freq_khz;
    LwU32   freq_khz;
} LWSWITCH_PLL_INFO;

// Per-unit interrupt masks
typedef struct
{
    LwU32   fatal;
    LwU32   nonfatal;
    LwU32   correctable;
} LWSWITCH_INTERRUPT_MASK;

// BIOS Image
typedef struct
{
    // Size of the image.
    LwU32 size;

    // pointer to the BIOS image.
    LwU8* pImage;
} LWSWITCH_BIOS_IMAGE;

struct LWSWITCH_CLIENT_EVENT
{
    LWListRec entry;
    LwU32     eventId;
    void      *private_driver_data;
};

//
// common device information
//
struct lwswitch_device
{
#ifdef INCLUDE_LWLINK_LIB
    lwlink_device   *lwlink_device;
#endif

    char            name[sizeof(LWSWITCH_DEVICE_NAME) + LWSWITCH_INSTANCE_LEN];

    void            *os_handle;
    LwU32           os_instance;

    LwBool  is_emulation;
    LwBool  is_rtlsim;
    LwBool  is_fmodel;

    LWSWITCH_REGKEY_TYPE regkeys;

    // Tasks
    LWSWITCH_TASK_TYPE                  *tasks;

    // Errors
    LwU64                               error_total;    // Total errors recorded across all error logs
    LWSWITCH_ERROR_LOG_TYPE             log_FATAL_ERRORS;
    LWSWITCH_ERROR_LOG_TYPE             log_NONFATAL_ERRORS;

    LWSWITCH_FIRMWARE                   firmware;

    // HAL connectivity
    lwswitch_hal hal;

    // SOE
    FLCNABLE *pSoe;
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    // CCI
    struct CCI *pCci;
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    // DMA
    LwU32 dma_addr_width;

    // InfoROM
    struct inforom                      *pInforom;

    // I2C
    struct LWSWITCH_OBJI2C              *pI2c;

    // SMBPBI
    struct smbpbi                       *pSmbpbi;

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    // LWSWITCH_BOARD_ID_TYPE
    LWSWITCH_BOARD_ID_TYPE              board_id;
    LwU16                               int_board_id;
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    // LWSWITCH_LINK_TYPE
    LWSWITCH_LINK_TYPE                  link[LWSWITCH_MAX_LINK_COUNT];

    // PLL
    LWSWITCH_PLL_INFO                   switch_pll;

    // Device specific information
    LwU32                               chip_arch;      // LWSWITCH_GET_INFO_INDEX_ARCH_*
    LwU32                               chip_impl;      // LWSWITCH_GET_INFO_INDEX_IMPL_*
                                                        //
    LwU32                               chip_id;        // LW_PSMC/PMC_BOOT_42_CHIP_ID_*
    void *                              chip_device;

    // UUID in big-endian format
    LwUuid uuid;

    // Fabric Manager timeout value for the heartbeat
    LwU32 fm_timeout;

    // Fabric State
    LWSWITCH_DRIVER_FABRIC_STATE driver_fabric_state;
    LWSWITCH_DEVICE_FABRIC_STATE device_fabric_state;
    LWSWITCH_DEVICE_BLACKLIST_REASON device_blacklist_reason;
    LwU64 fabric_state_timestamp;
    LwU32 fabric_state_sequence_number;

    // Full BIOS image
    LWSWITCH_BIOS_IMAGE                  biosImage;

    // List of client events
    LWListRec                            client_events_list;
};

#define LWSWITCH_IS_DEVICE_VALID(device) \
    ((device != NULL) &&                 \
     (device->lwlink_device->type == LWLINK_DEVICE_TYPE_LWSWITCH))

#define LWSWITCH_IS_DEVICE_ACCESSIBLE(device) \
    (LWSWITCH_IS_DEVICE_VALID(device) &&      \
     (device->lwlink_device->pciInfo.bars[0].pBar != NULL))

#define LWSWITCH_IS_DEVICE_INITIALIZED(device) \
    (LWSWITCH_IS_DEVICE_ACCESSIBLE(device) &&  \
     (device->lwlink_device->initialized))

//
// Error Function defines
//

LwlStatus
lwswitch_construct_error_log
(
    LWSWITCH_ERROR_LOG_TYPE *errors,
    LwU32 error_log_size,
    LwBool overwritable
);

void
lwswitch_destroy_error_log
(
    lwswitch_device *device,
    LWSWITCH_ERROR_LOG_TYPE *errors
);

void
lwswitch_record_error
(
    lwswitch_device *device,
    LWSWITCH_ERROR_LOG_TYPE *errors,
    LwU32   error_type,                     // LWSWITCH_ERR_*
    LwU32   instance,
    LwU32   subinstance,
    LWSWITCH_ERROR_SRC_TYPE error_src,      // LWSWITCH_ERROR_SRC_*
    LWSWITCH_ERROR_SEVERITY_TYPE severity,  // LWSWITCH_ERROR_SEVERITY_*
    LwBool  error_resolved,
    void    *data,
    LwU32   data_size,
    LwU32   line
);

void
lwswitch_discard_errors
(
    LWSWITCH_ERROR_LOG_TYPE *errors,
    LwU32 error_discard_count
);

void
lwswitch_get_error
(
    lwswitch_device *device,
    LWSWITCH_ERROR_LOG_TYPE *errors,
    LWSWITCH_ERROR_TYPE *error_entry,
    LwU32   error_idx,
    LwU32   *error_count
);

void
lwswitch_get_next_error
(
    lwswitch_device *device,
    LWSWITCH_ERROR_LOG_TYPE *errors,
    LWSWITCH_ERROR_TYPE *error_entry,
    LwU32   *error_count,
    LwBool  remove_from_list
);

void
lwswitch_get_link_handlers
(
    lwlink_link_handlers *lwswitch_link_handlers
);

//
// Timeout checking
//

typedef struct LWSWITCH_TIMEOUT
{
    LwU64   timeout_ns;
} LWSWITCH_TIMEOUT;

#define LWSWITCH_INTERVAL_1USEC_IN_NS     1000LL
#define LWSWITCH_INTERVAL_50USEC_IN_NS    50000LL
#define LWSWITCH_INTERVAL_1MSEC_IN_NS     1000000LL
#define LWSWITCH_INTERVAL_5MSEC_IN_NS     5000000LL
#define LWSWITCH_INTERVAL_1SEC_IN_NS      1000000000LL

#define LWSWITCH_HEARTBEAT_INTERVAL_NS    LWSWITCH_INTERVAL_1SEC_IN_NS

// This should only be used for short delays
#define LWSWITCH_NSEC_DELAY(nsec_delay)                         \
do                                                              \
{                                                               \
    if (!IS_FMODEL(device))                                     \
    {                                                           \
        LWSWITCH_TIMEOUT timeout;                               \
        lwswitch_timeout_create(nsec_delay, &timeout);          \
        do { }                                                  \
        while (!lwswitch_timeout_check(&timeout));              \
    }                                                           \
} while(0)

#define LWSWITCH_GET_CAP(tbl,cap,field) (((LwU8)tbl[((1?cap##field)>=cap##_TBL_SIZE) ? 0/0 : (1?cap##field)]) & (0?cap##field))
#define LWSWITCH_SET_CAP(tbl,cap,field) ((tbl[((1?cap##field)>=cap##_TBL_SIZE) ? 0/0 : (1?cap##field)]) |= (0?cap##field))

#if LWCFG(GLOBAL_LWSWITCH_IMPL_SVNP01)
LwBool lwswitch_is_sv10_device_id(LwU32 device_id);
#endif
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
LwBool lwswitch_is_lr10_device_id(LwU32 device_id);
#endif
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
LwBool lwswitch_is_ls10_device_id(LwU32 device_id);
#endif

LwU32 lwswitch_reg_read_32(lwswitch_device *device, LwU32 offset);
void lwswitch_reg_write_32(lwswitch_device *device, LwU32 offset, LwU32 data);
LwU64 lwswitch_read_64bit_counter(lwswitch_device *device, LwU32 lo_offset, LwU32 hi_offset);
void lwswitch_timeout_create(LwU64 timeout_ns, LWSWITCH_TIMEOUT *time);
LwBool lwswitch_timeout_check(LWSWITCH_TIMEOUT *time);
void lwswitch_task_create(lwswitch_device *device,
void (*task_fn)(lwswitch_device *device), LwU64 period_nsec, LwU32 flags);
void lwswitch_tasks_destroy(lwswitch_device *device);

void lwswitch_free_chipdevice(lwswitch_device *device);
LwlStatus lwswitch_create_link(lwswitch_device *device, LwU32 link_number, lwlink_link **link);
lwlink_link* lwswitch_get_link(lwswitch_device *device, LwU8 link_id);
LwU64 lwswitch_get_enabled_link_mask(lwswitch_device *device);
void lwswitch_destroy_link(lwlink_link *link);
LwlStatus lwswitch_validate_pll_config(lwswitch_device *device,
                    LWSWITCH_PLL_INFO *switch_pll,
                    LWSWITCH_PLL_LIMITS default_pll_limits);

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
void lwswitch_fetch_active_repeater_mask(lwswitch_device *device);
LwU64 lwswitch_get_active_repeater_mask(lwswitch_device *device);
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

LwlStatus lwswitch_poll_sublink_state(lwswitch_device *device, lwlink_link *link);
void      lwswitch_setup_link_loopback_mode(lwswitch_device *device, LwU32 linkNumber);
void      lwswitch_reset_persistent_link_hw_state(lwswitch_device *device, LwU32 linkNumber);
void      lwswitch_store_topology_information(lwswitch_device *device, lwlink_link *link);

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
LwlStatus lwswitch_launch_ALI(lwswitch_device *device);
#endif //LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
LwlStatus lwswitch_set_training_mode(lwswitch_device *device);
LwBool    lwswitch_is_link_in_reset(lwswitch_device *device, lwlink_link *link);
void      lwswitch_apply_recal_settings(lwswitch_device *device, lwlink_link *link);
void lwswitch_init_buffer_ready(lwswitch_device *device, lwlink_link *link, LwBool bNportBufferReady);

#endif //_COMMON_LWSWITCH_H_
