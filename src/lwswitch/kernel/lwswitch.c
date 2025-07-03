/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"
#include "rom_lwswitch.h"
#include "error_lwswitch.h"
#include "regkey_lwswitch.h"
#include "bios_lwswitch.h"
#include "haldef_lwswitch.h"
#include "flcn/haldefs_flcnable_lwswitch.h"
#include "flcn/flcn_lwswitch.h"
#include "soe/soe_lwswitch.h"
#include "lwVer.h"
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#include "cci/cci_lwswitch.h"
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#define LWSWITCH_DEV_CMD_CHECK_ADMIN    LWBIT64(0)
#define LWSWITCH_DEV_CMD_CHECK_FM       LWBIT64(1)

#define LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(cmd, function, type, private, flags)\
    case cmd:                                                                    \
    {                                                                            \
        if (sizeof(type) != size)                                                \
        {                                                                        \
            retval = -LWL_BAD_ARGS;                                              \
            break;                                                               \
        }                                                                        \
                                                                                 \
        retval = _lwswitch_lib_validate_privileged_ctrl(private, flags);         \
        if (retval != LWL_SUCCESS)                                               \
        {                                                                        \
            break;                                                               \
        }                                                                        \
                                                                                 \
        retval = function(device, params);                                       \
        break;                                                                   \
    }                                                                            \

//
// HW's device id list can be found here -
// P4hw:2001: hw\doc\engr\Dev_ID\DeviceID_master_list.txt
//
#if LWCFG(GLOBAL_LWSWITCH_IMPL_SVNP01)
const static LwU32 lwswitch_sv10_device_ids[] =
{
    0x10F5, 0x1AC0, 0x1AC1, 0x1AC2, 0x1AC3, 0x1AC4, 0x1AC5, 0x1AC6, 0x1AC7,
    0x1AC8, 0x1AC9, 0x1ACA, 0x1ACB, 0x1ACC, 0x1ACD, 0x1ACE, 0x1ACF
};
#endif

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
const static LwU32 lwswitch_lr10_device_ids[] =
{
    0x1AE8, 0x1AF0, 0x1AF1, 0x1AF2, 0x1AF3, 0x1AF4, 0x1AF5, 0x1AF6, 0x1AF7,
    0x1AF8, 0x1AF9, 0x1AFA, 0x1AFB, 0x1AFC, 0x1AFD, 0x1AFE, 0x1AFF
};
#endif

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
const static LwU32 lwswitch_ls10_device_ids[] =
{
    // PCIE endpoint to manage the LWLink switch HW
    0x22A0, 0x22A1, 0x22A2, 0x22A3, 0x22A4, 0x22A5, 0x22A6, 0x22A7,
    // PCI-PCI Bridge, Laguna Switch Function 0
    0x22A8, 0x22A9, 0x22AA, 0x22AB,
    // Non-Transparent Bridge, Laguna Switch Function 1
    0x22AC, 0x22AD, 0x22AE, 0x22AF
};
#endif

lwlink_link_handlers link_handlers;

static LwBool
_lwswitch_is_device_id_present
(
    const LwU32 *array,
    LwU32 array_len,
    LwU32 device_id
)
{
    LwU32 i = 0;

    for(i = 0; i < array_len; i++)
    {
        if (array[i] == device_id)
        {
            return LW_TRUE;
        }
    }

    return LW_FALSE;
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_SVNP01)
LwBool
lwswitch_is_sv10_device_id
(
    LwU32 device_id
)
{
    LwU32 count = (sizeof(lwswitch_sv10_device_ids) /
                        sizeof(lwswitch_sv10_device_ids[0]));

    return _lwswitch_is_device_id_present(lwswitch_sv10_device_ids, count, device_id);
}
#endif

LwBool
lwswitch_is_lr10_device_id
(
    LwU32 device_id
)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    LwU32 count = (sizeof(lwswitch_lr10_device_ids) /
                        sizeof(lwswitch_lr10_device_ids[0]));

    return _lwswitch_is_device_id_present(lwswitch_lr10_device_ids, count, device_id);
#else
    return LW_FALSE;
#endif
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
LwBool
lwswitch_is_ls10_device_id
(
    LwU32 device_id
)
{
    LwU32 count = (sizeof(lwswitch_ls10_device_ids) /
                        sizeof(lwswitch_ls10_device_ids[0]));

    return _lwswitch_is_device_id_present(lwswitch_ls10_device_ids, count, device_id);
}
#endif // LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

/*
 * LWLink corelib callbacks are used by the LWLink library separate from the
 * LWSwitch driver, therefore they do not take a device lock and can not modify
 * lwswitch_device state or use error logging.
 *
 * These LWSwitch functions modify link state outside of the corelib:
 *   _lwswitch_ctrl_inject_link_error - injects asynchronous link errors (MODS-only)
 */

static LW_API_CALL LwlStatus
_lwswitch_corelib_add_link
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_add_link(link);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_remove_link
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_remove_link(link);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_set_dl_link_mode
(
    lwlink_link *link,
    LwU64 mode,
    LwU32 flags
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_set_dl_link_mode(link, mode, flags);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_get_dl_link_mode
(
    lwlink_link *link,
    LwU64 *mode
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_get_dl_link_mode(link, mode);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_set_tl_link_mode
(
    lwlink_link *link,
    LwU64 mode,
    LwU32 flags
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_set_tl_link_mode(link, mode, flags);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_get_tl_link_mode
(
    lwlink_link *link,
    LwU64 *mode
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_get_tl_link_mode(link, mode);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_set_tx_mode
(
    lwlink_link *link,
    LwU64 mode,
    LwU32 flags
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_set_tx_mode(link, mode, flags);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_get_tx_mode
(
    lwlink_link *link,
    LwU64 *mode,
    LwU32 *subMode
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_get_tx_mode(link, mode, subMode);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_set_rx_mode
(
    lwlink_link *link,
    LwU64 mode,
    LwU32 flags
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_set_rx_mode(link, mode, flags);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_get_rx_mode
(
    lwlink_link *link,
    LwU64 *mode,
    LwU32 *subMode
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_get_rx_mode(link, mode, subMode);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_set_rx_detect
(
    lwlink_link *link,
    LwU32 flags
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_set_rx_detect(link, flags);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_get_rx_detect
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_get_rx_detect(link);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_write_discovery_token
(
    lwlink_link *link,
    LwU64 token
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_write_discovery_token(link, token);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_read_discovery_token
(
    lwlink_link *link,
    LwU64 *token
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_read_discovery_token(link, token);
}

static LW_API_CALL void
_lwswitch_corelib_training_complete
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    device->hal.lwswitch_corelib_training_complete(link);
}

static LW_API_CALL void
_lwswitch_corelib_get_uphy_load
(
    lwlink_link *link,
    LwBool *bUnlocked
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_get_uphy_load(link, bUnlocked);
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
static LW_API_CALL LwlStatus
_lwswitch_corelib_enable_optical_maintenance
(
    lwlink_link *link,
    LwBool bTx
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_enable_optical_maintenance(link, bTx);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_set_optical_infinite_mode
(
    lwlink_link *link,
    LwBool bEnable
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_set_optical_infinite_mode(link, bEnable);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_set_optical_iobist
(
    lwlink_link *link,
    LwBool bEnable
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_set_optical_iobist(link, bEnable);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_set_optical_pretrain
(
    lwlink_link *link,
    LwBool bTx,
    LwBool bEnable
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_set_optical_pretrain(link, bTx, bEnable);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_check_optical_pretrain
(
    lwlink_link *link,
    LwBool bTx,
    LwBool* bSuccess
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_check_optical_pretrain(link, bTx, bSuccess);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_init_optical_links
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_init_optical_links(link);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_set_optical_force_eq
(
    lwlink_link *link,
    LwBool bEnable
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_set_optical_force_eq(link, bEnable);
}

static LW_API_CALL LwlStatus
_lwswitch_corelib_check_optical_eom_status
(
    lwlink_link *link,
    LwBool* bEomLow
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    return device->hal.lwswitch_corelib_check_optical_eom_status(link, bEomLow);
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

void
lwswitch_get_link_handlers
(
    lwlink_link_handlers *lwswitch_link_handlers
)
{
    if (!lwswitch_link_handlers)
    {
        LWSWITCH_ASSERT(0);
        return;
    }

    lwswitch_link_handlers->add = _lwswitch_corelib_add_link;
    lwswitch_link_handlers->remove = _lwswitch_corelib_remove_link;
    lwswitch_link_handlers->set_dl_link_mode = _lwswitch_corelib_set_dl_link_mode;
    lwswitch_link_handlers->get_dl_link_mode = _lwswitch_corelib_get_dl_link_mode;
    lwswitch_link_handlers->set_tl_link_mode = _lwswitch_corelib_set_tl_link_mode;
    lwswitch_link_handlers->get_tl_link_mode = _lwswitch_corelib_get_tl_link_mode;
    lwswitch_link_handlers->set_tx_mode = _lwswitch_corelib_set_tx_mode;
    lwswitch_link_handlers->get_tx_mode = _lwswitch_corelib_get_tx_mode;
    lwswitch_link_handlers->set_rx_mode = _lwswitch_corelib_set_rx_mode;
    lwswitch_link_handlers->get_rx_mode = _lwswitch_corelib_get_rx_mode;
    lwswitch_link_handlers->set_rx_detect = _lwswitch_corelib_set_rx_detect;
    lwswitch_link_handlers->get_rx_detect = _lwswitch_corelib_get_rx_detect;
    lwswitch_link_handlers->write_discovery_token = _lwswitch_corelib_write_discovery_token;
    lwswitch_link_handlers->read_discovery_token = _lwswitch_corelib_read_discovery_token;
    lwswitch_link_handlers->training_complete = _lwswitch_corelib_training_complete;
    lwswitch_link_handlers->get_uphy_load = _lwswitch_corelib_get_uphy_load;
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    lwswitch_link_handlers->set_optical_infinite_mode = _lwswitch_corelib_set_optical_infinite_mode;
    lwswitch_link_handlers->enable_optical_maintenance = _lwswitch_corelib_enable_optical_maintenance;
    lwswitch_link_handlers->set_optical_pretrain = _lwswitch_corelib_set_optical_pretrain;
    lwswitch_link_handlers->set_optical_iobist = _lwswitch_corelib_set_optical_iobist;
    lwswitch_link_handlers->check_optical_pretrain = _lwswitch_corelib_check_optical_pretrain;
    lwswitch_link_handlers->init_optical_links = _lwswitch_corelib_init_optical_links;
    lwswitch_link_handlers->set_optical_force_eq = _lwswitch_corelib_set_optical_force_eq;
    lwswitch_link_handlers->check_optical_eom_status = _lwswitch_corelib_check_optical_eom_status;
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
}

#define LWSWITCH_INIT_REGKEY(_private, _regkey, _string, _default_val)          \
do                                                                              \
{                                                                               \
    LwU32 data;                                                                 \
                                                                                \
    device->regkeys._regkey = _default_val;                                     \
    if (LW_SWITCH_REGKEY_PRIVATE_ALLOWED || !LW_SWITCH_REGKEY##_private)        \
    {                                                                           \
        if (LWL_SUCCESS ==                                                      \
            lwswitch_os_read_registry_dword(device->os_handle, _string, &data)) \
        {                                                                       \
            LWSWITCH_PRINT(device, SETUP,                                       \
                "%s: Applying regkey %s=0x%x\n",                                \
                __FUNCTION__,                                                   \
                _string, data);                                                 \
            device->regkeys._regkey = data;                                     \
        }                                                                       \
    }                                                                           \
} while(0)

static void
_lwswitch_init_device_regkeys
(
    lwswitch_device *device
)
{
    //
    // Public external use regkeys
    //
    LWSWITCH_INIT_REGKEY(_PUBLIC, ato_control,
                         LW_SWITCH_REGKEY_ATO_CONTROL,
                         LW_SWITCH_REGKEY_ATO_CONTROL_DEFAULT);

    LWSWITCH_INIT_REGKEY(_PUBLIC, sto_control,
                         LW_SWITCH_REGKEY_STO_CONTROL,
                         LW_SWITCH_REGKEY_STO_CONTROL_DEFAULT);

    LWSWITCH_INIT_REGKEY(_PUBLIC, crc_bit_error_rate_short,
                         LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT,
                         LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_OFF);

    LWSWITCH_INIT_REGKEY(_PUBLIC, crc_bit_error_rate_long,
                         LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG,
                         LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_OFF);

    //
    // Private internal use regkeys
    // Not available on release build kernel drivers
    //
    LWSWITCH_INIT_REGKEY(_PRIVATE, external_fabric_mgmt,
                         LW_SWITCH_REGKEY_EXTERNAL_FABRIC_MGMT,
                         LW_SWITCH_REGKEY_EXTERNAL_FABRIC_MGMT_ENABLE);

    LWSWITCH_INIT_REGKEY(_PRIVATE, txtrain_control,
                         LW_SWITCH_REGKEY_TXTRAIN_CONTROL,
                         LW_SWITCH_REGKEY_TXTRAIN_CONTROL_NOP);

    LWSWITCH_INIT_REGKEY(_PRIVATE, crossbar_DBI,
                         LW_SWITCH_REGKEY_CROSSBAR_DBI,
                         LW_SWITCH_REGKEY_CROSSBAR_DBI_ENABLE);

    LWSWITCH_INIT_REGKEY(_PRIVATE, link_DBI,
                         LW_SWITCH_REGKEY_LINK_DBI,
                         LW_SWITCH_REGKEY_LINK_DBI_ENABLE);

    LWSWITCH_INIT_REGKEY(_PRIVATE, ac_coupled_mask,
                         LW_SWITCH_REGKEY_AC_COUPLED_MASK,
                         0);

    LWSWITCH_INIT_REGKEY(_PRIVATE, ac_coupled_mask2,
                         LW_SWITCH_REGKEY_AC_COUPLED_MASK2,
                         0);

    LWSWITCH_INIT_REGKEY(_PRIVATE, swap_clk,
                         LW_SWITCH_REGKEY_SWAP_CLK_OVERRIDE,
                         lwswitch_get_swap_clk_default(device));

    LWSWITCH_INIT_REGKEY(_PRIVATE, link_enable_mask,
                         LW_SWITCH_REGKEY_ENABLE_LINK_MASK,
                         LW_U32_MAX);

    LWSWITCH_INIT_REGKEY(_PRIVATE, link_enable_mask2,
                         LW_SWITCH_REGKEY_ENABLE_LINK_MASK2,
                         LW_U32_MAX);

    LWSWITCH_INIT_REGKEY(_PRIVATE, bandwidth_shaper,
                         LW_SWITCH_REGKEY_BANDWIDTH_SHAPER,
                         LW_SWITCH_REGKEY_BANDWIDTH_SHAPER_PROD);

    LWSWITCH_INIT_REGKEY(_PRIVATE, ssg_control,
                         LW_SWITCH_REGKEY_SSG_CONTROL,
                         0);

    LWSWITCH_INIT_REGKEY(_PRIVATE, skip_buffer_ready,
                         LW_SWITCH_REGKEY_SKIP_BUFFER_READY,
                         0);

    LWSWITCH_INIT_REGKEY(_PRIVATE, enable_pm,
                         LW_SWITCH_REGKEY_ENABLE_PM,
                         LW_SWITCH_REGKEY_ENABLE_PM_YES);

    LWSWITCH_INIT_REGKEY(_PRIVATE, chiplib_forced_config_link_mask,
                         LW_SWITCH_REGKEY_CHIPLIB_FORCED_LINK_CONFIG_MASK,
                         0);

    LWSWITCH_INIT_REGKEY(_PRIVATE, chiplib_forced_config_link_mask2,
                         LW_SWITCH_REGKEY_CHIPLIB_FORCED_LINK_CONFIG_MASK2,
                         0);

    LWSWITCH_INIT_REGKEY(_PRIVATE, soe_dma_self_test,
                         LW_SWITCH_REGKEY_SOE_DMA_SELFTEST,
                         LW_SWITCH_REGKEY_SOE_DMA_SELFTEST_ENABLE);

    LWSWITCH_INIT_REGKEY(_PRIVATE, soe_disable,
                         LW_SWITCH_REGKEY_SOE_DISABLE,
                         LW_SWITCH_REGKEY_SOE_DISABLE_NO);

    LWSWITCH_INIT_REGKEY(_PUBLIC, soe_boot_core,
                         LW_SWITCH_REGKEY_SOE_BOOT_CORE,
                         LW_SWITCH_REGKEY_SOE_BOOT_CORE_DEFAULT);

    LWSWITCH_INIT_REGKEY(_PRIVATE, minion_cache_seeds,
                         LW_SWITCH_REGKEY_MINION_CACHE_SEEDS,
                         LW_SWITCH_REGKEY_MINION_CACHE_SEEDS_DISABLE);

    LWSWITCH_INIT_REGKEY(_PRIVATE, latency_counter,
                         LW_SWITCH_REGKEY_LATENCY_COUNTER_LOGGING,
                         LW_SWITCH_REGKEY_LATENCY_COUNTER_LOGGING_ENABLE);

    LWSWITCH_INIT_REGKEY(_PRIVATE, lwlink_speed_control,
                         LW_SWITCH_REGKEY_SPEED_CONTROL,
                         LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_DEFAULT);

    LWSWITCH_INIT_REGKEY(_PRIVATE, inforom_bbx_periodic_flush,
                         LW_SWITCH_REGKEY_INFOROM_BBX_ENABLE_PERIODIC_FLUSHING,
                         LW_SWITCH_REGKEY_INFOROM_BBX_ENABLE_PERIODIC_FLUSHING_DISABLE);

    LWSWITCH_INIT_REGKEY(_PRIVATE, inforom_bbx_write_periodicity,
                         LW_SWITCH_REGKEY_INFOROM_BBX_WRITE_PERIODICITY,
                         LW_SWITCH_REGKEY_INFOROM_BBX_WRITE_PERIODICITY_DEFAULT);

    LWSWITCH_INIT_REGKEY(_PRIVATE, inforom_bbx_write_min_duration,
                         LW_SWITCH_REGKEY_INFOROM_BBX_WRITE_MIN_DURATION,
                         LW_SWITCH_REGKEY_INFOROM_BBX_WRITE_MIN_DURATION_DEFAULT);

    LWSWITCH_INIT_REGKEY(_PRIVATE, minion_disable,
                         LW_SWITCH_REGKEY_MINION_DISABLE,
                         LW_SWITCH_REGKEY_MINION_DISABLE_NO);

    LWSWITCH_INIT_REGKEY(_PRIVATE, set_ucode_target,
                         LW_SWITCH_REGKEY_MINION_SET_UCODE_TARGET,
                         LW_SWITCH_REGKEY_MINION_SET_UCODE_TARGET_DEFAULT);

    LWSWITCH_INIT_REGKEY(_PRIVATE, set_simmode,
                         LW_SWITCH_REGKEY_MINION_SET_SIMMODE,
                         LW_SWITCH_REGKEY_MINION_SET_SIMMODE_DEFAULT);

    LWSWITCH_INIT_REGKEY(_PRIVATE, set_smf_settings,
                         LW_SWITCH_REGKEY_MINION_SET_SMF_SETTINGS,
                         LW_SWITCH_REGKEY_MINION_SET_SMF_SETTINGS_DEFAULT);

    LWSWITCH_INIT_REGKEY(_PRIVATE, select_uphy_tables,
                         LW_SWITCH_REGKEY_MINION_SELECT_UPHY_TABLES,
                         LW_SWITCH_REGKEY_MINION_SELECT_UPHY_TABLES_DEFAULT);

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_INIT_REGKEY(_PRIVATE, link_training_mode,
                         LW_SWITCH_REGKEY_LINK_TRAINING_SELECT,
                         LW_SWITCH_REGKEY_LINK_TRAINING_SELECT_DEFAULT);
#endif

    LWSWITCH_INIT_REGKEY(_PRIVATE, i2c_access_control,
                         LW_SWITCH_REGKEY_I2C_ACCESS_CONTROL,
                         LW_SWITCH_REGKEY_I2C_ACCESS_CONTROL_DEFAULT);

    LWSWITCH_INIT_REGKEY(_PRIVATE, link_recal_settings,
                         LW_SWITCH_REGKEY_LINK_RECAL_SETTINGS,
                         LW_SWITCH_REGKEY_LINK_RECAL_SETTINGS_NOP);
}

void
lwswitch_save_lwlink_seed_data_from_minion_to_inforom
(
    lwswitch_device *device,
    LwU32 linkId
)
{
   device->hal.lwswitch_save_lwlink_seed_data_from_minion_to_inforom(device, linkId);
}

void
lwswitch_store_seed_data_from_inforom_to_corelib
(
    lwswitch_device *device
)
{
   device->hal.lwswitch_store_seed_data_from_inforom_to_corelib(device);
}

LwU64
lwswitch_lib_deferred_task_dispatcher
(
    lwswitch_device *device
)
{
    LwU64 time_nsec;
    LwU64 time_next_nsec = lwswitch_os_get_platform_time() + 100*LWSWITCH_INTERVAL_1MSEC_IN_NS;
    LWSWITCH_TASK_TYPE *task;

    if (!LWSWITCH_IS_DEVICE_VALID(device))
    {
        return LW_U64_MAX;
    }

    task = device->tasks;

    // Walk the task list, exelwting those whose next exelwtion interval is at hand
    while (task)
    {
        // Get current time (nsec) for scheduling
        time_nsec = lwswitch_os_get_platform_time();

        if (time_nsec >= task->last_run_nsec + task->period_nsec)
        {
            //
            // The task has never been run or it is time to run
            // Mark its last run time
            //
            task->last_run_nsec = time_nsec;
            // Run the task
            if (LWSWITCH_IS_DEVICE_INITIALIZED(device) ||
               (task->flags & LWSWITCH_TASK_TYPE_FLAGS_ALWAYS_RUN))
                (*task->task_fn)(device);
        }

        // Determine its next run time
        time_next_nsec = LW_MIN(task->last_run_nsec + task->period_nsec, time_next_nsec);
        task = task->next;
    }

    time_nsec = lwswitch_os_get_platform_time();

    // Return to the OS layer how long to wait before calling again
    return(time_next_nsec >= time_nsec ? time_next_nsec - time_nsec : 0);
}

static LwlStatus
_lwswitch_setup_hal
(
    lwswitch_device *device,
    LwU32 pci_device_id
)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_SVNP01)
    if (lwswitch_is_sv10_device_id(pci_device_id))
    {
        lwswitch_setup_hal_sv10(device);
        return LWL_SUCCESS;
    }
#endif
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    if (lwswitch_is_lr10_device_id(pci_device_id))
    {
        lwswitch_setup_hal_lr10(device);
        return LWL_SUCCESS;
    }
#endif
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10) && (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    if (lwswitch_is_ls10_device_id(pci_device_id))
    {
        lwswitch_setup_hal_ls10(device);
        return LWL_SUCCESS;
    }
#endif
    LWSWITCH_PRINT(device, ERROR,
        "LWSwitch HAL setup failed - Unrecognized PCI Device ID\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_lib_check_api_version
(
    const char *user_version,
    char *kernel_version,
    LwU32 length
)
{
    const LwLength VERSION_LENGTH = lwswitch_os_strlen(LW_VERSION_STRING);

    if (kernel_version == NULL || user_version == NULL)
    {
        return -LWL_BAD_ARGS;
    }

    if (length < VERSION_LENGTH)
    {
        return -LWL_NO_MEM;
    }

    lwswitch_os_memset(kernel_version, 0x0, length);
    lwswitch_os_strncpy(kernel_version, LW_VERSION_STRING, VERSION_LENGTH);

    kernel_version[length - 1] = '\0';

    if (lwswitch_os_strncmp(user_version, kernel_version, VERSION_LENGTH))
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return LWL_SUCCESS;
}

LwBool
lwswitch_is_inforom_supported
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_is_inforom_supported(device);
}

LwBool
lwswitch_is_spi_supported
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_is_spi_supported(device);
}

LwBool
lwswitch_is_smbpbi_supported
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_is_smbpbi_supported(device);
}

LwlStatus
lwswitch_soe_prepare_for_reset
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_soe_prepare_for_reset(device);
}

LwBool
lwswitch_is_soe_supported
(
    lwswitch_device *device
)
{
    if (device->regkeys.soe_disable == LW_SWITCH_REGKEY_SOE_DISABLE_YES)
    {
        LWSWITCH_PRINT(device, INFO, "SOE is disabled via regkey.\n");
        return LW_FALSE;
    }

    return device->hal.lwswitch_is_soe_supported(device);
}

LwlStatus
lwswitch_soe_set_ucode_core
(
    lwswitch_device *device,
    LwBool bFalcon
)
{
    return device->hal.lwswitch_soe_set_ucode_core(device, bFalcon);
}

LwlStatus
lwswitch_init_soe
(
    lwswitch_device *device
)
{
    if (device->regkeys.soe_disable == LW_SWITCH_REGKEY_SOE_DISABLE_YES)
    {
        LWSWITCH_PRINT(device, INFO, "SOE is disabled via regkey.\n");
        return LW_FALSE;
    }

    return device->hal.lwswitch_init_soe(device);
}

static LwlStatus
_lwswitch_construct_soe
(
    lwswitch_device *device
)
{
    FLCNABLE *pSoe = NULL;
    LwlStatus retval;

    device->pSoe = pSoe = (PFLCNABLE)soeAllocNew();
    if (pSoe == NULL)
    {
        LWSWITCH_PRINT(device, ERROR, "SOE allocation failed.\n");
        return -LWL_NO_MEM;
    }

    retval = soeInit(device, (PSOE)pSoe, device->lwlink_device->pciInfo.pciDeviceId);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "SOE init failed.\n");
        goto soe_init_failed;
    }

    if (flcnableConstruct_HAL(device, pSoe) != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "FALCON construct failed.\n");
        retval = -LWL_ERR_ILWALID_STATE;
        goto flcn_construct_failed;
    }

    return LWL_SUCCESS;

flcn_construct_failed:
    soeDestroy(device, (PSOE)pSoe);

soe_init_failed:
    lwswitch_os_free(pSoe);
    device->pSoe = NULL;

    return retval;
}

static void
_lwswitch_destruct_soe
(
    lwswitch_device *device
)
{
    FLCNABLE *pSoe = device->pSoe;

    if (pSoe == NULL)
    {
        return;
    }

    flcnableDestruct_HAL(device, pSoe);
    soeDestroy(device, (PSOE)pSoe);

    lwswitch_os_free(pSoe);
    device->pSoe = NULL;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

static LwlStatus
_lwswitch_construct_cci
(
    lwswitch_device *device
)
{
    CCI *pCci = NULL;
    LwlStatus retval;

    device->pCci = pCci = cciAllocNew();
    if (pCci == NULL)
    {
        LWSWITCH_PRINT(device, ERROR, "CCI allocation failed.\n");
        return -LWL_NO_MEM;
    }

    retval = cciInit(device, pCci, device->lwlink_device->pciInfo.pciDeviceId);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "CCI init failed.\n");
        goto cci_init_failed;
    }

    return LWL_SUCCESS;

cci_init_failed:
    lwswitch_os_free(pCci);
    device->pCci = NULL;

    return retval;
}

static void
_lwswitch_destruct_cci
(
    lwswitch_device *device
)
{
    CCI *pCci = device->pCci;

    if (pCci == NULL)
    {
        return;
    }

    cciDestroy(device, pCci);

    lwswitch_os_free(pCci);
    device->pCci = NULL;
}

#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

static LwlStatus
_lwswitch_initialize_device_state
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_initialize_device_state(device);
}

static LwlStatus
_lwswitch_post_init_device_setup
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_post_init_device_setup(device);
}

static LwlStatus
_lwswitch_setup_link_system_registers
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_setup_link_system_registers(device);
}

static void
_lwswitch_post_init_blacklist_device_setup
(
    lwswitch_device *device
)
{
    device->hal.lwswitch_post_init_blacklist_device_setup(device);
}

static void
_lwswitch_set_dma_mask
(
    lwswitch_device *device
)
{
    LwU32 hw_dma_width, retval;

    hw_dma_width = device->hal.lwswitch_get_device_dma_width(device);

    if (hw_dma_width == 0)
    {
        LWSWITCH_PRINT(device, INFO, "DMA is not supported on this device\n");
        return;
    }

    retval = lwswitch_os_set_dma_mask(device->os_handle, hw_dma_width);
    if (retval == LWL_SUCCESS)
    {
        device->dma_addr_width = hw_dma_width;
        return;
    }

    LWSWITCH_PRINT(device, SETUP,
                   "%s: Failed to set DMA mask, trying 32-bit fallback : %d\n",
                   __FUNCTION__, retval);

    retval = lwswitch_os_set_dma_mask(device->os_handle, 32);
    if (retval == LWL_SUCCESS)
    {
        device->dma_addr_width = 32;
        return;
    }

    // failure is not fatal, the driver will just restrict DMA functionality
    LWSWITCH_PRINT(device, ERROR, "Failed to set DMA mask : %d\n", retval);
}

LwlStatus
lwswitch_deassert_link_reset
(
    lwswitch_device *device,
    lwlink_link *link
)
{
    return device->hal.lwswitch_deassert_link_reset(device, link);
}

LwU32
lwswitch_get_sublink_width
(
    lwswitch_device *device,
    LwU32 linkNumber
)
{
    return device->hal.lwswitch_get_sublink_width(device, linkNumber);
}

static void
_lwswitch_unregister_links
(
    lwswitch_device *device
)
{
    lwlink_link *link = NULL;
    LwU32 link_num;
    LwBool is_blacklisted;


    if (!LWSWITCH_IS_DEVICE_INITIALIZED(device))
        return;

    device->lwlink_device->initialized = 0;
    is_blacklisted = (device->device_fabric_state == LWSWITCH_DEVICE_FABRIC_STATE_BLACKLISTED);

    for (link_num = 0; link_num < lwswitch_get_num_links(device); link_num++)
    {
        if (lwlink_lib_get_link(device->lwlink_device, link_num, &link) == LWL_SUCCESS)
        {
            if (!is_blacklisted)
                lwswitch_save_lwlink_seed_data_from_minion_to_inforom(device, link_num);
            lwlink_lib_unregister_link(link);
            lwswitch_destroy_link(link);
        }
    }

    if (!is_blacklisted)
        lwswitch_inforom_lwlink_flush(device);
}

LwlStatus LW_API_CALL
lwswitch_lib_read_fabric_state
(
    lwswitch_device *device,
    LWSWITCH_DEVICE_FABRIC_STATE *device_fabric_state,
    LWSWITCH_DEVICE_BLACKLIST_REASON *device_blacklist_reason,
    LWSWITCH_DRIVER_FABRIC_STATE *driver_fabric_state
)
{
    if (!LWSWITCH_IS_DEVICE_ACCESSIBLE(device))
        return -LWL_BAD_ARGS;

    if (device_fabric_state != NULL)
        *device_fabric_state = device->device_fabric_state;

    if (device_blacklist_reason != NULL)
        *device_blacklist_reason = device->device_blacklist_reason;

    if (driver_fabric_state != NULL)
        *driver_fabric_state = device->driver_fabric_state;

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_lib_blacklist_device
(
    lwswitch_device *device,
    LWSWITCH_DEVICE_BLACKLIST_REASON device_blacklist_reason
)
{
    LwlStatus status;

    if (!LWSWITCH_IS_DEVICE_ACCESSIBLE(device))
    {
        return -LWL_BAD_ARGS;
    }

    if (device->device_fabric_state == LWSWITCH_DEVICE_FABRIC_STATE_BLACKLISTED)
    {
        LWSWITCH_PRINT(device, WARN, "Device is already blacklisted\n");
        return -LWL_ERR_NOT_SUPPORTED;
    }

    device->device_fabric_state = LWSWITCH_DEVICE_FABRIC_STATE_BLACKLISTED;
    device->device_blacklist_reason = device_blacklist_reason;

    status = device->hal.lwswitch_write_fabric_state(device);
    if (status != LWL_SUCCESS)
        LWSWITCH_PRINT(device, INFO, "Cannot send fabric state to SOE\n");

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_ctrl_blacklist_device(
    lwswitch_device *device,
    LWSWITCH_BLACKLIST_DEVICE_PARAMS *p
)
{
    LwlStatus status;

    status = lwswitch_lib_blacklist_device(device, p->deviceReason);
    if (status != LWL_SUCCESS)
        return status;

    lwswitch_lib_disable_interrupts(device);

    // Unregister links from LWLinkCoreLib, so that link training is not
    // attempted
    _lwswitch_unregister_links(device);

    // Keep device registered for HAL access and Fabric State updates

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_ctrl_set_fm_driver_state(
    lwswitch_device *device,
    LWSWITCH_SET_FM_DRIVER_STATE_PARAMS *p
)
{
    if (!LWSWITCH_IS_DEVICE_ACCESSIBLE(device))
    {
        return -LWL_BAD_ARGS;
    }

    device->driver_fabric_state = p->driverState;
    device->fabric_state_timestamp = lwswitch_os_get_platform_time();

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_ctrl_set_device_fabric_state(
    lwswitch_device *device,
    LWSWITCH_SET_DEVICE_FABRIC_STATE_PARAMS *p
)
{
    if (!LWSWITCH_IS_DEVICE_ACCESSIBLE(device))
    {
        return -LWL_BAD_ARGS;
    }

    if (device->device_fabric_state == LWSWITCH_DEVICE_FABRIC_STATE_BLACKLISTED)
        return -LWL_ERR_NOT_SUPPORTED;

    device->device_fabric_state = p->deviceState;
    device->fabric_state_timestamp = lwswitch_os_get_platform_time();

    // If FM had exceeded timeout, reset the status to not timed-out
    if (device->driver_fabric_state == LWSWITCH_DRIVER_FABRIC_STATE_MANAGER_TIMEOUT)
        device->driver_fabric_state = LWSWITCH_DRIVER_FABRIC_STATE_CONFIGURED;

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_ctrl_set_fm_timeout(
    lwswitch_device *device,
    LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT_PARAMS *p
)
{
    if (!LWSWITCH_IS_DEVICE_ACCESSIBLE(device))
    {
        return -LWL_BAD_ARGS;
    }

    device->fm_timeout = p->fmTimeout;

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_ctrl_register_events(
    lwswitch_device *device,
    LWSWITCH_REGISTER_EVENTS_PARAMS *p,
    void *osPrivate
)
{
    LwlStatus status = LWL_SUCCESS;
    LwU32 i;
    LwBool many_events, os_descriptor;
    void *osDescriptor = osPrivate;

    if (!LWSWITCH_IS_DEVICE_ACCESSIBLE(device))
    {
        return -LWL_BAD_ARGS;
    }

    status = lwswitch_os_get_supported_register_events_params(&many_events,
                                                              &os_descriptor);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    if ((!many_events && (p->numEvents > 1)) ||
        (p->numEvents == 0))
    {
        return -LWL_BAD_ARGS;
    }

    if (os_descriptor)
    {
        osDescriptor = (void *) p->osDescriptor;
    }

    for (i = 0; i < p->numEvents; i++)
    {
        status = lwswitch_lib_add_client_event(device, osDescriptor, p->eventIds[i]);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR, "%s: Failed to add client event.\n", __FUNCTION__);
            return status;
        }
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_ctrl_unregister_events(
    lwswitch_device *device,
    LWSWITCH_UNREGISTER_EVENTS_PARAMS *p,
    void *osPrivate
)
{
    LwlStatus status = LWL_SUCCESS;
    LwBool many_events, os_descriptor;
    void *osDescriptor = osPrivate;

    if (!LWSWITCH_IS_DEVICE_ACCESSIBLE(device))
    {
        return -LWL_BAD_ARGS;
    }

    status = lwswitch_os_get_supported_register_events_params(&many_events,
                                                              &os_descriptor);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    if (os_descriptor)
    {
        osDescriptor = (void *) p->osDescriptor;
    }

    status = lwswitch_lib_remove_client_events(device, osDescriptor);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed to remove client event.\n", __FUNCTION__);
        return status;
    }

    return LWL_SUCCESS;
}

static void
lwswitch_fabric_state_heartbeat(
    lwswitch_device *device
)
{
    LwU64 age;

    if (!LWSWITCH_IS_DEVICE_VALID(device))
        return;

    age = lwswitch_os_get_platform_time() - device->fabric_state_timestamp;

    // Check to see if we have exceeded the FM timeout
    if (device->driver_fabric_state == LWSWITCH_DRIVER_FABRIC_STATE_CONFIGURED &&
        age > (LwU64)device->fm_timeout * 1000ULL * 1000ULL)
         device->driver_fabric_state = LWSWITCH_DRIVER_FABRIC_STATE_MANAGER_TIMEOUT;

    (void)device->hal.lwswitch_write_fabric_state(device);
}

static LwlStatus
_lwswitch_ctrl_set_training_error_info
(
    lwswitch_device *device,
    LWSWITCH_SET_TRAINING_ERROR_INFO_PARAMS *p
)
{
    return device->hal.lwswitch_set_training_error_info(device, p);
}

static LwlStatus
_lwswitch_ctrl_get_fatal_error_scope
(
    lwswitch_device *device,
    LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS *pParams
)
{
    return device->hal.lwswitch_ctrl_get_fatal_error_scope(device, pParams);
}

static LwlStatus
_lwswitch_ctrl_therm_get_temperature_limit
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_TEMPERATURE_LIMIT_PARAMS *pParams
)
{
    if (!LWSWITCH_IS_DEVICE_ACCESSIBLE(device))
    {
        return -LWL_BAD_ARGS;
    }

    return device->hal.lwswitch_ctrl_therm_get_temperature_limit(device, pParams);
}

LwlStatus
lwswitch_lib_initialize_device
(
    lwswitch_device *device
)
{
    LwlStatus retval = LWL_SUCCESS;
    LwU8 link_num;
    lwlink_link *link = NULL;
    LwBool is_blacklisted_by_os = LW_FALSE;

    if (!LWSWITCH_IS_DEVICE_ACCESSIBLE(device))
    {
        return -LWL_BAD_ARGS;
    }

    if (LWSWITCH_IS_DEVICE_INITIALIZED(device))
    {
        LWSWITCH_PRINT(device, SETUP, "Device is already initialized!\n");
        return LWL_SUCCESS;
    }

    LWSWITCH_PRINT(device, SETUP,
        "Initializing lwswitch at (%04x:%02x:%02x.%02x)\n",
        device->lwlink_device->pciInfo.domain,
        device->lwlink_device->pciInfo.bus,
        device->lwlink_device->pciInfo.device,
        device->lwlink_device->pciInfo.function);

    lwListInit(&device->client_events_list);

    retval = lwswitch_lib_load_platform_info(device);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to load platform information\n");
        return retval;
    }

    if (lwswitch_is_soe_supported(device))
    {
        retval = _lwswitch_construct_soe(device);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }
    }
    else
    {
        LWSWITCH_PRINT(device, INFO, "SOE is not supported, skipping construct\n");
    }
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    if (lwswitch_is_cci_supported(device))
    {
        retval = _lwswitch_construct_cci(device);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }
    }
    else
    {
        LWSWITCH_PRINT(device, INFO, "CCI is not supported, skipping construct\n");
    }
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    _lwswitch_set_dma_mask(device);

    retval = _lwswitch_initialize_device_state(device);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "Failed to initialize device state: %d!\n",
            retval);
        goto lwswitch_initialize_device_state_fail;
    }

    device->hal.lwswitch_load_uuid(device);

    /*
     * Check module parameters for blacklisted device
     */
    if (lwswitch_os_is_uuid_in_blacklist(&device->uuid) == LW_TRUE)
    {
        LWSWITCH_PRINT(device, SETUP,
            "Blacklisted lwswitch at (%04x:%02x:%02x.%02x)\n",
            device->lwlink_device->pciInfo.domain,
            device->lwlink_device->pciInfo.bus,
            device->lwlink_device->pciInfo.device,
            device->lwlink_device->pciInfo.function);
        is_blacklisted_by_os = LW_TRUE;
        // initialization continues until we have updated InfoROM...
    }

    if (lwswitch_is_inforom_supported(device))
    {
        retval = lwswitch_initialize_inforom(device);
        if (LWL_SUCCESS != retval)
        {
            LWSWITCH_PRINT(device, ERROR,
                    "Failed to initialize InfoROM rc: %d\n",
                    retval);
            goto lwswitch_initialize_device_state_fail;
        }

        retval = lwswitch_initialize_inforom_objects(device);
        if (LWL_SUCCESS != retval)
        {
            LWSWITCH_PRINT(device, ERROR,
                        "Failed to initialize InfoROM objects! rc:%d\n",
                        retval);
            goto lwswitch_initialize_inforom_fail;
        }
    }
    else
    {
        LWSWITCH_PRINT(device, INFO,
                "InfoROM is not supported, skipping init\n");
    }

    (void)device->hal.lwswitch_read_oob_blacklist_state(device);
    (void)device->hal.lwswitch_write_fabric_state(device);

    lwswitch_task_create(device, &lwswitch_fabric_state_heartbeat,
                         LWSWITCH_HEARTBEAT_INTERVAL_NS,
                         LWSWITCH_TASK_TYPE_FLAGS_ALWAYS_RUN);

    if (device->device_blacklist_reason == LWSWITCH_DEVICE_BLACKLIST_REASON_MANUAL_OUT_OF_BAND)
    {
        LWSWITCH_PRINT(device, SETUP,
            "Blacklisted lwswitch at (%04x:%02x:%02x.%02x)\n",
            device->lwlink_device->pciInfo.domain,
            device->lwlink_device->pciInfo.bus,
            device->lwlink_device->pciInfo.device,
            device->lwlink_device->pciInfo.function);
        return retval;
    }

    if (is_blacklisted_by_os)
    {
        (void)lwswitch_lib_blacklist_device(device, LWSWITCH_DEVICE_BLACKLIST_REASON_MANUAL_IN_BAND);
        return retval;
    }

    for (link_num=0; link_num < lwswitch_get_num_links(device); link_num++)
    {
        if (!lwswitch_is_link_valid(device, link_num))
        {
            continue;
        }

        retval = lwswitch_create_link(device, link_num, &link);
        if (LWL_SUCCESS != retval)
        {
            LWSWITCH_PRINT(device, ERROR,
                "Failed to create link %d : %d!\n",
                link_num,
                retval);
            goto lwswitch_link_fail;
        }

        retval = lwlink_lib_register_link(device->lwlink_device, link);
        if (LWL_SUCCESS != retval)
        {
            LWSWITCH_PRINT(device, ERROR,
                "Failed to register link %d with the lwlink core : %d!\n",
                link_num,
                retval);

            // Free the single dangling link.
            lwswitch_destroy_link(link);

            goto lwswitch_link_fail;
        }

        lwswitch_reset_persistent_link_hw_state(device, link_num);
    }

    retval = lwswitch_set_training_mode(device);

    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to determine link training mode! rc: %d\n", retval);
        goto lwswitch_link_fail;
    }

    // Pull seeds from inforom after all links have been registered
    lwswitch_store_seed_data_from_inforom_to_corelib(device);

    // Initialize select scratch registers to 0x0
    device->hal.lwswitch_init_scratch(device);

    retval = lwswitch_construct_error_log(&device->log_FATAL_ERRORS, 1024, LW_FALSE);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to construct log_FATAL_ERRORS! rc: %d\n", retval);
        goto lwswitch_construct_error_log_fail;
    }

    retval = lwswitch_construct_error_log(&device->log_NONFATAL_ERRORS, 1024, LW_TRUE);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Failed to construct log_NONFATAL_ERRORS! rc: %d\n", retval);
        goto lwswitch_construct_error_log_fail;
    }

    if (device->regkeys.latency_counter == LW_SWITCH_REGKEY_LATENCY_COUNTER_LOGGING_ENABLE)
    {
        lwswitch_task_create(device, &lwswitch_internal_latency_bin_log,
            lwswitch_get_latency_sample_interval_msec(device) * LWSWITCH_INTERVAL_1MSEC_IN_NS * 9/10, 0);
    }

    lwswitch_task_create(device, &lwswitch_ecc_writeback_task,
        (60 * LWSWITCH_INTERVAL_1SEC_IN_NS), 0);

    if (IS_RTLSIM(device) || IS_EMULATION(device) || IS_FMODEL(device))
    {
        LWSWITCH_PRINT(device, WARN,
        "%s: Skipping setup of LwSwitch thermal alert monitoring\n",
            __FUNCTION__);
    }
    else
    {
        lwswitch_task_create(device, &lwswitch_monitor_thermal_alert,
            100*LWSWITCH_INTERVAL_1MSEC_IN_NS, 0);
    }

    device->lwlink_device->initialized = 1;

    return LWL_SUCCESS;

lwswitch_construct_error_log_fail:
    //free allocated memory to avoid leaking
    lwswitch_destroy_error_log(device, &device->log_FATAL_ERRORS);
    lwswitch_destroy_error_log(device, &device->log_NONFATAL_ERRORS);

lwswitch_link_fail:
    // Track down all links that successfully registered.
    for (link_num = 0; link_num < lwswitch_get_num_links(device); link_num++)
    {
        if (lwlink_lib_get_link(device->lwlink_device, link_num, &link) == LWL_SUCCESS)
        {
            lwlink_lib_unregister_link(link);
            lwswitch_destroy_link(link);
        }
    }

    lwswitch_destroy_inforom_objects(device);

lwswitch_initialize_inforom_fail:
    lwswitch_destroy_inforom(device);

lwswitch_initialize_device_state_fail:
    _lwswitch_destruct_soe(device);
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    _lwswitch_destruct_cci(device);
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    return retval;
}

LwBool
lwswitch_lib_validate_device_id
(
    LwU32 device_id
)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_SVNP01)
    if (lwswitch_is_sv10_device_id(device_id))
    {
        return LW_TRUE;
    }
#endif
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    if (lwswitch_is_lr10_device_id(device_id))
    {
        return LW_TRUE;
    }
#endif
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    if (lwswitch_is_ls10_device_id(device_id))
    {
        return LW_TRUE;
    }
#endif
    return LW_FALSE;
}

LwlStatus
lwswitch_lib_post_init_device
(
    lwswitch_device *device
)
{
    LwlStatus retval;

    if (!LWSWITCH_IS_DEVICE_INITIALIZED(device))
    {
        return -LWL_ERR_ILWALID_STATE;
    }

    retval = _lwswitch_post_init_device_setup(device);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }
    
    if (lwswitch_is_spi_supported(device))
    {
        retval = lwswitch_bios_get_image(device);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }

        retval = lwswitch_parse_bios_image(device);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Skipping BIOS parsing since SPI is unsupported.\n",
            __FUNCTION__);
    }

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    lwswitch_fetch_active_repeater_mask(device);

    if (lwswitch_is_cci_supported(device))
    {
        retval = cciLoad(device);
        if (LWL_SUCCESS != retval)
        {
            LWSWITCH_PRINT(device, ERROR, "%s: Init CCI failed\n",
                __FUNCTION__);
            return retval;
        }
    }
    else
    {
        LWSWITCH_PRINT(device, INFO, "%s: Skipping CCI init.\n",
            __FUNCTION__);
    }
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    retval = _lwswitch_setup_link_system_registers(device);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    lwswitch_smbpbi_post_init(device);

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    (void)lwswitch_launch_ALI(device);
#endif

    return LWL_SUCCESS;
}

void
lwswitch_lib_post_init_blacklist_device
(
    lwswitch_device *device
)
{
    _lwswitch_post_init_blacklist_device_setup(device);
}

/*!
 * @brief: Gets the client event associated with the file descriptor
 *         if it already exists in the Device's client event list.
 */
LwlStatus
lwswitch_lib_get_client_event
(
    lwswitch_device       *device,
    void                  *osPrivate,
    LWSWITCH_CLIENT_EVENT **ppClientEvent
)
{
    LWSWITCH_CLIENT_EVENT *lwrr = NULL;

    *ppClientEvent = NULL;

    if(!LWSWITCH_IS_DEVICE_VALID(device))
    {
        return -LWL_BAD_ARGS;
    }

    lwListForEachEntry(lwrr, &device->client_events_list, entry)
    {
        if (lwrr->private_driver_data == osPrivate)
        {
            *ppClientEvent = lwrr;
            return LWL_SUCCESS;
        }
    }

    return -LWL_NOT_FOUND;
}

/*!
 * @brief: Adds an event to the front of the
 *         Device's client event list.
 */
LwlStatus
lwswitch_lib_add_client_event
(
    lwswitch_device *device,
    void            *osPrivate,
    LwU32           eventId
)
{
    LWSWITCH_CLIENT_EVENT *newEvent;
    LwlStatus status = LWL_SUCCESS;

    if (!LWSWITCH_IS_DEVICE_VALID(device))
    {
        return -LWL_BAD_ARGS;
    }

    if (eventId >= LWSWITCH_DEVICE_EVENT_COUNT)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Invalid event Id.\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    // Ilwoke OS specific API to add event.
    status = lwswitch_os_add_client_event(device->os_handle,
                                          osPrivate,
                                          eventId);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed to add client event.\n", __FUNCTION__);
        return status;
    }

    newEvent = lwswitch_os_malloc(sizeof(*newEvent));
    if (newEvent == NULL)
    {
        return -LWL_NO_MEM;
    }

    newEvent->eventId             = eventId;
    newEvent->private_driver_data = osPrivate;

    lwListAdd(&newEvent->entry, &device->client_events_list);

    return LWL_SUCCESS;
}

/*!
 * @brief: Removes all events corresponding to osPrivate,
 *         from the Device's client event list.
 */
LwlStatus
lwswitch_lib_remove_client_events
(
    lwswitch_device *device,
    void            *osPrivate
)
{
    LWSWITCH_CLIENT_EVENT *lwrr = NULL;
    LWSWITCH_CLIENT_EVENT *next = NULL;
    LwlStatus status = LWL_SUCCESS;

    //
    // Device shutdown may happen before this is called, so return
    // if device is gone
    //
    if (!LWSWITCH_IS_DEVICE_VALID(device))
    {
        return LWL_SUCCESS;
    }

    lwListForEachEntry_safe(lwrr, next, &device->client_events_list, entry)
    {
        if (lwrr->private_driver_data == osPrivate)
        {
            lwListDel(&lwrr->entry);
            lwswitch_os_free(lwrr);
        }
    }

    // Ilwoke OS specific API to remove event.
    status = lwswitch_os_remove_client_event(device->os_handle,
                                             osPrivate);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed to remove client events.\n", __FUNCTION__);
        return status;
    }

    return LWL_SUCCESS;
}

/*!
 * @brief: Notifies all events with matching event id in the
 *         Device's client event list.
 */
LwlStatus
lwswitch_lib_notify_client_events
(
    lwswitch_device *device,
    LwU32            eventId
)
{
    LwlStatus status;
    LWSWITCH_CLIENT_EVENT *lwrr = NULL;

    if (!LWSWITCH_IS_DEVICE_VALID(device))
    {
        return -LWL_BAD_ARGS;
    }

    if (eventId >= LWSWITCH_DEVICE_EVENT_COUNT)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Invalid event Id.\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    lwListForEachEntry(lwrr, &device->client_events_list, entry)
    {
        if (lwrr->eventId == eventId)
        {
            // OS specific event notification.
            status = lwswitch_os_notify_client_event(device->os_handle,
                                                     lwrr->private_driver_data,
                                                     eventId);
            if (status != LWL_SUCCESS)
            {
                return status;
            }
        }
    }

    return LWL_SUCCESS;
}

/*!
   @brief: Release ROM image from memory.
*/
void
_lwswitch_destroy_rom(lwswitch_device *device)
{
    if (device->biosImage.pImage != NULL)
    {
        lwswitch_os_free(device->biosImage.pImage);
        device->biosImage.pImage = NULL;
    }
}

/*!
 * @brief: Free the device's client event list
 */
static void
_lwswitch_destroy_event_list(lwswitch_device *device)
{
    LWSWITCH_CLIENT_EVENT *lwrr = NULL;
    LWSWITCH_CLIENT_EVENT *next = NULL;

    lwListForEachEntry_safe(lwrr, next, &device->client_events_list, entry)
    {
        lwListDel(&lwrr->entry);
        lwswitch_os_free(lwrr);
    }
}

LwlStatus
lwswitch_lib_shutdown_device
(
    lwswitch_device *device
)
{

    if (!LWSWITCH_IS_DEVICE_ACCESSIBLE(device))
    {
        return -LWL_BAD_ARGS;
    }

    //
    // Set fabric state to offline
    //
    if (device->device_fabric_state != LWSWITCH_DEVICE_FABRIC_STATE_BLACKLISTED)
        device->device_fabric_state = LWSWITCH_DEVICE_FABRIC_STATE_OFFLINE;
    device->driver_fabric_state = LWSWITCH_DRIVER_FABRIC_STATE_OFFLINE;
    (void)device->hal.lwswitch_write_fabric_state(device);

    lwswitch_hw_counter_shutdown(device);

    _lwswitch_unregister_links(device);

    lwswitch_destroy_error_log(device, &device->log_FATAL_ERRORS);
    lwswitch_destroy_error_log(device, &device->log_NONFATAL_ERRORS);

    lwswitch_smbpbi_unload(device);
    _lwswitch_destroy_event_list(device);

    lwswitch_destroy_inforom_objects(device);
    lwswitch_destroy_inforom(device);

    lwswitch_smbpbi_destroy(device);

    lwswitch_destroy_device_state(device);

    _lwswitch_destroy_rom(device);

    _lwswitch_destruct_soe(device);

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    _lwswitch_destruct_cci(device);
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    lwswitch_tasks_destroy(device);

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_lib_get_log_count
(
    lwswitch_device *device,
    LwU32 *fatal, LwU32 *nonfatal
)
{
    if (!LWSWITCH_IS_DEVICE_INITIALIZED(device) ||
        fatal == NULL || nonfatal == NULL)
    {
        return -LWL_BAD_ARGS;
    }

    *fatal = device->log_FATAL_ERRORS.error_count;
    *nonfatal = device->log_NONFATAL_ERRORS.error_count;
    // No report of log_INFO lwrrently

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_lib_load_platform_info
(
    lwswitch_device *device
)
{
    if (!LWSWITCH_IS_DEVICE_ACCESSIBLE(device))
    {
        return -LWL_BAD_ARGS;
    }

    device->hal.lwswitch_determine_platform(device);

    return LWL_SUCCESS;
}

void
lwswitch_lib_get_device_info
(
    lwswitch_device *device,
    struct lwlink_pci_info **pciInfo
)
{
    if (!LWSWITCH_IS_DEVICE_VALID(device) || pciInfo == NULL)
    {
        LWSWITCH_ASSERT(0);
        return;
    }

    *pciInfo = &device->lwlink_device->pciInfo;
}

LwlStatus
lwswitch_lib_get_bios_version
(
    lwswitch_device *device,
    LwU64 *version
)
{
    LWSWITCH_GET_BIOS_INFO_PARAMS p = { 0 };
    LwlStatus ret;

    if (!device)
        return -LWL_BAD_ARGS;

    ret = device->hal.lwswitch_ctrl_get_bios_info(device, &p);
    *version = p.version;

    return ret;
}

LwlStatus
lwswitch_lib_use_pin_irq
(
     lwswitch_device *device
)
{
    return IS_FMODEL(device);
}


LwlStatus
lwswitch_lib_register_device
(
    LwU16 pci_domain,
    LwU8 pci_bus,
    LwU8 pci_device,
    LwU8 pci_func,
    LwU16 pci_device_id,
    void *os_handle,
    LwU32 os_instance,
    lwswitch_device **return_device
)
{
    lwswitch_device *device  = NULL;
    lwlink_device   *coreDev = NULL;
    LwlStatus        retval  = LWL_SUCCESS;

    if (!lwlink_lib_is_initialized())
    {
        LWSWITCH_PRINT(device, ERROR,
            "LWLink core lib isn't initialized yet!\n");
        return -LWL_INITIALIZATION_TOTAL_FAILURE;
    }

    if (return_device == NULL || os_handle == NULL)
    {
        return -LWL_BAD_ARGS;
    }

    *return_device = NULL;

    device = lwswitch_os_malloc(sizeof(*device));
    if (NULL == device)
    {
        LWSWITCH_PRINT(device, ERROR,
            "lwswitch_os_malloc during device creation failed!\n");
        return -LWL_NO_MEM;
    }
    lwswitch_os_memset(device, 0, sizeof(*device));

    lwswitch_os_snprintf(device->name, sizeof(device->name),
         LWSWITCH_DEVICE_NAME "%d", os_instance);

    coreDev = lwswitch_os_malloc(sizeof(*coreDev));
    if (NULL == coreDev)
    {
        LWSWITCH_PRINT(device, ERROR,
            "lwswitch_os_malloc during device creation failed!\n");

        retval = -LWL_NO_MEM;
        goto lwlink_lib_register_device_fail;
    }
    lwswitch_os_memset(coreDev, 0, sizeof(*coreDev));

    coreDev->driverName =
        lwswitch_os_malloc(sizeof(LWSWITCH_DRIVER_NAME));
    if (coreDev->driverName == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "lwswitch_os_malloc during device creation failed!\n");

        retval = -LWL_NO_MEM;
        goto lwlink_lib_register_device_fail;
    }
    lwswitch_os_memcpy(coreDev->driverName, LWSWITCH_DRIVER_NAME,
                       sizeof(LWSWITCH_DRIVER_NAME));

    device->os_handle   = os_handle;
    device->os_instance = os_instance;

    device->lwlink_device             = coreDev;
    device->lwlink_device->deviceName = device->name;
    device->lwlink_device->uuid = NULL; // No UUID support for switch

    device->lwlink_device->pciInfo.domain      = pci_domain;
    device->lwlink_device->pciInfo.bus         = pci_bus;
    device->lwlink_device->pciInfo.device      = pci_device;
    device->lwlink_device->pciInfo.function    = pci_func;
    device->lwlink_device->pciInfo.pciDeviceId = pci_device_id;

    // lwlink_device has a back pointer to lwswitch_device
    device->lwlink_device->pDevInfo = device;
    device->lwlink_device->type = LWLINK_DEVICE_TYPE_LWSWITCH;

    //
    // Initialize the Fabric State
    //
    device->fm_timeout = LWSWITCH_DEFAULT_FM_HEARTBEAT_TIMEOUT_MSEC;
    device->fabric_state_sequence_number = 0;
    device->driver_fabric_state = LWSWITCH_DRIVER_FABRIC_STATE_STANDBY;
    device->device_fabric_state = LWSWITCH_DEVICE_FABRIC_STATE_STANDBY;
    device->device_blacklist_reason = LWSWITCH_DEVICE_BLACKLIST_REASON_NONE;

    //
    // Initialize HAL connectivity as early as possible so that other lib
    // interfaces can work.
    //
    retval = _lwswitch_setup_hal(device, device->lwlink_device->pciInfo.pciDeviceId);
    if (retval != LWL_SUCCESS)
    {
        goto lwlink_lib_register_device_fail;
    }

    //
    // Initialize regkeys as early as possible so that most routines can take
    // advantage of them.
    //
    _lwswitch_init_device_regkeys(device);

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    // After regkeys have been set then only set the enableALI field.
    device->lwlink_device->enableALI = (device->regkeys.link_training_mode ==
                        LW_SWITCH_REGKEY_LINK_TRAINING_SELECT_ALI) ? LW_TRUE:LW_FALSE;
#endif

    retval = lwlink_lib_register_device(device->lwlink_device);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "lwlinklib register device failed!\n");
        goto lwlink_lib_register_device_fail;
    }

    *return_device = device;

    LWSWITCH_PRINT(device, SETUP,
        "Successfully registered with lwlinkcore\n");

    return retval;

lwlink_lib_register_device_fail:

    if (NULL != coreDev)
    {
        lwswitch_os_free(coreDev->driverName);
        lwswitch_os_free(coreDev);
    }

    if (NULL != device)
        lwswitch_os_free(device);

    return retval;
}

void
lwswitch_lib_unregister_device
(
    lwswitch_device *device
)
{
    if (!LWSWITCH_IS_DEVICE_VALID(device))
    {
        LWSWITCH_ASSERT(0);
        return;
    }

    lwlink_lib_unregister_device(device->lwlink_device);

    lwswitch_os_free(device->lwlink_device->driverName);
    lwswitch_os_free(device->lwlink_device);
    lwswitch_os_free(device);

    return;
}

/*!
 * @brief: Gets the mask of valid I2C ports on the
 *         Device.
 */
LwlStatus
lwswitch_lib_get_valid_ports_mask
(
    lwswitch_device *device,
    LwU32 *validPortsMask
)
{
    LwlStatus status;
    LWSWITCH_CTRL_I2C_GET_PORT_INFO_PARAMS port_info;
    LwU32 i;
    LwU32 ports_mask = 0;
    LwBool is_i2c_access_allowed;
    LwBool is_port_allowed;

    if (!LWSWITCH_IS_DEVICE_VALID(device) ||
        (validPortsMask == NULL))
    {
        return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(&port_info, 0, sizeof(port_info));

    status = lwswitch_ctrl_i2c_get_port_info(device, &port_info);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get I2C port info.\n",
            __FUNCTION__);
        return status;
    }

    is_i2c_access_allowed = (device->regkeys.i2c_access_control ==
                             LW_SWITCH_REGKEY_I2C_ACCESS_CONTROL_ENABLE) ?
                                LW_TRUE : LW_FALSE;

    for (i = 0; i < LWSWITCH_MAX_I2C_PORTS; i++)
    {
        is_port_allowed = is_i2c_access_allowed ? LW_TRUE :
                              FLD_TEST_DRF(_I2C, _PORTINFO, _ACCESS_ALLOWED, _TRUE,
                                           port_info.info[i]);

        if (is_port_allowed &&
            FLD_TEST_DRF(_I2C, _PORTINFO, _DEFINED, _PRESENT,
                         port_info.info[i]))
        {
            ports_mask |= LWBIT(i);
        }
    }

    *validPortsMask = ports_mask;
    return status;
}

/*!
 * @brief: Returns if the I2C transactions are supported.
 */
LwBool
lwswitch_lib_is_i2c_supported
(
    lwswitch_device *device
)
{
    if (!LWSWITCH_IS_DEVICE_VALID(device))
    {
        LWSWITCH_ASSERT(0);
        return LW_FALSE;
    }

    return lwswitch_is_i2c_supported(device);
}

static LwlStatus
_lwswitch_perform_i2c_transfer
(
    lwswitch_device *device,
    LwU32 client,
    LwU8 type,
    LwU16 addr,
    LwU8 port,
    LwU8 cmd,
    LwU32 msgLength,
    LwU8 *pData
)
{
    LwlStatus status;
    LwU16 deviceAddr;
    LwU32 speedMode;
    LwBool bIsRead = LW_FALSE;
    LwU32 flags = 0;
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS i2c_params = {0};
    LwBool is_i2c_access_allowed;

    if (!lwswitch_os_is_admin())
    {
        return -LWL_ERR_INSUFFICIENT_PERMISSIONS;
    }

    is_i2c_access_allowed = (device->regkeys.i2c_access_control ==
                             LW_SWITCH_REGKEY_I2C_ACCESS_CONTROL_ENABLE) ?
                                LW_TRUE : LW_FALSE;

    //
    // The address needs to be shifted by 1,
    // See LWSWITCH_CTRL_I2C_INDEXED_PARAMS
    //
    deviceAddr = addr << 1;
    speedMode  = device->pI2c->Ports[port].defaultSpeedMode;
    flags      = DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START, _SEND)              |
                 DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP, _SEND)               |
                 DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT)       |
                 DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW)               |
                 DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _BLOCK_PROTOCOL, _DISABLED) |
                 DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _TRANSACTION_MODE, _NORMAL);

    switch (speedMode)
    {
        case LWSWITCH_I2C_SPEED_MODE_1000KHZ:
        {
            flags = FLD_SET_DRF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _1000KHZ, flags);
            break;
        }
        case LWSWITCH_I2C_SPEED_MODE_400KHZ:
        {
            flags = FLD_SET_DRF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _400KHZ, flags);
            break;
        }
        case LWSWITCH_I2C_SPEED_MODE_300KHZ:
        {
            flags = FLD_SET_DRF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _300KHZ, flags);
            break;
        }
        case LWSWITCH_I2C_SPEED_MODE_200KHZ:
        {
            flags = FLD_SET_DRF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _200KHZ, flags);
            break;
        }
        case LWSWITCH_I2C_SPEED_MODE_100KHZ:
        {
            flags = FLD_SET_DRF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _100KHZ, flags);
            break;
        }
        default:
        {
            LWSWITCH_PRINT(device, ERROR, "Invalid I2C speed!\n");
            status = -LWL_BAD_ARGS;
            goto end;
        }
    }

    switch (type)
    {
        case LWSWITCH_I2C_CMD_READ:
            bIsRead = LW_TRUE;
            // Fall through
        case LWSWITCH_I2C_CMD_WRITE:
        {
            flags = FLD_SET_DRF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ZERO, flags);
            break;
        }
        case LWSWITCH_I2C_CMD_SMBUS_READ:
        {
            bIsRead = LW_TRUE;
            flags = FLD_SET_DRF(SWITCH_CTRL, _I2C_FLAGS, _RESTART, _SEND, flags);
            // Fall through
        }
        case LWSWITCH_I2C_CMD_SMBUS_WRITE:
        {
            flags = FLD_SET_DRF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE, flags);
            break;
        }
        case LWSWITCH_I2C_CMD_SMBUS_QUICK_READ:
            bIsRead = LW_TRUE;
            // Fall through
        case LWSWITCH_I2C_CMD_SMBUS_QUICK_WRITE:
        {
            flags = FLD_SET_DRF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ZERO, flags);
            msgLength = 0;
            break;
        }
        default:
        {
            LWSWITCH_PRINT(device, ERROR, "Invalid SMBUS protocol! Protocol not supported.\n");
            status = -LWL_BAD_ARGS;
            goto end;
        }
    }

    if (!is_i2c_access_allowed &&
        !lwswitch_i2c_is_device_access_allowed(device, port, deviceAddr, bIsRead))
    {
        return -LWL_BAD_ARGS;
    }

    if (msgLength > LWSWITCH_CTRL_I2C_MESSAGE_LENGTH_MAX)
    {
        LWSWITCH_PRINT(device, ERROR,
            "Length of buffer (0x%x bytes) provided larger than max (0x%x bytes)\n",
             msgLength, LWSWITCH_CTRL_I2C_MESSAGE_LENGTH_MAX);
        status = -LWL_BAD_ARGS;
        goto end;
    }

    if (bIsRead)
    {
        i2c_params.bIsRead = LW_TRUE;
    }
    else
    {
        flags = FLD_SET_DRF(SWITCH_CTRL, _I2C_FLAGS, _RESTART, _NONE, flags);
        lwswitch_os_memcpy(i2c_params.message, pData, msgLength);
    }

    if (FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE, flags))
    {
        i2c_params.index[0] = cmd;
    }

    i2c_params.port     = port;
    i2c_params.address  = deviceAddr;
    i2c_params.acquirer = client;
    i2c_params.flags    = flags;
    i2c_params.messageLength = msgLength;

    status = lwswitch_ctrl_i2c_indexed(device, &i2c_params);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "I2C transfer Failed!\n");
        goto end;
    }

    if (bIsRead)
    {
        lwswitch_os_memcpy(pData, i2c_params.message, msgLength);
    }

end:
    return status;
}

/*!
 * @brief: Performs an I2C transaction.
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
)
{
    LwlStatus status;
    LWSWITCH_CTRL_I2C_GET_PORT_INFO_PARAMS port_info;
    LwBool is_i2c_access_allowed;
    LwBool is_port_allowed;

    if (!LWSWITCH_IS_DEVICE_VALID(device))
    {
        LWSWITCH_ASSERT(0);
        return -LWL_ERR_ILWALID_STATE;
    }

    lwswitch_os_memset(&port_info, 0, sizeof(port_info));

    status = lwswitch_ctrl_i2c_get_port_info(device, &port_info);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_CHECK_STATUS(device, status);
        return status;
    }

    is_i2c_access_allowed = (device->regkeys.i2c_access_control ==
                             LW_SWITCH_REGKEY_I2C_ACCESS_CONTROL_ENABLE) ?
                                LW_TRUE : LW_FALSE;
    is_port_allowed = is_i2c_access_allowed ? LW_TRUE :
                          FLD_TEST_DRF(_I2C, _PORTINFO, _ACCESS_ALLOWED, _TRUE,
                                       port_info.info[port]);

    if (!is_port_allowed ||
        !FLD_TEST_DRF(_I2C, _PORTINFO, _DEFINED, _PRESENT,
                      port_info.info[port]))
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Invalid port access %d.\n",
            __FUNCTION__, port);
        return (-LWL_BAD_ARGS);
    }

    status = _lwswitch_perform_i2c_transfer(device, LWSWITCH_I2C_ACQUIRER_EXTERNAL,
                                            type, (LwU16)addr, port, command, len, pData);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "I2C transaction failed!\n");
        return status;
    }

    return LWL_SUCCESS;
}

void
lwswitch_timeout_create
(
    LwU64   timeout_ns,
    LWSWITCH_TIMEOUT *time
)
{
    LwU64   time_lwrrent;

    time_lwrrent = lwswitch_os_get_platform_time();
    time->timeout_ns = time_lwrrent + timeout_ns;
}

LwBool
lwswitch_timeout_check
(
    LWSWITCH_TIMEOUT *time
)
{
    LwU64   time_lwrrent;

    time_lwrrent = lwswitch_os_get_platform_time();
    return (time->timeout_ns <= time_lwrrent);
}

void
lwswitch_task_create
(
    lwswitch_device *device,
    void (*task_fn)(lwswitch_device *device),
    LwU64 period_nsec,
    LwU32 flags
)
{
    LWSWITCH_TASK_TYPE *task = lwswitch_os_malloc(sizeof(*task));

    if (task == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Unable to allocate task.\n",
            __FUNCTION__);
    }
    else
    {
        task->task_fn = task_fn;
        task->period_nsec = period_nsec;
        task->last_run_nsec = 0;
        task->flags = flags;
        task->next = device->tasks;
        device->tasks = task;
    }
}

void
lwswitch_tasks_destroy
(
    lwswitch_device *device
)
{
    LWSWITCH_TASK_TYPE *task = device->tasks;
    LWSWITCH_TASK_TYPE *next_task;

    device->tasks = NULL;

    while (task)
    {
        next_task = task->next;
        lwswitch_os_free(task);
        task = next_task;
    }
}

void
lwswitch_destroy_device_state
(
    lwswitch_device *device
)
{
    device->hal.lwswitch_destroy_device_state(device);
}

static LwlStatus
_lwswitch_ctrl_get_info
(
    lwswitch_device *device,
    LWSWITCH_GET_INFO *p
)
{
    return device->hal.lwswitch_ctrl_get_info(device, p);
}

static LwlStatus
_lwswitch_ctrl_get_lwlink_status
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_STATUS_PARAMS *ret
)
{
    return device->hal.lwswitch_ctrl_get_lwlink_status(device, ret);
}

static LwlStatus
_lwswitch_ctrl_get_counters
(
    lwswitch_device *device,
    LWSWITCH_LWLINK_GET_COUNTERS_PARAMS *ret
)
{
    return device->hal.lwswitch_ctrl_get_counters(device, ret);
}

LwlStatus
lwswitch_set_nport_port_config
(
    lwswitch_device *device,
    LWSWITCH_SET_SWITCH_PORT_CONFIG *p
)
{
    return device->hal.lwswitch_set_nport_port_config(device, p);
}

static LwlStatus
_lwswitch_ctrl_set_switch_port_config
(
    lwswitch_device *device,
    LWSWITCH_SET_SWITCH_PORT_CONFIG *p
)
{
    return device->hal.lwswitch_ctrl_set_switch_port_config(device, p);
}

static LwlStatus
_lwswitch_ctrl_get_ingress_request_table
(
    lwswitch_device *device,
    LWSWITCH_GET_INGRESS_REQUEST_TABLE_PARAMS *params
)
{
    return device->hal.lwswitch_ctrl_get_ingress_request_table(device, params);
}

static LwlStatus
_lwswitch_ctrl_set_ingress_request_table
(
    lwswitch_device *device,
    LWSWITCH_SET_INGRESS_REQUEST_TABLE *p
)
{
    return device->hal.lwswitch_ctrl_set_ingress_request_table(device, p);
}

static LwlStatus
_lwswitch_ctrl_set_ingress_request_valid
(
    lwswitch_device *device,
    LWSWITCH_SET_INGRESS_REQUEST_VALID *p
)
{
    return device->hal.lwswitch_ctrl_set_ingress_request_valid(device, p);
}

static LwlStatus
_lwswitch_ctrl_get_ingress_response_table
(
    lwswitch_device *device,
    LWSWITCH_GET_INGRESS_RESPONSE_TABLE_PARAMS *params
)
{
    return device->hal.lwswitch_ctrl_get_ingress_response_table(device, params);
}

static LwlStatus
_lwswitch_ctrl_set_ingress_response_table
(
    lwswitch_device *device,
    LWSWITCH_SET_INGRESS_RESPONSE_TABLE *p
)
{
    return device->hal.lwswitch_ctrl_set_ingress_response_table(device, p);
}

static LwlStatus
_lwswitch_ctrl_set_ganged_link_table
(
    lwswitch_device *device,
    LWSWITCH_SET_GANGED_LINK_TABLE *p
)
{
    return device->hal.lwswitch_ctrl_set_ganged_link_table(device, p);
}

static LwlStatus
_lwswitch_ctrl_set_remap_policy
(
    lwswitch_device *device,
    LWSWITCH_SET_REMAP_POLICY *p
)
{
    return device->hal.lwswitch_ctrl_set_remap_policy(device, p);
}

static LwlStatus
_lwswitch_ctrl_get_remap_policy
(
    lwswitch_device *device,
    LWSWITCH_GET_REMAP_POLICY_PARAMS *params
)
{
    return device->hal.lwswitch_ctrl_get_remap_policy(device, params);
}

static LwlStatus
_lwswitch_ctrl_set_remap_policy_valid
(
    lwswitch_device *device,
    LWSWITCH_SET_REMAP_POLICY_VALID *p
)
{
    return device->hal.lwswitch_ctrl_set_remap_policy_valid(device, p);
}

static LwlStatus
_lwswitch_ctrl_set_routing_id
(
    lwswitch_device *device,
    LWSWITCH_SET_ROUTING_ID *p
)
{
    return device->hal.lwswitch_ctrl_set_routing_id(device, p);
}

static LwlStatus
_lwswitch_ctrl_get_routing_id
(
    lwswitch_device *device,
    LWSWITCH_GET_ROUTING_ID_PARAMS *params
)
{
    return device->hal.lwswitch_ctrl_get_routing_id(device, params);
}

static LwlStatus
_lwswitch_ctrl_set_routing_id_valid
(
    lwswitch_device *device,
    LWSWITCH_SET_ROUTING_ID_VALID *p
)
{
    return device->hal.lwswitch_ctrl_set_routing_id_valid(device, p);
}

static LwlStatus
_lwswitch_ctrl_set_routing_lan
(
    lwswitch_device *device,
    LWSWITCH_SET_ROUTING_LAN *p
)
{
    return device->hal.lwswitch_ctrl_set_routing_lan(device, p);
}

static LwlStatus
_lwswitch_ctrl_get_routing_lan
(
    lwswitch_device *device,
    LWSWITCH_GET_ROUTING_LAN_PARAMS *params
)
{
    return device->hal.lwswitch_ctrl_get_routing_lan(device, params);
}

static LwlStatus
_lwswitch_ctrl_set_routing_lan_valid
(
    lwswitch_device *device,
    LWSWITCH_SET_ROUTING_LAN_VALID *p
)
{
    return device->hal.lwswitch_ctrl_set_routing_lan_valid(device, p);
}

static LwlStatus
_lwswitch_ctrl_get_internal_latency
(
    lwswitch_device *device,
    LWSWITCH_GET_INTERNAL_LATENCY *p
)
{
    return device->hal.lwswitch_ctrl_get_internal_latency(device, p);
}

static LwlStatus
_lwswitch_ctrl_get_lwlipt_counters
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLIPT_COUNTERS *p
)
{
    //
    // This control call is now deprecated.
    // New control call to fetch throughput counters is:
    // _lwswitch_ctrl_get_throughput_counters
    //
    return -LWL_ERR_NOT_SUPPORTED;
}

static LwlStatus
_lwswitch_ctrl_set_lwlipt_counter_config
(
    lwswitch_device *device,
    LWSWITCH_SET_LWLIPT_COUNTER_CONFIG *p
)
{
    //
    // This control call is now deprecated.
    // New control call to fetch throughput counters is:
    // _lwswitch_ctrl_get_throughput_counters_lr10
    //
    // Setting counter config is not allowed on these
    // non-configurable counters. These counters are
    // expected to be used by monitoring clients.
    //
    return -LWL_ERR_NOT_SUPPORTED;
}

static LwlStatus
_lwswitch_ctrl_get_lwlipt_counter_config
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLIPT_COUNTER_CONFIG *p
)
{
    //
    // This control call is now deprecated.
    // New control call to fetch throughput counters is:
    // _lwswitch_ctrl_get_throughput_counters_lr10
    //
    // Getting counter config is useful if counters are
    // configurable. These counters are not configurable
    // and are expected to be used by monitoring clients.
    //
    return -LWL_ERR_NOT_SUPPORTED;
}

static LwlStatus
_lwswitch_ctrl_register_read
(
    lwswitch_device *device,
    LWSWITCH_REGISTER_READ *p
)
{
    return device->hal.lwswitch_ctrl_register_read(device, p);
}

static LwlStatus
_lwswitch_ctrl_register_write
(
    lwswitch_device *device,
    LWSWITCH_REGISTER_WRITE *p
)
{
    return device->hal.lwswitch_ctrl_register_write(device, p);
}

LwlStatus
lwswitch_pex_get_counter
(
    lwswitch_device *device,
    LwU32   counterType,
    LwU32   *pCount
)
{
    return device->hal.lwswitch_pex_get_counter(device, counterType, pCount);
}

LwlStatus
lwswitch_ctrl_i2c_get_port_info
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_GET_PORT_INFO_PARAMS *pParams
)
{
    return device->hal.lwswitch_ctrl_i2c_get_port_info(device, pParams);
}

LwlStatus
lwswitch_ctrl_i2c_indexed
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams
)
{
    return device->hal.lwswitch_ctrl_i2c_indexed(device, pParams);
}

static LwlStatus
_lwswitch_ctrl_therm_read_temperature
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_TEMPERATURE_PARAMS *info
)
{
    return device->hal.lwswitch_ctrl_therm_read_temperature(device, info);
}

static LwlStatus
_lwswitch_ctrl_get_bios_info
(
    lwswitch_device *device,
    LWSWITCH_GET_BIOS_INFO_PARAMS *p
)
{
    return device->hal.lwswitch_ctrl_get_bios_info(device, p);
}

LwlStatus
lwswitch_ctrl_set_latency_bins
(
    lwswitch_device *device,
    LWSWITCH_SET_LATENCY_BINS *p
)
{
    return device->hal.lwswitch_ctrl_set_latency_bins(device, p);
}

static LwlStatus
_lwswitch_ctrl_get_ingress_reqlinkid
(
    lwswitch_device *device,
    LWSWITCH_GET_INGRESS_REQLINKID_PARAMS *params
)
{
    return device->hal.lwswitch_ctrl_get_ingress_reqlinkid(device, params);
}

static LwlStatus
_lwswitch_ctrl_get_throughput_counters
(
    lwswitch_device *device,
    LWSWITCH_GET_THROUGHPUT_COUNTERS_PARAMS *p
)
{
    return device->hal.lwswitch_ctrl_get_throughput_counters(device, p);
}

static LwlStatus
_lwswitch_ctrl_unregister_link
(
    lwswitch_device *device,
    LWSWITCH_UNREGISTER_LINK_PARAMS *params
)
{
    lwlink_link *link = lwswitch_get_link(device, (LwU8)params->portNum);

    if (link == NULL)
    {
        return -LWL_BAD_ARGS;
    }

    if (device->hal.lwswitch_is_link_in_use(device, params->portNum))
    {
        return -LWL_ERR_STATE_IN_USE;
    }

    lwlink_lib_unregister_link(link);
    lwswitch_destroy_link(link);

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_ctrl_acquire_capability
(
    lwswitch_device *device,
    LWSWITCH_ACQUIRE_CAPABILITY_PARAMS *params,
    void *osPrivate
)
{
    return lwswitch_os_acquire_fabric_mgmt_cap(osPrivate,
                                               params->capDescriptor);
}

static LwlStatus
_lwswitch_ctrl_reset_and_drain_links
(
    lwswitch_device *device,
    LWSWITCH_RESET_AND_DRAIN_LINKS_PARAMS *params
)
{
    return device->hal.lwswitch_reset_and_drain_links(device, params->linkMask);
}

static LwlStatus
_lwswitch_ctrl_get_fom_values
(
    lwswitch_device *device,
    LWSWITCH_GET_FOM_VALUES_PARAMS *ret
)
{
    return device->hal.lwswitch_ctrl_get_fom_values(device, ret);
}

static LwlStatus
_lwswitch_ctrl_get_lwlink_ecc_errors
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_ECC_ERRORS_PARAMS *params
)
{
    return device->hal.lwswitch_get_lwlink_ecc_errors(device, params);
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
static LwlStatus
_lwswitch_ctrl_set_mc_rid_table
(
    lwswitch_device *device,
    LWSWITCH_SET_MC_RID_TABLE_PARAMS *p
)
{
    return device->hal.lwswitch_ctrl_set_mc_rid_table(device, p);
}

static LwlStatus
_lwswitch_ctrl_get_mc_rid_table
(
    lwswitch_device *device,
    LWSWITCH_GET_MC_RID_TABLE_PARAMS *p
)
{
    return device->hal.lwswitch_ctrl_get_mc_rid_table(device, p);
}
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

static LwlStatus
_lwswitch_ctrl_set_residency_bins
(
    lwswitch_device *device,
    LWSWITCH_SET_RESIDENCY_BINS *p
)
{
    return device->hal.lwswitch_ctrl_set_residency_bins(device, p);
}

static LwlStatus
_lwswitch_ctrl_get_residency_bins
(
    lwswitch_device *device,
    LWSWITCH_GET_RESIDENCY_BINS *p
)
{
    return device->hal.lwswitch_ctrl_get_residency_bins(device, p);
}

static LwlStatus
_lwswitch_ctrl_get_rb_stall_busy
(
    lwswitch_device *device,
    LWSWITCH_GET_RB_STALL_BUSY *p
)
{
    return device->hal.lwswitch_ctrl_get_rb_stall_busy(device, p);
}

static LwlStatus
_lwswitch_ctrl_inband_send_data
(
    lwswitch_device *device,
    LWSWITCH_INBAND_SEND_DATA_PARAMS *p
)
{
    return device->hal.lwswitch_ctrl_inband_send_data(device, p);
}

static LwlStatus
_lwswitch_ctrl_inband_read_data
(
    lwswitch_device *device,
    LWSWITCH_INBAND_READ_DATA_PARAMS *p
)
{
    return device->hal.lwswitch_ctrl_inband_read_data(device, p);
}

static LwlStatus
_lwswitch_ctrl_inband_flush_data
(
    lwswitch_device *device,
    LWSWITCH_INBAND_FLUSH_DATA_PARAMS *p
)
{
    return device->hal.lwswitch_ctrl_inband_flush_data(device, p);
}

static LwlStatus
_lwswitch_ctrl_inband_pending_data_stats
(
    lwswitch_device *device,
    LWSWITCH_INBAND_PENDING_DATA_STATS_PARAMS *p
)
{
    return device->hal.lwswitch_ctrl_inband_pending_data_stats(device, p);
}

static LwlStatus
_lwswitch_ctrl_i2c_smbus_command
(
    lwswitch_device *device,
    LWSWITCH_I2C_SMBUS_COMMAND_PARAMS *pParams
)
{
    LwlStatus status;
    LWSWITCH_CTRL_I2C_GET_PORT_INFO_PARAMS port_info;
    LwU32 port = pParams->port;
    LwU8 msgLen;
    LwU8 cmd;
    LwU16 addr;
    LwU8 cmdType;
    LwU8 *pData;
    LwBool is_i2c_access_allowed;
    LwBool is_port_allowed;

    lwswitch_os_memset(&port_info, 0, sizeof(port_info));

    status = lwswitch_ctrl_i2c_get_port_info(device, &port_info);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get I2C port info.\n",
            __FUNCTION__);
        return status;
    }

    is_i2c_access_allowed = (device->regkeys.i2c_access_control ==
                             LW_SWITCH_REGKEY_I2C_ACCESS_CONTROL_ENABLE) ?
                                LW_TRUE : LW_FALSE;
    is_port_allowed = is_i2c_access_allowed ? LW_TRUE :
                          FLD_TEST_DRF(_I2C, _PORTINFO, _ACCESS_ALLOWED, _TRUE,
                                       port_info.info[port]);

    if (!is_port_allowed ||
        !FLD_TEST_DRF(_I2C, _PORTINFO, _DEFINED, _PRESENT,
                      port_info.info[port]))
    {
        LWSWITCH_PRINT(device, ERROR, "Invalid port access %d.\n", port);
        return LWL_BAD_ARGS;
    }

    addr = pParams->deviceAddr;

    switch (pParams->cmdType)
    {
        case LWSWITCH_I2C_SMBUS_CMD_QUICK:
        {
            cmd = 0;
            msgLen = 0;
            cmdType = pParams->bRead ?
                          LWSWITCH_I2C_CMD_SMBUS_QUICK_READ :
                          LWSWITCH_I2C_CMD_SMBUS_QUICK_WRITE;
            pData = NULL;
            break;
        }
        case LWSWITCH_I2C_SMBUS_CMD_BYTE:
        {
            cmd = 0;
            msgLen = 1;
            cmdType = pParams->bRead ?
                          LWSWITCH_I2C_CMD_READ :
                          LWSWITCH_I2C_CMD_WRITE;
            pData = (LwU8 *)&pParams->transactionData.smbusByte.message;
            break;
        }
        case LWSWITCH_I2C_SMBUS_CMD_BYTE_DATA:
        {
            msgLen = 1;
            cmd = pParams->transactionData.smbusByteData.cmd;
            cmdType = pParams->bRead ?
                          LWSWITCH_I2C_CMD_SMBUS_READ :
                          LWSWITCH_I2C_CMD_SMBUS_WRITE;
            pData = (LwU8 *)&pParams->transactionData.smbusByteData.message;
            break;
        }
        case LWSWITCH_I2C_SMBUS_CMD_WORD_DATA:
        {
            msgLen = 2;
            cmd = pParams->transactionData.smbusWordData.cmd;
            cmdType = pParams->bRead ?
                          LWSWITCH_I2C_CMD_SMBUS_READ :
                          LWSWITCH_I2C_CMD_SMBUS_WRITE;
            pData = (LwU8 *)&pParams->transactionData.smbusWordData.message;
            break;
        }
        default:
        {
            LWSWITCH_PRINT(device, ERROR, "Invalid Smbus command: %d.\n", port);
            return LWL_BAD_ARGS;
        }
    }

    return _lwswitch_perform_i2c_transfer(device, LWSWITCH_I2C_ACQUIRER_IOCTL,
                                          cmdType, addr, port, cmd, msgLen, pData);
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
static LwlStatus
_lwswitch_ctrl_cci_cmis_presence
(
    lwswitch_device *device,
    LWSWITCH_CCI_CMIS_PRESENCE_PARAMS *pParams
)
{
    lwswitch_os_memset(pParams, 0, sizeof(LWSWITCH_CCI_CMIS_PRESENCE_PARAMS));
    if (device->pCci != NULL)
    {
        (void)cciGetXcvrMask(device, &pParams->cagesMask, &pParams->modulesMask);
    }

    // IOCTL will always succeed
    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_ctrl_cci_lwlink_mappings
(
    lwswitch_device *device,
    LWSWITCH_CCI_CMIS_LWLINK_MAPPING_PARAMS *pParams
)
{
    if (device->pCci == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return cciGetCageMapping(device, pParams->cageIndex, &pParams->linkMask, &pParams->encodedValue);
}

static LwlStatus
_lwswitch_ctrl_cci_memory_access_read
(
    lwswitch_device *device,
    LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ_PARAMS *pParams
)
{
    if (device->pCci == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return cciCmisRead(device, pParams->cageIndex, pParams->bank,
                       pParams->page, pParams->address, pParams->count,
                       pParams->data);
}

static LwlStatus
_lwswitch_ctrl_cci_memory_access_write
(
    lwswitch_device *device,
    LWSWITCH_CCI_CMIS_MEMORY_ACCESS_WRITE_PARAMS *pParams
)
{
    if (device->pCci == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return cciCmisWrite(device, pParams->cageIndex, pParams->bank,
                        pParams->page, pParams->address, pParams->count,
                        pParams->data);
}

static LwlStatus
_lwswitch_ctrl_cci_cage_bezel_marking
(
    lwswitch_device *device,
    LWSWITCH_CCI_CMIS_CAGE_BEZEL_MARKING_PARAMS *pParams
)
{
    if (device->pCci == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return cciCmisCageBezelMarking(device, pParams->cageIndex, pParams->bezelMarking);
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

static LwlStatus
_lwswitch_ctrl_get_inforom_lwlink_max_correctable_error_rate
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_MAX_CORRECTABLE_ERROR_RATES_PARAMS *params
)
{
    return lwswitch_inforom_lwlink_get_max_correctable_error_rate(device, params);
}

static LwlStatus
_lwswitch_ctrl_get_inforom_lwlink_errors
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_ERROR_COUNTS_PARAMS *params
)
{
    return lwswitch_inforom_lwlink_get_errors(device, params);
}

static LwlStatus
_lwswitch_ctrl_get_inforom_ecc_errors
(
    lwswitch_device *device,
    LWSWITCH_GET_ECC_ERROR_COUNTS_PARAMS *params
)
{
    return lwswitch_inforom_ecc_get_errors(device, params);
}

static LwlStatus
_lwswitch_ctrl_get_inforom_bbx_sxid
(
    lwswitch_device *device,
    LWSWITCH_GET_SXIDS_PARAMS *params
)
{
    return lwswitch_inforom_bbx_get_sxid(device, params);
}

static LwlStatus
_lwswitch_ctrl_get_lwlink_lp_counters
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_LP_COUNTERS_PARAMS *params
)
{
    return device->hal.lwswitch_ctrl_get_lwlink_lp_counters(device, params);
}

static LwlStatus
_lwswitch_lib_ctrl_test_only
(
    lwswitch_device *device,
    LwU32 cmd,
    void *params,
    LwU64 size
)
{
    LwlStatus retval;

    switch (cmd)
    {
        LWSWITCH_DEV_CMD_DISPATCH_TEST(CTRL_LWSWITCH_REGISTER_READ,
                _lwswitch_ctrl_register_read,
                LWSWITCH_REGISTER_READ);
        LWSWITCH_DEV_CMD_DISPATCH_TEST(CTRL_LWSWITCH_REGISTER_WRITE,
                _lwswitch_ctrl_register_write,
                LWSWITCH_REGISTER_WRITE);
        default:
            lwswitch_os_print(LWSWITCH_DBG_LEVEL_INFO, "unknown ioctl %x\n", cmd);
            retval = -LWL_BAD_ARGS;
            break;
    }

    return retval;
}

static LwlStatus
_lwswitch_lib_validate_privileged_ctrl
(
    void *osPrivate,
    LwU64 flags
)
{
    if (flags & LWSWITCH_DEV_CMD_CHECK_ADMIN)
    {
        if (lwswitch_os_is_admin())
        {
            return LWL_SUCCESS;
        }
    }

    if (flags & LWSWITCH_DEV_CMD_CHECK_FM)
    {
        if (lwswitch_os_is_fabric_manager(osPrivate))
        {
            return LWL_SUCCESS;
        }
    }

    return -LWL_ERR_INSUFFICIENT_PERMISSIONS;
}

/*
 * @Brief : Constructs an LWS link struct with the given data
 *
 * @Description :
 *
 * @param[in] device            LwSwitch device to contain this link
 * @param[in] link_num          link number of the link
 * @param[out] link             reference to store the created link into
 *
 * @returns                     LWL_SUCCESS if action succeeded,
 *                              -LWL_NO_MEM if memory allocation failed
 */
LwlStatus
lwswitch_create_link
(
    lwswitch_device *device,
    LwU32 link_number,
    lwlink_link **link
)
{
    LwlStatus   retval      = LWL_SUCCESS;
    lwlink_link *ret        = NULL;
    LINK_INFO   *link_info  = NULL;
    LwU64       ac_coupled_mask;

    LWSWITCH_ASSERT(lwswitch_get_num_links(device) <=  LWSWITCH_MAX_NUM_LINKS);

    ret = lwswitch_os_malloc(sizeof(*ret));
    if (NULL == ret)
    {
        LWSWITCH_PRINT(device, ERROR,
            "lwswitch_os_malloc during link creation failed!\n");
        retval = -LWL_NO_MEM;
        goto lwswitch_create_link_cleanup;
    }
    lwswitch_os_memset(ret, 0, sizeof(*ret));

    link_info = lwswitch_os_malloc(sizeof(*link_info));
    if (NULL == link_info)
    {
        LWSWITCH_PRINT(device, ERROR,
            "lwswitch_os_malloc during link creation failed!\n");
        retval = -LWL_NO_MEM;
        goto lwswitch_create_link_cleanup;
    }
    lwswitch_os_memset(link_info, 0, sizeof(*link_info));
    lwswitch_os_snprintf(link_info->name, sizeof(link_info->name), LWSWITCH_LINK_NAME "%d", link_number);

    ret->dev        = device->lwlink_device;
    ret->linkName   = link_info->name;
    ret->linkNumber = link_number;
    ret->state      = LWLINK_LINKSTATE_OFF;
    ret->ac_coupled = LW_FALSE;
    ret->version    = lwswitch_get_link_ip_version(device, link_number);

    ac_coupled_mask = ((LwU64)device->regkeys.ac_coupled_mask2 << 32 |
                       (LwU64)device->regkeys.ac_coupled_mask);

    if (ac_coupled_mask)
    {
        if (ac_coupled_mask & LWBIT64(link_number))
        {
            ret->ac_coupled = LW_TRUE;
        }
    }
    else if (device->firmware.lwlink.link_config_found)
    {
        if (device->firmware.lwlink.link_ac_coupled_mask & LWBIT64(link_number))
        {
            ret->ac_coupled = LW_TRUE;
        }
    }

    // Initialize LWLink corelib callbacks for switch
    lwswitch_get_link_handlers(&link_handlers);

    ret->link_handlers = &link_handlers;

    //
    // link_info is used to store private link information
    //

    ret->link_info = link_info;

    *link = ret;

    return retval;

lwswitch_create_link_cleanup:
    if (NULL != ret)
    {
        lwswitch_os_free(ret);
    }
    if (NULL != link_info)
    {
        lwswitch_os_free(link_info);
    }

    return retval;
}

void
lwswitch_destroy_link
(
    lwlink_link *link
)
{
    if (NULL != link->link_info)
    {
        lwswitch_os_free(link->link_info);
    }

    lwswitch_os_free(link);
}

LwU32
lwswitch_get_num_links
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_get_num_links(device);
}

LwBool
lwswitch_is_link_valid
(
    lwswitch_device *device,
    LwU32            link_id
)
{
    return device->hal.lwswitch_is_link_valid(device, link_id);
}

lwlink_link*
lwswitch_get_link(lwswitch_device *device, LwU8 link_id)
{
    lwlink_link *link = NULL;

    lwlink_lib_get_link(device->lwlink_device, link_id, &link);

    return link;
}

LwU64
lwswitch_get_enabled_link_mask
(
    lwswitch_device *device
)
{
    LwU64                    ret;
    lwlink_link             *link;
    LwU32 link_num;

    ret = 0x0;

    for (link_num = 0; link_num < lwswitch_get_num_links(device); link_num++)
    {
        if (lwlink_lib_get_link(device->lwlink_device, link_num, &link) == LWL_SUCCESS)
        {
            ret |= LWBIT64(link_num);
        }
    }

    return ret;
}

void
lwswitch_set_fatal_error
(
    lwswitch_device *device,
    LwBool           device_fatal,
    LwU32            link_id
)
{
    device->hal.lwswitch_set_fatal_error(device, device_fatal, link_id);
}

LwU32
lwswitch_get_swap_clk_default
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_get_swap_clk_default(device);
}

LwU32
lwswitch_get_latency_sample_interval_msec
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_get_latency_sample_interval_msec(device);
}

void
lwswitch_internal_latency_bin_log
(
    lwswitch_device *device
)
{
    device->hal.lwswitch_internal_latency_bin_log(device);
}

void
lwswitch_ecc_writeback_task
(
    lwswitch_device *device
)
{
    device->hal.lwswitch_ecc_writeback_task(device);
}

void
lwswitch_monitor_thermal_alert
(
    lwswitch_device *device
)
{
    device->hal.lwswitch_monitor_thermal_alert(device);
}

void
lwswitch_hw_counter_shutdown
(
    lwswitch_device *device
)
{
    device->hal.lwswitch_hw_counter_shutdown(device);
}

LwlStatus
lwswitch_get_rom_info
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom
)
{
    return device->hal.lwswitch_get_rom_info(device, eeprom);
}

void
lwswitch_lib_enable_interrupts
(
    lwswitch_device *device
)
{
    if (!LWSWITCH_IS_DEVICE_ACCESSIBLE(device))
    {
        LWSWITCH_ASSERT(0);
        return;
    }

    device->hal.lwswitch_lib_enable_interrupts(device);
}

void
lwswitch_lib_disable_interrupts
(
    lwswitch_device *device
)
{
    if (!LWSWITCH_IS_DEVICE_ACCESSIBLE(device))
    {
        LWSWITCH_ASSERT(0);
        return;
    }

    device->hal.lwswitch_lib_disable_interrupts(device);
}

LwlStatus
lwswitch_lib_check_interrupts
(
    lwswitch_device *device
)
{
    if (!LWSWITCH_IS_DEVICE_INITIALIZED(device))
    {
        return -LWL_BAD_ARGS;
    }

    return device->hal.lwswitch_lib_check_interrupts(device);
}

LwlStatus
lwswitch_lib_service_interrupts
(
    lwswitch_device *device
)
{
    if (!LWSWITCH_IS_DEVICE_INITIALIZED(device))
    {
        return -LWL_BAD_ARGS;
    }

    return device->hal.lwswitch_lib_service_interrupts(device);
}

LwU64
lwswitch_hw_counter_read_counter
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_hw_counter_read_counter(device);
}

LwU32
lwswitch_get_link_ip_version
(
    lwswitch_device *device,
    LwU32            link_id
)
{
    return device->hal.lwswitch_get_link_ip_version(device, link_id);
}

LwU32
lwswitch_reg_read_32
(
    lwswitch_device *device,
    LwU32 offset
)
{
    LwU32 val;

    if (device->lwlink_device->pciInfo.bars[0].pBar == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "register read failed at offset 0x%x\n", offset);

        return 0xFFFFFFFF;
    }

    val = lwswitch_os_mem_read32((LwU8 *)device->lwlink_device->pciInfo.bars[0].pBar + offset);

    if ((val & 0xFFFF0000) == 0xBADF0000)
    {
        LwU32 boot_0;
        LWSWITCH_PRINT(device, WARN,
            "Potential IO failure reading 0x%x (0x%x)\n", offset, val);
        boot_0 = lwswitch_os_mem_read32((LwU8 *)device->lwlink_device->pciInfo.bars[0].pBar + 0x0);

        if ((boot_0 & 0xFFFF0000) == 0xBADF0000)
        {
            LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_IO_FAILURE,
                "IO failure\n");
            LWSWITCH_PRINT(device, ERROR,
                "IO failure reading 0x%x (0x%x)\n", offset, val);
        }
    }

#ifdef _VERBOSE_REG_ACCESS
    LWSWITCH_PRINT(device, SETUP,
        "LWSWITCH read 0x%6x+%6x = 0x%08x\n",
        device->lwlink_device->pciInfo.bars[0].baseAddr, offset, val);
#endif

    return val;
}

void
lwswitch_reg_write_32
(
    lwswitch_device *device,
    LwU32 offset,
    LwU32 data
)
{
    if (device->lwlink_device->pciInfo.bars[0].pBar == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "register write failed at offset 0x%x\n", offset);

        return;
    }

#ifdef _VERBOSE_REG_ACCESS
    LWSWITCH_PRINT(device, SETUP,
        "LWSWITCH write 0x%6x+%6x = 0x%08x\n",
        device->lwlink_device->pciInfo.bars[0].baseAddr, offset, data);
#endif

    // Write the register
    lwswitch_os_mem_write32((LwU8 *)device->lwlink_device->pciInfo.bars[0].pBar + offset, data);

    return;
}

LwU64
lwswitch_read_64bit_counter
(
    lwswitch_device *device,
    LwU32 lo_offset,
    LwU32 hi_offset
)
{
    LwU32   hi0;
    LwU32   hi1;
    LwU32   lo;

    hi0 = lwswitch_reg_read_32(device, hi_offset);
    do
    {
        hi1 = hi0;
        lo  = lwswitch_reg_read_32(device, lo_offset);
        hi0 = lwswitch_reg_read_32(device, hi_offset);
    } while (hi0 != hi1);

    return (lo | ((LwU64)hi0 << 32));
}

LwlStatus
lwswitch_validate_pll_config
(
    lwswitch_device *device,
    LWSWITCH_PLL_INFO *switch_pll,
    LWSWITCH_PLL_LIMITS default_pll_limits
)
{
    LwU32 update_rate_khz;
    LwU32 vco_freq_khz;
    LWSWITCH_PLL_LIMITS pll_limits;

    LWSWITCH_PRINT(device, SETUP,
        "%s: Validating PLL: %dkHz * %d / (%d * %d * (1 << %d))\n",
        __FUNCTION__,
        switch_pll->src_freq_khz,
        switch_pll->N,
        switch_pll->M,
        switch_pll->PL,
        switch_pll->dist_mode);

    //
    // These parameters could come from schmoo'ing API, settings file or a ROM.
    // For now, hard code with POR.
    //
    if (device->firmware.firmware_size > 0 &&
        device->firmware.clocks.clocks_found &&
        device->firmware.clocks.sys_pll.valid)
    {
        pll_limits = device->firmware.clocks.sys_pll;
    }
    else
    {
        pll_limits = default_pll_limits;
    }

    LWSWITCH_ASSERT(switch_pll->M != 0);
    LWSWITCH_ASSERT(switch_pll->PL != 0);

    if ((switch_pll->src_freq_khz < pll_limits.ref_min_mhz * 1000) ||
        (switch_pll->src_freq_khz > pll_limits.ref_max_mhz * 1000))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ERROR: Ref(%d) out-of-range\n",
            __FUNCTION__,
            switch_pll->src_freq_khz);
        return -LWL_ERR_ILWALID_STATE;
    }

    if ((switch_pll->M < pll_limits.m_min) ||
        (switch_pll->M > pll_limits.m_max))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ERROR: M(%d) out-of-range\n",
            __FUNCTION__,
            switch_pll->M);
        return -LWL_ERR_ILWALID_STATE;
    }

    if ((switch_pll->N < pll_limits.n_min) ||
        (switch_pll->N > pll_limits.n_max))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ERROR: N(%d) out-of-range\n",
            __FUNCTION__,
            switch_pll->N);
        return -LWL_ERR_ILWALID_STATE;
    }

    if ((switch_pll->PL < pll_limits.pl_min) ||
        (switch_pll->PL > pll_limits.pl_max))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ERROR: PL(%d) out-of-range\n",
            __FUNCTION__,
            switch_pll->PL);
        return -LWL_ERR_ILWALID_STATE;
    }

    vco_freq_khz = switch_pll->src_freq_khz * switch_pll->N
        / switch_pll->M;
    if ((vco_freq_khz < pll_limits.vco_min_mhz * 1000) ||
        (vco_freq_khz > pll_limits.vco_max_mhz * 1000))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ERROR: VCO(%d) freq out-of-range\n",
            __FUNCTION__,
            vco_freq_khz);
        return -LWL_ERR_ILWALID_STATE;
    }

    update_rate_khz = switch_pll->src_freq_khz / switch_pll->M;
    if ((update_rate_khz < pll_limits.update_min_mhz * 1000) ||
        (update_rate_khz > pll_limits.update_max_mhz * 1000))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ERROR: update rate(%d) out-of-range\n",
            __FUNCTION__,
            update_rate_khz);
        return -LWL_ERR_ILWALID_STATE;
    }

    switch_pll->vco_freq_khz = vco_freq_khz;

    switch_pll->freq_khz =
        switch_pll->src_freq_khz * switch_pll->N /
        (switch_pll->M * switch_pll->PL * (1 << switch_pll->dist_mode));

    LWSWITCH_PRINT(device, SETUP,
        "%s: Validated PLL: %dkHz * %d / (%d * %d * (1 << %d)) = %dkHz\n",
        __FUNCTION__,
        switch_pll->src_freq_khz,
        switch_pll->N,
        switch_pll->M,
        switch_pll->PL,
        switch_pll->dist_mode,
        switch_pll->freq_khz);

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_init_pll_config
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_init_pll_config(device);
}

LwlStatus
lwswitch_init_pll
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_init_pll(device);
}

void
lwswitch_init_clock_gating
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_init_clock_gating(device);
}

void
lwswitch_lib_get_uuid
(
    lwswitch_device *device,
    LwUuid *uuid
)
{
    if (!LWSWITCH_IS_DEVICE_INITIALIZED(device) || (uuid == NULL))
    {
        return;
    }

    lwswitch_os_memcpy(uuid, &device->uuid, sizeof(device->uuid));
}

LwlStatus
lwswitch_lib_get_physid
(
    lwswitch_device *device,
    LwU32 *phys_id
)
{
    LWSWITCH_GET_INFO get_info;
    LwlStatus ret;

    if (phys_id == NULL || !LWSWITCH_IS_DEVICE_ACCESSIBLE(device))
    {
        return -LWL_BAD_ARGS;
    }

    get_info.count=1;
    get_info.index[0] = LWSWITCH_GET_INFO_INDEX_PHYSICAL_ID;

    ret = _lwswitch_ctrl_get_info(device, &get_info);
    if (ret != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "Failed to get physical ID\n");
        return ret;
    }

    *phys_id = get_info.info[0];

    return LWL_SUCCESS;
}

void
lwswitch_i2c_set_hw_speed_mode
(
    lwswitch_device *device,
    LwU32 port,
    LwU32 speedMode
)
{
    device->hal.lwswitch_i2c_set_hw_speed_mode(device, port, speedMode);
    return;
}

void
lwswitch_lib_smbpbi_log_sxid
(
    lwswitch_device *device,
    LwU32           sxid,
    const char      *pFormat,
    ...
)
{
    va_list arglist;
    int     msglen;
    char    string[80];

    va_start(arglist, pFormat);
    msglen = lwswitch_os_vsnprintf(string, sizeof(string), pFormat, arglist);
    va_end(arglist);

    if (!(msglen < 0))
    {
        msglen = LW_MIN(msglen + 1, (int) sizeof(string));
        lwswitch_smbpbi_log_message(device, sxid, msglen, (LwU8 *) string);
    }
}

LwlStatus
lwswitch_set_minion_initialized
(
    lwswitch_device *device,
    LwU32 idx_minion,
    LwBool initialized
)
{
    return device->hal.lwswitch_set_minion_initialized(device, idx_minion, initialized);
}

LwBool
lwswitch_is_minion_initialized
(
    lwswitch_device *device,
    LwU32 idx_minion
)
{
    return device->hal.lwswitch_is_minion_initialized(device, idx_minion);
}

LwlStatus
lwswitch_device_discovery
(
    lwswitch_device *device,
    LwU32 discovery_offset
)
{
    return device->hal.lwswitch_device_discovery(device, discovery_offset);
}

void
lwswitch_filter_discovery
(
    lwswitch_device *device
)
{
    device->hal.lwswitch_filter_discovery(device);
}

LwlStatus
lwswitch_process_discovery
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_process_discovery(device);
}

LwlStatus
lwswitch_init_minion
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_init_minion(device);
}

LwU32
lwswitch_get_link_eng_inst
(
    lwswitch_device *device,
    LwU32 link_id,
    LWSWITCH_ENGINE_ID eng_id
)
{
    return device->hal.lwswitch_get_link_eng_inst(device, link_id, eng_id);
}

void *
lwswitch_alloc_chipdevice
(
    lwswitch_device *device
)
{
    return(device->hal.lwswitch_alloc_chipdevice(device));
}

void
lwswitch_free_chipdevice
(
    lwswitch_device *device
)
{
    if (device->chip_device)
    {
        lwswitch_os_free(device->chip_device);
        device->chip_device = NULL;
    }
}

LwlStatus
lwswitch_init_thermal
(
    lwswitch_device *device
)
{
    return(device->hal.lwswitch_init_thermal(device));
}

LwU32
lwswitch_read_physical_id
(
    lwswitch_device *device
)
{
    return(device->hal.lwswitch_read_physical_id(device));
}

LwU32
lwswitch_get_caps_lwlink_version
(
    lwswitch_device *device
)
{
    return(device->hal.lwswitch_get_caps_lwlink_version(device));
}

void
lwswitch_initialize_interrupt_tree
(
    lwswitch_device *device
)
{
    device->hal.lwswitch_initialize_interrupt_tree(device);
}

void
lwswitch_init_dlpl_interrupts
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;

    device->hal.lwswitch_init_dlpl_interrupts(link);
}

LwlStatus
lwswitch_initialize_pmgr
(
    lwswitch_device *device
)
{
    return(device->hal.lwswitch_initialize_pmgr(device));
}

LwlStatus
lwswitch_initialize_ip_wrappers
(
    lwswitch_device *device
)
{
    return(device->hal.lwswitch_initialize_ip_wrappers(device));
}

LwlStatus
lwswitch_initialize_route
(
    lwswitch_device *device
)
{
    return(device->hal.lwswitch_initialize_route(device));
}

void
lwswitch_soe_unregister_events
(
    lwswitch_device *device
)
{
    device->hal.lwswitch_soe_unregister_events(device);
}

LwlStatus
lwswitch_soe_register_event_callbacks
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_soe_register_event_callbacks(device);
}

LWSWITCH_BIOS_LWLINK_CONFIG *
lwswitch_get_bios_lwlink_config
(
    lwswitch_device *device
)
{
    return(device->hal.lwswitch_get_bios_lwlink_config(device));
}

LwlStatus
lwswitch_minion_send_command
(
    lwswitch_device *device,
    LwU32            linkNumber,
    LwU32            command,
    LwU32            scratch0
)
{
    return(device->hal.lwswitch_minion_send_command(device, linkNumber,
                                                    command, scratch0));
}

LwlStatus
lwswitch_init_nport
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_init_nport(device);
}

LwlStatus
lwswitch_init_nxbar
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_init_nxbar(device);
}

LwlStatus
lwswitch_clear_nport_rams
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_clear_nport_rams(device);
}

LwlStatus
lwswitch_pri_ring_init
(
    lwswitch_device *device
)
{
    return(device->hal.lwswitch_pri_ring_init(device));
}

LwlStatus
lwswitch_get_soe_ucode_binaries
(
    lwswitch_device *device,
    const LwU32 **soe_ucode_data,
    const LwU32 **soe_ucode_header
)
{
    return device->hal.lwswitch_get_soe_ucode_binaries(device, soe_ucode_data, soe_ucode_header);
}

LwlStatus
lwswitch_get_remap_table_selector
(
    lwswitch_device *device,
    LWSWITCH_TABLE_SELECT_REMAP table_selector,
    LwU32 *remap_ram_sel
)
{
    return device->hal.lwswitch_get_remap_table_selector(device, table_selector, remap_ram_sel);
}

LwU32
lwswitch_get_ingress_ram_size
(
    lwswitch_device *device,
    LwU32 ingress_ram_selector      // LW_INGRESS_REQRSPMAPADDR_RAM_ADDRESS_*
)
{
    return device->hal.lwswitch_get_ingress_ram_size(device, ingress_ram_selector);
}

LwlStatus
lwswitch_minion_get_dl_status
(
    lwswitch_device *device,
    LwU32            linkId,
    LwU32            statusIdx,
    LwU32            statusArgs,
    LwU32           *statusData
)
{
    return device->hal.lwswitch_minion_get_dl_status(device, linkId, statusIdx, statusArgs, statusData);
}

LwBool
lwswitch_is_i2c_supported
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_is_i2c_supported(device);
}


LwlStatus
lwswitch_poll_sublink_state
(
    lwswitch_device *device,
    lwlink_link *link
)
{
    return device->hal.lwswitch_poll_sublink_state(device, link);
}

void
lwswitch_setup_link_loopback_mode
(
    lwswitch_device *device,
    LwU32            linkNumber
)
{
    return device->hal.lwswitch_setup_link_loopback_mode(device, linkNumber);
}

void
lwswitch_reset_persistent_link_hw_state
(
    lwswitch_device *device,
    LwU32            linkNumber
)
{
    return device->hal.lwswitch_reset_persistent_link_hw_state(device, linkNumber);
}

void
lwswitch_store_topology_information
(
    lwswitch_device *device,
    lwlink_link *link
)
{
    return device->hal.lwswitch_store_topology_information(device, link);
}

void
lwswitch_init_lpwr_regs
(
    lwlink_link *link
)
{
   lwswitch_device *device = link->dev->pDevInfo;
   device->hal.lwswitch_init_lpwr_regs(link);
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
LwlStatus
lwswitch_launch_ALI
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_launch_ALI(device);
}
#endif //LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

LwlStatus
lwswitch_set_training_mode
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_set_training_mode(device);
}

LwBool
lwswitch_is_link_in_reset
(
    lwswitch_device *device,
    lwlink_link     *link
)
{
    return device->hal.lwswitch_is_link_in_reset(device, link);
}

LwBool
lwswitch_i2c_is_device_access_allowed
(
    lwswitch_device *device,
    LwU32 port,
    LwU8 addr,
    LwBool bIsRead
)
{
    return device->hal.lwswitch_i2c_is_device_access_allowed(device, port, addr, bIsRead);
}

LwlStatus
lwswitch_parse_bios_image
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_parse_bios_image(device);
}

void
lwswitch_init_buffer_ready
(
    lwswitch_device *device,
    lwlink_link *link,
    LwBool bNportBufferReady
)
{
    return device->hal.lwswitch_init_buffer_ready(device, link, bNportBufferReady);
}

LwU32
lwswitch_read_iddq_dvdd
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_read_iddq_dvdd(device);
}

void
lwswitch_apply_recal_settings
(
    lwswitch_device *device,
    lwlink_link *link
)
{
    return device->hal.lwswitch_apply_recal_settings(device, link);
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
void
lwswitch_fetch_active_repeater_mask
(
    lwswitch_device *device
)
{
    device->hal.lwswitch_fetch_active_repeater_mask(device);
}

LwU64
lwswitch_get_active_repeater_mask
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_get_active_repeater_mask(device);
}

LwBool
lwswitch_is_cci_supported
(
    lwswitch_device *device
)
{
    return device->hal.lwswitch_is_cci_supported(device);
}

LwlStatus
lwswitch_get_board_id
(
    lwswitch_device *device,
    LwU16 *boardId
)
{
    return device->hal.lwswitch_get_board_id(device, boardId);
}

static LwlStatus
_lwswitch_ctrl_set_port_test_mode
(
    lwswitch_device *device,
    LWSWITCH_SET_PORT_TEST_MODE *p
)
{
    return device->hal.lwswitch_ctrl_set_port_test_mode(device, p);
}

static LwlStatus
_lwswitch_ctrl_jtag_chain_read
(
    lwswitch_device *device,
    LWSWITCH_JTAG_CHAIN_PARAMS *jtag_chain
)
{
    return device->hal.lwswitch_ctrl_jtag_chain_read(device, jtag_chain);
}

static LwlStatus
_lwswitch_ctrl_jtag_chain_write
(
    lwswitch_device *device,
    LWSWITCH_JTAG_CHAIN_PARAMS *jtag_chain
)
{
    return device->hal.lwswitch_ctrl_jtag_chain_write(device, jtag_chain);
}

static LwlStatus
_lwswitch_ctrl_pex_clear_counters
(
    lwswitch_device *device,
    LWSWITCH_PEX_CLEAR_COUNTERS_PARAMS *pParams
)
{
    return device->hal.lwswitch_ctrl_pex_clear_counters(device, pParams);
}

static LwlStatus
_lwswitch_ctrl_pex_get_lane_counters
(
    lwswitch_device *device,
    LWSWITCH_PEX_GET_LANE_COUNTERS_PARAMS *pParams
)
{
    return device->hal.lwswitch_ctrl_pex_get_lane_counters(device, pParams);
}

LwlStatus
lwswitch_ctrl_i2c_get_dev_info
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_GET_DEV_INFO_PARAMS *pParams
)
{
    return device->hal.lwswitch_ctrl_i2c_get_dev_info(device, pParams);
}

static LwlStatus
_lwswitch_ctrl_therm_read_voltage
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_VOLTAGE_PARAMS *info
)
{
    return device->hal.lwswitch_ctrl_therm_read_voltage(device, info);
}

static LwlStatus
_lwswitch_ctrl_config_eom
(
    lwswitch_device *device,
    LWSWITCH_CTRL_CONFIG_EOM *p
)
{
    return device->hal.lwswitch_ctrl_config_eom(device, p);
}

static LwlStatus
_lwswitch_ctrl_inject_link_error
(
    lwswitch_device *device,
    LWSWITCH_INJECT_LINK_ERROR *p
)
{
    return device->hal.lwswitch_ctrl_inject_link_error(device, p);
}

static LwlStatus
_lwswitch_ctrl_get_lwlink_caps
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_CAPS_PARAMS *ret
)
{
    return device->hal.lwswitch_ctrl_get_lwlink_caps(device, ret);
}

static LwlStatus
_lwswitch_ctrl_clear_counters
(
    lwswitch_device *device,
    LWSWITCH_LWLINK_CLEAR_COUNTERS_PARAMS *ret
)
{
    return device->hal.lwswitch_ctrl_clear_counters(device, ret);
}

static LwlStatus
_lwswitch_ctrl_get_err_info
(
    lwswitch_device *device,
    LWSWITCH_LWLINK_GET_ERR_INFO_PARAMS *ret
)
{
    return device->hal.lwswitch_ctrl_get_err_info(device, ret);
}

static LwlStatus
_lwswitch_ctrl_get_irq_info
(
    lwswitch_device *device,
    LWSWITCH_GET_IRQ_INFO_PARAMS *ret
)
{
    return device->hal.lwswitch_ctrl_get_irq_info(device, ret);
}

static LwlStatus
_lwswitch_ctrl_read_uphy_pad_lane_reg
(
    lwswitch_device *device,
    LWSWITCH_CTRL_READ_UPHY_PAD_LANE_REG *p
)
{
    return device->hal.lwswitch_ctrl_read_uphy_pad_lane_reg(device, p);
}

static LwlStatus
_lwswitch_ctrl_pex_set_eom
(
    lwswitch_device *device,
    LWSWITCH_PEX_CTRL_EOM *pParams
)
{
    return device->hal.lwswitch_ctrl_pex_set_eom(device, pParams);
}

static LwlStatus
_lwswitch_ctrl_pex_get_eom_status
(
    lwswitch_device *device,
    LWSWITCH_PEX_GET_EOM_STATUS_PARAMS *pParams
)
{
    return device->hal.lwswitch_ctrl_pex_get_eom_status(device, pParams);
}

LwlStatus
lwswitch_ctrl_get_uphy_dln_cfg_space
(
    lwswitch_device *device,
    LWSWITCH_GET_PEX_UPHY_DLN_CFG_SPACE_PARAMS *pParams
)
{
    return device->hal.lwswitch_ctrl_get_uphy_dln_cfg_space(device, pParams);
}

LwlStatus
lwswitch_ctrl_force_thermal_slowdown
(
    lwswitch_device *device,
    LWSWITCH_CTRL_SET_THERMAL_SLOWDOWN *p
)
{
    return device->hal.lwswitch_ctrl_force_thermal_slowdown(device, p);
}

LwlStatus
lwswitch_ctrl_set_pcie_link_speed
(
    lwswitch_device *device,
    LWSWITCH_SET_PCIE_LINK_SPEED_PARAMS *pParams
)
{
    return device->hal.lwswitch_ctrl_set_pcie_link_speed(device, pParams);
}

static LwlStatus
_lwswitch_lib_ctrl_mods_only
(
    lwswitch_device *device,
    LwU32 cmd,
    void *params,
    LwU64 size
)
{
    LwlStatus retval;

    switch (cmd)
    {
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_SET_PORT_TEST_MODE,
                _lwswitch_ctrl_set_port_test_mode,
                LWSWITCH_SET_PORT_TEST_MODE);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_READ_JTAG_CHAIN,
                _lwswitch_ctrl_jtag_chain_read,
                LWSWITCH_JTAG_CHAIN_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_WRITE_JTAG_CHAIN,
                _lwswitch_ctrl_jtag_chain_write,
                LWSWITCH_JTAG_CHAIN_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_PEX_GET_COUNTERS,
                lwswitch_ctrl_pex_get_counters,
                LWSWITCH_PEX_GET_COUNTERS_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_PEX_CLEAR_COUNTERS,
                _lwswitch_ctrl_pex_clear_counters,
                LWSWITCH_PEX_CLEAR_COUNTERS_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_PEX_GET_LANE_COUNTERS,
                _lwswitch_ctrl_pex_get_lane_counters,
                LWSWITCH_PEX_GET_LANE_COUNTERS_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_I2C_GET_PORT_INFO,
                lwswitch_ctrl_i2c_get_port_info,
                LWSWITCH_CTRL_I2C_GET_PORT_INFO_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_I2C_GET_DEV_INFO,
                lwswitch_ctrl_i2c_get_dev_info,
                LWSWITCH_CTRL_I2C_GET_DEV_INFO_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_I2C_INDEXED,
                lwswitch_ctrl_i2c_indexed,
                LWSWITCH_CTRL_I2C_INDEXED_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_GET_VOLTAGE,
                _lwswitch_ctrl_therm_read_voltage,
                LWSWITCH_CTRL_GET_VOLTAGE_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_CONFIG_EOM,
                _lwswitch_ctrl_config_eom,
                LWSWITCH_CTRL_CONFIG_EOM);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_INJECT_LINK_ERROR,
                _lwswitch_ctrl_inject_link_error,
                LWSWITCH_INJECT_LINK_ERROR);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_GET_LWLINK_CAPS,
                _lwswitch_ctrl_get_lwlink_caps,
                LWSWITCH_GET_LWLINK_CAPS_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_CLEAR_COUNTERS,
                _lwswitch_ctrl_clear_counters,
                LWSWITCH_LWLINK_CLEAR_COUNTERS_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_GET_ERR_INFO,
                _lwswitch_ctrl_get_err_info,
                LWSWITCH_LWLINK_GET_ERR_INFO_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_GET_IRQ_INFO,
                _lwswitch_ctrl_get_irq_info,
                LWSWITCH_GET_IRQ_INFO_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_READ_UPHY_PAD_LANE_REG,
                _lwswitch_ctrl_read_uphy_pad_lane_reg,
                LWSWITCH_CTRL_READ_UPHY_PAD_LANE_REG);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_PEX_SET_EOM,
                _lwswitch_ctrl_pex_set_eom,
                LWSWITCH_PEX_CTRL_EOM);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_PEX_GET_EOM_STATUS,
                _lwswitch_ctrl_pex_get_eom_status,
                LWSWITCH_PEX_GET_EOM_STATUS_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_PEX_GET_UPHY_DLN_CFG_SPACE,
                lwswitch_ctrl_get_uphy_dln_cfg_space,
                LWSWITCH_GET_PEX_UPHY_DLN_CFG_SPACE_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_SET_THERMAL_SLOWDOWN,
                lwswitch_ctrl_force_thermal_slowdown,
                LWSWITCH_CTRL_SET_THERMAL_SLOWDOWN);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_SET_PCIE_LINK_SPEED,
                lwswitch_ctrl_set_pcie_link_speed,
                LWSWITCH_SET_PCIE_LINK_SPEED_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_CCI_GET_CAPABILITIES,
                lwswitch_ctrl_get_cci_capabilities,
                LWSWITCH_CCI_GET_CAPABILITIES_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_CCI_GET_TEMPERATURE,
                lwswitch_ctrl_get_cci_temperature,
                LWSWITCH_CCI_GET_TEMPERATURE);
        LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_CCI_GET_FW_REVISIONS,
            lwswitch_ctrl_get_cci_fw_revisions,
            LWSWITCH_CCI_GET_FW_REVISION_PARAMS);
    LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_CCI_GET_MODULE_STATE,
            lwswitch_ctrl_get_module_state,
            LWSWITCH_CCI_GET_MODULE_STATE);
    LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_CCI_GET_MODULE_FLAGS,
            lwswitch_ctrl_get_module_flags,
            LWSWITCH_CCI_GET_MODULE_FLAGS);
    LWSWITCH_DEV_CMD_DISPATCH_MODS(CTRL_LWSWITCH_CCI_GET_VOLTAGE,
            lwswitch_ctrl_get_voltage,
            LWSWITCH_CCI_GET_VOLTAGE);
        default:
            retval = _lwswitch_lib_ctrl_test_only(device, cmd, params, size);
            break;
    }

    return retval;
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

LwlStatus
lwswitch_lib_ctrl
(
    lwswitch_device *device,
    LwU32 cmd,
    void *params,
    LwU64 size,
    void *osPrivate
)
{
    LwlStatus retval;
    LwU64 flags = 0;

    if (!LWSWITCH_IS_DEVICE_ACCESSIBLE(device) || params == NULL)
    {
        return -LWL_BAD_ARGS;
    }

    flags = LWSWITCH_DEV_CMD_CHECK_ADMIN | LWSWITCH_DEV_CMD_CHECK_FM;
    switch (cmd)
    {
        LWSWITCH_DEV_CMD_DISPATCH(CTRL_LWSWITCH_GET_INFO,
                _lwswitch_ctrl_get_info,
                LWSWITCH_GET_INFO);
        LWSWITCH_DEV_CMD_DISPATCH(CTRL_LWSWITCH_GET_INTERNAL_LATENCY,
                _lwswitch_ctrl_get_internal_latency,
                LWSWITCH_GET_INTERNAL_LATENCY);
        LWSWITCH_DEV_CMD_DISPATCH(CTRL_LWSWITCH_GET_LWLIPT_COUNTERS,
                _lwswitch_ctrl_get_lwlipt_counters,
                LWSWITCH_GET_LWLIPT_COUNTERS);
        LWSWITCH_DEV_CMD_DISPATCH(CTRL_LWSWITCH_GET_ERRORS,
                lwswitch_ctrl_get_errors,
                LWSWITCH_GET_ERRORS_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH(CTRL_LWSWITCH_GET_LWLINK_STATUS,
                _lwswitch_ctrl_get_lwlink_status,
                LWSWITCH_GET_LWLINK_STATUS_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_WITH_PRIVATE_DATA(
                CTRL_LWSWITCH_ACQUIRE_CAPABILITY,
                _lwswitch_ctrl_acquire_capability,
                LWSWITCH_ACQUIRE_CAPABILITY_PARAMS,
                osPrivate);
        LWSWITCH_DEV_CMD_DISPATCH(CTRL_LWSWITCH_GET_TEMPERATURE,
                _lwswitch_ctrl_therm_read_temperature,
                LWSWITCH_CTRL_GET_TEMPERATURE_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH(CTRL_LWSWITCH_GET_THROUGHPUT_COUNTERS,
                _lwswitch_ctrl_get_throughput_counters,
                LWSWITCH_GET_THROUGHPUT_COUNTERS_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH(CTRL_LWSWITCH_GET_FATAL_ERROR_SCOPE,
                _lwswitch_ctrl_get_fatal_error_scope,
                LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_SET_SWITCH_PORT_CONFIG,
                _lwswitch_ctrl_set_switch_port_config,
                LWSWITCH_SET_SWITCH_PORT_CONFIG,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_GET_INGRESS_REQUEST_TABLE,
                _lwswitch_ctrl_get_ingress_request_table,
                LWSWITCH_GET_INGRESS_REQUEST_TABLE_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_SET_INGRESS_REQUEST_TABLE,
                _lwswitch_ctrl_set_ingress_request_table,
                LWSWITCH_SET_INGRESS_REQUEST_TABLE,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_SET_INGRESS_REQUEST_VALID,
                _lwswitch_ctrl_set_ingress_request_valid,
                LWSWITCH_SET_INGRESS_REQUEST_VALID,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE,
                _lwswitch_ctrl_get_ingress_response_table,
                LWSWITCH_GET_INGRESS_RESPONSE_TABLE_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE,
                _lwswitch_ctrl_set_ingress_response_table,
                LWSWITCH_SET_INGRESS_RESPONSE_TABLE,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_SET_GANGED_LINK_TABLE,
                _lwswitch_ctrl_set_ganged_link_table,
                LWSWITCH_SET_GANGED_LINK_TABLE,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(CTRL_LWSWITCH_SET_LATENCY_BINS,
                lwswitch_ctrl_set_latency_bins,
                LWSWITCH_SET_LATENCY_BINS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_SET_LWLIPT_COUNTER_CONFIG,
                _lwswitch_ctrl_set_lwlipt_counter_config,
                LWSWITCH_SET_LWLIPT_COUNTER_CONFIG,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_GET_LWLIPT_COUNTER_CONFIG,
                _lwswitch_ctrl_get_lwlipt_counter_config,
                LWSWITCH_GET_LWLIPT_COUNTER_CONFIG,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(CTRL_LWSWITCH_SET_REMAP_POLICY,
                _lwswitch_ctrl_set_remap_policy,
                LWSWITCH_SET_REMAP_POLICY,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(CTRL_LWSWITCH_GET_REMAP_POLICY,
                _lwswitch_ctrl_get_remap_policy,
                LWSWITCH_GET_REMAP_POLICY_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_SET_REMAP_POLICY_VALID,
                _lwswitch_ctrl_set_remap_policy_valid,
                LWSWITCH_SET_REMAP_POLICY_VALID,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(CTRL_LWSWITCH_SET_ROUTING_ID,
                _lwswitch_ctrl_set_routing_id,
                LWSWITCH_SET_ROUTING_ID,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(CTRL_LWSWITCH_GET_ROUTING_ID,
                _lwswitch_ctrl_get_routing_id,
                LWSWITCH_GET_ROUTING_ID_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(CTRL_LWSWITCH_SET_ROUTING_ID_VALID,
                _lwswitch_ctrl_set_routing_id_valid,
                LWSWITCH_SET_ROUTING_LAN_VALID,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(CTRL_LWSWITCH_SET_ROUTING_LAN,
                _lwswitch_ctrl_set_routing_lan,
                LWSWITCH_SET_ROUTING_LAN,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(CTRL_LWSWITCH_GET_ROUTING_LAN,
                _lwswitch_ctrl_get_routing_lan,
                LWSWITCH_GET_ROUTING_LAN_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_SET_ROUTING_LAN_VALID,
                _lwswitch_ctrl_set_routing_lan_valid,
                LWSWITCH_SET_ROUTING_LAN_VALID,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_GET_INGRESS_REQLINKID,
                _lwswitch_ctrl_get_ingress_reqlinkid,
                LWSWITCH_GET_INGRESS_REQLINKID_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(CTRL_LWSWITCH_UNREGISTER_LINK,
                _lwswitch_ctrl_unregister_link,
                LWSWITCH_UNREGISTER_LINK_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_RESET_AND_DRAIN_LINKS,
                _lwswitch_ctrl_reset_and_drain_links,
                LWSWITCH_RESET_AND_DRAIN_LINKS_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH(CTRL_LWSWITCH_GET_BIOS_INFO,
                _lwswitch_ctrl_get_bios_info,
                LWSWITCH_GET_BIOS_INFO_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_BLACKLIST_DEVICE,
                lwswitch_ctrl_blacklist_device,
                LWSWITCH_BLACKLIST_DEVICE_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_SET_FM_DRIVER_STATE,
                lwswitch_ctrl_set_fm_driver_state,
                LWSWITCH_SET_FM_DRIVER_STATE_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_SET_DEVICE_FABRIC_STATE,
                lwswitch_ctrl_set_device_fabric_state,
                LWSWITCH_SET_DEVICE_FABRIC_STATE_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT,
                lwswitch_ctrl_set_fm_timeout,
                LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_WITH_PRIVATE_DATA(
                CTRL_LWSWITCH_REGISTER_EVENTS,
                _lwswitch_ctrl_register_events,
                LWSWITCH_REGISTER_EVENTS_PARAMS,
                osPrivate);
        LWSWITCH_DEV_CMD_DISPATCH_WITH_PRIVATE_DATA(
                CTRL_LWSWITCH_UNREGISTER_EVENTS,
                _lwswitch_ctrl_unregister_events,
                LWSWITCH_UNREGISTER_EVENTS_PARAMS,
                osPrivate);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_SET_TRAINING_ERROR_INFO,
                _lwswitch_ctrl_set_training_error_info,
                LWSWITCH_SET_TRAINING_ERROR_INFO_PARAMS,
                osPrivate, flags);
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_SET_MC_RID_TABLE,
                _lwswitch_ctrl_set_mc_rid_table,
                LWSWITCH_SET_MC_RID_TABLE_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_GET_MC_RID_TABLE,
                _lwswitch_ctrl_get_mc_rid_table,
                LWSWITCH_GET_MC_RID_TABLE_PARAMS,
                osPrivate, flags);
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_GET_COUNTERS,
                _lwswitch_ctrl_get_counters,
                LWSWITCH_LWLINK_GET_COUNTERS_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_GET_LWLINK_ECC_ERRORS,
                _lwswitch_ctrl_get_lwlink_ecc_errors,
                LWSWITCH_GET_LWLINK_ECC_ERRORS_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_I2C_SMBUS_COMMAND,
                _lwswitch_ctrl_i2c_smbus_command,
                LWSWITCH_I2C_SMBUS_COMMAND_PARAMS,
                osPrivate, LWSWITCH_DEV_CMD_CHECK_ADMIN);
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_CCI_CMIS_PRESENCE,
                _lwswitch_ctrl_cci_cmis_presence,
                LWSWITCH_CCI_CMIS_PRESENCE_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_CCI_CMIS_LWLINK_MAPPING,
                _lwswitch_ctrl_cci_lwlink_mappings,
                LWSWITCH_CCI_CMIS_LWLINK_MAPPING_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ,
                _lwswitch_ctrl_cci_memory_access_read,
                LWSWITCH_CCI_CMIS_MEMORY_ACCESS_READ_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_CCI_CMIS_MEMORY_ACCESS_WRITE,
                _lwswitch_ctrl_cci_memory_access_write,
                LWSWITCH_CCI_CMIS_MEMORY_ACCESS_WRITE_PARAMS,
                osPrivate, flags);
	LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_CCI_CMIS_CAGE_BEZEL_MARKING,
                _lwswitch_ctrl_cci_cage_bezel_marking,
                LWSWITCH_CCI_CMIS_CAGE_BEZEL_MARKING_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_CCI_GET_GRADING_VALUES,
                lwswitch_ctrl_get_grading_values,
                LWSWITCH_CCI_GET_GRADING_VALUES_PARAMS,
                osPrivate, flags);
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
        LWSWITCH_DEV_CMD_DISPATCH(
                CTRL_LWSWITCH_GET_TEMPERATURE_LIMIT,
                _lwswitch_ctrl_therm_get_temperature_limit,
                LWSWITCH_CTRL_GET_TEMPERATURE_LIMIT_PARAMS);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_GET_LWLINK_MAX_ERROR_RATES,
                _lwswitch_ctrl_get_inforom_lwlink_max_correctable_error_rate,
                LWSWITCH_GET_LWLINK_MAX_CORRECTABLE_ERROR_RATES_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_GET_LWLINK_ERROR_COUNTS,
                _lwswitch_ctrl_get_inforom_lwlink_errors,
                LWSWITCH_GET_LWLINK_ERROR_COUNTS_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_GET_ECC_ERROR_COUNTS,
                _lwswitch_ctrl_get_inforom_ecc_errors,
                LWSWITCH_GET_ECC_ERROR_COUNTS_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_GET_SXIDS,
                _lwswitch_ctrl_get_inforom_bbx_sxid,
                LWSWITCH_GET_SXIDS_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_GET_FOM_VALUES,
                _lwswitch_ctrl_get_fom_values,
                LWSWITCH_GET_FOM_VALUES_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_GET_LWLINK_LP_COUNTERS,
                _lwswitch_ctrl_get_lwlink_lp_counters,
                LWSWITCH_GET_LWLINK_LP_COUNTERS_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH(CTRL_LWSWITCH_GET_RESIDENCY_BINS,
                _lwswitch_ctrl_get_residency_bins,
                LWSWITCH_GET_RESIDENCY_BINS);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(CTRL_LWSWITCH_SET_RESIDENCY_BINS,
                _lwswitch_ctrl_set_residency_bins,
                LWSWITCH_SET_RESIDENCY_BINS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH(CTRL_LWSWITCH_GET_RB_STALL_BUSY,
                _lwswitch_ctrl_get_rb_stall_busy,
                LWSWITCH_GET_RB_STALL_BUSY);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_INBAND_SEND_DATA,
                _lwswitch_ctrl_inband_send_data,
                LWSWITCH_INBAND_SEND_DATA_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_INBAND_READ_DATA,
                _lwswitch_ctrl_inband_read_data,
                 LWSWITCH_INBAND_READ_DATA_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWTICH_INBAND_FLUSH_DATA,
                _lwswitch_ctrl_inband_flush_data,
                LWSWITCH_INBAND_FLUSH_DATA_PARAMS,
                osPrivate, flags);
        LWSWITCH_DEV_CMD_DISPATCH_PRIVILEGED(
                CTRL_LWSWITCH_INBAND_PENDING_DATA_STATS,
                _lwswitch_ctrl_inband_pending_data_stats,
                LWSWITCH_INBAND_PENDING_DATA_STATS_PARAMS,
                osPrivate, flags);
        default:
        #if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
            retval = _lwswitch_lib_ctrl_mods_only(device, cmd, params, size);
        #else
            retval = _lwswitch_lib_ctrl_test_only(device, cmd, params, size);
        #endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
            break;
    }

    return retval;
}
