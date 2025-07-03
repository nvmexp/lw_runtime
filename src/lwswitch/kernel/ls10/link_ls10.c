/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "lwlink_export.h"
#include "g_lwconfig.h"
#include "export_lwswitch.h"
#include "common_lwswitch.h"
#include "regkey_lwswitch.h"
#include "ls10/ls10.h"
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#include "cci/cci_lwswitch.h"
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#include "lwswitch/ls10/dev_lwldl_ip.h"
#include "lwswitch/ls10/dev_lwlipt_lnk_ip.h"
#include "lwswitch/ls10/dev_lwlphyctl_ip.h"
#include "lwswitch/ls10/dev_lwltlc_ip.h"
#include "lwswitch/ls10/dev_minion_ip.h"
#include "lwswitch/ls10/dev_lwlipt_lnk_ip.h"
#include "lwswitch/ls10/dev_lwlipt_ip.h"
#include "lwswitch/ls10/dev_nport_ip.h"

#if !defined(LW_MODS)
static void
_lwswitch_configure_reserved_throughput_counters
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 linkNum = link->linkNumber;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLTLC, link->linkNumber))
    {
        LWSWITCH_PRINT(device, INFO,
                       "Invalid link, skipping LWLink throughput counter config for link %d\n",
                       link->linkNumber);
        return;
    }

    //
    // Counters 0 and 2 will be reserved for monitoring tools
    // Counters 1 and 3 will be user-configurable and used by devtools
    //

    // Rx0 config
    LWSWITCH_LINK_WR32_IDX_LS10(device, linkNum, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, 0,
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _UNIT, _FLITS)           |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _DATA)      |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _VCSETFILTERMODE, _INIT) |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _ENABLE, _ENABLE));

    // Tx0 config
    LWSWITCH_LINK_WR32_IDX_LS10(device, linkNum, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, 0,
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _UNIT, _FLITS)           |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _DATA)      |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _VCSETFILTERMODE, _INIT) |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _ENABLE, _ENABLE));

    // Rx2 config
    LWSWITCH_LINK_WR32_IDX_LS10(device, linkNum, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, 2,
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _UNIT, _FLITS)           |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _HEAD)      |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _AE)        |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _BE)        |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _DATA)      |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _VCSETFILTERMODE, _INIT) |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _ENABLE, _ENABLE));

    // Tx2 config
    LWSWITCH_LINK_WR32_IDX_LS10(device, linkNum, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, 2,
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _UNIT, _FLITS)           |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _HEAD)      |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _AE)        |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _BE)        |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _DATA)      |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _VCSETFILTERMODE, _INIT) |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _ENABLE, _ENABLE));
}
#endif  //!defined(LW_MODS)

void
lwswitch_init_lpwr_regs_ls10
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    // LWSWITCH_BIOS_LWLINK_CONFIG *bios_config;
    LwU32 linkNum = link->linkNumber;
    LwU32 tempRegVal, lpEntryThreshold;
    LwU8  softwareDesired;
    LwBool bLpEnable;
  
    if (device->regkeys.enable_pm == LW_SWITCH_REGKEY_ENABLE_PM_NO)
    {
        return;
    }

    // bios_config = lwswitch_get_bios_lwlink_config(device);

    // IC Enter Threshold
    // TODO: get from bios
    lpEntryThreshold = 16110000;

    tempRegVal = 0;
    tempRegVal = FLD_SET_DRF_NUM(_LWLIPT, _LNK_PWRM_L1_ENTER_THRESHOLD, _THRESHOLD, lpEntryThreshold, tempRegVal);
    LWSWITCH_LINK_WR32_LS10(device, linkNum, LWLIPT, _LWLIPT_LNK, _PWRM_L1_ENTER_THRESHOLD, tempRegVal);        

    //LP Entry Enable
    bLpEnable = LW_TRUE;
    softwareDesired = (bLpEnable) ? 0x1 : 0x0;

    tempRegVal = LWSWITCH_LINK_RD32_LS10(device, linkNum, LWLIPT, _LWLIPT_LNK, _CTRL_SYSTEM_LINK_AN1_CTRL);
    tempRegVal = FLD_SET_DRF_NUM(_LWLIPT, _LNK_CTRL_SYSTEM_LINK_AN1_CTRL, _PWRM_L1_ENABLE, softwareDesired, tempRegVal);
    LWSWITCH_LINK_WR32_LS10(device, linkNum, LWLIPT, _LWLIPT_LNK, _CTRL_SYSTEM_LINK_AN1_CTRL, tempRegVal);
}

void
lwswitch_corelib_training_complete_ls10
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;

    lwswitch_init_dlpl_interrupts(link);
#if !defined(LW_MODS)
    _lwswitch_configure_reserved_throughput_counters(link);
#endif

    if (lwswitch_lib_notify_client_events(device,
                LWSWITCH_DEVICE_EVENT_PORT_UP) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed to notify PORT_UP event\n",
                     __FUNCTION__);
    }

    return;
}

static LwlStatus
_lwswitch_init_dl_pll
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwlStatus status;

    status = lwswitch_minion_send_command(device, link->linkNumber, LW_MINION_LWLINK_DL_CMD_COMMAND_INITPLL, 0);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: INITPLL failed for link %d.\n",
            __FUNCTION__, link->linkNumber);
        LWSWITCH_ASSERT_INFO(LW_ERR_LWLINK_CLOCK_ERROR, LWBIT32(link->linkNumber), INITPLL_ERROR);
        return LW_ERR_LWLINK_CLOCK_ERROR;
    }

    status = lwswitch_minion_send_command(device, link->linkNumber, LW_MINION_LWLINK_DL_CMD_COMMAND_INITPHY, 0);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: INITPHY failed for link %d.\n",
            __FUNCTION__, link->linkNumber);
        LWSWITCH_ASSERT_INFO(LW_ERR_LWLINK_INIT_ERROR, LWBIT32(link->linkNumber), INITPHY_ERROR);
        return LW_ERR_LWLINK_INIT_ERROR;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_set_tx_mode_ls10
(
    lwlink_link *link,
    LwU64 mode,
    LwU32 flags
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 val;
    LwlStatus status = LWL_SUCCESS;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLDL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    // check if link is in reset
    if (lwswitch_is_link_in_reset(device, link))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d is still in reset, cannot change sub-link state\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_ERR_ILWALID_STATE;
    }

    val = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLDL, _LWLDL_TX, _SLSM_STATUS_TX);

    // Check if Sublink State Machine is ready to accept a sublink change request.
    status = lwswitch_poll_sublink_state(device, link);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s : SLSM not ready to accept a state change request for(%s):(%s).\n",
            __FUNCTION__, device->name, link->linkName);
        return status;
    }

    switch (mode)
    {
        case LWLINK_SUBLINK_STATE_TX_COMMON_MODE:
        {
            val = _lwswitch_init_dl_pll(link);
            if (val != LWL_SUCCESS)
            {
                return val;
            }

#if defined(LW_MODS)
            // Break if SSG has a request for per-link break after LWLDL init.
            if (FLD_TEST_DRF(_SWITCH_REGKEY, _SSG_CONTROL, _BREAK_AFTER_UPHY_INIT, _YES,
                device->regkeys.ssg_control))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: SSG Control: Break after UPHY Init on link %d\n",
                    __FUNCTION__, link->linkNumber);
                LWSWITCH_ASSERT(0);
            }
#endif  //defined(LW_MODS)

            break;
        }

        default:
        {
           status = lwswitch_corelib_set_tx_mode_lr10(link, mode, flags);
        }
    }

    return status;
}

LwU32
lwswitch_get_sublink_width_ls10
(
    lwswitch_device *device,
    LwU32            linkNumber
)
{
    LwU32 data = LWSWITCH_LINK_RD32_LS10(device, linkNumber, LWLIPT,
                                     _LWLIPT_COMMON, _TOPOLOGY_LOCAL_LINK_CONFIGURATION);
    return DRF_VAL(_LWLIPT_COMMON, _TOPOLOGY_LOCAL_LINK_CONFIGURATION, _NUM_LANES_PER_LINK, data);
}

void
lwswitch_corelib_get_uphy_load_ls10
(
    lwlink_link *link,
    LwBool *bUnlocked
)
{
    *bUnlocked = LW_FALSE;
}

LwlStatus
lwswitch_corelib_set_dl_link_mode_ls10
(
    lwlink_link *link,
    LwU64 mode,
    LwU32 flags
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32            val;
    LwU32            *seedData;
    LwlStatus        status = LWL_SUCCESS;
    LwBool           keepPolling;
    LWSWITCH_TIMEOUT timeout;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLDL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    switch (mode)
    {
        case LWLINK_LINKSTATE_INITPHASE1:
        {
            // Apply appropriate SIMMODE settings
            status = lwswitch_minion_set_sim_mode_ls10(device, link);
            if (status != LWL_SUCCESS)
            {
                return LW_ERR_LWLINK_CONFIGURATION_ERROR;
            }

            // Apply appropriate SMF settings
            status = lwswitch_minion_set_smf_settings_ls10(device, link);
            if (status != LWL_SUCCESS)
            {
                return LW_ERR_LWLINK_CONFIGURATION_ERROR;
            }

            // Apply appropriate UPHY Table settings
            status = lwswitch_minion_select_uphy_tables_ls10(device, link);
            if (status != LWL_SUCCESS)
            {
                return LW_ERR_LWLINK_CONFIGURATION_ERROR;
            }

            // Before INITPHASE1, apply NEA setting
            lwswitch_setup_link_loopback_mode(device, link->linkNumber);

            // training seed restoration for ALT LWLINK training
            if (device->regkeys.minion_cache_seeds == LW_SWITCH_REGKEY_MINION_CACHE_SEEDS_ENABLE)
            {
                // get seed data cached in corelib
                seedData = link->seedData;

                // restore seed data back into minion before INITPHASE1
                status = lwswitch_minion_restore_seed_data_ls10(device, link->linkNumber, seedData);

                if (status != LWL_SUCCESS)
                {
                    LWSWITCH_PRINT(device, ERROR, "%s : Error Writing seed data for link (%s):(%s)\n",
                                    __FUNCTION__, device->name, link->linkName);
                    LWSWITCH_PRINT(device, INFO,
                        "%s : Failed to restore back seeds to minion for link (%s):(%s).\n",
                        __FUNCTION__, device->name, link->linkName);
                }
            }

            status = lwswitch_minion_send_command(device, link->linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_INITPHASE1, 0);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : INITPHASE1 failed for link (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                LWSWITCH_ASSERT_INFO(LW_ERR_LWLINK_CONFIGURATION_ERROR,
                    LWBIT32(link->linkNumber), INITPHASE1_ERROR);
                return LW_ERR_LWLINK_CONFIGURATION_ERROR;
            }

            break;
        }

        case LWLINK_LINKSTATE_POST_INITOPTIMIZE:
        {
            // Poll for TRAINING_GOOD
            status  = lwswitch_minion_get_initoptimize_status_ls10(device, link->linkNumber);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                            "%s Error polling for INITOPTIMIZE TRAINING_GOOD. Link (%s):(%s)\n",
                            __FUNCTION__, device->name, link->linkName);
                LWSWITCH_ASSERT_INFO(LW_ERR_LWLINK_TRAINING_ERROR, LWBIT32(link->linkNumber), INITOPTIMIZE_ERROR);
                return LW_ERR_LWLINK_TRAINING_ERROR;
            }
            break;
        }

        case LWLINK_LINKSTATE_INITTL:
        {
             status = lwswitch_minion_send_command(device, link->linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_INITTL, 0);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : INITTL failed for link (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                LWSWITCH_ASSERT_INFO(LW_ERR_LWLINK_TRAINING_ERROR, LWBIT32(link->linkNumber), INITTL_ERROR);
                return LW_ERR_LWLINK_CONFIGURATION_ERROR;
            }
            break;
        }
        case LWLINK_LINKSTATE_INITOPTIMIZE:
        {
            return lwswitch_corelib_set_dl_link_mode_lr10(link, mode, flags);
        }

        case LWLINK_LINKSTATE_INITPHASE5:
        {
            status = lwswitch_minion_send_command(device, link->linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_INITPHASE5A, 0);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : INITPHASE5A failed to be called for link (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                LWSWITCH_ASSERT_INFO(LW_ERR_LWLINK_TRAINING_ERROR, LWBIT32(link->linkNumber), INITPHASE5_ERROR);
                return LW_ERR_LWLINK_CONFIGURATION_ERROR;
            }

            lwswitch_timeout_create(10 * LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);

            do
            {
                keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

                val =  LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLDL, _LWLPHYCTL_COMMON, _PSAVE_UCODE_CTRL_STS);
                if(FLD_TEST_DRF(_LWLPHYCTL_COMMON, _PSAVE_UCODE_CTRL_STS, _PMSTS, _PSL0, val))
                {
                    break;
                }

                if(!keepPolling)
                {
                    LWSWITCH_PRINT(device, ERROR,
                        "%s : Failed to poll for L0 on link (%s):(%s).\n",
                        __FUNCTION__, device->name, link->linkName);
                    LWSWITCH_ASSERT_INFO(LW_ERR_LWLINK_TRAINING_ERROR, LWBIT32(link->linkNumber), INITPHASE5_ERROR);
                    return LW_ERR_LWLINK_CONFIGURATION_ERROR;

                }
            }
            while (keepPolling);

            break;
        }
        default:
        {
            status = lwswitch_corelib_set_dl_link_mode_lr10(link, mode, flags);
        }
    }

    return status;
}

LwlStatus
lwswitch_corelib_get_rx_detect_ls10
(
    lwlink_link *link
)
{
    LwlStatus status;
    lwswitch_device *device = link->dev->pDevInfo;

    status = lwswitch_minion_get_rxdet_status_ls10(device, link->linkNumber);

    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Get RXDET failed for link %d.\n",
            __FUNCTION__, link->linkNumber);
        return status;
    }
    return LWL_SUCCESS;
}

void
lwswitch_reset_persistent_link_hw_state_ls10
(
    lwswitch_device *device,
    LwU32            linkNumber
)
{
    LwU32 regData;
    LwU32 lwliptWarmResetDelayUs = (IS_RTLSIM(device) || IS_EMULATION(device)) ? 800:8;

    regData = LWSWITCH_LINK_RD32_LS10(device, linkNumber, LWLIPT_LNK,
                    _LWLIPT_LNK, _DEBUG_CLEAR);
    regData = FLD_SET_DRF_NUM(_LWLIPT_LNK, _DEBUG_CLEAR, _CLEAR,
                     LW_LWLIPT_LNK_DEBUG_CLEAR_CLEAR_ASSERT, regData);
    LWSWITCH_LINK_WR32_LS10(device, linkNumber, LWLIPT_LNK,
                    _LWLIPT_LNK, _DEBUG_CLEAR, regData);

    LWSWITCH_NSEC_DELAY(lwliptWarmResetDelayUs * LWSWITCH_INTERVAL_1USEC_IN_NS);

    regData = LWSWITCH_LINK_RD32_LS10(device, linkNumber, LWLIPT_LNK,
                    _LWLIPT_LNK, _DEBUG_CLEAR);
    regData = FLD_SET_DRF_NUM(_LWLIPT_LNK, _DEBUG_CLEAR, _CLEAR,
                     LW_LWLIPT_LNK_DEBUG_CLEAR_CLEAR_DEASSERT, regData);
    LWSWITCH_LINK_WR32_LS10(device, linkNumber, LWLIPT_LNK,
                    _LWLIPT_LNK, _DEBUG_CLEAR, regData);

    LWSWITCH_NSEC_DELAY(lwliptWarmResetDelayUs * LWSWITCH_INTERVAL_1USEC_IN_NS);

}

LwlStatus
lwswitch_corelib_get_tl_link_mode_ls10
(
    lwlink_link *link,
    LwU64 *mode
)
{
#if defined(INCLUDE_LWLINK_LIB)

    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 link_state;
    LwU32 val = 0;
    LwlStatus status = LWL_SUCCESS;

    *mode = LWLINK_LINKSTATE_OFF;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLDL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    // check if links are in reset
    if (lwswitch_is_link_in_reset(device, link))
    {
        *mode = LWLINK_LINKSTATE_RESET;
        return LWL_SUCCESS;
    }

    // Read state from LWLIPT HW
    val = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLIPT_LNK,
            _LWLIPT_LNK, _CTRL_LINK_STATE_STATUS);

    link_state = DRF_VAL(_LWLIPT_LNK, _CTRL_LINK_STATE_STATUS, _LWRRENTLINKSTATE,
            val);

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    if (cciIsLinkManaged(device, link->linkNumber))
    {
        if (link_state == LW_LWLIPT_LNK_CTRL_LINK_STATE_STATUS_LWRRENTLINKSTATE_RESET)
        {
            *mode = LWLINK_LINKSTATE_RESET;
            return LWL_SUCCESS;
        }
    }
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    switch(link_state)
    {
        case LW_LWLIPT_LNK_CTRL_LINK_STATE_STATUS_LWRRENTLINKSTATE_ACTIVE:

            // If using ALI, ensure that the request to active completed
            if (link->dev->enableALI)
            {
                status = lwswitch_wait_for_tl_request_ready_ls10(link);
            }

            *mode = (status == LWL_SUCCESS) ? LWLINK_LINKSTATE_HS:LWLINK_LINKSTATE_OFF;
            break;

        case LW_LWLIPT_LNK_CTRL_LINK_STATE_STATUS_LWRRENTLINKSTATE_L2:
            *mode = LWLINK_LINKSTATE_SLEEP;
            break;

        case LW_LWLIPT_LNK_CTRL_LINK_STATE_STATUS_LWRRENTLINKSTATE_CONTAIN:
            *mode = LWLINK_LINKSTATE_CONTAIN;
            break;

        default:
            // Lwrrently, only ACTIVE, L2 and CONTAIN states are supported
            return LWL_ERR_ILWALID_STATE;
            break;
    }

#endif

    return status;
}

LwBool
lwswitch_is_link_in_reset_ls10
(
    lwswitch_device *device,
    lwlink_link     *link
)
{
    LwU32 linkState;
    LwU32 clkStatus;
    LwU32 resetRequestStatus;

    linkState = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber,
            LWLIPT_LNK, _LWLIPT_LNK, _CTRL_LINK_STATE_STATUS);

    linkState = DRF_VAL(_LWLIPT_LNK, _CTRL_LINK_STATE_STATUS,
                                     _LWRRENTLINKSTATE, linkState);

    clkStatus = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber,
            LWLIPT_LNK, _LWLIPT_LNK, _CTRL_CLK_CTRL);

    // Read the reset request register
    resetRequestStatus = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber,
                LWLIPT_LNK, _LWLIPT_LNK, _RESET_RSTSEQ_LINK_RESET);

    //
    // For link to be in reset either of 2 conditions should be true
    // 1. On a cold-boot the RESET_RSTSEQ status should be ASSERTED reset
    // 2. A link's current TL link state should be _RESET
    // and all of the per link clocks, RXCLK, TXCLK and NCISOCCLK, should be off
    //
    if ((DRF_VAL(_LWLIPT_LNK, _RESET_RSTSEQ_LINK_RESET, _LINK_RESET_STATUS, resetRequestStatus) ==
               LW_LWLIPT_LNK_RESET_RSTSEQ_LINK_RESET_LINK_RESET_STATUS_ASSERTED)     ||
        (linkState == LW_LWLIPT_LNK_CTRL_LINK_STATE_STATUS_LWRRENTLINKSTATE_RESET    &&
        FLD_TEST_DRF(_LWLIPT_LNK, _CTRL_CLK_CTRL, _RXCLK_STS, _OFF, clkStatus)       &&
        FLD_TEST_DRF(_LWLIPT_LNK, _CTRL_CLK_CTRL, _TXCLK_STS, _OFF, clkStatus)       &&
        FLD_TEST_DRF(_LWLIPT_LNK, _CTRL_CLK_CTRL, _NCISOCCLK_STS, _OFF, clkStatus)))
    {
        return LW_TRUE;
    }

    return LW_FALSE;
}

void
lwswitch_init_buffer_ready_ls10
(
    lwswitch_device *device,
    lwlink_link *link,
    LwBool bNportBufferReady
)
{
    LwU32 val;
    LwU32 linkNum = link->linkNumber;
    LwU64 forcedConfigLinkMask;
    LwU32 localLinkNumber = linkNum % LWSWITCH_LINKS_PER_MINION_LS10;

    forcedConfigLinkMask = ((LwU64)device->regkeys.chiplib_forced_config_link_mask) +
                ((LwU64)device->regkeys.chiplib_forced_config_link_mask2 << 32);

    //
    // Use legacy LR10 function to set buffer ready if
    // running with forced config since MINION is not
    // booted
    //
    if (forcedConfigLinkMask != 0)
    {
        lwswitch_init_buffer_ready_lr10(device, link, bNportBufferReady);
    }

    if (FLD_TEST_DRF(_SWITCH_REGKEY, _SKIP_BUFFER_READY, _TLC, _NO,
                     device->regkeys.skip_buffer_ready))
    {
        LWSWITCH_MINION_WR32_LS10(device,
                LWSWITCH_GET_LINK_ENG_INST(device, linkNum, MINION),
                _MINION, _LWLINK_DL_CMD_DATA(localLinkNumber),
                LW_MINION_LWLINK_DL_CMD_DATA_DATA_SET_BUFFER_READY_TX_AND_RX);

        lwswitch_minion_send_command(device, linkNum,
            LW_MINION_LWLINK_DL_CMD_COMMAND_SET_BUFFER_READY, 0);
    }

    if (bNportBufferReady &&
        FLD_TEST_DRF(_SWITCH_REGKEY, _SKIP_BUFFER_READY, _NPORT, _NO,
                     device->regkeys.skip_buffer_ready))
    {
        val = DRF_NUM(_NPORT, _CTRL_BUFFER_READY, _BUFFERRDY, 0x1);
        LWSWITCH_LINK_WR32_LS10(device, linkNum, NPORT, _NPORT, _CTRL_BUFFER_READY, val);
    }
}

void
lwswitch_apply_recal_settings_ls10
(
    lwswitch_device *device,
    lwlink_link *link
)
{
    LwU32 linkNumber = link->linkNumber;
    LwU32 regVal;
    LwU32 settingVal;

    // If no recal settings are set then return early
    if (device->regkeys.link_recal_settings == LW_SWITCH_REGKEY_LINK_RECAL_SETTINGS_NOP)
    {
        return;
    }

    regVal = LWSWITCH_LINK_RD32_LS10(device, linkNumber, LWLIPT_LNK, _LWLIPT_LNK,
                _CTRL_SYSTEM_LINK_CHANNEL_CTRL2);

    settingVal = DRF_VAL(_SWITCH_REGKEY, _LINK_RECAL_SETTINGS, _MIN_RECAL_TIME_MANTISSA,
                    device->regkeys.link_recal_settings);
    if (settingVal != LW_SWITCH_REGKEY_LINK_RECAL_SETTINGS_NOP)
    {
        regVal = FLD_SET_DRF_NUM(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL2,
                _L1_MINIMUM_RECALIBRATION_TIME_MANTISSA, settingVal, regVal);
    }

    settingVal = DRF_VAL(_SWITCH_REGKEY, _LINK_RECAL_SETTINGS, _MIN_RECAL_TIME_EXPONENT,
                    device->regkeys.link_recal_settings);
    if (settingVal != LW_SWITCH_REGKEY_LINK_RECAL_SETTINGS_NOP)
    {
        regVal = FLD_SET_DRF_NUM(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL2,
                _L1_MINIMUM_RECALIBRATION_TIME_EXPONENT, 0x2, regVal);
    }

    settingVal = DRF_VAL(_SWITCH_REGKEY, _LINK_RECAL_SETTINGS, _MAX_RECAL_PERIOD_MANTISSA,
                    device->regkeys.link_recal_settings);
    if (settingVal != LW_SWITCH_REGKEY_LINK_RECAL_SETTINGS_NOP)
    {
        regVal = FLD_SET_DRF_NUM(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL2,
                _L1_MAXIMUM_RECALIBRATION_PERIOD_MANTISSA, 0xf, regVal);
    }

    settingVal = DRF_VAL(_SWITCH_REGKEY, _LINK_RECAL_SETTINGS, _MAX_RECAL_PERIOD_EXPONENT,
                    device->regkeys.link_recal_settings);
    if (settingVal != LW_SWITCH_REGKEY_LINK_RECAL_SETTINGS_NOP)
    {
        regVal = FLD_SET_DRF_NUM(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL2,
            _L1_MAXIMUM_RECALIBRATION_PERIOD_EXPONENT, 0x3, regVal);
    }

    LWSWITCH_LINK_WR32_LS10(device, linkNumber, LWLIPT_LNK, _LWLIPT_LNK,
            _CTRL_SYSTEM_LINK_CHANNEL_CTRL2, regVal);

    return;
}

LwlStatus
lwswitch_corelib_get_dl_link_mode_ls10
(
    lwlink_link *link,
    LwU64 *mode
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 link_state;
    LwU32 val = 0;
    LwU64 tlLinkMode;

    *mode = LWLINK_LINKSTATE_OFF;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLDL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    // check if links are in reset
    if (lwswitch_is_link_in_reset(device, link))
    {
        *mode = LWLINK_LINKSTATE_RESET;
        return LWL_SUCCESS;
    }

    val = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _LINK_STATE);

    link_state = DRF_VAL(_LWLDL_TOP, _LINK_STATE, _STATE, val);

    switch (link_state)
    {
        case LW_LWLDL_TOP_LINK_STATE_STATE_INIT:
            *mode = LWLINK_LINKSTATE_OFF;
            break;
        case LW_LWLDL_TOP_LINK_STATE_STATE_HWCFG:
            *mode = LWLINK_LINKSTATE_DETECT;
            break;
        case LW_LWLDL_TOP_LINK_STATE_STATE_SWCFG:
            *mode = LWLINK_LINKSTATE_SAFE;
            break;
        case LW_LWLDL_TOP_LINK_STATE_STATE_ACTIVE:
            *mode = LWLINK_LINKSTATE_HS;
            break;
        case LW_LWLDL_TOP_LINK_STATE_STATE_SLEEP:
            if (device->hal.lwswitch_corelib_get_tl_link_mode(link, &tlLinkMode) != LWL_SUCCESS ||
                tlLinkMode == LWLINK_LINKSTATE_SLEEP)
            {
                *mode = LWLINK_LINKSTATE_SLEEP;
            }
            else
            {
                *mode = LWLINK_LINKSTATE_HS;
            }
            break;
        case LW_LWLDL_TOP_LINK_STATE_STATE_FAULT:
            *mode = LWLINK_LINKSTATE_FAULT;
            break;
        case LW_LWLDL_TOP_LINK_STATE_STATE_RCVY_AC:
        case LW_LWLDL_TOP_LINK_STATE_STATE_RCVY_RX:
            *mode = LWLINK_LINKSTATE_RECOVERY;
            break;
        default:
            *mode = LWLINK_LINKSTATE_OFF;
            break;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_get_rx_mode_ls10
(
    lwlink_link *link,
    LwU64 *mode,
    LwU32 *subMode
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 rx_sublink_state;
    LwU32 data = 0;
    LwU64 dlLinkMode;
    *mode = LWLINK_SUBLINK_STATE_RX_OFF;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLDL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    // check if link is in reset
    if (lwswitch_is_link_in_reset(device, link))
    {
        *mode = LWLINK_SUBLINK_STATE_RX_OFF;
        return LWL_SUCCESS;
    }

    data = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLDL, _LWLDL_RX, _SLSM_STATUS_RX);

    rx_sublink_state = DRF_VAL(_LWLDL_RX, _SLSM_STATUS_RX, _PRIMARY_STATE, data);

    // Return LWLINK_SUBLINK_SUBSTATE_RX_STABLE for sub-state
    *subMode = LWLINK_SUBLINK_SUBSTATE_RX_STABLE;

    switch (rx_sublink_state)
    {
        case LW_LWLDL_RX_SLSM_STATUS_RX_PRIMARY_STATE_HS:
            *mode = LWLINK_SUBLINK_STATE_RX_HS;
            break;

        case LW_LWLDL_RX_SLSM_STATUS_RX_PRIMARY_STATE_TRAIN:
            *mode = LWLINK_SUBLINK_STATE_RX_TRAIN;
            break;

        case LW_LWLDL_RX_SLSM_STATUS_RX_PRIMARY_STATE_SAFE:
            *mode = LWLINK_SUBLINK_STATE_RX_SAFE;
            break;
        case LW_LWLDL_RX_SLSM_STATUS_RX_PRIMARY_STATE_OFF:
            if (device->hal.lwswitch_corelib_get_dl_link_mode(link, &dlLinkMode) != LWL_SUCCESS ||
                dlLinkMode != LWLINK_LINKSTATE_HS)
            {
                *mode = LWLINK_SUBLINK_STATE_RX_OFF;
            }
            else
            {
                *mode = LWLINK_SUBLINK_STATE_RX_LOW_POWER;
            }
            break;

        default:
            *mode = LWLINK_SUBLINK_STATE_RX_OFF;
            break;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_get_tx_mode_ls10
(
    lwlink_link *link,
    LwU64 *mode,
    LwU32 *subMode
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 tx_sublink_state;
    LwU64 dlLinkMode;
    LwU32 data = 0;

    *mode = LWLINK_SUBLINK_STATE_TX_OFF;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LS10(device, LWLDL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    // check if link is in reset
    if (lwswitch_is_link_in_reset(device, link))
    {
        *mode = LWLINK_SUBLINK_STATE_TX_OFF;
        return LWL_SUCCESS;
    }

    data = LWSWITCH_LINK_RD32_LS10(device, link->linkNumber, LWLDL, _LWLDL_TX, _SLSM_STATUS_TX);

    tx_sublink_state = DRF_VAL(_LWLDL_TX, _SLSM_STATUS_TX, _PRIMARY_STATE, data);

    // Return LWLINK_SUBLINK_SUBSTATE_TX_STABLE for sub-state
    *subMode = LWLINK_SUBLINK_SUBSTATE_TX_STABLE;

    switch (tx_sublink_state)
    {
        case LW_LWLDL_TX_SLSM_STATUS_TX_PRIMARY_STATE_HS:
            *mode = LWLINK_SUBLINK_STATE_TX_HS;
            break;

        case LW_LWLDL_TX_SLSM_STATUS_TX_PRIMARY_STATE_TRAIN:
            *mode = LWLINK_SUBLINK_STATE_TX_TRAIN;
            break;

        case LW_LWLDL_TX_SLSM_STATUS_TX_PRIMARY_STATE_SAFE:
            *mode = LWLINK_SUBLINK_STATE_TX_SAFE;
            break;

        case LW_LWLDL_TX_SLSM_STATUS_TX_PRIMARY_STATE_OFF:
            if (device->hal.lwswitch_corelib_get_dl_link_mode(link, &dlLinkMode) != LWL_SUCCESS ||
                dlLinkMode != LWLINK_LINKSTATE_HS)
            {
                *mode = LWLINK_SUBLINK_STATE_TX_OFF;
            }
            else
            {
                *mode = LWLINK_SUBLINK_STATE_TX_LOW_POWER;
            }
            break;

        default:
            *mode = LWLINK_SUBLINK_STATE_TX_OFF;
            break;
    }

    return LWL_SUCCESS;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
/* ADD NEW UNPUBLISHED CODE BELOW THIS LINE */

LwlStatus
lwswitch_corelib_enable_optical_maintenance_ls10
(
    lwlink_link *link,
    LwBool bTx
)
{
    return LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_corelib_set_optical_infinite_mode_ls10
(
    lwlink_link *link,
    LwBool bEnable
)
{
    return LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_corelib_set_optical_iobist_ls10
(
    lwlink_link *link,
    LwBool bEnable
)
{
    return LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_corelib_set_optical_pretrain_ls10
(
    lwlink_link *link,
    LwBool bTx,
    LwBool bEnable
)
{
    return LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_corelib_check_optical_pretrain_ls10
(
    lwlink_link *link,
    LwBool bTx,
    LwBool *bSuccess
)
{
    return LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_corelib_init_optical_links_ls10(lwlink_link *link)
{
    return LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_corelib_check_optical_eom_status_ls10
(
    lwlink_link *link,
    LwBool      *bEomLow
)
{
    return LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_corelib_set_optical_force_eq_ls10
(
    lwlink_link *link,
    LwBool bEnable
)
{
    return LWL_ERR_NOT_SUPPORTED;
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
