/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2021 by LWPU Corporation.  All rights reserved.  All
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
#include "sv10/sv10.h"
#include "sv10/minion_sv10.h"

#include "lwswitch/svnp01/dev_lwlipt_ip.h"
#include "lwswitch/svnp01/dev_lwltlc_ip.h"
#include "lwswitch/svnp01/dev_lwl_ip.h"
#include "lwswitch/svnp01/dev_lwl_ip_addendum.h"
#include "lwswitch/svnp01/dev_lwlctrl_ip.h"
#include "lwswitch/svnp01/dev_minion_ip.h"
#include "lwswitch/svnp01/dev_trim.h"
#include "lwswitch/svnp01/dev_nport_ip.h"
#include "lwswitch/svnp01/dev_nport_ip_addendum.h"
#include "lwswitch/svnp01/dev_pri_ringstation_sys.h"

#define LW_PCLOCK_LWSW_LWLINK_CTRL(idx)                                 \
    (LW_PCLOCK_LWSW_LWLINK0_CTRL +                                      \
     idx * (LW_PCLOCK_LWSW_LWLINK1_CTRL - LW_PCLOCK_LWSW_LWLINK0_CTRL))

#define LW_PCLOCK_LWSW_LWLINK_STATUS(idx)                                 \
    (LW_PCLOCK_LWSW_LWLINK0_STATUS +                                      \
     idx * (LW_PCLOCK_LWSW_LWLINK1_STATUS - LW_PCLOCK_LWSW_LWLINK0_STATUS))

// Forward declarations

static LwlStatus _lwswitch_init_link(lwlink_link *link);
void lwswitch_init_dlpl_interrupts_sv10(lwlink_link *link);
static void _lwswitch_disable_dlpl_interrupts(lwlink_link *link);
static void _lwswitch_enable_minion_generated_dlpl_interrupts(lwlink_link *link);
static void _lwswitch_disable_minion_generated_dlpl_interrupts(lwlink_link *link);

// Calls out of the lwlink core library should be made with great care
LwlStatus lwswitch_force_pwr_mgmt(lwswitch_device *device, LwU32 linkNumber, LwBool bEnable);

LwlStatus
lwswitch_corelib_add_link_sv10
(
    lwlink_link *link
)
{
    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_remove_link_sv10
(
    lwlink_link *link
)
{
    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_set_dl_link_mode_sv10
(
    lwlink_link *link,
    LwU64 mode,
    LwU32 flags
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32           val;
    LwU32           link_state;
    LwlStatus       status = LWL_SUCCESS;

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _LINK_STATE);

    link_state = DRF_VAL(_PLWL, _LINK_STATE, _STATE, val);

    switch (mode)
    {
        case LWLINK_LINKSTATE_SAFE:
        {
            if (link_state == LW_PLWL_LINK_STATE_STATE_SWCFG)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : Link is already in Safe mode for (%s).\n",
                    __FUNCTION__, link->linkName);
                break;
            }
            else if (link_state == LW_PLWL_LINK_STATE_STATE_HWCFG)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : Link already transitioning to Safe mode for (%s).\n",
                    __FUNCTION__, link->linkName);
                break;
            }

            LWSWITCH_PRINT(device, INFO,
                "LWRM: %s : Changing Link state to Safe for (%s):(%s).\n",
                __FUNCTION__, device->name, link->linkName);

            if (link_state == LW_PLWL_LINK_STATE_STATE_INIT)
            {
                val = 0;
                val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _LINK_CHANGE);
                val = FLD_SET_DRF(_PLWL, _LINK_CHANGE, _NEWSTATE, _HWCFG, val);
                val = FLD_SET_DRF(_PLWL, _LINK_CHANGE, _OLDSTATE_MASK, _DONTCARE, val);
                val = FLD_SET_DRF(_PLWL, _LINK_CHANGE, _ACTION, _LTSSM_CHANGE, val);
                LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL, _LINK_CHANGE, val);
            }
            else if (link_state == LW_PLWL_LINK_STATE_STATE_ACTIVE)
            {
                val = 0;
                val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _LINK_CHANGE);
                val = FLD_SET_DRF(_PLWL, _LINK_CHANGE, _NEWSTATE, _SWCFG, val);
                val = FLD_SET_DRF(_PLWL, _LINK_CHANGE, _OLDSTATE_MASK, _DONTCARE, val);
                val = FLD_SET_DRF(_PLWL, _LINK_CHANGE, _ACTION, _LTSSM_CHANGE, val);
                LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL, _LINK_CHANGE, val);
            }
            else
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : Link is in invalid state"
                    " cannot set to safe state (%s):(%s). (%x) (%x)\n",
                    __FUNCTION__, device->name, link->linkName, val, link_state);
                return -LWL_ERR_ILWALID_STATE;
            }

            break;
        }

        case LWLINK_LINKSTATE_HS:
        {
            if (link_state == LW_PLWL_LINK_STATE_STATE_ACTIVE)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : Link is already in Active mode (%s).\n",
                    __FUNCTION__, link->linkName);
                break;
            }
            else if (link_state == LW_PLWL_LINK_STATE_STATE_INIT)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : Link cannot be taken from INIT state to"
                    " Active mode for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                return -LWL_ERR_ILWALID_STATE;
            }
            else if (link_state == LW_PLWL_LINK_STATE_STATE_SWCFG)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : Changing Link state to Active for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);

                val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _LINK_CHANGE);
                val = FLD_SET_DRF(_PLWL, _LINK_CHANGE, _NEWSTATE, _ACTIVE, val);
                val = FLD_SET_DRF(_PLWL, _LINK_CHANGE, _OLDSTATE_MASK, _DONTCARE, val);
                val = FLD_SET_DRF(_PLWL, _LINK_CHANGE, _ACTION, _LTSSM_CHANGE, val);
                LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL, _LINK_CHANGE, val);
            }
            else
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : Link is in invalid state"
                    " cannot set to active state (%s):(%s). (%x) (%x)\n",
                    __FUNCTION__, device->name, link->linkName, val, link_state);
                return -LWL_ERR_ILWALID_STATE;
            }

            break;
        }

        case LWLINK_LINKSTATE_OFF:
        {
            break;
        }

        case LWLINK_LINKSTATE_RESET:
        {
            break;
        }

        case LWLINK_LINKSTATE_ENABLE_PM:
        {
            // Not a POR.
            if (device->regkeys.enable_pm == LW_SWITCH_REGKEY_ENABLE_PM_YES)
            {
                status = lwswitch_minion_send_command_sv10(device, link->linkNumber,
                    LW_MINION_LWLINK_DL_CMD_COMMAND_ENABLEPM, 0);

                if (status != LWL_SUCCESS)
                {
                    LWSWITCH_PRINT(device, ERROR,
                        "%s: ENABLEPM CMD failed for link %d.\n",
                        __FUNCTION__, link->linkNumber);
                    return status;
                }
            }

            break;
        }

        case LWLINK_LINKSTATE_DISABLE_PM:
        {
            // Not a POR.
            if (device->regkeys.enable_pm == LW_SWITCH_REGKEY_ENABLE_PM_YES)
            {
                status = lwswitch_minion_send_command_sv10(device, link->linkNumber,
                    LW_MINION_LWLINK_DL_CMD_COMMAND_DISABLEPM, 0);

                if (status != LWL_SUCCESS)
                {
                    LWSWITCH_PRINT(device, ERROR,
                        "%s: DISABLEPM CMD failed for link %d.\n",
                        __FUNCTION__, link->linkNumber);
                    return status;
                }
            }

            break;
        }

        case LWLINK_LINKSTATE_DISABLE_HEARTBEAT:
        {
            // NOP
            break;
        }

        case LWLINK_LINKSTATE_PRE_HS:
        {
            break;
        }

        case LWLINK_LINKSTATE_TRAFFIC_SETUP:
        {
            // NOP
            break;
        }

        case LWLINK_LINKSTATE_DISABLE_ERR_DETECT:
        {
            // Disable DL/PL interrupts
            _lwswitch_disable_dlpl_interrupts(link);
            break;
        }

        case LWLINK_LINKSTATE_LANE_DISABLE:
        {
            status = lwswitch_minion_send_command_sv10(device, link->linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_LANEDISABLE, 0);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : LANEDISABLE CMD failed for link (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                return status;
            }

            break;
        }

        case LWLINK_LINKSTATE_LANE_SHUTDOWN:
        {
            status = lwswitch_minion_send_command_sv10(device, link->linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_LANESHUTDOWN, 0);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : SHUTDOWN CMD failed for link (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                return status;
            }

            break;
        }

        case LWLINK_LINKSTATE_INITPHASE1:
        {
            //NOP
            break;
        }

        case LWLINK_LINKSTATE_INITNEGOTIATE:
        {
            //NOP
            break;
        }

        case LWLINK_LINKSTATE_POST_INITNEGOTIATE:
        {
            //NOP
            break;
        }
        default:
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s : Invalid mode specified.\n",
                __FUNCTION__);
            break;
        }
    }

    return LWL_SUCCESS;
}

LwU32
lwswitch_get_sublink_width_sv10
(
    lwswitch_device *device,
    LwU32            linkNumber
)
{
    return 0;
}

LwlStatus
lwswitch_corelib_get_dl_link_mode_sv10
(
    lwlink_link *link,
    LwU64 *mode
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 link_state;
    LwU32 val = 0;

    *mode = LWLINK_LINKSTATE_OFF;

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _LINK_STATE);

    link_state = DRF_VAL(_PLWL, _LINK_STATE, _STATE, val);

    switch (link_state)
    {
        case LW_PLWL_LINK_STATE_STATE_INIT:
            *mode = LWLINK_LINKSTATE_OFF;
            break;
        case LW_PLWL_LINK_STATE_STATE_HWCFG:
            *mode = LWLINK_LINKSTATE_DETECT;
            break;
        case LW_PLWL_LINK_STATE_STATE_SWCFG:
            *mode = LWLINK_LINKSTATE_SAFE;
            break;
        case LW_PLWL_LINK_STATE_STATE_ACTIVE:
            *mode = LWLINK_LINKSTATE_HS;
            break;
        case LW_PLWL_LINK_STATE_STATE_FAULT:
            *mode = LWLINK_LINKSTATE_FAULT;
            break;
        case LW_PLWL_LINK_STATE_STATE_RCVY_AC:
        case LW_PLWL_LINK_STATE_STATE_RCVY_SW:
        case LW_PLWL_LINK_STATE_STATE_RCVY_RX:
            *mode = LWLINK_LINKSTATE_RECOVERY;
            break;
        default:
            *mode = LWLINK_LINKSTATE_OFF;
            break;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_set_tl_link_mode_sv10
(
    lwlink_link *link,
    LwU64 mode,
    LwU32 flags
)
{
    return LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_corelib_get_tl_link_mode_sv10
(
    lwlink_link *link,
    LwU64 *mode
)
{
    return LWL_ERR_NOT_SUPPORTED;
}

void
lwswitch_corelib_get_uphy_load_sv10
(
    lwlink_link *link,
    LwBool *bUnlocked
)
{
    *bUnlocked = LW_FALSE;
}

static LwlStatus
_lwswitch_poll_sublink_state
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LWSWITCH_TIMEOUT timeout;
    LwU32 val;
    LwBool bPreSiPlatform = (IS_RTLSIM(device) || IS_EMULATION(device) || IS_FMODEL(device));

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1MSEC_IN_NS * (bPreSiPlatform ? 2000: 200), &timeout);

    val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _SUBLINK_CHANGE);
    while (!FLD_TEST_DRF(_PLWL, _SUBLINK_CHANGE, _STATUS, _DONE, val))
    {
        if (FLD_TEST_DRF(_PLWL, _SUBLINK_CHANGE, _STATUS, _FAULT, val))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s : Fault while changing sublink state (%s):(%s).\n",
                __FUNCTION__, device->name, link->linkName);
            return -LWL_ERR_ILWALID_STATE;
        }

        if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s : Timeout while waiting sublink state (%s):(%s).\n",
                __FUNCTION__, device->name, link->linkName);
            return -LWL_ERR_GENERIC;
        }

        lwswitch_os_sleep(1);

        val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _SUBLINK_CHANGE);
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_set_tx_mode_sv10
(
    lwlink_link *link,
    LwU64 mode,
    LwU32 flags
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 tx_sublink_state;
    LwU32 val;
    LwlStatus status = LWL_SUCCESS;

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL0, _SLSM_STATUS_TX);

    tx_sublink_state = DRF_VAL(_PLWL_SL0, _SLSM_STATUS_TX, _PRIMARY_STATE, val);

    // Check if Sublink State Machine is ready to accept a sublink change request.
    status = _lwswitch_poll_sublink_state(link);
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
            status = _lwswitch_init_link(link);
            if (status != LWL_SUCCESS)
            {
                return status;
            }

            {
                sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

                if (chip_device->link[link->linkNumber].nea)
                {
                    LWSWITCH_PRINT(device, ERROR,
                        "%s: Setting NEA on link %d\n",
                        __FUNCTION__, link->linkNumber);

                    status = lwswitch_minion_send_command_sv10(device, link->linkNumber,
                                LW_MINION_LWLINK_DL_CMD_COMMAND_SETNEA, 0);
                    if (status != LWL_SUCCESS)
                    {
                        LWSWITCH_PRINT(device, ERROR,
                            "%s: SETNEA CMD failed for link %d.\n",
                            __FUNCTION__, link->linkNumber);
                        return status;
                    }
                }
                else if (chip_device->link[link->linkNumber].ned)
                {
                    LWSWITCH_PRINT(device, ERROR,
                        "%s: Setting NED on link %d\n",
                        __FUNCTION__, link->linkNumber);

                    status = lwswitch_minion_send_command_sv10(device, link->linkNumber,
                                LW_MINION_LWLINK_DL_CMD_COMMAND_SETNEDR, 0);
                    if (status != LWL_SUCCESS)
                    {
                        LWSWITCH_PRINT(device, ERROR,
                            "%s: SETNEDR CMD failed for link %d.\n",
                            __FUNCTION__, link->linkNumber);
                        return status;
                    }

                    status = lwswitch_minion_send_command_sv10(device, link->linkNumber,
                                LW_MINION_LWLINK_DL_CMD_COMMAND_SETNEDW, 0);
                    if (status != LWL_SUCCESS)
                    {
                        LWSWITCH_PRINT(device, ERROR,
                            "%s: SETNEDW CMD failed for link %d.\n",
                            __FUNCTION__, link->linkNumber);
                        return status;
                    }
                }
            }

            status = lwswitch_minion_send_command_sv10(device, link->linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_INITRXTERM, 0);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, WARN,
                    "%s: _INITRXTERM CMD failed for link %d.\n",
                    __FUNCTION__, link->linkNumber);
            }

            if (link->ac_coupled)
            {
                val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _LINK_CONFIG);
                val = FLD_SET_DRF(_PLWL, _LINK_CONFIG, _AC_SAFE_EN, _ON, val);
                LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL, _LINK_CONFIG, val);

                status = lwswitch_minion_send_command_sv10(device, link->linkNumber,
                            LW_MINION_LWLINK_DL_CMD_COMMAND_SETACMODE, 0);
                if (status != LWL_SUCCESS)
                {
                    LWSWITCH_PRINT(device, ERROR,
                        "%s: SETACMODE CMD failed for link %d.\n",
                        __FUNCTION__, link->linkNumber);
                    return status;
                }
            }

            status = lwswitch_minion_send_command_sv10(device, link->linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_INITPHY, 0);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: INITPHY CMD failed for link %d.\n",
                    __FUNCTION__, link->linkNumber);
                return status;
            }

#if defined(LW_MODS)
            // Break if SSG has a request for per-link break after DLPL init.
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

        case LWLINK_SUBLINK_STATE_TX_COMMON_MODE_DISABLE:
        {
            // Not applicable for LW IP
            break;
        }

        case LWLINK_SUBLINK_STATE_TX_DATA_READY:
        {
            status = lwswitch_minion_send_command_sv10(device, link->linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_INITLANEENABLE, 0);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: INITLANEENABLE CMD failed for link %d.\n",
                    __FUNCTION__, link->linkNumber);
                return status;
            }

            status = lwswitch_minion_send_command_sv10(device, link->linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_INITDLPL, 0);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: INITDLPL CMD failed for link %d.\n",
                    __FUNCTION__, link->linkNumber);
                return status;
            }

#if defined(LW_MODS)
            // Break if SSG has a request for per-link break after DLPL init.
            if (FLD_TEST_DRF(_SWITCH_REGKEY, _SSG_CONTROL, _BREAK_AFTER_DLPL_INIT, _YES,
                device->regkeys.ssg_control))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: SSG Control: Break after DLPL Init on link %d\n",
                    __FUNCTION__, link->linkNumber);
                LWSWITCH_ASSERT(0);
            }
#endif  //defined(LW_MODS)

            break;
        }

        case LWLINK_SUBLINK_STATE_TX_PRBS_EN:
        {
            // WAR for Bug 1888034
            val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL0, _SAFE_CTRL2_TX);
            val = FLD_SET_DRF(_PLWL_SL0, _SAFE_CTRL2_TX, _CTR_INIT,    _INIT, val);
            val = FLD_SET_DRF(_PLWL_SL0, _SAFE_CTRL2_TX, _CTR_INITSCL, _INIT, val);
            LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL_SL0, _SAFE_CTRL2_TX, val);

            val = 0;
            LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _RXSLSM_TIMEOUT_2, val);

            // Enable PRBS pattern generator
            val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _TXIOBIST_CONFIGREG);
            val = FLD_SET_DRF_NUM(_PLWL, _TXIOBIST_CONFIGREG, _IO_BIST_MODE_IN, 0x1, val);
            LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL, _TXIOBIST_CONFIGREG, val);

            val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _TXIOBIST_CONFIG);
            val = FLD_SET_DRF_NUM(_PLWL, _TXIOBIST_CONFIG, _DPG_PRBSSEEDLD, 0x1, val);
            LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL, _TXIOBIST_CONFIG, val);

            val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _TXIOBIST_CONFIG);
            val = FLD_SET_DRF_NUM(_PLWL, _TXIOBIST_CONFIG, _DPG_PRBSSEEDLD, 0x0, val);
            LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL, _TXIOBIST_CONFIG, val);

            break;
        }

        case LWLINK_SUBLINK_STATE_TX_POST_HS:
        {
          // NOP: In general, there is no point to downgrade *_PRBS_* and *_SCRAM_* values.
          break;
        }

        case LWLINK_SUBLINK_STATE_TX_EQ:
        {
            //TODO: To be implemented
            break;
        }

        case LWLINK_SUBLINK_STATE_TX_HS:
        {
            if (tx_sublink_state == LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_HS)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : TX already in HSmode for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                break;
            }
            else if (tx_sublink_state == LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_OFF)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : TX cannot be taken from OFF to HS directly for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                return -LWL_ERR_ILWALID_STATE;
            }
            else if (tx_sublink_state == LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_SAFE)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : Changing TX sublink state to HS for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);

                val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _SUBLINK_CHANGE);
                val = FLD_SET_DRF(_PLWL, _SUBLINK_CHANGE, _NEWSTATE, _HS, val);
                val = FLD_SET_DRF(_PLWL, _SUBLINK_CHANGE, _SUBLINK, _TX, val);
                val = FLD_SET_DRF(_PLWL, _SUBLINK_CHANGE, _ACTION, _SLSM_CHANGE, val);
                LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL, _SUBLINK_CHANGE, val);
            }
            else
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : TX is in invalid state for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                return -LWL_ERR_ILWALID_STATE;
            }

            status = _lwswitch_poll_sublink_state(link);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : Error while changing TX sublink to HS for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                return status;
            }
            break;
        }

        case LWLINK_SUBLINK_STATE_TX_SAFE:
        {
            if (tx_sublink_state == LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_SAFE)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : TX already in Safe mode for  (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                break;
            }

            LWSWITCH_PRINT(device, INFO,
                  "%s : Changing TX sublink state to Safe mode for (%s):(%s).\n",
                  __FUNCTION__, device->name, link->linkName);

            val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _SUBLINK_CHANGE);
            val = FLD_SET_DRF(_PLWL, _SUBLINK_CHANGE, _NEWSTATE, _SAFE, val);
            val = FLD_SET_DRF(_PLWL, _SUBLINK_CHANGE, _SUBLINK, _TX, val);
            val = FLD_SET_DRF(_PLWL, _SUBLINK_CHANGE, _ACTION, _SLSM_CHANGE, val);
            LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL, _SUBLINK_CHANGE, val);

            status = _lwswitch_poll_sublink_state(link);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : Error while changing TX sublink to Safe Mode for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                return status;
            }

            break;
        }

        case LWLINK_SUBLINK_STATE_TX_OFF:
        {
            if (tx_sublink_state == LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_OFF)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : TX already OFF (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                break;
            }
            else if (tx_sublink_state == LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_HS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : TX cannot be taken from HS to OFF directly for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                return -LWL_ERR_GENERIC;
            }

            LWSWITCH_PRINT(device, INFO,
                "%s : Changing TX sublink state to OFF for (%s):(%s).\n",
                __FUNCTION__, device->name, link->linkName);

            val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _SUBLINK_CHANGE);
            val = FLD_SET_DRF(_PLWL, _SUBLINK_CHANGE, _COUNTDOWN, _IMMEDIATE, val);
            val = FLD_SET_DRF(_PLWL, _SUBLINK_CHANGE, _NEWSTATE, _OFF, val);
            val = FLD_SET_DRF(_PLWL, _SUBLINK_CHANGE, _SUBLINK, _TX, val);
            val = FLD_SET_DRF(_PLWL, _SUBLINK_CHANGE, _ACTION, _SLSM_CHANGE, val);
            LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL, _SUBLINK_CHANGE, val);

            status = _lwswitch_poll_sublink_state(link);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : Error while changing TX sublink to off Mode for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                return status;
            }
            break;
        }

        default:
            LWSWITCH_PRINT(device, ERROR,
                 "%s : Invalid TX sublink mode specified.\n",
                __FUNCTION__);
            break;
    }

    return status;
}

LwlStatus
lwswitch_corelib_get_tx_mode_sv10
(
    lwlink_link *link,
    LwU64 *mode,
    LwU32 *subMode
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 tx_sublink_state;
    LwU32 data = 0;

    *mode = LWLINK_SUBLINK_STATE_TX_OFF;

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL0, _SLSM_STATUS_TX);

    tx_sublink_state = DRF_VAL(_PLWL_SL0, _SLSM_STATUS_TX, _PRIMARY_STATE, data);

    // Return LWLINK_SUBLINK_SUBSTATE_TX_STABLE for sub-state
    *subMode = LWLINK_SUBLINK_SUBSTATE_TX_STABLE;

    switch (tx_sublink_state)
    {
        case LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_HS:
            *mode = LWLINK_SUBLINK_STATE_TX_HS;
            break;

        case LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_TRAIN:
            *mode = LWLINK_SUBLINK_STATE_TX_TRAIN;
            break;

        case LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_SAFE:
            *mode = LWLINK_SUBLINK_STATE_TX_SAFE;
            break;

        case LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_OFF:
            *mode = LWLINK_SUBLINK_STATE_TX_OFF;
            break;

        default:
            *mode = LWLINK_SUBLINK_STATE_TX_OFF;
            break;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_set_rx_mode_sv10
(
    lwlink_link *link,
    LwU64 mode,
    LwU32 flags
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 rx_sublink_state;
    LwU32 val;
    LwlStatus status = LWL_SUCCESS;
    LWSWITCH_TIMEOUT timeout;

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _SLSM_STATUS_RX);

    rx_sublink_state = DRF_VAL(_PLWL_SL1, _SLSM_STATUS_RX, _PRIMARY_STATE, val);

    // Check if Sublink State Machine is ready to accept a sublink change request.
    status = _lwswitch_poll_sublink_state(link);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s : SLSM not ready to accept a state change request for(%s):(%s).\n",
            __FUNCTION__, device->name, link->linkName);
        return status;
    }

    switch (mode)
    {
        case LWLINK_SUBLINK_STATE_RX_HS:
            break;

        case LWLINK_SUBLINK_STATE_RX_SAFE:
            break;

        case LWLINK_SUBLINK_STATE_RX_OFF:
        {
            if (rx_sublink_state == LW_PLWL_SL1_SLSM_STATUS_RX_PRIMARY_STATE_OFF)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : RX already OFF (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                break;
            }
            else if (rx_sublink_state == LW_PLWL_SL1_SLSM_STATUS_RX_PRIMARY_STATE_HS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : RX cannot be taken from HS to OFF directly for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                status = -LWL_ERR_GENERIC;
                return status;
            }

            LWSWITCH_PRINT(device, INFO,
                "%s : Changing RX sublink state to OFF for (%s):(%s).\n",
                __FUNCTION__, device->name, link->linkName);

            val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _SUBLINK_CHANGE);
            val = FLD_SET_DRF(_PLWL, _SUBLINK_CHANGE, _COUNTDOWN, _IMMEDIATE, val);
            val = FLD_SET_DRF(_PLWL, _SUBLINK_CHANGE, _OLDSTATE_MASK, _DONTCARE, val);
            val = FLD_SET_DRF(_PLWL, _SUBLINK_CHANGE, _NEWSTATE, _OFF, val);
            val = FLD_SET_DRF(_PLWL, _SUBLINK_CHANGE, _SUBLINK, _RX, val);

            // When changing RX sublink state use FORCE, otherwise it will fault.
            val = FLD_SET_DRF(_PLWL, _SUBLINK_CHANGE, _ACTION, _SLSM_FORCE, val);
            LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL, _SUBLINK_CHANGE, val);

            LWSWITCH_PRINT(device, INFO,
                "%s : LW_PLWL_SUBLINK_CHANGE = 0x%08x\n", __FUNCTION__, val);

            status = _lwswitch_poll_sublink_state(link);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : Error while changing RX sublink to Off Mode for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                return status;
            }
            break;
        }

        case LWLINK_SUBLINK_STATE_RX_RXCAL:
        {
            // Enable RXCAL in CFG_CTL_6 and Poll CFG_STATUS_0 for RXCAL_DONE=1.
            val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_BR0, _CFG_CTL_6);
            val = FLD_SET_DRF(_PLWL_BR0, _CFG_CTL_6, _RXCAL , _ON, val);
            LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL_BR0, _CFG_CTL_6, val);

            lwswitch_timeout_create(LWSWITCH_INTERVAL_1MSEC_IN_NS * 2000, &timeout);

            val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_BR0, _CFG_STATUS_0);
            while (!FLD_TEST_DRF_NUM(_PLWL_BR0, _CFG_STATUS_0, _RXCAL_DONE, 0x1, val))
            {
                if (lwswitch_timeout_check(&timeout))
                {
                    LWSWITCH_PRINT(device, ERROR,
                        "%s: Timeout while waiting for RXCAL_DONE on link %d.\n",
                        __FUNCTION__, link->linkNumber);
                    return -LWL_ERR_GENERIC;
                }

                lwswitch_os_sleep(1);
                val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_BR0, _CFG_STATUS_0);
            }
            break;
        }

        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s : Invalid RX sublink mode specified.\n",
                __FUNCTION__);
            break;
    }

    return status;
}

LwlStatus
lwswitch_corelib_get_rx_mode_sv10
(
    lwlink_link *link,
    LwU64 *mode,
    LwU32 *subMode
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 rx_sublink_state, rx_sublink_fence;
    LwU32 data = 0;

    *mode = LWLINK_SUBLINK_STATE_RX_OFF;

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    data = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _SLSM_STATUS_RX);

    rx_sublink_fence = DRF_VAL(_PLWL_SL1, _SLSM_STATUS_RX, _FENCE_STATUS, data);

    rx_sublink_state = DRF_VAL(_PLWL_SL1, _SLSM_STATUS_RX, _PRIMARY_STATE, data);

    if (rx_sublink_fence &&
        rx_sublink_state == LW_PLWL_SL1_SLSM_STATUS_RX_PRIMARY_STATE_SAFE)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s : Sublink is not ready for High Speed mode \n",__FUNCTION__);
        return -LWL_ERR_ILWALID_STATE;
    }

    // Return LWLINK_SUBLINK_SUBSTATE_RX_STABLE for sub-state
    *subMode = LWLINK_SUBLINK_SUBSTATE_RX_STABLE;

    switch (rx_sublink_state)
    {
        case LW_PLWL_SL1_SLSM_STATUS_RX_PRIMARY_STATE_HS:
            *mode = LWLINK_SUBLINK_STATE_RX_HS;
            break;

        case LW_PLWL_SL1_SLSM_STATUS_RX_PRIMARY_STATE_TRAIN:
            *mode = LWLINK_SUBLINK_STATE_RX_TRAIN;
            break;

        case LW_PLWL_SL1_SLSM_STATUS_RX_PRIMARY_STATE_SAFE:
            *mode = LWLINK_SUBLINK_STATE_RX_SAFE;
            break;

        case LW_PLWL_SL1_SLSM_STATUS_RX_PRIMARY_STATE_OFF:
            *mode = LWLINK_SUBLINK_STATE_RX_OFF;
            break;

        default:
            *mode = LWLINK_SUBLINK_STATE_RX_OFF;
            break;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_set_rx_detect_sv10
(
    lwlink_link *link,
    LwU32 flags
)
{
    // Receiver detect not supported on switch return success
    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_get_rx_detect_sv10
(
    lwlink_link *link
)
{
    // Receiver detect not supported on switch return success
    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_write_discovery_token_sv10
(
    lwlink_link *link,
    LwU64 token
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 command;
    LWSWITCH_TIMEOUT timeout;

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    LWSWITCH_PRINT(device, INFO,
        "%s: Sending token 0x%016llx to (%s).\n",
        __FUNCTION__, token, link->linkName);

    command = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL0, _R4TX_COMMAND);
    if (FLD_TEST_DRF(_PLWL_SL0, _R4TX_COMMAND, _READY, _NO, command))
    {
        LWSWITCH_PRINT(device, ERROR,
            "LWRM: %s: Unable to process packet injection request for link %s."
            " HW not ready for new request.\n",
            __FUNCTION__, link->linkName);
        return -LWL_ERR_GENERIC;
    }

    LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL_SL0, _R4TX_WDATA0, LwU64_LO32(token));
    LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL_SL0, _R4TX_WDATA1, LwU64_HI32(token));

    command = FLD_SET_DRF(_PLWL_SL0, _R4TX_COMMAND, _REQUEST,  _WRITE, command);
    command = FLD_SET_DRF(_PLWL_SL0, _R4TX_COMMAND, _COMPLETE, _READY_CLEAR, command);

    // SW is expected to use address 0x1 for topology detection
    command = FLD_SET_DRF(_PLWL_SL0, _R4TX_COMMAND, _WADDR, _SC, command);

    LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL_SL0, _R4TX_COMMAND, command);

    lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    do
    {
        command = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL0, _R4TX_COMMAND);
        if (FLD_TEST_DRF(_PLWL_SL0, _R4TX_COMMAND, _COMPLETE, _READY_CLEAR, command))
        {
            break;
        }
    }
    while (!lwswitch_timeout_check(&timeout));

    if (FLD_TEST_DRF(_PLWL_SL0, _R4TX_COMMAND, _COMPLETE, _READY_CLEAR, command))
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Discovery token = 0x%016llx sent for (%s).\n",
            __FUNCTION__, token, link->linkName);
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Timeout sending discovery token = 0x%016llx to (%s).\n",
            __FUNCTION__, token, link->linkName);
        return -LWL_ERR_GENERIC;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_read_discovery_token_sv10
(
    lwlink_link *link,
    LwU64 *token
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 value_hi;
    LwU32 value_lo;
    LwU32 command;
    LWSWITCH_TIMEOUT timeout;

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    command = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _R4LOCAL_COMMAND);
    if (FLD_TEST_DRF(_PLWL_SL1, _R4LOCAL_COMMAND, _READY, _NO, command))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Unable to read AN0 packet for link %s."
            " HW not ready for new request.\n",
            __FUNCTION__, link->linkName);
        return -LWL_ERR_GENERIC;
    }

    command = FLD_SET_DRF(_PLWL_SL1, _R4LOCAL_COMMAND, _REQUEST,  _READ, command);
    command = FLD_SET_DRF(_PLWL_SL1, _R4LOCAL_COMMAND, _COMPLETE, _READY_CLEAR, command);

    // SW is expected to use address 0x1 for topology detection
    command = FLD_SET_DRF(_PLWL_SL1, _R4LOCAL_COMMAND, _RADDR, _SC, command);

    // Issue the command to read the value
    LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _R4LOCAL_COMMAND, command);

    lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    do
    {
        command = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _R4LOCAL_COMMAND);
        if (FLD_TEST_DRF(_PLWL_SL1, _R4LOCAL_COMMAND, _COMPLETE, _READY_CLEAR, command))
        {
            break;
        }
    }
    while (!lwswitch_timeout_check(&timeout));

    if (FLD_TEST_DRF(_PLWL_SL1, _R4LOCAL_COMMAND, _COMPLETE, _READY_CLEAR, command))
    {
        // Read the value using little endian (aligned to data reg names)
        value_lo = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _R4LOCAL_RDATA0);
        value_hi = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _R4LOCAL_RDATA1);

        *token = ((LwU64)value_hi << 32) | (LwU64) value_lo;

        LWSWITCH_PRINT(device, INFO,
            "%s: Discovery token received on (%s) = 0x%016llx\n",
            __FUNCTION__, link->linkName, *token);

        //
        // RM uses memory address of each link as its unique
        // token for topology discovery. However, in the event of a driver
        // unload and reload, when links are freed and reallocated, some of
        // these addresses get reused. The old token values are still cached
        // in the registers which leads to incorrect topology. So, clear the
        // registers after reading the tokens to ensure that the tokens will
        // be truly unique across driver loads
        //
        LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _R4LOCAL_RDATA0, 0x0);
        LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _R4LOCAL_RDATA1, 0x0);

        LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _R4LOCAL_WDATA0, 0x0);
        LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _R4LOCAL_WDATA1, 0x0);

        // Send write command to clear out register.
        command = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _R4LOCAL_COMMAND);
        if (FLD_TEST_DRF(_PLWL_SL1, _R4LOCAL_COMMAND, _READY, _NO, command))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Unable to read AN0 packet for link %s."
                " HW not ready for new request.\n",
                __FUNCTION__, link->linkName);
            return -LWL_ERR_GENERIC;
        }

        command = FLD_SET_DRF(_PLWL_SL1, _R4LOCAL_COMMAND, _REQUEST, _WRITE, command);
        command = FLD_SET_DRF(_PLWL_SL1, _R4LOCAL_COMMAND, _COMPLETE, _READY_CLEAR, command);

        // SW is expected to use address 0x1 for topology detection
        command = FLD_SET_DRF(_PLWL_SL1, _R4LOCAL_COMMAND, _WADDR, _SC, command);

        // Issue the command to write (clear) the value
        LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _R4LOCAL_COMMAND, command);

        lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
        do
        {
            command = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL_SL1, _R4LOCAL_COMMAND);
            if (FLD_TEST_DRF(_PLWL_SL1, _R4LOCAL_COMMAND, _COMPLETE, _READY_CLEAR, command))
            {
                break;
            }
        }
        while (!lwswitch_timeout_check(&timeout));
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Timeout reading discovery token from (%s).\n",
            __FUNCTION__, link->linkName);
        return -LWL_ERR_GENERIC;
    }

    return LWL_SUCCESS;
}

void
lwswitch_corelib_training_complete_sv10
(
    lwlink_link *link
)
{
    lwswitch_init_dlpl_interrupts(link);
}

static LwlStatus
_lwswitch_init_dl_pll
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 cmd = LW_MINION_LWLINK_DL_CMD_COMMAND_INITPLL_0;
    LwU32 val;
    LwU32 master;
    LwU32 slave;
    LwBool pll_on;
    LWSWITCH_TIMEOUT timeout;
    LwlStatus status;

    if (IS_RTLSIM(device) || IS_EMULATION(device) || IS_FMODEL(device))
    {
        LWSWITCH_PRINT(device, WARN,
            "%s: Skipping setup of LwSwitch LWLIPT clocks\n",
            __FUNCTION__);
        return LWL_SUCCESS;
    }

    //
    // Link rate is selected per link pair, so really even & odd link
    // rate will both be set to the rate selected for the even link.
    //
    if (device->regkeys.lwlink_speed_control == LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_20G)
    {
        // Select 20Gbps
        cmd = LW_MINION_LWLINK_DL_CMD_COMMAND_INITPLL_1;
    }
    else
    {
        // Select 25.781Gbps
        cmd = LW_MINION_LWLINK_DL_CMD_COMMAND_INITPLL_0;
    }

    //
    // Send INITPLL command to minion if applicable. We store initpll_state
    // in LW_PLWL_SCRATCH_PRIVMASK1 to keep track of PLL state across driver
    // loads.
    //
    val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _SCRATCH_PRIVMASK1);

    master = (link->linkNumber & ~0x1) | 0x0;
    slave = (link->linkNumber & ~0x1) | 0x1;

    // Clear LW_PLWL_SCRATCH_PRIVMASK1 if set to init value
    if (FLD_TEST_DRF(_PLWL, _SCRATCH_PRIVMASK1, _DATA, _INIT, val))
    {
        LWSWITCH_LINK_WR32_SV10(device, master, DLPL, _PLWL, _SCRATCH_PRIVMASK1, 0);
        LWSWITCH_LINK_WR32_SV10(device, slave, DLPL, _PLWL, _SCRATCH_PRIVMASK1, 0);
        val = 0;
    }

    if (!FLD_TEST_DRF(_PLWL, _SCRATCH_PRIVMASK1, _INITPLL_LINK_STATE,
            _DONE, val))
    {
        // INITPLL has not been sent yet
        status = lwswitch_minion_send_command_sv10(device, master, cmd, 0);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: INITPLL CMD 0x%02x failed for link %d.\n",
                __FUNCTION__, cmd, master);
            return status;
        }

        // Save LW_PLWL_SCRATCH_PRIVMASK1 state for both master and slave
        val = LWSWITCH_LINK_RD32_SV10(device, master, DLPL, _PLWL, _SCRATCH_PRIVMASK1);
        val = FLD_SET_DRF(_PLWL, _SCRATCH_PRIVMASK1, _INITPLL_LINK_STATE, _DONE, val);
        LWSWITCH_LINK_WR32_SV10(device, master, DLPL, _PLWL, _SCRATCH_PRIVMASK1, val);

        val = LWSWITCH_LINK_RD32_SV10(device, slave, DLPL, _PLWL, _SCRATCH_PRIVMASK1);
        val = FLD_SET_DRF(_PLWL, _SCRATCH_PRIVMASK1, _INITPLL_LINK_STATE, _DONE, val);
        LWSWITCH_LINK_WR32_SV10(device, slave, DLPL, _PLWL, _SCRATCH_PRIVMASK1, val);
    }

    val = LWSWITCH_CLK_LWLINK_RD32_SV10(device, _CTRL, link->linkNumber);
    val = FLD_SET_DRF(_PCLOCK, _LWSW_LWLINK0_CTRL, _UNIT2CLKS_PLL_TURN_OFF, _NO, val);
    LWSWITCH_CLK_LWLINK_WR32_SV10(device, _CTRL, link->linkNumber, val);

    // 30ms timeout
    lwswitch_timeout_create(30 * LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);

    while(1)
    {
        val = LWSWITCH_CLK_LWLINK_RD32_SV10(device, _STATUS, link->linkNumber);
        pll_on = FLD_TEST_DRF_NUM(_PCLOCK, _LWSW_LWLINK0_STATUS,
                        _PLL_OFF, 0, val);

        if (pll_on)
        {
            break;
        }
        else if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: _PLL_OFF failed to toggle ON for link id %x\n",
                __FUNCTION__, link->linkNumber);
            return -LWL_ERR_ILWALID_STATE;
        }
    }

    return LWL_SUCCESS;
}

static void
_lwswitch_enable_minion_generated_dlpl_interrupts
(
    lwlink_link *link
)
{
    LwU32           intrEn;
    lwswitch_device *device    = link->dev->pDevInfo;
    LwU32           linkNumber = link->linkNumber;
    LwU8            dlpl_idx   = linkNumber % 2;

    intrEn = LWSWITCH_MINION_LINK_RD32_SV10(device, linkNumber, _MINION, _MINION_INTR_STALL_EN);

    intrEn = FLD_SET_DRF_NUM(_MINION, _MINION_INTR_STALL_EN, _LINK, LWBIT(dlpl_idx), intrEn);
    LWSWITCH_MINION_LINK_WR32_SV10(device, linkNumber, _MINION, _MINION_INTR_STALL_EN, intrEn);

    // Explicitly disable NONSTALL interrupts
    intrEn = LWSWITCH_MINION_LINK_RD32_SV10(device, linkNumber, _MINION, _MINION_INTR_NONSTALL_EN);
    intrEn = FLD_SET_DRF_NUM(_MINION, _MINION_INTR_NONSTALL_EN, _LINK, 0, intrEn);
    LWSWITCH_MINION_LINK_WR32_SV10(device, linkNumber, _MINION, _MINION_INTR_NONSTALL_EN, intrEn);
}

static void
_lwswitch_disable_minion_generated_dlpl_interrupts
(
    lwlink_link *link
)
{
    LwU32           intrEn, linkIntr;
    lwswitch_device *device    = link->dev->pDevInfo;
    LwU32           linkNumber = link->linkNumber;
    LwU8            dlpl_idx   = linkNumber % 2;

    intrEn = LWSWITCH_MINION_LINK_RD32_SV10(device, linkNumber, _MINION, _MINION_INTR_STALL_EN);

    // Disable interrupt bit for the given link
    linkIntr = DRF_VAL(_MINION, _MINION_INTR_STALL_EN, _LINK, intrEn);
    linkIntr &= ~LWBIT(dlpl_idx);

    intrEn = FLD_SET_DRF_NUM(_MINION, _MINION_INTR_STALL_EN, _LINK, linkIntr, intrEn);
    LWSWITCH_MINION_LINK_WR32_SV10(device, linkNumber, _MINION, _MINION_INTR_STALL_EN, intrEn);
}

void
lwswitch_init_dlpl_interrupts_sv10
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 linkNumber = link->linkNumber;

    // W1C any stale state.
    LWSWITCH_LINK_WR32_SV10(device, linkNumber, DLPL, _PLWL, _INTR, 0xffffffff);
    LWSWITCH_LINK_WR32_SV10(device, linkNumber, DLPL, _PLWL, _INTR_SW2, 0xffffffff);

    // Stall tree routes to INTR_A which is connected to LWLIPT fatal tree
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    // https://wiki.lwpu.com/gpuhwdept/index.php/LWLink/DL_programming_guide#Interrupts
#endif
    LWSWITCH_LINK_WR32_SV10(device, linkNumber, DLPL, _PLWL, _INTR_STALL_EN,
              DRF_DEF(_PLWL, _INTR_STALL_EN, _TX_REPLAY, _DISABLE)               |
              DRF_DEF(_PLWL, _INTR_STALL_EN, _TX_RECOVERY_SHORT, _DISABLE)       |
              DRF_DEF(_PLWL, _INTR_STALL_EN, _TX_RECOVERY_LONG, _ENABLE)         |
              DRF_DEF(_PLWL, _INTR_STALL_EN, _TX_FAULT_RAM, _ENABLE)             |
              DRF_DEF(_PLWL, _INTR_STALL_EN, _TX_FAULT_INTERFACE, _ENABLE)       |
              DRF_DEF(_PLWL, _INTR_STALL_EN, _TX_FAULT_SUBLINK_CHANGE, _DISABLE) |
              DRF_DEF(_PLWL, _INTR_STALL_EN, _RX_FAULT_SUBLINK_CHANGE, _DISABLE) |
              DRF_DEF(_PLWL, _INTR_STALL_EN, _RX_FAULT_DL_PROTOCOL, _ENABLE)     |
              DRF_DEF(_PLWL, _INTR_STALL_EN, _RX_SHORT_ERROR_RATE, _DISABLE)     |
              DRF_DEF(_PLWL, _INTR_STALL_EN, _RX_LONG_ERROR_RATE, _DISABLE)      |
              DRF_DEF(_PLWL, _INTR_STALL_EN, _RX_ILA_TRIGGER, _DISABLE)          |
              DRF_DEF(_PLWL, _INTR_STALL_EN, _RX_CRC_COUNTER, _DISABLE)          | // Disabled due to Bug:1890181
              DRF_DEF(_PLWL, _INTR_STALL_EN, _LTSSM_FAULT, _ENABLE)              |
              DRF_DEF(_PLWL, _INTR_STALL_EN, _LTSSM_PROTOCOL, _DISABLE)          |
              DRF_DEF(_PLWL, _INTR_STALL_EN, _MINION_REQUEST, _DISABLE));

    // All disabled
    LWSWITCH_LINK_WR32_SV10(device, linkNumber, DLPL, _PLWL, _INTR_NONSTALL_EN,
              DRF_DEF(_PLWL, _INTR_NONSTALL_EN, _TX_REPLAY, _DISABLE)               |
              DRF_DEF(_PLWL, _INTR_NONSTALL_EN, _TX_RECOVERY_SHORT, _DISABLE)       |
              DRF_DEF(_PLWL, _INTR_NONSTALL_EN, _TX_RECOVERY_LONG, _DISABLE)        |
              DRF_DEF(_PLWL, _INTR_NONSTALL_EN, _TX_FAULT_RAM, _DISABLE)            |
              DRF_DEF(_PLWL, _INTR_NONSTALL_EN, _TX_FAULT_INTERFACE, _DISABLE)      |
              DRF_DEF(_PLWL, _INTR_NONSTALL_EN, _TX_FAULT_SUBLINK_CHANGE, _DISABLE) |
              DRF_DEF(_PLWL, _INTR_NONSTALL_EN, _RX_FAULT_SUBLINK_CHANGE, _DISABLE) |
              DRF_DEF(_PLWL, _INTR_NONSTALL_EN, _RX_FAULT_DL_PROTOCOL, _DISABLE)    |
              DRF_DEF(_PLWL, _INTR_NONSTALL_EN, _RX_SHORT_ERROR_RATE, _DISABLE)     |
              DRF_DEF(_PLWL, _INTR_NONSTALL_EN, _RX_LONG_ERROR_RATE, _DISABLE)      |
              DRF_DEF(_PLWL, _INTR_NONSTALL_EN, _RX_ILA_TRIGGER, _DISABLE)          |
              DRF_DEF(_PLWL, _INTR_NONSTALL_EN, _RX_CRC_COUNTER, _DISABLE)          |
              DRF_DEF(_PLWL, _INTR_NONSTALL_EN, _LTSSM_FAULT, _DISABLE)             |
              DRF_DEF(_PLWL, _INTR_NONSTALL_EN, _LTSSM_PROTOCOL, _DISABLE)          |
              DRF_DEF(_PLWL, _INTR_NONSTALL_EN, _MINION_REQUEST, _DISABLE));

    // Enable MINION per-link interrupts
    _lwswitch_enable_minion_generated_dlpl_interrupts(link);

    LWSWITCH_FLUSH_MMIO(device);
}

static void
_lwswitch_disable_dlpl_interrupts
(
    lwlink_link *link
)
{
    lwswitch_device *device    = link->dev->pDevInfo;
    LwU32           linkNumber = link->linkNumber;

    LWSWITCH_LINK_WR32_SV10(device, linkNumber, DLPL, _PLWL, _INTR_STALL_EN,    0x0);
    LWSWITCH_LINK_WR32_SV10(device, linkNumber, DLPL, _PLWL, _INTR_NONSTALL_EN, 0x0);

    _lwswitch_disable_minion_generated_dlpl_interrupts(link);

    LWSWITCH_FLUSH_MMIO(device);
}

static void
_lwswitch_init_dlpl
(
    lwlink_link *link
)
{
    LwU32 val;
    lwswitch_device *device = link->dev->pDevInfo;

    if (IS_RTLSIM(device))
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: skipping DLPL init on RTL sim\n",
            __FUNCTION__);
        return;
    }

    val = LWSWITCH_LINK_RD32_SV10(device, link->linkNumber, DLPL, _PLWL, _LINK_CONFIG);
    val = FLD_SET_DRF(_PLWL, _LINK_CONFIG, _LINK_EN, _ON, val);
    LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL, _LINK_CONFIG, val);

    //
    // _BITS:
    //      * 0x80 - Bug #1810774
    //      * 0x10 - Bug #1774336
    //      * 0x08 - Bug #1762253
    //
    val = DRF_DEF(_PLWL, _SPARE_A, _CYA_LTSSM_WAIT_FOR_SAFE, __PROD)       |
          DRF_DEF(_PLWL, _SPARE_A, _CYA_LTSSM_ACTIVE_SLSM_FAULTS, __PROD)  |
          DRF_DEF(_PLWL, _SPARE_A, _CYA_LTSSM_RESET_RXSEQ_ON_RCVY, __PROD) |
          DRF_DEF(_PLWL, _SPARE_A, _BITS, __PROD);
    LWSWITCH_LINK_WR32_SV10(device, link->linkNumber, DLPL, _PLWL, _SPARE_A, val);

    lwswitch_init_dlpl_interrupts(link);
}

static void
_lwswitch_init_lwltlc_debug_counters
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 linkNum = link->linkNumber;

    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_RX, _DEBUG_TP_CNTR0_CTRL_0,
        DRF_DEF(_LWLTLC_RX, _DEBUG_TP_CNTR0_CTRL_0, _UNIT, _BYTES)      |
        DRF_DEF(_LWLTLC_RX, _DEBUG_TP_CNTR0_CTRL_0, _FLITFILTER, _DATA) |
        DRF_DEF(_LWLTLC_RX, _DEBUG_TP_CNTR0_CTRL_0, _VCSETFILTERMODE, _INIT));

    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_RX, _DEBUG_TP_CNTR1_CTRL_0,
        DRF_DEF(_LWLTLC_RX, _DEBUG_TP_CNTR1_CTRL_0, _UNIT, _PACKETS)    |
        DRF_DEF(_LWLTLC_RX, _DEBUG_TP_CNTR1_CTRL_0, _FLITFILTER, _DATA) |
        DRF_DEF(_LWLTLC_RX, _DEBUG_TP_CNTR1_CTRL_0, _VCSETFILTERMODE, _INIT));

    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_RX, _DEBUG_TP_CNTR_CTRL,
        DRF_NUM(_LWLTLC_RX, _DEBUG_TP_CNTR_CTRL, _ENRX0, 0x1) |
        DRF_NUM(_LWLTLC_RX, _DEBUG_TP_CNTR_CTRL, _ENRX1, 0x1));

    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_TX, _DEBUG_TP_CNTR0_CTRL_0,
        DRF_DEF(_LWLTLC_TX, _DEBUG_TP_CNTR0_CTRL_0, _UNIT, _BYTES)      |
        DRF_DEF(_LWLTLC_TX, _DEBUG_TP_CNTR0_CTRL_0, _FLITFILTER, _DATA) |
        DRF_DEF(_LWLTLC_TX, _DEBUG_TP_CNTR0_CTRL_0, _VCSETFILTERMODE, _INIT));

    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_TX, _DEBUG_TP_CNTR1_CTRL_0,
        DRF_DEF(_LWLTLC_TX, _DEBUG_TP_CNTR1_CTRL_0, _UNIT, _PACKETS)    |
        DRF_DEF(_LWLTLC_TX, _DEBUG_TP_CNTR1_CTRL_0, _FLITFILTER, _DATA) |
        DRF_DEF(_LWLTLC_TX, _DEBUG_TP_CNTR1_CTRL_0, _VCSETFILTERMODE, _INIT));

    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_TX, _DEBUG_TP_CNTR_CTRL,
        DRF_NUM(_LWLTLC_TX, _DEBUG_TP_CNTR_CTRL, _ENTX0, 0x1) |
        DRF_NUM(_LWLTLC_TX, _DEBUG_TP_CNTR_CTRL, _ENTX1, 0x1));
}

static void
_lwswitch_init_lwltlc_interrupts
(
    lwlink_link *link
)
{
    LwU32 val;
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 linkNum = link->linkNumber;

    // W1C any stale state.
    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_TX, _ERR_STATUS_0, 0xffffffff);
    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_RX, _ERR_STATUS_0, 0xffffffff);
    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_RX, _ERR_STATUS_1, 0xffffffff);

    val = DRF_DEF(_LWLTLC_TX, _ERR_LOG_EN_0, _TXHDRCREDITOVFERR, __PROD) |
        DRF_DEF(_LWLTLC_TX, _ERR_LOG_EN_0, _TXDATACREDITOVFERR, __PROD)  |
        DRF_DEF(_LWLTLC_TX, _ERR_LOG_EN_0, _TXDLCREDITOVFERR, __PROD)    |
        DRF_DEF(_LWLTLC_TX, _ERR_LOG_EN_0, _TXDLCREDITPARITYERR, __PROD) |
        DRF_DEF(_LWLTLC_TX, _ERR_LOG_EN_0, _TXRAMHDRPARITYERR, __PROD)   |
        DRF_DEF(_LWLTLC_TX, _ERR_LOG_EN_0, _TXRAMDATAPARITYERR, __PROD)  |
        DRF_DEF(_LWLTLC_TX, _ERR_LOG_EN_0, _TXUNSUPVCOVFERR, __PROD)     |
        DRF_DEF(_LWLTLC_TX, _ERR_LOG_EN_0, _TXSTOMPDET, __PROD)          |
        DRF_NUM(_LWLTLC_TX, _ERR_LOG_EN_0, _TXPOISONDET, 0)              |  // silent poison
        DRF_DEF(_LWLTLC_TX, _ERR_LOG_EN_0, _TARGETERR, __PROD)           |
        DRF_DEF(_LWLTLC_TX, _ERR_LOG_EN_0, _UNSUPPORTEDREQUESTERR, __PROD);
    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_TX, _ERR_LOG_EN_0, val);
    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_TX, _ERR_REPORT_EN_0, val);

    val = DRF_DEF(_LWLTLC_TX, _ERR_CONTAIN_EN_0, _TXHDRCREDITOVFERR, __PROD) |
        DRF_DEF(_LWLTLC_TX, _ERR_CONTAIN_EN_0, _TXDATACREDITOVFERR, __PROD)  |
        DRF_DEF(_LWLTLC_TX, _ERR_CONTAIN_EN_0, _TXDLCREDITOVFERR, __PROD)    |
        DRF_DEF(_LWLTLC_TX, _ERR_CONTAIN_EN_0, _TXDLCREDITPARITYERR, __PROD) |
        DRF_DEF(_LWLTLC_TX, _ERR_CONTAIN_EN_0, _TXRAMHDRPARITYERR, __PROD)   |
        DRF_NUM(_LWLTLC_TX, _ERR_CONTAIN_EN_0, _TXRAMDATAPARITYERR, 1)       |
        DRF_DEF(_LWLTLC_TX, _ERR_CONTAIN_EN_0, _TXUNSUPVCOVFERR, __PROD)     |
        DRF_DEF(_LWLTLC_TX, _ERR_CONTAIN_EN_0, _TXSTOMPDET, __PROD)          |
        DRF_DEF(_LWLTLC_TX, _ERR_CONTAIN_EN_0, _TXPOISONDET, __PROD)         | // should flow to destination
        DRF_NUM(_LWLTLC_TX, _ERR_CONTAIN_EN_0, _TARGETERR, 1)                |
        DRF_DEF(_LWLTLC_TX, _ERR_CONTAIN_EN_0, _UNSUPPORTEDREQUESTERR, __PROD);
    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_TX, _ERR_CONTAIN_EN_0, val);

    val = DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RXDLHDRPARITYERR, __PROD)      |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RXDLDATAPARITYERR, __PROD)       |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RXDLCTRLPARITYERR, __PROD)       |
        DRF_NUM(_LWLTLC_RX, _ERR_LOG_EN_0, _RXRAMDATAPARITYERR, 1)           |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RXRAMHDRPARITYERR, __PROD)       |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RXILWALIDAEERR, __PROD)          |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RXILWALIDBEERR, __PROD)          |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RXILWALIDADDRALIGNERR, __PROD)   |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RXPKTLENERR, __PROD)             |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RSVCMDENCERR, __PROD)            |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RSVDATLENENCERR, __PROD)         |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RSVADDRTYPEERR, __PROD)          |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RSVRSPSTATUSERR, __PROD)         |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RSVPKTSTATUSERR, __PROD)         |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RSVCACHEATTRPROBEREQERR, __PROD) |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RSVCACHEATTRPROBERSPERR, __PROD) |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _DATLENGTATOMICREQMAXERR, __PROD) |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _DATLENGTRMWREQMAXERR, __PROD)    |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _DATLENLTATRRSPMINERR, __PROD)    |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _ILWALIDCACHEATTRPOERR, __PROD)   |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _ILWALIDCRERR, __PROD)            |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RXRESPSTATUSTARGETERR, __PROD)   |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_0, _RXRESPSTATUSUNSUPPORTEDREQUESTERR, __PROD);
    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_RX, _ERR_LOG_EN_0, val);
    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_RX, _ERR_REPORT_EN_0, val);

    val = DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RXDLHDRPARITYERR, __PROD)      |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RXDLDATAPARITYERR, __PROD)       |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RXDLCTRLPARITYERR, __PROD)       |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RXRAMDATAPARITYERR, __PROD)      |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RXRAMHDRPARITYERR, __PROD)       |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RXILWALIDAEERR, __PROD)          |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RXILWALIDBEERR, __PROD)          |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RXILWALIDADDRALIGNERR, __PROD)   |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RXPKTLENERR, __PROD)             |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RSVCMDENCERR, __PROD)            |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RSVDATLENENCERR, __PROD)         |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RSVADDRTYPEERR, __PROD)          |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RSVRSPSTATUSERR, __PROD)         |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RSVPKTSTATUSERR, __PROD)         |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RSVCACHEATTRPROBEREQERR, __PROD) |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RSVCACHEATTRPROBERSPERR, __PROD) |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _DATLENGTATOMICREQMAXERR, __PROD) |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _DATLENGTRMWREQMAXERR, __PROD)    |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _DATLENLTATRRSPMINERR, __PROD)    |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _ILWALIDCACHEATTRPOERR, __PROD)   |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _ILWALIDCRERR, __PROD)            |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RXRESPSTATUSTARGETERR, __PROD)   |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_0, _RXRESPSTATUSUNSUPPORTEDREQUESTERR, __PROD);
    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_RX, _ERR_CONTAIN_EN_0, val);

    val = DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_1, _RXHDROVFERR, __PROD)             |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_1, _RXDATAOVFERR, __PROD)              |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_1, _STOMPDETERR, __PROD)               |
        DRF_NUM(_LWLTLC_RX, _ERR_LOG_EN_1, _RXPOISONERR, 0)                    | // silent poison
        DRF_NUM(_LWLTLC_RX, _ERR_LOG_EN_1, _CORRECTABLEINTERNALERR, 0)         | // HW not implemented
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_1, _RXUNSUPVCOVFERR, __PROD)           |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_1, _RXUNSUPLWLINKCREDITRELERR, __PROD) |
        DRF_DEF(_LWLTLC_RX, _ERR_LOG_EN_1, _RXUNSUPNCISOCCREDITRELERR, __PROD);
    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_RX, _ERR_LOG_EN_1, val);
    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_RX, _ERR_REPORT_EN_1, val);

    val = DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_1, _RXHDROVFERR, __PROD)             |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_1, _RXDATAOVFERR, __PROD)              |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_1, _STOMPDETERR, __PROD)               |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_1, _RXPOISONERR, __PROD)               |
        DRF_NUM(_LWLTLC_RX, _ERR_CONTAIN_EN_1, _CORRECTABLEINTERNALERR, 0)         |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_1, _RXUNSUPVCOVFERR, __PROD)           |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_1, _RXUNSUPLWLINKCREDITRELERR, __PROD) |
        DRF_DEF(_LWLTLC_RX, _ERR_CONTAIN_EN_1, _RXUNSUPNCISOCCREDITRELERR, __PROD);
    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_RX, _ERR_CONTAIN_EN_1, val);

    LWSWITCH_FLUSH_MMIO(device);
}

static void
_lwswitch_init_lwltlc_credits
(
    lwlink_link *link
)
{
    LwU32 val;
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 linkNum = link->linkNumber;

#define UPDATE_CREDITS(idx, vc)                                                              \
do {                                                                                         \
    val = DRF_DEF(_LWLTLC_RX, _CTRL_BUFFER_CREDITS_VC##vc, _HDRCREDITS, _INIT) |             \
          DRF_DEF(_LWLTLC_RX, _CTRL_BUFFER_CREDITS_VC##vc, _DATACREDITS, _INIT);             \
                                                                                             \
    LWSWITCH_LINK_WR32_SV10(device, idx, LWLTLC, _LWLTLC_RX, _CTRL_BUFFER_CREDITS_VC##vc, val);\
                                                                                             \
    val = DRF_DEF(_LWLTLC_TX, _CTRL_BUFFER_CREDITS_VC##vc, _HDRCREDITS, _INIT) |             \
          DRF_DEF(_LWLTLC_TX, _CTRL_BUFFER_CREDITS_VC##vc, _DATACREDITS, _INIT);             \
                                                                                             \
    LWSWITCH_LINK_WR32_SV10(device, idx, LWLTLC, _LWLTLC_TX, _CTRL_BUFFER_CREDITS_VC##vc, val);\
                                                                                             \
    val = DRF_DEF(_LWLTLC_TX, _CTRL_REMOTE_BUFFER_CREDIT_LIMIT_VC##vc, _LIMITHDR, __PROD) |  \
          DRF_DEF(_LWLTLC_TX, _CTRL_REMOTE_BUFFER_CREDIT_LIMIT_VC##vc, _LIMITDATA, __PROD);  \
                                                                                             \
    LWSWITCH_LINK_WR32_SV10(device, idx, LWLTLC, _LWLTLC_TX, _CTRL_REMOTE_BUFFER_CREDIT_LIMIT_VC##vc, val);\
} while(0)

    UPDATE_CREDITS(linkNum, 0);
    UPDATE_CREDITS(linkNum, 1);
    UPDATE_CREDITS(linkNum, 2);
    UPDATE_CREDITS(linkNum, 3);
    UPDATE_CREDITS(linkNum, 4);
    UPDATE_CREDITS(linkNum, 5);
    UPDATE_CREDITS(linkNum, 6);
    UPDATE_CREDITS(linkNum, 7);
}

static void
_lwswitch_init_lwltlc
(
    lwlink_link *link
)
{
    LwU32 val;
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 linkNum = link->linkNumber;

    // Disable compressed responses.  GPU supports them but LWSwitch doesn't.
    val = DRF_DEF(_LWLTLC_RX, _CTRL_LINK_CONFIG, _STORENFORWARD, _INIT) |
          DRF_DEF(_LWLTLC_RX, _CTRL_LINK_CONFIG, _ALLOWAEC, _INIT)      |
          DRF_NUM(_LWLTLC_RX, _CTRL_LINK_CONFIG, _ALLOWCR, 0x0);

    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_RX, _CTRL_LINK_CONFIG, val);

    // Disable PM. (Not a POR)
    val = DRF_NUM(_LWLTLC_TX, _PWRM_IC_SW_CTRL, _SOFTWAREDESIRED, 0) |
          DRF_NUM(_LWLTLC_TX, _PWRM_IC_SW_CTRL, _HARDWAREDISABLE, 1);

    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_TX, _PWRM_IC_SW_CTRL, val);

    val = DRF_NUM(_LWLTLC_RX, _PWRM_IC_SW_CTRL, _SOFTWAREDESIRED, 0) |
          DRF_NUM(_LWLTLC_RX, _PWRM_IC_SW_CTRL, _HARDWAREDISABLE, 1);

    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_RX, _PWRM_IC_SW_CTRL, val);

    // Init clock gating
    val = DRF_DEF(_LWLTLC_TX, _CTRL_CLOCK_GATING, _CG1_SLCG, __PROD);

    LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_TX, _CTRL_CLOCK_GATING, val);

    _lwswitch_init_lwltlc_interrupts(link);

    _lwswitch_init_lwltlc_credits(link);

    _lwswitch_init_lwltlc_debug_counters(link);
}

void
lwswitch_init_buffer_ready_sv10
(
    lwswitch_device *device,
    lwlink_link *link,
    LwBool bNportBufferReady
)
{
    LwU32 val;
    LwU32 linkNum = link->linkNumber;

    //
    // This bit is RW1S. Writes to this by SW can not clear this bit, only a
    // reset can clear this register. This regkey allows to skip setting it.
    //
    if (FLD_TEST_DRF(_SWITCH_REGKEY, _SKIP_BUFFER_READY, _TLC, _NO,
                     device->regkeys.skip_buffer_ready))
    {
        val = DRF_NUM(_LWLTLC_RX, _CTRL_BUFFER_READY, _BUFFERRDY, 0x1);
        LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_RX, _CTRL_BUFFER_READY, val);

        val = DRF_NUM(_LWLTLC_TX, _CTRL_BUFFER_READY, _BUFFERRDY, 0x1);
        LWSWITCH_LINK_WR32_SV10(device, linkNum, LWLTLC, _LWLTLC_TX, _CTRL_BUFFER_READY, val);

        LWSWITCH_FLUSH_MMIO(device);
    }

    // This can be cleared by SW, but we allow skipping it as well.
    if (bNportBufferReady &&
        FLD_TEST_DRF(_SWITCH_REGKEY, _SKIP_BUFFER_READY, _NPORT, _NO,
                     device->regkeys.skip_buffer_ready))
    {
        val = DRF_NUM(_NPORT, _CTRL_BUFFER_READY, _BUFFERRDY, 0x1);
        LWSWITCH_LINK_WR32_SV10(device, linkNum, NPORT, _NPORT, _CTRL_BUFFER_READY, val);

        LWSWITCH_FLUSH_MMIO(device);
    }
}

static LwlStatus
_lwswitch_init_link
(
    lwlink_link *link
)
{
    LwlStatus retval = LWL_SUCCESS;
    lwswitch_device *device = link->dev->pDevInfo;

    retval = _lwswitch_init_dl_pll(link);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    _lwswitch_init_dlpl(link);

    _lwswitch_init_lwltlc(link);

    // Note: buffer_rdy should be asserted last!
    lwswitch_init_buffer_ready(device, link, LW_TRUE);

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_poll_sublink_state_sv10
(
    lwswitch_device *device,
    lwlink_link     *link
)
{
    return LWL_ERR_NOT_IMPLEMENTED;
}

void
lwswitch_setup_link_loopback_mode_sv10
(
    lwswitch_device *device,
    LwU32            linkNumber
)
{
    // Not Implemented for SV10
}

void
lwswitch_reset_persistent_link_hw_state_sv10
(
    lwswitch_device *device,
    LwU32            linkNumber
)
{
    // Not Implemented for SV10
}

void
lwswitch_store_topology_information_sv10
(
    lwswitch_device *device,
    lwlink_link *link
)
{
    // Not Implemented for SV10
}

void
lwswitch_init_lpwr_regs_sv10
(
    lwlink_link *link
)
{
    // Not Implemented for SV10
}

LwBool
lwswitch_is_link_in_reset_sv10
(
    lwswitch_device *device,
    lwlink_link     *link
)
{
    return LW_FALSE;
}

void
lwswitch_apply_recal_settings_sv10
(
    lwswitch_device *device,
    lwlink_link *link
)
{
    // Not supported on SV10
    return;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
/* ADD NEW UNPUBLISHED CODE BELOW THIS LINE */

LwlStatus
lwswitch_corelib_enable_optical_maintenance_sv10
(
    lwlink_link *link,
    LwBool bTx
)
{
    return LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_corelib_set_optical_infinite_mode_sv10
(
    lwlink_link *link,
    LwBool bEnable
)
{
    return LWL_ERR_NOT_SUPPORTED;
}


LwlStatus
lwswitch_corelib_set_optical_iobist_sv10
(
    lwlink_link *link,
    LwBool bEnable
)
{
    return LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_corelib_set_optical_pretrain_sv10
(   
    lwlink_link *link,
    LwBool bTx,
    LwBool bEnable
)
{
    return LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_corelib_check_optical_pretrain_sv10
(
    lwlink_link *link,
    LwBool bTx,
    LwBool *bSuccess
)
{
    return LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_corelib_init_optical_links_sv10
(
    lwlink_link *link
)
{
    return LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_corelib_check_optical_eom_status_sv10
(
    lwlink_link *link,
    LwBool      *bEomLow
)
{
    return LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_corelib_set_optical_force_eq_sv10
(
    lwlink_link *link,
    LwBool bEnable
)
{
    return LWL_ERR_NOT_SUPPORTED;
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
