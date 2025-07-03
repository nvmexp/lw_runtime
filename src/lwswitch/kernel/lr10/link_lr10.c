/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2022 by LWPU Corporation.  All rights reserved.  All
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
#include "lr10/lr10.h"
#include "lr10/minion_lr10.h"
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#include "cci/cci_lwswitch.h"
#include "cci/cci_priv_lwswitch.h"
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#include "lwswitch/lr10/dev_lwldl_ip.h"
#include "lwswitch/lr10/dev_lwldl_ip_addendum.h"
#include "lwswitch/lr10/dev_minion_ip_addendum.h"
#include "lwswitch/lr10/dev_lwlipt_lnk_ip.h"
#include "lwswitch/lr10/dev_lwlphyctl_ip.h"
#include "lwswitch/lr10/dev_lwltlc_ip.h"
#include "lwswitch/lr10/dev_minion_ip.h"
#include "lwswitch/lr10/dev_trim.h"
#include "lwswitch/lr10/dev_pri_ringstation_sys.h"
#include "lwswitch/lr10/dev_lwlperf_ip.h"
#include "lwswitch/lr10/dev_lwlipt_ip.h"
#include "lwswitch/lr10/dev_nport_ip.h"

void
lwswitch_setup_link_loopback_mode_lr10
(
    lwswitch_device *device,
    LwU32            linkNumber
)
{
    lwlink_link *link;
    LW_STATUS status;

    link = lwswitch_get_link(device, linkNumber);

    if ((link == NULL) ||
        !LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber) ||
        (linkNumber >= LWSWITCH_LWLINK_MAX_LINKS))
    {
        return;
    }

    if (device->link[link->linkNumber].nea)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Setting NEA on link %d\n",
            __FUNCTION__, link->linkNumber);

        status = lwswitch_minion_send_command(device, link->linkNumber,
                    LW_MINION_LWLINK_DL_CMD_COMMAND_SETNEA, 0);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: SETNEA CMD failed for link %d.\n",
                __FUNCTION__, link->linkNumber);
        }
    }

    if (device->link[link->linkNumber].ned)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Setting NED on link %d\n",
            __FUNCTION__, link->linkNumber);

        // setting NEDR
        status = lwswitch_minion_send_command(device, link->linkNumber,
                    LW_MINION_LWLINK_DL_CMD_COMMAND_SETNEDR, 0);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: SETNEDR CMD failed for link %d.\n",
                __FUNCTION__, link->linkNumber);
        }
        
        // setting NEDW
        status = lwswitch_minion_send_command(device, link->linkNumber,
                    LW_MINION_LWLINK_DL_CMD_COMMAND_SETNEDW, 0);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: SETNEDW CMD failed for link %d.\n",
                __FUNCTION__, link->linkNumber);
        }
    }
}

static LW_STATUS
_lwswitch_ioctrl_setup_link_plls_lr10
(
    lwlink_link *link
)
{
    LW_STATUS status = LW_OK;
    LwU32     linkId, tempRegVal;
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;

    lwswitch_device *device = link->dev->pDevInfo;
    linkId = link->linkNumber;

    if (IS_EMULATION(device))
    {
        LWSWITCH_PRINT(device, ERROR,"Skipping PLL init on emulation. \n");
        return status;
    }

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1MSEC_IN_NS * 400, &timeout);

    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;
        tempRegVal = LWSWITCH_LINK_RD32_LR10(device, linkId, LWLIPT_LNK , _LWLIPT_LNK , _CTRL_CLK_CTRL);

        if (FLD_TEST_DRF(_LWLIPT_LNK, _CTRL_CLK_CTRL, _PLL_PWR_STS, _ON, tempRegVal))
            break;

        lwswitch_os_sleep(1);
    } while (keepPolling == LW_TRUE);

    if (FLD_TEST_DRF(_LWLIPT_LNK, _CTRL_CLK_CTRL, _PLL_PWR_STS, _OFF, tempRegVal))
    {
        LWSWITCH_PRINT(device, ERROR,
                  "PLL_PWR_STS did not turn _ON for linkId = 0x%x!!\n", linkId);
        return LW_ERR_TIMEOUT;
    }

    // Request Minion to setup the LWLink clocks
    status = lwswitch_minion_send_command(device, linkId,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_TXCLKSWITCH_PLL, 0); 
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
                  "Error sending TXCLKSWITCH_PLL command to MINION. Link = %d\n", linkId);
        LWSWITCH_ASSERT_INFO(LW_ERR_LWLINK_CLOCK_ERROR, LWBIT32(link->linkNumber), TXCLKSWITCH_PLL_ERROR);
        return LW_ERR_LWLINK_CLOCK_ERROR;
    }

    // Poll for the links to switch to LWLink clocks
    lwswitch_timeout_create(LWSWITCH_INTERVAL_1MSEC_IN_NS * 400, &timeout);

    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;
        tempRegVal = LWSWITCH_LINK_RD32_LR10(device, linkId, LWLIPT_LNK , _LWLIPT_LNK , _CTRL_CLK_CTRL);
        if (FLD_TEST_DRF(_LWLIPT_LNK, _CTRL_CLK_CTRL, _TXCLK_STS, _PLL_CLK, tempRegVal))
            break;

        lwswitch_os_sleep(1);
    } while (keepPolling == LW_TRUE);

    if (!FLD_TEST_DRF(_LWLIPT_LNK, _CTRL_CLK_CTRL, _TXCLK_STS, _PLL_CLK, tempRegVal))
    {
        // Print the links for which we were unable to switch to PLL clock
        LWSWITCH_PRINT(device, ERROR,
                  "TXCLK_STS did not switch to _PLL_CLOCK for linkId = 0x%x!!\n", linkId);
        return LW_ERR_TIMEOUT;
    }

    return status;
}

LwBool
lwswitch_is_link_in_reset_lr10
(
    lwswitch_device *device,
    lwlink_link     *link
)
{
    LwU32 val;
    val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber,
            LWLIPT_LNK, _LWLIPT_LNK, _RESET_RSTSEQ_LINK_RESET);

    return (FLD_TEST_DRF(_LWLIPT_LNK, _RESET_RSTSEQ_LINK_RESET, _LINK_RESET_STATUS,
                _ASSERTED, val)) ? LW_TRUE : LW_FALSE;
}

LwlStatus
lwswitch_poll_sublink_state_lr10
(
    lwswitch_device *device,
    lwlink_link *link
)
{
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;
    LwU32 val;
    LwBool bPreSiPlatform = (IS_RTLSIM(device) || IS_EMULATION(device) || IS_FMODEL(device));

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1MSEC_IN_NS * (bPreSiPlatform ? 2000: 200), &timeout);

    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _SUBLINK_CHANGE);

        if (FLD_TEST_DRF(_LWLDL_TOP, _SUBLINK_CHANGE, _STATUS, _FAULT, val))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s : Fault while changing sublink state (%s):(%s).\n",
                __FUNCTION__, device->name, link->linkName);
            return -LWL_ERR_ILWALID_STATE;
        }

        if (FLD_TEST_DRF(_LWLDL_TOP, _SUBLINK_CHANGE, _STATUS, _DONE, val))
        {
            break;
        }

        lwswitch_os_sleep(1);
    }
    while (keepPolling);

    if ((!FLD_TEST_DRF(_LWLDL_TOP, _SUBLINK_CHANGE, _STATUS, _DONE, val)))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s : Timeout while waiting sublink state (%s):(%s).\n",
            __FUNCTION__, device->name, link->linkName);
        return -LWL_ERR_GENERIC;
    }

    return LWL_SUCCESS;
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

    status = _lwswitch_ioctrl_setup_link_plls_lr10(link);
    if (status != LW_OK){
        return status;
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

LwU32
lwswitch_get_sublink_width_lr10
(
    lwswitch_device *device,
    LwU32            linkNumber
)
{
    return LWSWITCH_NUM_LANES_LR10;
}

void
lwswitch_init_dlpl_interrupts_lr10
(
    lwlink_link *link
)
{
    lwswitch_device *device            = link->dev->pDevInfo;
    LwU32            linkNumber        = link->linkNumber;
    LwU32            crcShortRegkeyVal = device->regkeys.crc_bit_error_rate_short;
    LwU32            crcLongRegkeyVal  = device->regkeys.crc_bit_error_rate_long;
    LwU32            intrRegVal;
    LwU32            crcRegVal;
    LwU32            shortRateMask;
    LwU32            longRateMask;

    ct_assert(DRF_BASE(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_THRESHOLD_MAN)   == 
              DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_SHORT_THRESHOLD_MAN));
    ct_assert(DRF_EXTENT(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_THRESHOLD_MAN) == 
              DRF_EXTENT(LW_LWLDL_RX_ERROR_RATE_CTRL_SHORT_THRESHOLD_MAN));
    ct_assert(DRF_BASE(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_THRESHOLD_EXP)   == 
              DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_SHORT_THRESHOLD_EXP));
    ct_assert(DRF_EXTENT(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_THRESHOLD_EXP) == 
              DRF_EXTENT(LW_LWLDL_RX_ERROR_RATE_CTRL_SHORT_THRESHOLD_EXP));
    ct_assert(DRF_BASE(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_TIMESCALE_MAN)   == 
              DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_SHORT_TIMESCALE_MAN));
    ct_assert(DRF_EXTENT(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_TIMESCALE_MAN) == 
              DRF_EXTENT(LW_LWLDL_RX_ERROR_RATE_CTRL_SHORT_TIMESCALE_MAN));
    ct_assert(DRF_BASE(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_TIMESCALE_EXP)   == 
              DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_SHORT_TIMESCALE_EXP));
    ct_assert(DRF_EXTENT(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_TIMESCALE_EXP) == 
              DRF_EXTENT(LW_LWLDL_RX_ERROR_RATE_CTRL_SHORT_TIMESCALE_EXP));
    
    ct_assert(DRF_BASE(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_THRESHOLD_MAN)    == 
             (DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_THRESHOLD_MAN)   - 
              DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_THRESHOLD_MAN)));
    ct_assert(DRF_EXTENT(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_THRESHOLD_MAN)  == 
             (DRF_EXTENT(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_THRESHOLD_MAN) - 
              DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_THRESHOLD_MAN)));
    ct_assert(DRF_BASE(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_THRESHOLD_EXP)    == 
             (DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_THRESHOLD_EXP)   - 
              DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_THRESHOLD_MAN)));
    ct_assert(DRF_EXTENT(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_THRESHOLD_EXP)  == 
             (DRF_EXTENT(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_THRESHOLD_EXP) - 
              DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_THRESHOLD_MAN)));
    ct_assert(DRF_BASE(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_TIMESCALE_MAN)    == 
             (DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_TIMESCALE_MAN)   - 
              DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_THRESHOLD_MAN)));
    ct_assert(DRF_EXTENT(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_TIMESCALE_MAN)  == 
             (DRF_EXTENT(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_TIMESCALE_MAN) - 
              DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_THRESHOLD_MAN)));
    ct_assert(DRF_BASE(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_TIMESCALE_EXP)    == 
             (DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_TIMESCALE_EXP)   - 
              DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_THRESHOLD_MAN)));
    ct_assert(DRF_EXTENT(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_TIMESCALE_EXP)  == 
             (DRF_EXTENT(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_TIMESCALE_EXP) - 
              DRF_BASE(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_THRESHOLD_MAN)));

    // W1C any stale state.
    LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TOP, _INTR, 0xffffffff);
    LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TOP, _INTR_SW2, 0xffffffff);

    // Stall tree routes to INTR_A which is connected to LWLIPT fatal tree
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    // https://wiki.lwpu.com/gpuhwdept/index.php/LWLink/DL_programming_guide#Interrupts
#endif
    LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TOP, _INTR_STALL_EN,
              DRF_DEF(_LWLDL_TOP, _INTR_STALL_EN, _TX_REPLAY, _DISABLE)               |
              DRF_DEF(_LWLDL_TOP, _INTR_STALL_EN, _TX_RECOVERY_SHORT, _DISABLE)       |
              DRF_DEF(_LWLDL_TOP, _INTR_STALL_EN, _LTSSM_FAULT_UP, _ENABLE)           |
              DRF_DEF(_LWLDL_TOP, _INTR_STALL_EN, _TX_FAULT_RAM, _ENABLE)             |
              DRF_DEF(_LWLDL_TOP, _INTR_STALL_EN, _TX_FAULT_INTERFACE, _ENABLE)       |
              DRF_DEF(_LWLDL_TOP, _INTR_STALL_EN, _TX_FAULT_SUBLINK_CHANGE, _DISABLE) |
              DRF_DEF(_LWLDL_TOP, _INTR_STALL_EN, _RX_FAULT_SUBLINK_CHANGE, _DISABLE) |
              DRF_DEF(_LWLDL_TOP, _INTR_STALL_EN, _RX_FAULT_DL_PROTOCOL, _ENABLE)     |
              DRF_DEF(_LWLDL_TOP, _INTR_STALL_EN, _RX_SHORT_ERROR_RATE, _DISABLE)     |
              DRF_DEF(_LWLDL_TOP, _INTR_STALL_EN, _RX_LONG_ERROR_RATE, _DISABLE)      |
              DRF_DEF(_LWLDL_TOP, _INTR_STALL_EN, _RX_ILA_TRIGGER, _DISABLE)          |
              DRF_DEF(_LWLDL_TOP, _INTR_STALL_EN, _RX_CRC_COUNTER, _DISABLE)          |
              DRF_DEF(_LWLDL_TOP, _INTR_STALL_EN, _LTSSM_PROTOCOL, _DISABLE)          |
              DRF_DEF(_LWLDL_TOP, _INTR_STALL_EN, _MINION_REQUEST, _DISABLE));

    // NONSTALL -> NONFATAL
    LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TOP, _INTR_NONSTALL_EN,
              DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _TX_REPLAY, _DISABLE)               |
              DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _TX_RECOVERY_SHORT, _DISABLE)       |
              DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _LTSSM_FAULT_UP, _DISABLE)          |
              DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _TX_FAULT_RAM, _DISABLE)            |
              DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _TX_FAULT_INTERFACE, _DISABLE)      |
              DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _TX_FAULT_SUBLINK_CHANGE, _DISABLE) |
              DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _RX_FAULT_SUBLINK_CHANGE, _DISABLE) |
              DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _RX_FAULT_DL_PROTOCOL, _DISABLE)    |
              DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _RX_SHORT_ERROR_RATE, _DISABLE)     |
              DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _RX_LONG_ERROR_RATE, _DISABLE)      |
              DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _RX_ILA_TRIGGER, _DISABLE)          |
              DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _RX_CRC_COUNTER, _ENABLE)           |
              DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _LTSSM_PROTOCOL, _DISABLE)          |
              DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _MINION_REQUEST, _DISABLE));

    intrRegVal = LWSWITCH_LINK_RD32_LR10(device, linkNumber, LWLDL, 
                                         _LWLDL_TOP, _INTR_NONSTALL_EN);
    crcRegVal  = LWSWITCH_LINK_RD32_LR10(device, linkNumber, LWLDL, 
                                         _LWLDL_RX, _ERROR_RATE_CTRL);

    // Enable RX error rate short interrupt if the regkey is set
    if (crcShortRegkeyVal != LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_OFF)
    {
        shortRateMask = DRF_SHIFTMASK(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_THRESHOLD_MAN)     |
                            DRF_SHIFTMASK(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_THRESHOLD_EXP) |
                            DRF_SHIFTMASK(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_TIMESCALE_MAN) |
                            DRF_SHIFTMASK(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_SHORT_TIMESCALE_EXP);
        
        intrRegVal |= DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _RX_SHORT_ERROR_RATE, _ENABLE);
        crcRegVal  &= ~shortRateMask;
        crcRegVal  |= crcShortRegkeyVal;
    }
    // Enable RX error rate long interrupt if the regkey is set
    if (crcLongRegkeyVal != LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_OFF)
    {
        longRateMask = DRF_SHIFTMASK(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_THRESHOLD_MAN)      |
                            DRF_SHIFTMASK(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_THRESHOLD_EXP) |
                            DRF_SHIFTMASK(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_TIMESCALE_MAN) |
                            DRF_SHIFTMASK(LW_SWITCH_REGKEY_CRC_BIT_ERROR_RATE_LONG_TIMESCALE_EXP);

        intrRegVal |= DRF_DEF(_LWLDL_TOP, _INTR_NONSTALL_EN, _RX_LONG_ERROR_RATE, _ENABLE);
        crcRegVal  &= ~longRateMask;
        crcRegVal  |= crcLongRegkeyVal << DRF_SHIFT(LW_LWLDL_RX_ERROR_RATE_CTRL_LONG_THRESHOLD_MAN);
    }

    LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, 
                            _LWLDL_TOP, _INTR_NONSTALL_EN, intrRegVal);
    LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, 
                            _LWLDL_RX, _ERROR_RATE_CTRL, crcRegVal);
}

static void
_lwswitch_disable_dlpl_interrupts
(
    lwlink_link *link
)
{
    lwswitch_device *device    = link->dev->pDevInfo;
    LwU32           linkNumber = link->linkNumber;

    LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TOP, _INTR_STALL_EN,    0x0);
    LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TOP, _INTR_NONSTALL_EN, 0x0);
}

void
lwswitch_store_topology_information_lr10
(
    lwswitch_device *device,
    lwlink_link *link
)
{
    LwU32            tempval;

    link->bInitnegotiateConfigGood = LW_TRUE;
    link->remoteSid = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLIPT_LNK,
                                             _LWLIPT_LNK, _TOPOLOGY_REMOTE_CHIP_SID_HI);
    link->remoteSid = link->remoteSid << 32;
    link->remoteSid |= LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLIPT_LNK,
                                             _LWLIPT_LNK, _TOPOLOGY_REMOTE_CHIP_SID_LO);

    tempval = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLIPT_LNK, _LWLIPT_LNK, _TOPOLOGY_REMOTE_LINK_INFO);
    link->remoteLinkId = DRF_VAL(_LWLIPT_LNK, _TOPOLOGY_REMOTE_LINK_INFO, _LINK_NUMBER, tempval);

    link->localSid = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLIPT,
                                            _LWLIPT_COMMON, _TOPOLOGY_LOCAL_CHIP_SID_HI);
    link->localSid = link->localSid << 32;
    link->localSid |= LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLIPT,
                                             _LWLIPT_COMMON, _TOPOLOGY_LOCAL_CHIP_SID_LO);

    tempval = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLIPT_LNK,
                                             _LWLIPT_LNK, _TOPOLOGY_REMOTE_CHIP_TYPE);

    // Update the remoteDeviceType with LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_TYPE values.
    switch(tempval)
    {
        case LW_LWLIPT_LNK_TOPOLOGY_REMOTE_CHIP_TYPE_TYPE_LW3P0AMP:
            link->remoteDeviceType = LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_TYPE_GPU;
        break;
        case LW_LWLIPT_LNK_TOPOLOGY_REMOTE_CHIP_TYPE_TYPE_LW3P0LRK:
            link->remoteDeviceType = LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_TYPE_SWITCH;
        break;
        default:
            link->remoteDeviceType = LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_TYPE_NONE;
        break;
    }
}

void
lwswitch_init_lpwr_regs_lr10
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 linkNum = link->linkNumber;
    LwU32 tempRegVal, icLimit, fbIcInc, lpIcInc, fbIcDec, lpIcDec, lpEntryThreshold;
    LwU32 lpExitThreshold;
    LwU8  softwareDesired, hardwareDisable;
    LwBool bLpEnable;

    if (device->regkeys.enable_pm == LW_SWITCH_REGKEY_ENABLE_PM_NO)
    {
        return;
    }

    //
    // Power Management threshold settings
    // These settings are lwrrently being hard coded.
    // They will be parsed from the VBIOS LWLink LPWR table once bug 2767390 is
    // implemented
    //

    // IC Limit
    icLimit = 16110000;

    tempRegVal = 0;
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_TX_LNK, _PWRM_IC_LIMIT, _LIMIT, icLimit,
        tempRegVal);
    LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_TX_LNK, _PWRM_IC_LIMIT,
        tempRegVal);

    tempRegVal = 0;
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_RX_LNK,_PWRM_IC_LIMIT, _LIMIT, icLimit,
        tempRegVal);
    LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_RX_LNK, _PWRM_IC_LIMIT,
        tempRegVal);

    //IC Inc
    fbIcInc = 1;
    lpIcInc = 1;

    tempRegVal = 0;
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_TX_LNK, _PWRM_IC_INC, _FBINC, fbIcInc, tempRegVal);
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_TX_LNK, _PWRM_IC_INC, _LPINC, lpIcInc, tempRegVal);
    LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_TX_LNK, _PWRM_IC_INC,
        tempRegVal);

    tempRegVal = 0;
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_RX_LNK, _PWRM_IC_INC, _FBINC, fbIcInc, tempRegVal);
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_RX_LNK, _PWRM_IC_INC, _LPINC, lpIcInc, tempRegVal);
    LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_RX_LNK, _PWRM_IC_INC,
        tempRegVal);

    //IC Dec
    fbIcDec = 1;
    lpIcDec = 65535;

    tempRegVal = 0;
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_TX_LNK, _PWRM_IC_DEC, _FBDEC, fbIcDec, tempRegVal);
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_TX_LNK, _PWRM_IC_DEC, _LPDEC, lpIcDec, tempRegVal);
    LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_TX_LNK, _PWRM_IC_DEC,
        tempRegVal);

    tempRegVal = 0;
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_RX_LNK, _PWRM_IC_DEC, _FBDEC, fbIcDec,   tempRegVal);
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_RX_LNK, _PWRM_IC_DEC, _LPDEC, lpIcDec, tempRegVal);
    LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_RX_LNK, _PWRM_IC_DEC,
        tempRegVal);

    //IC Enter Threshold
    lpEntryThreshold = 16110000;

    tempRegVal = 0;
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_TX_LNK, _PWRM_IC_LP_ENTER_THRESHOLD, _THRESHOLD, lpEntryThreshold, tempRegVal);
    LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_TX_LNK, _PWRM_IC_LP_ENTER_THRESHOLD,
        tempRegVal);

    tempRegVal = 0;
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_RX_LNK, _PWRM_IC_LP_ENTER_THRESHOLD, _THRESHOLD, lpEntryThreshold, tempRegVal);
    LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_RX_LNK, _PWRM_IC_LP_ENTER_THRESHOLD,
        tempRegVal);

    //IC Exit Threshold
    lpExitThreshold = 16044465;

    tempRegVal = 0;
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_TX_LNK, _PWRM_IC_LP_EXIT_THRESHOLD, _THRESHOLD, lpExitThreshold, tempRegVal);
    LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_TX_LNK, _PWRM_IC_LP_EXIT_THRESHOLD,
        tempRegVal);

    tempRegVal = 0;
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_RX_LNK, _PWRM_IC_LP_EXIT_THRESHOLD, _THRESHOLD, lpExitThreshold, tempRegVal);
    LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_RX_LNK, _PWRM_IC_LP_EXIT_THRESHOLD,
        tempRegVal);

    //LP Entry Enable
    bLpEnable = LW_TRUE;
    softwareDesired = (bLpEnable) ? 0x1 : 0x0;
    hardwareDisable = (bLpEnable) ? 0x0 : 0x1;

    tempRegVal = LWSWITCH_LINK_RD32_LR10(device, linkNum, LWLTLC, _LWLTLC_TX_LNK, _PWRM_IC_SW_CTRL);
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_TX_LNK, _PWRM_IC_SW_CTRL, _SOFTWAREDESIRED,
                    softwareDesired, tempRegVal);
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_TX_LNK, _PWRM_IC_SW_CTRL, _HARDWAREDISABLE,
                    hardwareDisable, tempRegVal);
    LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_TX_LNK, _PWRM_IC_SW_CTRL,
                    tempRegVal);

    tempRegVal = LWSWITCH_LINK_RD32_LR10(device, linkNum, LWLTLC, _LWLTLC_RX_LNK, _PWRM_IC_SW_CTRL);
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_RX_LNK, _PWRM_IC_SW_CTRL, _SOFTWAREDESIRED,
                    softwareDesired, tempRegVal);
    tempRegVal = FLD_SET_DRF_NUM(_LWLTLC_RX_LNK, _PWRM_IC_SW_CTRL, _HARDWAREDISABLE,
                    hardwareDisable, tempRegVal);
    LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_RX_LNK, _PWRM_IC_SW_CTRL,
                    tempRegVal);
}


void
lwswitch_init_buffer_ready_lr10
(
    lwswitch_device *device,
    lwlink_link *link,
    LwBool bNportBufferReady
)
{
    LwU32 val;
    LwU32 linkNum = link->linkNumber;

    if (FLD_TEST_DRF(_SWITCH_REGKEY, _SKIP_BUFFER_READY, _TLC, _NO,
                     device->regkeys.skip_buffer_ready))
    {
        val = DRF_NUM(_LWLTLC_RX_SYS, _CTRL_BUFFER_READY, _BUFFERRDY, 0x1);
        LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_RX_SYS, _CTRL_BUFFER_READY, val);
        val = DRF_NUM(_LWLTLC_TX_SYS, _CTRL_BUFFER_READY, _BUFFERRDY, 0x1);
        LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_TX_SYS, _CTRL_BUFFER_READY, val);
    }

    if (bNportBufferReady &&
        FLD_TEST_DRF(_SWITCH_REGKEY, _SKIP_BUFFER_READY, _NPORT, _NO,
                     device->regkeys.skip_buffer_ready))
    {
        val = DRF_NUM(_NPORT, _CTRL_BUFFER_READY, _BUFFERRDY, 0x1);
        LWSWITCH_LINK_WR32_LR10(device, linkNum, NPORT, _NPORT, _CTRL_BUFFER_READY, val);
    }
}

#if !defined(LW_MODS)
static void
_lwswitch_configure_reserved_throughput_counters
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 linkNum = link->linkNumber;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLTLC, link->linkNumber))
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
    LWSWITCH_LINK_WR32_IDX_LR10(device, linkNum, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, 0,
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _UNIT, _FLITS)      |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _DATA) |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _VCSETFILTERMODE, _INIT));

    // Tx0 config
    LWSWITCH_LINK_WR32_IDX_LR10(device, linkNum, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, 0,
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _UNIT, _FLITS)      |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _DATA) |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _VCSETFILTERMODE, _INIT));

    // Rx2 config
    LWSWITCH_LINK_WR32_IDX_LR10(device, linkNum, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, 2,
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _UNIT, _FLITS)      |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _HEAD) |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _AE) |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _BE) |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _DATA) |
        DRF_DEF(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL_0, _VCSETFILTERMODE, _INIT));

    // Tx2 config
    LWSWITCH_LINK_WR32_IDX_LR10(device, linkNum, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, 2,
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _UNIT, _FLITS)      |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _HEAD) |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _AE) |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _BE) |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _FLITFILTER, _DATA) |
        DRF_DEF(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL_0, _VCSETFILTERMODE, _INIT));

    // Enable Rx for counters 0, 2
    LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL,
        DRF_NUM(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL, _ENRX0, 0x1) |
        DRF_NUM(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL, _ENRX2, 0x1));

    // Enable Tx for counters 0, 2
    LWSWITCH_LINK_WR32_LR10(device, linkNum, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL,
        DRF_NUM(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL, _ENTX0, 0x1) |
        DRF_NUM(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL, _ENTX2, 0x1));
}
#endif  //!defined(LW_MODS)

static LwlStatus
_lwswitch_init_link_post_active
(
    lwlink_link *link,
    LwU32       flags
)
{
    LwlStatus       status = LWL_SUCCESS;
    lwswitch_device *device = link->dev->pDevInfo;

    lwswitch_init_lpwr_regs(link);
    status = lwswitch_request_tl_link_state_lr10(link,
                LW_LWLIPT_LNK_CTRL_LINK_STATE_REQUEST_REQUEST_ACTIVE,
                flags == LWLINK_STATE_CHANGE_SYNC);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    // Note: buffer_rdy should be asserted last!
    lwswitch_init_buffer_ready(device, link, LW_TRUE);

    return status;
}

static void
_lwswitch_power_down_link_plls
(
    lwlink_link *link
)
{
    LwlStatus       status = LWL_SUCCESS;
    lwswitch_device *device = link->dev->pDevInfo;

    if (IS_EMULATION(device))
    {
        LWSWITCH_PRINT(device, ERROR,"Skipping PLL init on emulation. \n");
        return;
    }

    status = lwswitch_minion_send_command(device, link->linkNumber,
        LW_MINION_LWLINK_DL_CMD_COMMAND_TXCLKSWITCH_ALT, 0);

    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: TXCLKSWITCH_ALT CMD failed for link %d.\n",
            __FUNCTION__, link->linkNumber);
        return;
    }

    return;
}

LwlStatus
lwswitch_corelib_add_link_lr10
(
    lwlink_link *link
)
{
    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_remove_link_lr10
(
    lwlink_link *link
)
{
    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_set_dl_link_mode_lr10
(
    lwlink_link *link,
    LwU64 mode,
    LwU32 flags
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32           val;
    LwU32           link_state;
    LwU32           *seedData;
    LwlStatus       status = LWL_SUCCESS;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    switch (mode)
    {
        case LWLINK_LINKSTATE_SAFE:
        {
            // check if link is in reset
            if (lwswitch_is_link_in_reset(device, link))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: link #%d is still in reset, cannot change link state\n",
                    __FUNCTION__, link->linkNumber);
                return LWL_ERR_ILWALID_STATE;
            }

            val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _LINK_STATE);
            link_state = DRF_VAL(_LWLDL_TOP, _LINK_STATE, _STATE, val);

            if (link_state == LW_LWLDL_TOP_LINK_STATE_STATE_SWCFG)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : Link is already in Safe mode for (%s).\n",
                    __FUNCTION__, link->linkName);
                break;
            }
            else if (link_state == LW_LWLDL_TOP_LINK_STATE_STATE_HWCFG)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : Link already transitioning to Safe mode for (%s).\n",
                    __FUNCTION__, link->linkName);
                break;
            }

            LWSWITCH_PRINT(device, INFO,
                "LWRM: %s : Changing Link state to Safe for (%s):(%s).\n",
                __FUNCTION__, device->name, link->linkName);

            if (link_state == LW_LWLDL_TOP_LINK_STATE_STATE_INIT)
            {
                val = 0;
                val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _LINK_CHANGE);
                val = FLD_SET_DRF(_LWLDL_TOP, _LINK_CHANGE, _NEWSTATE, _HWCFG, val);
                val = FLD_SET_DRF(_LWLDL_TOP, _LINK_CHANGE, _OLDSTATE_MASK, _DONTCARE, val);
                val = FLD_SET_DRF(_LWLDL_TOP, _LINK_CHANGE, _ACTION, _LTSSM_CHANGE, val);
                LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _LINK_CHANGE, val);
            }
            else if (link_state == LW_LWLDL_TOP_LINK_STATE_STATE_ACTIVE)
            {
                val = 0;
                val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _LINK_CHANGE);
                val = FLD_SET_DRF(_LWLDL_TOP, _LINK_CHANGE, _NEWSTATE, _SWCFG, val);
                val = FLD_SET_DRF(_LWLDL_TOP, _LINK_CHANGE, _OLDSTATE_MASK, _DONTCARE, val);
                val = FLD_SET_DRF(_LWLDL_TOP, _LINK_CHANGE, _ACTION, _LTSSM_CHANGE, val);
                LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _LINK_CHANGE, val);
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
            // check if link is in reset
            if (lwswitch_is_link_in_reset(device, link))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: link #%d is still in reset, cannot change link state\n",
                    __FUNCTION__, link->linkNumber);
                return LWL_ERR_ILWALID_STATE;
            }

            val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _LINK_STATE);
            link_state = DRF_VAL(_LWLDL_TOP, _LINK_STATE, _STATE, val);

            if (link_state == LW_LWLDL_TOP_LINK_STATE_STATE_ACTIVE)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : Link is already in Active mode (%s).\n",
                    __FUNCTION__, link->linkName);
                break;
            }
            else if (link_state == LW_LWLDL_TOP_LINK_STATE_STATE_INIT)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : Link cannot be taken from INIT state to"
                    " Active mode for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                return -LWL_ERR_ILWALID_STATE;
            }
            else if (link_state == LW_LWLDL_TOP_LINK_STATE_STATE_SWCFG)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : Changing Link state to Active for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);

                val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _LINK_CHANGE);
                val = FLD_SET_DRF(_LWLDL_TOP, _LINK_CHANGE, _NEWSTATE, _ACTIVE, val);
                val = FLD_SET_DRF(_LWLDL_TOP, _LINK_CHANGE, _OLDSTATE_MASK, _DONTCARE, val);
                val = FLD_SET_DRF(_LWLDL_TOP, _LINK_CHANGE, _ACTION, _LTSSM_CHANGE, val);
                LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _LINK_CHANGE, val);
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
            _lwswitch_power_down_link_plls(link);

            // save training seeds used for ALT training
            if (device->regkeys.minion_cache_seeds == LW_SWITCH_REGKEY_MINION_CACHE_SEEDS_ENABLE)
            {
                // reach back into corelib and get the seedData buffer to store into
                seedData = link->seedData;

                // poll minion to get seed data and store into corelib buffer
                status = lwswitch_minion_save_seed_data_lr10(device, link->linkNumber, seedData);

                if (status != LWL_SUCCESS)
                {
                    LWSWITCH_PRINT(device, ERROR, "%s : Error Storing seed data for link (%s):(%s)\n", 
                                    __FUNCTION__, device->name, link->linkName);
                }
            }

            if (lwswitch_lib_notify_client_events(device,
                        LWSWITCH_DEVICE_EVENT_PORT_DOWN) != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR, "%s: Failed to notify PORT_DOWN event\n",
                             __FUNCTION__);
            }

            break;
        }

        case LWLINK_LINKSTATE_RESET:
        {
            break;
        }

        case LWLINK_LINKSTATE_ENABLE_PM:
        {
            if (device->regkeys.enable_pm == LW_SWITCH_REGKEY_ENABLE_PM_YES)
            {
                status = lwswitch_minion_send_command(device, link->linkNumber,
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
            if (device->regkeys.enable_pm == LW_SWITCH_REGKEY_ENABLE_PM_YES)
            {
                status = lwswitch_minion_send_command(device, link->linkNumber,
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
            status = _lwswitch_init_link_post_active(link, flags);
            if (status != LWL_SUCCESS)
            {
                return status;
            }

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
            status = lwswitch_minion_send_command(device, link->linkNumber,
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
            status = lwswitch_minion_send_command(device, link->linkNumber,
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
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
            //
            // TODO: Remove these once pre-training is moved to FM for intranode as well.
            // Tracked in bug 3288612
            //
            if (cciIsLinkManaged(device, link->linkNumber))
            {
                if (device->pCci->preTrainingFailed)
                {
                    return LW_ERR_LWLINK_CONFIGURATION_ERROR;
                }

                if (!device->pCci->preTrainingComplete)
                {
                    status = cciOpticalPretrain(device);
                    if (status != LWL_SUCCESS)
                    {
                        LWSWITCH_PRINT(device, ERROR,
                            "%s : Optical pre-training failed on (%s).\n",
                            __FUNCTION__, device->name);

                        device->pCci->preTrainingFailed = LW_TRUE;
                        return LW_ERR_LWLINK_CONFIGURATION_ERROR;
                    }
                    device->pCci->preTrainingComplete = LW_TRUE;
                }

                LWSWITCH_PRINT(device, INFO,
                    "%s : Skipping Initphase1 on optical link (%s):(%s)\n",
                    __FUNCTION__, device->name, link->linkName);
   
                link->bRxDetected = LW_TRUE;
                link->bTxCommonModeFail = LW_FALSE;
                link->bOpticalLink = LW_TRUE;  
                return LWL_SUCCESS;
            }
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
            // training seed restoration for ALT LWLINK training
            if (device->regkeys.minion_cache_seeds == LW_SWITCH_REGKEY_MINION_CACHE_SEEDS_ENABLE)
            {
                // get seed data cached in corelib
                seedData = link->seedData;

                // restore seed data back into minion before INITPHASE1
                status = lwswitch_minion_restore_seed_data_lr10(device, link->linkNumber, seedData);

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

            // After INITPHASE1, apply NEA setting
            lwswitch_setup_link_loopback_mode(device, link->linkNumber);
            break;
        }

        case LWLINK_LINKSTATE_INITOPTIMIZE:
        {
            status = lwswitch_minion_send_command(device, link->linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_INITOPTIMIZE, 0);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : INITOPTIMIZE failed for link (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                return status;
            }

            break;
        }

        case LWLINK_LINKSTATE_POST_INITOPTIMIZE:
        {
            // Poll for TRAINING_GOOD
            status  = lwswitch_minion_get_initoptimize_status_lr10(device, link->linkNumber);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                            "%s Error polling for INITOPTIMIZE TRAINING_GOOD. Link (%s):(%s)\n",
                            __FUNCTION__, device->name, link->linkName);
                LWSWITCH_ASSERT_INFO(LW_ERR_LWLINK_TRAINING_ERROR, LWBIT32(link->linkNumber), INITOPTIMIZE_ERROR);
                return LW_ERR_LWLINK_TRAINING_ERROR;
            }

            // Send INITTL DLCMD
            status = lwswitch_minion_send_command(device, link->linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_INITTL, 0);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : INITTL failed for link (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                LWSWITCH_ASSERT_INFO(LW_ERR_LWLINK_TRAINING_ERROR, LWBIT32(link->linkNumber), INITTL_ERROR);
                return LW_ERR_LWLINK_TRAINING_ERROR;
            }

            break;
        }

        case LWLINK_LINKSTATE_INITNEGOTIATE:
        {
            status = lwswitch_minion_send_command(device, link->linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_INITNEGOTIATE, 0);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : INITNEGOTIATE failed for link (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                return status;
            }

            break;
        }

        case LWLINK_LINKSTATE_POST_INITNEGOTIATE:
        {
            // Poll for CONFIG_GOOD
            status  = lwswitch_minion_get_initnegotiate_status_lr10(device, link->linkNumber);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                            "%s Error polling for INITNEGOTIATE CONFIG_GOOD. Link (%s):(%s)\n",
                            __FUNCTION__, device->name, link->linkName);
                LWSWITCH_ASSERT_INFO(LW_ERR_LWLINK_CONFIGURATION_ERROR,
                    LWBIT32(link->linkNumber), INITNEGOTIATE_ERROR);
                return LW_ERR_LWLINK_CONFIGURATION_ERROR;
            }
            else
            {
                lwswitch_store_topology_information(device, link);
            }

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

LwlStatus
lwswitch_corelib_get_dl_link_mode_lr10
(
    lwlink_link *link,
    LwU64 *mode
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 link_state;
    LwU32 val = 0;

    *mode = LWLINK_LINKSTATE_OFF;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber))
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

    val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _LINK_STATE);

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

void
lwswitch_corelib_get_uphy_load_lr10
(
    lwlink_link *link,
    LwBool *bUnlocked
)
{
    *bUnlocked = LW_FALSE;
}

LwlStatus
lwswitch_corelib_set_tl_link_mode_lr10
(
    lwlink_link *link,
    LwU64 mode,
    LwU32 flags
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwlStatus       status = LWL_SUCCESS;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    switch (mode)
    {
        case LWLINK_LINKSTATE_RESET:
        {
            // perform TL reset
            LWSWITCH_PRINT(device, INFO,
                "%s: Performing TL Reset on link %d\n",
                __FUNCTION__, link->linkNumber);

            status = lwswitch_request_tl_link_state_lr10(link,
                LW_LWLIPT_LNK_CTRL_LINK_STATE_REQUEST_REQUEST_RESET,
                flags == LWLINK_STATE_CHANGE_SYNC);

            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: LwLink Reset has failed for link %d\n",
                    __FUNCTION__, link->linkNumber);
                return status;
            }
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

LwlStatus
lwswitch_corelib_get_tl_link_mode_lr10
(
    lwlink_link *link,
    LwU64 *mode
)
{
#if defined(INCLUDE_LWLINK_LIB)

    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 link_state;
    LwU32 val = 0;

    *mode = LWLINK_LINKSTATE_OFF;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber))
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
    val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLIPT_LNK,
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
            *mode = LWLINK_LINKSTATE_HS;
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

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_set_tx_mode_lr10
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

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber))
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

    val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TX, _SLSM_STATUS_TX);

    tx_sublink_state = DRF_VAL(_LWLDL_TX, _SLSM_STATUS_TX, _PRIMARY_STATE, val);

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

        case LWLINK_SUBLINK_STATE_TX_COMMON_MODE_DISABLE:
        {
            // Not applicable for LW IP
            break;
        }

        case LWLINK_SUBLINK_STATE_TX_DATA_READY:
        {
            status = lwswitch_minion_send_command(device, link->linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_INITDLPL, 0);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: INITLWLDL CMD failed for link %d.\n",
                    __FUNCTION__, link->linkNumber);
                LWSWITCH_ASSERT_INFO(LW_ERR_LWLINK_INIT_ERROR,
                    LWBIT32(link->linkNumber), INITDLPL_ERROR);
                return LW_ERR_LWLINK_INIT_ERROR;
            }

            status = lwswitch_minion_send_command(device, link->linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_INITLANEENABLE, 0);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: INITLANEENABLE CMD failed for link %d.\n",
                    __FUNCTION__, link->linkNumber);
                LWSWITCH_ASSERT_INFO(LW_ERR_LWLINK_INIT_ERROR, LWBIT32(link->linkNumber), INITLANEENABLE_ERROR);
                return LW_ERR_LWLINK_INIT_ERROR;
            }

#if defined(LW_MODS)
            // Break if SSG has a request for per-link break after LWLDL init.
            if (FLD_TEST_DRF(_SWITCH_REGKEY, _SSG_CONTROL, _BREAK_AFTER_DLPL_INIT, _YES,
                device->regkeys.ssg_control))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: SSG Control: Break after LWLDL Init on link %d\n",
                    __FUNCTION__, link->linkNumber);
                LWSWITCH_ASSERT(0);
            }
#endif  //defined(LW_MODS)

            break;
        }

        case LWLINK_SUBLINK_STATE_TX_PRBS_EN:
        {
            // Not needed with ALT
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
            // Not needed with ALT
            break;
        }

        case LWLINK_SUBLINK_STATE_TX_SAFE:
        {
            if (tx_sublink_state == LW_LWLDL_TX_SLSM_STATUS_TX_PRIMARY_STATE_SAFE)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : TX already in Safe mode for  (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                break;
            }

            LWSWITCH_PRINT(device, INFO,
                  "%s : Changing TX sublink state to Safe mode for (%s):(%s).\n",
                  __FUNCTION__, device->name, link->linkName);

            val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _SUBLINK_CHANGE);
            val = FLD_SET_DRF(_LWLDL_TOP, _SUBLINK_CHANGE, _NEWSTATE, _SAFE, val);
            val = FLD_SET_DRF(_LWLDL_TOP, _SUBLINK_CHANGE, _SUBLINK, _TX, val);
            val = FLD_SET_DRF(_LWLDL_TOP, _SUBLINK_CHANGE, _ACTION, _SLSM_CHANGE, val);
            LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _SUBLINK_CHANGE, val);

            status = lwswitch_poll_sublink_state(device, link);
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
            if (tx_sublink_state == LW_LWLDL_TX_SLSM_STATUS_TX_PRIMARY_STATE_OFF)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : TX already OFF (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                break;
            }
            else if (tx_sublink_state == LW_LWLDL_TX_SLSM_STATUS_TX_PRIMARY_STATE_HS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : TX cannot be taken from HS to OFF directly for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                return -LWL_ERR_GENERIC;
            }

            LWSWITCH_PRINT(device, INFO,
                "%s : Changing TX sublink state to OFF for (%s):(%s).\n",
                __FUNCTION__, device->name, link->linkName);

            val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _SUBLINK_CHANGE);
            val = FLD_SET_DRF(_LWLDL_TOP, _SUBLINK_CHANGE, _COUNTDOWN, _IMMEDIATE, val);
            val = FLD_SET_DRF(_LWLDL_TOP, _SUBLINK_CHANGE, _NEWSTATE, _OFF, val);
            val = FLD_SET_DRF(_LWLDL_TOP, _SUBLINK_CHANGE, _SUBLINK, _TX, val);
            val = FLD_SET_DRF(_LWLDL_TOP, _SUBLINK_CHANGE, _ACTION, _SLSM_CHANGE, val);
            LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _SUBLINK_CHANGE, val);

            status = lwswitch_poll_sublink_state(device, link);
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
lwswitch_corelib_get_tx_mode_lr10
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

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber))
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

    data = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TX, _SLSM_STATUS_TX);

    tx_sublink_state = DRF_VAL(_LWLDL_TX, _SLSM_STATUS_TX, _PRIMARY_STATE, data);

    // Return LWLINK_SUBLINK_SUBSTATE_TX_STABLE for sub-state
    *subMode = LWLINK_SUBLINK_SUBSTATE_TX_STABLE;

    switch (tx_sublink_state)
    {
        case LW_LWLDL_TX_SLSM_STATUS_TX_PRIMARY_STATE_EIGHTH:
            *mode = LWLINK_SUBLINK_STATE_TX_SINGLE_LANE;
            break;

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
            *mode = LWLINK_SUBLINK_STATE_TX_OFF;
            break;

        default:
            *mode = LWLINK_SUBLINK_STATE_TX_OFF;
            break;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_set_rx_mode_lr10
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
    LwU32 delay_ns;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber))
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


    val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _SLSM_STATUS_RX);

    rx_sublink_state = DRF_VAL(_LWLDL_RX, _SLSM_STATUS_RX, _PRIMARY_STATE, val);

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
        case LWLINK_SUBLINK_STATE_RX_HS:
            break;

        case LWLINK_SUBLINK_STATE_RX_SAFE:
            break;

        case LWLINK_SUBLINK_STATE_RX_OFF:
        {
            if (rx_sublink_state == LW_LWLDL_RX_SLSM_STATUS_RX_PRIMARY_STATE_OFF)
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s : RX already OFF (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                break;
            }
            else if (rx_sublink_state == LW_LWLDL_RX_SLSM_STATUS_RX_PRIMARY_STATE_HS)
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

            val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _SUBLINK_CHANGE);
            val = FLD_SET_DRF(_LWLDL_TOP, _SUBLINK_CHANGE, _COUNTDOWN, _IMMEDIATE, val);
            val = FLD_SET_DRF(_LWLDL_TOP, _SUBLINK_CHANGE, _OLDSTATE_MASK, _DONTCARE, val);
            val = FLD_SET_DRF(_LWLDL_TOP, _SUBLINK_CHANGE, _NEWSTATE, _OFF, val);
            val = FLD_SET_DRF(_LWLDL_TOP, _SUBLINK_CHANGE, _SUBLINK, _RX, val);

            // When changing RX sublink state use FORCE, otherwise it will fault.
            val = FLD_SET_DRF(_LWLDL_TOP, _SUBLINK_CHANGE, _ACTION, _SLSM_FORCE, val);
            LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _SUBLINK_CHANGE, val);

            LWSWITCH_PRINT(device, INFO,
                "%s : LW_LWLDL_TOP_SUBLINK_CHANGE = 0x%08x\n", __FUNCTION__, val);

            status = lwswitch_poll_sublink_state(device, link);
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
            // Enable RXCAL in CFG_CTL_6, Delay 200us (bug 2551877), and check CFG_STATUS_0 for RXCAL_DONE=1.
            val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLPHYCTL_COMMON, _CFG_CTL_6);
            val = FLD_SET_DRF(_LWLPHYCTL_COMMON, _CFG_CTL_6, _RXCAL , _ON, val);
            LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLPHYCTL_COMMON, _CFG_CTL_6, val);

            if (IS_FMODEL(device) || IS_EMULATION(device) || IS_RTLSIM(device))
            {
                delay_ns = LWSWITCH_INTERVAL_1SEC_IN_NS;
            }
            else
            {
                delay_ns = 200 * LWSWITCH_INTERVAL_1USEC_IN_NS;
            }

            LWSWITCH_NSEC_DELAY(delay_ns);

            val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLPHYCTL_COMMON, _CFG_STATUS_0);
            if (!FLD_TEST_DRF_NUM(_LWLPHYCTL_COMMON, _CFG_STATUS_0, _RXCAL_DONE, 0x1, val))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: Timeout while waiting for RXCAL_DONE on link %d.\n",
                    __FUNCTION__, link->linkNumber);
                return -LWL_ERR_GENERIC;
            }
            break;
        }

        case LWLINK_SUBLINK_STATE_RX_INIT_TERM:
        {
            // Ilwoke MINION routine to enable RX Termination
            status = lwswitch_minion_set_rx_term_lr10(device, link->linkNumber);

            if (status != LW_OK)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s : Error while setting RX INIT_TERM for (%s):(%s).\n",
                    __FUNCTION__, device->name, link->linkName);
                return status;
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
lwswitch_corelib_get_rx_mode_lr10
(
    lwlink_link *link,
    LwU64 *mode,
    LwU32 *subMode
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 rx_sublink_state;
    LwU32 data = 0;

    *mode = LWLINK_SUBLINK_STATE_RX_OFF;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber))
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

    data = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _SLSM_STATUS_RX);

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

        case LW_LWLDL_RX_SLSM_STATUS_RX_PRIMARY_STATE_EIGHTH:
            *mode = LWLINK_SUBLINK_STATE_RX_SINGLE_LANE;
            break;

        case LW_LWLDL_RX_SLSM_STATUS_RX_PRIMARY_STATE_OFF:
            *mode = LWLINK_SUBLINK_STATE_RX_OFF;
            break;

        default:
            *mode = LWLINK_SUBLINK_STATE_RX_OFF;
            break;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_set_rx_detect_lr10
(
    lwlink_link *link,
    LwU32 flags
)
{
    LwlStatus status;
    lwswitch_device *device = link->dev->pDevInfo;

    status = lwswitch_minion_send_command(device, link->linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_TURING_RXDET, 0);

    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Set RXDET failed for link %d.\n",
            __FUNCTION__, link->linkNumber);
        return status;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_get_rx_detect_lr10
(
    lwlink_link *link
)
{
    LwlStatus status;
    lwswitch_device *device = link->dev->pDevInfo;

    status = lwswitch_minion_get_rxdet_status_lr10(device, link->linkNumber);

    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Get RXDET failed for link %d.\n",
            __FUNCTION__, link->linkNumber);
        return status;
    }
    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_write_discovery_token_lr10
(
    lwlink_link *link,
    LwU64 token
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 command;
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    LWSWITCH_PRINT(device, INFO,
        "%s: Sending token 0x%016llx to (%s).\n",
        __FUNCTION__, token, link->linkName);

    command = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TX, _R4TX_COMMAND);
    if (FLD_TEST_DRF(_LWLDL_TX, _R4TX_COMMAND, _READY, _NO, command))
    {
        LWSWITCH_PRINT(device, ERROR,
            "LWRM: %s: Unable to process packet injection request for link %s."
            " HW not ready for new request.\n",
            __FUNCTION__, link->linkName);
        return -LWL_ERR_GENERIC;
    }

    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TX, _R4TX_WDATA0, LwU64_LO32(token));
    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TX, _R4TX_WDATA1, LwU64_HI32(token));

    command = FLD_SET_DRF(_LWLDL_TX, _R4TX_COMMAND, _REQUEST,  _WRITE,       command);
    command = FLD_SET_DRF(_LWLDL_TX, _R4TX_COMMAND, _COMPLETE, _READY_CLEAR, command);

    // SW is expected to use address 0x1 for topology detection
    command = FLD_SET_DRF(_LWLDL_TX, _R4TX_COMMAND, _WADDR,    _SC,          command);

    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TX, _R4TX_COMMAND, command);

    if (IS_FMODEL(device) || IS_EMULATION(device) || IS_RTLSIM(device))
    {
        lwswitch_timeout_create(100 * LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);
    }
    else
    {
        lwswitch_timeout_create(10 * LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);
    }

    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        command = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TX, _R4TX_COMMAND);
        if (FLD_TEST_DRF(_LWLDL_TX, _R4TX_COMMAND, _COMPLETE, _READY_CLEAR, command))
        {
            break;
        }

        lwswitch_os_sleep(1);
    }
    while (keepPolling);

    if (!FLD_TEST_DRF(_LWLDL_TX, _R4TX_COMMAND, _COMPLETE, _READY_CLEAR, command))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Timeout sending discovery token = 0x%016llx to (%s).\n",
            __FUNCTION__, token, link->linkName);
        return -LWL_ERR_GENERIC;
    }

    LWSWITCH_PRINT(device, INFO,
        "%s: Discovery token = 0x%016llx sent for (%s).\n",
        __FUNCTION__, token, link->linkName);

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_read_discovery_token_lr10
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
    LwBool           keepPolling;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    command = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _R4LOCAL_COMMAND);
    if (FLD_TEST_DRF(_LWLDL_RX, _R4LOCAL_COMMAND, _READY, _NO, command))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Unable to read AN0 packet for link %s."
            " HW not ready for new request.\n",
            __FUNCTION__, link->linkName);
        return -LWL_ERR_GENERIC;
    }

    command = FLD_SET_DRF(_LWLDL_RX, _R4LOCAL_COMMAND, _REQUEST,  _READ,        command);
    command = FLD_SET_DRF(_LWLDL_RX, _R4LOCAL_COMMAND, _COMPLETE, _READY_CLEAR, command);

    // SW is expected to use address 0x1 for topology detection
    command = FLD_SET_DRF(_LWLDL_RX, _R4LOCAL_COMMAND, _RADDR,    _SC,          command);

    // Issue the command to read the value
    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _R4LOCAL_COMMAND, command);

    if (IS_FMODEL(device) || IS_EMULATION(device) || IS_RTLSIM(device))
    {
        lwswitch_timeout_create(50 * LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);
    }
    else
    {
        lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    }

    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        command = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _R4LOCAL_COMMAND);
        if (FLD_TEST_DRF(_LWLDL_RX, _R4LOCAL_COMMAND, _COMPLETE, _READY_CLEAR, command))
        {
            break;
        }

        lwswitch_os_sleep(1);
    }
    while (keepPolling);

    if (!FLD_TEST_DRF(_LWLDL_RX, _R4LOCAL_COMMAND, _COMPLETE, _READY_CLEAR, command))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Timeout reading discovery token from (%s).\n",
            __FUNCTION__, link->linkName);
        return -LWL_ERR_GENERIC;
    }

    // Read the value using little endian (aligned to data reg names)
    value_lo = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _R4LOCAL_RDATA0);
    value_hi = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _R4LOCAL_RDATA1);

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
    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _R4LOCAL_RDATA0, 0x0);
    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _R4LOCAL_RDATA1, 0x0);

    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _R4LOCAL_WDATA0, 0x0);
    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _R4LOCAL_WDATA1, 0x0);

    // Send write command to clear out register.
    command = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _R4LOCAL_COMMAND);
    if (FLD_TEST_DRF(_LWLDL_RX, _R4LOCAL_COMMAND, _READY, _NO, command))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Unable to read AN0 packet for link %s."
            " HW not ready for new request.\n",
            __FUNCTION__, link->linkName);
        return -LWL_ERR_GENERIC;
    }

    command = FLD_SET_DRF(_LWLDL_RX, _R4LOCAL_COMMAND, _REQUEST, _WRITE, command);
    command = FLD_SET_DRF(_LWLDL_RX, _R4LOCAL_COMMAND, _COMPLETE, _READY_CLEAR, command);

    // SW is expected to use address 0x1 for topology detection
    command = FLD_SET_DRF(_LWLDL_RX, _R4LOCAL_COMMAND, _WADDR, _SC, command);

    // Issue the command to write (clear) the value
    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _R4LOCAL_COMMAND, command);

    lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);

    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        command = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _R4LOCAL_COMMAND);
        if (FLD_TEST_DRF(_LWLDL_RX, _R4LOCAL_COMMAND, _COMPLETE, _READY_CLEAR, command))
        {
            break;
        }

        if (keepPolling == LW_FALSE)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout reading discovery token from (%s).\n",
                __FUNCTION__, link->linkName);
            return -LWL_ERR_GENERIC;
        }

        lwswitch_os_sleep(1);
    }
    while (keepPolling);

    if (!FLD_TEST_DRF(_LWLDL_RX, _R4LOCAL_COMMAND, _COMPLETE, _READY_CLEAR, command))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Timeout reading discovery token from (%s).\n",
            __FUNCTION__, link->linkName);
        return -LWL_ERR_GENERIC;
    }

    return LWL_SUCCESS;
}

void
lwswitch_corelib_training_complete_lr10
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

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LWSWITCH_PRINT(device, INFO,
                "%s: Printing grading values after training complete.\n",
                __FUNCTION__); 

    cciPrintGradingValues(device, 0, link->linkNumber);
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
}

LwlStatus
lwswitch_wait_for_tl_request_ready_lr10
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;
    LwU32     linkRequest;
#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
    LwU32 linkStatus, linkErr;
#endif

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLIPT_LNK, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_BAD_ARGS;
    }

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1MSEC_IN_NS * 400, &timeout);

    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        linkRequest = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber,
                LWLIPT_LNK , _LWLIPT_LNK , _CTRL_LINK_STATE_REQUEST);

        if (FLD_TEST_DRF_NUM(_LWLIPT_LNK, _CTRL_LINK_STATE_REQUEST, _READY, 1, linkRequest))
        {
            return LWL_SUCCESS;
        }

        lwswitch_os_sleep(1);
    }
    while(keepPolling);

    //
    // LWSWITCH_PRINT is not defined for release builds,
    // so this keeps compiler happy
    //
#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
    linkStatus  = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber,
            LWLIPT_LNK , _LWLIPT_LNK , _CTRL_LINK_STATE_STATUS);
    linkErr     = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber,
            LWLIPT_LNK , _LWLIPT_LNK , _ERR_STATUS_0);
#endif

    LWSWITCH_PRINT(device, ERROR,
        "%s: Timeout waiting for TL link state ready on link #%d! "
              "LW_LWLIPT_LNK_CTRL_LINK_STATE_REQUEST = 0x%x, "
              "LW_LWLIPT_LNK_CTRL_LINK_STATE_STATUS = 0x%x, "
              "LW_LWLIPT_LNK_ERR_STATUS_0 = 0x%x\n",
        __FUNCTION__, link->linkNumber, linkRequest, linkStatus, linkErr);

    return -LWL_ERR_GENERIC;
}

LwlStatus
lwswitch_request_tl_link_state_lr10
(
    lwlink_link *link,
    LwU32        tlLinkState,
    LwBool       bSync
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwlStatus status = LWL_SUCCESS;
    LwU32 linkStatus;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLIPT_LNK, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return -LWL_UNBOUND_DEVICE;
    }

    // Wait for the TL link state register to report ready
    status = lwswitch_wait_for_tl_request_ready_lr10(link);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    // Request RESET state through CTRL_LINK_STATE_REQUEST
    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber,
            LWLIPT_LNK, _LWLIPT_LNK, _CTRL_LINK_STATE_REQUEST,
            DRF_NUM(_LWLIPT_LNK, _CTRL_LINK_STATE_REQUEST, _REQUEST, tlLinkState));

    if (bSync)
    {
        // Wait for the TL link state register to complete
        status = lwswitch_wait_for_tl_request_ready_lr10(link);
        if (status != LWL_SUCCESS)
        {
            return status;
        }

        // Check for state requested
        linkStatus  = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber,
                LWLIPT_LNK , _LWLIPT_LNK , _CTRL_LINK_STATE_STATUS);

        if (DRF_VAL(_LWLIPT_LNK, _CTRL_LINK_STATE_STATUS, _LWRRENTLINKSTATE, linkStatus) !=
                    tlLinkState)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: TL link state request to state 0x%x for link #%d did not complete!\n",
                __FUNCTION__, tlLinkState, link->linkNumber);
            return -LWL_ERR_GENERIC;
        }
    }

    return status;

}

void
lwswitch_exelwte_unilateral_link_shutdown_lr10
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link #%d invalid\n",
            __FUNCTION__, link->linkNumber);
        return;
    }

    //
    // Perform unilateral shutdown
    // This follows "Unilateral variant" from
    // LWLink 3.x Shutdown (confluence page ID: 164573291)
    //
    // Status is explicitly ignored here since we are required to soldier-on
    // in this scenario
    //
    lwswitch_corelib_set_dl_link_mode_lr10(link, LWLINK_LINKSTATE_DISABLE_PM, 0);
    lwswitch_corelib_set_dl_link_mode_lr10(link, LWLINK_LINKSTATE_DISABLE_ERR_DETECT, 0);
    lwswitch_corelib_set_dl_link_mode_lr10(link, LWLINK_LINKSTATE_LANE_DISABLE, 0);
    lwswitch_corelib_set_dl_link_mode_lr10(link, LWLINK_LINKSTATE_OFF, 0);
}

void 
lwswitch_reset_persistent_link_hw_state_lr10
(
    lwswitch_device *device,
    LwU32            linkNumber
)
{
    // Not Implemented for LR10
}

void
lwswitch_apply_recal_settings_lr10
(
    lwswitch_device *device,
    lwlink_link *link
)
{
    // Not supported on LR10
    return;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
/* ADD NEW UNPUBLISHED CODE BELOW THIS LINE */

LwlStatus
lwswitch_cci_initialization_sequence_lr10
(
    lwswitch_device *device,
    LwU32 linkNumber
)
{
    LwlStatus    status;
    LwU32        val;
    lwlink_link link;
    lwlink_device dev;

    link.linkNumber = linkNumber;
    link.dev = &dev;
    link.dev->pDevInfo = device;

    // Perform INITPHASE1
    status = lwswitch_minion_send_command(device, linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_INITPHASE1, 0);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s : LW_MINION_LWLINK_DL_CMD_COMMAND_INITPHY failed on link %d.\n",
            __FUNCTION__, linkNumber);
    }

    // Ilwoke MINION routine to enable RX Termination
    status = lwswitch_minion_set_rx_term_lr10(device, linkNumber);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s : lwswitch_minion_set_rx_term_lr10 failed on link %d.\n",
            __FUNCTION__, linkNumber);
    }

    // SET RX detect
    status = lwswitch_corelib_set_rx_detect_lr10(&link, 0);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Set RXDET failed for link %d.\n",
            __FUNCTION__, linkNumber);
    }

    // GET RX DETECT
    status = lwswitch_corelib_get_rx_detect_lr10(&link);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Get RXDET failed for link %d.\n",
            __FUNCTION__, linkNumber);
    }

    // Enable Common mode on  Tx
    status = _lwswitch_init_dl_pll(&link);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s:Failed to enable common mode for link %d.\n",
            __FUNCTION__, linkNumber);
    }

    // Put all Rx's in RXCAL
    val = LWSWITCH_LINK_RD32_LR10(device, linkNumber, LWLDL, _LWLPHYCTL_COMMON, _CFG_CTL_6);
    val = FLD_SET_DRF(_LWLPHYCTL_COMMON, _CFG_CTL_6, _RXCAL , _ON, val);
    LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLPHYCTL_COMMON, _CFG_CTL_6, val);

    LWSWITCH_NSEC_DELAY(200 * LWSWITCH_INTERVAL_1USEC_IN_NS);

    val = LWSWITCH_LINK_RD32_LR10(device, linkNumber, LWLDL, _LWLPHYCTL_COMMON, _CFG_STATUS_0);
    if (!FLD_TEST_DRF_NUM(_LWLPHYCTL_COMMON, _CFG_STATUS_0, _RXCAL_DONE, 0x1, val))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Timeout while waiting for RXCAL_DONE on link %d.\n",
            __FUNCTION__, linkNumber);
    }

    // Set Data Ready and Enable
    status = lwswitch_minion_send_command(device, linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_INITDLPL, 0);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
                 "%s : LW_MINION_LWLINK_DL_CMD_COMMAND_INITDLPL failed on link %d.\n",
                __FUNCTION__, linkNumber);
    }

    status = lwswitch_minion_send_command(device, linkNumber,
                        LW_MINION_LWLINK_DL_CMD_COMMAND_INITLANEENABLE, 0);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
                 "%s : LW_MINION_LWLINK_DL_CMD_COMMAND_INITLANEENABLE failed on link %d.\n",
                __FUNCTION__, linkNumber);
    }

    return LWL_SUCCESS;
}

/*
 * These settings must be moved to VBIOS. Remove this function once moved to VBIOS.
 */
LwlStatus
lwswitch_cci_setup_optical_links_lr10
(
    lwswitch_device *device,
    LwU64 linkMask
)
{
    LwU32 linkId;
    LwU32 val;

    FOR_EACH_INDEX_IN_MASK(64, linkId, linkMask)
    {
        val = LWSWITCH_LINK_RD32_LR10(device, linkId, LWLIPT_LNK , _LWLIPT_LNK , _CTRL_SYSTEM_LINK_CHANNEL_CTRL2);
        val = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL2, _RESTORE_PHY_TRAINING_PARAMS, _DISABLE, val);
        LWSWITCH_LINK_WR32_LR10(device, linkId, LWLIPT_LNK, _LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL2, val);

        val = LWSWITCH_LINK_RD32_LR10(device, linkId, LWLIPT_LNK, _LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL);
        val = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, _LINE_CODE_MODE, _PAM4, val);
        val = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, _AC_DC_MODE, _AC, val);
        val = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, _BLOCK_CODE_MODE, _ECC88_ENABLED, val);
        val = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, _TXTRAIN_FOM_FORMAT, _FOMC, val);
        val = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, _TXTRAIN_OPTIMIZATION_ALGORITHM, _A0, val);
        val = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, _TXTRAIN_ADJUSTMENT_ALGORITHM, _B0, val);
        val = FLD_SET_DRF_NUM(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, _TXTRAIN_MINIMUM_TRAIN_TIME_MANTISSA, 5, val);
        val = FLD_SET_DRF_NUM(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, _TXTRAIN_MINIMUM_TRAIN_TIME_EXPONENT, 4, val);
        LWSWITCH_LINK_WR32_LR10(device, linkId, LWLIPT_LNK, _LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, val);

        val = LWSWITCH_LINK_RD32_LR10(device, linkId, LWLIPT_LNK, _LWLIPT_LNK, _CTRL_SYSTEM_LINK_CLK_CTRL);
        val = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CLK_CTRL, _REFERENCE_CLOCK_MODE, _NON_COMMON_SS, val);
        val = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CLK_CTRL, _LINE_RATE, _53_12500_GBPS, val);
        LWSWITCH_LINK_WR32_LR10(device, linkId, LWLIPT_LNK, _LWLIPT_LNK, _CTRL_SYSTEM_LINK_CLK_CTRL, val);

        val = LWSWITCH_LINK_RD32_LR10(device, linkId, LWLIPT_LNK, _LWLIPT_LNK, _CTRL_SYSTEM_LINK_AN1_CTRL);
        val = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_AN1_CTRL, _PWRM_SL_ENABLE, _DISABLE, val);
        val = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_AN1_CTRL, _PWRM_L2_ENABLE, _DISABLE, val);
        LWSWITCH_LINK_WR32_LR10(device, linkId, LWLIPT_LNK, _LWLIPT_LNK, _CTRL_SYSTEM_LINK_AN1_CTRL, val);

    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_cci_enable_iobist_lr10
(
    lwswitch_device *device,
    LwU32 linkNumber,
    LwBool bEnable
)
{
    LwU32 val;

    if (bEnable)
    {
        val = LWSWITCH_LINK_RD32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _CONFIG);
        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _CONFIG, _CFGCLKGATEEN, _ENABLE, val);
        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _CONFIG, _PRBSALT, _NRZ, val);
        LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _CONFIG,  val);

        val = LWSWITCH_LINK_RD32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _SKIPCOMINSERTERGEN_2);
        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _SKIPCOMINSERTERGEN_2, _MASK_SKIP_OUT, _INIT, val);
        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _SKIPCOMINSERTERGEN_2, _MASK_COM_OUT, _INIT, val);
        LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _SKIPCOMINSERTERGEN_2,  val);

        val = LWSWITCH_LINK_RD32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _SKIPCOMINSERTERGEN_1);
        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _SKIPCOMINSERTERGEN_1, _SKIP_SYMBOL, _SYMBOL, val);
        LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _SKIPCOMINSERTERGEN_1,  val);

        val = LWSWITCH_LINK_RD32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _SKIPCOMINSERTERGEN);
        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _SKIPCOMINSERTERGEN, _COM_SYMBOL, _SYMBOL, val);
        LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _SKIPCOMINSERTERGEN,  val);

        val = LWSWITCH_LINK_RD32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _SKIPCOMINSERTERGEN_2);
        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _SKIPCOMINSERTERGEN_2, _SEND_DATA_OUT, _INIT, val);
        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _SKIPCOMINSERTERGEN_2, _RESET_WORD_CNT_OUT, _COUNT, val);
        LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _SKIPCOMINSERTERGEN_2,  val);

        val = LWSWITCH_LINK_RD32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _CONFIGREG);
        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _CONFIGREG, _TX_BIST_EN_IN, _ENABLE, val);
        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _CONFIGREG, _DISABLE_WIRED_ENABLE_IN, _ENABLE, val);
        LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _CONFIGREG, val);

        val = LWSWITCH_LINK_RD32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _CONFIG);
        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _CONFIG, _DPG_PRBSSEEDLD, _ENABLE, val);
        LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _CONFIG,  val);

        lwswitch_os_sleep(5);

        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _CONFIG, _DPG_PRBSSEEDLD, _INIT, val);
        LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _CONFIG,  val);

        val = LWSWITCH_LINK_RD32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _CONFIG);
        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _CONFIG, _STARTTEST, _ENABLE, val);
        LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _CONFIG,  val);
    }
    else
    {
        val = LWSWITCH_LINK_RD32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _CONFIG);
        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _CONFIG, _STARTTEST, _INIT, val);
        LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _CONFIG,  val);

        val = LWSWITCH_LINK_RD32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _CONFIGREG);
        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _CONFIGREG, _DISABLE_WIRED_ENABLE_IN, _INIT, val);
        LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _CONFIGREG, val);

        val = LWSWITCH_LINK_RD32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _CONFIGREG);
        val = FLD_SET_DRF(_LWLDL_TXIOBIST, _CONFIGREG, _TX_BIST_EN_IN, _INIT, val);
        LWSWITCH_LINK_WR32_LR10(device, linkNumber, LWLDL, _LWLDL_TXIOBIST, _CONFIGREG, val);
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_check_optical_eom_status_lr10
(
    lwlink_link *link,
    LwBool      *bEomLow
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LWSWITCH_TIMEOUT timeout;
    LwBool keepPolling;
    LwBool eomDone;
    LwU8 eom[4];
    LwU32 val, i;

    // Clear from any previous usage
    *bEomLow = LW_FALSE;

    val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLPHYCTL_LANE, _PAD_CTL_8(4));
    val = FLD_SET_DRF(_LWLPHYCTL_LANE, _PAD_CTL_8, _RX_EOM_OVRD, _ON, val);
    val = FLD_SET_DRF(_LWLPHYCTL_LANE, _PAD_CTL_8, _RX_EOM_EN, _ON, val);
    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLPHYCTL_LANE, _PAD_CTL_8(4), val);

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1MSEC_IN_NS * 20, &timeout);
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;
        eomDone = LW_TRUE;

        for (i = 0; i < 4; i++)
        {
            val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLPHYCTL_LANE, _PAD_CTL_4(i));
            if (FLD_TEST_DRF_NUM(_LWLPHYCTL_LANE, _PAD_CTL_4, _RX_EOM_DONE, 0, val))
            {
                eomDone = LW_FALSE;
            }
        }

        if (eomDone)
        {
            break;
        }
        lwswitch_os_sleep(1);
    }
    while(keepPolling);

    if (!eomDone)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to EOM data for link %d\n",
            __FUNCTION__, link->linkNumber);
        return LW_ERR_LWLINK_CONFIGURATION_ERROR;
    }

    val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLPHYCTL_LANE, _PAD_CTL_8(4));
    val = FLD_SET_DRF(_LWLPHYCTL_LANE, _PAD_CTL_8, _RX_EOM_OVRD, _OFF, val);
    val = FLD_SET_DRF(_LWLPHYCTL_LANE, _PAD_CTL_8, _RX_EOM_EN, _OFF, val);
    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLPHYCTL_LANE, _PAD_CTL_8(4), val);

    for (i = 0; i < 4; i++)
    {
        val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLPHYCTL_LANE, _PAD_CTL_4(i));
        eom[i] = DRF_VAL(_LWLPHYCTL_LANE, _PAD_CTL_4, _RX_EOM_STATUS, val) & 0xFF;
        if (eom[i] < LWLINK_OPTICAL_EOM_THRESHOLD)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: EOM is too low on lane %d link %d.\n",
                __FUNCTION__, i, link->linkNumber);
            *bEomLow = LW_TRUE;
        }
    }

    LWSWITCH_PRINT(device, INFO,
        "%s: EOM(0,3) : %d, %d, %d, %d for link %d\n",
        __FUNCTION__, eom[0], eom[1], eom[2], eom[3], link->linkNumber);

    LWSWITCH_PRINT(device, INFO,
                "%s: Printing grading values after EOM check.\n",
                __FUNCTION__); 

    cciPrintGradingValues(device, 0, link->linkNumber);    

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_set_optical_force_eq_lr10
(
    lwlink_link *link,
    LwBool bEnable
)
{
    LwlStatus status;
    lwswitch_device *device = link->dev->pDevInfo;
    
    if (bEnable)
    {
        status = lwswitch_minion_send_command(device, link->linkNumber,
                  LW_MINION_LWLINK_DL_CMD_COMMAND_FORCE_EQ_OVERRIDE_1, 0);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: FORCE_EQ_OVERRIDE CMD failed for link %d.\n",
                __FUNCTION__, link->linkNumber);
            return LW_ERR_LWLINK_CONFIGURATION_ERROR;
        }
    }
    else
    {
        status = lwswitch_minion_send_command(device, link->linkNumber,
                  LW_MINION_LWLINK_DL_CMD_COMMAND_RELEASE_EQ_OVERRIDE_1, 0);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: RELEASE_EQ_OVERRIDE CMD failed for link %d.\n",
                __FUNCTION__, link->linkNumber);
            return LW_ERR_LWLINK_CONFIGURATION_ERROR;
        }
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_set_optical_infinite_mode_lr10
(
    lwlink_link *link,
    LwBool bEnable
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwU32 val;

    if (!cciIsLinkManaged(device, link->linkNumber))
    {
        return LWL_SUCCESS;
    }

    // ENABLE or DISABLE INIFIFITE PRBS
    val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL, _TX_TRAIN0_TX);
    val = FLD_SET_DRF_NUM(_LWLDL, _TX_TRAIN0_TX, _PRBS_INFINITE, bEnable ? 1 : 0, val);
    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL, _TX_TRAIN0_TX, val);

    if (bEnable)
    {
        LWSWITCH_PRINT(device, INFO,
                "%s: Printing grading values after enable infinite PRBS.\n",
                __FUNCTION__);

        cciPrintGradingValues(device, 0, link->linkNumber);   
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_enable_optical_maintenance_lr10
(
    lwlink_link *link,
    LwBool       bTx
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwlStatus status;
    LwBool freeze_maintenance = LW_FALSE;
    LwBool restart_training = LW_FALSE;
    LwBool lwlink_mode = LW_TRUE;


    if (!cciIsLinkManaged(device, link->linkNumber))
    {
        return LWL_SUCCESS;
    }

    status = cciConfigureLwlinkMode(device, 0, link->linkNumber,
                 bTx, freeze_maintenance, restart_training, lwlink_mode);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to enable maintenance on link %d\n",
            __FUNCTION__, link->linkNumber);
        return LW_ERR_LWLINK_CONFIGURATION_ERROR;
    }

    return LWL_SUCCESS;
}

LwlStatus lwswitch_corelib_set_optical_iobist_lr10
(
    lwlink_link *link,
    LwBool bEnable
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwlStatus status;

    if (!cciIsLinkManaged(device, link->linkNumber))
    {
        return LWL_SUCCESS;
    }

    status = lwswitch_cci_enable_iobist_lr10(device, link->linkNumber, bEnable);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to set IOBIST on link %d\n",
            __FUNCTION__, link->linkNumber);
        return LW_ERR_LWLINK_CONFIGURATION_ERROR;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_set_optical_pretrain_lr10
(
    lwlink_link *link,
    LwBool      bTx,
    LwBool      bEnable
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwlStatus status;
    LwBool freeze_maintenance, restart_training;

    if (!cciIsLinkManaged(device, link->linkNumber))
    {
        return LWL_SUCCESS;
    }

    if (bEnable)
    {
        freeze_maintenance = LW_FALSE;
        restart_training = LW_TRUE;
    }
    else
    {
        freeze_maintenance = LW_TRUE;
        restart_training = LW_FALSE;
    }

    status = cciConfigureLwlinkMode(device, 0, link->linkNumber,
                bTx, freeze_maintenance, restart_training, LW_TRUE);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to set pretrain on link %d\n",
            __FUNCTION__, link->linkNumber);
        return LW_ERR_LWLINK_CONFIGURATION_ERROR;
    }

    return LWL_SUCCESS;
}

LwlStatus lwswitch_corelib_check_optical_pretrain_lr10
(
    lwlink_link *link,
    LwBool      isTx,
    LwBool      *bSuccess
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwlStatus status;

    if (!cciIsLinkManaged(device, link->linkNumber))
    {
        return LWL_SUCCESS;
    }

    status = cciPollForPreTraining(device, 0, link->linkNumber, isTx);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get pre-training status on link %d\n",
            __FUNCTION__, link->linkNumber);
        *bSuccess = LW_FALSE;
    }
    else
    {
        *bSuccess = LW_FALSE;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_corelib_init_optical_links_lr10
(
    lwlink_link *link
)
{
    lwswitch_device *device = link->dev->pDevInfo;
    LwlStatus status;

    if (!cciIsLinkManaged(device, link->linkNumber))
    {
        return LWL_SUCCESS;
    }

    status = lwswitch_cci_initialization_sequence_lr10(device, link->linkNumber);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get initialize optical link %d\n",
            __FUNCTION__, link->linkNumber);
        return LW_ERR_LWLINK_CONFIGURATION_ERROR;
    }

    link->bRxDetected = LW_TRUE;
    link->bTxCommonModeFail = LW_FALSE;
    link->bOpticalLink = LW_TRUE;

    return LWL_SUCCESS;
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
