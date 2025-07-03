/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "g_lwconfig.h"
#include "common_lwswitch.h"
#include "ls10/ls10.h"
#include "ls10/bus_ls10.h"
#include "soe/soe_lwswitch.h"

#include "lwswitch/ls10/dev_lw_xpl.h"
#include "lwswitch/ls10/dev_xtl_ep_pri.h"

/**
 * Function to get per lane counters.
 *
 * @param  The device object of the counters we want to get.
 * @param  The Lane Error Status mask.
 * @param  The counter value for each lane.
 * @return Returns LWL_SUCCESS on success.  Anything else should be considered a failure.
 */
static void
_lwswitch_pex_get_lane_error_counts_ls10
(
    lwswitch_device *device,
    LwU16  *laneErrorMask,
    LwU8   *laneCounter
)
{
    LwU32 val;
    LwU32 i;

    *laneErrorMask = (LwU16) LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _RXLANES_ERROR_STATUS);

    for (i = 0; i < LW_XPL_PL_RXLANES_ERRORS_COUNT__SIZE_1; i++)
    {
        val = LWSWITCH_ENG_RD32_IDX(device, XPL, , 0, _XPL_PL, _RXLANES_ERRORS_COUNT, i);
        laneCounter[i] = (LwU8) DRF_VAL(_XPL_PL, _RXLANES_ERRORS_COUNT, _VALUE, val);
    }
}

/*!
 * This function retrieves the counter values in PEX.
 *
 * @param  The device object of the counters we want to get.
 * @param  counterType The type specifying the required counter type.
 * @param  pCount      A pointer to the location where the count will be returned.
 *
 * @return Returns LWL_SUCCESS on success.  Anything else should be considered a failure.
 */
LwlStatus
lwswitch_pex_get_counter_ls10
(
    lwswitch_device *device,
    LwU32   counterType,
    LwU32   *pCount
)
{
    LwU32 tempRegVal;
    LwU8  i;
    LwU16 laneErrorStatusMask;
    LwU8  laneErrors[16];

    switch (counterType)
    {
        case LWSWITCH_PEX_COUNTER_RECEIVER_ERRORS:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_DL, _RECEIVER_ERRORS_COUNT);
            *pCount = DRF_VAL(_XPL_DL, _RECEIVER_ERRORS_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_REPLAY_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_DL, _REPLAY_COUNT);
            *pCount = DRF_VAL(_XPL_DL, _REPLAY_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_REPLAY_ROLLOVER_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_DL, _REPLAY_ROLLOVER_COUNT);
            *pCount = DRF_VAL(_XPL_DL, _REPLAY_ROLLOVER_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_BAD_DLLP_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_DL, _BAD_DLLP_COUNT);
            *pCount = DRF_VAL(_XPL_DL, _BAD_DLLP_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_BAD_TLP_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_DL, _BAD_TLP_COUNT);
            *pCount = DRF_VAL(_XPL_DL, _BAD_TLP_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_8B10B_ERRORS_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _RXLANES_8B10B_ERRORS_COUNT);
            *pCount = DRF_VAL(_XPL_PL, _RXLANES_8B10B_ERRORS_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_SYNC_HEADER_ERRORS_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _RXLANES_SYNC_HEADER_ERRORS_COUNT);
            *pCount = DRF_VAL(_XPL_PL, _RXLANES_SYNC_HEADER_ERRORS_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_LCRC_ERRORS_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_DL, _LCRC_ERRORS_COUNT);
            *pCount = DRF_VAL(_XPL_DL, _LCRC_ERRORS_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_NAKS_SENT_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_DL, _NAKS_SENT_COUNT);
            *pCount = DRF_VAL(_XPL_DL, _NAKS_SENT_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_NAKS_RCVD_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_DL, _NAKS_RCVD_COUNT);
            *pCount = DRF_VAL(_XPL_DL, _NAKS_RCVD_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_LANE_ERRORS:
        {
            *pCount = 0;

            _lwswitch_pex_get_lane_error_counts_ls10(device,
                &laneErrorStatusMask, &laneErrors[0]);
            if (laneErrorStatusMask)
            {
                for (i = 0; i < 16 ; i++)
                {
                    (*pCount) += laneErrors[i];
                }
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_L1_TO_RECOVERY_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _LTSSM_L1_TO_RECOVERY_COUNT);
            *pCount = DRF_VAL(_XPL_PL, _LTSSM_L1_TO_RECOVERY_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_L0_TO_RECOVERY_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _LTSSM_L0_TO_RECOVERY_COUNT);
            *pCount = DRF_VAL(_XPL_PL, _LTSSM_L0_TO_RECOVERY_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_RECOVERY_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _LTSSM_RECOVERY_COUNT);
            *pCount = DRF_VAL(_XPL_PL, _LTSSM_RECOVERY_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_L1_ENTRY_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _LTSSM_L1_ENTRY_COUNT);
            *pCount = DRF_VAL(_XPL_PL, _LTSSM_L1_ENTRY_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_L1P_ENTRY_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _LTSSM_L1_PLL_PD_ENTRY_COUNT);
            *pCount = DRF_VAL(_XPL_PL, _LTSSM_L1_PLL_PD_ENTRY_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_DEEP_L1_ENTRY_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _LTSSM_L1_CPM_ENTRY_COUNT);
            *pCount = DRF_VAL(_XPL_PL, _LTSSM_L1_CPM_ENTRY_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_ASLM_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _LTSSM_ASLM_COUNT);
            *pCount = DRF_VAL(_XPL_PL, _LTSSM_ASLM_COUNT, _VALUE, tempRegVal);
            break;
        }
        case  LWSWITCH_PEX_COUNTER_CORR_ERROR_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XTL, , 0, _XTL_EP_PRI, _ERROR_COUNTER);
            *pCount = DRF_VAL(_XTL_EP_PRI, _ERROR_COUNTER, _RSVD_CORR_ERROR_COUNT_VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_NON_FATAL_ERROR_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XTL, , 0, _XTL_EP_PRI, _ERROR_COUNTER);
            *pCount = DRF_VAL(_XTL_EP_PRI, _ERROR_COUNTER, _NON_FATAL_ERROR_COUNT_VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_FATAL_ERROR_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XTL, , 0, _XTL_EP_PRI, _ERROR_COUNTER);
            *pCount = DRF_VAL(_XTL_EP_PRI, _ERROR_COUNTER, _FATAL_ERROR_COUNT_VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_UNSUPP_REQ_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XTL, , 0, _XTL_EP_PRI, _ERROR_COUNTER);
            *pCount = DRF_VAL(_XTL_EP_PRI, _ERROR_COUNTER, _UNSUPP_REQUEST_COUNT_VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_L1_1_ENTRY_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _LTSSM_L1_1_ENTRY_COUNT);
            *pCount = DRF_VAL(_XPL_PL, _LTSSM_L1_1_ENTRY_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_L1_2_ENTRY_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _LTSSM_L1_2_ENTRY_COUNT);
            *pCount = DRF_VAL(_XPL_PL, _LTSSM_L1_2_ENTRY_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_L1_2_ABORT_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _LTSSM_L1_2_ABORT_COUNT);
            *pCount = DRF_VAL(_XPL_PL, _LTSSM_L1_2_ABORT_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_L1SS_TO_DEEP_L1_TIMEOUT_COUNT:
        {
            tempRegVal = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _LTSSM_L1_1_ABORT_COUNT);
            *pCount = DRF_VAL(_XPL_PL, _LTSSM_L1_1_ABORT_COUNT, _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_L1_SHORT_DURATION_COUNT:
        case LWSWITCH_PEX_COUNTER_FAILED_L0S_EXITS_COUNT:
        case LWSWITCH_PEX_COUNTER_CHIPSET_XMIT_L0S_ENTRY_COUNT:
        case LWSWITCH_PEX_COUNTER_GPU_XMIT_L0S_ENTRY_COUNT:
        {
            // Not supported
            *pCount = 0;
            break;
        }
        default:
        {
            return -LWL_BAD_ARGS;
        }
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_pex_get_lane_counters_ls10
(
    lwswitch_device *device,
    LWSWITCH_PEX_GET_LANE_COUNTERS_PARAMS *pParams
)
{
    _lwswitch_pex_get_lane_error_counts_ls10(device, &pParams->pexLaneErrorStatus,
        &pParams->pexLaneCounter[0]);

    return LWL_SUCCESS;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LwlStatus
lwswitch_ctrl_pex_clear_counters_ls10
(
    lwswitch_device *device,
    LWSWITCH_PEX_CLEAR_COUNTERS_PARAMS *pParams
)
{
    LwU32 counterMask = pParams->pexCounterMask;
    LwU32 v;
    LwU32 i;

    //
    // XPL DL
    //
    v = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_DL, _ERROR_COUNTER_RESET);
    if ((counterMask &  LWSWITCH_PEX_COUNTER_RECEIVER_ERRORS))
    {
        v = FLD_SET_DRF(_XPL_DL, _ERROR_COUNTER_RESET, _RECEIVER_ERRORS_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_LCRC_ERRORS_COUNT))
    {
        v = FLD_SET_DRF(_XPL_DL, _ERROR_COUNTER_RESET, _LCRC_ERRORS_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_NAKS_SENT_COUNT))
    {
        v = FLD_SET_DRF(_XPL_DL, _ERROR_COUNTER_RESET, _NAKS_SENT_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_NAKS_RCVD_COUNT))
    {
        v = FLD_SET_DRF(_XPL_DL, _ERROR_COUNTER_RESET, _NAKS_RCVD_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_REPLAY_COUNT))
    {
        v = FLD_SET_DRF(_XPL_DL, _ERROR_COUNTER_RESET, _REPLAY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_REPLAY_ROLLOVER_COUNT))
    {
        v = FLD_SET_DRF(_XPL_DL, _ERROR_COUNTER_RESET, _REPLAY_ROLLOVER_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_BAD_DLLP_COUNT))
    {
        v = FLD_SET_DRF(_XPL_DL, _ERROR_COUNTER_RESET, _BAD_DLLP_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_BAD_TLP_COUNT))
    {
        v = FLD_SET_DRF(_XPL_DL, _ERROR_COUNTER_RESET, _BAD_TLP_COUNT, _PENDING, v);
    }
    LWSWITCH_ENG_WR32(device, XPL, , 0, _XPL_DL, _ERROR_COUNTER_RESET, v);

    //
    // XPL PL
    //
    if ((counterMask &  LWSWITCH_PEX_COUNTER_8B10B_ERRORS_COUNT))
    {
        v = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _RXLANES_8B10B_ERRORS_COUNT);
        v = FLD_SET_DRF_NUM(_XPL_PL, _RXLANES_8B10B_ERRORS_COUNT, _RESET, 0x1, v);
        LWSWITCH_ENG_WR32(device, XPL, , 0, _XPL_PL, _RXLANES_8B10B_ERRORS_COUNT, v);
    }

    if ((counterMask &  LWSWITCH_PEX_COUNTER_SYNC_HEADER_ERRORS_COUNT))
    {
        v = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _RXLANES_SYNC_HEADER_ERRORS_COUNT);
        v = FLD_SET_DRF_NUM(_XPL_PL, _RXLANES_SYNC_HEADER_ERRORS_COUNT, _RESET, 0x1, v);
        LWSWITCH_ENG_WR32(device, XPL, , 0, _XPL_PL, _RXLANES_SYNC_HEADER_ERRORS_COUNT, v);
    }

    if ((counterMask &  LWSWITCH_PEX_COUNTER_LANE_ERRORS))
    {
        for (i = 0; i < LW_XPL_PL_RXLANES_ERRORS_COUNT__SIZE_1; i++)
        {
            v = LWSWITCH_ENG_RD32_IDX(device, XPL, , 0, _XPL_PL, _RXLANES_ERRORS_COUNT, i);
            v = FLD_SET_DRF_NUM(_XPL_PL, _RXLANES_ERRORS_COUNT, _RESET, 0x1, v);
            LWSWITCH_ENG_WR32_IDX(device, XPL, , 0, _XPL_PL, _RXLANES_ERRORS_COUNT, i, v);
        }
    }

    v = LWSWITCH_ENG_RD32(device, XPL, , 0, _XPL_PL, _LTSSM_ERROR_COUNTER_RESET);
    if ((counterMask &  LWSWITCH_PEX_COUNTER_L1_TO_RECOVERY_COUNT))
    {
        v = FLD_SET_DRF(_XPL_PL, _LTSSM_ERROR_COUNTER_RESET, _L1_TO_RECOVERY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_L0_TO_RECOVERY_COUNT))
    {
        v = FLD_SET_DRF(_XPL_PL, _LTSSM_ERROR_COUNTER_RESET, _L0_TO_RECOVERY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_RECOVERY_COUNT))
    {
        v = FLD_SET_DRF(_XPL_PL, _LTSSM_ERROR_COUNTER_RESET, _RECOVERY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_L1_ENTRY_COUNT))
    {
        v = FLD_SET_DRF(_XPL_PL, _LTSSM_ERROR_COUNTER_RESET, _L1_ENTRY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_L1P_ENTRY_COUNT))
    {
        v = FLD_SET_DRF(_XPL_PL, _LTSSM_ERROR_COUNTER_RESET, _L1_PLL_PD_ENTRY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_DEEP_L1_ENTRY_COUNT))
    {
        v = FLD_SET_DRF(_XPL_PL, _LTSSM_ERROR_COUNTER_RESET, _L1_CPM_ENTRY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_ASLM_COUNT))
    {
        v = FLD_SET_DRF(_XPL_PL, _LTSSM_ERROR_COUNTER_RESET, _ASLM_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_L1_1_ENTRY_COUNT))
    {
        v = FLD_SET_DRF(_XPL_PL, _LTSSM_ERROR_COUNTER_RESET, _L1_1_ENTRY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_L1_2_ENTRY_COUNT))
    {
        v = FLD_SET_DRF(_XPL_PL, _LTSSM_ERROR_COUNTER_RESET, _L1_2_ENTRY_COUNT, _PENDING, v);
    }
    LWSWITCH_ENG_WR32(device, XPL, , 0, _XPL_PL, _LTSSM_ERROR_COUNTER_RESET, v);

    //
    // XTL_EP_PRI
    //
    v = LWSWITCH_ENG_RD32(device, XTL, , 0, _XTL_EP_PRI, _ERROR_COUNTER_RESET);
    if ((counterMask &  LWSWITCH_PEX_COUNTER_CORR_ERROR_COUNT))
    {
        v = FLD_SET_DRF(_XTL_EP_PRI, _ERROR_COUNTER_RESET, _CORR_ERROR_COUNT, _TRIGGER, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_NON_FATAL_ERROR_COUNT))
    {
        v = FLD_SET_DRF(_XTL_EP_PRI, _ERROR_COUNTER_RESET, _NON_FATAL_ERROR_COUNT, _TRIGGER, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_FATAL_ERROR_COUNT))
    {
        v = FLD_SET_DRF(_XTL_EP_PRI, _ERROR_COUNTER_RESET, _FATAL_ERROR_COUNT, _TRIGGER, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_UNSUPP_REQ_COUNT))
    {
        v = FLD_SET_DRF(_XTL_EP_PRI, _ERROR_COUNTER_RESET, _UNSUPP_REQUEST_COUNT, _TRIGGER, v);
    }
    LWSWITCH_ENG_WR32(device, XTL, , 0, _XTL_EP_PRI, _ERROR_COUNTER_RESET, v);

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_pex_set_eom_ls10
(
    lwswitch_device *device,
    LWSWITCH_PEX_CTRL_EOM *pParams
)
{
    return soeSetPexEOM_HAL(device, pParams->mode, pParams->nblks, pParams->nerrs,
                            pParams->berEyeSel);
}

LwlStatus
lwswitch_ctrl_pex_get_eom_status_ls10
(
    lwswitch_device *device,
    LWSWITCH_PEX_GET_EOM_STATUS_PARAMS *pParams
)
{
    return soeGetPexEomStatus_HAL(device, pParams->mode, pParams->nblks,
        pParams->nerrs, pParams->berEyeSel, pParams->laneMask, pParams->eomStatus);
}

LwlStatus
lwswitch_ctrl_get_uphy_dln_cfg_space_ls10
(
    lwswitch_device *device,
    LWSWITCH_GET_PEX_UPHY_DLN_CFG_SPACE_PARAMS *pParams
)
{
    return soeGetUphyDlnCfgSpace_HAL(device, pParams->regAddress,
        pParams->laneSelectMask, &pParams->regValue);
}

LwlStatus
lwswitch_ctrl_set_pcie_link_speed_ls10
(
    lwswitch_device *device,
    LWSWITCH_SET_PCIE_LINK_SPEED_PARAMS *pParams
)
{
    LwlStatus status;

    // On ls10, link speed must not be greater than gen5.
    if ((pParams->linkSpeed ==   LWSWITCH_BIF_LINK_SPEED_ILWALID) ||
        (pParams->linkSpeed > LWSWITCH_BIF_LINK_SPEED_GEN5PCIE))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: linkSpeed is invalid or greater than maximum supported speed\n",
            __FUNCTION__);

        return -LWL_BAD_ARGS;
    }

    // Set Link speed in SOE.
    status = soeSetPcieLinkSpeed_HAL(device, pParams->linkSpeed);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed to set link speed for PCIEGEN%d\n",
            __FUNCTION__, pParams->linkSpeed);
    }

    return status;
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
