/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"
#include "sv10/sv10.h"
#include "sv10/bus_sv10.h"

#include "lwswitch/svnp01/dev_lw_xp.h"
#include "lwswitch/svnp01/dev_lw_xve.h"

/**
 * Function to get per lane counters.
 *
 * @param  The device object of the counters we want to get.
 * @param  The Lane Error Status mask.
 * @param  The counter value for each lane.
 * @return Returns LWL_SUCCESS on success.  Anything else should be considered a failure.
 */
static void
_lwswitch_pex_get_lane_error_counts
(
    lwswitch_device *device,
    LwU16  *laneErrorMask,
    LwU8   *laneCounter
)
{
    LwU32 val;

    *laneErrorMask = (LwU16) LWSWITCH_OFF_RD32(device, LW_XP_LANE_ERROR_STATUS);

    val = LWSWITCH_OFF_RD32(device, LW_XP_LANE_ERRORS_COUNT_0);
    laneCounter[0] = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_0, _LANE_0_VALUE, val);
    laneCounter[1] = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_0, _LANE_1_VALUE, val);
    laneCounter[2] = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_0, _LANE_2_VALUE, val);
    laneCounter[3] = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_0, _LANE_3_VALUE, val);

    val = LWSWITCH_OFF_RD32(device, LW_XP_LANE_ERRORS_COUNT_1);
    laneCounter[4] = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_1, _LANE_4_VALUE, val);
    laneCounter[5] = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_1, _LANE_5_VALUE, val);
    laneCounter[6] = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_1, _LANE_6_VALUE, val);
    laneCounter[7] = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_1, _LANE_7_VALUE, val);

    val = LWSWITCH_OFF_RD32(device, LW_XP_LANE_ERRORS_COUNT_2);
    laneCounter[8]  = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_2, _LANE_8_VALUE, val);
    laneCounter[9]  = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_2, _LANE_9_VALUE, val);
    laneCounter[10] = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_2, _LANE_10_VALUE, val);
    laneCounter[11] = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_2, _LANE_11_VALUE, val);

    val = LWSWITCH_OFF_RD32(device, LW_XP_LANE_ERRORS_COUNT_3);
    laneCounter[12] = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_3, _LANE_12_VALUE, val);
    laneCounter[13] = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_3, _LANE_13_VALUE, val);
    laneCounter[14] = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_3, _LANE_14_VALUE, val);
    laneCounter[15] = (LwU8) DRF_VAL(_XP, _LANE_ERRORS_COUNT_3, _LANE_15_VALUE, val);
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
lwswitch_pex_get_counter_sv10
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
        case LWSWITCH_PEX_COUNTER_L1_1_ENTRY_COUNT:
        {
            *pCount = 0;
            for (i = 0; i < LW_XP_L1_1_ENTRY_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_L1_1_ENTRY_COUNT(i));
                *pCount += DRF_VAL(_XP, _L1_1_ENTRY_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_L1_2_ENTRY_COUNT:
        {
            *pCount = 0;
            for (i = 0; i < LW_XP_L1_2_ENTRY_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_L1_2_ENTRY_COUNT(i));
                *pCount += DRF_VAL(_XP, _L1_2_ENTRY_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_L1_2_ABORT_COUNT:
        {
            *pCount = 0;
            for (i = 0; i < LW_XP_L1_2_ABORT_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_L1_2_ABORT_COUNT(i));
                *pCount += DRF_VAL(_XP, _L1_2_ABORT_COUNT,
                                       _VALUE , tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_L1SS_TO_DEEP_L1_TIMEOUT_COUNT:
        {
            *pCount = 0;
            for (i = 0; i < LW_XP_L1_SUBSTATE_TO_DEEP_L1_TIMEOUT_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_L1_SUBSTATE_TO_DEEP_L1_TIMEOUT_COUNT(i));
                *pCount += DRF_VAL(_XP, _L1_SUBSTATE_TO_DEEP_L1_TIMEOUT_COUNT,
                                       _VALUE , tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_L1_SHORT_DURATION_COUNT:
        {
            *pCount = 0;
            for (i = 0; i < LW_XP_L1_SHORT_DURATION_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_L1_SHORT_DURATION_COUNT(i));
                *pCount += DRF_VAL(_XP, _L1_SHORT_DURATION_COUNT,
                                       _VALUE , tempRegVal);
            }
            break;
        }

        case LWSWITCH_PEX_COUNTER_RECEIVER_ERRORS:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_RECEIVER_ERRORS_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_RECEIVER_ERRORS_COUNT(i));
                *pCount += DRF_VAL(_XP, _RECEIVER_ERRORS_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_REPLAY_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_REPLAY_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_REPLAY_COUNT(i));
                *pCount += DRF_VAL(_XP, _REPLAY_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_REPLAY_ROLLOVER_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_REPLAY_ROLLOVER_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_REPLAY_ROLLOVER_COUNT(i));
                *pCount += DRF_VAL(_XP, _REPLAY_ROLLOVER_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_BAD_DLLP_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_BAD_DLLP_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_BAD_DLLP_COUNT(i));
                *pCount += DRF_VAL(_XP, _BAD_DLLP_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_BAD_TLP_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_BAD_TLP_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_BAD_TLP_COUNT(i));
                *pCount += DRF_VAL(_XP, _BAD_TLP_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_8B10B_ERRORS_COUNT:
        {
            tempRegVal = LWSWITCH_OFF_RD32(device,
                                      LW_XP__8B10B_ERRORS_COUNT);
            *pCount = DRF_VAL(_XP, __8B10B_ERRORS_COUNT,
                                            _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_SYNC_HEADER_ERRORS_COUNT:
        {
            tempRegVal = LWSWITCH_OFF_RD32(device,
                                      LW_XP_SYNC_HEADER_ERRORS_COUNT);
            *pCount = DRF_VAL(_XP, _SYNC_HEADER_ERRORS_COUNT,
                                            _VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_LCRC_ERRORS_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_LCRC_ERRORS_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_LCRC_ERRORS_COUNT(i));
                *pCount += DRF_VAL(_XP, _LCRC_ERRORS_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_FAILED_L0S_EXITS_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_FAILED_L0S_EXITS_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_FAILED_L0S_EXITS_COUNT(i));
                *pCount += DRF_VAL(_XP, _FAILED_L0S_EXITS_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_NAKS_SENT_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_NAKS_SENT_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_NAKS_SENT_COUNT(i));
                *pCount += DRF_VAL(_XP, _NAKS_SENT_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_NAKS_RCVD_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_NAKS_RCVD_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_NAKS_RCVD_COUNT(i));
                *pCount += DRF_VAL(_XP, _NAKS_RCVD_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_LANE_ERRORS:
        {
            *pCount = 0;

            _lwswitch_pex_get_lane_error_counts(device,
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
            *pCount = 0;
            for (i =0; i < LW_XP_L1_TO_RECOVERY_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_L1_TO_RECOVERY_COUNT(i));
                *pCount += DRF_VAL(_XP, _L1_TO_RECOVERY_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_L0_TO_RECOVERY_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_L0_TO_RECOVERY_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_L0_TO_RECOVERY_COUNT(i));
                *pCount += DRF_VAL(_XP, _L0_TO_RECOVERY_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_RECOVERY_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_RECOVERY_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_RECOVERY_COUNT(i));
                *pCount += DRF_VAL(_XP, _RECOVERY_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_CHIPSET_XMIT_L0S_ENTRY_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_CHIPSET_XMIT_L0S_ENTRY_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_CHIPSET_XMIT_L0S_ENTRY_COUNT(i));
                *pCount += DRF_VAL(_XP, _CHIPSET_XMIT_L0S_ENTRY_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_GPU_XMIT_L0S_ENTRY_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_GPU_XMIT_L0S_ENTRY_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_GPU_XMIT_L0S_ENTRY_COUNT(i));
                *pCount += DRF_VAL(_XP, _GPU_XMIT_L0S_ENTRY_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_L1_ENTRY_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_L1_ENTRY_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_L1_ENTRY_COUNT(i));
                *pCount += DRF_VAL(_XP, _L1_ENTRY_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_L1P_ENTRY_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_L1P_ENTRY_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_L1P_ENTRY_COUNT(i));
                *pCount += DRF_VAL(_XP, _L1P_ENTRY_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_DEEP_L1_ENTRY_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_DEEP_L1_ENTRY_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_DEEP_L1_ENTRY_COUNT(i));
                *pCount += DRF_VAL(_XP, _DEEP_L1_ENTRY_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case LWSWITCH_PEX_COUNTER_ASLM_COUNT:
        {
            *pCount = 0;
            for (i =0; i < LW_XP_ASLM_COUNT__SIZE_1; i++)
            {
                tempRegVal = LWSWITCH_OFF_RD32(device,
                                          LW_XP_ASLM_COUNT(i));
                *pCount += DRF_VAL(_XP, _ASLM_COUNT,
                                            _VALUE, tempRegVal);
            }
            break;
        }
        case  LWSWITCH_PEX_COUNTER_CORR_ERROR_COUNT:
        {
            tempRegVal = LWSWITCH_OFF_RD32(device,
                    DEVICE_BASE(LW_PCFG) + LW_XVE_ERROR_COUNTER1);
            *pCount = DRF_VAL(_XVE, _ERROR_COUNTER1,
                                       _CORR_ERROR_COUNT_VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_NON_FATAL_ERROR_COUNT:
        {
            tempRegVal = LWSWITCH_OFF_RD32(device,
                    DEVICE_BASE(LW_PCFG) + LW_XVE_ERROR_COUNTER);
            *pCount = DRF_VAL(_XVE, _ERROR_COUNTER,
                                       _NON_FATAL_ERROR_COUNT_VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_FATAL_ERROR_COUNT:
        {
            tempRegVal = LWSWITCH_OFF_RD32(device,
                    DEVICE_BASE(LW_PCFG) + LW_XVE_ERROR_COUNTER);
            *pCount = DRF_VAL(_XVE, _ERROR_COUNTER,
                                       _FATAL_ERROR_COUNT_VALUE, tempRegVal);
            break;
        }
        case LWSWITCH_PEX_COUNTER_UNSUPP_REQ_COUNT:
        {
            tempRegVal = LWSWITCH_OFF_RD32(device,
                    DEVICE_BASE(LW_PCFG) + LW_XVE_ERROR_COUNTER);
            *pCount = DRF_VAL(_XVE, _ERROR_COUNTER,
                                       _UNSUPP_REQ_COUNT_VALUE, tempRegVal);
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
lwswitch_ctrl_pex_get_lane_counters_sv10
(
    lwswitch_device *device,
    LWSWITCH_PEX_GET_LANE_COUNTERS_PARAMS *pParams
)
{
    _lwswitch_pex_get_lane_error_counts(device, &pParams->pexLaneErrorStatus,
        &pParams->pexLaneCounter[0]);

    return LWL_SUCCESS;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LwlStatus
lwswitch_ctrl_pex_clear_counters_sv10
(
    lwswitch_device *device,
    LWSWITCH_PEX_CLEAR_COUNTERS_PARAMS *pParams
)
{
    LwU32 counterMask = pParams->pexCounterMask;
   LwU32 v = LWSWITCH_OFF_RD32(device, LW_XP_ERROR_COUNTER_RESET);

    if ((counterMask &  LWSWITCH_PEX_COUNTER_8B10B_ERRORS_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _8B10B_ERRORS_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_SYNC_HEADER_ERRORS_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _SYNC_HEADER_ERRORS_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_LANE_ERRORS))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _LANE_ERRORS_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_RECEIVER_ERRORS))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _RECEIVER_ERRORS_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_LCRC_ERRORS_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _LCRC_ERRORS_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_FAILED_L0S_EXITS_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _FAILED_L0S_EXITS_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_NAKS_SENT_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _NAKS_SENT_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_NAKS_RCVD_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _NAKS_RCVD_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_REPLAY_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _REPLAY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_REPLAY_ROLLOVER_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _REPLAY_ROLLOVER_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_L1_TO_RECOVERY_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _L1_TO_RECOVERY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_L0_TO_RECOVERY_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _L0_TO_RECOVERY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_RECOVERY_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _RECOVERY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_BAD_DLLP_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _BAD_DLLP_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_BAD_TLP_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _BAD_TLP_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_CHIPSET_XMIT_L0S_ENTRY_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _CHIPSET_XMIT_L0S_ENTRY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_GPU_XMIT_L0S_ENTRY_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _GPU_XMIT_L0S_ENTRY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_L1_ENTRY_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _L1_ENTRY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_L1P_ENTRY_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _L1P_ENTRY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_DEEP_L1_ENTRY_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _DEEP_L1_ENTRY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_ASLM_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _ASLM_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_L1_1_ENTRY_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _L1_1_ENTRY_COUNT, _PENDING, v);
    }
    if ((counterMask &  LWSWITCH_PEX_COUNTER_L1_2_ENTRY_COUNT))
    {
        v = FLD_SET_DRF(_XP, _ERROR_COUNTER_RESET, _L1_2_ENTRY_COUNT, _PENDING, v);
    }

    LWSWITCH_OFF_WR32(device, LW_XP_ERROR_COUNTER_RESET, v);

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_pex_set_eom_sv10
(
    lwswitch_device *device,
    LWSWITCH_PEX_CTRL_EOM *pParams
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_ctrl_pex_get_eom_status_sv10
(
    lwswitch_device *device,
    LWSWITCH_PEX_GET_EOM_STATUS_PARAMS *pParams
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_ctrl_get_uphy_dln_cfg_space_sv10
(
    lwswitch_device *device,
    LWSWITCH_GET_PEX_UPHY_DLN_CFG_SPACE_PARAMS *pParams
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_ctrl_set_pcie_link_speed_sv10
(
    lwswitch_device *device,
    LWSWITCH_SET_PCIE_LINK_SPEED_PARAMS *pParams
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
