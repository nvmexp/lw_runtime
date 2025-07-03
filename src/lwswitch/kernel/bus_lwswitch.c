/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "g_lwconfig.h"
#include "common_lwswitch.h"

LwlStatus
lwswitch_ctrl_pex_get_counters
(
    lwswitch_device *device,
    LWSWITCH_PEX_GET_COUNTERS_PARAMS *pParams
)
{
    LwU32 totalCorrErrors = 0;
    LwU32 cnt;
    LwU32 r;
    LwlStatus status;

    if ((pParams->pexCounterMask &
        (LWSWITCH_PEX_COUNTER_RECEIVER_ERRORS |
         LWSWITCH_PEX_COUNTER_TOTAL_CORR_ERROR_COUNT)))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_RECEIVER_ERRORS, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_RECEIVER_ERRORS;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
        totalCorrErrors += cnt;
    }

    if ((pParams->pexCounterMask &
        (LWSWITCH_PEX_COUNTER_REPLAY_COUNT |
         LWSWITCH_PEX_COUNTER_TOTAL_CORR_ERROR_COUNT)))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_REPLAY_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_REPLAY_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
        totalCorrErrors += cnt;
    }

    if ((pParams->pexCounterMask &
        (LWSWITCH_PEX_COUNTER_REPLAY_ROLLOVER_COUNT |
         LWSWITCH_PEX_COUNTER_TOTAL_CORR_ERROR_COUNT)))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_REPLAY_ROLLOVER_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_REPLAY_ROLLOVER_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
        totalCorrErrors += cnt;
    }

    if ((pParams->pexCounterMask &
        (LWSWITCH_PEX_COUNTER_BAD_DLLP_COUNT |
         LWSWITCH_PEX_COUNTER_TOTAL_CORR_ERROR_COUNT)))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_BAD_DLLP_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_BAD_DLLP_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
        totalCorrErrors += cnt;
    }

    if ((pParams->pexCounterMask &
        (LWSWITCH_PEX_COUNTER_BAD_TLP_COUNT |
         LWSWITCH_PEX_COUNTER_TOTAL_CORR_ERROR_COUNT)))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_BAD_TLP_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_BAD_TLP_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
        totalCorrErrors += cnt;
    }

    if ((pParams->pexCounterMask &
        (LWSWITCH_PEX_COUNTER_8B10B_ERRORS_COUNT |
         LWSWITCH_PEX_COUNTER_TOTAL_CORR_ERROR_COUNT)))
    {
        status = lwswitch_pex_get_counter(device,
            LWSWITCH_PEX_COUNTER_8B10B_ERRORS_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_8B10B_ERRORS_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
        totalCorrErrors += cnt;
    }

    if ((pParams->pexCounterMask &
        (LWSWITCH_PEX_COUNTER_SYNC_HEADER_ERRORS_COUNT |
         LWSWITCH_PEX_COUNTER_TOTAL_CORR_ERROR_COUNT)))
    {
        status = lwswitch_pex_get_counter(device,
            LWSWITCH_PEX_COUNTER_SYNC_HEADER_ERRORS_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_SYNC_HEADER_ERRORS_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
        totalCorrErrors += cnt;
    }

    if ((pParams->pexCounterMask &
        (LWSWITCH_PEX_COUNTER_LCRC_ERRORS_COUNT |
         LWSWITCH_PEX_COUNTER_TOTAL_CORR_ERROR_COUNT)))
    {
        status = lwswitch_pex_get_counter(device,
            LWSWITCH_PEX_COUNTER_LCRC_ERRORS_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_LCRC_ERRORS_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
        totalCorrErrors += cnt;
    }

    if ((pParams->pexCounterMask &
        (LWSWITCH_PEX_COUNTER_FAILED_L0S_EXITS_COUNT |
         LWSWITCH_PEX_COUNTER_TOTAL_CORR_ERROR_COUNT)))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_FAILED_L0S_EXITS_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_FAILED_L0S_EXITS_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
        totalCorrErrors += cnt;
    }

    if ((pParams->pexCounterMask &
        (LWSWITCH_PEX_COUNTER_NAKS_SENT_COUNT |
         LWSWITCH_PEX_COUNTER_TOTAL_CORR_ERROR_COUNT)))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_NAKS_SENT_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_NAKS_SENT_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
        totalCorrErrors += cnt;
    }

    if ((pParams->pexCounterMask &
        (LWSWITCH_PEX_COUNTER_NAKS_RCVD_COUNT |
         LWSWITCH_PEX_COUNTER_TOTAL_CORR_ERROR_COUNT)))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_NAKS_RCVD_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }

        r = LWSWITCH_PEX_COUNTER_NAKS_RCVD_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
        totalCorrErrors += cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_LANE_ERRORS))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_LANE_ERRORS, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_LANE_ERRORS;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_L1_TO_RECOVERY_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_L1_TO_RECOVERY_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_L1_TO_RECOVERY_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_L0_TO_RECOVERY_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_L0_TO_RECOVERY_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_L0_TO_RECOVERY_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_RECOVERY_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_RECOVERY_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_RECOVERY_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_CHIPSET_XMIT_L0S_ENTRY_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_CHIPSET_XMIT_L0S_ENTRY_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_CHIPSET_XMIT_L0S_ENTRY_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_GPU_XMIT_L0S_ENTRY_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_GPU_XMIT_L0S_ENTRY_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_GPU_XMIT_L0S_ENTRY_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_L1_ENTRY_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_L1_ENTRY_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_L1_ENTRY_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_L1P_ENTRY_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_L1P_ENTRY_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_L1P_ENTRY_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_DEEP_L1_ENTRY_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_DEEP_L1_ENTRY_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_DEEP_L1_ENTRY_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_ASLM_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_ASLM_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_ASLM_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_TOTAL_CORR_ERROR_COUNT))
    {
        pParams->pexTotalCorrectableErrors = totalCorrErrors;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_CORR_ERROR_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_CORR_ERROR_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        pParams->pexCorrectableErrors = (LwU16) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_NON_FATAL_ERROR_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_NON_FATAL_ERROR_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        pParams->pexTotalNonFatalErrors = (LwU8) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_FATAL_ERROR_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_FATAL_ERROR_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        pParams->pexTotalFatalErrors = (LwU8) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_UNSUPP_REQ_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_UNSUPP_REQ_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        pParams->pexTotalUnsupportedReqs = (LwU8) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_L1_1_ENTRY_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_L1_1_ENTRY_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_L1_1_ENTRY_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_L1_2_ENTRY_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_L1_2_ENTRY_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_L1_2_ENTRY_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_L1_2_ABORT_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_L1_2_ABORT_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_L1_2_ABORT_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_L1SS_TO_DEEP_L1_TIMEOUT_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_L1SS_TO_DEEP_L1_TIMEOUT_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_L1SS_TO_DEEP_L1_TIMEOUT_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
    }

    if ((pParams->pexCounterMask &
        LWSWITCH_PEX_COUNTER_L1_SHORT_DURATION_COUNT))
    {
        status = lwswitch_pex_get_counter(device,
                    LWSWITCH_PEX_COUNTER_L1_SHORT_DURATION_COUNT, &cnt);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
        r = LWSWITCH_PEX_COUNTER_L1_SHORT_DURATION_COUNT;
        HIGHESTBITIDX_32(r);
        pParams->pexCounters[r] = (LwU16) cnt;
    }

    return LWL_SUCCESS;
}

