/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include <sys/types.h>
#include "inc/chip.h"
#include "inc/disp.h"
#include "class/clc67b.h"
#include "class/clc67d.h"
#include "class/clc67e.h"
#include "class/clc57d.h"
#include "disp/v04_01/dev_disp.h"
#include "disp/v04_00/disp0400.h"
#include "ampere/ga102/dev_pwr_pri.h"
#include "ampere/ga102/dev_trim.h" 
#include "g_disp_private.h"

#define DBG_BREAKPOINT()

/*!
 * @brief Print Display Low Power HW state
 */
void dispReadDisplayLowPowerStatus_v04_01()
{
    LwU32 data32;
    LwU32 win;
    LwU32 lwr;
    LwU32 head;

    dprintf("|***************** General Summary *****************\n");

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_COMMON_MISC_CTL);

    if (!FLD_TEST_DRF(_PDISP, _IHUB_COMMON_MISC_CTL, _MSCG, _ENABLE, data32))
    {
        dprintf("Display MSCG disabled. Exiting\n");
        return;
    }
    
    dprintf("Display MSCG Enabled - %s\n", FLD_TEST_DRF(_PDISP, _IHUB_COMMON_MISC_CTL, _MSCG, _ENABLE, data32) ? "Yes":"No");

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_COMMON_MSCG_MIN_CYCLE_TIME);
    dprintf("MSCG Min Cycle Time(Us) - %d\n", DRF_VAL(_PDISP, _IHUB_COMMON_MSCG_MIN_CYCLE_TIME, _USEC, data32));
    dprintf("MSCG Exit Time(Us) - %d\n", DRF_VAL(_PDISP, _IHUB_COMMON_MSCG_MIN_CYCLE_TIME, _EXIT_USEC, data32));

    dprintf("Current ihub2pmu_mscg_ok Fetch Count - %d\n", GPU_REG_RD32(LW_PDISP_IHUB_COMMON_PMU_SIGNAL_MSCG_OK_FETCH));
    dprintf("Current ihub2pmu_mscg_ok VBlank Count - %d\n", GPU_REG_RD32(LW_PDISP_IHUB_COMMON_PMU_SIGNAL_MSCG_OK_DWCF));
    dprintf("Current ihub2pmu_mscg_wake Count - %d\n", GPU_REG_RD32(LW_PDISP_IHUB_COMMON_PMU_SIGNAL_MSCG_WAKE));
    dprintf("Current MSCG Efficiency Fetch - %d us\n", GPU_REG_RD32(LW_PDISP_IHUB_COMMON_PMU_SIGNAL_MSCG_EFFICIENCY_FETCH));
    dprintf("Current MSCG Efficiency VBlank - %d us\n", GPU_REG_RD32(LW_PDISP_IHUB_COMMON_PMU_SIGNAL_MSCG_EFFICIENCY_DWCF));

    data32 = GPU_REG_RD32(LW_PPWR_PMU_GPIO_1_INPUT);
    dprintf("Current done_with_contract_fetch State - %s\n", FLD_TEST_DRF(_PPWR, _PMU_GPIO_1_INPUT, _FBH_DWCF, _TRUE, data32) ? "Asserted":"Deasserted");
    dprintf("Current ihub2pmu_ok_to_switch State - %s\n", FLD_TEST_DRF(_PPWR, _PMU_GPIO_1_INPUT, _IHUB_OK_TO_SWITCH, _TRUE, data32) ? "Asserted":"Deasserted");
    dprintf("Current ihub2pmu_mempool_draining State - %s\n", FLD_TEST_DRF(_PPWR, _PMU_GPIO_1_INPUT, _MEMPOOL_DRAINING, _TRUE, data32) ? "Asserted":"Deasserted");
    dprintf("Current ihub2pmu_mscg_wakeup State - %s\n", FLD_TEST_DRF(_PPWR, _PMU_GPIO_1_INPUT, _MSPG_WAKE, _TRUE, data32) ? "Asserted":"Deasserted");

    dprintf("|***************** Head Summary *****************\n");
    for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); head++)
    {
        dprintf("|---- Head - %d\n", head);

        data32 = GPU_REG_RD32(LW_PDISP_RG_STATUS(head));

        if (FLD_TEST_DRF(_PDISP, _RG_STATUS, _ACT_HEAD_OPMODE, _AWAKE, data32))
        {
            dprintf("|-------- Current Head State - Awake\n");
            if (FLD_TEST_DRF(_PDISP, _RG_STATUS, _STALLED, _YES, data32))
            {
                dprintf("|-------- Current RG Status - Stalled\n");
            }
            else
            {
                dprintf("|-------- Current RG Status - Not Stalled\n");
            }
            dprintf("|-------- Frame Time(Us) - %d\n", GPU_REG_RD32(LW_PDISP_IHUB_LWRS_FRAME_TIME(head)));

            data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR(head) + LWC67D_HEAD_SET_DISPLAY_RATE(head));
            if (FLD_TEST_DRF(C67D, _HEAD_SET_DISPLAY_RATE, _RUN_MODE, _ONE_SHOT, data32))
            {
                dprintf("|-------- One Shot Mode - Enabled\n");
                dprintf("|---------------- Min Refresh Interval Enabled - %s\n", FLD_TEST_DRF(C67D, _HEAD_SET_DISPLAY_RATE, _MIN_REFRESH, _ENABLE, data32) ? "Yes":"No");
                dprintf("|---------------- Min Refresh Interval(Us) - %d\n", DRF_VAL(C67D, _HEAD_SET_DISPLAY_RATE, _MIN_REFRESH_INTERVAL, data32));

                data32 = GPU_REG_RD32(LW_PDISP_RG_UNSTALL_SPOOLUP(head));
                dprintf("|---------------- RG Unstall Spool Up Delay(Pixels) - %d\n", DRF_VAL(_PDISP, _RG_UNSTALL_SPOOLUP, _VALUE, data32));

                data32 = GPU_REG_RD32(LW_PDISP_FE_ONE_SHOT_START_DELAY(head));
                dprintf("|---------------- One Shot Start Delay(Pixels) - %d\n", DRF_VAL(_PDISP, _FE_ONE_SHOT_START_DELAY, _VALUE, data32));

                data32 = GPU_REG_RD32(LW_PDISP_RG_IN_LOADV_COUNTER(head));
                dprintf("|---------------- RG LoadV Counter - %d\n", DRF_VAL(_PDISP, _RG_IN_LOADV_COUNTER, _VALUE, data32));
            }
            else
            {
                dprintf("|-------- One Shot Mode - Disabled\n");
            }
            
        }
        else
        {
            dprintf("|-------- Current Head State - Sleep/Snooze\n");
            continue;
        }
    }

    dprintf("|***************** Window Parameters *****************\n");

    for (win = 0; win < pDisp[indexGpu].dispGetNumWindows(); win++)
    {
        dprintf("|---- Window - %d\n", win);
        data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_MSCG_CTLA(win));

        if (FLD_TEST_DRF(_PDISP, _IHUB_WINDOW_MSCG_CTLA, _MODE, _ENABLE, data32))
        {
            dprintf("|-------- Enabled - Yes\n");
        }
        else
        {
            dprintf("|-------- Enabled - No\n");
            continue;
        }
        
        dprintf("|------------ OSSD Based MSCG Enabled - %s\n",  FLD_TEST_DRF(_PDISP, _IHUB_WINDOW_MSCG_CTLA, _OSM_VBLANK_MSCG, _ENABLE, data32) ? "Yes" : "No");

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_POOL_CONFIG(win));
        dprintf("|------------ Allocated Mempool - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_WINDOW_POOL_CONFIG, _ENTRIES, data32));

        dprintf("|------------ Current Mempool Oclwpancy(in Pixels) - 0x%x\n",  GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_OCC(win)));

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_MSCG_CTLA(win));
        dprintf("|------------ LWM(in Pixels) - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_WINDOW_MSCG_CTLA, _MSCG_LOW_WATERMARK, data32));

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_MSCG_CTLB(win));
        dprintf("|------------ WWM(in Pixels) - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_WINDOW_MSCG_CTLB, _MSCG_POWERUP_WATERMARK, data32));

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_MSCG_CTLC(win));
        dprintf("|------------ HWM(in Pixels) - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_WINDOW_MSCG_CTLC, _MSCG_HIGH_WATERMARK, data32));

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_CRITICAL_CTL(win));
        dprintf("|------------ CWM(in Pixels) - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_WINDOW_CRITICAL_CTL, _CRITICAL_WATERMARK, data32));

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_MSCG_LOW_WATERMARK(win));
        dprintf("|------------ Current Low Watermark Hit Count - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_WINDOW_MSCG_LOW_WATERMARK, _COUNT, data32));

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_MSCG_HIGH_WATERMARK(win));
        dprintf("|------------ Current High Watermark Hit Count - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_WINDOW_MSCG_HIGH_WATERMARK, _COUNT, data32));

        dprintf("|------------ Mempool compression factor - %dx\n", GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_UNCOMPRESSED_SIZE(win)) / 
                                                                    GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_TOTAL_SIZE(win)));
    }

    dprintf("|***************** Cursor Parameters *****************\n");

    for (lwr = 0; lwr < pDisp[indexGpu].dispGetNumHeads(); lwr++)
    {
        dprintf("|---- Cursor - %d\n", lwr);
        data32 = GPU_REG_RD32(LW_PDISP_IHUB_LWRS_MSCG_CTLA(lwr));

        if (FLD_TEST_DRF(_PDISP, _IHUB_LWRS_MSCG_CTLA, _MODE, _ENABLE, data32))
        {
            dprintf("|-------- Enabled - Yes\n");
        }
        else
        {
            dprintf("|-------- Enabled - No\n");
            continue;
        }
        
        dprintf("|------------ OSSD Based MSCG Enabled - %s\n",  FLD_TEST_DRF(_PDISP, _IHUB_LWRS_MSCG_CTLA, _OSM_VBLANK_MSCG, _ENABLE, data32) ? "Yes" : "No");

        dprintf("|------------ Current Mempool Oclwpancy(in Pixels) - 0x%x\n",  GPU_REG_RD32(LW_PDISP_IHUB_LWRS_OCC(lwr)));

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_LWRS_MSCG_CTLA(lwr));
        dprintf("|------------ LWM - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_LWRS_MSCG_CTLA, _MSCG_LOW_WATERMARK, data32));

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_LWRS_MSCG_CTLB(lwr));
        dprintf("|------------ WWM - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_LWRS_MSCG_CTLB, _MSCG_POWERUP_WATERMARK, data32));

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_LWRS_MSCG_CTLC(lwr));
        dprintf("|------------ HWM - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_LWRS_MSCG_CTLC, _MSCG_HIGH_WATERMARK, data32));

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_LWRS_MSCG_LOW_WATERMARK(lwr));
        dprintf("|------------ Current Low Watermark Hit Count - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_WINDOW_MSCG_LOW_WATERMARK, _COUNT, data32));

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_LWRS_MSCG_HIGH_WATERMARK(lwr));
        dprintf("|------------ Current High Watermark Hit Count - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_WINDOW_MSCG_HIGH_WATERMARK, _COUNT, data32));
    }
}

/*!
 * @brief Analyze Display Low Power MSCG
 */
void dispAnalyzeDisplayLowPowerMscg_v04_01()
{
    LwU32 head;
    LwU32 data32;
    LwU32 preTestMscgEfficieny;
    LwU32 postTestMscgEfficieny;
    LwU32 preTestLoadv;
    LwU32 postTestLoadv;

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_COMMON_MISC_CTL);
    if (FLD_TEST_DRF(_PDISP, _IHUB_COMMON_MISC_CTL, _MSCG, _ENABLE, data32))
    {
        dprintf("Display MSCG Enabled\n");
    }
    else
    {
        dprintf("Display MSCG disabled. Exiting!\n");
        return;
    }

    for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); head++)
    {
        data32 = GPU_REG_RD32(LW_PDISP_RG_STATUS(head));

        if (FLD_TEST_DRF(_PDISP, _RG_STATUS, _ACT_HEAD_OPMODE, _AWAKE, data32))
        {
            dprintf("Head - %d is Awake\n", head);

            preTestMscgEfficieny = GPU_REG_RD32(LW_PDISP_IHUB_COMMON_PMU_SIGNAL_MSCG_EFFICIENCY_FETCH) + 
                                   GPU_REG_RD32(LW_PDISP_IHUB_COMMON_PMU_SIGNAL_MSCG_EFFICIENCY_DWCF);
            preTestLoadv = GPU_REG_RD32(LW_PDISP_RG_IN_LOADV_COUNTER(head));

            while ((postTestLoadv = GPU_REG_RD32(LW_PDISP_RG_IN_LOADV_COUNTER(head)) - preTestLoadv) < 10)
            {
                // Wait 5 seconds
                dprintf("Waiting for 5 seconds\n");
#ifdef WIN32
                Sleep(5000);
#elif defined CLIENT_SIDE_RESMAN
        // Don't sleep in MODS with RM
#else
                sleep(5000);
#endif
            }

            postTestLoadv = GPU_REG_RD32(LW_PDISP_RG_IN_LOADV_COUNTER(head));
            postTestMscgEfficieny = GPU_REG_RD32(LW_PDISP_IHUB_COMMON_PMU_SIGNAL_MSCG_EFFICIENCY_FETCH) +
                                    GPU_REG_RD32(LW_PDISP_IHUB_COMMON_PMU_SIGNAL_MSCG_EFFICIENCY_DWCF);

            dprintf("Number of LoadV - %d us\n", postTestLoadv - preTestLoadv);
            dprintf("Avg. MSCG Efficiency per frame - %d us\n", ((postTestMscgEfficieny - preTestMscgEfficieny) / (postTestLoadv - preTestLoadv)));
        }       
    }
}


/*!
 * @brief Clear MSCG counters
 */
void dispClearDisplayLowPowerMscgCounters_v04_01()
{
    GPU_REG_WR32(LW_PDISP_IHUB_COMMON_PMU_DEBUG_CLR, 0xFFFFFFFF);
}

/*!
 * @brief Print Display Low Power MSCG Counters
 */
void dispPrintDisplayLowPowerMscgCounters_v04_01()
{
    LwU32 data32;
    LwU32 win;
    LwU32 lwr;
    LwU32 head;

    data32 = GPU_REG_RD32(LW_PDISP_IHUB_COMMON_MISC_CTL);
    if (!FLD_TEST_DRF(_PDISP, _IHUB_COMMON_MISC_CTL, _MSCG, _ENABLE, data32))
    {
        dprintf("Display MSCG disabled. Exiting\n");
        return;
    }

    dprintf("Current ihub2pmu_mscg_ok Fetch Count - %d\n", GPU_REG_RD32(LW_PDISP_IHUB_COMMON_PMU_SIGNAL_MSCG_OK_FETCH));
    dprintf("Current ihub2pmu_mscg_ok VBlank Count - %d\n", GPU_REG_RD32(LW_PDISP_IHUB_COMMON_PMU_SIGNAL_MSCG_OK_DWCF));
    dprintf("Current ihub2pmu_mscg_wake Count - %d\n", GPU_REG_RD32(LW_PDISP_IHUB_COMMON_PMU_SIGNAL_MSCG_WAKE));
    dprintf("Current MSCG Efficiency Fetch - %d us\n", GPU_REG_RD32(LW_PDISP_IHUB_COMMON_PMU_SIGNAL_MSCG_EFFICIENCY_FETCH));
    dprintf("Current MSCG Efficiency VBlank - %d us\n", GPU_REG_RD32(LW_PDISP_IHUB_COMMON_PMU_SIGNAL_MSCG_EFFICIENCY_DWCF));

    data32 = GPU_REG_RD32(LW_PPWR_PMU_GPIO_1_INPUT);
    dprintf("Current done_with_contract_fetch State - %s\n", FLD_TEST_DRF(_PPWR, _PMU_GPIO_1_INPUT, _FBH_DWCF, _TRUE, data32) ? "Asserted":"Deasserted");
    dprintf("Current ihub2pmu_ok_to_switch State - %s\n", FLD_TEST_DRF(_PPWR, _PMU_GPIO_1_INPUT, _IHUB_OK_TO_SWITCH, _TRUE, data32) ? "Asserted":"Deasserted");
    dprintf("Current ihub2pmu_mempool_draining State - %s\n", FLD_TEST_DRF(_PPWR, _PMU_GPIO_1_INPUT, _MEMPOOL_DRAINING, _TRUE, data32) ? "Asserted":"Deasserted");
    dprintf("Current ihub2pmu_mscg_wakeup State - %s\n", FLD_TEST_DRF(_PPWR, _PMU_GPIO_1_INPUT, _MSPG_WAKE, _TRUE, data32) ? "Asserted":"Deasserted");

    dprintf("|***************** Head Counters *****************\n");
    for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); head++)
    {
        dprintf("|---- Head - %d\n", head);

        data32 = GPU_REG_RD32(LW_PDISP_RG_STATUS(head));

        if (FLD_TEST_DRF(_PDISP, _RG_STATUS, _ACT_HEAD_OPMODE, _AWAKE, data32))
        {

            data32 = GPU_REG_RD32(LW_PDISP_RG_IN_LOADV_COUNTER(head));
            dprintf("|---------------- RG LoadV Counter - %d\n", DRF_VAL(_PDISP, _RG_IN_LOADV_COUNTER, _VALUE, data32));            
        }
        else
        {
            continue;
        }
    }

    dprintf("|***************** Window Counters *****************\n");

    for (win = 0; win < pDisp[indexGpu].dispGetNumWindows(); win++)
    {
        dprintf("|---- Window - %d\n", win);
        data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_MSCG_CTLA(win));

        if (!FLD_TEST_DRF(_PDISP, _IHUB_WINDOW_MSCG_CTLA, _MODE, _ENABLE, data32))
        {
            continue;
        }

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_MSCG_LOW_WATERMARK(win));
        dprintf("|------------ Current Low Watermark Hit Count - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_WINDOW_MSCG_LOW_WATERMARK, _COUNT, data32));

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_WINDOW_MSCG_HIGH_WATERMARK(win));
        dprintf("|------------ Current High Watermark Hit Count - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_WINDOW_MSCG_HIGH_WATERMARK, _COUNT, data32));
    }

    dprintf("|***************** Cursor Counters *****************\n");

    for (lwr = 0; lwr < pDisp[indexGpu].dispGetNumHeads(); lwr++)
    {
        dprintf("|---- Cursor - %d\n", lwr);
        data32 = GPU_REG_RD32(LW_PDISP_IHUB_LWRS_MSCG_CTLA(lwr));

        if (!FLD_TEST_DRF(_PDISP, _IHUB_LWRS_MSCG_CTLA, _MODE, _ENABLE, data32))
        {
            continue;
        }

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_LWRS_MSCG_LOW_WATERMARK(win));
        dprintf("|------------ Current Low Watermark Hit Count - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_LWRS_MSCG_LOW_WATERMARK, _COUNT, data32));

        data32 = GPU_REG_RD32(LW_PDISP_IHUB_LWRS_MSCG_HIGH_WATERMARK(win));
        dprintf("|------------ Current High Watermark Hit Count - 0x%x\n",  DRF_VAL(_PDISP, _IHUB_LWRS_MSCG_HIGH_WATERMARK, _COUNT, data32));
    }
}

/*!
 * @brief Translate from device class protocol value to enum
 */
ORPROTOCOL dispGetOrProtocol_v04_01(LWOR orType, LwU32 protocolValue)
{
    switch (orType)
    {
        case LW_OR_SOR:
        {
            switch (protocolValue)
            {
                case LWC67D_SOR_SET_CONTROL_PROTOCOL_SINGLE_TMDS_A:
                    return sorProtocol_SingleTmdsA;
                case LWC67D_SOR_SET_CONTROL_PROTOCOL_SINGLE_TMDS_B:
                    return sorProtocol_SingleTmdsB;
                case LWC67D_SOR_SET_CONTROL_PROTOCOL_DUAL_TMDS:
                    return sorProtocol_DualTmds;
                case LWC67D_SOR_SET_CONTROL_PROTOCOL_DP_A:
                    return sorProtocol_DpA;
                case LWC67D_SOR_SET_CONTROL_PROTOCOL_DP_B:
                    return sorProtocol_DpB;
                case LWC67D_SOR_SET_CONTROL_PROTOCOL_HDMI_FRL:
                    return sorProtocol_HdmiFrl;
                case LWC67D_SOR_SET_CONTROL_PROTOCOL_LWSTOM:
                    return sorProtocol_Lwstom;
            }
            break;
        }       
        default:
            dprintf(" %s Invalid OR type : %d ", __FUNCTION__, orType);
            return sorProtocol_SingleTmdsA;
    }
    return protocolError;
}

/*!
 * @brief Get the Channel Ctl regs.
 * @param[in]  channelClass     Class of the channel
 * @param[in]  channelInstance  Channel instance #
 * @param[out] *pChnCtl         CHNCTL register offset for the channel
 *
 * @return LW_OK on success.
 */
LW_STATUS
dispGetChannelCtlRegs_v04_01
(
    ChnType_Lwdisplay   channelClass,
    LwU32               channelInstance,
    LwU32               *pChnCtl
)
{
    LW_STATUS status = LW_OK;

    if (pChnCtl == NULL)
    {
        return LW_ERR_ILWALID_ARGUMENT;
    }

    switch (channelClass)
    {
        case LWDISPLAY_CHNTYPE_LWRS:
            if (channelInstance < LW_PDISP_FE_CHNCTL_LWRS__SIZE_1)
            {
                *pChnCtl = LW_PDISP_FE_CHNCTL_LWRS(channelInstance);
            }
            break;

        case LWDISPLAY_CHNTYPE_WINIM:
            if (channelInstance < LW_PDISP_FE_PBBASE_WINIM__SIZE_1)
            {
                *pChnCtl = LW_PDISP_FE_CHNCTL_WINIM(channelInstance);
            }
            break;

        case LWDISPLAY_CHNTYPE_CORE:
            *pChnCtl = LW_PDISP_FE_CHNCTL_CORE;
            break;

        case LWDISPLAY_CHNTYPE_WIN:
            if (channelInstance < LW_PDISP_FE_PBBASE_WIN__SIZE_1)
            {
                *pChnCtl = LW_PDISP_FE_CHNCTL_WIN(channelInstance);
            }
            break;

        default:
            status = LW_ERR_ILWALID_ARGUMENT;
            break;
    }

    return status;
}

/*!
 * @brief Check if a channel has been allocated in HW.
 *
 * @param[in] channelClass      Class of the channel
 * @param[in] channelInstance   Channel instance #
 *
 * @return LW_TRUE if it is allocated, LW_FALSE otherwise.
 */
LwBool
dispIsChannelAllocated_v04_01
(
    ChnType_Lwdisplay channelClass,
    LwU32             channelInstance
)
{
    LwBool allocated = LW_FALSE;
    LwU32 channelCtl;
    LwU32 chnCtlOffset;
    LW_STATUS status;

    status = pDisp[indexGpu].dispGetChannelCtlRegs(channelClass, channelInstance, &chnCtlOffset);
    LW_ASSERT(status == LW_OK);

    switch (channelClass)
    {
        case LWDISPLAY_CHNTYPE_LWRS:
            channelCtl = GPU_REG_RD32(chnCtlOffset);
            allocated = (FLD_TEST_DRF(_PDISP, _FE_CHNCTL_LWRS, _ALLOCATION, _ALLOCATE, channelCtl)) ? LW_TRUE : LW_FALSE;
            break;

        case LWDISPLAY_CHNTYPE_WINIM:
            channelCtl = GPU_REG_RD32(chnCtlOffset);
            allocated = (FLD_TEST_DRF(_PDISP, _FE_CHNCTL_WINIM, _ALLOCATION, _ALLOCATE, channelCtl)) ? LW_TRUE : LW_FALSE;
            break;

        case LWDISPLAY_CHNTYPE_CORE:
            channelCtl = GPU_REG_RD32(chnCtlOffset);
            allocated = (FLD_TEST_DRF(_PDISP, _FE_CHNCTL_CORE, _ALLOCATION, _ALLOCATE, channelCtl)) ? LW_TRUE : LW_FALSE;
            break;

        case LWDISPLAY_CHNTYPE_WIN:
            channelCtl = GPU_REG_RD32(chnCtlOffset);
            allocated = (FLD_TEST_DRF(_PDISP, _FE_CHNCTL_WIN, _ALLOCATION, _ALLOCATE, channelCtl)) ? LW_TRUE : LW_FALSE;
            break;

        default:
            // We shouldn't reach here even if incorrect args are supplied
            DBG_BREAKPOINT();
            break;
    }

    return allocated;
}


/*!
 * @brief dispPrintChanMethodState - Print the ARM and ASSY values for a given LwDisplay channel
 *
 *  @param[in]  LwU32               chanNum                         Channel Number
 *  @param[in]  BOOL                printHeadless                   Print headless
 *  @param[in]  BOOL                printRegsWithoutEquivMethod     Print registers without equivalent method
 *  @param[in]  LwU32               coreHead                        Head to print (for core channel)
 *  @param[in]  LwU32               coreWin                         Window to print (for core channel)
 */
void
dispPrintChanMethodState_v04_01
(
    LwU32 chanNum,
    BOOL printHeadless,
    BOOL printRegsWithoutEquivMethod,
    LwS32 coreHead,
    LwS32 coreWin
)
{
    ChnType_Lwdisplay chanId = 0;
    LwU32 numInstance, head;
    LwU32 arm, assy;
    LwU32 scIndex;
    LwU32 i = 0, k = 0;
    char classString[32];
    char commandString[64];
    GetClassNum(classString);         

    chanId = dispChanState_v04_00[chanNum].id;
    numInstance = dispChanState_v04_00[chanNum].numInstance;
    scIndex = dispChanState_v04_00[chanNum].scIndex;

#ifndef LWC67D_HEAD_SET_DISPLAY_ID__SIZE_1
#define LWC67D_HEAD_SET_DISPLAY_ID__SIZE_1                          2
#endif

#ifndef LWC67D_SET_CONTEXT_DMAS_ISO__SIZE_1
#define LWC67DSET_CONTEXT_DMAS_ISO__SIZE_1                          6
#endif

#ifndef LWC67E_SET_PLANAR_STORAGE__SIZE_1
#define LWC67E_SET_PLANAR_STORAGE__SIZE_1                           3
#endif

#ifndef LWC67E_SET_POINT_IN__SIZE_1
#define LWC67E_SET_POINT_IN__SIZE_1                                 2
#endif

#ifndef LWC67E_SET_OPAQUE_POINT_IN__SIZE_1
#define LWC67E_SET_OPAQUE_POINT_IN__SIZE_1                          4
#endif

#ifndef LWC67E_SET_OPAQUE_SIZE_IN__SIZE_1
#define LWC67E_SET_OPAQUE_SIZE_IN__SIZE_1                           4
#endif

#ifndef LWC67B_SET_POINT_OUT__SIZE_1
#define LWC67B_SET_POINT_OUT__SIZE_1                                2
#endif

#ifndef LWC67D_HEAD_SET_CONTEXT_DMA_LWRSOR__SIZE_1
#define LWC67D_HEAD_SET_CONTEXT_DMA_LWRSOR__SIZE_1                  2
#endif

#ifndef LWC67D_HEAD_SET_OFFSET_LWRSOR__SIZE_1
#define LWC67D_HEAD_SET_OFFSET_LWRSOR__SIZE_1                       2
#endif

    if (pDisp[indexGpu].dispIsChannelAllocated(chanId, numInstance) == LW_FALSE) {

        char channelStatus[70];
        switch(chanId)
        {
            case LWDISPLAY_CHNTYPE_CORE:
                sprintf(channelStatus, "CORE CHANNEL: NOT ALLOCATED");
                break;
            case LWDISPLAY_CHNTYPE_WIN:
                sprintf(channelStatus, "WINDOWS CHANNEL %d: NOT ALLOCATED", pDisp[indexGpu].dispGetWinId(chanNum));
                break;
            case LWDISPLAY_CHNTYPE_WINIM:
                sprintf(channelStatus, "WINDOWS IMMEDIATE CHANNEL %d: NOT ALLOCATED", pDisp[indexGpu].dispGetWinId(chanNum));
                break;    
            default:
                sprintf(channelStatus, "UNKNOWN CHANNEL");
                break;    
        }   
             
        dprintf("-----------------------------------------------------------------------------------------------------\n");
        dprintf("                                    %s                                                \n", channelStatus);
        dprintf("-----------------------------------------------------------------------------------------------------\n");
        return ;               
    }

    switch(chanId)
    {
        case LWDISPLAY_CHNTYPE_CORE: // Core channel - C67D
            for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
            {
                if (coreHead >= 0 && head != (LwU32)coreHead)
                    continue;

                if (coreHead < 0 && coreWin >= 0)
                    continue;

                dprintf("-----------------------------------------------------------------------------------------------------\n");
                dprintf("CORE CHANNEL HEAD %u                                          ASY    |    ARM     | ASY-ARM Mismatch\n", head);
                dprintf("-----------------------------------------------------------------------------------------------------\n");
                //
                // The following list is based off of //sw/dev/gpu_drv/chips_a/sdk/lwpu/inc/class/clC67D.h#13
                // Note that it's implicit that the above comment applies only to core channel (d in C67D implies core)
                //                
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_PRESENT_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_VGA_CRC_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_SW_SPARE_A, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_SW_SPARE_B, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_SW_SPARE_C, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_SW_SPARE_D, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_DISPLAY_RATE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_CONTROL_OUTPUT_RESOURCE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_PIXEL_CLOCK_FREQUENCY, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_PIXEL_REORDER_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_DESKTOP_COLOR_ALPHA_RED, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_DESKTOP_COLOR_GREEN_BLUE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_LOCK_OFFSET, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_OVERSCAN_COLOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_RASTER_SIZE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_RASTER_SYNC_END, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_RASTER_BLANK_END, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_RASTER_BLANK_START, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_RASTER_VERT_BLANK2, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_LOCK_CHAIN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_CRC_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_CONTEXT_DMA_CRC, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_CONTROL_LWRSOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_PRESENT_CONTROL_LWRSOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_DITHER_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_CONTROL_OUTPUT_SCALER, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_PROCAMP, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_VIEWPORT_POINT_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_VIEWPORT_SIZE_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_VIEWPORT_POINT_OUT_ADJUST, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_VIEWPORT_SIZE_OUT, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_HDMI_CTRL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_PIXEL_REORDER_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_MAX_OUTPUT_SCALE_FACTOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_HEAD_USAGE_BOUNDS, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_HDMI_AUDIO_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_DP_AUDIO_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_VIEWPORT_VALID_SIZE_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_VIEWPORT_VALID_POINT_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_FRAME_PACKED_VACTIVE_COLOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_MIN_FRAME_IDLE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_DSC_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_DSC_PPS_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_RG_MERGE, head, scIndex);
                dprintf("-----------------------------------------------------------------------------------------------------\n");
                dprintf("HDR SPECIFIC REGISTERS FOR HEAD %u                            ASY    |    ARM     | ASY-ARM Mismatch\n", head);
                dprintf("-----------------------------------------------------------------------------------------------------\n");
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_OCSC0CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_CONTEXT_DMA_OLUT, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_OLUT_FP_NORM_SCALE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_OCSC1CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION, head, scIndex);

#ifdef LWC67D_HEAD_SET_STALL_LOCK
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_STALL_LOCK, head, scIndex);
#endif
                for (k = 0; k < LWC67D_HEAD_SET_DISPLAY_ID__SIZE_1; ++k)
                {
                    DISP_PRINT_SC_DOUBLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_DISPLAY_ID, head, k, scIndex);
                }

                for (k = 0; k < LWC67D_HEAD_SET_CONTEXT_DMA_LWRSOR__SIZE_1; ++k)
                {
                   DISP_PRINT_SC_DOUBLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_CONTEXT_DMA_LWRSOR, head, k, scIndex); 
                }

                for (k = 0; k < LWC67D_HEAD_SET_OFFSET_LWRSOR__SIZE_1; ++k)
                {
                    DISP_PRINT_SC_DOUBLE_IDX_COMP_V04_00(LWC67D_HEAD_SET_OFFSET_LWRSOR, head, k, scIndex); 
                }
            }

            if (printHeadless || coreWin >= 0)
            {
                LwU32 numWindows = pDisp[indexGpu].dispGetNumWindows();

                for (k = 0; k < numWindows; ++k)
                {
                    if (coreWin >= 0 && k != (LwU32)coreWin)
                        continue;

                    dprintf("------------------------------------------------------------------------------------------------------\n");
                    dprintf("CORE CHANNEL WINDOW %u                                        ASY    |    ARM     | ASY-ARM Mismatch\n", k);
                    dprintf("------------------------------------------------------------------------------------------------------\n");

                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_WINDOW_SET_CONTROL, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_WINDOW_SET_MAX_INPUT_SCALE_FACTOR, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_WINDOW_SET_WINDOW_USAGE_BOUNDS, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS, k, scIndex);
                }
            }

            if (printHeadless == TRUE)
            {
                LwU32 numSors = pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR);

                dprintf("-----------------------------------------------------------------------------------------------------\n");
                dprintf("CORE CHANNEL HEADLESS                                        ASY    |    ARM     | ASY-ARM Mismatch\n");
                dprintf("-----------------------------------------------------------------------------------------------------\n");

                DISP_PRINT_SC_NON_IDX_V04_00(LWC67D_SET_CONTROL, scIndex);
                DISP_PRINT_SC_NON_IDX_V04_00(LWC67D_SET_INTERLOCK_FLAGS, scIndex);
                DISP_PRINT_SC_NON_IDX_V04_00(LWC67D_SET_WINDOW_INTERLOCK_FLAGS, scIndex);

                for (k = 0; k < numSors; ++k)
                {
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_SOR_SET_CONTROL,           k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67D_SOR_SET_LWSTOM_REASON,     k, scIndex);
                }
                DISP_PRINT_SC_NON_IDX_V04_00(LWC67D_SET_INTERLOCK_FLAGS, scIndex);
                DISP_PRINT_SC_NON_IDX_V04_00(LWC67D_SET_WINDOW_INTERLOCK_FLAGS, scIndex);
                DISP_PRINT_SC_NON_IDX_V04_00(LWC67D_SET_CONTEXT_DMA_NOTIFIER, scIndex);
                DISP_PRINT_SC_NON_IDX_V04_00(LWC67D_SET_NOTIFIER_CONTROL, scIndex);
            }

            if (printRegsWithoutEquivMethod == TRUE)
            {
                for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
                {
                    dprintf("----------------------------------------------------------------------------------------------\n");
                    dprintf("CORE CHANNEL HEAD %u (SC w/o equiv method)             ASY    |    ARM     | ASY-ARM Mismatch\n", head);
                    dprintf("----------------------------------------------------------------------------------------------\n");
                }

                if (printHeadless == TRUE)
                {
                    dprintf("----------------------------------------------------------------------------------------------\n");
                    dprintf("CORE CHANNEL HEADLESS (SC w/o equiv method)           ASY    |    ARM     | ASY-ARM Mismatch\n");
                    dprintf("----------------------------------------------------------------------------------------------\n");
                }
            }
            break;

        case LWDISPLAY_CHNTYPE_WIN: // Window channel - C67E
            dprintf("-----------------------------------------------------------------------------------------------------\n");
            dprintf("WINDOW CHANNEL WINDOW %u                                      ASY    |    ARM     | ASY-ARM Mismatch\n", pDisp[indexGpu].dispGetWinId(chanNum));
            dprintf("-----------------------------------------------------------------------------------------------------\n");
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_SEMAPHORE_ACQUIRE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_SEMAPHORE_ACQUIRE_HI, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_SEMAPHORE_RELEASE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_SEMAPHORE_RELEASE_HI, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_SEMAPHORE_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_CONTEXT_DMA_SEMAPHORE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_ACQ_SEMAPHORE_VALUE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_ACQ_SEMAPHORE_VALUE_HI, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_CONTEXT_DMA_ACQ_SEMAPHORE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_CONTEXT_DMA_NOTIFIER, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_NOTIFIER_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_SIZE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_STORAGE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_PARAMS, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_SIZE_IN, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_VALID_POINT_IN, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_VALID_SIZE_IN, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_SIZE_OUT, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_CONTROL_INPUT_SCALER, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_COMPOSITION_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_COMPOSITION_CONSTANT_ALPHA, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_KEY_ALPHA, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_KEY_RED_CR, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_KEY_GREEN_Y, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_KEY_BLUE_CB, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_PRESENT_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_SCAN_DIRECTION, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_TIMESTAMP_ORIGIN_LO, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_TIMESTAMP_ORIGIN_HI, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_UPDATE_TIMESTAMP_LO, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_UPDATE_TIMESTAMP_HI, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_COMPOSITION_FACTOR_SELECT, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_SYNC_POINT_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_SYNC_POINT_ACQUIRE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_CDE_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_CDE_CTB_ENTRY, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_CDE_ZBC_COLOR, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_INTERLOCK_FLAGS, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_WINDOW_INTERLOCK_FLAGS, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_CONTEXT_DMA_ILUT, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_CLAMP_RANGE, scIndex);
            dprintf("-----------------------------------------------------------------------------------------------------\n");
            dprintf("HDR SPECIFIC REGISTERS FOR WINDOW %u                          ASY    |    ARM     | ASY-ARM Mismatch\n", pDisp[indexGpu].dispGetWinId(chanNum));
            dprintf("-----------------------------------------------------------------------------------------------------\n");
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_CSC00CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_CSC0LUT_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_CSC01CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_CONTEXT_DMA_TMO_LUT, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_TMO_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_TMO_LOW_INTENSITY_ZONE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_TMO_LOW_INTENSITY_VALUE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_TMO_MEDIUM_INTENSITY_ZONE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_TMO_MEDIUM_INTENSITY_VALUE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_TMO_HIGH_INTENSITY_ZONE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_TMO_HIGH_INTENSITY_VALUE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_CSC10CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_CSC1LUT_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_CSC11CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_COMPOSITION_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_FMT_COEFFICIENT_C00, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_FMT_COEFFICIENT_C01, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_FMT_COEFFICIENT_C02, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_FMT_COEFFICIENT_C10, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_FMT_COEFFICIENT_C11, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_FMT_COEFFICIENT_C12, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_FMT_COEFFICIENT_C20, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_FMT_COEFFICIENT_C21, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_FMT_COEFFICIENT_C22, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_FMT_COEFFICIENT_C03, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_FMT_COEFFICIENT_C13, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC67E_SET_FMT_COEFFICIENT_C23, scIndex);

            for (k = 0; k < LWC67DSET_CONTEXT_DMAS_ISO__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67E_SET_CONTEXT_DMA_ISO,       k, scIndex);
            }

            for (k = 0; k < LWC67E_SET_PLANAR_STORAGE__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67E_SET_PLANAR_STORAGE,        k, scIndex);
            }

            for (k = 0; k < LWC67E_SET_POINT_IN__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67E_SET_POINT_IN,              k, scIndex);
            }

            for (k = 0; k < LWC67E_SET_OPAQUE_POINT_IN__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67E_SET_OPAQUE_POINT_IN,              k, scIndex);
            }

            for (k = 0; k < LWC67E_SET_OPAQUE_SIZE_IN__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67E_SET_OPAQUE_SIZE_IN,              k, scIndex);
            }
            break;

        case LWDISPLAY_CHNTYPE_WINIM: // Window channel - C67B
            dprintf("----------------------------------------------------------------------------------------------\n");
            dprintf("WINDOW IMMEDIATE CHANNEL WINDOW %u                     ASY    |    ARM     | ASY-ARM Mismatch\n", pDisp[indexGpu].dispGetWinId(chanNum));
            dprintf("----------------------------------------------------------------------------------------------\n");
            for (k = 0; k < LWC67B_SET_POINT_OUT__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC67B_SET_POINT_OUT,              k, scIndex);
            }
            break;

        default:
            dprintf("LwDisplay channel %u not supported.\n", chanNum);
    }
}

/*!
 * @brief Helper function to return SLI Data.
 *
 *  @param[in]  LwU32      head       Head index in DSLI_DATA structure to fill
 *  @param[in]  DSLI_DATA *pDsliData  Pointer to DSLI data structure
 */
void dispGetSliData_v04_01
(
    LwU32      head,
    DSLI_DATA *pDsliData
)
{
    pDsliData[head].DsliRgDistRndr = 0x0;        // Register Not Valid on >=GV100
    pDsliData[head].DsliRgDistRndrSyncAdv = 0x0; // Register Not Valid on >=GV100

    // As of >=lwdisplay, LW_PDISP_RG_FLIPLOCK no longer exists,
    // LW_PDISP_RG_FLIPLOCK_MAX_SWAP_LOCKOUT_SKEW can no longer be programmed,
    // and SWAP_LOCKOUT_START is in LW_PDISP_RG_SWAP_LOCKOUT.
    pDsliData[head].DsliRgFlipLock = GPU_REG_RD32(LW_PDISP_RG_SWAP_LOCKOUT(head));
    pDsliData[head].DsliRgStatus = GPU_REG_RD32(LW_PDISP_RG_STATUS(head));
    pDsliData[head].DsliRgStatusLocked = DRF_VAL(_PDISP, _RG_STATUS, _LOCKED, pDsliData[head].DsliRgStatus);

    // As of >=lwdisplay, LW_C37D_SET_CONTROL_FLIP_LOCK_PIN0_ENABLE
    // determines whether external fliplock is enabled, and
    // LW_PDISP_RG_STATUS_FLIPLOCKED is no longer meaningful.
    pDsliData[head].DsliRgStatusFlipLocked = DRF_VAL(C57D, _SET_CONTROL, _FLIP_LOCK_PIN0,
                                                     GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE +
                                                                  LWC57D_SET_CONTROL));

    pDsliData[head].DsliClkRemVpllExtRef = 0x0; //Register not used >=Ampere
    pDsliData[head].DsliClkDriverSrc = 0x0; //Register Not valid for Ampere and Hopper

    pDsliData[head].DsliHeadSetCntrl = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE + LWC57D_HEAD_SET_CONTROL(head));
    pDsliData[head].DsliHeadSetSlaveLockMode = DRF_VAL(C57D, _HEAD_SET_CONTROL, _SLAVE_LOCK_MODE, pDsliData[head].DsliHeadSetCntrl);
    pDsliData[head].DsliHeadSetMasterLockMode = DRF_VAL(C57D, _HEAD_SET_CONTROL, _MASTER_LOCK_MODE, pDsliData[head].DsliHeadSetCntrl);
    pDsliData[head].DsliHeadSetSlaveLockPin = DRF_VAL(C57D, _HEAD_SET_CONTROL, _SLAVE_LOCK_PIN, pDsliData[head].DsliHeadSetCntrl);
    pDsliData[head].DsliHeadSetMasterLockPin = DRF_VAL(C57D, _HEAD_SET_CONTROL, _MASTER_LOCK_PIN, pDsliData[head].DsliHeadSetCntrl);
}
/*!
 * @brief Prints HDMI FRL Reg info
 *
 *  @param[in]  LwU32      headNum    Head number   
 *  @param[in]  LwU32      sorNum     Sor number
 */
LW_STATUS dispCheckHdmifrlStatus_v04_01
(
    LwU32 headNum, 
    LwU32 sorNum
)
{
    LwU32 data32;
    LwU32 val;
    LW_STATUS status = LW_OK;

    dprintf("\n=================================================================================");
    dprintf("\n        Checking HDMI FRL Link control status for Head: %d, Sor: %d ", headNum, sorNum);
    dprintf("\n=================================================================================");

    data32 = GPU_REG_RD32(LW_PDISP_SOR_HDMI_FRL_LINKCTL(sorNum));
    dprintf("\n%-55s 0x%08x", "LW_PDISP_SOR_HDMI_FRL_LINKCTL : ", data32);

    if (FLD_TEST_DRF(_PDISP, _SOR_HDMI_FRL_LINKCTL, _ENABLE, _YES, data32))
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_ENABLE", "YES");
    }
    else
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_ENABLE", "NO");
    }

    switch(DRF_VAL(_PDISP, _SOR_HDMI_FRL_LINKCTL, _LANECOUNT, data32))
    {
        case LW_PDISP_SOR_HDMI_FRL_LINKCTL_LANECOUNT_ONE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_LANECOUNT", "ONE");
            break;
        case LW_PDISP_SOR_HDMI_FRL_LINKCTL_LANECOUNT_TWO:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_LANECOUNT", "TWO");
            break;
        case LW_PDISP_SOR_HDMI_FRL_LINKCTL_LANECOUNT_THREE:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_LANECOUNT", "THREE");
            break;
        case LW_PDISP_SOR_HDMI_FRL_LINKCTL_LANECOUNT_FOUR:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_LANECOUNT", "FOUR");
            break;
        default:
            dprintf("\n%-55s  0x%02x", "Unknown LW_PDISP_SOR_HDMI_FRL_LINKCTL_LANECOUNT : ",
                    DRF_VAL(_PDISP, _SOR_HDMI_FRL_LINKCTL, _LANECOUNT, data32));
            status = LW_ERR_GENERIC;
            break;
    }

    val = DRF_VAL(_PDISP, _SOR_HDMI_FRL_LINKCTL, _MIN_ACTIVE_PACKET_SIZE, data32);
    if (val == LW_PDISP_SOR_HDMI_FRL_LINKCTL_MIN_ACTIVE_PACKET_SIZE_INIT)
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_MIN_ACTIVE_PACKET_SIZE", "INIT");
    }
    else
    {
        dprintf("\n\t%-55s 0x%02x", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_MIN_ACTIVE_PACKET_SIZE : ", val);
    }

    val = DRF_VAL(_PDISP, _SOR_HDMI_FRL_LINKCTL, _MIN_BLANKING_PACKET_SIZE, data32);
    if (val == LW_PDISP_SOR_HDMI_FRL_LINKCTL_MIN_BLANKING_PACKET_SIZE_INIT)
    {
        dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_MIN_BLANKING_PACKET_SIZE", "INIT");
    }
    else
    {
        dprintf("\n\t%-55s 0x%02x", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_MIN_BLANKING_PACKET_SIZE : ", val);
    }

    switch(DRF_VAL(_PDISP, _SOR_HDMI_FRL_LINKCTL, _RD_RESET_VAL, data32))
    {
        case LW_PDISP_SOR_HDMI_FRL_LINKCTL_RD_RESET_VAL_NEG1:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_RD_RESET_VAL", "NEG1");
            break;
        case LW_PDISP_SOR_HDMI_FRL_LINKCTL_RD_RESET_VAL_NEG3:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_RD_RESET_VAL", "NEG3");
            break;
        case LW_PDISP_SOR_HDMI_FRL_LINKCTL_RD_RESET_VAL_POS1:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_RD_RESET_VAL", "POS1");
            break;
        case LW_PDISP_SOR_HDMI_FRL_LINKCTL_RD_RESET_VAL_POS3:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_RD_RESET_VAL", "POS3");
            break;
        default:
            dprintf("\n%-55s  0x%02x", "Unknown LW_PDISP_SOR_HDMI_FRL_LINKCTL_RD_RESET_VAL : ",
                    DRF_VAL(_PDISP, _SOR_HDMI_FRL_LINKCTL, _RD_RESET_VAL, data32));
            status = LW_ERR_GENERIC;
            break;
    }

    switch(DRF_VAL(_PDISP, _SOR_HDMI_FRL_LINKCTL, _RD_RESET_CYA, data32))
    {
        case LW_PDISP_SOR_HDMI_FRL_LINKCTL_RD_RESET_CYA_YES:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_RD_RESET_CYA", "YES");
            break;
        case LW_PDISP_SOR_HDMI_FRL_LINKCTL_RD_RESET_CYA_NO:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_RD_RESET_CYA", "NO");
            break;
        default:
            dprintf("\n%-55s  0x%02x", "Unknown LW_PDISP_SOR_HDMI_FRL_LINKCTL_RD_RESET_CYA : ",
                    DRF_VAL(_PDISP, _SOR_HDMI_FRL_LINKCTL, _RD_RESET_CYA, data32));
            status = LW_ERR_GENERIC;
            break;
    }

    switch(DRF_VAL(_PDISP, _SOR_HDMI_FRL_LINKCTL, _REVERSE_BIT_ORDER, data32))
    {
        case LW_PDISP_SOR_HDMI_FRL_LINKCTL_REVERSE_BIT_ORDER_YES:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_REVERSE_BIT_ORDER", "YES");
            break;
        case LW_PDISP_SOR_HDMI_FRL_LINKCTL_REVERSE_BIT_ORDER_NO:
            dprintf("\n\t%-55s %-55s", "LW_PDISP_SOR_HDMI_FRL_LINKCTL_REVERSE_BIT_ORDER", "NO");
            break;
        default:
            dprintf("\n%-55s  0x%02x", "Unknown LW_PDISP_SOR_HDMI_FRL_LINKCTL_REVERSE_BIT_ORDER : ",
                    DRF_VAL(_PDISP, _SOR_HDMI_FRL_LINKCTL, _REVERSE_BIT_ORDER, data32));
            status = LW_ERR_GENERIC;
            break;
    }

    dprintf("\n---------------------------------------------------------------------------------");
    return status;
}
