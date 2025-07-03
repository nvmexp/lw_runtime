/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// disp0207.c - Disp V02_07 display routines 
// 
//*****************************************************

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "inc/chip.h"
#include "inc/disp.h"
#include "clk.h"
#include "disp/v02_07/dev_disp.h"
#include "disp/v02_01/disp0201.h"
#include "g_disp_private.h"

#include "class/cl977d.h"


// Print the ARM and ASSY values for a given EVO channel.
void
dispPrintChanMethodState_v02_07
(
    LwU32 chanNum,
    BOOL printHeadless,
    BOOL printRegsWithoutEquivMethod,
    LwS32 coreHead,
    LwS32 coreWin
)
{
    ChnType chanId;
    LwU32 headNum, head, k;
    LwU32 arm, assy;
    LwU32 i = 0;
    char classString[32];
    char commandString[52];
    GetClassNum(classString);         

    chanId = dispChanState_v02_01[chanNum].id;
    headNum = dispChanState_v02_01[chanNum].headNum;

#ifndef LW977D_HEAD_SET_DISPLAY_ID__SIZE_1
#define LW977D_HEAD_SET_DISPLAY_ID__SIZE_1                          2
#endif

    switch(chanId)
    {
        case CHNTYPE_CORE: // Core channel - 977D
            for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
            {
                if (coreHead >= 0 && head != (LwU32)coreHead)
                    continue;

                dprintf("----------------------------------------------------------------------------------------------\n");
                dprintf("CORE CHANNEL HEAD %u                                   ASY    |    ARM     | ASY-ARM Mismatch\n", head);
                dprintf("----------------------------------------------------------------------------------------------\n");
                //
                // The following list is based off of //sw/dev/gpu_drv/chips_a/sdk/lwpu/inc/class/cl977d.h#13
                // Note that it's implicit that the above comment applies only to core channel (d in 977d implies core)
                //
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_PRESENT_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CONTROL_OUTPUT_RESOURCE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_LOCK_OFFSET, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_OVERSCAN_COLOR, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_RASTER_SIZE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_RASTER_SYNC_END, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_RASTER_BLANK_END, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_RASTER_BLANK_START, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_RASTER_VERT_BLANK2, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_LOCK_CHAIN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_DEFAULT_BASE_COLOR, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CRC_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_LEGACY_CRC_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CONTEXT_DMA_CRC, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_BASE_LUT_LO, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_BASE_LUT_HI, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_OUTPUT_LUT_LO, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_OUTPUT_LUT_HI, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_PIXEL_CLOCK_FREQUENCY, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CONTEXT_DMA_LUT, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_OFFSET, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_SIZE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_STORAGE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_PARAMS, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CONTEXT_DMAS_ISO, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CONTROL_LWRSOR, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_PRESENT_CONTROL_LWRSOR, head, chanNum);
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW977D_HEAD_SET_OFFSETS_LWRSOR, head, 0, chanNum);
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW977D_HEAD_SET_OFFSETS_LWRSOR, head, 1, chanNum);
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW977D_HEAD_SET_CONTEXT_DMAS_LWRSOR, head, 0, chanNum);
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW977D_HEAD_SET_CONTEXT_DMAS_LWRSOR, head, 1, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_DITHER_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CONTROL_OUTPUT_SCALER, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_PROCAMP, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_VIEWPORT_POINT_IN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_VIEWPORT_SIZE_IN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_VIEWPORT_POINT_OUT_ADJUST, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_VIEWPORT_SIZE_OUT, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_VIEWPORT_SIZE_OUT_MIN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_VIEWPORT_SIZE_OUT_MAX, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_OVERLAY_USAGE_BOUNDS, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_PROCESSING, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_COLWERSION_RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_COLWERSION_GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_COLWERSION_BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CSC_RED2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CSC_GRN2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CSC_BLU2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CSC_CONSTANT2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CSC_RED2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CSC_GRN2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CSC_BLU2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CSC_CONSTANT2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CSC_RED2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CSC_GRN2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CSC_BLU2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CSC_CONSTANT2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_HDMI_CTRL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_VACTIVE_SPACE_COLOR, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_PIXEL_REORDER_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_GET_BLANKING_CTRL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_CONTROL_COMPRESSION, head, chanNum);
#ifdef LW977D_HEAD_SET_STALL_LOCK
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_STALL_LOCK, head, chanNum);
#endif
                for (k = 0; k < LW977D_HEAD_SET_DISPLAY_ID__SIZE_1; ++k)
                {
                    DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW977D_HEAD_SET_DISPLAY_ID, head, k, chanNum);
                }

                // It seems the following registers need not be printed
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_SW_METHOD_PLACEHOLDER_A, head, chanNum); 
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_SW_METHOD_PLACEHOLDER_B, head, chanNum); 
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_SW_METHOD_PLACEHOLDER_C, head, chanNum); 
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_HEAD_SET_SW_METHOD_PLACEHOLDER_D, head, chanNum); 
            }

            if (printHeadless == TRUE)
            {
                LwU32 numSors = pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR);
                LwU32 numPiors = pDisp[indexGpu].dispGetNumOrs(LW_OR_PIOR);

                dprintf("----------------------------------------------------------------------------------------------\n");
                dprintf("CORE CHANNEL HEADLESS                                 ASY    |    ARM     | ASY-ARM Mismatch\n");
                dprintf("----------------------------------------------------------------------------------------------\n");
                for (k = 0; k < numSors; ++k)
                {
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_SOR_SET_CONTROL,           k, chanNum);
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_SOR_SET_LWSTOM_REASON,     k, chanNum);
                }
                for (k = 0; k < numPiors; ++k)
                {
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_PIOR_SET_CONTROL,          k, chanNum);
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW977D_PIOR_SET_LWSTOM_REASON,    k, chanNum);
                }
                DISP_PRINT_SC_NON_IDX_V02_01(LW977D_SET_CONTEXT_DMA_NOTIFIER, chanNum);
                DISP_PRINT_SC_NON_IDX_V02_01(LW977D_SET_NOTIFIER_CONTROL, chanNum);
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

        case CHNTYPE_BASE: // Base channel - 927C
        case CHNTYPE_OVLY: // Ovly channel - 917E
            dispPrintChanMethodState_v02_01(chanNum, printHeadless, printRegsWithoutEquivMethod, coreHead, coreWin);
            break;

        case CHNTYPE_OVIM: // Ovim channel - 917B
            // Nothing to print.
            break;

        case CHNTYPE_LWRS: // Lwrs channel - 917A
            // Nothing to print.
            break;

        default:
            dprintf("EVO channel %u not supported.\n", chanNum);
    }
}

LwU32
dispGetMaxChan_v02_07(void)
{
    return LW_PDISP_CHANNELS;
}

void dispAnalyzeInterrupts_v02_07 (LwU32 all, LwU32 evt, LwU32 intr, LwU32 dispatch)
{
    LwU32 data32, tgt, pen, ebe, eve, en0,rmk, dmk, pmk, hind=0;
    LwU32 rip, pip, dip, rpe, dpe, ppe;
    int ind = 0;
    data32=tgt=pen=ebe=0;    
    dprintf("----------------------------------------------------------------------------------------------------------\n");
    dprintf("%s   %40s   %10s   %10s   %10s\n","NAME","PENDING?","ENABLED?","TARGET","SANITY CHECK");
    dprintf("----------------------------------------------------------------------------------------------------------\n");
    DIN_ANL_IDX(_AWAKEN, _AWAKEN_CHN, "AWAKEN_CHN_", 0);
    dprintf("----------------------------------------------------------------------------------------------------------\n");
    DIN_ANL_IDX(_EXCEPTION, _EXCEPTION_CHN, "EXCEPTION_CHN_", 0);
    dprintf("----------------------------------------------------------------------------------------------------------\n");
    DIN_ANL_IDX(_OR, _OR_SOR, "OR_SOR_", 0);
    DIN_ANL_IDX(_OR, _OR_PIOR, "OR_PIOR_", 0);
    dprintf("--------------------------------------------------------------------------------------------------------\n");
    DIN_ANL_IDX(_SV, _SV_SUPERVISOR, "SV_SUPERVISOR", 1); 
    DIN_ANL(_SV, _SV_VBIOS_RELEASE, "SV_VBIOS_RELEASE");
    DIN_ANL(_SV, _SV_VBIOS_ATTENTION, "SV_VBIOS_ATTENTION");

    #ifdef LW_CHIP_DISP_ZPW_ENABLE
    DIN_ANL(_SV, _SV_ZPW_DISABLE, "SV_ZPW_DISABLE");
    DIN_ANL(_SV, _SV_CORE_UPDATE_ARRIVE, "SV_CORE_UPDATE_ARRIVE");
    DIN_ANL(_SV, _SV_CORE_UPDATE_DONE, "SV_CORE_UPDATE_DONE");
    DIN_ANL_IDX(_SV, _SV_BASE_UPDATE_ARRIVE, "SV_BASE_UPDATE_ARRIVE_", 0);
    DIN_ANL_IDX(_SV, _SV_BASE_UPDATE_DONE, "SV_BASE_UPDATE_DONE_", 0);
    DIN_ANL_IDX(_SV, _SV_OVLY_UPDATE_ARRIVE, "SV_OVLY_UPDATE_ARRIVE_", 0);
    DIN_ANL_IDX(_SV, _SV_OVLY_UPDATE_DONE, "SV_OVLY_UPDATE_DONE_", 0);
    #endif
    //DIN_ANL(_SV, _SV_PMU_DIRECT, "SV_PMU_DIRECT");
    DIN_POPU(_SV);
    pen = (DRF_VAL(_PDISP, _DSI_EVENT, _SV_PMU_DIRECT, eve) == LW_PDISP_DSI_EVENT_SV_PMU_DIRECT_PENDING);
    rpe=(DRF_VAL(_PDISP,_DSI_RM_INTR,_SV_PMU_DIRECT,rip)==LW_PDISP_DSI_RM_INTR_SV_PMU_DIRECT_PENDING);
    //ppe=(DRF_VAL(_PDISP,_DSI_PMU_INTR,_SV_PMU_DIRECT,pip)==LW_PDISP_DSI_PMU_INTR_SV_PMU_DIRECT_PENDING);
    ppe=0;
    dpe=(DRF_VAL(_PDISP,_DSI_DPU_INTR,_SV_PMU_DIRECT,dip)==LW_PDISP_DSI_DPU_INTR_SV_PMU_DIRECT_PENDING);
    ebe=(DRF_VAL(_PDISP,_DSI_RM_INTR_EN0,_SV_PMU_DIRECT,en0)==LW_PDISP_DSI_RM_INTR_EN0_SV_PMU_DIRECT_ENABLE);
    if(DRF_VAL(_PDISP, _DSI_RM_INTR_MSK, _SV_PMU_DIRECT, rmk))
        tgt = INTR_TGT_RM;
    else if(DRF_VAL(_PDISP, _DSI_DPU_INTR_MSK, _SV_PMU_DIRECT, dmk))
        tgt = INTR_TGT_DPU;
    else
        tgt = INTR_TGT_NONE;
    DIN_PRINT("SV_PMU_DIRECT",ebe,pen,tgt);
    dprintf("%s\n",DIN_SANITY(pen, tgt, rpe, ppe, dpe));

    //DIN_ANL(_SV, _SV_DPU_DIRECT, "SV_DPU_DIRECT");
    DIN_POPU(_SV);
    pen = (DRF_VAL(_PDISP, _DSI_EVENT, _SV_DPU_DIRECT, eve) == LW_PDISP_DSI_EVENT_SV_DPU_DIRECT_PENDING);
    rpe=(DRF_VAL(_PDISP,_DSI_RM_INTR,_SV_DPU_DIRECT,rip)==LW_PDISP_DSI_RM_INTR_SV_DPU_DIRECT_PENDING);
    ppe=(DRF_VAL(_PDISP,_DSI_PMU_INTR,_SV_DPU_DIRECT,pip)==LW_PDISP_DSI_PMU_INTR_SV_DPU_DIRECT_PENDING);
    //dpe=(DRF_VAL(_PDISP,_DSI_DPU_INTR,_SV_DPU_DIRECT,dip)==LW_PDISP_DSI_DPU_INTR_SV_DPU_DIRECT__PENDING);
    dpe=0;
    ebe=(DRF_VAL(_PDISP,_DSI_RM_INTR_EN0,_SV_DPU_DIRECT,en0)==LW_PDISP_DSI_RM_INTR_EN0_SV_DPU_DIRECT_ENABLE);
    if(DRF_VAL(_PDISP, _DSI_RM_INTR_MSK, _SV_DPU_DIRECT, rmk))
        tgt = INTR_TGT_RM;
    else if(DRF_VAL(_PDISP, _DSI_PMU_INTR_MSK, _SV_DPU_DIRECT, pmk))
        tgt = INTR_TGT_PMU;
    else
        tgt = INTR_TGT_NONE;
    DIN_PRINT("SV_DPU_DIRECT",ebe,pen,tgt);
    dprintf("%s\n",DIN_SANITY(pen, tgt, rpe, ppe, dpe));

    DIN_ANL(_SV, _SV_TIMEOUT, "SV_TIMEOUT");
    dprintf("--------------------------------------------------------------------------------------------------------\n");
    while(hind < LW_PDISP_DSI_EVENT_HEAD__SIZE_1)
    {
        if(!(DRF_VAL(_PDISP, _DSI_SYS_CAP, _HEAD_EXISTS(hind), GPU_REG_RD32(LW_PDISP_DSI_SYS_CAP)) ==
               LW_PDISP_DSI_SYS_CAP_HEAD_EXISTS_YES))
        {
               dprintf("HEAD%d doesn't exist\n",hind);
               ++hind;
               continue;
        }
        dprintf("----------------------------------------------------HEAD%d-----------------------------------------------\n",hind);
        DIN_ANL(_HEAD(hind), _HEAD_VBLANK, "HEAD_VBLANK");
        DIN_ANL(_HEAD(hind), _HEAD_HBLANK, "HEAD_HBLANK");

        #ifdef LW_CHIP_DISP_PBUF_LARGE_LATENCY_BUFFER
        DIN_ANL(_HEAD(hind), _HEAD_PBUF_UFLOW, "HEAD_PBUF_UFLOW");
        DIN_ANL(_HEAD(hind), _HEAD_PBUF_UNRECOVERABLE_UFLOW, "HEAD_PBUF_UNRECOVERABLE_UFLOW");
        #endif
        DIN_ANL(_HEAD(hind), _HEAD_RG_UNDERFLOW, "HEAD_RG_UNDERFLOW");

        #ifdef LW_CHIP_DISP_LWDPS_1_5
        DIN_ANL(_HEAD(hind), _HEAD_LWDDS_STATISTIC_COUNTERS_MSB_SET, "HEAD_LWDDS_STATISTIC_COUNTERS_MSB_SET");
        DIN_ANL_IDX(_HEAD(hind), _HEAD_LWDDS_STATISTIC_GATHER, "HEAD_LWDDS_STATISTIC_GATHER",0);
        DIN_ANL_IDX(_HEAD(hind), _HEAD_LWDDS_STATISTIC_GATHER_UPPER_BOUND, "HEAD_LWDDS_STATISTIC_GATHER_UPPER_BOUND",0);
        DIN_ANL_IDX(_HEAD(hind), _HEAD_LWDDS_STATISTIC_GATHER_LOWER_BOUND, "HEAD_LWDDS_STATISTIC_GATHER_LOWER_BOUND",0);
        #endif
        DIN_ANL(_HEAD(hind), _HEAD_SD3_BUCKET_WALK_DONE, "HEAD_SD3_BUCKET_WALK_DONE");
        DIN_ANL(_HEAD(hind), _HEAD_RG_VBLANK, "HEAD_RG_VBLANK");
        #ifdef LW_CHIP_DISP_ZPW_ENABLE
        DIN_ANL(_HEAD(hind), _HEAD_RG_ZPW_CRC_ERROR, "HEAD_RG_ZPW_CRC_ERROR");
        #endif
        DIN_ANL(_HEAD(hind), _HEAD_PMU_DMI_LINE, "HEAD_PMU_DMI_LINE");
        DIN_ANL(_HEAD(hind), _HEAD_PMU_RG_LINE, "HEAD_PMU_RG_LINE");
        DIN_ANL(_HEAD(hind), _HEAD_RM_DMI_LINE, "HEAD_RM_DMI_LINE");
        DIN_ANL(_HEAD(hind), _HEAD_RM_RG_LINE, "HEAD_RM_RG_LINE");
        dprintf("--------------------------------------------------------------------------------------------------------\n");
        ++hind;
    }
}

/**
 * @brief Read PixelClk settings.
 *
 * @returns void
 */
void
dispReadPixelClkSettings_v02_07(void)
{
    LwU32 regVal;
    LwU32 VPLL;
    LwU32 rgMode;
    LwU32 rgDiv;
    LwU32 idx;

    dprintf("lw: All PixelClk settings\n");

    for (idx = 0; idx < pDisp[indexGpu].dispGetNumHeads(); idx++)
    {
        // Read RG settings
        regVal = GPU_REG_RD32(LW_PDISP_CLK_REM_RG(idx));

        // Check if the RG is enabled
        if (FLD_TEST_DRF(_PDISP, _CLK_REM_RG, _STATE, _DISABLE, regVal))
        {
            dprintf("lw:   RG%d[ RG_STATE: DISABLE ]\n", idx);
            dprintf("lw: RG%d_PCLK = N/A\n\n", idx);
            continue;
        }

        // Check which RG mode is selected
        rgMode = DRF_VAL(_PDISP, _CLK_REM_RG, _MODE, regVal);
        if (rgMode == LW_PDISP_CLK_REM_RG_MODE_XITION)
        {
            dprintf("lw:   RG%d[ RG_MODE: XITION ]\n", idx);
            dprintf("lw: RG%d_PCLK = N/A\n\n", idx);
        }
        else if (rgMode == LW_PDISP_CLK_REM_RG_MODE_SAFE)
        {
            dprintf("lw:   RG%d[ RG_MODE: SAFE ]\n", idx);
            dprintf("lw: RG%d_PCLK = %4d MHz\n\n",
                    idx, pClk[indexGpu].clkReadCrystalFreqKHz() / 1000);
        }
        else if (rgMode == LW_PDISP_CLK_REM_RG_MODE_NORMAL)
        {
            // Read the VPLL settings
            VPLL = pClk[indexGpu].clkGetVClkFreqKHz(idx) / 1000;
            dprintf("lw:   VPLL%d = %4d MHz\n", idx, VPLL);

            // Read the RG_DIV settings
            switch (DRF_VAL(_PDISP, _CLK_REM_RG, _DIV, regVal))
            {
                case LW_PDISP_CLK_REM_RG_DIV_BY_1:
                    rgDiv = 1;
                    dprintf("lw:   RG%d[ RG_DIV: BY_1 ]\n", idx);
                    break;

                case LW_PDISP_CLK_REM_RG_DIV_BY_2:
                    rgDiv = 2;
                    dprintf("lw:   RG%d[ RG_DIV: BY_2 ]\n", idx);
                    break;

                case LW_PDISP_CLK_REM_RG_DIV_BY_3:
                    rgDiv = 3;
                    dprintf("lw:   RG%d[ RG_DIV: BY_3 ]\n", idx);
                    break;

                case LW_PDISP_CLK_REM_RG_DIV_BY_4:
                    rgDiv = 4;
                    dprintf("lw:   RG%d[ RG_DIV: BY_4 ]\n", idx);
                    break;

                case LW_PDISP_CLK_REM_RG_DIV_BY_6:
                    rgDiv = 6;
                    dprintf("lw:   RG%d[ RG_DIV: BY_6 ]\n", idx);
                    break;

                case LW_PDISP_CLK_REM_RG_DIV_BY_8:
                    rgDiv = 8;
                    dprintf("lw:   RG%d[ RG_DIV: BY_8 ]\n", idx);
                    break;

                case LW_PDISP_CLK_REM_RG_DIV_BY_16:
                    rgDiv = 16;
                    dprintf("lw:   RG%d[ RG_DIV: BY_16 ]\n", idx);
                    break;

                default:
                    rgDiv = DRF_VAL(_PDISP, _CLK_REM_RG, _DIV, regVal) + 1;
                    dprintf("lw:   RG%d[ RG_DIV: invalid enum (%d) ]\n",
                            idx, DRF_VAL(_PDISP, _CLK_REM_RG, _DIV, regVal));
                    break;
            }
            dprintf("lw: RG%d_PCLK = %4d MHz\n\n", idx, (VPLL / rgDiv));
        }
        else
        {
            dprintf("lw:   RG%d[ RG_MODE: invalid enum (%d) ]\n", idx, rgMode);
            dprintf("lw: RG%d_PCLK = N/A\n\n", idx);
        }
    }
}

/**
 * @brief Read SorClk settings.
 *
 * @returns void
 */
void
dispReadSorClkSettings_v02_07(void)
{
    LwU32 regVal;
    LwU32 VPLL;
    LwU32 idx;
    LwU32 headNum;
    LwU32 sorMode;
    LwU32 sorDiv;
    LwU32 sorPllRefDiv;
    LwU32 sorModeBypass;
    LwU32 sorLinkSpeed;
    LwU32 sorClk;

    dprintf("lw: All SorClk settings\n");

    for (idx = 0; idx < pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR); idx++)
    {
        // Read SOR settings
        regVal = GPU_REG_RD32(LW_PDISP_CLK_REM_SOR(idx));

        // Check if the SOR is enabled
        if (FLD_TEST_DRF(_PDISP, _CLK_REM_SOR, _STATE, _DISABLE, regVal))
        {
            dprintf("lw:   SOR%d[ SOR_STATE: DISABLE ]\n", idx);
            dprintf("lw: SOR%d_CLK = N/A\n\n", idx);
            continue;
        }

        // Check which SOR mode is selected
        sorMode = DRF_VAL(_PDISP, _CLK_REM_SOR, _MODE, regVal);
        if (sorMode == LW_PDISP_CLK_REM_SOR_MODE_XITION)
        {
            dprintf("lw:   SOR%d[ SOR_MODE: XITION ]\n", idx);
            dprintf("lw: SOR%d_CLK = N/A\n\n", idx);
        }
        else if (sorMode == LW_PDISP_CLK_REM_SOR_MODE_SAFE)
        {
            dprintf("lw:   SOR%d[ SOR_MODE: SAFE ]\n", idx);
            dprintf("lw: SOR%d_CLK = %4d MHz\n\n",
                    idx, pClk[indexGpu].clkReadCrystalFreqKHz() / 1000);
        }
        else if (sorMode == LW_PDISP_CLK_REM_SOR_MODE_NORMAL)
        {
            headNum = DRF_VAL(_PDISP, _CLK_REM_SOR, _HEAD, regVal);
            if (headNum == LW_PDISP_CLK_REM_SOR_HEAD_NONE)
            {
                dprintf("lw:   SOR%d[ SOR_HEAD: NONE ]\n", idx);
                continue;
            }
            if (headNum > LW_PDISP_CLK_REM_SOR_HEAD_3)
            {
                dprintf("lw:   SOR%d[ SOR_HEAD: invalid enum (%d) ]\n", idx, headNum);
                continue;
            }

            sorLinkSpeed = DRF_VAL(_PDISP, _CLK_REM_SOR, _LINK_SPEED, regVal);
            sorModeBypass = DRF_VAL(_PDISP, _CLK_REM_SOR, _MODE_BYPASS, regVal);

            if (sorModeBypass == LW_PDISP_CLK_REM_SOR_MODE_BYPASS_NONE)
            {
                VPLL = pClk[indexGpu].clkGetVClkFreqKHz(headNum) / 1000;
                dprintf("lw:   VPLL%d[ %4d MHz ]\n", headNum, VPLL);
                dprintf("lw:   SOR%d[ SOR_MODE_BYPASS: NONE ]\n", idx);

                // sorClk = (VPLL freq / SOR_DIV) * LINK_SPEED / 10
                switch (DRF_VAL(_PDISP, _CLK_REM_SOR, _DIV, regVal))
                {
                    case LW_PDISP_CLK_REM_SOR_DIV_BY_1:
                        sorDiv = 1;
                        dprintf("lw:   SOR%d[ SOR_DIV: BY_1 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_DIV_BY_2:
                        sorDiv = 2;
                        dprintf("lw:   SOR%d[ SOR_DIV: BY_2 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_DIV_BY_4:
                        sorDiv = 4;
                        dprintf("lw:   SOR%d[ SOR_DIV: BY_4 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_DIV_BY_8:
                        sorDiv = 8;
                        dprintf("lw:   SOR%d[ SOR_DIV: BY_8 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_DIV_BY_16:
                        sorDiv = 16;
                        dprintf("lw:   SOR%d[ SOR_DIV: BY_16 ]\n", idx);
                        break;

                    default:
                        sorDiv = 2 << DRF_VAL(_PDISP, _CLK_REM_SOR, _DIV, regVal);
                        dprintf("lw:   SOR%d[ SOR_DIV: invalid enum (%d) ]\n",
                                idx, DRF_VAL(_PDISP, _CLK_REM_SOR, _DIV, regVal));
                        break;
                }
                sorClk = (VPLL / sorDiv) * (sorLinkSpeed / 10);
                dprintf("lw: SOR%d_CLK = %4d MHz\n\n", idx, sorClk);
            }
            else if (sorModeBypass == LW_PDISP_CLK_REM_SOR_MODE_BYPASS_FEEDBACK)
            {
                VPLL = pClk[indexGpu].clkGetVClkFreqKHz(headNum) / 1000;
                dprintf("lw:   VPLL%d[ %4d MHz ]\n", headNum, VPLL);
                dprintf("lw:   SOR%d[ SOR_MODE_BYPASS: FEEDBACK ]\n", idx);

                // sorClk = (VPLL freq / SOR_PLL_REF_DIV) * LINK_SPEED / 10
                switch (DRF_VAL(_PDISP, _CLK_REM_SOR, _PLL_REF_DIV, regVal))
                {
                    case LW_PDISP_CLK_REM_SOR_PLL_REF_DIV_BY_1:
                        sorPllRefDiv = 1;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: BY_1 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_PLL_REF_DIV_BY_2:
                        sorPllRefDiv = 2;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: BY_2 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_PLL_REF_DIV_BY_4:
                        sorPllRefDiv = 4;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: BY_4 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_PLL_REF_DIV_BY_8:
                        sorPllRefDiv = 8;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: BY_8 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_PLL_REF_DIV_BY_16:
                        sorPllRefDiv = 16;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: BY_16 ]\n", idx);
                        break;

                    default:
                        sorPllRefDiv = 2 << DRF_VAL(_PDISP, _CLK_REM_SOR, _PLL_REF_DIV, regVal);
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: invalid enum (%d) ]\n",
                                idx, DRF_VAL(_PDISP, _CLK_REM_SOR, _PLL_REF_DIV, regVal));
                        break;
                }
                sorClk = (VPLL / sorPllRefDiv) * (sorLinkSpeed / 10);
                dprintf("lw: SOR%d_CLK = %4d MHz\n\n", idx, sorClk);
            }
            else if (sorModeBypass == LW_PDISP_CLK_REM_SOR_MODE_BYPASS_DP_NORMAL)
            {
                dprintf("lw:   SOR%d[ SOR_MODE_BYPASS: DP_NORMAL ]\n", idx);

                // sorClk uses DP pad macro feedback clock
                switch (sorLinkSpeed)
                {
                    case LW_PDISP_CLK_REM_SOR_LINK_SPEED_DP_1_62GHZ:
                        dprintf("lw: SOR%d_CLK = 162 MHz\n\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_LINK_SPEED_DP_2_70GHZ:
                        dprintf("lw: SOR%d_CLK = 270 MHz\n\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_LINK_SPEED_DP_5_40GHZ:
                        dprintf("lw: SOR%d_CLK = 540 MHz\n\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_LINK_SPEED_DP_8_10GHZ:
                        dprintf("lw: SOR%d_CLK = 810 MHz\n\n", idx);
                        break;

                    default:
                        dprintf("lw:   SOR%d[ SOR_LINK_SPEED: invalid enum (%d) ]\n",
                                idx, sorLinkSpeed);
                        dprintf("lw: SOR%d_CLK = N/A\n\n", idx);
                        break;
                }
            }
            else if (sorModeBypass == LW_PDISP_CLK_REM_SOR_MODE_BYPASS_DP_SAFE)
            {
                // sorClk is Xtal safe clock
                dprintf("lw:   SOR%d[ SOR_MODE_BYPASS: DP_SAFE ]\n", idx);
                dprintf("lw: SOR%d_CLK = %4d MHz\n\n",
                        idx, pClk[indexGpu].clkReadCrystalFreqKHz() / 1000);
            }
            else
            {
                dprintf("lw:   SOR%d[ SOR_MODE_BYPASS: invalid enum (%d) ]\n",
                        idx, sorModeBypass);
                dprintf("lw: SOR%d_CLK = N/A\n\n", idx);
            }
        }
        else
        {
            dprintf("lw:   SOR%d[ SOR_MODE: invalid enum (%d) ]\n", idx, sorMode);
            dprintf("lw: SOR%d_CLK = N/A\n\n", idx);
        }
    }
}

LwU32
dispGetNumOrs_v02_07(LWOR orType)
{
    // Display on GP100 does not aclwrately report DACs.  DAC is actually
    // defeatured in GP100, but the logic wasn't updated on silicon, so
    // capabilities reported through both LW_PDISP_DSI_SYS_CAP and
    // LW_PDISP_CLK_REM_MISC_CONFIGA are wrong.
    // Instead, we hard-code the number of DACs to 0 here.  This is fixed in
    // GP10x (to actually report zero DACs correctly) -- see bug 200133797.
    if (orType == LW_OR_DAC)
    {
        return 0;
    }

    return dispGetNumOrs_v02_02(orType);
}
