/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// disp0204.c - Disp V02_04 display routines 
// 
//*****************************************************

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "inc/chip.h"
#include "inc/disp.h"
#include "disp/v02_04/dev_disp.h"
#include "disp/v02_01/disp0201.h"
#include "g_disp_private.h"
#include "dpaux.h"

#include "class/cl947d.h"

// Print the ARM and ASSY values for a given EVO channel.
void
dispPrintChanMethodState_v02_04
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

#ifndef LW947D_HEAD_SET_DISPLAY_ID__SIZE_1
#define LW947D_HEAD_SET_DISPLAY_ID__SIZE_1                          2
#endif

    switch(chanId)
    {
        case CHNTYPE_CORE: // Core channel - 947D
            for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
            {
                if (coreHead >= 0 && head != coreHead)
                    continue;

                dprintf("----------------------------------------------------------------------------------------------\n");
                dprintf("CORE CHANNEL HEAD %u                                   ASY    |    ARM     | ASY-ARM Mismatch\n", head);
                dprintf("----------------------------------------------------------------------------------------------\n");
                //
                // The following list is based off of //sw/dev/gpu_drv/chips_a/sdk/lwpu/inc/class/cl947d.h#13
                // Note that it's implicit that the above comment applies only to core channel (d in 947d implies core)
                //
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_PRESENT_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CONTROL_OUTPUT_RESOURCE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_LOCK_OFFSET, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_OVERSCAN_COLOR, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_RASTER_SIZE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_RASTER_SYNC_END, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_RASTER_BLANK_END, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_RASTER_BLANK_START, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_RASTER_VERT_BLANK2, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_LOCK_CHAIN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_DEFAULT_BASE_COLOR, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CRC_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_LEGACY_CRC_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CONTEXT_DMA_CRC, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_BASE_LUT_LO, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_BASE_LUT_HI, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_OUTPUT_LUT_LO, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_OUTPUT_LUT_HI, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_PIXEL_CLOCK_FREQUENCY, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CONTEXT_DMA_LUT, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_OFFSET, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_SIZE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_STORAGE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_PARAMS, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CONTEXT_DMAS_ISO, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CONTROL_LWRSOR, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_PRESENT_CONTROL_LWRSOR, head, chanNum);
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW947D_HEAD_SET_OFFSETS_LWRSOR, head, 0, chanNum);
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW947D_HEAD_SET_OFFSETS_LWRSOR, head, 1, chanNum);
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW947D_HEAD_SET_CONTEXT_DMAS_LWRSOR, head, 0, chanNum);
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW947D_HEAD_SET_CONTEXT_DMAS_LWRSOR, head, 1, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_DITHER_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CONTROL_OUTPUT_SCALER, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_PROCAMP, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_VIEWPORT_POINT_IN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_VIEWPORT_SIZE_IN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_VIEWPORT_POINT_OUT_ADJUST, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_VIEWPORT_SIZE_OUT, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_VIEWPORT_SIZE_OUT_MIN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_VIEWPORT_SIZE_OUT_MAX, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_OVERLAY_USAGE_BOUNDS, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_PROCESSING, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_COLWERSION_RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_COLWERSION_GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_COLWERSION_BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CSC_RED2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CSC_GRN2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CSC_BLU2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CSC_CONSTANT2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CSC_RED2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CSC_GRN2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CSC_BLU2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CSC_CONSTANT2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CSC_RED2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CSC_GRN2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CSC_BLU2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CSC_CONSTANT2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_HDMI_CTRL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_VACTIVE_SPACE_COLOR, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_PIXEL_REORDER_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_GET_BLANKING_CTRL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_CONTROL_COMPRESSION, head, chanNum);
#ifdef LW947D_HEAD_SET_STALL_LOCK
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_STALL_LOCK, head, chanNum);
#endif
                for (k = 0; k < LW947D_HEAD_SET_DISPLAY_ID__SIZE_1; ++k)
                {
                    DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW947D_HEAD_SET_DISPLAY_ID, head, k, chanNum);
                }

                // It seems the following registers need not be printed
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_SW_METHOD_PLACEHOLDER_A, head, chanNum); 
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_SW_METHOD_PLACEHOLDER_B, head, chanNum); 
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_SW_METHOD_PLACEHOLDER_C, head, chanNum); 
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_HEAD_SET_SW_METHOD_PLACEHOLDER_D, head, chanNum); 
            }

            if (printHeadless == TRUE)
            {
                LwU32 numDacs = pDisp[indexGpu].dispGetNumOrs(LW_OR_DAC);
                LwU32 numSors = pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR);
                LwU32 numPiors = pDisp[indexGpu].dispGetNumOrs(LW_OR_PIOR);

                dprintf("----------------------------------------------------------------------------------------------\n");
                dprintf("CORE CHANNEL HEADLESS                                 ASY    |    ARM     | ASY-ARM Mismatch\n");
                dprintf("----------------------------------------------------------------------------------------------\n");
                for (k = 0; k < numDacs; ++k)
                {
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_DAC_SET_CONTROL,           k, chanNum);
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_DAC_SET_LWSTOM_REASON,     k, chanNum);
                }
                for (k = 0; k < numSors; ++k)
                {
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_SOR_SET_CONTROL,           k, chanNum);
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_SOR_SET_LWSTOM_REASON,     k, chanNum);
                }
                for (k = 0; k < numPiors; ++k)
                {
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_PIOR_SET_CONTROL,          k, chanNum);
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW947D_PIOR_SET_LWSTOM_REASON,    k, chanNum);
                }
                DISP_PRINT_SC_NON_IDX_V02_01(LW947D_SET_CONTEXT_DMA_NOTIFIER, chanNum);
                DISP_PRINT_SC_NON_IDX_V02_01(LW947D_SET_NOTIFIER_CONTROL, chanNum);
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

        case CHNTYPE_BASE: // Base channel - 947C
        case CHNTYPE_OVLY: // Ovly channel - 947E
            dispPrintChanMethodState_v02_01(chanNum, printHeadless, printRegsWithoutEquivMethod, coreHead, coreWin);
            break;

        case CHNTYPE_OVIM: // Ovim channel - 947B
            // Nothing to print.
            break;

        case CHNTYPE_LWRS: // Lwrs channel - 947A
            // Nothing to print.
            break;

        default:
            dprintf("EVO channel %u not supported.\n", chanNum);
    }
}

// Read DP_LINKCTL register & add extra information in protocol string
void
dispReadDpLinkCtl_v02_04
(
    LwU32 orNum,
    LwU32 linkIndex,
    char * protocolString
)
{
    LwU32 data = 0;
    char * buffer = (char *)malloc(256 * sizeof(char));
    LwU32 temp = 0;

   temp =  sprintf(buffer, "%s", protocolString);

    // read the DP_SOR_LINKCTL register
    data = GPU_REG_RD32(LW_PDISP_SOR_DP_LINKCTL(orNum, linkIndex));

    switch(DRF_VAL(_PDISP, _SOR_DP_LINKCTL, _LANECOUNT, data))
    {
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_ONE:
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_TWO:
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_FOUR:
            temp += sprintf(buffer+temp, "-DP");
            break;
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_EIGHT:
            temp += sprintf(buffer+temp, "-8-Lane-DP");
            break;
    }

    switch(DRF_VAL(_PDISP, _SOR_DP_LINKCTL, _FORMAT_MODE, data))
    {
        case LW_PDISP_SOR_DP_LINKCTL_FORMAT_MODE_SINGLE_STREAM:
            temp += sprintf(buffer+temp, "-SST");
            break;
        case LW_PDISP_SOR_DP_LINKCTL_FORMAT_MODE_MULTI_STREAM:
            temp += sprintf(buffer+temp, "-MST");
            break;
    }

    if (temp < 256)
    {
        strcpy(protocolString, buffer);
    }
}

/*!
 *  Function to get links driven by specified SOR.
 *
 *  @param[in]  sorIndex    Index of SOR.
 *  @param[out] pLinks      Links driven by specified SOR.
 */
BOOL dispGetLinkBySor_v02_04
(
    LwU32 sorIndx,
    LwU32 *pLinks
)
{
    PADLINK sorLinkMatrix[LW_PDISP_MAX_SOR][LW_MAX_SUBLINK] =
    {
        {PADLINK_A, PADLINK_B},
        {PADLINK_C, PADLINK_D},
        {PADLINK_G, PADLINK_NONE},
        {PADLINK_E, PADLINK_F}
    };
    LwU32 reg;

    if (sorIndx >= LW_PDISP_MAX_SOR || pLinks == NULL)
        return FALSE;

    // Update mapping if split SOR.
    if (sorIndx == 0 || sorIndx == 3)
    {
        reg = GPU_REG_RD32(LW_PDISP_CLK_REM_SOR_CTRL(0));
        if (FLD_TEST_DRF(_PDISP, _CLK_REM_SOR_CTRL, _BACKEND, _SOR3, reg))
        {
            sorLinkMatrix[0][PRIMARY]   = PADLINK_NONE;
            sorLinkMatrix[0][SECONDARY] = sorLinkMatrix[3][SECONDARY];
            sorLinkMatrix[3][SECONDARY] = PADLINK_NONE;
        }

        reg = GPU_REG_RD32(LW_PDISP_CLK_REM_SOR_CTRL(3));
        if (FLD_TEST_DRF(_PDISP, _CLK_REM_SOR_CTRL, _BACKEND, _SOR0, reg))
        {
            sorLinkMatrix[3][PRIMARY]   = sorLinkMatrix[0][PRIMARY];
            sorLinkMatrix[3][SECONDARY] = PADLINK_NONE;
            sorLinkMatrix[0][PRIMARY]   = PADLINK_NONE;
        }
    }
    else if (sorIndx == 1 || sorIndx == 2)
    {
        reg = GPU_REG_RD32(LW_PDISP_CLK_REM_SOR_CTRL(2));
        if (FLD_TEST_DRF(_PDISP, _CLK_REM_SOR_CTRL, _BACKEND, _SOR1, reg))
        {
            sorLinkMatrix[2][PRIMARY]   = sorLinkMatrix[1][SECONDARY];
            sorLinkMatrix[2][SECONDARY] = PADLINK_NONE;
            sorLinkMatrix[1][SECONDARY] = PADLINK_NONE;
        }
    }

    pLinks[PRIMARY]   = sorLinkMatrix[sorIndx][PRIMARY];
    pLinks[SECONDARY] = sorLinkMatrix[sorIndx][SECONDARY];
    return TRUE;
}

/*!
 *  Function to print configuration of specified displayport.
 *
 *  @param[in]  port        Specified AUX port.
 *  @param[in]  sorIndex    Specified SOR.
 *  @param[in]  dpIndex     Specified sublink.
 */
void dispDisplayPortInfo_v02_04
(
    LwU32 port,
    LwU32 sorIndex,
    LwU32 dpIndex
)
{
    LwU32 reg, headMask;

    if (pDpaux[indexGpu].dpauxGetHpdStatus(port))
    {
        dprintf("================================================================================\n");
        dprintf("%-55s: %s\n\n", "LW_PMGR_DP_AUXSTAT_HPD_STATUS", "PLUGGED");
    }
    else
    {
        dprintf("ERROR: %s: DP not plugged in. Bailing out early\n",
            __FUNCTION__);
        return;
    }

    // Get right SOR & sublink index.
    if (dpIndex == LW_MAX_SUBLINK)
    {
        LwU32 i, j;
        LwU32 link[LW_MAX_SUBLINK];

        for (i = 0; i < LW_MAX_SUBLINK; i++)
            link[i] = PADLINK_NONE;

        for (i = 0; i < pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR); i++)
        {
            if (pDisp[indexGpu].dispGetLinkBySor(i, link))
            {
                for (j = 0; j < LW_MAX_SUBLINK; j++)
                {
                    if (pDisp[indexGpu].dispGetAuxPortByLink(link[j]) == port)
                    {
                        sorIndex = i;
                        dpIndex = j;
                        break;
                    }
                }
                if (dpIndex != LW_MAX_SUBLINK)
                    break;
            }
        }
        if (dpIndex == LW_MAX_SUBLINK)
        {
            dprintf("ERROR: %s: Can't get corrusponding SOR & sublink.\n",
                __FUNCTION__);
            return;
        }
    }

    dprintf("Tx:\n");
    dprintf("%-55s: %d\n", "sorIndex", sorIndex);
    dprintf("%-55s: %d\n", "dpIndex", dpIndex);

    reg = GPU_REG_RD32(LW_PDISP_SOR_DP_LINKCTL(sorIndex, dpIndex));

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_ENABLE",
        DRF_VAL(_PDISP, _SOR_DP_LINKCTL, _ENABLE, reg) ? "YES" : "NO");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_ENHANCEDFRAME",
        DRF_VAL(_PDISP, _SOR_DP_LINKCTL, _ENHANCEDFRAME, reg) ?
        "ENABLED" : "DISABLED");

    switch(DRF_VAL(_PDISP, _SOR_DP_LINKCTL, _LANECOUNT, reg))
    {
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_ZERO:
            dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_LANECOUNT",
                "ZERO");
            break;
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_ONE:
            dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_LANECOUNT",
                "ONE");
            break;
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_TWO:
            dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_LANECOUNT",
                "TWO");
            break;
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_FOUR:
            dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_LANECOUNT",
                "FOUR");
            break;
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_EIGHT:
            dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_LANECOUNT",
                "EIGHT");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_SOR_DP_LINKCTL_LANECOUNT value.\n",
                __FUNCTION__);
    }

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_FORMAT_MODE",
        DRF_VAL(_PDISP, _SOR_DP_LINKCTL, _FORMAT_MODE, reg) ?
        "MULTI_STREAM" : "SINGLE_STREAM");

    reg = GPU_REG_RD32(LW_PDISP_SOR_DP_PADCTL(sorIndex, dpIndex));

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_PADCTL_PD_TXD_0",
        DRF_VAL(_PDISP, _SOR_DP_PADCTL, _PD_TXD_0, reg) ?
        "NO" : "YES");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_PADCTL_PD_TXD_1",
        DRF_VAL(_PDISP, _SOR_DP_PADCTL, _PD_TXD_1, reg) ?
        "NO" : "YES");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_PADCTL_PD_TXD_2",
        DRF_VAL(_PDISP, _SOR_DP_PADCTL, _PD_TXD_2, reg) ?
        "NO" : "YES");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_PADCTL_PD_TXD_3",
        DRF_VAL(_PDISP, _SOR_DP_PADCTL, _PD_TXD_3, reg) ?
        "NO" : "YES");

    reg = GPU_REG_RD32(LW_PDISP_CLK_REM_SOR(sorIndex));

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _DIV, reg))
    {
        case LW_PDISP_CLK_REM_SOR_DIV_BY_1:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "BY_1");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_2:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "BY_2");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_4:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "BY_4");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_8:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "BY_4");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_16:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "BY_4");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_CLK_REM_SOR_DIV value.\n",
                __FUNCTION__);
    }

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _MODE, reg))
    {
        case LW_PDISP_CLK_REM_SOR_MODE_XITION:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "XITION");
            break;
        case LW_PDISP_CLK_REM_SOR_MODE_NORMAL:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE",
                "NORMAL");
            break;
        case LW_PDISP_CLK_REM_SOR_MODE_SAFE:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE",
                "SAFE");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_CLK_REM_SOR_MODE value.\n",
                __FUNCTION__);
    }

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _PLL_REF_DIV, reg))
    {
        case LW_PDISP_CLK_REM_SOR_DIV_BY_1:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_PLL_REF_DIV",
                "BY_1");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_2:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_PLL_REF_DIV",
                "BY_2");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_4:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_PLL_REF_DIV",
                "BY_4");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_8:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_PLL_REF_DIV",
                "BY_4");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_16:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_PLL_REF_DIV",
                "BY_4");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_CLK_REM_SOR_PLL_REF_DIV value.\n",
                __FUNCTION__);
    }

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _MODE_BYPASS, reg))
    {
        case LW_PDISP_CLK_REM_SOR_MODE_BYPASS_NONE:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE_BYPASS",
                "NONE");
            break;
        case LW_PDISP_CLK_REM_SOR_MODE_BYPASS_DP_NORMAL:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE_BYPASS",
                "DP_NORMAL");
            break;
        case LW_PDISP_CLK_REM_SOR_MODE_BYPASS_DP_SAFE:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE_BYPASS",
                "DP_SAFE");
            break;
        case LW_PDISP_CLK_REM_SOR_MODE_BYPASS_FEEDBACK:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE_BYPASS",
                "FEEDBACK");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_CLK_REM_SOR_MODE_BYPASS value.\n",
                __FUNCTION__);
    }

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _LINK_SPEED, reg))
    {
        case LW_PDISP_CLK_REM_SOR_LINK_SPEED_DP_1_62GHZ:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_LINK_SPEED",
                    "1_62GHZ");
            break;
        case LW_PDISP_CLK_REM_SOR_LINK_SPEED_DP_2_70GHZ:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_LINK_SPEED",
                    "2_70GHZ");
            break;
        case LW_PDISP_CLK_REM_SOR_LINK_SPEED_DP_5_40GHZ:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_LINK_SPEED",
                    "5_40GHZ");
            break;
        default:
            dprintf("WARNING: %-55s: %x\n",
                "LW_PDISP_CLK_REM_SOR_LINK_SPEED",
                DRF_VAL(_PDISP, _CLK_REM_SOR, _LINK_SPEED, reg));
            break;
    }

    dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_STATE",
        DRF_VAL(_PDISP, _CLK_REM_SOR, _STATE, reg) ? "ENABLE" : "DISABLE");

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _CLK_SOURCE, reg))
    {
        case LW_PDISP_CLK_REM_SOR_CLK_SOURCE_SINGLE_PCLK:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_CLK_SOURCE",
                "SINGLE_PCLK");
            break;
        case LW_PDISP_CLK_REM_SOR_CLK_SOURCE_DIFF_PCLK:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_CLK_SOURCE",
                "DIFF_PCLK");
            break;
        case LW_PDISP_CLK_REM_SOR_CLK_SOURCE_SINGLE_DPCLK:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_CLK_SOURCE",
                "SINGLE_DPCLK");
            break;
        case LW_PDISP_CLK_REM_SOR_CLK_SOURCE_DIFF_DPCLK:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_CLK_SOURCE",
                "DIFF_DPCLK");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_CLK_REM_SOR_CLK_SOURCE value.\n",
                __FUNCTION__);
    }

    reg = GPU_REG_RD32(LW_PDISP_SOR_CAP(sorIndex));

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_CAP_DP_A",
        DRF_VAL(_PDISP, _SOR_CAP, _DP_A, reg) ? "TRUE" : "FALSE");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_CAP_DP_B",
        DRF_VAL(_PDISP, _SOR_CAP, _DP_B, reg) ? "TRUE" : "FALSE");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_CAP_DP_INTERFACE",
        DRF_VAL(_PDISP, _SOR_CAP, _DP_INTERLACE, reg) ? "TRUE" : "FALSE");

    reg = GPU_REG_RD32(LW_PDISP_DSI_SOR_CAP(sorIndex));

    dprintf("%-55s: %s\n", "LW_PDISP_DSI_SOR_CAP_DP_A",
        DRF_VAL(_PDISP, _DSI_SOR_CAP, _DP_A, reg) ? "TRUE" : "FALSE");

    dprintf("%-55s: %s\n", "LW_PDISP_DSI_SOR_CAP_DP_B",
        DRF_VAL(_PDISP, _DSI_SOR_CAP, _DP_B, reg) ? "TRUE" : "FALSE");

    dprintf("%-55s: %s\n", "LW_PDISP_DSI_SOR_CAP_DP_INTERLACE",
        DRF_VAL(_PDISP, _DSI_SOR_CAP, _DP_INTERLACE, reg) ? "TRUE" : "FALSE");

    reg = GPU_REG_RD32(LW_PDISP_SOR_DP_HDCP_BKSV_MSB(sorIndex, dpIndex));

    dprintf("%-55s: %x\n", "LW_PDISP_SOR_DP_HDCP_BKSV_MSB_VALUE",
        DRF_VAL(_PDISP, _SOR_DP_HDCP_BKSV_MSB, _VALUE, reg));

    dprintf("%-55s: %x\n", "LW_PDISP_SOR_DP_HDCP_BKSV_MSB_REPEATER",
        DRF_VAL(_PDISP, _SOR_DP_HDCP_BKSV_MSB, _REPEATER, reg));

    dprintf("%-55s: %x\n", "LW_PDISP_SOR_DP_HDCP_BKSV_LSB_VALUE",
        GPU_REG_RD32(LW_PDISP_SOR_DP_HDCP_BKSV_LSB(sorIndex, dpIndex)));

    dprintf("%-55s: %x\n", "LW_PDISP_SOR_DP_HDCP_RI",
            GPU_REG_RD32(LW_PDISP_SOR_DP_HDCP_RI(sorIndex, dpIndex)));

    reg = GPU_REG_RD32(LW_PDISP_SOR_DP_SPARE(sorIndex, dpIndex));

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_SPARE_SEQ_ENABLE",
        DRF_VAL(_PDISP, _SOR_DP_SPARE, _SEQ_ENABLE, reg) ? "YES" : "NO");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_SPARE_PANEL",
        DRF_VAL(_PDISP, _SOR_DP_SPARE, _PANEL, reg) ? "INTERNAL" : "EXTERNAL");

    reg = GPU_REG_RD32(LW_PDISP_SOR_TEST(sorIndex));
    dprintf("%-55s: ", "LW_PDISP_SOR_TEST");
    headMask = DRF_VAL(_PDISP, _SOR_TEST, _OWNER_MASK, reg);
    if (headMask)
    {
        BOOL bHeadOnce;
        LwU32 i;

        // Print attached HEADs
        bHeadOnce = FALSE;
        for (i = 0; i < pDisp[indexGpu].dispGetNumHeads(); i++)
        {
            if (headMask & (1 << i))
            {
                if (bHeadOnce)
                    dprintf("|%d", i);
                else
                    dprintf("HEAD %d", i);

                bHeadOnce = TRUE;
            }
        }

        if (bHeadOnce)
            dprintf("\n");
        else
            dprintf("ERROR: head index can't be recognized\n");

        // Print relevant SF registers
        for (i = 0; i < pDisp[indexGpu].dispGetNumSfs(); i++)
        {
            LwU32 reg = GPU_REG_RD32(LW_PDISP_SF_TEST(i));

            if (DRF_VAL(_PDISP, _SF_TEST, _OWNER_MASK, reg) & headMask)
            {
                dprintf("%-55s: %d\n", "sfIndex", i);

                reg = GPU_REG_RD32(LW_PDISP_SF_DP_LINKCTL(i));

                dprintf("%-55s: %s\n", "LW_PDISP_SF_DP_LINKCTL_ENABLE",
                    DRF_VAL(_PDISP, _SF_DP_LINKCTL, _ENABLE,
                    reg) ? "YES" : "NO");

                dprintf("%-55s: %s\n", "LW_PDISP_SF_DP_LINKCTL_FORMAT_MODE",
                    DRF_VAL(_PDISP, _SF_DP_LINKCTL, _FORMAT_MODE,
                    reg) ? "MULTI_STREAM" : "SINGLE_STREAM");

                dprintf("%-55s: %d\n", "LW_PDISP_SF_DP_LINKCTL_LANECOUNT",
                    DRF_VAL(_PDISP, _SF_DP_LINKCTL, _LANECOUNT, reg));

                reg = GPU_REG_RD32(LW_PDISP_SF_DP_FLUSH(i));

                dprintf("%-55s: %s\n", "LW_PDISP_SF_DP_FLUSH_ENABLE",
                    DRF_VAL(_PDISP, _SF_DP_FLUSH, _ENABLE,
                    reg) ? "YES" : "NO");

                dprintf("%-55s: %s\n", "LW_PDISP_SF_DP_FLUSH_MODE",
                    DRF_VAL(_PDISP, _SF_DP_FLUSH, _MODE,
                    reg) ? "IMMEDIATE" : "LOADV");

                dprintf("%-55s: %s\n", "LW_PDISP_SF_DP_FLUSH_CNTL",
                    DRF_VAL(_PDISP, _SF_DP_FLUSH, _CNTL,
                    reg) ? "PENDING" : "DONE");

                reg = GPU_REG_RD32(LW_PDISP_SF_DP_STREAM_CTL(i));

                dprintf("%-55s: %d\n", "LW_PDISP_SF_DP_STREAM_CTL_START",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_CTL, _START, reg));

                dprintf("%-55s: %d\n", "LW_PDISP_SF_DP_STREAM_CTL_LENGTH",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_CTL, _LENGTH, reg));

                dprintf("%-55s: %d\n",
                    "LW_PDISP_SF_DP_STREAM_CTL_START_ACTIVE",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_CTL, _START_ACTIVE, reg));

                dprintf("%-55s: %d\n",
                    "LW_PDISP_SF_DP_STREAM_CTL_LENGTH_ACTIVE",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_CTL, _LENGTH_ACTIVE, reg));

                reg = GPU_REG_RD32(LW_PDISP_SF_DP_STREAM_BW(i));

                dprintf("%-55s: %d\n", "LW_PDISP_SF_DP_STREAM_BW_ALLOCATED",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_BW, _ALLOCATED, reg));

                dprintf("%-55s: %d\n", "LW_PDISP_SF_DP_STREAM_BW_TIMESLICE",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_BW, _TIMESLICE, reg));
            }
        }
    }

    dispPrintDpRxInfo(port);
}
