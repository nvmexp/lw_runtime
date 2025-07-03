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
// disp0206.c - Disp V02_06 display routines 
// 
//*****************************************************

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "inc/chip.h"
#include "inc/disp.h"
#include "disp/v02_06/dev_disp.h"
#include "disp/v02_01/disp0201.h"
#include "g_disp_private.h"

#include "class/cl967d.h"


// Print the ARM and ASSY values for a given EVO channel.
void
dispPrintChanMethodState_v02_06
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

#ifndef LW967D_HEAD_SET_DISPLAY_ID__SIZE_1
#define LW967D_HEAD_SET_DISPLAY_ID__SIZE_1                          2
#endif

    switch(chanId)
    {
        case CHNTYPE_CORE: // Core channel - 967D
            for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
            {
                if (coreHead >= 0 && head != (LwU32)coreHead)
                    continue;

                dprintf("----------------------------------------------------------------------------------------------\n");
                dprintf("CORE CHANNEL HEAD %u                                   ASY    |    ARM     | ASY-ARM Mismatch\n", head);
                dprintf("----------------------------------------------------------------------------------------------\n");
                //
                // The following list is based off of //sw/dev/gpu_drv/chips_a/sdk/lwpu/inc/class/cl967d.h#13
                // Note that it's implicit that the above comment applies only to core channel (d in 967d implies core)
                //
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_PRESENT_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CONTROL_OUTPUT_RESOURCE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_LOCK_OFFSET, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_OVERSCAN_COLOR, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_RASTER_SIZE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_RASTER_SYNC_END, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_RASTER_BLANK_END, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_RASTER_BLANK_START, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_RASTER_VERT_BLANK2, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_LOCK_CHAIN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_DEFAULT_BASE_COLOR, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CRC_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_LEGACY_CRC_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CONTEXT_DMA_CRC, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_BASE_LUT_LO, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_BASE_LUT_HI, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_OUTPUT_LUT_LO, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_OUTPUT_LUT_HI, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_PIXEL_CLOCK_FREQUENCY, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CONTEXT_DMA_LUT, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_OFFSET, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_SIZE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_STORAGE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_PARAMS, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CONTEXT_DMAS_ISO, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CONTROL_LWRSOR, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_PRESENT_CONTROL_LWRSOR, head, chanNum);
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW967D_HEAD_SET_OFFSETS_LWRSOR, head, 0, chanNum);
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW967D_HEAD_SET_OFFSETS_LWRSOR, head, 1, chanNum);
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW967D_HEAD_SET_CONTEXT_DMAS_LWRSOR, head, 0, chanNum);
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW967D_HEAD_SET_CONTEXT_DMAS_LWRSOR, head, 1, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_DITHER_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CONTROL_OUTPUT_SCALER, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_PROCAMP, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_VIEWPORT_POINT_IN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_VIEWPORT_SIZE_IN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_VIEWPORT_POINT_OUT_ADJUST, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_VIEWPORT_SIZE_OUT, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_VIEWPORT_SIZE_OUT_MIN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_VIEWPORT_SIZE_OUT_MAX, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_OVERLAY_USAGE_BOUNDS, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_PROCESSING, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_COLWERSION_RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_COLWERSION_GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_COLWERSION_BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CSC_RED2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CSC_GRN2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CSC_BLU2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CSC_CONSTANT2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CSC_RED2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CSC_GRN2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CSC_BLU2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CSC_CONSTANT2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CSC_RED2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CSC_GRN2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CSC_BLU2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CSC_CONSTANT2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_HDMI_CTRL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_VACTIVE_SPACE_COLOR, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_PIXEL_REORDER_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_GET_BLANKING_CTRL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_CONTROL_COMPRESSION, head, chanNum);
#ifdef LW967D_HEAD_SET_STALL_LOCK
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_STALL_LOCK, head, chanNum);
#endif
                for (k = 0; k < LW967D_HEAD_SET_DISPLAY_ID__SIZE_1; ++k)
                {
                    DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW967D_HEAD_SET_DISPLAY_ID, head, k, chanNum);
                }

                // It seems the following registers need not be printed
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_SW_METHOD_PLACEHOLDER_A, head, chanNum); 
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_SW_METHOD_PLACEHOLDER_B, head, chanNum); 
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_SW_METHOD_PLACEHOLDER_C, head, chanNum); 
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_HEAD_SET_SW_METHOD_PLACEHOLDER_D, head, chanNum); 
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
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_DAC_SET_CONTROL,           k, chanNum);
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_DAC_SET_LWSTOM_REASON,     k, chanNum);
                }
                for (k = 0; k < numSors; ++k)
                {
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_SOR_SET_CONTROL,           k, chanNum);
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_SOR_SET_LWSTOM_REASON,     k, chanNum);
                }
                for (k = 0; k < numPiors; ++k)
                {
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_PIOR_SET_CONTROL,          k, chanNum);
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW967D_PIOR_SET_LWSTOM_REASON,    k, chanNum);
                }
                DISP_PRINT_SC_NON_IDX_V02_01(LW967D_SET_CONTEXT_DMA_NOTIFIER, chanNum);
                DISP_PRINT_SC_NON_IDX_V02_01(LW967D_SET_NOTIFIER_CONTROL, chanNum);
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
