/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2019 by LWPU Corporation.  All rights reserved.  All
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
#include "disp/v04_00/dev_disp.h"
#include "disp/v04_00/disp0400.h"
#include "class/clc57b.h"
#include "class/clc57d.h"
#include "class/clc57e.h"
#include "turing/tu102/dev_trim.h"
#include "g_disp_private.h"

static LwS32 _disp2sCompToDec(LwU32 size, LwU32 data32);

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
dispPrintChanMethodState_v04_00
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

#ifndef LWC57D_HEAD_SET_DISPLAY_ID__SIZE_1
#define LWC57D_HEAD_SET_DISPLAY_ID__SIZE_1                          2
#endif

#ifndef LWC57D_SET_CONTEXT_DMAS_ISO__SIZE_1
#define LWC57DSET_CONTEXT_DMAS_ISO__SIZE_1                          6
#endif

#ifndef LWC57E_SET_PLANAR_STORAGE__SIZE_1
#define LWC57E_SET_PLANAR_STORAGE__SIZE_1                           3
#endif

#ifndef LWC57E_SET_POINT_IN__SIZE_1
#define LWC57E_SET_POINT_IN__SIZE_1                                 2
#endif

#ifndef LWC57E_SET_OPAQUE_POINT_IN__SIZE_1
#define LWC57E_SET_OPAQUE_POINT_IN__SIZE_1                          4
#endif

#ifndef LWC57E_SET_OPAQUE_SIZE_IN__SIZE_1
#define LWC57E_SET_OPAQUE_SIZE_IN__SIZE_1                           4
#endif

#ifndef LWC57B_SET_POINT_OUT__SIZE_1
#define LWC57B_SET_POINT_OUT__SIZE_1                                2
#endif

#ifndef LWC57D_HEAD_SET_CONTEXT_DMA_LWRSOR__SIZE_1
#define LWC57D_HEAD_SET_CONTEXT_DMA_LWRSOR__SIZE_1                  2
#endif

#ifndef LWC57D_HEAD_SET_OFFSET_LWRSOR__SIZE_1
#define LWC57D_HEAD_SET_OFFSET_LWRSOR__SIZE_1                       2
#endif

    switch(chanId)
    {
        case LWDISPLAY_CHNTYPE_CORE: // Core channel - C57D
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
                // The following list is based off of //sw/dev/gpu_drv/chips_a/sdk/lwpu/inc/class/clC57D.h#13
                // Note that it's implicit that the above comment applies only to core channel (d in C57D implies core)
                //                
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_PRESENT_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_VGA_CRC_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_SW_SPARE_A, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_SW_SPARE_B, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_SW_SPARE_C, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_SW_SPARE_D, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_DISPLAY_RATE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_CONTROL_OUTPUT_RESOURCE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_PIXEL_CLOCK_FREQUENCY, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_PIXEL_REORDER_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_DESKTOP_COLOR_ALPHA_RED, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_DESKTOP_COLOR_GREEN_BLUE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_LOCK_OFFSET, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_OVERSCAN_COLOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_RASTER_SIZE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_RASTER_SYNC_END, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_RASTER_BLANK_END, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_RASTER_BLANK_START, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_RASTER_VERT_BLANK2, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_LOCK_CHAIN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_CRC_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_CONTEXT_DMA_CRC, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_CONTROL_LWRSOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_PRESENT_CONTROL_LWRSOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_DITHER_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_CONTROL_OUTPUT_SCALER, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_PROCAMP, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_VIEWPORT_POINT_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_VIEWPORT_SIZE_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_VIEWPORT_POINT_OUT_ADJUST, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_VIEWPORT_SIZE_OUT, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_HDMI_CTRL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_PIXEL_REORDER_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_MAX_OUTPUT_SCALE_FACTOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_HEAD_USAGE_BOUNDS, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_HDMI_AUDIO_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_DP_AUDIO_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_VIEWPORT_VALID_SIZE_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_VIEWPORT_VALID_POINT_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_FRAME_PACKED_VACTIVE_COLOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_MIN_FRAME_IDLE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_DSC_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_DSC_PPS_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_RG_MERGE, head, scIndex);
                dprintf("-----------------------------------------------------------------------------------------------------\n");
                dprintf("HDR SPECIFIC REGISTERS FOR HEAD %u                            ASY    |    ARM     | ASY-ARM Mismatch\n", head);
                dprintf("-----------------------------------------------------------------------------------------------------\n");
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_OCSC0CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_CONTEXT_DMA_OLUT, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_OLUT_FP_NORM_SCALE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_OCSC1CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION, head, scIndex);

#ifdef LWC57D_HEAD_SET_STALL_LOCK
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_STALL_LOCK, head, scIndex);
#endif
                for (k = 0; k < LWC57D_HEAD_SET_DISPLAY_ID__SIZE_1; ++k)
                {
                    DISP_PRINT_SC_DOUBLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_DISPLAY_ID, head, k, scIndex);
                }

                for (k = 0; k < LWC57D_HEAD_SET_CONTEXT_DMA_LWRSOR__SIZE_1; ++k)
                {
                   DISP_PRINT_SC_DOUBLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_CONTEXT_DMA_LWRSOR, head, k, scIndex); 
                }

                for (k = 0; k < LWC57D_HEAD_SET_OFFSET_LWRSOR__SIZE_1; ++k)
                {
                    DISP_PRINT_SC_DOUBLE_IDX_COMP_V04_00(LWC57D_HEAD_SET_OFFSET_LWRSOR, head, k, scIndex); 
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

                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_WINDOW_SET_CONTROL, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_WINDOW_SET_MAX_INPUT_SCALE_FACTOR, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_WINDOW_SET_WINDOW_USAGE_BOUNDS, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS, k, scIndex);
                }
            }

            if (printHeadless == TRUE)
            {
                LwU32 numSors = pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR);

                dprintf("-----------------------------------------------------------------------------------------------------\n");
                dprintf("CORE CHANNEL HEADLESS                                        ASY    |    ARM     | ASY-ARM Mismatch\n");
                dprintf("-----------------------------------------------------------------------------------------------------\n");

                DISP_PRINT_SC_NON_IDX_V04_00(LWC57D_SET_CONTROL, scIndex);
                DISP_PRINT_SC_NON_IDX_V04_00(LWC57D_SET_INTERLOCK_FLAGS, scIndex);
                DISP_PRINT_SC_NON_IDX_V04_00(LWC57D_SET_WINDOW_INTERLOCK_FLAGS, scIndex);

                for (k = 0; k < numSors; ++k)
                {
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_SOR_SET_CONTROL,           k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57D_SOR_SET_LWSTOM_REASON,     k, scIndex);
                }
                DISP_PRINT_SC_NON_IDX_V04_00(LWC57D_SET_INTERLOCK_FLAGS, scIndex);
                DISP_PRINT_SC_NON_IDX_V04_00(LWC57D_SET_WINDOW_INTERLOCK_FLAGS, scIndex);
                DISP_PRINT_SC_NON_IDX_V04_00(LWC57D_SET_CONTEXT_DMA_NOTIFIER, scIndex);
                DISP_PRINT_SC_NON_IDX_V04_00(LWC57D_SET_NOTIFIER_CONTROL, scIndex);
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

        case LWDISPLAY_CHNTYPE_WIN: // Window channel - C57E
            dprintf("-----------------------------------------------------------------------------------------------------\n");
            dprintf("WINDOW CHANNEL WINDOW %u                                      ASY    |    ARM     | ASY-ARM Mismatch\n", pDisp[indexGpu].dispGetWinId(chanNum));
            dprintf("-----------------------------------------------------------------------------------------------------\n");
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_SEMAPHORE_ACQUIRE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_SEMAPHORE_RELEASE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_SEMAPHORE_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_CONTEXT_DMA_SEMAPHORE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_CONTEXT_DMA_NOTIFIER, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_NOTIFIER_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_SIZE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_STORAGE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_PARAMS, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_SIZE_IN, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_VALID_POINT_IN, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_VALID_SIZE_IN, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_SIZE_OUT, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_CONTROL_INPUT_SCALER, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_COMPOSITION_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_COMPOSITION_CONSTANT_ALPHA, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_KEY_ALPHA, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_KEY_RED_CR, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_KEY_GREEN_Y, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_KEY_BLUE_CB, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_PRESENT_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_SCAN_DIRECTION, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_TIMESTAMP_ORIGIN_LO, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_TIMESTAMP_ORIGIN_HI, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_UPDATE_TIMESTAMP_LO, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_UPDATE_TIMESTAMP_HI, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_COMPOSITION_FACTOR_SELECT, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_SYNC_POINT_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_SYNC_POINT_ACQUIRE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_STREAM_ID, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_RSB, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_CDE_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_CDE_CTB_ENTRY, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_CDE_ZBC_COLOR, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_INTERLOCK_FLAGS, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_WINDOW_INTERLOCK_FLAGS, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_CONTEXT_DMA_ILUT, scIndex);
            dprintf("-----------------------------------------------------------------------------------------------------\n");
            dprintf("HDR SPECIFIC REGISTERS FOR WINDOW %u                          ASY    |    ARM     | ASY-ARM Mismatch\n", pDisp[indexGpu].dispGetWinId(chanNum));
            dprintf("-----------------------------------------------------------------------------------------------------\n");
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_CSC00CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_CSC0LUT_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_CSC01CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_CONTEXT_DMA_TMO_LUT, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_TMO_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_TMO_LOW_INTENSITY_ZONE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_TMO_LOW_INTENSITY_VALUE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_TMO_MEDIUM_INTENSITY_ZONE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_TMO_MEDIUM_INTENSITY_VALUE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_TMO_HIGH_INTENSITY_ZONE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_TMO_HIGH_INTENSITY_VALUE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_CSC10CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_CSC1LUT_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_CSC11CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_COMPOSITION_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_FMT_COEFFICIENT_C00, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_FMT_COEFFICIENT_C01, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_FMT_COEFFICIENT_C02, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_FMT_COEFFICIENT_C10, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_FMT_COEFFICIENT_C11, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_FMT_COEFFICIENT_C12, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_FMT_COEFFICIENT_C20, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_FMT_COEFFICIENT_C21, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_FMT_COEFFICIENT_C22, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_FMT_COEFFICIENT_C03, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_FMT_COEFFICIENT_C13, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_00(LWC57E_SET_FMT_COEFFICIENT_C23, scIndex);

            for (k = 0; k < LWC57DSET_CONTEXT_DMAS_ISO__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57E_SET_CONTEXT_DMA_ISO,       k, scIndex);
            }

            for (k = 0; k < LWC57E_SET_PLANAR_STORAGE__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57E_SET_PLANAR_STORAGE,        k, scIndex);
            }

            for (k = 0; k < LWC57E_SET_POINT_IN__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57E_SET_POINT_IN,              k, scIndex);
            }

            for (k = 0; k < LWC57E_SET_OPAQUE_POINT_IN__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57E_SET_OPAQUE_POINT_IN,              k, scIndex);
            }

            for (k = 0; k < LWC57E_SET_OPAQUE_SIZE_IN__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57E_SET_OPAQUE_SIZE_IN,              k, scIndex);
            }
            break;

        case LWDISPLAY_CHNTYPE_WINIM: // Window channel - C57B
            dprintf("----------------------------------------------------------------------------------------------\n");
            dprintf("WINDOW IMMEDIATE CHANNEL WINDOW %u                     ASY    |    ARM     | ASY-ARM Mismatch\n", pDisp[indexGpu].dispGetWinId(chanNum));
            dprintf("----------------------------------------------------------------------------------------------\n");
            for (k = 0; k < LWC57B_SET_POINT_OUT__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(LWC57B_SET_POINT_OUT,              k, scIndex);
            }
            break;

        default:
            dprintf("LwDisplay channel %u not supported.\n", chanNum);
    }
}

/*!
 * @brief _disp2sCompToDec - colwert 2's complement number to decimal and return
 *
 *  @param[in]  LwU32               size                            Size of input number in terms of bit count ( including sign bit )
 *  @param[in]  data32              coreHead                        Input 2's complimented number
 */

static
LwS32 _disp2sCompToDec(LwU32 size, LwU32 data32)
{
    LwU32 mask;
    mask = (1<< size)-1;
    
    if( (data32 >> (size - 1)) >= 1 )
    {
        return  (-1 * ((~data32 & mask) + 1));
    }
    else
    {
        return data32 & mask;
    }
}

void dispReadDscStatus_v04_00(LwU32 head)
{
    LwU32 data32;
    LwU32 sor;
    LwU32 subSor;
    LwU32 temp;
    dprintf("--------------------------------------------------\n");
    dprintf("%5s \t | \t %5s \t | \t %5s \t |\n", "Head", "Status", "Mode");
    dprintf("--------------------------------------------------\n");

    if (head == 0xFFFFFFFF)
    {
        for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); head++)
        {
            char enable[50] = "DISABLE";
            char mode[50];

            data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_CONTROL(head));

            if (FLD_TEST_DRF(C57D, _HEAD_SET_DSC_CONTROL, _ENABLE, _TRUE, data32))
            {
                strcpy(enable, "ENABLE");

                if (FLD_TEST_DRF(C57D, _HEAD_SET_DSC_CONTROL, _MODE, _SINGLE, data32))
                {
                    strcpy(mode, "SINGLE");
                }
                else if (FLD_TEST_DRF(C57D, _HEAD_SET_DSC_CONTROL, _MODE, _DUAL, data32))
                {
                    strcpy(mode, "DUAL");
                }
                else if (FLD_TEST_DRF(C57D, _HEAD_SET_DSC_CONTROL, _MODE, _DROP, data32))
                {
                    strcpy(mode, "DROP");
                }
            }
            else
            {
                strcpy(mode, "N/A");
            }

            dprintf("%5d \t | \t %5s \t | \t %5s \t |\n", head, enable, mode);
        }
        dprintf("--------------------------------------------------\n");
    }
    else
    {
        char enable[50] = "DISABLE";
        char mode[50];

        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_CONTROL(head));

        if (FLD_TEST_DRF(C57D, _HEAD_SET_DSC_CONTROL, _ENABLE, _TRUE, data32))
        {
            strcpy(enable, "ENABLE");

            if (FLD_TEST_DRF(C57D, _HEAD_SET_DSC_CONTROL, _MODE, _SINGLE, data32))
            {
                strcpy(mode, "SINGLE");
            }
            else if (FLD_TEST_DRF(C57D, _HEAD_SET_DSC_CONTROL, _MODE, _DUAL, data32))
            {
                strcpy(mode, "DUAL");
            }
            else if (FLD_TEST_DRF(C57D, _HEAD_SET_DSC_CONTROL, _MODE, _DROP, data32))
            {
                strcpy(mode, "DROP");
            }
        }
        else
        {
            strcpy(mode, "N/A");
        }

        dprintf("%5d \t | \t %5s \t | \t %5s \t |\n", head, enable, mode);
        dprintf("--------------------------------------------------\n\n\n");
        
        data32 = GPU_REG_RD32(LW_PDISP_POSTCOMP_HEAD_DSC_STATUS(head));
        dprintf("POSTCOMP DSC STATUS:\n");
        dprintf("HINDEX = %5d\n",DRF_VAL(_PDISP,_POSTCOMP_HEAD_DSC_STATUS,_HINDEX, data32));
        dprintf("VINDEX = %5d\n",DRF_VAL(_PDISP,_POSTCOMP_HEAD_DSC_STATUS,_VINDEX, data32));
        dprintf("BUSY   = %5d\n",DRF_VAL(_PDISP,_POSTCOMP_HEAD_DSC_STATUS,_BUSY  , data32)); 
        
        dprintf("\nPPS - \n");
        
        dprintf("Raw PPS Data : \n");
        dprintf("------------------------------------------------------\n");

        dprintf("              Method                  |     Data       |\n");
        dprintf("------------------------------------------------------\n");

        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA0(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA0 (%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA1(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA1 (%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA2(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA2 (%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA3(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA3 (%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA4(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA4 (%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA5(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA5 (%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA6(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA6 (%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA7(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA7 (%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA8(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA8 (%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA9(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA9 (%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA10(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA10(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA11(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA11(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA12(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA12(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA13(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA13(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA14(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA14(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA15(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA15(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA16(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA16(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA17(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA17(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA18(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA18(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA19(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA19(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA20(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA20(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA21(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA21(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA22(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA22(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA23(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA23(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA24(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA24(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA25(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA25(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA26(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA26(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA27(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA27(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA28(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA28(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA29(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA29(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA30(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA30(%d)     |     0x%8.8x |\n", head, data32);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA31(head));
        dprintf("LWC57D_HEAD_SET_DSC_PPS_DATA31(%d)     |     0x%8.8x |\n", head, data32);

        dprintf("------------------------------------------------------\n");
        
        dprintf("=========================================\n");
        dprintf("|  PPS Set                              |\n");
        dprintf("-----------------------------------------\n");
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA0(head));
        dprintf("Version:                 %5d.%d\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA0, _DSC_VERSION_MAJOR ,data32),
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA0, _DSC_VERSION_MINOR, data32));
        dprintf("pps_identifier           %5d\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA0,_PPS_IDENTIFIER,data32));
        dprintf("bits_per_component       %5d bpc\n" ,
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA0, _BITS_PER_COMPONENT,data32));
        dprintf("linebuf_depth            %5d bits\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA0, _LINEBUF_DEPTH, data32));
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA1(head));
        dprintf("block_pred_enable        %5d\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA1, _BLOCK_PRED_ENABLE  ,data32));
        dprintf("colwert_rgb              %5d\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA1, _COLWERT_RGB  ,data32));
        dprintf("simple_422               %5d\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA1, _SIMPLE422 ,data32));
        dprintf("vbr_enable               %5d\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA1, _VBR_ENABLE ,data32));
        dprintf("bits_per_pixel           %5d\n",
            ((DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA1, _BITS_PER_PIXEL_HIGH  ,data32) << 8) |
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA1, _BITS_PER_PIXEL_LOW  ,data32)));
        dprintf("pic_height               %5d\n",
            ((DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA1, _PIC_HEIGHT_HIGH ,data32) << 8 ) |
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA1, _PIC_HEIGHT_LOW  ,data32))); 
            
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA2(head));
        dprintf("pic_width                %5d\n",
            ((DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA2, _PIC_WIDTH_HIGH ,data32) << 8 ) |
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA2, _PIC_WIDTH_LOW  ,data32))); 
        dprintf("slice_height             %5d\n",
            ((DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA2, _SLICE_HEIGHT_HIGH ,data32) << 8 ) |
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA2, _SLICE_HEIGHT_LOW  ,data32))); 
            
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA3(head));
        dprintf("slice_width              %5d\n",
            ((DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA3, _SLICE_WIDTH_HIGH ,data32) << 8 ) |
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA3, _SLICE_WIDTH_LOW  ,data32))); 
        dprintf("chunk_size               %5d\n",
            ((DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA3, _CHUNK_SIZE_HIGH ,data32) << 8 ) |
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA3, _CHUNK_SIZE_LOW  ,data32))); 
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA4(head));
        dprintf("initial_xmit_delay       %5d\n",
            ((DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA4,_INITIAL_XMIT_DELAY_HIGH ,data32) << 8 ) |
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA4,_INITIAL_XMIT_DELAY_LOW  ,data32))); 
        dprintf("initial_dec_delay        %5d\n",
            ((DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA4,_INITIAL_DEC_DELAY_HIGH ,data32) << 8 ) |
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA4,_INITIAL_DEC_DELAY_LOW  ,data32))); 
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA5(head));
        dprintf("initial_scale_value      %5d\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA5, _INITIAL_SCALE_VALUE  ,data32));
        dprintf("scale_increment_interval %5d\n",
            ((DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA5,_SCALE_INCREMENT_INTERVAL_HIGH  ,data32) << 8 ) |
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA5,_SCALE_INCREMENT_INTERVAL_LOW   ,data32))); 
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA6(head));
        dprintf("scale_decrement_interval %5d\n",
            ((DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA6,_SCALE_DECREMENT_INTERVAL_HIGH ,data32) << 8 ) |
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA6,_SCALE_DECREMENT_INTERVAL_LOW  ,data32))); 
        dprintf("first_line_bpg_ofs       %5d\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA6, _FIRST_LINE_BPG_OFFSET ,data32));
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA7(head));
        dprintf("nfl_bpg_offset           %5d\n",
            ((DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA7,_NFL_BPG_OFFSET_HIGH  ,data32) << 8 ) |
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA7,_NFL_BPG_OFFSET_LOW  ,data32))); 
        dprintf("slice_bpg_offset         %5d\n",
            ((DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA7,_SLICE_BPG_OFFSET_HIGH ,data32) << 8 ) |
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA7,_SLICE_BPG_OFFSET_LOW  ,data32))); 
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA8(head));
        dprintf("initial_offset           %5d\n",
            ((DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA8,_INITIAL_OFFSET_HIGH ,data32) << 8 ) |
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA8,_INITIAL_OFFSET_LOW  ,data32))); 
        dprintf("final_offset             %5d\n",
            ((DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA8,_FINAL_OFFSET_HIGH  ,data32) << 8 ) |
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA8,_FINAL_OFFSET_LOW  ,data32))); 
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA9(head));
        dprintf("flatness_min_qp          %5d\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA9, _FLATNESS_MIN_QP  ,data32));
        dprintf("flatness_max_qp          %5d\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA9, _FLATNESS_MAX_QP  ,data32));
        dprintf("\n");
        
        
        
        dprintf("------------------------------------------\n");
        dprintf("|  RC parameter set                      |\n");
        dprintf("------------------------------------------\n");
        dprintf("rc_model_size            %5d\n",
            ((DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA9,_RC_MODEL_SIZE_HIGH  ,data32) << 8 ) |
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA9,_RC_MODEL_SIZE_LOW  ,data32))); 
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA10(head));
        dprintf("rc_edge_factor           %5d\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA10, _RC_EDGE_FACTOR ,data32));
        dprintf("rc_quant_incr_limit0     %5d\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA10, _RC_QUANT_INCR_LIMIT0 ,data32));
        dprintf("rc_quant_incr_limit1     %5d\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA10, _RC_QUANT_INCR_LIMIT1 ,data32));
        dprintf("rc_tgt_offset_hi         %5d\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA10, _RC_TGT_OFFSET_HI ,data32));
        dprintf("rc_tgt_offset_lo         %5d\n",
            DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA10, _RC_TGT_OFFSET_LO ,data32));
            
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA11(head));
        dprintf("\nRC_BUF_THRESH0     %5d\n", DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA11, _RC_BUF_THRESH0, data32) << 6);
        dprintf("RC_BUF_THRESH1     %5d\n", DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA11, _RC_BUF_THRESH1, data32) << 6);
        dprintf("RC_BUF_THRESH2     %5d\n", DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA11, _RC_BUF_THRESH2, data32) << 6);
        dprintf("RC_BUF_THRESH3     %5d\n", DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA11, _RC_BUF_THRESH3, data32) << 6);
            
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA12(head));
        dprintf("RC_BUF_THRESH4     %5d\n", DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA12, _RC_BUF_THRESH4, data32) << 6);
        dprintf("RC_BUF_THRESH5     %5d\n", DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA12, _RC_BUF_THRESH5, data32) << 6);
        dprintf("RC_BUF_THRESH6     %5d\n", DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA12, _RC_BUF_THRESH6, data32) << 6);
        dprintf("RC_BUF_THRESH7     %5d\n", DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA12, _RC_BUF_THRESH7, data32) << 6);
            
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA13(head));
        dprintf("RC_BUF_THRESH8     %5d\n", DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA13, _RC_BUF_THRESH8, data32) << 6);
        dprintf("RC_BUF_THRESH9     %5d\n", DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA13, _RC_BUF_THRESH9, data32) << 6);
        dprintf("RC_BUF_THRESH10    %5d\n", DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA13, _RC_BUF_THRESH10, data32) << 6);
        dprintf("RC_BUF_THRESH11    %5d\n", DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA13, _RC_BUF_THRESH11, data32) << 6);
            
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA14(head));
        dprintf("RC_BUF_THRESH12    %5d\n", DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA14, _RC_BUF_THRESH12, data32) << 6);
        dprintf("RC_BUF_THRESH13    %5d\n", DRF_VAL(C57D, _HEAD_SET_DSC_PPS_DATA14, _RC_BUF_THRESH13, data32) << 6);
        
        //
        // Max QP has 3 MSB bits in MAX_QP_HIGH0 and 2 LSB bits in MAX_QP_LOW0
        // offset QP is signed 6 bits value
        //
        
        dprintf("\nRC Range Parameter [ 0]\n");
        dprintf("Max QP    [ 0] = %5d\n", 
                (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA14,_RC_RANGE_MAX_QP_HIGH0, data32) << 2) | 
                 DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA14, _RC_RANGE_MAX_QP_LOW0, data32));
        dprintf("Min QP    [ 0] = %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA14, _RC_RANGE_MIN_QP0, data32));
        temp = DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA14, _RC_RANGE_BPG_OFFSET0, data32);
        dprintf("Offset QP [ 0] = %5d\n",_disp2sCompToDec(6,temp));
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA15(head));
        dprintf("\nRC Range Parameter [ 1]\n");
        dprintf("Max QP    [ 1] = %5d\n", 
                (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA15,_RC_RANGE_MAX_QP_HIGH1, data32) << 2) | 
                 DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA15, _RC_RANGE_MAX_QP_LOW1, data32));
        dprintf("Min QP    [ 1] = %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA15, _RC_RANGE_MIN_QP1, data32));
        temp = DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA15, _RC_RANGE_BPG_OFFSET1, data32);
        dprintf("Offset QP [ 1] = %5d\n",_disp2sCompToDec(6,temp));
        
        dprintf("\nRC Range Parameter [ 2]\n");
        dprintf("Max QP    [ 2] = %5d\n", 
                (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA15,_RC_RANGE_MAX_QP_HIGH2, data32) << 2) | 
                 DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA15, _RC_RANGE_MAX_QP_LOW2, data32));
        dprintf("Min QP    [ 2] = %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA15, _RC_RANGE_MIN_QP2, data32));
        temp = DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA15, _RC_RANGE_BPG_OFFSET2, data32);
        dprintf("Offset QP [ 2] = %5d\n",_disp2sCompToDec(6,temp));
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA16(head));
        dprintf("\nRC Range Parameter [ 3]\n");
        dprintf("Max QP    [ 3] = %5d\n", 
                (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA16,_RC_RANGE_MAX_QP_HIGH4, data32) << 2) | 
                 DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA16, _RC_RANGE_MAX_QP_LOW4, data32));
        dprintf("Min QP    [ 3] = %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA16, _RC_RANGE_MIN_QP4, data32));
        temp = DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA16, _RC_RANGE_BPG_OFFSET4, data32);
        dprintf("Offset QP [ 3] = %5d\n",_disp2sCompToDec(6,temp));
        
        dprintf("\nRC Range Parameter [ 4]\n");
        dprintf("Max QP    [ 4] = %5d\n", 
                (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA16,_RC_RANGE_MAX_QP_HIGH4, data32) << 2) | 
                 DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA16, _RC_RANGE_MAX_QP_LOW4, data32));
        dprintf("Min QP    [ 4] = %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA16, _RC_RANGE_MIN_QP4, data32));
        temp = DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA16, _RC_RANGE_BPG_OFFSET4, data32);
        dprintf("Offset QP [ 4] = %5d\n",_disp2sCompToDec(6,temp));
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA17(head));
        dprintf("\nRC Range Parameter [ 5]\n");
        dprintf("Max QP    [ 5] = %5d\n", 
                (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA17,_RC_RANGE_MAX_QP_HIGH5, data32) << 2) | 
                 DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA17, _RC_RANGE_MAX_QP_LOW5, data32));
        dprintf("Min QP    [ 5] = %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA17, _RC_RANGE_MIN_QP5, data32));
        temp = DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA17, _RC_RANGE_BPG_OFFSET5, data32);
        dprintf("Offset QP [ 5] = %5d\n",_disp2sCompToDec(6,temp));
        
        dprintf("\nRC Range Parameter [ 6]\n");
        dprintf("Max QP    [ 6] = %5d\n", 
                (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA17,_RC_RANGE_MAX_QP_HIGH6, data32) << 2) | 
                 DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA17, _RC_RANGE_MAX_QP_LOW6, data32));
        dprintf("Min QP    [ 6] = %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA17, _RC_RANGE_MIN_QP6, data32));
        temp = DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA17, _RC_RANGE_BPG_OFFSET6, data32);
        dprintf("Offset QP [ 6] = %5d\n",_disp2sCompToDec(6,temp));
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA18(head));
        dprintf("\nRC Range Parameter [ 7]\n");
        dprintf("Max QP    [ 7] = %5d\n", 
                (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA18,_RC_RANGE_MAX_QP_HIGH7, data32) << 2) | 
                  DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA18, _RC_RANGE_MAX_QP_LOW7, data32));
        dprintf("Min QP    [ 7] = %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA18, _RC_RANGE_MIN_QP7, data32));
        temp = DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA18, _RC_RANGE_BPG_OFFSET7, data32);
        dprintf("Offset QP [ 7] = %5d\n",_disp2sCompToDec(6,temp));
        
        dprintf("\nRC Range Parameter [ 8]\n");
        dprintf("Max QP    [ 8] = %5d\n", 
                (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA18,_RC_RANGE_MAX_QP_HIGH8, data32) << 2) | 
                 DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA18, _RC_RANGE_MAX_QP_LOW8, data32));
        dprintf("Min QP    [ 8] = %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA18, _RC_RANGE_MIN_QP8, data32));
        temp = DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA18, _RC_RANGE_BPG_OFFSET8, data32);
        dprintf("Offset QP [ 8] = %5d\n",_disp2sCompToDec(6,temp));
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA19(head));
        dprintf("\nRC Range Parameter [ 9]\n");
        dprintf("Max QP    [ 9] = %5d\n", 
                (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA19,_RC_RANGE_MAX_QP_HIGH9, data32) << 2) | 
                 DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA19, _RC_RANGE_MAX_QP_LOW9, data32));
        dprintf("Min QP    [ 9] = %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA19, _RC_RANGE_MIN_QP9, data32));
        temp = DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA19, _RC_RANGE_BPG_OFFSET9, data32);
        dprintf("Offset QP [ 9] = %5d\n",_disp2sCompToDec(6,temp));
        
        dprintf("\nRC Range Parameter [10]\n");
        dprintf("Max QP    [10] = %5d\n", 
                (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA19,_RC_RANGE_MAX_QP_HIGH10, data32) << 2) | 
                 DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA19, _RC_RANGE_MAX_QP_LOW10, data32));
        dprintf("Min QP    [10] = %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA19, _RC_RANGE_MIN_QP10, data32));
        temp = DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA19, _RC_RANGE_BPG_OFFSET10, data32);
        dprintf("Offset QP [10] = %5d\n",_disp2sCompToDec(6,temp));
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA20(head));
        dprintf("\nRC Range Parameter [11]\n");
        dprintf("Max QP    [11] = %5d\n", 
                (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA20,_RC_RANGE_MAX_QP_HIGH11, data32) << 2) | 
                 DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA20, _RC_RANGE_MAX_QP_LOW11, data32));
        dprintf("Min QP    [11] = %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA20, _RC_RANGE_MIN_QP11, data32));
        temp = DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA20, _RC_RANGE_BPG_OFFSET11, data32);
        dprintf("Offset QP [11] = %5d\n",_disp2sCompToDec(6,temp));
        
        dprintf("\nRC Range Parameter [12]\n");
        dprintf("Max QP    [12] = %5d\n", 
                (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA20,_RC_RANGE_MAX_QP_HIGH12, data32) << 2) | 
                 DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA20, _RC_RANGE_MAX_QP_LOW12, data32));
        dprintf("Min QP    [12] = %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA20, _RC_RANGE_MIN_QP12, data32));
        temp = DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA20, _RC_RANGE_BPG_OFFSET12, data32);
        dprintf("Offset QP [12] = %5d\n",_disp2sCompToDec(6,temp));
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA21(head));
        dprintf("\nRC Range Parameter [13]\n");
        dprintf("Max QP    [13] = %5d\n", 
                (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA21,_RC_RANGE_MAX_QP_HIGH13, data32) << 2) | 
                 DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA21, _RC_RANGE_MAX_QP_LOW13, data32));
        dprintf("Min QP    [13] = %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA21, _RC_RANGE_MIN_QP13, data32));
        temp = DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA21, _RC_RANGE_BPG_OFFSET13, data32);
        dprintf("Offset QP [13] = %5d\n",_disp2sCompToDec(6,temp));
        
        dprintf("\nRC Range Parameter [14]\n");
        dprintf("Max QP    [14] = %5d\n", 
                (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA21,_RC_RANGE_MAX_QP_HIGH14, data32) << 2) | 
                 DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA21, _RC_RANGE_MAX_QP_LOW14, data32));
        dprintf("Min QP    [14] = %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA21, _RC_RANGE_MIN_QP14, data32));
        temp = DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA21, _RC_RANGE_BPG_OFFSET14, data32);
        dprintf("Offset QP [14] = %5d\n",_disp2sCompToDec(6,temp));
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA22(head));
        dprintf("\nnative_420              %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA22, _NATIVE420, data32));
        dprintf("native_422              %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA22, _NATIVE422, data32));
        dprintf("second_line_bpg_ofs     %5d\n", DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA22, _SECOND_LINE_BPG_OFFSET, data32));
        
        dprintf("nsl_bpg_offset          %5d\n",
            (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA22, _NSL_BPG_OFFSET_HIGH, data32) << 8 ) |
             DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA22, _NSL_BPG_OFFSETLOW, data32));
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_HEAD_SET_DSC_PPS_DATA23(head));
        dprintf("second_line_ofs_adj     %5d\n",
            (DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA23, _SECOND_LINE_OFFSET_ADJ_HIGH, data32) << 8 )|
             DRF_VAL(C57D,_HEAD_SET_DSC_PPS_DATA23, _SECOND_LINE_OFFSET_ADJ_LOW, data32));
        
        dprintf("------------------------------------------\n");
        dprintf("|  FEC Parameters                         |\n");
        dprintf("------------------------------------------\n");
        
        dprintf("\n---------------------------------------------------------------------------\n");
        dprintf("|SOR | ENABLE |   HDCP  | SCRAMBLER|  SEQ   |INTERLEAVING|  BIT  | ENABLE |\n");
        dprintf("|    |        |NOSKIP_PM| SKIP_PM  |OVERRIDE|   BYPASS   |REVERSE| STATUS |\n");
        dprintf("---------------------------------------------------------------------------\n");
        
        for(sor = 0; sor < pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR); sor++)
        {
            data32 = GPU_REG_RD32( LW_PDISP_FE_CMGR_CLK_SOR(sor));
            for(subSor = 0; subSor < 2; subSor++)
            {
                data32 = GPU_REG_RD32( LW_PDISP_SOR_DP_FEC_CTRL(sor, subSor));
                dprintf("|%d,%d |%8d|%9d|%10d|%8d|%12d|%7d|%8d|\n",
                        sor,
                        subSor,
                        DRF_VAL(_PDISP, _SOR_DP_FEC_CTRL,_ENABLE, data32),
                        DRF_VAL(_PDISP, _SOR_DP_FEC_CTRL,_HDCP_NOSKIP_PM, data32),
                        DRF_VAL(_PDISP, _SOR_DP_FEC_CTRL,_SCRAMBLER_SKIP_PM, data32),
                        DRF_VAL(_PDISP, _SOR_DP_FEC_CTRL,_SEQ_OVERRIDE, data32),
                        DRF_VAL(_PDISP, _SOR_DP_FEC_CTRL,_INTERLEAVING_BYPASS, data32),
                        DRF_VAL(_PDISP, _SOR_DP_FEC_CTRL,_BIT_REVERSE, data32),
                        DRF_VAL(_PDISP, _SOR_DP_FEC_CTRL,_ENABLE_STATUS, data32));
                        
                        
        
            }
        }
        dprintf("---------------------------------------------------------------------------\n");
    }
}

void dispHeadORConnection_v04_00(void)
{
    CHECK_INIT(MODE_LIVE);

    dprintf("===========================================================================================================\n");
    dprintf("OR#     OWNER  DISPLAY ID  PROTOCOL    MODE   STATE  WINDOWS   HSYNC   VSYNC   DATA    PWR     BLANKED?    \n");
    dprintf("-----------------------------------------------------------------------------------------------------------\n");

    pDisp[indexGpu].dispHeadSorConnection();

    dprintf("============================================================================================================\n");
}

void dispHeadSorConnection_v04_00(void)
{

    LwU32       orNum, data32, head, ownerMask, headDisplayId = 0, window;
    LwS32       numSpaces, winSpaces = 0;
    ORPROTOCOL  orProtocol;
    char        *protocolString;
    char        *orString = dispGetORString(LW_OR_SOR);
    BOOL        bAtLeastOneHeadPrinted;
    LwU32       comma;

    protocolString = (char *)malloc(256 * sizeof(char));

    for (orNum = 0; orNum < pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR); orNum++) 
    {
        if (pDisp[indexGpu].dispResourceExists(LW_OR_SOR, orNum) != TRUE)
        {
            continue;
        }
        
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE + LWC57D_SOR_SET_CONTROL(orNum));
        ownerMask = DRF_VAL(C57D, _SOR_SET_CONTROL, _OWNER_MASK, data32);

        if (!ownerMask)
        {
            dprintf("%s%d    NONE   N/A         N/A         ", orString, orNum);
        }
        else
        {
            bAtLeastOneHeadPrinted = FALSE;

            orProtocol = pDisp[indexGpu].dispGetOrProtocol(LW_OR_SOR, DRF_VAL(C57D, _SOR_SET_CONTROL, _PROTOCOL, data32));
            sprintf(protocolString, "%s", dispGetStringForOrProtocol(LW_OR_SOR, orProtocol));

            // Check extra information in case protocol was DP & add that to the data to be printed
            if ((orProtocol == sorProtocol_DpA) || (orProtocol == sorProtocol_DpB))
            {
                // Read DP_LINKCTL data & add appropriate sting to protocol
                pDisp[indexGpu].dispReadDpLinkCtl(orNum,
                                                  ((orProtocol == sorProtocol_DpA) ? 0 : 1),
                                                  protocolString);

            }

            dprintf("%s%d    HEAD", orString, orNum);

            numSpaces = 7;
            // If more that one owner is there, we need to print brackets
            if (ownerMask & (ownerMask  - 1))
            {    
                dprintf("(");
                --numSpaces;
            }
            for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
            {
                if (BIT(head) & ownerMask)
                {
                    if (bAtLeastOneHeadPrinted)
                    {
                        dprintf("|");
                        --numSpaces;
                    }
                    bAtLeastOneHeadPrinted = TRUE;
                    dprintf("%d  ", head);
                    headDisplayId = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE + LWC57D_HEAD_SET_DISPLAY_ID(head, 0));
                    dprintf ("0x%x", headDisplayId);
                    --numSpaces;
                }
            }
            // If more that one owner is there, we need to print brackets
            if (ownerMask & (ownerMask  - 1))
            {    
                dprintf(")");
                --numSpaces;
            }
            while (numSpaces > 0)
            {
                dprintf(" ");
                numSpaces--;
            }
            dprintf(" %-12s", protocolString);
        }

        data32 = GPU_REG_RD32(LW_PDISP_SOR_PWR(orNum));
        if(DRF_VAL(_PDISP, _SOR_PWR, _MODE, data32) == LW_PDISP_SOR_PWR_MODE_SAFE)
        {
            dprintf("SAFE     %-5s", (DRF_VAL(_PDISP, _SOR_PWR, _SAFE_STATE, data32) == LW_PDISP_SOR_PWR_SAFE_STATE_PU)? "PU" : "PD");
        }
        else
        {
            dprintf("NORMAL   %-5s", (DRF_VAL(_PDISP, _SOR_PWR, _NORMAL_STATE, data32) == LW_PDISP_SOR_PWR_NORMAL_STATE_PU)? "PU" : "PD");
        }

        winSpaces = 0;
        comma = 0;
        for (window = 0; window < pDisp[indexGpu].dispGetNumWindows(); ++window)
        {
            data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR_CORE + LWC57D_WINDOW_SET_CONTROL(window));
            data32 = DRF_VAL(C57D, _WINDOW_SET_CONTROL, _OWNER, data32);

            if (BIT(data32) & ownerMask)
            {
                if (comma > 0)
                {
                    dprintf(",");
                    winSpaces++;
                }
                dprintf("%d", window);
                winSpaces++;
                comma++;
                data32 = GPU_REG_RD32(LW_PDISP_FE_SW_PRECOMP_WIN_PIPE_HDR_CAPA(window));
                if (DRF_VAL(_PDISP, _FE_SW_PRECOMP_WIN_PIPE_HDR_CAPA, _TMO_PRESENT, data32) == LW_PDISP_FE_SW_PRECOMP_WIN_PIPE_HDR_CAPA_TMO_PRESENT_TRUE)
                {
                    dprintf("(TMO)");
                    winSpaces = winSpaces + 5;
                }
            }
        }
        numSpaces = 20 - winSpaces;

        while (numSpaces && (numSpaces > 0))
        {
            dprintf (" ");
            numSpaces--;
        }

        for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
        {
            if (head > 0)
            {
                dprintf("/");
            }
            if (BIT(head) & ownerMask)
            {
                data32 = GPU_REG_RD32(LW_PDISP_SF_BLANK(head));
                if (DRF_VAL(_PDISP, _SF_BLANK, _STATUS, data32) == LW_PDISP_SF_BLANK_STATUS_BLANKED)
                {
                    dprintf("YES%s", (DRF_VAL(_PDISP, _SF_BLANK, _OVERRIDE, data32) == LW_PDISP_SF_BLANK_OVERRIDE_TRUE)? " (because of override)" : "");
                }
                else
                {
                    dprintf("NO");
                }
            }
            else
            {
                dprintf("NA"); // If a head is not attached, we say Not Applicable.
            }
        }

        dprintf("\n");
    }
}

/*!
 * @brief dispPrintClkData - Function to print SLI-OR config data,
 * used by DSLI. It prints SLI register values for configuration
 *
 *  @param[in]  LwU32               head            Head Number
 *  @param[in]  DSLI_DATA          *pDsliData       Pointer to DSLI
 *                                                  datastructure
 *  @param[in]  DSLI_PRINT_PARAM   *pDsliPrintData  Pointer to print
 *                                                  Param datastructure
 *  @param[in]  LwU32               verbose         Verbose switch
 */

void dispPrintClkData_v04_00
(
    LwU32               head,
    DSLI_DATA          *pDsliData,
    DSLI_PRINT_PARAM   *pDsliPrintData,
    LwU32               verbose
)
{
    switch(pDsliData[head].DsliVclkRefSwitchFinalSel)
    {
        case LW_PVTRIM_SYS_VCLK_REF_SWITCH_FINALSEL_SLOWCLK:
            switch(pDsliData[head].DsliSlowClk)
            {
                case LW_PVTRIM_SYS_VCLK_REF_SWITCH_SLOWCLK_XTAL_IN:
                    pDsliPrintData[head].refClkForVpll = "XTAL";
                    break;

                case LW_PVTRIM_SYS_VCLK_REF_SWITCH_SLOWCLK_XTAL4X:
                    pDsliPrintData[head].refClkForVpll = "4X-XTAL";
                    break;
            }
            break;

        case LW_PVTRIM_SYS_VCLK_REF_SWITCH_FINALSEL_TESTORJTAGCLK:
            pDsliPrintData[head].refClkForVpll = "Test-Jtag";
            break;

        case LW_PVTRIM_SYS_VCLK_REF_SWITCH_FINALSEL_MISCCLK:
            switch(pDsliData[head].DsliMisCclk)
            {
                case LW_PVTRIM_SYS_VCLK_REF_SWITCH_MISCCLK_PEX_REFCLK_FAST:
                    pDsliPrintData[head].refClkForVpll = "PEX-REF";
                    break;

                case LW_PVTRIM_SYS_VCLK_REF_SWITCH_MISCCLK_EXT_REFCLK:
                    switch(pDsliData[head].DsliClkDriverSrc)
                    {
                        case LW_PVTRIM_SYS_VPLL_MISC_EXT_REF_CONFIG_SRC_EXT_REFCLKA_IB:
                            pDsliPrintData[head].refClkForVpll = "EXT-Ref-Clock-A";
                            break;

                        case LW_PVTRIM_SYS_VPLL_MISC_EXT_REF_CONFIG_SRC_FL_REFCLK_IN:
                            pDsliPrintData[head].refClkForVpll = "FL_REFCLK";
                            break;
                    }
                    break;
            }
            break;

        case LW_PVTRIM_SYS_VCLK_REF_SWITCH_FINALSEL_ONESRCCLK:
            pDsliPrintData[head].refClkForVpll = "ONESRC";
            break;
    }
}

/*!
 * @brief Helper function to return SLI Data.
 *
 *  @param[in]  LwU32      head       Head index in DSLI_DATA structure to fill
 *  @param[in]  DSLI_DATA *pDsliData  Pointer to DSLI data structure
 */
void dispGetSliData_v04_00
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
    pDsliData[head].DsliClkRemVpllExtRef = GPU_REG_RD32(LW_PVTRIM_SYS_VPLL_MISC(head));
    // As of turing, EXT_REF_CONFIG_SRC is 2 bits wide
    pDsliData[head].DsliClkDriverSrc = DRF_VAL(_PVTRIM, _SYS_VPLL_MISC, _EXT_REF_CONFIG_SRC, pDsliData[head].DsliClkRemVpllExtRef);
    pDsliData[head].DsliHeadSetCntrl = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE + LWC57D_HEAD_SET_CONTROL(head));
    pDsliData[head].DsliHeadSetSlaveLockMode = DRF_VAL(C57D, _HEAD_SET_CONTROL, _SLAVE_LOCK_MODE, pDsliData[head].DsliHeadSetCntrl);
    pDsliData[head].DsliHeadSetMasterLockMode = DRF_VAL(C57D, _HEAD_SET_CONTROL, _MASTER_LOCK_MODE, pDsliData[head].DsliHeadSetCntrl);
    pDsliData[head].DsliHeadSetSlaveLockPin = DRF_VAL(C57D, _HEAD_SET_CONTROL, _SLAVE_LOCK_PIN, pDsliData[head].DsliHeadSetCntrl);
    pDsliData[head].DsliHeadSetMasterLockPin = DRF_VAL(C57D, _HEAD_SET_CONTROL, _MASTER_LOCK_PIN, pDsliData[head].DsliHeadSetCntrl);
}

// Enumerate displayport pipe information, including source and sink.
void
dispDisplayPortEnum_v04_00(void)
{
    DPINFO_SF  *sf = 0;
    DPINFO_SOR *sor = 0;
    LwU32      numHead, numSor, numSf, reg, *headDisplayId = 0;
    LwU8        i, j, k;

    numHead = pDisp[indexGpu].dispGetNumHeads();
    if (numHead)
    {
        headDisplayId = (LwU32*)malloc(sizeof(LwU32) * numHead);
        if (headDisplayId == NULL)
        {
            dprintf("Failed to allocate memory for HEADs");
            return;
        }
        memset((void*)headDisplayId, 0, sizeof(LwU32) * numHead);
    }
    else
    {
        dprintf("No Head to enumerate.\n");
    }

    numSor = pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR);
    if (numSor)
    {
        sor = (DPINFO_SOR*)malloc(sizeof(DPINFO_SOR) * numSor);
        if (sor == NULL)
        {
            dprintf("Failed to allocate memory for SOR_CFG");
            if (numHead && headDisplayId)
                free(headDisplayId);
            return;
        }
        memset((void*)sor, 0, sizeof(DPINFO_SOR) * numSor);
    }
    else
    {
        dprintf("No SOR to enumerate.\n");
    }

    numSf = pDisp[indexGpu].dispGetNumSfs();
    if (numSf)
    {
        sf = (DPINFO_SF*)malloc(sizeof(DPINFO_SF) * numSf);
        if (sor == NULL)
        {
            dprintf("Failed to allocate memory for SF_CFG");
            if (numHead && headDisplayId)
                free(headDisplayId);
            if (numSor && sor)
                free(sor);
            return;
        }
        memset((void*)sf, 0, sizeof(DPINFO_SF) * numSf);
    }
    else
    {
        dprintf("No SF to enumerate.\n");
    }

    for (i = 0; i < numHead; i++)
    {
        headDisplayId[i] = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE +
                               LWC57D_HEAD_SET_DISPLAY_ID(i, 0));
    }

    for (i = 0; i < numSf; i++)
    {
        // Exit if no displayId assigned.
        reg = GPU_REG_RD32(LW_PDISP_SF_TEST(i));
        if (!(sf[i].headMask = DRF_VAL(_PDISP, _SF_TEST, _OWNER_MASK, reg)))
            continue;

        reg = GPU_REG_RD32(LW_PDISP_SF_DP_LINKCTL(i));
        sf[i].bDpEnabled  = FLD_TEST_DRF(_PDISP, _SF_DP_LINKCTL, _ENABLE, _YES,
                                         reg);
        sf[i].bMstEnabled = FLD_TEST_DRF(_PDISP, _SF_DP_LINKCTL, _FORMAT_MODE,
                                         _MULTI_STREAM, reg);

        sf[i].bSingleHeadMst = FLD_TEST_DRF(_PDISP, _SF_DP_LINKCTL,
                                            _SINGLE_HEAD_MST, _ENABLE, reg);

        reg = GPU_REG_RD32(LW_PDISP_SF_DP_FLUSH(i));
        sf[i].bFlushEnabled = FLD_TEST_DRF(_PDISP, _SF_DP_FLUSH, _ENABLE, _YES,
                                           reg);

        reg = GPU_REG_RD32(LW_PDISP_SF_DP_STREAM_CTL(i));
        sf[i].timeSlotStart  = DRF_VAL(_PDISP, _SF_DP_STREAM_CTL,
                                       _START_ACTIVE, reg);
        sf[i].timeSlotLength = DRF_VAL(_PDISP, _SF_DP_STREAM_CTL,
                                       _LENGTH_ACTIVE, reg);

        reg = GPU_REG_RD32(LW_PDISP_SF_DP_STREAM_BW(i));
        sf[i].pbn = DRF_VAL(_PDISP, _SF_DP_STREAM_BW, _ALLOCATED, reg);

        reg = GPU_REG_RD32(LW_PDISP_SF_DP_2ND_STREAM_CTL(i));
        sf[i].timeSlotStart2nd  = DRF_VAL(_PDISP, _SF_DP_2ND_STREAM_CTL,
                                       _START_ACTIVE, reg);
        sf[i].timeSlotLength2nd = DRF_VAL(_PDISP, _SF_DP_2ND_STREAM_CTL,
                                       _LENGTH_ACTIVE, reg);

        reg = GPU_REG_RD32(LW_PDISP_SF_DP_2ND_STREAM_BW(i));
        sf[i].pbn2nd = DRF_VAL(_PDISP, _SF_DP_2ND_STREAM_BW, _ALLOCATED, reg);
    }

    for (i = 0; i < numSor; i++)
    {
        // SOR info
        sor[i].bExist = pDisp[indexGpu].dispResourceExists(LW_OR_SOR, i);
        if (!sor[i].bExist)
        {
            continue;
        }

        reg = GPU_REG_RD32(LW_PDISP_SOR_TEST(i));
        sor[i].headMask = DRF_VAL(_PDISP, _SOR_TEST, _OWNER_MASK, reg);

        reg = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE +
                           LWC57D_SOR_SET_CONTROL(i));
        sor[i].protocol = DRF_VAL(C57D, _SOR_SET_CONTROL, _PROTOCOL, reg);

        sor[i].bDpActive[PRIMARY]   = FALSE;
        sor[i].bDpActive[SECONDARY] = FALSE;

        if (!pDisp[indexGpu].dispGetLinkBySor(i, sor[i].link))
        {
            dprintf("failed to get links to SOR%d\n", i);
            break;
        }

        for (j = 0; j < LW_MAX_SUBLINK; j++)
        {
            LwU32 link;

            reg = GPU_REG_RD32(LW_PDISP_SOR_DP_LINKCTL(i, j));
            sor[i].bDpActive[j] = FLD_TEST_DRF(_PDISP, _SOR_DP_LINKCTL,
                                               _ENABLE, _YES, reg);

            link = sor[i].link[j];
            sor[i].auxPort[j] = pDisp[indexGpu].dispGetAuxPortByLink(link);
        }
    }

    for (i = 0; i < numHead; i++)
    {
        if (!headDisplayId[i])
            continue;

        for (j = 0; j < numSf; j++)
        {
            if (sf[j].headMask & (1 << i))
            {
                sf[j].displayId = headDisplayId[i];
                break;
            }
        }
    }

    dprintf("Tx:\n"
            "---------------------------------------------------------------------------------------------------------------------\n"
            "HEAD  DISPLAYID  PROTOCOL         MODE  PBN     TIMESLOT(START:LENGTH)  OneHeadMST  FLUSH  SOR:SUBLINK  LINK  AUXPORT\n"
            "---------------------------------------------------------------------------------------------------------------------\n");
    for (i = 0; i < numHead; i++)
    {
        for (j = 0; j < numSf; j++)
        {
            if (sf[j].headMask != (1 << i))
                continue;

            // Exit if paired head&sf is not DP mode.
            if (!sf[j].bDpEnabled)
                break;

            dprintf("%-6d%-11x", i, headDisplayId[i]);
            for (k = 0; k < numSor; k++)
            {
                if (sor[k].bExist && (sor[k].headMask & (1 << i)))
                {
                    // Head = i, SF index = j, SOR = k
                    if (sf[j].bMstEnabled)
                    {
                        dprintf("%-17s%-6s%-8d%2d:%-21d%-12s%-7s",
                            dispGetStringForOrProtocol(LW_OR_SOR,
                            sor[k].protocol), "MST", sf[j].pbn,
                            sf[j].timeSlotStart, sf[j].timeSlotLength,
                            sf[j].bSingleHeadMst ? "YES" : "NO",
                            sf[j].bFlushEnabled ? "YES" : "NO");
                    }
                    else
                    {
                        dprintf("%-17s%-6s%-8s%-24s%-12s%-7s",
                            dispGetStringForOrProtocol(LW_OR_SOR,
                            sor[k].protocol), "SST", "NA", "NA",
                            sf[j].bSingleHeadMst ? "YES" : "NO",
                            sf[j].bFlushEnabled ? "YES" : "NO");
                    }

                    if (sor[k].bDpActive[PRIMARY])
                    {
                        dprintf("%d:Primary    %-6c%d\n", k,
                            (char)('A' + sor[k].link[PRIMARY]),
                            sor[k].auxPort[PRIMARY]);
                    }
                    else if (sor[k].bDpActive[SECONDARY])
                    {
                        dprintf("%d:Secondary  %-6c%d\n", k,
                            (char)('A' + sor[k].link[SECONDARY]),
                            sor[k].auxPort[SECONDARY]);
                    }

                    // Print single head MST's allocation
                    if (sf[j].bSingleHeadMst)
                    {
                        dprintf("%40s%-8d%2d:%-21d\n","", sf[j].pbn2nd,
                            sf[j].timeSlotStart2nd, sf[j].timeSlotLength2nd);
                    }

                    // Print secondary row if both sub link are on.
                    if (sor[k].bDpActive[PRIMARY] &&
                        sor[k].bDpActive[SECONDARY])
                    {
                        dprintf("%92d:Secondary  %-6c%d\n", k,
                                (char)('A' + sor[k].link[SECONDARY]),
                                sor[k].auxPort[SECONDARY]);
                    }
                    break;
                }
            }
        }
    }

    dprintf("---------------------------------------------------------------------------------------------------------------------\n");

    dispPrintDpRxEnum();

    if (numHead && headDisplayId)
        free(headDisplayId);
 
    if (numSor && sor)
        free(sor);
 
    if (numSf && sf)
        free(sf);
}
