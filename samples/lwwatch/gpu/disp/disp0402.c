/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
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
#include "disp/v04_02/disp0402.h"
#include "t23x/t234/address_map_new.h"
#include "g_disp_private.h"

/**
 * @brief Translate display address in GPU aperture to SOC aperture
 *
 * @return Translated address in SOC aperture if /p gpuAddress is within display range.
 *         Otherwise return /p gpuAddress.
 */
LwU64 dispTranslateGpuRegAddrToSoc_v04_02(LwU64 gpuAddress)
{
    /* Display range check */
    const LwU64 dispBase = DRF_BASE(LW_PDISP);
    const LwU64 dispExtent = DRF_EXTENT(LW_UDISP);
    const LwU64 gpuOffset = gpuAddress - LW_ADDRESS_MAP_GPU_BASE;

    if (gpuAddress < LW_ADDRESS_MAP_GPU_BASE) 
    {
        return gpuAddress;
    }

    if (gpuOffset >= dispBase && gpuOffset <= dispExtent)
    {
        const LwU64 dispOffset = gpuOffset - dispBase;
        return LW_ADDRESS_MAP_DISP_BASE + dispOffset;
    }

    return gpuAddress;
}

/*!
 *  Get Window ID from Channel number
 *
 *
 *  @param[in]   chanNum     channel number
 *
 *  @return   Window ID. Negative when illegal.
 */
LwU32
dispGetWinId_v04_02(LwU32 chanNum)
{
    if (chanNum >= pDisp[indexGpu].dispGetMaxChan())
        return -1;

    if ((dispChanState_v04_02[chanNum].id == LWDISPLAY_CHNTYPE_WIN) ||
        (dispChanState_v04_02[chanNum].id == LWDISPLAY_CHNTYPE_WINIM))
    {
        return dispChanState_v04_02[chanNum].numInstance;
    }
    else
    {
        return -1;
    }
}

LwU32
dispGetMaxChan_v04_02(void)
{
    return sizeof(dispChanState_v04_02)/sizeof(ChanDesc_t_Lwdisplay);
}

//
// Prints channel state.
//
void
dispPrintChanState_v04_02
(
    LwU32 chanNum
)
{
    ChanDesc_t_Lwdisplay *chnst;
    LwU32 chnStatus, state, val;
    LwU32 evtDispatch;

    if (chanNum >= pDisp[indexGpu].dispGetMaxChan())
        return;

    chnst = &dispChanState_v04_02[chanNum];

    chnStatus = val = GPU_REG_RD32(chnst->base);
    state = DCHN_GET_CHNSTATUS_STATE_V04_02(chanNum, chnStatus);
    if (state > chnst->numstate)
    {
        dprintf("invalid state value %x\n", state);
        return;
    }

    dprintf("%2d \t%s\t%2d  ", chanNum, chnst->name, chnst->headNum);
    dprintf("\t%13s",  DCHN_GET_DESC_V04_02(chanNum, state));

    evtDispatch = GPU_REG_RD32(LW_PDISP_FE_EVT_DISPATCH);

    if (chnst->cap & DISP_SPVSR)
    {
        int i, numpend = 0, idx = 0;

        if (FLD_TEST_DRF(_PDISP, _FE_EVT_DISPATCH, _CTRL_DISP, _PENDING, evtDispatch))
        {
            val = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_CTRL_DISP);
            for (i = 0; i < LW_PDISP_FE_EVT_STAT_CTRL_DISP_SUPERVISOR__SIZE_1; i++)
            {
                if (FLD_IDX_TEST_DRF(_PDISP, _FE_EVT_STAT_CTRL_DISP, _SUPERVISOR, i,  _PENDING, val)) {
                   idx = i + 1;
                   numpend += 1;
                }
            }
        }
        if (numpend ==  1)
        {
            dprintf("\t%3s#%d PENDING", "", idx);
        }
        else if (numpend ==  0)
        {
            dprintf("\t%13s","NOT PENDING");
        }
        else
        {
            dprintf("\t %10s  ","ERROR!!!");
        }
    }
    else
    {
        dprintf("\t%9s   ","N/A");
    }

    if (chnst->cap & DISP_EXCPT)
    {
        LwBool pending = LW_FALSE;

        switch (chnst->id) {
        case LWDISPLAY_CHNTYPE_WIN:
            if (FLD_TEST_DRF(_PDISP, _FE_EVT_DISPATCH, _EXC_WIN, _PENDING, evtDispatch))
            {
                val = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_EXC_WIN);
                pending = FLD_IDX_TEST_DRF(_PDISP, _FE_EVT_STAT_EXC_WIN, _CH, chnst->numInstance, _PENDING, val);
            }
            break;
        case LWDISPLAY_CHNTYPE_WINIM:
            if (FLD_TEST_DRF(_PDISP, _FE_EVT_DISPATCH, _EXC_WINIM, _PENDING, evtDispatch))
            {
                val = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_EXC_WINIM);
                pending = FLD_IDX_TEST_DRF(_PDISP, _FE_EVT_STAT_EXC_WINIM, _CH, chnst->numInstance, _PENDING, val);
            }
            break;
        default:
            if (FLD_TEST_DRF(_PDISP, _FE_EVT_DISPATCH, _EXC_OTHER, _PENDING, evtDispatch))
            {
                val = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_EXC_OTHER);
                if (chnst->id == LWDISPLAY_CHNTYPE_CORE)
                {
                    pending = FLD_TEST_DRF(_PDISP, _FE_EVT_STAT_EXC_OTHER, _CORE, _PENDING, val);
                }
                else if (chnst->id == LWDISPLAY_CHNTYPE_LWRS)
                {
                    pending = FLD_IDX_TEST_DRF(_PDISP, _FE_EVT_STAT_EXC_OTHER, _LWRS, chnst->numInstance, _PENDING, val);
                }
                else
                {
                    dprintf("Unexpected channel type!\n");
                }
            }
            break;
        }

        if (pending)
        {
            dprintf("\t%8s", "  PENDING  ");
        }
        else
        {
            dprintf("\t%12s", "TBD");
        }
    }
    else
    {
        dprintf("\t%5s  ","N/A");
    }
    dprintf("\t0x%08x\n", chnStatus);
}

//
// Prints Channel Name
//
void
dispPrintChanName_v04_02
(
    LwU32 chanNum
)
{
    LwU32 numDispChannels = pDisp[indexGpu].dispGetMaxChan();

    if (chanNum >= numDispChannels) {
        dprintf("<channelNumber> should be in the range 0 .. %d\n", numDispChannels - 1 );
        return;
    }

    if (dispChanState_v04_02[chanNum].cap & DISP_STATE)
    {
        if ((dispChanState_v04_02[chanNum].id == LWDISPLAY_CHNTYPE_WIN) ||
            (dispChanState_v04_02[chanNum].id == LWDISPLAY_CHNTYPE_WINIM))
        {
            dprintf("ChannelName : %s, Window : %d\n",
                    dispChanState_v04_02[chanNum].name, dispChanState_v04_02[chanNum].numInstance);
        }
        else
        {
            dprintf("ChannelName : %s, Head : %d\n",
                    dispChanState_v04_02[chanNum].name, dispChanState_v04_02[chanNum].headNum);
        }
    }
}

//
// Returns Channel State Descriptor
//
static int
dispGetChanDesc_v04_02
(
    char           *name,
    LwU32           headNum,
    void          **dchnst
)
{
    LwU32 chanNum = 0;
    LwU32 numDispChannels = pDisp[indexGpu].dispGetMaxChan();
    LwU32 i;

    // Default is core
    if (!name)
    {       
        name = "core";     
        chanNum = 0;
        headNum = 0;
    }
    else if (!strcmp(name, "core"))
    {
        chanNum = 0;
        headNum = 0;
    }
    else
    {        
        for (i = 0; i < numDispChannels; i++)
        {
            if (headNum == -1)
            {
                if (!strcmp(dispChanState_v04_02[i].name, name))
                {
                    chanNum = i;
                    break;
                }
            }

            if (!strcmp(dispChanState_v04_02[i].name, name) && 
                (headNum == dispChanState_v04_02[i].headNum))
            {
                chanNum = i;
                break;
            }
        }

        if (i == numDispChannels)
        {
            return -1 ;
        }
    }

    *dchnst = &dispChanState_v04_02[chanNum];    
    return chanNum;
}

//
// Prints channel number
//
LwS32
dispGetChanNum_v04_02
(
    char   *chanName,
    LwU32   headNum
)
{
    ChanDesc_t_Lwdisplay *chnst;
    LwS32 chanNum = dispGetChanDesc_v04_02(chanName, headNum, (void **)&chnst);

    return chanNum;
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
dispPrintChanMethodState_v04_02
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

    chanId = dispChanState_v04_02[chanNum].id;
    numInstance = dispChanState_v04_02[chanNum].numInstance;
    scIndex = dispChanState_v04_02[chanNum].scIndex;

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
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_PRESENT_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_VGA_CRC_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_SW_SPARE_A, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_SW_SPARE_B, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_SW_SPARE_C, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_SW_SPARE_D, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_DISPLAY_RATE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_CONTROL_OUTPUT_RESOURCE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_PIXEL_CLOCK_FREQUENCY, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_PIXEL_REORDER_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_DESKTOP_COLOR_ALPHA_RED, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_DESKTOP_COLOR_GREEN_BLUE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_LOCK_OFFSET, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_OVERSCAN_COLOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_RASTER_SIZE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_RASTER_SYNC_END, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_RASTER_BLANK_END, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_RASTER_BLANK_START, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_RASTER_VERT_BLANK2, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_LOCK_CHAIN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_CRC_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_CONTEXT_DMA_CRC, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_CONTROL_LWRSOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_PRESENT_CONTROL_LWRSOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_DITHER_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_CONTROL_OUTPUT_SCALER, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_PROCAMP, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_VIEWPORT_POINT_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_VIEWPORT_SIZE_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_VIEWPORT_POINT_OUT_ADJUST, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_VIEWPORT_SIZE_OUT, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_HDMI_CTRL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_PIXEL_REORDER_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_MAX_OUTPUT_SCALE_FACTOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_HEAD_USAGE_BOUNDS, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_HDMI_AUDIO_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_DP_AUDIO_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_VIEWPORT_VALID_SIZE_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_VIEWPORT_VALID_POINT_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_FRAME_PACKED_VACTIVE_COLOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_MIN_FRAME_IDLE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_DSC_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_DSC_PPS_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_RG_MERGE, head, scIndex);
                dprintf("-----------------------------------------------------------------------------------------------------\n");
                dprintf("HDR SPECIFIC REGISTERS FOR HEAD %u                            ASY    |    ARM     | ASY-ARM Mismatch\n", head);
                dprintf("-----------------------------------------------------------------------------------------------------\n");
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_OCSC0CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_CONTEXT_DMA_OLUT, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_OLUT_FP_NORM_SCALE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_OCSC1CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION, head, scIndex);

#ifdef LWC67D_HEAD_SET_STALL_LOCK
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_STALL_LOCK, head, scIndex);
#endif
                for (k = 0; k < LWC67D_HEAD_SET_DISPLAY_ID__SIZE_1; ++k)
                {
                    DISP_PRINT_SC_DOUBLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_DISPLAY_ID, head, k, scIndex);
                }

                for (k = 0; k < LWC67D_HEAD_SET_CONTEXT_DMA_LWRSOR__SIZE_1; ++k)
                {
                   DISP_PRINT_SC_DOUBLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_CONTEXT_DMA_LWRSOR, head, k, scIndex); 
                }

                for (k = 0; k < LWC67D_HEAD_SET_OFFSET_LWRSOR__SIZE_1; ++k)
                {
                    DISP_PRINT_SC_DOUBLE_IDX_COMP_V04_02(LWC67D_HEAD_SET_OFFSET_LWRSOR, head, k, scIndex); 
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

                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_WINDOW_SET_CONTROL, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_WINDOW_SET_MAX_INPUT_SCALE_FACTOR, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_WINDOW_SET_WINDOW_USAGE_BOUNDS, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS, k, scIndex);
                }
            }

            if (printHeadless == TRUE)
            {
                LwU32 numSors = pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR);

                dprintf("-----------------------------------------------------------------------------------------------------\n");
                dprintf("CORE CHANNEL HEADLESS                                        ASY    |    ARM     | ASY-ARM Mismatch\n");
                dprintf("-----------------------------------------------------------------------------------------------------\n");

                DISP_PRINT_SC_NON_IDX_V04_02(LWC67D_SET_CONTROL, scIndex);
                DISP_PRINT_SC_NON_IDX_V04_02(LWC67D_SET_INTERLOCK_FLAGS, scIndex);
                DISP_PRINT_SC_NON_IDX_V04_02(LWC67D_SET_WINDOW_INTERLOCK_FLAGS, scIndex);

                for (k = 0; k < numSors; ++k)
                {
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_SOR_SET_CONTROL,           k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67D_SOR_SET_LWSTOM_REASON,     k, scIndex);
                }
                DISP_PRINT_SC_NON_IDX_V04_02(LWC67D_SET_INTERLOCK_FLAGS, scIndex);
                DISP_PRINT_SC_NON_IDX_V04_02(LWC67D_SET_WINDOW_INTERLOCK_FLAGS, scIndex);
                DISP_PRINT_SC_NON_IDX_V04_02(LWC67D_SET_CONTEXT_DMA_NOTIFIER, scIndex);
                DISP_PRINT_SC_NON_IDX_V04_02(LWC67D_SET_NOTIFIER_CONTROL, scIndex);
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
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_SEMAPHORE_ACQUIRE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_SEMAPHORE_ACQUIRE_HI, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_SEMAPHORE_RELEASE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_SEMAPHORE_RELEASE_HI, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_SEMAPHORE_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_CONTEXT_DMA_SEMAPHORE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_ACQ_SEMAPHORE_VALUE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_ACQ_SEMAPHORE_VALUE_HI, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_CONTEXT_DMA_ACQ_SEMAPHORE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_CONTEXT_DMA_NOTIFIER, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_NOTIFIER_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_SIZE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_STORAGE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_PARAMS, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_SIZE_IN, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_VALID_POINT_IN, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_VALID_SIZE_IN, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_SIZE_OUT, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_CONTROL_INPUT_SCALER, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_COMPOSITION_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_COMPOSITION_CONSTANT_ALPHA, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_KEY_ALPHA, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_KEY_RED_CR, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_KEY_GREEN_Y, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_KEY_BLUE_CB, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_PRESENT_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_SCAN_DIRECTION, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_TIMESTAMP_ORIGIN_LO, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_TIMESTAMP_ORIGIN_HI, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_UPDATE_TIMESTAMP_LO, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_UPDATE_TIMESTAMP_HI, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_COMPOSITION_FACTOR_SELECT, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_SYNC_POINT_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_SYNC_POINT_ACQUIRE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_CDE_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_CDE_CTB_ENTRY, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_CDE_ZBC_COLOR, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_INTERLOCK_FLAGS, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_WINDOW_INTERLOCK_FLAGS, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_CONTEXT_DMA_ILUT, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_CLAMP_RANGE, scIndex);
            dprintf("-----------------------------------------------------------------------------------------------------\n");
            dprintf("HDR SPECIFIC REGISTERS FOR WINDOW %u                          ASY    |    ARM     | ASY-ARM Mismatch\n", pDisp[indexGpu].dispGetWinId(chanNum));
            dprintf("-----------------------------------------------------------------------------------------------------\n");
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_CSC00CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_CSC0LUT_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_CSC01CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_CONTEXT_DMA_TMO_LUT, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_TMO_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_TMO_LOW_INTENSITY_ZONE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_TMO_LOW_INTENSITY_VALUE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_TMO_MEDIUM_INTENSITY_ZONE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_TMO_MEDIUM_INTENSITY_VALUE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_TMO_HIGH_INTENSITY_ZONE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_TMO_HIGH_INTENSITY_VALUE, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_CSC10CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_CSC1LUT_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_CSC11CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_COMPOSITION_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_FMT_COEFFICIENT_C00, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_FMT_COEFFICIENT_C01, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_FMT_COEFFICIENT_C02, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_FMT_COEFFICIENT_C10, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_FMT_COEFFICIENT_C11, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_FMT_COEFFICIENT_C12, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_FMT_COEFFICIENT_C20, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_FMT_COEFFICIENT_C21, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_FMT_COEFFICIENT_C22, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_FMT_COEFFICIENT_C03, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_FMT_COEFFICIENT_C13, scIndex);
            DISP_PRINT_SC_NON_IDX_V04_02(LWC67E_SET_FMT_COEFFICIENT_C23, scIndex);

            for (k = 0; k < LWC67DSET_CONTEXT_DMAS_ISO__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67E_SET_CONTEXT_DMA_ISO,       k, scIndex);
            }

            for (k = 0; k < LWC67E_SET_PLANAR_STORAGE__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67E_SET_PLANAR_STORAGE,        k, scIndex);
            }

            for (k = 0; k < LWC67E_SET_POINT_IN__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67E_SET_POINT_IN,              k, scIndex);
            }

            for (k = 0; k < LWC67E_SET_OPAQUE_POINT_IN__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67E_SET_OPAQUE_POINT_IN,              k, scIndex);
            }

            for (k = 0; k < LWC67E_SET_OPAQUE_SIZE_IN__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67E_SET_OPAQUE_SIZE_IN,              k, scIndex);
            }
            break;

        case LWDISPLAY_CHNTYPE_WINIM: // Window channel - C67B
            dprintf("----------------------------------------------------------------------------------------------\n");
            dprintf("WINDOW IMMEDIATE CHANNEL WINDOW %u                     ASY    |    ARM     | ASY-ARM Mismatch\n", pDisp[indexGpu].dispGetWinId(chanNum));
            dprintf("----------------------------------------------------------------------------------------------\n");
            for (k = 0; k < LWC67B_SET_POINT_OUT__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V04_02(LWC67B_SET_POINT_OUT,              k, scIndex);
            }
            break;

        default:
            dprintf("LwDisplay channel %u not supported.\n", chanNum);
    }
}