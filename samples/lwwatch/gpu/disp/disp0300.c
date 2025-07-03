/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2021 by LWPU Corporation.  All rights reserved.  All
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
#include "regex/regex.h"
#include "inc/chip.h"
#include "inc/disp.h"
#include "clk.h"
#include "fb.h"
#include "disp/v03_00/disp0300.h"
#include "class/clc37b.h"
#include "class/clc37d.h"
#include "class/clc37e.h"
#include "class_mthd/mthd_type.h"
#include "class_mthd/clc37d_mthd.h"
#include "class_mthd/clc37e_mthd.h"
#include "class_mthd/clc37b_mthd.h"
#include "methodParse.h"
#include "print.h"
#include "volta/gv100/dev_trim.h"
#include "g_disp_private.h"
#include "dpaux.h"

#define LWDISPLAY_DISP_NUM_DMA_CHANNELS LW_PDISP_FE_DEBUG_CTL__SIZE_1
#define DISP_PUSH_BUFFER_SIZE           4096
#define PHYSICAL_ADDR                   47
#define WIN_PUSHBUFFER_OFFSET           1
#define WINIM_PUSHBUFFER_OFFSET         33

#define GET_BITS(num,h,l) (((num) >> l) & (0xffffffff >> (31 - h + l)))

//
// Global variables.
//
mthds_t *mthd[LW_PDISP_FE_DEBUG_CTL__SIZE_1];
int mthd_num[LW_PDISP_FE_DEBUG_CTL__SIZE_1];
static int mthdInitialized = 0;
LwU32 classHeaderNumLwdisplay[LWDISPLAY_CHNTYPE_WINIM + 1];

static void printDispPbParsed_v03_00(PhysAddr baseAddr, LwU32 OffsetDwords, LwU32* buffer, LwU32 numDwords, LwU32 classNum, LwU32 getoffset, LwU32 putoffset);
static void getPbData_v03_00(PhysAddr physOffset, LwU32 numDwords, char * buffer, LwU32 mem_Target);
void initializeClassHeaderNum_v03_00(char *classNames, LwU32 classHeaderNum[]);
LwU32 print_sc_dinj(unsigned baseAddr, unsigned addr, unsigned chanNum, char *name);
static void printExceptionReason(int reason);

static AUXPORT link2AuxPort[PADLINK_MAX] =
{
    AUXPORT_0,
    AUXPORT_1,
    AUXPORT_2,
    AUXPORT_3,
    AUXPORT_4,
    AUXPORT_5,
    AUXPORT_6
};

/*!
 * @brief Returns aux port by specified link.
 *
 * @param[in]  index        specified Link.
 *
 * @returns  aux port
 */
LwU32 dispGetAuxPortByLink_v03_00(LwU32 index)
{
    if (index < PADLINK_MAX)
        return link2AuxPort[index];
    else
        return AUXPORT_NONE;
}

/*!
 * @brief dispPrintClkData - Function to print SLI-OR config data,
 * used by DSLI. It prints SLI register values for configuration
 *
 *  @param[in]  LwU32               head            Head Number
 *  @param[in]  DSLI_DATA           *pDsliData      Pointer to DSLI
 *                                                  datastructure
 *  @param[in]  DSLI_PRINT_PARAM    *pDsliPrintData  Pointer to print
 *                                                  Param datastructure
 *  @param[in]  LwU32               verbose         Verbose switch
 */

void dispPrintClkData_v03_00
(
    LwU32               head,
    DSLI_DATA           *pDsliData,
    DSLI_PRINT_PARAM    *pDsliPrintData,
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

                case LW_PVTRIM_SYS_VCLK_REF_SWITCH_SLOWCLK_FL_REFCLK_IN:
                    pDsliPrintData[head].refClkForVpll = "FL_REFCLK";
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

                        case LW_PVTRIM_SYS_VPLL_MISC_EXT_REF_CONFIG_SRC_EXT_REFCLKB_IB:
                            pDsliPrintData[head].refClkForVpll = "EXT-Ref-Clock-B";
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

//
// Prints Channel Name
//
void
dispPrintChanName_v03_00
(
    LwU32 chanNum
)
{
    LwU32 numDispChannels = pDisp[indexGpu].dispGetMaxChan();

    if (chanNum >= numDispChannels) {
        dprintf("<channelNumber> should be in the range 0 .. %d\n", numDispChannels - 1 );
        return;
    }

    if (dispChanState_v03_00[chanNum].cap & DISP_STATE)
    {
        if ((dispChanState_v03_00[chanNum].id == LWDISPLAY_CHNTYPE_WIN) ||
            (dispChanState_v03_00[chanNum].id == LWDISPLAY_CHNTYPE_WINIM))
        {
            dprintf("ChannelName : %s, Window : %d\n",
                    dispChanState_v03_00[chanNum].name, dispChanState_v03_00[chanNum].numInstance);
        }
        else
        {
            dprintf("ChannelName : %s, Head : %d\n",
                    dispChanState_v03_00[chanNum].name, dispChanState_v03_00[chanNum].headNum);
        }
    }
}

/*!
 * @brief dispGetNumHeads - Returns number of heads supported by the hardware
 */
LwU32
dispGetNumHeads_v03_00(void)
{
    LwU32 miscConfigA = GPU_REG_RD32(LW_PDISP_FE_MISC_CONFIGA);

    return DRF_VAL(_PDISP, _FE_MISC_CONFIGA, _NUM_HEADS, miscConfigA);
}

/*!
 * @brief dispGetNumWindows - Returns number of windows supported by the hardwape
 */
LwU32
dispGetNumWindows_v03_00(void)
{
    LwU32 miscConfigA = GPU_REG_RD32(LW_PDISP_FE_MISC_CONFIGA);

    return DRF_VAL(_PDISP, _FE_MISC_CONFIGA, _NUM_WINDOWS, miscConfigA);
}

/*!
 * @brief dispGetNumOrs - Returns number of the given type of OR supported by the hardwape
 */
LwU32
dispGetNumOrs_v03_00(LWOR orType)
{
    LwU32 miscConfigA = GPU_REG_RD32(LW_PDISP_FE_MISC_CONFIGA);

    switch (orType)
    {
    case LW_OR_SOR:
        return DRF_VAL(_PDISP, _FE_MISC_CONFIGA, _NUM_SORS, miscConfigA);
    case LW_OR_PIOR:
        return DRF_VAL(_PDISP, _FE_MISC_CONFIGA, _NUM_PIORS, miscConfigA);
    case LW_OR_DAC:
        /* No DAC support in lwdisplay */
        return 0;
    default:
        dprintf("%s: Unexpected OR type", __FUNCTION__);
        return 0;
    }
}


//
// Returns Channel State Descriptor
//
int
dispGetChanDesc_v03_00
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
                if (!strcmp(dispChanState_v03_00[i].name, name))
                {
                    chanNum = i;
                    break;
                }
            }

            if (!strcmp(dispChanState_v03_00[i].name, name) && 
                (headNum == dispChanState_v03_00[i].headNum))
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

    *dchnst = &dispChanState_v03_00[chanNum];    
    return chanNum;
}

LwS32
dispGetChanDescriptor_v03_00(LwU32 chanNum, void **desc)
{
    LwU32 numDispChannels = pDisp[indexGpu].dispGetMaxChan();

    if (chanNum >= numDispChannels)
    {
        dprintf("chanNum should be less than %d\n", numDispChannels);
        return -1;
    }
    if (desc)
    {
        *desc = (void *)&dispChanState_v03_00[chanNum];
    }
    else
    {
        dprintf("ERROR: null pointer to descriptor\n");
        return -1;
    }
    return 0;
}

//
// Prints channel number
//
LwS32
dispGetChanNum_v03_00
(
    char   *chanName,
    LwU32   headNum
)
{
    ChanDesc_t_Lwdisplay *chnst;
    LwS32 chanNum;

    if ((chanNum = dispGetChanDesc_v03_00(chanName, headNum, (void **)&chnst)) == -1)
    {
        return -1;
    }
    else
    {
        return chanNum;
    }
    return -1;
}

/*!
 *  Get Channel type and Head num for the given channel num
 *
 *
 *  @param[in]   chanNum     channel number
 *  @param[out]  pHeadNum    head num for that channel
 *
 *  @return   channel num. negative when illegal.
 */
LwS32
dispGetChanType_v03_00(LwU32 chanNum, LwU32* pHeadNum)
{
    if (chanNum >= pDisp[indexGpu].dispGetMaxChan())
        return -1;

    if (pHeadNum)
    {
        *pHeadNum = dispChanState_v03_00[chanNum].headNum;
    }

    return dispChanState_v03_00[chanNum].id;
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
dispGetWinId_v03_00(LwU32 chanNum)
{
    if (chanNum >= pDisp[indexGpu].dispGetMaxChan())
        return -1;

    if ((dispChanState_v03_00[chanNum].id == LWDISPLAY_CHNTYPE_WIN) ||
        (dispChanState_v03_00[chanNum].id == LWDISPLAY_CHNTYPE_WINIM))
    {
        return dispChanState_v03_00[chanNum].numInstance;
    }
    else
    {
        return -1;
    }
}


LwU32
dispGetMaxChan_v03_00(void)
{
    return sizeof(dispChanState_v03_00)/sizeof(ChanDesc_t_Lwdisplay);
}

//
// Prints channel state.
//
void
dispPrintChanState_v03_00
(
    LwU32 chanNum
)
{
    ChanDesc_t_Lwdisplay *chnst;
    LwU32 chnStatus, state, val;
    LwU32 evtDispatch;

    if (chanNum >= pDisp[indexGpu].dispGetMaxChan())
        return;

    chnst = &dispChanState_v03_00[chanNum];

    chnStatus = val = GPU_REG_RD32(chnst->base);
    state = DCHN_GET_CHNSTATUS_STATE_V03_00(chanNum, chnStatus);
    if (state > chnst->numstate)
    {
        dprintf("invalid state value %x\n", state);
        return;
    }

    dprintf("%2d \t%s\t%2d  ", chanNum, chnst->name, chnst->headNum);
    dprintf("\t%13s",  DCHN_GET_DESC_V03_00(chanNum, state));

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

void
dispPrintScanoutOwner_v03_00(void)
{
    LwU32 scanoutOwnerArmed, scanoutOwnerActiv;
    LwU32 data32;
    LwU32 i;

    for (i = 0; i < pDisp[indexGpu].dispGetNumHeads(); ++i)
    {
        data32 = GPU_REG_RD32(LW_PDISP_FE_CORE_HEAD_STATE(i));
        scanoutOwnerArmed = DRF_VAL(_PDISP, _FE_CORE_HEAD_STATE, _SHOWING_ARMED, data32);
        scanoutOwnerActiv = DRF_VAL(_PDISP, _FE_CORE_HEAD_STATE, _SHOWING_ACTIVE, data32);

        if (scanoutOwnerActiv == LW_PDISP_FE_CORE_HEAD_STATE_SHOWING_ACTIVE_DRIVER)
        {
            dprintf("Scanout owner for head%d (ACTIV) = DRIVER\n", i);
        }
        else
        {
            dprintf("Scanout owner for head%d (ACTIV) = VBIOS\n", i);
        }


        if (scanoutOwnerArmed == LW_PDISP_FE_CORE_HEAD_STATE_SHOWING_ARMED_DRIVER)
        {
            dprintf("Scanout owner for head%d (ARMED) = DRIVER\n", i);
        }
        else
        {
            dprintf("Scanout owner for head%d (ARMED) =  VBIOS\n", i);
        }
    }
}

void dispDumpChannelState_v03_00(char *chName, LwS32 headNum, LwS32 winNum, BOOL printHeadless, BOOL printRegsWithoutEquivMethod)
{
    LwU32 minChan = 0, maxChan = 0, channel;

    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        if ( chName != NULL )
        {
            LwS32 chNum = -1;
            if (!strcmp(chName, "win") || !strcmp(chName, "winim"))
            {
                if ((winNum == -1) && (headNum != -1))
                {
                    dprintf("Bad argument for window channel dumping\n");
                    dprintf("lw: Usage: !lw.dchnmstate win/winim [-w<wId>] \n");
                    return;
                }
                chNum = pDisp[indexGpu].dispGetChanNum(chName, winNum);
            }
            else if (!strcmp(chName, "core"))
            {
                chNum = pDisp[indexGpu].dispGetChanNum(chName, 0);
            } else {
                dprintf("Bad channel name '%s'\n", chName);
                return;
            }

            if (chNum != -1)
            {
                minChan = chNum;
                maxChan = chNum + 1;
            }
        }

        if (minChan == maxChan)
        {
            maxChan = pDisp[indexGpu].dispGetMaxChan();
        }

        for (channel = minChan; channel < maxChan; channel++)
        {
            pDisp[indexGpu].dispPrintChanMethodState(channel, printHeadless,
                                                     printRegsWithoutEquivMethod,
                                                     headNum, winNum);
        }
    }
    MGPU_LOOP_END;
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
dispPrintChanMethodState_v03_00
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

    chanId = dispChanState_v03_00[chanNum].id;
    numInstance = dispChanState_v03_00[chanNum].numInstance;
    scIndex = dispChanState_v03_00[chanNum].scIndex;

#ifndef LWC37D_HEAD_SET_DISPLAY_ID__SIZE_1
#define LWC37D_HEAD_SET_DISPLAY_ID__SIZE_1                          2
#endif

#ifndef LWC37D_SET_CONTEXT_DMAS_ISO__SIZE_1
#define LWC37DSET_CONTEXT_DMAS_ISO__SIZE_1                          6
#endif

#ifndef LWC37E_SET_PLANAR_STORAGE__SIZE_1
#define LWC37E_SET_PLANAR_STORAGE__SIZE_1                           3
#endif

#ifndef LWC37E_SET_POINT_IN__SIZE_1
#define LWC37E_SET_POINT_IN__SIZE_1                                 2
#endif

#ifndef LWC37E_SET_OPAQUE_POINT_IN__SIZE_1
#define LWC37E_SET_OPAQUE_POINT_IN__SIZE_1                          4
#endif

#ifndef LWC37E_SET_OPAQUE_SIZE_IN__SIZE_1
#define LWC37E_SET_OPAQUE_SIZE_IN__SIZE_1                           4
#endif

#ifndef LWC37B_SET_POINT_OUT__SIZE_1
#define LWC37B_SET_POINT_OUT__SIZE_1                                2
#endif

#ifndef LWC37D_HEAD_SET_CONTEXT_DMA_LWRSOR__SIZE_1
#define LWC37D_HEAD_SET_CONTEXT_DMA_LWRSOR__SIZE_1                  2
#endif

#ifndef LWC37D_HEAD_SET_OFFSET_LWRSOR__SIZE_1
#define LWC37D_HEAD_SET_OFFSET_LWRSOR__SIZE_1                       2
#endif

    switch(chanId)
    {
        case LWDISPLAY_CHNTYPE_CORE: // Core channel - C37D
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
                // The following list is based off of //sw/dev/gpu_drv/chips_a/sdk/lwpu/inc/class/clC37D.h#13
                // Note that it's implicit that the above comment applies only to core channel (d in C37D implies core)
                //                
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_PRESENT_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_VGA_CRC_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_SW_SPARE_A, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_SW_SPARE_B, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_SW_SPARE_C, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_SW_SPARE_D, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_DISPLAY_RATE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_CONTROL_OUTPUT_RESOURCE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_PIXEL_CLOCK_FREQUENCY, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_PIXEL_REORDER_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_DESKTOP_COLOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_LOCK_OFFSET, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_OVERSCAN_COLOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_RASTER_SIZE, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_RASTER_SYNC_END, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_RASTER_BLANK_END, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_RASTER_BLANK_START, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_RASTER_VERT_BLANK2, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_LOCK_CHAIN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_CRC_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_CONTEXT_DMA_CRC, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_CONTROL_LWRSOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_PRESENT_CONTROL_LWRSOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_DITHER_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_CONTROL_OUTPUT_SCALER, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_PROCAMP, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_VIEWPORT_POINT_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_VIEWPORT_SIZE_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_VIEWPORT_POINT_OUT_ADJUST, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_VIEWPORT_SIZE_OUT, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_HDMI_CTRL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_PIXEL_REORDER_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_PIXEL_CLOCK_FREQUENCY_MAX, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_MAX_OUTPUT_SCALE_FACTOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_HEAD_USAGE_BOUNDS, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_HDMI_AUDIO_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_DP_AUDIO_CONTROL, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_VIEWPORT_VALID_SIZE_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_VIEWPORT_VALID_POINT_IN, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_FRAME_PACKED_VACTIVE_COLOR, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_CONTROL_LWRSOR_COMPOSITION, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_CONTROL_OUTPUT_LUT, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_OFFSET_OUTPUT_LUT, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_CONTEXT_DMA_OUTPUT_LUT, head, scIndex);
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_MIN_FRAME_IDLE, head, scIndex);
#ifdef LWC37D_HEAD_SET_STALL_LOCK
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_STALL_LOCK, head, scIndex);
#endif
                for (k = 0; k < LWC37D_HEAD_SET_DISPLAY_ID__SIZE_1; ++k)
                {
                    DISP_PRINT_SC_DOUBLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_DISPLAY_ID, head, k, scIndex);
                }

                for (k = 0; k < LWC37D_HEAD_SET_CONTEXT_DMA_LWRSOR__SIZE_1; ++k)
                {
                   DISP_PRINT_SC_DOUBLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_CONTEXT_DMA_LWRSOR, head, k, scIndex);
                }

                for (k = 0; k < LWC37D_HEAD_SET_OFFSET_LWRSOR__SIZE_1; ++k)
                {
                    DISP_PRINT_SC_DOUBLE_IDX_COMP_V03_00(LWC37D_HEAD_SET_OFFSET_LWRSOR, head, k, scIndex);
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

                    DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_WINDOW_SET_CONTROL, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_WINDOW_SET_MAX_INPUT_SCALE_FACTOR, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_WINDOW_SET_WINDOW_USAGE_BOUNDS, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_WINDOW_SET_WINDOW_FORMAT_USAGE_BOUNDS, k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_WINDOW_SET_WINDOW_ROTATED_FORMAT_USAGE_BOUNDS, k, scIndex);
                }
            }

            if (printHeadless == TRUE)
            {
                LwU32 numSors = pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR);
                LwU32 numPiors = pDisp[indexGpu].dispGetNumOrs(LW_OR_PIOR);

                dprintf("-----------------------------------------------------------------------------------------------------\n");
                dprintf("CORE CHANNEL HEADLESS                                        ASY    |    ARM     | ASY-ARM Mismatch\n");
                dprintf("-----------------------------------------------------------------------------------------------------\n");

                DISP_PRINT_SC_NON_IDX_V03_00(LWC37D_SET_CONTROL, scIndex);
                DISP_PRINT_SC_NON_IDX_V03_00(LWC37D_SET_INTERLOCK_FLAGS, scIndex);
                DISP_PRINT_SC_NON_IDX_V03_00(LWC37D_SET_WINDOW_INTERLOCK_FLAGS, scIndex);

                for (k = 0; k < numSors; ++k)
                {
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_SOR_SET_CONTROL,           k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_SOR_SET_LWSTOM_REASON,     k, scIndex);
                }
                for (k = 0; k < numPiors; ++k)
                {
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_PIOR_SET_CONTROL,          k, scIndex);
                    DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37D_PIOR_SET_LWSTOM_REASON,    k, scIndex);
                }
                DISP_PRINT_SC_NON_IDX_V03_00(LWC37D_SET_INTERLOCK_FLAGS, scIndex);
                DISP_PRINT_SC_NON_IDX_V03_00(LWC37D_SET_WINDOW_INTERLOCK_FLAGS, scIndex);
                DISP_PRINT_SC_NON_IDX_V03_00(LWC37D_SET_CONTEXT_DMA_NOTIFIER, scIndex);
                DISP_PRINT_SC_NON_IDX_V03_00(LWC37D_SET_NOTIFIER_CONTROL, scIndex);
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

        case LWDISPLAY_CHNTYPE_WIN: // Window channel - C37E
            dprintf("-----------------------------------------------------------------------------------------------------\n");
            dprintf("WINDOW CHANNEL WINDOW %u                                      ASY    |    ARM     | ASY-ARM Mismatch\n", pDisp[indexGpu].dispGetWinId(chanNum));
            dprintf("-----------------------------------------------------------------------------------------------------\n");
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_SEMAPHORE_ACQUIRE, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_SEMAPHORE_RELEASE, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_SEMAPHORE_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CONTEXT_DMA_SEMAPHORE, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CONTEXT_DMA_NOTIFIER, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_NOTIFIER_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_SIZE, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_STORAGE, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_PROCESSING, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_PARAMS, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_COLWERSION_RED, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_COLWERSION_GREEN, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_COLWERSION_BLUE, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_SIZE_IN, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_VALID_POINT_IN, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_VALID_SIZE_IN, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_SIZE_OUT, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CONTROL_INPUT_SCALER, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CONTROL_INPUT_LUT, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CONTEXT_DMA_INPUT_LUT, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_COMPOSITION_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_COMPOSITION_CONSTANT_ALPHA, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_KEY_ALPHA, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_KEY_RED_CR, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_KEY_GREEN_Y, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_KEY_BLUE_CB, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_PRESENT_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_DIST_RENDER_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_DIST_RENDER_CONFIG, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_DIST_RENDER_STRIP, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_DIST_RENDER_INHIBIT_FLIP_REGION, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_SCAN_DIRECTION, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_TIMESTAMP_ORIGIN_LO, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_TIMESTAMP_ORIGIN_HI, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_UPDATE_TIMESTAMP_LO, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_UPDATE_TIMESTAMP_HI, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_COMPOSITION_FACTOR_SELECT, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_SYNC_POINT_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_SYNC_POINT_ACQUIRE, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_STREAM_ID, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_RSB, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_CDE_CONTROL, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_CDE_CTB_ENTRY, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_CDE_ZBC_COLOR, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_INTERLOCK_FLAGS, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_WINDOW_INTERLOCK_FLAGS, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CSC_RED2RED, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CSC_GREEN2RED, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CSC_BLUE2RED, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CSC_CONSTANT2RED, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CSC_RED2GREEN, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CSC_GREEN2GREEN, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CSC_BLUE2GREEN, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CSC_CONSTANT2GREEN, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CSC_RED2BLUE, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CSC_GREEN2BLUE, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CSC_BLUE2BLUE, scIndex);
            DISP_PRINT_SC_NON_IDX_V03_00(LWC37E_SET_CSC_CONSTANT2BLUE, scIndex);

            for (k = 0; k < LWC37DSET_CONTEXT_DMAS_ISO__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37E_SET_CONTEXT_DMA_ISO,       k, scIndex);
            }

            for (k = 0; k < LWC37E_SET_PLANAR_STORAGE__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37E_SET_PLANAR_STORAGE,        k, scIndex);
            }

            for (k = 0; k < LWC37E_SET_POINT_IN__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37E_SET_POINT_IN,              k, scIndex);
            }

            for (k = 0; k < LWC37E_SET_OPAQUE_POINT_IN__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37E_SET_OPAQUE_POINT_IN,              k, scIndex);
            }

            for (k = 0; k < LWC37E_SET_OPAQUE_SIZE_IN__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37E_SET_OPAQUE_SIZE_IN,              k, scIndex);
            }
            break;

        case LWDISPLAY_CHNTYPE_WINIM: // Window channel - C37B
            dprintf("-----------------------------------------------------------------------------------------------------\n");
            dprintf("WINDOW IMMEDIATE CHANNEL WINDOW %u                            ASY    |    ARM     | ASY-ARM Mismatch\n", pDisp[indexGpu].dispGetWinId(chanNum));
            dprintf("-----------------------------------------------------------------------------------------------------\n");
            for (k = 0; k < LWC37B_SET_POINT_OUT__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_COMP_V03_00(LWC37B_SET_POINT_OUT,              k, scIndex);
            }
            break;

        default:
            dprintf("LwDipslay channel %u not supported.\n", chanNum);
    }
}

void dispHeadSorConnection_v03_00(void)
{
    LwU32       orNum, data32, head, ownerMask, headDisplayId = 0;
    LwS32       numSpaces;
    ORPROTOCOL  orProtocol;
    char        *protocolString;
    char        *orString = dispGetORString(LW_OR_SOR);
    BOOL        bAtLeastOneHeadPrinted;

    protocolString = (char *)malloc(256 * sizeof(char));

    for (orNum = 0; orNum < pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR); orNum++) 
    {
        if (pDisp[indexGpu].dispResourceExists(LW_OR_SOR, orNum) != TRUE)
        {
            continue;
        }

        //dispPrintOwnerProtocol_LW50(LW_OR_SOR, orNum);
        data32 = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE + LWC37D_SOR_SET_CONTROL(orNum));
        ownerMask = DRF_VAL(C37D, _SOR_SET_CONTROL, _OWNER_MASK, data32);
        if (!ownerMask)
        {
            dprintf("%s%d    NONE   N/A         N/A         ", orString, orNum);
        }
        else
        {
            bAtLeastOneHeadPrinted = FALSE;

            orProtocol = pDisp[indexGpu].dispGetOrProtocol(LW_OR_SOR, DRF_VAL(C37D, _SOR_SET_CONTROL, _PROTOCOL, data32));
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
                    headDisplayId = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE + LWC37D_HEAD_SET_DISPLAY_ID(head, 0));
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
            dprintf("SAFE     %-17s", (DRF_VAL(_PDISP, _SOR_PWR, _SAFE_STATE, data32) == LW_PDISP_SOR_PWR_SAFE_STATE_PU)? "PU" : "PD");
        }
        else
        {
            dprintf("NORMAL   %-17s", (DRF_VAL(_PDISP, _SOR_PWR, _NORMAL_STATE, data32) == LW_PDISP_SOR_PWR_NORMAL_STATE_PU)? "PU" : "PD");
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

BOOL
dispResourceExists_v03_00(LWOR orType, LwU32 index)
{
    LwU32 orHwCap = 0;
	LwU32 orCap    = GPU_REG_RD32(LW_PDISP_FE_SW_SYS_CAP);

    if (index >= pDisp[indexGpu].dispGetNumOrs(orType))
    {
        dprintf("lw: %s Illegal OR Index: %d\n", __FUNCTION__, index);
        return FALSE;
    }

    switch (orType)
    {

        case LW_OR_SOR:
            orHwCap = GPU_REG_RD32(LW_PDISP_SOR_CAP(index));
            return (DRF_IDX_VAL(_PDISP, _FE_SW_SYS_CAP, _SOR_EXISTS, index, orCap) ==
                     LW_PDISP_FE_SW_SYS_CAP_SOR_EXISTS_YES);
        default:
            dprintf(" %s Invalid OR type : %d ", __FUNCTION__, orType);
            return FALSE;
    }
}



ORPROTOCOL dispGetOrProtocol_v03_00(LWOR orType, LwU32 protocolValue)
{
    switch (orType)
    {
        case LW_OR_SOR:
        {
            switch (protocolValue)
            {
                case LWC37D_SOR_SET_CONTROL_PROTOCOL_SINGLE_TMDS_A:
                    return sorProtocol_SingleTmdsA;
                case LWC37D_SOR_SET_CONTROL_PROTOCOL_SINGLE_TMDS_B:
                    return sorProtocol_SingleTmdsB;
                case LWC37D_SOR_SET_CONTROL_PROTOCOL_DUAL_TMDS:
                    return sorProtocol_DualTmds;
                case LWC37D_SOR_SET_CONTROL_PROTOCOL_DP_A:
                    return sorProtocol_DpA;
                case LWC37D_SOR_SET_CONTROL_PROTOCOL_DP_B:
                    return sorProtocol_DpB;
                case LWC37D_SOR_SET_CONTROL_PROTOCOL_LWSTOM:
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

void dispHeadORConnection_v03_00(void)
{
    CHECK_INIT(MODE_LIVE);

    dprintf("================================================================================================\n");
    dprintf("OR#     OWNER  DISPLAY ID  PROTOCOL    MODE     STATE   HSYNC   VSYNC   DATA    PWR     BLANKED?\n");
    dprintf("------------------------------------------------------------------------------------------------\n");

    pDisp[indexGpu].dispHeadSorConnection();

    dprintf("================================================================================================\n");
}

/**
 * @brief Read PixelClk settings.
 *
 * @returns void
 */
void
dispReadPixelClkSettings_v03_00(void)
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
        regVal = GPU_REG_RD32(LW_PDISP_FE_CMGR_CLK_RG(idx));

        // Check if the RG is enabled
        if (FLD_TEST_DRF(_PDISP, _FE_CMGR_CLK_RG, _STATE, _DISABLE, regVal))
        {
            dprintf("lw:   RG%d[ RG_STATE: DISABLE ]\n", idx);
            dprintf("lw: RG%d_PCLK = N/A\n\n", idx);
            continue;
        }

        // Check which RG mode is selected
        rgMode = DRF_VAL(_PDISP, _FE_CMGR_CLK_RG, _MODE, regVal);
        if (rgMode == LW_PDISP_FE_CMGR_CLK_RG_MODE_SAFE)
        {
            dprintf("lw:   RG%d[ RG_MODE: SAFE ]\n", idx);
            dprintf("lw: RG%d_PCLK = %4d MHz\n\n",
                    idx, pClk[indexGpu].clkReadCrystalFreqKHz() / 1000);
        }
        else if (rgMode == LW_PDISP_FE_CMGR_CLK_RG_MODE_NORMAL)
        {
            // Read the VPLL settings
            VPLL = pClk[indexGpu].clkGetVClkFreqKHz(idx) / 1000;
            dprintf("lw:   VPLL%d = %4d MHz\n", idx, VPLL);

            // Read the RG_DIV settings
            switch (DRF_VAL(_PDISP, _FE_CMGR_CLK_RG, _DIV, regVal))
            {
                case LW_PDISP_FE_CMGR_CLK_RG_DIV_BY_1:
                    rgDiv = 1;
                    dprintf("lw:   RG%d[ RG_DIV: BY_1 ]\n", idx);
                    break;

                case LW_PDISP_FE_CMGR_CLK_RG_DIV_BY_2:
                    rgDiv = 2;
                    dprintf("lw:   RG%d[ RG_DIV: BY_2 ]\n", idx);
                    break;

                case LW_PDISP_FE_CMGR_CLK_RG_DIV_BY_3:
                    rgDiv = 3;
                    dprintf("lw:   RG%d[ RG_DIV: BY_3 ]\n", idx);
                    break;

                case LW_PDISP_FE_CMGR_CLK_RG_DIV_BY_4:
                    rgDiv = 4;
                    dprintf("lw:   RG%d[ RG_DIV: BY_4 ]\n", idx);
                    break;

                case LW_PDISP_FE_CMGR_CLK_RG_DIV_BY_6:
                    rgDiv = 6;
                    dprintf("lw:   RG%d[ RG_DIV: BY_6 ]\n", idx);
                    break;

                case LW_PDISP_FE_CMGR_CLK_RG_DIV_BY_8:
                    rgDiv = 8;
                    dprintf("lw:   RG%d[ RG_DIV: BY_8 ]\n", idx);
                    break;

                case LW_PDISP_FE_CMGR_CLK_RG_DIV_BY_16:
                    rgDiv = 16;
                    dprintf("lw:   RG%d[ RG_DIV: BY_16 ]\n", idx);
                    break;

                default:
                    rgDiv = DRF_VAL(_PDISP, _FE_CMGR_CLK_RG, _DIV, regVal) + 1;
                    dprintf("lw:   RG%d[ RG_DIV: invalid enum (%d) ]\n",
                            idx, DRF_VAL(_PDISP, _FE_CMGR_CLK_RG, _DIV, regVal));
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
dispReadSorClkSettings_v03_00(void)
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
        regVal = GPU_REG_RD32(LW_PDISP_FE_CMGR_CLK_SOR(idx));

        // Check if the SOR is enabled
        if (FLD_TEST_DRF(_PDISP, _FE_CMGR_CLK_SOR, _STATE, _DISABLE, regVal))
        {
            dprintf("lw:   SOR%d[ SOR_STATE: DISABLE ]\n", idx);
            dprintf("lw: SOR%d_CLK = N/A\n\n", idx);
            continue;
        }

        // Check which SOR mode is selected
        sorMode = DRF_VAL(_PDISP, _FE_CMGR_CLK_SOR, _MODE, regVal);
        if (sorMode == LW_PDISP_FE_CMGR_CLK_SOR_MODE_SAFE)
        {
            dprintf("lw:   SOR%d[ SOR_MODE: SAFE ]\n", idx);
            dprintf("lw: SOR%d_CLK = %4d MHz\n\n",
                    idx, pClk[indexGpu].clkReadCrystalFreqKHz() / 1000);
        }
        else if (sorMode == LW_PDISP_FE_CMGR_CLK_SOR_MODE_NORMAL)
        {
            headNum = DRF_VAL(_PDISP, _FE_CMGR_CLK_SOR, _HEAD, regVal);
            if (headNum == LW_PDISP_FE_CMGR_CLK_SOR_HEAD_NONE)
            {
                dprintf("lw:   SOR%d[ SOR_HEAD: NONE ]\n", idx);
                continue;
            }
            if (headNum > LW_PDISP_FE_CMGR_CLK_SOR_HEAD_3)
            {
                dprintf("lw:   SOR%d[ SOR_HEAD: invalid enum (%d) ]\n", idx, headNum);
                continue;
            }

            sorLinkSpeed = DRF_VAL(_PDISP, _FE_CMGR_CLK_SOR, _LINK_SPEED, regVal);
            sorModeBypass = DRF_VAL(_PDISP, _FE_CMGR_CLK_SOR, _MODE_BYPASS, regVal);

            if (sorModeBypass == LW_PDISP_FE_CMGR_CLK_SOR_MODE_BYPASS_NONE)
            {
                VPLL = pClk[indexGpu].clkGetVClkFreqKHz(headNum) / 1000;
                dprintf("lw:   VPLL%d[ %4d MHz ]\n", headNum, VPLL);
                dprintf("lw:   SOR%d[ SOR_MODE_BYPASS: NONE ]\n", idx);

                // sorClk = (VPLL freq / SOR_DIV) * LINK_SPEED / 10
                switch (DRF_VAL(_PDISP, _FE_CMGR_CLK_SOR, _DIV, regVal))
                {
                    case LW_PDISP_FE_CMGR_CLK_SOR_DIV_BY_1:
                        sorDiv = 1;
                        dprintf("lw:   SOR%d[ SOR_DIV: BY_1 ]\n", idx);
                        break;

                    case LW_PDISP_FE_CMGR_CLK_SOR_DIV_BY_2:
                        sorDiv = 2;
                        dprintf("lw:   SOR%d[ SOR_DIV: BY_2 ]\n", idx);
                        break;

                    case LW_PDISP_FE_CMGR_CLK_SOR_DIV_BY_4:
                        sorDiv = 4;
                        dprintf("lw:   SOR%d[ SOR_DIV: BY_4 ]\n", idx);
                        break;

                    case LW_PDISP_FE_CMGR_CLK_SOR_DIV_BY_8:
                        sorDiv = 8;
                        dprintf("lw:   SOR%d[ SOR_DIV: BY_8 ]\n", idx);
                        break;

                    case LW_PDISP_FE_CMGR_CLK_SOR_DIV_BY_16:
                        sorDiv = 16;
                        dprintf("lw:   SOR%d[ SOR_DIV: BY_16 ]\n", idx);
                        break;

                    default:
                        sorDiv = DRF_VAL(_PDISP, _FE_CMGR_CLK_SOR, _DIV, regVal) + 1;
                        dprintf("lw:   SOR%d[ SOR_DIV: invalid enum (%d) ]\n",
                                idx, DRF_VAL(_PDISP, _FE_CMGR_CLK_SOR, _DIV, regVal));
                        break;
                }
                sorClk = (VPLL / sorDiv) * (sorLinkSpeed / 10);
                dprintf("lw: SOR%d_CLK = %4d MHz\n\n", idx, sorClk);
            }
            else if (sorModeBypass == LW_PDISP_FE_CMGR_CLK_SOR_MODE_BYPASS_FEEDBACK)
            {
                VPLL = pClk[indexGpu].clkGetVClkFreqKHz(headNum) / 1000;
                dprintf("lw:   VPLL%d[ %4d MHz ]\n", headNum, VPLL);
                dprintf("lw:   SOR%d[ SOR_MODE_BYPASS: FEEDBACK ]\n", idx);

                // sorClk = (VPLL freq / SOR_PLL_REF_DIV) * LINK_SPEED / 10
                switch (DRF_VAL(_PDISP, _FE_CMGR_CLK_SOR, _PLL_REF_DIV, regVal))
                {
                    case LW_PDISP_FE_CMGR_CLK_SOR_PLL_REF_DIV_BY_1:
                        sorPllRefDiv = 1;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: BY_1 ]\n", idx);
                        break;

                    case LW_PDISP_FE_CMGR_CLK_SOR_PLL_REF_DIV_BY_2:
                        sorPllRefDiv = 2;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: BY_2 ]\n", idx);
                        break;

                    case LW_PDISP_FE_CMGR_CLK_SOR_PLL_REF_DIV_BY_4:
                        sorPllRefDiv = 4;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: BY_4 ]\n", idx);
                        break;

                    case LW_PDISP_FE_CMGR_CLK_SOR_PLL_REF_DIV_BY_8:
                        sorPllRefDiv = 8;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: BY_8 ]\n", idx);
                        break;

                    case LW_PDISP_FE_CMGR_CLK_SOR_PLL_REF_DIV_BY_16:
                        sorPllRefDiv = 16;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: BY_16 ]\n", idx);
                        break;

                    default:
                        sorPllRefDiv = DRF_VAL(_PDISP, _FE_CMGR_CLK_SOR, _PLL_REF_DIV, regVal) + 1;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: invalid enum (%d) ]\n",
                                idx, DRF_VAL(_PDISP, _FE_CMGR_CLK_SOR, _PLL_REF_DIV, regVal));
                        break;
                }
                sorClk = (VPLL / sorPllRefDiv) * (sorLinkSpeed / 10);
                dprintf("lw: SOR%d_CLK = %4d MHz\n\n", idx, sorClk);
            }
            else if (sorModeBypass == LW_PDISP_FE_CMGR_CLK_SOR_MODE_BYPASS_DP_NORMAL)
            {
                dprintf("lw:   SOR%d[ SOR_MODE_BYPASS: DP_NORMAL ]\n", idx);

                // sorClk uses DP pad macro feedback clock
                switch (sorLinkSpeed)
                {
                    case LW_PDISP_FE_CMGR_CLK_SOR_LINK_SPEED_DP_1_62GHZ:
                        dprintf("lw: SOR%d_CLK = 162 MHz\n\n", idx);
                        break;

                    case LW_PDISP_FE_CMGR_CLK_SOR_LINK_SPEED_DP_2_70GHZ:
                        dprintf("lw: SOR%d_CLK = 270 MHz\n\n", idx);
                        break;

                    case LW_PDISP_FE_CMGR_CLK_SOR_LINK_SPEED_DP_5_40GHZ:
                        dprintf("lw: SOR%d_CLK = 540 MHz\n\n", idx);
                        break;

                    case LW_PDISP_FE_CMGR_CLK_SOR_LINK_SPEED_DP_8_10GHZ:
                        dprintf("lw: SOR%d_CLK = 810 MHz\n\n", idx);
                        break;

                    default:
                        dprintf("lw:   SOR%d[ SOR_LINK_SPEED: invalid enum (%d) ]\n",
                                idx, sorLinkSpeed);
                        dprintf("lw: SOR%d_CLK = N/A\n\n", idx);
                        break;
                }
            }
            else if (sorModeBypass == LW_PDISP_FE_CMGR_CLK_SOR_MODE_BYPASS_DP_SAFE)
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

typedef struct
{
    LwU32 exists;
    LwU32 active;
    struct sc
    {
        LwU32 rasterWidth;
        LwU32 rasterHeight;
        LwU32 rasterSyncEndX;
        LwU32 rasterSyncEndY;
        LwU32 rasterBlankEndX;
        LwU32 rasterBlankEndY;
        LwU32 rasterBlankStartX;
        LwU32 rasterBlankStartY;
        LwU32 rasterVertBlank2Ystart;
        LwU32 rasterVertBlank2Yend;
        LwU32 pclkHz;
        LwU32 buffWidth;
        LwU32 buffHeight;
        struct vp
        {
            LwU32 vpHeight;
            LwU32 vpWidth;
            LwU32 vpPosX;
            LwU32 vpPosY;
        }vp[2];
        LwU32 hactive;
        LwU32 hfporch;
        LwU32 hbporch;
        LwU32 hsync;
        LwU32 vactive;
        LwU32 vfporch;
        LwU32 vbporch;
        LwU32 vsync;
    }armd, assy;
}dTimings;

#define GETARM_V03_00(r,f,idx,chan) DRF_VAL(C37D, _HEAD, r##f, GPU_REG_RD32(                       \
                             LW_UDISP_FE_CHN_ARMED_BASEADR_CORE+LWC37D_HEAD##r(idx)))
#define GETASY_V03_00(r,f,idx,chan) DRF_VAL(C37D, _HEAD, r##f, GPU_REG_RD32(                       \
                             LW_UDISP_FE_CHN_ASSY_BASEADR_CORE+LWC37D_HEAD##r(idx)))
#define PRINTREC(name, var)  dprintf("%-22s |",name);                                     \
                             for(head=0;head<numHead;head++)                     \
                               {if(!pMyTimings[head].exists)continue;                    \
                                dprintf("%10d, %-10d|",pMyTimings[head].armd.var,        \
                                pMyTimings[head].assy.var);}                             \
                             dprintf("\n");

void dispTimings_v03_00(void)
{
    LwU32 head=0, numHead=0;
    dTimings *pMyTimings;

    numHead = pDisp[indexGpu].dispGetNumHeads();
    pMyTimings = (dTimings*)malloc(sizeof(dTimings) * numHead);

    if (pMyTimings == NULL)
    {
        dprintf("lw: %s - malloc failed!\n", __FUNCTION__);
        return;
    }

    dprintf("--------------------------------------------------------------------------------------------------------\n");
    dprintf("                                        Display Timings                                                 \n");
    dprintf("--------------------------------------------------------------------------------------------------------\n");

    dprintf("%-22s |","Name (Armed , Assy)");
    for (head = 0; head < numHead; ++head)    
    {
        if(!(DRF_VAL(_PDISP, _FE_SW_SYS_CAP, _HEAD_EXISTS(head), GPU_REG_RD32(LW_PDISP_FE_SW_SYS_CAP)) ==
               LW_PDISP_FE_SW_SYS_CAP_HEAD_EXISTS_YES))
        {
               pMyTimings[head].exists = 0;
               continue;
        }
        dprintf("%s%d","HEAD",head);
        pMyTimings[head].exists = 1;
        pMyTimings[head].active = DRF_VAL(_PDISP, _FE_CORE_HEAD_STATE, _OPERATING_MODE,
                                              GPU_REG_RD32(LW_PDISP_FE_CORE_HEAD_STATE(head)));
        if(pMyTimings[head].active)
            dprintf("%-19s"," (Active)       | ");
        else
            dprintf("%-19s"," (Not Active)   | ");

        //Raster size
        pMyTimings[head].armd.rasterWidth = GETARM_V03_00(_SET_RASTER_SIZE, _WIDTH, head, 0);
        pMyTimings[head].armd.rasterHeight = GETARM_V03_00(_SET_RASTER_SIZE, _HEIGHT, head, 0);
        pMyTimings[head].assy.rasterWidth = GETASY_V03_00(_SET_RASTER_SIZE, _WIDTH, head, 0);
        pMyTimings[head].assy.rasterHeight = GETASY_V03_00(_SET_RASTER_SIZE, _HEIGHT, head, 0);

        //Raster Sync End
        pMyTimings[head].armd.rasterSyncEndX = GETARM_V03_00(_SET_RASTER_SYNC_END, _X, head, 0);
        pMyTimings[head].armd.rasterSyncEndY = GETARM_V03_00(_SET_RASTER_SYNC_END, _Y, head, 0);
        pMyTimings[head].assy.rasterSyncEndX = GETASY_V03_00(_SET_RASTER_SYNC_END, _X, head, 0);
        pMyTimings[head].assy.rasterSyncEndY = GETASY_V03_00(_SET_RASTER_SYNC_END, _Y, head, 0);

        //Raster Blank End
        pMyTimings[head].armd.rasterBlankEndX = GETARM_V03_00(_SET_RASTER_BLANK_END, _X, head, 0);
        pMyTimings[head].armd.rasterBlankEndY = GETARM_V03_00(_SET_RASTER_BLANK_END, _Y, head, 0);
        pMyTimings[head].assy.rasterBlankEndX = GETASY_V03_00(_SET_RASTER_BLANK_END, _X, head, 0);
        pMyTimings[head].assy.rasterBlankEndY = GETASY_V03_00(_SET_RASTER_BLANK_END, _Y, head, 0);
                
        //Raster Blank Start 
        pMyTimings[head].armd.rasterBlankStartX = GETARM_V03_00(_SET_RASTER_BLANK_START, _X, head, 0);
        pMyTimings[head].armd.rasterBlankStartY = GETARM_V03_00(_SET_RASTER_BLANK_START, _Y, head, 0);
        pMyTimings[head].assy.rasterBlankStartX = GETASY_V03_00(_SET_RASTER_BLANK_START, _X, head, 0);
        pMyTimings[head].assy.rasterBlankStartY = GETASY_V03_00(_SET_RASTER_BLANK_START, _Y, head, 0);

        //Raster Vert Blank 
        pMyTimings[head].armd.rasterVertBlank2Ystart=GETARM_V03_00(_SET_RASTER_VERT_BLANK2,_YSTART, head, 0);
        pMyTimings[head].armd.rasterVertBlank2Yend = GETARM_V03_00(_SET_RASTER_VERT_BLANK2, _YEND, head, 0);
        pMyTimings[head].assy.rasterVertBlank2Ystart=GETASY_V03_00(_SET_RASTER_VERT_BLANK2,_YSTART, head, 0);
        pMyTimings[head].assy.rasterVertBlank2Yend = GETASY_V03_00(_SET_RASTER_VERT_BLANK2, _YEND, head, 0);

        //Pclk
        pMyTimings[head].armd.pclkHz = GETARM_V03_00(_SET_PIXEL_CLOCK_FREQUENCY, _HERTZ, head, 0); 
        pMyTimings[head].assy.pclkHz = GETASY_V03_00(_SET_PIXEL_CLOCK_FREQUENCY, _HERTZ, head, 0); 
 
        //Vp[0]=vpin, vp[1]=vpout
      
        pMyTimings[head].armd.vp[0].vpHeight = GETARM_V03_00(_SET_VIEWPORT_SIZE_IN, _HEIGHT, head, 0);
        pMyTimings[head].armd.vp[0].vpWidth = GETARM_V03_00(_SET_VIEWPORT_SIZE_IN, _WIDTH, head, 0);
        pMyTimings[head].armd.vp[0].vpPosX = GETARM_V03_00(_SET_VIEWPORT_POINT_IN, _X, head, 0);
        pMyTimings[head].armd.vp[0].vpPosY = GETARM_V03_00(_SET_VIEWPORT_POINT_IN, _Y, head, 0);

        pMyTimings[head].assy.vp[0].vpHeight = GETASY_V03_00(_SET_VIEWPORT_SIZE_IN, _HEIGHT, head, 0);
        pMyTimings[head].assy.vp[0].vpWidth = GETASY_V03_00(_SET_VIEWPORT_SIZE_IN, _WIDTH, head, 0);
        pMyTimings[head].assy.vp[0].vpPosX = GETASY_V03_00(_SET_VIEWPORT_POINT_IN, _X, head, 0);
        pMyTimings[head].assy.vp[0].vpPosY = GETASY_V03_00(_SET_VIEWPORT_POINT_IN, _Y, head, 0);

        pMyTimings[head].armd.vp[1].vpHeight = GETARM_V03_00(_SET_VIEWPORT_SIZE_OUT, _HEIGHT, head, 0);
        pMyTimings[head].armd.vp[1].vpWidth = GETARM_V03_00(_SET_VIEWPORT_SIZE_OUT, _WIDTH, head, 0);
        pMyTimings[head].armd.vp[1].vpPosX = GETARM_V03_00(_SET_VIEWPORT_POINT_OUT_ADJUST, _X, head, 0);
        pMyTimings[head].armd.vp[1].vpPosY = GETARM_V03_00(_SET_VIEWPORT_POINT_OUT_ADJUST, _Y, head, 0);

        pMyTimings[head].assy.vp[1].vpHeight = GETASY_V03_00(_SET_VIEWPORT_SIZE_OUT, _HEIGHT, head, 0);
        pMyTimings[head].assy.vp[1].vpWidth = GETASY_V03_00(_SET_VIEWPORT_SIZE_OUT, _WIDTH, head, 0);
        pMyTimings[head].assy.vp[1].vpPosX = GETASY_V03_00(_SET_VIEWPORT_POINT_OUT_ADJUST, _X, head, 0);
        pMyTimings[head].assy.vp[1].vpPosY = GETASY_V03_00(_SET_VIEWPORT_POINT_OUT_ADJUST, _Y, head, 0);
        
        pMyTimings[head].armd.hfporch = pMyTimings[head].armd.rasterWidth - 1 - pMyTimings[head].armd.rasterBlankStartX;
        pMyTimings[head].armd.hbporch = pMyTimings[head].armd.rasterBlankEndX - pMyTimings[head].armd.rasterSyncEndX; 
        pMyTimings[head].armd.hactive = pMyTimings[head].armd.rasterBlankStartX - pMyTimings[head].armd.rasterBlankEndX;
        pMyTimings[head].armd.hsync   = pMyTimings[head].armd.rasterSyncEndX+1;
        pMyTimings[head].assy.hfporch = pMyTimings[head].assy.rasterWidth - 1 -pMyTimings[head].assy.rasterBlankStartX;
        pMyTimings[head].assy.hbporch = pMyTimings[head].assy.rasterBlankEndX - pMyTimings[head].assy.rasterSyncEndX; 
        pMyTimings[head].assy.hactive = pMyTimings[head].assy.rasterBlankStartX - pMyTimings[head].assy.rasterBlankEndX;
        pMyTimings[head].assy.hsync   = pMyTimings[head].assy.rasterSyncEndX+1;

        pMyTimings[head].armd.vfporch = pMyTimings[head].armd.rasterHeight -1 - pMyTimings[head].armd.rasterBlankStartY;
        pMyTimings[head].armd.vbporch = pMyTimings[head].armd.rasterBlankEndY - pMyTimings[head].armd.rasterSyncEndY; 
        pMyTimings[head].armd.vactive = pMyTimings[head].armd.rasterBlankStartY - pMyTimings[head].armd.rasterBlankEndY;
        pMyTimings[head].armd.vsync   = pMyTimings[head].armd.rasterSyncEndY+1;
        pMyTimings[head].assy.vfporch = pMyTimings[head].assy.rasterHeight - 1 -pMyTimings[head].assy.rasterBlankStartY;
        pMyTimings[head].assy.vbporch = pMyTimings[head].assy.rasterBlankEndY - pMyTimings[head].assy.rasterSyncEndY; 
        pMyTimings[head].assy.vactive = pMyTimings[head].assy.rasterBlankStartY - pMyTimings[head].assy.rasterBlankEndY;
        pMyTimings[head].assy.vsync   = pMyTimings[head].assy.rasterSyncEndY+1;
    }
    dprintf("\n");
    dprintf("--------------------------------------------------------------------------------------------------------\n");
    PRINTREC("Hvisible",hactive);
    PRINTREC("HFront Porch",hfporch);
    PRINTREC("HBack Porch",hbporch);
    PRINTREC("HSync",hsync);
    PRINTREC("Vvisible",vactive);
    PRINTREC("VFront Porch",vfporch);
    PRINTREC("VBack Porch",vbporch);
    PRINTREC("VSync",vsync);
    PRINTREC("Pixel Clock (Hz)",pclkHz);
    dprintf("--------------------------------------------------------------------------------------------------------\n");
    PRINTREC("VPIN Height",vp[0].vpHeight);
    PRINTREC("VPIN Width",vp[0].vpWidth);
    PRINTREC("VPIN PosX",vp[0].vpPosX);
    PRINTREC("VPIN PosY",vp[0].vpPosY);
    dprintf("--------------------------------------------------------------------------------------------------------\n");
    PRINTREC("VPOUT Height",vp[1].vpHeight);
    PRINTREC("VPOUT Width",vp[1].vpWidth);
    PRINTREC("VPOUT PosX",vp[1].vpPosX);
    PRINTREC("VPOUT PosY",vp[1].vpPosY);
    dprintf("--------------------------------------------------------------------------------------------------------\n");
    dprintf("\n");

    free(pMyTimings);
}

// This matches the layout of the LW_DMA definitions in
// dev_display_withoffset.ref.
typedef struct
{
    LwU32  misc;
    LwU32  baseLo;
    LwU32  baseHi;
    LwU32  limitLo;
    LwU32  limitHi;
} DMAOBJECT_v03;

void dispCtxDmaDescription_v03_00(LwU32 handle, LwS32 chanNum, BOOL searchAllHandles)
{
    LwU32 feInstMem0, feInstMem1;
    LwU32 i, clientID, instance, objMemOffset;
    HASH_TABLE_ENTRY htEntry[(LW_UDISP_HASH_LIMIT - LW_UDISP_HASH_BASE + 1)/sizeof(HASH_TABLE_ENTRY)]; 
    DMAOBJECT_v03 dmaObj;
    PhysAddr feInstMemAddr, dispHTAddr, objMemStart, tmpAddr;
    BOOL isPhysLwm = FALSE;
    BOOL searchAllChannels = (chanNum  == -1) ? TRUE : FALSE;
    BOOL handleFound = FALSE;
    LW_STATUS status;

    // Read the base address of the display instance memory
    feInstMem0 = GPU_REG_RD32(LW_PDISP_FE_INST_MEM0);

    // Check the status bit to know if the loaded address is valid.
    if (FLD_TEST_DRF(_PDISP, _FE_INST_MEM0, _STATUS, _ILWALID, feInstMem0))
    {
        dprintf("Handle couldn't be found in the Hash Table because instance memory in invalid.\n");
        return;
    }

    feInstMem1 = GPU_REG_RD32(LW_PDISP_FE_INST_MEM1);

    // obtain the starting address of display instance memory.
    feInstMemAddr = DRF_VAL(_PDISP, _FE_INST_MEM1, _ADDR, feInstMem1);
    feInstMemAddr <<= 16;

    // obtain the base address of the display hash table
    dispHTAddr = feInstMemAddr + LW_UDISP_HASH_BASE;

    if (FLD_TEST_DRF(_PDISP, _FE_INST_MEM0, _TARGET, _PHYS_LWM, feInstMem0))
    {
        isPhysLwm = TRUE;
    }

    if (isPhysLwm)
        status = pFb[indexGpu].fbRead(dispHTAddr, &htEntry, sizeof(htEntry));
    else
        status = readSystem(dispHTAddr, &htEntry, sizeof(htEntry));

    if (status != LW_OK)
    {
        dprintf("Failed to read hash table memory.\n");
        return;
    }

    for (i = 0 ; i < ((LW_UDISP_HASH_LIMIT - LW_UDISP_HASH_BASE + 1)/sizeof(HASH_TABLE_ENTRY)) ; i++)
    {
        instance = DRF_VAL(_UDISP, _HASH_TBL, _INSTANCE, htEntry[i].data);
        if (instance == LW_UDISP_HASH_TBL_INSTANCE_ILWALID)
        {
            continue;
        }

        if (searchAllChannels || (chanNum == (LwS32)(DRF_VAL(_UDISP, _HASH_TBL, _CHN, htEntry[i].data))))
        {
            chanNum = DRF_VAL(_UDISP, _HASH_TBL, _CHN, htEntry[i].data);
            if (((LwU32)chanNum) >= pDisp[indexGpu].dispGetMaxChan())
            {
                continue;
            }

            //
            // Channels where we can't use debug port to send methods
            // don't have any ctx dmas
            // XXXDISP: Fix this assumption in future.
            //
            if (!(dispChanState_v03_00[chanNum].cap & DISP_DEBUG))
            {
                continue;
            }

            if (searchAllHandles || (handle == DRF_VAL(_UDISP, _HASH_TBL, _HANDLE, htEntry[i].handle)))
            {
                handle = DRF_VAL(_UDISP, _HASH_TBL, _HANDLE, htEntry[i].handle);
                chanNum = DRF_VAL(_UDISP, _HASH_TBL, _CHN, htEntry[i].data);
                clientID = DRF_VAL(_UDISP, _HASH_TBL, _CLIENT_ID, htEntry[i].data);

                if (handle == 0)
                {
                    continue;
                }
                handleFound = TRUE;

                dprintf("===============================================================================================\n");
                dprintf("%-30s: " PhysAddr_FMT " in %s\n", "FE_INST_MEM", feInstMemAddr, isPhysLwm ? "VIDMEM" : "SYSMEM");
                dprintf("%-30s: 0x%x\n", "Handle", handle);
                dprintf("%-30s: %s(%d)\n", "Channel", dispChanState_v03_00[chanNum].name, dispChanState_v03_00[chanNum].headNum);
                dprintf("%-30s: 0x%x\n", "Client ID", clientID);

                // All objects are allocated in chunks of LW_DMA_ALIGN bytes, the offset read is multiplied by 32 to get the actual OBJ_MEM offset.
                objMemOffset = instance * LW_DMA_ALIGN;
                dprintf("%-30s: 0x%x bytes\n", "Offset from base of HT", objMemOffset);
                objMemStart = dispHTAddr + objMemOffset;
                dprintf("%-30s: " PhysAddr_FMT " in %s\n", "Object Address", objMemStart, isPhysLwm ? "VIDMEM" : "SYSMEM");

                if (isPhysLwm)
                    status = pFb[indexGpu].fbRead(objMemStart, &dmaObj, sizeof(dmaObj));
                else
                    status = readSystem(objMemStart, &dmaObj, sizeof(dmaObj));

                if (status != LW_OK)
                {
                    dprintf("Failed to read hash table entry.\n");
                    return;
                }

                switch (DRF_VAL(_DMA, _TARGET, _NODE, dmaObj.misc))
                {
                    case LW_DMA_TARGET_NODE_PHYSICAL_LWM:
                        dprintf("%-30s: %s\n", "LW_DMA_TARGET_NODE", "PHYSICAL_LWM");
                        break;
                    case LW_DMA_TARGET_NODE_PHYSICAL_PCI:
                        dprintf("%-30s: %s\n", "LW_DMA_TARGET_NODE", "PHYSICAL_PCI");
                        break;
                    case LW_DMA_TARGET_NODE_PHYSICAL_PCI_COHERENT:
                        dprintf("%-30s: %s\n", "LW_DMA_TARGET_NODE", "PHYSICAL_PCI_COHERENT");
                        break;
                    default:
                        dprintf("%-30s: %s\n", "LW_DMA_TARGET_NODE", "UNKNOWN");
                }

                switch (DR_VAL(_DMA, _ACCESS, dmaObj.misc))
                {
                    case LW_DMA_ACCESS_READ_ONLY:
                        dprintf("%-30s: %s\n", "LW_DMA_ACCESS", "READ_ONLY");
                        break;
                    case LW_DMA_ACCESS_READ_AND_WRITE:
                        dprintf("%-30s: %s\n", "LW_DMA_ACCESS", "READ_AND_WRITE");
                        break;
                    default:
                        dprintf("%-30s: %s\n", "LW_DMA_ACCESS", "UNKNOWN");
                }

                dprintf("%-30s: %s\n", "LW_DMA_PAGE_SIZE", (DR_VAL(_DMA, _PAGE_SIZE, dmaObj.misc) == LW_DMA_PAGE_SIZE_SMALL)? "SMALL" : "BIG");
                dprintf("%-30s: %s\n", "LW_DMA_KIND", (DR_VAL(_DMA, _KIND, dmaObj.misc) == LW_DMA_KIND_PITCH)? "PITCH" : "BLOCKLINEAR");

                tmpAddr = DRF_VAL(_DMA, _ADDRESS, _BASE_LO,  dmaObj.baseLo);
                tmpAddr |= ((PhysAddr)DRF_VAL(_DMA, _ADDRESS, _BASE_HI,  dmaObj.baseHi)) << DRF_SIZE(LW_DMA_ADDRESS_BASE_LO);
                dprintf("%-30s: " PhysAddr_FMT " bytes\n", "LW_DMA_ADDRESS_BASE",     tmpAddr << LW_DMA_ADDRESS_BASE_SHIFT);

                tmpAddr = DRF_VAL(_DMA, _ADDRESS, _LIMIT_LO, dmaObj.limitLo);
                tmpAddr |= ((PhysAddr)DRF_VAL(_DMA, _ADDRESS, _LIMIT_HI,  dmaObj.limitHi)) << DRF_SIZE(LW_DMA_ADDRESS_LIMIT_LO);
                dprintf("%-30s: " PhysAddr_FMT " bytes\n", "LW_DMA_ADDRESS_LIMIT",    tmpAddr << LW_DMA_ADDRESS_BASE_SHIFT);
            }
        }
    }

    if ( (handleFound == FALSE) && (searchAllHandles == FALSE) )
    {
        if (searchAllChannels)
        {
            dprintf("Queried handle 0x%x could not be found in the Hash Table!\n", handle);
        }
        else
        {
            dprintf("Queried handle 0x%x could not be found in channel number %d in the Hash Table!\n", handle, chanNum);
        }
    }
}

void dispAnalyzeInterrupts_v03_00 (LwU32 all, LwU32 evt, LwU32 intr, LwU32 dispatch)
{
    LwU32 evtDispatch;
    LwU32 rm_intrDispatch;
    LwU32 pmu_intrDispatch;
    LwU32 gsp_intrDispatch;
    LwU32 evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen;
    LwU32 evtEn, ie, rm_im, pmu_im, gsp_im;
    LwU32 head = 0, channelNum = 0, orNum = 0, svNum = 0, i = 0;
    LwU32 channelNumMax = 0, orNumMax = 0, headMax = 0;
    LwU32 evtStat, evtEnClr, rm_intrStat, pmu_intrStat, gsp_intrStat, intrEn,
        rm_intrMask, pmu_intrMask, gsp_intrMask, evtStat1;

    if (dispatch)
    {
        dprintf ("----------------------------------------------------------------------------------------------------------\n");
        dprintf ("----------------------------------------------------------------------------------------------------------\n");
        dprintf ("%-40s|", "");
        dprintf ("%-29s|%-29s\n", "", "INTERRUPT DISPATCH: PENDING?");
        dprintf ("%-40s|%-29s|%s\n", "HIGH LEVEL EVENT/INTERRUPT NAME", "EVENT DISPATCH: PENDING?",
            "-----------------------------");
        dprintf ("%-40s|%-29s|%-9s|%-9s|%-9s\n", "", "", "RM", "PMU", "GSP");
        dprintf ("----------------------------------------------------------------------------------------------------------\n");
        dprintf ("----------------------------------------------------------------------------------------------------------\n");


        evtDispatch = GPU_REG_RD32(LW_PDISP_FE_EVT_DISPATCH);
        //dprintf ("**EVTDISPATCH info: evtDispatch 0x%x\n", evtDispatch);

        rm_intrDispatch = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_DISPATCH);
        pmu_intrDispatch = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_DISPATCH);
        gsp_intrDispatch = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_DISPATCH);
        //dprintf ("**INTRDISPATCH info: rm_intrDispatch 0x%x, pmu_intrDispatch 0x%x, gsp_intrDispatch 0x%x\n",
        //rm_intrDispatch, pmu_intrDispatch, gsp_intrDispatch);


        ////////////////////////////
        //Awaken Events/Interrupts//
        ////////////////////////////

        dprintf ("%-40s|", "AWAKEN");
        evtPen = ((DRF_VAL(_PDISP, _FE_EVT_DISPATCH, _AWAKEN_WIN, evtDispatch) ==
            LW_PDISP_FE_EVT_DISPATCH_AWAKEN_WIN_PENDING) |
            (DRF_VAL(_PDISP, _FE_EVT_DISPATCH, _AWAKEN_OTHER, evtDispatch) ==
            LW_PDISP_FE_EVT_DISPATCH_AWAKEN_OTHER_PENDING));
        rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_DISPATCH, _CTRL_DISP, rm_intrDispatch) ==
            LW_PDISP_FE_RM_INTR_DISPATCH_CTRL_DISP_PENDING);
        pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_DISPATCH, _CTRL_DISP, pmu_intrDispatch) ==
            LW_PDISP_FE_PMU_INTR_DISPATCH_CTRL_DISP_PENDING);
        gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_DISPATCH, _CTRL_DISP, gsp_intrDispatch) ==
            LW_PDISP_FE_GSP_INTR_DISPATCH_CTRL_DISP_PENDING);
        dprintf ("%-29s|%-9s|%-9s|%-9s\n",
            ynfunc(evtPen), ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen));


        ///////////////////////////////
        //Exception Events/Interrupts//
        ///////////////////////////////

        dprintf ("%-40s|", "EXCEPTION");
        evtPen = ((DRF_VAL(_PDISP, _FE_EVT_DISPATCH, _EXC_WIN, evtDispatch) ==
            LW_PDISP_FE_EVT_DISPATCH_EXC_WIN_PENDING) |
            (DRF_VAL(_PDISP, _FE_EVT_DISPATCH, _EXC_WINIM, evtDispatch) ==
            LW_PDISP_FE_EVT_DISPATCH_EXC_WINIM_PENDING) |
            (DRF_VAL(_PDISP, _FE_EVT_DISPATCH, _EXC_OTHER, evtDispatch) ==
            LW_PDISP_FE_EVT_DISPATCH_EXC_OTHER_PENDING));

        rm_intrPen = ((DRF_VAL(_PDISP, _FE_RM_INTR_DISPATCH, _EXC_WIN, rm_intrDispatch) ==
            LW_PDISP_FE_RM_INTR_DISPATCH_EXC_WIN_PENDING) |
            (DRF_VAL(_PDISP, _FE_RM_INTR_DISPATCH, _EXC_WINIM, rm_intrDispatch) ==
            LW_PDISP_FE_RM_INTR_DISPATCH_EXC_WINIM_PENDING) |
            (DRF_VAL(_PDISP, _FE_RM_INTR_DISPATCH, _EXC_OTHER, rm_intrDispatch) ==
            LW_PDISP_FE_RM_INTR_DISPATCH_EXC_OTHER_PENDING));

        pmu_intrPen = ((DRF_VAL(_PDISP, _FE_PMU_INTR_DISPATCH, _EXC_WIN, pmu_intrDispatch) ==
            LW_PDISP_FE_PMU_INTR_DISPATCH_EXC_WIN_PENDING) |
            (DRF_VAL(_PDISP, _FE_PMU_INTR_DISPATCH, _EXC_WINIM, pmu_intrDispatch) ==
            LW_PDISP_FE_PMU_INTR_DISPATCH_EXC_WINIM_PENDING) |
            (DRF_VAL(_PDISP, _FE_PMU_INTR_DISPATCH, _EXC_OTHER, pmu_intrDispatch) ==
            LW_PDISP_FE_PMU_INTR_DISPATCH_EXC_OTHER_PENDING));

        gsp_intrPen = ((DRF_VAL(_PDISP, _FE_GSP_INTR_DISPATCH, _EXC_WIN, gsp_intrDispatch) ==
            LW_PDISP_FE_GSP_INTR_DISPATCH_EXC_WIN_PENDING) |
            (DRF_VAL(_PDISP, _FE_GSP_INTR_DISPATCH, _EXC_WINIM, gsp_intrDispatch) ==
            LW_PDISP_FE_GSP_INTR_DISPATCH_EXC_WINIM_PENDING) |
            (DRF_VAL(_PDISP, _FE_GSP_INTR_DISPATCH, _EXC_OTHER, gsp_intrDispatch) ==
            LW_PDISP_FE_GSP_INTR_DISPATCH_EXC_OTHER_PENDING));

        dprintf ("%-29s|%-9s|%-9s|%-9s\n",
            ynfunc(evtPen), ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen));


        ////////////////////////////
        //OR Events/Interrupts//
        ////////////////////////////

        dprintf ("%-40s|", "OR");
        evtPen = (DRF_VAL(_PDISP, _FE_EVT_DISPATCH, _OR, evtDispatch) ==
            LW_PDISP_FE_EVT_DISPATCH_OR_PENDING);
        rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_DISPATCH, _OR, rm_intrDispatch) ==
            LW_PDISP_FE_RM_INTR_DISPATCH_OR_PENDING);
        pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_DISPATCH, _OR, pmu_intrDispatch) ==
            LW_PDISP_FE_PMU_INTR_DISPATCH_OR_PENDING);
        gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_DISPATCH, _OR, gsp_intrDispatch) ==
            LW_PDISP_FE_GSP_INTR_DISPATCH_OR_PENDING);
        dprintf ("%-29s|%-9s|%-9s|%-9s\n",
            ynfunc(evtPen), ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen));


        ////////////////////////////////
        //Supervisor Events/Interrupts//
        ////////////////////////////////

        dprintf ("%-40s|", "SUPERVISOR");
        evtPen = (DRF_VAL(_PDISP, _FE_EVT_DISPATCH, _CTRL_DISP, evtDispatch) ==
            LW_PDISP_FE_EVT_DISPATCH_CTRL_DISP_PENDING);
        rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_DISPATCH, _CTRL_DISP, rm_intrDispatch) ==
            LW_PDISP_FE_RM_INTR_DISPATCH_CTRL_DISP_PENDING);
        pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_DISPATCH, _CTRL_DISP, pmu_intrDispatch) ==
            LW_PDISP_FE_PMU_INTR_DISPATCH_CTRL_DISP_PENDING);
        gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_DISPATCH, _CTRL_DISP, gsp_intrDispatch) ==
            LW_PDISP_FE_GSP_INTR_DISPATCH_CTRL_DISP_PENDING);
        dprintf ("%-29s|%-9s|%-9s|%-9s\n",
            ynfunc(evtPen), ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen));


        /////////////////////////////////
        //Head Timing Events/Interrupts//
        /////////////////////////////////

        for (head = 0; head < LW_PDISP_FE_EVT_DISPATCH_HEAD_TIMING__SIZE_1; ++head)
        {
            dprintf ("HEAD_%d_TIMING", head);
            dprintf ("%-27s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_DISPATCH, _HEAD_TIMING(head), evtDispatch) ==
                LW_PDISP_FE_EVT_DISPATCH_HEAD_TIMING_PENDING);
            rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_DISPATCH, _HEAD_TIMING(head), rm_intrDispatch) ==
                LW_PDISP_FE_RM_INTR_DISPATCH_HEAD_TIMING_PENDING);
            pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_DISPATCH, _HEAD_TIMING(head), pmu_intrDispatch) ==
                LW_PDISP_FE_PMU_INTR_DISPATCH_HEAD_TIMING_PENDING);
            gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_DISPATCH, _HEAD_TIMING(head), gsp_intrDispatch) ==
                LW_PDISP_FE_GSP_INTR_DISPATCH_HEAD_TIMING_PENDING);
            dprintf ("%-29s|%-9s|%-9s|%-9s\n",
                ynfunc(evtPen), ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen));
        }


        ////////////////////////////////
        //Head LWDPS Events/Interrupts//
        ////////////////////////////////

        rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_DISPATCH, _HEAD_LWDPS, rm_intrDispatch) ==
            LW_PDISP_FE_RM_INTR_DISPATCH_HEAD_LWDPS_PENDING);
        pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_DISPATCH, _HEAD_LWDPS, pmu_intrDispatch) ==
            LW_PDISP_FE_PMU_INTR_DISPATCH_HEAD_LWDPS_PENDING);
        gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_DISPATCH, _HEAD_LWDPS, gsp_intrDispatch) ==
            LW_PDISP_FE_GSP_INTR_DISPATCH_HEAD_LWDPS_PENDING);
        for (head = 0; head < LW_PDISP_FE_EVT_DISPATCH_HEAD_LWDPS__SIZE_1; ++head)
        {
            dprintf ("HEAD_%d_LWDPS", head);
            dprintf ("%-28s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_DISPATCH, _HEAD_LWDPS(head), evtDispatch) ==
                LW_PDISP_FE_EVT_DISPATCH_HEAD_LWDPS_PENDING);
            dprintf ("%-29s|%-9s|%-9s|%-9s\n",
                ynfunc(evtPen), ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen));
        }

        dprintf ("----------------------------------------------------------------------------------------------------------\n");

    }
    dprintf ("\n");


    if (evt)
    {
        //*************************************************************EVENTS*******************************************************
        dprintf ("----------------------------------------------------------------------------------------------------------\n");
        dprintf ("----------------------------------------------------------------------------------------------------------\n");
        dprintf ("%-40s|", "EVENT NAME");
        dprintf ("%-10s|", "PENDING?");
        dprintf ("%-10s\n", "ENABLED?");
        dprintf ("----------------------------------------------------------------------------------------------------------\n");
        dprintf ("----------------------------------------------------------------------------------------------------------\n");

        channelNumMax = (all ? LW_PDISP_FE_EVT_STAT_AWAKEN_WIN_CH__SIZE_1 : 8);
        //Awaken Window Events

        evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_AWAKEN_WIN);
        evtEnClr = GPU_REG_RD32 (LW_PDISP_FE_EVT_EN_CLR_AWAKEN_WIN);

        for (channelNum = 0; channelNum < channelNumMax; ++channelNum)
        {
            dprintf ("AWAKEN_WIN_CH_");
            dprintf ("%d", channelNum);
            channelNum < 10? dprintf ("%25s|", "") : dprintf ("%24s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_AWAKEN_WIN, _CH(channelNum), evtStat) ==
                LW_PDISP_FE_EVT_STAT_AWAKEN_WIN_CH_PENDING);
            evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_AWAKEN_WIN, _CH(channelNum), evtEnClr) ==
                LW_PDISP_FE_EVT_EN_CLR_AWAKEN_WIN_CH_ENABLE);
            dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));
        }
        dprintf ("----------------------------------------------------------------------------------------------------------\n");


        //Awaken Core Event

        evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_AWAKEN_OTHER);
        evtEnClr = GPU_REG_RD32(LW_PDISP_FE_EVT_EN_CLR_AWAKEN_OTHER);
        dprintf ("%-40s|", "AWAKEN_CORE");
        evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_AWAKEN_OTHER, _CORE, evtStat) ==
            LW_PDISP_FE_EVT_STAT_AWAKEN_OTHER_CORE_PENDING);
        evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_AWAKEN_OTHER, _CORE, evtEnClr) ==
            LW_PDISP_FE_EVT_EN_CLR_AWAKEN_OTHER_CORE_ENABLE);
        dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));
        dprintf ("----------------------------------------------------------------------------------------------------------\n");


        channelNumMax = (all ? LW_PDISP_FE_EVT_STAT_EXC_WIN_CH__SIZE_1 : 8);
        //Exception Window Events

        evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_EXC_WIN);
        evtEnClr = GPU_REG_RD32(LW_PDISP_FE_EVT_EN_CLR_EXC_WIN);
        for (channelNum = 0; channelNum < channelNumMax; ++channelNum)
        {
            dprintf ("EXCEPTION_WIN_CH_");
            dprintf ("%d", channelNum);
            channelNum < 10? dprintf ("%22s|", "") : dprintf ("%21s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_EXC_WIN, _CH(channelNum), evtStat) ==
                LW_PDISP_FE_EVT_STAT_EXC_WIN_CH_PENDING);
            evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_EXC_WIN, _CH(channelNum), evtEnClr) ==
                LW_PDISP_FE_EVT_EN_CLR_EXC_WIN_CH_ENABLE);
            dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));
        }
        dprintf ("----------------------------------------------------------------------------------------------------------\n");


        channelNumMax = (all ? LW_PDISP_FE_EVT_STAT_EXC_WINIM_CH__SIZE_1 : 8);
        //Exception Window Imm Events

        evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_EXC_WINIM);
        evtEnClr = GPU_REG_RD32(LW_PDISP_FE_EVT_EN_CLR_EXC_WINIM);
        for (channelNum = 0; channelNum < channelNumMax; ++channelNum)
        {
            dprintf ("EXCEPTION_WINIM_CH_");
            dprintf ("%d", channelNum);
            channelNum < 10? dprintf ("%20s|", "") : dprintf ("%19s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_EXC_WINIM, _CH(channelNum), evtStat) ==
                LW_PDISP_FE_EVT_STAT_EXC_WINIM_CH_PENDING);
            evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_EXC_WINIM, _CH(channelNum), evtEnClr) ==
                LW_PDISP_FE_EVT_EN_CLR_EXC_WINIM_CH_ENABLE);
            dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));
        }
        dprintf ("----------------------------------------------------------------------------------------------------------\n");


        //Exception Core Event

        evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_EXC_OTHER);
        evtEnClr = GPU_REG_RD32(LW_PDISP_FE_EVT_EN_CLR_EXC_OTHER);
        dprintf ("%-40s|", "EXCEPTION_CORE");
        evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_EXC_OTHER, _CORE, evtStat) ==
            LW_PDISP_FE_EVT_STAT_EXC_OTHER_CORE_PENDING);
        evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_EXC_OTHER, _CORE, evtEnClr) ==
            LW_PDISP_FE_EVT_EN_CLR_EXC_OTHER_CORE_ENABLE);
        dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));
        dprintf ("----------------------------------------------------------------------------------------------------------\n");


        channelNumMax = (all ? LW_PDISP_FE_EVT_STAT_EXC_OTHER_LWRS__SIZE_1 : 4);
        //Exception Cursor Events

        for (channelNum = 0; channelNum < channelNumMax; ++channelNum)
        {
            dprintf ("EXCEPTION_LWRS_");
            dprintf ("%d", channelNum);
            dprintf ("%24s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_EXC_OTHER, _LWRS(channelNum), evtStat) ==
                LW_PDISP_FE_EVT_STAT_EXC_OTHER_LWRS_PENDING);
            evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_EXC_OTHER, _LWRS(channelNum), evtEnClr) ==
                LW_PDISP_FE_EVT_EN_CLR_EXC_OTHER_LWRS_ENABLE);
            dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));
        }
        dprintf ("----------------------------------------------------------------------------------------------------------\n");


        orNumMax = (all ? LW_PDISP_FE_EVT_STAT_OR_SOR__SIZE_1 : 4);
        //OR_SOR Events

        evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_OR);
        evtEnClr = GPU_REG_RD32(LW_PDISP_FE_EVT_EN_CLR_OR);
        for (orNum = 0; orNum < orNumMax; ++orNum)
        {
            dprintf ("OR_SOR_");
            dprintf ("%d", orNum);
            dprintf ("%32s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_OR, _SOR(orNum), evtStat) ==
                LW_PDISP_FE_EVT_STAT_OR_SOR_PENDING);
            evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_OR, _SOR(orNum), evtEnClr) ==
                LW_PDISP_FE_EVT_EN_CLR_OR_SOR_ENABLE);
            dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));
        }
        dprintf ("----------------------------------------------------------------------------------------------------------\n");


        if (all)
        {
            //OR_PIOR Events

            for (orNum = 0; orNum < LW_PDISP_FE_EVT_STAT_OR_PIOR__SIZE_1; ++orNum)
            {
                dprintf ("OR_PIOR_");
                dprintf ("%d", orNum);
                dprintf ("%31s|", "");
                evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_OR, _PIOR(orNum), evtStat) ==
                    LW_PDISP_FE_EVT_STAT_OR_PIOR_PENDING);
                evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_OR, _PIOR(orNum), evtEnClr) ==
                    LW_PDISP_FE_EVT_EN_CLR_OR_PIOR_ENABLE);
                dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));
            }
            dprintf ("----------------------------------------------------------------------------------------------------------\n");
        }


        //Supervisor Events

        evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_CTRL_DISP);
        evtEnClr = GPU_REG_RD32(LW_PDISP_FE_EVT_EN_CLR_CTRL_DISP);
        for (svNum = 0; svNum < LW_PDISP_FE_EVT_STAT_CTRL_DISP_SUPERVISOR__SIZE_1; ++svNum)
        {
            dprintf ("SUPERVISOR_");
            dprintf ("%d", svNum + 1);
            dprintf ("%28s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_CTRL_DISP, _SUPERVISOR(svNum), evtStat) ==
                LW_PDISP_FE_EVT_STAT_CTRL_DISP_SUPERVISOR_PENDING);
            evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_CTRL_DISP, _SUPERVISOR(svNum), evtEnClr) ==
                LW_PDISP_FE_EVT_EN_CLR_CTRL_DISP_SUPERVISOR_ENABLE);
            dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));
        }
        dprintf ("----------------------------------------------------------------------------------------------------------\n");


        headMax = (all ?  LW_PDISP_FE_EVT_STAT_HEAD_TIMING__SIZE_1 : 4);
        //Head Timing Events

        for (head = 0; head < headMax; ++head)
        {
            evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_HEAD_TIMING(head));
            evtEnClr = GPU_REG_RD32(LW_PDISP_FE_EVT_EN_CLR_HEAD_TIMING(head));

            dprintf ("HEAD%d_TIMING_LOADV", head);
            dprintf ("%22s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _LOADV, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_LOADV_PENDING);
            evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_HEAD_TIMING, _LOADV, evtEnClr) ==
                LW_PDISP_FE_EVT_EN_CLR_HEAD_TIMING_LOADV_ENABLE);
            dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));

            dprintf ("HEAD%d_TIMING_LAST_DATA", head);
            dprintf ("%18s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _LAST_DATA, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_LAST_DATA_PENDING);
            evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_HEAD_TIMING, _LAST_DATA, evtEnClr) ==
                LW_PDISP_FE_EVT_EN_CLR_HEAD_TIMING_LAST_DATA_ENABLE);
            dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));

            dprintf ("HEAD%d_TIMING_VBLANK", head);
            dprintf ("%21s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _VBLANK, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_VBLANK_PENDING);
            evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_HEAD_TIMING, _VBLANK, evtEnClr) ==
                LW_PDISP_FE_EVT_EN_CLR_HEAD_TIMING_VBLANK_ENABLE);
            dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));

            dprintf ("HEAD%d_TIMING_VACTIVE_SPACE_VBLANK", head);
            dprintf ("%7s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _VACTIVE_SPACE_VBLANK, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_VACTIVE_SPACE_VBLANK_PENDING);
            evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_HEAD_TIMING, _VACTIVE_SPACE_VBLANK, evtEnClr) ==
                LW_PDISP_FE_EVT_EN_CLR_HEAD_TIMING_VACTIVE_SPACE_VBLANK_ENABLE);
            dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));

            dprintf ("HEAD%d_TIMING_RG_STALL", head);
            dprintf ("%19s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _RG_STALL, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_RG_STALL_PENDING);
            evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_HEAD_TIMING, _RG_STALL, evtEnClr) ==
                LW_PDISP_FE_EVT_EN_CLR_HEAD_TIMING_RG_STALL_ENABLE);
            dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));

            dprintf ("HEAD%d_TIMING_RG_LINE_A", head);
            dprintf ("%18s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _RG_LINE_A, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_RG_LINE_A_PENDING);
            evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_HEAD_TIMING, _RG_LINE_A, evtEnClr) ==
                LW_PDISP_FE_EVT_EN_CLR_HEAD_TIMING_RG_LINE_A_ENABLE);
            dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));

            dprintf ("HEAD%d_TIMING_RG_LINE_B", head);
            dprintf ("%18s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _RG_LINE_B, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_RG_LINE_B_PENDING);
            evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_HEAD_TIMING, _RG_LINE_B, evtEnClr) ==
                LW_PDISP_FE_EVT_EN_CLR_HEAD_TIMING_RG_LINE_B_ENABLE);
            dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));

            dprintf ("HEAD%d_TIMING_SD3_BUCKET_WALK_DONE", head);
            dprintf ("%7s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _SD3_BUCKET_WALK_DONE, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_SD3_BUCKET_WALK_DONE_PENDING);
            evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_HEAD_TIMING, _SD3_BUCKET_WALK_DONE, evtEnClr) ==
                LW_PDISP_FE_EVT_EN_CLR_HEAD_TIMING_SD3_BUCKET_WALK_DONE_ENABLE);
            dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));

            dprintf ("HEAD%d_TIMING_SEC_POLICY", head);
            dprintf ("%17s|", "");
            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _SEC_POLICY, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_SEC_POLICY_PENDING);
            evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_HEAD_TIMING, _SEC_POLICY, evtEnClr) ==
                LW_PDISP_FE_EVT_EN_CLR_HEAD_TIMING_SEC_POLICY_ENABLE);
            dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));

            dprintf ("----------------------------------------------------------------------------------------------------------\n");
        }


        if (all)
        {
            //Head LWDPS Events

            for (head = 0; head < LW_PDISP_FE_EVT_STAT_HEAD_LWDPS__SIZE_1; ++head)
            {
                evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_HEAD_LWDPS(head));
                evtEnClr = GPU_REG_RD32(LW_PDISP_FE_EVT_EN_CLR_HEAD_LWDPS(head));

                dprintf ("HEAD%d_LWDPS_STATISTIC_COUNTERS_MSB_SET", head);
                dprintf ("%2s|", "");
                evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_LWDPS, _STATISTIC_COUNTERS_MSB_SET, evtStat) ==
                    LW_PDISP_FE_EVT_STAT_HEAD_LWDPS_STATISTIC_COUNTERS_MSB_SET_PENDING);
                evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_HEAD_LWDPS, _STATISTIC_COUNTERS_MSB_SET, evtEnClr) ==
                    LW_PDISP_FE_EVT_EN_CLR_HEAD_LWDPS_STATISTIC_COUNTERS_MSB_SET_ENABLE);
                dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));

                for (i = 0; i < LW_PDISP_FE_EVT_STAT_HEAD_LWDPS_STATISTIC_GATHER__SIZE_1; ++i)
                {
                    dprintf ("HEAD%d_LWDPS_STATISTIC_GATHER_%d", head, i);
                    dprintf ("%10s|", "");
                    evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_LWDPS, _STATISTIC_GATHER(i), evtStat) ==
                        LW_PDISP_FE_EVT_STAT_HEAD_LWDPS_STATISTIC_GATHER_PENDING);
                    evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_HEAD_LWDPS, _STATISTIC_GATHER(i), evtEnClr) ==
                        LW_PDISP_FE_EVT_EN_CLR_HEAD_LWDPS_STATISTIC_GATHER_ENABLE);
                    dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));
                }

                for (i = 0; i < LW_PDISP_FE_EVT_STAT_HEAD_LWDPS_FILTERED_STATISTIC_GATHER_UPPER_BOUND__SIZE_1; ++i)
                {
                    dprintf ("HEAD%d_LWDPS_FILTERED_STAT_GTHR_UPR_BND_%d|", head, i);
                    evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_LWDPS, _FILTERED_STATISTIC_GATHER_UPPER_BOUND(i), evtStat) ==
                        LW_PDISP_FE_EVT_STAT_HEAD_LWDPS_FILTERED_STATISTIC_GATHER_UPPER_BOUND_PENDING);
                    evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_HEAD_LWDPS, _FILTERED_STATISTIC_GATHER_UPPER_BOUND(i), evtEnClr) ==
                        LW_PDISP_FE_EVT_EN_CLR_HEAD_LWDPS_FILTERED_STATISTIC_GATHER_UPPER_BOUND_ENABLE);
                    dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));
                }

                for (i = 0; i < LW_PDISP_FE_EVT_STAT_HEAD_LWDPS_FILTERED_STATISTIC_GATHER_LOWER_BOUND__SIZE_1; ++i)
                {
                    dprintf ("HEAD%d_LWDPS_FILTERED_STAT_GTHR_LWR_BND_%d|", head, i);
                    evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_LWDPS, _FILTERED_STATISTIC_GATHER_LOWER_BOUND(i), evtStat) ==
                        LW_PDISP_FE_EVT_STAT_HEAD_LWDPS_FILTERED_STATISTIC_GATHER_LOWER_BOUND_PENDING);
                    evtEn = (DRF_VAL(_PDISP, _FE_EVT_EN_CLR_HEAD_LWDPS, _FILTERED_STATISTIC_GATHER_LOWER_BOUND(i), evtEnClr) ==
                        LW_PDISP_FE_EVT_EN_CLR_HEAD_LWDPS_FILTERED_STATISTIC_GATHER_LOWER_BOUND_ENABLE);
                    dprintf ("%-10s|%-10s\n", ynfunc(evtPen), ynfunc(evtEn));
                }

                dprintf ("----------------------------------------------------------------------------------------------------------\n");
            }
        }

    }

    dprintf ("\n");



    if (intr)
    {
        //******************************************************INTERRUPTS**********************************************************
        dprintf ("----------------------------------------------------------------------------------------------------------\n");
        dprintf ("----------------------------------------------------------------------------------------------------------\n");
        dprintf ("%-40s|", "");
        dprintf ("%-11s|%-8s|%-11s|%-32s\n", "PENDING?", "ENABLED?", "TARGET", "");
        dprintf ("%-40s|%11.11s|%-8s|%11.11s|%-32s\n", "INTERRUPT NAME", "-------------------------","   RM",
            "----------------------", "SANITY TEST");
        dprintf ("%-40s|%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n", "", "RM", "PMU", "GSP", "  ONLY", "RM", "PMU", "GSP", "");
        dprintf ("----------------------------------------------------------------------------------------------------------\n");
        dprintf ("----------------------------------------------------------------------------------------------------------\n");



        //Awaken Interrupts

        evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_AWAKEN_WIN);
        evtStat1 = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_AWAKEN_OTHER);
        evtPen = (evtStat!=0)|(evtStat1!=0);

        rm_intrStat = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_STAT_CTRL_DISP);
        pmu_intrStat = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_STAT_CTRL_DISP);
        gsp_intrStat = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_STAT_CTRL_DISP);
        intrEn = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_EN_CTRL_DISP);
        rm_intrMask = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_MSK_CTRL_DISP);
        pmu_intrMask = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_MSK_CTRL_DISP);
        gsp_intrMask = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_MSK_CTRL_DISP);

        dprintf ("%-40s|", "AWAKEN");

        rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_CTRL_DISP, _AWAKEN, rm_intrStat) ==
            LW_PDISP_FE_RM_INTR_STAT_CTRL_DISP_AWAKEN_PENDING);
        pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_CTRL_DISP, _AWAKEN, pmu_intrStat) ==
            LW_PDISP_FE_PMU_INTR_STAT_CTRL_DISP_AWAKEN_PENDING);
        gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_CTRL_DISP, _AWAKEN, gsp_intrStat) ==
            LW_PDISP_FE_GSP_INTR_STAT_CTRL_DISP_AWAKEN_PENDING);
        ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_CTRL_DISP, _AWAKEN, intrEn) ==
            LW_PDISP_FE_RM_INTR_EN_CTRL_DISP_AWAKEN_ENABLE);
        rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_CTRL_DISP, _AWAKEN, rm_intrMask) ==
            LW_PDISP_FE_RM_INTR_MSK_CTRL_DISP_AWAKEN_ENABLE);
        pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_CTRL_DISP, _AWAKEN, pmu_intrMask) ==
            LW_PDISP_FE_PMU_INTR_MSK_CTRL_DISP_AWAKEN_ENABLE);
        gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_CTRL_DISP, _AWAKEN, gsp_intrMask) ==
            LW_PDISP_FE_GSP_INTR_MSK_CTRL_DISP_AWAKEN_ENABLE);

        dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
            ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
            santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));

        dprintf ("----------------------------------------------------------------------------------------------------------\n");



        channelNumMax = (all ? LW_PDISP_FE_RM_INTR_STAT_EXC_WIN_CH__SIZE_1 : 8);
        //Exception Window Interrupts

        evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_EXC_WIN);

        rm_intrStat = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_STAT_EXC_WIN);
        pmu_intrStat = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_STAT_EXC_WIN);
        gsp_intrStat = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_STAT_EXC_WIN);
        intrEn = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_EN_EXC_WIN);
        rm_intrMask = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_MSK_EXC_WIN);
        pmu_intrMask = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_MSK_EXC_WIN);
        gsp_intrMask = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_MSK_EXC_WIN);

        for (channelNum = 0; channelNum < channelNumMax; ++channelNum)
        {
            dprintf ("EXCEPTION_WIN_CH_%d", channelNum);
            channelNum < 10? dprintf ("%22s|", "") : dprintf ("%21s|", "");

            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_EXC_WIN, _CH(channelNum), evtStat) ==
                LW_PDISP_FE_EVT_STAT_EXC_WIN_CH_PENDING);
            rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_EXC_WIN, _CH(channelNum), rm_intrStat) ==
                LW_PDISP_FE_RM_INTR_STAT_EXC_WIN_CH_PENDING);
            pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_EXC_WIN, _CH(channelNum), pmu_intrStat) ==
                LW_PDISP_FE_PMU_INTR_STAT_EXC_WIN_CH_PENDING);
            gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_EXC_WIN, _CH(channelNum), gsp_intrStat) ==
                LW_PDISP_FE_GSP_INTR_STAT_EXC_WIN_CH_PENDING);
            ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_EXC_WIN, _CH(channelNum), intrEn) ==
                LW_PDISP_FE_RM_INTR_EN_EXC_WIN_CH_ENABLE);
            rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_EXC_WIN, _CH(channelNum), rm_intrMask) ==
                LW_PDISP_FE_RM_INTR_MSK_EXC_WIN_CH_ENABLE);
            pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_EXC_WIN, _CH(channelNum), pmu_intrMask) ==
                LW_PDISP_FE_PMU_INTR_MSK_EXC_WIN_CH_ENABLE);
            gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_EXC_WIN, _CH(channelNum), gsp_intrMask) ==
                LW_PDISP_FE_GSP_INTR_MSK_EXC_WIN_CH_ENABLE);

            dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));

        }
        dprintf ("----------------------------------------------------------------------------------------------------------\n");



        channelNumMax = (all ? LW_PDISP_FE_RM_INTR_STAT_EXC_WINIM_CH__SIZE_1 : 8);
        //Exception Window Imm Interrupts

        evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_EXC_WINIM);

        rm_intrStat = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_STAT_EXC_WINIM);
        pmu_intrStat = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_STAT_EXC_WINIM);
        gsp_intrStat = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_STAT_EXC_WINIM);
        intrEn = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_EN_EXC_WINIM);
        rm_intrMask = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_MSK_EXC_WINIM);
        pmu_intrMask = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_MSK_EXC_WINIM);
        gsp_intrMask = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_MSK_EXC_WINIM);

        for (channelNum = 0; channelNum < channelNumMax; ++channelNum)
        {
            dprintf ("EXCEPTION_WINIM_CH_%d", channelNum);
            channelNum < 10? dprintf ("%20s|", "") : dprintf ("%19s|", "");

            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_EXC_WINIM, _CH(channelNum), evtStat) ==
                LW_PDISP_FE_EVT_STAT_EXC_WINIM_CH_PENDING);
            rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_EXC_WINIM, _CH(channelNum), rm_intrStat) ==
                LW_PDISP_FE_RM_INTR_STAT_EXC_WINIM_CH_PENDING);
            pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_EXC_WINIM, _CH(channelNum), pmu_intrStat) ==
                LW_PDISP_FE_PMU_INTR_STAT_EXC_WINIM_CH_PENDING);
            gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_EXC_WINIM, _CH(channelNum), gsp_intrStat) ==
                LW_PDISP_FE_GSP_INTR_STAT_EXC_WINIM_CH_PENDING);
            ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_EXC_WINIM, _CH(channelNum), intrEn) ==
                LW_PDISP_FE_RM_INTR_EN_EXC_WINIM_CH_ENABLE);
            rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_EXC_WINIM, _CH(channelNum), rm_intrMask) ==
                LW_PDISP_FE_RM_INTR_MSK_EXC_WINIM_CH_ENABLE);
            pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_EXC_WINIM, _CH(channelNum), pmu_intrMask) ==
                LW_PDISP_FE_PMU_INTR_MSK_EXC_WINIM_CH_ENABLE);
            gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_EXC_WINIM, _CH(channelNum), gsp_intrMask) ==
                LW_PDISP_FE_GSP_INTR_MSK_EXC_WINIM_CH_ENABLE);

            dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));

        }
        dprintf ("----------------------------------------------------------------------------------------------------------\n");



        //Exception Core Interrupt

        evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_EXC_OTHER);

        rm_intrStat = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_STAT_EXC_OTHER);
        pmu_intrStat = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_STAT_EXC_OTHER);
        gsp_intrStat = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_STAT_EXC_OTHER);
        intrEn = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_EN_EXC_OTHER);
        rm_intrMask = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_MSK_EXC_OTHER);
        pmu_intrMask = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_MSK_EXC_OTHER);
        gsp_intrMask = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_MSK_EXC_OTHER);


        dprintf ("%-40s|", "EXCEPTION_CORE");

        evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_EXC_OTHER, _CORE, evtStat) ==
            LW_PDISP_FE_EVT_STAT_EXC_OTHER_CORE_PENDING);
        rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_EXC_OTHER, _CORE, rm_intrStat) ==
            LW_PDISP_FE_RM_INTR_STAT_EXC_OTHER_CORE_PENDING);
        pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_EXC_OTHER, _CORE, pmu_intrStat) ==
            LW_PDISP_FE_PMU_INTR_STAT_EXC_OTHER_CORE_PENDING);
        gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_EXC_OTHER, _CORE, gsp_intrStat) ==
            LW_PDISP_FE_GSP_INTR_STAT_EXC_OTHER_CORE_PENDING);
        ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_EXC_OTHER, _CORE, intrEn) ==
            LW_PDISP_FE_RM_INTR_EN_EXC_OTHER_CORE_ENABLE);
        rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_EXC_OTHER, _CORE, rm_intrMask) ==
            LW_PDISP_FE_RM_INTR_MSK_EXC_OTHER_CORE_ENABLE);
        pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_EXC_OTHER, _CORE, pmu_intrMask) ==
            LW_PDISP_FE_PMU_INTR_MSK_EXC_OTHER_CORE_ENABLE);
        gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_EXC_OTHER, _CORE, gsp_intrMask) ==
            LW_PDISP_FE_GSP_INTR_MSK_EXC_OTHER_CORE_ENABLE);

        dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
            ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
            santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));

        dprintf ("----------------------------------------------------------------------------------------------------------\n");



        channelNumMax = (all ? LW_PDISP_FE_RM_INTR_STAT_EXC_OTHER_LWRS__SIZE_1 : 4);
        //Exception Cursor Interrupts

        for (channelNum = 0; channelNum < channelNumMax; ++channelNum)
        {
            dprintf ("EXCEPTION_LWRS_%d", channelNum);
            dprintf ("%24s|", "");

            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_EXC_OTHER, _LWRS(channelNum), evtStat) ==
                LW_PDISP_FE_EVT_STAT_EXC_OTHER_LWRS_PENDING);
            rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_EXC_OTHER, _LWRS(channelNum), rm_intrStat) ==
                LW_PDISP_FE_RM_INTR_STAT_EXC_OTHER_LWRS_PENDING);
            pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_EXC_OTHER, _LWRS(channelNum), pmu_intrStat) ==
                LW_PDISP_FE_PMU_INTR_STAT_EXC_OTHER_LWRS_PENDING);
            gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_EXC_OTHER, _LWRS(channelNum), gsp_intrStat) ==
                LW_PDISP_FE_GSP_INTR_STAT_EXC_OTHER_LWRS_PENDING);
            ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_EXC_OTHER, _LWRS(channelNum), intrEn) ==
                LW_PDISP_FE_RM_INTR_EN_EXC_OTHER_LWRS_ENABLE);
            rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_EXC_OTHER, _LWRS(channelNum), rm_intrMask) ==
                LW_PDISP_FE_RM_INTR_MSK_EXC_OTHER_LWRS_ENABLE);
            pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_EXC_OTHER, _LWRS(channelNum), pmu_intrMask) ==
                LW_PDISP_FE_PMU_INTR_MSK_EXC_OTHER_LWRS_ENABLE);
            gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_EXC_OTHER, _LWRS(channelNum), gsp_intrMask) ==
                LW_PDISP_FE_GSP_INTR_MSK_EXC_OTHER_LWRS_ENABLE);

            dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));

        }
        dprintf ("----------------------------------------------------------------------------------------------------------\n");



        orNumMax = (all ?  LW_PDISP_FE_RM_INTR_STAT_OR_SOR__SIZE_1 : 4);
        //OR_SOR Interrupts

        evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_OR);

        rm_intrStat = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_STAT_OR);
        pmu_intrStat = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_STAT_OR);
        gsp_intrStat = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_STAT_OR);
        intrEn = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_EN_OR);
        rm_intrMask = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_MSK_OR);
        pmu_intrMask = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_MSK_OR);
        gsp_intrMask = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_MSK_OR);

        for (orNum = 0; orNum < orNumMax; ++orNum)
        {
            dprintf ("OR_SOR_%d", orNum);
            dprintf ("%32s|", "");

            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_OR, _SOR(orNum), evtStat) ==
                LW_PDISP_FE_EVT_STAT_OR_SOR_PENDING);
            rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_OR, _SOR(orNum), rm_intrStat) ==
                LW_PDISP_FE_RM_INTR_STAT_OR_SOR_PENDING);
            pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_OR, _SOR(orNum), pmu_intrStat) ==
                LW_PDISP_FE_PMU_INTR_STAT_OR_SOR_PENDING);
            gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_OR, _SOR(orNum), gsp_intrStat) ==
                LW_PDISP_FE_GSP_INTR_STAT_OR_SOR_PENDING);
            ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_OR, _SOR(orNum), intrEn) ==
                LW_PDISP_FE_RM_INTR_EN_OR_SOR_ENABLE);
            rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_OR, _SOR(orNum), rm_intrMask) ==
                LW_PDISP_FE_RM_INTR_MSK_OR_SOR_ENABLE);
            pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_OR, _SOR(orNum), pmu_intrMask) ==
                LW_PDISP_FE_PMU_INTR_MSK_OR_SOR_ENABLE);
            gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_OR, _SOR(orNum), gsp_intrMask) ==
                LW_PDISP_FE_GSP_INTR_MSK_OR_SOR_ENABLE);

            dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));

        }
        dprintf ("----------------------------------------------------------------------------------------------------------\n");



        if (all)
        {
            //OR_PIOR Interrupts

            for (orNum = 0; orNum < LW_PDISP_FE_RM_INTR_STAT_OR_PIOR__SIZE_1; ++orNum)
            {
                dprintf ("OR_PIOR_%d", orNum);
                dprintf ("%31s|", "");

                evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_OR, _PIOR(orNum), evtStat) ==
                    LW_PDISP_FE_EVT_STAT_OR_PIOR_PENDING);
                rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_OR, _PIOR(orNum), rm_intrStat) ==
                    LW_PDISP_FE_RM_INTR_STAT_OR_PIOR_PENDING);
                pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_OR, _PIOR(orNum), pmu_intrStat) ==
                    LW_PDISP_FE_PMU_INTR_STAT_OR_PIOR_PENDING);
                gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_OR, _PIOR(orNum), gsp_intrStat) ==
                    LW_PDISP_FE_GSP_INTR_STAT_OR_PIOR_PENDING);
                ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_OR, _PIOR(orNum), intrEn) ==
                    LW_PDISP_FE_RM_INTR_EN_OR_PIOR_ENABLE);
                rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_OR, _PIOR(orNum), rm_intrMask) ==
                    LW_PDISP_FE_RM_INTR_MSK_OR_PIOR_ENABLE);
                pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_OR, _PIOR(orNum), pmu_intrMask) ==
                    LW_PDISP_FE_PMU_INTR_MSK_OR_PIOR_ENABLE);
                gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_OR, _PIOR(orNum), gsp_intrMask) ==
                    LW_PDISP_FE_GSP_INTR_MSK_OR_PIOR_ENABLE);

                dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                    ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                    santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));

            }
            dprintf ("----------------------------------------------------------------------------------------------------------\n");
        }


        //Supervisor Interrupts

        evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_CTRL_DISP);

        rm_intrStat = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_STAT_CTRL_DISP);
        pmu_intrStat = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_STAT_CTRL_DISP);
        gsp_intrStat = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_STAT_CTRL_DISP);
        intrEn = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_EN_CTRL_DISP);
        rm_intrMask = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_MSK_CTRL_DISP);
        pmu_intrMask = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_MSK_CTRL_DISP);
        gsp_intrMask = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_MSK_CTRL_DISP);

        for (svNum = 0; svNum < LW_PDISP_FE_RM_INTR_STAT_CTRL_DISP_SUPERVISOR__SIZE_1; ++svNum)
        {
            dprintf ("SUPERVISOR_%d", svNum + 1);
            dprintf ("%28s|", "");

            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_CTRL_DISP, _SUPERVISOR(svNum), evtStat) ==
                LW_PDISP_FE_EVT_STAT_CTRL_DISP_SUPERVISOR_PENDING);
            rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_CTRL_DISP, _SUPERVISOR(svNum), rm_intrStat) ==
                LW_PDISP_FE_RM_INTR_STAT_CTRL_DISP_SUPERVISOR_PENDING);
            pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_CTRL_DISP, _SUPERVISOR(svNum), pmu_intrStat) ==
                LW_PDISP_FE_PMU_INTR_STAT_CTRL_DISP_SUPERVISOR_PENDING);
            gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_CTRL_DISP, _SUPERVISOR(svNum), gsp_intrStat) ==
                LW_PDISP_FE_GSP_INTR_STAT_CTRL_DISP_SUPERVISOR_PENDING);
            ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_CTRL_DISP, _SUPERVISOR(svNum), intrEn) ==
                LW_PDISP_FE_RM_INTR_EN_CTRL_DISP_SUPERVISOR_ENABLE);
            rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_CTRL_DISP, _SUPERVISOR(svNum), rm_intrMask) ==
                LW_PDISP_FE_RM_INTR_MSK_CTRL_DISP_SUPERVISOR_ENABLE);
            pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_CTRL_DISP, _SUPERVISOR(svNum), pmu_intrMask) ==
                LW_PDISP_FE_PMU_INTR_MSK_CTRL_DISP_SUPERVISOR_ENABLE);
            gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_CTRL_DISP, _SUPERVISOR(svNum), gsp_intrMask) ==
                LW_PDISP_FE_GSP_INTR_MSK_CTRL_DISP_SUPERVISOR_ENABLE);

            dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));

        }
        dprintf ("----------------------------------------------------------------------------------------------------------\n");



        headMax = (all ? LW_PDISP_FE_RM_INTR_STAT_HEAD_TIMING__SIZE_1 : 4);
        //Head Timing Interrupts

        for (head = 0; head < headMax; ++head)
        {
            evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_HEAD_TIMING(head));

            rm_intrStat = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_STAT_HEAD_TIMING(head));
            pmu_intrStat = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_STAT_HEAD_TIMING(head));
            gsp_intrStat = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_STAT_HEAD_TIMING(head));
            intrEn = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_EN_HEAD_TIMING(head));
            rm_intrMask = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_MSK_HEAD_TIMING(head));
            pmu_intrMask = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_MSK_HEAD_TIMING(head));
            gsp_intrMask = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_MSK_HEAD_TIMING(head));


            dprintf ("HEAD%d_TIMING_LOADV", head);
            dprintf ("%22s|", "");

            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _LOADV, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_LOADV_PENDING);
            rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_HEAD_TIMING, _LOADV, rm_intrStat) ==
                LW_PDISP_FE_RM_INTR_STAT_HEAD_TIMING_LOADV_PENDING);
            pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_HEAD_TIMING, _LOADV, pmu_intrStat) ==
                LW_PDISP_FE_PMU_INTR_STAT_HEAD_TIMING_LOADV_PENDING);
            gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_HEAD_TIMING, _LOADV, gsp_intrStat) ==
                LW_PDISP_FE_GSP_INTR_STAT_HEAD_TIMING_LOADV_PENDING);
            ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_HEAD_TIMING, _LOADV, intrEn) ==
                LW_PDISP_FE_RM_INTR_EN_HEAD_TIMING_LOADV_ENABLE);
            rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_HEAD_TIMING, _LOADV, rm_intrMask) ==
                LW_PDISP_FE_RM_INTR_MSK_HEAD_TIMING_LOADV_ENABLE);
            pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_HEAD_TIMING, _LOADV, pmu_intrMask) ==
                LW_PDISP_FE_PMU_INTR_MSK_HEAD_TIMING_LOADV_ENABLE);
            gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_HEAD_TIMING, _LOADV, gsp_intrMask) ==
                LW_PDISP_FE_GSP_INTR_MSK_HEAD_TIMING_LOADV_ENABLE);

            dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));


            dprintf ("HEAD%d_TIMING_LAST_DATA", head);
            dprintf ("%18s|", "");

            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _LAST_DATA, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_LAST_DATA_PENDING);
            rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_HEAD_TIMING, _LAST_DATA, rm_intrStat) ==
                LW_PDISP_FE_RM_INTR_STAT_HEAD_TIMING_LAST_DATA_PENDING);
            pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_HEAD_TIMING, _LAST_DATA, pmu_intrStat) ==
                LW_PDISP_FE_PMU_INTR_STAT_HEAD_TIMING_LAST_DATA_PENDING);
            gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_HEAD_TIMING, _LAST_DATA, gsp_intrStat) ==
                LW_PDISP_FE_GSP_INTR_STAT_HEAD_TIMING_LAST_DATA_PENDING);
            ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_HEAD_TIMING, _LAST_DATA, intrEn) ==
                LW_PDISP_FE_RM_INTR_EN_HEAD_TIMING_LAST_DATA_ENABLE);
            rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_HEAD_TIMING, _LAST_DATA, rm_intrMask) ==
                LW_PDISP_FE_RM_INTR_MSK_HEAD_TIMING_LAST_DATA_ENABLE);
            pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_HEAD_TIMING, _LAST_DATA, pmu_intrMask) ==
                LW_PDISP_FE_PMU_INTR_MSK_HEAD_TIMING_LAST_DATA_ENABLE);
            gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_HEAD_TIMING, _LAST_DATA, gsp_intrMask) ==
                LW_PDISP_FE_GSP_INTR_MSK_HEAD_TIMING_LAST_DATA_ENABLE);

            dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));


            dprintf ("HEAD%d_TIMING_VBLANK", head);
            dprintf ("%21s|", "");

            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _VBLANK, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_VBLANK_PENDING);
            rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_HEAD_TIMING, _VBLANK, rm_intrStat) ==
                LW_PDISP_FE_RM_INTR_STAT_HEAD_TIMING_VBLANK_PENDING);
            pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_HEAD_TIMING, _VBLANK, pmu_intrStat) ==
                LW_PDISP_FE_PMU_INTR_STAT_HEAD_TIMING_VBLANK_PENDING);
            gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_HEAD_TIMING, _VBLANK, gsp_intrStat) ==
                LW_PDISP_FE_GSP_INTR_STAT_HEAD_TIMING_VBLANK_PENDING);
            ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_HEAD_TIMING, _VBLANK, intrEn) ==
                LW_PDISP_FE_RM_INTR_EN_HEAD_TIMING_VBLANK_ENABLE);
            rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_HEAD_TIMING, _VBLANK, rm_intrMask) ==
                LW_PDISP_FE_RM_INTR_MSK_HEAD_TIMING_VBLANK_ENABLE);
            pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_HEAD_TIMING, _VBLANK, pmu_intrMask) ==
                LW_PDISP_FE_PMU_INTR_MSK_HEAD_TIMING_VBLANK_ENABLE);
            gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_HEAD_TIMING, _VBLANK, gsp_intrMask) ==
                LW_PDISP_FE_GSP_INTR_MSK_HEAD_TIMING_VBLANK_ENABLE);

            dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));


            dprintf ("HEAD%d_TIMING_VACTIVE_SPACE_VBLANK", head);
            dprintf ("%7s|", "");

            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _VACTIVE_SPACE_VBLANK, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_VACTIVE_SPACE_VBLANK_PENDING);
            rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_HEAD_TIMING, _VACTIVE_SPACE_VBLANK, rm_intrStat) ==
                LW_PDISP_FE_RM_INTR_STAT_HEAD_TIMING_VACTIVE_SPACE_VBLANK_PENDING);
            pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_HEAD_TIMING, _VACTIVE_SPACE_VBLANK, pmu_intrStat) ==
                LW_PDISP_FE_PMU_INTR_STAT_HEAD_TIMING_VACTIVE_SPACE_VBLANK_PENDING);
            gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_HEAD_TIMING, _VACTIVE_SPACE_VBLANK, gsp_intrStat) ==
                LW_PDISP_FE_GSP_INTR_STAT_HEAD_TIMING_VACTIVE_SPACE_VBLANK_PENDING);
            ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_HEAD_TIMING, _VACTIVE_SPACE_VBLANK, intrEn) ==
                LW_PDISP_FE_RM_INTR_EN_HEAD_TIMING_VACTIVE_SPACE_VBLANK_ENABLE);
            rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_HEAD_TIMING, _VACTIVE_SPACE_VBLANK, rm_intrMask) ==
                LW_PDISP_FE_RM_INTR_MSK_HEAD_TIMING_VACTIVE_SPACE_VBLANK_ENABLE);
            pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_HEAD_TIMING, _VACTIVE_SPACE_VBLANK, pmu_intrMask) ==
                LW_PDISP_FE_PMU_INTR_MSK_HEAD_TIMING_VACTIVE_SPACE_VBLANK_ENABLE);
            gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_HEAD_TIMING, _VACTIVE_SPACE_VBLANK, gsp_intrMask) ==
                LW_PDISP_FE_GSP_INTR_MSK_HEAD_TIMING_VACTIVE_SPACE_VBLANK_ENABLE);

            dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));


            dprintf ("HEAD%d_TIMING_RG_STALL", head);
            dprintf ("%19s|", "");

            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _RG_STALL, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_RG_STALL_PENDING);
            rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_HEAD_TIMING, _RG_STALL, rm_intrStat) ==
                LW_PDISP_FE_RM_INTR_STAT_HEAD_TIMING_RG_STALL_PENDING);
            pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_HEAD_TIMING, _RG_STALL, pmu_intrStat) ==
                LW_PDISP_FE_PMU_INTR_STAT_HEAD_TIMING_RG_STALL_PENDING);
            gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_HEAD_TIMING, _RG_STALL, gsp_intrStat) ==
                LW_PDISP_FE_GSP_INTR_STAT_HEAD_TIMING_RG_STALL_PENDING);
            ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_HEAD_TIMING, _RG_STALL, intrEn) ==
                LW_PDISP_FE_RM_INTR_EN_HEAD_TIMING_RG_STALL_ENABLE);
            rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_HEAD_TIMING, _RG_STALL, rm_intrMask) ==
                LW_PDISP_FE_RM_INTR_MSK_HEAD_TIMING_RG_STALL_ENABLE);
            pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_HEAD_TIMING, _RG_STALL, pmu_intrMask) ==
                LW_PDISP_FE_PMU_INTR_MSK_HEAD_TIMING_RG_STALL_ENABLE);
            gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_HEAD_TIMING, _RG_STALL, gsp_intrMask) ==
                LW_PDISP_FE_GSP_INTR_MSK_HEAD_TIMING_RG_STALL_ENABLE);

            dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));


            dprintf ("HEAD%d_TIMING_RG_LINE_A", head);
            dprintf ("%18s|", "");

            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _RG_LINE_A, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_RG_LINE_A_PENDING);
            rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_HEAD_TIMING, _RG_LINE_A, rm_intrStat) ==
                LW_PDISP_FE_RM_INTR_STAT_HEAD_TIMING_RG_LINE_A_PENDING);
            pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_HEAD_TIMING, _RG_LINE_A, pmu_intrStat) ==
                LW_PDISP_FE_PMU_INTR_STAT_HEAD_TIMING_RG_LINE_A_PENDING);
            gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_HEAD_TIMING, _RG_LINE_A, gsp_intrStat) ==
                LW_PDISP_FE_GSP_INTR_STAT_HEAD_TIMING_RG_LINE_A_PENDING);
            ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_HEAD_TIMING, _RG_LINE_A, intrEn) ==
                LW_PDISP_FE_RM_INTR_EN_HEAD_TIMING_RG_LINE_A_ENABLE);
            rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_HEAD_TIMING, _RG_LINE_A, rm_intrMask) ==
                LW_PDISP_FE_RM_INTR_MSK_HEAD_TIMING_RG_LINE_A_ENABLE);
            pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_HEAD_TIMING, _RG_LINE_A, pmu_intrMask) ==
                LW_PDISP_FE_PMU_INTR_MSK_HEAD_TIMING_RG_LINE_A_ENABLE);
            gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_HEAD_TIMING, _RG_LINE_A, gsp_intrMask) ==
                LW_PDISP_FE_GSP_INTR_MSK_HEAD_TIMING_RG_LINE_A_ENABLE);

            dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));


            dprintf ("HEAD%d_TIMING_RG_LINE_B", head);
            dprintf ("%18s|", "");

            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _RG_LINE_B, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_RG_LINE_B_PENDING);
            rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_HEAD_TIMING, _RG_LINE_B, rm_intrStat) ==
                LW_PDISP_FE_RM_INTR_STAT_HEAD_TIMING_RG_LINE_B_PENDING);
            pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_HEAD_TIMING, _RG_LINE_B, pmu_intrStat) ==
                LW_PDISP_FE_PMU_INTR_STAT_HEAD_TIMING_RG_LINE_B_PENDING);
            gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_HEAD_TIMING, _RG_LINE_B, gsp_intrStat) ==
                LW_PDISP_FE_GSP_INTR_STAT_HEAD_TIMING_RG_LINE_B_PENDING);
            ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_HEAD_TIMING, _RG_LINE_B, intrEn) ==
                LW_PDISP_FE_RM_INTR_EN_HEAD_TIMING_RG_LINE_B_ENABLE);
            rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_HEAD_TIMING, _RG_LINE_B, rm_intrMask) ==
                LW_PDISP_FE_RM_INTR_MSK_HEAD_TIMING_RG_LINE_B_ENABLE);
            pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_HEAD_TIMING, _RG_LINE_B, pmu_intrMask) ==
                LW_PDISP_FE_PMU_INTR_MSK_HEAD_TIMING_RG_LINE_B_ENABLE);
            gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_HEAD_TIMING, _RG_LINE_B, gsp_intrMask) ==
                LW_PDISP_FE_GSP_INTR_MSK_HEAD_TIMING_RG_LINE_B_ENABLE);

            dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));


            dprintf ("HEAD%d_TIMING_SD3_BUCKET_WALK_DONE", head);
            dprintf ("%7s|", "");

            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _SD3_BUCKET_WALK_DONE, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_SD3_BUCKET_WALK_DONE_PENDING);
            rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_HEAD_TIMING, _SD3_BUCKET_WALK_DONE, rm_intrStat) ==
                LW_PDISP_FE_RM_INTR_STAT_HEAD_TIMING_SD3_BUCKET_WALK_DONE_PENDING);
            pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_HEAD_TIMING, _SD3_BUCKET_WALK_DONE, pmu_intrStat) ==
                LW_PDISP_FE_PMU_INTR_STAT_HEAD_TIMING_SD3_BUCKET_WALK_DONE_PENDING);
            gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_HEAD_TIMING, _SD3_BUCKET_WALK_DONE, gsp_intrStat) ==
                LW_PDISP_FE_GSP_INTR_STAT_HEAD_TIMING_SD3_BUCKET_WALK_DONE_PENDING);
            ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_HEAD_TIMING, _SD3_BUCKET_WALK_DONE, intrEn) ==
                LW_PDISP_FE_RM_INTR_EN_HEAD_TIMING_SD3_BUCKET_WALK_DONE_ENABLE);
            rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_HEAD_TIMING, _SD3_BUCKET_WALK_DONE, rm_intrMask) ==
                LW_PDISP_FE_RM_INTR_MSK_HEAD_TIMING_SD3_BUCKET_WALK_DONE_ENABLE);
            pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_HEAD_TIMING, _SD3_BUCKET_WALK_DONE, pmu_intrMask) ==
                LW_PDISP_FE_PMU_INTR_MSK_HEAD_TIMING_SD3_BUCKET_WALK_DONE_ENABLE);
            gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_HEAD_TIMING, _SD3_BUCKET_WALK_DONE, gsp_intrMask) ==
                LW_PDISP_FE_GSP_INTR_MSK_HEAD_TIMING_SD3_BUCKET_WALK_DONE_ENABLE);

            dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));


            dprintf ("HEAD%d_TIMING_SEC_POLICY", head);
            dprintf ("%17s|", "");

            evtPen = (DRF_VAL(_PDISP, _FE_EVT_STAT_HEAD_TIMING, _SEC_POLICY, evtStat) ==
                LW_PDISP_FE_EVT_STAT_HEAD_TIMING_SEC_POLICY_PENDING);
            rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_HEAD_TIMING, _SEC_POLICY, rm_intrStat) ==
                LW_PDISP_FE_RM_INTR_STAT_HEAD_TIMING_SEC_POLICY_PENDING);
            pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_HEAD_TIMING, _SEC_POLICY, pmu_intrStat) ==
                LW_PDISP_FE_PMU_INTR_STAT_HEAD_TIMING_SEC_POLICY_PENDING);
            gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_HEAD_TIMING, _SEC_POLICY, gsp_intrStat) ==
                LW_PDISP_FE_GSP_INTR_STAT_HEAD_TIMING_SEC_POLICY_PENDING);
            ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_HEAD_TIMING, _SEC_POLICY, intrEn) ==
                LW_PDISP_FE_RM_INTR_EN_HEAD_TIMING_SEC_POLICY_ENABLE);
            rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_HEAD_TIMING, _SEC_POLICY, rm_intrMask) ==
                LW_PDISP_FE_RM_INTR_MSK_HEAD_TIMING_SEC_POLICY_ENABLE);
            pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_HEAD_TIMING, _SEC_POLICY, pmu_intrMask) ==
                LW_PDISP_FE_PMU_INTR_MSK_HEAD_TIMING_SEC_POLICY_ENABLE);
            gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_HEAD_TIMING, _SEC_POLICY, gsp_intrMask) ==
                LW_PDISP_FE_GSP_INTR_MSK_HEAD_TIMING_SEC_POLICY_ENABLE);

            dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));


            dprintf ("----------------------------------------------------------------------------------------------------------\n");
        }



        if (all)
        {
            //Head LWDPS Interrupts

            rm_intrStat = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_STAT_HEAD_LWDPS);
            pmu_intrStat = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_STAT_HEAD_LWDPS);
            gsp_intrStat = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_STAT_HEAD_LWDPS);
            intrEn = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_EN_HEAD_LWDPS);
            rm_intrMask = GPU_REG_RD32(LW_PDISP_FE_RM_INTR_MSK_HEAD_LWDPS);
            pmu_intrMask = GPU_REG_RD32(LW_PDISP_FE_PMU_INTR_MSK_HEAD_LWDPS);
            gsp_intrMask = GPU_REG_RD32(LW_PDISP_FE_GSP_INTR_MSK_HEAD_LWDPS);

            for (head=0; head < LW_PDISP_FE_RM_INTR_STAT_HEAD_LWDPS_HEAD__SIZE_1; ++head)
            {
                dprintf ("HEAD_LWDPS_HEAD%d", head);
                dprintf ("%24s|", "");

                evtStat = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_HEAD_LWDPS(head));
                evtPen = (evtStat!=0);
                rm_intrPen = (DRF_VAL(_PDISP, _FE_RM_INTR_STAT_HEAD_LWDPS, _HEAD(head), rm_intrStat) ==
                    LW_PDISP_FE_RM_INTR_STAT_HEAD_LWDPS_HEAD_PENDING);
                pmu_intrPen = (DRF_VAL(_PDISP, _FE_PMU_INTR_STAT_HEAD_LWDPS, _HEAD(head), pmu_intrStat) ==
                    LW_PDISP_FE_PMU_INTR_STAT_HEAD_LWDPS_HEAD_PENDING);
                gsp_intrPen = (DRF_VAL(_PDISP, _FE_GSP_INTR_STAT_HEAD_LWDPS, _HEAD(head), gsp_intrStat) ==
                    LW_PDISP_FE_GSP_INTR_STAT_HEAD_LWDPS_HEAD_PENDING);
                ie = (DRF_VAL(_PDISP, _FE_RM_INTR_EN_HEAD_LWDPS, _HEAD(head), intrEn) ==
                    LW_PDISP_FE_RM_INTR_EN_HEAD_LWDPS_HEAD_ENABLE);
                rm_im = (DRF_VAL(_PDISP, _FE_RM_INTR_MSK_HEAD_LWDPS, _HEAD(head), rm_intrMask) ==
                    LW_PDISP_FE_RM_INTR_MSK_HEAD_LWDPS_HEAD_ENABLE);
                pmu_im = (DRF_VAL(_PDISP, _FE_PMU_INTR_MSK_HEAD_LWDPS, _HEAD(head), pmu_intrMask) ==
                    LW_PDISP_FE_PMU_INTR_MSK_HEAD_LWDPS_HEAD_ENABLE);
                gsp_im = (DRF_VAL(_PDISP, _FE_GSP_INTR_MSK_HEAD_LWDPS, _HEAD(head), gsp_intrMask) ==
                    LW_PDISP_FE_GSP_INTR_MSK_HEAD_LWDPS_HEAD_ENABLE);

                dprintf ("%-3s|%-3s|%-3s|%-8s|%-3s|%-3s|%-3s|%-32s\n",
                    ynfunc(rm_intrPen), ynfunc(pmu_intrPen), ynfunc(gsp_intrPen), ynfunc(ie), ynfunc(rm_im), ynfunc(pmu_im), ynfunc(gsp_im),
                    santest(evtPen, rm_intrPen, pmu_intrPen, gsp_intrPen, ie, rm_im, pmu_im, gsp_im));

            }
            dprintf ("----------------------------------------------------------------------------------------------------------\n");

        }
        dprintf ("\n");
    }

}

// Print the SOR - Padlink connections.
void dispOrPadlinkConnection_v03_00(void)
{
    LwU32       data32, sor;
    PADLINK     link, sorStatus[LW_PDISP_FE_SW_SYS_CAP_SOR_EXISTS__SIZE_1][LW_MAX_SUBLINK];
    SOR_SUBLINK sub;

    // Initialization.
    for (sor = 0; sor < LW_PDISP_FE_SW_SYS_CAP_SOR_EXISTS__SIZE_1; sor++)
    {
        for (sub = PRIMARY; sub < LW_MAX_SUBLINK; sub++)
        {
            sorStatus[sor][sub] = PADLINK_NONE;
        }

    }

    // Get the SOR - Padlink connections.
    for (link = PADLINK_A; link < LW_PDISP_FE_CMGR_CLK_LINK_CTRL__SIZE_1; link++)
    {
        data32 = GPU_REG_RD32(LW_PDISP_FE_CMGR_CLK_LINK_CTRL(link));
        if (DRF_VAL(_PDISP, _FE_CMGR_CLK_LINK_CTRL, _FRONTEND, data32) != LW_PDISP_FE_CMGR_CLK_LINK_CTRL_FRONTEND_NONE)
        {
            switch(DRF_VAL(_PDISP, _FE_CMGR_CLK_LINK_CTRL, _FRONTEND, data32))
            {
                case LW_PDISP_FE_CMGR_CLK_LINK_CTRL_FRONTEND_SOR0:
                    sor = 0;
                    break;
                case LW_PDISP_FE_CMGR_CLK_LINK_CTRL_FRONTEND_SOR1:
                    sor = 1;
                    break;
                case LW_PDISP_FE_CMGR_CLK_LINK_CTRL_FRONTEND_SOR2:
                    sor = 2;
                    break;
                case LW_PDISP_FE_CMGR_CLK_LINK_CTRL_FRONTEND_SOR3:
                    sor = 3;
                    break;
                case LW_PDISP_FE_CMGR_CLK_LINK_CTRL_FRONTEND_SOR4:
                    sor = 4;
                    break;
                case LW_PDISP_FE_CMGR_CLK_LINK_CTRL_FRONTEND_SOR5:
                    sor = 5;
                    break;
                case LW_PDISP_FE_CMGR_CLK_LINK_CTRL_FRONTEND_SOR6:
                    sor = 6;
                    break;
                case LW_PDISP_FE_CMGR_CLK_LINK_CTRL_FRONTEND_SOR7:
                    sor = 7;
                    break;
            }
            switch(DRF_VAL(_PDISP, _FE_CMGR_CLK_LINK_CTRL, _FRONTEND_SOR, data32))
            {
                case LW_PDISP_FE_CMGR_CLK_LINK_CTRL_FRONTEND_SOR_PRIMARY:
                    sorStatus[sor][PRIMARY] = link;
                    break;
                case LW_PDISP_FE_CMGR_CLK_LINK_CTRL_FRONTEND_SOR_SECONDARY:
                    sorStatus[sor][SECONDARY] = link;
                    break;
            }
        }
    }

    // Print out the SOR - Padlink connections.
    dprintf("------------------------------------------------------------------\n");
    dprintf(" SorIndex     |     Exists    |    Routed to(Primary,  Secondary)\n");
    dprintf("------------------------------------------------------------------\n");
    for (sor = 0; sor < LW_PDISP_FE_SW_SYS_CAP_SOR_EXISTS__SIZE_1; sor++)
    {
        dprintf("   SOR%d", sor);
        data32 = GPU_REG_RD32(LW_PDISP_FE_SW_SYS_CAP);
        if (FLD_IDX_TEST_DRF(_PDISP, _FE_SW_SYS_CAP_SOR, _EXISTS, sor, _YES, data32))
            dprintf("       |      Yes      |     ");
        else
            dprintf("       |      No       |     ");

        for (sub = PRIMARY; sub < LW_MAX_SUBLINK; sub++)
        {
            dprintf("%s,  ", dispGetPadLinkString(sorStatus[sor][sub]));
        }
        dprintf("\n\n");
    }
}

LwS32 dispGetDebugMode_v03_00
(
    LwU32 chanNum
)
{
    ChanDesc_t_Lwdisplay *chnst;
    LwU32       val;
    LwU32       numDispChannels = pDisp[indexGpu].dispGetMaxChan();

    if (chanNum >= numDispChannels)
    {
        dprintf("chanNum should be less than %d\n", numDispChannels);
        return -1;
    }

    chnst=&dispChanState_v03_00[chanNum];
    if (! (chnst->cap & DISP_DEBUG) )
    {
        return -2;
    }
    val = GPU_REG_RD32(LW_PDISP_FE_DEBUG_CTL(chanNum));
    return (FLD_TEST_DRF(_PDISP, _FE_DEBUG_CTL, _MODE, _ENABLE, val)) ? TRUE: FALSE;
}

void dispSetDebugMode_v03_00(LwU32 chanNum, BOOL set)
{
    ChanDesc_t_Lwdisplay *chnst;
    LwU32       val;
    LwU32       numDispChannels = pDisp[indexGpu].dispGetMaxChan();

    if (chanNum >= numDispChannels)
    {
        dprintf("chanNum should be less than %d\n", numDispChannels);
        return;
    }

    // Check for cap.
    chnst = &dispChanState_v03_00[chanNum];

    if (! (chnst->cap & DISP_DEBUG) )
    {
        dprintf("DebugMode is not available for channel %d\n", chanNum);
        return ;
    }

    val = GPU_REG_RD32(LW_PDISP_FE_DEBUG_CTL(chanNum));
    if (set != FLD_TEST_DRF(_PDISP, _FE_DEBUG_CTL, _MODE, _ENABLE, val)) {
        if (set)
            val = FLD_SET_DRF(_PDISP, _FE_DEBUG_CTL, _MODE, _ENABLE, val);
        else
            val = FLD_SET_DRF(_PDISP, _FE_DEBUG_CTL, _MODE, _DISABLE, val);

        GPU_REG_WR32(LW_PDISP_FE_DEBUG_CTL(chanNum), val);
        val = GPU_REG_RD32(LW_PDISP_FE_DEBUG_CTL(chanNum));
        if (set != FLD_TEST_DRF(_PDISP, _FE_DEBUG_CTL, _MODE, _ENABLE, val))
            dprintf("Failed to set the debug mode..(0x%08x)\n", val);
    }
}

void dispDumpGetDebugMode_v03_00(char *chName, LwS32 headNum, LwU32 dArgc)
{
    ChanDesc_t_Lwdisplay *desc;
    LwS32 chNum = 0;
    LwS32 ret;

    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        dprintf("Ch#  Name  Head#    DBG MODE\n");
        dprintf("-----------------------------\n");
        if ( dArgc > 0 )
        {
            if ( (chNum = pDisp[indexGpu].dispGetChanNum(chName, headNum)) != -1)
            {
                int i, numheads = 1;

                if (pDisp[indexGpu].dispGetChanDescriptor(chNum, (void**)&desc))
                {
                    dprintf("error in dispGetChanDescriptor\n");
                    return;
                }
                if (!(desc->cap & DISP_DEBUG))
                {
                    dprintf("DebugMode is not available for channel %d\n", chNum);
                    return;
                }

                // if dArgc == 1, print all heads
                if (dArgc == 1)
                {
                    numheads = desc->numHeads;
                }

                for (i = 0; i < numheads ; i++)
                {
                    ret = pDisp[indexGpu].dispGetDebugMode(chNum + i);
                    dprintf("#%2d %5s %5d %8s\n",chNum + i,  chName, i, ret ? "ENABLED":"DISABLED");
                }
            }
            else
                dprintf("lw: Usage: !lw.dgetdbgmode [chName] [-h<hd>]\n");
        }
        else
        {
            LwU32 i, k;
            k = pDisp[indexGpu].dispGetMaxChan();
            for (i = 0; i < k; i++)
            {
                pDisp[indexGpu].dispGetChanDescriptor(i, (void**)&desc);
                if (!(desc->cap & DISP_DEBUG))
                    continue;
                ret = pDisp[indexGpu].dispGetDebugMode(i);
                if (ret >= 0)
                    dprintf("#%2d %5s %5d %8s\n",i,  desc->name, desc->headNum, ret ? "ENABLED":"DISABLED");
            }
        }
    }
    MGPU_LOOP_END;
}

void dispDumpSetDebugMode_v03_00(char *chName, LwS32 headNum, LwU32 debugMode)
{
    LwS32 chNum = 0;

    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        if ((chNum = pDisp[indexGpu].dispGetChanNum(chName, headNum)) != -1)
            if (debugMode != pDisp[indexGpu].dispGetDebugMode(chNum))
                pDisp[indexGpu].dispSetDebugMode(chNum,  debugMode);
    }
    MGPU_LOOP_END;
}

void dispAnalyzeHang_v03_00(void)
{
    LwU32 data32 = 0, head=0;
    LwU32 numOfHeads = 0;
    numOfHeads = pDisp[indexGpu].dispGetNumHeads();

    dprintf("===========Display Analyze Hang Start===========\n");

    for(head = 0; head < numOfHeads; head++)
    {
        dispAnalyzeCoreUpdSm_v03_00(head);
    }

    dprintf("===========Analyze Display Hang End=============\n");

    return;
}

//
// fills given buffer according to the target memory type and offset specified
//
static void getPbData_v03_00
(
    PhysAddr physOffset,
    LwU32 numDwords,
    char *buffer,
    LwU32 mem_Target
)
{
    LwU32 i;
    LwU32 * ptr;

    if ( mem_Target == LW_PDISP_FE_PBBASE_PUSHBUFFER_TARGET_PHYS_LWM)
    {
        if (buffer)
        {
            if (pFb[indexGpu].fbRead(physOffset, buffer, (LwU32)(numDwords*4)) == LW_ERR_GENERIC)
            {
                dprintf( "lw: ERROR READING VIDEO MEMORY\n");
            }
        }
    }
    else
    {
        if (buffer)
        {
            ptr = (LwU32 *) buffer;
            for (i = 0; i < numDwords; i++)
            {
                *ptr = SYSMEM_RD32(physOffset + (i * 4));
                ptr ++;
            }
        }
    }
}

//
// Prints the dump Push Buffer
//
LW_STATUS  dispDumpPB_v03_00
(
    LwU32 chanNum,
    LwS32 headNum,
    LwS32 numDwords,
    LwS32 OffsetDwords,
    LwU32 printParsed
)
{
    LW_STATUS   status = 0, data32 = 0;
    PhysAddr    physOffset = 0;
    LwU32       flagAllocated = 0, flagConnected = 0;
    LwU32       memTarget;
    LwU32       chnCtlOffset;
    PBCTLOFFSET pbCtlOffset;
    LwS32       channelClass = pDisp[indexGpu].dispGetChanType(chanNum, NULL);
    char*       buffer = NULL;
    LwU32       classNum;
    LwU32       getoffset = 0, putoffset = 0;

    status = pDisp[indexGpu].dispGetChnAndPbCtlRegOffsets(headNum,
                                                          chanNum,
                                                          channelClass,
                                                          &chnCtlOffset,
                                                          &pbCtlOffset);
    if (status == LW_ERR_GENERIC)
    {
            return status;
    }

    data32 = GPU_REG_RD32(chnCtlOffset);
    switch (channelClass)
    {
        case LWDISPLAY_CHNTYPE_CORE:
            flagAllocated = DRF_VAL(_PDISP, _FE_CHNCTL, _CORE_ALLOCATION, data32);
            flagConnected = DRF_VAL(_PDISP, _FE_CHNCTL, _CORE_CONNECTION, data32);

            dprintf("lw: LW_PDISP_FE_CHNCTL_CORE_ALLOCATION\t      %s\n",
                    flagAllocated ? "ALLOCATE" : "DEALLOCATE");

            dprintf("lw: LW_PDISP_FE_CHNCTL_CORE_CONNECT\t      %s\n",
                    flagConnected ? "CONNECT" : "DISCONNECT");
        break;

        case LWDISPLAY_CHNTYPE_WIN:
            flagAllocated = DRF_VAL(_PDISP, _FE_CHNCTL, _WIN_ALLOCATION, data32);
            flagConnected = DRF_VAL(_PDISP, _FE_CHNCTL, _WIN_CONNECTION, data32);

            dprintf("lw: LW_PDISP_FE_CHNCTL_WIN_ALLOCATION\t      %s\n",
                    flagAllocated ? "ALLOCATE" : "DEALLOCATE");

            dprintf("lw: LW_PDISP_FE_CHNCTL_WIN_CONNECT\t      %s\n",
                    flagConnected ? "CONNECT" : "DISCONNECT");
        break;

        case LWDISPLAY_CHNTYPE_WINIM:
            flagAllocated = DRF_VAL(_PDISP, _FE_CHNCTL, _WINIM_ALLOCATION, data32);
            flagConnected = DRF_VAL(_PDISP, _FE_CHNCTL, _WINIM_CONNECTION, data32);

            dprintf("lw: LW_PDISP_FE_CHNCTL_WINIM_ALLOCATION\t      %s\n",
                    flagAllocated ? "ALLOCATE" : "DEALLOCATE");

            dprintf("lw: LW_PDISP_FE_CHNCTL_WINIM_CONNECT\t      %s\n",
                    flagConnected ? "CONNECT" : "DISCONNECT");
        break;

        default:
            dprintf("Error : Channel is not DMA channel. Use core, win or winim\n\n");
    }

    //
    // if channel is not allocated or connected then return
    //
    if (!(flagConnected && flagAllocated))
    {
        dprintf("lw: Channel is not connected to any PB. Skipping the dump of push buffer.\n\n");
        return status;
    }

    //
    // Read the PB physical address by shifting 4 bit right and
    // then shift 12 bit left to get 40 bit physical address
    //

    data32 = GPU_REG_RD32(pbCtlOffset.PbCtlOffset[3]);

    physOffset = ((PhysAddr)(DRF_VAL(_PDISP, _FE_PBBASEHI, _PUSHBUFFER_ADDR, data32))) << 
                  (PHYSICAL_ADDR - DRF_SIZE(LW_PDISP_FE_PBBASEHI_PUSHBUFFER_ADDR));

    data32 = GPU_REG_RD32(pbCtlOffset.PbCtlOffset[0]);

    memTarget = DRF_VAL(_PDISP, _FE_PBBASE, _PUSHBUFFER_TARGET, data32);

    physOffset |= ((PhysAddr)(DRF_VAL(_PDISP, _FE_PBBASE, _PUSHBUFFER_ADDR, data32))) <<
                   (PHYSICAL_ADDR - DRF_SIZE(LW_PDISP_FE_PBBASE_PUSHBUFFER_ADDR) - 
                   DRF_SIZE(LW_PDISP_FE_PBBASEHI_PUSHBUFFER_ADDR));

    buffer = (char *)malloc((size_t)(DISP_PUSH_BUFFER_SIZE));
    if(!buffer)
    {
        return LW_ERR_GENERIC;
    }
    getPbData_v03_00(physOffset, (DISP_PUSH_BUFFER_SIZE / 4), buffer, memTarget);

    classNum = classHeaderNumLwdisplay[channelClass];

    dprintf( "lw: LW_PDISP_FE_PBBASE_PUSHBUFFER_TARGET(%d) :", chanNum);
    dprintf( (memTarget == LW_PDISP_FE_PBBASE_PUSHBUFFER_TARGET_PHYS_LWM) ? "VIDEO MEMORY\n" : "SYSTEM MEMORY\n");
    dprintf( "lw: LW_PDISP_FE_PBBASE(%d) \t\t            0x%x\n\n", chanNum, data32);

    putoffset = GPU_REG_RD32(LW_UDISP_FE_PUT(chanNum));
    getoffset = GPU_REG_RD32(LW_UDISP_FE_GET(chanNum));

    // printing out get and put pointers
    dprintf("lw: GET POINTER OFFSET: 0x%08x\n",getoffset);
    dprintf("lw: PUT POINTER OFFSET: 0x%08x\n",putoffset);

    if (!printParsed)
    {
        printBuffer((buffer + (OffsetDwords * 4)), (LwU32)(numDwords * 4), (physOffset + (OffsetDwords * 4)), 4);
    }
    else
    {
        printDispPbParsed_v03_00((LwU32)physOffset, OffsetDwords, (LwU32 *)buffer, (LwU32)numDwords, classNum, getoffset, putoffset);
    }

    free(buffer);
    return status;
}

//
// Parses and prints out the Display Push Buffer
//
static void printDispPbParsed_v03_00
(
    PhysAddr baseAddr,
    LwU32 OffsetDwords,
    LwU32 *buffer,
    LwU32 numDwords,
    LwU32 classNum,
    LwU32 getoffset,
    LwU32 putoffset)
{
    LwU32 i;
    LwU32 pbOffset, subdeviceMaskValue;
    PhysAddr addr, get, put;
    LwU32 * bufferstart;

    // 2 dws at least
    if (numDwords < 2)
    {
        numDwords = 8;
    }

    if (buffer == NULL)
    {
        dprintf("lw: Push buffer empty!\n");
        return;
    }

    if (!isValidClassHeader(classNum))
    {
        dprintf("lw: WARNING - Class Header file does not exist for \""
                CLASS_PATH_LOCAL "\"\n", classNum);
        goto EXIT;
    }

    addr = baseAddr + OffsetDwords *4;
    bufferstart = buffer + OffsetDwords;
    get = baseAddr + getoffset;
    put = baseAddr + putoffset;

    // Traverse the buffer by DWORDS
    for (i = 0; i < numDwords; i++)
    {
        LwU32 lwrMethHdr = bufferstart[i];
        LwU32 methCount, methAddr;
        LwU32 opcode;

        dprintf("\n" LwU40_FMT ": %08x\t", addr, lwrMethHdr);

        // The Get and Put pointers could be on a header
        if (addr == put || addr == get)
        {
            dprintf("lw: METHOD HEADER ADDR: " LwU40_FMT, addr);
            if (addr == get) dprintf(" <- _DISP_DMA_GET");
            if (addr == put)
            {
                dprintf(" <- _DISP_DMA_PUT\n");
                dprintf("lw: Parsing ends here\n");
                goto EXIT;
            }
            dprintf("\n");
        }

        // Increment current offset
        addr += 4;

        opcode  = DRF_VAL(_UDISP, _DMA, _OPCODE, lwrMethHdr);

        // JUMP is always to offset 0
        if (opcode == LW_UDISP_DMA_OPCODE_JUMP)
        {
            pbOffset = DRF_VAL(_UDISP, _DMA, _JUMP_OFFSET, lwrMethHdr);
            dprintf("lw: LW_UDISP_DMA_OPCODE_JUMP: OFFSET = 0x%08x\n", pbOffset * 4);
            printDispPbParsed_v03_00(baseAddr, pbOffset / 4, buffer, numDwords - (i+1), classNum, getoffset, putoffset);
            goto EXIT;
        }

        // SET SUBDEVICE MASK
        else if (opcode == LW_UDISP_DMA_OPCODE_SET_SUBDEVICE_MASK)
        {
            subdeviceMaskValue = DRF_VAL(_UDISP, _DMA, _SET_SUBDEVICE_MASK_VALUE, lwrMethHdr);
            dprintf("lw: LW_UDISP_SET_SUBDEVICE_MASK_VALUE: SUBDEVICE MASK VALUE = %x\n", subdeviceMaskValue);
        }

        // Method Header
        else if ((opcode == LW_UDISP_DMA_OPCODE_NONINC_METHOD ||
                  opcode == LW_UDISP_DMA_OPCODE_METHOD))
        {
            methCount = DRF_VAL(_UDISP, _DMA, _METHOD_COUNT, lwrMethHdr);
            methAddr = (DRF_VAL(_UDISP, _DMA, _METHOD_OFFSET, lwrMethHdr) * 4) & 0x1FFC;

            // When the methCount is zero, this is effectively a NOP
            if (!methCount)
            {
                dprintf("lw: 0x0000: NO_OPERATION\n");
            }

            // Parse each method to a data value (multiple data values for one header).
            for (methCount += i; (i+1) <= methCount && (i+1) < numDwords; i++)
            {
                LwU32 data = bufferstart[i + 1];

                dprintf("%08x:", data);

                if (!parseClassHeader(classNum, methAddr, data))
                {
                    dprintf("lw: 0x%04x: DATA: 0x%08x ", methAddr, data);
                }

                if (addr == get) dprintf(" <- _DISP_DMA_GET");
                if (addr == put)
                {
                    dprintf(" <- _DISP_DMA_PUT\n");
                    dprintf("lw: Parsing ends here\n");
                    goto EXIT;
                }

                // Increment current offset
                addr += 4;

                dprintf("\n");

                // Incrementing method
                if (opcode == LW_UDISP_DMA_OPCODE_METHOD)
                {
                    methAddr += 4;
                }
            }
        }
        else
        {
            // need to print out address also here!
            dprintf("lw: INVALID METHOD HEADER: 0x%08x at " LwU40_FMT "\n", lwrMethHdr, addr);
            goto EXIT;
        }
    }

EXIT:
    dprintf("\n");
}

LW_STATUS dispGetChnAndPbCtlRegOffsets_v03_00
(
    LwU32        headNum,
    LwU32        channelNum,
    LwU32        channelClass,
    LwU32        *pChnCtl,
    PBCTLOFFSET  *pPbCtl
)
{
    if ((pChnCtl == NULL) || (pPbCtl == NULL))
    {
        return LW_ERR_GENERIC;
    }

    if (pChnCtl)
    {
        switch (channelClass)
        {
            case CHNTYPE_CORE:
                *pChnCtl = LW_PDISP_FE_CHNCTL_CORE;
            break;

            case LWDISPLAY_CHNTYPE_WIN:
                *pChnCtl = LW_PDISP_FE_CHNCTL_WIN(headNum);
            break;

            case LWDISPLAY_CHNTYPE_WINIM:
                *pChnCtl = LW_PDISP_FE_CHNCTL_WINIM(headNum);
            break;

            default:
                dprintf("lw : Illegal channel type. "
                        "Use core, win or winim. Aborting\n");
            return LW_ERR_GENERIC;
        }
    }

    if (pPbCtl)
    {
        if (channelNum >= LW_PDISP_FE_PBBASE__SIZE_1)
        {
            return LW_ERR_GENERIC;
        }

        pPbCtl->PbCtlOffset[0] = LW_PDISP_FE_PBBASE(channelNum);
        pPbCtl->PbCtlOffset[1] = LW_PDISP_FE_PBSUBDEV(channelNum);
        pPbCtl->PbCtlOffset[2] = LW_PDISP_FE_PBCLIENT(channelNum);
        pPbCtl->PbCtlOffset[3] = LW_PDISP_FE_PBBASEHI(channelNum);
    }

    return LW_OK;
}

#define TIMEOUT (100) // 100ms. should be more than enough
//
// Inject Method
//
LwS32 dispInjectMethod_v03_00(LwU32 chanNum, LwU32 offset, LwU32 data)
{
    LwU32 val;
    int tout = 0;

    if (chanNum >= LWDISPLAY_DISP_NUM_DMA_CHANNELS) {
        dprintf("%s: chanNum must be smaller than %d\n", __FUNCTION__, LWDISPLAY_DISP_NUM_DMA_CHANNELS);
        return -1;
    }

    offset >>= 2;

    if(chanNum == 0)
    {
        val = GPU_REG_RD32(LW_PDISP_FE_CHNSTATUS_CORE);
        if(!FLD_TEST_DRF(_PDISP, _FE_CHNSTATUS, _CORE_STATE, _IDLE, val))
        {
            dprintf("Warning: There are pending methods in CORE channel push buffer\n");
        }
    }
    else if(chanNum < WINIM_PUSHBUFFER_OFFSET)
    {
        val = GPU_REG_RD32(LW_PDISP_FE_CHNSTATUS_WIN(chanNum));
        if(!FLD_TEST_DRF(_PDISP, _FE_CHNSTATUS, _WIN_STATE, _IDLE, val))
        {
            dprintf("Warning: There are pending methods in WIN channel push buffer\n");
        }
    }
    else if(chanNum < (WINIM_PUSHBUFFER_OFFSET + LW_PDISP_FE_CHNCTL_WINIM__SIZE_1))
    {
        val = GPU_REG_RD32(LW_PDISP_FE_CHNSTATUS_WINIM(chanNum));
        if(!FLD_TEST_DRF(_PDISP, _FE_CHNSTATUS, _WINIM_STATE, _IDLE, val))
        {
            dprintf("Warning: There are pending methods in WINIM channel push buffer\n");
        }
    }

    // Write data first
    GPU_REG_WR32(LW_PDISP_FE_DEBUG_DAT(chanNum), data);

    // Trigger
    val = GPU_REG_RD32(LW_PDISP_FE_DEBUG_CTL(chanNum));

    val = FLD_SET_DRF_NUM(_PDISP, _FE_DEBUG_CTL, _METHOD_OFS, offset, val);
    val = FLD_SET_DRF(_PDISP, _FE_DEBUG_CTL, _CTXDMA, _NORMAL, val);
    val = FLD_SET_DRF(_PDISP, _FE_DEBUG_CTL, _NEW_METHOD, _TRIGGER, val);

    GPU_REG_WR32(LW_PDISP_FE_DEBUG_CTL(chanNum), val);

    if (dispDispOwner_v02_01()) 
    {
        dprintf("Display owner is not DRIVER, bail out\n");
        return -1;
    }

    while(!FLD_TEST_DRF(_PDISP, _FE_DEBUG_CTL, _NEW_METHOD, _DONE, GPU_REG_RD32(LW_PDISP_FE_DEBUG_CTL(chanNum)))) 
    {
        if (++tout == TIMEOUT) 
        {
            dprintf("Method %x timedout for channel %d\n", offset, chanNum);
            if(chanNum == 0)
            {
                val = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_EXC_OTHER);
                if(FLD_TEST_DRF(_PDISP, _FE_EVT_STAT_EXC_OTHER, _CORE, _PENDING, val))
                {
                    dprintf("CORE Channel exception oclwred. Details below:\n");
                    val = GPU_REG_RD32(LW_PDISP_FE_EXCEPT(0));
                    dprintf("Method Offset: 0x%x\n", DRF_VAL(_PDISP, _FE_EXCEPT, _METHOD_OFFSET, val));
                    printExceptionReason(DRF_VAL(_PDISP, _FE_EXCEPT, _REASON, val));
                    val = GPU_REG_RD32(LW_PDISP_FE_EXCEPTARG(0));
                    dprintf("Except method arguments: 0x%x\n", val);
                    val = GPU_REG_RD32(LW_PDISP_FE_EXCEPTERR(0));
                    dprintf("Except Error Code: 0x%x\n", val);
                }
                else
                {
                    dispPrintChnFeState_v03_00(chanNum);
                }
            }
            else if (chanNum < WINIM_PUSHBUFFER_OFFSET)
            {
                val = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_EXC_WIN);
                if(val & (1 << (chanNum - 1)))
                {
                    dprintf("WIN Channel exception oclwred. Details below:\n");
                    val = GPU_REG_RD32(LW_PDISP_FE_EXCEPT(chanNum));
                    dprintf("Method Offset: 0x%x\n", DRF_VAL(_PDISP, _FE_EXCEPT, _METHOD_OFFSET, val));
                    printExceptionReason(DRF_VAL(_PDISP, _FE_EXCEPT, _REASON, val));
                    val = GPU_REG_RD32(LW_PDISP_FE_EXCEPTARG(chanNum));
                    dprintf("Except method arguments: 0x%x\n", val);
                    val = GPU_REG_RD32(LW_PDISP_FE_EXCEPTERR(chanNum));
                    dprintf("Except Error Code: 0x%x\n", val);
                }
                else
                {
                    dispPrintChnFeState_v03_00(chanNum);
                }
            }
            else if (chanNum < (WINIM_PUSHBUFFER_OFFSET + LW_PDISP_FE_CHNCTL_WINIM__SIZE_1))
            {
                val = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_EXC_WINIM);
                if(val & (1 << (chanNum - WINIM_PUSHBUFFER_OFFSET)))
                {
                    dprintf("WINIM Channel exception oclwred. Details below:\n");
                    val = GPU_REG_RD32(LW_PDISP_FE_EXCEPT(chanNum));
                    dprintf("Method Offset: 0x%x\n", DRF_VAL(_PDISP, _FE_EXCEPT, _METHOD_OFFSET, val));
                    printExceptionReason(DRF_VAL(_PDISP, _FE_EXCEPT, _REASON, val));
                }
                else
                {
                    dispPrintChnFeState_v03_00(chanNum);
                }
            }
            return -1;
        }
        // essentially we need some delay..
        osPerfDelay(1000);
    }
    return 0;
}

static void printExceptionReason(int reason)
{ 
    dprintf("Except Reason: ");
    switch(reason)
    {
        case LW_PDISP_FE_EXCEPT_REASON_PUSHBUFFER_ERR:
            dprintf("Pushbuffer error");
            break;
        case LW_PDISP_FE_EXCEPT_REASON_TRAP:
            dprintf("Trap");
            break;
        case LW_PDISP_FE_EXCEPT_REASON_RESERVED_METHOD:
            dprintf("Reserved Method");
            break;
        case LW_PDISP_FE_EXCEPT_REASON_ILWALID_ARG:
            dprintf("Invalid argument");
            break;
        case LW_PDISP_FE_EXCEPT_REASON_ILWALID_STATE:
            dprintf("Invalid state");
            break;
        case LW_PDISP_FE_EXCEPT_REASON_UNRESOLVABLE_HANDLE:
            dprintf("Unresolvable handle");
            break;
        case LW_PDISP_FE_EXCEPT_REASON_NONE:
            dprintf("No exception");
            break;
        default:
            dprintf("Invalid data");
    }
    dprintf("\n");
}

// Helper
static int getIdxMatch(int chanNum, char *name)
{
    int i;
    for (i = 0 ; i < mthd_num[chanNum] ; i++) 
    {
        if (!strcmp(name,mthd[chanNum][i].name))
        {
            return i;
        }
    }
    return -1;
}
static int getIdxFldMatch(int chanNum, char *name, int i)
{
    int fi;
    for (fi = 0 ; fi < mthd[chanNum][i].num_fld ; fi++) 
    {
        if (!strcmp(name,mthd[chanNum][i].fld[fi].name))
        {
            return fi;
        }
    }
    return -1;
}

#define IDX_HEAD (0x01)
#define IDX_SUB  (0x02)
#define IDX_MTD  (0x04)

#define PRINT_MTHD(cn,ni,i) \
    do { \
    dprintf("%s", mthd[cn][i].name);\
    if (ni & IDX_SUB)\
        dprintf("@%d[%s]", mthd[cn][i].sub->idx, mthd[cn][i].sub->name);\
    if (ni & IDX_MTD)\
        dprintf("@%d", mthd[cn][i].idx);\
    dprintf(" (offset: 0x%x)",  mthd[cn][i].base);\
    if (ni & IDX_HEAD)\
        dprintf(" [MULTI-HEADS]");\
    dprintf("\n");\
    } while (0)

#define PRINT_FLD(cn,ni,i,fi) \
    do { \
    dprintf("%s.%s", mthd[cn][i].name, mthd[cn][i].fld[fi].name);\
    if (ni & IDX_SUB)\
        dprintf("@%d[%s]", mthd[cn][i].sub->idx, mthd[cn][i].sub->name);\
    if (ni & IDX_MTD)\
        dprintf("@%d", mthd[cn][i].idx);\
    dprintf(" (mthd offset: 0x%x, h: %d, l: %d)",  mthd[cn][i].base, \
            mthd[cn][i].fld[fi].hbit, mthd[cn][i].fld[fi].lbit);\
    dprintf("\n");\
    } while (0)

#define PRINT_FLD_SC(cn,ni,i,fi,off,sc) \
    do { \
    LwU32 hi =  mthd[cn][i].fld[fi].hbit, lo = mthd[cn][i].fld[fi].lbit;\
    dprintf("%s.%s", mthd[cn][i].name, mthd[cn][i].fld[fi].name);\
    if (ni & IDX_SUB)\
        dprintf("@%d[%s]", mthd[cn][i].sub->idx, mthd[cn][i].sub->name);\
    if (ni & IDX_MTD)\
        dprintf("@%d", mthd[cn][i].idx);\
    if (sc) {   \
        LwU32 tmp;\
        tmp = GET_BITS(GPU_REG_RD32(sc),hi,lo);\
        dprintf(" SC(%04x|%4d) ", tmp, tmp);\
    }\
    dprintf(" (mthd offset: 0x%x, ", off);\
    if (sc) dprintf("sc offset: 0x%x, ", sc);\
    dprintf("bit %d:%d)", hi, lo);\
    dprintf("\n");\
    } while (0)

static unsigned int numNeedIdx(mthds_t *mtd)
{
    unsigned int needIdx = 0;
    if (mtd->sub != NULL && !(strcmp(mtd->sub->name, "HEAD")))
        needIdx |= IDX_HEAD;
    else if (mtd->sub != NULL && mtd->sub->idx > 1)
        needIdx |= IDX_SUB;

    if (mtd->idx > 1)
        needIdx |= IDX_MTD;

    return needIdx;
}

static int findMethod(int chanNum, int headNum, char *name, LwU32 *offset, LwU32 *hbit, LwU32 *lbit, LwU32 *sc)
{
    char mtd[1024], fld[1024], buf[1024];
    int ret = 0, hasIdx = 0, idx1 = 0, idx2 = 0, i = 0, fi = 0, di = 0;
    int printField=0, printDesc=0, needIdx = 0, useFld = 0;
    size_t len = 0;

    *sc  = 0;
    ret = sscanf(name, "%[^.@].%[^.@]@%d@%d", mtd, fld, &idx1, &idx2);

    if (ret == 0) 
    {
        dprintf("No matching method.\n");
        goto dump_mthd;
    }
    else if (ret > 0)
    {
        // check if it has !
        len = strlen(mtd);
        if (mtd[len - 1] == '*' || mtd[len-1] == '!')
        {
            mtd[len - 1] = '\0';
            printField = 1;
        }

        // search for the right method
        if ((i = getIdxMatch(chanNum, mtd)) == -1)
        {
            dprintf("No matching method.\n");
            goto dump_mthd;
        }

        // check if we need to specify idx1 or idx2
        needIdx = numNeedIdx(&mthd[chanNum][i]);

        // try this too, the user may want to assign values directly
        if (ret == 1)
        {
            // no field - Direct Mode
            useFld = 0;
            hasIdx = sscanf(name, "%[^.@]@%d@%d", buf, &idx1, &idx2) - 1;
            if (len = strlen(name), name[len -1] == '!' ||
                    name[len -1] == '*')
                printField = 1;

        }
        else if (ret > 1)
        {
            // use field
            useFld = 1;

            if (!printField)
            {
                // check if it has !
                len = strlen(fld);
                if ((fld[len - 1] == '*') || (fld[len - 1] == '!'))
                {
                    fld[len - 1] = '\0';
                    printDesc = 1;
                }
                else if (len = strlen(name), name[len -1] == '!' ||
                        name[len -1] == '*')
                    printDesc = 1;
                // search for the right method
                if ((fi = getIdxFldMatch(chanNum, fld, i)) == -1)
                {
                    dprintf("No matching field.\n");
                    printField = 1;
                    //goto dump_fld; // with field
                }
            }

            hasIdx = ret - 2;

        }
        else
        {
            dprintf("BUG\n");
            return -1;
        }

        if (needIdx == (IDX_SUB|IDX_MTD))
        {
            if (hasIdx != 2)
            {
                dprintf("This method needs 2 indices. Use @n@n form\n");
                PRINT_MTHD(chanNum,needIdx, i);
                if (hasIdx == 1)
                    idx2 = 0;
                // now has..
                hasIdx = 2;
                dprintf("Using @%d@%d\n", idx1, idx2);
            }
        }
        else if (needIdx & (IDX_SUB|IDX_MTD))
        {
            if (hasIdx != 1)
            {
                // only IDX_SUB or IDX_MTD is set
                dprintf("This method needs 1 index. Use @n form.\n");
                PRINT_MTHD(chanNum,needIdx, i);
                idx1 = 0;
                // now has
                hasIdx = 1;
                dprintf("Using @0\n");
            }
        }
        else if (!(needIdx & (IDX_SUB|IDX_MTD)) && hasIdx)
        {
            dprintf("This method needs no index.\n");
            hasIdx = 0;
        }

        // special case
        if (needIdx & IDX_HEAD)
        {
            if ((hasIdx == 1) && (needIdx & IDX_MTD))
            {
                idx2 = idx1;
                idx1 = headNum;
            }
            else if ((hasIdx == 0) && !(needIdx & IDX_MTD))
            {
                idx1 = headNum;
            } else
                dprintf("Inconsistent.. BUG\n");
        }

        // all good
        *offset = mthd[chanNum][i].base;

        if (mthd[chanNum][i].sc_avail)
        {
            *sc = mthd[chanNum][i].sc_offset;
            if (!(needIdx & IDX_HEAD))
            {
                // NOT core.
                *sc += headNum ? mthd[chanNum][i].sc_head : 0;
            }
        }
        else
        {
            *sc = 0;
        }

        // treat HEAD and SUB equiv.
        if ((needIdx & IDX_SUB) || (needIdx & IDX_HEAD) )
        {
            if (idx1 < mthd[chanNum][i].sub->idx)
            {
                *offset += mthd[chanNum][i].sub->size * idx1;
                if (*sc)
                    *sc += (needIdx & IDX_HEAD ?
                            mthd[chanNum][i].sc_head :  mthd[chanNum][i].sc_subidx) * idx1;
            }
            else
            {
                dprintf("index should be smaller than %d\n", mthd[chanNum][i].sub->idx);
                return -1;
            }
            if ((needIdx & IDX_MTD))
            {
                if (idx2 < mthd[chanNum][i].idx)
                {
                    *offset += mthd[chanNum][i].size * idx2;
                    if (*sc)
                        *sc += mthd[chanNum][i].sc_mtdidx * idx2;
                }
                else
                {
                    dprintf("index should be smaller than %d\n", mthd[chanNum][i].idx);
                    return -1;
                }
            }
        }
        else if ((needIdx & IDX_MTD))
        {
                if (idx1 < mthd[chanNum][i].idx)
                {
                    *offset += mthd[chanNum][i].size * idx1;
                    if (*sc)
                        *sc += mthd[chanNum][i].sc_mtdidx * idx1;
                }
                else
                {
                    dprintf("index should be smaller than %d\n", mthd[chanNum][i].idx);
                    return -1;
                }
        }

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
        if (printField)
        {
            goto dump_fld;
        }
        if (printDesc)
        {
            goto dump_desc;
        }
#else
        if (mthd[chanNum][i].sc_avail)
        {
            if (!print_sc_dinj(mthd[chanNum][i].sc_offset, (unsigned)*sc, chanNum, mthd[chanNum][i].name))
            {
                if (printField)
                {
                    goto dump_fld;
                }
                if (printDesc)
                {
                    goto dump_desc;
                }
            }
        }
        else
        {
            if (printField)
            {
                goto dump_fld;
            }
            if (printDesc)
            {
                goto dump_desc;
            }
        }
#endif

        if (useFld)
        {
            *hbit = mthd[chanNum][i].fld[fi].hbit;
            *lbit = mthd[chanNum][i].fld[fi].lbit;
        }
        else
        {
            *hbit = 31;
            *lbit = 0;
        }

    } 
    else
    {
        goto dump_mthd;
    }

    return 0;

dump_desc:
    // i - mthd index, fi - field index
    // TODO: print sc information as well
    PRINT_FLD_SC(chanNum, needIdx, i, fi, *offset, *sc);
    if (mthd[chanNum][i].fld[fi].num_desc)
    {
        for(di = 0; di < mthd[chanNum][i].fld[fi].num_desc ; di++)
        {
            dprintf("   %c |%04x|%4d| -> %s\n",
                ((LwU32)((*sc) && mthd[chanNum][i].fld[fi].desc[di].val) ==
                   GET_BITS(GPU_REG_RD32(*sc),mthd[chanNum][i].fld[fi].hbit,
                       mthd[chanNum][i].fld[fi].lbit) ? '*' : ' ' ) ,
                mthd[chanNum][i].fld[fi].desc[di].val,
                mthd[chanNum][i].fld[fi].desc[di].val,
                mthd[chanNum][i].fld[fi].desc[di].desc);
        }
    }
    else
    {
        dprintf("No descriptor for this field.\n");
    }
    return -1;
dump_fld:
    if (*sc)
    {
        dprintf("%s (sc offset 0x%08x): 0x%08x\n",
        mthd[chanNum][i].name, *sc, GPU_REG_RD32(*sc));
    }
    if (mthd[chanNum][i].num_fld)
    {
        for(fi = 0; fi < mthd[chanNum][i].num_fld ; fi++)
        {
            PRINT_FLD_SC(chanNum, needIdx, i, fi,*offset, *sc);
        }
    }
    else 
    {
        if (*sc)
            dprintf("%s.<No field> SC(0x%08x) (sc offset: 0x%08x)\n",
                    mthd[chanNum][i].name, GPU_REG_RD32(*sc), *sc);
        else
            dprintf("%s.<No field>\n", mthd[chanNum][i].name);
    }
    return -1;

dump_mthd:
    for (i = 0 ; i < mthd_num[chanNum] ; i++)
    {
        needIdx = numNeedIdx(&mthd[chanNum][i]);
        PRINT_MTHD(chanNum, needIdx, i);
    }
    return -1;
}

// mthdName format -  MTHD_NAME_FIELD@IDX@IDX
LwS32 dispMatchMethod_v03_00
(
    LwU32 chanNum,
    LwU32 headNum,
    char * mthdName,
    LwU32 *offset,
    LwU32 *hbit,
    LwU32 *lbit,
    LwU32 *sc
)
{
    LwU32 base = 0;
    char *check;
    if (chanNum >= LWDISPLAY_DISP_NUM_DMA_CHANNELS)
    {
        dprintf("%s: chanNum must be smaller than %d\n",
            __FUNCTION__, LWDISPLAY_DISP_NUM_DMA_CHANNELS);
        return -1;
    }
    // check if it's real direct mode. i.e. mthdName is offset
    base = strtoul(mthdName, &check, 0);
    if (*check == '\0') {
        // numeric, thus it's REAL direct mode.
        *offset = base;
        *hbit = 31;
        *lbit = 0;
        *sc = 0;
        dprintf("DIRECT MODE @ 0x%08x\n", base);
        return 0;
    }

    if(!mthdInitialized)
    {
        dprintf("Class Methods are not initialized\n");
        return -1;
    }
    return findMethod(chanNum, headNum, mthdName, offset, hbit, lbit, sc);
}

void dispGetAllClasses_v03_00(char* dest, char separator)
{
    char coreClass[16]  = "LWC37D";
    char winClass[16]   = "LWC37E";
    char winimClass[16] = "LWC37B";

    sprintf(dest, "%s%c%s%c%s%c", coreClass, separator, winClass, separator, winimClass, separator);
}

// Find a single class specified by chan_template in classes, seperated by separator
static LwU32 dispFindClass(mthds_t **mthds, int *num_mthds, char *classNames, char separator, char *chan_template, int n)
{
    char lwr_class[32];

    assert(n < 32);

    if (findDisplayClass(classNames, separator, chan_template, n, lwr_class)==0)
    {
        dprintf("\nLwrrent Class not found");
        return 0;
    }
    if (strncasecmp(lwr_class, "LWC37B", 6) == 0)
    {
        *mthds = lwc37b_mthds;
        *num_mthds = num_lwc37b_mthds;
        return 1;
    }
    else if (strncasecmp(lwr_class, "LWC37D", 6) == 0)
    {
        *mthds = lwc37d_mthds;
        *num_mthds = num_lwc37d_mthds;
        return 1;
    }
    else if (strncasecmp(lwr_class, "LWC37E", 6) == 0)
    {
        *mthds = lwc37e_mthds;
        *num_mthds = num_lwc37e_mthds;
        return 1;
    }
    else
    {
        dprintf("%s is not lwrrently supported\n", lwr_class);
        return 0;
    }
}

void
initializeDisp_v03_00(char *chipName)
{
    char *tmpScPath;
    char scPath[256];
    char classNames[256];
    char dispManualPath[256];
    mthds_t *methodsB, *methodsD, *methodsE;
    int num_mthdsB, num_mthdsD, num_mthdsE;

    memset(scPath, 0, sizeof(scPath));
    tmpScPath = getelw("LWW_MANUAL_SDK");
    if (tmpScPath == NULL)
    {
        dprintf("lw: Please set your LWW_MANUAL_SDK environment variable to point to your "
                INC_DIR_EXAMPLE " directory\n");
        return;
    }

    strcpy(scPath, tmpScPath);

    strcat(scPath, DIR_SLASH);


    if(!GetDispManualsDir(dispManualPath))
    {
        dprintf("lw:%s(): Failed to initialise for current chip",
            __FUNCTION__);
        return;
    }

    strcat(scPath, dispManualPath);

    strcat(scPath, DIR_SLASH "dev_disp.h");

    memset(classNames, 0, sizeof(classNames));
    //TODO: find a better way to dynamically find class names and fill methods and headers
    dispGetAllClasses_v03_00((char *)(classNames), ';');
    if (dispFindClass(&methodsB, &num_mthdsB, classNames, ';', "LW***B", 6) != 0 &&
        dispFindClass(&methodsD, &num_mthdsD, classNames, ';', "LW***D", 6) != 0 &&
        dispFindClass(&methodsE, &num_mthdsE, classNames, ';', "LW***E", 6) != 0 )
    {
        LwU32 mthdIndex = 0;

        mthd[mthdIndex]     = methodsD;
        mthd_num[mthdIndex] = num_mthdsD;

        for(mthdIndex = WIN_PUSHBUFFER_OFFSET; mthdIndex < LW_PDISP_FE_DEBUG_CTL__SIZE_1; mthdIndex++)
        {
            if(mthdIndex < WIN_PUSHBUFFER_OFFSET + LW_PDISP_FE_PBBASE_WIN__SIZE_1)
            {
                mthd[mthdIndex]     = methodsE;
                mthd_num[mthdIndex] = num_mthdsE;
            }
            else if(mthdIndex < WINIM_PUSHBUFFER_OFFSET + LW_PDISP_FE_PBBASE_WINIM__SIZE_1)
            {
                mthd[mthdIndex]     = methodsB;
                mthd_num[mthdIndex] = num_mthdsB;
            }
        }

        mthdInitialized = 1;
    }
    else
    {
        dprintf("Methods not initialized\n");
        return;
    }

    initializeClassHeaderNum_v03_00(classNames, classHeaderNumLwdisplay);
}

void initializeClassHeaderNum_v03_00(char *classNames, LwU32 classHeaderNum[])
{
    LwU32 classheaderB, classheaderD, classheaderE;

    if (dispFindHeaderNum(&classheaderB, classNames, ';', "LW***B", 6) != 0 &&
        dispFindHeaderNum(&classheaderD, classNames, ';', "LW***D", 6) != 0 &&
        dispFindHeaderNum(&classheaderE, classNames, ';', "LW***E", 6) != 0 )
    {
        classHeaderNum[0] = classheaderD;
        classHeaderNum[1] = classheaderE;
        classHeaderNum[2] = classheaderB;
    }
    else
    {
        dprintf("Class Headers not initialized\n");
    }
}

void dispPrintChnFeState_v03_00(LwU32 chanNum)
{
    unsigned int val = 0;
    //Core channel
    if(chanNum == 0)
    {
        dprintf("** CORE CHANNEL STATE **\n");
        val = GPU_REG_RD32(LW_PDISP_FE_CHNSTATUS_CORE);

        dprintf("STG1 STATE: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_CORE, _STG1_STATE, val))
        {
            case LW_PDISP_FE_CHNSTATUS_CORE_STG1_STATE_IDLE:
            {
                dprintf("IDLE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STG1_STATE_READ_METHOD:
            {
                dprintf("READ METHOD");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STG1_STATE_REQ_METHOD_INFO:
            {
                dprintf("REQ_METHOD_INFO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STG1_STATE_CHK_METHOD_INFO:
            {
                dprintf("CHK_METHOD_INFO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STG1_STATE_CHK_CTXDMA:
            {
                dprintf("CHK_CTXDMA");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STG1_STATE_CTX_DMA_LOOKUP:
            {
                dprintf("CTX_DMA_LOOKUP");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STG1_STATE_WAIT_FOR_STG2:
            {
                dprintf("WAIT_FOR_STG2");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STG1_STATE_WAIT_FOR_UPD:
            {
                dprintf("WAIT_FOR_UPD");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STG1_STATE_EXCEPTION:
            {
                dprintf("EXCEPTION");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("STG2 STATE: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_CORE, _STG2_STATE, val))
        {
            case LW_PDISP_FE_CHNSTATUS_CORE_STG2_STATE_IDLE:
            {
                dprintf("IDLE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STG2_STATE_SEND_PUBLIC:
            {
                dprintf("SEND_PUBLIC");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STG2_STATE_SEND_MISC:
            {
                dprintf("SEND_MISC");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STG2_STATE_SEND_LIMIT:
            {
                dprintf("SEND_LIMIT");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STG2_STATE_SEND_BASE:
            {
                dprintf("SEND_BASE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STG2_STATE_SEND_SETPARAMSCRSR:
            {
                dprintf("SEND_SETPARAMSCRSR");
                break;
            }
            default:
            dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("STATE: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_CORE, _STATE, val))
        {
            case LW_PDISP_FE_CHNSTATUS_CORE_STATE_DEALLOC:
            {
                dprintf("DEALLOC");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATE_DEALLOC_LIMBO:
            {
                dprintf("DEALLOC_LIMBO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATE_VBIOS_INIT1:
            {
                dprintf("VBIOS_INIT1");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATE_VBIOS_INIT2:
            {
                dprintf("VBIOS_INIT2");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATE_VBIOS_OPERATION:
            {
                dprintf("VBIOS_OPERATION");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATE_EFI_INIT1:
            {
                dprintf("EFI_INIT1");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATE_EFI_INIT2:
            {
                dprintf("EFI_INIT2");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATE_EFI_OPERATION:
            {
                dprintf("EFI_OPERATION");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATE_UNCONNECTED:
            {
                dprintf("UNCONNECTED");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATE_INIT1:
            {
                dprintf("INIT1");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATE_INIT2:
            {
                dprintf("INIT2");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATE_IDLE:
            {
                dprintf("IDLE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATE_BUSY:
            {
                dprintf("BUSY");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATE_SHUTDOWN1:
            {
                dprintf("SHUTDOWN1");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATE_SHUTDOWN2:
            {
                dprintf("SHUTDOWN2");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("FIRST TIME: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_CORE, _FIRSTTIME, val))
        {
            case LW_PDISP_FE_CHNSTATUS_CORE_FIRSTTIME_NO:
            {
                dprintf("NO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_FIRSTTIME_YES:
            {
                dprintf("YES");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("METHOD FIFO: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_CORE, _STATUS_METHOD_FIFO, val))
        {
            case LW_PDISP_FE_CHNSTATUS_CORE_STATUS_METHOD_FIFO_EMPTY:
            {
                dprintf("EMPTY");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATUS_METHOD_FIFO_NOTEMPTY:
            {
                dprintf("NOTEMPTY");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("READ PENDING: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_CORE, _STATUS_READ_PENDING, val))
        {
            case LW_PDISP_FE_CHNSTATUS_CORE_STATUS_READ_PENDING_NO:
            {
                dprintf("NO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATUS_READ_PENDING_YES:
            {
                dprintf("YES");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("NOTIF WRITE PENDING: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_CORE, _STATUS_NOTIF_WRITE_PENDING, val))
        {
            case LW_PDISP_FE_CHNSTATUS_CORE_STATUS_NOTIF_WRITE_PENDING_NO:
            {
                dprintf("NO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATUS_NOTIF_WRITE_PENDING_YES:
            {
                dprintf("YES");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("SUBDEVICE STATUS: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_CORE, _SUBDEVICE_STATUS, val))
        {
            case LW_PDISP_FE_CHNSTATUS_CORE_SUBDEVICE_STATUS_INACTIVE:
            {
                dprintf("INACTIVE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_SUBDEVICE_STATUS_ACTIVE:
            {
                dprintf("ACTIVE");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("QUIESCENT STATUS: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_CORE, _STATUS_QUIESCENT, val))
        {
            case LW_PDISP_FE_CHNSTATUS_CORE_STATUS_QUIESCENT_NO:
            {
                dprintf("NO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATUS_QUIESCENT_YES:
            {
                dprintf("YES");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("METHOD EXEC: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_CORE, _STATUS_METHOD_EXEC, val))
        {
            case LW_PDISP_FE_CHNSTATUS_CORE_STATUS_METHOD_EXEC_IDLE:
            {
                dprintf("IDLE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_CORE_STATUS_METHOD_EXEC_RUNNING:
            {
                dprintf("RUNNING");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");
    }

    // WIN CHANNEL
    else if (chanNum < WINIM_PUSHBUFFER_OFFSET)
    {
        dprintf("** WIN CHANNEL STATE **\n");
        val = GPU_REG_RD32(LW_PDISP_FE_CHNSTATUS_WIN(chanNum));
        dprintf("STG1 STATE: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WIN, _STG1_STATE, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WIN_STG1_STATE_IDLE:
            {
                dprintf("IDLE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STG1_STATE_READ_METHOD:
            {
                dprintf("READ METHOD");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STG1_STATE_REQ_METHOD_INFO:
            {
                dprintf("REQ_METHOD_INFO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STG1_STATE_CHK_METHOD_INFO:
            {
                dprintf("CHK_METHOD_INFO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STG1_STATE_CHK_CTXDMA:
            {
                dprintf("CHK_CTXDMA");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STG1_STATE_CTX_DMA_LOOKUP:
            {
                dprintf("CTX_DMA_LOOKUP");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STG1_STATE_WAIT_FOR_STG2:
            {
                dprintf("WAIT_FOR_STG2");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STG1_STATE_WAIT_FOR_UPD:
            {
                dprintf("WAIT_FOR_UPD");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STG1_STATE_EXCEPTION:
            {
                dprintf("EXCEPTION");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("STG2 STATE: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WIN, _STG2_STATE, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WIN_STG2_STATE_IDLE:
            {
                dprintf("IDLE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STG2_STATE_SEND_PUBLIC:
            {
                dprintf("SEND_PUBLIC");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STG2_STATE_SEND_MISC:
            {
                dprintf("SEND_MISC");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STG2_STATE_SEND_LIMIT: 
            {
                dprintf("SEND_LIMIT");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STG2_STATE_SEND_BASE:
            {
                dprintf("SEND_BASE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STG2_STATE_SEND_WIN_SETCONFIG:
            {
                dprintf("SEND_WIN_SETCONFIG");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("UPDATE STATE: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WIN, _UPD_STATE, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WIN_UPD_STATE_IDLE:
            {
                dprintf("IDLE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_UPD_STATE_WAIT_BLOCK:
            {
                dprintf("WAIT_BLOCK");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_UPD_STATE_WAIT_MPI:
            {
                dprintf("WAIT_MPI");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_UPD_STATE_WAIT_ILK_PH_1:
            {
                dprintf("WAIT_ILK_PH_1");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_UPD_STATE_WAIT_STATE_ERRCHK:
            {
                dprintf("WAIT_STATE_ERRCHK");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_UPD_STATE_WAIT_RDY_TO_FLIP:
            {
                dprintf("WAIT_RDY_TO_FLIP");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_UPD_STATE_WAIT_ILK_PH_2:
            {
                dprintf("WAIT_ILK_PH_2");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_UPD_STATE_CHECK_PEND_LOADV:
            {
                dprintf("CHECK_PEND_LOADV");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_UPD_STATE_SEND_UPD:
            {
                dprintf("SEND_UPD");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_UPD_STATE_WAIT_PRM:
            {
                dprintf("WAIT_PRM");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_UPD_STATE_EXCEPTION:
            {
                dprintf("EXCEPTION");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_UPD_STATE_WAIT_ILK_ABORT:
            {
                dprintf("WAIT_ILK_ABORT");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("STATE: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WIN, _STATE, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WIN_STATE_DEALLOC:
            {
                dprintf("DEALLOC");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STATE_UNCONNECTED:
            {
                dprintf("UNCONNECTED");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STATE_INIT1:
            {
                dprintf("INIT1");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STATE_INIT2:
            {
                dprintf("INIT2");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STATE_IDLE:
            {
                dprintf("IDLE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STATE_BUSY:
            {
                dprintf("BUSY");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STATE_SHUTDOWN1:
            {
                dprintf("SHUTDOWN1");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STATE_SHUTDOWN2:
            {
                dprintf("SHUTDOWN2");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("FIRST TIME: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WIN, _FIRSTTIME, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WIN_FIRSTTIME_NO:
            {
                dprintf("NO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_FIRSTTIME_YES:
            {
                dprintf("YES");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("METHOD FIFO: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WIN, _STATUS_METHOD_FIFO, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WIN_STATUS_METHOD_FIFO_EMPTY:
            {
                dprintf("EMPTY");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STATUS_METHOD_FIFO_NOTEMPTY:
            {
                dprintf("NOTEMPTY");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("READ PENDING: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WIN, _STATUS_READ_PENDING, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WIN_STATUS_READ_PENDING_NO:
            {
                dprintf("NO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STATUS_READ_PENDING_YES:
            {
                dprintf("YES");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("WRITE PENDING: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WIN, _STATUS_WRITE_PENDING, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WIN_STATUS_WRITE_PENDING_NO:
            {
                dprintf("NO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STATUS_WRITE_PENDING_YES:
            {
                dprintf("YES");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("SUBDEVICE STATUS: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WIN, _SUBDEVICE_STATUS, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WIN_SUBDEVICE_STATUS_INACTIVE:
            {
                dprintf("INACTIVE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_SUBDEVICE_STATUS_ACTIVE:
            {
                dprintf("ACTIVE");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("QUIESCENT STATUS: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WIN, _STATUS_QUIESCENT, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WIN_STATUS_QUIESCENT_NO:
            {
                dprintf("NO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STATUS_QUIESCENT_YES:
            {
                dprintf("YES");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("METHOD EXEC: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WIN, _STATUS_METHOD_EXEC, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WIN_STATUS_METHOD_EXEC_IDLE:
            {
                dprintf("IDLE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WIN_STATUS_METHOD_EXEC_RUNNING:
            {
                dprintf("RUNNING");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");
    }

    //WINIM CHANNEL
    else if (chanNum < (WINIM_PUSHBUFFER_OFFSET + LW_PDISP_FE_CHNCTL_WINIM__SIZE_1))
    {
        dprintf("** WINIM CHANNEL STATE **\n");
        val = GPU_REG_RD32(LW_PDISP_FE_CHNSTATUS_WINIM(chanNum));

        dprintf("MP_STATE: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WINIM, _MP_STATE, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WINIM_MP_STATE_IDLE:
            {
                dprintf("IDLE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_MP_STATE_EXCEPT:
            {
                dprintf("EXCEPT");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_MP_STATE_SEND_PUBLIC:
            {
                dprintf("SEND_PUBLIC");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_MP_STATE_WAIT_ILK1:
            {
                dprintf("WAIT_ILK1");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_MP_STATE_WAIT_FLIP:
            {
                dprintf("WAIT_FLIP");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_MP_STATE_WAIT_ILK2:
            {
                dprintf("WAIT_ILK2");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_MP_STATE_WAIT_LOADV:
            {
                dprintf("WAIT_LOADV");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_MP_STATE_SEND_UPDATE:
            {
                dprintf("SEND_UPDATE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_MP_STATE_WAIT_PRM:
            {
                dprintf("WAIT_PRM");
            break;
        }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("STATE: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WINIM, _STATE, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATE_DEALLOC:
            {
                dprintf("DEALLOC");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATE_UNCONNECTED:
            {
                dprintf("UNCONNECTED");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATE_INIT1:
            {
                dprintf("INIT1");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATE_INIT2:
            {
                dprintf("INIT2");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATE_IDLE:
            {
                dprintf("IDLE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATE_BUSY:
            {
                dprintf("BUSY");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATE_SHUTDOWN1:
            {
                dprintf("SHUTDOWN1");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATE_SHUTDOWN2:
            {
                dprintf("SHUTDOWN2");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("FIRST TIME: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WINIM, _FIRSTTIME, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WINIM_FIRSTTIME_NO:
            {
                dprintf("NO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_FIRSTTIME_YES:
            {
                dprintf("YES");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("METHOD FIFO: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WINIM, _STATUS_METHOD_FIFO, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATUS_METHOD_FIFO_EMPTY:
            {
                dprintf("EMPTY");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATUS_METHOD_FIFO_NOTEMPTY:
            {
                dprintf("NOTEMPTY");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("READ PENDING: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WINIM, _STATUS_READ_PENDING, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATUS_READ_PENDING_NO:
            {
                dprintf("NO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATUS_READ_PENDING_YES:
            {
                dprintf("YES");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("NOTIF WRITE PENDING: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WINIM, _STATUS_WRITE_PENDING, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATUS_WRITE_PENDING_NO:
            {
                dprintf("NO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATUS_WRITE_PENDING_YES:
            {
                dprintf("YES");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("SUBDEVICE STATUS: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WINIM, _SUBDEVICE_STATUS, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WINIM_SUBDEVICE_STATUS_INACTIVE:
            {
                dprintf("INACTIVE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_SUBDEVICE_STATUS_ACTIVE:
            {
                dprintf("ACTIVE");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("QUIESCENT STATUS: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WINIM, _STATUS_QUIESCENT, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATUS_QUIESCENT_NO:
            {
                dprintf("NO");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATUS_QUIESCENT_YES:
            {
                dprintf("YES");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");

        dprintf("METHOD EXEC: ");
        switch(DRF_VAL(_PDISP, _FE_CHNSTATUS_WINIM, _STATUS_METHOD_EXEC, val))
        {
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATUS_METHOD_EXEC_IDLE:
            {
                dprintf("IDLE");
                break;
            }
            case LW_PDISP_FE_CHNSTATUS_WINIM_STATUS_METHOD_EXEC_RUNNING:
            {
                dprintf("RUNNING");
                break;
            }
            default:
                dprintf("Invalid data");
        }
        dprintf("\n");
    }
}

static void generateHlsStringAndPrint(char* buffer, regmatch_t* pmatch, size_t  maxSize)
{
    int openBracket = 1;
    int tempOffset = pmatch->rm_eo + 1;
    int size;
    char* match = NULL;
    int i,j;

    while (openBracket > 0)
    {
        if ((size_t)tempOffset >= maxSize)
        {
            break;
        }

        if (buffer[tempOffset] == '{')
        {
            openBracket++;
        }
        else if (buffer[tempOffset] == '}')
        {
            openBracket--;
        }

        tempOffset++;
    }

    size = tempOffset - pmatch->rm_so;
    match = malloc(size);

    j = 0;
    for (i = pmatch->rm_so; i < tempOffset; i++)
    {
        match[j] = buffer[i];
        j++;
    }

    for (i=0; i < size + 1; i++)
    {
        dprintf("%c", match[i]);
    }
    dprintf("\n");
    free(match);
}

void dispParseHls_v03_00(LwBool isCore, LwU32 errNum)
{
    FILE *f = NULL;
    char filePath[256];
    char dispHlsPath[256];
    char *filePathTmp;
    char errorCode[10];
    const char *regex_a, *regex_b;
    char regex[100];
    char *buffer;
    size_t len;
    size_t size;
    size_t i = 0, j = 0;
    regex_t regx;
    size_t nmatch = 1;
    regmatch_t pmatch[1];

    // Generatign the regex string specific to LWDisplay.
    sprintf(errorCode,"%d",errNum);
    regex_a = "LWDisplay Error code: ";
    regex_b = "[^0-9][^{]*{";
    len = strlen(regex_a) + strlen(errorCode) + strlen(regex_b); 

    if (len > 100)
    {
       dprintf("Size of regex buffer is not enough\n");
       return;
    }

    regex[0] = '\0';
    strcpy(regex, regex_a);
    strcat(regex, errorCode);
    strcat(regex, regex_b);

    // grab the file Path where the hls files are imported. 
    memset(filePath, 0, sizeof(filePath));
    filePathTmp = getelw("LWW_MANUAL_SDK");

    if (filePathTmp == NULL)
    {
        dprintf("lw: Please set yourLWW_MANUAL_SDKS environment variable to point"
                "to the directory with HLS files\n");
        return;
    }

    strcpy(filePath, filePathTmp);
    strcat(filePath, DIR_SLASH);

    if(!GetDispManualsDir(dispHlsPath))
    {
        dprintf("lw:%s(): Failed to initialise for current chip",
            __FUNCTION__);
        return;
    }

    strcat(filePath, dispHlsPath);
    strcat(filePath, DIR_SLASH);

    strcat(filePath, "hls");
    strcat(filePath, DIR_SLASH);

    if(isCore)
    {
        strcat(filePath, "dispCoreUpdateErrorChecks_hls.c.txt");
    }
    else
    {
        strcat(filePath, "dispWindowUpdateErrorChecks_hls.c.txt");
    }

    f = fopen(filePath, "rb");
    if (!f)
    {
        dprintf("File: %s not found!\n", filePath);
        return;
    }

    //read the respective hls file to a buffer
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buffer = malloc(size + 1);
    fread(buffer, 1, size, f);
    fclose(f);

    // beging matching the regular expression.
    if (regcomp(&regx, regex, REG_EXTENDED))
    {
        dprintf("Failed To Compile Regex!\n");
        return;
    }

    if (regexec(&regx, buffer, nmatch, pmatch,0))
    {
        dprintf("Cannot Find Match");
        return;
    }

    generateHlsStringAndPrint(buffer, pmatch, size);
    free(buffer);
    regfree(&regx); 
}

void dispDumpPendingExcHls_v03_00(LwBool isCore, LwBool isWindow)
{
    LwU32 evtDispatch;
    LwU32 val;
    LwBool corePend = LW_FALSE;
    LwBool winPend = LW_FALSE;
    LwU32 chnNum;
    LwU32 errCode;
    LwU32 i;
    evtDispatch = GPU_REG_RD32(LW_PDISP_FE_EVT_DISPATCH);

    if (isCore)
    {
        if (FLD_TEST_DRF(_PDISP, _FE_EVT_DISPATCH, 
                         _EXC_OTHER, _PENDING, evtDispatch))
        {
            val = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_EXC_OTHER);

            corePend = FLD_TEST_DRF(_PDISP, _FE_EVT_STAT_EXC_OTHER, 
                               _CORE, _PENDING, val);
            if (corePend)
            {
                chnNum  = LW_PDISP_EXCEPT_CHN_NUM_CORE;
                errCode = GPU_REG_IDX_RD_DRF(_PDISP, _FE_EXCEPTERR, 
                                             chnNum, _CODE);
                dprintf("**********CORE CHANNEL EXCEPTION PENDING WITH ERROR CODE %d**********\n", errCode);
                dispParseHls_v03_00(LW_TRUE, errCode);
                dprintf("*********************************************************************\n");
            }
        }
    }

    if (isWindow)
    {
        if (FLD_TEST_DRF(_PDISP, _FE_EVT_DISPATCH,
                         _EXC_WIN, _PENDING, evtDispatch))
        {
            val = GPU_REG_RD32(LW_PDISP_FE_EVT_STAT_EXC_WIN);

            for (i =0; i < LW_PDISP_FE_EVT_STAT_EXC_WIN_CH__SIZE_1; i++)
            {
                winPend = LW_FALSE;

                winPend = FLD_IDX_TEST_DRF(_PDISP, _FE_EVT_STAT_EXC_WIN, _CH, i, _PENDING, val);
                if (winPend)
                {
                    chnNum = LW_PDISP_EXCEPT_CHN_NUM_WIN(i);
                    errCode = GPU_REG_IDX_RD_DRF(_PDISP, _FE_EXCEPTERR,
                                                 chnNum, _CODE);
                    dprintf("**********WINDOW %d HAS CHANNEL EXCEPTION PENDING WITH ERROR CODE %d**********\n", i, errCode);
                    dispParseHls_v03_00(LW_FALSE, errCode);            
                    dprintf("******************************************************************************\n");
                } 
            }
        }
    }
}

/*!
 * @brief Helper function to return SLI Data.
 *
 *  @param[in]  LwU32      head       Head index in DSLI_DATA structure to fill
 *  @param[in]  DSLI_DATA *pDsliData  Pointer to DSLI data structure
 */
void dispGetSliData_v03_00
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
    pDsliData[head].DsliRgStatusFlipLocked = DRF_VAL(C37D, _SET_CONTROL, _FLIP_LOCK_PIN0,
                                                     GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE +
                                                                  LWC37D_SET_CONTROL));
    pDsliData[head].DsliClkRemVpllExtRef = GPU_REG_RD32(LW_PVTRIM_SYS_VPLL_MISC(head));
    pDsliData[head].DsliClkDriverSrc = DRF_VAL(_PVTRIM, _SYS_VPLL_MISC, _EXT_REF_CONFIG_SRC, pDsliData[head].DsliClkRemVpllExtRef);
    pDsliData[head].DsliHeadSetCntrl = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR_CORE + LWC37D_HEAD_SET_CONTROL(head));
    pDsliData[head].DsliHeadSetSlaveLockMode = DRF_VAL(C37D, _HEAD_SET_CONTROL, _SLAVE_LOCK_MODE, pDsliData[head].DsliHeadSetCntrl);
    pDsliData[head].DsliHeadSetMasterLockMode = DRF_VAL(C37D, _HEAD_SET_CONTROL, _MASTER_LOCK_MODE, pDsliData[head].DsliHeadSetCntrl);
    pDsliData[head].DsliHeadSetSlaveLockPin = DRF_VAL(C37D, _HEAD_SET_CONTROL, _SLAVE_LOCK_PIN, pDsliData[head].DsliHeadSetCntrl);
    pDsliData[head].DsliHeadSetMasterLockPin = DRF_VAL(C37D, _HEAD_SET_CONTROL, _MASTER_LOCK_PIN, pDsliData[head].DsliHeadSetCntrl);
}

/*!
 * @brief Function to print SLI registers.
 *
 *  @param[in]  LwU32            numHead        Number of Heads
 *  @param[in]  LwU32            numPior        Number of PIORs (unused)
 *  @param[in]  DSLI_DATA       *pDsliData      Pointer to DSLI data structure
 *  @param[in]  DSLI_PIOR_DATA  *pDsliPiorData  Pointer to DSLI_PIOR data
 *                                              structure
 */
void dispPrintSliRegisters_v03_00
(
    LwU32           numHead,
    LwU32           numPior,
    DSLI_DATA      *pDsliData,
    DSLI_PIOR_DATA *pDsliPiorData
)
{
    LwU32 head = 0;

    dprintf("--------------------------------------------------------------------------------------------------------\n");
    dprintf("                                        Register Information                                            \n");
    dprintf("--------------------------------------------------------------------------------------------------------\n");

    dprintf("%40s |", "");
    for (head = 0; head < numHead; ++head)
    {
        dprintf("HEAD-%d       |", head);
    }
    dprintf("\n%40s |", "");
    for (head = 0; head < numHead; ++head)
    {
        dprintf("------       |");
    }
    dprintf("\n");

    PRINTCONFIGHEAD("LW_PDISP_RG_SWAP_LOCKOUT", pDsliData, DsliRgFlipLock);
    PRINTCONFIGHEAD("LW_PDISP_RG_STATUS", pDsliData, DsliRgStatus);
    PRINTCONFIGHEAD("LWC37D_SET_CONTROL_FLIP_LOCK_PIN0", pDsliData, DsliRgStatusFlipLocked);
    PRINTCONFIGHEAD("LW_PVTRIM_SYS_VPLL_MISC", pDsliData, DsliClkRemVpllExtRef);
    PRINTCONFIGHEAD("LWC37D_HEAD_SET_CONTROL", pDsliData, DsliHeadSetCntrl);
    PRINTCONFIGHEAD("LW_PVTRIM_SYS_VCLK_REF_SWITCH", pDsliData, DsliPvTrimSysVClkRefSwitch);

    dprintf("%40s |", "");
    for (head = 0; head < numHead; ++head)
    {
        dprintf("------       |");
    }
    dprintf("\n\n");

    PRINTCONFIG("LW_PDISP_FE_SW_LOCK_PIN_CAP", pDsliPiorData->DsliCap);
}

/*!
 * @brief dispPrintHeadData - Function to print SLI-HEAD config data,
 * used by DSLI. It prints SLI register values for configuration
 *
 *  @param[in]  LwU32              numHead         Number of Heads
 *  @param[in]  DSLI_DATA         *pDsliData       Pointer to DSLI data
 *                                                 structure
 *  @param[in]  DSLI_PRINT_PARAM  *pDsliPrintData  Pointer to print
 *                                                 Param datastructure
 *  @param[in]  LwU32              verbose         Verbose switch
 */
void dispPrintHeadData_v03_00
(
    LwU32               numHead,
    DSLI_DATA          *pDsliData,
    DSLI_PRINT_PARAM   *pDsliPrintData,
    LwU32               verbose
)

{
    LwU32 head  = 0;

    for (head = 0; head < numHead; ++head)
    {
        if (pDsliData[head].DsliHeadActive)
        {
            pDsliPrintData[head].headStatus = "Active";
        }

        PRINTLOCKMODE(DsliHeadSetSlaveLockMode, pDsliData, pDsliPrintData, slaveLock)
        PRINTLOCKPIN(DsliHeadSetSlaveLockPin, pDsliData, pDsliPrintData, slaveLockPin)
        PRINTLOCKMODE(DsliHeadSetMasterLockMode, pDsliData, pDsliPrintData, masterLock)
        PRINTLOCKPIN(DsliHeadSetMasterLockPin, pDsliData, pDsliPrintData, masterLockPin)

        switch(pDsliData[head].DsliRgStatusLocked)
        {
            case LW_PDISP_RG_STATUS_LOCKED_TRUE:
                pDsliPrintData[head].scanLockStatus = "Locked";
                break;

            default:
                pDsliPrintData[head].scanLockStatus = "N/A";
                break;
        }

        switch(pDsliData[head].DsliRgStatusFlipLocked)
        {
            case LWC37D_SET_CONTROL_FLIP_LOCK_PIN0_ENABLE:
                pDsliPrintData[head].flipLock = "Enabled";
                pDsliPrintData[head].flipLockStatus = "Locked";
                break;

            default:
                pDsliPrintData[head].flipLock = "disabled";
                pDsliPrintData[head].flipLockStatus = "N/A";

        }

        pDsliPrintData[head].syncAdvance = pDsliData[head].DsliRgDistRndrSyncAdv;

        // Print clock data for active Head
        pDisp[indexGpu].dispPrintClkData(head, pDsliData, pDsliPrintData, verbose);
    }

    dispPrintSliStatus (numHead, pDsliData, pDsliPrintData, verbose);

    dprintf("--------------------------------------------------------------------------------------------------------\n");
}

/*!
 * @brief dispDumpSLIConfig - Function to dump SLI Config Data.
 * It dumps SLI register values & print results related to GPU
 * configuration
 *
 *  @param[in] LwU32      Verbose - Switch to enable extra information
 *                                  logging.
 */
void dispDumpSliConfig_v03_00
(
    LwU32 verbose
)
{
    LwU32               head = 0, numHead = 0;
    LwU32               pior = 0, numPior = 0;
    DSLI_DATA           *pDsliData;
    DSLI_PIOR_DATA      *pDsliPiorData;
    DSLI_PRINT_PARAM    *pDsliPrintData;

    // Find out total number of heads
    numHead = pDisp[indexGpu].dispGetNumHeads();

    // Find number of Piors
    numPior = pDisp[indexGpu].dispGetNumOrs(LW_OR_PIOR);

    pDsliData = (DSLI_DATA*)malloc(sizeof(DSLI_DATA) * numHead);
    pDsliPiorData = (DSLI_PIOR_DATA*)malloc(sizeof(DSLI_PIOR_DATA));
    pDsliPrintData = (DSLI_PRINT_PARAM*)malloc(sizeof(DSLI_PRINT_PARAM) * numHead);

    if ((pDsliData == NULL) || (pDsliPiorData == NULL) || (pDsliPrintData == NULL))
    {
        dprintf("lw: %s - malloc failed!\n", __FUNCTION__);
        dprintf("lw: Failed Pointers : DSLI_DATA-%p, DSLI_PIOR_DATA-%p, DSLI_PRINT_PARAM-%p\n", \
                pDsliData, pDsliPiorData, pDsliPrintData);
        free(pDsliData);
        free(pDsliPiorData);
        free(pDsliPrintData);
        return;
    }

    dprintf("--------------------------------------------------------------------------------------------------------\n");
    dprintf("                                        Display SLI Configuration                                                 \n");
    dprintf("--------------------------------------------------------------------------------------------------------\n");

    for (head = 0; head < numHead; ++head)
    {
        // Make all heads inactive initially
        pDsliData[head].DsliHeadActive = FALSE;

        // Initialize the print parameters
        dispInitializeSliData(&pDsliPrintData[head]);

        // Function to get all SLI configuration
        pDisp[indexGpu].dispGetSliData(head, pDsliData);
    }

    pDsliPiorData->DsliCap = GPU_REG_RD32(LW_PDISP_FE_SW_LOCK_PIN_CAP);

    for (pior = 0; pior < numPior; ++pior)
    {
       pDisp[indexGpu].dispGetPiorData(pior, pDsliData, pDsliPiorData);
    }

    // Get the clock data
    pDisp[indexGpu].dispGetClockData(pDsliData);

    // Call the print function to print configuration results
    dispPrintSliData(numHead, numPior, 0 /* numPin */, pDsliData, pDsliPiorData, pDsliPrintData, verbose);

    free(pDsliData);
    free(pDsliPiorData);
    free(pDsliPrintData);
}

/*!
 * @brief dispPrintPinData - Function to print SLI-PIN config data,
 * used by DSLI. It prints SLI register values for configuration
 *
 *  @param[in]  LwU32            numPin         Number of Pins
 *  @param[in]  DSLI_PIOR_DATA  *pDsliPiorData  Pointer to PIOR
 *                                              datastructure
 */

void dispPrintPinData_v03_00
(
    LwU32            numPin,
    DSLI_PIOR_DATA  *pDsliPiorData
)
{
    dprintf("                                  LOCK-PIN-CAPABILITIES                                                 \n");
    dprintf("========================================================================================================\n");
    dprintf("Function  Pin count\n");
    dprintf("--------------------------------------------------------------------------------------------------------\n");
    dprintf("FLIP_LOCK %d\n", DRF_VAL(_PDISP, _FE_SW_LOCK_PIN_CAP, _FLIP_LOCK_PINS, pDsliPiorData->DsliCap));
    dprintf("SCAN_LOCK %d\n", DRF_VAL(_PDISP, _FE_SW_LOCK_PIN_CAP, _SCAN_LOCK_PINS, pDsliPiorData->DsliCap));
    dprintf("STEREO    %d\n", DRF_VAL(_PDISP, _FE_SW_LOCK_PIN_CAP, _STEREO_PINS,    pDsliPiorData->DsliCap));
    dprintf("--------------------------------------------------------------------------------------------------------\n");
}

LwU32 dispGetChannelStateCacheValue_v03_00(LwU32 chNum, BOOL isArmed, LwU32 offset)
{
    LwU32 scIndex = dispChanState_v03_00[chNum].scIndex;
    LwU32 baseAddr  = isArmed ? LW_UDISP_FE_CHN_ARMED_BASEADR(scIndex) : LW_UDISP_FE_CHN_ASSY_BASEADR(scIndex);
    return GPU_REG_RD32(baseAddr + offset);
}