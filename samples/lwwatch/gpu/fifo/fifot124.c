/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// includes
//
#include "t21x/t210/dev_arhost1x.h"
#include "t21x/t210/dev_arhost1x_channel.h"
#include "t21x/t210/dev_arhost1x_sync.h"
#include "t21x/t210/project_arhost1x_sw_defs.h"
#include "t21x/t210/dev_fifo.h"
#include "inst.h"
#include "fb.h"
#include "vmem.h"
#include "hwref/t21x/t210/class_ids.h"

#include "gpuanalyze.h"

#include "g_fifo_private.h"        // (rmconfig) implementation prototypes

#define CHANNEL_REGR(_chid_, _reg_)                                                     \
    DEV_REG_RD32(                                                                       \
    ( (_chid_ * LW_PHOST1X_CHANNEL_MAP_SIZE_BYTES) + LW_PHOST1X_CHANNEL0_BASE +         \
      LW_PHOST1X_CHANNEL_##_reg_ ), "HOST1X", 0 )


#define HOST1X_SYNC_REGR(_reg_)                                                         \
    DEV_REG_RD32(                                                                       \
    ( LW_PHOST1X_CHANNEL0_BASE + LW_PHOST1X_CHANNEL_SYNC_REG_BASE + _reg_), "HOST1X", 0 )

void _printClassIdName(LwU32 classId)
{
    switch(classId)
    {
        case LW_HOST1X_CLASS_ID:
            dprintf("LW_HOST1X_CLASS_ID");
            break;

        case LW_VIDEO_ENCODE_MPEG_CLASS_ID:
            dprintf("LW_VIDEO_ENCODE_MPEG_CLASS_ID");
            break;

        case LW_VIDEO_STREAMING_VI_CLASS_ID:
            dprintf("LW_VIDEO_STREAMING_VI_CLASS_ID");
            break;

        case LW_VIDEO_STREAMING_EPP_CLASS_ID:
            dprintf("LW_VIDEO_STREAMING_EPP_CLASS_ID");
            break;

        case LW_VIDEO_STREAMING_ISP_CLASS_ID:
            dprintf("LW_VIDEO_STREAMING_ISP_CLASS_ID");
            break;

        case LW_VIDEO_STREAMING_VCI_CLASS_ID:
            dprintf("LW_VIDEO_STREAMING_VCI_CLASS_ID");
            break;

        case LW_GRAPHICS_2D_DOWNLOAD_CLASS_ID:
            dprintf("LW_GRAPHICS_2D_DOWNLOAD_CLASS_ID");
            break;

        case LW_GRAPHICS_2D_CLASS_ID:
            dprintf("LW_GRAPHICS_2D_CLASS_ID");
            break;

        case LW_GRAPHICS_2D_SB_CLASS_ID:
            dprintf("LW_GRAPHICS_2D_SB_CLASS_ID");
            break;

        case LW_GRAPHICS_2D_DOWNLOAD_CTX1_CLASS_ID:
            dprintf("LW_GRAPHICS_2D_DOWNLOAD_CTX1_CLASS_ID");
            break;

        case LW_GRAPHICS_2D_CTX1_CLASS_ID:
            dprintf("LW_GRAPHICS_2D_CTX1_CLASS_ID");
            break;

        case LW_GRAPHICS_2D_SB_CTX1_CLASS_ID:
            dprintf("LW_GRAPHICS_2D_SB_CTX1_CLASS_ID");
            break;

        case LW_GRAPHICS_2D_DOWNLOAD_CTX2_CLASS_ID:
            dprintf("LW_GRAPHICS_2D_DOWNLOAD_CTX2_CLASS_ID");
            break;

        case LW_GRAPHICS_2D_SB_CTX2_CLASS_ID:
            dprintf("LW_GRAPHICS_2D_SB_CTX2_CLASS_ID");
            break;

        case LW_GRAPHICS_3D_CLASS_ID:
            dprintf("LW_GRAPHICS_3D_CLASS_ID");
            break;

        case LW_DISPLAY_CLASS_ID:
            dprintf("LW_DISPLAY_CLASS_ID");
            break;

        case LW_DISPLAYB_CLASS_ID:
            dprintf("LW_DISPLAYB_CLASS_ID");
            break;

        case LW_HDMI_CLASS_ID:
            dprintf("LW_HDMI_CLASS_ID");
            break;

        case LW_DISPLAY_TVO_CLASS_ID:
            dprintf("LW_DISPLAY_TVO_CLASS_ID");
            break;

        case LW_DISPLAY_DSI_CLASS_ID:
            dprintf("LW_DISPLAY_DSI_CLASS_ID");
            break;

        default:
            dprintf("Uknown CLASS_ID");
            break;
    }
}

//-----------------------------------------------------
// instmemDumpFifoCtx_T124
//
//-----------------------------------------------------
void instmemDumpFifoCtx_T124(ChannelId *pChannelId)
{
    LwU32 reg;
    LwU32 chid = pChannelId->id;

    if (chid > LW_HOST1X_CHANNELS)
    {
        dprintf("lw: %s Invalid channel: 0x%x\n", __FUNCTION__, chid);
        return;
    }

    //
    // arhost1x_channel.h state
    // Printing HOST1X_CHANNEL_INDDATA_0 results in a hang
    //
    dprintf("lw: Dumping arhost1x_channel.h State for Channel: 0x%x\n", chid);
    dprintf("lw: LW_PHOST1X_CHANNEL_FIFOSTAT:     0x%08x\n", CHANNEL_REGR(chid, FIFOSTAT));
    dprintf("lw: LW_PHOST1X_CHANNEL_INDOFF:       0x%08x\n", CHANNEL_REGR(chid, INDOFF));
    dprintf("lw: LW_PHOST1X_CHANNEL_INDCNT:       0x%08x\n", CHANNEL_REGR(chid, INDCNT));
    //dprintf("lw: LW_PHOST1X_CHANNEL_INDDATA:      0x%08x\n", CHANNEL_REGR(chid, INDDATA));
    dprintf("lw: LW_PHOST1X_CHANNEL_RAISE:        0x%08x\n", CHANNEL_REGR(chid, RAISE));
    dprintf("lw: LW_PHOST1X_CHANNEL_DMASTART:     0x%08x\n", CHANNEL_REGR(chid, DMASTART));
    dprintf("lw: LW_PHOST1X_CHANNEL_DMAPUT:       0x%08x\n", CHANNEL_REGR(chid, DMAPUT));
    dprintf("lw: LW_PHOST1X_CHANNEL_DMAGET:       0x%08x\n", CHANNEL_REGR(chid, DMAGET));
    dprintf("lw: LW_PHOST1X_CHANNEL_DMAEND:       0x%08x\n", CHANNEL_REGR(chid, DMAEND));
    dprintf("lw: LW_PHOST1X_CHANNEL_DMACTRL:      0x%08x\n", CHANNEL_REGR(chid, DMACTRL));
    dprintf("lw: LW_PHOST1X_CHANNEL_FBBUFBASE:    0x%08x\n", CHANNEL_REGR(chid, FBBUFBASE));
    dprintf("lw: LW_PHOST1X_CHANNEL_CMDSWAP:      0x%08x\n", CHANNEL_REGR(chid, CMDSWAP));
    dprintf("lw: LW_PHOST1X_CHANNEL_INDOFF2:      0x%08x\n", CHANNEL_REGR(chid, INDOFF2));
    dprintf("lw: LW_PHOST1X_CHANNEL_TICKCOUNT_HI: 0x%08x\n", CHANNEL_REGR(chid, TICKCOUNT_HI));
    dprintf("lw: LW_PHOST1X_CHANNEL_TICKCOUNT_LO: 0x%08x\n", CHANNEL_REGR(chid, TICKCOUNT_LO));
    dprintf("lw: LW_PHOST1X_CHANNEL_CHANNELCTRL:  0x%08x\n", CHANNEL_REGR(chid, CHANNELCTRL));
    dprintf("\n");

    //
    // arhost1x_sync.h state
    //
    dprintf("lw: Dumping arhost1x_sync.h State for Channel: 0x%x\n", chid);
    reg = HOST1X_SYNC_REGR(LW_PHOST1X_SYNC_CH0_STATUS + (chid*4));
    dprintf("lw: LW_PHOST1X_SYNC_CH%d_STATUS:      0x%08x\n", chid, reg);
    reg = DRF_VAL(_PHOST1X_SYNC, _CH0_STATUS, _CHOUT_CLASS0, reg);
    dprintf("lw:    CHOUT_CLASS0:               0x%x - ", reg);
    _printClassIdName(reg);
    dprintf("\n");
    reg = HOST1X_SYNC_REGR(LW_PHOST1X_SYNC_CBREAD0 + (chid*4));
    dprintf("lw: LW_PHOST1X_SYNC_CBREAD%d:         0x%08x\n", chid, reg);
    dprintf("\n");
}

//-----------------------------------------------------
// fifoGetInfo_T124
//
//-----------------------------------------------------
LW_STATUS fifoGetInfo_T124(void)
{
    LwU32 chid;
    ChannelId channelId;

    dprintf("lw: LW_PHOST1X_SYNC_INTSTATUS:       0x%08x\n", HOST1X_SYNC_REGR(LW_PHOST1X_SYNC_INTSTATUS));
    dprintf("lw: LW_PHOST1X_SYNC_HINTSTATUS:      0x%08x\n", HOST1X_SYNC_REGR(LW_PHOST1X_SYNC_HINTSTATUS));
    dprintf("lw: LW_PHOST1X_SYNC_HINTSTATUS_EXT:  0x%08x\n", HOST1X_SYNC_REGR(LW_PHOST1X_SYNC_HINTSTATUS_EXT));
    dprintf("lw: LW_PHOST1X_SYNC_SYNCPT_THRESH_CPU0_INT_STATUS: 0x%08x\n", HOST1X_SYNC_REGR(LW_PHOST1X_SYNC_SYNCPT_THRESH_CPU0_INT_STATUS));
    dprintf("\n");

    // Dump the state for each chid
    channelId.bRunListValid = LW_FALSE;
    channelId.bChramPriBaseValid  = LW_FALSE;
    for (chid = 0; chid < LW_HOST1X_CHANNELS; chid++)
    {
        channelId.id = chid;
        pInstmem[indexGpu].instmemDumpFifoCtx(&channelId);
        dprintf("\n");
    }

    pFifo[indexGpu].fifoGetInfo();
    return LW_OK;
}

/*!
 * @return The maximum number of channels provided by the chip
 */
LwU32 fifoGetNumChannels_T124(LwU32 runlistId)
{
    // Unused pre-Ampere
    (void) runlistId;
    return LW_PCCSR_CHANNEL__SIZE_1;
}

