/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2013-2014 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

//-----------------------------------------------------
//
// lwdec0100.c - LWDEC v1 routines
// 
//-----------------------------------------------------

#include "maxwell/gm107/dev_pri_ringstation_sys.h"
#include "maxwell/gm107/dev_lwdec_pri.h"
#include "maxwell/gm107/dev_falcon_v4.h"
#include "maxwell/gm107/dev_fifo.h"
#include "maxwell/gm107/dev_master.h"

#include "lwdec.h"

#include "g_lwdec_private.h"     // (rmconfig)  implementation prototypes

#define LW_A0B0_NOP                                                              (0x00000100)
#define LW_A0B0_PM_TRIGGER                                                       (0x00000140)
#define LW_A0B0_SET_APPLICATION_ID                                               (0x00000200)
#define LW_A0B0_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LW_A0B0_SEMAPHORE_A                                                      (0x00000240)
#define LW_A0B0_SEMAPHORE_B                                                      (0x00000244)
#define LW_A0B0_SEMAPHORE_C                                                      (0x00000248)
#define LW_A0B0_EXELWTE                                                          (0x00000300)
#define LW_A0B0_SEMAPHORE_D                                                      (0x00000304)
#define LW_A0B0_H264_SET_SEQ_PIC_CTRL_IN_OFFSET                                  (0x00000400)
#define LW_A0B0_H264_SET_MBHIST_BUF_OFFSET                                       (0x00000404)
#define LW_A0B0_H264_SET_MBHIST_BUF_SIZE                                         (0x00000408)
#define LW_A0B0_H264_SET_BASE_SLC_HDR_OFFSET                                     (0x0000040C)
#define LW_A0B0_H264_SET_PICTURE_OFFSET3                                         (0x00000410)
#define LW_A0B0_H264_SET_PICTURE_OFFSET4                                         (0x00000414)
#define LW_A0B0_H264_SET_PICTURE_OFFSET5                                         (0x00000418)
#define LW_A0B0_H264_SET_PICTURE_OFFSET6                                         (0x0000041C)
#define LW_A0B0_H264_SET_PICTURE_OFFSET7                                         (0x00000420)
#define LW_A0B0_H264_SET_PICTURE_OFFSET8                                         (0x00000424)
#define LW_A0B0_H264_SET_PICTURE_OFFSET9                                         (0x00000428)
#define LW_A0B0_H264_SET_PICTURE_OFFSET10                                        (0x0000042C)
#define LW_A0B0_H264_SET_PICTURE_OFFSET11                                        (0x00000430)
#define LW_A0B0_H264_SET_PICTURE_OFFSET12                                        (0x00000434)
#define LW_A0B0_H264_SET_PICTURE_OFFSET13                                        (0x00000438)
#define LW_A0B0_H264_SET_PICTURE_OFFSET14                                        (0x0000043C)
#define LW_A0B0_H264_SET_PICTURE_OFFSET15                                        (0x00000440)
#define LW_A0B0_H264_SET_PICTURE_OFFSET16                                        (0x00000444)
#define LW_A0B0_H264_SET_PICTURE_OFFSET17                                        (0x00000448)
#define LW_A0B0_H264_SET_SLICE_COUNT                                             (0x0000044C)
#define LW_A0B0_VP8_SET_PROB_DATA_OFFSET                                         (0x00000480)
#define LW_A0B0_VP8_SET_PICTURE_OFFSET3                                          (0x00000484)
#define LW_A0B0_VC1_SET_SEQ_CTRL_IN_OFFSET                                       (0x00000500)
#define LW_A0B0_VC1_SET_PIC_SCRATCH_BUF_OFFSET                                   (0x00000504)
#define LW_A0B0_VC1_SET_PIC_SCRATCH_BUF_SIZE                                     (0x00000508)
#define LW_A0B0_MPEG12_SET_SEQ_PIC_CTRL_IN_OFFSET                                (0x00000600)
#define LW_A0B0_SET_CONTROL_PARAMS                                               (0x00000700)
#define LW_A0B0_SET_DATA_INFO_BUFFER_IN_OFFSET                                   (0x00000704)
#define LW_A0B0_SET_IN_BUF_BASE_OFFSET                                           (0x00000708)
#define LW_A0B0_SET_PICTURE_INDEX                                                (0x0000070C)
#define LW_A0B0_SET_NUM_SLICE_CNT                                                (0x00000710)
#define LW_A0B0_SET_DRV_PIC_SETUP_OFFSET                                         (0x00000714)
#define LW_A0B0_SET_COLOC_DATA_OFFSET                                            (0x00000718)
#define LW_A0B0_SET_HISTORY_OFFSET                                               (0x0000071C)
#define LW_A0B0_SET_PICTURE_OFFSET0                                              (0x00000720)
#define LW_A0B0_SET_PICTURE_OFFSET1                                              (0x00000724)
#define LW_A0B0_SET_PICTURE_OFFSET2                                              (0x00000728)
#define LW_A0B0_SET_DISPLAY_BUF_OFFSET                                           (0x00000730)
#define LW_A0B0_SET_DISPLAY_BUF_SIZE                                             (0x00000734)
#define LW_A0B0_SET_HISTOGRAM_OFFSET                                             (0x00000738)
#define LW_A0B0_SET_CONTENT_INITIAL_VECTOR(b)                                    (0x00000C00 + (b)*0x00000004)
#define LW_A0B0_SET_CTL_COUNT                                                    (0x00000C10)
#define LW_A0B0_SET_UPPER_SRC                                                    (0x00000C20)
#define LW_A0B0_SET_LOWER_SRC                                                    (0x00000C24)
#define LW_A0B0_SET_UPPER_DST                                                    (0x00000C28)
#define LW_A0B0_SET_LOWER_DST                                                    (0x00000C2C)
#define LW_A0B0_SET_UPPER_CTL                                                    (0x00000C30)
#define LW_A0B0_SET_LOWER_CTL                                                    (0x00000C34)
#define LW_A0B0_SET_BLOCK_COUNT                                                  (0x00000C38)
#define LW_A0B0_SET_STRETCH_MASK                                                 (0x00000C3C)
#define LW_A0B0_SET_UCODE_LOADER_PARAMS                                          (0x00000D10)
#define LW_A0B0_SET_UCODE_LOADER_OFFSET                                          (0x00000D34)
#define LW_A0B0_MP4_SET_SEQ_CTRL_IN_OFFSET                                       (0x00000E00)
#define LW_A0B0_MP4_SET_PIC_SCRATCH_BUF_OFFSET                                   (0x00000E04)
#define LW_A0B0_MP4_SET_PIC_SCRATCH_BUF_SIZE                                     (0x00000E08)
#define LW_A0B0_SET_SESSION_KEY(b)                                               (0x00000F00 + (b)*0x00000004)
#define LW_A0B0_SET_CONTENT_KEY(b)                                               (0x00000F10 + (b)*0x00000004)
#define LW_A0B0_PM_TRIGGER_END                                                   (0x00001114)

dbg_lwdec_v01_01 lwdecMethodTable_v01_00[] =
{
    privInfo_lwdec_v01_01(LW_A0B0_NOP),
    privInfo_lwdec_v01_01(LW_A0B0_PM_TRIGGER),
    privInfo_lwdec_v01_01(LW_A0B0_SET_APPLICATION_ID),
    privInfo_lwdec_v01_01(LW_A0B0_SET_WATCHDOG_TIMER),
    privInfo_lwdec_v01_01(LW_A0B0_SEMAPHORE_A),
    privInfo_lwdec_v01_01(LW_A0B0_SEMAPHORE_B),
    privInfo_lwdec_v01_01(LW_A0B0_SEMAPHORE_C),
    privInfo_lwdec_v01_01(LW_A0B0_EXELWTE),
    privInfo_lwdec_v01_01(LW_A0B0_SEMAPHORE_D),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_SEQ_PIC_CTRL_IN_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_MBHIST_BUF_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_MBHIST_BUF_SIZE),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_BASE_SLC_HDR_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_PICTURE_OFFSET3),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_PICTURE_OFFSET4),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_PICTURE_OFFSET5),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_PICTURE_OFFSET6),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_PICTURE_OFFSET7),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_PICTURE_OFFSET8),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_PICTURE_OFFSET9),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_PICTURE_OFFSET10),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_PICTURE_OFFSET11),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_PICTURE_OFFSET12),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_PICTURE_OFFSET13),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_PICTURE_OFFSET14),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_PICTURE_OFFSET15),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_PICTURE_OFFSET16),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_PICTURE_OFFSET17),
    privInfo_lwdec_v01_01(LW_A0B0_H264_SET_SLICE_COUNT),
    privInfo_lwdec_v01_01(LW_A0B0_VP8_SET_PROB_DATA_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_VP8_SET_PICTURE_OFFSET3),
    privInfo_lwdec_v01_01(LW_A0B0_VC1_SET_SEQ_CTRL_IN_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_VC1_SET_PIC_SCRATCH_BUF_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_VC1_SET_PIC_SCRATCH_BUF_SIZE),
    privInfo_lwdec_v01_01(LW_A0B0_MPEG12_SET_SEQ_PIC_CTRL_IN_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_SET_CONTROL_PARAMS),
    privInfo_lwdec_v01_01(LW_A0B0_SET_DATA_INFO_BUFFER_IN_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_SET_IN_BUF_BASE_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_SET_PICTURE_INDEX),
    privInfo_lwdec_v01_01(LW_A0B0_SET_NUM_SLICE_CNT),
    privInfo_lwdec_v01_01(LW_A0B0_SET_DRV_PIC_SETUP_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_SET_COLOC_DATA_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_SET_HISTORY_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_SET_PICTURE_OFFSET0),
    privInfo_lwdec_v01_01(LW_A0B0_SET_PICTURE_OFFSET1),
    privInfo_lwdec_v01_01(LW_A0B0_SET_PICTURE_OFFSET2),
    privInfo_lwdec_v01_01(LW_A0B0_SET_DISPLAY_BUF_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_SET_DISPLAY_BUF_SIZE),
    privInfo_lwdec_v01_01(LW_A0B0_SET_HISTOGRAM_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_SET_CONTENT_INITIAL_VECTOR(0)),
    privInfo_lwdec_v01_01(LW_A0B0_SET_CONTENT_INITIAL_VECTOR(1)),
    privInfo_lwdec_v01_01(LW_A0B0_SET_CONTENT_INITIAL_VECTOR(2)),
    privInfo_lwdec_v01_01(LW_A0B0_SET_CONTENT_INITIAL_VECTOR(3)),
    privInfo_lwdec_v01_01(LW_A0B0_SET_CTL_COUNT),
    privInfo_lwdec_v01_01(LW_A0B0_SET_UPPER_SRC),
    privInfo_lwdec_v01_01(LW_A0B0_SET_LOWER_SRC),
    privInfo_lwdec_v01_01(LW_A0B0_SET_UPPER_DST),
    privInfo_lwdec_v01_01(LW_A0B0_SET_LOWER_DST),
    privInfo_lwdec_v01_01(LW_A0B0_SET_UPPER_CTL),
    privInfo_lwdec_v01_01(LW_A0B0_SET_LOWER_CTL),
    privInfo_lwdec_v01_01(LW_A0B0_SET_BLOCK_COUNT),
    privInfo_lwdec_v01_01(LW_A0B0_SET_STRETCH_MASK),
    privInfo_lwdec_v01_01(LW_A0B0_SET_UCODE_LOADER_PARAMS),
    privInfo_lwdec_v01_01(LW_A0B0_SET_UCODE_LOADER_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_MP4_SET_SEQ_CTRL_IN_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_MP4_SET_PIC_SCRATCH_BUF_OFFSET),
    privInfo_lwdec_v01_01(LW_A0B0_MP4_SET_PIC_SCRATCH_BUF_SIZE),
    privInfo_lwdec_v01_01(LW_A0B0_SET_SESSION_KEY(0)),
    privInfo_lwdec_v01_01(LW_A0B0_SET_SESSION_KEY(1)),
    privInfo_lwdec_v01_01(LW_A0B0_SET_SESSION_KEY(2)),
    privInfo_lwdec_v01_01(LW_A0B0_SET_SESSION_KEY(3)),
    privInfo_lwdec_v01_01(LW_A0B0_SET_CONTENT_KEY(0)),
    privInfo_lwdec_v01_01(LW_A0B0_SET_CONTENT_KEY(1)),
    privInfo_lwdec_v01_01(LW_A0B0_SET_CONTENT_KEY(2)),
    privInfo_lwdec_v01_01(LW_A0B0_SET_CONTENT_KEY(3)),
    privInfo_lwdec_v01_01(LW_A0B0_PM_TRIGGER_END),
    privInfo_lwdec_v01_01(0),
};

dbg_lwdec_v01_01 lwdecPrivReg_v01_00[] =
{
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQSSET),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQSCLR),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQSTAT),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQMODE),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQMSET),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQMCLR),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQMASK),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQDEST),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_GPTMRINT),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_GPTMRVAL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_GPTMRCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_PTIMER0),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_PTIMER1),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_WDTMRVAL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_WDTMRCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_MTHDDATA),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_MTHDID),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_MTHDWDAT),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_MTHDCOUNT),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_MTHDPOP),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_MTHDRAMSZ),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_LWRCTX),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_NXTCTX),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_CTXACK),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_MAILBOX0),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_MAILBOX1),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_ITFEN),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IDLESTATE),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_FHSTATE),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_PRIVSTATE),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_SFTRESET),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_OS),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_RM),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_SOFT_PM),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_SOFT_MODE),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DEBUG1),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DEBUGINFO),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IBRKPT1),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IBRKPT2),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IBRKPT3),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IBRKPT4),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IBRKPT5),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_CGCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_ENGCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_PMM),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_ADDR),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_CPUCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_BOOTVEC),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_HWCFG),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_HWCFG1),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMACTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMATRFBASE),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMATRFMOFFS),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMATRFCMD),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMATRFFBOFFS),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMAPOLL_FB),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMAPOLL_CP),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMSTAT),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_TRACEIDX),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_TRACEPC),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMFILLRNG0),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMFILLRNG1),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMFILLCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_CG1_SLCG),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_ICD_CMD),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_ICD_ADDR),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_ICD_WDATA),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_ICD_RDATA),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMEMC(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMEMC(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMEMC(2)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMEMC(3)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMEMD(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMEMD(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMEMD(2)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMEMD(3)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMEMT(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMEMT(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMEMT(2)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMEMT(3)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMC(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMC(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMC(2)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMC(3)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMC(4)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMC(5)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMC(6)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMC(7)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMD(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMD(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMD(2)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMD(3)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMD(4)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMD(5)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMD(6)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEMD(7)),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFINTRPTEN),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFINTRPTSTAT),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDBG1),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCLUMSCALE(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCLUMSCALE(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCLUMSCALE(2)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCLUMSCALE(3)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCLUMSHIFT(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCLUMSHIFT(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCLUMSHIFT(2)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCLUMSHIFT(3)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCINTRPTEN),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCINTRPTSTAT),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCDBG1),
    privInfo_lwdec_v01_01(LW_PLWDEC_VERSION),
    privInfo_lwdec_v01_01(LW_PLWDEC_CAP_REG0),
    privInfo_lwdec_v01_01(LW_PLWDEC_CAP_REG1),
    privInfo_lwdec_v01_01(LW_PLWDEC_CAP_REG2),
    privInfo_lwdec_v01_01(LW_PLWDEC_CAP_REG3),
    privInfo_lwdec_v01_01(LW_PLWDEC_CG),
    privInfo_lwdec_v01_01(LW_PLWDEC_CG1),
    privInfo_lwdec_v01_01(LW_PLWDEC_CG2),
    privInfo_lwdec_v01_01(LW_PLWDEC_CG3),
    privInfo_lwdec_v01_01(LW_PLWDEC_PMM),
    privInfo_lwdec_v01_01(LW_PLWDEC_RECCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_RECINTEN),
    privInfo_lwdec_v01_01(LW_PLWDEC_RECINTCSW),
    privInfo_lwdec_v01_01(LW_PLWDEC_RECDBG1),
    privInfo_lwdec_v01_01(LW_PLWDEC_RECDBG2),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_PIC_INFO_COMMON),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_PIC_INFO_H264),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_PIC_INFO_MPEG2),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_PIC_INFO_VC1GB),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_PIC_INFO_VC1T),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_PIC_INFO_VC1Q),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_PIC_INFO_VC1P),    
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_SLICE_INFO),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_EDOB_MB_INFO),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_EDOB_MB_POS),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_MB_INFO),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_MB_POS),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_PARSE_CMD),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_STATUS),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_RESULT),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_INTRPT_EN),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_INTRPT_STATUS),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_BIT_ERROR_CODE),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_SCP_CFG),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_TIMEOUT_VALUE),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_RESET_CTL),    
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA_BUFF_A(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA_BUFF_A(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA_BUFF_A(2)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA_BUFF_A(3)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA_BUFF_B(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA_BUFF_B(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA_BUFF_B(2)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA_BUFF_B(3)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_LWRR_IDMA_INFO),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_CLEAR_INPUT_BUFFERS),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_HIST_START_ADDR),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_HIST_SIZE),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_HIST_CTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_CBC_DEBUG_OP),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_DEBUG),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_VC1_DEBUG0),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_VC1_DEBUG1),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_MPG_DEBUG),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_BIT_ERROR_MASK),    
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_SLICE_LIMIT),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_SLICE_ERROR_CTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_MPEG4_DEBUG),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_PROBBUF_WRITE_REG),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_PROBBUF_READ_REG),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_PROBBUF_RW_OFFSET),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_A(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_A(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_A(2)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_A(3)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_B(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_B(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_B(2)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_B(3)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_LWRR_IDMA1_INFO),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_SEGMENT_ID_START_ADDR),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_SFC_WIDTH),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_TOP_SFC_LUMA),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_TOP_SFC_CHROMA),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_BOT_SFC_LUMA),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_BOT_SFC_CHROMA),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_RW_CTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_CTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_FETCH),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_INTRPTEN),    
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_INTRPTSTAT),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_DBG1),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_ROW_COUNTER),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_MB_COUNTER),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_SFC_WIDTH_FIELD_UNFILTERED),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_TOP_SFC_LUMA_UNFILTERED),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_TOP_SFC_CHROMA_UNFILTERED),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_BOT_SFC_LUMA_UNFILTERED),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_BOT_SFC_CHROMA_UNFILTERED),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_TF_OUTPUT_PIC_INFO),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_TF_OUTPUT_LUMA_TOP_FIELD),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_TF_OUTPUT_LUMA_BOTTOM_FIELD),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_TF_OUTPUT_CHROMA_TOP_FIELD),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_TF_OUTPUT_CHROMA_BOTTOM_FIELD),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_HISTOGRAM_WINDOW_START),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_HISTOGRAM_WINDOW_END),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_HISTOGRAM_RESULT),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_DITHER_MATRIX_WRITE_DATA),
    privInfo_lwdec_v01_01(LW_PLWDEC_HIST_CFG),
    privInfo_lwdec_v01_01(LW_PLWDEC_HIST_CTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_HIST_BUF_START),
    privInfo_lwdec_v01_01(LW_PLWDEC_HIST_DBG0),
    privInfo_lwdec_v01_01(LW_PLWDEC_HIST_DBG1),
    privInfo_lwdec_v01_01(LW_PLWDEC_HIST_DBG2),
    privInfo_lwdec_v01_01(LW_PLWDEC_HIST_DBG3),
    privInfo_lwdec_v01_01(LW_PLWDEC_HIST_DBG4),
    privInfo_lwdec_v01_01(LW_PLWDEC_HIST_DBG5),
    privInfo_lwdec_v01_01(LW_PLWDEC_HIST_DBG6),
    privInfo_lwdec_v01_01(LW_PLWDEC_HIST_DBG7),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTINTRPTEN),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTINTRPTSTAT),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTDBG1),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTDBG2),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTDBG3),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTTYPE),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTMP2),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTVC1),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTQMEM),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTDCTCTRL),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTDMA),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTRDOFFS),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTRDCNT),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTWROFFS),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTWRCNT),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTRDCROP),
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTWRCROP),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCFETCH_MCFCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCFETCH_FRMSFCW),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCFETCH_FLDSFCW),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCFETCH_FRMSIZE),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCFETCH_DBG_STAT(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVLOCAL_MEMORY(0,0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVLOCAL_MEMORY(1,0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVLOCAL_MEMORY(2,0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVLOCAL_MEMORY(3,0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVSEQSTART),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVFRMPARAM),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVWDCOUNTER),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVWDTIME),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVCTRL0),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVCTRL1),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVPSINFO),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVCTRL2),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVRPLBASE),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVSPLBASE),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVSTATUS),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVPSINFO2),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVINTCSW),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVINTEN),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVWT_CNTRL),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVWT_DATA),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVRCOLADDR),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVRCOLPARAM),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVRCOLLINE),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVRCOLOFFS),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVWCOLADDR),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVWCOLPARAM),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVWCOLLINE),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVWCOLOFFS),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVISSTART),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVISSIZE),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVISOFFS),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVISCOUNT),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVISCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVDBG_CTL),    
    privInfo_lwdec_v01_01(LW_PLWDEC_MVERRORCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVERRORMASK),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVERRORSTAT),
    privInfo_lwdec_v01_01(LW_PLWDEC_CTL_STAT),
    privInfo_lwdec_v01_01(LW_PLWDEC_CTL_CFG),
    privInfo_lwdec_v01_01(LW_PLWDEC_CTL_SCP),
    privInfo_lwdec_v01_01(LW_PLWDEC_CTL_HDCP0),
    privInfo_lwdec_v01_01(LW_PLWDEC_CTL_HDCP1),
    privInfo_lwdec_v01_01(LW_PLWDEC_BAR0_CSR),
    privInfo_lwdec_v01_01(LW_PLWDEC_BAR0_ADDR),
    privInfo_lwdec_v01_01(LW_PLWDEC_BAR0_DATA),
    privInfo_lwdec_v01_01(LW_PLWDEC_BAR0_TMOUT),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_TRANSCFG(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_TRANSCFG(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_TRANSCFG(2)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_TRANSCFG(3)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_TRANSCFG(4)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_TRANSCFG(5)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_TRANSCFG(6)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_TRANSCFG(7)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_INSTBLK),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_CTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_DBG_STAT(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_THROTTLE),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_ACHK_BLK(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_ACHK_BLK(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_ACHK_CTL(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_FBIF_ACHK_CTL(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_CTL0),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_CTL1),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_CTL_STAT),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_CTL_CFG),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_CFG0),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_CTL_SCP),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_CTL_PKEY),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_CTL_DEBUG),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_DEBUG0),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_DEBUG1),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_DEBUG2),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_DEBUG_CMD),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_ACL_FETCH),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_STATUS),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_STAT0),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_STAT1),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNG_STAT0),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNG_STAT1),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_INTR),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_ACL_VIO),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_SELWRITY_VIO),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_CMD_ERROR),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNDCTL0),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNDCTL1),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNDCTL2),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNDCTL3),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNDCTL4),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNDCTL5),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNDCTL6),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNDCTL7),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNDCTL8),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNDCTL9),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNDCTL10),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNDCTL11),
    privInfo_lwdec_v01_01(0),
};

//-----------------------------------------------------
// lwdecIsSupported_v01_00
//-----------------------------------------------------
BOOL lwdecIsSupported_v01_00(LwU32 indexGpu, LwU32 engineId)
{
    if (engineId != LWWATCH_LWDEC_0)
        return FALSE;
 
    pLwdecPrivReg[engineId] = lwdecPrivReg_v01_00;
    pLwdecMethodTable = lwdecMethodTable_v01_00;
    return TRUE;
}

//-----------------------------------------------------
// lwdecIsPrivBlocked_v01_00
//-----------------------------------------------------
BOOL lwdecIsPrivBlocked_v01_00(LwU32 indexGpu, LwU32 engineId)
{
    LwU32 bitmask;
    LwU32 regSysPrivFsConfig;

    if (engineId != LWWATCH_LWDEC_0)
        return TRUE;

    // Bit-fields within LW_PPRIV_SYS_PRIV_FS_CONFIG denote priv access for video.
    // All video engines must have priv access for lwdec command support.
    regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(engineId));

    bitmask = BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2lwdec_pri);

    return ((regSysPrivFsConfig & bitmask) != bitmask);
}

//-----------------------------------------------------
// lwdecPrintMethodData_v01_00
//-----------------------------------------------------
void lwdecPrintMethodData_v01_00(LwU32 clmn, char *tag, LwU32 method, LwU32 data)
{
    size_t len = strlen(tag);
    
    dprintf("lw: %s",tag);

    if((len>0)&&(len<(clmn+4)))
    {
        LwU32 i;
        for(i=0;i<clmn-len;i++)
        {
            dprintf(" ");
        }
    }
    dprintf("(0x%08X)  = 0x%08X\n",method,data);
}

//-----------------------------------------------------
// lwdecDumpImem_v01_00 - Dumps LWDEC instruction memory
//-----------------------------------------------------
LW_STATUS lwdecDumpImem_v01_00(LwU32 indexGpu, LwU32 engineId, LwU32 imemSize)
{
    LW_STATUS status = LW_OK;
    LwU32  imemSizeMax;
    LwU32 addrssImem=LW_PLWDEC_FALCON_IMEMD(0);
    LwU32 address2Imem=LW_PLWDEC_FALCON_IMEMC(0);
    LwU32 address2Imemt = LW_PLWDEC_FALCON_IMEMT(0);
    LwU32 u;
    LwU32 blk=0;

    if (engineId != LWWATCH_LWDEC_0)
        return LW_ERR_NOT_SUPPORTED;

    imemSizeMax = (GPU_REG_RD_DRF(_PLWDEC_FALCON, _HWCFG, _IMEM_SIZE)<<8) ;
    if (imemSize > 0)
        imemSize = min(imemSizeMax, imemSize);
    else
        imemSize = imemSizeMax;

    dprintf("\n");
    dprintf("lw: -- Gpu %u LWDEC IMEM -- \n", indexGpu);    
    dprintf("lw: -- Gpu %u LWDEC IMEM SIZE =  0x%08x-- \n", indexGpu,imemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    for(u=0;u<(imemSize+3)/4;u++)
    {
        LwU32 i;
        if((u%64)==0) {
            GPU_REG_WR32(address2Imemt, blk++);
        }
        i = (u<<(0?LW_PLWDEC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2Imem,i);
        if((u%8==0))
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  GPU_REG_RD32(addrssImem));
    }
    dprintf("\n");
    return status;  
}

//-----------------------------------------------------
// lwdecDumpDmem_v01_00 - Dumps LWDEC data memory
//-----------------------------------------------------
LW_STATUS lwdecDumpDmem_v01_00(LwU32 indexGpu, LwU32 engineId, LwU32 dmemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 dmemSizeMax;
    // these are the variables defined for use in parsing and printinf the methods and data
    LwU32 addrss, address2, u, i, comMthdOffs = 0, appMthdOffs = 0, classNum;
    LwU32 comMthd[CMNMETHODARRAYSIZE] = {0};
    LwU32 appMthd[APPMETHODARRAYSIZE] = {0};
    LwU32 methodIdx;

    if (engineId != LWWATCH_LWDEC_0)
        return LW_ERR_NOT_SUPPORTED;

    dmemSizeMax = (GPU_REG_RD_DRF(_PLWDEC_FALCON, _HWCFG, _DMEM_SIZE)<<8) ;

    if(dmemSize > 0)
        dmemSize = min(dmemSizeMax, dmemSize);
    else
       dmemSize = dmemSizeMax;

    addrss      = LW_PLWDEC_FALCON_DMEMD(0);
    address2    = LW_PLWDEC_FALCON_DMEMC(0);
    classNum    = 0xA0B0;

    dprintf("\n");
    dprintf("lw: -- Gpu %u LWDEC DMEM -- \n", indexGpu);
    dprintf("lw: -- Gpu %u LWDEC DMEM SIZE =  0x%08x-- \n", indexGpu,dmemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    

    for(u=0;u<(dmemSize+3)/4;u++)
    {
        i = (u<<(0?LW_PLWDEC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        if((u%8==0))
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  GPU_REG_RD32(addrss));
    }

    // get methods offset are in the DWORD#3 in dmem
    u = (3<<(0?LW_PLWDEC_FALCON_IMEMC_OFFS));
    GPU_REG_WR32(address2,u);
    comMthdOffs = (GPU_REG_RD32(addrss)) >> 2;
    appMthdOffs = comMthdOffs + 16;

    for(u=0; u<CMNMETHODARRAYSIZE;u++)
    {
        i = ((u+comMthdOffs)<<(0?LW_PLWDEC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        comMthd[u] = GPU_REG_RD32(addrss);
        i = ((u+appMthdOffs)<<(0?LW_PLWDEC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        appMthd[u] = GPU_REG_RD32(addrss);
    }

    dprintf("\n\n-----------------------------------------------------------------------\n");
    dprintf("%4s, %8s,    %4s, %8s,    %4s, %8s,    %4s, %8s\n", 
            "Mthd", "Data", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data");
    dprintf("[COMMON METHODS]\n");
    for (u=0; u<CMNMETHODARRAYSIZE; u++)
    {
        dprintf("%04X: %08X", CMNMETHODBASE_LWDEC_v01+4*u, comMthd[u]);
        if (((u % 4) == 3) || u == (CMNMETHODARRAYSIZE - 1))
        {
            dprintf("\n");
        }
        else
        {
            dprintf(",    ");
        }
    }
    dprintf("\n");
    dprintf("\n[APP METHODS]\n");
    for (u=0; u<APPMETHODARRAYSIZE; u++)
    {

        dprintf("%04X: %08X", APPMETHODBASE_LWDEC_v01+4*u, appMthd[u]);
        if (((u % 4) == 3) || u == (APPMETHODARRAYSIZE - 1))
        {
            dprintf("\n");
        }
        else
        {
            dprintf(",    ");
        }
    }

    // common methods
    // if this environment variable is present, parse and print out the methods
    if (getelw("LWW_CLASS_SDK") != NULL)
    {
        dprintf("\n[COMMON METHODS]\n");
        for(u=0;u<16;u++)
        {
            if(parseClassHeader(classNum, CMNMETHODBASE_LWDEC_v01+4*u, comMthd[u]))
                dprintf("\n");
        }
        dprintf("\n");

        // app methods
        dprintf("\n[APP METHODS]\n");
        for(u=0;u<16;u++)
        {
            if(parseClassHeader(classNum, APPMETHODBASE_LWDEC_v01+4*u, appMthd[u]))
                dprintf("\n");
        }
    }
    else
    {
        dprintf("\n[COMMON METHODS]\n");
        for(u=0;u<16;u++)
        {
            for(methodIdx=0;;methodIdx++)
            {
                if(pLwdecMethodTable[methodIdx].m_id == (CMNMETHODBASE_LWDEC_v01+4*u))
                {
                    lwdecPrintMethodData_v01_00(40,
                                                pLwdecMethodTable[methodIdx].m_tag, 
                                                pLwdecMethodTable[methodIdx].m_id, 
                                                comMthd[u]);
                    break;
                }
                else if (pLwdecMethodTable[methodIdx].m_id == 0)
                {
                    break;
                }
            }
        }
        dprintf("\n");
        // app methods
        dprintf("\n[APP METHODS]\n");
        for(u=0;u<16;u++)
        {
            for(methodIdx=0;;methodIdx++)
            {
                if(pLwdecMethodTable[methodIdx].m_id == (APPMETHODBASE_LWDEC_v01+4*u))
                {
                    lwdecPrintMethodData_v01_00(40,
                                                pLwdecMethodTable[methodIdx].m_tag, 
                                                pLwdecMethodTable[methodIdx].m_id, 
                                                appMthd[u]);
                    break;
                }
                else if (pLwdecMethodTable[methodIdx].m_id == 0)
                {
                    break;
                }
            }
        }
        dprintf("\nDefine the LWW_CLASS_SDK environment variable to the location "
                "of the class header files to view parsed methods and data \n");
    }
    return status;  
}

//-----------------------------------------------------
// lwdecTestState_v01_00 - Test basic lwdec state
//-----------------------------------------------------
LW_STATUS lwdecTestState_v01_00(LwU32 indexGpu, LwU32 engineId)
{
    LW_STATUS    status = LW_OK;
    LwU32   regIntr;
    LwU32   regIntrEn;
    LwU32   data32;

    if (engineId != LWWATCH_LWDEC_0)
        return LW_ERR_NOT_SUPPORTED;

    //check falcon interrupts
    regIntr = GPU_REG_RD32(LW_PLWDEC_FALCON_IRQSTAT);
    regIntrEn = GPU_REG_RD32(LW_PLWDEC_FALCON_IRQMASK);
    regIntr &= regIntrEn;

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _GPTMR, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_GPTMR disabled\n");

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _WDTMR, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_WDTMR disabled\n");

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _MTHD, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_MTHD disabled\n");

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _CTXSW, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_CTXSW disabled\n");

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _HALT, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_HALT disabled\n");

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _EXTERR, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_EXTERR disabled\n");

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _SWGEN0, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_SWGEN0 disabled\n");

    if ( !DRF_VAL(_PLWDEC, _FALCON_IRQMASK, _SWGEN1, regIntrEn))
        dprintf("lw: LW_PLWDEC_FALCON_IRQMASK_SWGEN1 disabled\n");

   
    //if any interrupt pending, set error
    if (regIntr != 0)
    {
        addUnitErr("\t LWDEC interrupts are pending\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _GPTMR, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_GPTMR pending\n");

        dprintf("lw: LW_PLWDEC_FALCON_GPTMRINT:    0x%08x\n", 
            GPU_REG_RD32(LW_PLWDEC_FALCON_GPTMRINT) );
        dprintf("lw: LW_PLWDEC_FALCON_GPTMRVAL:    0x%08x\n", 
            GPU_REG_RD32(LW_PLWDEC_FALCON_GPTMRVAL) );
        
    }
    
    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _WDTMR, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_WDTMR pending\n");
    }

    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _MTHD, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_MTHD pending\n");

        dprintf("lw: LW_PLWDEC_FALCON_MTHDDATA_DATA:    0x%08x\n", 
            GPU_REG_RD32(LW_PLWDEC_FALCON_MTHDDATA) );
        
        data32 = GPU_REG_RD32(LW_PLWDEC_FALCON_MTHDID);
        dprintf("lw: LW_PLWDEC_FALCON_MTHDID_ID:    0x%08x\n", 
           DRF_VAL( _PLWDEC,_FALCON_MTHDID, _ID, data32)  );
        dprintf("lw: LW_PLWDEC_FALCON_MTHDID_SUBCH:    0x%08x\n", 
           DRF_VAL( _PLWDEC,_FALCON_MTHDID, _SUBCH, data32)  );
        dprintf("lw: LW_PLWDEC_FALCON_MTHDID_PRIV:    0x%08x\n", 
           DRF_VAL( _PLWDEC,_FALCON_MTHDID, _PRIV, data32)  );
    }
    
    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _CTXSW, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_CTXSW pending\n");
    }
    
    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _HALT, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_HALT pending\n");
    }
    
    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _EXTERR, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_EXTERR pending\n");
    }
    
    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _SWGEN0, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_SWGEN0 pending\n");

        pFalcon[indexGpu].falconPrintMailbox(LW_FALCON_LWDEC_BASE);
    }

    if ( DRF_VAL( _PLWDEC,_FALCON_IRQSTAT, _SWGEN1, regIntr))
    {
        dprintf("lw: LW_PLWDEC_FALCON_IRQSTAT_SWGEN1 pending\n");
    }

     //
    //print falcon states
    //Bit |  Signal meaning
    //0      FALCON busy
    //

    data32 = GPU_REG_RD32(LW_PLWDEC_FALCON_IDLESTATE);

    if ( DRF_VAL( _PLWDEC, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_IDLESTATE_FALCON_BUSY\n");
        addUnitErr("\t LW_PLWDEC_FALCON_IDLESTATE_FALCON_BUSY\n");
        status = LW_ERR_GENERIC;
    }

  
    data32 = GPU_REG_RD32(LW_PLWDEC_FALCON_FHSTATE);
 
    if ( DRF_VAL( _PLWDEC, _FALCON_FHSTATE, _FALCON_HALTED, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_FHSTATE_FALCON_HALTED\n");
        addUnitErr("\t LW_PLWDEC_FALCON_FHSTATE_FALCON_HALTED\n");
        status = LW_ERR_GENERIC;
    }
    
    if ( DRF_VAL( _PLWDEC, _FALCON_FHSTATE, _ENGINE_FAULTED, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        addUnitErr("\t LW_PLWDEC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        status = LW_ERR_GENERIC;
    }
    
    if ( DRF_VAL( _PLWDEC, _FALCON_FHSTATE, _STALL_REQ, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_FHSTATE_STALL_REQ\n");
        addUnitErr("\t LW_PLWDEC_FALCON_FHSTATE_STALL_REQ\n");
        status = LW_ERR_GENERIC;
    }

    //print falcon ctl regs
    data32 = GPU_REG_RD32(LW_PLWDEC_FALCON_ENGCTL);
    
    if ( DRF_VAL( _PLWDEC, _FALCON_ENGCTL, _ILW_CONTEXT, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_ENGCTL_ILW_CONTEXT\n");
        addUnitErr("\t LW_PLWDEC_FALCON_ENGCTL_ILW_CONTEXT\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWDEC, _FALCON_ENGCTL, _STALLREQ, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_ENGCTL_STALLREQ\n");
        addUnitErr("\t LW_PLWDEC_FALCON_ENGCTL_STALLREQ\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PLWDEC_FALCON_CPUCTL);

    if ( DRF_VAL( _PLWDEC, _FALCON_CPUCTL, _IILWAL, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_CPUCTL_IILWAL\n");
        addUnitErr("\t LW_PLWDEC_FALCON_CPUCTL_IILWAL\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWDEC, _FALCON_CPUCTL, _HALTED, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_CPUCTL_HALTED\n");
        addUnitErr("\t LW_PLWDEC_FALCON_CPUCTL_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWDEC, _FALCON_CPUCTL, _STOPPED, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_CPUCTL_STOPPED\n");
        addUnitErr("\t Warning: LW_PLWDEC_FALCON_CPUCTL_STOPPED\n");
        //status = LW_ERR_GENERIC;
    }

    // state of mthd/ctx interface 
    data32 = GPU_REG_RD32(LW_PLWDEC_FALCON_ITFEN);

    if (DRF_VAL( _PLWDEC, _FALCON_ITFEN, _CTXEN, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_ITFEN_CTXEN enabled\n");
             
        if (pFalcon[indexGpu].falconTestCtxState(LW_FALCON_LWDEC_BASE, "PLWDEC") == LW_ERR_GENERIC)
        {
            dprintf("lw: Current ctx state invalid\n");
            addUnitErr("\t Current ctx state invalid\n");
            status = LW_ERR_GENERIC;
        }
        else
        {
            dprintf("lw: Current ctx state valid\n");
        }
    }
    else
    {
        dprintf("lw: + LW_PLWDEC_FALCON_ITFEN_CTXEN disabled\n");
    }

    if ( DRF_VAL( _PLWDEC, _FALCON_ITFEN, _MTHDEN, data32))
    {
        dprintf("lw: + LW_PLWDEC_FALCON_ITFEN_MTHDEN enabled\n");
    }
    else
    {
        dprintf("lw: + LW_PLWDEC_FALCON_ITFEN_MTHDEN disabled\n");
    }

    //check if falcon is hung (instr ptr)
    if ( pFalcon[indexGpu].falconTestPC(LW_FALCON_LWDEC_BASE, "PLWDEC") == LW_ERR_GENERIC )
    {
        dprintf("lw: Falcon instruction pointer is stuck or invalid\n");
        
        //TODO: treat falcon PC errors as warnings now, need to report as error
        addUnitErr("\t Warning: Falcon instruction pointer is stuck or invalid\n");
        //status = LW_ERR_GENERIC;
    }

    return status;  
}

//-----------------------------------------------------
// lwdecPrintPriv_v01_00
//-----------------------------------------------------
void lwdecPrintPriv_v01_00(LwU32 clmn, char *tag, LwU32 id)
{
    size_t len = strlen(tag);
    
    dprintf("lw: %s",tag);

    if((len>0)&&(len<(clmn+4)))
    {
        LwU32 i;
        for(i=0;i<clmn-len;i++)
        {
            dprintf(" ");
        }
    }
    dprintf("(0x%08X)  = 0x%08X\n",id,GPU_REG_RD32(id));
}

//-----------------------------------------------------
// lwdecDumpPriv_v01_00 - Dumps LWDEC priv reg space
//-----------------------------------------------------
LW_STATUS lwdecDumpPriv_v01_00(LwU32 indexGpu, LwU32 engineId)
{
    LwU32 u;

    if (engineId != LWWATCH_LWDEC_0)
        return LW_ERR_NOT_SUPPORTED;

    if (!pLwdecPrivReg[engineId])
    {
        dprintf("lw: -- Gpu %u LWDEC error: priv reg array uninitialized\n", indexGpu);
        return LW_ERR_ILWALID_PARAMETER;
    }

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWDEC priv registers -- \n", indexGpu);
    dprintf("lw:\n");

    for(u=0;;u++)
    {
        if(pLwdecPrivReg[engineId][u].m_id==0)
        {
            break;
        }
        pLwdec[indexGpu].lwdecPrintPriv(61,pLwdecPrivReg[engineId][u].m_tag,pLwdecPrivReg[engineId][u].m_id);
    }
    return LW_OK; 
}

//--------------------------------------------------------
// lwdecDisplayHwcfg_v01_00 - Display LWDEC HW config
//--------------------------------------------------------
LW_STATUS lwdecDisplayHwcfg_v01_00(LwU32 indexGpu, LwU32 engineId)
{
    LwU32 hwcfg, hwcfg1;

    if (engineId != LWWATCH_LWDEC_0)
        return LW_ERR_NOT_SUPPORTED;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWDEC HWCFG -- \n", indexGpu);
    dprintf("lw:\n");

    hwcfg  = GPU_REG_RD32(LW_PLWDEC_FALCON_HWCFG);
    dprintf("lw: LW_PLWDEC_FALCON_HWCFG:  0x%08x\n", hwcfg); 
    dprintf("lw:\n");
    dprintf("lw:  IMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",
            DRF_VAL(_PLWDEC, _FALCON_HWCFG, _IMEM_SIZE, hwcfg),
            DRF_VAL(_PLWDEC, _FALCON_HWCFG, _IMEM_SIZE, hwcfg)<<8); 
    dprintf("lw:  DMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",
            DRF_VAL(_PLWDEC, _FALCON_HWCFG, _DMEM_SIZE, hwcfg), 
            DRF_VAL(_PLWDEC, _FALCON_HWCFG, _DMEM_SIZE, hwcfg)<<8); 
    dprintf("lw:  METHODFIFO_DEPTH: 0x%08X\n", DRF_VAL(_PLWDEC, _FALCON_HWCFG, _METHODFIFO_DEPTH, hwcfg)); 
    dprintf("lw:  DMAQUEUE_DEPTH:   0x%08X\n", DRF_VAL(_PLWDEC, _FALCON_HWCFG, _DMAQUEUE_DEPTH, hwcfg)); 

    dprintf("lw:\n");

    hwcfg1 = GPU_REG_RD32(LW_PLWDEC_FALCON_HWCFG1);
    dprintf("lw: LW_PLWDEC_FALCON_HWCFG1: 0x%08x\n", hwcfg1); 
    dprintf("lw:\n");
    dprintf("lw:  CORE_REV:         0x%08X\n", DRF_VAL(_PLWDEC, _FALCON_HWCFG1, _CORE_REV, hwcfg1)); 
    dprintf("lw:  SELWRITY_MODEL:   0x%08X\n", DRF_VAL(_PLWDEC, _FALCON_HWCFG1, _SELWRITY_MODEL, hwcfg1)); 
    dprintf("lw:  IMEM_PORTS:       0x%08X\n", DRF_VAL(_PLWDEC, _FALCON_HWCFG1, _IMEM_PORTS, hwcfg1)); 
    dprintf("lw:  DMEM_PORTS:       0x%08X\n", DRF_VAL(_PLWDEC, _FALCON_HWCFG1, _DMEM_PORTS, hwcfg1)); 
    dprintf("lw:  TAG_WIDTH:        0x%08X\n", DRF_VAL(_PLWDEC, _FALCON_HWCFG1, _TAG_WIDTH, hwcfg1)); 

    return LW_OK;  
}

 /*
 Prints Falcon's Special purpose registers
0   IV0
1   IV1
3   EV
4   SP
5   PC
6   IMB
7   DMB
8   CSW
*/
// indx taken from Falcon 4.0 arch Table 3
LW_STATUS  lwdecDisplayFlcnSPR_v01_00(LwU32 indexGpu, LwU32 engineId)
{
    if (engineId != LWWATCH_LWDEC_0)
        return LW_ERR_NOT_SUPPORTED;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWDEC Special Purpose Registers -- \n", indexGpu);
    dprintf("lw:\n");

    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD, 0x1008);
    dprintf("lw: LWDEC IV0 :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA)); 
    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD, 0x1108);
    dprintf("lw: LWDEC IV1 :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA)); 
    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD, 0x1308);
    dprintf("lw: LWDEC EV  :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA)); 
    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD, 0x1408);
    dprintf("lw: LWDEC SP  :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA)); 
    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD, 0x1508);
    dprintf("lw: LWDEC PC  :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA)); 
    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD, 0x1608);
    dprintf("lw: LWDEC IMB :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA)); 
    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD, 0x1708);
    dprintf("lw: LWDEC DMB :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA)); 
    GPU_REG_WR32(LW_PLWDEC_FALCON_ICD_CMD, 0x1808);
    dprintf("lw: LWDEC CSW :    0x%08x\n", GPU_REG_RD32(LW_PLWDEC_FALCON_ICD_RDATA)); 
    dprintf("lw:\n\n");

    return LW_OK; 
}

/*!
 * @brief Checks if LWDEC DEBUG fuse is blown or not
 *
 */
LwBool
lwdecIsDebugMode_v01_00(LwU32 engineId)
{
    LwU32 ctlStat =  GPU_REG_RD32(LW_PLWDEC_SCP_CTL_STAT);

    if (engineId != LWWATCH_LWDEC_0)
        return FALSE;

    return !FLD_TEST_DRF(_PLWDEC, _SCP_CTL_STAT, _DEBUG_MODE, _DISABLED, ctlStat);
}

