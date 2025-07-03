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
// lwdec0101.c - LWDEC v1 routines
// 
//-----------------------------------------------------

#include "maxwell/gm200/dev_pri_ringstation_sys.h"
#include "maxwell/gm200/dev_lwdec_pri.h"
#include "maxwell/gm200/dev_falcon_v4.h"
#include "maxwell/gm200/dev_fifo.h"

#include "lwdec.h"

#include "g_lwdec_private.h"     // (rmconfig)  implementation prototypes

#include "class/clb0b0.h"

dbg_lwdec_v01_01 lwdecMethodTable_v01_01[] =
{
    privInfo_lwdec_v01_01(LWB0B0_NOP),
    privInfo_lwdec_v01_01(LWB0B0_PM_TRIGGER),
    privInfo_lwdec_v01_01(LWB0B0_SET_APPLICATION_ID),
    privInfo_lwdec_v01_01(LWB0B0_SET_WATCHDOG_TIMER),
    privInfo_lwdec_v01_01(LWB0B0_SEMAPHORE_A),
    privInfo_lwdec_v01_01(LWB0B0_SEMAPHORE_B),
    privInfo_lwdec_v01_01(LWB0B0_SEMAPHORE_C),
    privInfo_lwdec_v01_01(LWB0B0_EXELWTE),
    privInfo_lwdec_v01_01(LWB0B0_SEMAPHORE_D),
    privInfo_lwdec_v01_01(LWB0B0_SET_PREDICATION_OFFSET_UPPER),
    privInfo_lwdec_v01_01(LWB0B0_SET_PREDICATION_OFFSET_LOWER),
    privInfo_lwdec_v01_01(LWB0B0_SET_CONTROL_PARAMS),
    privInfo_lwdec_v01_01(LWB0B0_SET_DRV_PIC_SETUP_OFFSET),
    privInfo_lwdec_v01_01(LWB0B0_SET_IN_BUF_BASE_OFFSET),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_INDEX),
    privInfo_lwdec_v01_01(LWB0B0_SET_SLICE_OFFSETS_BUF_OFFSET),
    privInfo_lwdec_v01_01(LWB0B0_SET_COLOC_DATA_OFFSET),
    privInfo_lwdec_v01_01(LWB0B0_SET_HISTORY_OFFSET),
    privInfo_lwdec_v01_01(LWB0B0_SET_DISPLAY_BUF_SIZE),
    privInfo_lwdec_v01_01(LWB0B0_SET_HISTOGRAM_OFFSET),
    privInfo_lwdec_v01_01(LWB0B0_SET_LWDEC_STATUS_OFFSET),
    privInfo_lwdec_v01_01(LWB0B0_SET_DISPLAY_BUF_LUMA_OFFSET),
    privInfo_lwdec_v01_01(LWB0B0_SET_DISPLAY_BUF_CHROMA_OFFSET),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET0),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET1),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET2),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET3),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET4),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET5),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET6),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET7),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET8),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET9),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET10),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET11),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET12),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET13),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET14),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET15),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_LUMA_OFFSET16),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET0),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET1),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET2),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET3),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET4),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET5),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET6),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET7),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET8),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET9),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET10),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET11),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET12),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET13),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET14),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET15),
    privInfo_lwdec_v01_01(LWB0B0_SET_PICTURE_CHROMA_OFFSET16),
    privInfo_lwdec_v01_01(LWB0B0_H264_SET_MBHIST_BUF_OFFSET),
    privInfo_lwdec_v01_01(LWB0B0_VP8_SET_PROB_DATA_OFFSET),
    privInfo_lwdec_v01_01(LWB0B0_VP8_SET_HEADER_PARTITION_BUF_BASE_OFFSET),
    privInfo_lwdec_v01_01(LWB0B0_SET_CONTENT_INITIAL_VECTOR(0)),
    privInfo_lwdec_v01_01(LWB0B0_SET_CONTENT_INITIAL_VECTOR(1)),
    privInfo_lwdec_v01_01(LWB0B0_SET_CONTENT_INITIAL_VECTOR(2)),
    privInfo_lwdec_v01_01(LWB0B0_SET_CONTENT_INITIAL_VECTOR(3)),
    privInfo_lwdec_v01_01(LWB0B0_SET_CTL_COUNT),
    privInfo_lwdec_v01_01(LWB0B0_SET_UPPER_SRC),
    privInfo_lwdec_v01_01(LWB0B0_SET_LOWER_SRC),
    privInfo_lwdec_v01_01(LWB0B0_SET_UPPER_DST),
    privInfo_lwdec_v01_01(LWB0B0_SET_LOWER_DST),
    privInfo_lwdec_v01_01(LWB0B0_SET_BLOCK_COUNT),
    privInfo_lwdec_v01_01(LWB0B0_SET_SESSION_KEY(0)),
    privInfo_lwdec_v01_01(LWB0B0_SET_SESSION_KEY(1)),
    privInfo_lwdec_v01_01(LWB0B0_SET_SESSION_KEY(2)),
    privInfo_lwdec_v01_01(LWB0B0_SET_SESSION_KEY(3)),
    privInfo_lwdec_v01_01(LWB0B0_SET_CONTENT_KEY(0)),
    privInfo_lwdec_v01_01(LWB0B0_SET_CONTENT_KEY(1)),
    privInfo_lwdec_v01_01(LWB0B0_SET_CONTENT_KEY(2)),
    privInfo_lwdec_v01_01(LWB0B0_SET_CONTENT_KEY(3)),
    privInfo_lwdec_v01_01(LWB0B0_PM_TRIGGER_END),
    privInfo_lwdec_v01_01(0),
};

dbg_lwdec_v01_01 lwdecPrivReg_v01_01[] =
{
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMEM_PRIV_LEVEL_MASK),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEM_PRIV_LEVEL_MASK),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEM_DUMMY),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMEM_DUMMY__PRIV_LEVEL_MASK),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_CPUCTL_PRIV_LEVEL_MASK),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_EXE_PRIV_LEVEL_MASK),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQTMR_PRIV_LEVEL_MASK),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_MTHDCTX_PRIV_LEVEL_MASK),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_SCTL_PRIV_LEVEL_MASK),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_SCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_SCTL__PRIV_LEVEL_MASK),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_SCTL1),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_SCTL1__PRIV_LEVEL_MASK),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DBGCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQSSET),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQSCLR),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQSTAT),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQMODE),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQMSET),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQMCLR),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQMASK),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQDEST),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IRQDEST2),
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
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_EXCI),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_SVEC_SPR),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_RSTAT0),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_RSTAT3),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_CPUCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_CPUCTL_ALIAS),
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
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMAINFO_FINISHED_FBRD_LOW),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMAINFO_FINISHED_FBRD_HIGH),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMAINFO_FINISHED_FBWR_LOW),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMAINFO_FINISHED_FBWR_HIGH),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMAINFO_LWRRENT_FBRD_LOW),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMAINFO_LWRRENT_FBRD_HIGH),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMAINFO_LWRRENT_FBWR_LOW),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMAINFO_LWRRENT_FBWR_HIGH),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_DMAINFO_CTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMSTAT),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_TRACEIDX),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_TRACEPC),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMFILLRNG0),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMFILLRNG1),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_IMFILLCTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_CG2),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_CSBERRSTAT),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_CSBERR_INFO),
    privInfo_lwdec_v01_01(LW_PLWDEC_FALCON_CSBERR_ADDR),
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
    privInfo_lwdec_v01_01(LW_PLWDEC_MCLUMSHIFT(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCLUMSHIFT(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MCLUMSHIFT(2)),
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
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_SLICE_INFO_EXT),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_EDOB_MB_INFO),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_SLICE_INFO_SVC),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_SLICE_INFO_SVC_CROP),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_EDOB_MB_POS),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_MB_INFO),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_MB_POS),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_PARSE_CMD),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_STATUS),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_RESULT),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_INTRPT_EN),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_INTRPT_STATUS),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_BIT_ERROR_CODE),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_BIT_ERROR_CODE_CODE_VC1_MB_MBMODE),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_BIT_ERROR_CODE_CODE_MP4_MB_MCBPY),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_BIT_ERROR_CODE_CODE_H264_ILWALID_VLC_RUN_BEFORE),
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
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_CBC_DEBUG_DATA),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_DEBUG),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_VC1_DEBUG0),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_VC1_DEBUG1),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_MPG_DEBUG),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_BIT_ERROR_MASK),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_SLICE_LIMIT),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_SLICE_ERROR_CTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_MPEG4_DEBUG),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_MPEG4_BITPLANE_MEMORY(0,0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_MPEG4_BITPLANE_MEMORY(1,0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_MPEG4_BITPLANE_MEMORY(2,0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_ENTROPYDEC_MPEG4_BITPLANE_MEMORY(3,0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_PROBBUF_WRITE_REG),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_PROBBUF_READ_REG),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_PROBBUF_RW_OFFSET),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_A(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_A(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_A(2)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_A(3)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_A(4)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_A(5)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_A(6)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_A(7)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_B(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_B(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_B(2)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_B(3)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_B(4)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_B(5)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_B(6)),
    privInfo_lwdec_v01_01(LW_PLWDEC_VLD_IDMA1_BUFF_B(7)),
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
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_HISTOGRAM_RESULT_OFFSET),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_DITHER_MATRIX_WRITE_DATA),
    privInfo_lwdec_v01_01(LW_PLWDEC_DBFDMA_DITHER_MATRIX_WRITE_OFFSET),
    privInfo_lwdec_v01_01(LW_PLWDEC_HIST_CFG),
    privInfo_lwdec_v01_01(LW_PLWDEC_HIST_CTL),
    privInfo_lwdec_v01_01(LW_PLWDEC_HIST_BUF_START),
    privInfo_lwdec_v01_01(LW_PLWDEC_HIST_BUF_SIZE),
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
    privInfo_lwdec_v01_01(LW_PLWDEC_IQTQMEMCTRL),
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
    privInfo_lwdec_v01_01(LW_PLWDEC_MV_MB_COUNTER),
    privInfo_lwdec_v01_01(LW_PLWDEC_MV_MB_COUNTER1),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVDBG_STATE(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_MVSFCLAYOUT),
    privInfo_lwdec_v01_01(LW_PLWDEC_SA_CONTROL(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_SA_CONTROL(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_SA_START(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_SA_START(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_SA_INST_ARRAY(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_SA_INST_ARRAY(1)),
    privInfo_lwdec_v01_01(LW_PLWDEC_SA_INST_ARRAY(2)),
    privInfo_lwdec_v01_01(LW_PLWDEC_SA_INST_ARRAY(3)),
    privInfo_lwdec_v01_01(LW_PLWDEC_SA_STATUS(0)),
    privInfo_lwdec_v01_01(LW_PLWDEC_SA_STATUS(1)),
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
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_ACL_KEYABLE),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_ACL_WRITE),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_STATUS),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_STAT0),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_STAT1),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNG_STAT0),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_RNG_STAT1),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_INTR),
    privInfo_lwdec_v01_01(LW_PLWDEC_SCP_INTR_EN),
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
// lwdecIsSupported_v01_01
//-----------------------------------------------------
BOOL lwdecIsSupported_v01_01(LwU32 indexGpu , LwU32 engineId)
{
    if (engineId != LWWATCH_LWDEC_0)
        return FALSE;

    pLwdecPrivReg[engineId] = lwdecPrivReg_v01_01;
    pLwdecMethodTable = lwdecMethodTable_v01_01;
    return TRUE;
}


//-----------------------------------------------------
// lwdecDumpDmem_v01_01 - Dumps LWDEC data memory
//-----------------------------------------------------
LW_STATUS lwdecDumpDmem_v01_01(LwU32 indexGpu, LwU32 engineId, LwU32 dmemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 dmemSizeMax;
    // these are the variables defined for use in parsing and printinf the methods and data
    LwU32 addrss, address2, u, i, comMthdOffs = 0, appMthdOffs = 0, classNum;
    LwU32 comMthd[CMNMETHODARRAYSIZEB0B0] = {0};
    LwU32 appMthd[APPMETHODARRAYSIZEB0B0] = {0};
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
    classNum    = 0xB0B0;

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
    appMthdOffs = comMthdOffs + CMNMETHODARRAYSIZEB0B0;

    for(u=0; u<CMNMETHODARRAYSIZEB0B0;u++)
    {
        i = ((u+comMthdOffs)<<(0?LW_PLWDEC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        comMthd[u] = GPU_REG_RD32(addrss);
    }

    for(u=0; u<APPMETHODARRAYSIZEB0B0;u++)
    {
        i = ((u+appMthdOffs)<<(0?LW_PLWDEC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        appMthd[u] = GPU_REG_RD32(addrss);
    }

    dprintf("\n\n-----------------------------------------------------------------------\n");
    dprintf("%4s, %8s,    %4s, %8s,    %4s, %8s,    %4s, %8s\n", 
            "Mthd", "Data", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data");
    dprintf("[COMMON METHODS]\n");
    for (u=0; u<CMNMETHODARRAYSIZEB0B0; u++)
    {
        dprintf("%04X: %08X", CMNMETHODBASE_LWDEC_v01+4*u, comMthd[u]);
        if (((u % 4) == 3) || u == (CMNMETHODARRAYSIZEB0B0 - 1))
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
    for (u=0; u<APPMETHODARRAYSIZEB0B0; u++)
    {
        dprintf("%04X: %08X", APPMETHODBASE_LWDEC_v01+4*u, appMthd[u]);
        if (((u % 4) == 3) || u == (APPMETHODARRAYSIZEB0B0 - 1))
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
        for(u=0;u<CMNMETHODARRAYSIZEB0B0;u++)
        {
            if(parseClassHeader(classNum, CMNMETHODBASE_LWDEC_v01+4*u, comMthd[u]))
                dprintf("\n");
        }
        dprintf("\n");

        // app methods
        dprintf("\n[APP METHODS]\n");
        for(u=0;u<APPMETHODARRAYSIZEB0B0;u++)
        {
            if(parseClassHeader(classNum, APPMETHODBASE_LWDEC_v01+4*u, appMthd[u]))
                dprintf("\n");
        }
    }
    else
    {
        dprintf("\n[COMMON METHODS]\n");
        for(u=0;u<CMNMETHODARRAYSIZEB0B0;u++)
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
        for(u=0;u<APPMETHODARRAYSIZEB0B0;u++)
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
