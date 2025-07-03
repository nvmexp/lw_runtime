/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//-----------------------------------------------------
//
// msenc0500.c - LWENC routines
// 
//-----------------------------------------------------

#include "maxwell/gm200/dev_lwenc_pri_sw.h"
#include "maxwell/gm200/dev_falcon_v4.h"
#include "class/cld0b7.h"

#include "msenc.h"
#include "hwref/lwutil.h"
#include "g_msenc_private.h"     // (rmconfig)  implementation prototypes

#define USE_LWENC_5_0

#if defined(USE_LWENC_5_0)

dbg_msenc_v01_01 msencMethodTable_v05_00[] =
{
    privInfo_msenc_v01_01(LWD0B7_NOP),
    privInfo_msenc_v01_01(LWD0B7_PM_TRIGGER),
    privInfo_msenc_v01_01(LWD0B7_SET_APPLICATION_ID),
    privInfo_msenc_v01_01(LWD0B7_SET_WATCHDOG_TIMER),
    privInfo_msenc_v01_01(LWD0B7_SEMAPHORE_A),
    privInfo_msenc_v01_01(LWD0B7_SEMAPHORE_B),
    privInfo_msenc_v01_01(LWD0B7_SEMAPHORE_C),
    privInfo_msenc_v01_01(LWD0B7_EXELWTE),
    privInfo_msenc_v01_01(LWD0B7_SEMAPHORE_D),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC0),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC1),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC2),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC3),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC4),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC5),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC6),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC7),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC8),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC9),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC10),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC11),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC12),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC13),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC14),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC15),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC_LAST),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC_GOLDEN),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_REF_PIC_ALTREF),
    privInfo_msenc_v01_01(LWD0B7_SET_UCODE_STATE),
    privInfo_msenc_v01_01(LWD0B7_SET_IO_VP8_ENC_STATUS),
    privInfo_msenc_v01_01(LWD0B7_SET_OUT_BITSTREAM_RES),
    privInfo_msenc_v01_01(LWD0B7_SET_CONTROL_PARAMS),
    privInfo_msenc_v01_01(LWD0B7_SET_PICTURE_INDEX),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_RCDATA),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_DRV_PIC_SETUP),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_CEAHINTS_DATA),
    privInfo_msenc_v01_01(LWD0B7_SET_OUT_ENC_STATUS),
    privInfo_msenc_v01_01(LWD0B7_SET_OUT_BITSTREAM),
    privInfo_msenc_v01_01(LWD0B7_SET_IOHISTORY),
    privInfo_msenc_v01_01(LWD0B7_SET_IO_RC_PROCESS),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_COLOC_DATA),
    privInfo_msenc_v01_01(LWD0B7_SET_OUT_COLOC_DATA),
    privInfo_msenc_v01_01(LWD0B7_SET_OUT_REF_PIC),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_LWR_PIC),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_MEPRED_DATA),
    privInfo_msenc_v01_01(LWD0B7_SET_OUT_MEPRED_DATA),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_LWR_PIC_CHROMA_U),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_LWR_PIC_CHROMA_V),
    privInfo_msenc_v01_01(LWD0B7_SET_IN_QP_MAP),
    privInfo_msenc_v01_01(LWD0B7_PM_TRIGGER_END),
    privInfo_msenc_v01_01(0),
};

dbg_msenc_v01_01 msenc0PrivReg_v05_00[] =
{
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEM_PRIV_LEVEL_MASK(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEM_PRIV_LEVEL_MASK(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEM_DUMMY(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CPUCTL_PRIV_LEVEL_MASK(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_EXE_PRIV_LEVEL_MASK(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQTMR_PRIV_LEVEL_MASK(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MTHDCTX_PRIV_LEVEL_MASK(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_SCTL_PRIV_LEVEL_MASK(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_SCTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_SCTL1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DBGCTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQSSET(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQSCLR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQSTAT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQMODE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQMSET(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQMCLR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQMASK(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQDEST(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQDEST2(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQSCMASK(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_GPTMRINT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_GPTMRVAL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_GPTMRCTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_PTIMER0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_PTIMER1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_WDTMRVAL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_WDTMRCTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MTHDDATA(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MTHDID(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MTHDWDAT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MTHDCOUNT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MTHDPOP(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MTHDRAMSZ(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_LWRCTX(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_NXTCTX(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CTXACK(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MAILBOX0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MAILBOX1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_ITFEN(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IDLESTATE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_FHSTATE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_PRIVSTATE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_SFTRESET(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_OS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_RM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_SOFT_PM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_SOFT_MODE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DEBUG1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DEBUGINFO(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IBRKPT1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IBRKPT2(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IBRKPT3(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IBRKPT4(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IBRKPT5(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CGCTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_ENGCTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_PMM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_ADDR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_EXCI(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_SVEC_SPR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_RSTAT0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_RSTAT3(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CPUCTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CPUCTL_ALIAS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_BOOTVEC(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_HWCFG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_HWCFG1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMACTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMATRFBASE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMATRFMOFFS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMATRFCMD(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMATRFFBOFFS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAPOLL_FB(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAPOLL_CP(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_FINISHED_FBRD_LOW(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_FINISHED_FBRD_HIGH(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_FINISHED_FBWR_LOW(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_FINISHED_FBWR_HIGH(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_LWRRENT_FBRD_LOW(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_LWRRENT_FBRD_HIGH(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_LWRRENT_FBWR_LOW(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_LWRRENT_FBWR_HIGH(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_CTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMCTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMSTAT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_TRACEIDX(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_TRACEPC(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CG2(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CSBERRSTAT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CSBERR_INFO(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CSBERR_ADDR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_ICD_CMD(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_ICD_ADDR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_ICD_WDATA(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_ICD_RDATA(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMC(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMC(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMC(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMC(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMD(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMD(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMD(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMD(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMT(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMT(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMT(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMT(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(0,4)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(0,5)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(0,6)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(0,7)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(0,4)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(0,5)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(0,6)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(0,7)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_REGIONCFG_PRIV_LEVEL_MASK(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(0,4)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(0,5)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(0,6)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(0,7)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_INSTBLK(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_CTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_THROTTLE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_REGIONCFG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_ACHK_BLK(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_ACHK_BLK(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_ACHK_CTL(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_ACHK_CTL(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_DBG_STAT0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_DBG_STAT1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_DBG_STAT2(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_DBG_RDCOUNT_LO(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_DBG_RDCOUNT_HI(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_DBG_WRCOUNT_LO(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_DBG_WRCOUNT_HI(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RMVP_BASE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RMVP_STRIDE_IN_MB(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RMVP_MB_WIDTH(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RMVP_MB_HEIGHT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RMVP_CTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RMVP_DBG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_WMVP_BASE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_WMVP_CTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_WMVP_STRIDE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_WMVP_MB_WIDTH(0)),
    privInfo_msenc_v01_01(LW_PLWENC_WMVP_MB_HEIGHT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_WMVP_DBG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FRAME_CONFIG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_DELTA_PIC_ORDER_CNT_BOTTOM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FRAME_SIZE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_SEQ_PARAMETERS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_FRAME_ID(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PIC_PARAMETERS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PIC_PARAM_SET_ID(0)),
    privInfo_msenc_v01_01(LW_PLWENC_NUM_REF_IDX_ACTIVE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_IDR_PIC_ID(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FRAME_CTRL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_MISC_CTRL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_INTERNAL_BIAS_MULTIPLIER(0)),
    privInfo_msenc_v01_01(LW_PLWENC_QPP_CTRL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_QPP_8X8_VECTOR0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_QPP_8X8_VECTOR1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_IPCM_CTRL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_SLICE_CTRL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_SLICE_STAT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MB_CNT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA_BASE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA_BASEL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA_SIZE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA_OFFSET(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA_CTRL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_COLOC_WRDMA_ACTIVE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_COLOC_WRDMA_BASE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_INTR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_INTEN(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_INTR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_INTEN(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_LUMA_BASE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_CHROMA_BASE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HH_L0_REFPIC_MAP_0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HH_L0_REFPIC_MAP_1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HH_L1_REFPIC_MAP_0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HH_L1_REFPIC_MAP_1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(0,4)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(0,5)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(0,6)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(0,7)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(0,4)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(0,5)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(0,6)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(0,7)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_C4X4(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(0,4)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(0,5)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(0,6)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(0,7)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(0,4)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(0,5)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(0,6)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(0,7)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_CTRL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTRA_RESIDUAL_BITS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTER_RESIDUAL_BITS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTRA_MODE_BITS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTER_MODE_BITS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_IPCM_BITS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_TOTAL_BITS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_TOTAL_CABAC_BINS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_LWM_INTRA_COST(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_LWM_INTER_COST(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_LWM_IPCM_COST(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_LWM_SKIP_COST(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTRA_MB_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTER_MB_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_SKIP_MB_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_IPCM_MB_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_TYPE1_BIT_COUNT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_DBF_DMA_CTRL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_PIPE_STATUS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_DEBUG0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_DEBUG1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_DEBUG2(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_INTRA_SAT_CTRL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_CMD_CONFIG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(0,4)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(0,5)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(0,6)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(0,7)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(0,4)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(0,5)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(0,6)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(0,7)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(0,4)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(0,5)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(0,6)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(0,7)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MVC_NAL_HEADER(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_DEBUG_IF(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_DEBUG_STAT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_DEBUG_H264(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_INTER_SAT_CTRL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_FRAME_SIZE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_FRAME_PARAM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_SEG_QINDEX(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_SEG_FLTLVL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_LOOPFILTER(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_LOOPFILTER_REF_DELTA(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_LOOPFILTER_MODE_DELTA(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_QUANT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_REFFRAME(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_MPECRAM_LWRR_OFFSET(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_MPECRAM_ADDR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_SEG_PROBS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_YMODE_PROBS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_UVMODE_PROBS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_REF_PROBS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_MV_PROBS_UPD_Y(0)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_MV_PROBS_UPD_X(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA1_BASE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA1_BASEL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA1_SIZE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA1_OFFSET(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_STATE_BASE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_STATE_BASEL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_STATE_SIZE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_STATE_OFFSET(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA_ACK_OFFSET(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y1(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y1(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y1(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y1(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y2(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y2(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y2(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y2(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_UV(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_UV(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_UV(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_UV(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_DC(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_DC(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_DC(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_DC(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_AC(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_AC(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_AC(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_AC(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_LAMBDA(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_LAMBDA(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_FLTLVL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_COSTRAM_ADDR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_COSTRAM_DATA(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_H264_STAT_SLICE_BITS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_CHROMA_V_BASE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_STAT_TOTAL_INTRA_COST(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_SEQ_PARAMETERS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_PIC_PARAMETERS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_FRAME_CTRL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_REF_PIC_LIST_MODIFICATION_L0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_REF_PIC_LIST_MODIFICATION_L1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_SHORT_TERM_REF_PIC_SET(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_ENTRY_POINT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_TOTAL_INTRA_COST(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_TOTAL_INTER_COST(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTRA_LW32X32_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTER_LW32X32_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_SKIP_LW32X32_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_IPCM_LW32X32_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTRA_LW16X16_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTER_LW16X16_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_SKIP_LW16X16_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_IPCM_LW16X16_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTRA_LW8X8_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTER_LW8X8_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_SKIP_LW8X8_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_IPCM_LW8X8_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_STRPS_DELTA_POCS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_STRPS_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_STRPS_USED_BY_LWRR_PIC(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_LTRPS_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_LTRPS_FLAG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_REF_PIC_DATA(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_PREDICTOR_SOURCE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_GLOBAL_L0_MV(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_GLOBAL_L1_MV(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_PIC_SIZE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STATUS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SLICE_CONTROL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SRF_SLICE_CONTROL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_PIC_CONTROL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SPS_CONFIG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_CONSTANT_L0_PREDICTOR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_CONSTANT_L1_PREDICTOR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_LAMBDA(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_LONGTERM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_DBG_SEL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_DBG_STAT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SLICE_CONFIG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MV_BOUNDARY0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MV_BOUNDARY1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SELF_SCALE_L0(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SELF_SCALE_L0(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SELF_SCALE_L1(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SELF_SCALE_L1(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,4)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,5)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,6)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,7)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,8)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,9)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,10)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,11)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,12)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,13)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,14)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,15)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,16)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,17)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,18)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,19)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,20)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,21)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,22)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,23)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,24)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,25)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,26)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,27)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,28)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,29)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,30)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(0,31)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_INTR_STATUS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_INTR_ENABLE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_ICC_MODE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_DIFF_PIC_ORDER_CNT_ZERO(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_DIFF_PIC_ORDER_CNT_ZERO(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(0,4)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(0,5)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(0,6)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(0,7)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(0,4)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(0,5)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(0,6)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(0,7)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SELECT_EXT_L0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SELECT_EXT_L1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SELECT_INT_L0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SELECT_INT_L1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_CENTER(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MV_LIMIT0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MV_LIMIT1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SLICE_MB_CNT_HI(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_TEB_INFO(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_VP8_MVCOST_TABLE_ADDR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_VP8_MVCOST_TABLE_DATA(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_VP8_MV_SIGNCOST_X(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_VP8_MV_SIGNCOST_Y(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_VP8_MV_OFFSET_X(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_VP8_MV_OFFSET_Y(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_PART_MODE_CONFIG_LWSIZE_16X16(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_PART_MODE_CONFIG_LWSIZE_32X32(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_PART_MODE_CONFIG_LWSIZE_64X64(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_PART_DECISION_FPS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RCOL_BASE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RCOL_MB_NUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RCOL_OFFS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RCOL_CTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RCOL_RESET(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RCOL_DBG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_PROFILE_LEVEL_CONTROL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_PIC_SIZE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MODE_CONTROL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MOTION_VECTOR_COST_CONTROL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_INTRA_BIAS_1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_INTRA_BIAS_2(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MODE_BIAS_1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MODE_BIAS_2(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MODE_BIAS_3(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MB_CONTROL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_PSKIP_BIAS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_BDIRECT_BIAS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_LAMBDA_COEF(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_SET_MPEB_INTRA(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MULTI_REF_CONFIG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_COMMAND_ENCODE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_INTEN(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_INTR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MB_STATUS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_SELF_SCALE_L0(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_SELF_SCALE_L0(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_SELF_SCALE_L1(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_SELF_SCALE_L1(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_BDIRECT_CONTROL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_PRED_GEN(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_DEBUG0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_DEBUG1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_NEARMV_NEARESTMV_BIAS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_NEWMV_ZEROMV_BIAS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_LEFT4X4MV_ABOVE4X4MV_BIAS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_NEW4X4MV_ZERO4X4MV_BIAS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_MODE_CONFIG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_LAMBDA(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_SEGMENTID(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_FRAME_CYCLE_CNT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_IP_VER(0)),
    privInfo_msenc_v01_01(LW_PLWENC_CAP0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_CAP1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_CAP2(0)),
    privInfo_msenc_v01_01(LW_PLWENC_CAP3(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_REF_FRAME_COST_1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_REF_FRAME_COST_2(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_STAT_SUM_MV_X(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_STAT_SUM_MV_Y(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_STAT_SUM_MV_COUNT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_STAT_TOTAL_INTER_COST(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_INTRA_BIAS_3(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_INTER_MODE_BIAS_0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_INTER_MODE_BIAS_1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_INTER_MODE_BIAS_2(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_INTER_MODE_BIAS_3(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_LW_SIZE_BIAS_0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_LW_SIZE_BIAS_1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_IP1_EARLY_TERMINATION(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_QPFIFO(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_LAMBDA_FOR_MPEB(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_INTRA_CONFIG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_INTRA_LUMA4X4_MODE_ENABLE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_INTRA_LUMA8X8_MODE_ENABLE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_INTRA_LUMA16X16_MODE_ENABLE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_INTRA_LUMA32X32_MODE_ENABLE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_INTRA_LUMA64X64_MODE_ENABLE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_INTRA_LUMA_ENABLE_LEFTBITS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_TRANSFORM_SKIP_BIAS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_CFG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_CTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_BUF_START(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_BUF_SIZE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG2(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG3(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG4(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG5(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG6(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG7(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG8(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG9(0)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG10(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_LUMA_BASE_ADDR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_CHROMA_BASE_ADDR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_CHROMA_U_BASE_ADDR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_CHROMA_V_BASE_ADDR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_PLANAR_CHROMA_STRIDE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_PIC_INFO(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_BLKLINEAR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_CTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_ACSET_EN(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_ACSET_POP(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_ACSET_CNT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_ACSET_MAX(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_ACSET_MIN(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_ACSET_AVG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_SFC_TRANS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RHINT_BASE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RHINT_INTR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RHINT_INTEN(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RHINT_INTERVAL_BYTENUM(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RHINT_CTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_RHINT_DBG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_LUMA_OFFSET_TOP(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_LUMA_OFFSET_BOT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_SIZE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_MV_BOUNDARY(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_MAX_REQ_NUM_PER_MB(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_CTRL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFLIST0_DPBLUT(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFLIST0_DPBLUT(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFLIST1_DPBLUT(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFLIST1_DPBLUT(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,2)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,3)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,4)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,5)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,6)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,7)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,8)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,9)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,10)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,11)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,12)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,13)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,14)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(0,15)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_CHROMA_OFFSET_TOP(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_CHROMA_OFFSET_BOT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_CHROMA_U_OFFSET(0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_CHROMA_V_OFFSET(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PMM_MS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PMM_STALL_M0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PMM_STARVE_M0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PMM_STARVE_M1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PMM_STAT0(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PMM_STAT1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_PMM_STAT2(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MISC_BLCG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MISC_BLCG1(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MEDMA_BASE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MEDMA_OFFSET(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MEDMA_FRMCFG(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MEDMA_INTEN(0)),
    privInfo_msenc_v01_01(LW_PLWENC_MEDMA_INTS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_CG2(0)),
    privInfo_msenc_v01_01(LW_PLWENC_CG3(0)),
    privInfo_msenc_v01_01(LW_PLWENC_CG4(0)),
    privInfo_msenc_v01_01(0),
};

dbg_msenc_v01_01 msenc1PrivReg_v05_00[] =
{
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEM_PRIV_LEVEL_MASK(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEM_PRIV_LEVEL_MASK(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEM_DUMMY(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CPUCTL_PRIV_LEVEL_MASK(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_EXE_PRIV_LEVEL_MASK(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQTMR_PRIV_LEVEL_MASK(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MTHDCTX_PRIV_LEVEL_MASK(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_SCTL_PRIV_LEVEL_MASK(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_SCTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_SCTL1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DBGCTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQSSET(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQSCLR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQSTAT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQMODE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQMSET(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQMCLR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQMASK(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQDEST(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQDEST2(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQSCMASK(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_GPTMRINT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_GPTMRVAL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_GPTMRCTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_PTIMER0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_PTIMER1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_WDTMRVAL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_WDTMRCTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MTHDDATA(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MTHDID(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MTHDWDAT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MTHDCOUNT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MTHDPOP(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MTHDRAMSZ(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_LWRCTX(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_NXTCTX(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CTXACK(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MAILBOX0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_MAILBOX1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_ITFEN(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IDLESTATE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_FHSTATE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_PRIVSTATE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_SFTRESET(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_OS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_RM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_SOFT_PM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_SOFT_MODE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DEBUG1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DEBUGINFO(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IBRKPT1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IBRKPT2(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IBRKPT3(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IBRKPT4(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IBRKPT5(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CGCTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_ENGCTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_PMM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_ADDR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_EXCI(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_SVEC_SPR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_RSTAT0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_RSTAT3(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CPUCTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CPUCTL_ALIAS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_BOOTVEC(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_HWCFG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_HWCFG1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMACTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMATRFBASE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMATRFMOFFS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMATRFCMD(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMATRFFBOFFS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAPOLL_FB(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAPOLL_CP(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_FINISHED_FBRD_LOW(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_FINISHED_FBRD_HIGH(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_FINISHED_FBWR_LOW(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_FINISHED_FBWR_HIGH(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_LWRRENT_FBRD_LOW(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_LWRRENT_FBRD_HIGH(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_LWRRENT_FBWR_LOW(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_LWRRENT_FBWR_HIGH(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMAINFO_CTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMCTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMSTAT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_TRACEIDX(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_TRACEPC(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CG2(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CSBERRSTAT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CSBERR_INFO(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CSBERR_ADDR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_ICD_CMD(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_ICD_ADDR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_ICD_WDATA(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_ICD_RDATA(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMC(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMC(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMC(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMC(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMD(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMD(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMD(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMD(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMT(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMT(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMT(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMEMT(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(1,4)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(1,5)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(1,6)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMC(1,7)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(1,4)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(1,5)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(1,6)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_DMEMD(1,7)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_REGIONCFG_PRIV_LEVEL_MASK(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(1,4)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(1,5)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(1,6)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_TRANSCFG(1,7)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_INSTBLK(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_CTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_THROTTLE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_REGIONCFG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_ACHK_BLK(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_ACHK_BLK(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_ACHK_CTL(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_ACHK_CTL(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_DBG_STAT0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_DBG_STAT1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_DBG_STAT2(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_DBG_RDCOUNT_LO(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_DBG_RDCOUNT_HI(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_DBG_WRCOUNT_LO(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FBIF_DBG_WRCOUNT_HI(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RMVP_BASE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RMVP_STRIDE_IN_MB(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RMVP_MB_WIDTH(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RMVP_MB_HEIGHT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RMVP_CTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RMVP_DBG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_WMVP_BASE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_WMVP_CTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_WMVP_STRIDE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_WMVP_MB_WIDTH(1)),
    privInfo_msenc_v01_01(LW_PLWENC_WMVP_MB_HEIGHT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_WMVP_DBG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FRAME_CONFIG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_DELTA_PIC_ORDER_CNT_BOTTOM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FRAME_SIZE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_SEQ_PARAMETERS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_FRAME_ID(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PIC_PARAMETERS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PIC_PARAM_SET_ID(1)),
    privInfo_msenc_v01_01(LW_PLWENC_NUM_REF_IDX_ACTIVE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_IDR_PIC_ID(1)),
    privInfo_msenc_v01_01(LW_PLWENC_FRAME_CTRL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_MISC_CTRL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_INTERNAL_BIAS_MULTIPLIER(1)),
    privInfo_msenc_v01_01(LW_PLWENC_QPP_CTRL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_QPP_8X8_VECTOR0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_QPP_8X8_VECTOR1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_IPCM_CTRL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_SLICE_CTRL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_SLICE_STAT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MB_CNT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA_BASE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA_BASEL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA_SIZE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA_OFFSET(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA_CTRL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_COLOC_WRDMA_ACTIVE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_COLOC_WRDMA_BASE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_INTR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_INTEN(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_INTR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_INTEN(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_LUMA_BASE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_CHROMA_BASE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HH_L0_REFPIC_MAP_0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HH_L0_REFPIC_MAP_1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HH_L1_REFPIC_MAP_0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HH_L1_REFPIC_MAP_1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(1,4)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(1,5)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(1,6)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI4(1,7)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(1,4)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(1,5)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(1,6)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP4(1,7)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_C4X4(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(1,4)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(1,5)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(1,6)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YI8(1,7)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(1,4)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(1,5)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(1,6)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_DEADZONE_YP8(1,7)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_CTRL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTRA_RESIDUAL_BITS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTER_RESIDUAL_BITS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTRA_MODE_BITS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTER_MODE_BITS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_IPCM_BITS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_TOTAL_BITS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_TOTAL_CABAC_BINS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_LWM_INTRA_COST(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_LWM_INTER_COST(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_LWM_IPCM_COST(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_LWM_SKIP_COST(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTRA_MB_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTER_MB_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_SKIP_MB_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_IPCM_MB_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_TYPE1_BIT_COUNT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_DBF_DMA_CTRL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_PIPE_STATUS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_DEBUG0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_DEBUG1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_DEBUG2(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_INTRA_SAT_CTRL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_CMD_CONFIG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(1,4)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(1,5)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(1,6)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L0_(1,7)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(1,4)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(1,5)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(1,6)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_REF_PIC_REORDER_L1_(1,7)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(1,4)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(1,5)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(1,6)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MMCO_CMD_LIST_(1,7)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_MVC_NAL_HEADER(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_DEBUG_IF(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_DEBUG_STAT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_DEBUG_H264(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_QUANT_INTER_SAT_CTRL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_FRAME_SIZE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_FRAME_PARAM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_SEG_QINDEX(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_SEG_FLTLVL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_LOOPFILTER(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_LOOPFILTER_REF_DELTA(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_LOOPFILTER_MODE_DELTA(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_QUANT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_REFFRAME(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_MPECRAM_LWRR_OFFSET(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_MPECRAM_ADDR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_SEG_PROBS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_YMODE_PROBS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_UVMODE_PROBS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_REF_PROBS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_MV_PROBS_UPD_Y(1)),
    privInfo_msenc_v01_01(LW_PLWENC_VP8_MV_PROBS_UPD_X(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA1_BASE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA1_BASEL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA1_SIZE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA1_OFFSET(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_STATE_BASE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_STATE_BASEL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_STATE_SIZE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_STATE_OFFSET(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_WRDMA_ACK_OFFSET(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y1(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y1(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y1(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y1(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y2(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y2(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y2(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_Y2(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_UV(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_UV(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_UV(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_QUANT_UV(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_DC(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_DC(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_DC(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_DC(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_AC(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_AC(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_AC(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_IQUANT_AC(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_LAMBDA(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_LAMBDA(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_FLTLVL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_COSTRAM_ADDR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_VP8_COSTRAM_DATA(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEC_H264_STAT_SLICE_BITS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_CHROMA_V_BASE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MPEB_STAT_TOTAL_INTRA_COST(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_SEQ_PARAMETERS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_PIC_PARAMETERS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_FRAME_CTRL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_REF_PIC_LIST_MODIFICATION_L0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_REF_PIC_LIST_MODIFICATION_L1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_SHORT_TERM_REF_PIC_SET(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_ENTRY_POINT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_TOTAL_INTRA_COST(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_TOTAL_INTER_COST(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTRA_LW32X32_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTER_LW32X32_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_SKIP_LW32X32_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_IPCM_LW32X32_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTRA_LW16X16_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTER_LW16X16_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_SKIP_LW16X16_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_IPCM_LW16X16_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTRA_LW8X8_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_INTER_LW8X8_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_SKIP_LW8X8_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_STAT_IPCM_LW8X8_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_STRPS_DELTA_POCS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_STRPS_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_STRPS_USED_BY_LWRR_PIC(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_LTRPS_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_LTRPS_FLAG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_REF_PIC_DATA(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_PREDICTOR_SOURCE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_GLOBAL_L0_MV(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_GLOBAL_L1_MV(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_PIC_SIZE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STATUS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SLICE_CONTROL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SRF_SLICE_CONTROL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_PIC_CONTROL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SPS_CONFIG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_CONSTANT_L0_PREDICTOR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_CONSTANT_L1_PREDICTOR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_LAMBDA(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_LONGTERM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_DBG_SEL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_DBG_STAT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SLICE_CONFIG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MV_BOUNDARY0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MV_BOUNDARY1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SELF_SCALE_L0(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SELF_SCALE_L0(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SELF_SCALE_L1(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SELF_SCALE_L1(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,4)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,5)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,6)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,7)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,8)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,9)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,10)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,11)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,12)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,13)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,14)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,15)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,16)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,17)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,18)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,19)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,20)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,21)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,22)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,23)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,24)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,25)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,26)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,27)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,28)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,29)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,30)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_COLOC_DIST(1,31)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_INTR_STATUS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_INTR_ENABLE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_ICC_MODE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_DIFF_PIC_ORDER_CNT_ZERO(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_DIFF_PIC_ORDER_CNT_ZERO(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(1,4)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(1,5)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(1,6)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_LO(1,7)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(1,4)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(1,5)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(1,6)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SHAPE_HI(1,7)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SELECT_EXT_L0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SELECT_EXT_L1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SELECT_INT_L0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_SELECT_INT_L1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_STAMP_CENTER(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MV_LIMIT0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MV_LIMIT1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_SLICE_MB_CNT_HI(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_TEB_INFO(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_VP8_MVCOST_TABLE_ADDR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_VP8_MVCOST_TABLE_DATA(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_VP8_MV_SIGNCOST_X(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_VP8_MV_SIGNCOST_Y(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_VP8_MV_OFFSET_X(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_VP8_MV_OFFSET_Y(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_PART_MODE_CONFIG_LWSIZE_16X16(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_PART_MODE_CONFIG_LWSIZE_32X32(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_PART_MODE_CONFIG_LWSIZE_64X64(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_PART_DECISION_FPS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RCOL_BASE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RCOL_MB_NUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RCOL_OFFS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RCOL_CTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RCOL_RESET(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RCOL_DBG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_PROFILE_LEVEL_CONTROL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_PIC_SIZE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MODE_CONTROL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MOTION_VECTOR_COST_CONTROL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_INTRA_BIAS_1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_INTRA_BIAS_2(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MODE_BIAS_1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MODE_BIAS_2(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MODE_BIAS_3(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MB_CONTROL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_PSKIP_BIAS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_BDIRECT_BIAS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_LAMBDA_COEF(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_SET_MPEB_INTRA(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MULTI_REF_CONFIG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_COMMAND_ENCODE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_INTEN(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_INTR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_MB_STATUS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_SELF_SCALE_L0(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_SELF_SCALE_L0(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_SELF_SCALE_L1(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_SELF_SCALE_L1(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_BDIRECT_CONTROL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_PRED_GEN(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_DEBUG0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_DEBUG1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_NEARMV_NEARESTMV_BIAS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_NEWMV_ZEROMV_BIAS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_LEFT4X4MV_ABOVE4X4MV_BIAS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_NEW4X4MV_ZERO4X4MV_BIAS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_MODE_CONFIG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_LAMBDA(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_SEGMENTID(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_FRAME_CYCLE_CNT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_IP_VER(1)),
    privInfo_msenc_v01_01(LW_PLWENC_CAP0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_CAP1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_CAP2(1)),
    privInfo_msenc_v01_01(LW_PLWENC_CAP3(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_REF_FRAME_COST_1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_VP8_REF_FRAME_COST_2(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_STAT_SUM_MV_X(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_STAT_SUM_MV_Y(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_STAT_SUM_MV_COUNT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_STAT_TOTAL_INTER_COST(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_INTRA_BIAS_3(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_INTER_MODE_BIAS_0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_INTER_MODE_BIAS_1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_INTER_MODE_BIAS_2(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_INTER_MODE_BIAS_3(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_LW_SIZE_BIAS_0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_LW_SIZE_BIAS_1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_IP1_EARLY_TERMINATION(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_QPFIFO(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_LAMBDA_FOR_MPEB(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_INTRA_CONFIG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_INTRA_LUMA4X4_MODE_ENABLE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_INTRA_LUMA8X8_MODE_ENABLE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_INTRA_LUMA16X16_MODE_ENABLE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_INTRA_LUMA32X32_MODE_ENABLE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_INTRA_LUMA64X64_MODE_ENABLE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_H265_INTRA_LUMA_ENABLE_LEFTBITS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MDP_H265_TRANSFORM_SKIP_BIAS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_CFG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_CTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_BUF_START(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_BUF_SIZE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG2(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG3(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG4(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG5(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG6(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG7(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG8(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG9(1)),
    privInfo_msenc_v01_01(LW_PLWENC_HIST_DBG10(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_LUMA_BASE_ADDR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_CHROMA_BASE_ADDR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_CHROMA_U_BASE_ADDR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_CHROMA_V_BASE_ADDR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_PLANAR_CHROMA_STRIDE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_PIC_INFO(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_BLKLINEAR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_CTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_ACSET_EN(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_ACSET_POP(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_ACSET_CNT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_ACSET_MAX(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_ACSET_MIN(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_ACSET_AVG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PDMA_SFC_TRANS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RHINT_BASE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RHINT_INTR(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RHINT_INTEN(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RHINT_INTERVAL_BYTENUM(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RHINT_CTL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_RHINT_DBG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_LUMA_OFFSET_TOP(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_LUMA_OFFSET_BOT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_SIZE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_MV_BOUNDARY(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_MAX_REQ_NUM_PER_MB(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_CTRL(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFLIST0_DPBLUT(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFLIST0_DPBLUT(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFLIST1_DPBLUT(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFLIST1_DPBLUT(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,0)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,2)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,3)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,4)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,5)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,6)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,7)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,8)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,9)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,10)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,11)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,12)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,13)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,14)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_BASE(1,15)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_CHROMA_OFFSET_TOP(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_CHROMA_OFFSET_BOT(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_CHROMA_U_OFFSET(1)),
    privInfo_msenc_v01_01(LW_PLWENC_ME_MBC_REFPIC_CHROMA_V_OFFSET(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PMM_MS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PMM_STALL_M0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PMM_STARVE_M0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PMM_STARVE_M1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PMM_STAT0(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PMM_STAT1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_PMM_STAT2(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MISC_BLCG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MISC_BLCG1(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MEDMA_BASE(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MEDMA_OFFSET(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MEDMA_FRMCFG(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MEDMA_INTEN(1)),
    privInfo_msenc_v01_01(LW_PLWENC_MEDMA_INTS(1)),
    privInfo_msenc_v01_01(LW_PLWENC_CG2(1)),
    privInfo_msenc_v01_01(LW_PLWENC_CG3(1)),
    privInfo_msenc_v01_01(LW_PLWENC_CG4(1)),
    privInfo_msenc_v01_01(1),
};
#endif

//-----------------------------------------------------
// msencIsSupported_v05_00
//-----------------------------------------------------
BOOL msencIsSupported_v05_00( LwU32 indexGpu )
{
    if(lwencId != LWWATCH_MSENC_0 && lwencId != LWWATCH_MSENC_1)
    {
        dprintf("Only MSENC0 and MSENC1 supported on this GPU\n");
        return FALSE;
    }

    pMsencPrivReg[0] = msenc0PrivReg_v05_00;
    pMsencPrivReg[1] = msenc1PrivReg_v05_00;
    pMsencMethodTable = msencMethodTable_v05_00;

    engineId = lwencId;

    return TRUE;
}

//-----------------------------------------------------
// msencDumpImem_v05_00 - Dumps LWENC instruction memory
//-----------------------------------------------------
LW_STATUS msencDumpImem_v05_00( LwU32 indexGpu , LwU32 imemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 imemSizeMax;

    LwU32 addrssImem;
    LwU32 address2Imem;
    LwU32 address2Imemt;
    LwU32 u;
    LwU32 blk;

    // IF LWENC is not specified, Operate for all available LWENC engines

    dprintf("Dumping IMEM for LWENC%d\n", engineId);
    addrssImem    = LW_PLWENC_FALCON_IMEMD(engineId,0);
    address2Imem  = LW_PLWENC_FALCON_IMEMC(engineId,0);
    address2Imemt = LW_PLWENC_FALCON_IMEMT(engineId,0);
    blk=0;
    imemSizeMax = (GPU_REG_IDX_RD_DRF(_PLWENC_FALCON, _HWCFG, engineId, _IMEM_SIZE)<<8) ;
    if (imemSize > 0)
        imemSize = min(imemSizeMax, imemSize);
    else
        imemSize = imemSizeMax;

    dprintf("\n");
    dprintf("lw: -- Gpu %u LWENC%d IMEM -- \n", indexGpu, engineId);
    dprintf("lw: -- Gpu %u LWENC%d IMEM SIZE =  0x%08x-- \n", indexGpu,engineId,imemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    for(u=0;u<(imemSize+3)/4;u++)
    {
        LwU32 i;
        if((u%64)==0)
        {
            GPU_REG_WR32(address2Imemt, blk++);
        }
        i = (u<<(0?LW_PLWENC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2Imem,i);
        if((u%8==0))
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  GPU_REG_RD32(addrssImem));
    }
    dprintf("\n\n");
    return status;
}

//-----------------------------------------------------
// msencDumpDmem_v05_00 - Dumps LWENC data memory
//-----------------------------------------------------
LW_STATUS msencDumpDmem_v05_00( LwU32 indexGpu , LwU32 dmemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 dmemSizeMax;
    // these are the variables defined for use in parsing and printinf the methods and data
    LwU32 addrss, address2, u, i, comMthdOffs = 0, appMthdOffs = 0, classNum;
    LwU32 comMthd[CMNMETHODARRAYSIZE] = {0};
    LwU32 appMthd[APPMETHODARRAYSIZE] = {0};
    LwU32 methodIdx;

    dprintf("lw: Dumping DMEM for LWENC%d\n", engineId);
    dmemSizeMax = (GPU_REG_IDX_RD_DRF(_PLWENC_FALCON, _HWCFG, engineId, _DMEM_SIZE)<<8);

    if(dmemSize > 0)
        dmemSize = min(dmemSizeMax, dmemSize);
    else
       dmemSize = dmemSizeMax;

    addrss      = LW_PLWENC_FALCON_DMEMD(engineId,0);
    address2    = LW_PLWENC_FALCON_DMEMC(engineId,0);
    classNum    = 0xD0B7;

    dprintf("\n");
    dprintf("lw: -- Gpu %u LWENC%d DMEM -- \n", indexGpu,engineId);
    dprintf("lw: -- Gpu %u LWENC%d DMEM SIZE =  0x%08x-- \n", indexGpu,engineId,dmemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");

    for(u=0;u<(dmemSize+3)/4;u++)
    {
        i = (u<<(0?LW_PLWENC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        if((u%8==0))
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  GPU_REG_RD32(addrss));
    }

    // get methods offset are in the DWORD#3 in dmem
    u = (3<<(0?LW_PLWENC_FALCON_IMEMC_OFFS));
    GPU_REG_WR32(address2,u);
    comMthdOffs = (GPU_REG_RD32(addrss)) >> 2;
    appMthdOffs = comMthdOffs + 16;

    for(u=0; u<CMNMETHODARRAYSIZE;u++)
    {
        i = ((u+comMthdOffs)<<(0?LW_PLWENC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        comMthd[u] = GPU_REG_RD32(addrss);
        i = ((u+appMthdOffs)<<(0?LW_PLWENC_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        appMthd[u] = GPU_REG_RD32(addrss);
    }

    dprintf("\n\n-----------------------------------------------------------------------\n");
    dprintf("%4s, %8s,    %4s, %8s,    %4s, %8s,    %4s, %8s\n",
                                "Mthd", "Data", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data");
    dprintf("[COMMON METHODS]\n");
    for (u=0; u<CMNMETHODARRAYSIZE; u+=4)
    {
        dprintf("%04X: %08X,    %04X: %08X,    %04X: %08X,    %04X: %08X\n",
        CMNMETHODBASE_v02+4*u, comMthd[u], CMNMETHODBASE_v02+4*(u+1), comMthd[u+1], 
        CMNMETHODBASE_v02+4*(u+2), comMthd[u+2], CMNMETHODBASE_v02+4*(u+3), comMthd[u+3]);
    }
    dprintf("\n");
    dprintf("\n[APP METHODS]\n");
    for (u=0; u<APPMETHODARRAYSIZE; u+=4)
    {
        dprintf("%04X: %08X,    %04X: %08X,    %04X: %08X,    %04X: %08X\n",
        APPMETHODBASE_v02+4*u, appMthd[u], APPMETHODBASE_v02+4*(u+1), appMthd[u+1],
        APPMETHODBASE_v02+4*(u+2), appMthd[u+2], APPMETHODBASE_v02+4*(u+3), appMthd[u+3]);
    }

    // common methods
    // if this environment variable is present, parse and print out the methods
    if (getelw("LWW_CLASS_SDK") != NULL)
    {
        dprintf("\n[COMMON METHODS]\n");
        for(u=0;u<16;u++)
        {
            if(parseClassHeader(classNum, CMNMETHODBASE_v02+4*u, comMthd[u]))
                dprintf("\n");
        }
        dprintf("\n");

        // app methods
        dprintf("\n[APP METHODS]\n");
        for(u=0;u<16;u++)
        {
            if(parseClassHeader(classNum, APPMETHODBASE_v02+4*u, appMthd[u]))
                dprintf("\n");
        }
    }
    else
    {
    #if defined(USE_LWENC_5_0)
        dprintf("\n[COMMON METHODS]\n");
        for(u=0;u<16;u++)
        {
            for(methodIdx=0;;methodIdx++)
            {
                if(pMsencMethodTable[methodIdx].m_id == (CMNMETHODBASE_v02+4*u))
                {
                    msencPrintMethodData_v01_00(40,
                                                pMsencMethodTable[methodIdx].m_tag, 
                                                pMsencMethodTable[methodIdx].m_id,
                                                comMthd[u]);
                    break;
                }
                else if (pMsencMethodTable[methodIdx].m_id == 0)
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
                if(pMsencMethodTable[methodIdx].m_id == (APPMETHODBASE_v02+4*u))
                {
                    msencPrintMethodData_v01_00(40,
                                                pMsencMethodTable[methodIdx].m_tag,
                                                pMsencMethodTable[methodIdx].m_id,
                                                appMthd[u]);
                    break;
                }
                else if (pMsencMethodTable[methodIdx].m_id == 0)
                {
                    break;
                }
            }
        }
        dprintf("\n");
    #else
        dprintf("\nDefine the LWW_CLASS_SDK environment variable to the location \
                    of the class header files to view parsed methods and data \n");
    #endif
    }
    return status;
}

//-----------------------------------------------------
// msencTestState_v05_00 - Test basic lwenc state
//-----------------------------------------------------
LW_STATUS msencTestState_v05_00( LwU32 indexGpu )
{
    LW_STATUS    status = LW_OK;
    LwU32   regIntr;
    LwU32   regIntrEn;
    LwU32   data32;

    dprintf("lw: Checking states of LWENC%d\n", engineId);
    //check falcon interrupts
    regIntr = GPU_REG_RD32(LW_PLWENC_FALCON_IRQSTAT(engineId));
    regIntrEn = GPU_REG_RD32(LW_PLWENC_FALCON_IRQMASK(engineId));
    regIntr &= regIntrEn;

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _GPTMR, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_GPTMR disabled\n");

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _WDTMR, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_WDTMR disabled\n");

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _MTHD, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_MTHD disabled\n");

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _CTXSW, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_CTXSW disabled\n");

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _HALT, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_HALT disabled\n");

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _EXTERR, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_EXTERR disabled\n");

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _SWGEN0, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_SWGEN0 disabled\n");

    if ( !DRF_VAL(_PLWENC, _FALCON_IRQMASK, _SWGEN1, regIntrEn))
        dprintf("lw: LW_PLWENC_FALCON_IRQMASK_SWGEN1 disabled\n");

    //if any interrupt pending, set error
    if (regIntr != 0)
    {
        addUnitErr("\t LWENC interrupts are pending\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _GPTMR, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_GPTMR pending\n");
        dprintf("lw: LW_PLWENC_FALCON_GPTMRINT:    0x%08x\n", 
        GPU_REG_RD32(LW_PLWENC_FALCON_GPTMRINT(0)) );
        dprintf("lw: LW_PLWENC_FALCON_GPTMRVAL:    0x%08x\n", 
        GPU_REG_RD32(LW_PLWENC_FALCON_GPTMRVAL(0)) );
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _WDTMR, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_WDTMR pending\n");
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _MTHD, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_MTHD pending\n");
        dprintf("lw: LW_PLWENC_FALCON_MTHDDATA_DATA:    0x%08x\n", 
        GPU_REG_RD32(LW_PLWENC_FALCON_MTHDDATA(engineId)) );

        data32 = GPU_REG_RD32(LW_PLWENC_FALCON_MTHDID(engineId));
        dprintf("lw: LW_PLWENC_FALCON_MTHDID_ID:    0x%08x\n", 
        DRF_VAL( _PLWENC,_FALCON_MTHDID, _ID, data32)  );
        dprintf("lw: LW_PLWENC_FALCON_MTHDID_SUBCH:    0x%08x\n", 
        DRF_VAL( _PLWENC,_FALCON_MTHDID, _SUBCH, data32)  );
        dprintf("lw: LW_PLWENC_FALCON_MTHDID_PRIV:    0x%08x\n", 
        DRF_VAL( _PLWENC,_FALCON_MTHDID, _PRIV, data32)  );
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _CTXSW, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_CTXSW pending\n");
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _HALT, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_HALT pending\n");
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _EXTERR, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_EXTERR pending\n");
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _SWGEN0, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_SWGEN0 pending\n");

        pFalcon[indexGpu].falconPrintMailbox(LW_FALCON_LWENC_BASE);
    }

    if ( DRF_VAL( _PLWENC,_FALCON_IRQSTAT, _SWGEN1, regIntr))
    {
        dprintf("lw: LW_PLWENC_FALCON_IRQSTAT_SWGEN1 pending\n");
    }

    //
    //print falcon states
    //Bit |  Signal meaning
    //0      FALCON busy
    //

    data32 = GPU_REG_RD32(LW_PLWENC_FALCON_IDLESTATE(engineId));

    if ( DRF_VAL( _PLWENC, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_IDLESTATE_FALCON_BUSY\n");
        addUnitErr("\t LW_PLWENC_FALCON_IDLESTATE_FALCON_BUSY\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PLWENC_FALCON_FHSTATE(engineId));

    if ( DRF_VAL( _PLWENC, _FALCON_FHSTATE, _FALCON_HALTED, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_FHSTATE_FALCON_HALTED\n");
        addUnitErr("\t LW_PLWENC_FALCON_FHSTATE_FALCON_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWENC, _FALCON_FHSTATE, _ENGINE_FAULTED, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        addUnitErr("\t LW_PLWENC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWENC, _FALCON_FHSTATE, _STALL_REQ, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_FHSTATE_STALL_REQ\n");
        addUnitErr("\t LW_PLWENC_FALCON_FHSTATE_STALL_REQ\n");
        status = LW_ERR_GENERIC;
    }

    //print falcon ctl regs
    data32 = GPU_REG_RD32(LW_PLWENC_FALCON_ENGCTL(engineId));

    if ( DRF_VAL( _PLWENC, _FALCON_ENGCTL, _ILW_CONTEXT, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_ENGCTL_ILW_CONTEXT\n");
        addUnitErr("\t LW_PLWENC_FALCON_ENGCTL_ILW_CONTEXT\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWENC, _FALCON_ENGCTL, _STALLREQ, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_ENGCTL_STALLREQ\n");
        addUnitErr("\t LW_PLWENC_FALCON_ENGCTL_STALLREQ\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PLWENC_FALCON_CPUCTL(engineId));

    if ( DRF_VAL( _PLWENC, _FALCON_CPUCTL, _IILWAL, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_CPUCTL_IILWAL\n");
        addUnitErr("\t LW_PLWENC_FALCON_CPUCTL_IILWAL\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWENC, _FALCON_CPUCTL, _HALTED, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_CPUCTL_HALTED\n");
        addUnitErr("\t LW_PLWENC_FALCON_CPUCTL_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWENC, _FALCON_CPUCTL, _STOPPED, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_CPUCTL_STOPPED\n");
        addUnitErr("\t Warning: LW_PLWENC_FALCON_CPUCTL_STOPPED\n");
        //status = LW_ERR_GENERIC;
    }

    // state of mthd/ctx interface 
    data32 = GPU_REG_RD32(LW_PLWENC_FALCON_ITFEN(engineId));

    if (DRF_VAL( _PLWENC, _FALCON_ITFEN, _CTXEN, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_ITFEN_CTXEN enabled\n");

        if (pFalcon[indexGpu].falconTestCtxState(LW_FALCON_LWENC_BASE, "PLWENC") == LW_ERR_GENERIC)
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
        dprintf("lw: + LW_PLWENC_FALCON_ITFEN_CTXEN disabled\n");
    }

    if ( DRF_VAL( _PLWENC, _FALCON_ITFEN, _MTHDEN, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_ITFEN_MTHDEN enabled\n");
    }
    else
    {
        dprintf("lw: + LW_PLWENC_FALCON_ITFEN_MTHDEN disabled\n");
    }

    //check if falcon is hung (instr ptr)
    if ( pFalcon[indexGpu].falconTestPC(LW_FALCON_LWENC_BASE, "PLWENC") == LW_ERR_GENERIC )
    {
        dprintf("lw: Falcon instruction pointer is stuck or invalid\n");

        //TODO: treat falcon PC errors as warnings now, need to report as error
        addUnitErr("\t Warning: Falcon instruction pointer is stuck or invalid\n");
        //status = LW_ERR_GENERIC;
    }
    return status;
}

//-----------------------------------------------------
// msencPrintPriv_v05_00
//-----------------------------------------------------
void msencPrintPriv_v05_00(LwU32 clmn, char *tag, LwU32 id)
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
// msencDumpPriv_v05_00 - Dumps LWENC priv reg space
//-----------------------------------------------------
LW_STATUS msencDumpPriv_v05_00(LwU32 indexGpu)
{
    LwU32 u;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWENC%d priv registers -- \n", indexGpu,engineId);
    dprintf("lw:\n");

    for(u=0;;u++)
    {
        if(pMsencPrivReg[engineId][u].m_id==0)
        {
            break;
        }

        pMsenc[indexGpu].msencPrintPriv(50,pMsencPrivReg[engineId][u].m_tag,
                    pMsencPrivReg[engineId][u].m_id);
    }
    return LW_OK; 
}

//--------------------------------------------------------
// msencDisplayHwcfg_v05_00 - Display LWENC HW config
//--------------------------------------------------------
LW_STATUS msencDisplayHwcfg_v05_00(LwU32 indexGpu)
{
    LwU32 hwcfg, hwcfg1;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWENC%d HWCFG -- \n", indexGpu,engineId);
    dprintf("lw:\n");

    hwcfg  = GPU_REG_RD32(LW_PLWENC_FALCON_HWCFG(engineId));

    dprintf("lw: LW_PLWENC_FALCON_HWCFG:  0x%08x\n", hwcfg); 
    dprintf("lw:\n");
    dprintf("lw:  IMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",

    DRF_VAL(_PLWENC, _FALCON_HWCFG, _IMEM_SIZE, hwcfg),
    DRF_VAL(_PLWENC, _FALCON_HWCFG, _IMEM_SIZE, hwcfg)<<8);

    dprintf("lw:  DMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",

    DRF_VAL(_PLWENC, _FALCON_HWCFG, _DMEM_SIZE, hwcfg),
    DRF_VAL(_PLWENC, _FALCON_HWCFG, _DMEM_SIZE, hwcfg)<<8);

    dprintf("lw:  METHODFIFO_DEPTH: 0x%08X\n", DRF_VAL(_PLWENC, _FALCON_HWCFG, _METHODFIFO_DEPTH, hwcfg));
    dprintf("lw:  DMAQUEUE_DEPTH:   0x%08X\n", DRF_VAL(_PLWENC, _FALCON_HWCFG, _DMAQUEUE_DEPTH, hwcfg));
    dprintf("lw:\n");

    hwcfg1 = GPU_REG_RD32(LW_PLWENC_FALCON_HWCFG1(engineId));

    dprintf("lw: LW_PLWENC_FALCON_HWCFG1: 0x%08x\n", hwcfg1); 
    dprintf("lw:\n");
    dprintf("lw:  CORE_REV:     0x%08X\n", DRF_VAL(_PLWENC, _FALCON_HWCFG1, _CORE_REV, hwcfg1));
    dprintf("lw:  SELWRITY_MODEL:   0x%08X\n", DRF_VAL(_PLWENC, _FALCON_HWCFG1, _SELWRITY_MODEL, hwcfg1));
    dprintf("lw:  IMEM_PORTS:       0x%08X\n", DRF_VAL(_PLWENC, _FALCON_HWCFG1, _IMEM_PORTS, hwcfg1));
    dprintf("lw:  DMEM_PORTS:       0x%08X\n", DRF_VAL(_PLWENC, _FALCON_HWCFG1, _DMEM_PORTS, hwcfg1));
    dprintf("lw:  TAG_WIDTH:        0x%08X\n", DRF_VAL(_PLWENC, _FALCON_HWCFG1, _TAG_WIDTH, hwcfg1));

    return LW_OK;
}

 /*
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
LW_STATUS  msencDisplayFlcnSPR_v05_00(LwU32 indexGpu)
{
    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWENC%d Special Purpose Registers -- \n", indexGpu,engineId);
    dprintf("lw:\n");

    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1008);
    dprintf("lw: LWENC IV0 :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1108);
    dprintf("lw: LWENC IV1 :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1308);
    dprintf("lw: LWENC EV  :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1408);
    dprintf("lw: LWENC SP  :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1508);
    dprintf("lw: LWENC PC  :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1608);
    dprintf("lw: LWENC IMB :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1708);
    dprintf("lw: LWENC DMB :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(engineId), 0x1808);
    dprintf("lw: LWENC CSW :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(engineId))); 
    dprintf("lw:\n\n");
    return LW_OK; 
}
