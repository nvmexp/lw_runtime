/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2014 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//-----------------------------------------------------
//
// lwenc0400.c - LWENC routines
// 
//-----------------------------------------------------

#include "maxwell/gm107/dev_lwenc_pri_sw.h"
#include "maxwell/gm107/dev_falcon_v4.h"

#include "msenc.h"

#include "g_msenc_private.h"     // (rmconfig)  implementation prototypes

#define USE_LWENC_4_0

#if defined(USE_LWENC_4_0)
#define LW_C0B7_NOP                                                              (0x00000100)
#define LW_C0B7_PM_TRIGGER                                                       (0x00000140)
#define LW_C0B7_SET_APPLICATION_ID                                               (0x00000200)
#define LW_C0B7_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LW_C0B7_SEMAPHORE_A                                                      (0x00000240)
#define LW_C0B7_SEMAPHORE_B                                                      (0x00000244)
#define LW_C0B7_SEMAPHORE_C                                                      (0x00000248)
#define LW_C0B7_EXELWTE                                                          (0x00000300)
#define LW_C0B7_SEMAPHORE_D                                                      (0x00000304)
#define LW_C0B7_SET_IN_REF_PIC0                                                  (0x00000400)
#define LW_C0B7_SET_IN_REF_PIC1                                                  (0x00000404)
#define LW_C0B7_SET_IN_REF_PIC2                                                  (0x00000408)
#define LW_C0B7_SET_IN_REF_PIC3                                                  (0x0000040C)
#define LW_C0B7_SET_IN_REF_PIC4                                                  (0x00000410)
#define LW_C0B7_SET_IN_REF_PIC5                                                  (0x00000414)
#define LW_C0B7_SET_IN_REF_PIC6                                                  (0x00000418)
#define LW_C0B7_SET_IN_REF_PIC7                                                  (0x0000041C)
#define LW_C0B7_SET_IN_REF_PIC8                                                  (0x00000420)
#define LW_C0B7_SET_IN_REF_PIC9                                                  (0x00000424)
#define LW_C0B7_SET_IN_REF_PIC10                                                 (0x00000428)
#define LW_C0B7_SET_IN_REF_PIC11                                                 (0x0000042C)
#define LW_C0B7_SET_IN_REF_PIC12                                                 (0x00000430)
#define LW_C0B7_SET_IN_REF_PIC13                                                 (0x00000434)
#define LW_C0B7_SET_IN_REF_PIC14                                                 (0x00000438)
#define LW_C0B7_SET_IN_REF_PIC15                                                 (0x0000043C)
#define LW_C0B7_SET_CONTROL_PARAMS                                               (0x00000700)
#define LW_C0B7_SET_PICTURE_INDEX                                                (0x00000704)
#define LW_C0B7_SET_IN_RCDATA                                                    (0x0000070C)
#define LW_C0B7_SET_IN_DRV_PIC_SETUP                                             (0x00000710)
#define LW_C0B7_SET_IN_CEAHINTS_DATA                                             (0x00000714)
#define LW_C0B7_SET_OUT_ENC_STATUS                                               (0x00000718)
#define LW_C0B7_SET_OUT_BITSTREAM                                                (0x0000071C)
#define LW_C0B7_SET_IOHISTORY                                                    (0x00000720)
#define LW_C0B7_SET_IO_RC_PROCESS                                                (0x00000724)
#define LW_C0B7_SET_IN_COLOC_DATA                                                (0x00000728)
#define LW_C0B7_SET_OUT_COLOC_DATA                                               (0x0000072C)
#define LW_C0B7_SET_OUT_REF_PIC                                                  (0x00000730)
#define LW_C0B7_SET_IN_LWR_PIC                                                   (0x00000734)
#define LW_C0B7_SET_IN_MEPRED_DATA                                               (0x00000738)
#define LW_C0B7_SET_OUT_MEPRED_DATA                                              (0x0000073C)
#define LW_C0B7_PM_TRIGGER_END                                                   (0x00001114)

dbg_msenc_v01_01 msencMethodTable_v04_00[] =
{
    privInfo_msenc_v01_01(LW_C0B7_NOP),
    privInfo_msenc_v01_01(LW_C0B7_PM_TRIGGER),
    privInfo_msenc_v01_01(LW_C0B7_SET_APPLICATION_ID),
    privInfo_msenc_v01_01(LW_C0B7_SET_WATCHDOG_TIMER),
    privInfo_msenc_v01_01(LW_C0B7_SEMAPHORE_A),
    privInfo_msenc_v01_01(LW_C0B7_SEMAPHORE_B),
    privInfo_msenc_v01_01(LW_C0B7_SEMAPHORE_C),
    privInfo_msenc_v01_01(LW_C0B7_EXELWTE),
    privInfo_msenc_v01_01(LW_C0B7_SEMAPHORE_D),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC0),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC1),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC2),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC3),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC4),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC5),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC6),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC7),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC8),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC9),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC10),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC11),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC12),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC13),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC14),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_REF_PIC15),
    privInfo_msenc_v01_01(LW_C0B7_SET_CONTROL_PARAMS),
    privInfo_msenc_v01_01(LW_C0B7_SET_PICTURE_INDEX),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_RCDATA),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_DRV_PIC_SETUP),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_CEAHINTS_DATA),
    privInfo_msenc_v01_01(LW_C0B7_SET_OUT_ENC_STATUS),
    privInfo_msenc_v01_01(LW_C0B7_SET_OUT_BITSTREAM),
    privInfo_msenc_v01_01(LW_C0B7_SET_IOHISTORY),
    privInfo_msenc_v01_01(LW_C0B7_SET_IO_RC_PROCESS),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_COLOC_DATA),
    privInfo_msenc_v01_01(LW_C0B7_SET_OUT_COLOC_DATA),
    privInfo_msenc_v01_01(LW_C0B7_SET_OUT_REF_PIC),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_LWR_PIC),
    privInfo_msenc_v01_01(LW_C0B7_SET_IN_MEPRED_DATA),
    privInfo_msenc_v01_01(LW_C0B7_SET_OUT_MEPRED_DATA),
    privInfo_msenc_v01_01(LW_C0B7_PM_TRIGGER_END),
    privInfo_msenc_v01_01(0),
};

dbg_msenc_v01_01 msencPrivReg_v04_00[] =
{
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQSSET(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQSCLR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQSTAT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQMODE(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQMSET(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQMCLR(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQMASK(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IRQDEST(0)),
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
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CPUCTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CPUCTL_ALIAS(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CG2(0)),
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
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMCTL(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_IMSTAT(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_TRACEIDX(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_TRACEPC(0)),
    privInfo_msenc_v01_01(LW_PLWENC_FALCON_CG1_SLCG(0)),
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
#endif

//-----------------------------------------------------
// msencIsSupported_v04_00
//-----------------------------------------------------
BOOL msencIsSupported_v04_00( LwU32 indexGpu )
{
    if(lwencId != LWWATCH_MSENC_0)
    {
        dprintf("Only MSENC0 supported on this GPU\n");
        return FALSE;
    }

    pMsencPrivReg[0] = msencPrivReg_v04_00;
    pMsencMethodTable = msencMethodTable_v04_00;
    return TRUE;
}

//-----------------------------------------------------
// msencDumpImem_v04_00 - Dumps LWENC instruction memory
//-----------------------------------------------------
LW_STATUS msencDumpImem_v04_00( LwU32 indexGpu , LwU32 imemSize)
{
    LW_STATUS status = LW_OK;
    LwU32  imemSizeMax;
    LwU32 addrssImem = LW_PLWENC_FALCON_IMEMD(0,0);
    LwU32 address2Imem = LW_PLWENC_FALCON_IMEMC(0,0);
    LwU32 address2Imemt = LW_PLWENC_FALCON_IMEMT(0,0);
    LwU32 u;
    LwU32 blk=0;
    imemSizeMax = (GPU_REG_IDX_RD_DRF(_PLWENC_FALCON, _HWCFG, 0, _IMEM_SIZE)<<8) ;
    if (imemSize > 0)
        imemSize = min(imemSizeMax, imemSize);
    else
        imemSize = imemSizeMax;

    dprintf("\n");
    dprintf("lw: -- Gpu %u LWENC IMEM -- \n", indexGpu);    
    dprintf("lw: -- Gpu %u LWENC IMEM SIZE =  0x%08x-- \n", indexGpu,imemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    for(u=0;u<(imemSize+3)/4;u++)
    {
        LwU32 i;
        if((u%64)==0) {
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
    return status;  
}

//-----------------------------------------------------
// msencDumpDmem_v04_00 - Dumps LWENC data memory
//-----------------------------------------------------
LW_STATUS msencDumpDmem_v04_00( LwU32 indexGpu , LwU32 dmemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 dmemSizeMax;
    // these are the variables defined for use in parsing and printinf the methods and data
    LwU32 addrss, address2, u, i, comMthdOffs = 0, appMthdOffs = 0, classNum;
    LwU32 comMthd[CMNMETHODARRAYSIZE] = {0};
    LwU32 appMthd[APPMETHODARRAYSIZE] = {0};
    LwU32 methodIdx;

    dmemSizeMax = (GPU_REG_IDX_RD_DRF(_PLWENC_FALCON, _HWCFG, 0, _DMEM_SIZE)<<8) ;

    if(dmemSize > 0)
        dmemSize = min(dmemSizeMax, dmemSize);
    else
       dmemSize = dmemSizeMax;

    addrss      = LW_PLWENC_FALCON_DMEMD(0,0);
    address2    = LW_PLWENC_FALCON_DMEMC(0,0);
    classNum    = 0xC0B7;

    dprintf("\n");
    dprintf("lw: -- Gpu %u LWENC DMEM -- \n", indexGpu);
    dprintf("lw: -- Gpu %u LWENC DMEM SIZE =  0x%08x-- \n", indexGpu,dmemSize);
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
#if defined(USE_LWENC_4_0)
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
// msencTestState_v04_00 - Test basic lwenc state
//-----------------------------------------------------
LW_STATUS msencTestState_v04_00( LwU32 indexGpu )
{
    LW_STATUS    status = LW_OK;
    LwU32   regIntr;
    LwU32   regIntrEn;
    LwU32   data32;

    //check falcon interrupts
    regIntr = GPU_REG_RD32(LW_PLWENC_FALCON_IRQSTAT(0));
    regIntrEn = GPU_REG_RD32(LW_PLWENC_FALCON_IRQMASK(0));
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
            GPU_REG_RD32(LW_PLWENC_FALCON_MTHDDATA(0)) );
        
        data32 = GPU_REG_RD32(LW_PLWENC_FALCON_MTHDID(0));
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

    data32 = GPU_REG_RD32(LW_PLWENC_FALCON_IDLESTATE(0));

    if ( DRF_VAL( _PLWENC, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_PLWENC_FALCON_IDLESTATE_FALCON_BUSY\n");
        addUnitErr("\t LW_PLWENC_FALCON_IDLESTATE_FALCON_BUSY\n");
        status = LW_ERR_GENERIC;
    }

  
    data32 = GPU_REG_RD32(LW_PLWENC_FALCON_FHSTATE(0));
 
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
    data32 = GPU_REG_RD32(LW_PLWENC_FALCON_ENGCTL(0));
    
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

    data32 = GPU_REG_RD32(LW_PLWENC_FALCON_CPUCTL(0));

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
    data32 = GPU_REG_RD32(LW_PLWENC_FALCON_ITFEN(0));

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
// msencPrintPriv_v04_00
//-----------------------------------------------------
void msencPrintPriv_v04_00(LwU32 clmn, char *tag, LwU32 id)
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
// msencDumpPriv_v04_00 - Dumps LWENC priv reg space
//-----------------------------------------------------
LW_STATUS msencDumpPriv_v04_00(LwU32 indexGpu)
{
    LwU32 u;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWENC priv registers -- \n", indexGpu);
    dprintf("lw:\n");

    for(u=0;;u++)
    {
        if(pMsencPrivReg[0][u].m_id==0)
        {
            break;
        }
        
        pMsenc[indexGpu].msencPrintPriv(50,pMsencPrivReg[0][u].m_tag,pMsencPrivReg[0][u].m_id);
    }
    return LW_OK; 
}

//--------------------------------------------------------
// msencDisplayHwcfg_v04_00 - Display LWENC HW config
//--------------------------------------------------------
LW_STATUS msencDisplayHwcfg_v04_00(LwU32 indexGpu)
{
    LwU32 hwcfg, hwcfg1;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWENC HWCFG -- \n", indexGpu);
    dprintf("lw:\n");

    hwcfg  = GPU_REG_RD32(LW_PLWENC_FALCON_HWCFG(0));
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

    hwcfg1 = GPU_REG_RD32(LW_PLWENC_FALCON_HWCFG1(0));
    dprintf("lw: LW_PLWENC_FALCON_HWCFG1: 0x%08x\n", hwcfg1); 
    dprintf("lw:\n");
    dprintf("lw:  CORE_REV:         0x%08X\n", DRF_VAL(_PLWENC, _FALCON_HWCFG1, _CORE_REV, hwcfg1)); 
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
LW_STATUS  msencDisplayFlcnSPR_v04_00(LwU32 indexGpu)
{
    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWENC Special Purpose Registers -- \n", indexGpu);
    dprintf("lw:\n");

    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(0), 0x1008);
    dprintf("lw: LWENC IV0 :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(0))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(0), 0x1108);
    dprintf("lw: LWENC IV1 :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(0))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(0), 0x1308);
    dprintf("lw: LWENC EV  :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(0))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(0), 0x1408);
    dprintf("lw: LWENC SP  :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(0))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(0), 0x1508);
    dprintf("lw: LWENC PC  :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(0))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(0), 0x1608);
    dprintf("lw: LWENC IMB :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(0))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(0), 0x1708);
    dprintf("lw: LWENC DMB :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(0))); 
    GPU_REG_WR32(LW_PLWENC_FALCON_ICD_CMD(0), 0x1808);
    dprintf("lw: LWENC CSW :    0x%08x\n", GPU_REG_RD32(LW_PLWENC_FALCON_ICD_RDATA(0))); 
    dprintf("lw:\n\n");

    return LW_OK; 
}
