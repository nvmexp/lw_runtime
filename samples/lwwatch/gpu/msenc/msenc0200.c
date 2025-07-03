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
// msenc0100.c - MSENC routines
// 
//-----------------------------------------------------

#include "hal.h"
#include "tegrasys.h"
#include "t11x/t114/dev_msenc_pri.h"
#include "t11x/t114/dev_falcon_v5.h"

#include "msenc.h"

#include "g_msenc_private.h"     // (rmconfig)  implementation prototypes

#define USE_MSENC_2_0

// MSENC Device specific register access macros
#define MSENC_REG_RD32(reg)           (DEV_REG_RD32((reg - DRF_BASE(LW_PMSENC)), "MSENC", 0))
#define MSENC_REG_WR32(reg,val)       (DEV_REG_WR32((reg - DRF_BASE(LW_PMSENC)), val, "MSENC", 0))
#define MSENC_REG_RD_DRF(d,r,f)       (((MSENC_REG_RD32(LW ## d ## r))>>DRF_SHIFT(LW ## d ## r ## f))&DRF_MASK(LW ## d ## r ## f)) 

#if defined(USE_MSENC_2_0)
#define LW_A0B7_NOP                                                              (0x00000100)
#define LW_A0B7_PM_TRIGGER                                                       (0x00000140)
#define LW_A0B7_SET_APPLICATION_ID                                               (0x00000200)
#define LW_A0B7_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LW_A0B7_SEMAPHORE_A                                                      (0x00000240)
#define LW_A0B7_SEMAPHORE_B                                                      (0x00000244)
#define LW_A0B7_SEMAPHORE_C                                                      (0x00000248)
#define LW_A0B7_EXELWTE                                                          (0x00000300)
#define LW_A0B7_SEMAPHORE_D                                                      (0x00000304)
#define LW_A0B7_SET_IN_REF_PIC0                                                  (0x00000400)
#define LW_A0B7_SET_IN_REF_PIC1                                                  (0x00000404)
#define LW_A0B7_SET_IN_REF_PIC2                                                  (0x00000408)
#define LW_A0B7_SET_IN_REF_PIC3                                                  (0x0000040C)
#define LW_A0B7_SET_IN_REF_PIC4                                                  (0x00000410)
#define LW_A0B7_SET_IN_REF_PIC5                                                  (0x00000414)
#define LW_A0B7_SET_IN_REF_PIC6                                                  (0x00000418)
#define LW_A0B7_SET_IN_REF_PIC7                                                  (0x0000041C)
#define LW_A0B7_SET_IN_REF_PIC8                                                  (0x00000420)
#define LW_A0B7_SET_IN_REF_PIC9                                                  (0x00000424)
#define LW_A0B7_SET_IN_REF_PIC10                                                 (0x00000428)
#define LW_A0B7_SET_IN_REF_PIC11                                                 (0x0000042C)
#define LW_A0B7_SET_IN_REF_PIC12                                                 (0x00000430)
#define LW_A0B7_SET_IN_REF_PIC13                                                 (0x00000434)
#define LW_A0B7_SET_IN_REF_PIC14                                                 (0x00000438)
#define LW_A0B7_SET_IN_REF_PIC15                                                 (0x0000043C)
#define LW_A0B7_SET_CONTROL_PARAMS                                               (0x00000700)
#define LW_A0B7_SET_PICTURE_INDEX                                                (0x00000704)
#define LW_A0B7_SET_IN_RCDATA                                                    (0x0000070C)
#define LW_A0B7_SET_IN_DRV_PIC_SETUP                                             (0x00000710)
#define LW_A0B7_SET_IN_CEAHINTS_DATA                                             (0x00000714)
#define LW_A0B7_SET_OUT_ENC_STATUS                                               (0x00000718)
#define LW_A0B7_SET_OUT_BITSTREAM                                                (0x0000071C)
#define LW_A0B7_SET_IOHISTORY                                                    (0x00000720)
#define LW_A0B7_SET_IO_RC_PROCESS                                                (0x00000724)
#define LW_A0B7_SET_IN_COLOC_DATA                                                (0x00000728)
#define LW_A0B7_SET_OUT_COLOC_DATA                                               (0x0000072C)
#define LW_A0B7_SET_OUT_REF_PIC                                                  (0x00000730)
#define LW_A0B7_SET_IN_LWR_PIC                                                   (0x00000734)
#define LW_A0B7_SET_IN_MEPRED_DATA                                               (0x00000738)
#define LW_A0B7_SET_OUT_MEPRED_DATA                                              (0x0000073C)
#define LW_A0B7_PM_TRIGGER_END                                                   (0x00001114)

dbg_msenc_v01_01 msencMethodTable_v02_00[] =
{
    privInfo_msenc_v01_01(LW_A0B7_NOP),
    privInfo_msenc_v01_01(LW_A0B7_PM_TRIGGER),
    privInfo_msenc_v01_01(LW_A0B7_SET_APPLICATION_ID),
    privInfo_msenc_v01_01(LW_A0B7_SET_WATCHDOG_TIMER),
    privInfo_msenc_v01_01(LW_A0B7_SEMAPHORE_A),
    privInfo_msenc_v01_01(LW_A0B7_SEMAPHORE_B),
    privInfo_msenc_v01_01(LW_A0B7_SEMAPHORE_C),
    privInfo_msenc_v01_01(LW_A0B7_EXELWTE),
    privInfo_msenc_v01_01(LW_A0B7_SEMAPHORE_D),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC0),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC1),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC2),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC3),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC4),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC5),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC6),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC7),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC8),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC9),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC10),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC11),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC12),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC13),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC14),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_REF_PIC15),
    privInfo_msenc_v01_01(LW_A0B7_SET_CONTROL_PARAMS),
    privInfo_msenc_v01_01(LW_A0B7_SET_PICTURE_INDEX),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_RCDATA),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_DRV_PIC_SETUP),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_CEAHINTS_DATA),
    privInfo_msenc_v01_01(LW_A0B7_SET_OUT_ENC_STATUS),
    privInfo_msenc_v01_01(LW_A0B7_SET_OUT_BITSTREAM),
    privInfo_msenc_v01_01(LW_A0B7_SET_IOHISTORY),
    privInfo_msenc_v01_01(LW_A0B7_SET_IO_RC_PROCESS),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_COLOC_DATA),
    privInfo_msenc_v01_01(LW_A0B7_SET_OUT_COLOC_DATA),
    privInfo_msenc_v01_01(LW_A0B7_SET_OUT_REF_PIC),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_LWR_PIC),
    privInfo_msenc_v01_01(LW_A0B7_SET_IN_MEPRED_DATA),
    privInfo_msenc_v01_01(LW_A0B7_SET_OUT_MEPRED_DATA),
    privInfo_msenc_v01_01(LW_A0B7_PM_TRIGGER_END),
    privInfo_msenc_v01_01(0),
};

dbg_msenc_v01_01 msencPrivReg_v02_00[] =
{
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IRQSSET),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IRQSCLR),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IRQSTAT),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IRQMODE),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IRQMSET),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IRQMCLR),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IRQMASK),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IRQDEST),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_GPTMRINT),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_GPTMRVAL),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_GPTMRCTL),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_PTIMER0),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_PTIMER1),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_WDTMRVAL),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_WDTMRCTL),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_MTHDDATA),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_MTHDID),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_MTHDWDAT),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_MTHDCOUNT),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_MTHDPOP),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_MTHDRAMSZ),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_LWRCTX),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_NXTCTX),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_CTXACK),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_MAILBOX0),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_MAILBOX1),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_ITFEN),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IDLESTATE),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_FHSTATE),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_PRIVSTATE),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_SFTRESET),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_OS),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_RM),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_SOFT_PM),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_SOFT_MODE),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DEBUG1),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DEBUGINFO),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IBRKPT1),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IBRKPT2),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_CGCTL),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_ENGCTL),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_PMM),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_ADDR),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_CPUCTL),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_BOOTVEC),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_HWCFG),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_HWCFG1),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMACTL),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMATRFBASE),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMATRFMOFFS),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMATRFCMD),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMATRFFBOFFS),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMAPOLL_FB),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMAPOLL_CP),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IMCTL),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IMSTAT),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_TRACEIDX),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_TRACEPC),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_ICD_CMD),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_ICD_ADDR),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_ICD_WDATA),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_ICD_RDATA),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IMEMC(0)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IMEMC(1)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IMEMC(2)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IMEMC(3)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IMEMD(0)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IMEMD(1)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IMEMD(2)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IMEMD(3)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IMEMT(0)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IMEMT(1)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IMEMT(2)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_IMEMT(3)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMC(0)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMC(1)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMC(2)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMC(3)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMC(4)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMC(5)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMC(6)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMC(7)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMD(0)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMD(1)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMD(2)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMD(3)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMD(4)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMD(5)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMD(6)),
    privInfo_msenc_v01_01(LW_PMSENC_FALCON_DMEMD(7)),
    privInfo_msenc_v01_01(LW_PMSENC_RMVP_BASE),
    privInfo_msenc_v01_01(LW_PMSENC_RMVP_STRIDE_IN_MB),
    privInfo_msenc_v01_01(LW_PMSENC_RMVP_MB_WIDTH),
    privInfo_msenc_v01_01(LW_PMSENC_RMVP_MB_HEIGHT),
    privInfo_msenc_v01_01(LW_PMSENC_RMVP_CTL),
    privInfo_msenc_v01_01(LW_PMSENC_RMVP_DBG),
    privInfo_msenc_v01_01(LW_PMSENC_WMVP_BASE),
    privInfo_msenc_v01_01(LW_PMSENC_WMVP_CTL),
    privInfo_msenc_v01_01(LW_PMSENC_WMVP_STRIDE),
    privInfo_msenc_v01_01(LW_PMSENC_WMVP_MB_WIDTH),
    privInfo_msenc_v01_01(LW_PMSENC_WMVP_MB_HEIGHT),
    privInfo_msenc_v01_01(LW_PMSENC_WMVP_DBG),
    privInfo_msenc_v01_01(LW_PMSENC_FRAME_CONFIG),
    privInfo_msenc_v01_01(LW_PMSENC_DELTA_PIC_ORDER_CNT_BOTTOM),
    privInfo_msenc_v01_01(LW_PMSENC_FRAME_SIZE),
    privInfo_msenc_v01_01(LW_PMSENC_SEQ_PARAMETERS),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_FRAME_ID),
    privInfo_msenc_v01_01(LW_PMSENC_PIC_PARAMETERS),
    privInfo_msenc_v01_01(LW_PMSENC_PIC_PARAM_SET_ID),
    privInfo_msenc_v01_01(LW_PMSENC_NUM_REF_IDX_ACTIVE),
    privInfo_msenc_v01_01(LW_PMSENC_IDR_PIC_ID),
    privInfo_msenc_v01_01(LW_PMSENC_FRAME_CTRL),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_MISC_CTRL),
    privInfo_msenc_v01_01(LW_PMSENC_INTERNAL_BIAS_MULTIPLIER),
    privInfo_msenc_v01_01(LW_PMSENC_QPP_CTRL),
    privInfo_msenc_v01_01(LW_PMSENC_QPP_8X8_VECTOR0),
    privInfo_msenc_v01_01(LW_PMSENC_QPP_8X8_VECTOR1),
    privInfo_msenc_v01_01(LW_PMSENC_IPCM_CTRL),
    privInfo_msenc_v01_01(LW_PMSENC_SLICE_CTRL),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_SLICE_STAT),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_MB_CNT),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_WRDMA_BASE),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_WRDMA_BASEL),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_WRDMA_SIZE),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_WRDMA_OFFSET),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_WRDMA_CTRL),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_COLOC_WRDMA_ACTIVE),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_COLOC_WRDMA_BASE),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_INTR),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_INTEN),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_INTR),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_INTEN),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_LUMA_BASE),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_CHROMA_BASE),
    privInfo_msenc_v01_01(LW_PMSENC_HH_L0_REFPIC_MAP_0),
    privInfo_msenc_v01_01(LW_PMSENC_HH_L0_REFPIC_MAP_1),
    privInfo_msenc_v01_01(LW_PMSENC_HH_L1_REFPIC_MAP_0),
    privInfo_msenc_v01_01(LW_PMSENC_HH_L1_REFPIC_MAP_1),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI4(0)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI4(1)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI4(2)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI4(3)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI4(4)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI4(5)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI4(6)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI4(7)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP4(0)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP4(1)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP4(2)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP4(3)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP4(4)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP4(5)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP4(6)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP4(7)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_C4X4),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI8(0)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI8(1)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI8(2)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI8(3)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI8(4)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI8(5)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI8(6)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YI8(7)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP8(0)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP8(1)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP8(2)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP8(3)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP8(4)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP8(5)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP8(6)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_QUANT_DEADZONE_YP8(7)),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_CTRL),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_INTRA_RESIDUAL_BITS),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_INTER_RESIDUAL_BITS),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_INTRA_MODE_BITS),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_INTER_MODE_BITS),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_IPCM_BITS),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_TOTAL_BITS),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_TOTAL_CABAC_BINS),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_LWM_INTRA_COST),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_LWM_INTER_COST),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_LWM_IPCM_COST),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_LWM_SKIP_COST),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_INTRA_MB_NUM),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_INTER_MB_NUM),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_SKIP_MB_NUM),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_IPCM_MB_NUM),
    privInfo_msenc_v01_01(LW_PMSENC_STAT_TYPE1_BIT_COUNT),
    privInfo_msenc_v01_01(LW_PMSENC_DBF_DMA_CTRL),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_PIPE_STATUS),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_DEBUG0),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_DEBUG1),
    privInfo_msenc_v01_01(LW_PMSENC_MPEB_DEBUG2),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_CMD_CONFIG),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L0_(0)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L0_(1)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L0_(2)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L0_(3)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L0_(4)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L0_(5)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L0_(6)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L0_(7)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L1_(0)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L1_(1)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L1_(2)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L1_(3)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L1_(4)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L1_(5)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L1_(6)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_REF_PIC_REORDER_L1_(7)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_MMCO_CMD_LIST_(0)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_MMCO_CMD_LIST_(1)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_MMCO_CMD_LIST_(2)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_MMCO_CMD_LIST_(3)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_MMCO_CMD_LIST_(4)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_MMCO_CMD_LIST_(5)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_MMCO_CMD_LIST_(6)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_MMCO_CMD_LIST_(7)),
    privInfo_msenc_v01_01(LW_PMSENC_MPEC_MVC_NAL_HEADER),
    privInfo_msenc_v01_01(LW_PMSENC_ME_PREDICTOR_SOURCE),
    privInfo_msenc_v01_01(LW_PMSENC_ME_GLOBAL_L0_MV),
    privInfo_msenc_v01_01(LW_PMSENC_ME_GLOBAL_L1_MV),
    privInfo_msenc_v01_01(LW_PMSENC_ME_PIC_SIZE),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STATUS),
    privInfo_msenc_v01_01(LW_PMSENC_ME_SLICE_CONTROL),
    privInfo_msenc_v01_01(LW_PMSENC_ME_SRF_SLICE_CONTROL),
    privInfo_msenc_v01_01(LW_PMSENC_ME_PIC_CONTROL),
    privInfo_msenc_v01_01(LW_PMSENC_ME_SPS_CONFIG),
    privInfo_msenc_v01_01(LW_PMSENC_ME_CONSTANT_L0_PREDICTOR),
    privInfo_msenc_v01_01(LW_PMSENC_ME_CONSTANT_L1_PREDICTOR),
    privInfo_msenc_v01_01(LW_PMSENC_ME_LAMBDA),
    privInfo_msenc_v01_01(LW_PMSENC_ME_LONGTERM),
    privInfo_msenc_v01_01(LW_PMSENC_ME_DBG_SEL),
    privInfo_msenc_v01_01(LW_PMSENC_ME_DBG_STAT),
    privInfo_msenc_v01_01(LW_PMSENC_ME_SLICE_CONFIG),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MV_BOUNDARY0),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MV_BOUNDARY1),
    privInfo_msenc_v01_01(LW_PMSENC_ME_SELF_SCALE_L0(0)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_SELF_SCALE_L0(1)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_SELF_SCALE_L1(0)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_SELF_SCALE_L1(1)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(0)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(1)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(2)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(3)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(4)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(5)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(6)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(7)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(8)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(9)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(10)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(11)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(12)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(13)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(14)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(15)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(16)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(17)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(18)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(19)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(20)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(21)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(22)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(23)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(24)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(25)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(26)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(27)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(28)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(29)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(30)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_COLOC_DIST(31)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_INTR_STATUS),
    privInfo_msenc_v01_01(LW_PMSENC_ME_INTR_ENABLE),
    privInfo_msenc_v01_01(LW_PMSENC_ME_ICC_MODE),
    privInfo_msenc_v01_01(LW_PMSENC_ME_DIFF_PIC_ORDER_CNT_ZERO(0)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_DIFF_PIC_ORDER_CNT_ZERO(1)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_LO(0)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_LO(1)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_LO(2)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_LO(3)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_LO(4)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_LO(5)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_LO(6)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_LO(7)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_HI(0)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_HI(1)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_HI(2)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_HI(3)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_HI(4)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_HI(5)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_HI(6)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SHAPE_HI(7)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SELECT_EXT_L0),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SELECT_EXT_L1),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SELECT_INT_L0),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_SELECT_INT_L1),
    privInfo_msenc_v01_01(LW_PMSENC_ME_STAMP_CENTER),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MV_LIMIT0),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MV_LIMIT1),
    privInfo_msenc_v01_01(LW_PMSENC_RCOL_BASE),
    privInfo_msenc_v01_01(LW_PMSENC_RCOL_MB_NUM),
    privInfo_msenc_v01_01(LW_PMSENC_RCOL_OFFS),
    privInfo_msenc_v01_01(LW_PMSENC_RCOL_CTL),
    privInfo_msenc_v01_01(LW_PMSENC_RCOL_RESET),
    privInfo_msenc_v01_01(LW_PMSENC_RCOL_DBG),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_PROFILE_LEVEL_CONTROL),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_PIC_SIZE),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_MODE_CONTROL),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_MOTION_VECTOR_COST_CONTROL),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_INTRA_BIAS_1),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_INTRA_BIAS_2),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_MODE_BIAS_1),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_MODE_BIAS_2),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_MODE_BIAS_3),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_MB_CONTROL),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_PSKIP_BIAS),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_BDIRECT_BIAS),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_LAMBDA_COEF),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_SET_MPEB_INTRA),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_MULTI_REF_CONFIG),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_COMMAND_ENCODE),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_INTEN),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_INTR),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_MB_STATUS),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_SELF_SCALE_L0(0)),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_SELF_SCALE_L0(1)),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_SELF_SCALE_L1(0)),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_SELF_SCALE_L1(1)),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_BDIRECT_CONTROL),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_PRED_GEN),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_DEBUG0),
    privInfo_msenc_v01_01(LW_PMSENC_MDP_DEBUG1),
    privInfo_msenc_v01_01(LW_PMSENC_HIST_CFG),
    privInfo_msenc_v01_01(LW_PMSENC_HIST_CTL),
    privInfo_msenc_v01_01(LW_PMSENC_HIST_BUF_START),
    privInfo_msenc_v01_01(LW_PMSENC_HIST_BUF_SIZE),
    privInfo_msenc_v01_01(LW_PMSENC_HIST_DBG0),
    privInfo_msenc_v01_01(LW_PMSENC_HIST_DBG1),
    privInfo_msenc_v01_01(LW_PMSENC_HIST_DBG2),
    privInfo_msenc_v01_01(LW_PMSENC_HIST_DBG3),
    privInfo_msenc_v01_01(LW_PMSENC_HIST_DBG4),
    privInfo_msenc_v01_01(LW_PMSENC_HIST_DBG5),
    privInfo_msenc_v01_01(LW_PMSENC_HIST_DBG6),
    privInfo_msenc_v01_01(LW_PMSENC_HIST_DBG7),
    privInfo_msenc_v01_01(LW_PMSENC_HIST_DBG8),
    privInfo_msenc_v01_01(LW_PMSENC_HIST_DBG9),
    privInfo_msenc_v01_01(LW_PMSENC_HIST_DBG10),
    privInfo_msenc_v01_01(LW_PMSENC_PDMA_LUMA_BASE_ADDR),
    privInfo_msenc_v01_01(LW_PMSENC_PDMA_CHROMA_BASE_ADDR),
    privInfo_msenc_v01_01(LW_PMSENC_PDMA_PIC_INFO),
    privInfo_msenc_v01_01(LW_PMSENC_PDMA_BLKLINEAR),
    privInfo_msenc_v01_01(LW_PMSENC_PDMA_CTL),
    privInfo_msenc_v01_01(LW_PMSENC_PDMA_ACSET_EN),
    privInfo_msenc_v01_01(LW_PMSENC_PDMA_ACSET_POP),
    privInfo_msenc_v01_01(LW_PMSENC_PDMA_ACSET_CNT),
    privInfo_msenc_v01_01(LW_PMSENC_PDMA_ACSET_MAX),
    privInfo_msenc_v01_01(LW_PMSENC_PDMA_ACSET_MIN),
    privInfo_msenc_v01_01(LW_PMSENC_PDMA_ACSET_AVG),
    privInfo_msenc_v01_01(LW_PMSENC_RHINT_BASE),
    privInfo_msenc_v01_01(LW_PMSENC_RHINT_INTR),
    privInfo_msenc_v01_01(LW_PMSENC_RHINT_INTEN),
    privInfo_msenc_v01_01(LW_PMSENC_RHINT_INTERVAL_BYTENUM),
    privInfo_msenc_v01_01(LW_PMSENC_RHINT_CTL),
    privInfo_msenc_v01_01(LW_PMSENC_RHINT_DBG),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_LUMA_OFFSET_TOP),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_LUMA_OFFSET_BOT),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_MV_BOUNDARY),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_MAX_REQ_NUM_PER_MB),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_STATUS),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFLIST0_DPBLUT(0)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFLIST0_DPBLUT(1)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFLIST1_DPBLUT(0)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFLIST1_DPBLUT(1)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(0)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(1)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(2)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(3)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(4)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(5)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(6)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(7)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(8)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(9)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(10)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(11)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(12)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(13)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(14)),
    privInfo_msenc_v01_01(LW_PMSENC_ME_MBC_REFPIC_BASE(15)),
    privInfo_msenc_v01_01(LW_PMSENC_PMM_MS),
    privInfo_msenc_v01_01(LW_PMSENC_PMM_STALL_M0),
    privInfo_msenc_v01_01(LW_PMSENC_PMM_STARVE_M0),
    privInfo_msenc_v01_01(LW_PMSENC_PMM_STARVE_M1),
    privInfo_msenc_v01_01(LW_PMSENC_PMM_STAT0),
    privInfo_msenc_v01_01(LW_PMSENC_PMM_STAT1),
    privInfo_msenc_v01_01(LW_PMSENC_PMM_STAT2),
    privInfo_msenc_v01_01(LW_PMSENC_MISC_PG),
    privInfo_msenc_v01_01(LW_PMSENC_MISC_PG1),
    privInfo_msenc_v01_01(0),
};
#endif

//-----------------------------------------------------
// msencIsSupported_v02_00
//-----------------------------------------------------
BOOL msencIsSupported_v02_00( LwU32 indexGpu )
{
    if(lwencId != LWWATCH_MSENC_0)
    {
        dprintf("Only MSENC0 supported on this GPU\n");
        return FALSE;
    }

    pMsencPrivReg[0] = msencPrivReg_v02_00;
    pMsencMethodTable = msencMethodTable_v02_00;
    return TRUE;
}

//-----------------------------------------------------
// msencDumpImem_v02_00 - Dumps MSENC instruction memory
//-----------------------------------------------------
LW_STATUS msencDumpImem_v02_00( LwU32 indexGpu , LwU32 imemSize)
{
    LW_STATUS status = LW_OK;
    LwU32  imemSizeMax;
    LwU32 addrssImem = LW_PMSENC_FALCON_IMEMD(0);
    LwU32 address2Imem = LW_PMSENC_FALCON_IMEMC(0);
    LwU32 address2Imemt = LW_PMSENC_FALCON_IMEMT(0);
    LwU32 u;
    LwU32 blk=0;
    imemSizeMax = (MSENC_REG_RD_DRF(_PMSENC_FALCON, _HWCFG, _IMEM_SIZE)<<8) ;
    if (imemSize > 0)
        imemSize = min(imemSizeMax, imemSize);
    else
        imemSize = imemSizeMax;

    dprintf("\n");
    dprintf("lw: -- Gpu %u MSENC IMEM -- \n", indexGpu);    
    dprintf("lw: -- Gpu %u MSENC IMEM SIZE =  0x%08x-- \n", indexGpu,imemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    for(u=0;u<(imemSize+3)/4;u++)
    {
        LwU32 i;
        if((u%64)==0) {
            MSENC_REG_WR32(address2Imemt, blk++);
        }
        i = (u<<(0?LW_PMSENC_FALCON_IMEMC_OFFS));
        MSENC_REG_WR32(address2Imem,i);
        if((u%8==0))
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  MSENC_REG_RD32(addrssImem));
    }
    return status;  
}

//-----------------------------------------------------
// msencDumpDmem_v02_00 - Dumps MSENC data memory
//-----------------------------------------------------
LW_STATUS msencDumpDmem_v02_00( LwU32 indexGpu , LwU32 dmemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 dmemSizeMax;
    // these are the variables defined for use in parsing and printinf the methods and data
    LwU32 addrss, address2, u, i, comMthdOffs = 0, appMthdOffs = 0, classNum;
    LwU32 comMthd[CMNMETHODARRAYSIZE] = {0};
    LwU32 appMthd[APPMETHODARRAYSIZE] = {0};
    LwU32 methodIdx;

    dmemSizeMax = (MSENC_REG_RD_DRF(_PMSENC_FALCON, _HWCFG, _DMEM_SIZE)<<8) ;

    if(dmemSize > 0)
        dmemSize = min(dmemSizeMax, dmemSize);
    else
       dmemSize = dmemSizeMax;

    addrss      = LW_PMSENC_FALCON_DMEMD(0);
    address2    = LW_PMSENC_FALCON_DMEMC(0);
    classNum    = 0xA0B7;

    dprintf("\n");
    dprintf("lw: -- Gpu %u MSENC DMEM -- \n", indexGpu);
    dprintf("lw: -- Gpu %u MSENC DMEM SIZE =  0x%08x-- \n", indexGpu,dmemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    

    for(u=0;u<(dmemSize+3)/4;u++)
    {
        i = (u<<(0?LW_PMSENC_FALCON_IMEMC_OFFS));
        MSENC_REG_WR32(address2,i);
        if((u%8==0))
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  MSENC_REG_RD32(addrss));
    }

    // get methods offset are in the DWORD#3 in dmem
    u = (3<<(0?LW_PMSENC_FALCON_IMEMC_OFFS));
    MSENC_REG_WR32(address2,u);
    comMthdOffs = (MSENC_REG_RD32(addrss)) >> 2;
    appMthdOffs = comMthdOffs + 16;

    for(u=0; u<CMNMETHODARRAYSIZE;u++)
    {
        i = ((u+comMthdOffs)<<(0?LW_PMSENC_FALCON_IMEMC_OFFS));
        MSENC_REG_WR32(address2,i);
        comMthd[u] = MSENC_REG_RD32(addrss);
        i = ((u+appMthdOffs)<<(0?LW_PMSENC_FALCON_IMEMC_OFFS));
        MSENC_REG_WR32(address2,i);
        appMthd[u] = MSENC_REG_RD32(addrss);
    }

    dprintf("\n\n-----------------------------------------------------------------------\n");
    dprintf("%4s, %8s,    %4s, %8s,    %4s, %8s,    %4s, %8s\n", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data");
    dprintf("[COMMON METHODS]\n");
    for (u=0; u<CMNMETHODARRAYSIZE; u+=4)
    {
        dprintf("%04X: %08X,    %04X: %08X,    %04X: %08X,    %04X: %08X\n",
        CMNMETHODBASE+4*u, comMthd[u], CMNMETHODBASE+4*(u+1), comMthd[u+1], 
        CMNMETHODBASE+4*(u+2), comMthd[u+2], CMNMETHODBASE+4*(u+3), comMthd[u+3]);
    }
    dprintf("\n");
    dprintf("\n[APP METHODS]\n");
    for (u=0; u<APPMETHODARRAYSIZE; u+=4)
    {

        dprintf("%04X: %08X,    %04X: %08X,    %04X: %08X,    %04X: %08X\n",
        APPMETHODBASE+4*u, appMthd[u], APPMETHODBASE+4*(u+1), appMthd[u+1],
        APPMETHODBASE+4*(u+2), appMthd[u+2], APPMETHODBASE+4*(u+3), appMthd[u+3]);
    }

    // common methods
    // if this environment variable is present, parse and print out the methods
    if (getelw("LWW_CLASS_SDK") != NULL)
    {
        dprintf("\n[COMMON METHODS]\n");
        for(u=0;u<16;u++)
        {
            if(parseClassHeader(classNum, CMNMETHODBASE+4*u, comMthd[u]))
                dprintf("\n");
        }
        dprintf("\n");

        // app methods
        dprintf("\n[APP METHODS]\n");
        for(u=0;u<16;u++)
        {
            if(parseClassHeader(classNum, APPMETHODBASE+4*u, appMthd[u]))
                dprintf("\n");
        }
    }
    else
    {
#if defined(USE_MSENC_2_0)
        dprintf("\n[COMMON METHODS]\n");
        for(u=0;u<16;u++)
        {
            for(methodIdx=0;;methodIdx++)
            {
                if(pMsencMethodTable[methodIdx].m_id == (CMNMETHODBASE+4*u))
                {
                    msencPrintMethodData_v01_00(40,pMsencMethodTable[methodIdx].m_tag, pMsencMethodTable[methodIdx].m_id, comMthd[u]);
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
                if(pMsencMethodTable[methodIdx].m_id == (APPMETHODBASE+4*u))
                {
                    msencPrintMethodData_v01_00(40,pMsencMethodTable[methodIdx].m_tag, pMsencMethodTable[methodIdx].m_id, appMthd[u]);
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
        dprintf("\nDefine the LWW_CLASS_SDK environment variable to the location of the class header files to view parsed methods and data \n");
#endif
    }
    return status;  
}

//-----------------------------------------------------
// msencTestState_v02_00 - Test basic msenc state
//-----------------------------------------------------
LW_STATUS msencTestState_v02_00( LwU32 indexGpu )
{
    LW_STATUS    status = LW_OK;
    LwU32   regIntr;
    LwU32   regIntrEn;
    LwU32   data32;
    LwU32   msencBaseAddress;
    PDEVICE_RELOCATION pDev = NULL;

    pDev     = tegrasysGetDeviceReloc(&TegraSysObj[indexGpu], "MSENC", 0);
    assert(pDev);
    msencBaseAddress = (LwU32)pDev->start;

    //check falcon interrupts
    regIntr = MSENC_REG_RD32(LW_PMSENC_FALCON_IRQSTAT);
    regIntrEn = MSENC_REG_RD32(LW_PMSENC_FALCON_IRQMASK);
    regIntr &= regIntrEn;

    if ( !DRF_VAL(_PMSENC, _FALCON_IRQMASK, _GPTMR, regIntrEn))
        dprintf("lw: LW_PMSENC_FALCON_IRQMASK_GPTMR disabled\n");

    if ( !DRF_VAL(_PMSENC, _FALCON_IRQMASK, _WDTMR, regIntrEn))
        dprintf("lw: LW_PMSENC_FALCON_IRQMASK_WDTMR disabled\n");

    if ( !DRF_VAL(_PMSENC, _FALCON_IRQMASK, _MTHD, regIntrEn))
        dprintf("lw: LW_PMSENC_FALCON_IRQMASK_MTHD disabled\n");

    if ( !DRF_VAL(_PMSENC, _FALCON_IRQMASK, _CTXSW, regIntrEn))
        dprintf("lw: LW_PMSENC_FALCON_IRQMASK_CTXSW disabled\n");

    if ( !DRF_VAL(_PMSENC, _FALCON_IRQMASK, _HALT, regIntrEn))
        dprintf("lw: LW_PMSENC_FALCON_IRQMASK_HALT disabled\n");

    if ( !DRF_VAL(_PMSENC, _FALCON_IRQMASK, _EXTERR, regIntrEn))
        dprintf("lw: LW_PMSENC_FALCON_IRQMASK_EXTERR disabled\n");

    if ( !DRF_VAL(_PMSENC, _FALCON_IRQMASK, _SWGEN0, regIntrEn))
        dprintf("lw: LW_PMSENC_FALCON_IRQMASK_SWGEN0 disabled\n");

    if ( !DRF_VAL(_PMSENC, _FALCON_IRQMASK, _SWGEN1, regIntrEn))
        dprintf("lw: LW_PMSENC_FALCON_IRQMASK_SWGEN1 disabled\n");

   
    //if any interrupt pending, set error
    if (regIntr != 0)
    {
        addUnitErr("\t MSENC interrupts are pending\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSENC,_FALCON_IRQSTAT, _GPTMR, regIntr))
    {
        dprintf("lw: LW_PMSENC_FALCON_IRQSTAT_GPTMR pending\n");

        dprintf("lw: LW_PMSENC_FALCON_GPTMRINT:    0x%08x\n", 
            MSENC_REG_RD32(LW_PMSENC_FALCON_GPTMRINT) );
        dprintf("lw: LW_PMSENC_FALCON_GPTMRVAL:    0x%08x\n", 
            MSENC_REG_RD32(LW_PMSENC_FALCON_GPTMRVAL) );
        
    }
    
    if ( DRF_VAL( _PMSENC,_FALCON_IRQSTAT, _WDTMR, regIntr))
    {
        dprintf("lw: LW_PMSENC_FALCON_IRQSTAT_WDTMR pending\n");
    }

    if ( DRF_VAL( _PMSENC,_FALCON_IRQSTAT, _MTHD, regIntr))
    {
        dprintf("lw: LW_PMSENC_FALCON_IRQSTAT_MTHD pending\n");

        dprintf("lw: LW_PMSENC_FALCON_MTHDDATA_DATA:    0x%08x\n", 
            MSENC_REG_RD32(LW_PMSENC_FALCON_MTHDDATA) );
        
        data32 = MSENC_REG_RD32(LW_PMSENC_FALCON_MTHDID);
        dprintf("lw: LW_PMSENC_FALCON_MTHDID_ID:    0x%08x\n", 
           DRF_VAL( _PMSENC,_FALCON_MTHDID, _ID, data32)  );
        dprintf("lw: LW_PMSENC_FALCON_MTHDID_SUBCH:    0x%08x\n", 
           DRF_VAL( _PMSENC,_FALCON_MTHDID, _SUBCH, data32)  );
        dprintf("lw: LW_PMSENC_FALCON_MTHDID_PRIV:    0x%08x\n", 
           DRF_VAL( _PMSENC,_FALCON_MTHDID, _PRIV, data32)  );
    }
    
    if ( DRF_VAL( _PMSENC,_FALCON_IRQSTAT, _CTXSW, regIntr))
    {
        dprintf("lw: LW_PMSENC_FALCON_IRQSTAT_CTXSW pending\n");
    }
    
    if ( DRF_VAL( _PMSENC,_FALCON_IRQSTAT, _HALT, regIntr))
    {
        dprintf("lw: LW_PMSENC_FALCON_IRQSTAT_HALT pending\n");
    }
    
    if ( DRF_VAL( _PMSENC,_FALCON_IRQSTAT, _EXTERR, regIntr))
    {
        dprintf("lw: LW_PMSENC_FALCON_IRQSTAT_EXTERR pending\n");
    }
    
    if ( DRF_VAL( _PMSENC,_FALCON_IRQSTAT, _SWGEN0, regIntr))
    {
        dprintf("lw: LW_PMSENC_FALCON_IRQSTAT_SWGEN0 pending\n");

        pFalcon[indexGpu].falconPrintMailbox(msencBaseAddress);
    }

    if ( DRF_VAL( _PMSENC,_FALCON_IRQSTAT, _SWGEN1, regIntr))
    {
        dprintf("lw: LW_PMSENC_FALCON_IRQSTAT_SWGEN1 pending\n");
    }

     //
    //print falcon states
    //Bit |  Signal meaning
    //0      FALCON busy
    //

    data32 = MSENC_REG_RD32(LW_PMSENC_FALCON_IDLESTATE);

    if ( DRF_VAL( _PMSENC, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_PMSENC_FALCON_IDLESTATE_FALCON_BUSY\n");
        addUnitErr("\t LW_PMSENC_FALCON_IDLESTATE_FALCON_BUSY\n");
        status = LW_ERR_GENERIC;
    }

  
    data32 = MSENC_REG_RD32(LW_PMSENC_FALCON_FHSTATE);
 
    if ( DRF_VAL( _PMSENC, _FALCON_FHSTATE, _FALCON_HALTED, data32))
    {
        dprintf("lw: + LW_PMSENC_FALCON_FHSTATE_FALCON_HALTED\n");
        addUnitErr("\t LW_PMSENC_FALCON_FHSTATE_FALCON_HALTED\n");
        status = LW_ERR_GENERIC;
    }
    
    if ( DRF_VAL( _PMSENC, _FALCON_FHSTATE, _ENGINE_FAULTED, data32))
    {
        dprintf("lw: + LW_PMSENC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        addUnitErr("\t LW_PMSENC_FALCON_FHSTATE_ENGINE_FAULTED\n");
        status = LW_ERR_GENERIC;
    }
    
    if ( DRF_VAL( _PMSENC, _FALCON_FHSTATE, _STALL_REQ, data32))
    {
        dprintf("lw: + LW_PMSENC_FALCON_FHSTATE_STALL_REQ\n");
        addUnitErr("\t LW_PMSENC_FALCON_FHSTATE_STALL_REQ\n");
        status = LW_ERR_GENERIC;
    }

    //print falcon ctl regs
    data32 = MSENC_REG_RD32(LW_PMSENC_FALCON_ENGCTL);
    
    if ( DRF_VAL( _PMSENC, _FALCON_ENGCTL, _ILW_CONTEXT, data32))
    {
        dprintf("lw: + LW_PMSENC_FALCON_ENGCTL_ILW_CONTEXT\n");
        addUnitErr("\t LW_PMSENC_FALCON_ENGCTL_ILW_CONTEXT\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSENC, _FALCON_ENGCTL, _STALLREQ, data32))
    {
        dprintf("lw: + LW_PMSENC_FALCON_ENGCTL_STALLREQ\n");
        addUnitErr("\t LW_PMSENC_FALCON_ENGCTL_STALLREQ\n");
        status = LW_ERR_GENERIC;
    }

    data32 = MSENC_REG_RD32(LW_PMSENC_FALCON_CPUCTL);

    if ( DRF_VAL( _PMSENC, _FALCON_CPUCTL, _IILWAL, data32))
    {
        dprintf("lw: + LW_PMSENC_FALCON_CPUCTL_IILWAL\n");
        addUnitErr("\t LW_PMSENC_FALCON_CPUCTL_IILWAL\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSENC, _FALCON_CPUCTL, _HALTED, data32))
    {
        dprintf("lw: + LW_PMSENC_FALCON_CPUCTL_HALTED\n");
        addUnitErr("\t LW_PMSENC_FALCON_CPUCTL_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PMSENC, _FALCON_CPUCTL, _STOPPED, data32))
    {
        dprintf("lw: + LW_PMSENC_FALCON_CPUCTL_STOPPED\n");
        addUnitErr("\t Warning: LW_PMSENC_FALCON_CPUCTL_STOPPED\n");
        //status = LW_ERR_GENERIC;
    }

    // state of mthd/ctx interface 
    data32 = MSENC_REG_RD32(LW_PMSENC_FALCON_ITFEN);

    if (DRF_VAL( _PMSENC, _FALCON_ITFEN, _CTXEN, data32))
    {
        dprintf("lw: + LW_PMSENC_FALCON_ITFEN_CTXEN enabled\n");
             
        if (pFalcon[indexGpu].falconTestCtxState(msencBaseAddress, "PMSENC") == LW_ERR_GENERIC)
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
        dprintf("lw: + LW_PMSENC_FALCON_ITFEN_CTXEN disabled\n");
    }

    if ( DRF_VAL( _PMSENC, _FALCON_ITFEN, _MTHDEN, data32))
    {
        dprintf("lw: + LW_PMSENC_FALCON_ITFEN_MTHDEN enabled\n");
    }
    else
    {
        dprintf("lw: + LW_PMSENC_FALCON_ITFEN_MTHDEN disabled\n");
    }

    //check if falcon is hung (instr ptr)
    if ( pFalcon[indexGpu].falconTestPC(msencBaseAddress, "PMSENC") == LW_ERR_GENERIC )
    {
        dprintf("lw: Falcon instruction pointer is stuck or invalid\n");
        
        //TODO: treat falcon PC errors as warnings now, need to report as error
        addUnitErr("\t Warning: Falcon instruction pointer is stuck or invalid\n");
        //status = LW_ERR_GENERIC;
    }

    return status;  
}

//-----------------------------------------------------
// msencPrintPriv_v02_00
//-----------------------------------------------------
void msencPrintPriv_v02_00(LwU32 clmn, char *tag, LwU32 id)
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
    dprintf("(0x%08X)  = 0x%08X\n",id,MSENC_REG_RD32(id));
}

//-----------------------------------------------------
// msencDumpPriv_v02_00 - Dumps MSENC priv reg space
//-----------------------------------------------------
LW_STATUS msencDumpPriv_v02_00(LwU32 indexGpu)
{
    LwU32 u;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u MSENC priv registers -- \n", indexGpu);
    dprintf("lw:\n");

    for(u=0;;u++)
    {
        if(pMsencPrivReg[0][u].m_id==0)
        {
            break;
        }
        
        pMsenc[indexGpu].msencPrintPriv(40,pMsencPrivReg[0][u].m_tag,pMsencPrivReg[0][u].m_id);
    }
    return LW_OK; 
}

//--------------------------------------------------------
// msencDisplayHwcfg_v02_00 - Display MSENC HW config
//--------------------------------------------------------
LW_STATUS msencDisplayHwcfg_v02_00(LwU32 indexGpu)
{
    LwU32 hwcfg, hwcfg1;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u MSENC HWCFG -- \n", indexGpu);
    dprintf("lw:\n");

    hwcfg  = MSENC_REG_RD32(LW_PMSENC_FALCON_HWCFG);
    dprintf("lw: LW_PMSENC_FALCON_HWCFG:  0x%08x\n", hwcfg); 
    dprintf("lw:\n");
    dprintf("lw:  IMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",
            DRF_VAL(_PMSENC, _FALCON_HWCFG, _IMEM_SIZE, hwcfg),
            DRF_VAL(_PMSENC, _FALCON_HWCFG, _IMEM_SIZE, hwcfg)<<8); 
    dprintf("lw:  DMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",
            DRF_VAL(_PMSENC, _FALCON_HWCFG, _DMEM_SIZE, hwcfg), 
            DRF_VAL(_PMSENC, _FALCON_HWCFG, _DMEM_SIZE, hwcfg)<<8); 
    dprintf("lw:  METHODFIFO_DEPTH: 0x%08X\n", DRF_VAL(_PMSENC, _FALCON_HWCFG, _METHODFIFO_DEPTH, hwcfg)); 
    dprintf("lw:  DMAQUEUE_DEPTH:   0x%08X\n", DRF_VAL(_PMSENC, _FALCON_HWCFG, _DMAQUEUE_DEPTH, hwcfg)); 

    dprintf("lw:\n");

    hwcfg1 = MSENC_REG_RD32(LW_PMSENC_FALCON_HWCFG1);
    dprintf("lw: LW_PMSENC_FALCON_HWCFG1: 0x%08x\n", hwcfg1); 
    dprintf("lw:\n");
    dprintf("lw:  CORE_REV:         0x%08X\n", DRF_VAL(_PMSENC, _FALCON_HWCFG1, _CORE_REV, hwcfg1)); 
    dprintf("lw:  SELWRITY_MODEL:   0x%08X\n", DRF_VAL(_PMSENC, _FALCON_HWCFG1, _SELWRITY_MODEL, hwcfg1)); 
    dprintf("lw:  IMEM_PORTS:       0x%08X\n", DRF_VAL(_PMSENC, _FALCON_HWCFG1, _IMEM_PORTS, hwcfg1)); 
    dprintf("lw:  DMEM_PORTS:       0x%08X\n", DRF_VAL(_PMSENC, _FALCON_HWCFG1, _DMEM_PORTS, hwcfg1)); 
    dprintf("lw:  TAG_WIDTH:        0x%08X\n", DRF_VAL(_PMSENC, _FALCON_HWCFG1, _TAG_WIDTH, hwcfg1)); 

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
LW_STATUS  msencDisplayFlcnSPR_v02_00(LwU32 indexGpu)
{
    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u MSENC Special Purpose Registers -- \n", indexGpu);
    dprintf("lw:\n");

    MSENC_REG_WR32(LW_PMSENC_FALCON_ICD_CMD, 0x1008);
    dprintf("lw: MSENC IV0 :    0x%08x\n", MSENC_REG_RD32(LW_PMSENC_FALCON_ICD_RDATA)); 
    MSENC_REG_WR32(LW_PMSENC_FALCON_ICD_CMD, 0x1108);
    dprintf("lw: MSENC IV1 :    0x%08x\n", MSENC_REG_RD32(LW_PMSENC_FALCON_ICD_RDATA)); 
    MSENC_REG_WR32(LW_PMSENC_FALCON_ICD_CMD, 0x1308);
    dprintf("lw: MSENC EV  :    0x%08x\n", MSENC_REG_RD32(LW_PMSENC_FALCON_ICD_RDATA)); 
    MSENC_REG_WR32(LW_PMSENC_FALCON_ICD_CMD, 0x1408);
    dprintf("lw: MSENC SP  :    0x%08x\n", MSENC_REG_RD32(LW_PMSENC_FALCON_ICD_RDATA)); 
    MSENC_REG_WR32(LW_PMSENC_FALCON_ICD_CMD, 0x1508);
    dprintf("lw: MSENC PC  :    0x%08x\n", MSENC_REG_RD32(LW_PMSENC_FALCON_ICD_RDATA)); 
    MSENC_REG_WR32(LW_PMSENC_FALCON_ICD_CMD, 0x1608);
    dprintf("lw: MSENC IMB :    0x%08x\n", MSENC_REG_RD32(LW_PMSENC_FALCON_ICD_RDATA)); 
    MSENC_REG_WR32(LW_PMSENC_FALCON_ICD_CMD, 0x1708);
    dprintf("lw: MSENC DMB :    0x%08x\n", MSENC_REG_RD32(LW_PMSENC_FALCON_ICD_RDATA)); 
    MSENC_REG_WR32(LW_PMSENC_FALCON_ICD_CMD, 0x1808);
    dprintf("lw: MSENC CSW :    0x%08x\n", MSENC_REG_RD32(LW_PMSENC_FALCON_ICD_RDATA)); 
    dprintf("lw:\n\n");

    return LW_OK; 
}
