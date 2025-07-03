/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2017 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

//-----------------------------------------------------
//
// lwjpg0100.c - LWJPG v1 routines
// 
//-----------------------------------------------------

#include "ampere/ga100/dev_pri_ringstation_sys.h"
#include "ampere/ga100/dev_lwjpg_pri.h"
#include "ampere/ga100/dev_falcon_v4.h"
#include "ampere/ga100/dev_fifo.h"
#include "ampere/ga100/dev_master.h"

#include "lwjpg.h"
#include "g_lwjpg_private.h"     // (rmconfig)  implementation prototypes

#define LW_C4D1_NOP                                      (0x00000100)
#define LW_C4D1_PM_TRIGGER                               (0x00000140)
#define LW_C4D1_SET_APPLICATION_ID                       (0x00000200)
#define LW_C4D1_SET_WATCHDOG_TIMER                       (0x00000204)
#define LW_C4D1_SEMAPHORE_A                              (0x00000240)
#define LW_C4D1_SEMAPHORE_B                              (0x00000244)
#define LW_C4D1_SEMAPHORE_C                              (0x00000248)
#define LW_C4D1_CTX_SAVE_AREA                            (0x0000024C)
#define LW_C4D1_CTX_SWITCH                               (0x00000250)
#define LW_C4D1_EXELWTE                                  (0x00000300)
#define LW_C4D1_SEMAPHORE_D                              (0x00000304)
#define LW_C4D1_SET_CONTROL_PARAMS                       (0x00000700)
#define LW_C4D1_SET_TOTAL_CORE_NUM                       (0x00000704)
#define LW_C4D1_SET_IN_DRV_PIC_SETUP                     (0x00000708)
#define LW_C4D1_SET_PER_CORE_SET_OUT_STATUS              (0x0000070C)
#define LW_C4D1_SET_PER_CORE_SET_CORE_INDEX(i)           (0x00000710+(i)*0x20)
#define LW_C4D1_SET_PER_CORE_SET_BITSTREAM(i)            (0x00000714+(i)*0x20)
#define LW_C4D1_SET_PER_CORE_SET_LWR_PIC(i)              (0x00000718+(i)*0x20)
#define LW_C4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_U(i)     (0x0000071C+(i)*0x20)
#define LW_C4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_V(i)     (0x00000720+(i)*0x20)
#define LW_C4D1_PM_TRIGGER_END                           (0x00001114)


dbg_lwjpg_v02_00 lwjpgMethodTable_v02_00[] =
{
    privInfo_lwjpg_v02_00(LW_C4D1_NOP),
    privInfo_lwjpg_v02_00(LW_C4D1_PM_TRIGGER),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_APPLICATION_ID),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_WATCHDOG_TIMER),
    privInfo_lwjpg_v02_00(LW_C4D1_SEMAPHORE_A),
    privInfo_lwjpg_v02_00(LW_C4D1_SEMAPHORE_B),
    privInfo_lwjpg_v02_00(LW_C4D1_SEMAPHORE_C),
    privInfo_lwjpg_v02_00(LW_C4D1_EXELWTE),
    privInfo_lwjpg_v02_00(LW_C4D1_SEMAPHORE_D),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_CONTROL_PARAMS),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_TOTAL_CORE_NUM),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_IN_DRV_PIC_SETUP),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_OUT_STATUS),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_CORE_INDEX(0)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_BITSTREAM(0)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_LWR_PIC(0)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_U(0)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_V(0)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_CORE_INDEX(1)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_BITSTREAM(1)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_LWR_PIC(1)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_U(1)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_V(1)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_CORE_INDEX(2)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_BITSTREAM(2)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_LWR_PIC(2)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_U(2)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_V(2)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_CORE_INDEX(3)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_BITSTREAM(3)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_LWR_PIC(3)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_U(3)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_V(3)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_CORE_INDEX(4)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_BITSTREAM(4)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_LWR_PIC(4)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_U(4)),
    privInfo_lwjpg_v02_00(LW_C4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_V(4)),
    privInfo_lwjpg_v02_00(LW_C4D1_PM_TRIGGER_END),
    privInfo_lwjpg_v02_00(0),
};

dbg_lwjpg_v02_00 lwjpgPrivReg_v02_00[] =
{
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQSSET),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQSCLR),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQSTAT),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQMODE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQMSET),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQMCLR),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQMASK),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQDEST),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_GPTMRINT),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_GPTMRVAL),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_GPTMRCTL),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_PTIMER0),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_PTIMER1),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_WDTMRVAL),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_WDTMRCTL),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MTHDDATA),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MTHDID),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MTHDWDAT),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MTHDCOUNT),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MTHDPOP),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MTHDRAMSZ),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_LWRCTX),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_NXTCTX),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_CTXACK),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MAILBOX0),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MAILBOX1),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_ITFEN),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IDLESTATE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_FHSTATE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_PRIVSTATE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_SFTRESET),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_OS),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_RM),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_SOFT_PM),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_SOFT_MODE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DEBUG1),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DEBUGINFO),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IBRKPT1),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IBRKPT2),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IBRKPT3),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IBRKPT4),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IBRKPT5),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_CGCTL),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_ENGCTL),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_PMM),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_ADDR),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_CPUCTL),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_BOOTVEC),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_HWCFG),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_HWCFG1),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMACTL),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMATRFBASE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMATRFMOFFS),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMATRFCMD),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMATRFFBOFFS),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMAPOLL_FB),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMAPOLL_CP),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMCTL),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMSTAT),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_TRACEIDX),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_TRACEPC),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_ICD_CMD),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_ICD_ADDR),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_ICD_WDATA),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_ICD_RDATA),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMC(0)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMC(1)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMC(2)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMC(3)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMD(0)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMD(1)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMD(2)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMD(3)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMT(0)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMT(1)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMT(2)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMT(3)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(0)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(1)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(2)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(3)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(4)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(5)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(6)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(7)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(0)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(1)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(2)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(3)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(4)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(5)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(6)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(7)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(0)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(1)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(2)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(3)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(4)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(5)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(6)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(7)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_INSTBLK),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_CTL),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_THROTTLE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_ACHK_BLK(0)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_ACHK_BLK(1)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_ACHK_CTL(0)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_ACHK_CTL(1)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_TOP_CTL),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_TOP_RST_INTERVAL),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_FDCT_QT_ADDR),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_FDCT_QT_WDATA),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_FDCT_QT_RDATA),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_FDCT_RTCTRL),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_VLE_HT_ADDR),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_VLE_HT_WDATA),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_VLE_HT_RDATA),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_VLE_RTCTRL),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_PDMA_PIC_INFO),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_PDMA_PIC_INFO1),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_PDMA_PIC_LUMA_STRIDE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_PDMA_PIC_CHROMA_STRIDE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_PDMA_DEC_PACKET_INFO),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_SDMA_OFFSET),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_SDMA_SIZE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_FRAME_SIZE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_FRAME_PARAM),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_BLOCK_CTL(0)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_BLOCK_CTL(1)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_BLOCK_CTL(2)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_BLOCK_CTL(3)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_HUFFMAN_MINCODE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_HUFFMAN_SYMBOL),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_OUTPUT_INFO),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_YUV2RGB_PARAM(0)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_YUV2RGB_PARAM(1)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_YUV2RGB_PARAM(2)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_YUV2RGB_PARAM(3)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_YUV2RGB_PARAM(4)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_YUV2RGB_PARAM(5)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(0)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(1)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(2)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(3)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(4)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(5)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(6)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(7)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(8)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(9)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(10)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(11)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(12)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(13)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(14)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(15)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(16)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(17)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(18)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(19)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(20)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(21)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(22)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(23)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(24)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(25)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(26)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(27)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(28)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(29)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(30)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(31)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(32)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(33)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(34)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(35)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(36)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(37)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(38)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(39)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(40)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(41)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(42)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(43)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(44)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(45)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(46)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(47)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(48)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(49)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(50)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(51)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(52)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(53)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(54)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(55)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(56)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(57)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(58)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(59)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(60)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(61)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(62)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(63)),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_INDEX),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_PDMA_LUMA_BASE_ADDR),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_PDMA_CHROMA_U_BASE_ADDR),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_PDMA_CHROMA_V_BASE_ADDR),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_SDMA_BASE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_CYCLE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_STATUS),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_PDMA_PIC_INFO2_POS_8X8),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_DEC_READ_BYTES_OFFSET),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_HUFFMAN_TABLE_READ_ADDR),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_CFG),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_DEC_CAPABILITY),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_ENC_CAPABILITY),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_KICK_OFF),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_PIC_DONE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_ERROR),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_CYA),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_INTEN),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_INTR),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_SEPARATE_CORE_RESET),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_PARTIAL_CORE_RESET),
    privInfo_lwjpg_v02_00(LW_PLWJPG_MISC_BLCG),
    privInfo_lwjpg_v02_00(LW_PLWJPG_MISC_BLCG1),
    privInfo_lwjpg_v02_00(LW_PLWJPG_IP_VER),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CDMA_CFG),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CDMA_BASE),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CDMA_INTEN),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CDMA_INTR),
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CDMA_CORE_SET),
    privInfo_lwjpg_v02_00(LW_PLWJPG_PMM_MS),
    privInfo_lwjpg_v02_00(0),
};

//-----------------------------------------------------
// lwjpgIsSupported_v02_00
//-----------------------------------------------------
BOOL lwjpgIsSupported_v02_00(LwU32 indexGpu, LwU32 engineId)
{
    if (engineId != LWWATCH_LWJPG_0)
        return FALSE;

    pLwjpgPrivReg[engineId] = lwjpgPrivReg_v02_00;
    pLwjpgMethodTable = lwjpgMethodTable_v02_00;
    return TRUE;
}

//-----------------------------------------------------
// lwjpgIsPrivBlocked_v02_00
//-----------------------------------------------------
BOOL lwjpgIsPrivBlocked_v02_00(LwU32 indexGpu, LwU32 engineId)
{
    LwU32 idx;
    LwU32 bitmask;
    LwU32 regSysPrivFsConfig;

    if (engineId != LWWATCH_LWJPG_0)
        return TRUE;

    // Bit-fields within LW_PPRIV_SYS_PRIV_FS_CONFIG denote priv access for video.
    // All video engines must have priv access for lwjpg command support.
    idx = LW_PPRIV_SYS_PRI_MASTER_fecs2lwjpg_pri >> 5;
    regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));

    bitmask = BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2lwjpg_pri - (idx << 5));

    return ((regSysPrivFsConfig & bitmask) != bitmask);
}

//-----------------------------------------------------
// lwjpgPrintMethodData_v02_00
//-----------------------------------------------------
void lwjpgPrintMethodData_v02_00(LwU32 clmn, char *tag, LwU32 method, LwU32 data)
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
// lwjpgDumpImem_v02_00 - Dumps LWJPG instruction memory
//-----------------------------------------------------
LW_STATUS lwjpgDumpImem_v02_00(LwU32 indexGpu, LwU32 engineId, LwU32 imemSize)
{
    LW_STATUS status = LW_OK;
    LwU32  imemSizeMax;
    LwU32 addrssImem=LW_PLWJPG_FALCON_IMEMD(0);
    LwU32 address2Imem=LW_PLWJPG_FALCON_IMEMC(0);
    LwU32 address2Imemt = LW_PLWJPG_FALCON_IMEMT(0);
    LwU32 u;
    LwU32 blk=0;

    imemSizeMax = (GPU_REG_RD_DRF(_PLWJPG_FALCON, _HWCFG, _IMEM_SIZE)<<8) ;
    if (imemSize > 0)
        imemSize = min(imemSizeMax, imemSize);
    else
        imemSize = imemSizeMax;

    dprintf("\n");
    dprintf("lw: -- Gpu %u LWJPG IMEM -- \n", indexGpu);    
    dprintf("lw: -- Gpu %u LWJPG IMEM SIZE =  0x%08x-- \n", indexGpu,imemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    for(u=0;u<(imemSize+3)/4;u++)
    {
        LwU32 i;
        if((u%64)==0) {
            GPU_REG_WR32(address2Imemt, blk++);
        }
        i = (u<<(0?LW_PLWJPG_FALCON_IMEMC_OFFS));
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
// lwjpgDumpDmem_v02_00 - Dumps LWJPG data memory
//-----------------------------------------------------
LW_STATUS lwjpgDumpDmem_v02_00(LwU32 indexGpu, LwU32 engineId, LwU32 dmemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 dmemSizeMax;
    // these are the variables defined for use in parsing and printinf the methods and data
    LwU32 addrss, address2, u, i, comMthdOffs = 0, appMthdOffs = 0, classNum;
    LwU32 comMthd[CMNMETHODARRAYSIZE_LWJPG] = {0};
    LwU32 methodIdx;

    dmemSizeMax = (GPU_REG_RD_DRF(_PLWJPG_FALCON, _HWCFG, _DMEM_SIZE)<<8) ;

    if(dmemSize > 0)
        dmemSize = min(dmemSizeMax, dmemSize);
    else
       dmemSize = dmemSizeMax;

    addrss      = LW_PLWJPG_FALCON_DMEMD(0);
    address2    = LW_PLWJPG_FALCON_DMEMC(0);
    classNum    = 0xC4D1;

    dprintf("\n");
    dprintf("lw: -- Gpu %u LWJPG DMEM -- \n", indexGpu);
    dprintf("lw: -- Gpu %u LWJPG DMEM SIZE =  0x%08x-- \n", indexGpu,dmemSize);
    //dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    

    for(u=0;u<(dmemSize+3)/4;u++)
    {
        i = (u<<(0?LW_PLWJPG_FALCON_DMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        if((u%8==0))
        {
            dprintf("\n%04X: ", 4*u);
        }
        dprintf("%08X ",  GPU_REG_RD32(addrss));
    }

    // get methods offset are in the DWORD#0 in dmem
    u = (0<<(0?LW_PLWJPG_FALCON_DMEMC_OFFS));
    GPU_REG_WR32(address2,u);
    comMthdOffs = (GPU_REG_RD32(addrss)) >> 2;

    for(u=0; u<CMNMETHODARRAYSIZE_LWJPG;u++)
    {
        i = ((u+comMthdOffs)<<(0?LW_PLWJPG_FALCON_DMEMC_OFFS));
        GPU_REG_WR32(address2,i);
        comMthd[u] = GPU_REG_RD32(addrss);
    }

    dprintf("\n\n-----------------------------------------------------------------------\n");
    dprintf("%4s, %8s,    %4s, %8s,    %4s, %8s,    %4s, %8s\n", 
            "Mthd", "Data", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data");
    dprintf("[COMMON METHODS]\n");
    for (u=0; u<CMNMETHODARRAYSIZE_LWJPG; u++)
    {
        dprintf("%04X: %08X", CMNMETHODBASE_LWJPG_v02+4*u, comMthd[u]);
        if (((u % 4) == 3) || u == (CMNMETHODARRAYSIZE_LWJPG - 1))
        {
            dprintf("\n");
        }
        else
        {
            dprintf(",    ");
        }
    }
    dprintf("\n");

    // common methods
    // if this environment variable is present, parse and print out the methods
    if (getelw("LWW_CLASS_SDK") != NULL)
    {
        dprintf("\n[COMMON METHODS]\n");
        for (u = 0; u<CMNMETHODARRAYSIZE_LWJPG; u++)
        {
            if(parseClassHeader(classNum, CMNMETHODBASE_LWJPG_v02+4*u, comMthd[u]))
                dprintf("\n");
        }
        dprintf("\n");
    }
    else
    {
        dprintf("\n[COMMON METHODS]\n");
        for (u = 0; u<CMNMETHODARRAYSIZE_LWJPG; u++)
        {
            for(methodIdx=0;;methodIdx++)
            {
                if(pLwjpgMethodTable[methodIdx].m_id == (CMNMETHODBASE_LWJPG_v02+4*u))
                {
                    lwjpgPrintMethodData_v02_00(48,
                                                pLwjpgMethodTable[methodIdx].m_tag, 
                                                pLwjpgMethodTable[methodIdx].m_id, 
                                                comMthd[u]);
                    break;
                }
                else if (pLwjpgMethodTable[methodIdx].m_id == 0)
                {
                    break;
                }
            }
        }
        dprintf("\n");
        dprintf("\nDefine the LWW_CLASS_SDK environment variable to the location "
                "of the class header files to view parsed methods and data \n");
    }
    return status;  
}

//-----------------------------------------------------
// lwjpgTestState_v02_00 - Test basic lwjpg state
//-----------------------------------------------------
LW_STATUS lwjpgTestState_v02_00(LwU32 indexGpu, LwU32 engineId)
{
    LW_STATUS    status = LW_OK;
    LwU32   regIntr;
    LwU32   regIntrEn;
    LwU32   data32;

    //check falcon interrupts
    regIntr = GPU_REG_RD32(LW_PLWJPG_FALCON_IRQSTAT);
    regIntrEn = GPU_REG_RD32(LW_PLWJPG_FALCON_IRQMASK);
    regIntr &= regIntrEn;

    if ( !DRF_VAL(_PLWJPG, _FALCON_IRQMASK, _GPTMR, regIntrEn))
        dprintf("lw: LW_PLWJPG_FALCON_IRQMASK_GPTMR disabled\n");

    if ( !DRF_VAL(_PLWJPG, _FALCON_IRQMASK, _WDTMR, regIntrEn))
        dprintf("lw: LW_PLWJPG_FALCON_IRQMASK_WDTMR disabled\n");

    if ( !DRF_VAL(_PLWJPG, _FALCON_IRQMASK, _MTHD, regIntrEn))
        dprintf("lw: LW_PLWJPG_FALCON_IRQMASK_MTHD disabled\n");

    if ( !DRF_VAL(_PLWJPG, _FALCON_IRQMASK, _CTXSW, regIntrEn))
        dprintf("lw: LW_PLWJPG_FALCON_IRQMASK_CTXSW disabled\n");

    if ( !DRF_VAL(_PLWJPG, _FALCON_IRQMASK, _HALT, regIntrEn))
        dprintf("lw: LW_PLWJPG_FALCON_IRQMASK_HALT disabled\n");

    if ( !DRF_VAL(_PLWJPG, _FALCON_IRQMASK, _EXTERR, regIntrEn))
        dprintf("lw: LW_PLWJPG_FALCON_IRQMASK_EXTERR disabled\n");

    if ( !DRF_VAL(_PLWJPG, _FALCON_IRQMASK, _SWGEN0, regIntrEn))
        dprintf("lw: LW_PLWJPG_FALCON_IRQMASK_SWGEN0 disabled\n");

    if ( !DRF_VAL(_PLWJPG, _FALCON_IRQMASK, _SWGEN1, regIntrEn))
        dprintf("lw: LW_PLWJPG_FALCON_IRQMASK_SWGEN1 disabled\n");

   
    //if any interrupt pending, set error
    if (regIntr != 0)
    {
        addUnitErr("\t LWJPG interrupts are pending\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWJPG,_FALCON_IRQSTAT, _GPTMR, regIntr))
    {
        dprintf("lw: LW_PLWJPG_FALCON_IRQSTAT_GPTMR pending\n");

        dprintf("lw: LW_PLWJPG_FALCON_GPTMRINT:    0x%08x\n", 
            GPU_REG_RD32(LW_PLWJPG_FALCON_GPTMRINT) );
        dprintf("lw: LW_PLWJPG_FALCON_GPTMRVAL:    0x%08x\n", 
            GPU_REG_RD32(LW_PLWJPG_FALCON_GPTMRVAL) );
        
    }
    
    if ( DRF_VAL( _PLWJPG,_FALCON_IRQSTAT, _WDTMR, regIntr))
    {
        dprintf("lw: LW_PLWJPG_FALCON_IRQSTAT_WDTMR pending\n");
    }

    if ( DRF_VAL( _PLWJPG,_FALCON_IRQSTAT, _MTHD, regIntr))
    {
        dprintf("lw: LW_PLWJPG_FALCON_IRQSTAT_MTHD pending\n");

        dprintf("lw: LW_PLWJPG_FALCON_MTHDDATA_DATA:    0x%08x\n", 
            GPU_REG_RD32(LW_PLWJPG_FALCON_MTHDDATA) );
        
        data32 = GPU_REG_RD32(LW_PLWJPG_FALCON_MTHDID);
        dprintf("lw: LW_PLWJPG_FALCON_MTHDID_ID:    0x%08x\n", 
           DRF_VAL( _PLWJPG,_FALCON_MTHDID, _ID, data32)  );
        dprintf("lw: LW_PLWJPG_FALCON_MTHDID_SUBCH:    0x%08x\n", 
           DRF_VAL( _PLWJPG,_FALCON_MTHDID, _SUBCH, data32)  );
        dprintf("lw: LW_PLWJPG_FALCON_MTHDID_PRIV:    0x%08x\n", 
           DRF_VAL( _PLWJPG,_FALCON_MTHDID, _PRIV, data32)  );
    }
    
    if ( DRF_VAL( _PLWJPG,_FALCON_IRQSTAT, _CTXSW, regIntr))
    {
        dprintf("lw: LW_PLWJPG_FALCON_IRQSTAT_CTXSW pending\n");
    }
    
    if ( DRF_VAL( _PLWJPG,_FALCON_IRQSTAT, _HALT, regIntr))
    {
        dprintf("lw: LW_PLWJPG_FALCON_IRQSTAT_HALT pending\n");
    }
    
    if ( DRF_VAL( _PLWJPG,_FALCON_IRQSTAT, _EXTERR, regIntr))
    {
        dprintf("lw: LW_PLWJPG_FALCON_IRQSTAT_EXTERR pending\n");
    }
    
    if ( DRF_VAL( _PLWJPG,_FALCON_IRQSTAT, _SWGEN0, regIntr))
    {
        dprintf("lw: LW_PLWJPG_FALCON_IRQSTAT_SWGEN0 pending\n");

        pFalcon[indexGpu].falconPrintMailbox(/*LW_FALCON_LWJPG_BASE*/0x828000);
    }

    if ( DRF_VAL( _PLWJPG,_FALCON_IRQSTAT, _SWGEN1, regIntr))
    {
        dprintf("lw: LW_PLWJPG_FALCON_IRQSTAT_SWGEN1 pending\n");
    }

     //
    //print falcon states
    //Bit |  Signal meaning
    //0      FALCON busy
    //

    data32 = GPU_REG_RD32(LW_PLWJPG_FALCON_IDLESTATE);

    if ( DRF_VAL( _PLWJPG, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_PLWJPG_FALCON_IDLESTATE_FALCON_BUSY\n");
        addUnitErr("\t LW_PLWJPG_FALCON_IDLESTATE_FALCON_BUSY\n");
        status = LW_ERR_GENERIC;
    }

  
    data32 = GPU_REG_RD32(LW_PLWJPG_FALCON_FHSTATE);
 
    if ( DRF_VAL( _PLWJPG, _FALCON_FHSTATE, _FALCON_HALTED, data32))
    {
        dprintf("lw: + LW_PLWJPG_FALCON_FHSTATE_FALCON_HALTED\n");
        addUnitErr("\t LW_PLWJPG_FALCON_FHSTATE_FALCON_HALTED\n");
        status = LW_ERR_GENERIC;
    }
    
    if ( DRF_VAL( _PLWJPG, _FALCON_FHSTATE, _ENGINE_FAULTED, data32))
    {
        dprintf("lw: + LW_PLWJPG_FALCON_FHSTATE_ENGINE_FAULTED\n");
        addUnitErr("\t LW_PLWJPG_FALCON_FHSTATE_ENGINE_FAULTED\n");
        status = LW_ERR_GENERIC;
    }
    
    if ( DRF_VAL( _PLWJPG, _FALCON_FHSTATE, _STALL_REQ, data32))
    {
        dprintf("lw: + LW_PLWJPG_FALCON_FHSTATE_STALL_REQ\n");
        addUnitErr("\t LW_PLWJPG_FALCON_FHSTATE_STALL_REQ\n");
        status = LW_ERR_GENERIC;
    }

    //print falcon ctl regs
    data32 = GPU_REG_RD32(LW_PLWJPG_FALCON_ENGCTL);
    
    if ( DRF_VAL( _PLWJPG, _FALCON_ENGCTL, _ILW_CONTEXT, data32))
    {
        dprintf("lw: + LW_PLWJPG_FALCON_ENGCTL_ILW_CONTEXT\n");
        addUnitErr("\t LW_PLWJPG_FALCON_ENGCTL_ILW_CONTEXT\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWJPG, _FALCON_ENGCTL, _STALLREQ, data32))
    {
        dprintf("lw: + LW_PLWJPG_FALCON_ENGCTL_STALLREQ\n");
        addUnitErr("\t LW_PLWJPG_FALCON_ENGCTL_STALLREQ\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_PLWJPG_FALCON_CPUCTL);

    if ( DRF_VAL( _PLWJPG, _FALCON_CPUCTL, _IILWAL, data32))
    {
        dprintf("lw: + LW_PLWJPG_FALCON_CPUCTL_IILWAL\n");
        addUnitErr("\t LW_PLWJPG_FALCON_CPUCTL_IILWAL\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWJPG, _FALCON_CPUCTL, _HALTED, data32))
    {
        dprintf("lw: + LW_PLWJPG_FALCON_CPUCTL_HALTED\n");
        addUnitErr("\t LW_PLWJPG_FALCON_CPUCTL_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL( _PLWJPG, _FALCON_CPUCTL, _STOPPED, data32))
    {
        dprintf("lw: + LW_PLWJPG_FALCON_CPUCTL_STOPPED\n");
        addUnitErr("\t Warning: LW_PLWJPG_FALCON_CPUCTL_STOPPED\n");
        //status = LW_ERR_GENERIC;
    }

    // state of mthd/ctx interface 
    data32 = GPU_REG_RD32(LW_PLWJPG_FALCON_ITFEN);

    if (DRF_VAL( _PLWJPG, _FALCON_ITFEN, _CTXEN, data32))
    {
        dprintf("lw: + LW_PLWJPG_FALCON_ITFEN_CTXEN enabled\n");
             
        if (pFalcon[indexGpu].falconTestCtxState(/*LW_FALCON_LWJPG_BASE*/0x828000, "PLWJPG") == LW_ERR_GENERIC)
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
        dprintf("lw: + LW_PLWJPG_FALCON_ITFEN_CTXEN disabled\n");
    }

    if ( DRF_VAL( _PLWJPG, _FALCON_ITFEN, _MTHDEN, data32))
    {
        dprintf("lw: + LW_PLWJPG_FALCON_ITFEN_MTHDEN enabled\n");
    }
    else
    {
        dprintf("lw: + LW_PLWJPG_FALCON_ITFEN_MTHDEN disabled\n");
    }

    //check if falcon is hung (instr ptr)
    if ( pFalcon[indexGpu].falconTestPC(/*LW_FALCON_LWJPG_BASE*/0x828000, "PLWJPG") == LW_ERR_GENERIC )
    {
        dprintf("lw: Falcon instruction pointer is stuck or invalid\n");
        
        //TODO: treat falcon PC errors as warnings now, need to report as error
        addUnitErr("\t Warning: Falcon instruction pointer is stuck or invalid\n");
        //status = LW_ERR_GENERIC;
    }

    return status;  
}

//-----------------------------------------------------
// lwjpgPrintPriv_v02_00
//-----------------------------------------------------
void lwjpgPrintPriv_v02_00(LwU32 clmn, char *tag, LwU32 id)
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
// lwjpgDumpPriv_v02_00 - Dumps LWJPG priv reg space
//-----------------------------------------------------
LW_STATUS lwjpgDumpPriv_v02_00(LwU32 indexGpu, LwU32 engineId)
{
    LwU32 u;

    if (!pLwjpgPrivReg[engineId])
    {
        dprintf("lw: -- Gpu %u LWJPG error: priv reg array uninitialized\n", indexGpu);
        return LW_ERR_ILWALID_PARAMETER;
    }

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWJPG priv registers -- \n", indexGpu);
    dprintf("lw:\n");

    for(u=0;;u++)
    {
        if (pLwjpgPrivReg[engineId][u].m_id == 0)
        {
            break;
        }
        pLwjpg[indexGpu].lwjpgPrintPriv(61, pLwjpgPrivReg[engineId][u].m_tag, pLwjpgPrivReg[engineId][u].m_id);
    }
    return LW_OK; 
}

//--------------------------------------------------------
// lwjpgDisplayHwcfg_v02_00 - Display LWJPG HW config
//--------------------------------------------------------
LW_STATUS lwjpgDisplayHwcfg_v02_00(LwU32 indexGpu, LwU32 engineId)
{
    LwU32 hwcfg, hwcfg1;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWJPG HWCFG -- \n", indexGpu);
    dprintf("lw:\n");

    hwcfg  = GPU_REG_RD32(LW_PLWJPG_FALCON_HWCFG);
    dprintf("lw: LW_PLWJPG_FALCON_HWCFG:  0x%08x\n", hwcfg); 
    dprintf("lw:\n");
    dprintf("lw:  IMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",
            DRF_VAL(_PLWJPG, _FALCON_HWCFG, _IMEM_SIZE, hwcfg),
            DRF_VAL(_PLWJPG, _FALCON_HWCFG, _IMEM_SIZE, hwcfg)<<8); 
    dprintf("lw:  DMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",
            DRF_VAL(_PLWJPG, _FALCON_HWCFG, _DMEM_SIZE, hwcfg), 
            DRF_VAL(_PLWJPG, _FALCON_HWCFG, _DMEM_SIZE, hwcfg)<<8); 
    dprintf("lw:  METHODFIFO_DEPTH: 0x%08X\n", DRF_VAL(_PLWJPG, _FALCON_HWCFG, _METHODFIFO_DEPTH, hwcfg)); 
    dprintf("lw:  DMAQUEUE_DEPTH:   0x%08X\n", DRF_VAL(_PLWJPG, _FALCON_HWCFG, _DMAQUEUE_DEPTH, hwcfg)); 

    dprintf("lw:\n");

    hwcfg1 = GPU_REG_RD32(LW_PLWJPG_FALCON_HWCFG1);
    dprintf("lw: LW_PLWJPG_FALCON_HWCFG1: 0x%08x\n", hwcfg1); 
    dprintf("lw:\n");
    dprintf("lw:  CORE_REV:         0x%08X\n", DRF_VAL(_PLWJPG, _FALCON_HWCFG1, _CORE_REV, hwcfg1)); 
    dprintf("lw:  SELWRITY_MODEL:   0x%08X\n", DRF_VAL(_PLWJPG, _FALCON_HWCFG1, _SELWRITY_MODEL, hwcfg1)); 
    dprintf("lw:  IMEM_PORTS:       0x%08X\n", DRF_VAL(_PLWJPG, _FALCON_HWCFG1, _IMEM_PORTS, hwcfg1)); 
    dprintf("lw:  DMEM_PORTS:       0x%08X\n", DRF_VAL(_PLWJPG, _FALCON_HWCFG1, _DMEM_PORTS, hwcfg1)); 
    dprintf("lw:  TAG_WIDTH:        0x%08X\n", DRF_VAL(_PLWJPG, _FALCON_HWCFG1, _TAG_WIDTH, hwcfg1)); 

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
LW_STATUS  lwjpgDisplayFlcnSPR_v02_00(LwU32 indexGpu, LwU32 engineId)
{
    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWJPG Special Purpose Registers -- \n", indexGpu);
    dprintf("lw:\n");

    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD, 0x1008);
    dprintf("lw: LWJPG IV0 :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA)); 
    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD, 0x1108);
    dprintf("lw: LWJPG IV1 :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA)); 
    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD, 0x1308);
    dprintf("lw: LWJPG EV  :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA)); 
    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD, 0x1408);
    dprintf("lw: LWJPG SP  :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA)); 
    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD, 0x1508);
    dprintf("lw: LWJPG PC  :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA)); 
    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD, 0x1608);
    dprintf("lw: LWJPG IMB :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA)); 
    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD, 0x1708);
    dprintf("lw: LWJPG DMB :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA)); 
    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD, 0x1808);
    dprintf("lw: LWJPG CSW :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA)); 
    dprintf("lw:\n\n");

    return LW_OK; 
}

//-----------------------------------------------------
// lwjpgGetClassId_v02_00
//-----------------------------------------------------
LwU32
lwjpgGetClassId_v02_00(void)
{
    return 0x0000C4D1;/*LWC4D1_VIDEO_LWJPG*/
}

