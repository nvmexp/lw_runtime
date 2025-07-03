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

#include "hopper/gh100/dev_pri_ringstation_sys.h"
#include "hopper/gh100/dev_pri_ringstation_sysb.h"
#include "hopper/gh100/dev_pri_ringstation_sysc.h"
#include "hopper/gh100/dev_lwjpg_pri_sw.h"
#include "hopper/gh100/dev_falcon_v4.h"
#include "hopper/gh100/dev_fifo.h"
#include "hopper/gh100/dev_fuse.h"
#include "hopper/gh100/dev_master.h"

#include "lwjpg.h"
#include "chip.h"
#include "g_lwjpg_private.h"     // (rmconfig)  implementation prototypes

// TODO: Use LW_FALCON_LWJPG0_BASE when available in drivers\common\inc\hwref\hopper\gh100\dev_falcon_*.h
#define FALCON_LWJPG_BASE(id)  (/*LW_FALCON_LWJPG0_BASE*/ 0x828000 + (id * 0x4000))

#define LW_B8D1_NOP                                      (0x00000100)
#define LW_B8D1_PM_TRIGGER                               (0x00000140)
#define LW_B8D1_SET_APPLICATION_ID                       (0x00000200)
#define LW_B8D1_SET_WATCHDOG_TIMER                       (0x00000204)
#define LW_B8D1_SEMAPHORE_A                              (0x00000240)
#define LW_B8D1_SEMAPHORE_B                              (0x00000244)
#define LW_B8D1_SEMAPHORE_C                              (0x00000248)
#define LW_B8D1_CTX_SAVE_AREA                            (0x0000024C)
#define LW_B8D1_CTX_SWITCH                               (0x00000250)
#define LW_B8D1_EXELWTE                                  (0x00000300)
#define LW_B8D1_SEMAPHORE_D                              (0x00000304)
#define LW_B8D1_SET_CONTROL_PARAMS                       (0x00000700)
#define LW_B8D1_SET_TOTAL_CORE_NUM                       (0x00000704)
#define LW_B8D1_SET_IN_DRV_PIC_SETUP                     (0x00000708)
#define LW_B8D1_SET_PER_CORE_SET_OUT_STATUS              (0x0000070C)
#define LW_B8D1_SET_PER_CORE_SET_CORE_INDEX(i)           (0x00000710+(i)*0x20)
#define LW_B8D1_SET_PER_CORE_SET_BITSTREAM(i)            (0x00000714+(i)*0x20)
#define LW_B8D1_SET_PER_CORE_SET_LWR_PIC(i)              (0x00000718+(i)*0x20)
#define LW_B8D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_U(i)     (0x0000071C+(i)*0x20)
#define LW_B8D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_V(i)     (0x00000720+(i)*0x20)
#define LW_B8D1_PM_TRIGGER_END                           (0x00001114)


dbg_lwjpg_v02_00 lwjpgMethodTable_v02_02[] =
{
    privInfo_lwjpg_v02_00(LW_B8D1_NOP),
    privInfo_lwjpg_v02_00(LW_B8D1_PM_TRIGGER),
    privInfo_lwjpg_v02_00(LW_B8D1_SET_APPLICATION_ID),
    privInfo_lwjpg_v02_00(LW_B8D1_SET_WATCHDOG_TIMER),
    privInfo_lwjpg_v02_00(LW_B8D1_SEMAPHORE_A),
    privInfo_lwjpg_v02_00(LW_B8D1_SEMAPHORE_B),
    privInfo_lwjpg_v02_00(LW_B8D1_SEMAPHORE_C),
    privInfo_lwjpg_v02_00(LW_B8D1_EXELWTE),
    privInfo_lwjpg_v02_00(LW_B8D1_SEMAPHORE_D),
    privInfo_lwjpg_v02_00(LW_B8D1_SET_CONTROL_PARAMS),
    privInfo_lwjpg_v02_00(LW_B8D1_SET_TOTAL_CORE_NUM),
    privInfo_lwjpg_v02_00(LW_B8D1_SET_IN_DRV_PIC_SETUP),
    privInfo_lwjpg_v02_00(LW_B8D1_SET_PER_CORE_SET_OUT_STATUS),
    privInfo_lwjpg_v02_00(LW_B8D1_SET_PER_CORE_SET_CORE_INDEX(0)),
    privInfo_lwjpg_v02_00(LW_B8D1_SET_PER_CORE_SET_BITSTREAM(0)),
    privInfo_lwjpg_v02_00(LW_B8D1_SET_PER_CORE_SET_LWR_PIC(0)),
    privInfo_lwjpg_v02_00(LW_B8D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_U(0)),
    privInfo_lwjpg_v02_00(LW_B8D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_V(0)),
    privInfo_lwjpg_v02_00(LW_B8D1_PM_TRIGGER_END),
    privInfo_lwjpg_v02_00(0),
};

dbg_lwjpg_v02_00 lwjpgFuseReg_v02_02[] =
{
    // revocation
    privInfo_lwjpg_v02_00(LW_FUSE_OPT_FUSE_UCODE_LWJPG_REV),
    
    // debug vs prod
    privInfo_lwjpg_v02_00(LW_FUSE_OPT_LWJPG_PRIV_READ_DIS),
    privInfo_lwjpg_v02_00(LW_FUSE_OPT_LWJPG_PRIV_WRITE_DIS),
    privInfo_lwjpg_v02_00(LW_FUSE_OPT_LWJPG_ENGINE_RESET_FSM_HALT_CYA_DISABLE),
    
    // floorsweeping
    privInfo_lwjpg_v02_00(LW_FUSE_OPT_LWJPG_DISABLE),
    privInfo_lwjpg_v02_00(LW_FUSE_OPT_LWJPG_DISABLE_CP),
    privInfo_lwjpg_v02_00(LW_FUSE_OPT_LWJPG_DEFECTIVE),
    privInfo_lwjpg_v02_00(LW_FUSE_CTRL_OPT_LWJPG),
    privInfo_lwjpg_v02_00(LW_FUSE_STATUS_OPT_LWJPG),
    privInfo_lwjpg_v02_00(0)
};

#define LWJPG_PRIV_ACCESS(id)      \
{\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQSSET(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQSCLR(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQSTAT(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQMODE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQMSET(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQMCLR(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQMASK(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IRQDEST(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_GPTMRINT(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_GPTMRVAL(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_GPTMRCTL(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_PTIMER0(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_PTIMER1(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_WDTMRVAL(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_WDTMRCTL(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MTHDDATA(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MTHDID(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MTHDWDAT(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MTHDCOUNT(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MTHDPOP(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MTHDRAMSZ(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_LWRCTX(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_NXTCTX(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_CTXACK(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MAILBOX0(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_MAILBOX1(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_ITFEN(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IDLESTATE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_FHSTATE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_PRIVSTATE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_SFTRESET(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_OS(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_RM(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_SOFT_PM(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_SOFT_MODE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DEBUG1(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DEBUGINFO(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IBRKPT1(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IBRKPT2(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IBRKPT3(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IBRKPT4(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IBRKPT5(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_CGCTL(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_ENGCTL(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_PMM(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_ADDR(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_CPUCTL(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_BOOTVEC(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_HWCFG(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_HWCFG1(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMACTL(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMATRFBASE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMATRFMOFFS(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMATRFCMD(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMATRFFBOFFS(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMAPOLL_FB(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMAPOLL_CP(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMCTL(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMSTAT(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_TRACEIDX(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_TRACEPC(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_ICD_CMD(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_ICD_ADDR(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_ICD_WDATA(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_ICD_RDATA(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMC(id, 0)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMC(id, 1)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMC(id, 2)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMC(id, 3)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMD(id, 0)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMD(id, 1)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMD(id, 2)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMD(id, 3)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMT(id, 0)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMT(id, 1)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMT(id, 2)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_IMEMT(id, 3)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(id, 0)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(id, 1)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(id, 2)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(id, 3)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(id, 4)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(id, 5)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(id, 6)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMC(id, 7)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(id, 0)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(id, 1)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(id, 2)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(id, 3)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(id, 4)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(id, 5)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(id, 6)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FALCON_DMEMD(id, 7)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(id, 0)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(id, 1)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(id, 2)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(id, 3)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(id, 4)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(id, 5)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(id, 6)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_TRANSCFG(id, 7)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_INSTBLK(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_CTL(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_THROTTLE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_ACHK_BLK(id, 0)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_ACHK_BLK(id, 1)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_ACHK_CTL(id, 0)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_FBIF_ACHK_CTL(id, 1)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_TOP_CTL(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_TOP_RST_INTERVAL(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_FDCT_QT_ADDR(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_FDCT_QT_WDATA(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_FDCT_QT_RDATA(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_FDCT_RTCTRL(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_VLE_HT_ADDR(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_VLE_HT_WDATA(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_VLE_HT_RDATA(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_ENC_VLE_RTCTRL(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_PDMA_PIC_INFO(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_PDMA_PIC_INFO1(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_PDMA_PIC_LUMA_STRIDE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_PDMA_PIC_CHROMA_STRIDE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_PDMA_DEC_PACKET_INFO(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_SDMA_OFFSET(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_SDMA_SIZE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_FRAME_SIZE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_FRAME_PARAM(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_BLOCK_CTL(id, 0)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_BLOCK_CTL(id, 1)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_BLOCK_CTL(id, 2)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_BLOCK_CTL(id, 3)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_HUFFMAN_MINCODE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_HUFFMAN_SYMBOL(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_OUTPUT_INFO(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_YUV2RGB_PARAM(id, 0)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_YUV2RGB_PARAM(id, 1)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_YUV2RGB_PARAM(id, 2)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_YUV2RGB_PARAM(id, 3)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_YUV2RGB_PARAM(id, 4)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_YUV2RGB_PARAM(id, 5)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 0)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 1)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 2)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 3)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 4)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 5)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 6)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 7)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 8)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 9)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 10)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 11)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 12)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 13)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 14)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 15)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 16)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 17)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 18)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 19)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 20)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 21)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 22)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 23)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 24)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 25)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 26)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 27)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 28)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 29)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 30)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 31)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 32)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 33)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 34)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 35)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 36)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 37)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 38)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 39)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 40)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 41)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 42)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 43)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 44)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 45)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 46)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 47)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 48)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 49)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 50)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 51)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 52)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 53)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 54)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 55)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 56)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 57)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 58)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 59)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 60)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 61)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 62)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_CDMA_DEC_COEF_QUANT(id, 63)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_INDEX(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_PDMA_LUMA_BASE_ADDR(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_PDMA_LUMA_BASE_ADDR_HI(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_PDMA_CHROMA_U_BASE_ADDR(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_PDMA_CHROMA_U_BASE_ADDR_HI(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_PDMA_CHROMA_V_BASE_ADDR(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_PDMA_CHROMA_V_BASE_ADDR_HI(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_SDMA_BASE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_SDMA_BASE_HI(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_CYCLE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_STATUS(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_PDMA_PIC_INFO2_POS_8X8(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_DEC_READ_BYTES_OFFSET(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CORE_HUFFMAN_TABLE_READ_ADDR(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_CFG(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_DEC_CAPABILITY(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_ENC_CAPABILITY(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_KICK_OFF(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_PIC_DONE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_ERROR(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_CYA(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_INTEN(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_INTR(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_SEPARATE_CORE_RESET(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_ENGINE_PARTIAL_CORE_RESET(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_MISC_BLCG(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_MISC_BLCG1(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_IP_VER(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CDMA_CFG(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CDMA_BASE(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CDMA_BASE_HI(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CDMA_INTEN(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CDMA_INTR(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_JPEG_CDMA_CORE_SET(id)),\
    privInfo_lwjpg_v02_00(LW_PLWJPG_PMM_MS(id)),\
    privInfo_lwjpg_v02_00(0) \
};

dbg_lwjpg_v02_00 lwjpgPrivReg_v02_02_eng0[] = LWJPG_PRIV_ACCESS(0);
dbg_lwjpg_v02_00 lwjpgPrivReg_v02_02_eng1[] = LWJPG_PRIV_ACCESS(1);
dbg_lwjpg_v02_00 lwjpgPrivReg_v02_02_eng2[] = LWJPG_PRIV_ACCESS(2);
dbg_lwjpg_v02_00 lwjpgPrivReg_v02_02_eng3[] = LWJPG_PRIV_ACCESS(3);
dbg_lwjpg_v02_00 lwjpgPrivReg_v02_02_eng4[] = LWJPG_PRIV_ACCESS(4);
dbg_lwjpg_v02_00 lwjpgPrivReg_v02_02_eng5[] = LWJPG_PRIV_ACCESS(5);
dbg_lwjpg_v02_00 lwjpgPrivReg_v02_02_eng6[] = LWJPG_PRIV_ACCESS(6);
dbg_lwjpg_v02_00 lwjpgPrivReg_v02_02_eng7[] = LWJPG_PRIV_ACCESS(7);

//-----------------------------------------------------
// lwjpgIsValidEngineId_v02_02
//-----------------------------------------------------
BOOL lwjpgIsValidEngineId_v02_02(LwU32 indexGpu, LwU32 engineId)
{
    switch (engineId)
    {
    case LWWATCH_LWJPG_0:
    case LWWATCH_LWJPG_1:
    case LWWATCH_LWJPG_2:
    case LWWATCH_LWJPG_3:
    case LWWATCH_LWJPG_4:
    case LWWATCH_LWJPG_5:
    case LWWATCH_LWJPG_6:
    case LWWATCH_LWJPG_7:
        break;
    default:
        dprintf("GH100 supports upto 8 lwjpg instances only\n");
        return FALSE;
    }

    return TRUE;
}

//-----------------------------------------------------
// lwjpgIsSupported_v02_02
//-----------------------------------------------------
BOOL lwjpgIsSupported_v02_02( LwU32 indexGpu, LwU32 engineId )
{
    if (!pLwjpg[indexGpu].lwjpgIsValidEngineId(indexGpu, engineId))
        return FALSE;

    switch (engineId)
    {
    case LWWATCH_LWJPG_0:
        pLwjpgPrivReg[LWWATCH_LWJPG_0] = lwjpgPrivReg_v02_02_eng0;
        break;
    case LWWATCH_LWJPG_1:
        pLwjpgPrivReg[LWWATCH_LWJPG_1] = lwjpgPrivReg_v02_02_eng1;
        break;
    case LWWATCH_LWJPG_2:
        pLwjpgPrivReg[LWWATCH_LWJPG_2] = lwjpgPrivReg_v02_02_eng2;
        break;
    case LWWATCH_LWJPG_3:
        pLwjpgPrivReg[LWWATCH_LWJPG_3] = lwjpgPrivReg_v02_02_eng3;
        break;
    case LWWATCH_LWJPG_4:
        pLwjpgPrivReg[LWWATCH_LWJPG_4] = lwjpgPrivReg_v02_02_eng4;
        break;
    case LWWATCH_LWJPG_5:
        pLwjpgPrivReg[LWWATCH_LWJPG_5] = lwjpgPrivReg_v02_02_eng5;
        break;
    case LWWATCH_LWJPG_6:
        pLwjpgPrivReg[LWWATCH_LWJPG_6] = lwjpgPrivReg_v02_02_eng6;
        break;
    case LWWATCH_LWJPG_7:
        pLwjpgPrivReg[LWWATCH_LWJPG_7] = lwjpgPrivReg_v02_02_eng7;
        break;
    default:
        return FALSE;
    }

    pLwjpgFuseReg     = lwjpgFuseReg_v02_02;
    pLwjpgMethodTable = lwjpgMethodTable_v02_02;

    return TRUE;
}

//-----------------------------------------------------
// lwjpgIsPrivBlocked_v02_02
//-----------------------------------------------------
BOOL lwjpgIsPrivBlocked_v02_02( LwU32 indexGpu, LwU32 engineId )
{
    LwU32 idx;
    LwU32 bitmask;
    LwU32 privWarnReadDisable  = 0;
    LwU32 privWarnWriteDisable = 0;
    LwU32 regSysPrivFsConfig;

    if (!pLwjpg[indexGpu].lwjpgIsValidEngineId(indexGpu, engineId))
        return TRUE;

    // Bit-fields within LW_PPRIV_SYS_PRIV_FS_CONFIG denote priv access for video.
    // All video engines must have priv access for lwjpg command support.

    switch (engineId)
    {
    case LWWATCH_LWJPG_0:
        idx = LW_PPRIV_SYSC_PRI_MASTER_sys_pri_hub2lwjpg_pri0 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSC_PRI_MASTER_sys_pri_hub2lwjpg_pri0 - (idx << 5));
        break;
    case LWWATCH_LWJPG_1:
        idx = LW_PPRIV_SYSC_PRI_MASTER_sys_pri_hub2lwjpg_pri1 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSC_PRI_MASTER_sys_pri_hub2lwjpg_pri1 - (idx << 5));
        break;
    case LWWATCH_LWJPG_2:
        idx = LW_PPRIV_SYSC_PRI_MASTER_sys_pri_hub2lwjpg_pri2 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSC_PRI_MASTER_sys_pri_hub2lwjpg_pri2 - (idx << 5));
        break;
    case LWWATCH_LWJPG_3:
        idx = LW_PPRIV_SYSB_PRI_MASTER_sys_pri_hub2lwjpg_pri3 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSB_PRI_MASTER_sys_pri_hub2lwjpg_pri3 - (idx << 5));
        break;
    case LWWATCH_LWJPG_4:
        idx = LW_PPRIV_SYSB_PRI_MASTER_sys_pri_hub2lwjpg_pri4 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSB_PRI_MASTER_sys_pri_hub2lwjpg_pri4 - (idx << 5));
        break;
    case LWWATCH_LWJPG_5:
        idx = LW_PPRIV_SYSB_PRI_MASTER_sys_pri_hub2lwjpg_pri5 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSB_PRI_MASTER_sys_pri_hub2lwjpg_pri5 - (idx << 5));
        break;
    case LWWATCH_LWJPG_6:
        idx = LW_PPRIV_SYSB_PRI_MASTER_sys_pri_hub2lwjpg_pri6 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSB_PRI_MASTER_sys_pri_hub2lwjpg_pri6 - (idx << 5));
        break;
    case LWWATCH_LWJPG_7:
        idx = LW_PPRIV_SYSB_PRI_MASTER_sys_pri_hub2lwjpg_pri7 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSB_PRI_MASTER_sys_pri_hub2lwjpg_pri7 - (idx << 5));
        break;
    default:
        return TRUE;
    }

    if ((regSysPrivFsConfig & bitmask) != bitmask)
    {
        return TRUE;
    }
    else
    {
        privWarnReadDisable  = (GPU_REG_RD32(LW_FUSE_OPT_LWJPG_PRIV_READ_DIS)  & 0x1);
        if (privWarnReadDisable)
        {
            dprintf("WARNING: LWJPG%d: Fixed function HW unit register's priv READ is disabled by fuse, register reads all zeros, only FALCON, FBIF, CG, PMM registers are readable depending on PLM settings\n",
                    engineId);
        }
 
        privWarnWriteDisable = (GPU_REG_RD32(LW_FUSE_OPT_LWJPG_PRIV_WRITE_DIS) & 0x1); 
        if (privWarnWriteDisable)
        {
            dprintf("WARNING: LWJPG%d: Fixed function HW unit register's priv WRITE is disabled by fuse, register writes have no effect, only FALCON, FBIF, CG, PMM registers are writeable depending on PLM settings\n",
                    engineId); 
        }
    }

    return FALSE;
}

//-----------------------------------------------------
// lwjpgPrintMethodData_v02_02
//-----------------------------------------------------
void lwjpgPrintMethodData_v02_02(LwU32 clmn, char *tag, LwU32 method, LwU32 data)
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
// lwjpgDumpImem_v02_02 - Dumps LWJPG instruction memory
//-----------------------------------------------------
LW_STATUS lwjpgDumpImem_v02_02( LwU32 indexGpu, LwU32 engineId , LwU32 imemSize)
{
    LW_STATUS status = LW_OK;
    LwU32  imemSizeMax;
    LwU32 addrssImem = LW_PLWJPG_FALCON_IMEMD(engineId, 0);
    LwU32 address2Imem = LW_PLWJPG_FALCON_IMEMC(engineId, 0);
    LwU32 address2Imemt = LW_PLWJPG_FALCON_IMEMT(engineId, 0);
    LwU32 u;
    LwU32 blk=0;

    imemSizeMax = (GPU_REG_IDX_RD_DRF(_PLWJPG_FALCON, _HWCFG, engineId, _IMEM_SIZE) << 8);
    if (imemSize > 0)
        imemSize = min(imemSizeMax, imemSize);
    else
        imemSize = imemSizeMax;

    dprintf("\n");
    dprintf("lw: -- Gpu %u LWJPG %d IMEM -- \n", indexGpu, engineId);
    dprintf("lw: -- Gpu %u LWJPG %d IMEM SIZE =  0x%08x-- \n", indexGpu, engineId, imemSize);
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
// lwjpgDumpDmem_v02_02 - Dumps LWJPG data memory
//-----------------------------------------------------
LW_STATUS lwjpgDumpDmem_v02_02( LwU32 indexGpu, LwU32 engineId , LwU32 dmemSize)
{
    LW_STATUS status = LW_OK;
    LwU32 dmemSizeMax;
    // these are the variables defined for use in parsing and printinf the methods and data
    LwU32 addrss, address2, u, i, comMthdOffs = 0, appMthdOffs = 0, classNum;
    LwU32 comMthd[CMNMETHODARRAYSIZE_LWJPG] = {0};
    LwU32 methodIdx;

    dmemSizeMax = (GPU_REG_IDX_RD_DRF(_PLWJPG_FALCON, _HWCFG, engineId, _DMEM_SIZE) << 8);

    if(dmemSize > 0)
        dmemSize = min(dmemSizeMax, dmemSize);
    else
       dmemSize = dmemSizeMax;

    addrss = LW_PLWJPG_FALCON_DMEMD(engineId, 0);
    address2 = LW_PLWJPG_FALCON_DMEMC(engineId, 0);
    classNum    = 0xB8D1;

    dprintf("\n");
    dprintf("lw: -- Gpu %u LWJPG %d DMEM -- \n", indexGpu, engineId);
    dprintf("lw: -- Gpu %u LWJPG %d DMEM SIZE =  0x%08x-- \n", indexGpu, engineId, dmemSize);

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
                    lwjpgPrintMethodData_v02_02(48,
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
// lwjpgTestState_v02_02 - Test basic lwjpg state
//-----------------------------------------------------
LW_STATUS lwjpgTestState_v02_02( LwU32 indexGpu, LwU32 engineId )
{
    LW_STATUS    status = LW_OK;
    LwU32   regIntr;
    LwU32   regIntrEn;
    LwU32   data32;

    //check falcon interrupts
    regIntr = GPU_REG_RD32(LW_PLWJPG_FALCON_IRQSTAT(engineId));
    regIntrEn = GPU_REG_RD32(LW_PLWJPG_FALCON_IRQMASK(engineId));
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
            GPU_REG_RD32(LW_PLWJPG_FALCON_GPTMRINT(engineId)));
        dprintf("lw: LW_PLWJPG_FALCON_GPTMRVAL:    0x%08x\n", 
            GPU_REG_RD32(LW_PLWJPG_FALCON_GPTMRVAL(engineId)));
        
    }
    
    if ( DRF_VAL( _PLWJPG,_FALCON_IRQSTAT, _WDTMR, regIntr))
    {
        dprintf("lw: LW_PLWJPG_FALCON_IRQSTAT_WDTMR pending\n");
    }

    if ( DRF_VAL( _PLWJPG,_FALCON_IRQSTAT, _MTHD, regIntr))
    {
        dprintf("lw: LW_PLWJPG_FALCON_IRQSTAT_MTHD pending\n");

        dprintf("lw: LW_PLWJPG_FALCON_MTHDDATA_DATA:    0x%08x\n", 
            GPU_REG_RD32(LW_PLWJPG_FALCON_MTHDDATA(engineId)));
        
        data32 = GPU_REG_RD32(LW_PLWJPG_FALCON_MTHDID(engineId));
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

        pFalcon[indexGpu].falconPrintMailbox(FALCON_LWJPG_BASE(engineId)/*0x828000*/);
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

    data32 = GPU_REG_RD32(LW_PLWJPG_FALCON_IDLESTATE(engineId));

    if ( DRF_VAL( _PLWJPG, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_PLWJPG_FALCON_IDLESTATE_FALCON_BUSY\n");
        addUnitErr("\t LW_PLWJPG_FALCON_IDLESTATE_FALCON_BUSY\n");
        status = LW_ERR_GENERIC;
    }

  
    data32 = GPU_REG_RD32(LW_PLWJPG_FALCON_FHSTATE(engineId));
 
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
    data32 = GPU_REG_RD32(LW_PLWJPG_FALCON_ENGCTL(engineId));
    
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

    data32 = GPU_REG_RD32(LW_PLWJPG_FALCON_CPUCTL(engineId));

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
    data32 = GPU_REG_RD32(LW_PLWJPG_FALCON_ITFEN(engineId));

    if (DRF_VAL( _PLWJPG, _FALCON_ITFEN, _CTXEN, data32))
    {
        dprintf("lw: + LW_PLWJPG_FALCON_ITFEN_CTXEN enabled\n");
             
        if (pFalcon[indexGpu].falconTestCtxState(FALCON_LWJPG_BASE(engineId)/*0x828000*/, "PLWJPG") == LW_ERR_GENERIC)
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
    if ( pFalcon[indexGpu].falconTestPC(FALCON_LWJPG_BASE(engineId)/*0x828000*/, "PLWJPG") == LW_ERR_GENERIC )
    {
        dprintf("lw: Falcon instruction pointer is stuck or invalid\n");
        
        //TODO: treat falcon PC errors as warnings now, need to report as error
        addUnitErr("\t Warning: Falcon instruction pointer is stuck or invalid\n");
        //status = LW_ERR_GENERIC;
    }

    return status;  
}

//-----------------------------------------------------
// lwjpgPrintPriv_v02_02
//-----------------------------------------------------
void lwjpgPrintPriv_v02_02(LwU32 clmn, char *tag, LwU32 id)
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
// lwjpgDumpPriv_v02_02 - Dumps LWJPG priv reg space
//-----------------------------------------------------
LW_STATUS lwjpgDumpPriv_v02_02(LwU32 indexGpu, LwU32 engineId)
{
    LwU32 u;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWJPG %d priv registers -- \n", indexGpu, engineId);
    dprintf("lw:\n");

    for(u=0;;u++)
    {
        if (pLwjpgPrivReg[engineId][u].m_id == 0)
        {
            break;
        }
        pLwjpg[indexGpu].lwjpgPrintPriv(61,pLwjpgPrivReg[engineId][u].m_tag,pLwjpgPrivReg[engineId][u].m_id);
    }
    return LW_OK; 
}

//-----------------------------------------------------------
// lwjpgDumpFuse_v02_02 - Dumps LWJPG related fuse registers
//-----------------------------------------------------------
LW_STATUS lwjpgDumpFuse_v02_02(LwU32 indexGpu, LwU32 engineId)
{
    LwU32 u;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWJPG %d related fuse registers -- \n", indexGpu, engineId);
    dprintf("lw:\n");

    for(u=0;;u++)
    {
        if (pLwjpgFuseReg[u].m_id == 0)
        {
            break;
        }
        pLwjpg[indexGpu].lwjpgPrintPriv(61,pLwjpgFuseReg[u].m_tag,pLwjpgFuseReg[u].m_id);
    }
    return LW_OK; 
}

//--------------------------------------------------------
// lwjpgDisplayHwcfg_v02_02 - Display LWJPG HW config
//--------------------------------------------------------
LW_STATUS lwjpgDisplayHwcfg_v02_02(LwU32 indexGpu, LwU32 engineId)
{
    LwU32 hwcfg, hwcfg1;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWJPG %d HWCFG -- \n", indexGpu, engineId);
    dprintf("lw:\n");

    hwcfg = GPU_REG_RD32(LW_PLWJPG_FALCON_HWCFG(engineId));
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

    hwcfg1 = GPU_REG_RD32(LW_PLWJPG_FALCON_HWCFG1(engineId));
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
LW_STATUS  lwjpgDisplayFlcnSPR_v02_02(LwU32 indexGpu, LwU32 engineId)
{
    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWJPG %d Special Purpose Registers -- \n", indexGpu, engineId);
    dprintf("lw:\n");

    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD(engineId), 0x1008);
    dprintf("lw: LWJPG IV0 :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA(engineId)));
    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD(engineId), 0x1108);
    dprintf("lw: LWJPG IV1 :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA(engineId)));
    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD(engineId), 0x1308);
    dprintf("lw: LWJPG EV  :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA(engineId)));
    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD(engineId), 0x1408);
    dprintf("lw: LWJPG SP  :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA(engineId)));
    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD(engineId), 0x1508);
    dprintf("lw: LWJPG PC  :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA(engineId)));
    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD(engineId), 0x1608);
    dprintf("lw: LWJPG IMB :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA(engineId)));
    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD(engineId), 0x1708);
    dprintf("lw: LWJPG DMB :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA(engineId)));
    GPU_REG_WR32(LW_PLWJPG_FALCON_ICD_CMD(engineId), 0x1808);
    dprintf("lw: LWJPG CSW :    0x%08x\n", GPU_REG_RD32(LW_PLWJPG_FALCON_ICD_RDATA(engineId)));
    dprintf("lw:\n\n");

    return LW_OK; 
}

//-----------------------------------------------------
// lwjpgGetClassId_v02_02
//-----------------------------------------------------
LwU32
lwjpgGetClassId_v02_02(void)
{
    return 0x0000B8D1;/*LWB8D1_VIDEO_LWJPG*/
}

