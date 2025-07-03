/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/
#include "ofa.h"
#include "lwsym.h"
#include "g_ofa_private.h"
#include "ampere/ga100/dev_ofa_pri.h"
#include "g_ofa_private.h"     // (rmconfig)  implementation prototypes

#include <class/clc6fa.h>

#define GetAppMthdParam(mthd)   ((mthd - APPMETHODBASE) >> 2U)

dbg_ofa_v01_00 ofaMethodTable_v01_00[] =
{
    privInfo_ofa_v01_00(LWC6FA_NOP),
    privInfo_ofa_v01_00(LWC6FA_PM_TRIGGER),
    privInfo_ofa_v01_00(LWC6FA_SET_APPLICATION_ID),
    privInfo_ofa_v01_00(LWC6FA_SET_WATCHDOG_TIMER),
    privInfo_ofa_v01_00(LWC6FA_SEMAPHORE_A),
    privInfo_ofa_v01_00(LWC6FA_SEMAPHORE_B),
    privInfo_ofa_v01_00(LWC6FA_SEMAPHORE_C),
    privInfo_ofa_v01_00(LWC6FA_CTX_SAVE_AREA),
    privInfo_ofa_v01_00(LWC6FA_CTX_SWITCH),
    privInfo_ofa_v01_00(LWC6FA_EXELWTE),
    privInfo_ofa_v01_00(LWC6FA_SEMAPHORE_D),
    privInfo_ofa_v01_00(LWC6FA_SET_AUXILIARY_DATA_BUFFER),
    privInfo_ofa_v01_00(LWC6FA_SET_PICTURE_INDEX),
    privInfo_ofa_v01_00(LWC6FA_SET_CONTROL_PARAMS),
    privInfo_ofa_v01_00(LWC6FA_SET_TOTAL_LEVEL_NUM),
    privInfo_ofa_v01_00(LWC6FA_PM_TRIGGER_END),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_LEVEL_INDEX(0)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_LEVEL_INDEX(1)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_LEVEL_INDEX(2)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_LEVEL_INDEX(3)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_LEVEL_INDEX(4)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_DRV_SET_UP_ADDR(0)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_DRV_SET_UP_ADDR(1)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_DRV_SET_UP_ADDR(2)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_DRV_SET_UP_ADDR(3)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_DRV_SET_UP_ADDR(4)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_LWRR_PIC_ADDR(0)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_LWRR_PIC_ADDR(1)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_LWRR_PIC_ADDR(2)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_LWRR_PIC_ADDR(3)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_LWRR_PIC_ADDR(4)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_REF_PIC_ADDR(0)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_REF_PIC_ADDR(1)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_REF_PIC_ADDR(2)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_REF_PIC_ADDR(3)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_REF_PIC_ADDR(4)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_HINT_MV_ADDR(0)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_HINT_MV_ADDR(1)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_HINT_MV_ADDR(2)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_HINT_MV_ADDR(3)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_HINT_MV_ADDR(4)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_WINNER_COST_ADDR(0)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_WINNER_COST_ADDR(1)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_WINNER_COST_ADDR(2)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_WINNER_COST_ADDR(3)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_WINNER_COST_ADDR(4)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_WINNER_FLOW_ADDR(0)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_WINNER_FLOW_ADDR(1)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_WINNER_FLOW_ADDR(2)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_WINNER_FLOW_ADDR(3)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_WINNER_FLOW_ADDR(4)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_HISTORY_BUF_ADDR(0)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_HISTORY_BUF_ADDR(1)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_HISTORY_BUF_ADDR(2)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_HISTORY_BUF_ADDR(3)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_HISTORY_BUF_ADDR(4)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_TEMPORARY_BUF_ADDR(0)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_TEMPORARY_BUF_ADDR(1)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_TEMPORARY_BUF_ADDR(2)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_TEMPORARY_BUF_ADDR(3)),
    privInfo_ofa_v01_00(LWC6FA_SET_MULTI_LEVELS_SET_TEMPORARY_BUF_ADDR(4)),
    privInfo_ofa_v01_00(0),
};

dbg_ofa_v01_00 ofaPrivReg_v01_00[] =
{
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQSSET),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQSCLR),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQSTAT),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQMODE),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQMSET),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQMCLR),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQMASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQDEST),
    privInfo_ofa_v01_00(LW_POFA_FALCON_GPTMRINT),
    privInfo_ofa_v01_00(LW_POFA_FALCON_GPTMRVAL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_GPTMRCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PTIMER0),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PTIMER1),
    privInfo_ofa_v01_00(LW_POFA_FALCON_WDTMRVAL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_WDTMRCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQDEST2),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MAILBOX0),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MAILBOX1),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ITFEN),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IDLESTATE),
    privInfo_ofa_v01_00(LW_POFA_FALCON_LWRCTX),
    privInfo_ofa_v01_00(LW_POFA_FALCON_NXTCTX),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CTXACK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_FHSTATE),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PRIVSTATE),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDDATA),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDID),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDWDAT),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDCOUNT),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDPOP),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDRAMSZ),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SFTRESET),
    privInfo_ofa_v01_00(LW_POFA_FALCON_OS),
    privInfo_ofa_v01_00(LW_POFA_FALCON_RM),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SOFT_PM),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SOFT_MODE),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DEBUG1),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DEBUGINFO),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IBRKPT1),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IBRKPT2),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CGCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ENGCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PMM),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ADDR),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IBRKPT3),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IBRKPT4),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IBRKPT5),
    privInfo_ofa_v01_00(LW_POFA_FALCON_EXCI2),
    privInfo_ofa_v01_00(LW_POFA_FALCON_EXCI),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SVEC_SPR),
    privInfo_ofa_v01_00(LW_POFA_FALCON_RSTAT0),
    privInfo_ofa_v01_00(LW_POFA_FALCON_RSTAT3),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQSCMASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_HSCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_HSCTL_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_HWCFG2),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CPUCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_BOOTVEC),
    privInfo_ofa_v01_00(LW_POFA_FALCON_HWCFG),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMACTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMATRFBASE),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMATRFMOFFS),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMATRFCMD),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMATRFFBOFFS),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAPOLL_FB),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAPOLL_CP),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMATRFBASE1),
    privInfo_ofa_v01_00(LW_POFA_FALCON_HWCFG1),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CPUCTL_ALIAS),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CG2),
    privInfo_ofa_v01_00(LW_POFA_FALCON_STACKCFG),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMSTAT),
    privInfo_ofa_v01_00(LW_POFA_FALCON_TRACEIDX),
    privInfo_ofa_v01_00(LW_POFA_FALCON_TRACEPC),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMCTL_DEBUG),
    privInfo_ofa_v01_00(LW_POFA_FALCON_TRACEINFO),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEMC(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEMD(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEMT(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEMC(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEMD(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEMT(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEMC(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEMD(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEMT(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEMC(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEMD(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEMT(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMC(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMD(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMC(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMD(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMC(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMD(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMC(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMD(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMC(4)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMD(4)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMC(5)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMD(5)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMC(6)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMD(6)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMC(7)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMD(7)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ICD_CMD),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ICD_ADDR),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ICD_WDATA),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ICD_RDATA),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMT(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMT(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMT(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMT(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMT(4)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMT(5)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMT(6)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMT(7)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CSBERRSTAT),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CSBERR_INFO),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CSBERR_ADDR),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SCTL1),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DBGCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEM_DUMMY),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMSTAT),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMVACTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SP_MIN),
    privInfo_ofa_v01_00(LW_POFA_FALCON_HWCFG3),
    privInfo_ofa_v01_00(LW_POFA_FALCON_WDTMR_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEML(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEML(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEML(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEML(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEML(4)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEML(5)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEML(6)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEML(7)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAINFO_FINISHED_FBRD_LOW),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAINFO_FINISHED_FBRD_HIGH),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAINFO_FINISHED_FBWR_LOW),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAINFO_FINISHED_FBWR_HIGH),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAINFO_LWRRENT_FBRD_LOW),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAINFO_LWRRENT_FBRD_HIGH),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAINFO_LWRRENT_FBWR_LOW),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAINFO_LWRRENT_FBWR_HIGH),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAINFO_CTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAINFO_ERR_TAG_HIGH),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAINFO_ERR_TAG_LOW),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_0(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_0(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_0(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_0(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_1(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_1(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_1(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_1(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_2(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_2(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_2(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_2(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_3(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_3(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_3(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_3(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_LWRCTX2),
    privInfo_ofa_v01_00(LW_POFA_FALCON_NXTCTX2),
    privInfo_ofa_v01_00(LW_POFA_FALCON_INTR_CTRL_PRIV_LEVEL_MASK(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_INTR_CTRL_PRIV_LEVEL_MASK(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_INTR_CTRL(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_INTR_CTRL(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_INTR_RETRIGGER(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_INTR_RETRIGGER(1)),
    privInfo_ofa_v01_00(LW_POFA_CTRL),
    privInfo_ofa_v01_00(LW_POFA_COMMAND),
    privInfo_ofa_v01_00(LW_POFA_FRAME_SIZE),
    privInfo_ofa_v01_00(LW_POFA_SUBFRAME),
    privInfo_ofa_v01_00(LW_POFA_ROI_START),
    privInfo_ofa_v01_00(LW_POFA_ROI_END),
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_INTF_PARTA),
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_INTF_PARTB),
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_PARTA_GOLDEN_0),
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_PARTA_GOLDEN_1),
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_PARTA_GOLDEN_2),
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_PARTA_GOLDEN_3),
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_PARTB_GOLDEN_0),
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_PARTB_GOLDEN_1),
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_PARTB_GOLDEN_2),
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_PARTB_GOLDEN_3),
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_COMP_PARTA),
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_COMP_PARTB),
    privInfo_ofa_v01_00(LW_POFA_PMM),
    privInfo_ofa_v01_00(LW_POFA_PMM1),
    privInfo_ofa_v01_00(LW_POFA_PMM2),
    privInfo_ofa_v01_00(LW_POFA_EPI_CTRL),
    privInfo_ofa_v01_00(LW_POFA_EPIPOLE_X),
    privInfo_ofa_v01_00(LW_POFA_EPIPOLE_Y),
    privInfo_ofa_v01_00(LW_POFA_EPI_FUNDAMENTAL_M(0)),
    privInfo_ofa_v01_00(LW_POFA_EPI_FUNDAMENTAL_M(1)),
    privInfo_ofa_v01_00(LW_POFA_EPI_FUNDAMENTAL_M(2)),
    privInfo_ofa_v01_00(LW_POFA_EPI_FUNDAMENTAL_M(3)),
    privInfo_ofa_v01_00(LW_POFA_EPI_FUNDAMENTAL_M(4)),
    privInfo_ofa_v01_00(LW_POFA_EPI_FUNDAMENTAL_M(5)),
    privInfo_ofa_v01_00(LW_POFA_EPI_FUNDAMENTAL_M(6)),
    privInfo_ofa_v01_00(LW_POFA_EPI_FUNDAMENTAL_M(7)),
    privInfo_ofa_v01_00(LW_POFA_EPI_FUNDAMENTAL_M(8)),
    privInfo_ofa_v01_00(LW_POFA_EPI_ROTATION_M(0)),
    privInfo_ofa_v01_00(LW_POFA_EPI_ROTATION_M(1)),
    privInfo_ofa_v01_00(LW_POFA_EPI_ROTATION_M(2)),
    privInfo_ofa_v01_00(LW_POFA_EPI_ROTATION_M(3)),
    privInfo_ofa_v01_00(LW_POFA_EPI_ROTATION_M(4)),
    privInfo_ofa_v01_00(LW_POFA_EPI_ROTATION_M(5)),
    privInfo_ofa_v01_00(LW_POFA_EPI_ROTATION_M(6)),
    privInfo_ofa_v01_00(LW_POFA_EPI_ROTATION_M(7)),
    privInfo_ofa_v01_00(LW_POFA_EPI_ROTATION_M(8)),
    privInfo_ofa_v01_00(LW_POFA_CVC_CTRL),
    privInfo_ofa_v01_00(LW_POFA_SGM_CTRL),
    privInfo_ofa_v01_00(LW_POFA_SGM_PENALTY),
    privInfo_ofa_v01_00(LW_POFA_LWRR_SURFACE_OFFSET),
    privInfo_ofa_v01_00(LW_POFA_LWRR_SURFACE_STRIDE),
    privInfo_ofa_v01_00(LW_POFA_LWRR_SURFACE_FORMAT),
    privInfo_ofa_v01_00(LW_POFA_REF_SURFACE_OFFSET),
    privInfo_ofa_v01_00(LW_POFA_REF_SURFACE_STRIDE),
    privInfo_ofa_v01_00(LW_POFA_REF_SURFACE_FORMAT),
    privInfo_ofa_v01_00(LW_POFA_FLOW_SURFACE_OFFSET),
    privInfo_ofa_v01_00(LW_POFA_FLOW_SURFACE_STRIDE),
    privInfo_ofa_v01_00(LW_POFA_FLOW_SURFACE_FORMAT),
    privInfo_ofa_v01_00(LW_POFA_COST_SURFACE_OFFSET),
    privInfo_ofa_v01_00(LW_POFA_COST_SURFACE_STRIDE),
    privInfo_ofa_v01_00(LW_POFA_COST_SURFACE_FORMAT),
    privInfo_ofa_v01_00(LW_POFA_SGM_HIST_SURFACE_OFFSET),
    privInfo_ofa_v01_00(LW_POFA_SGM_TEMP_SURFACE_OFFSET),
    privInfo_ofa_v01_00(LW_POFA_BOOT_42),
    privInfo_ofa_v01_00(LW_POFA_OPT_FUSE_UCODE_OFA_REV),
    privInfo_ofa_v01_00(LW_POFA_PYD_CTRL),
    privInfo_ofa_v01_00(LW_POFA_PYD_HINT_SURFACE_OFFSET),
    privInfo_ofa_v01_00(LW_POFA_PYD_HINT_SURFACE_STRIDE),
    privInfo_ofa_v01_00(LW_POFA_PYD_HINT_SURFACE_FORMAT),
    privInfo_ofa_v01_00(LW_POFA_OFA_ENGINE_INTEN),
    privInfo_ofa_v01_00(LW_POFA_OFA_ENGINE_INTR),
    privInfo_ofa_v01_00(LW_POFA_OFA_SLCG_DIS),
    privInfo_ofa_v01_00(LW_POFA_MISC_BLCG),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(0)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(1)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(2)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(3)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(4)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(5)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(6)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(7)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_INSTBLK),
    privInfo_ofa_v01_00(LW_POFA_FBIF_CTL),
    privInfo_ofa_v01_00(LW_POFA_FBIF_THROTTLE),
    privInfo_ofa_v01_00(LW_POFA_FBIF_ACHK_BLK(0)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_ACHK_CTL(0)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_ACHK_BLK(1)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_ACHK_CTL(1)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_DBG_STAT0),
    privInfo_ofa_v01_00(LW_POFA_FBIF_DBG_STAT1),
    privInfo_ofa_v01_00(LW_POFA_FBIF_DBG_STAT2),
    privInfo_ofa_v01_00(LW_POFA_FBIF_BW_ALLOC),
    privInfo_ofa_v01_00(LW_POFA_FBIF_DBG_RDCOUNT_LO),
    privInfo_ofa_v01_00(LW_POFA_FBIF_DBG_RDCOUNT_HI),
    privInfo_ofa_v01_00(LW_POFA_FBIF_DBG_WRCOUNT_LO),
    privInfo_ofa_v01_00(LW_POFA_FBIF_DBG_WRCOUNT_HI),
    privInfo_ofa_v01_00(LW_POFA_FBIF_BIND_STATUS),
    privInfo_ofa_v01_00(LW_POFA_FBIF_REGIONCFG),
    privInfo_ofa_v01_00(LW_POFA_FBIF_REGIONCFG_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FBIF_CG1),
    privInfo_ofa_v01_00(LW_POFA_MISC_BLCG1),
    privInfo_ofa_v01_00(LW_POFA_FALCON_BOOTVEC__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEM_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEM_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CPUCTL__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CPUCTL_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMCTL__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SFTRESET__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DBGCTL__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ADDR__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMCTL__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_EXE_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_STACKCFG__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMACTL__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMSTAT__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAINFO_CTL__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMVACTL__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMSTAT__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SCTL1__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQDEST__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_GPTMRCTL__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQSCMASK__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_GPTMRVAL__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQMCLR__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQTMR_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_GPTMRINT__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQMODE__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQDEST2__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQMSET__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDWDAT__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDCOUNT__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_NXTCTX2__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_NXTCTX__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDCTX_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DEBUG1__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDDATA__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ITFEN__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDID__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CTXACK__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDPOP__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_LWRCTX2__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDRAMSZ__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_LWRCTX__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SCTL__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SCTL_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_WDTMRCTL__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_WDTMRVAL__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_0_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_0__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_1_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_1__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_2_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_2__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_3_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_3__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEM_DUMMY__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_HSCTL__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FBIF_REGIONCFG__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PRIVSTATE__PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_CONTROL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PRGLWER),
    privInfo_ofa_v01_00(LW_POFA_DBG_CPF2RPF_GRID),
    privInfo_ofa_v01_00(LW_POFA_DBG_SGM_POS),
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_INTF_PARTA),
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_INTF_PARTB),
    privInfo_ofa_v01_00(LW_POFA_DBG_OFA_CNT_CTRL),
    privInfo_ofa_v01_00(LW_POFA_DBG_ENGINE_HW),
    privInfo_ofa_v01_00(LW_POFA_DBG_SGM2DMA_WIN),
    privInfo_ofa_v01_00(LW_POFA_DBG_SGM2DMA_TEMP),
    privInfo_ofa_v01_00(LW_POFA_DBG_CVC2SGM_COST),
    privInfo_ofa_v01_00(LW_POFA_DBG_CPF2CVC_PIX),
    privInfo_ofa_v01_00(LW_POFA_DBG_RPC_MISS),
    privInfo_ofa_v01_00(LW_POFA_DBG_PARTA),
    privInfo_ofa_v01_00(LW_POFA_DBG_RPC2CVC_PIX),
    privInfo_ofa_v01_00(LW_POFA_DBG_PARTB),
    privInfo_ofa_v01_00(0),
};

static FLCN_ENGINE_IFACES flcnEngineIfaces_ofa =
{
    ofaGetFalconCoreIFace_STUB,         // flcnEngGetCoreIFace
    ofaGetFalconBase_STUB,              // flcnEngGetFalconBase
    ofaGetEngineName_v01_00,            // flcnEngGetEngineName
    ofaUcodeName_STUB,                  // flcnEngUcodeName
    ofaGetSymFilePath,                  // flcnEngGetSymFilePath
    ofaQueueGetNum_STUB,                // flcnEngQueueGetNum
    ofaQueueRead_STUB,                  // flcnEngQueueRead
    ofaGetDmemAccessPort,               // flcnEngGetDmemAccessPort
    ofaIsDmemRangeAccessible_STUB,      // flcnEngIsDmemRangeAccessible
    ofaEmemGetSize_STUB,                // flcnEngEmemGetSize
    ofaEmemGetOffsetInDmemVaSpace_STUB, // flcnEngEmemGetOffsetInDmemVaSpace
    ofaEmemGetNumPorts_STUB,            // flcnEngEmemGetNumPorts
    ofaEmemRead_STUB,                   // flcnEngEmemRead
    ofaEmemWrite_STUB,                  // flcnEngEmemWrite
};

const char* ofaGetEngineName_v01_00(void)
{
    return "OFA";
}

/*!
 * @return The falcon base address of OFA
 */
LwU32
ofaGetFalconBase_v01_00()
{
    return DEVICE_BASE(LW_POFA);
}

const char* ofaGetSymFilePath(void)
{
    return DIR_SLASH "ofa" DIR_SLASH "bin";
}

LW_STATUS ofaFillSymPath_v01_00(OBJFLCN *ofaFlcn)
{
    sprintf(ofaFlcn[indexGpu].symPath, "%s%s", LWSYM_VIRUTAL_PATH, "ofa/");
    ofaFlcn[indexGpu].bSympathSet = TRUE;
    return LW_OK;
}

LwBool ofaIsSupported_v01_00(void)
{
    return TRUE;
}

const FLCN_ENGINE_IFACES *ofaGetFalconEngineIFace_v01_00(void)
{
    const FLCN_CORE_IFACES   *pFCIF = pOfa[indexGpu].ofaGetFalconCoreIFace();
    FLCN_ENGINE_IFACES *pFEIF = NULL;

    // The Falcon Engine interface is supported only when the Core Interface exists.
    if (pFCIF)
    {
        pFEIF = &flcnEngineIfaces_ofa;

        pFEIF->flcnEngGetCoreIFace = pOfa[indexGpu].ofaGetFalconCoreIFace;
        pFEIF->flcnEngGetFalconBase = pOfa[indexGpu].ofaGetFalconBase;
        pFEIF->flcnEngQueueGetNum = pOfa[indexGpu].ofaQueueGetNum;
        pFEIF->flcnEngQueueRead = pOfa[indexGpu].ofaQueueRead;
        pFEIF->flcnEngIsDmemRangeAccessible = pOfa[indexGpu].ofaIsDmemRangeAccessible;
        pFEIF->flcnEngEmemGetSize = pOfa[indexGpu].ofaEmemGetSize;
        pFEIF->flcnEngEmemGetOffsetInDmemVaSpace = pOfa[indexGpu].ofaEmemGetOffsetInDmemVaSpace;
        pFEIF->flcnEngEmemGetNumPorts = pOfa[indexGpu].ofaEmemGetNumPorts;
        pFEIF->flcnEngEmemRead = pOfa[indexGpu].ofaEmemRead;
        pFEIF->flcnEngEmemWrite = pOfa[indexGpu].ofaEmemWrite;
    }
    return pFEIF;
}

/*!
 * @return Falcon core interface
 */
const FLCN_CORE_IFACES *
ofaGetFalconCoreIFace_v01_00()
{
    return &flcnCoreIfaces_v06_00;
}

//-----------------------------------------------------
// ofaIsGpuSupported_v01_00
//-----------------------------------------------------
BOOL ofaIsGpuSupported_v01_00(LwU32 indexGpu)
{
    if (ofaId != LWWATCH_OFA_0)
    {
        dprintf("Only OFA0 supported on this GPU\n");
        return FALSE;
    }
    dprintf("OFA0 supported on this GPU\n");
    pOfaPrivReg[0]  = ofaPrivReg_v01_00;
    pOfaMethodTable = ofaMethodTable_v01_00;
    return TRUE;
}

//-----------------------------------------------------
// ofaGetClassId_v01_00
//-----------------------------------------------------
LwU32
ofaGetClassId_v01_00(void)
{
    return LWC6FA_VIDEO_OFA;
}

//-----------------------------------------------------
// ofaDumpImem_v01_00 - Dumps OFA instruction memory
//-----------------------------------------------------
LW_STATUS ofaDumpImem_v01_00(LwU32 indexGpu, LwU32 imemSize)
{
    LW_STATUS status = LW_OK;
    LwU32  imemSizeMax;
    LwU32 addressImem = LW_POFA_FALCON_IMEMD(0);
    LwU32 address2Imem = LW_POFA_FALCON_IMEMC(0);
    LwU32 address2Imemt = LW_POFA_FALCON_IMEMT(0);
    LwU32 u;
    LwU32 blk = 0;
    LwU32 classNum;
    imemSizeMax = (GPU_REG_RD_DRF(_POFA_FALCON, _HWCFG, _IMEM_SIZE) << 8);
    if (imemSize > 0)
        imemSize = min(imemSizeMax, imemSize);
    else
        imemSize = imemSizeMax;
    classNum = pOfa[indexGpu].ofaGetClassId();

    dprintf("\n");
    dprintf("lw: -- Gpu %u OFA %X IMEM -- \n", indexGpu, classNum);
    dprintf("lw: -- Gpu %u OFA %X IMEM SIZE =  0x%08x-- \n", indexGpu, classNum, imemSize);
    dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");
    for (u = 0; u<(imemSize + 3) / 4; u++)
    {
        LwU32 i;
        if ((u % 64) == 0) {
            GPU_REG_WR32(address2Imemt, blk++);
        }
        i = (u << (0 ? LW_POFA_FALCON_IMEMC_OFFS));
        GPU_REG_WR32(address2Imem, i);
        if ((u % 8 == 0))
        {
            dprintf("\n%04X: ", 4 * u);
        }
        dprintf("%08X ", GPU_REG_RD32(addressImem));
    }
    return status;
}

//-----------------------------------------------------
// ofaTestState_v01_00 - Test basic ofa state
//-----------------------------------------------------
LW_STATUS ofaTestState_v01_00(LwU32 indexGpu)
{
    LW_STATUS    status = LW_OK;
    LwU32   regIntr;
    LwU32   regIntrEn;
    LwU32   data32;

    //check falcon interrupts
    regIntr = GPU_REG_RD32(LW_POFA_FALCON_IRQSTAT);
    regIntrEn = GPU_REG_RD32(LW_POFA_FALCON_IRQMASK);
    regIntr &= regIntrEn;

    if (!DRF_VAL(_POFA, _FALCON_IRQMASK, _GPTMR, regIntrEn))
        dprintf("lw: LW_POFA_FALCON_IRQMASK_GPTMR disabled\n");

    if (!DRF_VAL(_POFA, _FALCON_IRQMASK, _WDTMR, regIntrEn))
        dprintf("lw: LW_POFA_FALCON_IRQMASK_WDTMR disabled\n");

    if (!DRF_VAL(_POFA, _FALCON_IRQMASK, _MTHD, regIntrEn))
        dprintf("lw: LW_POFA_FALCON_IRQMASK_MTHD disabled\n");

    if (!DRF_VAL(_POFA, _FALCON_IRQMASK, _CTXSW, regIntrEn))
        dprintf("lw: LW_POFA_FALCON_IRQMASK_CTXSW disabled\n");

    if (!DRF_VAL(_POFA, _FALCON_IRQMASK, _HALT, regIntrEn))
        dprintf("lw: LW_POFA_FALCON_IRQMASK_HALT disabled\n");

    if (!DRF_VAL(_POFA, _FALCON_IRQMASK, _EXTERR, regIntrEn))
        dprintf("lw: LW_POFA_FALCON_IRQMASK_EXTERR disabled\n");

    if (!DRF_VAL(_POFA, _FALCON_IRQMASK, _SWGEN0, regIntrEn))
        dprintf("lw: LW_POFA_FALCON_IRQMASK_SWGEN0 disabled\n");

    if (!DRF_VAL(_POFA, _FALCON_IRQMASK, _SWGEN1, regIntrEn))
        dprintf("lw: LW_POFA_FALCON_IRQMASK_SWGEN1 disabled\n");


    //if any interrupt pending, set error
    if (regIntr != 0)
    {
        addUnitErr("\t OFA interrupts are pending\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_POFA, _FALCON_IRQSTAT, _GPTMR, regIntr))
    {
        dprintf("lw: LW_POFA_FALCON_IRQSTAT_GPTMR pending\n");

        dprintf("lw: LW_POFA_FALCON_GPTMRINT:    0x%08x\n",
            GPU_REG_RD32(LW_POFA_FALCON_GPTMRINT));
        dprintf("lw: LW_POFA_FALCON_GPTMRVAL:    0x%08x\n",
            GPU_REG_RD32(LW_POFA_FALCON_GPTMRVAL));

    }

    if (DRF_VAL(_POFA, _FALCON_IRQSTAT, _WDTMR, regIntr))
    {
        dprintf("lw: LW_POFA_FALCON_IRQSTAT_WDTMR pending\n");
    }

    if (DRF_VAL(_POFA, _FALCON_IRQSTAT, _MTHD, regIntr))
    {
        dprintf("lw: LW_POFA_FALCON_IRQSTAT_MTHD pending\n");

        dprintf("lw: LW_POFA_FALCON_MTHDDATA_DATA:    0x%08x\n",
            GPU_REG_RD32(LW_POFA_FALCON_MTHDDATA));

        data32 = GPU_REG_RD32(LW_POFA_FALCON_MTHDID);
        dprintf("lw: LW_POFA_FALCON_MTHDID_ID:    0x%08x\n",
            DRF_VAL(_POFA, _FALCON_MTHDID, _ID, data32));
        dprintf("lw: LW_POFA_FALCON_MTHDID_SUBCH:    0x%08x\n",
            DRF_VAL(_POFA, _FALCON_MTHDID, _SUBCH, data32));
        dprintf("lw: LW_POFA_FALCON_MTHDID_PRIV:    0x%08x\n",
            DRF_VAL(_POFA, _FALCON_MTHDID, _PRIV, data32));
    }

    if (DRF_VAL(_POFA, _FALCON_IRQSTAT, _CTXSW, regIntr))
    {
        dprintf("lw: LW_POFA_FALCON_IRQSTAT_CTXSW pending\n");
    }

    if (DRF_VAL(_POFA, _FALCON_IRQSTAT, _HALT, regIntr))
    {
        dprintf("lw: LW_POFA_FALCON_IRQSTAT_HALT pending\n");
    }

    if (DRF_VAL(_POFA, _FALCON_IRQSTAT, _EXTERR, regIntr))
    {
        dprintf("lw: LW_POFA_FALCON_IRQSTAT_EXTERR pending\n");
    }

    if (DRF_VAL(_POFA, _FALCON_IRQSTAT, _SWGEN0, regIntr))
    {
        dprintf("lw: LW_POFA_FALCON_IRQSTAT_SWGEN0 pending\n");
    }

    if (DRF_VAL(_POFA, _FALCON_IRQSTAT, _SWGEN1, regIntr))
    {
        dprintf("lw: LW_POFA_FALCON_IRQSTAT_SWGEN1 pending\n");
    }

    //
    //print falcon states
    //Bit |  Signal meaning
    //0      FALCON busy
    //

    data32 = GPU_REG_RD32(LW_POFA_FALCON_IDLESTATE);

    if (DRF_VAL(_POFA, _FALCON_IDLESTATE, _FALCON_BUSY, data32))
    {
        dprintf("lw: + LW_POFA_FALCON_IDLESTATE_FALCON_BUSY\n");
        addUnitErr("\t LW_POFA_FALCON_IDLESTATE_FALCON_BUSY\n");
        status = LW_ERR_GENERIC;
    }


    data32 = GPU_REG_RD32(LW_POFA_FALCON_FHSTATE);

    if (DRF_VAL(_POFA, _FALCON_FHSTATE, _FALCON_HALTED, data32))
    {
        dprintf("lw: + LW_POFA_FALCON_FHSTATE_FALCON_HALTED\n");
        addUnitErr("\t LW_POFA_FALCON_FHSTATE_FALCON_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_POFA, _FALCON_FHSTATE, _ENGINE_FAULTED, data32))
    {
        dprintf("lw: + LW_POFA_FALCON_FHSTATE_ENGINE_FAULTED\n");
        addUnitErr("\t LW_POFA_FALCON_FHSTATE_ENGINE_FAULTED\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_POFA, _FALCON_FHSTATE, _STALL_REQ, data32))
    {
        dprintf("lw: + LW_POFA_FALCON_FHSTATE_STALL_REQ\n");
        addUnitErr("\t LW_POFA_FALCON_FHSTATE_STALL_REQ\n");
        status = LW_ERR_GENERIC;
    }

    //print falcon ctl regs
    data32 = GPU_REG_RD32(LW_POFA_FALCON_ENGCTL);

    if (DRF_VAL(_POFA, _FALCON_ENGCTL, _ILW_CONTEXT, data32))
    {
        dprintf("lw: + LW_POFA_FALCON_ENGCTL_ILW_CONTEXT\n");
        addUnitErr("\t LW_POFA_FALCON_ENGCTL_ILW_CONTEXT\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_POFA, _FALCON_ENGCTL, _STALLREQ, data32))
    {
        dprintf("lw: + LW_POFA_FALCON_ENGCTL_STALLREQ\n");
        addUnitErr("\t LW_POFA_FALCON_ENGCTL_STALLREQ\n");
        status = LW_ERR_GENERIC;
    }

    data32 = GPU_REG_RD32(LW_POFA_FALCON_CPUCTL);

    if (DRF_VAL(_POFA, _FALCON_CPUCTL, _IILWAL, data32))
    {
        dprintf("lw: + LW_POFA_FALCON_CPUCTL_IILWAL\n");
        addUnitErr("\t LW_POFA_FALCON_CPUCTL_IILWAL\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_POFA, _FALCON_CPUCTL, _HALTED, data32))
    {
        dprintf("lw: + LW_POFA_FALCON_CPUCTL_HALTED\n");
        addUnitErr("\t LW_POFA_FALCON_CPUCTL_HALTED\n");
        status = LW_ERR_GENERIC;
    }

    if (DRF_VAL(_POFA, _FALCON_CPUCTL, _STOPPED, data32))
    {
        dprintf("lw: + LW_POFA_FALCON_CPUCTL_STOPPED\n");
        addUnitErr("\t Warning: LW_POFA_FALCON_CPUCTL_STOPPED\n");
        //status = LW_ERR_GENERIC;
    }

    // state of mthd/ctx interface 
    data32 = GPU_REG_RD32(LW_POFA_FALCON_ITFEN);

    if (DRF_VAL(_POFA, _FALCON_ITFEN, _CTXEN, data32))
    {
        dprintf("lw: + LW_POFA_FALCON_ITFEN_CTXEN enabled\n");
    }
    else
    {
        dprintf("lw: + LW_POFA_FALCON_ITFEN_CTXEN disabled\n");
    }

    if (DRF_VAL(_POFA, _FALCON_ITFEN, _MTHDEN, data32))
    {
        dprintf("lw: + LW_POFA_FALCON_ITFEN_MTHDEN enabled\n");
    }
    else
    {
        dprintf("lw: + LW_POFA_FALCON_ITFEN_MTHDEN disabled\n");
    }

    return status;
}

//-----------------------------------------------------
// ofaPrintMethodData
//-----------------------------------------------------
void ofaPrintMethodData(LwU32 clmn, char *tag, LwU32 method, LwU32 data)
{
    size_t len = strlen(tag);

    dprintf("lw: %s", tag);

    if ((len>0) && (len<(clmn + 4)))
    {
        LwU32 i;
        for (i = 0; i<clmn - len; i++)
        {
            dprintf(" ");
        }
    }
    dprintf("(0x%08X)  = 0x%08X\n", method, data);
}

LW_STATUS ofaGetOff2MthdOffs(LwU32 classNum, LwU32 *off2MthdOffs)
{
    char fName[MAX_STR_LEN];
    char lwrLine[MAX_STR_LEN];
    char varName[MAX_STR_LEN];
    char *ptr = NULL;
    FILE *pFile = NULL;
    LwU32 verNum;
    LwBool found = LW_FALSE;
    LwU32 offset;

    if ((ptr = getelw("LWW_UCODE_MAP")) && (ptr != NULL))
    {
        switch (classNum)
        {
            case 0xc6fa: verNum = 100; break;
            case 0xc7fa: verNum = 101; break;
            case 0xb8fa: verNum = 103; break;
            case 0xc9fa: verNum = 105; break;
            default: return LW_ERR_GENERIC;
        }

        if (ptr[strlen(ptr)-1] == DIR_SLASH_CHAR)
        {
            sprintf(fName, "%s" UCODE_MAP_FNAME_FORMAT, ptr, classNum, verNum);
        }
        else
        {
            sprintf(fName, "%s" DIR_SLASH UCODE_MAP_FNAME_FORMAT, ptr, classNum, verNum);
        }
        dprintf("Parsing map file \"%s\" to read \"%s\" offset...\n", fName, VAR_NAME_OFF2MTHDOFFS);

        pFile = fopen(fName, "r");
        if (pFile != NULL)
        {
            while (fgets(lwrLine, MAX_STR_LEN, pFile) != NULL)
            {
                if (strstr(lwrLine, VAR_NAME_OFF2MTHDOFFS) && sscanf(lwrLine, "1%x D _%s", &offset, varName))
                {
                    found = LW_TRUE;
                    break;
                }
            }
            fclose(pFile);
            if (found == LW_FALSE)
            {
                dprintf("No variable with name \"%s\" found in the mapfile\n", VAR_NAME_OFF2MTHDOFFS);
                return LW_ERR_GENERIC;
            }
            dprintf("\"%s\" offset in DMEM = 0x%x\n", VAR_NAME_OFF2MTHDOFFS, offset);
            *off2MthdOffs = offset;
            return LW_OK;
        }
        else
        {
            dprintf("Failed to open map file!\n");
            return LW_ERR_GENERIC;
        }
    }

    return LW_ERR_GENERIC;
}

//-----------------------------------------------------
// ofaDumpDmem - Dumps OFA data memory
//-----------------------------------------------------
LW_STATUS ofaDumpDmem_v01_00(LwU32 indexGpu, LwU32 dmemSize, LwU32 offs2MthdOffs)
{
    LW_STATUS status = LW_OK;
    LwU32 dmemSizeMax;
    // these are the variables defined for use in parsing and printinf the methods and data
    LwU32 address, address2, u, i, mthdOffs = 0, classNum;
    LwU32 mthd[MAXMETHODARRAYSIZE] = { 0 };
    LwU32 methodIdx, mthdArraySize;

    dmemSizeMax = (GPU_REG_RD_DRF(_POFA_FALCON, _HWCFG, _DMEM_SIZE) << 8);

    if (dmemSize > 0)
        dmemSize = min(dmemSizeMax, dmemSize);
    else
        dmemSize = dmemSizeMax;

    address = LW_POFA_FALCON_DMEMD(0);
    address2 = LW_POFA_FALCON_DMEMC(0);
    classNum = pOfa[indexGpu].ofaGetClassId();

    dprintf("\n");
    dprintf("lw: -- Gpu %u OFA %X DMEM -- \n", indexGpu, classNum);
    dprintf("lw: -- Gpu %u OFA %X DMEM SIZE =  0x%08x-- \n", indexGpu, classNum, dmemSize);
    dprintf("lw:\n");
    dprintf("\nADDR: 03....00 07....04 0B....08 0F....0C 13....10 17....14 1B....18 1F....1C");
    dprintf("\n-----------------------------------------------------------------------------");


    for (u = 0; u<(dmemSize + 3) / 4; u++)
    {
        i = (u << (0 ? LW_POFA_FALCON_DMEMC_OFFS));
        GPU_REG_WR32(address2, i);
        if ((u % 8 == 0))
        {
            dprintf("\n%04X: ", 4 * u);
        }
        dprintf("%08X ", GPU_REG_RD32(address));
    }
    dprintf("\n\n");

    if (offs2MthdOffs != ILWALID_OFFSET)  // value provided in command line takes higher precedence
    {
        dprintf("Method array offset is stored in DWORD#%d in DMEM\n\n", offs2MthdOffs >> 2);
    }
    else if (ofaGetOff2MthdOffs(classNum, &offs2MthdOffs) == LW_OK)
    {
        dprintf("Method array offset is stored in DWORD#%d in DMEM\n\n", offs2MthdOffs >> 2);
    }
    else
    {
        dprintf("NOTE: Assuming method array offset is stored in DWORD#0 in DMEM\n");
        dprintf("      Please check corresponding ucode map file for correctness\n\n");
        offs2MthdOffs = 0;
    }
    if (offs2MthdOffs >= dmemSizeMax)
    {
        dprintf("ERROR: off2MthdOffs 0x%x is bigger than the DMEM size 0x%x\n", offs2MthdOffs, dmemSizeMax);
        dprintf("       Setting the value to 0 and dumping methods\n\n");
        offs2MthdOffs = 0;
    }

    u = ((offs2MthdOffs >> 2) << (0 ? LW_POFA_FALCON_DMEMC_OFFS));
    GPU_REG_WR32(address2, u);
    mthdOffs = (GPU_REG_RD32(address)) >> 2;
    dprintf("Method array offset : 0x%08x\n\n", mthdOffs << 2);
    if (mthdOffs >= dmemSizeMax)
    {
        dprintf("ERROR: mthdOffs 0x%x is bigger than the DMEM size 0x%x\n", mthdOffs << 2, dmemSizeMax);
        dprintf("       Either the offset to \"%s\" is incorrect or DMEM read resulted in error\n", VAR_NAME_OFF2MTHDOFFS);
        dprintf("       Skipping method dump\n\n");
        return LW_OK;
    }

    if (classNum == 0xC9FA)
        mthdArraySize = METHODARRAYSIZEC9FA;
    else
        mthdArraySize = METHODARRAYSIZEC6FA;
    for (u = 0; u < mthdArraySize; u++)
    {
        i = ((u + mthdOffs) << (0 ? LW_POFA_FALCON_DMEMC_OFFS));
        GPU_REG_WR32(address2, i);
        mthd[u] = GPU_REG_RD32(address);
    }

    dprintf("\n\n[METHODS]\n");
    dprintf("%4s, %8s,    %4s, %8s,    %4s, %8s,    %4s, %8s\n", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data", "Mthd", "Data");
    dprintf("-----------------------------------------------------------------------\n");
    for (u = 0; u < mthdArraySize; u += 4)
    {
        dprintf("%04X: %08X,    %04X: %08X,    %04X: %08X,    %04X: %08X\n",
            METHODBASE + 4 * u, mthd[u], METHODBASE + 4 * (u + 1), mthd[u + 1],
            METHODBASE + 4 * (u + 2), mthd[u + 2], METHODBASE + 4 * (u + 3), mthd[u + 3]);
    }
    dprintf("\n");

    // common methods
    // if this environment variable is present, parse and print out the methods
    if (getelw("LWW_CLASS_SDK") != NULL)
    {
        dprintf("\n[METHODS]\n");
        for (u = 0; u < mthdArraySize; u++)
        {
            if (parseClassHeader(classNum, METHODBASE + 4 * u, mthd[u]))
                dprintf("\n");
        }
        dprintf("\n");
    }
    else
    {
        dprintf("\n[METHODS]\n");
        for (u = 0; u < mthdArraySize; u++)
        {
            for (methodIdx = 0;; methodIdx++)
            {
                if (pOfaMethodTable[methodIdx].m_id == (METHODBASE + 4 * u))
                {
                    ofaPrintMethodData(65, pOfaMethodTable[methodIdx].m_tag, pOfaMethodTable[methodIdx].m_id, mthd[u]);
                    break;
                }
                else if (pOfaMethodTable[methodIdx].m_id == 0)
                {
                    break;
                }
            }
        }
        dprintf("\n");
    }

    //dprintf("lw:\n");
    //dprintf("lw: -- Gpu %u OFA %X registers -- \n", indexGpu, classNum);
    //dprintf("lw:\n");
    //for (u = 0;; u++)
    //{
    //    if (pOfaMethodTable[u].m_id == 0)
    //    {
    //        break;
    //    }
    //    pOfa[indexGpu].ofaPrintPriv(65, pOfaMethodTable[u].m_tag, GetAppMthdParam(pOfaMethodTable[u].m_id));
    //    pOfa[indexGpu].ofaPrintPriv(65, pOfaMethodTable[u].m_tag, (LW_POFA_FALCON_DMATRFBASE << 8) + pOfaMethodTable[u].m_id);
    //    pOfa[indexGpu].ofaPrintPriv(65, pOfaMethodTable[u].m_tag, LW_POFA_FALCON_DMATRFBASE + pOfaMethodTable[u].m_id);
    //    pOfa[indexGpu].ofaPrintPriv(65, pOfaMethodTable[u].m_tag, (LW_POFA_FALCON_DMATRFBASE + pOfaMethodTable[u].m_id)<<8);
    //    dprintf("lw:\n");
    //}

    return LW_OK;

}

//-----------------------------------------------------
// ofaPrintPriv_v01_00
//-----------------------------------------------------
void ofaPrintPriv_v01_00(LwU32 clmn, char *tag, LwU32 id)
{
    size_t len = strlen(tag);

    dprintf("lw: %s", tag);

    if ((len>0) && (len<(clmn + 4)))
    {
        LwU32 i;
        for (i = 0; i<clmn - len; i++)
        {
            dprintf(" ");
        }
    }
    dprintf("(0x%08X)  = 0x%08X\n", id, GPU_REG_RD32(id));
}

//-----------------------------------------------------
// ofaDumpPriv_v01_00 - Dumps OFA priv reg space
//-----------------------------------------------------
LW_STATUS ofaDumpPriv_v01_00(LwU32 indexGpu)
{
    LwU32 u;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u OFA %X priv registers -- \n", indexGpu, pOfa[indexGpu].ofaGetClassId());
    dprintf("lw:\n");

    for (u = 0;; u++)
    {
        if (pOfaPrivReg[0][u].m_id == 0)
        {
            break;
        }
        pOfa[indexGpu].ofaPrintPriv(65, pOfaPrivReg[0][u].m_tag, pOfaPrivReg[0][u].m_id);
    }
    return LW_OK;
}

//--------------------------------------------------------
// ofaDisplayHwcfg_v01_00 - Display OFA HW config
//--------------------------------------------------------
LW_STATUS ofaDisplayHwcfg_v01_00(LwU32 indexGpu)
{
    LwU32 hwcfg, hwcfg1;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u OFA %X HWCFG -- \n", indexGpu, pOfa[indexGpu].ofaGetClassId());
    dprintf("lw:\n");

    hwcfg = GPU_REG_RD32(LW_POFA_FALCON_HWCFG);
    dprintf("lw: LW_POFA_FALCON_HWCFG:  0x%08x\n", hwcfg);
    dprintf("lw:\n");
    dprintf("lw:  IMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",
        DRF_VAL(_POFA, _FALCON_HWCFG, _IMEM_SIZE, hwcfg),
        DRF_VAL(_POFA, _FALCON_HWCFG, _IMEM_SIZE, hwcfg) << 8);
    dprintf("lw:  DMEM_SIZE:        0x%08X (or 0x%08X bytes)\n",
        DRF_VAL(_POFA, _FALCON_HWCFG, _DMEM_SIZE, hwcfg),
        DRF_VAL(_POFA, _FALCON_HWCFG, _DMEM_SIZE, hwcfg) << 8);
    dprintf("lw:  METHODFIFO_DEPTH: 0x%08X\n", DRF_VAL(_POFA, _FALCON_HWCFG, _METHODFIFO_DEPTH, hwcfg));
    dprintf("lw:  DMAQUEUE_DEPTH:   0x%08X\n", DRF_VAL(_POFA, _FALCON_HWCFG, _DMAQUEUE_DEPTH, hwcfg));

    dprintf("lw:\n");

    hwcfg1 = GPU_REG_RD32(LW_POFA_FALCON_HWCFG1);
    dprintf("lw: LW_POFA_FALCON_HWCFG1: 0x%08x\n", hwcfg1);
    dprintf("lw:\n");
    dprintf("lw:  CORE_REV:         0x%08X\n", DRF_VAL(_POFA, _FALCON_HWCFG1, _CORE_REV, hwcfg1));
    dprintf("lw:  SELWRITY_MODEL:   0x%08X\n", DRF_VAL(_POFA, _FALCON_HWCFG1, _SELWRITY_MODEL, hwcfg1));
    dprintf("lw:  IMEM_PORTS:       0x%08X\n", DRF_VAL(_POFA, _FALCON_HWCFG1, _IMEM_PORTS, hwcfg1));
    dprintf("lw:  DMEM_PORTS:       0x%08X\n", DRF_VAL(_POFA, _FALCON_HWCFG1, _DMEM_PORTS, hwcfg1));
    dprintf("lw:  TAG_WIDTH:        0x%08X\n", DRF_VAL(_POFA, _FALCON_HWCFG1, _TAG_WIDTH, hwcfg1));

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
LW_STATUS  ofaDisplayFlcnSPR_v01_00(LwU32 indexGpu)
{
    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u OFA %X Special Purpose Registers -- \n", indexGpu, pOfa[indexGpu].ofaGetClassId());
    dprintf("lw:\n");

    GPU_REG_WR32(LW_POFA_FALCON_ICD_CMD, 0x1008);
    dprintf("lw: OFA IV0 :    0x%08x\n", GPU_REG_RD32(LW_POFA_FALCON_ICD_RDATA));
    GPU_REG_WR32(LW_POFA_FALCON_ICD_CMD, 0x1108);
    dprintf("lw: OFA IV1 :    0x%08x\n", GPU_REG_RD32(LW_POFA_FALCON_ICD_RDATA));
    GPU_REG_WR32(LW_POFA_FALCON_ICD_CMD, 0x1308);
    dprintf("lw: OFA EV  :    0x%08x\n", GPU_REG_RD32(LW_POFA_FALCON_ICD_RDATA));
    GPU_REG_WR32(LW_POFA_FALCON_ICD_CMD, 0x1408);
    dprintf("lw: OFA SP  :    0x%08x\n", GPU_REG_RD32(LW_POFA_FALCON_ICD_RDATA));
    GPU_REG_WR32(LW_POFA_FALCON_ICD_CMD, 0x1508);
    dprintf("lw: OFA PC  :    0x%08x\n", GPU_REG_RD32(LW_POFA_FALCON_ICD_RDATA));
    GPU_REG_WR32(LW_POFA_FALCON_ICD_CMD, 0x1608);
    dprintf("lw: OFA IMB :    0x%08x\n", GPU_REG_RD32(LW_POFA_FALCON_ICD_RDATA));
    GPU_REG_WR32(LW_POFA_FALCON_ICD_CMD, 0x1708);
    dprintf("lw: OFA DMB :    0x%08x\n", GPU_REG_RD32(LW_POFA_FALCON_ICD_RDATA));
    GPU_REG_WR32(LW_POFA_FALCON_ICD_CMD, 0x1808);
    dprintf("lw: OFA CSW :    0x%08x\n", GPU_REG_RD32(LW_POFA_FALCON_ICD_RDATA));
    dprintf("lw:\n\n");

    return LW_OK;
}

