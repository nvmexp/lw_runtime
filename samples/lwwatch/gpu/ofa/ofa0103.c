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
#include "hopper/gh100/dev_ofa_pri.h"
#include "hopper/gh100/dev_fuse.h"
#include "g_ofa_private.h"     // (rmconfig)  implementation prototypes

#include <class/clb8fa.h>

dbg_ofa_v01_00 ofaMethodTable_v01_03[] =
{
    privInfo_ofa_v01_00(LWB8FA_NOP),
    privInfo_ofa_v01_00(LWB8FA_PM_TRIGGER),
    privInfo_ofa_v01_00(LWB8FA_SET_APPLICATION_ID),
    privInfo_ofa_v01_00(LWB8FA_SET_WATCHDOG_TIMER),
    privInfo_ofa_v01_00(LWB8FA_SEMAPHORE_A),
    privInfo_ofa_v01_00(LWB8FA_SEMAPHORE_B),
    privInfo_ofa_v01_00(LWB8FA_SEMAPHORE_C),
    privInfo_ofa_v01_00(LWB8FA_CTX_SAVE_AREA),
    privInfo_ofa_v01_00(LWB8FA_CTX_SWITCH),
    privInfo_ofa_v01_00(LWB8FA_SET_SEMAPHORE_PAYLOAD_LOWER),
    privInfo_ofa_v01_00(LWB8FA_SET_SEMAPHORE_PAYLOAD_UPPER),
    privInfo_ofa_v01_00(LWB8FA_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_A),
    privInfo_ofa_v01_00(LWB8FA_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_B),
    privInfo_ofa_v01_00(LWB8FA_EXELWTE),
    privInfo_ofa_v01_00(LWB8FA_SEMAPHORE_D),
    privInfo_ofa_v01_00(LWB8FA_SET_PREDICATION_OFFSET_UPPER),
    privInfo_ofa_v01_00(LWB8FA_SET_PREDICATION_OFFSET_LOWER),
    privInfo_ofa_v01_00(LWB8FA_SET_AUXILIARY_DATA_BUFFER),
    privInfo_ofa_v01_00(LWB8FA_SET_PICTURE_INDEX),
    privInfo_ofa_v01_00(LWB8FA_SET_CONTROL_PARAMS),
    privInfo_ofa_v01_00(LWB8FA_SET_TOTAL_LEVEL_NUM),
    privInfo_ofa_v01_00(LWB8FA_PM_TRIGGER_END),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_LEVEL_INDEX(0)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_LEVEL_INDEX(1)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_LEVEL_INDEX(2)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_LEVEL_INDEX(3)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_LEVEL_INDEX(4)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_DRV_SET_UP_ADDR(0)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_DRV_SET_UP_ADDR(1)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_DRV_SET_UP_ADDR(2)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_DRV_SET_UP_ADDR(3)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_DRV_SET_UP_ADDR(4)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_LWRR_PIC_ADDR(0)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_LWRR_PIC_ADDR(1)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_LWRR_PIC_ADDR(2)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_LWRR_PIC_ADDR(3)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_LWRR_PIC_ADDR(4)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_REF_PIC_ADDR(0)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_REF_PIC_ADDR(1)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_REF_PIC_ADDR(2)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_REF_PIC_ADDR(3)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_REF_PIC_ADDR(4)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_HINT_MV_ADDR(0)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_HINT_MV_ADDR(1)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_HINT_MV_ADDR(2)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_HINT_MV_ADDR(3)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_HINT_MV_ADDR(4)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_WINNER_COST_ADDR(0)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_WINNER_COST_ADDR(1)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_WINNER_COST_ADDR(2)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_WINNER_COST_ADDR(3)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_WINNER_COST_ADDR(4)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_WINNER_FLOW_ADDR(0)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_WINNER_FLOW_ADDR(1)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_WINNER_FLOW_ADDR(2)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_WINNER_FLOW_ADDR(3)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_WINNER_FLOW_ADDR(4)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_HISTORY_BUF_ADDR(0)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_HISTORY_BUF_ADDR(1)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_HISTORY_BUF_ADDR(2)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_HISTORY_BUF_ADDR(3)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_HISTORY_BUF_ADDR(4)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_TEMPORARY_BUF_ADDR(0)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_TEMPORARY_BUF_ADDR(1)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_TEMPORARY_BUF_ADDR(2)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_TEMPORARY_BUF_ADDR(3)),
    privInfo_ofa_v01_00(LWB8FA_SET_MULTI_LEVELS_SET_TEMPORARY_BUF_ADDR(4)),
    privInfo_ofa_v01_00(LWB8FA_SET_STATUS_ADDR),
    privInfo_ofa_v01_00(LWB8FA_PM_TRIGGER_END),
    privInfo_ofa_v01_00(0),
};

dbg_ofa_v01_00 ofaFuseReg_v01_03[] =
{
    // revocation
    privInfo_ofa_v01_00(LW_FUSE_OPT_FUSE_UCODE_OFA_REV),

    // debug vs prod
    privInfo_ofa_v01_00(LW_FUSE_OPT_OFA_ENGINE_RESET_FSM_HALT_CYA_DISABLE),
    privInfo_ofa_v01_00(LW_FUSE_OPT_OFA_PRIV_READ_DIS),
    privInfo_ofa_v01_00(LW_FUSE_OPT_OFA_PRIV_WRITE_DIS),

    // floorsweeping
    privInfo_ofa_v01_00(LW_FUSE_OPT_OFA_DISABLE),
    privInfo_ofa_v01_00(LW_FUSE_OPT_OFA_DISABLE_CP),
    privInfo_ofa_v01_00(LW_FUSE_OPT_OFA_DEFECTIVE),
    privInfo_ofa_v01_00(LW_FUSE_CTRL_OPT_OFA),
    privInfo_ofa_v01_00(LW_FUSE_STATUS_OPT_OFA),
    privInfo_ofa_v01_00(0),
};

dbg_ofa_v01_00 ofaPrivReg_v01_03[] =
{
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEM_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEM_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CPUCTL_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_EXE_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQTMR_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDCTX_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_HSCTL_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SCTL_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_WDTMR_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PRIVSTATE_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_0_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_1_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_2_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_COMMON_SCRATCH_GROUP_3_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_INTR_CTRL_PRIV_LEVEL_MASK(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_INTR_CTRL_PRIV_LEVEL_MASK(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PMB_IMEM_PRIV_LEVEL_MASK(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PMB_DMEM_PRIV_LEVEL_MASK(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PMB_DMEM_PRIV_LEVEL_MASK(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PMB_DMEM_PRIV_LEVEL_MASK(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PMB_DMEM_PRIV_LEVEL_MASK(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMA_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_BOOTVEC_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_AMAP_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_TRACEBUF_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_TMR_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQSCMASK_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DBGCTL_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQSSET),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQSCLR),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQSTAT),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQMODE),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQSCMASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_INTR_CTRL(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_INTR_CTRL(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_INTR_RETRIGGER(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_INTR_RETRIGGER(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQMSET),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQMCLR),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQMASK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQDEST),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IRQDEST2),
    privInfo_ofa_v01_00(LW_POFA_FALCON_GPTMRINT),
    privInfo_ofa_v01_00(LW_POFA_FALCON_GPTMRVAL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_GPTMRCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PTIMER0),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PTIMER1),
    privInfo_ofa_v01_00(LW_POFA_FALCON_WDTMRVAL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_WDTMRCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDDATA),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDID),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDWDAT),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDCOUNT),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDPOP),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MTHDRAMSZ),
    privInfo_ofa_v01_00(LW_POFA_FALCON_LWRCTX),
    privInfo_ofa_v01_00(LW_POFA_FALCON_NXTCTX),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CTXACK),
    privInfo_ofa_v01_00(LW_POFA_FALCON_LWRCTX2),
    privInfo_ofa_v01_00(LW_POFA_FALCON_NXTCTX2),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MAILBOX0),
    privInfo_ofa_v01_00(LW_POFA_FALCON_MAILBOX1),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ITFEN),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IDLESTATE),
    privInfo_ofa_v01_00(LW_POFA_FALCON_FHSTATE),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PRIVSTATE),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SFTRESET),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ENGCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ADDR),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMACTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMATRFBASE),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMATRFMOFFS),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMATRFCMD),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMATRFFBOFFS),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAPOLL_FB),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMAPOLL_CP),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMATRFBASE1),
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
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEMC(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEMD(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEMT(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMC(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMC(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMC(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMC(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMD(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMD(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMD(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMD(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMT(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMT(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMT(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEMT(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEML(0)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEML(1)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEML(2)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEML(3)),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMEM_DUMMY),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMEM_DUMMY),
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
    privInfo_ofa_v01_00(LW_POFA_FALCON_PRGLWER),
    privInfo_ofa_v01_00(LW_POFA_FALCON_HWCFG),
    privInfo_ofa_v01_00(LW_POFA_FALCON_HWCFG1),
    privInfo_ofa_v01_00(LW_POFA_FALCON_HWCFG2),
    privInfo_ofa_v01_00(LW_POFA_FALCON_HWCFG3),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CGCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CG2),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SCTL1),
    privInfo_ofa_v01_00(LW_POFA_FALCON_HSCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_OS),
    privInfo_ofa_v01_00(LW_POFA_FALCON_RM),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SOFT_PM),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SOFT_MODE),
    privInfo_ofa_v01_00(LW_POFA_FALCON_PMM),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DEBUG1),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DEBUGINFO),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SP_MIN),
    privInfo_ofa_v01_00(LW_POFA_FALCON_EXCI),
    privInfo_ofa_v01_00(LW_POFA_FALCON_EXCI2),
    privInfo_ofa_v01_00(LW_POFA_FALCON_SVEC_SPR),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CPUCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CPUCTL_ALIAS),
    privInfo_ofa_v01_00(LW_POFA_FALCON_STACKCFG),
    privInfo_ofa_v01_00(LW_POFA_FALCON_BOOTVEC),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DBGCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IBRKPT1),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IBRKPT2),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IBRKPT3),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IBRKPT4),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IBRKPT5),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ICD_CMD),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ICD_ADDR),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ICD_WDATA),
    privInfo_ofa_v01_00(LW_POFA_FALCON_ICD_RDATA),
    privInfo_ofa_v01_00(LW_POFA_FALCON_RSTAT0),
    privInfo_ofa_v01_00(LW_POFA_FALCON_RSTAT3),
    privInfo_ofa_v01_00(LW_POFA_FALCON_TRACEIDX),
    privInfo_ofa_v01_00(LW_POFA_FALCON_TRACEPC),
    privInfo_ofa_v01_00(LW_POFA_FALCON_TRACEINFO),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMCTL_DEBUG),
    privInfo_ofa_v01_00(LW_POFA_FALCON_IMSTAT),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMCTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMSTAT),
    privInfo_ofa_v01_00(LW_POFA_FALCON_DMVACTL),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CSBERRSTAT),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CSBERR_ADDR),
    privInfo_ofa_v01_00(LW_POFA_FALCON_CSBERR_INFO),
    privInfo_ofa_v01_00(LW_POFA_FBIF_REGIONCFG_PRIV_LEVEL_MASK),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(0)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(1)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(2)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(3)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(4)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(5)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(6)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_TRANSCFG(7)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_INSTBLK),
    privInfo_ofa_v01_00(LW_POFA_FBIF_BIND_STATUS),
    privInfo_ofa_v01_00(LW_POFA_FBIF_CTL),
    privInfo_ofa_v01_00(LW_POFA_FBIF_THROTTLE),
    privInfo_ofa_v01_00(LW_POFA_FBIF_REGIONCFG),
    privInfo_ofa_v01_00(LW_POFA_FBIF_CG1),
    privInfo_ofa_v01_00(LW_POFA_FBIF_ACHK_BLK(0)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_ACHK_BLK(1)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_ACHK_CTL(0)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_ACHK_CTL(1)),
    privInfo_ofa_v01_00(LW_POFA_FBIF_DBG_STAT0),
    privInfo_ofa_v01_00(LW_POFA_FBIF_DBG_STAT1),
    privInfo_ofa_v01_00(LW_POFA_FBIF_DBG_STAT2),
    privInfo_ofa_v01_00(LW_POFA_FBIF_DBG_RDCOUNT_LO),
    privInfo_ofa_v01_00(LW_POFA_FBIF_DBG_RDCOUNT_HI),
    privInfo_ofa_v01_00(LW_POFA_FBIF_DBG_WRCOUNT_LO),
    privInfo_ofa_v01_00(LW_POFA_FBIF_DBG_WRCOUNT_HI),
    privInfo_ofa_v01_00(LW_POFA_FBIF_BW_ALLOC),
    privInfo_ofa_v01_00(LW_POFA_CTRL),
    privInfo_ofa_v01_00(LW_POFA_COMMAND),
    privInfo_ofa_v01_00(LW_POFA_PYD_CTRL),
    privInfo_ofa_v01_00(LW_POFA_FRAME_SIZE),
    privInfo_ofa_v01_00(LW_POFA_ROI_START),
    privInfo_ofa_v01_00(LW_POFA_ROI_END),
    privInfo_ofa_v01_00(LW_POFA_SUBFRAME),
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
    privInfo_ofa_v01_00(LW_POFA_LWRR_SURFACE_BUFFER_SIZE),
    privInfo_ofa_v01_00(LW_POFA_REF_SURFACE_OFFSET),
    privInfo_ofa_v01_00(LW_POFA_REF_SURFACE_STRIDE),
    privInfo_ofa_v01_00(LW_POFA_REF_SURFACE_FORMAT),
    privInfo_ofa_v01_00(LW_POFA_REF_SURFACE_BUFFER_SIZE),
    privInfo_ofa_v01_00(LW_POFA_FLOW_SURFACE_OFFSET),
    privInfo_ofa_v01_00(LW_POFA_FLOW_SURFACE_STRIDE),
    privInfo_ofa_v01_00(LW_POFA_FLOW_SURFACE_FORMAT),
    privInfo_ofa_v01_00(LW_POFA_FLOW_SURFACE_BUFFER_SIZE),
    privInfo_ofa_v01_00(LW_POFA_COST_SURFACE_OFFSET),
    privInfo_ofa_v01_00(LW_POFA_COST_SURFACE_STRIDE),
    privInfo_ofa_v01_00(LW_POFA_COST_SURFACE_FORMAT),
    privInfo_ofa_v01_00(LW_POFA_COST_SURFACE_BUFFER_SIZE),
    privInfo_ofa_v01_00(LW_POFA_SGM_HIST_SURFACE_OFFSET),
    privInfo_ofa_v01_00(LW_POFA_SGM_HIST_SURFACE_BUFFER_SIZE),
    privInfo_ofa_v01_00(LW_POFA_SGM_TEMP_SURFACE_OFFSET),
    privInfo_ofa_v01_00(LW_POFA_SGM_TEMP_SURFACE_BUFFER_SIZE),
    privInfo_ofa_v01_00(LW_POFA_PYD_HINT_SURFACE_OFFSET),
    privInfo_ofa_v01_00(LW_POFA_PYD_HINT_SURFACE_STRIDE),
    privInfo_ofa_v01_00(LW_POFA_PYD_HINT_SURFACE_FORMAT),
    privInfo_ofa_v01_00(LW_POFA_PYD_HINT_SURFACE_BUFFER_SIZE),
    privInfo_ofa_v01_00(LW_POFA_OFA_ENGINE_INTEN),
    privInfo_ofa_v01_00(LW_POFA_OFA_ENGINE_INTR),
    privInfo_ofa_v01_00(LW_POFA_CG2),
    privInfo_ofa_v01_00(LW_POFA_BOOT_42),
    privInfo_ofa_v01_00(LW_POFA_OPT_FUSE_UCODE_OFA_REV),
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
    privInfo_ofa_v01_00(LW_POFA_DBG_CRC_CONTROL),
    privInfo_ofa_v01_00(LW_POFA_DBG_CPF2RPF_GRID),
    privInfo_ofa_v01_00(LW_POFA_DBG_SGM_POS),
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
    privInfo_ofa_v01_00(LW_POFA_PMM),
    privInfo_ofa_v01_00(LW_POFA_PMM1),
    privInfo_ofa_v01_00(LW_POFA_PMM2),
    privInfo_ofa_v01_00(LW_POFA_CG),
    privInfo_ofa_v01_00(LW_POFA_CG1),
    privInfo_ofa_v01_00(LW_POFA_LWRR_SURFACE_OFFSET_HI),
    privInfo_ofa_v01_00(LW_POFA_REF_SURFACE_OFFSET_HI),
    privInfo_ofa_v01_00(LW_POFA_FLOW_SURFACE_OFFSET_HI),
    privInfo_ofa_v01_00(LW_POFA_COST_SURFACE_OFFSET_HI),
    privInfo_ofa_v01_00(LW_POFA_SGM_HIST_SURFACE_OFFSET_HI),
    privInfo_ofa_v01_00(LW_POFA_SGM_TEMP_SURFACE_OFFSET_HI),
    privInfo_ofa_v01_00(LW_POFA_PYD_HINT_SURFACE_OFFSET_HI),
     privInfo_ofa_v01_00(0),
};


//-----------------------------------------------------
// ofaIsGpuSupported_v01_03
//-----------------------------------------------------
BOOL ofaIsGpuSupported_v01_03(LwU32 indexGpu)
{
    LwU32 privWarnReadDisable  = 0;
    LwU32 privWarnWriteDisable = 0;

    if (ofaId != LWWATCH_OFA_0)
    {
        dprintf("Only OFA0 supported on this GPU\n");
        return FALSE;
    }
    dprintf("OFA0 supported on this GPU\n");
    pOfaPrivReg[0]  = ofaPrivReg_v01_03;
    pOfaFuseReg     = ofaFuseReg_v01_03;
    pOfaMethodTable = ofaMethodTable_v01_03;

    privWarnReadDisable  = (GPU_REG_RD32(LW_FUSE_OPT_OFA_PRIV_READ_DIS) & 0x1);
    if (privWarnReadDisable)
    {
        dprintf("WARNING: OFA%d: Fixed function HW unit register's priv READ is disabled by fuse, register reads all zeros, only FALCON, FBIF, CG, PMM registers are readable depending on PLM settings\n",
                ofaId);
    }

    privWarnWriteDisable = (GPU_REG_RD32(LW_FUSE_OPT_OFA_PRIV_WRITE_DIS) & 0x1); 
    if (privWarnWriteDisable)
    {
        dprintf("WARNING: OFA%d: Fixed function HW unit register's priv WRITE is disabled by fuse, register writes have no effect, only FALCON, FBIF, CG, PMM registers are writeable depending on PLM settings\n",
                ofaId); 
    }

    return TRUE;
}

//-----------------------------------------------------
// ofaGetClassId_v01_03
//-----------------------------------------------------
LwU32
ofaGetClassId_v01_03(void)
{
    return LWB8FA_VIDEO_OFA;
}

//-----------------------------------------------------
// ofaDumpFuse_v01_03 - Dumps OFA related fuse registers 
//-----------------------------------------------------
LW_STATUS ofaDumpFuse_v01_03(LwU32 indexGpu)
{
    LwU32 u;

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u OFA %X fuse registers -- \n", indexGpu, pOfa[indexGpu].ofaGetClassId());
    dprintf("lw:\n");

    for (u = 0;; u++)
    {
        if (pOfaFuseReg[u].m_id == 0)
        {
            break;
        }
        pOfa[indexGpu].ofaPrintPriv(65, pOfaFuseReg[u].m_tag, pOfaFuseReg[u].m_id);
    }
    return LW_OK;
}
