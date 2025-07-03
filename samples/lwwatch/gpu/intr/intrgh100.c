/* Copyright 2019-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 *_LWRM_COPYRIGHT_END_
 */
#include <stdint.h>
#include <string.h>

#include "hopper/gh100/dev_ce.h"
#include "hopper/gh100/dev_ctrl.h"
#include "hopper/gh100/dev_falcon_v4.h"
#include "hopper/gh100/dev_gsp.h"
#include "hopper/gh100/dev_graphics_nobundle.h"
#include "hopper/gh100/dev_lwdec_pri.h"
#include "hopper/gh100/dev_lwjpg_pri_sw.h"
#include "hopper/gh100/dev_ofa_pri.h"
#include "hopper/gh100/dev_runlist.h"
#include "hopper/gh100/dev_sec_pri.h"
#include "hopper/gh100/dev_fb.h"
#include "hopper/gh100/pri_lw_xal_ep.h"
#include "hopper/gh100/dev_xtl_ep_pri.h"
#include "hopper/gh100/dev_pmgr.h"
#include "hopper/gh100/dev_cpr_ip.h"
#include "hopper/gh100/dev_fbpa.h"
#include "hopper/gh100/dev_ltc.h"
#include "hopper/gh100/dev_pri_ringmaster.h"
#include "hopper/gh100/dev_pri_ringmaster.h"
#include "hopper/gh100/dev_fsp_pri.h"
#include "hopper/gh100/dev_c2c.h"
#include "hopper/gh100/dev_pwr_pri.h"
#include "hopper/gh100/ioctrl_discovery.h"
#include "hopper/gh100/dev_vm.h"
#include "hopper/gh100/dev_vm_addendum.h"
#include "hopper/gh100/hwproject.h"

#include "deviceinfo.h"
#include "fifo.h"
#include "g_intr_private.h"
#include "hal.h"
#include "intr.h"
#include "intr_private.h"
#include "lwwatch.h"
#include "os.h"
#include "print.h"

#define INTR_TOPS_IMPLEMENTED_PER_FN (LW_CTRL_CPU_INTR_TOP__SIZE_1/MAX_GFIDS)
#define INTR_PER_TOP (LW_CTRL_CPU_INTR_LEAF_ARRAY_SIZE_PER_FN*REG_NUM_BITS/INTR_TOPS_IMPLEMENTED_PER_FN)
#define INTR_PER_GSP_TOP (LW_CTRL_GSP_INTR_LEAF__SIZE_1*REG_NUM_BITS/LW_CTRL_GSP_INTR_TOP__SIZE_1)

LwU32 intrFetchVectorID_GH100(LwU32 reg, LwBool *bad)
{
    LwU32 v = GPU_REG_RD32(reg);

    if ((v & 0xfff00000) == 0xbad00000)
    {
        *bad = LW_TRUE;
        dprintf("**** BAD ADDRESS *****\n");
    }
    else
    {
        *bad = LW_FALSE;
    }

    return DRF_VAL(_CTRL, _INTR_CTRL_ACCESS_DEFINES, _VECTOR, v);
}

LwU32 intrGetRunlistIntrCtrl_GH100(LwU32 idx)
{
    return LW_RUNLIST_INTR_CTRL(idx);
}

void intrAddLwjpg_GH100(void)
{
    const LwU32 lwjpgStride = LW_PLWJPG_FALCON_INTR_CTRL(1,0) - LW_PLWJPG_FALCON_INTR_CTRL(0,0);
    addEngineInfo(ENGINE_TAG_LWJPG, "lwjpg", LW_PLWJPG_FALCON_INTR_CTRL(0, 0), "LW_PLWJPG_FALCON_INTR_CTRL", LW_PLWJPG_FALCON_INTR_RETRIGGER(0, 0), "LW_PLWJPG_FALCON_INTR_RETRIGGER", LW_PLWJPG_FALCON_INTR_CTRL(0, 1), "LW_PLWJPG_FALCON_INTR_CTRL", lwjpgStride, 2, LW_TRUE);
}

/*
 * @brief   Initializes list of interrupts
 * @detail  The goal is to auto-generate this code in Hopper
 *          The table here mirrors mcGetStaticInterruptTable_GH100
 */
void intrInit_GH100()
{
    LwU32 i;
    const LwU32 lwdecStride = LW_PLWDEC_FALCON_INTR_CTRL(1,0) - LW_PLWDEC_FALCON_INTR_CTRL(0,0);
    const LwU32 ceStride    = LW_CE_LCE_INTR_CTRL(1) - LW_CE_LCE_INTR_CTRL(0);

    intr_info pStaticTable[] =
    {
        // All interrupts are pulse-based in Hopper

        // GSP RM
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PFB_VECTOR, LW_TRUE, "fb", GSP_PRESENT, VIRTUAL_ABSENT, LW_PFB_FBPA_INTR_RETRIGGER, "LW_PFB_FBPA_INTR_RETRIGGER"},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_THERMAL_VECTOR, LW_TRUE, "thermal", GSP_PRESENT, VIRTUAL_ABSENT, LW_PPWR_FALCON_INTR_RETRIGGER(1), "LW_PPWR_FALCON_INTR_RETRIGGER(1)"},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PMGR_VECTOR, LW_TRUE, "pmgr", GSP_PRESENT, VIRTUAL_ABSENT, LW_PMGR_RM_INTR_RETRIGGER, "LW_PMGR_RM_INTR_RETRIGGER"},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PMU_VECTOR, LW_TRUE, "pmu", GSP_PRESENT, VIRTUAL_ABSENT, LW_PPWR_FALCON_INTR_RETRIGGER(0), "LW_PPWR_FALCON_INTR_RETRIGGER(0)"},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_LTC_ALL_VECTOR, LW_TRUE, "ltc (all)", GSP_PRESENT, VIRTUAL_ABSENT, LW_PLTCG_LTCS_INTR_RETRIGGER, "LW_PLTCG_LTCS_INTR_RETRIGGER"},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PRIV_RING_VECTOR, LW_TRUE, "priv_ring", GSP_PRESENT, VIRTUAL_ABSENT, LW_PPRIV_MASTER_PRIV_RING_INTR_RETRIGGER, "LW_PPRIV_MASTER_PRIV_RING_INTR_RETRIGGER"},
        {LW_PFB_FBHUB_POISON_INTR_VECTOR_HW_INIT, LW_TRUE, "fbhub", GSP_PRESENT, VIRTUAL_ABSENT, 0, ""},
        {LW_PFB_PRI_MMU_INT_VECTOR_FAULT_NOTIFY_NON_REPLAYABLE, LW_TRUE, "mmu_nonreplayable_fault", GSP_PRESENT, VIRTUAL_PRESENT, 0, ""},
        {LW_PFB_PRI_MMU_INT_VECTOR_FAULT_ERROR_NON_REPLAYABLE, LW_TRUE, "mmu_nonreplayable_fault_error", GSP_PRESENT, VIRTUAL_PRESENT, 0, ""},
        {LW_PFB_PRI_MMU_INT_VECTOR_INFO_FAULT_VECTOR_INIT, LW_TRUE, "mmu_info_fault", GSP_PRESENT, VIRTUAL_PRESENT, 0, ""},
        {LW_PFB_PRI_MMU_INT_VECTOR_ECC_ERROR_VECTOR_INIT, LW_TRUE, "mmu_ecc_error", GSP_PRESENT, VIRTUAL_PRESENT, 0, ""},
        // TODO: We're only indexing the first CPR_INTR_RETRIGGER here because there is not broadcast one. Index more of them?
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_IOCTRL_INTR_0_VECTOR, LW_TRUE, "lwlink (intr 0)", GSP_PRESENT, VIRTUAL_ABSENT, LW_DISCOVERY_IOCTRL_UNICAST_0_SW_DEVICE_BASE_CPR + LW_CPR_SYS_INTR_RETRIGGER(0), "LW_CPR_SYS_INTR_RETRIGGER(0)"},
        // lwlink intr 1 interrupt is unused in Hopper, but still has a vector
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_IOCTRL_INTR_1_VECTOR, LW_TRUE, "lwlink (intr 1)", GSP_PRESENT, VIRTUAL_ABSENT, 0, ""},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PTIMER_ALARM_VECTOR, LW_TRUE, "software runlist (SWRL)", GSP_PRESENT, VIRTUAL_ABSENT, 0, ""},
        {LW_PFSP_INTR_VECTOR, LW_TRUE, "fsp", GSP_PRESENT, VIRTUAL_ABSENT, LW_PFSP_FALCON_INTR_RETRIGGER(0), "LW_PFSP_FALCON_INTR_RETRIGGER(0)"},

        // CPU RM aka Client RM
        {LW_PFB_PRI_MMU_INT_VECTOR_FAULT_NOTIFY_REPLAYABLE, LW_TRUE, "mmu_replayable_fault", GSP_PRESENT, VIRTUAL_PRESENT, 0, ""},
        {LW_PFB_PRI_MMU_INT_VECTOR_FAULT_ERROR_REPLAYABLE, LW_TRUE, "mmu_replayable_fault_error", GSP_PRESENT, VIRTUAL_PRESENT, 0, ""},
        {LW_PFB_FBHUB_ACCESS_COUNTER_INTR_VECTOR_HW_INIT, LW_TRUE, "access_counter", GSP_PRESENT, VIRTUAL_ABSENT, 0, ""},
        {LW_CTRL_SW_INTR_NOSTALL_VECTORID_VALUE_CONSTANT, LW_TRUE, "software", GSP_PRESENT, VIRTUAL_ABSENT, 0, ""},
        // XTL actually has four retriggers, see bifIntrRetrigger_GH100 in RM
        {DRF_VAL(_CTRL, _INTR_CTRL_ACCESS_DEFINES, _VECTOR, LW_XTL_EP_PRI_INTR_CTRL_0_MESSAGE_INIT), LW_TRUE, "bif", GSP_PRESENT, VIRTUAL_ABSENT, LW_XTL_EP_PRI_INTR_RETRIGGER(0) + LW_XTL_BASE_ADDRESS, "LW_XTL_EP_PRI_INTR_RETRIGGER(0)"},
        // C2C actually has two broadcast retriggers, see bifIntrRetriggerC2C_GH100 in RM
        {DRF_VAL(_CTRL, _INTR_CTRL_ACCESS_DEFINES, _VECTOR, LW_PC2C_C2CS0_TL_TX_C2C_INTR_CTRL_MSG_INIT), LW_TRUE, "bif", GSP_PRESENT, VIRTUAL_ABSENT, LW_PC2C_C2CS0_TL_TX_C2C_INTR_RETRIGGER, "LW_PC2C_C2CS0/1_TL_TX_C2C_INTR_RETRIGGER"},
        {DRF_VAL(_CTRL, _INTR_CTRL_ACCESS_DEFINES, _VECTOR, LW_XAL_EP_INTR_CTRL_MESSAGE_INIT), LW_TRUE, "bus", GSP_PRESENT, VIRTUAL_ABSENT, LW_XAL_EP_INTR_RETRIGGER, "LW_XAL_EP_INTR_RETRIGGER"},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PTIMER_VECTOR, LW_TRUE, "timer", GSP_PRESENT, VIRTUAL_PRESENT, 0, ""},
        {LW_CTRL_CPU_DOORBELL_VECTORID_VALUE_CONSTANT, LW_TRUE, "cpu_doorbell", GSP_PRESENT, VIRTUAL_PRESENT, 0, ""},
        {LW_CTRL_VIRTUAL_INTR_VECTORID_VALUE_CONSTANT, LW_TRUE, "priv_doorbell", GSP_PRESENT, VIRTUAL_PRESENT, 0, ""},
        // GR_FECS_LOGN have no interrupts
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_GSP_INTR_0_VECTOR, LW_TRUE, "gsp", GSP_ABSENT, VIRTUAL_ABSENT, 0, ""},
    };

    memset(intrEnumTable.interrupts, 0, sizeof(intrEnumTable.interrupts));
    intrEnumTable.intrCtr = 0;
    intrEnumTable.discCtr = 0;

    for (i = 0; i < sizeof(pStaticTable) / sizeof(intr_info); i++)
    {
        intr_info intr = pStaticTable[i];
        pIntr[indexGpu].intrRegister(intr.vector, intr.bPulse, intr.name, intr.bGspTree, intr.bVirtualized, intr.retriggerAddress, intr.retriggerName);
    }

    // TODO: This loop will be used when -intr can be reinitialized to pick up new interrupt programmings
    for (i = 0; i < INTR_MAX; i++)
    {
        intr_type *intr = &intrEnumTable.interrupts[i];
        while (intr != NULL)
        {
            intr_type *oldintr = intr;
            intr = intr->next;
            oldintr->next = NULL;

            if (oldintr != &intrEnumTable.interrupts[i])
            {
                free(oldintr);
            }
        }
    }

    memset(intrEnumTable.intrDisc, 0, sizeof(intrEnumTable.intrDisc));
    memset(intrEnumTable.discInfo, 0, sizeof(intrEnumTable.discInfo));

    addEngineInfo(ENGINE_TAG_GR, "gr", LW_PGRAPH_INTR_CTRL, "LW_PGRAPH_INTR_CTRL", LW_PGRAPH_INTR_RETRIGGER, "LW_PGRAPH_INTR_RETRIGGER", LW_PGRAPH_INTR_NOTIFY_CTRL, "LW_PGRAPH_INTR_NOTIFY_CTRL", 0, 0, LW_TRUE);
    addEngineInfo(ENGINE_TAG_CE, "ce", LW_CE_LCE_INTR_CTRL(0), "LW_CE_LCE_INTR_CTRL", LW_CE_LCE_INTR_RETRIGGER(0), "LW_CE_LCE_INTR_RETRIGGER", LW_CE_LCE_INTR_NOTIFY_CTRL(0), "LW_CE_LCE_INTR_NOTIFY_CTRL", ceStride, 1, LW_TRUE);
    addEngineInfo(ENGINE_TAG_LWDEC, "lwdec", LW_PLWDEC_FALCON_INTR_CTRL(0,0), "LW_PLWDEC_FALCON_INTR_CTRL", LW_PLWDEC_FALCON_INTR_RETRIGGER(0,0), "LW_PLWDEC_FALCON_INTR_RETRIGGER", LW_PLWDEC_FALCON_INTR_CTRL(0,1), "LW_PLWDEC_FALCON_INTR_CTRL", lwdecStride, 2, LW_TRUE);
    addEngineInfo(ENGINE_TAG_SEC2, "sec2_", LW_PSEC_FALCON_INTR_CTRL(0), "LW_PSEC_FALCON_INTR_CTRL", LW_PSEC_FALCON_INTR_RETRIGGER(0), "LW_PSEC_FALCON_INTR_RETRIGGER", LW_PSEC_FALCON_INTR_CTRL(1), "LW_PSEC_FALCON_INTR_CTRL", 0, 1, LW_TRUE);
    addEngineInfo(ENGINE_TAG_IOCTRL, "", 0, "", 0, "", 0, "", 0, 0, LW_FALSE);
    addEngineInfo(ENGINE_TAG_OFA, "ofa", LW_POFA_FALCON_INTR_CTRL(0), "LW_POFA_FALCON_INTR_CTRL", LW_POFA_FALCON_INTR_RETRIGGER(0), "LW_POFA_FALCON_INTR_RETRIGGER", LW_POFA_FALCON_INTR_CTRL(1), "LW_POFA_FALCON_INTR_CTRL", 0, 1, LW_TRUE);
    addEngineInfo(ENGINE_TAG_GSP, "gsp", LW_PGSP_FALCON_INTR_CTRL(0), "LW_PGSP_FALCON_INTR_CTRL", LW_PGSP_FALCON_INTR_RETRIGGER(0), "LW_PGSP_FALCON_INTR_RETRIGGER", LW_PGSP_FALCON_INTR_CTRL(1), "LW_PGSP_FALCON_INTR_CTRL", 0, 1, LW_TRUE);
    addEngineInfo(ENGINE_TAG_FLA, "", 0, "", 0, "", 0, "", 0, 0, LW_FALSE);
    addEngineInfo(ENGINE_TAG_UNKNOWN, "", 0, "", 0, "", 0, "", 0, 0, LW_FALSE);
    pIntr[indexGpu].intrAddLwenc();
    pIntr[indexGpu].intrAddLwjpg();

    pIntr[indexGpu].intrInitDiscAll();
}
