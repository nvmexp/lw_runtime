/* Copyright 2019-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 *_LWRM_COPYRIGHT_END_
 */
#include <stdint.h>
#include <string.h>

#include "ampere/ga102/dev_ce.h"
#include "ampere/ga102/dev_ctrl.h"
#include "ampere/ga102/dev_falcon_v4.h"
#include "ampere/ga102/dev_gsp.h"
#include "ampere/ga102/dev_graphics_nobundle.h"
#include "ampere/ga102/dev_lwdec_pri.h"
#include "ampere/ga102/dev_ofa_pri.h"
#include "ampere/ga102/dev_lwenc_pri_sw.h"
#include "ampere/ga102/dev_runlist.h"
#include "ampere/ga102/dev_sec_pri.h"
#include "ampere/ga102/dev_fb.h"
#include "ampere/ga102/dev_vm.h"
#include "ampere/ga102/dev_vm_addendum.h"

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

/*
 * @brief   Initializes list of interrupts
 */
void intrInit_GA102()
{
    LwU32 i;
    const LwU32 lwdecStride = LW_PLWDEC_FALCON_INTR_RETRIGGER(1,0) - LW_PLWDEC_FALCON_INTR_RETRIGGER(0,0);
    const LwU32 ceStride    = LW_CE_LCE_INTR_CTRL(1) - LW_CE_LCE_INTR_CTRL(0);

    intr_info pStaticTable[] =
    {
        // TODO: find out which of these are actually pulse-based in Ampere

        // GSP RM
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PFB_VECTOR, LW_FALSE, "fb", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_THERMAL_VECTOR, LW_FALSE, "thermal", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PMGR_VECTOR, LW_FALSE, "pmgr", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_DFD_VECTOR, LW_FALSE, "dfd", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PMU_VECTOR, LW_FALSE, "pmu", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_LTC_ALL_VECTOR, LW_FALSE, "ltc (all)", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PDISP_VECTOR, LW_FALSE, "disp", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PRIV_RING_VECTOR, LW_FALSE, "priv_ring", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_HDACODEC_VECTOR, LW_FALSE, "hdacodec", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_PFB_FBHUB_POISON_INTR_VECTOR_HW_INIT, LW_FALSE, "fbhub", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_PFB_PRI_MMU_INT_VECTOR_FAULT_NOTIFY_NON_REPLAYABLE, LW_TRUE, "mmu_nonreplayable_fault", GSP_PRESENT, VIRTUAL_PRESENT},
        {LW_PFB_PRI_MMU_INT_VECTOR_FAULT_ERROR_NON_REPLAYABLE, LW_TRUE, "mmu_nonreplayable_fault_error", GSP_PRESENT, VIRTUAL_PRESENT},
        {LW_PFB_PRI_MMU_INT_VECTOR_INFO_FAULT_VECTOR_INIT, LW_TRUE, "mmu_info_fault", GSP_PRESENT, VIRTUAL_PRESENT},
        {LW_PFB_PRI_MMU_INT_VECTOR_ECC_ERROR_VECTOR_INIT, LW_TRUE, "mmu_ecc_error", GSP_PRESENT, VIRTUAL_PRESENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_IOCTRL_INTR_0_VECTOR, LW_FALSE, "lwlink (intr 0)", GSP_PRESENT, VIRTUAL_ABSENT},
        // lwlink intr 1 interrupt is unused in Ampere, but still has a vector
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_IOCTRL_INTR_1_VECTOR, LW_FALSE, "lwlink (intr 1)", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PTIMER_ALARM_VECTOR, LW_TRUE, "software runlist (SWRL)", GSP_PRESENT, VIRTUAL_ABSENT},

        // CPU RM aka Client RM
        {LW_PFB_PRI_MMU_INT_VECTOR_FAULT_NOTIFY_REPLAYABLE, LW_TRUE, "mmu_replayable_fault", GSP_PRESENT, VIRTUAL_PRESENT},
        {LW_PFB_PRI_MMU_INT_VECTOR_FAULT_ERROR_REPLAYABLE, LW_TRUE, "mmu_replayable_fault_error", GSP_PRESENT, VIRTUAL_PRESENT},
        {LW_PFB_FBHUB_ACCESS_COUNTER_INTR_VECTOR_HW_INIT, LW_TRUE, "access_counter", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_CTRL_SW_INTR_NOSTALL_VECTORID_VALUE_CONSTANT, LW_FALSE, "software", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_XVE_VECTOR, LW_FALSE, "bif", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PBUS_VECTOR, LW_FALSE, "bus", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PTIMER_VECTOR, LW_TRUE, "timer", GSP_PRESENT, VIRTUAL_PRESENT},
        {LW_CTRL_CPU_DOORBELL_VECTORID_VALUE_CONSTANT, LW_TRUE, "cpu_doorbell", GSP_PRESENT, VIRTUAL_PRESENT},
        {LW_CTRL_VIRTUAL_INTR_VECTORID_VALUE_CONSTANT, LW_TRUE, "priv_doorbell", GSP_PRESENT, VIRTUAL_PRESENT},
        // GR_FECS_LOGN have no interrupts
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_GSP_INTR_0_VECTOR, LW_FALSE, "gsp", GSP_ABSENT, VIRTUAL_ABSENT},
    };

    memset(intrEnumTable.interrupts, 0, sizeof(intrEnumTable.interrupts));
    intrEnumTable.intrCtr = 0;
    intrEnumTable.discCtr = 0;

    for (i = 0; i < sizeof(pStaticTable) / sizeof(intr_info); i++)
    {
        intr_info intr = pStaticTable[i];
        pIntr[indexGpu].intrRegister(intr.vector, intr.bPulse, intr.name, intr.bGspTree, intr.bVirtualized, 0, NULL);
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

void intrAddLwenc_GA102(void)
{
    const LwU32 lwencStride = LW_PLWENC_FALCON_INTR_CTRL(1,0) - LW_PLWENC_FALCON_INTR_CTRL(0,0);
    addEngineInfo(ENGINE_TAG_LWENC, "lwenc", LW_PLWENC_FALCON_INTR_CTRL(0,0), "LW_PLWENC_FALCON_INTR_CTRL", LW_PLWENC_FALCON_INTR_RETRIGGER(0,0), "LW_PLWENC_FALCON_INTR_RETRIGGER", LW_PLWENC_FALCON_INTR_CTRL(0,1), "LW_PLWENC_FALCON_INTR_CTRL", lwencStride, 2, LW_TRUE);
}
