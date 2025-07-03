/* Copyright 2019-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 *_LWRM_COPYRIGHT_END_
 */
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "ampere/ga100/dev_ce.h"
#include "ampere/ga100/dev_ctrl.h"
#include "ampere/ga100/dev_falcon_v4.h"
#include "ampere/ga100/dev_gsp.h"
#include "ampere/ga100/dev_graphics_nobundle.h"
#include "ampere/ga100/dev_lwdec_pri.h"
#include "ampere/ga100/dev_lwjpg_pri.h"
#include "ampere/ga100/dev_ofa_pri.h"
#include "ampere/ga100/dev_runlist.h"
#include "ampere/ga100/dev_sec_pri.h"
#include "ampere/ga100/dev_fb.h"
#include "ampere/ga100/dev_vm.h"
#include "ampere/ga100/dev_vm_addendum.h"

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

// =============================== INIT FUNCS ============================== //

LwU32 intrFetchVectorID_GA100(LwU32 reg, LwBool *bad)
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
    //
    // Hack: this corresponds to the INTR_CTRL_VECTOR field, but
    // it's known to be in the same place for all Ampere units
    //
    return v & 0xFFF;
}

/**
 * @brief   Initializes interrupt info in the interrupts list
 */
void intrRegister_GA100
(
    LwU32 gpuIV,
    LwBool bPulse,
    char *name,
    LwBool bGspTree,
    LwBool bVirtualized,
    LwU32 retriggerAddress,
    char *retriggerName
)
{
    intr_type *i = &intrEnumTable.interrupts[gpuIV];
    i->gpuIV = gpuIV;
    i->bPulse = bPulse;
    i->name = name;
    i->leafReg = INTR_LEAF_IDX(gpuIV);
    i->leafBit = INTR_LEAF_BIT_IDX(gpuIV);
    i->topReg = INTR_TOP_IDX(gpuIV);
    i->topBit = INTR_TOP_BIT_IDX(gpuIV);
    i->bGspTree = bGspTree;
    i->bVirtualized = bVirtualized;
    // Retriggers are only used for non-host-based units on Hopper+
    i->retriggerAddress = retriggerAddress;
    i->retriggerName = retriggerName;

    i->bTag = LW_TRUE;
    i->bDiscovery = LW_FALSE;
    i->bFound = LW_FALSE;

    i->next = NULL;

    intrEnumTable.intrCtr++;
    return;
}

/**
 * @brief   Adds to list of interrupts whose vector ids cannot be cached
 */
void intrRegisterDiscWithRetrigger(LwU32 ctrlReg, char *name, LwU32 retriggerAddress,
                             char *retriggerName)
{
    intr_disc_type *i = &intrEnumTable.intrDisc[intrEnumTable.discCtr];
    i->ctrlReg = ctrlReg;
    strncpy(i->name, name, INTR_NAME_MAXLEN - 1);
    i->retriggerAddress = retriggerAddress;
    strncpy(i->retriggerName, retriggerName, INTR_NAME_MAXLEN - 1);

    intrEnumTable.discCtr++;
    return;
}

/**
 * @brief   Initializes interrupt info for discovered interrupt in list
 */
LwBool intrUpdateDisc(intr_disc_type *disc)
{
    LwU32 gpuIV;
    intr_type *i;
    LwBool bBadaddress;

    gpuIV = pIntr[indexGpu].intrFetchVectorID(disc->ctrlReg, &bBadaddress);
    if (bBadaddress)
    {
        dprintf("failed to register %s with addr 0x%x\n",
                disc->name, disc->ctrlReg);
    }
    else
    {
        i = &intrEnumTable.interrupts[gpuIV];

        if (i->bTag)
        {
            while (i->next != NULL && i->next->bTag == LW_TRUE)
            {
                i = i->next;
            }

            if (i->next == NULL)
                i->next = (intr_type *)calloc(1, sizeof(intr_type));

            if (i->next == NULL)
            {
                return !bBadaddress;
            }

            i = i->next;
        }
        else
        {
            intr_type *itr = i;
            while (itr != NULL)
            {
                itr->bTag = LW_FALSE;
                itr = itr->next;
            }
        }

        i->gpuIV = gpuIV;
        i->bPulse = LW_TRUE;
        i->name = disc->name;
        i->leafReg = INTR_LEAF_IDX(gpuIV);
        i->leafBit = INTR_LEAF_BIT_IDX(gpuIV);
        i->topReg = INTR_TOP_IDX(gpuIV);
        i->topBit = INTR_TOP_BIT_IDX(gpuIV);
        i->bGspTree = GSP_PRESENT;
        i->bVirtualized = VIRTUAL_ABSENT;
        i->retriggerAddress = disc->retriggerAddress;
        i->retriggerName = disc->retriggerName;

        i->bTag = LW_TRUE;
        i->bDiscovery = LW_TRUE;
    }

    return !bBadaddress;
}

LwU32 intrGetRunlistIntrCtrl_GA100(LwU32 idx)
{
    return LW_RUNLIST_INTR_VECTORID(idx);
}

void intrInitDiscAll_GA100(void)
{
    LwU32 i, u;

    // GET RUNLIST INTERRUPTS AND COUNT ENGINE INSTANCES
    if (LW_OK == pFifo[indexGpu].fifoGetDeviceInfo())
    {
        for (i = 0; i < deviceInfo.enginesCount; i++)
        {
            if (!deviceInfo.pEngines[i].bHostEng)
                continue;

            if (deviceInfo.cfg.version >= 2)
            {
                LwU32 runListPriBase, reg0, reg1, retrig0, retrig1;
                char name0[INTR_NAME_MAXLEN];
                char name1[INTR_NAME_MAXLEN];
                const char *engineName = deviceInfo.pEngines[i].engineName;

                LwU32 engineTag = deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_ENGINE_TAG];

                if (engineTag != ENGINE_TAG_ILWALID)
                {
                    intrEnumTable.discInfo[engineTag].count++;
                }

                runListPriBase = deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_RUNLIST_PRI_BASE];
                memset(name0, 0, INTR_NAME_MAXLEN * sizeof(char));
                memset(name1, 0, INTR_NAME_MAXLEN * sizeof(char));

                sprintf(name0, "esched_%s(0)", engineName);
                sprintf(name1, "esched_%s(1)", engineName);

                reg0 = runListPriBase + pIntr[indexGpu].intrGetRunlistIntrCtrl(0);
                reg1 = runListPriBase + pIntr[indexGpu].intrGetRunlistIntrCtrl(1);
                retrig0 = runListPriBase + LW_RUNLIST_INTR_RETRIGGER(0);
                retrig1 = runListPriBase + LW_RUNLIST_INTR_RETRIGGER(1);

                intrRegisterDiscWithRetrigger(reg0, name0, retrig0, "N/A");
                intrRegisterDiscWithRetrigger(reg1, name1, retrig1, "N/A");
            }
        }

        for (u = 0; u < ENGINE_TAG_ILWALID; u++)
        {
            engine_disc_info engineDiscInfo = intrEnumTable.discInfo[u];

            char name0[INTR_NAME_MAXLEN];
            char name1[INTR_NAME_MAXLEN];
            char retr0[INTR_NAME_MAXLEN];

            if (engineDiscInfo.count == 0 || !engineDiscInfo.bValid)
            {
                continue;
            }

             if ((engineDiscInfo.count > 1) && (engineDiscInfo.numArgs != 0))
            {
                assert(engineDiscInfo.stride != 0);
            }

            for (i = 0; i < engineDiscInfo.count; i++)
            {
                if (engineDiscInfo.numArgs == 0)
                {
                    // All engines report to the same INTR_CTRL (only GR does this)
                    snprintf(name0, sizeof(name0), "%s%d", engineDiscInfo.name, i);
                    snprintf(name1, sizeof(name1), "%s%d_notification", engineDiscInfo.name, i);
                    snprintf(retr0, sizeof(retr0), "%s", engineDiscInfo.retriggerName);
                }
                else if (engineDiscInfo.numArgs == 1)
                {
                    snprintf(name0, sizeof(name0), "%s%d", engineDiscInfo.name, i);
                    snprintf(name1, sizeof(name1), "%s%d_notification", engineDiscInfo.name, i);
                    snprintf(retr0, sizeof(retr0), "%s(%d)", engineDiscInfo.retriggerName, i);
                }
                else if (engineDiscInfo.numArgs == 2)
                {
                    snprintf(name0, sizeof(name0), "%s%d", engineDiscInfo.name, i);
                    snprintf(name1, sizeof(name1), "%s%d_notification", engineDiscInfo.name, i);
                    snprintf(retr0, sizeof(retr0), "%s(%d, 0)", engineDiscInfo.retriggerName, i);
                }

                intrRegisterDiscWithRetrigger(engineDiscInfo.ctrlReg             + engineDiscInfo.stride * i, name0, engineDiscInfo.retriggerReg + engineDiscInfo.stride * i, retr0);
                // Retrigger is not used (if present at all) for notification reg because there's no HW state to service
                intrRegisterDiscWithRetrigger(engineDiscInfo.ctrlNotificationReg + engineDiscInfo.stride * i, name1, 0, "N/A");
            }
        }
    }
    else
    {
        dprintf("Failed to get device info\n");
    }

}

/**
 * @brief   Retrieves interrupts IDs and updates the interrupt table
 */
LwBool intrUpdateAll(void)
{
    LwU32 i;
    LwBool bSuccess = LW_TRUE;

    for (i = 0; i < intrEnumTable.discCtr; i++)
    {
        LwBool bUpdateSuccess;
        bUpdateSuccess = intrUpdateDisc(&intrEnumTable.intrDisc[i]);
        if (!bUpdateSuccess)
            bSuccess = LW_FALSE;
    }

    intrEnumTable.bInterruptTableInit = LW_FALSE;
    return bSuccess;
}

/**
 * @brief   Prints details for interrupt with GPU Vector ID idx
 * @param   idx     GPU vector index of interrupt
 * @param   bVerb   Set to true for more detailed output
 * @param   gfid    Target gfid, used only when verbose
 * @param   bGsp    Whether or not querying gsp tree
 */
static void intrPrint(LwU32 idx, LwBool bVerb, LwU32 gfid, LwBool bGsp)
{
    intr_type *intr = &intrEnumTable.interrupts[idx];

    if (!intr->bFound && ((bGsp && !intr->bGspTree) ||
                        (gfid > 0 && !intr->bVirtualized) ||
                        (!intr->bTag)))
    {
        return;
    }

    if ((!intr->bTag) ||
        (bGsp && !intr->bGspTree) ||
        (gfid > 0 && !intr->bVirtualized))
    {
        dprintf("|%-30s|%-7d|%-7s", "UNKNOWN", idx, "?");
        if (!bGsp)
        {
            dprintf("| CPU_INTR_LEAF(%02d) => (%02d:%02d) |",
                INTR_LEAF_IDX(idx), INTR_LEAF_BIT_IDX(idx), INTR_LEAF_BIT_IDX(idx));
            dprintf(" CPU_INTR_TOP(%02d) =>  (%02d:%02d) |",
                INTR_TOP_IDX(idx), INTR_TOP_BIT_IDX(idx), INTR_TOP_BIT_IDX(idx));
        }
        else
        {
            dprintf("| GSP_INTR_LEAF(%02d) => (%02d:%02d) |",
                INTR_LEAF_IDX(idx), INTR_LEAF_BIT_IDX(idx), INTR_LEAF_BIT_IDX(idx));
            dprintf(" GSP_INTR_TOP(%02d) =>  (%02d:%02d) |",
                INTR_TOP_IDX(idx), INTR_TOP_BIT_IDX(idx), INTR_TOP_BIT_IDX(idx));
        }

        if (bVerb)
        {
            dprintf("%-40s", "?");
            dprintf("|%-15s", "");

            dprintf("|%-13s", INTR_LEAF_ENABLED(idx, gfid, bGsp) ? "Y":"N");
            dprintf("|%-13s|", INTR_TOP_ENABLED(idx, gfid, bGsp) ? "Y":"N");
        }

        dprintf("\n");
        intr->bFound = LW_FALSE;
        return;
    }

    while (intr != NULL && intr->bTag)
    {
        dprintf("|%-30s|%-7d|%-7s", intr->name, intr->gpuIV, intr->bPulse ? "Y":"N");
        if (!bGsp)
        {
            dprintf("| CPU_INTR_LEAF(%02d) => (%02d:%02d) |",
                    intr->leafReg, intr->leafBit, intr->leafBit);
            dprintf(" CPU_INTR_TOP(%02d) =>  (%02d:%02d) |",
                    intr->topReg, intr->topBit, intr->topBit);
        }
        else
        {
            dprintf("| GSP_INTR_LEAF(%02d) => (%02d:%02d) |",
                    intr->leafReg, intr->leafBit, intr->leafBit);
            dprintf(" GSP_INTR_TOP(%02d) =>  (%02d:%02d) |",
                    intr->topReg, intr->topBit, intr->topBit);
        }

        if (bVerb)
        {
            if (intr->retriggerAddress)
            {
                dprintf("%-40s",
                        intr->retriggerName ? intr->retriggerName : "");
                dprintf("|0x%-13x", intr->retriggerAddress);
            }
            else
            {
                dprintf("%-40s", "N/A");
                dprintf("|%-15s", "");
            }
            dprintf("|%-13s",
                    INTR_LEAF_ENABLED(intr->gpuIV, gfid, bGsp) ? "Y":"N");
            dprintf("|%-13s|",
                    INTR_TOP_ENABLED(intr->gpuIV, gfid, bGsp) ? "Y":"N");
        }
        dprintf("\n");

        intr = intr->next;
    }

    return;
}

/**
 * @brief   Prints interrupt list in readable format
 * @param   Array of indices to print in interrupts table
 */
void intrPrintList_GA100(LwBool bVerbose, LwU32 gfid, LwBool bGsp)
{
    LwU32 i;
    intrUpdateAll();

    dprintf("========\n");
    if (!bGsp)
        dprintf("  CPU %d\n", gfid);
    else
        dprintf("   GSP\n");
    dprintf("========\n");

    dprintf("|%-30s|%-7s|%-7s", "Unit", "Vect #", "Pulse");
    if (!bGsp)
    {
        dprintf("|%-30s", "LW_CTRL_CPU_INTR_LEAF");
        dprintf("|%-30s", "LW_CTRL_CPU_INTR_TOP");
    }
    else
    {
        dprintf("|%-30s", "LW_CTRL_GSP_INTR_LEAF");
        dprintf("|%-30s", "LW_CTRL_GSP_INTR_TOP");
    }
    if (bVerbose)
    {
        dprintf("|%-40s", "Retrigger Name");
        dprintf("|%-15s", "Retrigger Addr");

        dprintf("|%-13s", "Leaf Enabled");
        dprintf("|%-13s", "Top Enabled");
    }
    dprintf("|\n");

    dprintf("|%-30s|%-7s|%-7s", "----", "------", "-----");
    dprintf("|%-30s", "---------------------");
    dprintf("|%-30s", "--------------------");

    if (bVerbose)
    {
        dprintf("|%-40s", "--------------");
        dprintf("|%-15s", "--------------");

        dprintf("|%-13s", "------------");
        dprintf("|%-13s|", "-----------");
    }
    dprintf("\n");

    for (i = 0; i < INTR_MAX; i++)
    {
        LwU32 idx = i;

        if (INTR_VECT_LEAF_SET(idx, gfid, bGsp) || !bVerbose)
        {
            if (bVerbose)
                intrEnumTable.interrupts[idx].bFound = LW_TRUE;
            intrPrint(idx, bVerbose, gfid, bGsp);
        }

        if (intrEnumTable.interrupts[idx].bDiscovery)
        {
            intrEnumTable.interrupts[idx].bDiscovery = LW_FALSE;
            intrEnumTable.interrupts[idx].bTag = LW_FALSE;
        }
    }

    dprintf("\n");
    return;
}


void addEngineInfo
(
    LwU32 tag,
    char *name,
    LwU32 ctrlReg,
    char *ctrlRegName,
    LwU32 retriggerReg,
    char *retriggerName,
    LwU32 ctrlNotificationReg,
    char *ctrlNotificationRegName,
    LwU32  stride,
    LwU32  numArgs,
    LwBool bValid
)
{
    // TODO: ctrlRegName and ctrlNotificationRegName are not actually used anywhere
    engine_disc_info *info = &intrEnumTable.discInfo[tag];
    strcpy(info->name, name);
    info->ctrlReg = ctrlReg;
    strcpy(info->ctrlRegName, ctrlRegName);
    info->retriggerReg = retriggerReg;
    strcpy(info->retriggerName, retriggerName);
    info->ctrlNotificationReg = ctrlNotificationReg;
    strcpy(info->ctrlNotificationRegName, ctrlNotificationRegName);
    info->stride = stride;
    info->numArgs = numArgs;
    info->bValid = bValid;
}

void intrAddLwjpg_GA100(void)
{
    // Only one lwjpg on Ampere
   addEngineInfo(ENGINE_TAG_LWJPG, "lwjpg", LW_PLWJPG_FALCON_INTR_CTRL(0), "LW_PLWJPG_FALCON_INTR_CTRL", LW_PLWJPG_FALCON_INTR_RETRIGGER(0), "LW_PLWJPG_FALCON_INTR_RETRIGGER", LW_PLWJPG_FALCON_INTR_CTRL(1), "LW_PLWJPG_FALCON_INTR_CTRL", 0, 1, LW_TRUE);
}

/*
 * @brief   Initializes list of interrupts
 *          The table here mirrors mcGetStaticInterruptTable_GA100
 */
void intrInit_GA100()
{
    LwU32 i;
    const LwU32 lwdecStride = LW_PLWDEC_FALCON_INTR_CTRL(1,0) - LW_PLWDEC_FALCON_INTR_CTRL(0,0);
    const LwU32 ceStride    = LW_CE_LCE_INTR_CTRL(1) - LW_CE_LCE_INTR_CTRL(0);

    intr_info pStaticTable[] =
    {
        // GSP RM
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PFB_VECTOR, LW_FALSE, "fb", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_THERMAL_VECTOR, LW_FALSE, "thermal", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PMGR_VECTOR, LW_FALSE, "pmgr", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_DFD_VECTOR, LW_FALSE, "dfd", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PMU_VECTOR, LW_FALSE, "pmu", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_LTC_ALL_VECTOR, LW_FALSE, "ltc (all)", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PRIV_RING_VECTOR, LW_FALSE, "priv_ring", GSP_PRESENT, VIRTUAL_ABSENT},
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
