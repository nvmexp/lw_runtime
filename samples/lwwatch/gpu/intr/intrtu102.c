/* Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 *_LWRM_COPYRIGHT_END_
 */
#include <stdint.h>
#include <string.h>

#include "chip.h"
#include "deviceinfo.h"
#include "fifo.h"
#include "g_intr_private.h"
#include "hal.h"
#include "intr.h"
#include "intr_private.h"
#include "lwwatch.h"
#include "os.h"
#include "print.h"
#include "turing/tu102/dev_vm.h"
#include "turing/tu102/dev_ctrl.h"
#include "turing/tu102/dev_ce.h"
#include "turing/tu102/dev_fb.h"
#include "vgpu.h"

#define INTR_TOPS_IMPLEMENTED_PER_FN (LW_CTRL_CPU_INTR_TOP__SIZE_1/MAX_GFIDS)
#define INTR_PER_TOP (LW_CTRL_CPU_INTR_LEAF_ARRAY_SIZE_PER_FN*REG_NUM_BITS/INTR_TOPS_IMPLEMENTED_PER_FN)
#define INTR_PER_GSP_TOP (LW_CTRL_GSP_INTR_LEAF__SIZE_1*REG_NUM_BITS/LW_CTRL_GSP_INTR_TOP__SIZE_1)

// =============================== INIT FUNCS ============================== //

void intrRegister_TU102
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
    i->bTag = LW_TRUE;
    i->bFound = LW_FALSE;

    intrEnumTable.intrCtr++;
    return;
}

static void intrRegisterDisc(LwU32 gpuIV, LwBool bPulse, char *name,
                             LwBool bGspTree, LwBool bVirtualized)
{
    intr_type *i = &intrEnumTable.interrupts[gpuIV];

    if (i->bTag)
    {
        while (i->next != NULL)
        {
            i = i->next;
        }

        i->next = (intr_type *)calloc(1, sizeof(intr_type));
        if (i->next == NULL)
        {
            return;
        }

        i = i->next;
    }

    i->gpuIV = gpuIV;
    i->bPulse = bPulse;
    i->name = calloc(INTR_NAME_MAXLEN, sizeof(char));
    strncpy(i->name, name, INTR_NAME_MAXLEN);
    i->leafReg = INTR_LEAF_IDX(gpuIV);
    i->leafBit = INTR_LEAF_BIT_IDX(gpuIV);
    i->topReg = INTR_TOP_IDX(gpuIV);
    i->topBit = INTR_TOP_BIT_IDX(gpuIV);
    i->bGspTree = bGspTree;
    i->bVirtualized = bVirtualized;
    i->bTag = LW_TRUE;

    intrEnumTable.intrCtr++;
    return;
}

/**
 * @brief   Retrieves interrupt info from interrupt discovery table
 */
void intrInitDiscAll_TU102(void)
{
    LwU32 i, nostall, stall;

    if (LW_OK != pFifo[indexGpu].fifoGetDeviceInfo())
    {
        return;
    }

    nostall = GPU_REG_RD32(LW_CTRL_LEGACY_ENGINE_NONSTALL_INTR_BASE_VECTORID);
    stall = GPU_REG_RD32(LW_CTRL_LEGACY_ENGINE_STALL_INTR_BASE_VECTORID);

    for (i = 0; i < deviceInfo.enginesCount; i++)
    {
        LwU32 gpuIV;
        char name[INTR_NAME_MAXLEN];

        if (!deviceInfo.pEngines[i].bHostEng)
            continue;

        if (((deviceInfo.cfg.version >= 2) && deviceInfo.pEngines[i].bHostEng)
            || (deviceInfo.cfg.version < 2))
        {
            memset(name, 0, INTR_NAME_MAXLEN);
            strncpy(name, deviceInfo.pEngines[i].engineName,
                    INTR_NAME_MAXLEN - 1);
            gpuIV = deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_INTR];

            intrRegisterDisc(gpuIV + nostall, LW_TRUE, name, GSP_PRESENT, VIRTUAL_PRESENT);
            intrRegisterDisc(gpuIV + stall, LW_FALSE, name, GSP_PRESENT, VIRTUAL_PRESENT);
        }
    }

    return;
}

/**
 * @brief   Prints details for interrupt with GPU Vector ID idx
 * @param   idx     GPU vector index of interrupt
 * @param   gfid    Target gfid, used only when verbose
 */
static void intrPrint(LwU32 idx, LwS32 gfid, LwBool bVerbose, LwBool gsp)
{
    intr_type *intr = &intrEnumTable.interrupts[idx];

    if (!intr->bFound && ((gsp && !intr->bGspTree) ||
                        (gfid > 0 && !intr->bVirtualized) ||
                        (!intr->bTag)))
    {
        return;
    }

    if ((!intr->bTag) ||
        (gsp && !intr->bGspTree) ||
        (gfid > 0 && !intr->bVirtualized))
    {
        dprintf("|%-30s", "UNKNOWN");
        dprintf("|%-7d|%-7s", idx, "?");

        if (!gsp)
            dprintf("| CPU_INTR_LEAF(%02d) => (%02d:%02d) |",
                    INTR_LEAF_IDX(idx), INTR_LEAF_BIT_IDX(idx), INTR_LEAF_BIT_IDX(idx));
        else
            dprintf("| GPU_INTR_LEAF(%02d) => (%02d:%02d) |",
                    INTR_LEAF_IDX(idx), INTR_LEAF_BIT_IDX(idx), INTR_LEAF_BIT_IDX(idx));

        if (bVerbose)
        {
            dprintf("Leaf: %-5s|", INTR_LEAF_ENABLED(idx, gfid, gsp) ? "Y":"N");
        }
        dprintf("\n");

        dprintf("|%-30s|%-7s|%-7s", "", "", "");

        if (!gsp)
            dprintf("| CPU_INTR_TOP(%02d) =>  (%02d:%02d) |",
                    INTR_TOP_IDX(idx), INTR_TOP_BIT_IDX(idx), INTR_TOP_BIT_IDX(idx));
        else
            dprintf("| GPU_INTR_TOP(%02d) =>  (%02d:%02d) |",
                    INTR_TOP_IDX(idx), INTR_TOP_BIT_IDX(idx), INTR_TOP_BIT_IDX(idx));

        if (bVerbose)
        {
            dprintf("Top:  %-5s|", INTR_TOP_ENABLED(idx, gfid, gsp) ? "Y":"N");
        }
        dprintf("\n");
        intr->bFound = LW_FALSE;
        return;
    }

    while (intr != NULL && intr->bTag)
    {
        dprintf("|%-30s", intr->name);
        dprintf("|%-7d|%-7s", intr->gpuIV, (intr->bPulse ? "Y":"N"));

        if (!gsp)
            dprintf("| CPU_INTR_LEAF(%02d) => (%02d:%02d) |", intr->leafReg,
                    intr->leafBit, intr->leafBit);
        else
            dprintf("| GPU_INTR_LEAF(%02d) => (%02d:%02d) |", intr->leafReg,
                    intr->leafBit, intr->leafBit);

        if (bVerbose)
        {
            dprintf("Leaf: %-5s|", INTR_LEAF_ENABLED(intr->gpuIV, gfid, gsp) ? "Y":"N");
        }
        dprintf("\n");

        dprintf("|%-30s|%-7s|%-7s", "", "", "");
        if (!gsp)
            dprintf("| CPU_INTR_TOP(%02d) =>  (%02d:%02d) |", intr->topReg, intr->topBit,
                    intr->topBit);
        else
            dprintf("| GPU_INTR_TOP(%02d) =>  (%02d:%02d) |", intr->topReg, intr->topBit,
                    intr->topBit);

        if (bVerbose)
        {
            dprintf("Top:  %-5s|", INTR_TOP_ENABLED(intr->gpuIV, gfid, gsp) ? "Y":"N");
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
void intrPrintList_TU102(LwBool bVerbose, LwU32 gfid, LwBool gsp)
{
    LwU32 i;

    dprintf("|%-30s|%-7s|%-7s", "Unit", "Vect #", "Pulse");

    if (!gsp)
        dprintf("|%-30s|", "LW_CTRL_CPU");
    else
        dprintf("|%-30s|", "LW_CTRL_GSP");

    if (bVerbose)
    {
        dprintf("%-11s|", "Enabled");
    }
    dprintf("\n");

    dprintf("|%-30s|%-7s|%-7s", "----", "------", "-----");
    dprintf("|%-30s|", "-----------");

    if (bVerbose)
    {
        dprintf("%-11s|", "-------");
    }
    dprintf("\n");

    for (i = 0; i < INTR_MAX; i++)
    {
        LwU32 idx = i;

        if (INTR_VECT_LEAF_SET(idx, gfid, gsp) || !bVerbose)
        {
            intrPrint(idx, gfid, bVerbose, gsp);
        }
    }

    dprintf("\n");
    return;
}

/*
 * @brief   Initializes list of interrupts
 * @detail  The goal is to auto-generate this code eventually
 *          The table here mirrors mcGetStaticInterruptTable_TU102
 */
void intrInit_TU102()
{
    LwU32 i;

    intr_info pStaticTable[] =
    {
        // GSP RM
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PFIFO_INTR_0_VECTOR, LW_FALSE, "fifo (stall)", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PFIFO_INTR_1_VECTOR, LW_FALSE, "fifo (nonstall)", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PFB_VECTOR, LW_FALSE, "fb", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_THERMAL_VECTOR, LW_FALSE, "thermal", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PMGR_VECTOR, LW_FALSE, "pmgr", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_DFD_VECTOR, LW_FALSE, "dfd", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PMU_VECTOR, LW_FALSE, "pmu", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_LTC_ALL_VECTOR, LW_FALSE, "ltc (all)", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PDISP_VECTOR, LW_FALSE, "disp", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PRIV_RING_VECTOR, LW_FALSE, "priv_ring", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_HDACODEC_VECTOR, LW_FALSE, "hdacodec", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_PFB_PRI_MMU_INT_VECTOR_FAULT_NOTIFY_NON_REPLAYABLE, LW_TRUE, "mmu_nonreplayable_fault", GSP_PRESENT, VIRTUAL_PRESENT},
        {LW_PFB_PRI_MMU_INT_VECTOR_FAULT_ERROR_NON_REPLAYABLE, LW_TRUE, "mmu_nonreplayable_fault_error", GSP_PRESENT, VIRTUAL_PRESENT},
        {LW_PFB_PRI_MMU_INT_VECTOR_INFO_FAULT_VECTOR_INIT, LW_TRUE, "mmu_info_fault", GSP_PRESENT, VIRTUAL_PRESENT},
        {LW_PFB_PRI_MMU_INT_VECTOR_ECC_ERROR_VECTOR_INIT, LW_TRUE, "mmu_ecc_error", GSP_PRESENT, VIRTUAL_PRESENT},
        // lwlink interrupts are unused in Turing, but still have vectors
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_IOCTRL_INTR_0_VECTOR, LW_FALSE, "lwlink_stall (intr 0)", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_IOCTRL_INTR_1_VECTOR, LW_FALSE, "lwlink_nostall (intr 1)", GSP_PRESENT, VIRTUAL_ABSENT},

        // CPU RM aka Client RM
        {LW_PFB_PRI_MMU_INT_VECTOR_FAULT_NOTIFY_REPLAYABLE, LW_TRUE, "mmu_replayable_fault", GSP_PRESENT, VIRTUAL_PRESENT},
        {LW_PFB_PRI_MMU_INT_VECTOR_FAULT_ERROR_REPLAYABLE, LW_TRUE, "mmu_replayable_fault_error", GSP_PRESENT, VIRTUAL_PRESENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_HUB_ACCESS_CNTR_INTR_VECTOR, LW_TRUE, "access_counter", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_CTRL_SW_INTR_NOSTALL_VECTORID_VALUE_CONSTANT, LW_FALSE, "software", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_XVE_VECTOR, LW_FALSE, "bif", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PBUS_VECTOR, LW_FALSE, "bus", GSP_PRESENT, VIRTUAL_ABSENT},
        {LW_VIRTUAL_FUNCTION_PRIV_CPU_INTR_PTIMER_VECTOR, LW_TRUE, "timer", GSP_PRESENT, VIRTUAL_PRESENT},
        {LW_CTRL_CPU_DOORBELL_VECTORID_VALUE_CONSTANT, LW_TRUE, "cpu_doorbell", GSP_PRESENT, VIRTUAL_PRESENT},
        {LW_CTRL_VIRTUAL_INTR_VECTORID_VALUE_CONSTANT, LW_TRUE, "priv_doorbell", GSP_PRESENT, VIRTUAL_PRESENT},
        // GR_FECS_LOG has no interrupt
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

    pIntr[indexGpu].intrInitDiscAll();
}

/**
 * @brief   Sets an interrupt
 * @param   vect     gpu vector ID
 * @param   gfid     gpu function index
 */
void intrSetInterrupt_TU102(LwU32 vect, LwU32 gfid)
{
    if (vect >= INTR_MAX_VECTOR)
    {
        dprintf("Invalid gpu vector ID.\n");
        return;
    }
    if (gfid >= MAX_GFIDS)
    {
        dprintf("Invalid function ID.\n");
        return;
    }

    GPU_REG_WR32(LW_CTRL_CPU_INTR_LEAF_TRIGGER(gfid), vect);
    return;
}

/**
 * @brief Clears an interrupt
 * @param   vect    gpu vector ID
 * @param   func    gpu function index
 */
void intrClearInterrupt_TU102(LwU32 vect, LwU32 gfid)
{
    LwU32 leaf, leafBits, leafBaseIdx;

    if (vect >= INTR_MAX_VECTOR)
    {
        dprintf("Invalid gpu vector ID.\n");
        return;
    }
    if (gfid >= MAX_GFIDS)
    {
        dprintf("Invalid function ID.\n");
        return;
    }

    leafBaseIdx = gfid * LW_CTRL_CPU_INTR_LEAF_ARRAY_SIZE_PER_FN;
    leaf = LW_CTRL_CPU_INTR_LEAF(leafBaseIdx + INTR_LEAF_IDX(vect));
    leafBits = 0x1 << INTR_LEAF_BIT_IDX(vect);

    GPU_REG_WR32(leaf, leafBits);
    return;
}

/**
 * @brief   Sets enable for interrupt
 * @detail  Sets both leaf and top enables
 * @param   vect    gpu vector ID
 * @param   gfid    gpu function index
 * @param   bGsp    whether or not target is gsp
 */
void intrEnableInterrupt_TU102(LwU32 vect, LwU32 gfid, LwBool bGsp)
{
    LwU32 leafEnableSet, topEnableSet;

    if (vect >= INTR_MAX_VECTOR)
    {
        dprintf("Invalid gpu vector ID.\n");
        return;
    }
    if (gfid >= MAX_GFIDS && !bGsp)
    {
        dprintf("Invalid function ID.\n");
        return;
    }

    leafEnableSet = INTR_LEAF_ENABLE_SET_ADDR(vect, gfid, bGsp);
    GPU_REG_WR32(leafEnableSet, 0x1 << INTR_LEAF_BIT_IDX(vect));

    topEnableSet = INTR_TOP_ENABLE_SET_ADDR(vect, gfid, bGsp);
    GPU_REG_WR32(topEnableSet, 0x1 << INTR_TOP_BIT_IDX(vect));

    return;
}

/**
 * @brief   Sets disable for interrupt
 * @detail  Clears only leaf enable
 * @param   vect    gpu vector ID
 * @param   gfid    gpu function index
 * @param   bGsp    whether or not target is gsp
 */
void intrDisableInterrupt_TU102(LwU32 vect, LwU32 gfid, LwBool bGsp)
{
    LwU32 leafEnableClear;

    if (vect >= INTR_MAX_VECTOR)
    {
        dprintf("Invalid gpu vector ID.\n");
        return;
    }
    if (gfid >= MAX_GFIDS)
    {
        dprintf("Invalid function ID.\n");
        return;
    }

    leafEnableClear = INTR_LEAF_ENABLE_CLEAR_ADDR(vect, gfid, bGsp);
    GPU_REG_WR32(leafEnableClear, 0x1 << INTR_LEAF_BIT_IDX(vect));

    return;
}

/**
 * @brief   Dumps leaf and top register values for a function
 * @param   gfid    Target function ID
 * @param   bGsp    whether or not target is gsp tree
 */
void intrDumpRawRegs_TU102(LwU32 gfid, LwBool bGsp)
{
    LwU32 ctr, reg, val, en, idx;

    dprintf("BEGINNING REGISTER DUMP\n");
    dprintf("-----------------------\n");

    if (gfid >= MAX_GFIDS && !bGsp)
    {
        dprintf("Invalid function ID.\n");
        return;
    }

    if (!bGsp)
    {
        dprintf("\tCPU_INTR_LEAF(i)\t|\tVALUE\t\t|\tCPU_INTR_LEAF_EN(i)\n");
        for (ctr = 0; ctr < LW_CTRL_CPU_INTR_LEAF_ARRAY_SIZE_PER_FN; ctr++)
        {
            idx = LW_CTRL_CPU_INTR_LEAF_ARRAY_SIZE_PER_FN * gfid + ctr;
            reg = LW_CTRL_CPU_INTR_LEAF(idx);
            en = GPU_REG_RD32(LW_CTRL_CPU_INTR_LEAF_EN_SET(idx));
            val = GPU_REG_RD32(reg);
            dprintf("\t%16d\t|\t0x%08x\t|\t0x%08x\n",
                    ctr, (unsigned int)val, (unsigned int)en);
        }

        dprintf("\tCPU_INTR_TOP(i) \t|\tVALUE\t\t|\tCPU_INTR_TOP_EN(i)\n");
        for (ctr = 0; ctr < INTR_TOPS_IMPLEMENTED_PER_FN; ctr++)
        {
            idx = INTR_TOPS_IMPLEMENTED_PER_FN * gfid + ctr;
            reg = LW_CTRL_CPU_INTR_TOP(idx);
            en = GPU_REG_RD32(LW_CTRL_CPU_INTR_TOP_EN_SET(idx));
            val = GPU_REG_RD32(reg);
            dprintf("\t%16d\t|\t0x%08x\t|\t0x%08x\n",
                    ctr, (unsigned int)val, (unsigned int)en);
        }
    }
    else
    {
        dprintf("\tGSP_INTR_LEAF(i)\t|\tVALUE\t\t|\tGSP_INTR_LEAF_EN(i)\n");
        for (ctr = 0; ctr < LW_CTRL_CPU_INTR_LEAF_ARRAY_SIZE_PER_FN; ctr++)
        {
            en = GPU_REG_RD32(LW_CTRL_GSP_INTR_LEAF_EN_SET(ctr));
            val = GPU_REG_RD32(LW_CTRL_GSP_INTR_LEAF(ctr));
            dprintf("\t%16d\t|\t0x%08x\t|\t0x%08x\n",
                    ctr, (unsigned int)val, (unsigned int)en);
        }

        dprintf("\tGSP_INTR_TOP(i)\t|\tVALUE\n");
        for (ctr = 0; ctr < INTR_TOPS_IMPLEMENTED_PER_FN; ctr++)
        {
            val = GPU_REG_RD32(LW_CTRL_GSP_INTR_TOP(ctr));
            dprintf("\t%16d\t|\t0x%08x\n",
                ctr, (unsigned int)val);
        }
    }

    return;
}
