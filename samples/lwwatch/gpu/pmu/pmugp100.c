/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension for PMU
// pmugp100.c
//
//*****************************************************

//
// includes
//
#include "pmu.h"

#include "pascal/gp100/dev_master.h"
#include "pascal/gp100/dev_lw_xve.h"

#include "g_pmu_private.h"     // (rmconfig)  implementation prototypes

const char *
pmuUcodeName_GP100()
{
    return "g_c85b6_gp100";
}

//////////////////////////////////////////////////////////////////////////////
// PMU Sanity Tests
//////////////////////////////////////////////////////////////////////////////

// CFG Space
#define CFG_RD32(a)   PMU_REG_RD32(DRF_BASE(LW_PCFG) + a) 
#define CFG_WR32(a,b) PMU_REG_WR32(DRF_BASE(LW_PCFG) + a, b) 
// Prototypes
LW_STATUS pmuSanityTest_Latency_GP100      (LwU32, char *);

static PmuSanityTestEntry PmuSanityTests_GP100[] =
{
    // Check Image
    {
        pmuSanityTest_CheckImage_GK104,
        PMU_TEST_PROD_UCODE,
        "Check Image"
    },
    // Reset Test
    {
        pmuSanityTest_Reset_GK104,
        PMU_TEST_DESTRUCTIVE,
        "Reset Test"
    },
    // Low-Latency Test
    {
        pmuSanityTest_Latency_GP100,
        PMU_TEST_AUTO,
        "Low-Latency Test"
    },
    // Mutex and ID Generator Test
    {
        pmuSanityTest_MutexIdGen_GK104,
        PMU_TEST_AUTO,
        "Mutex and ID Generator Test"
    },
    // PBI Test
    {
        pmuSanityTest_PBI_GK104,
        PMU_TEST_AUTO,
        "PBI Test"
    },
    // BAR0 Master Test
    {
        pmuSanityTest_Bar0Master_GK104,
        PMU_TEST_AUTO,
        "Bar0 Master Test"
    },
    // BAR0 FECS Test
    {
        pmuSanityTest_Bar0FECS_GK104,
        PMU_TEST_AUTO,
        "Bar0 FECS Test"
    },
    // PBI Interface Test
    {
        pmuSanityTest_PbiInterface_GK104,
        PMU_TEST_AUTO|PMU_TEST_PROD_UCODE,
        "PBI Interface Test"
    },
    // PMU GPTIMER Test
    {
        pmuSanityTest_GPTMR_GK104,
        PMU_TEST_VERIF_UCODE|PMU_TEST_DESTRUCTIVE,
        "GPTMR Test"
    },
    // PMU Vblank IO Test
    {
        pmuSanityTest_Vblank_GK104,
        PMU_TEST_VERIF_UCODE|PMU_TEST_DESTRUCTIVE,
        "VBLANK1 (head0) Test"
    },
    // PMU Display Scanline IO Test
    {
        pmuSanityTest_ScanlineIO_GK104,
        PMU_TEST_VERIF_UCODE|PMU_TEST_DESTRUCTIVE,
        "RG/DMI Scanline I/O Test"
    },
    // PMU Display Scanline Interrupt Test
    {
        pmuSanityTest_ScanlineIntr_GK104,
        PMU_TEST_VERIF_UCODE|PMU_TEST_DESTRUCTIVE,
        "RG/DMI Scanline Interrupt Test"
    },
};

#define PMU_SANITY_TEST_NUM (sizeof(PmuSanityTests_GP100) / sizeof(PmuSanityTestEntry))

#define BAILOUT(cond, label) \
    do {status = cond; goto label;} while(0)

/*!
 *  @returns test table
 */
void *
pmuSanityTestGetEntryTable_GP100()
{
    return (void *) PmuSanityTests_GP100;
}

//
// PMU Sanity Test Cases
//

// Low-Latency Test
LW_STATUS pmuSanityTest_Latency_GP100(LwU32 verbose, char *arg)
{
    LwU32 intrMskSave;
    LwU32 reg32, intr;
    LW_STATUS status      = LW_OK;
    LwU32 xvePrivMisc = 0;
    LwU32 xveBase;

    // Find bit position of xve interrupt in LW_PMC_INTR
    xveBase = DRF_BASE(LW_PMC_INTR_XVE);

    // Save the current LW_XVE_PRIV_MISC_1
    xvePrivMisc = CFG_RD32(LW_XVE_PRIV_MISC_1);

    // Force route MSGBOX interrupt to HOST
    reg32 = FLD_SET_DRF(_XVE, _PRIV_MISC_1, _CYA_ROUTE_MSGBOX_CMD_INTR_TO_PMU, _DISABLE, xvePrivMisc);
    CFG_WR32(LW_XVE_PRIV_MISC_1, reg32);

    // Save LW_PMC_INTR(0)
    intrMskSave = PMU_REG_RD32(LW_PMC_INTR_EN(0));

    PMU_LOG(VB0, "<<Low-Latency Test>>\n");

    // Enable XVE interrupts for INTR(0)
    PMU_LOG(VB0, "Enable XVE interrupt\n");
    reg32 = FLD_IDX_SET_DRF(_PMC, _INTR_EN, _SET_DEVICE, xveBase, _SET, intrMskSave);
    PMU_REG_WR32(LW_PMC_INTR_EN_SET(0), reg32);

    // Enable XVE MSGBOX interrupts
    PMU_LOG(VB0, "Enable XVE MSGBOX interrupts.\n");
    reg32 = CFG_RD32(LW_XVE_PRIV_INTR_EN);
    PMU_LOG(VB2, "LW_XVE_PRIV_INTR_EN[%08x]\n", reg32);
    if (FLD_TEST_DRF(_XVE, _PRIV_INTR_EN, _MSGBOX_INTERRUPT, _DISABLED, reg32))
    {
        reg32 = FLD_SET_DRF(_XVE, _PRIV_INTR_EN, _MSGBOX_INTERRUPT, _ENABLED, reg32);
        CFG_WR32(LW_XVE_PRIV_INTR_EN, reg32);
        PMU_LOG(VB2, "Disabled previously, enabling MSGBOX(%08x)\n", reg32);
    }

    // Make sure no MSGBOX interrupt pending.
    PMU_LOG(VB0, "Check there's no pending MSGBOX interrupt pending.\n");
    reg32 = CFG_RD32(LW_XVE_PRIV_INTR);
    PMU_LOG(VB2, "LW_XVE_PRIV_INTR[%08x]\n", reg32);
    if (FLD_TEST_DRF(_XVE, _PRIV_INTR, _MSGBOX_INTERRUPT, _PENDING, reg32))
    {
        reg32 = FLD_SET_DRF(_XVE, _PRIV_INTR, _MSGBOX_INTERRUPT, _NOT_PENDING, reg32);
        CFG_WR32(LW_XVE_PRIV_INTR, reg32);
        PMU_LOG(VB0, "Pending MSGBOX interrupt. Clearing it.\n");
    }

    // Write command and generate interrupt
    PMU_LOG(VB0, "Write command and generate interrupt.\n");
    CFG_WR32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_COMMAND, 0x80100000);

    // Check XVE_MSGBOX_INTERRUPT pending
    PMU_LOG(VB0, "Check XVE_MSGBOX_INTERRUPT pending.\n");
    reg32 = CFG_RD32(LW_XVE_PRIV_INTR);
    if (FLD_TEST_DRF(_XVE, _PRIV_INTR, _MSGBOX_INTERRUPT, _NOT_PENDING, reg32))
    {
        PMU_LOG(VB0, "XVE_MSGBOX_INTERRUPT NOT PENDING.\n");
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    PMU_LOG(VB0, "XVE_MSGBOX_INTERRUPT PENDING. (OK)\n");

    // Check pending on INTR(0)
    PMU_LOG(VB0, "Check pending");
    intr = PMU_REG_RD32(LW_PMC_INTR(0));

    if (!(FLD_TEST_DRF(_PMC, _INTR, _XVE, _PENDING, intr)))
    {
        PMU_LOG(VB0, "FAILED. intr[%08x]\n", intr);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    PMU_LOG(VB1, "PASSED. intr[%08x]\n", intr);

quit:

    // Restore XVE_PRIV_MISC_1
    CFG_WR32(LW_XVE_PRIV_MISC_1, xvePrivMisc);

    // Clear all XVE pending interrupts
    reg32 = CFG_RD32(LW_XVE_PRIV_INTR);
    if (reg32)
    {
        CFG_WR32(LW_XVE_PRIV_INTR, reg32);
        reg32 = CFG_RD32(LW_XVE_PRIV_INTR);
        if (reg32)
        {
            PMU_LOG(VB0, "LW_XVE_PRIV_INTR not cleared. (%08x)\n", reg32);
            status = LW_ERR_GENERIC;
        }
    }
    // Restore LW_PMC_INTR_MSK
    PMU_REG_WR32(LW_PMC_INTR_EN_CLEAR(0), 0xFFFFFFFF);
    PMU_REG_WR32(LW_PMC_INTR_EN_SET(0), intrMskSave);
    return status;
}

/*!
 *  @returns the number of PMU sanity tests available.
 */
LW_STATUS 
pmuSanityTestGetNum_GP100
(
    void
)
{
    return PMU_SANITY_TEST_NUM;
}

