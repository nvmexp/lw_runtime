/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
 
//*****************************************************
//
// DPU Hal Functions
// dpu0201.c
//
//*****************************************************


/* ------------------------ Includes --------------------------------------- */
#include "dpu.h"
#include "dpu/v02_01/dev_disp_falcon.h"
#include "kepler/gk104/dev_falcon_v4.h"
#include "kepler/gk104/dev_bus.h"
#include "kepler/gk104/dev_master.h"
#include "kepler/gk104/dev_lw_xve.h"
#include "dpu/v02_01/dev_disp_falcon.h"
#include "g_dpu_private.h"     // (rmconfig)  implementation prototypes


/* ------------------------ Defines ---------------------------------------- */
/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Globals ---------------------------------------- */
const static FLCN_ENGINE_IFACES flcnEngineIfaces_dpu0201 =
{
    dpuGetFalconCoreIFace_v02_01,               // flcnEngGetCoreIFace
    dpuGetFalconBase_v02_01,                    // flcnEngGetFalconBase
    dpuGetEngineName,                           // flcnEngGetEngineName
    dpuUcodeName_v02_01,                        // flcnEngUcodeName
    dpuGetSymFilePath,                          // flcnEngGetSymFilePath
    dpuQueueGetNum_v02_01,                      // flcnEngQueueGetNum
    dpuQueueRead_v02_01,                        // flcnEngQueueRead
    dpuGetDmemAccessPort,                       // flcnEngGetDmemAccessPort
    dpuIsDmemRangeAccessible_STUB,              // flcnEngIsDmemRangeAccessible
    dpuEmemGetOffsetInDmemVaSpace_STUB,         // flcnEngEmemGetOffsetInDmemVaSpace
    dpuEmemGetSize_STUB,                        // flcnEngEmemGetSize
    dpuEmemGetNumPorts_STUB,                    // flcnEngEmemGetNumPorts
    dpuEmemRead_STUB,                           // flcnEngEmemRead
    dpuEmemWrite_STUB,                          // flcnEngEmemWrite
};  // falconEngineIfaces_dpu


/* ------------------------ Functions -------------------------------------- */
/*!
 * @return The falcon engine interface FLCN_ENGINE_IFACES*
 */
const FLCN_ENGINE_IFACES *
dpuGetFalconEngineIFace_v02_01()
{
    return &flcnEngineIfaces_dpu0201;
}

/*!
 * Return Ucode file name
 *
 * @return Ucode file name
 */
const char*
dpuUcodeName_v02_01()
{
    return "g_dpuuc0201";
}

/*!
 * Return Queue numbers of Command Queue
 */
LwU32 
dpuGetCmdQNum_v02_01()
{
    return LW_PDISP_FALCON_CMDQ_HEAD__SIZE_1;
}

#define CFG_RD32(a)   GPU_REG_RD32(0x88000 + a)
#define CFG_WR32(a,b) GPU_REG_WR32(0x88000 + a, b)

static DpuSanityTestEntry* DpuSanityTests = NULL;

static DpuSanityTestEntry DpuSanityTests_v02_01[] =
{
    // Reset Test
    {
        dpuSanityTest_Reset_v02_01,
        DPU_TEST_DESTRUCTIVE,
        "Reset Test"
    },
    // Low-Latency Test
    {
        dpuSanityTest_Latency_v02_01,
        DPU_TEST_AUTO,
        "Low-Latency Test"
    },
    // DPU GPTIMER Test
    {
        dpuSanityTest_GPTMR_v02_01,
        DPU_TEST_VERIF_UCODE|DPU_TEST_DESTRUCTIVE,
        "GPTMR Test"
    },
};

#define DPU_SANITY_TEST_NUM (sizeof(DpuSanityTests_v02_01) / sizeof(DpuSanityTestEntry))

#define DPU_SANITY_CHECK_TEST_ENTRY \
    do\
    {\
        if (DpuSanityTests == NULL)\
        {\
            DpuSanityTests = (DpuSanityTestEntry *) \
                pDpu[indexGpu].dpuSanityTestGetEntryTable();\
        }\
    } while(0)

#define BAILOUT(cond, label) \
    do {status = cond; goto label;} while(0)

#define MBOX_CMD            (0)
#define MBOX_CMD_INIT       (0)
#define MBOX_CMD_START      (1)
#define MBOX_CMD_STOP       (2)
#define MBOX_CMD_PING       (3)
#define MBOX_CMD_PWRCLK     (4)
#define MBOX_CMD_RST        (5)
#define MBOX_CMD_BLANK      (6)
#define MBOX_CMD_SETPWRCLK  (7)
#define MBOX_CMD_PERFTEST   (8)
#define MBOX_CMD_DWCF       (9)
#define MBOX_CMD_SCANLINE   (10)
#define MBOX_CMD_BLANK_VBLANK1  (0)
#define MBOX_CMD_BLANK_HBLANK1  (1)
#define MBOX_CMD_BLANK_VBLANK2  (2)
#define MBOX_CMD_BLANK_HBLANK2  (3)

#define MBOX_DATA_0    (1)
#define MBOX_DATA      MBOX_DATA_0
#define MBOX_DATA_1    (2)
#define MBOX_PTMR      MBOX_DATA_1
#define MBOX_DATA_2    (3)
#define MBOX_PTMR_LO   MBOX_DATA_2

#define EMULATION
#ifdef EMULATION
#define CMD_DELAY osPerfDelay(100 * 1000)
#else
#define CMD_DELAY osPerfDelay(5000)
#endif

//----------------------------------------------------------------------
// DPU Check if gptimer_*.bin is loaded
//----------------------------------------------------------------------
BOOL dpuCheckVerifUcode(LwU32 verbose)
{
    LwU32 reg32;

    // Ping iotest.bin
    GPU_REG_WR32(LW_PDISP_FALCON_MAILBOX0, MBOX_CMD_STOP);
    CMD_DELAY;
    GPU_REG_WR32(LW_PDISP_FALCON_MAILBOX0, MBOX_CMD_RST);
    CMD_DELAY;
    GPU_REG_WR32(LW_PDISP_FALCON_MAILBOX0, MBOX_CMD_PING);

    // delay
    CMD_DELAY;
    reg32 = GPU_REG_RD32(LW_PDISP_FALCON_MAILBOX1);
    if (reg32 != 0xface0000)
    {
        DPU_LOG(VB0, "MAILBOX1 : %x\n", reg32);
        DPU_LOG(VB0, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        DPU_LOG(VB0, "!!                                                                                !!\n");
        DPU_LOG(VB0, "!! gptimer_*.bin is not running. gptimer_*.bin must be loaded and running for this  !!\n");
        DPU_LOG(VB0, "!! test. iotest_*.bin is available at                                             !!\n");
        DPU_LOG(VB0, "!! //sw/dev/gpu_drv/chips_a/tools/restricted/pmu/testapps/bin/gptimer_*.bin       !!\n");
        DPU_LOG(VB0, "!! * Use !dpuqboot to load the binary.                                            !!\n");
        DPU_LOG(VB0, "!!                                                                                !!\n");
        DPU_LOG(VB0, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        return FALSE;
    }
    return TRUE;
}

//
// DPU Sanity Test Cases
//

//----------------------------------------------------------------------
// Reset Test
//----------------------------------------------------------------------
LW_STATUS dpuSanityTest_Reset_v02_01(LwU32 verbose, char *arg)
{
    LwU32 reg32;
    DPU_LOG(VB0, ("<<Reset Test>>\n"));

    //------------------------------------------------------------------
    // Step 1 : clear all PBUS pending interrupts
    //------------------------------------------------------------------
    reg32 = GPU_REG_RD32(LW_PBUS_INTR_0);
    DPU_LOG(VB2, "checking pending intr LW_PBUS_INTR0 -> %08x\n", reg32);
    if (reg32)
    {
        DPU_LOG(VB2, "INTR pending -> clearing\n");
        GPU_REG_WR32(LW_PBUS_INTR_0, reg32);
    }

    //------------------------------------------------------------------
    // Step 2 : Enable DISP in LW_PMC_ENABLE
    //------------------------------------------------------------------
    reg32 = GPU_REG_RD32(LW_PMC_ENABLE);
    DPU_LOG(VB0, "reading PMC_ENABLE (%08x)\n", reg32);

    if (FLD_TEST_DRF(_PMC, _ENABLE, _PDISP, _DISABLED, reg32))
    {
        DPU_LOG(VB1, "DISP is not enabled, we're enabling it explicitly\n");
        reg32 = FLD_SET_DRF(_PMC, _ENABLE, _PDISP, _ENABLED, reg32);
        GPU_REG_WR32(LW_PMC_ENABLE, reg32);
    }
    DPU_LOG(VB1, "DISP is enabled\n");

    //------------------------------------------------------------------
    // Step 3 : Write to MAILBOX0, Read to verify
    //------------------------------------------------------------------
    DPU_LOG(VB0, "Writing 0xfaceb00c to LW_PDISP_FALCON_MAILBOX0\n");
    GPU_REG_WR32(LW_PDISP_FALCON_MAILBOX0, 0xfaceb00c);

    // read DISP register (MAILBOX register)
    DPU_LOG(VB0, "Reading LW_PDISP_FALCON_MAILBOX0\n");
    reg32 = GPU_REG_RD32(LW_PDISP_FALCON_MAILBOX0);
    DPU_LOG(VB1, "MAILBOX0[%08x]\n", reg32);

    // verify
    if (0xfaceb00c != reg32)
    {
        DPU_LOG(VB0, "Value mismatch!! %08x != 0xfaceb00c.\n", reg32);
        return LW_ERR_GENERIC;
    }

    //------------------------------------------------------------------
    // Step 4 : Check if step 3 caused any errors
    //------------------------------------------------------------------
    DPU_LOG(VB0, "Checking if there's no PRI error.\n");

    reg32 = GPU_REG_RD32(LW_PBUS_INTR_0);
    DPU_LOG(VB1, "LW_PBUS_INTR_0 -> %08x\n", reg32);
    if (reg32)
    {
        DPU_LOG(VB0, "Unexpected interrupts pending, clearing...\n");
        GPU_REG_WR32(LW_PBUS_INTR_0, reg32);
        return LW_ERR_GENERIC;
    }
    DPU_LOG(VB0, "=> No PRI error. (OK)\n");

    //------------------------------------------------------------------
    // Step 5 : Try to read MAILBOX(0) after disabling DISP
    //------------------------------------------------------------------
    DPU_LOG(VB0, "Disabling PMU.\n");
    reg32 = GPU_REG_RD32(LW_PMC_ENABLE);
    reg32 = FLD_SET_DRF(_PMC, _ENABLE, _PDISP, _DISABLED, reg32);
    GPU_REG_WR32(LW_PMC_ENABLE, reg32);
    DPU_LOG(VB2, "LW_PMC_ENABLE[%08x]\n", GPU_REG_RD32(LW_PMC_ENABLE));
    DPU_LOG(VB0, "Reading MAILBOX0\n");
    reg32 = GPU_REG_RD32(LW_PDISP_FALCON_MAILBOX0);
    DPU_LOG(VB1, "LW_PDISP_FALCON_MAILBOX0[%08x] ", reg32);
    // error if MAILBOX0 returns 0xfaceb00c
    if (0xfaceb00c == reg32)
    {
        DPU_LOG(VB0, "MAILBOX0 returned %08x after the DPU is disabled.\n",
                      reg32);
        return LW_ERR_GENERIC;
    }

    // check for PRI errors. fail if none of them is pending
    DPU_LOG(VB0, "Verify that there are PRI errors.\n");
    reg32 = GPU_REG_RD32(LW_PBUS_INTR_0);
    if (FLD_TEST_DRF(_PBUS, _INTR_0, _PRI_SQUASH,  _NOT_PENDING, reg32) &&
        FLD_TEST_DRF(_PBUS, _INTR_0, _PRI_FECSERR, _NOT_PENDING, reg32) &&
        FLD_TEST_DRF(_PBUS, _INTR_0, _PRI_TIMEOUT, _NOT_PENDING, reg32))
    {
        DPU_LOG(VB0, "No PRI errors.\n");
        return LW_ERR_GENERIC;
    }
    DPU_LOG(VB1, "PRI Error pending => LW_PBUS_INTR_0[%08x]\n", reg32);
    // Clear pending interrupts
    DPU_LOG(VB1, "Clearing LW_PBUS_INTR_0\n");
    GPU_REG_WR32(LW_PBUS_INTR_0, reg32);

    //------------------------------------------------------------------
    // Step 6 : Try to read MAILBOX0 after renabling PDISP
    //------------------------------------------------------------------
    DPU_LOG(VB0, "Enable PDISP.\n");
    reg32 = GPU_REG_RD32(LW_PMC_ENABLE);
    reg32 = FLD_SET_DRF(_PMC, _ENABLE, _PDISP, _ENABLED, reg32);
    GPU_REG_WR32(LW_PMC_ENABLE, reg32);

    // read MAILBOX(0) and verify it's 0.
    DPU_LOG(VB0, "Read MAILBOX(0) It should be 0 after reset.\n");
    reg32 = GPU_REG_RD32(LW_PDISP_FALCON_MAILBOX0);
    if (0 != reg32)
    {
        DPU_LOG(VB0, "MAILBOX(0)[%08x] != 0\n", reg32);
        return LW_ERR_GENERIC;
    }

    DPU_LOG(VB0, "Write/Read 'deadface' to MAILBOX0.\n");
    GPU_REG_WR32(LW_PDISP_FALCON_MAILBOX0, 0xdeadface);
    reg32 = GPU_REG_RD32(LW_PDISP_FALCON_MAILBOX0);
    if (0xdeadface != reg32)
    {
        DPU_LOG(VB0, "ERROR: 'deadface' != MAILBOX(0).\n");
        return LW_ERR_GENERIC;
    }
    DPU_LOG(VB1, "0xdeadface written OK.\n");

    // write 0 to MAILBOX(0)
    GPU_REG_WR32(LW_PDISP_FALCON_MAILBOX0, 0);
    return LW_OK;
}

//----------------------------------------------------------------------
// Low-Latency Test
//----------------------------------------------------------------------
LW_STATUS dpuSanityTest_Latency_v02_01(LwU32 verbose, char *arg)
{
    LwU32 intrMsk0Save, intrMsk2Save;
    LwU32 reg32, intr0, intr2;
    LW_STATUS status = LW_OK;

    // Save LW_PMC_INTR_MSK_0/2
    intrMsk0Save = GPU_REG_RD32(LW_PMC_INTR_MSK_0);
    intrMsk2Save = GPU_REG_RD32(LW_PMC_INTR_MSK_2);

    DPU_LOG(VB0, "<<Low-Latency Test>>\n");

    // Enable XVE interrupts for INTR_0, disable for INTR_2
    DPU_LOG(VB0, "Enable XVE interrupts for INTR_0, disable for INTR_2\n");
    reg32 = FLD_SET_DRF(_PMC, _INTR_MSK_0, _XVE, _ENABLED,  intrMsk0Save);
    GPU_REG_WR32(LW_PMC_INTR_MSK_0, reg32);
    reg32 = FLD_SET_DRF(_PMC, _INTR_MSK_2, _XVE, _DISABLED, intrMsk2Save);
    GPU_REG_WR32(LW_PMC_INTR_MSK_2, reg32);

    // Enable XVE MSGBOX interrupts
    DPU_LOG(VB0, "Enable XVE MSGBOX interrupts.\n");
    reg32 = CFG_RD32(LW_XVE_PRIV_INTR_EN);
    DPU_LOG(VB2, "LW_XVE_PRIV_INTR_EN[%08x]\n", reg32);
    if (FLD_TEST_DRF(_XVE, _PRIV_INTR_EN, _MSGBOX_INTERRUPT, _DISABLED, reg32))
    {
        reg32 = FLD_SET_DRF(_XVE, _PRIV_INTR_EN, _MSGBOX_INTERRUPT, _ENABLED, reg32);
        CFG_WR32(LW_XVE_PRIV_INTR_EN, reg32);
        DPU_LOG(VB2, "Disabled previously, enabling MSGBOX(%08x)\n", reg32);
    }

    // Make sure no MSGBOX interrupt pending.
    DPU_LOG(VB0, "Check there's no pending MSGBOX interrupt pending.\n");
    reg32 = CFG_RD32(LW_XVE_PRIV_INTR);
    DPU_LOG(VB2, "LW_XVE_PRIV_INTR[%08x]\n", reg32);
    if (FLD_TEST_DRF(_XVE, _PRIV_INTR, _MSGBOX_INTERRUPT, _PENDING, reg32))
    {
        reg32 = FLD_SET_DRF(_XVE, _PRIV_INTR, _MSGBOX_INTERRUPT, _NOT_PENDING, reg32);
        CFG_WR32(LW_XVE_PRIV_INTR, reg32);
        DPU_LOG(VB0, "Pending MSGBOX interrupt. Clearing it.\n");
    }

    // Write command and generate interrupt
    DPU_LOG(VB0, "Write command and generate interrupt.\n");
    CFG_WR32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_COMMAND, 0x80100000);

    // Check XVE_MSGBOX_INTERRUPT pending
    DPU_LOG(VB0, "Check XVE_MSGBOX_INTERRUPT pending.\n");
    reg32 = CFG_RD32(LW_XVE_PRIV_INTR);
    if (FLD_TEST_DRF(_XVE, _PRIV_INTR, _MSGBOX_INTERRUPT, _NOT_PENDING, reg32))
    {
        DPU_LOG(VB0, "XVE_MSGBOX_INTERRUPT NOT PENDING.\n");
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    DPU_LOG(VB0, "XVE_MSGBOX_INTERRUPT PENDING. (OK)\n");

    // Check pending on INTR_0, not pending on INTR_2
    DPU_LOG(VB0, "Check pending on INTR_0, not pending on INTR_2");
    intr0 = GPU_REG_RD32(LW_PMC_INTR_0);
    intr2 = GPU_REG_RD32(LW_PMC_INTR_2);

    if (!(FLD_TEST_DRF(_PMC, _INTR_0, _XVE, _PENDING, intr0) &&
          FLD_TEST_DRF(_PMC, _INTR_2, _XVE, _NOT_PENDING, intr2)))
    {
        DPU_LOG(VB0, "FAILED. intr0[%08x], intr2[%08x]\n", intr0, intr2);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    DPU_LOG(VB1, "PASSED. intr0[%08x], intr2[%08x]\n", intr0, intr2);

    // Enable interrupt INTR_2
    DPU_LOG(VB0, "Enable XVE interrupts INTR_2\n");
    reg32 = FLD_SET_DRF(_PMC, _INTR_MSK_2, _XVE, _ENABLED, intrMsk2Save);
    GPU_REG_WR32(LW_PMC_INTR_MSK_2, reg32);

    intr0 = GPU_REG_RD32(LW_PMC_INTR_0);
    intr2 = GPU_REG_RD32(LW_PMC_INTR_2);

    // Check Pending
    if (FLD_TEST_DRF(_PMC, _INTR_0, _XVE, _NOT_PENDING, intr0) ||
        FLD_TEST_DRF(_PMC, _INTR_2, _XVE, _NOT_PENDING, intr2))
    {
        DPU_LOG(VB0, "INTR_0/2 not pending. Error."
                      " INTR0[%08x], INTR2[%08x]\n", intr0, intr2);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    DPU_LOG(VB0, "PASS: Enable XVE interrupts INTR_2\n");
    // clear interrupts
    reg32 = CFG_RD32(LW_XVE_PRIV_INTR);
    CFG_WR32(LW_XVE_PRIV_INTR, reg32);

    reg32 = CFG_RD32(LW_XVE_PRIV_INTR);
    if (reg32)
    {
        DPU_LOG(VB0, "LW_XVE_PRIV_INTR -> interrupts still pending!!\n");
        BAILOUT(LW_ERR_GENERIC, quit);
    }

    // Run the same test in the reverse order
    DPU_LOG(VB0, "Run the same test in the reverse order\n");

    // Enable XVE interrupts for INTR_2, disable for INTR_0
    DPU_LOG(VB0, "Enable XVE interrupts for INTR_2, disable for INTR_0\n");
    reg32 = FLD_SET_DRF(_PMC, _INTR_MSK_2, _XVE, _ENABLED,  intrMsk0Save);
    GPU_REG_WR32(LW_PMC_INTR_MSK_2, reg32);
    reg32 = FLD_SET_DRF(_PMC, _INTR_MSK_0, _XVE, _DISABLED, intrMsk2Save);
    GPU_REG_WR32(LW_PMC_INTR_MSK_0, reg32);

    // Make sure no MSGBOX interrupt pending.
    DPU_LOG(VB0, "Check there's no pending MSGBOX interrupt pending.\n");
    reg32 = CFG_RD32(LW_XVE_PRIV_INTR);
    DPU_LOG(VB2, "LW_XVE_PRIV_INTR[%08x]\n", reg32);
    if (FLD_TEST_DRF(_XVE, _PRIV_INTR, _MSGBOX_INTERRUPT, _PENDING, reg32))
    {
        reg32 = FLD_SET_DRF(_XVE, _PRIV_INTR, _MSGBOX_INTERRUPT, _NOT_PENDING, reg32);
        CFG_WR32(LW_XVE_PRIV_INTR, reg32);
        DPU_LOG(VB0, "Pending MSGBOX interrupt. Clearing it.\n");
    }

    // Write command and generate interrupt
    DPU_LOG(VB0, "Write command and generate interrupt.\n");
    CFG_WR32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_COMMAND, 0x80100000);

    // Check XVE_MSGBOX_INTERRUPT pending
    DPU_LOG(VB0, "Check XVE_MSGBOX_INTERRUPT pending.\n");
    reg32 = CFG_RD32(LW_XVE_PRIV_INTR);
    if (FLD_TEST_DRF(_XVE, _PRIV_INTR, _MSGBOX_INTERRUPT, _NOT_PENDING, reg32))
    {
        DPU_LOG(VB0, "XVE_MSGBOX_INTERRUPT NOT PENDING.\n");
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    DPU_LOG(VB0, "XVE_MSGBOX_INTERRUPT PENDING. (OK)\n");

    // Check pending on INTR_2, not pending on INTR_0
    DPU_LOG(VB0, "Check pending on INTR_0, not pending on INTR_2\n");
    intr0 = GPU_REG_RD32(LW_PMC_INTR_0);
    intr2 = GPU_REG_RD32(LW_PMC_INTR_2);

    if (!(FLD_TEST_DRF(_PMC, _INTR_2, _XVE, _PENDING, intr2) &&
          FLD_TEST_DRF(_PMC, _INTR_0, _XVE, _NOT_PENDING, intr0)))
    {
        DPU_LOG(VB0, "FAILED. intr0[%08x], intr2[%08x]\n", intr0, intr2);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    DPU_LOG(VB1, "PASSED. intr0[%08x], intr2[%08x]\n", intr0, intr2);

    // Enable interrupt INTR_0
    DPU_LOG(VB0, "Enable XVE interrupts INTR_2\n");
    reg32 = FLD_SET_DRF(_PMC, _INTR_MSK_0, _XVE, _ENABLED, intrMsk2Save);
    GPU_REG_WR32(LW_PMC_INTR_MSK_0, reg32);

    intr0 = GPU_REG_RD32(LW_PMC_INTR_0);
    intr2 = GPU_REG_RD32(LW_PMC_INTR_2);

    // Check Pending
    if (FLD_TEST_DRF(_PMC, _INTR_0, _XVE, _NOT_PENDING, intr0) ||
        FLD_TEST_DRF(_PMC, _INTR_2, _XVE, _NOT_PENDING, intr2))
    {
        DPU_LOG(VB0, "INTR_0/2 not pending. Error."
                      " INTR0[%08x], INTR2[%08x]\n", intr0, intr2);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    DPU_LOG(VB0, "PASS: Enable XVE interrupts INTR_0\n");

    // Disable INTR_2
    DPU_LOG(VB0, "Disable XVE interrupts for INTR_2\n");
    reg32 = FLD_SET_DRF(_PMC, _INTR_MSK_2, _XVE, _DISABLED, intrMsk2Save);
    GPU_REG_WR32(LW_PMC_INTR_MSK_2, reg32);

    // Check INTR_2 interrupts, should NOT_PENDING
    intr0 = GPU_REG_RD32(LW_PMC_INTR_0);
    intr2 = GPU_REG_RD32(LW_PMC_INTR_2);

    // Check Pending
    if (!(FLD_TEST_DRF(_PMC, _INTR_0, _XVE, _PENDING, intr0) &&
          FLD_TEST_DRF(_PMC, _INTR_2, _XVE, _NOT_PENDING, intr2)))
    {
        DPU_LOG(VB0, "Error, INTR_0 should be pending, INTR_2 should not."
                      " INTR0[%08x], INTR2[%08x]\n", intr0, intr2);
        BAILOUT(LW_ERR_GENERIC, quit);
    }

quit:
    // Clear all XVE pending interrupts
    reg32 = CFG_RD32(LW_XVE_PRIV_INTR);
    if (reg32)
    {
        CFG_WR32(LW_XVE_PRIV_INTR, reg32);
        reg32 = CFG_RD32(LW_XVE_PRIV_INTR);
        if (reg32)
        {
            DPU_LOG(VB0, "LW_XVE_PRIV_INTR not cleared. (%08x)\n", reg32);
            status = LW_ERR_GENERIC;
        }
    }
    // Restore LW_PMC_INTR_MSK_0/2
    GPU_REG_WR32(LW_PMC_INTR_MSK_0, intrMsk0Save);
    GPU_REG_WR32(LW_PMC_INTR_MSK_2, intrMsk2Save);
    return status;
}

//----------------------------------------------------------------------
// GPTMR Test (gptimer.bin required to be running)
// Source at: //sw/dev/gpu_drv/chips_a/tools/restricted/pmu/testapps/gptimer
//----------------------------------------------------------------------
LW_STATUS dpuSanityTest_GPTMR_v02_01(LwU32 verbose, char * arg)
{
    LW_STATUS status = LW_OK;
    LwU32 i, reg32, reg32a;
    DPU_LOG(VB0, "<<PMU GPTMR Test>>\n");

    DPU_LOG(VB0, "NOTE: this test requires gptimer.bin\n");

//----------------------------------------------------------------------
// Step 0 : Check gptimer.bin is loaded into DPU
//----------------------------------------------------------------------
    if (!dpuCheckVerifUcode(verbose))
    {
        BAILOUT(LW_ERR_BUSY_RETRY, quit);
    }

//----------------------------------------------------------------------
// Step 1 : Start test, perform delays of 1s 5 times in loop and stop.
//----------------------------------------------------------------------
    // Start the test
    DPU_LOG(VB0, "Starting test. Writing 0 to MAILBOX0\n");
    GPU_REG_WR32(LW_PDISP_FALCON_MAILBOX0, MBOX_CMD_START);

    // Delay (5 seconds)
    DPU_LOG(VB0, "5 Second Delay.\n");
    // little so that we read the data after 1Hz interrupt is handled.
    osPerfDelay(100 * 1000);

    for (i = 1 ; i <= 5 ; i++)
    {
        DPU_LOG(VB0, "%d secs elapsing...\n", i);
        osPerfDelay(1000 * 1000);

        // Check data
        reg32 = GPU_REG_RD32(LW_PDISP_FALCON_MAILBOX1);
        DPU_LOG(VB1, "MAILBOX1 Counter = %d\n", reg32);

        reg32a = GPU_REG_RD32(LW_PDISP_FALCON_OS);
        DPU_LOG(VB1, "FALCON_OS ptimer = %d\n", reg32a);
    }
    GPU_REG_WR32(LW_PDISP_FALCON_MAILBOX0, MBOX_CMD_STOP);

//----------------------------------------------------------------------
// Step 2 : Check that we got to ~5s.
//----------------------------------------------------------------------
    // Check data.
    reg32 = GPU_REG_RD32(LW_PDISP_FALCON_MAILBOX1);
    DPU_LOG(VB0, "MAILBOX1 = %d\n", reg32);

    reg32a = GPU_REG_RD32(LW_PDISP_FALCON_OS);
    DPU_LOG(VB0, "FALCON_OS = %d\n", reg32a);

    // Be forgiving
    if (reg32 < 4 || reg32 > 6)
    {
        DPU_LOG(VB0, "DATA should be 5(+/-1), but %d. "
                     "(compare this to %d in ms incremented by ptimer)\n", reg32, reg32a);
        GPU_REG_WR32(LW_PDISP_FALCON_MAILBOX0, MBOX_CMD_PWRCLK);
        reg32 = GPU_REG_RD32(LW_PDISP_FALCON_MAILBOX1);
        DPU_LOG(VB0, "Check PWRCLK = %d\n", reg32);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    DPU_LOG(VB0, "PASS! Counter = %d, Ptimer (ms) = %d\n", reg32, reg32a);

//----------------------------------------------------------------------
// Step 3 : Start test, perform delays of 1s 7 times in loop and stop.
//----------------------------------------------------------------------
    // Start the test
    DPU_LOG(VB0, "starting test. writing 0 to MAILBOX(0)\n");
    GPU_REG_WR32(LW_PDISP_FALCON_MAILBOX0, MBOX_CMD_START);
    // Delay (7 seconds)
    DPU_LOG(VB0, "7 Second Delay.\n");
    // little so that we read the data after 1Hz interrupt is handled.
    osPerfDelay(100 * 1000);
    for (i = 1 ; i <= 7 ; i++)
    {
        DPU_LOG(VB0, "%d secs elapsing...\n", i);
        osPerfDelay(1000 * 1000);
        // Check data.
        reg32 = GPU_REG_RD32(LW_PDISP_FALCON_MAILBOX1);
        DPU_LOG(VB1, "MAILBOX1 Counter = %d\n", reg32);

        reg32a = GPU_REG_RD32(LW_PDISP_FALCON_OS);
        DPU_LOG(VB1, "FALCON_OS ptimer = %d\n", reg32a);
    }
    GPU_REG_WR32(LW_PDISP_FALCON_MAILBOX0, MBOX_CMD_STOP);

//----------------------------------------------------------------------
// Step 4 : Check that we got to ~7s.
//----------------------------------------------------------------------
    // Check data.
    reg32 = GPU_REG_RD32(LW_PDISP_FALCON_MAILBOX1);
    DPU_LOG(VB0, "MAILBOX1 = %d\n", reg32);

    reg32a = GPU_REG_RD32(LW_PDISP_FALCON_OS);
    DPU_LOG(VB0, "FALCON_OS = %d\n", reg32a);
    if (reg32 < 6 || reg32 > 8)
    {
        DPU_LOG(VB0, "DATA should be 7(+/- 1), but %d. "
                     "(compare this to %d in ms incremented by ptimer)\n", reg32, reg32a);
        GPU_REG_WR32(LW_PDISP_FALCON_MAILBOX0, MBOX_CMD_PWRCLK);
        reg32 = GPU_REG_RD32(LW_PDISP_FALCON_MAILBOX1);
        DPU_LOG(VB0, "Check PWRCLK = %d\n", reg32);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    DPU_LOG(VB0, "PASS! Counter = %d, Ptimer (ms) = %d\n", reg32, reg32a);

quit:
    GPU_REG_WR32(LW_PDISP_FALCON_MAILBOX0, MBOX_CMD_STOP);
    return status;
}

/* ------------------------ Functions -------------------------------------- */
/*!
 *  Runs a specified test from the list of sanity tests available.
 *
 *  @param[in]  testNum  specifies the test to run
 *  @param[in]  verbose  specifies the verbose level (0 to mute)
 *  @param[in]  arg      optional arg
 *
 *  @returns LW_OK    if the test passes
 *  @returns LW_ERR_GENERIC if the test fails
 *  @returns LW_ERR_BUSY_RETRY when the test cannot be run because of PMU's current
 *                    state.  e.g. the test requires the PMU be bootstrapped
 *                    in order to complete the test.
 */
LW_STATUS
dpuSanityTestRun_v02_01
(
    LwU32 testNum,
    LwU32 verbose,
    char* arg
)
{
    DPU_SANITY_CHECK_TEST_ENTRY;
    if (testNum < pDpu[indexGpu].dpuSanityTestGetNum())
    {
        return DpuSanityTests[testNum].fnPtr(verbose, arg);
    }
    else
    {
        dprintf("Test Number (%d) is not available.\n", testNum);
        return LW_ERR_GENERIC;
    }
}

/*!
 *  @returns description of the specified sanity test.
 */
const char *
dpuSanityTestGetInfo_v02_01
(
    LwU32 testNum,
    LwU32 verbose
)
{
    static char buf[1024];
    DPU_SANITY_CHECK_TEST_ENTRY;
    if (testNum < pDpu[indexGpu].dpuSanityTestGetNum())
    {
        if (verbose && DpuSanityTests[testNum].flags)
        {
            LwU32 flags = DpuSanityTests[testNum].flags;
            LwU8  fc[32 * 2];
            LwU32 i;
            memset(fc,'|',sizeof(fc));
            for (i = 0; i < strlen(DPU_TEST_FLAGS_CODE); i++)
            {
                if (flags & BIT(i))
                {
                    fc[i*2] = DPU_TEST_FLAGS_CODE[i];
                }
                else
                {
                    fc[i*2] = ' ';
                }
            }
            fc[i*2 - 1] =  '\0';
            // can't get snprintf to work
            sprintf(buf, "%36s  [%s]",
                    DpuSanityTests[testNum].fnInfo, fc);
            return buf;
        }
        else
        {
            return DpuSanityTests[testNum].fnInfo;
        }
    }
    else
    {
        return "Not Available";
    }
}

/*!
 *  @returns flags of the specified sanity test.
 */
LwU32
dpuSanityTestGetFlags_v02_01
(
    LwU32 testNum
)
{
    DPU_SANITY_CHECK_TEST_ENTRY;
    if (testNum < pDpu[indexGpu].dpuSanityTestGetNum())
    {
        return DpuSanityTests[testNum].flags;
    }
    return 0;
}

/*!
 *  @returns the number of DPU sanity tests available.
 */
LwU32
dpuSanityTestGetNum_v02_01
(
    void
)
{
    return DPU_SANITY_TEST_NUM;
}

/*!
 *  @returns test table
 */
void *
dpuSanityTestGetEntryTable_v02_01()
{
    return (void *) DpuSanityTests_v02_01;
}

/*!
 * @return The falcon core interface FLCN_CORE_IFACES*
 */
const FLCN_CORE_IFACES *
dpuGetFalconCoreIFace_v02_01()
{
    return &flcnCoreIfaces_v04_00;
}

/*!
 * @return The falcon base address of DPU
 */
LwU32
dpuGetFalconBase_v02_01()
{
    return LW_FALCON_DISP_BASE;
}

/*!
 *  Returns the number of queues on the DPU, that includes command queues and
 *  message queues.
 *
 *  @return Number of queue on DPU.
 */
LwU32
dpuQueueGetNum_v02_01()
{
    LwU32 numCmdQs = pDpu[indexGpu].dpuGetCmdQNum();
    LwU32 numMsgQs = pDpu[indexGpu].dpuGetMsgQNum();

    // sum of #CMDQ and #MSGQ
    return numCmdQs + numMsgQs;
}

/*!
 *  Read the contents of a specific queue into queue. pQueue->id will be filled
 *  out automatically as well.
 *
 *  @param queueId Id of queue to get data for. If invalid, then this function
 *                 will return FALSE.
 *  @param pQueue  Pointer to queue structure to fill up.
 *
 *  @return FALSE if queueId is invalid or queue is NULL; TRUE on success.
 */
LwBool
dpuQueueRead_v02_01
(
    LwU32        queueId,
    PFLCN_QUEUE  pQueue
)
{
    const FLCN_ENGINE_IFACES *pFEIF = pDpu[indexGpu].dpuGetFalconEngineIFace();
    const FLCN_CORE_IFACES   *pFCIF = pDpu[indexGpu].dpuGetFalconCoreIFace();
    LwU32 engineBase = pFEIF->flcnEngGetFalconBase();
    LwU32 numQueues;
    LwU32 numCmdQs;
    LwU32 sizeInWords;
    LwU32 dmemPort;

    numQueues = pFEIF->flcnEngQueueGetNum();
    if (queueId >= numQueues || pQueue == NULL)
    {
        return LW_FALSE;
    }

    numCmdQs = pDpu[indexGpu].dpuGetCmdQNum();

    if (queueId < numCmdQs)
    {
        pQueue->head = GPU_REG_RD32(LW_PDISP_FALCON_CMDQ_HEAD(queueId));
        pQueue->tail = GPU_REG_RD32(LW_PDISP_FALCON_CMDQ_TAIL(queueId));
    }
    else
    {
        pQueue->head = GPU_REG_RD32(LW_PDISP_FALCON_MSGQ_HEAD(queueId-numCmdQs));
        pQueue->tail = GPU_REG_RD32(LW_PDISP_FALCON_MSGQ_TAIL(queueId-numCmdQs));
    }

    //
    // At the momement, we assume that tail <= head. This is not the case since
    // the queue wraps-around. Unfortunatly, we have no way of knowing the size
    // or offsets of the queues, and thus renders the parsing slightly
    // impossible. Lwrrently do not support.
    //
    if (pQueue->head < pQueue->tail)
    {
        dprintf("lw: Queue 0x%x is lwrrently in a wrap-around state.\n",
                queueId);
        dprintf("lw:     tail=0x%04x, head=0x%04x\n", pQueue->tail, pQueue->head);
        dprintf("lw:     It is lwrrently not possible to parse this queue.\n");
        return LW_FALSE;
    }

    sizeInWords    = (pQueue->head - pQueue->tail) / sizeof(LwU32);
    pQueue->length = sizeInWords;
    pQueue->id     = queueId;

    //
    // If the queue happens to be larger than normally allowed, print out an
    // error message and return an error.
    //
    if (sizeInWords >= LW_FLCN_MAX_QUEUE_SIZE)
    {
        dprintf("lw: %s: DPU queue 0x%x is larger than configured to read:\n",
                __FUNCTION__, queueId);
        dprintf("lw:     Queue Size: 0x%x     Supported Size: 0x%x\n",
                (LwU32)(sizeInWords * sizeof(LwU32)), (LwU32)(LW_FLCN_MAX_QUEUE_SIZE * sizeof(LwU32)));
        dprintf("lw:     Make LW_FLCN_MAX_QUEUE_SIZE larger and re-compile LW_WATCH\n");
        return LW_FALSE;
    }

    dmemPort = pFCIF->flcnDmemGetNumPorts(engineBase) - 1;

    // Simply read the queue into the buffer
    pFCIF->flcnDmemRead(engineBase, pQueue->tail, LW_TRUE, sizeInWords, dmemPort, pQueue->data);
    return LW_TRUE;
}

/*!
 * Return Ucode file name
 *
 * @return Ucode file name
 */

LwU32
dpuGetMsgQNum_v02_01()
{
    return LW_PDISP_FALCON_MSGQ_HEAD__SIZE_1;
}

/*!
 * Return the path to the ucode without .nm file extension
 *
 * @return Ucode path
 */

char *
dpuUcodeGetPath_v02_01()
{
    static char dpuUCodePath[32] = {0};

    LwU32   falconIpVer = GPU_REG_RD32(LW_PDISP_FALCON_IP_VER);
    LwU32   major       = DRF_VAL(_PDISP_FALCON, _IP_VER, _MAJOR, falconIpVer);
    LwU32   minor       = DRF_VAL(_PDISP_FALCON, _IP_VER, _MINOR, falconIpVer);

    //
    // The major and minor number can be at most just 0xFF,
    // buffer overflow is not possible
    //
    sprintf(dpuUCodePath, "dpu/bin/g_dpuuc%02d%02d", major, minor);

    return dpuUCodePath;
}

/*!
 *  Reset the DPU
*/

void dpuRst_v02_01()
{
    // Reset the DPU Falcon
    GPU_REG_WR32(LW_PDISP_FALCON_CPUCTL, DRF_NUM(_PDISP_FALCON, _CPUCTL, _SRESET, 1));

    // Add Delay
    osPerfDelay(5);
}

/*!
 *  Gets falcon engine base, returns  the given registerMap.
 *
 */
void
dpuFlcngdbGetRegMap_v02_01
(
    FLCNGDB_REGISTER_MAP* registerMap
)
{
   const FLCN_ENGINE_IFACES *pFEIF = pDpu[indexGpu].dpuGetFalconEngineIFace();
   flcnGetFlcngdbRegisterMap_v04_00(pFEIF->flcnEngGetFalconBase(), registerMap);
}
