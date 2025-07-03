/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension for PMU
// pmugk104.c
//
//*****************************************************

//
// includes
//
#include "pmu.h"
#include "print.h"
#include "fb.h"
#include "vmem.h"
#include "elpg.h"
#include "kepler/gk104/dev_pwr_pri.h"
#include "kepler/gk104/dev_bus.h"
#include "kepler/gk104/dev_top.h"
#include "kepler/gk104/dev_master.h"
#include "kepler/gk104/dev_lw_xve.h"

#include "g_pmu_private.h"     // (rmconfig)  implementation prototypes

#include "rmpmucmdif.h"
#include "rmpmusupersurfif.h"

#ifdef CLIENT_SIDE_RESMAN
#include "core/include/modsdrv.h"
#elif defined(OS_Linux)
#include <pthread.h>
#endif

void  vmemDumpInfo(LwU64, LwU64, VMemTypes);
static void _pmuDmemWrite_GK104(LwU32, LwU32, LwU32, LwU32);
static void _pmuImemWrite_GK104(LwU32, LwU32, LwU32, LwU32);

static LW_STATUS _pmuMutexIdGen_GK104(LwU32 *);
static void      _pmuMutexIdRel_GK104(LwU32);

static LwU32     _pmuQueueGetNum_STUB(void);
static LwBool    _pmuQueueRead_STUB(LwU32, PFLCN_QUEUE);


//////////////////////////////////////////////////////////////////////////////
// PMU Sanity Tests
//////////////////////////////////////////////////////////////////////////////

// macros

// CFG Space
#define CFG_RD32(a)   PMU_REG_RD32(DRF_BASE(LW_PCFG) + a)
#define CFG_WR32(a,b) PMU_REG_WR32(DRF_BASE(LW_PCFG) + a, b)
// Prototypes
LW_STATUS pmuSanityTest_CheckImage_GK104   (LwU32, char *);
LW_STATUS pmuSanityTest_Reset_GK104        (LwU32, char *);
LW_STATUS pmuSanityTest_Latency_GK104      (LwU32, char *);
LW_STATUS pmuSanityTest_MutexIdGen_GK104   (LwU32, char *);
LW_STATUS pmuSanityTest_PBI_GK104          (LwU32, char *);
LW_STATUS pmuSanityTest_Bar0Master_GK104   (LwU32, char *);
LW_STATUS pmuSanityTest_Bar0FECS_GK104     (LwU32, char *);
LW_STATUS pmuSanityTest_PbiInterface_GK104 (LwU32, char *);
LW_STATUS pmuSanityTest_GPTMR_GK104        (LwU32, char *);
LW_STATUS pmuSanityTest_Vblank_GK104       (LwU32, char *);
LW_STATUS pmuSanityTest_ScanlineIO_GK104   (LwU32, char *);
LW_STATUS pmuSanityTest_ScanlineIntr_GK104 (LwU32, char *);


static PmuSanityTestEntry PmuSanityTests_GK104[] =
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
        pmuSanityTest_Latency_GK104,
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

#define PMU_SANITY_TEST_NUM (sizeof(PmuSanityTests_GK104) / sizeof(PmuSanityTestEntry))

#define BAILOUT(cond, label) \
    do {status = cond; goto label;} while(0)

/*!
 *  @return Number of mutices on the PMU.
 */
LwU32
pmuMutexGetNum_GK104()
{
    return LW_PPWR_PMU_MUTEX__SIZE_1;
}

/*!
 *  Reset the PMU
 */
LW_STATUS pmuMasterReset_GK104()
{
    LwU32 reg32;
    reg32 = PMU_REG_RD32(LW_PMC_ENABLE);
    reg32 = FLD_SET_DRF(_PMC, _ENABLE, _PWR, _DISABLED, reg32);
    PMU_REG_WR32(LW_PMC_ENABLE, reg32);
    reg32 = FLD_SET_DRF(_PMC, _ENABLE, _PWR, _ENABLED, reg32);
    PMU_REG_WR32(LW_PMC_ENABLE, reg32);

    return LW_OK;
}

//
// PMU Sanity Test Cases
//

#define PMU_APP_OFFSET_DEFAULT  (0x100)

// Image Check Test
LW_STATUS pmuSanityTest_CheckImage_GK104(LwU32 verbose, char *arg)
{
    LwU32      reg32;
    LwU32      imemSize;
    LwU32     *pImemBuf = NULL;
    LwU32     *pSysBuf  = NULL;
    LwU32      wordsRead;
    LwU32      imemBlocks;
    LwU32      b;
    LwU32      w = 0;
    PmuBlock   blockInfo;
    VMemSpace  vMemSpace;
    LwU64      vAddr;
    LW_STATUS  retVal = LW_OK;
    LW_STATUS  status;
    LwBool     bMismatch = TRUE;
    LwU64      vabase;

    PMU_LOG(VB0, ("<<Image Check Test>>\n"));

    imemSize   = pPmu[indexGpu].pmuImemGetSize();
    imemBlocks = imemSize >> 8;

    if (vmemGet(&vMemSpace, VMEM_TYPE_PMU, NULL) != LW_OK)
    {
        PMU_LOG(VB0, "error finding PMU vaspace information\n");
        retVal = LW_ERR_GENERIC;
        goto pmuSanityTest_CheckImage_done;
    }

    pImemBuf = (LwU32 *)malloc(PMU_IMEM_BLOCK_SIZE_BYTES);
    if (pImemBuf == NULL)
    {
        PMU_LOG(VB0, "unable to create temporary buffer\n");
        retVal = LW_ERR_GENERIC;
        goto pmuSanityTest_CheckImage_done;
    }

    pSysBuf = (LwU32 *)malloc(PMU_IMEM_BLOCK_SIZE_BYTES);
    if (pSysBuf == NULL)
    {
        PMU_LOG(VB0, "unable to create temporary buffer\n");
        retVal = LW_ERR_GENERIC;
        goto pmuSanityTest_CheckImage_done;
    }

    vabase = pPmu[indexGpu].pmuGetVABase();

    for (b = 0; b < imemBlocks; b++)
    {
        if (bMismatch)
        {
            PMU_LOG(VB1, " Block  |Code Address| Code Virtual Addr  |Compare Result\n");
            PMU_LOG(VB1, "--------+------------+--------------------|--------------\n");
            bMismatch = FALSE;
        }

        (void)pPmu[indexGpu].pmuImblk(b, &blockInfo);
        if (blockInfo.bValid)
        {
            vAddr = vabase + PMU_APP_OFFSET_DEFAULT + (blockInfo.tag << 8);
            PMU_LOG(VB1, " 0x%04x |   0x%04x   | " LwU64_FMT " |",
                    b, ((blockInfo.tag << 8) + (w * 4)),
                    vAddr);

            // read in the code from FB or SYSMEM over the vaspace
            status = pVmem[indexGpu].vmemRead(
                         &vMemSpace,                    // pVMemSpace
                         vAddr,                         // va
                         PMU_IMEM_BLOCK_SIZE_BYTES,     // length
                         pSysBuf);                      // pData

            if (status != LW_OK)
            {
                PMU_LOG(VB0, "\n    error reading vapsace at address " LwU64_FMT "\n",
                        vAddr);
                retVal = LW_ERR_GENERIC;
                goto pmuSanityTest_CheckImage_done;
            }

            // read in the code from physical IMEM
            wordsRead = pPmu[indexGpu].pmuImemRead(
                         b * PMU_IMEM_BLOCK_SIZE_BYTES, // addr
                         PMU_IMEM_BLOCK_SIZE_WORDS,     // length
                         0,                             // port
                         pImemBuf);                     // pImem

            if (wordsRead != PMU_IMEM_BLOCK_SIZE_WORDS)
            {
                PMU_LOG(VB0, "\n    error reading IMEM at offset 0x%x.\n",
                        b * PMU_IMEM_BLOCK_SIZE_BYTES);
                retVal = LW_ERR_GENERIC;
                goto pmuSanityTest_CheckImage_done;
            }

            // verify the code buffers match
            for (w = 0; w < PMU_IMEM_BLOCK_SIZE_WORDS; w++)
            {
                if (pImemBuf[w] != pSysBuf[w])
                {
                    if (!bMismatch)
                    {
                        PMU_LOG(VB0, " >FAILED\n");
                    }
                    PMU_LOG(VB0, "    !!!Code mismatch: code address=0x%04x "
                                 "[expected 0x%08x, was 0x%08x]\n",
                                 (blockInfo.tag << 8) + (w * 4),
                                 pSysBuf[w],
                                 pImemBuf[w]);
                    bMismatch = TRUE;
                }
            }
            if (!bMismatch)
            {
                PMU_LOG(VB1, "PASSED\n");
            }
            else
            {
                PMU_LOG(VB0, "\n");
                retVal = LW_ERR_GENERIC;
            }
        }
        else
        {
            PMU_LOG(VB1, "Block 0x%04x is invalid.  Skipping...\n", b);
        }
    }

    if (retVal != LW_OK)
    {
        PMU_LOG(VB0, "Image check test failed. Looking for potential "
                     "causes...\n");

        PMU_LOG(VB0, "1) Uncorrectable XVE errors? ");
        reg32 = PMU_REG_RD32(DEVICE_BASE(LW_PCFG) + LW_XVE_AER_UNCORR_ERR);
        if (reg32 != 0x00)
        {
            PMU_LOG(VB0, "Yes\n");
            PMU_LOG(VB0, "::XVE is reporting uncorrectable errors (0x%08x):\n",
                    reg32);

            if (FLD_TEST_DRF(_XVE, _AER_UNCORR_ERR, _CPL_TIMEOUT, _ACTIVE,
                reg32))
            {
                PMU_LOG(VB0, "::   + Completion Timeout Pending (likely source "
                             "of error).\n");
            }
            PMU_LOG(VB0, "\n");
        }
        else
        {
            PMU_LOG(VB0, "No\n");
        }

        PMU_LOG(VB0, "2) PMU running (!halted)? ");
        if (!FLD_TEST_DRF_NUM(_PPWR_FALCON, _IRQSTAT, _HALT, 0x1,
                    PMU_REG_RD32(LW_PPWR_FALCON_IRQSTAT)))
        {
            PMU_LOG(VB0, "Yes\n");
            PMU_LOG(VB0, "::The PMU is not halted. Running this test while the "
                         "PMU is actively running will likely lead\n::to false "
                         "failures due to regular code-swapping activity. The "
                         "alternative is to run with DMA\n::suspended. To suspend"
                         " DMA, use !pmusym to find the address of the symbol "
                         "'uxDmaSuspended' and\n::force its value to 0x1. Be "
                         "sure to restore its original value before resuming.\n");
        }
        else
        {
            PMU_LOG(VB0, "No\n");
        }
        PMU_LOG(VB0, "\n");
    }

pmuSanityTest_CheckImage_done:
    free(pSysBuf);
    free(pImemBuf);
    return retVal;
}

// Reset Test
LW_STATUS pmuSanityTest_Reset_GK104(LwU32 verbose, char *arg)
{
    LwU32 reg32;
    PMU_LOG(VB0, ("<<Reset Test>>\n"));

    // clear all PBUS pending interrupts
    reg32 = PMU_REG_RD32(LW_PBUS_INTR_0);
    PMU_LOG(VB2, "checking pending intr LW_PBUS_INTR0 -> %08x\n", reg32);
    if (reg32)
    {
        PMU_LOG(VB2, "INTR pending -> clearing\n");
        PMU_REG_WR32(LW_PBUS_INTR_0, reg32);
    }
    // read PMC enable
    reg32 = PMU_REG_RD32(LW_PMC_ENABLE);
    PMU_LOG(VB0, "reading PMC_ENABLE (%08x)\n", reg32);

    if (FLD_TEST_DRF(_PMC, _ENABLE, _PWR, _DISABLED, reg32))
    {
        PMU_LOG(VB1, "PMU is not enabled, we're enabling it explicitly\n");
        reg32 = FLD_SET_DRF(_PMC, _ENABLE, _PWR, _ENABLED, reg32);
        PMU_REG_WR32(LW_PMC_ENABLE, reg32);
    }
    PMU_LOG(VB1, "PMU is enabled\n");

    // write to MAILBOX(0)
    PMU_LOG(VB0, "Writing 0xfaceb00c to LW_PPWR_PMU_MAILBOX(0)\n");
    pPmu[indexGpu].pmuWritePmuMailbox(0, 0xfaceb00c);

    // read PMU register (MAILBOX register)
    PMU_LOG(VB0, "Reading LW_PPWR_PMU_MAILBOX(0)\n");
    reg32 = pPmu[indexGpu].pmuReadPmuMailbox(0);
    PMU_LOG(VB1, "MAILBOX(0)[%08x]\n", reg32);

    // verify
    if (0xfaceb00c != reg32)
    {
        PMU_LOG(VB0, "Value mismatch!! %08x != 0xfaceb00c.\n", reg32);
        return LW_ERR_GENERIC;
    }

    // check if there's no PRI error
    PMU_LOG(VB0, "Checking if there's no PRI error.\n");

    reg32 = PMU_REG_RD32(LW_PBUS_INTR_0);
    PMU_LOG(VB1, "LW_PBUS_INTR_0 -> %08x\n", reg32);
    if (reg32)
    {
        PMU_LOG(VB0, "Unexpected interrupts pending, clearing...\n");
        PMU_REG_WR32(LW_PBUS_INTR_0, reg32);
        return LW_ERR_GENERIC;
    }
    PMU_LOG(VB0, "=> No PRI error. (OK)\n");

    // Try to read MAILBOX(0) after disabling the PMU
    PMU_LOG(VB0, "Disabling PMU.\n");
    reg32 = PMU_REG_RD32(LW_PMC_ENABLE);
    reg32 = FLD_SET_DRF(_PMC, _ENABLE, _PWR, _DISABLED, reg32);
    PMU_REG_WR32(LW_PMC_ENABLE, reg32);
    PMU_LOG(VB2, "LW_PMC_ENABLE[%08x]\n", PMU_REG_RD32(LW_PMC_ENABLE));
    PMU_LOG(VB0, "Reading MAILBOX(0)\n");
    reg32 = pPmu[indexGpu].pmuReadPmuMailbox(0);
    PMU_LOG(VB1, "LW_PPWR_PMU_MAILBOX(0)[%08x] ", reg32);
    // error if MAILBOX(0) returns 0xfaceb00c
    if (0xfaceb00c == reg32)
    {
        PMU_LOG(VB0, "MAILBOX(0) returned %08x after the PMU is disabled.\n",
                      reg32);
        return LW_ERR_GENERIC;
    }

    // check for PRI errors. fail if none of them is pending
    PMU_LOG(VB0, "Verify that there are PRI errors.\n");
    reg32 = PMU_REG_RD32(LW_PBUS_INTR_0);
    if (FLD_TEST_DRF(_PBUS, _INTR_0, _PRI_SQUASH,  _NOT_PENDING, reg32) &&
        FLD_TEST_DRF(_PBUS, _INTR_0, _PRI_FECSERR, _NOT_PENDING, reg32) &&
        FLD_TEST_DRF(_PBUS, _INTR_0, _PRI_TIMEOUT, _NOT_PENDING, reg32))
    {
        PMU_LOG(VB0, "No PRI errors.\n");
        return LW_ERR_GENERIC;
    }
    PMU_LOG(VB1, "PRI Error pending => LW_PBUS_INTR_0[%08x]\n", reg32);
    // Clear pending interrupts
    PMU_LOG(VB1, "Clearing LW_PBUS_INTR_0\n");
    PMU_REG_WR32(LW_PBUS_INTR_0, reg32);

    // re-enable PMU
    PMU_LOG(VB0, "Enable PMU.\n");
    reg32 = PMU_REG_RD32(LW_PMC_ENABLE);
    reg32 = FLD_SET_DRF(_PMC, _ENABLE, _PWR, _ENABLED, reg32);
    PMU_REG_WR32(LW_PMC_ENABLE, reg32);

    // read MAILBOX(0) and verify it's 0.
    PMU_LOG(VB0, "Read MAILBOX(0) It should be 0 after reset.\n");
    reg32 = pPmu[indexGpu].pmuReadPmuMailbox(0);
    if (0 != reg32)
    {
        PMU_LOG(VB0, "MAILBOX(0)[%08x] != 0\n", reg32);
        return LW_ERR_GENERIC;
    }

    PMU_LOG(VB0, "Write/Read 'deadface' to MAILBOX(0).\n");
    pPmu[indexGpu].pmuWritePmuMailbox(0, 0xdeadface);
    reg32 = pPmu[indexGpu].pmuReadPmuMailbox(0);
    if (0xdeadface != reg32)
    {
        PMU_LOG(VB0, "ERROR: 'deadface' != MAILBOX(0).\n");
        return LW_ERR_GENERIC;
    }
    PMU_LOG(VB1, "0xdeadface written OK.\n");

    // write 0 to MAILBOX(0)
    pPmu[indexGpu].pmuWritePmuMailbox(0, 0);
    return LW_OK;
}


// Low-Latency Test
LW_STATUS pmuSanityTest_Latency_GK104(LwU32 verbose, char *arg)
{
    LwU32 intrMsk0Save, intrMsk2Save;
    LwU32 reg32, intr0, intr2;
    LW_STATUS status      = LW_OK;
    LwU32 xvePrivMisc = 0;

    // Save the current LW_XVE_PRIV_MISC_1
    xvePrivMisc = CFG_RD32(LW_XVE_PRIV_MISC_1);

    // Force route MSGBOX interrupt to HOST
    reg32 = FLD_SET_DRF(_XVE, _PRIV_MISC_1, _CYA_ROUTE_MSGBOX_CMD_INTR_TO_PMU, _DISABLE, xvePrivMisc);
    CFG_WR32(LW_XVE_PRIV_MISC_1, reg32);

    // Save LW_PMC_INTR_MSK_0/2
    intrMsk0Save = PMU_REG_RD32(LW_PMC_INTR_MSK_0);
    intrMsk2Save = PMU_REG_RD32(LW_PMC_INTR_MSK_2);

    PMU_LOG(VB0, "<<Low-Latency Test>>\n");

    // Enable XVE interrupts for INTR_0, disable for INTR_2
    PMU_LOG(VB0, "Enable XVE interrupts for INTR_0, disable for INTR_2\n");
    reg32 = FLD_SET_DRF(_PMC, _INTR_MSK_0, _XVE, _ENABLED,  intrMsk0Save);
    PMU_REG_WR32(LW_PMC_INTR_MSK_0, reg32);
    reg32 = FLD_SET_DRF(_PMC, _INTR_MSK_2, _XVE, _DISABLED, intrMsk2Save);
    PMU_REG_WR32(LW_PMC_INTR_MSK_2, reg32);

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

    // Check pending on INTR_0, not pending on INTR_2
    PMU_LOG(VB0, "Check pending on INTR_0, not pending on INTR_2");
    intr0 = PMU_REG_RD32(LW_PMC_INTR_0);
    intr2 = PMU_REG_RD32(LW_PMC_INTR_2);

    if (!(FLD_TEST_DRF(_PMC, _INTR_0, _XVE, _PENDING, intr0) &&
          FLD_TEST_DRF(_PMC, _INTR_2, _XVE, _NOT_PENDING, intr2)))
    {
        PMU_LOG(VB0, "FAILED. intr0[%08x], intr2[%08x]\n", intr0, intr2);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    PMU_LOG(VB1, "PASSED. intr0[%08x], intr2[%08x]\n", intr0, intr2);

    // Enable interrupt INTR_2
    PMU_LOG(VB0, "Enable XVE interrupts INTR_2\n");
    reg32 = FLD_SET_DRF(_PMC, _INTR_MSK_2, _XVE, _ENABLED, intrMsk2Save);
    PMU_REG_WR32(LW_PMC_INTR_MSK_2, reg32);

    intr0 = PMU_REG_RD32(LW_PMC_INTR_0);
    intr2 = PMU_REG_RD32(LW_PMC_INTR_2);

    // Check Pending
    if (FLD_TEST_DRF(_PMC, _INTR_0, _XVE, _NOT_PENDING, intr0) ||
        FLD_TEST_DRF(_PMC, _INTR_2, _XVE, _NOT_PENDING, intr2))
    {
        PMU_LOG(VB0, "INTR_0/2 not pending. Error."
                      " INTR0[%08x], INTR2[%08x]\n", intr0, intr2);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    PMU_LOG(VB0, "PASS: Enable XVE interrupts INTR_2\n");
    // clear interrupts
    reg32 = CFG_RD32(LW_XVE_PRIV_INTR);
    CFG_WR32(LW_XVE_PRIV_INTR, reg32);

    reg32 = CFG_RD32(LW_XVE_PRIV_INTR);
    if (reg32)
    {
        PMU_LOG(VB0, "LW_XVE_PRIV_INTR -> interrupts still pending!!\n");
        BAILOUT(LW_ERR_GENERIC, quit);
    }


    // Run the same test in the reverse order
    PMU_LOG(VB0, "Run the same test in the reverse order\n");

    // Enable XVE interrupts for INTR_2, disable for INTR_0
    PMU_LOG(VB0, "Enable XVE interrupts for INTR_2, disable for INTR_0\n");
    reg32 = FLD_SET_DRF(_PMC, _INTR_MSK_2, _XVE, _ENABLED,  intrMsk0Save);
    PMU_REG_WR32(LW_PMC_INTR_MSK_2, reg32);
    reg32 = FLD_SET_DRF(_PMC, _INTR_MSK_0, _XVE, _DISABLED, intrMsk2Save);
    PMU_REG_WR32(LW_PMC_INTR_MSK_0, reg32);

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

    // Check pending on INTR_2, not pending on INTR_0
    PMU_LOG(VB0, "Check pending on INTR_0, not pending on INTR_2\n");
    intr0 = PMU_REG_RD32(LW_PMC_INTR_0);
    intr2 = PMU_REG_RD32(LW_PMC_INTR_2);

    if (!(FLD_TEST_DRF(_PMC, _INTR_2, _XVE, _PENDING, intr2) &&
          FLD_TEST_DRF(_PMC, _INTR_0, _XVE, _NOT_PENDING, intr0)))
    {
        PMU_LOG(VB0, "FAILED. intr0[%08x], intr2[%08x]\n", intr0, intr2);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    PMU_LOG(VB1, "PASSED. intr0[%08x], intr2[%08x]\n", intr0, intr2);

    // Enable interrupt INTR_0
    PMU_LOG(VB0, "Enable XVE interrupts INTR_2\n");
    reg32 = FLD_SET_DRF(_PMC, _INTR_MSK_0, _XVE, _ENABLED, intrMsk2Save);
    PMU_REG_WR32(LW_PMC_INTR_MSK_0, reg32);

    intr0 = PMU_REG_RD32(LW_PMC_INTR_0);
    intr2 = PMU_REG_RD32(LW_PMC_INTR_2);

    // Check Pending
    if (FLD_TEST_DRF(_PMC, _INTR_0, _XVE, _NOT_PENDING, intr0) ||
        FLD_TEST_DRF(_PMC, _INTR_2, _XVE, _NOT_PENDING, intr2))
    {
        PMU_LOG(VB0, "INTR_0/2 not pending. Error."
                      " INTR0[%08x], INTR2[%08x]\n", intr0, intr2);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    PMU_LOG(VB0, "PASS: Enable XVE interrupts INTR_0\n");

    // Disable INTR_2
    PMU_LOG(VB0, "Disable XVE interrupts for INTR_2\n");
    reg32 = FLD_SET_DRF(_PMC, _INTR_MSK_2, _XVE, _DISABLED, intrMsk2Save);
    PMU_REG_WR32(LW_PMC_INTR_MSK_2, reg32);

    // Check INTR_2 interrupts, should NOT_PENDING
    intr0 = PMU_REG_RD32(LW_PMC_INTR_0);
    intr2 = PMU_REG_RD32(LW_PMC_INTR_2);

    // Check Pending
    if (!(FLD_TEST_DRF(_PMC, _INTR_0, _XVE, _PENDING, intr0) &&
          FLD_TEST_DRF(_PMC, _INTR_2, _XVE, _NOT_PENDING, intr2)))
    {
        PMU_LOG(VB0, "Error, INTR_0 should be pending, INTR_2 should not."
                      " INTR0[%08x], INTR2[%08x]\n", intr0, intr2);
        BAILOUT(LW_ERR_GENERIC, quit);
    }

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
    // Restore LW_PMC_INTR_MSK_0/2
    PMU_REG_WR32(LW_PMC_INTR_MSK_0, intrMsk0Save);
    PMU_REG_WR32(LW_PMC_INTR_MSK_2, intrMsk2Save);
    return status;
}

#define MTX_SIZE    256
#define MTX_ILWALID 255

#define GEN_MUTEX(v) \
    do { \
        v = PMU_REG_RD32(LW_PPWR_PMU_MUTEX_ID);\
        if (v < MTX_ILWALID) \
        {\
            mutexByteMap[v] = 1;\
            PMU_LOG(VB2, "MUTEX_ID Generated [%x]\n", v);\
        }\
        else \
        {\
            PMU_LOG(VB1, "MUTEX ID ERROR => %x\n", v);\
        }\
    } while(0)
#define REL_MUTEX(i) \
    do { \
        PMU_REG_WR32(LW_PPWR_PMU_MUTEX_ID_RELEASE, i);\
        if (i < MTX_ILWALID) \
        {\
            mutexByteMap[i] = 0;\
            PMU_LOG(VB2, "MUTEX_ID Released [%x]\n", i);\
        }\
        else \
        {\
            PMU_LOG(VB1, "MUTEX ID ERROR => %x\n", i);\
        }\
    } while(0)

// Mutex and ID Generator Test
LW_STATUS pmuSanityTest_MutexIdGen_GK104(LwU32 verbose, char* arg)
{
    LwU32  savedMutex  = 0;
    LwU32  reg32       = 0;
    LW_STATUS  status      = LW_OK;
    LwU32  m1, m2, i, j;
    LwU8   mutexByteMap[MTX_SIZE];
    LwU8   tempMutexIdToRelease[15];

    // extend this test to all available mutexes when we have more time.
    // LwU32  totalMutex  = LW_PPWR_PMU_MUTEX__SIZE_1

    PMU_LOG(VB0, "<<Mutex and ID Generator Test>>\n");
    // Initialize mutexByteMap to 0
    memset(mutexByteMap, 0, sizeof(mutexByteMap));
    memset(tempMutexIdToRelease, 0, sizeof(tempMutexIdToRelease));

    // Save current mutex0
    savedMutex = PMU_REG_RD32(LW_PPWR_PMU_MUTEX(0));
    // Release mutex0
    PMU_REG_WR32(LW_PPWR_PMU_MUTEX(0), 0);

    // Basic ID Gen Test
    PMU_LOG(VB0, "Get two mutexes.\n");
    GEN_MUTEX(m1);
    GEN_MUTEX(m2);
    PMU_LOG(VB1, "m1(%x) m2(%x)\n", m1, m2);
    if (m1 == MTX_ILWALID || m2 == MTX_ILWALID)
    {
        PMU_LOG(VB0, "Not enough MUTEX IDs available\n");
        BAILOUT(LW_ERR_BUSY_RETRY, quit);
    }
    // Check for uniqueness
    if (m1 == m2)
    {
        PMU_LOG(VB0, "No unique IDs are generated.\n");
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    // Acquire mutex using m1.
    PMU_LOG(VB0, "Acquiring LW_PPWR_PMU_MUTEX(0) with m1(%x). (Should Pass)\n", m1);
    PMU_REG_WR32(LW_PPWR_PMU_MUTEX(0), m1);
    reg32 = PMU_REG_RD32(LW_PPWR_PMU_MUTEX(0));
    PMU_LOG(VB1, "LW_PPWR_PMU_MUTEX(0)[%x]\n", reg32);
    if (reg32 == m1)
    {
        PMU_LOG(VB0, "Acquire Success!\n");
    }
    else
    {
        PMU_LOG(VB0, "Acquire Failed m1[%x] != MUTEX(0)[%x]\n",
                        m1, reg32 );
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    // Acquire with m2, verify that mutex didn't change.
    PMU_LOG(VB0, "Acquiring LW_PPWR_PMU_MUTEX(0) with m2(%x). (Should Fail)\n", m2);
    PMU_REG_WR32(LW_PPWR_PMU_MUTEX(0), m2);
    reg32 = PMU_REG_RD32(LW_PPWR_PMU_MUTEX(0));
    PMU_LOG(VB1, "LW_PPWR_PMU_MUTEX(0)[%x]\n", reg32);
    if (m2 == reg32)
    {
        PMU_LOG(VB0, "MUTEX(0) actually acquired to m2[%x], this is wrong!!\n", m2);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    // release m1, retry with m2
    PMU_LOG(VB0, "Releasing LW_PPWR_PMU_MUTEX(0) -- m1\n");
    PMU_REG_WR32(LW_PPWR_PMU_MUTEX(0), 0);
    // Acquire mutex using m2.
    PMU_LOG(VB0, "Acquiring LW_PPWR_PMU_MUTEX(0) with m2(%x). (Should Pass)\n", m2);
    PMU_REG_WR32(LW_PPWR_PMU_MUTEX(0), m2);
    reg32 = PMU_REG_RD32(LW_PPWR_PMU_MUTEX(0));
    PMU_LOG(VB1, "LW_PPWR_PMU_MUTEX(0)[%x]\n", reg32);
    if (reg32 == m2)
    {
        PMU_LOG(VB0, "Acquire Success!\n");
    }
    else
    {
        PMU_LOG(VB0, "Acquire Failed m2[%x] != MUTEX(0)[%x]\n",
                        m2, reg32 );
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    PMU_LOG(VB0, "Releasing LW_PPWR_PMU_MUTEX(0) -- m2\n");
    PMU_REG_WR32(LW_PPWR_PMU_MUTEX(0), 0);

    // Now exhaust all available mutex ids.
    PMU_LOG(VB0, "Generating all available Mutex IDs\n");
    do
    {
        GEN_MUTEX(reg32);
    } while (reg32 != MTX_ILWALID);

    // verify generating one more id still gives an error
    PMU_LOG(VB0, "MutexIDs all used. Trying to generate one more time.\n");
    GEN_MUTEX(reg32);
    if (reg32 != MTX_ILWALID)
    {
        PMU_LOG(VB0, "Didn't get an error. This is not expected.\n");
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    PMU_LOG(VB0, "Got an error. (OK)\n");
    // let's release some.
    PMU_LOG(VB0, "Releasing some mutexe ids generated.\n");
    for (i = 0, j = 0; i < MTX_SIZE && j < sizeof(tempMutexIdToRelease); i++)
    {
        if (mutexByteMap[i])
        {
            // save mutex id to release
            tempMutexIdToRelease[j++] = (LwU8) i;
            REL_MUTEX(i);
            PMU_LOG(VB0, "--(%d) Releasing [%2x]\n", j, i);
        }
    }
    // Now let's generate mutex id and match them with the ones we've released.
    PMU_LOG(VB0, "Generating all available Mutex IDs\n");
    i = 0;
    do
    {
        GEN_MUTEX(reg32);
        if (reg32 == MTX_ILWALID)
            break;
        // i - # of generated mutex ids
        // j - previously released # of mutexids
        i++;
        if (i > j)
        {
            PMU_LOG(VB0, "ERROR!! we're trying to generate more than "
                         "%d mutexids we have released.\n", j);
            BAILOUT(LW_ERR_GENERIC, quit);
        }
        if (reg32 != tempMutexIdToRelease[i - 1])
        {
            PMU_LOG(VB0, "ERROR!! Expecting %x, but %x generated\n",
                    tempMutexIdToRelease[i - 1], reg32);
            BAILOUT(LW_ERR_GENERIC, quit);
        }
        PMU_LOG(VB0, "-- (%d) Generating [%2x]\n", i, reg32);
    } while (1);

    if (i != j)
    {
        PMU_LOG(VB0, "ERROR!! We've released %d mutex ids,"
                     " but only %d got generated\n", j, i);
        BAILOUT(LW_ERR_GENERIC, quit);
    }

    // finally we're done! let's release what's allocated!
    PMU_LOG(VB0, "DONE!!\n");

quit:
    // Restore saved mutex0
    PMU_REG_WR32(LW_PPWR_PMU_MUTEX(0), savedMutex);
    // Release acquired mutexes
    for (i = 0; i < sizeof(mutexByteMap) ; i++)
    {
        if (mutexByteMap[i])
            REL_MUTEX(i);
    }
    return status;
}

// PBI Test
LW_STATUS pmuSanityTest_PBI_GK104(LwU32 verbose, char* arg)
{
    LwU32 status = LW_OK;
    LwU32 savePrivIntrEn, savePrivCya, savePmuIntrEnR, savePmuIntrEnF;
    LwU32 reg32;
    PMU_LOG(VB0, "<<PBI Basic Test (doesn't require PMUSW to be running>>\n");

    PMU_LOG(VB0, "Checking for PBI capability "
                 "(XVE_PCI_EXPRESS_CAPABILITY_LIST_NEXT_CAPABILITY_PTR_MSGBOX)\n");
    // check for capability
    reg32 = CFG_RD32(LW_XVE_PCI_EXPRESS_CAPABILITY_LIST_NEXT_CAPABILITY_PTR_MSGBOX);
    if (FLD_TEST_DRF_NUM(_XVE, _PCI_EXPRESS_CAPABILITY, _SLOT_IMPLEMENTED, 0, reg32))
    {
        PMU_LOG(VB0, "MSGBOX not implemented! [%08x]\n", reg32);
        return LW_ERR_GENERIC;
    }
    PMU_LOG(VB0, "MSGBOX implemented! (PASS)\n");

    // Save some registers
    savePrivIntrEn = CFG_RD32(LW_XVE_PRIV_INTR_EN);
    savePrivCya    = CFG_RD32(LW_XVE_PRIV_MISC_1);
    savePmuIntrEnR = PMU_REG_RD32(LW_PPWR_PMU_GPIO_INTR_RISING_EN);
    savePmuIntrEnF = PMU_REG_RD32(LW_PPWR_PMU_GPIO_INTR_FALLING_EN);

    // Setting them up.
    reg32 = FLD_SET_DRF(_XVE, _PRIV_INTR_EN, _MSGBOX_INTERRUPT, _ENABLED, savePrivIntrEn);
    CFG_WR32(LW_XVE_PRIV_INTR_EN, reg32);
    PMU_LOG(VB1, "LW_XVE_PRIV_INTR_EN [%08x]\n", CFG_RD32(LW_XVE_PRIV_INTR_EN));
    reg32 = FLD_SET_DRF(_XVE, _PRIV_MISC_1, _CYA_MSGBOX, _ENABLE, savePrivCya);
    reg32 = FLD_SET_DRF(_XVE, _PRIV_MISC_1, _CYA_ROUTE_MSGBOX_CMD_INTR_TO_PMU, _ENABLE, reg32);
    CFG_WR32(LW_XVE_PRIV_MISC_1, reg32);
    PMU_LOG(VB1, "LW_XVE_PRIV_MISC_1 [%08x]\n", CFG_RD32(LW_XVE_PRIV_MISC_1));


    // Disable all PMU interrupts
    PMU_REG_WR32(LW_PPWR_PMU_GPIO_INTR_RISING_EN, 0);
    PMU_REG_WR32(LW_PPWR_PMU_GPIO_INTR_FALLING_EN, 0);

    // Clearing intr.
    reg32 = DRF_DEF(_PPWR, _PMU_GPIO_INTR_FALLING, _XVE_INTR, _PENDING);
    PMU_REG_WR32(LW_PPWR_PMU_GPIO_INTR_FALLING, reg32);
    reg32 = PMU_REG_RD32(LW_PPWR_PMU_GPIO_INTR_FALLING);
    if (FLD_TEST_DRF(_PPWR, _PMU_GPIO_INTR_FALLING, _XVE_INTR, _PENDING, reg32))
    {
        PMU_LOG(VB0, "XVE Interrupt not cleared.. What's wrong!? (%08x)\n", reg32);
        BAILOUT(LW_ERR_GENERIC, quit);
    }

    // Writing command
    PMU_LOG(VB0, "Writing Command to MSGBOX\n");
    CFG_WR32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_COMMAND, 0x80100000);

    // Check if we have GPIO10 interrupt pending.
    PMU_LOG(VB0, "Check if we have XVE interrupt pending.\n");
    reg32 = PMU_REG_RD32(LW_PPWR_PMU_GPIO_INTR_FALLING);
    if (FLD_TEST_DRF(_PPWR, _PMU_GPIO_INTR_FALLING, _XVE_INTR, _NOT_PENDING, reg32))
    {
        PMU_LOG(VB0, "No Interrupt Pending!! PMU_GPIO_INTR_FALLING [%08x]\n", reg32);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    PMU_LOG(VB0, "GPIO10 Interrupt Pending. Test Passed!!\n");
    // Clearing intr.
    reg32 = DRF_DEF(_PPWR, _PMU_GPIO_INTR_FALLING, _XVE_INTR, _PENDING);
    PMU_REG_WR32(LW_PPWR_PMU_GPIO_INTR_FALLING, reg32);
    reg32 = PMU_REG_RD32(LW_PPWR_PMU_GPIO_INTR_FALLING);
    if (FLD_TEST_DRF(_PPWR, _PMU_GPIO_INTR_FALLING, _XVE_INTR, _PENDING, reg32))
    {
        PMU_LOG(VB0, "GPIO10 Interrupt not cleared.. What's wrong!? (%08x)\n", reg32);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    PMU_LOG(VB0, "DONE!!!\n");
quit:
    // Save some registers
     CFG_WR32(LW_XVE_PRIV_INTR_EN, savePrivIntrEn);
     CFG_WR32(LW_XVE_PRIV_MISC_1,  savePrivCya);
     PMU_REG_WR32(LW_PPWR_PMU_GPIO_INTR_RISING_EN,  savePmuIntrEnR);
     PMU_REG_WR32(LW_PPWR_PMU_GPIO_INTR_FALLING_EN, savePmuIntrEnF);
    return status;
}


// BAR0 Master Test
LW_STATUS pmuSanityTest_Bar0Master_GK104(LwU32 verbose, char* arg)
{
    LW_STATUS status = LW_OK;
    LwU32 reg32;
    LwU32 val;
    LwU32 saveDebug11 = PMU_REG_RD32(LW_PBUS_DEBUG_11);
    PMU_LOG(VB0, "<< BAR0 Master Test >>\n");

    for (val = 0 ;val < 5; val++)
    {
        PMU_LOG(VB0, "writing 0xbeefface + %d to PBUS_DEBUG_11. (BAR0 READ)\n", val);
        PMU_REG_WR32(LW_PBUS_DEBUG_11, 0xbeefface + val);

        PMU_REG_WR32(LW_PPWR_PMU_BAR0_ADDR, LW_PBUS_DEBUG_11);
        PMU_REG_WR32(LW_PPWR_PMU_BAR0_CTL,
                DRF_DEF(_PPWR, _PMU_BAR0_CTL, _CMD, _READ)|
                DRF_NUM(_PPWR, _PMU_BAR0_CTL, _WRBE, 0xff)|
                DRF_DEF(_PPWR, _PMU_BAR0_CTL, _TRIG,_TRUE));

        PMU_LOG(VB1, "Issued BAR0_CTL, reading BAR0_CTL back\n");

        for (;;)
        {
            LwU32 bar0stat =
                DRF_VAL(_PPWR, _PMU_BAR0_CTL, _STATUS, PMU_REG_RD32(LW_PPWR_PMU_BAR0_CTL));
            if (bar0stat == LW_PPWR_PMU_BAR0_CTL_STATUS_INIT)
            {
                break;
            }
            else if (bar0stat == LW_PPWR_PMU_BAR0_CTL_STATUS_BUSY)
            {
                PMU_LOG(VB1, "bar0 busy (%x)\n", bar0stat);
                continue;
            }
            else
            {
                PMU_LOG(VB0, "_PMU_BAR0_CTL_STATUS has errors![%08x]\n", bar0stat);
                BAILOUT(LW_ERR_GENERIC, quit);
            }
        }

        reg32 = PMU_REG_RD32(LW_PPWR_PMU_BAR0_DATA);
        if (0xbeefface + val != reg32)
        {
            PMU_LOG(VB0, "DATA[%08x] read doesn't match x=0xbeefface + %d\n", reg32, val);
            BAILOUT(LW_ERR_GENERIC, quit);
        }
        PMU_LOG(VB0, "BAR0 READ OK!![%08x]\n", reg32);

        // now let's do some writing.
        PMU_LOG(VB0, "writing 0xfeedb0cc + %d to PBUS_DEBUG_11. (BAR0 WRITE)\n", val);
        PMU_REG_WR32(LW_PPWR_PMU_BAR0_DATA, 0xfeedb0cc + val);
        PMU_REG_WR32(LW_PPWR_PMU_BAR0_ADDR, LW_PBUS_DEBUG_11);
        PMU_REG_WR32(LW_PPWR_PMU_BAR0_CTL,
                DRF_DEF(_PPWR, _PMU_BAR0_CTL, _CMD, _WRITE)|
                DRF_NUM(_PPWR, _PMU_BAR0_CTL, _WRBE, 0xff)|
                DRF_DEF(_PPWR, _PMU_BAR0_CTL, _TRIG,_TRUE));

        PMU_LOG(VB1, "Issued BAR0_CTL, reading BAR0_CTL back\n");
        for (;;)
        {
            LwU32 bar0stat =
                DRF_VAL(_PPWR, _PMU_BAR0_CTL, _STATUS, PMU_REG_RD32(LW_PPWR_PMU_BAR0_CTL));
            if (bar0stat == LW_PPWR_PMU_BAR0_CTL_STATUS_INIT)
            {
                break;
            }
            else if (bar0stat == LW_PPWR_PMU_BAR0_CTL_STATUS_BUSY)
            {
                PMU_LOG(VB1, "bar0 busy (%x)\n", bar0stat);
                continue;
            }
            else
            {
                PMU_LOG(VB0, "_PMU_BAR0_CTL_STATUS has errors![%08x]\n", bar0stat);
                BAILOUT(LW_ERR_GENERIC, quit);
            }
        }
        reg32 = PMU_REG_RD32(LW_PBUS_DEBUG_11);
        if (0xfeedb0cc + val!= reg32)
        {
            PMU_LOG(VB0, "DATA[%08x] read doesn't match x=0xfeedb0cc + %d\n", reg32, val);
            BAILOUT(LW_ERR_GENERIC, quit);
        }
        PMU_LOG(VB0, "BAR0 WRITE OK!![%08x]\n", reg32);
    }

quit:
    // restore scratch debug11
    PMU_REG_WR32(LW_PBUS_DEBUG_11, saveDebug11);
    return status;
}

#define BAR0_TARGET_FECS       DRF_DEF(_PPWR_PMU, _BAR0_ADDR, _TARGET, _FECS)
#define BAR0_BLOCKING_ENABLE   DRF_DEF(_PPWR_PMU, _BAR0_ADDR, _BLOCKING, _ENABLE)
// BAR0 Master Test
LW_STATUS pmuSanityTest_Bar0FECS_GK104(LwU32 verbose, char* arg)
{
    LwU32 status = LW_OK;
    LwU32 reg32;
    LwU32 val;
    LwU32 saveScratch0 = PMU_REG_RD32(LW_PTOP_SCRATCH0);   // FECS register
    PMU_LOG(VB0, "<< BAR0 Master FECS Test >>\n");

    PMU_LOG(VB1, "Check BAR0_ERROR_STATUS is clear\n");
    reg32 = PMU_REG_RD32(LW_PPWR_PMU_BAR0_ERROR_STATUS);
    if (reg32 != 0)
    {
        PMU_LOG(VB0, "ERROR! LW_PPWR_PMU_BAR0_ERROR_STATUS = 0x%08x\n", reg32);
        PMU_LOG(VB0, "Clear these bits first before running this test\n");
        BAILOUT(LW_ERR_GENERIC, quit);
    }

    for (val = 0 ;val < 5; val++)
    {
        PMU_LOG(VB0, "writing 0xbeefface + %d to LW_PTOP_SCRATCH0(FECS). (BAR0 READ)\n", val);
        PMU_REG_WR32(LW_PTOP_SCRATCH0, 0xbeefface + val);

        PMU_REG_WR32(LW_PPWR_PMU_BAR0_ADDR, LW_PTOP_SCRATCH0 | BAR0_TARGET_FECS);
        PMU_REG_WR32(LW_PPWR_PMU_BAR0_CTL,
                DRF_DEF(_PPWR, _PMU_BAR0_CTL, _CMD, _READ)|
                DRF_NUM(_PPWR, _PMU_BAR0_CTL, _WRBE, 0xff)|
                DRF_DEF(_PPWR, _PMU_BAR0_CTL, _TRIG,_TRUE));

        PMU_LOG(VB1, "Issued BAR0_CTL, reading BAR0_CTL back\n");

        for (;;)
        {
            LwU32 bar0stat =
                DRF_VAL(_PPWR, _PMU_BAR0_CTL, _STATUS, PMU_REG_RD32(LW_PPWR_PMU_BAR0_CTL));
            if (bar0stat == LW_PPWR_PMU_BAR0_CTL_STATUS_INIT)
            {
                break;
            }
            else if (bar0stat == LW_PPWR_PMU_BAR0_CTL_STATUS_BUSY)
            {
                PMU_LOG(VB1, "bar0 busy (%x)\n", bar0stat);
                continue;
            }
            else
            {
                PMU_LOG(VB0, "_PMU_BAR0_CTL_STATUS has errors![%08x]\n", bar0stat);
                BAILOUT(LW_ERR_GENERIC, quit);
            }
        }

        reg32 = PMU_REG_RD32(LW_PPWR_PMU_BAR0_DATA);
        if (0xbeefface + val != reg32)
        {
            PMU_LOG(VB0, "DATA[%08x] read doesn't match x=0xbeefface + %d\n", reg32, val);
            BAILOUT(LW_ERR_GENERIC, quit);
        }
        PMU_LOG(VB0, "BAR0 READ OK!![%08x]\n", reg32);

        // now let's do some writing.
        PMU_LOG(VB0, "writing 0xfeedb0cc + %d to LW_PTOP_SCRATCH0(FECS). (BAR0 WRITE)\n", val);
        PMU_REG_WR32(LW_PPWR_PMU_BAR0_DATA, 0xfeedb0cc + val);
        PMU_REG_WR32(LW_PPWR_PMU_BAR0_ADDR, LW_PTOP_SCRATCH0 | BAR0_TARGET_FECS);
        PMU_REG_WR32(LW_PPWR_PMU_BAR0_CTL,
                DRF_DEF(_PPWR, _PMU_BAR0_CTL, _CMD, _WRITE)|
                DRF_NUM(_PPWR, _PMU_BAR0_CTL, _WRBE, 0xff)|
                DRF_DEF(_PPWR, _PMU_BAR0_CTL, _TRIG,_TRUE));

        PMU_LOG(VB1, "Issued BAR0_CTL, reading BAR0_CTL back\n");
        for (;;)
        {
            LwU32 bar0stat =
                DRF_VAL(_PPWR, _PMU_BAR0_CTL, _STATUS, PMU_REG_RD32(LW_PPWR_PMU_BAR0_CTL));
            if (bar0stat == LW_PPWR_PMU_BAR0_CTL_STATUS_INIT)
            {
                break;
            }
            else if (bar0stat == LW_PPWR_PMU_BAR0_CTL_STATUS_BUSY)
            {
                PMU_LOG(VB1, "bar0 busy (%x)\n", bar0stat);
                continue;
            }
            else
            {
                PMU_LOG(VB0, "_PMU_BAR0_CTL_STATUS has errors![%08x]\n", bar0stat);
                BAILOUT(LW_ERR_GENERIC, quit);
            }
        }
        reg32 = PMU_REG_RD32(LW_PTOP_SCRATCH0);
        if (0xfeedb0cc + val!= reg32)
        {
            PMU_LOG(VB0, "DATA[%08x] read doesn't match x=0xfeedb0cc + %d\n", reg32, val);
            BAILOUT(LW_ERR_GENERIC, quit);
        }
        PMU_LOG(VB0, "FECS BAR0 WRITE OK!![%08x]\n", reg32);

        // let's also do blocking FECS write
        PMU_LOG(VB0, "writing 0xabcd0000 + %d to LW_PTOP_SCRATCH0(FECS).\n", val);
        PMU_LOG(VB0, "(NEW!) [FECS BAR0 BLOCKING WRITE]\n");
        PMU_REG_WR32(LW_PPWR_PMU_BAR0_DATA, 0xabcd0000 + val);
        PMU_REG_WR32(LW_PPWR_PMU_BAR0_ADDR, LW_PTOP_SCRATCH0 | BAR0_TARGET_FECS | BAR0_BLOCKING_ENABLE);
        PMU_REG_WR32(LW_PPWR_PMU_BAR0_CTL,
                DRF_DEF(_PPWR, _PMU_BAR0_CTL, _CMD, _WRITE)|
                DRF_NUM(_PPWR, _PMU_BAR0_CTL, _WRBE, 0xff)|
                DRF_DEF(_PPWR, _PMU_BAR0_CTL, _TRIG,_TRUE));

        PMU_LOG(VB1, "Issued BAR0_CTL, reading BAR0_CTL back\n");
        for (;;)
        {
            LwU32 bar0stat =
                DRF_VAL(_PPWR, _PMU_BAR0_CTL, _STATUS, PMU_REG_RD32(LW_PPWR_PMU_BAR0_CTL));
            if (bar0stat == LW_PPWR_PMU_BAR0_CTL_STATUS_INIT)
            {
                break;
            }
            else if (bar0stat == LW_PPWR_PMU_BAR0_CTL_STATUS_BUSY)
            {
                PMU_LOG(VB1, "bar0 busy (%x)\n", bar0stat);
                continue;
            }
            else
            {
                PMU_LOG(VB0, "_PMU_BAR0_CTL_STATUS has errors![%08x]\n", bar0stat);
                BAILOUT(LW_ERR_GENERIC, quit);
            }
        }
        reg32 = PMU_REG_RD32(LW_PTOP_SCRATCH0);
        if (0xabcd0000 + val!= reg32)
        {
            PMU_LOG(VB0, "DATA[%08x] read doesn't match 0xabcd0000 + %d\n", reg32, val);
            BAILOUT(LW_ERR_GENERIC, quit);
        }
        PMU_LOG(VB0, "FECS BAR0 BLOCKING WRITE OK!![%08x]\n", reg32);
    }

quit:
    // restore scratch debug11
    PMU_REG_WR32(LW_PTOP_SCRATCH0, saveScratch0);
    return status;
}

// PBI Interface Test (PMU Ucode required to be running)
LW_STATUS pmuSanityTest_PbiInterface_GK104(LwU32 verbose, char * arg)
{
    LW_STATUS status = LW_OK;
    LwU32 savePrivIntrEn, savePrivCya, saveMsgMutex;
    LwU32 reg32;
    LwU32 i;
    PMU_LOG(VB0, "<<PBI Interface Test (Requires PMU ucode to be running) >>\n");

    PMU_LOG(VB0, "Checking for PBI capability "
                 "(XVE_PCI_EXPRESS_CAPABILITY_LIST_NEXT_CAPABILITY_PTR_MSGBOX)\n");
    // check for capability
    reg32 = CFG_RD32(LW_XVE_PCI_EXPRESS_CAPABILITY_LIST_NEXT_CAPABILITY_PTR_MSGBOX);
    if (FLD_TEST_DRF_NUM(_XVE, _PCI_EXPRESS_CAPABILITY, _SLOT_IMPLEMENTED, 0, reg32))
    {
        PMU_LOG(VB0, "MSGBOX not implemented! [%08x]\n", reg32);
        return LW_ERR_GENERIC;
    }
    PMU_LOG(VB0, "MSGBOX implemented! (PASS)\n");

    // PMU SW Check.
    PMU_LOG(VB0, "Checking for PMU SW state.\n");

    // Save some registers
    savePrivIntrEn = CFG_RD32(LW_XVE_PRIV_INTR_EN);
    savePrivCya    = CFG_RD32(LW_XVE_PRIV_MISC_1);
    saveMsgMutex   = CFG_RD32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_MUTEX);

    if ((FLD_TEST_DRF_NUM(_PPWR, _FALCON_CPUCTL, _HALTED, 1, PMU_REG_RD32(LW_PPWR_FALCON_CPUCTL))))
    {
        PMU_LOG(VB0, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        PMU_LOG(VB0, "!!                                                                                !!\n");
        PMU_LOG(VB0, "!! PMU is not running. This test requires the PMU ucode be properly bootstrapped  !!\n");
        PMU_LOG(VB0, "!! and running. Also the PMU ucode must be built with PBI capability.             !!\n");
        PMU_LOG(VB0, "!!                                                                                !!\n");
        PMU_LOG(VB0, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        BAILOUT(LW_ERR_BUSY_RETRY, quit);
    }

    // Setting them up.
    reg32 = FLD_SET_DRF(_XVE, _PRIV_INTR_EN, _MSGBOX_INTERRUPT, _ENABLED, savePrivIntrEn);
    CFG_WR32(LW_XVE_PRIV_INTR_EN, reg32);
    PMU_LOG(VB1, "LW_XVE_PRIV_INTR_EN [%08x]\n", CFG_RD32(LW_XVE_PRIV_INTR_EN));
    reg32 = FLD_SET_DRF(_XVE, _PRIV_MISC_1, _CYA_MSGBOX, _ENABLE, savePrivCya);
    reg32 = FLD_SET_DRF(_XVE, _PRIV_MISC_1, _CYA_ROUTE_MSGBOX_CMD_INTR_TO_PMU, _ENABLE, reg32);
    CFG_WR32(LW_XVE_PRIV_MISC_1, reg32);
    PMU_LOG(VB1, "LW_XVE_PRIV_MISC_1 [%08x]\n", CFG_RD32(LW_XVE_PRIV_MISC_1));
    // write 0 to msgbox mutex
    CFG_WR32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_MUTEX, 0);
    PMU_LOG(VB1, "LW_XVE_VENDOR_SPECIFIC_MSGBOX_MUTEX [%08x]\n", CFG_RD32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_MUTEX));

    CFG_WR32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_MUTEX, 0x11);
    CFG_WR32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_COMMAND, 0);
    if (CFG_RD32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_COMMAND) != 0)
    {
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    CFG_WR32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_DATA_IN, 0);
    if (CFG_RD32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_DATA_IN) != 0)
    {
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    CFG_WR32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_DATA_OUT, 0);
    if (CFG_RD32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_DATA_OUT) != 0)
    {
        BAILOUT(LW_ERR_GENERIC, quit);
    }

    // Writing command
    PMU_LOG(VB0, "Writing Command to MSGBOX\n");
    CFG_WR32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_COMMAND, 0x80100000);
    for (i = 0; i < 100; i++)
    {
        if (CFG_RD32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_COMMAND) & 1)
        {
            PMU_LOG(VB0, "DATA Ready!\n");
            break;
        }
        PMU_LOG(VB2, "DATA Not Ready, trying %d\n", i);
    }
    if (i == 100)
    {
        PMU_LOG(VB0, "Failed to get DATA.\n");
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    // verify
    PMU_LOG(VB0, "Checking DATA_OUT.\n");
    reg32 = CFG_RD32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_DATA_OUT);
    PMU_LOG(VB1, "LW_XVE_VENDOR_SPECIFIC_MSGBOX_DATA_OUT[%08x]\n", reg32);
    if (reg32 == 0)
    {
        PMU_LOG(VB0, "We did not get the expected DATA_OUT[%08x] != 0x7\n", reg32);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    PMU_LOG(VB0, "DONE!!.\n");

quit:
    // Save some registers
    CFG_WR32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_COMMAND, 0);
    CFG_WR32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_DATA_OUT,0);
    CFG_WR32(LW_XVE_PRIV_INTR_EN, savePrivIntrEn);
    CFG_WR32(LW_XVE_PRIV_MISC_1,  savePrivCya);
    CFG_WR32(LW_XVE_VENDOR_SPECIFIC_MSGBOX_MUTEX, saveMsgMutex);
    return status;
}


//////////////////////////////////////////////////////////////////////////////
// PMU Common Falcon Interfaces
//////////////////////////////////////////////////////////////////////////////

/*!
 * @return The falcon core interface FLCN_CORE_IFACES*
 */
const FLCN_CORE_IFACES *
pmuGetFalconCoreIFace_GK104()
{
    return &flcnCoreIfaces_v04_00;
}

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

BOOL pmuCheckVerifUcode(LwU32 verbose)
{
    LwU32 reg32;
    // Ping iotest.bin
    pPmu[indexGpu].pmuWritePmuMailbox(0, MBOX_CMD_STOP);
    CMD_DELAY;
    pPmu[indexGpu].pmuWritePmuMailbox(0, MBOX_CMD_RST);
    CMD_DELAY;
    pPmu[indexGpu].pmuWritePmuMailbox(0, MBOX_CMD_PING);

    // delay
    CMD_DELAY;
    reg32 = pPmu[indexGpu].pmuReadPmuMailbox(1);
    if (reg32 != 0xface0000)
    {
        PMU_LOG(VB0, "MAILBOX(1) : %x\n", reg32);
        PMU_LOG(VB0, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        PMU_LOG(VB0, "!!                                                                                !!\n");
        PMU_LOG(VB0, "!! iotest_*.bin is not running. iotest_*.bin must be loaded and running for this  !!\n");
        PMU_LOG(VB0, "!! test. iotest_*.bin is available at                                             !!\n");
        PMU_LOG(VB0, "!! //sw/dev/gpu_drv/chips_a/tools/restricted/pmu/testapps/bin/iotest_*.bin        !!\n");
        PMU_LOG(VB0, "!! * Use !pmuqboot to load the binary.                                            !!\n");
        PMU_LOG(VB0, "!!                                                                                !!\n");
        PMU_LOG(VB0, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        return FALSE;
    }
    return TRUE;
}

// GPTMR Test (iotest.bin required to be running)
LW_STATUS pmuSanityTest_GPTMR_GK104(LwU32 verbose, char * arg)
{
    LW_STATUS status = LW_OK;
    LwU32 i, reg32, reg32a;
    PMU_LOG(VB0, "<<PMU GPTMR Test>>\n");

    PMU_LOG(VB0, "NOTE: this test requires iotest.bin\n");

    if (!pmuCheckVerifUcode(verbose))
    {
        BAILOUT(LW_ERR_BUSY_RETRY, quit);
    }

    // Start the test
    PMU_LOG(VB0, "starting test. writing 0 to MAILBOX(0)\n");
    pPmu[indexGpu].pmuWritePmuMailbox(0, MBOX_CMD_START);

    // Delay (5 seconds)
    PMU_LOG(VB0, "5 Second Delay.\n");
    // little so that we read the data after 1Hz interrupt is handled.
    osPerfDelay(100 * 1000);
    for (i = 1 ; i <= 5 ; i++)
    {
        PMU_LOG(VB0, "%d secs elapsing...\n", i);
        osPerfDelay(1000 * 1000);
        // Check data.
        reg32 = pPmu[indexGpu].pmuReadPmuMailbox(1);
        PMU_LOG(VB1, "MAILBOX(1) Counter = %d\n", reg32);

        reg32a = pPmu[indexGpu].pmuReadPmuMailbox(2);
        PMU_LOG(VB1, "MAILBOX(2) ptimer = %d\n", reg32a);
    }
    pPmu[indexGpu].pmuWritePmuMailbox(0, MBOX_CMD_STOP);


    // Check data.
    reg32 = pPmu[indexGpu].pmuReadPmuMailbox(1);
    PMU_LOG(VB0, "MAILBOX(1) = %d\n", reg32);

    reg32a = pPmu[indexGpu].pmuReadPmuMailbox(2);
    PMU_LOG(VB0, "MAILBOX(2) = %d\n", reg32a);
    // be forgiving
    if (reg32 < 4 || reg32 > 6)
    {
        PMU_LOG(VB0, "DATA should be 5(+/-1), but %d. "
                     "(compare this to %d in ms incremented by ptimer)\n", reg32, reg32a);
        pPmu[indexGpu].pmuWritePmuMailbox(0, MBOX_CMD_PWRCLK);
        reg32 = pPmu[indexGpu].pmuReadPmuMailbox(1);
        PMU_LOG(VB0, "Check PWRCLK = %d\n", reg32);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    PMU_LOG(VB0, "PASS! Counter = %d, Ptimer (ms) = %d\n", reg32, reg32a);

    // Start the test
    PMU_LOG(VB0, "starting test. writing 0 to MAILBOX(0)\n");
    pPmu[indexGpu].pmuWritePmuMailbox(0, MBOX_CMD_START);
    // Delay (7 seconds)
    PMU_LOG(VB0, "7 Second Delay.\n");
    // little so that we read the data after 1Hz interrupt is handled.
    osPerfDelay(100 * 1000);
    for (i = 1 ; i <= 7 ; i++)
    {
        PMU_LOG(VB0, "%d secs elapsing...\n", i);
        osPerfDelay(1000 * 1000);
        // Check data.
        reg32 = pPmu[indexGpu].pmuReadPmuMailbox(1);
        PMU_LOG(VB1, "MAILBOX(1) Counter = %d\n", reg32);

        reg32a = pPmu[indexGpu].pmuReadPmuMailbox(2);
        PMU_LOG(VB1, "MAILBOX(2) ptimer = %d\n", reg32a);
    }
    pPmu[indexGpu].pmuWritePmuMailbox(0, MBOX_CMD_STOP);


    // Check data.
    reg32 = pPmu[indexGpu].pmuReadPmuMailbox(1);
    PMU_LOG(VB0, "MAILBOX(1) = %d\n", reg32);

    reg32a = pPmu[indexGpu].pmuReadPmuMailbox(2);
    PMU_LOG(VB0, "MAILBOX(2) = %d\n", reg32a);
    if (reg32 < 6 || reg32 > 8)
    {
        PMU_LOG(VB0, "DATA should be 7(+/- 1), but %d. "
                     "(compare this to %d in ms incremented by ptimer)\n", reg32, reg32a);
        pPmu[indexGpu].pmuWritePmuMailbox(0, MBOX_CMD_PWRCLK);
        reg32 = pPmu[indexGpu].pmuReadPmuMailbox(1);
        PMU_LOG(VB0, "Check PWRCLK = %d\n", reg32);
        BAILOUT(LW_ERR_GENERIC, quit);
    }
    PMU_LOG(VB0, "PASS! Counter = %d, Ptimer (ms) = %d\n", reg32, reg32a);

quit:
    pPmu[indexGpu].pmuWritePmuMailbox(0, MBOX_CMD_STOP);
    return status;
}


#define MBOX_CMD_BLANK_OK               (1)
#define MBOX_CMD_BLANK_NO_DMI_VBLANK    (2)
#define MBOX_CMD_BLANK_TIMEOUT          (3)
#define MBOX_CMD_BLANK_ERROR            (4)

// Vblank Test (iotest.bin required to be running)
LW_STATUS pmuSanityTest_Vblank_GK104 (LwU32 verbose, char * arg)
{
    LW_STATUS status = LW_OK;
    LwU32 reg32 ;
    PMU_LOG(VB0, "<<PMU VBLANK Test>>\n");

    PMU_LOG(VB0, "NOTE: this test requires iotest.bin\n");

    if (!pmuCheckVerifUcode(verbose))
    {
        BAILOUT(LW_ERR_BUSY_RETRY, quit);
    }

    // Start the test
    pPmu[indexGpu].pmuWritePmuMailbox(0, MBOX_CMD_RST);
    PMU_LOG(VB0, "Testing only vblank1 CMD_BLANK to MAILBOX(0)\n");
    pPmu[indexGpu].pmuWritePmuMailbox(1, MBOX_CMD_BLANK_VBLANK1);
    pPmu[indexGpu].pmuWritePmuMailbox(0, MBOX_CMD_BLANK);

    // delay
    osPerfDelay(100000);

    // Check data.
    reg32 = pPmu[indexGpu].pmuReadPmuMailbox(2);
    PMU_LOG(VB0, "MAILBOX(2) = %d\n", reg32);

    switch (reg32)
    {
        case MBOX_CMD_BLANK_OK:
            PMU_LOG(VB0, "VBLANK Found!\n");
            status = LW_OK;
            break;
        case MBOX_CMD_BLANK_NO_DMI_VBLANK:
            PMU_LOG(VB0, "VBLANK signal was found, but DMI_POS doesn't report VBLANK.\n");
            PMU_LOG(VB0, "Check if the display is set up correctly and DMI_VBLANK is not set to 0.\n");
            status = LW_ERR_BUSY_RETRY;
            break;
        case MBOX_CMD_BLANK_TIMEOUT:
            PMU_LOG(VB0, "Vblank not found. Is head 0 active?!\n");
            status = LW_ERR_GENERIC;
            break;
        case MBOX_CMD_BLANK_ERROR:
            PMU_LOG(VB0, "Invalid test mode!\n");
            status = LW_ERR_GENERIC;
            break;
        default:
            PMU_LOG(VB0, "Unexpected return code : %d!\n", reg32);
            status = LW_ERR_GENERIC;
    }

quit:
    return status;
}

#define MBOX_CMD_SCANLINE_OK            (1)
#define MBOX_CMD_SCANLINE_VBLANK        (2)
#define MBOX_CMD_SCANLINE_WAYOFF        (3)
#define MBOX_CMD_SCANLINE_TIMEOUT       (4)
#define MBOX_CMD_SCANLINE_MISMATCH      (5)

#define LW_MBOX_DATA0_SCANLINE_HEAD             3:0
#define LW_MBOX_DATA0_SCANLINE_HEAD_0             0
#define LW_MBOX_DATA0_SCANLINE_HEAD_1             1
#define LW_MBOX_DATA0_SCANLINE_INTR             4:4
#define LW_MBOX_DATA0_SCANLINE_INTR_DISABLE       0
#define LW_MBOX_DATA0_SCANLINE_INTR_ENABLE        1
#define LW_MBOX_DATA0_SCANLINE_SRC              8:8
#define LW_MBOX_DATA0_SCANLINE_SRC_RG             0
#define LW_MBOX_DATA0_SCANLINE_SRC_DMI            1
#define LW_MBOX_DATA0_SCANLINE_LINE           31:16
#define LW_MBOX_DATA0_SCANLINE_LINE_IGNORE   0x7fff

#define TEST_SRC_RG   LW_MBOX_DATA0_SCANLINE_SRC_RG
#define TEST_SRC_DMI  LW_MBOX_DATA0_SCANLINE_SRC_DMI
#define TEST_SCANLINE (0x100)
// GPTMR Test (iotest.bin required to be running)
static LW_STATUS pmuSanityTest_ScanlineTest_GK104 (LwU32 verbose, LwU32 src, BOOL bUseIntr)
{
    LW_STATUS status = LW_OK;
    LwU32 cmd, reg32;
    LwU32 head = 0;
    PMU_LOG(VB0, "<<%s Scanline %s Test>>\n",
            src == TEST_SRC_RG ? "RG" : "DMI",
            bUseIntr ? "Interrupt" : "I/O");

    PMU_LOG(VB0, "NOTE: this test requires iotest.bin\n");

    if (!pmuCheckVerifUcode(verbose))
    {
        BAILOUT(LW_ERR_BUSY_RETRY, quit);
    }

    PMU_LOG(VB0, "Testing head : %d, src:  %s,  Scanline : %d\n",
            head, src == TEST_SRC_RG ? "RG" : "DMI", TEST_SCANLINE);
    cmd = DRF_NUM(_MBOX_DATA0, _SCANLINE, _HEAD, head)|
          DRF_NUM(_MBOX_DATA0, _SCANLINE, _INTR, bUseIntr)|
          DRF_NUM(_MBOX_DATA0, _SCANLINE, _SRC , src)|
          DRF_NUM(_MBOX_DATA0, _SCANLINE, _LINE, TEST_SCANLINE);
    // Start the test
    pPmu[indexGpu].pmuWritePmuMailbox(0, MBOX_CMD_RST);
    pPmu[indexGpu].pmuWritePmuMailbox(1, cmd);
    pPmu[indexGpu].pmuWritePmuMailbox(0, MBOX_CMD_SCANLINE);

    // delay
    osPerfDelay(100000);

    // Check data.
    reg32 = pPmu[indexGpu].pmuReadPmuMailbox(2);
    PMU_LOG(VB0, "MAILBOX(2) = %d\n", reg32);

    switch (reg32)
    {
        case MBOX_CMD_SCANLINE_OK:
            PMU_LOG(VB0, "Scanline Test Passed!\n");
            status = LW_OK;
            break;
        case MBOX_CMD_SCANLINE_VBLANK:
            PMU_LOG(VB0, "We should not be in Vblank.\n");
            status = LW_ERR_GENERIC;
            break;
        case MBOX_CMD_SCANLINE_TIMEOUT:
            PMU_LOG(VB0, "Timed out. Is head 0 active?!\n");
            status = LW_ERR_GENERIC;
            break;
        case MBOX_CMD_SCANLINE_MISMATCH:
            PMU_LOG(VB0, "the requested scanline (%d)"
                         " did not match what we found!\n", TEST_SCANLINE );
            PMU_LOG(VB0, "actual scanline = %d\n",  pPmu[indexGpu].pmuReadPmuMailbox(3));
            status = LW_ERR_GENERIC;
            break;
        default:
            PMU_LOG(VB0, "Unexpected return code : %d!\n", reg32);
            status = LW_ERR_GENERIC;
    }

quit:
    return status;
}

LW_STATUS pmuSanityTest_ScanlineIO_GK104 (LwU32 verbose, char * arg)
{
    LW_STATUS status;
    status = pmuSanityTest_ScanlineTest_GK104(verbose, TEST_SRC_RG,
            LW_MBOX_DATA0_SCANLINE_INTR_DISABLE);
    if (status != LW_OK)
    {
        return status;
    }
    return  pmuSanityTest_ScanlineTest_GK104(verbose, TEST_SRC_DMI,
            LW_MBOX_DATA0_SCANLINE_INTR_DISABLE);
}
LW_STATUS pmuSanityTest_ScanlineIntr_GK104 (LwU32 verbose, char * arg)
{
    LW_STATUS status;
    status =  pmuSanityTest_ScanlineTest_GK104(verbose, TEST_SRC_RG,
            LW_MBOX_DATA0_SCANLINE_INTR_ENABLE);
    if (status != LW_OK)
    {
        return status;
    }
    return  pmuSanityTest_ScanlineTest_GK104(verbose, TEST_SRC_DMI,
            LW_MBOX_DATA0_SCANLINE_INTR_ENABLE);
}

/*!
 *  @returns the number of PMU sanity tests available.
 */
LwU32
pmuSanityTestGetNum_GK104
(
    void
)
{
    return PMU_SANITY_TEST_NUM;
}

/*!
 *  @returns test table
 */
void *
pmuSanityTestGetEntryTable_GK104()
{
    return (void *) PmuSanityTests_GK104;
}

LwU64
pmuGetVABase_GK104(void)
{
    return 0x8000000;
}

/*!
 * Return the value of a falcon register denoted by index
 *
 * @param[in]  regIdx  Falcon Register Index
 *
 * @return value of a falcon register specified by regIdx
 */
LwU32
pmuFalconGetRegister_GK104
(
    LwU32  regIdx
)
{
    LwU32 icdCmd;

    if (regIdx >= LW_FALCON_REG_SIZE)
    {
        return 0xffffffff;
    }

    icdCmd = DRF_DEF(_PPWR, _FALCON_ICD_CMD, _OPC, _RREG) |
             DRF_NUM(_PPWR, _FALCON_ICD_CMD, _IDX, regIdx);

    PMU_REG_WR32(LW_PPWR_FALCON_ICD_CMD, icdCmd);

    return PMU_REG_RD32(LW_PPWR_FALCON_ICD_RDATA);
}

/*!
 * Returns the number of physical mutexes supported by the chip.
 *
 * @param[out]  pCount
 *     Pointer to store the number of physical mutexes may by the chip.  The
 *     caller may assume that this number will always be greater than zero.
 *
 * @return 'LW_OK'                    If the count pointer is non-NULL
 * @return 'LW_ERR_ILWALID_ARGUMENT'  Otherwise
 */
LW_STATUS
pmuMutex_GetCount_GK104
(
    LwU32 *pCount
)
{
    if (pCount == NULL)
    {
        return LW_ERR_ILWALID_ARGUMENT;
    }
    *pCount = LW_PPWR_PMU_MUTEX__SIZE_1;
    return LW_OK;
}

/*!
 * Read LW_PPWR_PMU_MAILBOX(i)
 */
LwU32
pmuReadPmuMailbox_GK104
(
    LwU32 index
)
{
    return PMU_REG_RD32(LW_PPWR_PMU_MAILBOX(index));
}

/*!
 * Writes to LW_PPWR_PMU_MAILBOX(i)
 */
void
pmuWritePmuMailbox_GK104
(
    LwU32 index,
    LwU32 value
)
{
    PMU_REG_WR32(LW_PPWR_PMU_MAILBOX(index), value);
    return;
}

/*!
 * Reads LW_PPWR_PMU_NEW_INSTBLK
 */
LwU32 pmuReadPmuNewInstblk_GK104()
{
    return PMU_REG_RD32(LW_PPWR_PMU_NEW_INSTBLK);
}


static FLCN_ENGINE_IFACES flcnEngineIfaces_pmu =
{
    pmuGetFalconCoreIFace_STUB,                // flcnEngGetCoreIFace
    pmuGetFalconBase_STUB,                     // flcnEngGetFalconBase
    pmuGetEngineName,                          // flcnEngGetEngineName
    pmuUcodeName_STUB,                         // flcnEngUcodeName
    pmuGetSymFilePath,                         // flcnEngGetSymFilePath
    _pmuQueueGetNum_STUB,                      // flcnEngQueueGetNum
    _pmuQueueRead_STUB,                        // flcnEngQueueRead
    pmuGetDmemAccessPort,                      // flcnEngGetDmemAccessPort
    pmuIsDmemRangeAccessible_STUB,             // flcnEngIsDmemRangeAccessible
    pmuEmemGetOffsetInDmemVaSpace_STUB,        // flcnEngEmemGetOffsetInDmemVaSpace
    pmuEmemGetSize_STUB,                       // flcnEngEmemGetSize
    pmuEmemGetNumPorts_STUB,                   // flcnEngEmemGetNumPorts
    pmuEmemRead_STUB,                          // flcnEngEmemRead
    pmuEmemWrite_STUB,                         // flcnEngEmemWrite
};  // flcnEngineIfaces_pmu

/*!
 * @return The falcon engine interface FLCN_ENGINE_IFACES*
 */
const FLCN_ENGINE_IFACES *
pmuGetFalconEngineIFace_GK104()
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
          FLCN_ENGINE_IFACES *pFEIF = NULL;

    pFCIF = pPmu[indexGpu].pmuGetFalconCoreIFace();

    // The Falcon Engine interface is supported only when the Core Interface exists.
    if(pFCIF != NULL)
    {
        pFEIF = &flcnEngineIfaces_pmu;

        pFEIF->flcnEngGetCoreIFace          = pPmu[indexGpu].pmuGetFalconCoreIFace;
        pFEIF->flcnEngGetFalconBase         = pPmu[indexGpu].pmuGetFalconBase;
        pFEIF->flcnEngUcodeName             = pPmu[indexGpu].pmuUcodeName;
        pFEIF->flcnEngQueueGetNum           = pPmu[indexGpu].pmuQueueGetNum;
        pFEIF->flcnEngQueueRead             = pPmu[indexGpu].pmuQueueRead;
        pFEIF->flcnEngIsDmemRangeAccessible = pPmu[indexGpu].pmuIsDmemRangeAccessible;
    }

    return pFEIF;
}

/*!
 * @return The falcon base address of PMU
 */
LwU32
pmuGetFalconBase_GK104()
{
    return DEVICE_BASE(LW_PPWR);
}

/*!
 *  Placed here for backward compatibility. Can remove all instance of
 *  CDECL_LWWATCH from the code once the code starts running stdcall from
 *  video team on emulation environment using the dll exported directly
 *  from the falcon tool chain.
 */
#ifndef CDECL_LWWATCH
#define CDECL_LWWATCH
#endif

static PmuSanityTestEntry* PmuSanityTests = NULL;

#define PMU_SANITY_CHECK_TEST_ENTRY \
    do\
    {\
        if (PmuSanityTests == NULL)\
        {\
            PmuSanityTests = (PmuSanityTestEntry *) \
                pPmu[indexGpu].pmuSanityTestGetEntryTable();\
        }\
    } while(0)

/*!
 *  @return Size in bytes of the PMU DMEM.
 */
LwU32
pmuDmemGetSize_GK104()
{
    return DRF_VAL(_PPWR_FALCON, _HWCFG, _DMEM_SIZE,
                   PMU_REG_RD32(LW_PPWR_FALCON_HWCFG)) << 8;
}

/*!
 *  @return Number of PMU DMEM ports.
 */
LwU32
pmuDmemGetNumPorts_GK104()
{
    return DRF_VAL(_PPWR_FALCON, _HWCFG1, _DMEM_PORTS,
                   PMU_REG_RD32(LW_PPWR_FALCON_HWCFG1));
}

/*!
 *  Read length words of DMEM starting at offset addr. Addr will automatically
 *  be truncated down to 4-byte aligned value. If length spans out of the range
 *  of the DMEM, it will automatically  be truncated to fit into the DMEM
 *  range.
 *
 *  @param addr      Offset into the DMEM to start reading.
 *  @param bIsAddrVa Consider the addr as virtual address
 *                   (ignored if DMEM VA is not enabled).
 *  @param length    Number of 4-byte words to read.
 *  @param port      Port to read from.
 *  @param pDmem     Buffer to store DMEM into.
 *
 *  @return 0 on error, or number of 4-byte words read.
 */
LwU32
pmuDmemRead_GK104
(
    LwU32  addr,
    LwBool bIsAddrVa,
    LwU32  length,
    LwU32  port,
    LwU32 *pDmem
)
{
    LwU32  dmemSize    = 0x0;
    LwU32  dmemcOrig   = 0x0;
    LwU32  dmemc       = 0x0;
    LwU32  maxPort     = 0x0;
    LwU32  i           = 0x0;

    addr    &= ~(sizeof(LwU32) - 1);
    dmemSize = pPmu[indexGpu].pmuDmemGetSize();
    maxPort  = pPmu[indexGpu].pmuDmemGetNumPorts() - 1;

    // Fail if the port specified is not valid
    if (port > maxPort)
    {
        dprintf("lw:\tport 0x%x is out of range (max 0x%x).\n",
                port, maxPort);
        return 0;
    }

    // Fail if the address it out of range
    if (addr >= dmemSize)
    {
        dprintf("lw:\taddress 0x%x is out of range (max 0x%x).\n",
                addr, dmemSize - 1);
        return 0;
    }

    // Truncate the length down to a reasonable size
    if ((addr + (length * sizeof(LwU32))) > dmemSize)
    {
        length = (dmemSize - addr) / sizeof(LwU32);
        dprintf("lw:\twarning: length truncated to fit into DMEM range.\n");
    }

    //
    // Build the DMEMC command that auto-increments on each read
    // We take the address and mask it off with OFFSET and BLOCK region.
    //
    // Note: We also remember and restore the original command value
    //
    dmemc = (addr & (DRF_SHIFTMASK(LW_PPWR_FALCON_DMEMC_OFFS)  |
                     DRF_SHIFTMASK(LW_PPWR_FALCON_DMEMC_BLK))) |
                     DRF_NUM(_PPWR, _FALCON_DMEMC, _AINCR, 0x1);
    dmemcOrig = PMU_REG_RD32(LW_PPWR_FALCON_DMEMC(port));
    PMU_REG_WR32(LW_PPWR_FALCON_DMEMC(port), dmemc);

    // Perform the actual DMEM read operations
    for (i = 0; i < length; i++)
    {
        pDmem[i] = PMU_REG_RD32(LW_PPWR_FALCON_DMEMD(port));
    }

    // Restore the original DMEMC command
    PMU_REG_WR32(LW_PPWR_FALCON_DMEMC(port), dmemcOrig);
    return length;
}

/*!
 * Write 'val' to DMEM address 'addr'.  'val' may be a byte, half-word, or
 * word based on 'width'. There are no alignment restrictions on the address.
 * A length of 1 will perform a single write.  Lengths greater than one may
 * be used to stream writes multiple times to adjacent locations in DMEM.
 *
 * @code
 *     //
 *     // Write '0x00a5' four times starting at address 0x00, and advancing
 *     // the address by 2-bytes per write.
 *     //
 *     pmuDmemWrite(0x0, 0xa5, 0x2, 0x4, 0x0);
 * @endcode
 *
 * @param[in]  addr    Address in DMEM to write 'val'
 * @param[in]  val     Value to write
 * @param[in]  width   width of 'val'; 1=byte, 2=half-word, 4=word
 * @param[in]  length  Number of writes to perform
 * @param[in]  port    DMEM port to use for when reading and writing DMEM
 *
 * @return The number of bytes written; zero denotes a failure.
 */
LwU32
pmuDmemWrite_GK104
(
    LwU32  addr,
    LwBool bIsAddrVa,
    LwU32  val,
    LwU32  width,
    LwU32  length,
    LwU32  port
)
{
    LwU32 dmemSize  = 0x0;
    LwU32 dmemcOrig = 0x0;
    LwU32 maxPort   = 0x0;
    LwU32 i;

    dmemSize = pPmu[indexGpu].pmuDmemGetSize();
    maxPort  = pPmu[indexGpu].pmuDmemGetNumPorts() - 1;

    // Fail if the port specified is not valid
    if (port > maxPort)
    {
        dprintf("lw:\tport 0x%x is out of range (max 0x%x).\n",
                port, maxPort);
        return 0;
    }

    // Fail if the address it out of range
    if (addr >= dmemSize)
    {
        dprintf("lw:\taddress 0x%x is out of range (max 0x%x).\n",
                addr, dmemSize - 1);
        return 0;
    }

    // Width must be 1, 2, or 4.
    if ((width != 1) && (width != 2) && (width != 4))
    {
        dprintf("lw: Error: Width (%u) must be 1, 2, or 4\n", width);
        return 0;
    }

    // Fail if the write will go out-of-bounds
    if ((addr + (length * width)) > dmemSize)
    {
        dprintf("lw: Error: Cannot write to DMEM. Writes goes out-of-"
                "bounds.\n");
        return 0;
    }

    //
    // Verify that the value to be written will not be truncated due to the
    // transfer width.
    //
    if (width < 4)
    {
        if (val & ~((1 << (8 * width)) - 1))
        {
            dprintf("lw: Error: Value (0x%x) exceeds max-size (0x%x) for width "
                   "(%d byte(s)).\n", val, ((1 << (8 * width)) - 1), width);
            return 0;
        }
    }

    // save the current DMEMC register value (to be restored later)
    dmemcOrig = PMU_REG_RD32(LW_PPWR_FALCON_DMEMC(port));
    for (i = 0; i < length; i++)
    {
        _pmuDmemWrite_GK104(addr + (i * width), val, width, port);
    }
    PMU_REG_WR32(LW_PPWR_FALCON_DMEMC(port), dmemcOrig);
    return length * width;
}


/*!
 *  @return Size in bytes of the PMU IMEM.
 */
LwU32
pmuImemGetSize_GK104()
{
    return DRF_VAL(_PPWR_FALCON, _HWCFG, _IMEM_SIZE,
                   PMU_REG_RD32(LW_PPWR_FALCON_HWCFG)) << 8;
}

/*!
 *  @return Number of IMEM blocks in PMU.
 */
LwU32
pmuImemGetNumBlocks_GK104()
{
     return DRF_VAL(_PPWR_FALCON, _HWCFG, _IMEM_SIZE,
                   PMU_REG_RD32(LW_PPWR_FALCON_HWCFG));
}

/*!
 *  @return Number of PMU IMEM ports.
 */
LwU32
pmuImemGetNumPorts_GK104()
{
    return DRF_VAL(_PPWR_FALCON, _HWCFG1, _IMEM_PORTS,
                   PMU_REG_RD32(LW_PPWR_FALCON_HWCFG1));
}

/*!
 *  @return Bit width of PMU IMEM tags.
 */
LwU32
pmuImemGetTagWidth_GK104()
{
    return DRF_VAL(_PPWR_FALCON, _HWCFG1, _TAG_WIDTH,
                   PMU_REG_RD32(LW_PPWR_FALCON_HWCFG1));
}

/*!
 *  Read length words of IMEM starting at offset addr. Addr will automatically
 *  be truncated down to 4-byte aligned value. If length spans out of the range
 *  of the IMEM, it will automatically be truncated to fit into the IMEM range.
 *
 *  @param addr   Offset into the IMEM to start reading.
 *  @param length Number of 4-byte words to read.
 *  @param port   Port to read from.
 *  @param imem   Buffer to store IMEM into.
 *
 *  @return 0 on error, or number of 4-byte words read.
 */
LwU32
pmuImemRead_GK104
(
    LwU32  addr,
    LwU32  length,
    LwU32  port,
    LwU32 *pImem
)
{
    LwU32  imemSize    = 0x0;
    LwU32  imemcOrig   = 0x0;
    LwU32  imemc       = 0x0;
    LwU32  maxPort     = 0x0;
    LwU32  i           = 0x0;

    addr    &= ~(sizeof(LwU32) - 1);
    imemSize = pPmu[indexGpu].pmuImemGetSize();
    maxPort  = pPmu[indexGpu].pmuImemGetNumPorts() - 1;

    // Fail if the port specified is not valid
    if (port > maxPort)
    {
        dprintf("lw:\tport 0x%x is out of range (max 0x%x).\n",
                port, maxPort);
        return 0;
    }

    // Fail if the address it out of range
    if (addr >= imemSize)
    {
        dprintf("lw:\taddress 0x%x is out of range (max 0x%x).\n",
                addr, imemSize - 1);
        return 0;
    }

    // Truncate the length down to a reasonable size
    if (addr + length * sizeof(LwU32) > imemSize)
    {
        length = (imemSize - addr) / sizeof(LwU32);
        dprintf("lw:\twarning: length truncated to fit into IMEM range.\n");
    }

    //
    // Build the IMEMC command that auto-increments on each read
    // We take the address and mask it off with OFFSET and BLOCK region.
    //
    // Note: We also remember and restore the original command value
    //
    imemc = (addr & (DRF_SHIFTMASK(LW_PPWR_FALCON_IMEMC_OFFS)  |
                     DRF_SHIFTMASK(LW_PPWR_FALCON_IMEMC_BLK))) |
                     DRF_NUM(_PPWR, _FALCON_IMEMC, _AINCR, 0x1);
    imemcOrig = PMU_REG_RD32(LW_PPWR_FALCON_IMEMC(port));
    PMU_REG_WR32(LW_PPWR_FALCON_IMEMC(port), imemc);

    // Perform the actual IMEM read operations
    for (i = 0; i < length; i++)
    {
        pImem[i] = PMU_REG_RD32(LW_PPWR_FALCON_IMEMD(port));
    }

    // Restore the original IMEMC command
    PMU_REG_WR32(LW_PPWR_FALCON_IMEMC(port), imemcOrig);
    return length;
}

/*!
 * Write 'val' to IMEM address 'addr'.  'val' may be a byte, half-word, or
 * word based on 'width'. There are no alignment restrictions on the address.
 * A length of 1 will perform a single write.  Lengths greater than one may
 * be used to stream writes multiple times to adjacent locations in IMEM.
 *
 * @code
 *     //
 *     // Write '0x00a5' four times starting at address 0x00, and advancing
 *     // the address by 2-bytes per write.
 *     //
 *     pmuImemWrite(0x0, 0xa5, 0x2, 0x4, 0x0);
 * @endcode
 *
 * @param[in]  addr    Address in IMEM to write 'val'
 * @param[in]  val     Value to write
 * @param[in]  width   width of 'val'; 1=byte, 2=half-word, 4=word
 * @param[in]  length  Number of writes to perform
 * @param[in]  port    IMEM port to use for when reading and writing IMEM
 *
 * @return The number of bytes written; zero denotes a failure.
 */
LwU32
pmuImemWrite_GK104
(
    LwU32 addr,
    LwU32 val,
    LwU32 width,
    LwU32 length,
    LwU32 port
)
{
    LwU32 imemSize  = 0x0;
    LwU32 imemcOrig = 0x0;
    LwU32 maxPort   = 0x0;
    LwU32 i;

    imemSize = pPmu[indexGpu].pmuImemGetSize();
    maxPort  = pPmu[indexGpu].pmuImemGetNumPorts() - 1;

    // Fail if the port specified is not valid
    if (port > maxPort)
    {
        dprintf("lw:\tport 0x%x is out of range (max 0x%x).\n",
                port, maxPort);
        return 0;
    }

    // Fail if the address it out of range
    if (addr >= imemSize)
    {
        dprintf("lw:\taddress 0x%x is out of range (max 0x%x).\n",
                addr, imemSize - 1);
        return 0;
    }

    // Width must be 1, 2, or 4.
    if ((width != 1) && (width != 2) && (width != 4))
    {
        dprintf("lw: Error: Width (%u) must be 1, 2, or 4\n", width);
        return 0;
    }

    // Fail if the write will go out-of-bounds
    if ((addr + (length * width)) > imemSize)
    {
        dprintf("lw: Error: Cannot write to IMEM. Writes goes out-of-"
                "bounds.\n");
        return 0;
    }

    //
    // Verify that the value to be written will not be truncated due to the
    // transfer width.
    //
    if (width < 4)
    {
        if (val & ~((1 << (8 * width)) - 1))
        {
            dprintf("lw: Error: Value (0x%x) exceeds max-size (0x%x) for width "
                   "(%d byte(s)).\n", val, ((1 << (8 * width)) - 1), width);
            return 0;
        }
    }

    // save the current IMEMC register value (to be restored later)
    imemcOrig = PMU_REG_RD32(LW_PPWR_FALCON_IMEMC(port));
    for (i = 0; i < length; i++)
    {
        _pmuImemWrite_GK104(addr + (i * width), val, width, port);
    }
    PMU_REG_WR32(LW_PPWR_FALCON_IMEMC(port), imemcOrig);
    return length * width;
}

/*!
 * Sets IMEM Tag
 * Calls PPWR_FALCON_IMEMT(i)
 *
 * @param[in]  tag     Sets IMEM tag to 'tag'
 * @param[in]  port    IMEM port to use for when setting IMEM tag
 *
 * @return LW_OK if successful, LW_ERR_GENERIC otherwise
 */
LW_STATUS
pmuImemSetTag_GK104
(
    LwU32 tag,
    LwU32 port
)
{
    LwU32 maxPort   = 0x0;

    maxPort  = pPmu[indexGpu].pmuImemGetNumPorts() - 1;

    // Fail if the port specified is not valid
    if (port > maxPort)
    {
        dprintf("lw:\tport 0x%x is out of range (max 0x%x).\n",
                port, maxPort);
        return LW_ERR_GENERIC;
    }

    PMU_REG_WR32(LW_PPWR_FALCON_IMEMT(port), tag);
    return LW_OK;
}

/*!
 *  Returns the number of queues on the PMU. Note that there are N many
 *  "command queues" and one message queue. The queues are accessed by their
 *  number but it can be assumed that the maximum queue index is the message
 *  queue.
 *
 *  @return Number of queue on PMU.
 */
LwU32
pmuQueueGetNum_GK104()
{
    // We have command queues + 1 message queue
    return LW_PPWR_PMU_QUEUE_HEAD__SIZE_1 + 1;
}

/*! Returns the head of the 'command queue' on the PMU. */
LwU32
pmuQueueReadCommandHead_GK104
(
    LwU32 queueId
)
{
    return PMU_REG_RD32(LW_PPWR_PMU_QUEUE_HEAD(queueId));
}

/*! Returns the tail of the 'command queue' on the PMU. */
LwU32
pmuQueueReadCommandTail_GK104
(
    LwU32 queueId
)
{
    return PMU_REG_RD32(LW_PPWR_PMU_QUEUE_TAIL(queueId));
}

/*!
 *  Returns the next queue element index for a FB queue (not used for DMEM queues).
 *
 *  @param *pIndex  Current index to be incremented.
 *  @param  qSize   Size of the queue (number of elements).  Queue is indexes
 *                  are zero based.
 *
 *  @return incremented index.
 */
static void _nextFbQueueElement(LwU32 *pIndex, LwU32 qSize)
{
    if (++(*pIndex) >= qSize)
    {
        *pIndex = 0;
    }
}

/*!
 *  Dump the contents of a specific queue.  This funciton is for FB
 *  queues, and does not apply to DMEM queues.
 *
 *  @param queueId Id of queue to get data for.
 *  @param pQueue  Pointer to queue structurethat describes the queue.
 *
 *  @return void.
 */
void
_fbqDumpQueue
(
    LwU32         queueId,
    PFLCN_QUEUE   pQueue
)
{
    LwU64    superSurfaceFbOffset = 0;
    LwU64    queueOffset;
    PMU_SYM *pMatches;
    LwU32    count;
    LwU32    symValue1;
    LwU32    symValue2;
    LwU32    queueEntrySize = 0;
    LwU32    i;
    BOOL     bActive = LW_TRUE;
    BOOL     bExactFound;
    char     buffer[RM_PMU_FBQ_CMD_ELEMENT_SIZE] = {0};

    // Ensure PMU symbols are loaded.
    if (!pmuSymCheckIfLoaded())
    {
        dprintf("lw:     PMU symbols must be loaded to dump Queue Elements.\n");
        return;
    }

    // Get the Offset in FB (from the PMU) to the Super Surface
    pMatches = pmuSymFind("SuperSurfaceFbOffset", FALSE, &bExactFound, &count);
    if (pMatches == NULL)
    {
        dprintf("lw:     ERROR: Unable to find SuperSurfaceFbOffset in the PMU.\n");
        return;
    }

    if (bExactFound &&
        PMU_DEREF_DMEM_PTR(pMatches->addr, 1, &symValue1) &&
        PMU_DEREF_DMEM_PTR(pMatches->addr+4, 1, &symValue2))
    {
        superSurfaceFbOffset = ((LwU64)symValue2 << 32) + symValue1;
    }
    if (superSurfaceFbOffset == 0)
    {
        dprintf("lw:     ERROR: SuperSurfaceFbOffset is 0.\n");
        return;
    }

    if (queueId >= RM_PMU_FBQ_CMD_COUNT)
    {
        // Message queue
        if ((pQueue->head >= RM_PMU_FBQ_MSG_NUM_ELEMENTS)  ||
            (pQueue->tail >= RM_PMU_FBQ_MSG_NUM_ELEMENTS))
        {
            dprintf("lw:     ERROR: Invalid Head or Tail for FB Queue.\n");
            return;
        }

        // Loop through Queue Elements, oldest in use first.
        for (i = pQueue->tail; ; )
        {
            if (i == pQueue->head)
            {
                bActive = LW_FALSE;
            }

            queueOffset = superSurfaceFbOffset + LW_OFFSETOF(RM_PMU_SUPER_SURFACE, fbq.msgQueue.qElement[i]);
            queueEntrySize = RM_PMU_FBQ_MSG_ELEMENT_SIZE;

            // Read a queue element from FB at fbOffset.
            if (pFb[indexGpu].fbRead(queueOffset, (void*)buffer, queueEntrySize) != LW_ERR_GENERIC)
            {
                //
                // Removed below line, so that this works for now (will dump whole 64 byte MSG Queue Entry)
                // with both the old MSG queue format (started with FLCN HDR), and the new format (starts
                // with PRM_FLCN_FBQ_MSGQ_HDR).
                // queueEntrySize = buffer[1];
                //
                dprintf("lw:     queueId: %u  qElement: 0x%04x  %s\n", queueId, i, bActive ? "ACTIVE" : "COMPLETED");
                printBuffer(buffer, queueEntrySize, queueOffset /* 0 */, 1);
            }

            _nextFbQueueElement(&i, RM_PMU_FBQ_MSG_NUM_ELEMENTS);

            if (i == pQueue->tail)
            {
                break;
            }
        }
    }
    else
    {
        // Command queue
        if ((pQueue->head >= RM_PMU_FBQ_CMD_NUM_ELEMENTS)  ||
            (pQueue->tail >= RM_PMU_FBQ_CMD_NUM_ELEMENTS))
        {
            dprintf("lw:     ERROR: Invalid Head or Tail for FB Queue.\n");
            return;
        }

        // Loop through Queue Elements, oldest in use first.
        for (i = pQueue->tail; ; )
        {
            if (i == pQueue->head)
            {
                bActive = LW_FALSE;
            }

            queueOffset = superSurfaceFbOffset + LW_OFFSETOF(RM_PMU_SUPER_SURFACE, fbq.cmdQueues.queue[queueId].qElement[i]);
            queueEntrySize = RM_PMU_FBQ_CMD_ELEMENT_SIZE;

            // Read a queue element from FB at fbOffset.
            if (pFb[indexGpu].fbRead(queueOffset, (void*)buffer, queueEntrySize) != LW_ERR_GENERIC)
            {
                queueEntrySize = *(LwU16 *)&buffer[4];
                dprintf("lw:     queueId: %u  qElement: 0x%04x  %s\n", queueId, i, bActive ? "ACTIVE" : "COMPLETED");
                printBuffer(buffer, queueEntrySize, queueOffset /* 0 */, 1);
            }

            _nextFbQueueElement(&i, RM_PMU_FBQ_CMD_NUM_ELEMENTS);

            if (i == pQueue->tail)
            {
                break;
            }
        }
    }
}

/*!
 *  Read the contents of a specific queue into queue. Message queue Id comes
 *  sequentially after the command queues. pQueue->id will be filled out
 *  automatically as well.
 *
 *  @param queueId Id of queue to get data for. If invalid, then this function
 *                 will return FALSE.
 *  @param pQueue  Pointer to queue structure to fill up.
 *
 *  @return FALSE if queueId is invalid or queue is NULL; TRUE on success.
 */
LwBool
pmuQueueRead_GK104
(
    LwU32         queueId,
    PFLCN_QUEUE   pQueue
)
{
    LwU32  numQueues;
    LwU32  sizeInWords;

    numQueues = pPmu[indexGpu].pmuQueueGetNum();
    if (queueId >= numQueues || pQueue == NULL)
    {
        return LW_FALSE;
    }

    //
    // The "message" queue comes right after the command queues,
    // so we use a special case to get the information
    //
    if (queueId < (numQueues - 1))
    {
        pQueue->head = pPmu[indexGpu].pmuQueueReadCommandHead(queueId);
        pQueue->tail = pPmu[indexGpu].pmuQueueReadCommandTail(queueId);
    }
    else
    {
        pQueue->head = PMU_REG_RD32(LW_PPWR_PMU_MSGQ_HEAD);
        pQueue->tail = PMU_REG_RD32(LW_PPWR_PMU_MSGQ_TAIL);
    }

    // see if FB Queue
    if ((pQueue->head > 32) && (pQueue->tail > 32))
    {
        // DMEM Queue
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
            dprintf("lw: %s: PMU queue 0x%x is larger than configured to read:\n",
                    __FUNCTION__, queueId);
            dprintf("lw:     Queue Size: 0x%x     Supported Size: 0x%x\n",
                    (LwU32)(sizeInWords * sizeof(LwU32)), (LwU32)(LW_FLCN_MAX_QUEUE_SIZE * sizeof(LwU32)));
            dprintf("lw:     Make LW_FLCN_MAX_QUEUE_SIZE larger and re-compile LW_WATCH\n");
            return LW_FALSE;
        }

        // Simply read the queue into the buffer
        pPmu[indexGpu].pmuDmemRead(pQueue->tail, LW_TRUE, sizeInWords,
                                   LW_PMU_DMEMC_DEFAULT_PORT, pQueue->data);
    }
    else
    {
        // NOT DMEM Queue, it is an FB queue
        if (queueId < RM_PMU_FBQ_CMD_COUNT || queueId == (pPmu[indexGpu].pmuQueueGetNum()-1))
        {
            // FB Queue
            LwU32 numElementsInQueue;
            LwU32 queueSize;
            if (queueId < RM_PMU_FBQ_CMD_COUNT)
            {
                // CMD queue.
                queueSize = RM_PMU_FBQ_CMD_NUM_ELEMENTS;
            }
            else
            {
                // MSG queue
                queueSize = RM_PMU_FBQ_MSG_NUM_ELEMENTS;
            }


            if (pQueue->tail <= pQueue->head)
            {
                numElementsInQueue = pQueue->head - pQueue->tail;
            }
            else
            {
                numElementsInQueue = queueSize - (pQueue->tail - pQueue->head);
            }

            dprintf("lw:   PMU QUEUE 0x%02x: tail=0x%04x, head=0x%04x\n", queueId, pQueue->tail, pQueue->head);
            dprintf("lw:     Queue Type is FB.\n");
            dprintf("lw:     Queue Size: 0x%x.\n", queueSize);
            dprintf("lw:     There are lwrrently 0x%x elements in use.\n", numElementsInQueue);

            // Dump the FB Queue
            _fbqDumpQueue(queueId, pQueue);
        }
        return LW_FALSE;
    }
    return LW_TRUE;
}

/*!
 *  Get the status of an IMEM code block.
 *
 *  @see struct PmuBlock
 *  @param[in]  blockIndex  Specific block index to print status.
 *  @param[out] pBlockInfo  Pointer to block info structure to be filled.
 *
 *  @return TRUE on succes, FALSE otherwise.
 */
BOOL
pmuImblk_GK104
(
    LwU32       blockIndex,
    PmuBlock*   pBlockInfo
)
{
    LwU32  numBlocks   = 0x0;
    LwU32  tagWidth    = 0x0;
    LwU32  tagMask     = 0x0;
    LwU32  cmd         = 0x0;
    LwU32  result      = 0x0;

    numBlocks   = pPmu[indexGpu].pmuImemGetNumBlocks();
    tagWidth    = pPmu[indexGpu].pmuImemGetTagWidth();
    tagMask     = BIT(tagWidth) - 1;

    // Bail out for invalid block index
    if ((blockIndex >= numBlocks) || (pBlockInfo == NULL))
    {
        return FALSE;
    }

    // Create the command, write, and read result
    cmd     = FLD_SET_DRF(_PPWR_FALCON, _IMCTL, _CMD, _IMBLK, blockIndex);
    PMU_REG_WR32(LW_PPWR_FALCON_IMCTL, cmd);
    result  = PMU_REG_RD32(LW_PPWR_FALCON_IMSTAT);

    // Extract the TAG value from bits 8 to 7+TagWidth
    pBlockInfo->tag         = (result >> 8) & tagMask;
    pBlockInfo->bValid      = DRF_VAL(_PPWR_FALCON, _IMBLK, _VALID, result);
    pBlockInfo->bPending    = DRF_VAL(_PPWR_FALCON, _IMBLK, _PENDING, result);
    pBlockInfo->bSelwre     = DRF_VAL(_PPWR_FALCON, _IMBLK, _SELWRE, result);
    pBlockInfo->blockIndex  = blockIndex;
    return TRUE;
}

/*!
 *  Get the status of the block that codeAddr maps to. The IMEM may be "tagged".
 *  In this case, a code address in IMEM may be as large as 7+TagWidth bits. A
 *  specific code address has a tag specified by the bits 8 to 7+TagWidth. This
 *  tag may or may not be "mapped" to a IMEM 256 byte code block. This function
 *  returns the status of the block mapped to by a specific code address
 *  (or specifies it as un-mapped).
 *
 *  A codeAddr that exceeds the maximum code address size as specified above
 *  will automatically be masked to the lower bits allowed. Note that the lower
 *  8 bits of the address are ignored as they form the offset into the block.
 *
 *  @param[in]  codeAddr        Block address to look up mapped block status.
 *  @param[out] pPmuTagBlock    Pointer to tag mapping structure to store
 *                              information regarding block mapped to tag.
 *
 *  @return FALSE on error, TRUE on success.
 */
BOOL
pmuImtag_GK104
(
    LwU32           codeAddr,
    PmuTagBlock*    pPmuTagBlock
)
{
    LwU32  numBlocks   = 0x0;
    LwU32  blockMask   = 0x0;
    LwU32  blockIndex  = 0x0;
    LwU32  valid       = 0x0;
    LwU32  pending     = 0x0;
    LwU32  secure      = 0x0;
    LwU32  multiHit    = 0x0;
    LwU32  miss        = 0x0;
    LwU32  tagWidth    = 0x0;
    LwU32  maxAddr     = 0x0;
    LwU32  cmd         = 0x0;
    LwU32  result      = 0x0;

    // Quick check of argument pointer
    if (pPmuTagBlock == NULL)
    {
        return FALSE;
    }

    numBlocks   = pPmu[indexGpu].pmuImemGetNumBlocks();
    tagWidth    = pPmu[indexGpu].pmuImemGetTagWidth();
    maxAddr     = (BIT(tagWidth) << 8) - 1;

    blockMask = numBlocks;
    ROUNDUP_POW2(blockMask);
    blockMask--;

    //
    // Create the command, write, and read result
    // Command is created by taking:
    //       Bits T+7 - 0: Address
    //       Upper Bits  : LW_PPWR_FALCON_IMCTL_CMD_IMTAG
    // Result is fetched from IMSTAT register
    //
    cmd     = FLD_SET_DRF(_PPWR_FALCON, _IMCTL, _CMD, _IMTAG, codeAddr & maxAddr);
    PMU_REG_WR32(LW_PPWR_FALCON_IMCTL, cmd);
    result  = PMU_REG_RD32(LW_PPWR_FALCON_IMSTAT);

    // Extract the block index and other information
    blockIndex  = result & blockMask;
    valid       = DRF_VAL(_PPWR_FALCON, _IMTAG, _VALID, result);
    pending     = DRF_VAL(_PPWR_FALCON, _IMTAG, _PENDING, result);
    secure      = DRF_VAL(_PPWR_FALCON, _IMTAG, _SELWRE, result);
    multiHit    = DRF_VAL(_PPWR_FALCON, _IMTAG, _MULTI_HIT, result);
    miss        = DRF_VAL(_PPWR_FALCON, _IMTAG, _MISS, result);

    if (miss)
    {
        pPmuTagBlock->mapType = PMU_TAG_UNMAPPED;
    }
    else if (multiHit)
    {
        pPmuTagBlock->mapType = PMU_TAG_MULTI_MAPPED;
    }
    else
    {
        pPmuTagBlock->mapType = PMU_TAG_MAPPED;
    }

    pPmuTagBlock->blockInfo.tag         = (codeAddr & maxAddr) >> 8;
    pPmuTagBlock->blockInfo.blockIndex  = blockIndex;
    pPmuTagBlock->blockInfo.bPending    = pending;
    pPmuTagBlock->blockInfo.bValid      = valid;
    pPmuTagBlock->blockInfo.bSelwre     = secure;
    return TRUE;
}

/*!
 *  Get the contents of specified mutex. A mutexId is a number from 0 to
 *  numMutices-1. The number of mutices available can be retrieved via
 *  pmuGetNumMutex().
 *
 *  @see pmuGetNumMutex()
 *
 *  @param[in]  mutexId Mutex Id
 *  @param[out] pMutex  Mutex data/contents.
 *  @param[out] pFree   Is this mutex lock available or lwrrently held.
 *
 *  @return FALSE on error, TRUE on success.
 */
BOOL
pmuMutexRead_GK104
(
    LwU32    mutexId,
    LwU32*   pMutex,
    BOOL*    pFree
)
{
    LwU32 maxMutex = pPmu[indexGpu].pmuMutexGetNum() - 1;

    if ((mutexId > maxMutex) || (pMutex == NULL) || (pFree == NULL))
    {
        return FALSE;
    }

    *pMutex = DRF_VAL(_PPWR_PMU, _MUTEX, _VALUE,
                     PMU_REG_RD32(LW_PPWR_PMU_MUTEX(mutexId)));
    *pFree  = (*pMutex == LW_PPWR_PMU_MUTEX_VALUE_INITIAL_LOCK);
    return TRUE;
}

#define PMU_TCB_SIZE           0x13
#define PMU_TCB_PTOP_OF_STACK  0x00
#define PMU_TCB_PRIORITY       0x0B
#define PMU_TCB_PSTACK         0x0C
#define PMU_TCB_NUMBER         0x0D
#define PMU_TCB_PTASK_NAME     0x0E
#define PMU_TCB_STACK_DEPTH    0x10

/*!
 * Using the provided TCB address, goes and reads DMEM to populate the given
 * TCB structure.
 *
 * @param[in]   tcbAddress   DMEM address of the TCB to retrieve
 * @param[in]   port         Port to use when reading the DMEM
 * @param[out]  pTcb         Pointer to the TCB structure to populate
 *
 * @return TRUE if the TCB was retrieved without error; FALSE otherwise
 */
BOOL
pmuTcbGet_GK104
(
    LwU32    tcbAddress,
    LwU32    port,
    PMU_TCB *pTcb
)
{
    LwU32       size = 0;
    LwU32      *pBuffer = NULL;
    PMU_SYM    *pMatches;
    BOOL        bExactFound;
    LwU32       count;
    LwU64       lwrtosVerNum;
    const char *pSymLwrtosVersion = "LwosUcodeVersion";
    const char *pSymOsDebugEntryPoint = "OsDebugEntryPoint";
    LwU32       wordsRead;
    BOOL        ret = FALSE;

    pMatches = pmuSymFind(pSymLwrtosVersion, FALSE, &bExactFound, &count);
    pTcb->tcbAddr = tcbAddress;

    if (bExactFound)
    {
        LwU16 rtosVer;
        LwU8  tcbVer;

        if (!PMU_DEREF_DMEM_PTR_64(pMatches->addr, port, &lwrtosVerNum))
        {
            dprintf("lw: %s: Unable to retrieve lwrtosVersion symbol.\n", __FUNCTION__);
            return FALSE;
        }
        rtosVer = DRF_VAL64(_RM, _RTOS, _VERSION, lwrtosVerNum);
        tcbVer = DRF_VAL64(_RM, _TCB, _VERSION, lwrtosVerNum);

        if (rtosVer == LW_RM_RTOS_VERSION_SAFERTOS_V5160_LW12_FALCON)
        {
            if (0x0 == tcbVer)
            {
                // Original TCB version when defines were set for v5.16.0-lw1.2
                pTcb->tcbVer = PMU_TCB_VER_2;
                size = (sizeof(pTcb->pmuTcb.pmuTcb2) + sizeof(LwU32)) / sizeof(LwU32);
            }
            else if (0x1 == tcbVer)
            {
                pTcb->tcbVer = PMU_TCB_VER_3;
                size = (sizeof(pTcb->pmuTcb.pmuTcb3) + sizeof(LwU32)) / sizeof(LwU32);
            }
            else
            {
                dprintf("lw: %s: Invalid LW_RM_TCB_VERSION.\n", __FUNCTION__);
                return FALSE;
            }
        }
        else if (rtosVer == LW_RM_RTOS_VERSION_SAFERTOS_V5160_LW13_FALCON)
        {
            if (0x0 == tcbVer)
            {
                pTcb->tcbVer = PMU_TCB_VER_5;
                size = (sizeof(pTcb->pmuTcb.pmuTcb5) + sizeof(LwU32)) / sizeof(LwU32);
            }
            else
            {
                dprintf("lw: %s: Invalid LW_RM_TCB_VERSION.\n", __FUNCTION__);
                return FALSE;
            }
        }
        else if (rtosVer == LW_RM_RTOS_VERSION_OPENRTOS_V413_LW10_FALCON)
        {
            if (0x0 == tcbVer)
            {
                pTcb->tcbVer = PMU_TCB_VER_1;
                size = (sizeof(pTcb->pmuTcb.pmuTcb1) + sizeof(LwU32)) / sizeof(LwU32);
            }
            else if (0x1 == tcbVer)
            {
                pTcb->tcbVer = PMU_TCB_VER_4;
                size = (sizeof(pTcb->pmuTcb.pmuTcb4) + sizeof(LwU32)) / sizeof(LwU32);
            }
        }
        else
        {
            dprintf("lw: Unknown RTOS version %d.\n", rtosVer);
        }
    }
    else
    {
        //
        // We need to know we are doing the old or the new driver
        // The main difference is if we have a private tcb structure to hold the information
        // By checking the symbol OsDebugEntryPoint, we can know if it is new.
        //
        pMatches = pmuSymFind(pSymOsDebugEntryPoint, FALSE, &bExactFound, &count);
        if (bExactFound)
        {
            pTcb->tcbVer = PMU_TCB_VER_1;
            size = (sizeof(pTcb->pmuTcb.pmuTcb1) + sizeof(LwU32)) / sizeof(LwU32);
        }
        else
        {
            pTcb->tcbVer = PMU_TCB_VER_0;
            size = (sizeof(pTcb->pmuTcb.pmuTcb0) + sizeof(LwU32)) / sizeof(LwU32);
        }
    }

    // Create a temporary buffer to store data
    pBuffer = (LwU32 *)malloc(size * sizeof(LwU32));
    if (pBuffer == NULL)
    {
        dprintf("lw: %s: unable to create temporary buffer\n", __FUNCTION__);
        goto pmuTcbGet_exit;
    }

    // Read the raw TCB data from the DMEM
    wordsRead = pPmu[indexGpu].pmuDmemRead(tcbAddress,
                                           LW_TRUE,
                                           size,
                                           port,
                                           pBuffer);

    if (wordsRead != size)
    {
        dprintf("ERROR: Unable to read TCB data at address 0x%x\n", tcbAddress);
        goto pmuTcbGet_exit;
    }

    if (pTcb->tcbVer == PMU_TCB_VER_0)
    {
        memcpy(&pTcb->pmuTcb.pmuTcb0, pBuffer, sizeof(pTcb->pmuTcb.pmuTcb0));
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_1)
    {
        memcpy(&pTcb->pmuTcb.pmuTcb1, pBuffer, sizeof(pTcb->pmuTcb.pmuTcb1));
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_2)
    {
        memcpy(&pTcb->pmuTcb.pmuTcb2, pBuffer, sizeof(pTcb->pmuTcb.pmuTcb2));
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_3)
    {
        memcpy(&pTcb->pmuTcb.pmuTcb3, pBuffer, sizeof(pTcb->pmuTcb.pmuTcb3));
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_4)
    {
        memcpy(&pTcb->pmuTcb.pmuTcb4, pBuffer, sizeof(pTcb->pmuTcb.pmuTcb4));
    }
    else if (pTcb->tcbVer == PMU_TCB_VER_5)
    {
        memcpy(&pTcb->pmuTcb.pmuTcb5, pBuffer, sizeof(pTcb->pmuTcb.pmuTcb5));
    }
    else
    {
        dprintf("ERROR: Invalid TCB version number. \n");
        goto pmuTcbGet_exit;
    }

    // update return value
    ret = TRUE;

pmuTcbGet_exit:
    if (pBuffer)
    {
        free(pBuffer);
    }
    return ret;
}

LwU32
pmuUcodeGetVersion_GK104(void)
{
    return PMU_REG_RD32(LW_PPWR_FALCON_OS);
}

/*!
 *
 * Start PMU at the given bootvector
 *
 * @param[in]  addr   Specifies bootvector
 */
void pmuBootstrap_GK104(LwU32 bootvector)
{
    // Clear DMACTL
    PMU_REG_WR32(LW_PPWR_FALCON_DMACTL, 0);

    // Set Bootvec
    PMU_REG_WR32(LW_PPWR_FALCON_BOOTVEC,
            DRF_NUM(_PPWR, _FALCON_BOOTVEC, _VEC, bootvector));

    // Start CPU
    PMU_REG_WR32(LW_PPWR_FALCON_CPUCTL,
            DRF_NUM(_PPWR, _FALCON_CPUCTL, _STARTCPU, 1));
}

/*!
 * Write 'val' to DMEM address 'addr'.  'val' may be a byte, half-word, or
 * word based on 'width'. There are no alignment restrictions on the address.
 *
 * @param[in]  addr    Address in DMEM to write 'val'
 * @param[in]  val     Value to write
 * @param[in]  width   width of 'val'; 1=byte, 2=half-word, 4=word
 * @param[in]  port    DMEM port to use for when reading and writing DMEM
 */
static void
_pmuDmemWrite_GK104
(
    LwU32 addr,
    LwU32 val,
    LwU32 width,
    LwU32 port
)
{
    LwU32 unaligned;
    LwU32 addrAligned;
    LwU32 data32;
    LwU32 andMask;
    LwU32 lshift;
    LwU32 overflow = 0;
    LwU32 val2     = 0;

    //
    // DMEM transfers are always in 4-byte alignments/chunks. Callwlate the
    // misalignment and the aligned starting address of the transfer.
    //
    unaligned   = addr & 0x3;
    addrAligned = addr & ~0x3;
    lshift      = unaligned * 8;
    andMask     = (LwU32)~(((((LwU64)1) << (8 * width)) - 1) << lshift);

    if ((unaligned + width) > 4)
    {
        overflow = unaligned + width - 4;
        val2     = (val >> (8 * (width - overflow)));
    }

    PMU_REG_WR32(LW_PPWR_FALCON_DMEMC(port),
        (addrAligned & (DRF_SHIFTMASK(LW_PPWR_FALCON_DMEMC_OFFS)   |
                        DRF_SHIFTMASK(LW_PPWR_FALCON_DMEMC_BLK)))  |
                        DRF_NUM(_PPWR, _FALCON_DMEMC, _AINCR, 0x0) |
                        DRF_NUM(_PPWR, _FALCON_DMEMC, _AINCW, 0x1));

    //
    // Read-in the current 4-byte value and mask off the portion that will
    // be overwritten by this write.
    //
    data32  = PMU_REG_RD32(LW_PPWR_FALCON_DMEMD(port));
    data32 &= andMask;
    data32 |= (val << lshift);
    PMU_REG_WR32(LW_PPWR_FALCON_DMEMD(port), data32);

    if (overflow != 0)
    {
        addrAligned += 4;
        andMask      = ~((1 << (8 * overflow)) - 1);

        data32  = PMU_REG_RD32(LW_PPWR_FALCON_DMEMD(port));
        data32 &= andMask;
        data32 |= val2;
        PMU_REG_WR32(LW_PPWR_FALCON_DMEMD(port), data32);
    }
}

/*!
 * Write 'val' to IMEM address 'addr'.  'val' may be a byte, half-word, or
 * word based on 'width'. There are no alignment restrictions on the address.
 *
 * @param[in]  addr    Address in IMEM to write 'val'
 * @param[in]  val     Value to write
 * @param[in]  width   width of 'val'; 1=byte, 2=half-word, 4=word
 * @param[in]  port    IMEM port to use for when reading and writing IMEM
 */
static void
_pmuImemWrite_GK104
(
    LwU32 addr,
    LwU32 val,
    LwU32 width,
    LwU32 port
)
{
    LwU32 unaligned;
    LwU32 addrAligned;
    LwU32 data32;
    LwU32 andMask;
    LwU32 lshift;
    LwU32 overflow = 0;
    LwU32 val2     = 0;

    //
    // IMEM transfers are always in 4-byte alignments/chunks. Callwlate the
    // misalignment and the aligned starting address of the transfer.
    //
    unaligned   = addr & 0x3;
    addrAligned = addr & ~0x3;
    lshift      = unaligned * 8;
    andMask     = (LwU32)~(((((LwU64)1) << (8 * width)) - 1) << lshift);

    if ((unaligned + width) > 4)
    {
        overflow = unaligned + width - 4;
        val2     = (val >> (8 * (width - overflow)));
    }

    PMU_REG_WR32(LW_PPWR_FALCON_IMEMC(port),
        (addrAligned & (DRF_SHIFTMASK(LW_PPWR_FALCON_IMEMC_OFFS)   |
                        DRF_SHIFTMASK(LW_PPWR_FALCON_IMEMC_BLK)))  |
                        DRF_NUM(_PPWR, _FALCON_IMEMC, _AINCR, 0x0) |
                        DRF_NUM(_PPWR, _FALCON_IMEMC, _AINCW, 0x1));

    //
    // Read-in the current 4-byte value and mask off the portion that will
    // be overwritten by this write.
    //
    data32  = PMU_REG_RD32(LW_PPWR_FALCON_IMEMD(port));
    data32 &= andMask;
    data32 |= (val << lshift);
    PMU_REG_WR32(LW_PPWR_FALCON_IMEMD(port), data32);

    if (overflow != 0)
    {
        addrAligned += 4;
        andMask      = ~((1 << (8 * overflow)) - 1);

        data32  = PMU_REG_RD32(LW_PPWR_FALCON_IMEMD(port));
        data32 &= andMask;
        data32 |= val2;
        PMU_REG_WR32(LW_PPWR_FALCON_IMEMD(port), data32);
    }
}

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
pmuSanityTestRun_GK104
(
    LwU32 testNum,
    LwU32 verbose,
    char* arg
)
{
    PMU_SANITY_CHECK_TEST_ENTRY;
    if (testNum < pPmu[indexGpu].pmuSanityTestGetNum())
    {
        return PmuSanityTests[testNum].fnPtr(verbose, arg);
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
pmuSanityTestGetInfo_GK104
(
    LwU32 testNum,
    LwU32 verbose
)
{
    static char buf[1024];
    PMU_SANITY_CHECK_TEST_ENTRY;
    if (testNum < pPmu[indexGpu].pmuSanityTestGetNum())
    {
        if (verbose && PmuSanityTests[testNum].flags)
        {
            LwU32 flags = PmuSanityTests[testNum].flags;
            LwU8  fc[32 * 2];
            LwU32 i;
            memset(fc,'|',sizeof(fc));
            for (i = 0; i < strlen(PMU_TEST_FLAGS_CODE); i++)
            {
                if (flags & BIT(i))
                {
                    fc[i*2] = PMU_TEST_FLAGS_CODE[i];
                }
                else
                {
                    fc[i*2] = ' ';
                }
            }
            fc[i*2 - 1] =  '\0';
            // can't get snprintf to work
            sprintf(buf, "%36s  [%s]",
                    PmuSanityTests[testNum].fnInfo, fc);
            return buf;
        }
        else
        {
            return PmuSanityTests[testNum].fnInfo;
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
pmuSanityTestGetFlags_GK104
(
    LwU32 testNum
)
{
    PMU_SANITY_CHECK_TEST_ENTRY;
    if (testNum < pPmu[indexGpu].pmuSanityTestGetNum())
    {
        return PmuSanityTests[testNum].flags;
    }
    return 0;
}

/*!
 * Attempts to query the HW-state of the given PMU mutex (identified by
 * physical index) to determine the current owner of the mutex.
 *
 * @param[in]  mutexIndex  The physical mutex-index for the mutex to query
 * @param[in]  ownerId     The value/token that the owner used when acquiring
 *                         the mutex.  Or PMU_ILWALID_MUTEX_OWNER_ID if the
 *                         mutex is not lwrrently acquired.
 *
 * @return 'LW_OK' if the mutex was successfully queried for its owner
 */
LW_STATUS
pmuMutex_QueryOwnerByIndex_GK104
(
    LwU32  mutexIndex,
    LwU32 *pOwner
)
{
    *pOwner = GPU_REG_IDX_RD_DRF(_PPWR_PMU, _MUTEX, mutexIndex, _VALUE);
    return LW_OK;
}

/*!
 * Attempts to acquire the PMU mutex as specified by the given physical
 * mutex-index.
 *
 * @param[in]   mutexIndex  The physical mutex-index for the mutex to acquire
 * @param[out]  pOwnerId    Pointer to the id to write with the id generated
 *                          upon a successful lock of the mutex.  This value
 *                          will remain unchanged upon failure.
 *
 * @return 'LW_OK' if the mutex was successfully acquired.
 * @sa      pmuMutex_ReleaseByIndex_GK104
 */
LW_STATUS
pmuMutex_AcquireByIndex_GK104
(
    LwU32  mutexIndex,
    LwU32 *pOwnerId
)
{
    LwU32      ownerId;
    LwU32      value;
    LW_STATUS  status;

    // generate a unique mutex identifier
    status = _pmuMutexIdGen_GK104(&ownerId);
    if (status != LW_OK)
    {
        return status;
    }

    //
    // Write the id into the mutex register to attempt an "acquire"
    // of the mutex.
    //
    GPU_REG_IDX_WR_DRF_NUM(_PPWR_PMU, _MUTEX, mutexIndex, _VALUE, ownerId);

    //
    // Read the register back to see if the id stuck.  If the value
    // read back matches the id written, the mutex was successfully
    // acquired.  Otherwise, release the id and return an error.
    //
    value = GPU_REG_IDX_RD_DRF(_PPWR_PMU, _MUTEX, mutexIndex, _VALUE);
    if (value == ownerId)
    {
        *pOwnerId = ownerId;
    }
    else
    {
        _pmuMutexIdRel_GK104(ownerId);
        status = LW_ERR_STATE_IN_USE;
    }
    return status;
}


/*!
 * Attempts to release the PMU mutex as specified by the given physical
 * mutex-index.  It is the caller's responsibility to ensure that the mutex was
 * acquired before calling this function.  This function simply performs a
 * "release" operation on the given mutex, and frees the owner-id.
 *
 * @param[in]  mutexIndex  The physical mutex-index for the mutex to release
 * @param[in]  ownerId     The id returned when the mutex was initially
 *                          acquired.
 *
 * @return 'LW_OK' if the mutex and owner ID were successfully released.
 * @sa      pmuMutex_AcquireByIndex_GK104
 */
LW_STATUS
pmuMutex_ReleaseByIndex_GK104
(
    LwU32 mutexIndex,
    LwU32 ownerId
)
{
    GPU_REG_IDX_WR_DRF_DEF(_PPWR_PMU, _MUTEX, mutexIndex, _VALUE, _INITIAL_LOCK);

    // release the mutex identifer
    _pmuMutexIdRel_GK104(ownerId);
    return LW_OK;
}

/*!
 * Generate a unique identifier that may be used for locking the PMU's HW
 * mutexes.
 *
 * @param[out]  pMutexId  Pointer to write with the generated mutex identifier
 *
 * @return 'LW_OK' if a mutex identifier was successfully generated.
 */
static LW_STATUS
_pmuMutexIdGen_GK104
(
    LwU32 *pMutexId
)
{
    LwU32      reg32;
    LW_STATUS  status = LW_OK;

    //
    // Generate a mutex id by reading the MUTEX_ID register (this register
    // has a read side-effect; avoid unnecessary reads).
    //
    reg32 = PMU_REG_RD32(LW_PPWR_PMU_MUTEX_ID);

    //
    // Hardware will return _NOT_AVAIL if all identifiers have been used/
    // consumed. Also check against _INIT since zero is not a valid identifier
    // either (zero is used to release mutexes so it cannot be used as an id).
    //
    if ((!FLD_TEST_DRF(_PPWR_PMU, _MUTEX_ID, _VALUE, _INIT     , reg32)) &&
        (!FLD_TEST_DRF(_PPWR_PMU, _MUTEX_ID, _VALUE, _NOT_AVAIL, reg32)))
    {
        *pMutexId = DRF_VAL(_PPWR_PMU, _MUTEX_ID, _VALUE, reg32);
    }
    else
    {
        status = LW_ERR_INSUFFICIENT_RESOURCES;
        dprintf("LWRM: %s: Failed to generate a mutex identifier. "
                "Hardware indicates that all identifiers have been "
                "consumed.\n", __FUNCTION__);
    }
    return status;
}

/*!
 * Release the given mutex identifier thus making it available to other
 * clients.
 *
 * @param[in]  mutexId  The mutex identifier to release
 */
static void
_pmuMutexIdRel_GK104
(
    LwU32  mutexId
)
{
    GPU_FLD_WR_DRF_NUM(_PPWR_PMU, _MUTEX_ID_RELEASE, _VALUE, mutexId);
}

/*!
 *  default stubbed flcnEngQueueGetNum for flcnEngineIfaces_pmu
 */
static LwU32
_pmuQueueGetNum_STUB(void)
{
    return (LwU32) 0;
}

/*!
 *  default stubbed flcnEngQueueRead for flcnEngineIfaces_pmu
 */
static LwBool
_pmuQueueRead_STUB
(
    LwU32         queueId,
    PFLCN_QUEUE   pQueue
)
{
    return LW_FALSE;
}

LwU32 pmuTestElpgState_GK104(void)
{
    return elpgGetStatus();
}

LwU32 pmuTestLpwrState_GK104(void)
{
    return lpwrGetStatus();
}
