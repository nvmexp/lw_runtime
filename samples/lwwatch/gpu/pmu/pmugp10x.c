/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch PMU helper.
// pmugp10x.c
//
//*****************************************************

//
// includes
//
#include "pmu.h"
#include "pascal/gp102/dev_pwr_pri.h"

#define PMU_RESET_TIMEOUT                     0x1000 //us

static LwBool _pmuDmemWrite_GP102(LwU32  addr, LwBool bIsAddrVa,
                                  LwU32  val, LwU32  width, LwU32  port);
static LwU32 _pmuGetDmemVaBound_GP102(void);

/*!
 *  Reset the PMU 
 */
LW_STATUS pmuMasterReset_GP10X()
{
    LwU32 reg32;
    LwS32 timeoutUs = PMU_RESET_TIMEOUT;

    reg32 = GPU_REG_RD32(LW_PPWR_FALCON_ENGINE);
    reg32 = FLD_SET_DRF(_PPWR, _FALCON_ENGINE, _RESET , _TRUE, reg32);
    GPU_REG_WR32(LW_PPWR_FALCON_ENGINE, reg32);

    reg32 = GPU_REG_RD32(LW_PPWR_FALCON_ENGINE);
    reg32 = FLD_SET_DRF(_PPWR, _FALCON_ENGINE, _RESET , _FALSE, reg32);
    GPU_REG_WR32(LW_PPWR_FALCON_ENGINE, reg32);

    // Wait for SCRUBBING to complete
    while (timeoutUs > 0)
    {
        reg32 = GPU_REG_RD32(LW_PPWR_FALCON_DMACTL);

        if (FLD_TEST_DRF(_PPWR, _FALCON_DMACTL, _DMEM_SCRUBBING, _DONE, reg32) &&
            FLD_TEST_DRF(_PPWR, _FALCON_DMACTL, _IMEM_SCRUBBING, _DONE, reg32))
        {
            break;
        }
        osPerfDelay(20);
        timeoutUs -= 20;
    }

    if (timeoutUs <= 0)
    {
        return LW_ERR_GENERIC;
    }

    return LW_OK;
}

//////////////////////////////////////////////////////////////////////////////
// PMU Sanity Tests
//////////////////////////////////////////////////////////////////////////////

// CFG Space
#define CFG_RD32(a)   PMU_REG_RD32(DRF_BASE(LW_PCFG) + a) 
#define CFG_WR32(a,b) PMU_REG_WR32(DRF_BASE(LW_PCFG) + a, b) 
// Prototypes
LW_STATUS pmuSanityTest_Latency_GP10X      (LwU32, char *);

static PmuSanityTestEntry PmuSanityTests_GP10X[] =
{
    // Check Image
    {
        pmuSanityTest_CheckImage_GK104,
        PMU_TEST_PROD_UCODE,
        "Check Image"
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

#define PMU_SANITY_TEST_NUM (sizeof(PmuSanityTests_GP10X) / sizeof(PmuSanityTestEntry))

#define BAILOUT(cond, label) \
    do {status = cond; goto label;} while(0)

/*!
 *  @returns test table
 */
void *
pmuSanityTestGetEntryTable_GP10X()
{
    return (void *) PmuSanityTests_GP10X;
}

/*!
 *  @returns the number of PMU sanity tests available.
 */
LW_STATUS 
pmuSanityTestGetNum_GP10X
(
    void
)
{
    return PMU_SANITY_TEST_NUM;
}

/*!
 * @return Falcon core interface
 */
const FLCN_CORE_IFACES *
pmuGetFalconCoreIFace_GP104()
{
    return &flcnCoreIfaces_v06_00;
}

/*!
 *  Read length words of DMEM starting at virtual offset addr. Addr will
 *  automatically be truncated down to 4-byte aligned value. If length spans
 *  out of the range of the DMEM, it will automatically  be truncated to fit
 *  into the DMEM range.
 *
 *  @param addr      Virtual address of the DMEM to start reading.
 *  @param bIsAddrVa Consider the addr as a virtual address.
 *  @param length    Number of 4-byte words to read.
 *  @param port      Port to read from.
 *  @param pDmem     Buffer to store DMEM into.
 *
 *  @return 0 on error, or number of 4-byte words read.
 */
LwU32
pmuDmemRead_GP102
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
    dmemSize = pPmu[indexGpu].pmuDmemGetSize(); // This gives us the physical DMEM size
    maxPort  = pPmu[indexGpu].pmuDmemGetNumPorts() - 1;

    // Fail if the port specified is not valid
    if (port > maxPort)
    {
        dprintf("lw:\tport 0x%x is out of range (max 0x%x).\n",
                port, maxPort);
        return 0;
    }

    // Fail if the address it out of range
    if (!bIsAddrVa && (addr >= dmemSize))
    {
        dprintf("lw:\taddress 0x%x is out of range (max 0x%x).\n",
                addr, dmemSize - 1);
        return 0;
    }

    // Truncate the length down to a reasonable size
    if (!bIsAddrVa && ((addr + (length * sizeof(LwU32))) > dmemSize))
    {
        dprintf("lw: Error: Cannot read from DMEM. Reads go out-of-"
                "bounds.\n");
        return 0;
    }

    //
    // Build the DMEMC command that auto-increments on each read
    // We take the address and mask it off with OFFSET and BLOCK region.
    //
    // Note: We also remember and restore the original command value
    //
    dmemc = (addr & (DRF_SHIFTMASK(LW_PPWR_FALCON_DMEMC_OFFS)   |
                     DRF_SHIFTMASK(LW_PPWR_FALCON_DMEMC_BLK)))  |
                     DRF_NUM(_PPWR, _FALCON_DMEMC, _AINCR, 0x1) |
                     (((bIsAddrVa && (addr >= _pmuGetDmemVaBound_GP102()))) ?
                        DRF_DEF(_PPWR, _FALCON_DMEMC, _VA, _TRUE) :
                        DRF_DEF(_PPWR, _FALCON_DMEMC, _VA, _FALSE));

    dmemcOrig = PMU_REG_RD32(LW_PPWR_FALCON_DMEMC(port));
    PMU_REG_WR32(LW_PPWR_FALCON_DMEMC(port), dmemc);

    // Perform the actual DMEM read operations
    for (i = 0; i < length; i++)
    {
        pDmem[i] = PMU_REG_RD32(LW_PPWR_FALCON_DMEMD(port));

        // Check for a miss/multihit/lvlerr
        dmemc = PMU_REG_RD32(LW_PPWR_FALCON_DMEMC(port));
        if (FLD_TEST_DRF(_PPWR, _FALCON_DMEMC, _MISS, _TRUE, dmemc) ||
            FLD_TEST_DRF(_PPWR, _FALCON_DMEMC, _MULTIHIT, _TRUE, dmemc) ||
            FLD_TEST_DRF(_PPWR, _FALCON_DMEMC, _LVLERR, _TRUE, dmemc))
        {
            dprintf("lw:\tCannot read from address 0x%x. (%s)\n",
                    addr + i * 4,
                    FLD_TEST_DRF(_PPWR, _FALCON_DMEMC, _MISS, _TRUE, dmemc) ?
                        "DMEM MISS" :
                    FLD_TEST_DRF(_PPWR, _FALCON_DMEMC, _MULTIHIT, _TRUE, dmemc) ?
                        "DMEM MULTIHIT" : "DMEM LVLERR");
            break;
        }
    }

    // Restore the original DMEMC command
    PMU_REG_WR32(LW_PPWR_FALCON_DMEMC(port), dmemcOrig);
    return i;
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
 *     pmuDmemWrite(0x0, LW_FALSE, 0xa5, 0x2, 0x4, 0x0);
 * @endcode
 *
 * @param[in]  addr      Address in DMEM to write 'val'
 * @param[in]  bIsAddrVa Consider addr as a virtual address
 * @param[in]  val       Value to write
 * @param[in]  width     width of 'val'; 1=byte, 2=half-word, 4=word
 * @param[in]  length    Number of writes to perform
 * @param[in]  port      DMEM port to use for when reading and writing DMEM
 *
 * @return The number of bytes written; zero denotes a failure.
 */
LwU32
pmuDmemWrite_GP102
(
    LwU32 addr,
    LwBool bIsAddrVa,
    LwU32 val,
    LwU32 width,
    LwU32 length,
    LwU32 port
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

    // Fail if the address is out of range
    if (!bIsAddrVa && (addr >= dmemSize))
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
    if (!bIsAddrVa && ((addr + (length * width)) > dmemSize))
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
        if (!_pmuDmemWrite_GP102(addr + (i * width), bIsAddrVa, val, width, port))
        {
            // Can't write. Give up.
            break;
        }
    }
    PMU_REG_WR32(LW_PPWR_FALCON_DMEMC(port), dmemcOrig);
    return i * width;
}

const char *
pmuUcodeName_GP102()
{
    return "g_c85b6_gp10x";
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
static LwBool
_pmuDmemWrite_GP102
(
    LwU32  addr,
    LwBool bIsAddrVa,
    LwU32  val,
    LwU32  width,
    LwU32  port
)
{
    LwU32 unaligned;
    LwU32 addrAligned;
    LwU32 data32;
    LwU32 andMask;
    LwU32 lshift;
    LwU32 overflow = 0;
    LwU32 val2     = 0;
    LwU32 dmemc;

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
        (addrAligned & (DRF_SHIFTMASK(LW_PPWR_FALCON_DMEMC_OFFS)|
                   DRF_SHIFTMASK(LW_PPWR_FALCON_DMEMC_BLK)))    |
        DRF_NUM(_PPWR, _FALCON_DMEMC, _AINCR, 0x0)  |
        DRF_NUM(_PPWR, _FALCON_DMEMC, _AINCW, 0x1)  |
        ((bIsAddrVa && (addr >= _pmuGetDmemVaBound_GP102())) ?
                DRF_DEF(_PPWR, _FALCON_DMEMC, _VA, _TRUE) :
                DRF_DEF(_PPWR, _FALCON_DMEMC, _VA, _FALSE)));

    //
    // Read-in the current 4-byte value and mask off the portion that will
    // be overwritten by this write.
    //
    data32  = PMU_REG_RD32(LW_PPWR_FALCON_DMEMD(port));

    dmemc = PMU_REG_RD32(LW_PPWR_FALCON_DMEMC(port));
    if (FLD_TEST_DRF(_PPWR, _FALCON_DMEMC, _MISS, _TRUE, dmemc) ||
        FLD_TEST_DRF(_PPWR, _FALCON_DMEMC, _MULTIHIT, _TRUE, dmemc) ||
        FLD_TEST_DRF(_PPWR, _FALCON_DMEMC, _LVLERR, _TRUE, dmemc))
    {
        dprintf("lw:\tCannot write to address 0x%x. (%s)\n",
                addrAligned,
                FLD_TEST_DRF(_PPWR, _FALCON_DMEMC, _MISS, _TRUE, dmemc) ?
                    "DMEM MISS" :
                FLD_TEST_DRF(_PPWR, _FALCON_DMEMC, _MULTIHIT, _TRUE, dmemc) ?
                    "DMEM MULTIHIT" : "DMEM LVLERR");
        return LW_FALSE;
    }

    data32 &= andMask;
    data32 |= (val << lshift);
    PMU_REG_WR32(LW_PPWR_FALCON_DMEMD(port), data32);

    if (overflow != 0)
    {
        addrAligned += 4;
        andMask      = ~(BIT(8 * overflow) - 1);

        data32  = PMU_REG_RD32(LW_PPWR_FALCON_DMEMD(port));

        dmemc = PMU_REG_RD32(LW_PPWR_FALCON_DMEMC(port));
        if (FLD_TEST_DRF(_PPWR, _FALCON_DMEMC, _MISS, _TRUE, dmemc) ||
            FLD_TEST_DRF(_PPWR, _FALCON_DMEMC, _MULTIHIT, _TRUE, dmemc) ||
            FLD_TEST_DRF(_PPWR, _FALCON_DMEMC, _LVLERR, _TRUE, dmemc))
        {
            dprintf("lw:\tCannot write to address 0x%x. (%s)\n",
                    addrAligned,
                    FLD_TEST_DRF(_PPWR, _FALCON_DMEMC, _MISS, _TRUE, dmemc) ?
                        "DMEM MISS" :
                    FLD_TEST_DRF(_PPWR, _FALCON_DMEMC, _MULTIHIT, _TRUE, dmemc) ?
                        "DMEM MULTIHIT" : "DMEM LVLERR");
            return LW_FALSE;
        }

        data32 &= andMask;
        data32 |= val2;
        PMU_REG_WR32(LW_PPWR_FALCON_DMEMD(port), data32);
    }

    return LW_TRUE;
}

/*!
 * Retrieve the virtual address that is the threshold for VA to PA mapping.
 * For a virtual address below this threshold, PA = VA.
 *
 * @return  The DMEM VA bound
 */
LwU32
_pmuGetDmemVaBound_GP102(void)
{
    LwU32 boundInBlocks = DRF_VAL(_PPWR, _FALCON_DMVACTL, _BOUND,
                                  PMU_REG_RD32(LW_PPWR_FALCON_DMVACTL));
    return boundInBlocks * PMU_DMEM_BLOCK_SIZE_BYTES;
}
