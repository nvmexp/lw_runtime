/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2016-2018 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

/* ------------------------ Includes --------------------------------------- */
#include "pascal/gp102/dev_falcon_v4.h"
#include "falcon.h"

/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Static variables ------------------------------- */
/* ------------------------ Function Prototypes ---------------------------- */
static LwBool _flcnDmemWrite_v06_00(LwU32  engineBase, LwU32  addr, LwBool bIsAddrVa, LwU32  val, LwU32  width, LwU32  port);

/*!
 *  Read length words of DMEM starting at offset addr. Addr will automatically
 *  be truncated down to 4-byte aligned value. If length spans out of the range
 *  of the DMEM, it will automatically  be truncated to fit into the DMEM 
 *  range.
 * 
 *  @param engineBase  Base address of the Falcon engine
 *  @param addr        Offset into the DMEM to start reading.
 *  @param bIsAddrVa   Addr is a VA (ignored if DMEM VA is not enabled).
 *  @param length      Number of 4-byte words to read.
 *  @param port        Port to read from.
 *  @param dmem        Buffer to store DMEM into.
 *
 *  @return 0 on error, or number of 4-byte words read.
 */
LwU32
flcnDmemRead_v06_00
(
    LwU32  engineBase,    
    LwU32  addr,
    LwBool bIsAddrVa,
    LwU32  length,
    LwU32  port,
    LwU32* pDmem
)
{
    const FLCN_CORE_IFACES* pFCIF = &(flcnCoreIfaces_v06_00);
    LwU32    dmemSize    = 0x0;
    LwU32    dmemcOrig   = 0x0;
    LwU32    dmemc       = 0x0;
    LwU32    maxPort     = 0x0;
    LwU32    i           = 0x0;
    LwU32    dmemVaBound = 0x0;

    addr    &= ~(sizeof(LwU32) - 1);
    dmemSize = pFCIF->flcnDmemGetSize(engineBase);
    maxPort  = pFCIF->flcnDmemGetNumPorts(engineBase) - 1;

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

    // Fail if the read will go beyond the end of DMEM
    if (!bIsAddrVa && ((addr + (length * sizeof(LwU32))) > dmemSize))
    {
        dprintf("lw:\tError: Attempt to read beyond the end of DMEM.\n");
        return 0;
    }

    // Fail if we're trying to read across the DMEM VA bound
    dmemVaBound = pFCIF->flcnDmemVaBoundaryGet(engineBase);
    if (bIsAddrVa &&
        (addr < dmemVaBound) && ((addr + length * 4) > dmemVaBound))
    {
        dprintf("lw:\tError: Attempt to read across DMEM VA boundary.\n");
        return 0;
    }

    //
    // Build the DMEMC command that auto-increments on each read
    // We take the address and mask it off with OFFSET and BLOCK region.
    // 
    // Note: We also remember and restore the original command value
    //
    dmemc = (addr & (DRF_SHIFTMASK(LW_PFALCON_FALCON_DMEMC_OFFS)  |
                     DRF_SHIFTMASK(LW_PFALCON_FALCON_DMEMC_BLK))) |
                     DRF_NUM(_PFALCON, _FALCON_DMEMC, _AINCR, 0x1)|
                     ((bIsAddrVa && (addr >= dmemVaBound)) ?
                        DRF_DEF(_PFALCON, _FALCON_DMEMC, _VA, _TRUE) :
                        DRF_DEF(_PFALCON, _FALCON_DMEMC, _VA, _FALSE));

    dmemcOrig = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEMC(port));
    FLCN_REG_WR32(LW_PFALCON_FALCON_DMEMC(port), dmemc);

    // Perform the actual DMEM read operations
    for (i = 0; i < length; i++)
    {
        pDmem[i] = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEMD(port));

        // Check for a miss/multihit/lvlerr
        dmemc = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEMC(port));
        if (FLD_TEST_DRF(_PFALCON, _FALCON_DMEMC, _MISS, _TRUE, dmemc) ||
            FLD_TEST_DRF(_PFALCON, _FALCON_DMEMC, _MULTIHIT, _TRUE, dmemc) ||
            FLD_TEST_DRF(_PFALCON, _FALCON_DMEMC, _LVLERR, _TRUE, dmemc))
        {
            dprintf("lw:\tCannot read from address 0x%x. (%s)\n",
                    addr + i * 4,
                    FLD_TEST_DRF(_PFALCON, _FALCON_DMEMC, _MISS, _TRUE, dmemc) ?
                        "DMEM MISS" :
                    FLD_TEST_DRF(_PFALCON, _FALCON_DMEMC, _MULTIHIT, _TRUE, dmemc) ?
                        "DMEM MULTIHIT" : "DMEM LVLERR");
            break;
        }
    }

    // Restore the original DMEMC command
    FLCN_REG_WR32(LW_PFALCON_FALCON_DMEMC(port), dmemcOrig);
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
 *     // Write '0xffff' four times starting at address 0x4000, and advancing
 *     // the address by 2-bytes per write.
 *     //
 *     flcnDmemWrite(base, LW_FALSE, 0x4000, 0xffff, 0x2, 0x4, 0x0);
 * @endcode
 *
 * @param addr        Address in DMEM to write 'val'
 * @param bIsAddrVa   Addr is a VA (ignored if DMEM VA is not enabled)
 * @param val         Value to write
 * @param width       width of 'val'; 1=byte, 2=half-word, 4=word
 * @param length      Number of writes to perform
 * @param port        DMEM port to use for when reading and writing DMEM
 *
 * @return The number of bytes written; zero denotes a failure.
 */
LwU32
flcnDmemWrite_v06_00
(
    LwU32  engineBase,
    LwU32  addr,
    LwBool bIsAddrVa,
    LwU32  val,
    LwU32  width,
    LwU32  length,
    LwU32  port
)
{
    const FLCN_CORE_IFACES *pFCIF = &(flcnCoreIfaces_v06_00);
    LwU32  dmemSize  = 0x0;
    LwU32  dmemcOrig = 0x0;
    LwU32  maxPort   = 0x0;
    LwU32  i;
    LwU32  dmemVaBound;

    dmemSize = pFCIF->flcnDmemGetSize(engineBase);
    maxPort  = pFCIF->flcnDmemGetNumPorts(engineBase) - 1;
    
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

    // Fail if the write will go beyond the end of DMEM
    if (!bIsAddrVa && ((addr + (length * width)) > dmemSize))
    {
        dprintf("lw:\tError: Attempt to write beyond the end of DMEM.\n");
        return 0;
    }

    // Fail if we're writing across the DMEM VA bound
    dmemVaBound = pFCIF->flcnDmemVaBoundaryGet(engineBase);
    if (bIsAddrVa &&
        (addr < dmemVaBound) && ((addr + (length * width)) > dmemVaBound))
    {
        dprintf("lw: Error: Attempt to write across the DMEM VA bound.\n");
        return 0;
    }

    // Width must be 1, 2, or 4.
    if ((width != 1) && (width != 2) && (width != 4))
    {
        dprintf("lw: Error: Width (%u) must be 1, 2, or 4\n", width);
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
    dmemcOrig = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEMC(port));
    for (i = 0; i < length; i++)
    {
        if (_flcnDmemWrite_v06_00(engineBase, addr + (i * width), bIsAddrVa, val, width, port) == LW_FALSE)
            break;
    }
    FLCN_REG_WR32(LW_PFALCON_FALCON_DMEMC(port), dmemcOrig);    
    return i * width;
}

/*!
 * Write 'val' to DMEM address 'addr'.  'val' may be a byte, half-word, or
 * word based on 'width'. There are no alignment restrictions on the address.
 *
 * @param  addr       Address in DMEM to write 'val'
 * @param  bIsAddrVa  Addr is a VA (ignored if DMEM VA is not enabled)
 * @param  val        Value to write
 * @param  width      width of 'val'; 1=byte, 2=half-word, 4=word
 * @param  port       DMEM port to use for when reading and writing DMEM
 */
LwBool
_flcnDmemWrite_v06_00
(
    LwU32  engineBase,
    LwU32  addr,
    LwBool bIsAddrVa,
    LwU32  val,
    LwU32  width,
    LwU32  port
)
{
    const FLCN_CORE_IFACES* pFCIF = &(flcnCoreIfaces_v06_00);
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

    FLCN_REG_WR32(LW_PFALCON_FALCON_DMEMC(port),
        (addrAligned & (DRF_SHIFTMASK(LW_PFALCON_FALCON_DMEMC_OFFS)   |
                        DRF_SHIFTMASK(LW_PFALCON_FALCON_DMEMC_BLK)))  |
                        DRF_NUM(_PFALCON, _FALCON_DMEMC, _AINCR, 0x0) |
                        DRF_NUM(_PFALCON, _FALCON_DMEMC, _AINCW, 0x1) |
                        ((bIsAddrVa && (addr >= pFCIF->flcnDmemVaBoundaryGet(engineBase))) ?
                            DRF_DEF(_PFALCON, _FALCON_DMEMC, _VA, _TRUE) :
                            DRF_DEF(_PFALCON, _FALCON_DMEMC, _VA, _FALSE)));

    // 
    // Read-in the current 4-byte value and mask off the portion that will
    // be overwritten by this write.
    //
    data32  = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEMD(port));

    dmemc = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEMC(port));
    if (FLD_TEST_DRF(_PFALCON, _FALCON_DMEMC, _MISS, _TRUE, dmemc) ||
        FLD_TEST_DRF(_PFALCON, _FALCON_DMEMC, _MULTIHIT, _TRUE, dmemc) ||
        FLD_TEST_DRF(_PFALCON, _FALCON_DMEMC, _LVLERR, _TRUE, dmemc))
    {
        dprintf("lw:\tCannot write to address 0x%x. (%s)\n",
                addrAligned,
                FLD_TEST_DRF(_PFALCON, _FALCON_DMEMC, _MISS, _TRUE, dmemc) ?
                    "DMEM MISS" :
                FLD_TEST_DRF(_PFALCON, _FALCON_DMEMC, _MULTIHIT, _TRUE, dmemc) ?
                    "DMEM MULTIHIT" : "DMEM LVLERR");
        return LW_FALSE;
    }

    data32 &= andMask;
    data32 |= (val << lshift);
    FLCN_REG_WR32(LW_PFALCON_FALCON_DMEMD(port), data32);

    if (overflow != 0)
    {
        addrAligned += 4;
        andMask      = ~((1 << (8 * overflow)) - 1);

        data32  = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEMD(port));

        dmemc = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEMC(port));
        if (FLD_TEST_DRF(_PFALCON, _FALCON_DMEMC, _MISS, _TRUE, dmemc) ||
            FLD_TEST_DRF(_PFALCON, _FALCON_DMEMC, _MULTIHIT, _TRUE, dmemc) ||
            FLD_TEST_DRF(_PFALCON, _FALCON_DMEMC, _LVLERR, _TRUE, dmemc))
        {
            dprintf("lw:\tCannot write to address 0x%x. (%s)\n",
                    addrAligned,
                    FLD_TEST_DRF(_PFALCON, _FALCON_DMEMC, _MISS, _TRUE, dmemc) ?
                        "DMEM MISS" :
                    FLD_TEST_DRF(_PFALCON, _FALCON_DMEMC, _MULTIHIT, _TRUE, dmemc) ?
                        "DMEM MULTIHIT" : "DMEM LVLERR");
            return LW_FALSE;
        }

        data32 &= andMask;
        data32 |= val2;
        FLCN_REG_WR32(LW_PFALCON_FALCON_DMEMD(port), data32);
    }
    return LW_TRUE;
}
