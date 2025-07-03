/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2012-2018 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

/* ------------------------ Includes --------------------------------------- */
#include "kepler/gk104/dev_falcon_v4.h"
#include "falcon.h"
#include "gpuanalyze.h"

#include "g_falcon_private.h"     // (rmconfig)  implementation prototypes


/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Static variables ------------------------------- */
/* ------------------------ Function Prototypes ---------------------------- */
void        _flcnDmemWrite_v04_00 (LwU32, LwU32, LwU32, LwU32, LwU32);
void        _flcnImemWrite_v04_00 (LwU32, LwU32, LwU32, LwU32, LwU32);
void        _flcnImemWriteBuf_v04_00 (LwU32, LwU32, LwU32 *, LwU32, LwBool);


/*!
 *  Return the DMEM size 
 *
 *  @param engineBase  Base address of the Falcon engine
 *
 *  @return Size in bytes of the Falcon Engine DMEM.
 */
LwU32
flcnDmemGetSize_v04_00
(
    LwU32 engineBase
)
{
    return DRF_VAL(_PFALCON_FALCON, _HWCFG, _DMEM_SIZE,
                   FLCN_REG_RD32(LW_PFALCON_FALCON_HWCFG)) << 8;
}

/*!
 *  Return the DMEM ports
 * 
 *  @param engineBase  Base address of the Falcon engine
 *
 *  @return Number of Falcon engine DMEM ports.
 */
LwU32
flcnDmemGetNumPorts_v04_00
(
    LwU32 engineBase
)
{
    return DRF_VAL(_PFALCON_FALCON, _HWCFG1, _DMEM_PORTS,
                   FLCN_REG_RD32(LW_PFALCON_FALCON_HWCFG1));
}

/*!
 *  Read length words of DMEM starting at offset addr. Addr will automatically
 *  be truncated down to 4-byte aligned value. If length spans out of the range
 *  of the DMEM, it will automatically  be truncated to fit into the DMEM 
 *  range.
 * 
 *  @param engineBase  Base address of the Falcon engine
 *  @param addr        Offset into the DMEM to start reading.
 *  @param bIsAddrVa   Consider addr as a VA (ignored if DMEM VA is not enabled)
 *  @param length      Number of 4-byte words to read.
 *  @param port        Port to read from.
 *  @param dmem        Buffer to store DMEM into.
 *
 *  @return 0 on error, or number of 4-byte words read.
 */
LwU32
flcnDmemRead_v04_00
(
    LwU32  engineBase,    
    LwU32  addr,
    LwBool bIsAddrVa,
    LwU32  length,
    LwU32  port,
    LwU32* pDmem
)
{
    const FLCN_CORE_IFACES* pFCIF = &(flcnCoreIfaces_v04_00);
    LwU32    dmemSize    = 0x0;
    LwU32    dmemcOrig   = 0x0;
    LwU32    dmemc       = 0x0;
    LwU32    maxPort     = 0x0;
    LwU32    i           = 0x0;

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
    dmemc = (addr & (DRF_SHIFTMASK(LW_PFALCON_FALCON_DMEMC_OFFS)  |
                     DRF_SHIFTMASK(LW_PFALCON_FALCON_DMEMC_BLK))) |
                     DRF_NUM(_PFALCON, _FALCON_DMEMC, _AINCR, 0x1);
    dmemcOrig = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEMC(port));
    FLCN_REG_WR32(LW_PFALCON_FALCON_DMEMC(port), dmemc);

    // Perform the actual DMEM read operations
    for (i = 0; i < length; i++)
    {
        pDmem[i] = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEMD(port));
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
 *     flcnDmemWrite(base, 0x4000, 0xffff, 0x2, 0x4, 0x0);
 * @endcode
 *
 * @param[in]  addr    Address in DMEM to write 'val'
 * @param bIsAddrVa   Consider addr as a VA (ignored if DMEM VA is not enabled)
 * @param[in]  val     Value to write
 * @param[in]  width   width of 'val'; 1=byte, 2=half-word, 4=word
 * @param[in]  length  Number of writes to perform
 * @param[in]  port    DMEM port to use for when reading and writing DMEM
 *
 * @return The number of bytes written; zero denotes a failure.
 */
LwU32
flcnDmemWrite_v04_00
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
    const FLCN_CORE_IFACES *pFCIF = &(flcnCoreIfaces_v04_00);
    LwU32  dmemSize  = 0x0;
    LwU32  dmemcOrig = 0x0;
    LwU32  maxPort   = 0x0;
    LwU32  i;

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
    dmemcOrig = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEMC(port));
    for (i = 0; i < length; i++)
    {
        _flcnDmemWrite_v04_00(engineBase, addr + (i * width), val, width, port);
    }
    FLCN_REG_WR32(LW_PFALCON_FALCON_DMEMC(port), dmemcOrig);    
    return length * width;
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
void
_flcnDmemWrite_v04_00
(
    LwU32  engineBase,
    LwU32  addr,
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
                        DRF_NUM(_PFALCON, _FALCON_DMEMC, _AINCW, 0x1));

    // 
    // Read-in the current 4-byte value and mask off the portion that will
    // be overwritten by this write.
    //
    data32  = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEMD(port));
    data32 &= andMask;
    data32 |= (val << lshift);
    FLCN_REG_WR32(LW_PFALCON_FALCON_DMEMD(port), data32);

    if (overflow != 0)
    {
        addrAligned += 4;
        andMask      = ~((1 << (8 * overflow)) - 1);

        data32  = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEMD(port));
        data32 &= andMask;
        data32 |= val2;
        FLCN_REG_WR32(LW_PFALCON_FALCON_DMEMD(port), data32);
    }
}



/*!
 * @return Size in bytes of the Falcon engine IMEM.
 */
LwU32
flcnImemGetSize_v04_00
(
    LwU32  engineBase
)
{
    return DRF_VAL(_PFALCON_FALCON, _HWCFG, _IMEM_SIZE,
                   FLCN_REG_RD32(LW_PFALCON_FALCON_HWCFG)) << 8;
}

/*!
 * @return Number of IMEM blocks in Falcon engine.
 */
LwU32
flcnImemGetNumBlocks_v04_00
(
    LwU32  engineBase
)
{
     return DRF_VAL(_PFALCON_FALCON, _HWCFG, _IMEM_SIZE,
                   FLCN_REG_RD32(LW_PFALCON_FALCON_HWCFG));
}

/*!
 * @return Number of Falcon engine IMEM ports.
 */
LwU32
flcnImemGetNumPorts_v04_00
(
    LwU32  engineBase
)
{
    return DRF_VAL(_PFALCON_FALCON, _HWCFG1, _IMEM_PORTS,
                   FLCN_REG_RD32(LW_PFALCON_FALCON_HWCFG1));
}

/*!
 * Read length words of IMEM starting at offset addr. Addr will automatically
 * be truncated down to 4-byte aligned value. If length spans out of the range
 * of the IMEM, it will automatically be truncated to fit into the IMEM range.
 *
 * @param addr   Offset into the IMEM to start reading.
 * @param length Number of 4-byte words to read.
 * @param port   Port to read from.
 * @param imem   Buffer to store IMEM into.
 *
 * @return 0 on error, or number of 4-byte words read.
 */
LwU32
flcnImemRead_v04_00
(
    LwU32  engineBase,
    LwU32  addr,
    LwU32  length,
    LwU32  port,
    LwU32* pImem
)
{
    const FLCN_CORE_IFACES *pFCIF = &(flcnCoreIfaces_v04_00);
    LwU32    imemSize    = 0x0;
    LwU32    imemcOrig   = 0x0;
    LwU32    imemc       = 0x0;
    LwU32    maxPort     = 0x0;
    LwU32    i           = 0x0;
    
    addr    &= ~(sizeof(LwU32) - 1);
    imemSize = pFCIF->flcnImemGetSize(engineBase);
    maxPort  = pFCIF->flcnImemGetNumPorts(engineBase) - 1;
    
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
    imemc = (addr & (DRF_SHIFTMASK(LW_PFALCON_FALCON_IMEMC_OFFS)  |
                     DRF_SHIFTMASK(LW_PFALCON_FALCON_IMEMC_BLK))) |
                     DRF_NUM(_PFALCON, _FALCON_IMEMC, _AINCR, 0x1);
    imemcOrig = FLCN_REG_RD32(LW_PFALCON_FALCON_IMEMC(port));
    FLCN_REG_WR32(LW_PFALCON_FALCON_IMEMC(port), imemc);

    // Perform the actual IMEM read operations
    for (i = 0; i < length; i++)
    {
        pImem[i] = FLCN_REG_RD32(LW_PFALCON_FALCON_IMEMD(port));
    }

    // Restore the original IMEMC command
    FLCN_REG_WR32(LW_PFALCON_FALCON_IMEMC(port), imemcOrig);
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
 *     // Write '0xffff' four times starting at address 0x2000, and advancing
 *     // the address by 2-bytes per write.
 *     //
 *     flcnImemWrite(0x2000, 0xffff, 0x2, 0x4, 0x0);
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
flcnImemWrite_v04_00
(
    LwU32  engineBase,
    LwU32  addr,
    LwU32  val,
    LwU32  width,
    LwU32  length,
    LwU32  port
)
{
    const FLCN_CORE_IFACES *pFCIF = &(flcnCoreIfaces_v04_00);
    LwU32  imemSize  = 0x0;
    LwU32  imemcOrig = 0x0;
    LwU32  maxPort   = 0x0;
    LwU32  i;

    imemSize = pFCIF->flcnImemGetSize(engineBase);
    maxPort  = pFCIF->flcnImemGetNumPorts(engineBase) - 1;
    
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
    imemcOrig = FLCN_REG_RD32(LW_PFALCON_FALCON_IMEMC(port));
    for (i = 0; i < length; i++)
    {
        _flcnImemWrite_v04_00(engineBase, addr + (i * width), val, width, port);
    }
    FLCN_REG_WR32(LW_PFALCON_FALCON_IMEMC(port), imemcOrig);    
    return length * width;
}

/*!
 *  Return the IMEM tag width in bits
 * 
 *  @param[in] engineBase  Base address of the Falcon engine
 *
 *  @return Width of the tag in bits.
 */
LwU32
flcnImemGetTagWidth_v04_00
(
    LwU32 engineBase
)
{
    return DRF_VAL(_PFALCON_FALCON, _HWCFG1, _TAG_WIDTH,
                   FLCN_REG_RD32(LW_PFALCON_FALCON_HWCFG1));
}

/*!
 *  Return the IMEM block info given a block index
 * 
 *  @param[in]  engineBase  Base address of the Falcon engine
 *  @param[in]  blockIndex  The index of interested block
 *  @param[out] pBlockInfo  The buffer to store the results
 *
 *  @return True on success, false otherwise.
 */
BOOL
flcnImemBlk_v04_00
(
    LwU32           engineBase,
    LwU32           blockIndex,
    FLCN_BLOCK      *pBlockInfo
)
{
    const FLCN_CORE_IFACES *pFCIF = &(flcnCoreIfaces_v04_00);
    LwU32    numBlocks   = 0x0;
    LwU32    tagWidth    = 0x0;
    LwU32    tagMask     = 0x0;
    LwU32    cmd         = 0x0;
    LwU32    result      = 0x0;


    numBlocks   = pFCIF->flcnImemGetNumBlocks(engineBase);
    tagWidth    = pFCIF->flcnImemGetTagWidth(engineBase);
    tagMask     = BIT(tagWidth) - 1;
  
    // Bail out for invalid block index
    if ((blockIndex >= numBlocks) || (pBlockInfo == NULL))
    {
        return FALSE;
    }
    
    // Create the command, write, and read result
    cmd     = FLD_SET_DRF(_PFALCON_FALCON, _IMCTL, _CMD, _IMBLK, blockIndex);
    FLCN_REG_WR32(LW_PFALCON_FALCON_IMCTL, cmd);
    result  = FLCN_REG_RD32(LW_PFALCON_FALCON_IMSTAT);
    
    // Extract the TAG value from bits 8 to 7+TagWidth
    pBlockInfo->tag         = (result >> 8) & tagMask;    
    pBlockInfo->bValid      = DRF_VAL(_PFALCON_FALCON, _IMBLK, _VALID, result);
    pBlockInfo->bPending    = DRF_VAL(_PFALCON_FALCON, _IMBLK, _PENDING, result);
    pBlockInfo->bSelwre     = DRF_VAL(_PFALCON_FALCON, _IMBLK, _SELWRE, result); 
    
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
 *  @param[in]  engineBase      Base address of the engine
 *  @param[in]  codeAddr        Block address to look up mapped block status.
 *  @param[out] pTagInfo        Pointer to tag mapping structure to store
 *                              information regarding block mapped to tag.
 *
 *  @return FALSE on error, TRUE on success.
 */

BOOL
flcnImemTag_v04_00
(
    LwU32           engineBase,
    LwU32           codeAddr,
    FLCN_TAG*       pTagInfo
)
{
    
    const FLCN_CORE_IFACES *pFCIF = &(flcnCoreIfaces_v04_00);
    LwU32    numBlocks   = 0x0;
    LwU32    blockMask   = 0x0;
    LwU32    blockIndex  = 0x0;
    LwU32    valid       = 0x0;
    LwU32    pending     = 0x0;
    LwU32    secure      = 0x0;
    LwU32    multiHit    = 0x0;
    LwU32    miss        = 0x0;
    LwU32    tagWidth    = 0x0;
    LwU32    maxAddr     = 0x0;
    LwU32    cmd         = 0x0;
    LwU32    result      = 0x0;

    // Quick check of argument pointer
    if (pTagInfo == NULL)
    {
        return FALSE;
    }

    numBlocks   = pFCIF->flcnImemGetNumBlocks(engineBase);
    tagWidth    = pFCIF->flcnImemGetTagWidth(engineBase);
    maxAddr     = (BIT(tagWidth) << 8) - 1;

    blockMask   = numBlocks;
    ROUNDUP_POW2(blockMask);
    blockMask--;

    //
    // Create the command, write, and read result
    // Command is created by taking:
    //       Bits T+7 - 0: Address
    //       Upper Bits  : LW_PFALCON_FALCON_IMCTL_CMD_IMTAG
    // Result is fetched from IMSTAT register
    //
    cmd     = FLD_SET_DRF(_PFALCON_FALCON, _IMCTL, _CMD, _IMTAG, codeAddr & maxAddr);
    FLCN_REG_WR32(LW_PFALCON_FALCON_IMCTL, cmd);
    result  = FLCN_REG_RD32(LW_PFALCON_FALCON_IMSTAT);
    
    // Extract the block index and other information
    blockIndex  = result & blockMask;
    valid       = DRF_VAL(_PFALCON_FALCON, _IMTAG, _VALID, result);
    pending     = DRF_VAL(_PFALCON_FALCON, _IMTAG, _PENDING, result);
    secure      = DRF_VAL(_PFALCON_FALCON, _IMTAG, _SELWRE, result);
    multiHit    = DRF_VAL(_PFALCON_FALCON, _IMTAG, _MULTI_HIT, result);
    miss        = DRF_VAL(_PFALCON_FALCON, _IMTAG, _MISS, result);
    
    if (miss)
    {
        pTagInfo->mapType = FALCON_TAG_UNMAPPED;
    }
    else if (multiHit)
    {
        pTagInfo->mapType = FALCON_TAG_MULTI_MAPPED;
    }     
    else
    {
        pTagInfo->mapType = FALCON_TAG_MAPPED;
    }

    pTagInfo->blockInfo.tag         = (codeAddr & maxAddr) >> 8;
    pTagInfo->blockInfo.blockIndex  = blockIndex;
    pTagInfo->blockInfo.bPending    = pending;
    pTagInfo->blockInfo.bValid      = valid;
    pTagInfo->blockInfo.bSelwre     = secure;
    return TRUE;
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
void
_flcnImemWrite_v04_00
(
    LwU32  engineBase,
    LwU32  addr,
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

    FLCN_REG_WR32(LW_PFALCON_FALCON_IMEMC(port),
        (addrAligned & (DRF_SHIFTMASK(LW_PFALCON_FALCON_IMEMC_OFFS)   |
                        DRF_SHIFTMASK(LW_PFALCON_FALCON_IMEMC_BLK)))  |
                        DRF_NUM(_PFALCON, _FALCON_IMEMC, _AINCR, 0x0) |
                        DRF_NUM(_PFALCON, _FALCON_IMEMC, _AINCW, 0x1));
    
    // 
    // Read-in the current 4-byte value and mask off the portion that will
    // be overwritten by this write.
    //
    data32  = FLCN_REG_RD32(LW_PFALCON_FALCON_IMEMD(port));
    data32 &= andMask;
    data32 |= (val << lshift);
    FLCN_REG_WR32(LW_PFALCON_FALCON_IMEMD(port), data32);

    if (overflow != 0)
    {
        addrAligned += 4;
        andMask      = ~((1 << (8 * overflow)) - 1);

        data32  = FLCN_REG_RD32(LW_PFALCON_FALCON_IMEMD(port));
        data32 &= andMask;
        data32 |= val2;
        FLCN_REG_WR32(LW_PFALCON_FALCON_IMEMD(port), data32);
    }
}

/*!
 * Write multiple blocks of 256B each into IMEM. This function is meant to be used
 * for loading UCODE with size normally aligned to 256B.
 *
 * @param[in]  addr      Address in IMEM (256B aligned)
 * @param[in]  startPC   Starting PC for this imem data
 * @param[in]  pInBuf    Input buffer
 * @param[in]  size      Size of input buffer in bytes (256B aligned)
 * @param[in]  port      IMEM port to use for when reading and writing IMEM
 * @param[in]  bIsSelwre Is Ucode secure?
 *
 * @return The number of bytes written; zero denotes a failure.
 */
LwU32
flcnImemWriteBuf_v04_00
(
    LwU32  engineBase,
    LwU32  addr,
    LwU32  startPC,
    LwU32  *pInBuf,
    LwU32  size,
    LwU32  port,
    LwBool bIsSelwre
)
{
    const FLCN_CORE_IFACES *pFCIF = &(flcnCoreIfaces_v04_00);
    LwU32  imemSize               = 0x0;
    LwU32  imemcOrig              = 0x0;
    LwU32  maxPort                = 0x0;
    LwU32  bytesCopied            = 0x0;
    LwU32  i                      = 0x0;
    LwU32  tag                    = 0x0;

    // Need 256B aligned to make sure secure code is loaded properly
    size = LW_ALIGN_UP(size, 256);

    //Check the alignment first
    if ((addr % 256 ) || (startPC % 256))
    {
        // Unaligned address 
        dprintf("lw:\tUnaligned address 0x%0x  or startPC 0x%0x in flcnImemWriteBuf\n",
                addr, startPC);
        return 0;
    }

    imemSize = pFCIF->flcnImemGetSize(engineBase);
    maxPort  = pFCIF->flcnImemGetNumPorts(engineBase) - 1;

    // Fail if the port specified is not valid
    if (port > maxPort)
    {
        dprintf("lw:\tport 0x%x is out of range (max 0x%x).\n",
                port, maxPort);
        return 0;
    }

    // Fail if the address is out of range
    if ((addr >= imemSize) || ((addr + size) >= imemSize))
    {
        dprintf("lw:\taddress 0x%x is out of range (max 0x%x).\n",
                addr, imemSize - 1);
        return 0;
    }

    // save the current IMEMC register value (to be restored later)
    imemcOrig = FLCN_REG_RD32(LW_PFALCON_FALCON_IMEMC(port));

    // Start copying the content to IMEM
    FLCN_REG_WR32(LW_PFALCON_FALCON_IMEMC(port),
        (addr & (DRF_SHIFTMASK(LW_PFALCON_FALCON_IMEMC_OFFS)   |
                 DRF_SHIFTMASK(LW_PFALCON_FALCON_IMEMC_BLK)))  |
                 DRF_NUM(_PFALCON, _FALCON_IMEMC, _AINCR, 0x0) |
                 DRF_NUM(_PFALCON, _FALCON_IMEMC, _AINCW, 0x1) |
                 DRF_NUM(_PFALCON, _FALCON_IMEMC, _SELWRE, (bIsSelwre)?1:0)); 

    tag = startPC >> 8;
    for (i = 0; i < (size/4); i++)
    {
        // Set tag
        if ((i % 64) == 0)
        {
            FLCN_REG_WR32(LW_PFALCON_FALCON_IMEMT(port), DRF_NUM(_PFALCON,
                    _FALCON_IMEMT, _TAG, tag));
            tag++;
        }
        FLCN_REG_WR32(LW_PFALCON_FALCON_IMEMD(port), pInBuf[i]);
        bytesCopied += 4;
    }

    // Restore saved IMEMC
    FLCN_REG_WR32(LW_PFALCON_FALCON_IMEMC(port), imemcOrig);

    return bytesCopied;
}

/*!
 * Write multiple blocks of 256B each into DMEM. This function is meant to be used
 * for loading UCODE DATA with size normally aligned to 4B.
 *
 * @param[in]  addr      Address in DMEM (256B aligned)
 * @param[in]  pInBuf    Input buffer
 * @param[in]  size      Size of input buffer in bytes (4B aligned)
 * @param[in]  port      DMEM port to use for when reading and writing IMEM
 *
 * @return The number of bytes written; zero denotes a failure.
 */
LwU32
flcnDmemWriteBuf_v04_00
(
    LwU32  engineBase,
    LwU32  addr,
    LwU32  *pInBuf,
    LwU32  size,
    LwU32  port
)
{
    const FLCN_CORE_IFACES *pFCIF = &(flcnCoreIfaces_v04_00);
    LwU32  dmemSize               = 0x0;
    LwU32  dmemcOrig              = 0x0;
    LwU32  maxPort                = 0x0;
    LwU32  bytesCopied            = 0x0;
    LwU32  i                      = 0x0;

    // Align size to 4
    size = LW_ALIGN_UP(size, 4);

    //Check the alignment first
    if ((addr % 4 ) || (size % 4))
    {
        // Unaligned address or size
        dprintf("lw:\tUnaligned address 0x%0x in flcnDmemWriteBuf\n",
                addr);
        return 0;
    }

    dmemSize = pFCIF->flcnDmemGetSize(engineBase);
    maxPort  = pFCIF->flcnDmemGetNumPorts(engineBase) - 1;

    // Fail if the port specified is not valid
    if (port > maxPort)
    {
        dprintf("lw:\tport 0x%x is out of range (max 0x%x).\n",
                port, maxPort);
        return 0;
    }

    // Fail if the address is out of range
    if ((addr >= dmemSize) || ((addr + size) >= dmemSize))
    {
        dprintf("lw:\taddress 0x%x is out of range (max 0x%x).\n",
                addr, dmemSize - 1);
        return 0;
    }

    // save the current DMEMC register value (to be restored later)
    dmemcOrig = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEMC(port));

    // Start copying the content to DMEM
    FLCN_REG_WR32(LW_PFALCON_FALCON_DMEMC(port),
        (addr & (DRF_SHIFTMASK(LW_PFALCON_FALCON_DMEMC_OFFS)   |
                 DRF_SHIFTMASK(LW_PFALCON_FALCON_DMEMC_BLK)))  |
                 DRF_NUM(_PFALCON, _FALCON_DMEMC, _AINCR, 0x0) |
                 DRF_NUM(_PFALCON, _FALCON_DMEMC, _AINCW, 0x1)); 

    for (i = 0; i < (size/4); i++)
    {
        FLCN_REG_WR32(LW_PFALCON_FALCON_DMEMD(port), pInBuf[i]);
        bytesCopied += 4;
    }

    return bytesCopied;
}

/*!
 * Sets IMEM Tag 
 * Calls PFALCON_FALCON_IMEMT(i)  
 *
 * @param[in]  engineBase   Base address of the engine
 * @param[in]  tag          Sets IMEM tag to 'tag' 
 * @param[in]  port         IMEM port to use for when setting IMEM tag 
 *
 * @return TRUE if successful, FALSE otherwise 
 */
LwU32
flcnImemSetTag_v04_00
(
    LwU32  engineBase,
    LwU32  tag,
    LwU32  port
)
{
    const FLCN_CORE_IFACES *pFCIF = &(flcnCoreIfaces_v04_00);
    LwU32  maxPort   = 0x0;

    maxPort  = pFCIF->flcnImemGetNumPorts(engineBase) - 1;

    // Fail if the port specified is not valid
    if (port > maxPort)
    {
        dprintf("lw:\tport 0x%x is out of range (max 0x%x).\n",
                port, maxPort);
        return FALSE;
    }

    FLCN_REG_WR32(LW_PFALCON_FALCON_IMEMT(port), tag);    
    return TRUE; 
}

/*!
 * Return the value of a falcon register denoted by index
 * 
 * @param engineBase  Base address of the Falcon engine
 * @param regIdx      Falcon Register Index
 *
 * @return value of a falcon register specified by regIdx
 */
LwU32
flcnGetRegister_v04_00
(
    LwU32    engineBase,  
    LwU32    regIdx
)
{
    LwU32 icdCmd;

    if (regIdx >= LW_FLCN_REG_SIZE)
    {
        return 0xffffffff;
    }

    icdCmd = DRF_DEF(_PFALCON, _FALCON_ICD_CMD, _OPC, _RREG) |
             DRF_NUM(_PFALCON, _FALCON_ICD_CMD, _IDX, regIdx);

    FLCN_REG_WR32(LW_PFALCON_FALCON_ICD_CMD, icdCmd);

    return FLCN_REG_RD32(LW_PFALCON_FALCON_ICD_RDATA);
}


/*!
 * Return the OS version of Ucode
 * 
 * @param engineBase  Base address of the Falcon engine
 *
 * @return The OS version of Ucode
 */
LwU32
flcnUcodeGetVersion_v04_00
(
    LwU32 engineBase
)
{
    return FLCN_REG_RD32(LW_PFALCON_FALCON_OS);
}

/*!
 *  
 * Start PMU at the given bootvector
 *
 * @param[in]  engineBase   Specifies base address of the Falcon engine
 * @param[in]  bootvector   Specifies bootvector  
 */
void flcnBootstrap_v04_00
(
    LwU32 engineBase,
    LwU32 bootvector
)
{
    // Clear DMACTL 
    FLCN_REG_WR32(LW_PFALCON_FALCON_DMACTL, 0);

    // Set Bootvec
    FLCN_REG_WR32(LW_PFALCON_FALCON_BOOTVEC, 
            DRF_NUM(_PFALCON, _FALCON_BOOTVEC, _VEC, bootvector));

    // Start CPU
    FLCN_REG_WR32(LW_PFALCON_FALCON_CPUCTL, 
            DRF_NUM(_PFALCON, _FALCON_CPUCTL, _STARTCPU, 1));
}

/*!
 * Wait for Falcon to HALT
 *
 * @param[in]  engineBase   Specifies base address of the Falcon engine
 * @param[in]  timeoutUs    Timeout in micro seconds
 */
LW_STATUS flcnWaitForHalt_v04_00
(
    LwU32 engineBase,
    LwS32 timeoutUs 
)
{
    LwU32   cpuCtl    = 0;

    while (timeoutUs > 0)
    {
        cpuCtl = FLCN_REG_RD32(LW_PFALCON_FALCON_CPUCTL);
        if (FLD_TEST_DRF_NUM(_PFALCON, _FALCON_CPUCTL, _HALTED, 0x1, cpuCtl))
        {
            break;
        }
        osPerfDelay(20);
        timeoutUs -= 20;
    }
   
    if (timeoutUs <= 0)
        return LW_ERR_GENERIC;
    return LW_OK;
}

/*!
 *   
 * Returns register map based on falcon engine.
 *
 * @param[in]  engineBase   Specifies base address of the Falcon engine.
 * @param[in, out]  registerMap  FLCNGDB_REGISTER_MAP structure. 
 *
 */
void
flcnGetFlcngdbRegisterMap_v04_00
(
    LwU32                 engineBase,
    FLCNGDB_REGISTER_MAP* registerMap
)
{
    registerMap->registerBase = engineBase;

    registerMap->icdCmd   = LW_PFALCON_FALCON_ICD_CMD + engineBase;
    registerMap->icdAddr  = LW_PFALCON_FALCON_ICD_ADDR + engineBase;
    registerMap->icdWData = LW_PFALCON_FALCON_ICD_WDATA + engineBase;
    registerMap->icdRData = LW_PFALCON_FALCON_ICD_RDATA + engineBase;

    registerMap->numBreakpoints = 2;
    registerMap->firstIBRK = LW_PFALCON_FALCON_IBRKPT1 + engineBase;
}

LwU32
falconTrpcGetMaxIdx_v04_00
(
    LwU32    engineBase
)
{
    return DRF_VAL(_PFALCON, _FALCON_TRACEIDX, _MAXIDX,
                   FLCN_REG_RD32(LW_PFALCON_FALCON_TRACEIDX));
}


LwU32
falconTrpcGetPC_v04_00
(
    LwU32    engineBase,
    LwU32    idx,
    LwU32*   pCount
)
{
    FLCN_REG_WR32(LW_PFALCON_FALCON_TRACEIDX, idx);
    if (pCount != NULL)
    {
        *pCount = 0;
    }
    return DRF_VAL(_PFALCON, _FALCON_TRACEPC, _PC,
                   FLCN_REG_RD32(LW_PFALCON_FALCON_TRACEPC));
}

/*!
 *  Based on _GT212
 *  test of falcon context state: check if it is valid,
 *  fetch chid and see if it is active in host,
 *  check get/put pointers
 *  @param[in] engineBase - base of engine in register space
 *  @param[in] engName - name of the engine, e.g. "PMSPDEC"
 *
 *  @return   LW_OK , LW_ERR_GENERIC.
 */
LW_STATUS falconTestCtxState_v04_00(LwU32 engineBase, char* engName)
{
    PRINT_LWWATCH_NOT_IMPLEMENTED_MESSAGE_AND_RETURN0();
}

//defs from msdecos.h
#define MSDECOS_ERROR_NONE                                         (0x00000000) // default return code for app
#define MSDECOS_ERROR_EXELWTE_INSUFFICIENT_DATA                    (0x00000001) // to be returned by app to OS
#define MSDECOS_ERROR_SEMAPHORE_INSUFFICIENT_DATA                  (0x00000002) // insufficient semaphore methods received
#define MSDECOS_ERROR_ILWALID_METHOD                               (0x00000003) // unsupported method
#define MSDECOS_ERROR_ILWALID_DMA_PAGE                             (0x00000004) // not used lwrrently
#define MSDECOS_ERROR_UNHANDLED_INTERRUPT                          (0x00000005) // either app has no interrupt handler, or an unhandled os error
#define MSDECOS_ERROR_EXCEPTION                                    (0x00000006) // exception raised by falcon
#define MSDECOS_ERROR_ILWALID_CTXSW_REQUEST                        (0x00000007) // invalid ctxsw request to OS
#define MSDECOS_ERROR_APPLICATION                                  (0x00000008) // application returned nonzero error code
#define MSDECOS_ERROR_SWBREAKPT                                    (0x00000009) // exception raised to dump registers in debug mode
#define MSDECOS_INTERRUPT_EXELWTE_AWAKEN                           (0x00000100) // execute awaken enabled
#define MSDECOS_INTERRUPT_BACKEND_SEMAPHORE_AWAKEN                 (0x00000200) // backend semaphore awaken enabled (os assumes that bck_awkn addr = 2*exe_awkn addr)
#define MSDECOS_INTERRUPT_CTX_ERROR_FBIF                           (0x00000300) // ctx error from fbif
#define MSDECOS_INTERRUPT_LIMIT_VIOLATION                          (0x00000400) // limit violation
#define MSDECOS_INTERRUPT_LIMIT_AND_FBIF_CTX_ERROR                 (0x00000500) // limit violation and fbif ctx error (if both happen together)
#define MSDECOS_INTERRUPT_HALT_ENGINE                              (0x00000600) // wait for dma transfers and halt engine in response to interrupt from RM




//-----------------------------------------------------
// printMailbox_GK104
//-----------------------------------------------------
void falconPrintMailbox_v04_00(LwU32 engineBase)
{
    LwU32 mailbox0 = GPU_REG_RD32(LW_PFALCON_FALCON_MAILBOX0 + engineBase);
    LwU32 mailbox1 = GPU_REG_RD32(LW_PFALCON_FALCON_MAILBOX1 + engineBase);

    switch(mailbox0) {
        case MSDECOS_ERROR_NONE:
            dprintf("lw: + ERROR TYPE: ERROR_NONE ; MailBox0=0x%x. MailBox1=0x%x\n",
                    mailbox0, mailbox1);
            break ;
        case MSDECOS_ERROR_EXELWTE_INSUFFICIENT_DATA:
            dprintf("lw: + ERROR TYPE: ERROR_EXELWTE_INSUFFICIENT_DATA ; MailBox0=0x%x. "
                    "MailBox1=0x%x\n", mailbox0, mailbox1);
            break ;
        case MSDECOS_ERROR_SEMAPHORE_INSUFFICIENT_DATA:
            dprintf("lw: + ERROR TYPE: ERROR_SEMAPHORE_INSUFFICIENT_DATA ; MailBox0=0x%x. "
                    "MailBox1=0x%x\n",mailbox0, mailbox1);
            break ;
        case MSDECOS_ERROR_ILWALID_DMA_PAGE:
            dprintf("lw: + ERROR TYPE: ERROR_ILWALID_DMA_PAGE: ; MailBox0=0x%x. "
                    "MailBox1=0x%x\n",mailbox0, mailbox1);
            break ;
        case MSDECOS_ERROR_UNHANDLED_INTERRUPT:
            dprintf("ERROR TYPE: ERROR_UNHANDLED_INTERRUPT: Either App has No Interrupt Handler, "
                    "or an Unhandled OS Error: Mailbox0=0x%x. MailBox1=0x%x\n", mailbox0, mailbox1);
            break ;
        case MSDECOS_ERROR_APPLICATION:
            dprintf("lw: + ERROR TYPE: ERROR_APPLICATION; Application Error: "
                    "MailBox0=0x%x. MailBox1=0x%x\n",mailbox0, mailbox1);
            break ;
        case MSDECOS_ERROR_EXCEPTION:
            dprintf("lw: + ERROR TYPE: ERROR_EXCEPTION; Falcon Exception ; "
                    "MailBox0=0x%x. MailBox1=0x%x\n",mailbox0, mailbox1);
            break ;
        case MSDECOS_ERROR_ILWALID_CTXSW_REQUEST :
            dprintf("lw: + ERROR TYPE: ERROR_ILWALID_CTXSW_REQUEST ; MailBox0=0x%x. "
                "MailBox1=0x%x\n",mailbox0, mailbox1);
            break ;
        case MSDECOS_INTERRUPT_CTX_ERROR_FBIF:
            dprintf("lw: + ERROR TYPE: INTERRUPT_CTX_ERROR_FBIF ; MailBox0=0x%x. "
                    "MailBox1=0x%x\n",mailbox0, mailbox1);
            break ;                                      
        case MSDECOS_INTERRUPT_LIMIT_VIOLATION:
            dprintf("lw: + ERROR TYPE: INTERRUPT_LIMIT_VIOLATION ; MailBox0=0x%x. "
                    "MailBox1=0x%x\n",mailbox0, mailbox1);
            break ;              
        case MSDECOS_INTERRUPT_LIMIT_AND_FBIF_CTX_ERROR:
            dprintf("lw: + ERROR TYPE: INTERRUPT_LIMIT_AND_FBIF_CTX_ERROR ; "
                    "MailBox0=0x%x. MailBox1=0x%x\n",mailbox0, mailbox1);
            break ;
        case MSDECOS_INTERRUPT_HALT_ENGINE:
            dprintf("lw: + ERROR TYPE: INTERRUPT_HALT_ENGINE ; MailBox0=0x%x. "
                    "MailBox1=0x%x\n", mailbox0, mailbox1);
            break ;                
        default:
            dprintf("lw: + Unknown ERROR TYPE: MailBox0=0x%x. MailBox1=0x%x\n",
                    mailbox0, mailbox1);
            break ; 
        }
}


