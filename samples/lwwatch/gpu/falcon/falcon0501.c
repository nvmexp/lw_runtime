/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2013-2018 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

/* ------------------------ Includes --------------------------------------- */
#include "maxwell/gm200/dev_falcon_v4.h"
#include "falcon.h"

static void _flcnImemWrite_v05_01 (LwU32, LwU32, LwU32, LwU32, LwU32);

/*!
 *  Return the IMEM block info given a block index
 *
 *  Here, only change in function compared to that in falcon0400.c is that
 *  IMCTL_DEBUG is used instead of IMCTL, because GM20X blocked IMCTL
 *  and IMCTL_DEBUG gives same functionality as of IMCTL
 *  @param[in]  engineBase  Base address of the Falcon engine
 *  @param[in]  blockIndex  The index of interested block
 *  @param[out] pBlockInfo  The buffer to store the results
 *
 *  @return True on success, false otherwise.
 */
BOOL
flcnImemBlk_v05_01
(
    LwU32           engineBase,
    LwU32           blockIndex,
    FLCN_BLOCK      *pBlockInfo
)
{
    const FLCN_CORE_IFACES *pFCIF = &(flcnCoreIfaces_v05_01);
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
    cmd     = FLD_SET_DRF(_PFALCON_FALCON, _IMCTL_DEBUG, _CMD, _IMBLK, blockIndex);
    FLCN_REG_WR32(LW_PFALCON_FALCON_IMCTL_DEBUG, cmd);
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
 *  Here, only change in function compared to that in falcon0400.c is that
 *  IMCTL_DEBUG is used instead of IMCTL, because GM20X blocked IMCTL
 *  and IMCTL_DEBUG gives same functionality as of IMCTL

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
flcnImemTag_v05_01
(
    LwU32           engineBase,
    LwU32           codeAddr,
    FLCN_TAG*       pTagInfo
)
{
    const FLCN_CORE_IFACES *pFCIF = &(flcnCoreIfaces_v05_01);
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
    //       Upper Bits  : LW_PFALCON_FALCON_IMCTL_DEBUG_CMD_IMTAG
    // Result is fetched from IMSTAT register
    //
    cmd     = FLD_SET_DRF(_PFALCON_FALCON, _IMCTL_DEBUG, _CMD, _IMTAG, codeAddr & maxAddr);
    FLCN_REG_WR32(LW_PFALCON_FALCON_IMCTL_DEBUG, cmd);
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

LwBool
flcnIsDmemAccessAllowed_v05_01
(
    const FLCN_ENGINE_IFACES   *pFEIF,
    LwU32                       engineBase,
    LwU32                       addrLo,
    LwU32                       addrHi,
    LwBool                      bIsRead
)
{
    LwU32 privLevelMask, sctl;
    LwU32 blkLo, blkHi;

    blkLo = FLCN_DMEM_ADDR_TO_BLK(addrLo);
    blkHi = FLCN_DMEM_ADDR_TO_BLK(addrHi);

    privLevelMask = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEM_PRIV_LEVEL_MASK);
    sctl          = FLCN_REG_RD32(LW_PFALCON_FALCON_SCTL);

    if (DRF_VAL(_PFALCON_FALCON, _SCTL_, HSMODE, sctl) ==
        LW_PFALCON_FALCON_SCTL_HSMODE_TRUE)
    {
        // HS mode. No read/write access
        dprintf("lw: Falcon is in HS mode. No read/write access\n");
        return LW_FALSE;
    }
    else if (DRF_VAL(_PFALCON_FALCON, _SCTL_, LSMODE, sctl) == 
             LW_PFALCON_FALCON_SCTL_LSMODE_TRUE)
    {
        // LS mode
        if (bIsRead)
        {
            // Check read mask
            if (DRF_VAL(_PFALCON_FALCON, _DMEM_PRIV_LEVEL_MASK_, READ_PROTECTION_LEVEL0, privLevelMask)
                == LW_PFALCON_FALCON_DMEM_PRIV_LEVEL_MASK_READ_PROTECTION_LEVEL0_ENABLE)
            {
                // Read protection disabled
                return LW_TRUE;
            }
            // Read protection enabled
            return pFEIF->flcnEngIsDmemRangeAccessible(blkLo, blkHi);
        }
        else
        {
            // Check write mask
            if (DRF_VAL(_PFALCON_FALCON, _DMEM_PRIV_LEVEL_MASK_, WRITE_PROTECTION_LEVEL0, privLevelMask)
                == LW_PFALCON_FALCON_DMEM_PRIV_LEVEL_MASK_WRITE_PROTECTION_LEVEL0_ENABLE)
            {
                // Write protection disabled
                return LW_TRUE;
            }
            // Write protection enabled
            return pFEIF->flcnEngIsDmemRangeAccessible(blkLo, blkHi);
        }
    }

    // NS mode. Both read/write access
    return LW_TRUE;
}

LwU32
flcnImemRead_v05_01
(
    LwU32  engineBase,
    LwU32  addr,
    LwU32  length,
    LwU32  port,
    LwU32* pImem
)
{
    const FLCN_CORE_IFACES *pFCIF = &(flcnCoreIfaces_v05_01);
    LwU32 imemSize  = 0x0;
    LwU32 imemcOrig = 0x0;
    LwU32 imemc     = 0x0;
    LwU32 maxPort   = 0x0;
    LwU32 i;
    LwU32 privLevelMask = FLCN_REG_RD32(LW_PFALCON_FALCON_IMEM_PRIV_LEVEL_MASK);
    LwU32 sctl          = FLCN_REG_RD32(LW_PFALCON_FALCON_SCTL);

    if (DRF_VAL(_PFALCON_FALCON, _SCTL_, HSMODE, sctl) ==
        LW_PFALCON_FALCON_SCTL_HSMODE_TRUE)
    {
        dprintf("lw: Falcon is in HS mode. No read access\n");
        return 0;
    }

    if (DRF_VAL(_PFALCON_FALCON, _SCTL_, LSMODE, sctl) == 
        LW_PFALCON_FALCON_SCTL_LSMODE_TRUE)
    {
        // Check read mask
        if (DRF_VAL(_PFALCON_FALCON, _IMEM_PRIV_LEVEL_MASK_, READ_PROTECTION_LEVEL0, privLevelMask)
            != LW_PFALCON_FALCON_IMEM_PRIV_LEVEL_MASK_READ_PROTECTION_LEVEL0_ENABLE)
        {
            dprintf("lw: Falcon is in LS mode with insufficient read privileges\n");
            return 0;
        }
    }

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

LwU32
flcnImemWrite_v05_01
(
    LwU32  engineBase,
    LwU32  addr,
    LwU32  val,
    LwU32  width,
    LwU32  length,
    LwU32  port
)
{
    const FLCN_CORE_IFACES *pFCIF = &(flcnCoreIfaces_v05_01);
    LwU32 imemSize  = 0x0;
    LwU32 imemcOrig = 0x0;
    LwU32 maxPort   = 0x0;
    LwU32 i;
    LwU32 privLevelMask = FLCN_REG_RD32(LW_PFALCON_FALCON_DMEM_PRIV_LEVEL_MASK);
    LwU32 sctl          = FLCN_REG_RD32(LW_PFALCON_FALCON_SCTL);

    if (DRF_VAL(_PFALCON_FALCON, _SCTL_, HSMODE, sctl) ==
        LW_PFALCON_FALCON_SCTL_HSMODE_TRUE)
    {
        dprintf("lw: Falcon is in HS mode. No write access\n");
        return 0;
    }

    if (DRF_VAL(_PFALCON_FALCON, _SCTL_, LSMODE, sctl) == 
        LW_PFALCON_FALCON_SCTL_LSMODE_TRUE)
    {
        // Check write mask
        if (DRF_VAL(_PFALCON_FALCON, _IMEM_PRIV_LEVEL_MASK_, WRITE_PROTECTION_LEVEL0, privLevelMask)
            != LW_PFALCON_FALCON_IMEM_PRIV_LEVEL_MASK_READ_PROTECTION_LEVEL0_ENABLE)
        {
            dprintf("lw: Falcon is in LS mode with insufficient write privileges\n");
            return 0;
        }
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
        _flcnImemWrite_v05_01(engineBase, addr + (i * width), val, width, port);
    }
    FLCN_REG_WR32(LW_PFALCON_FALCON_IMEMC(port), imemcOrig);    
    return length * width;
}

static
void
_flcnImemWrite_v05_01
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
