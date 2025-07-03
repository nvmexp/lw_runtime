/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
 
//*****************************************************
//
// lwwatch WinDbg Extension for PMU
// pmugm20x.c
//
//*****************************************************

//
// includes
//
#include "pmu.h"
#include "maxwell/gm200/dev_pwr_pri.h"

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
pmuImblk_GM20X
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
    cmd     = FLD_SET_DRF(_PPWR_FALCON, _IMCTL_DEBUG, _CMD, _IMBLK, blockIndex);
    PMU_REG_WR32(LW_PPWR_FALCON_IMCTL_DEBUG, cmd);
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
pmuImtag_GM20X
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
    //       Upper Bits  : LW_PPWR_FALCON_IMCTL_DEBUG_CMD_IMTAG
    // Result is fetched from IMSTAT register
    //
    cmd     = FLD_SET_DRF(_PPWR_FALCON, _IMCTL_DEBUG, _CMD, _IMTAG, codeAddr & maxAddr);
    PMU_REG_WR32(LW_PPWR_FALCON_IMCTL_DEBUG, cmd);
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
 * @return Get the number of DMEM carveouts
 */
LwU32
pmuGetDmemNumPrivRanges_GM200(void)
{
    // No register to read this, so hardcode until HW adds support.
    return 2; // RANGE0/1
}

/*!
 * @return Get the DMEM Priv Range0/1
 */
void
pmuGetDmemPrivRange_GM200
(
    LwU32  index,
    LwU32 *rangeStart,
    LwU32 *rangeEnd
)
{
    LwU32 reg;

    switch(index)
    {
        case 0:
        {
            reg         = GPU_REG_RD32(LW_PPWR_FALCON_DMEM_PRIV_RANGE0);
            *rangeStart = DRF_VAL(_PPWR_FALCON, _DMEM_PRIV_RANGE0, _START_BLOCK, reg);
            *rangeEnd   = DRF_VAL(_PPWR_FALCON, _DMEM_PRIV_RANGE0, _END_BLOCK, reg);
            break;
        }

        case 1:
        {
            reg         = GPU_REG_RD32(LW_PPWR_FALCON_DMEM_PRIV_RANGE1);
            *rangeStart = DRF_VAL(_PPWR_FALCON, _DMEM_PRIV_RANGE1, _START_BLOCK, reg);
            *rangeEnd   = DRF_VAL(_PPWR_FALCON, _DMEM_PRIV_RANGE1, _END_BLOCK, reg);
            break;
        }

        default:
        {
            *rangeStart = FALCON_DMEM_PRIV_RANGE_ILWALID;
            *rangeEnd   = FALCON_DMEM_PRIV_RANGE_ILWALID;
            dprintf("lw: Invalid priv range index: %d\n", index);
            break;
        }
    }
}

/*!
 * @return LW_TRUE  DMEM range is accessible 
 *         LW_FALSE DMEM range is inaccessible 
 */
LwBool
pmuIsDmemRangeAccessible_GM200
(
    LwU32 blkLo,
    LwU32 blkHi
)
{
    LwU32  i, numPrivRanges;
    LwU32  rangeStart = FALCON_DMEM_PRIV_RANGE_ILWALID;
    LwU32  rangeEnd   = FALCON_DMEM_PRIV_RANGE_ILWALID;
    LwBool accessAllowed = LW_FALSE;

    numPrivRanges = pPmu[indexGpu].pmuGetDmemNumPrivRanges();

    for (i = 0; i < numPrivRanges; i++)
    {
        pPmu[indexGpu].pmuGetDmemPrivRange(i, &rangeStart, &rangeEnd);

        if (rangeStart >= rangeEnd)
        {
            // invalid range.
            continue;
        }

        if (blkLo >= rangeStart && blkHi <= rangeEnd)
        {
            // We're within range
            accessAllowed = LW_TRUE;
            break;
        }
    }

    if (!accessAllowed)
    {
        // Print out info message
        dprintf("lw: PMU is in LS mode. Requested address range is not within "
                "ranges accessible by CPU.\n");
    }

    return accessAllowed;
}

const char *
pmuUcodeName_GM204()
{
    return "g_c85b6_gm20x";
}
