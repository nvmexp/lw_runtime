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
#include "pascal/gp104/dev_falcon_v4.h"
#include "falcon.h"
#include "chip.h"

/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Static variables ------------------------------- */
/* ------------------------ Function Prototypes ---------------------------- */

/*!
 *  Return the DMEM tag width in bits
 *
 *  @param[in] engineBase  Base address of the Falcon engine
 *
 *  @return Width of the tag in bits.
 */
LwU32
flcnDmemGetTagWidth_v06_00
(
    LwU32 engineBase
)
{
    return DRF_VAL(_PFALCON_FALCON, _HWCFG1, _DMEM_TAG_WIDTH,
                   FLCN_REG_RD32(LW_PFALCON_FALCON_HWCFG1));
}


/*!
 *  Return the DMEM block info given a block index
 *
 *  @param[in]  engineBase  Base address of the Falcon engine
 *  @param[in]  blockIndex  The index of interested block
 *  @param[out] pBlockInfo  The buffer to store the results
 *
 *  @return True on success, false otherwise.
 */

BOOL
flcnDmemBlk_v06_00
(
    LwU32           engineBase,
    LwU32           blockIndex,
    FLCN_BLOCK      *pBlockInfo
)
{
    const FLCN_CORE_IFACES *pFCIF = &(flcnCoreIfaces_v06_00);
    LwU32    numBlocks   = 0x0;
    LwU32    tagWidth    = 0x0;
    LwU32    tagMask     = 0x0;
    LwU32    cmd         = 0x0;
    LwU32    result      = 0x0;

    numBlocks   = pFCIF->flcnDmemGetSize(engineBase);
    tagWidth    = pFCIF->flcnDmemGetTagWidth(engineBase);
    tagMask     = BIT(tagWidth) - 1;

    // Bail out for invalid block index
    if ((blockIndex >= numBlocks) || (pBlockInfo == NULL))
    {
        return FALSE;
    }

    // Create the command, write, and read result
    cmd     = FLD_SET_DRF(_PFALCON_FALCON, _DMCTL, _CMD, _DMBLK, blockIndex);
    FLCN_REG_WR32(LW_PFALCON_FALCON_DMCTL, cmd);
    result  = FLCN_REG_RD32(LW_PFALCON_FALCON_DMSTAT);

    // Extract the TAG value from bits 8 to 7+TagWidth
    pBlockInfo->tag         = (result >> 8) & tagMask;
    pBlockInfo->bValid      = DRF_VAL(_PFALCON_FALCON, _DMBLK, _VALID, result);
    pBlockInfo->bPending    = DRF_VAL(_PFALCON_FALCON, _DMBLK, _PENDING, result);
    pBlockInfo->bSelwre     = DRF_VAL(_PFALCON_FALCON, _DMBLK, _SELWRE, result);

    pBlockInfo->blockIndex  = blockIndex;
    return TRUE;


}
/*!
 *
 *  Get the status of the block that codeAddr maps to. The DMEM may be "tagged".
 *  In this case, a code address in DMEM may be as large as 7+TagWidth bits. A
 *  specific code address has a tag specified by the bits 8 to 7+TagWidth. This
 *  tag may or may not be "mapped" to a DMEM 256 byte code block. This function
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
flcnDmemTag_v06_00
(
    LwU32           engineBase,
    LwU32           codeAddr,
    FLCN_TAG*       pTagInfo
)
{   
    const FLCN_CORE_IFACES *pFCIF = &(flcnCoreIfaces_v06_00);
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

    numBlocks   = pFCIF->flcnDmemGetSize(engineBase);
    tagWidth    = pFCIF->flcnDmemGetTagWidth(engineBase);
    maxAddr     = (BIT(tagWidth) << 8) - 1;

    blockMask   = numBlocks;
    ROUNDUP_POW2(blockMask);
    blockMask--;

    //
    // Create the command, write, and read result
    // Command is created by taking:
    //       Bits T+7 - 0: Address
    //       Upper Bits  : LW_PFALCON_FALCON_DMCTL_CMD_DMTAG
    // Result is fetched from DMSTAT register
    //
    cmd     = FLD_SET_DRF(_PFALCON_FALCON, _DMCTL, _CMD, _DMTAG, codeAddr & maxAddr);
    FLCN_REG_WR32(LW_PFALCON_FALCON_DMCTL, cmd);
    result  = FLCN_REG_RD32(LW_PFALCON_FALCON_DMSTAT);

    // Extract the block index and other information
    blockIndex  = result & blockMask;
    valid       = DRF_VAL(_PFALCON_FALCON, _DMTAG, _VALID, result);
    pending     = DRF_VAL(_PFALCON_FALCON, _DMTAG, _PENDING, result);
    secure      = DRF_VAL(_PFALCON_FALCON, _DMTAG, _SELWRE, result);
    multiHit    = DRF_VAL(_PFALCON_FALCON, _DMTAG, _MULTI_HIT, result);
    miss        = DRF_VAL(_PFALCON_FALCON, _DMTAG, _MISS, result);

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
 * Retrieve the virtual address that is the threshold for VA to PA mapping.
 * For a virtual address below this threshold, PA = VA.
 *
 * @return  The DMEM VA bound
 */
LwU32
flcnDmemVaBoundaryGet_v06_00
(
    LwU32 engineBase
)
{
    LwU32 boundInBlocks;

    if (IsGP100())
    {
        return LW_FLCN_DMEM_VA_BOUND_NONE;
    }
    else
    {
        boundInBlocks = DRF_VAL(_PFALCON, _FALCON_DMVACTL, _BOUND,
                                FLCN_REG_RD32(LW_PFALCON_FALCON_DMVACTL));
        return FLCN_DMEM_BLK_TO_ADDR(boundInBlocks);
    }
}

LwBool
falconTrpcIsCompressed_GP104
(
    LwU32   engineBase
)
{
    return FLD_TEST_DRF(_PFALCON, _FALCON_DEBUG1, _TRACE_FORMAT, _COMPRESSED,
                        FLCN_REG_RD32(LW_PFALCON_FALCON_DEBUG1));
}

LwU32
falconTrpcGetPC_GP104
(
    LwU32   engineBase,
    LwU32   idx,
    LwU32*  pCount
)
{
    FLCN_REG_WR32(LW_PFALCON_FALCON_TRACEIDX, idx);
    if (pCount != NULL)
    {
        *pCount = DRF_VAL(_PFALCON, _FALCON_TRACEINFO, _COUNT,
                          FLCN_REG_RD32(LW_PFALCON_FALCON_TRACEINFO));
    }
    return DRF_VAL(_PFALCON, _FALCON_TRACEPC, _PC,
                   FLCN_REG_RD32(LW_PFALCON_FALCON_TRACEPC));
}
