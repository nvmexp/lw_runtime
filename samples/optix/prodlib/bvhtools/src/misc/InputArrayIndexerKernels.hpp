// Copyright LWPU Corporation 2017
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once
#include <prodlib/bvhtools/src/common/Intrinsics.hpp>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

#define INPUT_ARRAY_INDEXER_EXEC_WARPS_PER_BLOCK    8
#define INPUT_ARRAY_INDEXER_EXEC_BLOCKS_PER_SM      NUMBLOCKS_MAXWELL(32)

//------------------------------------------------------------------------

struct InputArrayIndexerExecParams
{
    int                           numInputs;
    int                           numBlocks;
    const unsigned int*           outArrayBaseGlobalIndex;
    unsigned int*                 outArrayTransitionBits;
    int*                          outBlockStartArrayIndex;
};

//------------------------------------------------------------------------

bool launchInputArrayIndexerExec  (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const InputArrayIndexerExecParams& p);

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
