// Copyright LWPU Corporation 2013
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
#include "../common/TypesInternal.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

#define WOOPER_EXEC_WARPS_PER_BLOCK 2   // 50% oclwpancy to avoid cache thrashing.
#define WOOPER_EXEC_BLOCKS_PER_SM   NUMBLOCKS_KEPLER(16)

//------------------------------------------------------------------------

struct WooperExecParams
{
    WoopTriangle*   outWoop;        // [maxRemapSize]
    int*            ioRemap;        // [maxRemapSize]
    const int*      inRemapSize;    // [1]
    ModelPointers   inModel;

    int             maxRemapSize;
    bool            uncomplementRemaps;
};

//------------------------------------------------------------------------

bool launchWooperExec   (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const WooperExecParams& p);

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
