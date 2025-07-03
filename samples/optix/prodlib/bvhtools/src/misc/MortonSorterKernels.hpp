// Copyright LWPU Corporation 2015
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
#include <prodlib/bvhtools/src/common/TypesInternal.hpp>
#include <prodlib/bvhtools/src/bounds/ApexPointMap.hpp>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

#define MORTON_CALC_WARPS_PER_BLOCK 4
#define MORTON_CALC_BLOCKS_PER_SM   NUMBLOCKS_KEPLER(16)

//------------------------------------------------------------------------

struct MortonCalcParams
{
    unsigned char*          outMortonCodes; // [maxPrims]
    int*                    outPrimOrder;   // [maxPrims]
    Range*                  outPrimRange;   // [1]

    const int*              inPrimOrder;    // [inPrimRange->end], NULL => identity mapping
    ModelPointers           inModel;
    const ApexPointMap*     inApexPointMap;
    const Range*            inPrimRange;    // [1], NULL => (0, maxPrims)

    int                     maxPrims;
};

//------------------------------------------------------------------------

bool launchMortonCalc   (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const MortonCalcParams& p, int bytesPerMortonCode);

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
