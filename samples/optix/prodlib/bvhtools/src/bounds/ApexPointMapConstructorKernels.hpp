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
#include "ApexPointMap.hpp"
#include <prodlib/bvhtools/src/common/TypesInternal.hpp>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

#define APM_CONSTRUCT_WARPS_PER_BLOCK   4
#define APM_CONSTRUCT_BATCH_SIZE        256             // Must be a multiple of 32.

#if !defined(__LWDA_ARCH__) || (__LWDA_ARCH__ >= 500)   // Compiling for Maxwell (or host).
#   define APM_CONSTRUCT_HASH_PER_WARP  1024            // In dwords. Must be power of two. 24 warps/SM.
#else                                                   // Compiling for Kepler (or Fermi).
#   define APM_CONSTRUCT_HASH_PER_WARP  256             // In dwords. Must be power of two. 48 warps/SM.
#endif

//------------------------------------------------------------------------

struct APMConstructParams
{
    ApexPointMap*           outApexPointMap;
    int*                    workCounter;        // Zeroed by the constructor.
    ModelPointers           inModel;
    int                     apmResolution;
    int                     numDirections;
};

//------------------------------------------------------------------------

bool launchAPMInit     (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const APMConstructParams& p);
bool launchAPMConstruct(dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const APMConstructParams& p);

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
