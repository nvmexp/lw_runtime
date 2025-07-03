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
#include <prodlib/bvhtools/src/common/TypesInternal.hpp>
#include <prodlib/bvhtools/src/bounds/ApexPointMap.hpp>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

#define MORTONSPLIT_INIT_WARPS_PER_BLOCK    8
#define MORTONSPLIT_INIT_BLOCKS_PER_SM      NUMBLOCKS_KEPLER(8)

#define MORTONSPLIT_TUNEB_WARPS_PER_BLOCK   4
#define MORTONSPLIT_TUNEB_BLOCKS_PER_SM     NUMBLOCKS_KEPLER(16)
#define MORTONSPLIT_TUNEB_TRIS_PER_THREAD   8

#define MORTONSPLIT_EXEC_WARPS_PER_BLOCK    25  // At most 32 warps per SM due to shared mem. Less => reduce L2 trashing.
#define MORTONSPLIT_EXEC_BLOCKS_PER_SM      NUMBLOCKS_KEPLER(1)
#define MORTONSPLIT_EXEC_MAX_PIECES_LOG2    16  // An input triangle is expanded by at most a factor of 2^N.

//------------------------------------------------------------------------

struct MortonSplitGlobals
{
    float   totalPriority;      // Must be zero initially.

    float   lwrScaleFactor;     // Must be zero initially.
    float   lwrScaleInterval;   // Must be zero initially.
    int     totalSplits1;       // with lwrScaleFactor + lwrScaleInterval * 1.0f
    int     totalSplits2;       // with lwrScaleFactor + lwrScaleInterval * 2.0f
    int     totalSplits3;       // with lwrScaleFactor + lwrScaleInterval * 3.0f

    int     execWorkCounter;    // Must be zero initially.
    int     freshAABBCounter;   // Must be zero initially.
};

//------------------------------------------------------------------------

struct MortonSplitInitParams
{
    int*                outPrimIndices;     // [maxOutputPrims]
    Range*              outPrimRange;       // [1]
    float*              outPriorities;      // [maxInputTris]
    MortonSplitGlobals* globals;            // [1]

    const int*          inTriOrder;         // [triRange.end] NULL => identity mapping
    const Range*        triRange;           // [1] NULL => (0, maxInputTris)
    ModelPointers       inModel;
    const ApexPointMap* inApexPointMap;

    int                 maxInputTris;
    float               splitBeta;
    float               priorityLog2X;
    float               priorityY;
    float               epsilon;
};

//------------------------------------------------------------------------

struct MortonSplitTuneParams
{
    MortonSplitGlobals* globals;            // [1]

    const float*        inPriorities;       // [maxInputTris]
    const Range*        triRange;           // [1] NULL => (0, maxInputTris)

    int                 maxInputTris;
    int                 maxOutputPrims;
    float               maxSplitsPerTriangle;
    float               splitBeta;
};

//------------------------------------------------------------------------

struct MortonSplitExecParams
{
    PrimitiveAABB*      outSplitAABBs;      // [maxSplitAABBs]
    int*                outPrimIndices;     // [maxOutputPrims]
    Range*              outPrimRange;       // [1]
    MortonSplitGlobals* globals;            // [1]

    const Range*        triRange;           // [1] NULL => (0, maxInputTris)
    const float*        inPriorities;       // [maxInputTris]
    ModelPointers       inModel;
    const ApexPointMap* inApexPointMap;

    int                 maxInputTris;
    int                 maxOutputPrims;
    int                 maxSplitAABBs;
    int                 maxAABBsPerTriangle;
    float               epsilon;
};

//------------------------------------------------------------------------

bool launchMortonSplitInit  (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const MortonSplitInitParams& p);
bool launchMortonSplitTuneA (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const MortonSplitTuneParams& p);
bool launchMortonSplitTuneB (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const MortonSplitTuneParams& p);
bool launchMortonSplitExec  (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const MortonSplitExecParams& p);

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
