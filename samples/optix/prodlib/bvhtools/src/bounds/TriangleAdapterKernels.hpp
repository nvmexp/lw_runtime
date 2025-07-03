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
#include <src/common/TypesInternal.hpp>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

#define TRIANGLE_ADAPTER_EXEC_WARPS_PER_BLOCK     8
#define TRIANGLE_ADAPTER_EXEC_BLOCKS_PER_SM       NUMBLOCKS_KEPLER(16)

#define TRIANGLE_ADAPTER_ITERATIONS_PER_WARP      1

//------------------------------------------------------------------------

struct TriangleAdapterExecParams
{
    int                           numPrimitives;
    int                           primitiveIndexBits;
    int                           computePrimBits;
    PrimBitsFormat                primBitsFormat;

    const InputTrianglesDesc*     trianglesDescArray;
    const InputTrianglesPointers* trianglesPtrArray;

    InputArrayIndexPointers       inArrayIndexing;

    float3*                       outVertices;   // [numPrimitives]
    unsigned char*                outPrimBits;
};

//-------------------------------------------------------------------------

bool launchTriangleAdapterExec  (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const TriangleAdapterExecParams& p);

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
