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
#include "ApexPointMapLookup.hpp"
#include <include/Types.hpp>
#include <prodlib/bvhtools/src/common/SharedKernelFunctions.hpp>


namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

#define INSTANCE_DATA_ADAPTER_EXEC_WARPS_PER_BLOCK    4
#define INSTANCE_DATA_ADAPTER_EXEC_BLOCKS_PER_SM      NUMBLOCKS_KEPLER(16)

//------------------------------------------------------------------------

struct InstanceDataAdapterExecParams
{
    BvhInstanceData*            outBvhInstanceData;
    PrimitiveAABB*              outWorldSpaceAabbs;
    float*                      outIlwMatrices;

    const InstanceDesc*         inInstanceDesc;
    const ApexPointMap* const*  inApexPointMaps;
    const float*                inTransforms;
    const int*                  inInstanceIds;

    int                         numInstances;
    int                         matrixStride;
};

//------------------------------------------------------------------------------

static INLINE void copyTransform(float* dst, const float* src)
{
  for(int i=0; i < 12; ++i)
    dst[i] = src[i];
}

//------------------------------------------------------------------------------
static INLINE void computeAabb(const float* transform, const ApexPointMap* apm, int primitiveId, PrimitiveAABB* primAabb)
{
  const float* m = transform;

  // Get world space bounds from apex point map.
  float2 bx = apm->getSlab(make_float3(m[0], m[1], m[2]));
  float2 by = apm->getSlab(make_float3(m[4], m[5], m[6]));
  float2 bz = apm->getSlab(make_float3(m[8], m[9], m[10]));

  // Write the PrimitiveAABB

  primAabb->lox = bx.x + m[3];
  primAabb->loy = by.x + m[7];
  primAabb->loz = bz.x + m[11];
  primAabb->hix = bx.y + m[3];
  primAabb->hiy = by.y + m[7];
  primAabb->hiz = bz.y + m[11];
  primAabb->primitiveIdx = primitiveId;
  primAabb->pad = 0;
}



//-------------------------------------------------------------------------

bool launchInstanceDataAdapterExec  (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const InstanceDataAdapterExecParams& p);

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
