/*
 *  Copyright (c) 2012, LWPU Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of LWPU Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __TRACERLWDAKERNELS_HPP__
#define __TRACERLWDAKERNELS_HPP__

#include "TracerTypes.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

#define EntrypointSentinel  0x76543210  // Bottom-most stack entry, indicating the end of traversal.

//------------------------------------------------------------------------

struct TracerParamsMesh
{
  const float4*   nodes;
  const float4*   triangles;
  const int3*     indices;
  const float*    vertices;
  const int*      remap;
  int             indexStride;
  int             vertexStride;
  int             rootNode;
  int             numNodes;
  int             numRemaps;
};

//------------------------------------------------------------------------

struct TracerParamsGroup
{
  const float4*   nodes;
  const int*      remap;
  const float*    ilwMatrices;
  const int*      modelId;
  int             rootNode;
  int             numEntities;
  int             matrixStride;
};

//------------------------------------------------------------------------

struct TracerLwdaParams
{
  const void*           rays;
  void*                 hits;
  int*                  warpCounter;
  const TracerParamsGroup* group;
  const TracerParamsMesh*  meshes;
  TracerParamsMesh      firstMesh;
  int                   numRays;
  int                   rayFormat;
  int                   hitFormat;
  bool                  anyhit;
};

//------------------------------------------------------------------------

struct TracerLwdaConfig
{
    bool    twoLevel;       // Two-level traversal of a group?
    bool    useWoop;        // Use Woop ray/triangle test?
    bool    useFloat4;      // Vertex data is float4 instead of float3?
    bool    useTex;         // Access BVH through linear textures?
    bool    useMasking;     // Enable geometry masking?
    bool    useWatertight;     // Watertight intersection test
};

//------------------------------------------------------------------------

typedef bool (*TraceLaunchFunc)(dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const TracerLwdaParams& p, const TracerLwdaConfig& c);

bool launchTrace20  (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const TracerLwdaParams& p, const TracerLwdaConfig& c);
bool launchTrace30  (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const TracerLwdaParams& p, const TracerLwdaConfig& c);
bool launchTrace35  (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const TracerLwdaParams& p, const TracerLwdaConfig& c);
bool launchTrace52  (dim3 gridDim, dim3 blockDim, lwdaStream_t stream, const TracerLwdaParams& p, const TracerLwdaConfig& c);

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
#endif // __TRACERLWDAKERNELS_HPP__
