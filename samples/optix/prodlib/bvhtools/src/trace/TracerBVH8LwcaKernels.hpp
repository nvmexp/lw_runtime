/*
 *  Copyright (c) 2016, LWPU Corporation
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

#ifndef __TRACERBVH8LWDAKERNELS_HPP__
#define __TRACERBVH8LWDAKERNELS_HPP__

#include "TracerLwdaKernels.hpp"
#include <prodlib/bvhtools/include/BVH8Types.hpp>
#include <stdint.h>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

struct TracerBVH8LwdaParams
{
  const void*           rays;
  void*                 hits;
  TracerParamsMesh      mesh;   // Valid if useInstancing=False.
  TracerParamsGroup     group;  // Valid if useInstancing=True.
  TracerParamsMesh*     meshes; // Valid if useInstancing=True.
  int*                  warpCounter;
  int                   numRays;
  int                   rayFormat;
  int                   hitFormat;
};

//------------------------------------------------------------------------

struct TracerBVH8LwdaConfig
{
    bool    useInstancing;      // Two-level traversal of a group?
    bool    useExternalTris;    // Access triangle data from an external buffer?
    bool    useWatertight;      // Require watertight traversal & intersection?
    bool    anyhit;
};

//------------------------------------------------------------------------

typedef bool (*TraceBVH8LaunchFunc)(int smArch, int numSMs, lwdaStream_t stream, const TracerBVH8LwdaParams& p, const TracerBVH8LwdaConfig& c);

bool launchBVH8Trace52  (int smArch, int numSMs, lwdaStream_t stream, const TracerBVH8LwdaParams& p, const TracerBVH8LwdaConfig& c);

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
#endif // __TRACERBVH8LWDAKERNELS_HPP__
