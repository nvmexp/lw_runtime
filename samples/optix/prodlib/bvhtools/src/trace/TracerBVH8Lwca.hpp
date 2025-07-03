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

#ifndef __TRACERBVH8LWDA_HPP__
#define __TRACERBVH8LWDA_HPP__

#include "Tracer.hpp"
#include "TracerBVH8LwdaKernels.hpp"
#include <prodlib/bvhtools/src/common/LwdaUtils.hpp>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

class TracerBVH8Lwda : public Tracer // for SM >= 5.2
{
public:
                        TracerBVH8Lwda      (void);
    virtual             ~TracerBVH8Lwda     (void);

    void                init                (int streamCount = 1); 

    virtual const char* getBackendName      (void) const    { return m_backendName; }
    virtual Layout      getBVHLayout        (void) const;
    
    virtual void        traceFromDeviceMem  (const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitsFormat, const int numRays, const bool anyhit, const bool watertight, const void* const stream = 0, const int streamIdx = 0);

private:
    void                configure           (const TracerData& bvhData, int rayFormat, bool watertight, bool anyhit);
    void                launch              (const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const void* const stream, const int streamIdx);
    void                setupKernelParams   (const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const void* const stream, const int streamIdx, TracerBVH8LwdaParams& params);

                        TracerBVH8Lwda      (const TracerBVH8Lwda&); // forbidden
    TracerBVH8Lwda&     operator=           (const TracerBVH8Lwda&); // forbidden

private:
    LwdaUtils           m_lwda;
    const char*         m_backendName;
    TraceBVH8LaunchFunc m_launchFunc;
    int*                m_warpCounters; // one per stream

    // TODO: put this in a kind of buffer
    TracerParamsMesh*   m_meshData_d;           // Pointer to TracerParamsMesh array on the device.
    std::vector<TracerParamsMesh> m_meshData_h; // TracerParamsMesh array on the host (will be copied from host to device).
    int                 m_streamCount;

    TracerBVH8LwdaConfig m_config;
    int                 m_numMeshes;
};

//------------------------------------------------------------------------

void copyMesh(TracerParamsMesh& dst, const TracerDataMeshBVH8& src);

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
#endif // __TRACERBVH8LWDA_HPP__
