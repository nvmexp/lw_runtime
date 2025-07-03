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

#ifndef __TRACERLWDA_HPP__
#define __TRACERLWDA_HPP__

#include "Tracer.hpp"
#include "TracerLwdaKernels.hpp"
#include "../common/LwdaUtils.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

class TracerLwda : public Tracer // for SM >= 1.2
{
public:
                        TracerLwda          (void);
    virtual             ~TracerLwda         (void);

    void                init                ( int streamCount=1 ); 

    virtual const char* getBackendName      (void) const    { return m_backendName; }
    virtual Layout      getBVHLayout        (void) const    { return m_bvhLayout; }
    
    virtual void        traceFromDeviceMem  (const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitsFormat, const int numRays, const bool anyhit, const bool watertight, const void* const stream = 0, const int streamIdx = 0);

private:
    void                configure           ( const TracerData& bvhData, int rayFormat, bool watertight );
    void                launch              ( const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyhit, const void* const stream, const int streamIdx );
    void                setupKernelParams   ( const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyhit, const void* const stream, const int streamIdx, TracerLwdaParams& params );

                        TracerLwda          (const TracerLwda&); // forbidden
    TracerLwda&         operator=           (const TracerLwda&); // forbidden

private:
    LwdaUtils           m_lwda;
    char                m_backendName[256];
    int                 m_warpsPerBlock;
    bool                m_usePersistentThreads;
    TraceLaunchFunc     m_launchFunc;
    Layout              m_bvhLayout;
    int                 m_numPersistentThreads;
    int*                m_warpCounters; // one per stream

    // TODO: put this in a kind of buffer
    char*               m_kernelData_d; // a ptr to the TracerParamsMesh/TracerParamsGroup data for the device. all put in one chunk of memory.
    std::vector<char>   m_kernelData_h; // a ptr to the TracerParamsMesh/TracerParamsGroup data on the host (data will be copied from host to device)
    int                 m_streamCount;

    TracerLwdaConfig    m_config;
    int                 m_numMeshes;
};

//------------------------------------------------------------------------

void copyMesh(TracerParamsMesh& dst, const TracerDataMesh& src);

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
#endif // __TRACERLWDA_HPP__
