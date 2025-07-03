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

#ifndef __TRACER_HPP__
#define __TRACER_HPP__

#include "TracerTypes.hpp"
#include "../common/TypesInternal.hpp"
#include "../common/LwdaUtils.hpp"



namespace prodlib
{
namespace bvhtools
{

size_t sizeOfRayBuffer( int numRays, int rayFormat );
size_t sizeOfHitBuffer( int numHits, int hitFormat );

//------------------------------------------------------------------------

class Tracer
{
public:
    struct Layout
    {
        bool    storeOnGPU;       // Store the BVH primarily in GPU memory.
        int     arrayAlign;       // Alignment of the data arrays.
        bool    reorderTriangles; // Store triangles in triangle list order. 
        bool    optixNodes;       // Use OptixNode structure instead of Node        
    };
   
                        Tracer              (void);
    virtual             ~Tracer             (void);

    virtual const char* getBackendName      (void) const = 0;
    virtual Layout      getBVHLayout        (void) const = 0;
    
    virtual void        traceFromHostMem(const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyHit, const bool watertight, const void* const stream = 0);
    virtual void        traceFromDeviceMem(const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyHit, const bool watertight, const void* const stream = 0, const int streamIdx = 0);

private:
                        Tracer              (const Tracer&); // forbidden
    Tracer&             operator=           (const Tracer&); // forbidden

private:
    LwdaUtils*          m_lwdaUtils;
    char*               m_cpuTemp;
    char*               m_gpuTemp;
    size_t              m_cpuTempSize;
    size_t              m_gpuTempSize;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
#endif // __TRACER_HPP__
