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

#include "Tracer.hpp"

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

Tracer::Tracer(void)
:   m_lwdaUtils     (NULL),
    m_cpuTemp       (NULL),
    m_gpuTemp       (NULL),
    m_cpuTempSize   (0),
    m_gpuTempSize   (0)
{
}

//------------------------------------------------------------------------

Tracer::~Tracer(void)
{
    if (m_cpuTemp)
        m_lwdaUtils->hostFree(m_cpuTemp);
    if (m_gpuTemp)
        m_lwdaUtils->deviceFree(m_gpuTemp);
    if (m_lwdaUtils)
        delete m_lwdaUtils;
}

//------------------------------------------------------------------------

void Tracer::traceFromHostMem(const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyHit, const bool watertight, const void* const stream)
{
    // We get here when the subclass doesn't implement traceFromHostMem

    // Create LwdaUtils.

    if (!m_lwdaUtils)
        m_lwdaUtils = new LwdaUtils;
    
    // Allocate GPU buffer.

    size_t sizeRays = sizeOfRayBuffer(numRays, rayFormat);
    size_t sizeHits = sizeOfHitBuffer(numRays, hitFormat);
    if (m_gpuTemp && m_gpuTempSize < sizeRays + sizeHits)
    {
        m_lwdaUtils->deviceFree(m_gpuTemp);
        m_gpuTemp = NULL;
    }

    if (!m_gpuTemp)
    {
        m_gpuTemp = m_lwdaUtils->deviceAlloc(sizeRays + sizeHits);
        m_gpuTempSize = sizeRays + sizeHits;
    }


    m_lwdaUtils->memcpyHtoDAsync(m_gpuTemp, rays, sizeRays, ( lwdaStream_t )stream );
    traceFromDeviceMem(bvh, m_gpuTemp, rayFormat, m_gpuTemp + sizeRays, hitFormat, numRays, anyHit, watertight, stream);
    m_lwdaUtils->memcpyDtoHAsync(hits, m_gpuTemp + sizeRays, sizeHits, ( lwdaStream_t )stream);
}

//------------------------------------------------------------------------

void Tracer::traceFromDeviceMem(const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyHit, const bool watertight, const void* const stream, const int streamIdx)
{
    // We get here when the subclass doesn't implement traceFromDeviceMem

    // Create LwdaUtils.

    if (!m_lwdaUtils)
        m_lwdaUtils = new LwdaUtils;
 
    // Allocate CPU buffer.

    size_t sizeRays = sizeOfRayBuffer(numRays, rayFormat);
    size_t sizeHits = sizeOfHitBuffer(numRays, hitFormat);
    if (m_cpuTemp && m_cpuTempSize < sizeRays + sizeHits)
    {
        m_lwdaUtils->hostFree(m_cpuTemp);
        m_cpuTemp = NULL;
    }

    if (!m_cpuTemp)
    {
        m_cpuTemp = m_lwdaUtils->hostAlloc(sizeRays + sizeHits);
        m_cpuTempSize = sizeRays + sizeHits;
    }

    m_lwdaUtils->memcpyDtoHAsync(m_cpuTemp, rays, sizeRays, ( lwdaStream_t )stream);
    traceFromHostMem(bvh, m_cpuTemp, rayFormat, m_cpuTemp + sizeRays, hitFormat, numRays, anyHit, watertight, stream);
    m_lwdaUtils->memcpyHtoDAsync(hits, m_cpuTemp + sizeRays, sizeHits, ( lwdaStream_t )stream);
}

//------------------------------------------------------------------------

size_t prodlib::bvhtools::sizeOfRayBuffer( int numRays, int rayFormat )
{
  switch( rayFormat )
  {
  case RAY_ORIGIN_TMIN_DIRECTION_TMAX: return numRays * 8 * sizeof(float); break;
  case RAY_ORIGIN_DIRECTION:           return numRays * 6 * sizeof(float); break;
  default:                             return 0;
  }
}

//------------------------------------------------------------------------

size_t prodlib::bvhtools::sizeOfHitBuffer( int numHits, int hitFormat )
{
  switch( hitFormat )
  {
  case HIT_T_TRIID_INSTID:      return numHits * 3 * sizeof(float); break;
  case HIT_T_TRIID_INSTID_U_V:  return numHits * 5 * sizeof(float); break;
  case HIT_T_TRIID_U_V:         return numHits * 4 * sizeof(float); break;
  case HIT_T_TRIID:             return numHits * 2 * sizeof(float); break;
  case HIT_T:                   return numHits * 1 * sizeof(float); break;
  case HIT_BITMASK:             return (numHits + 32-1)/32 * sizeof(int); break;
  default:                      return 0;
  }
}
