/*
 * Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "motionAabb.h"

//---------------------------------------------------------------
// Trampoline functions to call lwca kernels
//---------------------------------------------------------------

namespace optix_exp {
namespace motion {


// Resamples motion aabbs (with #geometryMotionSteps aabbs regularly distributed
//  over time [geometryTimeBegin,geometryTimeEnd]) to #buildMotionSteps aabbs (regularly
//  distributed over time [buildTimeBegin, buildTimeEnd]) such that the linearly interpolated
//  buildAabb(t) at time t \in [buildTimeBegin, buildTimeEnd] includes the linearly interpolated
//  geometryAabb(t).
// Done for all #geometryPrimCount motion aabbs in parallel.
void resampleMotionAabbsWithRegularDistributionDevice( void*            stream,                 // Size
                                                       const float      inputTimeBegin,
                                                       const float      inputTimeEnd,
                                                       unsigned int     geometryPrimCount,
                                                       const Aabb*      geometryAabbs,          // = geometryMotionSteps * geometryPrimCount
                                                       unsigned int     geometryMotionSteps,
                                                       float            geometryTimeBegin,
                                                       float            geometryTimeEnd,
                                                       Aabb*            buildAabbs,             // = buildMotionSteps * geometryPrimCount
                                                       unsigned int     buildMotionSteps,
                                                       float            buildTimeBegin,
                                                       float            buildTimeEnd );

// Resampling as above, but handles motion aabbs with irregularly distributed aabbs.
// Additional memory is required to store the timestamps of the keys of an input motion aabb.
struct MotionAabbDevice
{
    float        timeBegin;
    float        timeEnd;
    bool         timesRegularDistribution;
    unsigned int timesBufferOffset;  // offset for this motion aabb into an array of timestamps for many motion aabbs (see inputTimes below), should match prefixSum over timesCount of all previous MotionAabbDevice
    unsigned int keyCount;           // number of keys, defines the number of aabbs
    unsigned int aabbBufferOffset;   // offset for this motion aabb into an array of aabbs for many motion aabbs (see buildAabbs below), should match prefixSum over keyCount of all previous MotionAabbDevice
    // border modes? - treat all as BORDER_CLAMP
};
void resampleMotionAabbsDevice( size_t                  motionAabbsCount,                       // Size
                                const MotionAabbDevice* motionAabbs,                            // = motionAabbsCount
                                const Aabb*             inputAabbs,  // = motionAabbs[motionAabbsCount-1].aabbBufferOffset + motionAabbs[motionAabbsCount-1].keyCount (iff tightly packed: =(sum over all MotionAabbDevice::keyCount) )
                                const float*            inputTimes,  // = motionAabbs[motionAabbsCount-1].timesBufferOffset + motionAabbs[motionAabbsCount-1].timesCount (iff tightly packed: =(sum over all MotionAabbDevice::timesCount) )
                                Aabb*                   buildAabbs,  // = buildMotionSteps * motionAabbsCount
                                unsigned int            buildMotionSteps,
                                float                   buildTimeBegin,
                                float                   buildTimeEnd );

// Host variant that handles irregular distribution.
// However, resamples only a single motion aabb.
void resampleMotionAabbsHost( const bool   inputTimesRegularDistribution,
                              const float  inputTimeBegin,
                              const float  inputTimeEnd,
                              const Aabb*  inputAabbs,
                              unsigned int inputAabbsCount,
                              const float* inputTimes,
                              Aabb*        buildAabbs,
                              unsigned int buildMotionSteps,
                              float        buildTimeBegin,
                              float        buildTimeEnd );


}  // namespace motion
}  // namespace optix_exp
