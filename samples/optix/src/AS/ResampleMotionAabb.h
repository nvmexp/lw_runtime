// Copyright (c) 2017, LWPU CORPORATION.
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

#include <LWCA/ComputeCapability.h>
#include <optixu/optixu_aabb.h>

//---------------------------------------------------------------
// Trampoline functions to call lwca kernels
//---------------------------------------------------------------

namespace optix {

struct MotionAabbDevice
{
    // if( timesCount != keyCount ), timesCount must be 2!
    unsigned int timesCount;
    unsigned int timesBufferOffset;
    unsigned int keyCount;
    unsigned int aabbBufferOffset;
    // border modes? - treat all as BORDER_CLAMP
};

// Resamples motion aabbs (with #geometryMotionSteps aabbs regularly distributed
//  over time [geometryTimeBegin,geometryTimeEnd]) to #buildMotionSteps aabbs (regularly
//  distributed over time [buildTimeBegin, buildTimeEnd]) such that the linearly interpolated
//  buildAabb(t) at time t \in [buildTimeBegin, buildTimeEnd] includes the linearly interpolated
//  geometryAabb(t).
// Done for all #geometryPrimCount motion aabbs in parallel.
void resampleMotionAabbsWithRegularDistributionDevice( optix::lwca::ComputeCapability sm_ver,
                                                       void*                          stream,
                                                       unsigned int                   geometryPrimCount,
                                                       const optix::Aabb*             geometryAabbs,
                                                       unsigned int                   geometryMotionSteps,
                                                       float                          geometryTimeBegin,
                                                       float                          geometryTimeEnd,
                                                       optix::Aabb*                   buildAabbs,
                                                       unsigned int                   buildMotionSteps,
                                                       float                          buildTimeBegin,
                                                       float                          buildTimeEnd );

// Resampling as above, but handles motion aabbs with irregularly distributed aabbs.
// Additional memory is required to store the times for the aabbs of an input motion aabb.
void resampleMotionAabbsDevice( optix::lwca::ComputeCapability sm_ver,
                                size_t                         motionAabbsCount,
                                const MotionAabbDevice*        motionAabbs,
                                const optix::Aabb*             inputAabbs,
                                const float*                   inputTimes,
                                optix::Aabb*                   buildAabbs,
                                unsigned int                   buildMotionSteps,
                                float                          buildTimeBegin,
                                float                          buildTimeEnd );

// Host variant of the above variant that handles irregular distribution.
void resampleMotionAabbsHost( const optix::Aabb* inputAabbs,
                              unsigned int       inputAabbsCount,
                              const float*       inputTimes,
                              unsigned int       inputTimesCount,
                              optix::Aabb*       buildAabbs,
                              unsigned int       buildMotionSteps,
                              float              buildTimeBegin,
                              float              buildTimeEnd );

}  // namespace optix
