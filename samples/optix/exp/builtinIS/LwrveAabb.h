/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#pragma once

#include <optix_types.h>

#include <rtcore/interface/types.h>

#include <exp/accel/ExtendedAccelHeader.h>
#include <exp/context/ErrorHandling.h>

namespace optix_exp {

// Quantized optix Line-Swept-Sphere
struct OptixLSS
{
    unsigned char beginX;
    unsigned char beginY;
    unsigned char beginZ;
    unsigned char beginR;
    unsigned char endX;
    unsigned char endY;
    unsigned char endZ;
    unsigned char endR;
};

void makeLwrveLSSs( LWstream               stream,
                    const OptixBuildInput& bi,
                    unsigned int           motionStep,
                    unsigned int           numSplits,
                    LWdeviceptr            segments,
                    unsigned int           numSegments,
                    LWdeviceptr            aabbs,
                    size_t                 aabbStrideInBytes,
                    unsigned int           numLSSs );

void makeLwrveAabbs( LWstream stream, const OptixBuildInput& bi, unsigned int motionStep, unsigned int numSplits, LWdeviceptr aabbs, size_t aabbStrideInBytes );

void makeLwrveSegmentAabbs( LWstream               stream,
                            const OptixBuildInput& bi,
                            unsigned int           motionStep,
                            float                  numSplits,
                            LWdeviceptr            segments,
                            LWdeviceptr            aabbs,
                            size_t                 aabbStrideInBytes,
                            LWdeviceptr            inflectionPoints );

void alignbuiltinISData( LWstream stream, LWdeviceptr outputBuffer );

void copyLwrveVertices( LWstream stream, const OptixBuildInput& bi, LWdeviceptr outputBuffer, size_t vertexOffsetInBytes, unsigned int numMotionSteps );

void copyLwrveNormals( LWstream stream, const OptixBuildInput& bi, LWdeviceptr outputBuffer, size_t normalOffsetInBytes, unsigned int numMotionSteps );

void copyLwrveIndices( LWstream stream, const OptixBuildInput& bi, LWdeviceptr outputBuffer, size_t indexOffsetInBytes, unsigned int vertexOffset );

void storeLwrveSegmentData( LWstream               stream,
                            const OptixBuildInput& bi,
                            LWdeviceptr            indexMap,
                            LWdeviceptr            dataBuffer,
                            unsigned int           indexOffset,
                            unsigned int           vertexOffset,
                            unsigned int           numSplits,
                            unsigned int           numMotionSteps );

void storeAdaptiveLwrveSegmentData( LWstream               stream,
                                    const OptixBuildInput& bi,
                                    LWdeviceptr            indexMap,
                                    LWdeviceptr            dataBuffer,
                                    LWdeviceptr            segments,
                                    LWdeviceptr            inflectionPoints,
                                    unsigned int           indexOffset,
                                    unsigned int           vertexOffset,
                                    unsigned int           numSegments,
                                    unsigned int           numMotionSteps );

void storeLwrveIndexData( LWstream stream, const OptixBuildInput& bi, LWdeviceptr indexMap );

OptixResult copybuiltinISIndexOffsets( LWstream             stream,
                                       ExtendedAccelHeader* header,
                                       LWdeviceptr          tempBuffer,
                                       size_t               tempBufferSizeInBytes,
                                       LWdeviceptr          outputBuffer,
                                       unsigned int         numBuildInputs,
                                       unsigned int*        indexOffsets,
                                       ErrorDetails&        errDetails );

OptixResult copyExtendedAccelHeader( LWstream stream, ExtendedAccelHeader* header, LWdeviceptr outputBuffer, ErrorDetails& errDetails );

void setbuiltinISCompactedBufferSize( LWstream stream, size_t* compactedSizePtr, ExtendedAccelHeader* header );

void compactExtendedBuffer( LWstream stream, LWdeviceptr sourceBuffer, LWdeviceptr outputBuffer, size_t outputBufferSizeInBytes );

}  // namespace optix_exp
