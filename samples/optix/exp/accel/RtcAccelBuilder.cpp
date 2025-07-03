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

#include <optix_types.h>

#include <rtcore/interface/rtcore.h>
#include <rtcore/interface/types.h>

#include <exp/builtinIS/LwrveAabb.h>
#include <exp/builtinIS/LwrveAdaptiveSplitter.h>
#include <exp/builtinIS/SphereKernels.h>

#include <exp/accel/RtcAccelBuilder.h>

#include <exp/context/DeviceContext.h>
#include <exp/context/ErrorHandling.h>

#include <corelib/math/MathUtil.h>
#include <corelib/misc/String.h>
#include <corelib/system/LwdaDriver.h>
#include <corelib/system/Preprocessor.h>
#include <prodlib/system/Knobs.h>

#include <Util/ContainerAlgorithm.h>

#include <lwca.h>

#include <string.h>

#include <cmath>
#include <vector>

// Alignment for lwrve or sphere data groups, i.e., vertices, normals, indices, index offsets.
#define BUILTIN_IS_DATA_BYTE_ALIGNMENT 16ull
// Alignment of data blocks in adaptive splitting.
#define LWRVE_ADAPTIVE_SPLITTING_ALIGNMENT 4ull

namespace {
// clang-format off
    Knob<float>        k_lwrveSplitFactor( RT_DSTRING( "o7.accel.lwrveSplitFactor" ), 1.6f, RT_DSTRING( "Split factor for lwrve segments. Argument has to be at least 1. Higher values can increase performance at a cost of higher memory consumption." ) );
    Knob<bool>         k_lwrveAdaptiveSplitting( RT_DSTRING( "o7.accel.lwrveAdaptiveSplitting" ), true, RT_DSTRING( "Enable adaptive splitting of lwrve segments." ) );
    Knob<unsigned int> k_lwrveLSS( RT_DSTRING( "o7.accel.lwrveLSS" ), 0, RT_DSTRING( "LSS count per lwrve segments. A value of 0 means no LSS." ) );
    Knob<bool>         k_forceRandomAccess( RT_DSTRING( "o7.accel.forceRandomAccess" ), false, RT_DSTRING( "Force support for random triangle and instance data access for all ASs." ) );
    Knob<std::string>  k_dumpLSS( RT_DSTRING( "o7.accel.dumpLSS" ), "", RT_DSTRING( "Dump the lwrve LSS bounds to a .hair file with of linear lwrves." ) );
    Knob<bool>         k_sphereLowMem( RT_DSTRING( "o7.sphereLowMem" ), true, RT_DSTRING( "Low memory spheres." ) );
// clang-format on
}  // namespace

namespace optix_exp {

size_t getLwrveAabbSize()
{
    return sizeof( OptixAabb )
           + corelib::roundUp( k_lwrveLSS.get() * sizeof( OptixLSS ), static_cast<size_t>( OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ) );
}

OptixResult RtcAccelBuilder::computeMemoryUsage( OptixAccelBufferSizes* bufferSizes )
{
    RtcAccelBufferSizes rtcBufferSizes{};

    if( const RtcResult rtcResult = m_deviceContext->getRtcore().accelComputeMemoryUsage(
            m_deviceContext->getRtcDeviceContext(), &m_rtcAccelOptions,
            static_cast<unsigned int>( m_vecBuildInputs.size() ), m_vecBuildInputs.data(),
            m_vecBuildInputOverridePtrs.empty() ? nullptr : m_vecBuildInputOverridePtrs.data(), &rtcBufferSizes ) )
        return m_errDetails.logDetails( rtcResult, "Failed to compute memory usage" );

    bufferSizes->outputSizeInBytes     = rtcBufferSizes.outputSizeInBytes;
    bufferSizes->tempSizeInBytes       = rtcBufferSizes.tempSizeInBytes;
    bufferSizes->tempUpdateSizeInBytes = rtcBufferSizes.tempUpdateSizeInBytes;

    if( m_buildInputType == OPTIX_BUILD_INPUT_TYPE_LWRVES || m_buildInputType == OPTIX_BUILD_INPUT_TYPE_SPHERES )
    {
        bool isLwrves = m_buildInputType == OPTIX_BUILD_INPUT_TYPE_LWRVES;

        size_t dataSizeInBytes           = 0;
        size_t vertexDataSizeInBytes     = 0;
        size_t indexDataSizeInBytes      = 0;
        size_t remapIndexDataSizeInBytes = 0;

        size_t aabbSizeInBytes               = 0;
        size_t intersectorDataSizeInBytes    = 0;
        size_t lwrveAdaptiveSplitSizeInBytes = 0;
        size_t inflectionPointSizeInBytes    = 0;
        size_t sbtMappingSizeInBytes         = 0;

        const unsigned int headerSize       = EXTENDED_ACCEL_HEADER_SIZE;
        const unsigned int indexOffsetsSize = m_numBuildInputs * sizeof( int );

        const bool inputNormals = isLwrves
                                  && ( m_buildInputs[0].lwrveArray.normalBuffers != nullptr
                                       && m_buildInputs[0].lwrveArray.lwrveType != OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE
                                       && m_buildInputs[0].lwrveArray.lwrveType != OPTIX_PRIMITIVE_TYPE_ROUND_LWBIC_BSPLINE
                                       && m_buildInputs[0].lwrveArray.lwrveType != OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR
                                       && m_buildInputs[0].lwrveArray.lwrveType != OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM );

        if( isLwrves )
        {
            // Compute size of vertices, normals, and indices to be stored in lwrve data.
            // Compute size of aabbs to be stored at the end of the temp buffer.
            for( unsigned int i = 0; i < m_numBuildInputs; ++i )
            {
                vertexDataSizeInBytes += m_rtcAccelOptions.motionSteps
                                         * ( inputNormals ? m_buildInputs[i].lwrveArray.numVertices * 7 * sizeof( float ) :
                                                            m_buildInputs[i].lwrveArray.numVertices * 4 * sizeof( float ) );
                indexDataSizeInBytes += m_buildInputs[i].lwrveArray.numPrimitives * sizeof( int );
                const unsigned int numSegments =
                    static_cast<unsigned int>( m_numSplits * m_buildInputs[i].lwrveArray.numPrimitives );
                aabbSizeInBytes += m_rtcAccelOptions.motionSteps * numSegments * getLwrveAabbSize();
                if( !m_builtinISLowMem )
                    intersectorDataSizeInBytes += numSegments * sizeof( LwrveSegmentData );
                remapIndexDataSizeInBytes += numSegments * sizeof( int );
                if( m_lwrveAdaptiveSplitting )
                {
                    // Adding temp memory for adaptive splitting.
                    lwrveAdaptiveSplitSizeInBytes +=
                        m_buildInputs[i].lwrveArray.numPrimitives * 7 * sizeof( float ) + 2 * numSegments * sizeof( int );
                    inflectionPointSizeInBytes += m_buildInputs[i].lwrveArray.numPrimitives * sizeof( unsigned char );
                }
            }
            lwrveAdaptiveSplitSizeInBytes +=
                corelib::roundUp( inflectionPointSizeInBytes, static_cast<size_t>( LWRVE_ADAPTIVE_SPLITTING_ALIGNMENT ) );
        }
        else
        {
            // Compute size of vertices to be stored in sphere data.
            // Compute size of aabbs to be stored at the end of the temp buffer.
            for( unsigned int i = 0; i < m_numBuildInputs; ++i )
            {
                const unsigned int numVertices = m_buildInputs[i].sphereArray.numVertices;
                vertexDataSizeInBytes += m_rtcAccelOptions.motionSteps * numVertices * 4 * sizeof( float );
                aabbSizeInBytes += m_rtcAccelOptions.motionSteps * numVertices * sizeof( OptixAabb );
                if( !m_builtinISLowMem )
                    intersectorDataSizeInBytes += numVertices * sizeof( SphereIntersectorData );
                else
                    sbtMappingSizeInBytes += m_buildInputs[i].sphereArray.numSbtRecords * sizeof( unsigned int );
            }
        }

        // All groups of builtin primitive data, i.e., vertices, normals, indices, and indexOffsets have to be 16-byte aligned.
        dataSizeInBytes = corelib::roundUp( vertexDataSizeInBytes, static_cast<size_t>( BUILTIN_IS_DATA_BYTE_ALIGNMENT ) )
                          + corelib::roundUp( indexDataSizeInBytes, static_cast<size_t>( BUILTIN_IS_DATA_BYTE_ALIGNMENT ) )
                          + corelib::roundUp( sbtMappingSizeInBytes, static_cast<size_t>( BUILTIN_IS_DATA_BYTE_ALIGNMENT ) )
                          + indexOffsetsSize;

        // Take alignment into account.
        bufferSizes->outputSizeInBytes =
            corelib::roundUp( bufferSizes->outputSizeInBytes, static_cast<size_t>( BUILTIN_IS_DATA_BYTE_ALIGNMENT ) )
            + dataSizeInBytes + headerSize;
        bufferSizes->tempSizeInBytes =
            corelib::roundUp( bufferSizes->tempSizeInBytes, static_cast<size_t>( OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ) )
            + corelib::roundUp( lwrveAdaptiveSplitSizeInBytes, static_cast<size_t>( OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ) ) + aabbSizeInBytes
            + corelib::roundUp( remapIndexDataSizeInBytes, static_cast<size_t>( OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ) )
            + corelib::roundUp( static_cast<size_t>( intersectorDataSizeInBytes ), static_cast<size_t>( OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ) )
            + corelib::roundUp( static_cast<size_t>( indexOffsetsSize ), static_cast<size_t>( OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ) );

        // No need for lwrveAdaptiveSplitSizeInBytes because updates for adaptively split lwrve build inputs are not supported.
        bufferSizes->tempUpdateSizeInBytes =
            corelib::roundUp( bufferSizes->tempUpdateSizeInBytes, static_cast<size_t>( OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ) ) + aabbSizeInBytes
            + corelib::roundUp( remapIndexDataSizeInBytes, static_cast<size_t>( OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ) )
            + corelib::roundUp( static_cast<size_t>( intersectorDataSizeInBytes ), static_cast<size_t>( OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ) )
            + corelib::roundUp( static_cast<size_t>( indexOffsetsSize ), static_cast<size_t>( OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ) );
    }

    return OPTIX_SUCCESS;
}

bool RtcAccelBuilder::buildInputsAreEmpty() const
{
// TODO: 2650681 OptiX 7 should treat a traversable handle of zero as a no-op everywhere
#if 1
    return false;
#else
    return optix::algorithm::all_of( m_vecBuildInputs, [this]( const RtcBuildInput& buildInput ) {
        return buildInputIsEmpty( buildInput );
    } );
#endif
}

bool RtcAccelBuilder::buildInputIsEmpty( const RtcBuildInput& buildInput ) const
{
    switch( buildInput.type )
    {
        case RTC_BUILD_INPUT_TYPE_TRIANGLE_ARRAY:
            return triangleArrayBuildInputIsEmpty( buildInput.triangleArray );

        case RTC_BUILD_INPUT_TYPE_FAT_INSTANCE_ARRAY:
            return aabbArrayBuildInputIsEmpty( buildInput.aabbArray );

        // TODO: handle other input types
        case RTC_BUILD_INPUT_TYPE_AABB_ARRAY:
        case RTC_BUILD_INPUT_TYPE_INSTANCE_ARRAY:
        case RTC_BUILD_INPUT_TYPE_INSTANCE_POINTERS:
        case RTC_BUILD_INPUT_TYPE_MOTION_TRIANGLE_ARRAYS:
        case RTC_BUILD_INPUT_TYPE_MOTION_AABB_ARRAYS:
        case RTC_BUILD_INPUT_TYPE_FAT_INSTANCE_POINTERS:
            break;
    }

    return false;
}

bool RtcAccelBuilder::triangleArrayBuildInputIsEmpty( const RtcBuildInputTriangleArray& triangles ) const
{
    return triangles.numVertices == 0U && triangles.numIndices == 0U;
}

bool RtcAccelBuilder::aabbArrayBuildInputIsEmpty( const RtcBuildInputAabbArray& aabbArray ) const
{
    return aabbArray.numAabbs == 0U;
}

void RtcAccelBuilder::setRtcAccelBuffers( LWdeviceptr          tempBuffer,
                                          size_t               tempBufferSizeInBytes,
                                          LWdeviceptr          outputBuffer,
                                          size_t               outputBufferSizeInBytes,
                                          ExtendedAccelHeader& extendedHeader )
{
    const bool isBuiltinIntersector =
        ( m_buildInputType == OPTIX_BUILD_INPUT_TYPE_LWRVES || m_buildInputType == OPTIX_BUILD_INPUT_TYPE_SPHERES );

    rtcAccelBuffers.input  = 0;
    rtcAccelBuffers.output = outputBuffer + ( isBuiltinIntersector ? EXTENDED_ACCEL_HEADER_SIZE : 0 );
    rtcAccelBuffers.outputSizeInBytes =
        outputBufferSizeInBytes - ( isBuiltinIntersector ? EXTENDED_ACCEL_HEADER_SIZE + extendedHeader.dataSize : 0 );
    rtcAccelBuffers.temp            = tempBuffer;
    rtcAccelBuffers.tempSizeInBytes = tempBufferSizeInBytes;
}

void RtcAccelBuilder::setRtcEmittedProperties( const OptixAccelEmitDesc* emittedProperties, unsigned int numEmittedProperties )
{
    const bool hasMotion = ( m_accelOptions->motionOptions.numKeys >= 2 );  // allow 0 or 1 to disable motion

    rtcEmittedProperties.reserve( numEmittedProperties );
    rtcEmittedProperties.clear();
    for( unsigned int i = 0; i < numEmittedProperties; ++i )
    {
        RtcAccelEmitDesc rtcEmittedProperty;
        rtcEmittedProperty.resultVA = emittedProperties[i].result;
        switch( emittedProperties[i].type )
        {
            case OPTIX_PROPERTY_TYPE_COMPACTED_SIZE:
                rtcEmittedProperty.type = RTC_PROPERTY_TYPE_COMPACTED_SIZE;
                break;
            case OPTIX_PROPERTY_TYPE_AABBS:
                if( hasMotion )
                {
                    // WAR: motion blur AABBs can lwrrently not yet be emitted by rtcore during the build
                    // and instead need to be queried explicitly after the build
                    motionAabbsEmitDescIndex.push_back( i );
                    continue;
                }
                rtcEmittedProperty.type = RTC_PROPERTY_TYPE_AABB;
                break;
        }
        rtcEmittedProperties.push_back( rtcEmittedProperty );
    }
}

void RtcAccelBuilder::addSizeToRtcEmittedProperties( LWdeviceptr                    outputBuffer,
                                                     std::vector<RtcAccelEmitDesc>& rtcEmittedProperties,
                                                     bool&                          compactedProperty )
{
    // An emitted property for the BVH size is added.
    // Check if there is already a compacted size property in the property vector, otherwise add it.

    RtcAccelEmitDesc sizeProperty;

    sizeProperty.type     = RTC_PROPERTY_TYPE_LWRRENT_SIZE;
    sizeProperty.resultVA = outputBuffer + offsetof( ExtendedAccelHeader, dataOffset );

    rtcEmittedProperties.push_back( sizeProperty );

    // If there is no compacted size property in the property vector, add one.
    for( auto it = rtcEmittedProperties.begin(); it != rtcEmittedProperties.end() && !compactedProperty; ++it )
    {
        if( it->type == RTC_PROPERTY_TYPE_COMPACTED_SIZE )
        {
            compactedProperty = true;
        }
    }
    if( !compactedProperty )
    {
        RtcAccelEmitDesc compactedSizeProperty;

        compactedSizeProperty.type     = RTC_PROPERTY_TYPE_COMPACTED_SIZE;
        compactedSizeProperty.resultVA = outputBuffer + offsetof( ExtendedAccelHeader, dataCompactedOffset );

        rtcEmittedProperties.push_back( compactedSizeProperty );
    }
}

OptixResult RtcAccelBuilder::build( LWstream                  stream,
                                    LWdeviceptr               tempBuffer,
                                    size_t                    tempBufferSizeInBytes,
                                    LWdeviceptr               outputBuffer,
                                    size_t                    outputBufferSizeInBytes,
                                    OptixTraversableHandle*   outputHandle,
                                    const OptixAccelEmitDesc* emittedProperties,
                                    unsigned                  numEmittedProperties )
{
    ExtendedAccelHeader                   extendedHeader = {};
    std::vector<std::vector<LWdeviceptr>> lwrveAabbs;
    std::vector<std::vector<LWdeviceptr>> sphereAabbs;
    if( m_buildInputType == OPTIX_BUILD_INPUT_TYPE_LWRVES )
    {
        if( m_lwrveAdaptiveSplitting )
        {
            if( const OptixResult result = splitLwrveAdaptive( stream, tempBuffer, tempBufferSizeInBytes ) )
            {
                return result;
            }
        }
        if( const OptixResult result = computeLwrveAabbs( stream, tempBuffer, tempBufferSizeInBytes, lwrveAabbs ) )
        {
            return result;
        }

        if( k_lwrveLSS.get() )
        {
            if( const OptixResult result = computeLwrveLSSs( stream, tempBuffer, tempBufferSizeInBytes ) )
            {
                return result;
            }
        }

        if( const OptixResult result = initExtendedAccelHeader( stream, outputBuffer, &extendedHeader ) )
        {
            return result;
        }
    }
    else if( m_buildInputType == OPTIX_BUILD_INPUT_TYPE_SPHERES )
    {
        if( const OptixResult result = computeSphereAabbs( stream, tempBuffer, tempBufferSizeInBytes, sphereAabbs ) )
        {
            return result;
        }

        if( const OptixResult result = initExtendedAccelHeader( stream, outputBuffer, &extendedHeader ) )
        {
            return result;
        }
    }
    setRtcAccelBuffers( tempBuffer, tempBufferSizeInBytes, outputBuffer, outputBufferSizeInBytes, extendedHeader );
    setRtcEmittedProperties( emittedProperties, numEmittedProperties );

    bool compactedProperty = false;
    if( m_buildInputType == OPTIX_BUILD_INPUT_TYPE_LWRVES || m_buildInputType == OPTIX_BUILD_INPUT_TYPE_SPHERES )
    {
        // For compaction, the BVH size and the compacted BVH size need to be determined and stored inside the lwrve data header.
        // Check whether there is already a compacted size property inside the property vector.
        addSizeToRtcEmittedProperties( outputBuffer, rtcEmittedProperties, compactedProperty );
    }

    if( buildInputsAreEmpty() )
    {
        const OptixTraversableHandle NULL_TRAVERSABLE_HANDLE = 0ULL;
        *outputHandle                                        = NULL_TRAVERSABLE_HANDLE;
        return OPTIX_SUCCESS;
    }

    ScopedCommandList commandList( m_deviceContext );
    if( const OptixResult result = commandList.init( stream, m_errDetails ) )
    {
        return result;
    }

    if( const RtcResult rtcResult = m_deviceContext->getRtcore().accelBuild(
            commandList.get(), &m_rtcAccelOptions, static_cast<unsigned int>( m_vecBuildInputs.size() ),
            m_vecBuildInputs.data(), m_vecBuildInputOverridePtrs.empty() ? nullptr : m_vecBuildInputOverridePtrs.data(),
            &rtcAccelBuffers, static_cast<unsigned int>( rtcEmittedProperties.size() ), rtcEmittedProperties.data() ) )
    {
        commandList.destroy( m_errDetails );
        return m_errDetails.logDetails( rtcResult, "Failed to build acceleration structure" );
    }

    // the motion aabbs are emitted after construction using accelEmitProperties because accelBuild doesn't support these yet.
    for( unsigned int i : motionAabbsEmitDescIndex )
    {
        if( const RtcResult rtcResult = m_deviceContext->getRtcore().accelEmitProperties(
                commandList.get(), &outputBuffer, 1, RTC_PROPERTY_TYPE_MOTION_AABBS, emittedProperties[i].result,
                sizeof( RtcEmittedAccelPropertyAabb ) * m_accelOptions->motionOptions.numKeys ) )
        {
            commandList.destroy( m_errDetails );
            return m_errDetails.logDetails( rtcResult, "Failed to build acceleration structure" );
        }
    }

    if( const OptixResult result = commandList.destroy( m_errDetails ) )
    {
        return result;
    }

    if( outputHandle != nullptr )
    {
        RtcTraversableHandle rtcTraversableHandle = 0;

        if( m_vecBuildInputs.empty() )
            return logIlwalidValue( "Build inputs empty" );

        const RtcTraversableType rtcTraversableType = m_isIAS ? RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL :
                                                                ( m_buildInputType == OPTIX_BUILD_INPUT_TYPE_LWRVES
                                                                  || m_buildInputType == OPTIX_BUILD_INPUT_TYPE_SPHERES ) ?
                                                                RTC_TRAVERSABLE_TYPE_EXTENDED_BOTTOM_LEVEL_ACCEL :
                                                                RTC_TRAVERSABLE_TYPE_BOTTOM_LEVEL_ACCEL;
        if( const RtcResult rtcResult = m_deviceContext->getRtcore().colwertPointerToTraversableHandle(
                m_deviceContext->getRtcDeviceContext(), rtcAccelBuffers.output, rtcTraversableType,
                m_rtcAccelOptions.accelType, &rtcTraversableHandle ) )
        {
            return m_errDetails.logDetails( rtcResult, "Failed to colwert pointer to traversable handle" );
        }
        *outputHandle = rtcTraversableHandle;
    }

    if( m_buildInputType == OPTIX_BUILD_INPUT_TYPE_LWRVES || m_buildInputType == OPTIX_BUILD_INPUT_TYPE_SPHERES )
    {
        if( compactedProperty )
        {
            // Copy the size of the compacted BVH to the lwrve/sphere data header.
            // Increase the property result for including lwrve/sphere data.

            for( auto it = rtcEmittedProperties.begin(); it != rtcEmittedProperties.end(); ++it )
            {
                if( it->type == RTC_PROPERTY_TYPE_COMPACTED_SIZE )
                {
                    // only for primitives with builtin intersector (lwrves and spheres)
                    setbuiltinISCompactedBufferSize( stream, (size_t*)( it->resultVA ), (ExtendedAccelHeader*)outputBuffer );
                    break;
                }
            }
        }

        if( const OptixResult result = appendbuiltinISData( stream, tempBuffer, tempBufferSizeInBytes, outputBuffer, &extendedHeader ) )
        {
            return result;
        }
    }

    return OPTIX_SUCCESS;
}

RtcAccelBuilder::RtcAccelBuilder( DeviceContext*                context,
                                  const OptixAccelBuildOptions* accelOptions,
                                  bool                          computeMemory,
                                  ErrorDetails&                 errDetails,
                                  unsigned                      maxPrimsPerGAS,
                                  unsigned                      maxSbtRecordsPerGAS,
                                  unsigned                      maxInstancesPerIAS,
                                  int                           abiVersion,
                                  bool                          hasTTU,
                                  bool                          hasMotionTTU )
    : m_deviceContext( context )
    , m_accelOptions( accelOptions )
    , m_computeMemory( computeMemory )
    , m_errDetails( errDetails )
    , m_maxPrimsPerGAS( maxPrimsPerGAS )
    , m_maxSbtRecordsPerGAS( maxSbtRecordsPerGAS )
    , m_maxInstancesPerIAS( maxInstancesPerIAS )
    , m_abiVersion( (OptixABI)abiVersion )
    , m_hasTTU( hasTTU )
    , m_hasMotionTTU( hasMotionTTU )
{
}

RtcAccelBuilder::RtcAccelBuilder( DeviceContext* deviceContext, const OptixAccelBuildOptions* accelOptions, bool computeMemory, ErrorDetails& errDetails )
    : m_deviceContext( deviceContext )
    , m_accelOptions( accelOptions )
    , m_computeMemory( computeMemory )
    , m_errDetails( errDetails )
    , m_maxPrimsPerGAS( deviceContext->getMaxPrimsPerGAS() )
    , m_maxSbtRecordsPerGAS( m_deviceContext->getMaxSbtRecordsPerGAS() )
    , m_maxInstancesPerIAS( m_deviceContext->getMaxInstancesPerIAS() )
    , m_abiVersion( m_deviceContext->getAbiVersion() )
    , m_hasTTU( m_deviceContext->hasTTU() )
    , m_hasMotionTTU( m_deviceContext->hasMotionTTU() )
{
}

// Get internal split factor for lwrve segments.
// This is computed from the knob k_lwrveSplitFactor, the build flags and whether the geometry flag requires single any hit call.
// If the total number of segments in the GAS times the number of splits exceeds the maximum number of primitives in the GAS,
// the number of splits needs to be reduced.
// It's assumed that the aclwmulated number of primitives over all build inputs has already been computed, in validateBuildInputs.
//
void RtcAccelBuilder::getLwrveSplitFactor( const OptixBuildInput* buildInputs, unsigned int numBuildInputs )
{
    if( m_abiVersion >= OptixABI::ABI_51 )
        m_builtinISLowMem = m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    if( m_builtinISLowMem )
    {
        // no splitting, i.e. m_numSplits = 1 and non-adaptive
        m_lwrveAdaptiveSplitting = false;
        m_numSplits              = 1.f;
        return;
    }

    // For restricting the number of intersections per lwrve segment to one, do not split the lwrve segment.
    for( unsigned int i = 0; i < numBuildInputs; ++i )
    {
        if( buildInputs[i].lwrveArray.flag & OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL )
        {
            // no splitting, i.e. m_numSplits = 1 and non-adaptive
            m_lwrveAdaptiveSplitting = false;
            m_numSplits              = 1.f;
            return;
        }
    }

    // Default is adaptive splitting.
    // Fallback from adaptive splitting to uniform (non-adaptive) splitting, including no splitting, if
    // - the knob "o7.accel.lwrveAdaptiveSplitting" is set to false, or
    // - or the knob-specified split factor is <= 1 or >= 8, or
    // - OPTIX_BUILD_FLAG_PERFER_FAST_BUILD is set

    bool uniform = ( k_lwrveAdaptiveSplitting.get() == false );
    if( k_lwrveSplitFactor.get() <= 1.f || k_lwrveSplitFactor.get() >= 8.f
        || ( m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_PREFER_FAST_BUILD ) )
        uniform = true;

    if( !uniform && m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE )
    {
        // To support refits with adaptive splitting, the refit needs to preserve the topology of the adaptively split lwrves.
        // Lwrrently the adaptive lwrve splitting would just re-run, possibly resulting in a different segment topology.
        // Fall back to uniform lwrve splitting if refits are allowed.
        std::string msg =
            "Updating of adaptively split lwrve build inputs is not supported, fall back to uniform splitting.";
        m_deviceContext->getLogger().callback( DeviceContextLogger::LOG_LEVEL::Warning, "ACCEL_BUILDER", msg.c_str() );
        uniform = true;
    }

    if( uniform )
    {
        // non-adaptive splitting, because of knob settings or OPTIX_BUILD_FLAG_PREFER_FAST_BUILD (=split factor 1)
        m_lwrveAdaptiveSplitting = false;

        unsigned int numSplits = 1;
        if( m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_PREFER_FAST_TRACE )
            numSplits = 4;
        else if( m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_PREFER_FAST_BUILD )
            numSplits = 1;
        else
            numSplits = 2;

        if( k_lwrveSplitFactor.isSet() )
        {
            if( k_lwrveSplitFactor.get() < 1.f )
                numSplits = 1;
            else
                numSplits = static_cast<unsigned int>( k_lwrveSplitFactor.get() );
        }
        if( numSplits > 1 && m_numPrimsInGASForValidation * numSplits > m_maxPrimsPerGAS )
        {
            std::string msg = "Reaching memory limits when building AS for lwrves, performance might be affected.";
            m_deviceContext->getLogger().callback( DeviceContextLogger::LOG_LEVEL::Warning, "ACCEL_BUILDER", msg.c_str() );
            numSplits = m_maxPrimsPerGAS / m_numPrimsInGASForValidation;
            // From the validation routines we know that m_numPrimsInGASForValidation <= m_maxPrimsPerGAS,
            // which means numSplits >= 1.
        }
        m_numSplits = (float)numSplits;
    }
    else
    {
        // adaptive splitting
        m_lwrveAdaptiveSplitting = true;

        if( m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_PREFER_FAST_TRACE )
            m_numSplits = 3.8f;
        else
            // default
            m_numSplits = 1.6f;

        if( k_lwrveSplitFactor.isSet() )
            m_numSplits = k_lwrveSplitFactor.get();

        if( static_cast<unsigned int>( m_numPrimsInGASForValidation * m_numSplits ) > m_maxPrimsPerGAS )
        {
            std::string msg = "Reaching memory limits when building AS for lwrves, performance might be affected.";
            m_deviceContext->getLogger().callback( DeviceContextLogger::LOG_LEVEL::Warning, "ACCEL_BUILDER", msg.c_str() );
            m_numSplits = (float)m_maxPrimsPerGAS / (float)m_numPrimsInGASForValidation;
            // From the validation routines we know that m_numPrimsInGASForValidation <= m_maxPrimsPerGAS,
            // which means m_numSplits >= 1.
        }
    }
}

// To avoid mocking DeviceContext in tests for RtcAccelBuilder, query all DeviceContext
// properties in the c'tor and use the overload to allow test cases to set these properties
// explicitly.  Then the init() method here can be tested to validate hidden inputs it
// passes to rtCore.
//
OptixResult RtcAccelBuilder::init( const OptixBuildInput* buildInputs, unsigned int numBuildInputs )
{
    // the m_vecBuildInputOverridePtrs is populated with pointers to build input overrides entries.
    // the vector needs to be pre-sized as resizing may ilwalidate pointers.
    m_vecBuildInputOverrides.resize( numBuildInputs );

    m_buildInputs    = buildInputs;
    m_numBuildInputs = numBuildInputs;
    m_buildInputType = buildInputs[0].type;
    m_isIAS = m_buildInputType == OPTIX_BUILD_INPUT_TYPE_INSTANCES || m_buildInputType == OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS;

    if( m_buildInputType == OPTIX_BUILD_INPUT_TYPE_LWRVES )
        m_builtinPrimitiveType = buildInputs[0].lwrveArray.lwrveType;
    else if( m_buildInputType == OPTIX_BUILD_INPUT_TYPE_SPHERES )
        m_builtinPrimitiveType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    else if( m_buildInputType == OPTIX_BUILD_INPUT_TYPE_TRIANGLES )
        m_builtinPrimitiveType = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
    else
        m_builtinPrimitiveType = OPTIX_PRIMITIVE_TYPE_LWSTOM;

    const bool           hasMotion = ( m_accelOptions->motionOptions.numKeys >= 2 );  // allow 0 or 1 to disable motion
    const unsigned short motionNumKeys = hasMotion ? m_accelOptions->motionOptions.numKeys : 1;
    // the code below assumes numKeys >= 1
    m_numPrimsInGASForValidation      = 0;
    m_numSbtRecordsInGASForValidation = 0;

    for( unsigned int i = 0; i < numBuildInputs; ++i )
    {
        if( const OptixResult validateResult = validateBuildInput( i, buildInputs[i], hasMotion, motionNumKeys ) )
            return validateResult;
    }

    // Get m_numSplits. This needs to be done after input validation and before colwersion.
    if( m_buildInputType == OPTIX_BUILD_INPUT_TYPE_LWRVES )
    {
        getLwrveSplitFactor( buildInputs, numBuildInputs );
    }
    else if( m_buildInputType == OPTIX_BUILD_INPUT_TYPE_SPHERES )
    {
        m_builtinISLowMem = k_sphereLowMem.get();
    }

    if( m_buildInputType == OPTIX_BUILD_INPUT_TYPE_TRIANGLES )
    {
        // FIXME: colwertTrianglesBuildInputs always returns OPTIX_SUCCESS, is this useful then?
        if( const OptixResult colwertResult = colwertTrianglesBuildInputs( buildInputs, numBuildInputs, hasMotion, motionNumKeys ) )
            return colwertResult;
    }
    else
    {
        m_numRtcLwrvesInGAS  = 0;
        for( unsigned int i = 0; i < numBuildInputs; ++i )
        {
            if( const OptixResult colwertResult = colwertBuildInput( i, buildInputs[i], hasMotion, motionNumKeys ) )
                return colwertResult;
        }
    }

    memset( &m_rtcAccelOptions, 0, sizeof( RtcAccelOptions ) );

    if( hasMotion )
    {
        if( m_hasMotionTTU )
        {
            m_rtcAccelOptions.accelType = RTC_ACCEL_TYPE_MTTU;
        }
        else
        {
            m_rtcAccelOptions.accelType = RTC_ACCEL_TYPE_MBVH2;
        }
    }
    else
    {
        if( m_hasTTU )
        {
            m_rtcAccelOptions.accelType = RTC_ACCEL_TYPE_TTU;
        }
        else
        {
            m_rtcAccelOptions.accelType = RTC_ACCEL_TYPE_BVH2;
        }
    }

    m_rtcAccelOptions.useUniversalFormat = true;

    m_rtcAccelOptions.refit = m_accelOptions->operation == OPTIX_BUILD_OPERATION_UPDATE;

    m_rtcAccelOptions.bakeTriangles              = true;
    m_rtcAccelOptions.usePrimBits                = true;
    m_rtcAccelOptions.useRemapForPrimBits        = false;
    m_rtcAccelOptions.enableBuildReordering      = false;
    m_rtcAccelOptions.clampAabbsToValidRange     = hasMotion ? false : true;
    m_rtcAccelOptions.highPrecisionMath          = true;
    m_rtcAccelOptions.useProvizBuilderStrategies = true;

    // not sure if we should pass 0 or 1 to disable motion. Right now both should work.
    m_rtcAccelOptions.motionSteps = motionNumKeys;
    m_rtcAccelOptions.motionFlags = 0;
    if( hasMotion )
    {
        // note that accelOptions are already validated at this point
        m_rtcAccelOptions.motionTimeBegin = m_accelOptions->motionOptions.timeBegin;
        m_rtcAccelOptions.motionTimeEnd   = m_accelOptions->motionOptions.timeEnd;

        if( m_accelOptions->motionOptions.flags & OPTIX_MOTION_FLAG_START_VANISH )
            m_rtcAccelOptions.motionFlags |= RTC_MOTION_FLAG_START_VANISH;
        if( m_accelOptions->motionOptions.flags & OPTIX_MOTION_FLAG_END_VANISH )
            m_rtcAccelOptions.motionFlags |= RTC_MOTION_FLAG_END_VANISH;
    }

    m_rtcAccelOptions.buildFlags = 0;
    if( m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE )
        m_rtcAccelOptions.buildFlags |= RTC_BUILD_FLAG_ALLOW_UPDATE;
    if( m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION )
        m_rtcAccelOptions.buildFlags |= RTC_BUILD_FLAG_ALLOW_COMPACTION;
    if( m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_PREFER_FAST_TRACE )
        m_rtcAccelOptions.buildFlags |= RTC_BUILD_FLAG_PREFER_FAST_TRACE;
    if( m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_PREFER_FAST_BUILD )
        m_rtcAccelOptions.buildFlags |= RTC_BUILD_FLAG_PREFER_FAST_BUILD;
    if( m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS )
        m_rtcAccelOptions.buildFlags |= RTC_BUILD_FLAG_ALLOW_DATA_ACCESS;
    if( m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS )
        m_rtcAccelOptions.buildFlags |= RTC_BUILD_FLAG_ALLOW_DATA_ACCESS;

    if( !k_forceRandomAccess.isDefault() )
        m_rtcAccelOptions.buildFlags |= RTC_BUILD_FLAG_ALLOW_DATA_ACCESS;

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    if( m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_IAS_REFERENCES_GAS_WITH_DMM )
        m_rtcAccelOptions.buildFlags |= RTC_BUILD_FLAG_REFERENCED_BLAS_HAS_DMM;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::validateBuildInput( unsigned int i, const OptixBuildInput& bi, bool hasMotion, unsigned short motionNumKeys )
{
    if( const OptixResult result = validateBuildInputType( i, bi ) )
    {
        return result;
    }

    switch( bi.type )
    {
        case OPTIX_BUILD_INPUT_TYPE_TRIANGLES:
            if( const OptixResult result = validateTrianglesBuildInput( i, bi.triangleArray, motionNumKeys ) )
                return result;
            if( const OptixResult result = validateBuildOverridesInput( i, bi ) )
                return result;
            break;

        case OPTIX_BUILD_INPUT_TYPE_LWRVES:
            if( const OptixResult result = validateLwrvesBuildInput( i, bi.lwrveArray, motionNumKeys ) )
                return result;
            if( const OptixResult result = validateLwrvesBuildOverridesInput( i, bi, motionNumKeys ) )
                return result;
            break;

        case OPTIX_BUILD_INPUT_TYPE_SPHERES:
            if( const OptixResult result = validateSpheresBuildInput( i, bi.sphereArray, motionNumKeys ) )
                return result;
            if( const OptixResult result = validateBuildOverridesInput( i, bi ) )
                return result;
            break;

        case OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES:
            if( const OptixResult result = validateLwstomPrimitivesBuildInput( i, bi.lwstomPrimitiveArray, motionNumKeys ) )
                return result;
            if( const OptixResult result = validateBuildOverridesInput( i, bi ) )
                return result;
            break;

        case OPTIX_BUILD_INPUT_TYPE_INSTANCES:
        case OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS:
            if( const OptixResult result = validateInstancesBuildInput( i, bi.type, bi.instanceArray, hasMotion, motionNumKeys ) )
                return result;
            break;

        default:
            return logIlwalidValue( corelib::stringf( R"msg(Invalid build type (0x%x) for "buildInputs[%u].type")msg",
                                                      static_cast<unsigned int>( bi.type ), i ) );
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::colwertBuildInput( const unsigned int i, const OptixBuildInput& bi, const bool hasMotion, const unsigned short motionNumKeys )
{
    switch( bi.type )
    {
        case OPTIX_BUILD_INPUT_TYPE_TRIANGLES:
            // handled separately
            return OPTIX_SUCCESS;

        case OPTIX_BUILD_INPUT_TYPE_LWRVES:
            return colwertLwrvesBuildInput( i, bi.lwrveArray, hasMotion, motionNumKeys );

        case OPTIX_BUILD_INPUT_TYPE_SPHERES:
            return colwertSpheresBuildInput( i, bi.sphereArray, hasMotion, motionNumKeys );

        case OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES:
            return colwertLwstomPrimitivesBuildInput( i, bi.lwstomPrimitiveArray, hasMotion );

        case OPTIX_BUILD_INPUT_TYPE_INSTANCES:
        case OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS:
            return colwertInstancesBuildInput( bi.type, bi.instanceArray );
    }

    return logIlwalidValue( corelib::stringf( R"msg(Invalid build type (0x%x) for "buildInputs[%u].type")msg",
                                              static_cast<unsigned int>( bi.type ), i ) );
}

OptixResult RtcAccelBuilder::validateBuildInputType( unsigned int i, const OptixBuildInput& bi )
{
    if( bi.type != m_buildInputType )
    {
        return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].type" != "buildInputs[0].type". All build inputs for geometry acceleration structures must have the same type)msg",
                                                  i ) );
    }

    if( m_buildInputType == OPTIX_BUILD_INPUT_TYPE_LWRVES )
    {
        if( bi.lwrveArray.lwrveType != m_builtinPrimitiveType )
        {
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwrveArray.type" != "buildInputs[0].lwrveArray.type". All lwrve build inputs for geometry acceleration structures must have the same lwrve type)msg",
                                                      i ) );
        }
    }

    return OPTIX_SUCCESS;
}

template <typename T>
T selectBuildInput( const OptixBuildInput& bi, const T triangles, const T aabb, const T sphere )
{
    return bi.type == OPTIX_BUILD_INPUT_TYPE_TRIANGLES ? triangles :
                                                         bi.type == OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES ? aabb : sphere;
}

#define SELECT_BUILD_INPUT( bi_, field_ )                                                                              \
    selectBuildInput( bi, bi.triangleArray.field_, bi.lwstomPrimitiveArray.field_, bi.sphereArray.field_ )

OptixResult RtcAccelBuilder::validateBuildOverridesInput( unsigned int i, const OptixBuildInput& bi )
{
    const unsigned int        numSbtRecords               = SELECT_BUILD_INPUT( bi, numSbtRecords );
    const unsigned int* const flags                       = SELECT_BUILD_INPUT( bi, flags );
    const LWdeviceptr         sbtIndexOffsetBuffer        = SELECT_BUILD_INPUT( bi, sbtIndexOffsetBuffer );
    const unsigned int        sbtIndexOffsetSizeInBytes   = SELECT_BUILD_INPUT( bi, sbtIndexOffsetSizeInBytes );
    const unsigned int        sbtIndexOffsetStrideInBytes = SELECT_BUILD_INPUT( bi, sbtIndexOffsetStrideInBytes );
    const unsigned int        primitiveIndexOffset        = SELECT_BUILD_INPUT( bi, primitiveIndexOffset );
    const unsigned int        numPrimitives =
        selectBuildInput( bi,
                          bi.triangleArray.indexFormat != OPTIX_INDICES_FORMAT_NONE ? bi.triangleArray.numIndexTriplets :
                                                                                      bi.triangleArray.numVertices / 3,
                          bi.lwstomPrimitiveArray.numPrimitives, bi.sphereArray.numVertices );

    const char* const typeS = bi.type == OPTIX_BUILD_INPUT_TYPE_TRIANGLES ?
                                  "triangleArray" :
                                  bi.type == OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES ? "lwstomPrimitiveArray" : "sphereArray";

    m_numSbtRecordsInGASForValidation += numSbtRecords;
    if( m_numSbtRecordsInGASForValidation > m_maxSbtRecordsPerGAS )
    {
        // note m_vecBuildInputOverrides.size() == numBuildInputs
        return logIlwalidValue( corelib::stringf(
            R"msg(Sum of "numSbtRecords" (%llu) over first %u build inputs out of %u build inputs exceeds device context property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS (%u).)msg",
            m_numSbtRecordsInGASForValidation, i + 1, (uint32_t)m_vecBuildInputOverrides.size(), m_maxSbtRecordsPerGAS ) );
    }

    if( numSbtRecords == 0 )
    {
        return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].%s.numSbtRecords" is zero)msg", i, typeS ) );
    }
    else if( numSbtRecords > 1 )
    {
        if( !m_computeMemory && sbtIndexOffsetBuffer == 0 )
        {
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].%s", "numSbtRecords" is larger than one, but "sbtIndexOffsetBuffer" is null)msg",
                                                      i, typeS ) );
        }
        if( sbtIndexOffsetSizeInBytes != 1 && sbtIndexOffsetSizeInBytes != 2 && sbtIndexOffsetSizeInBytes != 4 )
        {
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].%s.sbtIndexOffsetSizeInBytes" (%u) must be either 1, 2, or 4.)msg",
                                                      i, typeS, sbtIndexOffsetSizeInBytes ) );
        }
        if( sbtIndexOffsetStrideInBytes != 0 && sbtIndexOffsetStrideInBytes < sbtIndexOffsetSizeInBytes )
        {
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].%s", "sbtIndexOffsetStrideInBytes" is smaller than "sbtIndexOffsetSizeInBytes")msg",
                                                      i, typeS ) );
        }
    }
    //else
    // ignore sbtIndexOffsetBuffer, sbtIndexOffsetSizeInBytes, sbtIndexOffsetStrideInBytes

    if( flags == nullptr )
    {
        return logIlwalidValue( corelib::stringf( R"msg(Invalid value (0) for "buildInputs[%u].%s.flags")msg", i, typeS ) );
    }
    for( unsigned int j = 0; j < numSbtRecords; ++j )
    {
        if( flags[j]
            & ~( OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_LWLLING
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
                 | OPTIX_GEOMETRY_FLAG_REPLACEABLE_VM_ARRAY
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
                 ) )
        {
            return logIlwalidValue( corelib::stringf( R"msg(Invalid value (%u) for "buildInputs[%u].%s.flags[%u]")msg",
                                                      flags[j], i, typeS, j ) );
        }
    }

    m_numPrimsInGASForValidation += numPrimitives;
    if( m_numPrimsInGASForValidation > m_maxPrimsPerGAS )
    {
        // note m_vecBuildInputOverrides.size() == numBuildInputs
        return logIlwalidValue( corelib::stringf(
            R"msg(Sum of number of triangles/primitives (%llu) over first %u build inputs out of %u build inputs exceeds device context property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS (%u).)msg",
            m_numPrimsInGASForValidation, i + 1, (uint32_t)m_vecBuildInputOverrides.size(), m_maxPrimsPerGAS ) );
    }

    if( (uint64_t)numPrimitives + primitiveIndexOffset > (uint64_t)std::numeric_limits<uint32_t>::max() )
    {
        return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].%s": number of triangles/primitives + "primitiveIndexOffset" overflows 32 bits.)msg",
                                                  i, typeS ) );
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::validateLwrvesBuildOverridesInput( unsigned int i, const OptixBuildInput& bi, unsigned short motionNumKeys )
{
    m_numSbtRecordsInGASForValidation++;  // numSbtRecords == 1
    if( m_numSbtRecordsInGASForValidation > m_maxSbtRecordsPerGAS )
    {
        // note m_vecBuildInputOverrides.size() == numBuildInputs
        return logIlwalidValue( corelib::stringf(
            R"msg(Sum of "numSbtRecords" (%llu) over first %u build inputs out of %u build inputs exceeds device context property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS (%u).)msg",
            m_numSbtRecordsInGASForValidation, i + 1, (uint32_t)m_vecBuildInputOverrides.size(), m_maxSbtRecordsPerGAS ) );
    }

    if( bi.lwrveArray.flag & ~( OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL ) )
    {
        return logIlwalidValue( corelib::stringf( R"msg(Invalid value (%u) for "buildInputs[%u].lwrveArray.flag")msg",
                                                  bi.lwrveArray.flag, i ) );
    }

    m_numPrimsInGASForValidation += bi.lwrveArray.numPrimitives;
    if( m_numPrimsInGASForValidation * motionNumKeys > m_maxPrimsPerGAS )
    {
        const char* const segmentMotionString =
            motionNumKeys > 1 ? ", which equals the number of lwrve segments times the number of motion keys," : "";
        if( m_vecBuildInputOverrides.size() > 1 )  // m_vecBuildInputOverrides.size() == numBuildInputs
        {
            return logIlwalidValue( corelib::stringf( R"msg(Sum of number of primitives (%llu)%s over first %u build inputs out of %u build inputs exceeds device context property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS (%u).)msg",
                                                      m_numPrimsInGASForValidation * motionNumKeys, segmentMotionString,
                                                      i + 1, (uint32_t)m_vecBuildInputOverrides.size(), m_maxPrimsPerGAS ) );
        }
        else
        {
            return logIlwalidValue(
                corelib::stringf( R"msg(Sum of number of primitives (%llu)%s exceeds device context property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS (%u).)msg",
                                  m_numPrimsInGASForValidation * motionNumKeys, segmentMotionString, m_maxPrimsPerGAS ) );
        }
    }

    if( (uint64_t)bi.lwrveArray.numPrimitives + bi.lwrveArray.primitiveIndexOffset > (uint64_t)std::numeric_limits<uint32_t>::max() )
    {
        return logIlwalidValue( corelib::stringf(
            R"msg("buildInputs[%u].lwrveArray": number of primitives + "primitiveIndexOffset" overflows 32 bits.)msg", i ) );
    }

    return OPTIX_SUCCESS;
}

// clang-format off
#define VALIDATE_VERTEX_STRIDE_CASE( enum_, stride_, align_ )                                                                                   \
    case enum_:                                                                                                                                 \
        if( const OptixResult result = validateFormatStride( i, bi.vertexStrideInBytes, stride_, align_, true, #stride_, #align_, #enum_ ) )    \
            return result;                                                                                                                      \
        break

#define VALIDATE_INDEX_STRIDE_CASE( enum_, stride_, align_ )                                                                                    \
    case enum_:                                                                                                                                 \
        if( const OptixResult result = validateFormatStride( i, bi.indexStrideInBytes, stride_, align_, false, #stride_, #align_, #enum_ ) )    \
            return result;                                                                                                                      \
// clang-format on

OptixResult RtcAccelBuilder::validateTrianglesBuildInput( const unsigned int                  i,
                                                          const OptixBuildInputTriangleArray& bi,
                                                          const unsigned short                motionNumKeys )
{
    if( bi.numVertices == 0 )
    {
        if( bi.vertexBuffers != nullptr )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].triangleArray", "numVertices" is zero, but "vertexBuffers" is not null)msg", i ) );
        if( bi.numIndexTriplets != 0 )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].triangleArray", "numVertices" is zero, but "numIndexTriplets" is not zero)msg", i ) );
        if( bi.indexBuffer != 0 )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].triangleArray", "numVertices" is zero, but "indexBuffer" is not null)msg", i ) );
    }
    else
    {
        if( !m_computeMemory )
        {
            if( bi.vertexBuffers == nullptr )
                return logIlwalidValue( corelib::stringf(
                    R"msg("buildInputs[%u].triangleArray", "numVertices" is non-zero, but "vertexBuffers" is null)msg", i ) );

            for( unsigned short motionKey = 0; motionKey < motionNumKeys; ++motionKey )
            {
                if( bi.vertexBuffers[motionKey] == 0 )
                    return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].triangleArray", "numVertices" is non-zero, but "vertexBuffers[%hu]" (vertex buffer for motion key %hu) is null)msg",
                        i, motionKey, motionKey ) );
            }
        }
        if( bi.vertexFormat == OPTIX_VERTEX_FORMAT_NONE )
        {
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].triangleArray", "numVertices" is non-zero, but "vertexFormat" is OPTIX_VERTEX_FORMAT_NONE)msg",
                                                      i ) );
        }
    }

    switch( bi.vertexFormat )
    {
        VALIDATE_VERTEX_STRIDE_CASE( OPTIX_VERTEX_FORMAT_FLOAT3, 3 * sizeof( float ), 4 );
        VALIDATE_VERTEX_STRIDE_CASE( OPTIX_VERTEX_FORMAT_FLOAT2, 2 * sizeof( float ), 4 );
        VALIDATE_VERTEX_STRIDE_CASE( OPTIX_VERTEX_FORMAT_HALF3, 3 * sizeof( float ) / 2, 2 );
        VALIDATE_VERTEX_STRIDE_CASE( OPTIX_VERTEX_FORMAT_HALF2, 2 * sizeof( float ) / 2, 2 );
        VALIDATE_VERTEX_STRIDE_CASE( OPTIX_VERTEX_FORMAT_SNORM16_3, 3 * sizeof( short ), 2 );
        VALIDATE_VERTEX_STRIDE_CASE( OPTIX_VERTEX_FORMAT_SNORM16_2, 2 * sizeof( short ), 2 );
        // case (bi.numVertices != 0 && bi.vertexFormat == OPTIX_VERTEX_FORMAT_NONE) is already handled above!
        // we effectively do not care about the format, stride, alignment if bi.numVertices == 0
        VALIDATE_VERTEX_STRIDE_CASE( OPTIX_VERTEX_FORMAT_NONE, 0, 0 );
        default:
            return logIlwalidValue( corelib::stringf( R"msg(Invalid value (0x%x) for "buildInputs[%u].triangleArray.vertexFormat")msg",
                                                      static_cast<unsigned int>( bi.vertexFormat ), i ) );
    }

    if( m_abiVersion > OptixABI::ABI_22 )
    {
        if( bi.vertexBuffers )
        {
            const unsigned int naturalBaseTypeAlignment =
                ( bi.vertexFormat == OPTIX_VERTEX_FORMAT_FLOAT3 || bi.vertexFormat == OPTIX_VERTEX_FORMAT_FLOAT2 ) ? 4 : 2;
            for( unsigned short motionKey = 0; motionKey < motionNumKeys; ++motionKey )
            {
                if( bi.vertexBuffers[motionKey] % naturalBaseTypeAlignment != 0 )
                    return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].triangleArray.vertexBuffers[%hu]" (vertex buffer for motion key %hu) is not %u-byte aligned)msg",
                                                              i, motionKey, motionKey, naturalBaseTypeAlignment ) );
            }
        }
    }

    if( bi.numIndexTriplets == 0 )
    {
        if( bi.indexBuffer != 0 )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].triangleArray", "numIndexTriplets" is zero, but "indexBuffer" is non-null)msg", i ) );
    }
    else
    {
        if( !m_computeMemory && bi.indexBuffer == 0 )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].triangleArray", "numIndexTriplets" is non-zero, but "indexBuffer" is null)msg", i ) );

        if( bi.indexFormat == OPTIX_INDICES_FORMAT_NONE )
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].triangleArray", "numIndexTriplets" is non-zero, but "indexFormat" is OPTIX_INDICES_FORMAT_NONE)msg",
                                                      i ) );
    }

    switch( bi.indexFormat )
    {
        VALIDATE_INDEX_STRIDE_CASE( OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3, 3 * sizeof( short ), 2 );
        if( m_abiVersion > OptixABI::ABI_22 )
        {
            if( bi.indexBuffer % 2 != 0 )
                return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].triangleArray", "indexFormat" is OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3, but "indexBuffer" is not 2-byte aligned)msg",
                    i ) );
        }
        break;
        VALIDATE_INDEX_STRIDE_CASE( OPTIX_INDICES_FORMAT_UNSIGNED_INT3, 3 * sizeof( int ), 4 );
        if( m_abiVersion > OptixABI::ABI_22 )
        {
            if( bi.indexBuffer % 4 != 0 )
                return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].triangleArray", "indexFormat" is OPTIX_INDICES_FORMAT_UNSIGNED_INT3, but "indexBuffer" is not 4-byte aligned)msg",
                    i ) );
        }
        break;
        // case (bi.numIndexTriplets != 0 && bi.indexFormat == OPTIX_INDICES_FORMAT_NONE) is already handled above!
        // we effectively do not care about the format, stride, alignment if bi.numIndexTriplets == 0
        VALIDATE_INDEX_STRIDE_CASE( OPTIX_INDICES_FORMAT_NONE, 0, 0 );
        break;
        default:
            return logIlwalidValue( corelib::stringf( R"msg(Invalid value (0x%x) for "buildInputs[%u].triangleArray.indexFormat")msg",
                static_cast<unsigned int>( bi.indexFormat ), i ) );
    }


    if( m_abiVersion <= OptixABI::ABI_22 )
    {
        if( bi.preTransform % OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT != 0 )
        {
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].triangleArray.preTransform" is not a multiple of OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT)msg",
                                                      i ) );
        }
    }
    else
    {
        if( bi.transformFormat == OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 )
        {
            if( !m_computeMemory )
            {
                if( bi.preTransform == 0 )
                    return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].triangleArray", "transformFormat" is OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12, but "preTransform" is null)msg",
                                                              i ) );
            }

            if( bi.preTransform % OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT != 0 )
            {
                return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].triangleArray", "preTransform" is not a multiple of OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT)msg",
                                                          i ) );
            }
        }
        else if( bi.transformFormat == OPTIX_TRANSFORM_FORMAT_NONE )
        {
            if( bi.preTransform != 0 )
                return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].triangleArray", "transformFormat" is OPTIX_TRANSFORM_FORMAT_NONE, but "preTransform" is not null)msg",
                                                          i ) );
        }
        else
            return logIlwalidValue( corelib::stringf( R"msg(Invalid value (0x%x) for "buildInputs[%u].triangleArray.transformFormat")msg",
                                                      static_cast<unsigned int>( bi.transformFormat ), i ) );
    }

    return OPTIX_SUCCESS;
}

#undef VALIDATE_VERTEX_STRIDE_CASE
#undef VALIDATE_INDEX_STRIDE_CASE

OptixResult RtcAccelBuilder::validateFormatStride( const unsigned int i,
                                                   const unsigned int strideInput,
                                                   const unsigned int naturalStride,
                                                   const unsigned int alignment,
                                                   const bool         vertexNotIndex,
                                                   const char* const  strideText,
                                                   const char* const  alignText,
                                                   const char* const  enumText )
{
    if( strideInput > 0 )
    {
        if( strideInput < naturalStride )
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].triangleArray.%sStrideInBytes" (%u) is smaller than %s for %s format %s.)msg",
                                                      i, vertexNotIndex ? "vertex" : "index", strideInput, strideText,
                                                      vertexNotIndex ? "vertex" : "index", enumText ) );

        if( strideInput % alignment != 0 )
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].triangleArray.%sStrideInBytes" (%u) is not a multiple of %s for %s format %s.)msg",
                                                      i, vertexNotIndex ? "vertex" : "index", strideInput, alignText,
                                                      vertexNotIndex ? "vertex" : "index", enumText ) );
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::validateLwrvesBuildInput( unsigned int i, const OptixBuildInputLwrveArray& bi, unsigned short motionNumKeys )
{
    if( bi.lwrveType != OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE &&
        bi.lwrveType != OPTIX_PRIMITIVE_TYPE_ROUND_LWBIC_BSPLINE &&
        bi.lwrveType != OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR &&
        bi.lwrveType != OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM )
    {
        return logIlwalidValue( corelib::stringf(
            R"msg("buildInputs[%u].lwrveArray", "lwrveType" is not a lwrve type)msg", i ) );
    }

    if( bi.numVertices == 0 )
    {
        if( bi.vertexBuffers != nullptr )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].lwrveArray", "numVertices" is zero, but "vertexBuffers" is not null)msg", i ) );
        if( bi.vertexStrideInBytes != 0 )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].lwrveArray", "numVertices" is zero, but "vertexStrideInBytes" is not zero)msg", i ) );
        if( bi.widthBuffers != nullptr )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].lwrveArray", "numVertices" is zero, but "widthBuffers" is not null)msg", i ) );
        if( bi.widthStrideInBytes != 0 )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].lwrveArray", "numVertices" is zero, but "widthStrideInBytes" is not zero)msg", i ) );
        if( bi.normalBuffers != nullptr )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].lwrveArray", "numVertices" is zero, but "normalBuffers" is not null)msg", i ) );
        if( bi.normalStrideInBytes != 0 )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].lwrveArray", "numVertices" is zero, but "normalStrideInBytes" is not zero)msg", i ) );
        if( bi.indexBuffer != 0 )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].lwrveArray", "numVertices" is zero, but "indexBuffer" is not null)msg", i ) );
        if( bi.indexStrideInBytes != 0 )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].lwrveArray", "numVertices" is zero, but "indexStrideInBytes" is not zero)msg", i ) );
        return OPTIX_SUCCESS;
    }

    if( bi.vertexStrideInBytes > 0 && bi.vertexStrideInBytes <  3 * sizeof( float ) )
        return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwrveArray.vertexStrideInBytes" (%u) is smaller than 3 * sizeof( float )"
                                                        " for vertex format OPTIX_VERTEX_FORMAT_FLOAT3.)msg",
                                                  i, bi.vertexStrideInBytes ) );
    if( bi.vertexStrideInBytes % 4 != 0 )
        return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwrveArray.vertexStrideInBytes" (%u) is not a multiple of 4"
                                                        " for vertex format OPTIX_VERTEX_FORMAT_FLOAT3.)msg",
                                                  i, bi.vertexStrideInBytes ) );

    if( bi.widthStrideInBytes > 0 && bi.widthStrideInBytes < sizeof( float ) )
        return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwrveArray.widthStrideInBytes" (%u) is smaller than sizeof( float )"
                                                        " for width format float".)msg",
                                                  i, bi.widthStrideInBytes ) );
    if( bi.widthStrideInBytes % 4 != 0 )
        return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwrveArray.widthStrideInBytes" (%u) is not a multiple of 4"
                                                        " for width format float.)msg",
                                                  i, bi.widthStrideInBytes ) );

    if( bi.normalStrideInBytes > 0 && bi.normalStrideInBytes < 3 * sizeof( float ) )
        return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwrveArray.normalStrideInBytes" (%u) is smaller than 3 * sizeof( float )"
                                                        " for normal format float3.)msg",
                                                  i, bi.normalStrideInBytes ) );
    if( bi.normalStrideInBytes % 4 != 0 )
        return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwrveArray.normalStrideInBytes" (%u) is not a multiple of 4"
                                                        " for normal format float3.)msg",
                                                  i, bi.normalStrideInBytes ) );

    if( !m_computeMemory )
    {
        if( bi.vertexBuffers == nullptr )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].lwrveArray", "numVertices" is non-zero, but "vertexBuffers" is null)msg", i ) );

        for( unsigned short motionKey = 0; motionKey < motionNumKeys; ++motionKey )
        {
            if( bi.vertexBuffers[motionKey] == 0 )
                return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwrveArray", "numVertices" is non-zero, but "vertexBuffers[%hu]" (vertex buffer for motion key %hu) is null)msg",
                    i, motionKey, motionKey ) );
        }
    }

    if( bi.numPrimitives != 0 )
    {
        if( !m_computeMemory && bi.indexBuffer == 0 )
        {
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].lwrveArray", "numPrimitives" is non-zero, but "indexBuffer" is null)msg", i ) );
        }

        if( bi.indexStrideInBytes > 0 )
        {
            if( bi.indexStrideInBytes < sizeof( int ) )
                return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwrveArray.indexStrideInBytes" (%u) is smaller than sizeof( int ) for index format unsigned int.)msg",
                    i, bi.indexStrideInBytes ) );

            if( bi.indexStrideInBytes % 4 != 0 )
                return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwrveArray.indexStrideInBytes" (%u) is not a multiple of 4 for index format unsigned int.)msg",
                    i, bi.indexStrideInBytes ) );
        }
    }

    for( unsigned short motionKey = 0; motionKey < motionNumKeys; ++motionKey )
    {
        if( bi.vertexBuffers != nullptr && bi.vertexBuffers[motionKey] % 4 != 0 )
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwrveArray.vertexBuffers[%hu]" must be a multiple of 4)msg",
                i, motionKey ) );
        if( bi.widthBuffers != nullptr && bi.widthBuffers[motionKey] % 4 != 0 )
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwrveArray.widthBuffers[%hu]" must be a multiple of 4)msg",
                i, motionKey ) );
        if( bi.normalBuffers != nullptr && bi.normalBuffers[motionKey] % 4 != 0 )
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwrveArray.normalBuffers[%hu]" must be a multiple of 4)msg",
                i, motionKey ) );
    }

    if( bi.indexBuffer % 4 != 0 )
        return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwrveArray.indexBuffer" must be a multiple of 4)msg", i ) );

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::validateSpheresBuildInput( unsigned int i, const OptixBuildInputSphereArray& bi, unsigned short motionNumKeys )
{
    if( bi.numVertices == 0 )
    {
        if( bi.vertexBuffers != nullptr )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].sphereArray", "numVertices" is zero, but "vertexBuffers" is not null)msg", i ) );
        if( bi.vertexStrideInBytes != 0 )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].sphereArray", "numVertices" is zero, but "vertexStrideInBytes" is not zero)msg", i ) );
        if( bi.radiusBuffers != nullptr )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].sphereArray", "numVertices" is zero, but "radiusBuffers" is not null)msg", i ) );
        if( bi.radiusStrideInBytes != 0 )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].sphereArray", "numVertices" is zero, but "radiusStrideInBytes" is not zero)msg", i ) );
        return OPTIX_SUCCESS;
    }

    if( bi.vertexStrideInBytes > 0 && bi.vertexStrideInBytes <  3 * sizeof( float ) )
        return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].sphereArray.vertexStrideInBytes" (%u) is smaller than 3 * sizeof( float )"
                                                        " for vertex format OPTIX_VERTEX_FORMAT_FLOAT3.)msg",
                                                  i, bi.vertexStrideInBytes ) );
    if( bi.vertexStrideInBytes % 4 != 0 )
        return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].sphereArray.vertexStrideInBytes" (%u) is not a multiple of 4"
                                                        " for vertex format OPTIX_VERTEX_FORMAT_FLOAT3.)msg",
                                                  i, bi.vertexStrideInBytes ) );

    if( bi.radiusStrideInBytes > 0 && bi.radiusStrideInBytes < sizeof( float ) )
        return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].sphereArray.radiusStrideInBytes" (%u) is smaller than sizeof( float )"
                                                        " for radius format float".)msg",
                                                  i, bi.radiusStrideInBytes ) );
    if( bi.radiusStrideInBytes % 4 != 0 )
        return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].sphereArray.radiusStrideInBytes" (%u) is not a multiple of 4"
                                                        " for radius format float.)msg",
                                                  i, bi.radiusStrideInBytes ) );

    if( !m_computeMemory )
    {
        if( bi.vertexBuffers == nullptr )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].sphereArray", "numVertices" is non-zero, but "vertexBuffers" is null)msg", i ) );

        for( unsigned short motionKey = 0; motionKey < motionNumKeys; ++motionKey )
        {
            if( bi.vertexBuffers[motionKey] == 0 )
                return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].sphereArray", "numVertices" is non-zero, but "vertexBuffers[%hu]" (vertex buffer for motion key %hu) is null)msg",
                    i, motionKey, motionKey ) );
        }
    }

    for( unsigned short motionKey = 0; motionKey < motionNumKeys; ++motionKey )
    {
        if( bi.vertexBuffers != nullptr && bi.vertexBuffers[motionKey] % 4 != 0 )
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].sphereArray.vertexBuffers[%hu]" must be a multiple of 4)msg",
                i, motionKey ) );
        if( bi.radiusBuffers != nullptr && bi.radiusBuffers[motionKey] % 4 != 0 )
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].sphereArray.radiusBuffers[%hu]" must be a multiple of 4)msg",
                i, motionKey ) );
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::validateLwstomPrimitivesBuildInput( unsigned int                               i,
                                                                 const OptixBuildInputLwstomPrimitiveArray& bi,
                                                                 unsigned short motionNumKeys )
{
    if( bi.numPrimitives == 0 )
    {
        if( bi.aabbBuffers != nullptr )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].lwstomPrimitiveArray", "numPrimitives" is zero, but "aabbBuffers" is non-null)msg", i ) );
    }

    if( bi.aabbBuffers )
    {
        for( unsigned short motionKey = 0; motionKey < motionNumKeys; ++motionKey )
        {
            if( !m_computeMemory && bi.aabbBuffers[motionKey] == 0 )
                return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwstomPrimitiveArray", "numPrimitives" is non-zero, but "aabbBuffers[%u]" is null)msg",
                                                            i, motionKey ) );
            if( bi.aabbBuffers[motionKey] % OPTIX_AABB_BUFFER_BYTE_ALIGNMENT != 0 )
                return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwstomPrimitiveArray.aabbBuffers[%u]" must be a multiple of OPTIX_AABB_BUFFER_BYTE_ALIGNMENT)msg",
                                                            i, motionKey ) );
        }
    }
    else if( !m_computeMemory && bi.numPrimitives != 0 )
    {
        return logIlwalidValue( corelib::stringf(
            R"msg("buildInputs[%u].lwstomPrimitiveArray", "numPrimitives" is non-zero, but "aabbBuffers" is null)msg", i ) );
    }

    if( bi.strideInBytes > 0 )
    {
        if( bi.strideInBytes < sizeof( OptixAabb ) )
        {
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwstomPrimitiveArray.strideInBytes" (%u) is smaller than sizeof( OptixAabb ).)msg",
                                                      i, bi.strideInBytes ) );
        }

        if( bi.strideInBytes % OPTIX_AABB_BUFFER_BYTE_ALIGNMENT != 0 )
        {
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].lwstomPrimitiveArray.strideInBytes" (%u) must be a multiple of OPTIX_AABB_BUFFER_BYTE_ALIGNMENT.)msg",
                                                      i, bi.strideInBytes ) );
        }
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::validateInstancesBuildInput( const unsigned int                  i,
                                                          const OptixBuildInputType           type,
                                                          const OptixBuildInputInstanceArray& bi,
                                                          const bool                          hasMotion,
                                                          const unsigned short                motionNumKeys )
{
    if( i > 0 )
        return logIlwalidValue( R"msg("numBuildInputs" must be 1 for instance acceleration builds)msg" );

    if( bi.instances && bi.numInstances == 0 )
    {
        return logIlwalidValue( corelib::stringf(
            R"msg("buildInputs[%u].instanceArray", "numInstances" is zero, but "instances" is non-null)msg", i ) );
    }

    if( !m_computeMemory )
    {
        if( !bi.instances && bi.numInstances != 0 )
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].instanceArray", "numInstances" is non-zero, but "instances" is null)msg", i ) );
    }

    if( bi.numInstances > m_maxInstancesPerIAS )
        return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].instanceArray.numInstances" (%u) exceeds device context property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS (%u))msg",
                                                  i, bi.numInstances, m_maxInstancesPerIAS ) );

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    if( ( !m_computeMemory ) && ( m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_IAS_USES_VM_REPLACEMENTS ) != 0 )
    {
        if( !bi.vmArrayReplacements )
        {
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].instanceArray", build flag OPTIX_BUILD_FLAG_IAS_USES_VM_REPLACEMENTS is set, but "vmArrayReplacements" is null)msg", i ) );
        }
    }
    if( ( m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_IAS_USES_VM_REPLACEMENTS ) == 0 )
    {
        if( bi.vmArrayReplacements )
        {
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].instanceArray", "vmArrayReplacements" is non-null, but build flag OPTIX_BUILD_FLAG_IAS_USES_VM_REPLACEMENTS is not set)msg", i ) );
        }
    }
#endif // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

    // The following checks are ok, even if we do NOT want to validate the devicepointer
    if( type == OPTIX_BUILD_INPUT_TYPE_INSTANCES )
    {
        if( bi.instances % OPTIX_INSTANCE_BYTE_ALIGNMENT != 0 )
        {
            return logIlwalidValue( corelib::stringf(
                R"msg("buildInputs[%u].instanceArray.instances" is not a multiple of OPTIX_INSTANCE_BYTE_ALIGNMENT)msg", i ) );
        }
    }
    else
    {
        // instanceArray.instances points to
        // arrays of pointers that need to have proper alignment (OPTIX_INSTANCE_BYTE_ALIGNMENT)
        // However, we cannot check those devices pointers here.

        // What we can do is validate that the device pointers to the arrays are properly aligned.
        if( bi.instances % sizeof( RtcGpuVA ) != 0 )
        {
            return logIlwalidValue( corelib::stringf( R"msg("buildInputs[%u].instanceArray.instances" is not a multiple of %zu)msg",
                                                      i, sizeof( RtcGpuVA ) ) );
        }
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::colwertTrianglesBuildInputs( const OptixBuildInput* buildInputs,
                                                          unsigned int           numBuildInputs,
                                                          const bool             hasMotion,
                                                          const unsigned short   motionNumKeys )
{
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    unsigned int vmCount = 0;
    unsigned int dmmCount = 0;
    for( unsigned int i = 0; i < numBuildInputs; ++i )
    {
        if( buildInputs[i].triangleArray.visibilityMap )
            vmCount++;
        if( buildInputs[i].triangleArray.displacement )
            dmmCount++;
    }
    m_vecBuildInputVm.resize( vmCount );
    m_vecBuildInputDmm.resize( dmmCount );
#endif // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

    unsigned int rtcVmArrayIndex  = 0;
    unsigned int rtcDmmArrayIndex = 0;
    for( unsigned int i = 0; i < numBuildInputs; ++i )
    {
        const OptixBuildInputTriangleArray& bi = buildInputs[i].triangleArray;
        RtcBuildInput   rtcBuildInput = {};
        RtcVertexFormat rtcVertexFormat;
        unsigned int    rtcVertexStride = bi.vertexStrideInBytes;
        switch( bi.vertexFormat )
        {
        case OPTIX_VERTEX_FORMAT_NONE:
            rtcVertexFormat = RTC_VERTEX_FORMAT_FLOAT3;
            rtcVertexStride = 0;
            break;
        case OPTIX_VERTEX_FORMAT_FLOAT3:
            rtcVertexFormat = RTC_VERTEX_FORMAT_FLOAT3;
            if( rtcVertexStride == 0 )
                rtcVertexStride = 3 * sizeof( float );
            break;
        case OPTIX_VERTEX_FORMAT_FLOAT2:
            rtcVertexFormat = RTC_VERTEX_FORMAT_FLOAT2;
            if( rtcVertexStride == 0 )
                rtcVertexStride = 2 * sizeof( float );
            break;
        case OPTIX_VERTEX_FORMAT_HALF3:
            rtcVertexFormat = RTC_VERTEX_FORMAT_HALF3;
            if( rtcVertexStride == 0 )
                rtcVertexStride = 3 * sizeof( float ) / 2;
            break;
        case OPTIX_VERTEX_FORMAT_HALF2:
            rtcVertexFormat = RTC_VERTEX_FORMAT_HALF2;
            if( rtcVertexStride == 0 )
                rtcVertexStride = 2 * sizeof( float ) / 2;
            break;
        case OPTIX_VERTEX_FORMAT_SNORM16_3:
            rtcVertexFormat = RTC_VERTEX_FORMAT_SNORM16_3;
            if( rtcVertexStride == 0 )
                rtcVertexStride = 3 * sizeof( short );
            break;
        case OPTIX_VERTEX_FORMAT_SNORM16_2:
            rtcVertexFormat = RTC_VERTEX_FORMAT_SNORM16_2;
            if( rtcVertexStride == 0 )
                rtcVertexStride = 2 * sizeof( short );
            break;
        }

        unsigned int rtcIndexSizeInBytes = 0;
        unsigned int rtcIndexStride      = bi.indexStrideInBytes;
        switch( bi.indexFormat )
        {
        case OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3:
            rtcIndexSizeInBytes = sizeof( short );
            break;
        case OPTIX_INDICES_FORMAT_UNSIGNED_INT3:
            rtcIndexSizeInBytes = sizeof( int );
            break;
        case OPTIX_INDICES_FORMAT_NONE:
            rtcIndexSizeInBytes = 0;
            rtcIndexStride      = 0;
            break;
        }
        if( rtcIndexStride == 0 )
            rtcIndexStride = 3 * rtcIndexSizeInBytes;

        if( hasMotion )
        {
            rtcBuildInput.type = RTC_BUILD_INPUT_TYPE_MOTION_TRIANGLE_ARRAYS;
            // although the type is RtcGpuVA, rtcore expects a host pointer in the motion case (pointing to an array of device pointers)!
            rtcBuildInput.triangleArray.vertexBuffer = reinterpret_cast<RtcGpuVA>( bi.vertexBuffers );
        }
        else
        {
            rtcBuildInput.type = RTC_BUILD_INPUT_TYPE_TRIANGLE_ARRAY;
            // according to the test above, we accept 0 numVertices as well as 0 as vertexBuffer (the test above checks for agreement between the two)
            rtcBuildInput.triangleArray.vertexBuffer = bi.vertexBuffers ? static_cast<RtcGpuVA>( bi.vertexBuffers[0] ) : 0;
        }

        rtcBuildInput.triangleArray.vertexFormat        = rtcVertexFormat;
        rtcBuildInput.triangleArray.numVertices         = bi.numVertices;
        rtcBuildInput.triangleArray.vertexStrideInBytes = rtcVertexStride;
        rtcBuildInput.triangleArray.indexBuffer         = bi.indexBuffer;
        rtcBuildInput.triangleArray.numIndices          = bi.numIndexTriplets * 3;
        rtcBuildInput.triangleArray.indexSizeInBytes    = rtcIndexSizeInBytes;
        rtcBuildInput.triangleArray.indexStrideInBytes  = rtcIndexStride;
        rtcBuildInput.triangleArray.flags               = bi.flags[0];

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
        // type equality is checked in RtcVmBuilder.cpp and RtcDmmBuilder.cpp
        // note that we also rely on rtcore behaving like optix, e.g., if indexStrideInBytes is 0, the stride is assumed to be indexSizeInBytes
        // Otherwise we would need to maintain an array of RtcBuildInputVisibilityMap, copying the optix values to rtc values.
        if( bi.visibilityMap )
        {
            RtcBuildInputVisibilityMap& vm = m_vecBuildInputVm[rtcVmArrayIndex++];
            vm = {};
            reinterpret_cast<OptixBuildInputVisibilityMap&>( vm ) = *bi.visibilityMap;
            rtcBuildInput.triangleArray.visibilityMap = &vm;
        }
        if( bi.displacement )
        {
            RtcBuildInputDisplacement& dmm = m_vecBuildInputDmm[rtcDmmArrayIndex++];
            dmm = {};
            reinterpret_cast<OptixBuildInputDisplacement&>( dmm ) = *bi.displacement;
            switch( bi.displacement->displacementVectorFormat )
            {
            case OPTIX_MICROMESH_DISPLACEMENT_VECTOR_FORMAT_HALF3:
                dmm.dispVectorFormat = RTC_DISPLACEMENT_VECTOR_FORMAT_FORMAT_HALF3;
                if( dmm.dispVectorStrideInBytes == 0 )
                    dmm.dispVectorStrideInBytes = 3 * sizeof( float ) / 2;
                break;
            }
            dmm.dispVectorStrideInBytes =
                ( bi.displacement->displacementVectorStrideInBytes == 0 ) ? 3 * sizeof( float ) / 2 : bi.displacement->displacementVectorStrideInBytes;
            rtcBuildInput.triangleArray.displacement = &dmm;
        }
#endif // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

        if( m_abiVersion <= OptixABI::ABI_22 )
        {
            // We did not have a way of distinguishing null device ptr for compute memory operation as
            // "not wanted" vs. "not yet set".
            // So always assume that a preTransform will be used for the build.
            // The assumption for that is that the memory requirement for w/ preTransform are strictly greater than w/o preTransform.
            if( m_computeMemory )
                rtcBuildInput.triangleArray.transform = (RtcGpuVA)OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT;
            else
                rtcBuildInput.triangleArray.transform = bi.preTransform;
        }
        else
        {
            if( bi.transformFormat == OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 )
            {
                if( bi.preTransform )
                    rtcBuildInput.triangleArray.transform = bi.preTransform;
                else
                    // use a dummy pointer
                    // validation must make sure that this case can only be reached in case of accel memory compute
                    rtcBuildInput.triangleArray.transform = (RtcGpuVA)OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT;
            }
        }

        RtcBuildInputOverrides& rtcBuildInputOverrides = m_vecBuildInputOverrides[i];

        rtcBuildInputOverrides.primitiveIndexBias = bi.primitiveIndexOffset;
        rtcBuildInputOverrides.numGeometries      = bi.numSbtRecords;
        if( bi.numSbtRecords > 1 )
        {
            rtcBuildInputOverrides.geometryOffsetBuffer      = bi.sbtIndexOffsetBuffer;
            rtcBuildInputOverrides.geometryOffsetSizeInBytes = bi.sbtIndexOffsetSizeInBytes;
            rtcBuildInputOverrides.geometryOffsetStrideInBytes =
                bi.sbtIndexOffsetStrideInBytes ? bi.sbtIndexOffsetStrideInBytes : bi.sbtIndexOffsetSizeInBytes;
        }

        m_vecBuildInputs.push_back( rtcBuildInput );

        if( rtcBuildInputOverrides.primitiveIndexBias == 0 && rtcBuildInputOverrides.numGeometries == 1 )
        {
            // vector will be filled lazily, see below
            if( !m_vecBuildInputOverridePtrs.empty() )
                m_vecBuildInputOverridePtrs.push_back( nullptr );
        }
        else
        {
            if( m_vecBuildInputOverridePtrs.empty() )
            {
                // lazily allocate m_vecBuildInputOverridePtrs since we may not need it at all
                // we know right now, that we need to store at least that many ptrs
                m_vecBuildInputOverridePtrs.reserve( m_vecBuildInputOverrides.size() + bi.numSbtRecords - 1 );
                // will do push backs in the future as we do not yet know how many pointers we need to store
                // resize to the size which we would have if we did push_backs until 'now'
                // later additions of pointers to the vector will use a push_back
                m_vecBuildInputOverridePtrs.resize( i, nullptr );
            }
            m_vecBuildInputOverridePtrs.push_back( &m_vecBuildInputOverrides[i] );
        }

        rtcBuildInput.triangleArray.vertexBuffer = reinterpret_cast<RtcGpuVA>( nullptr );
        rtcBuildInput.triangleArray.numVertices  = 0;
        rtcBuildInput.triangleArray.numIndices   = 0;

        for( unsigned int j = 1; j < bi.numSbtRecords; ++j )
        {
            rtcBuildInput.triangleArray.flags = bi.flags[j];

            m_vecBuildInputs.push_back( rtcBuildInput );
            m_vecBuildInputOverridePtrs.push_back( nullptr );
        }
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::colwertLwrvesBuildInput( const unsigned int                  i,
                                                      const OptixBuildInputLwrveArray&    bi,
                                                      const bool                          hasMotion,
                                                      const unsigned short                motionNumKeys )
{
    RtcBuildInput rtcBuildInput = {};
    rtcBuildInput.type = hasMotion ? RTC_BUILD_INPUT_TYPE_MOTION_AABB_ARRAYS : RTC_BUILD_INPUT_TYPE_AABB_ARRAY;
    // rtcBuildInput.aabbArray.aabbBuffer will be set later in computeLwrveAabbs()
    rtcBuildInput.aabbArray.numAabbs  = static_cast<unsigned int>( m_numSplits * bi.numPrimitives );
    rtcBuildInput.aabbArray.strideInBytes = getLwrveAabbSize();
    rtcBuildInput.aabbArray.flags         = bi.flag;
    rtcBuildInput.aabbArray.numSubBoxes   = k_lwrveLSS.get();

    RtcBuildInputOverrides& rtcBuildInputOverrides = m_vecBuildInputOverrides[i];

    rtcBuildInputOverrides.numGeometries = 1;
    rtcBuildInputOverrides.primitiveIndexBias          = bi.primitiveIndexOffset;
    rtcBuildInputOverrides.primLwstomVABuffer          = reinterpret_cast< RtcGpuVA >( nullptr );
    rtcBuildInputOverrides.primLwstomVAStrideInBytes   = m_builtinISLowMem ? 0 : sizeof( LwrveSegmentData );
    rtcBuildInputOverrides.primitiveIndexBuffer        = reinterpret_cast< RtcGpuVA >( nullptr );
    rtcBuildInputOverrides.primitiveIndexStrideInBytes = sizeof( unsigned int );

    m_numRtcLwrvesInGAS += rtcBuildInput.aabbArray.numAabbs;

    m_vecBuildInputs.push_back( rtcBuildInput );

    if( rtcBuildInputOverrides.primitiveIndexBias == 0 && 
        rtcBuildInputOverrides.primLwstomVAStrideInBytes == 0 &&
        rtcBuildInputOverrides.primitiveIndexStrideInBytes == 0 )
    {
        // vector will be filled lazily, see below
        if( !m_vecBuildInputOverridePtrs.empty() )
            m_vecBuildInputOverridePtrs.push_back( nullptr );
    }
    else
    {
        if( m_vecBuildInputOverridePtrs.empty() )
        {
            // lazily allocate m_vecBuildInputOverridePtrs since we may not need it at all
            // we know right now, that we need to store at least that many ptrs
            m_vecBuildInputOverridePtrs.reserve( m_vecBuildInputOverrides.size() );
            // will do push backs in the future as we do not yet know how many pointers we need to store
            // resize here to shrink to size as if we did push_backs up to now
            m_vecBuildInputOverridePtrs.resize( i, nullptr );
        }
        m_vecBuildInputOverridePtrs.push_back( &m_vecBuildInputOverrides[i] );
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::colwertSpheresBuildInput( const unsigned int                  i,
                                                       const OptixBuildInputSphereArray&   bi,
                                                       const bool                          hasMotion,
                                                       const unsigned short                motionNumKeys )
{
    RtcBuildInput rtcBuildInput = {};
    rtcBuildInput.type = hasMotion ? RTC_BUILD_INPUT_TYPE_MOTION_AABB_ARRAYS : RTC_BUILD_INPUT_TYPE_AABB_ARRAY;
    // rtcBuildInput.aabbArray.aabbBuffer will be set later in computeSphereAabbs()
    rtcBuildInput.aabbArray.numAabbs      = bi.numVertices;
    rtcBuildInput.aabbArray.strideInBytes = sizeof( OptixAabb );
    rtcBuildInput.aabbArray.flags         = *bi.flags;
    rtcBuildInput.aabbArray.numSubBoxes   = 0;

    RtcBuildInputOverrides& rtcBuildInputOverrides = m_vecBuildInputOverrides[i];

    rtcBuildInputOverrides.numGeometries      = bi.numSbtRecords;
    rtcBuildInputOverrides.primitiveIndexBias = bi.primitiveIndexOffset;

    if( bi.numSbtRecords > 1 )
    {
        rtcBuildInputOverrides.geometryOffsetBuffer      = bi.sbtIndexOffsetBuffer;
        rtcBuildInputOverrides.geometryOffsetSizeInBytes = bi.sbtIndexOffsetSizeInBytes;
        rtcBuildInputOverrides.geometryOffsetStrideInBytes =
            bi.sbtIndexOffsetStrideInBytes ? bi.sbtIndexOffsetStrideInBytes : bi.sbtIndexOffsetSizeInBytes;
    }

    rtcBuildInputOverrides.primLwstomVABuffer          = reinterpret_cast< RtcGpuVA >( nullptr );
    rtcBuildInputOverrides.primLwstomVAStrideInBytes   = m_builtinISLowMem ? 0 : sizeof( SphereIntersectorData );
    rtcBuildInputOverrides.primitiveIndexBuffer        = reinterpret_cast< RtcGpuVA >( nullptr );
    rtcBuildInputOverrides.primitiveIndexStrideInBytes = 0;// sizeof( unsigned int );

    m_vecBuildInputs.push_back( rtcBuildInput );

    if( rtcBuildInputOverrides.primitiveIndexBias == 0 && 
        rtcBuildInputOverrides.numGeometries == 1 && 
        rtcBuildInputOverrides.primLwstomVAStrideInBytes == 0 &&
        rtcBuildInputOverrides.primitiveIndexStrideInBytes == 0 )
    {
        // vector will be filled lazily, see below
        if( !m_vecBuildInputOverridePtrs.empty() )
            m_vecBuildInputOverridePtrs.push_back( nullptr );
    }
    else
    {
        if( m_vecBuildInputOverridePtrs.empty() )
        {
            // lazily allocate m_vecBuildInputOverridePtrs since we may not need it at all
            // we know right now, that we need to store at least that many ptrs
            m_vecBuildInputOverridePtrs.resize( m_vecBuildInputOverrides.size() + bi.numSbtRecords - 1, nullptr );
            // will do push backs in the future as we do not yet know how many pointers we need to store
            // resize here to shrink to size as if we did push_backs up to now
            m_vecBuildInputOverridePtrs.resize( i );
        }
        m_vecBuildInputOverridePtrs.push_back( &m_vecBuildInputOverrides[i] );
    }

    rtcBuildInput.aabbArray.aabbBuffer = reinterpret_cast<RtcGpuVA>( nullptr );
    rtcBuildInput.aabbArray.numAabbs   = 0;

    for( unsigned int j = 1; j < bi.numSbtRecords; ++j )
    {
        rtcBuildInput.aabbArray.flags       = bi.flags[j];
        rtcBuildInput.aabbArray.numSubBoxes = 0;

        m_vecBuildInputs.push_back( rtcBuildInput );
        m_vecBuildInputOverridePtrs.push_back( nullptr );
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::colwertLwstomPrimitivesBuildInput( const unsigned int                         i,
                                                                const OptixBuildInputLwstomPrimitiveArray& bi,
                                                                const bool                                 hasMotion )
{
    RtcBuildInput rtcBuildInput = {};
    
    if( hasMotion )
    {
        rtcBuildInput.type = RTC_BUILD_INPUT_TYPE_MOTION_AABB_ARRAYS;
        if( bi.numPrimitives > 0 )
            rtcBuildInput.aabbArray.aabbBuffer = reinterpret_cast< RtcGpuVA >( bi.aabbBuffers );
    }
    else
    {
        rtcBuildInput.type = RTC_BUILD_INPUT_TYPE_AABB_ARRAY;
        if( bi.numPrimitives > 0 )
            rtcBuildInput.aabbArray.aabbBuffer = bi.aabbBuffers ? static_cast< RtcGpuVA >( bi.aabbBuffers[0] ) : 0;
    }

    rtcBuildInput.aabbArray.numAabbs      = bi.numPrimitives;
    rtcBuildInput.aabbArray.strideInBytes = bi.strideInBytes ? bi.strideInBytes : sizeof( OptixAabb );
    rtcBuildInput.aabbArray.flags         = *bi.flags;
    rtcBuildInput.aabbArray.numSubBoxes   = 0;

    RtcBuildInputOverrides& rtcBuildInputOverrides = m_vecBuildInputOverrides[i];

    rtcBuildInputOverrides.primitiveIndexBias = bi.primitiveIndexOffset;
    rtcBuildInputOverrides.numGeometries      = bi.numSbtRecords;
    if( bi.numSbtRecords > 1 )
    {
        rtcBuildInputOverrides.geometryOffsetBuffer      = bi.sbtIndexOffsetBuffer;
        rtcBuildInputOverrides.geometryOffsetSizeInBytes = bi.sbtIndexOffsetSizeInBytes;
        rtcBuildInputOverrides.geometryOffsetStrideInBytes =
            bi.sbtIndexOffsetStrideInBytes ? bi.sbtIndexOffsetStrideInBytes : bi.sbtIndexOffsetSizeInBytes;
    }

    m_vecBuildInputs.push_back( rtcBuildInput );

    if( rtcBuildInputOverrides.primitiveIndexBias == 0 && rtcBuildInputOverrides.numGeometries == 1 )
    {
        // vector will be filled lazily, see below
        if( !m_vecBuildInputOverridePtrs.empty() )
            m_vecBuildInputOverridePtrs.push_back( nullptr );
    }
    else
    {
        if( m_vecBuildInputOverridePtrs.empty() )
        {
            // lazily allocate m_vecBuildInputOverridePtrs since we may not need it at all
            // we know right now, that we need to store at least that many ptrs
            m_vecBuildInputOverridePtrs.resize( m_vecBuildInputOverrides.size() + bi.numSbtRecords - 1, nullptr );
            // will do push backs in the future as we do not yet know how many pointers we need to store
            // resize here to shrink to size as if we did push_backs up to now
            m_vecBuildInputOverridePtrs.resize( i );
        }
        m_vecBuildInputOverridePtrs.push_back( &m_vecBuildInputOverrides[i] );
    }

    rtcBuildInput.aabbArray.aabbBuffer = reinterpret_cast<RtcGpuVA>( nullptr );
    rtcBuildInput.aabbArray.numAabbs   = 0;

    for( unsigned int j = 1; j < bi.numSbtRecords; ++j )
    {
        rtcBuildInput.aabbArray.flags       = bi.flags[j];
        rtcBuildInput.aabbArray.numSubBoxes = 0;

        m_vecBuildInputs.push_back( rtcBuildInput );
        m_vecBuildInputOverridePtrs.push_back( nullptr );
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::colwertInstancesBuildInput( OptixBuildInputType type, const OptixBuildInputInstanceArray& bi ) noexcept
{
    RtcBuildInput rtcBuildInput = {};

    // ABI 20 is first archived ABI variant and uses the DXR instance interface.
    // ABI 21 uses the FAT_INSTANCE interface.
    if( m_abiVersion <= OptixABI::ABI_20 )
    {
        if( type == OPTIX_BUILD_INPUT_TYPE_INSTANCES )
        {
            rtcBuildInput.type = RTC_BUILD_INPUT_TYPE_INSTANCE_ARRAY;
        }
        else
        {
            rtcBuildInput.type = RTC_BUILD_INPUT_TYPE_INSTANCE_POINTERS;
        }
    }
    else
    {
        if( type == OPTIX_BUILD_INPUT_TYPE_INSTANCES )
        {
            rtcBuildInput.type = RTC_BUILD_INPUT_TYPE_FAT_INSTANCE_ARRAY;
        }
        else
        {
            rtcBuildInput.type = RTC_BUILD_INPUT_TYPE_FAT_INSTANCE_POINTERS;
        }
    }

    rtcBuildInput.instanceArray.instanceDescs = bi.instances;
    rtcBuildInput.instanceArray.numInstances  = bi.numInstances;

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    // ABI < 43 had members OptixBuildInputInstanceArray::
    //   LWdeviceptr aabbs;
    //   unsigned int numAabbs;
    // These members now overlap with the added members below (ABI 54)
    // We need to make sure to not read that data if an application with an old ABI is run
    if( m_abiVersion >= OptixABI::ABI_54 )
    {
        rtcBuildInput.instanceArray.instanceStride        = bi.instanceStride;
        if( m_accelOptions->buildFlags & OPTIX_BUILD_FLAG_IAS_USES_VM_REPLACEMENTS )
        {
            if( bi.vmArrayReplacements )
            {
                rtcBuildInput.instanceArray.vmArrayReplacements   = bi.vmArrayReplacements;
                rtcBuildInput.instanceArray.vmReplacementsStride  = bi.vmReplacementsStride;
            }
            else
            {
                // use a dummy pointer
                // validation must make sure that this case can only be reached in case of accel memory compute
                rtcBuildInput.instanceArray.vmArrayReplacements = (RtcGpuVA)OPTIX_INSTANCE_VM_REPLACEMENTS_BYTE_ALIGNMENT;
                rtcBuildInput.instanceArray.vmReplacementsStride  = bi.vmReplacementsStride;
            }
        }
    }
#endif // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

    m_vecBuildInputs.push_back( rtcBuildInput );

    return OPTIX_SUCCESS;
}


OptixResult RtcAccelBuilder::validateTempBuffer( LWdeviceptr tempBuffer, size_t tempBufferSizeInBytes, OptixAccelBufferSizes bufferSizes )
{
    if( !tempBuffer )
    {
        if( tempBufferSizeInBytes )
        {
            return logIlwalidValue( R"msg("tempBufferSizeInBytes" is non-zero but "tempBuffer" is null)msg" );
        }
    }
    else if( !tempBufferSizeInBytes )
    {
        return logIlwalidValue( R"msg("tempBuffer" is non-null but "tempBufferSizeInBytes" is 0)msg" );
    }

    if( m_accelOptions->operation == OPTIX_BUILD_OPERATION_BUILD )
    {
        if( tempBufferSizeInBytes < bufferSizes.tempSizeInBytes )
        {
            return logIlwalidValue( corelib::stringf( R"msg("tempBufferSizeInBytes" (%zi b) is less than what is required according to OptixAccelBufferSizes::tempSizeInBytes as output by optixAccelComputeMemoryUsage() (%zi b) )msg",
                                                      tempBufferSizeInBytes, bufferSizes.tempSizeInBytes ) );
        }
    }
    else
    {
        if( tempBufferSizeInBytes < bufferSizes.tempUpdateSizeInBytes )
        {
            return logIlwalidValue( corelib::stringf( R"msg("tempBufferSizeInBytes" (%zi b) is less than what is required according to OptixAccelBufferSizes::tempUpdateSizeInBytes as output by optixAccelComputeMemoryUsage() (%zi b) )msg",
                                                      tempBufferSizeInBytes, bufferSizes.tempUpdateSizeInBytes ) );
        }
    }

    if( tempBuffer % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT != 0 )
    {
        return logIlwalidValue( R"msg("tempBuffer" is not a multiple of OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT)msg" );
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::validateOutputBuffer( LWdeviceptr outputBuffer, size_t outputBufferSizeInBytes, OptixAccelBufferSizes bufferSizes )
{
    if( outputBuffer % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT != 0 )
    {
        return logIlwalidValue( R"msg("outputBuffer" is not a multiple of OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT)msg" );
    }

    if( outputBufferSizeInBytes < bufferSizes.outputSizeInBytes )
    {
        return logIlwalidValue( corelib::stringf( R"msg("outputBufferSizeInBytes" (%zi b) is less than what is required according to OptixAccelBufferSizes::outputSizeInBytes as output by optixAccelComputeMemoryUsage() (%zi b) )msg",
                                                  outputBufferSizeInBytes, bufferSizes.outputSizeInBytes ) );
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::splitLwrveAdaptive( LWstream stream, LWdeviceptr tempBuffer, size_t tempBufferSizeInBytes )
{
    // TODO: reduce tempBuffer memory consumption
    // (f.e., by a higher utilization of the aabb memory which during splitting is not yet used for aabbs ).

    unsigned int numPrimitives = 0;
    unsigned int numSegments   = 0;
    for (unsigned int i = 0; i < m_numBuildInputs; ++i)
    {
        numPrimitives += m_buildInputs[i].lwrveArray.numPrimitives;
        numSegments += m_vecBuildInputs[i].aabbArray.numAabbs;
    }

    LwrveAdaptiveSplitInfo splitInfo   = {numPrimitives, numSegments, 0, m_buildInputs, m_numBuildInputs, m_numSplits, 0, m_errDetails};

    tempBufferSizeInBytes -= ( tempBufferSizeInBytes % OPTIX_AABB_BUFFER_BYTE_ALIGNMENT );
    unsigned int indexOffsetsSize =
        corelib::roundUp( m_numBuildInputs * sizeof( int ), static_cast<size_t>( OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ) );
    const unsigned int numMotionSteps = m_rtcAccelOptions.motionSteps;
    // Size of memory in tempBuffer used for temporary splitting data and aabb storage.
    splitInfo.memorySize = numSegments * numMotionSteps * getLwrveAabbSize()       // aabbs
                         + numSegments * sizeof( LwrveSegmentData ) // additional memory for per primitive VA
                         + numSegments * sizeof( unsigned int )     // additional memory for per primitive remapped index
                         + 2 * numSegments * sizeof( int )          // additional memory for resulting subsegments
                         + numPrimitives * 7 * sizeof( float )      // for precomputed SA reductions
                         + corelib::roundUp( numPrimitives * sizeof( unsigned char ), static_cast<size_t>( LWRVE_ADAPTIVE_SPLITTING_ALIGNMENT ) ); // for inflection points

    // Pointer to the free memory in tempBuffer where to store splitting data.
    splitInfo.data = ( LWdeviceptr )( tempBuffer ) + tempBufferSizeInBytes - splitInfo.memorySize - indexOffsetsSize;

    // Pre-compute SA reductions of segment splits and initialize the segment buffer.

    if( OptixResult result = computeLwrveSAReductions( stream, splitInfo ) )
        return result;

    // Split segments following an approximate SA reduction order.

    if( OptixResult result = splitSegments( stream, splitInfo ) )
        return result;

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::getLwrveAabbsAndSegments( LWdeviceptr&           aabbBuffer,
                                                       LWdeviceptr&           segmentMap,
                                                       LWdeviceptr&           indexMap,
                                                       LWdeviceptr&           segmentData,
                                                       LWdeviceptr&           inflectionPoints,
                                                       LWdeviceptr            tempBuffer,
                                                       size_t                 tempBufferSizeInBytes )
{
    unsigned int numAabbs = 0;
    unsigned int numPrimitives = 0;
    for( unsigned int i = 0; i < m_numBuildInputs; ++i )
    {
        numAabbs += m_vecBuildInputs[i].aabbArray.numAabbs;
        numPrimitives += m_buildInputs[i].lwrveArray.numPrimitives;
    }
    // The aabb data of size numAabbs * numMotionSteps * sizeof( OptixAabb) and, later, the index offsets per build input of
    // size m_numBuildInputs * sizeof( int ), are stored at the end of the temp buffer.
    // The actually required rtcore temp buffer size is not yet known.

    tempBufferSizeInBytes -= (tempBufferSizeInBytes % OPTIX_AABB_BUFFER_BYTE_ALIGNMENT);
    unsigned int indexOffsetsSize =
        corelib::roundUp( m_numBuildInputs * sizeof( int ), static_cast<size_t>(OPTIX_AABB_BUFFER_BYTE_ALIGNMENT) );
    size_t aabbsSize = m_rtcAccelOptions.motionSteps * numAabbs * getLwrveAabbSize();
    aabbBuffer       = (LWdeviceptr)(tempBuffer)+tempBufferSizeInBytes - aabbsSize - indexOffsetsSize;
    if( !m_builtinISLowMem )
    {
        segmentData = aabbBuffer - numAabbs * sizeof( LwrveSegmentData );
        indexMap = segmentData - numAabbs * sizeof( unsigned int );
        if (m_lwrveAdaptiveSplitting)
        {
            segmentMap = indexMap - numAabbs * sizeof( int );
            inflectionPoints = segmentMap - corelib::roundUp( numPrimitives * sizeof( unsigned char ), static_cast<size_t>(LWRVE_ADAPTIVE_SPLITTING_ALIGNMENT) );
        }
    }
    else
    {
        indexMap = aabbBuffer - numAabbs * sizeof( unsigned int );
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::getSphereAabbs( LWdeviceptr&           aabbBuffer,
                                             LWdeviceptr&           sphereData,
                                             LWdeviceptr            tempBuffer,
                                             size_t                 tempBufferSizeInBytes )
{
    unsigned int numAabbs = 0;
    for( unsigned int i = 0; i < m_numBuildInputs; ++i )
        numAabbs += m_buildInputs[i].sphereArray.numVertices;
 
    // The aabb data of size numAabbs * numMotionSteps * sizeof( OptixAabb) and, later, the primitive offsets per build input of
    // size m_numBuildInputs * sizeof( int ),  are stored at the end of the temp buffer.
    // The actually required rtcore temp buffer size is not yet known.

    tempBufferSizeInBytes -= ( tempBufferSizeInBytes % OPTIX_AABB_BUFFER_BYTE_ALIGNMENT );
    unsigned int primitiveOffsetsSize =
        corelib::roundUp( m_numBuildInputs * sizeof( int ), static_cast<size_t>(OPTIX_AABB_BUFFER_BYTE_ALIGNMENT) );
    size_t aabbsSize = m_rtcAccelOptions.motionSteps * numAabbs * sizeof( OptixAabb );
    aabbBuffer = (LWdeviceptr)( tempBuffer )+tempBufferSizeInBytes - aabbsSize - primitiveOffsetsSize;

    if( !m_builtinISLowMem )
        sphereData = aabbBuffer - numAabbs * sizeof( SphereIntersectorData );

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::computeLwrveLSSs( LWstream     stream,
                                                LWdeviceptr tempBuffer,
                                                size_t      tempBufferSizeInBytes )
{
    // Compute AABBs and store them in temp buffer.
    const unsigned int numMotionSteps = m_rtcAccelOptions.motionSteps;

    LWdeviceptr aabbBuffer     = 0;
    LWdeviceptr segmentMap     = 0; // mapping for adaptive splitting
    LWdeviceptr segmentData    = 0; // per-segment intersection data
    LWdeviceptr indexMap       = 0;
    LWdeviceptr inflectionData = 0;
    getLwrveAabbsAndSegments( aabbBuffer, segmentMap, indexMap, segmentData, inflectionData, tempBuffer, tempBufferSizeInBytes );

    // Host storage to hold quantized LSS data, downloaded from device. Only used when dumping LSS to file.
    std::vector<char>  dumpLSSAabbs;
    size_t             dumpLSSSegmentCount = 0;

    size_t aabbStrideInBytes = getLwrveAabbSize();
    for( unsigned int i = 0; i < m_numBuildInputs; ++i )
    {
        const OptixBuildInput &bi = m_buildInputs[i];
        RtcBuildInput &rtcbi = m_vecBuildInputs[i];

        for( int motionStep = 0; motionStep < numMotionSteps; motionStep++ )
            makeLwrveLSSs( stream, bi, motionStep, (unsigned int)m_numSplits, segmentMap,
                           rtcbi.aabbArray.numAabbs, aabbBuffer, aabbStrideInBytes, k_lwrveLSS.get() );

        const unsigned int numSegments = numMotionSteps * rtcbi.aabbArray.numAabbs;
        size_t aabbSizeInBytes   = aabbStrideInBytes * numSegments;

        if( !k_dumpLSS.isDefault() )
        {
            // Download the AABB and quantized LSS to a file.
            size_t aabbOffsetInBytes = dumpLSSAabbs.size();
            dumpLSSAabbs.resize( aabbOffsetInBytes + aabbSizeInBytes );
            if( lwdaError_t err = lwdaMemcpy( dumpLSSAabbs.data() + aabbOffsetInBytes, (void*)aabbBuffer, aabbSizeInBytes, lwdaMemcpyDeviceToHost ) )
                return OPTIX_ERROR_LWDA_ERROR;
            dumpLSSSegmentCount += numSegments;
        }

        if( aabbBuffer )
            aabbBuffer += aabbSizeInBytes;
        if( segmentMap )   
            segmentMap += rtcbi.aabbArray.numAabbs * sizeof( int );
    }

    if( !k_dumpLSS.isDefault() )
    {
        // Decompress the LSS and write per motion-key hair files.
        for( size_t m = 0; m < numMotionSteps; ++m )
        {
            std::vector<float> dumpLSSPointArray;
            std::vector<float> dumpLSSThicknessArray;

            for( size_t i = 0; i < dumpLSSSegmentCount; ++i )
            {
                OptixAabb* aabb = reinterpret_cast<OptixAabb*>(dumpLSSAabbs.data() + (i * numMotionSteps + m) * aabbStrideInBytes);
                OptixLSS*  lss  = reinterpret_cast<OptixLSS*>( aabb + 1 );

                for( size_t j = 0; j < k_lwrveLSS.get(); ++j )
                {
                    float wx = (aabb->maxX - aabb->minX);
                    float wy = (aabb->maxY - aabb->minY);
                    float wz = (aabb->maxZ - aabb->minZ);
                    float wr = fmaxf(wx,fmaxf(wy,wz)) * 2.f; // radius to thickness

                    dumpLSSPointArray.push_back(aabb->minX + (lss[j].beginX / 255.f) * wx);
                    dumpLSSPointArray.push_back(aabb->minY + (lss[j].beginY / 255.f) * wy);
                    dumpLSSPointArray.push_back(aabb->minZ + (lss[j].beginZ / 255.f) * wz);
                    dumpLSSThicknessArray.push_back((lss[j].beginR / 255.f) * wr);

                    dumpLSSPointArray.push_back(aabb->minX + (lss[j].endX / 255.f) * wx);
                    dumpLSSPointArray.push_back(aabb->minY + (lss[j].endY / 255.f) * wy);
                    dumpLSSPointArray.push_back(aabb->minZ + (lss[j].endZ / 255.f) * wz);
                    dumpLSSThicknessArray.push_back((lss[j].endR / 255.f) * wr);
                }
            }

            // .hair file header (http://www.cemyuksel.com/)
            struct HairHeader
            {
                char magic[4] = {'H','A','I','R'};;
                uint32_t numStrands = 0;
                uint32_t numPoints = 0;
                uint32_t flags = 6;
                uint32_t defaultNumSegments = 1;
                float defaultThickness = 1.f;
                float defaultTransparency = 1.f;
                float3 defaultColor = {1.f,1.f,1.f};
                char fileInfo[88];
            };

            HairHeader header;
            header.numStrands = dumpLSSSegmentCount * k_lwrveLSS.get();
            header.numPoints  = header.numStrands * 2;

            std::string filename = k_dumpLSS.get() + "_" + std::to_string(m) + ".hair";
            FILE *fp = fopen(filename.c_str(), "wb");
            fwrite(&header,sizeof(HairHeader),1,fp);
            fwrite(dumpLSSPointArray.data(),sizeof(float),dumpLSSPointArray.size(),fp);
            fwrite(dumpLSSThicknessArray.data(),sizeof(float),dumpLSSThicknessArray.size(),fp);
            fclose(fp);
        }
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::computeLwrveAabbs( LWstream               stream,
                                                LWdeviceptr            tempBuffer,
                                                size_t                 tempBufferSizeInBytes,
                                                std::vector<std::vector<LWdeviceptr>>& lwrveAabbs )
{
    // Compute AABBs and store them in temp buffer.

    const unsigned int numMotionSteps = m_rtcAccelOptions.motionSteps;

    LWdeviceptr aabbBuffer     = 0;
    LWdeviceptr segmentMap     = 0; // mapping for adaptive splitting
    LWdeviceptr segmentData    = 0; // per-segment intersection data
    LWdeviceptr indexMap       = 0;
    LWdeviceptr inflectionPoints = 0;
    getLwrveAabbsAndSegments( aabbBuffer, segmentMap, indexMap, segmentData, inflectionPoints, tempBuffer, tempBufferSizeInBytes );

    unsigned int indexOffset = 0;
    unsigned int vertexOffset = 0;
    for( unsigned int i = 0; i < m_numBuildInputs; ++i )
    {
        const OptixBuildInput  &bi                     = m_buildInputs[i];
        RtcBuildInput          &rtcbi                  = m_vecBuildInputs[i];
        RtcBuildInputOverrides &rtcBuildInputOverrides = m_vecBuildInputOverrides[i];

        if( numMotionSteps <= 1 )
        {
            // For buildInput[i] compute lwrve (sub-)segment aabbs and and pass the pointer to the builder.
            // If adaptive splitting is done, the aabbs of the lwrve (sub-)segments have already been computed
            // and are stored at aabbBuffer.
            if( !m_lwrveAdaptiveSplitting )
                makeLwrveAabbs( stream, bi, 0, (unsigned int)m_numSplits, aabbBuffer, getLwrveAabbSize() );
            else
                makeLwrveSegmentAabbs( stream, bi, 0, m_numSplits, segmentMap, aabbBuffer, getLwrveAabbSize(), inflectionPoints );

            rtcbi.aabbArray.aabbBuffer = static_cast<RtcGpuVA>( aabbBuffer );

            aabbBuffer += rtcbi.aabbArray.numAabbs * getLwrveAabbSize();
        }
        else
        {
            // For buildInput[i], for each motion step compute lwrve (sub-)segment aabbs and pass the list of pointers to the builder.
            std::vector<LWdeviceptr> tmp;
            for( int motionStep = 0; motionStep < numMotionSteps; motionStep++ )
            {
                LWdeviceptr aabbs = aabbBuffer + motionStep * rtcbi.aabbArray.numAabbs * getLwrveAabbSize();
                if( !m_lwrveAdaptiveSplitting )
                    makeLwrveAabbs( stream, bi, motionStep, (unsigned int)m_numSplits, aabbs, getLwrveAabbSize() );
                else
                    makeLwrveSegmentAabbs( stream, bi, motionStep, m_numSplits, segmentMap, aabbs, getLwrveAabbSize(), inflectionPoints );
                tmp.push_back( aabbs );
            }
            lwrveAabbs.push_back( tmp );
            rtcbi.aabbArray.aabbBuffer = reinterpret_cast<RtcGpuVA>( lwrveAabbs[i].data() );

            aabbBuffer += numMotionSteps * rtcbi.aabbArray.numAabbs * getLwrveAabbSize();
        }

        if( !m_builtinISLowMem )
        {
            rtcBuildInputOverrides.primLwstomVABuffer = static_cast<RtcGpuVA>(segmentData);
            rtcBuildInputOverrides.primitiveIndexBuffer = static_cast<RtcGpuVA>(indexMap);

            if( m_lwrveAdaptiveSplitting )
                storeAdaptiveLwrveSegmentData( stream, bi, indexMap, segmentData, segmentMap, inflectionPoints, indexOffset, vertexOffset, rtcbi.aabbArray.numAabbs, numMotionSteps );
            else
                storeLwrveSegmentData( stream, bi, indexMap, segmentData, indexOffset, vertexOffset, (unsigned int)m_numSplits, numMotionSteps );

            segmentData += rtcbi.aabbArray.numAabbs * sizeof( LwrveSegmentData );
            indexMap += rtcbi.aabbArray.numAabbs * sizeof( unsigned int );
            indexOffset += bi.lwrveArray.numPrimitives;
            vertexOffset += bi.lwrveArray.numVertices;
            segmentMap += rtcbi.aabbArray.numAabbs * sizeof( int );
            inflectionPoints += bi.lwrveArray.numPrimitives * sizeof( unsigned char );
        }
        else
        {
            rtcBuildInputOverrides.primitiveIndexBuffer = static_cast<RtcGpuVA>(indexMap);
            storeLwrveIndexData( stream, bi, indexMap );
            indexMap += rtcbi.aabbArray.numAabbs * sizeof( unsigned int );
        }
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::computeSphereAabbs( LWstream               stream,
                                                 LWdeviceptr            tempBuffer,
                                                 size_t                 tempBufferSizeInBytes,
                                                 std::vector<std::vector<LWdeviceptr>>& sphereAabbs )
{
    // Compute AABBs and store them in temp buffer.

    const unsigned int numMotionSteps = m_rtcAccelOptions.motionSteps;

    LWdeviceptr aabbBuffer = 0;
    LWdeviceptr sphereData = 0;
    // Get pointer where to store sphere aabbs inside tempBuffer.
    getSphereAabbs( aabbBuffer, sphereData, tempBuffer, tempBufferSizeInBytes );

    unsigned int vertexOffset = 0;
    unsigned int j = 0;
    for( unsigned int i = 0; i < m_numBuildInputs; ++i )
    {
        const OptixBuildInput  &bi = m_buildInputs[i];
        RtcBuildInput          &rtcbi = m_vecBuildInputs[j];
        RtcBuildInputOverrides &rtcBuildInputOverrides = m_vecBuildInputOverrides[i];

        if( numMotionSteps <= 1 )
        {
            // For buildInput[i] compute sphere aabbs and and pass the pointer to the builder.
            makeSphereAabbs( stream, bi, 0, aabbBuffer, sizeof( OptixAabb ) );

            rtcbi.aabbArray.aabbBuffer = static_cast<RtcGpuVA>(aabbBuffer);

            aabbBuffer += rtcbi.aabbArray.numAabbs * sizeof( OptixAabb );
        }
        else
        {
            // For buildInput[i], for each motion step compute sphere aabbs and pass the list of pointers to the builder.
            std::vector<LWdeviceptr> tmp;
            for( int motionStep = 0; motionStep < numMotionSteps; motionStep++ )
            {
                LWdeviceptr aabbs = aabbBuffer + motionStep * rtcbi.aabbArray.numAabbs * sizeof( OptixAabb );
                makeSphereAabbs( stream, bi, motionStep, aabbs, sizeof( OptixAabb ) );
                tmp.push_back( aabbs );
            }
            sphereAabbs.push_back( tmp );
            rtcbi.aabbArray.aabbBuffer = reinterpret_cast<RtcGpuVA>( sphereAabbs[i].data() );

            aabbBuffer += numMotionSteps * rtcbi.aabbArray.numAabbs * sizeof( OptixAabb );
        }

        if( !m_builtinISLowMem )
        {
            rtcBuildInputOverrides.primLwstomVABuffer   = static_cast<RtcGpuVA>( sphereData );

            storeSphereData( stream, bi.sphereArray.numVertices, sphereData, vertexOffset, numMotionSteps );

            sphereData += rtcbi.aabbArray.numAabbs * sizeof( SphereIntersectorData );
            vertexOffset += bi.sphereArray.numVertices;
        }
        j += bi.sphereArray.numSbtRecords;
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::initExtendedAccelHeader( LWstream               stream,
                                                      LWdeviceptr            outputBuffer,
                                                      ExtendedAccelHeader*   header )
{
    bool isSpheres = ( m_builtinPrimitiveType == OPTIX_PRIMITIVE_TYPE_SPHERE );
    bool isLwrves  = !isSpheres;

    size_t numVertices   = 0;
    size_t numPrimitives = 0;
    size_t numSegments   = 0;
    size_t numSbtRecords = 0;
    if( isLwrves )
    {
        for( unsigned int i = 0; i < m_numBuildInputs; ++i )
        {
            numVertices   += m_buildInputs[i].lwrveArray.numVertices;
            numPrimitives += m_buildInputs[i].lwrveArray.numPrimitives;
            if( m_lwrveAdaptiveSplitting )
                numSegments += m_vecBuildInputs[i].aabbArray.numAabbs;
        }
    }
    else
    {
        for( unsigned int i = 0; i < m_numBuildInputs; ++i )
        {
            numVertices += m_buildInputs[i].sphereArray.numVertices;
            numSbtRecords += m_buildInputs[i].sphereArray.numSbtRecords;
        }
        numPrimitives = numVertices;
    }

    numVertices *= m_rtcAccelOptions.motionSteps;

    const bool inputNormals = ( isLwrves && m_buildInputs[0].lwrveArray.normalBuffers != nullptr &&
        m_buildInputs[0].lwrveArray.lwrveType != OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE &&
        m_buildInputs[0].lwrveArray.lwrveType != OPTIX_PRIMITIVE_TYPE_ROUND_LWBIC_BSPLINE &&
        m_buildInputs[0].lwrveArray.lwrveType != OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR &&
        m_buildInputs[0].lwrveArray.lwrveType != OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM );
    const size_t vertexDataSizeInBytes = numVertices * 4 * sizeof( float );
    const size_t normalDataSizeInBytes = inputNormals ? corelib::roundUp( numVertices * 3 * sizeof( float ), static_cast<size_t>( BUILTIN_IS_DATA_BYTE_ALIGNMENT ) ) : 0;
    const size_t indexDataSizeInBytes  = isLwrves ? corelib::roundUp( numPrimitives * sizeof( int ), static_cast<size_t>( BUILTIN_IS_DATA_BYTE_ALIGNMENT ) ) : 0;
    const size_t sbtMappingSizeInBytes = ( isSpheres && m_builtinISLowMem ) ? corelib::roundUp( numSbtRecords * sizeof( int ), static_cast<size_t>( BUILTIN_IS_DATA_BYTE_ALIGNMENT ) ) : 0;
    const size_t primDataSizeInBytes   = vertexDataSizeInBytes + normalDataSizeInBytes +
                                         indexDataSizeInBytes + sbtMappingSizeInBytes +
                                         m_numBuildInputs * sizeof( int );

    // The offsets in the ExtendedAccelHeader are stored using 32 bits, plus implicitly four 0-bits because of 16byte data alignment.
    // Therefore the largest offset has to be smaller than 2^36, restricting the memory of the lwrve data to 64GB.
    if( ( ( vertexDataSizeInBytes + normalDataSizeInBytes + indexDataSizeInBytes + sbtMappingSizeInBytes ) >> 36 ) > 0 )
    {
         return logIlwalidValue( "Data size of build inputs in GAS exceeds the maximum." );
    }

    // The lwrve/sphere data groups are stored 16-byte aligned,
    // the offsets are stored without the 4 lowermost bits.
    unsigned int normalOffsetIn16Bytes         = static_cast<unsigned int>( ( inputNormals ? ( vertexDataSizeInBytes >> 4 ) : 0 ) );
    unsigned int indexOffsetIn16Bytes          = static_cast<unsigned int>( ( isLwrves ? ( ( vertexDataSizeInBytes + normalDataSizeInBytes ) >> 4 ) : 0 ) );
    unsigned int sbtOffsetIn16Bytes            = static_cast<unsigned int>( ( isSpheres && m_builtinISLowMem ) ? ( vertexDataSizeInBytes >> 4 ) : 0 );
    unsigned int primitiveIndexOffsetIn16Bytes = static_cast<unsigned int>( ( ( vertexDataSizeInBytes + normalDataSizeInBytes + indexDataSizeInBytes + sbtMappingSizeInBytes ) >> 4 ) );

    // The pointer to the data/dataCompacted will be computed during accel build.
    header->dataOffset          = 0;
    header->dataOffset32        = 0;
    header->dataCompactedOffset = 0;
    header->dataSize            = primDataSizeInBytes;

    header->normalOffset     = normalOffsetIn16Bytes;
    header->indexOffset      = indexOffsetIn16Bytes;
    header->sbtMappingOffset = sbtOffsetIn16Bytes;
    header->primIndexOffset  = primitiveIndexOffsetIn16Bytes;
    header->numBuildInputs   = m_numBuildInputs;

    header->primitiveType = m_builtinPrimitiveType;

    header->lowMem = m_builtinISLowMem;

    if( const OptixResult result = copyExtendedAccelHeader( stream, header, outputBuffer, m_errDetails ) )
    {
        return result;
    }

    return OPTIX_SUCCESS;
}

OptixResult RtcAccelBuilder::appendbuiltinISData( LWstream               stream,
                                                  LWdeviceptr            tempBuffer,
                                                  size_t                 tempBufferSizeInBytes,
                                                  LWdeviceptr            outputBuffer,
                                                  ExtendedAccelHeader*   header )
{
    alignbuiltinISData( stream, outputBuffer );

    bool isSpheres = ( m_builtinPrimitiveType == OPTIX_PRIMITIVE_TYPE_SPHERE );
    bool isLwrves  = !isSpheres;

    // offsets of the current build input relative to aligned header->data
    size_t vertexOffsetInBytes = 0;
    size_t normalOffsetInBytes = static_cast<size_t>( header->normalOffset ) << 4;
    size_t indexOffsetInBytes  = static_cast<size_t>( header->indexOffset ) << 4;
    const unsigned int numMotionSteps = m_rtcAccelOptions.motionSteps;

    const bool inputNormals = header->normalOffset > 0;

    // Copy vertices to builtin primitive data.

    if( isLwrves )
    {
        for( unsigned int i = 0; i < m_numBuildInputs; ++i )
        {
            const OptixBuildInput &bi = m_buildInputs[i];
            copyLwrveVertices( stream, bi, outputBuffer, vertexOffsetInBytes, numMotionSteps );
            vertexOffsetInBytes += 4 * numMotionSteps * sizeof( float ) * bi.lwrveArray.numVertices;
        }
    }
    else
    {
        for( unsigned int i = 0; i < m_numBuildInputs; ++i)
        {
            const OptixBuildInput &bi = m_buildInputs[i];
            copySphereVertices( stream, bi, outputBuffer, vertexOffsetInBytes, numMotionSteps );
            vertexOffsetInBytes += 4 * numMotionSteps * sizeof( float ) * bi.sphereArray.numVertices;
        }
    }

    // If there are normals, copy them to the lwrve data.

    if( inputNormals )
    {
        for( unsigned int i = 0; i < m_numBuildInputs; ++i )
        {
            const OptixBuildInput &bi = m_buildInputs[i];
            copyLwrveNormals( stream, bi, outputBuffer, normalOffsetInBytes, numMotionSteps );
            normalOffsetInBytes += 3 * numMotionSteps * sizeof( float ) * bi.lwrveArray.numVertices;
        }
    }

    if( isLwrves )
    {
        // Copy indices to the lwrve data.
        // The indices are referencing vertices, therefore the indices need
        // to be increased by the number of vertices of the preceding build inputs.

        unsigned int vertexOffset = 0;

        for( unsigned int i = 0; i < m_numBuildInputs; ++i )
        {
            const OptixBuildInput &bi = m_buildInputs[i];
            copyLwrveIndices( stream, bi, outputBuffer, indexOffsetInBytes, vertexOffset );

            indexOffsetInBytes += sizeof( int ) * bi.lwrveArray.numPrimitives;

            // number of vertices, summed up from build inputs [0,i]
            vertexOffset += bi.lwrveArray.numVertices;
        }
    }

    if( isSpheres && m_builtinISLowMem )
    {
        size_t sbtMappingOffsetInBytes = static_cast<size_t>( header->sbtMappingOffset ) << 4;

        for( unsigned int i = 0; i < m_numBuildInputs; ++i )
        {
            const OptixBuildInput &bi = m_buildInputs[i];
            storeSphereSbtMapping( stream, bi, i, outputBuffer, sbtMappingOffsetInBytes );

            sbtMappingOffsetInBytes += bi.sphereArray.numSbtRecords * sizeof( unsigned int );
        }
    }

    // Compute and store index offsets (-bias) in builtin primitive data.

    std::vector<unsigned int> primIndexOffsets( m_numBuildInputs );
    primIndexOffsets[0] = 0;
    for( unsigned int i = 1; i < m_numBuildInputs; ++i )
    {
        unsigned int numPrimitivesPrevBuildInput = isLwrves ? m_buildInputs[i - 1].lwrveArray.numPrimitives : m_buildInputs[i - 1].sphereArray.numVertices;
        primIndexOffsets[i] = primIndexOffsets[i - 1] + numPrimitivesPrevBuildInput;
    }

    // Take offset from input into account.
    // (Subtracting the input offset might produce negative numbers for the primIndexOffsets.
    //  Using type unsigned int is possible for primIndexOffsets because on device an overflow will happen and produce the correct result.)
    if( isLwrves )
    {
        for( unsigned int i = 0; i < m_numBuildInputs; ++i )
        {
            primIndexOffsets[i] -= m_buildInputs[i].lwrveArray.primitiveIndexOffset;
        }
    }
    else
    {
        for( unsigned int i = 0; i < m_numBuildInputs; ++i )
        {
            primIndexOffsets[i] -= m_buildInputs[i].sphereArray.primitiveIndexOffset;
        }
    }

    if( OptixResult result =
        copybuiltinISIndexOffsets( stream, header, tempBuffer, tempBufferSizeInBytes, outputBuffer,
                                   m_numBuildInputs, &primIndexOffsets[0], m_errDetails ) )
        return result;

    return OPTIX_SUCCESS;
}

}  // namespace optix_exp
