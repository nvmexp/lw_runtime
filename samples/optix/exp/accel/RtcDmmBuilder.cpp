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

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

#include <rtcore/interface/rtcore.h>
#include <rtcore/interface/types.h>

#include <exp/accel/RtcDmmBuilder.h>

#include <exp/context/DeviceContext.h>
#include <exp/context/ErrorHandling.h>

#include <corelib/misc/String.h>
#include <prodlib/system/Knobs.h>

#include <lwca.h>

namespace {
// clang-format off
    //Knob<std::string>  k_dumpLSS( RT_DSTRING( "o7.accel.dumpLSS" ), "", RT_DSTRING( "Dump the lwrve LSS bounds to a .hair file with of linear lwrves." ) );
// clang-format on
}  // namespace

namespace optix_exp {

#define OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( flag ) static_assert( (unsigned int)RTC_##flag == (unsigned int)OPTIX_##flag, "RTC_##flag != OPTIX_##flag" );
#define OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX2( rtcFlag, optixFlag ) static_assert( (unsigned int)rtcFlag == (unsigned int)optixFlag, "rtcFlag != optixFlag" );

#define OPTIX_DMMBUILDER_STATIC_ASSERT_TYPE_SIZE( rtcType, optixType ) static_assert( sizeof( rtcType ) == sizeof( optixType ), "sizeof( rtcType ) != sizeof( optixType )" );
#define OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( rtcType, optixType, member ) \
    static_assert( offsetof( rtcType, member ) == offsetof( optixType, member ), "offsetof( rtcType, member ) != offsetof( optixType, member )" ); \
    static_assert( sizeof( rtcType::member ) == sizeof( optixType::member ), "sizeof( rtcType::member ) != sizeof( optixType::member )" );
#define OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER2( rtcType, optixType, rtcMember, optixMember ) \
    static_assert( offsetof( rtcType, rtcMember ) == offsetof( optixType, optixMember ), "offsetof( rtcType, rtcMember ) != offsetof( optixType, optixMember )" ); \
    static_assert( sizeof( rtcType::rtcMember ) == sizeof( optixType::optixMember ), "sizeof( rtcType::rtcMember ) != sizeof( optixType::optixMember )" );



    //RtcDisplacedMicromeshFormat
    OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX2( RTC_DISPLACED_MICROMESH_FORMAT_DC1_64_TRIS_64_BYTES, OPTIX_DISPLACED_MICROMESH_FORMAT_64_TRIS_64_BYTES );
    OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX2( RTC_DISPLACED_MICROMESH_FORMAT_DC1_256_TRIS_128_BYTES, OPTIX_DISPLACED_MICROMESH_FORMAT_256_TRIS_128_BYTES );
    OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX2( RTC_DISPLACED_MICROMESH_FORMAT_DC1_1024_TRIS_128_BYTES, OPTIX_DISPLACED_MICROMESH_FORMAT_1024_TRIS_128_BYTES );

    //RtcDisplacedMicromeshFlags
    OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( DISPLACED_MICROMESH_FLAG_NONE );
    OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( DISPLACED_MICROMESH_FLAG_PREFER_FAST_TRACE );
    OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( DISPLACED_MICROMESH_FLAG_PREFER_FAST_BUILD );

    //RtcDisplacementPrimitiveFlags
    OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( DISPLACED_MICROMESH_PRIMITIVE_FLAG_DECIMATE_01 );
    OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( DISPLACED_MICROMESH_PRIMITIVE_FLAG_DECIMATE_12 );
    OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( DISPLACED_MICROMESH_PRIMITIVE_FLAG_DECIMATE_20 );

    //RtcBuildFlags
    OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( BUILD_FLAG_ALLOW_DMM_UPDATE );
    OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX2( RTC_BUILD_FLAG_REFERENCED_BLAS_HAS_DMM, OPTIX_BUILD_FLAG_IAS_REFERENCES_GAS_WITH_DMM );

    //RtcDisplacementVectorFormat
    OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX2( RTC_DISPLACEMENT_VECTOR_FORMAT_FORMAT_HALF3,
                                                       OPTIX_MICROMESH_DISPLACEMENT_VECTOR_FORMAT_HALF3 );

    OPTIX_DMMBUILDER_STATIC_ASSERT_TYPE_SIZE( RtcVisibilityMapDesc, OptixVisibilityMapDesc );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcVisibilityMapDesc, OptixVisibilityMapDesc, byteOffset );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcVisibilityMapDesc, OptixVisibilityMapDesc, subdivisionLevel );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcVisibilityMapDesc, OptixVisibilityMapDesc, format );

    OPTIX_DMMBUILDER_STATIC_ASSERT_TYPE_SIZE( RtcBuildInputDisplacement, OptixBuildInputDisplacement );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcBuildInputDisplacement, OptixBuildInputDisplacement, displacedMicromeshArray );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER2( RtcBuildInputDisplacement, OptixBuildInputDisplacement, dmmIndexBuffer, displacedMicromeshIndexBuffer );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER2( RtcBuildInputDisplacement, OptixBuildInputDisplacement, dispVectors, displacementVectorBuffer );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER2( RtcBuildInputDisplacement, OptixBuildInputDisplacement, biasAndScale, displacementBiasAndScaleBuffer );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER2( RtcBuildInputDisplacement, OptixBuildInputDisplacement, primitiveFlags, primitiveFlagBuffer );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcBuildInputDisplacement, OptixBuildInputDisplacement, baseLocation );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER2( RtcBuildInputDisplacement, OptixBuildInputDisplacement, dmmIndexStrideInBytes, displacedMicromeshIndexStrideInBytes );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER2( RtcBuildInputDisplacement, OptixBuildInputDisplacement, dmmIndexSizeInBytes, displacedMicromeshIndexSizeInBytes );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER2( RtcBuildInputDisplacement, OptixBuildInputDisplacement, dispVectorFormat, displacementVectorFormat );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER2( RtcBuildInputDisplacement, OptixBuildInputDisplacement, dispVectorStrideInBytes, displacementVectorStrideInBytes );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER2( RtcBuildInputDisplacement, OptixBuildInputDisplacement, biasAndScaleFormat, displacementBiasAndScaleFormat );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER2( RtcBuildInputDisplacement, OptixBuildInputDisplacement, biasAndScaleStrideInBytes, displacementBiasAndScaleStrideInBytes );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcBuildInputDisplacement, OptixBuildInputDisplacement, primitiveFlagsStrideInBytes );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcBuildInputDisplacement, OptixBuildInputDisplacement, numDmmUsageCounts );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcBuildInputDisplacement, OptixBuildInputDisplacement, dmmUsageCounts );

    OPTIX_DMMBUILDER_STATIC_ASSERT_TYPE_SIZE( RtcDisplacedMicromeshArrayBuildInput, OptixDisplacedMicromeshArrayBuildInput );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcDisplacedMicromeshArrayBuildInput, OptixDisplacedMicromeshArrayBuildInput, flags );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcDisplacedMicromeshArrayBuildInput, OptixDisplacedMicromeshArrayBuildInput, inputBuffer );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcDisplacedMicromeshArrayBuildInput, OptixDisplacedMicromeshArrayBuildInput, perDmmDescBuffer );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcDisplacedMicromeshArrayBuildInput, OptixDisplacedMicromeshArrayBuildInput, perDmmDescStrideInBytes );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcDisplacedMicromeshArrayBuildInput, OptixDisplacedMicromeshArrayBuildInput, numDmmUsageCounts );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcDisplacedMicromeshArrayBuildInput, OptixDisplacedMicromeshArrayBuildInput, dmmUsageCounts );

    OPTIX_DMMBUILDER_STATIC_ASSERT_TYPE_SIZE( RtcMicromeshBufferSizes, OptixMicromeshBufferSizes );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcMicromeshBufferSizes, OptixMicromeshBufferSizes, outputSizeInBytes );

    OPTIX_DMMBUILDER_STATIC_ASSERT_TYPE_SIZE( RtcMicromeshBuffers, OptixMicromeshBuffers );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcMicromeshBuffers, OptixMicromeshBuffers, output );
    OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER( RtcMicromeshBuffers, OptixMicromeshBuffers, outputSizeInBytes );

#undef OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX
#undef OPTIX_DMMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX2
#undef OPTIX_DMMBUILDER_STATIC_ASSERT_TYPE_SIZE
#undef OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER
#undef OPTIX_DMMBUILDER_STATIC_ASSERT_MEMBER2

RtcDmmBuilder::RtcDmmBuilder( DeviceContext* context, bool computeMemory, ErrorDetails& errDetails )
    : m_context( context )
    , m_computeMemory( computeMemory )
    , m_errDetails( errDetails )
{
}

OptixResult RtcDmmBuilder::init( const OptixDisplacedMicromeshArrayBuildInput* buildInput )
{
    m_buildInput = buildInput;

    m_rtcBuildInput = {};
    // above we assert that all optix members are also in and at the same offest in the rtc counterpart
    reinterpret_cast<OptixDisplacedMicromeshArrayBuildInput&>( m_rtcBuildInput ) = *buildInput;

    // this may also happening in rtcore, but we make sure to handle in optix land
    m_rtcBuildInput.perDmmDescStrideInBytes =
        ( buildInput->perDmmDescStrideInBytes == 0 ) ? sizeof( RtcDisplacedMicromeshDesc ) : buildInput->perDmmDescStrideInBytes;

    return OPTIX_SUCCESS;
}

OptixResult RtcDmmBuilder::computeMemoryUsage( OptixMicromeshBufferSizes* bufferSizes )
{
    RtcMicromeshBufferSizes* rtcBS = reinterpret_cast<RtcMicromeshBufferSizes*>( bufferSizes );

    if( const RtcResult rtcResult = m_context->getRtcore().displacedMicromeshArrayComputeMemoryUsage(
            m_context->getRtcDeviceContext(), &m_rtcBuildInput, rtcBS ) )
        return m_errDetails.logDetails( rtcResult, "Failed to compute memory usage" );

    return OPTIX_SUCCESS;
}


OptixResult RtcDmmBuilder::build( LWstream                      stream,
                                  const OptixMicromeshBuffers*  buffers,
                                  const OptixMicromeshEmitDesc* emittedProperties,
                                  unsigned int                  numEmittedProperties )
{
    const RtcMicromeshBuffers*  rtcB = reinterpret_cast<const RtcMicromeshBuffers*>( buffers );
    const RtcMicromeshEmitDesc* rtcE = reinterpret_cast<const RtcMicromeshEmitDesc*>( emittedProperties );

    ScopedCommandList commandList( m_context );
    if( const OptixResult result = commandList.init( stream, m_errDetails ) )
    {
        return result;
    }

    if( const RtcResult rtcResult = m_context->getRtcore().displacedMicromeshArrayBuild( commandList.get(), &m_rtcBuildInput,
                                                                                         rtcB, numEmittedProperties, rtcE ) )
        return m_errDetails.logDetails( rtcResult, "Failed to build displaced micromesh array" );

    if( const OptixResult result = commandList.destroy( m_errDetails ) )
    {
        return result;
    }

    return OPTIX_SUCCESS;
}

}  // namespace optix_exp

#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
