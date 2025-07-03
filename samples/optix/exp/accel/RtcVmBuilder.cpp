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

#include <exp/accel/RtcVmBuilder.h>

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

#define OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( flag ) static_assert( (unsigned int)RTC_##flag == (unsigned int)OPTIX_##flag, "RTC_##flag != OPTIX_##flag" );
#define OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX2( rtcFlag, optixFlag ) static_assert( (unsigned int)rtcFlag == (unsigned int)optixFlag, "rtcFlag != optixFlag" );

#define OPTIX_VMBUILDER_STATIC_ASSERT_TYPE_SIZE( rtcType, optixType ) static_assert( sizeof( rtcType ) == sizeof( optixType ), "sizeof( rtcType ) != sizeof( optixType )" );
#define OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( rtcType, optixType, member ) \
    static_assert( offsetof( rtcType, member ) == offsetof( optixType, member ), "offsetof( rtcType, member ) != offsetof( optixType, member )" ); \
    static_assert( sizeof( rtcType::member ) == sizeof( optixType::member ), "sizeof( rtcType::member ) != sizeof( optixType::member )" );
#define OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER2( rtcType, optixType, rtcMember, optixMember ) \
    static_assert( offsetof( rtcType, rtcMember ) == offsetof( optixType, optixMember ), "offsetof( rtcType, rtcMember ) != offsetof( optixType, optixMember )" ); \
    static_assert( sizeof( rtcType::rtcMember ) == sizeof( optixType::optixMember ), "sizeof( rtcType::rtcMember ) != sizeof( optixType::optixMember )" );


    OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( VISIBILITY_MAP_SPECIAL_INDEX_FULLY_TRANSPARENT );
    OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( VISIBILITY_MAP_SPECIAL_INDEX_FULLY_OPAQUE );
    OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( VISIBILITY_MAP_SPECIAL_INDEX_FULLY_UNKNOWN_TRANSPARENT );
    OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( VISIBILITY_MAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE );

    //RtcVisibilityMapFormat
    OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX2( RTC_VISIBILITY_MAP_FORMAT_OC1_2_STATE, OPTIX_VISIBILITY_MAP_FORMAT_2_STATE );
    OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX2( RTC_VISIBILITY_MAP_FORMAT_OC1_4_STATE, OPTIX_VISIBILITY_MAP_FORMAT_4_STATE );

    //RtcGeometryFlags
    OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( GEOMETRY_FLAG_REPLACEABLE_VM_ARRAY );

    //OptixMicromeshPropertyType
    OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( MICROMESH_PROPERTY_TYPE_LWRRENT_SIZE );

    //RtcVisibilityMapFlags
    OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( VISIBILITY_MAP_FLAG_NONE );
    OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( VISIBILITY_MAP_FLAG_PREFER_FAST_TRACE );
    OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( VISIBILITY_MAP_FLAG_PREFER_FAST_BUILD );

    //RtcBuildFlags
    OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( BUILD_FLAG_ALLOW_VM_UPDATE );
    OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX( BUILD_FLAG_ALLOW_DISABLE_VMS );

    OPTIX_VMBUILDER_STATIC_ASSERT_TYPE_SIZE( RtcVisibilityMapDesc, OptixVisibilityMapDesc );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcVisibilityMapDesc, OptixVisibilityMapDesc, byteOffset );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcVisibilityMapDesc, OptixVisibilityMapDesc, subdivisionLevel );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcVisibilityMapDesc, OptixVisibilityMapDesc, format );

    // OptixBuildInputVisibilityMap does not have member baseFormat, but we lwrrently rely on the other members to match their rtcore counterparts
    OPTIX_VMBUILDER_STATIC_ASSERT_TYPE_SIZE( RtcBuildInputVisibilityMap, OptixBuildInputVisibilityMap );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcBuildInputVisibilityMap, OptixBuildInputVisibilityMap, visibilityMapArray );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcBuildInputVisibilityMap, OptixBuildInputVisibilityMap, indexBuffer );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcBuildInputVisibilityMap, OptixBuildInputVisibilityMap, indexStrideInBytes );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcBuildInputVisibilityMap, OptixBuildInputVisibilityMap, indexSizeInBytes );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcBuildInputVisibilityMap, OptixBuildInputVisibilityMap, baseLocation );

    // OptixVisibilityMapArrayBuildInput does not have member baseFormat
    OPTIX_VMBUILDER_STATIC_ASSERT_TYPE_SIZE( RtcVisibilityMapArrayBuildInput, OptixVisibilityMapArrayBuildInput );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcVisibilityMapArrayBuildInput, OptixVisibilityMapArrayBuildInput, flags );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcVisibilityMapArrayBuildInput, OptixVisibilityMapArrayBuildInput, inputBuffer );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcVisibilityMapArrayBuildInput, OptixVisibilityMapArrayBuildInput, perVmDescBuffer );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcVisibilityMapArrayBuildInput, OptixVisibilityMapArrayBuildInput, perVmDescStrideInBytes );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcVisibilityMapArrayBuildInput, OptixVisibilityMapArrayBuildInput, numVmUsageCounts );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcVisibilityMapArrayBuildInput, OptixVisibilityMapArrayBuildInput, vmUsageCounts );

    OPTIX_VMBUILDER_STATIC_ASSERT_TYPE_SIZE( RtcMicromeshBufferSizes, OptixMicromeshBufferSizes );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcMicromeshBufferSizes, OptixMicromeshBufferSizes, outputSizeInBytes );

    OPTIX_VMBUILDER_STATIC_ASSERT_TYPE_SIZE( RtcMicromeshBuffers, OptixMicromeshBuffers );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcMicromeshBuffers, OptixMicromeshBuffers, output );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcMicromeshBuffers, OptixMicromeshBuffers, outputSizeInBytes );

    OPTIX_VMBUILDER_STATIC_ASSERT_TYPE_SIZE( RtcMicromeshEmitDesc, OptixMicromeshEmitDesc );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER2( RtcMicromeshEmitDesc, OptixMicromeshEmitDesc, resultVA, result );
    OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER( RtcMicromeshEmitDesc, OptixMicromeshEmitDesc, type );

#undef OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX
#undef OPTIX_VMBUILDER_STATIC_ASSERT_FLAG_RTC_EQ_OPTIX2
#undef OPTIX_VMBUILDER_STATIC_ASSERT_TYPE_SIZE
#undef OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER
#undef OPTIX_VMBUILDER_STATIC_ASSERT_MEMBER2

RtcVmBuilder::RtcVmBuilder( DeviceContext* context, bool computeMemory, ErrorDetails& errDetails )
    : m_context( context )
    , m_computeMemory( computeMemory )
    , m_errDetails( errDetails )
{
}

OptixResult RtcVmBuilder::init( const OptixVisibilityMapArrayBuildInput* buildInput )
{
    m_buildInput = buildInput;

    m_rtcBuildInput = {};
    // above we assert that all optix members are also in and at the same offest in the rtc counterpart
    reinterpret_cast<OptixVisibilityMapArrayBuildInput&>( m_rtcBuildInput ) = *buildInput;

    // this may also happening in rtcore, but we make sure to handle in optix land
    m_rtcBuildInput.perVmDescStrideInBytes =
        ( buildInput->perVmDescStrideInBytes == 0 ) ? sizeof( RtcVisibilityMapDesc ) : buildInput->perVmDescStrideInBytes;

    return OPTIX_SUCCESS;
}

OptixResult RtcVmBuilder::computeMemoryUsage( OptixMicromeshBufferSizes* bufferSizes )
{
    RtcMicromeshBufferSizes* rtcBS = reinterpret_cast<RtcMicromeshBufferSizes*>( bufferSizes );

    if( const RtcResult rtcResult = m_context->getRtcore().visibilityMapArrayComputeMemoryUsage(
            m_context->getRtcDeviceContext(), &m_rtcBuildInput, rtcBS ) )
        return m_errDetails.logDetails( rtcResult, "Failed to compute memory usage" );

    return OPTIX_SUCCESS;
}


OptixResult RtcVmBuilder::build( LWstream                      stream,
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

    if( const RtcResult rtcResult =
            m_context->getRtcore().visibilityMapArrayBuild( commandList.get(), &m_rtcBuildInput, rtcB, numEmittedProperties, rtcE ) )
        return m_errDetails.logDetails( rtcResult, "Failed to build visibility map array" );

    if( const OptixResult result = commandList.destroy( m_errDetails ) )
    {
        return result;
    }

    return OPTIX_SUCCESS;
}

}  // namespace optix_exp

#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
