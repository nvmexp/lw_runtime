//
//  Copyright (c) 2019 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include <srcTests.h>

#include <optix_types.h>
#include <private/optix_7_enum_printers.h>

#include <Util/MakeUnique.h>
#include <exp/accel/RtcAccelBuilder.h>

#include <memory>

using namespace testing;
using namespace optix_exp;

#define ASSERT_OPTIX_SUCCESS( call_ ) ASSERT_EQ( OPTIX_SUCCESS, call_ )

namespace {

const LWdeviceptr  nulldevptr                      = 0U;
const LWdeviceptr  ARBITRARY_DEVICE_PTR            = 0xbeef00U;
const LWdeviceptr  DISTINCT_DEVICE_PTR             = 0xface00U;
const float        ARBITRARY_MOTION_TIME_BEGIN     = 1.0f;
const float        ARBITRARY_MOTION_TIME_END       = 2.0f;
const unsigned int ARBITRARY_NUM_PRIMS_PER_GAS     = 100;
const unsigned int ARBITRARY_NUM_SBTS_PER_GAS      = 100;
const unsigned int ARBITRARY_NUM_INSTANCES_PER_IAS = 100;
const int          ABI_VERSION_PRE_FAT_INSTANCES   = 20;

struct TestRtcAccelBuilderOptions : Test
{
    void SetUp() override
    {
        m_options.operation     = OPTIX_BUILD_OPERATION_BUILD;
        const bool hasTTU       = false;
        const bool hasMotionTTU = false;
        makeBuilder( ARBITRARY_NUM_PRIMS_PER_GAS, ARBITRARY_NUM_SBTS_PER_GAS, ARBITRARY_NUM_INSTANCES_PER_IAS,
                     ABI_VERSION_PRE_FAT_INSTANCES, hasTTU, hasMotionTTU );
        m_buildInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        m_buildInput.instanceArray.instances    = ARBITRARY_DEVICE_PTR;
        m_buildInput.instanceArray.numInstances = 1;
    }

    void makeBuilder( const unsigned int maxPrimsPerGAS,
                      const unsigned int maxSbtRecordsPerGAS,
                      const unsigned int maxInstancesPerIAS,
                      const int          abiVersion,
                      const bool         hasTTU,
                      const bool         hasMotionTTU )
    {
        m_builder = makeUnique<RtcAccelBuilder>( nullptr, &m_options, false, m_errDetails, maxPrimsPerGAS,
                                                 maxSbtRecordsPerGAS, maxInstancesPerIAS, abiVersion, hasTTU, hasMotionTTU );
    }

    ErrorDetails                     m_errDetails;
    OptixAccelBuildOptions           m_options{};
    std::unique_ptr<RtcAccelBuilder> m_builder;
    OptixBuildInput                  m_buildInput{};
};

struct TestRtcAccelBuilderMotionOptions : TestRtcAccelBuilderOptions
{
    void SetUp() override
    {
        m_options.motionOptions.numKeys   = 2;
        m_options.motionOptions.timeBegin = ARBITRARY_MOTION_TIME_BEGIN;
        m_options.motionOptions.timeEnd   = ARBITRARY_MOTION_TIME_END;
        TestRtcAccelBuilderOptions::SetUp();
    }
};

}  // namespace

TEST_F( TestRtcAccelBuilderOptions, as_type_bvh2 )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_EQ( RTC_ACCEL_TYPE_BVH2, rtcAccelOptions.accelType );
}

TEST_F( TestRtcAccelBuilderOptions, as_type_ttu )
{
    const bool hasTTU       = true;
    const bool hasMotionTTU = true;
    makeBuilder( ARBITRARY_NUM_PRIMS_PER_GAS, ARBITRARY_NUM_SBTS_PER_GAS, ARBITRARY_NUM_INSTANCES_PER_IAS, 20, hasTTU, hasMotionTTU );

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_EQ( RTC_ACCEL_TYPE_TTU, rtcAccelOptions.accelType );
}

TEST_F( TestRtcAccelBuilderOptions, uses_universal_format )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_TRUE( rtcAccelOptions.useUniversalFormat );
}

TEST_F( TestRtcAccelBuilderOptions, build_doesnt_refit )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_FALSE( rtcAccelOptions.refit );
}

TEST_F( TestRtcAccelBuilderOptions, update_does_refit )
{
    m_options.operation  = OPTIX_BUILD_OPERATION_UPDATE;
    m_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_TRUE( rtcAccelOptions.refit );
}

TEST_F( TestRtcAccelBuilderOptions, bake_triangles )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_TRUE( rtcAccelOptions.bakeTriangles );
}

TEST_F( TestRtcAccelBuilderOptions, use_prim_bits )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_TRUE( rtcAccelOptions.usePrimBits );
}

TEST_F( TestRtcAccelBuilderOptions, no_use_remap_for_prim_bits )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_FALSE( rtcAccelOptions.useRemapForPrimBits );
}

TEST_F( TestRtcAccelBuilderOptions, no_build_reordering )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_FALSE( rtcAccelOptions.enableBuildReordering );
}

TEST_F( TestRtcAccelBuilderOptions, clamp_aabbs_to_valid_range )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_TRUE( rtcAccelOptions.clampAabbsToValidRange );
}

TEST_F( TestRtcAccelBuilderOptions, high_precision_math )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_TRUE( rtcAccelOptions.highPrecisionMath );
}

TEST_F( TestRtcAccelBuilderOptions, single_motion_step )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_EQ( 1U, rtcAccelOptions.motionSteps );
}

TEST_F( TestRtcAccelBuilderOptions, no_motion_flags )
{
    m_options.motionOptions.flags = OPTIX_MOTION_FLAG_START_VANISH | OPTIX_MOTION_FLAG_END_VANISH;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_EQ( 0U, rtcAccelOptions.motionFlags );
}

TEST_F( TestRtcAccelBuilderOptions, no_motion_times )
{
    m_options.motionOptions.timeBegin = ARBITRARY_MOTION_TIME_BEGIN;
    m_options.motionOptions.timeEnd   = ARBITRARY_MOTION_TIME_END;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    EXPECT_EQ( 0.0f, rtcAccelOptions.motionTimeBegin );
    EXPECT_EQ( 0.0f, rtcAccelOptions.motionTimeEnd );
}

TEST_F( TestRtcAccelBuilderOptions, translates_update_flag )
{
    m_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_EQ( RTC_BUILD_FLAG_ALLOW_UPDATE, rtcAccelOptions.buildFlags );
}

TEST_F( TestRtcAccelBuilderOptions, translates_compaction_flag )
{
    m_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_EQ( RTC_BUILD_FLAG_ALLOW_COMPACTION, rtcAccelOptions.buildFlags );
}

TEST_F( TestRtcAccelBuilderOptions, translates_fast_trace_flag )
{
    m_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_EQ( RTC_BUILD_FLAG_PREFER_FAST_TRACE, rtcAccelOptions.buildFlags );
}

TEST_F( TestRtcAccelBuilderMotionOptions, as_type_mbvh2 )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    ASSERT_EQ( RTC_ACCEL_TYPE_MBVH2, rtcAccelOptions.accelType );
}

TEST_F( TestRtcAccelBuilderMotionOptions, passes_motion_steps )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    EXPECT_EQ( 2U, rtcAccelOptions.motionSteps );
}

TEST_F( TestRtcAccelBuilderMotionOptions, passes_motion_begin_end )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    EXPECT_EQ( ARBITRARY_MOTION_TIME_BEGIN, rtcAccelOptions.motionTimeBegin );
    EXPECT_EQ( ARBITRARY_MOTION_TIME_END, rtcAccelOptions.motionTimeEnd );
}

TEST_F( TestRtcAccelBuilderMotionOptions, translates_start_vanish_flag )
{
    m_options.motionOptions.flags = OPTIX_MOTION_FLAG_START_VANISH;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    EXPECT_TRUE( ( rtcAccelOptions.motionFlags & RTC_MOTION_FLAG_START_VANISH ) != 0 );
}

TEST_F( TestRtcAccelBuilderMotionOptions, translates_start_end_flag )
{
    m_options.motionOptions.flags = OPTIX_MOTION_FLAG_END_VANISH;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcAccelOptions rtcAccelOptions = m_builder->getRtcAccelOptions();
    EXPECT_TRUE( ( rtcAccelOptions.motionFlags & RTC_MOTION_FLAG_END_VANISH ) != 0 );
}

struct TestRtcAccelBuilderTriangleInput : TestRtcAccelBuilderOptions
{
    void SetUp() override
    {
        TestRtcAccelBuilderOptions::SetUp();
        m_buildInput.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        m_buildInput.triangleArray               = OptixBuildInputTriangleArray{};
        m_buildInput.triangleArray.vertexBuffers = &m_vb;
        m_buildInput.triangleArray.numVertices   = 3;
        m_buildInput.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        m_buildInput.triangleArray.flags         = &m_sbtFlags;
        m_buildInput.triangleArray.numSbtRecords = 1;
    }

    LWdeviceptr  m_vb       = ARBITRARY_DEVICE_PTR;
    unsigned int m_sbtFlags = 0U;
};

TEST_F( TestRtcAccelBuilderTriangleInput, num_build_inputs )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const std::vector<RtcBuildInput>& inputs = m_builder->getBuildInputs();
    ASSERT_EQ( 1U, inputs.size() );
}

TEST_F( TestRtcAccelBuilderTriangleInput, build_type )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcBuildInput& input = m_builder->getBuildInputs()[0];
    ASSERT_EQ( RTC_BUILD_INPUT_TYPE_TRIANGLE_ARRAY, input.type );
}

struct VertexFormatTranslation
{
    OptixVertexFormat optixFormat;
    RtcVertexFormat   rtcoreFormat;
};

struct TriangleInputVertexFormat : TestRtcAccelBuilderTriangleInput, WithParamInterface<VertexFormatTranslation>
{
    VertexFormatTranslation m_param = GetParam();
};

std::ostream& operator<<( std::ostream& str, VertexFormatTranslation value )
{
    return str << value.optixFormat << ", " << value.rtcoreFormat;
}

TEST_P( TriangleInputVertexFormat, translated )
{
    m_buildInput.triangleArray.vertexFormat = m_param.optixFormat;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcBuildInputTriangleArray& input = m_builder->getBuildInputs()[0].triangleArray;
    ASSERT_EQ( m_param.rtcoreFormat, input.vertexFormat );
}

static VertexFormatTranslation translations[] = {
    {OPTIX_VERTEX_FORMAT_FLOAT3, RTC_VERTEX_FORMAT_FLOAT3},
    {OPTIX_VERTEX_FORMAT_FLOAT2, RTC_VERTEX_FORMAT_FLOAT2},
    {OPTIX_VERTEX_FORMAT_HALF3, RTC_VERTEX_FORMAT_HALF3},
    {OPTIX_VERTEX_FORMAT_HALF2, RTC_VERTEX_FORMAT_HALF2},
    {OPTIX_VERTEX_FORMAT_SNORM16_3, RTC_VERTEX_FORMAT_SNORM16_3},
    {OPTIX_VERTEX_FORMAT_SNORM16_2, RTC_VERTEX_FORMAT_SNORM16_2},
};

INSTANTIATE_TEST_SUITE_P( TestRtcAccelBuilderAllVertexFormats, TriangleInputVertexFormat, ValuesIn( translations ) );

TEST_F( TestRtcAccelBuilderTriangleInput, vertex_buffer )
{
    LWdeviceptr distinctVb                   = DISTINCT_DEVICE_PTR;
    m_buildInput.triangleArray.vertexBuffers = &distinctVb;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcBuildInputTriangleArray& input = m_builder->getBuildInputs()[0].triangleArray;
    ASSERT_EQ( DISTINCT_DEVICE_PTR, input.vertexBuffer );
}

TEST_F( TestRtcAccelBuilderTriangleInput, num_vertices )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcBuildInputTriangleArray& input = m_builder->getBuildInputs()[0].triangleArray;
    ASSERT_EQ( 3, input.numVertices );
}

struct VertexFormatStride
{
    OptixVertexFormat format;
    unsigned int      stride;
};

std::ostream& operator<<( std::ostream& str, VertexFormatStride value )
{
    return str << value.format << ", " << value.stride;
}

struct TriangleInputVertexStride : TestRtcAccelBuilderTriangleInput, WithParamInterface<VertexFormatStride>
{
    VertexFormatStride m_param = GetParam();
};

TEST_P( TriangleInputVertexStride, computed_stride )
{
    m_buildInput.triangleArray.vertexFormat = m_param.format;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcBuildInputTriangleArray& input = m_builder->getBuildInputs()[0].triangleArray;
    ASSERT_EQ( m_param.stride, input.vertexStrideInBytes );
}

static VertexFormatStride vertexStrides[] = {
    {OPTIX_VERTEX_FORMAT_FLOAT3, 3 * sizeof( float )},    {OPTIX_VERTEX_FORMAT_FLOAT2, 2 * sizeof( float )},
    {OPTIX_VERTEX_FORMAT_HALF3, 3 * sizeof( float ) / 2}, {OPTIX_VERTEX_FORMAT_HALF2, 2 * sizeof( float ) / 2},
    {OPTIX_VERTEX_FORMAT_SNORM16_3, 3 * sizeof( short )}, {OPTIX_VERTEX_FORMAT_SNORM16_2, 2 * sizeof( short )},
};

INSTANTIATE_TEST_SUITE_P( TestRtcAccelBuilderAllVertexFormats, TriangleInputVertexStride, ValuesIn( vertexStrides ) );

TEST_F( TestRtcAccelBuilderTriangleInput, lwstom_stride_forwarded )
{
    m_buildInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    const unsigned int lwstomStride                = 4 * sizeof( float );
    m_buildInput.triangleArray.vertexStrideInBytes = lwstomStride;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcBuildInputTriangleArray& input = m_builder->getBuildInputs()[0].triangleArray;
    ASSERT_EQ( lwstomStride, input.vertexStrideInBytes );
}

TEST_F( TestRtcAccelBuilderTriangleInput, no_index_buffer )
{
    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcBuildInputTriangleArray& input = m_builder->getBuildInputs()[0].triangleArray;
    EXPECT_EQ( nulldevptr, input.indexBuffer );
    EXPECT_EQ( 0U, input.numIndices );
    EXPECT_EQ( 0U, input.indexStrideInBytes );
    EXPECT_EQ( 0U, input.indexSizeInBytes );
}

TEST_F( TestRtcAccelBuilderTriangleInput, index_buffer )
{
    m_buildInput.triangleArray.indexBuffer        = DISTINCT_DEVICE_PTR;
    m_buildInput.triangleArray.numIndexTriplets   = 1;
    m_buildInput.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
    m_buildInput.triangleArray.indexStrideInBytes = 0;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcBuildInputTriangleArray& input = m_builder->getBuildInputs()[0].triangleArray;
    ASSERT_EQ( DISTINCT_DEVICE_PTR, input.indexBuffer );
}

TEST_F( TestRtcAccelBuilderTriangleInput, num_indices )
{
    m_buildInput.triangleArray.indexBuffer        = DISTINCT_DEVICE_PTR;
    m_buildInput.triangleArray.numIndexTriplets   = 3;
    m_buildInput.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
    m_buildInput.triangleArray.indexStrideInBytes = 0;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcBuildInputTriangleArray& input = m_builder->getBuildInputs()[0].triangleArray;
    ASSERT_EQ( 3 * 3U, input.numIndices );
}

struct IndexFormatSizeStride
{
    OptixIndicesFormat format;
    unsigned int       size;
    unsigned int       stride;
};

std::ostream& operator<<( std::ostream& str, IndexFormatSizeStride value )
{
    return str << value.format << ", " << value.size << ", " << value.stride;
}

struct TriangleInputIndexStride : TestRtcAccelBuilderTriangleInput, WithParamInterface<IndexFormatSizeStride>
{
    void SetUp() override
    {
        TestRtcAccelBuilderTriangleInput::SetUp();
        m_buildInput.triangleArray.indexBuffer        = DISTINCT_DEVICE_PTR;
        m_buildInput.triangleArray.numIndexTriplets   = 1;
        m_buildInput.triangleArray.indexStrideInBytes = 0;
    }

    IndexFormatSizeStride m_param = GetParam();
};

TEST_P( TriangleInputIndexStride, computed_stride )
{
    m_buildInput.triangleArray.indexFormat = m_param.format;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcBuildInputTriangleArray& input = m_builder->getBuildInputs()[0].triangleArray;
    ASSERT_EQ( m_param.stride, input.indexStrideInBytes );
}

TEST_P( TriangleInputIndexStride, size )
{
    m_buildInput.triangleArray.indexFormat = m_param.format;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcBuildInputTriangleArray& input = m_builder->getBuildInputs()[0].triangleArray;
    ASSERT_EQ( m_param.size, input.indexSizeInBytes );
}

static IndexFormatSizeStride indexSizeStrides[] = {
    {OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3, sizeof( unsigned short ), 3 * sizeof( unsigned short )},
    {OPTIX_INDICES_FORMAT_UNSIGNED_INT3, sizeof( unsigned int ), 3 * sizeof( unsigned int )},
};

INSTANTIATE_TEST_SUITE_P( TestRtcAccelBuilderAllIndexFormats, TriangleInputIndexStride, ValuesIn( indexSizeStrides ) );

TEST_F( TestRtcAccelBuilderTriangleInput, lwstom_index_stride )
{
    m_buildInput.triangleArray.indexBuffer        = DISTINCT_DEVICE_PTR;
    m_buildInput.triangleArray.numIndexTriplets   = 1;
    m_buildInput.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    const unsigned int lwstomStride               = 4 * sizeof( unsigned int );
    m_buildInput.triangleArray.indexStrideInBytes = lwstomStride;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcBuildInputTriangleArray& input = m_builder->getBuildInputs()[0].triangleArray;
    ASSERT_EQ( lwstomStride, input.indexStrideInBytes );
}

struct TriangleBuildFlags
{
    unsigned int optixFlags;
    unsigned int rtcoreFlags;
};

std::ostream& operator<<( std::ostream& str, const TriangleBuildFlags& param )
{
    return str << OptixGeometryFlagSet{param.optixFlags} << ", " << param.rtcoreFlags;
}

struct TriangleInputGeometryFlags : TestRtcAccelBuilderTriangleInput, WithParamInterface<TriangleBuildFlags>
{
    void SetUp() override
    {
        TestRtcAccelBuilderTriangleInput::SetUp();
        m_buildInput.triangleArray.flags = &m_param.optixFlags;
    }

    TriangleBuildFlags m_param = GetParam();
};

TEST_P( TriangleInputGeometryFlags, flags )
{
    m_buildInput.triangleArray.flags = &m_param.optixFlags;

    ASSERT_OPTIX_SUCCESS( m_builder->init( &m_buildInput, 1 ) );

    const RtcBuildInputTriangleArray& input = m_builder->getBuildInputs()[0].triangleArray;
    ASSERT_EQ( m_param.rtcoreFlags, input.flags );
}

static TriangleBuildFlags triangleBuildFlags[] = {
    {OPTIX_GEOMETRY_FLAG_NONE, RTC_GEOMETRY_FLAG_NONE},
    {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT, RTC_GEOMETRY_FLAG_OPAQUE},
    {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL, RTC_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_ILWOCATION},
    {OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_LWLLING, RTC_GEOMETRY_FLAG_TRIANGLE_LWLL_DISABLE}
};

INSTANTIATE_TEST_SUITE_P( TestRtcAccelBuilderAllGeometryBuildFlags, TriangleInputGeometryFlags, ValuesIn( triangleBuildFlags ) );
