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

#include <optix.h>
#include <optix_stubs.h>

#include <exptest/loggingHelpers.h>
#include <exptest/testTools.h>

#include <gmock/gmock.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "CommonAsserts.h"
#include "StrictMockLogger.h"

using namespace testing;

namespace {

const OptixTraversableHandle NULL_TRAVERSABLE_HANDLE          = 0ULL;
const OptixTraversableHandle TRAVERSABLE_HANDLE_INITIAL_VALUE = 0xdeadbeefULL;
const OptixBuildInputType    ILWALID_BUILD_INPUT_TYPE         = static_cast<OptixBuildInputType>( 0 );
const size_t                 TEMP_BUFFER_SIZE                 = 64 * 1024;  // 64 kb
const size_t                 OUTPUT_BUFFER_SIZE               = 64 * 1024;  // 64 kb
const LWdeviceptr            nulldevptr                       = static_cast<LWdeviceptr>( 0U );
const LWdeviceptr            ARBITRARY_DEVICE_PTR             = 0xbeef00U;
const unsigned int           DEVICE_PTR_ALIGNMENT             = static_cast<unsigned int>( sizeof( std::uint64_t ) );

template <OptixVertexFormat>
struct NaturalVertexStride;

// clang-format off
#define DECLARE_NATURAL_VERTEX_STRIDE( format_, stride_ )           \
template <>                                                         \
struct NaturalVertexStride<format_>                                 \
{                                                                   \
    static const unsigned int stride;                               \
};                                                                  \
const unsigned int NaturalVertexStride<format_>::stride = stride_
// clang-format on

DECLARE_NATURAL_VERTEX_STRIDE( OPTIX_VERTEX_FORMAT_FLOAT3, 3 * sizeof( float ) );
DECLARE_NATURAL_VERTEX_STRIDE( OPTIX_VERTEX_FORMAT_FLOAT2, 2 * sizeof( float ) );
DECLARE_NATURAL_VERTEX_STRIDE( OPTIX_VERTEX_FORMAT_HALF3, 3 * sizeof( float ) / 2 );
DECLARE_NATURAL_VERTEX_STRIDE( OPTIX_VERTEX_FORMAT_HALF2, 2 * sizeof( float ) / 2 );
DECLARE_NATURAL_VERTEX_STRIDE( OPTIX_VERTEX_FORMAT_SNORM16_3, 3 * sizeof( short ) );
DECLARE_NATURAL_VERTEX_STRIDE( OPTIX_VERTEX_FORMAT_SNORM16_2, 2 * sizeof( short ) );

#undef DECLARE_NATURAL_VERTEX_STRIDE

// clang-format off
#define NATURAL_VERTEX_STRIDE_CASE( fmt_ )          \
    case fmt_:                                      \
        return NaturalVertexStride<fmt_>::stride
// clang-format on

unsigned int naturalVertexStride( OptixVertexFormat fmt )
{
    switch( fmt )
    {
        NATURAL_VERTEX_STRIDE_CASE( OPTIX_VERTEX_FORMAT_FLOAT3 );
        NATURAL_VERTEX_STRIDE_CASE( OPTIX_VERTEX_FORMAT_FLOAT2 );
        NATURAL_VERTEX_STRIDE_CASE( OPTIX_VERTEX_FORMAT_HALF3 );
        NATURAL_VERTEX_STRIDE_CASE( OPTIX_VERTEX_FORMAT_HALF2 );
        NATURAL_VERTEX_STRIDE_CASE( OPTIX_VERTEX_FORMAT_SNORM16_3 );
        NATURAL_VERTEX_STRIDE_CASE( OPTIX_VERTEX_FORMAT_SNORM16_2 );
        case OPTIX_VERTEX_FORMAT_NONE:
            break;
    }
    throw std::runtime_error( "Unknown vertex format" );
}

#undef NATURAL_VERTEX_STRIDE_CASE

template <OptixIndicesFormat>
struct NaturalIndexStride;

// clang-format off
#define DECLARE_NATURAL_INDEX_STRIDE( format_, stride_ )            \
template <>                                                         \
struct NaturalIndexStride<format_>                                  \
{                                                                   \
    static const unsigned int stride;                               \
};                                                                  \
const unsigned int NaturalIndexStride<format_>::stride = stride_
// clang-format on

DECLARE_NATURAL_INDEX_STRIDE( OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3, 3 * sizeof( unsigned short ) );
DECLARE_NATURAL_INDEX_STRIDE( OPTIX_INDICES_FORMAT_UNSIGNED_INT3, 3 * sizeof( unsigned int ) );

#undef DECLARE_NATURAL_INDEX_STRIDE

// clang-format off
#define NATURAL_INDEX_STRIDE_CASE( fmt_ )           \
    case fmt_:                                      \
        return NaturalIndexStride<fmt_>::stride
// clang-format on

unsigned int naturalIndexStride( OptixIndicesFormat fmt )
{
    switch( fmt )
    {
        NATURAL_INDEX_STRIDE_CASE( OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 );
        NATURAL_INDEX_STRIDE_CASE( OPTIX_INDICES_FORMAT_UNSIGNED_INT3 );
        case OPTIX_INDICES_FORMAT_NONE:
            break;
    }
    throw std::runtime_error( "Unknown index format" );
}

struct TriangleBuildInput
{
    OptixBuildInput           buildInput;
    std::vector<unsigned int> flags;
};

TriangleBuildInput createEmptyTriangleInput()
{
    TriangleBuildInput result{};
    result.buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    result.flags.resize( 1 );
    result.buildInput.triangleArray.numSbtRecords = 1;
    result.buildInput.triangleArray.flags         = &result.flags[0];
    return result;
}

struct TestAccelBuild : Test
{
    void SetUp() override
    {
        exptest::lwdaInitialize();
        ASSERT_OPTIX_SUCCESS( optixInit() );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( nullptr, nullptr, &m_context ) );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextSetLogCallback( m_context, &StrictMockLogger::apiCallback, &m_logger, LOG_LEVEL_ERROR ) );
        ASSERT_LWDA_SUCCESS( lwdaStreamCreate( &m_stream ) );
        ASSERT_LWDA_SUCCESS( lwdaMalloc( reinterpret_cast<void**>( &m_tempBuffer ), TEMP_BUFFER_SIZE ) );
        ASSERT_LWDA_SUCCESS( lwdaMalloc( reinterpret_cast<void**>( &m_outputBuffer ), OUTPUT_BUFFER_SIZE ) );
        m_tempBufferSize         = TEMP_BUFFER_SIZE;
        m_outputBufferSize       = OUTPUT_BUFFER_SIZE;
        m_accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        m_buildInputs.resize( 1, OptixBuildInput{} );
    }

    void TearDown() override
    {
        ASSERT_LWDA_SUCCESS( lwdaFree( reinterpret_cast<void*>( m_outputBuffer ) ) );
        ASSERT_LWDA_SUCCESS( lwdaFree( reinterpret_cast<void*>( m_tempBuffer ) ) );
        ASSERT_LWDA_SUCCESS( lwdaStreamDestroy( m_stream ) );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextDestroy( m_context ) );
    }

    void resizeBuffers()
    {
        OptixAccelBufferSizes bufferSizes;
        ASSERT_OPTIX_SUCCESS( optixAccelComputeMemoryUsage( m_context, &m_accelOptions, m_buildInputs.data(), 1, &bufferSizes ) );
        if( bufferSizes.tempSizeInBytes > TEMP_BUFFER_SIZE )
        {
            ASSERT_LWDA_SUCCESS( lwdaFree( reinterpret_cast<void*>( m_tempBuffer ) ) );
            ASSERT_LWDA_SUCCESS( lwdaMalloc( reinterpret_cast<void**>( &m_tempBuffer ), bufferSizes.tempSizeInBytes ) );
            m_tempBufferSize = bufferSizes.tempSizeInBytes;
        }
        if( bufferSizes.outputSizeInBytes > OUTPUT_BUFFER_SIZE )
        {
            ASSERT_LWDA_SUCCESS( lwdaFree( reinterpret_cast<void*>( m_outputBuffer ) ) );
            ASSERT_LWDA_SUCCESS( lwdaMalloc( reinterpret_cast<void**>( &m_outputBuffer ), bufferSizes.outputSizeInBytes ) );
            m_outputBufferSize = bufferSizes.outputSizeInBytes;
        }
    }

    OptixResult memComputeAccelOneInput()
    {
        OptixAccelBufferSizes bufferSizes;
        return optixAccelComputeMemoryUsage( m_context, &m_accelOptions, &m_buildInputs[0],
                                             static_cast<unsigned int>( m_buildInputs.size() ), &bufferSizes );
    }

    OptixResult buildAccelOneInput()
    {
        OptixAccelBufferSizes bufferSizes;
        if( const OptixResult result = optixAccelComputeMemoryUsage( m_context, &m_accelOptions, &m_buildInputs[0],
                                                                     static_cast<unsigned int>( m_buildInputs.size() ), &bufferSizes ) )
            return result;

        return optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer,
                                m_tempBufferSize, m_outputBuffer, m_outputBufferSize, &m_handle, nullptr, 0 );
    }

    OptixResult buildAccelOneInputEmptyDevPtrsForComputeMemory()
    {
        OptixBuildInput tempBi = m_buildInputs[0];
        switch( m_buildInputs[0].type )
        {
            case OPTIX_BUILD_INPUT_TYPE_TRIANGLES:
                tempBi.triangleArray.vertexBuffers        = nullptr;
                tempBi.triangleArray.indexBuffer          = nulldevptr;
                tempBi.triangleArray.preTransform         = nulldevptr;
                tempBi.triangleArray.sbtIndexOffsetBuffer = nulldevptr;
                break;
            case OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES:
                tempBi.lwstomPrimitiveArray.aabbBuffers          = nullptr;
                tempBi.lwstomPrimitiveArray.sbtIndexOffsetBuffer = nulldevptr;
                break;
            case OPTIX_BUILD_INPUT_TYPE_INSTANCES:
                tempBi.instanceArray.instances = nulldevptr;
                break;
            case OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS:
                tempBi.instanceArray.instances = nulldevptr;
                break;
            default:
                break;
        }
        OptixAccelBufferSizes bufferSizes;
        if( const OptixResult result = optixAccelComputeMemoryUsage( m_context, &m_accelOptions, &tempBi, 1, &bufferSizes ) )
            return result;

        return optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer,
                                m_tempBufferSize, m_outputBuffer, m_outputBufferSize, &m_handle, nullptr, 0 );
    }

    OptixResult buildAccelFromInputs()
    {
        OptixAccelBufferSizes bufferSizes;
        if( const OptixResult result = optixAccelComputeMemoryUsage( m_context, &m_accelOptions, &m_buildInputs[0],
                                                                     static_cast<unsigned int>( m_buildInputs.size() ), &bufferSizes ) )
            return result;

        return optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0],
                                static_cast<unsigned int>( m_buildInputs.size() ), m_tempBuffer, m_tempBufferSize,
                                m_outputBuffer, m_outputBufferSize, &m_handle, nullptr, 0 );
    }

    void setupEmptyTriangleInput()
    {
        m_emptyTriangles = createEmptyTriangleInput();
        m_buildInputs[0] = m_emptyTriangles.buildInput;
    }

    OptixDeviceContext           m_context = nullptr;
    StrictMockLogger             m_logger;
    LWstream                     m_stream = nullptr;
    OptixAccelBuildOptions       m_accelOptions{};
    std::vector<OptixBuildInput> m_buildInputs;
    LWdeviceptr                  m_outputBuffer     = nulldevptr;
    LWdeviceptr                  m_tempBuffer       = nulldevptr;
    size_t                       m_outputBufferSize = 0;
    size_t                       m_tempBufferSize   = 0;
    OptixTraversableHandle       m_handle{TRAVERSABLE_HANDLE_INITIAL_VALUE};
    TriangleBuildInput           m_emptyTriangles{};
};

struct TestAccelUpdate : TestAccelBuild
{
    OptixResult buildAccelForUpdateOneInput()
    {
        m_accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        return buildAccelOneInput();
    }
};

struct TestAccelBuildMotionOptions : TestAccelBuild
{
    void SetUp() override
    {
        TestAccelBuild::SetUp();
        setupEmptyTriangleInput();
        m_accelOptions.motionOptions.numKeys = 2;
    }
};

struct TestAccelBuildTriangles : TestAccelBuild
{
    void SetUp() override
    {
        TestAccelBuild::SetUp();
        m_vertexBuffers.resize( 1 );
        m_vertexBuffers[0] = ARBITRARY_DEVICE_PTR;
        setupBasicTriangleInput( 0, &m_sbtFlags );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextGetProperty( m_context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,
                                                             &m_maxNumPrimsPerGAS, sizeof( unsigned int ) ) );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextGetProperty( m_context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS,
                                                             &m_maxSbtRecords, sizeof( unsigned int ) ) );
    }

    void setupBasicTriangleInput( unsigned int index, const unsigned int* sbtFlags )
    {
        m_buildInputs[index].type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        m_buildInputs[index].triangleArray.vertexBuffers = m_vertexBuffers.data();
        m_buildInputs[index].triangleArray.numVertices   = 3;
        m_buildInputs[index].triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        m_buildInputs[index].triangleArray.flags         = sbtFlags;
        m_buildInputs[index].triangleArray.numSbtRecords = 1;
    }

    void TearDown() override
    {
        if( m_vertexBuffer )
        {
            ASSERT_LWDA_SUCCESS( lwdaFree( reinterpret_cast<void*>( m_vertexBuffer ) ) );
        }
        if( m_indexBuffer )
        {
            ASSERT_LWDA_SUCCESS( lwdaFree( reinterpret_cast<void*>( m_indexBuffer ) ) );
        }
        if( m_preTransformBuffer )
        {
            ASSERT_LWDA_SUCCESS( lwdaFree( reinterpret_cast<void*>( m_preTransformBuffer ) ) );
        }
        TestAccelBuild::TearDown();
    }

    void allocateVertexBuffer( OptixVertexFormat fmt, unsigned int numVertices )
    {
        ASSERT_LWDA_SUCCESS( lwdaMalloc( reinterpret_cast<void**>( &m_vertexBuffer ), naturalVertexStride( fmt ) * numVertices ) );
    }

    void fillVertexBuffer3f( float x, float y, float z )
    {
        float v[3] = {x, y, z};
        ASSERT_LWDA_SUCCESS( lwdaMemcpy( reinterpret_cast<void*>( m_vertexBuffer ), &v[0], 3 * sizeof( float ), lwdaMemcpyHostToDevice ) );
    }

    void allocateIndexBuffer( OptixIndicesFormat fmt, unsigned int numTriplets )
    {
        ASSERT_LWDA_SUCCESS( lwdaMalloc( reinterpret_cast<void**>( &m_indexBuffer ), 3 * naturalIndexStride( fmt ) * numTriplets ) );
    }

    void allocateAndFillPreTransformBuffer()
    {
        ASSERT_LWDA_SUCCESS( lwdaMalloc( reinterpret_cast<void**>( &m_preTransformBuffer ), 12 * sizeof( float ) ) );
        float t[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1};
        ASSERT_LWDA_SUCCESS( lwdaMemcpy( reinterpret_cast<void*>( m_preTransformBuffer ), t, 12 * sizeof( float ),
                                         lwdaMemcpyHostToDevice ) );
    }

    void setZeroTriangle()
    {
        m_buildInputs[0].triangleArray.numVertices      = 0;
        m_buildInputs[0].triangleArray.vertexBuffers    = nullptr;
        m_buildInputs[0].triangleArray.numIndexTriplets = 0;
        m_buildInputs[0].triangleArray.indexBuffer      = nulldevptr;
    }

    void setupDegenerateTriangle( OptixVertexFormat fmt )
    {
        m_vertexBuffers.resize( 1 );
        m_buildInputs[0].triangleArray.numVertices   = 3;
        m_buildInputs[0].triangleArray.vertexFormat  = fmt;
        m_buildInputs[0].triangleArray.vertexBuffers = &m_vertexBuffers[0];
        allocateVertexBuffer( fmt, 3 );
        m_vertexBuffers[0] = m_vertexBuffer;
        ASSERT_LWDA_SUCCESS( lwdaMemset( reinterpret_cast<void*>( m_vertexBuffer ), 0, 3 * naturalVertexStride( fmt ) ) );
        m_buildInputs[0].triangleArray.indexBuffer        = nulldevptr;
        m_buildInputs[0].triangleArray.numIndexTriplets   = 0;
        m_buildInputs[0].triangleArray.indexStrideInBytes = 0;
    }


    void setupDegenerateIndexedTriangle( OptixIndicesFormat fmt )
    {
        m_vertexBuffers.resize( 1 );
        m_buildInputs[0].triangleArray.numVertices   = 1;
        m_buildInputs[0].triangleArray.vertexBuffers = &m_vertexBuffers[0];
        allocateVertexBuffer( OPTIX_VERTEX_FORMAT_FLOAT3, 1 );
        fillVertexBuffer3f( 1.0f, 1.0f, 1.0f );
        m_vertexBuffers[0] = m_vertexBuffer;
        allocateIndexBuffer( fmt, 1 );
        ASSERT_LWDA_SUCCESS( lwdaMemset( reinterpret_cast<void*>( m_indexBuffer ), 0, 3 * naturalIndexStride( fmt ) ) );
        m_buildInputs[0].triangleArray.indexBuffer        = m_indexBuffer;
        m_buildInputs[0].triangleArray.numIndexTriplets   = 1;
        m_buildInputs[0].triangleArray.indexFormat        = fmt;
        m_buildInputs[0].triangleArray.indexStrideInBytes = naturalIndexStride( fmt );
    }

    unsigned int             m_sbtFlags = 0U;
    std::vector<LWdeviceptr> m_vertexBuffers;
    LWdeviceptr              m_vertexBuffer       = nulldevptr;
    LWdeviceptr              m_indexBuffer        = nulldevptr;
    LWdeviceptr              m_preTransformBuffer = nulldevptr;
    unsigned int             m_maxNumPrimsPerGAS  = 0U;
    unsigned int             m_maxSbtRecords      = 0U;
};

struct TestAccelBuildLwstomPrimitives : TestAccelBuild
{
    void SetUp() override
    {
        TestAccelBuild::SetUp();
        m_buildInputs[0].type                               = OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES;
        m_buildInputs[0].lwstomPrimitiveArray.numPrimitives = 1;
        m_aabbBuffers.resize( m_buildInputs[0].lwstomPrimitiveArray.numPrimitives, ARBITRARY_DEVICE_PTR );
        m_buildInputs[0].lwstomPrimitiveArray.aabbBuffers   = &m_aabbBuffers[0];
        m_buildInputs[0].lwstomPrimitiveArray.numSbtRecords = 1;
        m_sbtFlags.resize( m_buildInputs[0].lwstomPrimitiveArray.numSbtRecords, 0U );
        m_buildInputs[0].lwstomPrimitiveArray.flags = &m_sbtFlags[0];
    }

    void TearDown() override
    {
        if( m_aabbBuffer )
        {
            ASSERT_LWDA_SUCCESS( lwdaFree( reinterpret_cast<void*>( m_aabbBuffer ) ) );
        }
        TestAccelBuild::TearDown();
    }

    void setZeroPrim()
    {
        m_buildInputs[0].lwstomPrimitiveArray.numPrimitives = 0;
        m_buildInputs[0].lwstomPrimitiveArray.aabbBuffers   = nullptr;
    }

    void allocateAabbBuffer( unsigned int numAabbs )
    {
        ASSERT_LWDA_SUCCESS( lwdaMalloc( reinterpret_cast<void**>( &m_aabbBuffer ), 6 * sizeof( float ) * numAabbs ) );
    }

    void fillAabbBuffer( float minXYZ, float maxXYZ )
    {
        float aabb[6] = {minXYZ, minXYZ, minXYZ, maxXYZ, maxXYZ, maxXYZ};
        ASSERT_LWDA_SUCCESS( lwdaMemcpy( reinterpret_cast<void*>( m_aabbBuffer ), &aabb[0], 6 * sizeof( float ), lwdaMemcpyHostToDevice ) );
    }

    std::vector<LWdeviceptr>  m_aabbBuffers;
    LWdeviceptr               m_aabbBuffer = nulldevptr;
    std::vector<unsigned int> m_sbtFlags;
};

struct TestAccelBuildInstances : TestAccelBuild
{
    void SetUp() override
    {
        TestAccelBuild::SetUp();
        m_buildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        ASSERT_OPTIX_SUCCESS( optixDeviceContextGetProperty( m_context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS,
                                                             &m_maxInstancesPerIAS, sizeof( unsigned int ) ) );
    }

    unsigned int m_maxInstancesPerIAS = 0U;
};

struct TestAccelBuildInstancePointers : TestAccelBuild
{
    void SetUp() override
    {
        TestAccelBuild::SetUp();
        m_buildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS;
    }
};

struct TestAccelDevicePointers : TestAccelBuild
{
    void verify( char* ptr, size_t size, OptixResult expectedResult )
    {
        if( expectedResult != OPTIX_SUCCESS )
            EXPECT_LOG_MSG( R"msg("outputBuffer" does not point to memory allocated with lwdaMalloc() or lwMemAlloc())msg" );

        m_outputBufferSize = size;

        const TriangleBuildInput emptyTriangles = createEmptyTriangleInput();

        // First check that our output buffer is actually large enough.
        OptixAccelBufferSizes bufferSizes;
        ASSERT_OPTIX_SUCCESS( optixAccelComputeMemoryUsage( m_context, &m_accelOptions, &emptyTriangles.buildInput, 1, &bufferSizes ) );
        ASSERT_GE( m_outputBufferSize, bufferSizes.outputSizeInBytes );

        ASSERT_EQ( expectedResult,
                   optixAccelBuild( m_context, nullptr, &m_accelOptions, &emptyTriangles.buildInput, 1, m_tempBuffer,
                                    TEMP_BUFFER_SIZE, (LWdeviceptr)ptr, m_outputBufferSize, &m_handle, nullptr, 0 ) );
    }

    static const int offsets[4];

    static const size_t bufferSize = 3000 + 384;
    static const size_t alignment  = 512;

    char* roundUp( char* ptr, size_t alignment )
    {
        return (char*)( ( ( size_t )( ptr ) + alignment - 1 ) & ~( alignment - 1 ) );
    }

    void test_lwMemHostAlloc( int flags )
    {
        void* ptr;
        ASSERT_LW_SUCCESS( lwMemHostAlloc( &ptr, bufferSize, flags ) );

        for( auto offset : offsets )
            verify( ( (char*)ptr ) + offset, bufferSize - offset, OPTIX_ERROR_ILWALID_VALUE );

        ASSERT_LW_SUCCESS( lwMemFreeHost( ptr ) );
    }

    void test_lwMemHostRegister( int flags )
    {
        std::unique_ptr<char[]> buffer( new char[bufferSize + alignment] );
        char*                   aligned = roundUp( buffer.get(), alignment );

        ASSERT_LW_SUCCESS( lwMemHostRegister( aligned, bufferSize, flags ) );

        for( auto offset : offsets )
            verify( aligned + offset, bufferSize - offset, OPTIX_ERROR_ILWALID_VALUE );

        ASSERT_LW_SUCCESS( lwMemHostUnregister( aligned ) );
    }
};

const int TestAccelDevicePointers::offsets[4] = {0, 128, 256, 384};

}  // namespace

// Return a LWdeviceptr that is not aligned to the given alignment
inline LWdeviceptr misalignDevicePtr( LWdeviceptr p, unsigned int alignment )
{
    return p + alignment - 1U;
}

TEST_F( TestAccelBuild, with_null_context )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT,
               optixAccelBuild( nullptr, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer,
                                TEMP_BUFFER_SIZE, m_outputBuffer, OUTPUT_BUFFER_SIZE, &m_handle, nullptr, 0 ) );
}

TEST_F( TestAccelBuild, null_stream_ok )
{
    const TriangleBuildInput emptyTriangles = createEmptyTriangleInput();

    ASSERT_OPTIX_SUCCESS( optixAccelBuild( m_context, nullptr, &m_accelOptions, &emptyTriangles.buildInput, 1, m_tempBuffer,
                                           TEMP_BUFFER_SIZE, m_outputBuffer, OUTPUT_BUFFER_SIZE, &m_handle, nullptr, 0 ) );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuild, with_null_accel_options )
{
    EXPECT_LOG_MSG( "accelOptions is null" );

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, nullptr, &m_buildInputs[0], 1, m_tempBuffer, TEMP_BUFFER_SIZE,
                                                 m_outputBuffer, OUTPUT_BUFFER_SIZE, &m_handle, nullptr, 0 ) );
}

TEST_F( TestAccelBuild, with_null_build_inputs )
{
    EXPECT_LOG_MSG( "buildInputs is null" );

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, nullptr, 1, m_tempBuffer, TEMP_BUFFER_SIZE,
                                                 m_outputBuffer, OUTPUT_BUFFER_SIZE, &m_handle, nullptr, 0 ) );
}

TEST_F( TestAccelBuild, with_zero_build_inputs )
{
    EXPECT_LOG_MSG( "numBuildInputs is 0" );

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 0, m_tempBuffer,
                                                 TEMP_BUFFER_SIZE, m_outputBuffer, OUTPUT_BUFFER_SIZE, &m_handle, nullptr, 0 ) );
}

TEST_F( TestAccelBuild, with_null_temp_buffer_nonzero_temp_buffer_size )
{
    setupEmptyTriangleInput();
    EXPECT_LOG_MSG( R"msg("tempBufferSizeInBytes" is non-zero but "tempBuffer" is null)msg" );

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, nulldevptr,
                                                 TEMP_BUFFER_SIZE, m_outputBuffer, OUTPUT_BUFFER_SIZE, &m_handle, nullptr, 0 ) );
}

TEST_F( TestAccelBuild, with_zero_temp_buffer_size )
{
    setupEmptyTriangleInput();
    EXPECT_LOG_MSG( R"msg("tempBuffer" is non-null but "tempBufferSizeInBytes" is 0)msg" );

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer,
                                                 0, m_outputBuffer, OUTPUT_BUFFER_SIZE, &m_handle, nullptr, 0 ) );
}

TEST_F( TestAccelBuild, with_misaligned_temp_buffer )
{
    setupEmptyTriangleInput();
    EXPECT_LOG_MSG( R"msg("tempBuffer" is not a multiple of OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT)msg" );

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1,
                                                 misalignDevicePtr( m_tempBuffer, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT ),
                                                 TEMP_BUFFER_SIZE, m_outputBuffer, OUTPUT_BUFFER_SIZE, &m_handle, nullptr, 0 ) );
}

TEST_F( TestAccelBuild, with_null_output_buffer )
{
    setupEmptyTriangleInput();
    EXPECT_LOG_MSG( "outputBuffer is 0" );

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer,
                                                 TEMP_BUFFER_SIZE, nulldevptr, OUTPUT_BUFFER_SIZE, &m_handle, nullptr, 0 ) );
}

TEST_F( TestAccelBuild, with_zero_output_buffer_size )
{
    setupEmptyTriangleInput();
    EXPECT_LOG_MSG( "outputBufferSizeInBytes is 0" );

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer,
                                                 TEMP_BUFFER_SIZE, m_outputBuffer, 0, &m_handle, nullptr, 0 ) );
}

TEST_F( TestAccelBuild, with_misaligned_output_buffer )
{
    setupEmptyTriangleInput();
    EXPECT_LOG_MSG( R"msg("outputBuffer" is not a multiple of OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT)msg" );

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer, TEMP_BUFFER_SIZE,
                                                 misalignDevicePtr( m_outputBuffer, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT ),
                                                 OUTPUT_BUFFER_SIZE, &m_handle, nullptr, 0 ) );
}

TEST_F( TestAccelBuild, with_null_output_handle )
{
    setupEmptyTriangleInput();
    EXPECT_LOG_MSG( R"msg("accelOptions.operation" is OPTIX_BUILD_OPERATION_BUILD but "outputHandle" is null)msg" );

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer,
                                                 TEMP_BUFFER_SIZE, m_outputBuffer, OUTPUT_BUFFER_SIZE, nullptr, nullptr, 0 ) );
}

TEST_F( TestAccelBuild, with_zero_emitted_properties_size )
{
    EXPECT_LOG_MSG( R"msg("emittedProperties" is non-null but "numEmittedProperties" is 0)msg" );
    OptixAccelEmitDesc emittedProperties{};

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer, TEMP_BUFFER_SIZE,
                                                 m_outputBuffer, OUTPUT_BUFFER_SIZE, &m_handle, &emittedProperties, 0 ) );
}

TEST_F( TestAccelBuild, with_null_emitted_properties_nonzero_emitted_properties_size )
{
    EXPECT_LOG_MSG( R"msg("numEmittedProperties" is non-zero but "emittedProperties" is null)msg" );

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer,
                                                 TEMP_BUFFER_SIZE, m_outputBuffer, OUTPUT_BUFFER_SIZE, &m_handle, nullptr, 1 ) );
}

TEST_F( TestAccelBuild, request_compaction_size_without_requesting_compaction )
{
    EXPECT_LOG_MSG( R"msg(Invalid value (8577) for "emittedProperties[0].type": Querying compacted size, but build flag "OPTIX_BUILD_FLAG_ALLOW_COMPACTION" is not set.)msg" );
    setupEmptyTriangleInput();
    OptixAccelEmitDesc emittedProperties{ARBITRARY_DEVICE_PTR, OPTIX_PROPERTY_TYPE_COMPACTED_SIZE};

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer, TEMP_BUFFER_SIZE,
                                                 m_outputBuffer, OUTPUT_BUFFER_SIZE, &m_handle, &emittedProperties, 1 ) );
}

TEST_F( TestAccelBuild, request_compaction_size_null_result_buffer )
{
    EXPECT_LOG_MSG( R"msg("emittedProperties[0].result" is zero)msg" );
    m_accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    setupEmptyTriangleInput();
    OptixAccelEmitDesc emittedProperties{nulldevptr, OPTIX_PROPERTY_TYPE_COMPACTED_SIZE};

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer, TEMP_BUFFER_SIZE,
                                                 m_outputBuffer, OUTPUT_BUFFER_SIZE, &m_handle, &emittedProperties, 1 ) );
}

TEST_F( TestAccelBuild, request_compaction_size_misaligned_result_buffer )
{
    EXPECT_LOG_MSG( R"msg("emittedProperties[0].result" is not a multiple of 8)msg" );
    m_accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    setupEmptyTriangleInput();
    OptixAccelEmitDesc emittedProperties{misalignDevicePtr( ARBITRARY_DEVICE_PTR, DEVICE_PTR_ALIGNMENT ),
                                         OPTIX_PROPERTY_TYPE_COMPACTED_SIZE};

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer, TEMP_BUFFER_SIZE,
                                                 m_outputBuffer, OUTPUT_BUFFER_SIZE, &m_handle, &emittedProperties, 1 ) );
}

TEST_F( TestAccelBuild, request_aabb_misaligned_result_buffer )
{
    EXPECT_LOG_MSG( R"msg("emittedProperties[0].result" is not a multiple of OPTIX_AABB_BUFFER_BYTE_ALIGNMENT)msg" );
    setupEmptyTriangleInput();
    OptixAccelEmitDesc emittedProperties{misalignDevicePtr( ARBITRARY_DEVICE_PTR, OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ),
                                         OPTIX_PROPERTY_TYPE_AABBS};

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer, TEMP_BUFFER_SIZE,
                                                 m_outputBuffer, OUTPUT_BUFFER_SIZE, &m_handle, &emittedProperties, 1 ) );
}

TEST_F( TestAccelBuild, with_lwdaMallocManaged_allocated_result_buffer )
{
    EXPECT_LOG_MSG( R"msg("outputBuffer" does not point to memory allocated with lwdaMalloc() or lwMemAlloc())msg" );
    setupEmptyTriangleInput();
    LWdeviceptr managed_ptr;
    ASSERT_LWDA_SUCCESS( lwdaMallocManaged( reinterpret_cast<void**>( &managed_ptr ), OUTPUT_BUFFER_SIZE ) );

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer,
                                                 TEMP_BUFFER_SIZE, managed_ptr, OUTPUT_BUFFER_SIZE, &m_handle, nullptr, 0 ) );
    ASSERT_LWDA_SUCCESS( lwdaFree( reinterpret_cast<void*>( managed_ptr ) ) );
}

TEST_F( TestAccelBuild, ilwalid_emitted_property_type )
{
    EXPECT_LOG_MSG( R"msg(Invalid value (0) for "emittedProperties[0].type")msg" );
    m_accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    setupEmptyTriangleInput();
    OptixAccelEmitDesc emittedProperties{ARBITRARY_DEVICE_PTR, static_cast<OptixAccelPropertyType>( 0 )};

    ASSERT_OPTIX_ILWALID_VALUE( optixAccelBuild( m_context, m_stream, &m_accelOptions, &m_buildInputs[0], 1, m_tempBuffer, TEMP_BUFFER_SIZE,
                                                 m_outputBuffer, OUTPUT_BUFFER_SIZE, &m_handle, &emittedProperties, 1 ) );
}

TEST_F( TestAccelBuild, ilwalid_option_flags )
{
    EXPECT_LOG_MSG( R"msg("accelOptions.buildFlags" contains invalid flags)msg" );
    setupEmptyTriangleInput();
    m_accelOptions.buildFlags = ~0U;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuild, ilwalid_operation )
{
    EXPECT_LOG_MSG( R"msg("accelOptions.operation" is not a valid build operation)msg" );
    setupEmptyTriangleInput();
    m_accelOptions.operation = static_cast<OptixBuildOperation>( ~0U );

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuild, mixed_triangle_lwstom_primitive_build_types )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[1].type" != "buildInputs[0].type". All build inputs for geometry acceleration structures must have the same type)msg" );
    m_buildInputs.resize( 2 );
    m_emptyTriangles      = createEmptyTriangleInput();
    m_buildInputs[0]      = m_emptyTriangles.buildInput;
    m_buildInputs[1].type = OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelFromInputs() );
}

TEST_F( TestAccelBuild, ilwalid_build_input_type )
{
    EXPECT_LOG_MSG( R"msg(Invalid build type (0x0) for "buildInputs[0].type")msg" );
    m_buildInputs[0].type = ILWALID_BUILD_INPUT_TYPE;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}


TEST_F( TestAccelDevicePointers, arbitraryStackPointer )
{
    int a;

    // Passing "bufferSize" here is incorrect, but we want the test to fail because of the nature of the pointer, not
    // because the buffer is too small.
    verify( (char*)&a, bufferSize, OPTIX_ERROR_ILWALID_VALUE );
}

TEST_F( TestAccelDevicePointers, arbitraryHeapPointer )
{
    int* ptr = new int;

    // Passing "bufferSize" here is incorrect, but we want the test to fail because of the nature of the pointer, not
    // because the buffer is too small.
    verify( (char*)ptr, bufferSize, OPTIX_ERROR_ILWALID_VALUE );

    delete ptr;
}

TEST_F( TestAccelDevicePointers, lwdaMalloc )
{
    LWdeviceptr ptr;
    LWDA_CHECK( lwdaMalloc( (void**)&ptr, bufferSize ) );

    for( auto offset : offsets )
        verify( ( (char*)ptr ) + offset, bufferSize - offset, OPTIX_SUCCESS );

    LWDA_CHECK( lwdaFree( (void*)ptr ) );
}

TEST_F( TestAccelDevicePointers, lwMemAlloc )
{
    LWdeviceptr ptr;
    LWDA_CHECK( lwMemAlloc( &ptr, bufferSize ) );

    for( auto offset : offsets )
        verify( ( (char*)ptr ) + offset, bufferSize - offset, OPTIX_SUCCESS );

    LWDA_CHECK( lwMemFree( ptr ) );
}

TEST_F( TestAccelDevicePointers, lwdaMallocManaged )
{
    LWdeviceptr ptr;
    LWDA_CHECK( lwdaMallocManaged( (void**)&ptr, bufferSize ) );

    for( auto offset : offsets )
        verify( ( (char*)ptr ) + offset, bufferSize - offset, OPTIX_ERROR_ILWALID_VALUE );

    LWDA_CHECK( lwdaFree( (void*)ptr ) );
}

TEST_F( TestAccelDevicePointers, lwMemHostRegister_noFlags )
{
    test_lwMemHostRegister( 0 );
}

TEST_F( TestAccelDevicePointers, lwMemHostRegister_Portable )
{
    test_lwMemHostRegister( LW_MEMHOSTREGISTER_PORTABLE );
}

TEST_F( TestAccelDevicePointers, lwMemHostRegister_DeviceMap )
{
    test_lwMemHostRegister( LW_MEMHOSTREGISTER_DEVICEMAP );
}

TEST_F( TestAccelDevicePointers, lwMemHostRegister_Portable_DeviceMap )
{
    test_lwMemHostRegister( LW_MEMHOSTREGISTER_PORTABLE | LW_MEMHOSTREGISTER_DEVICEMAP );
}

// No tests with LW_MEMHOSTREGISTER_IOMEMORY since it is not clear how to make lwMemHostRegister() succeed
// with this flag.

TEST_F( TestAccelDevicePointers, lwMemHostAlloc_noFlags )
{
    test_lwMemHostAlloc( 0 );
}

TEST_F( TestAccelDevicePointers, lwMemHostAlloc_Portable )
{
    test_lwMemHostAlloc( LW_MEMHOSTALLOC_PORTABLE );
}

TEST_F( TestAccelDevicePointers, lwMemHostAlloc_DeviceMap )
{
    test_lwMemHostAlloc( LW_MEMHOSTALLOC_DEVICEMAP );
}

TEST_F( TestAccelDevicePointers, lwMemHostAlloc_Portable_DeviceMap )
{
    test_lwMemHostAlloc( LW_MEMHOSTALLOC_PORTABLE | LW_MEMHOSTALLOC_DEVICEMAP );
}

TEST_F( TestAccelDevicePointers, lwMemHostAlloc_WriteCombined )
{
    test_lwMemHostAlloc( LW_MEMHOSTALLOC_WRITECOMBINED );
}

TEST_F( TestAccelDevicePointers, lwMemHostAlloc_Portable_WriteCombined )
{
    test_lwMemHostAlloc( LW_MEMHOSTALLOC_PORTABLE | LW_MEMHOSTALLOC_WRITECOMBINED );
}

TEST_F( TestAccelDevicePointers, lwMemHostAlloc_DeviceMap_WriteCombined )
{
    test_lwMemHostAlloc( LW_MEMHOSTALLOC_DEVICEMAP | LW_MEMHOSTALLOC_WRITECOMBINED );
}

TEST_F( TestAccelDevicePointers, lwMemHostAlloc_Portable_DeviceMap_WriteCombined )
{
    test_lwMemHostAlloc( LW_MEMHOSTALLOC_PORTABLE | LW_MEMHOSTALLOC_DEVICEMAP | LW_MEMHOSTALLOC_WRITECOMBINED );
}

TEST_F( TestAccelBuildMotionOptions, flags_ignored_when_no_motion )
{
    m_accelOptions.motionOptions.numKeys = 1;
    m_accelOptions.motionOptions.flags   = ~static_cast<unsigned short>( 0U );

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildMotionOptions, ilwalid_flags )
{
    EXPECT_LOG_MSG( R"msg("accelOptions.motionOptions.flags" contains invalid flags)msg" );
    m_accelOptions.motionOptions.flags = ~static_cast<unsigned short>( 0U );

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildMotionOptions, begin_time_after_end_time )
{
    EXPECT_LOG_MSG( R"msg("accelOptions.motionOptions.timeBegin" is larger than "accelOptions.motionOptions.timeEnd")msg" );
    m_accelOptions.motionOptions.timeBegin = 10.f;
    m_accelOptions.motionOptions.timeEnd   = 5.f;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildMotionOptions, begin_time_infinity )
{
    EXPECT_LOG_MSG( R"msg("accelOptions.motionOptions.timeBegin" is infinity)msg" );
    m_accelOptions.motionOptions.timeBegin = INFINITY;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildMotionOptions, end_time_infinity )
{
    EXPECT_LOG_MSG( R"msg("accelOptions.motionOptions.timeEnd" is infinity)msg" );
    m_accelOptions.motionOptions.timeEnd = INFINITY;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildMotionOptions, begin_time_nan )
{
    EXPECT_LOG_MSG( R"msg("accelOptions.motionOptions.timeBegin" is NaN)msg" );
    m_accelOptions.motionOptions.timeBegin = NAN;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildMotionOptions, end_time_nan )
{
    EXPECT_LOG_MSG( R"msg("accelOptions.motionOptions.timeEnd" is NaN)msg" );
    m_accelOptions.motionOptions.timeEnd = NAN;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildMotionOptions, begin_time_ignored_when_no_motion )
{
    m_accelOptions.motionOptions.numKeys   = 1;
    m_accelOptions.motionOptions.timeBegin = INFINITY;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildMotionOptions, end_time_ignored_when_no_motion )
{
    m_accelOptions.motionOptions.numKeys = 1;
    m_accelOptions.motionOptions.timeEnd = NAN;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, nonzero_num_vertices_null_vertex_buffers )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "numVertices" is non-zero, but "vertexBuffers" is null)msg" );
    m_buildInputs[0].triangleArray.numVertices   = 3;
    m_buildInputs[0].triangleArray.vertexBuffers = nullptr;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, nonzero_num_vertices_null_vertex_buffer_device_ptr )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "numVertices" is non-zero, but "vertexBuffers[0]" (vertex buffer for motion key 0) is null)msg" );
    m_buildInputs[0].triangleArray.numVertices   = 3;
    LWdeviceptr vb                               = nulldevptr;
    m_buildInputs[0].triangleArray.vertexBuffers = &vb;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, no_vertex_format )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "numVertices" is non-zero, but "vertexFormat" is OPTIX_VERTEX_FORMAT_NONE)msg" );
    m_buildInputs[0].triangleArray.vertexFormat = static_cast<OptixVertexFormat>( 0 );

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_float3_zero_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_FLOAT3 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = 0;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_float2_zero_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_FLOAT2 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = 0;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_half3_zero_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_HALF3 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = 0;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_half2_zero_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_HALF2 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = 0;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_snorm16_3_zero_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_SNORM16_3 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = 0;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_snorm16_2_zero_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_SNORM16_2 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = 0;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_float3_too_small_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (11) is smaller than 3 * sizeof( float ) for vertex format OPTIX_VERTEX_FORMAT_FLOAT3.)msg" );
    m_buildInputs[0].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_FLOAT3>::stride - 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (11) is smaller than 3 * sizeof( float ) for vertex format OPTIX_VERTEX_FORMAT_FLOAT3.)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_float2_too_small_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (7) is smaller than 2 * sizeof( float ) for vertex format OPTIX_VERTEX_FORMAT_FLOAT2.)msg" );
    m_buildInputs[0].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT2;
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_FLOAT2>::stride - 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (7) is smaller than 2 * sizeof( float ) for vertex format OPTIX_VERTEX_FORMAT_FLOAT2.)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_half3_too_small_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (5) is smaller than 3 * sizeof( float ) / 2 for vertex format OPTIX_VERTEX_FORMAT_HALF3.)msg" );
    m_buildInputs[0].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_HALF3;
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_HALF3>::stride - 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (5) is smaller than 3 * sizeof( float ) / 2 for vertex format OPTIX_VERTEX_FORMAT_HALF3.)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_half2_too_small_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (3) is smaller than 2 * sizeof( float ) / 2 for vertex format OPTIX_VERTEX_FORMAT_HALF2.)msg" );
    m_buildInputs[0].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_HALF2;
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_HALF2>::stride - 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (3) is smaller than 2 * sizeof( float ) / 2 for vertex format OPTIX_VERTEX_FORMAT_HALF2.)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_snorm16_3_too_small_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (5) is smaller than 3 * sizeof( short ) for vertex format OPTIX_VERTEX_FORMAT_SNORM16_3.)msg" );
    m_buildInputs[0].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_SNORM16_3;
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_SNORM16_3>::stride - 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (5) is smaller than 3 * sizeof( short ) for vertex format OPTIX_VERTEX_FORMAT_SNORM16_3.)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_snorm16_2_too_small_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (3) is smaller than 2 * sizeof( short ) for vertex format OPTIX_VERTEX_FORMAT_SNORM16_2.)msg" );
    m_buildInputs[0].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_SNORM16_2;
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_SNORM16_2>::stride - 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (3) is smaller than 2 * sizeof( short ) for vertex format OPTIX_VERTEX_FORMAT_SNORM16_2.)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_float3_exact_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_FLOAT3 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_FLOAT3>::stride;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_float2_exact_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_FLOAT2 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_FLOAT2>::stride;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_half3_exact_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_HALF3 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_HALF3>::stride;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_half2_exact_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_HALF2 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_HALF2>::stride;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_snorm16_3_exact_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_SNORM16_3 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_SNORM16_3>::stride;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_snorm16_2_exact_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_SNORM16_2 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_SNORM16_2>::stride;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_float3_excess_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_FLOAT3 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = 2 * NaturalVertexStride<OPTIX_VERTEX_FORMAT_FLOAT3>::stride;
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_float2_excess_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_FLOAT2 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = 2 * NaturalVertexStride<OPTIX_VERTEX_FORMAT_FLOAT2>::stride;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_half3_excess_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_HALF3 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = 2 * NaturalVertexStride<OPTIX_VERTEX_FORMAT_HALF3>::stride;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_half2_excess_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_HALF2 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = 2 * NaturalVertexStride<OPTIX_VERTEX_FORMAT_HALF2>::stride;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_snorm16_3_excess_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_SNORM16_3 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = 2 * NaturalVertexStride<OPTIX_VERTEX_FORMAT_SNORM16_3>::stride;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_snorm16_2_excess_stride )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_SNORM16_2 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = 2 * NaturalVertexStride<OPTIX_VERTEX_FORMAT_SNORM16_2>::stride;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_OPTIX_SUCCESS( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, vertex_format_float3_misaligned_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (13) is not a multiple of 4 for vertex format OPTIX_VERTEX_FORMAT_FLOAT3.)msg" );
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_FLOAT3 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_FLOAT3>::stride + 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (13) is not a multiple of 4 for vertex format OPTIX_VERTEX_FORMAT_FLOAT3.)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_float2_misaligned_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (9) is not a multiple of 4 for vertex format OPTIX_VERTEX_FORMAT_FLOAT2.)msg" );
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_FLOAT2 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_FLOAT2>::stride + 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_half3_misaligned_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (7) is not a multiple of 2 for vertex format OPTIX_VERTEX_FORMAT_HALF3.)msg" );
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_HALF3 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_HALF3>::stride + 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_half2_misaligned_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (5) is not a multiple of 2 for vertex format OPTIX_VERTEX_FORMAT_HALF2.)msg" );
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_HALF2 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_HALF2>::stride + 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_snorm16_3_misaligned_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (7) is not a multiple of 2 for vertex format OPTIX_VERTEX_FORMAT_SNORM16_3.)msg" );
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_SNORM16_3 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_SNORM16_3>::stride + 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_snorm16_2_misaligned_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexStrideInBytes" (5) is not a multiple of 2 for vertex format OPTIX_VERTEX_FORMAT_SNORM16_2.)msg" );
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_SNORM16_2 );
    m_buildInputs[0].triangleArray.vertexStrideInBytes = NaturalVertexStride<OPTIX_VERTEX_FORMAT_SNORM16_2>::stride + 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_float3_misaligned_vertex_buffer )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexBuffers[0]" (vertex buffer for motion key 0) is not 4-byte aligned)msg" );
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_FLOAT3 );
    m_vertexBuffers[0] += 2;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexBuffers[0]" (vertex buffer for motion key 0) is not 4-byte aligned)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInputEmptyDevPtrsForComputeMemory() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_float2_misaligned_vertex_buffer )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexBuffers[0]" (vertex buffer for motion key 0) is not 4-byte aligned)msg" );
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_FLOAT2 );
    m_vertexBuffers[0] += 2;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_half3_misaligned_vertex_buffer )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexBuffers[0]" (vertex buffer for motion key 0) is not 2-byte aligned)msg" );
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_HALF3 );
    m_vertexBuffers[0] += 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_half2_misaligned_vertex_buffer )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexBuffers[0]" (vertex buffer for motion key 0) is not 2-byte aligned)msg" );
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_HALF2 );
    m_vertexBuffers[0] += 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_snorm16_3_misaligned_vertex_buffer )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexBuffers[0]" (vertex buffer for motion key 0) is not 2-byte aligned)msg" );
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_SNORM16_3 );
    m_vertexBuffers[0] += 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, vertex_format_snorm16_2_misaligned_vertex_buffer )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.vertexBuffers[0]" (vertex buffer for motion key 0) is not 2-byte aligned)msg" );
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_SNORM16_2 );
    m_vertexBuffers[0] += 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, null_index_buffer_nonzero_num_indices )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "numIndexTriplets" is non-zero, but "indexBuffer" is null)msg" );
    m_buildInputs[0].triangleArray.vertexBuffers    = &ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.indexBuffer      = 0;
    m_buildInputs[0].triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    m_buildInputs[0].triangleArray.numIndexTriplets = 1;

    ASSERT_OPTIX_SUCCESS( memComputeAccelOneInput() );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, nonnull_index_buffer_zero_num_indices )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "numIndexTriplets" is zero, but "indexBuffer" is non-null)msg" );
    m_buildInputs[0].triangleArray.indexBuffer      = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    m_buildInputs[0].triangleArray.numIndexTriplets = 0;

    ASSERT_OPTIX_ILWALID_VALUE( memComputeAccelOneInput() );
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "numIndexTriplets" is zero, but "indexBuffer" is non-null)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, bad_index_format )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "numIndexTriplets" is non-zero, but "indexFormat" is OPTIX_INDICES_FORMAT_NONE)msg" );
    m_buildInputs[0].triangleArray.indexBuffer      = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.numIndexTriplets = 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, index_format_short3_zero_stride )
{
    setupDegenerateIndexedTriangle( OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 );
    m_buildInputs[0].triangleArray.indexStrideInBytes = 0;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, index_format_int3_zero_stride )
{
    setupDegenerateIndexedTriangle( OPTIX_INDICES_FORMAT_UNSIGNED_INT3 );
    m_buildInputs[0].triangleArray.indexStrideInBytes = 0;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, index_format_short3_too_small_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.indexStrideInBytes" (5) is smaller than 3 * sizeof( short ) for index format OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3.)msg" );
    setupDegenerateIndexedTriangle( OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 );
    m_buildInputs[0].triangleArray.indexStrideInBytes = NaturalIndexStride<OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3>::stride - 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, index_format_int3_too_small_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.indexStrideInBytes" (11) is smaller than 3 * sizeof( int ) for index format OPTIX_INDICES_FORMAT_UNSIGNED_INT3.)msg" );
    setupDegenerateIndexedTriangle( OPTIX_INDICES_FORMAT_UNSIGNED_INT3 );
    m_buildInputs[0].triangleArray.indexStrideInBytes = NaturalIndexStride<OPTIX_INDICES_FORMAT_UNSIGNED_INT3>::stride - 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, index_format_short3_exact_stride )
{
    setupDegenerateIndexedTriangle( OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 );
    m_buildInputs[0].triangleArray.indexStrideInBytes = NaturalIndexStride<OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3>::stride;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, index_format_int3_exact_stride )
{
    setupDegenerateIndexedTriangle( OPTIX_INDICES_FORMAT_UNSIGNED_INT3 );
    m_buildInputs[0].triangleArray.indexStrideInBytes = NaturalIndexStride<OPTIX_INDICES_FORMAT_UNSIGNED_INT3>::stride;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, index_format_short3_excess_stride )
{
    setupDegenerateIndexedTriangle( OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 );
    m_buildInputs[0].triangleArray.indexStrideInBytes = 2 * NaturalIndexStride<OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3>::stride;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, index_format_int3_excess_stride )
{
    setupDegenerateIndexedTriangle( OPTIX_INDICES_FORMAT_UNSIGNED_INT3 );
    m_buildInputs[0].triangleArray.indexStrideInBytes = 2 * NaturalIndexStride<OPTIX_INDICES_FORMAT_UNSIGNED_INT3>::stride;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, index_format_short3_misaligned_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.indexStrideInBytes" (7) is not a multiple of 2 for index format OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3.)msg" );
    setupDegenerateIndexedTriangle( OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 );
    m_buildInputs[0].triangleArray.indexStrideInBytes = NaturalIndexStride<OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3>::stride + 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, index_format_int3_misaligned_stride )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.indexStrideInBytes" (13) is not a multiple of 4 for index format OPTIX_INDICES_FORMAT_UNSIGNED_INT3.)msg" );
    setupDegenerateIndexedTriangle( OPTIX_INDICES_FORMAT_UNSIGNED_INT3 );
    m_buildInputs[0].triangleArray.indexStrideInBytes = NaturalIndexStride<OPTIX_INDICES_FORMAT_UNSIGNED_INT3>::stride + 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, index_format_short3_misaligned_buffer )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "indexFormat" is OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3, but "indexBuffer" is not 2-byte aligned)msg" );
    setupDegenerateIndexedTriangle( OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 );
    m_buildInputs[0].triangleArray.indexBuffer += 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, index_format_int3_misaligned_buffer )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "indexFormat" is OPTIX_INDICES_FORMAT_UNSIGNED_INT3, but "indexBuffer" is not 4-byte aligned)msg" );
    setupDegenerateIndexedTriangle( OPTIX_INDICES_FORMAT_UNSIGNED_INT3 );
    m_buildInputs[0].triangleArray.indexBuffer += 2;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, non_null_transform_format_null_pre_transform )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "transformFormat" is OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12, but "preTransform" is null)msg" );
    m_buildInputs[0].triangleArray.vertexBuffers   = &ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;

    ASSERT_OPTIX_SUCCESS( memComputeAccelOneInput() );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, null_transform_format_non_null_pre_transform )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "transformFormat" is OPTIX_TRANSFORM_FORMAT_NONE, but "preTransform" is not null)msg" );
    m_buildInputs[0].triangleArray.preTransform    = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;

    ASSERT_OPTIX_ILWALID_VALUE( memComputeAccelOneInput() );
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "transformFormat" is OPTIX_TRANSFORM_FORMAT_NONE, but "preTransform" is not null)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, misaligned_pre_transform )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "preTransform" is not a multiple of OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT)msg" );
    m_buildInputs[0].triangleArray.vertexBuffers   = &ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
    m_buildInputs[0].triangleArray.preTransform = misalignDevicePtr( ARBITRARY_DEVICE_PTR, OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT );

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, pre_transform )
{
    setupDegenerateTriangle( OPTIX_VERTEX_FORMAT_FLOAT3 );
    m_buildInputs[0].triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
    allocateAndFillPreTransformBuffer();
    m_buildInputs[0].triangleArray.preTransform = m_preTransformBuffer;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );
}

TEST_F( TestAccelBuildTriangles, null_sbt_flags )
{
    EXPECT_LOG_MSG( R"msg(Invalid value (0) for "buildInputs[0].triangleArray.flags")msg" );
    m_buildInputs[0].triangleArray.flags = nullptr;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, zero_sbts )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.numSbtRecords" is zero)msg" );
    m_buildInputs[0].triangleArray.numSbtRecords = 0;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, null_sbt_index_offset_buffer_multiple_sbt_records )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "numSbtRecords" is larger than one, but "sbtIndexOffsetBuffer" is null)msg" );
    const unsigned int              numSbtRecords = 2;
    const std::vector<unsigned int> sbtFlags( numSbtRecords, 0U );
    setupBasicTriangleInput( 0, &sbtFlags[0] );
    m_buildInputs[0].triangleArray.vertexBuffers             = &ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.sbtIndexOffsetBuffer      = nulldevptr;
    m_buildInputs[0].triangleArray.numSbtRecords             = numSbtRecords;
    m_buildInputs[0].triangleArray.sbtIndexOffsetSizeInBytes = 1;

    ASSERT_OPTIX_SUCCESS( memComputeAccelOneInput() );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, index_offset_buffer_offset_size_0 )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.sbtIndexOffsetSizeInBytes" (0) must be either 1, 2, or 4.)msg" );
    m_buildInputs[0].triangleArray.numSbtRecords             = 2;
    m_buildInputs[0].triangleArray.sbtIndexOffsetBuffer      = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.sbtIndexOffsetSizeInBytes = 0;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, index_offset_buffer_offset_size_3 )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.sbtIndexOffsetSizeInBytes" (3) must be either 1, 2, or 4.)msg" );
    m_buildInputs[0].triangleArray.numSbtRecords             = 2;
    m_buildInputs[0].triangleArray.sbtIndexOffsetBuffer      = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.sbtIndexOffsetSizeInBytes = 3;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, index_offset_buffer_offset_size_5 )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray.sbtIndexOffsetSizeInBytes" (5) must be either 1, 2, or 4.)msg" );
    m_buildInputs[0].triangleArray.numSbtRecords             = 2;
    m_buildInputs[0].triangleArray.sbtIndexOffsetBuffer      = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.sbtIndexOffsetSizeInBytes = 5;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, index_offset_buffer_offset_size_2_stride_1 )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "sbtIndexOffsetStrideInBytes" is smaller than "sbtIndexOffsetSizeInBytes")msg" );
    m_buildInputs[0].triangleArray.numSbtRecords               = 2;
    m_buildInputs[0].triangleArray.sbtIndexOffsetBuffer        = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.sbtIndexOffsetSizeInBytes   = 2;
    m_buildInputs[0].triangleArray.sbtIndexOffsetStrideInBytes = 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, index_offset_buffer_offset_size_4_stride_2 )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray", "sbtIndexOffsetStrideInBytes" is smaller than "sbtIndexOffsetSizeInBytes")msg" );
    m_buildInputs[0].triangleArray.numSbtRecords               = 2;
    m_buildInputs[0].triangleArray.sbtIndexOffsetBuffer        = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.sbtIndexOffsetSizeInBytes   = 4;
    m_buildInputs[0].triangleArray.sbtIndexOffsetStrideInBytes = 2;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

// TODO: 2664761 [O7] Creating an AS from a single triangle build input with OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS triangles asserts in ChunkedTrbvhBuilder
TEST_F( TestAccelBuildTriangles, DISABLED_max_triangles_single_input )
{
    // Choose a vertex format that is as small as possible in stride.
    const OptixVertexFormat fmt                 = OPTIX_VERTEX_FORMAT_HALF2;
    m_buildInputs[0].triangleArray.vertexFormat = fmt;
    const unsigned int numVertices              = 3 * m_maxNumPrimsPerGAS;
    m_buildInputs[0].triangleArray.numVertices  = numVertices;
    allocateVertexBuffer( fmt, m_maxNumPrimsPerGAS );
    ASSERT_LWDA_SUCCESS( lwdaMemset( reinterpret_cast<void*>( m_vertexBuffer ), 0, numVertices * naturalVertexStride( fmt ) ) );
    m_buildInputs[0].triangleArray.vertexBuffers = &m_vertexBuffer;

    resizeBuffers();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, too_many_triangles_direct_single_input )
{
    EXPECT_LOG_MSG( R"msg(Sum of number of triangles/primitives (536870913) over first 1 build inputs out of 1 build inputs exceeds device context property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS (536870912).)msg" );
    m_buildInputs[0].triangleArray.numVertices = 3 * ( m_maxNumPrimsPerGAS + 1 );

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, too_many_triangles_indirect_single_input )
{
    EXPECT_LOG_MSG( R"msg(Sum of number of triangles/primitives (536870913) over first 1 build inputs out of 1 build inputs exceeds device context property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS (536870912).)msg" );
    m_buildInputs[0].triangleArray.numIndexTriplets = m_maxNumPrimsPerGAS + 1;
    m_buildInputs[0].triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    m_buildInputs[0].triangleArray.indexBuffer      = m_indexBuffer;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, too_many_triangles_direct_multiple_inputs )
{
    EXPECT_LOG_MSG( R"msg(Sum of number of triangles/primitives (536870913) over first 2 build inputs out of 2 build inputs exceeds device context property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS (536870912).)msg" );
    m_buildInputs.resize( 2 );
    std::vector<unsigned int> sbtFlags( 2, 0U );
    setupBasicTriangleInput( 0, &sbtFlags[0] );
    setupBasicTriangleInput( 1, &sbtFlags[1] );
    m_buildInputs[0].triangleArray.numVertices   = 3 * m_maxNumPrimsPerGAS;
    LWdeviceptr vb0                              = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.vertexBuffers = &vb0;
    m_buildInputs[1].triangleArray.numVertices   = 3;
    LWdeviceptr vb1                              = ARBITRARY_DEVICE_PTR;
    m_buildInputs[1].triangleArray.vertexBuffers = &vb1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelFromInputs() );
}

TEST_F( TestAccelBuildTriangles, too_many_triangles_indirect_multiple_inputs )
{
    EXPECT_LOG_MSG( R"msg(Sum of number of triangles/primitives (536870913) over first 2 build inputs out of 2 build inputs exceeds device context property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS (536870912).)msg" );
    m_buildInputs.resize( 2 );
    std::vector<unsigned int> sbtFlags( 2, 0U );
    setupBasicTriangleInput( 0, &sbtFlags[0] );
    setupBasicTriangleInput( 1, &sbtFlags[1] );
    m_buildInputs[0].triangleArray.numVertices      = 3;
    LWdeviceptr vb0                                 = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.vertexBuffers    = &vb0;
    m_buildInputs[0].triangleArray.numIndexTriplets = m_maxNumPrimsPerGAS;
    m_buildInputs[0].triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    m_buildInputs[0].triangleArray.indexBuffer      = ARBITRARY_DEVICE_PTR;
    m_buildInputs[1].triangleArray.numVertices      = 3;
    LWdeviceptr vb1                                 = ARBITRARY_DEVICE_PTR;
    m_buildInputs[1].triangleArray.vertexBuffers    = &vb1;
    m_buildInputs[1].triangleArray.numIndexTriplets = 1;
    m_buildInputs[1].triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    m_buildInputs[1].triangleArray.indexBuffer      = ARBITRARY_DEVICE_PTR;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelFromInputs() );
}

TEST_F( TestAccelBuildTriangles, prim_index_too_large )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].triangleArray": number of triangles/primitives + "primitiveIndexOffset" overflows 32 bits.)msg" );
    m_buildInputs[0].triangleArray.numVertices          = 3;
    const LWdeviceptr vb                                = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.vertexBuffers        = &vb;
    m_buildInputs[0].triangleArray.primitiveIndexOffset = std::numeric_limits<std::uint32_t>::max();

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, DISABLED_max_sbts_single_input )
{
    // Choose a vertex format that is as small as possible in stride.
    const OptixVertexFormat fmt                 = OPTIX_VERTEX_FORMAT_HALF2;
    m_buildInputs[0].triangleArray.vertexFormat = fmt;
    const unsigned int numVertices              = 3;
    m_buildInputs[0].triangleArray.numVertices  = numVertices;
    allocateVertexBuffer( fmt, m_maxNumPrimsPerGAS );
    ASSERT_LWDA_SUCCESS( lwdaMemset( reinterpret_cast<void*>( m_vertexBuffer ), 0, numVertices * naturalVertexStride( fmt ) ) );
    m_buildInputs[0].triangleArray.vertexBuffers  = &m_vertexBuffer;
    const unsigned int              numSbtRecords = m_maxSbtRecords;
    const std::vector<unsigned int> sbtFlags( numSbtRecords, 0U );
    m_buildInputs[0].triangleArray.numSbtRecords = numSbtRecords;
    m_buildInputs[0].triangleArray.flags         = &sbtFlags[0];
    LWdeviceptr sbtIndexOffsets                  = nulldevptr;
    ASSERT_LWDA_SUCCESS( lwdaMalloc( reinterpret_cast<void**>( &sbtIndexOffsets ), 1 ) );
    ASSERT_LWDA_SUCCESS( lwdaMemset( reinterpret_cast<void*>( sbtIndexOffsets ), 0, 1 ) );
    m_buildInputs[0].triangleArray.sbtIndexOffsetBuffer        = sbtIndexOffsets;
    m_buildInputs[0].triangleArray.sbtIndexOffsetSizeInBytes   = 1;
    m_buildInputs[0].triangleArray.sbtIndexOffsetStrideInBytes = 0;

    resizeBuffers();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );

    ASSERT_LWDA_SUCCESS( lwdaFree( reinterpret_cast<void*>( sbtIndexOffsets ) ) );
}

TEST_F( TestAccelBuildTriangles, too_many_sbts_single_input )
{
    EXPECT_LOG_MSG( R"msg(Sum of "numSbtRecords" (268435457) over first 1 build inputs out of 1 build inputs exceeds device context property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS (268435456).)msg" );
    const unsigned int        numSbtRecords = m_maxSbtRecords + 1;
    std::vector<unsigned int> sbtFlags( numSbtRecords, 0U );
    m_buildInputs[0].triangleArray.numSbtRecords               = numSbtRecords;
    m_buildInputs[0].triangleArray.flags                       = &sbtFlags[0];
    m_buildInputs[0].triangleArray.sbtIndexOffsetBuffer        = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.sbtIndexOffsetSizeInBytes   = 2;
    m_buildInputs[0].triangleArray.sbtIndexOffsetStrideInBytes = 0;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildTriangles, too_many_sbts_multiple_inputs )
{
    EXPECT_LOG_MSG( R"msg(Sum of "numSbtRecords" (268435457) over first 2 build inputs out of 2 build inputs exceeds device context property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS (268435456).)msg" );
    m_buildInputs.resize( 2 );
    const std::vector<unsigned int> sbtFlags( m_maxSbtRecords + 1, 0U );
    setupBasicTriangleInput( 0, &sbtFlags[0] );
    setupBasicTriangleInput( 1, &sbtFlags[1] );
    m_buildInputs[0].triangleArray.numSbtRecords               = 1;
    m_buildInputs[0].triangleArray.sbtIndexOffsetBuffer        = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].triangleArray.sbtIndexOffsetSizeInBytes   = 2;
    m_buildInputs[0].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    m_buildInputs[1].triangleArray.numSbtRecords               = m_maxSbtRecords;
    m_buildInputs[1].triangleArray.flags                       = &sbtFlags[1];
    m_buildInputs[1].triangleArray.sbtIndexOffsetBuffer        = ARBITRARY_DEVICE_PTR;
    m_buildInputs[1].triangleArray.sbtIndexOffsetSizeInBytes   = 2;
    m_buildInputs[1].triangleArray.sbtIndexOffsetStrideInBytes = 0;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelFromInputs() );
}

TEST_F( TestAccelBuildTriangles, empty_input )
{
    setZeroTriangle();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    // TODO: 2650681 OptiX 7 should treat a traversable handle of zero as a no-op everywhere
    //ASSERT_EQ( NULL_TRAVERSABLE_HANDLE, m_handle );
}

TEST_F( TestAccelBuildLwstomPrimitives, zero_primitives_nonzero_aabb_buffers )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray", "numPrimitives" is zero, but "aabbBuffers" is non-null)msg" );
    m_buildInputs[0].lwstomPrimitiveArray.numPrimitives = 0;
    const LWdeviceptr aabbBuffer                        = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].lwstomPrimitiveArray.aabbBuffers   = &aabbBuffer;

    ASSERT_OPTIX_ILWALID_VALUE( memComputeAccelOneInput() );
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray", "numPrimitives" is zero, but "aabbBuffers" is non-null)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, zero_primitives_disabled_anyhit )
{
	m_accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
	m_accelOptions.motionOptions = { 0, OPTIX_MOTION_FLAG_NONE, 0.0f, 0.0f };

	m_buildInputs[0].lwstomPrimitiveArray.aabbBuffers = nullptr;
	m_buildInputs[0].lwstomPrimitiveArray.strideInBytes = sizeof(OptixAabb);
	m_buildInputs[0].lwstomPrimitiveArray.numPrimitives = 0;
	const unsigned int flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
	m_buildInputs[0].lwstomPrimitiveArray.flags = &flags;
	m_buildInputs[0].lwstomPrimitiveArray.numSbtRecords = 1;

	ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, primitives_zero_aabb_buffers )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray", "numPrimitives" is non-zero, but "aabbBuffers[0]" is null)msg" );
    m_aabbBuffers[0] = nulldevptr;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, null_aabbs )
{
    m_buildInputs[0].lwstomPrimitiveArray.aabbBuffers = nullptr;

    ASSERT_OPTIX_SUCCESS( memComputeAccelOneInput() );
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray", "numPrimitives" is non-zero, but "aabbBuffers" is null)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, misaligned_aabbs )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray.aabbBuffers[0]" must be a multiple of OPTIX_AABB_BUFFER_BYTE_ALIGNMENT)msg" );
    m_aabbBuffers[0] = misalignDevicePtr( ARBITRARY_DEVICE_PTR, OPTIX_AABB_BUFFER_BYTE_ALIGNMENT );

    ASSERT_OPTIX_ILWALID_VALUE( memComputeAccelOneInput() );
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray.aabbBuffers[0]" must be a multiple of OPTIX_AABB_BUFFER_BYTE_ALIGNMENT)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, stride_too_small )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray.strideInBytes" (23) is smaller than sizeof( OptixAabb ))msg" );
    m_buildInputs[0].lwstomPrimitiveArray.strideInBytes = sizeof( OptixAabb ) - 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );

    setZeroPrim();
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray.strideInBytes" (23) is smaller than sizeof( OptixAabb ))msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, aabb_stride_misaligned )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray.strideInBytes" (31) must be a multiple of OPTIX_AABB_BUFFER_BYTE_ALIGNMENT.)msg" );
    allocateAabbBuffer( 1 );
    fillAabbBuffer( 0, 1 );
    m_aabbBuffers[0]                                    = m_aabbBuffer;
    m_buildInputs[0].lwstomPrimitiveArray.numPrimitives = 1;
    m_buildInputs[0].lwstomPrimitiveArray.strideInBytes = sizeof( OptixAabb ) + OPTIX_AABB_BUFFER_BYTE_ALIGNMENT - 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );

    setZeroPrim();
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray.strideInBytes" (31) must be a multiple of OPTIX_AABB_BUFFER_BYTE_ALIGNMENT.)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, aabb_stride_exact )
{
    allocateAabbBuffer( 1 );
    fillAabbBuffer( 0, 1 );
    m_aabbBuffers[0]                                    = m_aabbBuffer;
    m_buildInputs[0].lwstomPrimitiveArray.numPrimitives = 1;
    m_buildInputs[0].lwstomPrimitiveArray.strideInBytes = sizeof( OptixAabb );

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );

    setZeroPrim();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, aabb_stride_excess )
{
    allocateAabbBuffer( 1 );
    fillAabbBuffer( 0, 1 );
    m_aabbBuffers[0]                                    = m_aabbBuffer;
    m_buildInputs[0].lwstomPrimitiveArray.numPrimitives = 1;
    m_buildInputs[0].lwstomPrimitiveArray.strideInBytes = sizeof( OptixAabb ) + OPTIX_AABB_BUFFER_BYTE_ALIGNMENT;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );

    setZeroPrim();
    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, null_flags )
{
    EXPECT_LOG_MSG( R"msg(Invalid value (0) for "buildInputs[0].lwstomPrimitiveArray.flags")msg" );
    m_aabbBuffers[0]                                    = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].lwstomPrimitiveArray.numSbtRecords = 1;
    m_buildInputs[0].lwstomPrimitiveArray.flags         = nullptr;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );

    setZeroPrim();
    EXPECT_LOG_MSG( R"msg(Invalid value (0) for "buildInputs[0].lwstomPrimitiveArray.flags")msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, bad_flags )
{
    EXPECT_LOG_MSG( R"msg(Invalid value (4294967295) for "buildInputs[0].lwstomPrimitiveArray.flags[0]")msg" );
    m_sbtFlags[0] = ~0U;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );

    setZeroPrim();
    EXPECT_LOG_MSG( R"msg(Invalid value (4294967295) for "buildInputs[0].lwstomPrimitiveArray.flags[0]")msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, zero_sbts )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray.numSbtRecords" is zero)msg" );
    m_buildInputs[0].lwstomPrimitiveArray.numSbtRecords = 0;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );

    setZeroPrim();
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray.numSbtRecords" is zero)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, null_sbt_index_offset_buffer_many_sbts )
{
    m_buildInputs[0].lwstomPrimitiveArray.numSbtRecords             = 2;
    m_buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetBuffer      = nulldevptr;
    m_buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetSizeInBytes = 1;
    m_sbtFlags.resize( 2, OPTIX_GEOMETRY_FLAG_NONE );
    m_buildInputs[0].lwstomPrimitiveArray.flags = &m_sbtFlags[0];

    ASSERT_OPTIX_SUCCESS( memComputeAccelOneInput() );
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray", "numSbtRecords" is larger than one, but "sbtIndexOffsetBuffer" is null)msg" );
    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, index_offset_buffer_offset_size_0 )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetSizeInBytes" (0) must be either 1, 2, or 4.)msg" );
    m_buildInputs[0].lwstomPrimitiveArray.numSbtRecords             = 2;
    m_buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetBuffer      = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, index_offset_buffer_offset_size_3 )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetSizeInBytes" (3) must be either 1, 2, or 4.)msg" );
    m_buildInputs[0].lwstomPrimitiveArray.numSbtRecords             = 2;
    m_buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetBuffer      = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetSizeInBytes = 3;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, index_offset_buffer_offset_size_5 )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetSizeInBytes" (5) must be either 1, 2, or 4.)msg" );
    m_buildInputs[0].lwstomPrimitiveArray.numSbtRecords             = 2;
    m_buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetBuffer      = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetSizeInBytes = 5;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, index_offset_buffer_offset_size_2_stride_1 )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray", "sbtIndexOffsetStrideInBytes" is smaller than "sbtIndexOffsetSizeInBytes")msg" );
    m_buildInputs[0].lwstomPrimitiveArray.numSbtRecords               = 2;
    m_buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetBuffer        = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetSizeInBytes   = 2;
    m_buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetStrideInBytes = 1;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, index_offset_buffer_offset_size_4_stride_2 )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray", "sbtIndexOffsetStrideInBytes" is smaller than "sbtIndexOffsetSizeInBytes")msg" );
    m_buildInputs[0].lwstomPrimitiveArray.numSbtRecords               = 2;
    m_buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetBuffer        = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetSizeInBytes   = 4;
    m_buildInputs[0].lwstomPrimitiveArray.sbtIndexOffsetStrideInBytes = 2;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, prim_index_too_large )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].lwstomPrimitiveArray": number of triangles/primitives + "primitiveIndexOffset" overflows 32 bits.)msg" );
    m_buildInputs[0].lwstomPrimitiveArray.primitiveIndexOffset = std::numeric_limits<std::uint32_t>::max();

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildLwstomPrimitives, empty_input )
{
    m_buildInputs[0].lwstomPrimitiveArray.numPrimitives = 0;
    m_buildInputs[0].lwstomPrimitiveArray.aabbBuffers   = nullptr;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    // TODO: 2650681 OptiX 7 should treat a traversable handle of zero as a no-op everywhere
    //ASSERT_EQ( NULL_TRAVERSABLE_HANDLE, m_handle );
}

TEST_F( TestAccelBuildInstances, null_instances_non_zero_num_instances )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].instanceArray", "numInstances" is non-zero, but "instances" is null)msg" );
    m_buildInputs[0].instanceArray.instances    = 0U;
    m_buildInputs[0].instanceArray.numInstances = 1U;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildInstances, non_null_instances_zero_num_instances )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].instanceArray", "numInstances" is zero, but "instances" is non-null)msg" );
    m_buildInputs[0].instanceArray.instances    = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].instanceArray.numInstances = 0U;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildInstances, misaligned_instances )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].instanceArray.instances" is not a multiple of OPTIX_INSTANCE_BYTE_ALIGNMENT)msg" );
    m_buildInputs[0].instanceArray.instances = misalignDevicePtr( ARBITRARY_DEVICE_PTR, OPTIX_INSTANCE_BYTE_ALIGNMENT );
    m_buildInputs[0].instanceArray.numInstances = 1U;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildInstances, multiple_instances )
{
    EXPECT_LOG_MSG( R"msg("numBuildInputs" must be 1 for instance acceleration builds)msg" );
    m_buildInputs.resize( 2 );
    std::fill( m_buildInputs.begin(), m_buildInputs.end(), OptixBuildInput{} );
    m_buildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    m_buildInputs[1].type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelFromInputs() );
}

TEST_F( TestAccelBuildInstances, too_many_instances )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].instanceArray.numInstances" (268435457) exceeds device context property OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS (268435456))msg" );
    const unsigned int numInstances             = m_maxInstancesPerIAS + 1;
    m_buildInputs[0].instanceArray.instances    = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].instanceArray.numInstances = numInstances;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildInstances, empty_input )
{
    m_buildInputs[0].instanceArray.numInstances = 0U;
    m_buildInputs[0].instanceArray.instances    = nulldevptr;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    // TODO: 2650681 OptiX 7 should treat a traversable handle of zero as a no-op everywhere
    //ASSERT_EQ( NULL_TRAVERSABLE_HANDLE, m_handle );
}

TEST_F( TestAccelBuildInstancePointers, null_instances_non_zero_num_instances )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].instanceArray", "numInstances" is non-zero, but "instances" is null)msg" );
    m_buildInputs[0].instanceArray.instances    = 0U;
    m_buildInputs[0].instanceArray.numInstances = 1U;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildInstancePointers, non_null_instances_zero_num_instances )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].instanceArray", "numInstances" is zero, but "instances" is non-null)msg" );
    m_buildInputs[0].instanceArray.instances    = ARBITRARY_DEVICE_PTR;
    m_buildInputs[0].instanceArray.numInstances = 0U;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildInstancePointers, misaligned_instances )
{
    EXPECT_LOG_MSG( R"msg("buildInputs[0].instanceArray.instances" is not a multiple of 8)msg" );
    m_buildInputs[0].instanceArray.instances    = misalignDevicePtr( ARBITRARY_DEVICE_PTR, DEVICE_PTR_ALIGNMENT );
    m_buildInputs[0].instanceArray.numInstances = 1U;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

TEST_F( TestAccelBuildInstancePointers, multiple_instances )
{
    EXPECT_LOG_MSG( R"msg("numBuildInputs" must be 1 for instance acceleration builds)msg" );
    m_buildInputs.resize( 2 );
    std::fill( m_buildInputs.begin(), m_buildInputs.end(), OptixBuildInput{} );
    m_buildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS;
    m_buildInputs[1].type = OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS;

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelFromInputs() );
}

TEST_F( TestAccelBuildInstancePointers, empty_input )
{
    m_buildInputs[0].instanceArray.numInstances = 0U;
    m_buildInputs[0].instanceArray.instances    = nulldevptr;

    ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    ASSERT_LWDA_SUCCESS( lwdaStreamSynchronize( m_stream ) );

    // TODO: 2650681 OptiX 7 should treat a traversable handle of zero as a no-op everywhere
    //ASSERT_EQ( NULL_TRAVERSABLE_HANDLE, m_handle );
}

TEST_F( TestAccelUpdate, no_update_flag )
{
    setupEmptyTriangleInput();
    ASSERT_OPTIX_SUCCESS( buildAccelForUpdateOneInput() );
    m_accelOptions.operation  = OPTIX_BUILD_OPERATION_UPDATE;
    m_accelOptions.buildFlags = 0U;
    EXPECT_LOG_MSG( R"msg("accelOptions.operation" is OPTIX_BUILD_OPERATION_UPDATE but "accelOptions.buildFlags" does not specify OPTIX_BUILD_FLAG_ALLOW_UPDATE)msg" );

    ASSERT_OPTIX_ILWALID_VALUE( buildAccelOneInput() );
}

//---------------------------------------------------------------------------
// 	optixAccelCheckRelocationCompatibility
//---------------------------------------------------------------------------

struct O7_API_optixAccelCheckRelocationCompatibility : TestAccelBuildLwstomPrimitives
{
    // taken from TestAccelBuildLwstomPrimitives.aabb_stride_exact()
    void SetUp() override
    {
        TestAccelBuildLwstomPrimitives::SetUp();
        ASSERT_OPTIX_SUCCESS( optixDeviceContextSetLogCallback( m_context, &OptixRecordingLogger::callback, &m_logger, 2 ) );

        allocateAabbBuffer( 1 );
        fillAabbBuffer( 0, 1 );
        m_aabbBuffers[0]                                    = m_aabbBuffer;
        m_buildInputs[0].lwstomPrimitiveArray.numPrimitives = 1;
        m_buildInputs[0].lwstomPrimitiveArray.strideInBytes = sizeof( OptixAabb );

        ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
        ASSERT_OPTIX_SUCCESS( optixAccelGetRelocationInfo( m_context, m_handle, &m_info ) );
    }
    OptixRecordingLogger     m_logger;
    OptixAccelRelocationInfo m_info;
};


TEST_F( O7_API_optixAccelCheckRelocationCompatibility, CheckWithIdenticalContext )
{
    int res = {};
    ASSERT_OPTIX_SUCCESS( optixAccelCheckRelocationCompatibility( m_context, &m_info, &res ) );
    ASSERT_EQ( 1, res );
}


TEST_F( O7_API_optixAccelCheckRelocationCompatibility, CheckWithDifferentContextOnSameDevice )
{
    int                res = {};
    OptixDeviceContext new_context;
    ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( nullptr, nullptr, &new_context ) );
    ASSERT_OPTIX_SUCCESS( optixAccelCheckRelocationCompatibility( new_context, &m_info, &res ) );
    ASSERT_EQ( 1, res );
}


TEST_F( O7_API_optixAccelCheckRelocationCompatibility, CallWithNullptrContext )
{
    int res = {};
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT, optixAccelCheckRelocationCompatibility( nullptr, &m_info, &res ) );
}


TEST_F( O7_API_optixAccelCheckRelocationCompatibility, CallWithNullptrInfoArg )
{
    int res = {};
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixAccelCheckRelocationCompatibility( m_context, nullptr, &res ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "info is null" ) );
}


TEST_F( O7_API_optixAccelCheckRelocationCompatibility, CallWithNullptrReturnArg )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixAccelCheckRelocationCompatibility( m_context, &m_info, nullptr ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "compatible is null" ) );
}


// Since rtcore is using internal APIs to retrieve chip architecture etc we cannot decide inside a test
// whether the relocation would be compatibel for sure. Hence we try to detect incompatible cases only.
TEST_F( O7_API_optixAccelCheckRelocationCompatibility, CallOnDifferentGPUs )
{
    int32_t device_count = 0;
    LWDA_CHECK( lwdaGetDeviceCount( &device_count ) );
    std::cout << "Total GPUs visible: " << device_count << std::endl;
    if( device_count > 1 )
    {
        std::vector<lwdaDeviceProp> prop( device_count );
        for( int i = 0; i < device_count; ++i )
        {
            LWDA_CHECK( lwdaGetDeviceProperties( &prop[i], i ) );
        }

        int lwrrent_device = {};
        LWDA_CHECK( lwdaGetDevice( &lwrrent_device ) );

        // create context on a "another" device and check for compatibility
        for( int i = 0; i < device_count; ++i )
        {
            if( i == lwrrent_device )
                continue;
            LWDA_CHECK( lwdaSetDevice( i ) );
            exptest::lwdaInitialize();
            ASSERT_OPTIX_SUCCESS( optixInit() );

            int                res = {};
            OptixDeviceContext new_context;
            ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( nullptr, nullptr, &new_context ) );
            ASSERT_OPTIX_SUCCESS( optixAccelCheckRelocationCompatibility( new_context, &m_info, &res ) );

            // TBD What exactly is the decision-making criterion? Lwrrently checking compute capabilities.
            // The following is not enough as the internal impl checks for chip architecture, driver version, TTU, ...
            // But inequality should be sufficient for being incompatible
            if( prop[i].major != prop[lwrrent_device].major || prop[i].minor != prop[lwrrent_device].minor )
            {
                ASSERT_EQ( 0, res );
            }
        }
    }
}


//---------------------------------------------------------------------------
// 	optixAccelCompact
//---------------------------------------------------------------------------

struct O7_API_optixAccelCompact : Test
{
    void SetUp() override
    {
        exptest::lwdaInitialize();
        ASSERT_OPTIX_SUCCESS( optixInit() );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( nullptr, nullptr, &m_context ) );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextSetLogCallback( m_context, &OptixRecordingLogger::callback, &m_logger, 2 ) );
        ASSERT_LWDA_SUCCESS( lwdaStreamCreate( &m_stream ) );

        m_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        m_accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

        buildAabb();
    }
    void TearDown() override
    {
        ASSERT_LWDA_SUCCESS( lwdaStreamDestroy( m_stream ) );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextDestroy( m_context ) );
    }

    void buildAabb()
    {
        // AABB build input
        OptixAabb   aabb = {-1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f};
        LWdeviceptr d_aabb_buffer;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_aabb_buffer ), sizeof( OptixAabb ) ) );
        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_aabb_buffer ), &aabb, sizeof( OptixAabb ), lwdaMemcpyHostToDevice ) );


        aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES;
        aabb_input.lwstomPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
        aabb_input.lwstomPrimitiveArray.numPrimitives = 1;

        uint32_t aabb_input_flags[1]                  = {OPTIX_GEOMETRY_FLAG_NONE};
        aabb_input.lwstomPrimitiveArray.flags         = aabb_input_flags;
        aabb_input.lwstomPrimitiveArray.numSbtRecords = 1;

        OPTIX_CHECK( optixAccelComputeMemoryUsage( m_context, &m_accel_options, &aabb_input, 1, &m_buffer_sizes ) );
        LWdeviceptr d_temp_buffer_gas;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), m_buffer_sizes.tempSizeInBytes ) );

        // non-compacted output
        size_t compactedSizeOffset = roundUp<size_t>( m_buffer_sizes.outputSizeInBytes, 8ull );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ), compactedSizeOffset + 8 ) );

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result             = reinterpret_cast<LWdeviceptr>(
            reinterpret_cast<char*>( d_buffer_temp_output_gas_and_compacted_size ) + compactedSizeOffset );

        OPTIX_CHECK( optixAccelBuild( m_context,
                                      m_stream,  // LWCA stream
                                      &m_accel_options, &aabb_input,
                                      1,  // num build inputs
                                      d_temp_buffer_gas, m_buffer_sizes.tempSizeInBytes,
                                      d_buffer_temp_output_gas_and_compacted_size, m_buffer_sizes.outputSizeInBytes, &m_gas_handle,
                                      &emitProperty,  // emitted property list
                                      1               // num emitted properties
                                      ) );
        //state.params.radius = 1.5f;

        //LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
        //LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( d_aabb_buffer ) ) );

        LWDA_CHECK( lwdaMemcpy( &m_compacted_gas_size, reinterpret_cast<void*>( emitProperty.result ), sizeof( size_t ),
                                lwdaMemcpyDeviceToHost ) );
        //LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( d_buffer_temp_output_gas_and_compacted_size ) ) );
    }
    template <typename IntegerType>
    IntegerType roundUp( IntegerType x, IntegerType y )
    {
        return ( ( x + y - 1 ) / y ) * y;
    }

    OptixDeviceContext     m_context       = {};
    LWstream               m_stream        = nullptr;
    OptixAccelBuildOptions m_accel_options = {};
    OptixTraversableHandle m_gas_handle    = 0;
    OptixBuildInput        aabb_input      = {};
    OptixRecordingLogger   m_logger;
    size_t                 m_compacted_gas_size;
    OptixAccelBufferSizes  m_buffer_sizes = {};
    LWdeviceptr            d_buffer_temp_output_gas_and_compacted_size;
};


TEST_F( O7_API_optixAccelCompact, RunWithSuccess )
{
    LWdeviceptr outputBuffer = 0;
    LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &outputBuffer ), m_compacted_gas_size ) );
    // use handle as input and output
    OPTIX_CHECK( optixAccelCompact( m_context,
                                    m_stream,  // LWCA stream
                                    m_gas_handle, outputBuffer, m_compacted_gas_size, &m_gas_handle ) );
    LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( outputBuffer ) ) );
}


TEST_F( O7_API_optixAccelCompact, NullPtrContext )
{
    LWdeviceptr outputBuffer = 0;
    LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &outputBuffer ), m_compacted_gas_size ) );
    // use handle as input and output
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT,
               optixAccelCompact( nullptr,
                                  m_stream,  // LWCA stream
                                  m_gas_handle, outputBuffer, m_compacted_gas_size, &m_gas_handle ) );
    LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( outputBuffer ) ) );
}

//---------------------------------------------------------------------------
// 	optixAccelComputeMemoryUsage
//---------------------------------------------------------------------------

struct O7_API_optixAccelComputeMemoryUsage : TestAccelBuildLwstomPrimitives
{
};


TEST_F( O7_API_optixAccelComputeMemoryUsage, EmptyTriangleInput )
{
    setupEmptyTriangleInput();
    OptixAccelBufferSizes bufferSizes = {};
    ASSERT_OPTIX_SUCCESS( optixAccelComputeMemoryUsage( m_context, &m_accelOptions, &m_buildInputs[0], 1, &bufferSizes ) );
    // we cannot check the returned sizes as they are HW dependent
}


TEST_F( O7_API_optixAccelComputeMemoryUsage, RunWithNullptrContext )
{
    setupEmptyTriangleInput();
    OptixAccelBufferSizes bufferSizes = {};
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT,
               optixAccelComputeMemoryUsage( nullptr, &m_accelOptions, &m_buildInputs[0], 1, &bufferSizes ) );
}

//---------------------------------------------------------------------------
// 	optixAccelGetRelocationInfo
//---------------------------------------------------------------------------

struct O7_API_optixAccelGetRelocationInfo : TestAccelBuildLwstomPrimitives
{
    // taken from TestAccelBuildLwstomPrimitives.aabb_stride_exact()
    void SetUp() override
    {
        TestAccelBuildLwstomPrimitives::SetUp();
        ASSERT_OPTIX_SUCCESS( optixDeviceContextSetLogCallback( m_context, &OptixRecordingLogger::callback, &m_logger, 2 ) );

        allocateAabbBuffer( 1 );
        fillAabbBuffer( 0, 1 );
        m_aabbBuffers[0]                                    = m_aabbBuffer;
        m_buildInputs[0].lwstomPrimitiveArray.numPrimitives = 1;
        m_buildInputs[0].lwstomPrimitiveArray.strideInBytes = sizeof( OptixAabb );

        ASSERT_OPTIX_SUCCESS( buildAccelOneInput() );
    }
    OptixRecordingLogger m_logger;
};


TEST_F( O7_API_optixAccelGetRelocationInfo, SuccessfulRun )
{
    OptixAccelRelocationInfo info;
    ASSERT_OPTIX_SUCCESS( optixAccelGetRelocationInfo( m_context, m_handle, &info ) );
}


TEST_F( O7_API_optixAccelGetRelocationInfo, RunWithNullptrContext )
{
    OptixAccelRelocationInfo info;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT, optixAccelGetRelocationInfo( nullptr, m_handle, &info ) );
}


TEST_F( O7_API_optixAccelGetRelocationInfo, CallWithNullHandleArg )
{
    OptixAccelRelocationInfo info;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixAccelGetRelocationInfo( m_context, 0, &info ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "handle is 0" ) );
}


TEST_F( O7_API_optixAccelGetRelocationInfo, CallWithNullptrInfoArg )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixAccelGetRelocationInfo( m_context, m_handle, nullptr ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "info is null" ) );
}


//---------------------------------------------------------------------------
// 	optixAccelRelocate
//---------------------------------------------------------------------------

struct O7_API_optixAccelRelocate : O7_API_optixAccelCompact
{
    void SetUp() override
    {
        O7_API_optixAccelCompact::SetUp();
        output_buffer_size = m_buffer_sizes.outputSizeInBytes;

        OPTIX_CHECK( optixAccelGetRelocationInfo( m_context, m_gas_handle, &relocationInfo ) );

        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_new_output_buffer ), output_buffer_size ) );
        LWDA_CHECK( lwdaMemcpy( (void*)d_new_output_buffer, (const void*)d_buffer_temp_output_gas_and_compacted_size,
                                output_buffer_size, lwdaMemcpyDeviceToDevice ) );
    }
    void TearDown() override
    {
        LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( d_new_output_buffer ) ) );
        O7_API_optixAccelCompact::TearDown();
    }
    OptixAccelRelocationInfo relocationInfo;
    LWdeviceptr              d_new_output_buffer;
    size_t                   output_buffer_size;
};


TEST_F( O7_API_optixAccelRelocate, RunWithSuccess )
{
    OptixTraversableHandle new_gas_handle;
    OPTIX_CHECK( optixAccelRelocate( m_context, 0, &relocationInfo, 0, 0, d_new_output_buffer, output_buffer_size, &new_gas_handle ) );
}


TEST_F( O7_API_optixAccelRelocate, NullPtrContext )
{
    OptixTraversableHandle new_gas_handle;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT, optixAccelRelocate( nullptr, 0, &relocationInfo, 0, 0, d_new_output_buffer,
                                                                       output_buffer_size, &new_gas_handle ) );
}


//---------------------------------------------------------------------------
// 	optixColwertPointerToTraversableHandle
//---------------------------------------------------------------------------

struct O7_API_optixColwertPointerToTraversableHandle : Test, testing::WithParamInterface<OptixTraversableType>
{
    void SetUp()
    {
        exptest::lwdaInitialize();
        ASSERT_OPTIX_SUCCESS( optixInit() );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( nullptr, nullptr, &m_context ) );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextSetLogCallback( m_context, &OptixRecordingLogger::callback, &m_logger, 2 ) );
        ASSERT_LWDA_SUCCESS( lwdaStreamCreate( &m_stream ) );

        // setup transformation
        switch( GetParam() )
        {
            case OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM:
            {
                const float D                      = 0.05f;
                const float staticTransform[12]    = {1.0, 0.0, 0.0, D, 0.0, 1.0, 0.0, D, 0.0, 0.0, 1.0, 0.0};
                const float staticIlwTransform[12] = {1.0, 0.0, 0.0, -D, 0.0, 1.0, 0.0, -D, 0.0, 0.0, 1.0, 0.0};
                OptixStaticTransform transform     = {};
                transform.child                    = 0;
                memcpy( transform.transform, staticTransform, 12 * sizeof( float ) );
                memcpy( transform.ilwTransform, staticIlwTransform, 12 * sizeof( float ) );

                LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_transform ), sizeof( OptixStaticTransform ) ) );
                LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_transform ), &transform,
                                        sizeof( OptixStaticTransform ), lwdaMemcpyHostToDevice ) );
                break;
            }
            case OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM:
            {
                const float motion_matrix_keys[2][12] = {{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f},
                                                         {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f}};

                OptixMatrixMotionTransform motion_transform = {};
                motion_transform.child = 0;  // not a valid handle, but doesn't seem to matter for these tests
                motion_transform.motionOptions.numKeys   = 2;
                motion_transform.motionOptions.timeBegin = 0.0f;
                motion_transform.motionOptions.timeEnd   = 1.0f;
                motion_transform.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;
                memcpy( motion_transform.transform, motion_matrix_keys, 2 * 12 * sizeof( float ) );

                LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_transform ), sizeof( OptixMatrixMotionTransform ) ) );
                LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_transform ), &motion_transform,
                                        sizeof( OptixMatrixMotionTransform ), lwdaMemcpyHostToDevice ) );
                break;
            }
            case OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM:
            {
                OptixSRTData transform = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                                          0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};

                LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_transform ), sizeof( OptixSRTData ) ) );
                LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_transform ), &transform, sizeof( OptixSRTData ),
                                        lwdaMemcpyHostToDevice ) );
                break;
            }
            default:
            {
                OptixSRTData transform = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                                          0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};

                LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_transform ), sizeof( OptixSRTData ) ) );
                LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_transform ), &transform, sizeof( OptixSRTData ),
                                        lwdaMemcpyHostToDevice ) );
                break;
            }
        }
    }
    void TearDown()
    {
        ASSERT_LWDA_SUCCESS( lwdaFree( reinterpret_cast<void*>( d_transform ) ) );
        ASSERT_LWDA_SUCCESS( lwdaStreamDestroy( m_stream ) );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextDestroy( m_context ) );
    }

    OptixDeviceContext   m_context = nullptr;
    OptixRecordingLogger m_logger;
    LWstream             m_stream = nullptr;
    LWdeviceptr          d_transform;
};


TEST_P( O7_API_optixColwertPointerToTraversableHandle, RunWithSuccess )
{
    OptixTraversableHandle outHandle;
    OptixResult res = optixColwertPointerToTraversableHandle( m_context, d_transform, GetParam(), &outHandle );
    // testing against poor man's OPTIX_TRAVERSABLE_TYPE_UNDEFINED
    if( GetParam() != OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM - 1 )
        ASSERT_EQ( OPTIX_SUCCESS, res );
    else
        ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, res );
}


TEST_P( O7_API_optixColwertPointerToTraversableHandle, RunWithNullptrContext )
{
    OptixTraversableHandle outHandle;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT,
               optixColwertPointerToTraversableHandle( nullptr, d_transform, GetParam(), &outHandle ) );
}


TEST_P( O7_API_optixColwertPointerToTraversableHandle, RunWithNullTraversableHandle )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixColwertPointerToTraversableHandle( m_context, d_transform, GetParam(), nullptr ) );
}


TEST_P( O7_API_optixColwertPointerToTraversableHandle, RunWithNullPointer )
{
    OptixTraversableHandle outHandle;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixColwertPointerToTraversableHandle( m_context, 0, GetParam(), &outHandle ) );
}


TEST_P( O7_API_optixColwertPointerToTraversableHandle, RunWithNonAlignedPointer )
{
    OptixTraversableHandle outHandle;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixColwertPointerToTraversableHandle( m_context, d_transform + 1, GetParam(), &outHandle ) );
}


INSTANTIATE_TEST_SUITE_P( RunWithAndAllTraversableTypes,
                          O7_API_optixColwertPointerToTraversableHandle,
                          testing::Values( OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM,
                                           OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM,
                                           OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM,
                                           // poor man's OPTIX_TRAVERSABLE_TYPE_UNDEFINED
                                           OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM - 1 ) );
// only available in newer GTest versions
#if 0
        []( const testing::TestParamInfo<O7_API_optixColwertPointerToTraversableHandle::ParamType>& info ) {
            std::string name;
            switch( info )
            {
                case OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM:
                    name = "STATIC_TRANSFORM";
                    break;
                case OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM:
                    name = "MOTION_TRANSFORM";
                    break;
                case OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM:
                    name = "SRT_TRANSFORM";
                    break;
            }
            return name;
        } );
#endif
