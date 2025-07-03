
//
//  Copyright (c) 2020 LWPU Corporation.  All rights reserved.
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

#define OPTIX_OPTIONAL_FEATURE_OPTIX7_LWRVES

#include <optix.h>
#include <optix_stubs.h>

#include <exptest/exptest.h>
#include <exptest/loggingHelpers.h>
#include <exptest/testTools.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>

#include "CommonAsserts.h"

#include "test_DeviceAPI_triangleHit.h"
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include <exptest/coverageFileWriter.h>
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include "tests/sanity/test_DeviceAPI_triangleHit_ptx_bin.h"

using namespace testing;

using RaygenSbtRecord    = SbtRecord<void>;
using ExceptionSbtRecord = SbtRecord<void>;
using MissSbtRecord      = SbtRecord<void>;
using HitgroupSbtRecord  = SbtRecord<void>;
using CallableSbtRecord  = SbtRecord<void>;

using GtestParam = std::tuple<OptixProgramTypeTriangleHit, ExpectedTriangleHitType, UseHitTypeArgument>;

struct O7_API_Device_Triangle_Hit : public Test, public WithParamInterface<GtestParam>
{
    void SetUp() override {}

    void TearDown() override {}

    void runTest();

    static OptixDeviceContext          s_context;
    static OptixRecordingLogger        s_logger;
    static LWdeviceptr                 s_d_gasOutputBuffer;
    static OptixTraversableHandle      s_gasHandle;
    static OptixModule                 s_ptxModule;
    static OptixPipelineCompileOptions s_pipelineCompileOptions;

    Params             m_params;
    TriangleHitId      m_triangleHitResultOut;
    OptixPrimitiveType m_optixPrimitiveTypeResultOut;
    float3             m_hitPoint;

    static void SetUpTestCase()
    {
        exptest::lwdaInitialize();
        OPTIX_CHECK( optixInit() );

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &OptixRecordingLogger::callback;
        options.logCallbackData           = &s_logger;
        options.logCallbackLevel          = 2;
        LWcontext lwCtx                   = 0;  // zero means take the current context

        OPTIX_CHECK( optixDeviceContextCreate( lwCtx, &options, &s_context ) );

        // Create a single triangle to intersect.
        const std::array<float3, 3> vertices = { { { -1.f, -1.f, 0.f }, { 1.f, -1.f, 0.f }, { 0.f, 1.f, 0.f } } };

        const size_t sizeOfVertices = sizeof( float3 ) * vertices.size();
        LWdeviceptr  d_vertices     = 0x0;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_vertices ), sizeOfVertices ) );
        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_vertices ), vertices.data(), sizeOfVertices, lwdaMemcpyHostToDevice ) );

        const uint32_t  triangleInputFlags[1]     = { OPTIX_GEOMETRY_FLAG_NONE };
        OptixBuildInput triangleInput             = {};
        triangleInput.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangleInput.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput.triangleArray.numVertices   = static_cast<uint32_t>( vertices.size() );
        triangleInput.triangleArray.vertexBuffers = &d_vertices;
        triangleInput.triangleArray.flags         = triangleInputFlags;
        triangleInput.triangleArray.numSbtRecords = 1;

        OptixAccelBuildOptions gasAccelOptions = {};
        gasAccelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        gasAccelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( s_context, &gasAccelOptions, &triangleInput, 1, &gasBufferSizes ) );

        LWdeviceptr d_tempBuffer;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_tempBuffer ), gasBufferSizes.tempSizeInBytes ) );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &s_d_gasOutputBuffer ), gasBufferSizes.outputSizeInBytes ) );

        OPTIX_CHECK( optixAccelBuild( s_context, 0, &gasAccelOptions, &triangleInput, 1, d_tempBuffer, gasBufferSizes.tempSizeInBytes,
                                      s_d_gasOutputBuffer, gasBufferSizes.outputSizeInBytes, &s_gasHandle, nullptr, 0 ) );

        LWDA_CHECK( lwdaFree( (void*)d_vertices ) );
        LWDA_CHECK( lwdaFree( (void*)d_tempBuffer ) );

        OptixModuleCompileOptions moduleCompileOptions = {};

        s_pipelineCompileOptions.usesMotionBlur                   = false;
        s_pipelineCompileOptions.numPayloadValues                 = 0;
        s_pipelineCompileOptions.numAttributeValues               = 2;
        s_pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

        OPTIX_CHECK_THROW( optixModuleCreateFromPTX( s_context, &moduleCompileOptions, &s_pipelineCompileOptions,
                                                     optix::data::gettest_DeviceAPI_triangleHitSources()[1],
                                                     optix::data::gettest_DeviceAPI_triangleHitSourceSizes()[0], 0, 0, &s_ptxModule ) );
    }

    static void TearDownTestCase()
    {
        LWDA_CHECK( lwdaFree( (void*)s_d_gasOutputBuffer ) );
        OPTIX_CHECK( optixDeviceContextDestroy( s_context ) );
    }
};

OptixDeviceContext          O7_API_Device_Triangle_Hit::s_context = nullptr;
OptixRecordingLogger        O7_API_Device_Triangle_Hit::s_logger{};
LWdeviceptr                 O7_API_Device_Triangle_Hit::s_d_gasOutputBuffer{};
OptixTraversableHandle      O7_API_Device_Triangle_Hit::s_gasHandle{};
OptixModule                 O7_API_Device_Triangle_Hit::s_ptxModule              = nullptr;
OptixPipelineCompileOptions O7_API_Device_Triangle_Hit::s_pipelineCompileOptions = {};

void O7_API_Device_Triangle_Hit::runTest()
{
    OptixProgramGroupOptions programGroupOptions = {};

    OptixProgramGroupDesc rgProgramGroupDesc    = {};
    rgProgramGroupDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgProgramGroupDesc.raygen.module            = s_ptxModule;
    rgProgramGroupDesc.raygen.entryFunctionName = "__raygen__";
    OptixProgramGroup rgProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &rgProgramGroupDesc, 1, &programGroupOptions, 0, 0, &rgProgramGroup ) );

    OptixProgramGroupDesc exProgramGroupDesc       = {};
    exProgramGroupDesc.kind                        = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    exProgramGroupDesc.exception.module            = s_ptxModule;
    exProgramGroupDesc.exception.entryFunctionName = "__exception__";
    OptixProgramGroup exProgramGroup;
    OPTIX_CHECK_THROW( optixProgramGroupCreate( s_context, &exProgramGroupDesc, 1, &programGroupOptions, 0, 0, &exProgramGroup ) );

    OptixProgramGroupDesc msProgramGroupDesc  = {};
    msProgramGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msProgramGroupDesc.miss.module            = nullptr;
    msProgramGroupDesc.miss.entryFunctionName = nullptr;
    OptixProgramGroup msProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &msProgramGroupDesc, 1, &programGroupOptions, 0, 0, &msProgramGroup ) );

    OptixProgramGroupDesc hitgroupProgramGroupDesc        = {};
    hitgroupProgramGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupProgramGroupDesc.hitgroup.moduleCH            = s_ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__";
    hitgroupProgramGroupDesc.hitgroup.moduleIS            = s_ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameIS = "__intersection__";
    hitgroupProgramGroupDesc.hitgroup.moduleAH            = s_ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameAH = "__anyhit__";
    OptixProgramGroup hitgroupProgramGroup;

    OPTIX_CHECK( optixProgramGroupCreate( s_context, &hitgroupProgramGroupDesc, 1, &programGroupOptions, 0, 0, &hitgroupProgramGroup ) );

    OptixPipeline            pipeline;
    OptixProgramGroup        programGroups[] = { rgProgramGroup, exProgramGroup, msProgramGroup, hitgroupProgramGroup };
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth            = 5;
    pipelineLinkOptions.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    OPTIX_CHECK( optixPipelineCreate( s_context, &s_pipelineCompileOptions, &pipelineLinkOptions, programGroups,
                                      sizeof( programGroups ) / sizeof( programGroups[0] ), 0, 0, &pipeline ) );

    RaygenSbtRecord rgSBT;
    OPTIX_CHECK( optixSbtRecordPackHeader( rgProgramGroup, &rgSBT ) );
    LWdeviceptr d_raygenRecord;
    size_t      raygenRecordSize = sizeof( RaygenSbtRecord );
    LWDA_CHECK( lwdaMalloc( (void**)&d_raygenRecord, raygenRecordSize ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_raygenRecord, &rgSBT, raygenRecordSize, lwdaMemcpyHostToDevice ) );

    ExceptionSbtRecord exSBT;
    OPTIX_CHECK( optixSbtRecordPackHeader( exProgramGroup, &exSBT ) );
    LWdeviceptr d_exceptionRecord;
    size_t      exceptionRecordSize = sizeof( ExceptionSbtRecord );
    LWDA_CHECK( lwdaMalloc( (void**)&d_exceptionRecord, exceptionRecordSize ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_exceptionRecord, &exSBT, exceptionRecordSize, lwdaMemcpyHostToDevice ) );

    MissSbtRecord msSBT;
    OPTIX_CHECK( optixSbtRecordPackHeader( msProgramGroup, &msSBT ) );
    LWdeviceptr d_missSbtRecord;
    size_t      missSbtRecordSize = sizeof( MissSbtRecord );
    LWDA_CHECK( lwdaMalloc( (void**)&d_missSbtRecord, missSbtRecordSize ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_missSbtRecord, &msSBT, missSbtRecordSize, lwdaMemcpyHostToDevice ) );

    HitgroupSbtRecord hgSBT;
    OPTIX_CHECK( optixSbtRecordPackHeader( hitgroupProgramGroup, &hgSBT ) );
    LWdeviceptr d_hitgroupSbtRecord;
    size_t      hitgroupSbtRecordSize = sizeof( HitgroupSbtRecord );
    LWDA_CHECK( lwdaMalloc( (void**)&d_hitgroupSbtRecord, hitgroupSbtRecordSize ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_hitgroupSbtRecord, &hgSBT, hitgroupSbtRecordSize, lwdaMemcpyHostToDevice ) );

    OptixShaderBindingTable sbt     = {};
    sbt.raygenRecord                = d_raygenRecord;
    sbt.exceptionRecord             = 0;  //m_sbtRecord_nullptr ? 0 : d_exceptionRecord;
    sbt.missRecordBase              = d_missSbtRecord;
    sbt.missRecordStrideInBytes     = (unsigned int)sizeof( MissSbtRecord );
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordBase          = d_hitgroupSbtRecord;
    sbt.hitgroupRecordStrideInBytes = (unsigned int)sizeof( HitgroupSbtRecord );
    sbt.hitgroupRecordCount         = 1;

    LWstream stream;
    LWDA_CHECK( lwdaStreamCreate( &stream ) );

#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    SETUP_API_CALL_COVERAGE( m_params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

    m_params.handle = s_gasHandle;

    TriangleHitId      initTriangleHitId = TRIANGLE_HIT_ID_NONE;
    OptixPrimitiveType initPrimitiveType = OPTIX_PRIMITIVE_TYPE_LWSTOM;
    float3             initHitpoint      = { -1.f, -1.f, -1.f };

    LWDA_CHECK( lwdaMalloc( (void**)&m_params.d_triangleHitResultOutPointer, sizeof( TriangleHitId* ) ) );

    LWDA_CHECK( lwdaMemcpy( (void*)m_params.d_triangleHitResultOutPointer, (void*)&initTriangleHitId,
                            sizeof( TriangleHitId ), lwdaMemcpyHostToDevice ) );

    LWDA_CHECK( lwdaMalloc( (void**)&m_params.d_optixPrimitiveTypeResultOutPointer, sizeof( OptixPrimitiveType* ) ) );

    LWDA_CHECK( lwdaMemcpy( (void*)m_params.d_optixPrimitiveTypeResultOutPointer, (void*)&initPrimitiveType,
                            sizeof( OptixPrimitiveType ), lwdaMemcpyHostToDevice ) );

    LWDA_CHECK( lwdaMalloc( (void**)&m_params.d_hitPoint, sizeof( float3 ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)m_params.d_hitPoint, (void*)&initHitpoint, sizeof( float3 ), lwdaMemcpyHostToDevice ) );

    LWdeviceptr d_params;
    LWDA_CHECK( lwdaMalloc( (void**)&d_params, sizeof( Params ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_params, &m_params, sizeof( Params ), lwdaMemcpyHostToDevice ) );

    OPTIX_CHECK( optixLaunch( pipeline, stream, d_params, sizeof( Params ), &sbt, 1, 1, 1 ) );

    LWDA_SYNC_CHECK();

    LWDA_CHECK( lwdaMemcpy( (void*)&m_triangleHitResultOut, (void*)m_params.d_triangleHitResultOutPointer,
                            sizeof( TriangleHitId ), lwdaMemcpyDeviceToHost ) );

    LWDA_CHECK( lwdaMemcpy( (void*)&m_optixPrimitiveTypeResultOut, (void*)m_params.d_optixPrimitiveTypeResultOutPointer,
                            sizeof( OptixPrimitiveType ), lwdaMemcpyDeviceToHost ) );

    LWDA_CHECK( lwdaMemcpy( (void*)&m_hitPoint, (void*)m_params.d_hitPoint, sizeof( float3 ), lwdaMemcpyDeviceToHost ) );

#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    ANALYZE_API_CALL_COVERAGE( m_params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

    LWDA_CHECK( lwdaFree( (void*)m_params.d_triangleHitResultOutPointer ) );
    LWDA_CHECK( lwdaFree( (void*)m_params.d_optixPrimitiveTypeResultOutPointer ) );
    LWDA_CHECK( lwdaFree( (void*)m_params.d_hitPoint ) );
    LWDA_CHECK( lwdaFree( (void*)d_params ) );
    LWDA_CHECK( lwdaFree( (void*)d_hitgroupSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_missSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_exceptionRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_raygenRecord ) );
}

TEST_P( O7_API_Device_Triangle_Hit, OptixIsTriangleHit )
{
    GtestParam param                 = GetParam();
    m_params.optixProgramType        = std::get<0>( param );
    m_params.expectedTriangleHitType = std::get<1>( param );
    m_params.useHitTypeArgument      = std::get<2>( param );

    runTest();

    switch( m_params.expectedTriangleHitType )
    {
        case EXPECTED_TRIANGLE_HIT_TYPE_HIT:
            EXPECT_EQ( TRIANGLE_HIT_ID_HIT, m_triangleHitResultOut );
            break;

        case EXPECTED_TRIANGLE_HIT_TYPE_FRONT_FACE_HIT:
            EXPECT_EQ( TRIANGLE_HIT_ID_FRONT, m_triangleHitResultOut );
            break;

        case EXPECTED_TRIANGLE_HIT_TYPE_BACK_FACE_HIT:
            EXPECT_EQ( TRIANGLE_HIT_ID_BACK, m_triangleHitResultOut );
            break;

        default:
            break;
    }

    EXPECT_FLOAT_EQ( EXPECTED_HIT_POINT.x, m_hitPoint.x );
    EXPECT_FLOAT_EQ( EXPECTED_HIT_POINT.y, m_hitPoint.y );
    EXPECT_FLOAT_EQ( EXPECTED_HIT_POINT.z, m_hitPoint.z );
}

TEST_P( O7_API_Device_Triangle_Hit, OptixGetPrimitiveType )
{
    GtestParam param                 = GetParam();
    m_params.optixProgramType        = std::get<0>( param );
    m_params.expectedTriangleHitType = std::get<1>( param );
    m_params.useHitTypeArgument      = std::get<2>( param );

    runTest();

    EXPECT_EQ( OPTIX_PRIMITIVE_TYPE_TRIANGLE, m_optixPrimitiveTypeResultOut );

    EXPECT_FLOAT_EQ( EXPECTED_HIT_POINT.x, m_hitPoint.x );
    EXPECT_FLOAT_EQ( EXPECTED_HIT_POINT.y, m_hitPoint.y );
    EXPECT_FLOAT_EQ( EXPECTED_HIT_POINT.z, m_hitPoint.z );
}

std::string getOptixProgramTypeName( OptixProgramTypeTriangleHit optixProgramType );
std::string getExpectedTriangleHitTypeName( ExpectedTriangleHitType expectedTriangleHitType );
std::string getUseHitTypeArgumentName( UseHitTypeArgument useHitTypeArgument );

INSTANTIATE_TEST_SUITE_P( RunThroughTriangleHitTypes,
                          O7_API_Device_Triangle_Hit,
                          Combine( Values( PROGRAM_TYPE_ANY_HIT, PROGRAM_TYPE_CLOSEST_HIT ),
                                   Values( EXPECTED_TRIANGLE_HIT_TYPE_HIT, EXPECTED_TRIANGLE_HIT_TYPE_FRONT_FACE_HIT, EXPECTED_TRIANGLE_HIT_TYPE_BACK_FACE_HIT ),
                                   Values( USE_HIT_TYPE_TRIANGLE_IMPLICIT, USE_HIT_TYPE_ARGUMENT, USE_HIT_TYPE_IMPLICIT ) ),
                          []( const testing::TestParamInfo<O7_API_Device_Triangle_Hit::ParamType>& info ) {
                              std::string name = getOptixProgramTypeName( std::get<0>( info.param ) );
                              name += "_";
                              name += getExpectedTriangleHitTypeName( std::get<1>( info.param ) );
                              name += "_";
                              name += getUseHitTypeArgumentName( std::get<2>( info.param ) );
                              return name;
                          } );

std::string getOptixProgramTypeName( OptixProgramTypeTriangleHit optixProgramType )
{
    switch( optixProgramType )
    {
        case PROGRAM_TYPE_ANY_HIT:
            return "OptixProgramTypeAnyHit";
        case PROGRAM_TYPE_CLOSEST_HIT:
            return "OptixProgramTypeClosestHit";
        default:
            return std::to_string( optixProgramType );
    }
}

std::string getExpectedTriangleHitTypeName( ExpectedTriangleHitType expectedTriangleHitType )
{
    switch( expectedTriangleHitType )
    {
        case EXPECTED_TRIANGLE_HIT_TYPE_HIT:
            return "ExpectedTriangleHitTypeHit";
        case EXPECTED_TRIANGLE_HIT_TYPE_FRONT_FACE_HIT:
            return "ExpectedTriangleHitTypeFrontFaceHit";
        case EXPECTED_TRIANGLE_HIT_TYPE_BACK_FACE_HIT:
            return "ExpectedTriangleHitTypeBackFaceHit";
        default:
            return std::to_string( expectedTriangleHitType );
    }
}

std::string getUseHitTypeArgumentName( UseHitTypeArgument useHitTypeArgument )
{
    switch( useHitTypeArgument )
    {
        case USE_HIT_TYPE_TRIANGLE_IMPLICIT:
            return "UseHitTypeTriangleImplicit";
        case USE_HIT_TYPE_ARGUMENT:
            return "UseHitTypeArgument";
        case USE_HIT_TYPE_IMPLICIT:
            return "UseHitTypeImplicit";
        default:
            return std::to_string( useHitTypeArgument );
    }
}
