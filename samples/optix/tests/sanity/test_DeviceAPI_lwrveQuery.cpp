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

#include <optix.h>
#include <optix_stubs.h>

#include <exptest/exptest.h>
#include <exptest/loggingHelpers.h>
#include <exptest/testTools.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>

#include "CommonAsserts.h"

#include "test_DeviceAPI_lwrveQuery.h"
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include <exptest/coverageFileWriter.h>
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include "tests/sanity/test_DeviceAPI_lwrveQuery_ptx_bin.h"

// Lwrve Data.
#define PAD .0f

// clang-format off

const LwrveData linearLwrveData = {
    LINEAR,
    { { -4.0f, -4.0f, 0.00001f, PAD },
       {  4.0f,  4.0f, 0.00002f, PAD }
    },
    { { 0.5f, PAD },
       { 0.5f, PAD }
    },
    OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR,
};

const LwrveData quadraticLwrveData = {
    QUADRATIC,
    {  { -4.1f,  0.1f, 0.00001f, PAD },
        {  0.0f,  0.2f, 0.00003f, PAD },
        {  4.1f,  0.3f, 0.00001f, PAD },
    },
    {  { 0.5f, PAD },
        { 0.5f, PAD },
        { 0.5f, PAD }
    },
    OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE,
};

const LwrveData lwbicLwrveData = {
    CUBIC,
    {  { -4.0f,  0.1f, 0.00001f, PAD },
        { -2.0f,  0.2f, 0.00002f, PAD },
        {  2.0f,  0.3f, 0.00003f, PAD },
        {  4.0f,  0.4f, 0.00001f, PAD },
    },
    {  { 0.5f, PAD },
        { 0.5f, PAD },
        { 0.5f, PAD },
        { 0.5f, PAD }
    },
    OPTIX_PRIMITIVE_TYPE_ROUND_LWBIC_BSPLINE,
};

// clang-format on

using RaygenSbtRecord    = SbtRecord<void>;
using ExceptionSbtRecord = SbtRecord<void>;
using MissSbtRecord      = SbtRecord<void>;
using HitgroupSbtRecord  = SbtRecord<void>;
using CallableSbtRecord  = SbtRecord<void>;

using namespace testing;

class O7_API_Device_Lwrve_Query : public testing::Test, public testing::WithParamInterface<OptixProgramTypeLwrveQuery>
{
    public:

    void SetUp() override {}

    void TearDown() override {}

    void runTest();

    static OptixDeviceContext          s_context;
    static OptixRecordingLogger        s_logger;
    static LWdeviceptr                 s_d_gasOutputBuffer;
    static OptixTraversableHandle      s_gasHandle;
    static OptixModule                 s_ptxModule;
    static OptixModule                 s_geometryModule;
    static OptixPipelineCompileOptions s_pipelineCompileOptions;

    Params    m_params;
    LwrveData m_lwrveData  = {};
    float4    m_dataOut[4] = {};
    float     m_lwrveParameterOut;

    static void SetUpTestCase()
    {

        exptest::lwdaInitialize();
        OPTIX_CHECK( optixInit() );

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &OptixRecordingLogger::callback;
        options.logCallbackData           = &s_logger;
        options.logCallbackLevel          = 2;
        LWcontext lwCtx                   = 0;

        OPTIX_CHECK( optixDeviceContextCreate( lwCtx, &options, &s_context ) );
    }

    static void TearDownTestCase()
    {
        LWDA_CHECK( lwdaFree( (void*)s_d_gasOutputBuffer ) );
        OPTIX_CHECK( optixDeviceContextDestroy( s_context ) );
    }
};

OptixDeviceContext          O7_API_Device_Lwrve_Query::s_context = nullptr;
OptixRecordingLogger        O7_API_Device_Lwrve_Query::s_logger{};
LWdeviceptr                 O7_API_Device_Lwrve_Query::s_d_gasOutputBuffer{};
OptixTraversableHandle      O7_API_Device_Lwrve_Query::s_gasHandle{};
OptixModule                 O7_API_Device_Lwrve_Query::s_ptxModule              = nullptr;
OptixModule                 O7_API_Device_Lwrve_Query::s_geometryModule         = nullptr;
OptixPipelineCompileOptions O7_API_Device_Lwrve_Query::s_pipelineCompileOptions = {};

void O7_API_Device_Lwrve_Query::runTest()
{
    const size_t verticesSize = sizeof( float4 ) * m_lwrveData.vertices.size();
    LWdeviceptr  d_vertices   = 0;
    LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_vertices ), verticesSize ) );
    LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_vertices ), m_lwrveData.vertices.data(), verticesSize, lwdaMemcpyHostToDevice ) );

    const size_t widthsSize = sizeof( float ) * m_lwrveData.widths.size() * 2;
    LWdeviceptr  d_widths   = 0;
    LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_widths ), widthsSize ) );
    LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_widths ), m_lwrveData.widths.data(), widthsSize, lwdaMemcpyHostToDevice ) );

    const std::array<int, 1> segmentIndices     = {0};
    const size_t             segmentIndicesSize = sizeof( int ) * segmentIndices.size();
    LWdeviceptr              d_segmentIndices   = 0;
    LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_segmentIndices ), segmentIndicesSize ) );
    LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_segmentIndices ), segmentIndices.data(), segmentIndicesSize,
                            lwdaMemcpyHostToDevice ) );

    OptixBuildInput lwrveInput = {};

    lwrveInput.type                            = OPTIX_BUILD_INPUT_TYPE_LWRVES;
    lwrveInput.lwrveArray.lwrveType            = m_lwrveData.primitiveType;
    lwrveInput.lwrveArray.numPrimitives        = 1;
    lwrveInput.lwrveArray.vertexBuffers        = &d_vertices;
    lwrveInput.lwrveArray.numVertices          = static_cast<uint32_t>( m_lwrveData.vertices.size() );
    lwrveInput.lwrveArray.vertexStrideInBytes  = sizeof( float4 );
    lwrveInput.lwrveArray.widthBuffers         = &d_widths;
    lwrveInput.lwrveArray.widthStrideInBytes   = sizeof( float ) * 2;
    lwrveInput.lwrveArray.normalBuffers        = 0;
    lwrveInput.lwrveArray.normalStrideInBytes  = 0;
    lwrveInput.lwrveArray.indexBuffer          = d_segmentIndices;
    lwrveInput.lwrveArray.indexStrideInBytes   = sizeof( int ) * 2;
    lwrveInput.lwrveArray.flag                 = OPTIX_GEOMETRY_FLAG_NONE;
    lwrveInput.lwrveArray.primitiveIndexOffset = 0;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( s_context, &accelOptions, &lwrveInput,
                                               1,  // Number of build inputs.
                                               &gasBufferSizes ) );
    LWdeviceptr d_tempGasBuffer;
    LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_tempGasBuffer ), gasBufferSizes.tempSizeInBytes ) );

    LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &s_d_gasOutputBuffer ), gasBufferSizes.outputSizeInBytes ) );

    OPTIX_CHECK( optixAccelBuild( s_context, 0, &accelOptions, &lwrveInput, 1, d_tempGasBuffer, gasBufferSizes.tempSizeInBytes,
                                  s_d_gasOutputBuffer, gasBufferSizes.outputSizeInBytes, &s_gasHandle, nullptr, 0 ) );

    LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( d_tempGasBuffer ) ) );
    LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( d_vertices ) ) );
    LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( d_widths ) ) );
    LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( d_segmentIndices ) ) );

    OptixModuleCompileOptions moduleCompileOptions = {};

    s_pipelineCompileOptions.usesMotionBlur                   = false;
    s_pipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    s_pipelineCompileOptions.numPayloadValues                 = 0;
    s_pipelineCompileOptions.numAttributeValues               = 2;
    s_pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    s_pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    s_pipelineCompileOptions.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR | OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE
        | OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LWBIC_BSPLINE | OPTIX_PRIMITIVE_TYPE_FLAGS_LWSTOM;

    OPTIX_CHECK_THROW( optixModuleCreateFromPTX( s_context, &moduleCompileOptions, &s_pipelineCompileOptions,
                                                 optix::data::gettest_DeviceAPI_lwrveQuerySources()[1],
                                                 optix::data::gettest_DeviceAPI_lwrveQuerySourceSizes()[0], 0, 0, &s_ptxModule ) );

    OptixBuiltinISOptions builtinISOptions = {};
    builtinISOptions.builtinISModuleType   = m_lwrveData.primitiveType;
    OPTIX_CHECK( optixBuiltinISModuleGet( s_context, &moduleCompileOptions, &s_pipelineCompileOptions,
                                          &builtinISOptions, &s_geometryModule ) );

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
    msProgramGroupDesc.miss.module            = s_ptxModule;
    msProgramGroupDesc.miss.entryFunctionName = "__miss__";
    OptixProgramGroup msProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &msProgramGroupDesc, 1, &programGroupOptions, 0, 0, &msProgramGroup ) );

    OptixProgramGroupDesc hitgroupProgramGroupDesc        = {};
    hitgroupProgramGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupProgramGroupDesc.hitgroup.moduleCH            = s_ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__";
    hitgroupProgramGroupDesc.hitgroup.moduleIS            = s_geometryModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameIS = 0;  // supplied by module.
    hitgroupProgramGroupDesc.hitgroup.moduleAH            = s_ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameAH = "__anyhit__";
    OptixProgramGroup hitgroupProgramGroup;

    OPTIX_CHECK( optixProgramGroupCreate( s_context, &hitgroupProgramGroupDesc, 1, &programGroupOptions, 0, 0, &hitgroupProgramGroup ) );

    OptixPipeline     pipeline;
    OptixProgramGroup programGroups[] = {rgProgramGroup, exProgramGroup, msProgramGroup, hitgroupProgramGroup};

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
    sbt.exceptionRecord             = d_exceptionRecord;
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

    m_params.gasHandle = s_gasHandle;

    LWDA_CHECK( lwdaMalloc( (void**)&m_params.dataOut, 4 * sizeof( float4 ) ) );
    LWDA_CHECK( lwdaMemset( (void*)m_params.dataOut, 0.f, 4 * sizeof( float4 ) ) );

    LWDA_CHECK( lwdaMalloc( (void**)&m_params.lwrveParameterOut, sizeof( float ) ) );
    LWDA_CHECK( lwdaMemset( (void*)m_params.lwrveParameterOut, 0.f, sizeof( float ) ) );

    LWdeviceptr d_params;
    LWDA_CHECK( lwdaMalloc( (void**)&d_params, sizeof( Params ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_params, &m_params, sizeof( Params ), lwdaMemcpyHostToDevice ) );

    OPTIX_CHECK( optixLaunch( pipeline, stream, d_params, sizeof( Params ), &sbt, 1, 1, 1 ) );

    LWDA_CHECK( lwdaMemcpy( (void*)m_dataOut, (void*)m_params.dataOut, 4 * sizeof( float4 ), lwdaMemcpyDeviceToHost ) );

    LWDA_CHECK( lwdaMemcpy( (void*)&m_lwrveParameterOut, (void*)m_params.lwrveParameterOut, sizeof( float ), lwdaMemcpyDeviceToHost ) );

    LWDA_SYNC_CHECK();

#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    ANALYZE_API_CALL_COVERAGE( m_params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

    LWDA_CHECK( lwdaFree( (void*)m_params.dataOut ) );
    LWDA_CHECK( lwdaFree( (void*)m_params.lwrveParameterOut ) );
    LWDA_CHECK( lwdaFree( (void*)d_params ) );
    LWDA_CHECK( lwdaFree( (void*)d_hitgroupSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_missSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_exceptionRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_raygenRecord ) );
}

TEST_P( O7_API_Device_Lwrve_Query, testOptixGetLinearLwrveData )
{
    m_params                  = {};
    m_lwrveData               = linearLwrveData;
    m_params.optixProgramType = GetParam();
    m_params.optixFunction    = OPTIX_FUNCTION_OPTIX_GET_LINEAR_LWRVE_VERTEX_DATA;

    runTest();

    EXPECT_EQ( m_lwrveData.vertices[0].x, m_dataOut[0].x );
    EXPECT_EQ( m_lwrveData.vertices[0].y, m_dataOut[0].y );
    EXPECT_EQ( m_lwrveData.vertices[0].z, m_dataOut[0].z );
    EXPECT_EQ( m_lwrveData.vertices[1].x, m_dataOut[1].x );
    EXPECT_EQ( m_lwrveData.vertices[1].y, m_dataOut[1].y );
    EXPECT_EQ( m_lwrveData.vertices[1].z, m_dataOut[1].z );
}

TEST_P( O7_API_Device_Lwrve_Query, testOptixGetQuadraticBSplineVertexData )
{
    m_params                  = {};
    m_lwrveData               = quadraticLwrveData;
    m_params.optixProgramType = GetParam();
    m_params.optixFunction    = OPTIX_FUNCTION_OPTIX_GET_QUADRATIC_BSPLINE_VERTEX_DATA;

    runTest();

    EXPECT_EQ( m_lwrveData.vertices[0].x, m_dataOut[0].x );
    EXPECT_EQ( m_lwrveData.vertices[0].y, m_dataOut[0].y );
    EXPECT_EQ( m_lwrveData.vertices[0].z, m_dataOut[0].z );
    EXPECT_EQ( m_lwrveData.vertices[1].x, m_dataOut[1].x );
    EXPECT_EQ( m_lwrveData.vertices[1].y, m_dataOut[1].y );
    EXPECT_EQ( m_lwrveData.vertices[1].z, m_dataOut[1].z );
    EXPECT_EQ( m_lwrveData.vertices[2].x, m_dataOut[2].x );
    EXPECT_EQ( m_lwrveData.vertices[2].y, m_dataOut[2].y );
    EXPECT_EQ( m_lwrveData.vertices[2].z, m_dataOut[2].z );
}

TEST_P( O7_API_Device_Lwrve_Query, testOptixGetLwbicBSplineVertexData )
{
    m_lwrveData               = lwbicLwrveData;
    m_params.optixProgramType = GetParam();
    m_params.optixFunction    = OPTIX_FUNCTION_OPTIX_GET_LWBIC_BSPLINE_VERTEX_DATA;

    runTest();

    EXPECT_EQ( m_lwrveData.vertices[0].x, m_dataOut[0].x );
    EXPECT_EQ( m_lwrveData.vertices[0].y, m_dataOut[0].y );
    EXPECT_EQ( m_lwrveData.vertices[0].z, m_dataOut[0].z );
    EXPECT_EQ( m_lwrveData.vertices[1].x, m_dataOut[1].x );
    EXPECT_EQ( m_lwrveData.vertices[1].y, m_dataOut[1].y );
    EXPECT_EQ( m_lwrveData.vertices[1].z, m_dataOut[1].z );
    EXPECT_EQ( m_lwrveData.vertices[2].x, m_dataOut[2].x );
    EXPECT_EQ( m_lwrveData.vertices[2].y, m_dataOut[2].y );
    EXPECT_EQ( m_lwrveData.vertices[2].z, m_dataOut[2].z );
    EXPECT_EQ( m_lwrveData.vertices[3].x, m_dataOut[3].x );
    EXPECT_EQ( m_lwrveData.vertices[3].y, m_dataOut[3].y );
    EXPECT_EQ( m_lwrveData.vertices[3].z, m_dataOut[3].z );
}

// For each OptiX program
// TODO : Add OPTIX_PROGRAM_TYPE_EXCEPTION once user exception issue is resolved.
INSTANTIATE_TEST_SUITE_P( ForEachValidOptixProgram,
                          O7_API_Device_Lwrve_Query,
                          Values( OPTIX_PROGRAM_TYPE_RAYGEN, OPTIX_PROGRAM_TYPE_ANYHIT, OPTIX_PROGRAM_TYPE_CLOSESTHIT, OPTIX_PROGRAM_TYPE_MISS ) );

// Help GTest pretty print with params when tests fail.
std::string getOptixProgramTypeName( OptixProgramTypeLwrveQuery optixProgramType )
{
    switch( optixProgramType )
    {
        case OPTIX_PROGRAM_TYPE_RAYGEN:
            return "OptixProgramTypeRaygen";
        case OPTIX_PROGRAM_TYPE_INTERSECTION:
            return "OptixProgramTypeIntersection";
        case OPTIX_PROGRAM_TYPE_ANYHIT:
            return "OptixProgramTypeAnyhit";
        case OPTIX_PROGRAM_TYPE_CLOSESTHIT:
            return "OptixProgramTypeClosestHit";
        case OPTIX_PROGRAM_TYPE_MISS:
            return "OptixProgramTypeMiss";
        case OPTIX_PROGRAM_TYPE_EXCEPTION:
            return "OptixProgramTypeException";
        default:
            return std::to_string( optixProgramType );
    }

    return "";
}

void PrintTo( const OptixProgramTypeLwrveQuery programType, std::ostream* stream )
{
    *stream << '{' << getOptixProgramTypeName( programType ) << '}';
}
