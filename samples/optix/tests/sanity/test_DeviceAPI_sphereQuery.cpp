//
//  Copyright (c) 2021 LWPU Corporation.  All rights reserved.
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

#include "test_DeviceAPI_sphereQuery.h"
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include <exptest/coverageFileWriter.h>
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include "tests/sanity/test_DeviceAPI_sphereQuery_ptx_bin.h"

using RaygenSbtRecord    = SbtRecord<void>;
using ExceptionSbtRecord = SbtRecord<void>;
using MissSbtRecord      = SbtRecord<void>;
using HitgroupSbtRecord  = SbtRecord<void>;
using CallableSbtRecord  = SbtRecord<void>;

using namespace testing;

class O7_API_Device_Sphere_Query : public testing::Test, public testing::WithParamInterface<OptixProgramTypeSphereQuery>
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
    float4    m_dataOut[1] = {};
    float     m_sphereParameterOut;

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

OptixDeviceContext          O7_API_Device_Sphere_Query::s_context = nullptr;
OptixRecordingLogger        O7_API_Device_Sphere_Query::s_logger{};
LWdeviceptr                 O7_API_Device_Sphere_Query::s_d_gasOutputBuffer{};
OptixTraversableHandle      O7_API_Device_Sphere_Query::s_gasHandle{};
OptixModule                 O7_API_Device_Sphere_Query::s_ptxModule              = nullptr;
OptixModule                 O7_API_Device_Sphere_Query::s_geometryModule         = nullptr;
OptixPipelineCompileOptions O7_API_Device_Sphere_Query::s_pipelineCompileOptions = {};

void O7_API_Device_Sphere_Query::runTest()
{
    std::vector<float4> vertices;
    std::vector<float>  radii;

    // vertices and radii with 0-padding

    vertices.push_back( make_float4( 0.f, 0.f, 0.f, 0.f ) );
    vertices.push_back( make_float4( 0.f, 2.f, 0.f, 0.f ) );
    radii.push_back( 0.5f ); radii.push_back( 0.0f );
    radii.push_back( 0.5f ); radii.push_back( 0.0f );

    const size_t verticesSize = sizeof( float4 ) * vertices.size();
    LWdeviceptr  d_vertices   = 0;
    LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_vertices ), verticesSize ) );
    LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_vertices ), vertices.data(), verticesSize, lwdaMemcpyHostToDevice ) );

    const size_t radiiSize = sizeof( float ) * radii.size();
    LWdeviceptr  d_radii   = 0;
    LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_radii ), radiiSize ) );
    LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_radii ), radii.data(), radiiSize, lwdaMemcpyHostToDevice ) );

    const uint32_t  sphereInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

    OptixBuildInput sphereInput = {};

    sphereInput.type                             = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    sphereInput.sphereArray.vertexBuffers        = &d_vertices;
    sphereInput.sphereArray.numVertices          = vertices.size();
    sphereInput.sphereArray.vertexStrideInBytes  = sizeof( float4 );
    sphereInput.sphereArray.radiusBuffers        = &d_radii;
    sphereInput.sphereArray.radiusStrideInBytes  = sizeof( float ) * 2;
    sphereInput.sphereArray.flags                = sphereInputFlags;
    sphereInput.sphereArray.numSbtRecords        = 1;
    sphereInput.sphereArray.primitiveIndexOffset = 0;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( s_context, &accelOptions, &sphereInput,
                                               1,  // Number of build inputs.
                                               &gasBufferSizes ) );
    LWdeviceptr d_tempGasBuffer;
    LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_tempGasBuffer ), gasBufferSizes.tempSizeInBytes ) );

    LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &s_d_gasOutputBuffer ), gasBufferSizes.outputSizeInBytes ) );

    OPTIX_CHECK( optixAccelBuild( s_context, 0, &accelOptions, &sphereInput, 1, d_tempGasBuffer, gasBufferSizes.tempSizeInBytes,
                                  s_d_gasOutputBuffer, gasBufferSizes.outputSizeInBytes, &s_gasHandle, nullptr, 0 ) );

    LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( d_tempGasBuffer ) ) );
    LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( d_vertices ) ) );
    LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( d_radii ) ) );

    OptixModuleCompileOptions moduleCompileOptions = {};

    s_pipelineCompileOptions.usesMotionBlur                   = false;
    s_pipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    s_pipelineCompileOptions.numPayloadValues                 = 0;
    s_pipelineCompileOptions.numAttributeValues               = 1;
    s_pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    s_pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    s_pipelineCompileOptions.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE | OPTIX_PRIMITIVE_TYPE_FLAGS_LWSTOM;

    OPTIX_CHECK_THROW( optixModuleCreateFromPTX( s_context, &moduleCompileOptions, &s_pipelineCompileOptions,
                                                 optix::data::gettest_DeviceAPI_sphereQuerySources()[1],
                                                 optix::data::gettest_DeviceAPI_sphereQuerySourceSizes()[0], 0, 0, &s_ptxModule ) );

    OptixBuiltinISOptions builtinISOptions = {};
    builtinISOptions.builtinISModuleType   = OPTIX_PRIMITIVE_TYPE_SPHERE;
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

    LWDA_CHECK( lwdaMalloc( (void**)&m_params.dataOut, sizeof( float4 ) ) );
    LWDA_CHECK( lwdaMemset( (void*)m_params.dataOut, 0.f, sizeof( float4 ) ) );

    LWdeviceptr d_params;
    LWDA_CHECK( lwdaMalloc( (void**)&d_params, sizeof( Params ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_params, &m_params, sizeof( Params ), lwdaMemcpyHostToDevice ) );

    OPTIX_CHECK( optixLaunch( pipeline, stream, d_params, sizeof( Params ), &sbt, 1, 1, 1 ) );

    LWDA_CHECK( lwdaMemcpy( (void*)m_dataOut, (void*)m_params.dataOut, sizeof( float4 ), lwdaMemcpyDeviceToHost ) );

    LWDA_SYNC_CHECK();

#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    ANALYZE_API_CALL_COVERAGE( m_params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

    LWDA_CHECK( lwdaFree( (void*)m_params.dataOut ) );
    LWDA_CHECK( lwdaFree( (void*)d_params ) );
    LWDA_CHECK( lwdaFree( (void*)d_hitgroupSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_missSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_exceptionRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_raygenRecord ) );
}

TEST_P( O7_API_Device_Sphere_Query, testOptixGetSphereData )
{
    m_params = {};

    runTest();

    float3 vertex = make_float3( 0.0f, 0.0f, 0.0f );
    float  radius = 0.5f;

    EXPECT_EQ( vertex.x, m_dataOut[0].x );
    EXPECT_EQ( vertex.y, m_dataOut[0].y );
    EXPECT_EQ( vertex.z, m_dataOut[0].z );
    EXPECT_EQ( radius, m_dataOut[0].w );
}

// For each OptiX program
INSTANTIATE_TEST_SUITE_P( ForEachValidOptixProgram,
                          O7_API_Device_Sphere_Query,
                          Values( OPTIX_PROGRAM_TYPE_RAYGEN, OPTIX_PROGRAM_TYPE_ANYHIT, OPTIX_PROGRAM_TYPE_CLOSESTHIT, OPTIX_PROGRAM_TYPE_MISS ) );

// Help GTest pretty print with params when tests fail.
std::string getOptixProgramTypeName( OptixProgramTypeSphereQuery optixProgramType )
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

void PrintTo( const OptixProgramTypeSphereQuery programType, std::ostream* stream )
{
    *stream << '{' << getOptixProgramTypeName( programType ) << '}';
}
