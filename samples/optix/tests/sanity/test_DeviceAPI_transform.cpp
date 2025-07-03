
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

#include <vector>

#include "CommonAsserts.h"

using namespace testing;

#include "test_DeviceAPI_transform.h"
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include <exptest/coverageFileWriter.h>
#endif
#include "tests/sanity/test_DeviceAPI_transform_ptx_bin.h"

using RaygenSbtRecord    = SbtRecord<void>;
using ExceptionSbtRecord = SbtRecord<void>;
using MissSbtRecord      = SbtRecord<void>;
using HitgroupSbtRecord  = SbtRecord<void>;
using CallableSbtRecord  = SbtRecord<void>;

using GtestParam = std::tuple<OptixProgramTypeTransform, TransformType>;

struct O7_API_Device_Transform : public testing::Test, public testing::WithParamInterface<GtestParam>
{
    void SetUp() override {}

    void TearDown() override {}

    void runTest();

    static OptixDeviceContext          s_context;
    static OptixRecordingLogger        s_logger;
    static LWdeviceptr                 s_d_gasOutputBuffer;
    static LWdeviceptr                 s_d_iasOutputBuffer;
    static OptixTraversableHandle      s_gasHandle;
    static OptixTraversableHandle      s_iasHandle;
    static OptixModule                 s_ptxModule;
    static OptixPipelineCompileOptions s_pipelineCompileOptions;

    Params m_params;

    float3* m_outputValues;
    float4* m_outputMatrixValues;

    static void SetUpTestCase()
    {
        exptest::lwdaInitialize();
        OPTIX_CHECK( optixInit() );

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &OptixRecordingLogger::callback;
        options.logCallbackData           = &s_logger;
        options.logCallbackLevel          = 3;
        LWcontext lwCtx                   = 0;  // zero means take the current context

        OPTIX_CHECK( optixDeviceContextCreate( lwCtx, &options, &s_context ) );

        OptixBuildInput gasInput{};

        OptixAabb aabbs = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

        LWdeviceptr d_aabbs;
        LWDA_CHECK( lwdaMalloc( (void**)&d_aabbs, 6 * sizeof( float ) ) );
        LWDA_CHECK( lwdaMemcpy( (void*)d_aabbs, &aabbs, 6 * sizeof( float ), lwdaMemcpyHostToDevice ) );

        gasInput.type                               = OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES;
        gasInput.lwstomPrimitiveArray.aabbBuffers   = &d_aabbs;
        gasInput.lwstomPrimitiveArray.numPrimitives = 1;
        gasInput.lwstomPrimitiveArray.strideInBytes = 6 * sizeof( float );

        unsigned int sbtIndexOffsets[] = {0};
        LWdeviceptr  d_sbtIndexOffsets;
        size_t       sbtIndexOffsetsSizeInBytes = sizeof( sbtIndexOffsets );
        assert( sbtIndexOffsetsSizeInBytes == 4 );
        LWDA_CHECK( lwdaMalloc( (void**)&d_sbtIndexOffsets, sbtIndexOffsetsSizeInBytes ) );
        LWDA_CHECK( lwdaMemcpy( (void*)d_sbtIndexOffsets, &sbtIndexOffsets[0], sbtIndexOffsetsSizeInBytes, lwdaMemcpyHostToDevice ) );

        unsigned int gasInputFlags[1]                             = {OPTIX_GEOMETRY_FLAG_NONE};
        gasInput.lwstomPrimitiveArray.flags                       = gasInputFlags;
        gasInput.lwstomPrimitiveArray.numSbtRecords               = 1;
        gasInput.lwstomPrimitiveArray.sbtIndexOffsetBuffer        = d_sbtIndexOffsets;
        gasInput.lwstomPrimitiveArray.sbtIndexOffsetSizeInBytes   = sizeof( unsigned int );
        gasInput.lwstomPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof( unsigned int );

        OptixAccelBuildOptions gasAccelOptions = {};

        gasAccelOptions.buildFlags            = OPTIX_BUILD_FLAG_NONE;
        gasAccelOptions.motionOptions.numKeys = 1;
        gasAccelOptions.operation             = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( s_context, &gasAccelOptions, &gasInput, 1, &gasBufferSizes ) );

        LWdeviceptr d_tempBuffer;
        LWDA_CHECK( lwdaMalloc( (void**)&d_tempBuffer, gasBufferSizes.tempSizeInBytes ) );
        LWDA_CHECK( lwdaMalloc( (void**)&s_d_gasOutputBuffer, gasBufferSizes.outputSizeInBytes ) );

        OPTIX_CHECK( optixAccelBuild( s_context, 0, &gasAccelOptions, &gasInput, 1, d_tempBuffer, gasBufferSizes.tempSizeInBytes,
                                      s_d_gasOutputBuffer, gasBufferSizes.outputSizeInBytes, &s_gasHandle, nullptr, 0 ) );

        LWDA_CHECK( lwdaFree( (void*)d_aabbs ) );
        LWDA_CHECK( lwdaFree( (void*)d_tempBuffer ) );
        LWDA_CHECK( lwdaFree( (void*)d_sbtIndexOffsets ) );

        // Create single instance so we can manipulate its transform

        OptixInstance instance = {};

        memcpy( instance.transform, TEST_TRANSFORM, sizeof( float ) * 12 );

        instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        instance.instanceId        = 0;
        instance.visibilityMask    = 255;
        instance.sbtOffset         = 0;
        instance.traversableHandle = s_gasHandle;

        LWdeviceptr d_instance;
        LWDA_CHECK( lwdaMalloc( (void**)&d_instance, sizeof( OptixInstance ) * 1 ) );
        LWDA_CHECK( lwdaMemcpy( (void*)d_instance, &instance, sizeof( OptixInstance ), lwdaMemcpyHostToDevice ) );

        OptixBuildInput buildInput            = {};
        buildInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        buildInput.instanceArray.instances    = d_instance;
        buildInput.instanceArray.numInstances = 1;

        unsigned int rayTypeCount = 1;
        unsigned int sbtOffset    = 0;

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes iasBufferSizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( s_context, &accelOptions, &buildInput, 1, &iasBufferSizes ) );

        LWdeviceptr d_tempIasBuffer;
        LWDA_CHECK( lwdaMalloc( (void**)&d_tempIasBuffer, iasBufferSizes.tempSizeInBytes ) );

        LWdeviceptr s_d_iasOutputBuffer;
        LWDA_CHECK( lwdaMalloc( (void**)&s_d_iasOutputBuffer, iasBufferSizes.outputSizeInBytes ) );

        OPTIX_CHECK( optixAccelBuild( s_context, nullptr, &accelOptions, &buildInput, 1, d_tempIasBuffer, iasBufferSizes.tempSizeInBytes,
                                      s_d_iasOutputBuffer, iasBufferSizes.outputSizeInBytes, &s_iasHandle, nullptr, 0 ) );

        // Compile modules

        OptixModuleCompileOptions moduleCompileOptions = {};

        s_pipelineCompileOptions.usesMotionBlur        = false;
        s_pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        s_pipelineCompileOptions.numPayloadValues      = 0;
        s_pipelineCompileOptions.numAttributeValues    = 2;
        s_pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

        OPTIX_CHECK_THROW( optixModuleCreateFromPTX( s_context, &moduleCompileOptions, &s_pipelineCompileOptions,
                                                     optix::data::gettest_DeviceAPI_transformSources()[1],
                                                     optix::data::gettest_DeviceAPI_transformSourceSizes()[0], 0, 0, &s_ptxModule ) );

        LWDA_CHECK( lwdaFree( (void*)d_tempIasBuffer ) );
    }

    static void TearDownTestCase()
    {
        LWDA_CHECK( lwdaFree( (void*)s_d_gasOutputBuffer ) );
        OPTIX_CHECK( optixDeviceContextDestroy( s_context ) );
    }

    // Do we have to deal with a transformation or transform matrix retrieval call?
    bool isTransformationCall()
    {
        return ::isTransformationCall( m_params.transformType );
    }

    void CheckOutputValues()
    {
        if( isTransformationCall() )
        {
            bool objectToWorld = m_params.transformType % 2 == 0;  // object to world are even, world to object are odd.

            std::vector<float3> EXPECTED_VALUES;

            // Points
            if( m_params.transformType == TRANSFORM_TYPE_POINT_FROM_OBJECT_TO_WORLD
                || m_params.transformType == TRANSFORM_TYPE_POINT_FROM_WORLD_TO_OBJECT )
            {
                EXPECTED_VALUES = objectToWorld ? EXPECTED_POINT_OBJECT_WORLD_VALUES : EXPECTED_POINT_WORLD_OBJECT_VALUES;
            }
            // Vectors
            else if( m_params.transformType == TRANSFORM_TYPE_VECTOR_FROM_OBJECT_TO_WORLD
                     || m_params.transformType == TRANSFORM_TYPE_VECTOR_FROM_WORLD_TO_OBJECT )
            {
                EXPECTED_VALUES = objectToWorld ? EXPECTED_VECTOR_OBJECT_WORLD_VALUES : EXPECTED_VECTOR_WORLD_OBJECT_VALUES;
            }
            // Normals
            else
            {
                EXPECTED_VALUES = objectToWorld ? EXPECTED_NORMAL_OBJECT_WORLD_VALUES : EXPECTED_NORMAL_WORLD_OBJECT_VALUES;
            }

            for( int i = 0; i < EXPECTED_VALUES.size(); ++i )
            {
                EXPECT_NEAR( m_outputValues[i].x, EXPECTED_VALUES[i].x, EPSILON );
                EXPECT_NEAR( m_outputValues[i].y, EXPECTED_VALUES[i].y, EPSILON );
                EXPECT_NEAR( m_outputValues[i].z, EXPECTED_VALUES[i].z, EPSILON );
            }
        }
        else
        {
            std::vector<float4> EXPECTED_VALUES;
            switch( m_params.transformType )
            {
                // Points
                case TRANSFORM_TYPE_MATRIX_FROM_OBJECT_TO_WORLD:
                    EXPECTED_VALUES = EXPECTED_MATRIX_OBJECT_WORLD_VALUES;
                    break;
                case TRANSFORM_TYPE_MATRIX_FROM_WORLD_TO_OBJECT:
                    EXPECTED_VALUES = EXPECTED_MATRIX_WORLD_OBJECT_VALUES;
                    break;
                default:
                    ASSERT_FALSE( "Wrong TransformType value used" );
                    break;
            }
            for( int i = 0; i < EXPECTED_VALUES.size(); ++i )
            {
                EXPECT_NEAR( m_outputMatrixValues[i].x, EXPECTED_VALUES[i].x, EPSILON );
                EXPECT_NEAR( m_outputMatrixValues[i].y, EXPECTED_VALUES[i].y, EPSILON );
                EXPECT_NEAR( m_outputMatrixValues[i].z, EXPECTED_VALUES[i].z, EPSILON );
                EXPECT_NEAR( m_outputMatrixValues[i].w, EXPECTED_VALUES[i].w, EPSILON );
            }
        }
    }
};

OptixDeviceContext          O7_API_Device_Transform::s_context = nullptr;
OptixRecordingLogger        O7_API_Device_Transform::s_logger{};
LWdeviceptr                 O7_API_Device_Transform::s_d_gasOutputBuffer{};
LWdeviceptr                 O7_API_Device_Transform::s_d_iasOutputBuffer{};
OptixTraversableHandle      O7_API_Device_Transform::s_gasHandle{};
OptixTraversableHandle      O7_API_Device_Transform::s_iasHandle{};
OptixModule                 O7_API_Device_Transform::s_ptxModule              = nullptr;
OptixPipelineCompileOptions O7_API_Device_Transform::s_pipelineCompileOptions = {};

void O7_API_Device_Transform::runTest()
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
    OptixProgramGroup        programGroups[] = {rgProgramGroup, exProgramGroup, msProgramGroup, hitgroupProgramGroup};
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

    // Set up launch
    LWstream stream;
    LWDA_CHECK( lwdaStreamCreate( &stream ) );

    // Set up params

#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    SETUP_API_CALL_COVERAGE( m_params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

    LWDA_CHECK( lwdaMalloc( (void**)&m_params.testValues, TEST_VALUES.size() * sizeof( float3 ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)m_params.testValues, (void*)TEST_VALUES.data(),
                            TEST_VALUES.size() * sizeof( float3 ), lwdaMemcpyHostToDevice ) );
    LWDA_CHECK( lwdaMalloc( (void**)&m_params.outputValues, TEST_VALUES.size() * sizeof( float3 ) ) );
    LWDA_CHECK( lwdaMalloc( (void**)&m_params.outputMatrixValues, TEST_VALUES.size() * sizeof( float4 ) ) );

    m_params.handle        = s_iasHandle;
    m_params.numTestValues = TEST_VALUES.size();

    LWdeviceptr d_params;
    LWDA_CHECK( lwdaMalloc( (void**)&d_params, sizeof( Params ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_params, &m_params, sizeof( Params ), lwdaMemcpyHostToDevice ) );

    // Launch
    OPTIX_CHECK( optixLaunch( pipeline, stream, d_params, sizeof( Params ), &sbt, 1, 1, 1 ) );
    LWDA_SYNC_CHECK();

    m_outputValues = new float3[TEST_VALUES.size()];
    LWDA_CHECK( lwdaMemcpy( (void*)m_outputValues, (void*)m_params.outputValues, TEST_VALUES.size() * sizeof( float3 ),
                            lwdaMemcpyDeviceToHost ) );
    m_outputMatrixValues = new float4[TEST_VALUES.size()];
    LWDA_CHECK( lwdaMemcpy( (void*)m_outputMatrixValues, (void*)m_params.outputMatrixValues,
                            TEST_VALUES.size() * sizeof( float4 ), lwdaMemcpyDeviceToHost ) );

#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    ANALYZE_API_CALL_COVERAGE( m_params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

    LWDA_CHECK( lwdaFree( (void*)d_params ) );
    LWDA_CHECK( lwdaFree( (void*)m_params.testValues ) );
    LWDA_CHECK( lwdaFree( (void*)m_params.outputValues ) );
    LWDA_CHECK( lwdaFree( (void*)m_params.outputMatrixValues ) );
    LWDA_CHECK( lwdaFree( (void*)d_hitgroupSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_missSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_exceptionRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_raygenRecord ) );
}

TEST_P( O7_API_Device_Transform, TestPerformTransform )
{
    GtestParam param       = GetParam();
    m_params               = {0};
    m_params.programType   = std::get<0>( param );
    m_params.transformType = std::get<1>( param );

    runTest();

    CheckOutputValues();
}

INSTANTIATE_TEST_SUITE_P( PerformTransformInAllApplicablePrograms,
                          O7_API_Device_Transform,
                          Combine( Values( OPTIX_PROGRAM_TYPE_INTERSECTION, OPTIX_PROGRAM_TYPE_ANY_HIT, OPTIX_PROGRAM_TYPE_CLOSEST_HIT ),
                                   Values( TRANSFORM_TYPE_POINT_FROM_OBJECT_TO_WORLD,
                                           TRANSFORM_TYPE_POINT_FROM_WORLD_TO_OBJECT,
                                           TRANSFORM_TYPE_VECTOR_FROM_OBJECT_TO_WORLD,
                                           TRANSFORM_TYPE_VECTOR_FROM_WORLD_TO_OBJECT,
                                           TRANSFORM_TYPE_NORMAL_FROM_OBJECT_TO_WORLD,
                                           TRANSFORM_TYPE_NORMAL_FROM_WORLD_TO_OBJECT,
                                           TRANSFORM_TYPE_MATRIX_FROM_OBJECT_TO_WORLD,
                                           TRANSFORM_TYPE_MATRIX_FROM_WORLD_TO_OBJECT ) ) );

std::string getOptixProgramTypeName( OptixProgramTypeTransform optixProgramType )
{
    switch( optixProgramType )
    {
        case OPTIX_PROGRAM_TYPE_INTERSECTION:
            return "OptixProgramTypeIntersection";
        case OPTIX_PROGRAM_TYPE_ANY_HIT:
            return "OptixProgramTypeAnyHit";
        case OPTIX_PROGRAM_TYPE_CLOSEST_HIT:
            return "OptixProgramTypeClosestHit";
        default:
            return std::to_string( optixProgramType );
    }
}

std::string getTransformTypeName( TransformType transformType )
{
    switch( transformType )
    {
        case TRANSFORM_TYPE_POINT_FROM_OBJECT_TO_WORLD:
            return "TransformTypePointFromObjectToWorld";
        case TRANSFORM_TYPE_POINT_FROM_WORLD_TO_OBJECT:
            return "TransformTypePointFromWorldToObject";
        case TRANSFORM_TYPE_VECTOR_FROM_OBJECT_TO_WORLD:
            return "TransformTypeVectorFromObjectToWorld";
        case TRANSFORM_TYPE_VECTOR_FROM_WORLD_TO_OBJECT:
            return "TransformTypeVectorFromWorldToObject";
        case TRANSFORM_TYPE_NORMAL_FROM_OBJECT_TO_WORLD:
            return "TransformTypeNormalFromObjectToWorld";
        case TRANSFORM_TYPE_NORMAL_FROM_WORLD_TO_OBJECT:
            return "TransformTypeNormalFromWorldToObject";
        case TRANSFORM_TYPE_MATRIX_FROM_OBJECT_TO_WORLD:
            return "TransformTypeMatrixFromObjectToWorld";
        case TRANSFORM_TYPE_MATRIX_FROM_WORLD_TO_OBJECT:
            return "TransformTypeMatrixFromWorldToObject";
        default:
            return std::to_string( transformType );
    }
}

void PrintTo( const GtestParam& param, std::ostream* stream )
{
    *stream << '{' << getOptixProgramTypeName( std::get<0>( param ) ) << ", "
            << getTransformTypeName( std::get<1>( param ) ) << '}';
}
