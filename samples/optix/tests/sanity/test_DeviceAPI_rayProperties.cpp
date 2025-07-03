
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
#include <iostream>

#include <optix.h>
#include <optix_stubs.h>

#include <exptest/exptest.h>
#include <exptest/loggingHelpers.h>
#include <exptest/testTools.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "CommonAsserts.h"

#include "test_DeviceAPI_rayProperties.h"
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include <exptest/coverageFileWriter.h>
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include "tests/sanity/test_DeviceAPI_rayProperties_ptx_bin.h"

using RaygenSbtRecord    = SbtRecord<void>;
using ExceptionSbtRecord = SbtRecord<void>;
using MissSbtRecord      = SbtRecord<void>;
using HitgroupSbtRecord  = SbtRecord<void>;
using CallableSbtRecord  = SbtRecord<void>;

enum RayInformationKind
{
    RAY_INFORMATION_KIND_ORIGIN_AND_DIRECTION,
    RAY_INFORMATION_KIND_FLAGS_AND_VISIBILITY_MASK,
    RAY_INFORMATION_KIND_RAY_EXTENT
    //,
    //RAY_INFORMATION_KIND_RAY_TIME
};

std::string retrieveEntryName( RayInformationKind entry )
{
    switch( entry )
    {
        case RAY_INFORMATION_KIND_ORIGIN_AND_DIRECTION:
            return "originAndDirectionRetrieval";
        case RAY_INFORMATION_KIND_FLAGS_AND_VISIBILITY_MASK:
            return "flagsAndMaskRetrieval";
        case RAY_INFORMATION_KIND_RAY_EXTENT:
            return "rayExtentRetrieval";
        default:
            return "UKNOWN";
    }
}

void PrintTo( RayInformationKind value, std::ostream* str )
{
  *str << retrieveEntryName( value );
}

struct O7_API_Device_RayProperties : public testing::Test
{
    void runTest( int numAttributeValues );

    static OptixDeviceContext     s_context;
    static OptixRecordingLogger   s_logger;
    static LWdeviceptr            s_d_gasOutputBuffer;
    static OptixTraversableHandle s_gasHandle;
    static int                    s_numPayloadValues;
    OptixModule                   m_ptxModule;

    // Default values used by runTest() unless changed upfront
    std::string         m_lwFile;
    std::string         m_rg;
    std::string         m_ex;
    std::string         m_ms;
    std::string         m_is;
    std::string         m_ah;
    std::string         m_ch;
    OptixRayFlags       m_rayFlags;
    OptixVisibilityMask m_visibilityMask;
    float               m_tMin;
    float               m_tMax;

    std::vector<unsigned int> m_payloads;

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

        // GAS

        OptixBuildInput gasInput{};

        OptixAabb aabbs = { -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f };

        LWdeviceptr d_aabbs;
        LWDA_CHECK( lwdaMalloc( (void**)&d_aabbs, 6 * sizeof( float ) ) );
        LWDA_CHECK( lwdaMemcpy( (void*)d_aabbs, &aabbs, 6 * sizeof( float ), lwdaMemcpyHostToDevice ) );

        gasInput.type                               = OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES;
        gasInput.lwstomPrimitiveArray.aabbBuffers   = &d_aabbs;
        gasInput.lwstomPrimitiveArray.numPrimitives = 1;
        gasInput.lwstomPrimitiveArray.strideInBytes = 6 * sizeof( float );

        unsigned int sbtIndexOffsets[] = { 0 };
        LWdeviceptr  d_sbtIndexOffsets;
        size_t       sbtIndexOffsetsSizeInBytes = sizeof( sbtIndexOffsets );
        assert( sbtIndexOffsetsSizeInBytes == 4 );
        LWDA_CHECK( lwdaMalloc( (void**)&d_sbtIndexOffsets, sbtIndexOffsetsSizeInBytes ) );
        LWDA_CHECK( lwdaMemcpy( (void*)d_sbtIndexOffsets, &sbtIndexOffsets[0], sbtIndexOffsetsSizeInBytes, lwdaMemcpyHostToDevice ) );

        unsigned int gasInputFlags[1]                             = { OPTIX_GEOMETRY_FLAG_NONE };
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
    }

    static void TearDownTestCase()
    {
        LWDA_CHECK( lwdaFree( (void*)s_d_gasOutputBuffer ) );
        OPTIX_CHECK( optixDeviceContextDestroy( s_context ) );
    }
};

struct O7_API_Device_RayPropertiesP : public O7_API_Device_RayProperties, public testing::WithParamInterface<RayInformationKind>
{
    std::string getFileName( const std::string& root ) const { return root + retrieveEntryName( GetParam() ); }

    void SetUp() override
    {
        m_rg     = "__raygen__";
        m_ex     = "__exception__nil";
        m_ms     = getFileName( "__miss__" );
        m_is     = getFileName( "__intersection__" );
        m_ah     = getFileName( "__anyhit__" );
        m_ch     = getFileName( "__closesthit__" );
        m_lwFile = "test_DeviceAPI_rayProperties.lw";

        // flags
        if( GetParam() != RAY_INFORMATION_KIND_FLAGS_AND_VISIBILITY_MASK )
        {
            m_rayFlags       = OPTIX_RAY_FLAG_NONE;
            m_visibilityMask = 1;
        }
        else
        {
            m_rayFlags       = OPTIX_RAY_FLAG_ENFORCE_ANYHIT;
            m_visibilityMask = 123;
        }

        // ray extent
        if( GetParam() != RAY_INFORMATION_KIND_RAY_EXTENT )
        {
            m_tMin = 0.f;
            m_tMax = 100.f;
        }
        else
        {
            m_tMin = 0.001f;
            m_tMax = 123.f;
        }

        // reset payloads each time to appropriate size with initial values
        m_payloads.clear();
        m_payloads.resize( s_numPayloadValues, 0 );
    }
};

OptixDeviceContext     O7_API_Device_RayProperties::s_context = nullptr;
OptixRecordingLogger   O7_API_Device_RayProperties::s_logger{};
LWdeviceptr            O7_API_Device_RayProperties::s_d_gasOutputBuffer{};
OptixTraversableHandle O7_API_Device_RayProperties::s_gasHandle{};
// as we are using all for the TestRayDirectionAndOriginInMiss() test case
int O7_API_Device_RayProperties::s_numPayloadValues = 8;

void O7_API_Device_RayProperties::runTest( int numAttributeValues )
{
    // Compile modules

    OptixModuleCompileOptions moduleCompileOptions = {};

    OptixPipelineCompileOptions pipelineCompileOptions      = {};
    pipelineCompileOptions.usesMotionBlur                   = false;
    pipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues                 = s_numPayloadValues;
    pipelineCompileOptions.numAttributeValues               = numAttributeValues;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.exceptionFlags =
#ifdef DEBUG  // Enables debug exceptions during optix launches. This may inlwr significant performance cost and should only be done during development.
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_DEBUG |
#endif
        OPTIX_EXCEPTION_FLAG_NONE;

    OPTIX_CHECK_THROW( optixModuleCreateFromPTX( s_context, &moduleCompileOptions, &pipelineCompileOptions,
                                                 optix::data::gettest_DeviceAPI_rayPropertiesSources()[1],
                                                 optix::data::gettest_DeviceAPI_rayPropertiesSourceSizes()[0], 0, 0, &m_ptxModule ) );

    // Set up program groups

    OptixProgramGroupOptions programGroupOptions = {};

    OptixProgramGroupDesc rgProgramGroupDesc    = {};
    rgProgramGroupDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgProgramGroupDesc.raygen.module            = m_ptxModule;
    rgProgramGroupDesc.raygen.entryFunctionName = m_rg.c_str();
    OptixProgramGroup rgProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &rgProgramGroupDesc, 1, &programGroupOptions, 0, 0, &rgProgramGroup ) );

    OptixProgramGroupDesc exProgramGroupDesc       = {};
    exProgramGroupDesc.kind                        = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    exProgramGroupDesc.exception.module            = m_ptxModule;
    exProgramGroupDesc.exception.entryFunctionName = m_ex.c_str();
    OptixProgramGroup exProgramGroup;
    OPTIX_CHECK_THROW( optixProgramGroupCreate( s_context, &exProgramGroupDesc, 1, &programGroupOptions, 0, 0, &exProgramGroup ) );

    OptixProgramGroupDesc msProgramGroupDesc  = {};
    msProgramGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msProgramGroupDesc.miss.module            = m_ptxModule;
    msProgramGroupDesc.miss.entryFunctionName = m_ms.c_str();
    OptixProgramGroup msProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &msProgramGroupDesc, 1, &programGroupOptions, 0, 0, &msProgramGroup ) );

    OptixProgramGroupDesc hitgroupProgramGroupDesc        = {};
    hitgroupProgramGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupProgramGroupDesc.hitgroup.moduleCH            = m_ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameCH = m_ch.c_str();
    hitgroupProgramGroupDesc.hitgroup.moduleIS            = m_ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameIS = m_is.c_str();
    hitgroupProgramGroupDesc.hitgroup.moduleAH            = m_ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameAH = m_ah.c_str();
    OptixProgramGroup hitgroupProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &hitgroupProgramGroupDesc, 1, &programGroupOptions, 0, 0, &hitgroupProgramGroup ) );


    // Link pipeline

    OptixPipeline            pipeline;
    OptixProgramGroup        programGroups[] = { rgProgramGroup, exProgramGroup, msProgramGroup, hitgroupProgramGroup };
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth            = 1;
    pipelineLinkOptions.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    OPTIX_CHECK( optixPipelineCreate( s_context, &pipelineCompileOptions, &pipelineLinkOptions, programGroups,
                                      sizeof( programGroups ) / sizeof( programGroups[0] ), 0, 0, &pipeline ) );

    // Set up SBT records

    // 1 SBT record for RG
    RaygenSbtRecord rgSBT;
    OPTIX_CHECK( optixSbtRecordPackHeader( rgProgramGroup, &rgSBT ) );
    LWdeviceptr d_raygenRecord;
    size_t      raygenRecordSize = sizeof( RaygenSbtRecord );
    LWDA_CHECK( lwdaMalloc( (void**)&d_raygenRecord, raygenRecordSize ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_raygenRecord, &rgSBT, raygenRecordSize, lwdaMemcpyHostToDevice ) );

    // 1 SBT record for EX
    ExceptionSbtRecord exSBT;
    OPTIX_CHECK( optixSbtRecordPackHeader( exProgramGroup, &exSBT ) );
    LWdeviceptr d_exceptionRecord;
    size_t      exceptionRecordSize = sizeof( ExceptionSbtRecord );
    LWDA_CHECK( lwdaMalloc( (void**)&d_exceptionRecord, exceptionRecordSize ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_exceptionRecord, &exSBT, exceptionRecordSize, lwdaMemcpyHostToDevice ) );

    // 1 SBT record for MS
    MissSbtRecord msSBT;
    OPTIX_CHECK( optixSbtRecordPackHeader( msProgramGroup, &msSBT ) );
    LWdeviceptr d_missSbtRecord;
    size_t      missSbtRecordSize = sizeof( MissSbtRecord );
    LWDA_CHECK( lwdaMalloc( (void**)&d_missSbtRecord, missSbtRecordSize ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_missSbtRecord, &msSBT, missSbtRecordSize, lwdaMemcpyHostToDevice ) );

    // 1 SBT record for CH/AH/IS
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
    Params params;
    LWDA_CHECK( lwdaMalloc( (void**)&params.payloads, m_payloads.size() * sizeof( unsigned int ) ) );
    // just to avoid accessing empty vector via index
    if( !m_payloads.empty() )
        LWDA_CHECK( lwdaMemcpy( (void*)params.payloads, (void*)&m_payloads[0],
                                m_payloads.size() * sizeof( unsigned int ), lwdaMemcpyHostToDevice ) );
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    SETUP_API_CALL_COVERAGE( params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    params.handle         = s_gasHandle;
    params.rayFlags       = m_rayFlags;
    params.visibilityMask = m_visibilityMask;
    params.tMin           = m_tMin;
    params.tMax           = m_tMax;

    LWdeviceptr d_param;
    LWDA_CHECK( lwdaMalloc( (void**)&d_param, sizeof( Params ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_param, &params, sizeof( Params ), lwdaMemcpyHostToDevice ) );

    // Launch
    OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, 1, 1, 1 ) );
    LWDA_SYNC_CHECK();

    // getting output value back from device
    // just to avoid accessing empty vector via index
    if( !m_payloads.empty() )
        LWDA_CHECK( lwdaMemcpy( (void*)&m_payloads[0], (void*)params.payloads,
                                m_payloads.size() * sizeof( unsigned int ), lwdaMemcpyDeviceToHost ) );
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    ANALYZE_API_CALL_COVERAGE( params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

    LWDA_CHECK( lwdaFree( (void*)d_param ) );
    LWDA_CHECK( lwdaFree( (void*)params.payloads ) );
    LWDA_CHECK( lwdaFree( (void*)d_hitgroupSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_missSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_exceptionRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_raygenRecord ) );
}

TEST_P( O7_API_Device_RayPropertiesP, TestInformationRetrieval )
{
    runTest( 6 );

    switch( GetParam() )
    {
        case RAY_INFORMATION_KIND_ORIGIN_AND_DIRECTION:
            // check both ray direction and origin comparisons in different semantic programs
            ASSERT_TRUE( m_payloads[PAYLOAD_IS_DIRECTION] );
            ASSERT_TRUE( m_payloads[PAYLOAD_IS_ORIGIN] );
            ASSERT_TRUE( m_payloads[PAYLOAD_CH_DIRECTION] );
            ASSERT_TRUE( m_payloads[PAYLOAD_CH_ORIGIN] );
            ASSERT_TRUE( m_payloads[PAYLOAD_AH_DIRECTION] );
            ASSERT_TRUE( m_payloads[PAYLOAD_AH_ORIGIN] );
            break;
        case RAY_INFORMATION_KIND_FLAGS_AND_VISIBILITY_MASK:
            // check retrieved data
            ASSERT_EQ( m_rayFlags, m_payloads[PAYLOAD_IS_PROGRAM_FLAGS] );
            ASSERT_EQ( m_visibilityMask, m_payloads[PAYLOAD_IS_PROGRAM_MASK] );
            ASSERT_EQ( m_rayFlags, m_payloads[PAYLOAD_CH_PROGRAM_FLAGS] );
            ASSERT_EQ( m_visibilityMask, m_payloads[PAYLOAD_CH_PROGRAM_MASK] );
            ASSERT_EQ( m_rayFlags, m_payloads[PAYLOAD_AH_PROGRAM_FLAGS] );
            ASSERT_EQ( m_visibilityMask, m_payloads[PAYLOAD_AH_PROGRAM_MASK] );
            break;
        case RAY_INFORMATION_KIND_RAY_EXTENT:
            // Tmin/Tmax checks for correctness done inside the programs, which fill in only true/false
            ASSERT_EQ( 1, m_payloads[PAYLOAD_IS_RAY_EXTENT_TMIN] );
            ASSERT_EQ( 1, m_payloads[PAYLOAD_IS_RAY_EXTENT_TMAX] );
            ASSERT_EQ( 1, m_payloads[PAYLOAD_AH_RAY_EXTENT_TMIN] );
            ASSERT_EQ( 1, m_payloads[PAYLOAD_AH_RAY_EXTENT_TMAX] );
            ASSERT_EQ( 1, m_payloads[PAYLOAD_CH_RAY_EXTENT_TMIN] );
            ASSERT_EQ( 1, m_payloads[PAYLOAD_CH_RAY_EXTENT_TMAX] );
            break;
        default:
            FAIL() << "We shouldn't get here.";
    }
}

TEST_P( O7_API_Device_RayPropertiesP, TestInformationRetrievalInMiss )
{
    m_rg = "__raygen__ilwertedDirection";

    // while we don't need all attributes everywhere, this is fixed inside the one LWCA/PTX input to avoid compilation issues
    runTest( 6 );

    switch( GetParam() )
    {
        case RAY_INFORMATION_KIND_ORIGIN_AND_DIRECTION:
            // check ray direction and origin in miss()
            ASSERT_TRUE( m_payloads[PAYLOAD_IS_DIRECTION] );
            ASSERT_TRUE( m_payloads[PAYLOAD_IS_ORIGIN] );
            ASSERT_TRUE( m_payloads[PAYLOAD_MS_DIRECTION] );
            ASSERT_TRUE( m_payloads[PAYLOAD_MS_ORIGIN] );
            break;
        case RAY_INFORMATION_KIND_FLAGS_AND_VISIBILITY_MASK:
            // check retrieved data
            ASSERT_EQ( m_rayFlags, m_payloads[PAYLOAD_IS_PROGRAM_FLAGS] );
            ASSERT_EQ( m_visibilityMask, m_payloads[PAYLOAD_IS_PROGRAM_MASK] );
            ASSERT_EQ( m_rayFlags, m_payloads[PAYLOAD_MS_PROGRAM_FLAGS] );
            ASSERT_EQ( m_visibilityMask, m_payloads[PAYLOAD_MS_PROGRAM_MASK] );
            break;
        case RAY_INFORMATION_KIND_RAY_EXTENT:
            // Tmin/Tmax checks for correctness done inside the programs, which fill in only true/false
            ASSERT_EQ( 1, m_payloads[PAYLOAD_IS_RAY_EXTENT_TMIN] );
            ASSERT_EQ( 1, m_payloads[PAYLOAD_IS_RAY_EXTENT_TMAX] );
            ASSERT_EQ( 1, m_payloads[PAYLOAD_MS_RAY_EXTENT_TMIN] );
            ASSERT_EQ( 1, m_payloads[PAYLOAD_MS_RAY_EXTENT_TMAX] );
            break;
        default:
            FAIL() << "We shouldn't get here.";
    }
}

INSTANTIATE_TEST_SUITE_P( RunThroughAllRayInformations,
                          O7_API_Device_RayPropertiesP,
                          testing::Values( RAY_INFORMATION_KIND_ORIGIN_AND_DIRECTION,
                                           RAY_INFORMATION_KIND_FLAGS_AND_VISIBILITY_MASK,
                                           RAY_INFORMATION_KIND_RAY_EXTENT ) );
#if 0
                         // support of better parameter naming - requires newer gtest version
                         []( const testing::TestParamInfo<O7_API_Device_RayPropertiesP::ParamType>& info ) {
                             return retrieveEntryName( info.param );
                         }
#endif
