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

#include "test_Regressions.h"
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include <exptest/coverageFileWriter.h>
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include "tests/sanity/test_Regressions_ptx_bin.h"

using RaygenSbtRecord    = SbtRecord<void>;
using ExceptionSbtRecord = SbtRecord<void>;
using MissSbtRecord      = SbtRecord<void>;
using HitgroupSbtRecord  = SbtRecord<void>;
using CallableSbtRecord  = SbtRecord<void>;

struct O7_API_Regressions : public testing::Test
{
    void SetUp() override
    {
        m_rg = "__raygen__";
        m_ex = "__exception__";
        m_ms = "__miss__";
        m_is = "__intersection__";
        m_ah = "__anyhit__";
        m_ch = "__closesthit__";

        m_width = m_height = m_depth = 1;
    }
    void TearDown() override {}

    void runTest();

    static OptixDeviceContext          s_context;
    static OptixRecordingLogger        s_logger;
    static LWdeviceptr                 s_d_gasOutputBuffer;
    static OptixTraversableHandle      s_gasHandle;
    static OptixModule                 s_ptxModule;
    static OptixPipelineCompileOptions s_pipelineCompileOptions;

    // Default values used by runTest() unless changed upfront
    static std::string s_lwFile;
    std::string        m_rg;
    std::string        m_ex;
    std::string        m_ms;
    std::string        m_is;
    std::string        m_ah;
    std::string        m_ch;
    unsigned int       m_width;
    unsigned int       m_height;
    unsigned int       m_depth;
    std::vector<int>   intResults;

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

        // Compile modules

        OptixModuleCompileOptions moduleCompileOptions = {};

        s_pipelineCompileOptions.usesMotionBlur                   = false;
        s_pipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        s_pipelineCompileOptions.numPayloadValues                 = 8;
        s_pipelineCompileOptions.numAttributeValues               = 2;
        s_pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

        OPTIX_CHECK_THROW( optixModuleCreateFromPTX( s_context, &moduleCompileOptions, &s_pipelineCompileOptions,
                                                     optix::data::gettest_RegressionsSources()[1],
                                                     optix::data::gettest_RegressionsSourceSizes()[0], 0, 0, &s_ptxModule ) );
    }

    static void TearDownTestCase()
    {
        LWDA_CHECK( lwdaFree( (void*)s_d_gasOutputBuffer ) );
        OPTIX_CHECK( optixDeviceContextDestroy( s_context ) );
    }
};

OptixDeviceContext          O7_API_Regressions::s_context = nullptr;
OptixRecordingLogger        O7_API_Regressions::s_logger{};
LWdeviceptr                 O7_API_Regressions::s_d_gasOutputBuffer{};
OptixTraversableHandle      O7_API_Regressions::s_gasHandle{};
OptixModule                 O7_API_Regressions::s_ptxModule              = nullptr;
std::string                 O7_API_Regressions::s_lwFile                 = "test_Regressions.lw";
OptixPipelineCompileOptions O7_API_Regressions::s_pipelineCompileOptions = {};

void O7_API_Regressions::runTest()
{
    // Set up program groups

    OptixProgramGroupOptions programGroupOptions = {};

    OptixProgramGroupDesc rgProgramGroupDesc    = {};
    rgProgramGroupDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgProgramGroupDesc.raygen.module            = s_ptxModule;
    rgProgramGroupDesc.raygen.entryFunctionName = m_rg.c_str();
    OptixProgramGroup rgProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &rgProgramGroupDesc, 1, &programGroupOptions, 0, 0, &rgProgramGroup ) );

    OptixProgramGroupDesc exProgramGroupDesc       = {};
    exProgramGroupDesc.kind                        = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    exProgramGroupDesc.exception.module            = s_ptxModule;
    exProgramGroupDesc.exception.entryFunctionName = m_ex.c_str();
    OptixProgramGroup exProgramGroup;
    OPTIX_CHECK_THROW( optixProgramGroupCreate( s_context, &exProgramGroupDesc, 1, &programGroupOptions, 0, 0, &exProgramGroup ) );

    OptixProgramGroupDesc msProgramGroupDesc  = {};
    msProgramGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msProgramGroupDesc.miss.module            = s_ptxModule;
    msProgramGroupDesc.miss.entryFunctionName = m_ms.c_str();
    OptixProgramGroup msProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &msProgramGroupDesc, 1, &programGroupOptions, 0, 0, &msProgramGroup ) );

    OptixProgramGroupDesc hitgroupProgramGroupDesc        = {};
    hitgroupProgramGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupProgramGroupDesc.hitgroup.moduleCH            = s_ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameCH = m_ch.c_str();
    hitgroupProgramGroupDesc.hitgroup.moduleIS            = s_ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameIS = m_is.c_str();
    hitgroupProgramGroupDesc.hitgroup.moduleAH            = s_ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameAH = m_ah.c_str();
    OptixProgramGroup hitgroupProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &hitgroupProgramGroupDesc, 1, &programGroupOptions, 0, 0, &hitgroupProgramGroup ) );


    // Link pipeline

    OptixPipeline            pipeline;
    OptixProgramGroup        programGroups[] = { rgProgramGroup, exProgramGroup, msProgramGroup, hitgroupProgramGroup };
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth            = 1;
    pipelineLinkOptions.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    OPTIX_CHECK( optixPipelineCreate( s_context, &s_pipelineCompileOptions, &pipelineLinkOptions, programGroups,
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
    LWDA_CHECK( lwdaMalloc( (void**)&params.intResult, m_width * sizeof( int ) ) );

#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    SETUP_API_CALL_COVERAGE( params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    params.handle = s_gasHandle;

    LWdeviceptr d_param;
    LWDA_CHECK( lwdaMalloc( (void**)&d_param, sizeof( Params ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_param, &params, sizeof( Params ), lwdaMemcpyHostToDevice ) );

    // Launch
    OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, m_width, m_height, m_depth ) );
    LWDA_SYNC_CHECK();

    // getting output value back from device
    intResults.resize( m_width );
    LWDA_CHECK( lwdaMemcpy( (void*)intResults.data(), (void*)params.intResult, m_width * sizeof( int ), lwdaMemcpyDeviceToHost ) );
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    ANALYZE_API_CALL_COVERAGE( params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

    LWDA_CHECK( lwdaFree( (void*)d_param ) );
    LWDA_CHECK( lwdaFree( (void*)params.intResult ) );
    LWDA_CHECK( lwdaFree( (void*)d_hitgroupSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_missSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_exceptionRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_raygenRecord ) );
}

TEST_F( O7_API_Regressions, Bug2761527_Broken_mul_hi_s32 )
{
    m_width = 133;
    m_rg    = "__raygen__Bug2761527";
    runTest();
    // Bug 2761527: The modulus operator is returning negative numbers given two positive inputs on device
    for( unsigned int i = 0; i < m_width; ++i )
        ASSERT_EQ( intResults[i], (int)( float( i ) / 12.f ) % 30 );
}
