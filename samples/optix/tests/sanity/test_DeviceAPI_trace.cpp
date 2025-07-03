
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

#include <string>
#include <vector>

#include "CommonAsserts.h"

#include "test_DeviceAPI_trace.h"
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include <exptest/coverageFileWriter.h>
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

// clang-format off
#include "tests/sanity/test_DeviceAPI_trace_0_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_1_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_2_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_3_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_4_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_5_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_6_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_7_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_8_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_9_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_10_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_11_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_12_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_13_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_14_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_15_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_16_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_17_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_18_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_19_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_20_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_21_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_22_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_23_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_24_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_25_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_26_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_27_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_28_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_29_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_30_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_31_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_trace_32_ptx_bin.h"
// clang-format on

using RaygenSbtRecord    = SbtRecord<void>;
using ExceptionSbtRecord = SbtRecord<void>;
using MissSbtRecord      = SbtRecord<void>;
using HitgroupSbtRecord  = SbtRecord<void>;
using CallableSbtRecord  = SbtRecord<void>;

using TraceTestsParam = std::tuple<bool, unsigned int>;

// Use PrintTo instead of operator<<.  Because tuple is in namespace std,
// so argument-dependent lookup looks for std::operator<< and it's undefined
// behavior to lwstomize std::operator<<.
//
void PrintTo( const TraceTestsParam& param, std::ostream* stream )
{
    unsigned int param1 = std::get<1>( param );
    std::string  str0   = std::get<0>( param ) ? "with Geometry" : "without Geometry";
    std::string  str1   = ( param1 ? ( std::to_string( param1 - 1 ) + " payloads" ).c_str() : "no Payloads" );
    *stream << '{' << str0.c_str() << ", " << str1.c_str() << '}';
}

struct O7_API_Device_Trace : public testing::Test, public testing::WithParamInterface<TraceTestsParam>
{
    void SetUp() override
    {
        // reset and initialize payloads each time. In the payload case we will store "program type | payload"
        if( unsigned int paramCount = std::get<1>( GetParam() ) )
        {
            m_payloads = { PAYLOAD_VAL_0,  PAYLOAD_VAL_1,  PAYLOAD_VAL_2,  PAYLOAD_VAL_3,  PAYLOAD_VAL_4,
                           PAYLOAD_VAL_5,  PAYLOAD_VAL_6,  PAYLOAD_VAL_7,  PAYLOAD_VAL_8,  PAYLOAD_VAL_9,
                           PAYLOAD_VAL_10, PAYLOAD_VAL_11, PAYLOAD_VAL_12, PAYLOAD_VAL_13, PAYLOAD_VAL_14,
                           PAYLOAD_VAL_15, PAYLOAD_VAL_16, PAYLOAD_VAL_17, PAYLOAD_VAL_18, PAYLOAD_VAL_19,
                           PAYLOAD_VAL_20, PAYLOAD_VAL_21, PAYLOAD_VAL_22, PAYLOAD_VAL_23, PAYLOAD_VAL_24,
                           PAYLOAD_VAL_25, PAYLOAD_VAL_26, PAYLOAD_VAL_27, PAYLOAD_VAL_28, PAYLOAD_VAL_29,
                           PAYLOAD_VAL_30, PAYLOAD_VAL_31 };
            m_payloads.resize( paramCount );
        }
        else
            // in the payload-less case we will store the program type here
            m_payloads = std::vector<unsigned int>( 1, 0 );

        setupGASandModule();
    }
    void TearDown() override {}

    void runTest();

    static OptixDeviceContext     s_context;
    static OptixRecordingLogger   s_logger;
    static LWdeviceptr            s_d_gasOutputBuffer;
    static OptixTraversableHandle s_gasHandle;
    static OptixModule            s_ptxModule;
    OptixPipelineCompileOptions   m_pipelineCompileOptions;

    // Default values used by runTest() unless changed upfront
    std::string m_rg = "__raygen__";
    std::string m_ex = "__exception__";
    std::string m_ms = "__miss__";
    std::string m_is = "__intersection__";
    std::string m_ah = "__anyhit__";
    std::string m_ch = "__closesthit__";

    std::vector<unsigned int> m_payloads;

    // helper utility to ease testing
    unsigned int getPayloadValue( unsigned int r )
    {
        switch( r )
        {
            case 0:
                return PAYLOAD_VAL_0;
                break;
            case 1:
                return PAYLOAD_VAL_1;
                break;
            case 2:
                return PAYLOAD_VAL_2;
                break;
            case 3:
                return PAYLOAD_VAL_3;
                break;
            case 4:
                return PAYLOAD_VAL_4;
                break;
            case 5:
                return PAYLOAD_VAL_5;
                break;
            case 6:
                return PAYLOAD_VAL_6;
                break;
            case 7:
                return PAYLOAD_VAL_7;
                break;
            case 8:
                return PAYLOAD_VAL_8;
                break;
            case 9:
                return PAYLOAD_VAL_9;
                break;
            case 10:
                return PAYLOAD_VAL_10;
                break;
            case 11:
                return PAYLOAD_VAL_11;
                break;
            case 12:
                return PAYLOAD_VAL_12;
                break;
            case 13:
                return PAYLOAD_VAL_13;
                break;
            case 14:
                return PAYLOAD_VAL_14;
                break;
            case 15:
                return PAYLOAD_VAL_15;
                break;
            case 16:
                return PAYLOAD_VAL_16;
                break;
            case 17:
                return PAYLOAD_VAL_17;
                break;
            case 18:
                return PAYLOAD_VAL_18;
                break;
            case 19:
                return PAYLOAD_VAL_19;
                break;
            case 20:
                return PAYLOAD_VAL_20;
                break;
            case 21:
                return PAYLOAD_VAL_21;
                break;
            case 22:
                return PAYLOAD_VAL_22;
                break;
            case 23:
                return PAYLOAD_VAL_23;
                break;
            case 24:
                return PAYLOAD_VAL_24;
                break;
            case 25:
                return PAYLOAD_VAL_25;
                break;
            case 26:
                return PAYLOAD_VAL_26;
                break;
            case 27:
                return PAYLOAD_VAL_27;
                break;
            case 28:
                return PAYLOAD_VAL_28;
                break;
            case 29:
                return PAYLOAD_VAL_29;
                break;
            case 30:
                return PAYLOAD_VAL_30;
                break;
            case 31:
                return PAYLOAD_VAL_31;
                break;
            default:
                break;
        }
        return PAYLOAD_VALUES_UNDEF;
    }
    std::string getParameterizedName( const std::string& name )
    {
        return name + "_" + std::to_string( std::get<1>( GetParam() ) );
    }

    const char* getPTX( unsigned int v )
    {
        switch( v )
        {
            case 0:
                return optix::data::gettest_DeviceAPI_trace_0Sources()[1];
                break;
            case 1:
                return optix::data::gettest_DeviceAPI_trace_1Sources()[1];
                break;
            case 2:
                return optix::data::gettest_DeviceAPI_trace_2Sources()[1];
                break;
            case 3:
                return optix::data::gettest_DeviceAPI_trace_3Sources()[1];
                break;
            case 4:
                return optix::data::gettest_DeviceAPI_trace_4Sources()[1];
                break;
            case 5:
                return optix::data::gettest_DeviceAPI_trace_5Sources()[1];
                break;
            case 6:
                return optix::data::gettest_DeviceAPI_trace_6Sources()[1];
                break;
            case 7:
                return optix::data::gettest_DeviceAPI_trace_7Sources()[1];
                break;
            case 8:
                return optix::data::gettest_DeviceAPI_trace_8Sources()[1];
                break;
            case 9:
                return optix::data::gettest_DeviceAPI_trace_9Sources()[1];
                break;
            case 10:
                return optix::data::gettest_DeviceAPI_trace_10Sources()[1];
                break;
            case 11:
                return optix::data::gettest_DeviceAPI_trace_11Sources()[1];
                break;
            case 12:
                return optix::data::gettest_DeviceAPI_trace_12Sources()[1];
                break;
            case 13:
                return optix::data::gettest_DeviceAPI_trace_13Sources()[1];
                break;
            case 14:
                return optix::data::gettest_DeviceAPI_trace_14Sources()[1];
                break;
            case 15:
                return optix::data::gettest_DeviceAPI_trace_15Sources()[1];
                break;
            case 16:
                return optix::data::gettest_DeviceAPI_trace_16Sources()[1];
                break;
            case 17:
                return optix::data::gettest_DeviceAPI_trace_17Sources()[1];
                break;
            case 18:
                return optix::data::gettest_DeviceAPI_trace_18Sources()[1];
                break;
            case 19:
                return optix::data::gettest_DeviceAPI_trace_19Sources()[1];
                break;
            case 20:
                return optix::data::gettest_DeviceAPI_trace_20Sources()[1];
                break;
            case 21:
                return optix::data::gettest_DeviceAPI_trace_21Sources()[1];
                break;
            case 22:
                return optix::data::gettest_DeviceAPI_trace_22Sources()[1];
                break;
            case 23:
                return optix::data::gettest_DeviceAPI_trace_23Sources()[1];
                break;
            case 24:
                return optix::data::gettest_DeviceAPI_trace_24Sources()[1];
                break;
            case 25:
                return optix::data::gettest_DeviceAPI_trace_25Sources()[1];
                break;
            case 26:
                return optix::data::gettest_DeviceAPI_trace_26Sources()[1];
                break;
            case 27:
                return optix::data::gettest_DeviceAPI_trace_27Sources()[1];
                break;
            case 28:
                return optix::data::gettest_DeviceAPI_trace_28Sources()[1];
                break;
            case 29:
                return optix::data::gettest_DeviceAPI_trace_29Sources()[1];
                break;
            case 30:
                return optix::data::gettest_DeviceAPI_trace_30Sources()[1];
                break;
            case 31:
                return optix::data::gettest_DeviceAPI_trace_31Sources()[1];
                break;
            case 32:
                return optix::data::gettest_DeviceAPI_trace_32Sources()[1];
                break;
            default:
                return 0;
                break;
        }
    }

    size_t getPTXSize( unsigned int v )
    {
        switch( v )
        {
            case 0:
                return optix::data::gettest_DeviceAPI_trace_0SourceSizes()[0];
                break;
            case 1:
                return optix::data::gettest_DeviceAPI_trace_1SourceSizes()[0];
                break;
            case 2:
                return optix::data::gettest_DeviceAPI_trace_2SourceSizes()[0];
                break;
            case 3:
                return optix::data::gettest_DeviceAPI_trace_3SourceSizes()[0];
                break;
            case 4:
                return optix::data::gettest_DeviceAPI_trace_4SourceSizes()[0];
                break;
            case 5:
                return optix::data::gettest_DeviceAPI_trace_5SourceSizes()[0];
                break;
            case 6:
                return optix::data::gettest_DeviceAPI_trace_6SourceSizes()[0];
                break;
            case 7:
                return optix::data::gettest_DeviceAPI_trace_7SourceSizes()[0];
                break;
            case 8:
                return optix::data::gettest_DeviceAPI_trace_8SourceSizes()[0];
                break;
            case 9:
                return optix::data::gettest_DeviceAPI_trace_9SourceSizes()[0];
                break;
            case 10:
                return optix::data::gettest_DeviceAPI_trace_10SourceSizes()[0];
                break;
            case 11:
                return optix::data::gettest_DeviceAPI_trace_11SourceSizes()[0];
                break;
            case 12:
                return optix::data::gettest_DeviceAPI_trace_12SourceSizes()[0];
                break;
            case 13:
                return optix::data::gettest_DeviceAPI_trace_13SourceSizes()[0];
                break;
            case 14:
                return optix::data::gettest_DeviceAPI_trace_14SourceSizes()[0];
                break;
            case 15:
                return optix::data::gettest_DeviceAPI_trace_15SourceSizes()[0];
                break;
            case 16:
                return optix::data::gettest_DeviceAPI_trace_16SourceSizes()[0];
                break;
            case 17:
                return optix::data::gettest_DeviceAPI_trace_17SourceSizes()[0];
                break;
            case 18:
                return optix::data::gettest_DeviceAPI_trace_18SourceSizes()[0];
                break;
            case 19:
                return optix::data::gettest_DeviceAPI_trace_19SourceSizes()[0];
                break;
            case 20:
                return optix::data::gettest_DeviceAPI_trace_20SourceSizes()[0];
                break;
            case 21:
                return optix::data::gettest_DeviceAPI_trace_21SourceSizes()[0];
                break;
            case 22:
                return optix::data::gettest_DeviceAPI_trace_22SourceSizes()[0];
                break;
            case 23:
                return optix::data::gettest_DeviceAPI_trace_23SourceSizes()[0];
                break;
            case 24:
                return optix::data::gettest_DeviceAPI_trace_24SourceSizes()[0];
                break;
            case 25:
                return optix::data::gettest_DeviceAPI_trace_25SourceSizes()[0];
                break;
            case 26:
                return optix::data::gettest_DeviceAPI_trace_26SourceSizes()[0];
                break;
            case 27:
                return optix::data::gettest_DeviceAPI_trace_27SourceSizes()[0];
                break;
            case 28:
                return optix::data::gettest_DeviceAPI_trace_28SourceSizes()[0];
                break;
            case 29:
                return optix::data::gettest_DeviceAPI_trace_29SourceSizes()[0];
                break;
            case 30:
                return optix::data::gettest_DeviceAPI_trace_30SourceSizes()[0];
                break;
            case 31:
                return optix::data::gettest_DeviceAPI_trace_31SourceSizes()[0];
                break;
            case 32:
                return optix::data::gettest_DeviceAPI_trace_32SourceSizes()[0];
                break;
            default:
                return 0;
                break;
        }
    }

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
    }

    void setupGASandModule()
    {
        if( std::get<0>( GetParam() ) )
        {
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

        // Compile modules
        OptixModuleCompileOptions moduleCompileOptions = {};

        m_pipelineCompileOptions                                  = {};
        m_pipelineCompileOptions.usesMotionBlur                   = false;
        m_pipelineCompileOptions.numPayloadValues                 = std::get<1>( GetParam() );
        m_pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

        OPTIX_CHECK_THROW( optixModuleCreateFromPTX( s_context, &moduleCompileOptions, &m_pipelineCompileOptions,
                                                     getPTX( std::get<1>( GetParam() ) ),
                                                     getPTXSize( std::get<1>( GetParam() ) ), 0, 0, &s_ptxModule ) );
    }

    static void TearDownTestCase()
    {
        LWDA_CHECK( lwdaFree( (void*)s_d_gasOutputBuffer ) );
        OPTIX_CHECK( optixDeviceContextDestroy( s_context ) );
    }

    // Check whether the returned payloads are equal to each initial payload value bitwise OR-ed with program type.
    void checkFinalPayload()
    {
        // with geometry we test that IS() got called, else MS()
        unsigned int programToken = std::get<0>( GetParam() ) ? PROGRAM_TYPE_PAYLOAD_IS : PROGRAM_TYPE_PAYLOAD_MS;
        // special treatment for case w/o any payload - here only either toke for IS or for MS gets set
        if( !std::get<1>( GetParam() ) )
        {
            ASSERT_EQ( programToken, m_payloads[0] );
            return;
        }
        for( unsigned int i = 0; i < std::get<1>( GetParam() ); ++i )
        {
            ASSERT_EQ( getPayloadValue( i ) | programToken, m_payloads[i] );
        }
    }
};

OptixDeviceContext     O7_API_Device_Trace::s_context = nullptr;
OptixRecordingLogger   O7_API_Device_Trace::s_logger{};
LWdeviceptr            O7_API_Device_Trace::s_d_gasOutputBuffer{};
OptixTraversableHandle O7_API_Device_Trace::s_gasHandle{};
OptixModule            O7_API_Device_Trace::s_ptxModule = nullptr;

void O7_API_Device_Trace::runTest()
{
    // Set up program groups

    OptixProgramGroupOptions programGroupOptions = {};

    OptixProgramGroupDesc rgProgramGroupDesc    = {};
    rgProgramGroupDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgProgramGroupDesc.raygen.module            = s_ptxModule;
    rgProgramGroupDesc.raygen.entryFunctionName = m_rg.c_str();
    OptixProgramGroup rgProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &rgProgramGroupDesc, 1, &programGroupOptions, 0, 0, &rgProgramGroup ) );

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

    OptixPipeline     pipeline;
    OptixProgramGroup programGroups[] = { rgProgramGroup, /*exProgramGroup,*/ msProgramGroup /*, hitgroupProgramGroup*/ };
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth            = 1;
    pipelineLinkOptions.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    OPTIX_CHECK( optixPipelineCreate( s_context, &m_pipelineCompileOptions, &pipelineLinkOptions, programGroups,
                                      sizeof( programGroups ) / sizeof( programGroups[0] ), 0, 0, &pipeline ) );

    // Set up SBT records

    // 1 SBT record for RG
    RaygenSbtRecord rgSBT;
    OPTIX_CHECK( optixSbtRecordPackHeader( rgProgramGroup, &rgSBT ) );
    LWdeviceptr d_raygenRecord;
    size_t      raygenRecordSize = sizeof( RaygenSbtRecord );
    LWDA_CHECK( lwdaMalloc( (void**)&d_raygenRecord, raygenRecordSize ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_raygenRecord, &rgSBT, raygenRecordSize, lwdaMemcpyHostToDevice ) );
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
    LWDA_CHECK( lwdaMemcpy( (void*)params.payloads, (void*)&m_payloads[0], m_payloads.size() * sizeof( unsigned int ),
                            lwdaMemcpyHostToDevice ) );
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    SETUP_API_CALL_COVERAGE( params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    params.handle = s_gasHandle;

    LWdeviceptr d_param;
    LWDA_CHECK( lwdaMalloc( (void**)&d_param, sizeof( Params ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_param, &params, sizeof( Params ), lwdaMemcpyHostToDevice ) );

    // Launch
    OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, 1, 1, 1 ) );
    LWDA_SYNC_CHECK();

    // getting output value back from device
    LWDA_CHECK( lwdaMemcpy( (void*)&m_payloads[0], (void*)params.payloads, m_payloads.size() * sizeof( unsigned int ),
                            lwdaMemcpyDeviceToHost ) );
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    ANALYZE_API_CALL_COVERAGE( params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

    LWDA_CHECK( lwdaFree( (void*)d_param ) );
    LWDA_CHECK( lwdaFree( (void*)params.payloads ) );
    LWDA_CHECK( lwdaFree( (void*)d_hitgroupSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_missSbtRecord ) );
    //LWDA_CHECK( lwdaFree( (void*)d_exceptionRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_raygenRecord ) );
}

TEST_P( O7_API_Device_Trace, RunTheTests )
{
    runTest();
    checkFinalPayload();
}

INSTANTIATE_TEST_SUITE_P( RunThroughAllTraceVariants,
                          O7_API_Device_Trace,
                          testing::Combine( testing::Bool(),
                                            testing::Range( static_cast<unsigned int>( 0 ), static_cast<unsigned int>( 33 ) ) ),
                          // support of better parameter naming - requires newer gtest version
                          []( const testing::TestParamInfo<O7_API_Device_Trace::ParamType>& info ) {
                              std::string name = std::get<0>( info.param ) ? "withGeometry" : "withoutGeometry";
                              name += "_";
                              unsigned int payloadCount = std::get<1>( info.param );
                              if( !payloadCount )
                                  name += "noPayloads";
                              else
                                  name += std::to_string( payloadCount ) + "payload" + ( payloadCount > 1 ? "s" : "" );
                              return name;
                          } );
