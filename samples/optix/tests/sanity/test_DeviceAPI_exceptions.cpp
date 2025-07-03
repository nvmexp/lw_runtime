
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

#include <optix_ext_feature_query.h>
#include <optix_ext_feature_query_function_table_definition.h>
#include <optix_ext_feature_query_stubs.h>

#include <exptest/exptest.h>
#include <exptest/loggingHelpers.h>
#include <exptest/testTools.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "CommonAsserts.h"

#include "test_DeviceAPI_exceptions.h"
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include <exptest/coverageFileWriter.h>
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include "tests/sanity/test_DeviceAPI_exceptions_ptx_bin.h"

using RaygenSbtRecord    = SbtRecord<void>;
using ExceptionSbtRecord = SbtRecord<void>;
using MissSbtRecord      = SbtRecord<int>;
using HitgroupSbtRecord  = SbtRecord<void>;
using CallableSbtRecord  = SbtRecord<void>;

using ExceptionTestsParam = unsigned int;

// Use PrintTo instead of operator<<.  Because tuple is in namespace std,
// so argument-dependent lookup looks for std::operator<< and it's undefined
// behavior to lwstomize std::operator<<.
//
void PrintTo( const ExceptionTestsParam& param, std::ostream* stream )
{
    *stream << '{' << ( std::to_string( param ) + " payloads" ) << '}';
}

#define LINE_INFO_BUFFER_SIZE 1024
// for testing OPTIX_EXCEPTION_CODE_ILWALID_RAY
#define RAYDATA_FLOAT_COUNT 9
// for testing OPTIX_EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH
#define PARAM_MISMATCH_INT_COUNT 3
#define PARAM_MISMATCH_STRING_LENGTH 256
#define PARAM_MISMATCH_CORRECT_ARGS_COUNT 1
#define PARAM_MISMATCH_UNDERFLOW_ARGS_COUNT 0
#define PARAM_MISMATCH_OVERFLOW_ARGS_COUNT 2
#define PARAM_MISMATCH_DC_CALLABLE_NAME "__direct_callable__param_mismatch"
#define PARAM_MISMATCH_CC_CALLABLE_NAME "__continuation_callable__param_mismatch"

struct O7_API_Device_Exceptions : public testing::Test
{
    void SetUp() override
    {
        // reset payloads each time
        m_payloads = {PAYLOAD_VAL_0, PAYLOAD_VAL_1, PAYLOAD_VAL_2, PAYLOAD_VAL_3,
                      PAYLOAD_VAL_4, PAYLOAD_VAL_5, PAYLOAD_VAL_6, PAYLOAD_VAL_7};
    }
    void TearDown() override {}

    void runTest();

    static OptixDeviceContext     s_context;
    static OptixRecordingLogger   s_logger;
    static LWdeviceptr            s_d_gasOutputBuffer;
    static OptixTraversableHandle s_gasHandle;

    // Default values used by runTest() unless changed upfront
    std::string m_rg = "__raygen__";
    std::string m_ex = "__exception__";
    std::string m_ms = "__miss__";
    std::string m_is = "__intersection__";
    std::string m_ah = "__anyhit__";
    std::string m_ch = "__closesthit__";
    std::string m_dc = "__direct_callable__";
    std::string m_cc = "__continuation_callable__";

    std::vector<unsigned int> m_payloads;
    Params                    m_params{};
    unsigned int              m_sbtOffset    = 0;
    unsigned int              m_missSBTIndex = 0;
    std::string               m_lineInfo;
    int                       m_callSiteLine = -1;
    // values will be passed to optixTrace() call
    float3 m_origin    = make_float3( 0.0f, 0.0f, 1.0f );
    float3 m_direction = make_float3( 0.0f, 0.0f, 1.0f );
    float  m_tmin      = 0.0f;
    float  m_tmax      = 100.0f;
    float  m_rayTime   = 0.f;
    // allows for testing both with and w/o motion in pipeline compile options
    bool  m_hasMotion                                           = false;
    float m_rayData[RAYDATA_FLOAT_COUNT]                        = {};
    int   m_missmatchIntData[PARAM_MISMATCH_INT_COUNT]          = {};
    char  m_missmatchCallableName[PARAM_MISMATCH_STRING_LENGTH] = {};
    int   m_missmatchCallableNameLength                         = 0;
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
            default:
                break;
        }
        return PAYLOAD_VALUES_UNDEF;
    }

    static void SetUpTestCase()
    {
        exptest::lwdaInitialize();
        void* handle;
        OPTIX_CHECK( optixInitWithHandle( &handle ) );
        OPTIX_CHECK( optixExtFeatureQueryInit( handle ) );

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &OptixRecordingLogger::callback;
        options.logCallbackData           = &s_logger;
        options.logCallbackLevel          = 3;
        LWcontext lwCtx                   = 0;  // zero means take the current context
        OPTIX_CHECK( optixDeviceContextCreate( lwCtx, &options, &s_context ) );

        // GAS

        OptixBuildInput gasInput{};

        OptixAabb aabbs = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

        LWdeviceptr d_aabbs;
        LWDA_CHECK( lwdaMalloc( (void**)&d_aabbs, 6 * sizeof( float ) ) );
        LWDA_CHECK( lwdaMemcpy( (void*)d_aabbs, &aabbs, 6 * sizeof( float ), lwdaMemcpyHostToDevice ) );

        gasInput.type                               = OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES;
        gasInput.lwstomPrimitiveArray.aabbBuffers   = &d_aabbs;
        gasInput.lwstomPrimitiveArray.numPrimitives = 1;
        gasInput.lwstomPrimitiveArray.strideInBytes = 6 * sizeof( float );
        unsigned int sbtIndexOffsets[]              = {0};
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
        OptixAccelBuildOptions gasAccelOptions                    = {};
        gasAccelOptions.buildFlags                                = OPTIX_BUILD_FLAG_NONE;
        gasAccelOptions.motionOptions.numKeys                     = 1;
        gasAccelOptions.operation                                 = OPTIX_BUILD_OPERATION_BUILD;
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

    void checkFinalPayload( unsigned int val ) { ASSERT_EQ( m_payloads[0], val ); }
};
OptixDeviceContext     O7_API_Device_Exceptions::s_context = nullptr;
OptixRecordingLogger   O7_API_Device_Exceptions::s_logger{};
LWdeviceptr            O7_API_Device_Exceptions::s_d_gasOutputBuffer{};
OptixTraversableHandle O7_API_Device_Exceptions::s_gasHandle{};

struct O7_API_Device_ExceptionsP : public O7_API_Device_Exceptions, public testing::WithParamInterface<ExceptionTestsParam>
{
    void SetUp() override
    {
        // reset payloads each time
        m_payloads = {PAYLOAD_VAL_0, PAYLOAD_VAL_1, PAYLOAD_VAL_2, PAYLOAD_VAL_3,
                      PAYLOAD_VAL_4, PAYLOAD_VAL_5, PAYLOAD_VAL_6, PAYLOAD_VAL_7};
        m_ex = getParameterizedName( "__exception__" );
    }
    std::string getParameterizedName( const std::string& name ) { return name + std::to_string( GetParam() ); }
    // Check whether the returned payloads are equal to each initial payload value bitwise OR-ed with val.
    void checkFinalPayload( unsigned int val )
    {
        if( !GetParam() )
            ASSERT_EQ( m_payloads[0], val );
        else
        {
            for( unsigned int i = 0; i < GetParam(); ++i )
                ASSERT_EQ( getPayloadValue( i ) | val, m_payloads[i] );
        }
    }
};

using IlwalidRayTestParams = std::tuple<int, float>;

// Pass invalid float values to optixTrace() and check that the exception program has set payload[0] appropriately.
struct O7_API_Device_ExceptionsIlwalidRayP : public O7_API_Device_Exceptions, public testing::WithParamInterface<IlwalidRayTestParams>
{
    void SetUp() override
    {
        m_payloads = {
            0, 0, 0, 0, 0, 0, 0, 0,
        };
    }
    // m_rayData contains now all data returned by optixGetExceptionIlwalidRay(), the invalid value
    // at position std::get<0>( GetParam() ) with value std::get<1>( GetParam() ). All other values
    // should be as initially configured.
    void checkFinalPayload()
    {
        int    positionIlwalidValue = std::get<0>( GetParam() );
        float* input                = reinterpret_cast<float*>( &m_origin );
        for( unsigned int i = 0; i < RAYDATA_FLOAT_COUNT; ++i )
        {
            if( i == positionIlwalidValue )
            {
                if( std::isnan( std::get<1>( GetParam() ) ) )
                    ASSERT_TRUE( std::isnan( m_rayData[i] ) );
                else
                    ASSERT_TRUE( std::isinf( m_rayData[i] ) );
            }
            else
                ASSERT_EQ( m_rayData[i], input[i] );
        }
    }
};

enum CALLABLE_TYPE
{
    directCallableType,
    continuationCallableType
};
enum PARAM_MISMATCH_TYPE
{
    underflowParamMismatchType,
    overflowParamMismatchType
};

using ParamsMismatchTestParams = std::tuple<CALLABLE_TYPE, PARAM_MISMATCH_TYPE>;

// Pass invalid float values to optixTrace() and check that the exception program has set payload[0] appropriately.
struct O7_API_Device_ExceptionsParamsMismatchP : public O7_API_Device_Exceptions,
                                                 public testing::WithParamInterface<ParamsMismatchTestParams>
{
    void SetUp() override
    {
        m_payloads = {
            0, 0, 0, 0, 0, 0, 0, 0,
        };
        // set up appropriate raygen program
        switch( std::get<0>( GetParam() ) )
        {
            case directCallableType:
                if( std::get<1>( GetParam() ) == underflowParamMismatchType )
                    m_rg = "__raygen__dc_missingArgs";
                else
                    m_rg = "__raygen__dc_tooManyArgs";
                break;
            case continuationCallableType:
                if( std::get<1>( GetParam() ) == underflowParamMismatchType )
                    m_rg = "__raygen__cc_missingArgs";
                else
                    m_rg = "__raygen__cc_tooManyArgs";
                break;
        }
        m_dc = "__direct_callable__param_mismatch";
        m_cc = "__continuation_callable__param_mismatch";
    }
};

void O7_API_Device_Exceptions::runTest()
{
    OptixPipelineCompileOptions pipelineCompileOptions{};
    OptixModule                 ptxModule{nullptr};
    // Compile modules
    OptixModuleCompileOptions moduleCompileOptions          = {};
    pipelineCompileOptions.usesMotionBlur                   = m_hasMotion;
    pipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numAttributeValues               = 2;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH
                                            | OPTIX_EXCEPTION_FLAG_USER | OPTIX_EXCEPTION_FLAG_DEBUG;
    OPTIX_CHECK_THROW( optixModuleCreateFromPTX( s_context, &moduleCompileOptions, &pipelineCompileOptions,
                                                 optix::data::gettest_DeviceAPI_exceptionsSources()[1],
                                                 optix::data::gettest_DeviceAPI_exceptionsSourceSizes()[0], 0, 0, &ptxModule ) );
    // Set up program groups

    OptixProgramGroupOptions programGroupOptions = {};
    OptixProgramGroupDesc    rgProgramGroupDesc  = {};
    rgProgramGroupDesc.kind                      = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgProgramGroupDesc.raygen.module             = ptxModule;
    rgProgramGroupDesc.raygen.entryFunctionName  = m_rg.c_str();
    OptixProgramGroup rgProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &rgProgramGroupDesc, 1, &programGroupOptions, 0, 0, &rgProgramGroup ) );

    OptixProgramGroupDesc exProgramGroupDesc       = {};
    exProgramGroupDesc.kind                        = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    exProgramGroupDesc.exception.module            = ptxModule;
    exProgramGroupDesc.exception.entryFunctionName = m_ex.c_str();
    OptixProgramGroup exProgramGroup;
    OPTIX_CHECK_THROW( optixProgramGroupCreate( s_context, &exProgramGroupDesc, 1, &programGroupOptions, 0, 0, &exProgramGroup ) );

    OptixProgramGroupDesc msProgramGroupDesc  = {};
    msProgramGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msProgramGroupDesc.miss.module            = ptxModule;
    msProgramGroupDesc.miss.entryFunctionName = m_ms.c_str();
    OptixProgramGroup msProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &msProgramGroupDesc, 1, &programGroupOptions, 0, 0, &msProgramGroup ) );

    OptixProgramGroupDesc hitgroupProgramGroupDesc        = {};
    hitgroupProgramGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupProgramGroupDesc.hitgroup.moduleCH            = ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameCH = m_ch.c_str();
    hitgroupProgramGroupDesc.hitgroup.moduleIS            = ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameIS = m_is.c_str();
    hitgroupProgramGroupDesc.hitgroup.moduleAH            = ptxModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameAH = m_ah.c_str();
    OptixProgramGroup hitgroupProgramGroup;
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &hitgroupProgramGroupDesc, 1, &programGroupOptions, 0, 0, &hitgroupProgramGroup ) );

    // use three program groups for the callables to ensure that the returned SBT index by the optixGetExceptionParameterMismatch is meaningful
    // index 0: unused
    // index 1: DC
    // index 2: CC
    OptixProgramGroupDesc callableProgramGroupDesc0         = {};
    callableProgramGroupDesc0.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callableProgramGroupDesc0.callables.moduleCC            = ptxModule;
    callableProgramGroupDesc0.callables.entryFunctionNameCC = "__continuation_callable__param_mismatch__unused";
    callableProgramGroupDesc0.callables.moduleDC            = ptxModule;
    callableProgramGroupDesc0.callables.entryFunctionNameDC = "__direct_callable__param_mismatch__unused";
    OptixProgramGroup callableProgramGroup0                 = {};
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &callableProgramGroupDesc0, 1, &programGroupOptions, 0, 0, &callableProgramGroup0 ) );

    OptixProgramGroupDesc callableProgramGroupDesc1         = {};
    callableProgramGroupDesc1.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callableProgramGroupDesc1.callables.moduleDC            = ptxModule;
    callableProgramGroupDesc1.callables.entryFunctionNameDC = m_dc.c_str();
    OptixProgramGroup callableProgramGroup1                 = {};
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &callableProgramGroupDesc1, 1, &programGroupOptions, 0, 0, &callableProgramGroup1 ) );

    OptixProgramGroupDesc callableProgramGroupDesc2         = {};
    callableProgramGroupDesc2.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callableProgramGroupDesc2.callables.moduleCC            = ptxModule;
    callableProgramGroupDesc2.callables.entryFunctionNameCC = m_cc.c_str();
    OptixProgramGroup callableProgramGroup2                 = {};
    OPTIX_CHECK( optixProgramGroupCreate( s_context, &callableProgramGroupDesc2, 1, &programGroupOptions, 0, 0, &callableProgramGroup2 ) );


    // Link pipeline

    OptixPipeline     pipeline;
    OptixProgramGroup programGroups[] = {rgProgramGroup,       exProgramGroup,        msProgramGroup,
                                         hitgroupProgramGroup, callableProgramGroup0, callableProgramGroup1,
                                         callableProgramGroup2};
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

    // 3 SBT records for CC/DC: one unused one, one for DC, one for CC
    LWdeviceptr d_callableSbtRecord;
    size_t      callableSbtRecordSize = sizeof( CallableSbtRecord );
    LWDA_CHECK( lwdaMalloc( (void**)&d_callableSbtRecord, callableSbtRecordSize * 3 ) );
    CallableSbtRecord callableSBT[3]{};
    OPTIX_CHECK( optixSbtRecordPackHeader( callableProgramGroup0, &callableSBT[0] ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( callableProgramGroup1, &callableSBT[1] ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( callableProgramGroup2, &callableSBT[2] ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_callableSbtRecord, &callableSBT, callableSbtRecordSize * 3, lwdaMemcpyHostToDevice ) );

    OptixShaderBindingTable sbt      = {};
    sbt.raygenRecord                 = d_raygenRecord;
    sbt.exceptionRecord              = d_exceptionRecord;
    sbt.missRecordBase               = d_missSbtRecord;
    sbt.missRecordStrideInBytes      = (unsigned int)sizeof( MissSbtRecord );
    sbt.missRecordCount              = 1;
    sbt.hitgroupRecordBase           = d_hitgroupSbtRecord;
    sbt.hitgroupRecordStrideInBytes  = (unsigned int)sizeof( HitgroupSbtRecord );
    sbt.hitgroupRecordCount          = 1;
    sbt.callablesRecordBase          = d_callableSbtRecord;
    sbt.callablesRecordStrideInBytes = (unsigned int)sizeof( CallableSbtRecord );
    sbt.callablesRecordCount         = 3;

    // Set up launch
    LWstream stream;
    LWDA_CHECK( lwdaStreamCreate( &stream ) );

    // Set up params
    LWDA_CHECK( lwdaMalloc( (void**)&m_params.payloads, m_payloads.size() * sizeof( unsigned int ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)m_params.payloads, (void*)&m_payloads[0], m_payloads.size() * sizeof( unsigned int ),
                            lwdaMemcpyHostToDevice ) );
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    SETUP_API_CALL_COVERAGE( m_params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    m_params.handle       = s_gasHandle;
    m_params.sbtOffset    = m_sbtOffset;
    m_params.missSBTIndex = m_missSBTIndex;
    LWDA_CHECK( lwdaMalloc( (void**)&m_params.rayData, RAYDATA_FLOAT_COUNT * sizeof( float ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)m_params.rayData, (void*)&m_origin, RAYDATA_FLOAT_COUNT * sizeof( float ), lwdaMemcpyHostToDevice ) );

    LWDA_CHECK( lwdaMalloc( (void**)&m_params.paramMismatchCallableName, PARAM_MISMATCH_STRING_LENGTH * sizeof( char ) ) );
    LWDA_CHECK( lwdaMemset( (void*)m_params.paramMismatchCallableName, 0, PARAM_MISMATCH_STRING_LENGTH * sizeof( char ) ) );
    m_params.paramMismatchCallableNameLength = PARAM_MISMATCH_STRING_LENGTH;
    LWDA_CHECK( lwdaMalloc( (void**)&m_params.paramMismatchIntData, PARAM_MISMATCH_INT_COUNT * sizeof( int ) ) );

    LWDA_CHECK( lwdaMalloc( (void**)&m_params.lineInfo, LINE_INFO_BUFFER_SIZE * sizeof( char ) ) );
    LWDA_CHECK( lwdaMemset( (void*)m_params.lineInfo, 0, LINE_INFO_BUFFER_SIZE * sizeof( char ) ) );
    m_params.lineInfoLength = LINE_INFO_BUFFER_SIZE;
    LWDA_CHECK( lwdaMalloc( (void**)&m_params.callSiteLine, sizeof( int ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)m_params.callSiteLine, &m_callSiteLine, sizeof( int ), lwdaMemcpyHostToDevice ) );

    LWdeviceptr d_param;
    LWDA_CHECK( lwdaMalloc( (void**)&d_param, sizeof( Params ) ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_param, &m_params, sizeof( Params ), lwdaMemcpyHostToDevice ) );

    // Launch
    OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, 1, 1, 1 ) );
    LWDA_SYNC_CHECK();

    // getting output value back from device
    LWDA_CHECK( lwdaMemcpy( (void*)&m_payloads[0], (void*)m_params.payloads, m_payloads.size() * sizeof( unsigned int ),
                            lwdaMemcpyDeviceToHost ) );
    LWDA_CHECK( lwdaMemcpy( (void*)m_rayData, (void*)m_params.rayData, RAYDATA_FLOAT_COUNT * sizeof( float ), lwdaMemcpyDeviceToHost ) );
    LWDA_CHECK( lwdaMemcpy( (void*)m_missmatchIntData, (void*)m_params.paramMismatchIntData,
                            PARAM_MISMATCH_INT_COUNT * sizeof( int ), lwdaMemcpyDeviceToHost ) );
    LWDA_CHECK( lwdaMemcpy( m_missmatchCallableName, (void*)m_params.paramMismatchCallableName,
                            PARAM_MISMATCH_STRING_LENGTH * sizeof( char ), lwdaMemcpyDeviceToHost ) );
    LWDA_CHECK( lwdaMemcpy( (void*)&m_callSiteLine, (void*)m_params.callSiteLine, sizeof( int ), lwdaMemcpyDeviceToHost ) );
    char lineInfo[LINE_INFO_BUFFER_SIZE];
    LWDA_CHECK( lwdaMemcpy( lineInfo, (void*)m_params.lineInfo, LINE_INFO_BUFFER_SIZE * sizeof( char ), lwdaMemcpyDeviceToHost ) );
    if( lineInfo[0] )
    {
        m_lineInfo = lineInfo;
    }

#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    ANALYZE_API_CALL_COVERAGE( m_params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

    // clean-up
    LWDA_CHECK( lwdaFree( (void*)d_param ) );
    LWDA_CHECK( lwdaFree( (void*)m_params.payloads ) );
    LWDA_CHECK( lwdaFree( (void*)m_params.rayData ) );
    LWDA_CHECK( lwdaFree( (void*)m_params.paramMismatchIntData ) );
    LWDA_CHECK( lwdaFree( (void*)m_params.paramMismatchCallableName ) );
    LWDA_CHECK( lwdaFree( (void*)m_params.lineInfo ) );
    LWDA_CHECK( lwdaFree( (void*)d_callableSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_hitgroupSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_missSbtRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_exceptionRecord ) );
    LWDA_CHECK( lwdaFree( (void*)d_raygenRecord ) );
}

TEST_F( O7_API_Device_Exceptions, IlwalidTraversalHitSBTOffset )
{
    // note that this has to be some number < 16 (ie, 8 bit)
    m_sbtOffset = 11;
    runTest();
    checkFinalPayload( m_sbtOffset );
}
TEST_F( O7_API_Device_Exceptions, IlwalidMissSBTIndex )
{
    // note that this has to be some number < 16 (ie, 8 bit)
    m_missSBTIndex = 14;
    m_rg           = "__raygen__miss";
    runTest();
    checkFinalPayload( m_missSBTIndex );
}
TEST_F( O7_API_Device_Exceptions, IlwalidTraversableHandle )
{
    unsigned int ilwalidHandle = 1;
    s_gasHandle                = static_cast<OptixTraversableHandle>( ilwalidHandle );
    runTest();
    checkFinalPayload( ilwalidHandle );
}

TEST_P( O7_API_Device_ExceptionsIlwalidRayP, RunChecks )
{
    // treat all input values as array of floats and set one array element to the given invalid value
    float* floatValues                     = reinterpret_cast<float*>( &m_origin );
    floatValues[std::get<0>( GetParam() )] = std::get<1>( GetParam() );
    runTest();
    checkFinalPayload();
}

TEST_P( O7_API_Device_ExceptionsP, ExceptionInRaygen )
{
    m_rg = getParameterizedName( "__raygen__withThrow_" );
    runTest();
    checkFinalPayload( PROGRAM_TYPE_PAYLOAD_RG );
}
TEST_P( O7_API_Device_ExceptionsP, ExceptionInIntersection )
{
    m_is = getParameterizedName( "__intersection__withThrow_" );
    runTest();
    checkFinalPayload( PROGRAM_TYPE_PAYLOAD_IS );
}
TEST_P( O7_API_Device_ExceptionsP, ExceptionInAnyhit )
{
    m_ah = getParameterizedName( "__anyhit__withThrow_" );
    runTest();
    checkFinalPayload( PROGRAM_TYPE_PAYLOAD_AH );
}
TEST_P( O7_API_Device_ExceptionsP, ExceptionInClosesthit )
{
    m_ch = getParameterizedName( "__closesthit__withThrow_" );
    runTest();
    checkFinalPayload( PROGRAM_TYPE_PAYLOAD_CH );
}
TEST_P( O7_API_Device_ExceptionsP, ExceptionInMiss )
{
    m_rg = "__raygen__miss";
    m_ms = getParameterizedName( "__miss__withThrow_" );
    runTest();
    checkFinalPayload( PROGRAM_TYPE_PAYLOAD_MS );
}
TEST_P( O7_API_Device_ExceptionsP, ExceptionInDirectCallable )
{
    m_rg = "__raygen__directcallable";
    m_dc = getParameterizedName( "__direct_callable__withThrow_" );
    runTest();
    checkFinalPayload( PROGRAM_TYPE_PAYLOAD_DC );
}
TEST_P( O7_API_Device_ExceptionsP, ExceptionInContinuationCallable )
{
    m_rg = "__raygen__continuationcallable";
    m_cc = getParameterizedName( "__continuation_callable__withThrow_" );
    runTest();
    checkFinalPayload( PROGRAM_TYPE_PAYLOAD_CC );
}

INSTANTIATE_TEST_SUITE_P( RunThroughAllExceptionDetails,
                          O7_API_Device_ExceptionsP,
                          testing::Range( static_cast<unsigned int>( 0 ), static_cast<unsigned int>( 9 ) ),
                          []( const testing::TestParamInfo<O7_API_Device_ExceptionsP::ParamType>& info ) {
                              unsigned int payloadCount = info.param;
                              std::string  name;
                              if( !payloadCount )
                                  name = "noPayloads";
                              else
                                  name = std::to_string( payloadCount - 1 ) + "payloads";
                              return name;
                          } );


INSTANTIATE_TEST_SUITE_P( RunThroughAllIlwalidValues,
                          O7_API_Device_ExceptionsIlwalidRayP,
                          testing::Combine( testing::Range( 0, RAYDATA_FLOAT_COUNT ),
                                            testing::Values( std::numeric_limits<float>::infinity(),
                                                             std::numeric_limits<float>::quiet_NaN(),
                                                             std::numeric_limits<float>::signaling_NaN(),
                                                             std::sqrt( -1.f ) ) ) );

TEST_P( O7_API_Device_ExceptionsParamsMismatchP, TestExceptionData )
{
    runTest();

    EXPECT_EQ( PARAM_MISMATCH_CORRECT_ARGS_COUNT, m_missmatchIntData[0] );
    if( std::get<1>( GetParam() ) == underflowParamMismatchType )
        EXPECT_EQ( PARAM_MISMATCH_UNDERFLOW_ARGS_COUNT, m_missmatchIntData[1] );
    else
        EXPECT_EQ( PARAM_MISMATCH_OVERFLOW_ARGS_COUNT, m_missmatchIntData[1] );
    if( std::get<0>( GetParam() ) == directCallableType )
    {
        EXPECT_STREQ( PARAM_MISMATCH_DC_CALLABLE_NAME, m_missmatchCallableName );
        EXPECT_EQ( SBT_INDEX_DIRECT_CALLABLE, m_missmatchIntData[2] );
    }
    else
    {
        EXPECT_STREQ( PARAM_MISMATCH_CC_CALLABLE_NAME, m_missmatchCallableName );
        EXPECT_EQ( SBT_INDEX_CONTINUATION_CALLABLE, m_missmatchIntData[2] );
    }
    int lwvmVersion;
    OPTIX_CHECK( optixExtFeatureQueryLwvmCompilerVersion( &lwvmVersion ) );
    int callSiteLine = m_callSiteLine;
    std::string expectedSourceLoc;
    if( lwvmVersion < 700 )
        expectedSourceLoc = std::to_string( m_callSiteLine ) + ":1 (approximately)";
    else
        expectedSourceLoc = std::to_string( m_callSiteLine - 1 ) + ":5";
    EXPECT_NE( std::string::npos, std::string( m_lineInfo ).find( expectedSourceLoc ) )
        << "Couldn't find '" << expectedSourceLoc << "' in line info '" << m_lineInfo << "'";
}

INSTANTIATE_TEST_SUITE_P( RunWithParamsMismatch,
                          O7_API_Device_ExceptionsParamsMismatchP,
                          testing::Combine( testing::Values( directCallableType, continuationCallableType ),
                                            testing::Values( underflowParamMismatchType, overflowParamMismatchType ) ),
                          []( const testing::TestParamInfo<O7_API_Device_ExceptionsParamsMismatchP::ParamType>& info ) {
                              std::string name;
                              switch( std::get<0>( info.param ) )
                              {
                                  case directCallableType:
                                      name = "DC_";
                                      break;
                                  case continuationCallableType:
                                      name = "CC_";
                                      break;
                              }
                              switch( std::get<1>( info.param ) )
                              {
                                  case underflowParamMismatchType:
                                      name += "UNDERFLOW";
                                      break;
                                  case overflowParamMismatchType:
                                      name += "OVERFLOW";
                                      break;
                              }
                              return name;
                          } );
