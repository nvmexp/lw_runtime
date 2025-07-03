
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

#include <exptest/exptest.h>
#include <exptest/loggingHelpers.h>
#include <exptest/testTools.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "CommonAsserts.h"

#include "test_DeviceAPI_reportIntersection.h"
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include <exptest/coverageFileWriter.h>
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include "tests/sanity/test_DeviceAPI_reportIntersection_0_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_reportIntersection_1_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_reportIntersection_2_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_reportIntersection_3_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_reportIntersection_4_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_reportIntersection_5_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_reportIntersection_6_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_reportIntersection_7_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_reportIntersection_optixIgnoreIntersection_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_reportIntersection_optixTerminateRay_ptx_bin.h"
#include "tests/sanity/test_DeviceAPI_reportIntersection_ptx_bin.h"

using RaygenSbtRecord    = SbtRecord<void>;
using ExceptionSbtRecord = SbtRecord<void>;
using MissSbtRecord      = SbtRecord<void>;
using HitgroupSbtRecord  = SbtRecord<void>;
using CallableSbtRecord  = SbtRecord<void>;

enum OptixTraversalABI
{
    OptixTraversal_UTRAV,
    OptixTraversal_TTU_A,
    OptixTraversal_MTTU,
    OptixTraversal_UNDEFINED
};

struct Instance
{
    float transform[12];
};
const Instance instance_transforms[] = {
    { 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 },
    { 1.0, 0.0, 0.0, 0.1f, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 }  // just to have the two transforms not exactly identical
};

struct O7_API_Device_ReportIntersection : public testing::Test
{
    void SetUp() override
    {
        m_rg = "__raygen__";
        m_ex = "__exception__nil";
        m_ms = "__miss__nil";
        m_is = "__intersection__";
        m_ah = "__anyhit__nil";
        m_ch = "__closesthit__";

        m_gasHandle         = 0;
        m_iasHandle         = 0;
        m_d_gasOutputBuffer = 0;
        m_d_iasOutputBuffer = 0;
        m_numInstances      = 0;
        // will be overridden by the child class
        m_traversalABI = OptixTraversal_UTRAV;
    }
    void TearDown() override
    {
        LWDA_CHECK( lwdaFree( (void*)m_d_gasOutputBuffer ) );
        LWDA_CHECK( lwdaFree( (void*)m_d_iasOutputBuffer ) );
    }

    void runTest( int numAttributeValues );

    static OptixDeviceContext   s_context;
    static OptixRecordingLogger s_logger;
    OptixModule                 m_ptxModule;
    OptixTraversalABI           m_traversalABI;
    OptixTraversableHandle      m_gasHandle;
    OptixTraversableHandle      m_iasHandle;
    LWdeviceptr                 m_d_gasOutputBuffer;
    LWdeviceptr                 m_d_iasOutputBuffer;
    unsigned int                m_numInstances;

    // Default values used by runTest() unless changed upfront
    std::string m_lwFile;
    std::string m_rg;
    std::string m_ex;
    std::string m_ms;
    std::string m_is;
    std::string m_ah;
    std::string m_ch;

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
            default:
                break;
        }
        return PAYLOAD_VALUES_UNDEF;
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

    static void TearDownTestCase() { OPTIX_CHECK( optixDeviceContextDestroy( s_context ) ); }

    void initializeBuild( int numAttributeValues )
    {
        // reset payloads each time to appropriate size with initial values
        m_payloads.clear();
        for( int i = 0; i < numAttributeValues; ++i )
        {
            m_payloads.push_back( getPayloadValue( i ) );
        }

        setupLwstomInput();
    }

    void setupLwstomInput()
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
        LWDA_CHECK( lwdaMalloc( (void**)&m_d_gasOutputBuffer, gasBufferSizes.outputSizeInBytes ) );

        OPTIX_CHECK( optixAccelBuild( s_context, 0, &gasAccelOptions, &gasInput, 1, d_tempBuffer, gasBufferSizes.tempSizeInBytes,
                                      m_d_gasOutputBuffer, gasBufferSizes.outputSizeInBytes, &m_gasHandle, nullptr, 0 ) );

        LWDA_CHECK( lwdaFree( (void*)d_aabbs ) );
        LWDA_CHECK( lwdaFree( (void*)d_tempBuffer ) );
        LWDA_CHECK( lwdaFree( (void*)d_sbtIndexOffsets ) );
    }

    void buildInstanceAccel()
    {
        ASSERT_TRUE( m_numInstances > 0 );
        LWdeviceptr d_instances;
        size_t      instance_size_in_bytes = sizeof( OptixInstance ) * m_numInstances;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_instances ), instance_size_in_bytes ) );

        OptixBuildInput instance_input            = {};
        instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instance_input.instanceArray.instances    = d_instances;
        instance_input.instanceArray.numInstances = m_numInstances;
        OptixAccelBuildOptions accel_options      = {};
        accel_options.buildFlags                  = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation                   = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes ias_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( s_context, &accel_options, &instance_input, 1, &ias_buffer_sizes ) );
        LWdeviceptr d_temp_buffer;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_temp_buffer ), ias_buffer_sizes.tempSizeInBytes ) );
        std::vector<OptixInstance> optix_instances( m_numInstances );
        memset( &optix_instances[0], 0, instance_size_in_bytes );
        for( uint32_t idx = 0; idx < m_numInstances; ++idx )
        {
            optix_instances[idx].traversableHandle = m_gasHandle;
            optix_instances[idx].flags             = OPTIX_INSTANCE_FLAG_NONE;
            optix_instances[idx].instanceId        = idx;
            optix_instances[idx].sbtOffset         = 0;  // <- all instances use the ONE hitgroup
            optix_instances[idx].visibilityMask    = 0xff;
            memcpy( optix_instances[idx].transform, instance_transforms[idx].transform, sizeof( float ) * 12 );
        }
        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_instances ), &optix_instances[0], instance_size_in_bytes,
                                lwdaMemcpyHostToDevice ) );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &m_d_iasOutputBuffer ), ias_buffer_sizes.outputSizeInBytes ) );
        OPTIX_CHECK( optixAccelBuild( s_context, 0, &accel_options, &instance_input, 1, d_temp_buffer, ias_buffer_sizes.tempSizeInBytes,
                                      m_d_iasOutputBuffer, ias_buffer_sizes.outputSizeInBytes, &m_iasHandle, nullptr, 0 ) );
        LWDA_CHECK( lwdaFree( (void*)d_temp_buffer ) );
        LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( d_instances ) ) );
    }


    // Check whether the returned payloads (aka val) are equal to each initial payload value bitwise OR-ed with val.
    void checkFinalPayload( unsigned int val, unsigned int param )
    {
        for( unsigned int i = 0; i < param; ++i )
            ASSERT_EQ( getPayloadValue( i ) | val, m_payloads[i] );
    }

    const char**  getSources();
    const size_t* getSourceSizes();
};

OptixDeviceContext   O7_API_Device_ReportIntersection::s_context = nullptr;
OptixRecordingLogger O7_API_Device_ReportIntersection::s_logger{};

const char** O7_API_Device_ReportIntersection::getSources()
{
    if( m_lwFile == "test_DeviceAPI_reportIntersection.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersectionSources();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_0.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_0Sources();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_1.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_1Sources();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_2.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_2Sources();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_3.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_3Sources();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_4.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_4Sources();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_5.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_5Sources();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_6.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_6Sources();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_7.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_7Sources();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_optixIgnoreIntersection.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_optixIgnoreIntersectionSources();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_optixTerminateRay.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_optixTerminateRaySources();
    else
        return nullptr;
}

const size_t* O7_API_Device_ReportIntersection::getSourceSizes()
{
    if( m_lwFile == "test_DeviceAPI_reportIntersection.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersectionSourceSizes();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_0.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_0SourceSizes();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_1.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_1SourceSizes();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_2.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_2SourceSizes();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_3.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_3SourceSizes();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_4.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_4SourceSizes();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_5.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_5SourceSizes();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_6.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_6SourceSizes();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_7.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_7SourceSizes();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_optixIgnoreIntersection.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_optixIgnoreIntersectionSourceSizes();
    else if( m_lwFile == "test_DeviceAPI_reportIntersection_optixTerminateRay.lw" )
        return optix::data::gettest_DeviceAPI_reportIntersection_optixTerminateRaySourceSizes();
    else
        return 0;
}

void O7_API_Device_ReportIntersection::runTest( int numAttributeValues )
{
    initializeBuild( numAttributeValues );

    // if not explicitly set, set it for TTU_A, but not for UTRAV
    if( m_numInstances == 0 )
    {
        if( m_traversalABI == OptixTraversal_TTU_A )
            m_numInstances = 2;
    }
    if( m_numInstances )
        buildInstanceAccel();

    // Compile modules

    OptixModuleCompileOptions moduleCompileOptions = {};

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur              = false;
    if( m_traversalABI == OptixTraversal_TTU_A )
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    else
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.numPayloadValues                 = 8;
    pipelineCompileOptions.numAttributeValues               = numAttributeValues;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    if( m_traversalABI == OptixTraversal_UTRAV )
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;

    OPTIX_CHECK_THROW( optixModuleCreateFromPTX( s_context, &moduleCompileOptions, &pipelineCompileOptions,
                                                 getSources()[1], getSourceSizes()[0], 0, 0, &m_ptxModule ) );

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
    if( m_iasHandle )
        params.handle = m_iasHandle;
    else
        params.handle = m_gasHandle;

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

TEST_F( O7_API_Device_ReportIntersection, RunningDefaultPrograms )
{
    m_lwFile = "test_DeviceAPI_reportIntersection.lw";
    runTest( 0 );
}

TEST_F( O7_API_Device_ReportIntersection, TestPrograms )
{
    m_lwFile = "test_DeviceAPI_reportIntersection_0.lw";
    m_is     = "__intersection__Default_First_0";
    m_ch     = "__closesthit__0";
    runTest( 1 );
    checkFinalPayload( PROGRAM_TYPE_PAYLOAD_CH, 0 + 1 );
}

TEST_F( O7_API_Device_ReportIntersection, optixIgnoreIntersectionTest )
{
    m_lwFile = "test_DeviceAPI_reportIntersection_optixIgnoreIntersection.lw";
    m_rg     = "__raygen__";
    m_is     = "__intersection__";
    m_ch     = "__closesthit__";
    m_ah     = "__anyhit__ignoreIntersection";
    runTest( 1 );
    ASSERT_EQ( PROGRAM_TYPE_PAYLOAD_UNDEF, m_payloads[0] );
}

TEST_F( O7_API_Device_ReportIntersection, optixTerminateRayTest )
{
    m_lwFile = "test_DeviceAPI_reportIntersection_optixTerminateRay.lw";
    m_rg     = "__raygen__";
    m_is     = "__intersection__";
    m_ch     = "__closesthit__";
    m_ah     = "__anyhit__terminateRay";
    runTest( 1 );
    ASSERT_EQ( PROGRAM_TYPE_PAYLOAD_UNDEF, m_payloads[0] );
}

using GtestParam = std::tuple<OptixTraversalABI, unsigned int>;

struct O7_API_Device_ReportIntersection_P : public O7_API_Device_ReportIntersection, public testing::WithParamInterface<GtestParam>
{
    std::string getParameterizedName( const std::string& name )
    {
        return name + std::to_string( int( std::get<1>( GetParam() ) ) );
    }

    void SetUp() override
    {
        O7_API_Device_ReportIntersection::SetUp();
        m_traversalABI = std::get<0>( GetParam() );
    }

    // Check whether the returned payloads (aka val) are equal to each initial payload value bitwise OR-ed with val.
    void checkFinalPayload( unsigned int val )
    {
        O7_API_Device_ReportIntersection::checkFinalPayload( val, std::get<1>( GetParam() ) );
    }
};

TEST_P( O7_API_Device_ReportIntersection_P, DefaultRun )
{
    m_lwFile = getParameterizedName( "test_DeviceAPI_reportIntersection_" ) + ".lw";
    m_is     = getParameterizedName( "__intersection__Default_First_" );
    m_ch     = getParameterizedName( "__closesthit__" );
    runTest( std::get<1>( GetParam() ) + 1 );
    checkFinalPayload( PROGRAM_TYPE_PAYLOAD_CH );

    m_is = getParameterizedName( "__intersection__Default_Second_" );
    m_ch = getParameterizedName( "__closesthit__" );
    runTest( std::get<1>( GetParam() ) + 1 );
    checkFinalPayload( PROGRAM_TYPE_PAYLOAD_CH );

    m_is = getParameterizedName( "__intersection__Default_Third_" );
    m_ch = getParameterizedName( "__closesthit__" );
    runTest( std::get<1>( GetParam() ) + 1 );
    checkFinalPayload( PROGRAM_TYPE_PAYLOAD_CH );
}

INSTANTIATE_TEST_SUITE_P( RunThroughAllReportIntersectionTests,
                          O7_API_Device_ReportIntersection_P,
                          testing::Combine( testing::Values( OptixTraversal_UTRAV, OptixTraversal_TTU_A, OptixTraversal_MTTU ),
                                            testing::Range( static_cast<unsigned int>( 0 ), static_cast<unsigned int>( 8 ) ) ),
                          []( const testing::TestParamInfo<O7_API_Device_ReportIntersection_P::ParamType>& info ) {
                              std::string name;
                              switch( std::get<0>( info.param ) )
                              {
                                  case OptixTraversal_UTRAV:
                                      name = "utrav";
                                      break;
                                  case OptixTraversal_TTU_A:
                                      name = "ttu_a";
                                      break;
                                  case OptixTraversal_MTTU:
                                      name = "mttu";
                                      break;
                                  default:
                                      name = "unknown";
                              }
                              name += "_with_";
                              name += std::to_string( std::get<1>( info.param ) + 1 );
                              name += "_attributes";

                              return name;
                          } );
