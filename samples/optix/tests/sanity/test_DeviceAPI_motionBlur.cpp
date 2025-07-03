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

#include "test_DeviceAPI_motionBlur.h"
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include <exptest/coverageFileWriter.h>
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include "tests/sanity/test_DeviceAPI_motionBlur_ptx_bin.h"

class TestMotionBlur : public testing::Test
{
  public:
    static OptixLogger                 s_logger;
    static OptixDeviceContext          s_context;
    static OptixDeviceContextOptions   s_options;
    static OptixModuleCompileOptions   s_moduleCompileOptions;
    static OptixPipelineCompileOptions s_pipelineCompileOptions;


    static void SetUpTestCase()
    {
        exptest::lwdaInitialize();
        OPTIX_CHECK( optixInit() );

        s_options.logCallbackFunction = &OptixLogger::callback;
        s_options.logCallbackData     = &s_logger;
        s_options.logCallbackLevel    = 4;
        LWcontext lwCtx               = 0;  // zero means take the current context
        OPTIX_CHECK( optixDeviceContextCreate( lwCtx, &s_options, &s_context ) );
    }

    static void TearDownTestCase() { OPTIX_CHECK( optixDeviceContextDestroy( s_context ) ); }

    static void buildStaticAS( OptixTraversableHandle& gas_handle, LWdeviceptr& d_gas_output_buffer )
    {
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

        // AABB build input
        OptixAabb   aabb = {-1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f};
        LWdeviceptr d_aabb_buffer;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_aabb_buffer ), sizeof( OptixAabb ) ) );
        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_aabb_buffer ), &aabb, sizeof( OptixAabb ), lwdaMemcpyHostToDevice ) );

        OptixBuildInput aabb_input = {};

        aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES;
        aabb_input.lwstomPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
        aabb_input.lwstomPrimitiveArray.numPrimitives = 1;

        uint32_t aabb_input_flags[1]                  = {OPTIX_GEOMETRY_FLAG_NONE};
        aabb_input.lwstomPrimitiveArray.flags         = aabb_input_flags;
        aabb_input.lwstomPrimitiveArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( s_context, &accel_options, &aabb_input, 1, &gas_buffer_sizes ) );
        LWdeviceptr d_temp_buffer_gas;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), gas_buffer_sizes.outputSizeInBytes ) );

        OPTIX_CHECK( optixAccelBuild( s_context,
                                      0,  // LWCA stream
                                      &accel_options, &aabb_input,
                                      1,  // num build inputs
                                      d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer,
                                      gas_buffer_sizes.outputSizeInBytes, &gas_handle,
                                      nullptr,  // emitted property list
                                      0         // num emitted properties
                                      ) );

        LWDA_CHECK( lwdaFree( (void*)d_temp_buffer_gas ) );
        LWDA_CHECK( lwdaFree( (void*)d_aabb_buffer ) );
    }

    static void buildMotionAS( OptixTraversableHandle& gas_handle, LWdeviceptr& d_gas_output_buffer )
    {
        OptixAccelBuildOptions accel_options  = {};
        accel_options.buildFlags              = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation               = OPTIX_BUILD_OPERATION_BUILD;
        accel_options.motionOptions.numKeys   = 2;
        accel_options.motionOptions.timeBegin = -3;
        accel_options.motionOptions.timeEnd   = 3;
        accel_options.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;

        // AABB build input
        OptixAabb   aabb = {-1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f};
        LWdeviceptr d_aabb_buffer;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_aabb_buffer ), sizeof( OptixAabb ) ) );
        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_aabb_buffer ), &aabb, sizeof( OptixAabb ), lwdaMemcpyHostToDevice ) );

        OptixBuildInput aabb_input = {};

        LWdeviceptr d_aabb_buffers[2] = {d_aabb_buffer, d_aabb_buffer};

        aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES;
        aabb_input.lwstomPrimitiveArray.aabbBuffers   = d_aabb_buffers;
        aabb_input.lwstomPrimitiveArray.numPrimitives = 1;

        uint32_t aabb_input_flags[1]                  = {OPTIX_GEOMETRY_FLAG_NONE};
        aabb_input.lwstomPrimitiveArray.flags         = aabb_input_flags;
        aabb_input.lwstomPrimitiveArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( s_context, &accel_options, &aabb_input, 1, &gas_buffer_sizes ) );
        LWdeviceptr d_temp_buffer_gas;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), gas_buffer_sizes.outputSizeInBytes ) );

        OPTIX_CHECK( optixAccelBuild( s_context,
                                      0,  // LWCA stream
                                      &accel_options, &aabb_input,
                                      1,  // num build inputs
                                      d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer,
                                      gas_buffer_sizes.outputSizeInBytes, &gas_handle,
                                      nullptr,  // emitted property list
                                      0         // num emitted properties
                                      ) );

        LWDA_CHECK( lwdaFree( (void*)d_temp_buffer_gas ) );
        LWDA_CHECK( lwdaFree( (void*)d_aabb_buffer ) );
    }

    static void runTest( bool enableMotion, bool useMotionGAS )
    {
        OptixModule module;
        s_pipelineCompileOptions.usesMotionBlur                   = enableMotion;
        s_pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
        OptixResult result = optixModuleCreateFromPTX( s_context, &s_moduleCompileOptions, &s_pipelineCompileOptions,
                                                       optix::data::gettest_DeviceAPI_motionBlurSources()[1],
                                                       optix::data::gettest_DeviceAPI_motionBlurSourceSizes()[0], 0, 0, &module );

        EXPECT_EQ( result, OPTIX_SUCCESS );

        OptixTraversableHandle gas_handle;
        LWdeviceptr            d_gas_output_buffer;

        if( useMotionGAS )
            buildMotionAS( gas_handle, d_gas_output_buffer );
        else
            buildStaticAS( gas_handle, d_gas_output_buffer );

        OptixProgramGroupOptions programGroupOptions = {};
        OptixProgramGroupDesc    rgProgramGroupDesc  = {};
        rgProgramGroupDesc.kind                      = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        rgProgramGroupDesc.raygen.module             = module;
        rgProgramGroupDesc.raygen.entryFunctionName  = "__raygen__writeMotionSettings";
        OptixProgramGroup rgProgramGroup;
        OPTIX_CHECK( optixProgramGroupCreate( s_context, &rgProgramGroupDesc, 1, &programGroupOptions, 0, 0, &rgProgramGroup ) );

        OptixProgramGroupDesc hgProgramGroupDesc        = {};
        hgProgramGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hgProgramGroupDesc.hitgroup.moduleIS            = module;
        hgProgramGroupDesc.hitgroup.entryFunctionNameIS = "__intersection__writeMotionSettings";
        OptixProgramGroup hgProgramGroup;
        OPTIX_CHECK( optixProgramGroupCreate( s_context, &hgProgramGroupDesc, 1, &programGroupOptions, 0, 0, &hgProgramGroup ) );

        OptixProgramGroupDesc msProgramGroupDesc = {};
        msProgramGroupDesc.kind                  = OPTIX_PROGRAM_GROUP_KIND_MISS;
        OptixProgramGroup msProgramGroup;
        OPTIX_CHECK( optixProgramGroupCreate( s_context, &msProgramGroupDesc, 1, &programGroupOptions, 0, 0, &msProgramGroup ) );

        OptixPipeline            pipeline;
        OptixProgramGroup        programGroups[]     = {rgProgramGroup, hgProgramGroup, msProgramGroup};
        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth            = 1;

        OPTIX_CHECK( optixPipelineCreate( s_context, &s_pipelineCompileOptions, &pipelineLinkOptions, programGroups,
                                          sizeof( programGroups ) / sizeof( programGroups[0] ), 0, 0, &pipeline ) );

        OutData* d_data;
        size_t   data_byte_size = sizeof( OutData );
        LWDA_CHECK( lwdaMalloc( (void**)&d_data, data_byte_size ) );

        Params params;
        params.outData = d_data;
        params.handle  = gas_handle;
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
        SETUP_API_CALL_COVERAGE( params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

        LWstream stream;
        LWDA_CHECK( lwdaStreamCreate( &stream ) );

        LWdeviceptr d_param;
        LWDA_CHECK( lwdaMalloc( (void**)&d_param, sizeof( Params ) ) );
        LWDA_CHECK( lwdaMemcpy( (void*)d_param, &params, sizeof( params ), lwdaMemcpyHostToDevice ) );

        LWdeviceptr raygenRecord;
        size_t      raygenRecordSize = sizeof( SbtRecord<void> );
        LWDA_CHECK( lwdaMalloc( (void**)&raygenRecord, raygenRecordSize ) );
        SbtRecord<void> rgSBT;
        OPTIX_CHECK( optixSbtRecordPackHeader( rgProgramGroup, &rgSBT ) );
        LWDA_CHECK( lwdaMemcpy( (void*)raygenRecord, &rgSBT, raygenRecordSize, lwdaMemcpyHostToDevice ) );

        LWdeviceptr hitgroupSbtRecord;
        size_t      hitgroupSbtRecordSize = sizeof( SbtRecord<void> );
        LWDA_CHECK( lwdaMalloc( (void**)&hitgroupSbtRecord, hitgroupSbtRecordSize ) );
        SbtRecord<void> hgSBT;
        OPTIX_CHECK( optixSbtRecordPackHeader( hgProgramGroup, &hgSBT ) );
        LWDA_CHECK( lwdaMemcpy( (void*)hitgroupSbtRecord, &hgSBT, hitgroupSbtRecordSize, lwdaMemcpyHostToDevice ) );

        LWdeviceptr missSbtRecord;
        size_t      missSbtRecordSize = sizeof( SbtRecord<void> );
        LWDA_CHECK( lwdaMalloc( (void**)&missSbtRecord, missSbtRecordSize ) );
        SbtRecord<void> msSBT;
        OPTIX_CHECK( optixSbtRecordPackHeader( msProgramGroup, &msSBT ) );
        LWDA_CHECK( lwdaMemcpy( (void*)missSbtRecord, &msSBT, missSbtRecordSize, lwdaMemcpyHostToDevice ) );

        OptixShaderBindingTable sbt     = {};
        sbt.raygenRecord                = raygenRecord;
        sbt.missRecordBase              = missSbtRecord;
        sbt.missRecordStrideInBytes     = sizeof( SbtRecord<void> );
        sbt.missRecordCount             = 1;
        sbt.hitgroupRecordBase          = hitgroupSbtRecord;
        sbt.hitgroupRecordStrideInBytes = sizeof( SbtRecord<void> );
        sbt.hitgroupRecordCount         = 1;

        OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, 1, 1, /*depth=*/1 ) );
        LWDA_SYNC_CHECK();

        OutData h_data;
        LWDA_CHECK( lwdaMemcpy( &h_data, d_data, data_byte_size, lwdaMemcpyDeviceToHost ) );
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
        ANALYZE_API_CALL_COVERAGE( params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

        if( useMotionGAS && enableMotion )
        {
            EXPECT_EQ( h_data.gasMotionTimeBegin, -3 );
            EXPECT_EQ( h_data.gasMotionTimeEnd, 3 );
            EXPECT_EQ( h_data.gasMotionStepCount, 2 );
        }
        else
        {
            EXPECT_EQ( h_data.gasMotionTimeBegin, 0 );
            EXPECT_EQ( h_data.gasMotionTimeEnd, 0 );
            EXPECT_EQ( h_data.gasMotionStepCount, 0 );
        }

        if( enableMotion )
        {
            EXPECT_EQ( h_data.lwrrentMotionTime, 5 );
        }
        else
        {
            EXPECT_EQ( h_data.lwrrentMotionTime, 0 );
        }

        LWDA_CHECK( lwdaFree( (void*)d_param ) );
        LWDA_CHECK( lwdaFree( (void*)raygenRecord ) );
        LWDA_CHECK( lwdaFree( (void*)hitgroupSbtRecord ) );
        LWDA_CHECK( lwdaFree( (void*)missSbtRecord ) );
        LWDA_CHECK( lwdaFree( (void*)d_gas_output_buffer ) );
        LWDA_CHECK( lwdaFree( d_data ) );

        OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
        OPTIX_CHECK( optixProgramGroupDestroy( rgProgramGroup ) );
        OPTIX_CHECK( optixProgramGroupDestroy( hgProgramGroup ) );
        OPTIX_CHECK( optixProgramGroupDestroy( msProgramGroup ) );
        OPTIX_CHECK( optixModuleDestroy( module ) );
    }
};

OptixLogger                 TestMotionBlur::s_logger( std::cerr );
OptixDeviceContext          TestMotionBlur::s_context                = 0;
OptixDeviceContextOptions   TestMotionBlur::s_options                = {};
OptixModuleCompileOptions   TestMotionBlur::s_moduleCompileOptions   = {};
OptixPipelineCompileOptions TestMotionBlur::s_pipelineCompileOptions = {};

TEST_F( TestMotionBlur, motionDisabledReturnMotionOptions )
{
    runTest( /*enableMotion=*/false, /*useMotionGAS=*/false );
}

TEST_F( TestMotionBlur, motionEnabledStaticGasReturnMotionOptions )
{
    runTest( /*enableMotion=*/true, /*useMotionGAS=*/false );
}

TEST_F( TestMotionBlur, motionEnabledMotionGasReturnMotionOptions )
{
    runTest( /*enableMotion=*/true, /*useMotionGAS=*/true );
}
