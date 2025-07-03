
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

#include <array>
#include <vector>

#include "CommonAsserts.h"

#include "test_DeviceAPI_rayFlags.h"
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include <exptest/coverageFileWriter.h>
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include "tests/sanity/test_DeviceAPI_rayFlags_ptx_bin.h"

using RaygenSbtRecord    = SbtRecord<void>;
using ExceptionSbtRecord = SbtRecord<void>;
using MissSbtRecord      = SbtRecord<void>;
using HitgroupSbtRecord  = SbtRecord<void>;
using CallableSbtRecord  = SbtRecord<void>;

enum OptixTraversalABI
{
    OptixTraversal_UTRAV,
    OptixTraversal_TTU_A,
    OptixTraversal_TTU_B,
    OptixTraversal_MTTU,
    OptixTraversal_UNDEFINED
};

struct O7_API_Device_RayFlags;

struct GAccelBuildControl
{
    GAccelBuildControl( O7_API_Device_RayFlags& owner );
    ~GAccelBuildControl();
    O7_API_Device_RayFlags& m_owner;
};

struct O7_API_Device_RayFlags : public testing::Test
{
    void SetUp() override
    {
        m_rg     = "__raygen__";
        m_ex     = "__exception__nil";
        m_ms     = "__miss__";
        m_is     = "__intersection__";
        m_ah     = "__anyhit__";
        m_ch     = "__closesthit__";
        m_lwFile = "test_DeviceAPI_rayFlags.lw";

        m_rayFlags           = OPTIX_RAY_FLAG_NONE;
        m_visibilityMask     = 1;
        m_numInstances       = 0;
        m_buildTriangleInput = false;
        m_backFacing         = false;
        m_geometryFlags      = OPTIX_GEOMETRY_FLAG_NONE;
        m_instanceFlags      = OPTIX_INSTANCE_FLAG_NONE;
        m_traversalABI       = OptixTraversal_UTRAV;

        m_gasHandle         = 0;
        m_iasHandle         = 0;
        m_d_gasOutputBuffer = 0;
        m_d_iasOutputBuffer = 0;

        resetPayload();
    }
    void TearDown() override {}

    void runTest( bool resetPayload = true );
    void buildInstanceAccel();

    static OptixDeviceContext   s_context;
    static OptixRecordingLogger s_logger;
    static int                  s_numPayloadValues;
    OptixModule                 m_ptxModule;
    LWdeviceptr                 m_d_gasOutputBuffer;
    LWdeviceptr                 m_d_iasOutputBuffer;
    OptixTraversableHandle      m_gasHandle;
    OptixTraversableHandle      m_iasHandle;
    unsigned int                m_numInstances;
    bool                        m_buildTriangleInput;
    bool                        m_backFacing;
    OptixGeometryFlags          m_geometryFlags;
    OptixInstanceFlags          m_instanceFlags;

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

    std::vector<unsigned int> m_payloads;
    OptixTraversalABI         m_traversalABI;

    //static void SetUpTestCase()
    void initializeBuild()
    {
        exptest::lwdaInitialize();
        OPTIX_CHECK( optixInit() );

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &OptixRecordingLogger::callback;
        options.logCallbackData           = &s_logger;
        options.logCallbackLevel          = 3;
        LWcontext lwCtx                   = 0;  // zero means take the current context
        OPTIX_CHECK( optixDeviceContextCreate( lwCtx, &options, &s_context ) );

        if( m_traversalABI == OptixTraversal_TTU_B )
            setupTriangleInput();
        else
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

        unsigned int gasInputFlags[1]                             = { m_geometryFlags };
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

    void setupTriangleInput()
    {
        // Use default options for simplicity.  In a real use case we would want to
        // enable compaction, etc
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

        // Triangle build input: simple list of three vertices
        std::array<float3, 3> vertices = { { { -0.5f, -0.5f, 0.0f }, { 0.5f, -0.5f, 0.0f }, { 0.0f, 0.5f, 0.0f } } };
        if( m_backFacing )
        {
            vertices = { { { -0.5f, -0.5f, 0.0f }, { 0.0f, 0.5f, 0.0f }, { 0.5f, -0.5f, 0.0f } } };
        }

        const size_t vertices_size = sizeof( float3 ) * vertices.size();
        LWdeviceptr  d_vertices    = 0;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_vertices ), vertices_size ) );
        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_vertices ), vertices.data(), vertices_size, lwdaMemcpyHostToDevice ) );

        // Our build input is a simple list of non-indexed triangle vertices
        unsigned int    triangle_input_flags[1]    = { m_geometryFlags };
        OptixBuildInput triangle_input             = {};
        triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices   = static_cast<uint32_t>( vertices.size() );
        triangle_input.triangleArray.vertexBuffers = &d_vertices;
        triangle_input.triangleArray.flags         = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( s_context, &accel_options, &triangle_input,
                                                   1,  // Number of build inputs
                                                   &gas_buffer_sizes ) );
        LWdeviceptr d_temp_buffer_gas;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &m_d_gasOutputBuffer ), gas_buffer_sizes.outputSizeInBytes ) );

        OPTIX_CHECK( optixAccelBuild( s_context,
                                      0,  // LWCA stream
                                      &accel_options, &triangle_input,
                                      1,  // num build inputs
                                      d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, m_d_gasOutputBuffer,
                                      gas_buffer_sizes.outputSizeInBytes, &m_gasHandle,
                                      nullptr,  // emitted property list
                                      0         // num emitted properties
                                      ) );

        // We can now free the scratch space buffer used during build and the vertex
        // inputs, since they are not needed by our trivial shading method
        LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
        LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( d_vertices ) ) );
    }

    //static void TearDownTestCase()
    void finalizeBuild() { OPTIX_CHECK( optixDeviceContextDestroy( s_context ) ); }

    void resetPayload()
    {
        // reset payloads each time to appropriate size with initial values
        m_payloads.resize( s_numPayloadValues );
        for( int i = 0; i < s_numPayloadValues; ++i )
        {
            m_payloads[i] = 0;
        }
    }
};

struct O7_API_Device_RayFlags_P : public O7_API_Device_RayFlags, public testing::WithParamInterface<OptixTraversalABI>
{
    void SetUp() override
    {
        O7_API_Device_RayFlags::SetUp();
        m_traversalABI = GetParam();
    }
    void TearDown() override { O7_API_Device_RayFlags::TearDown(); }
};

OptixDeviceContext   O7_API_Device_RayFlags::s_context = nullptr;
OptixRecordingLogger O7_API_Device_RayFlags::s_logger{};
// as we are using all for some test case and this has to be filewise-global
int O7_API_Device_RayFlags::s_numPayloadValues = 8;


GAccelBuildControl::GAccelBuildControl( O7_API_Device_RayFlags& owner )
    : m_owner( owner )
{
    m_owner.initializeBuild();
}
GAccelBuildControl::~GAccelBuildControl()
{
    m_owner.finalizeBuild();
}


struct Instance
{
    float transform[12];
};
const Instance instance_transforms[] = {
    { 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 },
    { 1.0, 0.0, 0.0, 0.1f, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 }  // just to have the two transforms not exactly identical
};

void O7_API_Device_RayFlags::runTest( bool resetPld )
{
    GAccelBuildControl guard( *this );
    // clean up payloads for each run
    if( resetPld )
        resetPayload();
    // hard-coded for now
    const int numAttributeValues = 6;

    // if not explicitly set, set it for both TTU_A and TTU_B, but not for UTRAV
    if( m_numInstances == 0 )
    {
        if( m_traversalABI == OptixTraversal_TTU_A || m_traversalABI == OptixTraversal_TTU_B )
            m_numInstances = 2;
    }
    if( m_numInstances )
        buildInstanceAccel();

    // Compile modules

    OptixModuleCompileOptions moduleCompileOptions = {};

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur              = false;
    if( m_traversalABI == OptixTraversal_TTU_A || m_traversalABI == OptixTraversal_TTU_B )
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    else
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.numPayloadValues                 = s_numPayloadValues;
    pipelineCompileOptions.numAttributeValues               = numAttributeValues;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    if( m_traversalABI == OptixTraversal_TTU_B )
        pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    if( m_traversalABI == OptixTraversal_UTRAV )
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;

    OPTIX_CHECK_THROW( optixModuleCreateFromPTX( s_context, &moduleCompileOptions, &pipelineCompileOptions,
                                                 optix::data::gettest_DeviceAPI_rayFlagsSources()[1],
                                                 optix::data::gettest_DeviceAPI_rayFlagsSourceSizes()[0], 0, 0, &m_ptxModule ) );

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
    int                            num_hitgroup_records = 1;  // even when using instances we use the same hitgroup here
    std::vector<HitgroupSbtRecord> hgSBT( num_hitgroup_records );
    OPTIX_CHECK( optixSbtRecordPackHeader( hitgroupProgramGroup, &hgSBT[0] ) );
    LWdeviceptr d_hitgroupSbtRecord;
    size_t      hitgroupSbtRecordSize = sizeof( HitgroupSbtRecord );
    LWDA_CHECK( lwdaMalloc( (void**)&d_hitgroupSbtRecord, num_hitgroup_records * hitgroupSbtRecordSize ) );
    LWDA_CHECK( lwdaMemcpy( (void*)d_hitgroupSbtRecord, &hgSBT[0], num_hitgroup_records * hitgroupSbtRecordSize, lwdaMemcpyHostToDevice ) );

    OptixShaderBindingTable sbt     = {};
    sbt.raygenRecord                = d_raygenRecord;
    sbt.exceptionRecord             = 0;  //m_sbtRecord_nullptr ? 0 : d_exceptionRecord;
    sbt.missRecordBase              = d_missSbtRecord;
    sbt.missRecordStrideInBytes     = (unsigned int)sizeof( MissSbtRecord );
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordBase          = d_hitgroupSbtRecord;
    sbt.hitgroupRecordStrideInBytes = (unsigned int)sizeof( HitgroupSbtRecord );
    sbt.hitgroupRecordCount         = num_hitgroup_records;

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

    params.rayFlags       = m_rayFlags;
    params.visibilityMask = m_visibilityMask;

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
    LWDA_CHECK( lwdaFree( (void*)m_d_gasOutputBuffer ) );
    LWDA_CHECK( lwdaFree( (void*)m_d_iasOutputBuffer ) );
}

void O7_API_Device_RayFlags::buildInstanceAccel()
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
        optix_instances[idx].flags             = m_instanceFlags;
        optix_instances[idx].instanceId        = idx;
        optix_instances[idx].sbtOffset         = 0;  // <- all instances use the ONE hitgroup
        optix_instances[idx].visibilityMask    = 0xff;
        memcpy( optix_instances[idx].transform, instance_transforms[idx].transform, sizeof( float ) * 12 );
    }
    LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_instances ), &optix_instances[0], instance_size_in_bytes, lwdaMemcpyHostToDevice ) );
    LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &m_d_iasOutputBuffer ), ias_buffer_sizes.outputSizeInBytes ) );
    OPTIX_CHECK( optixAccelBuild( s_context, 0, &accel_options, &instance_input, 1, d_temp_buffer, ias_buffer_sizes.tempSizeInBytes,
                                  m_d_iasOutputBuffer, ias_buffer_sizes.outputSizeInBytes, &m_iasHandle, nullptr, 0 ) );
    LWDA_CHECK( lwdaFree( (void*)d_temp_buffer ) );
    LWDA_CHECK( lwdaFree( reinterpret_cast<void*>( d_instances ) ) );
}

TEST_P( O7_API_Device_RayFlags_P, TestRayFlags_OPTIX_RAY_FLAG_NONE )
{
    m_ah = "__anyhit__rayFlags";
    m_ch = "__closesthit__rayFlags";
    m_ms = "__miss__rayFlags";

    m_rayFlags = OPTIX_RAY_FLAG_NONE;
    runTest();

    ASSERT_TRUE( m_payloads[0] & PROGRAM_TYPE_PAYLOAD_AH );
    ASSERT_TRUE( m_payloads[0] & PROGRAM_TYPE_PAYLOAD_CH );
}

TEST_P( O7_API_Device_RayFlags_P, TestRayFlags_OPTIX_RAY_FLAG_DISABLE_ANYHIT )
{
    m_ah = "__anyhit__rayFlags";
    m_ch = "__closesthit__rayFlags";
    m_ms = "__miss__rayFlags";

    // Disables anyhit programs for the ray.
    m_rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
    runTest();

    EXPECT_FALSE( m_payloads[0] & PROGRAM_TYPE_PAYLOAD_AH );
}

TEST_P( O7_API_Device_RayFlags_P, TestRayFlags_OPTIX_RAY_FLAG_ENFORCE_ANYHIT )
{
    m_ah = "__anyhit__rayFlags";
    m_ch = "__closesthit__rayFlags";
    m_ms = "__miss__rayFlags";

    // Forces anyhit program exelwtion for the ray.
    m_rayFlags = OPTIX_RAY_FLAG_ENFORCE_ANYHIT;
    runTest();

    EXPECT_TRUE( m_payloads[0] & PROGRAM_TYPE_PAYLOAD_AH );
}

TEST_P( O7_API_Device_RayFlags_P, TestRayFlags_OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT_ch )
{
    m_ah = "__anyhit__rayFlags";
    m_ch = "__closesthit__rayFlags";
    m_ms = "__miss__rayFlags";

    // Disables closesthit programs for the ray.
    m_rayFlags = OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    runTest();

    EXPECT_FALSE( m_payloads[0] & PROGRAM_TYPE_PAYLOAD_CH );
}

TEST_P( O7_API_Device_RayFlags_P, TestRayFlags_OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT_ms )
{
    m_ah = "__anyhit__rayFlags";
    m_ch = "__closesthit__rayFlags";
    m_ms = "__miss__rayFlags";

    // clean run
    runTest();

    EXPECT_TRUE( m_payloads[0] & PROGRAM_TYPE_PAYLOAD_AH );
    EXPECT_TRUE( m_payloads[0] & PROGRAM_TYPE_PAYLOAD_CH );

    // Disables closesthit programs for the ray - but check whether the miss() gets still called
    m_rayFlags = OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    runTest();

    EXPECT_TRUE( m_payloads[0] & PROGRAM_TYPE_PAYLOAD_AH );
    EXPECT_FALSE( m_payloads[0] & PROGRAM_TYPE_PAYLOAD_CH );

    m_rg           = "__raygen__ilwertedDirection";
    m_traversalABI = GetParam();

    // Disables closesthit programs for the ray - but check whether the miss() gets still called
    m_rayFlags = OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    runTest();

    EXPECT_TRUE( m_payloads[0] & PROGRAM_TYPE_PAYLOAD_MS );
}

TEST_P( O7_API_Device_RayFlags_P, TestRayFlags_OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT_off )
{
    m_ah = "__anyhit__counting";
    // for triangles, there is no way to get reliably two hits
    if( m_traversalABI == OptixTraversal_TTU_B )
        return;

    m_numInstances                     = 2;
    const unsigned int expectedAHCount = 2;
    runTest();
    ASSERT_EQ( expectedAHCount, m_payloads[0] );
}

TEST_P( O7_API_Device_RayFlags_P, TestRayFlags_OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT_on )
{
    m_ah = "__anyhit__counting";

    m_numInstances                     = 2;
    m_rayFlags                         = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;
    const unsigned int expectedAHCount = 1;
    runTest();
    ASSERT_EQ( expectedAHCount, m_payloads[0] );
}

TEST_F( O7_API_Device_RayFlags, TestRayFlags_FrontFacingTriangleHitTest )
{
    m_ah           = "__anyhit__counting";
    m_traversalABI = OptixTraversal_TTU_B;

    unsigned int expectedAHCount = 1;
    runTest();
    ASSERT_EQ( expectedAHCount, m_payloads[0] );
}

TEST_F( O7_API_Device_RayFlags, TestRayFlags_BackFacingTriangleHitTest )
{
    m_ah           = "__anyhit__counting";
    m_traversalABI = OptixTraversal_TTU_B;
    m_backFacing   = true;

    unsigned int expectedAHCount = 1;
    runTest();
    ASSERT_EQ( expectedAHCount, m_payloads[0] );
}

TEST_F( O7_API_Device_RayFlags, TestRayFlags_OPTIX_RAY_FLAG_LWLL_FRONT_FACING_TRIANGLES )
{
    m_ah           = "__anyhit__counting";
    m_rg           = "__raygen__movedOriginAndDirection";
    m_traversalABI = OptixTraversal_TTU_B;

    // test that anyhit() gets called once
    unsigned int expectedAHCount = 1;
    runTest();
    ASSERT_EQ( expectedAHCount, m_payloads[0] );

    // test that anyhit() doesn't get called once
    m_rayFlags      = OPTIX_RAY_FLAG_LWLL_FRONT_FACING_TRIANGLES;
    expectedAHCount = 0;
    runTest();
    ASSERT_EQ( expectedAHCount, m_payloads[0] );
}

TEST_F( O7_API_Device_RayFlags, TestRayFlags_OPTIX_RAY_FLAG_LWLL_BACK_FACING_TRIANGLES )
{
    m_ah           = "__anyhit__counting";
    m_rg           = "__raygen__movedOriginAndDirection";
    m_traversalABI = OptixTraversal_TTU_B;
    // either one of the following two does work, so let's test the instance flag here
    //m_backFacing         = true;
    m_instanceFlags = OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING;

    // test that anyhit() gets called once
    unsigned int expectedAHCount = 1;
    runTest();
    ASSERT_EQ( expectedAHCount, m_payloads[0] );

    // test that anyhit() doesn't get called once
    m_rayFlags      = OPTIX_RAY_FLAG_LWLL_BACK_FACING_TRIANGLES;
    expectedAHCount = 0;
    runTest();
    ASSERT_EQ( expectedAHCount, m_payloads[0] );
}

TEST_P( O7_API_Device_RayFlags_P, TestRayFlags_OPTIX_RAY_FLAG_LWLL_DISABLED_ANYHIT )
{
    m_ah           = "__anyhit__counting";
    m_rayFlags     = OPTIX_RAY_FLAG_LWLL_DISABLED_ANYHIT;
    m_numInstances = 2;

    // test that anyhit() gets called w/o proper geometry flags setting
    runTest();
    unsigned int expectedAHCount = 2;
    if( m_traversalABI == OptixTraversal_TTU_B )
        expectedAHCount = 1;
    ASSERT_EQ( expectedAHCount, m_payloads[0] );

    // test that anyhit() gets not called with proper geometry flags setting
    m_geometryFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    runTest();
    expectedAHCount = 0;
    ASSERT_EQ( expectedAHCount, m_payloads[0] );

    // test that anyhit() gets called w/o proper geometry flags setting
    m_geometryFlags = OPTIX_GEOMETRY_FLAG_NONE;
    m_instanceFlags = OPTIX_INSTANCE_FLAG_NONE;
    runTest();
    expectedAHCount = 2;
    if( m_traversalABI == OptixTraversal_TTU_B )
        expectedAHCount = 1;
    ASSERT_EQ( expectedAHCount, m_payloads[0] );

    // test that anyhit() gets not called with proper geometry flags setting
    m_instanceFlags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
    runTest();
    expectedAHCount = 0;
    ASSERT_EQ( expectedAHCount, m_payloads[0] );
}

TEST_P( O7_API_Device_RayFlags_P, TestRayFlags_OPTIX_RAY_FLAG_LWLL_ENFORCED_ANYHIT )
{
    m_ah           = "__anyhit__counting";
    m_numInstances = 2;

    // test that anyhit() gets called w/o proper ray flags setting
    runTest();
    unsigned int expectedAHCount = 2;
    if( m_traversalABI == OptixTraversal_TTU_B )
        expectedAHCount = 1;
    ASSERT_EQ( expectedAHCount, m_payloads[0] );

    // test that anyhit() gets not called with proper ray flags setting
    m_rayFlags = OPTIX_RAY_FLAG_LWLL_ENFORCED_ANYHIT;
    runTest();
    expectedAHCount = 0;
    ASSERT_EQ( expectedAHCount, m_payloads[0] );
}

TEST_F( O7_API_Device_RayFlags, TestRayFlags_8bit_handling_with_8bit )
{
    // setting all ray flags enabled OPTIX_RAY_FLAG_LWLL_ENFORCED_ANYHIT, hence no hit programs and
    // only MS program required
    m_rg          = "__raygen__8bit";
    m_ms          = "__miss__8bit";
    m_payloads[1] = 0xDEDEDEDE;

    // pass in rayflags as m_payloads[0], retrieval in m_payloads[1]
    m_payloads[0] = 0xFF;
    runTest( false );
    ASSERT_EQ( 0xFF, m_payloads[1] );
}

TEST_F( O7_API_Device_RayFlags, TestRayFlags_8bit_handling_with_16bit )
{
    // enabling OPTIX_RAY_FLAG_LWLL_ENFORCED_ANYHIT, hence no hit programs and only MS program required
    m_rg          = "__raygen__8bit";
    m_ms          = "__miss__8bit";
    m_payloads[1] = 0xDEDEDEDE;

    // pass in rayflags as m_payloads[0], retrieval in m_payloads[1]
    m_payloads[0] = 0xFF00 | OPTIX_RAY_FLAG_LWLL_ENFORCED_ANYHIT;
    runTest( false );
    // here we assume that at most 16 bits are used for ray flags and that it is not possible to pass and query 'unknown' ray flags to optix
    ASSERT_EQ( OPTIX_RAY_FLAG_LWLL_ENFORCED_ANYHIT, m_payloads[1] & 0xFF );
    ASSERT_GE( 0xFF00 | OPTIX_RAY_FLAG_LWLL_ENFORCED_ANYHIT, m_payloads[1] );
}

TEST_F( O7_API_Device_RayFlags, TestRayFlags_8bit_handling_with_32bit )
{
    // enabling OPTIX_RAY_FLAG_LWLL_ENFORCED_ANYHIT, hence no hit programs and only MS program required
    m_rg          = "__raygen__8bit";
    m_ms          = "__miss__8bit";
    m_payloads[1] = 0xDEDEDEDE;

    // pass in rayflags as m_payloads[0], retrieval as m_payloads[1]
    m_payloads[0] = 0xFFFF00 | OPTIX_RAY_FLAG_LWLL_ENFORCED_ANYHIT;
    runTest( false );
    // here we assume that at most 16 bits are used for ray flags and that it is not possible to pass and query 'unknown' ray flags to optix
    ASSERT_EQ( OPTIX_RAY_FLAG_LWLL_ENFORCED_ANYHIT, m_payloads[1] & 0xFF );
    ASSERT_GE( 0xFF00 | OPTIX_RAY_FLAG_LWLL_ENFORCED_ANYHIT, m_payloads[1] );
}

INSTANTIATE_TEST_SUITE_P( RunThroughAllTraversalABIs,
                          O7_API_Device_RayFlags_P,
                          testing::Values( OptixTraversal_UTRAV, OptixTraversal_TTU_A, OptixTraversal_TTU_B, OptixTraversal_MTTU ),
                          []( const testing::TestParamInfo<O7_API_Device_RayFlags_P::ParamType>& info ) {
                              std::string name;
                              switch( info.param )
                              {
                                  case OptixTraversal_UTRAV:
                                      name = "UTRAV";
                                      break;
                                  case OptixTraversal_TTU_A:
                                      name = "TTU_A";
                                      break;
                                  case OptixTraversal_TTU_B:
                                      name = "TTU_B";
                                      break;
                                  case OptixTraversal_MTTU:
                                      name = "MTTU";
                                      break;
                                  default:
                                      name = "???";
                                      break;
                              }
                              return name;
                          } );
