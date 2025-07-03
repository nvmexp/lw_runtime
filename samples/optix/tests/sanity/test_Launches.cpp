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

#include "CommonAsserts.h"

#include <iostream>
#include <vector>

#include "test_Launches.h"
#include "tests/sanity/test_Launches_ptx_bin.h"

using namespace testing;

struct GenericData
{
    float3 some_data;
};
typedef SbtRecord<GenericData> RayGenSbtRecord;
typedef SbtRecord<GenericData> MissSbtRecord;
typedef SbtRecord<GenericData> ExceptionSbtRecord;
typedef SbtRecord<GenericData> HitGroupSbtRecord;
typedef SbtRecord<GenericData> CallablesSbtRecord;

class O7_API_optixSbtRecordPackHeader : public Test
{
  protected:
    void SetUp() override
    {
        exptest::lwdaInitialize();
        ASSERT_OPTIX_SUCCESS( optixInit() );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( nullptr, nullptr, &m_context ) );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextSetLogCallback( m_context, &OptixRecordingLogger::callback, &m_logger, 2 ) );
        OPTIX_CHECK( optixModuleCreateFromPTX( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                               m_ptx_str.c_str(), m_ptx_str.size(), nullptr, nullptr, &m_module ) );
    }
    void createRayGenProgramGroup( bool withParams = false )
    {
        OptixProgramGroupOptions program_group_options  = {};
        OptixProgramGroupDesc    raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = m_module;
        if( withParams )
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg_withParams";
        else
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &raygen_prog_group_desc, 1, &program_group_options,
                                                       nullptr, nullptr, &m_raygen_prog_group ) );
    }
    void createMissProgramGroup( bool withParams = false )
    {
        OptixProgramGroupOptions program_group_options = {};
        OptixProgramGroupDesc    miss_prog_group_desc  = {};
        miss_prog_group_desc.kind                      = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module               = m_module;
        if( withParams )
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms_withParams";
        else
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &miss_prog_group_desc, 1, &program_group_options,
                                                       nullptr, nullptr, &m_miss_prog_group ) );
    }
    void createExceptionProgramGroup( bool withParams = false )
    {
        OptixProgramGroupOptions program_group_options  = {};
        OptixProgramGroupDesc    except_prog_group_desc = {};
        except_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
        except_prog_group_desc.exception.module         = m_module;
        if( withParams )
            except_prog_group_desc.exception.entryFunctionName = "__exception__ex_withParams";
        else
            except_prog_group_desc.exception.entryFunctionName = "__exception__ex";
        ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &except_prog_group_desc, 1, &program_group_options,
                                                       nullptr, nullptr, &m_exception_prog_group ) );
    }
    void createHitgroupProgramGroup_CH( bool withParams = false )
    {
        OptixProgramGroupOptions program_group_options = {};
        OptixProgramGroupDesc    hit_prog_group_desc   = {};
        hit_prog_group_desc.kind                       = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH          = m_module;
        if( withParams )
            hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch_withParams";
        else
            hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                       nullptr, nullptr, &m_hitgroup_prog_group ) );
    }
    void createHitgroupProgramGroup_AH( bool withParams = false )
    {
        OptixProgramGroupOptions program_group_options = {};
        OptixProgramGroupDesc    hit_prog_group_desc   = {};
        hit_prog_group_desc.kind                       = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleAH          = m_module;
        if( withParams )
            hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah_withParams";
        else
            hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
        ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                       nullptr, nullptr, &m_hitgroup_prog_group ) );
    }
    void createHitgroupProgramGroup_IS( bool withParams = false )
    {
        OptixProgramGroupOptions program_group_options = {};
        OptixProgramGroupDesc    hit_prog_group_desc   = {};
        hit_prog_group_desc.kind                       = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleIS          = m_module;
        if( withParams )
            hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is_withParams";
        else
            hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";
        ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                       nullptr, nullptr, &m_hitgroup_prog_group ) );
    }
    void createCallablesProgramGroup_DC( bool withParams = false )
    {
        OptixProgramGroupOptions program_group_options    = {};
        OptixProgramGroupDesc    callable_prog_group_desc = {};
        callable_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        callable_prog_group_desc.callables.moduleDC       = m_module;
        if( withParams )
            callable_prog_group_desc.callables.entryFunctionNameDC = "__direct_callable__dc_withParams";
        else
            callable_prog_group_desc.callables.entryFunctionNameDC = "__direct_callable__dc";
        ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &callable_prog_group_desc, 1, &program_group_options,
                                                       nullptr, nullptr, &m_callables_prog_group ) );
    }
    void createCallablesProgramGroup_CC( bool withParams = false )
    {
        OptixProgramGroupOptions program_group_options    = {};
        OptixProgramGroupDesc    callable_prog_group_desc = {};
        callable_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        callable_prog_group_desc.callables.moduleCC       = m_module;
        if( withParams )
            callable_prog_group_desc.callables.entryFunctionNameCC = "__continuation_callable__cc_withParams";
        else
            callable_prog_group_desc.callables.entryFunctionNameCC = "__continuation_callable__cc";
        ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &callable_prog_group_desc, 1, &program_group_options,
                                                       nullptr, nullptr, &m_callables_prog_group ) );
    }
    void TearDown() override { ASSERT_OPTIX_SUCCESS( optixDeviceContextDestroy( m_context ) ); }

    OptixDeviceContext   m_context = nullptr;
    OptixRecordingLogger m_logger;
    char                 logString[1024];
    size_t               logStringSize = 1024;

    OptixPipelineCompileOptions m_pipeline_compile_options = {};
    OptixPipelineLinkOptions    m_pipeline_link_options    = {};
    OptixModuleCompileOptions   m_module_compile_options   = {};
    OptixProgramGroup           m_raygen_prog_group        = nullptr;
    OptixProgramGroup           m_miss_prog_group          = nullptr;
    OptixProgramGroup           m_exception_prog_group     = nullptr;
    OptixProgramGroup           m_hitgroup_prog_group      = nullptr;
    OptixProgramGroup           m_callables_prog_group     = nullptr;
    OptixPipeline               m_pipeline                 = nullptr;
    OptixModule                 m_module                   = nullptr;

    // Aiming at a minimal ptx file.
    std::string m_ptx_str =
        ".version 6.4\n.target sm_30\n.address_size 64\n"
        ".visible .entry __raygen__rg()\n{ret;}\n"
        ".visible .entry __miss__ms()\n{ret;}\n"
        ".visible .entry __exception__ex()\n{ret;}\n"
        ".visible .entry __closesthit__ch()\n{ret;}\n"
        ".visible .entry __anyhit__ah()\n{ret;}\n"
        ".visible .entry __intersection__is()\n{ret;}\n"
        ".visible .entry __direct_callable__dc()\n{ret;}\n"
        ".visible .entry __continuation_callable__cc()\n{ret;}\n";
};

class O7_API_optixLaunch : public O7_API_optixSbtRecordPackHeader, public testing::WithParamInterface<bool>
{
  public:
    void SetUp() override
    {
        exptest::lwdaInitialize();
        ASSERT_OPTIX_SUCCESS( optixInit() );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( nullptr, nullptr, &m_context ) );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextSetLogCallback( m_context, &OptixRecordingLogger::callback, &m_logger, 2 ) );
    }
    // separate from SetupLaunch() to allow overrides
    void SetupPipelineCompileOptions()
    {
        m_pipeline_compile_options.usesMotionBlur = 0;
        // How much storage, in 32b words, to make available for the payload, [0..8]
        m_pipeline_compile_options.numPayloadValues = 3;
        // How much storage, in 32b words, to make available for the attributes. The
        // minimum number is 2. Values below that will automatically be changed to 2. [2..8]
        m_pipeline_compile_options.numAttributeValues = 3;
        /// A bitmask of OptixExceptionFlags indicating which exceptions are enabled.
        m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

        if( GetParam() )
            m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    }
    void SetupLaunch( bool skipExceptionProgram = false )
    {
        // module creation requires initialized pipeline compile options
        OPTIX_CHECK( optixModuleCreateFromPTX( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                               optix::data::gettest_LaunchesSources()[1],
                                               optix::data::gettest_LaunchesSourceSizes()[0], nullptr, nullptr, &m_module ) );
        // what program groups shall we use? We start with all.
        createRayGenProgramGroup( GetParam() );
        if( !skipExceptionProgram )
            createExceptionProgramGroup( GetParam() );
        createMissProgramGroup( GetParam() );
        createHitgroupProgramGroup( GetParam() );
        createCallablesProgramGroup( GetParam() );
        std::vector<OptixProgramGroup> program_groups;
        program_groups.push_back( m_raygen_prog_group );
        if( !skipExceptionProgram )
            program_groups.push_back( m_exception_prog_group );
        program_groups.push_back( m_miss_prog_group );
        program_groups.push_back( m_hitgroup_prog_group );
        program_groups.push_back( m_callables_prog_group );

        // setup sbt
        {
            LWdeviceptr  raygen_record;
            const size_t raygen_record_size = sizeof( RayGenSbtRecord );
            LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
            RayGenSbtRecord rg_sbt;
            rg_sbt.data = {};
            OPTIX_CHECK( optixSbtRecordPackHeader( m_raygen_prog_group, &rg_sbt ) );
            LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( raygen_record ), &rg_sbt, raygen_record_size, lwdaMemcpyHostToDevice ) );
            m_sbt.raygenRecord = raygen_record;
        }
        if( !skipExceptionProgram )
        {
            LWdeviceptr  exception_record;
            const size_t exception_record_size = sizeof( ExceptionSbtRecord );
            LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &exception_record ), exception_record_size ) );
            ExceptionSbtRecord ex_sbt;
            ex_sbt.data = {};
            OPTIX_CHECK( optixSbtRecordPackHeader( m_exception_prog_group, &ex_sbt ) );
            LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( exception_record ), &ex_sbt, exception_record_size, lwdaMemcpyHostToDevice ) );
            m_sbt.exceptionRecord = exception_record;
        }
        {
            LWdeviceptr  miss_record;
            const size_t miss_record_size = sizeof( MissSbtRecord );
            LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
            MissSbtRecord miss_sbt;
            miss_sbt.data = {};
            OPTIX_CHECK( optixSbtRecordPackHeader( m_miss_prog_group, &miss_sbt ) );
            LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( miss_record ), &miss_sbt, miss_record_size, lwdaMemcpyHostToDevice ) );
            m_sbt.missRecordBase          = miss_record;
            m_sbt.missRecordCount         = 1;
            m_sbt.missRecordStrideInBytes = miss_record_size;
        }
        {
            LWdeviceptr  hitgroup_record;
            const size_t hitgroup_record_size = sizeof( HitGroupSbtRecord );
            LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
            HitGroupSbtRecord hit_sbt;
            hit_sbt.data = {};
            OPTIX_CHECK( optixSbtRecordPackHeader( m_hitgroup_prog_group, &hit_sbt ) );
            LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( hitgroup_record ), &hit_sbt, hitgroup_record_size, lwdaMemcpyHostToDevice ) );
            m_sbt.hitgroupRecordBase          = hitgroup_record;
            m_sbt.hitgroupRecordCount         = 1;
            m_sbt.hitgroupRecordStrideInBytes = hitgroup_record_size;
        }
        {
            LWdeviceptr  callables_record;
            const size_t callables_record_size = sizeof( CallablesSbtRecord );
            LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &callables_record ), callables_record_size ) );
            CallablesSbtRecord callables_sbt;
            callables_sbt.data = {};
            OPTIX_CHECK( optixSbtRecordPackHeader( m_callables_prog_group, &callables_sbt ) );
            LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( callables_record ), &callables_sbt, callables_record_size,
                                    lwdaMemcpyHostToDevice ) );
            m_sbt.callablesRecordBase          = callables_record;
            m_sbt.callablesRecordCount         = 1;
            m_sbt.callablesRecordStrideInBytes = callables_record_size;
        }

        LWDA_CHECK( lwdaStreamCreate( &m_stream ) );
        ASSERT_OPTIX_SUCCESS( optixPipelineCreate( m_context, &m_pipeline_compile_options, &m_pipeline_link_options,
                                                   &program_groups[0], static_cast<unsigned int>( program_groups.size() ),
                                                   logString, &logStringSize, &m_pipeline ) );
        width  = 1;
        height = 1;
        depth  = 1;

        Params params        = {};
        m_pipelineParamsSize = sizeof( Params );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &m_pipelineParams ), m_pipelineParamsSize ) );
        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( m_pipelineParams ), &params, m_pipelineParamsSize, lwdaMemcpyHostToDevice ) );
    }
    void createHitgroupProgramGroup( bool withParams = false )
    {
        OptixProgramGroupOptions program_group_options = {};
        OptixProgramGroupDesc    hit_prog_group_desc   = {};
        hit_prog_group_desc.kind                       = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH          = m_module;
        hit_prog_group_desc.hitgroup.moduleAH          = m_module;
        hit_prog_group_desc.hitgroup.moduleIS          = m_module;
        if( withParams )
        {
            hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch_withParams";
            hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah_withParams";
            hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is_withParams";
        }
        else
        {
            hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
            hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";
        }
        ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                       nullptr, nullptr, &m_hitgroup_prog_group ) );
    }
    void createCallablesProgramGroup( bool withParams = false )
    {
        OptixProgramGroupOptions program_group_options    = {};
        OptixProgramGroupDesc    callable_prog_group_desc = {};
        callable_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        callable_prog_group_desc.callables.moduleDC       = m_module;
        callable_prog_group_desc.callables.moduleCC       = m_module;
        if( withParams )
        {
            callable_prog_group_desc.callables.entryFunctionNameDC = "__direct_callable__dc_withParams";
            callable_prog_group_desc.callables.entryFunctionNameCC = "__continuation_callable__cc_withParams";
        }
        else
        {
            callable_prog_group_desc.callables.entryFunctionNameDC = "__direct_callable__dc";
            callable_prog_group_desc.callables.entryFunctionNameCC = "__continuation_callable__cc";
        }
        ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &callable_prog_group_desc, 1, &program_group_options,
                                                       nullptr, nullptr, &m_callables_prog_group ) );
    }

    LWstream                m_stream             = {};
    LWdeviceptr             m_pipelineParams     = {};
    size_t                  m_pipelineParamsSize = 0;
    OptixShaderBindingTable m_sbt                = {};
    unsigned int            width                = 0;
    unsigned int            height               = 0;
    unsigned int            depth                = 0;
};

//---------------------------------------------------------------------------
// 	optixSbtRecordPackHeader
//---------------------------------------------------------------------------

TEST_F( O7_API_optixSbtRecordPackHeader, DontRun_with_nullProgramGroup )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixSbtRecordPackHeader( nullptr, nullptr ) );
}


TEST_F( O7_API_optixSbtRecordPackHeader, DontRun_with_nullHostPointer )
{
    createRayGenProgramGroup();
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixSbtRecordPackHeader( m_raygen_prog_group, nullptr ) );
}


TEST_F( O7_API_optixSbtRecordPackHeader, RunWithSuccess_with_Raygen )
{
    createRayGenProgramGroup();

    RayGenSbtRecord rg_sbt;
    rg_sbt.data = {};
    ASSERT_OPTIX_SUCCESS( optixSbtRecordPackHeader( m_raygen_prog_group, &rg_sbt ) );
    // does repeated filling work?
    ASSERT_OPTIX_SUCCESS( optixSbtRecordPackHeader( m_raygen_prog_group, &rg_sbt ) );
}


TEST_F( O7_API_optixSbtRecordPackHeader, RunWithSuccess_with_Miss )
{
    createMissProgramGroup();

    MissSbtRecord miss_sbt;
    miss_sbt.data = {};
    ASSERT_OPTIX_SUCCESS( optixSbtRecordPackHeader( m_miss_prog_group, &miss_sbt ) );
}


TEST_F( O7_API_optixSbtRecordPackHeader, RunWithSuccess_with_Exception )
{
    createExceptionProgramGroup();

    MissSbtRecord exceptions_sbt;
    exceptions_sbt.data = {};
    ASSERT_OPTIX_SUCCESS( optixSbtRecordPackHeader( m_exception_prog_group, &exceptions_sbt ) );
}


TEST_F( O7_API_optixSbtRecordPackHeader, RunWithSuccess_with_Hitgroup_CH )
{
    createHitgroupProgramGroup_CH();

    HitGroupSbtRecord hit_sbt;
    hit_sbt.data = {};
    ASSERT_OPTIX_SUCCESS( optixSbtRecordPackHeader( m_hitgroup_prog_group, &hit_sbt ) );
}


TEST_F( O7_API_optixSbtRecordPackHeader, RunWithSuccess_with_Hitgroup_AH )
{
    createHitgroupProgramGroup_AH();

    HitGroupSbtRecord hit_sbt;
    hit_sbt.data = {};
    ASSERT_OPTIX_SUCCESS( optixSbtRecordPackHeader( m_hitgroup_prog_group, &hit_sbt ) );
}


TEST_F( O7_API_optixSbtRecordPackHeader, RunWithSuccess_with_Hitgroup_IS )
{
    createHitgroupProgramGroup_IS();

    HitGroupSbtRecord hit_sbt;
    hit_sbt.data = {};
    ASSERT_OPTIX_SUCCESS( optixSbtRecordPackHeader( m_hitgroup_prog_group, &hit_sbt ) );
}


TEST_F( O7_API_optixSbtRecordPackHeader, RunWithSuccess_with_CallablesDC )
{
    createCallablesProgramGroup_DC();

    CallablesSbtRecord callables_sbt;
    callables_sbt.data = {};
    ASSERT_OPTIX_SUCCESS( optixSbtRecordPackHeader( m_callables_prog_group, &callables_sbt ) );
}


TEST_F( O7_API_optixSbtRecordPackHeader, RunWithSuccess_with_CallablesCC )
{
    createCallablesProgramGroup_CC();

    CallablesSbtRecord callables_sbt;
    callables_sbt.data = {};
    ASSERT_OPTIX_SUCCESS( optixSbtRecordPackHeader( m_callables_prog_group, &callables_sbt ) );
}

//---------------------------------------------------------------------------
// 	optixLaunch
//---------------------------------------------------------------------------

TEST_P( O7_API_optixLaunch, RunWithSuccess )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    ASSERT_OPTIX_SUCCESS( optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
}


TEST_P( O7_API_optixLaunch, Check_pipelineParamsButPipelineParamsSizeIsNull )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    OptixResult res = optixLaunch( m_pipeline, m_stream, m_pipelineParams, 0, &m_sbt, width, height, depth );
    if( GetParam() )
    {
        ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, res );
        EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"pipelineParams\" specified with zero "
                                                                           "\"pipelineParamsSize\"" ) );
    }
    else
        ASSERT_EQ( OPTIX_SUCCESS, res );
}


TEST_P( O7_API_optixLaunch, Check_pipelineParamsIsNullButPipelineParamsSizeIsNotNull )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    OptixResult res = optixLaunch( m_pipeline, m_stream, 0, m_pipelineParamsSize, &m_sbt, width, height, depth );
    if( GetParam() )
    {
        ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, res );
        EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"pipelineParamsSize\" specified with null "
                                                                           "\"pipelineParams\"" ) );
    }
    else
        ASSERT_EQ( OPTIX_SUCCESS, res );
}


// This error check inside checkPipelineLaunchParams() cannot be triggered. If there is no pipelineLaunchParamsVariableName
// the function extractPipelineParamsSize() during module compilation will leave the value at its default
// Module::s_ilwalidPipelineParamsSize and hence the pipeline will ignore any of its launchParams.
TEST_P( O7_API_optixLaunch, DISABLED_Check_pipelineParamsButpipelineLaunchParamsVariableNameIsNull )
{
    SetupPipelineCompileOptions();
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = 0;
    SetupLaunch();

    OptixResult res = optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth );
    if( GetParam() )
    {
        ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, res );
        EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "no variable name to bind "
                                                                           "\"pipelineParams\" to specified in "
                                                                           "pipeline compile options" ) );
    }
    else
        ASSERT_EQ( OPTIX_SUCCESS, res );
}


TEST_P( O7_API_optixLaunch, Check_pipelineParamsIsNullButPipelineLaunchParamsVariableNameIsNotNull )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    OptixResult res = optixLaunch( m_pipeline, m_stream, 0, 0, &m_sbt, width, height, depth );
    if( GetParam() )
    {
        ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, res );
        EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"pipelineParams\" is null, but pipeline "
                                                                           "compile options specify a variable name to "
                                                                           "bind \"pipelineParams\" to" ) );
    }
    else
        ASSERT_EQ( OPTIX_SUCCESS, res );
}


TEST_P( O7_API_optixLaunch, Check_pipelineParamsSizeIsGreaterThanPipelineLaunchParamSize )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    OptixResult res = optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize + 1, &m_sbt, width, height, depth );
    if( GetParam() )
    {
        ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, res );
        EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "pipeline launch param size configured by "
                                                                           "pipeline link options " ) );
    }
    else
        ASSERT_EQ( OPTIX_SUCCESS, res );
}


TEST_P( O7_API_optixLaunch, DontRun_raygenRecordIsNull )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.raygenRecord = 0;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->raygenRecord\" is null" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_raygenRecordIsNotAligned )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.raygenRecord += 1;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->raygenRecord\" points to a memory area "
                                                                       "which is not correctly aligned" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_exceptionRecordIsNotAligned )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.exceptionRecord += 1;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->exceptionRecord\" points to a memory "
                                                                       "area which is not correctly aligned" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_missRecordIsNull )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.missRecordBase = 0;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->missSbtRecord\" is null" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_missRecordIsNotAligned )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.missRecordBase += 1;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->missSbtRecord\" points to a memory area "
                                                                       "which is not correctly aligned" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_missRecordCountIsNull )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.missRecordCount = 0;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->missRecordCount\" is zero" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_missRecordStrideInBytesIsNull )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.missRecordStrideInBytes = 0;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->missRecordStrideInBytes\" is zero" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_missRecordStrideInBytesIsNotAligned )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.missRecordStrideInBytes = 1;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->missRecordStrideInBytes\" is not a "
                                                                       "multiple of OPTIX_SBT_RECORD_ALIGNMENT" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_hitgroupRecordBaseIsNull )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.hitgroupRecordBase = 0;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->hitgroupRecordBase\" is null" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_hitgroupRecordBaseIsNotAligned )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.hitgroupRecordBase += 1;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->hitgroupRecordBase\" points to a memory "
                                                                       "area which is not correctly aligned" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_hitgroupRecordCountIsZero )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.hitgroupRecordCount = 0;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->hitgroupRecordCount\" is zero" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_hitgroupRecordStrideInBytesIsZero )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.hitgroupRecordStrideInBytes = 0;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->hitgroupRecordStrideInBytes\" is "
                                                                       "zero" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_hitgroupRecordStrideInBytesIsNotAligned )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.hitgroupRecordStrideInBytes = 1;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->hitgroupRecordStrideInBytes\" is not a "
                                                                       "multiple of OPTIX_SBT_RECORD_ALIGNMENT" ) );
}


TEST_P( O7_API_optixLaunch, Run_callablesRecordBaseIsNullAndcallablesRecordCountIsNull )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.callablesRecordBase  = 0;
    m_sbt.callablesRecordCount = 0;
    ASSERT_OPTIX_SUCCESS( optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
}


TEST_P( O7_API_optixLaunch, DontRun_callablesRecordBaseIsNullAndcallablesRecordCountIsNotNull )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.callablesRecordBase  = 0;
    m_sbt.callablesRecordCount = 1;

    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->callablesRecordBase\" is null, but "
                                                                       "\"sbt->callablesRecordCount\" is non-zero" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_callablesRecordBaseIsNotAligned )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.callablesRecordBase += 1;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->callablesRecordBase\" points to a "
                                                                       "memory area which is not correctly aligned" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_callablesRecordCountIsZero )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.callablesRecordCount = 0;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->callablesRecordBase\" is non-null, but "
                                                                       "\"sbt->callablesRecordCount\" is zero" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_callablesRecordStrideInBytesIsZero )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.callablesRecordStrideInBytes = 0;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->callablesRecordBase\" is non-null, but "
                                                                       "\"sbt->callablesRecordStrideInBytes\" is "
                                                                       "zero" ) );
}


TEST_P( O7_API_optixLaunch, DontRun_callablesRecordStrideInBytesIsNotAligned )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    m_sbt.callablesRecordStrideInBytes = 1;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "\"sbt->callablesRecordStrideInBytes\" is not a "
                                                                       "multiple of OPTIX_SBT_RECORD_ALIGNMENT" ) );
}


TEST_P( O7_API_optixLaunch, UseDefaultExceptionProgram )
{
    SetupPipelineCompileOptions();
    m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    bool skipUserExceptionProgram             = true;
    SetupLaunch( skipUserExceptionProgram );

    ASSERT_OPTIX_SUCCESS( optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
}


TEST_P( O7_API_optixLaunch, DontRunWithNullArguments )
{
    SetupPipelineCompileOptions();
    SetupLaunch();

    // pipeline = nullptr
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( nullptr, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );

    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, nullptr, width, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "sbt is null" ) );

    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, 0, height, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "width is 0" ) );

    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, 0, depth ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "height is 0" ) );

    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, 0 ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "depth is 0" ) );

    // just to make sure that there aren't any persistent error states still alive
    ASSERT_OPTIX_SUCCESS( optixLaunch( m_pipeline, m_stream, m_pipelineParams, m_pipelineParamsSize, &m_sbt, width, height, depth ) );
}

INSTANTIATE_TEST_SUITE_P( RunWithAndWithoutParams, O7_API_optixLaunch, testing::Bool() );
