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

using namespace testing;

class O7_API_optixPipelineCreate : public Test
{
  protected:
    void SetUp() override
    {
        exptest::lwdaInitialize();
        ASSERT_OPTIX_SUCCESS( optixInit() );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( nullptr, nullptr, &m_context ) );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextSetLogCallback( m_context, &OptixRecordingLogger::callback, &m_logger, 2 ) );

        OptixModule module = {};
        {
            OPTIX_CHECK( optixModuleCreateFromPTX( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                   m_ptx_str.c_str(), m_ptx_str.size(), nullptr, nullptr, &module ) );
        }

        OptixProgramGroupOptions program_group_options  = {};
        OptixProgramGroupDesc    raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__trace";
        ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &raygen_prog_group_desc, 1, &program_group_options,
                                                       nullptr, nullptr, &m_raygen_prog_group ) );
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
    OptixPipeline               m_pipeline                 = nullptr;

    // Aiming at a minimal ptx file.
    std::string m_ptx_str =
        ".version 6.4\n.target sm_30\n.address_size 64\n"
        ".visible .entry __raygen__trace()\n{ret;}\n";
};


// This class allows to override the different options before a deferred setup call.
class O7_API_optixPipelineCreate_S : public O7_API_optixPipelineCreate
{
  protected:
    void SetUp() override {}
    void SetupDeferred()
    {
        exptest::lwdaInitialize();
        ASSERT_OPTIX_SUCCESS( optixInit() );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( nullptr, nullptr, &m_context ) );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextSetLogCallback( m_context, &OptixRecordingLogger::callback, &m_logger, 2 ) );

        OptixModule module = {};
        {
            OPTIX_CHECK( optixModuleCreateFromPTX( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                   m_ptx_str.c_str(), m_ptx_str.size(), nullptr, nullptr, &module ) );
        }

        OptixProgramGroupOptions program_group_options  = {};
        OptixProgramGroupDesc    raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__trace";
        ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &raygen_prog_group_desc, 1, &program_group_options,
                                                       nullptr, nullptr, &m_raygen_prog_group ) );
    }
};


// The sole reason for its existence is that the optixPipelineDestroy tests are properly named but
// can use the functionality to create a pipeline.
class O7_API_optixPipelineDestroy : public O7_API_optixPipelineCreate
{
};


//---------------------------------------------------------------------------
// 	optixPipelineCreate
//---------------------------------------------------------------------------

TEST_F( O7_API_optixPipelineCreate, RunWithNullContext )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT,
               optixPipelineCreate( nullptr, &m_pipeline_compile_options, &m_pipeline_link_options,
                                    &m_raygen_prog_group, 1, logString, &logStringSize, &m_pipeline ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixPipelineCreate, RunWithSuccess )
{
    ASSERT_OPTIX_SUCCESS( optixPipelineCreate( m_context, &m_pipeline_compile_options, &m_pipeline_link_options,
                                               &m_raygen_prog_group, 1, logString, &logStringSize, &m_pipeline ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixPipelineCreate, RunWithNoPipelineCompileOptions )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixPipelineCreate( m_context, nullptr, &m_pipeline_link_options, &m_raygen_prog_group,
                                                               1, logString, &logStringSize, &m_pipeline ) );
    ASSERT_STREQ( "pipelineCompileOptions is null", m_logger.getMessagesAsOneString().c_str() );
    ASSERT_STREQ( "pipelineCompileOptions is null", logString );
}


TEST_F( O7_API_optixPipelineCreate, RunWithNoPipelineLinkOptions )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixPipelineCreate( m_context, &m_pipeline_compile_options, nullptr, &m_raygen_prog_group,
                                                               1, logString, &logStringSize, &m_pipeline ) );
    ASSERT_STREQ( "pipelineLinkOptions is null", m_logger.getMessagesAsOneString().c_str() );
    ASSERT_STREQ( "pipelineLinkOptions is null", logString );
}


TEST_F( O7_API_optixPipelineCreate, RunWithNoProgramGroups )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixPipelineCreate( m_context, &m_pipeline_compile_options, &m_pipeline_link_options,
                                                               nullptr, 1, logString, &logStringSize, &m_pipeline ) );
    ASSERT_STREQ( "programGroups is null", m_logger.getMessagesAsOneString().c_str() );
    ASSERT_STREQ( "programGroups is null", logString );
}


TEST_F( O7_API_optixPipelineCreate, RunWithNoProgramGroupsCount )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixPipelineCreate( m_context, &m_pipeline_compile_options, &m_pipeline_link_options,
                                                               &m_raygen_prog_group, 0, logString, &logStringSize, &m_pipeline ) );
    ASSERT_STREQ( "numProgramGroups is 0", m_logger.getMessagesAsOneString().c_str() );
    ASSERT_STREQ( "numProgramGroups is 0", logString );
}


TEST_F( O7_API_optixPipelineCreate, RunWithNoPipelineAPI )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixPipelineCreate( m_context, &m_pipeline_compile_options, &m_pipeline_link_options,
                                                               &m_raygen_prog_group, 1, logString, &logStringSize, nullptr ) );
    ASSERT_STREQ( "pipelineAPI is null", m_logger.getMessagesAsOneString().c_str() );
    ASSERT_STREQ( "pipelineAPI is null", logString );
}


TEST_F( O7_API_optixPipelineCreate, RunWithEmptyProgramGroups )
{
    OptixProgramGroup raygen_prog_group = nullptr;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixPipelineCreate( m_context, &m_pipeline_compile_options, &m_pipeline_link_options,
                                                               &raygen_prog_group, 1, logString, &logStringSize, &m_pipeline ) );
    ASSERT_STREQ( "COMPILE ERROR: programGroups[0] argument is null\n", m_logger.getMessagesAsOneString().c_str() );
    ASSERT_STREQ( "COMPILE ERROR: programGroups[0] argument is null\n", logString );
}


//
// Check different legal and illegal options settings, eg OptixPipelineCompileOptions::usesMotionBlur=2
//

TEST_F( O7_API_optixPipelineCreate_S, OptixPipelineCompileOptions_RunWith_usesMotionBlur_1 )
{
    m_pipeline_compile_options.usesMotionBlur = 1;
    SetupDeferred();
    ASSERT_OPTIX_SUCCESS( optixPipelineCreate( m_context, &m_pipeline_compile_options, &m_pipeline_link_options,
                                               &m_raygen_prog_group, 1, logString, &logStringSize, &m_pipeline ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixPipelineCreate_S, OptixPipelineCompileOptions_DontRunWith_usesMotionBlur_2 )
{
    m_pipeline_compile_options.usesMotionBlur = 2;
    SetupDeferred();
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixPipelineCreate( m_context, &m_pipeline_compile_options, &m_pipeline_link_options,
                                                               &m_raygen_prog_group, 1, logString, &logStringSize, &m_pipeline ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "usesMotionBlur is neither 0 nor 1" ) );
    EXPECT_THAT( logString, HasSubstr( "usesMotionBlur is neither 0 nor 1" ) );
}


TEST_F( O7_API_optixPipelineCreate_S, OptixPipelineLinkOptions_RunWith_maxTraceDepth_0 )
{
    m_pipeline_link_options.maxTraceDepth = 0;
    SetupDeferred();
    ASSERT_OPTIX_SUCCESS( optixPipelineCreate( m_context, &m_pipeline_compile_options, &m_pipeline_link_options,
                                               &m_raygen_prog_group, 1, logString, &logStringSize, &m_pipeline ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixPipelineCreate_S, OptixPipelineLinkOptions_DontRunWith_maxTraceDepth_32 )
{
    m_pipeline_link_options.maxTraceDepth = 32;
    SetupDeferred();
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixPipelineCreate( m_context, &m_pipeline_compile_options, &m_pipeline_link_options,
                                                               &m_raygen_prog_group, 1, logString, &logStringSize, &m_pipeline ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "maxTraceDepth too high" ) );
    EXPECT_THAT( logString, HasSubstr( "maxTraceDepth too high" ) );
}


TEST_F( O7_API_optixPipelineCreate_S, Illegal_OptixPipelineCompileOptions_and_OptixPipelineLinkOptions )
{
    m_pipeline_compile_options.usesMotionBlur = 2;
    m_pipeline_link_options.maxTraceDepth     = 32;
    SetupDeferred();
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixPipelineCreate( m_context, &m_pipeline_compile_options, &m_pipeline_link_options,
                                                               &m_raygen_prog_group, 1, logString, &logStringSize, &m_pipeline ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "usesMotionBlur is neither 0 nor 1" ) );
    EXPECT_THAT( logString, HasSubstr( "usesMotionBlur is neither 0 nor 1" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "maxTraceDepth too high" ) );
    EXPECT_THAT( logString, HasSubstr( "maxTraceDepth too high" ) );
}


//---------------------------------------------------------------------------
// 	optixPipelineDestroy
//---------------------------------------------------------------------------

TEST_F( O7_API_optixPipelineDestroy, RunWithNullptr )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixPipelineDestroy( nullptr ) );
}

TEST_F( O7_API_optixPipelineDestroy, RunWithSuccess )
{
    ASSERT_OPTIX_SUCCESS( optixPipelineCreate( m_context, &m_pipeline_compile_options, &m_pipeline_link_options,
                                               &m_raygen_prog_group, 1, logString, &logStringSize, &m_pipeline ) );
    ASSERT_OPTIX_SUCCESS( optixPipelineDestroy( m_pipeline ) );
}
