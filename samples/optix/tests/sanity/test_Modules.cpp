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
#include <queue>

using namespace testing;

class O7_API_optixModuleCreate : public Test
{
  protected:
    void SetUp() override
    {
        exptest::lwdaInitialize();
        ASSERT_OPTIX_SUCCESS( optixInit() );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( nullptr, nullptr, &m_context ) );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextSetLogCallback( m_context, &OptixRecordingLogger::callback, &m_logger, 2 ) );
    }
    void TearDown() override { ASSERT_OPTIX_SUCCESS( optixDeviceContextDestroy( m_context ) ); }

    OptixDeviceContext   m_context = nullptr;
    OptixRecordingLogger m_logger;
    char                 logString[1024];
    size_t               logStringSize = 1024;

    OptixPipelineCompileOptions m_pipeline_compile_options = {};
    OptixModuleCompileOptions   m_module_compile_options   = {};

    // Aiming at a minimal ptx file.
    std::string m_ptx_str =
        ".version 6.4\n.target sm_30\n.address_size 64\n"
        ".visible .entry __raygen__trace()\n{ret;}\n";
};

class O7_API_optixModuleCreateWithTask : public O7_API_optixModuleCreate {};
class O7_API_optixModuleGetCompilationState : public O7_API_optixModuleCreate {};


// The sole reason for its existence is that the optixModuleDestroy tests are properly named but
// can use the functionality to create a module.
class O7_API_optixModuleDestroy : public O7_API_optixModuleCreate
{
};


TEST_F( O7_API_optixModuleCreate, RunWithSuccess )
{
    OptixModule module = {};
    ASSERT_OPTIX_SUCCESS( optixModuleCreateFromPTX( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                    m_ptx_str.c_str(), m_ptx_str.size(), logString, &logStringSize, &module ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixModuleCreate, RunWithInsufficientLogStringSize )
{
    OptixModule module = {};
    char        insufficientLogString[1];
    size_t      insufficientLogStringSize = 1;

    ASSERT_OPTIX_SUCCESS( optixModuleCreateFromPTX( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                    m_ptx_str.c_str(), m_ptx_str.size(), insufficientLogString,
                                                    &insufficientLogStringSize, &module ) );

    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


// Disabled due to Bug 2726201 "OptiX frontend does not recover from .ptx parsing failures"
TEST_F( O7_API_optixModuleCreate, DISABLED_RunWithBrokenPTX )
{
    OptixModule module = {};
    // break the ptx by obmitting the first char
    std::string ptx_str = m_ptx_str.substr( 1 );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_PTX,
               optixModuleCreateFromPTX( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                         ptx_str.c_str(), ptx_str.size(), logString, &logStringSize, &module ) );
    EXPECT_THAT( logString, HasSubstr( "Invalid PTX input" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "Invalid PTX input" ) );
}


TEST_F( O7_API_optixModuleCreate, OptixModuleCompileOptions_DontRunWith_conflictingOptions )
{
    OptixModule module                  = {};
    m_module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    m_module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_1;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixModuleCreateFromPTX( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                         m_ptx_str.c_str(), m_ptx_str.size(), logString, &logStringSize, &module ) );
    EXPECT_THAT( logString, HasSubstr( "requires optimization level" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "requires optimization level" ) );
}


TEST_F( O7_API_optixModuleCreate, OptixModuleCompileOptions_RunWith_nonconflictingOptions )
{
    OptixModule module                  = {};
    m_module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    m_module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    ASSERT_OPTIX_SUCCESS( optixModuleCreateFromPTX( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                    m_ptx_str.c_str(), m_ptx_str.size(), logString, &logStringSize, &module ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixModuleCreate, OptixModuleCompileOptions_RunWith_minimal )
{
    OptixModule module                  = {};
    m_module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    m_module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    ASSERT_OPTIX_SUCCESS( optixModuleCreateFromPTX( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                    m_ptx_str.c_str(), m_ptx_str.size(), logString, &logStringSize, &module ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixModuleCreate, OptixModuleCompileOptions_RunWith_moderate )
{
    OptixModule module                  = {};
    m_module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
    m_module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    ASSERT_OPTIX_SUCCESS( optixModuleCreateFromPTX( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                    m_ptx_str.c_str(), m_ptx_str.size(), logString, &logStringSize, &module ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixModuleCreate, OptixModuleCompileOptions_RunWith_ilwalidMaxRegisterCount )
{
    OptixModule module                        = {};
    m_module_compile_options.maxRegisterCount = -1;
    ASSERT_OPTIX_ILWALID_VALUE( optixModuleCreateFromPTX( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                          m_ptx_str.c_str(), m_ptx_str.size(), logString, &logStringSize, &module ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "maxRegisterCount must be >= 0." ) );
    ASSERT_LT( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixModuleCreateWithTask, RunWithSuccess )
{
    OptixModule module = {};
    OptixTask task;

    ASSERT_OPTIX_SUCCESS( optixModuleCreateFromPTXWithTasks( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                    m_ptx_str.c_str(), m_ptx_str.size(), logString, &logStringSize, &module, &task ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixModuleCreateWithTask, RunWithInsufficientLogStringSize )
{
    OptixModule module = {};
    OptixTask task;
    char        insufficientLogString[1];
    size_t      insufficientLogStringSize = 1;

    ASSERT_OPTIX_SUCCESS( optixModuleCreateFromPTXWithTasks( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                    m_ptx_str.c_str(), m_ptx_str.size(), insufficientLogString,
                                                    &insufficientLogStringSize, &module, &task ) );

    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixModuleCreateWithTask, OptixModuleCompileOptions_DontRunWith_conflictingOptions )
{
    OptixModule module                  = {};
    OptixTask task                      = {};
    m_module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    m_module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_1;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE,
               optixModuleCreateFromPTXWithTasks( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                         m_ptx_str.c_str(), m_ptx_str.size(), logString, &logStringSize, &module, &task ) );
    EXPECT_THAT( logString, HasSubstr( "requires optimization level" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "requires optimization level" ) );
}


TEST_F( O7_API_optixModuleCreateWithTask, OptixModuleCompileOptions_RunWith_nonconflictingOptions )
{
    OptixModule module                  = {};
    OptixTask task                      = {};
    m_module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    m_module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    ASSERT_OPTIX_SUCCESS( optixModuleCreateFromPTXWithTasks( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                    m_ptx_str.c_str(), m_ptx_str.size(), logString, &logStringSize, &module, &task ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixModuleCreateWithTask, OptixModuleCompileOptions_RunWith_minimal )
{
    OptixModule module                  = {};
    OptixTask task                      = {};
    m_module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    m_module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    ASSERT_OPTIX_SUCCESS( optixModuleCreateFromPTXWithTasks( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                    m_ptx_str.c_str(), m_ptx_str.size(), logString, &logStringSize, &module, &task ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixModuleCreateWithTask, OptixModuleCompileOptions_RunWith_moderate )
{
    OptixModule module                  = {};
    OptixTask task                      = {};
    m_module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
    m_module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    ASSERT_OPTIX_SUCCESS( optixModuleCreateFromPTXWithTasks( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                    m_ptx_str.c_str(), m_ptx_str.size(), logString, &logStringSize, &module, &task ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixModuleCreateWithTask, OptixModuleCompileOptions_RunWith_ilwalidMaxRegisterCount )
{
    OptixModule module                        = {};
    OptixTask task                            = {};
    m_module_compile_options.maxRegisterCount = -1;
    ASSERT_OPTIX_ILWALID_VALUE( optixModuleCreateFromPTXWithTasks( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                          m_ptx_str.c_str(), m_ptx_str.size(), logString, &logStringSize, &module, &task ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "maxRegisterCount must be >= 0." ) );
    ASSERT_LT( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixModuleGetCompilationState, RunWithTask_checkStartedAndCompleted )
{
    OptixModule module                       = {};
    OptixTask initialTask                    = {};
    OptixModuleCompileState state            = {};
    const unsigned int maxNumAdditionalTasks = 2;

    ASSERT_OPTIX_SUCCESS( optixModuleCreateFromPTXWithTasks( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                    m_ptx_str.c_str(), m_ptx_str.size(), logString, &logStringSize, &module, &initialTask ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );

    ASSERT_OPTIX_SUCCESS( optixModuleGetCompilationState( module, &state ) );
    ASSERT_EQ( OPTIX_MODULE_COMPILE_STATE_NOT_STARTED, state );

    std::queue<OptixTask> remainingTasks;
    remainingTasks.push( initialTask );

    while(!remainingTasks.empty())
    {
        OptixTask lwrrenttask = remainingTasks.front();
        std::vector<OptixTask> additionalTasks( maxNumAdditionalTasks );
        unsigned int numAdditionalTasksCreated;
        ASSERT_OPTIX_SUCCESS( optixTaskExelwte( lwrrenttask, additionalTasks.data(), maxNumAdditionalTasks, &numAdditionalTasksCreated ) );

        // If additional tasks are created, we may check for OPTIX_MODULE_COMPILE_STATE_STARTED prior to all tasks being exelwted.
        if( numAdditionalTasksCreated > 0)
        {
            ASSERT_OPTIX_SUCCESS( optixModuleGetCompilationState( module, &state ) );
            ASSERT_EQ( OPTIX_MODULE_COMPILE_STATE_STARTED, state );
        }

        remainingTasks.pop();
        for( int i = 0; i < numAdditionalTasksCreated; ++i )
            remainingTasks.push( additionalTasks[i] );
    }

    ASSERT_OPTIX_SUCCESS( optixModuleGetCompilationState( module, &state ) );
    ASSERT_EQ( OPTIX_MODULE_COMPILE_STATE_COMPLETED, state );
}


enum MaxRegisterCount
{
    MAX_REGISTER_COUNT_ZERO = 0,
    MAX_REGISTER_COUNT_1    = 1,
    MAX_REGISTER_COUNT_64   = 64,
    MAX_REGISTER_COUNT_96   = 96,
    MAX_REGISTER_COUNT_128  = 128
};

class O7_API_optixModuleCreateWithMaxRegisterCounts : public O7_API_optixModuleCreate,
                                                      public testing::WithParamInterface<MaxRegisterCount> {};
TEST_P( O7_API_optixModuleCreateWithMaxRegisterCounts, OptixModuleCompileOptions_RunWith_varyMaxRegisterCount )
{
    OptixModule module = {};
    m_module_compile_options.maxRegisterCount = GetParam();
    ASSERT_OPTIX_SUCCESS( optixModuleCreateFromPTX( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                          m_ptx_str.c_str(), m_ptx_str.size(), logString, &logStringSize, &module ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );

};

INSTANTIATE_TEST_SUITE_P( OptixModuleCompileOptions_RunWith_varyMaxRegisterCount,
                         O7_API_optixModuleCreateWithMaxRegisterCounts,
                         testing::Values( MAX_REGISTER_COUNT_ZERO,
                                          MAX_REGISTER_COUNT_1,
                                          MAX_REGISTER_COUNT_64,
                                          MAX_REGISTER_COUNT_96,
                                          MAX_REGISTER_COUNT_128 ) );


TEST_F( O7_API_optixModuleDestroy, RunWithSuccess )
{
    OptixModule module = {};
    ASSERT_OPTIX_SUCCESS( optixModuleCreateFromPTX( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                                    m_ptx_str.c_str(), m_ptx_str.size(), logString, &logStringSize, &module ) );
    ASSERT_OPTIX_SUCCESS( optixModuleDestroy( module ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixModuleDestroy, DontRunWithNullpointer )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixModuleDestroy( nullptr ) );
}
