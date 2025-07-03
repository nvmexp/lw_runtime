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

class O7_API_optixProgramGroupCreate : public Test
{
  protected:
    void SetUp() override {}
    void SetupDeferred()
    {
        exptest::lwdaInitialize();
        ASSERT_OPTIX_SUCCESS( optixInit() );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( nullptr, nullptr, &m_context ) );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextSetLogCallback( m_context, &OptixRecordingLogger::callback, &m_logger, 2 ) );
        OPTIX_CHECK( optixModuleCreateFromPTX( m_context, &m_module_compile_options, &m_pipeline_compile_options,
                                               m_ptx_str.c_str(), m_ptx_str.size(), nullptr, nullptr, &m_module ) );
    }
    void TearDown() override { ASSERT_OPTIX_SUCCESS( optixDeviceContextDestroy( m_context ) ); }

    OptixDeviceContext   m_context = nullptr;
    OptixRecordingLogger m_logger;
    char                 logString[1024];
    size_t               logStringSize = 1024;

    OptixPipelineCompileOptions m_pipeline_compile_options = {};
    OptixPipelineLinkOptions    m_pipeline_link_options    = {};
    OptixModuleCompileOptions   m_module_compile_options   = {};
    OptixProgramGroup           m_prog_group               = nullptr;
    OptixPipeline               m_pipeline                 = nullptr;
    OptixModule                 m_module                   = nullptr;

    // Aiming at a minimal ptx file.
    std::string m_ptx_str =
        ".version 6.4\n.target sm_30\n.address_size 64\n"
        ".visible .entry __raygen__trace()\n{ret;}\n"
        ".visible .entry __miss__ms()\n{ret;}\n"
        ".visible .entry __exception__ex()\n{ret;}\n"
        ".visible .entry __closesthit__ch()\n{ret;}\n"
        ".visible .entry __anyhit__ah()\n{ret;}\n"
        ".visible .entry __intersection__is()\n{ret;}\n"
        ".visible .entry __direct_callable__dc()\n{ret;}\n"
        ".visible .entry __continuation_callable__cc()\n{ret;}\n";
};


// The sole reason for its existence is that the optixProgramGroupDestroy tests are properly named but
// can use the functionality to create a program group.
class O7_API_optixProgramGroupDestroy : public O7_API_optixProgramGroupCreate
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

        OptixProgramGroupOptions program_group_options  = {};
        OptixProgramGroupDesc    raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = m_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__trace";
        ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &raygen_prog_group_desc, 1, &program_group_options,
                                                       nullptr, nullptr, &m_prog_group ) );
    }
};


// The sole reason for its existence is that the optixProgramGroupGetStackSize tests are properly named but
// can use the functionality to create a program group.
class O7_API_optixProgramGroupGetStackSize : public O7_API_optixProgramGroupCreate
{
};


//---------------------------------------------------------------------------
// 	optixProgramGroupCreate
//---------------------------------------------------------------------------

TEST_F( O7_API_optixProgramGroupCreate, RunWithSuccess_with_raygen )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options  = {};
    OptixProgramGroupDesc    raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = m_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__trace";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &raygen_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixProgramGroupCreate, RunWithSuccess_Empty_ProgramGroup )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options  = {};
    OptixProgramGroupDesc    raygen_prog_group_desc = {};
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &raygen_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_No_RaygenEntryFunctionName )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options  = {};
    OptixProgramGroupDesc    raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = m_module;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &raygen_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "raygen.entryFunctionName is null" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "raygen.entryFunctionName is null" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_RaygenEntryFunctionName_Doesnt_Start_With_Prefix )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options  = {};
    OptixProgramGroupDesc    raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.entryFunctionName = "trace";
    raygen_prog_group_desc.raygen.module            = m_module;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &raygen_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "raygen.entryFunctionName does not start with" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "raygen.entryFunctionName does not start "
                                                                       "with" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_RaygenEntryFunctionName_Not_In_Module )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options  = {};
    OptixProgramGroupDesc    raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__traceFOO";
    raygen_prog_group_desc.raygen.module            = m_module;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &raygen_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( raygen_prog_group_desc.raygen.entryFunctionName ) );
    EXPECT_THAT( logString, HasSubstr( "not found in" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( raygen_prog_group_desc.raygen.entryFunctionName ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "not found in" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, RunWithSuccess_with_miss )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc    miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                      = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module               = m_module;
    miss_prog_group_desc.miss.entryFunctionName    = "__miss__ms";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &miss_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_No_MissEntryFunctionName )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc    miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                      = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module               = m_module;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &miss_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "miss.entryFunctionName is null" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "miss.entryFunctionName is null" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_MissEntryFunctionName_Doesnt_Start_With_Prefix )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc    miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                      = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.entryFunctionName    = "ms";
    miss_prog_group_desc.miss.module               = m_module;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &miss_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "miss.entryFunctionName does not start with" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "miss.entryFunctionName does not start "
                                                                       "with" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_MissEntryFunctionName_Not_In_Module )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc    miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                      = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.entryFunctionName    = "__miss__msFOO";
    miss_prog_group_desc.miss.module               = m_module;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &miss_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( miss_prog_group_desc.miss.entryFunctionName ) );
    EXPECT_THAT( logString, HasSubstr( "not found in" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( miss_prog_group_desc.miss.entryFunctionName ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "not found in" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, RunWithSuccess_with_exception )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options     = {};
    OptixProgramGroupDesc    except_prog_group_desc    = {};
    except_prog_group_desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    except_prog_group_desc.exception.module            = m_module;
    except_prog_group_desc.exception.entryFunctionName = "__exception__ex";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &except_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_No_ExceptionEntryFunctionName )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options  = {};
    OptixProgramGroupDesc    except_prog_group_desc = {};
    except_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    except_prog_group_desc.exception.module         = m_module;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &except_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "exception.entryFunctionName is null" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "exception.entryFunctionName is null" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_ExceptionEntryFunctionName_Doesnt_Start_With_Prefix )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options     = {};
    OptixProgramGroupDesc    except_prog_group_desc    = {};
    except_prog_group_desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    except_prog_group_desc.exception.entryFunctionName = "ex";
    except_prog_group_desc.exception.module            = m_module;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &except_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "exception.entryFunctionName does not start with" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "exception.entryFunctionName does not start "
                                                                       "with" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_ExceptionEntryFunctionName_Not_In_Module )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options     = {};
    OptixProgramGroupDesc    except_prog_group_desc    = {};
    except_prog_group_desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    except_prog_group_desc.exception.entryFunctionName = "__exception__exFOO";
    except_prog_group_desc.exception.module            = m_module;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &except_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( except_prog_group_desc.exception.entryFunctionName ) );
    EXPECT_THAT( logString, HasSubstr( "not found in" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( except_prog_group_desc.exception.entryFunctionName ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "not found in" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, RunWithSuccess_with_HitgroupCH )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options   = {};
    OptixProgramGroupDesc    hit_prog_group_desc     = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH            = m_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_No_HitgroupCHEntryFunctionName )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc    hit_prog_group_desc   = {};
    hit_prog_group_desc.kind                       = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH          = m_module;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "hitgroup.entryFunctionNameCH is null" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "hitgroup.entryFunctionNameCH is null" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_HitgroupCHEntryFunctionName_Doesnt_Start_With_Prefix )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options   = {};
    OptixProgramGroupDesc    hit_prog_group_desc     = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH            = m_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "ch";
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "hitgroup.entryFunctionNameCH does not start with" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "hitgroup.entryFunctionNameCH does not start "
                                                                       "with" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_HitgroupCHEntryFunctionName_Not_In_Module )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options   = {};
    OptixProgramGroupDesc    hit_prog_group_desc     = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH            = m_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__chFOO";
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( hit_prog_group_desc.hitgroup.entryFunctionNameCH ) );
    EXPECT_THAT( logString, HasSubstr( "not found in" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( hit_prog_group_desc.hitgroup.entryFunctionNameCH ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "not found in" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, RunWithSuccess_with_HitgroupAH )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options   = {};
    OptixProgramGroupDesc    hit_prog_group_desc     = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleAH            = m_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_No_HitgroupAHEntryFunctionName )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc    hit_prog_group_desc   = {};
    hit_prog_group_desc.kind                       = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleAH          = m_module;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "hitgroup.entryFunctionNameAH is null" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "hitgroup.entryFunctionNameAH is null" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_HitgroupAHEntryFunctionName_Doesnt_Start_With_Prefix )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options   = {};
    OptixProgramGroupDesc    hit_prog_group_desc     = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleAH            = m_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameAH = "ah";
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "hitgroup.entryFunctionNameAH does not start with" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "hitgroup.entryFunctionNameAH does not start "
                                                                       "with" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_HitgroupAHEntryFunctionName_Not_In_Module )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options   = {};
    OptixProgramGroupDesc    hit_prog_group_desc     = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleAH            = m_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ahFOO";
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( hit_prog_group_desc.hitgroup.entryFunctionNameAH ) );
    EXPECT_THAT( logString, HasSubstr( "not found in" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( hit_prog_group_desc.hitgroup.entryFunctionNameAH ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "not found in" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, RunWithSuccess_with_HitgroupIS )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options   = {};
    OptixProgramGroupDesc    hit_prog_group_desc     = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleIS            = m_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_No_HitgroupISEntryFunctionName )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc    hit_prog_group_desc   = {};
    hit_prog_group_desc.kind                       = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleIS          = m_module;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "hitgroup.entryFunctionNameIS is null" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "hitgroup.entryFunctionNameIS is null" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_HitgroupISEntryFunctionName_Doesnt_Start_With_Prefix )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options   = {};
    OptixProgramGroupDesc    hit_prog_group_desc     = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleIS            = m_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = "is";
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "hitgroup.entryFunctionNameIS does not start with" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "hitgroup.entryFunctionNameIS does not start "
                                                                       "with" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_HitgroupISEntryFunctionName_Not_In_Module )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options   = {};
    OptixProgramGroupDesc    hit_prog_group_desc     = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleIS            = m_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__isFOO";
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( hit_prog_group_desc.hitgroup.entryFunctionNameIS ) );
    EXPECT_THAT( logString, HasSubstr( "not found in" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( hit_prog_group_desc.hitgroup.entryFunctionNameIS ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "not found in" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, RunWithSuccess_with_CallablesDC )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options         = {};
    OptixProgramGroupDesc    callable_prog_group_desc      = {};
    callable_prog_group_desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callable_prog_group_desc.callables.moduleDC            = m_module;
    callable_prog_group_desc.callables.entryFunctionNameDC = "__direct_callable__dc";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &callable_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_No_CallablesDCEntryFunctionName )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options    = {};
    OptixProgramGroupDesc    callable_prog_group_desc = {};
    callable_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callable_prog_group_desc.callables.moduleDC       = m_module;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &callable_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "callables.entryFunctionNameDC is null" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "callables.entryFunctionNameDC is null" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_CallablesDCEntryFunctionName_Doesnt_Start_With_Prefix )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options         = {};
    OptixProgramGroupDesc    callable_prog_group_desc      = {};
    callable_prog_group_desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callable_prog_group_desc.callables.moduleDC            = m_module;
    callable_prog_group_desc.callables.entryFunctionNameDC = "dc";
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &callable_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "callables.entryFunctionNameDC does not start with" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "callables.entryFunctionNameDC does not start "
                                                                       "with" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_CallablesDCEntryFunctionName_Not_In_Module )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options         = {};
    OptixProgramGroupDesc    callable_prog_group_desc      = {};
    callable_prog_group_desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callable_prog_group_desc.callables.moduleDC            = m_module;
    callable_prog_group_desc.callables.entryFunctionNameDC = "__direct_callable__dcFOO";
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &callable_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( callable_prog_group_desc.callables.entryFunctionNameDC ) );
    EXPECT_THAT( logString, HasSubstr( "not found in" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( callable_prog_group_desc.callables.entryFunctionNameDC ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "not found in" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, RunWithSuccess_with_CallablesCC )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options         = {};
    OptixProgramGroupDesc    callable_prog_group_desc      = {};
    callable_prog_group_desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callable_prog_group_desc.callables.moduleCC            = m_module;
    callable_prog_group_desc.callables.entryFunctionNameCC = "__continuation_callable__cc";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &callable_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, Not( HasSubstr( "COMPILE ERROR:" ) ) );
    ASSERT_EQ( size_t( 0 ), m_logger.getMessagesAsOneString().size() );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_No_CallablesCCEntryFunctionName )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options    = {};
    OptixProgramGroupDesc    callable_prog_group_desc = {};
    callable_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callable_prog_group_desc.callables.moduleCC       = m_module;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &callable_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "callables.entryFunctionNameCC is null" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "callables.entryFunctionNameCC is null" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_CallablesCCEntryFunctionName_Doesnt_Start_With_Prefix )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options         = {};
    OptixProgramGroupDesc    callable_prog_group_desc      = {};
    callable_prog_group_desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callable_prog_group_desc.callables.moduleCC            = m_module;
    callable_prog_group_desc.callables.entryFunctionNameCC = "cc";
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &callable_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( "callables.entryFunctionNameCC does not start with" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "callables.entryFunctionNameCC does not start "
                                                                       "with" ) );
}


TEST_F( O7_API_optixProgramGroupCreate, ProgramDescription_DontRunWith_CallablesCCEntryFunctionName_Not_In_Module )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options         = {};
    OptixProgramGroupDesc    callable_prog_group_desc      = {};
    callable_prog_group_desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callable_prog_group_desc.callables.moduleCC            = m_module;
    callable_prog_group_desc.callables.entryFunctionNameCC = "__continuation_callable__ccFOO";
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixProgramGroupCreate( m_context, &callable_prog_group_desc, 1, &program_group_options,
                                                                   logString, &logStringSize, &m_prog_group ) );
    EXPECT_THAT( logString, HasSubstr( callable_prog_group_desc.callables.entryFunctionNameCC ) );
    EXPECT_THAT( logString, HasSubstr( "not found in" ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( callable_prog_group_desc.callables.entryFunctionNameCC ) );
    EXPECT_THAT( m_logger.getMessagesAsOneString().c_str(), HasSubstr( "not found in" ) );
}


//---------------------------------------------------------------------------
// 	optixProgramGroupDestroy
//---------------------------------------------------------------------------

TEST_F( O7_API_optixProgramGroupDestroy, RunWithSuccess )
{
    ASSERT_OPTIX_SUCCESS( optixProgramGroupDestroy( m_prog_group ) );
}


//---------------------------------------------------------------------------
// 	optixProgramGroupGetStackSize
//---------------------------------------------------------------------------

TEST_F( O7_API_optixProgramGroupGetStackSize, RunWithSuccess_raygen )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options  = {};
    OptixProgramGroupDesc    raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = m_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__trace";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &raygen_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );

    OptixStackSizes stackSizes = {};
    ASSERT_OPTIX_SUCCESS( optixProgramGroupGetStackSize( m_prog_group, &stackSizes ) );
}


TEST_F( O7_API_optixProgramGroupGetStackSize, RunWithSuccess_miss )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc    miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                      = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module               = m_module;
    miss_prog_group_desc.miss.entryFunctionName    = "__miss__ms";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &miss_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );

    OptixStackSizes stackSizes = {};
    ASSERT_OPTIX_SUCCESS( optixProgramGroupGetStackSize( m_prog_group, &stackSizes ) );
}

// TODO
// continue similar for other program groups besides raygen and miss
TEST_F( O7_API_optixProgramGroupGetStackSize, RunWithSuccess_exception )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc    ex_prog_group_desc    = {};
    ex_prog_group_desc.kind                        = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    ex_prog_group_desc.exception.module            = m_module;
    ex_prog_group_desc.exception.entryFunctionName = "__exception__ex";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &ex_prog_group_desc, 1, &program_group_options, logString,
                                                   &logStringSize, &m_prog_group ) );

    OptixStackSizes stackSizes = {};
    ASSERT_OPTIX_SUCCESS( optixProgramGroupGetStackSize( m_prog_group, &stackSizes ) );
}


TEST_F( O7_API_optixProgramGroupGetStackSize, RunWithSuccess_hitgroup_ch )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options   = {};
    OptixProgramGroupDesc    hit_prog_group_desc     = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH            = m_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );

    OptixStackSizes stackSizes = {};
    ASSERT_OPTIX_SUCCESS( optixProgramGroupGetStackSize( m_prog_group, &stackSizes ) );
}


TEST_F( O7_API_optixProgramGroupGetStackSize, RunWithSuccess_hitgroup_ah )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options   = {};
    OptixProgramGroupDesc    hit_prog_group_desc     = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleAH            = m_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );

    OptixStackSizes stackSizes = {};
    ASSERT_OPTIX_SUCCESS( optixProgramGroupGetStackSize( m_prog_group, &stackSizes ) );
}


TEST_F( O7_API_optixProgramGroupGetStackSize, RunWithSuccess_hitgroup_is )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options   = {};
    OptixProgramGroupDesc    hit_prog_group_desc     = {};
    hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleIS            = m_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &hit_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );

    OptixStackSizes stackSizes = {};
    ASSERT_OPTIX_SUCCESS( optixProgramGroupGetStackSize( m_prog_group, &stackSizes ) );
}


TEST_F( O7_API_optixProgramGroupGetStackSize, RunWithSuccess_callables_dc )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options         = {};
    OptixProgramGroupDesc    callable_prog_group_desc      = {};
    callable_prog_group_desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callable_prog_group_desc.callables.moduleDC            = m_module;
    callable_prog_group_desc.callables.entryFunctionNameDC = "__direct_callable__dc";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &callable_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );

    OptixStackSizes stackSizes = {};
    ASSERT_OPTIX_SUCCESS( optixProgramGroupGetStackSize( m_prog_group, &stackSizes ) );
}


TEST_F( O7_API_optixProgramGroupGetStackSize, RunWithSuccess_callables_cc )
{
    SetupDeferred();
    OptixProgramGroupOptions program_group_options         = {};
    OptixProgramGroupDesc    callable_prog_group_desc      = {};
    callable_prog_group_desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callable_prog_group_desc.callables.moduleCC            = m_module;
    callable_prog_group_desc.callables.entryFunctionNameCC = "__continuation_callable__cc";
    ASSERT_OPTIX_SUCCESS( optixProgramGroupCreate( m_context, &callable_prog_group_desc, 1, &program_group_options,
                                                   logString, &logStringSize, &m_prog_group ) );

    OptixStackSizes stackSizes = {};
    ASSERT_OPTIX_SUCCESS( optixProgramGroupGetStackSize( m_prog_group, &stackSizes ) );
}
