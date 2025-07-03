
// Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
//
// LWPU Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software, related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from LWPU Corporation is strictly
// prohibited.
//
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
// AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
// INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
// SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
// LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
// BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES

#include <gtest/gtest.h>
#include <srcTests.h>

#include <optix_world.h>
#include <optixpp_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include <prodlib/system/Knobs.h>

#include <string>

using namespace optix;

const int LARGE_INDEX    = 10000;
const int TRACE_INIT     = 0;
const int TRACE_RG_START = 1;
const int TRACE_RG_END   = 2;
const int TRACE_EX_START = 3;
const int TRACE_EX_END   = 4;

// The integer template parameter is used by tests for BufferIds and TextureIds.
// It represents the id to be checked.
class TestRuntimeExceptions : public testing::TestWithParam<int>
{
  public:
    Context m_context;

    Buffer m_input;
    Buffer m_output;
    Buffer m_code;
    Buffer m_details32;
    Buffer m_details64;

    TestRuntimeExceptions() {}

    void SetUp() {}

    void TearDown() { m_context->destroy(); }

    // Use our own setup method, which needs to be ilwoked explicitly, such that there is a change to set knobs before.
    void SetUpDeferred( const char* ptxFile,
                        const char* rgName,
                        const char* exName,
                        const char* msName     = nullptr,
                        const char* boundsName = nullptr,
                        const char* isName     = nullptr,
                        const char* chName     = nullptr,
                        const char* ahName     = nullptr );

  private:
    TextureSampler m_bindlessTexSampler;
    TextureSampler m_texToDestroy;
};

void TestRuntimeExceptions::SetUpDeferred( const char* lwdaFile,
                                           const char* rgName,
                                           const char* exName,
                                           const char* msName,
                                           const char* boundsName,
                                           const char* isName,
                                           const char* chName,
                                           const char* ahName )
{
    // Delay context creation until this point such that overrides for the
    // exelwtion strategy are already in place.
    m_context = Context::create();

    // Disable all exceptions (each test enables the ones it needs).
    m_context->setExceptionEnabled( RT_EXCEPTION_ALL, false );

    // Enable printing so that rtPrintExceptionDetails has an effect.
    m_context->setPrintEnabled( true );

    m_context->setRayTypeCount( 1 );
    m_context->setEntryPointCount( 1 );

    std::string ptx_path = ptxPath( "test_ExelwtionStrategy", lwdaFile );

    assert( rgName );
    Program rayGen = m_context->createProgramFromPTXFile( ptx_path, rgName );
    m_context->setRayGenerationProgram( 0, rayGen );

    if( exName )
    {
        Program exceptionProg = m_context->createProgramFromPTXFile( ptx_path, exName );
        m_context->setExceptionProgram( 0, exceptionProg );
    }

    if( msName )
    {
        Program msProg = m_context->createProgramFromPTXFile( ptx_path, msName );
        m_context->setMissProgram( 0, msProg );
    }

    assert( !boundsName == !isName && !boundsName == !ahName && !boundsName == !chName );
    const bool useGeometry = boundsName && isName && chName && ahName;
    if( useGeometry )
    {
        Program bounds     = m_context->createProgramFromPTXFile( ptx_path, boundsName );
        Program intersect  = m_context->createProgramFromPTXFile( ptx_path, isName );
        Program closestHit = m_context->createProgramFromPTXFile( ptx_path, chName );
        Program anyHit     = m_context->createProgramFromPTXFile( ptx_path, ahName );

        Geometry geometry = m_context->createGeometry();
        geometry->setBoundingBoxProgram( bounds );
        geometry->setIntersectionProgram( intersect );
        geometry->setPrimitiveCount( 1u );

        Material material = m_context->createMaterial();
        material->setClosestHitProgram( 0, closestHit );
        material->setAnyHitProgram( 0, anyHit );

        GeometryInstance gi    = m_context->createGeometryInstance( geometry, &material, &material + 1 );
        GeometryGroup    group = m_context->createGeometryGroup();
        group->setChildCount( 1 );
        group->setChild( 0, gi );
        group->setAcceleration( m_context->createAcceleration( "NoAccel", "NoAccel" ) );
        m_context["top_object"]->set( group );
    }

    m_input = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 2 );
    m_context["input"]->set( m_input );
    int* input = static_cast<int*>( m_input->map() );
    input[0]   = RT_EXCEPTION_USER;
    input[1]   = RT_EXCEPTION_USER_MAX;
    m_input->unmap();

    m_output = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 2 );
    m_context["output"]->set( m_output );
    int* output = static_cast<int*>( m_output->map() );
    output[0]   = TRACE_INIT;
    output[1]   = -1;
    m_output->unmap();

    m_code = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1 );
    m_context["code"]->set( m_code );
    int* code = static_cast<int*>( m_code->map() );
    code[0]   = -1;
    m_code->unmap();

    m_details32 = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 9 );
    m_context["details32"]->set( m_details32 );
    int* details32 = static_cast<int*>( m_details32->map() );
    for( int i       = 0; i < 9; ++i )
        details32[i] = -1;
    m_details32->unmap();

    m_details64 = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_LONG_LONG, 7 );
    m_context["details64"]->set( m_details64 );
    long long* details64 = static_cast<long long*>( m_details64->map() );
    for( int i       = 0; i < 7; ++i )
        details64[i] = -1;
    m_details64->unmap();

    m_bindlessTexSampler  = m_context->createTextureSampler();
    Buffer textureBacking = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );
    m_bindlessTexSampler->setBuffer( 0, 0, textureBacking );
    m_texToDestroy = m_context->createTextureSampler();
    m_texToDestroy->setBuffer( 0, 0, textureBacking );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestRuntimeExceptions, TestNoException )
{
    SetUpDeferred( "exceptions.lw", "rg", "ex" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_ALL, false );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    EXPECT_EQ( TRACE_RG_END, output[0] );
    m_output->unmap();
}

TEST_F( TestRuntimeExceptions, TestNoException_NoExProgram )
{
    SetUpDeferred( "exceptions.lw", "rg", 0 );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_ALL, false );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    EXPECT_EQ( TRACE_RG_END, output[0] );
    m_output->unmap();
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestRuntimeExceptions, TestUserExceptionRG_NoExProgram )
{
    SetUpDeferred( "exceptions.lw", "rg_userexception_RG_constant_minimal", 0 );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    EXPECT_EQ( TRACE_RG_START, output[0] );  // the default EX program does not set output[0]
    m_output->unmap();
}

TEST_F( TestRuntimeExceptions, TestUserExceptionRG_Constant_Minimal )
{
    SetUpDeferred( "exceptions.lw", "rg_userexception_RG_constant_minimal", "ex" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_USER, code[0] );
    m_output->unmap();
    m_code->unmap();
}

TEST_F( TestRuntimeExceptions, TestUserExceptionRG_Constant_Maximal )
{
    SetUpDeferred( "exceptions.lw", "rg_userexception_RG_constant_maximal", "ex" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_USER_MAX, code[0] );
    m_output->unmap();
    m_code->unmap();
}

TEST_F( TestRuntimeExceptions, TestUserExceptionRG_Constant_TooLow )
{
    try
    {
        SetUpDeferred( "exceptions.lw", "rg_userexception_RG_constant_too_low", "ex" );
        EXPECT_TRUE( !"Expected compile-time exception missing" );
    }
    catch( const Exception& e )
    {
        const std::string& msg = e.getErrorString();
        EXPECT_THAT( msg, testing::HasSubstr(
                              "Compile Error: User exception must have exception code >= RT_EXCEPTION_USER" ) );
    }
}

TEST_F( TestRuntimeExceptions, TestUserExceptionRG_Constant_TooHigh )
{
    try
    {
        SetUpDeferred( "exceptions.lw", "rg_userexception_RG_constant_too_high", "ex" );
        EXPECT_TRUE( !"Expected compile-time exception missing" );
    }
    catch( const Exception& e )
    {
        const std::string& msg = e.getErrorString();
        EXPECT_THAT( msg, testing::HasSubstr(
                              "Compile Error: User exception must have exception code <= RT_EXCEPTION_USER_MAX" ) );
    }
}

TEST_F( TestRuntimeExceptions, TestUserExceptionRG_NonConstant_Minimal )
{
    SetUpDeferred( "exceptions.lw", "rg_userexception_RG_nonconstant_minimal", "ex" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_USER, code[0] );
    m_output->unmap();
    m_code->unmap();
}

TEST_F( TestRuntimeExceptions, TestUserExceptionRG_NonConstant_Maximal )
{
    SetUpDeferred( "exceptions.lw", "rg_userexception_RG_nonconstant_maximal", "ex" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_USER_MAX, code[0] );
    m_output->unmap();
    m_code->unmap();
}

TEST_F( TestRuntimeExceptions, TestUserExceptionRG_NonConstant_TooLow )
{
    SetUpDeferred( "exceptions.lw", "rg_userexception_RG_nonconstant_too_low", "ex" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output    = static_cast<int*>( m_output->map() );
    int* code      = static_cast<int*>( m_code->map() );
    int* details32 = static_cast<int*>( m_details32->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_USER_EXCEPTION_CODE_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( RT_EXCEPTION_USER - 1, details32[0] );
    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
}

TEST_F( TestRuntimeExceptions, TestUserExceptionRG_NonConstant_TooHigh )
{
    SetUpDeferred( "exceptions.lw", "rg_userexception_RG_nonconstant_too_high", "ex" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output    = static_cast<int*>( m_output->map() );
    int* code      = static_cast<int*>( m_code->map() );
    int* details32 = static_cast<int*>( m_details32->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_USER_EXCEPTION_CODE_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( RT_EXCEPTION_USER_MAX + 1, details32[0] );
    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
}

TEST_F( TestRuntimeExceptions, TestUserExceptionIS )
{
    SetUpDeferred( "exceptions.lw", "rg_userexception_IS", "ex", "ms", "bounds", "is_userexception_IS", "ch", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_USER + 1, code[0] );
    m_output->unmap();
    m_code->unmap();
}

TEST_F( TestRuntimeExceptions, TestUserExceptionAH )
{
    SetUpDeferred( "exceptions.lw", "rg_userexception_AH", "ex", "ms", "bounds", "is_always_hit", "ch",
                   "ah_userexception_AH" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_USER + 2, code[0] );
    m_output->unmap();
    m_code->unmap();
}

TEST_F( TestRuntimeExceptions, TestUserExceptionCH )
{
    SetUpDeferred( "exceptions.lw", "rg_userexception_CH", "ex", "ms", "bounds", "is_always_hit", "ch_userexception_CH",
                   "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_USER + 3, code[0] );
    m_output->unmap();
    m_code->unmap();
}

TEST_F( TestRuntimeExceptions, TestUserExceptionMS )
{
    SetUpDeferred( "exceptions.lw", "rg_userexception_MS", "ex", "ms_userexception_MS", "bounds", "is_userexception_MS",
                   "ch", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_USER + 4, code[0] );
    m_output->unmap();
    m_code->unmap();
}

#ifndef DEBUG
#ifndef DEVELOP
// Debug code in rtcore's verifyModule() causes this test to fail. This also happens in release mode if
// ENABLE_DEVELOP_ASSERTS is enabled for rtcore (but the OptiX build system does not propagate this flag as define).
TEST_F( TestRuntimeExceptions, TestUserExceptionEX )
{
    SetUpDeferred( "exceptions.lw", "rg_userexception_EX", "exception_userexception_EX" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    EXPECT_EQ( TRACE_EX_START, output[0] );  // the EX program is not run till its end
    m_output->unmap();
}
#endif
#endif

TEST_F( TestRuntimeExceptions, TestUserExceptionBindlessCP )
{
    SetUpDeferred( "exceptions.lw", "rg_userexception_bindlessCP", "ex" );

    std::string ptx_path = ptxPath( "test_ExelwtionStrategy", "exceptions.lw" );
    Program     cp       = m_context->createProgramFromPTXFile( ptx_path, "bindless_userexception_bindlessCP" );
    m_context["var_bindless_userexception_bindlessCP"]->setProgramId( cp );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_USER + 7, code[0] );
    m_output->unmap();
    m_code->unmap();
}

TEST_F( TestRuntimeExceptions, TestUserExceptionBoundCP )
{
    SetUpDeferred( "exceptions.lw", "rg_userexception_boundCP", "ex" );

    std::string ptx_path = ptxPath( "test_ExelwtionStrategy", "exceptions.lw" );
    Program     cp       = m_context->createProgramFromPTXFile( ptx_path, "bound_userexception_boundCP" );

    Program rg = m_context->getRayGenerationProgram( 0 );

    RTprogram  cp2 = cp->get();
    RTprogram  rg2 = rg->get();
    RTvariable variable;
    rtProgramDeclareVariable( rg2, "var_bound_userexception_boundCP", &variable );
    rtVariableSetObject( variable, cp2 );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_USER + 8, code[0] );
    m_output->unmap();
    m_code->unmap();
}

// TODO Add test for node visit programs (not yet supported by RTX)

////////////////////////////////////////////////////////////////////////////////

// Tests for stack overflow are different for RTX and MK/simpleES due to different approaches to
// trigger the stack overflow.

TEST_F_DEV( TestRuntimeExceptions, TestStackOverflowRG_Immediate_rtx )
{
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );
    ScopedKnobSetter knobCS( "rtx.continuationStackSize", 4096 );

    SetUpDeferred( "exceptions.lw", "rg_stackoverflow_RG_immediate_rtx", "ex", "ms", "bounds", "is", "ch", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_STACK_OVERFLOW, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_STACK_OVERFLOW, code[0] );
    m_output->unmap();
    m_code->unmap();
}

// Disabled because rtcore actually uses a larger continuation stack size.
TEST_F_DEV( TestRuntimeExceptions, DISABLED_TestStackOverflowRG_Attributes_rtx )
{
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );
    ScopedKnobSetter knobCS( "rtx.continuationStackSize", 4096 );

    SetUpDeferred( "exceptions.lw", "rg_stackoverflow_RG_attributes_rtx", "ex", "ms", "bounds",
                   "is_stackoverflow_RG_attributes_rtx", "ch_stackoverflow_RG_attributes_rtx", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_STACK_OVERFLOW, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_STACK_OVERFLOW, code[0] );
    m_output->unmap();
    m_code->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestStackOverflowRG_Huge_rtx )
{
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", "rg_stackoverflow_RG_huge_rtx", "ex", "ms", "bounds", "is", "ch", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_STACK_OVERFLOW, true );

    // Not really necessary, but to make the intention clearer and to prevent triggering other error checks.
    rtContextSetMaxTraceDepth( m_context->get(), 1 );

    try
    {
        m_context->launch( 0, 1 /* width */ );
        EXPECT_TRUE( !"Expected compile-time exception missing" );
    }
    catch( const Exception& e )
    {
        const std::string& msg = e.getErrorString();
        EXPECT_THAT( msg, testing::HasSubstr( "returned (7): Compile error" ) );
    }
}

TEST_F_DEV( TestRuntimeExceptions, TestStackOverflowRG_Huge_and_attributes_rtx )
{
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", "rg_stackoverflow_RG_huge_and_attributes_rtx", "ex", "ms", "bounds",
                   "is_stackoverflow_RG_huge_and_attributes_rtx", "ch_stackoverflow_RG_attributes_rtx", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_STACK_OVERFLOW, true );

    // Not really necessary, but to make the intention clearer and to prevent triggering other error checks.
    rtContextSetMaxTraceDepth( m_context->get(), 1 );

    try
    {
        m_context->launch( 0, 1 /* width */ );
        EXPECT_TRUE( !"Expected compile-time exception missing" );
    }
    catch( const Exception& e )
    {
        const std::string& msg = e.getErrorString();
        EXPECT_THAT( msg, testing::HasSubstr( "returned (8): Link error" ) );
    }
}

TEST_F_DEV( TestRuntimeExceptions, TestStackOverflowCH_Immediate_rtx )
{
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", "rg_stackoverflow_CH_rtx", "ex", "ms", "bounds", "is_always_hit",
                   "ch_stackoverflow_CH_immediate_rtx", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_STACK_OVERFLOW, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_STACK_OVERFLOW, code[0] );
    m_output->unmap();
    m_code->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestStackOverflowCH_Relwrsive_rtx )
{
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", "rg_stackoverflow_CH_rtx", "ex", "ms", "bounds", "is_always_hit",
                   "ch_stackoverflow_CH_relwrsive_rtx", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_STACK_OVERFLOW, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_STACK_OVERFLOW, code[0] );
    m_output->unmap();
    m_code->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestStackOverflowMS_Immediate_rtx )
{
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", "rg_stackoverflow_MS_rtx", "ex", "ms_stackoverflow_MS_immediate_rtx", "bounds",
                   "is", "ch", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_STACK_OVERFLOW, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_STACK_OVERFLOW, code[0] );
    m_output->unmap();
    m_code->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestStackOverflowMS_Relwrsive_rtx )
{
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", "rg_stackoverflow_MS_rtx", "ex", "ms_stackoverflow_MS_relwrsive_rtx", "bounds",
                   "is", "ch", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_STACK_OVERFLOW, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_STACK_OVERFLOW, code[0] );
    m_output->unmap();
    m_code->unmap();
}

////////////////////////////////////////////////////////////////////////////////

TEST_F_DEV( TestRuntimeExceptions, TestStackOverflowRG_Immediate_mk )
{
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "megakernel" ) );

    // Force stack small enough to allocate initial memory but small enough so we
    // can't even call the RG program.
    ScopedKnobSetter knobSS( "megakernel.stackSize", 112 );

    SetUpDeferred( "exceptions.lw", "rg_stackoverflow_RG_mk", "ex", "ms_stackoverflow_RG_mk", "bounds", "is", "ch",
                   "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_STACK_OVERFLOW, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_STACK_OVERFLOW, code[0] );
    m_output->unmap();
    m_code->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestStackOverflowRG_Relwrsive_mk )
{
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "megakernel" ) );

    // Force size of stack to make sure that traversal can start but will
    // overflow after a few relwrsions.
    ScopedKnobSetter knobSS( "megakernel.stackSize", 10000 );

    SetUpDeferred( "exceptions.lw", "rg_stackoverflow_RG_mk", "ex", "ms_stackoverflow_RG_mk", "bounds", "is", "ch",
                   "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_STACK_OVERFLOW, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_STACK_OVERFLOW, code[0] );
    m_output->unmap();
    m_code->unmap();
}

// Regression test: This checks if stack overflow handling still works in presence of
// other exception handling code.
TEST_F_DEV( TestRuntimeExceptions, TestStackOverflowRG_RelwrsiveAllEnabled_mk )
{
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "megakernel" ) );

    // Force size of stack to make sure that traversal can start but will
    // overflow after a few relwrsions.
    ScopedKnobSetter knobSS( "megakernel.stackSize", 10000 );

    SetUpDeferred( "exceptions.lw", "rg_stackoverflow_RG_mk", "ex", "ms_stackoverflow_RG_mk", "bounds", "is", "ch",
                   "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_ALL, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_STACK_OVERFLOW, code[0] );
    m_output->unmap();
    m_code->unmap();
}

////////////////////////////////////////////////////////////////////////////////

TEST_F_DEV( TestRuntimeExceptions, TestTraceDepthExceededRG_Immediate_0_rtx )
{
    // Enforce RTX since this exception is not implemented for MK and there are no plans to fix that.
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );
    ScopedKnobSetter knobRD( "rtx.maxTraceRelwrsionDepth", 0 );

    SetUpDeferred( "exceptions.lw", "rg_tracedepthexceeded_RG_rtx", "ex", "ms", "bounds", "is", "ch", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_TRACE_DEPTH_EXCEEDED, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_TRACE_DEPTH_EXCEEDED, code[0] );
    m_output->unmap();
    m_code->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestTraceDepthExceededRG_Immediate_1_rtx )
{
    // Enforce RTX since this exception is not implemented for MK and there are no plans to fix that.
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );
    ScopedKnobSetter knobRD( "rtx.maxTraceRelwrsionDepth", 1 );

    SetUpDeferred( "exceptions.lw", "rg_tracedepthexceeded_RG_rtx", "ex", "ms", "bounds", "is", "ch", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_TRACE_DEPTH_EXCEEDED, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    EXPECT_EQ( TRACE_RG_END, output[0] );
    m_output->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestTraceDepthExceededCH_Relwrsive_3_rtx )
{
    // Enforce RTX since this exception is not implemented for MK and there are no plans to fix that.
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );
    ScopedKnobSetter knobRD( "rtx.maxTraceRelwrsionDepth", 3 );

    SetUpDeferred( "exceptions.lw", "rg_tracedepthexceeded_CH_rtx", "ex", "ms", "bounds", "is_always_hit",
                   "ch_tracedepthexceeded_CH_4_rtx", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_TRACE_DEPTH_EXCEEDED, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_TRACE_DEPTH_EXCEEDED, code[0] );
    m_output->unmap();
    m_code->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestTraceDepthExceededCH_Relwrsive_4_rtx )
{
    // Enforce RTX since this exception is not implemented for MK and there are no plans to fix that.
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );
    ScopedKnobSetter knobRD( "rtx.maxTraceRelwrsionDepth", 4 );

    SetUpDeferred( "exceptions.lw", "rg_tracedepthexceeded_CH_rtx", "ex", "ms", "bounds", "is_always_hit",
                   "ch_tracedepthexceeded_CH_4_rtx", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_TRACE_DEPTH_EXCEEDED, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    EXPECT_EQ( TRACE_RG_END, output[0] );
    m_output->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestTraceDepthExceededCH_Relwrsive_31_rtx )
{
    // Enforce RTX since this exception is not implemented for MK and there are no plans to fix that.
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );
    // Use default of 31 for "rtx.maxTraceRelwrsionDepth"

    SetUpDeferred( "exceptions.lw", "rg_tracedepthexceeded_CH_rtx", "ex", "ms", "bounds", "is_always_hit",
                   "ch_tracedepthexceeded_CH_32_rtx", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_TRACE_DEPTH_EXCEEDED, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    int* code   = static_cast<int*>( m_code->map() );
    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_TRACE_DEPTH_EXCEEDED, code[0] );
    m_output->unmap();
    m_code->unmap();
}

////////////////////////////////////////////////////////////////////////////////

class TestRuntimeExceptionsBufferId : public TestRuntimeExceptions
{
};

INSTANTIATE_TEST_SUITE_P( TestNullId, TestRuntimeExceptionsBufferId, testing::Values( static_cast<int>( RT_BUFFER_ID_NULL ) ) );
INSTANTIATE_TEST_SUITE_P( TestIlwalidId, TestRuntimeExceptionsBufferId, testing::Values( 13 ) );
INSTANTIATE_TEST_SUITE_P( TestNegativeId, TestRuntimeExceptionsBufferId, testing::Values( -1 ) );

TEST_P( TestRuntimeExceptionsBufferId, TestBufferId )
{
    int ilwalidBufferId = GetParam();

    SetUpDeferred( "exceptions.lw", "rg_ilwalidbufferid", "ex" );

    Buffer idBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_BUFFER_ID, 2 );
    m_context["idBuffer"]->set( idBuffer );

    int* buffers = static_cast<int*>( idBuffer->map() );
    // Set the first element to a valid buffer id.
    buffers[0] = m_output->getId();
    // Set the second element to an invalid buffer id.
    buffers[1] = ilwalidBufferId;
    idBuffer->unmap();

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_BUFFER_ID_ILWALID, true );

    m_context->launch( 0, 1 /* width */ );

    int* output    = static_cast<int*>( m_output->map() );
    int* code      = static_cast<int*>( m_code->map() );
    int* details32 = static_cast<int*>( m_details32->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_BUFFER_ID_ILWALID, code[0] );
    int expected_response = ilwalidBufferId == RT_BUFFER_ID_NULL ? 1 : ilwalidBufferId == 13 ? 2 : 3;
    EXPECT_EQ( expected_response, details32[1] );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
}

////////////////////////////////////////////////////////////////////////////////

class TestRuntimeExceptionsTextureId : public TestRuntimeExceptions
{
};

INSTANTIATE_TEST_SUITE_P( TestNullId, TestRuntimeExceptionsTextureId, testing::Values( static_cast<int>( RT_TEXTURE_ID_NULL ) ) );
INSTANTIATE_TEST_SUITE_P( TestIlwalidId, TestRuntimeExceptionsTextureId, testing::Values( 13 ) );
INSTANTIATE_TEST_SUITE_P( TestNegativeId, TestRuntimeExceptionsTextureId, testing::Values( -1 ) );

TEST_P( TestRuntimeExceptionsTextureId, TestTextureId )
{
    int ilwalidTextureId = GetParam();

    SetUpDeferred( "exceptions.lw", "rg_ilwalidtextureid", "ex" );

    Buffer texIdBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 2 );
    m_context["texIdBuffer"]->set( texIdBuffer );

    int* textures = static_cast<int*>( texIdBuffer->map() );
    // Set the first element to a valid buffer id.
    textures[0] = m_output->getId();
    // Set the second element to an invalid texture id.
    textures[1] = ilwalidTextureId;
    texIdBuffer->unmap();

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_TEXTURE_ID_ILWALID, true );

    m_context->launch( 0, 1 /* width */ );

    int* output    = static_cast<int*>( m_output->map() );
    int* code      = static_cast<int*>( m_code->map() );
    int* details32 = static_cast<int*>( m_details32->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_TEXTURE_ID_ILWALID, code[0] );
    int expected_response = ilwalidTextureId == RT_TEXTURE_ID_NULL ? 1 : ilwalidTextureId == 13 ? 2 : 3;
    EXPECT_EQ( expected_response, details32[1] );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestRuntimeExceptions, TestBufferIndexOutOfBounds_GetBufferElement_1d )
{
    SetUpDeferred( "exceptions.lw", "rg_bufferindexoutofbounds_getBufferElement_1d", "ex" );

    Buffer input1d = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );
    m_context["input1d"]->set( input1d );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    int*       details32 = static_cast<int*>( m_details32->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 1, details32[0] );
    EXPECT_EQ( 4, details32[1] );
    EXPECT_TRUE( /*MK*/ details32[2] == -1 || /*RTX*/ details32[2] == input1d->getId() );
    EXPECT_EQ( 1, details64[1] );
    EXPECT_EQ( 1, details64[2] );
    EXPECT_EQ( 1, details64[3] );
    EXPECT_EQ( LARGE_INDEX, details64[4] );
    EXPECT_EQ( 0, details64[5] );
    EXPECT_EQ( 0, details64[6] );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
    m_details64->unmap();
}

TEST_F( TestRuntimeExceptions, TestBufferIndexOutOfBounds_GetBufferElement_2d )
{
    SetUpDeferred( "exceptions.lw", "rg_bufferindexoutofbounds_getBufferElement_2d", "ex" );

    Buffer input2d = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1, 1 );
    m_context["input2d"]->set( input2d );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    int*       details32 = static_cast<int*>( m_details32->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 2, details32[0] );
    EXPECT_EQ( 4, details32[1] );
    EXPECT_TRUE( /*MK*/ details32[2] == -1 || /*RTX*/ details32[2] == input2d->getId() );
    EXPECT_EQ( 1, details64[1] );
    EXPECT_EQ( 1, details64[2] );
    EXPECT_EQ( 1, details64[3] );
    EXPECT_EQ( 0, details64[4] );
    EXPECT_EQ( LARGE_INDEX, details64[5] );
    EXPECT_EQ( 0, details64[6] );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
    m_details64->unmap();
}

TEST_F( TestRuntimeExceptions, TestBufferIndexOutOfBounds_GetBufferElement_3d )
{
    SetUpDeferred( "exceptions.lw", "rg_bufferindexoutofbounds_getBufferElement_3d", "ex" );

    Buffer input3d = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1, 1, 1 );
    m_context["input3d"]->set( input3d );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    int*       details32 = static_cast<int*>( m_details32->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 3, details32[0] );
    EXPECT_EQ( 4, details32[1] );
    EXPECT_TRUE( /*MK*/ details32[2] == -1 || /*RTX*/ details32[2] == input3d->getId() );
    EXPECT_EQ( 1, details64[1] );
    EXPECT_EQ( 1, details64[2] );
    EXPECT_EQ( 1, details64[3] );
    EXPECT_EQ( 0, details64[4] );
    EXPECT_EQ( 0, details64[5] );
    EXPECT_EQ( LARGE_INDEX, details64[6] );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
    m_details64->unmap();
}

TEST_F( TestRuntimeExceptions, TestBufferIndexOutOfBounds_SetBufferElement_1d )
{
    SetUpDeferred( "exceptions.lw", "rg_bufferindexoutofbounds_setBufferElement_1d", "ex" );

    Buffer output1d = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );
    m_context["output1d"]->set( output1d );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    int*       details32 = static_cast<int*>( m_details32->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 1, details32[0] );
    EXPECT_EQ( 4, details32[1] );
    EXPECT_TRUE( /*MK*/ details32[2] == -1 || /*RTX*/ details32[2] == output1d->getId() );
    EXPECT_EQ( 1, details64[1] );
    EXPECT_EQ( 1, details64[2] );
    EXPECT_EQ( 1, details64[3] );
    EXPECT_EQ( LARGE_INDEX, details64[4] );
    EXPECT_EQ( 0, details64[5] );
    EXPECT_EQ( 0, details64[6] );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
    m_details64->unmap();
}

TEST_F( TestRuntimeExceptions, TestBufferIndexOutOfBounds_SetBufferElement_2d )
{
    SetUpDeferred( "exceptions.lw", "rg_bufferindexoutofbounds_setBufferElement_2d", "ex" );

    Buffer output2d = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1, 1 );
    m_context["output2d"]->set( output2d );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    int*       details32 = static_cast<int*>( m_details32->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 2, details32[0] );
    EXPECT_EQ( 4, details32[1] );
    EXPECT_TRUE( /*MK*/ details32[2] == -1 || /*RTX*/ details32[2] == output2d->getId() );
    EXPECT_EQ( 1, details64[1] );
    EXPECT_EQ( 1, details64[2] );
    EXPECT_EQ( 1, details64[3] );
    EXPECT_EQ( 0, details64[4] );
    EXPECT_EQ( LARGE_INDEX, details64[5] );
    EXPECT_EQ( 0, details64[6] );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
    m_details64->unmap();
}

TEST_F( TestRuntimeExceptions, TestBufferIndexOutOfBounds_SetBufferElement_3d )
{
    SetUpDeferred( "exceptions.lw", "rg_bufferindexoutofbounds_setBufferElement_3d", "ex" );

    Buffer output3d = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1, 1, 1 );
    m_context["output3d"]->set( output3d );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    int*       details32 = static_cast<int*>( m_details32->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 3, details32[0] );
    EXPECT_EQ( 4, details32[1] );
    EXPECT_TRUE( /*MK*/ details32[2] == -1 || /*RTX*/ details32[2] == output3d->getId() );
    EXPECT_EQ( 1, details64[1] );
    EXPECT_EQ( 1, details64[2] );
    EXPECT_EQ( 1, details64[3] );
    EXPECT_EQ( 0, details64[4] );
    EXPECT_EQ( 0, details64[5] );
    EXPECT_EQ( LARGE_INDEX, details64[6] );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
    m_details64->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestBufferIndexOutOfBounds_GetBufferElementFromId_1d_rtx )
{
    // Enforce RTX since this instrinsic is not instrumented for MK and there are no plans to fix that.
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", "rg_bufferindexoutofbounds_getBufferElementFromId_1d", "ex" );

    Buffer inputId1d = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );
    rtBufferId<int, 1> bufferId( inputId1d->getId() );
    m_context["inputId1d"]->setUserData( sizeof( bufferId ), &bufferId );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    int*       details32 = static_cast<int*>( m_details32->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 1, details32[0] );
    EXPECT_EQ( 4, details32[1] );
    EXPECT_EQ( inputId1d->getId(), details32[2] );
    EXPECT_EQ( 1, details64[1] );
    EXPECT_EQ( 1, details64[2] );
    EXPECT_EQ( 1, details64[3] );
    EXPECT_EQ( LARGE_INDEX, details64[4] );
    EXPECT_EQ( 0, details64[5] );
    EXPECT_EQ( 0, details64[6] );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
    m_details64->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestBufferIndexOutOfBounds_GetBufferElementFromId_2d_rtx )
{
    // Enforce RTX since this instrinsic is not instrumented for MK and there are no plans to fix that.
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", "rg_bufferindexoutofbounds_getBufferElementFromId_2d", "ex" );

    Buffer inputId2d = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1, 1 );
    rtBufferId<int, 2> bufferId( inputId2d->getId() );
    m_context["inputId2d"]->setUserData( sizeof( bufferId ), &bufferId );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    int*       details32 = static_cast<int*>( m_details32->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 2, details32[0] );
    EXPECT_EQ( 4, details32[1] );
    EXPECT_EQ( inputId2d->getId(), details32[2] );
    EXPECT_EQ( 1, details64[1] );
    EXPECT_EQ( 1, details64[2] );
    EXPECT_EQ( 1, details64[3] );
    EXPECT_EQ( 0, details64[4] );
    EXPECT_EQ( LARGE_INDEX, details64[5] );
    EXPECT_EQ( 0, details64[6] );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
    m_details64->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestBufferIndexOutOfBounds_GetBufferElementFromId_3d_rtx )
{
    // Enforce RTX since this instrinsic is not instrumented for MK and there are no plans to fix that.
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", "rg_bufferindexoutofbounds_getBufferElementFromId_3d", "ex" );

    Buffer inputId3d = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1, 1, 1 );
    rtBufferId<int, 3> bufferId( inputId3d->getId() );
    m_context["inputId3d"]->setUserData( sizeof( bufferId ), &bufferId );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    int*       details32 = static_cast<int*>( m_details32->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 3, details32[0] );
    EXPECT_EQ( 4, details32[1] );
    EXPECT_EQ( inputId3d->getId(), details32[2] );
    EXPECT_EQ( 1, details64[1] );
    EXPECT_EQ( 1, details64[2] );
    EXPECT_EQ( 1, details64[3] );
    EXPECT_EQ( 0, details64[4] );
    EXPECT_EQ( 0, details64[5] );
    EXPECT_EQ( LARGE_INDEX, details64[6] );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
    m_details64->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestBufferIndexOutOfBounds_SetBufferElementFromId_1d_rtx )
{
    // Enforce RTX since this instrinsic is not instrumented for MK and there are no plans to fix that.
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", "rg_bufferindexoutofbounds_setBufferElementFromId_1d", "ex" );

    Buffer outputId1d = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1 );
    m_context["outputId1d"]->setInt( outputId1d->getId() );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    int*       details32 = static_cast<int*>( m_details32->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 1, details32[0] );
    EXPECT_EQ( 4, details32[1] );
    EXPECT_EQ( outputId1d->getId(), details32[2] );
    EXPECT_EQ( 1, details64[1] );
    EXPECT_EQ( 1, details64[2] );
    EXPECT_EQ( 1, details64[3] );
    EXPECT_EQ( LARGE_INDEX, details64[4] );
    EXPECT_EQ( 0, details64[5] );
    EXPECT_EQ( 0, details64[6] );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
    m_details64->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestBufferIndexOutOfBounds_SetBufferElementFromId_2d_rtx )
{
    // Enforce RTX since this instrinsic is not instrumented for MK and there are no plans to fix that.
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", "rg_bufferindexoutofbounds_setBufferElementFromId_2d", "ex" );

    Buffer outputId2d = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1, 1 );
    m_context["outputId2d"]->setInt( outputId2d->getId() );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    int*       details32 = static_cast<int*>( m_details32->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 2, details32[0] );
    EXPECT_EQ( 4, details32[1] );
    EXPECT_EQ( outputId2d->getId(), details32[2] );
    EXPECT_EQ( 1, details64[1] );
    EXPECT_EQ( 1, details64[2] );
    EXPECT_EQ( 1, details64[3] );
    EXPECT_EQ( 0, details64[4] );
    EXPECT_EQ( LARGE_INDEX, details64[5] );
    EXPECT_EQ( 0, details64[6] );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
    m_details64->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestBufferIndexOutOfBounds_SetBufferElementFromId_3d_rtx )
{
    // Enforce RTX since this instrinsic is not instrumented for MK and there are no plans to fix that.
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", "rg_bufferindexoutofbounds_setBufferElementFromId_3d", "ex" );

    Buffer outputId3d = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1, 1, 1 );
    m_context["outputId3d"]->setInt( outputId3d->getId() );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    int*       details32 = static_cast<int*>( m_details32->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 3, details32[0] );
    EXPECT_EQ( 4, details32[1] );
    EXPECT_EQ( outputId3d->getId(), details32[2] );
    EXPECT_EQ( 1, details64[1] );
    EXPECT_EQ( 1, details64[2] );
    EXPECT_EQ( 1, details64[3] );
    EXPECT_EQ( 0, details64[4] );
    EXPECT_EQ( 0, details64[5] );
    EXPECT_EQ( LARGE_INDEX, details64[6] );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
    m_details64->unmap();
}

////////////////////////////////////////////////////////////////////////////////

#define __int_as_float( x ) ( *reinterpret_cast<float*>( &( x ) ) )
#define EPS 0.001

int lwdart_nan_f = 0x7fffffff;
int lwdart_inf_f = 0x7f800000;
#define LWDART_NAN_F __int_as_float( lwdart_nan_f )
#define LWDART_INF_F __int_as_float( lwdart_inf_f )

TEST_F( TestRuntimeExceptions, TestIlwalidRay )
{
    SetUpDeferred( "exceptions.lw", "rg_ilwalidray", "ex", "ms", "bounds", "is", "ch", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_ILWALID_RAY, true );

    m_context->launch( 0, 1 /* width */ );

    int* output    = static_cast<int*>( m_output->map() );
    int* code      = static_cast<int*>( m_code->map() );
    int* details32 = static_cast<int*>( m_details32->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_ILWALID_RAY, code[0] );
    // EXPECT_EQ( LWDART_NAN_F, __int_as_float( details32[0] ) );
    EXPECT_EQ( 0.0, __int_as_float( details32[1] ) );
    EXPECT_EQ( 0.0, __int_as_float( details32[2] ) );
    EXPECT_EQ( 1.0, __int_as_float( details32[3] ) );
    EXPECT_EQ( 0.0, __int_as_float( details32[4] ) );
    EXPECT_EQ( 0.0, __int_as_float( details32[5] ) );
    EXPECT_EQ( 0, details32[6] );
    EXPECT_NEAR( 0.001, __int_as_float( details32[7] ), EPS );
    EXPECT_EQ( RT_DEFAULT_MAX, __int_as_float( details32[8] ) );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
}

TEST_F( TestRuntimeExceptions, TestIlwalidRay2 )
{
    SetUpDeferred( "exceptions.lw", "rg_ilwalidray2", "ex", "ms", "bounds", "is", "ch", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_ILWALID_RAY, true );
    m_context->launch( 0, 1 /* width */ );

    int* output    = static_cast<int*>( m_output->map() );
    int* code      = static_cast<int*>( m_code->map() );
    int* details32 = static_cast<int*>( m_details32->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_ILWALID_RAY, code[0] );
    EXPECT_EQ( 0.0, __int_as_float( details32[0] ) );
    // EXPECT_EQ( LWDART_NAN_F, __int_as_float( details32[1] ) );
    EXPECT_EQ( 0.0, __int_as_float( details32[2] ) );
    EXPECT_EQ( 1.0, __int_as_float( details32[3] ) );
    EXPECT_EQ( 0.0, __int_as_float( details32[4] ) );
    EXPECT_EQ( 0.0, __int_as_float( details32[5] ) );
    EXPECT_EQ( 0, details32[6] );
    EXPECT_NEAR( 0.001, __int_as_float( details32[7] ), EPS );
    EXPECT_EQ( RT_DEFAULT_MAX, __int_as_float( details32[8] ) );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
}

TEST_F( TestRuntimeExceptions, TestIlwalidRay3 )
{
    SetUpDeferred( "exceptions.lw", "rg_ilwalidray3", "ex", "ms", "bounds", "is", "ch", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_ILWALID_RAY, true );

    m_context->launch( 0, 1 /* width */ );

    int* output    = static_cast<int*>( m_output->map() );
    int* code      = static_cast<int*>( m_code->map() );
    int* details32 = static_cast<int*>( m_details32->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_ILWALID_RAY, code[0] );
    EXPECT_EQ( 0.0, __int_as_float( details32[0] ) );
    EXPECT_EQ( 0.0, __int_as_float( details32[1] ) );
    EXPECT_EQ( 0.0, __int_as_float( details32[2] ) );
    EXPECT_EQ( 1.0, __int_as_float( details32[3] ) );
    // EXPECT_EQ( LWDART_INF_F, __int_as_float( details32[4] ) );
    EXPECT_EQ( 0.0, __int_as_float( details32[5] ) );
    EXPECT_EQ( 0, details32[6] );
    EXPECT_NEAR( 0.001, __int_as_float( details32[7] ), EPS );
    EXPECT_EQ( RT_DEFAULT_MAX, __int_as_float( details32[8] ) );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
}

TEST_F( TestRuntimeExceptions, TestIlwalidRayIlwalidTMax )
{
    SetUpDeferred( "exceptions.lw", "rg_ilwalidray4", "ex", "ms", "bounds", "is", "ch", "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_ILWALID_RAY, true );

    m_context->launch( 0, 1 /* width */ );

    int* output    = static_cast<int*>( m_output->map() );
    int* code      = static_cast<int*>( m_code->map() );
    int* details32 = static_cast<int*>( m_details32->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_ILWALID_RAY, code[0] );
    // EXPECT_EQ( LWDART_NAN_F, __int_as_float( details32[0] ) );
    EXPECT_EQ( 0.0, __int_as_float( details32[1] ) );
    EXPECT_EQ( 0.0, __int_as_float( details32[2] ) );
    EXPECT_EQ( 1.0, __int_as_float( details32[3] ) );
    EXPECT_EQ( 0.0, __int_as_float( details32[4] ) );
    EXPECT_EQ( 0.0, __int_as_float( details32[5] ) );
    EXPECT_EQ( 0, details32[6] );
    EXPECT_NEAR( 0.001, __int_as_float( details32[7] ), EPS );
    // EXPECT_EQ( RT_DEFAULT_MAX, __int_as_float( details32[8] ) );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestRuntimeExceptions, TestIndexOutOfBounds )
{
    SetUpDeferred( "exceptions.lw", "rg_indexoutofbounds", "ex", "ms_indexoutofbounds", "bounds_indexoutofbounds",
                   "is_indexoutofbounds", "ch_indexoutofbounds", "ah_indexoutofbounds" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_INDEX_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_INDEX_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 1, details64[1] );
    EXPECT_EQ( LARGE_INDEX, details64[2] );

    m_output->unmap();
    m_code->unmap();
    m_details64->unmap();
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestRuntimeExceptions, TestIlwalidProgramId )
{
    SetUpDeferred( "exceptions.lw", "rg_ilwalidprogramid", "ex" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_PROGRAM_ID_ILWALID, true );

    // Launch with a valid id.
    std::string ptx_path = ptxPath( "test_ExelwtionStrategy", "exceptions.lw" );
    Program     cp       = m_context->createProgramFromPTXFile( ptx_path, "bindless_ilwalidprogramid" );
    m_context["var_bindless_ilwalidprogramid"]->setProgramId( cp );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    EXPECT_EQ( TRACE_RG_END, output[0] );
    m_output->unmap();

    // Now set an invalid id and launch again.
    m_context["var_bindless_ilwalidprogramid"]->setInt( 13 );

    m_context->launch( 0, 1 /* width */ );

    output         = static_cast<int*>( m_output->map() );
    int* code      = static_cast<int*>( m_code->map() );
    int* details32 = static_cast<int*>( m_details32->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_PROGRAM_ID_ILWALID, code[0] );
    EXPECT_EQ( 1, details32[1] );

    m_output->unmap();
    m_code->unmap();
    m_details32->unmap();
}

////////////////////////////////////////////////////////////////////////////////

class TestRuntimeExceptionsPayloadAccessOutOfBounds : public TestRuntimeExceptions
{
};

INSTANTIATE_TEST_SUITE_P( 6, TestRuntimeExceptionsPayloadAccessOutOfBounds, testing::Values( 6 ) );
INSTANTIATE_TEST_SUITE_P( 7, TestRuntimeExceptionsPayloadAccessOutOfBounds, testing::Values( 7 ) );
INSTANTIATE_TEST_SUITE_P( 8, TestRuntimeExceptionsPayloadAccessOutOfBounds, testing::Values( 8 ) );

TEST_P_DEV( TestRuntimeExceptionsPayloadAccessOutOfBounds, GetPayloadValue_rtx )
{
    int regs = GetParam();

    std::string rg = std::string( "rg_payloadaccessoutofbounds" ) + std::to_string( regs );
    std::string ch = std::string( "ch_payloadaccessoutofbounds" ) + std::to_string( regs ) + "_getPayloadValue";

    // Enforce RTX since this exception is not implemented for MK and there are no plans to fix that.
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", rg.c_str(), "ex", "ms", "bounds", "is_always_hit", ch.c_str(), "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 4 * regs, details64[1] );
    EXPECT_EQ( 4, details64[2] );
    EXPECT_EQ( 4 * regs, details64[3] );

    m_output->unmap();
    m_code->unmap();
    m_details64->unmap();
}

TEST_P_DEV( TestRuntimeExceptionsPayloadAccessOutOfBounds, SetPayloadValue_rtx )
{
    int regs = GetParam();

    std::string rg = std::string( "rg_payloadaccessoutofbounds" ) + std::to_string( regs );
    std::string ch = std::string( "ch_payloadaccessoutofbounds" ) + std::to_string( regs ) + "_setPayloadValue";

    // Enforce RTX since this exception is not implemented for MK and there are no plans to fix that.
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", rg.c_str(), "ex", "ms", "bounds", "is_always_hit", ch.c_str(), "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 4 * regs, details64[1] );
    EXPECT_EQ( 4, details64[2] );
    EXPECT_EQ( 4 * regs, details64[3] );

    m_output->unmap();
    m_code->unmap();
    m_details64->unmap();
}

TEST_P_DEV( TestRuntimeExceptionsPayloadAccessOutOfBounds, GetPayloadValue_globalTrace_rtx )
{
    int regs = GetParam();

    std::string rg = std::string( "rg_payloadaccessoutofbounds" ) + std::to_string( regs );
    std::string ch =
        std::string( "ch_payloadaccessoutofbounds" ) + std::to_string( regs ) + "_getPayloadValue_globalTrace";

    // Enforce RTX since this exception is not implemented for MK and there are no plans to fix that.
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", rg.c_str(), "ex", "ms", "bounds", "is_always_hit", ch.c_str(), "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 4 * regs, details64[1] );
    EXPECT_EQ( 4, details64[2] );
    EXPECT_EQ( 4 * regs, details64[3] );

    m_output->unmap();
    m_code->unmap();
    m_details64->unmap();
}

TEST_P_DEV( TestRuntimeExceptionsPayloadAccessOutOfBounds, SetPayloadValue_globalTrace_rtx )
{
    int regs = GetParam();

    std::string rg = std::string( "rg_payloadaccessoutofbounds" ) + std::to_string( regs );
    std::string ch =
        std::string( "ch_payloadaccessoutofbounds" ) + std::to_string( regs ) + "_setPayloadValue_globalTrace";

    // Enforce RTX since this exception is not implemented for MK and there are no plans to fix that.
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    SetUpDeferred( "exceptions.lw", rg.c_str(), "ex", "ms", "bounds", "is_always_hit", ch.c_str(), "ah" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS, true );

    m_context->launch( 0, 1 /* width */ );

    int*       output    = static_cast<int*>( m_output->map() );
    int*       code      = static_cast<int*>( m_code->map() );
    long long* details64 = static_cast<long long*>( m_details64->map() );

    EXPECT_EQ( TRACE_EX_END, output[0] );
    EXPECT_EQ( RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS, code[0] );
    EXPECT_EQ( 4 * regs, details64[1] );
    EXPECT_EQ( 4, details64[2] );
    EXPECT_EQ( 4 * regs, details64[3] );

    m_output->unmap();
    m_code->unmap();
    m_details64->unmap();
}

////////////////////////////////////////////////////////////////////////////////

TEST_F_DEV( TestRuntimeExceptions, TestForceTrivialExceptionProgram_mk )
{
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "megakernel" ) );

    ScopedKnobSetter knobEX( "context.forceTrivialExceptionProgram", true );

    SetUpDeferred( "exceptions.lw", "rg_userexception_RG_constant_minimal", "ex" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    EXPECT_EQ( TRACE_RG_START, output[0] );  // the trivial EX program does not set output[0]
    m_output->unmap();
}

TEST_F_DEV( TestRuntimeExceptions, TestForceTrivialExceptionProgram_rtx )
{
    ScopedKnobSetter knobES( "context.forceExelwtionStrategy", std::string( "rtx" ) );

    ScopedKnobSetter knobEX( "context.forceTrivialExceptionProgram", true );

    SetUpDeferred( "exceptions.lw", "rg_userexception_RG_constant_minimal", "ex" );

    rtContextSetExceptionEnabled( m_context->get(), RT_EXCEPTION_USER, true );

    m_context->launch( 0, 1 /* width */ );

    int* output = static_cast<int*>( m_output->map() );
    EXPECT_EQ( TRACE_RG_START, output[0] );  // the trivial EX program does not set output[0]
    m_output->unmap();
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestRuntimeExceptions, TestValidation_EXCallsBindlessCP )
{
    try
    {
        SetUpDeferred( "exceptions.lw", "rg_userexception_bindlessCP", "ex_calls_bindlessCP" );
        EXPECT_TRUE( !"Expected compile-time exception missing" );
    }
    catch( const Exception& e )
    {
        const std::string& msg = e.getErrorString();
        EXPECT_THAT( msg,
                     testing::HasSubstr( "Validation error: Call of bindless callable program is not allowed in" ) );
    }
}

TEST_F( TestRuntimeExceptions, TestValidation_EXCallsBoundCP )
{
    try
    {
        SetUpDeferred( "exceptions.lw", "rg_userexception_boundCP", "ex_calls_boundCP" );
        EXPECT_TRUE( !"Expected compile-time exception missing" );
    }
    catch( const Exception& e )
    {
        const std::string& msg = e.getErrorString();
        EXPECT_THAT( msg, testing::HasSubstr( "Validation error: Call of bound callable program is not allowed in" ) );
    }
}
