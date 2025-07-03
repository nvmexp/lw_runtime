// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <optixpp_namespace.h>

#include <gtest/gtest.h>
#include <srcTests.h>

#include <optix_world.h>
#include <optixu/optixu_math.h>
#include <optixu/optixu_matrix_namespace.h>

using namespace optix;
using namespace testing;

class TestPointerArithmetics : public testing::TestWithParam<std::string>
{
  public:
    Context          m_context;
    Buffer           m_outputInt;
    Buffer           m_outputInt3;
    Buffer           m_inputInt;
    Buffer           m_inputInt3;
    Variable         m_bufferIndex1D;
    Variable         m_bufferIndex2D;
    static const int BUFFER_WIDTH;
    static const int X_COORDINATE;
    static const int VALUE;

    void SetUp()
    {
        m_context = Context::create();
        m_context->setRayTypeCount( 1 );
        m_context->setEntryPointCount( 1 );
        m_bufferIndex1D = m_context->declareVariable( "buffer_index_1d" );
        m_outputInt     = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1 );
        m_outputInt3    = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT3, 1 );
        m_context["outputInt"]->set( m_outputInt );
        m_context["outputInt3"]->set( m_outputInt3 );
    }

    void TearDown() { m_context->destroy(); }

    void setup_TestPointerArithmetics( const std::string& programName );
};

void TestPointerArithmetics::setup_TestPointerArithmetics( const std::string& programName )
{
    std::string ptx_path = ptxPath( "test_Pointers", "pointer_arithmetics.lw" );
    Program     rayGen   = m_context->createProgramFromPTXFile( ptx_path, programName );

    m_context->setRayGenerationProgram( 0, rayGen );

    m_inputInt = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, BUFFER_WIDTH );
    m_context["inputInt1D"]->set( m_inputInt );
    m_inputInt3 = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, BUFFER_WIDTH );
    m_context["inputInt31D"]->set( m_inputInt3 );

    m_bufferIndex1D->setInt( X_COORDINATE );

    int* outputPointer = static_cast<int*>( m_outputInt->map() );
    outputPointer[0]   = 0;
    m_outputInt->unmap();
}

const int TestPointerArithmetics::BUFFER_WIDTH = 20;
const int TestPointerArithmetics::X_COORDINATE = 4;
const int TestPointerArithmetics::VALUE        = 42;

// -----------------------------------------------------------------------------
class AccessInt : public TestPointerArithmetics
{
};

TEST_P( AccessInt, Test )
{
    std::string programName = GetParam();
    setup_TestPointerArithmetics( programName );

    int* hostPointer          = static_cast<int*>( m_inputInt->map() );
    hostPointer[X_COORDINATE] = VALUE;
    m_inputInt->unmap();

    m_context->launch( 0, 1 );

    int* outputPointer = static_cast<int*>( m_outputInt->map() );
    EXPECT_EQ( outputPointer[0], VALUE );
    m_outputInt->unmap();
}

INSTANTIATE_TEST_SUITE_P( NormalAccessInt, AccessInt, Values( std::string( "normalAccessInt1D" ) ) );
INSTANTIATE_TEST_SUITE_P( PointerAccessInt, AccessInt, Values( std::string( "pointerArithmeticsInt1D" ) ) );

// -----------------------------------------------------------------------------
class AccessInt3 : public TestPointerArithmetics
{
};

TEST_P( AccessInt3, Test )
{
    std::string programName = GetParam();
    setup_TestPointerArithmetics( programName );

    int3* hostPointer         = static_cast<int3*>( m_inputInt3->map() );
    hostPointer[X_COORDINATE] = make_int3( VALUE, VALUE + 1, VALUE + 2 );
    m_inputInt3->unmap();

    m_context->launch( 0, 1 );

    int* outputPointer = static_cast<int*>( m_outputInt->map() );
    EXPECT_EQ( outputPointer[0], VALUE + 1 );
    m_outputInt->unmap();
}

INSTANTIATE_TEST_SUITE_P( NormalAccessInt3, AccessInt3, Values( std::string( "normalAccessInt31D" ) ) );
INSTANTIATE_TEST_SUITE_P( PointerAccessInt3, AccessInt3, Values( std::string( "pointerArithmeticsInt31D" ) ) );

// -----------------------------------------------------------------------------
class AccessNextInt3 : public TestPointerArithmetics
{
};

TEST_P( AccessNextInt3, Test )
{
    std::string programName = GetParam();
    setup_TestPointerArithmetics( programName );

    int3* hostPointer             = static_cast<int3*>( m_inputInt3->map() );
    hostPointer[X_COORDINATE]     = make_int3( VALUE, VALUE + 1, VALUE + 2 );
    hostPointer[X_COORDINATE + 1] = make_int3( VALUE + 3, VALUE + 4, VALUE + 5 );
    m_inputInt3->unmap();

    m_context->launch( 0, 1 );

    int* outputPointer = static_cast<int*>( m_outputInt->map() );
    EXPECT_EQ( outputPointer[0], VALUE + 4 );
    m_outputInt->unmap();
}

INSTANTIATE_TEST_SUITE_P( AccessNextInt3, AccessNextInt3, Values( std::string( "normalAccessNextInt31D" ) ) );
INSTANTIATE_TEST_SUITE_P( PointerAccessNextInt3,
                          AccessNextInt3,
                          Values( std::string( "pointerArithmeticsAccessNextInt31D" ) ) );

// -----------------------------------------------------------------------------
class AccessPreviousInt3 : public TestPointerArithmetics
{
};

TEST_P( AccessPreviousInt3, Test )
{
    std::string programName = GetParam();
    setup_TestPointerArithmetics( programName );

    int3* hostPointer             = static_cast<int3*>( m_inputInt3->map() );
    hostPointer[X_COORDINATE - 1] = make_int3( VALUE - 3, VALUE - 2, VALUE - 1 );
    hostPointer[X_COORDINATE]     = make_int3( VALUE, VALUE + 1, VALUE + 2 );
    m_inputInt3->unmap();

    m_context->launch( 0, 1 );

    int* outputPointer = static_cast<int*>( m_outputInt->map() );
    EXPECT_EQ( outputPointer[0], VALUE - 1 );
    m_outputInt->unmap();
}

INSTANTIATE_TEST_SUITE_P( AccessPreviousInt3,
                          AccessPreviousInt3,
                          Values( std::string( "normalAccessPreviousInt31D" ) ) );
INSTANTIATE_TEST_SUITE_P( PointerAccessPreviousInt3,
                          AccessPreviousInt3,
                          Values( std::string( "pointerArithmeticsAccessPreviousInt31D" ) ) );

// -----------------------------------------------------------------------------
class AccessWithOverflowingOffsetInt3 : public TestPointerArithmetics
{
};

TEST_P( AccessWithOverflowingOffsetInt3, Test )
{
    std::string programName = GetParam();
    setup_TestPointerArithmetics( programName );

    int3* hostPointer             = static_cast<int3*>( m_inputInt3->map() );
    hostPointer[X_COORDINATE]     = make_int3( VALUE, VALUE + 1, VALUE + 2 );
    hostPointer[X_COORDINATE + 1] = make_int3( VALUE + 3, VALUE + 4, VALUE + 5 );
    hostPointer[X_COORDINATE + 2] = make_int3( VALUE + 6, VALUE + 7, VALUE + 8 );
    m_inputInt3->unmap();

    m_context->launch( 0, 1 );

    int* outputPointer = static_cast<int*>( m_outputInt->map() );
    EXPECT_EQ( outputPointer[0], VALUE + 7 );
    m_outputInt->unmap();
}

INSTANTIATE_TEST_SUITE_P( AccessWithOverflowingOffsetInt3,
                          AccessWithOverflowingOffsetInt3,
                          Values( std::string( "normalAccessOverflowingOffsetInt31D" ) ) );
INSTANTIATE_TEST_SUITE_P( PointerAccessWithOverflowingOffsetInt3,
                          AccessWithOverflowingOffsetInt3,
                          Values( std::string( "pointerArithmeticsAccessOverflowingOffsetInt31D" ) ) );

// -----------------------------------------------------------------------------
class AccessWithNonConstantOffset : public TestPointerArithmetics
{
};

TEST_P( AccessWithNonConstantOffset, Test )
{
    std::string programName = GetParam();
    setup_TestPointerArithmetics( programName );

    Buffer offsetBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );
    m_context["offsetBuffer"]->set( offsetBuffer );
    const int OFFSET = 2;
    int*      ptr    = static_cast<int*>( offsetBuffer->map() );
    ptr[0]           = OFFSET;
    offsetBuffer->unmap();

    int* hostPointer                   = static_cast<int*>( m_inputInt->map() );
    hostPointer[X_COORDINATE + OFFSET] = VALUE;
    m_inputInt->unmap();

    m_context->launch( 0, 1 );

    int  result        = 0;
    int* outputPointer = static_cast<int*>( m_outputInt->map() );
    result             = outputPointer[0];
    m_outputInt->unmap();

    EXPECT_EQ( result, VALUE );
}

INSTANTIATE_TEST_SUITE_P( AccessWithNonConstantOffset,
                          AccessWithNonConstantOffset,
                          Values( std::string( "normalAccessNonConstantOffsetInt1D" ) ) );
INSTANTIATE_TEST_SUITE_P( PointerAccessWithNonConstantOffset,
                          AccessWithNonConstantOffset,
                          Values( std::string( "pointerArithmeticsAccessNonConstantOffsetInt1D" ) ) );

// -----------------------------------------------------------------------------
class AccessWithNegativeNonConstantOffset : public TestPointerArithmetics
{
};

TEST_P( AccessWithNegativeNonConstantOffset, Test )
{
    std::string programName = GetParam();
    setup_TestPointerArithmetics( programName );

    Buffer offsetBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );
    m_context["offsetBuffer"]->set( offsetBuffer );
    const int OFFSET = 2;
    int*      ptr    = static_cast<int*>( offsetBuffer->map() );
    ptr[0]           = OFFSET;
    offsetBuffer->unmap();

    int* hostPointer                   = static_cast<int*>( m_inputInt->map() );
    hostPointer[X_COORDINATE - OFFSET] = VALUE;
    m_inputInt->unmap();

    m_context->launch( 0, 1 );

    int  result        = 0;
    int* outputPointer = static_cast<int*>( m_outputInt->map() );
    result             = outputPointer[0];
    m_outputInt->unmap();

    EXPECT_EQ( result, VALUE );
}

INSTANTIATE_TEST_SUITE_P( AccessWithNegativeNonConstantOffset,
                          AccessWithNegativeNonConstantOffset,
                          Values( std::string( "normalAccessNegativeNonConstantOffsetInt1D" ) ) );
INSTANTIATE_TEST_SUITE_P( PointerAccessWithNegativeNonConstantOffset,
                          AccessWithNegativeNonConstantOffset,
                          Values( std::string( "pointerArithmeticsAccessNegativeNonConstantOffsetInt1D" ) ) );

// -----------------------------------------------------------------------------
TEST( DiffPointers, CanDiffTwoPointers )
{
    Context context = Context::create();
    context->setRayTypeCount( 1 );
    context->setEntryPointCount( 1 );
    const int BUFFER_WIDTH = 20;
    Buffer    inputInt1D   = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, BUFFER_WIDTH );
    Buffer    outputInt    = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1 );
    context["outputInt"]->set( outputInt );
    context["inputInt1D"]->set( inputInt1D );

    std::string ptx_path = ptxPath( "test_Pointers", "pointer_arithmetics.lw" );
    Program     rayGen   = context->createProgramFromPTXFile( ptx_path, "diffPointers" );

    context->setRayGenerationProgram( 0, rayGen );

    context->launch( 0, 1 );

    int* outputPointer = static_cast<int*>( outputInt->map() );
    EXPECT_EQ( outputPointer[0], 2 );
    outputInt->unmap();

    context->destroy();
}

// -----------------------------------------------------------------------------
class TestPointerArithmeticsAfterPHI : public testing::TestWithParam<std::tuple<std::string, int>>
{
  public:
    Context          m_context;
    Buffer           m_outputInt;
    Buffer           m_outputInt3;
    Buffer           m_inputInt;
    Buffer           m_inputInt3;
    Variable         m_bufferIndex1D;
    Variable         m_bufferIndex2D;
    static const int BUFFER_WIDTH;
    static const int X_COORDINATE;
    static const int VALUE;

    void SetUp()
    {
        m_context = Context::create();
        m_context->setRayTypeCount( 1 );
        m_context->setEntryPointCount( 1 );
        m_bufferIndex1D = m_context->declareVariable( "buffer_index_1d" );
        m_outputInt     = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1 );
        m_outputInt3    = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT3, 1 );
        m_context["outputInt"]->set( m_outputInt );
        m_context["outputInt3"]->set( m_outputInt3 );
    }

    void TearDown() { m_context->destroy(); }

    void setup_TestPointerArithmeticsAfterPHI( const std::string& programName );
};

void TestPointerArithmeticsAfterPHI::setup_TestPointerArithmeticsAfterPHI( const std::string& programName )
{
    std::string ptx_path = ptxPath( "test_Pointers", "pointer_arithmetics.lw" );
    Program     rayGen   = m_context->createProgramFromPTXFile( ptx_path, programName );

    m_context->setRayGenerationProgram( 0, rayGen );

    m_inputInt = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, BUFFER_WIDTH );
    m_context["inputInt1D"]->set( m_inputInt );
    m_inputInt3 = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, BUFFER_WIDTH );
    m_context["inputInt31D"]->set( m_inputInt3 );

    m_bufferIndex1D->setInt( X_COORDINATE );

    int* outputPointer = static_cast<int*>( m_outputInt->map() );
    outputPointer[0]   = 0;
    m_outputInt->unmap();
}

const int TestPointerArithmeticsAfterPHI::BUFFER_WIDTH = 20;
const int TestPointerArithmeticsAfterPHI::X_COORDINATE = 4;
const int TestPointerArithmeticsAfterPHI::VALUE        = 42;

TEST_P( TestPointerArithmeticsAfterPHI, TestPhiWithBufferAndBuffer )
{
    auto        param        = GetParam();
    std::string programName  = std::get<0>( param );
    int         controlValue = std::get<1>( param );

    setup_TestPointerArithmeticsAfterPHI( programName );

    Variable controlVariable = m_context->declareVariable( "control" );
    m_context["control"]->setInt( controlValue );

    const int  INT_VALUE  = 35;
    const int3 INT3_VALUE = make_int3( 48, 49, 50 );

    int* hostPointer              = static_cast<int*>( m_inputInt->map() );
    hostPointer[X_COORDINATE + 2] = INT_VALUE;
    m_inputInt->unmap();

    int3* hostPointerInt3         = static_cast<int3*>( m_inputInt3->map() );
    hostPointerInt3[X_COORDINATE] = INT3_VALUE;
    m_inputInt3->unmap();

    m_context->launch( 0, 1 );

    int  result        = 0;
    int* outputPointer = static_cast<int*>( m_outputInt->map() );
    result             = outputPointer[0];
    m_outputInt->unmap();

    if( controlValue < 10 )
    {
        EXPECT_EQ( result, INT_VALUE );
    }
    else
    {
        EXPECT_EQ( result, INT3_VALUE.z );
    }
}

INSTANTIATE_TEST_SUITE_P( TestArithmeticsAfterPhi_BB_If,
                          TestPointerArithmeticsAfterPHI,
                          Values( std::make_tuple( std::string( "pointerArithmeticsAfterPhiBufferBuffer" ), 5 ) ) );
INSTANTIATE_TEST_SUITE_P( TestArithmeticsAfterPhi_BB_Else,
                          TestPointerArithmeticsAfterPHI,
                          Values( std::make_tuple( std::string( "pointerArithmeticsAfterPhiBufferBuffer" ), 15 ) ) );
