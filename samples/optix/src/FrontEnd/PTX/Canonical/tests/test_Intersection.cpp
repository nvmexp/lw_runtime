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

class TestIntersection : public Test
{
  public:
    Context     m_context;
    std::string ptx_path = ptxPath( "test_C14n", "intersection.lw" );

    void SetUp()
    {
        m_context = Context::create();
        m_context->setRayTypeCount( 1 );
        m_context->setEntryPointCount( 1 );
    }

    void TearDown() { m_context->destroy(); }
};

// For all these test cases PI stands for rtPotentialIntersection, RI stands for rtReportInsersection.
// -----------------------------------------------------------------------------
TEST_F( TestIntersection, CannotCallPIWithoutRI )
{
    ASSERT_ANY_THROW( m_context->createProgramFromPTXFile( ptx_path, "callOnlyPI" ) );
}

TEST_F( TestIntersection, CannotCallRIWithoutPI )
{
    ASSERT_ANY_THROW( m_context->createProgramFromPTXFile( ptx_path, "callOnlyRI" ) );
}

TEST_F( TestIntersection, CannotCallRIUnconditionally )
{
    ASSERT_ANY_THROW( m_context->createProgramFromPTXFile( ptx_path, "alwaysCallRI" ) );
}

TEST_F( TestIntersection, CannotAvoidRIIfPITrue )
{
    ASSERT_ANY_THROW( m_context->createProgramFromPTXFile( ptx_path, "avoidRIIfPITrue" ) );
}

TEST_F( TestIntersection, CannotDeferRIWithFlag )
{
    ASSERT_ANY_THROW( m_context->createProgramFromPTXFile( ptx_path, "controlRIWithFlag" ) );
}

TEST_F( TestIntersection, CannotHaveMultipleRIControlledBySinglePI )
{
    ASSERT_ANY_THROW( m_context->createProgramFromPTXFile( ptx_path, "twoRIsWithSinglePI" ) );
}

TEST_F( TestIntersection, CannotWriteAttributeBeforePI )
{
    ASSERT_ANY_THROW( m_context->createProgramFromPTXFile( ptx_path, "writeAttributeBeforePI" ) );
}

TEST_F( TestIntersection, CannotWriteAttributeAfterRI )
{
    ASSERT_ANY_THROW( m_context->createProgramFromPTXFile( ptx_path, "writeAttributeAfterRI" ) );
}

// Except for endlessLoopAroundRI which is clearly invalid, there is not really a "correct" result
// for the other tests. One could easily declare all programs with endless loops as invalid. The
// tests here mainly exist to check that the endless loops do not cause a crash in
// ControlDependenceGraph.
TEST_F( TestIntersection, EndlessLoopAroundPI )
{
    ASSERT_ANY_THROW( m_context->createProgramFromPTXFile( ptx_path, "endlessLoopAroundPI" ) );
}

TEST_F( TestIntersection, EndlessLoopAfterPI )
{
    // Only check for hard crashes and unhandled exceptions; OptiX may either
    // throw an exception or accept the program
    try
    {
        m_context->createProgramFromPTXFile( ptx_path, "endlessLoopAfterPI" );
    }
    catch( optix::Exception& )
    {
    }
}

TEST_F( TestIntersection, EndlessLoopAroundRI )
{
    // Only check for hard crashes and unhandled exceptions; OptiX may either
    // throw an exception or accept the program
    try
    {
        m_context->createProgramFromPTXFile( ptx_path, "endlessLoopAroundRI" );
    }
    catch( optix::Exception& )
    {
    }
}

TEST_F( TestIntersection, EndlessLoopAfterRI )
{
    // Only check for hard crashes and unhandled exceptions; OptiX may either
    // throw an exception or accept the program
    try
    {
        m_context->createProgramFromPTXFile( ptx_path, "endlessLoopAfterRI" );
    }
    catch( optix::Exception& )
    {
    }
}
