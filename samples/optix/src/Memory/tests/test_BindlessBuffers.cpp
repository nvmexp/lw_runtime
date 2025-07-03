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

#include <srcTests.h>

#include <optix_world.h>
#include <optixpp_namespace.h>

#include <iostream>

using namespace optix;
using namespace testing;


class TestBindlessBuffers : public Test
{
  public:
    static Context     m_context;
    static std::string m_ptxPath;

    static void SetUpTestCase()
    {
        m_context = Context::create();
        m_context->setRayTypeCount( 1 );
        m_context->setEntryPointCount( 1 );
    }

    static void TearDownTestCase() { m_context->destroy(); }

    void createProgram( const std::string& programName )
    {
        Program program = m_context->createProgramFromPTXFile( m_ptxPath, programName );
        m_context->setRayGenerationProgram( 0, program );
    }
};

Context     TestBindlessBuffers::m_context;
std::string TestBindlessBuffers::m_ptxPath( ptxPath( "test_Memory", "bindlessBuffer.lw" ) );

// Load from buffer id.

TEST_F( TestBindlessBuffers, CanLoadFromBufferIdVariable )
{
    createProgram( "loadFromBufferId" );
    const int CONSTANT    = 42;
    Buffer    inputBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );
    int*      input       = reinterpret_cast<int*>( inputBuffer->map() );
    input[0]              = CONSTANT;
    inputBuffer->unmap();

    Buffer outputBuffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1 );

    m_context["output"]->set( outputBuffer );
    m_context["bufferId"]->setInt( inputBuffer->getId() );
    m_context->launch( 0, 1 );

    int* output = reinterpret_cast<int*>( outputBuffer->map() );
    EXPECT_THAT( output[0], Eq( CONSTANT ) );
    outputBuffer->unmap();
}

TEST_F( TestBindlessBuffers, CanLoadMultipleTimesFromBufferIdVariable )
{
    createProgram( "loadTwiceFromBufferId" );
    const int CONSTANT    = 42;
    Buffer    inputBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 2 );
    int*      input       = reinterpret_cast<int*>( inputBuffer->map() );
    input[0]              = CONSTANT;
    input[1]              = CONSTANT + 1;
    inputBuffer->unmap();

    Buffer outputBuffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 2 );

    m_context["output"]->set( outputBuffer );
    m_context["bufferId"]->setInt( inputBuffer->getId() );
    m_context->launch( 0, 1 );

    int* output = reinterpret_cast<int*>( outputBuffer->map() );
    EXPECT_THAT( output[0], Eq( CONSTANT ) );
    EXPECT_THAT( output[1], Eq( CONSTANT + 1 ) );
    outputBuffer->unmap();
}

TEST_F( TestBindlessBuffers, CanLoadFromBufferOfBuffers )
{
    createProgram( "loadFromBufferOfBuffers" );

    const int CONSTANT    = 42;
    Buffer    inputBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );
    int*      input       = reinterpret_cast<int*>( inputBuffer->map() );
    input[0]              = CONSTANT;
    inputBuffer->unmap();

    Buffer bufferOfBuffers   = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );
    int*   bufferIdContainer = reinterpret_cast<int*>( bufferOfBuffers->map() );
    bufferIdContainer[0]     = inputBuffer->getId();
    bufferOfBuffers->unmap();

    Buffer outputBuffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 2 );

    m_context["bufferOfBuffers"]->set( bufferOfBuffers );
    m_context["output"]->set( outputBuffer );
    m_context->launch( 0, 1 );

    int* output = reinterpret_cast<int*>( outputBuffer->map() );
    EXPECT_THAT( output[0], Eq( CONSTANT ) );
    outputBuffer->unmap();
}

// Store to buffer id.

TEST_F( TestBindlessBuffers, CanStoreToBufferIdVariable )
{
    createProgram( "storeToBufferId" );
    Buffer outputBuffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1 );

    const int CONSTANT = 42;
    m_context["constantVariable"]->setInt( CONSTANT );
    m_context["output"]->set( outputBuffer );
    m_context["bufferId"]->setInt( outputBuffer->getId() );

    m_context->launch( 0, 1 );

    int* output = reinterpret_cast<int*>( outputBuffer->map() );
    EXPECT_THAT( output[0], Eq( CONSTANT ) );
    outputBuffer->unmap();
}

TEST_F( TestBindlessBuffers, CanStoreMultipleTimesToBufferIdVariable )
{
    createProgram( "storeTwiceToBufferId" );
    Buffer outputBuffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 2 );

    const int CONSTANT = 42;
    m_context["constantVariable"]->setInt( CONSTANT );
    m_context["output"]->set( outputBuffer );
    m_context["bufferId"]->setInt( outputBuffer->getId() );

    m_context->launch( 0, 1 );

    int* output = reinterpret_cast<int*>( outputBuffer->map() );
    EXPECT_THAT( output[0], Eq( CONSTANT ) );
    EXPECT_THAT( output[1], Eq( CONSTANT + 1 ) );
    outputBuffer->unmap();
}

TEST_F( TestBindlessBuffers, CanStoreToBufferIdFromBufferOfBuffers )
{
    createProgram( "storeToBufferIdFromBufferOfBuffers" );
    Buffer outputBuffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1 );

    Buffer bufferOfBuffers   = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );
    int*   bufferIdContainer = reinterpret_cast<int*>( bufferOfBuffers->map() );
    bufferIdContainer[0]     = outputBuffer->getId();
    bufferOfBuffers->unmap();

    const int CONSTANT = 42;
    m_context["constantVariable"]->setInt( CONSTANT );
    m_context["output"]->set( outputBuffer );
    m_context["bufferOfBuffers"]->set( bufferOfBuffers );

    m_context->launch( 0, 1 );

    int* output = reinterpret_cast<int*>( outputBuffer->map() );
    EXPECT_THAT( output[0], Eq( CONSTANT ) );
    outputBuffer->unmap();
}

// Query size of buffer.

TEST_F( TestBindlessBuffers, CanQuerySize )
{
    createProgram( "getSize" );
    Buffer outputBuffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1 );

    const int BUFFER_SIZE = 42;
    Buffer    inputBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, BUFFER_SIZE );

    m_context["output"]->set( outputBuffer );
    m_context["bufferId"]->setInt( inputBuffer->getId() );

    m_context->launch( 0, 1 );

    int*   output          = reinterpret_cast<int*>( outputBuffer->map() );
    RTsize inputBufferSize = 0;
    inputBuffer->getSize( inputBufferSize );
    EXPECT_THAT( output[0], Eq( (int)inputBufferSize ) );
    outputBuffer->unmap();
}
