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

using namespace optix;
using namespace testing;


class TestZeroSizeAllocation : public testing::Test
{
  public:
    Context   m_context;
    Buffer    m_buffer;
    const int ZERO_SIZE;

    TestZeroSizeAllocation()
        : ZERO_SIZE( 0 )
    {
    }

    void SetUp()
    {
        m_context = Context::create();
        m_context->setRayTypeCount( 1 );
        m_context->setEntryPointCount( 1 );

        m_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_BYTE, ZERO_SIZE );
    }

    void TearDown()
    {
        if( m_context )
            m_context->destroy();
    }

    void setupEmptyProgram();
    void setupSetBufferProgram();
    void createBuffer();
    void mapAndUnmapBuffer();
    void launchProgram();
    void attachBufferToTextureSampler();

  private:
    void setupProgram( const char* raygenName );
};

void TestZeroSizeAllocation::setupEmptyProgram()
{
    setupProgram( "emptyProgram" );
}

void TestZeroSizeAllocation::setupSetBufferProgram()
{
    setupProgram( "emptyProgram" );
}

void TestZeroSizeAllocation::setupProgram( const char* raygenName )
{
    std::string ptx_path( ptxPath( "test_Memory", "zeroSize.lw" ) );

    ASSERT_TRUE( raygenName );
    Program program = m_context->createProgramFromPTXFile( ptx_path, raygenName );
    m_context->setRayGenerationProgram( 0, program );
}

void TestZeroSizeAllocation::mapAndUnmapBuffer()
{
    void* mappedPointer = m_buffer->map();
    ASSERT_THAT( mappedPointer, NotNull() );
    m_buffer->unmap();
}

void TestZeroSizeAllocation::launchProgram()
{
    m_context["buffer"]->set( m_buffer );
    m_context->launch( 0, 1 );
}

void TestZeroSizeAllocation::attachBufferToTextureSampler()
{
    TextureSampler textureSampler = m_context->createTextureSampler();
    textureSampler->setArraySize( 1 );
    textureSampler->setMipLevelCount( 1 );
    textureSampler->setBuffer( 0, 0, m_buffer );
}

TEST_F( TestZeroSizeAllocation, ZeroSizeBufferMapsToNonNullPointer )
{
    void* mappedPointer = m_buffer->map();
    EXPECT_THAT( mappedPointer, NotNull() );
    m_buffer->unmap();
}

// The following test cases test that the allocation of buffers of a zero size
// pointer does not cause an exception when launching the program.
// Multiple tests are necessary to cover the different allocation policies.

TEST_F( TestZeroSizeAllocation, LwdaBufferIsAllocatedSuccessfully )
{
    setupSetBufferProgram();
    mapAndUnmapBuffer();
    EXPECT_NO_THROW( launchProgram() );
}

TEST_F( TestZeroSizeAllocation, UnusedBufferIsAllocatedSuccessfully )
{
    setupEmptyProgram();
    mapAndUnmapBuffer();
    EXPECT_NO_THROW( launchProgram() );
}

TEST_F( TestZeroSizeAllocation, TextureBackingBufferIsAllocatedSuccessfully )
{
    setupEmptyProgram();
    attachBufferToTextureSampler();
    mapAndUnmapBuffer();
    EXPECT_NO_THROW( launchProgram() );
}
