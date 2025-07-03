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


class TestBindlessTextures : public Test
{
  public:
    static Context     m_context;
    static std::string m_ptxPath;

    Buffer m_inputBufferInt1D;
    Buffer m_inputBufferInt2D;
    Buffer m_inputBufferInt3D;
    Buffer m_outputBufferInt;
    Buffer m_inputBufferFloat1D;
    Buffer m_inputBufferFloat2D;
    Buffer m_inputBufferFloat3D;
    Buffer m_outputBufferFloat;

    Variable m_texIdVariable;
    Variable m_xVariable;
    Variable m_yVariable;
    Variable m_zVariable;

    TextureSampler   m_boundTexSampler;
    TextureSampler   m_bindlessTexSampler;
    int              texId = 0;
    static const int CONSTANT;
    static const int BUFFER_WIDTH  = 1;
    static const int BUFFER_HEIGHT = 1;
    static const int BUFFER_DEPTH  = 1;
    static const int X             = 0;
    static const int Y             = 0;
    static const int Z             = 0;

    static void SetUpTestCase()
    {
        m_context = Context::create();
        m_context->setRayTypeCount( 1 );
        m_context->setEntryPointCount( 1 );
    }

    static void TearDownTestCase() { m_context->destroy(); }

    void createBuffers( bool applyDiscardFlag )
    {
        unsigned int inputBufferType = RT_BUFFER_INPUT;
        if( applyDiscardFlag )
        {
            inputBufferType |= RT_BUFFER_DISCARD_HOST_MEMORY;
        }
        m_inputBufferInt1D = m_context->createBuffer( inputBufferType, RT_FORMAT_INT, BUFFER_WIDTH );
        m_inputBufferInt2D = m_context->createBuffer( inputBufferType, RT_FORMAT_INT, BUFFER_WIDTH, BUFFER_HEIGHT );
        m_inputBufferInt3D = m_context->createBuffer( inputBufferType, RT_FORMAT_INT, BUFFER_WIDTH, BUFFER_HEIGHT, BUFFER_DEPTH );
        m_outputBufferInt = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1 );

        m_inputBufferFloat1D = m_context->createBuffer( inputBufferType, RT_FORMAT_FLOAT, BUFFER_WIDTH );
        m_inputBufferFloat2D = m_context->createBuffer( inputBufferType, RT_FORMAT_FLOAT, BUFFER_WIDTH, BUFFER_HEIGHT );
        m_inputBufferFloat3D =
            m_context->createBuffer( inputBufferType, RT_FORMAT_FLOAT, BUFFER_WIDTH, BUFFER_HEIGHT, BUFFER_DEPTH );
        m_outputBufferFloat = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, 1 );

        m_boundTexSampler = m_context->createTextureSampler();
        m_boundTexSampler->setBuffer( 0, 0, m_inputBufferInt1D );

        m_bindlessTexSampler = m_context->createTextureSampler();
        texId                = m_bindlessTexSampler->getId();

        m_context["outputInt"]->set( m_outputBufferInt );
        m_context["outputFloat"]->set( m_outputBufferFloat );

        m_context["boundTexSampler"]->set( m_boundTexSampler );

        m_context["texId"]->setInt( texId );
        m_context["x"]->setInt( X );
        m_context["y"]->setInt( Y );
        m_context["z"]->setInt( Z );
    }

    template <typename T>
    void createProgram( const std::string& programName, Buffer& buffer, bool declareInputBuffersAsDiscard )
    {
        createBuffers( declareInputBuffersAsDiscard );
        Program program = m_context->createProgramFromPTXFile( m_ptxPath, programName );
        m_context->setRayGenerationProgram( 0, program );
        m_bindlessTexSampler->setBuffer( 0, 0, buffer );
        T* input = static_cast<T*>( buffer->map() );
        input[X] = CONSTANT;
        buffer->unmap();
    }
};

const int TestBindlessTextures::CONSTANT = 42;

std::string TestBindlessTextures::m_ptxPath( ptxPath( "test_Memory", "bindless_textures.lw" ) );
Context     TestBindlessTextures::m_context;

// -----------------------------------------------------------------------------
TEST_F( TestBindlessTextures, CanReadBoundTexture )
{
    bool declareInputBuffersAsDiscard = false;
    createProgram<int>( "readFromBoundTextureInt1D", m_inputBufferInt1D, declareInputBuffersAsDiscard );

    m_context->launch( 0, 1 );

    int* output = static_cast<int*>( m_outputBufferInt->map() );
    EXPECT_EQ( output[0], CONSTANT );
    m_outputBufferInt->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestBindlessTextures, CanReadBindlessTextureInt1D )
{
    bool declareInputBuffersAsDiscard = false;
    createProgram<int>( "readFromBindlessTextureInt1D", m_inputBufferInt1D, declareInputBuffersAsDiscard );

    m_context->launch( 0, 1 );

    int* output = static_cast<int*>( m_outputBufferInt->map() );
    EXPECT_EQ( output[0], CONSTANT );
    m_outputBufferInt->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestBindlessTextures, CanReadBindlessTextureInt2D )
{
    bool declareInputBuffersAsDiscard = false;
    createProgram<int>( "readFromBindlessTextureInt2D", m_inputBufferInt2D, declareInputBuffersAsDiscard );

    m_context->launch( 0, 1 );

    int* output = static_cast<int*>( m_outputBufferInt->map() );
    EXPECT_EQ( output[0], CONSTANT );
    m_outputBufferInt->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestBindlessTextures, CanReadBindlessTextureInt3D )
{
    bool declareInputBuffersAsDiscard = false;
    createProgram<int>( "readFromBindlessTextureInt3D", m_inputBufferInt3D, declareInputBuffersAsDiscard );

    m_context->launch( 0, 1 );

    int* output = static_cast<int*>( m_outputBufferInt->map() );
    EXPECT_EQ( output[0], CONSTANT );
    m_outputBufferInt->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestBindlessTextures, CanReadBindlessTextureFloat1D )
{
    bool declareInputBuffersAsDiscard = false;
    createProgram<float>( "readFromBindlessTextureFloat1D", m_inputBufferFloat1D, declareInputBuffersAsDiscard );

    m_context->launch( 0, 1 );

    float* output = static_cast<float*>( m_outputBufferFloat->map() );
    EXPECT_EQ( output[0], CONSTANT );
    m_outputBufferFloat->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestBindlessTextures, CanReadBindlessTextureFloat2D )
{
    bool declareInputBuffersAsDiscard = false;
    createProgram<float>( "readFromBindlessTextureFloat2D", m_inputBufferFloat2D, declareInputBuffersAsDiscard );

    m_context->launch( 0, 1 );

    float* output = static_cast<float*>( m_outputBufferFloat->map() );
    EXPECT_EQ( output[0], CONSTANT );
    m_outputBufferFloat->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestBindlessTextures, CanReadBindlessTextureFloat3D )
{
    bool declareInputBuffersAsDiscard = false;
    createProgram<float>( "readFromBindlessTextureFloat3D", m_inputBufferFloat3D, declareInputBuffersAsDiscard );

    m_context->launch( 0, 1 );

    float* output = static_cast<float*>( m_outputBufferFloat->map() );
    EXPECT_EQ( output[0], CONSTANT );
    m_outputBufferFloat->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestBindlessTextures, CanReadBoundTextureWithDiscard )
{
    bool declareInputBuffersAsDiscard = true;
    createProgram<int>( "readFromBoundTextureInt1D", m_inputBufferInt1D, declareInputBuffersAsDiscard );

    m_context->launch( 0, 1 );

    int* output = static_cast<int*>( m_outputBufferInt->map() );
    EXPECT_EQ( output[0], CONSTANT );
    m_outputBufferInt->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestBindlessTextures, CanReadBindlessTextureInt1DWithDiscard )
{
    bool declareInputBuffersAsDiscard = true;
    createProgram<int>( "readFromBindlessTextureInt1D", m_inputBufferInt1D, declareInputBuffersAsDiscard );

    m_context->launch( 0, 1 );

    int* output = static_cast<int*>( m_outputBufferInt->map() );
    EXPECT_EQ( output[0], CONSTANT );
    m_outputBufferInt->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestBindlessTextures, CanReadBindlessTextureInt2DWithDiscard )
{
    bool declareInputBuffersAsDiscard = true;
    createProgram<int>( "readFromBindlessTextureInt2D", m_inputBufferInt2D, declareInputBuffersAsDiscard );

    m_context->launch( 0, 1 );

    int* output = static_cast<int*>( m_outputBufferInt->map() );
    EXPECT_EQ( output[0], CONSTANT );
    m_outputBufferInt->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestBindlessTextures, CanReadBindlessTextureInt3DWithDiscard )
{
    bool declareInputBuffersAsDiscard = true;
    createProgram<int>( "readFromBindlessTextureInt3D", m_inputBufferInt3D, declareInputBuffersAsDiscard );

    m_context->launch( 0, 1 );

    int* output = static_cast<int*>( m_outputBufferInt->map() );
    EXPECT_EQ( output[0], CONSTANT );
    m_outputBufferInt->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestBindlessTextures, CanReadBindlessTextureFloat1DWithDiscard )
{
    bool declareInputBuffersAsDiscard = true;
    createProgram<float>( "readFromBindlessTextureFloat1D", m_inputBufferFloat1D, declareInputBuffersAsDiscard );

    m_context->launch( 0, 1 );

    float* output = static_cast<float*>( m_outputBufferFloat->map() );
    EXPECT_EQ( output[0], CONSTANT );
    m_outputBufferFloat->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestBindlessTextures, CanReadBindlessTextureFloat2DWithDiscard )
{
    bool declareInputBuffersAsDiscard = true;
    createProgram<float>( "readFromBindlessTextureFloat2D", m_inputBufferFloat2D, declareInputBuffersAsDiscard );

    m_context->launch( 0, 1 );

    float* output = static_cast<float*>( m_outputBufferFloat->map() );
    EXPECT_EQ( output[0], CONSTANT );
    m_outputBufferFloat->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestBindlessTextures, CanReadBindlessTextureFloat3DWithDiscard )
{
    bool declareInputBuffersAsDiscard = true;
    createProgram<float>( "readFromBindlessTextureFloat3D", m_inputBufferFloat3D, declareInputBuffersAsDiscard );

    m_context->launch( 0, 1 );

    float* output = static_cast<float*>( m_outputBufferFloat->map() );
    EXPECT_EQ( output[0], CONSTANT );
    m_outputBufferFloat->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestBindlessTextures, CanModifyBindlessTextureFloat2D )
{
    const bool declareInputBuffersAsDiscard = false;
    createProgram<float>( "readFromBindlessTextureFloat2D", m_inputBufferFloat2D, declareInputBuffersAsDiscard );

    m_context->launch( 0, 1 );
    {
        float* output = static_cast<float*>( m_outputBufferFloat->map() );
        EXPECT_EQ( output[0], CONSTANT );
        m_outputBufferFloat->unmap();
    }

    // Modify something and relaunch.

    m_bindlessTexSampler->setWrapMode( 0, RT_WRAP_CLAMP_TO_BORDER );

    m_context->launch( 0, 1 );
    {
        float* output = static_cast<float*>( m_outputBufferFloat->map() );
        EXPECT_EQ( output[0], CONSTANT / 2 );  // Note: lerped with border color (black)
        m_outputBufferFloat->unmap();
    }
}
