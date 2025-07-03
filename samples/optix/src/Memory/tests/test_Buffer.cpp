
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

#include <srcTests.h>

#include <optix_world.h>
#include <optixpp_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include <string>

using namespace optix;
using namespace testing;


class TestBuffers : public testing::Test
{
  public:
    Context     m_context;
    Buffer      m_inputBuffer;
    Buffer      m_outputBuffer;
    std::string m_ptxPath;

    TestBuffers()
        : m_ptxPath( ptxPath( "test_Memory", "buffer.lw" ) )
    {
    }

    void SetUp()
    {
        m_context = Context::create();
        m_context->setRayTypeCount( 1 );
        m_context->setEntryPointCount( 1 );
    }

    void TearDown()
    {
        if( m_context )
            m_context->destroy();
    }

    void setupProgram( const char* raygenName, bool applyDiscardFlag, bool bindInputBuffer )
    {
        Program rg = m_context->createProgramFromPTXFile( m_ptxPath, raygenName );
        m_context->setRayGenerationProgram( 0, rg );

        m_outputBuffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1 );
        m_context["output"]->set( m_outputBuffer );

        unsigned int bufferType = RT_BUFFER_INPUT;
        if( applyDiscardFlag )
        {
            bufferType |= RT_BUFFER_DISCARD_HOST_MEMORY;
        }
        m_inputBuffer = m_context->createBuffer( bufferType, RT_FORMAT_INT, 1 );

        if( bindInputBuffer )
            m_context["input"]->set( m_inputBuffer );

        Buffer unusedBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );
        m_context["unusedBuffer"]->set( unusedBuffer );
    }

    GeometryGroup makeGeometryGroup()
    {

        Program  ch       = m_context->createProgramFromPTXFile( m_ptxPath, "ch_simple" );
        Material material = m_context->createMaterial();
        material->setClosestHitProgram( 0, ch );

        Geometry geometry = m_context->createGeometry();
        geometry->setPrimitiveCount( 1u );

        Program bounds    = m_context->createProgramFromPTXFile( m_ptxPath, "bounds" );
        Program intersect = m_context->createProgramFromPTXFile( m_ptxPath, "intersect" );

        geometry->setBoundingBoxProgram( bounds );
        geometry->setIntersectionProgram( intersect );

        GeometryInstance gi = m_context->createGeometryInstance( geometry, &material, &material + 1 );
        GeometryGroup    gg = m_context->createGeometryGroup( &gi, &gi + 1 );
        Acceleration     as = m_context->createAcceleration( "NoAccel", "NoAccel" );
        gg->setAcceleration( as );

        return gg;
    }

    void canMapAndUnmapBeforeAndAfterSingleLaunch( bool applyDiscardFlag )
    {
        bool bindInputBuffer = true;
        setupProgram( "rg", applyDiscardFlag, bindInputBuffer );

        const int val   = 42;
        int*      input = static_cast<int*>( m_inputBuffer->map() );
        input[0]        = val;
        m_inputBuffer->unmap();
        m_context["top_object"]->set( makeGeometryGroup() );

        m_context->launch( 0, 1 /* width */ );

        int* results = static_cast<int*>( m_outputBuffer->map() );

        EXPECT_THAT( val, Eq( results[0] ) );
        m_outputBuffer->unmap();
    }

    void canLeaveBufferMapped( bool applyDiscardFlag )
    {
        bool bindInputBuffer = true;
        setupProgram( "rg", applyDiscardFlag, bindInputBuffer );

        const int val   = 42;
        int*      input = static_cast<int*>( m_inputBuffer->map() );
        input[0]        = val;

        EXPECT_NO_THROW( m_context->destroy() );
        // Make sure not to call destroy a second time during the test teardown.
        m_context = nullptr;
    }

    void inputBufferCanBeMappedBetweenLaunches( bool applyDiscardFlag )
    {
        bool bindInputBuffer = true;
        setupProgram( "rg", applyDiscardFlag, bindInputBuffer );

        const int val = 42;
        for( int i = 13; i < 16; ++i )
        {
            int* input = static_cast<int*>( m_inputBuffer->map() );
            input[0]   = val + i;
            m_inputBuffer->unmap();
            m_context["top_object"]->set( makeGeometryGroup() );

            m_context->launch( 0, 1 /* width */ );

            int* results = static_cast<int*>( m_outputBuffer->map() );

            EXPECT_THAT( val + i, Eq( results[0] ) );
            m_outputBuffer->unmap();
        }
    }

    void canQueryVariableInContext( bool applyDiscardFlag )
    {
        bool bindInputBuffer = true;
        setupProgram( "rg", applyDiscardFlag, bindInputBuffer );

        Variable v;
        EXPECT_NO_THROW( v = m_context->queryVariable( "input" ) );
        EXPECT_THAT( v.get(), NotNull() );
    }

    void canRemoveVariableFromContext( bool applyDiscardFlag )
    {
        bool bindInputBuffer = true;
        setupProgram( "rg", applyDiscardFlag, bindInputBuffer );

        Variable v = m_context->queryVariable( "input" );
        ASSERT_THAT( v.get(), NotNull() );

        m_context->removeVariable( v );

        Variable v2 = m_context->queryVariable( "input" );
        EXPECT_THAT( v2.get(), IsNull() );
    }

    void launchWithRemovedVariableThrows( bool applyDiscardFlag )
    {
        bool bindInputBuffer = true;
        setupProgram( "rg", applyDiscardFlag, bindInputBuffer );

        Variable v = m_context->queryVariable( "input" );
        m_context->removeVariable( v );
        m_context["top_object"]->set( makeGeometryGroup() );

        // This should throw "Non-initialized variable input".
        EXPECT_ANY_THROW( m_context->launch( 0, 1 /* width */ ) );
    }

    void canRemoveAndReaddVariableInContext( bool applyDiscardFlag )
    {
        bool bindInputBuffer = true;
        setupProgram( "rg", applyDiscardFlag, bindInputBuffer );
        m_context["top_object"]->set( makeGeometryGroup() );

        Variable v = m_context->queryVariable( "input" );
        m_context->removeVariable( v );

        ASSERT_NO_THROW( m_context["input"]->set( m_inputBuffer ) );

        Variable v2 = m_context->queryVariable( "input" );
        EXPECT_THAT( v2.get(), NotNull() );

        EXPECT_NO_THROW( m_context->launch( 0, 1 /* width */ ) );
    }

    void canRemoveUnusedVariableFromContext( bool applyDiscardFlag )
    {
        bool bindInputBuffer = true;
        setupProgram( "rg", applyDiscardFlag, bindInputBuffer );
        m_context["top_object"]->set( makeGeometryGroup() );

        // Remove the unused buffer variable from the context.
        Variable v = m_context->queryVariable( "unusedBuffer" );
        m_context->removeVariable( v );

        // Launch. This should still work since the buffer is unused.
        EXPECT_NO_THROW( m_context->launch( 0, 1 /* width */ ) );
    }

    void canRemoveVariableFromGeometryInstance( bool applyDiscardFlag )
    {
        bool bindInputBuffer = true;
        setupProgram( "rg", applyDiscardFlag, bindInputBuffer );

        // Create a sphere.
        Geometry sphere = m_context->createGeometry();
        sphere->setPrimitiveCount( 1 );
        sphere->setBoundingBoxProgram( m_context->createProgramFromPTXFile( m_ptxPath, "bounds" ) );
        sphere->setIntersectionProgram( m_context->createProgramFromPTXFile( m_ptxPath, "intersect" ) );
        sphere["sphere"]->setFloat( 0.f, 0.f, 0.f, 1.f );

        // Create material for the sphere.
        Material material   = m_context->createMaterial();
        Program  closestHit = m_context->createProgramFromPTXFile( m_ptxPath, "ch" );
        material->setClosestHitProgram( 0, closestHit );

        // Create additional buffers, one of which is not used.
        Buffer giBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );

        // Create geometry instance from the sphere and the material, and attach the
        // buffer to it.
        GeometryInstance gi = m_context->createGeometryInstance( sphere, &material, &material + 1 );
        gi["giBuffer"]->set( giBuffer );

        // Create a geometry group with the single instance.
        GeometryGroup group = m_context->createGeometryGroup();
        group->setAcceleration( m_context->createAcceleration( "NoAccel", "NoAccel" ) );
        group->setChildCount( 1 );
        group->setChild( 0, gi );
        m_context["top_object"]->set( group );

        // Launch once to make sure this works.
        ASSERT_NO_THROW( m_context->launch( 0, 1 /* width */ ) );

        // Now remove the gi buffer variable from the geometry instance.
        Variable v = gi->queryVariable( "giBuffer" );
        gi->removeVariable( v );

        // This should throw "Non-initialized variable giBuffer".
        EXPECT_ANY_THROW( m_context->launch( 0, 1 /* width */ ) );
    }

    void canRemoveVariableFromGIAndResolveFromGlobalScope( bool applyDiscardFlag )
    {
        bool bindInputBuffer = true;
        setupProgram( "rg", applyDiscardFlag, bindInputBuffer );

        // Create a sphere.
        Geometry sphere = m_context->createGeometry();
        sphere->setPrimitiveCount( 1 );
        sphere->setBoundingBoxProgram( m_context->createProgramFromPTXFile( m_ptxPath, "bounds" ) );
        sphere->setIntersectionProgram( m_context->createProgramFromPTXFile( m_ptxPath, "intersect" ) );
        sphere["sphere"]->setFloat( 0.f, 0.f, 0.f, 1.f );

        // Create material for the sphere.
        Material material   = m_context->createMaterial();
        Program  closestHit = m_context->createProgramFromPTXFile( m_ptxPath, "ch" );
        material->setClosestHitProgram( 0, closestHit );

        // Create additional buffer.
        Buffer giBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );

        // Create geometry instance from the sphere and the material, and attach the
        // buffer to it.
        GeometryInstance gi = m_context->createGeometryInstance( sphere, &material, &material + 1 );
        gi["giBuffer"]->set( giBuffer );

        // Create a geometry group with the single instance.
        GeometryGroup group = m_context->createGeometryGroup();
        group->setAcceleration( m_context->createAcceleration( "NoAccel", "NoAccel" ) );
        group->setChildCount( 1 );
        group->setChild( 0, gi );
        m_context["top_object"]->set( group );

        // Launch once to make sure this works.
        ASSERT_NO_THROW( m_context->launch( 0, 1 /* width */ ) );

        // Remove the buffer variable from the geometry instance.
        Variable v = gi->queryVariable( "giBuffer" );
        gi->removeVariable( v );

        // Add the buffer variable to the global scope.
        m_context["giBuffer"]->set( giBuffer );

        // This should work fine, the variable should be resolved from the global scope.
        EXPECT_NO_THROW( m_context->launch( 0, 1 /* width */ ) );
    }
};


////////////////////////////////////////////////////////////////////////////////
/// Tests
////////////////////////////////////////////////////////////////////////////////

TEST_F( TestBuffers, CanMapAndUnmapBeforeAndAfterSingleLaunch )
{
    canMapAndUnmapBeforeAndAfterSingleLaunch( /*applyDiscardFlag=*/false );
}

TEST_F( TestBuffers, CanLeaveBufferMapped )
{
    canLeaveBufferMapped( /*applyDiscardFlag=*/false );
}

TEST_F( TestBuffers, InputBufferCanBeMappedBetweenLaunches )
{
    inputBufferCanBeMappedBetweenLaunches( /*applyDiscardFlag=*/false );
}

TEST_F( TestBuffers, CanQueryVariableInContext )
{
    canQueryVariableInContext( /*applyDiscardFlag=*/false );
}

TEST_F( TestBuffers, CanRemoveVariableFromContext )
{
    canRemoveVariableFromContext( /*applyDiscardFlag=*/false );
}

TEST_F( TestBuffers, LaunchWithRemovedVariableThrows )
{
    launchWithRemovedVariableThrows( /*applyDiscardFlag=*/false );
}

TEST_F( TestBuffers, CanRemoveAndReaddVariableInContext )
{
    canRemoveAndReaddVariableInContext( /*applyDiscardFlag=*/false );
}

TEST_F( TestBuffers, CanRemoveUnusedVariableFromContext )
{
    canRemoveUnusedVariableFromContext( /*applyDiscardFlag=*/false );
}

TEST_F( TestBuffers, CanRemoveVariableFromGeometryInstance )
{
    canRemoveVariableFromGeometryInstance( /*applyDiscardFlag=*/false );
}

TEST_F( TestBuffers, CanRemoveVariableFromGIAndResolveFromGlobalScope )
{
    canRemoveVariableFromGIAndResolveFromGlobalScope( /*applyDiscardFlag=*/false );
}

TEST_F( TestBuffers, CanMapAndUnmapBeforeAndAfterSingleLaunchWithDiscard )
{
    canMapAndUnmapBeforeAndAfterSingleLaunch( /*applyDiscardFlag=*/true );
}

TEST_F( TestBuffers, CanLeaveBufferMappedWithDiscard )
{
    canLeaveBufferMapped( /*applyDiscardFlag=*/true );
}

TEST_F( TestBuffers, InputBufferCanBeMappedBetweenLaunchesWithDiscard )
{
    inputBufferCanBeMappedBetweenLaunches( /*applyDiscardFlag=*/true );
}

TEST_F( TestBuffers, CanQueryVariableInContextWithDiscard )
{
    canQueryVariableInContext( /*applyDiscardFlag=*/true );
}

TEST_F( TestBuffers, CanRemoveVariableFromContextWithDiscard )
{
    canRemoveVariableFromContext( /*applyDiscardFlag=*/true );
}

TEST_F( TestBuffers, LaunchWithRemovedVariableThrowsWithDiscard )
{
    launchWithRemovedVariableThrows( /*applyDiscardFlag=*/true );
}

TEST_F( TestBuffers, CanRemoveAndReaddVariableInContextWithDiscard )
{
    canRemoveAndReaddVariableInContext( /*applyDiscardFlag=*/true );
}

TEST_F( TestBuffers, CanRemoveUnusedVariableFromContextWithDiscard )
{
    canRemoveUnusedVariableFromContext( /*applyDiscardFlag=*/true );
}

TEST_F( TestBuffers, CanRemoveVariableFromGeometryInstanceWithDiscard )
{
    canRemoveVariableFromGeometryInstance( /*applyDiscardFlag=*/true );
}

TEST_F( TestBuffers, CanRemoveVariableFromGIAndResolveFromGlobalScopeWithDiscard )
{
    canRemoveVariableFromGIAndResolveFromGlobalScope( /*applyDiscardFlag=*/true );
}
TEST_F( TestBuffers, DiscardHostMemoryOnUnmapBindAfterUnmap )
{
    setupProgram( "rg", /*applyDiscardFlag=*/true, /*bindInputBuffer=*/false );

    RTsize memCreated = m_context->getUsedHostMemory();

    const int val       = 42;
    int*      input     = static_cast<int*>( m_inputBuffer->map() );
    RTsize    memMapped = m_context->getUsedHostMemory();
    input[0]            = val;
    m_inputBuffer->unmap();
    RTsize memUnmapped = m_context->getUsedHostMemory();

    // bind input buffer after unmapping it
    m_context["input"]->set( m_inputBuffer );

    m_context["top_object"]->set( makeGeometryGroup() );
    EXPECT_NO_THROW( m_context->launch( 0, 1 /* width */ ) );

    // maps using RT_BUFFER_MAP (no DISCARD -> trigger synch from device)
    input             = static_cast<int*>( m_inputBuffer->map() );
    const int valRead = input[0];
    m_inputBuffer->unmap();

    EXPECT_THAT( memMapped, Eq( memCreated + m_inputBuffer->getElementSize() ) );
    EXPECT_THAT( memCreated, Eq( memUnmapped ) );
    EXPECT_THAT( val, Eq( valRead ) );

    int* results = static_cast<int*>( m_outputBuffer->map() );
    EXPECT_THAT( val, Eq( results[0] ) );
    m_outputBuffer->unmap();
}

TEST_F( TestBuffers, DiscardHostMemoryOnUnmapBindBeforeUnmap )
{
    setupProgram( "rg", /*applyDiscardFlag=*/true, /*bindInputBuffer=*/true );

    RTsize memCreated = m_context->getUsedHostMemory();

    const int val       = 42;
    int*      input     = static_cast<int*>( m_inputBuffer->map() );
    RTsize    memMapped = m_context->getUsedHostMemory();
    input[0]            = val;
    m_inputBuffer->unmap();
    RTsize memUnmapped = m_context->getUsedHostMemory();

    m_context["top_object"]->set( makeGeometryGroup() );
    EXPECT_NO_THROW( m_context->launch( 0, 1 /* width */ ) );

    // maps using RT_BUFFER_MAP (no DISCARD -> trigger synch from device)
    input             = static_cast<int*>( m_inputBuffer->map() );
    const int valRead = input[0];
    m_inputBuffer->unmap();

    EXPECT_THAT( memMapped, Eq( memCreated + m_inputBuffer->getElementSize() ) );
    EXPECT_THAT( memCreated, Eq( memUnmapped ) );
    EXPECT_THAT( val, Eq( valRead ) );

    int* results = static_cast<int*>( m_outputBuffer->map() );
    EXPECT_THAT( val, Eq( results[0] ) );
    m_outputBuffer->unmap();
}

void runBufferMappingTest( Context context, Buffer& buffer2D, unsigned int bufferdesc, optix::TextureSampler* sampler2D = nullptr, bool skipCheck = false )
{
    const RTsize       width       = 8;
    const RTsize       height      = 8;
    const unsigned int levels      = 4;
    unsigned int       elementSize = 1;

    unsigned int mapFlags = RT_BUFFER_MAP_READ_WRITE;
    buffer2D              = context->createMipmappedBuffer( bufferdesc, RT_FORMAT_UNSIGNED_INT, width, height, levels );

    // fill the buffer
    {
        std::vector<void*> destinations;
        for( unsigned int level = 0; level < levels; ++level )
        {
            destinations.push_back( buffer2D->map( level, mapFlags ) );
        }
        // fill levels with increasing values
        // starting with bufferVal = 0 on base level
        RTsize       s         = buffer2D->getElementSize();
        unsigned int bufferVal = 0;
        for( unsigned int level = 0; level < levels; ++level )
        {
            unsigned int              w = width >> level;
            unsigned int              h = height >> level;
            std::vector<unsigned int> v( w * h );
            for( size_t i = 0; i < w * h; ++i )
                v[i]      = bufferVal++;
            memcpy( destinations[level], &v[0], v.size() * s );
        }
        // now unmap all levels
        for( unsigned int level = 0; level < levels; ++level )
        {
            buffer2D->unmap( level );
        }
    }

    if( sampler2D )
    {
        *sampler2D = context->createTextureSampler();
        ( *sampler2D )->setWrapMode( 0, RT_WRAP_CLAMP_TO_EDGE );
        ( *sampler2D )->setWrapMode( 1, RT_WRAP_CLAMP_TO_EDGE );
        ( *sampler2D )->setWrapMode( 2, RT_WRAP_CLAMP_TO_EDGE );
        ( *sampler2D )->setFilteringModes( RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NEAREST );
        ( *sampler2D )->setIndexingMode( RT_TEXTURE_INDEX_ARRAY_INDEX );
        ( *sampler2D )->setReadMode( RT_TEXTURE_READ_ELEMENT_TYPE );
        ( *sampler2D )->setMaxAnisotropy( 1.0f );

        // NOTE: This triggers the Policy change and DtoArray copy. Would not be necessary if
        //       TextureSampler would have been attached before the map/unmap stuff.
        EXPECT_NO_THROW( ( *sampler2D )->setBuffer( buffer2D ) );
    }

    if( skipCheck )
        return;

    // now do the real test; no need to do it efficiently and can access level by level sequentially
    unsigned int expectedValue = 0;
    for( unsigned int level = 0; level < levels; ++level )
    {
        unsigned int w = width >> level;
        unsigned int h = height >> level;

        void*         vec = buffer2D->map( level, mapFlags );
        unsigned int* ptr = static_cast<unsigned int*>( vec );

        for( size_t i = 0; i < w * h; ++i )
        {
            ASSERT_EQ( expectedValue++, *ptr++ );
        }
        buffer2D->unmap( level );
    }
}

TEST_F( TestBuffers, RemapMipmap )
{
    Buffer buffer2D;
    runBufferMappingTest( m_context, buffer2D, RT_BUFFER_INPUT );
}

TEST_F( TestBuffers, RemapMipmapWithTexSampler )
{
    Buffer                buffer2D;
    optix::TextureSampler sampler2D;
    runBufferMappingTest( m_context, buffer2D, RT_BUFFER_INPUT, &sampler2D );
}

TEST_F( TestBuffers, RemapMipmapWithDiscardHostMemory )
{
    Buffer buffer2D;
    runBufferMappingTest( m_context, buffer2D, RT_BUFFER_INPUT | RT_BUFFER_DISCARD_HOST_MEMORY );
}

TEST_F( TestBuffers, RemapMipmapWithDiscardHostMemoryAndTexSampler )
{
    Buffer                buffer2D;
    optix::TextureSampler sampler2D;
    runBufferMappingTest( m_context, buffer2D, RT_BUFFER_INPUT | RT_BUFFER_DISCARD_HOST_MEMORY, &sampler2D );
}

TEST_F( TestBuffers, AccessMipmapOnDevice )
{
    Buffer                buffer2D;
    optix::TextureSampler sampler2D;
    runBufferMappingTest( m_context, buffer2D, RT_BUFFER_INPUT, &sampler2D, false );

    Program rg = m_context->createProgramFromPTXFile( m_ptxPath, "accessMipmapSquarredArray" );
    m_context->setRayGenerationProgram( 0, rg );
    //m_context->setPrintEnabled(1);
    //m_context->setPrintBufferSize(4096);

    const RTsize       width  = 8;
    const RTsize       height = 8;
    const unsigned int levels = 4;

    m_context["tex_2d_sampler"]->setTextureSampler( sampler2D );
    m_context["texId"]->setInt( sampler2D->getId() );
    m_context["levels"]->setInt( levels );
    m_context["width"]->setInt( width );
    m_context["height"]->setInt( height );

    // for returning success or failure
    Buffer successBuffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, 1 );
    int*   successPtr    = static_cast<int*>( successBuffer->map() );
    *successPtr          = 0;
    successBuffer->unmap();
    m_context["success"]->set( successBuffer );

    EXPECT_NO_THROW( m_context->launch( 0, 1 /* width */ ) );

    // check for success or failure
    int* success = static_cast<int*>( successBuffer->map() );
    EXPECT_TRUE( *success == 1 );
    successBuffer->unmap();
}

TEST_F( TestBuffers, AccessMipmapOnDeviceWithDiscardHostMemory )
{
    Buffer                buffer2D;
    optix::TextureSampler sampler2D;
    runBufferMappingTest( m_context, buffer2D, RT_BUFFER_INPUT | RT_BUFFER_DISCARD_HOST_MEMORY, &sampler2D, false );

    Program rg = m_context->createProgramFromPTXFile( m_ptxPath, "accessMipmapSquarredArray" );
    m_context->setRayGenerationProgram( 0, rg );
    //m_context->setPrintEnabled(1);
    //m_context->setPrintBufferSize(4096);

    const RTsize       width  = 8;
    const RTsize       height = 8;
    const unsigned int levels = 4;

    m_context["tex_2d_sampler"]->setTextureSampler( sampler2D );
    m_context["texId"]->setInt( sampler2D->getId() );
    m_context["levels"]->setInt( levels );
    m_context["width"]->setInt( width );
    m_context["height"]->setInt( height );

    // for returning success or failure
    Buffer successBuffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, 1 );
    int*   successPtr    = static_cast<int*>( successBuffer->map() );
    *successPtr          = 0;
    successBuffer->unmap();
    m_context["success"]->set( successBuffer );

    EXPECT_NO_THROW( m_context->launch( 0, 1 /* width */ ) );

    // check for success or failure
    int* success = static_cast<int*>( successBuffer->map() );
    EXPECT_TRUE( *success == 1 );
    successBuffer->unmap();
}

TEST_F( TestBuffers, DiscardHostMemoryMipmappedBuffers )
{
    const RTsize       width  = 8;
    const RTsize       height = 8;
    const RTsize       depth  = 8;  // == layers, 3D depth
    const unsigned int levels = 4;  // == mipmaps
    const unsigned int faces  = 6;  // == lwbeface

    // 1D layered
    //    optix::Buffer buffer1D = m_context->create1DLayeredBuffer( RT_BUFFER_INPUT | RT_BUFFER_DISCARD_HOST_MEMORY, RT_FORMAT_UNSIGNED_BYTE4, width, depth, 1 );
    optix::Buffer buffer1D = m_context->createBuffer( RT_BUFFER_INPUT | RT_BUFFER_LAYERED | RT_BUFFER_DISCARD_HOST_MEMORY,
                                                      RT_FORMAT_UNSIGNED_BYTE4, width, depth );

    buffer1D->map( 0, RT_BUFFER_MAP_WRITE_DISCARD );
    EXPECT_NO_THROW( buffer1D->unmap( 0 ) );  // Trigger the discard.

    optix::TextureSampler sampler1D = m_context->createTextureSampler();

    sampler1D->setWrapMode( 0, RT_WRAP_REPEAT );
    sampler1D->setWrapMode( 1, RT_WRAP_REPEAT );
    sampler1D->setWrapMode( 2, RT_WRAP_REPEAT );
    sampler1D->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );
    sampler1D->setIndexingMode( RT_TEXTURE_INDEX_NORMALIZED_COORDINATES );
    sampler1D->setReadMode( RT_TEXTURE_READ_NORMALIZED_FLOAT );
    sampler1D->setMaxAnisotropy( 1.0f );
    EXPECT_NO_THROW( sampler1D->setBuffer( buffer1D ) );

    // 2D layered mipmapped
    optix::Buffer buffer2D = m_context->createBuffer( RT_BUFFER_INPUT | RT_BUFFER_LAYERED | RT_BUFFER_DISCARD_HOST_MEMORY,
                                                      RT_FORMAT_UNSIGNED_BYTE4, width, height, depth );
    buffer2D->setMipLevelCount( levels );

    for( unsigned int level = 0; level < levels; ++level )
    {
        // NOTE: RT_BUFFER_MAP_WRITE_DISCARD gets rewritten to RT_BUFFER_READ_WRITE
        //       in Buffer::map, if( getMipLevelCount() > 1 ). So this will trigger
        //       a sync DtoH with deviceKind == LINEAR and hostKind == PITCHED_LINEAR
        //       on the second iteration.
        EXPECT_NO_THROW( buffer2D->map( level, RT_BUFFER_MAP_WRITE_DISCARD ) );

        // NOTE: Buffer does not yet have a TextureSampler attached, so this will
        //       sync HtoD with hostKind == PITCHED_LINEAR and deviceKind == LINEAR.
        EXPECT_NO_THROW( buffer2D->unmap( level ) );
    }

    optix::TextureSampler sampler2D = m_context->createTextureSampler();
    sampler2D->setWrapMode( 0, RT_WRAP_REPEAT );
    sampler2D->setWrapMode( 1, RT_WRAP_REPEAT );
    sampler2D->setWrapMode( 2, RT_WRAP_REPEAT );
    sampler2D->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );
    sampler2D->setIndexingMode( RT_TEXTURE_INDEX_NORMALIZED_COORDINATES );
    sampler2D->setReadMode( RT_TEXTURE_READ_NORMALIZED_FLOAT );
    sampler2D->setMaxAnisotropy( 1.0f );

    // NOTE: This triggers the Policy change and DtoArray copy. Would not be necessary if
    //       TextureSampler would have been attached before the map/unmap stuff.
    EXPECT_NO_THROW( sampler2D->setBuffer( buffer2D ) );

    // 3D
    optix::Buffer buffer3D = m_context->createBuffer( RT_BUFFER_INPUT /*| RT_BUFFER_DISCARD_HOST_MEMORY*/,
                                                      RT_FORMAT_UNSIGNED_BYTE4, width, height, depth );
    buffer3D->setMipLevelCount( levels );

    for( unsigned int level = 0; level < levels; ++level )
    {
        EXPECT_NO_THROW( buffer3D->map( level, RT_BUFFER_MAP_WRITE_DISCARD ) );
        EXPECT_NO_THROW( buffer3D->unmap( level ) );
    }
    optix::TextureSampler sampler3D = m_context->createTextureSampler();


    sampler3D->setWrapMode( 0, RT_WRAP_REPEAT );
    sampler3D->setWrapMode( 1, RT_WRAP_REPEAT );
    sampler3D->setWrapMode( 2, RT_WRAP_REPEAT );
    sampler3D->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );
    sampler3D->setIndexingMode( RT_TEXTURE_INDEX_NORMALIZED_COORDINATES );
    sampler3D->setReadMode( RT_TEXTURE_READ_NORMALIZED_FLOAT );
    sampler3D->setMaxAnisotropy( 1.0f );
    EXPECT_NO_THROW( sampler3D->setBuffer( buffer3D ) );

    // Lwbemap
    optix::Buffer bufferLwbe = m_context->createBuffer( RT_BUFFER_INPUT | RT_BUFFER_LWBEMAP | RT_BUFFER_DISCARD_HOST_MEMORY,
                                                        RT_FORMAT_UNSIGNED_BYTE4, width, height, faces );
    bufferLwbe->setMipLevelCount( levels );

    for( unsigned int level = 0; level < levels; ++level )
    {
        EXPECT_NO_THROW( bufferLwbe->map( level, RT_BUFFER_MAP_WRITE_DISCARD ) );
        EXPECT_NO_THROW( bufferLwbe->unmap( level ) );
    }

    optix::TextureSampler samplerLwbe = m_context->createTextureSampler();

    samplerLwbe->setWrapMode( 0, RT_WRAP_REPEAT );
    samplerLwbe->setWrapMode( 1, RT_WRAP_REPEAT );
    samplerLwbe->setWrapMode( 2, RT_WRAP_REPEAT );
    samplerLwbe->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );
    samplerLwbe->setIndexingMode( RT_TEXTURE_INDEX_NORMALIZED_COORDINATES );
    samplerLwbe->setReadMode( RT_TEXTURE_READ_NORMALIZED_FLOAT );
    samplerLwbe->setMaxAnisotropy( 1.0f );
    EXPECT_NO_THROW( samplerLwbe->setBuffer( bufferLwbe ) );
}

TEST_F( TestBuffers, IlwalidBufferOptionsForDiscardHostMemoryThrows )
{
    EXPECT_ANY_THROW( m_context->createBuffer( RT_BUFFER_OUTPUT | RT_BUFFER_DISCARD_HOST_MEMORY, RT_FORMAT_INT, 1 ) );
    EXPECT_ANY_THROW( m_context->createBuffer( RT_BUFFER_GPU_LOCAL | RT_BUFFER_DISCARD_HOST_MEMORY, RT_FORMAT_INT, 1 ) );
}
