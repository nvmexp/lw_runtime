//
//  Copyright (c) 2017 LWPU Corporation.  All rights reserved.
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

// LwdaDriver expects lwca types to be in the global namespace.  We can force the optix headers
// to use the global namespace lwca types by including these two lwca headers first.
#include <vector_functions.h>
#include <vector_types.h>

#include <srcTests.h>
#include <tests/Image.h>
#include <tests/ObjLoader.h>

#include <optix_world.h>
#include <optixu/optixu_matrix_namespace.h>

#include <corelib/system/LwdaDriver.h>

#include <algorithm>
#include <ostream>
#include <sstream>

// TODO: Make these true unit tests. They take a very long time to run. We
// can only do this when we link against a static optix library

using namespace optix;
using namespace testing;
using corelib::lwdaDriver;

#define LWDA_CHK( code ) EXPECT_THAT( code, Eq( LWDA_SUCCESS ) )

namespace {

const unsigned int MAX_GOLD_IMAGE_ERRORS = 64;

struct PTXModule
{
    const char* description;
    const char* metadata;
    const char* code;
};

}  // namespace

// clang-format off
#define PTX_MODULE( desc, ... )                                                                                        \
  {                                                                                                                    \
    desc, "", #__VA_ARGS__                                                                                             \
  }

PTXModule progPtxEmpty = PTX_MODULE( "prog_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32

  .entry prog
  {
    ret;
  }

);
// clang-format on

namespace {

//------------------------------------------------------------------------------
class ActiveDevice
{
  public:
    ActiveDevice( int dev )
        : m_dev( dev )
    {
        LWcontext newCtx;
        LWDA_CHK( lwdaDriver().LwDevicePrimaryCtxRetain( &newCtx, m_dev ) );
        EXPECT_TRUE( newCtx != nullptr );
        LWcontext context = nullptr;
        LWDA_CHK( lwdaDriver().LwCtxGetLwrrent( &context ) );
        if( context != newCtx )
        {
            m_prevCtx = context;
            LWDA_CHK( lwdaDriver().LwCtxSetLwrrent( newCtx ) );
        }
    }

    ~ActiveDevice()
    {
        if( m_prevCtx )
        {
            LWDA_CHK( lwdaDriver().LwCtxSetLwrrent( m_prevCtx ) );
        }
        LWDA_CHK( lwdaDriver().LwDevicePrimaryCtxRelease( m_dev ) );
    }

  private:
    int       m_dev     = 0;
    LWcontext m_prevCtx = nullptr;
};

class LwdaMemHog
{
  public:
    LwdaMemHog( int device )
        : m_device( device )
    {
        // allocating the amount of free memory doesn't always work so allocate in
        // smaller chunks
        size_t       free;
        ActiveDevice ad( m_device );
        LWDA_CHK( lwdaDriver().LwMemGetInfo( &free, nullptr ) );
        float        fraction        = 0.8f;
        const size_t MINIMUM_INITIAL = 32 << 20;  // 32 MB
        const size_t MINIMUM_FINAL   = 1 << 20;   //  1 MB
        while( free > MINIMUM_FINAL )
        {
            char*  block     = nullptr;
            size_t blockSize = size_t( free * fraction );
            if( lwdaDriver().LwMemAlloc( (LWdeviceptr*)&block, blockSize ) != LWDA_SUCCESS )
            {
                if( m_blocks.size() == 0 )
                {
                    // Fail if we can't even allocate 32 MB
                    if( blockSize <= MINIMUM_INITIAL )
                    {
                        printf( "WARNING: Could not allocate even %llu bytes\n", static_cast<unsigned long long>( MINIMUM_INITIAL ) );
                        break;
                    }

                    // Throttle fraction
                    fraction *= 0.5f;
                }
                else
                {
                    break;
                }
            }
            m_blocks.push_back( block );
            LWDA_CHK( lwdaDriver().LwMemGetInfo( &free, nullptr ) );
            printf( "free = %llu, blocksize = %llu\n", static_cast<unsigned long long>( free ),
                    static_cast<unsigned long long>( blockSize ) );
        }
    }

    ~LwdaMemHog()
    {
        ActiveDevice as( m_device );
        for( int i = (int)m_blocks.size() - 1; i >= 0; --i )
            LWDA_CHK( lwdaDriver().LwMemFree( (LWdeviceptr)m_blocks[i] ) );
    }

  private:
    std::vector<char*> m_blocks;
    int                m_device;
};

//------------------------------------------------------------------------------
class SimpleOptix : public Test
{
  public:
    SimpleOptix() {}

    //------------------------------------------------------------------------------
    void SetUp() override
    {
        createContext();
        if( m_loadGeometryInSetup )
            loadGeometry();
    }

    //------------------------------------------------------------------------------
    void TearDown() override { destroyContext(); }

    //------------------------------------------------------------------------------
    void createContext()
    {
        m_context = Context::create();
        m_context->setRayTypeCount( 1 );
        m_context->setEntryPointCount( 1 );
        m_context["radiance_ray_type"]->setUint( 0u );
        m_context["scene_epsilon"]->setFloat( 1.e-4f );
        m_context->setAttribute( RT_CONTEXT_ATTRIBUTE_GPU_PAGING_FORCED_OFF, 1 );

        // Ray gen program
        std::string ptx_path = ptxPath( m_target, "orthographic_camera.lw" );
        m_context->setRayGenerationProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "orthographic_camera" ) );

        // Exception program
        m_context->setExceptionProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );
        m_context["bad_color"]->setFloat( 1.0f, 1.0f, 0.0f );

        // Miss program
        ptx_path = ptxPath( m_target, "constantbg.lw" );
        m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "miss" ) );
        m_context["bg_color"]->setFloat( make_float3( .3f, .3f, .3f ) );

        // Material
        m_context["shading_offset"]->setFloat( make_float3( 0.0f ) );
        m_material = m_context->createMaterial();
        ptx_path   = ptxPath( m_target, "normal_shader.lw" );
        m_material->setClosestHitProgram( 0u, m_context->createProgramFromPTXFile( ptx_path, "closest_hit_radiance" ) );

        // Output buffer
        m_outputBuffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height );
        m_context["output_buffer"]->set( m_outputBuffer );

        m_goldImagePrefix       = "cow";
        m_goldImageMaxNumErrors = 0;
    }

    //------------------------------------------------------------------------------
    void createCowModel()
    {
        m_topGeometryGroup       = m_context->createGeometryGroup();
        std::string    filename  = dataPath() + "/cow.obj";
        std::string    ptx_path  = ptxPath( m_target, "triangle_mesh.lw" );
        optix::Program intersect = m_context->createProgramFromPTXFile( ptx_path, "mesh_intersect" );
        optix::Program bbox      = m_context->createProgramFromPTXFile( ptx_path, "mesh_bounds" );
        ObjLoader      loader( filename, m_context, m_topGeometryGroup, intersect, bbox, m_material, m_accelType );
        loader.load();
        m_sceneBounds = loader.getSceneBBox();
        m_context["top_object"]->set( m_topGeometryGroup );
    }

    //------------------------------------------------------------------------------
    void createCowsGroupModel()
    {
        createCowModel();
        GeometryGroup gg   = m_topGeometryGroup;
        m_topGeometryGroup = nullptr;

        float width = m_sceneBounds.extent( 0 );
        m_sceneBounds.m_max.x += ( m_groupSize - 1 ) * width;

        m_topGroup = m_context->createGroup();
        m_topGroup->setChildCount( m_groupSize );
        for( int i = 0; i < m_groupSize; ++i )
        {
            float     matrix[]    = {1, 0, 0, i * width, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            float     ilwMatrix[] = {1, 0, 0, -i * width, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            Transform transform   = m_context->createTransform();
            transform->setChild( gg );
            transform->setMatrix( false, matrix, ilwMatrix );

            m_topGroup->setChild( i, transform );
        }
        m_context["top_object"]->set( m_topGroup );

        m_goldImagePrefix       = "cows";
        m_goldImageMaxNumErrors = MAX_GOLD_IMAGE_ERRORS;
    }

    //------------------------------------------------------------------------------
    void createSpheres( int numSpheres, std::vector<float4>& spheres, Aabb& bounds )
    {
        float dTheta       = float( 2 * M_PI / numSpheres );
        float radius       = 1;
        float sphereRadius = radius * dTheta * 0.25f;
        float r            = radius - sphereRadius;
        spheres.resize( numSpheres );
        for( int i = 0; i < numSpheres; ++i )
        {
            float theta = i * dTheta;
            spheres[i]  = make_float4( r * cosf( theta ), r * sinf( theta ), 0, sphereRadius );
        }
        bounds.set( make_float3( -radius, -radius, -radius ), make_float3( radius, radius, radius ) );
    }

    //------------------------------------------------------------------------------
    void createSpheresModel()
    {
        std::vector<float4> spheres;
        createSpheres( m_groupSize, spheres, m_sceneBounds );

        Buffer sphereBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, m_groupSize );
        Buffer matBuffer    = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, m_groupSize );

        float4* spherePtr = (float4*)sphereBuffer->map();
        int*    matPtr    = (int*)matBuffer->map();
        for( int i = 0; i < m_groupSize; ++i )
        {
            spherePtr[i] = spheres[i];
            matPtr[i]    = 0;
        }
        sphereBuffer->unmap();
        matBuffer->unmap();

        const std::string sphereListPtx( ptxPath( m_target, "sphere_list.lw" ) );
        Geometry          geometry = m_context->createGeometry();
        geometry->setPrimitiveCount( m_groupSize );
        geometry->setIntersectionProgram( m_context->createProgramFromPTXFile( sphereListPtx, "intersect" ) );
        if( m_boundsType == "default" )
            geometry->setBoundingBoxProgram( m_context->createProgramFromPTXFile( sphereListPtx, "bounds" ) );
        else
        {
            const std::string boundsPtx = ptxPath( m_target, "bounds.lw" );
            geometry->setBoundingBoxProgram(
                m_context->createProgramFromPTXFile( boundsPtx, m_boundsType + "_bounds" ) );
        }
        geometry["sphere_buffer"]->setBuffer( sphereBuffer );
        geometry["material_buffer"]->setBuffer( matBuffer );

        GeometryInstance geomInstance = m_context->createGeometryInstance();
        geomInstance->addMaterial( m_material );
        geomInstance->setGeometry( geometry );

        m_topGeometryGroup = m_context->createGeometryGroup();
        m_topGeometryGroup->setChildCount( 1 );
        m_topGeometryGroup->setChild( 0, geomInstance );

        m_context["top_object"]->set( m_topGeometryGroup );

        m_goldImagePrefix       = "spheres";
        m_goldImageMaxNumErrors = MAX_GOLD_IMAGE_ERRORS;
    }

    //------------------------------------------------------------------------------
    void createSpheresGroupModel()
    {
        std::vector<float4> spheres;
        createSpheres( m_groupSize, spheres, m_sceneBounds );

        std::string sphere_ptx( ptxPath( m_target, "sphere.lw" ) );
        m_topGroup = m_context->createGroup();
        m_topGroup->setChildCount( m_groupSize );
        for( int i = 0; i < m_groupSize; ++i )
        {
            Geometry geometry = m_context->createGeometry();
            geometry->setPrimitiveCount( 1 );
            geometry->setIntersectionProgram( m_context->createProgramFromPTXFile( sphere_ptx, "intersect" ) );
            geometry->setBoundingBoxProgram( m_context->createProgramFromPTXFile( sphere_ptx, "bounds" ) );
            geometry["sphere"]->setFloat( spheres[i] );

            Acceleration giAccel = m_context->createAcceleration( m_accelType, m_builderType );

            GeometryInstance geomInstance = m_context->createGeometryInstance();
            geomInstance->addMaterial( m_material );
            geomInstance->setGeometry( geometry );

            GeometryGroup geomGroup = m_context->createGeometryGroup();
            geomGroup->setChildCount( 1 );
            geomGroup->setChild( 0, geomInstance );
            geomGroup->setAcceleration( giAccel );

            m_topGroup->setChild( i, geomGroup );
        }

        m_context["top_object"]->set( m_topGroup );

        m_goldImagePrefix       = "spheres";
        m_goldImageMaxNumErrors = MAX_GOLD_IMAGE_ERRORS;
    }

    //------------------------------------------------------------------------------
    void createSpheresTransformedModel()
    {
        std::vector<float4> spheres;
        createSpheres( m_groupSize, spheres, m_sceneBounds );

        std::string sphere_ptx( ptxPath( m_target, "sphere.lw" ) );

        // Shared geometry
        Geometry geometry = m_context->createGeometry();
        geometry->setPrimitiveCount( 1 );
        geometry->setIntersectionProgram( m_context->createProgramFromPTXFile( sphere_ptx, "intersect" ) );
        geometry->setBoundingBoxProgram( m_context->createProgramFromPTXFile( sphere_ptx, "bounds" ) );

        {
            const float radius = spheres[0].w;  // assume same radius for all spheres in group
            geometry["sphere"]->setFloat( make_float4( 0, 0, 0, radius ) );
        }

        // Shared bottom-level accel
        Acceleration ggAccel = m_context->createAcceleration( m_accelType, m_builderType );

        // Transformed instances

        m_topGroup = m_context->createGroup();
        m_topGroup->setChildCount( m_groupSize );
        for( int i = 0; i < m_groupSize; ++i )
        {
            // This is where users would attach variables to change the color, etc. of an instance
            GeometryInstance geomInstance = m_context->createGeometryInstance();
            geomInstance->addMaterial( m_material );
            geomInstance->setGeometry( geometry );  // shared geometry

            GeometryGroup geomGroup = m_context->createGeometryGroup();
            geomGroup->setChildCount( 1 );
            geomGroup->setChild( 0, geomInstance );
            geomGroup->setAcceleration( ggAccel );  // shared accel

            float3           pos    = make_float3( spheres[i] );
            optix::Matrix4x4 matrix = optix::Matrix4x4::translate( pos );

            Transform transform = m_context->createTransform();
            transform->setChild( geomGroup );
            transform->setMatrix( false, matrix.getData(), nullptr );

            m_topGroup->setChild( i, transform );
        }

        m_context["top_object"]->set( m_topGroup );

        m_goldImagePrefix       = "spheres";
        m_goldImageMaxNumErrors = MAX_GOLD_IMAGE_ERRORS;
    }

    //------------------------------------------------------------------------------
    void createSpheresGroupModelWithEmptyChild()
    {
        std::vector<float4> spheres;
        createSpheres( m_groupSize, spheres, m_sceneBounds );

        std::string sphere_ptx( ptxPath( m_target, "sphere.lw" ) );

        m_topGroup = m_context->createGroup();
        m_topGroup->setChildCount( m_groupSize + 1 );  // +1 for empty child
        for( int i = 0; i < m_groupSize; ++i )
        {
            Geometry geometry = m_context->createGeometry();
            geometry->setPrimitiveCount( 1 );
            geometry->setIntersectionProgram( m_context->createProgramFromPTXFile( sphere_ptx, "intersect" ) );
            geometry->setBoundingBoxProgram( m_context->createProgramFromPTXFile( sphere_ptx, "bounds" ) );
            geometry["sphere"]->setFloat( spheres[i] );

            Acceleration giAccel = m_context->createAcceleration( m_accelType, m_builderType );

            GeometryInstance geomInstance = m_context->createGeometryInstance();
            geomInstance->addMaterial( m_material );
            geomInstance->setGeometry( geometry );

            GeometryGroup geomGroup = m_context->createGeometryGroup();
            geomGroup->setChildCount( 1 );
            geomGroup->setChild( 0, geomInstance );
            geomGroup->setAcceleration( giAccel );

            m_topGroup->setChild( i, geomGroup );
        }

        // Add empty geometry group
        {
            GeometryGroup geomGroup = m_context->createGeometryGroup();
            geomGroup->setChildCount( 0 );
            Acceleration giAccel = m_context->createAcceleration( m_accelType, m_builderType );
            geomGroup->setAcceleration( giAccel );
            m_topGroup->setChild( m_groupSize, geomGroup );
        }

        m_context["top_object"]->set( m_topGroup );

        m_goldImagePrefix       = "spheres";
        m_goldImageMaxNumErrors = 65;
    }

    //------------------------------------------------------------------------------
    void createEmptySelector()
    {
        m_topGroup        = m_context->createGroup();
        Selector emptySel = m_context->createSelector();

        optix::Program emptyVisit = m_context->createProgramFromPTXString( progPtxEmpty.code, "prog" );

        emptySel->setVisitProgram( emptyVisit );

        m_topGroup->setChildCount( 1 );
        m_topGroup->setChild( 0, emptySel );

        m_sceneBounds.include( make_float3( 0.f ) );
        m_sceneBounds.include( make_float3( 1.f ) );
        m_context["top_object"]->set( m_topGroup );
    }

    //------------------------------------------------------------------------------
    void createEmptyGeometryGroup()
    {
        m_topGroup = m_context->createGroup();
        m_topGroup->setChildCount( 1 );

        GeometryGroup gg    = m_context->createGeometryGroup();
        Acceleration  accel = m_context->createAcceleration( m_accelType, m_builderType );
        gg->setAcceleration( accel );
        m_topGroup->setChild( 0, gg );

        m_sceneBounds.include( make_float3( 0.f ) );
        m_sceneBounds.include( make_float3( 1.f ) );
        m_context["top_object"]->set( m_topGroup );
    }

    //------------------------------------------------------------------------------
    void createZeroPrimitives()
    {
        Buffer vBuffer      = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 0 );
        Buffer vIndexBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, 0 );

        createIndexedTriangleMeshGraph( vBuffer, vIndexBuffer, 0 );

        m_goldImagePrefix       = "zeroPrimitives";
        m_goldImageMaxNumErrors = 0;
    }

    //------------------------------------------------------------------------------
    void createMixedValidPrimitives()
    {
        Buffer  vBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 3 );
        float3* verts   = static_cast<float3*>( vBuffer->map() );
        verts[0]        = make_float3( 0.f, 0.0f, 0.5f );
        verts[1]        = make_float3( 100.0f, 0.0f, 0.5f );
        verts[2]        = make_float3( 100.0f, 100.0f, 0.5f );
        vBuffer->unmap();

        Buffer  normalsBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1 );
        float3* normals       = static_cast<float3*>( normalsBuffer->map() );
        normals[0]            = make_float3( 0.f, 0.f, -1.0f );
        normalsBuffer->unmap();

        Buffer normalIndicesBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, 2 );
        int3*  normalIndices       = static_cast<int3*>( normalIndicesBuffer->map() );
        normalIndices[0]           = make_int3( 0 );
        normalIndices[1]           = make_int3( 0 );
        normalIndicesBuffer->unmap();

        Buffer vIndexBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, 2 );
        int3*  indices      = static_cast<int3*>( vIndexBuffer->map() );
        indices[0]          = make_int3( 0 );
        indices[1]          = make_int3( 0, 1, 2 );
        vIndexBuffer->unmap();

        createIndexedTriangleMeshGraph( vBuffer, vIndexBuffer, normalsBuffer, normalIndicesBuffer, 2 );

        m_goldImagePrefix       = "mixedValidPrimitives";
        m_goldImageMaxNumErrors = MAX_GOLD_IMAGE_ERRORS;
    }

    //------------------------------------------------------------------------------
    void createSinglePrimitive()
    {
        Buffer  vBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 3 );
        float3* verts   = static_cast<float3*>( vBuffer->map() );
        verts[0]        = make_float3( 0.f, 0.0f, 0.5f );
        verts[1]        = make_float3( 100.0f, 0.0f, 0.5f );
        verts[2]        = make_float3( 100.0f, 100.0f, 0.5f );
        vBuffer->unmap();

        Buffer  normalsBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1 );
        float3* normals       = static_cast<float3*>( normalsBuffer->map() );
        normals[0]            = make_float3( 0.f, 0.f, -1.0f );
        normalsBuffer->unmap();

        Buffer normalIndicesBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, 1 );
        int3*  normalIndices       = static_cast<int3*>( normalIndicesBuffer->map() );
        normalIndices[0]           = make_int3( 0 );
        normalIndicesBuffer->unmap();

        Buffer vIndexBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, 1 );
        int3*  indices      = static_cast<int3*>( vIndexBuffer->map() );
        indices[0]          = make_int3( 0, 1, 2 );
        vIndexBuffer->unmap();

        createIndexedTriangleMeshGraph( vBuffer, vIndexBuffer, normalsBuffer, normalIndicesBuffer, 1 );

        m_goldImagePrefix       = "singlePrimitive";
        m_goldImageMaxNumErrors = MAX_GOLD_IMAGE_ERRORS;
    }

    //------------------------------------------------------------------------------
    void createPrimitivesWithOffset()
    {
        const unsigned int NUM_PRIMITIVES = 2;
        const unsigned int NUM_VERTICES   = 3 * NUM_PRIMITIVES;

        Buffer  vBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, NUM_VERTICES );
        float3* verts   = static_cast<float3*>( vBuffer->map() );
        verts[0]        = make_float3( 0.f, 0.0f, 0.5f );
        verts[1]        = make_float3( 100.0f, 0.0f, 0.5f );
        verts[2]        = make_float3( 100.0f, 100.0f, 0.5f );
        verts[3]        = make_float3( 0.f, 0.0f, 0.5f );
        verts[4]        = make_float3( 100.0f, 0.0f, 0.5f );
        verts[5]        = make_float3( 100.0f, 100.0f, 0.5f );
        vBuffer->unmap();

        Buffer  normalsBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, NUM_VERTICES );
        float3* normals       = static_cast<float3*>( normalsBuffer->map() );
        std::fill( &normals[0], &normals[NUM_VERTICES], make_float3( 0.f, 0.f, -1.0f ) );
        normalsBuffer->unmap();

        createTriangleMeshGraph( vBuffer, normalsBuffer, 1, 1 );

        m_goldImagePrefix       = "primitivesWithOffset";
        m_goldImageMaxNumErrors = MAX_GOLD_IMAGE_ERRORS;
    }

    //------------------------------------------------------------------------------
    void createIndexedPrimitivesWithOffset()
    {
        const unsigned int NUM_PRIMITIVES = 2;

        Buffer  vBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 3 * NUM_PRIMITIVES );
        float3* verts   = static_cast<float3*>( vBuffer->map() );
        verts[0]        = make_float3( 0.f, 0.0f, 0.5f );
        verts[1]        = make_float3( 100.0f, 0.0f, 0.5f );
        verts[2]        = make_float3( 100.0f, 100.0f, 0.5f );
        verts[3]        = make_float3( 0.f, 0.0f, 0.5f );
        verts[4]        = make_float3( 100.0f, 0.0f, 0.5f );
        verts[5]        = make_float3( 100.0f, 100.0f, 0.5f );
        vBuffer->unmap();

        Buffer  normalsBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, NUM_PRIMITIVES );
        float3* normals       = static_cast<float3*>( normalsBuffer->map() );
        std::fill( &normals[0], &normals[NUM_PRIMITIVES], make_float3( 0.f, 0.f, -1.0f ) );
        normalsBuffer->unmap();

        Buffer normalIndicesBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, NUM_PRIMITIVES );
        int3*  normalIndices       = static_cast<int3*>( normalIndicesBuffer->map() );
        std::fill( &normalIndices[0], &normalIndices[NUM_PRIMITIVES], make_int3( 0 ) );
        normalIndicesBuffer->unmap();

        Buffer vIndexBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, NUM_PRIMITIVES );
        int3*  indices      = static_cast<int3*>( vIndexBuffer->map() );
        std::fill( &indices[0], &indices[NUM_PRIMITIVES], make_int3( 0, 1, 2 ) );
        vIndexBuffer->unmap();

        createIndexedTriangleMeshGraph( vBuffer, vIndexBuffer, normalsBuffer, normalIndicesBuffer, 1, 1 );

        m_goldImagePrefix       = "indexedPrimitivesWithOffset";
        m_goldImageMaxNumErrors = MAX_GOLD_IMAGE_ERRORS;
    }

    //------------------------------------------------------------------------------
    Buffer createNalwertices()
    {
        const unsigned int NUM_PRIMITIVES = 3;

        Buffer  vBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 3 * NUM_PRIMITIVES );
        float3* verts   = static_cast<float3*>( vBuffer->map() );
        verts[0]        = make_float3( NAN, NAN, NAN );
        verts[1]        = make_float3( 100.0f, 0.0f, 0.5f );
        verts[2]        = make_float3( 100.0f, 100.0f, 0.5f );
        verts[3]        = make_float3( 0.f, 0.0f, 0.5f );
        verts[4]        = make_float3( NAN, NAN, NAN );
        verts[5]        = make_float3( 100.0f, 100.0f, 0.5f );
        verts[6]        = make_float3( 0.f, 0.0f, 0.5f );
        verts[7]        = make_float3( 100.0f, 0.0f, 0.5f );
        verts[8]        = make_float3( NAN, NAN, NAN );
        vBuffer->unmap();

        return vBuffer;
    }

    //------------------------------------------------------------------------------
    void createIndexedPrimitivesWithNalwertices()
    {
        const unsigned int NUM_PRIMITIVES = 3;

        Buffer vBuffer = createNalwertices();

        Buffer  normalsBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, NUM_PRIMITIVES );
        float3* normals       = static_cast<float3*>( normalsBuffer->map() );
        std::fill( &normals[0], &normals[NUM_PRIMITIVES], make_float3( 0.f, 0.f, -1.0f ) );
        normalsBuffer->unmap();

        Buffer normalIndicesBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, NUM_PRIMITIVES );
        int3*  normalIndices       = static_cast<int3*>( normalIndicesBuffer->map() );
        std::fill( &normalIndices[0], &normalIndices[NUM_PRIMITIVES], make_int3( 0 ) );
        normalIndicesBuffer->unmap();

        Buffer vIndexBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, NUM_PRIMITIVES );
        int3*  indices      = static_cast<int3*>( vIndexBuffer->map() );
        indices[0]          = make_int3( 0, 1, 2 );
        indices[1]          = make_int3( 3, 4, 5 );
        indices[2]          = make_int3( 6, 7, 8 );
        vIndexBuffer->unmap();

        createIndexedTriangleMeshGraph( vBuffer, vIndexBuffer, normalsBuffer, normalIndicesBuffer, NUM_PRIMITIVES );

        m_goldImagePrefix       = "indexedPrimitivesWithNalwertices";
        m_goldImageMaxNumErrors = MAX_GOLD_IMAGE_ERRORS;
    }

    //------------------------------------------------------------------------------
    void createPrimitivesWithNalwertices()
    {
        const unsigned int NUM_PRIMITIVES = 3;
        Buffer             vBuffer        = createNalwertices();

        Buffer  normalsBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 3 * NUM_PRIMITIVES );
        float3* normals       = static_cast<float3*>( normalsBuffer->map() );
        std::fill( &normals[0], &normals[3 * NUM_PRIMITIVES], make_float3( 0.f, 0.f, -1.0f ) );
        normalsBuffer->unmap();

        createTriangleMeshGraph( vBuffer, normalsBuffer, NUM_PRIMITIVES );

        m_goldImagePrefix       = "primitivesWithNalwertices";
        m_goldImageMaxNumErrors = MAX_GOLD_IMAGE_ERRORS;
    }

    //------------------------------------------------------------------------------
    void createDegenerateTriangle()
    {
        Buffer  vBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1 );
        float3* verts   = static_cast<float3*>( vBuffer->map() );
        verts[0]        = make_float3( 0.f );
        vBuffer->unmap();

        Buffer vIndexBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, 1 );
        int3*  indices      = static_cast<int3*>( vIndexBuffer->map() );
        indices[0]          = make_int3( 0 );
        vIndexBuffer->unmap();

        createIndexedTriangleMeshGraph( vBuffer, vIndexBuffer, 1 );
    }

    //------------------------------------------------------------------------------
    void createIndexedTriangleMeshGraph( Buffer vBuffer, Buffer vIndexBuffer, unsigned int primitiveCount )
    {
        createIndexedTriangleMeshGraph( vBuffer, vIndexBuffer, vBuffer, vIndexBuffer, primitiveCount, 0 );
    }

    //------------------------------------------------------------------------------
    Geometry createMeshGeometry( Buffer vBuffer, Buffer nBuffer, unsigned primitiveCount, unsigned primitiveOffset, bool indexed )
    {
        Geometry mesh = m_context->createGeometry();

        mesh["vertex_buffer"]->setBuffer( vBuffer );
        mesh["normal_buffer"]->setBuffer( nBuffer );

        mesh->setPrimitiveCount( primitiveCount );
        const std::string triangleMeshPtxPath = ptxPath( m_target, "triangle_mesh.lw" );
        optix::Program    intersect =
            m_context->createProgramFromPTXFile( triangleMeshPtxPath,
                                                 indexed ? "mesh_intersect" : "mesh_intersect_list" );
        optix::Program bbox =
            m_context->createProgramFromPTXFile( triangleMeshPtxPath, indexed ? "mesh_bounds" : "mesh_bounds_list" );
        mesh->setIntersectionProgram( intersect );
        mesh->setBoundingBoxProgram( bbox );

        unsigned int primCount = primitiveCount + primitiveOffset;
        RTsize       vertexCount;
        vBuffer->getSize( vertexCount );
        Buffer tBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, indexed ? vertexCount : 3 * primCount );
        mesh["texcoord_buffer"]->setBuffer( tBuffer );

        Buffer mIndexBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, primCount );
        uint*  matIndices   = static_cast<uint*>( mIndexBuffer->map() );
        std::fill( &matIndices[0], &matIndices[primCount], 0 );
        mIndexBuffer->unmap();
        mesh["material_buffer"]->setBuffer( mIndexBuffer );

        return mesh;
    }

    //------------------------------------------------------------------------------
    Geometry createMeshGeometryGraph( Buffer vBuffer, Buffer nBuffer, unsigned int primitiveCount, unsigned int primitiveOffset, bool indexed )
    {
        m_topGeometryGroup = m_context->createGeometryGroup();

        Geometry mesh = createMeshGeometry( vBuffer, nBuffer, primitiveCount, primitiveOffset, indexed );

        GeometryInstance gi = m_context->createGeometryInstance( mesh, &m_material, &m_material + 1 );

        m_topGeometryGroup->setChildCount( 1 );
        m_topGeometryGroup->setChild( 0, gi );

        m_sceneBounds.include( make_float3( 0.f ) );
        m_sceneBounds.include( make_float3( 1.f ) );
        m_context["top_object"]->set( m_topGeometryGroup );

        return mesh;
    }

    //------------------------------------------------------------------------------
    void createIndexedTriangleMeshGraph( Buffer       vBuffer,
                                         Buffer       vIndexBuffer,
                                         Buffer       nBuffer,
                                         Buffer       nIndexBuffer,
                                         unsigned int primitiveCount,
                                         unsigned int primitiveOffset )
    {
        Geometry mesh = createMeshGeometryGraph( vBuffer, nBuffer, primitiveCount, primitiveOffset, true );
        mesh["vindex_buffer"]->setBuffer( vIndexBuffer );
        mesh["nindex_buffer"]->setBuffer( nIndexBuffer );
        mesh["tindex_buffer"]->setBuffer( vIndexBuffer );
        mesh->setPrimitiveIndexOffset( primitiveOffset );
    }

    void createIndexedTriangleMeshGraph( Buffer vBuffer, Buffer vIndexBuffer, Buffer nBuffer, Buffer nIndexBuffer, unsigned int primitiveCount )
    {
        createIndexedTriangleMeshGraph( vBuffer, vIndexBuffer, nBuffer, nIndexBuffer, primitiveCount, 0 );
    }


    //------------------------------------------------------------------------------
    void createTriangleMeshGraph( Buffer vBuffer, Buffer nBuffer, unsigned int primitiveCount )
    {
        createMeshGeometryGraph( vBuffer, nBuffer, primitiveCount, 0, false );
    }

    void createTriangleMeshGraph( Buffer vBuffer, Buffer nBuffer, unsigned int primitiveCount, unsigned int primitiveOffset )
    {
        Geometry mesh = createMeshGeometryGraph( vBuffer, nBuffer, primitiveCount, primitiveOffset, false );
        mesh->setPrimitiveIndexOffset( primitiveOffset );
    }

    //------------------------------------------------------------------------------
    void createSharedAsTopology()
    {
        const std::string normalShaderPtx = ptxPath( m_target, "normal_shader.lw" );
        m_material->setClosestHitProgram( 0u, m_context->createProgramFromPTXFile( normalShaderPtx,
                                                                                   "closest_hit_radiance_offset" ) );

        // Gr ->
        //      T -> Gr ----> GG -> GI -+
        //           |        |         |
        //           |-> AS   |-> AS    |-> Ge
        //           |        |         |
        //      T -> Gr ----> GG -> GI -+
        m_topGroup = m_context->createGroup();
        m_topGroup->setChildCount( 2 );
        Acceleration as1 = m_context->createAcceleration( m_accelType, m_builderType );
        Acceleration as2 = m_context->createAcceleration( m_accelType, m_builderType );

        m_indexedPrimitives = false;
        Buffer  vertsBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 3 );
        float3* verts       = static_cast<float3*>( vertsBuffer->map() );
        verts[0]            = make_float3( 0.f, 0.0f, 0.5f );
        verts[1]            = make_float3( 0.5f, 0.0f, 0.5f );
        verts[2]            = make_float3( 0.5f, 0.5f, 0.5f );
        vertsBuffer->unmap();

        Buffer  normalsBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 3 );
        float3* normals       = static_cast<float3*>( normalsBuffer->map() );
        std::fill( &normals[0], &normals[3], make_float3( 0.f, 0.f, -1.0f ) );
        normalsBuffer->unmap();

        Geometry geom = createMeshGeometry( vertsBuffer, normalsBuffer, 1, 0, false );

        for( unsigned int i = 0; i < 2U; ++i )
        {
            Transform xform = m_context->createTransform();
            xform->setMatrix( false, Matrix4x4::translate( make_float3( 0.5f * i, 0.25f, 0.0f ) ).getData(), nullptr );
            m_topGroup->setChild( i, xform );

            Group group = m_context->createGroup();
            group->setChildCount( 1 );
            group->setAcceleration( as1 );
            xform->setChild( group );

            GeometryGroup geomGroup = m_context->createGeometryGroup();
            geomGroup->setChildCount( 1 );
            geomGroup->setAcceleration( as2 );
            group->setChild( 0, geomGroup );

            GeometryInstance gi = m_context->createGeometryInstance();
            gi->setMaterialCount( 1 );
            gi->setMaterial( 0, m_material );
            gi->setGeometry( geom );
            gi["shading_offset"]->setFloat( i == 0 ? make_float3( 0.0f, 0.0f, 1.0f ) : make_float3( 0.0f ) );
            geomGroup->setChild( 0, gi );
        }

        m_sceneBounds.include( make_float3( 0.f ) );
        m_sceneBounds.include( make_float3( 1.f ) );
        m_context["top_object"]->set( m_topGroup );

        m_goldImagePrefix       = "sharedAsTopology";
        m_goldImageMaxNumErrors = MAX_GOLD_IMAGE_ERRORS;
    }

    //------------------------------------------------------------------------------
    void setTopAcceleration()
    {
        if( m_topGeometryGroup )
            m_topGeometryGroup->setAcceleration( m_accel );
        if( m_topGroup )
            m_topGroup->setAcceleration( m_accel );
    }

    //------------------------------------------------------------------------------
    void recreateAccleration()
    {
        if( m_accel )
            m_accel->destroy();
        m_accel = m_context->createAcceleration( m_accelType, m_builderType );
        if( m_indexedPrimitives )
        {
            m_accel->setProperty( "vertex_buffer_name", "vertex_buffer" );
            m_accel->setProperty( "index_buffer_name", "vindex_buffer" );
        }
        m_accel->markDirty();

        setTopAcceleration();
    }

    //------------------------------------------------------------------------------
    void setupOrthoCamera( const Aabb& aabb, float margin = 0.05f )
    {
        float3 center = aabb.center();
        float3 extent = aabb.extent();
        float  aspect = float( m_width ) / m_height;
        float  v      = ( extent.y / 2 ) * ( 1 + margin );
        float  u      = v * aspect;
        if( extent.x / extent.y > aspect )
        {
            u = ( extent.x / 2 ) * ( 1 + margin );
            v = u / aspect;
        }

        center.y += 1e-8f;  // GOLDENROD: This epsilon is to keep Trbvh from failing. The cause of the failure is that rays end up in a splitting plane due to the way the plane locations are quantized.
        float3 U   = make_float3( u, 0, 0 );
        float3 V   = make_float3( 0, v, 0 );
        float3 W   = make_float3( 0, 0, 1 );
        float3 eye = center - make_float3( 0, 0, 2 * extent.z );

        // Declare camera variables
        m_context["eye"]->setFloat( eye );
        m_context["U"]->setFloat( U );
        m_context["V"]->setFloat( V );
        m_context["W"]->setFloat( W );
    }

    //------------------------------------------------------------------------------
    void loadGeometry()
    {
        switch( m_modelType )
        {
            case ModelType::COW:
                createCowModel();
                break;
            case ModelType::COWS_GROUP:
                createCowsGroupModel();
                break;
            case ModelType::SPHERES:
                createSpheresModel();
                break;
            case ModelType::SPHERES_GROUP:
                createSpheresGroupModel();
                break;
            case ModelType::SPHERES_GROUP_WITH_EMPTY_CHILD:
                createSpheresGroupModelWithEmptyChild();
                break;
            case ModelType::SPHERES_TRANSFORMED:
                createSpheresTransformedModel();
                break;
            case ModelType::EMPTY_SELECTOR:
                createEmptySelector();
                break;
            case ModelType::EMPTY_GEOMETRY_GROUP:
                createEmptyGeometryGroup();
                break;
            case ModelType::DEGENERATE_TRIANGLE:
                createDegenerateTriangle();
                break;
            case ModelType::ZERO_PRIMITIVES:
                createZeroPrimitives();
                break;
            case ModelType::MIXED_VALID_PRIMITIVES:
                createMixedValidPrimitives();
                break;
            case ModelType::SINGLE_PRIMITIVE:
                createSinglePrimitive();
                break;
            case ModelType::PRIMITIVES_WITH_OFFSET:
                createPrimitivesWithOffset();
                break;
            case ModelType::INDEXED_PRIMITIVES_WITH_OFFSET:
                createIndexedPrimitivesWithOffset();
                break;
            case ModelType::INDEXED_PRIMITIVES_WITH_NAN_VERTICES:
                createIndexedPrimitivesWithNalwertices();
                break;
            case ModelType::PRIMITIVES_WITH_NAN_VERTICES:
                createPrimitivesWithNalwertices();
                break;
            case ModelType::SHARED_AS_TOPOLOGY:
                createSharedAsTopology();
                break;
            default:
                FAIL();
        }

        recreateAccleration();

        setupOrthoCamera( m_sceneBounds );
    }


    //------------------------------------------------------------------------------
    void launch()
    {
        if( !m_topGeometryGroup && !m_topGroup )
            loadGeometry();
        m_context->launch( 0, m_width, m_height );
    }

    std::string referenceFileName() const
    {
        std::stringstream refFilename;
        refFilename << dataPath() + "/" + m_target + "/" << m_goldImagePrefix << m_width << "x" << m_height << ".ppm";
        return refFilename.str();
    }

    std::string outputFileName() const
    {
        const TestInfo*   testInfo     = UnitTest::GetInstance()->lwrrent_test_info();
        const std::string testCaseName = testInfo->test_case_name();
        std::string       fullTestName = testCaseName + '.' + testInfo->name();
        std::replace( std::begin( fullTestName ), std::end( fullTestName ), '/', '_' );
        std::stringstream outputFileName;
        outputFileName << m_target + "_" << fullTestName << '_' << m_goldImagePrefix << m_width << "x" << m_height << ".ppm";
        return outputFileName.str();
    }

    std::string renderMatchFailureMessage() const
    {
        return referenceFileName() + " doesn't match rendered output " + outputFileName();
    }

    //------------------------------------------------------------------------------
    bool compareImageToReference()
    {
        m_refImage.readPPM( referenceFileName() );

        float tolerance = 2.0f / 255;
        int   numErrors;
        float avgError, maxError;
        Image::compare( m_image, m_refImage, tolerance, numErrors, avgError, maxError );
        if( numErrors > m_goldImageMaxNumErrors )
        {
            //std::cout << "number of wrong pixels ( " << numErrors << " ) exceeded threshold ( " << m_goldImageMaxNumErrors << " )\n";
            return false;
        }
        return true;
    }

    //------------------------------------------------------------------------------
    bool renderMatchesReference()
    {
        launch();
        m_image.init( m_outputBuffer );
        m_image.writePPM( outputFileName(), false );
        return compareImageToReference();
    }

    //------------------------------------------------------------------------------
    void destroyContext()
    {
        if( m_outputBuffer )
            m_outputBuffer->destroy();
        if( m_accel )
            m_accel->destroy();
        if( m_material )
            m_material->destroy();
        if( m_context )
            m_context->destroy();
    }

    //------------------------------------------------------------------------------
    void readOldAccelCache( std::vector<char>& data )
    {
        std::stringstream cacheFilename;
        cacheFilename << dataPath() + "/" + m_target + "/"
                      << "oldBldVersion.cache";
        FILE* input = fopen( cacheFilename.str().c_str(), "rb" );
        fseek( input, 0, SEEK_END );  // compute file size
        unsigned bytesToRead = ftell( input );
        fseek( input, 0, SEEK_SET );  // reset to beginning of file
        data.resize( bytesToRead );
        if( fread( &data[0], sizeof( char ), bytesToRead, input ) != bytesToRead )
            FAIL();
        fclose( input );
    }

    bool skipAccelType( const char* accelType )
    {
        if( accelType == m_accelType )
        {
            std::cerr << "WARNING: Skipping for accelType = " << m_accelType << '\n';
            return true;
        }
        return false;
    }

    // Parameters
    unsigned    m_width       = 320;
    unsigned    m_height      = 240;
    std::string m_accelType   = "Trbvh";
    std::string m_builderType = "Bvh";
    enum class ModelType
    {
        COW,
        COWS_GROUP,
        SPHERES,
        SPHERES_GROUP,
        SPHERES_GROUP_WITH_EMPTY_CHILD,
        SPHERES_TRANSFORMED,
        EMPTY_SELECTOR,
        DEGENERATE_TRIANGLE,
        ZERO_PRIMITIVES,
        MIXED_VALID_PRIMITIVES,
        SINGLE_PRIMITIVE,
        INDEXED_PRIMITIVES_WITH_OFFSET,
        INDEXED_PRIMITIVES_WITH_NAN_VERTICES,
        PRIMITIVES_WITH_NAN_VERTICES,
        PRIMITIVES_WITH_OFFSET,
        SHARED_AS_TOPOLOGY,
        EMPTY_GEOMETRY_GROUP
    };
    ModelType   m_modelType           = ModelType::COW;
    bool        m_loadGeometryInSetup = true;
    std::string m_boundsType          = "default";

    Context          m_context;
    Buffer           m_outputBuffer;
    Acceleration     m_accel;
    bool             m_indexedPrimitives = true;
    Material         m_material;
    GeometryGroup    m_topGeometryGroup;
    Group            m_topGroup;
    Image            m_refImage;
    Image            m_image;
    optix::Aabb      m_sceneBounds;
    std::string      m_goldImagePrefix;
    int              m_goldImageMaxNumErrors = 0;
    static const int m_groupSize             = 3;
    std::string      m_target                = "test_Accels";
};


class Trbvh : public SimpleOptix
{
};

}  // namespace

//------------------------------------------------------------------------------
TEST_F( Trbvh, CanBuildCpuInGpuContext )
{
    m_accel->setProperty( "build_type", "CPU" );
    m_accel->markDirty();

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

//------------------------------------------------------------------------------
TEST_F( Trbvh, CanBuildGpuInGpuContext )
{
    m_accel->setProperty( "build_type", "GPU" );
    m_accel->markDirty();

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

//------------------------------------------------------------------------------
TEST_F( Trbvh, DISABLED_BuildGpuThrowsOnOutOfMemory )
{
    launch();
    recreateAccleration();  // free up device memory
    m_accel->setProperty( "build_type", "GPU" );
    m_accel->markDirty();
    LwdaMemHog allocateMostOfMemory( 0 );

    ASSERT_ANY_THROW( launch() );
}

//------------------------------------------------------------------------------
TEST_F( Trbvh, DISABLED_CanBuildWithCpuFallbackWhenGpuOutOfMemory )
{
    launch();
    recreateAccleration();  // free up device memory
    m_accel->setProperty( "build_type", "GPU/CPU" );
    m_accel->markDirty();
    LwdaMemHog allocateMostOfMemory( 0 );

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

//------------------------------------------------------------------------------
TEST_F( Trbvh, CanSerializeAndDeserialize )
{
    m_context->launch( 0, 0 );
    std::vector<char> data( m_accel->getDataSize() );
    m_accel->getData( &data[0] );
    m_accel->setData( &data[0], data.size() );

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

//------------------------------------------------------------------------------
TEST_F( Trbvh, DISABLED_CanSerializeAndDeserializeWithFreshBuilder )
{
    m_context->launch( 0, 0 );
    std::vector<char> data( m_accel->getDataSize() );
    m_accel->getData( &data[0] );
    m_accel->setBuilder( "Bvh" );               // swap out builder
    m_accel->setData( &data[0], data.size() );  // creates new builder

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

//------------------------------------------------------------------------------
TEST_F( Trbvh, GetDataThrowsWhenDirty )
{
    m_context->launch( 0, 0 );
    std::vector<char> data( m_accel->getDataSize() );
    m_accel->markDirty();

    ASSERT_ANY_THROW( m_accel->getData( &data[0] ) );
}

//------------------------------------------------------------------------------
TEST_F( Trbvh, GetDataSizeThrowsWhenDirty )
{
    m_accel->markDirty();

    ASSERT_ANY_THROW( m_accel->getDataSize() );
}

//------------------------------------------------------------------------------
TEST_F( Trbvh, SetDataThrowsWithOldBuilder )
{
    std::vector<char> oldBuilderData;
    readOldAccelCache( oldBuilderData );

    ASSERT_ANY_THROW( m_accel->setData( &oldBuilderData[0], oldBuilderData.size() ) );
}

//------------------------------------------------------------------------------
TEST_F( Trbvh, SetDataMakesNonDirty )
{
    m_context->launch( 0, 0 );
    std::vector<char> data( m_accel->getDataSize() );
    m_accel->getData( &data[0] );
    m_accel->markDirty();

    m_accel->setData( &data[0], data.size() );

    ASSERT_FALSE( m_accel->isDirty() );
}

//------------------------------------------------------------------------------
TEST_F( Trbvh, SetDataSetsBuilder )
{
    m_context->launch( 0, 0 );
    std::string       bldName = m_accel->getBuilder();
    std::vector<char> data( m_accel->getDataSize() );
    m_accel->getData( &data[0] );
    m_accel->setData( &data[0], data.size() );

    ASSERT_EQ( bldName, m_accel->getBuilder() );
}

//------------------------------------------------------------------------------
TEST_F( Trbvh, SetDataFailingKeepsNonDirty )
{
    m_context->launch( 0, 0 );
    char bogusData = 0;
    EXPECT_ANY_THROW( m_accel->setData( &bogusData, 1 ) );

    ASSERT_FALSE( m_accel->isDirty() );
}

//------------------------------------------------------------------------------
TEST_F( Trbvh, SetDataFailingKeepsDirty )
{
    m_accel->markDirty();
    char bogusData = 0;
    EXPECT_ANY_THROW( m_accel->setData( &bogusData, 1 ) );

    ASSERT_TRUE( m_accel->isDirty() );
}

//------------------------------------------------------------------------------
TEST_F( Trbvh, CpuBuildWorksWithRefit )
{
    m_modelType = ModelType::SPHERES;
    loadGeometry();
    m_accel->setProperty( "build_type", "CPU" );
    m_accel->setProperty( "refit", "1" );
    launch();
    m_accel->markDirty();

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

///////////////////////////////////////////////////////////////////////////////
//
// Test other accels
//
///////////////////////////////////////////////////////////////////////////////

class Accel : public SimpleOptix, public WithParamInterface<std::string>
{
  public:
    Accel()
    {
        m_accelType           = GetParam();
        m_loadGeometryInSetup = false;
    }
};

TEST_P( Accel, WorksWithTriangleInputs )
{
    m_modelType = ModelType::COW;

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithAabbInputs )
{
    m_modelType = ModelType::SPHERES;

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithInfAabbInputs_rtx )
{
    if( skipAccelType( "Bvh" ) || skipAccelType( "Sbvh" ) || skipAccelType( "Bvh8" ) || skipAccelType( "Trbvh" ) )
        return;
    m_modelType  = ModelType::SPHERES;
    m_boundsType = "inf";
    // In contrast to exelwtion strategy megakernel, using exelwtion strategy rtx a top level tree may be built.
    // This means, due to computations on "inf" input values when combining the aabbs,
    // an empty aabb at the root is created, matching the "miss" reference image instead of the "spheres" one.
    // (Since the result image depends on the GPU type, TTU or non-TTU, and whether watertight traversal is used, we accept both results.)
    loadGeometry();

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, FailsWithIlwalidAabbInputs )
{
    m_modelType  = ModelType::SPHERES;
    m_boundsType = "invalid";

    ASSERT_FALSE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, FailsWithNanAabbInputs )
{
    // OP-2080
    if( skipAccelType( "Trbvh" ) )
        return;
    m_modelType  = ModelType::SPHERES;
    m_boundsType = "nan";
    // Traversal should ignore NaN aabbs, so change the gold image
    // after loading geometry
    loadGeometry();
    m_goldImagePrefix = "miss";

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithGroupNodes )
{
    m_modelType = ModelType::SPHERES_GROUP;

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithGroupNodesWithSomeEmptyChildren )
{
    m_modelType = ModelType::SPHERES_GROUP_WITH_EMPTY_CHILD;

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithTriGroupNodes )
{
    m_modelType = ModelType::COWS_GROUP;

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithEmptySelector )
{
    m_modelType = ModelType::EMPTY_SELECTOR;
    launch();
}

TEST_P( Accel, WorksWithEmptyGeometryGroups )
{
    m_modelType = ModelType::EMPTY_GEOMETRY_GROUP;
    launch();
}

TEST_P( Accel, WorksWithDegenerateTriangle )
{
    m_modelType = ModelType::DEGENERATE_TRIANGLE;
    launch();
}

TEST_P( Accel, DISABLED_WorksWithNoTexMem )
{
    loadGeometry();
    // TODO: rewrite this test to use memorymanager.maxTexHeapSize to turn off the texheap.
    m_accel->setProperty( "use_texmem", "0" );

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithRefit )
{
    m_modelType = ModelType::SPHERES;
    loadGeometry();
    m_accel->setProperty( "refit", "1" );
    launch();
    m_accel->markDirty();

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithRefine )
{
    m_modelType = ModelType::SPHERES;
    loadGeometry();
    m_accel->setProperty( "refine", "1" );
    launch();
    m_accel->markDirty();

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithTransformedInstances )
{
    m_modelType = ModelType::SPHERES_TRANSFORMED;

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithDynamicTransformedInstances )
{
    m_modelType = ModelType::SPHERES_TRANSFORMED;
    loadGeometry();
    // remove some spheres from the scene
    const int              n = m_topGroup->getChildCount();
    std::vector<Transform> transforms;
    for( int i = 1; i < n; ++i )
        transforms.push_back( m_topGroup->getChild<Transform>( i ) );
    m_topGroup->setChildCount( 1 );
    // build accels over scene without all the spheres
    launch();
    // add spheres back
    m_topGroup->setChildCount( n );
    for( size_t i = 0; i < transforms.size(); ++i )
    {
        m_topGroup->setChild( i + 1, transforms[i] );
    }
    // Important: only need to rebuild the top-level accel, not the shared bottom level accel!
    m_topGroup->getAcceleration()->markDirty();

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithDegenerateBounds )
{
    m_modelType  = ModelType::SPHERES;
    m_boundsType = "degenerate";
    loadGeometry();
    m_goldImagePrefix.append( "Degenerate" );

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithZeroPrimitives )
{
    m_modelType = ModelType::ZERO_PRIMITIVES;
    loadGeometry();

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithIlwalidPrimitivesMixedWithValidPrimitives )
{
    m_modelType = ModelType::MIXED_VALID_PRIMITIVES;
    loadGeometry();

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithSinglePrimitive )
{
    m_modelType = ModelType::SINGLE_PRIMITIVE;
    loadGeometry();

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithIndexedPrimitiveOffset )
{
    m_modelType = ModelType::INDEXED_PRIMITIVES_WITH_OFFSET;
    loadGeometry();

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithPrimitiveOffset )
{
    m_indexedPrimitives = false;
    m_modelType         = ModelType::PRIMITIVES_WITH_OFFSET;
    loadGeometry();

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithIndexedPrimitiveWithNalwertices )
{
    if( skipAccelType( "Bvh8" ) )
        return;
#ifndef NDEBUG
    // Sbvh uses Aabb methods that assert() that the Aabb is valid
    if( skipAccelType( "Sbvh" ) )
        return;
#endif
    m_modelType = ModelType::INDEXED_PRIMITIVES_WITH_NAN_VERTICES;
    loadGeometry();

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithPrimitiveWithNalwertices )
{
    if( skipAccelType( "Bvh8" ) )
        return;
    m_indexedPrimitives = false;
    m_modelType         = ModelType::PRIMITIVES_WITH_NAN_VERTICES;
    loadGeometry();

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

TEST_P( Accel, WorksWithSharedAsTopology )
{
    // Disabled for rtx, the rendered image doesn't show different material colors, see bug 2485684.
    m_modelType = ModelType::SHARED_AS_TOPOLOGY;
    loadGeometry();

    ASSERT_TRUE( renderMatchesReference() ) << renderMatchFailureMessage();
}

static const std::string g_accelTypes[] = {"Bvh", "Sbvh", "Trbvh", "Bvh8"};
INSTANTIATE_TEST_SUITE_P( AllBvh, Accel, ValuesIn( g_accelTypes ) );
