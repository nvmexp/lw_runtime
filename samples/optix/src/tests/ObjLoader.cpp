
/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <tests/Image.h>
#include <tests/ObjLoader.h>

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace optix;


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

namespace {
std::string getExtension( const std::string& filename )
{
    // Get the filename extension
    std::string::size_type extension_index = filename.find_last_of( "." );
    return extension_index != std::string::npos ? filename.substr( extension_index + 1 ) : std::string();
}
}


//------------------------------------------------------------------------------
//
//  ObjLoader class definition
//
//------------------------------------------------------------------------------

ObjLoader::ObjLoader( const std::string& filename,
                      Context            context,
                      GeometryGroup      geometrygroup,
                      Program            intersect_program,
                      Program            bbox_program,
                      Material           material,
                      const std::string& builder )
    : _filename( filename )
    , _context( context )
    , _geometrygroup( geometrygroup )
    , _builder( builder )
    , _vbuffer( nullptr )
    , _nbuffer( nullptr )
    , _tbuffer( nullptr )
    , _material( material )
    , _intersect_program( intersect_program )
    , _bbox_program( bbox_program )
    , _aabb()
{
    _pathname = _filename.substr( 0, _filename.find_last_of( "/\\" ) + 1 );
}

void ObjLoader::load()
{
    load( optix::Matrix4x4::identity() );
}

void ObjLoader::load( const optix::Matrix4x4& transform )
{
    // parse the OBJ file
    GLMmodel* model = glmReadOBJ( _filename.c_str() );
    if( !model )
    {
        std::stringstream ss;
        ss << "ObjLoader::loadImpl - glmReadOBJ( '" << _filename << "' ) failed" << std::endl;
        throw Exception( ss.str() );
    }

    // Create vertex data buffers to be shared by all Geometries
    loadVertexData( model, transform );

    // Create a GeometryInstance and Geometry for each obj group
    createGeometryInstances( model, _intersect_program, _bbox_program );

    glmDelete( model );
}

void ObjLoader::loadVertexData( GLMmodel* model, const optix::Matrix4x4& transform )
{
    unsigned int num_vertices  = model->numvertices;
    unsigned int num_texcoords = model->numtexcoords;
    unsigned int num_normals   = model->numnormals;

    // Create vertex buffer
    _vbuffer             = _context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_vertices );
    float3* vbuffer_data = static_cast<float3*>( _vbuffer->map() );

    // Create normal buffer
    _nbuffer             = _context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_normals );
    float3* nbuffer_data = static_cast<float3*>( _nbuffer->map() );

    // Create texcoord buffer
    _tbuffer             = _context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, num_texcoords );
    float2* tbuffer_data = static_cast<float2*>( _tbuffer->map() );

    // Transform and copy vertices.
    for( unsigned int i = 0; i < num_vertices; ++i )
    {
        const float3 v3 = *( (float3*)&model->vertices[( i + 1 ) * 3] );
        float4       v4 = make_float4( v3, 1.0f );
        vbuffer_data[i] = make_float3( transform * v4 );
    }

    // Transform and copy normals.
    const optix::Matrix4x4 norm_transform = transform.ilwerse().transpose();
    for( unsigned int i = 0; i < num_normals; ++i )
    {
        const float3 v3 = *( (float3*)&model->normals[( i + 1 ) * 3] );
        float4       v4 = make_float4( v3, 0.0f );
        nbuffer_data[i] = make_float3( norm_transform * v4 );
    }

    // Copy texture coordinates.
    memcpy( static_cast<void*>( tbuffer_data ), static_cast<void*>( &( model->texcoords[2] ) ), sizeof( float ) * num_texcoords * 2 );

    // Callwlate bbox of model
    for( unsigned int i = 0; i < num_vertices; ++i )
        _aabb.include( vbuffer_data[i] );

    // Unmap buffers.
    _vbuffer->unmap();
    _nbuffer->unmap();
    _tbuffer->unmap();
}


void ObjLoader::createGeometryInstances( GLMmodel* model, Program mesh_intersect, Program mesh_bbox )
{
    std::vector<GeometryInstance> instances;

    // Loop over all groups -- grab the triangles and material props from each group
    unsigned int triangle_count = 0u;
    unsigned int group_count    = 0u;
    for( GLMgroup *obj_group = model->groups; obj_group != nullptr; obj_group = obj_group->next, group_count++ )
    {

        unsigned int num_triangles = obj_group->numtriangles;
        if( num_triangles == 0 )
            continue;

        // Create vertex index buffers
        Buffer vindex_buffer      = _context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, num_triangles );
        int3*  vindex_buffer_data = static_cast<int3*>( vindex_buffer->map() );

        Buffer tindex_buffer      = _context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, num_triangles );
        int3*  tindex_buffer_data = static_cast<int3*>( tindex_buffer->map() );

        Buffer nindex_buffer      = _context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, num_triangles );
        int3*  nindex_buffer_data = static_cast<int3*>( nindex_buffer->map() );

        // TODO: Create empty buffer for mat indices, have obj_material check for zero length
        Buffer mbuffer      = _context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, num_triangles );
        uint*  mbuffer_data = static_cast<uint*>( mbuffer->map() );

        // Create the mesh object
        Geometry mesh = _context->createGeometry();
        mesh->setPrimitiveCount( num_triangles );
        mesh->setIntersectionProgram( mesh_intersect );
        mesh->setBoundingBoxProgram( mesh_bbox );
        mesh["vertex_buffer"]->setBuffer( _vbuffer );
        mesh["normal_buffer"]->setBuffer( _nbuffer );
        mesh["texcoord_buffer"]->setBuffer( _tbuffer );
        mesh["vindex_buffer"]->setBuffer( vindex_buffer );
        mesh["tindex_buffer"]->setBuffer( tindex_buffer );
        mesh["nindex_buffer"]->setBuffer( nindex_buffer );
        mesh["material_buffer"]->setBuffer( mbuffer );

        // Create the geom instance to hold mesh and material params
        GeometryInstance instance = _context->createGeometryInstance( mesh, &_material, &_material + 1 );
        instances.push_back( instance );

        for( unsigned int i = 0; i < obj_group->numtriangles; ++i, ++triangle_count )
        {

            unsigned int tindex = obj_group->triangles[i];
            int3         vindices;
            vindices.x = model->triangles[tindex].vindices[0] - 1;
            vindices.y = model->triangles[tindex].vindices[1] - 1;
            vindices.z = model->triangles[tindex].vindices[2] - 1;
            assert( vindices.x <= static_cast<int>( model->numvertices ) );
            assert( vindices.y <= static_cast<int>( model->numvertices ) );
            assert( vindices.z <= static_cast<int>( model->numvertices ) );

            int3 nindices;
            nindices.x = model->triangles[tindex].nindices[0] - 1;
            nindices.y = model->triangles[tindex].nindices[1] - 1;
            nindices.z = model->triangles[tindex].nindices[2] - 1;
            assert( nindices.x <= static_cast<int>( model->numnormals ) );
            assert( nindices.y <= static_cast<int>( model->numnormals ) );
            assert( nindices.z <= static_cast<int>( model->numnormals ) );

            int3 tindices;
            tindices.x = model->triangles[tindex].tindices[0] - 1;
            tindices.y = model->triangles[tindex].tindices[1] - 1;
            tindices.z = model->triangles[tindex].tindices[2] - 1;
            assert( tindices.x <= static_cast<int>( model->numtexcoords ) );
            assert( tindices.y <= static_cast<int>( model->numtexcoords ) );
            assert( tindices.z <= static_cast<int>( model->numtexcoords ) );

            vindex_buffer_data[i] = vindices;
            nindex_buffer_data[i] = nindices;
            tindex_buffer_data[i] = tindices;
            mbuffer_data[i]       = 0;  // See above TODO
        }

        vindex_buffer->unmap();
        tindex_buffer->unmap();
        nindex_buffer->unmap();
        mbuffer->unmap();
    }

    assert( triangle_count == model->numtriangles );

    // Set up group
    _geometrygroup->setChildCount( static_cast<unsigned int>( instances.size() ) );
    Acceleration acceleration = _context->createAcceleration( _builder.c_str(), "Bvh" );
    acceleration->setProperty( "vertex_buffer_name", "vertex_buffer" );
    acceleration->setProperty( "index_buffer_name", "vindex_buffer" );
    _geometrygroup->setAcceleration( acceleration );
    acceleration->markDirty();


    for( unsigned int i = 0; i < instances.size(); ++i )
        _geometrygroup->setChild( i, instances[i] );
}


bool ObjLoader::isMyFile( const std::string& filename )
{
    return getExtension( filename ) == "obj";
}
