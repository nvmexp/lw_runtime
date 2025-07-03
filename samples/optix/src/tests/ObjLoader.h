
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

#pragma once

#include <optixu/optixpp_namespace.h>

#include <optix_world.h>
#include <string>
#include <tests/glm.h>


#ifndef SRCTESTSAPI
#define SRCTESTSAPI
#endif

//-----------------------------------------------------------------------------
//
//  ObjLoader class declaration
//
//-----------------------------------------------------------------------------

class ObjLoader
{
  public:
    SRCTESTSAPI ObjLoader( const std::string&   filename,
                           optix::Context       context,
                           optix::GeometryGroup geometrygroup,
                           optix::Program       intersect_program,
                           optix::Program       bbox_program,
                           optix::Material      material,
                           const std::string&   builder );

    SRCTESTSAPI ~ObjLoader() {}  // makes sure CRT objects are destroyed on the correct heap

    SRCTESTSAPI void load();
    SRCTESTSAPI void load( const optix::Matrix4x4& transform );

    SRCTESTSAPI optix::Aabb getSceneBBox() const { return _aabb; }

    SRCTESTSAPI static bool isMyFile( const std::string& filename );

  private:
    void createGeometryInstances( GLMmodel* model, optix::Program mesh_intersect, optix::Program mesh_bbox );
    void loadVertexData( GLMmodel* model, const optix::Matrix4x4& transform );


    std::string          _pathname;
    std::string          _filename;
    optix::Context       _context;
    optix::GeometryGroup _geometrygroup;
    std::string          _builder;
    optix::Buffer        _vbuffer;
    optix::Buffer        _nbuffer;
    optix::Buffer        _tbuffer;
    optix::Material      _material;
    optix::Program       _intersect_program;
    optix::Program       _bbox_program;
    optix::Aabb          _aabb;
};
