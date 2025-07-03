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

#pragma once

#include <internal/optix_declarations.h>

#include <prodlib/misc/GLFunctions.h>

#include <stdlib.h>


namespace optix {

struct GfxInteropResource
{
    enum ResourceKind
    {
        NONE,
        OGL_TEXTURE,        // gl texture
        OGL_BUFFER_OBJECT,  // gl buffer object (pbo, vbo)
        OGL_RENDER_BUFFER,  // gl renderbuffer
    };

    ResourceKind kind;
    union
    {
        struct
        {
            GLuint     glId;
            RTgltarget target;
        } gl;
    };

    GfxInteropResource();
    GfxInteropResource( ResourceKind kind, GLuint glId, RTgltarget target );  // texture or render buffer
    GfxInteropResource( ResourceKind kind, GLuint glId );                     // buffer objects

    bool isOGL() const;
    bool isArray() const;

    struct Properties
    {
        RTformat     format         = RTformat( 0 );
        unsigned int dimensionality = 0;
        size_t       width          = 0;
        size_t       height         = 0;
        size_t       depth          = 0;
        unsigned int levelCount     = 0;
        size_t       elementSize    = 0;
        bool         lwbe           = false;
        bool         layered        = false;
        bool         srgb           = false;
        float        anisotropy     = 1.0f;
        float        baseLevel      = 0.0f;
        float        minLevel       = 0.0f;
        float        maxLevel       = 100.0f;  //default value for OpenGL

        // GL specific
        GLint   glInternalFormat  = 0;        // textures and renderbuffers only
        GLenum  glBufferUsage     = 0;        // buffer objects only
        GLuint* glRenderbufferFBO = nullptr;  // in/out, set if you want to cache the FBO used to query a renderbuffer
    };

    // Copy mode
    enum CopyDirection
    {
        ToResource,
        FromResource
    };

    // Utility functions
    void copyToOrFromGLResource( CopyDirection direction, void* ptr, size_t bufferSize, unsigned int* cachedFB );
    void queryProperties( Properties* props );

  private:
    void createInteropFBO( GLuint* fbo );
};
}
