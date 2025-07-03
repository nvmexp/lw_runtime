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

#include <Memory/GfxInteropResource.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/misc/GLFunctions.h>
#include <prodlib/misc/GLInternalFormats.h>
#include <prodlib/misc/RTFormatUtil.h>

#include <cstring>

using namespace optix;
using namespace prodlib;
using namespace corelib;

GfxInteropResource::GfxInteropResource()
    : kind( NONE )
{
}

GfxInteropResource::GfxInteropResource( ResourceKind kind, GLuint glId, RTgltarget target )
    : kind( kind )
{
    RT_ASSERT( kind == OGL_TEXTURE || kind == OGL_RENDER_BUFFER );
    gl.glId   = glId;
    gl.target = target;
}

GfxInteropResource::GfxInteropResource( ResourceKind kind, GLuint glId )
    : kind( kind )
{
    RT_ASSERT( kind == OGL_BUFFER_OBJECT );
    gl.glId   = glId;
    gl.target = (RTgltarget)~0;
}

bool GfxInteropResource::isOGL() const
{
    return kind == OGL_TEXTURE || kind == OGL_BUFFER_OBJECT || kind == OGL_RENDER_BUFFER;
}

bool GfxInteropResource::isArray() const
{
    return kind == OGL_TEXTURE || kind == OGL_RENDER_BUFFER;
}

void GfxInteropResource::copyToOrFromGLResource( CopyDirection direction, void* ptr, size_t bufferSize, unsigned int* cachedFBO )
{
    GL::checkGLError();
    RT_ASSERT( ptr );
    RT_ASSERT( direction == FromResource || kind == GfxInteropResource::OGL_BUFFER_OBJECT );

    if( kind == GfxInteropResource::OGL_RENDER_BUFFER )
    {
        if( GL::IsRenderbuffer( gl.glId ) != GL_TRUE )
            throw IlwalidValue( RT_EXCEPTION_INFO, "OpenGL ID does not belong to a renderbuffer.", gl.glId );

        createInteropFBO( cachedFBO );

        // what is the name of the current framebuffer
        GLint lwrrentFramebufferName;
        GL::GetIntegerv( GL_FRAMEBUFFER_BINDING, &lwrrentFramebufferName );

        // what is the name of the current renderbuffer
        GLint lwrrentRenderbufferName;
        GL::GetIntegerv( GL_RENDERBUFFER_BINDING, &lwrrentRenderbufferName );

        // setup our framebuffer and attach the users renderbuffer that is associated with this object
        GL::BindFramebuffer( GL_FRAMEBUFFER, *cachedFBO );
        GL::BindRenderbuffer( GL_RENDERBUFFER, gl.glId );
        GL::FramebufferRenderbuffer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, gl.glId );

        // check if our framebuffer is complete
        RT_ASSERT( GL::CheckFramebufferStatus( GL_FRAMEBUFFER ) == GL_FRAMEBUFFER_COMPLETE_EXT );

        // setup host data and memory
        GLint rbWidth, rbHeight, internal_format, depth_bits;
        GL::GetRenderbufferParameteriv( GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &rbWidth );
        GL::GetRenderbufferParameteriv( GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &rbHeight );
        GL::GetRenderbufferParameteriv( GL_RENDERBUFFER, GL_RENDERBUFFER_INTERNAL_FORMAT, &internal_format );
        GL::GetRenderbufferParameteriv( GL_RENDERBUFFER, GL_RENDERBUFFER_DEPTH_SIZE, &depth_bits );

        GLint        type, format;
        unsigned int elementSize;

        if( depth_bits == 0 )
        {
            type                 = getGLType( internal_format );
            format               = getGLFormat( internal_format );
            RTformat optixFormat = glToOptixFormat( internal_format );
            if( !isSupportedTextureFormat( optixFormat ) )
                throw IlwalidValue( RT_EXCEPTION_INFO, "Unsupported texture format for LWCA array: ", format );
            elementSize = getElementSize( optixFormat );
        }
        else
        {
            type                 = getDepthType( internal_format );
            format               = GL_DEPTH_COMPONENT;
            RTformat optixFormat = glDepthToOptixFormat( internal_format );
            if( !isSupportedTextureFormat( optixFormat ) )
                throw IlwalidValue( RT_EXCEPTION_INFO, "Unsupported texture format for LWCA array: ", format );
            elementSize = getElementSize( optixFormat );
        }

        const size_t size = rbWidth * rbHeight * elementSize;

        if( bufferSize != size )
        {
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "OpenGL buffer objects must be the same size as OptiX buffer objects" );
        }

        // read back the users data
        GL::ReadPixels( 0, 0, rbWidth, rbHeight, format, type, ptr );

        // restore user bindings
        GL::BindRenderbuffer( GL_RENDERBUFFER, lwrrentRenderbufferName );
        GL::BindFramebuffer( GL_FRAMEBUFFER, lwrrentFramebufferName );
    }
    else if( kind == GfxInteropResource::OGL_BUFFER_OBJECT )
    {

        GLint oldBinding;
        GL::GetIntegerv( GL_ARRAY_BUFFER_BINDING, &oldBinding );
        GL::BindBuffer( GL_ARRAY_BUFFER, gl.glId );

        GLint size  = 0;
        GLint usage = 0;
        GL::GetBufferParameteriv( GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint*)&size );
        GL::GetBufferParameteriv( GL_ARRAY_BUFFER, GL_BUFFER_USAGE, (GLint*)&usage );

        // check for correct opengl buffer size
        if( bufferSize != static_cast<size_t>( size ) )
        {
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "OpenGL buffer objects must be the same size as OptiX buffer objects" );
        }
        void* resourcePtr = GL::MapBuffer( GL_ARRAY_BUFFER, direction == FromResource ? GL_READ_ONLY : GL_WRITE_ONLY );
        if( resourcePtr == nullptr )
            throw IlwalidValue( RT_EXCEPTION_INFO, "GL buffer mapping failed." );

        if( direction == FromResource )
        {
            std::memcpy( ptr, resourcePtr, size );
        }
        else
        {
            std::memcpy( resourcePtr, ptr, size );
        }
        GL::UnmapBuffer( GL_ARRAY_BUFFER );
        GL::BindBuffer( GL_ARRAY_BUFFER, oldBinding );
    }
    else if( kind == GfxInteropResource::OGL_TEXTURE )
    {
        if( GL::IsTexture( gl.glId ) != GL_TRUE )
            throw IlwalidValue( RT_EXCEPTION_INFO, "OpenGL ID does not belong to a texture.", gl.glId );

        // Save old texture target and set the new one
        GLint oldTexId;
        GLint glTarget = getGLTextureTarget( gl.target );
        GL::GetIntegerv( getGLTextureBindingTarget( gl.target ), &oldTexId );
        GL::BindTexture( glTarget, gl.glId );

        size_t ptrOffset = 0;
        for( int level = 0;; level++ )
        {
            GLint glWidth, glHeight, glDepth;
            GL::GetTexLevelParameteriv( glTarget, level, GL_TEXTURE_WIDTH, &glWidth );
            GL::GetTexLevelParameteriv( glTarget, level, GL_TEXTURE_HEIGHT, &glHeight );
            GL::GetTexLevelParameteriv( glTarget, level, GL_TEXTURE_DEPTH, &glDepth );

            if( glWidth == 0 || glHeight == 0 || glDepth == 0 )
                break;  // the level == mip level count

            int glInternalFormat;
            GL::GetTexLevelParameteriv( glTarget, level, GL_TEXTURE_INTERNAL_FORMAT, &glInternalFormat );

            const RTformat optixFormat = glToOptixFormat( glInternalFormat );
            if( !isSupportedTextureFormat( optixFormat ) )
                throw IlwalidValue( RT_EXCEPTION_INFO, "Unsupported texture format for LWCA array: ", optixFormat );

            if( prodlib::isCompressed( optixFormat ) )
            {
                GLint glLevelSize = 0;
                GL::GetTexLevelParameteriv( glTarget, level, GL_TEXTURE_COMPRESSED_IMAGE_SIZE, &glLevelSize );
                if( ptrOffset + glLevelSize > bufferSize )
                {
                    throw IlwalidValue( RT_EXCEPTION_INFO,
                                        "OpenGL buffer objects must be the same size as OptiX buffer objects" );
                }

                GL::GetCompressedTexImage( glTarget, level, (char*)ptr + ptrOffset );
                ptrOffset += glLevelSize;
            }
            else
            {
                const GLint        type        = getGLType( glInternalFormat );
                const GLint        format      = getGLFormat( glInternalFormat );
                const unsigned int elementSize = getElementSize( optixFormat );
                const size_t       glLevelSize = glWidth * glHeight * glDepth * elementSize;
                if( ptrOffset + glLevelSize > bufferSize )
                {
                    throw IlwalidValue( RT_EXCEPTION_INFO,
                                        "OpenGL buffer objects must be the same size as OptiX buffer objects" );
                }

                GL::GetTexImage( glTarget, level, format, type, (char*)ptr + ptrOffset );
                ptrOffset += glLevelSize;
            }
        }

        // Reset the old texture
        GL::BindTexture( glTarget, oldTexId );
    }
    else
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "Unknown GL interop type", kind );
    }

    RT_ASSERT( GL::GetError() == GL_NO_ERROR );
}

void GfxInteropResource::queryProperties( Properties* props )
{
    // make sure there are no previous GL errors
    GL::checkGLError();

    // Query the size of the interop resource.
    if( kind == GfxInteropResource::OGL_BUFFER_OBJECT )
    {

#if !defined( __APPLE__ )
        if( !GL::IsBuffer( gl.glId ) )
        {
            throw IlwalidValue( RT_EXCEPTION_INFO, "The GL id does not belong to a GL buffer", gl.glId );
        }
#endif

        // Query the buffer size directly. - Bug 688327 probably prevents us from using this function
        // GLGetNamedBufferParameterivEXT(name, GL_BUFFER_SIZE, (GLint *)&size);

        // This works but is more expensive
        GLint bufId, bw, usage;
        GL::GetIntegerv( GL_PIXEL_PACK_BUFFER_BINDING, &bufId );
        GL::BindBuffer( GL_PIXEL_PACK_BUFFER, gl.glId );
        GL::GetBufferParameteriv( GL_PIXEL_PACK_BUFFER, GL_BUFFER_SIZE, &bw );
        GL::GetBufferParameteriv( GL_PIXEL_PACK_BUFFER, GL_BUFFER_USAGE, &usage );

        GL::BindBuffer( GL_PIXEL_PACK_BUFFER, bufId );

        // Update the size of the buffer. We do not know the format,
        // therefore this will not override the Buffer format as it gets
        // passed back up the chain.
        props->format         = RT_FORMAT_UNKNOWN;
        props->dimensionality = 1;
        props->elementSize    = 1;
        props->width          = bw;
        props->height         = 1;
        props->depth          = 1;
        props->glBufferUsage  = usage;
    }
    else if( kind == GfxInteropResource::OGL_TEXTURE )
    {
        if( GL::IsTexture( gl.glId ) != GL_TRUE )
            throw IlwalidValue( RT_EXCEPTION_INFO, "OpenGL ID does not belong to a texture.", gl.glId );

        // Save old texture target and set the new one
        GLint oldTexId;
        GLint glTarget = getGLTextureTarget( gl.target );
        GL::GetIntegerv( getGLTextureBindingTarget( gl.target ), &oldTexId );
        GL::BindTexture( glTarget, gl.glId );

        // Query the size
        GLint glWidth, glHeight, glDepth, glLevelCount;
        GL::GetTexLevelParameteriv( glTarget, 0, GL_TEXTURE_WIDTH, &glWidth );
        GL::GetTexLevelParameteriv( glTarget, 0, GL_TEXTURE_HEIGHT, &glHeight );
        GL::GetTexLevelParameteriv( glTarget, 0, GL_TEXTURE_DEPTH, &glDepth );

        // Query the gl max level
        {
            // In OGL there is no way to directly get exact number of levels
            int minDimSize = 0;
            if( gl.target == RT_TARGET_GL_TEXTURE_1D || gl.target == RT_TARGET_GL_TEXTURE_1D_ARRAY )
                minDimSize = glWidth;
            else if( gl.target == RT_TARGET_GL_TEXTURE_3D )
                minDimSize = std::min( std::min( glWidth, glHeight ), glDepth );
            else
                minDimSize = std::min( glWidth, glHeight );

            glLevelCount = 1;
            if( gl.target != RT_TARGET_GL_TEXTURE_RECTANGLE )  // texture rectangle doesn't support levels
            {
                while( minDimSize >>= 1 )
                {
                    GLint glLevelWidth, glLevelHeight, glLevelDepth;
                    GL::GetTexLevelParameteriv( glTarget, glLevelCount, GL_TEXTURE_WIDTH, &glLevelWidth );
                    GL::GetTexLevelParameteriv( glTarget, glLevelCount, GL_TEXTURE_HEIGHT, &glLevelHeight );
                    GL::GetTexLevelParameteriv( glTarget, glLevelCount, GL_TEXTURE_DEPTH, &glLevelDepth );
                    if( glLevelWidth == 0 || glLevelHeight == 0 || glLevelDepth == 0 )
                        break;
                    ++glLevelCount;
                }
            }
        }

        // Query LOD parameters
        GLfloat glAniso, glBaseLevel, glMinLevel, glMaxLevel;
        GL::GetTexParameterfv( glTarget, GL_TEXTURE_MAX_ANISOTROPY_EXT, &glAniso );
        GL::GetTexParameterfv( glTarget, GL_TEXTURE_BASE_LEVEL, &glBaseLevel );
        GL::GetTexParameterfv( glTarget, GL_TEXTURE_MIN_LOD, &glMinLevel );
        GL::GetTexParameterfv( glTarget, GL_TEXTURE_MAX_LOD, &glMaxLevel );

        // Query the format
        GLint glInternalFormat;
        GL::GetTexLevelParameteriv( glTarget, 0, GL_TEXTURE_INTERNAL_FORMAT, &glInternalFormat );

        // Reset the old texture
        GL::BindTexture( glTarget, oldTexId );

        // Colwert and return the size
        props->format      = glToOptixFormat( glInternalFormat );
        props->elementSize = getElementSize( props->format );
        props->width       = glWidth;
        props->height      = glHeight;
        props->depth       = glDepth;
        props->levelCount  = glLevelCount;
        props->layered     = getGLTextureLayeredFlag( gl.target );
        props->lwbe        = getGLTextureLwbeFlag( gl.target );
        props->srgb        = glInternalFormat == GL_SRGB8_ALPHA8_EXT;
        props->anisotropy  = glAniso;
        props->baseLevel   = glBaseLevel;
        props->minLevel    = glMinLevel;
        props->maxLevel    = glMaxLevel;
        if( prodlib::isCompressed( props->format ) )
        {
            props->width /= 4;
            props->height /= 4;
        }

        if( glWidth > 1 && glHeight > 1 && glDepth > 1 )
        {
            props->dimensionality = 3;
        }
        else if( glWidth > 1 && glHeight > 1 )
        {
            props->dimensionality = 2;
        }
        else
        {
            props->dimensionality = 1;
        }

        props->glInternalFormat = glInternalFormat;
    }
    else if( kind == GfxInteropResource::OGL_RENDER_BUFFER )
    {
        // Until OpenGL 4.5 the only way to query info about a renderbuffer is
        // to use glGetRenderbufferParameteriv() which returns info about the lwrrently
        // bound renderbuffer. But because there is no way to retrieve the lwrrently
        // bound renderbuffer, we have to create an fbo, bind rbId to it and then
        // query its properties. OpenGL 4.5 introduces glGetNameRenderbufferParameteriv()
        // which can query rbId directly.
        //
        // TODO: Use glGetNameRenderbufferParameteriv()
        //
        // BL: Keeping around an fbo makes a pretty big perf difference.
        // Kostya: OpenGL 4.5 is supported since 340.65 driver but OSX doesn't support 4.5
        // mstich: we should try to avoid requiring recent GL versions, that'll make it easier to run on CPU-only systems,
        // VMs, etc

        // Save old bindings
        GLint oldFBO     = 0;
        GLint oldBinding = 0;
        GL::GetIntegerv( GL_READ_FRAMEBUFFER_BINDING, &oldFBO );
        GL::GetIntegerv( GL_RENDERBUFFER_BINDING, &oldBinding );

        // If a cached FBO was passed in, use it
        GLuint fbo = props->glRenderbufferFBO ? *props->glRenderbufferFBO : 0;
        createInteropFBO( &fbo );
        GL::BindFramebuffer( GL_FRAMEBUFFER, fbo );
        GL::BindRenderbuffer( GL_RENDERBUFFER, gl.glId );

        // Query the size
        GLint bw, bh;
        GL::GetRenderbufferParameteriv( GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &bw );
        GL::GetRenderbufferParameteriv( GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &bh );

        // Query properties
        GLint depthBits = 0;
        GL::GetRenderbufferParameteriv( GL_RENDERBUFFER, GL_RENDERBUFFER_DEPTH_SIZE, &depthBits );
        bool  isDepthFormat = depthBits != 0;
        GLint samples       = 0;
        GL::GetRenderbufferParameteriv( GL_RENDERBUFFER, GL_RENDERBUFFER_SAMPLES, &samples );
        bool isMultisampleFormat = samples > 1;

        // Query the format
        GLint internalFormat;
        GL::GetRenderbufferParameteriv( GL_RENDERBUFFER, GL_RENDERBUFFER_INTERNAL_FORMAT, &internalFormat );

        // The current Lwca drivers don't support these!
        if( isDepthFormat )
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "Depth attachments are lwrrently not supported for GL render buffers." );
        if( isMultisampleFormat )
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "Multisample buffers are lwrrently not supported for GL render buffers." );

        /****************************************************************************\
     *
     * This call fails on Mac since the GL_RENDERBUFFER_COLOR_SAMPLES_LW
     * extension is not available.  Since the above GL_RENDERBUFFER_SAMPLES is
     * sufficient for determining multisampling mode, removing this for now.
     * see http://lwbugs/1039375
     *
    GL_CHECK_CALL( GLGetRenderbufferParameteriv( GL_RENDERBUFFER, GL_RENDERBUFFER_COLOR_SAMPLES_LW, &samples ) );
    if (samples > 1)
      info.m_isMultisampleFormat |= true;
    else
      info.m_isMultisampleFormat |= false;
     *
    \****************************************************************************/

        // restore previous bindingS
        GL::BindRenderbuffer( GL_RENDERBUFFER, oldBinding );
        GL::BindFramebuffer( GL_FRAMEBUFFER, oldFBO );

        // If caller wants to keep the FBO, make sure we return it. Otherwise delete the temp FBO.
        if( props->glRenderbufferFBO )
        {
            *props->glRenderbufferFBO = fbo;
        }
        else
        {
            GL::DeleteFramebuffers( 1, &fbo );
        }

        // Colwert and return the size
        props->format      = isDepthFormat ? glDepthToOptixFormat( internalFormat ) : glToOptixFormat( internalFormat );
        props->elementSize = getElementSize( props->format );
        props->dimensionality   = 2;
        props->width            = bw;
        props->height           = bh;
        props->depth            = 1;
        props->glInternalFormat = internalFormat;
    }
    else
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "Unexpected GL resource type", kind );
    }

    // make sure there are no current GL errors
    GL::checkGLError();
}

void GfxInteropResource::createInteropFBO( GLuint* fbo )
{
    if( *fbo )
        return;
    // create fbo
    GL::GenFramebuffers( 1, fbo );
    /*
    TODO:

    Lwrrently depth maps are not supported. Make sure to add depth
    maps support.
  */

    GLuint readBinding;
    // What is the lwrrently bound FBO?
    GL::GetIntegerv( GL_FRAMEBUFFER_BINDING, (GLint*)&readBinding );
    // Bind our FBO as the read frame buffer.
    GL::BindFramebuffer( GL_FRAMEBUFFER, *fbo );
    // Attach renderbuffer
    GL::FramebufferRenderbuffer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, gl.glId );
    // Reset FBO binding
    GL::BindFramebuffer( GL_FRAMEBUFFER, readBinding );
}
