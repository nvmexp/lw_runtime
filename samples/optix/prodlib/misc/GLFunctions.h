// Copyright LWPU Corporation 2008
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

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/gl3.h>
#else
#include <GL/gl.h>
#endif

#include <prodlib/misc/glext.h>

namespace corelib {
class ExelwtableModule;
}

namespace prodlib {

// Accesses a static instance of the GL wrapper
class GL;
GL& gl();

//
class GL
{
  public:
    ~GL();

    static bool ExtensionExtExternalObjectsAvailable();
    static bool ExtensionExtExternalObjectsWin32Available();

    static void BindBuffer( GLenum target, GLuint buffer );
    static void BindFramebuffer( GLenum target, GLuint framebuffer );
    static void BindRenderbuffer( GLenum target, GLuint renderbuffer );
    static void BindTexture( GLenum target, GLuint texture );
    static GLenum CheckFramebufferStatus( GLenum target );
    static void DeleteFramebuffers( GLsizei n, GLuint* framebuffers );
    static void FramebufferRenderbuffer( GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer );
    static void GenBuffers( GLsizei n, GLuint* ids );
    static void GenFramebuffers( GLsizei n, GLuint* ids );
    static void GenTextures( GLsizei n, GLuint* ids );
    static void GetBufferParameteriv( GLenum target, GLenum value, GLint* data );
    static GLenum         GetError();
    static const GLubyte* GetString( GLenum name );
    static void GetIntegerv( GLenum pname, GLint* params );
    static void GetUnsignedBytevEXT( GLenum pname, GLchar* data );
    static void GetRenderbufferParameteriv( GLenum target, GLenum pname, GLint* params );
    static void GetTexImage( GLenum target, GLint level, GLenum format, GLenum type, GLvoid* pixels );
    static void GetTexLevelParameteriv( GLenum target, GLint level, GLenum pname, GLint* params );
    static void GetTexParameterfv( GLenum target, GLenum pname, GLfloat* params );
    static void GetTexParameteriv( GLenum target, GLenum pname, GLint* params );
    static GLboolean IsRenderbuffer( GLuint renderbuffer );
    static GLboolean IsBuffer( GLuint buffer );
    static GLboolean IsTexture( GLuint texture );
    static void* MapBuffer( GLenum target, GLenum access );
    static void ReadPixels( GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid* pixels );
    static void TexImage2D( GLenum        target,
                            GLint         level,
                            GLint         internalformat,
                            GLsizei       width,
                            GLsizei       height,
                            GLint         border,
                            GLenum        format,
                            GLenum        type,
                            const GLvoid* data );
    static void TexImage3D( GLenum        target,
                            GLint         level,
                            GLint         internalformat,
                            GLsizei       width,
                            GLsizei       height,
                            GLsizei       depth,
                            GLint         border,
                            GLenum        format,
                            GLenum        type,
                            const GLvoid* data );
    static void TexSubImage2D( GLenum        target,
                               GLint         level,
                               GLint         xoffset,
                               GLint         yoffset,
                               GLsizei       width,
                               GLsizei       height,
                               GLenum        format,
                               GLenum        type,
                               const GLvoid* data );
    static void TexSubImage3D( GLenum        target,
                               GLint         level,
                               GLint         xoffset,
                               GLint         yoffset,
                               GLint         zoffset,
                               GLsizei       width,
                               GLsizei       height,
                               GLsizei       depth,
                               GLenum        format,
                               GLenum        type,
                               const GLvoid* data );
    static void TexParameterf( GLenum target, GLenum pname, GLfloat param );
    static void TexParameteri( GLenum target, GLenum pname, GLint param );
    static GLboolean UnmapBuffer( GLenum target );
    static void BufferData( GLenum target, GLsizeiptr size, const GLvoid* data, GLenum usage );
    static void BufferSubData( GLenum target, GLintptr offset, GLsizeiptr size, const GLvoid* data );
    static void GetBufferSubData( GLenum target, GLintptr offset, GLsizeiptr size, GLvoid* data );
    static void GetCompressedTexImage( GLenum target, GLint lod, GLvoid* data );
    static const char* ErrorString( GLenum error );

    // Throws exception if glGetError() != GL_NO_ERROR
    static void checkGLError();

  private:
    void glBindBuffer( GLenum target, GLuint buffer );
    void glBindFramebuffer( GLenum target, GLuint framebuffer );
    void glBindRenderbuffer( GLenum target, GLuint renderbuffer );
    void glBindTexture( GLenum target, GLuint texture );
    GLenum glCheckFramebufferStatus( GLenum target );
    void glDeleteFramebuffers( GLsizei n, GLuint* framebuffers );
    void glFramebufferRenderbuffer( GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer );
    void glGenBuffers( GLsizei n, GLuint* ids );
    void glGenFramebuffers( GLsizei n, GLuint* ids );
    void glGenTextures( GLsizei n, GLuint* ids );
    void glGetBufferParameteriv( GLenum target, GLenum value, GLint* data );
    GLenum         glGetError();
    const GLubyte* glGetString( GLenum name );
    void glGetIntegerv( GLenum pname, GLint* params );
    void glGetUnsignedBytevEXT( GLenum pname, GLchar* data );
    void glGetRenderbufferParameteriv( GLenum target, GLenum pname, GLint* params );
    void glGetTexImage( GLenum target, GLint level, GLenum format, GLenum type, GLvoid* pixels );
    void glGetTexLevelParameteriv( GLenum target, GLint level, GLenum pname, GLint* params );
    void glGetTexParameterfv( GLenum target, GLenum pname, GLfloat* params );
    void glGetTexParameteriv( GLenum target, GLenum pname, GLint* params );
    GLboolean glIsRenderbuffer( GLuint renderbuffer );
    GLboolean glIsBuffer( GLuint buffer );
    GLboolean glIsTexture( GLuint texture );
    void* glMapBuffer( GLenum target, GLenum access );
    void glReadPixels( GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid* pixels );
    void glTexImage2D( GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid* data );
    void glTexImage3D( GLenum        target,
                       GLint         level,
                       GLint         internalformat,
                       GLsizei       width,
                       GLsizei       height,
                       GLsizei       depth,
                       GLint         border,
                       GLenum        format,
                       GLenum        type,
                       const GLvoid* data );
    void glTexSubImage2D( GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid* data );
    void glTexSubImage3D( GLenum        target,
                          GLint         level,
                          GLint         xoffset,
                          GLint         yoffset,
                          GLint         zoffset,
                          GLsizei       width,
                          GLsizei       height,
                          GLsizei       depth,
                          GLenum        format,
                          GLenum        type,
                          const GLvoid* data );
    void glTexParameterf( GLenum target, GLenum pname, GLfloat param );
    void glTexParameteri( GLenum target, GLenum pname, GLint param );
    GLboolean glUnmapBuffer( GLenum target );
    void glBufferData( GLenum target, GLsizeiptr size, const GLvoid* data, GLenum usage );
    void glBufferSubData( GLenum target, GLintptr offset, GLsizeiptr size, const GLvoid* data );
    void glGetBufferSubData( GLenum target, GLintptr offset, GLsizeiptr size, GLvoid* data );
    void glGetCompressedTexImage( GLenum target, GLint lod, GLvoid* data );
    const char* gluErrorString( GLenum error );

    void getGlExtensions();

  private:
    typedef void ( *glBindBuffer_t )( GLenum target, GLuint buffer );
    typedef void ( *glBindFramebuffer_t )( GLenum target, GLuint framebuffer );
    typedef void ( *glBindRenderbuffer_t )( GLenum target, GLuint renderbuffer );
    typedef void ( *glBindTexture_t )( GLenum target, GLuint texture );
    typedef GLenum ( *glCheckFramebufferStatus_t )( GLenum target );
    typedef void ( *glDeleteFramebuffers_t )( GLsizei n, GLuint* framebuffers );
    typedef void ( *glFramebufferRenderbuffer_t )( GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer );
    typedef void ( *glGenBuffers_t )( GLsizei n, GLuint* ids );
    typedef void ( *glGenFramebuffers_t )( GLsizei n, GLuint* ids );
    typedef void ( *glGenTextures_t )( GLsizei n, GLuint* ids );
    typedef void ( *glGetBufferParameteriv_t )( GLenum target, GLenum value, GLint* data );
    typedef GLenum ( *glGetError_t )();
    typedef const GLubyte* ( *glGetString_t )( GLenum name );
    typedef void ( *glGetIntegerv_t )( GLenum pname, GLint* params );
    typedef void ( *glGetUnsignedBytevEXT_t )( GLenum pname, GLchar* data );
    typedef void ( *glGetRenderbufferParameteriv_t )( GLenum target, GLenum pname, GLint* params );
    typedef void ( *glGetTexImage_t )( GLenum target, GLint level, GLenum format, GLenum type, GLvoid* pixels );
    typedef void ( *glGetTexLevelParameteriv_t )( GLenum target, GLint level, GLenum pname, GLint* params );
    typedef void ( *glGetTexParameterfv_t )( GLenum target, GLenum pname, GLfloat* params );
    typedef void ( *glGetTexParameteriv_t )( GLenum target, GLenum pname, GLint* params );
    typedef GLboolean ( *glIsRenderbuffer_t )( GLuint renderbuffer );
    typedef GLboolean ( *glIsBuffer_t )( GLuint buffer );
    typedef GLboolean ( *glIsTexture_t )( GLuint texture );
    typedef void* ( *glMapBuffer_t )( GLenum target, GLenum access );
    typedef void ( *glReadPixels_t )( GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid* pixels );
    typedef void ( *glTexImage2D_t )( GLenum        target,
                                      GLint         level,
                                      GLint         internalformat,
                                      GLsizei       width,
                                      GLsizei       height,
                                      GLint         border,
                                      GLenum        format,
                                      GLenum        type,
                                      const GLvoid* data );
    typedef void ( *glTexImage3D_t )( GLenum        target,
                                      GLint         level,
                                      GLint         internalformat,
                                      GLsizei       width,
                                      GLsizei       height,
                                      GLsizei       depth,
                                      GLint         border,
                                      GLenum        format,
                                      GLenum        type,
                                      const GLvoid* data );
    typedef void ( *glTexSubImage2D_t )( GLenum        target,
                                         GLint         level,
                                         GLint         xoffset,
                                         GLint         yoffset,
                                         GLsizei       width,
                                         GLsizei       height,
                                         GLenum        format,
                                         GLenum        type,
                                         const GLvoid* data );
    typedef void ( *glTexSubImage3D_t )( GLenum        target,
                                         GLint         level,
                                         GLint         xoffset,
                                         GLint         yoffset,
                                         GLint         zoffset,
                                         GLsizei       width,
                                         GLsizei       height,
                                         GLsizei       depth,
                                         GLenum        format,
                                         GLenum        type,
                                         const GLvoid* data );
    typedef void ( *glTexParameterf_t )( GLenum target, GLenum pname, GLfloat param );
    typedef void ( *glTexParameteri_t )( GLenum target, GLenum pname, GLint param );
    typedef GLboolean ( *glUnmapBuffer_t )( GLenum target );
    typedef void ( *glBufferData_t )( GLenum target, GLsizeiptr size, const GLvoid* data, GLenum usage );
    typedef void ( *glBufferSubData_t )( GLenum target, GLintptr offset, GLsizeiptr size, const GLvoid* data );
    typedef void ( *glGetBufferSubData_t )( GLenum target, GLintptr offset, GLsizeiptr size, GLvoid* data );
    typedef void ( *glGetCompressedTexImage_t )( GLenum target, GLint lod, GLvoid* data );

    glBindBuffer_t                 p_glBindBuffer;
    glBindFramebuffer_t            p_glBindFramebuffer;
    glBindRenderbuffer_t           p_glBindRenderbuffer;
    glBindTexture_t                p_glBindTexture;
    glCheckFramebufferStatus_t     p_glCheckFramebufferStatus;
    glDeleteFramebuffers_t         p_glDeleteFramebuffers;
    glFramebufferRenderbuffer_t    p_glFramebufferRenderbuffer;
    glGenBuffers_t                 p_glGenBuffers;
    glGenFramebuffers_t            p_glGenFramebuffers;
    glGenTextures_t                p_glGenTextures;
    glGetBufferParameteriv_t       p_glGetBufferParameteriv;
    glGetError_t                   p_glGetError;
    glGetString_t                  p_glGetString;
    glGetIntegerv_t                p_glGetIntegerv;
    glGetUnsignedBytevEXT_t        p_glGetUnsignedBytevEXT;
    glGetRenderbufferParameteriv_t p_glGetRenderbufferParameteriv;
    glGetTexImage_t                p_glGetTexImage;
    glGetTexLevelParameteriv_t     p_glGetTexLevelParameteriv;
    glGetTexParameterfv_t          p_glGetTexParameterfv;
    glGetTexParameteriv_t          p_glGetTexParameteriv;
    glIsRenderbuffer_t             p_glIsRenderbuffer;
    glIsBuffer_t                   p_glIsBuffer;
    glIsTexture_t                  p_glIsTexture;
    glMapBuffer_t                  p_glMapBuffer;
    glReadPixels_t                 p_glReadPixels;
    glTexImage2D_t                 p_glTexImage2D;
    glTexImage3D_t                 p_glTexImage3D;
    glTexSubImage2D_t              p_glTexSubImage2D;
    glTexSubImage3D_t              p_glTexSubImage3D;
    glTexParameterf_t              p_glTexParameterf;
    glTexParameteri_t              p_glTexParameteri;
    glUnmapBuffer_t                p_glUnmapBuffer;
    glBufferData_t                 p_glBufferData;
    glBufferSubData_t              p_glBufferSubData;
    glGetBufferSubData_t           p_glGetBufferSubData;
    glGetCompressedTexImage_t      p_glGetCompressedTexImage;

    bool m_extExternalObjectsAvailable      = false;
    bool m_extExternalObjectsWin32Available = false;

    bool                       m_gl_available;
    corelib::ExelwtableModule* m_gl_lib;

    // All access should be through static accessors or the gl function
    GL();
    GL( const GL& ) = delete;
    GL& operator=( const GL& ) = delete;

    // Throws exception if GL was not able to be loaded
    void assertGLAvailable();

    // Accesses a static instance of the GL wrapper
    friend GL& gl();
};

}  // end namespace prodlib
