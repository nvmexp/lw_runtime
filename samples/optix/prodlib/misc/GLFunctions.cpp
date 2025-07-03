// Copyright LWPU Corporation 2014
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL
// LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
// CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
// LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
// INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGES

#include <prodlib/misc/GLFunctions.h>

#include <corelib/misc/String.h>
#include <corelib/system/ExelwtableModule.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/AssertionFailure.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#include <sstream>
#include <string>

using namespace prodlib;

namespace {
// clang-format off
PublicKnob<bool> k_enableAllGLChecks( RT_PUBLIC_DSTRING( "context.enableAllGLChecks" ), false, RT_PUBLIC_DSTRING( "Enable glGetError on every OpenGL API call" ) );
// clang-format on
}

// checking glGetError every GL call can be expensive for release build
// clang-format off
#define GL_CHECK_CALL( call )                                         \
do {                                                                  \
  call;                                                               \
  if( k_enableAllGLChecks.get() ) {                                   \
    GLenum err = GL::GetError();                                      \
    if( err != GL_NO_ERROR ) {                                        \
      std::ostringstream oss;                                         \
      do {                                                            \
        oss << "GL error: " << GL::ErrorString( err ) << "\n";        \
        err = GL::GetError();                                         \
      } while( err != GL_NO_ERROR );                                  \
      throw prodlib::AssertionFailure( RT_EXCEPTION_INFO, oss.str() ); \
    }                                                                 \
  }                                                                   \
} while(false)                                                        \

#define GL_CHECK()                                                    \
do {                                                                  \
  GLenum err = GL::GetError();                                        \
  if( err != GL_NO_ERROR ) {                                          \
    std::ostringstream oss;                                           \
    do {                                                              \
      oss << "GL error: " << GL::ErrorString( err ) << "\n";          \
      err = GL::GetError();                                           \
    } while( err != GL_NO_ERROR );                                    \
    throw prodlib::AssertionFailure( RT_EXCEPTION_INFO, oss.str() );   \
  }                                                                   \
} while(false)                                                        \
// clang-format on

void GL::checkGLError()
{
  GL_CHECK();
}

prodlib::GL& prodlib::gl()
{
  static GL instance;
  instance.assertGLAvailable();
  return instance;
}

void GL::assertGLAvailable()
{
  if( !m_gl_available )
    throw prodlib::AssertionFailure( RT_EXCEPTION_INFO, "OpenGL not available" );
}

GL::GL()
    : p_glBindBuffer( nullptr )
    , p_glBindFramebuffer( nullptr )
    , p_glBindRenderbuffer( nullptr )
    , p_glBindTexture( nullptr )
    , p_glCheckFramebufferStatus( nullptr )
    , p_glDeleteFramebuffers( nullptr )
    , p_glFramebufferRenderbuffer( nullptr )
    , p_glGenBuffers( nullptr )
    , p_glGenFramebuffers( nullptr )
    , p_glGenTextures( nullptr )
    , p_glGetBufferParameteriv( nullptr )
    , p_glGetError( nullptr )
    , p_glGetString( nullptr )
    , p_glGetIntegerv( nullptr )
    , p_glGetUnsignedBytevEXT( nullptr )
    , p_glGetRenderbufferParameteriv( nullptr )
    , p_glGetTexImage( nullptr )
    , p_glGetTexLevelParameteriv( nullptr )
    , p_glGetTexParameterfv( nullptr )
    , p_glGetTexParameteriv( nullptr )
    , p_glIsRenderbuffer( nullptr )
    , p_glIsBuffer( nullptr )
    , p_glIsTexture( nullptr )
    , p_glMapBuffer( nullptr )
    , p_glReadPixels( nullptr )
    , p_glTexImage2D( nullptr )
    , p_glTexImage3D( nullptr )
    , p_glTexSubImage2D( nullptr )
    , p_glTexSubImage3D( nullptr )
    , p_glTexParameterf( nullptr )
    , p_glTexParameteri( nullptr )
    , p_glUnmapBuffer( nullptr )
    , p_glBufferData( nullptr )
    , p_glBufferSubData( nullptr )
    , p_glGetBufferSubData( nullptr )
    , p_glGetCompressedTexImage( nullptr )
    , m_gl_available( false )
    , m_gl_lib( nullptr )
{
#if defined( _WIN32 )
#define GET_PROC_ADDR( name ) ( ( PROC( * )( LPCSTR ) )( getProcAddress ) )( #name )

  // Use an absolute path to load the driver's library (http://lwbugs/1297905)
  char gl_dll_path[MAX_PATH];
  if( !GetSystemDirectory( gl_dll_path, MAX_PATH ) )
  {
    lwarn << "GL library not found: Could not determine system path" << std::endl;
    delete m_gl_lib;
    m_gl_lib = 0;
    return;
  }

  std::ostringstream path;
  path << gl_dll_path << "/Opengl32.dll";

  m_gl_lib             = new corelib::ExelwtableModule( path.str().c_str() );

  if( path.str().length() > MAX_PATH || !m_gl_lib->init() )
  {
    lwarn << "GL library not found: Invalid driver system path for GL library" << std::endl;
    delete m_gl_lib;
    m_gl_lib = 0;
    return;
  }

  void* getProcAddress = m_gl_lib->getFunction( "wglGetProcAddress" );
  if( !getProcAddress )
    return;

#elif defined( __linux__ )
#define GET_PROC_ADDR( name ) ( ( void* (*)(const GLubyte*))( getProcAddress ) )( (const GLubyte*)#name )

  m_gl_lib             = new corelib::ExelwtableModule( "libGL.so" );
  if( !m_gl_lib->init() )
  {
    lwarn << "GL library not found: Invalid driver system path for GL library"
          << std::endl;
    delete m_gl_lib;
    m_gl_lib = 0;
    return;
  }

  void* getProcAddress = m_gl_lib->getFunction( "glXGetProcAddressARB" );
  if( !getProcAddress )
    return;

#else  // MAC
#define GET_PROC_ADDR( name ) (::name )
#endif

// I do not know why, but for Windows older GL function ptrs (<=GL1.2?) cannot be queried
// via wglGetProcAddress, but must be queried with GetProcAddress. so, i am just using
// m_gl_lib->getFunction for this!

#ifndef _WIN32
  p_glGetError               = ( glGetError_t )( GET_PROC_ADDR( glGetError ) );
  p_glGetString              = ( glGetString_t )( GET_PROC_ADDR( glGetString ) );
  p_glGetIntegerv            = ( glGetIntegerv_t )( GET_PROC_ADDR( glGetIntegerv ) );
  p_glGetTexImage            = ( glGetTexImage_t )( GET_PROC_ADDR( glGetTexImage ) );
  p_glReadPixels             = ( glReadPixels_t )( GET_PROC_ADDR( glReadPixels ) );
  p_glTexImage2D             = ( glTexImage2D_t )( GET_PROC_ADDR( glTexImage2D ) );
  p_glTexImage3D             = ( glTexImage3D_t )( GET_PROC_ADDR( glTexImage3D ) );
  p_glTexSubImage2D          = ( glTexSubImage2D_t )( GET_PROC_ADDR( glTexSubImage2D ) );
  p_glTexSubImage3D          = ( glTexSubImage3D_t )( GET_PROC_ADDR( glTexSubImage3D ) );
  p_glTexParameterf          = ( glTexParameterf_t )( GET_PROC_ADDR( glTexParameterf ) );
  p_glTexParameteri          = ( glTexParameteri_t )( GET_PROC_ADDR( glTexParameteri ) );
  p_glGetTexLevelParameteriv = ( glGetTexLevelParameteriv_t )( GET_PROC_ADDR( glGetTexLevelParameteriv ) );
  p_glGetTexParameterfv      = ( glGetTexParameterfv_t )( GET_PROC_ADDR( glGetTexParameterfv ) );
  p_glGetTexParameteriv      = ( glGetTexParameteriv_t )( GET_PROC_ADDR( glGetTexParameteriv ) );
#else
  p_glGetError               = ( glGetError_t )( m_gl_lib->getFunction( "glGetError" ) );
  p_glGetString              = ( glGetString_t )( m_gl_lib->getFunction( "glGetString" ) );
  p_glGetIntegerv            = ( glGetIntegerv_t )( m_gl_lib->getFunction( "glGetIntegerv" ) );
  p_glGetTexImage            = ( glGetTexImage_t )( m_gl_lib->getFunction( "glGetTexImage" ) );
  p_glReadPixels             = ( glReadPixels_t )( m_gl_lib->getFunction( "glReadPixels" ) );
  p_glTexImage2D             = ( glTexImage2D_t )( m_gl_lib->getFunction( "glTexImage2D" ) );
  p_glTexImage3D             = ( glTexImage3D_t )( m_gl_lib->getFunction( "glTexImage3D" ) );
  p_glTexSubImage2D          = ( glTexSubImage2D_t )( m_gl_lib->getFunction( "glTexSubImage2D" ) );
  p_glTexSubImage3D          = ( glTexSubImage3D_t )( m_gl_lib->getFunction( "glTexSubImage3D" ) );
  p_glTexParameterf          = ( glTexParameterf_t )( m_gl_lib->getFunction( "glTexParameterf" ) );
  p_glTexParameteri          = ( glTexParameteri_t )( m_gl_lib->getFunction( "glTexParameteri" ) );
  p_glGetTexLevelParameteriv = ( glGetTexLevelParameteriv_t )( m_gl_lib->getFunction( "glGetTexLevelParameteriv" ) );
  p_glGetTexParameterfv      = ( glGetTexParameterfv_t )( m_gl_lib->getFunction( "glGetTexParameterfv" ) );
  p_glGetTexParameteriv      = ( glGetTexParameteriv_t )( m_gl_lib->getFunction( "glGetTexParameteriv" ) );
#endif
  p_glBindBuffer                 = ( glBindBuffer_t )( GET_PROC_ADDR( glBindBuffer ) );
  p_glBindFramebuffer            = ( glBindFramebuffer_t )( GET_PROC_ADDR( glBindFramebuffer ) );
  p_glBindRenderbuffer           = ( glBindRenderbuffer_t )( GET_PROC_ADDR( glBindRenderbuffer ) );
  p_glBindTexture                = ( glBindTexture_t )( GET_PROC_ADDR( glBindTexture ) );
  p_glCheckFramebufferStatus     = ( glCheckFramebufferStatus_t )( GET_PROC_ADDR( glCheckFramebufferStatus ) );
  p_glDeleteFramebuffers         = ( glDeleteFramebuffers_t )( GET_PROC_ADDR( glDeleteFramebuffers ) );
  p_glFramebufferRenderbuffer    = ( glFramebufferRenderbuffer_t )( GET_PROC_ADDR( glFramebufferRenderbuffer ) );
  p_glGenBuffers                 = ( glGenBuffers_t )( GET_PROC_ADDR( glGenBuffers ) );
  p_glGenFramebuffers            = ( glGenFramebuffers_t )( GET_PROC_ADDR( glGenFramebuffers ) );
  p_glGenTextures                = ( glGenTextures_t )( GET_PROC_ADDR( glGenTextures ) );
  p_glGetBufferParameteriv       = ( glGetBufferParameteriv_t )( GET_PROC_ADDR( glGetBufferParameteriv ) );
  p_glGetRenderbufferParameteriv = ( glGetRenderbufferParameteriv_t )( GET_PROC_ADDR( glGetRenderbufferParameteriv ) );
  p_glIsBuffer                   = ( glIsBuffer_t )( GET_PROC_ADDR( glIsBuffer ) );
  p_glIsRenderbuffer             = ( glIsRenderbuffer_t )( GET_PROC_ADDR( glIsRenderbuffer ) );
  p_glIsTexture                  = ( glIsTexture_t )( GET_PROC_ADDR( glIsTexture ) );
  p_glMapBuffer                  = ( glMapBuffer_t )( GET_PROC_ADDR( glMapBuffer ) );
  p_glUnmapBuffer                = ( glUnmapBuffer_t )( GET_PROC_ADDR( glUnmapBuffer ) );
  p_glBufferData                 = ( glBufferData_t )( GET_PROC_ADDR( glBufferData ) );
  p_glBufferSubData              = ( glBufferSubData_t )( GET_PROC_ADDR( glBufferSubData ) );
  p_glGetBufferSubData           = ( glGetBufferSubData_t )( GET_PROC_ADDR( glGetBufferSubData ) );
  p_glGetCompressedTexImage      = ( glGetCompressedTexImage_t )( GET_PROC_ADDR( glGetCompressedTexImage ) );

  getGlExtensions();

#if !defined(__APPLE__)
  // This is provided by the EXT_external_objects extension.
  if( m_extExternalObjectsAvailable )
    p_glGetUnsignedBytevEXT        = ( glGetUnsignedBytevEXT_t )( GET_PROC_ADDR( glGetUnsignedBytevEXT ) );
#endif

#undef GET_PROC_ADDR

  m_gl_available = true;
}

GL::~GL()
{
  if( m_gl_lib )
    delete m_gl_lib;
}

void GL::getGlExtensions()
{
  std::string extensionString = std::string( (char*)glGetString( GL_EXTENSIONS ) );
  std::istringstream extensionStream( extensionString );
  std::string lwrrExtension;
  while( extensionStream >> lwrrExtension )
  {
    if( lwrrExtension.compare( "GL_EXT_memory_object" ) == 0 )
      m_extExternalObjectsAvailable = true;
#ifdef _WIN32
    if( lwrrExtension.compare( "GL_EXT_memory_object_win32" ) == 0 )
      m_extExternalObjectsWin32Available = true;
#endif // _WIN32
  }
}

bool GL::ExtensionExtExternalObjectsAvailable()
{
  return gl().m_extExternalObjectsAvailable;
}

bool GL::ExtensionExtExternalObjectsWin32Available()
{
  return gl().m_extExternalObjectsWin32Available;
}

void GL::BindBuffer( GLenum target, GLuint buffer )
{
  GL_CHECK_CALL( gl().glBindBuffer( target, buffer ) );
}

void GL::BindFramebuffer( GLenum target, GLuint framebuffer )
{
  GL_CHECK_CALL( gl().glBindFramebuffer( target, framebuffer ) );
}

void GL::BindRenderbuffer( GLenum target, GLuint renderbuffer )
{
  GL_CHECK_CALL( gl().glBindRenderbuffer( target, renderbuffer ) );
}

void GL::BindTexture( GLenum target, GLuint texture )
{
  GL_CHECK_CALL( gl().glBindTexture( target, texture ) );
}

GLenum GL::CheckFramebufferStatus( GLenum target )
{
  GLenum result = 0;
  GL_CHECK_CALL( result = gl().glCheckFramebufferStatus( target ) );
  return result;
}

void GL::DeleteFramebuffers( GLsizei n, GLuint* framebuffers )
{
  GL_CHECK_CALL( gl().glDeleteFramebuffers( n, framebuffers ) );
}

void GL::FramebufferRenderbuffer( GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer )
{
  GL_CHECK_CALL( gl().glFramebufferRenderbuffer( target, attachment, renderbuffertarget, renderbuffer ) );
}

void GL::GenBuffers( GLsizei n, GLuint* ids )
{
  GL_CHECK_CALL( gl().glGenBuffers( n, ids ) );
}

void GL::GenFramebuffers( GLsizei n, GLuint* ids )
{
  gl().glGenFramebuffers( n, ids );
}

void GL::GenTextures( GLsizei n, GLuint* ids )
{
  GL_CHECK_CALL( gl().glGenTextures( n, ids ) );
}

void GL::GetBufferParameteriv( GLenum target, GLenum value, GLint* data )
{
  GL_CHECK_CALL( gl().glGetBufferParameteriv( target, value, data ) );
}

GLenum GL::GetError()
{
  GLenum result = 0;
  GL_CHECK_CALL( result = gl().glGetError() );
  return result;
}

const GLubyte * GL::GetString( GLenum name )
{
  const GLubyte *result;
  GL_CHECK_CALL( result = gl().glGetString( name ) );
  return result;
}

void GL::GetIntegerv( GLenum pname, GLint* params )
{
  GL_CHECK_CALL( gl().glGetIntegerv( pname, params ) );
}

void GL::GetUnsignedBytevEXT( GLenum pname, GLchar* data )
{
  GL_CHECK_CALL( gl().glGetUnsignedBytevEXT( pname, data ) );
}

void GL::GetRenderbufferParameteriv( GLenum target, GLenum pname, GLint *params )
{
  GL_CHECK_CALL( gl().glGetRenderbufferParameteriv( target, pname, params ) );
}

void GL::GetTexImage( GLenum target, GLint level, GLenum format, GLenum type, GLvoid *pixels )
{
  GL_CHECK_CALL( gl().glGetTexImage( target, level, format, type, pixels ) );
}

void GL::GetTexLevelParameteriv( GLenum target, GLint level, GLenum pname, GLint *params )
{
  GL_CHECK_CALL( gl().glGetTexLevelParameteriv( target, level, pname, params ) );
}

void GL::GetTexParameterfv( GLenum target, GLenum pname, GLfloat *params )
{
  GL_CHECK_CALL( gl().glGetTexParameterfv( target, pname, params ) );
}

void GL::GetTexParameteriv( GLenum target, GLenum pname, GLint *params )
{
  GL_CHECK_CALL( gl().glGetTexParameteriv( target, pname, params ) );
}

GLboolean GL::IsRenderbuffer( GLuint renderbuffer )
{
  GLboolean result = 0;
  GL_CHECK_CALL( result = gl().glIsRenderbuffer( renderbuffer ) );
  return result;
}

GLboolean GL::IsBuffer( GLuint buffer )
{
  GLboolean result = 0;
  GL_CHECK_CALL( result = gl().glIsBuffer( buffer ) );
  return result;
}

GLboolean GL::IsTexture( GLuint texture )
{
  GLboolean result = 0;
  GL_CHECK_CALL( result = gl().glIsTexture( texture ) );
  return result;
}

void* GL::MapBuffer( GLenum target, GLenum access )
{
  void* result = 0;
  GL_CHECK_CALL( result = gl().glMapBuffer( target, access ) );
  return result;
}

void GL::ReadPixels( GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid* pixels )
{
  GL_CHECK_CALL( gl().glReadPixels( x, y, width, height, format, type, pixels ) );
}

void GL::TexImage2D( GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid* data )
{
  GL_CHECK_CALL( gl().glTexImage2D( target, level, internalformat, width, height, border, format, type, data ) );
}

void GL::TexImage3D( GLenum        target,
                     GLint         level,
                     GLint         internalformat,
                     GLsizei       width,
                     GLsizei       height,
                     GLsizei       depth,
                     GLint         border,
                     GLenum        format,
                     GLenum        type,
                     const GLvoid* data )
{
  GL_CHECK_CALL( gl().glTexImage3D( target, level, internalformat, width, height, depth, border, format, type, data ) );
}

void GL::TexSubImage2D( GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid* data )
{
  GL_CHECK_CALL( gl().glTexSubImage2D( target, level, xoffset, yoffset, width, height, format, type, data ) );
}

void GL::TexSubImage3D( GLenum        target,
                        GLint         level,
                        GLint         xoffset,
                        GLint         yoffset,
                        GLint         zoffset,
                        GLsizei       width,
                        GLsizei       height,
                        GLsizei       depth,
                        GLenum        format,
                        GLenum        type,
                        const GLvoid* data )
{
  GL_CHECK_CALL( gl().glTexSubImage3D( target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, data ) );
}

void GL::TexParameterf( GLenum target, GLenum pname, GLfloat param )
{
  GL_CHECK_CALL( gl().glTexParameterf( target, pname, param ) );
}

void GL::TexParameteri( GLenum target, GLenum pname, GLint param )
{
  GL_CHECK_CALL( gl().glTexParameteri( target, pname, param ) );
}

GLboolean GL::UnmapBuffer( GLenum target )
{
  GLboolean result = 0;
  GL_CHECK_CALL( result = gl().glUnmapBuffer( target ) );
  return result;
}

void GL::BufferData( GLenum target, GLsizeiptr size, const GLvoid* data, GLenum usage )
{
  GL_CHECK_CALL( gl().glBufferData( target, size, data, usage ) );
}

void GL::BufferSubData( GLenum target, GLintptr offset, GLsizeiptr size, const GLvoid* data )
{
  GL_CHECK_CALL( gl().glBufferSubData(target, offset, size, data) );
}

void GL::GetBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, GLvoid* data)
{
  GL_CHECK_CALL( gl().glGetBufferSubData( target, offset, size, data ) );
}

void GL::GetCompressedTexImage( GLenum target, GLint lod, GLvoid* data )
{
  GL_CHECK_CALL( gl().glGetCompressedTexImage(target, lod, data ) );
}

const char* GL::ErrorString( GLenum error )
{
  const char* result = 0;
  GL_CHECK_CALL( result = gl().gluErrorString( error ) );
  return result;
}

//--------------------------- implementations -------------------------------

void GL::glBindBuffer( GLenum target, GLuint buffer )
{
  p_glBindBuffer( target, buffer );
}

void GL::glBindFramebuffer( GLenum target, GLuint framebuffer )
{
  p_glBindFramebuffer( target, framebuffer );
}

void GL::glBindRenderbuffer( GLenum target, GLuint renderbuffer )
{
  p_glBindRenderbuffer( target, renderbuffer );
}

GLboolean GL::glIsBuffer( GLuint buffer )
{
  return p_glIsBuffer( buffer );
}

void GL::glBindTexture( GLenum target, GLuint texture )
{
  p_glBindTexture( target, texture );
}

GLenum GL::glCheckFramebufferStatus( GLenum target )
{
  return p_glCheckFramebufferStatus( target );
}

void GL::glDeleteFramebuffers( GLsizei n, GLuint* framebuffers )
{
  p_glDeleteFramebuffers( n, framebuffers );
}

void GL::glFramebufferRenderbuffer( GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer )
{
  p_glFramebufferRenderbuffer( target, attachment, renderbuffertarget, renderbuffer );
}

void GL::glGenBuffers( GLsizei n, GLuint* ids )
{
  p_glGenBuffers( n, ids );
}

void GL::glGenFramebuffers( GLsizei n, GLuint* ids )
{
  p_glGenFramebuffers( n, ids );
}

void GL::glGenTextures( GLsizei n, GLuint* ids )
{
  p_glGenTextures( n, ids );
}

void GL::glGetBufferParameteriv( GLenum target, GLenum value, GLint* data )
{
  p_glGetBufferParameteriv( target, value, data );
}

GLenum GL::glGetError()
{
  return p_glGetError();
}

const GLubyte *GL::glGetString( GLenum name )
{
  return p_glGetString( name );
}


void GL::glGetIntegerv( GLenum pname, GLint* params )
{
  p_glGetIntegerv( pname, params );
}

void GL::glGetUnsignedBytevEXT( GLenum pname, GLchar* data )
{
  p_glGetUnsignedBytevEXT( pname, data );
}

void GL::glGetRenderbufferParameteriv( GLenum target, GLenum pname, GLint* params )
{
  p_glGetRenderbufferParameteriv( target, pname, params );
}

void GL::glGetTexLevelParameteriv( GLenum target, GLint level, GLenum pname, GLint* params )
{
  p_glGetTexLevelParameteriv( target, level, pname, params );
}

void GL::glGetTexParameterfv( GLenum target, GLenum pname, GLfloat* params )
{
  p_glGetTexParameterfv( target, pname, params );
}

void GL::glGetTexParameteriv( GLenum target, GLenum pname, GLint* params )
{
  p_glGetTexParameteriv( target, pname, params );
}

void GL::glGetTexImage( GLenum target, GLint level, GLenum format, GLenum type, GLvoid* pixels )
{
  p_glGetTexImage( target, level, format, type, pixels );
}

GLboolean GL::glIsRenderbuffer( GLuint renderbuffer )
{
  return p_glIsRenderbuffer( renderbuffer );
}

GLboolean GL::glIsTexture( GLuint texture )
{
  return p_glIsTexture( texture );
}

void* GL::glMapBuffer( GLenum target, GLenum access )
{
  return p_glMapBuffer( target, access );
}

void GL::glReadPixels( GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid* pixels )
{
  p_glReadPixels( x, y, width, height, format, type, pixels );
}

void GL::glTexImage2D( GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid* data )
{
  p_glTexImage2D( target, level, internalformat, width, height, border, format, type, data );
}

void GL::glTexImage3D( GLenum        target,
                       GLint         level,
                       GLint         internalformat,
                       GLsizei       width,
                       GLsizei       height,
                       GLsizei       depth,
                       GLint         border,
                       GLenum        format,
                       GLenum        type,
                       const GLvoid* data )
{
  p_glTexImage3D( target, level, internalformat, width, height, depth, border, format, type, data );
}

void GL::glTexSubImage2D( GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid* data )
{
  p_glTexSubImage2D( target, level, xoffset, yoffset, width, height, format, type, data );
}

void GL::glTexSubImage3D( GLenum        target,
                          GLint         level,
                          GLint         xoffset,
                          GLint         yoffset,
                          GLint         zoffset,
                          GLsizei       width,
                          GLsizei       height,
                          GLsizei       depth,
                          GLenum        format,
                          GLenum        type,
                          const GLvoid* data )
{
  p_glTexSubImage3D( target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, data );
}

void GL::glTexParameterf( GLenum target, GLenum pname, GLfloat param )
{
  p_glTexParameterf( target, pname, param );
}

void GL::glTexParameteri( GLenum target, GLenum pname, GLint param )
{
  p_glTexParameteri( target, pname, param );
}

GLboolean GL::glUnmapBuffer( GLenum target )
{
  return p_glUnmapBuffer( target );
}

void GL::glBufferData( GLenum target, GLsizeiptr size, const GLvoid* data, GLenum usage )
{
  p_glBufferData( target, size, data, usage );
}

void GL::glBufferSubData( GLenum target, GLintptr offset, GLsizeiptr size, const GLvoid* data )
{
  p_glBufferSubData( target, offset, size, data );
}

void GL::glGetBufferSubData( GLenum target, GLintptr offset, GLsizeiptr size, GLvoid* data )
{
  p_glGetBufferSubData( target, offset, size, data );
}

void GL::glGetCompressedTexImage( GLenum target, GLint lod, GLvoid* data )
{
  p_glGetCompressedTexImage( target, lod, data );
}

const char* GL::gluErrorString( GLenum error )
{
  switch( error )
  {
    case GL_NO_ERROR:
      return "No error";
    case GL_ILWALID_ENUM:
      return "Invalid enum";
    case GL_ILWALID_VALUE:
      return "Invalid value";
    case GL_ILWALID_OPERATION:
      return "Invalid operation";
    case GL_STACK_OVERFLOW:
      return "Stack overflow";
    case GL_STACK_UNDERFLOW:
      return "Stack underflow";
    case GL_OUT_OF_MEMORY:
      return "Out of memory";
    case GL_TABLE_TOO_LARGE:
      return "Table too large";
    default:
      return "Unknown GL error";
  }
}
