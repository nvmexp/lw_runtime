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

#include <c-api/ApiCapture.h>

#include <LWCA/Memory.h>
#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Memory/MemoryManager.h>
#include <Objects/Buffer.h>
#include <Objects/TextureSampler.h>
#include <Util/LWML.h>

#include <prodlib/misc/Encryption.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>
#include <prodlib/system/System.h>

#include <corelib/misc/String.h>
#include <corelib/system/LwdaDriver.h>
#include <corelib/system/Preprocessor.h>
#include <corelib/system/System.h>

#include <private/optix_version_string.h>

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace optix;
using namespace prodlib;
using namespace corelib;


ApiCapture::ApiCapture()
    : m_capture_enabled( false )
    , m_log_enabled( false )
    , m_initialized( false )
    , m_nextFileNumber( 0 )
{
}

void ApiCapture::init()
{
    if( m_initialized )
        return;
    m_initialized = true;

// Check if logging is enabled (available in non-public builds only).
#if defined( OPTIX_ENABLE_LOGGING )
    m_log_enabled = log::active( LogLevel );
#endif

    //
    // NOTE: API capture is disabled in public builds for now, until we
    // decide on if/how we want to expose it.
    //

    // Check if capture is enabled.
    const char* elw   = getelw( "OPTIX_API_CAPTURE" );
    m_capture_enabled = elw != nullptr;

    if( m_capture_enabled )
    {
        if( !create_capture_dir( elw ) )
        {
            m_capture_enabled = false;
            return;
        }

        // Capture the trace version. Increase this number whenever
        // the trace format becomes incompatible for the replayer.
        //
        // Changelog:
        //    5 ->  6      capture ID values on rtXXXGetId()
        //      -> 10      pre-Goldenrod -> Goldenrod
        //      -> 12      encrypt buffers on disk
        //      -> 13      support for BC textures
        //      -> 14      support for Triangle API
        //      -> 15      support for Triangle API motion blur
        //      -> 16      capture buffer for rtContextGetAttribute()
        //      -> 17      modified Triangle API
        //
        capture( "17\n" );

        // capture bitness
        capture( to_string( buildBitness() ) + "\n" );

        // capture info about this system
        sys_info();

        // End delimiter.
        capture( "%%\n" );

        // capture optix.props if there is one
        if( fileExists( KnobRegistry::getOptixPropsLocation().c_str() ) )
            copyFile( KnobRegistry::getOptixPropsLocation().c_str(), ( m_dir + "/optix.props" ).c_str() );
    }
}

void ApiCapture::capture_and_log( const std::string& str )
{
    capture( str );
    llog( LogLevel ) << "OAC: " << str;
}

void ApiCapture::capture( const std::string& str )
{
    if( !m_capture_enabled )
        return;
    const std::string fname = m_dir + "/trace.oac";
    FILE*             fp    = fopen( fname.c_str(), "a" );
    if( !fp )
    {
        lerr << "OptiX API capture: error opening \"" << fname << "\"\n";
        return;
    }
    const size_t sz = str.size();
    if( fwrite( str.c_str(), 1, sz, fp ) != sz )
    {
        lerr << "OptiX API capture: fwrite error trying to write " << sz << " bytes\n";
    }
    fclose( fp );
}

std::string ApiCapture::capture_buffer( size_t sz, const void* ptr, const char* tag, const char* grouping, const char* comment )
{
    if( !m_capture_enabled || !ptr )
        return "";
    const std::string dst     = unique_filename( grouping );
    const std::string dstpath = m_dir + "/" + dst;
    FILE*             fp      = fopen( dstpath.c_str(), "wb" );
    if( !fp )
    {
        lerr << "OptiX API capture: error opening \"" << dstpath << "\"\n";
        return "";
    }

    // optionally encrypt
    std::vector<unsigned char> scratch;
    const unsigned char*       bytes = reinterpret_cast<const unsigned char*>( ptr );

    const bool do_encrypt = ( std::string( grouping ) != "ptx" );
    if( do_encrypt )
    {
        scratch.assign( bytes, bytes + sz );

        // Note: same key must be used when decrypting in oaclib
        static const uint32_t key[4] = {0x3a984218, 0x2b10f73c, 0x2312bacb, 0x1c0c2670};
        prodlib::tea_encrypt( scratch.data(), sz, key );
        bytes = scratch.data();
    }

    if( fwrite( bytes, 1, sz, fp ) != sz )
    {
        lerr << "OptiX API capture: fwrite error trying to write " << sz << " bytes\n";
    }
    fclose( fp );

    std::string capstr = std::string( "  " ) + tag + " = " + dst;
    if( comment )
        capstr += std::string( " // " ) + comment;
    capture( capstr + "\n" );
    return dst;
}

std::string ApiCapture::capture_ptx( const char* ptx )
{
    if( !m_capture_enabled || !ptx )
        return "";

    // Check if we've already captured this PTX, if so, point the metadata to the existing file
    PtxMap::iterator it = m_ptx_map.find( ptx );
    if( it != m_ptx_map.end() )
    {
        capture( std::string( "  file = " ) + it->second + "\n" );
        return it->second;
    }
    else
    {
        // Capture and fill map.
        const std::string fname = capture_buffer( strlen( ptx ), ptx, "file", "ptx" );
        m_ptx_map.insert( std::make_pair( std::string( ptx ), fname ) );
        return fname;
    }
}

void ApiCapture::capture_ptx_strings( unsigned int n, const char** ptx_strings )
{
    if( !m_capture_enabled || n == 0 || !ptx_strings )
        return;

    for( unsigned int i = 0; i < n; ++i )
    {
        const char* ptx = ptx_strings[i];

        // Check if we've already captured this PTX, if so, point the metadata to the existing file
        PtxMap::iterator it       = m_ptx_map.find( ptx );
        std::string      filename = "file" + to_string( i );
        if( it != m_ptx_map.end() )
        {
            capture( std::string( "  " ) + filename + " = " + it->second + "\n" );
        }
        else
        {
            // Capture and fill map.
            const std::string fname = capture_buffer( strlen( ptx ), ptx, filename.c_str(), "ptx" );
            m_ptx_map.insert( std::make_pair( std::string( ptx ), fname ) );
        }
    }
}

void ApiCapture::capture_lwda_buffer_size( Buffer* buffer )
{
    if( !m_capture_enabled )
        return;

    if( buffer->getMipLevelCount() > 1 )
    {
        lerr << "Optix API capture: LWCA interop doesn't support mipmaps\n";
    }

    capture( "  size = " + to_string( buffer->getLevelSizeInBytes( 0 ) ) + "\n" );
}


void ApiCapture::capture_lwda_buffer_address( void* address )
{
    if( !m_capture_enabled )
        return;

    capture( "  captime_address = " + to_string( address ) + "\n" );
}


void ApiCapture::capture_lwda_input_buffers( Context* context )
{
    if( !m_capture_enabled || !context )
        return;

    // get all buffers from context
    for( const auto& buffer : context->getObjectManager()->getBuffers() )
    {
        // if it's a LWCA interop buffer, dump it
        if( buffer->isLwdaInterop() && ( buffer->getType() & RT_BUFFER_INPUT ) )
        {
            capture_lwda_data( context, buffer->getMBuffer(), "LWDAbuf" );
        }
    }
}

void ApiCapture::capture_gl_input_buffers( Context* context )
{
    if( !m_capture_enabled || !context )
        return;


    // get all buffers from context
    for( const auto& buffer : context->getObjectManager()->getBuffers() )
    {
        // if it's a gl input buffer, dump it
        if( buffer->getGfxInteropResource().kind == GfxInteropResource::OGL_BUFFER_OBJECT &&  // skip images
            buffer->getType() & RT_BUFFER_INPUT )
        {
            capture_gl_data( context, buffer->getMBuffer(), buffer->getGfxInteropResource().gl.glId, "glbuf", nullptr );
        }
    }
}

void ApiCapture::capture_gl_images( Context* context )
{
    if( !m_capture_enabled || !context )
        return;

    // get all texture samplers from context
    for( const auto& tex : context->getObjectManager()->getTextureSamplers() )
    {
        // if it's a gl interop texture, dump it
        if( tex->isInteropTexture() )
        {
            GfxInteropResource resource( tex->getGfxInteropResource() );
            if( resource.isOGL() )
                capture_gl_data( context, tex->getBackingMBuffer(), resource.gl.glId, "gltex", nullptr );
        }
    }
}

void ApiCapture::capture_gl_buffer_info( unsigned int glId )
{
    if( !m_capture_enabled )
        return;

    GfxInteropResource             resource( GfxInteropResource::OGL_BUFFER_OBJECT, glId );
    GfxInteropResource::Properties props;
    resource.queryProperties( &props );

    capture( "  size = " + to_string( props.width ) + "\n" );
    capture( "  usage = " + to_string( props.glBufferUsage ) + "\n" );
}

void ApiCapture::capture_gl_image_info( unsigned int glId, RTgltarget target )
{
    if( !m_capture_enabled )
        return;

    const GfxInteropResource::ResourceKind kind =
        target == RT_TARGET_GL_RENDER_BUFFER ? GfxInteropResource::OGL_RENDER_BUFFER : GfxInteropResource::OGL_TEXTURE;
    GfxInteropResource             resource( kind, glId, target );
    GfxInteropResource::Properties props;
    resource.queryProperties( &props );

    capture( "  internalformat = " + to_string( props.glInternalFormat ) + "\n" );
    capture( "  width = " + to_string( props.width ) + "\n" );
    capture( "  height = " + to_string( props.height ) + "\n" );
    capture( "  depth = " + to_string( props.depth ) + "\n" );
    capture( "  levels = " + to_string( props.levelCount ) + "\n" );
    capture( "  lwbe = " + to_string( (int)props.lwbe ) + "\n" );
    capture( "  layered = " + to_string( (int)props.layered ) + "\n" );
}

// Use stubs for these, because we don't support DX interop in 4.x.
void ApiCapture::capture_d3d9_input_buffers( Context* context )
{
}
void ApiCapture::capture_d3d9_images( Context* context )
{
}
void ApiCapture::capture_d3d9_buffer_info( IDirect3DResource9* resource )
{
}
void ApiCapture::capture_d3d9_image_info( IDirect3DResource9* resource )
{
}

bool ApiCapture::create_capture_dir( const std::string& elw )
{
    // If the elw var is set to "1", auto-create a free directory name.
    // Otherwise, use the elw var value as trace directory name.
    m_dir = elw == "1" ? unique_dirname() : elw;

    if( !createDir( m_dir.c_str() ) )
    {
        lerr << "OptiX API capture: could not create trace directory \"" << m_dir << "\"\n";
        return false;
    }
    return true;
}

void ApiCapture::sys_info()
{
// Capture some general info.
// TODO: more info on OS, driver, etc
// TODO: linux/mac path

// Platform.
#ifdef _WIN32
    capture( "Platform: Windows\n" );
#elif defined( __APPLE__ )
    capture( "Platform: Apple\n" );
#else
    capture( "Platform: Linux\n" );
#endif

    // Driver version
    capture( "Driver Version: " + LWML::driverVersion() + "\n" );

    // Optix build description
    capture( std::string( OPTIX_BUILD_DESCRIPTION ) + "\n" );

// Command line.
#ifdef _WIN32
    capture( "Command line: " + std::string( GetCommandLine() ) + "\n" );
#endif

    // Timestamp.
    capture( "Capture time: " + getTimestamp() + "\n" );
}

std::string ApiCapture::unique_filename( const char* grouping )
{
    // Note: we use the .potx extension only because it's part of our p4's type
    // map and maps to binary. Any extension that is mapped to binary would do.
    // If we didn't consider this, one would have to manually ensure that trace
    // files are checked in as binary, which is too error prone.
    char name[512];
    for( int i = m_nextFileNumber;; ++i )
    {
        sprintf( name, "oac.%s.%06d.potx", grouping, i );
        const std::string path = m_dir + "/" + name;
        if( !fileExists( path.c_str() ) )
        {
            m_nextFileNumber = i;
            break;
        }
    }
    return std::string( name );
}

std::string ApiCapture::unique_dirname()
{
    char name[512];
    for( int i = 0;; ++i )
    {
        sprintf( name, "oac%05d", i );
        if( !dirExists( name ) )
            break;
    }
    return std::string( name );
}

void ApiCapture::capture_gl_data( Context* context, const MBufferHandle& buffer, const unsigned int glid, const char* subtag, const char* comment )
{
    const void*  data = context->getMemoryManager()->mapToHost( buffer, MAP_READ );
    const size_t size = buffer->getDimensions().getTotalSizeInBytes();
    capture_buffer( size, data, ( std::string( "file::" ) + subtag + "::" + to_string( glid ) ).c_str(), subtag, comment );
    context->getMemoryManager()->unmapFromHost( buffer );
}

void ApiCapture::capture_lwda_data( Context* context, const MBufferHandle& buffer, const char* grouping )
{
    RT_ASSERT( buffer->getDimensions().mipLevelCount() == 1 );
    MemoryManager*    mm      = context->getMemoryManager();
    DeviceSet         devices = mm->getLwdaInteropDevices( buffer );
    std::vector<char> hostmem( buffer->getDimensions().getLevelSizeInBytes( 0 ) );
    for( const auto devIdx : devices )
    {
        Device* device = context->getDeviceManager()->allDevices()[devIdx];
        deviceCast<LWDADevice>( device )->makeLwrrent();

        void* devPtr = mm->getLwdaInteropPointer( buffer, device );
        optix::lwca::memcpyDtoH( hostmem.data(), (LWdeviceptr)devPtr, hostmem.size() );

        // NOTE that we use device_ptr as ID here, in hex format
        std::string device_ptr_string = ptr_to_string( devPtr, buildBitness() );
        std::string tag               = std::string( "file::" ) + std::string( grouping ) + "::" + device_ptr_string;
        capture_buffer( hostmem.size(), hostmem.data(), tag.c_str(), grouping );
    }
}

ApiCapture& optix::getApiCapture()
{
    static ApiCapture a;
    return a;
}
