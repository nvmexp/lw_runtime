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

#include <Context/Context.h>
#include <Memory/GfxInteropResource.h>
#include <Memory/MBuffer.h>
#include <Util/ApiTime.h>
#include <prodlib/misc/TimeViz.h>

#include <corelib/system/Preprocessor.h>

#include <sstream>
#include <string>
#include <unordered_map>


// clang-format off
#define OAC_TRACE_START()                   try { getApiCapture().init(); TIMEVIZ_FUNC; API_TIME_SCOPE;
#define OAC_CAPTURE4( p0, p1, p2, p3 )      getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3 );
#define OAC_TRACE0()                        OAC_TRACE_START(); getApiCapture().trace( RTAPI_FUNC_NAME );
#define OAC_TRACE1( p0 )                    OAC_TRACE_START(); getApiCapture().trace( RTAPI_FUNC_NAME, p0 );
#define OAC_TRACE2( p0, p1 )                OAC_TRACE_START(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1 );
#define OAC_TRACE3( p0, p1, p2 )            OAC_TRACE_START(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2 );
#define OAC_TRACE4( p0, p1, p2, p3 )        OAC_TRACE_START(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3 );
#define OAC_TRACE5( p0, p1, p2, p3, p4 )    OAC_TRACE_START(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4 );
#define OAC_TRACE6( p0, p1, p2, p3, p4, p5 )    OAC_TRACE_START(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5 );
#define OAC_TRACE7( p0, p1, p2, p3, p4, p5, p6 )    OAC_TRACE_START(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6 );
#define OAC_TRACE8( p0, p1, p2, p3, p4, p5, p6, p7 )    OAC_TRACE_START(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6, p7 );
#define OAC_TRACE9( p0, p1, p2, p3, p4, p5, p6, p7, p8 )    OAC_TRACE_START(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6, p7, p8 );
#define OAC_TRACE10( p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 )    OAC_TRACE_START(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 );
#define OAC_TRACE11( p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 )    OAC_TRACE_START(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 );
#define OAC_TRACE12( p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 )    OAC_TRACE_START(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 );
#define OAC_TRACE13( p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12 )    OAC_TRACE_START(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12 );
#define OAC_TRACE14( p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13 )    OAC_TRACE_START(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13 );
#define OAC_TRACE_START_NOTIMEVIZ()                   try { getApiCapture().init();
#define OAC_TRACE0_NOTIMEVIZ()                        OAC_TRACE_START_NOTIMEVIZ(); getApiCapture().trace( RTAPI_FUNC_NAME );
#define OAC_TRACE1_NOTIMEVIZ( p0 )                    OAC_TRACE_START_NOTIMEVIZ(); getApiCapture().trace( RTAPI_FUNC_NAME, p0 );
#define OAC_TRACE2_NOTIMEVIZ( p0, p1 )                OAC_TRACE_START_NOTIMEVIZ(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1 );
#define OAC_TRACE3_NOTIMEVIZ( p0, p1, p2 )            OAC_TRACE_START_NOTIMEVIZ(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2 );
#define OAC_TRACE4_NOTIMEVIZ( p0, p1, p2, p3 )        OAC_TRACE_START_NOTIMEVIZ(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3 );
#define OAC_TRACE5_NOTIMEVIZ( p0, p1, p2, p3, p4 )    OAC_TRACE_START_NOTIMEVIZ(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4 );
#define OAC_TRACE6_NOTIMEVIZ( p0, p1, p2, p3, p4, p5 )    OAC_TRACE_START_NOTIMEVIZ(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5 );
#define OAC_TRACE7_NOTIMEVIZ( p0, p1, p2, p3, p4, p5, p6 )    OAC_TRACE_START_NOTIMEVIZ(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6 );
#define OAC_TRACE8_NOTIMEVIZ( p0, p1, p2, p3, p4, p5, p6, p7 )    OAC_TRACE_START_NOTIMEVIZ(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6, p7 );
#define OAC_TRACE9_NOTIMEVIZ( p0, p1, p2, p3, p4, p5, p6, p7, p8 )    OAC_TRACE_START_NOTIMEVIZ(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6, p7, p8 );
#define OAC_TRACE10_NOTIMEVIZ( p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 )    OAC_TRACE_START_NOTIMEVIZ(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 );
#define OAC_TRACE11_NOTIMEVIZ( p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 )    OAC_TRACE_START_NOTIMEVIZ(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 );
#define OAC_TRACE12_NOTIMEVIZ( p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 )    OAC_TRACE_START_NOTIMEVIZ(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 );
#define OAC_TRACE13_NOTIMEVIZ( p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12 )    OAC_TRACE_START_NOTIMEVIZ(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12 );
#define OAC_TRACE14_NOTIMEVIZ( p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13 )    OAC_TRACE_START_NOTIMEVIZ(); getApiCapture().trace( RTAPI_FUNC_NAME, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p13 );

#define OAC_RESULT( p0 )                    { if (getApiCapture().capture_enabled()) { getApiCapture().capture( "  res = " + to_string(p0) + "\n" ); } }
#define OAC_NEW_HANDLE( p0 )                { if (getApiCapture().capture_enabled()) { getApiCapture().capture( "  hdl = " + to_string(p0) + "\n" ); } }
#define OAC_ID( p0 )                        { if (getApiCapture().capture_enabled()) { getApiCapture().capture( "  id = " + to_string(p0) + "\n" ); } }
#define OAC_VALUES( n, p0 )                 { if (getApiCapture().capture_enabled()) { getApiCapture().capture_values( n, p0, "val" ); } }
#define OAC_VALUES2( n, p0 )                { if (getApiCapture().capture_enabled()) { getApiCapture().capture_values( n, p0, "val2" ); } }
#define OAC_BUFFER( n, p0 )                 { if (getApiCapture().capture_enabled()) { getApiCapture().capture_buffer( n, p0, "file", "buf" ); } }
#define OAC_COMMENT( c )                    { if (getApiCapture().capture_enabled()) { getApiCapture().capture( std::string("  // ")+(c)+"\n" ); } }
#define OAC_RETURN( result )                return result; } catch (...) { return RT_ERROR_UNKNOWN; }
#define OAC_NORETURN( code )                } catch (...) { code; }
// clang-format on


struct IDirect3DResource9;
struct ID3D10Resource;
struct ID3D11Resource;

namespace optix {

class ApiCapture
{
  public:
    ApiCapture();

    // Initialize capture.
    void init();

    // Capture and log a string.
    void capture_and_log( const std::string& str );

    // Capture a string.
    void capture( const std::string& str );

    // Capture the content of a buffer. Returns the filename
    // (without path) in which the buffer was captured.
    std::string capture_buffer( size_t sz, const void* ptr, const char* tag, const char* grouping, const char* comment = nullptr );

    // Capture PTX. Returns the filename (without path) in
    // which the string was captured.
    std::string capture_ptx( const char* ptx );

    // Capture multiple PTX strings for the same program
    void capture_ptx_strings( unsigned int n, const char** ptx_strings );

    // Capture a function call trace and its parameters.
    void trace( const char* func )
    {
        if( !m_capture_enabled && !m_log_enabled )
            return;
        std::ostringstream ss;
        ss << func << " ()\n";
        capture_and_log( ss.str() );
    }
    template <typename T, typename... Ts>
    void trace( const char* func, const T& p, const Ts&... rest )
    {
        if( !m_capture_enabled && !m_log_enabled )
            return;

        using unpack = int[];
        std::ostringstream ss;
        ss << func << "( " << p;
        (void)unpack{0, ( ( ss << ", " << rest ), 0 )...};
        ss << " )\n";
        capture_and_log( ss.str() );
    }

    // Capture a number of values of type T.
    template <typename T>
    void capture_values( int n, const T* values, const char* tag )
    {
        if( !m_capture_enabled || !values )
            return;
        std::stringstream ss;
        for( int i = 0; i < n; ++i )
            ss << " " << values[i];
        capture( "  " + std::string( tag ) + " =" + ss.str() + "\n" );
    }

    // Return whether API capture is turned on.
    bool capture_enabled() const { return m_capture_enabled; }

    //
    // LWCA interop
    //

    // Capture the contents of all LWCA input buffers in the context.
    void capture_lwda_input_buffers( Context* context );

    // Capture LWCA buffer size (not its content).
    void capture_lwda_buffer_size( Buffer* buffer );

    // Capture LWCA buffer address (not its content).
    void capture_lwda_buffer_address( void* address );

    //
    // GL interop
    //

    // Capture the contents of all gl input buffers in the context.
    void capture_gl_input_buffers( Context* context );

    // Capture the contents of all gl images in the context.
    void capture_gl_images( Context* context );

    // Capture creation data for a gl buffer (not its content).
    void capture_gl_buffer_info( unsigned int gl_id );

    // Capture creation data for a gl image (not its content).
    void capture_gl_image_info( unsigned int gl_id, RTgltarget target );

    //
    // D3D interop
    //

    // Capture the contents of all d3d9 input buffers in the context.
    void capture_d3d9_input_buffers( Context* context );

    // Capture the contents of all d3d9 images in the context.
    void capture_d3d9_images( Context* context );

    // Capture creation data for a d3d9 buffer (not its contents).
    void capture_d3d9_buffer_info( IDirect3DResource9* resource );

    // Capture creation data for a d3d9 image (not its contents).
    void capture_d3d9_image_info( IDirect3DResource9* resource );

  private:
    // Create the capture target directory.
    bool create_capture_dir( const std::string& elw );

    // Capture system information.
    void sys_info();

    // Find a unique filename within the capture directory. 'grouping' is
    // just a string that will be part of the filename, so it's easier
    // to gain an overview of the contents of a trace directory.
    std::string unique_filename( const char* grouping );

    // Find a unique directory name for the capture.
    std::string unique_dirname();

    // Capture the contents of a LWCA interop object.
    void capture_lwda_data( Context* context, const MBufferHandle& buffer, const char* grouping );

    // Capture the contents of a GL interop object.
    void capture_gl_data( Context* context, const MBufferHandle& buffer, unsigned int glid, const char* subtag, const char* comment = nullptr );

    static const int LogLevel = 50;  // level on which API calls are logged

    typedef std::unordered_map<std::string, std::string> PtxMap;

    bool        m_capture_enabled;  // full capture, enabled through elwvar
    bool        m_log_enabled;      // log calls, enabled if not release build
    bool        m_initialized;      // one time init done?
    int         m_nextFileNumber;   // what file number to try next
    std::string m_dir;              // capture directory
    PtxMap      m_ptx_map;          // map from ptx -> captured filename
};

// Singleton access.
ApiCapture& getApiCapture();

}  // namespace
