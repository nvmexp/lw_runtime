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

//
// Entry points and Optix API Capture wrapper
//

// clang-format off
#ifndef RTAPI
#  if defined( _WIN32 )
#    define RTAPI __declspec(dllexport)
#  elif defined( __linux__ ) || defined ( __CYGWIN__ )
#    define RTAPI __attribute__ ((visibility ("default")))
#  elif defined( __APPLE__ ) && defined( __MACH__ )
#    define RTAPI __attribute__ ((visibility ("default")))
#  else
#    error "CODE FOR THIS OS HAS NOT YET BEEN DEFINED"
#  endif
#endif
// clang-format on

#include <o6/optix.h>

#include <c-api/rtapi.h>

#include <Objects/TextureSampler.h>
#include <Util/Enum2String.h>
#include <c-api/ApiCapture.h>

#include <corelib/misc/String.h>
#include <corelib/system/System.h>
#include <prodlib/system/Logger.h>

using namespace optix;
using namespace prodlib;
using namespace corelib;


extern "C" {

RTresult RTAPI rtAccelerationCreate( RTcontext context_api, RTacceleration* acceleration )
{
    OAC_TRACE2( context_api, acceleration );
    const RTresult _res = _rtAccelerationCreate( context_api, acceleration );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *acceleration );
    OAC_RETURN( _res );
}

RTresult RTAPI rtAccelerationDestroy( RTacceleration acceleration_api )
{
    OAC_TRACE1( acceleration_api );
    const RTresult _res = _rtAccelerationDestroy( acceleration_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtAccelerationGetBuilder( RTacceleration acceleration_api, const char** return_string )
{
    OAC_TRACE2( acceleration_api, return_string );
    const RTresult _res = _rtAccelerationGetBuilder( acceleration_api, return_string );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtAccelerationGetContext( RTacceleration acceleration_api, RTcontext* c )
{
    OAC_TRACE2( acceleration_api, c );
    const RTresult _res = _rtAccelerationGetContext( acceleration_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtAccelerationGetData( RTacceleration acceleration_api, void* data )
{
    OAC_TRACE2( acceleration_api, data );
    const RTresult _res = _rtAccelerationGetData( acceleration_api, data );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtAccelerationGetDataSize( RTacceleration acceleration_api, RTsize* size )
{
    OAC_TRACE2( acceleration_api, size );
    const RTresult _res = _rtAccelerationGetDataSize( acceleration_api, size );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtAccelerationGetProperty( RTacceleration acceleration_api, const char* name, const char** return_string )
{
    OAC_TRACE3( acceleration_api, name, return_string );
    const RTresult _res = _rtAccelerationGetProperty( acceleration_api, name, return_string );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtAccelerationGetTraverser( RTacceleration acceleration_api, const char** return_string )
{
    OAC_TRACE2( acceleration_api, return_string );
    const RTresult _res = _rtAccelerationGetTraverser( acceleration_api, return_string );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtAccelerationIsDirty( RTacceleration acceleration_api, int* dirty )
{
    OAC_TRACE2( acceleration_api, dirty );
    const RTresult _res = _rtAccelerationIsDirty( acceleration_api, dirty );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtAccelerationMarkDirty( RTacceleration acceleration_api )
{
    OAC_TRACE1( acceleration_api );
    const RTresult _res = _rtAccelerationMarkDirty( acceleration_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtAccelerationSetBuilder( RTacceleration acceleration_api, const char* builder )
{
    OAC_TRACE2( acceleration_api, builder );
    const RTresult _res = _rtAccelerationSetBuilder( acceleration_api, builder );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtAccelerationSetData( RTacceleration acceleration_api, const void* data, RTsize size )
{
    OAC_TRACE3( acceleration_api, data, size );
    OAC_BUFFER( size, data );
    const RTresult _res = _rtAccelerationSetData( acceleration_api, data, size );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtAccelerationSetProperty( RTacceleration acceleration_api, const char* name, const char* value )
{
    OAC_TRACE3( acceleration_api, name, value );
    const RTresult _res = _rtAccelerationSetProperty( acceleration_api, name, value );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtAccelerationSetTraverser( RTacceleration acceleration_api, const char* traverser )
{
    OAC_TRACE2( acceleration_api, traverser );
    const RTresult _res = _rtAccelerationSetTraverser( acceleration_api, traverser );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtAcceleratiolwalidate( RTacceleration acceleration_api )
{
    OAC_TRACE1( acceleration_api );
    const RTresult _res = _rtAcceleratiolwalidate( acceleration_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferCreate( RTcontext context_api, unsigned int type, RTbuffer* buffer )
{
    OAC_TRACE3( context_api, type, buffer );
    OAC_COMMENT( bufferdesc2string( type ) );
    const RTresult _res = _rtBufferCreate( context_api, type, buffer );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *buffer );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferCreateForLWDA( RTcontext context_api, unsigned int type, RTbuffer* buffer )
{
    OAC_TRACE3( context_api, type, buffer );
    OAC_COMMENT( bufferdesc2string( type ) );
    RTresult _res = _rtBufferCreateForLWDA( context_api, type, buffer );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *buffer );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferCreateFromGLBO( RTcontext context_api, unsigned int type, unsigned int gl_id, RTbuffer* buffer )
{
    OAC_TRACE4( context_api, type, gl_id, buffer );
    OAC_COMMENT( bufferdesc2string( type ) );
    const RTresult _res = _rtBufferCreateFromGLBO( context_api, type, gl_id, buffer );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *buffer );

    if( getApiCapture().capture_enabled() && _res == RT_SUCCESS )
        getApiCapture().capture_gl_buffer_info( gl_id );

    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferCreateFromCallback( RTcontext context_api, unsigned int bufferDesc, RTbuffercallback callback, void* callbackContext, RTbuffer* buffer )
{
    OAC_TRACE4( context_api, bufferDesc, reinterpret_cast<unsigned long long>( callback ),
                reinterpret_cast<unsigned long long>( callbackContext ) );
    OAC_COMMENT( bufferdesc2string( bufferDesc ) );
    const RTresult res = _rtBufferCreateFromCallback( context_api, bufferDesc, callback, callbackContext, buffer );
    OAC_RESULT( res );
    OAC_NEW_HANDLE( *buffer );
    OAC_RETURN( res );
}

RTresult RTAPI rtBufferSetDevicePointer( RTbuffer buffer_api, int optix_device_ordinal, void* device_pointer )
{
    OAC_TRACE3( buffer_api, optix_device_ordinal, device_pointer );
    RTresult _res = _rtBufferSetDevicePointer( buffer_api, optix_device_ordinal, device_pointer );
    OAC_RESULT( _res );

    if( getApiCapture().capture_enabled() && _res == RT_SUCCESS )
    {
        Buffer* buffer = (Buffer*)buffer_api;
        getApiCapture().capture_lwda_buffer_size( buffer );
    }

    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferMarkDirty( RTbuffer buffer_api )
{
    OAC_TRACE1( buffer_api );
    RTresult _res = _rtBufferMarkDirty( buffer_api );
    OAC_RESULT( _res );

    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferDestroy( RTbuffer buffer_api )
{
    OAC_TRACE1( buffer_api );
    const RTresult _res = _rtBufferDestroy( buffer_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGLRegister( RTbuffer buffer_api )
{
    OAC_TRACE1( buffer_api );
    const RTresult _res = _rtBufferGLRegister( buffer_api );
    OAC_RESULT( _res );

    if( getApiCapture().capture_enabled() && _res == RT_SUCCESS )
    {
        Buffer* buffer = (Buffer*)buffer_api;
        if( buffer && buffer->getClass() == RT_OBJECT_BUFFER )
        {
            const GfxInteropResource& resource = buffer->getGfxInteropResource();
            getApiCapture().capture_gl_buffer_info( resource.gl.glId );
        }
    }

    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGLUnregister( RTbuffer buffer_api )
{
    OAC_TRACE1( buffer_api );
    const RTresult _res = _rtBufferGLUnregister( buffer_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetDevicePointer( RTbuffer buffer_api, int optix_device_ordinal, void** device_pointer )
{
    OAC_TRACE3( buffer_api, optix_device_ordinal, device_pointer );
    RTresult _res = _rtBufferGetDevicePointer( buffer_api, optix_device_ordinal, device_pointer );
    if( getApiCapture().capture_enabled() && _res == RT_SUCCESS )
    {
        // It's just oclwred to me that we could capture this data by capturing the call info like this (note the deref in
        // the last param):
        // OAC_TRACE3( buffer_api, optix_device_ordinal, *device_pointer );
        // But this 1) feels a little too obfuscated for my liking,
        // and 2) it would have to be called _after_ the actual API call for the pointer to be valid,
        // (which would be inconsistent with all the other captures), so I'm Keeping It Simple for now.
        getApiCapture().capture_lwda_buffer_address( *device_pointer );
    }
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}


RTresult RTAPI rtBufferGetContext( RTbuffer buffer_api, RTcontext* c )
{
    OAC_TRACE2( buffer_api, c );
    const RTresult _res = _rtBufferGetContext( buffer_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetDimensionality( RTbuffer buffer_api, unsigned int* dimensionality )
{
    OAC_TRACE2( buffer_api, dimensionality );
    const RTresult _res = _rtBufferGetDimensionality( buffer_api, dimensionality );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetElementSize( RTbuffer buffer_api, RTsize* size_of_element )
{
    OAC_TRACE2( buffer_api, size_of_element );
    const RTresult _res = _rtBufferGetElementSize( buffer_api, size_of_element );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetFormat( RTbuffer buffer_api, RTformat* format )
{
    OAC_TRACE2( buffer_api, format );
    const RTresult _res = _rtBufferGetFormat( buffer_api, format );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetGLBOId( RTbuffer buffer_api, unsigned int* gl_id )
{
    OAC_TRACE2( buffer_api, gl_id );
    const RTresult _res = _rtBufferGetGLBOId( buffer_api, gl_id );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetId( RTbuffer buffer_api, int* buffer_id )
{
    OAC_TRACE2( buffer_api, buffer_id );
    const RTresult _res = _rtBufferGetId( buffer_api, buffer_id );
    OAC_RESULT( _res );
    OAC_ID( *buffer_id );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetMipLevelSize1D( RTbuffer buffer_api, unsigned int level, RTsize* width )
{
    OAC_TRACE3( buffer_api, level, width );
    const RTresult _res = _rtBufferGetMipLevelSize1D( buffer_api, level, width );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetMipLevelSize2D( RTbuffer buffer_api, unsigned int level, RTsize* width, RTsize* height )
{
    OAC_TRACE4( buffer_api, level, width, height );
    const RTresult _res = _rtBufferGetMipLevelSize2D( buffer_api, level, width, height );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetMipLevelSize3D( RTbuffer buffer_api, unsigned int level, RTsize* width, RTsize* height, RTsize* depth )
{
    OAC_TRACE5( buffer_api, level, width, height, depth );
    const RTresult _res = _rtBufferGetMipLevelSize3D( buffer_api, level, width, height, depth );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetMipLevelCount( RTbuffer buffer_api, unsigned int* level )
{
    OAC_TRACE2( buffer_api, level );
    const RTresult _res = _rtBufferGetMipLevelCount( buffer_api, level );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetSize1D( RTbuffer buffer_api, RTsize* width )
{
    OAC_TRACE2( buffer_api, width );
    const RTresult _res = _rtBufferGetSize1D( buffer_api, width );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetSize2D( RTbuffer buffer_api, RTsize* width, RTsize* height )
{
    OAC_TRACE3( buffer_api, width, height );
    const RTresult _res = _rtBufferGetSize2D( buffer_api, width, height );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetSize3D( RTbuffer buffer_api, RTsize* width, RTsize* height, RTsize* depth )
{
    OAC_TRACE4( buffer_api, width, height, depth );
    const RTresult _res = _rtBufferGetSize3D( buffer_api, width, height, depth );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetSizev( RTbuffer buffer_api, unsigned int maxdim, RTsize* outdims )
{
    OAC_TRACE3( buffer_api, maxdim, outdims );
    const RTresult _res = _rtBufferGetSizev( buffer_api, maxdim, outdims );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferMap( RTbuffer buffer_api, void** user_pointer )
{
    OAC_TRACE2( buffer_api, user_pointer );
    const RTresult _res = _rtBufferMap( buffer_api, user_pointer );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferMapEx( RTbuffer buffer_api, unsigned int map_flag, unsigned int level, void* user_owned, void** optix_owned )
{
    OAC_TRACE5( buffer_api, map_flag, level, user_owned, optix_owned );
    const RTresult _res = _rtBufferMapEx( buffer_api, map_flag, level, user_owned, optix_owned );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferSetElementSize( RTbuffer buffer_api, RTsize size_of_element )
{
    OAC_TRACE2( buffer_api, size_of_element );
    const RTresult _res = _rtBufferSetElementSize( buffer_api, size_of_element );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferSetFormat( RTbuffer buffer_api, RTformat type )
{
    OAC_TRACE2( buffer_api, type );
    OAC_COMMENT( format2string( type ) );
    const RTresult _res = _rtBufferSetFormat( buffer_api, type );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferSetMipLevelCount( RTbuffer buffer_api, unsigned int levels )
{
    OAC_TRACE2( buffer_api, levels );
    const RTresult _res = _rtBufferSetMipLevelCount( buffer_api, levels );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferSetSize1D( RTbuffer buffer_api, RTsize width )
{
    OAC_TRACE2( buffer_api, width );
    const RTresult _res = _rtBufferSetSize1D( buffer_api, width );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferSetSize2D( RTbuffer buffer_api, RTsize width, RTsize height )
{
    OAC_TRACE3( buffer_api, width, height );
    const RTresult _res = _rtBufferSetSize2D( buffer_api, width, height );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferSetSize3D( RTbuffer buffer_api, RTsize width, RTsize height, RTsize depth )
{
    OAC_TRACE4( buffer_api, width, height, depth );
    const RTresult _res = _rtBufferSetSize3D( buffer_api, width, height, depth );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferSetSizev( RTbuffer buffer_api, unsigned int dimensionality, const RTsize* indims )
{
    OAC_TRACE3( buffer_api, dimensionality, indims );
    OAC_VALUES( dimensionality, indims );
    const RTresult _res = _rtBufferSetSizev( buffer_api, dimensionality, indims );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferUnmap( RTbuffer buffer_api )
{
    OAC_TRACE1( buffer_api );

    // Capture buffer contents on unmap if this is an input buffer
    if( getApiCapture().capture_enabled() )
    {
        optix::Buffer* buffer = (optix::Buffer*)buffer_api;
        if( buffer && buffer->getClass() == RT_OBJECT_BUFFER && ( buffer->getType() & RT_BUFFER_INPUT ) && buffer->isMappedHost( 0 ) )
        {
            const size_t sz  = buffer->getTotalSizeInBytes();
            const void*  ptr = buffer->getMappedHostPtr( 0 );
            OAC_BUFFER( sz, ptr );
        }
    }

    const RTresult _res = _rtBufferUnmap( buffer_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferUnmapEx( RTbuffer buffer_api, unsigned int level )
{
    OAC_TRACE2( buffer_api, level );

    // Capture buffer contents on unmap if this is an input buffer
    if( getApiCapture().capture_enabled() )
    {
        {
            optix::Buffer* buffer = (optix::Buffer*)buffer_api;
            if( buffer && buffer->getClass() == RT_OBJECT_BUFFER && ( buffer->getType() & RT_BUFFER_INPUT )
                && buffer->isMappedHost( level ) )
            {
                const size_t sz  = buffer->getLevelSizeInBytes( level );
                const void*  ptr = buffer->getMappedHostPtr( level );
                OAC_BUFFER( sz, ptr );
            }
        }
    }

    const RTresult _res = _rtBufferUnmapEx( buffer_api, level );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferValidate( RTbuffer buffer_api )
{
    OAC_TRACE1( buffer_api );
    const RTresult _res = _rtBufferValidate( buffer_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetProgressiveUpdateReady( RTbuffer buffer_api, int* ready, unsigned int* subframe_count, unsigned int* max_subframes )
{
    // The call is too high frequency and bloats the timeviz log unnecessarily, do not add timeviz info.
    OAC_TRACE4_NOTIMEVIZ( buffer_api, ready, subframe_count, max_subframes );
    const RTresult _res = _rtBufferGetProgressiveUpdateReady( buffer_api, ready, subframe_count, max_subframes );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferBindProgressiveStream( RTbuffer buffer_api, RTbuffer source )
{
    OAC_TRACE2( buffer_api, source );
    const RTresult _res = _rtBufferBindProgressiveStream( buffer_api, source );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferSetAttribute( RTbuffer buffer_api, RTbufferattribute attrib, RTsize size, const void* p )
{
    OAC_TRACE4( buffer_api, attrib, size, p );
    OAC_BUFFER( size, p );
    const RTresult _res = _rtBufferSetAttribute( buffer_api, attrib, size, p );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetAttribute( RTbuffer buffer_api, RTbufferattribute attrib, RTsize size, void* p )
{
    OAC_TRACE4( buffer_api, attrib, size, p );
    const RTresult _res = _rtBufferGetAttribute( buffer_api, attrib, size, p );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtCommandListCreate( RTcontext context_api, RTcommandlist* list_api )
{
    OAC_TRACE2( context_api, list_api );
    RTresult _res = _rtCommandListCreate( context_api, list_api );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *list_api );
    OAC_RETURN( _res );
}

RTresult RTAPI rtCommandListDestroy( RTcommandlist list_api )
{
    OAC_TRACE1( list_api );
    RTresult _res = _rtCommandListDestroy( list_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtCommandListAppendPostprocessingStage( RTcommandlist list_api, RTpostprocessingstage stage_api, RTsize launch_width, RTsize launch_height )
{
    OAC_TRACE4( list_api, stage_api, launch_width, launch_height );
    RTresult _res = _rtCommandListAppendPostprocessingStage( list_api, stage_api, launch_width, launch_height );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtCommandListAppendLaunch1D( RTcommandlist list_api, unsigned int entry_point_index, RTsize launch_width )
{
    OAC_TRACE3( list_api, entry_point_index, launch_width );
    RTresult _res = _rtCommandListAppendLaunch1D( list_api, entry_point_index, launch_width );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtCommandListAppendLaunch2D( RTcommandlist list_api, unsigned int entry_point_index, RTsize launch_width, RTsize launch_height )
{
    OAC_TRACE4( list_api, entry_point_index, launch_width, launch_height );
    RTresult _res = _rtCommandListAppendLaunch2D( list_api, entry_point_index, launch_width, launch_height );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtCommandListAppendLaunch3D( RTcommandlist list_api, unsigned int entry_point_index, RTsize launch_width, RTsize launch_height, RTsize launch_depth )
{
    OAC_TRACE5( list_api, entry_point_index, launch_width, launch_height, launch_depth );
    RTresult _res = _rtCommandListAppendLaunch3D( list_api, entry_point_index, launch_width, launch_height, launch_depth );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtCommandListSetDevices( RTcommandlist list_api, unsigned int count, const int* devices )
{
    OAC_TRACE3( list_api, count, devices );
    OAC_VALUES( count, devices );
    const RTresult _res = _rtCommandListSetDevices( list_api, count, devices );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtCommandListGetDeviceCount( RTcommandlist list_api, unsigned int* count )
{
    OAC_TRACE2( list_api, count );
    const RTresult _res = _rtCommandListGetDeviceCount( list_api, count );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtCommandListGetDevices( RTcommandlist list_api, int* devices )
{
    OAC_TRACE2( list_api, devices );
    const RTresult _res = _rtCommandListGetDevices( list_api, devices );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtCommandListFinalize( RTcommandlist list_api )
{
    OAC_TRACE1( list_api );
    RTresult _res = _rtCommandListFinalize( list_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtCommandListExelwte( RTcommandlist list_api )
{
    OAC_TRACE1( list_api );
    RTresult _res = _rtCommandListExelwte( list_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtCommandListSetLwdaStream( RTcommandlist list_api, void* stream )
{
    OAC_TRACE2( list_api, stream );
    RTresult _res = _rtCommandListSetLwdaStream( list_api, stream );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtCommandListGetLwdaStream( RTcommandlist list_api, void** stream )
{
    OAC_TRACE2( list_api, stream );
    RTresult _res = _rtCommandListGetLwdaStream( list_api, stream );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtCommandListGetContext( RTcommandlist list_api, RTcontext* c )
{
    OAC_TRACE1( list_api );
    RTresult _res = _rtCommandListGetContext( list_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextCompile( RTcontext context_api )
{
    OAC_TRACE1( context_api );
    const RTresult _res = _rtContextCompile( context_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextCreateABI16( RTcontext* context )
{
    OAC_TRACE1( context );
    RTresult _res = _rtContextCreateABI16( context );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *context );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextCreateABI17( RTcontext* context )
{
    OAC_TRACE1( context );
    RTresult _res = _rtContextCreateABI17( context );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *context );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextCreate( RTcontext* context )
{
    OAC_TRACE1( context );
    RTresult _res = _rtContextCreateABI18( context );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *context );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextDeclareVariable( RTcontext context_api, const char* name, RTvariable* v )
{
    OAC_TRACE3( context_api, name, v );
    const RTresult _res = _rtContextDeclareVariable( context_api, name, v );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *v );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextDestroy( RTcontext context_api )
{
    OAC_TRACE1( context_api );
    const RTresult _res = _rtContextDestroy( context_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetAttribute( RTcontext context_api, RTcontextattribute attrib, RTsize size, const void* p )
{
    OAC_TRACE_START();
    RTcontextattribute recorded_attrib = ( static_cast<int>( attrib ) == 0x31415926 ) ? RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS : attrib;
    if( attrib == RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION )
    {
        // Capture strings directly in the trace
        std::string str( reinterpret_cast<const char*>( p ) );
        OAC_CAPTURE4( context_api, recorded_attrib, size, str.c_str() );
    }
    else
    {
        OAC_CAPTURE4( context_api, recorded_attrib, size, p );
        OAC_BUFFER( size, p );
    }
    const RTresult _res = _rtContextSetAttribute( context_api, attrib, size, p );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetAttribute( RTcontext context_api, RTcontextattribute attrib, RTsize size, void* p )
{
    OAC_TRACE4( context_api, attrib, size, p );
    const RTresult _res = _rtContextGetAttribute( context_api, attrib, size, p );
    OAC_RESULT( _res );
    if( attrib == RT_CONTEXT_ATTRIBUTE_OPTIX_SALT )
    {
        OAC_BUFFER( size, p );
    }
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetBufferFromId( RTcontext context_api, int buffer_id, RTbuffer* buffer )
{
    OAC_TRACE3( context_api, buffer_id, buffer );
    const RTresult _res = _rtContextGetBufferFromId( context_api, buffer_id, buffer );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetDeviceCount( RTcontext context_api, unsigned int* count )
{
    OAC_TRACE2( context_api, count );
    const RTresult _res = _rtContextGetDeviceCount( context_api, count );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetDevices( RTcontext context_api, int* devices )
{
    OAC_TRACE2( context_api, devices );
    const RTresult _res = _rtContextGetDevices( context_api, devices );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetEntryPointCount( RTcontext context_api, unsigned int* num_entry_points )
{
    OAC_TRACE2( context_api, num_entry_points );
    const RTresult _res = _rtContextGetEntryPointCount( context_api, num_entry_points );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

void RTAPI rtContextGetErrorString( RTcontext context_api, RTresult code, const char** return_string )
{
    OAC_TRACE3( context_api, code, return_string );
    _rtContextGetErrorString( context_api, code, return_string );
    OAC_RESULT( RT_SUCCESS );  // avoid a special case in the trace by acting as if this function returned a result
    OAC_NORETURN( *return_string = "Caught exception while processing error string" );
}

RTresult RTAPI rtContextGetExceptionEnabled( RTcontext context_api, RTexception exception, int* enabled )
{
    OAC_TRACE3( context_api, exception, enabled );
    const RTresult _res = _rtContextGetExceptionEnabled( context_api, exception, enabled );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetExceptionProgram( RTcontext context_api, unsigned int entry_point_index, RTprogram* program )
{
    OAC_TRACE3( context_api, entry_point_index, program );
    const RTresult _res = _rtContextGetExceptionProgram( context_api, entry_point_index, program );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetMaxCallableProgramDepth( RTcontext context_api, unsigned int* max_depth )
{
    OAC_TRACE2( context_api, max_depth );
    const RTresult _res = _rtContextGetMaxCallableProgramDepth( context_api, max_depth );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetMaxTraceDepth( RTcontext context_api, unsigned int* max_depth )
{
    OAC_TRACE2( context_api, max_depth );
    const RTresult _res = _rtContextGetMaxTraceDepth( context_api, max_depth );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetMissProgram( RTcontext context_api, unsigned int ray_type_index, RTprogram* program )
{
    OAC_TRACE3( context_api, ray_type_index, program );
    const RTresult _res = _rtContextGetMissProgram( context_api, ray_type_index, program );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetPrintBufferSize( RTcontext context_api, RTsize* buffer_size_bytes )
{
    OAC_TRACE2( context_api, buffer_size_bytes );
    const RTresult _res = _rtContextGetPrintBufferSize( context_api, buffer_size_bytes );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetPrintEnabled( RTcontext context_api, int* enabled )
{
    OAC_TRACE2( context_api, enabled );
    const RTresult _res = _rtContextGetPrintEnabled( context_api, enabled );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetPrintLaunchIndex( RTcontext context_api, int* x, int* y, int* z )
{
    OAC_TRACE4( context_api, x, y, z );
    const RTresult _res = _rtContextGetPrintLaunchIndex( context_api, x, y, z );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetProgramFromId( RTcontext context_api, int program_id, RTprogram* program )
{
    OAC_TRACE3( context_api, program_id, program );
    const RTresult _res = _rtContextGetProgramFromId( context_api, program_id, program );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetRayGenerationProgram( RTcontext context_api, unsigned int entry_point_index, RTprogram* program )
{
    OAC_TRACE3( context_api, entry_point_index, program );
    const RTresult _res = _rtContextGetRayGenerationProgram( context_api, entry_point_index, program );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetRayTypeCount( RTcontext context_api, unsigned int* num_ray_types )
{
    OAC_TRACE2( context_api, num_ray_types );
    const RTresult _res = _rtContextGetRayTypeCount( context_api, num_ray_types );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetRunningState( RTcontext context_api, int* running )
{
    OAC_TRACE2( context_api, running );
    RTresult _res = _rtContextGetRunningState( context_api, running );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetTextureSamplerFromId( RTcontext context_api, int sampler_id, RTtexturesampler* sampler )
{
    OAC_TRACE3( context_api, sampler_id, sampler );
    const RTresult _res = _rtContextGetTextureSamplerFromId( context_api, sampler_id, sampler );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetStackSize( RTcontext context_api, RTsize* stack_size_bytes )
{
    OAC_TRACE2( context_api, stack_size_bytes );
    const RTresult _res = _rtContextGetStackSize( context_api, stack_size_bytes );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetVariable( RTcontext context_api, unsigned int index, RTvariable* v )
{
    OAC_TRACE3( context_api, index, v );
    const RTresult _res = _rtContextGetVariable( context_api, index, v );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextGetVariableCount( RTcontext context_api, unsigned int* c )
{
    OAC_TRACE2( context_api, c );
    const RTresult _res = _rtContextGetVariableCount( context_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextLaunch1D( RTcontext context_api, unsigned int entry_point_index, RTsize width )
{
    OAC_TRACE3( context_api, entry_point_index, width );
    if( getApiCapture().capture_enabled() )
    {
        Context* context = (Context*)context_api;
        getApiCapture().capture_lwda_input_buffers( context );
        getApiCapture().capture_gl_input_buffers( context );
        getApiCapture().capture_gl_images( context );
        getApiCapture().capture_d3d9_input_buffers( context );
        getApiCapture().capture_d3d9_images( context );
    }
    const RTresult _res = _rtContextLaunch1D( context_api, entry_point_index, width );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextLaunch2D( RTcontext context_api, unsigned int entry_point_index, RTsize width, RTsize height )
{
    OAC_TRACE4( context_api, entry_point_index, width, height );
    if( getApiCapture().capture_enabled() )
    {
        Context* context = (Context*)context_api;
        getApiCapture().capture_lwda_input_buffers( context );
        getApiCapture().capture_gl_input_buffers( context );
        getApiCapture().capture_gl_images( context );
        getApiCapture().capture_d3d9_input_buffers( context );
        getApiCapture().capture_d3d9_images( context );
    }
    const RTresult _res = _rtContextLaunch2D( context_api, entry_point_index, width, height );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextLaunch3D( RTcontext context_api, unsigned int entry_point_index, RTsize width, RTsize height, RTsize depth )
{
    OAC_TRACE5( context_api, entry_point_index, width, height, depth );
    if( getApiCapture().capture_enabled() )
    {
        Context* context = (Context*)context_api;
        getApiCapture().capture_lwda_input_buffers( context );
        getApiCapture().capture_gl_input_buffers( context );
        getApiCapture().capture_gl_images( context );
        getApiCapture().capture_d3d9_input_buffers( context );
        getApiCapture().capture_d3d9_images( context );
    }
    const RTresult _res = _rtContextLaunch3D( context_api, entry_point_index, width, height, depth );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextLaunchProgressive2D( RTcontext context_api, unsigned int entry_point_index, RTsize width, RTsize height, unsigned int max_subframes )
{
    // This call is high frequency, but this is something we want to catch, do that in another later point. No Timeviz here.
    OAC_TRACE5_NOTIMEVIZ( context_api, entry_point_index, width, height, max_subframes );
    if( getApiCapture().capture_enabled() )
    {
        Context* context = (Context*)context_api;
        getApiCapture().capture_gl_input_buffers( context );
        getApiCapture().capture_gl_images( context );
    }
    const RTresult _res = _rtContextLaunchProgressive2D( context_api, entry_point_index, width, height, max_subframes );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextStopProgressive( RTcontext context_api )
{
    OAC_TRACE1( context_api );
    const RTresult _res = _rtContextStopProgressive( context_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextQueryVariable( RTcontext context_api, const char* name, RTvariable* v )
{
    OAC_TRACE3( context_api, name, v );
    const RTresult _res = _rtContextQueryVariable( context_api, name, v );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextRemoveVariable( RTcontext context_api, RTvariable v_api )
{
    OAC_TRACE2( context_api, v_api );
    const RTresult _res = _rtContextRemoveVariable( context_api, v_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetDevices( RTcontext context_api, unsigned int count, const int* devices )
{
    OAC_TRACE3( context_api, count, devices );
    OAC_VALUES( count, devices );
    const RTresult _res = _rtContextSetDevices( context_api, count, devices );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetEntryPointCount( RTcontext context_api, unsigned int num_entry_points )
{
    OAC_TRACE2( context_api, num_entry_points );
    const RTresult _res = _rtContextSetEntryPointCount( context_api, num_entry_points );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetExceptionEnabled( RTcontext context_api, RTexception exception, int enabled )
{
    OAC_TRACE3( context_api, exception, enabled );
    const RTresult _res = _rtContextSetExceptionEnabled( context_api, exception, enabled );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetExceptionProgram( RTcontext context_api, unsigned int entry_point_index, RTprogram program_api )
{
    OAC_TRACE3( context_api, entry_point_index, program_api );
    const RTresult _res = _rtContextSetExceptionProgram( context_api, entry_point_index, program_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetMaxCallableProgramDepth( RTcontext context_api, unsigned int max_depth )
{
    OAC_TRACE2( context_api, max_depth );
    const RTresult _res = _rtContextSetMaxCallableProgramDepth( context_api, max_depth );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetMaxTraceDepth( RTcontext context_api, unsigned int max_depth )
{
    OAC_TRACE2( context_api, max_depth );
    const RTresult _res = _rtContextSetMaxTraceDepth( context_api, max_depth );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetMissProgram( RTcontext context_api, unsigned int ray_type_index, RTprogram program_api )
{
    OAC_TRACE3( context_api, ray_type_index, program_api );
    const RTresult _res = _rtContextSetMissProgram( context_api, ray_type_index, program_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetPrintBufferSize( RTcontext context_api, RTsize buffer_size_bytes )
{
    OAC_TRACE2( context_api, buffer_size_bytes );
    const RTresult _res = _rtContextSetPrintBufferSize( context_api, buffer_size_bytes );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetPrintEnabled( RTcontext context_api, int enabled )
{
    OAC_TRACE2( context_api, enabled );
    const RTresult _res = _rtContextSetPrintEnabled( context_api, enabled );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetPrintLaunchIndex( RTcontext context_api, int x, int y, int z )
{
    OAC_TRACE4( context_api, x, y, z );
    const RTresult _res = _rtContextSetPrintLaunchIndex( context_api, x, y, z );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetRayGenerationProgram( RTcontext context_api, unsigned int entry_point_index, RTprogram program_api )
{
    OAC_TRACE3( context_api, entry_point_index, program_api );
    const RTresult _res = _rtContextSetRayGenerationProgram( context_api, entry_point_index, program_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetRayTypeCount( RTcontext context_api, unsigned int num_ray_types )
{
    OAC_TRACE2( context_api, num_ray_types );
    const RTresult _res = _rtContextSetRayTypeCount( context_api, num_ray_types );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetStackSize( RTcontext context_api, RTsize stack_size_bytes )
{
    OAC_TRACE2( context_api, stack_size_bytes );
    const RTresult _res = _rtContextSetStackSize( context_api, stack_size_bytes );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetTimeoutCallback( RTcontext context_api, RTtimeoutcallback callback, double min_polling_seconds )
{
    OAC_TRACE3( context_api, callback, min_polling_seconds );
    RTresult _res = _rtContextSetTimeoutCallback( context_api, callback, min_polling_seconds );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetUsageReportCallback( RTcontext context_api, RTusagereportcallback callback, int verbosity, void* cbdata )
{
    OAC_TRACE4( context_api, callback, verbosity, cbdata );
    RTresult _res = _rtContextSetUsageReportCallback( context_api, callback, verbosity, cbdata );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextValidate( RTcontext context_api )
{
    OAC_TRACE1( context_api );
    const RTresult _res = _rtContextValidate( context_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtDeviceGetAttribute( int ordinal, RTdeviceattribute attrib, RTsize size, void* p )
{
    OAC_TRACE4( ordinal, attrib, size, p );
    RTresult _res = _rtDeviceGetAttribute( ordinal, attrib, size, p );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtDeviceGetDeviceCount( unsigned int* count )
{
    OAC_TRACE1( count );
    RTresult _res = _rtDeviceGetDeviceCount( count );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryCreate( RTcontext context_api, RTgeometry* geometry )
{
    OAC_TRACE2( context_api, geometry );
    const RTresult _res = _rtGeometryCreate( context_api, geometry );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *geometry );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryDeclareVariable( RTgeometry geometry_api, const char* name, RTvariable* v )
{
    OAC_TRACE3( geometry_api, name, v );
    const RTresult _res = _rtGeometryDeclareVariable( geometry_api, name, v );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *v );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryDestroy( RTgeometry geometry_api )
{
    OAC_TRACE1( geometry_api );
    const RTresult _res = _rtGeometryDestroy( geometry_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGetBoundingBoxProgram( RTgeometry geometry_api, RTprogram* program )
{
    OAC_TRACE2( geometry_api, program );
    const RTresult _res = _rtGeometryGetBoundingBoxProgram( geometry_api, program );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGetContext( RTgeometry geometry_api, RTcontext* c )
{
    OAC_TRACE2( geometry_api, c );
    const RTresult _res = _rtGeometryGetContext( geometry_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGetIntersectionProgram( RTgeometry geometry_api, RTprogram* program )
{
    OAC_TRACE2( geometry_api, program );
    const RTresult _res = _rtGeometryGetIntersectionProgram( geometry_api, program );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGetVariable( RTgeometry geometry_api, unsigned int index, RTvariable* v )
{
    OAC_TRACE3( geometry_api, index, v );
    const RTresult _res = _rtGeometryGetVariable( geometry_api, index, v );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGetVariableCount( RTgeometry geometry_api, unsigned int* c )
{
    OAC_TRACE2( geometry_api, c );
    const RTresult _res = _rtGeometryGetVariableCount( geometry_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGroupCreate( RTcontext context_api, RTgeometrygroup* geometrygroup )
{
    OAC_TRACE2( context_api, geometrygroup );
    const RTresult _res = _rtGeometryGroupCreate( context_api, geometrygroup );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *geometrygroup );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGroupDestroy( RTgeometrygroup geometrygroup_api )
{
    OAC_TRACE1( geometrygroup_api );
    const RTresult _res = _rtGeometryGroupDestroy( geometrygroup_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGroupGetAcceleration( RTgeometrygroup geometrygroup_api, RTacceleration* acceleration )
{
    OAC_TRACE2( geometrygroup_api, acceleration );
    const RTresult _res = _rtGeometryGroupGetAcceleration( geometrygroup_api, acceleration );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGroupGetChild( RTgeometrygroup geometrygroup_api, unsigned int index, RTgeometryinstance* geometryinstance )
{
    OAC_TRACE3( geometrygroup_api, index, geometryinstance );
    const RTresult _res = _rtGeometryGroupGetChild( geometrygroup_api, index, geometryinstance );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGroupGetChildCount( RTgeometrygroup geometrygroup_api, unsigned int* count )
{
    OAC_TRACE2( geometrygroup_api, count );
    const RTresult _res = _rtGeometryGroupGetChildCount( geometrygroup_api, count );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGroupGetContext( RTgeometrygroup geometrygroup_api, RTcontext* c )
{
    OAC_TRACE2( geometrygroup_api, c );
    const RTresult _res = _rtGeometryGroupGetContext( geometrygroup_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGroupSetAcceleration( RTgeometrygroup geometrygroup_api, RTacceleration acceleration_api )
{
    OAC_TRACE2( geometrygroup_api, acceleration_api );
    const RTresult _res = _rtGeometryGroupSetAcceleration( geometrygroup_api, acceleration_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGroupSetChild( RTgeometrygroup geometrygroup_api, unsigned int index, RTgeometryinstance geometryinstance_api )
{
    OAC_TRACE3( geometrygroup_api, index, geometryinstance_api );
    const RTresult _res = _rtGeometryGroupSetChild( geometrygroup_api, index, geometryinstance_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGroupSetChildCount( RTgeometrygroup geometrygroup_api, unsigned int count )
{
    OAC_TRACE2( geometrygroup_api, count );
    const RTresult _res = _rtGeometryGroupSetChildCount( geometrygroup_api, count );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGroupValidate( RTgeometrygroup geometrygroup_api )
{
    OAC_TRACE1( geometrygroup_api );
    const RTresult _res = _rtGeometryGroupValidate( geometrygroup_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceCreate( RTcontext context_api, RTgeometryinstance* geometryinstance )
{
    OAC_TRACE2( context_api, geometryinstance );
    const RTresult _res = _rtGeometryInstanceCreate( context_api, geometryinstance );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *geometryinstance );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceDeclareVariable( RTgeometryinstance gi_api, const char* name, RTvariable* v )
{
    OAC_TRACE3( gi_api, name, v );
    const RTresult _res = _rtGeometryInstanceDeclareVariable( gi_api, name, v );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *v );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceDestroy( RTgeometryinstance geometryinstance_api )
{
    OAC_TRACE1( geometryinstance_api );
    const RTresult _res = _rtGeometryInstanceDestroy( geometryinstance_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceGetContext( RTgeometryinstance gi_api, RTcontext* c )
{
    OAC_TRACE2( gi_api, c );
    const RTresult _res = _rtGeometryInstanceGetContext( gi_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceGetGeometry( RTgeometryinstance gi_api, RTgeometry* geo )
{
    OAC_TRACE2( gi_api, geo );
    const RTresult _res = _rtGeometryInstanceGetGeometry( gi_api, geo );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceGetGeometryTriangles( RTgeometryinstance gi_api, RTgeometrytriangles* geo )
{
    OAC_TRACE2( gi_api, geo );
    const RTresult _res = _rtGeometryInstanceGetGeometryTriangles( gi_api, geo );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceGetMaterial( RTgeometryinstance gi_api, unsigned int material_idx, RTmaterial* mat )
{
    OAC_TRACE3( gi_api, material_idx, mat );
    const RTresult _res = _rtGeometryInstanceGetMaterial( gi_api, material_idx, mat );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceGetMaterialCount( RTgeometryinstance gi_api, unsigned int* num_materials )
{
    OAC_TRACE2( gi_api, num_materials );
    const RTresult _res = _rtGeometryInstanceGetMaterialCount( gi_api, num_materials );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceGetVariable( RTgeometryinstance gi_api, unsigned int index, RTvariable* v )
{
    OAC_TRACE3( gi_api, index, v );
    const RTresult _res = _rtGeometryInstanceGetVariable( gi_api, index, v );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceGetVariableCount( RTgeometryinstance gi_api, unsigned int* c )
{
    OAC_TRACE2( gi_api, c );
    const RTresult _res = _rtGeometryInstanceGetVariableCount( gi_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceQueryVariable( RTgeometryinstance gi_api, const char* name, RTvariable* v )
{
    OAC_TRACE3( gi_api, name, v );
    const RTresult _res = _rtGeometryInstanceQueryVariable( gi_api, name, v );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceRemoveVariable( RTgeometryinstance gi_api, RTvariable v_api )
{
    OAC_TRACE2( gi_api, v_api );
    const RTresult _res = _rtGeometryInstanceRemoveVariable( gi_api, v_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceSetGeometry( RTgeometryinstance gi_api, RTgeometry geo_api )
{
    OAC_TRACE2( gi_api, geo_api );
    const RTresult _res = _rtGeometryInstanceSetGeometry( gi_api, geo_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceSetGeometryTriangles( RTgeometryinstance gi_api, RTgeometrytriangles geo_api )
{
    OAC_TRACE2( gi_api, geo_api );
    const RTresult _res = _rtGeometryInstanceSetGeometryTriangles( gi_api, geo_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceSetMaterial( RTgeometryinstance gi_api, unsigned int material_idx, RTmaterial mat_api )
{
    OAC_TRACE3( gi_api, material_idx, mat_api );
    const RTresult _res = _rtGeometryInstanceSetMaterial( gi_api, material_idx, mat_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceSetMaterialCount( RTgeometryinstance gi_api, unsigned int num_materials )
{
    OAC_TRACE2( gi_api, num_materials );
    const RTresult _res = _rtGeometryInstanceSetMaterialCount( gi_api, num_materials );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryInstanceValidate( RTgeometryinstance geometryinstance_api )
{
    OAC_TRACE1( geometryinstance_api );
    const RTresult _res = _rtGeometryInstanceValidate( geometryinstance_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryIsDirty( RTgeometry geometry_api, int* dirty )
{
    OAC_TRACE2( geometry_api, dirty );
    const RTresult _res = _rtGeometryIsDirty( geometry_api, dirty );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryMarkDirty( RTgeometry geometry_api )
{
    OAC_TRACE1( geometry_api );
    const RTresult _res = _rtGeometryMarkDirty( geometry_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryQueryVariable( RTgeometry geometry_api, const char* name, RTvariable* v )
{
    OAC_TRACE3( geometry_api, name, v );
    const RTresult _res = _rtGeometryQueryVariable( geometry_api, name, v );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryRemoveVariable( RTgeometry geometry_api, RTvariable v_api )
{
    OAC_TRACE2( geometry_api, v_api );
    const RTresult _res = _rtGeometryRemoveVariable( geometry_api, v_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometrySetBoundingBoxProgram( RTgeometry geometry_api, RTprogram program_api )
{
    OAC_TRACE2( geometry_api, program_api );
    const RTresult _res = _rtGeometrySetBoundingBoxProgram( geometry_api, program_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometrySetIntersectionProgram( RTgeometry geometry_api, RTprogram program_api )
{
    OAC_TRACE2( geometry_api, program_api );
    const RTresult _res = _rtGeometrySetIntersectionProgram( geometry_api, program_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometrySetPrimitiveCount( RTgeometry geometry_api, unsigned int num_primitives )
{
    OAC_TRACE2( geometry_api, num_primitives );
    const RTresult _res = _rtGeometrySetPrimitiveCount( geometry_api, num_primitives );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGetPrimitiveCount( RTgeometry geometry_api, unsigned int* num_primitives )
{
    OAC_TRACE2( geometry_api, num_primitives );
    const RTresult _res = _rtGeometryGetPrimitiveCount( geometry_api, num_primitives );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometrySetPrimitiveIndexOffset( RTgeometry geometry_api, unsigned int index_offset )
{
    OAC_TRACE2( geometry_api, index_offset );
    const RTresult _res = _rtGeometrySetPrimitiveIndexOffset( geometry_api, index_offset );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGetPrimitiveIndexOffset( RTgeometry geometry_api, unsigned int* index_offset )
{
    OAC_TRACE2( geometry_api, index_offset );
    const RTresult _res = _rtGeometryGetPrimitiveIndexOffset( geometry_api, index_offset );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometrySetMotionRange( RTgeometry geometry_api, float timeBegin, float timeEnd )
{
    OAC_TRACE3( geometry_api, timeBegin, timeEnd );
    const RTresult _res = _rtGeometrySetMotionRange( geometry_api, timeBegin, timeEnd );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGetMotionRange( RTgeometry geometry_api, float* timeBegin, float* timeEnd )
{
    OAC_TRACE3( geometry_api, timeBegin, timeEnd );
    const RTresult _res = _rtGeometryGetMotionRange( geometry_api, timeBegin, timeEnd );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometrySetMotionBorderMode( RTgeometry geometry_api, RTmotionbordermode beginMode, RTmotionbordermode endMode )
{
    OAC_TRACE3( geometry_api, beginMode, endMode );
    const RTresult _res = _rtGeometrySetMotionBorderMode( geometry_api, beginMode, endMode );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGetMotionBorderMode( RTgeometry geometry_api, RTmotionbordermode* beginMode, RTmotionbordermode* endMode )
{
    OAC_TRACE3( geometry_api, beginMode, endMode );
    const RTresult _res = _rtGeometryGetMotionBorderMode( geometry_api, beginMode, endMode );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometrySetMotionSteps( RTgeometry geometry_api, unsigned int n )
{
    OAC_TRACE2( geometry_api, n );
    const RTresult _res = _rtGeometrySetMotionSteps( geometry_api, n );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGetMotionSteps( RTgeometry geometry_api, unsigned int* n )
{
    OAC_TRACE2( geometry_api, n );
    const RTresult _res = _rtGeometryGetMotionSteps( geometry_api, n );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryValidate( RTgeometry geometry_api )
{
    OAC_TRACE1( geometry_api );
    const RTresult _res = _rtGeometryValidate( geometry_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesCreate( RTcontext context_api, RTgeometrytriangles* geometrytriangles )
{
    OAC_TRACE2( context_api, geometrytriangles );
    const RTresult _res = _rtGeometryTrianglesCreate( context_api, geometrytriangles );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *geometrytriangles );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesDestroy( RTgeometrytriangles geometrytriangles_api )
{
    OAC_TRACE1( geometrytriangles_api );
    const RTresult _res = _rtGeometryTrianglesDestroy( geometrytriangles_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesGetContext( RTgeometrytriangles geometrytriangles_api, RTcontext* c )
{
    OAC_TRACE2( geometrytriangles_api, c );
    const RTresult _res = _rtGeometryTrianglesGetContext( geometrytriangles_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesSetPrimitiveIndexOffset( RTgeometrytriangles geometrytriangles_api, unsigned int index_offset )
{
    OAC_TRACE2( geometrytriangles_api, index_offset );
    const RTresult _res = _rtGeometryTrianglesSetPrimitiveIndexOffset( geometrytriangles_api, index_offset );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesGetPrimitiveIndexOffset( RTgeometrytriangles geometrytriangles_api, unsigned int* index_offset )
{
    OAC_TRACE2( geometrytriangles_api, index_offset );
    const RTresult _res = _rtGeometryTrianglesGetPrimitiveIndexOffset( geometrytriangles_api, index_offset );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesGetPreTransformMatrix( RTgeometrytriangles geometrytriangles_api, int transpose, float* matrix )
{
    OAC_TRACE3( geometrytriangles_api, transpose, matrix );
    const RTresult _res = _rtGeometryTrianglesGetPreTransformMatrix( geometrytriangles_api, transpose, matrix );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesSetPreTransformMatrix( RTgeometrytriangles geometrytriangles_api, int transpose, const float* matrix )
{
    OAC_TRACE3( geometrytriangles_api, transpose, matrix );
    OAC_VALUES( 16, matrix );
    const RTresult _res = _rtGeometryTrianglesSetPreTransformMatrix( geometrytriangles_api, transpose, matrix );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesSetPrimitiveCount( RTgeometrytriangles geometrytriangles_api, unsigned int num_triangles )
{
    OAC_TRACE2( geometrytriangles_api, num_triangles );
    const RTresult _res = _rtGeometryTrianglesSetPrimitiveCount( geometrytriangles_api, num_triangles );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesGetPrimitiveCount( RTgeometrytriangles geometrytriangles_api, unsigned int* num_triangles )
{
    OAC_TRACE2( geometrytriangles_api, num_triangles );
    const RTresult _res = _rtGeometryTrianglesGetPrimitiveCount( geometrytriangles_api, num_triangles );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesSetTriangleIndices( RTgeometrytriangles geometrytriangles_api,
                                                      RTbuffer            index_buffer_api,
                                                      RTsize              index_buffer_byte_offset,
                                                      RTsize              tri_indices_byte_stride,
                                                      RTformat            tri_indices_format )
{
    OAC_TRACE5( geometrytriangles_api, index_buffer_api, index_buffer_byte_offset, tri_indices_byte_stride, tri_indices_format );
    const RTresult _res = _rtGeometryTrianglesSetTriangleIndices( geometrytriangles_api, index_buffer_api, index_buffer_byte_offset,
                                                                  tri_indices_byte_stride, tri_indices_format );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesSetVertices( RTgeometrytriangles geometrytriangles_api,
                                               unsigned int        num_vertices,
                                               RTbuffer            vertex_buffer_api,
                                               RTsize              vertex_buffer_byte_offset,
                                               RTsize              vertex_byte_stride,
                                               RTformat            position_format )
{
    OAC_TRACE6( geometrytriangles_api, num_vertices, vertex_buffer_api, vertex_buffer_byte_offset, vertex_byte_stride, position_format );
    const RTresult _res = _rtGeometryTrianglesSetVertices( geometrytriangles_api, num_vertices, vertex_buffer_api,
                                                           vertex_buffer_byte_offset, vertex_byte_stride, position_format );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesSetMotiolwertices( RTgeometrytriangles geometrytriangles_api,
                                                     unsigned int        num_vertices,
                                                     RTbuffer            vertex_buffer_api,
                                                     RTsize              vertex_buffer_byte_offset,
                                                     RTsize              vertex_byte_stride,
                                                     RTsize              vertex_motion_step_byte_stride,
                                                     RTformat            position_format )
{
    OAC_TRACE7( geometrytriangles_api, num_vertices, vertex_buffer_api, vertex_buffer_byte_offset, vertex_byte_stride,
                vertex_motion_step_byte_stride, position_format );
    const RTresult _res = _rtGeometryTrianglesSetMotiolwertices( geometrytriangles_api, num_vertices, vertex_buffer_api,
                                                                 vertex_buffer_byte_offset, vertex_byte_stride,
                                                                 vertex_motion_step_byte_stride, position_format );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesSetMotiolwerticesMultiBuffer( RTgeometrytriangles geometrytriangles_api,
                                                                unsigned int        num_vertices,
                                                                RTbuffer*           vertex_buffers_api,
                                                                unsigned int        vertex_buffer_count,
                                                                RTsize              vertex_buffer_byte_offset,
                                                                RTsize              vertex_byte_stride,
                                                                RTformat            position_format )
{
    OAC_TRACE7( geometrytriangles_api, num_vertices, vertex_buffers_api, vertex_buffer_count, vertex_buffer_byte_offset,
                vertex_byte_stride, position_format );
    const RTresult _res = _rtGeometryTrianglesSetMotiolwerticesMultiBuffer( geometrytriangles_api, num_vertices, vertex_buffers_api,
                                                                            vertex_buffer_count, vertex_buffer_byte_offset,
                                                                            vertex_byte_stride, position_format );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesSetMotionRange( RTgeometrytriangles geometrytriangles_api, float timeBegin, float timeEnd )
{
    OAC_TRACE3( geometrytriangles_api, timeBegin, timeEnd );
    const RTresult _res = _rtGeometryTrianglesSetMotionRange( geometrytriangles_api, timeBegin, timeEnd );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesGetMotionRange( RTgeometrytriangles geometrytriangles_api, float* timeBegin, float* timeEnd )
{
    OAC_TRACE3( geometrytriangles_api, timeBegin, timeEnd );
    const RTresult _res = _rtGeometryTrianglesGetMotionRange( geometrytriangles_api, timeBegin, timeEnd );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesSetMotionBorderMode( RTgeometrytriangles geometrytriangles_api,
                                                       RTmotionbordermode  beginMode,
                                                       RTmotionbordermode  endMode )
{
    OAC_TRACE3( geometrytriangles_api, beginMode, endMode );
    const RTresult _res = _rtGeometryTrianglesSetMotionBorderMode( geometrytriangles_api, beginMode, endMode );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesGetMotionBorderMode( RTgeometrytriangles geometrytriangles_api,
                                                       RTmotionbordermode* beginMode,
                                                       RTmotionbordermode* endMode )
{
    OAC_TRACE3( geometrytriangles_api, beginMode, endMode );
    const RTresult _res = _rtGeometryTrianglesGetMotionBorderMode( geometrytriangles_api, beginMode, endMode );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesSetMotionSteps( RTgeometrytriangles geometrytriangles_api, unsigned int num_motion_steps )
{
    OAC_TRACE2( geometrytriangles_api, num_motion_steps );
    const RTresult _res = _rtGeometryTrianglesSetMotionSteps( geometrytriangles_api, num_motion_steps );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesGetMotionSteps( RTgeometrytriangles geometrytriangles_api, unsigned int* num_motion_steps )
{
    OAC_TRACE2( geometrytriangles_api, num_motion_steps );
    const RTresult _res = _rtGeometryTrianglesGetMotionSteps( geometrytriangles_api, num_motion_steps );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesSetBuildFlags( RTgeometrytriangles geometrytriangles_api, RTgeometrybuildflags build_flags )
{
    OAC_TRACE2( geometrytriangles_api, build_flags );
    const RTresult _res = _rtGeometryTrianglesSetBuildFlags( geometrytriangles_api, build_flags );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesGetMaterialCount( RTgeometrytriangles geometrytriangles_api, unsigned int* num_materials )
{
    OAC_TRACE2( geometrytriangles_api, num_materials );
    const RTresult _res = _rtGeometryTrianglesGetMaterialCount( geometrytriangles_api, num_materials );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesSetMaterialCount( RTgeometrytriangles geometrytriangles_api, unsigned int num_materials )
{
    OAC_TRACE2( geometrytriangles_api, num_materials );
    const RTresult _res = _rtGeometryTrianglesSetMaterialCount( geometrytriangles_api, num_materials );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesSetMaterialIndices( RTgeometrytriangles geometrytriangles_api,
                                                      RTbuffer            material_index_buffer,
                                                      RTsize              material_index_buffer_byte_offset,
                                                      RTsize              material_index_byte_stride,
                                                      RTformat            material_index_format )
{
    OAC_TRACE5( geometrytriangles_api, material_index_buffer, material_index_buffer_byte_offset,
                material_index_byte_stride, material_index_format );
    const RTresult _res = _rtGeometryTrianglesSetMaterialIndices( geometrytriangles_api, material_index_buffer,
                                                                  material_index_buffer_byte_offset,
                                                                  material_index_byte_stride, material_index_format );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesSetFlagsPerMaterial( RTgeometrytriangles geometrytriangles_api, unsigned int material_index, RTgeometryflags flags )
{
    OAC_TRACE3( geometrytriangles_api, material_index, flags );
    const RTresult _res = _rtGeometryTrianglesSetFlagsPerMaterial( geometrytriangles_api, material_index, flags );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesGetFlagsPerMaterial( RTgeometrytriangles geometrytriangles_api,
                                                       unsigned int        material_index,
                                                       RTgeometryflags*    flags )
{
    OAC_TRACE3( geometrytriangles_api, material_index, flags );
    const RTresult _res = _rtGeometryTrianglesGetFlagsPerMaterial( geometrytriangles_api, material_index, flags );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesValidate( RTgeometrytriangles geometrytriangles_api )
{
    OAC_TRACE1( geometrytriangles_api );
    const RTresult _res = _rtGeometryTrianglesValidate( geometrytriangles_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesSetAttributeProgram( RTgeometrytriangles geometrytriangles_api, RTprogram program_api )
{
    OAC_TRACE2( geometrytriangles_api, program_api );
    const RTresult _res = _rtGeometryTrianglesSetAttributeProgram( geometrytriangles_api, program_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesGetAttributeProgram( RTgeometrytriangles geometrytriangles_api, RTprogram* program_api )
{
    OAC_TRACE2( geometrytriangles_api, program_api );
    const RTresult _res = _rtGeometryTrianglesGetAttributeProgram( geometrytriangles_api, program_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesDeclareVariable( RTgeometrytriangles geometrytriangles_api, const char* name, RTvariable* v )
{
    OAC_TRACE3( geometrytriangles_api, name, v );
    const RTresult _res = _rtGeometryTrianglesDeclareVariable( geometrytriangles_api, name, v );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *v );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesQueryVariable( RTgeometrytriangles geometrytriangles_api, const char* name, RTvariable* v )
{
    OAC_TRACE3( geometrytriangles_api, name, v );
    const RTresult _res = _rtGeometryTrianglesQueryVariable( geometrytriangles_api, name, v );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesRemoveVariable( RTgeometrytriangles geometrytriangles_api, RTvariable v )
{
    OAC_TRACE2( geometrytriangles_api, v );
    const RTresult _res = _rtGeometryTrianglesRemoveVariable( geometrytriangles_api, v );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesGetVariableCount( RTgeometrytriangles geometrytriangles_api, unsigned int* count )
{
    OAC_TRACE2( geometrytriangles_api, count );
    const RTresult _res = _rtGeometryTrianglesGetVariableCount( geometrytriangles_api, count );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryTrianglesGetVariable( RTgeometrytriangles geometrytriangles, unsigned int index, RTvariable* v )
{
    OAC_TRACE3( geometrytriangles, index, v );
    const RTresult _res = _rtGeometryTrianglesGetVariable( geometrytriangles, index, v );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGetBuildVersion( const char** result )
{
    // This function is part of the lwtopix API, but is not part of the optix API. It is used
    // exclusively by the wrapper to select the correct DLL. Calls to this method should not be
    // traced.
    return _rtGetBuildVersion( result );
}

RTresult RTAPI rtGetVersion( unsigned int* version )
{
    OAC_TRACE1( version );
    RTresult _res = _rtGetVersion( version );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGlobalSetAttribute( RTglobalattribute attrib, RTsize size, const void* p )
{
    RTglobalattribute recorded_attrib = attrib;
    OAC_TRACE3( recorded_attrib, size, p );
    OAC_BUFFER( size, p );
    // NOTE: Since there is no context we cannot distinguish between remote and local
    // operation. So we would need to cache this and apply to the next remote context.
    RTresult _res = _rtGlobalSetAttribute( attrib, size, p );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGlobalGetAttribute( RTglobalattribute attrib, RTsize size, void* p )
{
    OAC_TRACE3( attrib, size, p );
    // NOTE: Since there is no context we cannot distinguish between remote and local
    // operation. So we just return the local value.
    RTresult _res = _rtGlobalGetAttribute( attrib, size, p );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGroupCreate( RTcontext context_api, RTgroup* group )
{
    OAC_TRACE2( context_api, group );
    const RTresult _res = _rtGroupCreate( context_api, group );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *group );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGroupDestroy( RTgroup group_api )
{
    OAC_TRACE1( group_api );
    const RTresult _res = _rtGroupDestroy( group_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGroupGetAcceleration( RTgroup group_api, RTacceleration* acceleration )
{
    OAC_TRACE2( group_api, acceleration );
    const RTresult _res = _rtGroupGetAcceleration( group_api, acceleration );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGroupGetChild( RTgroup group_api, unsigned int index, RTobject* child )
{
    OAC_TRACE3( group_api, index, child );
    const RTresult _res = _rtGroupGetChild( group_api, index, child );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGroupGetChildCount( RTgroup group_api, unsigned int* count )
{
    OAC_TRACE2( group_api, count );
    const RTresult _res = _rtGroupGetChildCount( group_api, count );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGroupGetChildType( RTgroup group_api, unsigned int index, RTobjecttype* type )
{
    OAC_TRACE3( group_api, index, type );
    const RTresult _res = _rtGroupGetChildType( group_api, index, type );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGroupGetContext( RTgroup group_api, RTcontext* c )
{
    OAC_TRACE2( group_api, c );
    const RTresult _res = _rtGroupGetContext( group_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGroupSetAcceleration( RTgroup group_api, RTacceleration acceleration_api )
{
    OAC_TRACE2( group_api, acceleration_api );
    const RTresult _res = _rtGroupSetAcceleration( group_api, acceleration_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGroupSetChild( RTgroup group_api, unsigned int index, RTobject child )
{
    OAC_TRACE3( group_api, index, child );
    const RTresult _res = _rtGroupSetChild( group_api, index, child );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGroupSetChildCount( RTgroup group_api, unsigned int count )
{
    OAC_TRACE2( group_api, count );
    const RTresult _res = _rtGroupSetChildCount( group_api, count );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGroupValidate( RTgroup group_api )
{
    OAC_TRACE1( group_api );
    const RTresult _res = _rtGroupValidate( group_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGroupSetVisibilityMask( RTgroup group, RTvisibilitymask mask )
{
    OAC_TRACE2( group, mask );
    const RTresult _res = _rtGroupSetVisibilityMask( group, mask );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGroupGetVisibilityMask( RTgroup group, RTvisibilitymask* mask )
{
    OAC_TRACE2( group, mask );
    const RTresult _res = _rtGroupGetVisibilityMask( group, mask );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGroupSetFlags( RTgeometrygroup group, RTinstanceflags flags )
{
    OAC_TRACE2( group, flags );
    const RTresult _res = _rtGeometryGroupSetFlags( group, flags );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGroupGetFlags( RTgeometrygroup group, RTinstanceflags* flags )
{
    OAC_TRACE2( group, flags );
    const RTresult _res = _rtGeometryGroupGetFlags( group, flags );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGroupSetVisibilityMask( RTgeometrygroup group, RTvisibilitymask mask )
{
    OAC_TRACE2( group, mask );
    const RTresult _res = _rtGeometryGroupSetVisibilityMask( group, mask );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGroupGetVisibilityMask( RTgeometrygroup group, RTvisibilitymask* mask )
{
    OAC_TRACE2( group, mask );
    const RTresult _res = _rtGeometryGroupGetVisibilityMask( group, mask );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometrySetFlags( RTgeometry geometry, RTgeometryflags flags )
{
    OAC_TRACE2( geometry, flags );
    const RTresult _res = _rtGeometrySetFlags( geometry, flags );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtGeometryGetFlags( RTgeometry geometry, RTgeometryflags* flags )
{
    OAC_TRACE2( geometry, flags );
    const RTresult _res = _rtGeometryGetFlags( geometry, flags );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtMaterialCreate( RTcontext context_api, RTmaterial* material )
{
    OAC_TRACE2( context_api, material );
    const RTresult _res = _rtMaterialCreate( context_api, material );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *material );
    OAC_RETURN( _res );
}

RTresult RTAPI rtMaterialDeclareVariable( RTmaterial material_api, const char* name, RTvariable* v )
{
    OAC_TRACE3( material_api, name, v );
    const RTresult _res = _rtMaterialDeclareVariable( material_api, name, v );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *v );
    OAC_RETURN( _res );
}

RTresult RTAPI rtMaterialDestroy( RTmaterial material_api )
{
    OAC_TRACE1( material_api );
    const RTresult _res = _rtMaterialDestroy( material_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtMaterialGetAnyHitProgram( RTmaterial material_api, unsigned int ray_type_index, RTprogram* program )
{
    OAC_TRACE3( material_api, ray_type_index, program );
    const RTresult _res = _rtMaterialGetAnyHitProgram( material_api, ray_type_index, program );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtMaterialGetClosestHitProgram( RTmaterial material_api, unsigned int ray_type_index, RTprogram* program )
{
    OAC_TRACE3( material_api, ray_type_index, program );
    const RTresult _res = _rtMaterialGetClosestHitProgram( material_api, ray_type_index, program );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtMaterialGetContext( RTmaterial material_api, RTcontext* c )
{
    OAC_TRACE2( material_api, c );
    const RTresult _res = _rtMaterialGetContext( material_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtMaterialGetVariable( RTmaterial material_api, unsigned int index, RTvariable* v )
{
    OAC_TRACE3( material_api, index, v );
    const RTresult _res = _rtMaterialGetVariable( material_api, index, v );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtMaterialGetVariableCount( RTmaterial material_api, unsigned int* c )
{
    OAC_TRACE2( material_api, c );
    const RTresult _res = _rtMaterialGetVariableCount( material_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtMaterialQueryVariable( RTmaterial material_api, const char* name, RTvariable* v )
{
    OAC_TRACE3( material_api, name, v );
    const RTresult _res = _rtMaterialQueryVariable( material_api, name, v );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtMaterialRemoveVariable( RTmaterial material_api, RTvariable v_api )
{
    OAC_TRACE2( material_api, v_api );
    const RTresult _res = _rtMaterialRemoveVariable( material_api, v_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtMaterialSetAnyHitProgram( RTmaterial material_api, unsigned int ray_type_index, RTprogram program_api )
{
    OAC_TRACE3( material_api, ray_type_index, program_api );
    const RTresult _res = _rtMaterialSetAnyHitProgram( material_api, ray_type_index, program_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtMaterialSetClosestHitProgram( RTmaterial material_api, unsigned int ray_type_index, RTprogram program_api )
{
    OAC_TRACE3( material_api, ray_type_index, program_api );
    const RTresult _res = _rtMaterialSetClosestHitProgram( material_api, ray_type_index, program_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtMaterialValidate( RTmaterial material_api )
{
    OAC_TRACE1( material_api );
    const RTresult _res = _rtMaterialValidate( material_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtOverridesOtherVersion( const char* otherVersion, int* result )
{
    // This function is part of the lwtopix API, but is not part of the optix API. It is used
    // exclusively by the wrapper to select the correct DLL. Calls to this method should not be
    // traced.
    return _rtOverridesOtherVersion( otherVersion, result );
}

// NOTE: This has a different name than the official API because it has a different signature.
// This allows to load the denoiser & ssim libraries in the wrapper
// and pass a created library object into the function.
RTresult RTAPI rtPostProcessingStageCreateBuiltinInternal( RTcontext              context_api,
                                                           const char*            builtin_name,
                                                           void*                  denoiser,
                                                           void*                  ssim_predictor,
                                                           RTpostprocessingstage* stage )
{
    // NOTE: The name of the function diverges here, because the externally visible API function has a different signature than the implementation in
    // lwoptix. This is done to allow loading shared libraries from the wrapper and keep them outside the driver.
    OAC_TRACE_START();
    getApiCapture().trace( "rtPostProcessingStageCreateBuiltin", context_api, builtin_name, stage );
    RTresult _res = _rtPostProcessingStageCreateBuiltin( context_api, builtin_name, denoiser, ssim_predictor, stage );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *stage );
    OAC_RETURN( _res );
}

RTresult RTAPI rtPostProcessingStageDeclareVariable( RTpostprocessingstage stage_api, const char* name, RTvariable* v )
{
    OAC_TRACE3( stage_api, name, v );
    RTresult _res = _rtPostProcessingStageDeclareVariable( stage_api, name, v );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *v );
    OAC_RETURN( _res );
}

RTresult RTAPI rtPostProcessingStageDestroy( RTpostprocessingstage stage_api )
{
    OAC_TRACE1( stage_api );
    RTresult _res = _rtPostProcessingStageDestroy( stage_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtPostProcessingStageGetContext( RTpostprocessingstage stage_api, RTcontext* c )
{
    OAC_TRACE1( stage_api );
    RTresult _res = _rtPostProcessingStageGetContext( stage_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtPostProcessingStageQueryVariable( RTpostprocessingstage stage_api, const char* name, RTvariable* variable )
{
    OAC_TRACE3( stage_api, name, variable );
    RTresult _res = _rtPostProcessingStageQueryVariable( stage_api, name, variable );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtPostProcessingStageGetVariableCount( RTpostprocessingstage stage_api, unsigned int* count )
{
    OAC_TRACE2( stage_api, count );
    RTresult _res = _rtPostProcessingStageGetVariableCount( stage_api, count );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtPostProcessingStageGetVariable( RTpostprocessingstage stage_api, unsigned int index, RTvariable* variable )
{
    OAC_TRACE3( stage_api, index, variable );
    RTresult _res = _rtPostProcessingStageGetVariable( stage_api, index, variable );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtProgramCreateFromPTXFile( RTcontext context_api, const char* filename, const char* program_name, RTprogram* program )
{
    OAC_TRACE4( context_api, filename, program_name, program );
    if( getApiCapture().capture_enabled() )
    {
        const long long   sz = fileSize( filename );
        std::vector<char> str;
        if( sz > 0 )
        {
            str.resize( unsigned( sz ) );
            if( !loadFile( filename, &str[0], sz ) )
            {
                lerr << "OptiX API capture: error loading \"" << filename << "\"\n";
            }
        }
        str.push_back( 0 );
        getApiCapture().capture_ptx( &str[0] );
    }
    const RTresult _res = _rtProgramCreateFromPTXFile( context_api, filename, program_name, program );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *program );
    OAC_RETURN( _res );
}

RTresult RTAPI rtProgramCreateFromPTXFiles( RTcontext context_api, unsigned int n, const char** filenames, const char* program_name, RTprogram* program )
{
    OAC_TRACE5( context_api, n, filenames, program_name, program );
    if( getApiCapture().capture_enabled() )
    {
        std::vector<std::vector<char>> ptx_str( n );
        std::vector<const char*>       cstrings( n );
        for( unsigned int i = 0; i < n; ++i )
        {
            const char*        filename = filenames[i];
            const long long    sz       = fileSize( filename );
            std::vector<char>& str      = ptx_str[i];
            if( sz > 0 )
            {
                str.resize( unsigned( sz ) );
                if( !loadFile( filename, &str[0], sz ) )
                {
                    lerr << "OptiX API capture: error loading \"" << filename << "\"\n";
                }
            }
            str.push_back( 0 );
            cstrings[i] = &str[0];
        }
        getApiCapture().capture_ptx_strings( n, &cstrings[0] );
    }
    const RTresult _res = _rtProgramCreateFromPTXFiles( context_api, n, filenames, program_name, program );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *program );
    OAC_RETURN( _res );
}

RTresult RTAPI rtProgramCreateFromPTXString( RTcontext context_api, const char* ptx, const char* program_name, RTprogram* program )
{
    OAC_TRACE4( context_api, "<str>", program_name, program );
    getApiCapture().capture_ptx( ptx );
    const RTresult _res = _rtProgramCreateFromPTXString( context_api, ptx, program_name, program );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *program );
    OAC_RETURN( _res );
}

RTresult RTAPI rtProgramCreateFromPTXStrings( RTcontext context_api, unsigned int n, const char** ptx_strings, const char* program_name, RTprogram* program )
{
    OAC_TRACE5( context_api, n, "<multiple str>", program_name, program );
    getApiCapture().capture_ptx_strings( n, ptx_strings );
    const RTresult _res = _rtProgramCreateFromPTXStrings( context_api, n, ptx_strings, program_name, program );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *program );
    OAC_RETURN( _res );
}

RTresult RTAPI rtProgramCreateFromProgram( RTcontext context_api, RTprogram program_in_api, RTprogram* program )
{
    OAC_TRACE3( context_api, program_in_api, program );
    const RTresult _res = _rtProgramCreateFromProgram( context_api, program_in_api, program );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *program );
    OAC_RETURN( _res );
}


RTresult RTAPI rtProgramDeclareVariable( RTprogram program_api, const char* name, RTvariable* v )
{
    OAC_TRACE3( program_api, name, v );
    const RTresult _res = _rtProgramDeclareVariable( program_api, name, v );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *v );
    OAC_RETURN( _res );
}

RTresult RTAPI rtProgramDestroy( RTprogram program_api )
{
    OAC_TRACE1( program_api );
    const RTresult _res = _rtProgramDestroy( program_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtProgramGetContext( RTprogram program_api, RTcontext* c )
{
    OAC_TRACE2( program_api, c );
    const RTresult _res = _rtProgramGetContext( program_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtProgramGetId( RTprogram program_api, int* program_id )
{
    OAC_TRACE2( program_api, program_id );
    const RTresult _res = _rtProgramGetId( program_api, program_id );
    OAC_RESULT( _res );
    OAC_ID( *program_id );
    OAC_RETURN( _res );
}

RTresult RTAPI rtProgramGetVariable( RTprogram program_api, unsigned int index, RTvariable* v )
{
    OAC_TRACE3( program_api, index, v );
    const RTresult _res = _rtProgramGetVariable( program_api, index, v );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtProgramGetVariableCount( RTprogram program_api, unsigned int* c )
{
    OAC_TRACE2( program_api, c );
    const RTresult _res = _rtProgramGetVariableCount( program_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtProgramQueryVariable( RTprogram program_api, const char* name, RTvariable* v )
{
    OAC_TRACE3( program_api, name, v );
    const RTresult _res = _rtProgramQueryVariable( program_api, name, v );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtProgramRemoveVariable( RTprogram program_api, RTvariable v_api )
{
    OAC_TRACE2( program_api, v_api );
    const RTresult _res = _rtProgramRemoveVariable( program_api, v_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtProgramValidate( RTprogram program_api )
{
    OAC_TRACE1( program_api );
    const RTresult _res = _rtProgramValidate( program_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtProgramCallsiteSetPotentialCallees( RTprogram program_api, const char* name, const int* ids, int numIds )
{
    OAC_TRACE4( program_api, name, ids, numIds );
    OAC_VALUES( numIds, ids );
    const RTresult _res = _rtProgramCallsiteSetPotentialCallees( program_api, name, ids, numIds );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorCreate( RTcontext context_api, RTselector* selector )
{
    OAC_TRACE2( context_api, selector );
    const RTresult _res = _rtSelectorCreate( context_api, selector );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *selector );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorDeclareVariable( RTselector selector_api, const char* name, RTvariable* v )
{
    OAC_TRACE3( selector_api, name, v );
    const RTresult _res = _rtSelectorDeclareVariable( selector_api, name, v );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *v );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorDestroy( RTselector selector_api )
{
    OAC_TRACE1( selector_api );
    const RTresult _res = _rtSelectorDestroy( selector_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorGetChild( RTselector selector_api, unsigned int index, RTobject* child )
{
    OAC_TRACE3( selector_api, index, child );
    const RTresult _res = _rtSelectorGetChild( selector_api, index, child );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorGetChildCount( RTselector selector_api, unsigned int* count )
{
    OAC_TRACE2( selector_api, count );
    const RTresult _res = _rtSelectorGetChildCount( selector_api, count );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorGetChildType( RTselector selector_api, unsigned int index, RTobjecttype* type )
{
    OAC_TRACE3( selector_api, index, type );
    const RTresult _res = _rtSelectorGetChildType( selector_api, index, type );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorGetContext( RTselector selector_api, RTcontext* c )
{
    OAC_TRACE2( selector_api, c );
    const RTresult _res = _rtSelectorGetContext( selector_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorGetVariable( RTselector selector_api, unsigned int index, RTvariable* v )
{
    OAC_TRACE3( selector_api, index, v );
    const RTresult _res = _rtSelectorGetVariable( selector_api, index, v );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorGetVariableCount( RTselector selector_api, unsigned int* c )
{
    OAC_TRACE2( selector_api, c );
    const RTresult _res = _rtSelectorGetVariableCount( selector_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorGetVisitProgram( RTselector selector_api, RTprogram* program )
{
    OAC_TRACE2( selector_api, program );
    const RTresult _res = _rtSelectorGetVisitProgram( selector_api, program );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorQueryVariable( RTselector selector_api, const char* name, RTvariable* v )
{
    OAC_TRACE3( selector_api, name, v );
    const RTresult _res = _rtSelectorQueryVariable( selector_api, name, v );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorRemoveVariable( RTselector selector_api, RTvariable v_api )
{
    OAC_TRACE2( selector_api, v_api );
    const RTresult _res = _rtSelectorRemoveVariable( selector_api, v_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorSetChild( RTselector selector_api, unsigned int index, RTobject child )
{
    OAC_TRACE3( selector_api, index, child );
    const RTresult _res = _rtSelectorSetChild( selector_api, index, child );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorSetChildCount( RTselector selector_api, unsigned int count )
{
    OAC_TRACE2( selector_api, count );
    const RTresult _res = _rtSelectorSetChildCount( selector_api, count );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorSetVisitProgram( RTselector selector_api, RTprogram program_api )
{
    OAC_TRACE2( selector_api, program_api );
    const RTresult _res = _rtSelectorSetVisitProgram( selector_api, program_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSelectorValidate( RTselector selector_api )
{
    OAC_TRACE1( selector_api );
    const RTresult _res = _rtSelectorValidate( selector_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtSetLibraryVariant( bool isLibraryFromSdk )
{
    // This function is part of the lwtopix API, but is not part of the optix API. It is used
    // exclusively by the wrapper to select the correct DLL. Calls to this method should not be
    // traced.
    return _rtSetLibraryVariant( isLibraryFromSdk );
}

RTresult RTAPI rtSupportsLwrrentDriver()
{
    // This function is part of the lwtopix API, but is not part of the optix API. It is used
    // exclusively by the wrapper to select the correct DLL. Calls to this method should not be
    // traced.
    return _rtSupportsLwrrentDriver();
}

RTresult RTAPI rtTextureSamplerCreate( RTcontext context_api, RTtexturesampler* textureSampler )
{
    OAC_TRACE2( context_api, textureSampler );
    const RTresult _res = _rtTextureSamplerCreate( context_api, textureSampler );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *textureSampler );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerCreateFromGLImage( RTcontext context_api, unsigned int gl_id, RTgltarget target, RTtexturesampler* textureSampler )
{
    OAC_TRACE4( context_api, gl_id, target, textureSampler );
    OAC_COMMENT( gltarget2string( target ) );
    RTresult _res = _rtTextureSamplerCreateFromGLImage( context_api, gl_id, target, textureSampler );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *textureSampler );

    // Record GL object info regardless the API call result to know what was it even if it failed in the API call
    if( getApiCapture().capture_enabled() )
        getApiCapture().capture_gl_image_info( gl_id, target );

    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerDestroy( RTtexturesampler textureSampler_api )
{
    OAC_TRACE1( textureSampler_api );
    const RTresult _res = _rtTextureSamplerDestroy( textureSampler_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGLRegister( RTtexturesampler textureSampler_api )
{
    OAC_TRACE1( textureSampler_api );
    RTresult _res = _rtTextureSamplerGLRegister( textureSampler_api );
    OAC_RESULT( _res );

    // Record GL object info regardless the API call result to know what was it even if it failed in the API call
    if( getApiCapture().capture_enabled() )
    {
        TextureSampler* sampler = (TextureSampler*)textureSampler_api;
        if( sampler->isInteropTexture() )
        {
            const GfxInteropResource& resource = sampler->getGfxInteropResource();
            getApiCapture().capture_gl_image_info( resource.gl.glId, resource.gl.target );
        }
    }

    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGLUnregister( RTtexturesampler textureSampler_api )
{
    OAC_TRACE1( textureSampler_api );
    RTresult _res = _rtTextureSamplerGLUnregister( textureSampler_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetArraySize( RTtexturesampler textureSampler_api, unsigned int* deprecated0 )
{
    OAC_TRACE2( textureSampler_api, deprecated0 );
    const RTresult _res = _rtTextureSamplerGetArraySize( textureSampler_api, deprecated0 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetBuffer( RTtexturesampler textureSampler_api, unsigned int deprecated0, unsigned int deprecated1, RTbuffer* buffer )
{
    OAC_TRACE4( textureSampler_api, deprecated0, deprecated1, buffer );
    const RTresult _res = _rtTextureSamplerGetBuffer( textureSampler_api, deprecated0, deprecated1, buffer );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetContext( RTtexturesampler textureSampler_api, RTcontext* c )
{
    OAC_TRACE2( textureSampler_api, c );
    const RTresult _res = _rtTextureSamplerGetContext( textureSampler_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetFilteringModes( RTtexturesampler textureSampler_api,
                                                  RTfiltermode*    minFilter,
                                                  RTfiltermode*    magFilter,
                                                  RTfiltermode*    mipFilter )
{
    OAC_TRACE4( textureSampler_api, minFilter, magFilter, mipFilter );
    const RTresult _res = _rtTextureSamplerGetFilteringModes( textureSampler_api, minFilter, magFilter, mipFilter );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetGLImageId( RTtexturesampler textureSampler_api, unsigned int* gl_id )
{
    OAC_TRACE2( textureSampler_api, gl_id );
    RTresult _res = _rtTextureSamplerGetGLImageId( textureSampler_api, gl_id );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetIndexingMode( RTtexturesampler textureSampler_api, RTtextureindexmode* indexmode )
{
    OAC_TRACE2( textureSampler_api, indexmode );
    const RTresult _res = _rtTextureSamplerGetIndexingMode( textureSampler_api, indexmode );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetMaxAnisotropy( RTtexturesampler textureSampler_api, float* maxAnisotropy )
{
    OAC_TRACE2( textureSampler_api, maxAnisotropy );
    const RTresult _res = _rtTextureSamplerGetMaxAnisotropy( textureSampler_api, maxAnisotropy );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetMipLevelClamp( RTtexturesampler textureSampler_api, float* minLevel, float* maxLevel )
{
    OAC_TRACE3( textureSampler_api, minLevel, maxLevel );
    const RTresult _res = _rtTextureSamplerGetMipLevelClamp( textureSampler_api, minLevel, maxLevel );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetMipLevelBias( RTtexturesampler textureSampler_api, float* bias )
{
    OAC_TRACE2( textureSampler_api, bias );
    const RTresult _res = _rtTextureSamplerGetMipLevelBias( textureSampler_api, bias );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetMipLevelCount( RTtexturesampler textureSampler_api, unsigned int* deprecated0 )
{
    OAC_TRACE2( textureSampler_api, deprecated0 );
    const RTresult _res = _rtTextureSamplerGetMipLevelCount( textureSampler_api, deprecated0 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetReadMode( RTtexturesampler textureSampler_api, RTtexturereadmode* readmode )
{
    OAC_TRACE2( textureSampler_api, readmode );
    const RTresult _res = _rtTextureSamplerGetReadMode( textureSampler_api, readmode );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetWrapMode( RTtexturesampler textureSampler_api, unsigned int dim, RTwrapmode* wm )
{
    OAC_TRACE3( textureSampler_api, dim, wm );
    const RTresult _res = _rtTextureSamplerGetWrapMode( textureSampler_api, dim, wm );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerSetArraySize( RTtexturesampler textureSampler_api, unsigned int deprecated0 )
{
    OAC_TRACE2( textureSampler_api, deprecated0 );
    const RTresult _res = _rtTextureSamplerSetArraySize( textureSampler_api, deprecated0 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerSetBuffer( RTtexturesampler textureSampler_api, unsigned int deprecated0, unsigned int deprecated1, RTbuffer buffer_api )
{
    OAC_TRACE4( textureSampler_api, deprecated0, deprecated1, buffer_api );
    const RTresult _res = _rtTextureSamplerSetBuffer( textureSampler_api, deprecated0, deprecated1, buffer_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerSetFilteringModes( RTtexturesampler textureSampler_api, RTfiltermode minFilter, RTfiltermode magFilter, RTfiltermode mipFilter )
{
    OAC_TRACE4( textureSampler_api, minFilter, magFilter, mipFilter );
    const RTresult _res = _rtTextureSamplerSetFilteringModes( textureSampler_api, minFilter, magFilter, mipFilter );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerSetIndexingMode( RTtexturesampler textureSampler_api, RTtextureindexmode indexmode )
{
    OAC_TRACE2( textureSampler_api, indexmode );
    const RTresult _res = _rtTextureSamplerSetIndexingMode( textureSampler_api, indexmode );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerSetMaxAnisotropy( RTtexturesampler textureSampler_api, float maxAnisotropy )
{
    OAC_TRACE2( textureSampler_api, maxAnisotropy );
    const RTresult _res = _rtTextureSamplerSetMaxAnisotropy( textureSampler_api, maxAnisotropy );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerSetMipLevelClamp( RTtexturesampler textureSampler_api, float minLevel, float maxLevel )
{
    OAC_TRACE3( textureSampler_api, minLevel, maxLevel );
    const RTresult _res = _rtTextureSamplerSetMipLevelClamp( textureSampler_api, minLevel, maxLevel );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerSetMipLevelBias( RTtexturesampler textureSampler_api, float bias )
{
    OAC_TRACE2( textureSampler_api, bias );
    const RTresult _res = _rtTextureSamplerSetMipLevelBias( textureSampler_api, bias );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerSetMipLevelCount( RTtexturesampler textureSampler_api, unsigned int deprecated0 )
{
    OAC_TRACE2( textureSampler_api, deprecated0 );
    const RTresult _res = _rtTextureSamplerSetMipLevelCount( textureSampler_api, deprecated0 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerSetReadMode( RTtexturesampler textureSampler_api, RTtexturereadmode readmode )
{
    OAC_TRACE2( textureSampler_api, readmode );
    const RTresult _res = _rtTextureSamplerSetReadMode( textureSampler_api, readmode );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerSetWrapMode( RTtexturesampler textureSampler_api, unsigned int dim, RTwrapmode wm )
{
    OAC_TRACE3( textureSampler_api, dim, wm );
    const RTresult _res = _rtTextureSamplerSetWrapMode( textureSampler_api, dim, wm );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerValidate( RTtexturesampler textureSampler_api )
{
    OAC_TRACE1( textureSampler_api );
    const RTresult _res = _rtTextureSamplerValidate( textureSampler_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetId( RTtexturesampler textureSampler_api, int* texture_id )
{
    OAC_TRACE2( textureSampler_api, texture_id );
    const RTresult _res = _rtTextureSamplerGetId( textureSampler_api, texture_id );
    OAC_RESULT( _res );
    OAC_ID( *texture_id );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformCreate( RTcontext context_api, RTtransform* transform )
{
    OAC_TRACE2( context_api, transform );
    const RTresult _res = _rtTransformCreate( context_api, transform );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *transform );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformDestroy( RTtransform transform_api )
{
    OAC_TRACE1( transform_api );
    const RTresult _res = _rtTransformDestroy( transform_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformGetChild( RTtransform transform_api, RTobject* child )
{
    OAC_TRACE2( transform_api, child );
    const RTresult _res = _rtTransformGetChild( transform_api, child );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformGetChildType( RTtransform transform_api, RTobjecttype* type )
{
    OAC_TRACE2( transform_api, type );
    const RTresult _res = _rtTransformGetChildType( transform_api, type );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformGetContext( RTtransform transform_api, RTcontext* c )
{
    OAC_TRACE2( transform_api, c );
    const RTresult _res = _rtTransformGetContext( transform_api, c );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformGetMatrix( RTtransform transform_api, int transpose, float* matrix, float* ilwerse_matrix )
{
    OAC_TRACE4( transform_api, transpose, matrix, ilwerse_matrix );
    const RTresult _res = _rtTransformGetMatrix( transform_api, transpose, matrix, ilwerse_matrix );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformSetChild( RTtransform transform_api, RTobject child )
{
    OAC_TRACE2( transform_api, child );
    const RTresult _res = _rtTransformSetChild( transform_api, child );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformSetMatrix( RTtransform transform_api, int transpose, const float* matrix, const float* ilwerse_matrix )
{
    OAC_TRACE4( transform_api, transpose, matrix, ilwerse_matrix );
    OAC_VALUES( 16, matrix );
    OAC_VALUES2( 16, ilwerse_matrix );
    const RTresult _res = _rtTransformSetMatrix( transform_api, transpose, matrix, ilwerse_matrix );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformSetMotionRange( RTtransform transform_api, float timeBegin, float timeEnd )
{
    OAC_TRACE3( transform_api, timeBegin, timeEnd );
    const RTresult _res = _rtTransformSetMotionRange( transform_api, timeBegin, timeEnd );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformGetMotionRange( RTtransform transform_api, float* timeBegin, float* timeEnd )
{
    OAC_TRACE3( transform_api, timeBegin, timeEnd );
    const RTresult _res = _rtTransformGetMotionRange( transform_api, timeBegin, timeEnd );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformSetMotionBorderMode( RTtransform transform_api, RTmotionbordermode beginMode, RTmotionbordermode endMode )
{
    OAC_TRACE3( transform_api, beginMode, endMode );
    const RTresult _res = _rtTransformSetMotionBorderMode( transform_api, beginMode, endMode );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformGetMotionBorderMode( RTtransform transform_api, RTmotionbordermode* beginMode, RTmotionbordermode* endMode )
{
    OAC_TRACE3( transform_api, beginMode, endMode );
    const RTresult _res = _rtTransformGetMotionBorderMode( transform_api, beginMode, endMode );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformSetMotionKeys( RTtransform transform_api, unsigned int n, RTmotionkeytype type, const float* keys )
{
    OAC_TRACE4( transform_api, n, type, keys );
    const size_t keySize = type == RT_MOTIONKEYTYPE_MATRIX_FLOAT12 ? 12 : 16;
    OAC_VALUES( n * keySize, keys );
    const RTresult _res = _rtTransformSetMotionKeys( transform_api, n, type, keys );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformGetMotionKeyCount( RTtransform transform_api, unsigned int* n )
{
    OAC_TRACE2( transform_api, n );
    const RTresult _res = _rtTransformGetMotionKeyCount( transform_api, n );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformGetMotionKeyType( RTtransform transform_api, RTmotionkeytype* type )
{
    OAC_TRACE2( transform_api, type );
    const RTresult _res = _rtTransformGetMotionKeyType( transform_api, type );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformGetMotionKeys( RTtransform transform_api, float* keys )
{
    OAC_TRACE2( transform_api, keys );
    const RTresult _res = _rtTransformGetMotionKeys( transform_api, keys );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTransformValidate( RTtransform transform_api )
{
    OAC_TRACE1( transform_api );
    const RTresult _res = _rtTransformValidate( transform_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet1f( RTvariable v, float* f1 )
{
    OAC_TRACE2( v, f1 );
    const RTresult _res = _rtVariableGet1f( v, f1 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet1fv( RTvariable v, float* f )
{
    OAC_TRACE2( v, f );
    const RTresult _res = _rtVariableGet1fv( v, f );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet1i( RTvariable v, int* i1 )
{
    OAC_TRACE2( v, i1 );
    const RTresult _res = _rtVariableGet1i( v, i1 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet1iv( RTvariable v, int* i )
{
    OAC_TRACE2( v, i );
    const RTresult _res = _rtVariableGet1iv( v, i );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet1ui( RTvariable v, unsigned int* u1 )
{
    OAC_TRACE2( v, u1 );
    const RTresult _res = _rtVariableGet1ui( v, u1 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet1uiv( RTvariable v, unsigned int* u )
{
    OAC_TRACE2( v, u );
    const RTresult _res = _rtVariableGet1uiv( v, u );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet1ll( RTvariable v, long long* ll1 )
{
    OAC_TRACE2( v, ll1 );
    const RTresult _res = _rtVariableGet1ll( v, ll1 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet1llv( RTvariable v, long long* ll )
{
    OAC_TRACE2( v, ll );
    const RTresult _res = _rtVariableGet1llv( v, ll );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet1ull( RTvariable v, unsigned long long* ull1 )
{
    OAC_TRACE2( v, ull1 );
    const RTresult _res = _rtVariableGet1ull( v, ull1 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet1ullv( RTvariable v, unsigned long long* ull )
{
    OAC_TRACE2( v, ull );
    const RTresult _res = _rtVariableGet1ullv( v, ull );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet2f( RTvariable v, float* f1, float* f2 )
{
    OAC_TRACE3( v, f1, f2 );
    const RTresult _res = _rtVariableGet2f( v, f1, f2 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet2fv( RTvariable v, float* f )
{
    OAC_TRACE2( v, f );
    const RTresult _res = _rtVariableGet2fv( v, f );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet2i( RTvariable v, int* i1, int* i2 )
{
    OAC_TRACE3( v, i1, i2 );
    const RTresult _res = _rtVariableGet2i( v, i1, i2 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet2iv( RTvariable v, int* i )
{
    OAC_TRACE2( v, i );
    const RTresult _res = _rtVariableGet2iv( v, i );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet2ui( RTvariable v, unsigned int* u1, unsigned int* u2 )
{
    OAC_TRACE3( v, u1, u2 );
    const RTresult _res = _rtVariableGet2ui( v, u1, u2 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet2uiv( RTvariable v, unsigned int* u )
{
    OAC_TRACE2( v, u );
    const RTresult _res = _rtVariableGet2uiv( v, u );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet2ll( RTvariable v, long long* ll1, long long* ll2 )
{
    OAC_TRACE3( v, ll1, ll2 );
    const RTresult _res = _rtVariableGet2ll( v, ll1, ll2 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet2llv( RTvariable v, long long* ll )
{
    OAC_TRACE2( v, ll );
    const RTresult _res = _rtVariableGet2llv( v, ll );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet2ull( RTvariable v, unsigned long long* ull1, unsigned long long* ull2 )
{
    OAC_TRACE3( v, ull1, ull2 );
    const RTresult _res = _rtVariableGet2ull( v, ull1, ull2 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet2ullv( RTvariable v, unsigned long long* ull )
{
    OAC_TRACE2( v, ull );
    const RTresult _res = _rtVariableGet2ullv( v, ull );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet3f( RTvariable v, float* f1, float* f2, float* f3 )
{
    OAC_TRACE4( v, f1, f2, f3 );
    const RTresult _res = _rtVariableGet3f( v, f1, f2, f3 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet3fv( RTvariable v, float* f )
{
    OAC_TRACE2( v, f );
    const RTresult _res = _rtVariableGet3fv( v, f );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet3i( RTvariable v, int* i1, int* i2, int* i3 )
{
    OAC_TRACE4( v, i1, i2, i3 );
    const RTresult _res = _rtVariableGet3i( v, i1, i2, i3 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet3iv( RTvariable v, int* i )
{
    OAC_TRACE2( v, i );
    const RTresult _res = _rtVariableGet3iv( v, i );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet3ui( RTvariable v, unsigned int* u1, unsigned int* u2, unsigned int* u3 )
{
    OAC_TRACE4( v, u1, u2, u3 );
    const RTresult _res = _rtVariableGet3ui( v, u1, u2, u3 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet3uiv( RTvariable v, unsigned int* u )
{
    OAC_TRACE2( v, u );
    const RTresult _res = _rtVariableGet3uiv( v, u );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet3ll( RTvariable v, long long* ll1, long long* ll2, long long* ll3 )
{
    OAC_TRACE4( v, ll1, ll2, ll3 );
    const RTresult _res = _rtVariableGet3ll( v, ll1, ll2, ll3 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet3llv( RTvariable v, long long* ll )
{
    OAC_TRACE2( v, ll );
    const RTresult _res = _rtVariableGet3llv( v, ll );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet3ull( RTvariable v, unsigned long long* ull1, unsigned long long* ull2, unsigned long long* ull3 )
{
    OAC_TRACE4( v, ull1, ull2, ull3 );
    const RTresult _res = _rtVariableGet3ull( v, ull1, ull2, ull3 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet3ullv( RTvariable v, unsigned long long* ull )
{
    OAC_TRACE2( v, ull );
    const RTresult _res = _rtVariableGet3ullv( v, ull );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet4f( RTvariable v, float* f1, float* f2, float* f3, float* f4 )
{
    OAC_TRACE5( v, f1, f2, f3, f4 );
    const RTresult _res = _rtVariableGet4f( v, f1, f2, f3, f4 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet4fv( RTvariable v, float* f )
{
    OAC_TRACE2( v, f );
    const RTresult _res = _rtVariableGet4fv( v, f );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet4i( RTvariable v, int* i1, int* i2, int* i3, int* i4 )
{
    OAC_TRACE5( v, i1, i2, i3, i4 );
    const RTresult _res = _rtVariableGet4i( v, i1, i2, i3, i4 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet4iv( RTvariable v, int* i )
{
    OAC_TRACE2( v, i );
    const RTresult _res = _rtVariableGet4iv( v, i );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet4ui( RTvariable v, unsigned int* u1, unsigned int* u2, unsigned int* u3, unsigned int* u4 )
{
    OAC_TRACE5( v, u1, u2, u3, u4 );
    const RTresult _res = _rtVariableGet4ui( v, u1, u2, u3, u4 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet4uiv( RTvariable v, unsigned int* u )
{
    OAC_TRACE2( v, u );
    const RTresult _res = _rtVariableGet4uiv( v, u );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet4ll( RTvariable v, long long* ll1, long long* ll2, long long* ll3, long long* ll4 )
{
    OAC_TRACE5( v, ll1, ll2, ll3, ll4 );
    const RTresult _res = _rtVariableGet4ll( v, ll1, ll2, ll3, ll4 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet4llv( RTvariable v, long long* ll )
{
    OAC_TRACE2( v, ll );
    const RTresult _res = _rtVariableGet4llv( v, ll );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet4ull( RTvariable v, unsigned long long* ull1, unsigned long long* ull2, unsigned long long* ull3, unsigned long long* ull4 )
{
    OAC_TRACE5( v, ull1, ull2, ull3, ull4 );
    const RTresult _res = _rtVariableGet4ull( v, ull1, ull2, ull3, ull4 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGet4ullv( RTvariable v, unsigned long long* ull )
{
    OAC_TRACE2( v, ull );
    const RTresult _res = _rtVariableGet4ullv( v, ull );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetAnnotation( RTvariable v, const char** annotation_return )
{
    OAC_TRACE2( v, annotation_return );
    const RTresult _res = _rtVariableGetAnnotation( v, annotation_return );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetContext( RTvariable v, RTcontext* context )
{
    OAC_TRACE2( v, context );
    const RTresult _res = _rtVariableGetContext( v, context );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetMatrix2x2fv( RTvariable v, int transpose, float* m )
{
    OAC_TRACE3( v, transpose, m );
    const RTresult _res = _rtVariableGetMatrix2x2fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetMatrix2x3fv( RTvariable v, int transpose, float* m )
{
    OAC_TRACE3( v, transpose, m );
    const RTresult _res = _rtVariableGetMatrix2x3fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetMatrix2x4fv( RTvariable v, int transpose, float* m )
{
    OAC_TRACE3( v, transpose, m );
    const RTresult _res = _rtVariableGetMatrix2x4fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetMatrix3x2fv( RTvariable v, int transpose, float* m )
{
    OAC_TRACE3( v, transpose, m );
    const RTresult _res = _rtVariableGetMatrix3x2fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetMatrix3x3fv( RTvariable v, int transpose, float* m )
{
    OAC_TRACE3( v, transpose, m );
    const RTresult _res = _rtVariableGetMatrix3x3fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetMatrix3x4fv( RTvariable v, int transpose, float* m )
{
    OAC_TRACE3( v, transpose, m );
    const RTresult _res = _rtVariableGetMatrix3x4fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetMatrix4x2fv( RTvariable v, int transpose, float* m )
{
    OAC_TRACE3( v, transpose, m );
    const RTresult _res = _rtVariableGetMatrix4x2fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetMatrix4x3fv( RTvariable v, int transpose, float* m )
{
    OAC_TRACE3( v, transpose, m );
    const RTresult _res = _rtVariableGetMatrix4x3fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetMatrix4x4fv( RTvariable v, int transpose, float* m )
{
    OAC_TRACE3( v, transpose, m );
    const RTresult _res = _rtVariableGetMatrix4x4fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetName( RTvariable v, const char** name_return )
{
    OAC_TRACE2( v, name_return );
    const RTresult _res = _rtVariableGetName( v, name_return );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetObject( RTvariable v, RTobject* object )
{
    OAC_TRACE2( v, object );
    const RTresult _res = _rtVariableGetObject( v, object );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetSize( RTvariable v, RTsize* size )
{
    OAC_TRACE2( v, size );
    const RTresult _res = _rtVariableGetSize( v, size );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetType( RTvariable v, RTobjecttype* type_return )
{
    OAC_TRACE2( v, type_return );
    const RTresult _res = _rtVariableGetType( v, type_return );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableGetUserData( RTvariable v, RTsize size, void* ptr )
{
    OAC_TRACE3( v, size, ptr );
    const RTresult _res = _rtVariableGetUserData( v, size, ptr );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet1f( RTvariable v, float f1 )
{
    OAC_TRACE2( v, f1 );
    const RTresult _res = _rtVariableSet1f( v, f1 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet1fv( RTvariable v, const float* f )
{
    OAC_TRACE2( v, f );
    OAC_VALUES( 1, f );
    const RTresult _res = _rtVariableSet1fv( v, f );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet1i( RTvariable v, int i1 )
{
    OAC_TRACE2( v, i1 );
    const RTresult _res = _rtVariableSet1i( v, i1 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet1iv( RTvariable v, const int* i )
{
    OAC_TRACE2( v, i );
    OAC_VALUES( 1, i );
    const RTresult _res = _rtVariableSet1iv( v, i );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet1ui( RTvariable v, unsigned int u1 )
{
    OAC_TRACE2( v, u1 );
    const RTresult _res = _rtVariableSet1ui( v, u1 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet1uiv( RTvariable v, const unsigned int* u )
{
    OAC_TRACE2( v, u );
    OAC_VALUES( 1, u );
    const RTresult _res = _rtVariableSet1uiv( v, u );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet1ll( RTvariable v, long long ll1 )
{
    OAC_TRACE2( v, ll1 );
    const RTresult _res = _rtVariableSet1ll( v, ll1 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet1llv( RTvariable v, const long long* ll )
{
    OAC_TRACE2( v, ll );
    OAC_VALUES( 1, ll );
    const RTresult _res = _rtVariableSet1llv( v, ll );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet1ull( RTvariable v, unsigned long long ull1 )
{
    OAC_TRACE2( v, ull1 );
    const RTresult _res = _rtVariableSet1ull( v, ull1 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet1ullv( RTvariable v, const unsigned long long* ull )
{
    OAC_TRACE2( v, ull );
    OAC_VALUES( 1, ull );
    const RTresult _res = _rtVariableSet1ullv( v, ull );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet2f( RTvariable v, float f1, float f2 )
{
    OAC_TRACE3( v, f1, f2 );
    const RTresult _res = _rtVariableSet2f( v, f1, f2 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet2fv( RTvariable v, const float* f )
{
    OAC_TRACE2( v, f );
    OAC_VALUES( 2, f );
    const RTresult _res = _rtVariableSet2fv( v, f );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet2i( RTvariable v, int i1, int i2 )
{
    OAC_TRACE3( v, i1, i2 );
    const RTresult _res = _rtVariableSet2i( v, i1, i2 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet2iv( RTvariable v, const int* i )
{
    OAC_TRACE2( v, i );
    OAC_VALUES( 2, i );
    const RTresult _res = _rtVariableSet2iv( v, i );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet2ui( RTvariable v, unsigned int u1, unsigned int u2 )
{
    OAC_TRACE3( v, u1, u2 );
    const RTresult _res = _rtVariableSet2ui( v, u1, u2 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet2uiv( RTvariable v, const unsigned int* u )
{
    OAC_TRACE2( v, u );
    OAC_VALUES( 2, u );
    const RTresult _res = _rtVariableSet2uiv( v, u );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet2ll( RTvariable v, long long ll1, long long ll2 )
{
    OAC_TRACE3( v, ll1, ll2 );
    const RTresult _res = _rtVariableSet2ll( v, ll1, ll2 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet2llv( RTvariable v, const long long* ll )
{
    OAC_TRACE2( v, ll );
    OAC_VALUES( 2, ll );
    const RTresult _res = _rtVariableSet2llv( v, ll );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet2ull( RTvariable v, unsigned long long ull1, unsigned long long ull2 )
{
    OAC_TRACE3( v, ull1, ull2 );
    const RTresult _res = _rtVariableSet2ull( v, ull1, ull2 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet2ullv( RTvariable v, const unsigned long long* ull )
{
    OAC_TRACE2( v, ull );
    OAC_VALUES( 2, ull );
    const RTresult _res = _rtVariableSet2ullv( v, ull );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet3f( RTvariable v, float f1, float f2, float f3 )
{
    OAC_TRACE4( v, f1, f2, f3 );
    const RTresult _res = _rtVariableSet3f( v, f1, f2, f3 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet3fv( RTvariable v, const float* f )
{
    OAC_TRACE2( v, f );
    OAC_VALUES( 3, f );
    const RTresult _res = _rtVariableSet3fv( v, f );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet3i( RTvariable v, int i1, int i2, int i3 )
{
    OAC_TRACE4( v, i1, i2, i3 );
    const RTresult _res = _rtVariableSet3i( v, i1, i2, i3 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet3iv( RTvariable v, const int* i )
{
    OAC_TRACE2( v, i );
    OAC_VALUES( 3, i );
    const RTresult _res = _rtVariableSet3iv( v, i );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet3ui( RTvariable v, unsigned int u1, unsigned int u2, unsigned int u3 )
{
    OAC_TRACE4( v, u1, u2, u3 );
    const RTresult _res = _rtVariableSet3ui( v, u1, u2, u3 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet3uiv( RTvariable v, const unsigned int* u )
{
    OAC_TRACE2( v, u );
    OAC_VALUES( 3, u );
    const RTresult _res = _rtVariableSet3uiv( v, u );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet3ll( RTvariable v, long long ll1, long long ll2, long long ll3 )
{
    OAC_TRACE4( v, ll1, ll2, ll3 );
    const RTresult _res = _rtVariableSet3ll( v, ll1, ll2, ll3 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet3llv( RTvariable v, const long long* ll )
{
    OAC_TRACE2( v, ll );
    OAC_VALUES( 3, ll );
    const RTresult _res = _rtVariableSet3llv( v, ll );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet3ull( RTvariable v, unsigned long long ull1, unsigned long long ull2, unsigned long long ull3 )
{
    OAC_TRACE4( v, ull1, ull2, ull3 );
    const RTresult _res = _rtVariableSet3ull( v, ull1, ull2, ull3 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet3ullv( RTvariable v, const unsigned long long* ull )
{
    OAC_TRACE2( v, ull );
    OAC_VALUES( 3, ull );
    const RTresult _res = _rtVariableSet3ullv( v, ull );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet4f( RTvariable v, float f1, float f2, float f3, float f4 )
{
    OAC_TRACE5( v, f1, f2, f3, f4 );
    const RTresult _res = _rtVariableSet4f( v, f1, f2, f3, f4 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet4fv( RTvariable v, const float* f )
{
    OAC_TRACE2( v, f );
    OAC_VALUES( 4, f );
    const RTresult _res = _rtVariableSet4fv( v, f );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet4i( RTvariable v, int i1, int i2, int i3, int i4 )
{
    OAC_TRACE5( v, i1, i2, i3, i4 );
    const RTresult _res = _rtVariableSet4i( v, i1, i2, i3, i4 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet4iv( RTvariable v, const int* i )
{
    OAC_TRACE2( v, i );
    OAC_VALUES( 4, i );
    const RTresult _res = _rtVariableSet4iv( v, i );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet4ui( RTvariable v, unsigned int u1, unsigned int u2, unsigned int u3, unsigned int u4 )
{
    OAC_TRACE5( v, u1, u2, u3, u4 );
    const RTresult _res = _rtVariableSet4ui( v, u1, u2, u3, u4 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet4uiv( RTvariable v, const unsigned int* u )
{
    OAC_TRACE2( v, u );
    OAC_VALUES( 4, u );
    const RTresult _res = _rtVariableSet4uiv( v, u );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet4ll( RTvariable v, long long ll1, long long ll2, long long ll3, long long ll4 )
{
    OAC_TRACE5( v, ll1, ll2, ll3, ll4 );
    const RTresult _res = _rtVariableSet4ll( v, ll1, ll2, ll3, ll4 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet4llv( RTvariable v, const long long* ll )
{
    OAC_TRACE2( v, ll );
    OAC_VALUES( 4, ll );
    const RTresult _res = _rtVariableSet4llv( v, ll );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet4ull( RTvariable v, unsigned long long ull1, unsigned long long ull2, unsigned long long ull3, unsigned long long ull4 )
{
    OAC_TRACE5( v, ull1, ull2, ull3, ull4 );
    const RTresult _res = _rtVariableSet4ull( v, ull1, ull2, ull3, ull4 );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSet4ullv( RTvariable v, const unsigned long long* ull )
{
    OAC_TRACE2( v, ull );
    OAC_VALUES( 4, ull );
    const RTresult _res = _rtVariableSet4ullv( v, ull );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSetMatrix2x2fv( RTvariable v, int transpose, const float* m )
{
    OAC_TRACE3( v, transpose, m );
    OAC_VALUES( 4, m );
    const RTresult _res = _rtVariableSetMatrix2x2fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSetMatrix2x3fv( RTvariable v, int transpose, const float* m )
{
    OAC_TRACE3( v, transpose, m );
    OAC_VALUES( 6, m );
    const RTresult _res = _rtVariableSetMatrix2x3fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSetMatrix2x4fv( RTvariable v, int transpose, const float* m )
{
    OAC_TRACE3( v, transpose, m );
    OAC_VALUES( 8, m );
    const RTresult _res = _rtVariableSetMatrix2x4fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSetMatrix3x2fv( RTvariable v, int transpose, const float* m )
{
    OAC_TRACE3( v, transpose, m );
    OAC_VALUES( 6, m );
    const RTresult _res = _rtVariableSetMatrix3x2fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSetMatrix3x3fv( RTvariable v, int transpose, const float* m )
{
    OAC_TRACE3( v, transpose, m );
    OAC_VALUES( 9, m );
    const RTresult _res = _rtVariableSetMatrix3x3fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSetMatrix3x4fv( RTvariable v, int transpose, const float* m )
{
    OAC_TRACE3( v, transpose, m );
    OAC_VALUES( 12, m );
    const RTresult _res = _rtVariableSetMatrix3x4fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSetMatrix4x2fv( RTvariable v, int transpose, const float* m )
{
    OAC_TRACE3( v, transpose, m );
    OAC_VALUES( 8, m );
    const RTresult _res = _rtVariableSetMatrix4x2fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSetMatrix4x3fv( RTvariable v, int transpose, const float* m )
{
    OAC_TRACE3( v, transpose, m );
    OAC_VALUES( 12, m );
    const RTresult _res = _rtVariableSetMatrix4x3fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSetMatrix4x4fv( RTvariable v, int transpose, const float* m )
{
    OAC_TRACE3( v, transpose, m );
    OAC_VALUES( 16, m );
    const RTresult _res = _rtVariableSetMatrix4x4fv( v, transpose, m );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSetObject( RTvariable v, RTobject object )
{
    OAC_TRACE2( v, object );
    const RTresult _res = _rtVariableSetObject( v, object );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtVariableSetUserData( RTvariable v, RTsize size, const void* ptr )
{
    OAC_TRACE3( v, size, ptr );
    OAC_BUFFER( size, ptr );
    const RTresult _res = _rtVariableSetUserData( v, size, ptr );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

#ifdef _WIN32  ////////////////////////// Windows-only section ///////////////////////////////

RTresult RTAPI rtDeviceGetWGLDevice( int* device, HGPULW hGpu )
{
    OAC_TRACE2( device, hGpu );
    RTresult _res = _rtDeviceGetWGLDevice( device, hGpu );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferCreateFromD3D10Resource( RTcontext context_api, unsigned int type, ID3D10Resource* pResource, RTbuffer* buffer )
{
    OAC_TRACE4( context_api, type, pResource, buffer );
    OAC_COMMENT( bufferdesc2string( type ) );
    RTresult _res = _rtBufferCreateFromD3D10Resource( context_api, type, pResource, buffer );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *buffer );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferCreateFromD3D11Resource( RTcontext context_api, unsigned int type, ID3D11Resource* pResource, RTbuffer* buffer )
{
    OAC_TRACE4( context_api, type, pResource, buffer );
    OAC_COMMENT( bufferdesc2string( type ) );
    RTresult _res = _rtBufferCreateFromD3D11Resource( context_api, type, pResource, buffer );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *buffer );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferCreateFromD3D9Resource( RTcontext context_api, unsigned int type, IDirect3DResource9* pResource, RTbuffer* buffer )
{
    OAC_TRACE4( context_api, type, pResource, buffer );
    OAC_COMMENT( bufferdesc2string( type ) );
    RTresult _res = _rtBufferCreateFromD3D9Resource( context_api, type, pResource, buffer );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *buffer );

    /*
  if( getApiCapture().capture_enabled() && _res == RT_SUCCESS )
    getApiCapture().capture_d3d9_buffer_info( pResource );
   */

    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferD3D10Register( RTbuffer buffer_api )
{
    OAC_TRACE1( buffer_api );
    RTresult _res = _rtBufferD3D10Register( buffer_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferD3D10Unregister( RTbuffer buffer_api )
{
    OAC_TRACE1( buffer_api );
    RTresult _res = _rtBufferD3D10Unregister( buffer_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferD3D11Register( RTbuffer buffer_api )
{
    OAC_TRACE1( buffer_api );
    RTresult _res = _rtBufferD3D11Register( buffer_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferD3D11Unregister( RTbuffer buffer_api )
{
    OAC_TRACE1( buffer_api );
    RTresult _res = _rtBufferD3D11Unregister( buffer_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferD3D9Register( RTbuffer buffer_api )
{
    OAC_TRACE1( buffer_api );
    RTresult _res = _rtBufferD3D9Register( buffer_api );
    OAC_RESULT( _res );

    /*
  if( getApiCapture().capture_enabled() && _res == RT_SUCCESS )
  {
    Buffer* buffer = (Buffer*)buffer_api;
    BufferD3D* buffer_d3d = dynamic_cast<BufferD3D*>( buffer );
    if( buffer && buffer_d3d &&
        buffer->getClass() == RT_OBJECT_BUFFER )
    {
      IDirect3DResource9* resource = buffer_d3d->getD3D9Resource();
      getApiCapture().capture_d3d9_buffer_info( resource );
    }
  }
  */

    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferD3D9Unregister( RTbuffer buffer_api )
{
    OAC_TRACE1( buffer_api );
    RTresult _res = _rtBufferD3D9Unregister( buffer_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetD3D10Resource( RTbuffer buffer_api, ID3D10Resource** pResource )
{
    OAC_TRACE2( buffer_api, pResource );
    RTresult _res = _rtBufferGetD3D10Resource( buffer_api, pResource );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetD3D11Resource( RTbuffer buffer_api, ID3D11Resource** pResource )
{
    OAC_TRACE2( buffer_api, pResource );
    RTresult _res = _rtBufferGetD3D11Resource( buffer_api, pResource );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtBufferGetD3D9Resource( RTbuffer buffer_api, IDirect3DResource9** pResource )
{
    OAC_TRACE2( buffer_api, pResource );
    RTresult _res = _rtBufferGetD3D9Resource( buffer_api, pResource );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetD3D10Device( RTcontext context_api, ID3D10Device* matchingDevice )
{
    OAC_TRACE2( context_api, matchingDevice );
    RTresult _res = _rtContextSetD3D10Device( context_api, matchingDevice );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetD3D11Device( RTcontext context_api, ID3D11Device* matchingDevice )
{
    OAC_TRACE2( context_api, matchingDevice );
    RTresult _res = _rtContextSetD3D11Device( context_api, matchingDevice );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtContextSetD3D9Device( RTcontext context_api, IDirect3DDevice9* matchingDevice )
{
    OAC_TRACE2( context_api, matchingDevice );
    RTresult _res = _rtContextSetD3D9Device( context_api, matchingDevice );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtDeviceGetD3D10Device( int* device, IDXGIAdapter* pAdapter )
{
    OAC_TRACE2( device, pAdapter );
    RTresult _res = _rtDeviceGetD3D10Device( device, pAdapter );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtDeviceGetD3D11Device( int* device, IDXGIAdapter* pAdapter )
{
    OAC_TRACE2( device, pAdapter );
    RTresult _res = _rtDeviceGetD3D11Device( device, pAdapter );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtDeviceGetD3D9Device( int* device, const char* pszAdapterName )
{
    OAC_TRACE2( device, pszAdapterName );
    RTresult _res = _rtDeviceGetD3D9Device( device, pszAdapterName );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerCreateFromD3D10Resource( RTcontext context_api, ID3D10Resource* pResource, RTtexturesampler* textureSampler )
{
    OAC_TRACE3( context_api, pResource, textureSampler );
    RTresult _res = _rtTextureSamplerCreateFromD3D10Resource( context_api, pResource, textureSampler );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *textureSampler );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerCreateFromD3D11Resource( RTcontext context_api, ID3D11Resource* pResource, RTtexturesampler* textureSampler )
{
    OAC_TRACE3( context_api, pResource, textureSampler );
    RTresult _res = _rtTextureSamplerCreateFromD3D11Resource( context_api, pResource, textureSampler );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *textureSampler );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerCreateFromD3D9Resource( RTcontext context_api, IDirect3DResource9* pResource, RTtexturesampler* textureSampler )
{
    OAC_TRACE3( context_api, pResource, textureSampler );
    RTresult _res = _rtTextureSamplerCreateFromD3D9Resource( context_api, pResource, textureSampler );
    OAC_RESULT( _res );
    OAC_NEW_HANDLE( *textureSampler );

    /*
  if( getApiCapture().capture_enabled() && _res == RT_SUCCESS )
    getApiCapture().capture_d3d9_image_info( pResource );
  */

    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerD3D10Register( RTtexturesampler textureSampler_api )
{
    OAC_TRACE1( textureSampler_api );
    RTresult _res = _rtTextureSamplerD3D10Register( textureSampler_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerD3D10Unregister( RTtexturesampler textureSampler_api )
{
    OAC_TRACE1( textureSampler_api );
    RTresult _res = _rtTextureSamplerD3D10Unregister( textureSampler_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerD3D11Register( RTtexturesampler textureSampler_api )
{
    OAC_TRACE1( textureSampler_api );
    RTresult _res = _rtTextureSamplerD3D11Register( textureSampler_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerD3D11Unregister( RTtexturesampler textureSampler_api )
{
    OAC_TRACE1( textureSampler_api );
    RTresult _res = _rtTextureSamplerD3D11Unregister( textureSampler_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerD3D9Register( RTtexturesampler textureSampler_api )
{
    OAC_TRACE1( textureSampler_api );
    RTresult _res = _rtTextureSamplerD3D9Register( textureSampler_api );
    OAC_RESULT( _res );

    /*
  if( getApiCapture().capture_enabled() && _res == RT_SUCCESS )
  {
    TextureSampler* sampler = (TextureSampler*)textureSampler_api;
    sampler->updateInteropBuffers(); // see bug 794273
    Buffer* buffer = sampler->getBuffer();
    BufferD3D* buffer_d3d = dynamic_cast<BufferD3D*>( buffer );
    if( buffer && buffer_d3d &&
        buffer->getClass() == RT_OBJECT_BUFFER )
    {
      IDirect3DResource9* resource = buffer_d3d->getD3D9Resource();
      getApiCapture().capture_d3d9_image_info( resource );
    }
  }
  */

    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerD3D9Unregister( RTtexturesampler textureSampler_api )
{
    OAC_TRACE1( textureSampler_api );
    RTresult _res = _rtTextureSamplerD3D9Unregister( textureSampler_api );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetD3D10Resource( RTtexturesampler textureSampler_api, ID3D10Resource** pResource )
{
    OAC_TRACE2( textureSampler_api, pResource );
    RTresult _res = _rtTextureSamplerGetD3D10Resource( textureSampler_api, pResource );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetD3D11Resource( RTtexturesampler textureSampler_api, ID3D11Resource** pResource )
{
    OAC_TRACE2( textureSampler_api, pResource );
    RTresult _res = _rtTextureSamplerGetD3D11Resource( textureSampler_api, pResource );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

RTresult RTAPI rtTextureSamplerGetD3D9Resource( RTtexturesampler textureSampler_api, IDirect3DResource9** pResource )
{
    OAC_TRACE2( textureSampler_api, pResource );
    RTresult _res = _rtTextureSamplerGetD3D9Resource( textureSampler_api, pResource );
    OAC_RESULT( _res );
    OAC_RETURN( _res );
}

#endif  ////////////////////////// END Windows-only section ///////////////////////////////

}  // extern "C"
