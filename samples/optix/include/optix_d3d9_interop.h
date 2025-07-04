
/*
 * Copyright (c) 2021 LWPU Corporation.  All rights reserved.
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

 /**
 * @file   optix_d3d9_interop.h
 * @author LWPU Corporation
 * @brief  OptiX public API declarations D3D9 interop
 *
 * OptiX public API declarations for D3D9 interoperability
 */

#ifndef __optix_optix_dx9_interop_h__
#define __optix_optix_dx9_interop_h__

/************************************
 **
 **    DX9 Interop functions
 **
 ***********************************/

/*

On Windows you will need to include windows.h before this file:

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include<windows.h>
#endif
#include <optix_d3d9_interop.h>

*/

#if defined( _WIN32 )

#ifdef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#include "optix.h"
#else // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#include "o6/optix.h"
#endif // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD

/** IDirect3DDevice9 structure */
typedef struct IDirect3DDevice9   IDirect3DDevice9;
/** IDirect3DResource9 structure */
typedef struct IDirect3DResource9 IDirect3DResource9;

#ifdef __cplusplus
  extern "C"
  {
#endif

  /**
  * @brief Binds a D3D9 device to a context and enables interop
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetD3D9Device binds \a device to \a context and enables D3D9 interop capabilities in \a context. This
  * function must be exelwted once for \a context before any call to @ref rtBufferCreateFromD3D9Resource or @ref rtTextureSamplerCreateFromD3D9Resource can
  * take place. A context can only be bound to one device. Once \a device is bound to \a context, the binding is immutable and remains upon destruction of \a context.
  *
  * @param[in]   context      The context to bind the device with
  * @param[in]   device       The D3D9 device to be used for interop with the associated context
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_VALUE
  * - @ref RT_ERROR_ILWALID_CONTEXT
  *
  * <B>History</B>
  *
  * @ref rtContextSetD3D9Device was introduced in OptiX 2.0.
  *
  * <B>See also</B>
  * @ref rtBufferCreateFromD3D9Resource,
  * @ref rtTextureSamplerCreateFromD3D9Resource
  *
  */
  RTresult RTAPI rtContextSetD3D9Device                ( RTcontext context, IDirect3DDevice9* device );

  /**
  * @brief Returns the OptiX device number associated with the specified name of a D3D9 adapter
  *
  * @ingroup ContextFreeFunctions
  *
  * <B>Description</B>
  *
  * @ref rtDeviceGetD3D9Device returns in \a device the OptiX device ID of the adapter represented by \a pszAdapterName.
  * \a pszAdapterName is the DeviceName field in the \a D3DADAPTER_IDENTIFIER9 struct. In combination with @ref rtContextSetDevices,
  * this function can be used to restrict OptiX to use only one device. The same device the D3D9 commands will be sent to.
  *
  * This function is only supported on Windows platforms.
  *
  * @param[in]   device               A handle to the memory location where the OptiX device ordinal associated with \a pszAdapterName will be stored
  * @param[out]   pszAdapterName      The name of an adapter as can be found in the DeviceName field in the \a D3DADAPTER_IDENTIFIER9 struct
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtDeviceGetD3D9Device was introduced in OptiX 2.5.
  *
  * <B>See also</B>
  * @ref rtDeviceGetDeviceCount
  *
  */
  RTresult RTAPI rtDeviceGetD3D9Device                 ( int* device, const char* pszAdapterName );

  /**
  * @brief Creates a new buffer object from a D3D9 resource
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferCreateFromD3D9Resource allocates and returns a handle to a new buffer object in \a *buffer associated with
  * \a context. If the allocated size of the D3D resource is \a 0, @ref RT_ERROR_MEMORY_ALLOCATION_FAILED will be returned. Supported D3D9 buffer types are:
  *
  * - IDirect3DVertexBuffer9
  * - IDirect3DIndexBuffer9
  *
  * These buffers can be used to share data with D3D9; changes of the content in \a buffer, either done by D3D9 or OptiX,
  * will be reflected automatically in both APIs. If the size, or format, of a D3D9 buffer is changed, appropriate OptiX
  * calls have to be used to update \a buffer accordingly. OptiX keeps only a reference to D3D9 data, when \a buffer is
  * destroyed, the state of \a resource is unaltered.
  *
  * The \a type of this buffer is specified by one of the following values in \a bufferdesc:
  *
  * - @ref RT_BUFFER_INPUT
  * - @ref RT_BUFFER_OUTPUT
  * - @ref RT_BUFFER_INPUT_OUTPUT
  *
  * The type values are used to specify the direction of data flow from the host to the OptiX devices.
  * @ref RT_BUFFER_INPUT specifies that the host may only write to the buffer and the device may only read from the buffer.
  * @ref RT_BUFFER_OUTPUT specifies the opposite, read only access on the host and write only access on the device.
  * Devices and the host may read and write from buffers of type @ref RT_BUFFER_INPUT_OUTPUT.  Reading or writing to
  * a buffer of the incorrect type (e.g., the host writing to a buffer of type @ref RT_BUFFER_OUTPUT) is undefined.
  *
  * Flags can be used to optimize data transfers between the host and it's devices. Lwrrently no \a flags are supported for
  * interop buffers.
  *
  * @param[in]   context              The context to create the buffer in
  * @param[in]   bufferdesc           Bitwise \a or combination of the \a type and \a flags of the new buffer
  * @param[in]   resource             The D3D9 resource handle for use in OptiX
  * @param[out]  buffer               The return handle for the buffer object
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_VALUE
  * - @ref RT_ERROR_ILWALID_CONTEXT
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferCreateFromD3D9Resource was introduced in OptiX 2.0.
  *
  * <B>See also</B>
  * @ref rtBufferCreate,
  * @ref rtBufferDestroy
  *
  */
  RTresult RTAPI rtBufferCreateFromD3D9Resource        ( RTcontext context, unsigned int bufferdesc, IDirect3DResource9* resource,  RTbuffer* buffer );

  /**
  * @brief Creates a new texture sampler object from a D3D9 resource
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerCreateFromD3D9Resource allocates and returns a
  * handle to a new texture sampler object in \a *texturesampler
  * associated with \a context. If the allocated size of the D3D resource
  * is \a 0, @ref RT_ERROR_MEMORY_ALLOCATION_FAILED will be returned. Supported
  * D3D9 texture types are:
  *
  * - IDirect3DSurface9
  *
  * - (derivatives of) IDirect3DBaseTexture9
  *
  * These texture samplers can be used to share data with D3D9; changes of
  * the content and size of \a texturesampler done by D3D9 will be
  * reflected automatically in OptiX. Lwrrently texture sampler data are
  * read only in OptiX programs. OptiX keeps only a reference to
  * D3D9 data, when \a texturesampler is destroyed, the state of the
  * \a resource is unaltered.
  *
  * The array size and number of mipmap levels can't be changed for
  * texture samplers that encapsulate a D3D9 resource. Furthermore no
  * buffer objects can be queried. Please refer to the @ref InteropTypes for a
  * complete list of supported texture formats.
  *
  * @param[in]   context              The context to create the texture sampler in
  * @param[in]   resource             The D3D9 resource handle for use in OptiX
  * @param[out]  textureSampler       The return handle for the texture sampler object
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_VALUE
  * - @ref RT_ERROR_ILWALID_CONTEXT
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerCreateFromD3D9Resource was introduced in OptiX 2.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerCreate,
  * @ref rtTextureSamplerDestroy
  *
  */
  RTresult RTAPI rtTextureSamplerCreateFromD3D9Resource( RTcontext context, IDirect3DResource9* resource,  RTtexturesampler* textureSampler );

  /**
  * @brief Gets the D3D9 resource associated with this buffer
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetD3D9Resource stores the D3D9 resource pointer in \a **resource if \a buffer was created with
  * @ref rtBufferCreateFromD3D9Resource.  If \a buffer was not created from a D3D9 resource \a **resource will be \a 0 after
  * the call and @ref RT_ERROR_ILWALID_VALUE is returned.
  *
  * @param[in]    buffer              The buffer to be queried for its D3D9 resource
  * @param[out]   resource            The return handle for the resource
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_VALUE
  * - @ref RT_ERROR_ILWALID_CONTEXT
  *
  * <B>History</B>
  *
  * @ref rtBufferGetD3D9Resource was introduced in OptiX 2.0.
  *
  * <B>See also</B>
  * @ref rtBufferCreateFromD3D9Resource
  *
  */
  RTresult RTAPI rtBufferGetD3D9Resource               ( RTbuffer buffer, IDirect3DResource9** resource );

  /**
  * @brief Gets the D3D9 resource associated with this texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerGetD3D9Resource stores the D3D9 resource pointer in \a **resource if \a sampler was created with
  * @ref rtTextureSamplerGetD3D9Resource.  If \a sampler was not created from a D3D9 resource \a resource will be 0 after
  * the call and @ref RT_ERROR_ILWALID_VALUE is returned
  *
  * @param[in]    textureSampler              The texture sampler to be queried for its D3D9 resource
  * @param[out]   pResource                   The return handle for the resource
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_VALUE
  * - @ref RT_ERROR_ILWALID_CONTEXT
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerGetD3D9Resource was introduced in OptiX 2.0.
  *
  * <B>See also</B>
  * @ref rtBufferCreateFromD3D9Resource
  *
  */
  RTresult RTAPI rtTextureSamplerGetD3D9Resource       ( RTtexturesampler textureSampler, IDirect3DResource9** pResource );

  /**
  * @brief Declares a D3D9 buffer as immutable and accessible by OptiX
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * An OptiX buffer in an unregistered state can be registered to OptiX again via @ref rtBufferD3D9Register. Once registered,
  * properties like the size of the original D3D9 resource cannot be modified anymore. Calls to the corresponding D3D9 functions
  * will return with an error code. However, the data of the D3D9 resource can still be read and written by the appropriate D3D9 commands.
  * When a buffer is already in a registered state @ref rtBufferD3D9Register will return @ref RT_ERROR_RESOURCE_ALREADY_REGISTERED. A resource
  * must be registered in order to be used by OptiX. If a resource is not registered @ref RT_ERROR_ILWALID_VALUE will be returned.
  *
  * @param[in]    buffer              The handle for the buffer object
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_VALUE
  * - @ref RT_ERROR_ILWALID_CONTEXT
  * - @ref RT_ERROR_RESOURCE_ALREADY_REGISTERED
  *
  * <B>History</B>
  *
  * @ref rtBufferD3D9Register was introduced in OptiX 2.0.
  *
  * <B>See also</B>
  * @ref rtBufferCreateFromD3D11Resource
  *
  */
  RTresult RTAPI rtBufferD3D9Register                  ( RTbuffer buffer );

  /**
  * @brief Declares a D3D9 buffer as mutable and inaccessible by OptiX
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * An OptiX buffer in a registered state can be unregistered via @ref rtBufferD3D9Register. Once unregistered,
  * properties like the size of the original D3D9 resource can be changed. As long as a resource is unregistered,
  * OptiX will not be able to access the data and will fail with @ref RT_ERROR_ILWALID_VALUE. When a buffer is already
  * in an unregistered state @ref rtBufferD3D9Unregister will return @ref RT_ERROR_RESOURCE_NOT_REGISTERED.
  *
  * @param[in]    buffer              The handle for the buffer object
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_VALUE
  * - @ref RT_ERROR_ILWALID_CONTEXT
  * - @ref RT_ERROR_RESOURCE_NOT_REGISTERED
  *
  * <B>History</B>
  *
  * @ref rtBufferD3D9Unregister was introduced in OptiX 2.0.
  *
  * <B>See also</B>
  * @ref rtBufferCreateFromD3D11Resource
  *
  */
  RTresult RTAPI rtBufferD3D9Unregister                ( RTbuffer buffer );

  /**
  * @brief Declares a D3D9 texture as immutable and accessible by OptiX
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * An OptiX texture sampler in an unregistered state can be registered to OptiX again via @ref rtTextureSamplerD3D9Register.
  * Once registered, properties like the size of the original D3D9 resource cannot be modified anymore. Calls to the corresponding
  * D3D9 functions will return with an error code. However, the data of the D3D9 resource can still be read and written by the appropriate
  * D3D9 commands. When a texture sampler is already in a registered state @ref rtTextureSamplerD3D9Register will return @ref RT_ERROR_RESOURCE_ALREADY_REGISTERED.
  * A resource must be registered in order to be used by OptiX. If a resource is not registered @ref RT_ERROR_ILWALID_VALUE will be returned.
  *
  * @param[in]    textureSampler              The handle for the texture object
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_VALUE
  * - @ref RT_ERROR_ILWALID_CONTEXT
  * - @ref RT_ERROR_RESOURCE_ALREADY_REGISTERED
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerD3D9Register was introduced in OptiX 2.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerCreateFromD3D9Resource
  *
  */
  RTresult RTAPI rtTextureSamplerD3D9Register          ( RTtexturesampler textureSampler );

  /**
  * @brief Declares a D3D9 texture as mutable and inaccessible by OptiX
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * An OptiX texture sampler in a registered state can be unregistered via @ref rtTextureSamplerD3D9Unregister. Once unregistered,
  * properties like the size of the original D3D9 resource can be changed. As long as a resource is unregistered, OptiX will not
  * be able to access the data and will fail with @ref RT_ERROR_ILWALID_VALUE. When a buffer is already in an unregistered state
  * @ref rtBufferD3D9Unregister will return @ref RT_ERROR_RESOURCE_NOT_REGISTERED.
  *
  * @param[in]    textureSampler              The handle for the texture object
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_VALUE
  * - @ref RT_ERROR_ILWALID_CONTEXT
  * - @ref RT_ERROR_RESOURCE_NOT_REGISTERED
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerD3D9Unregister was introduced in OptiX 2.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerCreateFromD3D9Resource
  *
  */
  RTresult RTAPI rtTextureSamplerD3D9Unregister        ( RTtexturesampler textureSampler );

#ifdef __cplusplus
  }
#endif

#endif /* defined( _WIN32 ) */

#endif /* __optix_optix_dx9_interop_h__ */
