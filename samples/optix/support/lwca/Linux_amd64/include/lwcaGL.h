/*
 * Copyright 1993-2014 LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to LWPU intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and
 * conditions of a form of LWPU software license agreement by and
 * between LWPU and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of LWPU is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * LWPU DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL LWPU BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef LWDAGL_H
#define LWDAGL_H

#include <lwca.h>
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#if defined(__LWDA_API_VERSION_INTERNAL) || defined(__DOXYGEN_ONLY__) || defined(LWDA_ENABLE_DEPRECATED)
#define __LWDA_DEPRECATED
#elif defined(_MSC_VER)
#define __LWDA_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__)
#define __LWDA_DEPRECATED __attribute__((deprecated))
#else
#define __LWDA_DEPRECATED
#endif

#ifdef LWDA_FORCE_API_VERSION
#error "LWDA_FORCE_API_VERSION is no longer supported."
#endif

#if defined(__LWDA_API_VERSION_INTERNAL) || defined(LWDA_API_PER_THREAD_DEFAULT_STREAM)
    #define __LWDA_API_PER_THREAD_DEFAULT_STREAM
    #define __LWDA_API_PTDS(api) api ## _ptds
    #define __LWDA_API_PTSZ(api) api ## _ptsz
#else
    #define __LWDA_API_PTDS(api) api
    #define __LWDA_API_PTSZ(api) api
#endif

#define lwGLCtxCreate            lwGLCtxCreate_v2
#define lwGLMapBufferObject      __LWDA_API_PTDS(lwGLMapBufferObject_v2)
#define lwGLMapBufferObjectAsync __LWDA_API_PTSZ(lwGLMapBufferObjectAsync_v2)
#define lwGLGetDevices           lwGLGetDevices_v2

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \file lwdaGL.h
 * \brief Header file for the OpenGL interoperability functions of the
 * low-level LWCA driver application programming interface.
 */

/**
 * \defgroup LWDA_GL OpenGL Interoperability
 * \ingroup LWDA_DRIVER
 *
 * ___MANBRIEF___ OpenGL interoperability functions of the low-level LWCA
 * driver API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the OpenGL interoperability functions of the
 * low-level LWCA driver application programming interface. Note that mapping 
 * of OpenGL resources is performed with the graphics API agnostic, resource 
 * mapping interface described in \ref LWDA_GRAPHICS "Graphics Interoperability".
 *
 * @{
 */

#if defined(_WIN32)
#if !defined(WGL_LW_gpu_affinity)
typedef void* HGPULW;
#endif
#endif /* _WIN32 */

/**
 * \brief Registers an OpenGL buffer object
 *
 * Registers the buffer object specified by \p buffer for access by
 * LWCA.  A handle to the registered object is returned as \p
 * pLwdaResource.  The register flags \p Flags specify the intended usage,
 * as follows:
 *
 * - ::LW_GRAPHICS_REGISTER_FLAGS_NONE: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by LWCA. This is the default value.
 * - ::LW_GRAPHICS_REGISTER_FLAGS_READ_ONLY: Specifies that LWCA
 *   will not write to this resource.
 * - ::LW_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: Specifies that
 *   LWCA will not read from this resource and will write over the
 *   entire contents of the resource, so none of the data previously
 *   stored in the resource will be preserved.
 *
 * \param pLwdaResource - Pointer to the returned object handle
 * \param buffer - name of buffer object to be registered
 * \param Flags - Register flags
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ALREADY_MAPPED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * \notefnerr
 *
 * \sa 
 * ::lwGraphicsUnregisterResource,
 * ::lwGraphicsMapResources,
 * ::lwGraphicsResourceGetMappedPointer,
 * ::lwdaGraphicsGLRegisterBuffer
 */
LWresult LWDAAPI lwGraphicsGLRegisterBuffer(LWgraphicsResource *pLwdaResource, GLuint buffer, unsigned int Flags);

/**
 * \brief Register an OpenGL texture or renderbuffer object
 *
 * Registers the texture or renderbuffer object specified by \p image for access by LWCA.  
 * A handle to the registered object is returned as \p pLwdaResource.  
 *
 * \p target must match the type of the object, and must be one of ::GL_TEXTURE_2D, 
 * ::GL_TEXTURE_RECTANGLE, ::GL_TEXTURE_LWBE_MAP, ::GL_TEXTURE_3D, ::GL_TEXTURE_2D_ARRAY, 
 * or ::GL_RENDERBUFFER.
 *
 * The register flags \p Flags specify the intended usage, as follows:
 *
 * - ::LW_GRAPHICS_REGISTER_FLAGS_NONE: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by LWCA. This is the default value.
 * - ::LW_GRAPHICS_REGISTER_FLAGS_READ_ONLY: Specifies that LWCA
 *   will not write to this resource.
 * - ::LW_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: Specifies that
 *   LWCA will not read from this resource and will write over the
 *   entire contents of the resource, so none of the data previously
 *   stored in the resource will be preserved.
 * - ::LW_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST: Specifies that LWCA will
 *   bind this resource to a surface reference.
 * - ::LW_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER: Specifies that LWCA will perform
 *   texture gather operations on this resource.
 *
 * The following image formats are supported. For brevity's sake, the list is abbreviated.
 * For ex., {GL_R, GL_RG} X {8, 16} would expand to the following 4 formats 
 * {GL_R8, GL_R16, GL_RG8, GL_RG16} :
 * - GL_RED, GL_RG, GL_RGBA, GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA, GL_INTENSITY
 * - {GL_R, GL_RG, GL_RGBA} X {8, 16, 16F, 32F, 8UI, 16UI, 32UI, 8I, 16I, 32I}
 * - {GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA, GL_INTENSITY} X
 * {8, 16, 16F_ARB, 32F_ARB, 8UI_EXT, 16UI_EXT, 32UI_EXT, 8I_EXT, 16I_EXT, 32I_EXT}
 *
 * The following image classes are lwrrently disallowed:
 * - Textures with borders
 * - Multisampled renderbuffers
 *
 * \param pLwdaResource - Pointer to the returned object handle
 * \param image - name of texture or renderbuffer object to be registered
 * \param target - Identifies the type of object specified by \p image
 * \param Flags - Register flags
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ALREADY_MAPPED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * \notefnerr
 *
 * \sa 
 * ::lwGraphicsUnregisterResource,
 * ::lwGraphicsMapResources,
 * ::lwGraphicsSubResourceGetMappedArray,
 * ::lwdaGraphicsGLRegisterImage
 */
LWresult LWDAAPI lwGraphicsGLRegisterImage(LWgraphicsResource *pLwdaResource, GLuint image, GLenum target, unsigned int Flags);

#ifdef _WIN32
/**
 * \brief Gets the LWCA device associated with hGpu
 *
 * Returns in \p *pDevice the LWCA device associated with a \p hGpu, if
 * applicable.
 *
 * \param pDevice - Device associated with hGpu
 * \param hGpu    - Handle to a GPU, as queried via ::WGL_LW_gpu_affinity()
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwGLMapBufferObject,
 * ::lwGLRegisterBufferObject, ::lwGLUnmapBufferObject,
 * ::lwGLUnregisterBufferObject, ::lwGLUnmapBufferObjectAsync,
 * ::lwGLSetBufferObjectMapFlags,
 * ::lwdaWGLGetDevice
 */
LWresult LWDAAPI lwWGLGetDevice(LWdevice *pDevice, HGPULW hGpu);
#endif /* _WIN32 */

/**
 * LWCA devices corresponding to an OpenGL device
 */
typedef enum LWGLDeviceList_enum {
    LW_GL_DEVICE_LIST_ALL            = 0x01, /**< The LWCA devices for all GPUs used by the current OpenGL context */
    LW_GL_DEVICE_LIST_LWRRENT_FRAME  = 0x02, /**< The LWCA devices for the GPUs used by the current OpenGL context in its lwrrently rendering frame */
    LW_GL_DEVICE_LIST_NEXT_FRAME     = 0x03, /**< The LWCA devices for the GPUs to be used by the current OpenGL context in the next frame */
} LWGLDeviceList;

/**
 * \brief Gets the LWCA devices associated with the current OpenGL context
 *
 * Returns in \p *pLwdaDeviceCount the number of LWCA-compatible devices 
 * corresponding to the current OpenGL context. Also returns in \p *pLwdaDevices 
 * at most lwdaDeviceCount of the LWCA-compatible devices corresponding to 
 * the current OpenGL context. If any of the GPUs being used by the current OpenGL
 * context are not LWCA capable then the call will return LWDA_ERROR_NO_DEVICE.
 *
 * The \p deviceList argument may be any of the following:
 * - ::LW_GL_DEVICE_LIST_ALL: Query all devices used by the current OpenGL context.
 * - ::LW_GL_DEVICE_LIST_LWRRENT_FRAME: Query the devices used by the current OpenGL context to
 *   render the current frame (in SLI).
 * - ::LW_GL_DEVICE_LIST_NEXT_FRAME: Query the devices used by the current OpenGL context to
 *   render the next frame (in SLI). Note that this is a prediction, it can't be guaranteed that
 *   this is correct in all cases.
 *
 * \param pLwdaDeviceCount - Returned number of LWCA devices.
 * \param pLwdaDevices     - Returned LWCA devices.
 * \param lwdaDeviceCount  - The size of the output device array pLwdaDevices.
 * \param deviceList       - The set of devices to return.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_NO_DEVICE,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_GRAPHICS_CONTEXT
 *
 * \note This function is not supported on Mac OS X.
 * \notefnerr
 *
 * \sa
 * ::lwWGLGetDevice,
 * ::lwdaGLGetDevices
 */
LWresult LWDAAPI lwGLGetDevices(unsigned int *pLwdaDeviceCount, LWdevice *pLwdaDevices, unsigned int lwdaDeviceCount, LWGLDeviceList deviceList);

/**
 * \defgroup LWDA_GL_DEPRECATED OpenGL Interoperability [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated OpenGL interoperability functions of the low-level
 * LWCA driver API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes deprecated OpenGL interoperability functionality.
 *
 * @{
 */

/** Flags to map or unmap a resource */
typedef enum LWGLmap_flags_enum {
    LW_GL_MAP_RESOURCE_FLAGS_NONE          = 0x00,
    LW_GL_MAP_RESOURCE_FLAGS_READ_ONLY     = 0x01,
    LW_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02,    
} LWGLmap_flags;

/**
 * \brief Create a LWCA context for interoperability with OpenGL
 *
 * \deprecated This function is deprecated as of Lwca 5.0. 
 *
 * This function is deprecated and should no longer be used.  It is
 * no longer necessary to associate a LWCA context with an OpenGL
 * context in order to achieve maximum interoperability performance.
 *
 * \param pCtx   - Returned LWCA context
 * \param Flags  - Options for LWCA context creation
 * \param device - Device on which to create the context
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa ::lwCtxCreate, ::lwGLInit, ::lwGLMapBufferObject,
 * ::lwGLRegisterBufferObject, ::lwGLUnmapBufferObject,
 * ::lwGLUnregisterBufferObject, ::lwGLMapBufferObjectAsync,
 * ::lwGLUnmapBufferObjectAsync, ::lwGLSetBufferObjectMapFlags,
 * ::lwWGLGetDevice
 */
__LWDA_DEPRECATED LWresult LWDAAPI lwGLCtxCreate(LWcontext *pCtx, unsigned int Flags, LWdevice device );

/**
 * \brief Initializes OpenGL interoperability
 *
 * \deprecated This function is deprecated as of Lwca 3.0. 
 *
 * Initializes OpenGL interoperability. This function is deprecated
 * and calling it is no longer required. It may fail if the needed
 * OpenGL driver facilities are not available.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa ::lwGLMapBufferObject,
 * ::lwGLRegisterBufferObject, ::lwGLUnmapBufferObject,
 * ::lwGLUnregisterBufferObject, ::lwGLMapBufferObjectAsync,
 * ::lwGLUnmapBufferObjectAsync, ::lwGLSetBufferObjectMapFlags,
 * ::lwWGLGetDevice
 */
__LWDA_DEPRECATED LWresult LWDAAPI lwGLInit(void);

/**
 * \brief Registers an OpenGL buffer object
 *
 * \deprecated This function is deprecated as of Lwca 3.0. 
 *
 * Registers the buffer object specified by \p buffer for access by
 * LWCA. This function must be called before LWCA can map the buffer
 * object.  There must be a valid OpenGL context bound to the current
 * thread when this function is called, and the buffer name is
 * resolved by that context.
 *
 * \param buffer - The name of the buffer object to register.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ALREADY_MAPPED
 * \notefnerr
 *
 * \sa ::lwGraphicsGLRegisterBuffer
 */
__LWDA_DEPRECATED LWresult LWDAAPI lwGLRegisterBufferObject(GLuint buffer);

/**
 * \brief Maps an OpenGL buffer object
 *
 * \deprecated This function is deprecated as of Lwca 3.0. 
 *
 * Maps the buffer object specified by \p buffer into the address space of the
 * current LWCA context and returns in \p *dptr and \p *size the base pointer
 * and size of the resulting mapping.
 *
 * There must be a valid OpenGL context bound to the current thread
 * when this function is called.  This must be the same context, or a
 * member of the same shareGroup, as the context that was bound when
 * the buffer was registered.
 *
 * All streams in the current LWCA context are synchronized with the
 * current GL context.
 *
 * \param dptr   - Returned mapped base pointer
 * \param size   - Returned size of mapping
 * \param buffer - The name of the buffer object to map
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_MAP_FAILED
 * \notefnerr
 *
 * \sa ::lwGraphicsMapResources
 */
__LWDA_DEPRECATED LWresult LWDAAPI lwGLMapBufferObject(LWdeviceptr *dptr, size_t *size,  GLuint buffer);  

/**
 * \brief Unmaps an OpenGL buffer object
 *
 * \deprecated This function is deprecated as of Lwca 3.0. 
 *
 * Unmaps the buffer object specified by \p buffer for access by LWCA.
 *
 * There must be a valid OpenGL context bound to the current thread
 * when this function is called.  This must be the same context, or a
 * member of the same shareGroup, as the context that was bound when
 * the buffer was registered.
 *
 * All streams in the current LWCA context are synchronized with the
 * current GL context.
 *
 * \param buffer - Buffer object to unmap
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwGraphicsUnmapResources
 */
__LWDA_DEPRECATED LWresult LWDAAPI lwGLUnmapBufferObject(GLuint buffer);

/**
 * \brief Unregister an OpenGL buffer object
 *
 * \deprecated This function is deprecated as of Lwca 3.0. 
 *
 * Unregisters the buffer object specified by \p buffer.  This
 * releases any resources associated with the registered buffer.
 * After this call, the buffer may no longer be mapped for access by
 * LWCA.
 *
 * There must be a valid OpenGL context bound to the current thread
 * when this function is called.  This must be the same context, or a
 * member of the same shareGroup, as the context that was bound when
 * the buffer was registered.
 *
 * \param buffer - Name of the buffer object to unregister
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwGraphicsUnregisterResource
 */
__LWDA_DEPRECATED LWresult LWDAAPI lwGLUnregisterBufferObject(GLuint buffer);

/**
 * \brief Set the map flags for an OpenGL buffer object
 *
 * \deprecated This function is deprecated as of Lwca 3.0. 
 *
 * Sets the map flags for the buffer object specified by \p buffer.
 *
 * Changes to \p Flags will take effect the next time \p buffer is mapped.
 * The \p Flags argument may be any of the following:
 * - ::LW_GL_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by LWCA kernels. This is the default value.
 * - ::LW_GL_MAP_RESOURCE_FLAGS_READ_ONLY: Specifies that LWCA kernels which
 *   access this resource will not write to this resource.
 * - ::LW_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD: Specifies that LWCA kernels
 *   which access this resource will not read from this resource and will
 *   write over the entire contents of the resource, so none of the data
 *   previously stored in the resource will be preserved.
 *
 * If \p buffer has not been registered for use with LWCA, then
 * ::LWDA_ERROR_ILWALID_HANDLE is returned. If \p buffer is presently
 * mapped for access by LWCA, then ::LWDA_ERROR_ALREADY_MAPPED is returned.
 *
 * There must be a valid OpenGL context bound to the current thread
 * when this function is called.  This must be the same context, or a
 * member of the same shareGroup, as the context that was bound when
 * the buffer was registered.
 *
 * \param buffer - Buffer object to unmap
 * \param Flags  - Map flags
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ALREADY_MAPPED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * \notefnerr
 *
 * \sa ::lwGraphicsResourceSetMapFlags
 */
__LWDA_DEPRECATED LWresult LWDAAPI lwGLSetBufferObjectMapFlags(GLuint buffer, unsigned int Flags);

/**
 * \brief Maps an OpenGL buffer object
 *
 * \deprecated This function is deprecated as of Lwca 3.0. 
 *
 * Maps the buffer object specified by \p buffer into the address space of the
 * current LWCA context and returns in \p *dptr and \p *size the base pointer
 * and size of the resulting mapping.
 *
 * There must be a valid OpenGL context bound to the current thread
 * when this function is called.  This must be the same context, or a
 * member of the same shareGroup, as the context that was bound when
 * the buffer was registered.
 *
 * Stream \p hStream in the current LWCA context is synchronized with
 * the current GL context.
 *
 * \param dptr    - Returned mapped base pointer
 * \param size    - Returned size of mapping
 * \param buffer  - The name of the buffer object to map
 * \param hStream - Stream to synchronize
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_MAP_FAILED
 * \notefnerr
 *
 * \sa ::lwGraphicsMapResources
 */
__LWDA_DEPRECATED LWresult LWDAAPI lwGLMapBufferObjectAsync(LWdeviceptr *dptr, size_t *size,  GLuint buffer, LWstream hStream);

/**
 * \brief Unmaps an OpenGL buffer object
 *
 * \deprecated This function is deprecated as of Lwca 3.0. 
 *
 * Unmaps the buffer object specified by \p buffer for access by LWCA.
 *
 * There must be a valid OpenGL context bound to the current thread
 * when this function is called.  This must be the same context, or a
 * member of the same shareGroup, as the context that was bound when
 * the buffer was registered.
 *
 * Stream \p hStream in the current LWCA context is synchronized with
 * the current GL context.
 *
 * \param buffer  - Name of the buffer object to unmap
 * \param hStream - Stream to synchronize
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwGraphicsUnmapResources
 */
__LWDA_DEPRECATED LWresult LWDAAPI lwGLUnmapBufferObjectAsync(GLuint buffer, LWstream hStream);

/** @} */ /* END LWDA_GL_DEPRECATED */
/** @} */ /* END LWDA_GL */


#if defined(__LWDA_API_VERSION_INTERNAL)
    #undef lwGLCtxCreate
    #undef lwGLMapBufferObject
    #undef lwGLMapBufferObjectAsync
    #undef lwGLGetDevices

    LWresult LWDAAPI lwGLGetDevices(unsigned int *pLwdaDeviceCount, LWdevice *pLwdaDevices, unsigned int lwdaDeviceCount, LWGLDeviceList deviceList);
    LWresult LWDAAPI lwGLMapBufferObject_v2(LWdeviceptr *dptr, size_t *size,  GLuint buffer);
    LWresult LWDAAPI lwGLMapBufferObjectAsync_v2(LWdeviceptr *dptr, size_t *size,  GLuint buffer, LWstream hStream);
    LWresult LWDAAPI lwGLCtxCreate(LWcontext *pCtx, unsigned int Flags, LWdevice device );
    LWresult LWDAAPI lwGLMapBufferObject(LWdeviceptr_v1 *dptr, unsigned int *size,  GLuint buffer);
    LWresult LWDAAPI lwGLMapBufferObjectAsync(LWdeviceptr_v1 *dptr, unsigned int *size,  GLuint buffer, LWstream hStream);
#endif /* __LWDA_API_VERSION_INTERNAL */

#ifdef __cplusplus
};
#endif

#undef __LWDA_DEPRECATED

#endif
