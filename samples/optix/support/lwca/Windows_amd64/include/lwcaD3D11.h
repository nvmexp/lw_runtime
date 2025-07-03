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

#ifndef LWDAD3D11_H
#define LWDAD3D11_H

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

#define lwD3D11CtxCreate lwD3D11CtxCreate_v2

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup LWDA_D3D11 Direct3D 11 Interoperability
 * \ingroup LWDA_DRIVER
 *
 * ___MANBRIEF___ Direct3D 11 interoperability functions of the low-level LWCA
 * driver API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the Direct3D 11 interoperability functions of the
 * low-level LWCA driver application programming interface. Note that mapping 
 * of Direct3D 11 resources is performed with the graphics API agnostic, resource 
 * mapping interface described in \ref LWDA_GRAPHICS "Graphics Interoperability".
 *
 * @{
 */

/**
 * LWCA devices corresponding to a D3D11 device
 */
typedef enum LWd3d11DeviceList_enum {
    LW_D3D11_DEVICE_LIST_ALL            = 0x01, /**< The LWCA devices for all GPUs used by a D3D11 device */
    LW_D3D11_DEVICE_LIST_LWRRENT_FRAME  = 0x02, /**< The LWCA devices for the GPUs used by a D3D11 device in its lwrrently rendering frame */
    LW_D3D11_DEVICE_LIST_NEXT_FRAME     = 0x03, /**< The LWCA devices for the GPUs to be used by a D3D11 device in the next frame */
} LWd3d11DeviceList;

/**
 * \brief Gets the LWCA device corresponding to a display adapter.
 *
 * Returns in \p *pLwdaDevice the LWCA-compatible device corresponding to the
 * adapter \p pAdapter obtained from ::IDXGIFactory::EnumAdapters.
 *
 * If no device on \p pAdapter is LWCA-compatible the call will return
 * ::LWDA_ERROR_NO_DEVICE.
 *
 * \param pLwdaDevice - Returned LWCA device corresponding to \p pAdapter
 * \param pAdapter    - Adapter to query for LWCA device
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_NO_DEVICE,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_NOT_FOUND,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::lwD3D11GetDevices,
 * ::lwdaD3D11GetDevice
 */
LWresult LWDAAPI lwD3D11GetDevice(LWdevice *pLwdaDevice, IDXGIAdapter *pAdapter);

/**
 * \brief Gets the LWCA devices corresponding to a Direct3D 11 device
 *
 * Returns in \p *pLwdaDeviceCount the number of LWCA-compatible device corresponding
 * to the Direct3D 11 device \p pD3D11Device.
 * Also returns in \p *pLwdaDevices at most \p lwdaDeviceCount of the LWCA-compatible devices
 * corresponding to the Direct3D 11 device \p pD3D11Device.
 *
 * If any of the GPUs being used to render \p pDevice are not LWCA capable then the
 * call will return ::LWDA_ERROR_NO_DEVICE.
 *
 * \param pLwdaDeviceCount - Returned number of LWCA devices corresponding to \p pD3D11Device
 * \param pLwdaDevices     - Returned LWCA devices corresponding to \p pD3D11Device
 * \param lwdaDeviceCount  - The size of the output device array \p pLwdaDevices
 * \param pD3D11Device     - Direct3D 11 device to query for LWCA devices
 * \param deviceList       - The set of devices to return.  This set may be
 *                           ::LW_D3D11_DEVICE_LIST_ALL for all devices,
 *                           ::LW_D3D11_DEVICE_LIST_LWRRENT_FRAME for the devices used to
 *                           render the current frame (in SLI), or
 *                           ::LW_D3D11_DEVICE_LIST_NEXT_FRAME for the devices used to
 *                           render the next frame (in SLI).
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_NO_DEVICE,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_NOT_FOUND,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::lwD3D11GetDevice,
 * ::lwdaD3D11GetDevices
 */
LWresult LWDAAPI lwD3D11GetDevices(unsigned int *pLwdaDeviceCount, LWdevice *pLwdaDevices, unsigned int lwdaDeviceCount, ID3D11Device *pD3D11Device, LWd3d11DeviceList deviceList);

/**
 * \brief Register a Direct3D 11 resource for access by LWCA
 *
 * Registers the Direct3D 11 resource \p pD3DResource for access by LWCA and
 * returns a LWCA handle to \p pD3Dresource in \p pLwdaResource.
 * The handle returned in \p pLwdaResource may be used to map and unmap this
 * resource until it is unregistered.
 * On success this call will increase the internal reference count on
 * \p pD3DResource. This reference count will be decremented when this
 * resource is unregistered through ::lwGraphicsUnregisterResource().
 *
 * This call is potentially high-overhead and should not be called every frame
 * in interactive applications.
 *
 * The type of \p pD3DResource must be one of the following.
 * - ::ID3D11Buffer: may be accessed through a device pointer.
 * - ::ID3D11Texture1D: individual subresources of the texture may be accessed via arrays
 * - ::ID3D11Texture2D: individual subresources of the texture may be accessed via arrays
 * - ::ID3D11Texture3D: individual subresources of the texture may be accessed via arrays
 *
 * The \p Flags argument may be used to specify additional parameters at register
 * time.  The valid values for this parameter are
 * - ::LW_GRAPHICS_REGISTER_FLAGS_NONE: Specifies no hints about how this
 *   resource will be used.
 * - ::LW_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST: Specifies that LWCA will
 *   bind this resource to a surface reference.
 * - ::LW_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER: Specifies that LWCA will perform
 *   texture gather operations on this resource.
 *
 * Not all Direct3D resources of the above types may be used for
 * interoperability with LWCA.  The following are some limitations.
 * - The primary rendertarget may not be registered with LWCA.
 * - Textures which are not of a format which is 1, 2, or 4 channels of 8, 16,
 *   or 32-bit integer or floating-point data cannot be shared.
 * - Surfaces of depth or stencil formats cannot be shared.
 *
 * A complete list of supported DXGI formats is as follows. For compactness the
 * notation A_{B,C,D} represents A_B, A_C, and A_D.
 * - DXGI_FORMAT_A8_UNORM
 * - DXGI_FORMAT_B8G8R8A8_UNORM
 * - DXGI_FORMAT_B8G8R8X8_UNORM
 * - DXGI_FORMAT_R16_FLOAT
 * - DXGI_FORMAT_R16G16B16A16_{FLOAT,SINT,SNORM,UINT,UNORM}
 * - DXGI_FORMAT_R16G16_{FLOAT,SINT,SNORM,UINT,UNORM}
 * - DXGI_FORMAT_R16_{SINT,SNORM,UINT,UNORM}
 * - DXGI_FORMAT_R32_FLOAT
 * - DXGI_FORMAT_R32G32B32A32_{FLOAT,SINT,UINT}
 * - DXGI_FORMAT_R32G32_{FLOAT,SINT,UINT}
 * - DXGI_FORMAT_R32_{SINT,UINT}
 * - DXGI_FORMAT_R8G8B8A8_{SINT,SNORM,UINT,UNORM,UNORM_SRGB}
 * - DXGI_FORMAT_R8G8_{SINT,SNORM,UINT,UNORM}
 * - DXGI_FORMAT_R8_{SINT,SNORM,UINT,UNORM}
 *
 * If \p pD3DResource is of incorrect type or is already registered then
 * ::LWDA_ERROR_ILWALID_HANDLE is returned.
 * If \p pD3DResource cannot be registered then ::LWDA_ERROR_UNKNOWN is returned.
 * If \p Flags is not one of the above specified value then ::LWDA_ERROR_ILWALID_VALUE
 * is returned.
 *
 * \param pLwdaResource - Returned graphics resource handle
 * \param pD3DResource  - Direct3D resource to register
 * \param Flags         - Parameters for resource registration
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::lwGraphicsUnregisterResource,
 * ::lwGraphicsMapResources,
 * ::lwGraphicsSubResourceGetMappedArray,
 * ::lwGraphicsResourceGetMappedPointer,
 * ::lwdaGraphicsD3D11RegisterResource
 */
LWresult LWDAAPI lwGraphicsD3D11RegisterResource(LWgraphicsResource *pLwdaResource, ID3D11Resource *pD3DResource, unsigned int Flags);

/**
 * \defgroup LWDA_D3D11_DEPRECATED Direct3D 11 Interoperability [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated Direct3D 11 interoperability functions of the
 * low-level LWCA driver API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes deprecated Direct3D 11 interoperability functionality.
 * @{
 */

/**
 * \brief Create a LWCA context for interoperability with Direct3D 11
 *
 * \deprecated This function is deprecated as of LWCA 5.0.
 *
 * This function is deprecated and should no longer be used.  It is
 * no longer necessary to associate a LWCA context with a D3D11
 * device in order to achieve maximum interoperability performance.
 *
 * \param pCtx        - Returned newly created LWCA context
 * \param pLwdaDevice - Returned pointer to the device on which the context was created
 * \param Flags       - Context creation flags (see ::lwCtxCreate() for details)
 * \param pD3DDevice  - Direct3D device to create interoperability context with
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::lwD3D11GetDevice,
 * ::lwGraphicsD3D11RegisterResource
 */
__LWDA_DEPRECATED LWresult LWDAAPI lwD3D11CtxCreate(LWcontext *pCtx, LWdevice *pLwdaDevice, unsigned int Flags, ID3D11Device *pD3DDevice);

/**
 * \brief Create a LWCA context for interoperability with Direct3D 11
 *
 * \deprecated This function is deprecated as of LWCA 5.0.
 *
 * This function is deprecated and should no longer be used.  It is
 * no longer necessary to associate a LWCA context with a D3D11
 * device in order to achieve maximum interoperability performance.
 *
 * \param pCtx        - Returned newly created LWCA context
 * \param flags       - Context creation flags (see ::lwCtxCreate() for details)
 * \param pD3DDevice  - Direct3D device to create interoperability context with
 * \param lwdaDevice  - The LWCA device on which to create the context.  This device
 *                      must be among the devices returned when querying
 *                      ::LW_D3D11_DEVICES_ALL from  ::lwD3D11GetDevices.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::lwD3D11GetDevices,
 * ::lwGraphicsD3D11RegisterResource
 */
__LWDA_DEPRECATED LWresult LWDAAPI lwD3D11CtxCreateOnDevice(LWcontext *pCtx, unsigned int flags, ID3D11Device *pD3DDevice, LWdevice lwdaDevice);

/**
 * \brief Get the Direct3D 11 device against which the current LWCA context was
 * created
 *
 * \deprecated This function is deprecated as of LWCA 5.0.
 *
 * This function is deprecated and should no longer be used.  It is
 * no longer necessary to associate a LWCA context with a D3D11
 * device in order to achieve maximum interoperability performance.
 *
 * \param ppD3DDevice - Returned Direct3D device corresponding to LWCA context
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT
 * \notefnerr
 *
 * \sa
 * ::lwD3D11GetDevice
 */
__LWDA_DEPRECATED LWresult LWDAAPI lwD3D11GetDirect3DDevice(ID3D11Device **ppD3DDevice);

/** @} */ /* END LWDA_D3D11_DEPRECATED */
/** @} */ /* END LWDA_D3D11 */


#if defined(__LWDA_API_VERSION_INTERNAL)
    #undef lwD3D11CtxCreate

    LWresult LWDAAPI lwD3D11CtxCreate(LWcontext *pCtx, LWdevice *pLwdaDevice, unsigned int Flags, ID3D11Device *pD3DDevice);
#endif /* __LWDA_API_VERSION_INTERNAL */

#ifdef __cplusplus
};
#endif

#undef __LWDA_DEPRECATED

#endif

