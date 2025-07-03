/*
 * Copyright 2010-2014 LWPU Corporation.  All rights reserved.
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

#ifndef LWDAVDPAU_H
#define LWDAVDPAU_H

/** 
 * LWCA API versioning support
 */
#if defined(LWDA_FORCE_API_VERSION)
    #if (LWDA_FORCE_API_VERSION == 3010)
        #define __LWDA_API_VERSION 3010
    #else
        #error "Unsupported value of LWDA_FORCE_API_VERSION"
    #endif
#else
    #define __LWDA_API_VERSION 3020
#endif /* LWDA_FORCE_API_VERSION */

#if defined(__LWDA_API_VERSION_INTERNAL) || __LWDA_API_VERSION >= 3020
    #define lwVDPAUCtxCreate lwVDPAUCtxCreate_v2
#endif /* __LWDA_API_VERSION_INTERNAL || __LWDA_API_VERSION >= 3020 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup LWDA_VDPAU VDPAU Interoperability
 * \ingroup LWDA_DRIVER
 *
 * ___MANBRIEF___ VDPAU interoperability functions of the low-level LWCA driver
 * API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the VDPAU interoperability functions of the
 * low-level LWCA driver application programming interface.
 *
 * @{
 */

/**
 * \brief Gets the LWCA device associated with a VDPAU device
 *
 * Returns in \p *pDevice the LWCA device associated with a \p vdpDevice, if
 * applicable.
 *
 * \param pDevice           - Device associated with vdpDevice
 * \param vdpDevice         - A VdpDevice handle
 * \param vdpGetProcAddress - VDPAU's VdpGetProcAddress function pointer
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwCtxCreate, ::lwVDPAUCtxCreate, ::lwGraphicsVDPAURegisterVideoSurface,
 * ::lwGraphicsVDPAURegisterOutputSurface, ::lwGraphicsUnregisterResource,
 * ::lwGraphicsResourceSetMapFlags, ::lwGraphicsMapResources,
 * ::lwGraphicsUnmapResources, ::lwGraphicsSubResourceGetMappedArray,
 * ::lwdaVDPAUGetDevice
 */
LWresult LWDAAPI lwVDPAUGetDevice(LWdevice *pDevice, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress);

#if __LWDA_API_VERSION >= 3020
/**
 * \brief Create a LWCA context for interoperability with VDPAU
 *
 * Creates a new LWCA context, initializes VDPAU interoperability, and
 * associates the LWCA context with the calling thread. It must be called
 * before performing any other VDPAU interoperability operations. It may fail
 * if the needed VDPAU driver facilities are not available. For usage of the
 * \p flags parameter, see ::lwCtxCreate().
 *
 * \param pCtx              - Returned LWCA context
 * \param flags             - Options for LWCA context creation
 * \param device            - Device on which to create the context
 * \param vdpDevice         - The VdpDevice to interop with
 * \param vdpGetProcAddress - VDPAU's VdpGetProcAddress function pointer
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
 * \sa ::lwCtxCreate, ::lwGraphicsVDPAURegisterVideoSurface,
 * ::lwGraphicsVDPAURegisterOutputSurface, ::lwGraphicsUnregisterResource,
 * ::lwGraphicsResourceSetMapFlags, ::lwGraphicsMapResources,
 * ::lwGraphicsUnmapResources, ::lwGraphicsSubResourceGetMappedArray,
 * ::lwVDPAUGetDevice
 */
LWresult LWDAAPI lwVDPAUCtxCreate(LWcontext *pCtx, unsigned int flags, LWdevice device, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress);
#endif /* __LWDA_API_VERSION >= 3020 */

/**
 * \brief Registers a VDPAU VdpVideoSurface object
 *
 * Registers the VdpVideoSurface specified by \p vdpSurface for access by
 * LWCA. A handle to the registered object is returned as \p pLwdaResource.
 * The surface's intended usage is specified using \p flags, as follows:
 *
 * - ::LW_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by LWCA. This is the default value.
 * - ::LW_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY: Specifies that LWCA
 *   will not write to this resource.
 * - ::LW_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD: Specifies that
 *   LWCA will not read from this resource and will write over the
 *   entire contents of the resource, so none of the data previously
 *   stored in the resource will be preserved.
 *
 * The VdpVideoSurface is presented as an array of subresources that may be
 * accessed using pointers returned by ::lwGraphicsSubResourceGetMappedArray.
 * The exact number of valid \p arrayIndex values depends on the VDPAU surface
 * format. The mapping is shown in the table below. \p mipLevel must be 0.
 *
 * \htmlonly
 * <table>
 * <tr><th>VdpChromaType                               </th><th>arrayIndex</th><th>Size     </th><th>Format</th><th>Content            </th></tr>
 * <tr><td rowspan="4" valign="top">VDP_CHROMA_TYPE_420</td><td>0         </td><td>w   x h/2</td><td>R8    </td><td>Top-field luma     </td></tr>
 * <tr>                                                     <td>1         </td><td>w   x h/2</td><td>R8    </td><td>Bottom-field luma  </td></tr>
 * <tr>                                                     <td>2         </td><td>w/2 x h/4</td><td>R8G8  </td><td>Top-field chroma   </td></tr>
 * <tr>                                                     <td>3         </td><td>w/2 x h/4</td><td>R8G8  </td><td>Bottom-field chroma</td></tr>
 * <tr><td rowspan="4" valign="top">VDP_CHROMA_TYPE_422</td><td>0         </td><td>w   x h/2</td><td>R8    </td><td>Top-field luma     </td></tr>
 * <tr>                                                     <td>1         </td><td>w   x h/2</td><td>R8    </td><td>Bottom-field luma  </td></tr>
 * <tr>                                                     <td>2         </td><td>w/2 x h/2</td><td>R8G8  </td><td>Top-field chroma   </td></tr>
 * <tr>                                                     <td>3         </td><td>w/2 x h/2</td><td>R8G8  </td><td>Bottom-field chroma</td></tr>
 * </table>
 * \endhtmlonly
 *
 * \latexonly
 * \begin{tabular}{|l|l|l|l|l|}
 * \hline
 * VdpChromaType          & arrayIndex & Size      & Format & Content             \\
 * \hline
 * VDP\_CHROMA\_TYPE\_420 & 0          & w x h/2   & R8     & Top-field luma      \\
 *                        & 1          & w x h/2   & R8     & Bottom-field luma   \\
 *                        & 2          & w/2 x h/4 & R8G8   & Top-field chroma    \\
 *                        & 3          & w/2 x h/4 & R8G8   & Bottom-field chroma \\
 * \hline
 * VDP\_CHROMA\_TYPE\_422 & 0          & w x h/2   & R8     & Top-field luma      \\
 *                        & 1          & w x h/2   & R8     & Bottom-field luma   \\
 *                        & 2          & w/2 x h/2 & R8G8   & Top-field chroma    \\
 *                        & 3          & w/2 x h/2 & R8G8   & Bottom-field chroma \\
 * \hline
 * \end{tabular}
 * \endlatexonly
 *
 * \param pLwdaResource - Pointer to the returned object handle
 * \param vdpSurface    - The VdpVideoSurface to be registered
 * \param flags         - Map flags
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ALREADY_MAPPED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * \notefnerr
 *
 * \sa ::lwCtxCreate, ::lwVDPAUCtxCreate,
 * ::lwGraphicsVDPAURegisterOutputSurface, ::lwGraphicsUnregisterResource,
 * ::lwGraphicsResourceSetMapFlags, ::lwGraphicsMapResources,
 * ::lwGraphicsUnmapResources, ::lwGraphicsSubResourceGetMappedArray,
 * ::lwVDPAUGetDevice,
 * ::lwdaGraphicsVDPAURegisterVideoSurface
 */
LWresult LWDAAPI lwGraphicsVDPAURegisterVideoSurface(LWgraphicsResource *pLwdaResource, VdpVideoSurface vdpSurface, unsigned int flags);

/**
 * \brief Registers a VDPAU VdpOutputSurface object
 *
 * Registers the VdpOutputSurface specified by \p vdpSurface for access by
 * LWCA. A handle to the registered object is returned as \p pLwdaResource.
 * The surface's intended usage is specified using \p flags, as follows:
 *
 * - ::LW_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by LWCA. This is the default value.
 * - ::LW_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY: Specifies that LWCA
 *   will not write to this resource.
 * - ::LW_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD: Specifies that
 *   LWCA will not read from this resource and will write over the
 *   entire contents of the resource, so none of the data previously
 *   stored in the resource will be preserved.
 *
 * The VdpOutputSurface is presented as an array of subresources that may be
 * accessed using pointers returned by ::lwGraphicsSubResourceGetMappedArray.
 * The exact number of valid \p arrayIndex values depends on the VDPAU surface
 * format. The mapping is shown in the table below. \p mipLevel must be 0.
 *
 * \htmlonly
 * <table>
 * <tr><th>VdpRGBAFormat              </th><th>arrayIndex</th><th>Size </th><th>Format </th><th>Content       </th></tr>
 * <tr><td>VDP_RGBA_FORMAT_B8G8R8A8   </td><td>0         </td><td>w x h</td><td>ARGB8  </td><td>Entire surface</td></tr>
 * <tr><td>VDP_RGBA_FORMAT_R10G10B10A2</td><td>0         </td><td>w x h</td><td>A2BGR10</td><td>Entire surface</td></tr>
 * </table>
 * \endhtmlonly
 *
 * \latexonly
 * \begin{tabular}{|l|l|l|l|l|}
 * \hline
 * VdpRGBAFormat                  & arrayIndex & Size  & Format  & Content        \\
 * \hline
 * VDP\_RGBA\_FORMAT\_B8G8R8A8    & 0          & w x h & ARGB8   & Entire surface \\
 * VDP\_RGBA\_FORMAT\_R10G10B10A2 & 0          & w x h & A2BGR10 & Entire surface \\
 * \hline
 * \end{tabular}
 * \endlatexonly
 *
 * \param pLwdaResource - Pointer to the returned object handle
 * \param vdpSurface    - The VdpOutputSurface to be registered
 * \param flags         - Map flags
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ALREADY_MAPPED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * \notefnerr
 *
 * \sa ::lwCtxCreate, ::lwVDPAUCtxCreate,
 * ::lwGraphicsVDPAURegisterVideoSurface, ::lwGraphicsUnregisterResource,
 * ::lwGraphicsResourceSetMapFlags, ::lwGraphicsMapResources,
 * ::lwGraphicsUnmapResources, ::lwGraphicsSubResourceGetMappedArray,
 * ::lwVDPAUGetDevice,
 * ::lwdaGraphicsVDPAURegisterOutputSurface
 */
LWresult LWDAAPI lwGraphicsVDPAURegisterOutputSurface(LWgraphicsResource *pLwdaResource, VdpOutputSurface vdpSurface, unsigned int flags);

/** @} */ /* END LWDA_VDPAU */

/** 
 * LWCA API versioning support
 */
#if defined(__LWDA_API_VERSION_INTERNAL)
    #undef lwVDPAUCtxCreate
#endif /* __LWDA_API_VERSION_INTERNAL */

/** 
 * LWCA API made obselete at API version 3020
 */
#if defined(__LWDA_API_VERSION_INTERNAL) || __LWDA_API_VERSION < 3020
LWresult LWDAAPI lwVDPAUCtxCreate(LWcontext *pCtx, unsigned int flags, LWdevice device, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress);
#endif /* __LWDA_API_VERSION_INTERNAL || __LWDA_API_VERSION < 3020 */ 

#ifdef __cplusplus
};
#endif

#undef __LWDA_API_VERSION

#endif

