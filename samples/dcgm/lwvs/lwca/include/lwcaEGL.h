/*
 * Copyright 2014 LWPU Corporation.  All rights reserved.
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

#ifndef LWDAEGL_H
#define LWDAEGL_H

/**
 * LWCA API versioning support
 */

#include "lwca.h"
#include "EGL/egl.h"
#include "EGL/eglext.h"


#if defined(LWDA_FORCE_API_VERSION)
    #if (LWDA_FORCE_API_VERSION == 3010)
        #define __LWDA_API_VERSION 3010
    #else
        #error "Unsupported value of LWDA_FORCE_API_VERSION"
    #endif
#else
    #define __LWDA_API_VERSION 6050
#endif /* LWDA_FORCE_API_VERSION */

#ifdef __cplusplus
extern "C" {
#endif

#if __LWDA_API_VERSION >= 6050

/**
  * \addtogroup LWDA_TYPES
  * @{
  */

/**
 * Maximum number of planes per frame
 */
#define MAX_PLANES 3

/**
  * LWCA EglFrame type - array or pointer
  */
typedef enum LWeglFrameType_enum {
    LW_EGL_FRAME_TYPE_ARRAY = 0,  /**< Frame type LWCA array */
    LW_EGL_FRAME_TYPE_PITCH = 1,  /**< Frame type pointer */
} LWeglFrameType;

/**
 * Resource location flags- sysmem or vidmem
 *
 * For LWCA context on iGPU, since video and system memory are equivalent -
 * these flags will not have an effect on the exelwtion.
 *
 * For LWCA context on dGPU, applications can use the flag ::LWeglResourceLocationFlags
 * to give a hint about the desired location.
 *
 * ::LW_EGL_RESOURCE_LOCATION_SYSMEM - the frame data is made resident on the system memory
 * to be accessed by LWCA.
 *
 * ::LW_EGL_RESOURCE_LOCATION_VIDMEM - the frame data is made resident on the dedicated
 * video memory to be accessed by LWCA.
 *
 * There may be an additional latency due to new allocation and data migration,
 * if the frame is produced on a different memory.

  */
typedef enum LWeglResourceLocationFlags_enum {
    LW_EGL_RESOURCE_LOCATION_SYSMEM   = 0x00,       /**< Resource location sysmem */
    LW_EGL_RESOURCE_LOCATION_VIDMEM   = 0x01        /**< Resource location vidmem */
} LWeglResourceLocationFlags;

/**
  * LWCA EGL Color Format - The different planar and multiplanar formats lwrrently supported for LWDA_EGL interops.
  */
typedef enum LWeglColorFormat_enum {
    LW_EGL_COLOR_FORMAT_YUV420_PLANAR       = 0x00,   /**< Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    LW_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR   = 0x01,   /**< Y, UV in two surfaces (UV as one surface), width, height ratio same as YUV420Planar. */
    LW_EGL_COLOR_FORMAT_YUV422_PLANAR       = 0x02,  /**< Y, U, V  each in a separate  surface, U/V width = 1/2 Y width, U/V height = Y height. */
    LW_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR   = 0x03,  /**< Y, UV in two surfaces, width, height ratio same as YUV422Planar. */
    LW_EGL_COLOR_FORMAT_RGB                 = 0x04,  /**< R/G/B three channels in one surface with RGB byte ordering. */
    LW_EGL_COLOR_FORMAT_BGR                 = 0x05,  /**< R/G/B three channels in one surface with BGR byte ordering. */
    LW_EGL_COLOR_FORMAT_ARGB                = 0x06,  /**< R/G/B/A four channels in one surface with ARGB byte ordering. */
    LW_EGL_COLOR_FORMAT_RGBA                = 0x07,  /**< R/G/B/A four channels in one surface with RGBA byte ordering. */
    LW_EGL_COLOR_FORMAT_L                   = 0x08,  /**< single luminance channel in one surface. */
    LW_EGL_COLOR_FORMAT_R                   = 0x09,  /**< single color channel in one surface. */
    LW_EGL_COLOR_FORMAT_YUV444_PLANAR       = 0xA,   /**< Y, U, V in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height. */
    LW_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR   = 0xB,   /**< Y, UV in two surfaces (UV as one surface), width, height ratio same as YUV444Planar. */
    LW_EGL_COLOR_FORMAT_YUYV_422            = 0xC,   /**< Y, U, V in one surface, interleaved as YUYV. */
    LW_EGL_COLOR_FORMAT_UYVY_422            = 0xD,   /**< Y, U, V in one surface, interleaved as UYVY. */
    LW_EGL_COLOR_FORMAT_MAX
} LWeglColorFormat;

/**
 * LWCA EGLFrame structure Descriptor - structure defining one frame of EGL.
 *
 * Each frame may contain one or more planes depending on whether the surface  * is Multiplanar or not.
 */
typedef struct LWeglFrame_st {
    union {
        LWarray pArray[MAX_PLANES];     /**< Array of LWarray corresponding to each plane*/
        void*   pPitch[MAX_PLANES];     /**< Array of Pointers corresponding to each plane*/
    } frame;
    unsigned int width;                 /**< Width of first plane */
    unsigned int height;                /**< Height of first plane */
    unsigned int depth;                 /**< Depth of first plane */
    unsigned int pitch;                 /**< Pitch of first plane */
    unsigned int planeCount;            /**< Number of planes */
    unsigned int numChannels;           /**< Number of channels for the plane */
    LWeglFrameType frameType;           /**< Array or Pitch */
    LWeglColorFormat eglColorFormat;    /**< LWCA EGL Color Format*/
    LWarray_format lwFormat;            /**< LWCA Array Format*/
} LWeglFrame;

/**
  * LWCA EGLSream Connection
  */
typedef struct LWeglStreamConnection_st* LWeglStreamConnection;

/** @} */ /* END LWDA_TYPES */

/**
 * \file lwdaEGL.h
 * \brief Header file for the EGL interoperability functions of the
 * low-level LWCA driver application programming interface.
 */

/**
 * \defgroup LWDA_EGL EGL Interoperability
 * \ingroup LWDA_DRIVER
 *
 * ___MANBRIEF___ EGL interoperability functions of the low-level LWCA
 * driver API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the EGL interoperability functions of the
 * low-level LWCA driver application programming interface.
 *
 * @{
 */

/**
 * \brief Registers an EGL image
 *
 * Registers the EGLImageKHR specified by \p image for access by
 * LWCA. A handle to the registered object is returned as \p pLwdaResource.
 * Additional Mapping/Unmapping is not required for the registered resource and
 * ::lwGraphicsResourceGetMappedEglFrame can be directly called on the \p pLwdaResource.
 *
 * The application will be responsible for synchronizing access to shared objects.
 * The application must ensure that any pending operation which access the objects have completed
 * before passing control to LWCA. This may be accomplished by issuing and waiting for
 * glFinish command on all GLcontexts (for OpenGL and likewise for other APIs).
 * The application will be also responsible for ensuring that any pending operation on the
 * registered LWCA resource has completed prior to exelwting subsequent commands in other APIs
 * accesing the same memory objects.
 * This can be accomplished by calling lwCtxSynchronize or lwEventSynchronize (preferably).
 *
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
 * The EGLImageKHR is an object which can be used to create EGLImage target resource. It is defined as a void pointer.
 * typedef void* EGLImageKHR
 *
 * \param pLwdaResource   - Pointer to the returned object handle
 * \param image           - An EGLImageKHR image which can be used to create target resource.
 * \param flags           - Map flags
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ALREADY_MAPPED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 *
 * \sa ::lwGraphicsEGLRegisterImage, ::lwGraphicsUnregisterResource,
 * ::lwGraphicsResourceSetMapFlags, ::lwGraphicsMapResources,
 * ::lwGraphicsUnmapResources,
 * ::lwdaGraphicsEGLRegisterImage
 */
LWresult LWDAAPI lwGraphicsEGLRegisterImage(LWgraphicsResource *pLwdaResource, EGLImageKHR image, unsigned int flags);

/**
 * \brief Connect LWCA to EGLStream as a consumer.
 *
 * Connect LWCA as a consumer to EGLStreamKHR specified by \p stream.
 *
 * The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one
 * API to another.
 *
 * \param conn            - Pointer to the returned connection handle
 * \param stream          - EGLStreamKHR handle
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 *
 * \sa ::lwEGLStreamConsumerConnect, ::lwEGLStreamConsumerDisconnect,
 * ::lwEGLStreamConsumerAcquireFrame, ::lwEGLStreamConsumerReleaseFrame,
 * ::lwdaEGLStreamConsumerConnect
 */
LWresult LWDAAPI lwEGLStreamConsumerConnect(LWeglStreamConnection *conn, EGLStreamKHR stream);

/**
 * \brief Connect LWCA to EGLStream as a consumer with given flags.
 *
 * Connect LWCA as a consumer to EGLStreamKHR specified by \p stream with specified \p flags defined by LWeglResourceLocationFlags.
 *
 * The flags specify whether the consumer wants to access frames from system memory or video memory.
 * Default is ::LW_EGL_RESOURCE_LOCATION_VIDMEM.
 *
 * \param conn              - Pointer to the returned connection handle
 * \param stream            - EGLStreamKHR handle
 * \param flags             - Flags denote intended location - system or video.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 *
 * \sa ::lwEGLStreamConsumerConnect, ::lwEGLStreamConsumerDisconnect,
 * ::lwEGLStreamConsumerAcquireFrame, ::lwEGLStreamConsumerReleaseFrame,
 * ::lwdaEGLStreamConsumerConnectWithFlags
 */

LWresult LWDAAPI lwEGLStreamConsumerConnectWithFlags(LWeglStreamConnection *conn, EGLStreamKHR stream, unsigned int flags);

/**
 * \brief Disconnect LWCA as a consumer to EGLStream .
 *
 * Disconnect LWCA as a consumer to EGLStreamKHR.
 *
 * The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one
 * API to another.

 * \param conn            - Conection to disconnect.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 *
 * \sa ::lwEGLStreamConsumerConnect, ::lwEGLStreamConsumerDisconnect,
 * ::lwEGLStreamConsumerAcquireFrame, ::lwEGLStreamConsumerReleaseFrame,
 * ::lwdaEGLStreamConsumerDisconnect
 */
LWresult LWDAAPI lwEGLStreamConsumerDisconnect(LWeglStreamConnection *conn);

/**
 * \brief Acquire an image frame from the EGLStream with LWCA as a consumer.
 *
 * Acquire an image frame from EGLStreamKHR.
 * ::lwGraphicsResourceGetMappedEglFrame can be called on \p pLwdaResource to get
 * ::LWeglFrame.
 *
 * \param conn            - Connection on which to acquire
 * \param pLwdaResource   - LWCA resource on which the stream frame will be mapped for use.
 * \param pStream         - LWCA stream for synchronization and any data migrations
 *                          implied by ::LWeglResourceLocationFlags.
 * \param timeout         - Desired timeout in usec.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 *
 * \sa ::lwEGLStreamConsumerConnect, ::lwEGLStreamConsumerDisconnect,
 * ::lwEGLStreamConsumerAcquireFrame, ::lwEGLStreamConsumerReleaseFrame,
 * ::lwdaEGLStreamConsumerAcquireFrame
 */
LWresult LWDAAPI lwEGLStreamConsumerAcquireFrame(LWeglStreamConnection *conn,
                                                  LWgraphicsResource *pLwdaResource, LWstream *pStream, unsigned int timeout);
/**
 * \brief Releases the last frame acquired from the EGLStream.
 *
 * Release the acquired image frame specified by \p pLwdaResource to EGLStreamKHR.
 *
 * The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one
 * API to another.
 *
 * \param conn            - Connection on which to release
 * \param pLwdaResource   - LWCA resource whose corresponding frame is to be released
 * \param pStream         - LWCA stream on which release will be done.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 *
 * \sa ::lwEGLStreamConsumerConnect, ::lwEGLStreamConsumerDisconnect,
 * ::lwEGLStreamConsumerAcquireFrame, ::lwEGLStreamConsumerReleaseFrame,
 * ::lwdaEGLStreamConsumerReleaseFrame
 */
LWresult LWDAAPI lwEGLStreamConsumerReleaseFrame(LWeglStreamConnection *conn,
                                                  LWgraphicsResource pLwdaResource, LWstream *pStream);

/**
 * \brief Connect LWCA to EGLStream as a producer.
 *
 * Connect LWCA as a producer to EGLStreamKHR specified by \p stream.
 *
 * The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one
 * API to another.
 *
 * \param conn   - Pointer to the returned connection handle
 * \param stream - EGLStreamKHR handle
 * \param width  - width of the image to be submitted to the stream
 * \param height - height of the image to be submitted to the stream
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 *
 * \sa ::lwEGLStreamProducerConnect, ::lwEGLStreamProducerDisconnect,
 * ::lwEGLStreamProducerPresentFrame,
 * ::lwdaEGLStreamProducerConnect
 */
LWresult LWDAAPI lwEGLStreamProducerConnect(LWeglStreamConnection *conn, EGLStreamKHR stream,
                                             EGLint width, EGLint height);

/**
 * \brief Disconnect LWCA as a producer  to EGLStream .
 *
 * Disconnect LWCA as a producer to EGLStreamKHR.
 *
 * The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one
 * API to another.

 * \param conn            - Conection to disconnect.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 *
 * \sa ::lwEGLStreamProducerConnect, ::lwEGLStreamProducerDisconnect,
 * ::lwEGLStreamProducerPresentFrame,
 * ::lwdaEGLStreamProducerDisconnect
 */
LWresult LWDAAPI lwEGLStreamProducerDisconnect(LWeglStreamConnection *conn);

/**
 * \brief Present a LWCA eglFrame to the EGLStream with LWCA as a producer.
 *
 * The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one
 * API to another.

 * The ::LWeglFrame is defined as:
 * \code
 * typedef struct LWeglFrame_st {
 *     union {
 *         LWarray pArray[MAX_PLANES];
 *         void*   pPitch[MAX_PLANES];
 *     } frame;
 *     unsigned int width;
 *     unsigned int height;
 *     unsigned int depth;
 *     unsigned int pitch;
 *     unsigned int planeCount;
 *     unsigned int numChannels;
 *     LWeglFrameType frameType;
 *     LWeglColorFormat eglColorFormat;
 *     LWarray_format lwFormat;
 * } LWeglFrame;
 * \endcode
 *
 * \param conn            - Connection on which to present the LWCA array
 * \param eglframe        - LWCA Eglstream Proucer Frame handle to be sent to the consumer over EglStream.
 * \param pStream         - LWCA stream on which to present the frame.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 *
 * \sa ::lwEGLStreamProducerConnect, ::lwEGLStreamProducerDisconnect,
 * ::lwEGLStreamProducerReturnFrame,
 * ::lwdaEGLStreamProducerPresentFrame
 */
LWresult LWDAAPI lwEGLStreamProducerPresentFrame(LWeglStreamConnection *conn,
                                                 LWeglFrame eglframe, LWstream *pStream);

/**
 * \brief Return the LWCA eglFrame to the EGLStream released by the consumer.
 *
 * The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one
 * API to another.
 *
 * This API can potentially return LWDA_ERROR_LAUNCH_TIMEOUT if the consumer has not 
 * returned a frame to EGL stream. If timeout is returned the application can retry.
 *
 *
 * \param conn            - Connection on which to return
 * \param eglframe        - LWCA Eglstream Proucer Frame handle returned from the consumer over EglStream.
 * \param pStream         - LWCA stream on which to return the frame.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_LAUNCH_TIMEOUT
 *
 * \sa ::lwEGLStreamProducerConnect, ::lwEGLStreamProducerDisconnect,
 * ::lwEGLStreamProducerPresentFrame,
 * ::lwdaEGLStreamProducerReturnFrame
 */
LWresult LWDAAPI lwEGLStreamProducerReturnFrame(LWeglStreamConnection *conn,
                                                LWeglFrame *eglframe, LWstream *pStream);

/**
 * \brief Get an eglFrame through which to access a registered EGL graphics resource.
 *
 * Returns in \p *eglFrame an eglFrame pointer through which the registered graphics resource
 * \p resource may be accessed.
 * This API can only be called for EGL graphics resources.
 *
 * The ::LWeglFrame is defined as:
 * \code
 * typedef struct LWeglFrame_st {
 *     union {
 *         LWarray pArray[MAX_PLANES];
 *         void*   pPitch[MAX_PLANES];
 *     } frame;
 *     unsigned int width;
 *     unsigned int height;
 *     unsigned int depth;
 *     unsigned int pitch;
 *     unsigned int planeCount;
 *     unsigned int numChannels;
 *     LWeglFrameType frameType;
 *     LWeglColorFormat eglColorFormat;
 *     LWarray_format lwFormat;
 * } LWeglFrame;
 * \endcode
 *
 * If \p resource is not registered then ::LWDA_ERROR_NOT_MAPPED is returned.
 * *
 * \param eglFrame   - Returned eglFrame.
 * \param resource   - Registered resource to access.
 * \param index      - Index for lwbemap surfaces.
 * \param mipLevel   - Mipmap level for the subresource to access.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_NOT_MAPPED
 *
 * \sa
 * ::lwGraphicsMapResources,
 * ::lwGraphicsSubResourceGetMappedArray,
 * ::lwGraphicsResourceGetMappedPointer,
 * ::lwdaGraphicsResourceGetMappedEglFrame
 */
LWresult LWDAAPI lwGraphicsResourceGetMappedEglFrame(LWeglFrame* eglFrame, LWgraphicsResource resource, unsigned int index, unsigned int mipLevel);

/**
 * \brief Creates an event from EGLSync object
 *
 * Creates an event *phEvent from an EGLSyncKHR eglSync with the flages specified
 * via \p flags. Valid flags include:
 * - ::LW_EVENT_DEFAULT: Default event creation flag.
 * - ::LW_EVENT_BLOCKING_SYNC: Specifies that the created event should use blocking
 * synchronization.  A CPU thread that uses ::lwEventSynchronize() to wait on
 * an event created with this flag will block until the event has actually
 * been completed.
 *
 * ::lwEventRecord and TimingData are not supported for events created from EGLSync.
 *
 * The EGLSyncKHR is an opaque handle to an EGL sync object.
 * typedef void* EGLSyncKHR
 *
 * \param phEvent - Returns newly created event
 * \param eglSync - Opaque handle to EGLSync object
 * \param flags   - Event creation flags
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_OUT_OF_MEMORY
 *
 * \sa
 * ::lwEventQuery,
 * ::lwEventSynchronize,
 * ::lwEventDestroy
 */
LWresult LWDAAPI lwEventCreateFromEGLSync(LWevent *phEvent, EGLSyncKHR eglSync, unsigned int flags);

#endif

/** @} */ /* END LWDA_EGL */

#ifdef __cplusplus
};
#endif

#undef __LWDA_API_VERSION

#endif

