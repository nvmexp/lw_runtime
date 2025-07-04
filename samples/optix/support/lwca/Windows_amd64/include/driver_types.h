/*
 * Copyright 1993-2018 LWPU Corporation.  All rights reserved.
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

#if !defined(__DRIVER_TYPES_H__)
#define __DRIVER_TYPES_H__

#if !defined(__LWDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __LWDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_LWDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DRIVER_TYPES_H__
#endif

#ifndef __DOXYGEN_ONLY__
#include "crt/host_defines.h"
#endif
#include "vector_types.h"

/**
 * \defgroup LWDART_TYPES Data types used by LWCA Runtime
 * \ingroup LWDART
 *
 * @{
 */

/*******************************************************************************
*                                                                              *
*  TYPE DEFINITIONS USED BY RUNTIME API                                        *
*                                                                              *
*******************************************************************************/

#if !defined(__LWDA_INTERNAL_COMPILATION__)

#if !defined(__LWDACC_RTC__)
#include <limits.h>
#include <stddef.h>
#endif /* !defined(__LWDACC_RTC__) */

#define lwdaHostAllocDefault                0x00  /**< Default page-locked allocation flag */
#define lwdaHostAllocPortable               0x01  /**< Pinned memory accessible by all LWCA contexts */
#define lwdaHostAllocMapped                 0x02  /**< Map allocation into device space */
#define lwdaHostAllocWriteCombined          0x04  /**< Write-combined memory */

#define lwdaHostRegisterDefault             0x00  /**< Default host memory registration flag */
#define lwdaHostRegisterPortable            0x01  /**< Pinned memory accessible by all LWCA contexts */
#define lwdaHostRegisterMapped              0x02  /**< Map registered memory into device space */
#define lwdaHostRegisterIoMemory            0x04  /**< Memory-mapped I/O space */
#define lwdaHostRegisterReadOnly            0x08  /**< Memory-mapped read-only */

#define lwdaPeerAccessDefault               0x00  /**< Default peer addressing enable flag */

#define lwdaStreamDefault                   0x00  /**< Default stream flag */
#define lwdaStreamNonBlocking               0x01  /**< Stream does not synchronize with stream 0 (the NULL stream) */

 /**
 * Legacy stream handle
 *
 * Stream handle that can be passed as a lwdaStream_t to use an implicit stream
 * with legacy synchronization behavior.
 *
 * See details of the \link_sync_behavior
 */
#define lwdaStreamLegacy                    ((lwdaStream_t)0x1)

/**
 * Per-thread stream handle
 *
 * Stream handle that can be passed as a lwdaStream_t to use an implicit stream
 * with per-thread synchronization behavior.
 *
 * See details of the \link_sync_behavior
 */
#define lwdaStreamPerThread                 ((lwdaStream_t)0x2)

#define lwdaEventDefault                    0x00  /**< Default event flag */
#define lwdaEventBlockingSync               0x01  /**< Event uses blocking synchronization */
#define lwdaEventDisableTiming              0x02  /**< Event will not record timing data */
#define lwdaEventInterprocess               0x04  /**< Event is suitable for interprocess use. lwdaEventDisableTiming must be set */

#define lwdaEventRecordDefault              0x00  /**< Default event record flag */
#define lwdaEventRecordExternal             0x01  /**< Event is captured in the graph as an external event node when performing stream capture */

#define lwdaEventWaitDefault                0x00  /**< Default event wait flag */
#define lwdaEventWaitExternal               0x01  /**< Event is captured in the graph as an external event node when performing stream capture */

#define lwdaDeviceScheduleAuto              0x00  /**< Device flag - Automatic scheduling */
#define lwdaDeviceScheduleSpin              0x01  /**< Device flag - Spin default scheduling */
#define lwdaDeviceScheduleYield             0x02  /**< Device flag - Yield default scheduling */
#define lwdaDeviceScheduleBlockingSync      0x04  /**< Device flag - Use blocking synchronization */
#define lwdaDeviceBlockingSync              0x04  /**< Device flag - Use blocking synchronization 
                                                    *  \deprecated This flag was deprecated as of LWCA 4.0 and
                                                    *  replaced with ::lwdaDeviceScheduleBlockingSync. */
#define lwdaDeviceScheduleMask              0x07  /**< Device schedule flags mask */
#define lwdaDeviceMapHost                   0x08  /**< Device flag - Support mapped pinned allocations */
#define lwdaDeviceLmemResizeToMax           0x10  /**< Device flag - Keep local memory allocation after launch */
#define lwdaDeviceMask                      0x1f  /**< Device flags mask */

#define lwdaArrayDefault                    0x00  /**< Default LWCA array allocation flag */
#define lwdaArrayLayered                    0x01  /**< Must be set in lwdaMalloc3DArray to create a layered LWCA array */
#define lwdaArraySurfaceLoadStore           0x02  /**< Must be set in lwdaMallocArray or lwdaMalloc3DArray in order to bind surfaces to the LWCA array */
#define lwdaArrayLwbemap                    0x04  /**< Must be set in lwdaMalloc3DArray to create a lwbemap LWCA array */
#define lwdaArrayTextureGather              0x08  /**< Must be set in lwdaMallocArray or lwdaMalloc3DArray in order to perform texture gather operations on the LWCA array */
#define lwdaArrayColorAttachment            0x20  /**< Must be set in lwdaExternalMemoryGetMappedMipmappedArray if the mipmapped array is used as a color target in a graphics API */
#define lwdaArraySparse                     0x40  /**< Must be set in lwdaMallocArray, lwdaMalloc3DArray or lwdaMallocMipmappedArray in order to create a sparse LWCA array or LWCA mipmapped array */

#define lwdaIpcMemLazyEnablePeerAccess      0x01  /**< Automatically enable peer access between remote devices as needed */

#define lwdaMemAttachGlobal                 0x01  /**< Memory can be accessed by any stream on any device*/
#define lwdaMemAttachHost                   0x02  /**< Memory cannot be accessed by any stream on any device */
#define lwdaMemAttachSingle                 0x04  /**< Memory can only be accessed by a single stream on the associated device */

#define lwdaOclwpancyDefault                0x00  /**< Default behavior */
#define lwdaOclwpancyDisableCachingOverride 0x01  /**< Assume global caching is enabled and cannot be automatically turned off */

#define lwdaCpuDeviceId                     ((int)-1) /**< Device id that represents the CPU */
#define lwdaIlwalidDeviceId                 ((int)-2) /**< Device id that represents an invalid device */

/**
 * If set, each kernel launched as part of ::lwdaLaunchCooperativeKernelMultiDevice only
 * waits for prior work in the stream corresponding to that GPU to complete before the
 * kernel begins exelwtion.
 */
#define lwdaCooperativeLaunchMultiDeviceNoPreSync  0x01

/**
 * If set, any subsequent work pushed in a stream that participated in a call to
 * ::lwdaLaunchCooperativeKernelMultiDevice will only wait for the kernel launched on
 * the GPU corresponding to that stream to complete before it begins exelwtion.
 */
#define lwdaCooperativeLaunchMultiDeviceNoPostSync 0x02

#endif /* !__LWDA_INTERNAL_COMPILATION__ */

/** \cond impl_private */
#if defined(__DOXYGEN_ONLY__) || defined(LWDA_ENABLE_DEPRECATED)
#define __LWDA_DEPRECATED
#elif defined(_MSC_VER)
#define __LWDA_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__)
#define __LWDA_DEPRECATED __attribute__((deprecated))
#else
#define __LWDA_DEPRECATED
#endif
/** \endcond impl_private */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/**
 * LWCA error types
 */
enum __device_builtin__ lwdaError
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * also means that the operation being queried is complete (see
     * ::lwdaEventQuery() and ::lwdaStreamQuery()).
     */
    lwdaSuccess                           =      0,
  
    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    lwdaErrorIlwalidValue                 =     1,
  
    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    lwdaErrorMemoryAllocation             =      2,
  
    /**
     * The API call failed because the LWCA driver and runtime could not be
     * initialized.
     */
    lwdaErrorInitializationError          =      3,
  
    /**
     * This indicates that a LWCA Runtime API call cannot be exelwted because
     * it is being called during process shut down, at a point in time after
     * LWCA driver has been unloaded.
     */
    lwdaErrorLwdartUnloading              =     4,

    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    lwdaErrorProfilerDisabled             =     5,

    /**
     * \deprecated
     * This error return is deprecated as of LWCA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::lwdaProfilerStart or
     * ::lwdaProfilerStop without initialization.
     */
    lwdaErrorProfilerNotInitialized       =     6,

    /**
     * \deprecated
     * This error return is deprecated as of LWCA 5.0. It is no longer an error
     * to call lwdaProfilerStart() when profiling is already enabled.
     */
    lwdaErrorProfilerAlreadyStarted       =     7,

    /**
     * \deprecated
     * This error return is deprecated as of LWCA 5.0. It is no longer an error
     * to call lwdaProfilerStop() when profiling is already disabled.
     */
     lwdaErrorProfilerAlreadyStopped       =    8,
  
    /**
     * This indicates that a kernel launch is requesting resources that can
     * never be satisfied by the current device. Requesting more shared memory
     * per block than the device supports will trigger this error, as will
     * requesting too many threads or blocks. See ::lwdaDeviceProp for more
     * device limitations.
     */
    lwdaErrorIlwalidConfiguration         =      9,
  
    /**
     * This indicates that one or more of the pitch-related parameters passed
     * to the API call is not within the acceptable range for pitch.
     */
    lwdaErrorIlwalidPitchValue            =     12,
  
    /**
     * This indicates that the symbol name/identifier passed to the API call
     * is not a valid name or identifier.
     */
    lwdaErrorIlwalidSymbol                =     13,
  
    /**
     * This indicates that at least one host pointer passed to the API call is
     * not a valid host pointer.
     * \deprecated
     * This error return is deprecated as of LWCA 10.1.
     */
    lwdaErrorIlwalidHostPointer           =     16,
  
    /**
     * This indicates that at least one device pointer passed to the API call is
     * not a valid device pointer.
     * \deprecated
     * This error return is deprecated as of LWCA 10.1.
     */
    lwdaErrorIlwalidDevicePointer         =     17,
  
    /**
     * This indicates that the texture passed to the API call is not a valid
     * texture.
     */
    lwdaErrorIlwalidTexture               =     18,
  
    /**
     * This indicates that the texture binding is not valid. This oclwrs if you
     * call ::lwdaGetTextureAlignmentOffset() with an unbound texture.
     */
    lwdaErrorIlwalidTextureBinding        =     19,
  
    /**
     * This indicates that the channel descriptor passed to the API call is not
     * valid. This oclwrs if the format is not one of the formats specified by
     * ::lwdaChannelFormatKind, or if one of the dimensions is invalid.
     */
    lwdaErrorIlwalidChannelDescriptor     =     20,
  
    /**
     * This indicates that the direction of the memcpy passed to the API call is
     * not one of the types specified by ::lwdaMemcpyKind.
     */
    lwdaErrorIlwalidMemcpyDirection       =     21,
  
    /**
     * This indicated that the user has taken the address of a constant variable,
     * which was forbidden up until the LWCA 3.1 release.
     * \deprecated
     * This error return is deprecated as of LWCA 3.1. Variables in constant
     * memory may now have their address taken by the runtime via
     * ::lwdaGetSymbolAddress().
     */
    lwdaErrorAddressOfConstant            =     22,
  
    /**
     * This indicated that a texture fetch was not able to be performed.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of LWCA 3.1. Device emulation mode was
     * removed with the LWCA 3.1 release.
     */
    lwdaErrorTextureFetchFailed           =     23,
  
    /**
     * This indicated that a texture was not bound for access.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of LWCA 3.1. Device emulation mode was
     * removed with the LWCA 3.1 release.
     */
    lwdaErrorTextureNotBound              =     24,
  
    /**
     * This indicated that a synchronization operation had failed.
     * This was previously used for some device emulation functions.
     * \deprecated
     * This error return is deprecated as of LWCA 3.1. Device emulation mode was
     * removed with the LWCA 3.1 release.
     */
    lwdaErrorSynchronizationError         =     25,
  
    /**
     * This indicates that a non-float texture was being accessed with linear
     * filtering. This is not supported by LWCA.
     */
    lwdaErrorIlwalidFilterSetting         =     26,
  
    /**
     * This indicates that an attempt was made to read a non-float texture as a
     * normalized float. This is not supported by LWCA.
     */
    lwdaErrorIlwalidNormSetting           =     27,
  
    /**
     * Mixing of device and device emulation code was not allowed.
     * \deprecated
     * This error return is deprecated as of LWCA 3.1. Device emulation mode was
     * removed with the LWCA 3.1 release.
     */
    lwdaErrorMixedDeviceExelwtion         =     28,

    /**
     * This indicates that the API call is not yet implemented. Production
     * releases of LWCA will never return this error.
     * \deprecated
     * This error return is deprecated as of LWCA 4.1.
     */
    lwdaErrorNotYetImplemented            =     31,
  
    /**
     * This indicated that an emulated device pointer exceeded the 32-bit address
     * range.
     * \deprecated
     * This error return is deprecated as of LWCA 3.1. Device emulation mode was
     * removed with the LWCA 3.1 release.
     */
    lwdaErrorMemoryValueTooLarge          =     32,
  
    /**
     * This indicates that the LWCA driver that the application has loaded is a
     * stub library. Applications that run with the stub rather than a real
     * driver loaded will result in LWCA API returning this error.
     */
    lwdaErrorStubLibrary                  =     34,

    /**
     * This indicates that the installed LWPU LWCA driver is older than the
     * LWCA runtime library. This is not a supported configuration. Users should
     * install an updated LWPU display driver to allow the application to run.
     */
    lwdaErrorInsufficientDriver           =     35,

    /**
     * This indicates that the API call requires a newer LWCA driver than the one
     * lwrrently installed. Users should install an updated LWPU LWCA driver
     * to allow the API call to succeed.
     */
    lwdaErrorCallRequiresNewerDriver      =     36,
  
    /**
     * This indicates that the surface passed to the API call is not a valid
     * surface.
     */
    lwdaErrorIlwalidSurface               =     37,
  
    /**
     * This indicates that multiple global or constant variables (across separate
     * LWCA source files in the application) share the same string name.
     */
    lwdaErrorDuplicateVariableName        =     43,
  
    /**
     * This indicates that multiple textures (across separate LWCA source
     * files in the application) share the same string name.
     */
    lwdaErrorDuplicateTextureName         =     44,
  
    /**
     * This indicates that multiple surfaces (across separate LWCA source
     * files in the application) share the same string name.
     */
    lwdaErrorDuplicateSurfaceName         =     45,
  
    /**
     * This indicates that all LWCA devices are busy or unavailable at the current
     * time. Devices are often busy/unavailable due to use of
     * ::lwdaComputeModeExclusive, ::lwdaComputeModeProhibited or when long
     * running LWCA kernels have filled up the GPU and are blocking new work
     * from starting. They can also be unavailable due to memory constraints
     * on a device that already has active LWCA work being performed.
     */
    lwdaErrorDevicesUnavailable           =     46,
  
    /**
     * This indicates that the current context is not compatible with this
     * the LWCA Runtime. This can only occur if you are using LWCA
     * Runtime/Driver interoperability and have created an existing Driver
     * context using the driver API. The Driver context may be incompatible
     * either because the Driver context was created using an older version 
     * of the API, because the Runtime API call expects a primary driver 
     * context and the Driver context is not primary, or because the Driver 
     * context has been destroyed. Please see \ref LWDART_DRIVER "Interactions 
     * with the LWCA Driver API" for more information.
     */
    lwdaErrorIncompatibleDriverContext    =     49,
    
    /**
     * The device function being ilwoked (usually via ::lwdaLaunchKernel()) was not
     * previously configured via the ::lwdaConfigureCall() function.
     */
    lwdaErrorMissingConfiguration         =      52,
  
    /**
     * This indicated that a previous kernel launch failed. This was previously
     * used for device emulation of kernel launches.
     * \deprecated
     * This error return is deprecated as of LWCA 3.1. Device emulation mode was
     * removed with the LWCA 3.1 release.
     */
    lwdaErrorPriorLaunchFailure           =      53,

    /**
     * This error indicates that a device runtime grid launch did not occur 
     * because the depth of the child grid would exceed the maximum supported
     * number of nested grid launches. 
     */
    lwdaErrorLaunchMaxDepthExceeded       =     65,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped textures which are unsupported by the device runtime. 
     * Kernels launched via the device runtime only support textures created with 
     * the Texture Object API's.
     */
    lwdaErrorLaunchFileScopedTex          =     66,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped surfaces which are unsupported by the device runtime.
     * Kernels launched via the device runtime only support surfaces created with
     * the Surface Object API's.
     */
    lwdaErrorLaunchFileScopedSurf         =     67,

    /**
     * This error indicates that a call to ::lwdaDeviceSynchronize made from
     * the device runtime failed because the call was made at grid depth greater
     * than than either the default (2 levels of grids) or user specified device 
     * limit ::lwdaLimitDevRuntimeSyncDepth. To be able to synchronize on 
     * launched grids at a greater depth successfully, the maximum nested 
     * depth at which ::lwdaDeviceSynchronize will be called must be specified 
     * with the ::lwdaLimitDevRuntimeSyncDepth limit to the ::lwdaDeviceSetLimit
     * api before the host-side launch of a kernel using the device runtime. 
     * Keep in mind that additional levels of sync depth require the runtime 
     * to reserve large amounts of device memory that cannot be used for 
     * user allocations.
     */
    lwdaErrorSyncDepthExceeded            =     68,

    /**
     * This error indicates that a device runtime grid launch failed because
     * the launch would exceed the limit ::lwdaLimitDevRuntimePendingLaunchCount.
     * For this launch to proceed successfully, ::lwdaDeviceSetLimit must be
     * called to set the ::lwdaLimitDevRuntimePendingLaunchCount to be higher 
     * than the upper bound of outstanding launches that can be issued to the
     * device runtime. Keep in mind that raising the limit of pending device
     * runtime launches will require the runtime to reserve device memory that
     * cannot be used for user allocations.
     */
    lwdaErrorLaunchPendingCountExceeded   =     69,
  
    /**
     * The requested device function does not exist or is not compiled for the
     * proper device architecture.
     */
    lwdaErrorIlwalidDeviceFunction        =      98,
  
    /**
     * This indicates that no LWCA-capable devices were detected by the installed
     * LWCA driver.
     */
    lwdaErrorNoDevice                     =     100,
  
    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid LWCA device.
     */
    lwdaErrorIlwalidDevice                =     101,

    /**
     * This indicates that the device doesn't have a valid Grid License.
     */
    lwdaErrorDeviceNotLicensed            =     102,

   /**
    * By default, the LWCA runtime may perform a minimal set of self-tests,
    * as well as LWCA driver tests, to establish the validity of both.
    * Introduced in LWCA 11.2, this error return indicates that at least one
    * of these tests has failed and the validity of either the runtime
    * or the driver could not be established.
    */
   lwdaErrorSoftwareValidityNotEstablished  =     103,

    /**
     * This indicates an internal startup failure in the LWCA runtime.
     */
    lwdaErrorStartupFailure               =    127,
  
    /**
     * This indicates that the device kernel image is invalid.
     */
    lwdaErrorIlwalidKernelImage           =     200,

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::lwCtxDestroy() ilwoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::lwCtxGetApiVersion() for more details.
     */
    lwdaErrorDeviceUninitialized          =     201,

    /**
     * This indicates that the buffer object could not be mapped.
     */
    lwdaErrorMapBufferObjectFailed        =     205,
  
    /**
     * This indicates that the buffer object could not be unmapped.
     */
    lwdaErrorUnmapBufferObjectFailed      =     206,

    /**
     * This indicates that the specified array is lwrrently mapped and thus
     * cannot be destroyed.
     */
    lwdaErrorArrayIsMapped                =     207,

    /**
     * This indicates that the resource is already mapped.
     */
    lwdaErrorAlreadyMapped                =     208,
  
    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular LWCA source file that do not include the
     * corresponding device configuration.
     */
    lwdaErrorNoKernelImageForDevice       =     209,

    /**
     * This indicates that a resource has already been acquired.
     */
    lwdaErrorAlreadyAcquired              =     210,

    /**
     * This indicates that a resource is not mapped.
     */
    lwdaErrorNotMapped                    =     211,

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    lwdaErrorNotMappedAsArray             =     212,

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    lwdaErrorNotMappedAsPointer           =     213,
  
    /**
     * This indicates that an uncorrectable ECC error was detected during
     * exelwtion.
     */
    lwdaErrorECLWncorrectable             =     214,
  
    /**
     * This indicates that the ::lwdaLimit passed to the API call is not
     * supported by the active device.
     */
    lwdaErrorUnsupportedLimit             =     215,
    
    /**
     * This indicates that a call tried to access an exclusive-thread device that 
     * is already in use by a different thread.
     */
    lwdaErrorDeviceAlreadyInUse           =     216,

    /**
     * This error indicates that P2P access is not supported across the given
     * devices.
     */
    lwdaErrorPeerAccessUnsupported        =     217,

    /**
     * A PTX compilation failed. The runtime may fall back to compiling PTX if
     * an application does not contain a suitable binary for the current device.
     */
    lwdaErrorIlwalidPtx                   =     218,

    /**
     * This indicates an error with the OpenGL or DirectX context.
     */
    lwdaErrorIlwalidGraphicsContext       =     219,

    /**
     * This indicates that an uncorrectable LWLink error was detected during the
     * exelwtion.
     */
    lwdaErrorLwlinkUncorrectable          =     220,

    /**
     * This indicates that the PTX JIT compiler library was not found. The JIT Compiler
     * library is used for PTX compilation. The runtime may fall back to compiling PTX
     * if an application does not contain a suitable binary for the current device.
     */
    lwdaErrorJitCompilerNotFound          =     221,

    /**
     * This indicates that the provided PTX was compiled with an unsupported toolchain.
     * The most common reason for this, is the PTX was generated by a compiler newer
     * than what is supported by the LWCA driver and PTX JIT compiler.
     */
    lwdaErrorUnsupportedPtxVersion        =     222,

    /**
     * This indicates that the JIT compilation was disabled. The JIT compilation compiles
     * PTX. The runtime may fall back to compiling PTX if an application does not contain
     * a suitable binary for the current device.
     */
    lwdaErrorJitCompilationDisabled       =     223,

    /**
     * This indicates that the device kernel source is invalid.
     */
    lwdaErrorIlwalidSource                =     300,

    /**
     * This indicates that the file specified was not found.
     */
    lwdaErrorFileNotFound                 =     301,
  
    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    lwdaErrorSharedObjectSymbolNotFound   =     302,
  
    /**
     * This indicates that initialization of a shared object failed.
     */
    lwdaErrorSharedObjectInitFailed       =     303,

    /**
     * This error indicates that an OS call failed.
     */
    lwdaErrorOperatingSystem              =     304,
  
    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::lwdaStream_t and
     * ::lwdaEvent_t.
     */
    lwdaErrorIlwalidResourceHandle        =     400,

    /**
     * This indicates that a resource required by the API call is not in a
     * valid state to perform the requested operation.
     */
    lwdaErrorIllegalState                 =     401,

    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, driver function names, texture names,
     * and surface names.
     */
    lwdaErrorSymbolNotFound               =     500,
  
    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::lwdaSuccess (which indicates completion). Calls that
     * may return this value include ::lwdaEventQuery() and ::lwdaStreamQuery().
     */
    lwdaErrorNotReady                     =     600,

    /**
     * The device encountered a load or store instruction on an invalid memory address.
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    lwdaErrorIllegalAddress               =     700,
  
    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. Although this error is similar to
     * ::lwdaErrorIlwalidConfiguration, this error usually indicates that the
     * user has attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register count.
     */
    lwdaErrorLaunchOutOfResources         =      701,
  
    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device property
     * \ref ::lwdaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
     * for more information.
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    lwdaErrorLaunchTimeout                =      702,

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    lwdaErrorLaunchIncompatibleTexturing  =     703,
      
    /**
     * This error indicates that a call to ::lwdaDeviceEnablePeerAccess() is
     * trying to re-enable peer addressing on from a context which has already
     * had peer addressing enabled.
     */
    lwdaErrorPeerAccessAlreadyEnabled     =     704,
    
    /**
     * This error indicates that ::lwdaDeviceDisablePeerAccess() is trying to 
     * disable peer addressing which has not been enabled yet via 
     * ::lwdaDeviceEnablePeerAccess().
     */
    lwdaErrorPeerAccessNotEnabled         =     705,
  
    /**
     * This indicates that the user has called ::lwdaSetValidDevices(),
     * ::lwdaSetDeviceFlags(), ::lwdaD3D9SetDirect3DDevice(),
     * ::lwdaD3D10SetDirect3DDevice, ::lwdaD3D11SetDirect3DDevice(), or
     * ::lwdaVDPAUSetVDPAUDevice() after initializing the LWCA runtime by
     * calling non-device management operations (allocating memory and
     * launching kernels are examples of non-device management operations).
     * This error can also be returned if using runtime/driver
     * interoperability and there is an existing ::LWcontext active on the
     * host thread.
     */
    lwdaErrorSetOnActiveProcess           =     708,

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::lwCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    lwdaErrorContextIsDestroyed           =     709,

    /**
     * An assert triggered in device code during kernel exelwtion. The device
     * cannot be used again. All existing allocations are invalid. To continue
     * using LWCA, the process must be terminated and relaunched.
     */
    lwdaErrorAssert                        =    710,
  
    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices 
     * passed to ::lwdaEnablePeerAccess().
     */
    lwdaErrorTooManyPeers                 =     711,
  
    /**
     * This error indicates that the memory range passed to ::lwdaHostRegister()
     * has already been registered.
     */
    lwdaErrorHostMemoryAlreadyRegistered  =     712,
        
    /**
     * This error indicates that the pointer passed to ::lwdaHostUnregister()
     * does not correspond to any lwrrently registered memory region.
     */
    lwdaErrorHostMemoryNotRegistered      =     713,

    /**
     * Device encountered an error in the call stack during kernel exelwtion,
     * possibly due to stack corruption or exceeding the stack size limit.
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    lwdaErrorHardwareStackError           =     714,

    /**
     * The device encountered an illegal instruction during kernel exelwtion
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    lwdaErrorIllegalInstruction           =     715,

    /**
     * The device encountered a load or store instruction
     * on a memory address which is not aligned.
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    lwdaErrorMisalignedAddress            =     716,

    /**
     * While exelwting a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    lwdaErrorIlwalidAddressSpace          =     717,

    /**
     * The device encountered an invalid program counter.
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    lwdaErrorIlwalidPc                    =     718,
  
    /**
     * An exception oclwrred on the device while exelwting a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. Less common cases can be system specific - more
     * information about these cases can be found in the system specific user guide.
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    lwdaErrorLaunchFailure                =      719,

    /**
     * This error indicates that the number of blocks launched per grid for a kernel that was
     * launched via either ::lwdaLaunchCooperativeKernel or ::lwdaLaunchCooperativeKernelMultiDevice
     * exceeds the maximum number of blocks as allowed by ::lwdaOclwpancyMaxActiveBlocksPerMultiprocessor
     * or ::lwdaOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
     * as specified by the device attribute ::lwdaDevAttrMultiProcessorCount.
     */
    lwdaErrorCooperativeLaunchTooLarge    =     720,
    
    /**
     * This error indicates the attempted operation is not permitted.
     */
    lwdaErrorNotPermitted                 =     800,

    /**
     * This error indicates the attempted operation is not supported
     * on the current system or device.
     */
    lwdaErrorNotSupported                 =     801,

    /**
     * This error indicates that the system is not yet ready to start any LWCA
     * work.  To continue using LWCA, verify the system configuration is in a
     * valid state and all required driver daemons are actively running.
     * More information about this error can be found in the system specific
     * user guide.
     */
    lwdaErrorSystemNotReady               =     802,

    /**
     * This error indicates that there is a mismatch between the versions of
     * the display driver and the LWCA driver. Refer to the compatibility documentation
     * for supported versions.
     */
    lwdaErrorSystemDriverMismatch         =     803,

    /**
     * This error indicates that the system was upgraded to run with forward compatibility
     * but the visible hardware detected by LWCA does not support this configuration.
     * Refer to the compatibility documentation for the supported hardware matrix or ensure
     * that only supported hardware is visible during initialization via the LWDA_VISIBLE_DEVICES
     * environment variable.
     */
    lwdaErrorCompatNotSupportedOnDevice   =     804,

    /**
     * This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
     */
    lwdaErrorMpsConnectionFailed          =     805,

    /**
     * This error indicates that the remote procedural call between the MPS server and the MPS client failed.
     */
    lwdaErrorMpsRpcFailure                =     806,

    /**
     * This error indicates that the MPS server is not ready to accept new MPS client requests.
     * This error can be returned when the MPS server is in the process of recovering from a fatal failure.
     */
    lwdaErrorMpsServerNotReady            =     807,

    /**
     * This error indicates that the hardware resources required to create MPS client have been exhausted.
     */
    lwdaErrorMpsMaxClientsReached         =     808,

    /**
     * This error indicates the the hardware resources required to device connections have been exhausted.
     */
    lwdaErrorMpsMaxConnectionsReached     =     809,

    /**
     * The operation is not permitted when the stream is capturing.
     */
    lwdaErrorStreamCaptureUnsupported     =    900,

    /**
     * The current capture sequence on the stream has been ilwalidated due to
     * a previous error.
     */
    lwdaErrorStreamCaptureIlwalidated     =    901,

    /**
     * The operation would have resulted in a merge of two independent capture
     * sequences.
     */
    lwdaErrorStreamCaptureMerge           =    902,

    /**
     * The capture was not initiated in this stream.
     */
    lwdaErrorStreamCaptureUnmatched       =    903,

    /**
     * The capture sequence contains a fork that was not joined to the primary
     * stream.
     */
    lwdaErrorStreamCaptureUnjoined        =    904,

    /**
     * A dependency would have been created which crosses the capture sequence
     * boundary. Only implicit in-stream ordering dependencies are allowed to
     * cross the boundary.
     */
    lwdaErrorStreamCaptureIsolation       =    905,

    /**
     * The operation would have resulted in a disallowed implicit dependency on
     * a current capture sequence from lwdaStreamLegacy.
     */
    lwdaErrorStreamCaptureImplicit        =    906,

    /**
     * The operation is not permitted on an event which was last recorded in a
     * capturing stream.
     */
    lwdaErrorCapturedEvent                =    907,
  
    /**
     * A stream capture sequence not initiated with the ::lwdaStreamCaptureModeRelaxed
     * argument to ::lwdaStreamBeginCapture was passed to ::lwdaStreamEndCapture in a
     * different thread.
     */
    lwdaErrorStreamCaptureWrongThread     =    908,

    /**
     * This indicates that the wait operation has timed out.
     */
    lwdaErrorTimeout                      =    909,

    /**
     * This error indicates that the graph update was not performed because it included 
     * changes which violated constraints specific to instantiated graph update.
     */
    lwdaErrorGraphExelwpdateFailure       =    910,

    /**
     * This indicates that an unknown internal error has oclwrred.
     */
    lwdaErrorUnknown                      =    999,

    /**
     * Any unhandled LWCA driver error is added to this value and returned via
     * the runtime. Production releases of LWCA should not return such errors.
     * \deprecated
     * This error return is deprecated as of LWCA 4.1.
     */
    lwdaErrorApiFailureBase               =  10000
};

/**
 * Channel format kind
 */
enum __device_builtin__ lwdaChannelFormatKind
{
    lwdaChannelFormatKindSigned                 =   0,      /**< Signed channel format */
    lwdaChannelFormatKindUnsigned               =   1,      /**< Unsigned channel format */
    lwdaChannelFormatKindFloat                  =   2,      /**< Float channel format */
    lwdaChannelFormatKindNone                   =   3,      /**< No channel format */
    lwdaChannelFormatKindLW12                   =   4,      /**< Unsigned 8-bit integers, planar 4:2:0 YUV format */
    lwdaChannelFormatKindUnsignedNormalized8X1  =   5,      /**< 1 channel unsigned 8-bit normalized integer */
    lwdaChannelFormatKindUnsignedNormalized8X2  =   6,      /**< 2 channel unsigned 8-bit normalized integer */
    lwdaChannelFormatKindUnsignedNormalized8X4  =   7,      /**< 4 channel unsigned 8-bit normalized integer */
    lwdaChannelFormatKindUnsignedNormalized16X1 =   8,      /**< 1 channel unsigned 16-bit normalized integer */
    lwdaChannelFormatKindUnsignedNormalized16X2 =   9,      /**< 2 channel unsigned 16-bit normalized integer */
    lwdaChannelFormatKindUnsignedNormalized16X4 =   10,     /**< 4 channel unsigned 16-bit normalized integer */
    lwdaChannelFormatKindSignedNormalized8X1    =   11,     /**< 1 channel signed 8-bit normalized integer */
    lwdaChannelFormatKindSignedNormalized8X2    =   12,     /**< 2 channel signed 8-bit normalized integer */
    lwdaChannelFormatKindSignedNormalized8X4    =   13,     /**< 4 channel signed 8-bit normalized integer */
    lwdaChannelFormatKindSignedNormalized16X1   =   14,     /**< 1 channel signed 16-bit normalized integer */
    lwdaChannelFormatKindSignedNormalized16X2   =   15,     /**< 2 channel signed 16-bit normalized integer */
    lwdaChannelFormatKindSignedNormalized16X4   =   16      /**< 4 channel signed 16-bit normalized integer */
};

/**
 * LWCA Channel format descriptor
 */
struct __device_builtin__ lwdaChannelFormatDesc
{
    int                        x; /**< x */
    int                        y; /**< y */
    int                        z; /**< z */
    int                        w; /**< w */
    enum lwdaChannelFormatKind f; /**< Channel format kind */
};

/**
 * LWCA array
 */
typedef struct lwdaArray *lwdaArray_t;

/**
 * LWCA array (as source copy argument)
 */
typedef const struct lwdaArray *lwdaArray_const_t;

struct lwdaArray;

/**
 * LWCA mipmapped array
 */
typedef struct lwdaMipmappedArray *lwdaMipmappedArray_t;

/**
 * LWCA mipmapped array (as source argument)
 */
typedef const struct lwdaMipmappedArray *lwdaMipmappedArray_const_t;

struct lwdaMipmappedArray;

/**
 * Indicates that the layered sparse LWCA array or LWCA mipmapped array has a single mip tail region for all layers
 */
#define lwdaArraySparsePropertiesSingleMipTail   0x1

/**
 * Sparse LWCA array and LWCA mipmapped array properties
 */
struct __device_builtin__ lwdaArraySparseProperties {
    struct {
        unsigned int width;             /**< Tile width in elements */
        unsigned int height;            /**< Tile height in elements */
        unsigned int depth;             /**< Tile depth in elements */
    } tileExtent;
    unsigned int miptailFirstLevel;     /**< First mip level at which the mip tail begins */   
    unsigned long long miptailSize;     /**< Total size of the mip tail. */
    unsigned int flags;                 /**< Flags will either be zero or ::lwdaArraySparsePropertiesSingleMipTail */
    unsigned int reserved[4];
};

/**
 * LWCA memory types
 */
enum __device_builtin__ lwdaMemoryType
{
    lwdaMemoryTypeUnregistered = 0, /**< Unregistered memory */
    lwdaMemoryTypeHost         = 1, /**< Host memory */
    lwdaMemoryTypeDevice       = 2, /**< Device memory */
    lwdaMemoryTypeManaged      = 3  /**< Managed memory */
};

/**
 * LWCA memory copy types
 */
enum __device_builtin__ lwdaMemcpyKind
{
    lwdaMemcpyHostToHost          =   0,      /**< Host   -> Host */
    lwdaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
    lwdaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
    lwdaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
    lwdaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

/**
 * LWCA Pitched memory pointer
 *
 * \sa ::make_lwdaPitchedPtr
 */
struct __device_builtin__ lwdaPitchedPtr
{
    void   *ptr;      /**< Pointer to allocated memory */
    size_t  pitch;    /**< Pitch of allocated memory in bytes */
    size_t  xsize;    /**< Logical width of allocation in elements */
    size_t  ysize;    /**< Logical height of allocation in elements */
};

/**
 * LWCA extent
 *
 * \sa ::make_lwdaExtent
 */
struct __device_builtin__ lwdaExtent
{
    size_t width;     /**< Width in elements when referring to array memory, in bytes when referring to linear memory */
    size_t height;    /**< Height in elements */
    size_t depth;     /**< Depth in elements */
};

/**
 * LWCA 3D position
 *
 * \sa ::make_lwdaPos
 */
struct __device_builtin__ lwdaPos
{
    size_t x;     /**< x */
    size_t y;     /**< y */
    size_t z;     /**< z */
};

/**
 * LWCA 3D memory copying parameters
 */
struct __device_builtin__ lwdaMemcpy3DParms
{
    lwdaArray_t            srcArray;  /**< Source memory address */
    struct lwdaPos         srcPos;    /**< Source position offset */
    struct lwdaPitchedPtr  srcPtr;    /**< Pitched source memory address */
  
    lwdaArray_t            dstArray;  /**< Destination memory address */
    struct lwdaPos         dstPos;    /**< Destination position offset */
    struct lwdaPitchedPtr  dstPtr;    /**< Pitched destination memory address */
  
    struct lwdaExtent      extent;    /**< Requested memory copy size */
    enum lwdaMemcpyKind    kind;      /**< Type of transfer */
};

/**
 * LWCA 3D cross-device memory copying parameters
 */
struct __device_builtin__ lwdaMemcpy3DPeerParms
{
    lwdaArray_t            srcArray;  /**< Source memory address */
    struct lwdaPos         srcPos;    /**< Source position offset */
    struct lwdaPitchedPtr  srcPtr;    /**< Pitched source memory address */
    int                    srcDevice; /**< Source device */
  
    lwdaArray_t            dstArray;  /**< Destination memory address */
    struct lwdaPos         dstPos;    /**< Destination position offset */
    struct lwdaPitchedPtr  dstPtr;    /**< Pitched destination memory address */
    int                    dstDevice; /**< Destination device */
  
    struct lwdaExtent      extent;    /**< Requested memory copy size */
};

/**
 * LWCA Memset node parameters
 */
struct __device_builtin__  lwdaMemsetParams {
    void *dst;                              /**< Destination device pointer */
    size_t pitch;                           /**< Pitch of destination device pointer. Unused if height is 1 */
    unsigned int value;                     /**< Value to be set */
    unsigned int elementSize;               /**< Size of each element in bytes. Must be 1, 2, or 4. */
    size_t width;                           /**< Width of the row in elements */
    size_t height;                          /**< Number of rows */
};

/**
 * Specifies performance hint with ::lwdaAccessPolicyWindow for hitProp and missProp members.
 */
enum __device_builtin__  lwdaAccessProperty {
    lwdaAccessPropertyNormal = 0,       /**< Normal cache persistence. */
    lwdaAccessPropertyStreaming = 1,    /**< Streaming access is less likely to persit from cache. */
    lwdaAccessPropertyPersisting = 2    /**< Persisting access is more likely to persist in cache.*/
};

/**
 * Specifies an access policy for a window, a contiguous extent of memory
 * beginning at base_ptr and ending at base_ptr + num_bytes.
 * Partition into many segments and assign segments such that.
 * sum of "hit segments" / window == approx. ratio.
 * sum of "miss segments" / window == approx 1-ratio.
 * Segments and ratio specifications are fitted to the capabilities of
 * the architecture.
 * Accesses in a hit segment apply the hitProp access policy.
 * Accesses in a miss segment apply the missProp access policy.
 */
struct __device_builtin__ lwdaAccessPolicyWindow {
    void *base_ptr;                     /**< Starting address of the access policy window. LWCA driver may align it. */
    size_t num_bytes;                   /**< Size in bytes of the window policy. LWCA driver may restrict the maximum size and alignment. */
    float hitRatio;                     /**< hitRatio specifies percentage of lines assigned hitProp, rest are assigned missProp. */
    enum lwdaAccessProperty hitProp;    /**< ::LWaccessProperty set for hit. */
    enum lwdaAccessProperty missProp;   /**< ::LWaccessProperty set for miss. Must be either NORMAL or STREAMING. */
};

#ifdef _WIN32
#define LWDART_CB __stdcall
#else
#define LWDART_CB
#endif

/**
 * LWCA host function
 * \param userData Argument value passed to the function
 */
typedef void (LWDART_CB *lwdaHostFn_t)(void *userData);

/**
 * LWCA host node parameters
 */
struct __device_builtin__ lwdaHostNodeParams {
    lwdaHostFn_t fn;    /**< The function to call when the node exelwtes */
    void* userData; /**< Argument to pass to the function */
};

/**
 * Possible stream capture statuses returned by ::lwdaStreamIsCapturing
 */
enum __device_builtin__ lwdaStreamCaptureStatus {
    lwdaStreamCaptureStatusNone        = 0, /**< Stream is not capturing */
    lwdaStreamCaptureStatusActive      = 1, /**< Stream is actively capturing */
    lwdaStreamCaptureStatusIlwalidated = 2  /**< Stream is part of a capture sequence that
                                                   has been ilwalidated, but not terminated */
};

/**
 * Possible modes for stream capture thread interactions. For more details see
 * ::lwdaStreamBeginCapture and ::lwdaThreadExchangeStreamCaptureMode
 */
enum __device_builtin__ lwdaStreamCaptureMode {
    lwdaStreamCaptureModeGlobal      = 0,
    lwdaStreamCaptureModeThreadLocal = 1,
    lwdaStreamCaptureModeRelaxed     = 2
};

enum __device_builtin__ lwdaSynchronizationPolicy {
    lwdaSyncPolicyAuto = 1,
    lwdaSyncPolicySpin = 2,
    lwdaSyncPolicyYield = 3,
    lwdaSyncPolicyBlockingSync = 4
};

/**
 * Stream Attributes
 */
enum __device_builtin__ lwdaStreamAttrID {
    lwdaStreamAttributeAccessPolicyWindow     = 1,  /**< Identifier for ::lwdaStreamAttrValue::accessPolicyWindow. */
    lwdaStreamAttributeSynchronizationPolicy  = 3   /**< ::lwdaSynchronizationPolicy for work queued up in this stream */
};

/**
 * Stream attributes union used with ::lwdaStreamSetAttribute/::lwdaStreamGetAttribute
 */
union __device_builtin__ lwdaStreamAttrValue {
    struct lwdaAccessPolicyWindow accessPolicyWindow;
    enum lwdaSynchronizationPolicy syncPolicy;
};

/**
 * Flags for ::lwdaStreamUpdateCaptureDependencies
 */
enum __device_builtin__ lwdaStreamUpdateCaptureDependenciesFlags {
    lwdaStreamAddCaptureDependencies = 0x0, /**< Add new nodes to the dependency set */
    lwdaStreamSetCaptureDependencies = 0x1  /**< Replace the dependency set with the new nodes */
};

/**
 * Flags for user objects for graphs
 */
enum __device_builtin__ lwdaUserObjectFlags {
    lwdaUserObjectNoDestructorSync = 0x1  /**< Indicates the destructor exelwtion is not synchronized by any LWCA handle. */
};

/**
 * Flags for retaining user object references for graphs
 */
enum __device_builtin__ lwdaUserObjectRetainFlags {
    lwdaGraphUserObjectMove = 0x1  /**< Transfer references from the caller rather than creating new references. */
};

/**
 * LWCA graphics interop resource
 */
struct lwdaGraphicsResource;

/**
 * LWCA graphics interop register flags
 */
enum __device_builtin__ lwdaGraphicsRegisterFlags
{
    lwdaGraphicsRegisterFlagsNone             = 0,  /**< Default */
    lwdaGraphicsRegisterFlagsReadOnly         = 1,  /**< LWCA will not write to this resource */ 
    lwdaGraphicsRegisterFlagsWriteDiscard     = 2,  /**< LWCA will only write to and will not read from this resource */
    lwdaGraphicsRegisterFlagsSurfaceLoadStore = 4,  /**< LWCA will bind this resource to a surface reference */
    lwdaGraphicsRegisterFlagsTextureGather    = 8   /**< LWCA will perform texture gather operations on this resource */
};

/**
 * LWCA graphics interop map flags
 */
enum __device_builtin__ lwdaGraphicsMapFlags
{
    lwdaGraphicsMapFlagsNone         = 0,  /**< Default; Assume resource can be read/written */
    lwdaGraphicsMapFlagsReadOnly     = 1,  /**< LWCA will not write to this resource */
    lwdaGraphicsMapFlagsWriteDiscard = 2   /**< LWCA will only write to and will not read from this resource */
};

/**
 * LWCA graphics interop array indices for lwbe maps
 */
enum __device_builtin__ lwdaGraphicsLwbeFace 
{
    lwdaGraphicsLwbeFacePositiveX = 0x00, /**< Positive X face of lwbemap */
    lwdaGraphicsLwbeFaceNegativeX = 0x01, /**< Negative X face of lwbemap */
    lwdaGraphicsLwbeFacePositiveY = 0x02, /**< Positive Y face of lwbemap */
    lwdaGraphicsLwbeFaceNegativeY = 0x03, /**< Negative Y face of lwbemap */
    lwdaGraphicsLwbeFacePositiveZ = 0x04, /**< Positive Z face of lwbemap */
    lwdaGraphicsLwbeFaceNegativeZ = 0x05  /**< Negative Z face of lwbemap */
};

/**
 * Graph kernel node Attributes
 */
enum __device_builtin__ lwdaKernelNodeAttrID {
    lwdaKernelNodeAttributeAccessPolicyWindow   = 1,   /**< Identifier for ::lwdaKernelNodeAttrValue::accessPolicyWindow. */
    lwdaKernelNodeAttributeCooperative          = 2    /**< Allows a kernel node to be cooperative (see ::lwdaLaunchCooperativeKernel). */
};

/**
 * Graph kernel node attributes union, used with ::lwdaGraphKernelNodeSetAttribute/::lwdaGraphKernelNodeGetAttribute
 */
union __device_builtin__ lwdaKernelNodeAttrValue {
    struct lwdaAccessPolicyWindow accessPolicyWindow;          /**< Attribute ::LWaccessPolicyWindow. */
    int cooperative;
};

/**
 * LWCA resource types
 */
enum __device_builtin__ lwdaResourceType
{
    lwdaResourceTypeArray          = 0x00, /**< Array resource */
    lwdaResourceTypeMipmappedArray = 0x01, /**< Mipmapped array resource */
    lwdaResourceTypeLinear         = 0x02, /**< Linear resource */
    lwdaResourceTypePitch2D        = 0x03  /**< Pitch 2D resource */
};

/**
 * LWCA texture resource view formats
 */
enum __device_builtin__ lwdaResourceViewFormat
{
    lwdaResViewFormatNone                      = 0x00, /**< No resource view format (use underlying resource format) */
    lwdaResViewFormatUnsignedChar1             = 0x01, /**< 1 channel unsigned 8-bit integers */
    lwdaResViewFormatUnsignedChar2             = 0x02, /**< 2 channel unsigned 8-bit integers */
    lwdaResViewFormatUnsignedChar4             = 0x03, /**< 4 channel unsigned 8-bit integers */
    lwdaResViewFormatSignedChar1               = 0x04, /**< 1 channel signed 8-bit integers */
    lwdaResViewFormatSignedChar2               = 0x05, /**< 2 channel signed 8-bit integers */
    lwdaResViewFormatSignedChar4               = 0x06, /**< 4 channel signed 8-bit integers */
    lwdaResViewFormatUnsignedShort1            = 0x07, /**< 1 channel unsigned 16-bit integers */
    lwdaResViewFormatUnsignedShort2            = 0x08, /**< 2 channel unsigned 16-bit integers */
    lwdaResViewFormatUnsignedShort4            = 0x09, /**< 4 channel unsigned 16-bit integers */
    lwdaResViewFormatSignedShort1              = 0x0a, /**< 1 channel signed 16-bit integers */
    lwdaResViewFormatSignedShort2              = 0x0b, /**< 2 channel signed 16-bit integers */
    lwdaResViewFormatSignedShort4              = 0x0c, /**< 4 channel signed 16-bit integers */
    lwdaResViewFormatUnsignedInt1              = 0x0d, /**< 1 channel unsigned 32-bit integers */
    lwdaResViewFormatUnsignedInt2              = 0x0e, /**< 2 channel unsigned 32-bit integers */
    lwdaResViewFormatUnsignedInt4              = 0x0f, /**< 4 channel unsigned 32-bit integers */
    lwdaResViewFormatSignedInt1                = 0x10, /**< 1 channel signed 32-bit integers */
    lwdaResViewFormatSignedInt2                = 0x11, /**< 2 channel signed 32-bit integers */
    lwdaResViewFormatSignedInt4                = 0x12, /**< 4 channel signed 32-bit integers */
    lwdaResViewFormatHalf1                     = 0x13, /**< 1 channel 16-bit floating point */
    lwdaResViewFormatHalf2                     = 0x14, /**< 2 channel 16-bit floating point */
    lwdaResViewFormatHalf4                     = 0x15, /**< 4 channel 16-bit floating point */
    lwdaResViewFormatFloat1                    = 0x16, /**< 1 channel 32-bit floating point */
    lwdaResViewFormatFloat2                    = 0x17, /**< 2 channel 32-bit floating point */
    lwdaResViewFormatFloat4                    = 0x18, /**< 4 channel 32-bit floating point */
    lwdaResViewFormatUnsignedBlockCompressed1  = 0x19, /**< Block compressed 1 */
    lwdaResViewFormatUnsignedBlockCompressed2  = 0x1a, /**< Block compressed 2 */
    lwdaResViewFormatUnsignedBlockCompressed3  = 0x1b, /**< Block compressed 3 */
    lwdaResViewFormatUnsignedBlockCompressed4  = 0x1c, /**< Block compressed 4 unsigned */
    lwdaResViewFormatSignedBlockCompressed4    = 0x1d, /**< Block compressed 4 signed */
    lwdaResViewFormatUnsignedBlockCompressed5  = 0x1e, /**< Block compressed 5 unsigned */
    lwdaResViewFormatSignedBlockCompressed5    = 0x1f, /**< Block compressed 5 signed */
    lwdaResViewFormatUnsignedBlockCompressed6H = 0x20, /**< Block compressed 6 unsigned half-float */
    lwdaResViewFormatSignedBlockCompressed6H   = 0x21, /**< Block compressed 6 signed half-float */
    lwdaResViewFormatUnsignedBlockCompressed7  = 0x22  /**< Block compressed 7 */
};

/**
 * LWCA resource descriptor
 */
struct __device_builtin__ lwdaResourceDesc {
    enum lwdaResourceType resType;             /**< Resource type */
    
    union {
        struct {
            lwdaArray_t array;                 /**< LWCA array */
        } array;
        struct {
            lwdaMipmappedArray_t mipmap;       /**< LWCA mipmapped array */
        } mipmap;
        struct {
            void *devPtr;                      /**< Device pointer */
            struct lwdaChannelFormatDesc desc; /**< Channel descriptor */
            size_t sizeInBytes;                /**< Size in bytes */
        } linear;
        struct {
            void *devPtr;                      /**< Device pointer */
            struct lwdaChannelFormatDesc desc; /**< Channel descriptor */
            size_t width;                      /**< Width of the array in elements */
            size_t height;                     /**< Height of the array in elements */
            size_t pitchInBytes;               /**< Pitch between two rows in bytes */
        } pitch2D;
    } res;
};

/**
 * LWCA resource view descriptor
 */
struct __device_builtin__ lwdaResourceViewDesc
{
    enum lwdaResourceViewFormat format;           /**< Resource view format */
    size_t                      width;            /**< Width of the resource view */
    size_t                      height;           /**< Height of the resource view */
    size_t                      depth;            /**< Depth of the resource view */
    unsigned int                firstMipmapLevel; /**< First defined mipmap level */
    unsigned int                lastMipmapLevel;  /**< Last defined mipmap level */
    unsigned int                firstLayer;       /**< First layer index */
    unsigned int                lastLayer;        /**< Last layer index */
};

/**
 * LWCA pointer attributes
 */
struct __device_builtin__ lwdaPointerAttributes
{
    /**
     * The type of memory - ::lwdaMemoryTypeUnregistered, ::lwdaMemoryTypeHost,
     * ::lwdaMemoryTypeDevice or ::lwdaMemoryTypeManaged.
     */
    enum lwdaMemoryType type;

    /** 
     * The device against which the memory was allocated or registered.
     * If the memory type is ::lwdaMemoryTypeDevice then this identifies 
     * the device on which the memory referred physically resides.  If
     * the memory type is ::lwdaMemoryTypeHost or::lwdaMemoryTypeManaged then
     * this identifies the device which was current when the memory was allocated
     * or registered (and if that device is deinitialized then this allocation
     * will vanish with that device's state).
     */
    int device;

    /**
     * The address which may be dereferenced on the current device to access 
     * the memory or NULL if no such address exists.
     */
    void *devicePointer;

    /**
     * The address which may be dereferenced on the host to access the
     * memory or NULL if no such address exists.
     *
     * \note LWCA doesn't check if unregistered memory is allocated so this field
     * may contain invalid pointer if an invalid pointer has been passed to LWCA.
     */
    void *hostPointer;
};

/**
 * LWCA function attributes
 */
struct __device_builtin__ lwdaFuncAttributes
{
   /**
    * The size in bytes of statically-allocated shared memory per block
    * required by this function. This does not include dynamically-allocated
    * shared memory requested by the user at runtime.
    */
   size_t sharedSizeBytes;

   /**
    * The size in bytes of user-allocated constant memory required by this
    * function.
    */
   size_t constSizeBytes;

   /**
    * The size in bytes of local memory used by each thread of this function.
    */
   size_t localSizeBytes;

   /**
    * The maximum number of threads per block, beyond which a launch of the
    * function would fail. This number depends on both the function and the
    * device on which the function is lwrrently loaded.
    */
   int maxThreadsPerBlock;

   /**
    * The number of registers used by each thread of this function.
    */
   int numRegs;

   /**
    * The PTX virtual architecture version for which the function was
    * compiled. This value is the major PTX version * 10 + the minor PTX
    * version, so a PTX version 1.3 function would return the value 13.
    */
   int ptxVersion;

   /**
    * The binary architecture version for which the function was compiled.
    * This value is the major binary version * 10 + the minor binary version,
    * so a binary version 1.3 function would return the value 13.
    */
   int binaryVersion;

   /**
    * The attribute to indicate whether the function has been compiled with 
    * user specified option "-Xptxas --dlcm=ca" set.
    */
   int cacheModeCA;

   /**
    * The maximum size in bytes of dynamic shared memory per block for 
    * this function. Any launch must have a dynamic shared memory size
    * smaller than this value.
    */
   int maxDynamicSharedSizeBytes;

   /**
    * On devices where the L1 cache and shared memory use the same hardware resources, 
    * this sets the shared memory carveout preference, in percent of the maximum shared memory. 
    * Refer to ::lwdaDevAttrMaxSharedMemoryPerMultiprocessor.
    * This is only a hint, and the driver can choose a different ratio if required to execute the function.
    * See ::lwdaFuncSetAttribute
    */
   int preferredShmemCarveout;
};

/**
 * LWCA function attributes that can be set using ::lwdaFuncSetAttribute
 */
enum __device_builtin__ lwdaFuncAttribute
{
    lwdaFuncAttributeMaxDynamicSharedMemorySize = 8, /**< Maximum dynamic shared memory size */
    lwdaFuncAttributePreferredSharedMemoryCarveout = 9, /**< Preferred shared memory-L1 cache split */
    lwdaFuncAttributeMax
};

/**
 * LWCA function cache configurations
 */
enum __device_builtin__ lwdaFuncCache
{
    lwdaFuncCachePreferNone   = 0,    /**< Default function cache configuration, no preference */
    lwdaFuncCachePreferShared = 1,    /**< Prefer larger shared memory and smaller L1 cache  */
    lwdaFuncCachePreferL1     = 2,    /**< Prefer larger L1 cache and smaller shared memory */
    lwdaFuncCachePreferEqual  = 3     /**< Prefer equal size L1 cache and shared memory */
};

/**
 * LWCA shared memory configuration
 */

enum __device_builtin__ lwdaSharedMemConfig
{
    lwdaSharedMemBankSizeDefault   = 0,
    lwdaSharedMemBankSizeFourByte  = 1,
    lwdaSharedMemBankSizeEightByte = 2
};

/** 
 * Shared memory carveout configurations. These may be passed to lwdaFuncSetAttribute
 */
enum __device_builtin__ lwdaSharedCarveout {
    lwdaSharedmemCarveoutDefault      = -1,  /**< No preference for shared memory or L1 (default) */
    lwdaSharedmemCarveoutMaxShared    = 100, /**< Prefer maximum available shared memory, minimum L1 cache */
    lwdaSharedmemCarveoutMaxL1        = 0    /**< Prefer maximum available L1 cache, minimum shared memory */
};

/**
 * LWCA device compute modes
 */
enum __device_builtin__ lwdaComputeMode
{
    lwdaComputeModeDefault          = 0,  /**< Default compute mode (Multiple threads can use ::lwdaSetDevice() with this device) */
    lwdaComputeModeExclusive        = 1,  /**< Compute-exclusive-thread mode (Only one thread in one process will be able to use ::lwdaSetDevice() with this device) */
    lwdaComputeModeProhibited       = 2,  /**< Compute-prohibited mode (No threads can use ::lwdaSetDevice() with this device) */
    lwdaComputeModeExclusiveProcess = 3   /**< Compute-exclusive-process mode (Many threads in one process will be able to use ::lwdaSetDevice() with this device) */
};

/**
 * LWCA Limits
 */
enum __device_builtin__ lwdaLimit
{
    lwdaLimitStackSize                    = 0x00, /**< GPU thread stack size */
    lwdaLimitPrintfFifoSize               = 0x01, /**< GPU printf FIFO size */
    lwdaLimitMallocHeapSize               = 0x02, /**< GPU malloc heap size */
    lwdaLimitDevRuntimeSyncDepth          = 0x03, /**< GPU device runtime synchronize depth */
    lwdaLimitDevRuntimePendingLaunchCount = 0x04, /**< GPU device runtime pending launch count */
    lwdaLimitMaxL2FetchGranularity        = 0x05, /**< A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint */
    lwdaLimitPersistingL2CacheSize        = 0x06  /**< A size in bytes for L2 persisting lines cache size */
};

/**
 * LWCA Memory Advise values
 */
enum __device_builtin__ lwdaMemoryAdvise
{
    lwdaMemAdviseSetReadMostly          = 1, /**< Data will mostly be read and only occassionally be written to */
    lwdaMemAdviseUnsetReadMostly        = 2, /**< Undo the effect of ::lwdaMemAdviseSetReadMostly */
    lwdaMemAdviseSetPreferredLocation   = 3, /**< Set the preferred location for the data as the specified device */
    lwdaMemAdviseUnsetPreferredLocation = 4, /**< Clear the preferred location for the data */
    lwdaMemAdviseSetAccessedBy          = 5, /**< Data will be accessed by the specified device, so prevent page faults as much as possible */
    lwdaMemAdviseUnsetAccessedBy        = 6  /**< Let the Unified Memory subsystem decide on the page faulting policy for the specified device */
};

/**
 * LWCA range attributes
 */
enum __device_builtin__ lwdaMemRangeAttribute
{
    lwdaMemRangeAttributeReadMostly           = 1, /**< Whether the range will mostly be read and only occassionally be written to */
    lwdaMemRangeAttributePreferredLocation    = 2, /**< The preferred location of the range */
    lwdaMemRangeAttributeAccessedBy           = 3, /**< Memory range has ::lwdaMemAdviseSetAccessedBy set for specified device */
    lwdaMemRangeAttributeLastPrefetchLocation = 4  /**< The last location to which the range was prefetched */
};

/**
 * LWCA Profiler Output modes
 */
enum __device_builtin__ lwdaOutputMode
{
    lwdaKeyValuePair    = 0x00, /**< Output mode Key-Value pair format. */
    lwdaCSV             = 0x01  /**< Output mode Comma separated values format. */
};

/**
 * LWCA GPUDirect RDMA flush writes APIs supported on the device
 */
enum __device_builtin__ lwdaFlushGPUDirectRDMAWritesOptions {
    lwdaFlushGPUDirectRDMAWritesOptionHost   = 1<<0, /**< ::lwdaDeviceFlushGPUDirectRDMAWrites() and its LWCA Driver API counterpart are supported on the device. */
    lwdaFlushGPUDirectRDMAWritesOptionMemOps = 1<<1  /**< The ::LW_STREAM_WAIT_VALUE_FLUSH flag and the ::LW_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the LWCA device. */
};

/**
 * LWCA GPUDirect RDMA flush writes ordering features of the device
 */
enum __device_builtin__ lwdaGPUDirectRDMAWritesOrdering {
    lwdaGPUDirectRDMAWritesOrderingNone       = 0,   /**< The device does not natively support ordering of GPUDirect RDMA writes. ::lwdaFlushGPUDirectRDMAWrites() can be leveraged if supported. */
    lwdaGPUDirectRDMAWritesOrderingOwner      = 100, /**< Natively, the device can consistently consume GPUDirect RDMA writes, although other LWCA devices may not. */
    lwdaGPUDirectRDMAWritesOrderingAllDevices = 200  /**< Any LWCA device in the system can consistently consume GPUDirect RDMA writes to this device. */
};

/**
 * LWCA GPUDirect RDMA flush writes scopes
 */
enum __device_builtin__ lwdaFlushGPUDirectRDMAWritesScope {
    lwdaFlushGPUDirectRDMAWritesToOwner      = 100, /**< Blocks until remote writes are visible to the LWCA device context owning the data. */
    lwdaFlushGPUDirectRDMAWritesToAllDevices = 200  /**< Blocks until remote writes are visible to all LWCA device contexts. */
};

/**
 * LWCA GPUDirect RDMA flush writes targets
 */
enum __device_builtin__ lwdaFlushGPUDirectRDMAWritesTarget {
    lwdaFlushGPUDirectRDMAWritesTargetLwrrentDevice /**< Sets the target for ::lwdaDeviceFlushGPUDirectRDMAWrites() to the lwrrently active LWCA device context. */
};


/**
 * LWCA device attributes
 */
enum __device_builtin__ lwdaDeviceAttr
{
    lwdaDevAttrMaxThreadsPerBlock             = 1,  /**< Maximum number of threads per block */
    lwdaDevAttrMaxBlockDimX                   = 2,  /**< Maximum block dimension X */
    lwdaDevAttrMaxBlockDimY                   = 3,  /**< Maximum block dimension Y */
    lwdaDevAttrMaxBlockDimZ                   = 4,  /**< Maximum block dimension Z */
    lwdaDevAttrMaxGridDimX                    = 5,  /**< Maximum grid dimension X */
    lwdaDevAttrMaxGridDimY                    = 6,  /**< Maximum grid dimension Y */
    lwdaDevAttrMaxGridDimZ                    = 7,  /**< Maximum grid dimension Z */
    lwdaDevAttrMaxSharedMemoryPerBlock        = 8,  /**< Maximum shared memory available per block in bytes */
    lwdaDevAttrTotalConstantMemory            = 9,  /**< Memory available on device for __constant__ variables in a LWCA C kernel in bytes */
    lwdaDevAttrWarpSize                       = 10, /**< Warp size in threads */
    lwdaDevAttrMaxPitch                       = 11, /**< Maximum pitch in bytes allowed by memory copies */
    lwdaDevAttrMaxRegistersPerBlock           = 12, /**< Maximum number of 32-bit registers available per block */
    lwdaDevAttrClockRate                      = 13, /**< Peak clock frequency in kilohertz */
    lwdaDevAttrTextureAlignment               = 14, /**< Alignment requirement for textures */
    lwdaDevAttrGpuOverlap                     = 15, /**< Device can possibly copy memory and execute a kernel conlwrrently */
    lwdaDevAttrMultiProcessorCount            = 16, /**< Number of multiprocessors on device */
    lwdaDevAttrKernelExecTimeout              = 17, /**< Specifies whether there is a run time limit on kernels */
    lwdaDevAttrIntegrated                     = 18, /**< Device is integrated with host memory */
    lwdaDevAttrCanMapHostMemory               = 19, /**< Device can map host memory into LWCA address space */
    lwdaDevAttrComputeMode                    = 20, /**< Compute mode (See ::lwdaComputeMode for details) */
    lwdaDevAttrMaxTexture1DWidth              = 21, /**< Maximum 1D texture width */
    lwdaDevAttrMaxTexture2DWidth              = 22, /**< Maximum 2D texture width */
    lwdaDevAttrMaxTexture2DHeight             = 23, /**< Maximum 2D texture height */
    lwdaDevAttrMaxTexture3DWidth              = 24, /**< Maximum 3D texture width */
    lwdaDevAttrMaxTexture3DHeight             = 25, /**< Maximum 3D texture height */
    lwdaDevAttrMaxTexture3DDepth              = 26, /**< Maximum 3D texture depth */
    lwdaDevAttrMaxTexture2DLayeredWidth       = 27, /**< Maximum 2D layered texture width */
    lwdaDevAttrMaxTexture2DLayeredHeight      = 28, /**< Maximum 2D layered texture height */
    lwdaDevAttrMaxTexture2DLayeredLayers      = 29, /**< Maximum layers in a 2D layered texture */
    lwdaDevAttrSurfaceAlignment               = 30, /**< Alignment requirement for surfaces */
    lwdaDevAttrConlwrrentKernels              = 31, /**< Device can possibly execute multiple kernels conlwrrently */
    lwdaDevAttrEccEnabled                     = 32, /**< Device has ECC support enabled */
    lwdaDevAttrPciBusId                       = 33, /**< PCI bus ID of the device */
    lwdaDevAttrPciDeviceId                    = 34, /**< PCI device ID of the device */
    lwdaDevAttrTccDriver                      = 35, /**< Device is using TCC driver model */
    lwdaDevAttrMemoryClockRate                = 36, /**< Peak memory clock frequency in kilohertz */
    lwdaDevAttrGlobalMemoryBusWidth           = 37, /**< Global memory bus width in bits */
    lwdaDevAttrL2CacheSize                    = 38, /**< Size of L2 cache in bytes */
    lwdaDevAttrMaxThreadsPerMultiProcessor    = 39, /**< Maximum resident threads per multiprocessor */
    lwdaDevAttrAsyncEngineCount               = 40, /**< Number of asynchronous engines */
    lwdaDevAttrUnifiedAddressing              = 41, /**< Device shares a unified address space with the host */    
    lwdaDevAttrMaxTexture1DLayeredWidth       = 42, /**< Maximum 1D layered texture width */
    lwdaDevAttrMaxTexture1DLayeredLayers      = 43, /**< Maximum layers in a 1D layered texture */
    lwdaDevAttrMaxTexture2DGatherWidth        = 45, /**< Maximum 2D texture width if lwdaArrayTextureGather is set */
    lwdaDevAttrMaxTexture2DGatherHeight       = 46, /**< Maximum 2D texture height if lwdaArrayTextureGather is set */
    lwdaDevAttrMaxTexture3DWidthAlt           = 47, /**< Alternate maximum 3D texture width */
    lwdaDevAttrMaxTexture3DHeightAlt          = 48, /**< Alternate maximum 3D texture height */
    lwdaDevAttrMaxTexture3DDepthAlt           = 49, /**< Alternate maximum 3D texture depth */
    lwdaDevAttrPciDomainId                    = 50, /**< PCI domain ID of the device */
    lwdaDevAttrTexturePitchAlignment          = 51, /**< Pitch alignment requirement for textures */
    lwdaDevAttrMaxTextureLwbemapWidth         = 52, /**< Maximum lwbemap texture width/height */
    lwdaDevAttrMaxTextureLwbemapLayeredWidth  = 53, /**< Maximum lwbemap layered texture width/height */
    lwdaDevAttrMaxTextureLwbemapLayeredLayers = 54, /**< Maximum layers in a lwbemap layered texture */
    lwdaDevAttrMaxSurface1DWidth              = 55, /**< Maximum 1D surface width */
    lwdaDevAttrMaxSurface2DWidth              = 56, /**< Maximum 2D surface width */
    lwdaDevAttrMaxSurface2DHeight             = 57, /**< Maximum 2D surface height */
    lwdaDevAttrMaxSurface3DWidth              = 58, /**< Maximum 3D surface width */
    lwdaDevAttrMaxSurface3DHeight             = 59, /**< Maximum 3D surface height */
    lwdaDevAttrMaxSurface3DDepth              = 60, /**< Maximum 3D surface depth */
    lwdaDevAttrMaxSurface1DLayeredWidth       = 61, /**< Maximum 1D layered surface width */
    lwdaDevAttrMaxSurface1DLayeredLayers      = 62, /**< Maximum layers in a 1D layered surface */
    lwdaDevAttrMaxSurface2DLayeredWidth       = 63, /**< Maximum 2D layered surface width */
    lwdaDevAttrMaxSurface2DLayeredHeight      = 64, /**< Maximum 2D layered surface height */
    lwdaDevAttrMaxSurface2DLayeredLayers      = 65, /**< Maximum layers in a 2D layered surface */
    lwdaDevAttrMaxSurfaceLwbemapWidth         = 66, /**< Maximum lwbemap surface width */
    lwdaDevAttrMaxSurfaceLwbemapLayeredWidth  = 67, /**< Maximum lwbemap layered surface width */
    lwdaDevAttrMaxSurfaceLwbemapLayeredLayers = 68, /**< Maximum layers in a lwbemap layered surface */
    lwdaDevAttrMaxTexture1DLinearWidth        = 69, /**< Maximum 1D linear texture width */
    lwdaDevAttrMaxTexture2DLinearWidth        = 70, /**< Maximum 2D linear texture width */
    lwdaDevAttrMaxTexture2DLinearHeight       = 71, /**< Maximum 2D linear texture height */
    lwdaDevAttrMaxTexture2DLinearPitch        = 72, /**< Maximum 2D linear texture pitch in bytes */
    lwdaDevAttrMaxTexture2DMipmappedWidth     = 73, /**< Maximum mipmapped 2D texture width */
    lwdaDevAttrMaxTexture2DMipmappedHeight    = 74, /**< Maximum mipmapped 2D texture height */
    lwdaDevAttrComputeCapabilityMajor         = 75, /**< Major compute capability version number */ 
    lwdaDevAttrComputeCapabilityMinor         = 76, /**< Minor compute capability version number */
    lwdaDevAttrMaxTexture1DMipmappedWidth     = 77, /**< Maximum mipmapped 1D texture width */
    lwdaDevAttrStreamPrioritiesSupported      = 78, /**< Device supports stream priorities */
    lwdaDevAttrGlobalL1CacheSupported         = 79, /**< Device supports caching globals in L1 */
    lwdaDevAttrLocalL1CacheSupported          = 80, /**< Device supports caching locals in L1 */
    lwdaDevAttrMaxSharedMemoryPerMultiprocessor = 81, /**< Maximum shared memory available per multiprocessor in bytes */
    lwdaDevAttrMaxRegistersPerMultiprocessor  = 82, /**< Maximum number of 32-bit registers available per multiprocessor */
    lwdaDevAttrManagedMemory                  = 83, /**< Device can allocate managed memory on this system */
    lwdaDevAttrIsMultiGpuBoard                = 84, /**< Device is on a multi-GPU board */
    lwdaDevAttrMultiGpuBoardGroupID           = 85, /**< Unique identifier for a group of devices on the same multi-GPU board */
    lwdaDevAttrHostNativeAtomicSupported      = 86, /**< Link between the device and the host supports native atomic operations */
    lwdaDevAttrSingleToDoublePrecisionPerfRatio = 87, /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    lwdaDevAttrPageableMemoryAccess           = 88, /**< Device supports coherently accessing pageable memory without calling lwdaHostRegister on it */
    lwdaDevAttrConlwrrentManagedAccess        = 89, /**< Device can coherently access managed memory conlwrrently with the CPU */
    lwdaDevAttrComputePreemptionSupported     = 90, /**< Device supports Compute Preemption */
    lwdaDevAttrCanUseHostPointerForRegisteredMem = 91, /**< Device can access host registered memory at the same virtual address as the CPU */
    lwdaDevAttrReserved92                     = 92,
    lwdaDevAttrReserved93                     = 93,
    lwdaDevAttrReserved94                     = 94,
    lwdaDevAttrCooperativeLaunch              = 95, /**< Device supports launching cooperative kernels via ::lwdaLaunchCooperativeKernel*/
    lwdaDevAttrCooperativeMultiDeviceLaunch   = 96, /**< Deprecated, lwdaLaunchCooperativeKernelMultiDevice is deprecated. */
    lwdaDevAttrMaxSharedMemoryPerBlockOptin   = 97, /**< The maximum optin shared memory per block. This value may vary by chip. See ::lwdaFuncSetAttribute */
    lwdaDevAttrCanFlushRemoteWrites           = 98, /**< Device supports flushing of outstanding remote writes. */
    lwdaDevAttrHostRegisterSupported          = 99, /**< Device supports host memory registration via ::lwdaHostRegister. */
    lwdaDevAttrPageableMemoryAccessUsesHostPageTables = 100, /**< Device accesses pageable memory via the host's page tables. */
    lwdaDevAttrDirectManagedMemAccessFromHost = 101, /**< Host can directly access managed memory on the device without migration. */
    lwdaDevAttrMaxBlocksPerMultiprocessor     = 106, /**< Maximum number of blocks per multiprocessor */
    lwdaDevAttrMaxPersistingL2CacheSize       = 108, /**< Maximum L2 persisting lines capacity setting in bytes. */
    lwdaDevAttrMaxAccessPolicyWindowSize      = 109, /**< Maximum value of lwdaAccessPolicyWindow::num_bytes. */
    lwdaDevAttrReservedSharedMemoryPerBlock   = 111, /**< Shared memory reserved by LWCA driver per block in bytes */
    lwdaDevAttrSparseLwdaArraySupported       = 112, /**< Device supports sparse LWCA arrays and sparse LWCA mipmapped arrays */
    lwdaDevAttrHostRegisterReadOnlySupported  = 113,  /**< Device supports using the ::lwdaHostRegister flag lwdaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
    lwdaDevAttrMaxTimelineSemaphoreInteropSupported = 114,  /**< External timeline semaphore interop is supported on the device */
    lwdaDevAttrMemoryPoolsSupported           = 115, /**< Device supports using the ::lwdaMallocAsync and ::lwdaMemPool family of APIs */
    lwdaDevAttrGPUDirectRDMASupported         = 116, /**< Device supports GPUDirect RDMA APIs, like lwidia_p2p_get_pages (see https://docs.lwpu.com/lwca/gpudirect-rdma for more information) */
    lwdaDevAttrGPUDirectRDMAFlushWritesOptions = 117, /**< The returned attribute shall be interpreted as a bitmask, where the individual bits are listed in the ::lwdaFlushGPUDirectRDMAWritesOptions enum */
    lwdaDevAttrGPUDirectRDMAWritesOrdering    = 118, /**< GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::lwdaGPUDirectRDMAWritesOrdering for the numerical values returned here. */
    lwdaDevAttrMemoryPoolSupportedHandleTypes = 119, /**< Handle types supported with mempool based IPC */
    lwdaDevAttrMax
};

/**
 * LWCA memory pool attributes
 */
enum __device_builtin__ lwdaMemPoolAttr
{
    /**
     * (value type = int)
     * Allow lwMemAllocAsync to use memory asynchronously freed
     * in another streams as long as a stream ordering dependency
     * of the allocating stream on the free action exists.
     * Lwca events and null stream interactions can create the required
     * stream ordered dependencies. (default enabled)
     */
    lwdaMemPoolReuseFollowEventDependencies   = 0x1,

    /**
     * (value type = int)
     * Allow reuse of already completed frees when there is no dependency
     * between the free and allocation. (default enabled)
     */
    lwdaMemPoolReuseAllowOpportunistic        = 0x2,

    /**
     * (value type = int)
     * Allow lwMemAllocAsync to insert new stream dependencies
     * in order to establish the stream ordering required to reuse
     * a piece of memory released by lwFreeAsync (default enabled).
     */
    lwdaMemPoolReuseAllowInternalDependencies = 0x3,


    /**
     * (value type = lwuint64_t)
     * Amount of reserved memory in bytes to hold onto before trying
     * to release memory back to the OS. When more than the release
     * threshold bytes of memory are held by the memory pool, the
     * allocator will try to release memory back to the OS on the
     * next call to stream, event or context synchronize. (default 0)
     */
    lwdaMemPoolAttrReleaseThreshold           = 0x4,

    /**
     * (value type = lwuint64_t)
     * Amount of backing memory lwrrently allocated for the mempool.
     */
    lwdaMemPoolAttrReservedMemLwrrent         = 0x5,

    /**
     * (value type = lwuint64_t)
     * High watermark of backing memory allocated for the mempool since the
     * last time it was reset. High watermark can only be reset to zero.
     */
    lwdaMemPoolAttrReservedMemHigh            = 0x6,

    /**
     * (value type = lwuint64_t)
     * Amount of memory from the pool that is lwrrently in use by the application.
     */
    lwdaMemPoolAttrUsedMemLwrrent             = 0x7,

    /**
     * (value type = lwuint64_t)
     * High watermark of the amount of memory from the pool that was in use by the application since
     * the last time it was reset. High watermark can only be reset to zero.
     */
    lwdaMemPoolAttrUsedMemHigh                = 0x8
};

/**
 * Specifies the type of location 
 */
enum __device_builtin__ lwdaMemLocationType {
    lwdaMemLocationTypeIlwalid = 0,
    lwdaMemLocationTypeDevice = 1  /**< Location is a device location, thus id is a device ordinal */
};

/**
 * Specifies a memory location.
 *
 * To specify a gpu, set type = ::lwdaMemLocationTypeDevice and set id = the gpu's device ordinal.
 */
struct __device_builtin__ lwdaMemLocation {
    enum lwdaMemLocationType type;  /**< Specifies the location type, which modifies the meaning of id. */
    int id;                         /**< identifier for a given this location's ::LWmemLocationType. */
};

/**
 * Specifies the memory protection flags for mapping.
 */
enum __device_builtin__ lwdaMemAccessFlags {
    lwdaMemAccessFlagsProtNone      = 0,  /**< Default, make the address range not accessible */
    lwdaMemAccessFlagsProtRead      = 1,  /**< Make the address range read accessible */
    lwdaMemAccessFlagsProtReadWrite = 3   /**< Make the address range read-write accessible */
};

/**
 * Memory access descriptor
 */
struct __device_builtin__ lwdaMemAccessDesc {
    struct lwdaMemLocation  location; /**< Location on which the request is to change it's accessibility */
    enum lwdaMemAccessFlags flags;    /**< ::LWmemProt accessibility flags to set on the request */
};

/**
 * Defines the allocation types available
 */
enum __device_builtin__ lwdaMemAllocationType {
    lwdaMemAllocationTypeIlwalid = 0x0,
    /** This allocation type is 'pinned', i.e. cannot migrate from its current
      * location while the application is actively using it
      */
    lwdaMemAllocationTypePinned  = 0x1,
    lwdaMemAllocationTypeMax     = 0x7FFFFFFF 
};

/**
 * Flags for specifying particular handle types
 */
enum __device_builtin__ lwdaMemAllocationHandleType {
    lwdaMemHandleTypeNone                    = 0x0,  /**< Does not allow any export mechanism. > */
    lwdaMemHandleTypePosixFileDescriptor     = 0x1,  /**< Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int) */
    lwdaMemHandleTypeWin32                   = 0x2,  /**< Allows a Win32 NT handle to be used for exporting. (HANDLE) */
    lwdaMemHandleTypeWin32Kmt                = 0x4   /**< Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE) */
};

/**
 * Specifies the properties of allocations made from the pool.
 */
struct __device_builtin__ lwdaMemPoolProps {
    enum lwdaMemAllocationType         allocType;   /**< Allocation type. Lwrrently must be specified as lwdaMemAllocationTypePinned */
    enum lwdaMemAllocationHandleType   handleTypes; /**< Handle types that will be supported by allocations from the pool. */
    struct lwdaMemLocation             location;    /**< Location allocations should reside. */
    /**
     * Windows-specific LPSELWRITYATTRIBUTES required when
     * ::lwdaMemHandleTypeWin32 is specified.  This security attribute defines
     * the scope of which exported allocations may be tranferred to other
     * processes.  In all other cases, this field is required to be zero.
     */
    void                              *win32SelwrityAttributes;
    unsigned char                      reserved[64]; /**< reserved for future use, must be 0 */
};

/**
 * Opaque data for exporting a pool allocation
 */
struct __device_builtin__ lwdaMemPoolPtrExportData {
    unsigned char reserved[64];
};

/**
 * LWCA device P2P attributes
 */

enum __device_builtin__ lwdaDeviceP2PAttr {
    lwdaDevP2PAttrPerformanceRank              = 1, /**< A relative value indicating the performance of the link between two devices */
    lwdaDevP2PAttrAccessSupported              = 2, /**< Peer access is enabled */
    lwdaDevP2PAttrNativeAtomicSupported        = 3, /**< Native atomic operation over the link supported */
    lwdaDevP2PAttrLwdaArrayAccessSupported     = 4  /**< Accessing LWCA arrays over the link supported */
};

/**
 * LWCA UUID types
 */
#ifndef LW_UUID_HAS_BEEN_DEFINED
#define LW_UUID_HAS_BEEN_DEFINED
struct __device_builtin__ LWuuid_st {     /**< LWCA definition of UUID */
    char bytes[16];
};
typedef __device_builtin__ struct LWuuid_st LWuuid;
#endif
typedef __device_builtin__ struct LWuuid_st lwdaUUID_t;

/**
 * LWCA device properties
 */
struct __device_builtin__ lwdaDeviceProp
{
    char         name[256];                  /**< ASCII string identifying device */
    lwdaUUID_t   uuid;                       /**< 16-byte unique identifier */
    char         luid[8];                    /**< 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms */
    unsigned int luidDeviceNodeMask;         /**< LUID device node mask. Value is undefined on TCC and non-Windows platforms */
    size_t       totalGlobalMem;             /**< Global memory available on device in bytes */
    size_t       sharedMemPerBlock;          /**< Shared memory available per block in bytes */
    int          regsPerBlock;               /**< 32-bit registers available per block */
    int          warpSize;                   /**< Warp size in threads */
    size_t       memPitch;                   /**< Maximum pitch in bytes allowed by memory copies */
    int          maxThreadsPerBlock;         /**< Maximum number of threads per block */
    int          maxThreadsDim[3];           /**< Maximum size of each dimension of a block */
    int          maxGridSize[3];             /**< Maximum size of each dimension of a grid */
    int          clockRate;                  /**< Clock frequency in kilohertz */
    size_t       totalConstMem;              /**< Constant memory available on device in bytes */
    int          major;                      /**< Major compute capability */
    int          minor;                      /**< Minor compute capability */
    size_t       textureAlignment;           /**< Alignment requirement for textures */
    size_t       texturePitchAlignment;      /**< Pitch alignment requirement for texture references bound to pitched memory */
    int          deviceOverlap;              /**< Device can conlwrrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
    int          multiProcessorCount;        /**< Number of multiprocessors on device */
    int          kernelExecTimeoutEnabled;   /**< Specified whether there is a run time limit on kernels */
    int          integrated;                 /**< Device is integrated as opposed to discrete */
    int          canMapHostMemory;           /**< Device can map host memory with lwdaHostAlloc/lwdaHostGetDevicePointer */
    int          computeMode;                /**< Compute mode (See ::lwdaComputeMode) */
    int          maxTexture1D;               /**< Maximum 1D texture size */
    int          maxTexture1DMipmap;         /**< Maximum 1D mipmapped texture size */
    int          maxTexture1DLinear;         /**< Deprecated, do not use. Use lwdaDeviceGetTexture1DLinearMaxWidth() or lwDeviceGetTexture1DLinearMaxWidth() instead. */
    int          maxTexture2D[2];            /**< Maximum 2D texture dimensions */
    int          maxTexture2DMipmap[2];      /**< Maximum 2D mipmapped texture dimensions */
    int          maxTexture2DLinear[3];      /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    int          maxTexture2DGather[2];      /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
    int          maxTexture3D[3];            /**< Maximum 3D texture dimensions */
    int          maxTexture3DAlt[3];         /**< Maximum alternate 3D texture dimensions */
    int          maxTextureLwbemap;          /**< Maximum Lwbemap texture dimensions */
    int          maxTexture1DLayered[2];     /**< Maximum 1D layered texture dimensions */
    int          maxTexture2DLayered[3];     /**< Maximum 2D layered texture dimensions */
    int          maxTextureLwbemapLayered[2];/**< Maximum Lwbemap layered texture dimensions */
    int          maxSurface1D;               /**< Maximum 1D surface size */
    int          maxSurface2D[2];            /**< Maximum 2D surface dimensions */
    int          maxSurface3D[3];            /**< Maximum 3D surface dimensions */
    int          maxSurface1DLayered[2];     /**< Maximum 1D layered surface dimensions */
    int          maxSurface2DLayered[3];     /**< Maximum 2D layered surface dimensions */
    int          maxSurfaceLwbemap;          /**< Maximum Lwbemap surface dimensions */
    int          maxSurfaceLwbemapLayered[2];/**< Maximum Lwbemap layered surface dimensions */
    size_t       surfaceAlignment;           /**< Alignment requirements for surfaces */
    int          conlwrrentKernels;          /**< Device can possibly execute multiple kernels conlwrrently */
    int          ECCEnabled;                 /**< Device has ECC support enabled */
    int          pciBusID;                   /**< PCI bus ID of the device */
    int          pciDeviceID;                /**< PCI device ID of the device */
    int          pciDomainID;                /**< PCI domain ID of the device */
    int          tccDriver;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
    int          asyncEngineCount;           /**< Number of asynchronous engines */
    int          unifiedAddressing;          /**< Device shares a unified address space with the host */
    int          memoryClockRate;            /**< Peak memory clock frequency in kilohertz */
    int          memoryBusWidth;             /**< Global memory bus width in bits */
    int          l2CacheSize;                /**< Size of L2 cache in bytes */
    int          persistingL2CacheMaxSize;   /**< Device's maximum l2 persisting lines capacity setting in bytes */
    int          maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */
    int          streamPrioritiesSupported;  /**< Device supports stream priorities */
    int          globalL1CacheSupported;     /**< Device supports caching globals in L1 */
    int          localL1CacheSupported;      /**< Device supports caching locals in L1 */
    size_t       sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes */
    int          regsPerMultiprocessor;      /**< 32-bit registers available per multiprocessor */
    int          managedMemory;              /**< Device supports allocating managed memory on this system */
    int          isMultiGpuBoard;            /**< Device is on a multi-GPU board */
    int          multiGpuBoardGroupID;       /**< Unique identifier for a group of devices on the same multi-GPU board */
    int          hostNativeAtomicSupported;  /**< Link between the device and the host supports native atomic operations */
    int          singleToDoublePrecisionPerfRatio; /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    int          pageableMemoryAccess;       /**< Device supports coherently accessing pageable memory without calling lwdaHostRegister on it */
    int          conlwrrentManagedAccess;    /**< Device can coherently access managed memory conlwrrently with the CPU */
    int          computePreemptionSupported; /**< Device supports Compute Preemption */
    int          canUseHostPointerForRegisteredMem; /**< Device can access host registered memory at the same virtual address as the CPU */
    int          cooperativeLaunch;          /**< Device supports launching cooperative kernels via ::lwdaLaunchCooperativeKernel */
    int          cooperativeMultiDeviceLaunch; /**< Deprecated, lwdaLaunchCooperativeKernelMultiDevice is deprecated. */
    size_t       sharedMemPerBlockOptin;     /**< Per device maximum shared memory per block usable by special opt in */
    int          pageableMemoryAccessUsesHostPageTables; /**< Device accesses pageable memory via the host's page tables */
    int          directManagedMemAccessFromHost; /**< Host can directly access managed memory on the device without migration. */
    int          maxBlocksPerMultiProcessor; /**< Maximum number of resident blocks per multiprocessor */
    int          accessPolicyMaxWindowSize;  /**< The maximum value of ::lwdaAccessPolicyWindow::num_bytes. */
    size_t       reservedSharedMemPerBlock;  /**< Shared memory reserved by LWCA driver per block in bytes */
};

#define lwdaDevicePropDontCare                                 \
        {                                                      \
          {'\0'},    /* char         name[256];               */ \
          {{0}},     /* lwdaUUID_t   uuid;                    */ \
          {'\0'},    /* char         luid[8];                 */ \
          0,         /* unsigned int luidDeviceNodeMask       */ \
          0,         /* size_t       totalGlobalMem;          */ \
          0,         /* size_t       sharedMemPerBlock;       */ \
          0,         /* int          regsPerBlock;            */ \
          0,         /* int          warpSize;                */ \
          0,         /* size_t       memPitch;                */ \
          0,         /* int          maxThreadsPerBlock;      */ \
          {0, 0, 0}, /* int          maxThreadsDim[3];        */ \
          {0, 0, 0}, /* int          maxGridSize[3];          */ \
          0,         /* int          clockRate;               */ \
          0,         /* size_t       totalConstMem;           */ \
          -1,        /* int          major;                   */ \
          -1,        /* int          minor;                   */ \
          0,         /* size_t       textureAlignment;        */ \
          0,         /* size_t       texturePitchAlignment    */ \
          -1,        /* int          deviceOverlap;           */ \
          0,         /* int          multiProcessorCount;     */ \
          0,         /* int          kernelExecTimeoutEnabled */ \
          0,         /* int          integrated               */ \
          0,         /* int          canMapHostMemory         */ \
          0,         /* int          computeMode              */ \
          0,         /* int          maxTexture1D             */ \
          0,         /* int          maxTexture1DMipmap       */ \
          0,         /* int          maxTexture1DLinear       */ \
          {0, 0},    /* int          maxTexture2D[2]          */ \
          {0, 0},    /* int          maxTexture2DMipmap[2]    */ \
          {0, 0, 0}, /* int          maxTexture2DLinear[3]    */ \
          {0, 0},    /* int          maxTexture2DGather[2]    */ \
          {0, 0, 0}, /* int          maxTexture3D[3]          */ \
          {0, 0, 0}, /* int          maxTexture3DAlt[3]       */ \
          0,         /* int          maxTextureLwbemap        */ \
          {0, 0},    /* int          maxTexture1DLayered[2]   */ \
          {0, 0, 0}, /* int          maxTexture2DLayered[3]   */ \
          {0, 0},    /* int          maxTextureLwbemapLayered[2] */ \
          0,         /* int          maxSurface1D             */ \
          {0, 0},    /* int          maxSurface2D[2]          */ \
          {0, 0, 0}, /* int          maxSurface3D[3]          */ \
          {0, 0},    /* int          maxSurface1DLayered[2]   */ \
          {0, 0, 0}, /* int          maxSurface2DLayered[3]   */ \
          0,         /* int          maxSurfaceLwbemap        */ \
          {0, 0},    /* int          maxSurfaceLwbemapLayered[2] */ \
          0,         /* size_t       surfaceAlignment         */ \
          0,         /* int          conlwrrentKernels        */ \
          0,         /* int          ECCEnabled               */ \
          0,         /* int          pciBusID                 */ \
          0,         /* int          pciDeviceID              */ \
          0,         /* int          pciDomainID              */ \
          0,         /* int          tccDriver                */ \
          0,         /* int          asyncEngineCount         */ \
          0,         /* int          unifiedAddressing        */ \
          0,         /* int          memoryClockRate          */ \
          0,         /* int          memoryBusWidth           */ \
          0,         /* int          l2CacheSize              */ \
          0,         /* int          persistingL2CacheMaxSize   */ \
          0,         /* int          maxThreadsPerMultiProcessor */ \
          0,         /* int          streamPrioritiesSupported */ \
          0,         /* int          globalL1CacheSupported   */ \
          0,         /* int          localL1CacheSupported    */ \
          0,         /* size_t       sharedMemPerMultiprocessor; */ \
          0,         /* int          regsPerMultiprocessor;   */ \
          0,         /* int          managedMemory            */ \
          0,         /* int          isMultiGpuBoard          */ \
          0,         /* int          multiGpuBoardGroupID     */ \
          0,         /* int          hostNativeAtomicSupported */ \
          0,         /* int          singleToDoublePrecisionPerfRatio */ \
          0,         /* int          pageableMemoryAccess     */ \
          0,         /* int          conlwrrentManagedAccess  */ \
          0,         /* int          computePreemptionSupported */ \
          0,         /* int          canUseHostPointerForRegisteredMem */ \
          0,         /* int          cooperativeLaunch */ \
          0,         /* int          cooperativeMultiDeviceLaunch */ \
          0,         /* size_t       sharedMemPerBlockOptin */ \
          0,         /* int          pageableMemoryAccessUsesHostPageTables */ \
          0,         /* int          directManagedMemAccessFromHost */ \
          0,         /* int          accessPolicyMaxWindowSize */ \
          0,         /* size_t       reservedSharedMemPerBlock */ \
        } /**< Empty device properties */

/**
 * LWCA IPC Handle Size
 */
#define LWDA_IPC_HANDLE_SIZE 64

/**
 * LWCA IPC event handle
 */
typedef __device_builtin__ struct __device_builtin__ lwdaIpcEventHandle_st
{
    char reserved[LWDA_IPC_HANDLE_SIZE];
}lwdaIpcEventHandle_t;

/**
 * LWCA IPC memory handle
 */
typedef __device_builtin__ struct __device_builtin__ lwdaIpcMemHandle_st 
{
    char reserved[LWDA_IPC_HANDLE_SIZE];
}lwdaIpcMemHandle_t;

/**
 * External memory handle types
 */
enum __device_builtin__ lwdaExternalMemoryHandleType {
    /**
     * Handle is an opaque file descriptor
     */
    lwdaExternalMemoryHandleTypeOpaqueFd         = 1,
    /**
     * Handle is an opaque shared NT handle
     */
    lwdaExternalMemoryHandleTypeOpaqueWin32      = 2,
    /**
     * Handle is an opaque, globally shared handle
     */
    lwdaExternalMemoryHandleTypeOpaqueWin32Kmt   = 3,
    /**
     * Handle is a D3D12 heap object
     */
    lwdaExternalMemoryHandleTypeD3D12Heap        = 4,
    /**
     * Handle is a D3D12 committed resource
     */
    lwdaExternalMemoryHandleTypeD3D12Resource    = 5,
    /**
    *  Handle is a shared NT handle to a D3D11 resource
    */
    lwdaExternalMemoryHandleTypeD3D11Resource    = 6,
    /**
    *  Handle is a globally shared handle to a D3D11 resource
    */
    lwdaExternalMemoryHandleTypeD3D11ResourceKmt = 7,
    /**
    *  Handle is an LwSciBuf object
    */
    lwdaExternalMemoryHandleTypeLwSciBuf         = 8
};

/**
 * Indicates that the external memory object is a dedicated resource
 */
#define lwdaExternalMemoryDedicated   0x1

/** When the /p flags parameter of ::lwdaExternalSemaphoreSignalParams
 * contains this flag, it indicates that signaling an external semaphore object
 * should skip performing appropriate memory synchronization operations over all
 * the external memory objects that are imported as ::lwdaExternalMemoryHandleTypeLwSciBuf,
 * which otherwise are performed by default to ensure data coherency with other
 * importers of the same LwSciBuf memory objects.
 */
#define lwdaExternalSemaphoreSignalSkipLwSciBufMemSync     0x01

/** When the /p flags parameter of ::lwdaExternalSemaphoreWaitParams
 * contains this flag, it indicates that waiting an external semaphore object
 * should skip performing appropriate memory synchronization operations over all
 * the external memory objects that are imported as ::lwdaExternalMemoryHandleTypeLwSciBuf,
 * which otherwise are performed by default to ensure data coherency with other
 * importers of the same LwSciBuf memory objects.
 */
#define lwdaExternalSemaphoreWaitSkipLwSciBufMemSync       0x02

/**
 * When /p flags of ::lwdaDeviceGetLwSciSyncAttributes is set to this,
 * it indicates that application need signaler specific LwSciSyncAttr
 * to be filled by ::lwdaDeviceGetLwSciSyncAttributes.
 */
#define lwdaLwSciSyncAttrSignal       0x1

/**
 * When /p flags of ::lwdaDeviceGetLwSciSyncAttributes is set to this,
 * it indicates that application need waiter specific LwSciSyncAttr
 * to be filled by ::lwdaDeviceGetLwSciSyncAttributes.
 */
#define lwdaLwSciSyncAttrWait         0x2

/**
 * External memory handle descriptor
 */
struct __device_builtin__ lwdaExternalMemoryHandleDesc {
    /**
     * Type of the handle
     */
    enum lwdaExternalMemoryHandleType type;
    union {
        /**
         * File descriptor referencing the memory object. Valid
         * when type is
         * ::lwdaExternalMemoryHandleTypeOpaqueFd
         */
        int fd;
        /**
         * Win32 handle referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::lwdaExternalMemoryHandleTypeOpaqueWin32
         * - ::lwdaExternalMemoryHandleTypeOpaqueWin32Kmt
         * - ::lwdaExternalMemoryHandleTypeD3D12Heap 
         * - ::lwdaExternalMemoryHandleTypeD3D12Resource
		 * - ::lwdaExternalMemoryHandleTypeD3D11Resource
		 * - ::lwdaExternalMemoryHandleTypeD3D11ResourceKmt
         * Exactly one of 'handle' and 'name' must be non-NULL. If
         * type is one of the following: 
         * ::lwdaExternalMemoryHandleTypeOpaqueWin32Kmt
         * ::lwdaExternalMemoryHandleTypeD3D11ResourceKmt
         * then 'name' must be NULL.
         */
        struct {
            /**
             * Valid NT handle. Must be NULL if 'name' is non-NULL
             */
            void *handle;
            /**
             * Name of a valid memory object.
             * Must be NULL if 'handle' is non-NULL.
             */
            const void *name;
        } win32;
        /**
         * A handle representing LwSciBuf Object. Valid when type
         * is ::lwdaExternalMemoryHandleTypeLwSciBuf
         */
        const void *lwSciBufObject;
    } handle;
    /**
     * Size of the memory allocation
     */
    unsigned long long size;
    /**
     * Flags must either be zero or ::lwdaExternalMemoryDedicated
     */
    unsigned int flags;
};

/**
 * External memory buffer descriptor
 */
struct __device_builtin__ lwdaExternalMemoryBufferDesc {
    /**
     * Offset into the memory object where the buffer's base is
     */
    unsigned long long offset;
    /**
     * Size of the buffer
     */
    unsigned long long size;
    /**
     * Flags reserved for future use. Must be zero.
     */
    unsigned int flags;
};
 
/**
 * External memory mipmap descriptor
 */
struct __device_builtin__ lwdaExternalMemoryMipmappedArrayDesc {
    /**
     * Offset into the memory object where the base level of the
     * mipmap chain is.
     */
    unsigned long long offset;
    /**
     * Format of base level of the mipmap chain
     */
    struct lwdaChannelFormatDesc formatDesc;
    /**
     * Dimensions of base level of the mipmap chain
     */
    struct lwdaExtent extent;
    /**
     * Flags associated with LWCA mipmapped arrays.
     * See ::lwdaMallocMipmappedArray
     */
    unsigned int flags;
    /**
     * Total number of levels in the mipmap chain
     */
    unsigned int numLevels;
};
 
/**
 * External semaphore handle types
 */
enum __device_builtin__ lwdaExternalSemaphoreHandleType {
    /**
     * Handle is an opaque file descriptor
     */
    lwdaExternalSemaphoreHandleTypeOpaqueFd       = 1,
    /**
     * Handle is an opaque shared NT handle
     */
    lwdaExternalSemaphoreHandleTypeOpaqueWin32    = 2,
    /**
     * Handle is an opaque, globally shared handle
     */
    lwdaExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
    /**
     * Handle is a shared NT handle referencing a D3D12 fence object
     */
    lwdaExternalSemaphoreHandleTypeD3D12Fence     = 4,
    /**
     * Handle is a shared NT handle referencing a D3D11 fence object
     */
    lwdaExternalSemaphoreHandleTypeD3D11Fence     = 5,
    /**
     * Opaque handle to LwSciSync Object
     */
     lwdaExternalSemaphoreHandleTypeLwSciSync     = 6,
    /**
     * Handle is a shared NT handle referencing a D3D11 keyed mutex object
     */
    lwdaExternalSemaphoreHandleTypeKeyedMutex     = 7,
    /**
     * Handle is a shared KMT handle referencing a D3D11 keyed mutex object
     */
    lwdaExternalSemaphoreHandleTypeKeyedMutexKmt  = 8,
    /**
     * Handle is an opaque handle file descriptor referencing a timeline semaphore
     */
    lwdaExternalSemaphoreHandleTypeTimelineSemaphoreFd  = 9,
    /**
     * Handle is an opaque handle file descriptor referencing a timeline semaphore
     */
    lwdaExternalSemaphoreHandleTypeTimelineSemaphoreWin32  = 10
};

/**
 * External semaphore handle descriptor
 */
struct __device_builtin__ lwdaExternalSemaphoreHandleDesc {
    /**
     * Type of the handle
     */
    enum lwdaExternalSemaphoreHandleType type;
    union {
        /**
         * File descriptor referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::lwdaExternalSemaphoreHandleTypeOpaqueFd
         * - ::lwdaExternalSemaphoreHandleTypeTimelineSemaphoreFd
         */
        int fd;
        /**
         * Win32 handle referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::lwdaExternalSemaphoreHandleTypeOpaqueWin32
         * - ::lwdaExternalSemaphoreHandleTypeOpaqueWin32Kmt
         * - ::lwdaExternalSemaphoreHandleTypeD3D12Fence
         * - ::lwdaExternalSemaphoreHandleTypeD3D11Fence
         * - ::lwdaExternalSemaphoreHandleTypeKeyedMutex
         * - ::lwdaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
         * Exactly one of 'handle' and 'name' must be non-NULL. If
         * type is one of the following:
         * ::lwdaExternalSemaphoreHandleTypeOpaqueWin32Kmt
         * ::lwdaExternalSemaphoreHandleTypeKeyedMutexKmt
         * then 'name' must be NULL.
         */
        struct {
            /**
             * Valid NT handle. Must be NULL if 'name' is non-NULL
             */
            void *handle;
            /**
             * Name of a valid synchronization primitive.
             * Must be NULL if 'handle' is non-NULL.
             */
            const void *name;
        } win32;
        /**
         * Valid LwSciSyncObj. Must be non NULL
         */
        const void* lwSciSyncObj;
    } handle;
    /**
     * Flags reserved for the future. Must be zero.
     */
    unsigned int flags;
};

#if defined(__LWDA_API_VERSION_INTERNAL)
/**
 * External semaphore signal parameters(deprecated)
 */
struct __device_builtin__ lwdaExternalSemaphoreSignalParams_v1 {
    struct {
        /**
         * Parameters for fence objects
         */
        struct {
            /**
             * Value of fence to be signaled
             */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to LwSciSyncFence. Valid if ::lwdaExternalSemaphoreHandleType
             * is of type ::lwdaExternalSemaphoreHandleTypeLwSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } lwSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /*
             * Value of key to release the mutex with
             */
            unsigned long long key;
        } keyedMutex;
    } params;
    /**
     * Only when ::lwdaExternalSemaphoreSignalParams is used to
     * signal a ::lwdaExternalSemaphore_t of type
     * ::lwdaExternalSemaphoreHandleTypeLwSciSync, the valid flag is 
     * ::lwdaExternalSemaphoreSignalSkipLwSciBufMemSync: which indicates
     * that while signaling the ::lwdaExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::lwdaExternalMemoryHandleTypeLwSciBuf.
     * For all other types of ::lwdaExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
};

/**
* External semaphore wait parameters(deprecated)
*/
struct __device_builtin__ lwdaExternalSemaphoreWaitParams_v1 {
    struct {
        /**
        * Parameters for fence objects
        */
        struct {
            /**
            * Value of fence to be waited on
            */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to LwSciSyncFence. Valid if ::lwdaExternalSemaphoreHandleType
             * is of type ::lwdaExternalSemaphoreHandleTypeLwSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } lwSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /**
             * Value of key to acquire the mutex with
             */
            unsigned long long key;
            /**
             * Timeout in milliseconds to wait to acquire the mutex
             */
            unsigned int timeoutMs;
        } keyedMutex;
    } params;
    /**
     * Only when ::lwdaExternalSemaphoreSignalParams is used to
     * signal a ::lwdaExternalSemaphore_t of type
     * ::lwdaExternalSemaphoreHandleTypeLwSciSync, the valid flag is 
     * ::lwdaExternalSemaphoreSignalSkipLwSciBufMemSync: which indicates
     * that while waiting for the ::lwdaExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::lwdaExternalMemoryHandleTypeLwSciBuf.
     * For all other types of ::lwdaExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
};
#endif

/**
 * External semaphore signal parameters, compatible with driver type
 */
struct __device_builtin__ lwdaExternalSemaphoreSignalParams{
    struct {
        /**
         * Parameters for fence objects
         */
        struct {
            /**
             * Value of fence to be signaled
             */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to LwSciSyncFence. Valid if ::lwdaExternalSemaphoreHandleType
             * is of type ::lwdaExternalSemaphoreHandleTypeLwSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } lwSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /*
             * Value of key to release the mutex with
             */
            unsigned long long key;
        } keyedMutex;
        unsigned int reserved[12];
    } params;
    /**
     * Only when ::lwdaExternalSemaphoreSignalParams is used to
     * signal a ::lwdaExternalSemaphore_t of type
     * ::lwdaExternalSemaphoreHandleTypeLwSciSync, the valid flag is 
     * ::lwdaExternalSemaphoreSignalSkipLwSciBufMemSync: which indicates
     * that while signaling the ::lwdaExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::lwdaExternalMemoryHandleTypeLwSciBuf.
     * For all other types of ::lwdaExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
    unsigned int reserved[16];
};

/**
 * External semaphore wait parameters, compatible with driver type
 */
struct __device_builtin__ lwdaExternalSemaphoreWaitParams {
    struct {
        /**
        * Parameters for fence objects
        */
        struct {
            /**
            * Value of fence to be waited on
            */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to LwSciSyncFence. Valid if ::lwdaExternalSemaphoreHandleType
             * is of type ::lwdaExternalSemaphoreHandleTypeLwSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } lwSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /**
             * Value of key to acquire the mutex with
             */
            unsigned long long key;
            /**
             * Timeout in milliseconds to wait to acquire the mutex
             */
            unsigned int timeoutMs;
        } keyedMutex;
        unsigned int reserved[10];
    } params;
    /**
     * Only when ::lwdaExternalSemaphoreSignalParams is used to
     * signal a ::lwdaExternalSemaphore_t of type
     * ::lwdaExternalSemaphoreHandleTypeLwSciSync, the valid flag is 
     * ::lwdaExternalSemaphoreSignalSkipLwSciBufMemSync: which indicates
     * that while waiting for the ::lwdaExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::lwdaExternalMemoryHandleTypeLwSciBuf.
     * For all other types of ::lwdaExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
    unsigned int reserved[16];
};


/*******************************************************************************
*                                                                              *
*  SHORTHAND TYPE DEFINITION USED BY RUNTIME API                               *
*                                                                              *
*******************************************************************************/

/**
 * LWCA Error types
 */
typedef __device_builtin__ enum lwdaError lwdaError_t;

/**
 * LWCA stream
 */
typedef __device_builtin__ struct LWstream_st *lwdaStream_t;

/**
 * LWCA event types
 */
typedef __device_builtin__ struct LWevent_st *lwdaEvent_t;

/**
 * LWCA graphics resource types
 */
typedef __device_builtin__ struct lwdaGraphicsResource *lwdaGraphicsResource_t;

/**
 * LWCA output file modes
 */
typedef __device_builtin__ enum lwdaOutputMode lwdaOutputMode_t;

/**
 * LWCA external memory
 */
typedef __device_builtin__ struct LWexternalMemory_st *lwdaExternalMemory_t;

/**
 * LWCA external semaphore
 */
typedef __device_builtin__ struct LWexternalSemaphore_st *lwdaExternalSemaphore_t;

/**
 * LWCA graph
 */
typedef __device_builtin__ struct LWgraph_st *lwdaGraph_t;

/**
 * LWCA graph node.
 */
typedef __device_builtin__ struct LWgraphNode_st *lwdaGraphNode_t;

/**
 * LWCA user object for graphs
 */
typedef __device_builtin__ struct LWuserObject_st *lwdaUserObject_t;

/**
 * LWCA function
 */
typedef __device_builtin__ struct LWfunc_st *lwdaFunction_t;

/**
 * LWCA memory pool
 */
typedef __device_builtin__ struct LWmemPoolHandle_st *lwdaMemPool_t;

/**
 * LWCA cooperative group scope
 */
enum __device_builtin__ lwdaCGScope {
    lwdaCGScopeIlwalid   = 0, /**< Invalid cooperative group scope */
    lwdaCGScopeGrid      = 1, /**< Scope represented by a grid_group */
    lwdaCGScopeMultiGrid = 2  /**< Scope represented by a multi_grid_group */
};

/**
 * LWCA launch parameters
 */
struct __device_builtin__ lwdaLaunchParams
{
    void *func;          /**< Device function symbol */
    dim3 gridDim;        /**< Grid dimentions */
    dim3 blockDim;       /**< Block dimentions */
    void **args;         /**< Arguments */
    size_t sharedMem;    /**< Shared memory */
    lwdaStream_t stream; /**< Stream identifier */
};

/**
 * LWCA GPU kernel node parameters
 */
struct __device_builtin__ lwdaKernelNodeParams {
    void* func;                     /**< Kernel to launch */
    dim3 gridDim;                   /**< Grid dimensions */
    dim3 blockDim;                  /**< Block dimensions */
    unsigned int sharedMemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
    void **kernelParams;            /**< Array of pointers to individual kernel arguments*/
    void **extra;                   /**< Pointer to kernel arguments in the "extra" format */
};

/**
 * External semaphore signal node parameters
 */
struct __device_builtin__ lwdaExternalSemaphoreSignalNodeParams {
    lwdaExternalSemaphore_t* extSemArray;                        /**< Array of external semaphore handles. */
    const struct lwdaExternalSemaphoreSignalParams* paramsArray; /**< Array of external semaphore signal parameters. */
    unsigned int numExtSems;                                     /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

/**
 * External semaphore wait node parameters
 */
struct __device_builtin__ lwdaExternalSemaphoreWaitNodeParams {
    lwdaExternalSemaphore_t* extSemArray;                      /**< Array of external semaphore handles. */
    const struct lwdaExternalSemaphoreWaitParams* paramsArray; /**< Array of external semaphore wait parameters. */
    unsigned int numExtSems;                                   /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

/**
* LWCA Graph node types
*/
enum __device_builtin__ lwdaGraphNodeType {
    lwdaGraphNodeTypeKernel      = 0x00, /**< GPU kernel node */
    lwdaGraphNodeTypeMemcpy      = 0x01, /**< Memcpy node */
    lwdaGraphNodeTypeMemset      = 0x02, /**< Memset node */
    lwdaGraphNodeTypeHost        = 0x03, /**< Host (exelwtable) node */
    lwdaGraphNodeTypeGraph       = 0x04, /**< Node which exelwtes an embedded graph */
    lwdaGraphNodeTypeEmpty       = 0x05, /**< Empty (no-op) node */
    lwdaGraphNodeTypeWaitEvent   = 0x06, /**< External event wait node */
    lwdaGraphNodeTypeEventRecord = 0x07, /**< External event record node */
    lwdaGraphNodeTypeCount
};

/**
 * LWCA exelwtable (launchable) graph
 */
typedef struct LWgraphExec_st* lwdaGraphExec_t;

/**
* LWCA Graph Update error types
*/
enum __device_builtin__ lwdaGraphExelwpdateResult {
    lwdaGraphExelwpdateSuccess                = 0x0, /**< The update succeeded */
    lwdaGraphExelwpdateError                  = 0x1, /**< The update failed for an unexpected reason which is described in the return value of the function */
    lwdaGraphExelwpdateErrorTopologyChanged   = 0x2, /**< The update failed because the topology changed */
    lwdaGraphExelwpdateErrorNodeTypeChanged   = 0x3, /**< The update failed because a node type changed */
    lwdaGraphExelwpdateErrorFunctionChanged   = 0x4, /**< The update failed because the function of a kernel node changed (LWCA driver < 11.2) */
    lwdaGraphExelwpdateErrorParametersChanged = 0x5, /**< The update failed because the parameters changed in a way that is not supported */
    lwdaGraphExelwpdateErrorNotSupported      = 0x6, /**< The update failed because something about the node is not supported */
    lwdaGraphExelwpdateErrorUnsupportedFunctionChange = 0x7 /**< The update failed because the function of a kernel node changed in an unsupported way */
};

/**
 * Flags to specify search options to be used with ::lwdaGetDriverEntryPoint
 * For more details see ::lwGetProcAddress
 */ 
enum __device_builtin__ lwdaGetDriverEntryPointFlags {
    lwdaEnableDefault                = 0x0, /**< Default search mode for driver symbols. */
    lwdaEnableLegacyStream           = 0x1, /**< Search for legacy versions of driver symbols. */
    lwdaEnablePerThreadDefaultStream = 0x2  /**< Search for per-thread versions of driver symbols. */
};

/**
 * LWCA Graph debug write options
 */
enum __device_builtin__ lwdaGraphDebugDotFlags {
    lwdaGraphDebugDotFlagsVerbose                  = 1<<0,  /** Output all debug data as if every debug flag is enabled */
    lwdaGraphDebugDotFlagsKernelNodeParams         = 1<<2,  /** Adds lwdaKernelNodeParams to output */
    lwdaGraphDebugDotFlagsMemcpyNodeParams         = 1<<3,  /** Adds lwdaMemcpy3DParms to output */
    lwdaGraphDebugDotFlagsMemsetNodeParams         = 1<<4,  /** Adds lwdaMemsetParams to output */
    lwdaGraphDebugDotFlagsHostNodeParams           = 1<<5,  /** Adds lwdaHostNodeParams to output */
    lwdaGraphDebugDotFlagsEventNodeParams          = 1<<6,  /** Adds lwdaEvent_t handle from record and wait nodes to output */
    lwdaGraphDebugDotFlagsExtSemasSignalNodeParams = 1<<7,  /** Adds lwdaExternalSemaphoreSignalNodeParams values to output */
    lwdaGraphDebugDotFlagsExtSemasWaitNodeParams   = 1<<8,  /** Adds lwdaExternalSemaphoreWaitNodeParams to output */
    lwdaGraphDebugDotFlagsKernelNodeAttributes     = 1<<9,  /** Adds lwdaKernelNodeAttrID values to output */
    lwdaGraphDebugDotFlagsHandles                  = 1<<10  /** Adds node handles and every kernel function handle to output */
};

/** @} */
/** @} */ /* END LWDART_TYPES */

#if defined(__UNDEF_LWDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DRIVER_TYPES_H__)
#undef __LWDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_LWDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DRIVER_TYPES_H__
#endif

#undef __LWDA_DEPRECATED

#endif /* !__DRIVER_TYPES_H__ */
