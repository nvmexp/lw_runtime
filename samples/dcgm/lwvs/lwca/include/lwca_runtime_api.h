/*
 * Copyright 1993-2017 LWPU Corporation.  All rights reserved.
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

#if !defined(__LWDA_RUNTIME_API_H__)
#define __LWDA_RUNTIME_API_H__

/**
 * \latexonly
 * \page sync_async API synchronization behavior
 *
 * \section memcpy_sync_async_behavior Memcpy
 * The API provides memcpy/memset functions in both synchronous and asynchronous forms,
 * the latter having an \e "Async" suffix. This is a misnomer as each function
 * may exhibit synchronous or asynchronous behavior depending on the arguments
 * passed to the function. In the reference documentation, each memcpy function is
 * categorized as \e synchronous or \e asynchronous, corresponding to the definitions
 * below.
 * 
 * \subsection MemcpySynchronousBehavior Synchronous
 * 
 * <ol>
 * <li> For transfers from pageable host memory to device memory, a stream sync is performed
 * before the copy is initiated. The function will return once the pageable
 * buffer has been copied to the staging memory for DMA transfer to device memory,
 * but the DMA to final destination may not have completed.
 * 
 * <li> For transfers from pinned host memory to device memory, the function is synchronous
 * with respect to the host.
 *
 * <li> For transfers from device to either pageable or pinned host memory, the function returns
 * only once the copy has completed.
 * 
 * <li> For transfers from device memory to device memory, no host-side synchronization is
 * performed.
 *
 * <li> For transfers from any host memory to any host memory, the function is fully
 * synchronous with respect to the host.
 * </ol>
 * 
 * \subsection MemcpyAsynchronousBehavior Asynchronous
 *
 * <ol>
 * <li> For transfers from device memory to pageable host memory, the function
 * will return only once the copy has completed.
 *
 * <li> For transfers from any host memory to any host memory, the function is fully
 * synchronous with respect to the host.
 * 
 * <li> For all other transfers, the function is fully asynchronous. If pageable
 * memory must first be staged to pinned memory, this will be handled
 * asynchronously with a worker thread.
 * </ol>
 *
 * \section memset_sync_async_behavior Memset
 * The lwdaMemset functions are asynchronous with respect to the host
 * except when the target memory is pinned host memory. The \e Async
 * versions are always asynchronous with respect to the host.
 *
 * \section kernel_launch_details Kernel Launches
 * Kernel launches are asynchronous with respect to the host. Details of
 * conlwrrent kernel exelwtion and data transfers can be found in the LWCA
 * Programmers Guide.
 *
 * \endlatexonly
 */

/**
 * There are two levels for the runtime API.
 *
 * The C API (<i>lwda_runtime_api.h</i>) is
 * a C-style interface that does not require compiling with \p lwcc.
 *
 * The \ref LWDART_HIGHLEVEL "C++ API" (<i>lwda_runtime.h</i>) is a
 * C++-style interface built on top of the C API. It wraps some of the
 * C API routines, using overloading, references and default arguments.
 * These wrappers can be used from C++ code and can be compiled with any C++
 * compiler. The C++ API also has some LWCA-specific wrappers that wrap
 * C API routines that deal with symbols, textures, and device functions.
 * These wrappers require the use of \p lwcc because they depend on code being
 * generated by the compiler. For example, the exelwtion configuration syntax
 * to ilwoke kernels is only available in source code compiled with \p lwcc.
 */

/** LWCA Runtime API Version */
#define LWDART_VERSION  9000

#include "host_defines.h"
#include "builtin_types.h"

#include "lwda_device_runtime_api.h"

#if defined(LWDA_API_PER_THREAD_DEFAULT_STREAM) || defined(__LWDA_API_VERSION_INTERNAL)
    #define __LWDART_API_PER_THREAD_DEFAULT_STREAM
    #define __LWDART_API_PTDS(api) api ## _ptds
    #define __LWDART_API_PTSZ(api) api ## _ptsz
#else
    #define __LWDART_API_PTDS(api) api
    #define __LWDART_API_PTSZ(api) api
#endif

#if defined(__LWDART_API_PER_THREAD_DEFAULT_STREAM)
    #define lwdaMemcpy                     __LWDART_API_PTDS(lwdaMemcpy)
    #define lwdaMemcpyToSymbol             __LWDART_API_PTDS(lwdaMemcpyToSymbol)
    #define lwdaMemcpyFromSymbol           __LWDART_API_PTDS(lwdaMemcpyFromSymbol)
    #define lwdaMemcpy2D                   __LWDART_API_PTDS(lwdaMemcpy2D)
    #define lwdaMemcpyToArray              __LWDART_API_PTDS(lwdaMemcpyToArray)
    #define lwdaMemcpy2DToArray            __LWDART_API_PTDS(lwdaMemcpy2DToArray)
    #define lwdaMemcpyFromArray            __LWDART_API_PTDS(lwdaMemcpyFromArray)
    #define lwdaMemcpy2DFromArray          __LWDART_API_PTDS(lwdaMemcpy2DFromArray)
    #define lwdaMemcpyArrayToArray         __LWDART_API_PTDS(lwdaMemcpyArrayToArray)
    #define lwdaMemcpy2DArrayToArray       __LWDART_API_PTDS(lwdaMemcpy2DArrayToArray)
    #define lwdaMemcpy3D                   __LWDART_API_PTDS(lwdaMemcpy3D)
    #define lwdaMemcpy3DPeer               __LWDART_API_PTDS(lwdaMemcpy3DPeer)
    #define lwdaMemset                     __LWDART_API_PTDS(lwdaMemset)
    #define lwdaMemset2D                   __LWDART_API_PTDS(lwdaMemset2D)
    #define lwdaMemset3D                   __LWDART_API_PTDS(lwdaMemset3D)
    #define lwdaMemcpyAsync                __LWDART_API_PTSZ(lwdaMemcpyAsync)
    #define lwdaMemcpyToSymbolAsync        __LWDART_API_PTSZ(lwdaMemcpyToSymbolAsync)
    #define lwdaMemcpyFromSymbolAsync      __LWDART_API_PTSZ(lwdaMemcpyFromSymbolAsync)
    #define lwdaMemcpy2DAsync              __LWDART_API_PTSZ(lwdaMemcpy2DAsync)
    #define lwdaMemcpyToArrayAsync         __LWDART_API_PTSZ(lwdaMemcpyToArrayAsync)
    #define lwdaMemcpy2DToArrayAsync       __LWDART_API_PTSZ(lwdaMemcpy2DToArrayAsync)
    #define lwdaMemcpyFromArrayAsync       __LWDART_API_PTSZ(lwdaMemcpyFromArrayAsync)
    #define lwdaMemcpy2DFromArrayAsync     __LWDART_API_PTSZ(lwdaMemcpy2DFromArrayAsync)
    #define lwdaMemcpy3DAsync              __LWDART_API_PTSZ(lwdaMemcpy3DAsync)
    #define lwdaMemcpy3DPeerAsync          __LWDART_API_PTSZ(lwdaMemcpy3DPeerAsync)
    #define lwdaMemsetAsync                __LWDART_API_PTSZ(lwdaMemsetAsync)
    #define lwdaMemset2DAsync              __LWDART_API_PTSZ(lwdaMemset2DAsync)
    #define lwdaMemset3DAsync              __LWDART_API_PTSZ(lwdaMemset3DAsync)
    #define lwdaStreamQuery                __LWDART_API_PTSZ(lwdaStreamQuery)
    #define lwdaStreamGetFlags             __LWDART_API_PTSZ(lwdaStreamGetFlags)
    #define lwdaStreamGetPriority          __LWDART_API_PTSZ(lwdaStreamGetPriority)
    #define lwdaEventRecord                __LWDART_API_PTSZ(lwdaEventRecord)
    #define lwdaStreamWaitEvent            __LWDART_API_PTSZ(lwdaStreamWaitEvent)
    #define lwdaStreamAddCallback          __LWDART_API_PTSZ(lwdaStreamAddCallback)
    #define lwdaStreamAttachMemAsync       __LWDART_API_PTSZ(lwdaStreamAttachMemAsync)
    #define lwdaStreamSynchronize          __LWDART_API_PTSZ(lwdaStreamSynchronize)
    #define lwdaLaunch                     __LWDART_API_PTSZ(lwdaLaunch)
    #define lwdaLaunchKernel               __LWDART_API_PTSZ(lwdaLaunchKernel)
    #define lwdaMemPrefetchAsync           __LWDART_API_PTSZ(lwdaMemPrefetchAsync)
    #define lwdaLaunchCooperativeKernel    __LWDART_API_PTSZ(lwdaLaunchCooperativeKernel)
#endif

/** \cond impl_private */
#if !defined(__dv)

#if defined(__cplusplus)

#define __dv(v) \
        = v

#else /* __cplusplus */

#define __dv(v)

#endif /* __cplusplus */

#endif /* !__dv */
/** \endcond impl_private */

#if (!defined(__LWDA_ARCH__) || (__LWDA_ARCH__ >= 350))   /** Visible to SM>=3.5 and "__host__ __device__" only **/

#define LWDART_DEVICE __device__ 

#else

#define LWDART_DEVICE

#endif /** LWDART_DEVICE */

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * \defgroup LWDART_DEVICE Device Management
 *
 * ___MANBRIEF___ device management functions of the LWCA runtime API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the device management functions of the LWCA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Destroy all allocations and reset all state on the current device
 * in the current process.
 *
 * Explicitly destroys and cleans up all resources associated with the current
 * device in the current process.  Any subsequent API call to this device will 
 * reinitialize the device.
 *
 * Note that this function will reset the device immediately.  It is the caller's
 * responsibility to ensure that the device is not being accessed by any 
 * other host threads from the process when this function is called.
 *
 * \return
 * ::lwdaSuccess
 * \notefnerr
 *
 * \sa ::lwdaDeviceSynchronize
 */
extern __host__ lwdaError_t LWDARTAPI lwdaDeviceReset(void);

/**
 * \brief Wait for compute device to finish
 *
 * Blocks until the device has completed all preceding requested tasks.
 * ::lwdaDeviceSynchronize() returns an error if one of the preceding tasks
 * has failed. If the ::lwdaDeviceScheduleBlockingSync flag was set for 
 * this device, the host thread will block until the device has finished 
 * its work.
 *
 * \return
 * ::lwdaSuccess
 * \notefnerr
 *
 * \sa
 * ::lwdaDeviceReset,
 * ::lwCtxSynchronize
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaDeviceSynchronize(void);

/**
 * \brief Set resource limits
 *
 * Setting \p limit to \p value is a request by the application to update
 * the current limit maintained by the device.  The driver is free to
 * modify the requested value to meet h/w requirements (this could be
 * clamping to minimum or maximum values, rounding up to nearest element
 * size, etc).  The application can use ::lwdaDeviceGetLimit() to find out
 * exactly what the limit has been set to.
 *
 * Setting each ::lwdaLimit has its own specific restrictions, so each is
 * dislwssed here.
 *
 * - ::lwdaLimitStackSize controls the stack size in bytes of each GPU thread.
 *
 * - ::lwdaLimitPrintfFifoSize controls the size in bytes of the shared FIFO
 *   used by the ::printf() and ::fprintf() device system calls. Setting
 *   ::lwdaLimitPrintfFifoSize must not be performed after launching any kernel
 *   that uses the ::printf() or ::fprintf() device system calls - in such case
 *   ::lwdaErrorIlwalidValue will be returned.
 *
 * - ::lwdaLimitMallocHeapSize controls the size in bytes of the heap used by
 *   the ::malloc() and ::free() device system calls. Setting
 *   ::lwdaLimitMallocHeapSize must not be performed after launching any kernel
 *   that uses the ::malloc() or ::free() device system calls - in such case
 *   ::lwdaErrorIlwalidValue will be returned.
 *
 * - ::lwdaLimitDevRuntimeSyncDepth controls the maximum nesting depth of a
 *   grid at which a thread can safely call ::lwdaDeviceSynchronize(). Setting
 *   this limit must be performed before any launch of a kernel that uses the
 *   device runtime and calls ::lwdaDeviceSynchronize() above the default sync
 *   depth, two levels of grids. Calls to ::lwdaDeviceSynchronize() will fail
 *   with error code ::lwdaErrorSyncDepthExceeded if the limitation is
 *   violated. This limit can be set smaller than the default or up the maximum
 *   launch depth of 24. When setting this limit, keep in mind that additional
 *   levels of sync depth require the runtime to reserve large amounts of
 *   device memory which can no longer be used for user allocations. If these
 *   reservations of device memory fail, ::lwdaDeviceSetLimit will return
 *   ::lwdaErrorMemoryAllocation, and the limit can be reset to a lower value.
 *   This limit is only applicable to devices of compute capability 3.5 and
 *   higher. Attempting to set this limit on devices of compute capability less
 *   than 3.5 will result in the error ::lwdaErrorUnsupportedLimit being
 *   returned.
 *
 * - ::lwdaLimitDevRuntimePendingLaunchCount controls the maximum number of
 *   outstanding device runtime launches that can be made from the current
 *   device. A grid is outstanding from the point of launch up until the grid
 *   is known to have been completed. Device runtime launches which violate 
 *   this limitation fail and return ::lwdaErrorLaunchPendingCountExceeded when
 *   ::lwdaGetLastError() is called after launch. If more pending launches than
 *   the default (2048 launches) are needed for a module using the device
 *   runtime, this limit can be increased. Keep in mind that being able to
 *   sustain additional pending launches will require the runtime to reserve
 *   larger amounts of device memory upfront which can no longer be used for
 *   allocations. If these reservations fail, ::lwdaDeviceSetLimit will return
 *   ::lwdaErrorMemoryAllocation, and the limit can be reset to a lower value.
 *   This limit is only applicable to devices of compute capability 3.5 and
 *   higher. Attempting to set this limit on devices of compute capability less
 *   than 3.5 will result in the error ::lwdaErrorUnsupportedLimit being
 *   returned. 
 *
 * \param limit - Limit to set
 * \param value - Size of limit
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorUnsupportedLimit,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa
 * ::lwdaDeviceGetLimit,
 * ::lwCtxSetLimit
 */
extern __host__ lwdaError_t LWDARTAPI lwdaDeviceSetLimit(enum lwdaLimit limit, size_t value);

/**
 * \brief Returns resource limits
 *
 * Returns in \p *pValue the current size of \p limit.  The supported
 * ::lwdaLimit values are:
 * - ::lwdaLimitStackSize: stack size in bytes of each GPU thread;
 * - ::lwdaLimitPrintfFifoSize: size in bytes of the shared FIFO used by the
 *   ::printf() and ::fprintf() device system calls.
 * - ::lwdaLimitMallocHeapSize: size in bytes of the heap used by the
 *   ::malloc() and ::free() device system calls;
 * - ::lwdaLimitDevRuntimeSyncDepth: maximum grid depth at which a
 *   thread can isssue the device runtime call ::lwdaDeviceSynchronize()
 *   to wait on child grid launches to complete.
 * - ::lwdaLimitDevRuntimePendingLaunchCount: maximum number of outstanding
 *   device runtime launches.
 *
 * \param limit  - Limit to query
 * \param pValue - Returned size of the limit
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorUnsupportedLimit,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \sa
 * ::lwdaDeviceSetLimit,
 * ::lwCtxGetLimit
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaDeviceGetLimit(size_t *pValue, enum lwdaLimit limit);

/**
 * \brief Returns the preferred cache configuration for the current device.
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this returns through \p pCacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute functions.
 *
 * This will return a \p pCacheConfig of ::lwdaFuncCachePreferNone on devices
 * where the size of the L1 cache and shared memory are fixed.
 *
 * The supported cache configurations are:
 * - ::lwdaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::lwdaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::lwdaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 * - ::lwdaFuncCachePreferEqual: prefer equal size L1 cache and shared memory
 *
 * \param pCacheConfig - Returned cache configuration
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorInitializationError
 * \notefnerr
 *
 * \sa lwdaDeviceSetCacheConfig,
 * \ref ::lwdaFuncSetCacheConfig(const void*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C API)",
 * \ref ::lwdaFuncSetCacheConfig(T*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C++ API)",
 * ::lwCtxGetCacheConfig
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaDeviceGetCacheConfig(enum lwdaFuncCache *pCacheConfig);

/**
 * \brief Returns numerical values that correspond to the least and
 * greatest stream priorities.
 *
 * Returns in \p *leastPriority and \p *greatestPriority the numerical values that correspond
 * to the least and greatest stream priorities respectively. Stream priorities
 * follow a convention where lower numbers imply greater priorities. The range of
 * meaningful stream priorities is given by [\p *greatestPriority, \p *leastPriority].
 * If the user attempts to create a stream with a priority value that is
 * outside the the meaningful range as specified by this API, the priority is
 * automatically clamped down or up to either \p *leastPriority or \p *greatestPriority
 * respectively. See ::lwdaStreamCreateWithPriority for details on creating a
 * priority stream.
 * A NULL may be passed in for \p *leastPriority or \p *greatestPriority if the value
 * is not desired.
 *
 * This function will return '0' in both \p *leastPriority and \p *greatestPriority if
 * the current context's device does not support stream priorities
 * (see ::lwdaDeviceGetAttribute).
 *
 * \param leastPriority    - Pointer to an int in which the numerical value for least
 *                           stream priority is returned
 * \param greatestPriority - Pointer to an int in which the numerical value for greatest
 *                           stream priority is returned
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorInitializationError
 * \notefnerr
 *
 * \sa ::lwdaStreamCreateWithPriority,
 * ::lwdaStreamGetPriority,
 * ::lwCtxGetStreamPriorityRange
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority);

/**
 * \brief Sets the preferred cache configuration for the current device.
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p cacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute the function. Any
 * function preference set via
 * \ref ::lwdaFuncSetCacheConfig(const void*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C API)"
 * or
 * \ref ::lwdaFuncSetCacheConfig(T*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C++ API)"
 * will be preferred over this device-wide setting. Setting the device-wide
 * cache configuration to ::lwdaFuncCachePreferNone will cause subsequent
 * kernel launches to prefer to not change the cache configuration unless
 * required to launch the kernel.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 * - ::lwdaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::lwdaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::lwdaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 * - ::lwdaFuncCachePreferEqual: prefer equal size L1 cache and shared memory
 *
 * \param cacheConfig - Requested cache configuration
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorInitializationError
 * \notefnerr
 *
 * \sa ::lwdaDeviceGetCacheConfig,
 * \ref ::lwdaFuncSetCacheConfig(const void*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C API)",
 * \ref ::lwdaFuncSetCacheConfig(T*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C++ API)",
 * ::lwCtxSetCacheConfig
 */
extern __host__ lwdaError_t LWDARTAPI lwdaDeviceSetCacheConfig(enum lwdaFuncCache cacheConfig);

/**
 * \brief Returns the shared memory configuration for the current device.
 *
 * This function will return in \p pConfig the current size of shared memory banks
 * on the current device. On devices with configurable shared memory banks, 
 * ::lwdaDeviceSetSharedMemConfig can be used to change this setting, so that all 
 * subsequent kernel launches will by default use the new bank size. When 
 * ::lwdaDeviceGetSharedMemConfig is called on devices without configurable shared 
 * memory, it will return the fixed bank size of the hardware.
 *
 * The returned bank configurations can be either:
 * - ::lwdaSharedMemBankSizeFourByte - shared memory bank width is four bytes.
 * - ::lwdaSharedMemBankSizeEightByte - shared memory bank width is eight bytes.
 *
 * \param pConfig - Returned cache configuration
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorInitializationError
 * \notefnerr
 *
 * \sa ::lwdaDeviceSetCacheConfig,
 * ::lwdaDeviceGetCacheConfig,
 * ::lwdaDeviceSetSharedMemConfig,
 * ::lwdaFuncSetCacheConfig,
 * ::lwCtxGetSharedMemConfig
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaDeviceGetSharedMemConfig(enum lwdaSharedMemConfig *pConfig);

/**
 * \brief Sets the shared memory configuration for the current device.
 *
 * On devices with configurable shared memory banks, this function will set
 * the shared memory bank size which is used for all subsequent kernel launches.
 * Any per-function setting of shared memory set via ::lwdaFuncSetSharedMemConfig
 * will override the device wide setting.
 *
 * Changing the shared memory configuration between launches may introduce
 * a device side synchronization point.
 *
 * Changing the shared memory bank size will not increase shared memory usage
 * or affect oclwpancy of kernels, but may have major effects on performance. 
 * Larger bank sizes will allow for greater potential bandwidth to shared memory,
 * but will change what kinds of accesses to shared memory will result in bank 
 * conflicts.
 *
 * This function will do nothing on devices with fixed shared memory bank size.
 *
 * The supported bank configurations are:
 * - ::lwdaSharedMemBankSizeDefault: set bank width the device default (lwrrently,
 *   four bytes)
 * - ::lwdaSharedMemBankSizeFourByte: set shared memory bank width to be four bytes
 *   natively.
 * - ::lwdaSharedMemBankSizeEightByte: set shared memory bank width to be eight 
 *   bytes natively.
 *
 * \param config - Requested cache configuration
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorInitializationError
 * \notefnerr
 *
 * \sa ::lwdaDeviceSetCacheConfig,
 * ::lwdaDeviceGetCacheConfig,
 * ::lwdaDeviceGetSharedMemConfig,
 * ::lwdaFuncSetCacheConfig,
 * ::lwCtxSetSharedMemConfig
 */
extern __host__ lwdaError_t LWDARTAPI lwdaDeviceSetSharedMemConfig(enum lwdaSharedMemConfig config);

/**
 * \brief Returns a handle to a compute device
 *
 * Returns in \p *device a device ordinal given a PCI bus ID string.
 *
 * \param device   - Returned device ordinal
 *
 * \param pciBusId - String in one of the following forms: 
 * [domain]:[bus]:[device].[function]
 * [domain]:[bus]:[device]
 * [bus]:[device].[function]
 * where \p domain, \p bus, \p device, and \p function are all hexadecimal values
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidDevice
 * \notefnerr
 *
 * \sa
 * ::lwdaDeviceGetPCIBusId,
 * ::lwDeviceGetByPCIBusId
 */
extern __host__ lwdaError_t LWDARTAPI lwdaDeviceGetByPCIBusId(int *device, const char *pciBusId);

/**
 * \brief Returns a PCI Bus Id string for the device
 *
 * Returns an ASCII string identifying the device \p dev in the NULL-terminated
 * string pointed to by \p pciBusId. \p len specifies the maximum length of the
 * string that may be returned.
 *
 * \param pciBusId - Returned identifier string for the device in the following format
 * [domain]:[bus]:[device].[function]
 * where \p domain, \p bus, \p device, and \p function are all hexadecimal values.
 * pciBusId should be large enough to store 13 characters including the NULL-terminator.
 *
 * \param len      - Maximum length of string to store in \p name
 *
 * \param device   - Device to get identifier string for
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidDevice
 * \notefnerr
 *
 * \sa
 * ::lwdaDeviceGetByPCIBusId,
 * ::lwDeviceGetPCIBusId
 */
extern __host__ lwdaError_t LWDARTAPI lwdaDeviceGetPCIBusId(char *pciBusId, int len, int device);

/**
 * \brief Gets an interprocess handle for a previously allocated event
 *
 * Takes as input a previously allocated event. This event must have been 
 * created with the ::lwdaEventInterprocess and ::lwdaEventDisableTiming
 * flags set. This opaque handle may be copied into other processes and
 * opened with ::lwdaIpcOpenEventHandle to allow efficient hardware
 * synchronization between GPU work in different processes.
 *
 * After the event has been been opened in the importing process, 
 * ::lwdaEventRecord, ::lwdaEventSynchronize, ::lwdaStreamWaitEvent and 
 * ::lwdaEventQuery may be used in either process. Performing operations 
 * on the imported event after the exported event has been freed 
 * with ::lwdaEventDestroy will result in undefined behavior.
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems. IPC functionality is not supported
 * on CheetAh platforms.
 *
 * \param handle - Pointer to a user allocated lwdaIpcEventHandle
 *                    in which to return the opaque event handle
 * \param event   - Event allocated with ::lwdaEventInterprocess and 
 *                    ::lwdaEventDisableTiming flags.
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorMemoryAllocation,
 * ::lwdaErrorMapBufferObjectFailed,
 * ::lwdaErrorNotSupported
 *
 * \sa
 * ::lwdaEventCreate,
 * ::lwdaEventDestroy,
 * ::lwdaEventSynchronize,
 * ::lwdaEventQuery,
 * ::lwdaStreamWaitEvent,
 * ::lwdaIpcOpenEventHandle,
 * ::lwdaIpcGetMemHandle,
 * ::lwdaIpcOpenMemHandle,
 * ::lwdaIpcCloseMemHandle,
 * ::lwIpcGetEventHandle
 */
extern __host__ lwdaError_t LWDARTAPI lwdaIpcGetEventHandle(lwdaIpcEventHandle_t *handle, lwdaEvent_t event);

/**
 * \brief Opens an interprocess event handle for use in the current process
 *
 * Opens an interprocess event handle exported from another process with 
 * ::lwdaIpcGetEventHandle. This function returns a ::lwdaEvent_t that behaves like 
 * a locally created event with the ::lwdaEventDisableTiming flag specified. 
 * This event must be freed with ::lwdaEventDestroy.
 *
 * Performing operations on the imported event after the exported event has 
 * been freed with ::lwdaEventDestroy will result in undefined behavior.
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems. IPC functionality is not supported
 * on CheetAh platforms.
 *
 * \param event - Returns the imported event
 * \param handle  - Interprocess handle to open
 *
 * \returns
 * ::lwdaSuccess,
 * ::lwdaErrorMapBufferObjectFailed,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorNotSupported
 *
  * \sa
 * ::lwdaEventCreate,
 * ::lwdaEventDestroy,
 * ::lwdaEventSynchronize,
 * ::lwdaEventQuery,
 * ::lwdaStreamWaitEvent,
 * ::lwdaIpcGetEventHandle,
 * ::lwdaIpcGetMemHandle,
 * ::lwdaIpcOpenMemHandle,
 * ::lwdaIpcCloseMemHandle,
 * ::lwIpcOpenEventHandle
 */
extern __host__ lwdaError_t LWDARTAPI lwdaIpcOpenEventHandle(lwdaEvent_t *event, lwdaIpcEventHandle_t handle);


/**
 * \brief Gets an interprocess memory handle for an existing device memory
 *          allocation
 *
 * Takes a pointer to the base of an existing device memory allocation created 
 * with ::lwdaMalloc and exports it for use in another process. This is a 
 * lightweight operation and may be called multiple times on an allocation
 * without adverse effects. 
 *
 * If a region of memory is freed with ::lwdaFree and a subsequent call
 * to ::lwdaMalloc returns memory with the same device address,
 * ::lwdaIpcGetMemHandle will return a unique handle for the
 * new memory. 
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems. IPC functionality is not supported
 * on CheetAh platforms.
 *
 * \param handle - Pointer to user allocated ::lwdaIpcMemHandle to return
 *                    the handle in.
 * \param devPtr - Base pointer to previously allocated device memory 
 *
 * \returns
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorMemoryAllocation,
 * ::lwdaErrorMapBufferObjectFailed,
 * ::lwdaErrorNotSupported
 *
 * \sa
 * ::lwdaMalloc,
 * ::lwdaFree,
 * ::lwdaIpcGetEventHandle,
 * ::lwdaIpcOpenEventHandle,
 * ::lwdaIpcOpenMemHandle,
 * ::lwdaIpcCloseMemHandle,
 * ::lwIpcGetMemHandle
 */
extern __host__ lwdaError_t LWDARTAPI lwdaIpcGetMemHandle(lwdaIpcMemHandle_t *handle, void *devPtr);

/**
 * \brief Opens an interprocess memory handle exported from another process
 *          and returns a device pointer usable in the local process.
 *
 * Maps memory exported from another process with ::lwdaIpcGetMemHandle into
 * the current device address space. For contexts on different devices 
 * ::lwdaIpcOpenMemHandle can attempt to enable peer access between the
 * devices as if the user called ::lwdaDeviceEnablePeerAccess. This behavior is 
 * controlled by the ::lwdaIpcMemLazyEnablePeerAccess flag. 
 * ::lwdaDeviceCanAccessPeer can determine if a mapping is possible.
 *
 * Contexts that may open ::lwdaIpcMemHandles are restricted in the following way.
 * ::lwdaIpcMemHandles from each device in a given process may only be opened 
 * by one context per device per other process.
 *
 * Memory returned from ::lwdaIpcOpenMemHandle must be freed with
 * ::lwdaIpcCloseMemHandle.
 *
 * Calling ::lwdaFree on an exported memory region before calling
 * ::lwdaIpcCloseMemHandle in the importing context will result in undefined
 * behavior.
 * 
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems. IPC functionality is not supported
 * on CheetAh platforms.
 *
 * \param devPtr - Returned device pointer
 * \param handle - ::lwdaIpcMemHandle to open
 * \param flags  - Flags for this operation. Must be specified as ::lwdaIpcMemLazyEnablePeerAccess
 *
 * \returns
 * ::lwdaSuccess,
 * ::lwdaErrorMapBufferObjectFailed,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorTooManyPeers,
 * ::lwdaErrorNotSupported
 *
 * \note No guarantees are made about the address returned in \p *devPtr.  
 * In particular, multiple processes may not receive the same address for the same \p handle.
 *
 * \sa
 * ::lwdaMalloc,
 * ::lwdaFree,
 * ::lwdaIpcGetEventHandle,
 * ::lwdaIpcOpenEventHandle,
 * ::lwdaIpcGetMemHandle,
 * ::lwdaIpcCloseMemHandle,
 * ::lwdaDeviceEnablePeerAccess,
 * ::lwdaDeviceCanAccessPeer,
 * ::lwIpcOpenMemHandle
 */
extern __host__ lwdaError_t LWDARTAPI lwdaIpcOpenMemHandle(void **devPtr, lwdaIpcMemHandle_t handle, unsigned int flags);

/**
 * \brief Close memory mapped with lwdaIpcOpenMemHandle
 * 
 * Unmaps memory returnd by ::lwdaIpcOpenMemHandle. The original allocation
 * in the exporting process as well as imported mappings in other processes
 * will be unaffected.
 *
 * Any resources used to enable peer access will be freed if this is the
 * last mapping using them.
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems. IPC functionality is not supported
 * on CheetAh platforms.
 *
 * \param devPtr - Device pointer returned by ::lwdaIpcOpenMemHandle
 * 
 * \returns
 * ::lwdaSuccess,
 * ::lwdaErrorMapBufferObjectFailed,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorNotSupported
 *
 * \sa
 * ::lwdaMalloc,
 * ::lwdaFree,
 * ::lwdaIpcGetEventHandle,
 * ::lwdaIpcOpenEventHandle,
 * ::lwdaIpcGetMemHandle,
 * ::lwdaIpcOpenMemHandle,
 * ::lwIpcCloseMemHandle
 */
extern __host__ lwdaError_t LWDARTAPI lwdaIpcCloseMemHandle(void *devPtr);

/** @} */ /* END LWDART_DEVICE */

/**
 * \defgroup LWDART_THREAD_DEPRECATED Thread Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated thread management functions of the LWCA runtime
 * API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes deprecated thread management functions of the LWCA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Exit and clean up from LWCA launches
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::lwdaDeviceReset(), which should be used
 * instead.
 *
 * Explicitly destroys all cleans up all resources associated with the current
 * device in the current process.  Any subsequent API call to this device will 
 * reinitialize the device.  
 *
 * Note that this function will reset the device immediately.  It is the caller's
 * responsibility to ensure that the device is not being accessed by any 
 * other host threads from the process when this function is called.
 *
 * \return
 * ::lwdaSuccess
 * \notefnerr
 *
 * \sa ::lwdaDeviceReset
 */
extern __host__ lwdaError_t LWDARTAPI lwdaThreadExit(void);

/**
 * \brief Wait for compute device to finish
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is similar to the 
 * non-deprecated function ::lwdaDeviceSynchronize(), which should be used
 * instead.
 *
 * Blocks until the device has completed all preceding requested tasks.
 * ::lwdaThreadSynchronize() returns an error if one of the preceding tasks
 * has failed. If the ::lwdaDeviceScheduleBlockingSync flag was set for 
 * this device, the host thread will block until the device has finished 
 * its work.
 *
 * \return
 * ::lwdaSuccess
 * \notefnerr
 *
 * \sa ::lwdaDeviceSynchronize
 */
extern __host__ lwdaError_t LWDARTAPI lwdaThreadSynchronize(void);

/**
 * \brief Set resource limits
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::lwdaDeviceSetLimit(), which should be used
 * instead.
 *
 * Setting \p limit to \p value is a request by the application to update
 * the current limit maintained by the device.  The driver is free to
 * modify the requested value to meet h/w requirements (this could be
 * clamping to minimum or maximum values, rounding up to nearest element
 * size, etc).  The application can use ::lwdaThreadGetLimit() to find out
 * exactly what the limit has been set to.
 *
 * Setting each ::lwdaLimit has its own specific restrictions, so each is
 * dislwssed here.
 *
 * - ::lwdaLimitStackSize controls the stack size of each GPU thread.
 *
 * - ::lwdaLimitPrintfFifoSize controls the size of the shared FIFO
 *   used by the ::printf() and ::fprintf() device system calls.
 *   Setting ::lwdaLimitPrintfFifoSize must be performed before
 *   launching any kernel that uses the ::printf() or ::fprintf() device
 *   system calls, otherwise ::lwdaErrorIlwalidValue will be returned.
 *
 * - ::lwdaLimitMallocHeapSize controls the size of the heap used
 *   by the ::malloc() and ::free() device system calls.  Setting
 *   ::lwdaLimitMallocHeapSize must be performed before launching
 *   any kernel that uses the ::malloc() or ::free() device system calls,
 *   otherwise ::lwdaErrorIlwalidValue will be returned.
 *
 * \param limit - Limit to set
 * \param value - Size in bytes of limit
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorUnsupportedLimit,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \sa ::lwdaDeviceSetLimit
 */
extern __host__ lwdaError_t LWDARTAPI lwdaThreadSetLimit(enum lwdaLimit limit, size_t value);

/**
 * \brief Returns resource limits
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::lwdaDeviceGetLimit(), which should be used
 * instead.
 *
 * Returns in \p *pValue the current size of \p limit.  The supported
 * ::lwdaLimit values are:
 * - ::lwdaLimitStackSize: stack size of each GPU thread;
 * - ::lwdaLimitPrintfFifoSize: size of the shared FIFO used by the
 *   ::printf() and ::fprintf() device system calls.
 * - ::lwdaLimitMallocHeapSize: size of the heap used by the
 *   ::malloc() and ::free() device system calls;
 *
 * \param limit  - Limit to query
 * \param pValue - Returned size in bytes of limit
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorUnsupportedLimit,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \sa ::lwdaDeviceGetLimit
 */
extern __host__ lwdaError_t LWDARTAPI lwdaThreadGetLimit(size_t *pValue, enum lwdaLimit limit);

/**
 * \brief Returns the preferred cache configuration for the current device.
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::lwdaDeviceGetCacheConfig(), which should be 
 * used instead.
 * 
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this returns through \p pCacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute functions.
 *
 * This will return a \p pCacheConfig of ::lwdaFuncCachePreferNone on devices
 * where the size of the L1 cache and shared memory are fixed.
 *
 * The supported cache configurations are:
 * - ::lwdaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::lwdaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::lwdaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 *
 * \param pCacheConfig - Returned cache configuration
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorInitializationError
 * \notefnerr
 *
 * \sa ::lwdaDeviceGetCacheConfig
 */
extern __host__ lwdaError_t LWDARTAPI lwdaThreadGetCacheConfig(enum lwdaFuncCache *pCacheConfig);

/**
 * \brief Sets the preferred cache configuration for the current device.
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::lwdaDeviceSetCacheConfig(), which should be 
 * used instead.
 * 
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p cacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute the function. Any
 * function preference set via
 * \ref ::lwdaFuncSetCacheConfig(const void*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C API)"
 * or
 * \ref ::lwdaFuncSetCacheConfig(T*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C++ API)"
 * will be preferred over this device-wide setting. Setting the device-wide
 * cache configuration to ::lwdaFuncCachePreferNone will cause subsequent
 * kernel launches to prefer to not change the cache configuration unless
 * required to launch the kernel.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 * - ::lwdaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::lwdaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::lwdaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 *
 * \param cacheConfig - Requested cache configuration
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorInitializationError
 * \notefnerr
 *
 * \sa ::lwdaDeviceSetCacheConfig
 */
extern __host__ lwdaError_t LWDARTAPI lwdaThreadSetCacheConfig(enum lwdaFuncCache cacheConfig);

/** @} */ /* END LWDART_THREAD_DEPRECATED */

/**
 * \defgroup LWDART_ERROR Error Handling
 *
 * ___MANBRIEF___ error handling functions of the LWCA runtime API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the error handling functions of the LWCA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Returns the last error from a runtime call
 *
 * Returns the last error that has been produced by any of the runtime calls
 * in the same host thread and resets it to ::lwdaSuccess.
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorMissingConfiguration,
 * ::lwdaErrorMemoryAllocation,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorLaunchFailure,
 * ::lwdaErrorLaunchTimeout,
 * ::lwdaErrorLaunchOutOfResources,
 * ::lwdaErrorIlwalidDeviceFunction,
 * ::lwdaErrorIlwalidConfiguration,
 * ::lwdaErrorIlwalidDevice,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidPitchValue,
 * ::lwdaErrorIlwalidSymbol,
 * ::lwdaErrorUnmapBufferObjectFailed,
 * ::lwdaErrorIlwalidDevicePointer,
 * ::lwdaErrorIlwalidTexture,
 * ::lwdaErrorIlwalidTextureBinding,
 * ::lwdaErrorIlwalidChannelDescriptor,
 * ::lwdaErrorIlwalidMemcpyDirection,
 * ::lwdaErrorIlwalidFilterSetting,
 * ::lwdaErrorIlwalidNormSetting,
 * ::lwdaErrorUnknown,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorInsufficientDriver,
 * ::lwdaErrorSetOnActiveProcess,
 * ::lwdaErrorStartupFailure,
 * ::lwdaErrorIlwalidPtx,
 * ::lwdaErrorNoKernelImageForDevice,
 * ::lwdaErrorJitCompilerNotFound
 * \notefnerr
 *
 * \sa ::lwdaPeekAtLastError, ::lwdaGetErrorName, ::lwdaGetErrorString, ::lwdaError
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaGetLastError(void);

/**
 * \brief Returns the last error from a runtime call
 *
 * Returns the last error that has been produced by any of the runtime calls
 * in the same host thread. Note that this call does not reset the error to
 * ::lwdaSuccess like ::lwdaGetLastError().
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorMissingConfiguration,
 * ::lwdaErrorMemoryAllocation,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorLaunchFailure,
 * ::lwdaErrorLaunchTimeout,
 * ::lwdaErrorLaunchOutOfResources,
 * ::lwdaErrorIlwalidDeviceFunction,
 * ::lwdaErrorIlwalidConfiguration,
 * ::lwdaErrorIlwalidDevice,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidPitchValue,
 * ::lwdaErrorIlwalidSymbol,
 * ::lwdaErrorUnmapBufferObjectFailed,
 * ::lwdaErrorIlwalidDevicePointer,
 * ::lwdaErrorIlwalidTexture,
 * ::lwdaErrorIlwalidTextureBinding,
 * ::lwdaErrorIlwalidChannelDescriptor,
 * ::lwdaErrorIlwalidMemcpyDirection,
 * ::lwdaErrorIlwalidFilterSetting,
 * ::lwdaErrorIlwalidNormSetting,
 * ::lwdaErrorUnknown,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorInsufficientDriver,
 * ::lwdaErrorSetOnActiveProcess,
 * ::lwdaErrorStartupFailure,
 * ::lwdaErrorIlwalidPtx,
 * ::lwdaErrorNoKernelImageForDevice,
 * ::lwdaErrorJitCompilerNotFound
 * \notefnerr
 *
 * \sa ::lwdaGetLastError, ::lwdaGetErrorName, ::lwdaGetErrorString, ::lwdaError
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaPeekAtLastError(void);

/**
 * \brief Returns the string representation of an error code enum name
 *
 * Returns a string containing the name of an error code in the enum.  If the error
 * code is not recognized, "unrecognized error code" is returned.
 *
 * \param error - Error code to colwert to string
 *
 * \return
 * \p char* pointer to a NULL-terminated string
 *
 * \sa ::lwdaGetErrorString, ::lwdaGetLastError, ::lwdaPeekAtLastError, ::lwdaError,
 * ::lwGetErrorName
 */
extern __host__ __lwdart_builtin__ const char* LWDARTAPI lwdaGetErrorName(lwdaError_t error);

/**
 * \brief Returns the description string for an error code
 *
 * Returns the description string for an error code.  If the error
 * code is not recognized, "unrecognized error code" is returned.
 *
 * \param error - Error code to colwert to string
 *
 * \return
 * \p char* pointer to a NULL-terminated string
 *
 * \sa ::lwdaGetErrorName, ::lwdaGetLastError, ::lwdaPeekAtLastError, ::lwdaError,
 * ::lwGetErrorString
 */
extern __host__ __lwdart_builtin__ const char* LWDARTAPI lwdaGetErrorString(lwdaError_t error);
/** @} */ /* END LWDART_ERROR */

/**
 * \addtogroup LWDART_DEVICE 
 *
 * @{
 */

/**
 * \brief Returns the number of compute-capable devices
 *
 * Returns in \p *count the number of devices with compute capability greater
 * or equal to 2.0 that are available for exelwtion.  If there is no such
 * device then ::lwdaGetDeviceCount() will return ::lwdaErrorNoDevice.
 * If no driver can be loaded to determine if any such devices exist then
 * ::lwdaGetDeviceCount() will return ::lwdaErrorInsufficientDriver.
 *
 * \param count - Returns the number of devices with compute capability
 * greater or equal to 2.0
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorNoDevice,
 * ::lwdaErrorInsufficientDriver
 * \notefnerr
 *
 * \sa ::lwdaGetDevice, ::lwdaSetDevice, ::lwdaGetDeviceProperties,
 * ::lwdaChooseDevice,
 * ::lwDeviceGetCount
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaGetDeviceCount(int *count);

/**
 * \brief Returns information about the compute-device
 *
 * Returns in \p *prop the properties of device \p dev. The ::lwdaDeviceProp
 * structure is defined as:
 * \code
    struct lwdaDeviceProp {
        char name[256];
        size_t totalGlobalMem;
        size_t sharedMemPerBlock;
        int regsPerBlock;
        int warpSize;
        size_t memPitch;
        int maxThreadsPerBlock;
        int maxThreadsDim[3];
        int maxGridSize[3];
        int clockRate;
        size_t totalConstMem;
        int major;
        int minor;
        size_t textureAlignment;
        size_t texturePitchAlignment;
        int deviceOverlap;
        int multiProcessorCount;
        int kernelExecTimeoutEnabled;
        int integrated;
        int canMapHostMemory;
        int computeMode;
        int maxTexture1D;
        int maxTexture1DMipmap;
        int maxTexture1DLinear;
        int maxTexture2D[2];
        int maxTexture2DMipmap[2];
        int maxTexture2DLinear[3];
        int maxTexture2DGather[2];
        int maxTexture3D[3];
        int maxTexture3DAlt[3];
        int maxTextureLwbemap;
        int maxTexture1DLayered[2];
        int maxTexture2DLayered[3];
        int maxTextureLwbemapLayered[2];
        int maxSurface1D;
        int maxSurface2D[2];
        int maxSurface3D[3];
        int maxSurface1DLayered[2];
        int maxSurface2DLayered[3];
        int maxSurfaceLwbemap;
        int maxSurfaceLwbemapLayered[2];
        size_t surfaceAlignment;
        int conlwrrentKernels;
        int ECCEnabled;
        int pciBusID;
        int pciDeviceID;
        int pciDomainID;
        int tccDriver;
        int asyncEngineCount;
        int unifiedAddressing;
        int memoryClockRate;
        int memoryBusWidth;
        int l2CacheSize;
        int maxThreadsPerMultiProcessor;
        int streamPrioritiesSupported;
        int globalL1CacheSupported;
        int localL1CacheSupported;
        size_t sharedMemPerMultiprocessor;
        int regsPerMultiprocessor;
        int managedMemSupported;
        int isMultiGpuBoard;
        int multiGpuBoardGroupID;
        int singleToDoublePrecisionPerfRatio;
        int pageableMemoryAccess;
        int conlwrrentManagedAccess;
        int computePreemptionSupported;
        int canUseHostPointerForRegisteredMem;
        int cooperativeLaunch;
        int cooperativeMultiDeviceLaunch;
    }
 \endcode
 * where:
 * - \ref ::lwdaDeviceProp::name "name[256]" is an ASCII string identifying
 *   the device;
 * - \ref ::lwdaDeviceProp::totalGlobalMem "totalGlobalMem" is the total
 *   amount of global memory available on the device in bytes;
 * - \ref ::lwdaDeviceProp::sharedMemPerBlock "sharedMemPerBlock" is the
 *   maximum amount of shared memory available to a thread block in bytes;
 * - \ref ::lwdaDeviceProp::regsPerBlock "regsPerBlock" is the maximum number
 *   of 32-bit registers available to a thread block;
 * - \ref ::lwdaDeviceProp::warpSize "warpSize" is the warp size in threads;
 * - \ref ::lwdaDeviceProp::memPitch "memPitch" is the maximum pitch in
 *   bytes allowed by the memory copy functions that involve memory regions
 *   allocated through ::lwdaMallocPitch();
 * - \ref ::lwdaDeviceProp::maxThreadsPerBlock "maxThreadsPerBlock" is the
 *   maximum number of threads per block;
 * - \ref ::lwdaDeviceProp::maxThreadsDim "maxThreadsDim[3]" contains the
 *   maximum size of each dimension of a block;
 * - \ref ::lwdaDeviceProp::maxGridSize "maxGridSize[3]" contains the
 *   maximum size of each dimension of a grid;
 * - \ref ::lwdaDeviceProp::clockRate "clockRate" is the clock frequency in
 *   kilohertz;
 * - \ref ::lwdaDeviceProp::totalConstMem "totalConstMem" is the total amount
 *   of constant memory available on the device in bytes;
 * - \ref ::lwdaDeviceProp::major "major",
 *   \ref ::lwdaDeviceProp::minor "minor" are the major and minor revision
 *   numbers defining the device's compute capability;
 * - \ref ::lwdaDeviceProp::textureAlignment "textureAlignment" is the
 *   alignment requirement; texture base addresses that are aligned to
 *   \ref ::lwdaDeviceProp::textureAlignment "textureAlignment" bytes do not
 *   need an offset applied to texture fetches;
 * - \ref ::lwdaDeviceProp::texturePitchAlignment "texturePitchAlignment" is the
 *   pitch alignment requirement for 2D texture references that are bound to 
 *   pitched memory;
 * - \ref ::lwdaDeviceProp::deviceOverlap "deviceOverlap" is 1 if the device
 *   can conlwrrently copy memory between host and device while exelwting a
 *   kernel, or 0 if not.  Deprecated, use instead asyncEngineCount.
 * - \ref ::lwdaDeviceProp::multiProcessorCount "multiProcessorCount" is the
 *   number of multiprocessors on the device;
 * - \ref ::lwdaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
 *   is 1 if there is a run time limit for kernels exelwted on the device, or
 *   0 if not.
 * - \ref ::lwdaDeviceProp::integrated "integrated" is 1 if the device is an
 *   integrated (motherboard) GPU and 0 if it is a discrete (card) component.
 * - \ref ::lwdaDeviceProp::canMapHostMemory "canMapHostMemory" is 1 if the
 *   device can map host memory into the LWCA address space for use with
 *   ::lwdaHostAlloc()/::lwdaHostGetDevicePointer(), or 0 if not;
 * - \ref ::lwdaDeviceProp::computeMode "computeMode" is the compute mode
 *   that the device is lwrrently in. Available modes are as follows:
 *   - lwdaComputeModeDefault: Default mode - Device is not restricted and
 *     multiple threads can use ::lwdaSetDevice() with this device.
 *   - lwdaComputeModeExclusive: Compute-exclusive mode - Only one thread will
 *     be able to use ::lwdaSetDevice() with this device.
 *   - lwdaComputeModeProhibited: Compute-prohibited mode - No threads can use
 *     ::lwdaSetDevice() with this device.
 *   - lwdaComputeModeExclusiveProcess: Compute-exclusive-process mode - Many 
 *     threads in one process will be able to use ::lwdaSetDevice() with this device.
 *   <br> If ::lwdaSetDevice() is called on an already oclwpied \p device with 
 *   computeMode ::lwdaComputeModeExclusive, ::lwdaErrorDeviceAlreadyInUse
 *   will be immediately returned indicating the device cannot be used.
 *   When an oclwpied exclusive mode device is chosen with ::lwdaSetDevice,
 *   all subsequent non-device management runtime functions will return
 *   ::lwdaErrorDevicesUnavailable.
 * - \ref ::lwdaDeviceProp::maxTexture1D "maxTexture1D" is the maximum 1D
 *   texture size.
 * - \ref ::lwdaDeviceProp::maxTexture1DMipmap "maxTexture1DMipmap" is the maximum
 *   1D mipmapped texture texture size.
 * - \ref ::lwdaDeviceProp::maxTexture1DLinear "maxTexture1DLinear" is the maximum
 *   1D texture size for textures bound to linear memory.
 * - \ref ::lwdaDeviceProp::maxTexture2D "maxTexture2D[2]" contains the maximum
 *   2D texture dimensions.
 * - \ref ::lwdaDeviceProp::maxTexture2DMipmap "maxTexture2DMipmap[2]" contains the
 *   maximum 2D mipmapped texture dimensions.
 * - \ref ::lwdaDeviceProp::maxTexture2DLinear "maxTexture2DLinear[3]" contains the 
 *   maximum 2D texture dimensions for 2D textures bound to pitch linear memory.
 * - \ref ::lwdaDeviceProp::maxTexture2DGather "maxTexture2DGather[2]" contains the 
 *   maximum 2D texture dimensions if texture gather operations have to be performed.
 * - \ref ::lwdaDeviceProp::maxTexture3D "maxTexture3D[3]" contains the maximum
 *   3D texture dimensions.
 * - \ref ::lwdaDeviceProp::maxTexture3DAlt "maxTexture3DAlt[3]"
 *   contains the maximum alternate 3D texture dimensions.
 * - \ref ::lwdaDeviceProp::maxTextureLwbemap "maxTextureLwbemap" is the 
 *   maximum lwbemap texture width or height.
 * - \ref ::lwdaDeviceProp::maxTexture1DLayered "maxTexture1DLayered[2]" contains
 *   the maximum 1D layered texture dimensions.
 * - \ref ::lwdaDeviceProp::maxTexture2DLayered "maxTexture2DLayered[3]" contains
 *   the maximum 2D layered texture dimensions.
 * - \ref ::lwdaDeviceProp::maxTextureLwbemapLayered "maxTextureLwbemapLayered[2]"
 *   contains the maximum lwbemap layered texture dimensions.
 * - \ref ::lwdaDeviceProp::maxSurface1D "maxSurface1D" is the maximum 1D
 *   surface size.
 * - \ref ::lwdaDeviceProp::maxSurface2D "maxSurface2D[2]" contains the maximum
 *   2D surface dimensions.
 * - \ref ::lwdaDeviceProp::maxSurface3D "maxSurface3D[3]" contains the maximum
 *   3D surface dimensions.
 * - \ref ::lwdaDeviceProp::maxSurface1DLayered "maxSurface1DLayered[2]" contains
 *   the maximum 1D layered surface dimensions.
 * - \ref ::lwdaDeviceProp::maxSurface2DLayered "maxSurface2DLayered[3]" contains
 *   the maximum 2D layered surface dimensions.
 * - \ref ::lwdaDeviceProp::maxSurfaceLwbemap "maxSurfaceLwbemap" is the maximum 
 *   lwbemap surface width or height.
 * - \ref ::lwdaDeviceProp::maxSurfaceLwbemapLayered "maxSurfaceLwbemapLayered[2]"
 *   contains the maximum lwbemap layered surface dimensions.
 * - \ref ::lwdaDeviceProp::surfaceAlignment "surfaceAlignment" specifies the
 *   alignment requirements for surfaces.
 * - \ref ::lwdaDeviceProp::conlwrrentKernels "conlwrrentKernels" is 1 if the
 *   device supports exelwting multiple kernels within the same context
 *   simultaneously, or 0 if not. It is not guaranteed that multiple kernels
 *   will be resident on the device conlwrrently so this feature should not be
 *   relied upon for correctness;
 * - \ref ::lwdaDeviceProp::ECCEnabled "ECCEnabled" is 1 if the device has ECC
 *   support turned on, or 0 if not.
 * - \ref ::lwdaDeviceProp::pciBusID "pciBusID" is the PCI bus identifier of
 *   the device.
 * - \ref ::lwdaDeviceProp::pciDeviceID "pciDeviceID" is the PCI device
 *   (sometimes called slot) identifier of the device.
 * - \ref ::lwdaDeviceProp::pciDomainID "pciDomainID" is the PCI domain identifier
 *   of the device.
 * - \ref ::lwdaDeviceProp::tccDriver "tccDriver" is 1 if the device is using a
 *   TCC driver or 0 if not.
 * - \ref ::lwdaDeviceProp::asyncEngineCount "asyncEngineCount" is 1 when the
 *   device can conlwrrently copy memory between host and device while exelwting
 *   a kernel. It is 2 when the device can conlwrrently copy memory between host
 *   and device in both directions and execute a kernel at the same time. It is
 *   0 if neither of these is supported.
 * - \ref ::lwdaDeviceProp::unifiedAddressing "unifiedAddressing" is 1 if the device 
 *   shares a unified address space with the host and 0 otherwise.
 * - \ref ::lwdaDeviceProp::memoryClockRate "memoryClockRate" is the peak memory 
 *   clock frequency in kilohertz.
 * - \ref ::lwdaDeviceProp::memoryBusWidth "memoryBusWidth" is the memory bus width  
 *   in bits.
 * - \ref ::lwdaDeviceProp::l2CacheSize "l2CacheSize" is L2 cache size in bytes. 
 * - \ref ::lwdaDeviceProp::maxThreadsPerMultiProcessor "maxThreadsPerMultiProcessor"  
 *   is the number of maximum resident threads per multiprocessor.
 * - \ref ::lwdaDeviceProp::streamPrioritiesSupported "streamPrioritiesSupported"
 *   is 1 if the device supports stream priorities, or 0 if it is not supported.
 * - \ref ::lwdaDeviceProp::globalL1CacheSupported "globalL1CacheSupported"
 *   is 1 if the device supports caching of globals in L1 cache, or 0 if it is not supported.
 * - \ref ::lwdaDeviceProp::localL1CacheSupported "localL1CacheSupported"
 *   is 1 if the device supports caching of locals in L1 cache, or 0 if it is not supported.
 * - \ref ::lwdaDeviceProp::sharedMemPerMultiprocessor "sharedMemPerMultiprocessor" is the
 *   maximum amount of shared memory available to a multiprocessor in bytes; this amount is
 *   shared by all thread blocks simultaneously resident on a multiprocessor;
 * - \ref ::lwdaDeviceProp::regsPerMultiprocessor "regsPerMultiprocessor" is the maximum number
 *   of 32-bit registers available to a multiprocessor; this number is shared
 *   by all thread blocks simultaneously resident on a multiprocessor;
 * - \ref ::lwdaDeviceProp::managedMemory "managedMemory"
 *   is 1 if the device supports allocating managed memory on this system, or 0 if it is not supported.
 * - \ref ::lwdaDeviceProp::isMultiGpuBoard "isMultiGpuBoard"
 *   is 1 if the device is on a multi-GPU board (e.g. Gemini cards), and 0 if not;
 * - \ref ::lwdaDeviceProp::multiGpuBoardGroupID "multiGpuBoardGroupID" is a unique identifier
 *   for a group of devices associated with the same board.
 *   Devices on the same multi-GPU board will share the same identifier;
 * - \ref ::lwdaDeviceProp::singleToDoublePrecisionPerfRatio "singleToDoublePrecisionPerfRatio"  
 *   is the ratio of single precision performance (in floating-point operations per second)
 *   to double precision performance.
 * - \ref ::lwdaDeviceProp::pageableMemoryAccess "pageableMemoryAccess" is 1 if the device supports
 *   coherently accessing pageable memory without calling lwdaHostRegister on it, and 0 otherwise.
 * - \ref ::lwdaDeviceProp::conlwrrentManagedAccess "conlwrrentManagedAccess" is 1 if the device can
 *   coherently access managed memory conlwrrently with the CPU, and 0 otherwise.
 * - \ref ::lwdaDeviceProp::computePreemptionSupported "computePreemptionSupported" is 1 if the device
 *   supports Compute Preemption, and 0 otherwise.
 * - \ref ::lwdaDeviceProp::canUseHostPointerForRegisteredMem "canUseHostPointerForRegisteredMem" is 1 if
 *   the device can access host registered memory at the same virtual address as the CPU, and 0 otherwise.
 * - \ref ::lwdaDeviceProp::cooperativeLaunch "cooperativeLaunch" is 1 if the device supports launching
 *   cooperative kernels via ::lwdaLaunchCooperativeKernel, and 0 otherwise.
 * - \ref ::lwdaDeviceProp::cooperativeMultiDeviceLaunch "cooperativeMultiDeviceLaunch" is 1 if the device
 *   supports launching cooperative kernels via ::lwdaLaunchCooperativeKernelMultiDevice, and 0 otherwise.
 *
 * \param prop   - Properties for the specified device
 * \param device - Device number to get properties for
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidDevice
 *
 * \sa ::lwdaGetDeviceCount, ::lwdaGetDevice, ::lwdaSetDevice, ::lwdaChooseDevice,
 * ::lwdaDeviceGetAttribute,
 * ::lwDeviceGetAttribute,
 * ::lwDeviceGetName
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaGetDeviceProperties(struct lwdaDeviceProp *prop, int device);

/**
 * \brief Returns information about the device
 *
 * Returns in \p *value the integer value of the attribute \p attr on device
 * \p device. The supported attributes are:
 * - ::lwdaDevAttrMaxThreadsPerBlock: Maximum number of threads per block;
 * - ::lwdaDevAttrMaxBlockDimX: Maximum x-dimension of a block;
 * - ::lwdaDevAttrMaxBlockDimY: Maximum y-dimension of a block;
 * - ::lwdaDevAttrMaxBlockDimZ: Maximum z-dimension of a block;
 * - ::lwdaDevAttrMaxGridDimX: Maximum x-dimension of a grid;
 * - ::lwdaDevAttrMaxGridDimY: Maximum y-dimension of a grid;
 * - ::lwdaDevAttrMaxGridDimZ: Maximum z-dimension of a grid;
 * - ::lwdaDevAttrMaxSharedMemoryPerBlock: Maximum amount of shared memory
 *   available to a thread block in bytes;
 * - ::lwdaDevAttrTotalConstantMemory: Memory available on device for
 *   __constant__ variables in a LWCA C kernel in bytes;
 * - ::lwdaDevAttrWarpSize: Warp size in threads;
 * - ::lwdaDevAttrMaxPitch: Maximum pitch in bytes allowed by the memory copy
 *   functions that involve memory regions allocated through ::lwdaMallocPitch();
 * - ::lwdaDevAttrMaxTexture1DWidth: Maximum 1D texture width;
 * - ::lwdaDevAttrMaxTexture1DLinearWidth: Maximum width for a 1D texture bound
 *   to linear memory;
 * - ::lwdaDevAttrMaxTexture1DMipmappedWidth: Maximum mipmapped 1D texture width;
 * - ::lwdaDevAttrMaxTexture2DWidth: Maximum 2D texture width;
 * - ::lwdaDevAttrMaxTexture2DHeight: Maximum 2D texture height;
 * - ::lwdaDevAttrMaxTexture2DLinearWidth: Maximum width for a 2D texture
 *   bound to linear memory;
 * - ::lwdaDevAttrMaxTexture2DLinearHeight: Maximum height for a 2D texture
 *   bound to linear memory;
 * - ::lwdaDevAttrMaxTexture2DLinearPitch: Maximum pitch in bytes for a 2D
 *   texture bound to linear memory;
 * - ::lwdaDevAttrMaxTexture2DMipmappedWidth: Maximum mipmapped 2D texture
 *   width;
 * - ::lwdaDevAttrMaxTexture2DMipmappedHeight: Maximum mipmapped 2D texture
 *   height;
 * - ::lwdaDevAttrMaxTexture3DWidth: Maximum 3D texture width;
 * - ::lwdaDevAttrMaxTexture3DHeight: Maximum 3D texture height;
 * - ::lwdaDevAttrMaxTexture3DDepth: Maximum 3D texture depth;
 * - ::lwdaDevAttrMaxTexture3DWidthAlt: Alternate maximum 3D texture width,
 *   0 if no alternate maximum 3D texture size is supported;
 * - ::lwdaDevAttrMaxTexture3DHeightAlt: Alternate maximum 3D texture height,
 *   0 if no alternate maximum 3D texture size is supported;
 * - ::lwdaDevAttrMaxTexture3DDepthAlt: Alternate maximum 3D texture depth,
 *   0 if no alternate maximum 3D texture size is supported;
 * - ::lwdaDevAttrMaxTextureLwbemapWidth: Maximum lwbemap texture width or
 *   height;
 * - ::lwdaDevAttrMaxTexture1DLayeredWidth: Maximum 1D layered texture width;
 * - ::lwdaDevAttrMaxTexture1DLayeredLayers: Maximum layers in a 1D layered
 *   texture;
 * - ::lwdaDevAttrMaxTexture2DLayeredWidth: Maximum 2D layered texture width;
 * - ::lwdaDevAttrMaxTexture2DLayeredHeight: Maximum 2D layered texture height;
 * - ::lwdaDevAttrMaxTexture2DLayeredLayers: Maximum layers in a 2D layered
 *   texture;
 * - ::lwdaDevAttrMaxTextureLwbemapLayeredWidth: Maximum lwbemap layered
 *   texture width or height;
 * - ::lwdaDevAttrMaxTextureLwbemapLayeredLayers: Maximum layers in a lwbemap
 *   layered texture;
 * - ::lwdaDevAttrMaxSurface1DWidth: Maximum 1D surface width;
 * - ::lwdaDevAttrMaxSurface2DWidth: Maximum 2D surface width;
 * - ::lwdaDevAttrMaxSurface2DHeight: Maximum 2D surface height;
 * - ::lwdaDevAttrMaxSurface3DWidth: Maximum 3D surface width;
 * - ::lwdaDevAttrMaxSurface3DHeight: Maximum 3D surface height;
 * - ::lwdaDevAttrMaxSurface3DDepth: Maximum 3D surface depth;
 * - ::lwdaDevAttrMaxSurface1DLayeredWidth: Maximum 1D layered surface width;
 * - ::lwdaDevAttrMaxSurface1DLayeredLayers: Maximum layers in a 1D layered
 *   surface;
 * - ::lwdaDevAttrMaxSurface2DLayeredWidth: Maximum 2D layered surface width;
 * - ::lwdaDevAttrMaxSurface2DLayeredHeight: Maximum 2D layered surface height;
 * - ::lwdaDevAttrMaxSurface2DLayeredLayers: Maximum layers in a 2D layered
 *   surface;
 * - ::lwdaDevAttrMaxSurfaceLwbemapWidth: Maximum lwbemap surface width;
 * - ::lwdaDevAttrMaxSurfaceLwbemapLayeredWidth: Maximum lwbemap layered
 *   surface width;
 * - ::lwdaDevAttrMaxSurfaceLwbemapLayeredLayers: Maximum layers in a lwbemap
 *   layered surface;
 * - ::lwdaDevAttrMaxRegistersPerBlock: Maximum number of 32-bit registers 
 *   available to a thread block;
 * - ::lwdaDevAttrClockRate: Peak clock frequency in kilohertz;
 * - ::lwdaDevAttrTextureAlignment: Alignment requirement; texture base
 *   addresses aligned to ::textureAlign bytes do not need an offset applied
 *   to texture fetches;
 * - ::lwdaDevAttrTexturePitchAlignment: Pitch alignment requirement for 2D
 *   texture references bound to pitched memory;
 * - ::lwdaDevAttrGpuOverlap: 1 if the device can conlwrrently copy memory
 *   between host and device while exelwting a kernel, or 0 if not;
 * - ::lwdaDevAttrMultiProcessorCount: Number of multiprocessors on the device;
 * - ::lwdaDevAttrKernelExecTimeout: 1 if there is a run time limit for kernels
 *   exelwted on the device, or 0 if not;
 * - ::lwdaDevAttrIntegrated: 1 if the device is integrated with the memory
 *   subsystem, or 0 if not;
 * - ::lwdaDevAttrCanMapHostMemory: 1 if the device can map host memory into
 *   the LWCA address space, or 0 if not;
 * - ::lwdaDevAttrComputeMode: Compute mode is the compute mode that the device
 *   is lwrrently in. Available modes are as follows:
 *   - ::lwdaComputeModeDefault: Default mode - Device is not restricted and
 *     multiple threads can use ::lwdaSetDevice() with this device.
 *   - ::lwdaComputeModeExclusive: Compute-exclusive mode - Only one thread will
 *     be able to use ::lwdaSetDevice() with this device.
 *   - ::lwdaComputeModeProhibited: Compute-prohibited mode - No threads can use
 *     ::lwdaSetDevice() with this device.
 *   - ::lwdaComputeModeExclusiveProcess: Compute-exclusive-process mode - Many 
 *     threads in one process will be able to use ::lwdaSetDevice() with this
 *     device.
 * - ::lwdaDevAttrConlwrrentKernels: 1 if the device supports exelwting
 *   multiple kernels within the same context simultaneously, or 0 if
 *   not. It is not guaranteed that multiple kernels will be resident on the
 *   device conlwrrently so this feature should not be relied upon for
 *   correctness;
 * - ::lwdaDevAttrEccEnabled: 1 if error correction is enabled on the device,
 *   0 if error correction is disabled or not supported by the device;
 * - ::lwdaDevAttrPciBusId: PCI bus identifier of the device;
 * - ::lwdaDevAttrPciDeviceId: PCI device (also known as slot) identifier of
 *   the device;
 * - ::lwdaDevAttrTccDriver: 1 if the device is using a TCC driver. TCC is only
 *   available on Tesla hardware running Windows Vista or later;
 * - ::lwdaDevAttrMemoryClockRate: Peak memory clock frequency in kilohertz;
 * - ::lwdaDevAttrGlobalMemoryBusWidth: Global memory bus width in bits;
 * - ::lwdaDevAttrL2CacheSize: Size of L2 cache in bytes. 0 if the device
 *   doesn't have L2 cache;
 * - ::lwdaDevAttrMaxThreadsPerMultiProcessor: Maximum resident threads per 
 *   multiprocessor;
 * - ::lwdaDevAttrUnifiedAddressing: 1 if the device shares a unified address
 *   space with the host, or 0 if not;
 * - ::lwdaDevAttrComputeCapabilityMajor: Major compute capability version
 *   number;
 * - ::lwdaDevAttrComputeCapabilityMinor: Minor compute capability version
 *   number;
 * - ::lwdaDevAttrStreamPrioritiesSupported: 1 if the device supports stream
 *   priorities, or 0 if not;
 * - ::lwdaDevAttrGlobalL1CacheSupported: 1 if device supports caching globals 
 *    in L1 cache, 0 if not;
 * - ::lwdaDevAttrLocalL1CacheSupported: 1 if device supports caching locals 
 *    in L1 cache, 0 if not;
 * - ::lwdaDevAttrMaxSharedMemoryPerMultiprocessor: Maximum amount of shared memory
 *   available to a multiprocessor in bytes; this amount is shared by all 
 *   thread blocks simultaneously resident on a multiprocessor;
 * - ::lwdaDevAttrMaxRegistersPerMultiprocessor: Maximum number of 32-bit registers 
 *   available to a multiprocessor; this number is shared by all thread blocks
 *   simultaneously resident on a multiprocessor;
 * - ::lwdaDevAttrManagedMemSupported: 1 if device supports allocating
 *   managed memory, 0 if not;
 * - ::lwdaDevAttrIsMultiGpuBoard: 1 if device is on a multi-GPU board, 0 if not;
 * - ::lwdaDevAttrMultiGpuBoardGroupID: Unique identifier for a group of devices on the
 *   same multi-GPU board;
 * - ::lwdaDevAttrHostNativeAtomicSupported: 1 if the link between the device and the
 *   host supports native atomic operations;
 * - ::lwdaDevAttrSingleToDoublePrecisionPerfRatio: Ratio of single precision performance
 *   (in floating-point operations per second) to double precision performance;
 * - ::lwdaDevAttrPageableMemoryAccess: 1 if the device supports coherently accessing
 *   pageable memory without calling lwdaHostRegister on it, and 0 otherwise.
 * - ::lwdaDevAttrConlwrrentManagedAccess: 1 if the device can coherently access managed
 *   memory conlwrrently with the CPU, and 0 otherwise.
 * - ::lwdaDevAttrComputePreemptionSupported: 1 if the device supports
 *   Compute Preemption, 0 if not.
 * - ::lwdaDevAttrCanUseHostPointerForRegisteredMem: 1 if the device can access host
 *   registered memory at the same virtual address as the CPU, and 0 otherwise.
 * - ::lwdaDevAttrCooperativeLaunch: 1 if the device supports launching cooperative kernels
 *   via ::lwdaLaunchCooperativeKernel, and 0 otherwise.
 * - ::lwdaDevAttrCooperativeMultiDeviceLaunch: 1 if the device supports launching cooperative
 *   kernels via ::lwdaLaunchCooperativeKernelMultiDevice, and 0 otherwise.
 *
 * \param value  - Returned device attribute value
 * \param attr   - Device attribute to query
 * \param device - Device number to query 
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidDevice,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \sa ::lwdaGetDeviceCount, ::lwdaGetDevice, ::lwdaSetDevice, ::lwdaChooseDevice,
 * ::lwdaGetDeviceProperties,
 * ::lwDeviceGetAttribute
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaDeviceGetAttribute(int *value, enum lwdaDeviceAttr attr, int device);

/**
 * \brief Queries attributes of the link between two devices.
 *
 * Returns in \p *value the value of the requested attribute \p attrib of the
 * link between \p srcDevice and \p dstDevice. The supported attributes are:
 * - ::LwdaDevP2PAttrPerformanceRank: A relative value indicating the
 *   performance of the link between two devices. Lower value means better
 *   performance (0 being the value used for most performant link).
 * - ::LwdaDevP2PAttrAccessSupported: 1 if peer access is enabled.
 * - ::LwdaDevP2PAttrNativeAtomicSupported: 1 if native atomic operations over
 *   the link are supported.
 *
 * Returns ::lwdaErrorIlwalidDevice if \p srcDevice or \p dstDevice are not valid
 * or if they represent the same device.
 *
 * Returns ::lwdaErrorIlwalidValue if \p attrib is not valid or if \p value is
 * a null pointer.
 *
 * \param value         - Returned value of the requested attribute
 * \param attrib        - The requested attribute of the link between \p srcDevice and \p dstDevice.
 * \param srcDevice     - The source device of the target link.
 * \param dstDevice     - The destination device of the target link.
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidDevice,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \sa ::lwdaCtxEnablePeerAccess,
 * ::lwdaCtxDisablePeerAccess,
 * ::lwdaCtxCanAccessPeer,
 * ::lwDeviceGetP2PAttribute
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaDeviceGetP2PAttribute(int *value, enum lwdaDeviceP2PAttr attr, int srcDevice, int dstDevice);

/**
 * \brief Select compute-device which best matches criteria
 *
 * Returns in \p *device the device which has properties that best match
 * \p *prop.
 *
 * \param device - Device with best match
 * \param prop   - Desired device properties
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \sa ::lwdaGetDeviceCount, ::lwdaGetDevice, ::lwdaSetDevice,
 * ::lwdaGetDeviceProperties
 */
extern __host__ lwdaError_t LWDARTAPI lwdaChooseDevice(int *device, const struct lwdaDeviceProp *prop);

/**
 * \brief Set device to be used for GPU exelwtions
 *
 * Sets \p device as the current device for the calling host thread.
 * Valid device id's are 0 to (::lwdaGetDeviceCount() - 1).
 *
 * Any device memory subsequently allocated from this host thread
 * using ::lwdaMalloc(), ::lwdaMallocPitch() or ::lwdaMallocArray()
 * will be physically resident on \p device.  Any host memory allocated
 * from this host thread using ::lwdaMallocHost() or ::lwdaHostAlloc() 
 * or ::lwdaHostRegister() will have its lifetime associated  with
 * \p device.  Any streams or events created from this host thread will 
 * be associated with \p device.  Any kernels launched from this host
 * thread using the <<<>>> operator or ::lwdaLaunchKernel() will be exelwted
 * on \p device.
 *
 * This call may be made from any host thread, to any device, and at 
 * any time.  This function will do no synchronization with the previous 
 * or new device, and should be considered a very low overhead call.
 *
 * \param device - Device on which the active host thread should execute the
 * device code.
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidDevice,
 * ::lwdaErrorDeviceAlreadyInUse
 * \notefnerr
 *
 * \sa ::lwdaGetDeviceCount, ::lwdaGetDevice, ::lwdaGetDeviceProperties,
 * ::lwdaChooseDevice,
 * ::lwCtxSetLwrrent
 */
extern __host__ lwdaError_t LWDARTAPI lwdaSetDevice(int device);

/**
 * \brief Returns which device is lwrrently being used
 *
 * Returns in \p *device the current device for the calling host thread.
 *
 * \param device - Returns the device on which the active host thread
 * exelwtes the device code.
 *
 * \return
 * ::lwdaSuccess
 * \notefnerr
 *
 * \sa ::lwdaGetDeviceCount, ::lwdaSetDevice, ::lwdaGetDeviceProperties,
 * ::lwdaChooseDevice,
 * ::lwCtxGetLwrrent
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaGetDevice(int *device);

/**
 * \brief Set a list of devices that can be used for LWCA
 *
 * Sets a list of devices for LWCA exelwtion in priority order using
 * \p device_arr. The parameter \p len specifies the number of elements in the
 * list.  LWCA will try devices from the list sequentially until it finds one
 * that works.  If this function is not called, or if it is called with a \p len
 * of 0, then LWCA will go back to its default behavior of trying devices
 * sequentially from a default list containing all of the available LWCA
 * devices in the system. If a specified device ID in the list does not exist,
 * this function will return ::lwdaErrorIlwalidDevice. If \p len is not 0 and
 * \p device_arr is NULL or if \p len exceeds the number of devices in
 * the system, then ::lwdaErrorIlwalidValue is returned.
 *
 * \param device_arr - List of devices to try
 * \param len        - Number of devices in specified list
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidDevice
 * \notefnerr
 *
 * \sa ::lwdaGetDeviceCount, ::lwdaSetDevice, ::lwdaGetDeviceProperties,
 * ::lwdaSetDeviceFlags,
 * ::lwdaChooseDevice
 */
extern __host__ lwdaError_t LWDARTAPI lwdaSetValidDevices(int *device_arr, int len);

/**
 * \brief Sets flags to be used for device exelwtions
 *
 * Records \p flags as the flags to use when initializing the current 
 * device.  If no device has been made current to the calling thread,
 * then \p flags will be applied to the initialization of any device
 * initialized by the calling host thread, unless that device has had
 * its initialization flags set explicitly by this or any host thread.
 * 
 * If the current device has been set and that device has already been 
 * initialized then this call will fail with the error 
 * ::lwdaErrorSetOnActiveProcess.  In this case it is necessary 
 * to reset \p device using ::lwdaDeviceReset() before the device's
 * initialization flags may be set.
 *
 * The two LSBs of the \p flags parameter can be used to control how the CPU
 * thread interacts with the OS scheduler when waiting for results from the
 * device.
 *
 * - ::lwdaDeviceScheduleAuto: The default value if the \p flags parameter is
 * zero, uses a heuristic based on the number of active LWCA contexts in the
 * process \p C and the number of logical processors in the system \p P. If
 * \p C \> \p P, then LWCA will yield to other OS threads when waiting for the
 * device, otherwise LWCA will not yield while waiting for results and
 * actively spin on the processor.
 * - ::lwdaDeviceScheduleSpin: Instruct LWCA to actively spin when waiting for
 * results from the device. This can decrease latency when waiting for the
 * device, but may lower the performance of CPU threads if they are performing
 * work in parallel with the LWCA thread.
 * - ::lwdaDeviceScheduleYield: Instruct LWCA to yield its thread when waiting
 * for results from the device. This can increase latency when waiting for the
 * device, but can increase the performance of CPU threads performing work in
 * parallel with the device.
 * - ::lwdaDeviceScheduleBlockingSync: Instruct LWCA to block the CPU thread 
 * on a synchronization primitive when waiting for the device to finish work.
 * - ::lwdaDeviceBlockingSync: Instruct LWCA to block the CPU thread on a 
 * synchronization primitive when waiting for the device to finish work. <br>
 * \ref deprecated "Deprecated:" This flag was deprecated as of LWCA 4.0 and
 * replaced with ::lwdaDeviceScheduleBlockingSync.
 * - ::lwdaDeviceMapHost: This flag enables allocating pinned
 * host memory that is accessible to the device. It is implicit for the
 * runtime but may be absent if a context is created using the driver API.
 * If this flag is not set, ::lwdaHostGetDevicePointer() will always return
 * a failure code.
 * - ::lwdaDeviceLmemResizeToMax: Instruct LWCA to not reduce local memory
 * after resizing local memory for a kernel. This can prevent thrashing by
 * local memory allocations when launching many kernels with high local
 * memory usage at the cost of potentially increased memory usage.
 *
 * \param flags - Parameters for device operation
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidDevice,
 * ::lwdaErrorSetOnActiveProcess
 *
 * \sa ::lwdaGetDeviceFlags, ::lwdaGetDeviceCount, ::lwdaGetDevice, ::lwdaGetDeviceProperties,
 * ::lwdaSetDevice, ::lwdaSetValidDevices,
 * ::lwdaChooseDevice,
 * ::lwDevicePrimaryCtxSetFlags
 */
extern __host__ lwdaError_t LWDARTAPI lwdaSetDeviceFlags( unsigned int flags );

/**
 * \brief Gets the flags for the current device
 *
 * Returns in \p flags the flags for the current device.  If there is a
 * current device for the calling thread, and the device has been initialized
 * or flags have been set on that device specifically, the flags for the
 * device are returned.  If there is no current device, but flags have been
 * set for the thread with ::lwdaSetDeviceFlags, the thread flags are returned.
 * Finally, if there is no current device and no thread flags, the flags for
 * the first device are returned, which may be the default flags.  Compare
 * to the behavior of ::lwdaSetDeviceFlags.
 *
 * Typically, the flags returned should match the behavior that will be seen
 * if the calling thread uses a device after this call, without any change to
 * the flags or current device inbetween by this or another thread.  Note that
 * if the device is not initialized, it is possible for another thread to
 * change the flags for the current device before it is initialized.
 * Additionally, when using exclusive mode, if this thread has not requested a
 * specific device, it may use a device other than the first device, contrary
 * to the assumption made by this function.
 *
 * If a context has been created via the driver API and is current to the
 * calling thread, the flags for that context are always returned.
 *
 * Flags returned by this function may specifically include ::lwdaDeviceMapHost
 * even though it is not accepted by ::lwdaSetDeviceFlags because it is
 * implicit in runtime API flags.  The reason for this is that the current
 * context may have been created via the driver API in which case the flag is
 * not implicit and may be unset.
 *
 * \param flags - Pointer to store the device flags
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidDevice
 *
 * \sa ::lwdaGetDevice, ::lwdaGetDeviceProperties,
 * ::lwdaSetDevice, ::lwdaSetDeviceFlags,
 * ::lwCtxGetFlags,
 * ::lwDevicePrimaryCtxGetState
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGetDeviceFlags( unsigned int *flags );
/** @} */ /* END LWDART_DEVICE */

/**
 * \defgroup LWDART_STREAM Stream Management
 *
 * ___MANBRIEF___ stream management functions of the LWCA runtime API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the stream management functions of the LWCA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Create an asynchronous stream
 *
 * Creates a new asynchronous stream.
 *
 * \param pStream - Pointer to new stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \sa ::lwdaStreamCreateWithPriority,
 * ::lwdaStreamCreateWithFlags,
 * ::lwdaStreamGetPriority,
 * ::lwdaStreamGetFlags,
 * ::lwdaStreamQuery,
 * ::lwdaStreamSynchronize,
 * ::lwdaStreamWaitEvent,
 * ::lwdaStreamAddCallback,
 * ::lwdaStreamDestroy,
 * ::lwStreamCreate
 */
extern __host__ lwdaError_t LWDARTAPI lwdaStreamCreate(lwdaStream_t *pStream);

/**
 * \brief Create an asynchronous stream
 *
 * Creates a new asynchronous stream.  The \p flags argument determines the 
 * behaviors of the stream.  Valid values for \p flags are
 * - ::lwdaStreamDefault: Default stream creation flag.
 * - ::lwdaStreamNonBlocking: Specifies that work running in the created 
 *   stream may run conlwrrently with work in stream 0 (the NULL stream), and that
 *   the created stream should perform no implicit synchronization with stream 0.
 *
 * \param pStream - Pointer to new stream identifier
 * \param flags   - Parameters for stream creation
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \sa ::lwdaStreamCreate,
 * ::lwdaStreamCreateWithPriority,
 * ::lwdaStreamGetFlags,
 * ::lwdaStreamQuery,
 * ::lwdaStreamSynchronize,
 * ::lwdaStreamWaitEvent,
 * ::lwdaStreamAddCallback,
 * ::lwdaStreamDestroy,
 * ::lwStreamCreate
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaStreamCreateWithFlags(lwdaStream_t *pStream, unsigned int flags);

/**
 * \brief Create an asynchronous stream with the specified priority
 *
 * Creates a stream with the specified priority and returns a handle in \p pStream.
 * This API alters the scheduler priority of work in the stream. Work in a higher
 * priority stream may preempt work already exelwting in a low priority stream.
 *
 * \p priority follows a convention where lower numbers represent higher priorities.
 * '0' represents default priority. The range of meaningful numerical priorities can
 * be queried using ::lwdaDeviceGetStreamPriorityRange. If the specified priority is
 * outside the numerical range returned by ::lwdaDeviceGetStreamPriorityRange,
 * it will automatically be clamped to the lowest or the highest number in the range.
 *
 * \param pStream  - Pointer to new stream identifier
 * \param flags    - Flags for stream creation. See ::lwdaStreamCreateWithFlags for a list of valid flags that can be passed
 * \param priority - Priority of the stream. Lower numbers represent higher priorities.
 *                   See ::lwdaDeviceGetStreamPriorityRange for more information about
 *                   the meaningful stream priorities that can be passed.
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \note Stream priorities are supported only on GPUs
 * with compute capability 3.5 or higher.
 *
 * \note In the current implementation, only compute kernels launched in
 * priority streams are affected by the stream's priority. Stream priorities have
 * no effect on host-to-device and device-to-host memory operations.
 *
 * \sa ::lwdaStreamCreate,
 * ::lwdaStreamCreateWithFlags,
 * ::lwdaDeviceGetStreamPriorityRange,
 * ::lwdaStreamGetPriority,
 * ::lwdaStreamQuery,
 * ::lwdaStreamWaitEvent,
 * ::lwdaStreamAddCallback,
 * ::lwdaStreamSynchronize,
 * ::lwdaStreamDestroy,
 * ::lwStreamCreateWithPriority
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaStreamCreateWithPriority(lwdaStream_t *pStream, unsigned int flags, int priority);

/**
 * \brief Query the priority of a stream
 *
 * Query the priority of a stream. The priority is returned in in \p priority.
 * Note that if the stream was created with a priority outside the meaningful
 * numerical range returned by ::lwdaDeviceGetStreamPriorityRange,
 * this function returns the clamped priority.
 * See ::lwdaStreamCreateWithPriority for details about priority clamping.
 *
 * \param hStream    - Handle to the stream to be queried
 * \param priority   - Pointer to a signed integer in which the stream's priority is returned
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidResourceHandle
 * \notefnerr
 *
 * \sa ::lwdaStreamCreateWithPriority,
 * ::lwdaDeviceGetStreamPriorityRange,
 * ::lwdaStreamGetFlags,
 * ::lwStreamGetPriority
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaStreamGetPriority(lwdaStream_t hStream, int *priority);

/**
 * \brief Query the flags of a stream
 *
 * Query the flags of a stream. The flags are returned in \p flags.
 * See ::lwdaStreamCreateWithFlags for a list of valid flags.
 *
 * \param hStream - Handle to the stream to be queried
 * \param flags   - Pointer to an unsigned integer in which the stream's flags are returned
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidResourceHandle
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwdaStreamCreateWithPriority,
 * ::lwdaStreamCreateWithFlags,
 * ::lwdaStreamGetPriority,
 * ::lwStreamGetFlags
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaStreamGetFlags(lwdaStream_t hStream, unsigned int *flags);

/**
 * \brief Destroys and cleans up an asynchronous stream
 *
 * Destroys and cleans up the asynchronous stream specified by \p stream.
 *
 * In case the device is still doing work in the stream \p stream
 * when ::lwdaStreamDestroy() is called, the function will return immediately 
 * and the resources associated with \p stream will be released automatically 
 * once the device has completed all work in \p stream.
 *
 * \param stream - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidResourceHandle
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwdaStreamCreate,
 * ::lwdaStreamCreateWithFlags,
 * ::lwdaStreamQuery,
 * ::lwdaStreamWaitEvent,
 * ::lwdaStreamSynchronize,
 * ::lwdaStreamAddCallback,
 * ::lwStreamDestroy
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaStreamDestroy(lwdaStream_t stream);

/**
 * \brief Make a compute stream wait on an event
 *
 * Makes all future work submitted to \p stream wait until \p event reports
 * completion before beginning exelwtion.  This synchronization will be
 * performed efficiently on the device.  The event \p event may
 * be from a different context than \p stream, in which case this function
 * will perform cross-device synchronization.
 *
 * The stream \p stream will wait only for the completion of the most recent
 * host call to ::lwdaEventRecord() on \p event.  Once this call has returned,
 * any functions (including ::lwdaEventRecord() and ::lwdaEventDestroy()) may be
 * called on \p event again, and the subsequent calls will not have any effect
 * on \p stream.
 *
 * If ::lwdaEventRecord() has not been called on \p event, this call acts as if
 * the record has already completed, and so is a functional no-op.
 *
 * \param stream - Stream to wait
 * \param event  - Event to wait on
 * \param flags  - Parameters for the operation (must be 0)
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidResourceHandle
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwdaStreamCreate, ::lwdaStreamCreateWithFlags, ::lwdaStreamQuery, ::lwdaStreamSynchronize, ::lwdaStreamAddCallback, ::lwdaStreamDestroy,
 * ::lwStreamWaitEvent
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaStreamWaitEvent(lwdaStream_t stream, lwdaEvent_t event, unsigned int flags);

#ifdef _WIN32
#define LWDART_CB __stdcall
#else
#define LWDART_CB
#endif

/**
 * Type of stream callback functions.
 * \param stream The stream as passed to ::lwdaStreamAddCallback, may be NULL.
 * \param status ::lwdaSuccess or any persistent error on the stream.
 * \param userData User parameter provided at registration.
 */
typedef void (LWDART_CB *lwdaStreamCallback_t)(lwdaStream_t stream, lwdaError_t status, void *userData);

/**
 * \brief Add a callback to a compute stream
 *
 * Adds a callback to be called on the host after all lwrrently enqueued
 * items in the stream have completed.  For each 
 * lwdaStreamAddCallback call, a callback will be exelwted exactly once.
 * The callback will block later work in the stream until it is finished.
 *
 * The callback may be passed ::lwdaSuccess or an error code.  In the event
 * of a device error, all subsequently exelwted callbacks will receive an
 * appropriate ::lwdaError_t.
 *
 * Callbacks must not make any LWCA API calls.  Attempting to use LWCA APIs
 * will result in ::lwdaErrorNotPermitted.  Callbacks must not perform any
 * synchronization that may depend on outstanding device work or other callbacks
 * that are not mandated to run earlier.  Callbacks without a mandated order
 * (in independent streams) execute in undefined order and may be serialized.
 *
 * For the purposes of Unified Memory, callback exelwtion makes a number of
 * guarantees:
 * <ul>
 *   <li>The callback stream is considered idle for the duration of the
 *   callback.  Thus, for example, a callback may always use memory attached
 *   to the callback stream.</li>
 *   <li>The start of exelwtion of a callback has the same effect as
 *   synchronizing an event recorded in the same stream immediately prior to
 *   the callback.  It thus synchronizes streams which have been "joined"
 *   prior to the callback.</li>
 *   <li>Adding device work to any stream does not have the effect of making
 *   the stream active until all preceding callbacks have exelwted.  Thus, for
 *   example, a callback might use global attached memory even if work has
 *   been added to another stream, if it has been properly ordered with an
 *   event.</li>
 *   <li>Completion of a callback does not cause a stream to become
 *   active except as described above.  The callback stream will remain idle
 *   if no device work follows the callback, and will remain idle across
 *   conselwtive callbacks without device work in between.  Thus, for example,
 *   stream synchronization can be done by signaling from a callback at the
 *   end of the stream.</li>
 * </ul>
 *
 * \param stream   - Stream to add callback to
 * \param callback - The function to call once preceding stream operations are complete
 * \param userData - User specified data to be passed to the callback function
 * \param flags    - Reserved for future use, must be 0
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorNotSupported
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwdaStreamCreate, ::lwdaStreamCreateWithFlags, ::lwdaStreamQuery, ::lwdaStreamSynchronize, ::lwdaStreamWaitEvent, ::lwdaStreamDestroy, ::lwdaMallocManaged, ::lwdaStreamAttachMemAsync,
 * ::lwStreamAddCallback
 */
extern __host__ lwdaError_t LWDARTAPI lwdaStreamAddCallback(lwdaStream_t stream,
        lwdaStreamCallback_t callback, void *userData, unsigned int flags);

/**
 * \brief Waits for stream tasks to complete
 *
 * Blocks until \p stream has completed all operations. If the
 * ::lwdaDeviceScheduleBlockingSync flag was set for this device, 
 * the host thread will block until the stream is finished with 
 * all of its tasks.
 *
 * \param stream - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidResourceHandle
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwdaStreamCreate, ::lwdaStreamCreateWithFlags, ::lwdaStreamQuery, ::lwdaStreamWaitEvent, ::lwdaStreamAddCallback, ::lwdaStreamDestroy,
 * ::lwStreamSynchronize
 */
extern __host__ lwdaError_t LWDARTAPI lwdaStreamSynchronize(lwdaStream_t stream);

/**
 * \brief Queries an asynchronous stream for completion status
 *
 * Returns ::lwdaSuccess if all operations in \p stream have
 * completed, or ::lwdaErrorNotReady if not.
 *
 * For the purposes of Unified Memory, a return value of ::lwdaSuccess
 * is equivalent to having called ::lwdaStreamSynchronize().
 *
 * \param stream - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorNotReady,
 * ::lwdaErrorIlwalidResourceHandle
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwdaStreamCreate, ::lwdaStreamCreateWithFlags, ::lwdaStreamWaitEvent, ::lwdaStreamSynchronize, ::lwdaStreamAddCallback, ::lwdaStreamDestroy,
 * ::lwStreamQuery
 */
extern __host__ lwdaError_t LWDARTAPI lwdaStreamQuery(lwdaStream_t stream);

/**
 * \brief Attach memory to a stream asynchronously
 *
 * Enqueues an operation in \p stream to specify stream association of
 * \p length bytes of memory starting from \p devPtr. This function is a
 * stream-ordered operation, meaning that it is dependent on, and will
 * only take effect when, previous work in stream has completed. Any
 * previous association is automatically replaced.
 *
 * \p devPtr must point to an address within managed memory space declared
 * using the __managed__ keyword or allocated with ::lwdaMallocManaged.
 *
 * \p length must be zero, to indicate that the entire allocation's
 * stream association is being changed.  Lwrrently, it's not possible
 * to change stream association for a portion of an allocation. The default
 * value for \p length is zero.
 *
 * The stream association is specified using \p flags which must be
 * one of ::lwdaMemAttachGlobal, ::lwdaMemAttachHost or ::lwdaMemAttachSingle.
 * The default value for \p flags is ::lwdaMemAttachSingle
 * If the ::lwdaMemAttachGlobal flag is specified, the memory can be accessed
 * by any stream on any device.
 * If the ::lwdaMemAttachHost flag is specified, the program makes a guarantee
 * that it won't access the memory on the device from any stream on a device that
 * has a zero value for the device attribute ::lwdaDevAttrConlwrrentManagedAccess.
 * If the ::lwdaMemAttachSingle flag is specified and \p stream is associated with
 * a device that has a zero value for the device attribute ::lwdaDevAttrConlwrrentManagedAccess,
 * the program makes a guarantee that it will only access the memory on the device
 * from \p stream. It is illegal to attach singly to the NULL stream, because the
 * NULL stream is a virtual global stream and not a specific stream. An error will
 * be returned in this case.
 *
 * When memory is associated with a single stream, the Unified Memory system will
 * allow CPU access to this memory region so long as all operations in \p stream
 * have completed, regardless of whether other streams are active. In effect,
 * this constrains exclusive ownership of the managed memory region by
 * an active GPU to per-stream activity instead of whole-GPU activity.
 *
 * Accessing memory on the device from streams that are not associated with
 * it will produce undefined results. No error checking is performed by the
 * Unified Memory system to ensure that kernels launched into other streams
 * do not access this region. 
 *
 * It is a program's responsibility to order calls to ::lwdaStreamAttachMemAsync
 * via events, synchronization or other means to ensure legal access to memory
 * at all times. Data visibility and coherency will be changed appropriately
 * for all kernels which follow a stream-association change.
 *
 * If \p stream is destroyed while data is associated with it, the association is
 * removed and the association reverts to the default visibility of the allocation
 * as specified at ::lwdaMallocManaged. For __managed__ variables, the default
 * association is always ::lwdaMemAttachGlobal. Note that destroying a stream is an
 * asynchronous operation, and as a result, the change to default association won't
 * happen until all work in the stream has completed.
 *
 * \param stream  - Stream in which to enqueue the attach operation
 * \param devPtr  - Pointer to memory (must be a pointer to managed memory)
 * \param length  - Length of memory (must be zero, defaults to zero)
 * \param flags   - Must be one of ::lwdaMemAttachGlobal, ::lwdaMemAttachHost or ::lwdaMemAttachSingle (defaults to ::lwdaMemAttachSingle)
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorNotReady,
 * ::lwdaErrorIlwalidValue
 * ::lwdaErrorIlwalidResourceHandle
 * \notefnerr
 *
 * \sa ::lwdaStreamCreate, ::lwdaStreamCreateWithFlags, ::lwdaStreamWaitEvent, ::lwdaStreamSynchronize, ::lwdaStreamAddCallback, ::lwdaStreamDestroy, ::lwdaMallocManaged,
 * ::lwStreamAttachMemAsync
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaStreamAttachMemAsync(lwdaStream_t stream, void *devPtr, size_t length __dv(0), unsigned int flags __dv(lwdaMemAttachSingle));

/** @} */ /* END LWDART_STREAM */

/**
 * \defgroup LWDART_EVENT Event Management
 *
 * ___MANBRIEF___ event management functions of the LWCA runtime API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the event management functions of the LWCA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Creates an event object
 *
 * Creates an event object using ::lwdaEventDefault.
 *
 * \param event - Newly created event
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorLaunchFailure,
 * ::lwdaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa \ref ::lwdaEventCreate(lwdaEvent_t*, unsigned int) "lwdaEventCreate (C++ API)",
 * ::lwdaEventCreateWithFlags, ::lwdaEventRecord, ::lwdaEventQuery,
 * ::lwdaEventSynchronize, ::lwdaEventDestroy, ::lwdaEventElapsedTime,
 * ::lwdaStreamWaitEvent,
 * ::lwEventCreate
 */
extern __host__ lwdaError_t LWDARTAPI lwdaEventCreate(lwdaEvent_t *event);

/**
 * \brief Creates an event object with the specified flags
 *
 * Creates an event object with the specified flags. Valid flags include:
 * - ::lwdaEventDefault: Default event creation flag.
 * - ::lwdaEventBlockingSync: Specifies that event should use blocking
 *   synchronization. A host thread that uses ::lwdaEventSynchronize() to wait
 *   on an event created with this flag will block until the event actually
 *   completes.
 * - ::lwdaEventDisableTiming: Specifies that the created event does not need
 *   to record timing data.  Events created with this flag specified and
 *   the ::lwdaEventBlockingSync flag not specified will provide the best
 *   performance when used with ::lwdaStreamWaitEvent() and ::lwdaEventQuery().
 * - ::lwdaEventInterprocess: Specifies that the created event may be used as an
 *   interprocess event by ::lwdaIpcGetEventHandle(). ::lwdaEventInterprocess must
 *   be specified along with ::lwdaEventDisableTiming.
 *
 * \param event - Newly created event
 * \param flags - Flags for new event
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorLaunchFailure,
 * ::lwdaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa \ref ::lwdaEventCreate(lwdaEvent_t*) "lwdaEventCreate (C API)",
 * ::lwdaEventSynchronize, ::lwdaEventDestroy, ::lwdaEventElapsedTime,
 * ::lwdaStreamWaitEvent,
 * ::lwEventCreate
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaEventCreateWithFlags(lwdaEvent_t *event, unsigned int flags);

/**
 * \brief Records an event
 *
 * Records an event. See note about NULL stream behavior. Since operation
 * is asynchronous, ::lwdaEventQuery() or ::lwdaEventSynchronize() must
 * be used to determine when the event has actually been recorded.
 *
 * If ::lwdaEventRecord() has previously been called on \p event, then this
 * call will overwrite any existing state in \p event.  Any subsequent calls
 * which examine the status of \p event will only examine the completion of
 * this most recent call to ::lwdaEventRecord().
 *
 * \param event  - Event to record
 * \param stream - Stream in which to record event
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorLaunchFailure
 * \note_null_stream
 * \notefnerr
 *
 * \sa \ref ::lwdaEventCreate(lwdaEvent_t*) "lwdaEventCreate (C API)",
 * ::lwdaEventCreateWithFlags, ::lwdaEventQuery,
 * ::lwdaEventSynchronize, ::lwdaEventDestroy, ::lwdaEventElapsedTime,
 * ::lwdaStreamWaitEvent,
 * ::lwEventRecord
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaEventRecord(lwdaEvent_t event, lwdaStream_t stream __dv(0));

/**
 * \brief Queries an event's status
 *
 * Query the status of all device work preceding the most recent call to
 * ::lwdaEventRecord() (in the appropriate compute streams, as specified by the
 * arguments to ::lwdaEventRecord()).
 *
 * If this work has successfully been completed by the device, or if
 * ::lwdaEventRecord() has not been called on \p event, then ::lwdaSuccess is
 * returned. If this work has not yet been completed by the device then
 * ::lwdaErrorNotReady is returned.
 *
 * For the purposes of Unified Memory, a return value of ::lwdaSuccess
 * is equivalent to having called ::lwdaEventSynchronize().
 *
 * \param event - Event to query
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorNotReady,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorLaunchFailure
 * \notefnerr
 *
 * \sa \ref ::lwdaEventCreate(lwdaEvent_t*) "lwdaEventCreate (C API)",
 * ::lwdaEventCreateWithFlags, ::lwdaEventRecord,
 * ::lwdaEventSynchronize, ::lwdaEventDestroy, ::lwdaEventElapsedTime,
 * ::lwEventQuery
 */
extern __host__ lwdaError_t LWDARTAPI lwdaEventQuery(lwdaEvent_t event);

/**
 * \brief Waits for an event to complete
 *
 * Wait until the completion of all device work preceding the most recent
 * call to ::lwdaEventRecord() (in the appropriate compute streams, as specified
 * by the arguments to ::lwdaEventRecord()).
 *
 * If ::lwdaEventRecord() has not been called on \p event, ::lwdaSuccess is
 * returned immediately.
 *
 * Waiting for an event that was created with the ::lwdaEventBlockingSync
 * flag will cause the calling CPU thread to block until the event has
 * been completed by the device.  If the ::lwdaEventBlockingSync flag has
 * not been set, then the CPU thread will busy-wait until the event has
 * been completed by the device.
 *
 * \param event - Event to wait for
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorLaunchFailure
 * \notefnerr
 *
 * \sa \ref ::lwdaEventCreate(lwdaEvent_t*) "lwdaEventCreate (C API)",
 * ::lwdaEventCreateWithFlags, ::lwdaEventRecord,
 * ::lwdaEventQuery, ::lwdaEventDestroy, ::lwdaEventElapsedTime,
 * ::lwEventSynchronize
 */
extern __host__ lwdaError_t LWDARTAPI lwdaEventSynchronize(lwdaEvent_t event);

/**
 * \brief Destroys an event object
 *
 * Destroys the event specified by \p event.
 *
 * In case \p event has been recorded but has not yet been completed
 * when ::lwdaEventDestroy() is called, the function will return immediately and 
 * the resources associated with \p event will be released automatically once
 * the device has completed \p event.
 *
 * \param event - Event to destroy
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorLaunchFailure
 * \notefnerr
 *
 * \sa \ref ::lwdaEventCreate(lwdaEvent_t*) "lwdaEventCreate (C API)",
 * ::lwdaEventCreateWithFlags, ::lwdaEventQuery,
 * ::lwdaEventSynchronize, ::lwdaEventRecord, ::lwdaEventElapsedTime,
 * ::lwEventDestroy
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaEventDestroy(lwdaEvent_t event);

/**
 * \brief Computes the elapsed time between events
 *
 * Computes the elapsed time between two events (in milliseconds with a
 * resolution of around 0.5 microseconds).
 *
 * If either event was last recorded in a non-NULL stream, the resulting time
 * may be greater than expected (even if both used the same stream handle). This
 * happens because the ::lwdaEventRecord() operation takes place asynchronously
 * and there is no guarantee that the measured latency is actually just between
 * the two events. Any number of other different stream operations could execute
 * in between the two measured events, thus altering the timing in a significant
 * way.
 *
 * If ::lwdaEventRecord() has not been called on either event, then
 * ::lwdaErrorIlwalidResourceHandle is returned. If ::lwdaEventRecord() has been
 * called on both events but one or both of them has not yet been completed
 * (that is, ::lwdaEventQuery() would return ::lwdaErrorNotReady on at least one
 * of the events), ::lwdaErrorNotReady is returned. If either event was created
 * with the ::lwdaEventDisableTiming flag, then this function will return
 * ::lwdaErrorIlwalidResourceHandle.
 *
 * \param ms    - Time between \p start and \p end in ms
 * \param start - Starting event
 * \param end   - Ending event
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorNotReady,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorLaunchFailure
 * \notefnerr
 *
 * \sa \ref ::lwdaEventCreate(lwdaEvent_t*) "lwdaEventCreate (C API)",
 * ::lwdaEventCreateWithFlags, ::lwdaEventQuery,
 * ::lwdaEventSynchronize, ::lwdaEventDestroy, ::lwdaEventRecord,
 * ::lwEventElapsedTime
 */
extern __host__ lwdaError_t LWDARTAPI lwdaEventElapsedTime(float *ms, lwdaEvent_t start, lwdaEvent_t end);

/** @} */ /* END LWDART_EVENT */

/**
 * \defgroup LWDART_EXELWTION Exelwtion Control
 *
 * ___MANBRIEF___ exelwtion control functions of the LWCA runtime API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the exelwtion control functions of the LWCA runtime
 * application programming interface.
 *
 * Some functions have overloaded C++ API template versions dolwmented separately in the
 * \ref LWDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Launches a device function
 *
 * The function ilwokes kernel \p func on \p gridDim (\p gridDim.x × \p gridDim.y
 * × \p gridDim.z) grid of blocks. Each block contains \p blockDim (\p blockDim.x ×
 * \p blockDim.y × \p blockDim.z) threads.
 *
 * If the kernel has N parameters the \p args should point to array of N pointers.
 * Each pointer, from <tt>args[0]</tt> to <tt>args[N - 1]</tt>, point to the region
 * of memory from which the actual parameter will be copied.
 *
 * For templated functions, pass the function symbol as follows:
 * func_name<template_arg_0,...,template_arg_N>
 *
 * \p sharedMem sets the amount of dynamic shared memory that will be available to
 * each thread block.
 *
 * \p stream specifies a stream the invocation is associated to.
 *
 * \param func        - Device function symbol
 * \param gridDim     - Grid dimentions
 * \param blockDim    - Block dimentions
 * \param args        - Arguments
 * \param sharedMem   - Shared memory
 * \param stream      - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidDeviceFunction,
 * ::lwdaErrorIlwalidConfiguration,
 * ::lwdaErrorLaunchFailure,
 * ::lwdaErrorLaunchTimeout,
 * ::lwdaErrorLaunchOutOfResources,
 * ::lwdaErrorSharedObjectInitFailed,
 * ::lwdaErrorIlwalidPtx,
 * ::lwdaErrorNoKernelImageForDevice,
 * ::lwdaErrorJitCompilerNotFound
 * \note_null_stream
 * \notefnerr
 *
 * \sa
 * \ref ::lwdaLaunchKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, lwdaStream_t stream) "lwdaLaunchKernel (C++ API)",
 * ::lwLaunchKernel
 */
extern __host__ lwdaError_t LWDARTAPI lwdaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, lwdaStream_t stream);

/**
 * \brief Launches a device function where thread blocks can cooperate and synchronize as they execute
 *
 * The function ilwokes kernel \p func on \p gridDim (\p gridDim.x × \p gridDim.y
 * × \p gridDim.z) grid of blocks. Each block contains \p blockDim (\p blockDim.x ×
 * \p blockDim.y × \p blockDim.z) threads.
 *
 * The device on which this kernel is ilwoked must have a non-zero value for
 * the device attribute ::lwdaDevAttrCooperativeLaunch.
 *
 * The total number of blocks launched cannot exceed the maximum number of blocks per
 * multiprocessor as returned by ::lwdaOclwpancyMaxActiveBlocksPerMultiprocessor (or
 * ::lwdaOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors
 * as specified by the device attribute ::lwdaDevAttrMultiProcessorCount.
 *
 * The kernel cannot make use of LWCA dynamic parallelism.
 *
 * If the kernel has N parameters the \p args should point to array of N pointers.
 * Each pointer, from <tt>args[0]</tt> to <tt>args[N - 1]</tt>, point to the region
 * of memory from which the actual parameter will be copied.
 *
 * For templated functions, pass the function symbol as follows:
 * func_name<template_arg_0,...,template_arg_N>
 *
 * \p sharedMem sets the amount of dynamic shared memory that will be available to
 * each thread block.
 *
 * \p stream specifies a stream the invocation is associated to.
 *
 * \param func        - Device function symbol
 * \param gridDim     - Grid dimentions
 * \param blockDim    - Block dimentions
 * \param args        - Arguments
 * \param sharedMem   - Shared memory
 * \param stream      - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidDeviceFunction,
 * ::lwdaErrorIlwalidConfiguration,
 * ::lwdaErrorLaunchFailure,
 * ::lwdaErrorLaunchTimeout,
 * ::lwdaErrorLaunchOutOfResources,
 * ::lwdaErrorCooperativeLaunchTooLarge,
 * ::lwdaErrorSharedObjectInitFailed
 * \note_null_stream
 * \notefnerr
 *
 * \sa
 * \ref ::lwdaLaunchCooperativeKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, lwdaStream_t stream) "lwdaLaunchCooperativeKernel (C++ API)",
 * ::lwdaLaunchCooperativeKernelMultiDevice,
 * ::lwLaunchCooperativeKernel
 */
extern __host__ lwdaError_t LWDARTAPI lwdaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, lwdaStream_t stream);

/**
 * \brief Launches device functions on multiple devices where thread blocks can cooperate and synchronize as they execute
 *
 * Ilwokes kernels as specified in the \p launchParamsList array where each element
 * of the array specifies all the parameters required to perform a single kernel launch.
 * These kernels can cooperate and synchronize as they execute. The size of the array is
 * specified by \p numDevices.
 *
 * No two kernels can be launched on the same device. All the devices targeted by this
 * multi-device launch must be identical. All devices must have a non-zero value for the
 * device attribute ::lwdaDevAttrCooperativeLaunch.
 *
 * The same kernel must be launched on all devices. Note that any __device__ or __constant__
 * variables are independently instantiated on every device. It is the application's
 * responsiblity to ensure these variables are initialized and used appropriately.
 *
 * The size of the grids as specified in blocks, the size of the blocks themselves and the
 * amount of shared memory used by each thread block must also match across all launched kernels.
 *
 * The streams used to launch these kernels must have been created via either ::lwdaStreamCreate
 * or ::lwdaStreamCreateWithPriority or ::lwdaStreamCreateWithPriority. The NULL stream or
 * ::lwdaStreamLegacy or ::lwdaStreamPerThread cannot be used.
 *
 * The total number of blocks launched per kernel cannot exceed the maximum number of blocks
 * per multiprocessor as returned by ::lwdaOclwpancyMaxActiveBlocksPerMultiprocessor (or
 * ::lwdaOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors
 * as specified by the device attribute ::lwdaDevAttrMultiProcessorCount. Since the
 * total number of blocks launched per device has to match across all devices, the maximum
 * number of blocks that can be launched per device will be limited by the device with the
 * least number of multiprocessors.
 *
 * The kernel cannot make use of LWCA dynamic parallelism.
 *
 * The ::lwdaLaunchParams structure is defined as:
 * \code
        struct lwdaLaunchParams
        {
            void *func;
            dim3 gridDim;
            dim3 blockDim;
            void **args;
            size_t sharedMem;
            lwdaStream_t stream;
        };
 * \endcode
 * where:
 * - ::lwdaLaunchParams::func specifies the kernel to be launched. This same functions must
 *   be launched on all devices. For templated functions, pass the function symbol as follows:
 *   func_name<template_arg_0,...,template_arg_N>
 * - ::lwdaLaunchParams::gridDim specifies the width, height and depth of the grid in blocks.
 *   This must match across all kernels launched.
 * - ::lwdaLaunchParams::blockDim is the width, height and depth of each thread block. This
 *   must match across all kernels launched.
 * - ::lwdaLaunchParams::args specifies the arguments to the kernel. If the kernel has
 *   N parameters then ::lwdaLaunchParams::args should point to array of N pointers. Each
 *   pointer, from <tt>::lwdaLaunchParams::args[0]</tt> to <tt>::lwdaLaunchParams::args[N - 1]</tt>,
 *   point to the region of memory from which the actual parameter will be copied.
 * - ::lwdaLaunchParams::sharedMem is the dynamic shared-memory size per thread block in bytes.
 *   This must match across all kernels launched.
 * - ::lwdaLaunchParams::stream is the handle to the stream to perform the launch in. This cannot
 *   be the NULL stream or ::lwdaStreamLegacy or ::lwdaStreamPerThread.
 *
 * By default, the kernel won't begin exelwtion on any GPU until all prior work in all the specified
 * streams has completed. This behavior can be overridden by specifying the flag
 * ::lwdaCooperativeLaunchMultiDeviceNoPreSync. When this flag is specified, each kernel
 * will only wait for prior work in the stream corresponding to that GPU to complete before it begins
 * exelwtion.
 *
 * Similarly, by default, any subsequent work pushed in any of the specified streams will not begin
 * exelwtion until the kernels on all GPUs have completed. This behavior can be overridden by specifying
 * the flag ::lwdaCooperativeLaunchMultiDeviceNoPostSync. When this flag is specified,
 * any subsequent work pushed in any of the specified streams will only wait for the kernel launched
 * on the GPU corresponding to that stream to complete before it begins exelwtion.
 *
 * \param launchParamsList - List of launch parameters, one per device
 * \param numDevices       - Size of the \p launchParamsList array
 * \param flags            - Flags to control launch behavior
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidDeviceFunction,
 * ::lwdaErrorIlwalidConfiguration,
 * ::lwdaErrorLaunchFailure,
 * ::lwdaErrorLaunchTimeout,
 * ::lwdaErrorLaunchOutOfResources,
 * ::lwdaErrorCooperativeLaunchTooLarge,
 * ::lwdaErrorSharedObjectInitFailed
 * \note_null_stream
 * \notefnerr
 *
 * \sa
 * \ref ::lwdaLaunchCooperativeKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, lwdaStream_t stream) "lwdaLaunchCooperativeKernel (C++ API)",
 * ::lwdaLaunchCooperativeKernel,
 * ::lwLaunchCooperativeKernelMultiDevice
 */
extern __host__ lwdaError_t LWDARTAPI lwdaLaunchCooperativeKernelMultiDevice(struct lwdaLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags  __dv(0));

/**
 * \brief Sets the preferred cache configuration for a device function
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p cacheConfig the preferred cache configuration
 * for the function specified via \p func. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute \p func.
 *
 * \p func is a device function symbol and must be declared as a
 * \c __global__ function. If the specified function does not exist,
 * then ::lwdaErrorIlwalidDeviceFunction is returned. For templated functions,
 * pass the function symbol as follows: func_name<template_arg_0,...,template_arg_N>
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 * - ::lwdaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::lwdaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::lwdaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 * - ::lwdaFuncCachePreferEqual: prefer equal size L1 cache and shared memory
 *
 * \param func        - Device function symbol
 * \param cacheConfig - Requested cache configuration
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorIlwalidDeviceFunction
 * \notefnerr
 * \note_string_api_deprecation2
 *
 * \sa ::lwdaConfigureCall,
 * \ref ::lwdaFuncSetCacheConfig(T*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C++ API)",
 * \ref ::lwdaFuncGetAttributes(struct lwdaFuncAttributes*, const void*) "lwdaFuncGetAttributes (C API)",
 * \ref ::lwdaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, lwdaStream_t stream) "lwdaLaunchKernel (C API)",
 * ::lwdaSetDoubleForDevice,
 * ::lwdaSetDoubleForHost,
 * \ref ::lwdaSetupArgument(const void*, size_t, size_t) "lwdaSetupArgument (C API)",
 * ::lwdaThreadGetCacheConfig,
 * ::lwdaThreadSetCacheConfig,
 * ::lwFuncSetCacheConfig
 */
extern __host__ lwdaError_t LWDARTAPI lwdaFuncSetCacheConfig(const void *func, enum lwdaFuncCache cacheConfig);

/**
 * \brief Sets the shared memory configuration for a device function
 *
 * On devices with configurable shared memory banks, this function will 
 * force all subsequent launches of the specified device function to have
 * the given shared memory bank size configuration. On any given launch of the
 * function, the shared memory configuration of the device will be temporarily
 * changed if needed to suit the function's preferred configuration. Changes in
 * shared memory configuration between subsequent launches of functions, 
 * may introduce a device side synchronization point.
 *
 * Any per-function setting of shared memory bank size set via 
 * ::lwdaFuncSetSharedMemConfig will override the device wide setting set by
 * ::lwdaDeviceSetSharedMemConfig.
 *
 * Changing the shared memory bank size will not increase shared memory usage
 * or affect oclwpancy of kernels, but may have major effects on performance. 
 * Larger bank sizes will allow for greater potential bandwidth to shared memory,
 * but will change what kinds of accesses to shared memory will result in bank 
 * conflicts.
 *
 * This function will do nothing on devices with fixed shared memory bank size.
 *
 * For templated functions, pass the function symbol as follows:
 * func_name<template_arg_0,...,template_arg_N>
 *
 * The supported bank configurations are:
 * - ::lwdaSharedMemBankSizeDefault: use the device's shared memory configuration
 *   when launching this function.
 * - ::lwdaSharedMemBankSizeFourByte: set shared memory bank width to be 
 *   four bytes natively when launching this function.
 * - ::lwdaSharedMemBankSizeEightByte: set shared memory bank width to be eight 
 *   bytes natively when launching this function.
 *
 * \param func   - Device function symbol
 * \param config - Requested shared memory configuration
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorIlwalidDeviceFunction,
 * ::lwdaErrorIlwalidValue,
 * \notefnerr
 * \note_string_api_deprecation2
 *
 * \sa ::lwdaConfigureCall,
 * ::lwdaDeviceSetSharedMemConfig,
 * ::lwdaDeviceGetSharedMemConfig,
 * ::lwdaDeviceSetCacheConfig,
 * ::lwdaDeviceGetCacheConfig,
 * ::lwdaFuncSetCacheConfig,
 * ::lwFuncSetSharedMemConfig
 */
extern __host__ lwdaError_t LWDARTAPI lwdaFuncSetSharedMemConfig(const void *func, enum lwdaSharedMemConfig config);

/**
 * \brief Find out attributes for a given function
 *
 * This function obtains the attributes of a function specified via \p func.
 * \p func is a device function symbol and must be declared as a
 * \c __global__ function. The fetched attributes are placed in \p attr.
 * If the specified function does not exist, then
 * ::lwdaErrorIlwalidDeviceFunction is returned. For templated functions, pass
 * the function symbol as follows: func_name<template_arg_0,...,template_arg_N>
 *
 * Note that some function attributes such as
 * \ref ::lwdaFuncAttributes::maxThreadsPerBlock "maxThreadsPerBlock"
 * may vary based on the device that is lwrrently being used.
 *
 * \param attr - Return pointer to function's attributes
 * \param func - Device function symbol
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorIlwalidDeviceFunction
 * \notefnerr
 * \note_string_api_deprecation2
 *
 * \sa ::lwdaConfigureCall,
 * \ref ::lwdaFuncSetCacheConfig(const void*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C API)",
 * \ref ::lwdaFuncGetAttributes(struct lwdaFuncAttributes*, T*) "lwdaFuncGetAttributes (C++ API)",
 * \ref ::lwdaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, lwdaStream_t stream) "lwdaLaunchKernel (C API)",
 * ::lwdaSetDoubleForDevice,
 * ::lwdaSetDoubleForHost,
 * \ref ::lwdaSetupArgument(const void*, size_t, size_t) "lwdaSetupArgument (C API)",
 * ::lwFuncGetAttribute
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaFuncGetAttributes(struct lwdaFuncAttributes *attr, const void *func);


/**
 * \brief Set attributes for a given function
 *
 * This function sets the attributes of a function specified via \p func.
 * The parameter \p func must be a pointer to a function that exelwtes
 * on the device. The parameter specified by \p func must be declared as a \p __global__
 * function. The enumeration defined by \p attr is set to the value defined by \p value.
 * If the specified function does not exist, then ::lwdaErrorIlwalidDeviceFunction is returned.
 * If the specified attribute cannot be written, or if the value is incorrect, 
 * then ::lwdaErrorIlwalidValue is returned.
 *
 * Valid values for \p attr are:
 * - ::lwdaFuncAttributeMaxDynamicSharedMemorySize - Maximum size of dynamic shared memory per block
 * - ::lwdaFuncAttributePreferredSharedMemoryCarveout - Preferred shared memory-L1 cache split ratio in percent of maximum shared memory
 *
 * \param func  - Function to get attributes of
 * \param attr  - Attribute to set
 * \param value - Value to set
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorIlwalidDeviceFunction,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \ref ::lwdaLaunchKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, lwdaStream_t stream) "lwdaLaunchKernel (C++ API)",
 * \ref ::lwdaFuncSetCacheConfig(T*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C++ API)",
 * \ref ::lwdaFuncGetAttributes(struct lwdaFuncAttributes*, const void*) "lwdaFuncGetAttributes (C API)",
 * ::lwdaSetDoubleForDevice,
 * ::lwdaSetDoubleForHost,
 * \ref ::lwdaSetupArgument(T, size_t) "lwdaSetupArgument (C++ API)"
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaFuncSetAttribute(const void *func, enum lwdaFuncAttribute attr, int value);

/**
 * \brief Colwerts a double argument to be exelwted on a device
 *
 * \param d - Double to colwert
 *
 * \deprecated This function is deprecated as of LWCA 7.5
 *
 * Colwerts the double value of \p d to an internal float representation if
 * the device does not support double arithmetic. If the device does natively
 * support doubles, then this function does nothing.
 *
 * \return
 * ::lwdaSuccess
 * \notefnerr
 *
 * \sa
 * \ref ::lwdaLaunch(const void*) "lwdaLaunch (C API)",
 * \ref ::lwdaFuncSetCacheConfig(const void*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C API)",
 * \ref ::lwdaFuncGetAttributes(struct lwdaFuncAttributes*, const void*) "lwdaFuncGetAttributes (C API)",
 * ::lwdaSetDoubleForHost,
 * \ref ::lwdaSetupArgument(const void*, size_t, size_t) "lwdaSetupArgument (C API)"
 */
extern __host__ lwdaError_t LWDARTAPI lwdaSetDoubleForDevice(double *d);

/**
 * \brief Colwerts a double argument after exelwtion on a device
 *
 * \deprecated This function is deprecated as of LWCA 7.5
 *
 * Colwerts the double value of \p d from a potentially internal float
 * representation if the device does not support double arithmetic. If the
 * device does natively support doubles, then this function does nothing.
 *
 * \param d - Double to colwert
 *
 * \return
 * ::lwdaSuccess
 * \notefnerr
 *
 * \sa
 * \ref ::lwdaLaunch(const void*) "lwdaLaunch (C API)",
 * \ref ::lwdaFuncSetCacheConfig(const void*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C API)",
 * \ref ::lwdaFuncGetAttributes(struct lwdaFuncAttributes*, const void*) "lwdaFuncGetAttributes (C API)",
 * ::lwdaSetDoubleForDevice,
 * \ref ::lwdaSetupArgument(const void*, size_t, size_t) "lwdaSetupArgument (C API)"
 */
extern __host__ lwdaError_t LWDARTAPI lwdaSetDoubleForHost(double *d);

/** @} */ /* END LWDART_EXELWTION */

/**
 * \defgroup LWDART_OCLWPANCY Oclwpancy
 *
 * ___MANBRIEF___ oclwpancy callwlation functions of the LWCA runtime API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the oclwpancy callwlation functions of the LWCA runtime
 * application programming interface.
 *
 * Besides the oclwpancy calculator functions
 * (\ref ::lwdaOclwpancyMaxActiveBlocksPerMultiprocessor and \ref ::lwdaOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags),
 * there are also C++ only oclwpancy-based launch configuration functions dolwmented in
 * \ref LWDART_HIGHLEVEL "C++ API Routines" module.
 *
 * See
 * \ref ::lwdaOclwpancyMaxPotentialBlockSize(int*, int*, T, size_t, int) "lwdaOclwpancyMaxPotentialBlockSize (C++ API)",
 * \ref ::lwdaOclwpancyMaxPotentialBlockSizeWithFlags(int*, int*, T, size_t, int, unsigned int) "lwdaOclwpancyMaxPotentialBlockSize (C++ API)",
 * \ref ::lwdaOclwpancyMaxPotentialBlockSizeVariableSMem(int*, int*, T, UnaryFunction, int) "lwdaOclwpancyMaxPotentialBlockSizeVariableSMem (C++ API)",
 * \ref ::lwdaOclwpancyMaxPotentialBlockSizeVariableSMemWithFlags(int*, int*, T, UnaryFunction, int, unsigned int) "lwdaOclwpancyMaxPotentialBlockSizeVariableSMem (C++ API)"
 *
 * @{
 */

/**
 * \brief Returns oclwpancy for a device function
 *
 * Returns in \p *numBlocks the maximum number of active blocks per
 * streaming multiprocessor for the device function.
 *
 * \param numBlocks       - Returned oclwpancy
 * \param func            - Kernel function for which oclwpancy is callwlated
 * \param blockSize       - Block size the kernel is intended to be launched with
 * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorLwdartUnloading,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorIlwalidDevice,
 * ::lwdaErrorIlwalidDeviceFunction,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorUnknown,
 * \notefnerr
 *
 * \sa ::lwdaOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags,
 * \ref ::lwdaOclwpancyMaxPotentialBlockSize(int*, int*, T, size_t, int) "lwdaOclwpancyMaxPotentialBlockSize (C++ API)",
 * \ref ::lwdaOclwpancyMaxPotentialBlockSizeWithFlags(int*, int*, T, size_t, int, unsigned int) "lwdaOclwpancyMaxPotentialBlockSizeWithFlags (C++ API)",
 * \ref ::lwdaOclwpancyMaxPotentialBlockSizeVariableSMem(int*, int*, T, UnaryFunction, int) "lwdaOclwpancyMaxPotentialBlockSizeVariableSMem (C++ API)",
 * \ref ::lwdaOclwpancyMaxPotentialBlockSizeVariableSMemWithFlags(int*, int*, T, UnaryFunction, int, unsigned int) "lwdaOclwpancyMaxPotentialBlockSizeVariableSMemWithFlags (C++ API)",
 * ::lwOclwpancyMaxActiveBlocksPerMultiprocessor
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaOclwpancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize);

/**
 * \brief Returns oclwpancy for a device function with the specified flags
 *
 * Returns in \p *numBlocks the maximum number of active blocks per
 * streaming multiprocessor for the device function.
 *
 * The \p flags parameter controls how special cases are handled. Valid flags include:
 *
 * - ::lwdaOclwpancyDefault: keeps the default behavior as
 *   ::lwdaOclwpancyMaxActiveBlocksPerMultiprocessor
 *
 * - ::lwdaOclwpancyDisableCachingOverride: This flag suppresses the default behavior
 *   on platform where global caching affects oclwpancy. On such platforms, if caching
 *   is enabled, but per-block SM resource usage would result in zero oclwpancy, the
 *   oclwpancy calculator will callwlate the oclwpancy as if caching is disabled.
 *   Setting this flag makes the oclwpancy calculator to return 0 in such cases.
 *   More information can be found about this feature in the "Unified L1/Texture Cache"
 *   section of the Maxwell tuning guide.
 *
 * \param numBlocks       - Returned oclwpancy
 * \param func            - Kernel function for which oclwpancy is callwlated
 * \param blockSize       - Block size the kernel is intended to be launched with
 * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
 * \param flags           - Requested behavior for the oclwpancy calculator
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorLwdartUnloading,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorIlwalidDevice,
 * ::lwdaErrorIlwalidDeviceFunction,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorUnknown,
 * \notefnerr
 *
 * \sa ::lwdaOclwpancyMaxActiveBlocksPerMultiprocessor,
 * \ref ::lwdaOclwpancyMaxPotentialBlockSize(int*, int*, T, size_t, int) "lwdaOclwpancyMaxPotentialBlockSize (C++ API)",
 * \ref ::lwdaOclwpancyMaxPotentialBlockSizeWithFlags(int*, int*, T, size_t, int, unsigned int) "lwdaOclwpancyMaxPotentialBlockSizeWithFlags (C++ API)",
 * \ref ::lwdaOclwpancyMaxPotentialBlockSizeVariableSMem(int*, int*, T, UnaryFunction, int) "lwdaOclwpancyMaxPotentialBlockSizeVariableSMem (C++ API)",
 * \ref ::lwdaOclwpancyMaxPotentialBlockSizeVariableSMemWithFlags(int*, int*, T, UnaryFunction, int, unsigned int) "lwdaOclwpancyMaxPotentialBlockSizeVariableSMemWithFlags (C++ API)",
 * ::lwOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags);

/** @} */ /* END LWDA_OCLWPANCY */

/**
 * \defgroup LWDART_EXELWTION_DEPRECATED Exelwtion Control [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated exelwtion control functions of the LWCA runtime API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the deprecated exelwtion control functions of the LWCA runtime
 * application programming interface.
 *
 * Some functions have overloaded C++ API template versions dolwmented separately in the
 * \ref LWDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Configure a device-launch
 *
 * \deprecated This function is deprecated as of LWCA 7.0
 *
 * Specifies the grid and block dimensions for the device call to be exelwted
 * similar to the exelwtion configuration syntax. ::lwdaConfigureCall() is
 * stack based. Each call pushes data on top of an exelwtion stack. This data
 * contains the dimension for the grid and thread blocks, together with any
 * arguments for the call.
 *
 * \param gridDim   - Grid dimensions
 * \param blockDim  - Block dimensions
 * \param sharedMem - Shared memory
 * \param stream    - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidConfiguration
 * \note_null_stream
 * \notefnerr
 *
 * \sa
 * \ref ::lwdaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, lwdaStream_t stream) "lwdaLaunchKernel (C API)",
 * \ref ::lwdaFuncSetCacheConfig(const void*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C API)",
 * \ref ::lwdaFuncGetAttributes(struct lwdaFuncAttributes*, const void*) "lwdaFuncGetAttributes (C API)",
 * \ref ::lwdaLaunch(const void*) "lwdaLaunch (C API)",
 * ::lwdaSetDoubleForDevice,
 * ::lwdaSetDoubleForHost,
 * \ref ::lwdaSetupArgument(const void*, size_t, size_t) "lwdaSetupArgument (C API)",
 */
extern __host__ lwdaError_t LWDARTAPI lwdaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), lwdaStream_t stream __dv(0));

/**
 * \brief Configure a device launch
 *
 * \deprecated This function is deprecated as of LWCA 7.0
 *
 * Pushes \p size bytes of the argument pointed to by \p arg at \p offset
 * bytes from the start of the parameter passing area, which starts at
 * offset 0. The arguments are stored in the top of the exelwtion stack.
 * \ref ::lwdaSetupArgument(const void*, size_t, size_t) "lwdaSetupArgument()"
 * must be preceded by a call to ::lwdaConfigureCall().
 *
 * \param arg    - Argument to push for a kernel launch
 * \param size   - Size of argument
 * \param offset - Offset in argument stack to push new arg
 *
 * \return
 * ::lwdaSuccess
 * \notefnerr
 *
 * \ref ::lwdaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, lwdaStream_t stream) "lwdaLaunchKernel (C API)",
 * \ref ::lwdaFuncSetCacheConfig(const void*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C API)",
 * \ref ::lwdaFuncGetAttributes(struct lwdaFuncAttributes*, const void*) "lwdaFuncGetAttributes (C API)",
 * \ref ::lwdaLaunch(const void*) "lwdaLaunch (C API)",
 * ::lwdaSetDoubleForDevice,
 * ::lwdaSetDoubleForHost,
 * \ref ::lwdaSetupArgument(T, size_t) "lwdaSetupArgument (C++ API)",
 */
extern __host__ lwdaError_t LWDARTAPI lwdaSetupArgument(const void *arg, size_t size, size_t offset);

/**
 * \brief Launches a device function
 *
 * \deprecated This function is deprecated as of LWCA 7.0
 *
 * Launches the function \p func on the device. The parameter \p func must
 * be a device function symbol. The parameter specified by \p func must be
 * declared as a \p __global__ function. For templated functions, pass the
 * function symbol as follows: func_name<template_arg_0,...,template_arg_N>
 * \ref ::lwdaLaunch(const void*) "lwdaLaunch()" must be preceded by a call to
 * ::lwdaConfigureCall() since it pops the data that was pushed by
 * ::lwdaConfigureCall() from the exelwtion stack.
 *
 * \param func - Device function symbol
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidDeviceFunction,
 * ::lwdaErrorIlwalidConfiguration,
 * ::lwdaErrorLaunchFailure,
 * ::lwdaErrorLaunchTimeout,
 * ::lwdaErrorLaunchOutOfResources,
 * ::lwdaErrorSharedObjectInitFailed,
 * ::lwdaErrorIlwalidPtx,
 * ::lwdaErrorNoKernelImageForDevice,
 * ::lwdaErrorJitCompilerNotFound
 * \notefnerr
 * \note_string_api_deprecation_50
 *
 * \ref ::lwdaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, lwdaStream_t stream) "lwdaLaunchKernel (C API)",
 * \ref ::lwdaFuncSetCacheConfig(const void*, enum lwdaFuncCache) "lwdaFuncSetCacheConfig (C API)",
 * \ref ::lwdaFuncGetAttributes(struct lwdaFuncAttributes*, const void*) "lwdaFuncGetAttributes (C API)",
 * \ref ::lwdaLaunch(T*) "lwdaLaunch (C++ API)",
 * ::lwdaSetDoubleForDevice,
 * ::lwdaSetDoubleForHost,
 * \ref ::lwdaSetupArgument(const void*, size_t, size_t) "lwdaSetupArgument (C API)",
 * ::lwdaThreadGetCacheConfig,
 * ::lwdaThreadSetCacheConfig
 */
extern __host__ lwdaError_t LWDARTAPI lwdaLaunch(const void *func);


/** @} */ /* END LWDART_EXELWTION_DEPRECATED */


/**
 * \defgroup LWDART_MEMORY Memory Management
 *
 * ___MANBRIEF___ memory management functions of the LWCA runtime API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the memory management functions of the LWCA runtime
 * application programming interface.
 *
 * Some functions have overloaded C++ API template versions dolwmented separately in the
 * \ref LWDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Allocates memory that will be automatically managed by the Unified Memory system
 *
 * Allocates \p size bytes of managed memory on the device and returns in
 * \p *devPtr a pointer to the allocated memory. If the device doesn't support
 * allocating managed memory, ::lwdaErrorNotSupported is returned. Support
 * for managed memory can be queried using the device attribute
 * ::lwdaDevAttrManagedMemory. The allocated memory is suitably
 * aligned for any kind of variable. The memory is not cleared. If \p size
 * is 0, ::lwdaMallocManaged returns ::lwdaErrorIlwalidValue. The pointer
 * is valid on the CPU and on all GPUs in the system that support managed memory.
 * All accesses to this pointer must obey the Unified Memory programming model.
 *
 * \p flags specifies the default stream association for this allocation.
 * \p flags must be one of ::lwdaMemAttachGlobal or ::lwdaMemAttachHost. The
 * default value for \p flags is ::lwdaMemAttachGlobal.
 * If ::lwdaMemAttachGlobal is specified, then this memory is accessible from
 * any stream on any device. If ::lwdaMemAttachHost is specified, then the
 * allocation should not be accessed from devices that have a zero value for the
 * device attribute ::lwdaDevAttrConlwrrentManagedAccess; an explicit call to
 * ::lwdaStreamAttachMemAsync will be required to enable access on such devices.
 *
 * If the association is later changed via ::lwdaStreamAttachMemAsync to
 * a single stream, the default association, as specifed during ::lwdaMallocManaged,
 * is restored when that stream is destroyed. For __managed__ variables, the
 * default association is always ::lwdaMemAttachGlobal. Note that destroying a
 * stream is an asynchronous operation, and as a result, the change to default
 * association won't happen until all work in the stream has completed.
 *
 * Memory allocated with ::lwdaMallocManaged should be released with ::lwdaFree.
 *
 * Device memory oversubscription is possible for GPUs that have a non-zero value for the
 * device attribute ::lwdaDevAttrConlwrrentManagedAccess. Managed memory on
 * such GPUs may be evicted from device memory to host memory at any time by the Unified
 * Memory driver in order to make room for other allocations.
 *
 * In a multi-GPU system where all GPUs have a non-zero value for the device attribute
 * ::lwdaDevAttrConlwrrentManagedAccess, managed memory may not be populated when this
 * API returns and instead may be populated on access. In such systems, managed memory can
 * migrate to any processor's memory at any time. The Unified Memory driver will employ heuristics to
 * maintain data locality and prevent excessive page faults to the extent possible. The application
 * can also guide the driver about memory usage patterns via ::lwdaMemAdvise. The application
 * can also explicitly migrate memory to a desired processor's memory via
 * ::lwdaMemPrefetchAsync.
 *
 * In a multi-GPU system where all of the GPUs have a zero value for the device attribute
 * ::lwdaDevAttrConlwrrentManagedAccess and all the GPUs have peer-to-peer support
 * with each other, the physical storage for managed memory is created on the GPU which is active
 * at the time ::lwdaMallocManaged is called. All other GPUs will reference the data at reduced
 * bandwidth via peer mappings over the PCIe bus. The Unified Memory driver does not migrate
 * memory among such GPUs.
 *
 * In a multi-GPU system where not all GPUs have peer-to-peer support with each other and
 * where the value of the device attribute ::lwdaDevAttrConlwrrentManagedAccess
 * is zero for at least one of those GPUs, the location chosen for physical storage of managed
 * memory is system-dependent.
 * - On Linux, the location chosen will be device memory as long as the current set of active
 * contexts are on devices that either have peer-to-peer support with each other or have a
 * non-zero value for the device attribute ::lwdaDevAttrConlwrrentManagedAccess.
 * If there is an active context on a GPU that does not have a non-zero value for that device
 * attribute and it does not have peer-to-peer support with the other devices that have active
 * contexts on them, then the location for physical storage will be 'zero-copy' or host memory.
 * Note that this means that managed memory that is located in device memory is migrated to
 * host memory if a new context is created on a GPU that doesn't have a non-zero value for
 * the device attribute and does not support peer-to-peer with at least one of the other devices
 * that has an active context. This in turn implies that context creation may fail if there is
 * insufficient host memory to migrate all managed allocations.
 * - On Windows, the physical storage is always created in 'zero-copy' or host memory.
 * All GPUs will reference the data at reduced bandwidth over the PCIe bus. In these
 * cirlwmstances, use of the environment variable LWDA_VISIBLE_DEVICES is recommended to
 * restrict LWCA to only use those GPUs that have peer-to-peer support.
 * Alternatively, users can also set LWDA_MANAGED_FORCE_DEVICE_ALLOC to a non-zero
 * value to force the driver to always use device memory for physical storage.
 * When this environment variable is set to a non-zero value, all devices used in
 * that process that support managed memory have to be peer-to-peer compatible
 * with each other. The error ::lwdaErrorIlwalidDevice will be returned if a device
 * that supports managed memory is used and it is not peer-to-peer compatible with
 * any of the other managed memory supporting devices that were previously used in
 * that process, even if ::lwdaDeviceReset has been called on those devices. These
 * environment variables are described in the LWCA programming guide under the
 * "LWCA environment variables" section.
 *
 * \param devPtr - Pointer to allocated device memory
 * \param size   - Requested allocation size in bytes
 * \param flags  - Must be either ::lwdaMemAttachGlobal or ::lwdaMemAttachHost (defaults to ::lwdaMemAttachGlobal)
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorMemoryAllocation,
 * ::lwdaErrorNotSupported,
 * ::lwdaErrorIlwalidValue
 *
 * \sa ::lwdaMallocPitch, ::lwdaFree, ::lwdaMallocArray, ::lwdaFreeArray,
 * ::lwdaMalloc3D, ::lwdaMalloc3DArray,
 * \ref ::lwdaMallocHost(void**, size_t) "lwdaMallocHost (C API)",
 * ::lwdaFreeHost, ::lwdaHostAlloc, ::lwdaDeviceGetAttribute, ::lwdaStreamAttachMemAsync,
 * ::lwMemAllocManaged
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaMallocManaged(void **devPtr, size_t size, unsigned int flags __dv(lwdaMemAttachGlobal));


/**
 * \brief Allocate memory on the device
 *
 * Allocates \p size bytes of linear memory on the device and returns in
 * \p *devPtr a pointer to the allocated memory. The allocated memory is
 * suitably aligned for any kind of variable. The memory is not cleared.
 * ::lwdaMalloc() returns ::lwdaErrorMemoryAllocation in case of failure.
 *
 * The device version of ::lwdaFree cannot be used with a \p *devPtr
 * allocated using the host API, and vice versa.
 *
 * \param devPtr - Pointer to allocated device memory
 * \param size   - Requested allocation size in bytes
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorMemoryAllocation
 *
 * \sa ::lwdaMallocPitch, ::lwdaFree, ::lwdaMallocArray, ::lwdaFreeArray,
 * ::lwdaMalloc3D, ::lwdaMalloc3DArray,
 * \ref ::lwdaMallocHost(void**, size_t) "lwdaMallocHost (C API)",
 * ::lwdaFreeHost, ::lwdaHostAlloc,
 * ::lwMemAlloc
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaMalloc(void **devPtr, size_t size);

/**
 * \brief Allocates page-locked memory on the host
 *
 * Allocates \p size bytes of host memory that is page-locked and accessible
 * to the device. The driver tracks the virtual memory ranges allocated with
 * this function and automatically accelerates calls to functions such as
 * ::lwdaMemcpy*(). Since the memory can be accessed directly by the device,
 * it can be read or written with much higher bandwidth than pageable memory
 * obtained with functions such as ::malloc(). Allocating excessive amounts of
 * memory with ::lwdaMallocHost() may degrade system performance, since it
 * reduces the amount of memory available to the system for paging. As a
 * result, this function is best used sparingly to allocate staging areas for
 * data exchange between host and device.
 *
 * \param ptr  - Pointer to allocated host memory
 * \param size - Requested allocation size in bytes
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::lwdaMalloc, ::lwdaMallocPitch, ::lwdaMallocArray, ::lwdaMalloc3D,
 * ::lwdaMalloc3DArray, ::lwdaHostAlloc, ::lwdaFree, ::lwdaFreeArray,
 * \ref ::lwdaMallocHost(void**, size_t, unsigned int) "lwdaMallocHost (C++ API)",
 * ::lwdaFreeHost, ::lwdaHostAlloc,
 * ::lwMemAllocHost
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMallocHost(void **ptr, size_t size);

/**
 * \brief Allocates pitched memory on the device
 *
 * Allocates at least \p width (in bytes) * \p height bytes of linear memory
 * on the device and returns in \p *devPtr a pointer to the allocated memory.
 * The function may pad the allocation to ensure that corresponding pointers
 * in any given row will continue to meet the alignment requirements for
 * coalescing as the address is updated from row to row. The pitch returned in
 * \p *pitch by ::lwdaMallocPitch() is the width in bytes of the allocation.
 * The intended usage of \p pitch is as a separate parameter of the allocation,
 * used to compute addresses within the 2D array. Given the row and column of
 * an array element of type \p T, the address is computed as:
 * \code
    T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
   \endcode
 *
 * For allocations of 2D arrays, it is recommended that programmers consider
 * performing pitch allocations using ::lwdaMallocPitch(). Due to pitch
 * alignment restrictions in the hardware, this is especially true if the
 * application will be performing 2D memory copies between different regions
 * of device memory (whether linear memory or LWCA arrays).
 *
 * \param devPtr - Pointer to allocated pitched device memory
 * \param pitch  - Pitch for allocation
 * \param width  - Requested pitched allocation width (in bytes)
 * \param height - Requested pitched allocation height
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::lwdaMalloc, ::lwdaFree, ::lwdaMallocArray, ::lwdaFreeArray,
 * \ref ::lwdaMallocHost(void**, size_t) "lwdaMallocHost (C API)",
 * ::lwdaFreeHost, ::lwdaMalloc3D, ::lwdaMalloc3DArray,
 * ::lwdaHostAlloc,
 * ::lwMemAllocPitch
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);

/**
 * \brief Allocate an array on the device
 *
 * Allocates a LWCA array according to the ::lwdaChannelFormatDesc structure
 * \p desc and returns a handle to the new LWCA array in \p *array.
 *
 * The ::lwdaChannelFormatDesc is defined as:
 * \code
    struct lwdaChannelFormatDesc {
        int x, y, z, w;
    enum lwdaChannelFormatKind f;
    };
    \endcode
 * where ::lwdaChannelFormatKind is one of ::lwdaChannelFormatKindSigned,
 * ::lwdaChannelFormatKindUnsigned, or ::lwdaChannelFormatKindFloat.
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::lwdaArrayDefault: This flag's value is defined to be 0 and provides default array allocation
 * - ::lwdaArraySurfaceLoadStore: Allocates an array that can be read from or written to using a surface reference
 * - ::lwdaArrayTextureGather: This flag indicates that texture gather operations will be performed on the array.
 *
 * \p width and \p height must meet certain size requirements. See ::lwdaMalloc3DArray() for more details.
 *
 * \param array  - Pointer to allocated array in device memory
 * \param desc   - Requested channel format
 * \param width  - Requested array allocation width
 * \param height - Requested array allocation height
 * \param flags  - Requested properties of allocated array
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::lwdaMalloc, ::lwdaMallocPitch, ::lwdaFree, ::lwdaFreeArray,
 * \ref ::lwdaMallocHost(void**, size_t) "lwdaMallocHost (C API)",
 * ::lwdaFreeHost, ::lwdaMalloc3D, ::lwdaMalloc3DArray,
 * ::lwdaHostAlloc,
 * ::lwArrayCreate
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMallocArray(lwdaArray_t *array, const struct lwdaChannelFormatDesc *desc, size_t width, size_t height __dv(0), unsigned int flags __dv(0));

/**
 * \brief Frees memory on the device
 *
 * Frees the memory space pointed to by \p devPtr, which must have been
 * returned by a previous call to ::lwdaMalloc() or ::lwdaMallocPitch().
 * Otherwise, or if ::lwdaFree(\p devPtr) has already been called before,
 * an error is returned. If \p devPtr is 0, no operation is performed.
 * ::lwdaFree() returns ::lwdaErrorIlwalidDevicePointer in case of failure.
 *
 * The device version of ::lwdaFree cannot be used with a \p *devPtr
 * allocated using the host API, and vice versa.
 *
 * \param devPtr - Device pointer to memory to free
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidDevicePointer,
 * ::lwdaErrorInitializationError
 * \notefnerr
 *
 * \sa ::lwdaMalloc, ::lwdaMallocPitch, ::lwdaMallocArray, ::lwdaFreeArray,
 * \ref ::lwdaMallocHost(void**, size_t) "lwdaMallocHost (C API)",
 * ::lwdaFreeHost, ::lwdaMalloc3D, ::lwdaMalloc3DArray,
 * ::lwdaHostAlloc,
 * ::lwMemFree
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaFree(void *devPtr);

/**
 * \brief Frees page-locked memory
 *
 * Frees the memory space pointed to by \p hostPtr, which must have been
 * returned by a previous call to ::lwdaMallocHost() or ::lwdaHostAlloc().
 *
 * \param ptr - Pointer to memory to free
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorInitializationError
 * \notefnerr
 *
 * \sa ::lwdaMalloc, ::lwdaMallocPitch, ::lwdaFree, ::lwdaMallocArray,
 * ::lwdaFreeArray,
 * \ref ::lwdaMallocHost(void**, size_t) "lwdaMallocHost (C API)",
 * ::lwdaMalloc3D, ::lwdaMalloc3DArray, ::lwdaHostAlloc,
 * ::lwMemFreeHost
 */
extern __host__ lwdaError_t LWDARTAPI lwdaFreeHost(void *ptr);

/**
 * \brief Frees an array on the device
 *
 * Frees the LWCA array \p array, which must have been * returned by a
 * previous call to ::lwdaMallocArray(). If ::lwdaFreeArray(\p array) has
 * already been called before, ::lwdaErrorIlwalidValue is returned. If
 * \p devPtr is 0, no operation is performed.
 *
 * \param array - Pointer to array to free
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorInitializationError
 * \notefnerr
 *
 * \sa ::lwdaMalloc, ::lwdaMallocPitch, ::lwdaFree, ::lwdaMallocArray,
 * \ref ::lwdaMallocHost(void**, size_t) "lwdaMallocHost (C API)",
 * ::lwdaFreeHost, ::lwdaHostAlloc,
 * ::lwArrayDestroy
 */
extern __host__ lwdaError_t LWDARTAPI lwdaFreeArray(lwdaArray_t array);

/**
 * \brief Frees a mipmapped array on the device
 *
 * Frees the LWCA mipmapped array \p mipmappedArray, which must have been 
 * returned by a previous call to ::lwdaMallocMipmappedArray(). 
 * If ::lwdaFreeMipmappedArray(\p mipmappedArray) has already been called before,
 * ::lwdaErrorIlwalidValue is returned.
 *
 * \param mipmappedArray - Pointer to mipmapped array to free
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorInitializationError
 * \notefnerr
 *
 * \sa ::lwdaMalloc, ::lwdaMallocPitch, ::lwdaFree, ::lwdaMallocArray,
 * \ref ::lwdaMallocHost(void**, size_t) "lwdaMallocHost (C API)",
 * ::lwdaFreeHost, ::lwdaHostAlloc,
 * ::lwMipmappedArrayDestroy
 */
extern __host__ lwdaError_t LWDARTAPI lwdaFreeMipmappedArray(lwdaMipmappedArray_t mipmappedArray);


/**
 * \brief Allocates page-locked memory on the host
 *
 * Allocates \p size bytes of host memory that is page-locked and accessible
 * to the device. The driver tracks the virtual memory ranges allocated with
 * this function and automatically accelerates calls to functions such as
 * ::lwdaMemcpy(). Since the memory can be accessed directly by the device, it
 * can be read or written with much higher bandwidth than pageable memory
 * obtained with functions such as ::malloc(). Allocating excessive amounts of
 * pinned memory may degrade system performance, since it reduces the amount
 * of memory available to the system for paging. As a result, this function is
 * best used sparingly to allocate staging areas for data exchange between host
 * and device.
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::lwdaHostAllocDefault: This flag's value is defined to be 0 and causes
 * ::lwdaHostAlloc() to emulate ::lwdaMallocHost().
 * - ::lwdaHostAllocPortable: The memory returned by this call will be
 * considered as pinned memory by all LWCA contexts, not just the one that
 * performed the allocation.
 * - ::lwdaHostAllocMapped: Maps the allocation into the LWCA address space.
 * The device pointer to the memory may be obtained by calling
 * ::lwdaHostGetDevicePointer().
 * - ::lwdaHostAllocWriteCombined: Allocates the memory as write-combined (WC).
 * WC memory can be transferred across the PCI Express bus more quickly on some
 * system configurations, but cannot be read efficiently by most CPUs.  WC
 * memory is a good option for buffers that will be written by the CPU and read
 * by the device via mapped pinned memory or host->device transfers.
 *
 * All of these flags are orthogonal to one another: a developer may allocate
 * memory that is portable, mapped and/or write-combined with no restrictions.
 *
 * ::lwdaSetDeviceFlags() must have been called with the ::lwdaDeviceMapHost
 * flag in order for the ::lwdaHostAllocMapped flag to have any effect.
 *
 * The ::lwdaHostAllocMapped flag may be specified on LWCA contexts for devices
 * that do not support mapped pinned memory. The failure is deferred to
 * ::lwdaHostGetDevicePointer() because the memory may be mapped into other
 * LWCA contexts via the ::lwdaHostAllocPortable flag.
 *
 * Memory allocated by this function must be freed with ::lwdaFreeHost().
 *
 * \param pHost - Device pointer to allocated memory
 * \param size  - Requested allocation size in bytes
 * \param flags - Requested properties of allocated memory
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::lwdaSetDeviceFlags,
 * \ref ::lwdaMallocHost(void**, size_t) "lwdaMallocHost (C API)",
 * ::lwdaFreeHost,
 * ::lwMemHostAlloc
 */
extern __host__ lwdaError_t LWDARTAPI lwdaHostAlloc(void **pHost, size_t size, unsigned int flags);

/**
 * \brief Registers an existing host memory range for use by LWCA
 *
 * Page-locks the memory range specified by \p ptr and \p size and maps it
 * for the device(s) as specified by \p flags. This memory range also is added
 * to the same tracking mechanism as ::lwdaHostAlloc() to automatically accelerate
 * calls to functions such as ::lwdaMemcpy(). Since the memory can be accessed 
 * directly by the device, it can be read or written with much higher bandwidth 
 * than pageable memory that has not been registered.  Page-locking excessive
 * amounts of memory may degrade system performance, since it reduces the amount
 * of memory available to the system for paging. As a result, this function is
 * best used sparingly to register staging areas for data exchange between
 * host and device.
 *
 * ::lwdaHostRegister is not supported on non I/O coherent devices.
 *
 * The \p flags parameter enables different options to be specified that
 * affect the allocation, as follows.
 *
 * - ::lwdaHostRegisterDefault: On a system with unified virtual addressing,
 *   the memory will be both mapped and portable.  On a system with no unified
 *   virtual addressing, the memory will be neither mapped nor portable.
 *
 * - ::lwdaHostRegisterPortable: The memory returned by this call will be
 *   considered as pinned memory by all LWCA contexts, not just the one that
 *   performed the allocation.
 *
 * - ::lwdaHostRegisterMapped: Maps the allocation into the LWCA address
 *   space. The device pointer to the memory may be obtained by calling
 *   ::lwdaHostGetDevicePointer().
 *
 * - ::lwdaHostRegisterIoMemory: The passed memory pointer is treated as
 *   pointing to some memory-mapped I/O space, e.g. belonging to a
 *   third-party PCIe device, and it will marked as non cache-coherent and
 *   contiguous.
 *
 * All of these flags are orthogonal to one another: a developer may page-lock
 * memory that is portable or mapped with no restrictions.
 *
 * The LWCA context must have been created with the ::lwdaMapHost flag in
 * order for the ::lwdaHostRegisterMapped flag to have any effect.
 *
 * The ::lwdaHostRegisterMapped flag may be specified on LWCA contexts for
 * devices that do not support mapped pinned memory. The failure is deferred
 * to ::lwdaHostGetDevicePointer() because the memory may be mapped into
 * other LWCA contexts via the ::lwdaHostRegisterPortable flag.
 *
 * For devices that have a non-zero value for the device attribute
 * ::lwdaDevAttrCanUseHostPointerForRegisteredMem, the memory
 * can also be accessed from the device using the host pointer \p ptr.
 * The device pointer returned by ::lwdaHostGetDevicePointer() may or may not
 * match the original host pointer \p ptr and depends on the devices visible to the
 * application. If all devices visible to the application have a non-zero value for the
 * device attribute, the device pointer returned by ::lwdaHostGetDevicePointer()
 * will match the original pointer \p ptr. If any device visible to the application
 * has a zero value for the device attribute, the device pointer returned by
 * ::lwdaHostGetDevicePointer() will not match the original host pointer \p ptr,
 * but it will be suitable for use on all devices provided Unified Virtual Addressing
 * is enabled. In such systems, it is valid to access the memory using either pointer
 * on devices that have a non-zero value for the device attribute. Note however that
 * such devices should access the memory using only of the two pointers and not both.
 *
 * The memory page-locked by this function must be unregistered with ::lwdaHostUnregister().
 *
 * \param ptr   - Host pointer to memory to page-lock
 * \param size  - Size in bytes of the address range to page-lock in bytes
 * \param flags - Flags for allocation request
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorMemoryAllocation,
 * ::lwdaErrorHostMemoryAlreadyRegistered,
 * ::lwdaErrorNotSupported
 * \notefnerr
 *
 * \sa ::lwdaHostUnregister, ::lwdaHostGetFlags, ::lwdaHostGetDevicePointer,
 * ::lwMemHostRegister
 */
extern __host__ lwdaError_t LWDARTAPI lwdaHostRegister(void *ptr, size_t size, unsigned int flags);

/**
 * \brief Unregisters a memory range that was registered with lwdaHostRegister
 *
 * Unmaps the memory range whose base address is specified by \p ptr, and makes
 * it pageable again.
 *
 * The base address must be the same one specified to ::lwdaHostRegister().
 *
 * \param ptr - Host pointer to memory to unregister
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorHostMemoryNotRegistered
 * \notefnerr
 *
 * \sa ::lwdaHostUnregister,
 * ::lwMemHostUnregister
 */
extern __host__ lwdaError_t LWDARTAPI lwdaHostUnregister(void *ptr);

/**
 * \brief Passes back device pointer of mapped host memory allocated by
 * lwdaHostAlloc or registered by lwdaHostRegister
 *
 * Passes back the device pointer corresponding to the mapped, pinned host
 * buffer allocated by ::lwdaHostAlloc() or registered by ::lwdaHostRegister().
 *
 * ::lwdaHostGetDevicePointer() will fail if the ::lwdaDeviceMapHost flag was
 * not specified before deferred context creation oclwrred, or if called on a
 * device that does not support mapped, pinned memory.
 *
 * For devices that have a non-zero value for the device attribute
 * ::lwdaDevAttrCanUseHostPointerForRegisteredMem, the memory
 * can also be accessed from the device using the host pointer \p pHost.
 * The device pointer returned by ::lwdaHostGetDevicePointer() may or may not
 * match the original host pointer \p pHost and depends on the devices visible to the
 * application. If all devices visible to the application have a non-zero value for the
 * device attribute, the device pointer returned by ::lwdaHostGetDevicePointer()
 * will match the original pointer \p pHost. If any device visible to the application
 * has a zero value for the device attribute, the device pointer returned by
 * ::lwdaHostGetDevicePointer() will not match the original host pointer \p pHost,
 * but it will be suitable for use on all devices provided Unified Virtual Addressing
 * is enabled. In such systems, it is valid to access the memory using either pointer
 * on devices that have a non-zero value for the device attribute. Note however that
 * such devices should access the memory using only of the two pointers and not both.
 *
 * \p flags provides for future releases.  For now, it must be set to 0.
 *
 * \param pDevice - Returned device pointer for mapped memory
 * \param pHost   - Requested host pointer mapping
 * \param flags   - Flags for extensions (must be 0 for now)
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::lwdaSetDeviceFlags, ::lwdaHostAlloc,
 * ::lwMemHostGetDevicePointer
 */
extern __host__ lwdaError_t LWDARTAPI lwdaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);

/**
 * \brief Passes back flags used to allocate pinned host memory allocated by
 * lwdaHostAlloc
 *
 * ::lwdaHostGetFlags() will fail if the input pointer does not
 * reside in an address range allocated by ::lwdaHostAlloc().
 *
 * \param pFlags - Returned flags word
 * \param pHost - Host pointer
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \sa ::lwdaHostAlloc,
 * ::lwMemHostGetFlags
 */
extern __host__ lwdaError_t LWDARTAPI lwdaHostGetFlags(unsigned int *pFlags, void *pHost);

/**
 * \brief Allocates logical 1D, 2D, or 3D memory objects on the device
 *
 * Allocates at least \p width * \p height * \p depth bytes of linear memory
 * on the device and returns a ::lwdaPitchedPtr in which \p ptr is a pointer
 * to the allocated memory. The function may pad the allocation to ensure
 * hardware alignment requirements are met. The pitch returned in the \p pitch
 * field of \p pitchedDevPtr is the width in bytes of the allocation.
 *
 * The returned ::lwdaPitchedPtr contains additional fields \p xsize and
 * \p ysize, the logical width and height of the allocation, which are
 * equivalent to the \p width and \p height \p extent parameters provided by
 * the programmer during allocation.
 *
 * For allocations of 2D and 3D objects, it is highly recommended that
 * programmers perform allocations using ::lwdaMalloc3D() or
 * ::lwdaMallocPitch(). Due to alignment restrictions in the hardware, this is
 * especially true if the application will be performing memory copies
 * ilwolving 2D or 3D objects (whether linear memory or LWCA arrays).
 *
 * \param pitchedDevPtr  - Pointer to allocated pitched device memory
 * \param extent         - Requested allocation size (\p width field in bytes)
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::lwdaMallocPitch, ::lwdaFree, ::lwdaMemcpy3D, ::lwdaMemset3D,
 * ::lwdaMalloc3DArray, ::lwdaMallocArray, ::lwdaFreeArray,
 * \ref ::lwdaMallocHost(void**, size_t) "lwdaMallocHost (C API)",
 * ::lwdaFreeHost, ::lwdaHostAlloc, ::make_lwdaPitchedPtr, ::make_lwdaExtent,
 * ::lwMemAllocPitch
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMalloc3D(struct lwdaPitchedPtr* pitchedDevPtr, struct lwdaExtent extent);

/**
 * \brief Allocate an array on the device
 *
 * Allocates a LWCA array according to the ::lwdaChannelFormatDesc structure
 * \p desc and returns a handle to the new LWCA array in \p *array.
 *
 * The ::lwdaChannelFormatDesc is defined as:
 * \code
    struct lwdaChannelFormatDesc {
        int x, y, z, w;
        enum lwdaChannelFormatKind f;
    };
    \endcode
 * where ::lwdaChannelFormatKind is one of ::lwdaChannelFormatKindSigned,
 * ::lwdaChannelFormatKindUnsigned, or ::lwdaChannelFormatKindFloat.
 *
 * ::lwdaMalloc3DArray() can allocate the following:
 *
 * - A 1D array is allocated if the height and depth extents are both zero.
 * - A 2D array is allocated if only the depth extent is zero.
 * - A 3D array is allocated if all three extents are non-zero.
 * - A 1D layered LWCA array is allocated if only the height extent is zero and
 * the lwdaArrayLayered flag is set. Each layer is a 1D array. The number of layers is 
 * determined by the depth extent.
 * - A 2D layered LWCA array is allocated if all three extents are non-zero and 
 * the lwdaArrayLayered flag is set. Each layer is a 2D array. The number of layers is 
 * determined by the depth extent.
 * - A lwbemap LWCA array is allocated if all three extents are non-zero and the
 * lwdaArrayLwbemap flag is set. Width must be equal to height, and depth must be six. A lwbemap is
 * a special type of 2D layered LWCA array, where the six layers represent the six faces of a lwbe. 
 * The order of the six layers in memory is the same as that listed in ::lwdaGraphicsLwbeFace.
 * - A lwbemap layered LWCA array is allocated if all three extents are non-zero, and both,
 * lwdaArrayLwbemap and lwdaArrayLayered flags are set. Width must be equal to height, and depth must be 
 * a multiple of six. A lwbemap layered LWCA array is a special type of 2D layered LWCA array that consists 
 * of a collection of lwbemaps. The first six layers represent the first lwbemap, the next six layers form 
 * the second lwbemap, and so on.
 *
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::lwdaArrayDefault: This flag's value is defined to be 0 and provides default array allocation
 * - ::lwdaArrayLayered: Allocates a layered LWCA array, with the depth extent indicating the number of layers
 * - ::lwdaArrayLwbemap: Allocates a lwbemap LWCA array. Width must be equal to height, and depth must be six.
 *   If the lwdaArrayLayered flag is also set, depth must be a multiple of six.
 * - ::lwdaArraySurfaceLoadStore: Allocates a LWCA array that could be read from or written to using a surface
 *   reference.
 * - ::lwdaArrayTextureGather: This flag indicates that texture gather operations will be performed on the LWCA 
 *   array. Texture gather can only be performed on 2D LWCA arrays.
 *
 * The width, height and depth extents must meet certain size requirements as listed in the following table.
 * All values are specified in elements.
 *
 * Note that 2D LWCA arrays have different size requirements if the ::lwdaArrayTextureGather flag is set. In that
 * case, the valid range for (width, height, depth) is ((1,maxTexture2DGather[0]), (1,maxTexture2DGather[1]), 0).
 *
 * \xmlonly
 * <table outputclass="xmlonly">
 * <tgroup cols="3" colsep="1" rowsep="1">
 * <colspec colname="c1" colwidth="1.0*"/>
 * <colspec colname="c2" colwidth="3.0*"/>
 * <colspec colname="c3" colwidth="3.0*"/>
 * <thead>
 * <row>
 * <entry>LWCA array type</entry>
 * <entry>Valid extents that must always be met {(width range in elements),
 * (height range), (depth range)}</entry>
 * <entry>Valid extents with lwdaArraySurfaceLoadStore set {(width range in
 * elements), (height range), (depth range)}</entry>
 * </row>
 * </thead>
 * <tbody>
 * <row>
 * <entry>1D</entry>
 * <entry>{ (1,maxTexture1D), 0, 0 }</entry>
 * <entry>{ (1,maxSurface1D), 0, 0 }</entry>
 * </row>
 * <row>
 * <entry>2D</entry>
 * <entry>{ (1,maxTexture2D[0]), (1,maxTexture2D[1]), 0 }</entry>
 * <entry>{ (1,maxSurface2D[0]), (1,maxSurface2D[1]), 0 }</entry>
 * </row>
 * <row>
 * <entry>3D</entry>
 * <entry>{ (1,maxTexture3D[0]), (1,maxTexture3D[1]), (1,maxTexture3D[2]) }
 * OR { (1,maxTexture3DAlt[0]), (1,maxTexture3DAlt[1]),
 * (1,maxTexture3DAlt[2]) }</entry>
 * <entry>{ (1,maxSurface3D[0]), (1,maxSurface3D[1]), (1,maxSurface3D[2]) }</entry>
 * </row>
 * <row>
 * <entry>1D Layered</entry>
 * <entry>{ (1,maxTexture1DLayered[0]), 0, (1,maxTexture1DLayered[1]) }</entry>
 * <entry>{ (1,maxSurface1DLayered[0]), 0, (1,maxSurface1DLayered[1]) }</entry>
 * </row>
 * <row>
 * <entry>2D Layered</entry>
 * <entry>{ (1,maxTexture2DLayered[0]), (1,maxTexture2DLayered[1]),
 * (1,maxTexture2DLayered[2]) }</entry>
 * <entry>{ (1,maxSurface2DLayered[0]), (1,maxSurface2DLayered[1]),
 * (1,maxSurface2DLayered[2]) }</entry>
 * </row>
 * <row>
 * <entry>Lwbemap</entry>
 * <entry>{ (1,maxTextureLwbemap), (1,maxTextureLwbemap), 6 }</entry>
 * <entry>{ (1,maxSurfaceLwbemap), (1,maxSurfaceLwbemap), 6 }</entry>
 * </row>
 * <row>
 * <entry>Lwbemap Layered</entry>
 * <entry>{ (1,maxTextureLwbemapLayered[0]), (1,maxTextureLwbemapLayered[0]),
 * (1,maxTextureLwbemapLayered[1]) }</entry>
 * <entry>{ (1,maxSurfaceLwbemapLayered[0]), (1,maxSurfaceLwbemapLayered[0]),
 * (1,maxSurfaceLwbemapLayered[1]) }</entry>
 * </row>
 * </tbody>
 * </tgroup>
 * </table>
 * \endxmlonly
 *
 * \param array  - Pointer to allocated array in device memory
 * \param desc   - Requested channel format
 * \param extent - Requested allocation size (\p width field in elements)
 * \param flags  - Flags for extensions
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::lwdaMalloc3D, ::lwdaMalloc, ::lwdaMallocPitch, ::lwdaFree,
 * ::lwdaFreeArray,
 * \ref ::lwdaMallocHost(void**, size_t) "lwdaMallocHost (C API)",
 * ::lwdaFreeHost, ::lwdaHostAlloc,
 * ::make_lwdaExtent,
 * ::lwArray3DCreate
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMalloc3DArray(lwdaArray_t *array, const struct lwdaChannelFormatDesc* desc, struct lwdaExtent extent, unsigned int flags __dv(0));

/**
 * \brief Allocate a mipmapped array on the device
 *
 * Allocates a LWCA mipmapped array according to the ::lwdaChannelFormatDesc structure
 * \p desc and returns a handle to the new LWCA mipmapped array in \p *mipmappedArray.
 * \p numLevels specifies the number of mipmap levels to be allocated. This value is
 * clamped to the range [1, 1 + floor(log2(max(width, height, depth)))].
 *
 * The ::lwdaChannelFormatDesc is defined as:
 * \code
    struct lwdaChannelFormatDesc {
        int x, y, z, w;
        enum lwdaChannelFormatKind f;
    };
    \endcode
 * where ::lwdaChannelFormatKind is one of ::lwdaChannelFormatKindSigned,
 * ::lwdaChannelFormatKindUnsigned, or ::lwdaChannelFormatKindFloat.
 *
 * ::lwdaMallocMipmappedArray() can allocate the following:
 *
 * - A 1D mipmapped array is allocated if the height and depth extents are both zero.
 * - A 2D mipmapped array is allocated if only the depth extent is zero.
 * - A 3D mipmapped array is allocated if all three extents are non-zero.
 * - A 1D layered LWCA mipmapped array is allocated if only the height extent is zero and
 * the lwdaArrayLayered flag is set. Each layer is a 1D mipmapped array. The number of layers is 
 * determined by the depth extent.
 * - A 2D layered LWCA mipmapped array is allocated if all three extents are non-zero and 
 * the lwdaArrayLayered flag is set. Each layer is a 2D mipmapped array. The number of layers is 
 * determined by the depth extent.
 * - A lwbemap LWCA mipmapped array is allocated if all three extents are non-zero and the
 * lwdaArrayLwbemap flag is set. Width must be equal to height, and depth must be six.
 * The order of the six layers in memory is the same as that listed in ::lwdaGraphicsLwbeFace.
 * - A lwbemap layered LWCA mipmapped array is allocated if all three extents are non-zero, and both,
 * lwdaArrayLwbemap and lwdaArrayLayered flags are set. Width must be equal to height, and depth must be 
 * a multiple of six. A lwbemap layered LWCA mipmapped array is a special type of 2D layered LWCA mipmapped
 * array that consists of a collection of lwbemap mipmapped arrays. The first six layers represent the 
 * first lwbemap mipmapped array, the next six layers form the second lwbemap mipmapped array, and so on.
 *
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::lwdaArrayDefault: This flag's value is defined to be 0 and provides default mipmapped array allocation
 * - ::lwdaArrayLayered: Allocates a layered LWCA mipmapped array, with the depth extent indicating the number of layers
 * - ::lwdaArrayLwbemap: Allocates a lwbemap LWCA mipmapped array. Width must be equal to height, and depth must be six.
 *   If the lwdaArrayLayered flag is also set, depth must be a multiple of six.
 * - ::lwdaArraySurfaceLoadStore: This flag indicates that individual mipmap levels of the LWCA mipmapped array 
 *   will be read from or written to using a surface reference.
 * - ::lwdaArrayTextureGather: This flag indicates that texture gather operations will be performed on the LWCA 
 *   array. Texture gather can only be performed on 2D LWCA mipmapped arrays, and the gather operations are
 *   performed only on the most detailed mipmap level.
 *
 * The width, height and depth extents must meet certain size requirements as listed in the following table.
 * All values are specified in elements.
 *
 * \xmlonly
 * <table outputclass="xmlonly">
 * <tgroup cols="3" colsep="1" rowsep="1">
 * <colspec colname="c1" colwidth="1.0*"/>
 * <colspec colname="c2" colwidth="3.0*"/>
 * <colspec colname="c3" colwidth="3.0*"/>
 * <thead>
 * <row>
 * <entry>LWCA array type</entry>
 * <entry>Valid extents that must always be met {(width range in elements),
 * (height range), (depth range)}</entry>
 * <entry>Valid extents with lwdaArraySurfaceLoadStore set {(width range in
 * elements), (height range), (depth range)}</entry>
 * </row>
 * </thead>
 * <tbody>
 * <row>
 * <entry>1D</entry>
 * <entry>{ (1,maxTexture1DMipmap), 0, 0 }</entry>
 * <entry>{ (1,maxSurface1D), 0, 0 }</entry>
 * </row>
 * <row>
 * <entry>2D</entry>
 * <entry>{ (1,maxTexture2DMipmap[0]), (1,maxTexture2DMipmap[1]), 0 }</entry>
 * <entry>{ (1,maxSurface2D[0]), (1,maxSurface2D[1]), 0 }</entry>
 * </row>
 * <row>
 * <entry>3D</entry>
 * <entry>{ (1,maxTexture3D[0]), (1,maxTexture3D[1]), (1,maxTexture3D[2]) }
 * OR { (1,maxTexture3DAlt[0]), (1,maxTexture3DAlt[1]),
 * (1,maxTexture3DAlt[2]) }</entry>
 * <entry>{ (1,maxSurface3D[0]), (1,maxSurface3D[1]), (1,maxSurface3D[2]) }</entry>
 * </row>
 * <row>
 * <entry>1D Layered</entry>
 * <entry>{ (1,maxTexture1DLayered[0]), 0, (1,maxTexture1DLayered[1]) }</entry>
 * <entry>{ (1,maxSurface1DLayered[0]), 0, (1,maxSurface1DLayered[1]) }</entry>
 * </row>
 * <row>
 * <entry>2D Layered</entry>
 * <entry>{ (1,maxTexture2DLayered[0]), (1,maxTexture2DLayered[1]),
 * (1,maxTexture2DLayered[2]) }</entry>
 * <entry>{ (1,maxSurface2DLayered[0]), (1,maxSurface2DLayered[1]),
 * (1,maxSurface2DLayered[2]) }</entry>
 * </row>
 * <row>
 * <entry>Lwbemap</entry>
 * <entry>{ (1,maxTextureLwbemap), (1,maxTextureLwbemap), 6 }</entry>
 * <entry>{ (1,maxSurfaceLwbemap), (1,maxSurfaceLwbemap), 6 }</entry>
 * </row>
 * <row>
 * <entry>Lwbemap Layered</entry>
 * <entry>{ (1,maxTextureLwbemapLayered[0]), (1,maxTextureLwbemapLayered[0]),
 * (1,maxTextureLwbemapLayered[1]) }</entry>
 * <entry>{ (1,maxSurfaceLwbemapLayered[0]), (1,maxSurfaceLwbemapLayered[0]),
 * (1,maxSurfaceLwbemapLayered[1]) }</entry>
 * </row>
 * </tbody>
 * </tgroup>
 * </table>
 * \endxmlonly
 *
 * \param mipmappedArray  - Pointer to allocated mipmapped array in device memory
 * \param desc            - Requested channel format
 * \param extent          - Requested allocation size (\p width field in elements)
 * \param numLevels       - Number of mipmap levels to allocate
 * \param flags           - Flags for extensions
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorMemoryAllocation
 * \notefnerr
 *
 * \sa ::lwdaMalloc3D, ::lwdaMalloc, ::lwdaMallocPitch, ::lwdaFree,
 * ::lwdaFreeArray,
 * \ref ::lwdaMallocHost(void**, size_t) "lwdaMallocHost (C API)",
 * ::lwdaFreeHost, ::lwdaHostAlloc,
 * ::make_lwdaExtent,
 * ::lwMipmappedArrayCreate
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMallocMipmappedArray(lwdaMipmappedArray_t *mipmappedArray, const struct lwdaChannelFormatDesc* desc, struct lwdaExtent extent, unsigned int numLevels, unsigned int flags __dv(0));

/**
 * \brief Gets a mipmap level of a LWCA mipmapped array
 *
 * Returns in \p *levelArray a LWCA array that represents a single mipmap level
 * of the LWCA mipmapped array \p mipmappedArray.
 *
 * If \p level is greater than the maximum number of levels in this mipmapped array,
 * ::lwdaErrorIlwalidValue is returned.
 *
 * \param levelArray     - Returned mipmap level LWCA array
 * \param mipmappedArray - LWCA mipmapped array
 * \param level          - Mipmap level
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \sa ::lwdaMalloc3D, ::lwdaMalloc, ::lwdaMallocPitch, ::lwdaFree,
 * ::lwdaFreeArray,
 * \ref ::lwdaMallocHost(void**, size_t) "lwdaMallocHost (C API)",
 * ::lwdaFreeHost, ::lwdaHostAlloc,
 * ::make_lwdaExtent,
 * ::lwMipmappedArrayGetLevel
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGetMipmappedArrayLevel(lwdaArray_t *levelArray, lwdaMipmappedArray_const_t mipmappedArray, unsigned int level);

/**
 * \brief Copies data between 3D objects
 *
\code
struct lwdaExtent {
  size_t width;
  size_t height;
  size_t depth;
};
struct lwdaExtent make_lwdaExtent(size_t w, size_t h, size_t d);

struct lwdaPos {
  size_t x;
  size_t y;
  size_t z;
};
struct lwdaPos make_lwdaPos(size_t x, size_t y, size_t z);

struct lwdaMemcpy3DParms {
  lwdaArray_t           srcArray;
  struct lwdaPos        srcPos;
  struct lwdaPitchedPtr srcPtr;
  lwdaArray_t           dstArray;
  struct lwdaPos        dstPos;
  struct lwdaPitchedPtr dstPtr;
  struct lwdaExtent     extent;
  enum lwdaMemcpyKind   kind;
};
\endcode
 *
 * ::lwdaMemcpy3D() copies data betwen two 3D objects. The source and
 * destination objects may be in either host memory, device memory, or a LWCA
 * array. The source, destination, extent, and kind of copy performed is
 * specified by the ::lwdaMemcpy3DParms struct which should be initialized to
 * zero before use:
\code
lwdaMemcpy3DParms myParms = {0};
\endcode
 *
 * The struct passed to ::lwdaMemcpy3D() must specify one of \p srcArray or
 * \p srcPtr and one of \p dstArray or \p dstPtr. Passing more than one
 * non-zero source or destination will cause ::lwdaMemcpy3D() to return an
 * error.
 *
 * The \p srcPos and \p dstPos fields are optional offsets into the source and
 * destination objects and are defined in units of each object's elements. The
 * element for a host or device pointer is assumed to be <b>unsigned char</b>.
 *
 * The \p extent field defines the dimensions of the transferred area in
 * elements. If a LWCA array is participating in the copy, the extent is
 * defined in terms of that array's elements. If no LWCA array is
 * participating in the copy then the extents are defined in elements of
 * <b>unsigned char</b>.
 *
 * The \p kind field defines the direction of the copy. It must be one of
 * ::lwdaMemcpyHostToHost, ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * For ::lwdaMemcpyHostToHost or ::lwdaMemcpyHostToDevice or ::lwdaMemcpyDeviceToHost
 * passed as kind and lwdaArray type passed as source or destination, if the kind
 * implies lwdaArray type to be present on the host, ::lwdaMemcpy3D() will
 * disregard that implication and silently correct the kind based on the fact that
 * lwdaArray type can only be present on the device.
 *
 * If the source and destination are both arrays, ::lwdaMemcpy3D() will return
 * an error if they do not have the same element size.
 *
 * The source and destination object may not overlap. If overlapping source
 * and destination objects are specified, undefined behavior will result.
 *
 * The source object must lie entirely within the region defined by \p srcPos
 * and \p extent. The destination object must lie entirely within the region
 * defined by \p dstPos and \p extent.
 *
 * ::lwdaMemcpy3D() returns an error if the pitch of \p srcPtr or \p dstPtr
 * exceeds the maximum allowed. The pitch of a ::lwdaPitchedPtr allocated
 * with ::lwdaMalloc3D() will always be valid.
 *
 * \param p - 3D memory copy parameters
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidPitchValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwdaMalloc3D, ::lwdaMalloc3DArray, ::lwdaMemset3D, ::lwdaMemcpy3DAsync,
 * ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::make_lwdaExtent, ::make_lwdaPos,
 * ::lwMemcpy3D
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy3D(const struct lwdaMemcpy3DParms *p);

/**
 * \brief Copies memory between devices
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p p.  See the definition of the ::lwdaMemcpy3DPeerParms structure
 * for documentation of its parameters.
 *
 * Note that this function is synchronous with respect to the host only if
 * the source or destination of the transfer is host memory.  Note also 
 * that this copy is serialized with respect to all pending and future 
 * asynchronous work in to the current device, the copy's source device,
 * and the copy's destination device (use ::lwdaMemcpy3DPeerAsync to avoid 
 * this synchronization).
 *
 * \param p - Parameters for the memory copy
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidDevice
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpyPeer, ::lwdaMemcpyAsync, ::lwdaMemcpyPeerAsync,
 * ::lwdaMemcpy3DPeerAsync,
 * ::lwMemcpy3DPeer
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy3DPeer(const struct lwdaMemcpy3DPeerParms *p);

/**
 * \brief Copies data between 3D objects
 *
\code
struct lwdaExtent {
  size_t width;
  size_t height;
  size_t depth;
};
struct lwdaExtent make_lwdaExtent(size_t w, size_t h, size_t d);

struct lwdaPos {
  size_t x;
  size_t y;
  size_t z;
};
struct lwdaPos make_lwdaPos(size_t x, size_t y, size_t z);

struct lwdaMemcpy3DParms {
  lwdaArray_t           srcArray;
  struct lwdaPos        srcPos;
  struct lwdaPitchedPtr srcPtr;
  lwdaArray_t           dstArray;
  struct lwdaPos        dstPos;
  struct lwdaPitchedPtr dstPtr;
  struct lwdaExtent     extent;
  enum lwdaMemcpyKind   kind;
};
\endcode
 *
 * ::lwdaMemcpy3DAsync() copies data betwen two 3D objects. The source and
 * destination objects may be in either host memory, device memory, or a LWCA
 * array. The source, destination, extent, and kind of copy performed is
 * specified by the ::lwdaMemcpy3DParms struct which should be initialized to
 * zero before use:
\code
lwdaMemcpy3DParms myParms = {0};
\endcode
 *
 * The struct passed to ::lwdaMemcpy3DAsync() must specify one of \p srcArray
 * or \p srcPtr and one of \p dstArray or \p dstPtr. Passing more than one
 * non-zero source or destination will cause ::lwdaMemcpy3DAsync() to return an
 * error.
 *
 * The \p srcPos and \p dstPos fields are optional offsets into the source and
 * destination objects and are defined in units of each object's elements. The
 * element for a host or device pointer is assumed to be <b>unsigned char</b>.
 * For LWCA arrays, positions must be in the range [0, 2048) for any
 * dimension.
 *
 * The \p extent field defines the dimensions of the transferred area in
 * elements. If a LWCA array is participating in the copy, the extent is
 * defined in terms of that array's elements. If no LWCA array is
 * participating in the copy then the extents are defined in elements of
 * <b>unsigned char</b>.
 *
 * The \p kind field defines the direction of the copy. It must be one of
 * ::lwdaMemcpyHostToHost, ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * For ::lwdaMemcpyHostToHost or ::lwdaMemcpyHostToDevice or ::lwdaMemcpyDeviceToHost
 * passed as kind and lwdaArray type passed as source or destination, if the kind
 * implies lwdaArray type to be present on the host, ::lwdaMemcpy3DAsync() will
 * disregard that implication and silently correct the kind based on the fact that
 * lwdaArray type can only be present on the device.
 *
 * If the source and destination are both arrays, ::lwdaMemcpy3DAsync() will
 * return an error if they do not have the same element size.
 *
 * The source and destination object may not overlap. If overlapping source
 * and destination objects are specified, undefined behavior will result.
 *
 * The source object must lie entirely within the region defined by \p srcPos
 * and \p extent. The destination object must lie entirely within the region
 * defined by \p dstPos and \p extent.
 *
 * ::lwdaMemcpy3DAsync() returns an error if the pitch of \p srcPtr or
 * \p dstPtr exceeds the maximum allowed. The pitch of a
 * ::lwdaPitchedPtr allocated with ::lwdaMalloc3D() will always be valid.
 *
 * ::lwdaMemcpy3DAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::lwdaMemcpyHostToDevice or ::lwdaMemcpyDeviceToHost and \p stream
 * is non-zero, the copy may overlap with operations in other streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param p      - 3D memory copy parameters
 * \param stream - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidPitchValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwdaMalloc3D, ::lwdaMalloc3DArray, ::lwdaMemset3D, ::lwdaMemcpy3D,
 * ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::make_lwdaExtent, ::make_lwdaPos,
 * ::lwMemcpy3DAsync
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaMemcpy3DAsync(const struct lwdaMemcpy3DParms *p, lwdaStream_t stream __dv(0));

/**
 * \brief Copies memory between devices asynchronously.
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p p.  See the definition of the ::lwdaMemcpy3DPeerParms structure
 * for documentation of its parameters.
 *
 * \param p      - Parameters for the memory copy
 * \param stream - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpyPeer, ::lwdaMemcpyAsync, ::lwdaMemcpyPeerAsync,
 * ::lwdaMemcpy3DPeerAsync,
 * ::lwMemcpy3DPeerAsync
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy3DPeerAsync(const struct lwdaMemcpy3DPeerParms *p, lwdaStream_t stream __dv(0));

/**
 * \brief Gets free and total device memory
 *
 * Returns in \p *free and \p *total respectively, the free and total amount of
 * memory available for allocation by the device in bytes.
 *
 * \param free  - Returned free memory in bytes
 * \param total - Returned total memory in bytes
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorInitializationError,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorLaunchFailure
 * \notefnerr
 *
 * \sa
 * ::lwMemGetInfo
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemGetInfo(size_t *free, size_t *total);

/**
 * \brief Gets info about the specified lwdaArray
 * 
 * Returns in \p *desc, \p *extent and \p *flags respectively, the type, shape 
 * and flags of \p array.
 *
 * Any of \p *desc, \p *extent and \p *flags may be specified as NULL.
 *
 * \param desc   - Returned array type
 * \param extent - Returned array shape. 2D arrays will have depth of zero
 * \param flags  - Returned array flags
 * \param array  - The ::lwdaArray to get info for
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \sa
 * ::lwArrayGetDescriptor,
 * ::lwArray3DGetDescriptor
 */
extern __host__ lwdaError_t LWDARTAPI lwdaArrayGetInfo(struct lwdaChannelFormatDesc *desc, struct lwdaExtent *extent, unsigned int *flags, lwdaArray_t array);

/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * memory area pointed to by \p dst, where \p kind specifies the direction
 * of the copy, and must be one of ::lwdaMemcpyHostToHost,
 * ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing. Calling
 * ::lwdaMemcpy() with dst and src pointers that do not match the direction of
 * the copy results in an undefined behavior.
 *
 * \param dst   - Destination memory address
 * \param src   - Source memory address
 * \param count - Size in bytes to copy
 * \param kind  - Type of transfer
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 *
 * \note_sync
 *
 * \sa ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpyDtoH,
 * ::lwMemcpyHtoD,
 * ::lwMemcpyDtoD,
 * ::lwMemcpy
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy(void *dst, const void *src, size_t count, enum lwdaMemcpyKind kind);

/**
 * \brief Copies memory between two devices
 *
 * Copies memory from one device to memory on another device.  \p dst is the 
 * base device pointer of the destination memory and \p dstDevice is the 
 * destination device.  \p src is the base device pointer of the source memory 
 * and \p srcDevice is the source device.  \p count specifies the number of bytes 
 * to copy.
 *
 * Note that this function is asynchronous with respect to the host, but 
 * serialized with respect all pending and future asynchronous work in to the 
 * current device, \p srcDevice, and \p dstDevice (use ::lwdaMemcpyPeerAsync 
 * to avoid this synchronization).
 *
 * \param dst       - Destination device pointer
 * \param dstDevice - Destination device
 * \param src       - Source device pointer
 * \param srcDevice - Source device
 * \param count     - Size of memory copy in bytes
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidDevice
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpyAsync, ::lwdaMemcpyPeerAsync,
 * ::lwdaMemcpy3DPeerAsync,
 * ::lwMemcpyPeer
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count);

/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * LWCA array \p dst starting at the upper left corner
 * (\p wOffset, \p hOffset), where \p kind specifies the direction
 * of the copy, and must be one of ::lwdaMemcpyHostToHost,
 * ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset
 * \param hOffset - Destination starting Y offset
 * \param src     - Source memory address
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpyHtoA,
 * ::lwMemcpyDtoA
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyToArray(lwdaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum lwdaMemcpyKind kind);

/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the LWCA array \p src starting at the upper
 * left corner (\p wOffset, hOffset) to the memory area pointed to by \p dst,
 * where \p kind specifies the direction of the copy, and must be one of
 * ::lwdaMemcpyHostToHost, ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * \param dst     - Destination memory address
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset
 * \param hOffset - Source starting Y offset
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpyAtoH,
 * ::lwMemcpyAtoD
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyFromArray(void *dst, lwdaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum lwdaMemcpyKind kind);

/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the LWCA array \p src starting at the upper
 * left corner (\p wOffsetSrc, \p hOffsetSrc) to the LWCA array \p dst
 * starting at the upper left corner (\p wOffsetDst, \p hOffsetDst) where
 * \p kind specifies the direction of the copy, and must be one of
 * ::lwdaMemcpyHostToHost, ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * \param dst        - Destination memory address
 * \param wOffsetDst - Destination starting X offset
 * \param hOffsetDst - Destination starting Y offset
 * \param src        - Source memory address
 * \param wOffsetSrc - Source starting X offset
 * \param hOffsetSrc - Source starting Y offset
 * \param count      - Size in bytes to copy
 * \param kind       - Type of transfer
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpyAtoA
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyArrayToArray(lwdaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, lwdaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum lwdaMemcpyKind kind __dv(lwdaMemcpyDeviceToDevice));

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the memory area pointed to by \p dst, where
 * \p kind specifies the direction of the copy, and must be one of
 * ::lwdaMemcpyHostToHost, ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing. \p dpitch and
 * \p spitch are the widths in memory in bytes of the 2D arrays pointed to by
 * \p dst and \p src, including any padding added to the end of each row. The
 * memory areas may not overlap. \p width must not exceed either \p dpitch or
 * \p spitch. Calling ::lwdaMemcpy2D() with \p dst and \p src pointers that do
 * not match the direction of the copy results in an undefined behavior.
 * ::lwdaMemcpy2D() returns an error if \p dpitch or \p spitch exceeds
 * the maximum allowed.
 *
 * \param dst    - Destination memory address
 * \param dpitch - Pitch of destination memory
 * \param src    - Source memory address
 * \param spitch - Pitch of source memory
 * \param width  - Width of matrix transfer (columns in bytes)
 * \param height - Height of matrix transfer (rows)
 * \param kind   - Type of transfer
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidPitchValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpy2D,
 * ::lwMemcpy2DUnaligned
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum lwdaMemcpyKind kind);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the LWCA array \p dst starting at the
 * upper left corner (\p wOffset, \p hOffset) where \p kind specifies the
 * direction of the copy, and must be one of ::lwdaMemcpyHostToHost,
 * ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * \p spitch is the width in memory in bytes of the 2D array pointed to by
 * \p src, including any padding added to the end of each row. \p wOffset +
 * \p width must not exceed the width of the LWCA array \p dst. \p width must
 * not exceed \p spitch. ::lwdaMemcpy2DToArray() returns an error if \p spitch
 * exceeds the maximum allowed.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset
 * \param hOffset - Destination starting Y offset
 * \param src     - Source memory address
 * \param spitch  - Pitch of source memory
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidPitchValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpy2D,
 * ::lwMemcpy2DUnaligned
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy2DToArray(lwdaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum lwdaMemcpyKind kind);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the LWCA
 * array \p srcArray starting at the upper left corner
 * (\p wOffset, \p hOffset) to the memory area pointed to by \p dst, where
 * \p kind specifies the direction of the copy, and must be one of
 * ::lwdaMemcpyHostToHost, ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing. \p dpitch is the
 * width in memory in bytes of the 2D array pointed to by \p dst, including any
 * padding added to the end of each row. \p wOffset + \p width must not exceed
 * the width of the LWCA array \p src. \p width must not exceed \p dpitch.
 * ::lwdaMemcpy2DFromArray() returns an error if \p dpitch exceeds the maximum
 * allowed.
 *
 * \param dst     - Destination memory address
 * \param dpitch  - Pitch of destination memory
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset
 * \param hOffset - Source starting Y offset
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidPitchValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpy2D,
 * ::lwMemcpy2DUnaligned
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy2DFromArray(void *dst, size_t dpitch, lwdaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum lwdaMemcpyKind kind);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the LWCA
 * array \p srcArray starting at the upper left corner
 * (\p wOffsetSrc, \p hOffsetSrc) to the LWCA array \p dst starting at
 * the upper left corner (\p wOffsetDst, \p hOffsetDst), where \p kind
 * specifies the direction of the copy, and must be one of
 * ::lwdaMemcpyHostToHost, ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * \p wOffsetDst + \p width must not exceed the width of the LWCA array \p dst.
 * \p wOffsetSrc + \p width must not exceed the width of the LWCA array \p src.
 *
 * \param dst        - Destination memory address
 * \param wOffsetDst - Destination starting X offset
 * \param hOffsetDst - Destination starting Y offset
 * \param src        - Source memory address
 * \param wOffsetSrc - Source starting X offset
 * \param hOffsetSrc - Source starting Y offset
 * \param width      - Width of matrix transfer (columns in bytes)
 * \param height     - Height of matrix transfer (rows)
 * \param kind       - Type of transfer
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpy2D,
 * ::lwMemcpy2DUnaligned
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy2DArrayToArray(lwdaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, lwdaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum lwdaMemcpyKind kind __dv(lwdaMemcpyDeviceToDevice));

/**
 * \brief Copies data to the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p src
 * to the memory area pointed to by \p offset bytes from the start of symbol
 * \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault.
 * Passing ::lwdaMemcpyDefault is recommended, in which case the type of
 * transfer is inferred from the pointer values. However, ::lwdaMemcpyDefault
 * is only allowed on systems that support unified virtual addressing.
 *
 * \param symbol - Device symbol address
 * \param src    - Source memory address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidSymbol,
 * ::lwdaErrorIlwalidMemcpyDirection,
 * ::lwdaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_sync
 * \note_string_api_deprecation
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpy,
 * ::lwMemcpyHtoD,
 * ::lwMemcpyDtoD
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset __dv(0), enum lwdaMemcpyKind kind __dv(lwdaMemcpyHostToDevice));

/**
 * \brief Copies data from the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p offset bytes
 * from the start of symbol \p symbol to the memory area pointed to by \p dst.
 * The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::lwdaMemcpyDeviceToHost, ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault.
 * Passing ::lwdaMemcpyDefault is recommended, in which case the type of
 * transfer is inferred from the pointer values. However, ::lwdaMemcpyDefault
 * is only allowed on systems that support unified virtual addressing.
 *
 * \param dst    - Destination memory address
 * \param symbol - Device symbol address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidSymbol,
 * ::lwdaErrorIlwalidMemcpyDirection,
 * ::lwdaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_sync
 * \note_string_api_deprecation
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpy,
 * ::lwMemcpyDtoH,
 * ::lwMemcpyDtoD
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset __dv(0), enum lwdaMemcpyKind kind __dv(lwdaMemcpyDeviceToHost));


/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * memory area pointed to by \p dst, where \p kind specifies the
 * direction of the copy, and must be one of ::lwdaMemcpyHostToHost,
 * ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * 
 * The memory areas may not overlap. Calling ::lwdaMemcpyAsync() with \p dst and
 * \p src pointers that do not match the direction of the copy results in an
 * undefined behavior.
 *
 * ::lwdaMemcpyAsync() is asynchronous with respect to the host, so the call
 * may return before the copy is complete. The copy can optionally be
 * associated to a stream by passing a non-zero \p stream argument. If \p kind
 * is ::lwdaMemcpyHostToDevice or ::lwdaMemcpyDeviceToHost and the \p stream is
 * non-zero, the copy may overlap with operations in other streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param dst    - Destination memory address
 * \param src    - Source memory address
 * \param count  - Size in bytes to copy
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync
 * ::lwMemcpyAsync,
 * ::lwMemcpyDtoHAsync,
 * ::lwMemcpyHtoDAsync,
 * ::lwMemcpyDtoDAsync
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaMemcpyAsync(void *dst, const void *src, size_t count, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));

/**
 * \brief Copies memory between two devices asynchronously.
 *
 * Copies memory from one device to memory on another device.  \p dst is the 
 * base device pointer of the destination memory and \p dstDevice is the 
 * destination device.  \p src is the base device pointer of the source memory 
 * and \p srcDevice is the source device.  \p count specifies the number of bytes 
 * to copy.
 *
 * Note that this function is asynchronous with respect to the host and all work
 * on other devices.
 *
 * \param dst       - Destination device pointer
 * \param dstDevice - Destination device
 * \param src       - Source device pointer
 * \param srcDevice - Source device
 * \param count     - Size of memory copy in bytes
 * \param stream    - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpyPeer, ::lwdaMemcpyAsync,
 * ::lwdaMemcpy3DPeerAsync,
 * ::lwMemcpyPeerAsync
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, lwdaStream_t stream __dv(0));

/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * LWCA array \p dst starting at the upper left corner
 * (\p wOffset, \p hOffset), where \p kind specifies the
 * direction of the copy, and must be one of ::lwdaMemcpyHostToHost,
 * ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * ::lwdaMemcpyToArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If \p
 * kind is ::lwdaMemcpyHostToDevice or ::lwdaMemcpyDeviceToHost and \p stream
 * is non-zero, the copy may overlap with operations in other streams.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset
 * \param hOffset - Destination starting Y offset
 * \param src     - Source memory address
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpyHtoAAsync,
 * ::lwMemcpy2DAsync
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyToArrayAsync(lwdaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));

/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the LWCA array \p src starting at the upper
 * left corner (\p wOffset, hOffset) to the memory area pointed to by \p dst,
 * where \p kind specifies the direction of the copy, and must be one of
 * ::lwdaMemcpyHostToHost, ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * ::lwdaMemcpyFromArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If \p
 * kind is ::lwdaMemcpyHostToDevice or ::lwdaMemcpyDeviceToHost and \p stream
 * is non-zero, the copy may overlap with operations in other streams.
 *
 * \param dst     - Destination memory address
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset
 * \param hOffset - Source starting Y offset
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpyAtoHAsync,
 * ::lwMemcpy2DAsync
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyFromArrayAsync(void *dst, lwdaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the memory area pointed to by \p dst, where
 * \p kind specifies the direction of the copy, and must be one of
 * ::lwdaMemcpyHostToHost, ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * \p dpitch and \p spitch are the widths in memory in bytes of the 2D arrays
 * pointed to by \p dst and \p src, including any padding added to the end of
 * each row. The memory areas may not overlap. \p width must not exceed either
 * \p dpitch or \p spitch.
 *
 * Calling ::lwdaMemcpy2DAsync() with \p dst and \p src pointers that do not
 * match the direction of the copy results in an undefined behavior.
 * ::lwdaMemcpy2DAsync() returns an error if \p dpitch or \p spitch is greater
 * than the maximum allowed.
 *
 * ::lwdaMemcpy2DAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::lwdaMemcpyHostToDevice or ::lwdaMemcpyDeviceToHost and
 * \p stream is non-zero, the copy may overlap with operations in other
 * streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param dst    - Destination memory address
 * \param dpitch - Pitch of destination memory
 * \param src    - Source memory address
 * \param spitch - Pitch of source memory
 * \param width  - Width of matrix transfer (columns in bytes)
 * \param height - Height of matrix transfer (rows)
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidPitchValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpy2DAsync
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the LWCA array \p dst starting at the
 * upper left corner (\p wOffset, \p hOffset) where \p kind specifies the
 * direction of the copy, and must be one of ::lwdaMemcpyHostToHost,
 * ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * \p spitch is the width in memory in bytes of the 2D array pointed to by
 * \p src, including any padding added to the end of each row. \p wOffset +
 * \p width must not exceed the width of the LWCA array \p dst. \p width must
 * not exceed \p spitch. ::lwdaMemcpy2DToArrayAsync() returns an error if
 * \p spitch exceeds the maximum allowed.
 *
 * ::lwdaMemcpy2DToArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::lwdaMemcpyHostToDevice or ::lwdaMemcpyDeviceToHost and
 * \p stream is non-zero, the copy may overlap with operations in other
 * streams.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset
 * \param hOffset - Destination starting Y offset
 * \param src     - Source memory address
 * \param spitch  - Pitch of source memory
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidPitchValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpy2DAsync
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy2DToArrayAsync(lwdaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the LWCA
 * array \p srcArray starting at the upper left corner
 * (\p wOffset, \p hOffset) to the memory area pointed to by \p dst, where
 * \p kind specifies the direction of the copy, and must be one of
 * ::lwdaMemcpyHostToHost, ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToHost,
 * ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault. Passing
 * ::lwdaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * \p dpitch is the width in memory in bytes of the 2D
 * array pointed to by \p dst, including any padding added to the end of each
 * row. \p wOffset + \p width must not exceed the width of the LWCA array
 * \p src. \p width must not exceed \p dpitch. ::lwdaMemcpy2DFromArrayAsync()
 * returns an error if \p dpitch exceeds the maximum allowed.
 *
 * ::lwdaMemcpy2DFromArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally be
 * associated to a stream by passing a non-zero \p stream argument. If \p kind
 * is ::lwdaMemcpyHostToDevice or ::lwdaMemcpyDeviceToHost and \p stream is
 * non-zero, the copy may overlap with operations in other streams.
 *
 * \param dst     - Destination memory address
 * \param dpitch  - Pitch of destination memory
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset
 * \param hOffset - Source starting Y offset
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidPitchValue,
 * ::lwdaErrorIlwalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync, ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpy2DAsync
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, lwdaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));

/**
 * \brief Copies data to the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p src
 * to the memory area pointed to by \p offset bytes from the start of symbol
 * \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::lwdaMemcpyHostToDevice, ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault.
 * Passing ::lwdaMemcpyDefault is recommended, in which case the type of transfer
 * is inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * ::lwdaMemcpyToSymbolAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::lwdaMemcpyHostToDevice and \p stream is non-zero, the copy
 * may overlap with operations in other streams.
 *
 * \param symbol - Device symbol address
 * \param src    - Source memory address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidSymbol,
 * ::lwdaErrorIlwalidMemcpyDirection,
 * ::lwdaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_string_api_deprecation
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyFromSymbolAsync,
 * ::lwMemcpyAsync,
 * ::lwMemcpyHtoDAsync,
 * ::lwMemcpyDtoDAsync
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));

/**
 * \brief Copies data from the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p offset bytes
 * from the start of symbol \p symbol to the memory area pointed to by \p dst.
 * The memory areas may not overlap. \p symbol is a variable that resides in
 * global or constant memory space. \p kind can be either
 * ::lwdaMemcpyDeviceToHost, ::lwdaMemcpyDeviceToDevice, or ::lwdaMemcpyDefault.
 * Passing ::lwdaMemcpyDefault is recommended, in which case the type of transfer
 * is inferred from the pointer values. However, ::lwdaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * ::lwdaMemcpyFromSymbolAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally be
 * associated to a stream by passing a non-zero \p stream argument. If \p kind
 * is ::lwdaMemcpyDeviceToHost and \p stream is non-zero, the copy may overlap
 * with operations in other streams.
 *
 * \param dst    - Destination memory address
 * \param symbol - Device symbol address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidSymbol,
 * ::lwdaErrorIlwalidMemcpyDirection,
 * ::lwdaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_string_api_deprecation
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpy2D, ::lwdaMemcpyToArray,
 * ::lwdaMemcpy2DToArray, ::lwdaMemcpyFromArray, ::lwdaMemcpy2DFromArray,
 * ::lwdaMemcpyArrayToArray, ::lwdaMemcpy2DArrayToArray, ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol, ::lwdaMemcpyAsync, ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpyToArrayAsync, ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpyFromArrayAsync, ::lwdaMemcpy2DFromArrayAsync,
 * ::lwdaMemcpyToSymbolAsync,
 * ::lwMemcpyAsync,
 * ::lwMemcpyDtoHAsync,
 * ::lwMemcpyDtoDAsync
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));


/**
 * \brief Initializes or sets device memory to a value
 *
 * Fills the first \p count bytes of the memory area pointed to by \p devPtr
 * with the constant byte value \p value.
 *
 * Note that this function is asynchronous with respect to the host unless
 * \p devPtr refers to pinned host memory.
 *
 * \param devPtr - Pointer to device memory
 * \param value  - Value to set for each byte of specified memory
 * \param count  - Size in bytes to set
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * \notefnerr
 * \note_memset
 *
 * \sa
 * ::lwMemsetD8,
 * ::lwMemsetD16,
 * ::lwMemsetD32
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemset(void *devPtr, int value, size_t count);

/**
 * \brief Initializes or sets device memory to a value
 *
 * Sets to the specified value \p value a matrix (\p height rows of \p width
 * bytes each) pointed to by \p dstPtr. \p pitch is the width in bytes of the
 * 2D array pointed to by \p dstPtr, including any padding added to the end
 * of each row. This function performs fastest when the pitch is one that has
 * been passed back by ::lwdaMallocPitch().
 *
 * Note that this function is asynchronous with respect to the host unless
 * \p devPtr refers to pinned host memory.
 *
 * \param devPtr - Pointer to 2D device memory
 * \param pitch  - Pitch in bytes of 2D device memory
 * \param value  - Value to set for each byte of specified memory
 * \param width  - Width of matrix set (columns in bytes)
 * \param height - Height of matrix set (rows)
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * \notefnerr
 * \note_memset
 *
 * \sa ::lwdaMemset, ::lwdaMemset3D, ::lwdaMemsetAsync,
 * ::lwdaMemset2DAsync, ::lwdaMemset3DAsync,
 * ::lwMemsetD2D8,
 * ::lwMemsetD2D16,
 * ::lwMemsetD2D32
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);

/**
 * \brief Initializes or sets device memory to a value
 *
 * Initializes each element of a 3D array to the specified value \p value.
 * The object to initialize is defined by \p pitchedDevPtr. The \p pitch field
 * of \p pitchedDevPtr is the width in memory in bytes of the 3D array pointed
 * to by \p pitchedDevPtr, including any padding added to the end of each row.
 * The \p xsize field specifies the logical width of each row in bytes, while
 * the \p ysize field specifies the height of each 2D slice in rows.
 *
 * The extents of the initialized region are specified as a \p width in bytes,
 * a \p height in rows, and a \p depth in slices.
 *
 * Extents with \p width greater than or equal to the \p xsize of
 * \p pitchedDevPtr may perform significantly faster than extents narrower
 * than the \p xsize. Secondarily, extents with \p height equal to the
 * \p ysize of \p pitchedDevPtr will perform faster than when the \p height is
 * shorter than the \p ysize.
 *
 * This function performs fastest when the \p pitchedDevPtr has been allocated
 * by ::lwdaMalloc3D().
 *
 * Note that this function is asynchronous with respect to the host unless
 * \p pitchedDevPtr refers to pinned host memory.
 *
 * \param pitchedDevPtr - Pointer to pitched device memory
 * \param value         - Value to set for each byte of specified memory
 * \param extent        - Size parameters for where to set device memory (\p width field in bytes)
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * \notefnerr
 * \note_memset
 *
 * \sa ::lwdaMemset, ::lwdaMemset2D,
 * ::lwdaMemsetAsync, ::lwdaMemset2DAsync, ::lwdaMemset3DAsync,
 * ::lwdaMalloc3D, ::make_lwdaPitchedPtr,
 * ::make_lwdaExtent
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemset3D(struct lwdaPitchedPtr pitchedDevPtr, int value, struct lwdaExtent extent);

/**
 * \brief Initializes or sets device memory to a value
 *
 * Fills the first \p count bytes of the memory area pointed to by \p devPtr
 * with the constant byte value \p value.
 *
 * ::lwdaMemsetAsync() is asynchronous with respect to the host, so
 * the call may return before the memset is complete. The operation can optionally
 * be associated to a stream by passing a non-zero \p stream argument.
 * If \p stream is non-zero, the operation may overlap with operations in other streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param devPtr - Pointer to device memory
 * \param value  - Value to set for each byte of specified memory
 * \param count  - Size in bytes to set
 * \param stream - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::lwdaMemset, ::lwdaMemset2D, ::lwdaMemset3D,
 * ::lwdaMemset2DAsync, ::lwdaMemset3DAsync,
 * ::lwMemsetD8Async,
 * ::lwMemsetD16Async,
 * ::lwMemsetD32Async
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaMemsetAsync(void *devPtr, int value, size_t count, lwdaStream_t stream __dv(0));

/**
 * \brief Initializes or sets device memory to a value
 *
 * Sets to the specified value \p value a matrix (\p height rows of \p width
 * bytes each) pointed to by \p dstPtr. \p pitch is the width in bytes of the
 * 2D array pointed to by \p dstPtr, including any padding added to the end
 * of each row. This function performs fastest when the pitch is one that has
 * been passed back by ::lwdaMallocPitch().
 *
 * ::lwdaMemset2DAsync() is asynchronous with respect to the host, so
 * the call may return before the memset is complete. The operation can optionally
 * be associated to a stream by passing a non-zero \p stream argument.
 * If \p stream is non-zero, the operation may overlap with operations in other streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param devPtr - Pointer to 2D device memory
 * \param pitch  - Pitch in bytes of 2D device memory
 * \param value  - Value to set for each byte of specified memory
 * \param width  - Width of matrix set (columns in bytes)
 * \param height - Height of matrix set (rows)
 * \param stream - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::lwdaMemset, ::lwdaMemset2D, ::lwdaMemset3D,
 * ::lwdaMemsetAsync, ::lwdaMemset3DAsync,
 * ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16Async,
 * ::lwMemsetD2D32Async
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, lwdaStream_t stream __dv(0));

/**
 * \brief Initializes or sets device memory to a value
 *
 * Initializes each element of a 3D array to the specified value \p value.
 * The object to initialize is defined by \p pitchedDevPtr. The \p pitch field
 * of \p pitchedDevPtr is the width in memory in bytes of the 3D array pointed
 * to by \p pitchedDevPtr, including any padding added to the end of each row.
 * The \p xsize field specifies the logical width of each row in bytes, while
 * the \p ysize field specifies the height of each 2D slice in rows.
 *
 * The extents of the initialized region are specified as a \p width in bytes,
 * a \p height in rows, and a \p depth in slices.
 *
 * Extents with \p width greater than or equal to the \p xsize of
 * \p pitchedDevPtr may perform significantly faster than extents narrower
 * than the \p xsize. Secondarily, extents with \p height equal to the
 * \p ysize of \p pitchedDevPtr will perform faster than when the \p height is
 * shorter than the \p ysize.
 *
 * This function performs fastest when the \p pitchedDevPtr has been allocated
 * by ::lwdaMalloc3D().
 *
 * ::lwdaMemset3DAsync() is asynchronous with respect to the host, so
 * the call may return before the memset is complete. The operation can optionally
 * be associated to a stream by passing a non-zero \p stream argument.
 * If \p stream is non-zero, the operation may overlap with operations in other streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param pitchedDevPtr - Pointer to pitched device memory
 * \param value         - Value to set for each byte of specified memory
 * \param extent        - Size parameters for where to set device memory (\p width field in bytes)
 * \param stream - Stream identifier
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::lwdaMemset, ::lwdaMemset2D, ::lwdaMemset3D,
 * ::lwdaMemsetAsync, ::lwdaMemset2DAsync,
 * ::lwdaMalloc3D, ::make_lwdaPitchedPtr,
 * ::make_lwdaExtent
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaMemset3DAsync(struct lwdaPitchedPtr pitchedDevPtr, int value, struct lwdaExtent extent, lwdaStream_t stream __dv(0));

/**
 * \brief Finds the address associated with a LWCA symbol
 *
 * Returns in \p *devPtr the address of symbol \p symbol on the device.
 * \p symbol is a variable that resides in global or constant memory space.
 * If \p symbol cannot be found, or if \p symbol is not declared in the
 * global or constant memory space, \p *devPtr is unchanged and the error
 * ::lwdaErrorIlwalidSymbol is returned.
 *
 * \param devPtr - Return device pointer associated with symbol
 * \param symbol - Device symbol address
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidSymbol,
 * ::lwdaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_string_api_deprecation
 *
 * \sa
 * \ref ::lwdaGetSymbolAddress(void**, const T&) "lwdaGetSymbolAddress (C++ API)",
 * \ref ::lwdaGetSymbolSize(size_t*, const void*) "lwdaGetSymbolSize (C API)",
 * ::lwModuleGetGlobal
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGetSymbolAddress(void **devPtr, const void *symbol);

/**
 * \brief Finds the size of the object associated with a LWCA symbol
 *
 * Returns in \p *size the size of symbol \p symbol. \p symbol is a variable that
 * resides in global or constant memory space. If \p symbol cannot be found, or
 * if \p symbol is not declared in global or constant memory space, \p *size is
 * unchanged and the error ::lwdaErrorIlwalidSymbol is returned.
 *
 * \param size   - Size of object associated with symbol
 * \param symbol - Device symbol address
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidSymbol,
 * ::lwdaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_string_api_deprecation
 *
 * \sa
 * \ref ::lwdaGetSymbolAddress(void**, const void*) "lwdaGetSymbolAddress (C API)",
 * \ref ::lwdaGetSymbolSize(size_t*, const T&) "lwdaGetSymbolSize (C++ API)",
 * ::lwModuleGetGlobal
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGetSymbolSize(size_t *size, const void *symbol);

/**
 * \brief Prefetches memory to the specified destination device
 *
 * Prefetches memory to the specified destination device.  \p devPtr is the 
 * base device pointer of the memory to be prefetched and \p dstDevice is the 
 * destination device. \p count specifies the number of bytes to copy. \p stream
 * is the stream in which the operation is enqueued. The memory range must refer
 * to managed memory allocated via ::lwdaMallocManaged or declared via __managed__ variables.
 *
 * Passing in lwdaCpuDeviceId for \p dstDevice will prefetch the data to host memory. If
 * \p dstDevice is a GPU, then the device attribute ::lwdaDevAttrConlwrrentManagedAccess
 * must be non-zero. Additionally, \p stream must be associated with a device that has a
 * non-zero value for the device attribute ::lwdaDevAttrConlwrrentManagedAccess.
 *
 * The start address and end address of the memory range will be rounded down and rounded up
 * respectively to be aligned to CPU page size before the prefetch operation is enqueued
 * in the stream.
 *
 * If no physical memory has been allocated for this region, then this memory region
 * will be populated and mapped on the destination device. If there's insufficient
 * memory to prefetch the desired region, the Unified Memory driver may evict pages from other
 * ::lwdaMallocManaged allocations to host memory in order to make room. Device memory
 * allocated using ::lwdaMalloc or ::lwdaMallocArray will not be evicted.
 *
 * By default, any mappings to the previous location of the migrated pages are removed and
 * mappings for the new location are only setup on \p dstDevice. The exact behavior however
 * also depends on the settings applied to this memory range via ::lwdaMemAdvise as described
 * below:
 *
 * If ::lwdaMemAdviseSetReadMostly was set on any subset of this memory range,
 * then that subset will create a read-only copy of the pages on \p dstDevice.
 *
 * If ::lwdaMemAdviseSetPreferredLocation was called on any subset of this memory
 * range, then the pages will be migrated to \p dstDevice even if \p dstDevice is not the
 * preferred location of any pages in the memory range.
 *
 * If ::lwdaMemAdviseSetAccessedBy was called on any subset of this memory range,
 * then mappings to those pages from all the appropriate processors are updated to
 * refer to the new location if establishing such a mapping is possible. Otherwise,
 * those mappings are cleared.
 *
 * Note that this API is not required for functionality and only serves to improve performance
 * by allowing the application to migrate data to a suitable location before it is accessed.
 * Memory accesses to this range are always coherent and are allowed even when the data is
 * actively being migrated.
 *
 * Note that this function is asynchronous with respect to the host and all work
 * on other devices.
 *
 * \param devPtr    - Pointer to be prefetched
 * \param count     - Size in bytes
 * \param dstDevice - Destination device to prefetch to
 * \param stream    - Stream to enqueue prefetch operation
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpyPeer, ::lwdaMemcpyAsync,
 * ::lwdaMemcpy3DPeerAsync, ::lwdaMemAdvise,
 * ::lwMemPrefetchAsync
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, lwdaStream_t stream __dv(0));

/**
 * \brief Advise about the usage of a given memory range
 *
 * Advise the Unified Memory subsystem about the usage pattern for the memory range
 * starting at \p devPtr with a size of \p count bytes. The start address and end address of the memory
 * range will be rounded down and rounded up respectively to be aligned to CPU page size before the
 * advice is applied. The memory range must refer to managed memory allocated via ::lwdaMallocManaged
 * or declared via __managed__ variables.
 *
 * The \p advice parameter can take the following values:
 * - ::lwdaMemAdviseSetReadMostly: This implies that the data is mostly going to be read
 * from and only occasionally written to. Any read accesses from any processor to this region will create a
 * read-only copy of at least the accessed pages in that processor's memory. Additionally, if ::lwdaMemPrefetchAsync
 * is called on this region, it will create a read-only copy of the data on the destination processor.
 * If any processor writes to this region, all copies of the corresponding page will be ilwalidated
 * except for the one where the write oclwrred. The \p device argument is ignored for this advice.
 * Note that for a page to be read-duplicated, the accessing processor must either be the CPU or a GPU
 * that has a non-zero value for the device attribute ::lwdaDevAttrConlwrrentManagedAccess.
 * Also, if a context is created on a device that does not have the device attribute
 * ::lwdaDevAttrConlwrrentManagedAccess set, then read-duplication will not occur until
 * all such contexts are destroyed.
 * - ::lwdaMemAdviceUnsetReadMostly: Undoes the effect of ::lwdaMemAdviceReadMostly and also prevents the
 * Unified Memory driver from attempting heuristic read-duplication on the memory range. Any read-duplicated
 * copies of the data will be collapsed into a single copy. The location for the collapsed
 * copy will be the preferred location if the page has a preferred location and one of the read-duplicated
 * copies was resident at that location. Otherwise, the location chosen is arbitrary.
 * - ::lwdaMemAdviseSetPreferredLocation: This advice sets the preferred location for the
 * data to be the memory belonging to \p device. Passing in lwdaCpuDeviceId for \p device sets the
 * preferred location as host memory. If \p device is a GPU, then it must have a non-zero value for the
 * device attribute ::lwdaDevAttrConlwrrentManagedAccess. Setting the preferred location
 * does not cause data to migrate to that location immediately. Instead, it guides the migration policy
 * when a fault oclwrs on that memory region. If the data is already in its preferred location and the
 * faulting processor can establish a mapping without requiring the data to be migrated, then
 * data migration will be avoided. On the other hand, if the data is not in its preferred location
 * or if a direct mapping cannot be established, then it will be migrated to the processor accessing
 * it. It is important to note that setting the preferred location does not prevent data prefetching
 * done using ::lwdaMemPrefetchAsync.
 * Having a preferred location can override the page thrash detection and resolution logic in the Unified
 * Memory driver. Normally, if a page is detected to be constantly thrashing between for example host and device
 * memory, the page may eventually be pinned to host memory by the Unified Memory driver. But
 * if the preferred location is set as device memory, then the page will continue to thrash indefinitely.
 * If ::lwdaMemAdviseSetReadMostly is also set on this memory region or any subset of it, then the
 * policies associated with that advice will override the policies of this advice.
 * - ::lwdaMemAdviseUnsetPreferredLocation: Undoes the effect of ::lwdaMemAdviseSetPreferredLocation
 * and changes the preferred location to none.
 * - ::lwdaMemAdviseSetAccessedBy: This advice implies that the data will be accessed by \p device.
 * Passing in ::lwdaCpuDeviceId for \p device will set the advice for the CPU. If \p device is a GPU, then
 * the device attribute ::lwdaDevAttrConlwrrentManagedAccess must be non-zero.
 * This advice does not cause data migration and has no impact on the location of the data per se. Instead,
 * it causes the data to always be mapped in the specified processor's page tables, as long as the
 * location of the data permits a mapping to be established. If the data gets migrated for any reason,
 * the mappings are updated accordingly.
 * This advice is recommended in scenarios where data locality is not important, but avoiding faults is.
 * Consider for example a system containing multiple GPUs with peer-to-peer access enabled, where the
 * data located on one GPU is occasionally accessed by peer GPUs. In such scenarios, migrating data
 * over to the other GPUs is not as important because the accesses are infrequent and the overhead of
 * migration may be too high. But preventing faults can still help improve performance, and so having
 * a mapping set up in advance is useful. Note that on CPU access of this data, the data may be migrated
 * to host memory because the CPU typically cannot access device memory directly. Any GPU that had the
 * ::lwdaMemAdviceSetAccessedBy flag set for this data will now have its mapping updated to point to the
 * page in host memory.
 * If ::lwdaMemAdviseSetReadMostly is also set on this memory region or any subset of it, then the
 * policies associated with that advice will override the policies of this advice. Additionally, if the
 * preferred location of this memory region or any subset of it is also \p device, then the policies
 * associated with ::lwdaMemAdviseSetPreferredLocation will override the policies of this advice.
 * - ::lwdaMemAdviseUnsetAccessedBy: Undoes the effect of ::lwdaMemAdviseSetAccessedBy. Any mappings to
 * the data from \p device may be removed at any time causing accesses to result in non-fatal page faults.
 *
 * \param devPtr - Pointer to memory to set the advice for
 * \param count  - Size in bytes of the memory range
 * \param advice - Advice to be applied for the specified memory range
 * \param device - Device to apply the advice for
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwdaMemcpy, ::lwdaMemcpyPeer, ::lwdaMemcpyAsync,
 * ::lwdaMemcpy3DPeerAsync, ::lwdaMemPrefetchAsync,
 * ::lwMemAdvise
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemAdvise(const void *devPtr, size_t count, enum lwdaMemoryAdvise advice, int device);

/**
* \brief Query an attribute of a given memory range
*
* Query an attribute about the memory range starting at \p devPtr with a size of \p count bytes. The
* memory range must refer to managed memory allocated via ::lwdaMallocManaged or declared via
* __managed__ variables.
*
* The \p attribute parameter can take the following values:
* - ::lwdaMemRangeAttributeReadMostly: If this attribute is specified, \p data will be interpreted
* as a 32-bit integer, and \p dataSize must be 4. The result returned will be 1 if all pages in the given
* memory range have read-duplication enabled, or 0 otherwise.
* - ::lwdaMemRangeAttributePreferredLocation: If this attribute is specified, \p data will be
* interpreted as a 32-bit integer, and \p dataSize must be 4. The result returned will be a GPU device
* id if all pages in the memory range have that GPU as their preferred location, or it will be lwdaCpuDeviceId
* if all pages in the memory range have the CPU as their preferred location, or it will be lwdaIlwalidDeviceId
* if either all the pages don't have the same preferred location or some of the pages don't have a
* preferred location at all. Note that the actual location of the pages in the memory range at the time of
* the query may be different from the preferred location.
* - ::lwdaMemRangeAttributeAccessedBy: If this attribute is specified, \p data will be interpreted
* as an array of 32-bit integers, and \p dataSize must be a non-zero multiple of 4. The result returned
* will be a list of device ids that had ::lwdaMemAdviceSetAccessedBy set for that entire memory range.
* If any device does not have that advice set for the entire memory range, that device will not be included.
* If \p data is larger than the number of devices that have that advice set for that memory range,
* lwdaIlwalidDeviceId will be returned in all the extra space provided. For ex., if \p dataSize is 12
* (i.e. \p data has 3 elements) and only device 0 has the advice set, then the result returned will be
* { 0, lwdaIlwalidDeviceId, lwdaIlwalidDeviceId }. If \p data is smaller than the number of devices that have
* that advice set, then only as many devices will be returned as can fit in the array. There is no
* guarantee on which specific devices will be returned, however.
* - ::lwdaMemRangeAttributeLastPrefetchLocation: If this attribute is specified, \p data will be
* interpreted as a 32-bit integer, and \p dataSize must be 4. The result returned will be the last location
* to which all pages in the memory range were prefetched explicitly via ::lwdaMemPrefetchAsync. This will either be
* a GPU id or lwdaCpuDeviceId depending on whether the last location for prefetch was a GPU or the CPU
* respectively. If any page in the memory range was never explicitly prefetched or if all pages were not
* prefetched to the same location, lwdaIlwalidDeviceId will be returned. Note that this simply returns the
* last location that the applicaton requested to prefetch the memory range to. It gives no indication as to
* whether the prefetch operation to that location has completed or even begun.
*
* \param data      - A pointers to a memory location where the result
*                    of each attribute query will be written to.
* \param dataSize  - Array containing the size of data
* \param attribute - The attribute to query
* \param devPtr    - Start of the range to query
* \param count     - Size of the range to query
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwdaMemRangeGetAttributes, ::lwdaMemPrefetchAsync,
 * ::lwdaMemAdvise,
 * ::lwMemRangeGetAttribute
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemRangeGetAttribute(void *data, size_t dataSize, enum lwdaMemRangeAttribute attribute, const void *devPtr, size_t count);

/**
 * \brief Query attributes of a given memory range.
 *
 * Query attributes of the memory range starting at \p devPtr with a size of \p count bytes. The
 * memory range must refer to managed memory allocated via ::lwdaMallocManaged or declared via
 * __managed__ variables. The \p attributes array will be interpreted to have \p numAttributes
 * entries. The \p dataSizes array will also be interpreted to have \p numAttributes entries.
 * The results of the query will be stored in \p data.
 *
 * The list of supported attributes are given below. Please refer to ::lwdaMemRangeGetAttribute for
 * attribute descriptions and restrictions.
 *
 * - ::lwdaMemRangeAttributeReadMostly
 * - ::lwdaMemRangeAttributePreferredLocation
 * - ::lwdaMemRangeAttributeAccessedBy
 * - ::lwdaMemRangeAttributeLastPrefetchLocation
 *
 * \param data          - A two-dimensional array containing pointers to memory
 *                        locations where the result of each attribute query will be written to.
 * \param dataSizes     - Array containing the sizes of each result
 * \param attributes    - An array of attributes to query
 *                        (numAttributes and the number of attributes in this array should match)
 * \param numAttributes - Number of attributes to query
 * \param devPtr        - Start of the range to query
 * \param count         - Size of the range to query
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \sa ::lwdaMemRangeGetAttribute, ::lwdaMemAdvise
 * ::lwdaMemPrefetchAsync,
 * ::lwMemRangeGetAttributes
 */
extern __host__ lwdaError_t LWDARTAPI lwdaMemRangeGetAttributes(void **data, size_t *dataSizes, enum lwdaMemRangeAttribute *attributes, size_t numAttributes, const void *devPtr, size_t count);

/** @} */ /* END LWDART_MEMORY */

/**
 * \defgroup LWDART_UNIFIED Unified Addressing
 *
 * ___MANBRIEF___ unified addressing functions of the LWCA runtime API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the unified addressing functions of the LWCA 
 * runtime application programming interface.
 *
 * @{
 *
 * \section LWDART_UNIFIED_overview Overview
 *
 * LWCA devices can share a unified address space with the host.  
 * For these devices there is no distinction between a device
 * pointer and a host pointer -- the same pointer value may be 
 * used to access memory from the host program and from a kernel 
 * running on the device (with exceptions enumerated below).
 *
 * \section LWDART_UNIFIED_support Supported Platforms
 * 
 * Whether or not a device supports unified addressing may be 
 * queried by calling ::lwdaGetDeviceProperties() with the device 
 * property ::lwdaDeviceProp::unifiedAddressing.
 *
 * Unified addressing is automatically enabled in 64-bit processes .
 *
 * Unified addressing is not yet supported on Windows Vista or
 * Windows 7 for devices that do not use the TCC driver model.
 *
 * \section LWDART_UNIFIED_lookup Looking Up Information from Pointer Values
 *
 * It is possible to look up information about the memory which backs a 
 * pointer value.  For instance, one may want to know if a pointer points
 * to host or device memory.  As another example, in the case of device 
 * memory, one may want to know on which LWCA device the memory 
 * resides.  These properties may be queried using the function 
 * ::lwdaPointerGetAttributes()
 *
 * Since pointers are unique, it is not necessary to specify information
 * about the pointers specified to ::lwdaMemcpy() and other copy functions.  
 * The copy direction ::lwdaMemcpyDefault may be used to specify that the 
 * LWCA runtime should infer the location of the pointer from its value.
 *
 * \section LWDART_UNIFIED_automaphost Automatic Mapping of Host Allocated Host Memory
 *
 * All host memory allocated through all devices using ::lwdaMallocHost() and
 * ::lwdaHostAlloc() is always directly accessible from all devices that 
 * support unified addressing.  This is the case regardless of whether or 
 * not the flags ::lwdaHostAllocPortable and ::lwdaHostAllocMapped are 
 * specified.
 *
 * The pointer value through which allocated host memory may be accessed 
 * in kernels on all devices that support unified addressing is the same 
 * as the pointer value through which that memory is accessed on the host.
 * It is not necessary to call ::lwdaHostGetDevicePointer() to get the device 
 * pointer for these allocations.  
 *
 * Note that this is not the case for memory allocated using the flag
 * ::lwdaHostAllocWriteCombined, as dislwssed below.
 *
 * \section LWDART_UNIFIED_autopeerregister Direct Access of Peer Memory
 
 * Upon enabling direct access from a device that supports unified addressing 
 * to another peer device that supports unified addressing using 
 * ::lwdaDeviceEnablePeerAccess() all memory allocated in the peer device using 
 * ::lwdaMalloc() and ::lwdaMallocPitch() will immediately be accessible 
 * by the current device.  The device pointer value through 
 * which any peer's memory may be accessed in the current device 
 * is the same pointer value through which that memory may be 
 * accessed from the peer device. 
 *
 * \section LWDART_UNIFIED_exceptions Exceptions, Disjoint Addressing
 * 
 * Not all memory may be accessed on devices through the same pointer
 * value through which they are accessed on the host.  These exceptions
 * are host memory registered using ::lwdaHostRegister() and host memory
 * allocated using the flag ::lwdaHostAllocWriteCombined.  For these 
 * exceptions, there exists a distinct host and device address for the
 * memory.  The device address is guaranteed to not overlap any valid host
 * pointer range and is guaranteed to have the same value across all devices
 * that support unified addressing.  
 * 
 * This device address may be queried using ::lwdaHostGetDevicePointer() 
 * when a device using unified addressing is current.  Either the host 
 * or the unified device pointer value may be used to refer to this memory 
 * in ::lwdaMemcpy() and similar functions using the ::lwdaMemcpyDefault 
 * memory direction.
 *
 */

/**
 * \brief Returns attributes about a specified pointer
 *
 * Returns in \p *attributes the attributes of the pointer \p ptr.
 * If pointer was not allocated in, mapped by or registered with context
 * supporting unified addressing ::lwdaErrorIlwalidValue is returned.
 *
 * The ::lwdaPointerAttributes structure is defined as:
 * \code
    struct lwdaPointerAttributes {
        enum lwdaMemoryType memoryType;
        int device;
        void *devicePointer;
        void *hostPointer;
        int isManaged;
    }
    \endcode
 * In this structure, the individual fields mean
 *
 * - \ref ::lwdaPointerAttributes::memoryType "memoryType" identifies the physical 
 *   location of the memory associated with pointer \p ptr.  It can be
 *   ::lwdaMemoryTypeHost for host memory or ::lwdaMemoryTypeDevice for device
 *   memory.
 *
 * - \ref ::lwdaPointerAttributes::device "device" is the device against which
 *   \p ptr was allocated.  If \p ptr has memory type ::lwdaMemoryTypeDevice
 *   then this identifies the device on which the memory referred to by \p ptr
 *   physically resides.  If \p ptr has memory type ::lwdaMemoryTypeHost then this
 *   identifies the device which was current when the allocation was made
 *   (and if that device is deinitialized then this allocation will vanish
 *   with that device's state).
 *
 * - \ref ::lwdaPointerAttributes::devicePointer "devicePointer" is
 *   the device pointer alias through which the memory referred to by \p ptr
 *   may be accessed on the current device.
 *   If the memory referred to by \p ptr cannot be accessed directly by the 
 *   current device then this is NULL.  
 *
 * - \ref ::lwdaPointerAttributes::hostPointer "hostPointer" is
 *   the host pointer alias through which the memory referred to by \p ptr
 *   may be accessed on the host.
 *   If the memory referred to by \p ptr cannot be accessed directly by the
 *   host then this is NULL.
 *
 * - \ref ::lwdaPointerAttributes::isManaged "isManaged" indicates if
 *   the pointer \p ptr points to managed memory or not.
 *
 * \param attributes - Attributes for the specified pointer
 * \param ptr        - Pointer to get attributes for
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidDevice,
 * ::lwdaErrorIlwalidValue
 *
 * \sa ::lwdaGetDeviceCount, ::lwdaGetDevice, ::lwdaSetDevice,
 * ::lwdaChooseDevice,
 * ::lwPointerGetAttributes
 */
extern __host__ lwdaError_t LWDARTAPI lwdaPointerGetAttributes(struct lwdaPointerAttributes *attributes, const void *ptr);

/** @} */ /* END LWDART_UNIFIED */

/**
 * \defgroup LWDART_PEER Peer Device Memory Access
 *
 * ___MANBRIEF___ peer device memory access functions of the LWCA runtime API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the peer device memory access functions of the LWCA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Queries if a device may directly access a peer device's memory.
 *
 * Returns in \p *canAccessPeer a value of 1 if device \p device is capable of
 * directly accessing memory from \p peerDevice and 0 otherwise.  If direct
 * access of \p peerDevice from \p device is possible, then access may be
 * enabled by calling ::lwdaDeviceEnablePeerAccess().
 *
 * \param canAccessPeer - Returned access capability
 * \param device        - Device from which allocations on \p peerDevice are to
 *                        be directly accessed.
 * \param peerDevice    - Device on which the allocations to be directly accessed 
 *                        by \p device reside.
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidDevice
 * \notefnerr
 *
 * \sa ::lwdaDeviceEnablePeerAccess,
 * ::lwdaDeviceDisablePeerAccess,
 * ::lwDeviceCanAccessPeer
 */
extern __host__ lwdaError_t LWDARTAPI lwdaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice);

/**
 * \brief Enables direct access to memory allocations on a peer device.
 *
 * On success, all allocations from \p peerDevice will immediately be accessible by
 * the current device.  They will remain accessible until access is explicitly
 * disabled using ::lwdaDeviceDisablePeerAccess() or either device is reset using
 * ::lwdaDeviceReset().
 *
 * Note that access granted by this call is unidirectional and that in order to access
 * memory on the current device from \p peerDevice, a separate symmetric call 
 * to ::lwdaDeviceEnablePeerAccess() is required.
 *
 * Each device can support a system-wide maximum of eight peer connections.
 *
 * Peer access is not supported in 32 bit applications.
 *
 * Returns ::lwdaErrorIlwalidDevice if ::lwdaDeviceCanAccessPeer() indicates
 * that the current device cannot directly access memory from \p peerDevice.
 *
 * Returns ::lwdaErrorPeerAccessAlreadyEnabled if direct access of
 * \p peerDevice from the current device has already been enabled.
 *
 * Returns ::lwdaErrorIlwalidValue if \p flags is not 0.
 *
 * \param peerDevice  - Peer device to enable direct access to from the current device
 * \param flags       - Reserved for future use and must be set to 0
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidDevice,
 * ::lwdaErrorPeerAccessAlreadyEnabled,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \sa ::lwdaDeviceCanAccessPeer,
 * ::lwdaDeviceDisablePeerAccess,
 * ::lwCtxEnablePeerAccess
 */
extern __host__ lwdaError_t LWDARTAPI lwdaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);

/**
 * \brief Disables direct access to memory allocations on a peer device.
 *
 * Returns ::lwdaErrorPeerAccessNotEnabled if direct access to memory on
 * \p peerDevice has not yet been enabled from the current device.
 *
 * \param peerDevice - Peer device to disable direct access to
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorPeerAccessNotEnabled,
 * ::lwdaErrorIlwalidDevice
 * \notefnerr
 *
 * \sa ::lwdaDeviceCanAccessPeer,
 * ::lwdaDeviceEnablePeerAccess,
 * ::lwCtxDisablePeerAccess
 */
extern __host__ lwdaError_t LWDARTAPI lwdaDeviceDisablePeerAccess(int peerDevice);

/** @} */ /* END LWDART_PEER */

/** \defgroup LWDART_OPENGL OpenGL Interoperability */

/** \defgroup LWDART_OPENGL_DEPRECATED OpenGL Interoperability [DEPRECATED] */

/** \defgroup LWDART_D3D9 Direct3D 9 Interoperability */

/** \defgroup LWDART_D3D9_DEPRECATED Direct3D 9 Interoperability [DEPRECATED] */

/** \defgroup LWDART_D3D10 Direct3D 10 Interoperability */

/** \defgroup LWDART_D3D10_DEPRECATED Direct3D 10 Interoperability [DEPRECATED] */

/** \defgroup LWDART_D3D11 Direct3D 11 Interoperability */

/** \defgroup LWDART_D3D11_DEPRECATED Direct3D 11 Interoperability [DEPRECATED] */

/** \defgroup LWDART_VDPAU VDPAU Interoperability */

/** \defgroup LWDART_EGL EGL Interoperability */

/**
 * \defgroup LWDART_INTEROP Graphics Interoperability
 *
 * ___MANBRIEF___ graphics interoperability functions of the LWCA runtime API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the graphics interoperability functions of the LWCA
 * runtime application programming interface.
 *
 * @{
 */

/**
 * \brief Unregisters a graphics resource for access by LWCA
 *
 * Unregisters the graphics resource \p resource so it is not accessible by
 * LWCA unless registered again.
 *
 * If \p resource is invalid then ::lwdaErrorIlwalidResourceHandle is
 * returned.
 *
 * \param resource - Resource to unregister
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorUnknown
 * \notefnerr
 *
 * \sa
 * ::lwdaGraphicsD3D9RegisterResource,
 * ::lwdaGraphicsD3D10RegisterResource,
 * ::lwdaGraphicsD3D11RegisterResource,
 * ::lwdaGraphicsGLRegisterBuffer,
 * ::lwdaGraphicsGLRegisterImage,
 * ::lwGraphicsUnregisterResource
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGraphicsUnregisterResource(lwdaGraphicsResource_t resource);

/**
 * \brief Set usage flags for mapping a graphics resource
 *
 * Set \p flags for mapping the graphics resource \p resource.
 *
 * Changes to \p flags will take effect the next time \p resource is mapped.
 * The \p flags argument may be any of the following:
 * - ::lwdaGraphicsMapFlagsNone: Specifies no hints about how \p resource will
 *     be used. It is therefore assumed that LWCA may read from or write to \p resource.
 * - ::lwdaGraphicsMapFlagsReadOnly: Specifies that LWCA will not write to \p resource.
 * - ::lwdaGraphicsMapFlagsWriteDiscard: Specifies LWCA will not read from \p resource and will
 *   write over the entire contents of \p resource, so none of the data
 *   previously stored in \p resource will be preserved.
 *
 * If \p resource is presently mapped for access by LWCA then ::lwdaErrorUnknown is returned.
 * If \p flags is not one of the above values then ::lwdaErrorIlwalidValue is returned.
 *
 * \param resource - Registered resource to set flags for
 * \param flags    - Parameters for resource mapping
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorUnknown,
 * \notefnerr
 *
 * \sa
 * ::lwdaGraphicsMapResources,
 * ::lwGraphicsResourceSetMapFlags
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGraphicsResourceSetMapFlags(lwdaGraphicsResource_t resource, unsigned int flags);

/**
 * \brief Map graphics resources for access by LWCA
 *
 * Maps the \p count graphics resources in \p resources for access by LWCA.
 *
 * The resources in \p resources may be accessed by LWCA until they
 * are unmapped. The graphics API from which \p resources were registered
 * should not access any resources while they are mapped by LWCA. If an
 * application does so, the results are undefined.
 *
 * This function provides the synchronization guarantee that any graphics calls
 * issued before ::lwdaGraphicsMapResources() will complete before any subsequent LWCA
 * work issued in \p stream begins.
 *
 * If \p resources contains any duplicate entries then ::lwdaErrorIlwalidResourceHandle
 * is returned. If any of \p resources are presently mapped for access by
 * LWCA then ::lwdaErrorUnknown is returned.
 *
 * \param count     - Number of resources to map
 * \param resources - Resources to map for LWCA
 * \param stream    - Stream for synchronization
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorUnknown
 * \note_null_stream
 * \notefnerr
 *
 * \sa
 * ::lwdaGraphicsResourceGetMappedPointer,
 * ::lwdaGraphicsSubResourceGetMappedArray,
 * ::lwdaGraphicsUnmapResources,
 * ::lwGraphicsMapResources
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGraphicsMapResources(int count, lwdaGraphicsResource_t *resources, lwdaStream_t stream __dv(0));

/**
 * \brief Unmap graphics resources.
 *
 * Unmaps the \p count graphics resources in \p resources.
 *
 * Once unmapped, the resources in \p resources may not be accessed by LWCA
 * until they are mapped again.
 *
 * This function provides the synchronization guarantee that any LWCA work issued
 * in \p stream before ::lwdaGraphicsUnmapResources() will complete before any
 * subsequently issued graphics work begins.
 *
 * If \p resources contains any duplicate entries then ::lwdaErrorIlwalidResourceHandle
 * is returned. If any of \p resources are not presently mapped for access by
 * LWCA then ::lwdaErrorUnknown is returned.
 *
 * \param count     - Number of resources to unmap
 * \param resources - Resources to unmap
 * \param stream    - Stream for synchronization
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorUnknown
 * \note_null_stream
 * \notefnerr
 *
 * \sa
 * ::lwdaGraphicsMapResources,
 * ::lwGraphicsUnmapResources
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGraphicsUnmapResources(int count, lwdaGraphicsResource_t *resources, lwdaStream_t stream __dv(0));

/**
 * \brief Get an device pointer through which to access a mapped graphics resource.
 *
 * Returns in \p *devPtr a pointer through which the mapped graphics resource
 * \p resource may be accessed.
 * Returns in \p *size the size of the memory in bytes which may be accessed from that pointer.
 * The value set in \p devPtr may change every time that \p resource is mapped.
 *
 * If \p resource is not a buffer then it cannot be accessed via a pointer and
 * ::lwdaErrorUnknown is returned.
 * If \p resource is not mapped then ::lwdaErrorUnknown is returned.
 * *
 * \param devPtr     - Returned pointer through which \p resource may be accessed
 * \param size       - Returned size of the buffer accessible starting at \p *devPtr
 * \param resource   - Mapped resource to access
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorUnknown
 * \notefnerr
 *
 * \sa
 * ::lwdaGraphicsMapResources,
 * ::lwdaGraphicsSubResourceGetMappedArray,
 * ::lwGraphicsResourceGetMappedPointer
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, lwdaGraphicsResource_t resource);

/**
 * \brief Get an array through which to access a subresource of a mapped graphics resource.
 *
 * Returns in \p *array an array through which the subresource of the mapped
 * graphics resource \p resource which corresponds to array index \p arrayIndex
 * and mipmap level \p mipLevel may be accessed.  The value set in \p array may
 * change every time that \p resource is mapped.
 *
 * If \p resource is not a texture then it cannot be accessed via an array and
 * ::lwdaErrorUnknown is returned.
 * If \p arrayIndex is not a valid array index for \p resource then
 * ::lwdaErrorIlwalidValue is returned.
 * If \p mipLevel is not a valid mipmap level for \p resource then
 * ::lwdaErrorIlwalidValue is returned.
 * If \p resource is not mapped then ::lwdaErrorUnknown is returned.
 *
 * \param array       - Returned array through which a subresource of \p resource may be accessed
 * \param resource    - Mapped resource to access
 * \param arrayIndex  - Array index for array textures or lwbemap face
 *                      index as defined by ::lwdaGraphicsLwbeFace for
 *                      lwbemap textures for the subresource to access
 * \param mipLevel    - Mipmap level for the subresource to access
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorUnknown
 * \notefnerr
 *
 * \sa
 * ::lwdaGraphicsResourceGetMappedPointer,
 * ::lwGraphicsSubResourceGetMappedArray
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGraphicsSubResourceGetMappedArray(lwdaArray_t *array, lwdaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel);

/**
 * \brief Get a mipmapped array through which to access a mapped graphics resource.
 *
 * Returns in \p *mipmappedArray a mipmapped array through which the mapped
 * graphics resource \p resource may be accessed. The value set in \p mipmappedArray may
 * change every time that \p resource is mapped.
 *
 * If \p resource is not a texture then it cannot be accessed via an array and
 * ::lwdaErrorUnknown is returned.
 * If \p resource is not mapped then ::lwdaErrorUnknown is returned.
 *
 * \param mipmappedArray - Returned mipmapped array through which \p resource may be accessed
 * \param resource       - Mapped resource to access
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidResourceHandle,
 * ::lwdaErrorUnknown
 * \notefnerr
 *
 * \sa
 * ::lwdaGraphicsResourceGetMappedPointer,
 * ::lwGraphicsResourceGetMappedMipmappedArray
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGraphicsResourceGetMappedMipmappedArray(lwdaMipmappedArray_t *mipmappedArray, lwdaGraphicsResource_t resource);

/** @} */ /* END LWDART_INTEROP */

/**
 * \defgroup LWDART_TEXTURE Texture Reference Management
 *
 * ___MANBRIEF___ texture reference management functions of the LWCA runtime
 * API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level texture reference management functions
 * of the LWCA runtime application programming interface.
 *
 * Some functions have overloaded C++ API template versions dolwmented separately in the
 * \ref LWDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Get the channel descriptor of an array
 *
 * Returns in \p *desc the channel descriptor of the LWCA array \p array.
 *
 * \param desc  - Channel format
 * \param array - Memory array on device
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \sa \ref ::lwdaCreateChannelDesc(int, int, int, int, lwdaChannelFormatKind) "lwdaCreateChannelDesc (C API)",
 * ::lwdaGetTextureReference,
 * \ref ::lwdaBindTexture(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t) "lwdaBindTexture (C API)",
 * \ref ::lwdaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t, size_t, size_t) "lwdaBindTexture2D (C API)",
 * \ref ::lwdaBindTextureToArray(const struct textureReference*, lwdaArray_const_t, const struct lwdaChannelFormatDesc*) "lwdaBindTextureToArray (C API)",
 * \ref ::lwdaUnbindTexture(const struct textureReference*) "lwdaUnbindTexture (C API)",
 * \ref ::lwdaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "lwdaGetTextureAlignmentOffset (C API)"
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGetChannelDesc(struct lwdaChannelFormatDesc *desc, lwdaArray_const_t array);

/**
 * \brief Returns a channel descriptor using the specified format
 *
 * Returns a channel descriptor with format \p f and number of bits of each
 * component \p x, \p y, \p z, and \p w.  The ::lwdaChannelFormatDesc is
 * defined as:
 * \code
  struct lwdaChannelFormatDesc {
    int x, y, z, w;
    enum lwdaChannelFormatKind f;
  };
 * \endcode
 *
 * where ::lwdaChannelFormatKind is one of ::lwdaChannelFormatKindSigned,
 * ::lwdaChannelFormatKindUnsigned, or ::lwdaChannelFormatKindFloat.
 *
 * \param x - X component
 * \param y - Y component
 * \param z - Z component
 * \param w - W component
 * \param f - Channel format
 *
 * \return
 * Channel descriptor with format \p f
 *
 * \sa \ref ::lwdaCreateChannelDesc(void) "lwdaCreateChannelDesc (C++ API)",
 * ::lwdaGetChannelDesc, ::lwdaGetTextureReference,
 * \ref ::lwdaBindTexture(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t) "lwdaBindTexture (C API)",
 * \ref ::lwdaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t, size_t, size_t) "lwdaBindTexture2D (C API)",
 * \ref ::lwdaBindTextureToArray(const struct textureReference*, lwdaArray_const_t, const struct lwdaChannelFormatDesc*) "lwdaBindTextureToArray (C API)",
 * \ref ::lwdaUnbindTexture(const struct textureReference*) "lwdaUnbindTexture (C API)",
 * \ref ::lwdaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "lwdaGetTextureAlignmentOffset (C API)",
 * ::lwTexRefSetFormat
 */
extern __host__ struct lwdaChannelFormatDesc LWDARTAPI lwdaCreateChannelDesc(int x, int y, int z, int w, enum lwdaChannelFormatKind f);


/**
 * \brief Binds a memory area to a texture
 *
 * Binds \p size bytes of the memory area pointed to by \p devPtr to the
 * texture reference \p texref. \p desc describes how the memory is interpreted
 * when fetching values from the texture. Any memory previously bound to
 * \p texref is unbound.
 *
 * Since the hardware enforces an alignment requirement on texture base
 * addresses,
 * \ref ::lwdaBindTexture(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t) "lwdaBindTexture()"
 * returns in \p *offset a byte offset that
 * must be applied to texture fetches in order to read from the desired memory.
 * This offset must be divided by the texel size and passed to kernels that
 * read from the texture so they can be applied to the ::tex1Dfetch() function.
 * If the device memory pointer was returned from ::lwdaMalloc(), the offset is
 * guaranteed to be 0 and NULL may be passed as the \p offset parameter.
 *
 * The total number of elements (or texels) in the linear address range
 * cannot exceed ::lwdaDeviceProp::maxTexture1DLinear[0].
 * The number of elements is computed as (\p size / elementSize),
 * where elementSize is determined from \p desc.
 *
 * \param offset - Offset in bytes
 * \param texref - Texture to bind
 * \param devPtr - Memory area on device
 * \param desc   - Channel format
 * \param size   - Size of the memory area pointed to by devPtr
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidTexture
 * \notefnerr
 *
 * \sa \ref ::lwdaCreateChannelDesc(int, int, int, int, lwdaChannelFormatKind) "lwdaCreateChannelDesc (C API)",
 * ::lwdaGetChannelDesc, ::lwdaGetTextureReference,
 * \ref ::lwdaBindTexture(size_t*, const struct texture< T, dim, readMode>&, const void*, const struct lwdaChannelFormatDesc&, size_t) "lwdaBindTexture (C++ API)",
 * \ref ::lwdaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t, size_t, size_t) "lwdaBindTexture2D (C API)",
 * \ref ::lwdaBindTextureToArray(const struct textureReference*, lwdaArray_const_t, const struct lwdaChannelFormatDesc*) "lwdaBindTextureToArray (C API)",
 * \ref ::lwdaUnbindTexture(const struct textureReference*) "lwdaUnbindTexture (C API)",
 * \ref ::lwdaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "lwdaGetTextureAlignmentOffset (C API)",
 * ::lwTexRefSetAddress,
 * ::lwTexRefSetAddressMode,
 * ::lwTexRefSetFormat,
 * ::lwTexRefSetFlags,
 * ::lwTexRefSetBorderColor
 */
extern __host__ lwdaError_t LWDARTAPI lwdaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct lwdaChannelFormatDesc *desc, size_t size __dv(UINT_MAX));

/**
 * \brief Binds a 2D memory area to a texture
 *
 * Binds the 2D memory area pointed to by \p devPtr to the
 * texture reference \p texref. The size of the area is constrained by
 * \p width in texel units, \p height in texel units, and \p pitch in byte
 * units. \p desc describes how the memory is interpreted when fetching values
 * from the texture. Any memory previously bound to \p texref is unbound.
 *
 * Since the hardware enforces an alignment requirement on texture base
 * addresses, ::lwdaBindTexture2D() returns in \p *offset a byte offset that
 * must be applied to texture fetches in order to read from the desired memory.
 * This offset must be divided by the texel size and passed to kernels that
 * read from the texture so they can be applied to the ::tex2D() function.
 * If the device memory pointer was returned from ::lwdaMalloc(), the offset is
 * guaranteed to be 0 and NULL may be passed as the \p offset parameter.
 *
 * \p width and \p height, which are specified in elements (or texels), cannot
 * exceed ::lwdaDeviceProp::maxTexture2DLinear[0] and ::lwdaDeviceProp::maxTexture2DLinear[1]
 * respectively. \p pitch, which is specified in bytes, cannot exceed
 * ::lwdaDeviceProp::maxTexture2DLinear[2].
 *
 * The driver returns ::lwdaErrorIlwalidValue if \p pitch is not a multiple of
 * ::lwdaDeviceProp::texturePitchAlignment.
 *
 * \param offset - Offset in bytes
 * \param texref - Texture reference to bind
 * \param devPtr - 2D memory area on device
 * \param desc   - Channel format
 * \param width  - Width in texel units
 * \param height - Height in texel units
 * \param pitch  - Pitch in bytes
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidTexture
 * \notefnerr
 *
 * \sa \ref ::lwdaCreateChannelDesc(int, int, int, int, lwdaChannelFormatKind) "lwdaCreateChannelDesc (C API)",
 * ::lwdaGetChannelDesc, ::lwdaGetTextureReference,
 * \ref ::lwdaBindTexture(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t) "lwdaBindTexture (C API)",
 * \ref ::lwdaBindTexture2D(size_t*, const struct texture< T, dim, readMode>&, const void*, const struct lwdaChannelFormatDesc&, size_t, size_t, size_t) "lwdaBindTexture2D (C++ API)",
 * \ref ::lwdaBindTexture2D(size_t*, const struct texture<T, dim, readMode>&, const void*, size_t, size_t, size_t) "lwdaBindTexture2D (C++ API, inherited channel descriptor)",
 * \ref ::lwdaBindTextureToArray(const struct textureReference*, lwdaArray_const_t, const struct lwdaChannelFormatDesc*) "lwdaBindTextureToArray (C API)",
 * \ref ::lwdaUnbindTexture(const struct textureReference*) "lwdaBindTextureToArray (C API)",
 * \ref ::lwdaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "lwdaGetTextureAlignmentOffset (C API)",
 * ::lwTexRefSetAddress2D,
 * ::lwTexRefSetFormat,
 * ::lwTexRefSetFlags,
 * ::lwTexRefSetAddressMode,
 * ::lwTexRefSetBorderColor
 */
extern __host__ lwdaError_t LWDARTAPI lwdaBindTexture2D(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct lwdaChannelFormatDesc *desc, size_t width, size_t height, size_t pitch);

/**
 * \brief Binds an array to a texture
 *
 * Binds the LWCA array \p array to the texture reference \p texref.
 * \p desc describes how the memory is interpreted when fetching values from
 * the texture. Any LWCA array previously bound to \p texref is unbound.
 *
 * \param texref - Texture to bind
 * \param array  - Memory array on device
 * \param desc   - Channel format
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidTexture
 * \notefnerr
 *
 * \sa \ref ::lwdaCreateChannelDesc(int, int, int, int, lwdaChannelFormatKind) "lwdaCreateChannelDesc (C API)",
 * ::lwdaGetChannelDesc, ::lwdaGetTextureReference,
 * \ref ::lwdaBindTexture(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t) "lwdaBindTexture (C API)",
 * \ref ::lwdaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t, size_t, size_t) "lwdaBindTexture2D (C API)",
 * \ref ::lwdaBindTextureToArray(const struct texture< T, dim, readMode>&, lwdaArray_const_t, const struct lwdaChannelFormatDesc&) "lwdaBindTextureToArray (C++ API)",
 * \ref ::lwdaUnbindTexture(const struct textureReference*) "lwdaUnbindTexture (C API)",
 * \ref ::lwdaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "lwdaGetTextureAlignmentOffset (C API)",
 * ::lwTexRefSetArray,
 * ::lwTexRefSetFormat,
 * ::lwTexRefSetFlags,
 * ::lwTexRefSetAddressMode,
 * ::lwTexRefSetFilterMode,
 * ::lwTexRefSetBorderColor,
 * ::lwTexRefSetMaxAnisotropy
 */
extern __host__ lwdaError_t LWDARTAPI lwdaBindTextureToArray(const struct textureReference *texref, lwdaArray_const_t array, const struct lwdaChannelFormatDesc *desc);

/**
 * \brief Binds a mipmapped array to a texture
 *
 * Binds the LWCA mipmapped array \p mipmappedArray to the texture reference \p texref.
 * \p desc describes how the memory is interpreted when fetching values from
 * the texture. Any LWCA mipmapped array previously bound to \p texref is unbound.
 *
 * \param texref         - Texture to bind
 * \param mipmappedArray - Memory mipmapped array on device
 * \param desc           - Channel format
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidTexture
 * \notefnerr
 *
 * \sa \ref ::lwdaCreateChannelDesc(int, int, int, int, lwdaChannelFormatKind) "lwdaCreateChannelDesc (C API)",
 * ::lwdaGetChannelDesc, ::lwdaGetTextureReference,
 * \ref ::lwdaBindTexture(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t) "lwdaBindTexture (C API)",
 * \ref ::lwdaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t, size_t, size_t) "lwdaBindTexture2D (C API)",
 * \ref ::lwdaBindTextureToArray(const struct texture< T, dim, readMode>&, lwdaArray_const_t, const struct lwdaChannelFormatDesc&) "lwdaBindTextureToArray (C++ API)",
 * \ref ::lwdaUnbindTexture(const struct textureReference*) "lwdaUnbindTexture (C API)",
 * \ref ::lwdaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "lwdaGetTextureAlignmentOffset (C API)",
 * ::lwTexRefSetMipmappedArray,
 * ::lwTexRefSetMipmapFilterMode
 * ::lwTexRefSetMipmapLevelClamp,
 * ::lwTexRefSetMipmapLevelBias,
 * ::lwTexRefSetFormat,
 * ::lwTexRefSetFlags,
 * ::lwTexRefSetAddressMode,
 * ::lwTexRefSetBorderColor,
 * ::lwTexRefSetMaxAnisotropy
 */
extern __host__ lwdaError_t LWDARTAPI lwdaBindTextureToMipmappedArray(const struct textureReference *texref, lwdaMipmappedArray_const_t mipmappedArray, const struct lwdaChannelFormatDesc *desc);

/**
 * \brief Unbinds a texture
 *
 * Unbinds the texture bound to \p texref.
 *
 * \param texref - Texture to unbind
 *
 * \return
 * ::lwdaSuccess
 * \notefnerr
 *
 * \sa \ref ::lwdaCreateChannelDesc(int, int, int, int, lwdaChannelFormatKind) "lwdaCreateChannelDesc (C API)",
 * ::lwdaGetChannelDesc, ::lwdaGetTextureReference,
 * \ref ::lwdaBindTexture(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t) "lwdaBindTexture (C API)",
 * \ref ::lwdaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t, size_t, size_t) "lwdaBindTexture2D (C API)",
 * \ref ::lwdaBindTextureToArray(const struct textureReference*, lwdaArray_const_t, const struct lwdaChannelFormatDesc*) "lwdaBindTextureToArray (C API)",
 * \ref ::lwdaUnbindTexture(const struct texture< T, dim, readMode>&) "lwdaUnbindTexture (C++ API)",
 * \ref ::lwdaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "lwdaGetTextureAlignmentOffset (C API)"
 */
extern __host__ lwdaError_t LWDARTAPI lwdaUnbindTexture(const struct textureReference *texref);

/**
 * \brief Get the alignment offset of a texture
 *
 * Returns in \p *offset the offset that was returned when texture reference
 * \p texref was bound.
 *
 * \param offset - Offset of texture reference in bytes
 * \param texref - Texture to get offset of
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidTexture,
 * ::lwdaErrorIlwalidTextureBinding
 * \notefnerr
 *
 * \sa \ref ::lwdaCreateChannelDesc(int, int, int, int, lwdaChannelFormatKind) "lwdaCreateChannelDesc (C API)",
 * ::lwdaGetChannelDesc, ::lwdaGetTextureReference,
 * \ref ::lwdaBindTexture(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t) "lwdaBindTexture (C API)",
 * \ref ::lwdaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t, size_t, size_t) "lwdaBindTexture2D (C API)",
 * \ref ::lwdaBindTextureToArray(const struct textureReference*, lwdaArray_const_t, const struct lwdaChannelFormatDesc*) "lwdaBindTextureToArray (C API)",
 * \ref ::lwdaUnbindTexture(const struct textureReference*) "lwdaUnbindTexture (C API)",
 * \ref ::lwdaGetTextureAlignmentOffset(size_t*, const struct texture< T, dim, readMode>&) "lwdaGetTextureAlignmentOffset (C++ API)"
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref);

/**
 * \brief Get the texture reference associated with a symbol
 *
 * Returns in \p *texref the structure associated to the texture reference
 * defined by symbol \p symbol.
 *
 * \param texref - Texture reference associated with symbol
 * \param symbol - Texture to get reference for
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidTexture
 * \notefnerr
 * \note_string_api_deprecation_50
 *
 * \sa \ref ::lwdaCreateChannelDesc(int, int, int, int, lwdaChannelFormatKind) "lwdaCreateChannelDesc (C API)",
 * ::lwdaGetChannelDesc,
 * \ref ::lwdaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "lwdaGetTextureAlignmentOffset (C API)",
 * \ref ::lwdaBindTexture(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t) "lwdaBindTexture (C API)",
 * \ref ::lwdaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct lwdaChannelFormatDesc*, size_t, size_t, size_t) "lwdaBindTexture2D (C API)",
 * \ref ::lwdaBindTextureToArray(const struct textureReference*, lwdaArray_const_t, const struct lwdaChannelFormatDesc*) "lwdaBindTextureToArray (C API)",
 * \ref ::lwdaUnbindTexture(const struct textureReference*) "lwdaUnbindTexture (C API)",
 * ::lwModuleGetTexRef
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGetTextureReference(const struct textureReference **texref, const void *symbol);

/** @} */ /* END LWDART_TEXTURE */

/**
 * \defgroup LWDART_SURFACE Surface Reference Management
 *
 * ___MANBRIEF___ surface reference management functions of the LWCA runtime
 * API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level surface reference management functions
 * of the LWCA runtime application programming interface.
 *
 * Some functions have overloaded C++ API template versions dolwmented separately in the
 * \ref LWDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Binds an array to a surface
 *
 * Binds the LWCA array \p array to the surface reference \p surfref.
 * \p desc describes how the memory is interpreted when fetching values from
 * the surface. Any LWCA array previously bound to \p surfref is unbound.
 *
 * \param surfref - Surface to bind
 * \param array  - Memory array on device
 * \param desc   - Channel format
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue,
 * ::lwdaErrorIlwalidSurface
 * \notefnerr
 *
 * \sa \ref ::lwdaBindSurfaceToArray(const struct surface< T, dim>&, lwdaArray_const_t, const struct lwdaChannelFormatDesc&) "lwdaBindSurfaceToArray (C++ API)",
 * \ref ::lwdaBindSurfaceToArray(const struct surface< T, dim>&, lwdaArray_const_t) "lwdaBindSurfaceToArray (C++ API, inherited channel descriptor)",
 * ::lwdaGetSurfaceReference,
 * ::lwSurfRefSetArray
 */
extern __host__ lwdaError_t LWDARTAPI lwdaBindSurfaceToArray(const struct surfaceReference *surfref, lwdaArray_const_t array, const struct lwdaChannelFormatDesc *desc);

/**
 * \brief Get the surface reference associated with a symbol
 *
 * Returns in \p *surfref the structure associated to the surface reference
 * defined by symbol \p symbol.
 *
 * \param surfref - Surface reference associated with symbol
 * \param symbol - Surface to get reference for
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidSurface
 * \notefnerr
 * \note_string_api_deprecation_50
 *
 * \sa
 * \ref ::lwdaBindSurfaceToArray(const struct surfaceReference*, lwdaArray_const_t, const struct lwdaChannelFormatDesc*) "lwdaBindSurfaceToArray (C API)",
 * ::lwModuleGetSurfRef
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGetSurfaceReference(const struct surfaceReference **surfref, const void *symbol);

/** @} */ /* END LWDART_SURFACE */

/**
 * \defgroup LWDART_TEXTURE_OBJECT Texture Object Management
 *
 * ___MANBRIEF___ texture object management functions of the LWCA runtime API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level texture object management functions
 * of the LWCA runtime application programming interface. The texture
 * object API is only supported on devices of compute capability 3.0 or higher.
 *
 * @{
 */

/**
 * \brief Creates a texture object
 *
 * Creates a texture object and returns it in \p pTexObject. \p pResDesc describes
 * the data to texture from. \p pTexDesc describes how the data should be sampled.
 * \p pResViewDesc is an optional argument that specifies an alternate format for
 * the data described by \p pResDesc, and also describes the subresource region
 * to restrict access to when texturing. \p pResViewDesc can only be specified if
 * the type of resource is a LWCA array or a LWCA mipmapped array.
 *
 * Texture objects are only supported on devices of compute capability 3.0 or higher.
 * Additionally, a texture object is an opaque value, and, as such, should only be
 * accessed through LWCA API calls.
 *
 * The ::lwdaResourceDesc structure is defined as:
 * \code
        struct lwdaResourceDesc {
	        enum lwdaResourceType resType;
        	
	        union {
		        struct {
			        lwdaArray_t array;
		        } array;
                struct {
                    lwdaMipmappedArray_t mipmap;
                } mipmap;
		        struct {
			        void *devPtr;
			        struct lwdaChannelFormatDesc desc;
			        size_t sizeInBytes;
		        } linear;
		        struct {
			        void *devPtr;
			        struct lwdaChannelFormatDesc desc;
			        size_t width;
			        size_t height;
			        size_t pitchInBytes;
		        } pitch2D;
	        } res;
        };
 * \endcode
 * where:
 * - ::lwdaResourceDesc::resType specifies the type of resource to texture from.
 * LWresourceType is defined as:
 * \code
        enum lwdaResourceType {
            lwdaResourceTypeArray          = 0x00,
            lwdaResourceTypeMipmappedArray = 0x01,
            lwdaResourceTypeLinear         = 0x02,
            lwdaResourceTypePitch2D        = 0x03
        };
 * \endcode
 *
 * \par
 * If ::lwdaResourceDesc::resType is set to ::lwdaResourceTypeArray, ::lwdaResourceDesc::res::array::array
 * must be set to a valid LWCA array handle.
 *
 * \par
 * If ::lwdaResourceDesc::resType is set to ::lwdaResourceTypeMipmappedArray, ::lwdaResourceDesc::res::mipmap::mipmap
 * must be set to a valid LWCA mipmapped array handle and ::lwdaTextureDesc::normalizedCoords must be set to true.
 *
 * \par
 * If ::lwdaResourceDesc::resType is set to ::lwdaResourceTypeLinear, ::lwdaResourceDesc::res::linear::devPtr
 * must be set to a valid device pointer, that is aligned to ::lwdaDeviceProp::textureAlignment.
 * ::lwdaResourceDesc::res::linear::desc describes the format and the number of components per array element. ::lwdaResourceDesc::res::linear::sizeInBytes
 * specifies the size of the array in bytes. The total number of elements in the linear address range cannot exceed 
 * ::lwdaDeviceProp::maxTexture1DLinear. The number of elements is computed as (sizeInBytes / sizeof(desc)).
 *
 * \par
 * If ::lwdaResourceDesc::resType is set to ::lwdaResourceTypePitch2D, ::lwdaResourceDesc::res::pitch2D::devPtr
 * must be set to a valid device pointer, that is aligned to ::lwdaDeviceProp::textureAlignment.
 * ::lwdaResourceDesc::res::pitch2D::desc describes the format and the number of components per array element. ::lwdaResourceDesc::res::pitch2D::width
 * and ::lwdaResourceDesc::res::pitch2D::height specify the width and height of the array in elements, and cannot exceed
 * ::lwdaDeviceProp::maxTexture2DLinear[0] and ::lwdaDeviceProp::maxTexture2DLinear[1] respectively.
 * ::lwdaResourceDesc::res::pitch2D::pitchInBytes specifies the pitch between two rows in bytes and has to be aligned to 
 * ::lwdaDeviceProp::texturePitchAlignment. Pitch cannot exceed ::lwdaDeviceProp::maxTexture2DLinear[2].
 *
 *
 * The ::lwdaTextureDesc struct is defined as
 * \code
        struct lwdaTextureDesc {
            enum lwdaTextureAddressMode addressMode[3];
            enum lwdaTextureFilterMode  filterMode;
            enum lwdaTextureReadMode    readMode;
            int                         sRGB;
            float                       borderColor[4];
            int                         normalizedCoords;
            unsigned int                maxAnisotropy;
            enum lwdaTextureFilterMode  mipmapFilterMode;
            float                       mipmapLevelBias;
            float                       minMipmapLevelClamp;
            float                       maxMipmapLevelClamp;
        };
 * \endcode
 * where
 * - ::lwdaTextureDesc::addressMode specifies the addressing mode for each dimension of the texture data. ::lwdaTextureAddressMode is defined as:
 *   \code
        enum lwdaTextureAddressMode {
            lwdaAddressModeWrap   = 0,
            lwdaAddressModeClamp  = 1,
            lwdaAddressModeMirror = 2,
            lwdaAddressModeBorder = 3
        };
 *   \endcode
 *   This is ignored if ::lwdaResourceDesc::resType is ::lwdaResourceTypeLinear. Also, if ::lwdaTextureDesc::normalizedCoords
 *   is set to zero, ::lwdaAddressModeWrap and ::lwdaAddressModeMirror won't be supported and will be switched to ::lwdaAddressModeClamp.
 *
 * - ::lwdaTextureDesc::filterMode specifies the filtering mode to be used when fetching from the texture. ::lwdaTextureFilterMode is defined as:
 *   \code
        enum lwdaTextureFilterMode {
            lwdaFilterModePoint  = 0,
            lwdaFilterModeLinear = 1
        };
 *   \endcode
 *   This is ignored if ::lwdaResourceDesc::resType is ::lwdaResourceTypeLinear.
 *
 * - ::lwdaTextureDesc::readMode specifies whether integer data should be colwerted to floating point or not. ::lwdaTextureReadMode is defined as:
 *   \code
        enum lwdaTextureReadMode {
            lwdaReadModeElementType     = 0,
            lwdaReadModeNormalizedFloat = 1
        };
 *   \endcode
 *   Note that this applies only to 8-bit and 16-bit integer formats. 32-bit integer format would not be promoted, regardless of 
 *   whether or not this ::lwdaTextureDesc::readMode is set ::lwdaReadModeNormalizedFloat is specified.
 *
 * - ::lwdaTextureDesc::sRGB specifies whether sRGB to linear colwersion should be performed during texture fetch.
 *
 * - ::lwdaTextureDesc::borderColor specifies the float values of color. where:
 *   ::lwdaTextureDesc::borderColor[0] contains value of 'R', 
 *   ::lwdaTextureDesc::borderColor[1] contains value of 'G',
 *   ::lwdaTextureDesc::borderColor[2] contains value of 'B', 
 *   ::lwdaTextureDesc::borderColor[3] contains value of 'A'
 *   Note that application using integer border color values will need to <reinterpret_cast> these values to float.
 *   The values are set only when the addressing mode specified by ::lwdaTextureDesc::addressMode is lwdaAddressModeBorder.
 *
 * - ::lwdaTextureDesc::normalizedCoords specifies whether the texture coordinates will be normalized or not.
 *
 * - ::lwdaTextureDesc::maxAnisotropy specifies the maximum anistropy ratio to be used when doing anisotropic filtering. This value will be
 *   clamped to the range [1,16].
 *
 * - ::lwdaTextureDesc::mipmapFilterMode specifies the filter mode when the callwlated mipmap level lies between two defined mipmap levels.
 *
 * - ::lwdaTextureDesc::mipmapLevelBias specifies the offset to be applied to the callwlated mipmap level.
 *
 * - ::lwdaTextureDesc::minMipmapLevelClamp specifies the lower end of the mipmap level range to clamp access to.
 *
 * - ::lwdaTextureDesc::maxMipmapLevelClamp specifies the upper end of the mipmap level range to clamp access to.
 *
 *
 * The ::lwdaResourceViewDesc struct is defined as
 * \code
        struct lwdaResourceViewDesc {
            enum lwdaResourceViewFormat format;
            size_t                      width;
            size_t                      height;
            size_t                      depth;
            unsigned int                firstMipmapLevel;
            unsigned int                lastMipmapLevel;
            unsigned int                firstLayer;
            unsigned int                lastLayer;
        };
 * \endcode
 * where:
 * - ::lwdaResourceViewDesc::format specifies how the data contained in the LWCA array or LWCA mipmapped array should
 *   be interpreted. Note that this can inlwr a change in size of the texture data. If the resource view format is a block
 *   compressed format, then the underlying LWCA array or LWCA mipmapped array has to have a 32-bit unsigned integer format
 *   with 2 or 4 channels, depending on the block compressed format. For ex., BC1 and BC4 require the underlying LWCA array to have
 *   a 32-bit unsigned int with 2 channels. The other BC formats require the underlying resource to have the same 32-bit unsigned int
 *   format but with 4 channels.
 *
 * - ::lwdaResourceViewDesc::width specifies the new width of the texture data. If the resource view format is a block
 *   compressed format, this value has to be 4 times the original width of the resource. For non block compressed formats,
 *   this value has to be equal to that of the original resource.
 *
 * - ::lwdaResourceViewDesc::height specifies the new height of the texture data. If the resource view format is a block
 *   compressed format, this value has to be 4 times the original height of the resource. For non block compressed formats,
 *   this value has to be equal to that of the original resource.
 *
 * - ::lwdaResourceViewDesc::depth specifies the new depth of the texture data. This value has to be equal to that of the
 *   original resource.
 *
 * - ::lwdaResourceViewDesc::firstMipmapLevel specifies the most detailed mipmap level. This will be the new mipmap level zero.
 *   For non-mipmapped resources, this value has to be zero.::lwdaTextureDesc::minMipmapLevelClamp and ::lwdaTextureDesc::maxMipmapLevelClamp
 *   will be relative to this value. For ex., if the firstMipmapLevel is set to 2, and a minMipmapLevelClamp of 1.2 is specified,
 *   then the actual minimum mipmap level clamp will be 3.2.
 *
 * - ::lwdaResourceViewDesc::lastMipmapLevel specifies the least detailed mipmap level. For non-mipmapped resources, this value
 *   has to be zero.
 *
 * - ::lwdaResourceViewDesc::firstLayer specifies the first layer index for layered textures. This will be the new layer zero.
 *   For non-layered resources, this value has to be zero.
 *
 * - ::lwdaResourceViewDesc::lastLayer specifies the last layer index for layered textures. For non-layered resources, 
 *   this value has to be zero.
 *
 *
 * \param pTexObject   - Texture object to create
 * \param pResDesc     - Resource descriptor
 * \param pTexDesc     - Texture descriptor
 * \param pResViewDesc - Resource view descriptor
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 *
 * \sa
 * ::lwdaDestroyTextureObject,
 * ::lwTexObjectCreate
 */

extern __host__ lwdaError_t LWDARTAPI lwdaCreateTextureObject(lwdaTextureObject_t *pTexObject, const struct lwdaResourceDesc *pResDesc, const struct lwdaTextureDesc *pTexDesc, const struct lwdaResourceViewDesc *pResViewDesc);

/**
 * \brief Destroys a texture object
 *
 * Destroys the texture object specified by \p texObject.
 *
 * \param texObject - Texture object to destroy
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 *
 * \sa
 * ::lwdaCreateTextureObject,
 * ::lwTexObjectDestroy
 */
extern __host__ lwdaError_t LWDARTAPI lwdaDestroyTextureObject(lwdaTextureObject_t texObject);

/**
 * \brief Returns a texture object's resource descriptor
 *
 * Returns the resource descriptor for the texture object specified by \p texObject.
 *
 * \param pResDesc  - Resource descriptor
 * \param texObject - Texture object
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 *
 * \sa
 * ::lwdaCreateTextureObject,
 * ::lwTexObjectGetResourceDesc
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGetTextureObjectResourceDesc(struct lwdaResourceDesc *pResDesc, lwdaTextureObject_t texObject);

/**
 * \brief Returns a texture object's texture descriptor
 *
 * Returns the texture descriptor for the texture object specified by \p texObject.
 *
 * \param pTexDesc  - Texture descriptor
 * \param texObject - Texture object
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 *
 * \sa
 * ::lwdaCreateTextureObject,
 * ::lwTexObjectGetTextureDesc
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGetTextureObjectTextureDesc(struct lwdaTextureDesc *pTexDesc, lwdaTextureObject_t texObject);

/**
 * \brief Returns a texture object's resource view descriptor
 *
 * Returns the resource view descriptor for the texture object specified by \p texObject.
 * If no resource view was specified, ::lwdaErrorIlwalidValue is returned.
 *
 * \param pResViewDesc - Resource view descriptor
 * \param texObject    - Texture object
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 *
 * \sa
 * ::lwdaCreateTextureObject,
 * ::lwTexObjectGetResourceViewDesc
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGetTextureObjectResourceViewDesc(struct lwdaResourceViewDesc *pResViewDesc, lwdaTextureObject_t texObject);

/** @} */ /* END LWDART_TEXTURE_OBJECT */

/**
 * \defgroup LWDART_SURFACE_OBJECT Surface Object Management
 *
 * ___MANBRIEF___ surface object management functions of the LWCA runtime API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level texture object management functions
 * of the LWCA runtime application programming interface. The surface object 
 * API is only supported on devices of compute capability 3.0 or higher.
 *
 * @{
 */

/**
 * \brief Creates a surface object
 *
 * Creates a surface object and returns it in \p pSurfObject. \p pResDesc describes
 * the data to perform surface load/stores on. ::lwdaResourceDesc::resType must be 
 * ::lwdaResourceTypeArray and  ::lwdaResourceDesc::res::array::array
 * must be set to a valid LWCA array handle.
 *
 * Surface objects are only supported on devices of compute capability 3.0 or higher.
 * Additionally, a surface object is an opaque value, and, as such, should only be
 * accessed through LWCA API calls.
 *
 * \param pSurfObject - Surface object to create
 * \param pResDesc    - Resource descriptor
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 *
 * \sa
 * ::lwdaDestroySurfaceObject,
 * ::lwSurfObjectCreate
 */

extern __host__ lwdaError_t LWDARTAPI lwdaCreateSurfaceObject(lwdaSurfaceObject_t *pSurfObject, const struct lwdaResourceDesc *pResDesc);

/**
 * \brief Destroys a surface object
 *
 * Destroys the surface object specified by \p surfObject.
 *
 * \param surfObject - Surface object to destroy
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 *
 * \sa
 * ::lwdaCreateSurfaceObject,
 * ::lwSurfObjectDestroy
 */
extern __host__ lwdaError_t LWDARTAPI lwdaDestroySurfaceObject(lwdaSurfaceObject_t surfObject);

/**
 * \brief Returns a surface object's resource descriptor
 * Returns the resource descriptor for the surface object specified by \p surfObject.
 *
 * \param pResDesc   - Resource descriptor
 * \param surfObject - Surface object
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 *
 * \sa
 * ::lwdaCreateSurfaceObject,
 * ::lwSurfObjectGetResourceDesc
 */
extern __host__ lwdaError_t LWDARTAPI lwdaGetSurfaceObjectResourceDesc(struct lwdaResourceDesc *pResDesc, lwdaSurfaceObject_t surfObject);

/** @} */ /* END LWDART_SURFACE_OBJECT */

/**
 * \defgroup LWDART__VERSION Version Management
 *
 * @{
 */

/**
 * \brief Returns the LWCA driver version
 *
 * Returns in \p *driverVersion the version number of the installed LWCA
 * driver. If no driver is installed, then 0 is returned as the driver
 * version (via \p driverVersion). This function automatically returns
 * ::lwdaErrorIlwalidValue if the \p driverVersion argument is NULL.
 *
 * \param driverVersion - Returns the LWCA driver version.
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 * \notefnerr
 *
 * \sa
 * ::lwdaRuntimeGetVersion,
 * ::lwDriverGetVersion
 */
extern __host__ lwdaError_t LWDARTAPI lwdaDriverGetVersion(int *driverVersion);

/**
 * \brief Returns the LWCA Runtime version
 *
 * Returns in \p *runtimeVersion the version number of the installed LWCA
 * Runtime. This function automatically returns ::lwdaErrorIlwalidValue if
 * the \p runtimeVersion argument is NULL.
 *
 * \param runtimeVersion - Returns the LWCA Runtime version.
 *
 * \return
 * ::lwdaSuccess,
 * ::lwdaErrorIlwalidValue
 *
 * \sa
 * ::lwdaDriverGetVersion,
 * ::lwDriverGetVersion
 */
extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaRuntimeGetVersion(int *runtimeVersion);

/** @} */ /* END LWDART__VERSION */

/** \cond impl_private */
extern __host__ lwdaError_t LWDARTAPI lwdaGetExportTable(const void **ppExportTable, const lwdaUUID_t *pExportTableId);
/** \endcond impl_private */

/**
 * \defgroup LWDART_HIGHLEVEL C++ API Routines
 *
 * ___MANBRIEF___ C++ high level API functions of the LWCA runtime API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the C++ high level API functions of the LWCA runtime
 * application programming interface. To use these functions, your
 * application needs to be compiled with the \p lwcc compiler.
 *
 * \brief C++-style interface built on top of LWCA runtime API
 */

/**
 * \defgroup LWDART_DRIVER Interactions with the LWCA Driver API
 *
 * ___MANBRIEF___ interactions between LWCA Driver API and LWCA Runtime API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the interactions between the LWCA Driver API and the LWCA Runtime API
 *
 * @{
 *
 * \section LWDART_LWDA_primary Primary Contexts
 *
 * There exists a one to one relationship between LWCA devices in the LWCA Runtime
 * API and ::LWcontext s in the LWCA Driver API within a process.  The specific
 * context which the LWCA Runtime API uses for a device is called the device's
 * primary context.  From the perspective of the LWCA Runtime API, a device and 
 * its primary context are synonymous.
 *
 * \section LWDART_LWDA_init Initialization and Tear-Down
 *
 * LWCA Runtime API calls operate on the LWCA Driver API ::LWcontext which is current to
 * to the calling host thread.  
 *
 * The function ::lwdaSetDevice() makes the primary context for the
 * specified device current to the calling thread by calling ::lwCtxSetLwrrent().
 *
 * The LWCA Runtime API will automatically initialize the primary context for
 * a device at the first LWCA Runtime API call which requires an active context.
 * If no ::LWcontext is current to the calling thread when a LWCA Runtime API call 
 * which requires an active context is made, then the primary context for a device 
 * will be selected, made current to the calling thread, and initialized.
 *
 * The context which the LWCA Runtime API initializes will be initialized using 
 * the parameters specified by the LWCA Runtime API functions
 * ::lwdaSetDeviceFlags(), 
 * ::lwdaD3D9SetDirect3DDevice(), 
 * ::lwdaD3D10SetDirect3DDevice(), 
 * ::lwdaD3D11SetDirect3DDevice(), 
 * ::lwdaGLSetGLDevice(), and
 * ::lwdaVDPAUSetVDPAUDevice().
 * Note that these functions will fail with ::lwdaErrorSetOnActiveProcess if they are 
 * called when the primary context for the specified device has already been initialized.
 * (or if the current device has already been initialized, in the case of 
 * ::lwdaSetDeviceFlags()). 
 *
 * Primary contexts will remain active until they are explicitly deinitialized 
 * using ::lwdaDeviceReset().  The function ::lwdaDeviceReset() will deinitialize the 
 * primary context for the calling thread's current device immediately.  The context 
 * will remain current to all of the threads that it was current to.  The next LWCA 
 * Runtime API call on any thread which requires an active context will trigger the 
 * reinitialization of that device's primary context.
 *
 * Note that there is no reference counting of the primary context's lifetime.  It is
 * recommended that the primary context not be deinitialized except just before exit
 * or to recover from an unspecified launch failure.
 * 
 * \section LWDART_LWDA_context Context Interoperability
 *
 * Note that the use of multiple ::LWcontext s per device within a single process 
 * will substantially degrade performance and is strongly discouraged.  Instead,
 * it is highly recommended that the implicit one-to-one device-to-context mapping
 * for the process provided by the LWCA Runtime API be used.
 *
 * If a non-primary ::LWcontext created by the LWCA Driver API is current to a
 * thread then the LWCA Runtime API calls to that thread will operate on that 
 * ::LWcontext, with some exceptions listed below.  Interoperability between data
 * types is dislwssed in the following sections.
 *
 * The function ::lwdaPointerGetAttributes() will return the error 
 * ::lwdaErrorIncompatibleDriverContext if the pointer being queried was allocated by a 
 * non-primary context.  The function ::lwdaDeviceEnablePeerAccess() and the rest of 
 * the peer access API may not be called when a non-primary ::LWcontext is current.  
 * To use the pointer query and peer access APIs with a context created using the 
 * LWCA Driver API, it is necessary that the LWCA Driver API be used to access
 * these features.
 *
 * All LWCA Runtime API state (e.g, global variables' addresses and values) travels
 * with its underlying ::LWcontext.  In particular, if a ::LWcontext is moved from one 
 * thread to another then all LWCA Runtime API state will move to that thread as well.
 *
 * Please note that attaching to legacy contexts (those with a version of 3010 as returned
 * by ::lwCtxGetApiVersion()) is not possible. The LWCA Runtime will return
 * ::lwdaErrorIncompatibleDriverContext in such cases.
 *
 * \section LWDART_LWDA_stream Interactions between LWstream and lwdaStream_t
 *
 * The types ::LWstream and ::lwdaStream_t are identical and may be used interchangeably.
 *
 * \section LWDART_LWDA_event Interactions between LWevent and lwdaEvent_t
 *
 * The types ::LWevent and ::lwdaEvent_t are identical and may be used interchangeably.
 *
 * \section LWDART_LWDA_array Interactions between LWarray and lwdaArray_t 
 *
 * The types ::LWarray and struct ::lwdaArray * represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::LWarray in a LWCA Runtime API function which takes a struct ::lwdaArray *,
 * it is necessary to explicitly cast the ::LWarray to a struct ::lwdaArray *.
 *
 * In order to use a struct ::lwdaArray * in a LWCA Driver API function which takes a ::LWarray,
 * it is necessary to explicitly cast the struct ::lwdaArray * to a ::LWarray .
 *
 * \section LWDART_LWDA_graphicsResource Interactions between LWgraphicsResource and lwdaGraphicsResource_t
 *
 * The types ::LWgraphicsResource and ::lwdaGraphicsResource_t represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::LWgraphicsResource in a LWCA Runtime API function which takes a 
 * ::lwdaGraphicsResource_t, it is necessary to explicitly cast the ::LWgraphicsResource 
 * to a ::lwdaGraphicsResource_t.
 *
 * In order to use a ::lwdaGraphicsResource_t in a LWCA Driver API function which takes a
 * ::LWgraphicsResource, it is necessary to explicitly cast the ::lwdaGraphicsResource_t 
 * to a ::LWgraphicsResource.
 *
 * @}
 */

#if defined(__LWDA_API_VERSION_INTERNAL)
    #undef lwdaMemcpy
    #undef lwdaMemcpyToSymbol
    #undef lwdaMemcpyFromSymbol
    #undef lwdaMemcpy2D
    #undef lwdaMemcpyToArray
    #undef lwdaMemcpy2DToArray
    #undef lwdaMemcpyFromArray
    #undef lwdaMemcpy2DFromArray
    #undef lwdaMemcpyArrayToArray
    #undef lwdaMemcpy2DArrayToArray
    #undef lwdaMemcpy3D
    #undef lwdaMemcpy3DPeer
    #undef lwdaMemset
    #undef lwdaMemset2D
    #undef lwdaMemset3D
    #undef lwdaMemcpyAsync
    #undef lwdaMemcpyToSymbolAsync
    #undef lwdaMemcpyFromSymbolAsync
    #undef lwdaMemcpy2DAsync
    #undef lwdaMemcpyToArrayAsync
    #undef lwdaMemcpy2DToArrayAsync
    #undef lwdaMemcpyFromArrayAsync
    #undef lwdaMemcpy2DFromArrayAsync
    #undef lwdaMemcpy3DAsync
    #undef lwdaMemcpy3DPeerAsync
    #undef lwdaMemsetAsync
    #undef lwdaMemset2DAsync
    #undef lwdaMemset3DAsync
    #undef lwdaStreamQuery
    #undef lwdaStreamGetFlags
    #undef lwdaStreamGetPriority
    #undef lwdaEventRecord
    #undef lwdaStreamWaitEvent
    #undef lwdaStreamAddCallback
    #undef lwdaStreamAttachMemAsync
    #undef lwdaStreamSynchronize
    #undef lwdaLaunch
    #undef lwdaLaunchKernel
    #undef lwdaMemPrefetchAsync
    #undef lwdaLaunchCooperativeKernel
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy(void *dst, const void *src, size_t count, enum lwdaMemcpyKind kind);
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset __dv(0), enum lwdaMemcpyKind kind __dv(lwdaMemcpyHostToDevice));
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset __dv(0), enum lwdaMemcpyKind kind __dv(lwdaMemcpyDeviceToHost));
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum lwdaMemcpyKind kind);
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyToArray(lwdaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum lwdaMemcpyKind kind);
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy2DToArray(lwdaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum lwdaMemcpyKind kind);
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyFromArray(void *dst, lwdaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum lwdaMemcpyKind kind);
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy2DFromArray(void *dst, size_t dpitch, lwdaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum lwdaMemcpyKind kind);
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyArrayToArray(lwdaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, lwdaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum lwdaMemcpyKind kind __dv(lwdaMemcpyDeviceToDevice));
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy2DArrayToArray(lwdaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, lwdaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum lwdaMemcpyKind kind __dv(lwdaMemcpyDeviceToDevice));
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy3D(const struct lwdaMemcpy3DParms *p);
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy3DPeer(const struct lwdaMemcpy3DPeerParms *p);
    extern __host__ lwdaError_t LWDARTAPI lwdaMemset(void *devPtr, int value, size_t count);
    extern __host__ lwdaError_t LWDARTAPI lwdaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);
    extern __host__ lwdaError_t LWDARTAPI lwdaMemset3D(struct lwdaPitchedPtr pitchedDevPtr, int value, struct lwdaExtent extent);
    extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaMemcpyAsync(void *dst, const void *src, size_t count, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));
    extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyToArrayAsync(lwdaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy2DToArrayAsync(lwdaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpyFromArrayAsync(void *dst, lwdaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, lwdaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum lwdaMemcpyKind kind, lwdaStream_t stream __dv(0));
    extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaMemcpy3DAsync(const struct lwdaMemcpy3DParms *p, lwdaStream_t stream __dv(0));
    extern __host__ lwdaError_t LWDARTAPI lwdaMemcpy3DPeerAsync(const struct lwdaMemcpy3DPeerParms *p, lwdaStream_t stream __dv(0));
    extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaMemsetAsync(void *devPtr, int value, size_t count, lwdaStream_t stream __dv(0));
    extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, lwdaStream_t stream __dv(0));
    extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaMemset3DAsync(struct lwdaPitchedPtr pitchedDevPtr, int value, struct lwdaExtent extent, lwdaStream_t stream __dv(0));
    extern __host__ lwdaError_t LWDARTAPI lwdaStreamQuery(lwdaStream_t stream);
    extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaStreamGetFlags(lwdaStream_t hStream, unsigned int *flags);
    extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaStreamGetPriority(lwdaStream_t hStream, int *priority);
    extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaEventRecord(lwdaEvent_t event, lwdaStream_t stream __dv(0));
    extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaStreamWaitEvent(lwdaStream_t stream, lwdaEvent_t event, unsigned int flags);
    extern __host__ lwdaError_t LWDARTAPI lwdaStreamAddCallback(lwdaStream_t stream, lwdaStreamCallback_t callback, void *userData, unsigned int flags);
    extern __host__ __lwdart_builtin__ lwdaError_t LWDARTAPI lwdaStreamAttachMemAsync(lwdaStream_t stream, void *devPtr, size_t length, unsigned int flags);
    extern __host__ lwdaError_t LWDARTAPI lwdaStreamSynchronize(lwdaStream_t stream);
    extern __host__ lwdaError_t LWDARTAPI lwdaLaunch(const void *func);
    extern __host__ lwdaError_t LWDARTAPI lwdaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, lwdaStream_t stream);
    extern __host__ lwdaError_t LWDARTAPI lwdaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, lwdaStream_t stream);
    extern __host__ lwdaError_t LWDARTAPI lwdaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, lwdaStream_t stream);
#elif defined(__LWDART_API_PER_THREAD_DEFAULT_STREAM)
    // lwcc stubs reference the 'lwdaLaunch' identifier even if it was defined
    // to 'lwdaLaunch_ptsz'. Redirect through a static inline function.
    #undef lwdaLaunch
    static __inline__ __host__ lwdaError_t lwdaLaunch(const void *func)
    {
        return lwdaLaunch_ptsz(func);
    }
    #define lwdaLaunch __LWDART_API_PTSZ(lwdaLaunch)
#endif

#if defined(__cplusplus)
}

#endif /* __cplusplus */

#undef __dv

#endif /* !__LWDA_RUNTIME_API_H__ */
