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

#ifndef __lwda_lwda_h__
#define __lwda_lwda_h__

#include <stdlib.h>
#ifdef _MSC_VER
typedef unsigned __int32 lwuint32_t;
typedef unsigned __int64 lwuint64_t;
#else
#include <stdint.h>
typedef uint32_t lwuint32_t;
typedef uint64_t lwuint64_t;
#endif

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
    #define __LWDA_API_VERSION 9000
#endif /* LWDA_FORCE_API_VERSION */

#if defined(__LWDA_API_VERSION_INTERNAL) || defined(LWDA_API_PER_THREAD_DEFAULT_STREAM)
    #define __LWDA_API_PER_THREAD_DEFAULT_STREAM
    #define __LWDA_API_PTDS(api) api ## _ptds
    #define __LWDA_API_PTSZ(api) api ## _ptsz
#else
    #define __LWDA_API_PTDS(api) api
    #define __LWDA_API_PTSZ(api) api
#endif

#if defined(__LWDA_API_VERSION_INTERNAL) || __LWDA_API_VERSION >= 3020
    #define lwDeviceTotalMem                    lwDeviceTotalMem_v2
    #define lwCtxCreate                         lwCtxCreate_v2
    #define lwModuleGetGlobal                   lwModuleGetGlobal_v2
    #define lwMemGetInfo                        lwMemGetInfo_v2
    #define lwMemAlloc                          lwMemAlloc_v2
    #define lwMemAllocPitch                     lwMemAllocPitch_v2
    #define lwMemFree                           lwMemFree_v2
    #define lwMemGetAddressRange                lwMemGetAddressRange_v2
    #define lwMemAllocHost                      lwMemAllocHost_v2
    #define lwMemHostGetDevicePointer           lwMemHostGetDevicePointer_v2
    #define lwMemcpyHtoD                        __LWDA_API_PTDS(lwMemcpyHtoD_v2)
    #define lwMemcpyDtoH                        __LWDA_API_PTDS(lwMemcpyDtoH_v2)
    #define lwMemcpyDtoD                        __LWDA_API_PTDS(lwMemcpyDtoD_v2)
    #define lwMemcpyDtoA                        __LWDA_API_PTDS(lwMemcpyDtoA_v2)
    #define lwMemcpyAtoD                        __LWDA_API_PTDS(lwMemcpyAtoD_v2)
    #define lwMemcpyHtoA                        __LWDA_API_PTDS(lwMemcpyHtoA_v2)
    #define lwMemcpyAtoH                        __LWDA_API_PTDS(lwMemcpyAtoH_v2)
    #define lwMemcpyAtoA                        __LWDA_API_PTDS(lwMemcpyAtoA_v2)
    #define lwMemcpyHtoAAsync                   __LWDA_API_PTSZ(lwMemcpyHtoAAsync_v2)
    #define lwMemcpyAtoHAsync                   __LWDA_API_PTSZ(lwMemcpyAtoHAsync_v2)
    #define lwMemcpy2D                          __LWDA_API_PTDS(lwMemcpy2D_v2)
    #define lwMemcpy2DUnaligned                 __LWDA_API_PTDS(lwMemcpy2DUnaligned_v2)
    #define lwMemcpy3D                          __LWDA_API_PTDS(lwMemcpy3D_v2)
    #define lwMemcpyHtoDAsync                   __LWDA_API_PTSZ(lwMemcpyHtoDAsync_v2)
    #define lwMemcpyDtoHAsync                   __LWDA_API_PTSZ(lwMemcpyDtoHAsync_v2)
    #define lwMemcpyDtoDAsync                   __LWDA_API_PTSZ(lwMemcpyDtoDAsync_v2)
    #define lwMemcpy2DAsync                     __LWDA_API_PTSZ(lwMemcpy2DAsync_v2)
    #define lwMemcpy3DAsync                     __LWDA_API_PTSZ(lwMemcpy3DAsync_v2)
    #define lwMemsetD8                          __LWDA_API_PTDS(lwMemsetD8_v2)
    #define lwMemsetD16                         __LWDA_API_PTDS(lwMemsetD16_v2)
    #define lwMemsetD32                         __LWDA_API_PTDS(lwMemsetD32_v2)
    #define lwMemsetD2D8                        __LWDA_API_PTDS(lwMemsetD2D8_v2)
    #define lwMemsetD2D16                       __LWDA_API_PTDS(lwMemsetD2D16_v2)
    #define lwMemsetD2D32                       __LWDA_API_PTDS(lwMemsetD2D32_v2)
    #define lwArrayCreate                       lwArrayCreate_v2
    #define lwArrayGetDescriptor                lwArrayGetDescriptor_v2
    #define lwArray3DCreate                     lwArray3DCreate_v2
    #define lwArray3DGetDescriptor              lwArray3DGetDescriptor_v2
    #define lwTexRefSetAddress                  lwTexRefSetAddress_v2
    #define lwTexRefGetAddress                  lwTexRefGetAddress_v2
    #define lwGraphicsResourceGetMappedPointer  lwGraphicsResourceGetMappedPointer_v2
#endif /* __LWDA_API_VERSION_INTERNAL || __LWDA_API_VERSION >= 3020 */
#if defined(__LWDA_API_VERSION_INTERNAL) || __LWDA_API_VERSION >= 4000
    #define lwCtxDestroy                        lwCtxDestroy_v2
    #define lwCtxPopLwrrent                     lwCtxPopLwrrent_v2
    #define lwCtxPushLwrrent                    lwCtxPushLwrrent_v2
    #define lwStreamDestroy                     lwStreamDestroy_v2
    #define lwEventDestroy                      lwEventDestroy_v2
#endif /* __LWDA_API_VERSION_INTERNAL || __LWDA_API_VERSION >= 4000 */
#if defined(__LWDA_API_VERSION_INTERNAL) || __LWDA_API_VERSION >= 4010
    #define lwTexRefSetAddress2D                lwTexRefSetAddress2D_v3
#endif /* __LWDA_API_VERSION_INTERNAL || __LWDA_API_VERSION >= 4010 */
#if defined(__LWDA_API_VERSION_INTERNAL) || __LWDA_API_VERSION >= 6050
    #define lwLinkCreate                        lwLinkCreate_v2
    #define lwLinkAddData                       lwLinkAddData_v2
    #define lwLinkAddFile                       lwLinkAddFile_v2
#endif /* __LWDA_API_VERSION_INTERNAL || __LWDA_API_VERSION >= 6050 */
#if defined(__LWDA_API_VERSION_INTERNAL) || __LWDA_API_VERSION >= 6050
    #define lwMemHostRegister                   lwMemHostRegister_v2
    #define lwGraphicsResourceSetMapFlags       lwGraphicsResourceSetMapFlags_v2
#endif /* __LWDA_API_VERSION_INTERNAL || __LWDA_API_VERSION >= 6050 */

#if !defined(__LWDA_API_VERSION_INTERNAL)
#if defined(__LWDA_API_VERSION) && __LWDA_API_VERSION >= 3020 && __LWDA_API_VERSION < 4010
    #define lwTexRefSetAddress2D                lwTexRefSetAddress2D_v2
#endif /* __LWDA_API_VERSION && __LWDA_API_VERSION >= 3020 && __LWDA_API_VERSION < 4010 */
#endif /* __LWDA_API_VERSION_INTERNAL */

#if defined(__LWDA_API_PER_THREAD_DEFAULT_STREAM)
    #define lwMemcpy                            __LWDA_API_PTDS(lwMemcpy)
    #define lwMemcpyAsync                       __LWDA_API_PTSZ(lwMemcpyAsync)
    #define lwMemcpyPeer                        __LWDA_API_PTDS(lwMemcpyPeer)
    #define lwMemcpyPeerAsync                   __LWDA_API_PTSZ(lwMemcpyPeerAsync)
    #define lwMemcpy3DPeer                      __LWDA_API_PTDS(lwMemcpy3DPeer)
    #define lwMemcpy3DPeerAsync                 __LWDA_API_PTSZ(lwMemcpy3DPeerAsync)
    #define lwMemPrefetchAsync                  __LWDA_API_PTSZ(lwMemPrefetchAsync)

    #define lwMemsetD8Async                     __LWDA_API_PTSZ(lwMemsetD8Async)
    #define lwMemsetD16Async                    __LWDA_API_PTSZ(lwMemsetD16Async)
    #define lwMemsetD32Async                    __LWDA_API_PTSZ(lwMemsetD32Async)
    #define lwMemsetD2D8Async                   __LWDA_API_PTSZ(lwMemsetD2D8Async)
    #define lwMemsetD2D16Async                  __LWDA_API_PTSZ(lwMemsetD2D16Async)
    #define lwMemsetD2D32Async                  __LWDA_API_PTSZ(lwMemsetD2D32Async)

    #define lwStreamGetPriority                 __LWDA_API_PTSZ(lwStreamGetPriority)
    #define lwStreamGetFlags                    __LWDA_API_PTSZ(lwStreamGetFlags)
    #define lwStreamWaitEvent                   __LWDA_API_PTSZ(lwStreamWaitEvent)
    #define lwStreamAddCallback                 __LWDA_API_PTSZ(lwStreamAddCallback)
    #define lwStreamAttachMemAsync              __LWDA_API_PTSZ(lwStreamAttachMemAsync)
    #define lwStreamQuery                       __LWDA_API_PTSZ(lwStreamQuery)
    #define lwStreamSynchronize                 __LWDA_API_PTSZ(lwStreamSynchronize)
    #define lwEventRecord                       __LWDA_API_PTSZ(lwEventRecord)
    #define lwLaunchKernel                      __LWDA_API_PTSZ(lwLaunchKernel)
    #define lwGraphicsMapResources              __LWDA_API_PTSZ(lwGraphicsMapResources)
    #define lwGraphicsUnmapResources            __LWDA_API_PTSZ(lwGraphicsUnmapResources)

    #define lwStreamWriteValue32                __LWDA_API_PTSZ(lwStreamWriteValue32)
    #define lwStreamWaitValue32                 __LWDA_API_PTSZ(lwStreamWaitValue32)
    #define lwStreamWriteValue64                __LWDA_API_PTSZ(lwStreamWriteValue64)
    #define lwStreamWaitValue64                 __LWDA_API_PTSZ(lwStreamWaitValue64)
    #define lwStreamBatchMemOp                  __LWDA_API_PTSZ(lwStreamBatchMemOp)

    #define lwLaunchCooperativeKernel           __LWDA_API_PTSZ(lwLaunchCooperativeKernel)

#endif

/**
 * \file lwca.h
 * \brief Header file for the LWCA Toolkit application programming interface.
 *
 * \file lwdaGL.h
 * \brief Header file for the OpenGL interoperability functions of the
 * low-level LWCA driver application programming interface.
 *
 * \file lwdaD3D9.h
 * \brief Header file for the Direct3D 9 interoperability functions of the
 * low-level LWCA driver application programming interface.
 */

/**
 * \defgroup LWDA_TYPES Data types used by LWCA driver
 * @{
 */

/**
 * LWCA API version number
 */
#define LWDA_VERSION 9000

#ifdef __cplusplus
extern "C" {
#endif

/**
 * LWCA device pointer
 * LWdeviceptr is defined as an unsigned integer type whose size matches the size of a pointer on the target platform.
 */ 
#if __LWDA_API_VERSION >= 3020

#if defined(_WIN64) || defined(__LP64__)
typedef unsigned long long LWdeviceptr;
#else
typedef unsigned int LWdeviceptr;
#endif

#endif /* __LWDA_API_VERSION >= 3020 */

typedef int LWdevice;                                     /**< LWCA device */
typedef struct LWctx_st *LWcontext;                       /**< LWCA context */
typedef struct LWmod_st *LWmodule;                        /**< LWCA module */
typedef struct LWfunc_st *LWfunction;                     /**< LWCA function */
typedef struct LWarray_st *LWarray;                       /**< LWCA array */
typedef struct LWmipmappedArray_st *LWmipmappedArray;     /**< LWCA mipmapped array */
typedef struct LWtexref_st *LWtexref;                     /**< LWCA texture reference */
typedef struct LWsurfref_st *LWsurfref;                   /**< LWCA surface reference */
typedef struct LWevent_st *LWevent;                       /**< LWCA event */
typedef struct LWstream_st *LWstream;                     /**< LWCA stream */
typedef struct LWgraphicsResource_st *LWgraphicsResource; /**< LWCA graphics interop resource */
typedef unsigned long long LWtexObject;                   /**< An opaque value that represents a LWCA texture object */
typedef unsigned long long LWsurfObject;                  /**< An opaque value that represents a LWCA surface object */

typedef struct LWuuid_st {                                /**< LWCA definition of UUID */
    char bytes[16];
} LWuuid;


#if __LWDA_API_VERSION >= 4010

/**
 * LWCA IPC handle size 
 */
#define LW_IPC_HANDLE_SIZE 64

/**
 * LWCA IPC event handle
 */
typedef struct LWipcEventHandle_st {
    char reserved[LW_IPC_HANDLE_SIZE];
} LWipcEventHandle;

/**
 * LWCA IPC mem handle
 */
typedef struct LWipcMemHandle_st {
    char reserved[LW_IPC_HANDLE_SIZE];
} LWipcMemHandle;

/**
 * LWCA Ipc Mem Flags
 */
typedef enum LWipcMem_flags_enum {
    LW_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1 /**< Automatically enable peer access between remote devices as needed */
} LWipcMem_flags;

#endif

/**
 * LWCA Mem Attach Flags
 */
typedef enum LWmemAttach_flags_enum {
    LW_MEM_ATTACH_GLOBAL = 0x1, /**< Memory can be accessed by any stream on any device */
    LW_MEM_ATTACH_HOST   = 0x2, /**< Memory cannot be accessed by any stream on any device */
    LW_MEM_ATTACH_SINGLE = 0x4  /**< Memory can only be accessed by a single stream on the associated device */
} LWmemAttach_flags;

/**
 * Context creation flags
 */
typedef enum LWctx_flags_enum {
    LW_CTX_SCHED_AUTO          = 0x00, /**< Automatic scheduling */
    LW_CTX_SCHED_SPIN          = 0x01, /**< Set spin as default scheduling */
    LW_CTX_SCHED_YIELD         = 0x02, /**< Set yield as default scheduling */
    LW_CTX_SCHED_BLOCKING_SYNC = 0x04, /**< Set blocking synchronization as default scheduling */
    LW_CTX_BLOCKING_SYNC       = 0x04, /**< Set blocking synchronization as default scheduling
                                         *  \deprecated This flag was deprecated as of LWCA 4.0
                                         *  and was replaced with ::LW_CTX_SCHED_BLOCKING_SYNC. */
    LW_CTX_SCHED_MASK          = 0x07, 
    LW_CTX_MAP_HOST            = 0x08, /**< Support mapped pinned allocations */
    LW_CTX_LMEM_RESIZE_TO_MAX  = 0x10, /**< Keep local memory allocation after launch */
    LW_CTX_FLAGS_MASK          = 0x1f
} LWctx_flags;

/**
 * Stream creation flags
 */
typedef enum LWstream_flags_enum {
    LW_STREAM_DEFAULT      = 0x0, /**< Default stream flag */
    LW_STREAM_NON_BLOCKING = 0x1  /**< Stream does not synchronize with stream 0 (the NULL stream) */
} LWstream_flags;

/**
 * Legacy stream handle
 *
 * Stream handle that can be passed as a LWstream to use an implicit stream
 * with legacy synchronization behavior.
 *
 * See details of the \link_sync_behavior
 */
#define LW_STREAM_LEGACY     ((LWstream)0x1)

/**
 * Per-thread stream handle
 *
 * Stream handle that can be passed as a LWstream to use an implicit stream
 * with per-thread synchronization behavior.
 *
 * See details of the \link_sync_behavior
 */
#define LW_STREAM_PER_THREAD ((LWstream)0x2)

/**
 * Event creation flags
 */
typedef enum LWevent_flags_enum {
    LW_EVENT_DEFAULT        = 0x0, /**< Default event flag */
    LW_EVENT_BLOCKING_SYNC  = 0x1, /**< Event uses blocking synchronization */
    LW_EVENT_DISABLE_TIMING = 0x2, /**< Event will not record timing data */
    LW_EVENT_INTERPROCESS   = 0x4  /**< Event is suitable for interprocess use. LW_EVENT_DISABLE_TIMING must be set */
} LWevent_flags;

#if __LWDA_API_VERSION >= 8000
/**
 * Flags for ::lwStreamWaitValue32 and ::lwStreamWaitValue64
 */
typedef enum LWstreamWaitValue_flags_enum {
    LW_STREAM_WAIT_VALUE_GEQ   = 0x0,   /**< Wait until (int32_t)(*addr - value) >= 0 (or int64_t for 64 bit
                                             values). Note this is a cyclic comparison which ignores wraparound.
                                             (Default behavior.) */
    LW_STREAM_WAIT_VALUE_EQ    = 0x1,   /**< Wait until *addr == value. */
    LW_STREAM_WAIT_VALUE_AND   = 0x2,   /**< Wait until (*addr & value) != 0. */
    LW_STREAM_WAIT_VALUE_NOR   = 0x3,   /**< Wait until ~(*addr | value) != 0. Support for this operation can be
                                             queried with ::lwDeviceGetAttribute() and
                                             ::LW_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR. Generally, this
                                             requires compute capability 7.0 or greater. */
    LW_STREAM_WAIT_VALUE_FLUSH = 1<<30  /**< Follow the wait operation with a flush of outstanding remote writes. This
                                             means that, if a remote write operation is guaranteed to have reached the
                                             device before the wait can be satisfied, that write is guaranteed to be
                                             visible to downstream device work. The device is permitted to reorder
                                             remote writes internally. For example, this flag would be required if
                                             two remote writes arrive in a defined order, the wait is satisfied by the
                                             second write, and downstream work needs to observe the first write. */
} LWstreamWaitValue_flags;

/**
 * Flags for ::lwStreamWriteValue32
 */
typedef enum LWstreamWriteValue_flags_enum {
    LW_STREAM_WRITE_VALUE_DEFAULT           = 0x0, /**< Default behavior */
    LW_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 0x1  /**< Permits the write to be reordered with writes which were issued
                                                        before it, as a performance optimization. Normally,
                                                        ::lwStreamWriteValue32 will provide a memory fence before the
                                                        write, which has similar semantics to
                                                        __threadfence_system() but is scoped to the stream
                                                        rather than a LWCA thread. */
} LWstreamWriteValue_flags;

/**
 * Operations for ::lwStreamBatchMemOp
 */
typedef enum LWstreamBatchMemOpType_enum {
    LW_STREAM_MEM_OP_WAIT_VALUE_32  = 1,     /**< Represents a ::lwStreamWaitValue32 operation */
    LW_STREAM_MEM_OP_WRITE_VALUE_32 = 2,     /**< Represents a ::lwStreamWriteValue32 operation */
    LW_STREAM_MEM_OP_WAIT_VALUE_64  = 4,     /**< Represents a ::lwStreamWaitValue64 operation */
    LW_STREAM_MEM_OP_WRITE_VALUE_64 = 5,     /**< Represents a ::lwStreamWriteValue64 operation */
    LW_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3 /**< This has the same effect as ::LW_STREAM_WAIT_VALUE_FLUSH, but as a
                                                  standalone operation. */
} LWstreamBatchMemOpType;

/**
 * Per-operation parameters for ::lwStreamBatchMemOp
 */
typedef union LWstreamBatchMemOpParams_union {
    LWstreamBatchMemOpType operation;
    struct LWstreamMemOpWaitValueParams_st {
        LWstreamBatchMemOpType operation;
        LWdeviceptr address;
        union {
            lwuint32_t value;
            lwuint64_t value64;
        };
        unsigned int flags;
        LWdeviceptr alias; /**< For driver internal use. Initial value is unimportant. */
    } waitValue;
    struct LWstreamMemOpWriteValueParams_st {
        LWstreamBatchMemOpType operation;
        LWdeviceptr address;
        union {
            lwuint32_t value;
            lwuint64_t value64;
        };
        unsigned int flags;
        LWdeviceptr alias; /**< For driver internal use. Initial value is unimportant. */
    } writeValue;
    struct LWstreamMemOpFlushRemoteWritesParams_st {
        LWstreamBatchMemOpType operation;
        unsigned int flags;
    } flushRemoteWrites;
    lwuint64_t pad[6];
} LWstreamBatchMemOpParams;
#endif /* __LWDA_API_VERSION >= 8000 */

/**
 * Oclwpancy calculator flag
 */
typedef enum LWoclwpancy_flags_enum {
    LW_OCLWPANCY_DEFAULT                  = 0x0, /**< Default behavior */
    LW_OCLWPANCY_DISABLE_CACHING_OVERRIDE = 0x1  /**< Assume global caching is enabled and cannot be automatically turned off */
} LWoclwpancy_flags;

/**
 * Array formats
 */
typedef enum LWarray_format_enum {
    LW_AD_FORMAT_UNSIGNED_INT8  = 0x01, /**< Unsigned 8-bit integers */
    LW_AD_FORMAT_UNSIGNED_INT16 = 0x02, /**< Unsigned 16-bit integers */
    LW_AD_FORMAT_UNSIGNED_INT32 = 0x03, /**< Unsigned 32-bit integers */
    LW_AD_FORMAT_SIGNED_INT8    = 0x08, /**< Signed 8-bit integers */
    LW_AD_FORMAT_SIGNED_INT16   = 0x09, /**< Signed 16-bit integers */
    LW_AD_FORMAT_SIGNED_INT32   = 0x0a, /**< Signed 32-bit integers */
    LW_AD_FORMAT_HALF           = 0x10, /**< 16-bit floating point */
    LW_AD_FORMAT_FLOAT          = 0x20  /**< 32-bit floating point */
} LWarray_format;

/**
 * Texture reference addressing modes
 */
typedef enum LWaddress_mode_enum {
    LW_TR_ADDRESS_MODE_WRAP   = 0, /**< Wrapping address mode */
    LW_TR_ADDRESS_MODE_CLAMP  = 1, /**< Clamp to edge address mode */
    LW_TR_ADDRESS_MODE_MIRROR = 2, /**< Mirror address mode */
    LW_TR_ADDRESS_MODE_BORDER = 3  /**< Border address mode */
} LWaddress_mode;

/**
 * Texture reference filtering modes
 */
typedef enum LWfilter_mode_enum {
    LW_TR_FILTER_MODE_POINT  = 0, /**< Point filter mode */
    LW_TR_FILTER_MODE_LINEAR = 1  /**< Linear filter mode */
} LWfilter_mode;

/**
 * Device properties
 */
typedef enum LWdevice_attribute_enum {
    LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,              /**< Maximum number of threads per block */
    LW_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,                    /**< Maximum block dimension X */
    LW_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,                    /**< Maximum block dimension Y */
    LW_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,                    /**< Maximum block dimension Z */
    LW_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,                     /**< Maximum grid dimension X */
    LW_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,                     /**< Maximum grid dimension Y */
    LW_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,                     /**< Maximum grid dimension Z */
    LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,        /**< Maximum shared memory available per block in bytes */
    LW_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,            /**< Deprecated, use LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK */
    LW_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,              /**< Memory available on device for __constant__ variables in a LWCA C kernel in bytes */
    LW_DEVICE_ATTRIBUTE_WARP_SIZE = 10,                         /**< Warp size in threads */
    LW_DEVICE_ATTRIBUTE_MAX_PITCH = 11,                         /**< Maximum pitch in bytes allowed by memory copies */
    LW_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,           /**< Maximum number of 32-bit registers available per block */
    LW_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,               /**< Deprecated, use LW_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK */
    LW_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,                        /**< Typical clock frequency in kilohertz */
    LW_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,                 /**< Alignment requirement for textures */
    LW_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,                       /**< Device can possibly copy memory and execute a kernel conlwrrently. Deprecated. Use instead LW_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT. */
    LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,              /**< Number of multiprocessors on device */
    LW_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,               /**< Specifies whether there is a run time limit on kernels */
    LW_DEVICE_ATTRIBUTE_INTEGRATED = 18,                        /**< Device is integrated with host memory */
    LW_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,               /**< Device can map host memory into LWCA address space */
    LW_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,                      /**< Compute mode (See ::LWcomputemode for details) */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,           /**< Maximum 1D texture width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,           /**< Maximum 2D texture width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,          /**< Maximum 2D texture height */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,           /**< Maximum 3D texture width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,          /**< Maximum 3D texture height */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,           /**< Maximum 3D texture depth */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,   /**< Maximum 2D layered texture width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,  /**< Maximum 2D layered texture height */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,  /**< Maximum layers in a 2D layered texture */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,     /**< Deprecated, use LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,    /**< Deprecated, use LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29, /**< Deprecated, use LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS */
    LW_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,                 /**< Alignment requirement for surfaces */
    LW_DEVICE_ATTRIBUTE_CONLWRRENT_KERNELS = 31,                /**< Device can possibly execute multiple kernels conlwrrently */
    LW_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,                       /**< Device has ECC support enabled */
    LW_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,                        /**< PCI bus ID of the device */
    LW_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,                     /**< PCI device ID of the device */
    LW_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,                        /**< Device is using TCC driver model */
    LW_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,                 /**< Peak memory clock frequency in kilohertz */
    LW_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,           /**< Global memory bus width in bits */
    LW_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,                     /**< Size of L2 cache in bytes */
    LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,    /**< Maximum resident threads per multiprocessor */
    LW_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,                /**< Number of asynchronous engines */
    LW_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,                /**< Device shares a unified address space with the host */    
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,   /**< Maximum 1D layered texture width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,  /**< Maximum layers in a 1D layered texture */
    LW_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,                  /**< Deprecated, do not use. */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,    /**< Maximum 2D texture width if LWDA_ARRAY3D_TEXTURE_GATHER is set */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,   /**< Maximum 2D texture height if LWDA_ARRAY3D_TEXTURE_GATHER is set */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47, /**< Alternate maximum 3D texture width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,/**< Alternate maximum 3D texture height */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49, /**< Alternate maximum 3D texture depth */
    LW_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,                     /**< PCI domain ID of the device */
    LW_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,           /**< Pitch alignment requirement for textures */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURELWBEMAP_WIDTH = 52,      /**< Maximum lwbemap texture width/height */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURELWBEMAP_LAYERED_WIDTH = 53,  /**< Maximum lwbemap layered texture width/height */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURELWBEMAP_LAYERED_LAYERS = 54, /**< Maximum layers in a lwbemap layered texture */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,           /**< Maximum 1D surface width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,           /**< Maximum 2D surface width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,          /**< Maximum 2D surface height */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,           /**< Maximum 3D surface width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,          /**< Maximum 3D surface height */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,           /**< Maximum 3D surface depth */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,   /**< Maximum 1D layered surface width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,  /**< Maximum layers in a 1D layered surface */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,   /**< Maximum 2D layered surface width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,  /**< Maximum 2D layered surface height */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,  /**< Maximum layers in a 2D layered surface */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACELWBEMAP_WIDTH = 66,      /**< Maximum lwbemap surface width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACELWBEMAP_LAYERED_WIDTH = 67,  /**< Maximum lwbemap layered surface width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACELWBEMAP_LAYERED_LAYERS = 68, /**< Maximum layers in a lwbemap layered surface */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,    /**< Maximum 1D linear texture width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,    /**< Maximum 2D linear texture width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,   /**< Maximum 2D linear texture height */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,    /**< Maximum 2D linear texture pitch in bytes */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73, /**< Maximum mipmapped 2D texture width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,/**< Maximum mipmapped 2D texture height */
    LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,          /**< Major compute capability version number */     
    LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,          /**< Minor compute capability version number */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77, /**< Maximum mipmapped 1D texture width */
    LW_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,       /**< Device supports stream priorities */
    LW_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,         /**< Device supports caching globals in L1 */
    LW_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,          /**< Device supports caching locals in L1 */
    LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,  /**< Maximum shared memory available per multiprocessor in bytes */
    LW_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,  /**< Maximum number of 32-bit registers available per multiprocessor */
    LW_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,                    /**< Device can allocate managed memory on this system */
    LW_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,                    /**< Device is on a multi-GPU board */ 
    LW_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,           /**< Unique id for a group of devices on the same multi-GPU board */
    LW_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,       /**< Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)*/
    LW_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,  /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    LW_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,            /**< Device supports coherently accessing pageable memory without calling lwdaHostRegister on it */
    LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS = 89,         /**< Device can coherently access managed memory conlwrrently with the CPU */
    LW_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,      /**< Device supports compute preemption. */
    LW_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91, /**< Device can access host registered memory at the same virtual address as the CPU */
    LW_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,            /**< ::lwStreamBatchMemOp and related APIs are supported. */
    LW_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,     /**< 64-bit operations are supported in ::lwStreamBatchMemOp and related APIs. */
    LW_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,     /**< ::LW_STREAM_WAIT_VALUE_NOR is supported. */
    LW_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,                /**< Device supports launching cooperative kernels via ::lwLaunchCooperativeKernel */
    LW_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,   /**< Device can participate in cooperative kernels launched via ::lwLaunchCooperativeKernelMultiDevice */
    LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97, /**< Maximum optin shared memory per block */
    LW_DEVICE_ATTRIBUTE_MAX
} LWdevice_attribute;

/**
 * Legacy device properties
 */
typedef struct LWdevprop_st {
    int maxThreadsPerBlock;     /**< Maximum number of threads per block */
    int maxThreadsDim[3];       /**< Maximum size of each dimension of a block */
    int maxGridSize[3];         /**< Maximum size of each dimension of a grid */
    int sharedMemPerBlock;      /**< Shared memory available per block in bytes */
    int totalConstantMemory;    /**< Constant memory available on device in bytes */
    int SIMDWidth;              /**< Warp size in threads */
    int memPitch;               /**< Maximum pitch in bytes allowed by memory copies */
    int regsPerBlock;           /**< 32-bit registers available per block */
    int clockRate;              /**< Clock frequency in kilohertz */
    int textureAlign;           /**< Alignment requirement for textures */
} LWdevprop;

/**
 * Pointer information
 */
typedef enum LWpointer_attribute_enum {
    LW_POINTER_ATTRIBUTE_CONTEXT = 1,        /**< The ::LWcontext on which a pointer was allocated or registered */
    LW_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,    /**< The ::LWmemorytype describing the physical location of a pointer */
    LW_POINTER_ATTRIBUTE_DEVICE_POINTER = 3, /**< The address at which a pointer's memory may be accessed on the device */
    LW_POINTER_ATTRIBUTE_HOST_POINTER = 4,   /**< The address at which a pointer's memory may be accessed on the host */
    LW_POINTER_ATTRIBUTE_P2P_TOKENS = 5,     /**< A pair of tokens for use with the lw-p2p.h Linux kernel interface */
    LW_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,    /**< Synchronize every synchronous memory operation initiated on this region */
    LW_POINTER_ATTRIBUTE_BUFFER_ID = 7,      /**< A process-wide unique ID for an allocated memory region*/
    LW_POINTER_ATTRIBUTE_IS_MANAGED = 8      /**< Indicates if the pointer points to managed memory */
} LWpointer_attribute;

/**
 * Function properties
 */
typedef enum LWfunction_attribute_enum {
    /**
     * The maximum number of threads per block, beyond which a launch of the
     * function would fail. This number depends on both the function and the
     * device on which the function is lwrrently loaded.
     */
    LW_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,

    /**
     * The size in bytes of statically-allocated shared memory required by
     * this function. This does not include dynamically-allocated shared
     * memory requested by the user at runtime.
     */
    LW_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,

    /**
     * The size in bytes of user-allocated constant memory required by this
     * function.
     */
    LW_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,

    /**
     * The size in bytes of local memory used by each thread of this function.
     */
    LW_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,

    /**
     * The number of registers used by each thread of this function.
     */
    LW_FUNC_ATTRIBUTE_NUM_REGS = 4,

    /**
     * The PTX virtual architecture version for which the function was
     * compiled. This value is the major PTX version * 10 + the minor PTX
     * version, so a PTX version 1.3 function would return the value 13.
     * Note that this may return the undefined value of 0 for lwbins
     * compiled prior to LWCA 3.0.
     */
    LW_FUNC_ATTRIBUTE_PTX_VERSION = 5,

    /**
     * The binary architecture version for which the function was compiled.
     * This value is the major binary version * 10 + the minor binary version,
     * so a binary version 1.3 function would return the value 13. Note that
     * this will return a value of 10 for legacy lwbins that do not have a
     * properly-encoded binary architecture version.
     */
    LW_FUNC_ATTRIBUTE_BINARY_VERSION = 6,

    /**
     * The attribute to indicate whether the function has been compiled with 
     * user specified option "-Xptxas --dlcm=ca" set .
     */
    LW_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,

    /**
     * The maximum size in bytes of dynamically-allocated shared memory that can be used by
     * this function. If the user-specified dynamic shared memory size is larger than this
     * value, the launch will fail.
     */
    LW_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,

    /**
     * On devices where the L1 cache and shared memory use the same hardware resources, 
     * this sets the shared memory carveout preference, in percent of the total resources. 
     * This is only a hint, and the driver can choose a different ratio if required to execute the function.
     */
    LW_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,

    LW_FUNC_ATTRIBUTE_MAX
} LWfunction_attribute;

/**
 * Function cache configurations
 */
typedef enum LWfunc_cache_enum {
    LW_FUNC_CACHE_PREFER_NONE    = 0x00, /**< no preference for shared memory or L1 (default) */
    LW_FUNC_CACHE_PREFER_SHARED  = 0x01, /**< prefer larger shared memory and smaller L1 cache */
    LW_FUNC_CACHE_PREFER_L1      = 0x02, /**< prefer larger L1 cache and smaller shared memory */
    LW_FUNC_CACHE_PREFER_EQUAL   = 0x03  /**< prefer equal sized L1 cache and shared memory */
} LWfunc_cache;

/**
 * Shared memory configurations
 */
typedef enum LWsharedconfig_enum {
    LW_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE    = 0x00, /**< set default shared memory bank size */
    LW_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE  = 0x01, /**< set shared memory bank width to four bytes */
    LW_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 0x02  /**< set shared memory bank width to eight bytes */
} LWsharedconfig;

/**
 * Shared memory carveout configurations
 */
typedef enum LWshared_carveout_enum {
    LW_SHAREDMEM_CARVEOUT_DEFAULT       = -1,  /** < no preference for shared memory or L1 (default) */
    LW_SHAREDMEM_CARVEOUT_MAX_SHARED    = 100, /** < prefer maximum available shared memory, minimum L1 cache */
    LW_SHAREDMEM_CARVEOUT_MAX_L1        = 0    /** < prefer maximum available L1 cache, minimum shared memory */
} LWshared_carveout;

/**
 * Memory types
 */
typedef enum LWmemorytype_enum {
    LW_MEMORYTYPE_HOST    = 0x01,    /**< Host memory */
    LW_MEMORYTYPE_DEVICE  = 0x02,    /**< Device memory */
    LW_MEMORYTYPE_ARRAY   = 0x03,    /**< Array memory */
    LW_MEMORYTYPE_UNIFIED = 0x04     /**< Unified device or host memory */
} LWmemorytype;

/**
 * Compute Modes
 */
typedef enum LWcomputemode_enum {
    LW_COMPUTEMODE_DEFAULT           = 0, /**< Default compute mode (Multiple contexts allowed per device) */
    LW_COMPUTEMODE_PROHIBITED        = 2, /**< Compute-prohibited mode (No contexts can be created on this device at this time) */
    LW_COMPUTEMODE_EXCLUSIVE_PROCESS = 3  /**< Compute-exclusive-process mode (Only one context used by a single process can be present on this device at a time) */
} LWcomputemode;

/**
 * Memory advise values
 */
typedef enum LWmem_advise_enum {
    LW_MEM_ADVISE_SET_READ_MOSTLY          = 1, /**< Data will mostly be read and only occassionally be written to */
    LW_MEM_ADVISE_UNSET_READ_MOSTLY        = 2, /**< Undo the effect of ::LW_MEM_ADVISE_SET_READ_MOSTLY */
    LW_MEM_ADVISE_SET_PREFERRED_LOCATION   = 3, /**< Set the preferred location for the data as the specified device */
    LW_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4, /**< Clear the preferred location for the data */
    LW_MEM_ADVISE_SET_ACCESSED_BY          = 5, /**< Data will be accessed by the specified device, so prevent page faults as much as possible */
    LW_MEM_ADVISE_UNSET_ACCESSED_BY        = 6  /**< Let the Unified Memory subsystem decide on the page faulting policy for the specified device */
} LWmem_advise;

typedef enum LWmem_range_attribute_enum {
    LW_MEM_RANGE_ATTRIBUTE_READ_MOSTLY            = 1, /**< Whether the range will mostly be read and only occassionally be written to */
    LW_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION     = 2, /**< The preferred location of the range */
    LW_MEM_RANGE_ATTRIBUTE_ACCESSED_BY            = 3, /**< Memory range has ::LW_MEM_ADVISE_SET_ACCESSED_BY set for specified device */
    LW_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4  /**< The last location to which the range was prefetched */
} LWmem_range_attribute;

/**
 * Online compiler and linker options
 */
typedef enum LWjit_option_enum
{
    /**
     * Max number of registers that a thread may use.\n
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    LW_JIT_MAX_REGISTERS = 0,

    /**
     * IN: Specifies minimum number of threads per block to target compilation
     * for\n
     * OUT: Returns the number of threads the compiler actually targeted.
     * This restricts the resource utilization fo the compiler (e.g. max
     * registers) such that a block with the given number of threads should be
     * able to launch based on register limitations. Note, this option does not
     * lwrrently take into account any other resource limitations, such as
     * shared memory utilization.\n
     * Cannot be combined with ::LW_JIT_TARGET.\n
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    LW_JIT_THREADS_PER_BLOCK,

    /**
     * Overwrites the option value with the total wall clock time, in
     * milliseconds, spent in the compiler and linker\n
     * Option type: float\n
     * Applies to: compiler and linker
     */
    LW_JIT_WALL_TIME,

    /**
     * Pointer to a buffer in which to print any log messages
     * that are informational in nature (the buffer size is specified via
     * option ::LW_JIT_INFO_LOG_BUFFER_SIZE_BYTES)\n
     * Option type: char *\n
     * Applies to: compiler and linker
     */
    LW_JIT_INFO_LOG_BUFFER,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int\n
     * Applies to: compiler and linker
     */
    LW_JIT_INFO_LOG_BUFFER_SIZE_BYTES,

    /**
     * Pointer to a buffer in which to print any log messages that
     * reflect errors (the buffer size is specified via option
     * ::LW_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)\n
     * Option type: char *\n
     * Applies to: compiler and linker
     */
    LW_JIT_ERROR_LOG_BUFFER,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int\n
     * Applies to: compiler and linker
     */
    LW_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,

    /**
     * Level of optimizations to apply to generated code (0 - 4), with 4
     * being the default and highest level of optimizations.\n
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    LW_JIT_OPTIMIZATION_LEVEL,

    /**
     * No option value required. Determines the target based on the current
     * attached context (default)\n
     * Option type: No option value needed\n
     * Applies to: compiler and linker
     */
    LW_JIT_TARGET_FROM_LWCONTEXT,

    /**
     * Target is chosen based on supplied ::LWjit_target.  Cannot be
     * combined with ::LW_JIT_THREADS_PER_BLOCK.\n
     * Option type: unsigned int for enumerated type ::LWjit_target\n
     * Applies to: compiler and linker
     */
    LW_JIT_TARGET,

    /**
     * Specifies choice of fallback strategy if matching lwbin is not found.
     * Choice is based on supplied ::LWjit_fallback.  This option cannot be
     * used with lwLink* APIs as the linker requires exact matches.\n
     * Option type: unsigned int for enumerated type ::LWjit_fallback\n
     * Applies to: compiler only
     */
    LW_JIT_FALLBACK_STRATEGY,

    /**
     * Specifies whether to create debug information in output (-g)
     * (0: false, default)\n
     * Option type: int\n
     * Applies to: compiler and linker
     */
    LW_JIT_GENERATE_DEBUG_INFO,

    /**
     * Generate verbose log messages (0: false, default)\n
     * Option type: int\n
     * Applies to: compiler and linker
     */
    LW_JIT_LOG_VERBOSE,

    /**
     * Generate line number information (-lineinfo) (0: false, default)\n
     * Option type: int\n
     * Applies to: compiler only
     */
    LW_JIT_GENERATE_LINE_INFO,

    /**
     * Specifies whether to enable caching explicitly (-dlcm) \n
     * Choice is based on supplied ::LWjit_cacheMode_enum.\n
     * Option type: unsigned int for enumerated type ::LWjit_cacheMode_enum\n
     * Applies to: compiler only
     */
    LW_JIT_CACHE_MODE,

    /**
     * The below jit options are used for internal purposes only, in this version of LWCA
     */
    LW_JIT_NEW_SM3X_OPT,
    LW_JIT_FAST_COMPILE,

    LW_JIT_NUM_OPTIONS

} LWjit_option;

/**
 * Online compilation targets
 */
typedef enum LWjit_target_enum
{
    LW_TARGET_COMPUTE_20 = 20,       /**< Compute device class 2.0 */
    LW_TARGET_COMPUTE_21 = 21,       /**< Compute device class 2.1 */
    LW_TARGET_COMPUTE_30 = 30,       /**< Compute device class 3.0 */
    LW_TARGET_COMPUTE_32 = 32,       /**< Compute device class 3.2 */
    LW_TARGET_COMPUTE_35 = 35,       /**< Compute device class 3.5 */
    LW_TARGET_COMPUTE_37 = 37,       /**< Compute device class 3.7 */
    LW_TARGET_COMPUTE_50 = 50,       /**< Compute device class 5.0 */
    LW_TARGET_COMPUTE_52 = 52,       /**< Compute device class 5.2 */
    LW_TARGET_COMPUTE_53 = 53,       /**< Compute device class 5.3 */
    LW_TARGET_COMPUTE_60 = 60,       /**< Compute device class 6.0.*/
    LW_TARGET_COMPUTE_61 = 61,       /**< Compute device class 6.1.*/
    LW_TARGET_COMPUTE_62 = 62,       /**< Compute device class 6.2.*/
    LW_TARGET_COMPUTE_70 = 70        /**< Compute device class 7.0.*/
} LWjit_target;

/**
 * Lwbin matching fallback strategies
 */
typedef enum LWjit_fallback_enum
{
    LW_PREFER_PTX = 0,  /**< Prefer to compile ptx if exact binary match not found */

    LW_PREFER_BINARY    /**< Prefer to fall back to compatible binary code if exact match not found */

} LWjit_fallback;

/**
 * Caching modes for dlcm 
 */
typedef enum LWjit_cacheMode_enum
{
    LW_JIT_CACHE_OPTION_NONE = 0, /**< Compile with no -dlcm flag specified */
    LW_JIT_CACHE_OPTION_CG,       /**< Compile with L1 cache disabled */
    LW_JIT_CACHE_OPTION_CA        /**< Compile with L1 cache enabled */
} LWjit_cacheMode;

/**
 * Device code formats
 */
typedef enum LWjitInputType_enum
{
    /**
     * Compiled device-class-specific device code\n
     * Applicable options: none
     */
    LW_JIT_INPUT_LWBIN = 0,

    /**
     * PTX source code\n
     * Applicable options: PTX compiler options
     */
    LW_JIT_INPUT_PTX,

    /**
     * Bundle of multiple lwbins and/or PTX of some device code\n
     * Applicable options: PTX compiler options, ::LW_JIT_FALLBACK_STRATEGY
     */
    LW_JIT_INPUT_FATBINARY,

    /**
     * Host object with embedded device code\n
     * Applicable options: PTX compiler options, ::LW_JIT_FALLBACK_STRATEGY
     */
    LW_JIT_INPUT_OBJECT,

    /**
     * Archive of host objects with embedded device code\n
     * Applicable options: PTX compiler options, ::LW_JIT_FALLBACK_STRATEGY
     */
    LW_JIT_INPUT_LIBRARY,

    LW_JIT_NUM_INPUT_TYPES
} LWjitInputType;

#if __LWDA_API_VERSION >= 5050
typedef struct LWlinkState_st *LWlinkState;
#endif /* __LWDA_API_VERSION >= 5050 */

/**
 * Flags to register a graphics resource
 */
typedef enum LWgraphicsRegisterFlags_enum {
    LW_GRAPHICS_REGISTER_FLAGS_NONE           = 0x00,
    LW_GRAPHICS_REGISTER_FLAGS_READ_ONLY      = 0x01,
    LW_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD  = 0x02,
    LW_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST   = 0x04,
    LW_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 0x08
} LWgraphicsRegisterFlags;

/**
 * Flags for mapping and unmapping interop resources
 */
typedef enum LWgraphicsMapResourceFlags_enum {
    LW_GRAPHICS_MAP_RESOURCE_FLAGS_NONE          = 0x00,
    LW_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY     = 0x01,
    LW_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02
} LWgraphicsMapResourceFlags;

/**
 * Array indices for lwbe faces
 */
typedef enum LWarray_lwbemap_face_enum {
    LW_LWBEMAP_FACE_POSITIVE_X  = 0x00, /**< Positive X face of lwbemap */
    LW_LWBEMAP_FACE_NEGATIVE_X  = 0x01, /**< Negative X face of lwbemap */
    LW_LWBEMAP_FACE_POSITIVE_Y  = 0x02, /**< Positive Y face of lwbemap */
    LW_LWBEMAP_FACE_NEGATIVE_Y  = 0x03, /**< Negative Y face of lwbemap */
    LW_LWBEMAP_FACE_POSITIVE_Z  = 0x04, /**< Positive Z face of lwbemap */
    LW_LWBEMAP_FACE_NEGATIVE_Z  = 0x05  /**< Negative Z face of lwbemap */
} LWarray_lwbemap_face;

/**
 * Limits
 */
typedef enum LWlimit_enum {
    LW_LIMIT_STACK_SIZE                       = 0x00, /**< GPU thread stack size */
    LW_LIMIT_PRINTF_FIFO_SIZE                 = 0x01, /**< GPU printf FIFO size */
    LW_LIMIT_MALLOC_HEAP_SIZE                 = 0x02, /**< GPU malloc heap size */
    LW_LIMIT_DEV_RUNTIME_SYNC_DEPTH           = 0x03, /**< GPU device runtime launch synchronize depth */
    LW_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04, /**< GPU device runtime pending launch count */
    LW_LIMIT_MAX
} LWlimit;

/**
 * Resource types
 */
typedef enum LWresourcetype_enum {
    LW_RESOURCE_TYPE_ARRAY           = 0x00, /**< Array resoure */
    LW_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01, /**< Mipmapped array resource */
    LW_RESOURCE_TYPE_LINEAR          = 0x02, /**< Linear resource */
    LW_RESOURCE_TYPE_PITCH2D         = 0x03  /**< Pitch 2D resource */
} LWresourcetype;

/**
 * Error codes
 */
typedef enum lwdaError_enum {
    /**
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * ::lwEventQuery() and ::lwStreamQuery()).
     */
    LWDA_SUCCESS                              = 0,

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    LWDA_ERROR_ILWALID_VALUE                  = 1,

    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    LWDA_ERROR_OUT_OF_MEMORY                  = 2,

    /**
     * This indicates that the LWCA driver has not been initialized with
     * ::lwInit() or that initialization has failed.
     */
    LWDA_ERROR_NOT_INITIALIZED                = 3,

    /**
     * This indicates that the LWCA driver is in the process of shutting down.
     */
    LWDA_ERROR_DEINITIALIZED                  = 4,

    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    LWDA_ERROR_PROFILER_DISABLED              = 5,

    /**
     * \deprecated
     * This error return is deprecated as of LWCA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::lwProfilerStart or
     * ::lwProfilerStop without initialization.
     */
    LWDA_ERROR_PROFILER_NOT_INITIALIZED       = 6,

    /**
     * \deprecated
     * This error return is deprecated as of LWCA 5.0. It is no longer an error
     * to call lwProfilerStart() when profiling is already enabled.
     */
    LWDA_ERROR_PROFILER_ALREADY_STARTED       = 7,

    /**
     * \deprecated
     * This error return is deprecated as of LWCA 5.0. It is no longer an error
     * to call lwProfilerStop() when profiling is already disabled.
     */
    LWDA_ERROR_PROFILER_ALREADY_STOPPED       = 8,

    /**
     * This indicates that no LWCA-capable devices were detected by the installed
     * LWCA driver.
     */
    LWDA_ERROR_NO_DEVICE                      = 100,

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid LWCA device.
     */
    LWDA_ERROR_ILWALID_DEVICE                 = 101,


    /**
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid LWCA module.
     */
    LWDA_ERROR_ILWALID_IMAGE                  = 200,

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::lwCtxDestroy() ilwoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::lwCtxGetApiVersion() for more details.
     */
    LWDA_ERROR_ILWALID_CONTEXT                = 201,

    /**
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of LWCA 3.2. It is no longer an
     * error to attempt to push the active context via ::lwCtxPushLwrrent().
     */
    LWDA_ERROR_CONTEXT_ALREADY_LWRRENT        = 202,

    /**
     * This indicates that a map or register operation has failed.
     */
    LWDA_ERROR_MAP_FAILED                     = 205,

    /**
     * This indicates that an unmap or unregister operation has failed.
     */
    LWDA_ERROR_UNMAP_FAILED                   = 206,

    /**
     * This indicates that the specified array is lwrrently mapped and thus
     * cannot be destroyed.
     */
    LWDA_ERROR_ARRAY_IS_MAPPED                = 207,

    /**
     * This indicates that the resource is already mapped.
     */
    LWDA_ERROR_ALREADY_MAPPED                 = 208,

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular LWCA source file that do not include the
     * corresponding device configuration.
     */
    LWDA_ERROR_NO_BINARY_FOR_GPU              = 209,

    /**
     * This indicates that a resource has already been acquired.
     */
    LWDA_ERROR_ALREADY_ACQUIRED               = 210,

    /**
     * This indicates that a resource is not mapped.
     */
    LWDA_ERROR_NOT_MAPPED                     = 211,

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    LWDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212,

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    LWDA_ERROR_NOT_MAPPED_AS_POINTER          = 213,

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * exelwtion.
     */
    LWDA_ERROR_ECC_UNCORRECTABLE              = 214,

    /**
     * This indicates that the ::LWlimit passed to the API call is not
     * supported by the active device.
     */
    LWDA_ERROR_UNSUPPORTED_LIMIT              = 215,

    /**
     * This indicates that the ::LWcontext passed to the API call can
     * only be bound to a single CPU thread at a time but is already 
     * bound to a CPU thread.
     */
    LWDA_ERROR_CONTEXT_ALREADY_IN_USE         = 216,

    /**
     * This indicates that peer access is not supported across the given
     * devices.
     */
    LWDA_ERROR_PEER_ACCESS_UNSUPPORTED        = 217,

    /**
     * This indicates that a PTX JIT compilation failed.
     */
    LWDA_ERROR_ILWALID_PTX                    = 218,

    /**
     * This indicates an error with OpenGL or DirectX context.
     */
    LWDA_ERROR_ILWALID_GRAPHICS_CONTEXT       = 219,

    /**
    * This indicates that an uncorrectable LWLink error was detected during the
    * exelwtion.
    */
    LWDA_ERROR_LWLINK_UNCORRECTABLE           = 220,

    /**
    * This indicates that the PTX JIT compiler library was not found.
    */
    LWDA_ERROR_JIT_COMPILER_NOT_FOUND         = 221,

    /**
     * This indicates that the device kernel source is invalid.
     */
    LWDA_ERROR_ILWALID_SOURCE                 = 300,

    /**
     * This indicates that the file specified was not found.
     */
    LWDA_ERROR_FILE_NOT_FOUND                 = 301,

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    LWDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,

    /**
     * This indicates that initialization of a shared object failed.
     */
    LWDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303,

    /**
     * This indicates that an OS call failed.
     */
    LWDA_ERROR_OPERATING_SYSTEM               = 304,

    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::LWstream and ::LWevent.
     */
    LWDA_ERROR_ILWALID_HANDLE                 = 400,

    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, texture names, and surface names.
     */
    LWDA_ERROR_NOT_FOUND                      = 500,

    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::LWDA_SUCCESS (which indicates completion). Calls that
     * may return this value include ::lwEventQuery() and ::lwStreamQuery().
     */
    LWDA_ERROR_NOT_READY                      = 600,

    /**
     * While exelwting a kernel, the device encountered a
     * load or store instruction on an invalid memory address.
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    LWDA_ERROR_ILLEGAL_ADDRESS                = 700,

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    LWDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701,

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::LW_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information.
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    LWDA_ERROR_LAUNCH_TIMEOUT                 = 702,

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    LWDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,
    
    /**
     * This error indicates that a call to ::lwCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     */
    LWDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    = 704,

    /**
     * This error indicates that ::lwCtxDisablePeerAccess() is 
     * trying to disable peer access which has not been enabled yet 
     * via ::lwCtxEnablePeerAccess(). 
     */
    LWDA_ERROR_PEER_ACCESS_NOT_ENABLED        = 705,

    /**
     * This error indicates that the primary context for the specified device
     * has already been initialized.
     */
    LWDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708,

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::lwCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    LWDA_ERROR_CONTEXT_IS_DESTROYED           = 709,

    /**
     * A device-side assert triggered during kernel exelwtion. The context
     * cannot be used anymore, and must be destroyed. All existing device 
     * memory allocations from this context are invalid and must be 
     * reconstructed if the program is to continue using LWCA.
     */
    LWDA_ERROR_ASSERT                         = 710,

    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices 
     * passed to ::lwCtxEnablePeerAccess().
     */
    LWDA_ERROR_TOO_MANY_PEERS                 = 711,

    /**
     * This error indicates that the memory range passed to ::lwMemHostRegister()
     * has already been registered.
     */
    LWDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,

    /**
     * This error indicates that the pointer passed to ::lwMemHostUnregister()
     * does not correspond to any lwrrently registered memory region.
     */
    LWDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713,

    /**
     * While exelwting a kernel, the device encountered a stack error.
     * This can be due to stack corruption or exceeding the stack size limit.
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    LWDA_ERROR_HARDWARE_STACK_ERROR           = 714,

    /**
     * While exelwting a kernel, the device encountered an illegal instruction.
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    LWDA_ERROR_ILLEGAL_INSTRUCTION            = 715,

    /**
     * While exelwting a kernel, the device encountered a load or store instruction
     * on a memory address which is not aligned.
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    LWDA_ERROR_MISALIGNED_ADDRESS             = 716,

    /**
     * While exelwting a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    LWDA_ERROR_ILWALID_ADDRESS_SPACE          = 717,

    /**
     * While exelwting a kernel, the device program counter wrapped its address space.
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    LWDA_ERROR_ILWALID_PC                     = 718,

    /**
     * An exception oclwrred on the device while exelwting a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory.
     * This leaves the process in an inconsistent state and any further LWCA work
     * will return the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    LWDA_ERROR_LAUNCH_FAILED                  = 719,

    /**
     * This error indicates that the number of blocks launched per grid for a kernel that was
     * launched via either ::lwLaunchCooperativeKernel or ::lwLaunchCooperativeKernelMultiDevice
     * exceeds the maximum number of blocks as allowed by ::lwOclwpancyMaxActiveBlocksPerMultiprocessor
     * or ::lwOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
     * as specified by the device attribute ::LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
     */
    LWDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE   = 720,

    /**
     * This error indicates that the attempted operation is not permitted.
     */
    LWDA_ERROR_NOT_PERMITTED                  = 800,

    /**
     * This error indicates that the attempted operation is not supported
     * on the current system or device.
     */
    LWDA_ERROR_NOT_SUPPORTED                  = 801,

    /**
     * This indicates that an unknown internal error has oclwrred.
     */
    LWDA_ERROR_UNKNOWN                        = 999
} LWresult;

/**
 * P2P Attributes
 */
typedef enum LWdevice_P2PAttribute_enum {
    LW_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK        = 0x01, /**< A relative value indicating the performance of the link between two devices */
    LW_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED        = 0x02, /**< P2P Access is enable */
    LW_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 0x03  /**< Atomic operation over the link supported */
} LWdevice_P2PAttribute;

#ifdef _WIN32
#define LWDA_CB __stdcall
#else
#define LWDA_CB
#endif

/**
 * LWCA stream callback
 * \param hStream The stream the callback was added to, as passed to ::lwStreamAddCallback.  May be NULL.
 * \param status ::LWDA_SUCCESS or any persistent error on the stream.
 * \param userData User parameter provided at registration.
 */
typedef void (LWDA_CB *LWstreamCallback)(LWstream hStream, LWresult status, void *userData);

/**
 * Block size to per-block dynamic shared memory mapping for a certain
 * kernel \param blockSize Block size of the kernel.
 *
 * \return The dynamic shared memory needed by a block.
 */
typedef size_t (LWDA_CB *LWoclwpancyB2DSize)(int blockSize);

/**
 * If set, host memory is portable between LWCA contexts.
 * Flag for ::lwMemHostAlloc()
 */
#define LW_MEMHOSTALLOC_PORTABLE        0x01

/**
 * If set, host memory is mapped into LWCA address space and
 * ::lwMemHostGetDevicePointer() may be called on the host pointer.
 * Flag for ::lwMemHostAlloc()
 */
#define LW_MEMHOSTALLOC_DEVICEMAP       0x02

/**
 * If set, host memory is allocated as write-combined - fast to write,
 * faster to DMA, slow to read except via SSE4 streaming load instruction
 * (MOVNTDQA).
 * Flag for ::lwMemHostAlloc()
 */
#define LW_MEMHOSTALLOC_WRITECOMBINED   0x04

/**
 * If set, host memory is portable between LWCA contexts.
 * Flag for ::lwMemHostRegister()
 */
#define LW_MEMHOSTREGISTER_PORTABLE     0x01

/**
 * If set, host memory is mapped into LWCA address space and
 * ::lwMemHostGetDevicePointer() may be called on the host pointer.
 * Flag for ::lwMemHostRegister()
 */
#define LW_MEMHOSTREGISTER_DEVICEMAP    0x02

/**
 * If set, the passed memory pointer is treated as pointing to some
 * memory-mapped I/O space, e.g. belonging to a third-party PCIe device.
 * On Windows the flag is a no-op.
 * On Linux that memory is marked as non cache-coherent for the GPU and
 * is expected to be physically contiguous. It may return
 * LWDA_ERROR_NOT_PERMITTED if run as an unprivileged user,
 * LWDA_ERROR_NOT_SUPPORTED on older Linux kernel versions.
 * On all other platforms, it is not supported and LWDA_ERROR_NOT_SUPPORTED
 * is returned.
 * Flag for ::lwMemHostRegister()
 */
#define LW_MEMHOSTREGISTER_IOMEMORY     0x04

#if __LWDA_API_VERSION >= 3020

/**
 * 2D memory copy parameters
 */
typedef struct LWDA_MEMCPY2D_st {
    size_t srcXInBytes;         /**< Source X in bytes */
    size_t srcY;                /**< Source Y */

    LWmemorytype srcMemoryType; /**< Source memory type (host, device, array) */
    const void *srcHost;        /**< Source host pointer */
    LWdeviceptr srcDevice;      /**< Source device pointer */
    LWarray srcArray;           /**< Source array reference */
    size_t srcPitch;            /**< Source pitch (ignored when src is array) */

    size_t dstXInBytes;         /**< Destination X in bytes */
    size_t dstY;                /**< Destination Y */

    LWmemorytype dstMemoryType; /**< Destination memory type (host, device, array) */
    void *dstHost;              /**< Destination host pointer */
    LWdeviceptr dstDevice;      /**< Destination device pointer */
    LWarray dstArray;           /**< Destination array reference */
    size_t dstPitch;            /**< Destination pitch (ignored when dst is array) */

    size_t WidthInBytes;        /**< Width of 2D memory copy in bytes */
    size_t Height;              /**< Height of 2D memory copy */
} LWDA_MEMCPY2D;

/**
 * 3D memory copy parameters
 */
typedef struct LWDA_MEMCPY3D_st {
    size_t srcXInBytes;         /**< Source X in bytes */
    size_t srcY;                /**< Source Y */
    size_t srcZ;                /**< Source Z */
    size_t srcLOD;              /**< Source LOD */
    LWmemorytype srcMemoryType; /**< Source memory type (host, device, array) */
    const void *srcHost;        /**< Source host pointer */
    LWdeviceptr srcDevice;      /**< Source device pointer */
    LWarray srcArray;           /**< Source array reference */
    void *reserved0;            /**< Must be NULL */
    size_t srcPitch;            /**< Source pitch (ignored when src is array) */
    size_t srcHeight;           /**< Source height (ignored when src is array; may be 0 if Depth==1) */

    size_t dstXInBytes;         /**< Destination X in bytes */
    size_t dstY;                /**< Destination Y */
    size_t dstZ;                /**< Destination Z */
    size_t dstLOD;              /**< Destination LOD */
    LWmemorytype dstMemoryType; /**< Destination memory type (host, device, array) */
    void *dstHost;              /**< Destination host pointer */
    LWdeviceptr dstDevice;      /**< Destination device pointer */
    LWarray dstArray;           /**< Destination array reference */
    void *reserved1;            /**< Must be NULL */
    size_t dstPitch;            /**< Destination pitch (ignored when dst is array) */
    size_t dstHeight;           /**< Destination height (ignored when dst is array; may be 0 if Depth==1) */

    size_t WidthInBytes;        /**< Width of 3D memory copy in bytes */
    size_t Height;              /**< Height of 3D memory copy */
    size_t Depth;               /**< Depth of 3D memory copy */
} LWDA_MEMCPY3D;

/**
 * 3D memory cross-context copy parameters
 */
typedef struct LWDA_MEMCPY3D_PEER_st {
    size_t srcXInBytes;         /**< Source X in bytes */
    size_t srcY;                /**< Source Y */
    size_t srcZ;                /**< Source Z */
    size_t srcLOD;              /**< Source LOD */
    LWmemorytype srcMemoryType; /**< Source memory type (host, device, array) */
    const void *srcHost;        /**< Source host pointer */
    LWdeviceptr srcDevice;      /**< Source device pointer */
    LWarray srcArray;           /**< Source array reference */
    LWcontext srcContext;       /**< Source context (ignored with srcMemoryType is ::LW_MEMORYTYPE_ARRAY) */
    size_t srcPitch;            /**< Source pitch (ignored when src is array) */
    size_t srcHeight;           /**< Source height (ignored when src is array; may be 0 if Depth==1) */

    size_t dstXInBytes;         /**< Destination X in bytes */
    size_t dstY;                /**< Destination Y */
    size_t dstZ;                /**< Destination Z */
    size_t dstLOD;              /**< Destination LOD */
    LWmemorytype dstMemoryType; /**< Destination memory type (host, device, array) */
    void *dstHost;              /**< Destination host pointer */
    LWdeviceptr dstDevice;      /**< Destination device pointer */
    LWarray dstArray;           /**< Destination array reference */
    LWcontext dstContext;       /**< Destination context (ignored with dstMemoryType is ::LW_MEMORYTYPE_ARRAY) */
    size_t dstPitch;            /**< Destination pitch (ignored when dst is array) */
    size_t dstHeight;           /**< Destination height (ignored when dst is array; may be 0 if Depth==1) */

    size_t WidthInBytes;        /**< Width of 3D memory copy in bytes */
    size_t Height;              /**< Height of 3D memory copy */
    size_t Depth;               /**< Depth of 3D memory copy */
} LWDA_MEMCPY3D_PEER;

/**
 * Array descriptor
 */
typedef struct LWDA_ARRAY_DESCRIPTOR_st
{
    size_t Width;             /**< Width of array */
    size_t Height;            /**< Height of array */

    LWarray_format Format;    /**< Array format */
    unsigned int NumChannels; /**< Channels per array element */
} LWDA_ARRAY_DESCRIPTOR;

/**
 * 3D array descriptor
 */
typedef struct LWDA_ARRAY3D_DESCRIPTOR_st
{
    size_t Width;             /**< Width of 3D array */
    size_t Height;            /**< Height of 3D array */
    size_t Depth;             /**< Depth of 3D array */

    LWarray_format Format;    /**< Array format */
    unsigned int NumChannels; /**< Channels per array element */
    unsigned int Flags;       /**< Flags */
} LWDA_ARRAY3D_DESCRIPTOR;

#endif /* __LWDA_API_VERSION >= 3020 */

#if __LWDA_API_VERSION >= 5000

/**
 * LWCA Resource descriptor
 */
typedef struct LWDA_RESOURCE_DESC_st
{
    LWresourcetype resType;                   /**< Resource type */

    union {
        struct {
            LWarray hArray;                   /**< LWCA array */
        } array;
        struct {
            LWmipmappedArray hMipmappedArray; /**< LWCA mipmapped array */
        } mipmap;
        struct {
            LWdeviceptr devPtr;               /**< Device pointer */
            LWarray_format format;            /**< Array format */
            unsigned int numChannels;         /**< Channels per array element */
            size_t sizeInBytes;               /**< Size in bytes */
        } linear;
        struct {
            LWdeviceptr devPtr;               /**< Device pointer */
            LWarray_format format;            /**< Array format */
            unsigned int numChannels;         /**< Channels per array element */
            size_t width;                     /**< Width of the array in elements */
            size_t height;                    /**< Height of the array in elements */
            size_t pitchInBytes;              /**< Pitch between two rows in bytes */
        } pitch2D;
        struct {
            int reserved[32];
        } reserved;
    } res;

    unsigned int flags;                       /**< Flags (must be zero) */
} LWDA_RESOURCE_DESC;

/**
 * Texture descriptor
 */
typedef struct LWDA_TEXTURE_DESC_st {
    LWaddress_mode addressMode[3];  /**< Address modes */
    LWfilter_mode filterMode;       /**< Filter mode */
    unsigned int flags;             /**< Flags */
    unsigned int maxAnisotropy;     /**< Maximum anisotropy ratio */
    LWfilter_mode mipmapFilterMode; /**< Mipmap filter mode */
    float mipmapLevelBias;          /**< Mipmap level bias */
    float minMipmapLevelClamp;      /**< Mipmap minimum level clamp */
    float maxMipmapLevelClamp;      /**< Mipmap maximum level clamp */ 
    float borderColor[4];           /**< Border Color */
    int reserved[12];
} LWDA_TEXTURE_DESC;

/**
 * Resource view format
 */
typedef enum LWresourceViewFormat_enum
{
    LW_RES_VIEW_FORMAT_NONE          = 0x00, /**< No resource view format (use underlying resource format) */
    LW_RES_VIEW_FORMAT_UINT_1X8      = 0x01, /**< 1 channel unsigned 8-bit integers */
    LW_RES_VIEW_FORMAT_UINT_2X8      = 0x02, /**< 2 channel unsigned 8-bit integers */
    LW_RES_VIEW_FORMAT_UINT_4X8      = 0x03, /**< 4 channel unsigned 8-bit integers */
    LW_RES_VIEW_FORMAT_SINT_1X8      = 0x04, /**< 1 channel signed 8-bit integers */
    LW_RES_VIEW_FORMAT_SINT_2X8      = 0x05, /**< 2 channel signed 8-bit integers */
    LW_RES_VIEW_FORMAT_SINT_4X8      = 0x06, /**< 4 channel signed 8-bit integers */
    LW_RES_VIEW_FORMAT_UINT_1X16     = 0x07, /**< 1 channel unsigned 16-bit integers */
    LW_RES_VIEW_FORMAT_UINT_2X16     = 0x08, /**< 2 channel unsigned 16-bit integers */
    LW_RES_VIEW_FORMAT_UINT_4X16     = 0x09, /**< 4 channel unsigned 16-bit integers */
    LW_RES_VIEW_FORMAT_SINT_1X16     = 0x0a, /**< 1 channel signed 16-bit integers */
    LW_RES_VIEW_FORMAT_SINT_2X16     = 0x0b, /**< 2 channel signed 16-bit integers */
    LW_RES_VIEW_FORMAT_SINT_4X16     = 0x0c, /**< 4 channel signed 16-bit integers */
    LW_RES_VIEW_FORMAT_UINT_1X32     = 0x0d, /**< 1 channel unsigned 32-bit integers */
    LW_RES_VIEW_FORMAT_UINT_2X32     = 0x0e, /**< 2 channel unsigned 32-bit integers */
    LW_RES_VIEW_FORMAT_UINT_4X32     = 0x0f, /**< 4 channel unsigned 32-bit integers */
    LW_RES_VIEW_FORMAT_SINT_1X32     = 0x10, /**< 1 channel signed 32-bit integers */
    LW_RES_VIEW_FORMAT_SINT_2X32     = 0x11, /**< 2 channel signed 32-bit integers */
    LW_RES_VIEW_FORMAT_SINT_4X32     = 0x12, /**< 4 channel signed 32-bit integers */
    LW_RES_VIEW_FORMAT_FLOAT_1X16    = 0x13, /**< 1 channel 16-bit floating point */
    LW_RES_VIEW_FORMAT_FLOAT_2X16    = 0x14, /**< 2 channel 16-bit floating point */
    LW_RES_VIEW_FORMAT_FLOAT_4X16    = 0x15, /**< 4 channel 16-bit floating point */
    LW_RES_VIEW_FORMAT_FLOAT_1X32    = 0x16, /**< 1 channel 32-bit floating point */
    LW_RES_VIEW_FORMAT_FLOAT_2X32    = 0x17, /**< 2 channel 32-bit floating point */
    LW_RES_VIEW_FORMAT_FLOAT_4X32    = 0x18, /**< 4 channel 32-bit floating point */
    LW_RES_VIEW_FORMAT_UNSIGNED_BC1  = 0x19, /**< Block compressed 1 */
    LW_RES_VIEW_FORMAT_UNSIGNED_BC2  = 0x1a, /**< Block compressed 2 */
    LW_RES_VIEW_FORMAT_UNSIGNED_BC3  = 0x1b, /**< Block compressed 3 */
    LW_RES_VIEW_FORMAT_UNSIGNED_BC4  = 0x1c, /**< Block compressed 4 unsigned */
    LW_RES_VIEW_FORMAT_SIGNED_BC4    = 0x1d, /**< Block compressed 4 signed */
    LW_RES_VIEW_FORMAT_UNSIGNED_BC5  = 0x1e, /**< Block compressed 5 unsigned */
    LW_RES_VIEW_FORMAT_SIGNED_BC5    = 0x1f, /**< Block compressed 5 signed */
    LW_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x20, /**< Block compressed 6 unsigned half-float */
    LW_RES_VIEW_FORMAT_SIGNED_BC6H   = 0x21, /**< Block compressed 6 signed half-float */
    LW_RES_VIEW_FORMAT_UNSIGNED_BC7  = 0x22  /**< Block compressed 7 */
} LWresourceViewFormat;

/**
 * Resource view descriptor
 */
typedef struct LWDA_RESOURCE_VIEW_DESC_st
{
    LWresourceViewFormat format;   /**< Resource view format */
    size_t width;                  /**< Width of the resource view */
    size_t height;                 /**< Height of the resource view */
    size_t depth;                  /**< Depth of the resource view */
    unsigned int firstMipmapLevel; /**< First defined mipmap level */
    unsigned int lastMipmapLevel;  /**< Last defined mipmap level */
    unsigned int firstLayer;       /**< First layer index */
    unsigned int lastLayer;        /**< Last layer index */
    unsigned int reserved[16];
} LWDA_RESOURCE_VIEW_DESC;

/**
 * GPU Direct v3 tokens
 */
typedef struct LWDA_POINTER_ATTRIBUTE_P2P_TOKENS_st {
    unsigned long long p2pToken;
    unsigned int vaSpaceToken;
} LWDA_POINTER_ATTRIBUTE_P2P_TOKENS;

#endif /* __LWDA_API_VERSION >= 5000 */

#if __LWDA_API_VERSION >= 9000

/**
 * Kernel launch parameters
 */
typedef struct LWDA_LAUNCH_PARAMS_st {
    LWfunction function;         /**< Kernel to launch */
    unsigned int gridDimX;       /**< Width of grid in blocks */
    unsigned int gridDimY;       /**< Height of grid in blocks */
    unsigned int gridDimZ;       /**< Depth of grid in blocks */
    unsigned int blockDimX;      /**< X dimension of each thread block */
    unsigned int blockDimY;      /**< Y dimension of each thread block */
    unsigned int blockDimZ;      /**< Z dimension of each thread block */
    unsigned int sharedMemBytes; /**< Dynamic shared-memory size per thread block in bytes */
    LWstream hStream;            /**< Stream identifier */
    void **kernelParams;         /**< Array of pointers to kernel parameters */
} LWDA_LAUNCH_PARAMS;

#endif /* __LWDA_API_VERSION >= 9000 */

/**
 * If set, each kernel launched as part of ::lwLaunchCooperativeKernelMultiDevice only
 * waits for prior work in the stream corresponding to that GPU to complete before the
 * kernel begins exelwtion.
 */
#define LWDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC   0x01

/**
 * If set, any subsequent work pushed in a stream that participated in a call to
 * ::lwLaunchCooperativeKernelMultiDevice will only wait for the kernel launched on
 * the GPU corresponding to that stream to complete before it begins exelwtion.
 */
#define LWDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC  0x02

/**
 * If set, the LWCA array is a collection of layers, where each layer is either a 1D
 * or a 2D array and the Depth member of LWDA_ARRAY3D_DESCRIPTOR specifies the number 
 * of layers, not the depth of a 3D array.
 */
#define LWDA_ARRAY3D_LAYERED        0x01

/**
 * Deprecated, use LWDA_ARRAY3D_LAYERED
 */
#define LWDA_ARRAY3D_2DARRAY        0x01

/**
 * This flag must be set in order to bind a surface reference
 * to the LWCA array
 */
#define LWDA_ARRAY3D_SURFACE_LDST   0x02

/**
 * If set, the LWCA array is a collection of six 2D arrays, representing faces of a lwbe. The
 * width of such a LWCA array must be equal to its height, and Depth must be six.
 * If ::LWDA_ARRAY3D_LAYERED flag is also set, then the LWCA array is a collection of lwbemaps
 * and Depth must be a multiple of six.
 */
#define LWDA_ARRAY3D_LWBEMAP        0x04

/**
 * This flag must be set in order to perform texture gather operations
 * on a LWCA array.
 */
#define LWDA_ARRAY3D_TEXTURE_GATHER 0x08

/**
 * This flag if set indicates that the LWCA
 * array is a DEPTH_TEXTURE.
*/
#define LWDA_ARRAY3D_DEPTH_TEXTURE 0x10

/**
 * Override the texref format with a format inferred from the array.
 * Flag for ::lwTexRefSetArray()
 */
#define LW_TRSA_OVERRIDE_FORMAT 0x01

/**
 * Read the texture as integers rather than promoting the values to floats
 * in the range [0,1].
 * Flag for ::lwTexRefSetFlags()
 */
#define LW_TRSF_READ_AS_INTEGER         0x01

/**
 * Use normalized texture coordinates in the range [0,1) instead of [0,dim).
 * Flag for ::lwTexRefSetFlags()
 */
#define LW_TRSF_NORMALIZED_COORDINATES  0x02

/**
 * Perform sRGB->linear colwersion during texture read.
 * Flag for ::lwTexRefSetFlags()
 */
#define LW_TRSF_SRGB  0x10

/**
 * End of array terminator for the \p extra parameter to
 * ::lwLaunchKernel
 */
#define LW_LAUNCH_PARAM_END            ((void*)0x00)

/**
 * Indicator that the next value in the \p extra parameter to
 * ::lwLaunchKernel will be a pointer to a buffer containing all kernel
 * parameters used for launching kernel \p f.  This buffer needs to
 * honor all alignment/padding requirements of the individual parameters.
 * If ::LW_LAUNCH_PARAM_BUFFER_SIZE is not also specified in the
 * \p extra array, then ::LW_LAUNCH_PARAM_BUFFER_POINTER will have no
 * effect.
 */
#define LW_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)

/**
 * Indicator that the next value in the \p extra parameter to
 * ::lwLaunchKernel will be a pointer to a size_t which contains the
 * size of the buffer specified with ::LW_LAUNCH_PARAM_BUFFER_POINTER.
 * It is required that ::LW_LAUNCH_PARAM_BUFFER_POINTER also be specified
 * in the \p extra array if the value associated with
 * ::LW_LAUNCH_PARAM_BUFFER_SIZE is not zero.
 */
#define LW_LAUNCH_PARAM_BUFFER_SIZE    ((void*)0x02)

/**
 * For texture references loaded into the module, use default texunit from
 * texture reference.
 */
#define LW_PARAM_TR_DEFAULT -1

/**
 * Device that represents the CPU
 */
#define LW_DEVICE_CPU               ((LWdevice)-1)

/**
 * Device that represents an invalid device
 */
#define LW_DEVICE_ILWALID           ((LWdevice)-2)

/** @} */ /* END LWDA_TYPES */

#ifdef _WIN32
#define LWDAAPI __stdcall
#else
#define LWDAAPI
#endif

/**
 * \defgroup LWDA_ERROR Error Handling
 *
 * ___MANBRIEF___ error handling functions of the low-level LWCA driver API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the error handling functions of the low-level LWCA
 * driver application programming interface.
 *
 * @{
 */

/**
 * \brief Gets the string description of an error code
 *
 * Sets \p *pStr to the address of a NULL-terminated string description
 * of the error code \p error.
 * If the error code is not recognized, ::LWDA_ERROR_ILWALID_VALUE
 * will be returned and \p *pStr will be set to the NULL address.
 *
 * \param error - Error code to colwert to string
 * \param pStr - Address of the string pointer.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa
 * ::LWresult,
 * ::lwdaGetErrorString
 */
LWresult LWDAAPI lwGetErrorString(LWresult error, const char **pStr);

/**
 * \brief Gets the string representation of an error code enum name
 *
 * Sets \p *pStr to the address of a NULL-terminated string representation
 * of the name of the enum error code \p error.
 * If the error code is not recognized, ::LWDA_ERROR_ILWALID_VALUE
 * will be returned and \p *pStr will be set to the NULL address.
 *
 * \param error - Error code to colwert to string
 * \param pStr - Address of the string pointer.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa
 * ::LWresult,
 * ::lwdaGetErrorName
 */
LWresult LWDAAPI lwGetErrorName(LWresult error, const char **pStr);

/** @} */ /* END LWDA_ERROR */

/**
 * \defgroup LWDA_INITIALIZE Initialization
 *
 * ___MANBRIEF___ initialization functions of the low-level LWCA driver API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the initialization functions of the low-level LWCA
 * driver application programming interface.
 *
 * @{
 */

/**
 * \brief Initialize the LWCA driver API
 *
 * Initializes the driver API and must be called before any other function from
 * the driver API. Lwrrently, the \p Flags parameter must be 0. If ::lwInit()
 * has not been called, any function from the driver API will return
 * ::LWDA_ERROR_NOT_INITIALIZED.
 *
 * \param Flags - Initialization flag for LWCA.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 */
LWresult LWDAAPI lwInit(unsigned int Flags);

/** @} */ /* END LWDA_INITIALIZE */

/**
 * \defgroup LWDA_VERSION Version Management
 *
 * ___MANBRIEF___ version management functions of the low-level LWCA driver
 * API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the version management functions of the low-level
 * LWCA driver application programming interface.
 *
 * @{
 */

/**
 * \brief Returns the LWCA driver version
 *
 * Returns in \p *driverVersion the version number of the installed LWCA
 * driver. This function automatically returns ::LWDA_ERROR_ILWALID_VALUE if
 * the \p driverVersion argument is NULL.
 *
 * \param driverVersion - Returns the LWCA driver version
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa
 * ::lwdaDriverGetVersion,
 * ::lwdaRuntimeGetVersion
 */
LWresult LWDAAPI lwDriverGetVersion(int *driverVersion);

/** @} */ /* END LWDA_VERSION */

/**
 * \defgroup LWDA_DEVICE Device Management
 *
 * ___MANBRIEF___ device management functions of the low-level LWCA driver API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the device management functions of the low-level
 * LWCA driver application programming interface.
 *
 * @{
 */

/**
 * \brief Returns a handle to a compute device
 *
 * Returns in \p *device a device handle given an ordinal in the range <b>[0,
 * ::lwDeviceGetCount()-1]</b>.
 *
 * \param device  - Returned device handle
 * \param ordinal - Device number to get handle for
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::lwDeviceGetAttribute,
 * ::lwDeviceGetCount,
 * ::lwDeviceGetName,
 * ::lwDeviceTotalMem
 */
LWresult LWDAAPI lwDeviceGet(LWdevice *device, int ordinal);

/**
 * \brief Returns the number of compute-capable devices
 *
 * Returns in \p *count the number of devices with compute capability greater
 * than or equal to 2.0 that are available for exelwtion. If there is no such
 * device, ::lwDeviceGetCount() returns 0.
 *
 * \param count - Returned number of compute-capable devices
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa
 * ::lwDeviceGetAttribute,
 * ::lwDeviceGetName,
 * ::lwDeviceGet,
 * ::lwDeviceTotalMem,
 * ::lwdaGetDeviceCount
 */
LWresult LWDAAPI lwDeviceGetCount(int *count);

/**
 * \brief Returns an identifer string for the device
 *
 * Returns an ASCII string identifying the device \p dev in the NULL-terminated
 * string pointed to by \p name. \p len specifies the maximum length of the
 * string that may be returned.
 *
 * \param name - Returned identifier string for the device
 * \param len  - Maximum length of string to store in \p name
 * \param dev  - Device to get identifier string for
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::lwDeviceGetAttribute,
 * ::lwDeviceGetCount,
 * ::lwDeviceGet,
 * ::lwDeviceTotalMem,
 * ::lwdaGetDeviceProperties
 */
LWresult LWDAAPI lwDeviceGetName(char *name, int len, LWdevice dev);

#if __LWDA_API_VERSION >= 3020
/**
 * \brief Returns the total amount of memory on the device
 *
 * Returns in \p *bytes the total amount of memory available on the device
 * \p dev in bytes.
 *
 * \param bytes - Returned memory available on device in bytes
 * \param dev   - Device handle
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::lwDeviceGetAttribute,
 * ::lwDeviceGetCount,
 * ::lwDeviceGetName,
 * ::lwDeviceGet,
 * ::lwdaMemGetInfo
 */
LWresult LWDAAPI lwDeviceTotalMem(size_t *bytes, LWdevice dev);
#endif /* __LWDA_API_VERSION >= 3020 */

/**
 * \brief Returns information about the device
 *
 * Returns in \p *pi the integer value of the attribute \p attrib on device
 * \p dev. The supported attributes are:
 * - ::LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: Maximum number of threads per
 *   block;
 * - ::LW_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: Maximum x-dimension of a block;
 * - ::LW_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: Maximum y-dimension of a block;
 * - ::LW_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: Maximum z-dimension of a block;
 * - ::LW_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: Maximum x-dimension of a grid;
 * - ::LW_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: Maximum y-dimension of a grid;
 * - ::LW_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: Maximum z-dimension of a grid;
 * - ::LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: Maximum amount of
 *   shared memory available to a thread block in bytes;
 * - ::LW_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: Memory available on device for
 *   __constant__ variables in a LWCA C kernel in bytes;
 * - ::LW_DEVICE_ATTRIBUTE_WARP_SIZE: Warp size in threads;
 * - ::LW_DEVICE_ATTRIBUTE_MAX_PITCH: Maximum pitch in bytes allowed by the
 *   memory copy functions that involve memory regions allocated through
 *   ::lwMemAllocPitch();
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH: Maximum 1D 
 *  texture width;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH: Maximum width
 *  for a 1D texture bound to linear memory;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH: Maximum 
 *  mipmapped 1D texture width;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH: Maximum 2D 
 *  texture width;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT: Maximum 2D 
 *  texture height;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH: Maximum width
 *  for a 2D texture bound to linear memory;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT: Maximum height
 *  for a 2D texture bound to linear memory;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH: Maximum pitch
 *  in bytes for a 2D texture bound to linear memory;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH: Maximum 
 *  mipmapped 2D texture width;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT: Maximum
 *  mipmapped 2D texture height;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH: Maximum 3D 
 *  texture width;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT: Maximum 3D 
 *  texture height;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH: Maximum 3D 
 *  texture depth;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE: 
 *  Alternate maximum 3D texture width, 0 if no alternate
 *  maximum 3D texture size is supported;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE: 
 *  Alternate maximum 3D texture height, 0 if no alternate
 *  maximum 3D texture size is supported;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE: 
 *  Alternate maximum 3D texture depth, 0 if no alternate
 *  maximum 3D texture size is supported;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURELWBEMAP_WIDTH:
 *  Maximum lwbemap texture width or height;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH: 
 *  Maximum 1D layered texture width;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS: 
 *   Maximum layers in a 1D layered texture;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH: 
 *  Maximum 2D layered texture width;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT: 
 *   Maximum 2D layered texture height;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS: 
 *   Maximum layers in a 2D layered texture;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURELWBEMAP_LAYERED_WIDTH: 
 *   Maximum lwbemap layered texture width or height;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURELWBEMAP_LAYERED_LAYERS: 
 *   Maximum layers in a lwbemap layered texture;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH:
 *   Maximum 1D surface width;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH:
 *   Maximum 2D surface width;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT:
 *   Maximum 2D surface height;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH:
 *   Maximum 3D surface width;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT:
 *   Maximum 3D surface height;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH:
 *   Maximum 3D surface depth;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH:
 *   Maximum 1D layered surface width;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS:
 *   Maximum layers in a 1D layered surface;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH:
 *   Maximum 2D layered surface width;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT:
 *   Maximum 2D layered surface height;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS:
 *   Maximum layers in a 2D layered surface;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACELWBEMAP_WIDTH:
 *   Maximum lwbemap surface width;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACELWBEMAP_LAYERED_WIDTH:
 *   Maximum lwbemap layered surface width;
 * - ::LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACELWBEMAP_LAYERED_LAYERS:
 *   Maximum layers in a lwbemap layered surface;
 * - ::LW_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: Maximum number of 32-bit
 *   registers available to a thread block;
 * - ::LW_DEVICE_ATTRIBUTE_CLOCK_RATE: The typical clock frequency in kilohertz;
 * - ::LW_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: Alignment requirement; texture
 *   base addresses aligned to ::textureAlign bytes do not need an offset
 *   applied to texture fetches;
 * - ::LW_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT: Pitch alignment requirement
 *   for 2D texture references bound to pitched memory;
 * - ::LW_DEVICE_ATTRIBUTE_GPU_OVERLAP: 1 if the device can conlwrrently copy
 *   memory between host and device while exelwting a kernel, or 0 if not;
 * - ::LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: Number of multiprocessors on
 *   the device;
 * - ::LW_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT: 1 if there is a run time limit
 *   for kernels exelwted on the device, or 0 if not;
 * - ::LW_DEVICE_ATTRIBUTE_INTEGRATED: 1 if the device is integrated with the
 *   memory subsystem, or 0 if not;
 * - ::LW_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: 1 if the device can map host
 *   memory into the LWCA address space, or 0 if not;
 * - ::LW_DEVICE_ATTRIBUTE_COMPUTE_MODE: Compute mode that device is lwrrently
 *   in. Available modes are as follows:
 *   - ::LW_COMPUTEMODE_DEFAULT: Default mode - Device is not restricted and
 *     can have multiple LWCA contexts present at a single time.
 *   - ::LW_COMPUTEMODE_PROHIBITED: Compute-prohibited mode - Device is
 *     prohibited from creating new LWCA contexts.
 *   - ::LW_COMPUTEMODE_EXCLUSIVE_PROCESS:  Compute-exclusive-process mode - Device
 *     can have only one context used by a single process at a time.
 * - ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_KERNELS: 1 if the device supports
 *   exelwting multiple kernels within the same context simultaneously, or 0 if
 *   not. It is not guaranteed that multiple kernels will be resident
 *   on the device conlwrrently so this feature should not be relied upon for
 *   correctness;
 * - ::LW_DEVICE_ATTRIBUTE_ECC_ENABLED: 1 if error correction is enabled on the
 *    device, 0 if error correction is disabled or not supported by the device;
 * - ::LW_DEVICE_ATTRIBUTE_PCI_BUS_ID: PCI bus identifier of the device;
 * - ::LW_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: PCI device (also known as slot) identifier
 *   of the device;
 * - ::LW_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: PCI domain identifier of the device
 * - ::LW_DEVICE_ATTRIBUTE_TCC_DRIVER: 1 if the device is using a TCC driver. TCC
 *    is only available on Tesla hardware running Windows Vista or later;
 * - ::LW_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: Peak memory clock frequency in kilohertz;
 * - ::LW_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: Global memory bus width in bits;
 * - ::LW_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: Size of L2 cache in bytes. 0 if the device doesn't have L2 cache;
 * - ::LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: Maximum resident threads per multiprocessor;
 * - ::LW_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: 1 if the device shares a unified address space with 
 *   the host, or 0 if not;
 * - ::LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: Major compute capability version number;
 * - ::LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: Minor compute capability version number;
 * - ::LW_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED: 1 if device supports caching globals 
 *    in L1 cache, 0 if caching globals in L1 cache is not supported by the device;
 * - ::LW_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED: 1 if device supports caching locals 
 *    in L1 cache, 0 if caching locals in L1 cache is not supported by the device;
 * - ::LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: Maximum amount of
 *   shared memory available to a multiprocessor in bytes; this amount is shared
 *   by all thread blocks simultaneously resident on a multiprocessor;
 * - ::LW_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: Maximum number of 32-bit
 *   registers available to a multiprocessor; this number is shared by all thread
 *   blocks simultaneously resident on a multiprocessor;
 * - ::LW_DEVICE_ATTRIBUTE_MANAGED_MEMORY: 1 if device supports allocating managed memory
 *   on this system, 0 if allocating managed memory is not supported by the device on this system.
 * - ::LW_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD: 1 if device is on a multi-GPU board, 0 if not.
 * - ::LW_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID: Unique identifier for a group of devices
 *   associated with the same board. Devices on the same multi-GPU board will share the same identifier.
 * - ::LW_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED: 1 if Link between the device and the host
 *   supports native atomic operations.
 * - ::LW_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO: Ratio of single precision performance
 *   (in floating-point operations per second) to double precision performance.
 * - ::LW_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS: Device suppports coherently accessing
 *   pageable memory without calling lwdaHostRegister on it.
 * - ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS: Device can coherently access managed memory
 *   conlwrrently with the CPU.
 * - ::LW_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED: Device supports Compute Preemption.
 * - ::LW_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM: Device can access host registered
 *   memory at the same virtual address as the CPU.
 * -  ::LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: The maximum per block shared memory size
 *    suported on this device. This is the maximum value that can be opted into when using the lwFuncSetAttribute() call.
 *    For more details see ::LW_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
 *
 * \param pi     - Returned device attribute value
 * \param attrib - Device attribute to query
 * \param dev    - Device handle
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::lwDeviceGetCount,
 * ::lwDeviceGetName,
 * ::lwDeviceGet,
 * ::lwDeviceTotalMem,
 * ::lwdaDeviceGetAttribute,
 * ::lwdaGetDeviceProperties
 */
LWresult LWDAAPI lwDeviceGetAttribute(int *pi, LWdevice_attribute attrib, LWdevice dev);

/** @} */ /* END LWDA_DEVICE */

/**
 * \defgroup LWDA_DEVICE_DEPRECATED Device Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated device management functions of the low-level LWCA
 * driver API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the device management functions of the low-level
 * LWCA driver application programming interface.
 *
 * @{
 */

/**
 * \brief Returns properties for a selected device
 *
 * \deprecated
 *
 * This function was deprecated as of LWCA 5.0 and replaced by ::lwDeviceGetAttribute().
 *
 * Returns in \p *prop the properties of device \p dev. The ::LWdevprop
 * structure is defined as:
 *
 * \code
     typedef struct LWdevprop_st {
     int maxThreadsPerBlock;
     int maxThreadsDim[3];
     int maxGridSize[3];
     int sharedMemPerBlock;
     int totalConstantMemory;
     int SIMDWidth;
     int memPitch;
     int regsPerBlock;
     int clockRate;
     int textureAlign
  } LWdevprop;
 * \endcode
 * where:
 *
 * - ::maxThreadsPerBlock is the maximum number of threads per block;
 * - ::maxThreadsDim[3] is the maximum sizes of each dimension of a block;
 * - ::maxGridSize[3] is the maximum sizes of each dimension of a grid;
 * - ::sharedMemPerBlock is the total amount of shared memory available per
 *   block in bytes;
 * - ::totalConstantMemory is the total amount of constant memory available on
 *   the device in bytes;
 * - ::SIMDWidth is the warp size;
 * - ::memPitch is the maximum pitch allowed by the memory copy functions that
 *   involve memory regions allocated through ::lwMemAllocPitch();
 * - ::regsPerBlock is the total number of registers available per block;
 * - ::clockRate is the clock frequency in kilohertz;
 * - ::textureAlign is the alignment requirement; texture base addresses that
 *   are aligned to ::textureAlign bytes do not need an offset applied to
 *   texture fetches.
 *
 * \param prop - Returned properties of device
 * \param dev  - Device to get properties for
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::lwDeviceGetAttribute,
 * ::lwDeviceGetCount,
 * ::lwDeviceGetName,
 * ::lwDeviceGet,
 * ::lwDeviceTotalMem
 */
LWresult LWDAAPI lwDeviceGetProperties(LWdevprop *prop, LWdevice dev);

/**
 * \brief Returns the compute capability of the device
 *
 * \deprecated
 *
 * This function was deprecated as of LWCA 5.0 and its functionality superceded
 * by ::lwDeviceGetAttribute(). 
 *
 * Returns in \p *major and \p *minor the major and minor revision numbers that
 * define the compute capability of the device \p dev.
 *
 * \param major - Major revision number
 * \param minor - Minor revision number
 * \param dev   - Device handle
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::lwDeviceGetAttribute,
 * ::lwDeviceGetCount,
 * ::lwDeviceGetName,
 * ::lwDeviceGet,
 * ::lwDeviceTotalMem
 */
LWresult LWDAAPI lwDeviceComputeCapability(int *major, int *minor, LWdevice dev);

/** @} */ /* END LWDA_DEVICE_DEPRECATED */

/**
 * \defgroup LWDA_PRIMARY_CTX Primary Context Management
 *
 * ___MANBRIEF___ primary context management functions of the low-level LWCA driver
 * API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the primary context management functions of the low-level
 * LWCA driver application programming interface.
 *
 * The primary context is unique per device and shared with the LWCA runtime API.
 * These functions allow integration with other libraries using LWCA.
 *
 * @{
 */

#if __LWDA_API_VERSION >= 7000

/**
 * \brief Retain the primary context on the GPU
 *
 * Retains the primary context on the device, creating it if necessary,
 * increasing its usage count. The caller must call
 * ::lwDevicePrimaryCtxRelease() when done using the context.
 * Unlike ::lwCtxCreate() the newly created context is not pushed onto the stack.
 *
 * Context creation will fail with ::LWDA_ERROR_UNKNOWN if the compute mode of
 * the device is ::LW_COMPUTEMODE_PROHIBITED.  The function ::lwDeviceGetAttribute() 
 * can be used with ::LW_DEVICE_ATTRIBUTE_COMPUTE_MODE to determine the compute mode 
 * of the device. 
 * The <i>lwpu-smi</i> tool can be used to set the compute mode for
 * devices. Documentation for <i>lwpu-smi</i> can be obtained by passing a
 * -h option to it.
 *
 * Please note that the primary context always supports pinned allocations. Other
 * flags can be specified by ::lwDevicePrimaryCtxSetFlags().
 *
 * \param pctx  - Returned context handle of the new context
 * \param dev   - Device for which primary context is requested
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_DEVICE,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa ::lwDevicePrimaryCtxRelease,
 * ::lwDevicePrimaryCtxSetFlags,
 * ::lwCtxCreate,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize
 */
LWresult LWDAAPI lwDevicePrimaryCtxRetain(LWcontext *pctx, LWdevice dev);

/**
 * \brief Release the primary context on the GPU
 *
 * Releases the primary context interop on the device by decreasing the usage
 * count by 1. If the usage drops to 0 the primary context of device \p dev
 * will be destroyed regardless of how many threads it is current to.
 *
 * Please note that unlike ::lwCtxDestroy() this method does not pop the context
 * from stack in any cirlwmstances.
 *
 * \param dev - Device which primary context is released
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 *
 * \sa ::lwDevicePrimaryCtxRetain,
 * ::lwCtxDestroy,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize
 */
LWresult LWDAAPI lwDevicePrimaryCtxRelease(LWdevice dev);

/**
 * \brief Set flags for the primary context
 *
 * Sets the flags for the primary context on the device overwriting perviously
 * set ones. If the primary context is already created
 * ::LWDA_ERROR_PRIMARY_CONTEXT_ACTIVE is returned.
 *
 * The three LSBs of the \p flags parameter can be used to control how the OS
 * thread, which owns the LWCA context at the time of an API call, interacts
 * with the OS scheduler when waiting for results from the GPU. Only one of
 * the scheduling flags can be set when creating a context.
 *
 * - ::LW_CTX_SCHED_SPIN: Instruct LWCA to actively spin when waiting for
 * results from the GPU. This can decrease latency when waiting for the GPU,
 * but may lower the performance of CPU threads if they are performing work in
 * parallel with the LWCA thread.
 *
 * - ::LW_CTX_SCHED_YIELD: Instruct LWCA to yield its thread when waiting for
 * results from the GPU. This can increase latency when waiting for the GPU,
 * but can increase the performance of CPU threads performing work in parallel
 * with the GPU.
 *
 * - ::LW_CTX_SCHED_BLOCKING_SYNC: Instruct LWCA to block the CPU thread on a
 * synchronization primitive when waiting for the GPU to finish work.
 *
 * - ::LW_CTX_BLOCKING_SYNC: Instruct LWCA to block the CPU thread on a
 * synchronization primitive when waiting for the GPU to finish work. <br>
 * <b>Deprecated:</b> This flag was deprecated as of LWCA 4.0 and was
 * replaced with ::LW_CTX_SCHED_BLOCKING_SYNC.
 *
 * - ::LW_CTX_SCHED_AUTO: The default value if the \p flags parameter is zero,
 * uses a heuristic based on the number of active LWCA contexts in the
 * process \e C and the number of logical processors in the system \e P. If
 * \e C > \e P, then LWCA will yield to other OS threads when waiting for
 * the GPU (::LW_CTX_SCHED_YIELD), otherwise LWCA will not yield while
 * waiting for results and actively spin on the processor (::LW_CTX_SCHED_SPIN).
 * However, on low power devices like CheetAh, it always defaults to
 * ::LW_CTX_SCHED_BLOCKING_SYNC.
 *
 * - ::LW_CTX_LMEM_RESIZE_TO_MAX: Instruct LWCA to not reduce local memory
 * after resizing local memory for a kernel. This can prevent thrashing by
 * local memory allocations when launching many kernels with high local
 * memory usage at the cost of potentially increased memory usage.
 *
 * \param dev   - Device for which the primary context flags are set
 * \param flags - New flags for the device
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_DEVICE,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_PRIMARY_CONTEXT_ACTIVE
 * \notefnerr
 *
 * \sa ::lwDevicePrimaryCtxRetain,
 * ::lwDevicePrimaryCtxGetState,
 * ::lwCtxCreate,
 * ::lwCtxGetFlags,
 * ::lwdaSetDeviceFlags
 */
LWresult LWDAAPI lwDevicePrimaryCtxSetFlags(LWdevice dev, unsigned int flags);

/**
 * \brief Get the state of the primary context
 *
 * Returns in \p *flags the flags for the primary context of \p dev, and in
 * \p *active whether it is active.  See ::lwDevicePrimaryCtxSetFlags for flag
 * values.
 *
 * \param dev    - Device to get primary context flags for
 * \param flags  - Pointer to store flags
 * \param active - Pointer to store context state; 0 = inactive, 1 = active
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_DEVICE,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * \notefnerr
 *
 * \sa
 * ::lwDevicePrimaryCtxSetFlags,
 * ::lwCtxGetFlags,
 * ::lwdaGetDeviceFlags
 */
LWresult LWDAAPI lwDevicePrimaryCtxGetState(LWdevice dev, unsigned int *flags, int *active);

/**
 * \brief Destroy all allocations and reset all state on the primary context
 *
 * Explicitly destroys and cleans up all resources associated with the current
 * device in the current process.
 *
 * Note that it is responsibility of the calling function to ensure that no
 * other module in the process is using the device any more. For that reason
 * it is recommended to use ::lwDevicePrimaryCtxRelease() in most cases.
 * However it is safe for other modules to call ::lwDevicePrimaryCtxRelease()
 * even after resetting the device.
 *
 * \param dev - Device for which primary context is destroyed
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_DEVICE,
 * ::LWDA_ERROR_PRIMARY_CONTEXT_ACTIVE
 * \notefnerr
 *
 * \sa ::lwDevicePrimaryCtxRetain,
 * ::lwDevicePrimaryCtxRelease,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize,
 * ::lwdaDeviceReset
 */
LWresult LWDAAPI lwDevicePrimaryCtxReset(LWdevice dev);

#endif /* __LWDA_API_VERSION >= 7000 */

/** @} */ /* END LWDA_PRIMARY_CTX */


/**
 * \defgroup LWDA_CTX Context Management
 *
 * ___MANBRIEF___ context management functions of the low-level LWCA driver
 * API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the context management functions of the low-level
 * LWCA driver application programming interface.
 *
 * @{
 */

#if __LWDA_API_VERSION >= 3020
/**
 * \brief Create a LWCA context
 *
 * Creates a new LWCA context and associates it with the calling thread. The
 * \p flags parameter is described below. The context is created with a usage
 * count of 1 and the caller of ::lwCtxCreate() must call ::lwCtxDestroy() or
 * when done using the context. If a context is already current to the thread, 
 * it is supplanted by the newly created context and may be restored by a subsequent 
 * call to ::lwCtxPopLwrrent().
 *
 * The three LSBs of the \p flags parameter can be used to control how the OS
 * thread, which owns the LWCA context at the time of an API call, interacts
 * with the OS scheduler when waiting for results from the GPU. Only one of
 * the scheduling flags can be set when creating a context.
 *
 * - ::LW_CTX_SCHED_SPIN: Instruct LWCA to actively spin when waiting for
 * results from the GPU. This can decrease latency when waiting for the GPU,
 * but may lower the performance of CPU threads if they are performing work in
 * parallel with the LWCA thread.
 *
 * - ::LW_CTX_SCHED_YIELD: Instruct LWCA to yield its thread when waiting for
 * results from the GPU. This can increase latency when waiting for the GPU,
 * but can increase the performance of CPU threads performing work in parallel
 * with the GPU.
 * 
 * - ::LW_CTX_SCHED_BLOCKING_SYNC: Instruct LWCA to block the CPU thread on a
 * synchronization primitive when waiting for the GPU to finish work.
 *
 * - ::LW_CTX_BLOCKING_SYNC: Instruct LWCA to block the CPU thread on a
 * synchronization primitive when waiting for the GPU to finish work. <br>
 * <b>Deprecated:</b> This flag was deprecated as of LWCA 4.0 and was
 * replaced with ::LW_CTX_SCHED_BLOCKING_SYNC. 
 *
 * - ::LW_CTX_SCHED_AUTO: The default value if the \p flags parameter is zero,
 * uses a heuristic based on the number of active LWCA contexts in the
 * process \e C and the number of logical processors in the system \e P. If
 * \e C > \e P, then LWCA will yield to other OS threads when waiting for 
 * the GPU (::LW_CTX_SCHED_YIELD), otherwise LWCA will not yield while 
 * waiting for results and actively spin on the processor (::LW_CTX_SCHED_SPIN). 
 * However, on low power devices like CheetAh, it always defaults to 
 * ::LW_CTX_SCHED_BLOCKING_SYNC.
 *
 * - ::LW_CTX_MAP_HOST: Instruct LWCA to support mapped pinned allocations.
 * This flag must be set in order to allocate pinned host memory that is
 * accessible to the GPU.
 *
 * - ::LW_CTX_LMEM_RESIZE_TO_MAX: Instruct LWCA to not reduce local memory
 * after resizing local memory for a kernel. This can prevent thrashing by
 * local memory allocations when launching many kernels with high local
 * memory usage at the cost of potentially increased memory usage.
 *
 * Context creation will fail with ::LWDA_ERROR_UNKNOWN if the compute mode of
 * the device is ::LW_COMPUTEMODE_PROHIBITED. The function ::lwDeviceGetAttribute() 
 * can be used with ::LW_DEVICE_ATTRIBUTE_COMPUTE_MODE to determine the 
 * compute mode of the device. The <i>lwpu-smi</i> tool can be used to set 
 * the compute mode for * devices. 
 * Documentation for <i>lwpu-smi</i> can be obtained by passing a
 * -h option to it.
 *
 * \param pctx  - Returned context handle of the new context
 * \param flags - Context creation flags
 * \param dev   - Device to create context on
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_DEVICE,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa ::lwCtxDestroy,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize
 */
LWresult LWDAAPI lwCtxCreate(LWcontext *pctx, unsigned int flags, LWdevice dev);
#endif /* __LWDA_API_VERSION >= 3020 */

#if __LWDA_API_VERSION >= 4000
/**
 * \brief Destroy a LWCA context
 *
 * Destroys the LWCA context specified by \p ctx.  The context \p ctx will be
 * destroyed regardless of how many threads it is current to.
 * It is the responsibility of the calling function to ensure that no API
 * call issues using \p ctx while ::lwCtxDestroy() is exelwting.
 *
 * If \p ctx is current to the calling thread then \p ctx will also be 
 * popped from the current thread's context stack (as though ::lwCtxPopLwrrent()
 * were called).  If \p ctx is current to other threads, then \p ctx will
 * remain current to those threads, and attempting to access \p ctx from
 * those threads will result in the error ::LWDA_ERROR_CONTEXT_IS_DESTROYED.
 *
 * \param ctx - Context to destroy
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwCtxCreate,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize
 */
LWresult LWDAAPI lwCtxDestroy(LWcontext ctx);
#endif /* __LWDA_API_VERSION >= 4000 */

#if __LWDA_API_VERSION >= 4000
/**
 * \brief Pushes a context on the current CPU thread
 *
 * Pushes the given context \p ctx onto the CPU thread's stack of current
 * contexts. The specified context becomes the CPU thread's current context, so
 * all LWCA functions that operate on the current context are affected.
 *
 * The previous current context may be made current again by calling
 * ::lwCtxDestroy() or ::lwCtxPopLwrrent().
 *
 * \param ctx - Context to push
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwCtxCreate,
 * ::lwCtxDestroy,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize
 */
LWresult LWDAAPI lwCtxPushLwrrent(LWcontext ctx);

/**
 * \brief Pops the current LWCA context from the current CPU thread.
 *
 * Pops the current LWCA context from the CPU thread and passes back the 
 * old context handle in \p *pctx. That context may then be made current 
 * to a different CPU thread by calling ::lwCtxPushLwrrent().
 *
 * If a context was current to the CPU thread before ::lwCtxCreate() or
 * ::lwCtxPushLwrrent() was called, this function makes that context current to
 * the CPU thread again.
 *
 * \param pctx - Returned new context handle
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT
 * \notefnerr
 *
 * \sa ::lwCtxCreate,
 * ::lwCtxDestroy,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize
 */
LWresult LWDAAPI lwCtxPopLwrrent(LWcontext *pctx);

/**
 * \brief Binds the specified LWCA context to the calling CPU thread
 *
 * Binds the specified LWCA context to the calling CPU thread.
 * If \p ctx is NULL then the LWCA context previously bound to the
 * calling CPU thread is unbound and ::LWDA_SUCCESS is returned.
 *
 * If there exists a LWCA context stack on the calling CPU thread, this
 * will replace the top of that stack with \p ctx.  
 * If \p ctx is NULL then this will be equivalent to popping the top
 * of the calling CPU thread's LWCA context stack (or a no-op if the
 * calling CPU thread's LWCA context stack is empty).
 *
 * \param ctx - Context to bind to the calling CPU thread
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT
 * \notefnerr
 *
 * \sa
 * ::lwCtxGetLwrrent,
 * ::lwCtxCreate,
 * ::lwCtxDestroy,
 * ::lwdaSetDevice
 */
LWresult LWDAAPI lwCtxSetLwrrent(LWcontext ctx);

/**
 * \brief Returns the LWCA context bound to the calling CPU thread.
 *
 * Returns in \p *pctx the LWCA context bound to the calling CPU thread.
 * If no context is bound to the calling CPU thread then \p *pctx is
 * set to NULL and ::LWDA_SUCCESS is returned.
 *
 * \param pctx - Returned context handle
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * \notefnerr
 *
 * \sa
 * ::lwCtxSetLwrrent,
 * ::lwCtxCreate,
 * ::lwCtxDestroy,
 * ::lwdaGetDevice
 */
LWresult LWDAAPI lwCtxGetLwrrent(LWcontext *pctx);
#endif /* __LWDA_API_VERSION >= 4000 */

/**
 * \brief Returns the device ID for the current context
 *
 * Returns in \p *device the ordinal of the current context's device.
 *
 * \param device - Returned device ID for the current context
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * \notefnerr
 *
 * \sa ::lwCtxCreate,
 * ::lwCtxDestroy,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize,
 * ::lwdaGetDevice
 */
LWresult LWDAAPI lwCtxGetDevice(LWdevice *device);

#if __LWDA_API_VERSION >= 7000
/**
 * \brief Returns the flags for the current context
 *
 * Returns in \p *flags the flags of the current context. See ::lwCtxCreate
 * for flag values.
 *
 * \param flags - Pointer to store flags of current context
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * \notefnerr
 *
 * \sa ::lwCtxCreate,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetLwrrent,
 * ::lwCtxGetDevice
 * ::lwCtxGetLimit,
 * ::lwCtxGetSharedMemConfig,
 * ::lwCtxGetStreamPriorityRange,
 * ::lwdaGetDeviceFlags
 */
LWresult LWDAAPI lwCtxGetFlags(unsigned int *flags);
#endif /* __LWDA_API_VERSION >= 7000 */

/**
 * \brief Block for a context's tasks to complete
 *
 * Blocks until the device has completed all preceding requested tasks.
 * ::lwCtxSynchronize() returns an error if one of the preceding tasks failed.
 * If the context was created with the ::LW_CTX_SCHED_BLOCKING_SYNC flag, the 
 * CPU thread will block until the GPU context has finished its work.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT
 * \notefnerr
 *
 * \sa ::lwCtxCreate,
 * ::lwCtxDestroy,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxSetLimit,
 * ::lwdaDeviceSynchronize
 */
LWresult LWDAAPI lwCtxSynchronize(void);

/**
 * \brief Set resource limits
 *
 * Setting \p limit to \p value is a request by the application to update
 * the current limit maintained by the context. The driver is free to
 * modify the requested value to meet h/w requirements (this could be
 * clamping to minimum or maximum values, rounding up to nearest element
 * size, etc). The application can use ::lwCtxGetLimit() to find out exactly
 * what the limit has been set to.
 *
 * Setting each ::LWlimit has its own specific restrictions, so each is
 * dislwssed here.
 *
 * - ::LW_LIMIT_STACK_SIZE controls the stack size in bytes of each GPU thread.
 *
 * - ::LW_LIMIT_PRINTF_FIFO_SIZE controls the size in bytes of the FIFO used
 *   by the ::printf() device system call. Setting ::LW_LIMIT_PRINTF_FIFO_SIZE
 *   must be performed before launching any kernel that uses the ::printf()
 *   device system call, otherwise ::LWDA_ERROR_ILWALID_VALUE will be returned.
 *
 * - ::LW_LIMIT_MALLOC_HEAP_SIZE controls the size in bytes of the heap used
 *   by the ::malloc() and ::free() device system calls. Setting
 *   ::LW_LIMIT_MALLOC_HEAP_SIZE must be performed before launching any kernel
 *   that uses the ::malloc() or ::free() device system calls, otherwise
 *   ::LWDA_ERROR_ILWALID_VALUE will be returned.
 *
 * - ::LW_LIMIT_DEV_RUNTIME_SYNC_DEPTH controls the maximum nesting depth of
 *   a grid at which a thread can safely call ::lwdaDeviceSynchronize(). Setting
 *   this limit must be performed before any launch of a kernel that uses the 
 *   device runtime and calls ::lwdaDeviceSynchronize() above the default sync
 *   depth, two levels of grids. Calls to ::lwdaDeviceSynchronize() will fail 
 *   with error code ::lwdaErrorSyncDepthExceeded if the limitation is 
 *   violated. This limit can be set smaller than the default or up the maximum
 *   launch depth of 24. When setting this limit, keep in mind that additional
 *   levels of sync depth require the driver to reserve large amounts of device
 *   memory which can no longer be used for user allocations. If these 
 *   reservations of device memory fail, ::lwCtxSetLimit will return 
 *   ::LWDA_ERROR_OUT_OF_MEMORY, and the limit can be reset to a lower value.
 *   This limit is only applicable to devices of compute capability 3.5 and
 *   higher. Attempting to set this limit on devices of compute capability less
 *   than 3.5 will result in the error ::LWDA_ERROR_UNSUPPORTED_LIMIT being 
 *   returned.
 *
 * - ::LW_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT controls the maximum number of
 *   outstanding device runtime launches that can be made from the current
 *   context. A grid is outstanding from the point of launch up until the grid
 *   is known to have been completed. Device runtime launches which violate 
 *   this limitation fail and return ::lwdaErrorLaunchPendingCountExceeded when
 *   ::lwdaGetLastError() is called after launch. If more pending launches than
 *   the default (2048 launches) are needed for a module using the device
 *   runtime, this limit can be increased. Keep in mind that being able to
 *   sustain additional pending launches will require the driver to reserve
 *   larger amounts of device memory upfront which can no longer be used for
 *   allocations. If these reservations fail, ::lwCtxSetLimit will return
 *   ::LWDA_ERROR_OUT_OF_MEMORY, and the limit can be reset to a lower value.
 *   This limit is only applicable to devices of compute capability 3.5 and
 *   higher. Attempting to set this limit on devices of compute capability less
 *   than 3.5 will result in the error ::LWDA_ERROR_UNSUPPORTED_LIMIT being
 *   returned.
 *
 * \param limit - Limit to set
 * \param value - Size of limit
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_UNSUPPORTED_LIMIT,
 * ::LWDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa ::lwCtxCreate,
 * ::lwCtxDestroy,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxSynchronize,
 * ::lwdaDeviceSetLimit
 */
LWresult LWDAAPI lwCtxSetLimit(LWlimit limit, size_t value);

/**
 * \brief Returns resource limits
 *
 * Returns in \p *pvalue the current size of \p limit.  The supported
 * ::LWlimit values are:
 * - ::LW_LIMIT_STACK_SIZE: stack size in bytes of each GPU thread.
 * - ::LW_LIMIT_PRINTF_FIFO_SIZE: size in bytes of the FIFO used by the
 *   ::printf() device system call.
 * - ::LW_LIMIT_MALLOC_HEAP_SIZE: size in bytes of the heap used by the
 *   ::malloc() and ::free() device system calls.
 * - ::LW_LIMIT_DEV_RUNTIME_SYNC_DEPTH: maximum grid depth at which a thread
 *   can issue the device runtime call ::lwdaDeviceSynchronize() to wait on
 *   child grid launches to complete.
 * - ::LW_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT: maximum number of outstanding
 *   device runtime launches that can be made from this context.
 *
 * \param limit  - Limit to query
 * \param pvalue - Returned size of limit
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_UNSUPPORTED_LIMIT
 * \notefnerr
 *
 * \sa ::lwCtxCreate,
 * ::lwCtxDestroy,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize,
 * ::lwdaDeviceGetLimit
 */
LWresult LWDAAPI lwCtxGetLimit(size_t *pvalue, LWlimit limit);

/**
 * \brief Returns the preferred cache configuration for the current context.
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this function returns through \p pconfig the preferred cache configuration
 * for the current context. This is only a preference. The driver will use
 * the requested configuration if possible, but it is free to choose a different
 * configuration if required to execute functions.
 *
 * This will return a \p pconfig of ::LW_FUNC_CACHE_PREFER_NONE on devices
 * where the size of the L1 cache and shared memory are fixed.
 *
 * The supported cache configurations are:
 * - ::LW_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)
 * - ::LW_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache
 * - ::LW_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory
 * - ::LW_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory
 *
 * \param pconfig - Returned cache configuration
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwCtxCreate,
 * ::lwCtxDestroy,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize,
 * ::lwFuncSetCacheConfig,
 * ::lwdaDeviceGetCacheConfig
 */
LWresult LWDAAPI lwCtxGetCacheConfig(LWfunc_cache *pconfig);

/**
 * \brief Sets the preferred cache configuration for the current context.
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p config the preferred cache configuration for
 * the current context. This is only a preference. The driver will use
 * the requested configuration if possible, but it is free to choose a different
 * configuration if required to execute the function. Any function preference
 * set via ::lwFuncSetCacheConfig() will be preferred over this context-wide
 * setting. Setting the context-wide cache configuration to
 * ::LW_FUNC_CACHE_PREFER_NONE will cause subsequent kernel launches to prefer
 * to not change the cache configuration unless required to launch the kernel.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 * - ::LW_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)
 * - ::LW_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache
 * - ::LW_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory
 * - ::LW_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory
 *
 * \param config - Requested cache configuration
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwCtxCreate,
 * ::lwCtxDestroy,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize,
 * ::lwFuncSetCacheConfig,
 * ::lwdaDeviceSetCacheConfig
 */
LWresult LWDAAPI lwCtxSetCacheConfig(LWfunc_cache config);

#if __LWDA_API_VERSION >= 4020
/**
 * \brief Returns the current shared memory configuration for the current context.
 *
 * This function will return in \p pConfig the current size of shared memory banks
 * in the current context. On devices with configurable shared memory banks, 
 * ::lwCtxSetSharedMemConfig can be used to change this setting, so that all 
 * subsequent kernel launches will by default use the new bank size. When 
 * ::lwCtxGetSharedMemConfig is called on devices without configurable shared 
 * memory, it will return the fixed bank size of the hardware.
 *
 * The returned bank configurations can be either:
 * - ::LW_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE:  shared memory bank width is 
 *   four bytes.
 * - ::LW_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: shared memory bank width will
 *   eight bytes.
 *
 * \param pConfig - returned shared memory configuration
 * \return 
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwCtxCreate,
 * ::lwCtxDestroy,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize,
 * ::lwCtxGetSharedMemConfig,
 * ::lwFuncSetCacheConfig,
 * ::lwdaDeviceGetSharedMemConfig
 */
LWresult LWDAAPI lwCtxGetSharedMemConfig(LWsharedconfig *pConfig);

/**
 * \brief Sets the shared memory configuration for the current context.
 *
 * On devices with configurable shared memory banks, this function will set
 * the context's shared memory bank size which is used for subsequent kernel 
 * launches. 
 *
 * Changed the shared memory configuration between launches may insert a device
 * side synchronization point between those launches.
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
 * - ::LW_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE: set bank width to the default initial
 *   setting (lwrrently, four bytes).
 * - ::LW_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: set shared memory bank width to
 *   be natively four bytes.
 * - ::LW_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: set shared memory bank width to
 *   be natively eight bytes.
 *
 * \param config - requested shared memory configuration
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwCtxCreate,
 * ::lwCtxDestroy,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize,
 * ::lwCtxGetSharedMemConfig,
 * ::lwFuncSetCacheConfig,
 * ::lwdaDeviceSetSharedMemConfig
 */
LWresult LWDAAPI lwCtxSetSharedMemConfig(LWsharedconfig config);
#endif

/**
 * \brief Gets the context's API version.
 *
 * Returns a version number in \p version corresponding to the capabilities of
 * the context (e.g. 3010 or 3020), which library developers can use to direct
 * callers to a specific API version. If \p ctx is NULL, returns the API version
 * used to create the lwrrently bound context.
 *
 * Note that new API versions are only introduced when context capabilities are
 * changed that break binary compatibility, so the API version and driver version
 * may be different. For example, it is valid for the API version to be 3020 while
 * the driver version is 4020.
 *
 * \param ctx     - Context to check
 * \param version - Pointer to version
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa ::lwCtxCreate,
 * ::lwCtxDestroy,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize
 */
LWresult LWDAAPI lwCtxGetApiVersion(LWcontext ctx, unsigned int *version);

/**
 * \brief Returns numerical values that correspond to the least and
 * greatest stream priorities.
 *
 * Returns in \p *leastPriority and \p *greatestPriority the numerical values that correspond
 * to the least and greatest stream priorities respectively. Stream priorities
 * follow a convention where lower numbers imply greater priorities. The range of
 * meaningful stream priorities is given by [\p *greatestPriority, \p *leastPriority].
 * If the user attempts to create a stream with a priority value that is
 * outside the meaningful range as specified by this API, the priority is
 * automatically clamped down or up to either \p *leastPriority or \p *greatestPriority
 * respectively. See ::lwStreamCreateWithPriority for details on creating a
 * priority stream.
 * A NULL may be passed in for \p *leastPriority or \p *greatestPriority if the value
 * is not desired.
 *
 * This function will return '0' in both \p *leastPriority and \p *greatestPriority if
 * the current context's device does not support stream priorities
 * (see ::lwDeviceGetAttribute).
 *
 * \param leastPriority    - Pointer to an int in which the numerical value for least
 *                           stream priority is returned
 * \param greatestPriority - Pointer to an int in which the numerical value for greatest
 *                           stream priority is returned
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * \notefnerr
 *
 * \sa ::lwStreamCreateWithPriority,
 * ::lwStreamGetPriority,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize,
 * ::lwdaDeviceGetStreamPriorityRange
 */
LWresult LWDAAPI lwCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority);

/** @} */ /* END LWDA_CTX */

/**
 * \defgroup LWDA_CTX_DEPRECATED Context Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated context management functions of the low-level LWCA
 * driver API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the deprecated context management functions of the low-level
 * LWCA driver application programming interface.
 *
 * @{
 */

/**
 * \brief Increment a context's usage-count
 *
 * \deprecated
 *
 * Note that this function is deprecated and should not be used.
 *
 * Increments the usage count of the context and passes back a context handle
 * in \p *pctx that must be passed to ::lwCtxDetach() when the application is
 * done with the context. ::lwCtxAttach() fails if there is no context current
 * to the thread.
 *
 * Lwrrently, the \p flags parameter must be 0.
 *
 * \param pctx  - Returned context handle of the current context
 * \param flags - Context attach flags (must be 0)
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwCtxCreate,
 * ::lwCtxDestroy,
 * ::lwCtxDetach,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize
 */
LWresult LWDAAPI lwCtxAttach(LWcontext *pctx, unsigned int flags);

/**
 * \brief Decrement a context's usage-count
 *
 * \deprecated
 *
 * Note that this function is deprecated and should not be used.
 *
 * Decrements the usage count of the context \p ctx, and destroys the context
 * if the usage count goes to 0. The context must be a handle that was passed
 * back by ::lwCtxCreate() or ::lwCtxAttach(), and must be current to the
 * calling thread.
 *
 * \param ctx - Context to destroy
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT
 * \notefnerr
 *
 * \sa ::lwCtxCreate,
 * ::lwCtxDestroy,
 * ::lwCtxGetApiVersion,
 * ::lwCtxGetCacheConfig,
 * ::lwCtxGetDevice,
 * ::lwCtxGetFlags,
 * ::lwCtxGetLimit,
 * ::lwCtxPopLwrrent,
 * ::lwCtxPushLwrrent,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxSetLimit,
 * ::lwCtxSynchronize
 */
LWresult LWDAAPI lwCtxDetach(LWcontext ctx);

/** @} */ /* END LWDA_CTX_DEPRECATED */


/**
 * \defgroup LWDA_MODULE Module Management
 *
 * ___MANBRIEF___ module management functions of the low-level LWCA driver API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the module management functions of the low-level LWCA
 * driver application programming interface.
 *
 * @{
 */

/**
 * \brief Loads a compute module
 *
 * Takes a filename \p fname and loads the corresponding module \p module into
 * the current context. The LWCA driver API does not attempt to lazily
 * allocate the resources needed by a module; if the memory for functions and
 * data (constant and global) needed by the module cannot be allocated,
 * ::lwModuleLoad() fails. The file should be a \e lwbin file as output by
 * \b lwcc, or a \e PTX file either as output by \b lwcc or handwritten, or
 * a \e fatbin file as output by \b lwcc from toolchain 4.0 or later.
 *
 * \param module - Returned module
 * \param fname  - Filename of module to load
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_PTX,
 * ::LWDA_ERROR_NOT_FOUND,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_FILE_NOT_FOUND,
 * ::LWDA_ERROR_NO_BINARY_FOR_GPU,
 * ::LWDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
 * ::LWDA_ERROR_SHARED_OBJECT_INIT_FAILED,
 * ::LWDA_ERROR_JIT_COMPILER_NOT_FOUND
 * \notefnerr
 *
 * \sa ::lwModuleGetFunction,
 * ::lwModuleGetGlobal,
 * ::lwModuleGetTexRef,
 * ::lwModuleLoadData,
 * ::lwModuleLoadDataEx,
 * ::lwModuleLoadFatBinary,
 * ::lwModuleUnload
 */
LWresult LWDAAPI lwModuleLoad(LWmodule *module, const char *fname);

/**
 * \brief Load a module's data
 *
 * Takes a pointer \p image and loads the corresponding module \p module into
 * the current context. The pointer may be obtained by mapping a \e lwbin or
 * \e PTX or \e fatbin file, passing a \e lwbin or \e PTX or \e fatbin file
 * as a NULL-terminated text string, or incorporating a \e lwbin or \e fatbin
 * object into the exelwtable resources and using operating system calls such
 * as Windows \c FindResource() to obtain the pointer.
 *
 * \param module - Returned module
 * \param image  - Module data to load
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_PTX,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_NO_BINARY_FOR_GPU,
 * ::LWDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
 * ::LWDA_ERROR_SHARED_OBJECT_INIT_FAILED,
 * ::LWDA_ERROR_JIT_COMPILER_NOT_FOUND
 * \notefnerr
 *
 * \sa ::lwModuleGetFunction,
 * ::lwModuleGetGlobal,
 * ::lwModuleGetTexRef,
 * ::lwModuleLoad,
 * ::lwModuleLoadDataEx,
 * ::lwModuleLoadFatBinary,
 * ::lwModuleUnload
 */
LWresult LWDAAPI lwModuleLoadData(LWmodule *module, const void *image);

/**
 * \brief Load a module's data with options
 *
 * Takes a pointer \p image and loads the corresponding module \p module into
 * the current context. The pointer may be obtained by mapping a \e lwbin or
 * \e PTX or \e fatbin file, passing a \e lwbin or \e PTX or \e fatbin file
 * as a NULL-terminated text string, or incorporating a \e lwbin or \e fatbin
 * object into the exelwtable resources and using operating system calls such
 * as Windows \c FindResource() to obtain the pointer. Options are passed as
 * an array via \p options and any corresponding parameters are passed in
 * \p optiolwalues. The number of total options is supplied via \p numOptions.
 * Any outputs will be returned via \p optiolwalues. 
 *
 * \param module       - Returned module
 * \param image        - Module data to load
 * \param numOptions   - Number of options
 * \param options      - Options for JIT
 * \param optiolwalues - Option values for JIT
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_PTX,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_NO_BINARY_FOR_GPU,
 * ::LWDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
 * ::LWDA_ERROR_SHARED_OBJECT_INIT_FAILED,
 * ::LWDA_ERROR_JIT_COMPILER_NOT_FOUND
 * \notefnerr
 *
 * \sa ::lwModuleGetFunction,
 * ::lwModuleGetGlobal,
 * ::lwModuleGetTexRef,
 * ::lwModuleLoad,
 * ::lwModuleLoadData,
 * ::lwModuleLoadFatBinary,
 * ::lwModuleUnload
 */
LWresult LWDAAPI lwModuleLoadDataEx(LWmodule *module, const void *image, unsigned int numOptions, LWjit_option *options, void **optiolwalues);

/**
 * \brief Load a module's data
 *
 * Takes a pointer \p fatLwbin and loads the corresponding module \p module
 * into the current context. The pointer represents a <i>fat binary</i> object,
 * which is a collection of different \e lwbin and/or \e PTX files, all
 * representing the same device code, but compiled and optimized for different
 * architectures.
 *
 * Prior to LWCA 4.0, there was no dolwmented API for constructing and using
 * fat binary objects by programmers.  Starting with LWCA 4.0, fat binary
 * objects can be constructed by providing the <i>-fatbin option</i> to \b lwcc.
 * More information can be found in the \b lwcc document.
 *
 * \param module   - Returned module
 * \param fatLwbin - Fat binary to load
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_PTX,
 * ::LWDA_ERROR_NOT_FOUND,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_NO_BINARY_FOR_GPU,
 * ::LWDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
 * ::LWDA_ERROR_SHARED_OBJECT_INIT_FAILED,
 * ::LWDA_ERROR_JIT_COMPILER_NOT_FOUND
 * \notefnerr
 *
 * \sa ::lwModuleGetFunction,
 * ::lwModuleGetGlobal,
 * ::lwModuleGetTexRef,
 * ::lwModuleLoad,
 * ::lwModuleLoadData,
 * ::lwModuleLoadDataEx,
 * ::lwModuleUnload
 */
LWresult LWDAAPI lwModuleLoadFatBinary(LWmodule *module, const void *fatLwbin);

/**
 * \brief Unloads a module
 *
 * Unloads a module \p hmod from the current context.
 *
 * \param hmod - Module to unload
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwModuleGetFunction,
 * ::lwModuleGetGlobal,
 * ::lwModuleGetTexRef,
 * ::lwModuleLoad,
 * ::lwModuleLoadData,
 * ::lwModuleLoadDataEx,
 * ::lwModuleLoadFatBinary
 */
LWresult LWDAAPI lwModuleUnload(LWmodule hmod);

/**
 * \brief Returns a function handle
 *
 * Returns in \p *hfunc the handle of the function of name \p name located in
 * module \p hmod. If no function of that name exists, ::lwModuleGetFunction()
 * returns ::LWDA_ERROR_NOT_FOUND.
 *
 * \param hfunc - Returned function handle
 * \param hmod  - Module to retrieve function from
 * \param name  - Name of function to retrieve
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_NOT_FOUND
 * \notefnerr
 *
 * \sa ::lwModuleGetGlobal,
 * ::lwModuleGetTexRef,
 * ::lwModuleLoad,
 * ::lwModuleLoadData,
 * ::lwModuleLoadDataEx,
 * ::lwModuleLoadFatBinary,
 * ::lwModuleUnload
 */
LWresult LWDAAPI lwModuleGetFunction(LWfunction *hfunc, LWmodule hmod, const char *name);

#if __LWDA_API_VERSION >= 3020
/**
 * \brief Returns a global pointer from a module
 *
 * Returns in \p *dptr and \p *bytes the base pointer and size of the
 * global of name \p name located in module \p hmod. If no variable of that name
 * exists, ::lwModuleGetGlobal() returns ::LWDA_ERROR_NOT_FOUND. Both
 * parameters \p dptr and \p bytes are optional. If one of them is
 * NULL, it is ignored.
 *
 * \param dptr  - Returned global device pointer
 * \param bytes - Returned global size in bytes
 * \param hmod  - Module to retrieve global from
 * \param name  - Name of global to retrieve
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_NOT_FOUND
 * \notefnerr
 *
 * \sa ::lwModuleGetFunction,
 * ::lwModuleGetTexRef,
 * ::lwModuleLoad,
 * ::lwModuleLoadData,
 * ::lwModuleLoadDataEx,
 * ::lwModuleLoadFatBinary,
 * ::lwModuleUnload,
 * ::lwdaGetSymbolAddress,
 * ::lwdaGetSymbolSize
 */
LWresult LWDAAPI lwModuleGetGlobal(LWdeviceptr *dptr, size_t *bytes, LWmodule hmod, const char *name);
#endif /* __LWDA_API_VERSION >= 3020 */

/**
 * \brief Returns a handle to a texture reference
 *
 * Returns in \p *pTexRef the handle of the texture reference of name \p name
 * in the module \p hmod. If no texture reference of that name exists,
 * ::lwModuleGetTexRef() returns ::LWDA_ERROR_NOT_FOUND. This texture reference
 * handle should not be destroyed, since it will be destroyed when the module
 * is unloaded.
 *
 * \param pTexRef  - Returned texture reference
 * \param hmod     - Module to retrieve texture reference from
 * \param name     - Name of texture reference to retrieve
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_NOT_FOUND
 * \notefnerr
 *
 * \sa ::lwModuleGetFunction,
 * ::lwModuleGetGlobal,
 * ::lwModuleGetSurfRef,
 * ::lwModuleLoad,
 * ::lwModuleLoadData,
 * ::lwModuleLoadDataEx,
 * ::lwModuleLoadFatBinary,
 * ::lwModuleUnload,
 * ::lwdaGetTextureReference
 */
LWresult LWDAAPI lwModuleGetTexRef(LWtexref *pTexRef, LWmodule hmod, const char *name);

/**
 * \brief Returns a handle to a surface reference
 *
 * Returns in \p *pSurfRef the handle of the surface reference of name \p name
 * in the module \p hmod. If no surface reference of that name exists,
 * ::lwModuleGetSurfRef() returns ::LWDA_ERROR_NOT_FOUND.
 *
 * \param pSurfRef  - Returned surface reference
 * \param hmod     - Module to retrieve surface reference from
 * \param name     - Name of surface reference to retrieve
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_NOT_FOUND
 * \notefnerr
 *
 * \sa ::lwModuleGetFunction,
 * ::lwModuleGetGlobal,
 * ::lwModuleGetTexRef,
 * ::lwModuleLoad,
 * ::lwModuleLoadData,
 * ::lwModuleLoadDataEx,
 * ::lwModuleLoadFatBinary,
 * ::lwModuleUnload,
 * ::lwdaGetSurfaceReference
 */
LWresult LWDAAPI lwModuleGetSurfRef(LWsurfref *pSurfRef, LWmodule hmod, const char *name);

#if __LWDA_API_VERSION >= 5050

/**
 * \brief Creates a pending JIT linker invocation.
 *
 * If the call is successful, the caller owns the returned LWlinkState, which
 * should eventually be destroyed with ::lwLinkDestroy.  The
 * device code machine size (32 or 64 bit) will match the calling application.
 *
 * Both linker and compiler options may be specified.  Compiler options will
 * be applied to inputs to this linker action which must be compiled from PTX.
 * The options ::LW_JIT_WALL_TIME,
 * ::LW_JIT_INFO_LOG_BUFFER_SIZE_BYTES, and ::LW_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
 * will accumulate data until the LWlinkState is destroyed.
 *
 * \p optiolwalues must remain valid for the life of the LWlinkState if output
 * options are used.  No other references to inputs are maintained after this
 * call returns.
 *
 * \param numOptions   Size of options arrays
 * \param options      Array of linker and compiler options
 * \param optiolwalues Array of option values, each cast to void *
 * \param stateOut     On success, this will contain a LWlinkState to specify
 *                     and complete this action
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_JIT_COMPILER_NOT_FOUND
 * \notefnerr
 *
 * \sa ::lwLinkAddData,
 * ::lwLinkAddFile,
 * ::lwLinkComplete,
 * ::lwLinkDestroy
 */
LWresult LWDAAPI
lwLinkCreate(unsigned int numOptions, LWjit_option *options, void **optiolwalues, LWlinkState *stateOut);

/**
 * \brief Add an input to a pending linker invocation
 *
 * Ownership of \p data is retained by the caller.  No reference is retained to any
 * inputs after this call returns.
 *
 * This method accepts only compiler options, which are used if the data must
 * be compiled from PTX, and does not accept any of
 * ::LW_JIT_WALL_TIME, ::LW_JIT_INFO_LOG_BUFFER, ::LW_JIT_ERROR_LOG_BUFFER,
 * ::LW_JIT_TARGET_FROM_LWCONTEXT, or ::LW_JIT_TARGET.
 *
 * \param state        A pending linker action.
 * \param type         The type of the input data.
 * \param data         The input data.  PTX must be NULL-terminated.
 * \param size         The length of the input data.
 * \param name         An optional name for this input in log messages.
 * \param numOptions   Size of options.
 * \param options      Options to be applied only for this input (overrides options from ::lwLinkCreate).
 * \param optiolwalues Array of option values, each cast to void *.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_IMAGE,
 * ::LWDA_ERROR_ILWALID_PTX,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_NO_BINARY_FOR_GPU
 *
 * \sa ::lwLinkCreate,
 * ::lwLinkAddFile,
 * ::lwLinkComplete,
 * ::lwLinkDestroy
 */
LWresult LWDAAPI
lwLinkAddData(LWlinkState state, LWjitInputType type, void *data, size_t size, const char *name,
    unsigned int numOptions, LWjit_option *options, void **optiolwalues);

/**
 * \brief Add a file input to a pending linker invocation
 *
 * No reference is retained to any inputs after this call returns.
 *
 * This method accepts only compiler options, which are used if the input
 * must be compiled from PTX, and does not accept any of
 * ::LW_JIT_WALL_TIME, ::LW_JIT_INFO_LOG_BUFFER, ::LW_JIT_ERROR_LOG_BUFFER,
 * ::LW_JIT_TARGET_FROM_LWCONTEXT, or ::LW_JIT_TARGET.
 *
 * This method is equivalent to ilwoking ::lwLinkAddData on the contents
 * of the file.
 *
 * \param state        A pending linker action
 * \param type         The type of the input data
 * \param path         Path to the input file
 * \param numOptions   Size of options
 * \param options      Options to be applied only for this input (overrides options from ::lwLinkCreate)
 * \param optiolwalues Array of option values, each cast to void *
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_FILE_NOT_FOUND
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_IMAGE,
 * ::LWDA_ERROR_ILWALID_PTX,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_NO_BINARY_FOR_GPU
 *
 * \sa ::lwLinkCreate,
 * ::lwLinkAddData,
 * ::lwLinkComplete,
 * ::lwLinkDestroy
 */
LWresult LWDAAPI
lwLinkAddFile(LWlinkState state, LWjitInputType type, const char *path,
    unsigned int numOptions, LWjit_option *options, void **optiolwalues);

/**
 * \brief Complete a pending linker invocation
 *
 * Completes the pending linker action and returns the lwbin image for the linked
 * device code, which can be used with ::lwModuleLoadData.  The lwbin is owned by
 * \p state, so it should be loaded before \p state is destroyed via ::lwLinkDestroy.
 * This call does not destroy \p state.
 *
 * \param state    A pending linker invocation
 * \param lwbinOut On success, this will point to the output image
 * \param sizeOut  Optional parameter to receive the size of the generated image
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_OUT_OF_MEMORY
 *
 * \sa ::lwLinkCreate,
 * ::lwLinkAddData,
 * ::lwLinkAddFile,
 * ::lwLinkDestroy,
 * ::lwModuleLoadData
 */
LWresult LWDAAPI
lwLinkComplete(LWlinkState state, void **lwbinOut, size_t *sizeOut);

/**
 * \brief Destroys state for a JIT linker invocation.
 *
 * \param state State object for the linker invocation
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE
 *
 * \sa ::lwLinkCreate
 */
LWresult LWDAAPI
lwLinkDestroy(LWlinkState state);

#endif /* __LWDA_API_VERSION >= 5050 */

/** @} */ /* END LWDA_MODULE */


/**
 * \defgroup LWDA_MEM Memory Management
 *
 * ___MANBRIEF___ memory management functions of the low-level LWCA driver API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the memory management functions of the low-level LWCA
 * driver application programming interface.
 *
 * @{
 */

#if __LWDA_API_VERSION >= 3020
/**
 * \brief Gets free and total memory
 *
 * Returns in \p *free and \p *total respectively, the free and total amount of
 * memory available for allocation by the LWCA context, in bytes.
 *
 * \param free  - Returned free memory in bytes
 * \param total - Returned total memory in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMemGetInfo
 */
LWresult LWDAAPI lwMemGetInfo(size_t *free, size_t *total);

/**
 * \brief Allocates device memory
 *
 * Allocates \p bytesize bytes of linear memory on the device and returns in
 * \p *dptr a pointer to the allocated memory. The allocated memory is suitably
 * aligned for any kind of variable. The memory is not cleared. If \p bytesize
 * is 0, ::lwMemAlloc() returns ::LWDA_ERROR_ILWALID_VALUE.
 *
 * \param dptr     - Returned device pointer
 * \param bytesize - Requested allocation size in bytes
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
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMalloc
 */
LWresult LWDAAPI lwMemAlloc(LWdeviceptr *dptr, size_t bytesize);

/**
 * \brief Allocates pitched device memory
 *
 * Allocates at least \p WidthInBytes * \p Height bytes of linear memory on
 * the device and returns in \p *dptr a pointer to the allocated memory. The
 * function may pad the allocation to ensure that corresponding pointers in
 * any given row will continue to meet the alignment requirements for
 * coalescing as the address is updated from row to row. \p ElementSizeBytes
 * specifies the size of the largest reads and writes that will be performed
 * on the memory range. \p ElementSizeBytes may be 4, 8 or 16 (since coalesced
 * memory transactions are not possible on other data sizes). If
 * \p ElementSizeBytes is smaller than the actual read/write size of a kernel,
 * the kernel will run correctly, but possibly at reduced speed. The pitch
 * returned in \p *pPitch by ::lwMemAllocPitch() is the width in bytes of the
 * allocation. The intended usage of pitch is as a separate parameter of the
 * allocation, used to compute addresses within the 2D array. Given the row
 * and column of an array element of type \b T, the address is computed as:
 * \code
   T* pElement = (T*)((char*)BaseAddress + Row * Pitch) + Column;
 * \endcode
 *
 * The pitch returned by ::lwMemAllocPitch() is guaranteed to work with
 * ::lwMemcpy2D() under all cirlwmstances. For allocations of 2D arrays, it is
 * recommended that programmers consider performing pitch allocations using
 * ::lwMemAllocPitch(). Due to alignment restrictions in the hardware, this is
 * especially true if the application will be performing 2D memory copies
 * between different regions of device memory (whether linear memory or LWCA
 * arrays).
 *
 * The byte alignment of the pitch returned by ::lwMemAllocPitch() is guaranteed
 * to match or exceed the alignment requirement for texture binding with
 * ::lwTexRefSetAddress2D().
 *
 * \param dptr             - Returned device pointer
 * \param pPitch           - Returned pitch of allocation in bytes
 * \param WidthInBytes     - Requested allocation width in bytes
 * \param Height           - Requested allocation height in rows
 * \param ElementSizeBytes - Size of largest reads/writes for range
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
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMallocPitch
 */
LWresult LWDAAPI lwMemAllocPitch(LWdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);

/**
 * \brief Frees device memory
 *
 * Frees the memory space pointed to by \p dptr, which must have been returned
 * by a previous call to ::lwMemAlloc() or ::lwMemAllocPitch().
 *
 * \param dptr - Pointer to memory to free
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaFree
 */
LWresult LWDAAPI lwMemFree(LWdeviceptr dptr);

/**
 * \brief Get information on memory allocations
 *
 * Returns the base address in \p *pbase and size in \p *psize of the
 * allocation by ::lwMemAlloc() or ::lwMemAllocPitch() that contains the input
 * pointer \p dptr. Both parameters \p pbase and \p psize are optional. If one
 * of them is NULL, it is ignored.
 *
 * \param pbase - Returned base address
 * \param psize - Returned size of device memory allocation
 * \param dptr  - Device pointer to query
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_NOT_FOUND,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32
 */
LWresult LWDAAPI lwMemGetAddressRange(LWdeviceptr *pbase, size_t *psize, LWdeviceptr dptr);

/**
 * \brief Allocates page-locked host memory
 *
 * Allocates \p bytesize bytes of host memory that is page-locked and
 * accessible to the device. The driver tracks the virtual memory ranges
 * allocated with this function and automatically accelerates calls to
 * functions such as ::lwMemcpy(). Since the memory can be accessed directly by
 * the device, it can be read or written with much higher bandwidth than
 * pageable memory obtained with functions such as ::malloc(). Allocating
 * excessive amounts of memory with ::lwMemAllocHost() may degrade system
 * performance, since it reduces the amount of memory available to the system
 * for paging. As a result, this function is best used sparingly to allocate
 * staging areas for data exchange between host and device.
 *
 * Note all host memory allocated using ::lwMemHostAlloc() will automatically
 * be immediately accessible to all contexts on all devices which support unified
 * addressing (as may be queried using ::LW_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING).
 * The device pointer that may be used to access this host memory from those 
 * contexts is always equal to the returned host pointer \p *pp.
 * See \ref LWDA_UNIFIED for additional details.
 *
 * \param pp       - Returned host pointer to page-locked memory
 * \param bytesize - Requested allocation size in bytes
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
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMallocHost
 */
LWresult LWDAAPI lwMemAllocHost(void **pp, size_t bytesize);
#endif /* __LWDA_API_VERSION >= 3020 */

/**
 * \brief Frees page-locked host memory
 *
 * Frees the memory space pointed to by \p p, which must have been returned by
 * a previous call to ::lwMemAllocHost().
 *
 * \param p - Pointer to memory to free
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaFreeHost
 */
LWresult LWDAAPI lwMemFreeHost(void *p);

/**
 * \brief Allocates page-locked host memory
 *
 * Allocates \p bytesize bytes of host memory that is page-locked and accessible
 * to the device. The driver tracks the virtual memory ranges allocated with
 * this function and automatically accelerates calls to functions such as
 * ::lwMemcpyHtoD(). Since the memory can be accessed directly by the device,
 * it can be read or written with much higher bandwidth than pageable memory
 * obtained with functions such as ::malloc(). Allocating excessive amounts of
 * pinned memory may degrade system performance, since it reduces the amount
 * of memory available to the system for paging. As a result, this function is
 * best used sparingly to allocate staging areas for data exchange between
 * host and device.
 *
 * The \p Flags parameter enables different options to be specified that
 * affect the allocation, as follows.
 *
 * - ::LW_MEMHOSTALLOC_PORTABLE: The memory returned by this call will be
 *   considered as pinned memory by all LWCA contexts, not just the one that
 *   performed the allocation.
 *
 * - ::LW_MEMHOSTALLOC_DEVICEMAP: Maps the allocation into the LWCA address
 *   space. The device pointer to the memory may be obtained by calling
 *   ::lwMemHostGetDevicePointer().
 *
 * - ::LW_MEMHOSTALLOC_WRITECOMBINED: Allocates the memory as write-combined
 *   (WC). WC memory can be transferred across the PCI Express bus more
 *   quickly on some system configurations, but cannot be read efficiently by
 *   most CPUs. WC memory is a good option for buffers that will be written by
 *   the CPU and read by the GPU via mapped pinned memory or host->device
 *   transfers.
 *
 * All of these flags are orthogonal to one another: a developer may allocate
 * memory that is portable, mapped and/or write-combined with no restrictions.
 *
 * The LWCA context must have been created with the ::LW_CTX_MAP_HOST flag in
 * order for the ::LW_MEMHOSTALLOC_DEVICEMAP flag to have any effect.
 *
 * The ::LW_MEMHOSTALLOC_DEVICEMAP flag may be specified on LWCA contexts for
 * devices that do not support mapped pinned memory. The failure is deferred
 * to ::lwMemHostGetDevicePointer() because the memory may be mapped into
 * other LWCA contexts via the ::LW_MEMHOSTALLOC_PORTABLE flag.
 *
 * The memory allocated by this function must be freed with ::lwMemFreeHost().
 *
 * Note all host memory allocated using ::lwMemHostAlloc() will automatically
 * be immediately accessible to all contexts on all devices which support unified
 * addressing (as may be queried using ::LW_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING).
 * Unless the flag ::LW_MEMHOSTALLOC_WRITECOMBINED is specified, the device pointer 
 * that may be used to access this host memory from those contexts is always equal 
 * to the returned host pointer \p *pp.  If the flag ::LW_MEMHOSTALLOC_WRITECOMBINED
 * is specified, then the function ::lwMemHostGetDevicePointer() must be used
 * to query the device pointer, even if the context supports unified addressing.
 * See \ref LWDA_UNIFIED for additional details.
 *
 * \param pp       - Returned host pointer to page-locked memory
 * \param bytesize - Requested allocation size in bytes
 * \param Flags    - Flags for allocation request
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
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaHostAlloc
 */
LWresult LWDAAPI lwMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags);

#if __LWDA_API_VERSION >= 3020
/**
 * \brief Passes back device pointer of mapped pinned memory
 *
 * Passes back the device pointer \p pdptr corresponding to the mapped, pinned
 * host buffer \p p allocated by ::lwMemHostAlloc.
 *
 * ::lwMemHostGetDevicePointer() will fail if the ::LW_MEMHOSTALLOC_DEVICEMAP
 * flag was not specified at the time the memory was allocated, or if the
 * function is called on a GPU that does not support mapped pinned memory.
 *
 * For devices that have a non-zero value for the device attribute
 * ::LW_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, the memory
 * can also be accessed from the device using the host pointer \p p.
 * The device pointer returned by ::lwMemHostGetDevicePointer() may or may not
 * match the original host pointer \p p and depends on the devices visible to the
 * application. If all devices visible to the application have a non-zero value for the
 * device attribute, the device pointer returned by ::lwMemHostGetDevicePointer()
 * will match the original pointer \p p. If any device visible to the application
 * has a zero value for the device attribute, the device pointer returned by
 * ::lwMemHostGetDevicePointer() will not match the original host pointer \p p,
 * but it will be suitable for use on all devices provided Unified Virtual Addressing
 * is enabled. In such systems, it is valid to access the memory using either pointer
 * on devices that have a non-zero value for the device attribute. Note however that
 * such devices should access the memory using only of the two pointers and not both.
 *
 * \p Flags provides for future releases. For now, it must be set to 0.
 *
 * \param pdptr - Returned device pointer
 * \param p     - Host pointer
 * \param Flags - Options (must be 0)
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaHostGetDevicePointer
 */
LWresult LWDAAPI lwMemHostGetDevicePointer(LWdeviceptr *pdptr, void *p, unsigned int Flags);
#endif /* __LWDA_API_VERSION >= 3020 */

/**
 * \brief Passes back flags that were used for a pinned allocation
 *
 * Passes back the flags \p pFlags that were specified when allocating
 * the pinned host buffer \p p allocated by ::lwMemHostAlloc.
 *
 * ::lwMemHostGetFlags() will fail if the pointer does not reside in
 * an allocation performed by ::lwMemAllocHost() or ::lwMemHostAlloc().
 *
 * \param pFlags - Returned flags word
 * \param p     - Host pointer
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa
 * ::lwMemAllocHost,
 * ::lwMemHostAlloc,
 * ::lwdaHostGetFlags
 */
LWresult LWDAAPI lwMemHostGetFlags(unsigned int *pFlags, void *p);

#if __LWDA_API_VERSION >= 6000

/**
 * \brief Allocates memory that will be automatically managed by the Unified Memory system
 *
 * Allocates \p bytesize bytes of managed memory on the device and returns in
 * \p *dptr a pointer to the allocated memory. If the device doesn't support
 * allocating managed memory, ::LWDA_ERROR_NOT_SUPPORTED is returned. Support
 * for managed memory can be queried using the device attribute
 * ::LW_DEVICE_ATTRIBUTE_MANAGED_MEMORY. The allocated memory is suitably
 * aligned for any kind of variable. The memory is not cleared. If \p bytesize
 * is 0, ::lwMemAllocManaged returns ::LWDA_ERROR_ILWALID_VALUE. The pointer
 * is valid on the CPU and on all GPUs in the system that support managed memory.
 * All accesses to this pointer must obey the Unified Memory programming model.
 *
 * \p flags specifies the default stream association for this allocation.
 * \p flags must be one of ::LW_MEM_ATTACH_GLOBAL or ::LW_MEM_ATTACH_HOST. If
 * ::LW_MEM_ATTACH_GLOBAL is specified, then this memory is accessible from
 * any stream on any device. If ::LW_MEM_ATTACH_HOST is specified, then the
 * allocation should not be accessed from devices that have a zero value for the
 * device attribute ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS; an explicit call to
 * ::lwStreamAttachMemAsync will be required to enable access on such devices.
 *
 * If the association is later changed via ::lwStreamAttachMemAsync to
 * a single stream, the default association as specifed during ::lwMemAllocManaged
 * is restored when that stream is destroyed. For __managed__ variables, the
 * default association is always ::LW_MEM_ATTACH_GLOBAL. Note that destroying a
 * stream is an asynchronous operation, and as a result, the change to default
 * association won't happen until all work in the stream has completed.
 *
 * Memory allocated with ::lwMemAllocManaged should be released with ::lwMemFree.
 *
 * Device memory oversubscription is possible for GPUs that have a non-zero value for the
 * device attribute ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS. Managed memory on
 * such GPUs may be evicted from device memory to host memory at any time by the Unified
 * Memory driver in order to make room for other allocations.
 *
 * In a multi-GPU system where all GPUs have a non-zero value for the device attribute
 * ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS, managed memory may not be populated when this
 * API returns and instead may be populated on access. In such systems, managed memory can
 * migrate to any processor's memory at any time. The Unified Memory driver will employ heuristics to
 * maintain data locality and prevent excessive page faults to the extent possible. The application
 * can also guide the driver about memory usage patterns via ::lwMemAdvise. The application
 * can also explicitly migrate memory to a desired processor's memory via
 * ::lwMemPrefetchAsync.
 *
 * In a multi-GPU system where all of the GPUs have a zero value for the device attribute
 * ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS and all the GPUs have peer-to-peer support
 * with each other, the physical storage for managed memory is created on the GPU which is active
 * at the time ::lwMemAllocManaged is called. All other GPUs will reference the data at reduced
 * bandwidth via peer mappings over the PCIe bus. The Unified Memory driver does not migrate
 * memory among such GPUs.
 *
 * In a multi-GPU system where not all GPUs have peer-to-peer support with each other and
 * where the value of the device attribute ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS
 * is zero for at least one of those GPUs, the location chosen for physical storage of managed
 * memory is system-dependent.
 * - On Linux, the location chosen will be device memory as long as the current set of active
 * contexts are on devices that either have peer-to-peer support with each other or have a
 * non-zero value for the device attribute ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS.
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
 * Alternatively, users can also set LWDA_MANAGED_FORCE_DEVICE_ALLOC to a
 * non-zero value to force the driver to always use device memory for physical storage.
 * When this environment variable is set to a non-zero value, all contexts created in
 * that process on devices that support managed memory have to be peer-to-peer compatible
 * with each other. Context creation will fail if a context is created on a device that
 * supports managed memory and is not peer-to-peer compatible with any of the other
 * managed memory supporting devices on which contexts were previously created, even if
 * those contexts have been destroyed. These environment variables are described
 * in the LWCA programming guide under the "LWCA environment variables" section.
 * - On ARM, managed memory is not available on discrete gpu with Drive PX-2.
 *
 * \param dptr     - Returned device pointer
 * \param bytesize - Requested allocation size in bytes
 * \param flags    - Must be one of ::LW_MEM_ATTACH_GLOBAL or ::LW_MEM_ATTACH_HOST
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_NOT_SUPPORTED,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwDeviceGetAttribute, ::lwStreamAttachMemAsync,
 * ::lwdaMallocManaged
 */
LWresult LWDAAPI lwMemAllocManaged(LWdeviceptr *dptr, size_t bytesize, unsigned int flags);

#endif /* __LWDA_API_VERSION >= 6000 */

#if __LWDA_API_VERSION >= 4010

/**
 * \brief Returns a handle to a compute device
 *
 * Returns in \p *device a device handle given a PCI bus ID string.
 *
 * \param dev      - Returned device handle
 *
 * \param pciBusId - String in one of the following forms: 
 * [domain]:[bus]:[device].[function]
 * [domain]:[bus]:[device]
 * [bus]:[device].[function]
 * where \p domain, \p bus, \p device, and \p function are all hexadecimal values
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::lwDeviceGet,
 * ::lwDeviceGetAttribute,
 * ::lwDeviceGetPCIBusId,
 * ::lwdaDeviceGetByPCIBusId
 */
LWresult LWDAAPI lwDeviceGetByPCIBusId(LWdevice *dev, const char *pciBusId);

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
 * \param dev      - Device to get identifier string for
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::lwDeviceGet,
 * ::lwDeviceGetAttribute,
 * ::lwDeviceGetByPCIBusId,
 * ::lwdaDeviceGetPCIBusId
 */
LWresult LWDAAPI lwDeviceGetPCIBusId(char *pciBusId, int len, LWdevice dev);

/**
 * \brief Gets an interprocess handle for a previously allocated event
 *
 * Takes as input a previously allocated event. This event must have been 
 * created with the ::LW_EVENT_INTERPROCESS and ::LW_EVENT_DISABLE_TIMING 
 * flags set. This opaque handle may be copied into other processes and
 * opened with ::lwIpcOpenEventHandle to allow efficient hardware
 * synchronization between GPU work in different processes.
 *
 * After the event has been opened in the importing process, 
 * ::lwEventRecord, ::lwEventSynchronize, ::lwStreamWaitEvent and 
 * ::lwEventQuery may be used in either process. Performing operations 
 * on the imported event after the exported event has been freed 
 * with ::lwEventDestroy will result in undefined behavior.
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems.
 *
 * \param pHandle - Pointer to a user allocated LWipcEventHandle
 *                    in which to return the opaque event handle
 * \param event   - Event allocated with ::LW_EVENT_INTERPROCESS and 
 *                    ::LW_EVENT_DISABLE_TIMING flags.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_MAP_FAILED
 *
 * \sa 
 * ::lwEventCreate, 
 * ::lwEventDestroy, 
 * ::lwEventSynchronize,
 * ::lwEventQuery,
 * ::lwStreamWaitEvent,
 * ::lwIpcOpenEventHandle,
 * ::lwIpcGetMemHandle,
 * ::lwIpcOpenMemHandle,
 * ::lwIpcCloseMemHandle,
 * ::lwdaIpcGetEventHandle
 */
LWresult LWDAAPI lwIpcGetEventHandle(LWipcEventHandle *pHandle, LWevent event);

/**
 * \brief Opens an interprocess event handle for use in the current process
 *
 * Opens an interprocess event handle exported from another process with 
 * ::lwIpcGetEventHandle. This function returns a ::LWevent that behaves like 
 * a locally created event with the ::LW_EVENT_DISABLE_TIMING flag specified. 
 * This event must be freed with ::lwEventDestroy.
 *
 * Performing operations on the imported event after the exported event has 
 * been freed with ::lwEventDestroy will result in undefined behavior.
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems.
 *
 * \param phEvent - Returns the imported event
 * \param handle  - Interprocess handle to open
 *
 * \returns
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_MAP_FAILED,
 * ::LWDA_ERROR_PEER_ACCESS_UNSUPPORTED,
 * ::LWDA_ERROR_ILWALID_HANDLE
 *
 * \sa
 * ::lwEventCreate, 
 * ::lwEventDestroy, 
 * ::lwEventSynchronize,
 * ::lwEventQuery,
 * ::lwStreamWaitEvent,
 * ::lwIpcGetEventHandle,
 * ::lwIpcGetMemHandle,
 * ::lwIpcOpenMemHandle,
 * ::lwIpcCloseMemHandle,
 * ::lwdaIpcOpenEventHandle
 */
LWresult LWDAAPI lwIpcOpenEventHandle(LWevent *phEvent, LWipcEventHandle handle);

/**
 * \brief Gets an interprocess memory handle for an existing device memory
 * allocation
 *
 * Takes a pointer to the base of an existing device memory allocation created 
 * with ::lwMemAlloc and exports it for use in another process. This is a 
 * lightweight operation and may be called multiple times on an allocation
 * without adverse effects. 
 *
 * If a region of memory is freed with ::lwMemFree and a subsequent call
 * to ::lwMemAlloc returns memory with the same device address,
 * ::lwIpcGetMemHandle will return a unique handle for the
 * new memory. 
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems.
 *
 * \param pHandle - Pointer to user allocated ::LWipcMemHandle to return
 *                    the handle in.
 * \param dptr    - Base pointer to previously allocated device memory 
 *
 * \returns
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_MAP_FAILED,
 * 
 * \sa
 * ::lwMemAlloc,
 * ::lwMemFree,
 * ::lwIpcGetEventHandle,
 * ::lwIpcOpenEventHandle,
 * ::lwIpcOpenMemHandle,
 * ::lwIpcCloseMemHandle,
 * ::lwdaIpcGetMemHandle
 */
LWresult LWDAAPI lwIpcGetMemHandle(LWipcMemHandle *pHandle, LWdeviceptr dptr);

/**
 * \brief Opens an interprocess memory handle exported from another process
 * and returns a device pointer usable in the local process.
 *
 * Maps memory exported from another process with ::lwIpcGetMemHandle into
 * the current device address space. For contexts on different devices 
 * ::lwIpcOpenMemHandle can attempt to enable peer access between the
 * devices as if the user called ::lwCtxEnablePeerAccess. This behavior is 
 * controlled by the ::LW_IPC_MEM_LAZY_ENABLE_PEER_ACCESS flag. 
 * ::lwDeviceCanAccessPeer can determine if a mapping is possible.
 *
 * Contexts that may open ::LWipcMemHandles are restricted in the following way.
 * ::LWipcMemHandles from each ::LWdevice in a given process may only be opened 
 * by one ::LWcontext per ::LWdevice per other process.
 *
 * Memory returned from ::lwIpcOpenMemHandle must be freed with
 * ::lwIpcCloseMemHandle.
 *
 * Calling ::lwMemFree on an exported memory region before calling
 * ::lwIpcCloseMemHandle in the importing context will result in undefined
 * behavior.
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems.
 * 
 * \param pdptr  - Returned device pointer
 * \param handle - ::LWipcMemHandle to open
 * \param Flags  - Flags for this operation. Must be specified as ::LW_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
 *
 * \returns
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_MAP_FAILED,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_TOO_MANY_PEERS
 *
 * \note No guarantees are made about the address returned in \p *pdptr.  
 * In particular, multiple processes may not receive the same address for the same \p handle.
 *
 * \sa
 * ::lwMemAlloc,
 * ::lwMemFree,
 * ::lwIpcGetEventHandle,
 * ::lwIpcOpenEventHandle,
 * ::lwIpcGetMemHandle,
 * ::lwIpcCloseMemHandle,
 * ::lwCtxEnablePeerAccess,
 * ::lwDeviceCanAccessPeer,
 * ::lwdaIpcOpenMemHandle
 */
LWresult LWDAAPI lwIpcOpenMemHandle(LWdeviceptr *pdptr, LWipcMemHandle handle, unsigned int Flags);

/**
 * \brief Close memory mapped with ::lwIpcOpenMemHandle
 * 
 * Unmaps memory returnd by ::lwIpcOpenMemHandle. The original allocation
 * in the exporting process as well as imported mappings in other processes
 * will be unaffected.
 *
 * Any resources used to enable peer access will be freed if this is the
 * last mapping using them.
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems.
 *
 * \param dptr - Device pointer returned by ::lwIpcOpenMemHandle
 * 
 * \returns
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_MAP_FAILED,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 *
 * \sa
 * ::lwMemAlloc,
 * ::lwMemFree,
 * ::lwIpcGetEventHandle,
 * ::lwIpcOpenEventHandle,
 * ::lwIpcGetMemHandle,
 * ::lwIpcOpenMemHandle,
 * ::lwdaIpcCloseMemHandle
 */
LWresult LWDAAPI lwIpcCloseMemHandle(LWdeviceptr dptr);

#endif /* __LWDA_API_VERSION >= 4010 */

#if __LWDA_API_VERSION >= 4000
/**
 * \brief Registers an existing host memory range for use by LWCA
 *
 * Page-locks the memory range specified by \p p and \p bytesize and maps it
 * for the device(s) as specified by \p Flags. This memory range also is added
 * to the same tracking mechanism as ::lwMemHostAlloc to automatically accelerate
 * calls to functions such as ::lwMemcpyHtoD(). Since the memory can be accessed 
 * directly by the device, it can be read or written with much higher bandwidth 
 * than pageable memory that has not been registered.  Page-locking excessive
 * amounts of memory may degrade system performance, since it reduces the amount
 * of memory available to the system for paging. As a result, this function is
 * best used sparingly to register staging areas for data exchange between
 * host and device.
 *
 * This function has limited support on Mac OS X. OS 10.7 or higher is required.
 *
 * The \p Flags parameter enables different options to be specified that
 * affect the allocation, as follows.
 *
 * - ::LW_MEMHOSTREGISTER_PORTABLE: The memory returned by this call will be
 *   considered as pinned memory by all LWCA contexts, not just the one that
 *   performed the allocation.
 *
 * - ::LW_MEMHOSTREGISTER_DEVICEMAP: Maps the allocation into the LWCA address
 *   space. The device pointer to the memory may be obtained by calling
 *   ::lwMemHostGetDevicePointer().
 *
 * - ::LW_MEMHOSTREGISTER_IOMEMORY: The pointer is treated as pointing to some
 *   I/O memory space, e.g. the PCI Express resource of a 3rd party device.
 *
 * All of these flags are orthogonal to one another: a developer may page-lock
 * memory that is portable or mapped with no restrictions.
 *
 * The LWCA context must have been created with the ::LW_CTX_MAP_HOST flag in
 * order for the ::LW_MEMHOSTREGISTER_DEVICEMAP flag to have any effect.
 *
 * The ::LW_MEMHOSTREGISTER_DEVICEMAP flag may be specified on LWCA contexts for
 * devices that do not support mapped pinned memory. The failure is deferred
 * to ::lwMemHostGetDevicePointer() because the memory may be mapped into
 * other LWCA contexts via the ::LW_MEMHOSTREGISTER_PORTABLE flag.
 *
 * For devices that have a non-zero value for the device attribute
 * ::LW_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, the memory
 * can also be accessed from the device using the host pointer \p p.
 * The device pointer returned by ::lwMemHostGetDevicePointer() may or may not
 * match the original host pointer \p ptr and depends on the devices visible to the
 * application. If all devices visible to the application have a non-zero value for the
 * device attribute, the device pointer returned by ::lwMemHostGetDevicePointer()
 * will match the original pointer \p ptr. If any device visible to the application
 * has a zero value for the device attribute, the device pointer returned by
 * ::lwMemHostGetDevicePointer() will not match the original host pointer \p ptr,
 * but it will be suitable for use on all devices provided Unified Virtual Addressing
 * is enabled. In such systems, it is valid to access the memory using either pointer
 * on devices that have a non-zero value for the device attribute. Note however that
 * such devices should access the memory using only of the two pointers and not both.
 *
 * The memory page-locked by this function must be unregistered with 
 * ::lwMemHostUnregister().
 *
 * \param p        - Host pointer to memory to page-lock
 * \param bytesize - Size in bytes of the address range to page-lock
 * \param Flags    - Flags for allocation request
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
 * ::LWDA_ERROR_NOT_PERMITTED,
 * ::LWDA_ERROR_NOT_SUPPORTED
 * \notefnerr
 *
 * \sa
 * ::lwMemHostUnregister,
 * ::lwMemHostGetFlags,
 * ::lwMemHostGetDevicePointer,
 * ::lwdaHostRegister
 */
LWresult LWDAAPI lwMemHostRegister(void *p, size_t bytesize, unsigned int Flags);

/**
 * \brief Unregisters a memory range that was registered with lwMemHostRegister.
 *
 * Unmaps the memory range whose base address is specified by \p p, and makes
 * it pageable again.
 *
 * The base address must be the same one specified to ::lwMemHostRegister().
 *
 * \param p - Host pointer to memory to unregister
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
 * \notefnerr
 *
 * \sa
 * ::lwMemHostRegister,
 * ::lwdaHostUnregister
 */
LWresult LWDAAPI lwMemHostUnregister(void *p);

/**
 * \brief Copies memory
 *
 * Copies data between two pointers. 
 * \p dst and \p src are base pointers of the destination and source, respectively.  
 * \p ByteCount specifies the number of bytes to copy.
 * Note that this function infers the type of the transfer (host to host, host to 
 *   device, device to device, or device to host) from the pointer values.  This
 *   function is only allowed in contexts which support unified addressing.
 *
 * \param dst - Destination unified virtual address space pointer
 * \param src - Source unified virtual address space pointer
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMemcpy,
 * ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol
 */
LWresult LWDAAPI lwMemcpy(LWdeviceptr dst, LWdeviceptr src, size_t ByteCount);

/**
 * \brief Copies device memory between two contexts
 *
 * Copies from device memory in one context to device memory in another
 * context. \p dstDevice is the base device pointer of the destination memory 
 * and \p dstContext is the destination context.  \p srcDevice is the base 
 * device pointer of the source memory and \p srcContext is the source pointer.  
 * \p ByteCount specifies the number of bytes to copy.
 *
 * \param dstDevice  - Destination device pointer
 * \param dstContext - Destination context
 * \param srcDevice  - Source device pointer
 * \param srcContext - Source context
 * \param ByteCount  - Size of memory copy in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwMemcpyDtoD, ::lwMemcpy3DPeer, ::lwMemcpyDtoDAsync, ::lwMemcpyPeerAsync,
 * ::lwMemcpy3DPeerAsync,
 * ::lwdaMemcpyPeer
 */
LWresult LWDAAPI lwMemcpyPeer(LWdeviceptr dstDevice, LWcontext dstContext, LWdeviceptr srcDevice, LWcontext srcContext, size_t ByteCount);

#endif /* __LWDA_API_VERSION >= 4000 */

#if __LWDA_API_VERSION >= 3020
/**
 * \brief Copies memory from Host to Device
 *
 * Copies from host memory to device memory. \p dstDevice and \p srcHost are
 * the base addresses of the destination and source, respectively. \p ByteCount
 * specifies the number of bytes to copy.
 *
 * \param dstDevice - Destination device pointer
 * \param srcHost   - Source host pointer
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMemcpy,
 * ::lwdaMemcpyToSymbol
 */
LWresult LWDAAPI lwMemcpyHtoD(LWdeviceptr dstDevice, const void *srcHost, size_t ByteCount);

/**
 * \brief Copies memory from Device to Host
 *
 * Copies from device to host memory. \p dstHost and \p srcDevice specify the
 * base pointers of the destination and source, respectively. \p ByteCount
 * specifies the number of bytes to copy.
 *
 * \param dstHost   - Destination host pointer
 * \param srcDevice - Source device pointer
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMemcpy,
 * ::lwdaMemcpyFromSymbol
 */
LWresult LWDAAPI lwMemcpyDtoH(void *dstHost, LWdeviceptr srcDevice, size_t ByteCount);

/**
 * \brief Copies memory from Device to Device
 *
 * Copies from device memory to device memory. \p dstDevice and \p srcDevice
 * are the base pointers of the destination and source, respectively.
 * \p ByteCount specifies the number of bytes to copy.
 *
 * \param dstDevice - Destination device pointer
 * \param srcDevice - Source device pointer
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMemcpy,
 * ::lwdaMemcpyToSymbol,
 * ::lwdaMemcpyFromSymbol
 */
LWresult LWDAAPI lwMemcpyDtoD(LWdeviceptr dstDevice, LWdeviceptr srcDevice, size_t ByteCount);

/**
 * \brief Copies memory from Device to Array
 *
 * Copies from device memory to a 1D LWCA array. \p dstArray and \p dstOffset
 * specify the LWCA array handle and starting index of the destination data.
 * \p srcDevice specifies the base pointer of the source. \p ByteCount
 * specifies the number of bytes to copy.
 *
 * \param dstArray  - Destination array
 * \param dstOffset - Offset in bytes of destination array
 * \param srcDevice - Source device pointer
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMemcpyToArray
 */
LWresult LWDAAPI lwMemcpyDtoA(LWarray dstArray, size_t dstOffset, LWdeviceptr srcDevice, size_t ByteCount);

/**
 * \brief Copies memory from Array to Device
 *
 * Copies from one 1D LWCA array to device memory. \p dstDevice specifies the
 * base pointer of the destination and must be naturally aligned with the LWCA
 * array elements. \p srcArray and \p srcOffset specify the LWCA array handle
 * and the offset in bytes into the array where the copy is to begin.
 * \p ByteCount specifies the number of bytes to copy and must be evenly
 * divisible by the array element size.
 *
 * \param dstDevice - Destination device pointer
 * \param srcArray  - Source array
 * \param srcOffset - Offset in bytes of source array
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMemcpyFromArray
 */
LWresult LWDAAPI lwMemcpyAtoD(LWdeviceptr dstDevice, LWarray srcArray, size_t srcOffset, size_t ByteCount);

/**
 * \brief Copies memory from Host to Array
 *
 * Copies from host memory to a 1D LWCA array. \p dstArray and \p dstOffset
 * specify the LWCA array handle and starting offset in bytes of the destination
 * data.  \p pSrc specifies the base address of the source. \p ByteCount specifies
 * the number of bytes to copy.
 *
 * \param dstArray  - Destination array
 * \param dstOffset - Offset in bytes of destination array
 * \param srcHost   - Source host pointer
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMemcpyToArray
 */
LWresult LWDAAPI lwMemcpyHtoA(LWarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);

/**
 * \brief Copies memory from Array to Host
 *
 * Copies from one 1D LWCA array to host memory. \p dstHost specifies the base
 * pointer of the destination. \p srcArray and \p srcOffset specify the LWCA
 * array handle and starting offset in bytes of the source data.
 * \p ByteCount specifies the number of bytes to copy.
 *
 * \param dstHost   - Destination device pointer
 * \param srcArray  - Source array
 * \param srcOffset - Offset in bytes of source array
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMemcpyFromArray
 */
LWresult LWDAAPI lwMemcpyAtoH(void *dstHost, LWarray srcArray, size_t srcOffset, size_t ByteCount);

/**
 * \brief Copies memory from Array to Array
 *
 * Copies from one 1D LWCA array to another. \p dstArray and \p srcArray
 * specify the handles of the destination and source LWCA arrays for the copy,
 * respectively. \p dstOffset and \p srcOffset specify the destination and
 * source offsets in bytes into the LWCA arrays. \p ByteCount is the number of
 * bytes to be copied. The size of the elements in the LWCA arrays need not be
 * the same format, but the elements must be the same size; and count must be
 * evenly divisible by that size.
 *
 * \param dstArray  - Destination array
 * \param dstOffset - Offset in bytes of destination array
 * \param srcArray  - Source array
 * \param srcOffset - Offset in bytes of source array
 * \param ByteCount - Size of memory copy in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMemcpyArrayToArray
 */
LWresult LWDAAPI lwMemcpyAtoA(LWarray dstArray, size_t dstOffset, LWarray srcArray, size_t srcOffset, size_t ByteCount);

/**
 * \brief Copies memory for 2D arrays
 *
 * Perform a 2D memory copy according to the parameters specified in \p pCopy.
 * The ::LWDA_MEMCPY2D structure is defined as:
 *
 * \code
   typedef struct LWDA_MEMCPY2D_st {
      unsigned int srcXInBytes, srcY;
      LWmemorytype srcMemoryType;
          const void *srcHost;
          LWdeviceptr srcDevice;
          LWarray srcArray;
          unsigned int srcPitch;

      unsigned int dstXInBytes, dstY;
      LWmemorytype dstMemoryType;
          void *dstHost;
          LWdeviceptr dstDevice;
          LWarray dstArray;
          unsigned int dstPitch;

      unsigned int WidthInBytes;
      unsigned int Height;
   } LWDA_MEMCPY2D;
 * \endcode
 * where:
 * - ::srcMemoryType and ::dstMemoryType specify the type of memory of the
 *   source and destination, respectively; ::LWmemorytype_enum is defined as:
 *
 * \code
   typedef enum LWmemorytype_enum {
      LW_MEMORYTYPE_HOST = 0x01,
      LW_MEMORYTYPE_DEVICE = 0x02,
      LW_MEMORYTYPE_ARRAY = 0x03,
      LW_MEMORYTYPE_UNIFIED = 0x04
   } LWmemorytype;
 * \endcode
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_UNIFIED, ::srcDevice and ::srcPitch
 *   specify the (unified virtual address space) base address of the source data 
 *   and the bytes per row to apply.  ::srcArray is ignored.  
 * This value may be used only if unified addressing is supported in the calling 
 *   context.
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_HOST, ::srcHost and ::srcPitch
 * specify the (host) base address of the source data and the bytes per row to
 * apply. ::srcArray is ignored.
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_DEVICE, ::srcDevice and ::srcPitch
 * specify the (device) base address of the source data and the bytes per row
 * to apply. ::srcArray is ignored.
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_ARRAY, ::srcArray specifies the
 * handle of the source data. ::srcHost, ::srcDevice and ::srcPitch are
 * ignored.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_HOST, ::dstHost and ::dstPitch
 * specify the (host) base address of the destination data and the bytes per
 * row to apply. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_UNIFIED, ::dstDevice and ::dstPitch
 *   specify the (unified virtual address space) base address of the source data 
 *   and the bytes per row to apply.  ::dstArray is ignored.  
 * This value may be used only if unified addressing is supported in the calling 
 *   context.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_DEVICE, ::dstDevice and ::dstPitch
 * specify the (device) base address of the destination data and the bytes per
 * row to apply. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_ARRAY, ::dstArray specifies the
 * handle of the destination data. ::dstHost, ::dstDevice and ::dstPitch are
 * ignored.
 *
 * - ::srcXInBytes and ::srcY specify the base address of the source data for
 *   the copy.
 *
 * \par
 * For host pointers, the starting address is
 * \code
  void* Start = (void*)((char*)srcHost+srcY*srcPitch + srcXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  LWdeviceptr Start = srcDevice+srcY*srcPitch+srcXInBytes;
 * \endcode
 *
 * \par
 * For LWCA arrays, ::srcXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::dstXInBytes and ::dstY specify the base address of the destination data
 *   for the copy.
 *
 * \par
 * For host pointers, the base address is
 * \code
  void* dstStart = (void*)((char*)dstHost+dstY*dstPitch + dstXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  LWdeviceptr dstStart = dstDevice+dstY*dstPitch+dstXInBytes;
 * \endcode
 *
 * \par
 * For LWCA arrays, ::dstXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::WidthInBytes and ::Height specify the width (in bytes) and height of
 *   the 2D copy being performed.
 * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
 *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
 *   ::WidthInBytes + dstXInBytes.
 *
 * \par
 * ::lwMemcpy2D() returns an error if any pitch is greater than the maximum
 * allowed (::LW_DEVICE_ATTRIBUTE_MAX_PITCH). ::lwMemAllocPitch() passes back
 * pitches that always work with ::lwMemcpy2D(). On intra-device memory copies
 * (device to device, LWCA array to device, LWCA array to LWCA array),
 * ::lwMemcpy2D() may fail for pitches not computed by ::lwMemAllocPitch().
 * ::lwMemcpy2DUnaligned() does not have this restriction, but may run
 * significantly slower in the cases where ::lwMemcpy2D() would have returned
 * an error code.
 *
 * \param pCopy - Parameters for the memory copy
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMemcpy2D,
 * ::lwdaMemcpy2DToArray,
 * ::lwdaMemcpy2DFromArray
 */
LWresult LWDAAPI lwMemcpy2D(const LWDA_MEMCPY2D *pCopy);

/**
 * \brief Copies memory for 2D arrays
 *
 * Perform a 2D memory copy according to the parameters specified in \p pCopy.
 * The ::LWDA_MEMCPY2D structure is defined as:
 *
 * \code
   typedef struct LWDA_MEMCPY2D_st {
      unsigned int srcXInBytes, srcY;
      LWmemorytype srcMemoryType;
      const void *srcHost;
      LWdeviceptr srcDevice;
      LWarray srcArray;
      unsigned int srcPitch;
      unsigned int dstXInBytes, dstY;
      LWmemorytype dstMemoryType;
      void *dstHost;
      LWdeviceptr dstDevice;
      LWarray dstArray;
      unsigned int dstPitch;
      unsigned int WidthInBytes;
      unsigned int Height;
   } LWDA_MEMCPY2D;
 * \endcode
 * where:
 * - ::srcMemoryType and ::dstMemoryType specify the type of memory of the
 *   source and destination, respectively; ::LWmemorytype_enum is defined as:
 *
 * \code
   typedef enum LWmemorytype_enum {
      LW_MEMORYTYPE_HOST = 0x01,
      LW_MEMORYTYPE_DEVICE = 0x02,
      LW_MEMORYTYPE_ARRAY = 0x03,
      LW_MEMORYTYPE_UNIFIED = 0x04
   } LWmemorytype;
 * \endcode
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_UNIFIED, ::srcDevice and ::srcPitch
 *   specify the (unified virtual address space) base address of the source data 
 *   and the bytes per row to apply.  ::srcArray is ignored.  
 * This value may be used only if unified addressing is supported in the calling 
 *   context.
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_HOST, ::srcHost and ::srcPitch
 * specify the (host) base address of the source data and the bytes per row to
 * apply. ::srcArray is ignored.
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_DEVICE, ::srcDevice and ::srcPitch
 * specify the (device) base address of the source data and the bytes per row
 * to apply. ::srcArray is ignored.
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_ARRAY, ::srcArray specifies the
 * handle of the source data. ::srcHost, ::srcDevice and ::srcPitch are
 * ignored.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_UNIFIED, ::dstDevice and ::dstPitch
 *   specify the (unified virtual address space) base address of the source data 
 *   and the bytes per row to apply.  ::dstArray is ignored.  
 * This value may be used only if unified addressing is supported in the calling 
 *   context.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_HOST, ::dstHost and ::dstPitch
 * specify the (host) base address of the destination data and the bytes per
 * row to apply. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_DEVICE, ::dstDevice and ::dstPitch
 * specify the (device) base address of the destination data and the bytes per
 * row to apply. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_ARRAY, ::dstArray specifies the
 * handle of the destination data. ::dstHost, ::dstDevice and ::dstPitch are
 * ignored.
 *
 * - ::srcXInBytes and ::srcY specify the base address of the source data for
 *   the copy.
 *
 * \par
 * For host pointers, the starting address is
 * \code
  void* Start = (void*)((char*)srcHost+srcY*srcPitch + srcXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  LWdeviceptr Start = srcDevice+srcY*srcPitch+srcXInBytes;
 * \endcode
 *
 * \par
 * For LWCA arrays, ::srcXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::dstXInBytes and ::dstY specify the base address of the destination data
 *   for the copy.
 *
 * \par
 * For host pointers, the base address is
 * \code
  void* dstStart = (void*)((char*)dstHost+dstY*dstPitch + dstXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  LWdeviceptr dstStart = dstDevice+dstY*dstPitch+dstXInBytes;
 * \endcode
 *
 * \par
 * For LWCA arrays, ::dstXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::WidthInBytes and ::Height specify the width (in bytes) and height of
 *   the 2D copy being performed.
 * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
 *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
 *   ::WidthInBytes + dstXInBytes.
 *
 * \par
 * ::lwMemcpy2D() returns an error if any pitch is greater than the maximum
 * allowed (::LW_DEVICE_ATTRIBUTE_MAX_PITCH). ::lwMemAllocPitch() passes back
 * pitches that always work with ::lwMemcpy2D(). On intra-device memory copies
 * (device to device, LWCA array to device, LWCA array to LWCA array),
 * ::lwMemcpy2D() may fail for pitches not computed by ::lwMemAllocPitch().
 * ::lwMemcpy2DUnaligned() does not have this restriction, but may run
 * significantly slower in the cases where ::lwMemcpy2D() would have returned
 * an error code.
 *
 * \param pCopy - Parameters for the memory copy
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMemcpy2D,
 * ::lwdaMemcpy2DToArray,
 * ::lwdaMemcpy2DFromArray
 */
LWresult LWDAAPI lwMemcpy2DUnaligned(const LWDA_MEMCPY2D *pCopy);

/**
 * \brief Copies memory for 3D arrays
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p pCopy. The ::LWDA_MEMCPY3D structure is defined as:
 *
 * \code
        typedef struct LWDA_MEMCPY3D_st {

            unsigned int srcXInBytes, srcY, srcZ;
            unsigned int srcLOD;
            LWmemorytype srcMemoryType;
                const void *srcHost;
                LWdeviceptr srcDevice;
                LWarray srcArray;
                unsigned int srcPitch;  // ignored when src is array
                unsigned int srcHeight; // ignored when src is array; may be 0 if Depth==1

            unsigned int dstXInBytes, dstY, dstZ;
            unsigned int dstLOD;
            LWmemorytype dstMemoryType;
                void *dstHost;
                LWdeviceptr dstDevice;
                LWarray dstArray;
                unsigned int dstPitch;  // ignored when dst is array
                unsigned int dstHeight; // ignored when dst is array; may be 0 if Depth==1

            unsigned int WidthInBytes;
            unsigned int Height;
            unsigned int Depth;
        } LWDA_MEMCPY3D;
 * \endcode
 * where:
 * - ::srcMemoryType and ::dstMemoryType specify the type of memory of the
 *   source and destination, respectively; ::LWmemorytype_enum is defined as:
 *
 * \code
   typedef enum LWmemorytype_enum {
      LW_MEMORYTYPE_HOST = 0x01,
      LW_MEMORYTYPE_DEVICE = 0x02,
      LW_MEMORYTYPE_ARRAY = 0x03,
      LW_MEMORYTYPE_UNIFIED = 0x04
   } LWmemorytype;
 * \endcode
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_UNIFIED, ::srcDevice and ::srcPitch
 *   specify the (unified virtual address space) base address of the source data 
 *   and the bytes per row to apply.  ::srcArray is ignored.  
 * This value may be used only if unified addressing is supported in the calling 
 *   context.
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_HOST, ::srcHost, ::srcPitch and
 * ::srcHeight specify the (host) base address of the source data, the bytes
 * per row, and the height of each 2D slice of the 3D array. ::srcArray is
 * ignored.
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_DEVICE, ::srcDevice, ::srcPitch and
 * ::srcHeight specify the (device) base address of the source data, the bytes
 * per row, and the height of each 2D slice of the 3D array. ::srcArray is
 * ignored.
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_ARRAY, ::srcArray specifies the
 * handle of the source data. ::srcHost, ::srcDevice, ::srcPitch and
 * ::srcHeight are ignored.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_UNIFIED, ::dstDevice and ::dstPitch
 *   specify the (unified virtual address space) base address of the source data 
 *   and the bytes per row to apply.  ::dstArray is ignored.  
 * This value may be used only if unified addressing is supported in the calling 
 *   context.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_HOST, ::dstHost and ::dstPitch
 * specify the (host) base address of the destination data, the bytes per row,
 * and the height of each 2D slice of the 3D array. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_DEVICE, ::dstDevice and ::dstPitch
 * specify the (device) base address of the destination data, the bytes per
 * row, and the height of each 2D slice of the 3D array. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_ARRAY, ::dstArray specifies the
 * handle of the destination data. ::dstHost, ::dstDevice, ::dstPitch and
 * ::dstHeight are ignored.
 *
 * - ::srcXInBytes, ::srcY and ::srcZ specify the base address of the source
 *   data for the copy.
 *
 * \par
 * For host pointers, the starting address is
 * \code
  void* Start = (void*)((char*)srcHost+(srcZ*srcHeight+srcY)*srcPitch + srcXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  LWdeviceptr Start = srcDevice+(srcZ*srcHeight+srcY)*srcPitch+srcXInBytes;
 * \endcode
 *
 * \par
 * For LWCA arrays, ::srcXInBytes must be evenly divisible by the array
 * element size.
 *
 * - dstXInBytes, ::dstY and ::dstZ specify the base address of the
 *   destination data for the copy.
 *
 * \par
 * For host pointers, the base address is
 * \code
  void* dstStart = (void*)((char*)dstHost+(dstZ*dstHeight+dstY)*dstPitch + dstXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  LWdeviceptr dstStart = dstDevice+(dstZ*dstHeight+dstY)*dstPitch+dstXInBytes;
 * \endcode
 *
 * \par
 * For LWCA arrays, ::dstXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::WidthInBytes, ::Height and ::Depth specify the width (in bytes), height
 *   and depth of the 3D copy being performed.
 * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
 *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
 *   ::WidthInBytes + dstXInBytes.
 * - If specified, ::srcHeight must be greater than or equal to ::Height +
 *   ::srcY, and ::dstHeight must be greater than or equal to ::Height + ::dstY.
 *
 * \par
 * ::lwMemcpy3D() returns an error if any pitch is greater than the maximum
 * allowed (::LW_DEVICE_ATTRIBUTE_MAX_PITCH).
 *
 * The ::srcLOD and ::dstLOD members of the ::LWDA_MEMCPY3D structure must be
 * set to 0.
 *
 * \param pCopy - Parameters for the memory copy
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMemcpy3D
 */
LWresult LWDAAPI lwMemcpy3D(const LWDA_MEMCPY3D *pCopy);
#endif /* __LWDA_API_VERSION >= 3020 */

#if __LWDA_API_VERSION >= 4000
/**
 * \brief Copies memory between contexts
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p pCopy.  See the definition of the ::LWDA_MEMCPY3D_PEER structure
 * for documentation of its parameters.
 *
 * \param pCopy - Parameters for the memory copy
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_sync
 *
 * \sa ::lwMemcpyDtoD, ::lwMemcpyPeer, ::lwMemcpyDtoDAsync, ::lwMemcpyPeerAsync,
 * ::lwMemcpy3DPeerAsync,
 * ::lwdaMemcpy3DPeer
 */
LWresult LWDAAPI lwMemcpy3DPeer(const LWDA_MEMCPY3D_PEER *pCopy);

/**
 * \brief Copies memory asynchronously
 *
 * Copies data between two pointers. 
 * \p dst and \p src are base pointers of the destination and source, respectively.  
 * \p ByteCount specifies the number of bytes to copy.
 * Note that this function infers the type of the transfer (host to host, host to 
 *   device, device to device, or device to host) from the pointer values.  This
 *   function is only allowed in contexts which support unified addressing.
 *
 * \param dst       - Destination unified virtual address space pointer
 * \param src       - Source unified virtual address space pointer
 * \param ByteCount - Size of memory copy in bytes
 * \param hStream   - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemcpyAsync,
 * ::lwdaMemcpyToSymbolAsync,
 * ::lwdaMemcpyFromSymbolAsync
 */
LWresult LWDAAPI lwMemcpyAsync(LWdeviceptr dst, LWdeviceptr src, size_t ByteCount, LWstream hStream);

/**
 * \brief Copies device memory between two contexts asynchronously.
 *
 * Copies from device memory in one context to device memory in another
 * context. \p dstDevice is the base device pointer of the destination memory 
 * and \p dstContext is the destination context.  \p srcDevice is the base 
 * device pointer of the source memory and \p srcContext is the source pointer.  
 * \p ByteCount specifies the number of bytes to copy.
 *
 * \param dstDevice  - Destination device pointer
 * \param dstContext - Destination context
 * \param srcDevice  - Source device pointer
 * \param srcContext - Source context
 * \param ByteCount  - Size of memory copy in bytes
 * \param hStream    - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwMemcpyDtoD, ::lwMemcpyPeer, ::lwMemcpy3DPeer, ::lwMemcpyDtoDAsync, 
 * ::lwMemcpy3DPeerAsync,
 * ::lwdaMemcpyPeerAsync
 */
LWresult LWDAAPI lwMemcpyPeerAsync(LWdeviceptr dstDevice, LWcontext dstContext, LWdeviceptr srcDevice, LWcontext srcContext, size_t ByteCount, LWstream hStream);
#endif /* __LWDA_API_VERSION >= 4000 */

#if __LWDA_API_VERSION >= 3020
/**
 * \brief Copies memory from Host to Device
 *
 * Copies from host memory to device memory. \p dstDevice and \p srcHost are
 * the base addresses of the destination and source, respectively. \p ByteCount
 * specifies the number of bytes to copy.
 *
 * \param dstDevice - Destination device pointer
 * \param srcHost   - Source host pointer
 * \param ByteCount - Size of memory copy in bytes
 * \param hStream   - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemcpyAsync,
 * ::lwdaMemcpyToSymbolAsync
 */
LWresult LWDAAPI lwMemcpyHtoDAsync(LWdeviceptr dstDevice, const void *srcHost, size_t ByteCount, LWstream hStream);

/**
 * \brief Copies memory from Device to Host
 *
 * Copies from device to host memory. \p dstHost and \p srcDevice specify the
 * base pointers of the destination and source, respectively. \p ByteCount
 * specifies the number of bytes to copy.
 *
 * \param dstHost   - Destination host pointer
 * \param srcDevice - Source device pointer
 * \param ByteCount - Size of memory copy in bytes
 * \param hStream   - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemcpyAsync,
 * ::lwdaMemcpyFromSymbolAsync
 */
LWresult LWDAAPI lwMemcpyDtoHAsync(void *dstHost, LWdeviceptr srcDevice, size_t ByteCount, LWstream hStream);

/**
 * \brief Copies memory from Device to Device
 *
 * Copies from device memory to device memory. \p dstDevice and \p srcDevice
 * are the base pointers of the destination and source, respectively.
 * \p ByteCount specifies the number of bytes to copy.
 *
 * \param dstDevice - Destination device pointer
 * \param srcDevice - Source device pointer
 * \param ByteCount - Size of memory copy in bytes
 * \param hStream   - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemcpyAsync,
 * ::lwdaMemcpyToSymbolAsync,
 * ::lwdaMemcpyFromSymbolAsync
 */
LWresult LWDAAPI lwMemcpyDtoDAsync(LWdeviceptr dstDevice, LWdeviceptr srcDevice, size_t ByteCount, LWstream hStream);

/**
 * \brief Copies memory from Host to Array
 *
 * Copies from host memory to a 1D LWCA array. \p dstArray and \p dstOffset
 * specify the LWCA array handle and starting offset in bytes of the
 * destination data. \p srcHost specifies the base address of the source.
 * \p ByteCount specifies the number of bytes to copy.
 *
 * \param dstArray  - Destination array
 * \param dstOffset - Offset in bytes of destination array
 * \param srcHost   - Source host pointer
 * \param ByteCount - Size of memory copy in bytes
 * \param hStream   - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemcpyToArrayAsync
 */
LWresult LWDAAPI lwMemcpyHtoAAsync(LWarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, LWstream hStream);

/**
 * \brief Copies memory from Array to Host
 *
 * Copies from one 1D LWCA array to host memory. \p dstHost specifies the base
 * pointer of the destination. \p srcArray and \p srcOffset specify the LWCA
 * array handle and starting offset in bytes of the source data.
 * \p ByteCount specifies the number of bytes to copy.
 *
 * \param dstHost   - Destination pointer
 * \param srcArray  - Source array
 * \param srcOffset - Offset in bytes of source array
 * \param ByteCount - Size of memory copy in bytes
 * \param hStream   - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemcpyFromArrayAsync
 */
LWresult LWDAAPI lwMemcpyAtoHAsync(void *dstHost, LWarray srcArray, size_t srcOffset, size_t ByteCount, LWstream hStream);

/**
 * \brief Copies memory for 2D arrays
 *
 * Perform a 2D memory copy according to the parameters specified in \p pCopy.
 * The ::LWDA_MEMCPY2D structure is defined as:
 *
 * \code
   typedef struct LWDA_MEMCPY2D_st {
      unsigned int srcXInBytes, srcY;
      LWmemorytype srcMemoryType;
      const void *srcHost;
      LWdeviceptr srcDevice;
      LWarray srcArray;
      unsigned int srcPitch;
      unsigned int dstXInBytes, dstY;
      LWmemorytype dstMemoryType;
      void *dstHost;
      LWdeviceptr dstDevice;
      LWarray dstArray;
      unsigned int dstPitch;
      unsigned int WidthInBytes;
      unsigned int Height;
   } LWDA_MEMCPY2D;
 * \endcode
 * where:
 * - ::srcMemoryType and ::dstMemoryType specify the type of memory of the
 *   source and destination, respectively; ::LWmemorytype_enum is defined as:
 *
 * \code
   typedef enum LWmemorytype_enum {
      LW_MEMORYTYPE_HOST = 0x01,
      LW_MEMORYTYPE_DEVICE = 0x02,
      LW_MEMORYTYPE_ARRAY = 0x03,
      LW_MEMORYTYPE_UNIFIED = 0x04
   } LWmemorytype;
 * \endcode
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_HOST, ::srcHost and ::srcPitch
 * specify the (host) base address of the source data and the bytes per row to
 * apply. ::srcArray is ignored.
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_UNIFIED, ::srcDevice and ::srcPitch
 *   specify the (unified virtual address space) base address of the source data 
 *   and the bytes per row to apply.  ::srcArray is ignored.  
 * This value may be used only if unified addressing is supported in the calling 
 *   context.
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_DEVICE, ::srcDevice and ::srcPitch
 * specify the (device) base address of the source data and the bytes per row
 * to apply. ::srcArray is ignored.
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_ARRAY, ::srcArray specifies the
 * handle of the source data. ::srcHost, ::srcDevice and ::srcPitch are
 * ignored.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_UNIFIED, ::dstDevice and ::dstPitch
 *   specify the (unified virtual address space) base address of the source data 
 *   and the bytes per row to apply.  ::dstArray is ignored.  
 * This value may be used only if unified addressing is supported in the calling 
 *   context.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_HOST, ::dstHost and ::dstPitch
 * specify the (host) base address of the destination data and the bytes per
 * row to apply. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_DEVICE, ::dstDevice and ::dstPitch
 * specify the (device) base address of the destination data and the bytes per
 * row to apply. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_ARRAY, ::dstArray specifies the
 * handle of the destination data. ::dstHost, ::dstDevice and ::dstPitch are
 * ignored.
 *
 * - ::srcXInBytes and ::srcY specify the base address of the source data for
 *   the copy.
 *
 * \par
 * For host pointers, the starting address is
 * \code
  void* Start = (void*)((char*)srcHost+srcY*srcPitch + srcXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  LWdeviceptr Start = srcDevice+srcY*srcPitch+srcXInBytes;
 * \endcode
 *
 * \par
 * For LWCA arrays, ::srcXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::dstXInBytes and ::dstY specify the base address of the destination data
 *   for the copy.
 *
 * \par
 * For host pointers, the base address is
 * \code
  void* dstStart = (void*)((char*)dstHost+dstY*dstPitch + dstXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  LWdeviceptr dstStart = dstDevice+dstY*dstPitch+dstXInBytes;
 * \endcode
 *
 * \par
 * For LWCA arrays, ::dstXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::WidthInBytes and ::Height specify the width (in bytes) and height of
 *   the 2D copy being performed.
 * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
 *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
 *   ::WidthInBytes + dstXInBytes.
 * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
 *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
 *   ::WidthInBytes + dstXInBytes.
 * - If specified, ::srcHeight must be greater than or equal to ::Height +
 *   ::srcY, and ::dstHeight must be greater than or equal to ::Height + ::dstY.
 *
 * \par
 * ::lwMemcpy2DAsync() returns an error if any pitch is greater than the maximum
 * allowed (::LW_DEVICE_ATTRIBUTE_MAX_PITCH). ::lwMemAllocPitch() passes back
 * pitches that always work with ::lwMemcpy2D(). On intra-device memory copies
 * (device to device, LWCA array to device, LWCA array to LWCA array),
 * ::lwMemcpy2DAsync() may fail for pitches not computed by ::lwMemAllocPitch().
 *
 * \param pCopy   - Parameters for the memory copy
 * \param hStream - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemcpy2DAsync,
 * ::lwdaMemcpy2DToArrayAsync,
 * ::lwdaMemcpy2DFromArrayAsync
 */
LWresult LWDAAPI lwMemcpy2DAsync(const LWDA_MEMCPY2D *pCopy, LWstream hStream);

/**
 * \brief Copies memory for 3D arrays
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p pCopy. The ::LWDA_MEMCPY3D structure is defined as:
 *
 * \code
        typedef struct LWDA_MEMCPY3D_st {

            unsigned int srcXInBytes, srcY, srcZ;
            unsigned int srcLOD;
            LWmemorytype srcMemoryType;
                const void *srcHost;
                LWdeviceptr srcDevice;
                LWarray srcArray;
                unsigned int srcPitch;  // ignored when src is array
                unsigned int srcHeight; // ignored when src is array; may be 0 if Depth==1

            unsigned int dstXInBytes, dstY, dstZ;
            unsigned int dstLOD;
            LWmemorytype dstMemoryType;
                void *dstHost;
                LWdeviceptr dstDevice;
                LWarray dstArray;
                unsigned int dstPitch;  // ignored when dst is array
                unsigned int dstHeight; // ignored when dst is array; may be 0 if Depth==1

            unsigned int WidthInBytes;
            unsigned int Height;
            unsigned int Depth;
        } LWDA_MEMCPY3D;
 * \endcode
 * where:
 * - ::srcMemoryType and ::dstMemoryType specify the type of memory of the
 *   source and destination, respectively; ::LWmemorytype_enum is defined as:
 *
 * \code
   typedef enum LWmemorytype_enum {
      LW_MEMORYTYPE_HOST = 0x01,
      LW_MEMORYTYPE_DEVICE = 0x02,
      LW_MEMORYTYPE_ARRAY = 0x03,
      LW_MEMORYTYPE_UNIFIED = 0x04
   } LWmemorytype;
 * \endcode
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_UNIFIED, ::srcDevice and ::srcPitch
 *   specify the (unified virtual address space) base address of the source data 
 *   and the bytes per row to apply.  ::srcArray is ignored.  
 * This value may be used only if unified addressing is supported in the calling 
 *   context.
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_HOST, ::srcHost, ::srcPitch and
 * ::srcHeight specify the (host) base address of the source data, the bytes
 * per row, and the height of each 2D slice of the 3D array. ::srcArray is
 * ignored.
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_DEVICE, ::srcDevice, ::srcPitch and
 * ::srcHeight specify the (device) base address of the source data, the bytes
 * per row, and the height of each 2D slice of the 3D array. ::srcArray is
 * ignored.
 *
 * \par
 * If ::srcMemoryType is ::LW_MEMORYTYPE_ARRAY, ::srcArray specifies the
 * handle of the source data. ::srcHost, ::srcDevice, ::srcPitch and
 * ::srcHeight are ignored.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_UNIFIED, ::dstDevice and ::dstPitch
 *   specify the (unified virtual address space) base address of the source data 
 *   and the bytes per row to apply.  ::dstArray is ignored.  
 * This value may be used only if unified addressing is supported in the calling 
 *   context.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_HOST, ::dstHost and ::dstPitch
 * specify the (host) base address of the destination data, the bytes per row,
 * and the height of each 2D slice of the 3D array. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_DEVICE, ::dstDevice and ::dstPitch
 * specify the (device) base address of the destination data, the bytes per
 * row, and the height of each 2D slice of the 3D array. ::dstArray is ignored.
 *
 * \par
 * If ::dstMemoryType is ::LW_MEMORYTYPE_ARRAY, ::dstArray specifies the
 * handle of the destination data. ::dstHost, ::dstDevice, ::dstPitch and
 * ::dstHeight are ignored.
 *
 * - ::srcXInBytes, ::srcY and ::srcZ specify the base address of the source
 *   data for the copy.
 *
 * \par
 * For host pointers, the starting address is
 * \code
  void* Start = (void*)((char*)srcHost+(srcZ*srcHeight+srcY)*srcPitch + srcXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  LWdeviceptr Start = srcDevice+(srcZ*srcHeight+srcY)*srcPitch+srcXInBytes;
 * \endcode
 *
 * \par
 * For LWCA arrays, ::srcXInBytes must be evenly divisible by the array
 * element size.
 *
 * - dstXInBytes, ::dstY and ::dstZ specify the base address of the
 *   destination data for the copy.
 *
 * \par
 * For host pointers, the base address is
 * \code
  void* dstStart = (void*)((char*)dstHost+(dstZ*dstHeight+dstY)*dstPitch + dstXInBytes);
 * \endcode
 *
 * \par
 * For device pointers, the starting address is
 * \code
  LWdeviceptr dstStart = dstDevice+(dstZ*dstHeight+dstY)*dstPitch+dstXInBytes;
 * \endcode
 *
 * \par
 * For LWCA arrays, ::dstXInBytes must be evenly divisible by the array
 * element size.
 *
 * - ::WidthInBytes, ::Height and ::Depth specify the width (in bytes), height
 *   and depth of the 3D copy being performed.
 * - If specified, ::srcPitch must be greater than or equal to ::WidthInBytes +
 *   ::srcXInBytes, and ::dstPitch must be greater than or equal to
 *   ::WidthInBytes + dstXInBytes.
 * - If specified, ::srcHeight must be greater than or equal to ::Height +
 *   ::srcY, and ::dstHeight must be greater than or equal to ::Height + ::dstY.
 *
 * \par
 * ::lwMemcpy3DAsync() returns an error if any pitch is greater than the maximum
 * allowed (::LW_DEVICE_ATTRIBUTE_MAX_PITCH).
 *
 * The ::srcLOD and ::dstLOD members of the ::LWDA_MEMCPY3D structure must be
 * set to 0.
 *
 * \param pCopy - Parameters for the memory copy
 * \param hStream - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemcpy3DAsync
 */
LWresult LWDAAPI lwMemcpy3DAsync(const LWDA_MEMCPY3D *pCopy, LWstream hStream);
#endif /* __LWDA_API_VERSION >= 3020 */

#if __LWDA_API_VERSION >= 4000
/**
 * \brief Copies memory between contexts asynchronously.
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p pCopy.  See the definition of the ::LWDA_MEMCPY3D_PEER structure
 * for documentation of its parameters.
 *
 * \param pCopy - Parameters for the memory copy
 * \param hStream - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwMemcpyDtoD, ::lwMemcpyPeer, ::lwMemcpyDtoDAsync, ::lwMemcpyPeerAsync,
 * ::lwMemcpy3DPeerAsync,
 * ::lwdaMemcpy3DPeerAsync
 */
LWresult LWDAAPI lwMemcpy3DPeerAsync(const LWDA_MEMCPY3D_PEER *pCopy, LWstream hStream);
#endif /* __LWDA_API_VERSION >= 4000 */

#if __LWDA_API_VERSION >= 3020
/**
 * \brief Initializes device memory
 *
 * Sets the memory range of \p N 8-bit values to the specified value
 * \p uc.
 *
 * \param dstDevice - Destination device pointer
 * \param uc        - Value to set
 * \param N         - Number of elements
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_memset
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemset
 */
LWresult LWDAAPI lwMemsetD8(LWdeviceptr dstDevice, unsigned char uc, size_t N);

/**
 * \brief Initializes device memory
 *
 * Sets the memory range of \p N 16-bit values to the specified value
 * \p us. The \p dstDevice pointer must be two byte aligned.
 *
 * \param dstDevice - Destination device pointer
 * \param us        - Value to set
 * \param N         - Number of elements
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_memset
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemset
 */
LWresult LWDAAPI lwMemsetD16(LWdeviceptr dstDevice, unsigned short us, size_t N);

/**
 * \brief Initializes device memory
 *
 * Sets the memory range of \p N 32-bit values to the specified value
 * \p ui. The \p dstDevice pointer must be four byte aligned.
 *
 * \param dstDevice - Destination device pointer
 * \param ui        - Value to set
 * \param N         - Number of elements
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_memset
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32Async,
 * ::lwdaMemset
 */
LWresult LWDAAPI lwMemsetD32(LWdeviceptr dstDevice, unsigned int ui, size_t N);

/**
 * \brief Initializes device memory
 *
 * Sets the 2D memory range of \p Width 8-bit values to the specified value
 * \p uc. \p Height specifies the number of rows to set, and \p dstPitch
 * specifies the number of bytes between each row. This function performs
 * fastest when the pitch is one that has been passed back by
 * ::lwMemAllocPitch().
 *
 * \param dstDevice - Destination device pointer
 * \param dstPitch  - Pitch of destination device pointer
 * \param uc        - Value to set
 * \param Width     - Width of row
 * \param Height    - Number of rows
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_memset
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemset2D
 */
LWresult LWDAAPI lwMemsetD2D8(LWdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);

/**
 * \brief Initializes device memory
 *
 * Sets the 2D memory range of \p Width 16-bit values to the specified value
 * \p us. \p Height specifies the number of rows to set, and \p dstPitch
 * specifies the number of bytes between each row. The \p dstDevice pointer
 * and \p dstPitch offset must be two byte aligned. This function performs
 * fastest when the pitch is one that has been passed back by
 * ::lwMemAllocPitch().
 *
 * \param dstDevice - Destination device pointer
 * \param dstPitch  - Pitch of destination device pointer
 * \param us        - Value to set
 * \param Width     - Width of row
 * \param Height    - Number of rows
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_memset
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemset2D
 */
LWresult LWDAAPI lwMemsetD2D16(LWdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);

/**
 * \brief Initializes device memory
 *
 * Sets the 2D memory range of \p Width 32-bit values to the specified value
 * \p ui. \p Height specifies the number of rows to set, and \p dstPitch
 * specifies the number of bytes between each row. The \p dstDevice pointer
 * and \p dstPitch offset must be four byte aligned. This function performs
 * fastest when the pitch is one that has been passed back by
 * ::lwMemAllocPitch().
 *
 * \param dstDevice - Destination device pointer
 * \param dstPitch  - Pitch of destination device pointer
 * \param ui        - Value to set
 * \param Width     - Width of row
 * \param Height    - Number of rows
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_memset
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemset2D
 */
LWresult LWDAAPI lwMemsetD2D32(LWdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);

/**
 * \brief Sets device memory
 *
 * Sets the memory range of \p N 8-bit values to the specified value
 * \p uc.
 *
 * \param dstDevice - Destination device pointer
 * \param uc        - Value to set
 * \param N         - Number of elements
 * \param hStream   - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemsetAsync
 */
LWresult LWDAAPI lwMemsetD8Async(LWdeviceptr dstDevice, unsigned char uc, size_t N, LWstream hStream);

/**
 * \brief Sets device memory
 *
 * Sets the memory range of \p N 16-bit values to the specified value
 * \p us. The \p dstDevice pointer must be two byte aligned.
 *
 * \param dstDevice - Destination device pointer
 * \param us        - Value to set
 * \param N         - Number of elements
 * \param hStream   - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemsetAsync
 */
LWresult LWDAAPI lwMemsetD16Async(LWdeviceptr dstDevice, unsigned short us, size_t N, LWstream hStream);

/**
 * \brief Sets device memory
 *
 * Sets the memory range of \p N 32-bit values to the specified value
 * \p ui. The \p dstDevice pointer must be four byte aligned.
 *
 * \param dstDevice - Destination device pointer
 * \param ui        - Value to set
 * \param N         - Number of elements
 * \param hStream   - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async, ::lwMemsetD32,
 * ::lwdaMemsetAsync
 */
LWresult LWDAAPI lwMemsetD32Async(LWdeviceptr dstDevice, unsigned int ui, size_t N, LWstream hStream);

/**
 * \brief Sets device memory
 *
 * Sets the 2D memory range of \p Width 8-bit values to the specified value
 * \p uc. \p Height specifies the number of rows to set, and \p dstPitch
 * specifies the number of bytes between each row. This function performs
 * fastest when the pitch is one that has been passed back by
 * ::lwMemAllocPitch().
 *
 * \param dstDevice - Destination device pointer
 * \param dstPitch  - Pitch of destination device pointer
 * \param uc        - Value to set
 * \param Width     - Width of row
 * \param Height    - Number of rows
 * \param hStream   - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemset2DAsync
 */
LWresult LWDAAPI lwMemsetD2D8Async(LWdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, LWstream hStream);

/**
 * \brief Sets device memory
 *
 * Sets the 2D memory range of \p Width 16-bit values to the specified value
 * \p us. \p Height specifies the number of rows to set, and \p dstPitch
 * specifies the number of bytes between each row. The \p dstDevice pointer 
 * and \p dstPitch offset must be two byte aligned. This function performs
 * fastest when the pitch is one that has been passed back by
 * ::lwMemAllocPitch().
 *
 * \param dstDevice - Destination device pointer
 * \param dstPitch  - Pitch of destination device pointer
 * \param us        - Value to set
 * \param Width     - Width of row
 * \param Height    - Number of rows
 * \param hStream   - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D32, ::lwMemsetD2D32Async,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemset2DAsync
 */
LWresult LWDAAPI lwMemsetD2D16Async(LWdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, LWstream hStream);

/**
 * \brief Sets device memory
 *
 * Sets the 2D memory range of \p Width 32-bit values to the specified value
 * \p ui. \p Height specifies the number of rows to set, and \p dstPitch
 * specifies the number of bytes between each row. The \p dstDevice pointer
 * and \p dstPitch offset must be four byte aligned. This function performs
 * fastest when the pitch is one that has been passed back by
 * ::lwMemAllocPitch().
 *
 * \param dstDevice - Destination device pointer
 * \param dstPitch  - Pitch of destination device pointer
 * \param ui        - Value to set
 * \param Width     - Width of row
 * \param Height    - Number of rows
 * \param hStream   - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 * \note_memset
 * \note_null_stream
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D8Async,
 * ::lwMemsetD2D16, ::lwMemsetD2D16Async, ::lwMemsetD2D32,
 * ::lwMemsetD8, ::lwMemsetD8Async, ::lwMemsetD16, ::lwMemsetD16Async,
 * ::lwMemsetD32, ::lwMemsetD32Async,
 * ::lwdaMemset2DAsync
 */
LWresult LWDAAPI lwMemsetD2D32Async(LWdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, LWstream hStream);

/**
 * \brief Creates a 1D or 2D LWCA array
 *
 * Creates a LWCA array according to the ::LWDA_ARRAY_DESCRIPTOR structure
 * \p pAllocateArray and returns a handle to the new LWCA array in \p *pHandle.
 * The ::LWDA_ARRAY_DESCRIPTOR is defined as:
 *
 * \code
    typedef struct {
        unsigned int Width;
        unsigned int Height;
        LWarray_format Format;
        unsigned int NumChannels;
    } LWDA_ARRAY_DESCRIPTOR;
 * \endcode
 * where:
 *
 * - \p Width, and \p Height are the width, and height of the LWCA array (in
 * elements); the LWCA array is one-dimensional if height is 0, two-dimensional
 * otherwise;
 * - ::Format specifies the format of the elements; ::LWarray_format is
 * defined as:
 * \code
    typedef enum LWarray_format_enum {
        LW_AD_FORMAT_UNSIGNED_INT8 = 0x01,
        LW_AD_FORMAT_UNSIGNED_INT16 = 0x02,
        LW_AD_FORMAT_UNSIGNED_INT32 = 0x03,
        LW_AD_FORMAT_SIGNED_INT8 = 0x08,
        LW_AD_FORMAT_SIGNED_INT16 = 0x09,
        LW_AD_FORMAT_SIGNED_INT32 = 0x0a,
        LW_AD_FORMAT_HALF = 0x10,
        LW_AD_FORMAT_FLOAT = 0x20
    } LWarray_format;
 *  \endcode
 * - \p NumChannels specifies the number of packed components per LWCA array
 * element; it may be 1, 2, or 4;
 *
 * Here are examples of LWCA array descriptions:
 *
 * Description for a LWCA array of 2048 floats:
 * \code
    LWDA_ARRAY_DESCRIPTOR desc;
    desc.Format = LW_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = 2048;
    desc.Height = 1;
 * \endcode
 *
 * Description for a 64 x 64 LWCA array of floats:
 * \code
    LWDA_ARRAY_DESCRIPTOR desc;
    desc.Format = LW_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = 64;
    desc.Height = 64;
 * \endcode
 *
 * Description for a \p width x \p height LWCA array of 64-bit, 4x16-bit
 * float16's:
 * \code
    LWDA_ARRAY_DESCRIPTOR desc;
    desc.FormatFlags = LW_AD_FORMAT_HALF;
    desc.NumChannels = 4;
    desc.Width = width;
    desc.Height = height;
 * \endcode
 *
 * Description for a \p width x \p height LWCA array of 16-bit elements, each
 * of which is two 8-bit unsigned chars:
 * \code
    LWDA_ARRAY_DESCRIPTOR arrayDesc;
    desc.FormatFlags = LW_AD_FORMAT_UNSIGNED_INT8;
    desc.NumChannels = 2;
    desc.Width = width;
    desc.Height = height;
 * \endcode
 *
 * \param pHandle        - Returned array
 * \param pAllocateArray - Array descriptor
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMallocArray
 */
LWresult LWDAAPI lwArrayCreate(LWarray *pHandle, const LWDA_ARRAY_DESCRIPTOR *pAllocateArray);

/**
 * \brief Get a 1D or 2D LWCA array descriptor
 *
 * Returns in \p *pArrayDescriptor a descriptor containing information on the
 * format and dimensions of the LWCA array \p hArray. It is useful for
 * subroutines that have been passed a LWCA array, but need to know the LWCA
 * array parameters for validation or other purposes.
 *
 * \param pArrayDescriptor - Returned array descriptor
 * \param hArray           - Array to get descriptor of
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_HANDLE
 * \notefnerr
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaArrayGetInfo
 */
LWresult LWDAAPI lwArrayGetDescriptor(LWDA_ARRAY_DESCRIPTOR *pArrayDescriptor, LWarray hArray);
#endif /* __LWDA_API_VERSION >= 3020 */


/**
 * \brief Destroys a LWCA array
 *
 * Destroys the LWCA array \p hArray.
 *
 * \param hArray - Array to destroy
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ARRAY_IS_MAPPED
 * \notefnerr
 *
 * \sa ::lwArray3DCreate, ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaFreeArray
 */
LWresult LWDAAPI lwArrayDestroy(LWarray hArray);

#if __LWDA_API_VERSION >= 3020
/**
 * \brief Creates a 3D LWCA array
 *
 * Creates a LWCA array according to the ::LWDA_ARRAY3D_DESCRIPTOR structure
 * \p pAllocateArray and returns a handle to the new LWCA array in \p *pHandle.
 * The ::LWDA_ARRAY3D_DESCRIPTOR is defined as:
 *
 * \code
    typedef struct {
        unsigned int Width;
        unsigned int Height;
        unsigned int Depth;
        LWarray_format Format;
        unsigned int NumChannels;
        unsigned int Flags;
    } LWDA_ARRAY3D_DESCRIPTOR;
 * \endcode
 * where:
 *
 * - \p Width, \p Height, and \p Depth are the width, height, and depth of the
 * LWCA array (in elements); the following types of LWCA arrays can be allocated:
 *     - A 1D array is allocated if \p Height and \p Depth extents are both zero.
 *     - A 2D array is allocated if only \p Depth extent is zero.
 *     - A 3D array is allocated if all three extents are non-zero.
 *     - A 1D layered LWCA array is allocated if only \p Height is zero and the 
 *       ::LWDA_ARRAY3D_LAYERED flag is set. Each layer is a 1D array. The number 
 *       of layers is determined by the depth extent.
 *     - A 2D layered LWCA array is allocated if all three extents are non-zero and 
 *       the ::LWDA_ARRAY3D_LAYERED flag is set. Each layer is a 2D array. The number 
 *       of layers is determined by the depth extent.
 *     - A lwbemap LWCA array is allocated if all three extents are non-zero and the
 *       ::LWDA_ARRAY3D_LWBEMAP flag is set. \p Width must be equal to \p Height, and 
 *       \p Depth must be six. A lwbemap is a special type of 2D layered LWCA array, 
 *       where the six layers represent the six faces of a lwbe. The order of the six 
 *       layers in memory is the same as that listed in ::LWarray_lwbemap_face.
 *     - A lwbemap layered LWCA array is allocated if all three extents are non-zero, 
 *       and both, ::LWDA_ARRAY3D_LWBEMAP and ::LWDA_ARRAY3D_LAYERED flags are set. 
 *       \p Width must be equal to \p Height, and \p Depth must be a multiple of six. 
 *       A lwbemap layered LWCA array is a special type of 2D layered LWCA array that 
 *       consists of a collection of lwbemaps. The first six layers represent the first 
 *       lwbemap, the next six layers form the second lwbemap, and so on.
 *
 * - ::Format specifies the format of the elements; ::LWarray_format is
 * defined as:
 * \code
    typedef enum LWarray_format_enum {
        LW_AD_FORMAT_UNSIGNED_INT8 = 0x01,
        LW_AD_FORMAT_UNSIGNED_INT16 = 0x02,
        LW_AD_FORMAT_UNSIGNED_INT32 = 0x03,
        LW_AD_FORMAT_SIGNED_INT8 = 0x08,
        LW_AD_FORMAT_SIGNED_INT16 = 0x09,
        LW_AD_FORMAT_SIGNED_INT32 = 0x0a,
        LW_AD_FORMAT_HALF = 0x10,
        LW_AD_FORMAT_FLOAT = 0x20
    } LWarray_format;
 *  \endcode
 *
 * - \p NumChannels specifies the number of packed components per LWCA array
 * element; it may be 1, 2, or 4;
 *
 * - ::Flags may be set to 
 *   - ::LWDA_ARRAY3D_LAYERED to enable creation of layered LWCA arrays. If this flag is set, 
 *     \p Depth specifies the number of layers, not the depth of a 3D array.
 *   - ::LWDA_ARRAY3D_SURFACE_LDST to enable surface references to be bound to the LWCA array.  
 *     If this flag is not set, ::lwSurfRefSetArray will fail when attempting to bind the LWCA array 
 *     to a surface reference.
 *   - ::LWDA_ARRAY3D_LWBEMAP to enable creation of lwbemaps. If this flag is set, \p Width must be
 *     equal to \p Height, and \p Depth must be six. If the ::LWDA_ARRAY3D_LAYERED flag is also set,
 *     then \p Depth must be a multiple of six.
 *   - ::LWDA_ARRAY3D_TEXTURE_GATHER to indicate that the LWCA array will be used for texture gather.
 *     Texture gather can only be performed on 2D LWCA arrays.
 *
 * \p Width, \p Height and \p Depth must meet certain size requirements as listed in the following table. 
 * All values are specified in elements. Note that for brevity's sake, the full name of the device attribute 
 * is not specified. For ex., TEXTURE1D_WIDTH refers to the device attribute 
 * ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH.
 *
 * Note that 2D LWCA arrays have different size requirements if the ::LWDA_ARRAY3D_TEXTURE_GATHER flag 
 * is set. \p Width and \p Height must not be greater than ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH 
 * and ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT respectively, in that case.
 *
 * <table>
 * <tr><td><b>LWCA array type</b></td>
 * <td><b>Valid extents that must always be met<br>{(width range in elements), (height range), 
 * (depth range)}</b></td>
 * <td><b>Valid extents with LWDA_ARRAY3D_SURFACE_LDST set<br> 
 * {(width range in elements), (height range), (depth range)}</b></td></tr>
 * <tr><td>1D</td>
 * <td><small>{ (1,TEXTURE1D_WIDTH), 0, 0 }</small></td>
 * <td><small>{ (1,SURFACE1D_WIDTH), 0, 0 }</small></td></tr>
 * <tr><td>2D</td>
 * <td><small>{ (1,TEXTURE2D_WIDTH), (1,TEXTURE2D_HEIGHT), 0 }</small></td>
 * <td><small>{ (1,SURFACE2D_WIDTH), (1,SURFACE2D_HEIGHT), 0 }</small></td></tr>
 * <tr><td>3D</td>
 * <td><small>{ (1,TEXTURE3D_WIDTH), (1,TEXTURE3D_HEIGHT), (1,TEXTURE3D_DEPTH) }
 * <br>OR<br>{ (1,TEXTURE3D_WIDTH_ALTERNATE), (1,TEXTURE3D_HEIGHT_ALTERNATE), 
 * (1,TEXTURE3D_DEPTH_ALTERNATE) }</small></td>
 * <td><small>{ (1,SURFACE3D_WIDTH), (1,SURFACE3D_HEIGHT), 
 * (1,SURFACE3D_DEPTH) }</small></td></tr>
 * <tr><td>1D Layered</td>
 * <td><small>{ (1,TEXTURE1D_LAYERED_WIDTH), 0, 
 * (1,TEXTURE1D_LAYERED_LAYERS) }</small></td>
 * <td><small>{ (1,SURFACE1D_LAYERED_WIDTH), 0, 
 * (1,SURFACE1D_LAYERED_LAYERS) }</small></td></tr>
 * <tr><td>2D Layered</td>
 * <td><small>{ (1,TEXTURE2D_LAYERED_WIDTH), (1,TEXTURE2D_LAYERED_HEIGHT), 
 * (1,TEXTURE2D_LAYERED_LAYERS) }</small></td>
 * <td><small>{ (1,SURFACE2D_LAYERED_WIDTH), (1,SURFACE2D_LAYERED_HEIGHT), 
 * (1,SURFACE2D_LAYERED_LAYERS) }</small></td></tr>
 * <tr><td>Lwbemap</td>
 * <td><small>{ (1,TEXTURELWBEMAP_WIDTH), (1,TEXTURELWBEMAP_WIDTH), 6 }</small></td>
 * <td><small>{ (1,SURFACELWBEMAP_WIDTH), 
 * (1,SURFACELWBEMAP_WIDTH), 6 }</small></td></tr>
 * <tr><td>Lwbemap Layered</td>
 * <td><small>{ (1,TEXTURELWBEMAP_LAYERED_WIDTH), (1,TEXTURELWBEMAP_LAYERED_WIDTH), 
 * (1,TEXTURELWBEMAP_LAYERED_LAYERS) }</small></td>
 * <td><small>{ (1,SURFACELWBEMAP_LAYERED_WIDTH), (1,SURFACELWBEMAP_LAYERED_WIDTH), 
 * (1,SURFACELWBEMAP_LAYERED_LAYERS) }</small></td></tr>
 * </table>
 *
 * Here are examples of LWCA array descriptions:
 *
 * Description for a LWCA array of 2048 floats:
 * \code
    LWDA_ARRAY3D_DESCRIPTOR desc;
    desc.Format = LW_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = 2048;
    desc.Height = 0;
    desc.Depth = 0;
 * \endcode
 *
 * Description for a 64 x 64 LWCA array of floats:
 * \code
    LWDA_ARRAY3D_DESCRIPTOR desc;
    desc.Format = LW_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = 64;
    desc.Height = 64;
    desc.Depth = 0;
 * \endcode
 *
 * Description for a \p width x \p height x \p depth LWCA array of 64-bit,
 * 4x16-bit float16's:
 * \code
    LWDA_ARRAY3D_DESCRIPTOR desc;
    desc.FormatFlags = LW_AD_FORMAT_HALF;
    desc.NumChannels = 4;
    desc.Width = width;
    desc.Height = height;
    desc.Depth = depth;
 * \endcode
 *
 * \param pHandle        - Returned array
 * \param pAllocateArray - 3D array descriptor
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa ::lwArray3DGetDescriptor, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaMalloc3DArray
 */
LWresult LWDAAPI lwArray3DCreate(LWarray *pHandle, const LWDA_ARRAY3D_DESCRIPTOR *pAllocateArray);

/**
 * \brief Get a 3D LWCA array descriptor
 *
 * Returns in \p *pArrayDescriptor a descriptor containing information on the
 * format and dimensions of the LWCA array \p hArray. It is useful for
 * subroutines that have been passed a LWCA array, but need to know the LWCA
 * array parameters for validation or other purposes.
 *
 * This function may be called on 1D and 2D arrays, in which case the \p Height
 * and/or \p Depth members of the descriptor struct will be set to 0.
 *
 * \param pArrayDescriptor - Returned 3D array descriptor
 * \param hArray           - 3D array to get descriptor of
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_HANDLE
 * \notefnerr
 *
 * \sa ::lwArray3DCreate, ::lwArrayCreate,
 * ::lwArrayDestroy, ::lwArrayGetDescriptor, ::lwMemAlloc, ::lwMemAllocHost,
 * ::lwMemAllocPitch, ::lwMemcpy2D, ::lwMemcpy2DAsync, ::lwMemcpy2DUnaligned,
 * ::lwMemcpy3D, ::lwMemcpy3DAsync, ::lwMemcpyAtoA, ::lwMemcpyAtoD,
 * ::lwMemcpyAtoH, ::lwMemcpyAtoHAsync, ::lwMemcpyDtoA, ::lwMemcpyDtoD, ::lwMemcpyDtoDAsync,
 * ::lwMemcpyDtoH, ::lwMemcpyDtoHAsync, ::lwMemcpyHtoA, ::lwMemcpyHtoAAsync,
 * ::lwMemcpyHtoD, ::lwMemcpyHtoDAsync, ::lwMemFree, ::lwMemFreeHost,
 * ::lwMemGetAddressRange, ::lwMemGetInfo, ::lwMemHostAlloc,
 * ::lwMemHostGetDevicePointer, ::lwMemsetD2D8, ::lwMemsetD2D16,
 * ::lwMemsetD2D32, ::lwMemsetD8, ::lwMemsetD16, ::lwMemsetD32,
 * ::lwdaArrayGetInfo
 */
LWresult LWDAAPI lwArray3DGetDescriptor(LWDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, LWarray hArray);
#endif /* __LWDA_API_VERSION >= 3020 */

#if __LWDA_API_VERSION >= 5000

/**
 * \brief Creates a LWCA mipmapped array
 *
 * Creates a LWCA mipmapped array according to the ::LWDA_ARRAY3D_DESCRIPTOR structure
 * \p pMipmappedArrayDesc and returns a handle to the new LWCA mipmapped array in \p *pHandle.
 * \p numMipmapLevels specifies the number of mipmap levels to be allocated. This value is
 * clamped to the range [1, 1 + floor(log2(max(width, height, depth)))].
 *
 * The ::LWDA_ARRAY3D_DESCRIPTOR is defined as:
 *
 * \code
    typedef struct {
        unsigned int Width;
        unsigned int Height;
        unsigned int Depth;
        LWarray_format Format;
        unsigned int NumChannels;
        unsigned int Flags;
    } LWDA_ARRAY3D_DESCRIPTOR;
 * \endcode
 * where:
 *
 * - \p Width, \p Height, and \p Depth are the width, height, and depth of the
 * LWCA array (in elements); the following types of LWCA arrays can be allocated:
 *     - A 1D mipmapped array is allocated if \p Height and \p Depth extents are both zero.
 *     - A 2D mipmapped array is allocated if only \p Depth extent is zero.
 *     - A 3D mipmapped array is allocated if all three extents are non-zero.
 *     - A 1D layered LWCA mipmapped array is allocated if only \p Height is zero and the 
 *       ::LWDA_ARRAY3D_LAYERED flag is set. Each layer is a 1D array. The number 
 *       of layers is determined by the depth extent.
 *     - A 2D layered LWCA mipmapped array is allocated if all three extents are non-zero and 
 *       the ::LWDA_ARRAY3D_LAYERED flag is set. Each layer is a 2D array. The number 
 *       of layers is determined by the depth extent.
 *     - A lwbemap LWCA mipmapped array is allocated if all three extents are non-zero and the
 *       ::LWDA_ARRAY3D_LWBEMAP flag is set. \p Width must be equal to \p Height, and 
 *       \p Depth must be six. A lwbemap is a special type of 2D layered LWCA array, 
 *       where the six layers represent the six faces of a lwbe. The order of the six 
 *       layers in memory is the same as that listed in ::LWarray_lwbemap_face.
 *     - A lwbemap layered LWCA mipmapped array is allocated if all three extents are non-zero, 
 *       and both, ::LWDA_ARRAY3D_LWBEMAP and ::LWDA_ARRAY3D_LAYERED flags are set. 
 *       \p Width must be equal to \p Height, and \p Depth must be a multiple of six. 
 *       A lwbemap layered LWCA array is a special type of 2D layered LWCA array that 
 *       consists of a collection of lwbemaps. The first six layers represent the first 
 *       lwbemap, the next six layers form the second lwbemap, and so on.
 *
 * - ::Format specifies the format of the elements; ::LWarray_format is
 * defined as:
 * \code
    typedef enum LWarray_format_enum {
        LW_AD_FORMAT_UNSIGNED_INT8 = 0x01,
        LW_AD_FORMAT_UNSIGNED_INT16 = 0x02,
        LW_AD_FORMAT_UNSIGNED_INT32 = 0x03,
        LW_AD_FORMAT_SIGNED_INT8 = 0x08,
        LW_AD_FORMAT_SIGNED_INT16 = 0x09,
        LW_AD_FORMAT_SIGNED_INT32 = 0x0a,
        LW_AD_FORMAT_HALF = 0x10,
        LW_AD_FORMAT_FLOAT = 0x20
    } LWarray_format;
 *  \endcode
 *
 * - \p NumChannels specifies the number of packed components per LWCA array
 * element; it may be 1, 2, or 4;
 *
 * - ::Flags may be set to 
 *   - ::LWDA_ARRAY3D_LAYERED to enable creation of layered LWCA mipmapped arrays. If this flag is set, 
 *     \p Depth specifies the number of layers, not the depth of a 3D array.
 *   - ::LWDA_ARRAY3D_SURFACE_LDST to enable surface references to be bound to individual mipmap levels of
 *     the LWCA mipmapped array. If this flag is not set, ::lwSurfRefSetArray will fail when attempting to 
 *     bind a mipmap level of the LWCA mipmapped array to a surface reference.
  *   - ::LWDA_ARRAY3D_LWBEMAP to enable creation of mipmapped lwbemaps. If this flag is set, \p Width must be
 *     equal to \p Height, and \p Depth must be six. If the ::LWDA_ARRAY3D_LAYERED flag is also set,
 *     then \p Depth must be a multiple of six.
 *   - ::LWDA_ARRAY3D_TEXTURE_GATHER to indicate that the LWCA mipmapped array will be used for texture gather.
 *     Texture gather can only be performed on 2D LWCA mipmapped arrays.
 *
 * \p Width, \p Height and \p Depth must meet certain size requirements as listed in the following table. 
 * All values are specified in elements. Note that for brevity's sake, the full name of the device attribute 
 * is not specified. For ex., TEXTURE1D_MIPMAPPED_WIDTH refers to the device attribute 
 * ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH.
 *
 * <table>
 * <tr><td><b>LWCA array type</b></td>
 * <td><b>Valid extents that must always be met<br>{(width range in elements), (height range), 
 * (depth range)}</b></td>
 * <td><b>Valid extents with LWDA_ARRAY3D_SURFACE_LDST set<br> 
 * {(width range in elements), (height range), (depth range)}</b></td></tr>
 * <tr><td>1D</td>
 * <td><small>{ (1,TEXTURE1D_MIPMAPPED_WIDTH), 0, 0 }</small></td>
 * <td><small>{ (1,SURFACE1D_WIDTH), 0, 0 }</small></td></tr>
 * <tr><td>2D</td>
 * <td><small>{ (1,TEXTURE2D_MIPMAPPED_WIDTH), (1,TEXTURE2D_MIPMAPPED_HEIGHT), 0 }</small></td>
 * <td><small>{ (1,SURFACE2D_WIDTH), (1,SURFACE2D_HEIGHT), 0 }</small></td></tr>
 * <tr><td>3D</td>
 * <td><small>{ (1,TEXTURE3D_WIDTH), (1,TEXTURE3D_HEIGHT), (1,TEXTURE3D_DEPTH) }
 * <br>OR<br>{ (1,TEXTURE3D_WIDTH_ALTERNATE), (1,TEXTURE3D_HEIGHT_ALTERNATE), 
 * (1,TEXTURE3D_DEPTH_ALTERNATE) }</small></td>
 * <td><small>{ (1,SURFACE3D_WIDTH), (1,SURFACE3D_HEIGHT), 
 * (1,SURFACE3D_DEPTH) }</small></td></tr>
 * <tr><td>1D Layered</td>
 * <td><small>{ (1,TEXTURE1D_LAYERED_WIDTH), 0, 
 * (1,TEXTURE1D_LAYERED_LAYERS) }</small></td>
 * <td><small>{ (1,SURFACE1D_LAYERED_WIDTH), 0, 
 * (1,SURFACE1D_LAYERED_LAYERS) }</small></td></tr>
 * <tr><td>2D Layered</td>
 * <td><small>{ (1,TEXTURE2D_LAYERED_WIDTH), (1,TEXTURE2D_LAYERED_HEIGHT), 
 * (1,TEXTURE2D_LAYERED_LAYERS) }</small></td>
 * <td><small>{ (1,SURFACE2D_LAYERED_WIDTH), (1,SURFACE2D_LAYERED_HEIGHT), 
 * (1,SURFACE2D_LAYERED_LAYERS) }</small></td></tr>
 * <tr><td>Lwbemap</td>
 * <td><small>{ (1,TEXTURELWBEMAP_WIDTH), (1,TEXTURELWBEMAP_WIDTH), 6 }</small></td>
 * <td><small>{ (1,SURFACELWBEMAP_WIDTH), 
 * (1,SURFACELWBEMAP_WIDTH), 6 }</small></td></tr>
 * <tr><td>Lwbemap Layered</td>
 * <td><small>{ (1,TEXTURELWBEMAP_LAYERED_WIDTH), (1,TEXTURELWBEMAP_LAYERED_WIDTH), 
 * (1,TEXTURELWBEMAP_LAYERED_LAYERS) }</small></td>
 * <td><small>{ (1,SURFACELWBEMAP_LAYERED_WIDTH), (1,SURFACELWBEMAP_LAYERED_WIDTH), 
 * (1,SURFACELWBEMAP_LAYERED_LAYERS) }</small></td></tr>
 * </table>
 *
 *
 * \param pHandle             - Returned mipmapped array
 * \param pMipmappedArrayDesc - mipmapped array descriptor
 * \param numMipmapLevels     - Number of mipmap levels
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_OUT_OF_MEMORY,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::lwMipmappedArrayDestroy,
 * ::lwMipmappedArrayGetLevel,
 * ::lwArrayCreate,
 * ::lwdaMallocMipmappedArray
 */
LWresult LWDAAPI lwMipmappedArrayCreate(LWmipmappedArray *pHandle, const LWDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, unsigned int numMipmapLevels);

/**
 * \brief Gets a mipmap level of a LWCA mipmapped array
 *
 * Returns in \p *pLevelArray a LWCA array that represents a single mipmap level
 * of the LWCA mipmapped array \p hMipmappedArray.
 *
 * If \p level is greater than the maximum number of levels in this mipmapped array,
 * ::LWDA_ERROR_ILWALID_VALUE is returned.
 *
 * \param pLevelArray     - Returned mipmap level LWCA array
 * \param hMipmappedArray - LWCA mipmapped array
 * \param level           - Mipmap level
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_HANDLE
 * \notefnerr
 *
 * \sa
 * ::lwMipmappedArrayCreate,
 * ::lwMipmappedArrayDestroy,
 * ::lwArrayCreate,
 * ::lwdaGetMipmappedArrayLevel
 */
LWresult LWDAAPI lwMipmappedArrayGetLevel(LWarray *pLevelArray, LWmipmappedArray hMipmappedArray, unsigned int level);

/**
 * \brief Destroys a LWCA mipmapped array
 *
 * Destroys the LWCA mipmapped array \p hMipmappedArray.
 *
 * \param hMipmappedArray - Mipmapped array to destroy
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ARRAY_IS_MAPPED
 * \notefnerr
 *
 * \sa
 * ::lwMipmappedArrayCreate,
 * ::lwMipmappedArrayGetLevel,
 * ::lwArrayCreate,
 * ::lwdaFreeMipmappedArray
 */
LWresult LWDAAPI lwMipmappedArrayDestroy(LWmipmappedArray hMipmappedArray);

#endif /* __LWDA_API_VERSION >= 5000 */

/** @} */ /* END LWDA_MEM */

/**
 * \defgroup LWDA_UNIFIED Unified Addressing
 *
 * ___MANBRIEF___ unified addressing functions of the low-level LWCA driver
 * API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the unified addressing functions of the 
 * low-level LWCA driver application programming interface.
 *
 * @{
 *
 * \section LWDA_UNIFIED_overview Overview
 *
 * LWCA devices can share a unified address space with the host.  
 * For these devices there is no distinction between a device
 * pointer and a host pointer -- the same pointer value may be 
 * used to access memory from the host program and from a kernel 
 * running on the device (with exceptions enumerated below).
 *
 * \section LWDA_UNIFIED_support Supported Platforms
 * 
 * Whether or not a device supports unified addressing may be 
 * queried by calling ::lwDeviceGetAttribute() with the device 
 * attribute ::LW_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING.
 *
 * Unified addressing is automatically enabled in 64-bit processes 
 *
 * \section LWDA_UNIFIED_lookup Looking Up Information from Pointer Values
 *
 * It is possible to look up information about the memory which backs a 
 * pointer value.  For instance, one may want to know if a pointer points
 * to host or device memory.  As another example, in the case of device 
 * memory, one may want to know on which LWCA device the memory 
 * resides.  These properties may be queried using the function 
 * ::lwPointerGetAttribute()
 *
 * Since pointers are unique, it is not necessary to specify information
 * about the pointers specified to the various copy functions in the 
 * LWCA API.  The function ::lwMemcpy() may be used to perform a copy
 * between two pointers, ignoring whether they point to host or device
 * memory (making ::lwMemcpyHtoD(), ::lwMemcpyDtoD(), and ::lwMemcpyDtoH()
 * unnecessary for devices supporting unified addressing).  For 
 * multidimensional copies, the memory type ::LW_MEMORYTYPE_UNIFIED may be
 * used to specify that the LWCA driver should infer the location of the
 * pointer from its value.
 *
 * \section LWDA_UNIFIED_automaphost Automatic Mapping of Host Allocated Host Memory
 *
 * All host memory allocated in all contexts using ::lwMemAllocHost() and
 * ::lwMemHostAlloc() is always directly accessible from all contexts on
 * all devices that support unified addressing.  This is the case regardless 
 * of whether or not the flags ::LW_MEMHOSTALLOC_PORTABLE and
 * ::LW_MEMHOSTALLOC_DEVICEMAP are specified.
 *
 * The pointer value through which allocated host memory may be accessed 
 * in kernels on all devices that support unified addressing is the same 
 * as the pointer value through which that memory is accessed on the host,
 * so it is not necessary to call ::lwMemHostGetDevicePointer() to get the device 
 * pointer for these allocations.
 * 
 * Note that this is not the case for memory allocated using the flag
 * ::LW_MEMHOSTALLOC_WRITECOMBINED, as dislwssed below.
 *
 * \section LWDA_UNIFIED_autopeerregister Automatic Registration of Peer Memory
 *
 * Upon enabling direct access from a context that supports unified addressing 
 * to another peer context that supports unified addressing using 
 * ::lwCtxEnablePeerAccess() all memory allocated in the peer context using 
 * ::lwMemAlloc() and ::lwMemAllocPitch() will immediately be accessible 
 * by the current context.  The device pointer value through
 * which any peer memory may be accessed in the current context
 * is the same pointer value through which that memory may be
 * accessed in the peer context.
 *
 * \section LWDA_UNIFIED_exceptions Exceptions, Disjoint Addressing
 * 
 * Not all memory may be accessed on devices through the same pointer
 * value through which they are accessed on the host.  These exceptions
 * are host memory registered using ::lwMemHostRegister() and host memory
 * allocated using the flag ::LW_MEMHOSTALLOC_WRITECOMBINED.  For these 
 * exceptions, there exists a distinct host and device address for the
 * memory.  The device address is guaranteed to not overlap any valid host
 * pointer range and is guaranteed to have the same value across all 
 * contexts that support unified addressing.  
 * 
 * This device address may be queried using ::lwMemHostGetDevicePointer() 
 * when a context using unified addressing is current.  Either the host 
 * or the unified device pointer value may be used to refer to this memory 
 * through ::lwMemcpy() and similar functions using the 
 * ::LW_MEMORYTYPE_UNIFIED memory type.
 *
 */

#if __LWDA_API_VERSION >= 4000
/**
 * \brief Returns information about a pointer
 * 
 * The supported attributes are:
 * 
 * - ::LW_POINTER_ATTRIBUTE_CONTEXT: 
 * 
 *      Returns in \p *data the ::LWcontext in which \p ptr was allocated or 
 *      registered.   
 *      The type of \p data must be ::LWcontext *.  
 *      
 *      If \p ptr was not allocated by, mapped by, or registered with
 *      a ::LWcontext which uses unified virtual addressing then 
 *      ::LWDA_ERROR_ILWALID_VALUE is returned.
 * 
 * - ::LW_POINTER_ATTRIBUTE_MEMORY_TYPE: 
 *    
 *      Returns in \p *data the physical memory type of the memory that 
 *      \p ptr addresses as a ::LWmemorytype enumerated value.
 *      The type of \p data must be unsigned int.
 *      
 *      If \p ptr addresses device memory then \p *data is set to 
 *      ::LW_MEMORYTYPE_DEVICE.  The particular ::LWdevice on which the 
 *      memory resides is the ::LWdevice of the ::LWcontext returned by the 
 *      ::LW_POINTER_ATTRIBUTE_CONTEXT attribute of \p ptr.
 *      
 *      If \p ptr addresses host memory then \p *data is set to 
 *      ::LW_MEMORYTYPE_HOST.
 *      
 *      If \p ptr was not allocated by, mapped by, or registered with
 *      a ::LWcontext which uses unified virtual addressing then 
 *      ::LWDA_ERROR_ILWALID_VALUE is returned.
 *
 *      If the current ::LWcontext does not support unified virtual 
 *      addressing then ::LWDA_ERROR_ILWALID_CONTEXT is returned.
 *    
 * - ::LW_POINTER_ATTRIBUTE_DEVICE_POINTER:
 * 
 *      Returns in \p *data the device pointer value through which
 *      \p ptr may be accessed by kernels running in the current 
 *      ::LWcontext.
 *      The type of \p data must be LWdeviceptr *.
 * 
 *      If there exists no device pointer value through which
 *      kernels running in the current ::LWcontext may access
 *      \p ptr then ::LWDA_ERROR_ILWALID_VALUE is returned.
 * 
 *      If there is no current ::LWcontext then 
 *      ::LWDA_ERROR_ILWALID_CONTEXT is returned.
 *      
 *      Except in the exceptional disjoint addressing cases dislwssed 
 *      below, the value returned in \p *data will equal the input 
 *      value \p ptr.
 * 
 * - ::LW_POINTER_ATTRIBUTE_HOST_POINTER:
 * 
 *      Returns in \p *data the host pointer value through which 
 *      \p ptr may be accessed by by the host program.
 *      The type of \p data must be void **.
 *      If there exists no host pointer value through which
 *      the host program may directly access \p ptr then 
 *      ::LWDA_ERROR_ILWALID_VALUE is returned.
 * 
 *      Except in the exceptional disjoint addressing cases dislwssed 
 *      below, the value returned in \p *data will equal the input 
 *      value \p ptr.
 *
 * - ::LW_POINTER_ATTRIBUTE_P2P_TOKENS:
 *
 *      Returns in \p *data two tokens for use with the lw-p2p.h Linux
 *      kernel interface. \p data must be a struct of type
 *      LWDA_POINTER_ATTRIBUTE_P2P_TOKENS.
 *
 *      \p ptr must be a pointer to memory obtained from :lwMemAlloc().
 *      Note that p2pToken and vaSpaceToken are only valid for the
 *      lifetime of the source allocation. A subsequent allocation at
 *      the same address may return completely different tokens.
 *      Querying this attribute has a side effect of setting the attribute
 *      ::LW_POINTER_ATTRIBUTE_SYNC_MEMOPS for the region of memory that
 *      \p ptr points to.
 * 
 * - ::LW_POINTER_ATTRIBUTE_SYNC_MEMOPS:
 *
 *      A boolean attribute which when set, ensures that synchronous memory operations
 *      initiated on the region of memory that \p ptr points to will always synchronize.
 *      See further documentation in the section titled "API synchronization behavior"
 *      to learn more about cases when synchronous memory operations can
 *      exhibit asynchronous behavior.
 *
 * - ::LW_POINTER_ATTRIBUTE_BUFFER_ID:
 *
 *      Returns in \p *data a buffer ID which is guaranteed to be unique within the process.
 *      \p data must point to an unsigned long long.
 *
 *      \p ptr must be a pointer to memory obtained from a LWCA memory allocation API.
 *      Every memory allocation from any of the LWCA memory allocation APIs will
 *      have a unique ID over a process lifetime. Subsequent allocations do not reuse IDs
 *      from previous freed allocations. IDs are only unique within a single process.
 *
 *
 * - ::LW_POINTER_ATTRIBUTE_IS_MANAGED:
 *
 *      Returns in \p *data a boolean that indicates whether the pointer points to
 *      managed memory or not.
 *
 * \par
 *
 * Note that for most allocations in the unified virtual address space
 * the host and device pointer for accessing the allocation will be the 
 * same.  The exceptions to this are
 *  - user memory registered using ::lwMemHostRegister 
 *  - host memory allocated using ::lwMemHostAlloc with the 
 *    ::LW_MEMHOSTALLOC_WRITECOMBINED flag
 * For these types of allocation there will exist separate, disjoint host 
 * and device addresses for accessing the allocation.  In particular 
 *  - The host address will correspond to an invalid unmapped device address 
 *    (which will result in an exception if accessed from the device) 
 *  - The device address will correspond to an invalid unmapped host address 
 *    (which will result in an exception if accessed from the host).
 * For these types of allocations, querying ::LW_POINTER_ATTRIBUTE_HOST_POINTER 
 * and ::LW_POINTER_ATTRIBUTE_DEVICE_POINTER may be used to retrieve the host 
 * and device addresses from either address.
 *
 * \param data      - Returned pointer attribute value
 * \param attribute - Pointer attribute to query
 * \param ptr       - Pointer
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::lwPointerSetAttribute,
 * ::lwMemAlloc,
 * ::lwMemFree,
 * ::lwMemAllocHost,
 * ::lwMemFreeHost,
 * ::lwMemHostAlloc,
 * ::lwMemHostRegister,
 * ::lwMemHostUnregister,
 * ::lwdaPointerGetAttributes
 */
LWresult LWDAAPI lwPointerGetAttribute(void *data, LWpointer_attribute attribute, LWdeviceptr ptr);
#endif /* __LWDA_API_VERSION >= 4000 */

#if __LWDA_API_VERSION >= 8000
/**
 * \brief Prefetches memory to the specified destination device
 *
 * Prefetches memory to the specified destination device.  \p devPtr is the 
 * base device pointer of the memory to be prefetched and \p dstDevice is the 
 * destination device. \p count specifies the number of bytes to copy. \p hStream
 * is the stream in which the operation is enqueued. The memory range must refer
 * to managed memory allocated via ::lwMemAllocManaged or declared via __managed__ variables.
 *
 * Passing in LW_DEVICE_CPU for \p dstDevice will prefetch the data to host memory. If
 * \p dstDevice is a GPU, then the device attribute ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS
 * must be non-zero. Additionally, \p hStream must be associated with a device that has a
 * non-zero value for the device attribute ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS.
 *
 * The start address and end address of the memory range will be rounded down and rounded up
 * respectively to be aligned to CPU page size before the prefetch operation is enqueued
 * in the stream.
 *
 * If no physical memory has been allocated for this region, then this memory region
 * will be populated and mapped on the destination device. If there's insufficient
 * memory to prefetch the desired region, the Unified Memory driver may evict pages from other
 * ::lwMemAllocManaged allocations to host memory in order to make room. Device memory
 * allocated using ::lwMemAlloc or ::lwArrayCreate will not be evicted.
 *
 * By default, any mappings to the previous location of the migrated pages are removed and
 * mappings for the new location are only setup on \p dstDevice. The exact behavior however
 * also depends on the settings applied to this memory range via ::lwMemAdvise as described
 * below:
 *
 * If ::LW_MEM_ADVISE_SET_READ_MOSTLY was set on any subset of this memory range,
 * then that subset will create a read-only copy of the pages on \p dstDevice.
 *
 * If ::LW_MEM_ADVISE_SET_PREFERRED_LOCATION was called on any subset of this memory
 * range, then the pages will be migrated to \p dstDevice even if \p dstDevice is not the
 * preferred location of any pages in the memory range.
 *
 * If ::LW_MEM_ADVISE_SET_ACCESSED_BY was called on any subset of this memory range,
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
 * \param hStream    - Stream to enqueue prefetch operation
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwMemcpy, ::lwMemcpyPeer, ::lwMemcpyAsync,
 * ::lwMemcpy3DPeerAsync, ::lwMemAdvise,
 * ::lwdaMemPrefetchAsync
 */
LWresult LWDAAPI lwMemPrefetchAsync(LWdeviceptr devPtr, size_t count, LWdevice dstDevice, LWstream hStream);

/**
 * \brief Advise about the usage of a given memory range
 *
 * Advise the Unified Memory subsystem about the usage pattern for the memory range
 * starting at \p devPtr with a size of \p count bytes. The start address and end address of the memory
 * range will be rounded down and rounded up respectively to be aligned to CPU page size before the
 * advice is applied. The memory range must refer to managed memory allocated via ::lwMemAllocManaged
 * or declared via __managed__ variables.
 *
 * The \p advice parameter can take the following values:
 * - ::LW_MEM_ADVISE_SET_READ_MOSTLY: This implies that the data is mostly going to be read
 * from and only occasionally written to. Any read accesses from any processor to this region will create a
 * read-only copy of at least the accessed pages in that processor's memory. Additionally, if ::lwMemPrefetchAsync
 * is called on this region, it will create a read-only copy of the data on the destination processor.
 * If any processor writes to this region, all copies of the corresponding page will be ilwalidated
 * except for the one where the write oclwrred. The \p device argument is ignored for this advice.
 * Note that for a page to be read-duplicated, the accessing processor must either be the CPU or a GPU
 * that has a non-zero value for the device attribute ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS.
 * Also, if a context is created on a device that does not have the device attribute
 * ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS set, then read-duplication will not occur until
 * all such contexts are destroyed.
 * - ::LW_MEM_ADVISE_UNSET_READ_MOSTLY:  Undoes the effect of ::LW_MEM_ADVISE_SET_READ_MOSTLY and also prevents the
 * Unified Memory driver from attempting heuristic read-duplication on the memory range. Any read-duplicated
 * copies of the data will be collapsed into a single copy. The location for the collapsed
 * copy will be the preferred location if the page has a preferred location and one of the read-duplicated
 * copies was resident at that location. Otherwise, the location chosen is arbitrary.
 * - ::LW_MEM_ADVISE_SET_PREFERRED_LOCATION: This advice sets the preferred location for the
 * data to be the memory belonging to \p device. Passing in LW_DEVICE_CPU for \p device sets the
 * preferred location as host memory. If \p device is a GPU, then it must have a non-zero value for the
 * device attribute ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS. Setting the preferred location
 * does not cause data to migrate to that location immediately. Instead, it guides the migration policy
 * when a fault oclwrs on that memory region. If the data is already in its preferred location and the
 * faulting processor can establish a mapping without requiring the data to be migrated, then
 * data migration will be avoided. On the other hand, if the data is not in its preferred location
 * or if a direct mapping cannot be established, then it will be migrated to the processor accessing
 * it. It is important to note that setting the preferred location does not prevent data prefetching
 * done using ::lwMemPrefetchAsync.
 * Having a preferred location can override the page thrash detection and resolution logic in the Unified
 * Memory driver. Normally, if a page is detected to be constantly thrashing between for example host and device
 * memory, the page may eventually be pinned to host memory by the Unified Memory driver. But
 * if the preferred location is set as device memory, then the page will continue to thrash indefinitely.
 * If ::LW_MEM_ADVISE_SET_READ_MOSTLY is also set on this memory region or any subset of it, then the
 * policies associated with that advice will override the policies of this advice.
 * - ::LW_MEM_ADVISE_UNSET_PREFERRED_LOCATION: Undoes the effect of ::LW_MEM_ADVISE_SET_PREFERRED_LOCATION
 * and changes the preferred location to none.
 * - ::LW_MEM_ADVISE_SET_ACCESSED_BY: This advice implies that the data will be accessed by \p device.
 * Passing in ::LW_DEVICE_CPU for \p device will set the advice for the CPU. If \p device is a GPU, then
 * the device attribute ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS must be non-zero.
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
 * ::LW_MEM_ADVISE_SET_ACCESSED_BY flag set for this data will now have its mapping updated to point to the
 * page in host memory.
 * If ::LW_MEM_ADVISE_SET_READ_MOSTLY is also set on this memory region or any subset of it, then the
 * policies associated with that advice will override the policies of this advice. Additionally, if the
 * preferred location of this memory region or any subset of it is also \p device, then the policies
 * associated with ::LW_MEM_ADVISE_SET_PREFERRED_LOCATION will override the policies of this advice.
 * - ::LW_MEM_ADVISE_UNSET_ACCESSED_BY: Undoes the effect of ::LW_MEM_ADVISE_SET_ACCESSED_BY. Any mappings to
 * the data from \p device may be removed at any time causing accesses to result in non-fatal page faults.
 *
 * \param devPtr - Pointer to memory to set the advice for
 * \param count  - Size in bytes of the memory range
 * \param advice - Advice to be applied for the specified memory range
 * \param device - Device to apply the advice for
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwMemcpy, ::lwMemcpyPeer, ::lwMemcpyAsync,
 * ::lwMemcpy3DPeerAsync, ::lwMemPrefetchAsync,
 * ::lwdaMemAdvise
 */
LWresult LWDAAPI lwMemAdvise(LWdeviceptr devPtr, size_t count, LWmem_advise advice, LWdevice device);

/**
 * \brief Query an attribute of a given memory range
 * 
 * Query an attribute about the memory range starting at \p devPtr with a size of \p count bytes. The
 * memory range must refer to managed memory allocated via ::lwMemAllocManaged or declared via
 * __managed__ variables.
 *
 * The \p attribute parameter can take the following values:
 * - ::LW_MEM_RANGE_ATTRIBUTE_READ_MOSTLY: If this attribute is specified, \p data will be interpreted
 * as a 32-bit integer, and \p dataSize must be 4. The result returned will be 1 if all pages in the given
 * memory range have read-duplication enabled, or 0 otherwise.
 * - ::LW_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION: If this attribute is specified, \p data will be
 * interpreted as a 32-bit integer, and \p dataSize must be 4. The result returned will be a GPU device
 * id if all pages in the memory range have that GPU as their preferred location, or it will be LW_DEVICE_CPU
 * if all pages in the memory range have the CPU as their preferred location, or it will be LW_DEVICE_ILWALID
 * if either all the pages don't have the same preferred location or some of the pages don't have a
 * preferred location at all. Note that the actual location of the pages in the memory range at the time of
 * the query may be different from the preferred location. 
 * - ::LW_MEM_RANGE_ATTRIBUTE_ACCESSED_BY: If this attribute is specified, \p data will be interpreted
 * as an array of 32-bit integers, and \p dataSize must be a non-zero multiple of 4. The result returned
 * will be a list of device ids that had ::LW_MEM_ADVISE_SET_ACCESSED_BY set for that entire memory range.
 * If any device does not have that advice set for the entire memory range, that device will not be included.
 * If \p data is larger than the number of devices that have that advice set for that memory range,
 * LW_DEVICE_ILWALID will be returned in all the extra space provided. For ex., if \p dataSize is 12
 * (i.e. \p data has 3 elements) and only device 0 has the advice set, then the result returned will be
 * { 0, LW_DEVICE_ILWALID, LW_DEVICE_ILWALID }. If \p data is smaller than the number of devices that have
 * that advice set, then only as many devices will be returned as can fit in the array. There is no
 * guarantee on which specific devices will be returned, however.
 * - ::LW_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION: If this attribute is specified, \p data will be
 * interpreted as a 32-bit integer, and \p dataSize must be 4. The result returned will be the last location
 * to which all pages in the memory range were prefetched explicitly via ::lwMemPrefetchAsync. This will either be
 * a GPU id or LW_DEVICE_CPU depending on whether the last location for prefetch was a GPU or the CPU
 * respectively. If any page in the memory range was never explicitly prefetched or if all pages were not
 * prefetched to the same location, LW_DEVICE_ILWALID will be returned. Note that this simply returns the
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
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 * \note_async
 * \note_null_stream
 *
 * \sa ::lwMemRangeGetAttributes, ::lwMemPrefetchAsync,
 * ::lwMemAdvise,
 * ::lwdaMemRangeGetAttribute
 */
LWresult LWDAAPI lwMemRangeGetAttribute(void *data, size_t dataSize, LWmem_range_attribute attribute, LWdeviceptr devPtr, size_t count);

/**
 * \brief Query attributes of a given memory range.
 *
 * Query attributes of the memory range starting at \p devPtr with a size of \p count bytes. The
 * memory range must refer to managed memory allocated via ::lwMemAllocManaged or declared via
 * __managed__ variables. The \p attributes array will be interpreted to have \p numAttributes
 * entries. The \p dataSizes array will also be interpreted to have \p numAttributes entries.
 * The results of the query will be stored in \p data.
 *
 * The list of supported attributes are given below. Please refer to ::lwMemRangeGetAttribute for
 * attribute descriptions and restrictions.
 *
 * - ::LW_MEM_RANGE_ATTRIBUTE_READ_MOSTLY
 * - ::LW_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION
 * - ::LW_MEM_RANGE_ATTRIBUTE_ACCESSED_BY
 * - ::LW_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION
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
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 *
 * \sa ::lwMemRangeGetAttribute, ::lwMemAdvise
 * ::lwMemPrefetchAsync,
 * ::lwdaMemRangeGetAttributes
 */
LWresult LWDAAPI lwMemRangeGetAttributes(void **data, size_t *dataSizes, LWmem_range_attribute *attributes, size_t numAttributes, LWdeviceptr devPtr, size_t count);
#endif /* __LWDA_API_VERSION >= 8000 */

#if __LWDA_API_VERSION >= 6000
/**
 * \brief Set attributes on a previously allocated memory region
 *
 * The supported attributes are:
 *
 * - ::LW_POINTER_ATTRIBUTE_SYNC_MEMOPS:
 *
 *      A boolean attribute that can either be set (1) or unset (0). When set,
 *      the region of memory that \p ptr points to is guaranteed to always synchronize
 *      memory operations that are synchronous. If there are some previously initiated
 *      synchronous memory operations that are pending when this attribute is set, the
 *      function does not return until those memory operations are complete.
 *      See further documentation in the section titled "API synchronization behavior"
 *      to learn more about cases when synchronous memory operations can
 *      exhibit asynchronous behavior.
 *      \p value will be considered as a pointer to an unsigned integer to which this attribute is to be set.
 *
 * \param value     - Pointer to memory containing the value to be set
 * \param attribute - Pointer attribute to set
 * \param ptr       - Pointer to a memory region allocated using LWCA memory allocation APIs
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 *
 * \sa ::lwPointerGetAttribute,
 * ::lwPointerGetAttributes,
 * ::lwMemAlloc,
 * ::lwMemFree,
 * ::lwMemAllocHost,
 * ::lwMemFreeHost,
 * ::lwMemHostAlloc,
 * ::lwMemHostRegister,
 * ::lwMemHostUnregister
 */
LWresult LWDAAPI lwPointerSetAttribute(const void *value, LWpointer_attribute attribute, LWdeviceptr ptr);
#endif /* __LWDA_API_VERSION >= 6000 */

#if __LWDA_API_VERSION >= 7000
/**
 * \brief Returns information about a pointer.
 *
 * The supported attributes are (refer to ::lwPointerGetAttribute for attribute descriptions and restrictions):
 *
 * - ::LW_POINTER_ATTRIBUTE_CONTEXT
 * - ::LW_POINTER_ATTRIBUTE_MEMORY_TYPE
 * - ::LW_POINTER_ATTRIBUTE_DEVICE_POINTER
 * - ::LW_POINTER_ATTRIBUTE_HOST_POINTER
 * - ::LW_POINTER_ATTRIBUTE_SYNC_MEMOPS
 * - ::LW_POINTER_ATTRIBUTE_BUFFER_ID
 * - ::LW_POINTER_ATTRIBUTE_IS_MANAGED
 *
 * \param numAttributes - Number of attributes to query
 * \param attributes    - An array of attributes to query
 *                      (numAttributes and the number of attributes in this array should match)
 * \param data          - A two-dimensional array containing pointers to memory
 *                      locations where the result of each attribute query will be written to.
 * \param ptr           - Pointer to query
 *
 * Unlike ::lwPointerGetAttribute, this function will not return an error when the \p ptr
 * encountered is not a valid LWCA pointer. Instead, the attributes are assigned default NULL values
 * and LWDA_SUCCESS is returned.
 *
 * If \p ptr was not allocated by, mapped by, or registered with a ::LWcontext which uses UVA
 * (Unified Virtual Addressing), ::LWDA_ERROR_ILWALID_CONTEXT is returned.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::lwPointerGetAttribute,
 * ::lwPointerSetAttribute,
 * ::lwdaPointerGetAttributes
 */
LWresult LWDAAPI lwPointerGetAttributes(unsigned int numAttributes, LWpointer_attribute *attributes, void **data, LWdeviceptr ptr);
#endif /* __LWDA_API_VERSION >= 7000 */

/** @} */ /* END LWDA_UNIFIED */

/**
 * \defgroup LWDA_STREAM Stream Management
 *
 * ___MANBRIEF___ stream management functions of the low-level LWCA driver API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the stream management functions of the low-level LWCA
 * driver application programming interface.
 *
 * @{
 */

/**
 * \brief Create a stream
 *
 * Creates a stream and returns a handle in \p phStream.  The \p Flags argument
 * determines behaviors of the stream.  Valid values for \p Flags are:
 * - ::LW_STREAM_DEFAULT: Default stream creation flag.
 * - ::LW_STREAM_NON_BLOCKING: Specifies that work running in the created 
 *   stream may run conlwrrently with work in stream 0 (the NULL stream), and that
 *   the created stream should perform no implicit synchronization with stream 0.
 *
 * \param phStream - Returned newly created stream
 * \param Flags    - Parameters for stream creation
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
 * \sa ::lwStreamDestroy,
 * ::lwStreamCreateWithPriority,
 * ::lwStreamGetPriority,
 * ::lwStreamGetFlags,
 * ::lwStreamWaitEvent,
 * ::lwStreamQuery,
 * ::lwStreamSynchronize,
 * ::lwStreamAddCallback,
 * ::lwdaStreamCreate,
 * ::lwdaStreamCreateWithFlags
 */
LWresult LWDAAPI lwStreamCreate(LWstream *phStream, unsigned int Flags);

/**
 * \brief Create a stream with the given priority
 *
 * Creates a stream with the specified priority and returns a handle in \p phStream.
 * This API alters the scheduler priority of work in the stream. Work in a higher
 * priority stream may preempt work already exelwting in a low priority stream.
 *
 * \p priority follows a convention where lower numbers represent higher priorities.
 * '0' represents default priority. The range of meaningful numerical priorities can
 * be queried using ::lwCtxGetStreamPriorityRange. If the specified priority is
 * outside the numerical range returned by ::lwCtxGetStreamPriorityRange,
 * it will automatically be clamped to the lowest or the highest number in the range.
 *
 * \param phStream    - Returned newly created stream
 * \param flags       - Flags for stream creation. See ::lwStreamCreate for a list of
 *                      valid flags
 * \param priority    - Stream priority. Lower numbers represent higher priorities.
 *                      See ::lwCtxGetStreamPriorityRange for more information about
 *                      meaningful stream priorities that can be passed.
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
 * \note Stream priorities are supported only on GPUs
 * with compute capability 3.5 or higher.
 *
 * \note In the current implementation, only compute kernels launched in
 * priority streams are affected by the stream's priority. Stream priorities have
 * no effect on host-to-device and device-to-host memory operations.
 *
 * \sa ::lwStreamDestroy,
 * ::lwStreamCreate,
 * ::lwStreamGetPriority,
 * ::lwCtxGetStreamPriorityRange,
 * ::lwStreamGetFlags,
 * ::lwStreamWaitEvent,
 * ::lwStreamQuery,
 * ::lwStreamSynchronize,
 * ::lwStreamAddCallback,
 * ::lwdaStreamCreateWithPriority
 */
LWresult LWDAAPI lwStreamCreateWithPriority(LWstream *phStream, unsigned int flags, int priority);


/**
 * \brief Query the priority of a given stream
 *
 * Query the priority of a stream created using ::lwStreamCreate or ::lwStreamCreateWithPriority
 * and return the priority in \p priority. Note that if the stream was created with a
 * priority outside the numerical range returned by ::lwCtxGetStreamPriorityRange,
 * this function returns the clamped priority.
 * See ::lwStreamCreateWithPriority for details about priority clamping.
 *
 * \param hStream    - Handle to the stream to be queried
 * \param priority   - Pointer to a signed integer in which the stream's priority is returned
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa ::lwStreamDestroy,
 * ::lwStreamCreate,
 * ::lwStreamCreateWithPriority,
 * ::lwCtxGetStreamPriorityRange,
 * ::lwStreamGetFlags,
 * ::lwdaStreamGetPriority
 */
LWresult LWDAAPI lwStreamGetPriority(LWstream hStream, int *priority);

/**
 * \brief Query the flags of a given stream
 *
 * Query the flags of a stream created using ::lwStreamCreate or ::lwStreamCreateWithPriority
 * and return the flags in \p flags.
 *
 * \param hStream    - Handle to the stream to be queried
 * \param flags      - Pointer to an unsigned integer in which the stream's flags are returned
 *                     The value returned in \p flags is a logical 'OR' of all flags that
 *                     were used while creating this stream. See ::lwStreamCreate for the list
 *                     of valid flags
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa ::lwStreamDestroy,
 * ::lwStreamCreate,
 * ::lwStreamGetPriority,
 * ::lwdaStreamGetFlags
 */
LWresult LWDAAPI lwStreamGetFlags(LWstream hStream, unsigned int *flags);


/**
 * \brief Make a compute stream wait on an event
 *
 * Makes all future work submitted to \p hStream wait until \p hEvent
 * reports completion before beginning exelwtion.  This synchronization
 * will be performed efficiently on the device.  The event \p hEvent may
 * be from a different context than \p hStream, in which case this function
 * will perform cross-device synchronization.
 *
 * The stream \p hStream will wait only for the completion of the most recent
 * host call to ::lwEventRecord() on \p hEvent.  Once this call has returned,
 * any functions (including ::lwEventRecord() and ::lwEventDestroy()) may be
 * called on \p hEvent again, and subsequent calls will not have any
 * effect on \p hStream.
 *
 * If ::lwEventRecord() has not been called on \p hEvent, this call acts as if
 * the record has already completed, and so is a functional no-op.
 *
 * \param hStream - Stream to wait
 * \param hEvent  - Event to wait on (may not be NULL)
 * \param Flags   - Parameters for the operation (must be 0)
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwStreamCreate,
 * ::lwEventRecord,
 * ::lwStreamQuery,
 * ::lwStreamSynchronize,
 * ::lwStreamAddCallback,
 * ::lwStreamDestroy,
 * ::lwdaStreamWaitEvent
 */
LWresult LWDAAPI lwStreamWaitEvent(LWstream hStream, LWevent hEvent, unsigned int Flags);

/**
 * \brief Add a callback to a compute stream
 *
 * Adds a callback to be called on the host after all lwrrently enqueued
 * items in the stream have completed.  For each 
 * lwStreamAddCallback call, the callback will be exelwted exactly once.
 * The callback will block later work in the stream until it is finished.
 *
 * The callback may be passed ::LWDA_SUCCESS or an error code.  In the event
 * of a device error, all subsequently exelwted callbacks will receive an
 * appropriate ::LWresult.
 *
 * Callbacks must not make any LWCA API calls.  Attempting to use a LWCA API
 * will result in ::LWDA_ERROR_NOT_PERMITTED.  Callbacks must not perform any
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
 * \param hStream  - Stream to add callback to
 * \param callback - The function to call once preceding stream operations are complete
 * \param userData - User specified data to be passed to the callback function
 * \param flags    - Reserved for future use, must be 0
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_NOT_SUPPORTED
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwStreamCreate,
 * ::lwStreamQuery,
 * ::lwStreamSynchronize,
 * ::lwStreamWaitEvent,
 * ::lwStreamDestroy,
 * ::lwMemAllocManaged,
 * ::lwStreamAttachMemAsync,
 * ::lwdaStreamAddCallback
 */
LWresult LWDAAPI lwStreamAddCallback(LWstream hStream, LWstreamCallback callback, void *userData, unsigned int flags);

#if __LWDA_API_VERSION >= 6000

/**
 * \brief Attach memory to a stream asynchronously
 *
 * Enqueues an operation in \p hStream to specify stream association of
 * \p length bytes of memory starting from \p dptr. This function is a
 * stream-ordered operation, meaning that it is dependent on, and will
 * only take effect when, previous work in stream has completed. Any
 * previous association is automatically replaced.
 *
 * \p dptr must point to an address within managed memory space declared
 * using the __managed__ keyword or allocated with ::lwMemAllocManaged.
 *
 * \p length must be zero, to indicate that the entire allocation's
 * stream association is being changed. Lwrrently, it's not possible
 * to change stream association for a portion of an allocation.
 *
 * The stream association is specified using \p flags which must be
 * one of ::LWmemAttach_flags.
 * If the ::LW_MEM_ATTACH_GLOBAL flag is specified, the memory can be accessed
 * by any stream on any device.
 * If the ::LW_MEM_ATTACH_HOST flag is specified, the program makes a guarantee
 * that it won't access the memory on the device from any stream on a device that
 * has a zero value for the device attribute ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS.
 * If the ::LW_MEM_ATTACH_SINGLE flag is specified and \p hStream is associated with
 * a device that has a zero value for the device attribute ::LW_DEVICE_ATTRIBUTE_CONLWRRENT_MANAGED_ACCESS,
 * the program makes a guarantee that it will only access the memory on the device
 * from \p hStream. It is illegal to attach singly to the NULL stream, because the
 * NULL stream is a virtual global stream and not a specific stream. An error will
 * be returned in this case.
 *
 * When memory is associated with a single stream, the Unified Memory system will
 * allow CPU access to this memory region so long as all operations in \p hStream
 * have completed, regardless of whether other streams are active. In effect,
 * this constrains exclusive ownership of the managed memory region by
 * an active GPU to per-stream activity instead of whole-GPU activity.
 *
 * Accessing memory on the device from streams that are not associated with
 * it will produce undefined results. No error checking is performed by the
 * Unified Memory system to ensure that kernels launched into other streams
 * do not access this region. 
 *
 * It is a program's responsibility to order calls to ::lwStreamAttachMemAsync
 * via events, synchronization or other means to ensure legal access to memory
 * at all times. Data visibility and coherency will be changed appropriately
 * for all kernels which follow a stream-association change.
 *
 * If \p hStream is destroyed while data is associated with it, the association is
 * removed and the association reverts to the default visibility of the allocation
 * as specified at ::lwMemAllocManaged. For __managed__ variables, the default
 * association is always ::LW_MEM_ATTACH_GLOBAL. Note that destroying a stream is an
 * asynchronous operation, and as a result, the change to default association won't
 * happen until all work in the stream has completed.
 *
 * \param hStream - Stream in which to enqueue the attach operation
 * \param dptr    - Pointer to memory (must be a pointer to managed memory)
 * \param length  - Length of memory (must be zero)
 * \param flags   - Must be one of ::LWmemAttach_flags
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_NOT_SUPPORTED
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwStreamCreate,
 * ::lwStreamQuery,
 * ::lwStreamSynchronize,
 * ::lwStreamWaitEvent,
 * ::lwStreamDestroy,
 * ::lwMemAllocManaged,
 * ::lwdaStreamAttachMemAsync
 */
LWresult LWDAAPI lwStreamAttachMemAsync(LWstream hStream, LWdeviceptr dptr, size_t length, unsigned int flags);

#endif /* __LWDA_API_VERSION >= 6000 */

/**
 * \brief Determine status of a compute stream
 *
 * Returns ::LWDA_SUCCESS if all operations in the stream specified by
 * \p hStream have completed, or ::LWDA_ERROR_NOT_READY if not.
 *
 * For the purposes of Unified Memory, a return value of ::LWDA_SUCCESS
 * is equivalent to having called ::lwStreamSynchronize().
 *
 * \param hStream - Stream to query status of
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_NOT_READY
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwStreamCreate,
 * ::lwStreamWaitEvent,
 * ::lwStreamDestroy,
 * ::lwStreamSynchronize,
 * ::lwStreamAddCallback,
 * ::lwdaStreamQuery
 */
LWresult LWDAAPI lwStreamQuery(LWstream hStream);

/**
 * \brief Wait until a stream's tasks are completed
 *
 * Waits until the device has completed all operations in the stream specified
 * by \p hStream. If the context was created with the 
 * ::LW_CTX_SCHED_BLOCKING_SYNC flag, the CPU thread will block until the
 * stream is finished with all of its tasks.
 *
 * \param hStream - Stream to wait for
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwStreamCreate,
 * ::lwStreamDestroy,
 * ::lwStreamWaitEvent,
 * ::lwStreamQuery,
 * ::lwStreamAddCallback,
 * ::lwdaStreamSynchronize
 */
LWresult LWDAAPI lwStreamSynchronize(LWstream hStream);

#if __LWDA_API_VERSION >= 4000
/**
 * \brief Destroys a stream
 *
 * Destroys the stream specified by \p hStream.  
 *
 * In case the device is still doing work in the stream \p hStream
 * when ::lwStreamDestroy() is called, the function will return immediately 
 * and the resources associated with \p hStream will be released automatically 
 * once the device has completed all work in \p hStream.
 *
 * \param hStream - Stream to destroy
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwStreamCreate,
 * ::lwStreamWaitEvent,
 * ::lwStreamQuery,
 * ::lwStreamSynchronize,
 * ::lwStreamAddCallback,
 * ::lwdaStreamDestroy
 */
LWresult LWDAAPI lwStreamDestroy(LWstream hStream);
#endif /* __LWDA_API_VERSION >= 4000 */

/** @} */ /* END LWDA_STREAM */


/**
 * \defgroup LWDA_EVENT Event Management
 *
 * ___MANBRIEF___ event management functions of the low-level LWCA driver API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the event management functions of the low-level LWCA
 * driver application programming interface.
 *
 * @{
 */

/**
 * \brief Creates an event
 *
 * Creates an event *phEvent with the flags specified via \p Flags. Valid flags
 * include:
 * - ::LW_EVENT_DEFAULT: Default event creation flag.
 * - ::LW_EVENT_BLOCKING_SYNC: Specifies that the created event should use blocking
 *   synchronization.  A CPU thread that uses ::lwEventSynchronize() to wait on
 *   an event created with this flag will block until the event has actually
 *   been recorded.
 * - ::LW_EVENT_DISABLE_TIMING: Specifies that the created event does not need
 *   to record timing data.  Events created with this flag specified and
 *   the ::LW_EVENT_BLOCKING_SYNC flag not specified will provide the best
 *   performance when used with ::lwStreamWaitEvent() and ::lwEventQuery().
 * - ::LW_EVENT_INTERPROCESS: Specifies that the created event may be used as an
 *   interprocess event by ::lwIpcGetEventHandle(). ::LW_EVENT_INTERPROCESS must
 *   be specified along with ::LW_EVENT_DISABLE_TIMING.
 *
 * \param phEvent - Returns newly created event
 * \param Flags   - Event creation flags
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
 * \sa
 * ::lwEventRecord,
 * ::lwEventQuery,
 * ::lwEventSynchronize,
 * ::lwEventDestroy,
 * ::lwEventElapsedTime,
 * ::lwdaEventCreate,
 * ::lwdaEventCreateWithFlags
 */
LWresult LWDAAPI lwEventCreate(LWevent *phEvent, unsigned int Flags);

/**
 * \brief Records an event
 *
 * Records an event. See note on NULL stream behavior. Since operation is
 * asynchronous, ::lwEventQuery or ::lwEventSynchronize() must be used
 * to determine when the event has actually been recorded.
 *
 * If ::lwEventRecord() has previously been called on \p hEvent, then this
 * call will overwrite any existing state in \p hEvent.  Any subsequent calls
 * which examine the status of \p hEvent will only examine the completion of
 * this most recent call to ::lwEventRecord().
 *
 * It is necessary that \p hEvent and \p hStream be created on the same context.
 *
 * \param hEvent  - Event to record
 * \param hStream - Stream to record event for
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwEventCreate,
 * ::lwEventQuery,
 * ::lwEventSynchronize,
 * ::lwStreamWaitEvent,
 * ::lwEventDestroy,
 * ::lwEventElapsedTime,
 * ::lwdaEventRecord
 */
LWresult LWDAAPI lwEventRecord(LWevent hEvent, LWstream hStream);

/**
 * \brief Queries an event's status
 *
 * Query the status of all device work preceding the most recent
 * call to ::lwEventRecord() (in the appropriate compute streams,
 * as specified by the arguments to ::lwEventRecord()).
 *
 * If this work has successfully been completed by the device, or if
 * ::lwEventRecord() has not been called on \p hEvent, then ::LWDA_SUCCESS is
 * returned. If this work has not yet been completed by the device then
 * ::LWDA_ERROR_NOT_READY is returned.
 *
 * For the purposes of Unified Memory, a return value of ::LWDA_SUCCESS
 * is equivalent to having called ::lwEventSynchronize().
 *
 * \param hEvent - Event to query
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_NOT_READY
 * \notefnerr
 *
 * \sa ::lwEventCreate,
 * ::lwEventRecord,
 * ::lwEventSynchronize,
 * ::lwEventDestroy,
 * ::lwEventElapsedTime,
 * ::lwdaEventQuery
 */
LWresult LWDAAPI lwEventQuery(LWevent hEvent);

/**
 * \brief Waits for an event to complete
 *
 * Wait until the completion of all device work preceding the most recent
 * call to ::lwEventRecord() (in the appropriate compute streams, as specified
 * by the arguments to ::lwEventRecord()).
 *
 * If ::lwEventRecord() has not been called on \p hEvent, ::LWDA_SUCCESS is
 * returned immediately.
 *
 * Waiting for an event that was created with the ::LW_EVENT_BLOCKING_SYNC
 * flag will cause the calling CPU thread to block until the event has
 * been completed by the device.  If the ::LW_EVENT_BLOCKING_SYNC flag has
 * not been set, then the CPU thread will busy-wait until the event has
 * been completed by the device.
 *
 * \param hEvent - Event to wait for
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE
 * \notefnerr
 *
 * \sa ::lwEventCreate,
 * ::lwEventRecord,
 * ::lwEventQuery,
 * ::lwEventDestroy,
 * ::lwEventElapsedTime,
 * ::lwdaEventSynchronize
 */
LWresult LWDAAPI lwEventSynchronize(LWevent hEvent);

#if __LWDA_API_VERSION >= 4000
/**
 * \brief Destroys an event
 *
 * Destroys the event specified by \p hEvent.
 *
 * In case \p hEvent has been recorded but has not yet been completed
 * when ::lwEventDestroy() is called, the function will return immediately and 
 * the resources associated with \p hEvent will be released automatically once
 * the device has completed \p hEvent.
 *
 * \param hEvent - Event to destroy
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE
 * \notefnerr
 *
 * \sa ::lwEventCreate,
 * ::lwEventRecord,
 * ::lwEventQuery,
 * ::lwEventSynchronize,
 * ::lwEventElapsedTime,
 * ::lwdaEventDestroy
 */
LWresult LWDAAPI lwEventDestroy(LWevent hEvent);
#endif /* __LWDA_API_VERSION >= 4000 */

/**
 * \brief Computes the elapsed time between two events
 *
 * Computes the elapsed time between two events (in milliseconds with a
 * resolution of around 0.5 microseconds).
 *
 * If either event was last recorded in a non-NULL stream, the resulting time
 * may be greater than expected (even if both used the same stream handle). This
 * happens because the ::lwEventRecord() operation takes place asynchronously
 * and there is no guarantee that the measured latency is actually just between
 * the two events. Any number of other different stream operations could execute
 * in between the two measured events, thus altering the timing in a significant
 * way.
 *
 * If ::lwEventRecord() has not been called on either event then
 * ::LWDA_ERROR_ILWALID_HANDLE is returned. If ::lwEventRecord() has been called
 * on both events but one or both of them has not yet been completed (that is,
 * ::lwEventQuery() would return ::LWDA_ERROR_NOT_READY on at least one of the
 * events), ::LWDA_ERROR_NOT_READY is returned. If either event was created with
 * the ::LW_EVENT_DISABLE_TIMING flag, then this function will return
 * ::LWDA_ERROR_ILWALID_HANDLE.
 *
 * \param pMilliseconds - Time between \p hStart and \p hEnd in ms
 * \param hStart        - Starting event
 * \param hEnd          - Ending event
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_NOT_READY
 * \notefnerr
 *
 * \sa ::lwEventCreate,
 * ::lwEventRecord,
 * ::lwEventQuery,
 * ::lwEventSynchronize,
 * ::lwEventDestroy,
 * ::lwdaEventElapsedTime
 */
LWresult LWDAAPI lwEventElapsedTime(float *pMilliseconds, LWevent hStart, LWevent hEnd);

#if __LWDA_API_VERSION >= 8000
/**
 * \brief Wait on a memory location
 *
 * Enqueues a synchronization of the stream on the given memory location. Work
 * ordered after the operation will block until the given condition on the
 * memory is satisfied. By default, the condition is to wait for
 * (int32_t)(*addr - value) >= 0, a cyclic greater-or-equal.
 * Other condition types can be specified via \p flags.
 *
 * If the memory was registered via ::lwMemHostRegister(), the device pointer
 * should be obtained with ::lwMemHostGetDevicePointer(). This function cannot
 * be used with managed memory (::lwMemAllocManaged).
 *
 * Support for this can be queried with ::lwDeviceGetAttribute() and
 * ::LW_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS. The only requirement for basic
 * support is that on Windows, a device must be in TCC mode.
 *
 * \param stream The stream to synchronize on the memory location.
 * \param addr The memory location to wait on.
 * \param value The value to compare with the memory location.
 * \param flags See ::LWstreamWaitValue_flags.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_NOT_SUPPORTED
 * \notefnerr
 *
 * \sa ::lwStreamWaitValue64,
 * ::lwStreamWriteValue32,
 * ::lwStreamWriteValue64
 * ::lwStreamBatchMemOp,
 * ::lwMemHostRegister,
 * ::lwStreamWaitEvent
 */
LWresult LWDAAPI lwStreamWaitValue32(LWstream stream, LWdeviceptr addr, lwuint32_t value, unsigned int flags);

/**
 * \brief Wait on a memory location
 *
 * Enqueues a synchronization of the stream on the given memory location. Work
 * ordered after the operation will block until the given condition on the
 * memory is satisfied. By default, the condition is to wait for
 * (int64_t)(*addr - value) >= 0, a cyclic greater-or-equal.
 * Other condition types can be specified via \p flags.
 *
 * If the memory was registered via ::lwMemHostRegister(), the device pointer
 * should be obtained with ::lwMemHostGetDevicePointer().
 *
 * Support for this can be queried with ::lwDeviceGetAttribute() and
 * ::LW_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS. The requirements are
 * compute capability 7.0 or greater, and on Windows, that the device be in
 * TCC mode.
 *
 * \param stream The stream to synchronize on the memory location.
 * \param addr The memory location to wait on.
 * \param value The value to compare with the memory location.
 * \param flags See ::LWstreamWaitValue_flags.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_NOT_SUPPORTED
 * \notefnerr
 *
 * \sa ::lwStreamWaitValue32,
 * ::lwStreamWriteValue32,
 * ::lwStreamWriteValue64,
 * ::lwStreamBatchMemOp,
 * ::lwMemHostRegister,
 * ::lwStreamWaitEvent
 */
LWresult LWDAAPI lwStreamWaitValue64(LWstream stream, LWdeviceptr addr, lwuint64_t value, unsigned int flags);

/**
 * \brief Write a value to memory
 *
 * Write a value to memory. Unless the ::LW_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER
 * flag is passed, the write is preceded by a system-wide memory fence,
 * equivalent to a __threadfence_system() but scoped to the stream
 * rather than a LWCA thread.
 *
 * If the memory was registered via ::lwMemHostRegister(), the device pointer
 * should be obtained with ::lwMemHostGetDevicePointer(). This function cannot
 * be used with managed memory (::lwMemAllocManaged).
 *
 * Support for this can be queried with ::lwDeviceGetAttribute() and
 * ::LW_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS. The only requirement for basic
 * support is that on Windows, a device must be in TCC mode.
 *
 * \param stream The stream to do the write in.
 * \param addr The device address to write to.
 * \param value The value to write.
 * \param flags See ::LWstreamWriteValue_flags.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_NOT_SUPPORTED
 * \notefnerr
 *
 * \sa ::lwStreamWriteValue64,
 * ::lwStreamWaitValue32,
 * ::lwStreamWaitValue64,
 * ::lwStreamBatchMemOp,
 * ::lwMemHostRegister,
 * ::lwEventRecord
 */
LWresult LWDAAPI lwStreamWriteValue32(LWstream stream, LWdeviceptr addr, lwuint32_t value, unsigned int flags);

/**
 * \brief Write a value to memory
 *
 * Write a value to memory. Unless the ::LW_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER
 * flag is passed, the write is preceded by a system-wide memory fence,
 * equivalent to a __threadfence_system() but scoped to the stream
 * rather than a LWCA thread.
 *
 * If the memory was registered via ::lwMemHostRegister(), the device pointer
 * should be obtained with ::lwMemHostGetDevicePointer().
 *
 * Support for this can be queried with ::lwDeviceGetAttribute() and
 * ::LW_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS. The requirements are
 * compute capability 7.0 or greater, and on Windows, that the device be in
 * TCC mode.
 *
 * \param stream The stream to do the write in.
 * \param addr The device address to write to.
 * \param value The value to write.
 * \param flags See ::LWstreamWriteValue_flags.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_NOT_SUPPORTED
 * \notefnerr
 *
 * \sa ::lwStreamWriteValue32,
 * ::lwStreamWaitValue32,
 * ::lwStreamWaitValue64,
 * ::lwStreamBatchMemOp,
 * ::lwMemHostRegister,
 * ::lwEventRecord
 */
LWresult LWDAAPI lwStreamWriteValue64(LWstream stream, LWdeviceptr addr, lwuint64_t value, unsigned int flags);

/**
 * \brief Batch operations to synchronize the stream via memory operations
 *
 * This is a batch version of ::lwStreamWaitValue32() and ::lwStreamWriteValue32().
 * Batching operations may avoid some performance overhead in both the API call
 * and the device exelwtion versus adding them to the stream in separate API
 * calls. The operations are enqueued in the order they appear in the array.
 *
 * See ::LWstreamBatchMemOpType for the full set of supported operations, and
 * ::lwStreamWaitValue32(), ::lwStreamWaitValue64(), ::lwStreamWriteValue32(),
 * and ::lwStreamWriteValue64() for details of specific operations.
 *
 * Basic support for this can be queried with ::lwDeviceGetAttribute() and
 * ::LW_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS. See related APIs for details
 * on querying support for specific operations.
 *
 * \param stream The stream to enqueue the operations in.
 * \param count The number of operations in the array. Must be less than 256.
 * \param paramArray The types and parameters of the individual operations.
 * \param flags Reserved for future expansion; must be 0.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_NOT_SUPPORTED
 * \notefnerr
 *
 * \sa ::lwStreamWaitValue32,
 * ::lwStreamWaitValue64,
 * ::lwStreamWriteValue32,
 * ::lwStreamWriteValue64,
 * ::lwMemHostRegister
 */
LWresult LWDAAPI lwStreamBatchMemOp(LWstream stream, unsigned int count, LWstreamBatchMemOpParams *paramArray, unsigned int flags);
#endif /* __LWDA_API_VERSION >= 8000 */

/** @} */ /* END LWDA_EVENT */

/**
 * \defgroup LWDA_EXEC Exelwtion Control
 *
 * ___MANBRIEF___ exelwtion control functions of the low-level LWCA driver API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the exelwtion control functions of the low-level LWCA
 * driver application programming interface.
 *
 * @{
 */

/**
 * \brief Returns information about a function
 *
 * Returns in \p *pi the integer value of the attribute \p attrib on the kernel
 * given by \p hfunc. The supported attributes are:
 * - ::LW_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: The maximum number of threads
 *   per block, beyond which a launch of the function would fail. This number
 *   depends on both the function and the device on which the function is
 *   lwrrently loaded.
 * - ::LW_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: The size in bytes of
 *   statically-allocated shared memory per block required by this function.
 *   This does not include dynamically-allocated shared memory requested by
 *   the user at runtime.
 * - ::LW_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: The size in bytes of user-allocated
 *   constant memory required by this function.
 * - ::LW_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: The size in bytes of local memory
 *   used by each thread of this function.
 * - ::LW_FUNC_ATTRIBUTE_NUM_REGS: The number of registers used by each thread
 *   of this function.
 * - ::LW_FUNC_ATTRIBUTE_PTX_VERSION: The PTX virtual architecture version for
 *   which the function was compiled. This value is the major PTX version * 10
 *   + the minor PTX version, so a PTX version 1.3 function would return the
 *   value 13. Note that this may return the undefined value of 0 for lwbins
 *   compiled prior to LWCA 3.0.
 * - ::LW_FUNC_ATTRIBUTE_BINARY_VERSION: The binary architecture version for
 *   which the function was compiled. This value is the major binary
 *   version * 10 + the minor binary version, so a binary version 1.3 function
 *   would return the value 13. Note that this will return a value of 10 for
 *   legacy lwbins that do not have a properly-encoded binary architecture
 *   version.
 * - ::LW_FUNC_CACHE_MODE_CA: The attribute to indicate whether the function has  
 *   been compiled with user specified option "-Xptxas --dlcm=ca" set .
 * - ::LW_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: The maximum size in bytes of
 *   dynamically-allocated shared memory. 
 * - ::LW_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: Preferred shared memory-L1 
 *   cache split ratio in percent of shared memory.
 *
 * \param pi     - Returned attribute value
 * \param attrib - Attribute requested
 * \param hfunc  - Function to query attribute of
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwCtxGetCacheConfig,
 * ::lwCtxSetCacheConfig,
 * ::lwFuncSetCacheConfig,
 * ::lwLaunchKernel,
 * ::lwdaFuncGetAttributes
 * ::lwdaFuncSetAttribute
 */
LWresult LWDAAPI lwFuncGetAttribute(int *pi, LWfunction_attribute attrib, LWfunction hfunc);

#if __LWDA_API_VERSION >= 9000

/**
 * \brief Sets information about a function
 *
 * This call sets the value of a specified attribute \p attrib on the kernel given
 * by \p hfunc to an integer value specified by \p val
 * This function returns LWDA_SUCCESS if the new value of the attribute could be
 * successfully set. If the set fails, this call will return an error.
 * Not all attributes can have values set. Attempting to set a value on a read-only
 * attribute will result in an error (LWDA_ERROR_ILWALID_VALUE)
 *
 * Supported attributes for the lwFuncSetAttribute call are:
 * - ::LW_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: This maximum size in bytes of
 *   dynamically-allocated shared memory. The value should contain the requested
 *   maximum size of dynamically-allocated shared memory. The sum of this value and
 *   the function attribute ::LW_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES cannot exceed the
 *   device attribute ::LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN.
 *   The maximal size of requestable dynamic shared memory may differ by GPU
 *   architecture.
 * - ::LW_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: On devices where the L1 
 *   cache and shared memory use the same hardware resources, this sets the shared memory
 *   carveout preference, in percent of the total resources. This is only a hint, and the
 *   driver can choose a different ratio if required to execute the function.
 *
 * \param hfunc  - Function to query attribute of
 * \param attrib - Attribute requested
 * \param value   - The value to set
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwCtxGetCacheConfig,
 * ::lwCtxSetCacheConfig,
 * ::lwFuncSetCacheConfig,
 * ::lwLaunchKernel,
 * ::lwdaFuncGetAttributes
 * ::lwdaFuncSetAttribute
 */
LWresult LWDAAPI lwFuncSetAttribute(LWfunction hfunc, LWfunction_attribute attrib, int value);
#endif // __LWDA_API_VERSION >= 9000

/**
 * \brief Sets the preferred cache configuration for a device function
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p config the preferred cache configuration for
 * the device function \p hfunc. This is only a preference. The driver will use
 * the requested configuration if possible, but it is free to choose a different
 * configuration if required to execute \p hfunc.  Any context-wide preference
 * set via ::lwCtxSetCacheConfig() will be overridden by this per-function
 * setting unless the per-function setting is ::LW_FUNC_CACHE_PREFER_NONE. In
 * that case, the current context-wide setting will be used.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 *
 * The supported cache configurations are:
 * - ::LW_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)
 * - ::LW_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache
 * - ::LW_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory
 * - ::LW_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory
 *
 * \param hfunc  - Kernel to configure cache for
 * \param config - Requested cache configuration
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT
 * \notefnerr
 *
 * \sa ::lwCtxGetCacheConfig,
 * ::lwCtxSetCacheConfig,
 * ::lwFuncGetAttribute,
 * ::lwLaunchKernel,
 * ::lwdaFuncSetCacheConfig
 */
LWresult LWDAAPI lwFuncSetCacheConfig(LWfunction hfunc, LWfunc_cache config);

#if __LWDA_API_VERSION >= 4020
/**
 * \brief Sets the shared memory configuration for a device function.
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
 * ::lwFuncSetSharedMemConfig will override the context wide setting set with
 * ::lwCtxSetSharedMemConfig.
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
 * - ::LW_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE: use the context's shared memory 
 *   configuration when launching this function.
 * - ::LW_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: set shared memory bank width to
 *   be natively four bytes when launching this function.
 * - ::LW_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: set shared memory bank width to
 *   be natively eight bytes when launching this function.
 *
 * \param hfunc  - kernel to be given a shared memory config
 * \param config - requested shared memory configuration
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT
 * \notefnerr
 *
 * \sa ::lwCtxGetCacheConfig,
 * ::lwCtxSetCacheConfig,
 * ::lwCtxGetSharedMemConfig,
 * ::lwCtxSetSharedMemConfig,
 * ::lwFuncGetAttribute,
 * ::lwLaunchKernel,
 * ::lwdaFuncSetSharedMemConfig
 */
LWresult LWDAAPI lwFuncSetSharedMemConfig(LWfunction hfunc, LWsharedconfig config);
#endif

#if __LWDA_API_VERSION >= 4000
/**
 * \brief Launches a LWCA function
 *
 * Ilwokes the kernel \p f on a \p gridDimX x \p gridDimY x \p gridDimZ
 * grid of blocks. Each block contains \p blockDimX x \p blockDimY x
 * \p blockDimZ threads.
 *
 * \p sharedMemBytes sets the amount of dynamic shared memory that will be
 * available to each thread block.
 *
 * Kernel parameters to \p f can be specified in one of two ways:
 *
 * 1) Kernel parameters can be specified via \p kernelParams.  If \p f
 * has N parameters, then \p kernelParams needs to be an array of N
 * pointers.  Each of \p kernelParams[0] through \p kernelParams[N-1]
 * must point to a region of memory from which the actual kernel
 * parameter will be copied.  The number of kernel parameters and their
 * offsets and sizes do not need to be specified as that information is
 * retrieved directly from the kernel's image.
 *
 * 2) Kernel parameters can also be packaged by the application into
 * a single buffer that is passed in via the \p extra parameter.
 * This places the burden on the application of knowing each kernel
 * parameter's size and alignment/padding within the buffer.  Here is
 * an example of using the \p extra parameter in this manner:
 * \code
    size_t argBufferSize;
    char argBuffer[256];

    // populate argBuffer and argBufferSize

    void *config[] = {
        LW_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
        LW_LAUNCH_PARAM_BUFFER_SIZE,    &argBufferSize,
        LW_LAUNCH_PARAM_END
    };
    status = lwLaunchKernel(f, gx, gy, gz, bx, by, bz, sh, s, NULL, config);
 * \endcode
 *
 * The \p extra parameter exists to allow ::lwLaunchKernel to take
 * additional less commonly used arguments.  \p extra specifies a list of
 * names of extra settings and their corresponding values.  Each extra
 * setting name is immediately followed by the corresponding value.  The
 * list must be terminated with either NULL or ::LW_LAUNCH_PARAM_END.
 *
 * - ::LW_LAUNCH_PARAM_END, which indicates the end of the \p extra
 *   array;
 * - ::LW_LAUNCH_PARAM_BUFFER_POINTER, which specifies that the next
 *   value in \p extra will be a pointer to a buffer containing all
 *   the kernel parameters for launching kernel \p f;
 * - ::LW_LAUNCH_PARAM_BUFFER_SIZE, which specifies that the next
 *   value in \p extra will be a pointer to a size_t containing the
 *   size of the buffer specified with ::LW_LAUNCH_PARAM_BUFFER_POINTER;
 *
 * The error ::LWDA_ERROR_ILWALID_VALUE will be returned if kernel
 * parameters are specified with both \p kernelParams and \p extra
 * (i.e. both \p kernelParams and \p extra are non-NULL).
 *
 * Calling ::lwLaunchKernel() sets persistent function state that is
 * the same as function state set through the following deprecated APIs:
 *  ::lwFuncSetBlockShape(),
 *  ::lwFuncSetSharedSize(),
 *  ::lwParamSetSize(),
 *  ::lwParamSeti(),
 *  ::lwParamSetf(),
 *  ::lwParamSetv().
 *
 * When the kernel \p f is launched via ::lwLaunchKernel(), the previous
 * block shape, shared size and parameter info associated with \p f
 * is overwritten.
 *
 * Note that to use ::lwLaunchKernel(), the kernel \p f must either have
 * been compiled with toolchain version 3.2 or later so that it will
 * contain kernel parameter information, or have no kernel parameters.
 * If either of these conditions is not met, then ::lwLaunchKernel() will
 * return ::LWDA_ERROR_ILWALID_IMAGE.
 *
 * \param f              - Kernel to launch
 * \param gridDimX       - Width of grid in blocks
 * \param gridDimY       - Height of grid in blocks
 * \param gridDimZ       - Depth of grid in blocks
 * \param blockDimX      - X dimension of each thread block
 * \param blockDimY      - Y dimension of each thread block
 * \param blockDimZ      - Z dimension of each thread block
 * \param sharedMemBytes - Dynamic shared-memory size per thread block in bytes
 * \param hStream        - Stream identifier
 * \param kernelParams   - Array of pointers to kernel parameters
 * \param extra          - Extra options
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_IMAGE,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_LAUNCH_FAILED,
 * ::LWDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
 * ::LWDA_ERROR_LAUNCH_TIMEOUT,
 * ::LWDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
 * ::LWDA_ERROR_SHARED_OBJECT_INIT_FAILED
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwCtxGetCacheConfig,
 * ::lwCtxSetCacheConfig,
 * ::lwFuncSetCacheConfig,
 * ::lwFuncGetAttribute,
 * ::lwdaLaunchKernel
 */
LWresult LWDAAPI lwLaunchKernel(LWfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                LWstream hStream,
                                void **kernelParams,
                                void **extra);
#endif /* __LWDA_API_VERSION >= 4000 */
#if __LWDA_API_VERSION >= 9000
/**
 * \brief Launches a LWCA function where thread blocks can cooperate and synchronize as they execute
 *
 * Ilwokes the kernel \p f on a \p gridDimX x \p gridDimY x \p gridDimZ
 * grid of blocks. Each block contains \p blockDimX x \p blockDimY x
 * \p blockDimZ threads.
 *
 * \p sharedMemBytes sets the amount of dynamic shared memory that will be
 * available to each thread block.
 *
 * The device on which this kernel is ilwoked must have a non-zero value for
 * the device attribute ::LW_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH.
 *
 * The total number of blocks launched cannot exceed the maximum number of blocks per
 * multiprocessor as returned by ::lwOclwpancyMaxActiveBlocksPerMultiprocessor (or
 * ::lwOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors
 * as specified by the device attribute ::LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
 *
 * The kernel cannot make use of LWCA dynamic parallelism.
 *
 * Kernel parameters must be specified via \p kernelParams.  If \p f
 * has N parameters, then \p kernelParams needs to be an array of N
 * pointers.  Each of \p kernelParams[0] through \p kernelParams[N-1]
 * must point to a region of memory from which the actual kernel
 * parameter will be copied.  The number of kernel parameters and their
 * offsets and sizes do not need to be specified as that information is
 * retrieved directly from the kernel's image.
 *
 * Calling ::lwLaunchCooperativeKernel() sets persistent function state that is
 * the same as function state set through ::lwLaunchKernel API
 *
 * When the kernel \p f is launched via ::lwLaunchCooperativeKernel(), the previous
 * block shape, shared size and parameter info associated with \p f
 * is overwritten.
 *
 * Note that to use ::lwLaunchCooperativeKernel(), the kernel \p f must either have
 * been compiled with toolchain version 3.2 or later so that it will
 * contain kernel parameter information, or have no kernel parameters.
 * If either of these conditions is not met, then ::lwLaunchCooperativeKernel() will
 * return ::LWDA_ERROR_ILWALID_IMAGE.
 *
 * \param f              - Kernel to launch
 * \param gridDimX       - Width of grid in blocks
 * \param gridDimY       - Height of grid in blocks
 * \param gridDimZ       - Depth of grid in blocks
 * \param blockDimX      - X dimension of each thread block
 * \param blockDimY      - Y dimension of each thread block
 * \param blockDimZ      - Z dimension of each thread block
 * \param sharedMemBytes - Dynamic shared-memory size per thread block in bytes
 * \param hStream        - Stream identifier
 * \param kernelParams   - Array of pointers to kernel parameters
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_IMAGE,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_LAUNCH_FAILED,
 * ::LWDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
 * ::LWDA_ERROR_LAUNCH_TIMEOUT,
 * ::LWDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
 * ::LWDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
 * ::LWDA_ERROR_SHARED_OBJECT_INIT_FAILED
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwCtxGetCacheConfig,
 * ::lwCtxSetCacheConfig,
 * ::lwFuncSetCacheConfig,
 * ::lwFuncGetAttribute,
 * ::lwLaunchCooperativeKernelMultiDevice,
 * ::lwdaLaunchCooperativeKernel
 */
LWresult LWDAAPI lwLaunchCooperativeKernel(LWfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                LWstream hStream,
                                void **kernelParams);

/**
 * \brief Launches LWCA functions on multiple devices where thread blocks can cooperate and synchronize as they execute
 *
 * Ilwokes kernels as specified in the \p launchParamsList array where each element
 * of the array specifies all the parameters required to perform a single kernel launch.
 * These kernels can cooperate and synchronize as they execute. The size of the array is
 * specified by \p numDevices.
 *
 * No two kernels can be launched on the same device. All the devices targeted by this
 * multi-device launch must be identical. All devices must have a non-zero value for the
 * device attribute ::LW_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH.
 * 
 * All kernels launched must be identical with respect to the compiled code. Note that
 * any __device__, __constant__ or __managed__ variables present in the module that owns
 * the kernel launched on each device, are independently instantiated on every device.
 * It is the application's responsiblity to ensure these variables are initialized and
 * used appropriately.
 *
 * The size of the grids as specified in blocks, the size of the blocks themselves
 * and the amount of shared memory used by each thread block must also match across
 * all launched kernels.
 *
 * The streams used to launch these kernels must have been created via either ::lwStreamCreate
 * or ::lwStreamCreateWithPriority. The NULL stream or ::LW_STREAM_LEGACY or ::LW_STREAM_PER_THREAD
 * cannot be used.
 *
 * The total number of blocks launched per kernel cannot exceed the maximum number of blocks
 * per multiprocessor as returned by ::lwOclwpancyMaxActiveBlocksPerMultiprocessor (or
 * ::lwOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors
 * as specified by the device attribute ::LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT. Since the
 * total number of blocks launched per device has to match across all devices, the maximum
 * number of blocks that can be launched per device will be limited by the device with the
 * least number of multiprocessors.
 *
 * The kernels cannot make use of LWCA dynamic parallelism.
 *
 * The ::LWDA_LAUNCH_PARAMS structure is defined as:
 * \code
        typedef struct LWDA_LAUNCH_PARAMS_st
        {
            LWfunction function;
            unsigned int gridDimX;
            unsigned int gridDimY;
            unsigned int gridDimZ;
            unsigned int blockDimX;
            unsigned int blockDimY;
            unsigned int blockDimZ;
            unsigned int sharedMemBytes;
            LWstream hStream;
            void **kernelParams;
        } LWDA_LAUNCH_PARAMS;
 * \endcode
 * where:
 * - ::LWDA_LAUNCH_PARAMS::function specifies the kernel to be launched. All functions must
 *   be identical with respect to the compiled code.
 * - ::LWDA_LAUNCH_PARAMS::gridDimX is the width of the grid in blocks. This must match across
 *   all kernels launched.
 * - ::LWDA_LAUNCH_PARAMS::gridDimY is the height of the grid in blocks. This must match across
 *   all kernels launched.
 * - ::LWDA_LAUNCH_PARAMS::gridDimZ is the depth of the grid in blocks. This must match across
 *   all kernels launched.
 * - ::LWDA_LAUNCH_PARAMS::blockDimX is the X dimension of each thread block. This must match across
 *   all kernels launched.
 * - ::LWDA_LAUNCH_PARAMS::blockDimX is the Y dimension of each thread block. This must match across
 *   all kernels launched.
 * - ::LWDA_LAUNCH_PARAMS::blockDimZ is the Z dimension of each thread block. This must match across
 *   all kernels launched.
 * - ::LWDA_LAUNCH_PARAMS::sharedMemBytes is the dynamic shared-memory size per thread block in bytes.
 *   This must match across all kernels launched.
 * - ::LWDA_LAUNCH_PARAMS::hStream is the handle to the stream to perform the launch in. This cannot
 *   be the NULL stream or ::LW_STREAM_LEGACY or ::LW_STREAM_PER_THREAD. The LWCA context associated
 *   with this stream must match that associated with ::LWDA_LAUNCH_PARAMS::function.
 * - ::LWDA_LAUNCH_PARAMS::kernelParams is an array of pointers to kernel parameters. If
 *   ::LWDA_LAUNCH_PARAMS::function has N parameters, then ::LWDA_LAUNCH_PARAMS::kernelParams
 *   needs to be an array of N pointers. Each of ::LWDA_LAUNCH_PARAMS::kernelParams[0] through
 *   ::LWDA_LAUNCH_PARAMS::kernelParams[N-1] must point to a region of memory from which the actual
 *   kernel parameter will be copied. The number of kernel parameters and their offsets and sizes
 *   do not need to be specified as that information is retrieved directly from the kernel's image.
 *
 * By default, the kernel won't begin exelwtion on any GPU until all prior work in all the specified
 * streams has completed. This behavior can be overridden by specifying the flag
 * ::LWDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC. When this flag is specified, each kernel
 * will only wait for prior work in the stream corresponding to that GPU to complete before it begins
 * exelwtion.
 *
 * Similarly, by default, any subsequent work pushed in any of the specified streams will not begin
 * exelwtion until the kernels on all GPUs have completed. This behavior can be overridden by specifying
 * the flag ::LWDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC. When this flag is specified,
 * any subsequent work pushed in any of the specified streams will only wait for the kernel launched
 * on the GPU corresponding to that stream to complete before it begins exelwtion.
 *
 * Calling ::lwLaunchCooperativeKernelMultiDevice() sets persistent function state that is
 * the same as function state set through ::lwLaunchKernel API when called individually for each
 * element in \p launchParamsList.
 *
 * When kernels are launched via ::lwLaunchCooperativeKernelMultiDevice(), the previous
 * block shape, shared size and parameter info associated with each ::LWDA_LAUNCH_PARAMS::function
 * in \p launchParamsList is overwritten.
 *
 * Note that to use ::lwLaunchCooperativeKernelMultiDevice(), the kernels must either have
 * been compiled with toolchain version 3.2 or later so that it will
 * contain kernel parameter information, or have no kernel parameters.
 * If either of these conditions is not met, then ::lwLaunchCooperativeKernelMultiDevice() will
 * return ::LWDA_ERROR_ILWALID_IMAGE.
 *
 * \param launchParamsList - List of launch parameters, one per device
 * \param numDevices       - Size of the \p launchParamsList array
 * \param flags            - Flags to control launch behavior
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_IMAGE,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_LAUNCH_FAILED,
 * ::LWDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
 * ::LWDA_ERROR_LAUNCH_TIMEOUT,
 * ::LWDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
 * ::LWDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
 * ::LWDA_ERROR_SHARED_OBJECT_INIT_FAILED
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwCtxGetCacheConfig,
 * ::lwCtxSetCacheConfig,
 * ::lwFuncSetCacheConfig,
 * ::lwFuncGetAttribute,
 * ::lwLaunchCooperativeKernel,
 * ::lwdaLaunchCooperativeKernelMultiDevice
 */
LWresult LWDAAPI lwLaunchCooperativeKernelMultiDevice(LWDA_LAUNCH_PARAMS *launchParamsList, unsigned int numDevices, unsigned int flags);

#endif /* __LWDA_API_VERSION >= 9000 */

/** @} */ /* END LWDA_EXEC */

/**
 * \defgroup LWDA_EXEC_DEPRECATED Exelwtion Control [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated exelwtion control functions of the low-level LWCA
 * driver API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the deprecated exelwtion control functions of the
 * low-level LWCA driver application programming interface.
 *
 * @{
 */

/**
 * \brief Sets the block-dimensions for the function
 *
 * \deprecated
 *
 * Specifies the \p x, \p y, and \p z dimensions of the thread blocks that are
 * created when the kernel given by \p hfunc is launched.
 *
 * \param hfunc - Kernel to specify dimensions of
 * \param x     - X dimension
 * \param y     - Y dimension
 * \param z     - Z dimension
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwFuncSetSharedSize,
 * ::lwFuncSetCacheConfig,
 * ::lwFuncGetAttribute,
 * ::lwParamSetSize,
 * ::lwParamSeti,
 * ::lwParamSetf,
 * ::lwParamSetv,
 * ::lwLaunch,
 * ::lwLaunchGrid,
 * ::lwLaunchGridAsync,
 * ::lwLaunchKernel
 */
LWresult LWDAAPI lwFuncSetBlockShape(LWfunction hfunc, int x, int y, int z);

/**
 * \brief Sets the dynamic shared-memory size for the function
 *
 * \deprecated
 *
 * Sets through \p bytes the amount of dynamic shared memory that will be
 * available to each thread block when the kernel given by \p hfunc is launched.
 *
 * \param hfunc - Kernel to specify dynamic shared-memory size for
 * \param bytes - Dynamic shared-memory size per thread in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwFuncSetBlockShape,
 * ::lwFuncSetCacheConfig,
 * ::lwFuncGetAttribute,
 * ::lwParamSetSize,
 * ::lwParamSeti,
 * ::lwParamSetf,
 * ::lwParamSetv,
 * ::lwLaunch,
 * ::lwLaunchGrid,
 * ::lwLaunchGridAsync,
 * ::lwLaunchKernel
 */
LWresult LWDAAPI lwFuncSetSharedSize(LWfunction hfunc, unsigned int bytes);

/**
 * \brief Sets the parameter size for the function
 *
 * \deprecated
 *
 * Sets through \p numbytes the total size in bytes needed by the function
 * parameters of the kernel corresponding to \p hfunc.
 *
 * \param hfunc    - Kernel to set parameter size for
 * \param numbytes - Size of parameter list in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwFuncSetBlockShape,
 * ::lwFuncSetSharedSize,
 * ::lwFuncGetAttribute,
 * ::lwParamSetf,
 * ::lwParamSeti,
 * ::lwParamSetv,
 * ::lwLaunch,
 * ::lwLaunchGrid,
 * ::lwLaunchGridAsync,
 * ::lwLaunchKernel
 */
LWresult LWDAAPI lwParamSetSize(LWfunction hfunc, unsigned int numbytes);

/**
 * \brief Adds an integer parameter to the function's argument list
 *
 * \deprecated
 *
 * Sets an integer parameter that will be specified the next time the
 * kernel corresponding to \p hfunc will be ilwoked. \p offset is a byte offset.
 *
 * \param hfunc  - Kernel to add parameter to
 * \param offset - Offset to add parameter to argument list
 * \param value  - Value of parameter
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwFuncSetBlockShape,
 * ::lwFuncSetSharedSize,
 * ::lwFuncGetAttribute,
 * ::lwParamSetSize,
 * ::lwParamSetf,
 * ::lwParamSetv,
 * ::lwLaunch,
 * ::lwLaunchGrid,
 * ::lwLaunchGridAsync,
 * ::lwLaunchKernel
 */
LWresult LWDAAPI lwParamSeti(LWfunction hfunc, int offset, unsigned int value);

/**
 * \brief Adds a floating-point parameter to the function's argument list
 *
 * \deprecated
 *
 * Sets a floating-point parameter that will be specified the next time the
 * kernel corresponding to \p hfunc will be ilwoked. \p offset is a byte offset.
 *
 * \param hfunc  - Kernel to add parameter to
 * \param offset - Offset to add parameter to argument list
 * \param value  - Value of parameter
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwFuncSetBlockShape,
 * ::lwFuncSetSharedSize,
 * ::lwFuncGetAttribute,
 * ::lwParamSetSize,
 * ::lwParamSeti,
 * ::lwParamSetv,
 * ::lwLaunch,
 * ::lwLaunchGrid,
 * ::lwLaunchGridAsync,
 * ::lwLaunchKernel
 */
LWresult LWDAAPI lwParamSetf(LWfunction hfunc, int offset, float value);

/**
 * \brief Adds arbitrary data to the function's argument list
 *
 * \deprecated
 *
 * Copies an arbitrary amount of data (specified in \p numbytes) from \p ptr
 * into the parameter space of the kernel corresponding to \p hfunc. \p offset
 * is a byte offset.
 *
 * \param hfunc    - Kernel to add data to
 * \param offset   - Offset to add data to argument list
 * \param ptr      - Pointer to arbitrary data
 * \param numbytes - Size of data to copy in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa ::lwFuncSetBlockShape,
 * ::lwFuncSetSharedSize,
 * ::lwFuncGetAttribute,
 * ::lwParamSetSize,
 * ::lwParamSetf,
 * ::lwParamSeti,
 * ::lwLaunch,
 * ::lwLaunchGrid,
 * ::lwLaunchGridAsync,
 * ::lwLaunchKernel
 */
LWresult LWDAAPI lwParamSetv(LWfunction hfunc, int offset, void *ptr, unsigned int numbytes);

/**
 * \brief Launches a LWCA function
 *
 * \deprecated
 *
 * Ilwokes the kernel \p f on a 1 x 1 x 1 grid of blocks. The block
 * contains the number of threads specified by a previous call to
 * ::lwFuncSetBlockShape().
 *
 * \param f - Kernel to launch
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_LAUNCH_FAILED,
 * ::LWDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
 * ::LWDA_ERROR_LAUNCH_TIMEOUT,
 * ::LWDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
 * ::LWDA_ERROR_SHARED_OBJECT_INIT_FAILED
 * \notefnerr
 *
 * \sa ::lwFuncSetBlockShape,
 * ::lwFuncSetSharedSize,
 * ::lwFuncGetAttribute,
 * ::lwParamSetSize,
 * ::lwParamSetf,
 * ::lwParamSeti,
 * ::lwParamSetv,
 * ::lwLaunchGrid,
 * ::lwLaunchGridAsync,
 * ::lwLaunchKernel
 */
LWresult LWDAAPI lwLaunch(LWfunction f);

/**
 * \brief Launches a LWCA function
 *
 * \deprecated
 *
 * Ilwokes the kernel \p f on a \p grid_width x \p grid_height grid of
 * blocks. Each block contains the number of threads specified by a previous
 * call to ::lwFuncSetBlockShape().
 *
 * \param f           - Kernel to launch
 * \param grid_width  - Width of grid in blocks
 * \param grid_height - Height of grid in blocks
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_LAUNCH_FAILED,
 * ::LWDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
 * ::LWDA_ERROR_LAUNCH_TIMEOUT,
 * ::LWDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
 * ::LWDA_ERROR_SHARED_OBJECT_INIT_FAILED
 * \notefnerr
 *
 * \sa ::lwFuncSetBlockShape,
 * ::lwFuncSetSharedSize,
 * ::lwFuncGetAttribute,
 * ::lwParamSetSize,
 * ::lwParamSetf,
 * ::lwParamSeti,
 * ::lwParamSetv,
 * ::lwLaunch,
 * ::lwLaunchGridAsync,
 * ::lwLaunchKernel
 */
LWresult LWDAAPI lwLaunchGrid(LWfunction f, int grid_width, int grid_height);

/**
 * \brief Launches a LWCA function
 *
 * \deprecated
 *
 * Ilwokes the kernel \p f on a \p grid_width x \p grid_height grid of
 * blocks. Each block contains the number of threads specified by a previous
 * call to ::lwFuncSetBlockShape().
 *
 * \param f           - Kernel to launch
 * \param grid_width  - Width of grid in blocks
 * \param grid_height - Height of grid in blocks
 * \param hStream     - Stream identifier
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_LAUNCH_FAILED,
 * ::LWDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
 * ::LWDA_ERROR_LAUNCH_TIMEOUT,
 * ::LWDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
 * ::LWDA_ERROR_SHARED_OBJECT_INIT_FAILED
 *
 * \note In certain cases where lwbins are created with no ABI (i.e., using \p ptxas \p --abi-compile \p no), 
 *       this function may serialize kernel launches. In order to force the LWCA driver to retain 
 *		 asynchronous behavior, set the ::LW_CTX_LMEM_RESIZE_TO_MAX flag during context creation (see ::lwCtxCreate).
 *       
 * \note_null_stream
 * \notefnerr
 *
 * \sa ::lwFuncSetBlockShape,
 * ::lwFuncSetSharedSize,
 * ::lwFuncGetAttribute,
 * ::lwParamSetSize,
 * ::lwParamSetf,
 * ::lwParamSeti,
 * ::lwParamSetv,
 * ::lwLaunch,
 * ::lwLaunchGrid,
 * ::lwLaunchKernel
 */
LWresult LWDAAPI lwLaunchGridAsync(LWfunction f, int grid_width, int grid_height, LWstream hStream);


/**
 * \brief Adds a texture-reference to the function's argument list
 *
 * \deprecated
 *
 * Makes the LWCA array or linear memory bound to the texture reference
 * \p hTexRef available to a device program as a texture. In this version of
 * LWCA, the texture-reference must be obtained via ::lwModuleGetTexRef() and
 * the \p texunit parameter must be set to ::LW_PARAM_TR_DEFAULT.
 *
 * \param hfunc   - Kernel to add texture-reference to
 * \param texunit - Texture unit (must be ::LW_PARAM_TR_DEFAULT)
 * \param hTexRef - Texture-reference to add to argument list
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 */
LWresult LWDAAPI lwParamSetTexRef(LWfunction hfunc, int texunit, LWtexref hTexRef);
/** @} */ /* END LWDA_EXEC_DEPRECATED */


#if __LWDA_API_VERSION >= 6050
/**
 * \defgroup LWDA_OCLWPANCY Oclwpancy
 *
 * ___MANBRIEF___ oclwpancy callwlation functions of the low-level LWCA driver
 * API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the oclwpancy callwlation functions of the low-level LWCA
 * driver application programming interface.
 *
 * @{
 */

/**
 * \brief Returns oclwpancy of a function
 *
 * Returns in \p *numBlocks the number of the maximum active blocks per
 * streaming multiprocessor.
 *
 * \param numBlocks       - Returned oclwpancy
 * \param func            - Kernel for which oclwpancy is callwlated
 * \param blockSize       - Block size the kernel is intended to be launched with
 * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::lwdaOclwpancyMaxActiveBlocksPerMultiprocessor
 */
LWresult LWDAAPI lwOclwpancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, LWfunction func, int blockSize, size_t dynamicSMemSize);

/**
 * \brief Returns oclwpancy of a function
 *
 * Returns in \p *numBlocks the number of the maximum active blocks per
 * streaming multiprocessor.
 *
 * The \p Flags parameter controls how special cases are handled. The
 * valid flags are:
 *
 * - ::LW_OCLWPANCY_DEFAULT, which maintains the default behavior as
 *   ::lwOclwpancyMaxActiveBlocksPerMultiprocessor;
 *
 * - ::LW_OCLWPANCY_DISABLE_CACHING_OVERRIDE, which suppresses the
 *   default behavior on platform where global caching affects
 *   oclwpancy. On such platforms, if caching is enabled, but
 *   per-block SM resource usage would result in zero oclwpancy, the
 *   oclwpancy calculator will callwlate the oclwpancy as if caching
 *   is disabled. Setting ::LW_OCLWPANCY_DISABLE_CACHING_OVERRIDE makes
 *   the oclwpancy calculator to return 0 in such cases. More information
 *   can be found about this feature in the "Unified L1/Texture Cache"
 *   section of the Maxwell tuning guide.
 *
 * \param numBlocks       - Returned oclwpancy
 * \param func            - Kernel for which oclwpancy is callwlated
 * \param blockSize       - Block size the kernel is intended to be launched with
 * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
 * \param flags           - Requested behavior for the oclwpancy calculator
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::lwdaOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags
 */
LWresult LWDAAPI lwOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, LWfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
    
/**
 * \brief Suggest a launch configuration with reasonable oclwpancy
 *
 * Returns in \p *blockSize a reasonable block size that can achieve
 * the maximum oclwpancy (or, the maximum number of active warps with
 * the fewest blocks per multiprocessor), and in \p *minGridSize the
 * minimum grid size to achieve the maximum oclwpancy.
 *
 * If \p blockSizeLimit is 0, the configurator will use the maximum
 * block size permitted by the device / function instead.
 *
 * If per-block dynamic shared memory allocation is not needed, the
 * user should leave both \p blockSizeToDynamicSMemSize and \p
 * dynamicSMemSize as 0.
 *
 * If per-block dynamic shared memory allocation is needed, then if
 * the dynamic shared memory size is constant regardless of block
 * size, the size should be passed through \p dynamicSMemSize, and \p
 * blockSizeToDynamicSMemSize should be NULL.
 *
 * Otherwise, if the per-block dynamic shared memory size varies with
 * different block sizes, the user needs to provide a unary function
 * through \p blockSizeToDynamicSMemSize that computes the dynamic
 * shared memory needed by \p func for any given block size. \p
 * dynamicSMemSize is ignored. An example signature is:
 *
 * \code
 *    // Take block size, returns dynamic shared memory needed
 *    size_t blockToSmem(int blockSize);
 * \endcode
 *
 * \param minGridSize - Returned minimum grid size needed to achieve the maximum oclwpancy
 * \param blockSize   - Returned maximum block size that can achieve the maximum oclwpancy
 * \param func        - Kernel for which launch configuration is callwlated
 * \param blockSizeToDynamicSMemSize - A function that callwlates how much per-block dynamic shared memory \p func uses based on the block size
 * \param dynamicSMemSize - Dynamic shared memory usage intended, in bytes
 * \param blockSizeLimit  - The maximum block size \p func is designed to handle
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::lwdaOclwpancyMaxPotentialBlockSize
 */
LWresult LWDAAPI lwOclwpancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, LWfunction func, LWoclwpancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit);

/**
 * \brief Suggest a launch configuration with reasonable oclwpancy
 *
 * An extended version of ::lwOclwpancyMaxPotentialBlockSize. In
 * addition to arguments passed to ::lwOclwpancyMaxPotentialBlockSize,
 * ::lwOclwpancyMaxPotentialBlockSizeWithFlags also takes a \p Flags
 * parameter.
 *
 * The \p Flags parameter controls how special cases are handled. The
 * valid flags are:
 *
 * - ::LW_OCLWPANCY_DEFAULT, which maintains the default behavior as
 *   ::lwOclwpancyMaxPotentialBlockSize;
 *
 * - ::LW_OCLWPANCY_DISABLE_CACHING_OVERRIDE, which suppresses the
 *   default behavior on platform where global caching affects
 *   oclwpancy. On such platforms, the launch configurations that
 *   produces maximal oclwpancy might not support global
 *   caching. Setting ::LW_OCLWPANCY_DISABLE_CACHING_OVERRIDE
 *   guarantees that the the produced launch configuration is global
 *   caching compatible at a potential cost of oclwpancy. More information
 *   can be found about this feature in the "Unified L1/Texture Cache"
 *   section of the Maxwell tuning guide.
 *
 * \param minGridSize - Returned minimum grid size needed to achieve the maximum oclwpancy
 * \param blockSize   - Returned maximum block size that can achieve the maximum oclwpancy
 * \param func        - Kernel for which launch configuration is callwlated
 * \param blockSizeToDynamicSMemSize - A function that callwlates how much per-block dynamic shared memory \p func uses based on the block size
 * \param dynamicSMemSize - Dynamic shared memory usage intended, in bytes
 * \param blockSizeLimit  - The maximum block size \p func is designed to handle
 * \param flags       - Options
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::lwdaOclwpancyMaxPotentialBlockSizeWithFlags
 */
LWresult LWDAAPI lwOclwpancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, LWfunction func, LWoclwpancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags);

/** @} */ /* END LWDA_OCLWPANCY */
#endif /* __LWDA_API_VERSION >= 6050 */

/**
 * \defgroup LWDA_TEXREF Texture Reference Management
 *
 * ___MANBRIEF___ texture reference management functions of the low-level LWCA
 * driver API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the texture reference management functions of the
 * low-level LWCA driver application programming interface.
 *
 * @{
 */

/**
 * \brief Binds an array as a texture reference
 *
 * Binds the LWCA array \p hArray to the texture reference \p hTexRef. Any
 * previous address or LWCA array state associated with the texture reference
 * is superseded by this function. \p Flags must be set to
 * ::LW_TRSA_OVERRIDE_FORMAT. Any LWCA array previously bound to \p hTexRef is
 * unbound.
 *
 * \param hTexRef - Texture reference to bind
 * \param hArray  - Array to bind
 * \param Flags   - Options (must be ::LW_TRSA_OVERRIDE_FORMAT)
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode,
 * ::lwTexRefSetFilterMode, ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat,
 * ::lwdaBindTextureToArray
 */
LWresult LWDAAPI lwTexRefSetArray(LWtexref hTexRef, LWarray hArray, unsigned int Flags);

/**
 * \brief Binds a mipmapped array to a texture reference
 *
 * Binds the LWCA mipmapped array \p hMipmappedArray to the texture reference \p hTexRef.
 * Any previous address or LWCA array state associated with the texture reference
 * is superseded by this function. \p Flags must be set to ::LW_TRSA_OVERRIDE_FORMAT. 
 * Any LWCA array previously bound to \p hTexRef is unbound.
 *
 * \param hTexRef         - Texture reference to bind
 * \param hMipmappedArray - Mipmapped array to bind
 * \param Flags           - Options (must be ::LW_TRSA_OVERRIDE_FORMAT)
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode,
 * ::lwTexRefSetFilterMode, ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat,
 * ::lwdaBindTextureToMipmappedArray
 */
LWresult LWDAAPI lwTexRefSetMipmappedArray(LWtexref hTexRef, LWmipmappedArray hMipmappedArray, unsigned int Flags);

#if __LWDA_API_VERSION >= 3020
/**
 * \brief Binds an address as a texture reference
 *
 * Binds a linear address range to the texture reference \p hTexRef. Any
 * previous address or LWCA array state associated with the texture reference
 * is superseded by this function. Any memory previously bound to \p hTexRef
 * is unbound.
 *
 * Since the hardware enforces an alignment requirement on texture base
 * addresses, ::lwTexRefSetAddress() passes back a byte offset in
 * \p *ByteOffset that must be applied to texture fetches in order to read from
 * the desired memory. This offset must be divided by the texel size and
 * passed to kernels that read from the texture so they can be applied to the
 * ::tex1Dfetch() function.
 *
 * If the device memory pointer was returned from ::lwMemAlloc(), the offset
 * is guaranteed to be 0 and NULL may be passed as the \p ByteOffset parameter.
 *
 * The total number of elements (or texels) in the linear address range
 * cannot exceed ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH.
 * The number of elements is computed as (\p bytes / bytesPerElement),
 * where bytesPerElement is determined from the data format and number of 
 * components set using ::lwTexRefSetFormat().
 *
 * \param ByteOffset - Returned byte offset
 * \param hTexRef    - Texture reference to bind
 * \param dptr       - Device pointer to bind
 * \param bytes      - Size of memory to bind in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFilterMode, ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat,
 * ::lwdaBindTexture
 */
LWresult LWDAAPI lwTexRefSetAddress(size_t *ByteOffset, LWtexref hTexRef, LWdeviceptr dptr, size_t bytes);

/**
 * \brief Binds an address as a 2D texture reference
 *
 * Binds a linear address range to the texture reference \p hTexRef. Any
 * previous address or LWCA array state associated with the texture reference
 * is superseded by this function. Any memory previously bound to \p hTexRef
 * is unbound.
 *
 * Using a ::tex2D() function inside a kernel requires a call to either
 * ::lwTexRefSetArray() to bind the corresponding texture reference to an
 * array, or ::lwTexRefSetAddress2D() to bind the texture reference to linear
 * memory.
 *
 * Function calls to ::lwTexRefSetFormat() cannot follow calls to
 * ::lwTexRefSetAddress2D() for the same texture reference.
 *
 * It is required that \p dptr be aligned to the appropriate hardware-specific
 * texture alignment. You can query this value using the device attribute
 * ::LW_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT. If an unaligned \p dptr is
 * supplied, ::LWDA_ERROR_ILWALID_VALUE is returned.
 *
 * \p Pitch has to be aligned to the hardware-specific texture pitch alignment.
 * This value can be queried using the device attribute 
 * ::LW_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT. If an unaligned \p Pitch is 
 * supplied, ::LWDA_ERROR_ILWALID_VALUE is returned.
 *
 * Width and Height, which are specified in elements (or texels), cannot exceed
 * ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH and
 * ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT respectively.
 * \p Pitch, which is specified in bytes, cannot exceed 
 * ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH.
 *
 * \param hTexRef - Texture reference to bind
 * \param desc    - Descriptor of LWCA array
 * \param dptr    - Device pointer to bind
 * \param Pitch   - Line pitch in bytes
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFilterMode, ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat,
 * ::lwdaBindTexture2D
 */
LWresult LWDAAPI lwTexRefSetAddress2D(LWtexref hTexRef, const LWDA_ARRAY_DESCRIPTOR *desc, LWdeviceptr dptr, size_t Pitch);
#endif /* __LWDA_API_VERSION >= 3020 */

/**
 * \brief Sets the format for a texture reference
 *
 * Specifies the format of the data to be read by the texture reference
 * \p hTexRef. \p fmt and \p NumPackedComponents are exactly analogous to the
 * ::Format and ::NumChannels members of the ::LWDA_ARRAY_DESCRIPTOR structure:
 * They specify the format of each component and the number of components per
 * array element.
 *
 * \param hTexRef             - Texture reference
 * \param fmt                 - Format to set
 * \param NumPackedComponents - Number of components per array element
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFilterMode, ::lwTexRefSetFlags,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat,
 * ::lwdaCreateChannelDesc,
 * ::lwdaBindTexture,
 * ::lwdaBindTexture2D,
 * ::lwdaBindTextureToArray,
 * ::lwdaBindTextureToMipmappedArray
 */
LWresult LWDAAPI lwTexRefSetFormat(LWtexref hTexRef, LWarray_format fmt, int NumPackedComponents);

/**
 * \brief Sets the addressing mode for a texture reference
 *
 * Specifies the addressing mode \p am for the given dimension \p dim of the
 * texture reference \p hTexRef. If \p dim is zero, the addressing mode is
 * applied to the first parameter of the functions used to fetch from the
 * texture; if \p dim is 1, the second, and so on. ::LWaddress_mode is defined
 * as:
 * \code
   typedef enum LWaddress_mode_enum {
      LW_TR_ADDRESS_MODE_WRAP = 0,
      LW_TR_ADDRESS_MODE_CLAMP = 1,
      LW_TR_ADDRESS_MODE_MIRROR = 2,
      LW_TR_ADDRESS_MODE_BORDER = 3
   } LWaddress_mode;
 * \endcode
 *
 * Note that this call has no effect if \p hTexRef is bound to linear memory.
 * Also, if the flag, ::LW_TRSF_NORMALIZED_COORDINATES, is not set, the only 
 * supported address mode is ::LW_TR_ADDRESS_MODE_CLAMP.
 *
 * \param hTexRef - Texture reference
 * \param dim     - Dimension
 * \param am      - Addressing mode to set
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetArray,
 * ::lwTexRefSetFilterMode, ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat,
 * ::lwdaBindTexture,
 * ::lwdaBindTexture2D,
 * ::lwdaBindTextureToArray,
 * ::lwdaBindTextureToMipmappedArray
 */
LWresult LWDAAPI lwTexRefSetAddressMode(LWtexref hTexRef, int dim, LWaddress_mode am);

/**
 * \brief Sets the filtering mode for a texture reference
 *
 * Specifies the filtering mode \p fm to be used when reading memory through
 * the texture reference \p hTexRef. ::LWfilter_mode_enum is defined as:
 *
 * \code
   typedef enum LWfilter_mode_enum {
      LW_TR_FILTER_MODE_POINT = 0,
      LW_TR_FILTER_MODE_LINEAR = 1
   } LWfilter_mode;
 * \endcode
 *
 * Note that this call has no effect if \p hTexRef is bound to linear memory.
 *
 * \param hTexRef - Texture reference
 * \param fm      - Filtering mode to set
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat,
 * ::lwdaBindTextureToArray
 */
LWresult LWDAAPI lwTexRefSetFilterMode(LWtexref hTexRef, LWfilter_mode fm);

/**
 * \brief Sets the mipmap filtering mode for a texture reference
 *
 * Specifies the mipmap filtering mode \p fm to be used when reading memory through
 * the texture reference \p hTexRef. ::LWfilter_mode_enum is defined as:
 *
 * \code
   typedef enum LWfilter_mode_enum {
      LW_TR_FILTER_MODE_POINT = 0,
      LW_TR_FILTER_MODE_LINEAR = 1
   } LWfilter_mode;
 * \endcode
 *
 * Note that this call has no effect if \p hTexRef is not bound to a mipmapped array.
 *
 * \param hTexRef - Texture reference
 * \param fm      - Filtering mode to set
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat,
 * ::lwdaBindTextureToMipmappedArray
 */
LWresult LWDAAPI lwTexRefSetMipmapFilterMode(LWtexref hTexRef, LWfilter_mode fm);

/**
 * \brief Sets the mipmap level bias for a texture reference
 *
 * Specifies the mipmap level bias \p bias to be added to the specified mipmap level when 
 * reading memory through the texture reference \p hTexRef.
 *
 * Note that this call has no effect if \p hTexRef is not bound to a mipmapped array.
 *
 * \param hTexRef - Texture reference
 * \param bias    - Mipmap level bias
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat,
 * ::lwdaBindTextureToMipmappedArray
 */
LWresult LWDAAPI lwTexRefSetMipmapLevelBias(LWtexref hTexRef, float bias);

/**
 * \brief Sets the mipmap min/max mipmap level clamps for a texture reference
 *
 * Specifies the min/max mipmap level clamps, \p minMipmapLevelClamp and \p maxMipmapLevelClamp
 * respectively, to be used when reading memory through the texture reference 
 * \p hTexRef.
 *
 * Note that this call has no effect if \p hTexRef is not bound to a mipmapped array.
 *
 * \param hTexRef        - Texture reference
 * \param minMipmapLevelClamp - Mipmap min level clamp
 * \param maxMipmapLevelClamp - Mipmap max level clamp
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat,
 * ::lwdaBindTextureToMipmappedArray
 */
LWresult LWDAAPI lwTexRefSetMipmapLevelClamp(LWtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);

/**
 * \brief Sets the maximum anisotropy for a texture reference
 *
 * Specifies the maximum anisotropy \p maxAniso to be used when reading memory through
 * the texture reference \p hTexRef. 
 *
 * Note that this call has no effect if \p hTexRef is bound to linear memory.
 *
 * \param hTexRef  - Texture reference
 * \param maxAniso - Maximum anisotropy
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat,
 * ::lwdaBindTextureToArray,
 * ::lwdaBindTextureToMipmappedArray
 */
LWresult LWDAAPI lwTexRefSetMaxAnisotropy(LWtexref hTexRef, unsigned int maxAniso);

/**
 * \brief Sets the border color for a texture reference
 *
 * Specifies the value of the RGBA color via the \p pBorderColor to the texture reference
 * \p hTexRef. The color value supports only float type and holds color components in
 * the following sequence:
 * pBorderColor[0] holds 'R' component
 * pBorderColor[1] holds 'G' component
 * pBorderColor[2] holds 'B' component
 * pBorderColor[3] holds 'A' component
 *
 * Note that the color values can be set only when the Address mode is set to
 * LW_TR_ADDRESS_MODE_BORDER using ::lwTexRefSetAddressMode.
 * Applications using integer border color values have to "reinterpret_cast" their values to float.
 *
 * \param hTexRef       - Texture reference
 * \param pBorderColor  - RGBA color
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddressMode,
 * ::lwTexRefGetAddressMode, ::lwTexRefGetBorderColor,
 * ::lwdaBindTexture,
 * ::lwdaBindTexture2D,
 * ::lwdaBindTextureToArray,
 * ::lwdaBindTextureToMipmappedArray
 */
LWresult LWDAAPI lwTexRefSetBorderColor(LWtexref hTexRef, float *pBorderColor);

/**
 * \brief Sets the flags for a texture reference
 *
 * Specifies optional flags via \p Flags to specify the behavior of data
 * returned through the texture reference \p hTexRef. The valid flags are:
 *
 * - ::LW_TRSF_READ_AS_INTEGER, which suppresses the default behavior of
 *   having the texture promote integer data to floating point data in the
 *   range [0, 1]. Note that texture with 32-bit integer format
 *   would not be promoted, regardless of whether or not this
 *   flag is specified;
 * - ::LW_TRSF_NORMALIZED_COORDINATES, which suppresses the 
 *   default behavior of having the texture coordinates range
 *   from [0, Dim) where Dim is the width or height of the LWCA
 *   array. Instead, the texture coordinates [0, 1.0) reference
 *   the entire breadth of the array dimension;
 *
 * \param hTexRef - Texture reference
 * \param Flags   - Optional flags to set
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFilterMode, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat,
 * ::lwdaBindTexture,
 * ::lwdaBindTexture2D,
 * ::lwdaBindTextureToArray,
 * ::lwdaBindTextureToMipmappedArray
 */
LWresult LWDAAPI lwTexRefSetFlags(LWtexref hTexRef, unsigned int Flags);

#if __LWDA_API_VERSION >= 3020
/**
 * \brief Gets the address associated with a texture reference
 *
 * Returns in \p *pdptr the base address bound to the texture reference
 * \p hTexRef, or returns ::LWDA_ERROR_ILWALID_VALUE if the texture reference
 * is not bound to any device memory range.
 *
 * \param pdptr   - Returned device address
 * \param hTexRef - Texture reference
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFilterMode, ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat
 */
LWresult LWDAAPI lwTexRefGetAddress(LWdeviceptr *pdptr, LWtexref hTexRef);
#endif /* __LWDA_API_VERSION >= 3020 */

/**
 * \brief Gets the array bound to a texture reference
 *
 * Returns in \p *phArray the LWCA array bound to the texture reference
 * \p hTexRef, or returns ::LWDA_ERROR_ILWALID_VALUE if the texture reference
 * is not bound to any LWCA array.
 *
 * \param phArray - Returned array
 * \param hTexRef - Texture reference
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFilterMode, ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat
 */
LWresult LWDAAPI lwTexRefGetArray(LWarray *phArray, LWtexref hTexRef);

/**
 * \brief Gets the mipmapped array bound to a texture reference
 *
 * Returns in \p *phMipmappedArray the LWCA mipmapped array bound to the texture 
 * reference \p hTexRef, or returns ::LWDA_ERROR_ILWALID_VALUE if the texture reference
 * is not bound to any LWCA mipmapped array.
 *
 * \param phMipmappedArray - Returned mipmapped array
 * \param hTexRef          - Texture reference
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFilterMode, ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat
 */
LWresult LWDAAPI lwTexRefGetMipmappedArray(LWmipmappedArray *phMipmappedArray, LWtexref hTexRef);

/**
 * \brief Gets the addressing mode used by a texture reference
 *
 * Returns in \p *pam the addressing mode corresponding to the
 * dimension \p dim of the texture reference \p hTexRef. Lwrrently, the only
 * valid value for \p dim are 0 and 1.
 *
 * \param pam     - Returned addressing mode
 * \param hTexRef - Texture reference
 * \param dim     - Dimension
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFilterMode, ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat
 */
LWresult LWDAAPI lwTexRefGetAddressMode(LWaddress_mode *pam, LWtexref hTexRef, int dim);

/**
 * \brief Gets the filter-mode used by a texture reference
 *
 * Returns in \p *pfm the filtering mode of the texture reference
 * \p hTexRef.
 *
 * \param pfm     - Returned filtering mode
 * \param hTexRef - Texture reference
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFilterMode, ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFlags, ::lwTexRefGetFormat
 */
LWresult LWDAAPI lwTexRefGetFilterMode(LWfilter_mode *pfm, LWtexref hTexRef);

/**
 * \brief Gets the format used by a texture reference
 *
 * Returns in \p *pFormat and \p *pNumChannels the format and number
 * of components of the LWCA array bound to the texture reference \p hTexRef.
 * If \p pFormat or \p pNumChannels is NULL, it will be ignored.
 *
 * \param pFormat      - Returned format
 * \param pNumChannels - Returned number of components
 * \param hTexRef      - Texture reference
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFilterMode, ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags
 */
LWresult LWDAAPI lwTexRefGetFormat(LWarray_format *pFormat, int *pNumChannels, LWtexref hTexRef);

/**
 * \brief Gets the mipmap filtering mode for a texture reference
 *
 * Returns the mipmap filtering mode in \p pfm that's used when reading memory through
 * the texture reference \p hTexRef.
 *
 * \param pfm     - Returned mipmap filtering mode
 * \param hTexRef - Texture reference
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat
 */
LWresult LWDAAPI lwTexRefGetMipmapFilterMode(LWfilter_mode *pfm, LWtexref hTexRef);

/**
 * \brief Gets the mipmap level bias for a texture reference
 *
 * Returns the mipmap level bias in \p pBias that's added to the specified mipmap
 * level when reading memory through the texture reference \p hTexRef.
 *
 * \param pbias   - Returned mipmap level bias
 * \param hTexRef - Texture reference
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat
 */
LWresult LWDAAPI lwTexRefGetMipmapLevelBias(float *pbias, LWtexref hTexRef);

/**
 * \brief Gets the min/max mipmap level clamps for a texture reference
 *
 * Returns the min/max mipmap level clamps in \p pminMipmapLevelClamp and \p pmaxMipmapLevelClamp
 * that's used when reading memory through the texture reference \p hTexRef. 
 *
 * \param pminMipmapLevelClamp - Returned mipmap min level clamp
 * \param pmaxMipmapLevelClamp - Returned mipmap max level clamp
 * \param hTexRef              - Texture reference
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat
 */
LWresult LWDAAPI lwTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, LWtexref hTexRef);

/**
 * \brief Gets the maximum anisotropy for a texture reference
 *
 * Returns the maximum anisotropy in \p pmaxAniso that's used when reading memory through
 * the texture reference \p hTexRef. 
 *
 * \param pmaxAniso - Returned maximum anisotropy
 * \param hTexRef   - Texture reference
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFlags, ::lwTexRefGetFormat
 */
LWresult LWDAAPI lwTexRefGetMaxAnisotropy(int *pmaxAniso, LWtexref hTexRef);

/**
 * \brief Gets the border color used by a texture reference
 *
 * Returns in \p pBorderColor, values of the RGBA color used by
 * the texture reference \p hTexRef.
 * The color value is of type float and holds color components in
 * the following sequence:
 * pBorderColor[0] holds 'R' component
 * pBorderColor[1] holds 'G' component
 * pBorderColor[2] holds 'B' component
 * pBorderColor[3] holds 'A' component
 *
 * \param hTexRef  - Texture reference
 * \param pBorderColor   - Returned Type and Value of RGBA color
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddressMode,
 * ::lwTexRefSetAddressMode, ::lwTexRefSetBorderColor
 */
LWresult LWDAAPI lwTexRefGetBorderColor(float *pBorderColor, LWtexref hTexRef); 

/**
 * \brief Gets the flags used by a texture reference
 *
 * Returns in \p *pFlags the flags of the texture reference \p hTexRef.
 *
 * \param pFlags  - Returned flags
 * \param hTexRef - Texture reference
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefSetAddress,
 * ::lwTexRefSetAddress2D, ::lwTexRefSetAddressMode, ::lwTexRefSetArray,
 * ::lwTexRefSetFilterMode, ::lwTexRefSetFlags, ::lwTexRefSetFormat,
 * ::lwTexRefGetAddress, ::lwTexRefGetAddressMode, ::lwTexRefGetArray,
 * ::lwTexRefGetFilterMode, ::lwTexRefGetFormat
 */
LWresult LWDAAPI lwTexRefGetFlags(unsigned int *pFlags, LWtexref hTexRef);

/** @} */ /* END LWDA_TEXREF */

/**
 * \defgroup LWDA_TEXREF_DEPRECATED Texture Reference Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated texture reference management functions of the
 * low-level LWCA driver API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the deprecated texture reference management
 * functions of the low-level LWCA driver application programming interface.
 *
 * @{
 */

/**
 * \brief Creates a texture reference
 *
 * \deprecated
 *
 * Creates a texture reference and returns its handle in \p *pTexRef. Once
 * created, the application must call ::lwTexRefSetArray() or
 * ::lwTexRefSetAddress() to associate the reference with allocated memory.
 * Other texture reference functions are used to specify the format and
 * interpretation (addressing, filtering, etc.) to be used when the memory is
 * read through this texture reference.
 *
 * \param pTexRef - Returned texture reference
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefDestroy
 */
LWresult LWDAAPI lwTexRefCreate(LWtexref *pTexRef);

/**
 * \brief Destroys a texture reference
 *
 * \deprecated
 *
 * Destroys the texture reference specified by \p hTexRef.
 *
 * \param hTexRef - Texture reference to destroy
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwTexRefCreate
 */
LWresult LWDAAPI lwTexRefDestroy(LWtexref hTexRef);

/** @} */ /* END LWDA_TEXREF_DEPRECATED */


/**
 * \defgroup LWDA_SURFREF Surface Reference Management
 *
 * ___MANBRIEF___ surface reference management functions of the low-level LWCA
 * driver API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the surface reference management functions of the
 * low-level LWCA driver application programming interface.
 *
 * @{
 */

/**
 * \brief Sets the LWCA array for a surface reference.
 *
 * Sets the LWCA array \p hArray to be read and written by the surface reference
 * \p hSurfRef.  Any previous LWCA array state associated with the surface
 * reference is superseded by this function.  \p Flags must be set to 0.
 * The ::LWDA_ARRAY3D_SURFACE_LDST flag must have been set for the LWCA array.
 * Any LWCA array previously bound to \p hSurfRef is unbound.

 * \param hSurfRef - Surface reference handle
 * \param hArray - LWCA array handle
 * \param Flags - set to 0
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa
 * ::lwModuleGetSurfRef,
 * ::lwSurfRefGetArray,
 * ::lwdaBindSurfaceToArray
 */
LWresult LWDAAPI lwSurfRefSetArray(LWsurfref hSurfRef, LWarray hArray, unsigned int Flags);

/**
 * \brief Passes back the LWCA array bound to a surface reference.
 *
 * Returns in \p *phArray the LWCA array bound to the surface reference
 * \p hSurfRef, or returns ::LWDA_ERROR_ILWALID_VALUE if the surface reference
 * is not bound to any LWCA array.

 * \param phArray - Surface reference handle
 * \param hSurfRef - Surface reference handle
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa ::lwModuleGetSurfRef, ::lwSurfRefSetArray
 */
LWresult LWDAAPI lwSurfRefGetArray(LWarray *phArray, LWsurfref hSurfRef);

/** @} */ /* END LWDA_SURFREF */

#if __LWDA_API_VERSION >= 5000
/**
 * \defgroup LWDA_TEXOBJECT Texture Object Management
 *
 * ___MANBRIEF___ texture object management functions of the low-level LWCA
 * driver API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the texture object management functions of the
 * low-level LWCA driver application programming interface. The texture
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
 * The ::LWDA_RESOURCE_DESC structure is defined as:
 * \code
        typedef struct LWDA_RESOURCE_DESC_st
        {
            LWresourcetype resType;

            union {
                struct {
                    LWarray hArray;
                } array;
                struct {
                    LWmipmappedArray hMipmappedArray;
                } mipmap;
                struct {
                    LWdeviceptr devPtr;
                    LWarray_format format;
                    unsigned int numChannels;
                    size_t sizeInBytes;
                } linear;
                struct {
                    LWdeviceptr devPtr;
                    LWarray_format format;
                    unsigned int numChannels;
                    size_t width;
                    size_t height;
                    size_t pitchInBytes;
                } pitch2D;
            } res;

            unsigned int flags;
        } LWDA_RESOURCE_DESC;

 * \endcode
 * where:
 * - ::LWDA_RESOURCE_DESC::resType specifies the type of resource to texture from.
 * LWresourceType is defined as:
 * \code
        typedef enum LWresourcetype_enum {
            LW_RESOURCE_TYPE_ARRAY           = 0x00,
            LW_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01,
            LW_RESOURCE_TYPE_LINEAR          = 0x02,
            LW_RESOURCE_TYPE_PITCH2D         = 0x03
        } LWresourcetype;
 * \endcode
 *
 * \par
 * If ::LWDA_RESOURCE_DESC::resType is set to ::LW_RESOURCE_TYPE_ARRAY, ::LWDA_RESOURCE_DESC::res::array::hArray
 * must be set to a valid LWCA array handle.
 *
 * \par
 * If ::LWDA_RESOURCE_DESC::resType is set to ::LW_RESOURCE_TYPE_MIPMAPPED_ARRAY, ::LWDA_RESOURCE_DESC::res::mipmap::hMipmappedArray
 * must be set to a valid LWCA mipmapped array handle.
 *
 * \par
 * If ::LWDA_RESOURCE_DESC::resType is set to ::LW_RESOURCE_TYPE_LINEAR, ::LWDA_RESOURCE_DESC::res::linear::devPtr
 * must be set to a valid device pointer, that is aligned to ::LW_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT.
 * ::LWDA_RESOURCE_DESC::res::linear::format and ::LWDA_RESOURCE_DESC::res::linear::numChannels
 * describe the format of each component and the number of components per array element. ::LWDA_RESOURCE_DESC::res::linear::sizeInBytes
 * specifies the size of the array in bytes. The total number of elements in the linear address range cannot exceed 
 * ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH. The number of elements is computed as (sizeInBytes / (sizeof(format) * numChannels)).
 *
 * \par
 * If ::LWDA_RESOURCE_DESC::resType is set to ::LW_RESOURCE_TYPE_PITCH2D, ::LWDA_RESOURCE_DESC::res::pitch2D::devPtr
 * must be set to a valid device pointer, that is aligned to ::LW_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT.
 * ::LWDA_RESOURCE_DESC::res::pitch2D::format and ::LWDA_RESOURCE_DESC::res::pitch2D::numChannels
 * describe the format of each component and the number of components per array element. ::LWDA_RESOURCE_DESC::res::pitch2D::width
 * and ::LWDA_RESOURCE_DESC::res::pitch2D::height specify the width and height of the array in elements, and cannot exceed
 * ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH and ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT respectively.
 * ::LWDA_RESOURCE_DESC::res::pitch2D::pitchInBytes specifies the pitch between two rows in bytes and has to be aligned to 
 * ::LW_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT. Pitch cannot exceed ::LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH.
 *
 * - ::flags must be set to zero.
 *
 *
 * The ::LWDA_TEXTURE_DESC struct is defined as
 * \code
        typedef struct LWDA_TEXTURE_DESC_st {
            LWaddress_mode addressMode[3];
            LWfilter_mode filterMode;
            unsigned int flags;
            unsigned int maxAnisotropy;
            LWfilter_mode mipmapFilterMode;
            float mipmapLevelBias;
            float minMipmapLevelClamp;
            float maxMipmapLevelClamp;
        } LWDA_TEXTURE_DESC;
 * \endcode
 * where
 * - ::LWDA_TEXTURE_DESC::addressMode specifies the addressing mode for each dimension of the texture data. ::LWaddress_mode is defined as:
 *   \code
        typedef enum LWaddress_mode_enum {
            LW_TR_ADDRESS_MODE_WRAP = 0,
            LW_TR_ADDRESS_MODE_CLAMP = 1,
            LW_TR_ADDRESS_MODE_MIRROR = 2,
            LW_TR_ADDRESS_MODE_BORDER = 3
        } LWaddress_mode;
 *   \endcode
 *   This is ignored if ::LWDA_RESOURCE_DESC::resType is ::LW_RESOURCE_TYPE_LINEAR. Also, if the flag, ::LW_TRSF_NORMALIZED_COORDINATES 
 *   is not set, the only supported address mode is ::LW_TR_ADDRESS_MODE_CLAMP.
 *
 * - ::LWDA_TEXTURE_DESC::filterMode specifies the filtering mode to be used when fetching from the texture. LWfilter_mode is defined as:
 *   \code
        typedef enum LWfilter_mode_enum {
            LW_TR_FILTER_MODE_POINT = 0,
            LW_TR_FILTER_MODE_LINEAR = 1
        } LWfilter_mode;
 *   \endcode
 *   This is ignored if ::LWDA_RESOURCE_DESC::resType is ::LW_RESOURCE_TYPE_LINEAR.
 *
 * - ::LWDA_TEXTURE_DESC::flags can be any combination of the following:
 *   - ::LW_TRSF_READ_AS_INTEGER, which suppresses the default behavior of having the texture promote integer data to floating point data in the
 *     range [0, 1]. Note that texture with 32-bit integer format would not be promoted, regardless of whether or not this flag is specified.
 *   - ::LW_TRSF_NORMALIZED_COORDINATES, which suppresses the default behavior of having the texture coordinates range from [0, Dim) where Dim is
 *     the width or height of the LWCA array. Instead, the texture coordinates [0, 1.0) reference the entire breadth of the array dimension; Note
 *     that for LWCA mipmapped arrays, this flag has to be set.
 *
 * - ::LWDA_TEXTURE_DESC::maxAnisotropy specifies the maximum anisotropy ratio to be used when doing anisotropic filtering. This value will be
 *   clamped to the range [1,16].
 *
 * - ::LWDA_TEXTURE_DESC::mipmapFilterMode specifies the filter mode when the callwlated mipmap level lies between two defined mipmap levels.
 *
 * - ::LWDA_TEXTURE_DESC::mipmapLevelBias specifies the offset to be applied to the callwlated mipmap level.
 *
 * - ::LWDA_TEXTURE_DESC::minMipmapLevelClamp specifies the lower end of the mipmap level range to clamp access to.
 *
 * - ::LWDA_TEXTURE_DESC::maxMipmapLevelClamp specifies the upper end of the mipmap level range to clamp access to.
 *
 *
 * The ::LWDA_RESOURCE_VIEW_DESC struct is defined as
 * \code
        typedef struct LWDA_RESOURCE_VIEW_DESC_st
        {
            LWresourceViewFormat format;
            size_t width;
            size_t height;
            size_t depth;
            unsigned int firstMipmapLevel;
            unsigned int lastMipmapLevel;
            unsigned int firstLayer;
            unsigned int lastLayer;
        } LWDA_RESOURCE_VIEW_DESC;
 * \endcode
 * where:
 * - ::LWDA_RESOURCE_VIEW_DESC::format specifies how the data contained in the LWCA array or LWCA mipmapped array should
 *   be interpreted. Note that this can inlwr a change in size of the texture data. If the resource view format is a block
 *   compressed format, then the underlying LWCA array or LWCA mipmapped array has to have a base of format ::LW_AD_FORMAT_UNSIGNED_INT32.
 *   with 2 or 4 channels, depending on the block compressed format. For ex., BC1 and BC4 require the underlying LWCA array to have
 *   a format of ::LW_AD_FORMAT_UNSIGNED_INT32 with 2 channels. The other BC formats require the underlying resource to have the same base
 *   format but with 4 channels.
 *
 * - ::LWDA_RESOURCE_VIEW_DESC::width specifies the new width of the texture data. If the resource view format is a block
 *   compressed format, this value has to be 4 times the original width of the resource. For non block compressed formats,
 *   this value has to be equal to that of the original resource.
 *
 * - ::LWDA_RESOURCE_VIEW_DESC::height specifies the new height of the texture data. If the resource view format is a block
 *   compressed format, this value has to be 4 times the original height of the resource. For non block compressed formats,
 *   this value has to be equal to that of the original resource.
 *
 * - ::LWDA_RESOURCE_VIEW_DESC::depth specifies the new depth of the texture data. This value has to be equal to that of the
 *   original resource.
 *
 * - ::LWDA_RESOURCE_VIEW_DESC::firstMipmapLevel specifies the most detailed mipmap level. This will be the new mipmap level zero.
 *   For non-mipmapped resources, this value has to be zero.::LWDA_TEXTURE_DESC::minMipmapLevelClamp and ::LWDA_TEXTURE_DESC::maxMipmapLevelClamp
 *   will be relative to this value. For ex., if the firstMipmapLevel is set to 2, and a minMipmapLevelClamp of 1.2 is specified,
 *   then the actual minimum mipmap level clamp will be 3.2.
 *
 * - ::LWDA_RESOURCE_VIEW_DESC::lastMipmapLevel specifies the least detailed mipmap level. For non-mipmapped resources, this value
 *   has to be zero.
 *
 * - ::LWDA_RESOURCE_VIEW_DESC::firstLayer specifies the first layer index for layered textures. This will be the new layer zero.
 *   For non-layered resources, this value has to be zero.
 *
 * - ::LWDA_RESOURCE_VIEW_DESC::lastLayer specifies the last layer index for layered textures. For non-layered resources, 
 *   this value has to be zero.
 *
 *
 * \param pTexObject   - Texture object to create
 * \param pResDesc     - Resource descriptor
 * \param pTexDesc     - Texture descriptor
 * \param pResViewDesc - Resource view descriptor 
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa
 * ::lwTexObjectDestroy,
 * ::lwdaCreateTextureObject
 */
LWresult LWDAAPI lwTexObjectCreate(LWtexObject *pTexObject, const LWDA_RESOURCE_DESC *pResDesc, const LWDA_TEXTURE_DESC *pTexDesc, const LWDA_RESOURCE_VIEW_DESC *pResViewDesc);

/**
 * \brief Destroys a texture object
 *
 * Destroys the texture object specified by \p texObject.
 *
 * \param texObject - Texture object to destroy
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa
 * ::lwTexObjectCreate,
 * ::lwdaDestroyTextureObject
 */
LWresult LWDAAPI lwTexObjectDestroy(LWtexObject texObject);

/**
 * \brief Returns a texture object's resource descriptor
 *
 * Returns the resource descriptor for the texture object specified by \p texObject.
 *
 * \param pResDesc  - Resource descriptor
 * \param texObject - Texture object
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa
 * ::lwTexObjectCreate,
 * ::lwdaGetTextureObjectResourceDesc,
 */
LWresult LWDAAPI lwTexObjectGetResourceDesc(LWDA_RESOURCE_DESC *pResDesc, LWtexObject texObject);

/**
 * \brief Returns a texture object's texture descriptor
 *
 * Returns the texture descriptor for the texture object specified by \p texObject.
 *
 * \param pTexDesc  - Texture descriptor
 * \param texObject - Texture object
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa
 * ::lwTexObjectCreate,
 * ::lwdaGetTextureObjectTextureDesc
 */
LWresult LWDAAPI lwTexObjectGetTextureDesc(LWDA_TEXTURE_DESC *pTexDesc, LWtexObject texObject);

/**
 * \brief Returns a texture object's resource view descriptor
 *
 * Returns the resource view descriptor for the texture object specified by \p texObject.
 * If no resource view was set for \p texObject, the ::LWDA_ERROR_ILWALID_VALUE is returned.
 *
 * \param pResViewDesc - Resource view descriptor
 * \param texObject    - Texture object
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa
 * ::lwTexObjectCreate,
 * ::lwdaGetTextureObjectResourceViewDesc
 */
LWresult LWDAAPI lwTexObjectGetResourceViewDesc(LWDA_RESOURCE_VIEW_DESC *pResViewDesc, LWtexObject texObject);

/** @} */ /* END LWDA_TEXOBJECT */

/**
 * \defgroup LWDA_SURFOBJECT Surface Object Management
 *
 * ___MANBRIEF___ surface object management functions of the low-level LWCA
 * driver API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the surface object management functions of the
 * low-level LWCA driver application programming interface. The surface
 * object API is only supported on devices of compute capability 3.0 or higher.
 *
 * @{
 */

/**
 * \brief Creates a surface object
 *
 * Creates a surface object and returns it in \p pSurfObject. \p pResDesc describes
 * the data to perform surface load/stores on. ::LWDA_RESOURCE_DESC::resType must be 
 * ::LW_RESOURCE_TYPE_ARRAY and  ::LWDA_RESOURCE_DESC::res::array::hArray
 * must be set to a valid LWCA array handle. ::LWDA_RESOURCE_DESC::flags must be set to zero.
 *
 * Surface objects are only supported on devices of compute capability 3.0 or higher.
 * Additionally, a surface object is an opaque value, and, as such, should only be
 * accessed through LWCA API calls.
 *
 * \param pSurfObject - Surface object to create
 * \param pResDesc    - Resource descriptor
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa
 * ::lwSurfObjectDestroy,
 * ::lwdaCreateSurfaceObject
 */
LWresult LWDAAPI lwSurfObjectCreate(LWsurfObject *pSurfObject, const LWDA_RESOURCE_DESC *pResDesc);

/**
 * \brief Destroys a surface object
 *
 * Destroys the surface object specified by \p surfObject.
 *
 * \param surfObject - Surface object to destroy
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa
 * ::lwSurfObjectCreate,
 * ::lwdaDestroySurfaceObject
 */
LWresult LWDAAPI lwSurfObjectDestroy(LWsurfObject surfObject);

/**
 * \brief Returns a surface object's resource descriptor
 *
 * Returns the resource descriptor for the surface object specified by \p surfObject.
 *
 * \param pResDesc   - Resource descriptor
 * \param surfObject - Surface object
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE
 *
 * \sa
 * ::lwSurfObjectCreate,
 * ::lwdaGetSurfaceObjectResourceDesc
 */
LWresult LWDAAPI lwSurfObjectGetResourceDesc(LWDA_RESOURCE_DESC *pResDesc, LWsurfObject surfObject);

/** @} */ /* END LWDA_SURFOBJECT */
#endif /* __LWDA_API_VERSION >= 5000 */

/**
 * \defgroup LWDA_PEER_ACCESS Peer Context Memory Access
 *
 * ___MANBRIEF___ direct peer context memory access functions of the low-level
 * LWCA driver API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the direct peer context memory access functions 
 * of the low-level LWCA driver application programming interface.
 *
 * @{
 */

#if __LWDA_API_VERSION >= 4000

/**
 * \brief Queries if a device may directly access a peer device's memory.
 *
 * Returns in \p *canAccessPeer a value of 1 if contexts on \p dev are capable of
 * directly accessing memory from contexts on \p peerDev and 0 otherwise.
 * If direct access of \p peerDev from \p dev is possible, then access may be
 * enabled on two specific contexts by calling ::lwCtxEnablePeerAccess().
 *
 * \param canAccessPeer - Returned access capability
 * \param dev           - Device from which allocations on \p peerDev are to
 *                        be directly accessed.
 * \param peerDev       - Device on which the allocations to be directly accessed 
 *                        by \p dev reside.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_DEVICE
 * \notefnerr
 *
 * \sa
 * ::lwCtxEnablePeerAccess,
 * ::lwCtxDisablePeerAccess,
 * ::lwdaDeviceCanAccessPeer
 */
LWresult LWDAAPI lwDeviceCanAccessPeer(int *canAccessPeer, LWdevice dev, LWdevice peerDev);

/**
 * \brief Enables direct access to memory allocations in a peer context.
 *
 * If both the current context and \p peerContext are on devices which support unified
 * addressing (as may be queried using ::LW_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING) and same
 * major compute capability, then on success all allocations from \p peerContext will
 * immediately be accessible by the current context.  See \ref LWDA_UNIFIED for additional
 * details.
 *
 * Note that access granted by this call is unidirectional and that in order to access
 * memory from the current context in \p peerContext, a separate symmetric call 
 * to ::lwCtxEnablePeerAccess() is required.
 *
 * There is a system-wide maximum of eight peer connections per device.
 *
 * Returns ::LWDA_ERROR_PEER_ACCESS_UNSUPPORTED if ::lwDeviceCanAccessPeer() indicates
 * that the ::LWdevice of the current context cannot directly access memory
 * from the ::LWdevice of \p peerContext.
 *
 * Returns ::LWDA_ERROR_PEER_ACCESS_ALREADY_ENABLED if direct access of
 * \p peerContext from the current context has already been enabled.
 *
 * Returns ::LWDA_ERROR_TOO_MANY_PEERS if direct peer access is not possible 
 * because hardware resources required for peer access have been exhausted.
 *
 * Returns ::LWDA_ERROR_ILWALID_CONTEXT if there is no current context, \p peerContext
 * is not a valid context, or if the current context is \p peerContext.
 *
 * Returns ::LWDA_ERROR_ILWALID_VALUE if \p Flags is not 0.
 *
 * \param peerContext - Peer context to enable direct access to from the current context
 * \param Flags       - Reserved for future use and must be set to 0
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
 * ::LWDA_ERROR_TOO_MANY_PEERS,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_PEER_ACCESS_UNSUPPORTED,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa
 * ::lwDeviceCanAccessPeer,
 * ::lwCtxDisablePeerAccess,
 * ::lwdaDeviceEnablePeerAccess
 */
LWresult LWDAAPI lwCtxEnablePeerAccess(LWcontext peerContext, unsigned int Flags);

/**
 * \brief Disables direct access to memory allocations in a peer context and 
 * unregisters any registered allocations.
 *
  Returns ::LWDA_ERROR_PEER_ACCESS_NOT_ENABLED if direct peer access has 
 * not yet been enabled from \p peerContext to the current context.
 *
 * Returns ::LWDA_ERROR_ILWALID_CONTEXT if there is no current context, or if
 * \p peerContext is not a valid context.
 *
 * \param peerContext - Peer context to disable direct access to
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_PEER_ACCESS_NOT_ENABLED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * \notefnerr
 *
 * \sa
 * ::lwDeviceCanAccessPeer,
 * ::lwCtxEnablePeerAccess,
 * ::lwdaDeviceDisablePeerAccess
 */
LWresult LWDAAPI lwCtxDisablePeerAccess(LWcontext peerContext);

#endif /* __LWDA_API_VERSION >= 4000 */

#if __LWDA_API_VERSION >= 8000

/**
 * \brief Queries attributes of the link between two devices.
 *
 * Returns in \p *value the value of the requested attribute \p attrib of the
 * link between \p srcDevice and \p dstDevice. The supported attributes are:
 * - ::LW_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK: A relative value indicating the
 *   performance of the link between two devices.
 * - ::LW_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED P2P: 1 if P2P Access is enable.
 * - ::LW_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED: 1 if Atomic operations over
 *   the link are supported.
 *
 * Returns ::LWDA_ERROR_ILWALID_DEVICE if \p srcDevice or \p dstDevice are not valid
 * or if they represent the same device.
 *
 * Returns ::LWDA_ERROR_ILWALID_VALUE if \p attrib is not valid or if \p value is
 * a null pointer.
 *
 * \param value         - Returned value of the requested attribute
 * \param attrib        - The requested attribute of the link between \p srcDevice and \p dstDevice.
 * \param srcDevice     - The source device of the target link.
 * \param dstDevice     - The destination device of the target link.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_DEVICE,
 * ::LWDA_ERROR_ILWALID_VALUE
 * \notefnerr
 *
 * \sa
 * ::lwCtxEnablePeerAccess,
 * ::lwCtxDisablePeerAccess,
 * ::lwDeviceCanAccessPeer,
 * ::lwdaDeviceGetP2PAttribute
 */
LWresult LWDAAPI lwDeviceGetP2PAttribute(int* value, LWdevice_P2PAttribute attrib, LWdevice srcDevice, LWdevice dstDevice);

#endif /* __LWDA_API_VERSION >= 8000 */

/** @} */ /* END LWDA_PEER_ACCESS */

/**
 * \defgroup LWDA_GRAPHICS Graphics Interoperability
 *
 * ___MANBRIEF___ graphics interoperability functions of the low-level LWCA
 * driver API (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the graphics interoperability functions of the
 * low-level LWCA driver application programming interface.
 *
 * @{
 */

/**
 * \brief Unregisters a graphics resource for access by LWCA
 *
 * Unregisters the graphics resource \p resource so it is not accessible by
 * LWCA unless registered again.
 *
 * If \p resource is invalid then ::LWDA_ERROR_ILWALID_HANDLE is
 * returned.
 *
 * \param resource - Resource to unregister
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::lwGraphicsD3D9RegisterResource,
 * ::lwGraphicsD3D10RegisterResource,
 * ::lwGraphicsD3D11RegisterResource,
 * ::lwGraphicsGLRegisterBuffer,
 * ::lwGraphicsGLRegisterImage,
 * ::lwdaGraphicsUnregisterResource
 */
LWresult LWDAAPI lwGraphicsUnregisterResource(LWgraphicsResource resource);

/**
 * \brief Get an array through which to access a subresource of a mapped graphics resource.
 *
 * Returns in \p *pArray an array through which the subresource of the mapped
 * graphics resource \p resource which corresponds to array index \p arrayIndex
 * and mipmap level \p mipLevel may be accessed.  The value set in \p *pArray may
 * change every time that \p resource is mapped.
 *
 * If \p resource is not a texture then it cannot be accessed via an array and
 * ::LWDA_ERROR_NOT_MAPPED_AS_ARRAY is returned.
 * If \p arrayIndex is not a valid array index for \p resource then
 * ::LWDA_ERROR_ILWALID_VALUE is returned.
 * If \p mipLevel is not a valid mipmap level for \p resource then
 * ::LWDA_ERROR_ILWALID_VALUE is returned.
 * If \p resource is not mapped then ::LWDA_ERROR_NOT_MAPPED is returned.
 *
 * \param pArray      - Returned array through which a subresource of \p resource may be accessed
 * \param resource    - Mapped resource to access
 * \param arrayIndex  - Array index for array textures or lwbemap face
 *                      index as defined by ::LWarray_lwbemap_face for
 *                      lwbemap textures for the subresource to access
 * \param mipLevel    - Mipmap level for the subresource to access
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_NOT_MAPPED,
 * ::LWDA_ERROR_NOT_MAPPED_AS_ARRAY
 * \notefnerr
 *
 * \sa
 * ::lwGraphicsResourceGetMappedPointer,
 * ::lwdaGraphicsSubResourceGetMappedArray
 */
LWresult LWDAAPI lwGraphicsSubResourceGetMappedArray(LWarray *pArray, LWgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel);

#if __LWDA_API_VERSION >= 5000

/**
 * \brief Get a mipmapped array through which to access a mapped graphics resource.
 *
 * Returns in \p *pMipmappedArray a mipmapped array through which the mapped graphics 
 * resource \p resource. The value set in \p *pMipmappedArray may change every time 
 * that \p resource is mapped.
 *
 * If \p resource is not a texture then it cannot be accessed via a mipmapped array and
 * ::LWDA_ERROR_NOT_MAPPED_AS_ARRAY is returned.
 * If \p resource is not mapped then ::LWDA_ERROR_NOT_MAPPED is returned.
 *
 * \param pMipmappedArray - Returned mipmapped array through which \p resource may be accessed
 * \param resource        - Mapped resource to access
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_NOT_MAPPED,
 * ::LWDA_ERROR_NOT_MAPPED_AS_ARRAY
 * \notefnerr
 *
 * \sa
 * ::lwGraphicsResourceGetMappedPointer,
 * ::lwdaGraphicsResourceGetMappedMipmappedArray
 */
LWresult LWDAAPI lwGraphicsResourceGetMappedMipmappedArray(LWmipmappedArray *pMipmappedArray, LWgraphicsResource resource);

#endif /* __LWDA_API_VERSION >= 5000 */

#if __LWDA_API_VERSION >= 3020
/**
 * \brief Get a device pointer through which to access a mapped graphics resource.
 *
 * Returns in \p *pDevPtr a pointer through which the mapped graphics resource
 * \p resource may be accessed.
 * Returns in \p pSize the size of the memory in bytes which may be accessed from that pointer.
 * The value set in \p pPointer may change every time that \p resource is mapped.
 *
 * If \p resource is not a buffer then it cannot be accessed via a pointer and
 * ::LWDA_ERROR_NOT_MAPPED_AS_POINTER is returned.
 * If \p resource is not mapped then ::LWDA_ERROR_NOT_MAPPED is returned.
 * *
 * \param pDevPtr    - Returned pointer through which \p resource may be accessed
 * \param pSize      - Returned size of the buffer accessible starting at \p *pPointer
 * \param resource   - Mapped resource to access
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_NOT_MAPPED,
 * ::LWDA_ERROR_NOT_MAPPED_AS_POINTER
 * \notefnerr
 *
 * \sa
 * ::lwGraphicsMapResources,
 * ::lwGraphicsSubResourceGetMappedArray,
 * ::lwdaGraphicsResourceGetMappedPointer
 */
LWresult LWDAAPI lwGraphicsResourceGetMappedPointer(LWdeviceptr *pDevPtr, size_t *pSize, LWgraphicsResource resource);
#endif /* __LWDA_API_VERSION >= 3020 */

/**
 * \brief Set usage flags for mapping a graphics resource
 *
 * Set \p flags for mapping the graphics resource \p resource.
 *
 * Changes to \p flags will take effect the next time \p resource is mapped.
 * The \p flags argument may be any of the following:

 * - ::LW_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by LWCA kernels.  This is the default value.
 * - ::LW_GRAPHICS_MAP_RESOURCE_FLAGS_READONLY: Specifies that LWCA kernels which
 *   access this resource will not write to this resource.
 * - ::LW_GRAPHICS_MAP_RESOURCE_FLAGS_WRITEDISCARD: Specifies that LWCA kernels
 *   which access this resource will not read from this resource and will
 *   write over the entire contents of the resource, so none of the data
 *   previously stored in the resource will be preserved.
 *
 * If \p resource is presently mapped for access by LWCA then
 * ::LWDA_ERROR_ALREADY_MAPPED is returned.
 * If \p flags is not one of the above values then ::LWDA_ERROR_ILWALID_VALUE is returned.
 *
 * \param resource - Registered resource to set flags for
 * \param flags    - Parameters for resource mapping
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ALREADY_MAPPED
 * \notefnerr
 *
 * \sa
 * ::lwGraphicsMapResources,
 * ::lwdaGraphicsResourceSetMapFlags
 */
LWresult LWDAAPI lwGraphicsResourceSetMapFlags(LWgraphicsResource resource, unsigned int flags);

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
 * issued before ::lwGraphicsMapResources() will complete before any subsequent LWCA
 * work issued in \p stream begins.
 *
 * If \p resources includes any duplicate entries then ::LWDA_ERROR_ILWALID_HANDLE is returned.
 * If any of \p resources are presently mapped for access by LWCA then ::LWDA_ERROR_ALREADY_MAPPED is returned.
 *
 * \param count      - Number of resources to map
 * \param resources  - Resources to map for LWCA usage
 * \param hStream    - Stream with which to synchronize
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_ALREADY_MAPPED,
 * ::LWDA_ERROR_UNKNOWN
 * \note_null_stream
 * \notefnerr
 *
 * \sa
 * ::lwGraphicsResourceGetMappedPointer,
 * ::lwGraphicsSubResourceGetMappedArray,
 * ::lwGraphicsUnmapResources,
 * ::lwdaGraphicsMapResources
 */
LWresult LWDAAPI lwGraphicsMapResources(unsigned int count, LWgraphicsResource *resources, LWstream hStream);

/**
 * \brief Unmap graphics resources.
 *
 * Unmaps the \p count graphics resources in \p resources.
 *
 * Once unmapped, the resources in \p resources may not be accessed by LWCA
 * until they are mapped again.
 *
 * This function provides the synchronization guarantee that any LWCA work issued
 * in \p stream before ::lwGraphicsUnmapResources() will complete before any
 * subsequently issued graphics work begins.
 *
 *
 * If \p resources includes any duplicate entries then ::LWDA_ERROR_ILWALID_HANDLE is returned.
 * If any of \p resources are not presently mapped for access by LWCA then ::LWDA_ERROR_NOT_MAPPED is returned.
 *
 * \param count      - Number of resources to unmap
 * \param resources  - Resources to unmap
 * \param hStream    - Stream with which to synchronize
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_DEINITIALIZED,
 * ::LWDA_ERROR_NOT_INITIALIZED,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_HANDLE,
 * ::LWDA_ERROR_NOT_MAPPED,
 * ::LWDA_ERROR_UNKNOWN
 * \note_null_stream
 * \notefnerr
 *
 * \sa
 * ::lwGraphicsMapResources,
 * ::lwdaGraphicsUnmapResources
 */
LWresult LWDAAPI lwGraphicsUnmapResources(unsigned int count, LWgraphicsResource *resources, LWstream hStream);

/** @} */ /* END LWDA_GRAPHICS */

LWresult LWDAAPI lwGetExportTable(const void **ppExportTable, const LWuuid *pExportTableId);


/**
 * LWCA API versioning support
 */
#if defined(__LWDA_API_VERSION_INTERNAL)
    #undef lwMemHostRegister
    #undef lwGraphicsResourceSetMapFlags
    #undef lwLinkCreate
    #undef lwLinkAddData
    #undef lwLinkAddFile
    #undef lwDeviceTotalMem
    #undef lwCtxCreate
    #undef lwModuleGetGlobal
    #undef lwMemGetInfo
    #undef lwMemAlloc
    #undef lwMemAllocPitch
    #undef lwMemFree
    #undef lwMemGetAddressRange
    #undef lwMemAllocHost
    #undef lwMemHostGetDevicePointer
    #undef lwMemcpyHtoD
    #undef lwMemcpyDtoH
    #undef lwMemcpyDtoD
    #undef lwMemcpyDtoA
    #undef lwMemcpyAtoD
    #undef lwMemcpyHtoA
    #undef lwMemcpyAtoH
    #undef lwMemcpyAtoA
    #undef lwMemcpyHtoAAsync
    #undef lwMemcpyAtoHAsync
    #undef lwMemcpy2D
    #undef lwMemcpy2DUnaligned
    #undef lwMemcpy3D
    #undef lwMemcpyHtoDAsync
    #undef lwMemcpyDtoHAsync
    #undef lwMemcpyDtoDAsync
    #undef lwMemcpy2DAsync
    #undef lwMemcpy3DAsync
    #undef lwMemsetD8
    #undef lwMemsetD16
    #undef lwMemsetD32
    #undef lwMemsetD2D8
    #undef lwMemsetD2D16
    #undef lwMemsetD2D32
    #undef lwArrayCreate
    #undef lwArrayGetDescriptor
    #undef lwArray3DCreate
    #undef lwArray3DGetDescriptor
    #undef lwTexRefSetAddress
    #undef lwTexRefSetAddress2D
    #undef lwTexRefGetAddress
    #undef lwGraphicsResourceGetMappedPointer
    #undef lwCtxDestroy
    #undef lwCtxPopLwrrent
    #undef lwCtxPushLwrrent
    #undef lwStreamDestroy
    #undef lwEventDestroy
    #undef lwMemcpy
    #undef lwMemcpyAsync
    #undef lwMemcpyPeer
    #undef lwMemcpyPeerAsync
    #undef lwMemcpy3DPeer
    #undef lwMemcpy3DPeerAsync
    #undef lwMemsetD8Async
    #undef lwMemsetD16Async
    #undef lwMemsetD32Async
    #undef lwMemsetD2D8Async
    #undef lwMemsetD2D16Async
    #undef lwMemsetD2D32Async
    #undef lwStreamGetPriority
    #undef lwStreamGetFlags
    #undef lwStreamWaitEvent
    #undef lwStreamAddCallback
    #undef lwStreamAttachMemAsync
    #undef lwStreamQuery
    #undef lwStreamSynchronize
    #undef lwEventRecord
    #undef lwLaunchKernel
    #undef lwGraphicsMapResources
    #undef lwGraphicsUnmapResources
    #undef lwStreamWriteValue32
    #undef lwStreamWaitValue32
    #undef lwStreamWriteValue64
    #undef lwStreamWaitValue64
    #undef lwStreamBatchMemOp
    #undef lwMemPrefetchAsync
    #undef lwLaunchCooperativeKernel
#endif /* __LWDA_API_VERSION_INTERNAL */

#if defined(__LWDA_API_VERSION_INTERNAL) || (__LWDA_API_VERSION >= 4000 && __LWDA_API_VERSION < 6050)
LWresult LWDAAPI lwMemHostRegister(void *p, size_t bytesize, unsigned int Flags);
#endif /* defined(__LWDA_API_VERSION_INTERNAL) || (__LWDA_API_VERSION >= 4000 && __LWDA_API_VERSION < 6050) */

#if defined(__LWDA_API_VERSION_INTERNAL) || __LWDA_API_VERSION < 6050
LWresult LWDAAPI lwGraphicsResourceSetMapFlags(LWgraphicsResource resource, unsigned int flags);
#endif /* defined(__LWDA_API_VERSION_INTERNAL) || __LWDA_API_VERSION < 6050 */

#if defined(__LWDA_API_VERSION_INTERNAL) || (__LWDA_API_VERSION >= 5050 && __LWDA_API_VERSION < 6050)
LWresult LWDAAPI lwLinkCreate(unsigned int numOptions, LWjit_option *options, void **optiolwalues, LWlinkState *stateOut);
LWresult LWDAAPI lwLinkAddData(LWlinkState state, LWjitInputType type, void *data, size_t size, const char *name,
    unsigned int numOptions, LWjit_option *options, void **optiolwalues);
LWresult LWDAAPI lwLinkAddFile(LWlinkState state, LWjitInputType type, const char *path,
    unsigned int numOptions, LWjit_option *options, void **optiolwalues);
#endif /* __LWDA_API_VERSION_INTERNAL || (__LWDA_API_VERSION >= 5050 && __LWDA_API_VERSION < 6050) */

#if defined(__LWDA_API_VERSION_INTERNAL) || (__LWDA_API_VERSION >= 3020 && __LWDA_API_VERSION < 4010)
LWresult LWDAAPI lwTexRefSetAddress2D_v2(LWtexref hTexRef, const LWDA_ARRAY_DESCRIPTOR *desc, LWdeviceptr dptr, size_t Pitch);
#endif /* __LWDA_API_VERSION_INTERNAL || (__LWDA_API_VERSION >= 3020 && __LWDA_API_VERSION < 4010) */

/**
 * LWCA API made obselete at API version 3020
 */
#if defined(__LWDA_API_VERSION_INTERNAL)
    #define LWdeviceptr                  LWdeviceptr_v1
    #define LWDA_MEMCPY2D_st             LWDA_MEMCPY2D_v1_st
    #define LWDA_MEMCPY2D                LWDA_MEMCPY2D_v1
    #define LWDA_MEMCPY3D_st             LWDA_MEMCPY3D_v1_st
    #define LWDA_MEMCPY3D                LWDA_MEMCPY3D_v1
    #define LWDA_ARRAY_DESCRIPTOR_st     LWDA_ARRAY_DESCRIPTOR_v1_st
    #define LWDA_ARRAY_DESCRIPTOR        LWDA_ARRAY_DESCRIPTOR_v1
    #define LWDA_ARRAY3D_DESCRIPTOR_st   LWDA_ARRAY3D_DESCRIPTOR_v1_st
    #define LWDA_ARRAY3D_DESCRIPTOR      LWDA_ARRAY3D_DESCRIPTOR_v1
#endif /* LWDA_FORCE_LEGACY32_INTERNAL */

#if defined(__LWDA_API_VERSION_INTERNAL) || __LWDA_API_VERSION < 3020

typedef unsigned int LWdeviceptr;

typedef struct LWDA_MEMCPY2D_st
{
    unsigned int srcXInBytes;   /**< Source X in bytes */
    unsigned int srcY;          /**< Source Y */
    LWmemorytype srcMemoryType; /**< Source memory type (host, device, array) */
    const void *srcHost;        /**< Source host pointer */
    LWdeviceptr srcDevice;      /**< Source device pointer */
    LWarray srcArray;           /**< Source array reference */
    unsigned int srcPitch;      /**< Source pitch (ignored when src is array) */

    unsigned int dstXInBytes;   /**< Destination X in bytes */
    unsigned int dstY;          /**< Destination Y */
    LWmemorytype dstMemoryType; /**< Destination memory type (host, device, array) */
    void *dstHost;              /**< Destination host pointer */
    LWdeviceptr dstDevice;      /**< Destination device pointer */
    LWarray dstArray;           /**< Destination array reference */
    unsigned int dstPitch;      /**< Destination pitch (ignored when dst is array) */

    unsigned int WidthInBytes;  /**< Width of 2D memory copy in bytes */
    unsigned int Height;        /**< Height of 2D memory copy */
} LWDA_MEMCPY2D;

typedef struct LWDA_MEMCPY3D_st
{
    unsigned int srcXInBytes;   /**< Source X in bytes */
    unsigned int srcY;          /**< Source Y */
    unsigned int srcZ;          /**< Source Z */
    unsigned int srcLOD;        /**< Source LOD */
    LWmemorytype srcMemoryType; /**< Source memory type (host, device, array) */
    const void *srcHost;        /**< Source host pointer */
    LWdeviceptr srcDevice;      /**< Source device pointer */
    LWarray srcArray;           /**< Source array reference */
    void *reserved0;            /**< Must be NULL */
    unsigned int srcPitch;      /**< Source pitch (ignored when src is array) */
    unsigned int srcHeight;     /**< Source height (ignored when src is array; may be 0 if Depth==1) */

    unsigned int dstXInBytes;   /**< Destination X in bytes */
    unsigned int dstY;          /**< Destination Y */
    unsigned int dstZ;          /**< Destination Z */
    unsigned int dstLOD;        /**< Destination LOD */
    LWmemorytype dstMemoryType; /**< Destination memory type (host, device, array) */
    void *dstHost;              /**< Destination host pointer */
    LWdeviceptr dstDevice;      /**< Destination device pointer */
    LWarray dstArray;           /**< Destination array reference */
    void *reserved1;            /**< Must be NULL */
    unsigned int dstPitch;      /**< Destination pitch (ignored when dst is array) */
    unsigned int dstHeight;     /**< Destination height (ignored when dst is array; may be 0 if Depth==1) */

    unsigned int WidthInBytes;  /**< Width of 3D memory copy in bytes */
    unsigned int Height;        /**< Height of 3D memory copy */
    unsigned int Depth;         /**< Depth of 3D memory copy */
} LWDA_MEMCPY3D;

typedef struct LWDA_ARRAY_DESCRIPTOR_st
{
    unsigned int Width;         /**< Width of array */
    unsigned int Height;        /**< Height of array */

    LWarray_format Format;      /**< Array format */
    unsigned int NumChannels;   /**< Channels per array element */
} LWDA_ARRAY_DESCRIPTOR;

typedef struct LWDA_ARRAY3D_DESCRIPTOR_st
{
    unsigned int Width;         /**< Width of 3D array */
    unsigned int Height;        /**< Height of 3D array */
    unsigned int Depth;         /**< Depth of 3D array */

    LWarray_format Format;      /**< Array format */
    unsigned int NumChannels;   /**< Channels per array element */
    unsigned int Flags;         /**< Flags */
} LWDA_ARRAY3D_DESCRIPTOR;

LWresult LWDAAPI lwDeviceTotalMem(unsigned int *bytes, LWdevice dev);
LWresult LWDAAPI lwCtxCreate(LWcontext *pctx, unsigned int flags, LWdevice dev);
LWresult LWDAAPI lwModuleGetGlobal(LWdeviceptr *dptr, unsigned int *bytes, LWmodule hmod, const char *name);
LWresult LWDAAPI lwMemGetInfo(unsigned int *free, unsigned int *total);
LWresult LWDAAPI lwMemAlloc(LWdeviceptr *dptr, unsigned int bytesize);
LWresult LWDAAPI lwMemAllocPitch(LWdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes);
LWresult LWDAAPI lwMemFree(LWdeviceptr dptr);
LWresult LWDAAPI lwMemGetAddressRange(LWdeviceptr *pbase, unsigned int *psize, LWdeviceptr dptr);
LWresult LWDAAPI lwMemAllocHost(void **pp, unsigned int bytesize);
LWresult LWDAAPI lwMemHostGetDevicePointer(LWdeviceptr *pdptr, void *p, unsigned int Flags);
LWresult LWDAAPI lwMemcpyHtoD(LWdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount);
LWresult LWDAAPI lwMemcpyDtoH(void *dstHost, LWdeviceptr srcDevice, unsigned int ByteCount);
LWresult LWDAAPI lwMemcpyDtoD(LWdeviceptr dstDevice, LWdeviceptr srcDevice, unsigned int ByteCount);
LWresult LWDAAPI lwMemcpyDtoA(LWarray dstArray, unsigned int dstOffset, LWdeviceptr srcDevice, unsigned int ByteCount);
LWresult LWDAAPI lwMemcpyAtoD(LWdeviceptr dstDevice, LWarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
LWresult LWDAAPI lwMemcpyHtoA(LWarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount);
LWresult LWDAAPI lwMemcpyAtoH(void *dstHost, LWarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
LWresult LWDAAPI lwMemcpyAtoA(LWarray dstArray, unsigned int dstOffset, LWarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
LWresult LWDAAPI lwMemcpyHtoAAsync(LWarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, LWstream hStream);
LWresult LWDAAPI lwMemcpyAtoHAsync(void *dstHost, LWarray srcArray, unsigned int srcOffset, unsigned int ByteCount, LWstream hStream);
LWresult LWDAAPI lwMemcpy2D(const LWDA_MEMCPY2D *pCopy);
LWresult LWDAAPI lwMemcpy2DUnaligned(const LWDA_MEMCPY2D *pCopy);
LWresult LWDAAPI lwMemcpy3D(const LWDA_MEMCPY3D *pCopy);
LWresult LWDAAPI lwMemcpyHtoDAsync(LWdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, LWstream hStream);
LWresult LWDAAPI lwMemcpyDtoHAsync(void *dstHost, LWdeviceptr srcDevice, unsigned int ByteCount, LWstream hStream);
LWresult LWDAAPI lwMemcpyDtoDAsync(LWdeviceptr dstDevice, LWdeviceptr srcDevice, unsigned int ByteCount, LWstream hStream);
LWresult LWDAAPI lwMemcpy2DAsync(const LWDA_MEMCPY2D *pCopy, LWstream hStream);
LWresult LWDAAPI lwMemcpy3DAsync(const LWDA_MEMCPY3D *pCopy, LWstream hStream);
LWresult LWDAAPI lwMemsetD8(LWdeviceptr dstDevice, unsigned char uc, unsigned int N);
LWresult LWDAAPI lwMemsetD16(LWdeviceptr dstDevice, unsigned short us, unsigned int N);
LWresult LWDAAPI lwMemsetD32(LWdeviceptr dstDevice, unsigned int ui, unsigned int N);
LWresult LWDAAPI lwMemsetD2D8(LWdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height);
LWresult LWDAAPI lwMemsetD2D16(LWdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height);
LWresult LWDAAPI lwMemsetD2D32(LWdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height);
LWresult LWDAAPI lwArrayCreate(LWarray *pHandle, const LWDA_ARRAY_DESCRIPTOR *pAllocateArray);
LWresult LWDAAPI lwArrayGetDescriptor(LWDA_ARRAY_DESCRIPTOR *pArrayDescriptor, LWarray hArray);
LWresult LWDAAPI lwArray3DCreate(LWarray *pHandle, const LWDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
LWresult LWDAAPI lwArray3DGetDescriptor(LWDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, LWarray hArray);
LWresult LWDAAPI lwTexRefSetAddress(unsigned int *ByteOffset, LWtexref hTexRef, LWdeviceptr dptr, unsigned int bytes);
LWresult LWDAAPI lwTexRefSetAddress2D(LWtexref hTexRef, const LWDA_ARRAY_DESCRIPTOR *desc, LWdeviceptr dptr, unsigned int Pitch);
LWresult LWDAAPI lwTexRefGetAddress(LWdeviceptr *pdptr, LWtexref hTexRef);
LWresult LWDAAPI lwGraphicsResourceGetMappedPointer(LWdeviceptr *pDevPtr, unsigned int *pSize, LWgraphicsResource resource);
#endif /* __LWDA_API_VERSION_INTERNAL || __LWDA_API_VERSION < 3020 */
#if defined(__LWDA_API_VERSION_INTERNAL) || __LWDA_API_VERSION < 4000
LWresult LWDAAPI lwCtxDestroy(LWcontext ctx);
LWresult LWDAAPI lwCtxPopLwrrent(LWcontext *pctx);
LWresult LWDAAPI lwCtxPushLwrrent(LWcontext ctx);
LWresult LWDAAPI lwStreamDestroy(LWstream hStream);
LWresult LWDAAPI lwEventDestroy(LWevent hEvent);
#endif /* __LWDA_API_VERSION_INTERNAL || __LWDA_API_VERSION < 4000 */
#if defined(__LWDA_API_VERSION_INTERNAL)
    #undef LWdeviceptr
    #undef LWDA_MEMCPY2D_st
    #undef LWDA_MEMCPY2D
    #undef LWDA_MEMCPY3D_st
    #undef LWDA_MEMCPY3D
    #undef LWDA_ARRAY_DESCRIPTOR_st
    #undef LWDA_ARRAY_DESCRIPTOR
    #undef LWDA_ARRAY3D_DESCRIPTOR_st
    #undef LWDA_ARRAY3D_DESCRIPTOR
#endif /* __LWDA_API_VERSION_INTERNAL */

#if defined(__LWDA_API_VERSION_INTERNAL)
    LWresult LWDAAPI lwMemcpyHtoD_v2(LWdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
    LWresult LWDAAPI lwMemcpyDtoH_v2(void *dstHost, LWdeviceptr srcDevice, size_t ByteCount);
    LWresult LWDAAPI lwMemcpyDtoD_v2(LWdeviceptr dstDevice, LWdeviceptr srcDevice, size_t ByteCount);
    LWresult LWDAAPI lwMemcpyDtoA_v2(LWarray dstArray, size_t dstOffset, LWdeviceptr srcDevice, size_t ByteCount);
    LWresult LWDAAPI lwMemcpyAtoD_v2(LWdeviceptr dstDevice, LWarray srcArray, size_t srcOffset, size_t ByteCount);
    LWresult LWDAAPI lwMemcpyHtoA_v2(LWarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);
    LWresult LWDAAPI lwMemcpyAtoH_v2(void *dstHost, LWarray srcArray, size_t srcOffset, size_t ByteCount);
    LWresult LWDAAPI lwMemcpyAtoA_v2(LWarray dstArray, size_t dstOffset, LWarray srcArray, size_t srcOffset, size_t ByteCount);
    LWresult LWDAAPI lwMemcpyHtoAAsync_v2(LWarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, LWstream hStream);
    LWresult LWDAAPI lwMemcpyAtoHAsync_v2(void *dstHost, LWarray srcArray, size_t srcOffset, size_t ByteCount, LWstream hStream);
    LWresult LWDAAPI lwMemcpy2D_v2(const LWDA_MEMCPY2D *pCopy);
    LWresult LWDAAPI lwMemcpy2DUnaligned_v2(const LWDA_MEMCPY2D *pCopy);
    LWresult LWDAAPI lwMemcpy3D_v2(const LWDA_MEMCPY3D *pCopy);
    LWresult LWDAAPI lwMemcpyHtoDAsync_v2(LWdeviceptr dstDevice, const void *srcHost, size_t ByteCount, LWstream hStream);
    LWresult LWDAAPI lwMemcpyDtoHAsync_v2(void *dstHost, LWdeviceptr srcDevice, size_t ByteCount, LWstream hStream);
    LWresult LWDAAPI lwMemcpyDtoDAsync_v2(LWdeviceptr dstDevice, LWdeviceptr srcDevice, size_t ByteCount, LWstream hStream);
    LWresult LWDAAPI lwMemcpy2DAsync_v2(const LWDA_MEMCPY2D *pCopy, LWstream hStream);
    LWresult LWDAAPI lwMemcpy3DAsync_v2(const LWDA_MEMCPY3D *pCopy, LWstream hStream);
    LWresult LWDAAPI lwMemsetD8_v2(LWdeviceptr dstDevice, unsigned char uc, size_t N);
    LWresult LWDAAPI lwMemsetD16_v2(LWdeviceptr dstDevice, unsigned short us, size_t N);
    LWresult LWDAAPI lwMemsetD32_v2(LWdeviceptr dstDevice, unsigned int ui, size_t N);
    LWresult LWDAAPI lwMemsetD2D8_v2(LWdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
    LWresult LWDAAPI lwMemsetD2D16_v2(LWdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
    LWresult LWDAAPI lwMemsetD2D32_v2(LWdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);
    LWresult LWDAAPI lwMemcpy(LWdeviceptr dst, LWdeviceptr src, size_t ByteCount);
    LWresult LWDAAPI lwMemcpyAsync(LWdeviceptr dst, LWdeviceptr src, size_t ByteCount, LWstream hStream);
    LWresult LWDAAPI lwMemcpyPeer(LWdeviceptr dstDevice, LWcontext dstContext, LWdeviceptr srcDevice, LWcontext srcContext, size_t ByteCount);
    LWresult LWDAAPI lwMemcpyPeerAsync(LWdeviceptr dstDevice, LWcontext dstContext, LWdeviceptr srcDevice, LWcontext srcContext, size_t ByteCount, LWstream hStream);
    LWresult LWDAAPI lwMemcpy3DPeer(const LWDA_MEMCPY3D_PEER *pCopy);
    LWresult LWDAAPI lwMemcpy3DPeerAsync(const LWDA_MEMCPY3D_PEER *pCopy, LWstream hStream);

    LWresult LWDAAPI lwMemsetD8Async(LWdeviceptr dstDevice, unsigned char uc, size_t N, LWstream hStream);
    LWresult LWDAAPI lwMemsetD16Async(LWdeviceptr dstDevice, unsigned short us, size_t N, LWstream hStream);
    LWresult LWDAAPI lwMemsetD32Async(LWdeviceptr dstDevice, unsigned int ui, size_t N, LWstream hStream);
    LWresult LWDAAPI lwMemsetD2D8Async(LWdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, LWstream hStream);
    LWresult LWDAAPI lwMemsetD2D16Async(LWdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, LWstream hStream);
    LWresult LWDAAPI lwMemsetD2D32Async(LWdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, LWstream hStream);

    LWresult LWDAAPI lwStreamGetPriority(LWstream hStream, int *priority);
    LWresult LWDAAPI lwStreamGetFlags(LWstream hStream, unsigned int *flags);
    LWresult LWDAAPI lwStreamWaitEvent(LWstream hStream, LWevent hEvent, unsigned int Flags);
    LWresult LWDAAPI lwStreamAddCallback(LWstream hStream, LWstreamCallback callback, void *userData, unsigned int flags);
    LWresult LWDAAPI lwStreamAttachMemAsync(LWstream hStream, LWdeviceptr dptr, size_t length, unsigned int flags);
    LWresult LWDAAPI lwStreamQuery(LWstream hStream);
    LWresult LWDAAPI lwStreamSynchronize(LWstream hStream);
    LWresult LWDAAPI lwEventRecord(LWevent hEvent, LWstream hStream);
    LWresult LWDAAPI lwLaunchKernel(LWfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, LWstream hStream, void **kernelParams, void **extra);
    LWresult LWDAAPI lwGraphicsMapResources(unsigned int count, LWgraphicsResource *resources, LWstream hStream);
    LWresult LWDAAPI lwGraphicsUnmapResources(unsigned int count, LWgraphicsResource *resources, LWstream hStream);
    LWresult LWDAAPI lwStreamWriteValue32(LWstream stream, LWdeviceptr addr, lwuint32_t value, unsigned int flags);
    LWresult LWDAAPI lwStreamWaitValue32(LWstream stream, LWdeviceptr addr, lwuint32_t value, unsigned int flags);
    LWresult LWDAAPI lwStreamWriteValue64(LWstream stream, LWdeviceptr addr, lwuint64_t value, unsigned int flags);
    LWresult LWDAAPI lwStreamWaitValue64(LWstream stream, LWdeviceptr addr, lwuint64_t value, unsigned int flags);
    LWresult LWDAAPI lwStreamBatchMemOp(LWstream stream, unsigned int count, LWstreamBatchMemOpParams *paramArray, unsigned int flags);
    LWresult LWDAAPI lwMemPrefetchAsync(LWdeviceptr devPtr, size_t count, LWdevice dstDevice, LWstream hStream);
    LWresult LWDAAPI lwLaunchCooperativeKernel(LWfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, LWstream hStream, void **kernelParams);
#endif

#ifdef __cplusplus
}
#endif

#undef __LWDA_API_VERSION

#endif /* __lwda_lwda_h__ */

