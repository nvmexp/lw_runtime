/*
 * Copyright 1993-2015 LWPU Corporation.  All rights reserved.
 *
 * Please refer to the LWPU end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#ifndef __lwda_drvapi_dynlink_lwda_h__
#define __lwda_drvapi_dynlink_lwda_h__

#include <stdlib.h>


#define __lwda_lwda_h__ 1

/**
 * LWCA API versioning support
 */
#define __LWDA_API_VERSION 5000

/**
 * \defgroup LWDA_DRIVER LWCA Driver API
 *
 * This section describes the low-level LWCA driver application programming
 * interface.
 *
 * @{
 */

/**
 * \defgroup LWDA_TYPES Data types used by LWCA driver
 * @{
 */

/**
 * LWCA API version number
 */
#define LWDA_VERSION 3020 /* 3.2 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * LWCA device pointer
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
typedef unsigned long long LWtexObject;                   /**< LWCA texture object */
typedef unsigned long long LWsurfObject;                  /**< LWCA surface object */

typedef struct LWuuid_st                                  /**< LWCA definition of UUID */
{
    char bytes[16];
} LWuuid;

/**
 * Context creation flags
 */
typedef enum LWctx_flags_enum
{
    LW_CTX_SCHED_AUTO          = 0x00, /**< Automatic scheduling */
    LW_CTX_SCHED_SPIN          = 0x01, /**< Set spin as default scheduling */
    LW_CTX_SCHED_YIELD         = 0x02, /**< Set yield as default scheduling */
    LW_CTX_SCHED_BLOCKING_SYNC = 0x04, /**< Set blocking synchronization as default scheduling */
    LW_CTX_BLOCKING_SYNC       = 0x04, /**< Set blocking synchronization as default scheduling \deprecated */
    LW_CTX_MAP_HOST            = 0x08, /**< Support mapped pinned allocations */
    LW_CTX_LMEM_RESIZE_TO_MAX  = 0x10, /**< Keep local memory allocation after launch */
#if __LWDA_API_VERSION < 4000
    LW_CTX_SCHED_MASK          = 0x03,
    LW_CTX_FLAGS_MASK          = 0x1f
#else
    LW_CTX_SCHED_MASK          = 0x07,
    LW_CTX_PRIMARY             = 0x20, /**< Initialize and return the primary context */
    LW_CTX_FLAGS_MASK          = 0x3f
#endif
} LWctx_flags;

/**
 * Event creation flags
 */
typedef enum LWevent_flags_enum
{
    LW_EVENT_DEFAULT        = 0, /**< Default event flag */
    LW_EVENT_BLOCKING_SYNC  = 1, /**< Event uses blocking synchronization */
    LW_EVENT_DISABLE_TIMING = 2  /**< Event will not record timing data */
} LWevent_flags;

/**
 * Array formats
 */
typedef enum LWarray_format_enum
{
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
typedef enum LWaddress_mode_enum
{
    LW_TR_ADDRESS_MODE_WRAP   = 0, /**< Wrapping address mode */
    LW_TR_ADDRESS_MODE_CLAMP  = 1, /**< Clamp to edge address mode */
    LW_TR_ADDRESS_MODE_MIRROR = 2, /**< Mirror address mode */
    LW_TR_ADDRESS_MODE_BORDER = 3  /**< Border address mode */
} LWaddress_mode;

/**
 * Texture reference filtering modes
 */
typedef enum LWfilter_mode_enum
{
    LW_TR_FILTER_MODE_POINT  = 0, /**< Point filter mode */
    LW_TR_FILTER_MODE_LINEAR = 1  /**< Linear filter mode */
} LWfilter_mode;

/**
 * Device properties
 */
typedef enum LWdevice_attribute_enum
{
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
    LW_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,                        /**< Peak clock frequency in kilohertz */
    LW_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,                 /**< Alignment requirement for textures */
    LW_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,                       /**< Device can possibly copy memory and execute a kernel conlwrrently */
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
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,     /**< Maximum texture array width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,    /**< Maximum texture array height */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29, /**< Maximum slices in a texture array */
    LW_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,                 /**< Alignment requirement for surfaces */
    LW_DEVICE_ATTRIBUTE_CONLWRRENT_KERNELS = 31,                /**< Device can possibly execute multiple kernels conlwrrently */
    LW_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,                       /**< Device has ECC support enabled */
    LW_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,                        /**< PCI bus ID of the device */
    LW_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,                     /**< PCI device ID of the device */
    LW_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,                         /**< Device is using TCC driver model */
    LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,          /**< Major compute capability version number */
    LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76           /**< Minor compute capability version number */
#if __LWDA_API_VERSION >= 4000
                                     , LW_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,                 /**< Peak memory clock frequency in kilohertz */
    LW_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,           /**< Global memory bus width in bits */
    LW_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,                     /**< Size of L2 cache in bytes */
    LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,    /**< Maximum resident threads per multiprocessor */
    LW_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,                /**< Number of asynchronous engines */
    LW_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,                /**< Device uses shares a unified address space with the host */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,   /**< Maximum 1D layered texture width */
    LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43   /**< Maximum layers in a 1D layered texture */
#endif
} LWdevice_attribute;

/**
 * Legacy device properties
 */
typedef struct LWdevprop_st
{
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
 * Function properties
 */
typedef enum LWfunction_attribute_enum
{
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

    LW_FUNC_ATTRIBUTE_MAX
} LWfunction_attribute;

/**
 * Function cache configurations
 */
typedef enum LWfunc_cache_enum
{
    LW_FUNC_CACHE_PREFER_NONE    = 0x00, /**< no preference for shared memory or L1 (default) */
    LW_FUNC_CACHE_PREFER_SHARED  = 0x01, /**< prefer larger shared memory and smaller L1 cache */
    LW_FUNC_CACHE_PREFER_L1      = 0x02  /**< prefer larger L1 cache and smaller shared memory */
} LWfunc_cache;

/**
 * Shared memory configurations
 */
typedef enum LWsharedconfig_enum
{
    LW_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE    = 0x00, /**< set default shared memory bank size */
    LW_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE  = 0x01, /**< set shared memory bank width to four bytes */
    LW_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 0x02  /**< set shared memory bank width to eight bytes */
} LWsharedconfig;

/**
 * Memory types
 */
typedef enum LWmemorytype_enum
{
    LW_MEMORYTYPE_HOST    = 0x01,    /**< Host memory */
    LW_MEMORYTYPE_DEVICE  = 0x02,    /**< Device memory */
    LW_MEMORYTYPE_ARRAY   = 0x03     /**< Array memory */
#if __LWDA_API_VERSION >= 4000
                            , LW_MEMORYTYPE_UNIFIED = 0x04     /**< Unified device or host memory */
#endif
} LWmemorytype;

/**
 * Compute Modes
 */
typedef enum LWcomputemode_enum
{
    LW_COMPUTEMODE_DEFAULT           = 0,  /**< Default compute mode (Multiple contexts allowed per device) */
    LW_COMPUTEMODE_PROHIBITED        = 2  /**< Compute-prohibited mode (No contexts can be created on this device at this time) */
#if __LWDA_API_VERSION >= 4000
                                       , LW_COMPUTEMODE_EXCLUSIVE_PROCESS = 3  /**< Compute-exclusive-process mode (Only one context used by a single process can be present on this device at a time) */
#endif
} LWcomputemode;

/**
 * Online compiler options
 */
typedef enum LWjit_option_enum
{
    /**
     * Max number of registers that a thread may use.\n
     * Option type: unsigned int
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
     * Option type: unsigned int
     */
    LW_JIT_THREADS_PER_BLOCK,

    /**
     * Returns a float value in the option of the wall clock time, in
     * milliseconds, spent creating the lwbin\n
     * Option type: float
     */
    LW_JIT_WALL_TIME,

    /**
     * Pointer to a buffer in which to print any log messsages from PTXAS
     * that are informational in nature (the buffer size is specified via
     * option ::LW_JIT_INFO_LOG_BUFFER_SIZE_BYTES) \n
     * Option type: char*
     */
    LW_JIT_INFO_LOG_BUFFER,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int
     */
    LW_JIT_INFO_LOG_BUFFER_SIZE_BYTES,

    /**
     * Pointer to a buffer in which to print any log messages from PTXAS that
     * reflect errors (the buffer size is specified via option
     * ::LW_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)\n
     * Option type: char*
     */
    LW_JIT_ERROR_LOG_BUFFER,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int
     */
    LW_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,

    /**
     * Level of optimizations to apply to generated code (0 - 4), with 4
     * being the default and highest level of optimizations.\n
     * Option type: unsigned int
     */
    LW_JIT_OPTIMIZATION_LEVEL,

    /**
     * No option value required. Determines the target based on the current
     * attached context (default)\n
     * Option type: No option value needed
     */
    LW_JIT_TARGET_FROM_LWCONTEXT,

    /**
     * Target is chosen based on supplied ::LWjit_target_enum.\n
     * Option type: unsigned int for enumerated type ::LWjit_target_enum
     */
    LW_JIT_TARGET,

    /**
     * Specifies choice of fallback strategy if matching lwbin is not found.
     * Choice is based on supplied ::LWjit_fallback_enum.\n
     * Option type: unsigned int for enumerated type ::LWjit_fallback_enum
     */
    LW_JIT_FALLBACK_STRATEGY

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
    LW_PREFER_PTX = 0,  /**< Prefer to compile ptx */
    LW_PREFER_BINARY    /**< Prefer to fall back to compatible binary code */
} LWjit_fallback;

/**
 * Flags to register a graphics resource
 */
typedef enum LWgraphicsRegisterFlags_enum
{
    LW_GRAPHICS_REGISTER_FLAGS_NONE          = 0x00,
    LW_GRAPHICS_REGISTER_FLAGS_READ_ONLY     = 0x01,
    LW_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x02,
    LW_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST  = 0x04
} LWgraphicsRegisterFlags;

/**
 * Flags for mapping and unmapping interop resources
 */
typedef enum LWgraphicsMapResourceFlags_enum
{
    LW_GRAPHICS_MAP_RESOURCE_FLAGS_NONE          = 0x00,
    LW_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY     = 0x01,
    LW_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02
} LWgraphicsMapResourceFlags;

/**
 * Array indices for lwbe faces
 */
typedef enum LWarray_lwbemap_face_enum
{
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
typedef enum LWlimit_enum
{
    LW_LIMIT_STACK_SIZE        = 0x00, /**< GPU thread stack size */
    LW_LIMIT_PRINTF_FIFO_SIZE  = 0x01, /**< GPU printf FIFO size */
    LW_LIMIT_MALLOC_HEAP_SIZE  = 0x02  /**< GPU malloc heap size */
} LWlimit;

/**
 * Resource types
 */
typedef enum LWresourcetype_enum
{
    LW_RESOURCE_TYPE_ARRAY           = 0x00, /**< Array resoure */
    LW_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01, /**< Mipmapped array resource */
    LW_RESOURCE_TYPE_LINEAR          = 0x02, /**< Linear resource */
    LW_RESOURCE_TYPE_PITCH2D         = 0x03  /**< Pitch 2D resource */
} LWresourcetype;

/**
 * Error codes
 */
typedef enum lwdaError_enum
{
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
     * This indicates profiling APIs are called while application is running
     * in visual profiler mode.
    */
    LWDA_ERROR_PROFILER_DISABLED           = 5,
    /**
     * This indicates profiling has not been initialized for this context.
     * Call lwProfilerInitialize() to resolve this.
    */
    LWDA_ERROR_PROFILER_NOT_INITIALIZED       = 6,
    /**
     * This indicates profiler has already been started and probably
     * lwProfilerStart() is incorrectly called.
    */
    LWDA_ERROR_PROFILER_ALREADY_STARTED       = 7,
    /**
     * This indicates profiler has already been stopped and probably
     * lwProfilerStop() is incorrectly called.
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
     * An exception oclwrred on the device while exelwting a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The context cannot be used, so it must
     * be destroyed (and a new one should be created). All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using LWCA.
     */
    LWDA_ERROR_LAUNCH_FAILED                  = 700,

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
     * ::LW_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
     * context cannot be used (and must be destroyed similar to
     * ::LWDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
     * this context are invalid and must be reconstructed if the program is to
     * continue using LWCA.
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
    LWDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,

    /**
     * This error indicates that a call to ::lwMemPeerRegister is trying to
     * register memory from a context which has not had peer access
     * enabled yet via ::lwCtxEnablePeerAccess(), or that
     * ::lwCtxDisablePeerAccess() is trying to disable peer access
     * which has not been enabled yet.
     */
    LWDA_ERROR_PEER_ACCESS_NOT_ENABLED    = 705,

    /**
     * This error indicates that a call to ::lwMemPeerRegister is trying to
     * register already-registered memory.
     */
    LWDA_ERROR_PEER_MEMORY_ALREADY_REGISTERED = 706,

    /**
     * This error indicates that a call to ::lwMemPeerUnregister is trying to
     * unregister memory that has not been registered.
     */
    LWDA_ERROR_PEER_MEMORY_NOT_REGISTERED     = 707,

    /**
     * This error indicates that ::lwCtxCreate was called with the flag
     * ::LW_CTX_PRIMARY on a device which already has initialized its
     * primary context.
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
     * This indicates that an unknown internal error has oclwrred.
     */
    LWDA_ERROR_UNKNOWN                        = 999
} LWresult;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
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

#if __LWDA_API_VERSION >= 4000
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
 * If set, peer memory is mapped into LWCA address space and
 * ::lwMemPeerGetDevicePointer() may be called on the host pointer.
 * Flag for ::lwMemPeerRegister()
 */
#define LW_MEMPEERREGISTER_DEVICEMAP    0x02
#endif

#if __LWDA_API_VERSION >= 3020

/**
 * 2D memory copy parameters
 */
typedef struct LWDA_MEMCPY2D_st
{
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
typedef struct LWDA_MEMCPY3D_st
{
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
typedef struct LWDA_MEMCPY3D_PEER_st
{
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

    union
    {
        struct
        {
            LWarray hArray;                   /**< LWCA array */
        } array;
        struct
        {
            LWmipmappedArray hMipmappedArray; /**< LWCA mipmapped array */
        } mipmap;
        struct
        {
            LWdeviceptr devPtr;               /**< Device pointer */
            LWarray_format format;            /**< Array format */
            unsigned int numChannels;         /**< Channels per array element */
            size_t sizeInBytes;               /**< Size in bytes */
        } linear;
        struct
        {
            LWdeviceptr devPtr;               /**< Device pointer */
            LWarray_format format;            /**< Array format */
            unsigned int numChannels;         /**< Channels per array element */
            size_t width;                     /**< Width of the array in elements */
            size_t height;                    /**< Height of the array in elements */
            size_t pitchInBytes;              /**< Pitch between two rows in bytes */
        } pitch2D;
        struct
        {
            int reserved[32];
        } __reserved;
    } res;

    unsigned int flags;                       /**< Flags (must be zero) */
} LWDA_RESOURCE_DESC;

/**
 * Texture descriptor
 */
typedef struct LWDA_TEXTURE_DESC_st
{
    LWaddress_mode addressMode[3];  /**< Address modes */
    LWfilter_mode filterMode;       /**< Filter mode */
    unsigned int flags;             /**< Flags */
    unsigned int maxAnisotropy;     /**< Maximum anistropy ratio */
    LWfilter_mode mipmapFilterMode; /**< Mipmap filter mode */
    float mipmapLevelBias;          /**< Mipmap level bias */
    float minMipmapLevelClamp;      /**< Mipmap minimum level clamp */
    float maxMipmapLevelClamp;      /**< Mipmap maximum level clamp */
    int _reserved[16];
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
    unsigned int _reserved[16];
} LWDA_RESOURCE_VIEW_DESC;

/**
 * GPU Direct v3 tokens
 */
typedef struct LWDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
{
    unsigned long long p2pToken;
    unsigned int vaSpaceToken;
} LWDA_POINTER_ATTRIBUTE_P2P_TOKENS;
#endif



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

#endif /* (__LWDA_API_VERSION_INTERNAL) || __LWDA_API_VERSION < 3020 */

/*
 * If set, the LWCA array contains an array of 2D slices
 * and the Depth member of LWDA_ARRAY3D_DESCRIPTOR specifies
 * the number of slices, not the depth of a 3D array.
 */
#define LWDA_ARRAY3D_2DARRAY        0x01

/**
 * This flag must be set in order to bind a surface reference
 * to the LWCA array
 */
#define LWDA_ARRAY3D_SURFACE_LDST   0x02

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
 * For texture references loaded into the module, use default texunit from
 * texture reference.
 */
#define LW_PARAM_TR_DEFAULT -1

/** @} */ /* END LWDA_TYPES */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define LWDAAPI __stdcall
#else
#define LWDAAPI
#endif

/**
 * \defgroup LWDA_INITIALIZE Initialization
 *
 * This section describes the initialization functions of the low-level LWCA
 * driver application programming interface.
 *
 * @{
 */

/*********************************
 ** Initialization
 *********************************/
typedef LWresult  LWDAAPI tlwInit(unsigned int Flags);

/*********************************
 ** Driver Version Query
 *********************************/
typedef LWresult  LWDAAPI tlwDriverGetVersion(int *driverVersion);

/************************************
 **
 **    Device management
 **
 ***********************************/

typedef LWresult  LWDAAPI tlwDeviceGet(LWdevice *device, int ordinal);
typedef LWresult  LWDAAPI tlwDeviceGetCount(int *count);
typedef LWresult  LWDAAPI tlwDeviceGetName(char *name, int len, LWdevice dev);
typedef LWresult  LWDAAPI tlwDeviceComputeCapability(int *major, int *minor, LWdevice dev);
#if __LWDA_API_VERSION >= 3020
typedef LWresult  LWDAAPI tlwDeviceTotalMem(size_t *bytes, LWdevice dev);
#else
typedef LWresult  LWDAAPI tlwDeviceTotalMem(unsigned int *bytes, LWdevice dev);
#endif

typedef LWresult  LWDAAPI tlwDeviceGetProperties(LWdevprop *prop, LWdevice dev);
typedef LWresult  LWDAAPI tlwDeviceGetAttribute(int *pi, LWdevice_attribute attrib, LWdevice dev);
typedef LWresult  LWDAAPI tlwGetErrorString(LWresult error, const char **pStr);

/************************************
 **
 **    Context management
 **
 ***********************************/

typedef LWresult  LWDAAPI tlwCtxCreate(LWcontext *pctx, unsigned int flags, LWdevice dev);
typedef LWresult  LWDAAPI tlwCtxDestroy(LWcontext ctx);
typedef LWresult  LWDAAPI tlwCtxAttach(LWcontext *pctx, unsigned int flags);
typedef LWresult  LWDAAPI tlwCtxDetach(LWcontext ctx);
typedef LWresult  LWDAAPI tlwCtxPushLwrrent(LWcontext ctx);
typedef LWresult  LWDAAPI tlwCtxPopLwrrent(LWcontext *pctx);

typedef LWresult  LWDAAPI tlwCtxSetLwrrent(LWcontext ctx);
typedef LWresult  LWDAAPI tlwCtxGetLwrrent(LWcontext *pctx);

typedef LWresult  LWDAAPI tlwCtxGetDevice(LWdevice *device);
typedef LWresult  LWDAAPI tlwCtxSynchronize(void);


/************************************
 **
 **    Module management
 **
 ***********************************/

typedef LWresult  LWDAAPI tlwModuleLoad(LWmodule *module, const char *fname);
typedef LWresult  LWDAAPI tlwModuleLoadData(LWmodule *module, const void *image);
typedef LWresult  LWDAAPI tlwModuleLoadDataEx(LWmodule *module, const void *image, unsigned int numOptions, LWjit_option *options, void **optiolwalues);
typedef LWresult  LWDAAPI tlwModuleLoadFatBinary(LWmodule *module, const void *fatLwbin);
typedef LWresult  LWDAAPI tlwModuleUnload(LWmodule hmod);
typedef LWresult  LWDAAPI tlwModuleGetFunction(LWfunction *hfunc, LWmodule hmod, const char *name);

#if __LWDA_API_VERSION >= 3020
typedef LWresult  LWDAAPI tlwModuleGetGlobal(LWdeviceptr *dptr, size_t *bytes, LWmodule hmod, const char *name);
#else
typedef LWresult  LWDAAPI tlwModuleGetGlobal(LWdeviceptr *dptr, unsigned int *bytes, LWmodule hmod, const char *name);
#endif

typedef LWresult  LWDAAPI tlwModuleGetTexRef(LWtexref *pTexRef, LWmodule hmod, const char *name);
typedef LWresult  LWDAAPI tlwModuleGetSurfRef(LWsurfref *pSurfRef, LWmodule hmod, const char *name);

/************************************
 **
 **    Memory management
 **
 ***********************************/
#if __LWDA_API_VERSION >= 3020
typedef LWresult LWDAAPI tlwMemGetInfo(size_t *free, size_t *total);
typedef LWresult LWDAAPI tlwMemAlloc(LWdeviceptr *dptr, size_t bytesize);
typedef LWresult LWDAAPI tlwMemGetAddressRange(LWdeviceptr *pbase, size_t *psize, LWdeviceptr dptr);
typedef LWresult LWDAAPI tlwMemAllocPitch(LWdeviceptr *dptr,
                                          size_t *pPitch,
                                          size_t WidthInBytes,
                                          size_t Height,
                                          // size of biggest r/w to be performed by kernels on this memory
                                          // 4, 8 or 16 bytes
                                          unsigned int ElementSizeBytes
                                         );
#else
typedef LWresult LWDAAPI tlwMemGetInfo(unsigned int *free, unsigned int *total);
typedef LWresult LWDAAPI tlwMemAlloc(LWdeviceptr *dptr, unsigned int bytesize);
typedef LWresult LWDAAPI tlwMemGetAddressRange(LWdeviceptr *pbase, unsigned int *psize, LWdeviceptr dptr);
typedef LWresult LWDAAPI tlwMemAllocPitch(LWdeviceptr *dptr,
                                          unsigned int *pPitch,
                                          unsigned int WidthInBytes,
                                          unsigned int Height,
                                          // size of biggest r/w to be performed by kernels on this memory
                                          // 4, 8 or 16 bytes
                                          unsigned int ElementSizeBytes
                                         );
#endif

typedef LWresult LWDAAPI tlwMemFree(LWdeviceptr dptr);

#if __LWDA_API_VERSION >= 3020
typedef LWresult LWDAAPI tlwMemAllocHost(void **pp, size_t bytesize);
typedef LWresult LWDAAPI tlwMemHostGetDevicePointer(LWdeviceptr *pdptr, void *p, unsigned int Flags);
#else
typedef LWresult LWDAAPI tlwMemAllocHost(void **pp, unsigned int bytesize);
#endif

typedef LWresult LWDAAPI tlwMemFreeHost(void *p);
typedef LWresult LWDAAPI tlwMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags);

typedef LWresult LWDAAPI tlwMemHostGetFlags(unsigned int *pFlags, void *p);

#if __LWDA_API_VERSION >= 4010
/**
 * Interprocess Handles
 */
#define LW_IPC_HANDLE_SIZE 64

typedef struct LWipcEventHandle_st
{
    char reserved[LW_IPC_HANDLE_SIZE];
} LWipcEventHandle;

typedef struct LWipcMemHandle_st
{
    char reserved[LW_IPC_HANDLE_SIZE];
} LWipcMemHandle;

typedef enum LWipcMem_flags_enum
{
    LW_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1 /**< Automatically enable peer access between remote devices as needed */
} LWipcMem_flags;

typedef LWresult LWDAAPI tlwDeviceGetByPCIBusId(LWdevice *dev, char *pciBusId);
typedef LWresult LWDAAPI tlwDeviceGetPCIBusId(char *pciBusId, int len, LWdevice dev);
typedef LWresult LWDAAPI tlwIpcGetEventHandle(LWipcEventHandle *pHandle, LWevent event);
typedef LWresult LWDAAPI tlwIpcOpenEventHandle(LWevent *phEvent, LWipcEventHandle handle);
typedef LWresult LWDAAPI tlwIpcGetMemHandle(LWipcMemHandle *pHandle, LWdeviceptr dptr);
typedef LWresult LWDAAPI tlwIpcOpenMemHandle(LWdeviceptr *pdptr, LWipcMemHandle handle, unsigned int Flags);
typedef LWresult LWDAAPI tlwIpcCloseMemHandle(LWdeviceptr dptr);
#endif

typedef LWresult LWDAAPI tlwMemHostRegister(void *p, size_t bytesize, unsigned int Flags);
typedef LWresult LWDAAPI tlwMemHostUnregister(void *p);;
typedef LWresult LWDAAPI tlwMemcpy(LWdeviceptr dst, LWdeviceptr src, size_t ByteCount);
typedef LWresult LWDAAPI tlwMemcpyPeer(LWdeviceptr dstDevice, LWcontext dstContext, LWdeviceptr srcDevice, LWcontext srcContext, size_t ByteCount);

/************************************
 **
 **    Synchronous Memcpy
 **
 ** Intra-device memcpy's done with these functions may execute in parallel with the CPU,
 ** but if host memory is ilwolved, they wait until the copy is done before returning.
 **
 ***********************************/

// 1D functions
#if __LWDA_API_VERSION >= 3020
// system <-> device memory
typedef LWresult  LWDAAPI tlwMemcpyHtoD(LWdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
typedef LWresult  LWDAAPI tlwMemcpyDtoH(void *dstHost, LWdeviceptr srcDevice, size_t ByteCount);

// device <-> device memory
typedef LWresult  LWDAAPI tlwMemcpyDtoD(LWdeviceptr dstDevice, LWdeviceptr srcDevice, size_t ByteCount);

// device <-> array memory
typedef LWresult  LWDAAPI tlwMemcpyDtoA(LWarray dstArray, size_t dstOffset, LWdeviceptr srcDevice, size_t ByteCount);
typedef LWresult  LWDAAPI tlwMemcpyAtoD(LWdeviceptr dstDevice, LWarray srcArray, size_t srcOffset, size_t ByteCount);

// system <-> array memory
typedef LWresult  LWDAAPI tlwMemcpyHtoA(LWarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);
typedef LWresult  LWDAAPI tlwMemcpyAtoH(void *dstHost, LWarray srcArray, size_t srcOffset, size_t ByteCount);

// array <-> array memory
typedef LWresult  LWDAAPI tlwMemcpyAtoA(LWarray dstArray, size_t dstOffset, LWarray srcArray, size_t srcOffset, size_t ByteCount);
#else
// system <-> device memory
typedef LWresult  LWDAAPI tlwMemcpyHtoD(LWdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount);
typedef LWresult  LWDAAPI tlwMemcpyDtoH(void *dstHost, LWdeviceptr srcDevice, unsigned int ByteCount);

// device <-> device memory
typedef LWresult  LWDAAPI tlwMemcpyDtoD(LWdeviceptr dstDevice, LWdeviceptr srcDevice, unsigned int ByteCount);

// device <-> array memory
typedef LWresult  LWDAAPI tlwMemcpyDtoA(LWarray dstArray, unsigned int dstOffset, LWdeviceptr srcDevice, unsigned int ByteCount);
typedef LWresult  LWDAAPI tlwMemcpyAtoD(LWdeviceptr dstDevice, LWarray srcArray, unsigned int srcOffset, unsigned int ByteCount);

// system <-> array memory
typedef LWresult  LWDAAPI tlwMemcpyHtoA(LWarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount);
typedef LWresult  LWDAAPI tlwMemcpyAtoH(void *dstHost, LWarray srcArray, unsigned int srcOffset, unsigned int ByteCount);

// array <-> array memory
typedef LWresult  LWDAAPI tlwMemcpyAtoA(LWarray dstArray, unsigned int dstOffset, LWarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
#endif

// 2D memcpy

typedef LWresult  LWDAAPI tlwMemcpy2D(const LWDA_MEMCPY2D *pCopy);
typedef LWresult  LWDAAPI tlwMemcpy2DUnaligned(const LWDA_MEMCPY2D *pCopy);

// 3D memcpy

typedef LWresult  LWDAAPI tlwMemcpy3D(const LWDA_MEMCPY3D *pCopy);

/************************************
 **
 **    Asynchronous Memcpy
 **
 ** Any host memory ilwolved must be DMA'able (e.g., allocated with lwMemAllocHost).
 ** memcpy's done with these functions execute in parallel with the CPU and, if
 ** the hardware is available, may execute in parallel with the GPU.
 ** Asynchronous memcpy must be accompanied by appropriate stream synchronization.
 **
 ***********************************/

// 1D functions
#if __LWDA_API_VERSION >= 3020
// system <-> device memory
typedef LWresult  LWDAAPI tlwMemcpyHtoDAsync(LWdeviceptr dstDevice,
                                             const void *srcHost, size_t ByteCount, LWstream hStream);
typedef LWresult  LWDAAPI tlwMemcpyDtoHAsync(void *dstHost,
                                             LWdeviceptr srcDevice, size_t ByteCount, LWstream hStream);

// device <-> device memory
typedef LWresult  LWDAAPI tlwMemcpyDtoDAsync(LWdeviceptr dstDevice,
                                             LWdeviceptr srcDevice, size_t ByteCount, LWstream hStream);

// system <-> array memory
typedef LWresult  LWDAAPI tlwMemcpyHtoAAsync(LWarray dstArray, size_t dstOffset,
                                             const void *srcHost, size_t ByteCount, LWstream hStream);
typedef LWresult  LWDAAPI tlwMemcpyAtoHAsync(void *dstHost, LWarray srcArray, size_t srcOffset,
                                             size_t ByteCount, LWstream hStream);

#else
// system <-> device memory
typedef LWresult  LWDAAPI tlwMemcpyHtoDAsync(LWdeviceptr dstDevice,
                                             const void *srcHost, unsigned int ByteCount, LWstream hStream);
typedef LWresult  LWDAAPI tlwMemcpyDtoHAsync(void *dstHost,
                                             LWdeviceptr srcDevice, unsigned int ByteCount, LWstream hStream);

// device <-> device memory
typedef LWresult  LWDAAPI tlwMemcpyDtoDAsync(LWdeviceptr dstDevice,
                                             LWdeviceptr srcDevice, unsigned int ByteCount, LWstream hStream);

// system <-> array memory
typedef LWresult  LWDAAPI tlwMemcpyHtoAAsync(LWarray dstArray, unsigned int dstOffset,
                                             const void *srcHost, unsigned int ByteCount, LWstream hStream);
typedef LWresult  LWDAAPI tlwMemcpyAtoHAsync(void *dstHost, LWarray srcArray, unsigned int srcOffset,
                                             unsigned int ByteCount, LWstream hStream);
#endif

// 2D memcpy
typedef LWresult  LWDAAPI tlwMemcpy2DAsync(const LWDA_MEMCPY2D *pCopy, LWstream hStream);

// 3D memcpy
typedef LWresult  LWDAAPI tlwMemcpy3DAsync(const LWDA_MEMCPY3D *pCopy, LWstream hStream);

/************************************
 **
 **    Memset
 **
 ***********************************/
typedef LWresult  LWDAAPI tlwMemsetD8(LWdeviceptr dstDevice, unsigned char uc, unsigned int N);
typedef LWresult  LWDAAPI tlwMemsetD16(LWdeviceptr dstDevice, unsigned short us, unsigned int N);
typedef LWresult  LWDAAPI tlwMemsetD32(LWdeviceptr dstDevice, unsigned int ui, unsigned int N);

#if __LWDA_API_VERSION >= 3020
typedef LWresult  LWDAAPI tlwMemsetD2D8(LWdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, size_t Width, size_t Height);
typedef LWresult  LWDAAPI tlwMemsetD2D16(LWdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, size_t Width, size_t Height);
typedef LWresult  LWDAAPI tlwMemsetD2D32(LWdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, size_t Width, size_t Height);
#else
typedef LWresult  LWDAAPI tlwMemsetD2D8(LWdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height);
typedef LWresult  LWDAAPI tlwMemsetD2D16(LWdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height);
typedef LWresult  LWDAAPI tlwMemsetD2D32(LWdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height);
#endif

/************************************
 **
 **    Function management
 **
 ***********************************/


typedef LWresult LWDAAPI tlwFuncSetBlockShape(LWfunction hfunc, int x, int y, int z);
typedef LWresult LWDAAPI tlwFuncSetSharedSize(LWfunction hfunc, unsigned int bytes);
typedef LWresult LWDAAPI tlwFuncGetAttribute(int *pi, LWfunction_attribute attrib, LWfunction hfunc);
typedef LWresult LWDAAPI tlwFuncSetCacheConfig(LWfunction hfunc, LWfunc_cache config);
typedef LWresult LWDAAPI tlwFuncSetSharedMemConfig(LWfunction hfunc, LWsharedconfig config);

typedef LWresult LWDAAPI tlwLaunchKernel(LWfunction f,
                                         unsigned int gridDimX,  unsigned int gridDimY,  unsigned int gridDimZ,
                                         unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                         unsigned int sharedMemBytes,
                                         LWstream hStream, void **kernelParams, void **extra);

/************************************
 **
 **    Array management
 **
 ***********************************/

typedef LWresult  LWDAAPI tlwArrayCreate(LWarray *pHandle, const LWDA_ARRAY_DESCRIPTOR *pAllocateArray);
typedef LWresult  LWDAAPI tlwArrayGetDescriptor(LWDA_ARRAY_DESCRIPTOR *pArrayDescriptor, LWarray hArray);
typedef LWresult  LWDAAPI tlwArrayDestroy(LWarray hArray);

typedef LWresult  LWDAAPI tlwArray3DCreate(LWarray *pHandle, const LWDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
typedef LWresult  LWDAAPI tlwArray3DGetDescriptor(LWDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, LWarray hArray);

#if __LWDA_API_VERSION >= 5000
typedef LWresult LWDAAPI tlwMipmappedArrayCreate(LWmipmappedArray *pHandle, const LWDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, unsigned int numMipmapLevels);
typedef LWresult LWDAAPI tlwMipmappedArrayGetLevel(LWarray *pLevelArray, LWmipmappedArray hMipmappedArray, unsigned int level);
typedef LWresult LWDAAPI tlwMipmappedArrayDestroy(LWmipmappedArray hMipmappedArray);
#endif


/************************************
 **
 **    Texture reference management
 **
 ***********************************/
typedef LWresult  LWDAAPI tlwTexRefCreate(LWtexref *pTexRef);
typedef LWresult  LWDAAPI tlwTexRefDestroy(LWtexref hTexRef);

typedef LWresult  LWDAAPI tlwTexRefSetArray(LWtexref hTexRef, LWarray hArray, unsigned int Flags);

#if __LWDA_API_VERSION >= 3020
typedef LWresult  LWDAAPI tlwTexRefSetAddress(size_t *ByteOffset, LWtexref hTexRef, LWdeviceptr dptr, size_t bytes);
typedef LWresult  LWDAAPI tlwTexRefSetAddress2D(LWtexref hTexRef, const LWDA_ARRAY_DESCRIPTOR *desc, LWdeviceptr dptr, size_t Pitch);
#else
typedef LWresult  LWDAAPI tlwTexRefSetAddress(unsigned int *ByteOffset, LWtexref hTexRef, LWdeviceptr dptr, unsigned int bytes);
typedef LWresult  LWDAAPI tlwTexRefSetAddress2D(LWtexref hTexRef, const LWDA_ARRAY_DESCRIPTOR *desc, LWdeviceptr dptr, unsigned int Pitch);
#endif

typedef LWresult  LWDAAPI tlwTexRefSetFormat(LWtexref hTexRef, LWarray_format fmt, int NumPackedComponents);
typedef LWresult  LWDAAPI tlwTexRefSetAddressMode(LWtexref hTexRef, int dim, LWaddress_mode am);
typedef LWresult  LWDAAPI tlwTexRefSetFilterMode(LWtexref hTexRef, LWfilter_mode fm);
typedef LWresult  LWDAAPI tlwTexRefSetFlags(LWtexref hTexRef, unsigned int Flags);

typedef LWresult  LWDAAPI tlwTexRefGetAddress(LWdeviceptr *pdptr, LWtexref hTexRef);
typedef LWresult  LWDAAPI tlwTexRefGetArray(LWarray *phArray, LWtexref hTexRef);
typedef LWresult  LWDAAPI tlwTexRefGetAddressMode(LWaddress_mode *pam, LWtexref hTexRef, int dim);
typedef LWresult  LWDAAPI tlwTexRefGetFilterMode(LWfilter_mode *pfm, LWtexref hTexRef);
typedef LWresult  LWDAAPI tlwTexRefGetFormat(LWarray_format *pFormat, int *pNumChannels, LWtexref hTexRef);
typedef LWresult  LWDAAPI tlwTexRefGetFlags(unsigned int *pFlags, LWtexref hTexRef);

/************************************
 **
 **    Surface reference management
 **
 ***********************************/

typedef LWresult  LWDAAPI tlwSurfRefSetArray(LWsurfref hSurfRef, LWarray hArray, unsigned int Flags);
typedef LWresult  LWDAAPI tlwSurfRefGetArray(LWarray *phArray, LWsurfref hSurfRef);

/************************************
 **
 **    Parameter management
 **
 ***********************************/

typedef LWresult  LWDAAPI tlwParamSetSize(LWfunction hfunc, unsigned int numbytes);
typedef LWresult  LWDAAPI tlwParamSeti(LWfunction hfunc, int offset, unsigned int value);
typedef LWresult  LWDAAPI tlwParamSetf(LWfunction hfunc, int offset, float value);
typedef LWresult  LWDAAPI tlwParamSetv(LWfunction hfunc, int offset, void *ptr, unsigned int numbytes);
typedef LWresult  LWDAAPI tlwParamSetTexRef(LWfunction hfunc, int texunit, LWtexref hTexRef);


/************************************
 **
 **    Launch functions
 **
 ***********************************/

typedef LWresult LWDAAPI tlwLaunch(LWfunction f);
typedef LWresult LWDAAPI tlwLaunchGrid(LWfunction f, int grid_width, int grid_height);
typedef LWresult LWDAAPI tlwLaunchGridAsync(LWfunction f, int grid_width, int grid_height, LWstream hStream);

/************************************
 **
 **    Events
 **
 ***********************************/
typedef LWresult LWDAAPI tlwEventCreate(LWevent *phEvent, unsigned int Flags);
typedef LWresult LWDAAPI tlwEventRecord(LWevent hEvent, LWstream hStream);
typedef LWresult LWDAAPI tlwEventQuery(LWevent hEvent);
typedef LWresult LWDAAPI tlwEventSynchronize(LWevent hEvent);
typedef LWresult LWDAAPI tlwEventDestroy(LWevent hEvent);
typedef LWresult LWDAAPI tlwEventElapsedTime(float *pMilliseconds, LWevent hStart, LWevent hEnd);

/************************************
 **
 **    Streams
 **
 ***********************************/
typedef LWresult LWDAAPI tlwStreamCreate(LWstream *phStream, unsigned int Flags);
typedef LWresult LWDAAPI tlwStreamWaitEvent(LWstream hStream, LWevent hEvent, unsigned int Flags);
typedef LWresult LWDAAPI tlwStreamAddCallback(LWstream hStream, LWstreamCallback callback, void *userData, unsigned int flags);

typedef LWresult LWDAAPI tlwStreamQuery(LWstream hStream);
typedef LWresult LWDAAPI tlwStreamSynchronize(LWstream hStream);
typedef LWresult LWDAAPI tlwStreamDestroy(LWstream hStream);

/************************************
 **
 **    Graphics interop
 **
 ***********************************/
typedef LWresult LWDAAPI tlwGraphicsUnregisterResource(LWgraphicsResource resource);
typedef LWresult LWDAAPI tlwGraphicsSubResourceGetMappedArray(LWarray *pArray, LWgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel);

#if __LWDA_API_VERSION >= 3020
typedef LWresult LWDAAPI tlwGraphicsResourceGetMappedPointer(LWdeviceptr *pDevPtr, size_t *pSize, LWgraphicsResource resource);
#else
typedef LWresult LWDAAPI tlwGraphicsResourceGetMappedPointer(LWdeviceptr *pDevPtr, unsigned int *pSize, LWgraphicsResource resource);
#endif

typedef LWresult LWDAAPI tlwGraphicsResourceSetMapFlags(LWgraphicsResource resource, unsigned int flags);
typedef LWresult LWDAAPI tlwGraphicsMapResources(unsigned int count, LWgraphicsResource *resources, LWstream hStream);
typedef LWresult LWDAAPI tlwGraphicsUnmapResources(unsigned int count, LWgraphicsResource *resources, LWstream hStream);

/************************************
 **
 **    Export tables
 **
 ***********************************/
typedef LWresult LWDAAPI tlwGetExportTable(const void **ppExportTable, const LWuuid *pExportTableId);

/************************************
 **
 **    Limits
 **
 ***********************************/

typedef LWresult LWDAAPI tlwCtxSetLimit(LWlimit limit, size_t value);
typedef LWresult LWDAAPI tlwCtxGetLimit(size_t *pvalue, LWlimit limit);
typedef LWresult LWDAAPI tlwCtxGetCacheConfig(LWfunc_cache *pconfig);
typedef LWresult LWDAAPI tlwCtxSetCacheConfig(LWfunc_cache config);
typedef LWresult LWDAAPI tlwCtxGetSharedMemConfig(LWsharedconfig *pConfig);
typedef LWresult LWDAAPI tlwCtxSetSharedMemConfig(LWsharedconfig config);
typedef LWresult LWDAAPI tlwCtxGetApiVersion(LWcontext ctx, unsigned int *version);

/************************************
 **
 **    Profiler
 **
 ***********************************/
typedef LWresult LWDAAPI tlwProfilerStop(void);

/************************************
 ************************************/

extern LWresult LWDAAPI lwInit(unsigned int, int lwdaVersion);

extern tlwDriverGetVersion             *lwDriverGetVersion;
extern tlwDeviceGet                    *lwDeviceGet;
extern tlwDeviceGetCount               *lwDeviceGetCount;
extern tlwDeviceGetName                *lwDeviceGetName;
extern tlwDeviceComputeCapability      *lwDeviceComputeCapability;
extern tlwDeviceGetProperties          *lwDeviceGetProperties;
extern tlwDeviceGetAttribute           *lwDeviceGetAttribute;
extern tlwGetErrorString               *lwGetErrorString;
extern tlwCtxDestroy                   *lwCtxDestroy;
extern tlwCtxAttach                    *lwCtxAttach;
extern tlwCtxDetach                    *lwCtxDetach;
extern tlwCtxPushLwrrent               *lwCtxPushLwrrent;
extern tlwCtxPopLwrrent                *lwCtxPopLwrrent;

extern tlwCtxSetLwrrent                *lwCtxSetLwrrent;
extern tlwCtxGetLwrrent                *lwCtxGetLwrrent;

extern tlwCtxGetDevice                 *lwCtxGetDevice;
extern tlwCtxSynchronize               *lwCtxSynchronize;
extern tlwModuleLoad                   *lwModuleLoad;
extern tlwModuleLoadData               *lwModuleLoadData;
extern tlwModuleLoadDataEx             *lwModuleLoadDataEx;
extern tlwModuleLoadFatBinary          *lwModuleLoadFatBinary;
extern tlwModuleUnload                 *lwModuleUnload;
extern tlwModuleGetFunction            *lwModuleGetFunction;
extern tlwModuleGetTexRef              *lwModuleGetTexRef;
extern tlwModuleGetSurfRef             *lwModuleGetSurfRef;
extern tlwMemFreeHost                  *lwMemFreeHost;
extern tlwMemHostAlloc                 *lwMemHostAlloc;
extern tlwMemHostGetFlags              *lwMemHostGetFlags;

extern tlwMemHostRegister              *lwMemHostRegister;
extern tlwMemHostUnregister            *lwMemHostUnregister;
extern tlwMemcpy                       *lwMemcpy;
extern tlwMemcpyPeer                   *lwMemcpyPeer;

extern tlwDeviceTotalMem               *lwDeviceTotalMem;
extern tlwCtxCreate                    *lwCtxCreate;
extern tlwModuleGetGlobal              *lwModuleGetGlobal;
extern tlwMemGetInfo                   *lwMemGetInfo;
extern tlwMemAlloc                     *lwMemAlloc;
extern tlwMemAllocPitch                *lwMemAllocPitch;
extern tlwMemFree                      *lwMemFree;
extern tlwMemGetAddressRange           *lwMemGetAddressRange;
extern tlwMemAllocHost                 *lwMemAllocHost;
extern tlwMemHostGetDevicePointer      *lwMemHostGetDevicePointer;
extern tlwFuncSetBlockShape            *lwFuncSetBlockShape;
extern tlwFuncSetSharedSize            *lwFuncSetSharedSize;
extern tlwFuncGetAttribute             *lwFuncGetAttribute;
extern tlwFuncSetCacheConfig           *lwFuncSetCacheConfig;
extern tlwFuncSetSharedMemConfig       *lwFuncSetSharedMemConfig;
extern tlwLaunchKernel                 *lwLaunchKernel;
extern tlwArrayDestroy                 *lwArrayDestroy;
extern tlwTexRefCreate                 *lwTexRefCreate;
extern tlwTexRefDestroy                *lwTexRefDestroy;
extern tlwTexRefSetArray               *lwTexRefSetArray;
extern tlwTexRefSetFormat              *lwTexRefSetFormat;
extern tlwTexRefSetAddressMode         *lwTexRefSetAddressMode;
extern tlwTexRefSetFilterMode          *lwTexRefSetFilterMode;
extern tlwTexRefSetFlags               *lwTexRefSetFlags;
extern tlwTexRefGetArray               *lwTexRefGetArray;
extern tlwTexRefGetAddressMode         *lwTexRefGetAddressMode;
extern tlwTexRefGetFilterMode          *lwTexRefGetFilterMode;
extern tlwTexRefGetFormat              *lwTexRefGetFormat;
extern tlwTexRefGetFlags               *lwTexRefGetFlags;
extern tlwSurfRefSetArray              *lwSurfRefSetArray;
extern tlwSurfRefGetArray              *lwSurfRefGetArray;
extern tlwParamSetSize                 *lwParamSetSize;
extern tlwParamSeti                    *lwParamSeti;
extern tlwParamSetf                    *lwParamSetf;
extern tlwParamSetv                    *lwParamSetv;
extern tlwParamSetTexRef               *lwParamSetTexRef;
extern tlwLaunch                       *lwLaunch;
extern tlwLaunchGrid                   *lwLaunchGrid;
extern tlwLaunchGridAsync              *lwLaunchGridAsync;
extern tlwEventCreate                  *lwEventCreate;
extern tlwEventRecord                  *lwEventRecord;
extern tlwEventQuery                   *lwEventQuery;
extern tlwEventSynchronize             *lwEventSynchronize;
extern tlwEventDestroy                 *lwEventDestroy;
extern tlwEventElapsedTime             *lwEventElapsedTime;
extern tlwStreamCreate                 *lwStreamCreate;
extern tlwStreamQuery                  *lwStreamQuery;
extern tlwStreamWaitEvent              *lwStreamWaitEvent;
extern tlwStreamAddCallback            *lwStreamAddCallback;
extern tlwStreamSynchronize            *lwStreamSynchronize;
extern tlwStreamDestroy                *lwStreamDestroy;
extern tlwGraphicsUnregisterResource         *lwGraphicsUnregisterResource;
extern tlwGraphicsSubResourceGetMappedArray  *lwGraphicsSubResourceGetMappedArray;
extern tlwGraphicsResourceSetMapFlags        *lwGraphicsResourceSetMapFlags;
extern tlwGraphicsMapResources               *lwGraphicsMapResources;
extern tlwGraphicsUnmapResources             *lwGraphicsUnmapResources;
extern tlwGetExportTable                     *lwGetExportTable;
extern tlwCtxSetLimit                        *lwCtxSetLimit;
extern tlwCtxGetLimit                        *lwCtxGetLimit;

// These functions could be using the LWCA 3.2 interface (_v2)
extern tlwMemcpyHtoD                   *lwMemcpyHtoD;
extern tlwMemcpyDtoH                   *lwMemcpyDtoH;
extern tlwMemcpyDtoD                   *lwMemcpyDtoD;
extern tlwMemcpyDtoA                   *lwMemcpyDtoA;
extern tlwMemcpyAtoD                   *lwMemcpyAtoD;
extern tlwMemcpyHtoA                   *lwMemcpyHtoA;
extern tlwMemcpyAtoH                   *lwMemcpyAtoH;
extern tlwMemcpyAtoA                   *lwMemcpyAtoA;
extern tlwMemcpy2D                     *lwMemcpy2D;
extern tlwMemcpy2DUnaligned            *lwMemcpy2DUnaligned;
extern tlwMemcpy3D                     *lwMemcpy3D;
extern tlwMemcpyHtoDAsync              *lwMemcpyHtoDAsync;
extern tlwMemcpyDtoHAsync              *lwMemcpyDtoHAsync;
extern tlwMemcpyDtoDAsync              *lwMemcpyDtoDAsync;
extern tlwMemcpyHtoAAsync              *lwMemcpyHtoAAsync;
extern tlwMemcpyAtoHAsync              *lwMemcpyAtoHAsync;
extern tlwMemcpy2DAsync                *lwMemcpy2DAsync;
extern tlwMemcpy3DAsync                *lwMemcpy3DAsync;
extern tlwMemsetD8                     *lwMemsetD8;
extern tlwMemsetD16                    *lwMemsetD16;
extern tlwMemsetD32                    *lwMemsetD32;
extern tlwMemsetD2D8                   *lwMemsetD2D8;
extern tlwMemsetD2D16                  *lwMemsetD2D16;
extern tlwMemsetD2D32                  *lwMemsetD2D32;
extern tlwArrayCreate                  *lwArrayCreate;
extern tlwArrayGetDescriptor           *lwArrayGetDescriptor;
extern tlwArray3DCreate                *lwArray3DCreate;
extern tlwArray3DGetDescriptor         *lwArray3DGetDescriptor;
extern tlwTexRefSetAddress             *lwTexRefSetAddress;
extern tlwTexRefSetAddress2D           *lwTexRefSetAddress2D;
extern tlwTexRefGetAddress             *lwTexRefGetAddress;
extern tlwGraphicsResourceGetMappedPointer   *lwGraphicsResourceGetMappedPointer;

extern tlwMipmappedArrayCreate         *lwMipmappedArrayCreate;
extern tlwMipmappedArrayGetLevel       *lwMipmappedArrayGetLevel;
extern tlwMipmappedArrayDestroy        *lwMipmappedArrayDestroy;

extern tlwProfilerStop                    *lwProfilerStop;

#ifdef __cplusplus
}
#endif

//#undef __LWDA_API_VERSION

#endif //__lwda_drvapi_dynlink_lwda_h__
