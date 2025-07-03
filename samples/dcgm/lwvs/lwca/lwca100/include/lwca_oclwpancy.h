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

/**
 * LWCA Oclwpancy Calculator
 *
 * NAME
 *
 *   lwdaOccMaxActiveBlocksPerMultiprocessor,
 *   lwdaOccMaxPotentialOclwpancyBlockSize,
 *   lwdaOccMaxPotentialOclwpancyBlockSizeVariableSMem
 *
 * DESCRIPTION
 *
 *   The LWCA oclwpancy calculator provides a standalone, programmatical
 *   interface to compute the oclwpancy of a function on a device. It can also
 *   provide oclwpancy-oriented launch configuration suggestions.
 *
 *   The function and device are defined by the user through
 *   lwdaOccFuncAttributes, lwdaOccDeviceProp, and lwdaOccDeviceState
 *   structures. All APIs require all 3 of them.
 *
 *   See the structure definition for more details about the device / function
 *   descriptors.
 *
 *   See each API's prototype for API usage.
 *
 * COMPATIBILITY
 *
 *   The oclwpancy calculator will be updated on each major LWCA toolkit
 *   release. It does not provide forward compatibility, i.e. new hardwares
 *   released after this implementation's release will not be supported.
 *
 * NOTE
 *
 *   If there is access to LWCA runtime, and the sole intent is to callwlate
 *   oclwpancy related values on one of the accessible LWCA devices, using LWCA
 *   runtime's oclwpancy callwlation APIs is recommended.
 *
 */

#ifndef __lwda_oclwpancy_h__
#define __lwda_oclwpancy_h__

#include <stddef.h>
#include <limits.h>

// __OCC_INLINE will be undefined at the end of this header
//
#ifdef __LWDACC__
#define __OCC_INLINE inline __host__ __device__
#elif defined _MSC_VER
#define __OCC_INLINE __inline
#else // GNUCC assumed
#define __OCC_INLINE inline
#endif

enum lwdaOccError_enum {
    LWDA_OCC_SUCCESS              = 0,  // no error encountered
    LWDA_OCC_ERROR_ILWALID_INPUT  = 1,  // input parameter is invalid
    LWDA_OCC_ERROR_UNKNOWN_DEVICE = 2,  // requested device is not supported in
                                        // current implementation or device is
                                        // invalid
};
typedef enum lwdaOccError_enum       lwdaOccError;

typedef struct lwdaOccResult         lwdaOccResult;
typedef struct lwdaOccDeviceProp     lwdaOccDeviceProp;
typedef struct lwdaOccFuncAttributes lwdaOccFuncAttributes;
typedef struct lwdaOccDeviceState    lwdaOccDeviceState;

/**
 * The LWCA oclwpancy calculator computes the oclwpancy of the function
 * described by attributes with the given block size (blockSize), static device
 * properties (properties), dynamic device states (states) and per-block dynamic
 * shared memory allocation (dynamicSMemSize) in bytes, and output it through
 * result along with other useful information. The oclwpancy is computed in
 * terms of the maximum number of active blocks per multiprocessor. The user can
 * then colwert it to other metrics, such as number of active warps.
 *
 * RETURN VALUE
 *
 * The oclwpancy and related information is returned through result.
 *
 * If result->activeBlocksPerMultiprocessor is 0, then the given parameter
 * combination cannot run on the device.
 *
 * ERRORS
 *
 *     LWDA_OCC_ERROR_ILWALID_INPUT   input parameter is invalid.
 *     LWDA_OCC_ERROR_UNKNOWN_DEVICE  requested device is not supported in
 *     current implementation or device is invalid
 */
static __OCC_INLINE
lwdaOccError lwdaOccMaxActiveBlocksPerMultiprocessor(
    lwdaOccResult               *result,           // out
    const lwdaOccDeviceProp     *properties,       // in
    const lwdaOccFuncAttributes *attributes,       // in
    const lwdaOccDeviceState    *state,            // in
    int                          blockSize,        // in
    size_t                       dynamicSmemSize); // in

/**
 * The LWCA launch configurator C API suggests a grid / block size pair (in
 * minGridSize and blockSize) that achieves the best potential oclwpancy
 * (i.e. maximum number of active warps with the smallest number of blocks) for
 * the given function described by attributes, on a device described by
 * properties with settings in state.
 *
 * If per-block dynamic shared memory allocation is not needed, the user should
 * leave both blockSizeToDynamicSMemSize and dynamicSMemSize as 0.
 *
 * If per-block dynamic shared memory allocation is needed, then if the dynamic
 * shared memory size is constant regardless of block size, the size should be
 * passed through dynamicSMemSize, and blockSizeToDynamicSMemSize should be
 * NULL.
 *
 * Otherwise, if the per-block dynamic shared memory size varies with different
 * block sizes, the user needs to provide a pointer to an unary function through
 * blockSizeToDynamicSMemSize that computes the dynamic shared memory needed by
 * a block of the function for any given block size. dynamicSMemSize is
 * ignored. An example signature is:
 *
 *    // Take block size, returns dynamic shared memory needed
 *    size_t blockToSmem(int blockSize);
 *
 * RETURN VALUE
 *
 * The suggested block size and the minimum number of blocks needed to achieve
 * the maximum oclwpancy are returned through blockSize and minGridSize.
 *
 * If *blockSize is 0, then the given combination cannot run on the device.
 *
 * ERRORS
 *
 *     LWDA_OCC_ERROR_ILWALID_INPUT   input parameter is invalid.
 *     LWDA_OCC_ERROR_UNKNOWN_DEVICE  requested device is not supported in
 *     current implementation or device is invalid
 *
 */
static __OCC_INLINE
lwdaOccError lwdaOccMaxPotentialOclwpancyBlockSize(
    int                         *minGridSize,      // out
    int                         *blockSize,        // out
    const lwdaOccDeviceProp     *properties,       // in
    const lwdaOccFuncAttributes *attributes,       // in
    const lwdaOccDeviceState    *state,            // in
    size_t                     (*blockSizeToDynamicSMemSize)(int), // in
    size_t                       dynamicSMemSize); // in

/**
 * The LWCA launch configurator C++ API suggests a grid / block size pair (in
 * minGridSize and blockSize) that achieves the best potential oclwpancy
 * (i.e. the maximum number of active warps with the smallest number of blocks)
 * for the given function described by attributes, on a device described by
 * properties with settings in state.
 *
 * If per-block dynamic shared memory allocation is 0 or constant regardless of
 * block size, the user can use lwdaOccMaxPotentialOclwpancyBlockSize to
 * configure the launch. A constant dynamic shared memory allocation size in
 * bytes can be passed through dynamicSMemSize.
 *
 * Otherwise, if the per-block dynamic shared memory size varies with different
 * block sizes, the user needs to use
 * lwdaOccMaxPotentialOclwpancyBlockSizeVariableSmem instead, and provide a
 * functor / pointer to an unary function (blockSizeToDynamicSMemSize) that
 * computes the dynamic shared memory needed by func for any given block
 * size. An example signature is:
 *
 *  // Take block size, returns per-block dynamic shared memory needed
 *  size_t blockToSmem(int blockSize);
 *
 * RETURN VALUE
 *
 * The suggested block size and the minimum number of blocks needed to achieve
 * the maximum oclwpancy are returned through blockSize and minGridSize.
 *
 * If *blockSize is 0, then the given combination cannot run on the device.
 *
 * ERRORS
 *
 *     LWDA_OCC_ERROR_ILWALID_INPUT   input parameter is invalid.
 *     LWDA_OCC_ERROR_UNKNOWN_DEVICE  requested device is not supported in
 *     current implementation or device is invalid
 *
 */
#if defined(__cplusplus)
namespace {

__OCC_INLINE
lwdaOccError lwdaOccMaxPotentialOclwpancyBlockSize(
    int                         *minGridSize,          // out
    int                         *blockSize,            // out
    const lwdaOccDeviceProp     *properties,           // in
    const lwdaOccFuncAttributes *attributes,           // in
    const lwdaOccDeviceState    *state,                // in
    size_t                       dynamicSMemSize = 0); // in

template <typename UnaryFunction>
__OCC_INLINE
lwdaOccError lwdaOccMaxPotentialOclwpancyBlockSizeVariableSMem(
    int                         *minGridSize,          // out
    int                         *blockSize,            // out
    const lwdaOccDeviceProp     *properties,           // in
    const lwdaOccFuncAttributes *attributes,           // in
    const lwdaOccDeviceState    *state,                // in
    UnaryFunction                blockSizeToDynamicSMemSize); // in

} // namespace anonymous
#endif // defined(__cplusplus)

/**
 * Data structures
 *
 * These structures are subject to change for future architecture and LWCA
 * releases. C users should initialize the structure as {0}.
 *
 */

/**
 * Device descriptor
 *
 * This structure describes a device.
 */
struct lwdaOccDeviceProp {
    int    computeMajor;                // Compute capability major version
    int    computeMinor;                // Compute capability minor
                                        // version. None supported minor version
                                        // may cause error
    int    maxThreadsPerBlock;          // Maximum number of threads per block
    int    maxThreadsPerMultiprocessor; // Maximum number of threads per SM
                                        // i.e. (Max. number of warps) x (warp
                                        // size)
    int    regsPerBlock;                // Maximum number of registers per block
    int    regsPerMultiprocessor;       // Maximum number of registers per SM
    int    warpSize;                    // Warp size
    size_t sharedMemPerBlock;           // Maximum shared memory size per block
    size_t sharedMemPerMultiprocessor;  // Maximum shared memory size per SM
    int    numSms;                      // Number of SMs available
    size_t sharedMemPerBlockOptin;      // Maximum optin shared memory size per block

#ifdef __cplusplus
    // This structure can be colwerted from a lwdaDeviceProp structure for users
    // that use this header in their LWCA applications.
    //
    // If the application have access to the LWCA Runtime API, the application
    // can obtain the device properties of a LWCA device through
    // lwdaGetDeviceProperties, and initialize a lwdaOccDeviceProp with the
    // lwdaDeviceProp structure.
    //
    // Example:
    /*
     {
         lwdaDeviceProp prop;

         lwdaGetDeviceProperties(&prop, ...);

         lwdaOccDeviceProp occProp = prop;

         ...

         lwdaOccMaxPotentialOclwpancyBlockSize(..., &occProp, ...);
     }
     */
    //
    template<typename DeviceProp>
    __OCC_INLINE
    lwdaOccDeviceProp(const DeviceProp &props)
    :   computeMajor                (props.major),
        computeMinor                (props.minor),
        maxThreadsPerBlock          (props.maxThreadsPerBlock),
        maxThreadsPerMultiprocessor (props.maxThreadsPerMultiProcessor),
        regsPerBlock                (props.regsPerBlock),
        regsPerMultiprocessor       (props.regsPerMultiprocessor),
        warpSize                    (props.warpSize),
        sharedMemPerBlock           (props.sharedMemPerBlock),
        sharedMemPerMultiprocessor  (props.sharedMemPerMultiprocessor),
        numSms                      (props.multiProcessorCount),
        sharedMemPerBlockOptin      (props.sharedMemPerBlockOptin)
    {}

    __OCC_INLINE
    lwdaOccDeviceProp()
    :   computeMajor                (0),
        computeMinor                (0),
        maxThreadsPerBlock          (0),
        maxThreadsPerMultiprocessor (0),
        regsPerBlock                (0),
        regsPerMultiprocessor       (0),
        warpSize                    (0),
        sharedMemPerBlock           (0),
        sharedMemPerMultiprocessor  (0),
        numSms                      (0),
        sharedMemPerBlockOptin      (0)
    {}
#endif // __cplusplus
};

/**
 * Partitioned global caching option
 */
typedef enum lwdaOccPartitionedGCConfig_enum {
    PARTITIONED_GC_OFF,        // Disable partitioned global caching
    PARTITIONED_GC_ON,         // Prefer partitioned global caching
    PARTITIONED_GC_ON_STRICT   // Force partitioned global caching
} lwdaOccPartitionedGCConfig;

/**
 * Per function opt in maximum dynamic shared memory limit
 */
typedef enum lwdaOccFuncShmemConfig_enum {
    FUNC_SHMEM_LIMIT_DEFAULT,   // Default shmem limit
    FUNC_SHMEM_LIMIT_OPTIN,     // Use the optin shmem limit
} lwdaOccFuncShmemConfig;

/**
 * Function descriptor
 *
 * This structure describes a LWCA function.
 */
struct lwdaOccFuncAttributes {
    int maxThreadsPerBlock; // Maximum block size the function can work with. If
                            // unlimited, use INT_MAX or any value greater than
                            // or equal to maxThreadsPerBlock of the device
    int numRegs;            // Number of registers used. When the function is
                            // launched on device, the register count may change
                            // due to internal tools requirements.
    size_t sharedSizeBytes; // Number of static shared memory used

    lwdaOccPartitionedGCConfig partitionedGCConfig; 
                            // Partitioned global caching is required to enable
                            // caching on certain chips, such as sm_52
                            // devices. Partitioned global caching can be
                            // automatically disabled if the oclwpancy
                            // requirement of the launch cannot support caching.
                            //
                            // To override this behavior with caching on and
                            // callwlate oclwpancy strictly according to the
                            // preference, set partitionedGCConfig to
                            // PARTITIONED_GC_ON_STRICT. This is especially
                            // useful for experimenting and finding launch
                            // configurations (MaxPotentialOclwpancyBlockSize)
                            // that allow global caching to take effect.
                            //
                            // This flag only affects the oclwpancy callwlation.

    lwdaOccFuncShmemConfig shmemLimitConfig;
                            // Certain chips like sm_70 allow a user to opt into
                            // a higher per block limit of dynamic shared memory
                            // This optin is performed on a per function basis
                            // using the lwFuncSetAttribute function

    size_t maxDynamicSharedSizeBytes;
                            // User set limit on maximum dynamic shared memory
                            // usable by the kernel
                            // This limit is set using the lwFuncSetAttribute
                            // function.
#ifdef __cplusplus
    // This structure can be colwerted from a lwdaFuncAttributes structure for
    // users that use this header in their LWCA applications.
    //
    // If the application have access to the LWCA Runtime API, the application
    // can obtain the function attributes of a LWCA kernel function through
    // lwdaFuncGetAttributes, and initialize a lwdaOccFuncAttributes with the
    // lwdaFuncAttributes structure.
    //
    // Example:
    /*
      __global__ void foo() {...}

      ...

      {
          lwdaFuncAttributes attr;

          lwdaFuncGetAttributes(&attr, foo);

          lwdaOccFuncAttributes occAttr = attr;

          ...

          lwdaOccMaxPotentialOclwpancyBlockSize(..., &occAttr, ...);
      }
     */
    //
    template<typename FuncAttributes>
    __OCC_INLINE
    lwdaOccFuncAttributes(const FuncAttributes &attr)
    :   maxThreadsPerBlock  (attr.maxThreadsPerBlock),
        numRegs             (attr.numRegs),
        sharedSizeBytes     (attr.sharedSizeBytes),
        partitionedGCConfig (PARTITIONED_GC_OFF),
        shmemLimitConfig    (FUNC_SHMEM_LIMIT_OPTIN),
        maxDynamicSharedSizeBytes (attr.maxDynamicSharedSizeBytes)
    {}

    __OCC_INLINE
    lwdaOccFuncAttributes()
    :   maxThreadsPerBlock  (0),
        numRegs             (0),
        sharedSizeBytes     (0),
        partitionedGCConfig (PARTITIONED_GC_OFF),
        shmemLimitConfig    (FUNC_SHMEM_LIMIT_DEFAULT),
        maxDynamicSharedSizeBytes (0)
    {}
#endif
};

typedef enum lwdaOccCacheConfig_enum {
    CACHE_PREFER_NONE   = 0x00, // no preference for shared memory or L1 (default)
    CACHE_PREFER_SHARED = 0x01, // prefer larger shared memory and smaller L1 cache
    CACHE_PREFER_L1     = 0x02, // prefer larger L1 cache and smaller shared memory
    CACHE_PREFER_EQUAL  = 0x03  // prefer equal sized L1 cache and shared memory
} lwdaOccCacheConfig;

typedef enum lwdaOccCarveoutConfig_enum {
    SHAREDMEM_CARVEOUT_DEFAULT       = -1,  // no preference for shared memory or L1 (default)
    SHAREDMEM_CARVEOUT_MAX_SHARED    = 100, // prefer maximum available shared memory, minimum L1 cache
    SHAREDMEM_CARVEOUT_MAX_L1        = 0,    // prefer maximum available L1 cache, minimum shared memory
    SHAREDMEM_CARVEOUT_HALF          = 50   // prefer half of maximum available shared memory, with the rest as L1 cache
} lwdaOccCarveoutConfig;

/**
 * Device state descriptor
 *
 * This structure describes device settings that affect oclwpancy callwlation.
 */
struct lwdaOccDeviceState
{
    // Cache / shared memory split preference. Deprecated on Volta 
    lwdaOccCacheConfig cacheConfig; 
    // Shared memory / L1 split preference. Supported on only Volta
    int carveoutConfig;

#ifdef __cplusplus
    __OCC_INLINE
    lwdaOccDeviceState()
    :   cacheConfig     (CACHE_PREFER_NONE),
        carveoutConfig  (SHAREDMEM_CARVEOUT_DEFAULT)
    {}
#endif
};

typedef enum lwdaOccLimitingFactor_enum {
                                    // Oclwpancy limited due to:
    OCC_LIMIT_WARPS         = 0x01, // - warps available
    OCC_LIMIT_REGISTERS     = 0x02, // - registers available
    OCC_LIMIT_SHARED_MEMORY = 0x04, // - shared memory available
    OCC_LIMIT_BLOCKS        = 0x08  // - blocks available
} lwdaOccLimitingFactor;

/**
 * Oclwpancy output
 *
 * This structure contains oclwpancy calculator's output.
 */
struct lwdaOccResult {
    int activeBlocksPerMultiprocessor; // Oclwpancy
    unsigned int limitingFactors;      // Factors that limited oclwpancy. A bit
                                       // field that counts the limiting
                                       // factors, see lwdaOccLimitingFactor
    int blockLimitRegs;                // Oclwpancy due to register
                                       // usage, INT_MAX if the kernel does not
                                       // use any register.
    int blockLimitSharedMem;           // Oclwpancy due to shared memory
                                       // usage, INT_MAX if the kernel does not
                                       // use shared memory.
    int blockLimitWarps;               // Oclwpancy due to block size limit
    int blockLimitBlocks;              // Oclwpancy due to maximum number of blocks
                                       // managable per SM
    int allocatedRegistersPerBlock;    // Actual number of registers allocated per
                                       // block
    size_t allocatedSharedMemPerBlock; // Actual size of shared memory allocated
                                       // per block
    lwdaOccPartitionedGCConfig partitionedGCConfig;
                                       // Report if partitioned global caching
                                       // is actually enabled.
};

/**
 * Partitioned global caching support
 *
 * See lwdaOccPartitionedGlobalCachingModeSupport
 */
typedef enum lwdaOccPartitionedGCSupport_enum {
    PARTITIONED_GC_NOT_SUPPORTED,  // Partitioned global caching is not supported
    PARTITIONED_GC_SUPPORTED,      // Partitioned global caching is supported
    PARTITIONED_GC_ALWAYS_ON       // This is only needed for Pascal. This, and
                                   // all references / explanations for this,
                                   // should be removed from the header before
                                   // exporting to toolkit.
} lwdaOccPartitionedGCSupport;

/**
 * Implementation
 */

/**
 * Max compute capability supported
 */
#define __LWDA_OCC_MAJOR__ 7
#define __LWDA_OCC_MINOR__ 0

//////////////////////////////////////////
//    Mathematical Helper Functions     //
//////////////////////////////////////////

static __OCC_INLINE int __occMin(int lhs, int rhs)
{
    return rhs < lhs ? rhs : lhs;
}

static __OCC_INLINE int __occDivideRoundUp(int x, int y)
{
    return (x + (y - 1)) / y;
}

static __OCC_INLINE int __occRoundUp(int x, int y)
{
    return y * __occDivideRoundUp(x, y);
}

//////////////////////////////////////////
//      Architectural Properties        //
//////////////////////////////////////////

/**
 * Granularity of shared memory allocation
 */
static __OCC_INLINE lwdaOccError lwdaOccSMemAllocationGranularity(int *limit, const lwdaOccDeviceProp *properties)
{
    int value;

    switch(properties->computeMajor) {
        case 2:
            value = 128;
            break;
        case 3:
        case 5:
        case 6:
        case 7:
            value = 256;
            break;
        default:
            return LWDA_OCC_ERROR_UNKNOWN_DEVICE;
    }

    *limit = value;

    return LWDA_OCC_SUCCESS;
}

/**
 * Granularity of register allocation
 */
static __OCC_INLINE lwdaOccError lwdaOccRegAllocationGranularity(int *limit, const lwdaOccDeviceProp *properties, int regsPerThread)
{
    int value;

    switch(properties->computeMajor) {
        case 2:
            // Fermi+ allocates registers to warps
            //
            switch(regsPerThread) {
                case 21:
                case 22:
                case 29:
                case 30:
                case 37:
                case 38:
                case 45:
                case 46:
                    value = 128;
                    break;
                default:
                    value = 64;
                    break;
            }
            break;
        case 3:
        case 5:
        case 6:
        case 7:
            value = 256;
            break;
        default:
            return LWDA_OCC_ERROR_UNKNOWN_DEVICE;
    }

    *limit = value;

    return LWDA_OCC_SUCCESS;
}

/**
 * Number of sub-partitions
 */
static __OCC_INLINE lwdaOccError lwdaOccSubPartitionsPerMultiprocessor(int *limit, const lwdaOccDeviceProp *properties)
{
    int value;

    switch(properties->computeMajor) {
        case 2:
            value = 2;
            break;
        case 3:
        case 5:
        case 6:
        case 7:
            value = 4;
            break;
        default:
            return LWDA_OCC_ERROR_UNKNOWN_DEVICE;
    }

    *limit = value;

    return LWDA_OCC_SUCCESS;
}


/**
 * Maximum number of blocks that can run simultaneously on a multiprocessor
 */
static __OCC_INLINE lwdaOccError lwdaOccMaxBlocksPerMultiprocessor(int* limit, const lwdaOccDeviceProp *properties)
{
    int value;

    switch(properties->computeMajor) {
        case 2:
            value = 8;
            break;
        case 3:
            value = 16;
            break;
        case 5:
        case 6:
            value = 32;
            break;
        case 7: {
            int isTuring = properties->computeMinor == 5;
            value = (isTuring) ? 16 : 32;
            break;
        }
        default:
            return LWDA_OCC_ERROR_UNKNOWN_DEVICE;
    }

    *limit = value;

    return LWDA_OCC_SUCCESS;
}

/** 
 * Align up shared memory based on compute major configurations
 */
static __OCC_INLINE lwdaOccError lwdaOccAlignUpShmemSizeVoltaPlus(size_t *shMemSize, const lwdaOccDeviceProp *properties)
{
    // Volta and Turing have shared L1 cache / shared memory, and support cache
    // configuration to trade one for the other. These values are needed to
    // map carveout config ratio to the next available architecture size
    size_t size = *shMemSize;

    switch (properties->computeMajor) {
    case 7: {
        // Turing supports 32KB and 64KB shared mem.
        int isTuring = properties->computeMinor == 5;
        if (isTuring) {
            if (size <= 32768) {
                *shMemSize = 32768;
            }
            else {
                *shMemSize = properties->sharedMemPerMultiprocessor;
            }
        }
        // Volta supports 0KB, 8KB, 16KB, 32KB, 64KB, and 96KB shared mem.
        else {
            if (size == 0) {
                *shMemSize = 0;
            }
            else if (size <= 8192) {
                *shMemSize = 8192;
            }
            else if (size <= 16384) {
                *shMemSize = 16384;
            }
            else if (size <= 32768) {
                *shMemSize = 32768;
            }
            else if (size <= 65536) {
                *shMemSize = 65536;
            }
            else {
                *shMemSize = properties->sharedMemPerMultiprocessor;
            }
        }
        break;
    }
    default:
        return LWDA_OCC_ERROR_UNKNOWN_DEVICE;
    }

    return LWDA_OCC_SUCCESS;
}

/**
 * Shared memory based on the new carveoutConfig API introduced with Volta
 */
static __OCC_INLINE lwdaOccError lwdaOccSMemPreferenceVoltaPlus(size_t *limit, const lwdaOccDeviceProp *properties, const lwdaOccDeviceState *state)
{
    lwdaOccError status = LWDA_OCC_SUCCESS;
    size_t preferenceShmemSize;

    // LWCA 9.0 introduces a new API to set shared memory - L1 configuration on supported
    // devices. This preference will take precedence over the older cacheConfig setting.
    // Map cacheConfig to its effective preference value.
    int effectivePreference = state->carveoutConfig;
    if ((effectivePreference < SHAREDMEM_CARVEOUT_DEFAULT) || (effectivePreference > SHAREDMEM_CARVEOUT_MAX_SHARED)) {
        return LWDA_OCC_ERROR_ILWALID_INPUT;
    }
    
    if (effectivePreference == SHAREDMEM_CARVEOUT_DEFAULT) {
        switch (state->cacheConfig)
        {
        case CACHE_PREFER_L1:
            effectivePreference = SHAREDMEM_CARVEOUT_MAX_L1;
            break;
        case CACHE_PREFER_SHARED:
            effectivePreference = SHAREDMEM_CARVEOUT_MAX_SHARED;
            break;
        case CACHE_PREFER_EQUAL:
            effectivePreference = SHAREDMEM_CARVEOUT_HALF;
            break;
        default:
            effectivePreference = SHAREDMEM_CARVEOUT_DEFAULT;
            break;
        }
    }

    if (effectivePreference == SHAREDMEM_CARVEOUT_DEFAULT) {
        preferenceShmemSize = properties->sharedMemPerMultiprocessor;
    }
    else {
        preferenceShmemSize = (size_t) (effectivePreference * properties->sharedMemPerMultiprocessor) / 100;
    }

    status = lwdaOccAlignUpShmemSizeVoltaPlus(&preferenceShmemSize, properties);
    *limit = preferenceShmemSize;
    return status;
}

/**
 * Shared memory based on the cacheConfig
 */
static __OCC_INLINE lwdaOccError lwdaOccSMemPreference(size_t *limit, const lwdaOccDeviceProp *properties, const lwdaOccDeviceState *state)
{
    size_t bytes                          = 0;
    size_t sharedMemPerMultiprocessorHigh = properties->sharedMemPerMultiprocessor;
    lwdaOccCacheConfig cacheConfig        = state->cacheConfig;

    // Fermi and Kepler has shared L1 cache / shared memory, and support cache
    // configuration to trade one for the other. These values are needed to
    // callwlate the correct shared memory size for user requested cache
    // configuration.
    //
    size_t minCacheSize                   = 16384;
    size_t maxCacheSize                   = 49152;
    size_t cacheAndSharedTotal            = sharedMemPerMultiprocessorHigh + minCacheSize;
    size_t sharedMemPerMultiprocessorLow  = cacheAndSharedTotal - maxCacheSize;

    switch (properties->computeMajor) {
        case 2:
            // Fermi supports 48KB / 16KB or 16KB / 48KB partitions for shared /
            // L1.
            //
            switch (cacheConfig) {
                default :
                case CACHE_PREFER_NONE:
                case CACHE_PREFER_SHARED:
                case CACHE_PREFER_EQUAL:
                    bytes = sharedMemPerMultiprocessorHigh;
                    break;
                case CACHE_PREFER_L1:
                    bytes = sharedMemPerMultiprocessorLow;
                    break;
            }
            break;
        case 3:
            // Kepler supports 16KB, 32KB, or 48KB partitions for L1. The rest
            // is shared memory.
            //
            switch (cacheConfig) {
                default :
                case CACHE_PREFER_NONE:
                case CACHE_PREFER_SHARED:
                    bytes = sharedMemPerMultiprocessorHigh;
                    break;
                case CACHE_PREFER_L1:
                    bytes = sharedMemPerMultiprocessorLow;
                    break;
                case CACHE_PREFER_EQUAL:
                    // Equal is the mid-point between high and low. It should be
                    // equivalent to low + 16KB.
                    //
                    bytes = (sharedMemPerMultiprocessorHigh + sharedMemPerMultiprocessorLow) / 2;
                    break;
            }
            break;
        case 5:
        case 6:
            // Maxwell and Pascal have dedicated shared memory.
            //
            bytes = sharedMemPerMultiprocessorHigh;
            break;
        default:
            return LWDA_OCC_ERROR_UNKNOWN_DEVICE;
    }

    *limit = bytes;

    return LWDA_OCC_SUCCESS;
}

/**
 * Shared memory based on config requested by User
 */
static __OCC_INLINE lwdaOccError lwdaOccSMemPerMultiprocessor(size_t *limit, const lwdaOccDeviceProp *properties, const lwdaOccDeviceState *state)
{
    // Volta introduces a new API that allows for shared memory carveout preference. Because it is a shared memory preference,
    // it is handled separately from the cache config preference.
    if (properties->computeMajor == 7) {
        return lwdaOccSMemPreferenceVoltaPlus(limit, properties, state);
    }
    return lwdaOccSMemPreference(limit, properties, state);
}

/**
 * Return the per block shared memory limit based on function config
 */
static __OCC_INLINE lwdaOccError lwdaOccSMemPerBlock(size_t *limit, const lwdaOccDeviceProp *properties, lwdaOccFuncShmemConfig shmemLimitConfig, size_t smemPerCta)
{
    switch (properties->computeMajor) {
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
            *limit = properties->sharedMemPerBlock;
            break;
        case 7:
            switch (shmemLimitConfig) {
                default:
                case FUNC_SHMEM_LIMIT_DEFAULT:
                    *limit = properties->sharedMemPerBlock;
                    break;
                case FUNC_SHMEM_LIMIT_OPTIN:
                    if (smemPerCta > properties->sharedMemPerBlock) {
                        *limit = properties->sharedMemPerBlockOptin;
                    }
                    else {
                        *limit = properties->sharedMemPerBlock;
                    }
                    break;
            }
            break;
        default:
            return LWDA_OCC_ERROR_UNKNOWN_DEVICE;
    }

    return LWDA_OCC_SUCCESS;
}

/**
 * Partitioned global caching mode support
 */
static __OCC_INLINE lwdaOccError lwdaOccPartitionedGlobalCachingModeSupport(lwdaOccPartitionedGCSupport *limit, const lwdaOccDeviceProp *properties)
{
    *limit = PARTITIONED_GC_NOT_SUPPORTED;

    if ((properties->computeMajor == 5 && (properties->computeMinor == 2 || properties->computeMinor == 3)) ||
        properties->computeMajor == 6) {
        *limit = PARTITIONED_GC_SUPPORTED;
    }

    if (properties->computeMajor == 6 && properties->computeMinor == 0) {
        *limit = PARTITIONED_GC_NOT_SUPPORTED;
    }

    return LWDA_OCC_SUCCESS;
}

///////////////////////////////////////////////
//            User Input Sanity              //
///////////////////////////////////////////////

static __OCC_INLINE lwdaOccError lwdaOccDevicePropCheck(const lwdaOccDeviceProp *properties)
{
    // Verify device properties
    //
    // Each of these limits must be a positive number.
    //
    // Compute capacity is checked during the oclwpancy callwlation
    //
    if (properties->maxThreadsPerBlock          <= 0 ||
        properties->maxThreadsPerMultiprocessor <= 0 ||
        properties->regsPerBlock                <= 0 ||
        properties->regsPerMultiprocessor       <= 0 ||
        properties->warpSize                    <= 0 ||
        properties->sharedMemPerBlock           <= 0 ||
        properties->sharedMemPerMultiprocessor  <= 0 ||
        properties->numSms                      <= 0) {
        return LWDA_OCC_ERROR_ILWALID_INPUT;
    }

    return LWDA_OCC_SUCCESS;
}

static __OCC_INLINE lwdaOccError lwdaOccFuncAttributesCheck(const lwdaOccFuncAttributes *attributes)
{
    // Verify function attributes
    //
    if (attributes->maxThreadsPerBlock <= 0 ||
        attributes->numRegs < 0) {            // Compiler may choose not to use
                                              // any register (empty kernels,
                                              // etc.)
        return LWDA_OCC_ERROR_ILWALID_INPUT;
    }

    return LWDA_OCC_SUCCESS;
}

static __OCC_INLINE lwdaOccError lwdaOccDeviceStateCheck(const lwdaOccDeviceState *state)
{
    (void)state;   // silence unused-variable warning
    // Placeholder
    //

    return LWDA_OCC_SUCCESS;
}

static __OCC_INLINE lwdaOccError lwdaOccInputCheck(
    const lwdaOccDeviceProp     *properties,
    const lwdaOccFuncAttributes *attributes,
    const lwdaOccDeviceState    *state)
{
    lwdaOccError status = LWDA_OCC_SUCCESS;

    status = lwdaOccDevicePropCheck(properties);
    if (status != LWDA_OCC_SUCCESS) {
        return status;
    }

    status = lwdaOccFuncAttributesCheck(attributes);
    if (status != LWDA_OCC_SUCCESS) {
        return status;
    }

    status = lwdaOccDeviceStateCheck(state);
    if (status != LWDA_OCC_SUCCESS) {
        return status;
    }

    return status;
}

///////////////////////////////////////////////
//    Oclwpancy callwlation Functions        //
///////////////////////////////////////////////

static __OCC_INLINE int lwdaOccPartitionedGCForced(
    const lwdaOccDeviceProp *properties)
{
    lwdaOccPartitionedGCSupport gcSupport;

    lwdaOccPartitionedGlobalCachingModeSupport(&gcSupport, properties);

    return gcSupport == PARTITIONED_GC_ALWAYS_ON;
}

static __OCC_INLINE lwdaOccPartitionedGCConfig lwdaOccPartitionedGCExpected(
    const lwdaOccDeviceProp     *properties,
    const lwdaOccFuncAttributes *attributes)
{
    lwdaOccPartitionedGCSupport gcSupport;
    lwdaOccPartitionedGCConfig gcConfig;

    lwdaOccPartitionedGlobalCachingModeSupport(&gcSupport, properties);

    gcConfig = attributes->partitionedGCConfig;

    if (gcSupport == PARTITIONED_GC_NOT_SUPPORTED) {
        gcConfig = PARTITIONED_GC_OFF;
    }

    if (lwdaOccPartitionedGCForced(properties)) {
        gcConfig = PARTITIONED_GC_ON;
    }

    return gcConfig;
}

// Warp limit
//
static __OCC_INLINE lwdaOccError lwdaOccMaxBlocksPerSMWarpsLimit(
    int                         *limit,
    lwdaOccPartitionedGCConfig   gcConfig,
    const lwdaOccDeviceProp     *properties,
    const lwdaOccFuncAttributes *attributes,
    int                          blockSize)
{
    lwdaOccError status = LWDA_OCC_SUCCESS;
    int maxWarpsPerSm;
    int warpsAllocatedPerCTA;
    int maxBlocks;
    (void)attributes;   // silence unused-variable warning

    if (blockSize > properties->maxThreadsPerBlock) {
        maxBlocks = 0;
    }
    else {
        maxWarpsPerSm = properties->maxThreadsPerMultiprocessor / properties->warpSize;
        warpsAllocatedPerCTA = __occDivideRoundUp(blockSize, properties->warpSize);
        maxBlocks = 0;

        if (gcConfig != PARTITIONED_GC_OFF) {
            int maxBlocksPerSmPartition;
            int maxWarpsPerSmPartition;

            // If partitioned global caching is on, then a CTA can only use a SM
            // partition (a half SM), and thus a half of the warp slots
            // available per SM
            //
            maxWarpsPerSmPartition  = maxWarpsPerSm / 2;
            maxBlocksPerSmPartition = maxWarpsPerSmPartition / warpsAllocatedPerCTA;
            maxBlocks               = maxBlocksPerSmPartition * 2;
        }
        // On hardware that supports partitioned global caching, each half SM is
        // guaranteed to support at least 32 warps (maximum number of warps of a
        // CTA), so caching will not cause 0 oclwpancy due to insufficient warp
        // allocation slots.
        //
        else {
            maxBlocks = maxWarpsPerSm / warpsAllocatedPerCTA;
        }
    }

    *limit = maxBlocks;

    return status;
}

// Shared memory limit
//
static __OCC_INLINE lwdaOccError lwdaOccMaxBlocksPerSMSmemLimit(
    int                         *limit,
    lwdaOccResult               *result,
    const lwdaOccDeviceProp     *properties,
    const lwdaOccFuncAttributes *attributes,
    const lwdaOccDeviceState    *state,
    int                          blockSize,
    size_t                       dynamicSmemSize)
{
    lwdaOccError status = LWDA_OCC_SUCCESS;
    int allocationGranularity;
    size_t userSmemPreference = 0;
    size_t totalSmemUsagePerCTA;
    size_t maxSmemUsagePerCTA;
    size_t smemAllocatedPerCTA;
    size_t sharedMemPerMultiprocessor;
    size_t smemLimitPerCTA;
    int maxBlocks;
    int dynamicSmemSizeExceeded = 0;
    int totalSmemSizeExceeded = 0;
    (void)blockSize;   // silence unused-variable warning

    status = lwdaOccSMemAllocationGranularity(&allocationGranularity, properties);
    if (status != LWDA_OCC_SUCCESS) {
        return status;
    }

    // Obtain the user preferred shared memory size. This setting is ignored if
    // user requests more shared memory than preferred.
    //
    status = lwdaOccSMemPerMultiprocessor(&userSmemPreference, properties, state);
    if (status != LWDA_OCC_SUCCESS) {
        return status;
    }

    totalSmemUsagePerCTA = attributes->sharedSizeBytes + dynamicSmemSize;
    smemAllocatedPerCTA = __occRoundUp((int)totalSmemUsagePerCTA, (int)allocationGranularity);

    maxSmemUsagePerCTA = attributes->sharedSizeBytes + attributes->maxDynamicSharedSizeBytes;

    dynamicSmemSizeExceeded = 0;
    totalSmemSizeExceeded   = 0;

    // Obtain the user set maximum dynamic size if it exists
    // If so, the current launch dynamic shared memory must not
    // exceed the set limit
    if (attributes->shmemLimitConfig != FUNC_SHMEM_LIMIT_DEFAULT &&
        dynamicSmemSize > attributes->maxDynamicSharedSizeBytes) {
        dynamicSmemSizeExceeded = 1;
    }

    status = lwdaOccSMemPerBlock(&smemLimitPerCTA, properties, attributes->shmemLimitConfig, maxSmemUsagePerCTA);
    if (status != LWDA_OCC_SUCCESS) {
        return status;
    }

    if (smemAllocatedPerCTA > smemLimitPerCTA) {
        totalSmemSizeExceeded = 1;
    }

    if (dynamicSmemSizeExceeded || totalSmemSizeExceeded) {
        maxBlocks = 0;
    }
    else {
        // User requested shared memory limit is used as long as it is greater
        // than the total shared memory used per CTA, i.e. as long as at least
        // one CTA can be launched.
        if (userSmemPreference >= smemAllocatedPerCTA) {
            sharedMemPerMultiprocessor = userSmemPreference;
        }
        else {
            // On Volta+, user requested shared memory will limit oclwpancy
            // if it's less than shared memory per CTA. Otherwise, the
            // maximum shared memory limit is used.
            if (properties->computeMajor == 7) {
                sharedMemPerMultiprocessor = smemAllocatedPerCTA;
                lwdaOccAlignUpShmemSizeVoltaPlus(&sharedMemPerMultiprocessor, properties);
            }
            else {
                sharedMemPerMultiprocessor = properties->sharedMemPerMultiprocessor;
            }
        }

        if (smemAllocatedPerCTA > 0) {
            maxBlocks = (int)(sharedMemPerMultiprocessor / smemAllocatedPerCTA);
        }
        else {
            maxBlocks = INT_MAX;
        }
    }

    result->allocatedSharedMemPerBlock = smemAllocatedPerCTA;

    *limit = maxBlocks;

    return status;
}

static __OCC_INLINE
lwdaOccError lwdaOccMaxBlocksPerSMRegsLimit(
    int                         *limit,
    lwdaOccPartitionedGCConfig  *gcConfig,
    lwdaOccResult               *result,
    const lwdaOccDeviceProp     *properties,
    const lwdaOccFuncAttributes *attributes,
    int                          blockSize)
{
    lwdaOccError status = LWDA_OCC_SUCCESS;
    int allocationGranularity;
    int warpsAllocatedPerCTA;
    int regsAllocatedPerCTA;
    int regsAssumedPerCTA;
    int regsPerWarp;
    int regsAllocatedPerWarp;
    int numSubPartitions;
    int numRegsPerSubPartition;
    int numWarpsPerSubPartition;
    int numWarpsPerSM;
    int maxBlocks;

    status = lwdaOccRegAllocationGranularity(
        &allocationGranularity,
        properties,
        attributes->numRegs);   // Fermi requires special handling of certain register usage
    if (status != LWDA_OCC_SUCCESS) {
        return status;
    }

    status = lwdaOccSubPartitionsPerMultiprocessor(&numSubPartitions, properties);
    if (status != LWDA_OCC_SUCCESS) {
        return status;
    }

    warpsAllocatedPerCTA = __occDivideRoundUp(blockSize, properties->warpSize);

    // GPUs of compute capability 2.x and higher allocate registers to warps
    //
    // Number of regs per warp is regs per thread x warp size, rounded up to
    // register allocation granularity
    //
    regsPerWarp          = attributes->numRegs * properties->warpSize;
    regsAllocatedPerWarp = __occRoundUp(regsPerWarp, allocationGranularity);
    regsAllocatedPerCTA  = regsAllocatedPerWarp * warpsAllocatedPerCTA;

    // Hardware verifies if a launch fits the per-CTA register limit. For
    // historical reasons, the verification logic assumes register
    // allocations are made to all partitions simultaneously. Therefore, to
    // simulate the hardware check, the warp allocation needs to be rounded
    // up to the number of partitions.
    //
    regsAssumedPerCTA = regsAllocatedPerWarp * __occRoundUp(warpsAllocatedPerCTA, numSubPartitions);

    if (properties->regsPerBlock < regsAssumedPerCTA ||   // Hardware check
        properties->regsPerBlock < regsAllocatedPerCTA) { // Software check
        maxBlocks = 0;
    }
    else {
        if (regsAllocatedPerWarp > 0) {
            // Registers are allocated in each sub-partition. The max number
            // of warps that can fit on an SM is equal to the max number of
            // warps per sub-partition x number of sub-partitions.
            //
            numRegsPerSubPartition  = properties->regsPerMultiprocessor / numSubPartitions;
            numWarpsPerSubPartition = numRegsPerSubPartition / regsAllocatedPerWarp;

            maxBlocks = 0;

            if (*gcConfig != PARTITIONED_GC_OFF) {
                int numSubPartitionsPerSmPartition;
                int numWarpsPerSmPartition;
                int maxBlocksPerSmPartition;

                // If partitioned global caching is on, then a CTA can only
                // use a half SM, and thus a half of the registers available
                // per SM
                //
                numSubPartitionsPerSmPartition = numSubPartitions / 2;
                numWarpsPerSmPartition         = numWarpsPerSubPartition * numSubPartitionsPerSmPartition;
                maxBlocksPerSmPartition        = numWarpsPerSmPartition / warpsAllocatedPerCTA;
                maxBlocks                      = maxBlocksPerSmPartition * 2;
            }

            // Try again if partitioned global caching is not enabled, or if
            // the CTA cannot fit on the SM with caching on. In the latter
            // case, the device will automatically turn off caching, except
            // if the device forces it. The user can also override this
            // assumption with PARTITIONED_GC_ON_STRICT to callwlate
            // oclwpancy and launch configuration.
            //
            {
                int gcOff = (*gcConfig == PARTITIONED_GC_OFF);
                int zeroOclwpancy = (maxBlocks == 0);
                int cachingForced = (*gcConfig == PARTITIONED_GC_ON_STRICT ||
                                     lwdaOccPartitionedGCForced(properties));

                if (gcOff || (zeroOclwpancy && (!cachingForced))) {
                    *gcConfig = PARTITIONED_GC_OFF;
                    numWarpsPerSM = numWarpsPerSubPartition * numSubPartitions;
                    maxBlocks     = numWarpsPerSM / warpsAllocatedPerCTA;
                }
            }
        }
        else {
            maxBlocks = INT_MAX;
        }
    }


    result->allocatedRegistersPerBlock = regsAllocatedPerCTA;

    *limit = maxBlocks;

    return status;
}

///////////////////////////////////
//      API Implementations      //
///////////////////////////////////

static __OCC_INLINE
lwdaOccError lwdaOccMaxActiveBlocksPerMultiprocessor(
    lwdaOccResult               *result,
    const lwdaOccDeviceProp     *properties,
    const lwdaOccFuncAttributes *attributes,
    const lwdaOccDeviceState    *state,
    int                          blockSize,
    size_t                       dynamicSmemSize)
{
    lwdaOccError status          = LWDA_OCC_SUCCESS;
    int          ctaLimitWarps   = 0;
    int          ctaLimitBlocks  = 0;
    int          ctaLimitSMem    = 0;
    int          ctaLimitRegs    = 0;
    int          ctaLimit        = 0;
    unsigned int limitingFactors = 0;
    
    lwdaOccPartitionedGCConfig gcConfig = PARTITIONED_GC_OFF;

    if (!result || !properties || !attributes || !state || blockSize <= 0) {
        return LWDA_OCC_ERROR_ILWALID_INPUT;
    }

    ///////////////////////////
    // Check user input
    ///////////////////////////

    status = lwdaOccInputCheck(properties, attributes, state);
    if (status != LWDA_OCC_SUCCESS) {
        return status;
    }

    ///////////////////////////
    // Initialization
    ///////////////////////////

    gcConfig = lwdaOccPartitionedGCExpected(properties, attributes);

    ///////////////////////////
    // Compute oclwpancy
    ///////////////////////////

    // Limits due to registers/SM
    // Also compute if partitioned global caching has to be turned off
    //
    status = lwdaOccMaxBlocksPerSMRegsLimit(&ctaLimitRegs, &gcConfig, result, properties, attributes, blockSize);
    if (status != LWDA_OCC_SUCCESS) {
        return status;
    }

    // Limits due to warps/SM
    //
    status = lwdaOccMaxBlocksPerSMWarpsLimit(&ctaLimitWarps, gcConfig, properties, attributes, blockSize);
    if (status != LWDA_OCC_SUCCESS) {
        return status;
    }

    // Limits due to blocks/SM
    //
    status = lwdaOccMaxBlocksPerMultiprocessor(&ctaLimitBlocks, properties);
    if (status != LWDA_OCC_SUCCESS) {
        return status;
    }

    // Limits due to shared memory/SM
    //
    status = lwdaOccMaxBlocksPerSMSmemLimit(&ctaLimitSMem, result, properties, attributes, state, blockSize, dynamicSmemSize);
    if (status != LWDA_OCC_SUCCESS) {
        return status;
    }

    ///////////////////////////
    // Overall oclwpancy
    ///////////////////////////

    // Overall limit is min() of limits due to above reasons
    //
    ctaLimit = __occMin(ctaLimitRegs, __occMin(ctaLimitSMem, __occMin(ctaLimitWarps, ctaLimitBlocks)));

    // Fill in the return values
    //
    // Determine oclwpancy limiting factors
    //
    if (ctaLimit == ctaLimitWarps) {
        limitingFactors |= OCC_LIMIT_WARPS;
    }
    if (ctaLimit == ctaLimitRegs) {
        limitingFactors |= OCC_LIMIT_REGISTERS;
    }
    if (ctaLimit == ctaLimitSMem) {
        limitingFactors |= OCC_LIMIT_SHARED_MEMORY;
    }
    if (ctaLimit == ctaLimitBlocks) {
        limitingFactors |= OCC_LIMIT_BLOCKS;
    }
    result->limitingFactors = limitingFactors;

    result->blockLimitRegs      = ctaLimitRegs;
    result->blockLimitSharedMem = ctaLimitSMem;
    result->blockLimitWarps     = ctaLimitWarps;
    result->blockLimitBlocks    = ctaLimitBlocks;
    result->partitionedGCConfig = gcConfig;

    // Final oclwpancy
    result->activeBlocksPerMultiprocessor = ctaLimit;

    return LWDA_OCC_SUCCESS;
}

static __OCC_INLINE
lwdaOccError lwdaOccMaxPotentialOclwpancyBlockSize(
    int                         *minGridSize,
    int                         *blockSize,
    const lwdaOccDeviceProp     *properties,
    const lwdaOccFuncAttributes *attributes,
    const lwdaOccDeviceState    *state,
    size_t                     (*blockSizeToDynamicSMemSize)(int),
    size_t                       dynamicSMemSize)
{
    lwdaOccError  status = LWDA_OCC_SUCCESS;
    lwdaOccResult result;

    // Limits
    int oclwpancyLimit;
    int granularity;
    int blockSizeLimit;

    // Recorded maximum
    int maxBlockSize = 0;
    int numBlocks    = 0;
    int maxOclwpancy = 0;

    // Temporary
    int blockSizeToTryAligned;
    int blockSizeToTry;
    int blockSizeLimitAligned;
    int oclwpancyInBlocks;
    int oclwpancyInThreads;

    ///////////////////////////
    // Check user input
    ///////////////////////////

    if (!minGridSize || !blockSize || !properties || !attributes || !state) {
        return LWDA_OCC_ERROR_ILWALID_INPUT;
    }

    status = lwdaOccInputCheck(properties, attributes, state);
    if (status != LWDA_OCC_SUCCESS) {
        return status;
    }

    /////////////////////////////////////////////////////////////////////////////////
    // Try each block size, and pick the block size with maximum oclwpancy
    /////////////////////////////////////////////////////////////////////////////////

    oclwpancyLimit = properties->maxThreadsPerMultiprocessor;
    granularity    = properties->warpSize;

    blockSizeLimit        = __occMin(properties->maxThreadsPerBlock, attributes->maxThreadsPerBlock);
    blockSizeLimitAligned = __occRoundUp(blockSizeLimit, granularity);

    for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) {
        blockSizeToTry = __occMin(blockSizeLimit, blockSizeToTryAligned);

        // Ignore dynamicSMemSize if the user provides a mapping
        //
        if (blockSizeToDynamicSMemSize) {
            dynamicSMemSize = (*blockSizeToDynamicSMemSize)(blockSizeToTry);
        }

        status = lwdaOccMaxActiveBlocksPerMultiprocessor(
            &result,
            properties,
            attributes,
            state,
            blockSizeToTry,
            dynamicSMemSize);

        if (status != LWDA_OCC_SUCCESS) {
            return status;
        }

        oclwpancyInBlocks = result.activeBlocksPerMultiprocessor;
        oclwpancyInThreads = blockSizeToTry * oclwpancyInBlocks;

        if (oclwpancyInThreads > maxOclwpancy) {
            maxBlockSize = blockSizeToTry;
            numBlocks    = oclwpancyInBlocks;
            maxOclwpancy = oclwpancyInThreads;
        }

        // Early out if we have reached the maximum
        //
        if (oclwpancyLimit == maxOclwpancy) {
            break;
        }
    }

    ///////////////////////////
    // Return best available
    ///////////////////////////

    // Suggested min grid size to achieve a full machine launch
    //
    *minGridSize = numBlocks * properties->numSms;
    *blockSize = maxBlockSize;

    return status;
}


#if defined(__cplusplus)

namespace {

__OCC_INLINE
lwdaOccError lwdaOccMaxPotentialOclwpancyBlockSize(
    int                         *minGridSize,
    int                         *blockSize,
    const lwdaOccDeviceProp     *properties,
    const lwdaOccFuncAttributes *attributes,
    const lwdaOccDeviceState    *state,
    size_t                       dynamicSMemSize)
{
    return lwdaOccMaxPotentialOclwpancyBlockSize(
        minGridSize,
        blockSize,
        properties,
        attributes,
        state,
        NULL,
        dynamicSMemSize);
}

template <typename UnaryFunction>
__OCC_INLINE
lwdaOccError lwdaOccMaxPotentialOclwpancyBlockSizeVariableSMem(
    int                         *minGridSize,
    int                         *blockSize,
    const lwdaOccDeviceProp     *properties,
    const lwdaOccFuncAttributes *attributes,
    const lwdaOccDeviceState    *state,
    UnaryFunction                blockSizeToDynamicSMemSize)
{
    lwdaOccError  status = LWDA_OCC_SUCCESS;
    lwdaOccResult result;

    // Limits
    int oclwpancyLimit;
    int granularity;
    int blockSizeLimit;

    // Recorded maximum
    int maxBlockSize = 0;
    int numBlocks    = 0;
    int maxOclwpancy = 0;

    // Temporary
    int blockSizeToTryAligned;
    int blockSizeToTry;
    int blockSizeLimitAligned;
    int oclwpancyInBlocks;
    int oclwpancyInThreads;
    size_t dynamicSMemSize;

    ///////////////////////////
    // Check user input
    ///////////////////////////

    if (!minGridSize || !blockSize || !properties || !attributes || !state) {
        return LWDA_OCC_ERROR_ILWALID_INPUT;
    }

    status = lwdaOccInputCheck(properties, attributes, state);
    if (status != LWDA_OCC_SUCCESS) {
        return status;
    }

    /////////////////////////////////////////////////////////////////////////////////
    // Try each block size, and pick the block size with maximum oclwpancy
    /////////////////////////////////////////////////////////////////////////////////

    oclwpancyLimit = properties->maxThreadsPerMultiprocessor;
    granularity    = properties->warpSize;
    blockSizeLimit        = __occMin(properties->maxThreadsPerBlock, attributes->maxThreadsPerBlock);
    blockSizeLimitAligned = __occRoundUp(blockSizeLimit, granularity);

    for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) {
        blockSizeToTry = __occMin(blockSizeLimit, blockSizeToTryAligned);

        dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry);

        status = lwdaOccMaxActiveBlocksPerMultiprocessor(
            &result,
            properties,
            attributes,
            state,
            blockSizeToTry,
            dynamicSMemSize);

        if (status != LWDA_OCC_SUCCESS) {
            return status;
        }

        oclwpancyInBlocks = result.activeBlocksPerMultiprocessor;

        oclwpancyInThreads = blockSizeToTry * oclwpancyInBlocks;

        if (oclwpancyInThreads > maxOclwpancy) {
            maxBlockSize = blockSizeToTry;
            numBlocks    = oclwpancyInBlocks;
            maxOclwpancy = oclwpancyInThreads;
        }

        // Early out if we have reached the maximum
        //
        if (oclwpancyLimit == maxOclwpancy) {
            break;
        }
    }

    ///////////////////////////
    // Return best available
    ///////////////////////////

    // Suggested min grid size to achieve a full machine launch
    //
    *minGridSize = numBlocks * properties->numSms;
    *blockSize = maxBlockSize;

    return status;
}

} // namespace anonymous

#endif /*__cplusplus */

#undef __OCC_INLINE

#endif /*__lwda_oclwpancy_h__*/
