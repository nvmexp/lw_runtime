#ifndef LWPERF_VULKAN_TARGET_H
#define LWPERF_VULKAN_TARGET_H

/*
 * Copyright 2014-2021  LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to LWPU ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and conditions
 * of a form of LWPU software license agreement.
 *
 * LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include <stddef.h>
#include <stdint.h>
#include "lwperf_common.h"
#include "lwperf_target.h"

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility push(default)
    #if !defined(LWPW_LOCAL)
        #define LWPW_LOCAL __attribute__ ((visibility ("hidden")))
    #endif
#else
    #if !defined(LWPW_LOCAL)
        #define LWPW_LOCAL
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @file   lwperf_vulkan_target.h
 */

/***************************************************************************//**
 *  @name   External Types
 *  @{
 */


    struct VkInstance_T;
    typedef struct VkInstance_T* VkInstance;
    struct VkPhysicalDevice_T;
    typedef struct VkPhysicalDevice_T* VkPhysicalDevice;
    struct VkDevice_T;
    typedef struct VkDevice_T* VkDevice;
    struct VkQueue_T;
    typedef struct VkQueue_T* VkQueue;
    struct VkCommandBuffer_T;
    typedef struct VkCommandBuffer_T* VkCommandBuffer;



#if defined(__LP64__) || defined(_WIN64) || (defined(__x86_64__) && !defined(__ILP32__) ) || defined(_M_X64) || defined(__ia64) || defined (_M_IA64) || defined(__aarch64__) || defined(__powerpc64__)
    typedef struct VkDeviceMemory_T *VkDeviceMemory;
#else
    typedef uint64_t VkDeviceMemory;
#endif


/**
 *  @}
 ******************************************************************************/
 
    typedef struct LWPW_VK_Profiler_CounterDataImageOptions
    {
        /// [in]
        size_t structSize;
        /// The CounterDataPrefix generated from e.g.    lwperf2 initdata   or
        /// LWPW_CounterDataBuilder_GetCounterDataPrefix().  Must be align(8).
        const uint8_t* pCounterDataPrefix;
        size_t counterDataPrefixSize;
        /// max number of ranges that can be specified
        uint32_t maxNumRanges;
        /// max number of RangeTree nodes; must be >= maxNumRanges
        uint32_t maxNumRangeTreeNodes;
        /// max string length of each RangeName, including the trailing NUL character
        uint32_t maxRangeNameLength;
    } LWPW_VK_Profiler_CounterDataImageOptions;
#define LWPW_VK_Profiler_CounterDataImageOptions_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CounterDataImageOptions, maxRangeNameLength)

    typedef struct LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t counterDataImageOptionsSize;
        /// [in]
        const LWPW_VK_Profiler_CounterDataImageOptions* pOptions;
        /// [out]
        size_t counterDataImageSize;
    } LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Params;
#define LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Params, counterDataImageSize)

    LWPA_Status LWPW_VK_Profiler_CounterDataImage_CallwlateSize(LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Params* pParams);

    typedef struct LWPW_VK_Profiler_CounterDataImage_Initialize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t counterDataImageOptionsSize;
        /// [in]
        const LWPW_VK_Profiler_CounterDataImageOptions* pOptions;
        /// [in]
        size_t counterDataImageSize;
        /// [in] The buffer to be written.
        uint8_t* pCounterDataImage;
    } LWPW_VK_Profiler_CounterDataImage_Initialize_Params;
#define LWPW_VK_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CounterDataImage_Initialize_Params, pCounterDataImage)

    LWPA_Status LWPW_VK_Profiler_CounterDataImage_Initialize(LWPW_VK_Profiler_CounterDataImage_Initialize_Params* pParams);

    typedef struct LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t counterDataImageSize;
        /// [in]
        uint8_t* pCounterDataImage;
        /// [out]
        size_t counterDataScratchBufferSize;
    } LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params;
#define LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params, counterDataScratchBufferSize)

    LWPA_Status LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize(LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params* pParams);

    typedef struct LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t counterDataImageSize;
        /// [in]
        uint8_t* pCounterDataImage;
        /// [in]
        size_t counterDataScratchBufferSize;
        /// [in] The scratch buffer to be written.
        uint8_t* pCounterDataScratchBuffer;
    } LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params;
#define LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params, pCounterDataScratchBuffer)

    LWPA_Status LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer(LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams);

    typedef struct LWPW_VK_LoadDriver_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkInstance instance;
    } LWPW_VK_LoadDriver_Params;
#define LWPW_VK_LoadDriver_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_LoadDriver_Params, instance)

    LWPA_Status LWPW_VK_LoadDriver(LWPW_VK_LoadDriver_Params* pParams);

    typedef struct LWPW_VK_Device_GetDeviceIndex_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkPhysicalDevice physicalDevice;
        /// [in]
        size_t sliIndex;
        /// [out]
        size_t deviceIndex;
        /// [in]
        VkInstance instance;
        /// [in]
        VkDevice device;
        /// [in] Either a pointer to the loaders vkGetInstanceProcAddr or a pointer to the next layers
        /// vkGetInstanceProcAddr
        void* pfnGetInstanceProcAddr;
        /// [in] Either a pointer to the loaders vkGetDeviceProcAddr or a pointer to the next layers vkGetDeviceProcAddr
        void* pfnGetDeviceProcAddr;
    } LWPW_VK_Device_GetDeviceIndex_Params;
#define LWPW_VK_Device_GetDeviceIndex_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Device_GetDeviceIndex_Params, pfnGetDeviceProcAddr)

    LWPA_Status LWPW_VK_Device_GetDeviceIndex(LWPW_VK_Device_GetDeviceIndex_Params* pParams);

    typedef struct LWPW_VK_Profiler_GetRequiredInstanceExtensions_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [out]
        const char* const* ppInstanceExtensionNames;
        /// [out]
        size_t numInstanceExtensionNames;
        /// [in] Vulkan API version (VK_API_VERSION_*)
        uint32_t apiVersion;
        /// [out] is apiVersion officially supported by the LwPerf API
        LWPA_Bool isOfficiallySupportedVersion;
    } LWPW_VK_Profiler_GetRequiredInstanceExtensions_Params;
#define LWPW_VK_Profiler_GetRequiredInstanceExtensions_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_GetRequiredInstanceExtensions_Params, isOfficiallySupportedVersion)

    LWPA_Status LWPW_VK_Profiler_GetRequiredInstanceExtensions(LWPW_VK_Profiler_GetRequiredInstanceExtensions_Params* pParams);

    typedef struct LWPW_VK_Profiler_GetRequiredDeviceExtensions_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [out]
        const char* const* ppDeviceExtensionNames;
        /// [out]
        size_t numDeviceExtensionNames;
        /// [in] Vulkan API version (VK_API_VERSION_*)
        uint32_t apiVersion;
        /// [out] is apiVersion officially supported by the LwPerf API
        LWPA_Bool isOfficiallySupportedVersion;
    } LWPW_VK_Profiler_GetRequiredDeviceExtensions_Params;
#define LWPW_VK_Profiler_GetRequiredDeviceExtensions_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_GetRequiredDeviceExtensions_Params, isOfficiallySupportedVersion)

    LWPA_Status LWPW_VK_Profiler_GetRequiredDeviceExtensions(LWPW_VK_Profiler_GetRequiredDeviceExtensions_Params* pParams);

    typedef struct LWPW_VK_Profiler_CalcTraceBufferSize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] Maximum number of Push/Pop pairs that can be recorded in a single pass.
        size_t maxRangesPerPass;
        /// [in] for sizing internal buffers
        size_t avgRangeNameLength;
        /// [out] TraceBuffer size for a single pass.  Pass this to
        /// LWPW_VK_Profiler_BeginSession_Params::traceBufferSize.
        size_t traceBufferSize;
    } LWPW_VK_Profiler_CalcTraceBufferSize_Params;
#define LWPW_VK_Profiler_CalcTraceBufferSize_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CalcTraceBufferSize_Params, traceBufferSize)

    LWPA_Status LWPW_VK_Profiler_CalcTraceBufferSize(LWPW_VK_Profiler_CalcTraceBufferSize_Params* pParams);

    typedef struct LWPW_VK_Profiler_Queue_BeginSession_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkDevice device;
        /// [in]
        VkQueue queue;
        /// [in] Set to 1 if every pass is synchronized with CPU; for asynchronous collection, increase to
        /// (softwarePipelineDepth + 2).
        size_t numTraceBuffers;
        /// [in] Size of the per-pass TraceBuffer in bytes.  The profiler allocates a numTraceBuffers * traceBufferSize
        /// internally.
        size_t traceBufferSize;
        /// [in]
        uint8_t* pTraceArena;
        /// [in] VkDeviceAddress of pTraceArena.
        uint64_t traceArenaGpuAddress;
        /// [in] The VK object holding tracebuffer. It must be created at the index of
        /// VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT flags.
        VkDeviceMemory traceArenaDeviceMemory;
        /// [in] size of PerfmonBuffer.
        size_t perfmonBufferSize;
        /// [in] pointer to page-aligned PerfmonBuffer.
        uint8_t* pPerfmonBuffer;
        /// [in]
        VkInstance instance;
        /// [in]
        VkPhysicalDevice physicalDevice;
        /// [in] Either a pointer to the loaders vkGetInstanceProcAddr or a pointer to the next layers
        /// vkGetInstanceProcAddr
        void* pfnGetInstanceProcAddr;
        /// [in] Either a pointer to the loaders vkGetDeviceProcAddr or a pointer to the next layers vkGetDeviceProcAddr
        void* pfnGetDeviceProcAddr;
    } LWPW_VK_Profiler_Queue_BeginSession_Params;
#define LWPW_VK_Profiler_Queue_BeginSession_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_Queue_BeginSession_Params, pfnGetDeviceProcAddr)

    LWPA_Status LWPW_VK_Profiler_Queue_BeginSession(LWPW_VK_Profiler_Queue_BeginSession_Params* pParams);

    typedef struct LWPW_VK_Profiler_Queue_EndSession_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
    } LWPW_VK_Profiler_Queue_EndSession_Params;
#define LWPW_VK_Profiler_Queue_EndSession_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_Queue_EndSession_Params, queue)

    LWPA_Status LWPW_VK_Profiler_Queue_EndSession(LWPW_VK_Profiler_Queue_EndSession_Params* pParams);

    typedef struct LWPW_VK_Profiler_Queue_SetConfig_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
        /// [in] Config created by e.g.    lwperf2 configure   or   LWPW_RawMetricsConfig_GetConfigImage().  Must be
        /// align(8).
        const uint8_t* pConfig;
        size_t configSize;
        /// [in] the lowest nesting level to be profiled; must be >= 1
        uint16_t minNestingLevel;
        /// [in] the number of nesting levels to profile; must be >= 1
        uint16_t numNestingLevels;
        /// [in] Set this to zero for in-app replay.  Set this to the output of EndPass() for application replay.
        size_t passIndex;
        /// [in] Set this to minNestingLevel for in-app replay.  Set this to the output of EndPass() for application
        /// replay.
        uint16_t targetNestingLevel;
    } LWPW_VK_Profiler_Queue_SetConfig_Params;
#define LWPW_VK_Profiler_Queue_SetConfig_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_Queue_SetConfig_Params, targetNestingLevel)

    LWPA_Status LWPW_VK_Profiler_Queue_SetConfig(LWPW_VK_Profiler_Queue_SetConfig_Params* pParams);

    typedef struct LWPW_VK_Profiler_Queue_ClearConfig_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
    } LWPW_VK_Profiler_Queue_ClearConfig_Params;
#define LWPW_VK_Profiler_Queue_ClearConfig_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_Queue_ClearConfig_Params, queue)

    LWPA_Status LWPW_VK_Profiler_Queue_ClearConfig(LWPW_VK_Profiler_Queue_ClearConfig_Params* pParams);

    typedef struct LWPW_VK_Profiler_CommandBuffer_BeginPass_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkCommandBuffer commandBuffer;
    } LWPW_VK_Profiler_CommandBuffer_BeginPass_Params;
#define LWPW_VK_Profiler_CommandBuffer_BeginPass_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CommandBuffer_BeginPass_Params, commandBuffer)

    LWPA_Status LWPW_VK_Profiler_CommandBuffer_BeginPass(LWPW_VK_Profiler_CommandBuffer_BeginPass_Params* pParams);

    typedef struct LWPW_VK_Profiler_CommandBuffer_EndPass_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkCommandBuffer commandBuffer;
    } LWPW_VK_Profiler_CommandBuffer_EndPass_Params;
#define LWPW_VK_Profiler_CommandBuffer_EndPass_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CommandBuffer_EndPass_Params, commandBuffer)

    LWPA_Status LWPW_VK_Profiler_CommandBuffer_EndPass(LWPW_VK_Profiler_CommandBuffer_EndPass_Params* pParams);

    typedef struct LWPW_VK_Profiler_Queue_GetLastError_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
    } LWPW_VK_Profiler_Queue_GetLastError_Params;
#define LWPW_VK_Profiler_Queue_GetLastError_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_Queue_GetLastError_Params, queue)

    LWPA_Status LWPW_VK_Profiler_Queue_GetLastError(LWPW_VK_Profiler_Queue_GetLastError_Params* pParams);

    typedef struct LWPW_VK_Profiler_CommandBuffer_PushRange_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkCommandBuffer commandBuffer;
        /// [in] specifies the range that subsequent launches' counters will be assigned to; must not be NULL
        const char* pRangeName;
        /// [in] assign to strlen(pRangeName) if known; if set to zero, the library will call strlen()
        size_t rangeNameLength;
    } LWPW_VK_Profiler_CommandBuffer_PushRange_Params;
#define LWPW_VK_Profiler_CommandBuffer_PushRange_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CommandBuffer_PushRange_Params, rangeNameLength)

    LWPA_Status LWPW_VK_Profiler_CommandBuffer_PushRange(LWPW_VK_Profiler_CommandBuffer_PushRange_Params* pParams);

    typedef struct LWPW_VK_Profiler_CommandBuffer_PopRange_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkCommandBuffer commandBuffer;
    } LWPW_VK_Profiler_CommandBuffer_PopRange_Params;
#define LWPW_VK_Profiler_CommandBuffer_PopRange_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_CommandBuffer_PopRange_Params, commandBuffer)

    LWPA_Status LWPW_VK_Profiler_CommandBuffer_PopRange(LWPW_VK_Profiler_CommandBuffer_PopRange_Params* pParams);

    typedef struct LWPW_VK_Profiler_Queue_DecodeCounters_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
        /// [in]
        size_t counterDataImageSize;
        /// [in]
        uint8_t* pCounterDataImage;
        /// [in]
        size_t counterDataScratchBufferSize;
        /// [in]
        uint8_t* pCounterDataScratchBuffer;
        /// [out] number of ranges whose data was dropped in the processed pass
        size_t numRangesDropped;
        /// [out] number of bytes not written to TraceBuffer due to buffer full
        size_t numTraceBytesDropped;
        /// [out] true if a pass was successfully decoded
        LWPA_Bool onePassCollected;
        /// [out] becomes true when the last pass has been decoded
        LWPA_Bool allPassesCollected;
        /// [out] the Config decoded by this call
        const uint8_t* pConfigDecoded;
        /// [out] the passIndex decoded
        size_t passIndexDecoded;
    } LWPW_VK_Profiler_Queue_DecodeCounters_Params;
#define LWPW_VK_Profiler_Queue_DecodeCounters_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_Queue_DecodeCounters_Params, passIndexDecoded)

    LWPA_Status LWPW_VK_Profiler_Queue_DecodeCounters(LWPW_VK_Profiler_Queue_DecodeCounters_Params* pParams);

    typedef struct LWPW_VK_Profiler_IsGpuSupported_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        size_t deviceIndex;
        /// [out]
        LWPA_Bool isSupported;
        /// [out]
        LWPW_GpuArchitectureSupportLevel gpuArchitectureSupportLevel;
        /// [out]
        LWPW_SliSupportLevel sliSupportLevel;
    } LWPW_VK_Profiler_IsGpuSupported_Params;
#define LWPW_VK_Profiler_IsGpuSupported_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_Profiler_IsGpuSupported_Params, sliSupportLevel)

    /// LWPW_VK_LoadDriver must be called prior to this API
    LWPA_Status LWPW_VK_Profiler_IsGpuSupported(LWPW_VK_Profiler_IsGpuSupported_Params* pParams);



#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility pop
#endif

#endif // LWPERF_VULKAN_TARGET_H
