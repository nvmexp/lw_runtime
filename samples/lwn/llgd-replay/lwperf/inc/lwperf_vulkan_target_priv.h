#ifndef LWPERF_VULKAN_TARGET_PRIV_H
#define LWPERF_VULKAN_TARGET_PRIV_H

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
#include "lwperf_common_priv.h"
#include "lwperf_target_priv.h"

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
 *  @file   lwperf_vulkan_target_priv.h
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
 
    typedef struct LWPW_VK_LoadDriver_PrivParams
    {
        /// [in] deprecated - use pfnGetInstanceProcAddr and pfnGetDeviceProcAddr params of GetDeviceIndex and
        /// BeginSession.
        LWPA_Bool useLoader;
    } LWPW_VK_LoadDriver_PrivParams;
#define LWPW_VK_LoadDriver_PrivParams_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_LoadDriver_PrivParams, useLoader)

/***************************************************************************//**
 *  @name   PC Sampling
 *  @{
 */


    typedef struct LWPW_EventStream LWPW_EventStream;

    typedef struct LWPW_VK_PcSampling_Queue_BeginSession_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkInstance instance;
        /// [in]
        VkPhysicalDevice physicalDevice;
        /// [in]
        VkDevice device;
        /// [in]
        VkQueue queue;
        /// [in] Either a pointer to the loaders vkGetInstanceProcAddr or a pointer to the next layers
        /// vkGetInstanceProcAddr
        void* pfnGetInstanceProcAddr;
        /// [in] Either a pointer to the loaders vkGetDeviceProcAddr or a pointer to the next layers vkGetDeviceProcAddr
        void* pfnGetDeviceProcAddr;
        /// [out] pointer to OS-allocated PerfmonBuffer (freed at EndSession)
        uint8_t* pPerfmonBuffer;
        /// [in] size of PerfmonBuffer
        size_t perfmonBufferSizeInBytes;
        /// [in] sample period of 2 ^ (5 + timespan) cycles
        size_t timespan;
        /// [in] list of requested counterIds returned by LWPW_PcSampling_GetCounterProperties()
        const uint64_t* pCounterIds;
        /// [in] count of pCounterIds
        size_t numCounterIds;
        /// [in] scratch buffer for raw PC counter data downloaded from perfmon buffer
        uint8_t* pPcCountersBuffer;
        /// [in] size of scratch buffer
        size_t pcCountersBufferSizeInBytes;
    } LWPW_VK_PcSampling_Queue_BeginSession_Params;
#define LWPW_VK_PcSampling_Queue_BeginSession_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_PcSampling_Queue_BeginSession_Params, pcCountersBufferSizeInBytes)

    LWPW_LOCAL
    LWPA_Status LWPW_VK_PcSampling_Queue_BeginSession(LWPW_VK_PcSampling_Queue_BeginSession_Params* pParams);

    typedef struct LWPW_VK_PcSampling_Queue_EndSession_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
    } LWPW_VK_PcSampling_Queue_EndSession_Params;
#define LWPW_VK_PcSampling_Queue_EndSession_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_PcSampling_Queue_EndSession_Params, queue)

    LWPW_LOCAL
    LWPA_Status LWPW_VK_PcSampling_Queue_EndSession(LWPW_VK_PcSampling_Queue_EndSession_Params* pParams);

    typedef struct LWPW_VK_PcSampling_Queue_GetSampleStats_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
        /// [out] number of unacknowledged bytes in perfmon buffer
        size_t numPerfmonBytesUnacknowledged;
        /// [out] true if PerfmonBuffer is full, possibly because buffer is not large enough or buffer was not consumed
        /// frequently enough
        LWPA_Bool perfmonBufferFull;
    } LWPW_VK_PcSampling_Queue_GetSampleStats_Params;
#define LWPW_VK_PcSampling_Queue_GetSampleStats_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_PcSampling_Queue_GetSampleStats_Params, perfmonBufferFull)

    /// This function should be called on one thread at a time, and not be conlwrrent with BeginSession/EndSession
    LWPW_LOCAL
    LWPA_Status LWPW_VK_PcSampling_Queue_GetSampleStats(LWPW_VK_PcSampling_Queue_GetSampleStats_Params* pParams);

    typedef struct LWPW_VK_PcSampling_Queue_GatherData_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
        /// [in] number of bytes available in perfmon buffer to decode. "bytesToDecode" is for backward compatibility
        /// only
        size_t numPerfmonBytesUnacknowledged;
        /// [in] Event stream's varDataBufferSizeInBytes must be at least: sizeof(LWPW_VK_PcSampling_DecodeStats) +
        /// maxNumPcs * GetPcCounterValuesStride()
        LWPW_EventStream* pEventStream;
        /// [in] Type by which counter data is grouped and aclwmulated
        LWPW_PcSampling_CounterDataGroup counterDataGroup;
        /// [out] number of bytes decoded
        size_t numPerfmonBytesDecoded;
        /// [out] The event stream is full, data in hEventStream needs to be consumed before next GatherData call
        LWPA_Bool eventStreamFull;
        /// [out]number of bytes dropped, possibly due to back pressure when sampler frequency is too high
        uint32_t droppedBytes;
        /// [out]number of recovery attempts
        uint32_t resyncCount;
        /// [out]one or more samplers have dropped too many bytes and are unrecoverable
        uint8_t overflow;
        /// [out]scratch buffer is full
        uint8_t scratchBufferFull;
    } LWPW_VK_PcSampling_Queue_GatherData_Params;
#define LWPW_VK_PcSampling_Queue_GatherData_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_PcSampling_Queue_GatherData_Params, scratchBufferFull)

    /// This function should be called on one thread at a time, and not be conlwrrent with BeginSession/EndSession.
    /// Perfmon buffer needs to be acknowledged by LWPW_Vk_PcSampling_Queue_AcknowledgePerfmonBuffer.
    LWPW_LOCAL
    LWPA_Status LWPW_VK_PcSampling_Queue_GatherData(LWPW_VK_PcSampling_Queue_GatherData_Params* pParams);

    typedef struct LWPW_VK_PcSampling_Queue_GatherData_V2_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
        /// [in] number of bytes available in perfmon buffer to decode. "bytesToDecode" is for backward compatibility
        /// only
        size_t numPerfmonBytesUnacknowledged;
        /// [in] Type by which counter data is grouped and aclwmulated
        LWPW_PcSampling_CounterDataGroup counterDataGroup;
        /// [out] number of bytes decoded
        size_t numPerfmonBytesDecoded;
        /// [out]number of bytes dropped, possibly due to back pressure when sampler frequency is too high
        uint32_t droppedBytes;
        /// [out]number of recovery attempts
        uint32_t resyncCount;
        /// [out]one or more samplers have dropped too many bytes and are unrecoverable
        uint8_t overflow;
        /// [in]user function to get the number of available slots in pc counter values buffer
        LWPW_PcSampling_GetNumAvailablePcCounterValuesFn pfnGetNumAvailablePcCounterValues;
        /// [in]user data pointer passed to pfnGetNumAvailablePcCounterValues
        void* pGetNumAvailablePcCounterValuesUserData;
        /// [in]user function to get the address to store pc counter values
        LWPW_PcSampling_GetPcCounterValuesFn pfnGetPcCounterValues;
        /// [in]user data pointer passed to pfnGetPcCounterValues
        void* pGetPcCounterValuesUserData;
    } LWPW_VK_PcSampling_Queue_GatherData_V2_Params;
#define LWPW_VK_PcSampling_Queue_GatherData_V2_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_PcSampling_Queue_GatherData_V2_Params, pGetPcCounterValuesUserData)

    /// This function should be called on one thread at a time, and not be conlwrrent with BeginSession/EndSession.
    /// Perfmon buffer needs to be acknowledged by LWPW_Vk_PcSampling_Queue_AcknowledgePerfmonBuffer.
    LWPW_LOCAL
    LWPA_Status LWPW_VK_PcSampling_Queue_GatherData_V2(LWPW_VK_PcSampling_Queue_GatherData_V2_Params* pParams);

    typedef struct LWPW_VK_PcSampling_Queue_AcknowledgePerfmonBuffer_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkQueue queue;
        /// [in] number of bytes in perfmon buffer to acknowledge.
        size_t numPerfmonBytesUnacknowledged;
    } LWPW_VK_PcSampling_Queue_AcknowledgePerfmonBuffer_Params;
#define LWPW_VK_PcSampling_Queue_AcknowledgePerfmonBuffer_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_PcSampling_Queue_AcknowledgePerfmonBuffer_Params, numPerfmonBytesUnacknowledged)

    /// This function should be called on one thread at a time, and not be conlwrrent with BeginSession/EndSession
    LWPW_LOCAL
    LWPA_Status LWPW_VK_PcSampling_Queue_AcknowledgePerfmonBuffer(LWPW_VK_PcSampling_Queue_AcknowledgePerfmonBuffer_Params* pParams);

    typedef struct LWPW_VK_PcSampling_CommandBuffer_StartMeasuring_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkCommandBuffer commandBuffer;
        /// [in] If true, pushes into 3D subchannel; else compute subchannel
        LWPA_Bool push3D;
    } LWPW_VK_PcSampling_CommandBuffer_StartMeasuring_Params;
#define LWPW_VK_PcSampling_CommandBuffer_StartMeasuring_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_PcSampling_CommandBuffer_StartMeasuring_Params, push3D)

    LWPW_LOCAL
    LWPA_Status LWPW_VK_PcSampling_CommandBuffer_StartMeasuring(LWPW_VK_PcSampling_CommandBuffer_StartMeasuring_Params* pParams);

    typedef struct LWPW_VK_PcSampling_CommandBuffer_StopMeasuring_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkCommandBuffer commandBuffer;
        /// [in] If true, pushes into 3D subchannel; else compute subchannel
        LWPA_Bool push3D;
    } LWPW_VK_PcSampling_CommandBuffer_StopMeasuring_Params;
#define LWPW_VK_PcSampling_CommandBuffer_StopMeasuring_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_PcSampling_CommandBuffer_StopMeasuring_Params, push3D)

    LWPW_LOCAL
    LWPA_Status LWPW_VK_PcSampling_CommandBuffer_StopMeasuring(LWPW_VK_PcSampling_CommandBuffer_StopMeasuring_Params* pParams);

    typedef struct LWPW_VK_PcSampling_CommandBuffer_WaitForIdle_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        VkCommandBuffer commandBuffer;
    } LWPW_VK_PcSampling_CommandBuffer_WaitForIdle_Params;
#define LWPW_VK_PcSampling_CommandBuffer_WaitForIdle_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_PcSampling_CommandBuffer_WaitForIdle_Params, commandBuffer)

    LWPW_LOCAL
    LWPA_Status LWPW_VK_PcSampling_CommandBuffer_WaitForIdle(LWPW_VK_PcSampling_CommandBuffer_WaitForIdle_Params* pParams);

    typedef struct LWPW_VK_PcSampling_GetMinimumScratchBufferSize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] number of requested counters
        size_t numCounterIds;
        /// [out]
        size_t scratchBufferSizeInBytes;
    } LWPW_VK_PcSampling_GetMinimumScratchBufferSize_Params;
#define LWPW_VK_PcSampling_GetMinimumScratchBufferSize_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_PcSampling_GetMinimumScratchBufferSize_Params, scratchBufferSizeInBytes)

    LWPW_LOCAL
    LWPA_Status LWPW_VK_PcSampling_GetMinimumScratchBufferSize(LWPW_VK_PcSampling_GetMinimumScratchBufferSize_Params* pParams);

    typedef struct LWPW_VK_PcSampling_GetMinimumVarDataBufferSize_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in] number of requested counters
        size_t numCounterIds;
        /// [out]
        size_t varDataBufferSizeInBytes;
    } LWPW_VK_PcSampling_GetMinimumVarDataBufferSize_Params;
#define LWPW_VK_PcSampling_GetMinimumVarDataBufferSize_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_PcSampling_GetMinimumVarDataBufferSize_Params, varDataBufferSizeInBytes)

    LWPW_LOCAL
    LWPA_Status LWPW_VK_PcSampling_GetMinimumVarDataBufferSize(LWPW_VK_PcSampling_GetMinimumVarDataBufferSize_Params* pParams);

/**
 *  @}
 ******************************************************************************/
 
    typedef struct LWPW_VK_PcSampling_IsGpuSupported_Params
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
    } LWPW_VK_PcSampling_IsGpuSupported_Params;
#define LWPW_VK_PcSampling_IsGpuSupported_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_PcSampling_IsGpuSupported_Params, sliSupportLevel)

    /// LWPW_VK_LoadDriver must be called prior to this API
    LWPW_LOCAL
    LWPA_Status LWPW_VK_PcSampling_IsGpuSupported(LWPW_VK_PcSampling_IsGpuSupported_Params* pParams);



#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility pop
#endif

#endif // LWPERF_VULKAN_TARGET_PRIV_H
