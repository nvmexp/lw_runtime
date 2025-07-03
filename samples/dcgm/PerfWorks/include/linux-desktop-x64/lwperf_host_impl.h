/*
 * Copyright 2014-2019  LWPU Corporation.  All rights reserved.
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

#include <stdlib.h>
#include <string.h>

#if _WIN32
#include <wchar.h>
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#include <lwperf_host.h>
#include <lwperf_target.h>
#include <lwperf_lwda_host.h>
#include <lwperf_lwda_target.h>
#include <lwperf_vulkan_host.h>
#include <lwperf_vulkan_target.h>
#include <lwperf_dcgm_host.h>
#include <lwperf_dcgm_target.h>
#include <lwperf_opengl_host.h>
#include <lwperf_opengl_target.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef LWPA_GenericFn (*LWPA_GetProcAddress_Fn)(const char* pFunctionName);
typedef LWPA_Status (*LWPW_SetLibraryLoadPaths_Fn)(LWPW_SetLibraryLoadPaths_Params* pParams);
typedef LWPA_Status (*LWPW_SetLibraryLoadPathsW_Fn)(LWPW_SetLibraryLoadPathsW_Params* pParams);
typedef LWPA_Status (*LWPA_InitializeHost_Fn)(void);
typedef LWPA_Status (*LWPW_InitializeHost_Fn)(LWPW_InitializeHost_Params* pParams);
typedef LWPA_Status (*LWPA_CounterData_CallwlateCounterDataImageCopySize_Fn)(const LWPA_CounterDataImageCopyOptions* pCounterDataImageCopyOptions, const uint8_t* pCounterDataSrc, size_t* pCopyDataImageCounterSize);
typedef LWPA_Status (*LWPW_CounterData_CallwlateCounterDataImageCopySize_Fn)(LWPW_CounterData_CallwlateCounterDataImageCopySize_Params* pParams);
typedef LWPA_Status (*LWPA_CounterData_InitializeCounterDataImageCopy_Fn)(const LWPA_CounterDataImageCopyOptions* pCounterDataImageCopyOptions, const uint8_t* pCounterDataSrc, uint8_t* pCounterDataDst);
typedef LWPA_Status (*LWPW_CounterData_InitializeCounterDataImageCopy_Fn)(LWPW_CounterData_InitializeCounterDataImageCopy_Params* pParams);
typedef LWPA_Status (*LWPA_CounterDataCombiner_Create_Fn)(const LWPA_CounterDataCombinerOptions* pCounterDataCombinerOptions, LWPA_CounterDataCombiner** ppCounterDataCombiner);
typedef LWPA_Status (*LWPW_CounterDataCombiner_Create_Fn)(LWPW_CounterDataCombiner_Create_Params* pParams);
typedef LWPA_Status (*LWPA_CounterDataCombiner_Destroy_Fn)(LWPA_CounterDataCombiner* pCounterDataCombiner);
typedef LWPA_Status (*LWPW_CounterDataCombiner_Destroy_Fn)(LWPW_CounterDataCombiner_Destroy_Params* pParams);
typedef LWPA_Status (*LWPA_CounterDataCombiner_CreateRange_Fn)(LWPA_CounterDataCombiner* pCounterDataCombiner, size_t numDescriptions, const char* const* ppDescriptions, size_t* pRangeIndexDst);
typedef LWPA_Status (*LWPW_CounterDataCombiner_CreateRange_Fn)(LWPW_CounterDataCombiner_CreateRange_Params* pParams);
typedef LWPA_Status (*LWPA_CounterDataCombiner_AclwmulateIntoRange_Fn)(LWPA_CounterDataCombiner* pCounterDataCombiner, size_t rangeIndexDst, uint32_t dstMultiplier, const uint8_t* pCounterDataSrc, size_t rangeIndexSrc, uint32_t srcMultiplier);
typedef LWPA_Status (*LWPW_CounterDataCombiner_AclwmulateIntoRange_Fn)(LWPW_CounterDataCombiner_AclwmulateIntoRange_Params* pParams);
typedef LWPA_Status (*LWPW_CounterDataCombiner_SumIntoRange_Fn)(LWPW_CounterDataCombiner_SumIntoRange_Params* pParams);
typedef LWPA_Status (*LWPW_CounterDataCombiner_WeightedSumIntoRange_Fn)(LWPW_CounterDataCombiner_WeightedSumIntoRange_Params* pParams);
typedef LWPA_Status (*LWPA_GetSupportedChipNames_Fn)(LWPA_SupportedChipNames* pSupportedChipNames);
typedef LWPA_Status (*LWPW_GetSupportedChipNames_Fn)(LWPW_GetSupportedChipNames_Params* pParams);
typedef LWPA_Status (*LWPA_RawMetricsConfig_Create_Fn)(const LWPA_RawMetricsConfigOptions* pMetricsConfigOptions, LWPA_RawMetricsConfig** ppRawMetricsConfig);
typedef LWPA_Status (*LWPA_RawMetricsConfig_Destroy_Fn)(LWPA_RawMetricsConfig* pRawMetricsConfig);
typedef LWPA_Status (*LWPW_RawMetricsConfig_Destroy_Fn)(LWPW_RawMetricsConfig_Destroy_Params* pParams);
typedef LWPA_Status (*LWPA_RawMetricsConfig_BeginPassGroup_Fn)(LWPA_RawMetricsConfig* pRawMetricsConfig, const LWPA_RawMetricsPassGroupOptions* pRawMetricsPassGroupOptions);
typedef LWPA_Status (*LWPW_RawMetricsConfig_BeginPassGroup_Fn)(LWPW_RawMetricsConfig_BeginPassGroup_Params* pParams);
typedef LWPA_Status (*LWPA_RawMetricsConfig_EndPassGroup_Fn)(LWPA_RawMetricsConfig* pRawMetricsConfig);
typedef LWPA_Status (*LWPW_RawMetricsConfig_EndPassGroup_Fn)(LWPW_RawMetricsConfig_EndPassGroup_Params* pParams);
typedef LWPA_Status (*LWPA_RawMetricsConfig_GetNumMetrics_Fn)(const LWPA_RawMetricsConfig* pRawMetricsConfig, size_t* pNumMetrics);
typedef LWPA_Status (*LWPW_RawMetricsConfig_GetNumMetrics_Fn)(LWPW_RawMetricsConfig_GetNumMetrics_Params* pParams);
typedef LWPA_Status (*LWPA_RawMetricsConfig_GetMetricProperties_Fn)(const LWPA_RawMetricsConfig* pRawMetricsConfig, size_t metricIndex, LWPA_RawMetricProperties* pRawMetricProperties);
typedef LWPA_Status (*LWPW_RawMetricsConfig_GetMetricProperties_Fn)(LWPW_RawMetricsConfig_GetMetricProperties_Params* pParams);
typedef LWPA_Status (*LWPA_RawMetricsConfig_AddMetrics_Fn)(LWPA_RawMetricsConfig* pRawMetricsConfig, const LWPA_RawMetricRequest* pRawMetricRequests, size_t numMetricRequests);
typedef LWPA_Status (*LWPW_RawMetricsConfig_AddMetrics_Fn)(LWPW_RawMetricsConfig_AddMetrics_Params* pParams);
typedef LWPA_Status (*LWPA_RawMetricsConfig_IsAddMetricsPossible_Fn)(const LWPA_RawMetricsConfig* pRawMetricsConfig, const LWPA_RawMetricRequest* pRawMetricRequests, size_t numMetricRequests, LWPA_Bool* pIsPossible);
typedef LWPA_Status (*LWPW_RawMetricsConfig_IsAddMetricsPossible_Fn)(LWPW_RawMetricsConfig_IsAddMetricsPossible_Params* pParams);
typedef LWPA_Status (*LWPA_RawMetricsConfig_GenerateConfigImage_Fn)(LWPA_RawMetricsConfig* pRawMetricsConfig);
typedef LWPA_Status (*LWPW_RawMetricsConfig_GenerateConfigImage_Fn)(LWPW_RawMetricsConfig_GenerateConfigImage_Params* pParams);
typedef LWPA_Status (*LWPA_RawMetricsConfig_GetConfigImage_Fn)(const LWPA_RawMetricsConfig* pRawMetricsConfig, size_t bufferSize, uint8_t* pBuffer, size_t* pBufferSize);
typedef LWPA_Status (*LWPW_RawMetricsConfig_GetConfigImage_Fn)(LWPW_RawMetricsConfig_GetConfigImage_Params* pParams);
typedef LWPA_Status (*LWPA_RawMetricsConfig_GetNumPasses_Fn)(const LWPA_RawMetricsConfig* pRawMetricsConfig, size_t* pNumPipelinedPasses, size_t* pNumIsolatedPasses);
typedef LWPA_Status (*LWPW_RawMetricsConfig_GetNumPasses_Fn)(LWPW_RawMetricsConfig_GetNumPasses_Params* pParams);
typedef LWPA_Status (*LWPA_CounterDataBuilder_Create_Fn)(const LWPA_CounterDataBuilderOptions* pOptions, LWPA_CounterDataBuilder** ppCounterDataBuilder);
typedef LWPA_Status (*LWPW_CounterDataBuilder_Create_Fn)(LWPW_CounterDataBuilder_Create_Params* pParams);
typedef LWPA_Status (*LWPA_CounterDataBuilder_Destroy_Fn)(LWPA_CounterDataBuilder* pCounterDataBuilder);
typedef LWPA_Status (*LWPW_CounterDataBuilder_Destroy_Fn)(LWPW_CounterDataBuilder_Destroy_Params* pParams);
typedef LWPA_Status (*LWPA_CounterDataBuilder_AddMetrics_Fn)(LWPA_CounterDataBuilder* pCounterDataBuilder, const LWPA_RawMetricRequest* pRawMetricRequests, size_t numMetricRequests);
typedef LWPA_Status (*LWPW_CounterDataBuilder_AddMetrics_Fn)(LWPW_CounterDataBuilder_AddMetrics_Params* pParams);
typedef LWPA_Status (*LWPA_CounterDataBuilder_GetCounterDataPrefix_Fn)(LWPA_CounterDataBuilder* pCounterDataBuilder, size_t bufferSize, uint8_t* pBuffer, size_t* pBufferSize);
typedef LWPA_Status (*LWPW_CounterDataBuilder_GetCounterDataPrefix_Fn)(LWPW_CounterDataBuilder_GetCounterDataPrefix_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_Create_Fn)(const LWPA_MetricsContextOptions* pMetricsContextOptions, LWPA_MetricsContext** ppMetricsContext);
typedef LWPA_Status (*LWPA_MetricsContext_Destroy_Fn)(LWPA_MetricsContext* pMetricsContext);
typedef LWPA_Status (*LWPW_MetricsContext_Destroy_Fn)(LWPW_MetricsContext_Destroy_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_RunScript_Fn)(LWPA_MetricsContext* pMetricsContext, const LWPA_MetricsScriptOptions* pOptions);
typedef LWPA_Status (*LWPW_MetricsContext_RunScript_Fn)(LWPW_MetricsContext_RunScript_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_ExecScript_Begin_Fn)(LWPA_MetricsContext* pMetricsContext, LWPA_MetricsExecOptions* pOptions);
typedef LWPA_Status (*LWPW_MetricsContext_ExecScript_Begin_Fn)(LWPW_MetricsContext_ExecScript_Begin_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_ExecScript_End_Fn)(LWPA_MetricsContext* pMetricsContext);
typedef LWPA_Status (*LWPW_MetricsContext_ExecScript_End_Fn)(LWPW_MetricsContext_ExecScript_End_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_GetCounterNames_Begin_Fn)(LWPA_MetricsContext* pMetricsContext, size_t* pNumCounters, const char* const** pppCounterNames);
typedef LWPA_Status (*LWPW_MetricsContext_GetCounterNames_Begin_Fn)(LWPW_MetricsContext_GetCounterNames_Begin_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_GetCounterNames_End_Fn)(LWPA_MetricsContext* pMetricsContext);
typedef LWPA_Status (*LWPW_MetricsContext_GetCounterNames_End_Fn)(LWPW_MetricsContext_GetCounterNames_End_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_GetThroughputNames_Begin_Fn)(LWPA_MetricsContext* pMetricsContext, size_t* pNumThroughputs, const char* const** pppThroughputName);
typedef LWPA_Status (*LWPW_MetricsContext_GetThroughputNames_Begin_Fn)(LWPW_MetricsContext_GetThroughputNames_Begin_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_GetThroughputNames_End_Fn)(LWPA_MetricsContext* pMetricsContext);
typedef LWPA_Status (*LWPW_MetricsContext_GetThroughputNames_End_Fn)(LWPW_MetricsContext_GetThroughputNames_End_Params* pParams);
typedef LWPA_Status (*LWPW_MetricsContext_GetRatioNames_Begin_Fn)(LWPW_MetricsContext_GetRatioNames_Begin_Params* pParams);
typedef LWPA_Status (*LWPW_MetricsContext_GetRatioNames_End_Fn)(LWPW_MetricsContext_GetRatioNames_End_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_GetMetricNames_Begin_Fn)(LWPA_MetricsContext* pMetricsContext, LWPA_MetricsEnumerationOptions* pOptions);
typedef LWPA_Status (*LWPW_MetricsContext_GetMetricNames_Begin_Fn)(LWPW_MetricsContext_GetMetricNames_Begin_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_GetMetricNames_End_Fn)(LWPA_MetricsContext* pMetricsContext);
typedef LWPA_Status (*LWPW_MetricsContext_GetMetricNames_End_Fn)(LWPW_MetricsContext_GetMetricNames_End_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_GetThroughputBreakdown_Begin_Fn)(LWPA_MetricsContext* pMetricsContext, const char* pThroughputName, const char* const** pppCounterNames, const char* const** pppSubThroughputNames);
typedef LWPA_Status (*LWPW_MetricsContext_GetThroughputBreakdown_Begin_Fn)(LWPW_MetricsContext_GetThroughputBreakdown_Begin_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_GetThroughputBreakdown_End_Fn)(LWPA_MetricsContext* pMetricsContext);
typedef LWPA_Status (*LWPW_MetricsContext_GetThroughputBreakdown_End_Fn)(LWPW_MetricsContext_GetThroughputBreakdown_End_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_GetMetricProperties_Begin_Fn)(LWPA_MetricsContext* pMetricsContext, const char* pMetricName, LWPA_MetricProperties* pMetricProperties);
typedef LWPA_Status (*LWPW_MetricsContext_GetMetricProperties_Begin_Fn)(LWPW_MetricsContext_GetMetricProperties_Begin_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_GetMetricProperties_End_Fn)(LWPA_MetricsContext* pMetricsContext);
typedef LWPA_Status (*LWPW_MetricsContext_GetMetricProperties_End_Fn)(LWPW_MetricsContext_GetMetricProperties_End_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_SetCounterData_Fn)(LWPA_MetricsContext* pMetricsContext, const uint8_t* pCounterDataImage, size_t rangeIndex, LWPA_Bool isolated);
typedef LWPA_Status (*LWPW_MetricsContext_SetCounterData_Fn)(LWPW_MetricsContext_SetCounterData_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_SetUserData_Fn)(LWPA_MetricsContext* pMetricsContext, const LWPA_MetrilwserData* pMetrilwserData);
typedef LWPA_Status (*LWPW_MetricsContext_SetUserData_Fn)(LWPW_MetricsContext_SetUserData_Params* pParams);
typedef LWPA_Status (*LWPA_MetricsContext_EvaluateToGpuValues_Fn)(LWPA_MetricsContext* pMetricsContext, size_t numMetrics, const char* const* ppMetricNames, double* pMetricValues);
typedef LWPA_Status (*LWPW_MetricsContext_EvaluateToGpuValues_Fn)(LWPW_MetricsContext_EvaluateToGpuValues_Params* pParams);
typedef LWPA_Status (*LWPW_MetricsContext_GetMetricSuffix_Begin_Fn)(LWPW_MetricsContext_GetMetricSuffix_Begin_Params* pParams);
typedef LWPA_Status (*LWPW_MetricsContext_GetMetricSuffix_End_Fn)(LWPW_MetricsContext_GetMetricSuffix_End_Params* pParams);
typedef LWPA_Status (*LWPW_MetricsContext_GetMetricBaseNames_Begin_Fn)(LWPW_MetricsContext_GetMetricBaseNames_Begin_Params* pParams);
typedef LWPA_Status (*LWPW_MetricsContext_GetMetricBaseNames_End_Fn)(LWPW_MetricsContext_GetMetricBaseNames_End_Params* pParams);
typedef LWPA_Status (*LWPA_InitializeTarget_Fn)(void);
typedef LWPA_Status (*LWPA_GetDeviceCount_Fn)(size_t* pNumDevices);
typedef LWPA_Status (*LWPA_Device_GetNames_Fn)(size_t deviceIndex, const char** ppDeviceName, const char** ppChipName);
typedef LWPA_Status (*LWPA_CounterData_GetNumRanges_Fn)(const uint8_t* pCounterDataImage, size_t* pNumRanges);
typedef LWPA_Status (*LWPA_CounterData_GetRangeDescriptions_Fn)(const uint8_t* pCounterDataImage, size_t rangeIndex, size_t numDescriptions, const char** ppDescriptions, size_t* pNumDescriptions);
typedef LWPA_Status (*LWPW_InitializeTarget_Fn)(LWPW_InitializeTarget_Params* pParams);
typedef LWPA_Status (*LWPW_GetDeviceCount_Fn)(LWPW_GetDeviceCount_Params* pParams);
typedef LWPA_Status (*LWPW_Device_GetNames_Fn)(LWPW_Device_GetNames_Params* pParams);
typedef LWPA_Status (*LWPW_Device_GetPciBusIds_Fn)(LWPW_Device_GetPciBusIds_Params* pParams);
typedef LWPA_Status (*LWPW_Adapter_GetDeviceIndex_Fn)(LWPW_Adapter_GetDeviceIndex_Params* pParams);
typedef LWPA_Status (*LWPW_CounterData_GetNumRanges_Fn)(LWPW_CounterData_GetNumRanges_Params* pParams);
typedef LWPA_Status (*LWPW_Config_GetNumPasses_Fn)(LWPW_Config_GetNumPasses_Params* pParams);
typedef LWPA_Status (*LWPW_CounterData_GetRangeDescriptions_Fn)(LWPW_CounterData_GetRangeDescriptions_Params* pParams);
typedef LWPA_Status (*LWPW_Profiler_CounterData_GetRangeDescriptions_Fn)(LWPW_Profiler_CounterData_GetRangeDescriptions_Params* pParams);
typedef LWPA_Status (*LWPW_PeriodicSampler_CounterData_GetDelimiters_Fn)(LWPW_PeriodicSampler_CounterData_GetDelimiters_Params* pParams);
typedef LWPA_Status (*LWPW_PeriodicSampler_CounterData_GetSampleTime_Fn)(LWPW_PeriodicSampler_CounterData_GetSampleTime_Params* pParams);
typedef LWPA_Status (*LWPW_PeriodicSampler_CounterData_TrimInPlace_Fn)(LWPW_PeriodicSampler_CounterData_TrimInPlace_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_MetricsContext_Create_Fn)(LWPW_LWDA_MetricsContext_Create_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_RawMetricsConfig_Create_Fn)(LWPW_LWDA_RawMetricsConfig_Create_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_CounterDataImage_CallwlateSize_Fn)(LWPW_LWDA_Profiler_CounterDataImage_CallwlateSize_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_CounterDataImage_Initialize_Fn)(LWPW_LWDA_Profiler_CounterDataImage_Initialize_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_CounterDataImage_CallwlateScratchBufferSize_Fn)(LWPW_LWDA_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_CounterDataImage_InitializeScratchBuffer_Fn)(LWPW_LWDA_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_GetDeviceOrdinals_Fn)(LWPW_LWDA_GetDeviceOrdinals_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_LoadDriver_Fn)(LWPW_LWDA_LoadDriver_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_CalcTraceBufferSize_Fn)(LWPW_LWDA_Profiler_CalcTraceBufferSize_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_BeginSession_Fn)(LWPW_LWDA_Profiler_BeginSession_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_EndSession_Fn)(LWPW_LWDA_Profiler_EndSession_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_SetConfig_Fn)(LWPW_LWDA_Profiler_SetConfig_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_ClearConfig_Fn)(LWPW_LWDA_Profiler_ClearConfig_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_BeginPass_Fn)(LWPW_LWDA_Profiler_BeginPass_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_EndPass_Fn)(LWPW_LWDA_Profiler_EndPass_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_DecodeCounters_Fn)(LWPW_LWDA_Profiler_DecodeCounters_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_EnablePerLaunchProfiling_Fn)(LWPW_LWDA_Profiler_EnablePerLaunchProfiling_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_DisablePerLaunchProfiling_Fn)(LWPW_LWDA_Profiler_DisablePerLaunchProfiling_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_PushRange_Fn)(LWPW_LWDA_Profiler_PushRange_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_Profiler_PopRange_Fn)(LWPW_LWDA_Profiler_PopRange_Params* pParams);
typedef LWPA_Status (*LWPW_VK_MetricsContext_Create_Fn)(LWPW_VK_MetricsContext_Create_Params* pParams);
typedef LWPA_Status (*LWPW_VK_RawMetricsConfig_Create_Fn)(LWPW_VK_RawMetricsConfig_Create_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Fn)(LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_CounterDataImage_Initialize_Fn)(LWPW_VK_Profiler_CounterDataImage_Initialize_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Fn)(LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Fn)(LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams);
typedef LWPA_Status (*LWPW_VK_LoadDriver_Fn)(LWPW_VK_LoadDriver_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Device_GetDeviceIndex_Fn)(LWPW_VK_Device_GetDeviceIndex_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_GetRequiredInstanceExtensions_Fn)(LWPW_VK_Profiler_GetRequiredInstanceExtensions_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_GetRequiredDeviceExtensions_Fn)(LWPW_VK_Profiler_GetRequiredDeviceExtensions_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_CalcTraceBufferSize_Fn)(LWPW_VK_Profiler_CalcTraceBufferSize_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_Queue_BeginSession_Fn)(LWPW_VK_Profiler_Queue_BeginSession_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_Queue_EndSession_Fn)(LWPW_VK_Profiler_Queue_EndSession_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Queue_ServicePendingGpuOperations_Fn)(LWPW_VK_Queue_ServicePendingGpuOperations_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_Queue_SetConfig_Fn)(LWPW_VK_Profiler_Queue_SetConfig_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_Queue_ClearConfig_Fn)(LWPW_VK_Profiler_Queue_ClearConfig_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_Queue_BeginPass_Fn)(LWPW_VK_Profiler_Queue_BeginPass_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_Queue_EndPass_Fn)(LWPW_VK_Profiler_Queue_EndPass_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_CommandBuffer_PushRange_Fn)(LWPW_VK_Profiler_CommandBuffer_PushRange_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_CommandBuffer_PopRange_Fn)(LWPW_VK_Profiler_CommandBuffer_PopRange_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_Queue_DecodeCounters_Fn)(LWPW_VK_Profiler_Queue_DecodeCounters_Params* pParams);
typedef LWPA_Status (*LWPW_VK_Profiler_IsGpuSupported_Fn)(LWPW_VK_Profiler_IsGpuSupported_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_CallwlateMemoryOverhead_Fn)(LWPW_VK_PeriodicSampler_CallwlateMemoryOverhead_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_Device_CallwlateMemoryOverhead_Fn)(LWPW_VK_PeriodicSampler_Device_CallwlateMemoryOverhead_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_Queue_BeginSession_Fn)(LWPW_VK_PeriodicSampler_Queue_BeginSession_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_Queue_EndSession_Fn)(LWPW_VK_PeriodicSampler_Queue_EndSession_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_CommandBuffer_StartSampling_Fn)(LWPW_VK_PeriodicSampler_CommandBuffer_StartSampling_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_CommandBuffer_StopSampling_Fn)(LWPW_VK_PeriodicSampler_CommandBuffer_StopSampling_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_CommandBuffer_InsertDelimiter_Fn)(LWPW_VK_PeriodicSampler_CommandBuffer_InsertDelimiter_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_CommandBuffer_BeginFrame_Fn)(LWPW_VK_PeriodicSampler_CommandBuffer_BeginFrame_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_CommandBuffer_InsertTrigger_Fn)(LWPW_VK_PeriodicSampler_CommandBuffer_InsertTrigger_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_Queue_GetLastError_Fn)(LWPW_VK_PeriodicSampler_Queue_GetLastError_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_CounterDataImage_CallwlateSize_Fn)(LWPW_VK_PeriodicSampler_CounterDataImage_CallwlateSize_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_CounterDataImage_Initialize_Fn)(LWPW_VK_PeriodicSampler_CounterDataImage_Initialize_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_Queue_DecodeCounters_Fn)(LWPW_VK_PeriodicSampler_Queue_DecodeCounters_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_IsGpuSupported_Fn)(LWPW_VK_PeriodicSampler_IsGpuSupported_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_Queue_DiscardFrame_Fn)(LWPW_VK_PeriodicSampler_Queue_DiscardFrame_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_Queue_CallwlateRecordBufferSize_Fn)(LWPW_VK_PeriodicSampler_Queue_CallwlateRecordBufferSize_Params* pParams);
typedef LWPA_Status (*LWPW_VK_PeriodicSampler_Queue_SetConfig_Fn)(LWPW_VK_PeriodicSampler_Queue_SetConfig_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_MetricsContext_Create_Fn)(LWPW_OpenGL_MetricsContext_Create_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_RawMetricsConfig_Create_Fn)(LWPW_OpenGL_RawMetricsConfig_Create_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_LoadDriver_Fn)(LWPW_OpenGL_LoadDriver_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_GetLwrrentGraphicsContext_Fn)(LWPW_OpenGL_GetLwrrentGraphicsContext_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_GraphicsContext_GetDeviceIndex_Fn)(LWPW_OpenGL_GraphicsContext_GetDeviceIndex_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_Profiler_IsGpuSupported_Fn)(LWPW_OpenGL_Profiler_IsGpuSupported_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_Profiler_CounterDataImage_CallwlateSize_Fn)(LWPW_OpenGL_Profiler_CounterDataImage_CallwlateSize_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_Profiler_CounterDataImage_Initialize_Fn)(LWPW_OpenGL_Profiler_CounterDataImage_Initialize_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_Profiler_CounterDataImage_CallwlateScratchBufferSize_Fn)(LWPW_OpenGL_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_Profiler_CounterDataImage_InitializeScratchBuffer_Fn)(LWPW_OpenGL_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_Profiler_CalcTraceBufferSize_Fn)(LWPW_OpenGL_Profiler_CalcTraceBufferSize_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_Profiler_GraphicsContext_BeginSession_Fn)(LWPW_OpenGL_Profiler_GraphicsContext_BeginSession_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_Profiler_GraphicsContext_EndSession_Fn)(LWPW_OpenGL_Profiler_GraphicsContext_EndSession_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_Profiler_GraphicsContext_SetConfig_Fn)(LWPW_OpenGL_Profiler_GraphicsContext_SetConfig_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_Profiler_GraphicsContext_ClearConfig_Fn)(LWPW_OpenGL_Profiler_GraphicsContext_ClearConfig_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_Profiler_GraphicsContext_BeginPass_Fn)(LWPW_OpenGL_Profiler_GraphicsContext_BeginPass_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_Profiler_GraphicsContext_EndPass_Fn)(LWPW_OpenGL_Profiler_GraphicsContext_EndPass_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_Profiler_GraphicsContext_PushRange_Fn)(LWPW_OpenGL_Profiler_GraphicsContext_PushRange_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_Profiler_GraphicsContext_PopRange_Fn)(LWPW_OpenGL_Profiler_GraphicsContext_PopRange_Params* pParams);
typedef LWPA_Status (*LWPW_OpenGL_Profiler_GraphicsContext_DecodeCounters_Fn)(LWPW_OpenGL_Profiler_GraphicsContext_DecodeCounters_Params* pParams);

// Default implementations
static LWPA_Status g_defaultStatus = LWPA_STATUS_NOT_LOADED;

static LWPA_GenericFn LWPA_GetProcAddress_Default(const char* pFunctionName)
{
    (void)pFunctionName;
    return NULL;
}
static LWPA_GenericFn LWPA_GetProcAddress_Default(const char* pFunctionName);
static LWPA_Status LWPW_SetLibraryLoadPaths_Default(LWPW_SetLibraryLoadPaths_Params* pParams);
static LWPA_Status LWPW_SetLibraryLoadPathsW_Default(LWPW_SetLibraryLoadPathsW_Params* pParams);
static LWPA_Status LWPA_InitializeHost_Default(void);
static LWPA_Status LWPW_InitializeHost_Default(LWPW_InitializeHost_Params* pParams);
static LWPA_Status LWPA_CounterData_CallwlateCounterDataImageCopySize_Default(const LWPA_CounterDataImageCopyOptions* pCounterDataImageCopyOptions, const uint8_t* pCounterDataSrc, size_t* pCopyDataImageCounterSize)
{
    (void)pCounterDataImageCopyOptions;
    (void)pCounterDataSrc;
    (void)pCopyDataImageCounterSize;
    return g_defaultStatus;
}
static LWPA_Status LWPW_CounterData_CallwlateCounterDataImageCopySize_Default(LWPW_CounterData_CallwlateCounterDataImageCopySize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_CounterData_InitializeCounterDataImageCopy_Default(const LWPA_CounterDataImageCopyOptions* pCounterDataImageCopyOptions, const uint8_t* pCounterDataSrc, uint8_t* pCounterDataDst)
{
    (void)pCounterDataImageCopyOptions;
    (void)pCounterDataSrc;
    (void)pCounterDataDst;
    return g_defaultStatus;
}
static LWPA_Status LWPW_CounterData_InitializeCounterDataImageCopy_Default(LWPW_CounterData_InitializeCounterDataImageCopy_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_CounterDataCombiner_Create_Default(const LWPA_CounterDataCombinerOptions* pCounterDataCombinerOptions, LWPA_CounterDataCombiner** ppCounterDataCombiner)
{
    (void)pCounterDataCombinerOptions;
    (void)ppCounterDataCombiner;
    return g_defaultStatus;
}
static LWPA_Status LWPW_CounterDataCombiner_Create_Default(LWPW_CounterDataCombiner_Create_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_CounterDataCombiner_Destroy_Default(LWPA_CounterDataCombiner* pCounterDataCombiner)
{
    (void)pCounterDataCombiner;
    return g_defaultStatus;
}
static LWPA_Status LWPW_CounterDataCombiner_Destroy_Default(LWPW_CounterDataCombiner_Destroy_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_CounterDataCombiner_CreateRange_Default(LWPA_CounterDataCombiner* pCounterDataCombiner, size_t numDescriptions, const char* const* ppDescriptions, size_t* pRangeIndexDst)
{
    (void)pCounterDataCombiner;
    (void)numDescriptions;
    (void)ppDescriptions;
    (void)pRangeIndexDst;
    return g_defaultStatus;
}
static LWPA_Status LWPW_CounterDataCombiner_CreateRange_Default(LWPW_CounterDataCombiner_CreateRange_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_CounterDataCombiner_AclwmulateIntoRange_Default(LWPA_CounterDataCombiner* pCounterDataCombiner, size_t rangeIndexDst, uint32_t dstMultiplier, const uint8_t* pCounterDataSrc, size_t rangeIndexSrc, uint32_t srcMultiplier)
{
    (void)pCounterDataCombiner;
    (void)rangeIndexDst;
    (void)dstMultiplier;
    (void)pCounterDataSrc;
    (void)rangeIndexSrc;
    (void)srcMultiplier;
    return g_defaultStatus;
}
static LWPA_Status LWPW_CounterDataCombiner_AclwmulateIntoRange_Default(LWPW_CounterDataCombiner_AclwmulateIntoRange_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_CounterDataCombiner_SumIntoRange_Default(LWPW_CounterDataCombiner_SumIntoRange_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_CounterDataCombiner_WeightedSumIntoRange_Default(LWPW_CounterDataCombiner_WeightedSumIntoRange_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_GetSupportedChipNames_Default(LWPA_SupportedChipNames* pSupportedChipNames)
{
    (void)pSupportedChipNames;
    return g_defaultStatus;
}
static LWPA_Status LWPW_GetSupportedChipNames_Default(LWPW_GetSupportedChipNames_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_RawMetricsConfig_Create_Default(const LWPA_RawMetricsConfigOptions* pMetricsConfigOptions, LWPA_RawMetricsConfig** ppRawMetricsConfig)
{
    (void)pMetricsConfigOptions;
    (void)ppRawMetricsConfig;
    return g_defaultStatus;
}
static LWPA_Status LWPA_RawMetricsConfig_Destroy_Default(LWPA_RawMetricsConfig* pRawMetricsConfig)
{
    (void)pRawMetricsConfig;
    return g_defaultStatus;
}
static LWPA_Status LWPW_RawMetricsConfig_Destroy_Default(LWPW_RawMetricsConfig_Destroy_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_RawMetricsConfig_BeginPassGroup_Default(LWPA_RawMetricsConfig* pRawMetricsConfig, const LWPA_RawMetricsPassGroupOptions* pRawMetricsPassGroupOptions)
{
    (void)pRawMetricsConfig;
    (void)pRawMetricsPassGroupOptions;
    return g_defaultStatus;
}
static LWPA_Status LWPW_RawMetricsConfig_BeginPassGroup_Default(LWPW_RawMetricsConfig_BeginPassGroup_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_RawMetricsConfig_EndPassGroup_Default(LWPA_RawMetricsConfig* pRawMetricsConfig)
{
    (void)pRawMetricsConfig;
    return g_defaultStatus;
}
static LWPA_Status LWPW_RawMetricsConfig_EndPassGroup_Default(LWPW_RawMetricsConfig_EndPassGroup_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_RawMetricsConfig_GetNumMetrics_Default(const LWPA_RawMetricsConfig* pRawMetricsConfig, size_t* pNumMetrics)
{
    (void)pRawMetricsConfig;
    (void)pNumMetrics;
    return g_defaultStatus;
}
static LWPA_Status LWPW_RawMetricsConfig_GetNumMetrics_Default(LWPW_RawMetricsConfig_GetNumMetrics_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_RawMetricsConfig_GetMetricProperties_Default(const LWPA_RawMetricsConfig* pRawMetricsConfig, size_t metricIndex, LWPA_RawMetricProperties* pRawMetricProperties)
{
    (void)pRawMetricsConfig;
    (void)metricIndex;
    (void)pRawMetricProperties;
    return g_defaultStatus;
}
static LWPA_Status LWPW_RawMetricsConfig_GetMetricProperties_Default(LWPW_RawMetricsConfig_GetMetricProperties_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_RawMetricsConfig_AddMetrics_Default(LWPA_RawMetricsConfig* pRawMetricsConfig, const LWPA_RawMetricRequest* pRawMetricRequests, size_t numMetricRequests)
{
    (void)pRawMetricsConfig;
    (void)pRawMetricRequests;
    (void)numMetricRequests;
    return g_defaultStatus;
}
static LWPA_Status LWPW_RawMetricsConfig_AddMetrics_Default(LWPW_RawMetricsConfig_AddMetrics_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_RawMetricsConfig_IsAddMetricsPossible_Default(const LWPA_RawMetricsConfig* pRawMetricsConfig, const LWPA_RawMetricRequest* pRawMetricRequests, size_t numMetricRequests, LWPA_Bool* pIsPossible)
{
    (void)pRawMetricsConfig;
    (void)pRawMetricRequests;
    (void)numMetricRequests;
    (void)pIsPossible;
    return g_defaultStatus;
}
static LWPA_Status LWPW_RawMetricsConfig_IsAddMetricsPossible_Default(LWPW_RawMetricsConfig_IsAddMetricsPossible_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_RawMetricsConfig_GenerateConfigImage_Default(LWPA_RawMetricsConfig* pRawMetricsConfig)
{
    (void)pRawMetricsConfig;
    return g_defaultStatus;
}
static LWPA_Status LWPW_RawMetricsConfig_GenerateConfigImage_Default(LWPW_RawMetricsConfig_GenerateConfigImage_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_RawMetricsConfig_GetConfigImage_Default(const LWPA_RawMetricsConfig* pRawMetricsConfig, size_t bufferSize, uint8_t* pBuffer, size_t* pBufferSize)
{
    (void)pRawMetricsConfig;
    (void)bufferSize;
    (void)pBuffer;
    (void)pBufferSize;
    return g_defaultStatus;
}
static LWPA_Status LWPW_RawMetricsConfig_GetConfigImage_Default(LWPW_RawMetricsConfig_GetConfigImage_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_RawMetricsConfig_GetNumPasses_Default(const LWPA_RawMetricsConfig* pRawMetricsConfig, size_t* pNumPipelinedPasses, size_t* pNumIsolatedPasses)
{
    (void)pRawMetricsConfig;
    (void)pNumPipelinedPasses;
    (void)pNumIsolatedPasses;
    return g_defaultStatus;
}
static LWPA_Status LWPW_RawMetricsConfig_GetNumPasses_Default(LWPW_RawMetricsConfig_GetNumPasses_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_CounterDataBuilder_Create_Default(const LWPA_CounterDataBuilderOptions* pOptions, LWPA_CounterDataBuilder** ppCounterDataBuilder)
{
    (void)pOptions;
    (void)ppCounterDataBuilder;
    return g_defaultStatus;
}
static LWPA_Status LWPW_CounterDataBuilder_Create_Default(LWPW_CounterDataBuilder_Create_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_CounterDataBuilder_Destroy_Default(LWPA_CounterDataBuilder* pCounterDataBuilder)
{
    (void)pCounterDataBuilder;
    return g_defaultStatus;
}
static LWPA_Status LWPW_CounterDataBuilder_Destroy_Default(LWPW_CounterDataBuilder_Destroy_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_CounterDataBuilder_AddMetrics_Default(LWPA_CounterDataBuilder* pCounterDataBuilder, const LWPA_RawMetricRequest* pRawMetricRequests, size_t numMetricRequests)
{
    (void)pCounterDataBuilder;
    (void)pRawMetricRequests;
    (void)numMetricRequests;
    return g_defaultStatus;
}
static LWPA_Status LWPW_CounterDataBuilder_AddMetrics_Default(LWPW_CounterDataBuilder_AddMetrics_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_CounterDataBuilder_GetCounterDataPrefix_Default(LWPA_CounterDataBuilder* pCounterDataBuilder, size_t bufferSize, uint8_t* pBuffer, size_t* pBufferSize)
{
    (void)pCounterDataBuilder;
    (void)bufferSize;
    (void)pBuffer;
    (void)pBufferSize;
    return g_defaultStatus;
}
static LWPA_Status LWPW_CounterDataBuilder_GetCounterDataPrefix_Default(LWPW_CounterDataBuilder_GetCounterDataPrefix_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_Create_Default(const LWPA_MetricsContextOptions* pMetricsContextOptions, LWPA_MetricsContext** ppMetricsContext)
{
    (void)pMetricsContextOptions;
    (void)ppMetricsContext;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_Destroy_Default(LWPA_MetricsContext* pMetricsContext)
{
    (void)pMetricsContext;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_Destroy_Default(LWPW_MetricsContext_Destroy_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_RunScript_Default(LWPA_MetricsContext* pMetricsContext, const LWPA_MetricsScriptOptions* pOptions)
{
    (void)pMetricsContext;
    (void)pOptions;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_RunScript_Default(LWPW_MetricsContext_RunScript_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_ExecScript_Begin_Default(LWPA_MetricsContext* pMetricsContext, LWPA_MetricsExecOptions* pOptions)
{
    (void)pMetricsContext;
    (void)pOptions;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_ExecScript_Begin_Default(LWPW_MetricsContext_ExecScript_Begin_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_ExecScript_End_Default(LWPA_MetricsContext* pMetricsContext)
{
    (void)pMetricsContext;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_ExecScript_End_Default(LWPW_MetricsContext_ExecScript_End_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_GetCounterNames_Begin_Default(LWPA_MetricsContext* pMetricsContext, size_t* pNumCounters, const char* const** pppCounterNames)
{
    (void)pMetricsContext;
    (void)pNumCounters;
    (void)pppCounterNames;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetCounterNames_Begin_Default(LWPW_MetricsContext_GetCounterNames_Begin_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_GetCounterNames_End_Default(LWPA_MetricsContext* pMetricsContext)
{
    (void)pMetricsContext;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetCounterNames_End_Default(LWPW_MetricsContext_GetCounterNames_End_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_GetThroughputNames_Begin_Default(LWPA_MetricsContext* pMetricsContext, size_t* pNumThroughputs, const char* const** pppThroughputName)
{
    (void)pMetricsContext;
    (void)pNumThroughputs;
    (void)pppThroughputName;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetThroughputNames_Begin_Default(LWPW_MetricsContext_GetThroughputNames_Begin_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_GetThroughputNames_End_Default(LWPA_MetricsContext* pMetricsContext)
{
    (void)pMetricsContext;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetThroughputNames_End_Default(LWPW_MetricsContext_GetThroughputNames_End_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetRatioNames_Begin_Default(LWPW_MetricsContext_GetRatioNames_Begin_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetRatioNames_End_Default(LWPW_MetricsContext_GetRatioNames_End_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_GetMetricNames_Begin_Default(LWPA_MetricsContext* pMetricsContext, LWPA_MetricsEnumerationOptions* pOptions)
{
    (void)pMetricsContext;
    (void)pOptions;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetMetricNames_Begin_Default(LWPW_MetricsContext_GetMetricNames_Begin_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_GetMetricNames_End_Default(LWPA_MetricsContext* pMetricsContext)
{
    (void)pMetricsContext;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetMetricNames_End_Default(LWPW_MetricsContext_GetMetricNames_End_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_GetThroughputBreakdown_Begin_Default(LWPA_MetricsContext* pMetricsContext, const char* pThroughputName, const char* const** pppCounterNames, const char* const** pppSubThroughputNames)
{
    (void)pMetricsContext;
    (void)pThroughputName;
    (void)pppCounterNames;
    (void)pppSubThroughputNames;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetThroughputBreakdown_Begin_Default(LWPW_MetricsContext_GetThroughputBreakdown_Begin_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_GetThroughputBreakdown_End_Default(LWPA_MetricsContext* pMetricsContext)
{
    (void)pMetricsContext;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetThroughputBreakdown_End_Default(LWPW_MetricsContext_GetThroughputBreakdown_End_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_GetMetricProperties_Begin_Default(LWPA_MetricsContext* pMetricsContext, const char* pMetricName, LWPA_MetricProperties* pMetricProperties)
{
    (void)pMetricsContext;
    (void)pMetricName;
    (void)pMetricProperties;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetMetricProperties_Begin_Default(LWPW_MetricsContext_GetMetricProperties_Begin_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_GetMetricProperties_End_Default(LWPA_MetricsContext* pMetricsContext)
{
    (void)pMetricsContext;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetMetricProperties_End_Default(LWPW_MetricsContext_GetMetricProperties_End_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_SetCounterData_Default(LWPA_MetricsContext* pMetricsContext, const uint8_t* pCounterDataImage, size_t rangeIndex, LWPA_Bool isolated)
{
    (void)pMetricsContext;
    (void)pCounterDataImage;
    (void)rangeIndex;
    (void)isolated;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_SetCounterData_Default(LWPW_MetricsContext_SetCounterData_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_SetUserData_Default(LWPA_MetricsContext* pMetricsContext, const LWPA_MetrilwserData* pMetrilwserData)
{
    (void)pMetricsContext;
    (void)pMetrilwserData;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_SetUserData_Default(LWPW_MetricsContext_SetUserData_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_MetricsContext_EvaluateToGpuValues_Default(LWPA_MetricsContext* pMetricsContext, size_t numMetrics, const char* const* ppMetricNames, double* pMetricValues)
{
    (void)pMetricsContext;
    (void)numMetrics;
    (void)ppMetricNames;
    (void)pMetricValues;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_EvaluateToGpuValues_Default(LWPW_MetricsContext_EvaluateToGpuValues_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetMetricSuffix_Begin_Default(LWPW_MetricsContext_GetMetricSuffix_Begin_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetMetricSuffix_End_Default(LWPW_MetricsContext_GetMetricSuffix_End_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetMetricBaseNames_Begin_Default(LWPW_MetricsContext_GetMetricBaseNames_Begin_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_MetricsContext_GetMetricBaseNames_End_Default(LWPW_MetricsContext_GetMetricBaseNames_End_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPA_InitializeTarget_Default(void);
static LWPA_Status LWPA_GetDeviceCount_Default(size_t* pNumDevices)
{
    (void)pNumDevices;
    return g_defaultStatus;
}
static LWPA_Status LWPA_Device_GetNames_Default(size_t deviceIndex, const char** ppDeviceName, const char** ppChipName)
{
    (void)deviceIndex;
    (void)ppDeviceName;
    (void)ppChipName;
    return g_defaultStatus;
}
static LWPA_Status LWPA_CounterData_GetNumRanges_Default(const uint8_t* pCounterDataImage, size_t* pNumRanges)
{
    (void)pCounterDataImage;
    (void)pNumRanges;
    return g_defaultStatus;
}
static LWPA_Status LWPA_CounterData_GetRangeDescriptions_Default(const uint8_t* pCounterDataImage, size_t rangeIndex, size_t numDescriptions, const char** ppDescriptions, size_t* pNumDescriptions)
{
    (void)pCounterDataImage;
    (void)rangeIndex;
    (void)numDescriptions;
    (void)ppDescriptions;
    (void)pNumDescriptions;
    return g_defaultStatus;
}
static LWPA_Status LWPW_InitializeTarget_Default(LWPW_InitializeTarget_Params* pParams);
static LWPA_Status LWPW_GetDeviceCount_Default(LWPW_GetDeviceCount_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_Device_GetNames_Default(LWPW_Device_GetNames_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_Device_GetPciBusIds_Default(LWPW_Device_GetPciBusIds_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_Adapter_GetDeviceIndex_Default(LWPW_Adapter_GetDeviceIndex_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_CounterData_GetNumRanges_Default(LWPW_CounterData_GetNumRanges_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_Config_GetNumPasses_Default(LWPW_Config_GetNumPasses_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_CounterData_GetRangeDescriptions_Default(LWPW_CounterData_GetRangeDescriptions_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_Profiler_CounterData_GetRangeDescriptions_Default(LWPW_Profiler_CounterData_GetRangeDescriptions_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_PeriodicSampler_CounterData_GetDelimiters_Default(LWPW_PeriodicSampler_CounterData_GetDelimiters_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_PeriodicSampler_CounterData_GetSampleTime_Default(LWPW_PeriodicSampler_CounterData_GetSampleTime_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_PeriodicSampler_CounterData_TrimInPlace_Default(LWPW_PeriodicSampler_CounterData_TrimInPlace_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_MetricsContext_Create_Default(LWPW_LWDA_MetricsContext_Create_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_RawMetricsConfig_Create_Default(LWPW_LWDA_RawMetricsConfig_Create_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_CounterDataImage_CallwlateSize_Default(LWPW_LWDA_Profiler_CounterDataImage_CallwlateSize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_CounterDataImage_Initialize_Default(LWPW_LWDA_Profiler_CounterDataImage_Initialize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_CounterDataImage_CallwlateScratchBufferSize_Default(LWPW_LWDA_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_CounterDataImage_InitializeScratchBuffer_Default(LWPW_LWDA_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_GetDeviceOrdinals_Default(LWPW_LWDA_GetDeviceOrdinals_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_LoadDriver_Default(LWPW_LWDA_LoadDriver_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_CalcTraceBufferSize_Default(LWPW_LWDA_Profiler_CalcTraceBufferSize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_BeginSession_Default(LWPW_LWDA_Profiler_BeginSession_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_EndSession_Default(LWPW_LWDA_Profiler_EndSession_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_SetConfig_Default(LWPW_LWDA_Profiler_SetConfig_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_ClearConfig_Default(LWPW_LWDA_Profiler_ClearConfig_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_BeginPass_Default(LWPW_LWDA_Profiler_BeginPass_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_EndPass_Default(LWPW_LWDA_Profiler_EndPass_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_DecodeCounters_Default(LWPW_LWDA_Profiler_DecodeCounters_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_EnablePerLaunchProfiling_Default(LWPW_LWDA_Profiler_EnablePerLaunchProfiling_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_DisablePerLaunchProfiling_Default(LWPW_LWDA_Profiler_DisablePerLaunchProfiling_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_PushRange_Default(LWPW_LWDA_Profiler_PushRange_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_Profiler_PopRange_Default(LWPW_LWDA_Profiler_PopRange_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_MetricsContext_Create_Default(LWPW_VK_MetricsContext_Create_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_RawMetricsConfig_Create_Default(LWPW_VK_RawMetricsConfig_Create_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Default(LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_CounterDataImage_Initialize_Default(LWPW_VK_Profiler_CounterDataImage_Initialize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Default(LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Default(LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_LoadDriver_Default(LWPW_VK_LoadDriver_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Device_GetDeviceIndex_Default(LWPW_VK_Device_GetDeviceIndex_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_GetRequiredInstanceExtensions_Default(LWPW_VK_Profiler_GetRequiredInstanceExtensions_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_GetRequiredDeviceExtensions_Default(LWPW_VK_Profiler_GetRequiredDeviceExtensions_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_CalcTraceBufferSize_Default(LWPW_VK_Profiler_CalcTraceBufferSize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_Queue_BeginSession_Default(LWPW_VK_Profiler_Queue_BeginSession_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_Queue_EndSession_Default(LWPW_VK_Profiler_Queue_EndSession_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Queue_ServicePendingGpuOperations_Default(LWPW_VK_Queue_ServicePendingGpuOperations_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_Queue_SetConfig_Default(LWPW_VK_Profiler_Queue_SetConfig_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_Queue_ClearConfig_Default(LWPW_VK_Profiler_Queue_ClearConfig_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_Queue_BeginPass_Default(LWPW_VK_Profiler_Queue_BeginPass_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_Queue_EndPass_Default(LWPW_VK_Profiler_Queue_EndPass_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_CommandBuffer_PushRange_Default(LWPW_VK_Profiler_CommandBuffer_PushRange_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_CommandBuffer_PopRange_Default(LWPW_VK_Profiler_CommandBuffer_PopRange_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_Queue_DecodeCounters_Default(LWPW_VK_Profiler_Queue_DecodeCounters_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_Profiler_IsGpuSupported_Default(LWPW_VK_Profiler_IsGpuSupported_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_CallwlateMemoryOverhead_Default(LWPW_VK_PeriodicSampler_CallwlateMemoryOverhead_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_Device_CallwlateMemoryOverhead_Default(LWPW_VK_PeriodicSampler_Device_CallwlateMemoryOverhead_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_Queue_BeginSession_Default(LWPW_VK_PeriodicSampler_Queue_BeginSession_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_Queue_EndSession_Default(LWPW_VK_PeriodicSampler_Queue_EndSession_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_CommandBuffer_StartSampling_Default(LWPW_VK_PeriodicSampler_CommandBuffer_StartSampling_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_CommandBuffer_StopSampling_Default(LWPW_VK_PeriodicSampler_CommandBuffer_StopSampling_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_CommandBuffer_InsertDelimiter_Default(LWPW_VK_PeriodicSampler_CommandBuffer_InsertDelimiter_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_CommandBuffer_BeginFrame_Default(LWPW_VK_PeriodicSampler_CommandBuffer_BeginFrame_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_CommandBuffer_InsertTrigger_Default(LWPW_VK_PeriodicSampler_CommandBuffer_InsertTrigger_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_Queue_GetLastError_Default(LWPW_VK_PeriodicSampler_Queue_GetLastError_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_CounterDataImage_CallwlateSize_Default(LWPW_VK_PeriodicSampler_CounterDataImage_CallwlateSize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_CounterDataImage_Initialize_Default(LWPW_VK_PeriodicSampler_CounterDataImage_Initialize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_Queue_DecodeCounters_Default(LWPW_VK_PeriodicSampler_Queue_DecodeCounters_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_IsGpuSupported_Default(LWPW_VK_PeriodicSampler_IsGpuSupported_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_Queue_DiscardFrame_Default(LWPW_VK_PeriodicSampler_Queue_DiscardFrame_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_Queue_CallwlateRecordBufferSize_Default(LWPW_VK_PeriodicSampler_Queue_CallwlateRecordBufferSize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_VK_PeriodicSampler_Queue_SetConfig_Default(LWPW_VK_PeriodicSampler_Queue_SetConfig_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_MetricsContext_Create_Default(LWPW_OpenGL_MetricsContext_Create_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_RawMetricsConfig_Create_Default(LWPW_OpenGL_RawMetricsConfig_Create_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_LoadDriver_Default(LWPW_OpenGL_LoadDriver_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_GetLwrrentGraphicsContext_Default(LWPW_OpenGL_GetLwrrentGraphicsContext_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_GraphicsContext_GetDeviceIndex_Default(LWPW_OpenGL_GraphicsContext_GetDeviceIndex_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_Profiler_IsGpuSupported_Default(LWPW_OpenGL_Profiler_IsGpuSupported_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_Profiler_CounterDataImage_CallwlateSize_Default(LWPW_OpenGL_Profiler_CounterDataImage_CallwlateSize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_Profiler_CounterDataImage_Initialize_Default(LWPW_OpenGL_Profiler_CounterDataImage_Initialize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_Profiler_CounterDataImage_CallwlateScratchBufferSize_Default(LWPW_OpenGL_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_Profiler_CounterDataImage_InitializeScratchBuffer_Default(LWPW_OpenGL_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_Profiler_CalcTraceBufferSize_Default(LWPW_OpenGL_Profiler_CalcTraceBufferSize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_BeginSession_Default(LWPW_OpenGL_Profiler_GraphicsContext_BeginSession_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_EndSession_Default(LWPW_OpenGL_Profiler_GraphicsContext_EndSession_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_SetConfig_Default(LWPW_OpenGL_Profiler_GraphicsContext_SetConfig_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_ClearConfig_Default(LWPW_OpenGL_Profiler_GraphicsContext_ClearConfig_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_BeginPass_Default(LWPW_OpenGL_Profiler_GraphicsContext_BeginPass_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_EndPass_Default(LWPW_OpenGL_Profiler_GraphicsContext_EndPass_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_PushRange_Default(LWPW_OpenGL_Profiler_GraphicsContext_PushRange_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_PopRange_Default(LWPW_OpenGL_Profiler_GraphicsContext_PopRange_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_DecodeCounters_Default(LWPW_OpenGL_Profiler_GraphicsContext_DecodeCounters_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
typedef struct PerfworksApi {
    LWPA_GetProcAddress_Fn                                       LWPA_GetProcAddress;
    LWPW_SetLibraryLoadPaths_Fn                                  LWPW_SetLibraryLoadPaths;
    LWPW_SetLibraryLoadPathsW_Fn                                 LWPW_SetLibraryLoadPathsW;
    LWPA_InitializeHost_Fn                                       LWPA_InitializeHost;
    LWPW_InitializeHost_Fn                                       LWPW_InitializeHost;
    LWPA_CounterData_CallwlateCounterDataImageCopySize_Fn        LWPA_CounterData_CallwlateCounterDataImageCopySize;
    LWPW_CounterData_CallwlateCounterDataImageCopySize_Fn        LWPW_CounterData_CallwlateCounterDataImageCopySize;
    LWPA_CounterData_InitializeCounterDataImageCopy_Fn           LWPA_CounterData_InitializeCounterDataImageCopy;
    LWPW_CounterData_InitializeCounterDataImageCopy_Fn           LWPW_CounterData_InitializeCounterDataImageCopy;
    LWPA_CounterDataCombiner_Create_Fn                           LWPA_CounterDataCombiner_Create;
    LWPW_CounterDataCombiner_Create_Fn                           LWPW_CounterDataCombiner_Create;
    LWPA_CounterDataCombiner_Destroy_Fn                          LWPA_CounterDataCombiner_Destroy;
    LWPW_CounterDataCombiner_Destroy_Fn                          LWPW_CounterDataCombiner_Destroy;
    LWPA_CounterDataCombiner_CreateRange_Fn                      LWPA_CounterDataCombiner_CreateRange;
    LWPW_CounterDataCombiner_CreateRange_Fn                      LWPW_CounterDataCombiner_CreateRange;
    LWPA_CounterDataCombiner_AclwmulateIntoRange_Fn              LWPA_CounterDataCombiner_AclwmulateIntoRange;
    LWPW_CounterDataCombiner_AclwmulateIntoRange_Fn              LWPW_CounterDataCombiner_AclwmulateIntoRange;
    LWPW_CounterDataCombiner_SumIntoRange_Fn                     LWPW_CounterDataCombiner_SumIntoRange;
    LWPW_CounterDataCombiner_WeightedSumIntoRange_Fn             LWPW_CounterDataCombiner_WeightedSumIntoRange;
    LWPA_GetSupportedChipNames_Fn                                LWPA_GetSupportedChipNames;
    LWPW_GetSupportedChipNames_Fn                                LWPW_GetSupportedChipNames;
    LWPA_RawMetricsConfig_Create_Fn                              LWPA_RawMetricsConfig_Create;
    LWPA_RawMetricsConfig_Destroy_Fn                             LWPA_RawMetricsConfig_Destroy;
    LWPW_RawMetricsConfig_Destroy_Fn                             LWPW_RawMetricsConfig_Destroy;
    LWPA_RawMetricsConfig_BeginPassGroup_Fn                      LWPA_RawMetricsConfig_BeginPassGroup;
    LWPW_RawMetricsConfig_BeginPassGroup_Fn                      LWPW_RawMetricsConfig_BeginPassGroup;
    LWPA_RawMetricsConfig_EndPassGroup_Fn                        LWPA_RawMetricsConfig_EndPassGroup;
    LWPW_RawMetricsConfig_EndPassGroup_Fn                        LWPW_RawMetricsConfig_EndPassGroup;
    LWPA_RawMetricsConfig_GetNumMetrics_Fn                       LWPA_RawMetricsConfig_GetNumMetrics;
    LWPW_RawMetricsConfig_GetNumMetrics_Fn                       LWPW_RawMetricsConfig_GetNumMetrics;
    LWPA_RawMetricsConfig_GetMetricProperties_Fn                 LWPA_RawMetricsConfig_GetMetricProperties;
    LWPW_RawMetricsConfig_GetMetricProperties_Fn                 LWPW_RawMetricsConfig_GetMetricProperties;
    LWPA_RawMetricsConfig_AddMetrics_Fn                          LWPA_RawMetricsConfig_AddMetrics;
    LWPW_RawMetricsConfig_AddMetrics_Fn                          LWPW_RawMetricsConfig_AddMetrics;
    LWPA_RawMetricsConfig_IsAddMetricsPossible_Fn                LWPA_RawMetricsConfig_IsAddMetricsPossible;
    LWPW_RawMetricsConfig_IsAddMetricsPossible_Fn                LWPW_RawMetricsConfig_IsAddMetricsPossible;
    LWPA_RawMetricsConfig_GenerateConfigImage_Fn                 LWPA_RawMetricsConfig_GenerateConfigImage;
    LWPW_RawMetricsConfig_GenerateConfigImage_Fn                 LWPW_RawMetricsConfig_GenerateConfigImage;
    LWPA_RawMetricsConfig_GetConfigImage_Fn                      LWPA_RawMetricsConfig_GetConfigImage;
    LWPW_RawMetricsConfig_GetConfigImage_Fn                      LWPW_RawMetricsConfig_GetConfigImage;
    LWPA_RawMetricsConfig_GetNumPasses_Fn                        LWPA_RawMetricsConfig_GetNumPasses;
    LWPW_RawMetricsConfig_GetNumPasses_Fn                        LWPW_RawMetricsConfig_GetNumPasses;
    LWPA_CounterDataBuilder_Create_Fn                            LWPA_CounterDataBuilder_Create;
    LWPW_CounterDataBuilder_Create_Fn                            LWPW_CounterDataBuilder_Create;
    LWPA_CounterDataBuilder_Destroy_Fn                           LWPA_CounterDataBuilder_Destroy;
    LWPW_CounterDataBuilder_Destroy_Fn                           LWPW_CounterDataBuilder_Destroy;
    LWPA_CounterDataBuilder_AddMetrics_Fn                        LWPA_CounterDataBuilder_AddMetrics;
    LWPW_CounterDataBuilder_AddMetrics_Fn                        LWPW_CounterDataBuilder_AddMetrics;
    LWPA_CounterDataBuilder_GetCounterDataPrefix_Fn              LWPA_CounterDataBuilder_GetCounterDataPrefix;
    LWPW_CounterDataBuilder_GetCounterDataPrefix_Fn              LWPW_CounterDataBuilder_GetCounterDataPrefix;
    LWPA_MetricsContext_Create_Fn                                LWPA_MetricsContext_Create;
    LWPA_MetricsContext_Destroy_Fn                               LWPA_MetricsContext_Destroy;
    LWPW_MetricsContext_Destroy_Fn                               LWPW_MetricsContext_Destroy;
    LWPA_MetricsContext_RunScript_Fn                             LWPA_MetricsContext_RunScript;
    LWPW_MetricsContext_RunScript_Fn                             LWPW_MetricsContext_RunScript;
    LWPA_MetricsContext_ExecScript_Begin_Fn                      LWPA_MetricsContext_ExecScript_Begin;
    LWPW_MetricsContext_ExecScript_Begin_Fn                      LWPW_MetricsContext_ExecScript_Begin;
    LWPA_MetricsContext_ExecScript_End_Fn                        LWPA_MetricsContext_ExecScript_End;
    LWPW_MetricsContext_ExecScript_End_Fn                        LWPW_MetricsContext_ExecScript_End;
    LWPA_MetricsContext_GetCounterNames_Begin_Fn                 LWPA_MetricsContext_GetCounterNames_Begin;
    LWPW_MetricsContext_GetCounterNames_Begin_Fn                 LWPW_MetricsContext_GetCounterNames_Begin;
    LWPA_MetricsContext_GetCounterNames_End_Fn                   LWPA_MetricsContext_GetCounterNames_End;
    LWPW_MetricsContext_GetCounterNames_End_Fn                   LWPW_MetricsContext_GetCounterNames_End;
    LWPA_MetricsContext_GetThroughputNames_Begin_Fn              LWPA_MetricsContext_GetThroughputNames_Begin;
    LWPW_MetricsContext_GetThroughputNames_Begin_Fn              LWPW_MetricsContext_GetThroughputNames_Begin;
    LWPA_MetricsContext_GetThroughputNames_End_Fn                LWPA_MetricsContext_GetThroughputNames_End;
    LWPW_MetricsContext_GetThroughputNames_End_Fn                LWPW_MetricsContext_GetThroughputNames_End;
    LWPW_MetricsContext_GetRatioNames_Begin_Fn                   LWPW_MetricsContext_GetRatioNames_Begin;
    LWPW_MetricsContext_GetRatioNames_End_Fn                     LWPW_MetricsContext_GetRatioNames_End;
    LWPA_MetricsContext_GetMetricNames_Begin_Fn                  LWPA_MetricsContext_GetMetricNames_Begin;
    LWPW_MetricsContext_GetMetricNames_Begin_Fn                  LWPW_MetricsContext_GetMetricNames_Begin;
    LWPA_MetricsContext_GetMetricNames_End_Fn                    LWPA_MetricsContext_GetMetricNames_End;
    LWPW_MetricsContext_GetMetricNames_End_Fn                    LWPW_MetricsContext_GetMetricNames_End;
    LWPA_MetricsContext_GetThroughputBreakdown_Begin_Fn          LWPA_MetricsContext_GetThroughputBreakdown_Begin;
    LWPW_MetricsContext_GetThroughputBreakdown_Begin_Fn          LWPW_MetricsContext_GetThroughputBreakdown_Begin;
    LWPA_MetricsContext_GetThroughputBreakdown_End_Fn            LWPA_MetricsContext_GetThroughputBreakdown_End;
    LWPW_MetricsContext_GetThroughputBreakdown_End_Fn            LWPW_MetricsContext_GetThroughputBreakdown_End;
    LWPA_MetricsContext_GetMetricProperties_Begin_Fn             LWPA_MetricsContext_GetMetricProperties_Begin;
    LWPW_MetricsContext_GetMetricProperties_Begin_Fn             LWPW_MetricsContext_GetMetricProperties_Begin;
    LWPA_MetricsContext_GetMetricProperties_End_Fn               LWPA_MetricsContext_GetMetricProperties_End;
    LWPW_MetricsContext_GetMetricProperties_End_Fn               LWPW_MetricsContext_GetMetricProperties_End;
    LWPA_MetricsContext_SetCounterData_Fn                        LWPA_MetricsContext_SetCounterData;
    LWPW_MetricsContext_SetCounterData_Fn                        LWPW_MetricsContext_SetCounterData;
    LWPA_MetricsContext_SetUserData_Fn                           LWPA_MetricsContext_SetUserData;
    LWPW_MetricsContext_SetUserData_Fn                           LWPW_MetricsContext_SetUserData;
    LWPA_MetricsContext_EvaluateToGpuValues_Fn                   LWPA_MetricsContext_EvaluateToGpuValues;
    LWPW_MetricsContext_EvaluateToGpuValues_Fn                   LWPW_MetricsContext_EvaluateToGpuValues;
    LWPW_MetricsContext_GetMetricSuffix_Begin_Fn                 LWPW_MetricsContext_GetMetricSuffix_Begin;
    LWPW_MetricsContext_GetMetricSuffix_End_Fn                   LWPW_MetricsContext_GetMetricSuffix_End;
    LWPW_MetricsContext_GetMetricBaseNames_Begin_Fn              LWPW_MetricsContext_GetMetricBaseNames_Begin;
    LWPW_MetricsContext_GetMetricBaseNames_End_Fn                LWPW_MetricsContext_GetMetricBaseNames_End;
    LWPA_InitializeTarget_Fn                                     LWPA_InitializeTarget;
    LWPA_GetDeviceCount_Fn                                       LWPA_GetDeviceCount;
    LWPA_Device_GetNames_Fn                                      LWPA_Device_GetNames;
    LWPA_CounterData_GetNumRanges_Fn                             LWPA_CounterData_GetNumRanges;
    LWPA_CounterData_GetRangeDescriptions_Fn                     LWPA_CounterData_GetRangeDescriptions;
    LWPW_InitializeTarget_Fn                                     LWPW_InitializeTarget;
    LWPW_GetDeviceCount_Fn                                       LWPW_GetDeviceCount;
    LWPW_Device_GetNames_Fn                                      LWPW_Device_GetNames;
    LWPW_Device_GetPciBusIds_Fn                                  LWPW_Device_GetPciBusIds;
    LWPW_Adapter_GetDeviceIndex_Fn                               LWPW_Adapter_GetDeviceIndex;
    LWPW_CounterData_GetNumRanges_Fn                             LWPW_CounterData_GetNumRanges;
    LWPW_Config_GetNumPasses_Fn                                  LWPW_Config_GetNumPasses;
    LWPW_CounterData_GetRangeDescriptions_Fn                     LWPW_CounterData_GetRangeDescriptions;
    LWPW_Profiler_CounterData_GetRangeDescriptions_Fn            LWPW_Profiler_CounterData_GetRangeDescriptions;
    LWPW_PeriodicSampler_CounterData_GetDelimiters_Fn            LWPW_PeriodicSampler_CounterData_GetDelimiters;
    LWPW_PeriodicSampler_CounterData_GetSampleTime_Fn            LWPW_PeriodicSampler_CounterData_GetSampleTime;
    LWPW_PeriodicSampler_CounterData_TrimInPlace_Fn              LWPW_PeriodicSampler_CounterData_TrimInPlace;
    LWPW_LWDA_MetricsContext_Create_Fn                           LWPW_LWDA_MetricsContext_Create;
    LWPW_LWDA_RawMetricsConfig_Create_Fn                         LWPW_LWDA_RawMetricsConfig_Create;
    LWPW_LWDA_Profiler_CounterDataImage_CallwlateSize_Fn         LWPW_LWDA_Profiler_CounterDataImage_CallwlateSize;
    LWPW_LWDA_Profiler_CounterDataImage_Initialize_Fn            LWPW_LWDA_Profiler_CounterDataImage_Initialize;
    LWPW_LWDA_Profiler_CounterDataImage_CallwlateScratchBufferSize_Fn LWPW_LWDA_Profiler_CounterDataImage_CallwlateScratchBufferSize;
    LWPW_LWDA_Profiler_CounterDataImage_InitializeScratchBuffer_Fn LWPW_LWDA_Profiler_CounterDataImage_InitializeScratchBuffer;
    LWPW_LWDA_GetDeviceOrdinals_Fn                               LWPW_LWDA_GetDeviceOrdinals;
    LWPW_LWDA_LoadDriver_Fn                                      LWPW_LWDA_LoadDriver;
    LWPW_LWDA_Profiler_CalcTraceBufferSize_Fn                    LWPW_LWDA_Profiler_CalcTraceBufferSize;
    LWPW_LWDA_Profiler_BeginSession_Fn                           LWPW_LWDA_Profiler_BeginSession;
    LWPW_LWDA_Profiler_EndSession_Fn                             LWPW_LWDA_Profiler_EndSession;
    LWPW_LWDA_Profiler_SetConfig_Fn                              LWPW_LWDA_Profiler_SetConfig;
    LWPW_LWDA_Profiler_ClearConfig_Fn                            LWPW_LWDA_Profiler_ClearConfig;
    LWPW_LWDA_Profiler_BeginPass_Fn                              LWPW_LWDA_Profiler_BeginPass;
    LWPW_LWDA_Profiler_EndPass_Fn                                LWPW_LWDA_Profiler_EndPass;
    LWPW_LWDA_Profiler_DecodeCounters_Fn                         LWPW_LWDA_Profiler_DecodeCounters;
    LWPW_LWDA_Profiler_EnablePerLaunchProfiling_Fn               LWPW_LWDA_Profiler_EnablePerLaunchProfiling;
    LWPW_LWDA_Profiler_DisablePerLaunchProfiling_Fn              LWPW_LWDA_Profiler_DisablePerLaunchProfiling;
    LWPW_LWDA_Profiler_PushRange_Fn                              LWPW_LWDA_Profiler_PushRange;
    LWPW_LWDA_Profiler_PopRange_Fn                               LWPW_LWDA_Profiler_PopRange;
    LWPW_VK_MetricsContext_Create_Fn                             LWPW_VK_MetricsContext_Create;
    LWPW_VK_RawMetricsConfig_Create_Fn                           LWPW_VK_RawMetricsConfig_Create;
    LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Fn           LWPW_VK_Profiler_CounterDataImage_CallwlateSize;
    LWPW_VK_Profiler_CounterDataImage_Initialize_Fn              LWPW_VK_Profiler_CounterDataImage_Initialize;
    LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Fn LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize;
    LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Fn LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer;
    LWPW_VK_LoadDriver_Fn                                        LWPW_VK_LoadDriver;
    LWPW_VK_Device_GetDeviceIndex_Fn                             LWPW_VK_Device_GetDeviceIndex;
    LWPW_VK_Profiler_GetRequiredInstanceExtensions_Fn            LWPW_VK_Profiler_GetRequiredInstanceExtensions;
    LWPW_VK_Profiler_GetRequiredDeviceExtensions_Fn              LWPW_VK_Profiler_GetRequiredDeviceExtensions;
    LWPW_VK_Profiler_CalcTraceBufferSize_Fn                      LWPW_VK_Profiler_CalcTraceBufferSize;
    LWPW_VK_Profiler_Queue_BeginSession_Fn                       LWPW_VK_Profiler_Queue_BeginSession;
    LWPW_VK_Profiler_Queue_EndSession_Fn                         LWPW_VK_Profiler_Queue_EndSession;
    LWPW_VK_Queue_ServicePendingGpuOperations_Fn                 LWPW_VK_Queue_ServicePendingGpuOperations;
    LWPW_VK_Profiler_Queue_SetConfig_Fn                          LWPW_VK_Profiler_Queue_SetConfig;
    LWPW_VK_Profiler_Queue_ClearConfig_Fn                        LWPW_VK_Profiler_Queue_ClearConfig;
    LWPW_VK_Profiler_Queue_BeginPass_Fn                          LWPW_VK_Profiler_Queue_BeginPass;
    LWPW_VK_Profiler_Queue_EndPass_Fn                            LWPW_VK_Profiler_Queue_EndPass;
    LWPW_VK_Profiler_CommandBuffer_PushRange_Fn                  LWPW_VK_Profiler_CommandBuffer_PushRange;
    LWPW_VK_Profiler_CommandBuffer_PopRange_Fn                   LWPW_VK_Profiler_CommandBuffer_PopRange;
    LWPW_VK_Profiler_Queue_DecodeCounters_Fn                     LWPW_VK_Profiler_Queue_DecodeCounters;
    LWPW_VK_Profiler_IsGpuSupported_Fn                           LWPW_VK_Profiler_IsGpuSupported;
    LWPW_VK_PeriodicSampler_CallwlateMemoryOverhead_Fn           LWPW_VK_PeriodicSampler_CallwlateMemoryOverhead;
    LWPW_VK_PeriodicSampler_Device_CallwlateMemoryOverhead_Fn    LWPW_VK_PeriodicSampler_Device_CallwlateMemoryOverhead;
    LWPW_VK_PeriodicSampler_Queue_BeginSession_Fn                LWPW_VK_PeriodicSampler_Queue_BeginSession;
    LWPW_VK_PeriodicSampler_Queue_EndSession_Fn                  LWPW_VK_PeriodicSampler_Queue_EndSession;
    LWPW_VK_PeriodicSampler_CommandBuffer_StartSampling_Fn       LWPW_VK_PeriodicSampler_CommandBuffer_StartSampling;
    LWPW_VK_PeriodicSampler_CommandBuffer_StopSampling_Fn        LWPW_VK_PeriodicSampler_CommandBuffer_StopSampling;
    LWPW_VK_PeriodicSampler_CommandBuffer_InsertDelimiter_Fn     LWPW_VK_PeriodicSampler_CommandBuffer_InsertDelimiter;
    LWPW_VK_PeriodicSampler_CommandBuffer_BeginFrame_Fn          LWPW_VK_PeriodicSampler_CommandBuffer_BeginFrame;
    LWPW_VK_PeriodicSampler_CommandBuffer_InsertTrigger_Fn       LWPW_VK_PeriodicSampler_CommandBuffer_InsertTrigger;
    LWPW_VK_PeriodicSampler_Queue_GetLastError_Fn                LWPW_VK_PeriodicSampler_Queue_GetLastError;
    LWPW_VK_PeriodicSampler_CounterDataImage_CallwlateSize_Fn    LWPW_VK_PeriodicSampler_CounterDataImage_CallwlateSize;
    LWPW_VK_PeriodicSampler_CounterDataImage_Initialize_Fn       LWPW_VK_PeriodicSampler_CounterDataImage_Initialize;
    LWPW_VK_PeriodicSampler_Queue_DecodeCounters_Fn              LWPW_VK_PeriodicSampler_Queue_DecodeCounters;
    LWPW_VK_PeriodicSampler_IsGpuSupported_Fn                    LWPW_VK_PeriodicSampler_IsGpuSupported;
    LWPW_VK_PeriodicSampler_Queue_DiscardFrame_Fn                LWPW_VK_PeriodicSampler_Queue_DiscardFrame;
    LWPW_VK_PeriodicSampler_Queue_CallwlateRecordBufferSize_Fn   LWPW_VK_PeriodicSampler_Queue_CallwlateRecordBufferSize;
    LWPW_VK_PeriodicSampler_Queue_SetConfig_Fn                   LWPW_VK_PeriodicSampler_Queue_SetConfig;
    LWPW_OpenGL_MetricsContext_Create_Fn                         LWPW_OpenGL_MetricsContext_Create;
    LWPW_OpenGL_RawMetricsConfig_Create_Fn                       LWPW_OpenGL_RawMetricsConfig_Create;
    LWPW_OpenGL_LoadDriver_Fn                                    LWPW_OpenGL_LoadDriver;
    LWPW_OpenGL_GetLwrrentGraphicsContext_Fn                     LWPW_OpenGL_GetLwrrentGraphicsContext;
    LWPW_OpenGL_GraphicsContext_GetDeviceIndex_Fn                LWPW_OpenGL_GraphicsContext_GetDeviceIndex;
    LWPW_OpenGL_Profiler_IsGpuSupported_Fn                       LWPW_OpenGL_Profiler_IsGpuSupported;
    LWPW_OpenGL_Profiler_CounterDataImage_CallwlateSize_Fn       LWPW_OpenGL_Profiler_CounterDataImage_CallwlateSize;
    LWPW_OpenGL_Profiler_CounterDataImage_Initialize_Fn          LWPW_OpenGL_Profiler_CounterDataImage_Initialize;
    LWPW_OpenGL_Profiler_CounterDataImage_CallwlateScratchBufferSize_Fn LWPW_OpenGL_Profiler_CounterDataImage_CallwlateScratchBufferSize;
    LWPW_OpenGL_Profiler_CounterDataImage_InitializeScratchBuffer_Fn LWPW_OpenGL_Profiler_CounterDataImage_InitializeScratchBuffer;
    LWPW_OpenGL_Profiler_CalcTraceBufferSize_Fn                  LWPW_OpenGL_Profiler_CalcTraceBufferSize;
    LWPW_OpenGL_Profiler_GraphicsContext_BeginSession_Fn         LWPW_OpenGL_Profiler_GraphicsContext_BeginSession;
    LWPW_OpenGL_Profiler_GraphicsContext_EndSession_Fn           LWPW_OpenGL_Profiler_GraphicsContext_EndSession;
    LWPW_OpenGL_Profiler_GraphicsContext_SetConfig_Fn            LWPW_OpenGL_Profiler_GraphicsContext_SetConfig;
    LWPW_OpenGL_Profiler_GraphicsContext_ClearConfig_Fn          LWPW_OpenGL_Profiler_GraphicsContext_ClearConfig;
    LWPW_OpenGL_Profiler_GraphicsContext_BeginPass_Fn            LWPW_OpenGL_Profiler_GraphicsContext_BeginPass;
    LWPW_OpenGL_Profiler_GraphicsContext_EndPass_Fn              LWPW_OpenGL_Profiler_GraphicsContext_EndPass;
    LWPW_OpenGL_Profiler_GraphicsContext_PushRange_Fn            LWPW_OpenGL_Profiler_GraphicsContext_PushRange;
    LWPW_OpenGL_Profiler_GraphicsContext_PopRange_Fn             LWPW_OpenGL_Profiler_GraphicsContext_PopRange;
    LWPW_OpenGL_Profiler_GraphicsContext_DecodeCounters_Fn       LWPW_OpenGL_Profiler_GraphicsContext_DecodeCounters;
} PerfworksApi;

#if _WIN32
typedef wchar_t LWPW_User_PathCharType;
#else
typedef char LWPW_User_PathCharType;
#endif

typedef struct LWPW_User_Api
{
    void* hModPerfworks;
    LWPA_GetProcAddress_Fn perfworksGetProcAddress;
    PerfworksApi fn;
    size_t numSearchPaths;
    LWPW_User_PathCharType** ppSearchPaths;
} LWPW_User_Api;

static LWPW_User_Api g_api = {
      0 /* hModPerfworks */
    , 0 /* perfworksGetProcAddress */
    , {
          &LWPA_GetProcAddress_Default
        , &LWPW_SetLibraryLoadPaths_Default
        , &LWPW_SetLibraryLoadPathsW_Default
        , &LWPA_InitializeHost_Default
        , &LWPW_InitializeHost_Default
        , &LWPA_CounterData_CallwlateCounterDataImageCopySize_Default
        , &LWPW_CounterData_CallwlateCounterDataImageCopySize_Default
        , &LWPA_CounterData_InitializeCounterDataImageCopy_Default
        , &LWPW_CounterData_InitializeCounterDataImageCopy_Default
        , &LWPA_CounterDataCombiner_Create_Default
        , &LWPW_CounterDataCombiner_Create_Default
        , &LWPA_CounterDataCombiner_Destroy_Default
        , &LWPW_CounterDataCombiner_Destroy_Default
        , &LWPA_CounterDataCombiner_CreateRange_Default
        , &LWPW_CounterDataCombiner_CreateRange_Default
        , &LWPA_CounterDataCombiner_AclwmulateIntoRange_Default
        , &LWPW_CounterDataCombiner_AclwmulateIntoRange_Default
        , &LWPW_CounterDataCombiner_SumIntoRange_Default
        , &LWPW_CounterDataCombiner_WeightedSumIntoRange_Default
        , &LWPA_GetSupportedChipNames_Default
        , &LWPW_GetSupportedChipNames_Default
        , &LWPA_RawMetricsConfig_Create_Default
        , &LWPA_RawMetricsConfig_Destroy_Default
        , &LWPW_RawMetricsConfig_Destroy_Default
        , &LWPA_RawMetricsConfig_BeginPassGroup_Default
        , &LWPW_RawMetricsConfig_BeginPassGroup_Default
        , &LWPA_RawMetricsConfig_EndPassGroup_Default
        , &LWPW_RawMetricsConfig_EndPassGroup_Default
        , &LWPA_RawMetricsConfig_GetNumMetrics_Default
        , &LWPW_RawMetricsConfig_GetNumMetrics_Default
        , &LWPA_RawMetricsConfig_GetMetricProperties_Default
        , &LWPW_RawMetricsConfig_GetMetricProperties_Default
        , &LWPA_RawMetricsConfig_AddMetrics_Default
        , &LWPW_RawMetricsConfig_AddMetrics_Default
        , &LWPA_RawMetricsConfig_IsAddMetricsPossible_Default
        , &LWPW_RawMetricsConfig_IsAddMetricsPossible_Default
        , &LWPA_RawMetricsConfig_GenerateConfigImage_Default
        , &LWPW_RawMetricsConfig_GenerateConfigImage_Default
        , &LWPA_RawMetricsConfig_GetConfigImage_Default
        , &LWPW_RawMetricsConfig_GetConfigImage_Default
        , &LWPA_RawMetricsConfig_GetNumPasses_Default
        , &LWPW_RawMetricsConfig_GetNumPasses_Default
        , &LWPA_CounterDataBuilder_Create_Default
        , &LWPW_CounterDataBuilder_Create_Default
        , &LWPA_CounterDataBuilder_Destroy_Default
        , &LWPW_CounterDataBuilder_Destroy_Default
        , &LWPA_CounterDataBuilder_AddMetrics_Default
        , &LWPW_CounterDataBuilder_AddMetrics_Default
        , &LWPA_CounterDataBuilder_GetCounterDataPrefix_Default
        , &LWPW_CounterDataBuilder_GetCounterDataPrefix_Default
        , &LWPA_MetricsContext_Create_Default
        , &LWPA_MetricsContext_Destroy_Default
        , &LWPW_MetricsContext_Destroy_Default
        , &LWPA_MetricsContext_RunScript_Default
        , &LWPW_MetricsContext_RunScript_Default
        , &LWPA_MetricsContext_ExecScript_Begin_Default
        , &LWPW_MetricsContext_ExecScript_Begin_Default
        , &LWPA_MetricsContext_ExecScript_End_Default
        , &LWPW_MetricsContext_ExecScript_End_Default
        , &LWPA_MetricsContext_GetCounterNames_Begin_Default
        , &LWPW_MetricsContext_GetCounterNames_Begin_Default
        , &LWPA_MetricsContext_GetCounterNames_End_Default
        , &LWPW_MetricsContext_GetCounterNames_End_Default
        , &LWPA_MetricsContext_GetThroughputNames_Begin_Default
        , &LWPW_MetricsContext_GetThroughputNames_Begin_Default
        , &LWPA_MetricsContext_GetThroughputNames_End_Default
        , &LWPW_MetricsContext_GetThroughputNames_End_Default
        , &LWPW_MetricsContext_GetRatioNames_Begin_Default
        , &LWPW_MetricsContext_GetRatioNames_End_Default
        , &LWPA_MetricsContext_GetMetricNames_Begin_Default
        , &LWPW_MetricsContext_GetMetricNames_Begin_Default
        , &LWPA_MetricsContext_GetMetricNames_End_Default
        , &LWPW_MetricsContext_GetMetricNames_End_Default
        , &LWPA_MetricsContext_GetThroughputBreakdown_Begin_Default
        , &LWPW_MetricsContext_GetThroughputBreakdown_Begin_Default
        , &LWPA_MetricsContext_GetThroughputBreakdown_End_Default
        , &LWPW_MetricsContext_GetThroughputBreakdown_End_Default
        , &LWPA_MetricsContext_GetMetricProperties_Begin_Default
        , &LWPW_MetricsContext_GetMetricProperties_Begin_Default
        , &LWPA_MetricsContext_GetMetricProperties_End_Default
        , &LWPW_MetricsContext_GetMetricProperties_End_Default
        , &LWPA_MetricsContext_SetCounterData_Default
        , &LWPW_MetricsContext_SetCounterData_Default
        , &LWPA_MetricsContext_SetUserData_Default
        , &LWPW_MetricsContext_SetUserData_Default
        , &LWPA_MetricsContext_EvaluateToGpuValues_Default
        , &LWPW_MetricsContext_EvaluateToGpuValues_Default
        , &LWPW_MetricsContext_GetMetricSuffix_Begin_Default
        , &LWPW_MetricsContext_GetMetricSuffix_End_Default
        , &LWPW_MetricsContext_GetMetricBaseNames_Begin_Default
        , &LWPW_MetricsContext_GetMetricBaseNames_End_Default
        , &LWPA_InitializeTarget_Default
        , &LWPA_GetDeviceCount_Default
        , &LWPA_Device_GetNames_Default
        , &LWPA_CounterData_GetNumRanges_Default
        , &LWPA_CounterData_GetRangeDescriptions_Default
        , &LWPW_InitializeTarget_Default
        , &LWPW_GetDeviceCount_Default
        , &LWPW_Device_GetNames_Default
        , &LWPW_Device_GetPciBusIds_Default
        , &LWPW_Adapter_GetDeviceIndex_Default
        , &LWPW_CounterData_GetNumRanges_Default
        , &LWPW_Config_GetNumPasses_Default
        , &LWPW_CounterData_GetRangeDescriptions_Default
        , &LWPW_Profiler_CounterData_GetRangeDescriptions_Default
        , &LWPW_PeriodicSampler_CounterData_GetDelimiters_Default
        , &LWPW_PeriodicSampler_CounterData_GetSampleTime_Default
        , &LWPW_PeriodicSampler_CounterData_TrimInPlace_Default
        , &LWPW_LWDA_MetricsContext_Create_Default
        , &LWPW_LWDA_RawMetricsConfig_Create_Default
        , &LWPW_LWDA_Profiler_CounterDataImage_CallwlateSize_Default
        , &LWPW_LWDA_Profiler_CounterDataImage_Initialize_Default
        , &LWPW_LWDA_Profiler_CounterDataImage_CallwlateScratchBufferSize_Default
        , &LWPW_LWDA_Profiler_CounterDataImage_InitializeScratchBuffer_Default
        , &LWPW_LWDA_GetDeviceOrdinals_Default
        , &LWPW_LWDA_LoadDriver_Default
        , &LWPW_LWDA_Profiler_CalcTraceBufferSize_Default
        , &LWPW_LWDA_Profiler_BeginSession_Default
        , &LWPW_LWDA_Profiler_EndSession_Default
        , &LWPW_LWDA_Profiler_SetConfig_Default
        , &LWPW_LWDA_Profiler_ClearConfig_Default
        , &LWPW_LWDA_Profiler_BeginPass_Default
        , &LWPW_LWDA_Profiler_EndPass_Default
        , &LWPW_LWDA_Profiler_DecodeCounters_Default
        , &LWPW_LWDA_Profiler_EnablePerLaunchProfiling_Default
        , &LWPW_LWDA_Profiler_DisablePerLaunchProfiling_Default
        , &LWPW_LWDA_Profiler_PushRange_Default
        , &LWPW_LWDA_Profiler_PopRange_Default
        , &LWPW_VK_MetricsContext_Create_Default
        , &LWPW_VK_RawMetricsConfig_Create_Default
        , &LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Default
        , &LWPW_VK_Profiler_CounterDataImage_Initialize_Default
        , &LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Default
        , &LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Default
        , &LWPW_VK_LoadDriver_Default
        , &LWPW_VK_Device_GetDeviceIndex_Default
        , &LWPW_VK_Profiler_GetRequiredInstanceExtensions_Default
        , &LWPW_VK_Profiler_GetRequiredDeviceExtensions_Default
        , &LWPW_VK_Profiler_CalcTraceBufferSize_Default
        , &LWPW_VK_Profiler_Queue_BeginSession_Default
        , &LWPW_VK_Profiler_Queue_EndSession_Default
        , &LWPW_VK_Queue_ServicePendingGpuOperations_Default
        , &LWPW_VK_Profiler_Queue_SetConfig_Default
        , &LWPW_VK_Profiler_Queue_ClearConfig_Default
        , &LWPW_VK_Profiler_Queue_BeginPass_Default
        , &LWPW_VK_Profiler_Queue_EndPass_Default
        , &LWPW_VK_Profiler_CommandBuffer_PushRange_Default
        , &LWPW_VK_Profiler_CommandBuffer_PopRange_Default
        , &LWPW_VK_Profiler_Queue_DecodeCounters_Default
        , &LWPW_VK_Profiler_IsGpuSupported_Default
        , &LWPW_VK_PeriodicSampler_CallwlateMemoryOverhead_Default
        , &LWPW_VK_PeriodicSampler_Device_CallwlateMemoryOverhead_Default
        , &LWPW_VK_PeriodicSampler_Queue_BeginSession_Default
        , &LWPW_VK_PeriodicSampler_Queue_EndSession_Default
        , &LWPW_VK_PeriodicSampler_CommandBuffer_StartSampling_Default
        , &LWPW_VK_PeriodicSampler_CommandBuffer_StopSampling_Default
        , &LWPW_VK_PeriodicSampler_CommandBuffer_InsertDelimiter_Default
        , &LWPW_VK_PeriodicSampler_CommandBuffer_BeginFrame_Default
        , &LWPW_VK_PeriodicSampler_CommandBuffer_InsertTrigger_Default
        , &LWPW_VK_PeriodicSampler_Queue_GetLastError_Default
        , &LWPW_VK_PeriodicSampler_CounterDataImage_CallwlateSize_Default
        , &LWPW_VK_PeriodicSampler_CounterDataImage_Initialize_Default
        , &LWPW_VK_PeriodicSampler_Queue_DecodeCounters_Default
        , &LWPW_VK_PeriodicSampler_IsGpuSupported_Default
        , &LWPW_VK_PeriodicSampler_Queue_DiscardFrame_Default
        , &LWPW_VK_PeriodicSampler_Queue_CallwlateRecordBufferSize_Default
        , &LWPW_VK_PeriodicSampler_Queue_SetConfig_Default
        , &LWPW_OpenGL_MetricsContext_Create_Default
        , &LWPW_OpenGL_RawMetricsConfig_Create_Default
        , &LWPW_OpenGL_LoadDriver_Default
        , &LWPW_OpenGL_GetLwrrentGraphicsContext_Default
        , &LWPW_OpenGL_GraphicsContext_GetDeviceIndex_Default
        , &LWPW_OpenGL_Profiler_IsGpuSupported_Default
        , &LWPW_OpenGL_Profiler_CounterDataImage_CallwlateSize_Default
        , &LWPW_OpenGL_Profiler_CounterDataImage_Initialize_Default
        , &LWPW_OpenGL_Profiler_CounterDataImage_CallwlateScratchBufferSize_Default
        , &LWPW_OpenGL_Profiler_CounterDataImage_InitializeScratchBuffer_Default
        , &LWPW_OpenGL_Profiler_CalcTraceBufferSize_Default
        , &LWPW_OpenGL_Profiler_GraphicsContext_BeginSession_Default
        , &LWPW_OpenGL_Profiler_GraphicsContext_EndSession_Default
        , &LWPW_OpenGL_Profiler_GraphicsContext_SetConfig_Default
        , &LWPW_OpenGL_Profiler_GraphicsContext_ClearConfig_Default
        , &LWPW_OpenGL_Profiler_GraphicsContext_BeginPass_Default
        , &LWPW_OpenGL_Profiler_GraphicsContext_EndPass_Default
        , &LWPW_OpenGL_Profiler_GraphicsContext_PushRange_Default
        , &LWPW_OpenGL_Profiler_GraphicsContext_PopRange_Default
        , &LWPW_OpenGL_Profiler_GraphicsContext_DecodeCounters_Default
      }
    , 0 /* numSearchPaths */
    , 0 /* ppSearchPaths */
};

static LWPA_GenericFn GetPerfworksProc(const char* pName, LWPA_GenericFn pDefault);
static int InitPerfworks(void);

static void InitPerfworksProcs(void)
{
    g_api.fn.LWPA_GetProcAddress = (LWPA_GetProcAddress_Fn)GetPerfworksProc("LWPA_GetProcAddress", (LWPA_GenericFn)g_api.fn.LWPA_GetProcAddress);
    g_api.fn.LWPW_SetLibraryLoadPaths = (LWPW_SetLibraryLoadPaths_Fn)GetPerfworksProc("LWPW_SetLibraryLoadPaths", (LWPA_GenericFn)g_api.fn.LWPW_SetLibraryLoadPaths);
    g_api.fn.LWPW_SetLibraryLoadPathsW = (LWPW_SetLibraryLoadPathsW_Fn)GetPerfworksProc("LWPW_SetLibraryLoadPathsW", (LWPA_GenericFn)g_api.fn.LWPW_SetLibraryLoadPathsW);
    g_api.fn.LWPA_InitializeHost = (LWPA_InitializeHost_Fn)GetPerfworksProc("LWPA_InitializeHost", (LWPA_GenericFn)g_api.fn.LWPA_InitializeHost);
    g_api.fn.LWPW_InitializeHost = (LWPW_InitializeHost_Fn)GetPerfworksProc("LWPW_InitializeHost", (LWPA_GenericFn)g_api.fn.LWPW_InitializeHost);
    g_api.fn.LWPA_CounterData_CallwlateCounterDataImageCopySize = (LWPA_CounterData_CallwlateCounterDataImageCopySize_Fn)GetPerfworksProc("LWPA_CounterData_CallwlateCounterDataImageCopySize", (LWPA_GenericFn)g_api.fn.LWPA_CounterData_CallwlateCounterDataImageCopySize);
    g_api.fn.LWPW_CounterData_CallwlateCounterDataImageCopySize = (LWPW_CounterData_CallwlateCounterDataImageCopySize_Fn)GetPerfworksProc("LWPW_CounterData_CallwlateCounterDataImageCopySize", (LWPA_GenericFn)g_api.fn.LWPW_CounterData_CallwlateCounterDataImageCopySize);
    g_api.fn.LWPA_CounterData_InitializeCounterDataImageCopy = (LWPA_CounterData_InitializeCounterDataImageCopy_Fn)GetPerfworksProc("LWPA_CounterData_InitializeCounterDataImageCopy", (LWPA_GenericFn)g_api.fn.LWPA_CounterData_InitializeCounterDataImageCopy);
    g_api.fn.LWPW_CounterData_InitializeCounterDataImageCopy = (LWPW_CounterData_InitializeCounterDataImageCopy_Fn)GetPerfworksProc("LWPW_CounterData_InitializeCounterDataImageCopy", (LWPA_GenericFn)g_api.fn.LWPW_CounterData_InitializeCounterDataImageCopy);
    g_api.fn.LWPA_CounterDataCombiner_Create = (LWPA_CounterDataCombiner_Create_Fn)GetPerfworksProc("LWPA_CounterDataCombiner_Create", (LWPA_GenericFn)g_api.fn.LWPA_CounterDataCombiner_Create);
    g_api.fn.LWPW_CounterDataCombiner_Create = (LWPW_CounterDataCombiner_Create_Fn)GetPerfworksProc("LWPW_CounterDataCombiner_Create", (LWPA_GenericFn)g_api.fn.LWPW_CounterDataCombiner_Create);
    g_api.fn.LWPA_CounterDataCombiner_Destroy = (LWPA_CounterDataCombiner_Destroy_Fn)GetPerfworksProc("LWPA_CounterDataCombiner_Destroy", (LWPA_GenericFn)g_api.fn.LWPA_CounterDataCombiner_Destroy);
    g_api.fn.LWPW_CounterDataCombiner_Destroy = (LWPW_CounterDataCombiner_Destroy_Fn)GetPerfworksProc("LWPW_CounterDataCombiner_Destroy", (LWPA_GenericFn)g_api.fn.LWPW_CounterDataCombiner_Destroy);
    g_api.fn.LWPA_CounterDataCombiner_CreateRange = (LWPA_CounterDataCombiner_CreateRange_Fn)GetPerfworksProc("LWPA_CounterDataCombiner_CreateRange", (LWPA_GenericFn)g_api.fn.LWPA_CounterDataCombiner_CreateRange);
    g_api.fn.LWPW_CounterDataCombiner_CreateRange = (LWPW_CounterDataCombiner_CreateRange_Fn)GetPerfworksProc("LWPW_CounterDataCombiner_CreateRange", (LWPA_GenericFn)g_api.fn.LWPW_CounterDataCombiner_CreateRange);
    g_api.fn.LWPA_CounterDataCombiner_AclwmulateIntoRange = (LWPA_CounterDataCombiner_AclwmulateIntoRange_Fn)GetPerfworksProc("LWPA_CounterDataCombiner_AclwmulateIntoRange", (LWPA_GenericFn)g_api.fn.LWPA_CounterDataCombiner_AclwmulateIntoRange);
    g_api.fn.LWPW_CounterDataCombiner_AclwmulateIntoRange = (LWPW_CounterDataCombiner_AclwmulateIntoRange_Fn)GetPerfworksProc("LWPW_CounterDataCombiner_AclwmulateIntoRange", (LWPA_GenericFn)g_api.fn.LWPW_CounterDataCombiner_AclwmulateIntoRange);
    g_api.fn.LWPW_CounterDataCombiner_SumIntoRange = (LWPW_CounterDataCombiner_SumIntoRange_Fn)GetPerfworksProc("LWPW_CounterDataCombiner_SumIntoRange", (LWPA_GenericFn)g_api.fn.LWPW_CounterDataCombiner_SumIntoRange);
    g_api.fn.LWPW_CounterDataCombiner_WeightedSumIntoRange = (LWPW_CounterDataCombiner_WeightedSumIntoRange_Fn)GetPerfworksProc("LWPW_CounterDataCombiner_WeightedSumIntoRange", (LWPA_GenericFn)g_api.fn.LWPW_CounterDataCombiner_WeightedSumIntoRange);
    g_api.fn.LWPA_GetSupportedChipNames = (LWPA_GetSupportedChipNames_Fn)GetPerfworksProc("LWPA_GetSupportedChipNames", (LWPA_GenericFn)g_api.fn.LWPA_GetSupportedChipNames);
    g_api.fn.LWPW_GetSupportedChipNames = (LWPW_GetSupportedChipNames_Fn)GetPerfworksProc("LWPW_GetSupportedChipNames", (LWPA_GenericFn)g_api.fn.LWPW_GetSupportedChipNames);
    g_api.fn.LWPA_RawMetricsConfig_Create = (LWPA_RawMetricsConfig_Create_Fn)GetPerfworksProc("LWPA_RawMetricsConfig_Create", (LWPA_GenericFn)g_api.fn.LWPA_RawMetricsConfig_Create);
    g_api.fn.LWPA_RawMetricsConfig_Destroy = (LWPA_RawMetricsConfig_Destroy_Fn)GetPerfworksProc("LWPA_RawMetricsConfig_Destroy", (LWPA_GenericFn)g_api.fn.LWPA_RawMetricsConfig_Destroy);
    g_api.fn.LWPW_RawMetricsConfig_Destroy = (LWPW_RawMetricsConfig_Destroy_Fn)GetPerfworksProc("LWPW_RawMetricsConfig_Destroy", (LWPA_GenericFn)g_api.fn.LWPW_RawMetricsConfig_Destroy);
    g_api.fn.LWPA_RawMetricsConfig_BeginPassGroup = (LWPA_RawMetricsConfig_BeginPassGroup_Fn)GetPerfworksProc("LWPA_RawMetricsConfig_BeginPassGroup", (LWPA_GenericFn)g_api.fn.LWPA_RawMetricsConfig_BeginPassGroup);
    g_api.fn.LWPW_RawMetricsConfig_BeginPassGroup = (LWPW_RawMetricsConfig_BeginPassGroup_Fn)GetPerfworksProc("LWPW_RawMetricsConfig_BeginPassGroup", (LWPA_GenericFn)g_api.fn.LWPW_RawMetricsConfig_BeginPassGroup);
    g_api.fn.LWPA_RawMetricsConfig_EndPassGroup = (LWPA_RawMetricsConfig_EndPassGroup_Fn)GetPerfworksProc("LWPA_RawMetricsConfig_EndPassGroup", (LWPA_GenericFn)g_api.fn.LWPA_RawMetricsConfig_EndPassGroup);
    g_api.fn.LWPW_RawMetricsConfig_EndPassGroup = (LWPW_RawMetricsConfig_EndPassGroup_Fn)GetPerfworksProc("LWPW_RawMetricsConfig_EndPassGroup", (LWPA_GenericFn)g_api.fn.LWPW_RawMetricsConfig_EndPassGroup);
    g_api.fn.LWPA_RawMetricsConfig_GetNumMetrics = (LWPA_RawMetricsConfig_GetNumMetrics_Fn)GetPerfworksProc("LWPA_RawMetricsConfig_GetNumMetrics", (LWPA_GenericFn)g_api.fn.LWPA_RawMetricsConfig_GetNumMetrics);
    g_api.fn.LWPW_RawMetricsConfig_GetNumMetrics = (LWPW_RawMetricsConfig_GetNumMetrics_Fn)GetPerfworksProc("LWPW_RawMetricsConfig_GetNumMetrics", (LWPA_GenericFn)g_api.fn.LWPW_RawMetricsConfig_GetNumMetrics);
    g_api.fn.LWPA_RawMetricsConfig_GetMetricProperties = (LWPA_RawMetricsConfig_GetMetricProperties_Fn)GetPerfworksProc("LWPA_RawMetricsConfig_GetMetricProperties", (LWPA_GenericFn)g_api.fn.LWPA_RawMetricsConfig_GetMetricProperties);
    g_api.fn.LWPW_RawMetricsConfig_GetMetricProperties = (LWPW_RawMetricsConfig_GetMetricProperties_Fn)GetPerfworksProc("LWPW_RawMetricsConfig_GetMetricProperties", (LWPA_GenericFn)g_api.fn.LWPW_RawMetricsConfig_GetMetricProperties);
    g_api.fn.LWPA_RawMetricsConfig_AddMetrics = (LWPA_RawMetricsConfig_AddMetrics_Fn)GetPerfworksProc("LWPA_RawMetricsConfig_AddMetrics", (LWPA_GenericFn)g_api.fn.LWPA_RawMetricsConfig_AddMetrics);
    g_api.fn.LWPW_RawMetricsConfig_AddMetrics = (LWPW_RawMetricsConfig_AddMetrics_Fn)GetPerfworksProc("LWPW_RawMetricsConfig_AddMetrics", (LWPA_GenericFn)g_api.fn.LWPW_RawMetricsConfig_AddMetrics);
    g_api.fn.LWPA_RawMetricsConfig_IsAddMetricsPossible = (LWPA_RawMetricsConfig_IsAddMetricsPossible_Fn)GetPerfworksProc("LWPA_RawMetricsConfig_IsAddMetricsPossible", (LWPA_GenericFn)g_api.fn.LWPA_RawMetricsConfig_IsAddMetricsPossible);
    g_api.fn.LWPW_RawMetricsConfig_IsAddMetricsPossible = (LWPW_RawMetricsConfig_IsAddMetricsPossible_Fn)GetPerfworksProc("LWPW_RawMetricsConfig_IsAddMetricsPossible", (LWPA_GenericFn)g_api.fn.LWPW_RawMetricsConfig_IsAddMetricsPossible);
    g_api.fn.LWPA_RawMetricsConfig_GenerateConfigImage = (LWPA_RawMetricsConfig_GenerateConfigImage_Fn)GetPerfworksProc("LWPA_RawMetricsConfig_GenerateConfigImage", (LWPA_GenericFn)g_api.fn.LWPA_RawMetricsConfig_GenerateConfigImage);
    g_api.fn.LWPW_RawMetricsConfig_GenerateConfigImage = (LWPW_RawMetricsConfig_GenerateConfigImage_Fn)GetPerfworksProc("LWPW_RawMetricsConfig_GenerateConfigImage", (LWPA_GenericFn)g_api.fn.LWPW_RawMetricsConfig_GenerateConfigImage);
    g_api.fn.LWPA_RawMetricsConfig_GetConfigImage = (LWPA_RawMetricsConfig_GetConfigImage_Fn)GetPerfworksProc("LWPA_RawMetricsConfig_GetConfigImage", (LWPA_GenericFn)g_api.fn.LWPA_RawMetricsConfig_GetConfigImage);
    g_api.fn.LWPW_RawMetricsConfig_GetConfigImage = (LWPW_RawMetricsConfig_GetConfigImage_Fn)GetPerfworksProc("LWPW_RawMetricsConfig_GetConfigImage", (LWPA_GenericFn)g_api.fn.LWPW_RawMetricsConfig_GetConfigImage);
    g_api.fn.LWPA_RawMetricsConfig_GetNumPasses = (LWPA_RawMetricsConfig_GetNumPasses_Fn)GetPerfworksProc("LWPA_RawMetricsConfig_GetNumPasses", (LWPA_GenericFn)g_api.fn.LWPA_RawMetricsConfig_GetNumPasses);
    g_api.fn.LWPW_RawMetricsConfig_GetNumPasses = (LWPW_RawMetricsConfig_GetNumPasses_Fn)GetPerfworksProc("LWPW_RawMetricsConfig_GetNumPasses", (LWPA_GenericFn)g_api.fn.LWPW_RawMetricsConfig_GetNumPasses);
    g_api.fn.LWPA_CounterDataBuilder_Create = (LWPA_CounterDataBuilder_Create_Fn)GetPerfworksProc("LWPA_CounterDataBuilder_Create", (LWPA_GenericFn)g_api.fn.LWPA_CounterDataBuilder_Create);
    g_api.fn.LWPW_CounterDataBuilder_Create = (LWPW_CounterDataBuilder_Create_Fn)GetPerfworksProc("LWPW_CounterDataBuilder_Create", (LWPA_GenericFn)g_api.fn.LWPW_CounterDataBuilder_Create);
    g_api.fn.LWPA_CounterDataBuilder_Destroy = (LWPA_CounterDataBuilder_Destroy_Fn)GetPerfworksProc("LWPA_CounterDataBuilder_Destroy", (LWPA_GenericFn)g_api.fn.LWPA_CounterDataBuilder_Destroy);
    g_api.fn.LWPW_CounterDataBuilder_Destroy = (LWPW_CounterDataBuilder_Destroy_Fn)GetPerfworksProc("LWPW_CounterDataBuilder_Destroy", (LWPA_GenericFn)g_api.fn.LWPW_CounterDataBuilder_Destroy);
    g_api.fn.LWPA_CounterDataBuilder_AddMetrics = (LWPA_CounterDataBuilder_AddMetrics_Fn)GetPerfworksProc("LWPA_CounterDataBuilder_AddMetrics", (LWPA_GenericFn)g_api.fn.LWPA_CounterDataBuilder_AddMetrics);
    g_api.fn.LWPW_CounterDataBuilder_AddMetrics = (LWPW_CounterDataBuilder_AddMetrics_Fn)GetPerfworksProc("LWPW_CounterDataBuilder_AddMetrics", (LWPA_GenericFn)g_api.fn.LWPW_CounterDataBuilder_AddMetrics);
    g_api.fn.LWPA_CounterDataBuilder_GetCounterDataPrefix = (LWPA_CounterDataBuilder_GetCounterDataPrefix_Fn)GetPerfworksProc("LWPA_CounterDataBuilder_GetCounterDataPrefix", (LWPA_GenericFn)g_api.fn.LWPA_CounterDataBuilder_GetCounterDataPrefix);
    g_api.fn.LWPW_CounterDataBuilder_GetCounterDataPrefix = (LWPW_CounterDataBuilder_GetCounterDataPrefix_Fn)GetPerfworksProc("LWPW_CounterDataBuilder_GetCounterDataPrefix", (LWPA_GenericFn)g_api.fn.LWPW_CounterDataBuilder_GetCounterDataPrefix);
    g_api.fn.LWPA_MetricsContext_Create = (LWPA_MetricsContext_Create_Fn)GetPerfworksProc("LWPA_MetricsContext_Create", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_Create);
    g_api.fn.LWPA_MetricsContext_Destroy = (LWPA_MetricsContext_Destroy_Fn)GetPerfworksProc("LWPA_MetricsContext_Destroy", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_Destroy);
    g_api.fn.LWPW_MetricsContext_Destroy = (LWPW_MetricsContext_Destroy_Fn)GetPerfworksProc("LWPW_MetricsContext_Destroy", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_Destroy);
    g_api.fn.LWPA_MetricsContext_RunScript = (LWPA_MetricsContext_RunScript_Fn)GetPerfworksProc("LWPA_MetricsContext_RunScript", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_RunScript);
    g_api.fn.LWPW_MetricsContext_RunScript = (LWPW_MetricsContext_RunScript_Fn)GetPerfworksProc("LWPW_MetricsContext_RunScript", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_RunScript);
    g_api.fn.LWPA_MetricsContext_ExecScript_Begin = (LWPA_MetricsContext_ExecScript_Begin_Fn)GetPerfworksProc("LWPA_MetricsContext_ExecScript_Begin", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_ExecScript_Begin);
    g_api.fn.LWPW_MetricsContext_ExecScript_Begin = (LWPW_MetricsContext_ExecScript_Begin_Fn)GetPerfworksProc("LWPW_MetricsContext_ExecScript_Begin", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_ExecScript_Begin);
    g_api.fn.LWPA_MetricsContext_ExecScript_End = (LWPA_MetricsContext_ExecScript_End_Fn)GetPerfworksProc("LWPA_MetricsContext_ExecScript_End", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_ExecScript_End);
    g_api.fn.LWPW_MetricsContext_ExecScript_End = (LWPW_MetricsContext_ExecScript_End_Fn)GetPerfworksProc("LWPW_MetricsContext_ExecScript_End", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_ExecScript_End);
    g_api.fn.LWPA_MetricsContext_GetCounterNames_Begin = (LWPA_MetricsContext_GetCounterNames_Begin_Fn)GetPerfworksProc("LWPA_MetricsContext_GetCounterNames_Begin", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_GetCounterNames_Begin);
    g_api.fn.LWPW_MetricsContext_GetCounterNames_Begin = (LWPW_MetricsContext_GetCounterNames_Begin_Fn)GetPerfworksProc("LWPW_MetricsContext_GetCounterNames_Begin", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetCounterNames_Begin);
    g_api.fn.LWPA_MetricsContext_GetCounterNames_End = (LWPA_MetricsContext_GetCounterNames_End_Fn)GetPerfworksProc("LWPA_MetricsContext_GetCounterNames_End", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_GetCounterNames_End);
    g_api.fn.LWPW_MetricsContext_GetCounterNames_End = (LWPW_MetricsContext_GetCounterNames_End_Fn)GetPerfworksProc("LWPW_MetricsContext_GetCounterNames_End", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetCounterNames_End);
    g_api.fn.LWPA_MetricsContext_GetThroughputNames_Begin = (LWPA_MetricsContext_GetThroughputNames_Begin_Fn)GetPerfworksProc("LWPA_MetricsContext_GetThroughputNames_Begin", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_GetThroughputNames_Begin);
    g_api.fn.LWPW_MetricsContext_GetThroughputNames_Begin = (LWPW_MetricsContext_GetThroughputNames_Begin_Fn)GetPerfworksProc("LWPW_MetricsContext_GetThroughputNames_Begin", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetThroughputNames_Begin);
    g_api.fn.LWPA_MetricsContext_GetThroughputNames_End = (LWPA_MetricsContext_GetThroughputNames_End_Fn)GetPerfworksProc("LWPA_MetricsContext_GetThroughputNames_End", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_GetThroughputNames_End);
    g_api.fn.LWPW_MetricsContext_GetThroughputNames_End = (LWPW_MetricsContext_GetThroughputNames_End_Fn)GetPerfworksProc("LWPW_MetricsContext_GetThroughputNames_End", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetThroughputNames_End);
    g_api.fn.LWPW_MetricsContext_GetRatioNames_Begin = (LWPW_MetricsContext_GetRatioNames_Begin_Fn)GetPerfworksProc("LWPW_MetricsContext_GetRatioNames_Begin", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetRatioNames_Begin);
    g_api.fn.LWPW_MetricsContext_GetRatioNames_End = (LWPW_MetricsContext_GetRatioNames_End_Fn)GetPerfworksProc("LWPW_MetricsContext_GetRatioNames_End", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetRatioNames_End);
    g_api.fn.LWPA_MetricsContext_GetMetricNames_Begin = (LWPA_MetricsContext_GetMetricNames_Begin_Fn)GetPerfworksProc("LWPA_MetricsContext_GetMetricNames_Begin", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_GetMetricNames_Begin);
    g_api.fn.LWPW_MetricsContext_GetMetricNames_Begin = (LWPW_MetricsContext_GetMetricNames_Begin_Fn)GetPerfworksProc("LWPW_MetricsContext_GetMetricNames_Begin", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetMetricNames_Begin);
    g_api.fn.LWPA_MetricsContext_GetMetricNames_End = (LWPA_MetricsContext_GetMetricNames_End_Fn)GetPerfworksProc("LWPA_MetricsContext_GetMetricNames_End", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_GetMetricNames_End);
    g_api.fn.LWPW_MetricsContext_GetMetricNames_End = (LWPW_MetricsContext_GetMetricNames_End_Fn)GetPerfworksProc("LWPW_MetricsContext_GetMetricNames_End", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetMetricNames_End);
    g_api.fn.LWPA_MetricsContext_GetThroughputBreakdown_Begin = (LWPA_MetricsContext_GetThroughputBreakdown_Begin_Fn)GetPerfworksProc("LWPA_MetricsContext_GetThroughputBreakdown_Begin", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_GetThroughputBreakdown_Begin);
    g_api.fn.LWPW_MetricsContext_GetThroughputBreakdown_Begin = (LWPW_MetricsContext_GetThroughputBreakdown_Begin_Fn)GetPerfworksProc("LWPW_MetricsContext_GetThroughputBreakdown_Begin", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetThroughputBreakdown_Begin);
    g_api.fn.LWPA_MetricsContext_GetThroughputBreakdown_End = (LWPA_MetricsContext_GetThroughputBreakdown_End_Fn)GetPerfworksProc("LWPA_MetricsContext_GetThroughputBreakdown_End", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_GetThroughputBreakdown_End);
    g_api.fn.LWPW_MetricsContext_GetThroughputBreakdown_End = (LWPW_MetricsContext_GetThroughputBreakdown_End_Fn)GetPerfworksProc("LWPW_MetricsContext_GetThroughputBreakdown_End", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetThroughputBreakdown_End);
    g_api.fn.LWPA_MetricsContext_GetMetricProperties_Begin = (LWPA_MetricsContext_GetMetricProperties_Begin_Fn)GetPerfworksProc("LWPA_MetricsContext_GetMetricProperties_Begin", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_GetMetricProperties_Begin);
    g_api.fn.LWPW_MetricsContext_GetMetricProperties_Begin = (LWPW_MetricsContext_GetMetricProperties_Begin_Fn)GetPerfworksProc("LWPW_MetricsContext_GetMetricProperties_Begin", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetMetricProperties_Begin);
    g_api.fn.LWPA_MetricsContext_GetMetricProperties_End = (LWPA_MetricsContext_GetMetricProperties_End_Fn)GetPerfworksProc("LWPA_MetricsContext_GetMetricProperties_End", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_GetMetricProperties_End);
    g_api.fn.LWPW_MetricsContext_GetMetricProperties_End = (LWPW_MetricsContext_GetMetricProperties_End_Fn)GetPerfworksProc("LWPW_MetricsContext_GetMetricProperties_End", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetMetricProperties_End);
    g_api.fn.LWPA_MetricsContext_SetCounterData = (LWPA_MetricsContext_SetCounterData_Fn)GetPerfworksProc("LWPA_MetricsContext_SetCounterData", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_SetCounterData);
    g_api.fn.LWPW_MetricsContext_SetCounterData = (LWPW_MetricsContext_SetCounterData_Fn)GetPerfworksProc("LWPW_MetricsContext_SetCounterData", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_SetCounterData);
    g_api.fn.LWPA_MetricsContext_SetUserData = (LWPA_MetricsContext_SetUserData_Fn)GetPerfworksProc("LWPA_MetricsContext_SetUserData", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_SetUserData);
    g_api.fn.LWPW_MetricsContext_SetUserData = (LWPW_MetricsContext_SetUserData_Fn)GetPerfworksProc("LWPW_MetricsContext_SetUserData", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_SetUserData);
    g_api.fn.LWPA_MetricsContext_EvaluateToGpuValues = (LWPA_MetricsContext_EvaluateToGpuValues_Fn)GetPerfworksProc("LWPA_MetricsContext_EvaluateToGpuValues", (LWPA_GenericFn)g_api.fn.LWPA_MetricsContext_EvaluateToGpuValues);
    g_api.fn.LWPW_MetricsContext_EvaluateToGpuValues = (LWPW_MetricsContext_EvaluateToGpuValues_Fn)GetPerfworksProc("LWPW_MetricsContext_EvaluateToGpuValues", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_EvaluateToGpuValues);
    g_api.fn.LWPW_MetricsContext_GetMetricSuffix_Begin = (LWPW_MetricsContext_GetMetricSuffix_Begin_Fn)GetPerfworksProc("LWPW_MetricsContext_GetMetricSuffix_Begin", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetMetricSuffix_Begin);
    g_api.fn.LWPW_MetricsContext_GetMetricSuffix_End = (LWPW_MetricsContext_GetMetricSuffix_End_Fn)GetPerfworksProc("LWPW_MetricsContext_GetMetricSuffix_End", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetMetricSuffix_End);
    g_api.fn.LWPW_MetricsContext_GetMetricBaseNames_Begin = (LWPW_MetricsContext_GetMetricBaseNames_Begin_Fn)GetPerfworksProc("LWPW_MetricsContext_GetMetricBaseNames_Begin", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetMetricBaseNames_Begin);
    g_api.fn.LWPW_MetricsContext_GetMetricBaseNames_End = (LWPW_MetricsContext_GetMetricBaseNames_End_Fn)GetPerfworksProc("LWPW_MetricsContext_GetMetricBaseNames_End", (LWPA_GenericFn)g_api.fn.LWPW_MetricsContext_GetMetricBaseNames_End);
    g_api.fn.LWPA_InitializeTarget = (LWPA_InitializeTarget_Fn)GetPerfworksProc("LWPA_InitializeTarget", (LWPA_GenericFn)g_api.fn.LWPA_InitializeTarget);
    g_api.fn.LWPA_GetDeviceCount = (LWPA_GetDeviceCount_Fn)GetPerfworksProc("LWPA_GetDeviceCount", (LWPA_GenericFn)g_api.fn.LWPA_GetDeviceCount);
    g_api.fn.LWPA_Device_GetNames = (LWPA_Device_GetNames_Fn)GetPerfworksProc("LWPA_Device_GetNames", (LWPA_GenericFn)g_api.fn.LWPA_Device_GetNames);
    g_api.fn.LWPA_CounterData_GetNumRanges = (LWPA_CounterData_GetNumRanges_Fn)GetPerfworksProc("LWPA_CounterData_GetNumRanges", (LWPA_GenericFn)g_api.fn.LWPA_CounterData_GetNumRanges);
    g_api.fn.LWPA_CounterData_GetRangeDescriptions = (LWPA_CounterData_GetRangeDescriptions_Fn)GetPerfworksProc("LWPA_CounterData_GetRangeDescriptions", (LWPA_GenericFn)g_api.fn.LWPA_CounterData_GetRangeDescriptions);
    g_api.fn.LWPW_InitializeTarget = (LWPW_InitializeTarget_Fn)GetPerfworksProc("LWPW_InitializeTarget", (LWPA_GenericFn)g_api.fn.LWPW_InitializeTarget);
    g_api.fn.LWPW_GetDeviceCount = (LWPW_GetDeviceCount_Fn)GetPerfworksProc("LWPW_GetDeviceCount", (LWPA_GenericFn)g_api.fn.LWPW_GetDeviceCount);
    g_api.fn.LWPW_Device_GetNames = (LWPW_Device_GetNames_Fn)GetPerfworksProc("LWPW_Device_GetNames", (LWPA_GenericFn)g_api.fn.LWPW_Device_GetNames);
    g_api.fn.LWPW_Device_GetPciBusIds = (LWPW_Device_GetPciBusIds_Fn)GetPerfworksProc("LWPW_Device_GetPciBusIds", (LWPA_GenericFn)g_api.fn.LWPW_Device_GetPciBusIds);
    g_api.fn.LWPW_Adapter_GetDeviceIndex = (LWPW_Adapter_GetDeviceIndex_Fn)GetPerfworksProc("LWPW_Adapter_GetDeviceIndex", (LWPA_GenericFn)g_api.fn.LWPW_Adapter_GetDeviceIndex);
    g_api.fn.LWPW_CounterData_GetNumRanges = (LWPW_CounterData_GetNumRanges_Fn)GetPerfworksProc("LWPW_CounterData_GetNumRanges", (LWPA_GenericFn)g_api.fn.LWPW_CounterData_GetNumRanges);
    g_api.fn.LWPW_Config_GetNumPasses = (LWPW_Config_GetNumPasses_Fn)GetPerfworksProc("LWPW_Config_GetNumPasses", (LWPA_GenericFn)g_api.fn.LWPW_Config_GetNumPasses);
    g_api.fn.LWPW_CounterData_GetRangeDescriptions = (LWPW_CounterData_GetRangeDescriptions_Fn)GetPerfworksProc("LWPW_CounterData_GetRangeDescriptions", (LWPA_GenericFn)g_api.fn.LWPW_CounterData_GetRangeDescriptions);
    g_api.fn.LWPW_Profiler_CounterData_GetRangeDescriptions = (LWPW_Profiler_CounterData_GetRangeDescriptions_Fn)GetPerfworksProc("LWPW_Profiler_CounterData_GetRangeDescriptions", (LWPA_GenericFn)g_api.fn.LWPW_Profiler_CounterData_GetRangeDescriptions);
    g_api.fn.LWPW_PeriodicSampler_CounterData_GetDelimiters = (LWPW_PeriodicSampler_CounterData_GetDelimiters_Fn)GetPerfworksProc("LWPW_PeriodicSampler_CounterData_GetDelimiters", (LWPA_GenericFn)g_api.fn.LWPW_PeriodicSampler_CounterData_GetDelimiters);
    g_api.fn.LWPW_PeriodicSampler_CounterData_GetSampleTime = (LWPW_PeriodicSampler_CounterData_GetSampleTime_Fn)GetPerfworksProc("LWPW_PeriodicSampler_CounterData_GetSampleTime", (LWPA_GenericFn)g_api.fn.LWPW_PeriodicSampler_CounterData_GetSampleTime);
    g_api.fn.LWPW_PeriodicSampler_CounterData_TrimInPlace = (LWPW_PeriodicSampler_CounterData_TrimInPlace_Fn)GetPerfworksProc("LWPW_PeriodicSampler_CounterData_TrimInPlace", (LWPA_GenericFn)g_api.fn.LWPW_PeriodicSampler_CounterData_TrimInPlace);
    g_api.fn.LWPW_LWDA_MetricsContext_Create = (LWPW_LWDA_MetricsContext_Create_Fn)GetPerfworksProc("LWPW_LWDA_MetricsContext_Create", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_MetricsContext_Create);
    g_api.fn.LWPW_LWDA_RawMetricsConfig_Create = (LWPW_LWDA_RawMetricsConfig_Create_Fn)GetPerfworksProc("LWPW_LWDA_RawMetricsConfig_Create", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_RawMetricsConfig_Create);
    g_api.fn.LWPW_LWDA_Profiler_CounterDataImage_CallwlateSize = (LWPW_LWDA_Profiler_CounterDataImage_CallwlateSize_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_CounterDataImage_CallwlateSize", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_CounterDataImage_CallwlateSize);
    g_api.fn.LWPW_LWDA_Profiler_CounterDataImage_Initialize = (LWPW_LWDA_Profiler_CounterDataImage_Initialize_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_CounterDataImage_Initialize", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_CounterDataImage_Initialize);
    g_api.fn.LWPW_LWDA_Profiler_CounterDataImage_CallwlateScratchBufferSize = (LWPW_LWDA_Profiler_CounterDataImage_CallwlateScratchBufferSize_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_CounterDataImage_CallwlateScratchBufferSize", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_CounterDataImage_CallwlateScratchBufferSize);
    g_api.fn.LWPW_LWDA_Profiler_CounterDataImage_InitializeScratchBuffer = (LWPW_LWDA_Profiler_CounterDataImage_InitializeScratchBuffer_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_CounterDataImage_InitializeScratchBuffer", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_CounterDataImage_InitializeScratchBuffer);
    g_api.fn.LWPW_LWDA_GetDeviceOrdinals = (LWPW_LWDA_GetDeviceOrdinals_Fn)GetPerfworksProc("LWPW_LWDA_GetDeviceOrdinals", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_GetDeviceOrdinals);
    g_api.fn.LWPW_LWDA_LoadDriver = (LWPW_LWDA_LoadDriver_Fn)GetPerfworksProc("LWPW_LWDA_LoadDriver", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_LoadDriver);
    g_api.fn.LWPW_LWDA_Profiler_CalcTraceBufferSize = (LWPW_LWDA_Profiler_CalcTraceBufferSize_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_CalcTraceBufferSize", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_CalcTraceBufferSize);
    g_api.fn.LWPW_LWDA_Profiler_BeginSession = (LWPW_LWDA_Profiler_BeginSession_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_BeginSession", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_BeginSession);
    g_api.fn.LWPW_LWDA_Profiler_EndSession = (LWPW_LWDA_Profiler_EndSession_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_EndSession", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_EndSession);
    g_api.fn.LWPW_LWDA_Profiler_SetConfig = (LWPW_LWDA_Profiler_SetConfig_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_SetConfig", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_SetConfig);
    g_api.fn.LWPW_LWDA_Profiler_ClearConfig = (LWPW_LWDA_Profiler_ClearConfig_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_ClearConfig", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_ClearConfig);
    g_api.fn.LWPW_LWDA_Profiler_BeginPass = (LWPW_LWDA_Profiler_BeginPass_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_BeginPass", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_BeginPass);
    g_api.fn.LWPW_LWDA_Profiler_EndPass = (LWPW_LWDA_Profiler_EndPass_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_EndPass", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_EndPass);
    g_api.fn.LWPW_LWDA_Profiler_DecodeCounters = (LWPW_LWDA_Profiler_DecodeCounters_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_DecodeCounters", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_DecodeCounters);
    g_api.fn.LWPW_LWDA_Profiler_EnablePerLaunchProfiling = (LWPW_LWDA_Profiler_EnablePerLaunchProfiling_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_EnablePerLaunchProfiling", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_EnablePerLaunchProfiling);
    g_api.fn.LWPW_LWDA_Profiler_DisablePerLaunchProfiling = (LWPW_LWDA_Profiler_DisablePerLaunchProfiling_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_DisablePerLaunchProfiling", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_DisablePerLaunchProfiling);
    g_api.fn.LWPW_LWDA_Profiler_PushRange = (LWPW_LWDA_Profiler_PushRange_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_PushRange", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_PushRange);
    g_api.fn.LWPW_LWDA_Profiler_PopRange = (LWPW_LWDA_Profiler_PopRange_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_PopRange", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_PopRange);
    g_api.fn.LWPW_VK_MetricsContext_Create = (LWPW_VK_MetricsContext_Create_Fn)GetPerfworksProc("LWPW_VK_MetricsContext_Create", (LWPA_GenericFn)g_api.fn.LWPW_VK_MetricsContext_Create);
    g_api.fn.LWPW_VK_RawMetricsConfig_Create = (LWPW_VK_RawMetricsConfig_Create_Fn)GetPerfworksProc("LWPW_VK_RawMetricsConfig_Create", (LWPA_GenericFn)g_api.fn.LWPW_VK_RawMetricsConfig_Create);
    g_api.fn.LWPW_VK_Profiler_CounterDataImage_CallwlateSize = (LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Fn)GetPerfworksProc("LWPW_VK_Profiler_CounterDataImage_CallwlateSize", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_CounterDataImage_CallwlateSize);
    g_api.fn.LWPW_VK_Profiler_CounterDataImage_Initialize = (LWPW_VK_Profiler_CounterDataImage_Initialize_Fn)GetPerfworksProc("LWPW_VK_Profiler_CounterDataImage_Initialize", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_CounterDataImage_Initialize);
    g_api.fn.LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize = (LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Fn)GetPerfworksProc("LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize);
    g_api.fn.LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer = (LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Fn)GetPerfworksProc("LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer);
    g_api.fn.LWPW_VK_LoadDriver = (LWPW_VK_LoadDriver_Fn)GetPerfworksProc("LWPW_VK_LoadDriver", (LWPA_GenericFn)g_api.fn.LWPW_VK_LoadDriver);
    g_api.fn.LWPW_VK_Device_GetDeviceIndex = (LWPW_VK_Device_GetDeviceIndex_Fn)GetPerfworksProc("LWPW_VK_Device_GetDeviceIndex", (LWPA_GenericFn)g_api.fn.LWPW_VK_Device_GetDeviceIndex);
    g_api.fn.LWPW_VK_Profiler_GetRequiredInstanceExtensions = (LWPW_VK_Profiler_GetRequiredInstanceExtensions_Fn)GetPerfworksProc("LWPW_VK_Profiler_GetRequiredInstanceExtensions", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_GetRequiredInstanceExtensions);
    g_api.fn.LWPW_VK_Profiler_GetRequiredDeviceExtensions = (LWPW_VK_Profiler_GetRequiredDeviceExtensions_Fn)GetPerfworksProc("LWPW_VK_Profiler_GetRequiredDeviceExtensions", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_GetRequiredDeviceExtensions);
    g_api.fn.LWPW_VK_Profiler_CalcTraceBufferSize = (LWPW_VK_Profiler_CalcTraceBufferSize_Fn)GetPerfworksProc("LWPW_VK_Profiler_CalcTraceBufferSize", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_CalcTraceBufferSize);
    g_api.fn.LWPW_VK_Profiler_Queue_BeginSession = (LWPW_VK_Profiler_Queue_BeginSession_Fn)GetPerfworksProc("LWPW_VK_Profiler_Queue_BeginSession", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_Queue_BeginSession);
    g_api.fn.LWPW_VK_Profiler_Queue_EndSession = (LWPW_VK_Profiler_Queue_EndSession_Fn)GetPerfworksProc("LWPW_VK_Profiler_Queue_EndSession", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_Queue_EndSession);
    g_api.fn.LWPW_VK_Queue_ServicePendingGpuOperations = (LWPW_VK_Queue_ServicePendingGpuOperations_Fn)GetPerfworksProc("LWPW_VK_Queue_ServicePendingGpuOperations", (LWPA_GenericFn)g_api.fn.LWPW_VK_Queue_ServicePendingGpuOperations);
    g_api.fn.LWPW_VK_Profiler_Queue_SetConfig = (LWPW_VK_Profiler_Queue_SetConfig_Fn)GetPerfworksProc("LWPW_VK_Profiler_Queue_SetConfig", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_Queue_SetConfig);
    g_api.fn.LWPW_VK_Profiler_Queue_ClearConfig = (LWPW_VK_Profiler_Queue_ClearConfig_Fn)GetPerfworksProc("LWPW_VK_Profiler_Queue_ClearConfig", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_Queue_ClearConfig);
    g_api.fn.LWPW_VK_Profiler_Queue_BeginPass = (LWPW_VK_Profiler_Queue_BeginPass_Fn)GetPerfworksProc("LWPW_VK_Profiler_Queue_BeginPass", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_Queue_BeginPass);
    g_api.fn.LWPW_VK_Profiler_Queue_EndPass = (LWPW_VK_Profiler_Queue_EndPass_Fn)GetPerfworksProc("LWPW_VK_Profiler_Queue_EndPass", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_Queue_EndPass);
    g_api.fn.LWPW_VK_Profiler_CommandBuffer_PushRange = (LWPW_VK_Profiler_CommandBuffer_PushRange_Fn)GetPerfworksProc("LWPW_VK_Profiler_CommandBuffer_PushRange", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_CommandBuffer_PushRange);
    g_api.fn.LWPW_VK_Profiler_CommandBuffer_PopRange = (LWPW_VK_Profiler_CommandBuffer_PopRange_Fn)GetPerfworksProc("LWPW_VK_Profiler_CommandBuffer_PopRange", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_CommandBuffer_PopRange);
    g_api.fn.LWPW_VK_Profiler_Queue_DecodeCounters = (LWPW_VK_Profiler_Queue_DecodeCounters_Fn)GetPerfworksProc("LWPW_VK_Profiler_Queue_DecodeCounters", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_Queue_DecodeCounters);
    g_api.fn.LWPW_VK_Profiler_IsGpuSupported = (LWPW_VK_Profiler_IsGpuSupported_Fn)GetPerfworksProc("LWPW_VK_Profiler_IsGpuSupported", (LWPA_GenericFn)g_api.fn.LWPW_VK_Profiler_IsGpuSupported);
    g_api.fn.LWPW_VK_PeriodicSampler_CallwlateMemoryOverhead = (LWPW_VK_PeriodicSampler_CallwlateMemoryOverhead_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_CallwlateMemoryOverhead", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_CallwlateMemoryOverhead);
    g_api.fn.LWPW_VK_PeriodicSampler_Device_CallwlateMemoryOverhead = (LWPW_VK_PeriodicSampler_Device_CallwlateMemoryOverhead_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_Device_CallwlateMemoryOverhead", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_Device_CallwlateMemoryOverhead);
    g_api.fn.LWPW_VK_PeriodicSampler_Queue_BeginSession = (LWPW_VK_PeriodicSampler_Queue_BeginSession_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_Queue_BeginSession", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_Queue_BeginSession);
    g_api.fn.LWPW_VK_PeriodicSampler_Queue_EndSession = (LWPW_VK_PeriodicSampler_Queue_EndSession_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_Queue_EndSession", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_Queue_EndSession);
    g_api.fn.LWPW_VK_PeriodicSampler_CommandBuffer_StartSampling = (LWPW_VK_PeriodicSampler_CommandBuffer_StartSampling_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_CommandBuffer_StartSampling", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_CommandBuffer_StartSampling);
    g_api.fn.LWPW_VK_PeriodicSampler_CommandBuffer_StopSampling = (LWPW_VK_PeriodicSampler_CommandBuffer_StopSampling_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_CommandBuffer_StopSampling", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_CommandBuffer_StopSampling);
    g_api.fn.LWPW_VK_PeriodicSampler_CommandBuffer_InsertDelimiter = (LWPW_VK_PeriodicSampler_CommandBuffer_InsertDelimiter_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_CommandBuffer_InsertDelimiter", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_CommandBuffer_InsertDelimiter);
    g_api.fn.LWPW_VK_PeriodicSampler_CommandBuffer_BeginFrame = (LWPW_VK_PeriodicSampler_CommandBuffer_BeginFrame_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_CommandBuffer_BeginFrame", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_CommandBuffer_BeginFrame);
    g_api.fn.LWPW_VK_PeriodicSampler_CommandBuffer_InsertTrigger = (LWPW_VK_PeriodicSampler_CommandBuffer_InsertTrigger_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_CommandBuffer_InsertTrigger", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_CommandBuffer_InsertTrigger);
    g_api.fn.LWPW_VK_PeriodicSampler_Queue_GetLastError = (LWPW_VK_PeriodicSampler_Queue_GetLastError_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_Queue_GetLastError", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_Queue_GetLastError);
    g_api.fn.LWPW_VK_PeriodicSampler_CounterDataImage_CallwlateSize = (LWPW_VK_PeriodicSampler_CounterDataImage_CallwlateSize_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_CounterDataImage_CallwlateSize", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_CounterDataImage_CallwlateSize);
    g_api.fn.LWPW_VK_PeriodicSampler_CounterDataImage_Initialize = (LWPW_VK_PeriodicSampler_CounterDataImage_Initialize_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_CounterDataImage_Initialize", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_CounterDataImage_Initialize);
    g_api.fn.LWPW_VK_PeriodicSampler_Queue_DecodeCounters = (LWPW_VK_PeriodicSampler_Queue_DecodeCounters_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_Queue_DecodeCounters", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_Queue_DecodeCounters);
    g_api.fn.LWPW_VK_PeriodicSampler_IsGpuSupported = (LWPW_VK_PeriodicSampler_IsGpuSupported_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_IsGpuSupported", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_IsGpuSupported);
    g_api.fn.LWPW_VK_PeriodicSampler_Queue_DiscardFrame = (LWPW_VK_PeriodicSampler_Queue_DiscardFrame_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_Queue_DiscardFrame", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_Queue_DiscardFrame);
    g_api.fn.LWPW_VK_PeriodicSampler_Queue_CallwlateRecordBufferSize = (LWPW_VK_PeriodicSampler_Queue_CallwlateRecordBufferSize_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_Queue_CallwlateRecordBufferSize", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_Queue_CallwlateRecordBufferSize);
    g_api.fn.LWPW_VK_PeriodicSampler_Queue_SetConfig = (LWPW_VK_PeriodicSampler_Queue_SetConfig_Fn)GetPerfworksProc("LWPW_VK_PeriodicSampler_Queue_SetConfig", (LWPA_GenericFn)g_api.fn.LWPW_VK_PeriodicSampler_Queue_SetConfig);
    g_api.fn.LWPW_OpenGL_MetricsContext_Create = (LWPW_OpenGL_MetricsContext_Create_Fn)GetPerfworksProc("LWPW_OpenGL_MetricsContext_Create", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_MetricsContext_Create);
    g_api.fn.LWPW_OpenGL_RawMetricsConfig_Create = (LWPW_OpenGL_RawMetricsConfig_Create_Fn)GetPerfworksProc("LWPW_OpenGL_RawMetricsConfig_Create", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_RawMetricsConfig_Create);
    g_api.fn.LWPW_OpenGL_LoadDriver = (LWPW_OpenGL_LoadDriver_Fn)GetPerfworksProc("LWPW_OpenGL_LoadDriver", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_LoadDriver);
    g_api.fn.LWPW_OpenGL_GetLwrrentGraphicsContext = (LWPW_OpenGL_GetLwrrentGraphicsContext_Fn)GetPerfworksProc("LWPW_OpenGL_GetLwrrentGraphicsContext", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_GetLwrrentGraphicsContext);
    g_api.fn.LWPW_OpenGL_GraphicsContext_GetDeviceIndex = (LWPW_OpenGL_GraphicsContext_GetDeviceIndex_Fn)GetPerfworksProc("LWPW_OpenGL_GraphicsContext_GetDeviceIndex", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_GraphicsContext_GetDeviceIndex);
    g_api.fn.LWPW_OpenGL_Profiler_IsGpuSupported = (LWPW_OpenGL_Profiler_IsGpuSupported_Fn)GetPerfworksProc("LWPW_OpenGL_Profiler_IsGpuSupported", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_Profiler_IsGpuSupported);
    g_api.fn.LWPW_OpenGL_Profiler_CounterDataImage_CallwlateSize = (LWPW_OpenGL_Profiler_CounterDataImage_CallwlateSize_Fn)GetPerfworksProc("LWPW_OpenGL_Profiler_CounterDataImage_CallwlateSize", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_Profiler_CounterDataImage_CallwlateSize);
    g_api.fn.LWPW_OpenGL_Profiler_CounterDataImage_Initialize = (LWPW_OpenGL_Profiler_CounterDataImage_Initialize_Fn)GetPerfworksProc("LWPW_OpenGL_Profiler_CounterDataImage_Initialize", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_Profiler_CounterDataImage_Initialize);
    g_api.fn.LWPW_OpenGL_Profiler_CounterDataImage_CallwlateScratchBufferSize = (LWPW_OpenGL_Profiler_CounterDataImage_CallwlateScratchBufferSize_Fn)GetPerfworksProc("LWPW_OpenGL_Profiler_CounterDataImage_CallwlateScratchBufferSize", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_Profiler_CounterDataImage_CallwlateScratchBufferSize);
    g_api.fn.LWPW_OpenGL_Profiler_CounterDataImage_InitializeScratchBuffer = (LWPW_OpenGL_Profiler_CounterDataImage_InitializeScratchBuffer_Fn)GetPerfworksProc("LWPW_OpenGL_Profiler_CounterDataImage_InitializeScratchBuffer", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_Profiler_CounterDataImage_InitializeScratchBuffer);
    g_api.fn.LWPW_OpenGL_Profiler_CalcTraceBufferSize = (LWPW_OpenGL_Profiler_CalcTraceBufferSize_Fn)GetPerfworksProc("LWPW_OpenGL_Profiler_CalcTraceBufferSize", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_Profiler_CalcTraceBufferSize);
    g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_BeginSession = (LWPW_OpenGL_Profiler_GraphicsContext_BeginSession_Fn)GetPerfworksProc("LWPW_OpenGL_Profiler_GraphicsContext_BeginSession", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_BeginSession);
    g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_EndSession = (LWPW_OpenGL_Profiler_GraphicsContext_EndSession_Fn)GetPerfworksProc("LWPW_OpenGL_Profiler_GraphicsContext_EndSession", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_EndSession);
    g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_SetConfig = (LWPW_OpenGL_Profiler_GraphicsContext_SetConfig_Fn)GetPerfworksProc("LWPW_OpenGL_Profiler_GraphicsContext_SetConfig", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_SetConfig);
    g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_ClearConfig = (LWPW_OpenGL_Profiler_GraphicsContext_ClearConfig_Fn)GetPerfworksProc("LWPW_OpenGL_Profiler_GraphicsContext_ClearConfig", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_ClearConfig);
    g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_BeginPass = (LWPW_OpenGL_Profiler_GraphicsContext_BeginPass_Fn)GetPerfworksProc("LWPW_OpenGL_Profiler_GraphicsContext_BeginPass", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_BeginPass);
    g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_EndPass = (LWPW_OpenGL_Profiler_GraphicsContext_EndPass_Fn)GetPerfworksProc("LWPW_OpenGL_Profiler_GraphicsContext_EndPass", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_EndPass);
    g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_PushRange = (LWPW_OpenGL_Profiler_GraphicsContext_PushRange_Fn)GetPerfworksProc("LWPW_OpenGL_Profiler_GraphicsContext_PushRange", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_PushRange);
    g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_PopRange = (LWPW_OpenGL_Profiler_GraphicsContext_PopRange_Fn)GetPerfworksProc("LWPW_OpenGL_Profiler_GraphicsContext_PopRange", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_PopRange);
    g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_DecodeCounters = (LWPW_OpenGL_Profiler_GraphicsContext_DecodeCounters_Fn)GetPerfworksProc("LWPW_OpenGL_Profiler_GraphicsContext_DecodeCounters", (LWPA_GenericFn)g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_DecodeCounters);
}
static LWPA_Status LWPA_InitializeHost_Default(void)
{
    InitPerfworks();
    if (g_api.fn.LWPA_InitializeHost != &LWPA_InitializeHost_Default && g_api.fn.LWPA_InitializeHost != &LWPA_InitializeHost)
    {
        return g_api.fn.LWPA_InitializeHost();
    }
    return g_defaultStatus;
}
static LWPA_Status LWPW_InitializeHost_Default(LWPW_InitializeHost_Params* pParams)
{
    InitPerfworks();
    if (g_api.fn.LWPW_InitializeHost != &LWPW_InitializeHost_Default && g_api.fn.LWPW_InitializeHost != &LWPW_InitializeHost)
    {
        return g_api.fn.LWPW_InitializeHost(pParams);
    }
    return g_defaultStatus;
}
static LWPA_Status LWPA_InitializeTarget_Default(void)
{
    InitPerfworks();
    if (g_api.fn.LWPA_InitializeTarget != &LWPA_InitializeTarget_Default && g_api.fn.LWPA_InitializeTarget != &LWPA_InitializeTarget)
    {
        return g_api.fn.LWPA_InitializeTarget();
    }
    return g_defaultStatus;
}
static LWPA_Status LWPW_InitializeTarget_Default(LWPW_InitializeTarget_Params* pParams)
{
    InitPerfworks();
    if (g_api.fn.LWPW_InitializeTarget != &LWPW_InitializeTarget_Default && g_api.fn.LWPW_InitializeTarget != &LWPW_InitializeTarget)
    {
        return g_api.fn.LWPW_InitializeTarget(pParams);
    }
    return g_defaultStatus;
}
LWPA_GenericFn LWPA_GetProcAddress(const char* pFunctionName)
{
    return g_api.fn.LWPA_GetProcAddress(pFunctionName);
}
LWPA_Status LWPW_SetLibraryLoadPaths(LWPW_SetLibraryLoadPaths_Params* pParams)
{
    return g_api.fn.LWPW_SetLibraryLoadPaths(pParams);
}
LWPA_Status LWPW_SetLibraryLoadPathsW(LWPW_SetLibraryLoadPathsW_Params* pParams)
{
    return g_api.fn.LWPW_SetLibraryLoadPathsW(pParams);
}
LWPA_Status LWPA_InitializeHost(void)
{
    return g_api.fn.LWPA_InitializeHost();
}
LWPA_Status LWPW_InitializeHost(LWPW_InitializeHost_Params* pParams)
{
    return g_api.fn.LWPW_InitializeHost(pParams);
}
LWPA_Status LWPA_CounterData_CallwlateCounterDataImageCopySize(const LWPA_CounterDataImageCopyOptions* pCounterDataImageCopyOptions, const uint8_t* pCounterDataSrc, size_t* pCopyDataImageCounterSize)
{
    return g_api.fn.LWPA_CounterData_CallwlateCounterDataImageCopySize(pCounterDataImageCopyOptions, pCounterDataSrc, pCopyDataImageCounterSize);
}
LWPA_Status LWPW_CounterData_CallwlateCounterDataImageCopySize(LWPW_CounterData_CallwlateCounterDataImageCopySize_Params* pParams)
{
    return g_api.fn.LWPW_CounterData_CallwlateCounterDataImageCopySize(pParams);
}
LWPA_Status LWPA_CounterData_InitializeCounterDataImageCopy(const LWPA_CounterDataImageCopyOptions* pCounterDataImageCopyOptions, const uint8_t* pCounterDataSrc, uint8_t* pCounterDataDst)
{
    return g_api.fn.LWPA_CounterData_InitializeCounterDataImageCopy(pCounterDataImageCopyOptions, pCounterDataSrc, pCounterDataDst);
}
LWPA_Status LWPW_CounterData_InitializeCounterDataImageCopy(LWPW_CounterData_InitializeCounterDataImageCopy_Params* pParams)
{
    return g_api.fn.LWPW_CounterData_InitializeCounterDataImageCopy(pParams);
}
LWPA_Status LWPA_CounterDataCombiner_Create(const LWPA_CounterDataCombinerOptions* pCounterDataCombinerOptions, LWPA_CounterDataCombiner** ppCounterDataCombiner)
{
    return g_api.fn.LWPA_CounterDataCombiner_Create(pCounterDataCombinerOptions, ppCounterDataCombiner);
}
LWPA_Status LWPW_CounterDataCombiner_Create(LWPW_CounterDataCombiner_Create_Params* pParams)
{
    return g_api.fn.LWPW_CounterDataCombiner_Create(pParams);
}
LWPA_Status LWPA_CounterDataCombiner_Destroy(LWPA_CounterDataCombiner* pCounterDataCombiner)
{
    return g_api.fn.LWPA_CounterDataCombiner_Destroy(pCounterDataCombiner);
}
LWPA_Status LWPW_CounterDataCombiner_Destroy(LWPW_CounterDataCombiner_Destroy_Params* pParams)
{
    return g_api.fn.LWPW_CounterDataCombiner_Destroy(pParams);
}
LWPA_Status LWPA_CounterDataCombiner_CreateRange(LWPA_CounterDataCombiner* pCounterDataCombiner, size_t numDescriptions, const char* const* ppDescriptions, size_t* pRangeIndexDst)
{
    return g_api.fn.LWPA_CounterDataCombiner_CreateRange(pCounterDataCombiner, numDescriptions, ppDescriptions, pRangeIndexDst);
}
LWPA_Status LWPW_CounterDataCombiner_CreateRange(LWPW_CounterDataCombiner_CreateRange_Params* pParams)
{
    return g_api.fn.LWPW_CounterDataCombiner_CreateRange(pParams);
}
LWPA_Status LWPA_CounterDataCombiner_AclwmulateIntoRange(LWPA_CounterDataCombiner* pCounterDataCombiner, size_t rangeIndexDst, uint32_t dstMultiplier, const uint8_t* pCounterDataSrc, size_t rangeIndexSrc, uint32_t srcMultiplier)
{
    return g_api.fn.LWPA_CounterDataCombiner_AclwmulateIntoRange(pCounterDataCombiner, rangeIndexDst, dstMultiplier, pCounterDataSrc, rangeIndexSrc, srcMultiplier);
}
LWPA_Status LWPW_CounterDataCombiner_AclwmulateIntoRange(LWPW_CounterDataCombiner_AclwmulateIntoRange_Params* pParams)
{
    return g_api.fn.LWPW_CounterDataCombiner_AclwmulateIntoRange(pParams);
}
LWPA_Status LWPW_CounterDataCombiner_SumIntoRange(LWPW_CounterDataCombiner_SumIntoRange_Params* pParams)
{
    return g_api.fn.LWPW_CounterDataCombiner_SumIntoRange(pParams);
}
LWPA_Status LWPW_CounterDataCombiner_WeightedSumIntoRange(LWPW_CounterDataCombiner_WeightedSumIntoRange_Params* pParams)
{
    return g_api.fn.LWPW_CounterDataCombiner_WeightedSumIntoRange(pParams);
}
LWPA_Status LWPA_GetSupportedChipNames(LWPA_SupportedChipNames* pSupportedChipNames)
{
    return g_api.fn.LWPA_GetSupportedChipNames(pSupportedChipNames);
}
LWPA_Status LWPW_GetSupportedChipNames(LWPW_GetSupportedChipNames_Params* pParams)
{
    return g_api.fn.LWPW_GetSupportedChipNames(pParams);
}
LWPA_Status LWPA_RawMetricsConfig_Create(const LWPA_RawMetricsConfigOptions* pMetricsConfigOptions, LWPA_RawMetricsConfig** ppRawMetricsConfig)
{
    return g_api.fn.LWPA_RawMetricsConfig_Create(pMetricsConfigOptions, ppRawMetricsConfig);
}
LWPA_Status LWPA_RawMetricsConfig_Destroy(LWPA_RawMetricsConfig* pRawMetricsConfig)
{
    return g_api.fn.LWPA_RawMetricsConfig_Destroy(pRawMetricsConfig);
}
LWPA_Status LWPW_RawMetricsConfig_Destroy(LWPW_RawMetricsConfig_Destroy_Params* pParams)
{
    return g_api.fn.LWPW_RawMetricsConfig_Destroy(pParams);
}
LWPA_Status LWPA_RawMetricsConfig_BeginPassGroup(LWPA_RawMetricsConfig* pRawMetricsConfig, const LWPA_RawMetricsPassGroupOptions* pRawMetricsPassGroupOptions)
{
    return g_api.fn.LWPA_RawMetricsConfig_BeginPassGroup(pRawMetricsConfig, pRawMetricsPassGroupOptions);
}
LWPA_Status LWPW_RawMetricsConfig_BeginPassGroup(LWPW_RawMetricsConfig_BeginPassGroup_Params* pParams)
{
    return g_api.fn.LWPW_RawMetricsConfig_BeginPassGroup(pParams);
}
LWPA_Status LWPA_RawMetricsConfig_EndPassGroup(LWPA_RawMetricsConfig* pRawMetricsConfig)
{
    return g_api.fn.LWPA_RawMetricsConfig_EndPassGroup(pRawMetricsConfig);
}
LWPA_Status LWPW_RawMetricsConfig_EndPassGroup(LWPW_RawMetricsConfig_EndPassGroup_Params* pParams)
{
    return g_api.fn.LWPW_RawMetricsConfig_EndPassGroup(pParams);
}
LWPA_Status LWPA_RawMetricsConfig_GetNumMetrics(const LWPA_RawMetricsConfig* pRawMetricsConfig, size_t* pNumMetrics)
{
    return g_api.fn.LWPA_RawMetricsConfig_GetNumMetrics(pRawMetricsConfig, pNumMetrics);
}
LWPA_Status LWPW_RawMetricsConfig_GetNumMetrics(LWPW_RawMetricsConfig_GetNumMetrics_Params* pParams)
{
    return g_api.fn.LWPW_RawMetricsConfig_GetNumMetrics(pParams);
}
LWPA_Status LWPA_RawMetricsConfig_GetMetricProperties(const LWPA_RawMetricsConfig* pRawMetricsConfig, size_t metricIndex, LWPA_RawMetricProperties* pRawMetricProperties)
{
    return g_api.fn.LWPA_RawMetricsConfig_GetMetricProperties(pRawMetricsConfig, metricIndex, pRawMetricProperties);
}
LWPA_Status LWPW_RawMetricsConfig_GetMetricProperties(LWPW_RawMetricsConfig_GetMetricProperties_Params* pParams)
{
    return g_api.fn.LWPW_RawMetricsConfig_GetMetricProperties(pParams);
}
LWPA_Status LWPA_RawMetricsConfig_AddMetrics(LWPA_RawMetricsConfig* pRawMetricsConfig, const LWPA_RawMetricRequest* pRawMetricRequests, size_t numMetricRequests)
{
    return g_api.fn.LWPA_RawMetricsConfig_AddMetrics(pRawMetricsConfig, pRawMetricRequests, numMetricRequests);
}
LWPA_Status LWPW_RawMetricsConfig_AddMetrics(LWPW_RawMetricsConfig_AddMetrics_Params* pParams)
{
    return g_api.fn.LWPW_RawMetricsConfig_AddMetrics(pParams);
}
LWPA_Status LWPA_RawMetricsConfig_IsAddMetricsPossible(const LWPA_RawMetricsConfig* pRawMetricsConfig, const LWPA_RawMetricRequest* pRawMetricRequests, size_t numMetricRequests, LWPA_Bool* pIsPossible)
{
    return g_api.fn.LWPA_RawMetricsConfig_IsAddMetricsPossible(pRawMetricsConfig, pRawMetricRequests, numMetricRequests, pIsPossible);
}
LWPA_Status LWPW_RawMetricsConfig_IsAddMetricsPossible(LWPW_RawMetricsConfig_IsAddMetricsPossible_Params* pParams)
{
    return g_api.fn.LWPW_RawMetricsConfig_IsAddMetricsPossible(pParams);
}
LWPA_Status LWPA_RawMetricsConfig_GenerateConfigImage(LWPA_RawMetricsConfig* pRawMetricsConfig)
{
    return g_api.fn.LWPA_RawMetricsConfig_GenerateConfigImage(pRawMetricsConfig);
}
LWPA_Status LWPW_RawMetricsConfig_GenerateConfigImage(LWPW_RawMetricsConfig_GenerateConfigImage_Params* pParams)
{
    return g_api.fn.LWPW_RawMetricsConfig_GenerateConfigImage(pParams);
}
LWPA_Status LWPA_RawMetricsConfig_GetConfigImage(const LWPA_RawMetricsConfig* pRawMetricsConfig, size_t bufferSize, uint8_t* pBuffer, size_t* pBufferSize)
{
    return g_api.fn.LWPA_RawMetricsConfig_GetConfigImage(pRawMetricsConfig, bufferSize, pBuffer, pBufferSize);
}
LWPA_Status LWPW_RawMetricsConfig_GetConfigImage(LWPW_RawMetricsConfig_GetConfigImage_Params* pParams)
{
    return g_api.fn.LWPW_RawMetricsConfig_GetConfigImage(pParams);
}
LWPA_Status LWPA_RawMetricsConfig_GetNumPasses(const LWPA_RawMetricsConfig* pRawMetricsConfig, size_t* pNumPipelinedPasses, size_t* pNumIsolatedPasses)
{
    return g_api.fn.LWPA_RawMetricsConfig_GetNumPasses(pRawMetricsConfig, pNumPipelinedPasses, pNumIsolatedPasses);
}
LWPA_Status LWPW_RawMetricsConfig_GetNumPasses(LWPW_RawMetricsConfig_GetNumPasses_Params* pParams)
{
    return g_api.fn.LWPW_RawMetricsConfig_GetNumPasses(pParams);
}
LWPA_Status LWPA_CounterDataBuilder_Create(const LWPA_CounterDataBuilderOptions* pOptions, LWPA_CounterDataBuilder** ppCounterDataBuilder)
{
    return g_api.fn.LWPA_CounterDataBuilder_Create(pOptions, ppCounterDataBuilder);
}
LWPA_Status LWPW_CounterDataBuilder_Create(LWPW_CounterDataBuilder_Create_Params* pParams)
{
    return g_api.fn.LWPW_CounterDataBuilder_Create(pParams);
}
LWPA_Status LWPA_CounterDataBuilder_Destroy(LWPA_CounterDataBuilder* pCounterDataBuilder)
{
    return g_api.fn.LWPA_CounterDataBuilder_Destroy(pCounterDataBuilder);
}
LWPA_Status LWPW_CounterDataBuilder_Destroy(LWPW_CounterDataBuilder_Destroy_Params* pParams)
{
    return g_api.fn.LWPW_CounterDataBuilder_Destroy(pParams);
}
LWPA_Status LWPA_CounterDataBuilder_AddMetrics(LWPA_CounterDataBuilder* pCounterDataBuilder, const LWPA_RawMetricRequest* pRawMetricRequests, size_t numMetricRequests)
{
    return g_api.fn.LWPA_CounterDataBuilder_AddMetrics(pCounterDataBuilder, pRawMetricRequests, numMetricRequests);
}
LWPA_Status LWPW_CounterDataBuilder_AddMetrics(LWPW_CounterDataBuilder_AddMetrics_Params* pParams)
{
    return g_api.fn.LWPW_CounterDataBuilder_AddMetrics(pParams);
}
LWPA_Status LWPA_CounterDataBuilder_GetCounterDataPrefix(LWPA_CounterDataBuilder* pCounterDataBuilder, size_t bufferSize, uint8_t* pBuffer, size_t* pBufferSize)
{
    return g_api.fn.LWPA_CounterDataBuilder_GetCounterDataPrefix(pCounterDataBuilder, bufferSize, pBuffer, pBufferSize);
}
LWPA_Status LWPW_CounterDataBuilder_GetCounterDataPrefix(LWPW_CounterDataBuilder_GetCounterDataPrefix_Params* pParams)
{
    return g_api.fn.LWPW_CounterDataBuilder_GetCounterDataPrefix(pParams);
}
LWPA_Status LWPA_MetricsContext_Create(const LWPA_MetricsContextOptions* pMetricsContextOptions, LWPA_MetricsContext** ppMetricsContext)
{
    return g_api.fn.LWPA_MetricsContext_Create(pMetricsContextOptions, ppMetricsContext);
}
LWPA_Status LWPA_MetricsContext_Destroy(LWPA_MetricsContext* pMetricsContext)
{
    return g_api.fn.LWPA_MetricsContext_Destroy(pMetricsContext);
}
LWPA_Status LWPW_MetricsContext_Destroy(LWPW_MetricsContext_Destroy_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_Destroy(pParams);
}
LWPA_Status LWPA_MetricsContext_RunScript(LWPA_MetricsContext* pMetricsContext, const LWPA_MetricsScriptOptions* pOptions)
{
    return g_api.fn.LWPA_MetricsContext_RunScript(pMetricsContext, pOptions);
}
LWPA_Status LWPW_MetricsContext_RunScript(LWPW_MetricsContext_RunScript_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_RunScript(pParams);
}
LWPA_Status LWPA_MetricsContext_ExecScript_Begin(LWPA_MetricsContext* pMetricsContext, LWPA_MetricsExecOptions* pOptions)
{
    return g_api.fn.LWPA_MetricsContext_ExecScript_Begin(pMetricsContext, pOptions);
}
LWPA_Status LWPW_MetricsContext_ExecScript_Begin(LWPW_MetricsContext_ExecScript_Begin_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_ExecScript_Begin(pParams);
}
LWPA_Status LWPA_MetricsContext_ExecScript_End(LWPA_MetricsContext* pMetricsContext)
{
    return g_api.fn.LWPA_MetricsContext_ExecScript_End(pMetricsContext);
}
LWPA_Status LWPW_MetricsContext_ExecScript_End(LWPW_MetricsContext_ExecScript_End_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_ExecScript_End(pParams);
}
LWPA_Status LWPA_MetricsContext_GetCounterNames_Begin(LWPA_MetricsContext* pMetricsContext, size_t* pNumCounters, const char* const** pppCounterNames)
{
    return g_api.fn.LWPA_MetricsContext_GetCounterNames_Begin(pMetricsContext, pNumCounters, pppCounterNames);
}
LWPA_Status LWPW_MetricsContext_GetCounterNames_Begin(LWPW_MetricsContext_GetCounterNames_Begin_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetCounterNames_Begin(pParams);
}
LWPA_Status LWPA_MetricsContext_GetCounterNames_End(LWPA_MetricsContext* pMetricsContext)
{
    return g_api.fn.LWPA_MetricsContext_GetCounterNames_End(pMetricsContext);
}
LWPA_Status LWPW_MetricsContext_GetCounterNames_End(LWPW_MetricsContext_GetCounterNames_End_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetCounterNames_End(pParams);
}
LWPA_Status LWPA_MetricsContext_GetThroughputNames_Begin(LWPA_MetricsContext* pMetricsContext, size_t* pNumThroughputs, const char* const** pppThroughputName)
{
    return g_api.fn.LWPA_MetricsContext_GetThroughputNames_Begin(pMetricsContext, pNumThroughputs, pppThroughputName);
}
LWPA_Status LWPW_MetricsContext_GetThroughputNames_Begin(LWPW_MetricsContext_GetThroughputNames_Begin_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetThroughputNames_Begin(pParams);
}
LWPA_Status LWPA_MetricsContext_GetThroughputNames_End(LWPA_MetricsContext* pMetricsContext)
{
    return g_api.fn.LWPA_MetricsContext_GetThroughputNames_End(pMetricsContext);
}
LWPA_Status LWPW_MetricsContext_GetThroughputNames_End(LWPW_MetricsContext_GetThroughputNames_End_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetThroughputNames_End(pParams);
}
LWPA_Status LWPW_MetricsContext_GetRatioNames_Begin(LWPW_MetricsContext_GetRatioNames_Begin_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetRatioNames_Begin(pParams);
}
LWPA_Status LWPW_MetricsContext_GetRatioNames_End(LWPW_MetricsContext_GetRatioNames_End_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetRatioNames_End(pParams);
}
LWPA_Status LWPA_MetricsContext_GetMetricNames_Begin(LWPA_MetricsContext* pMetricsContext, LWPA_MetricsEnumerationOptions* pOptions)
{
    return g_api.fn.LWPA_MetricsContext_GetMetricNames_Begin(pMetricsContext, pOptions);
}
LWPA_Status LWPW_MetricsContext_GetMetricNames_Begin(LWPW_MetricsContext_GetMetricNames_Begin_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetMetricNames_Begin(pParams);
}
LWPA_Status LWPA_MetricsContext_GetMetricNames_End(LWPA_MetricsContext* pMetricsContext)
{
    return g_api.fn.LWPA_MetricsContext_GetMetricNames_End(pMetricsContext);
}
LWPA_Status LWPW_MetricsContext_GetMetricNames_End(LWPW_MetricsContext_GetMetricNames_End_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetMetricNames_End(pParams);
}
LWPA_Status LWPA_MetricsContext_GetThroughputBreakdown_Begin(LWPA_MetricsContext* pMetricsContext, const char* pThroughputName, const char* const** pppCounterNames, const char* const** pppSubThroughputNames)
{
    return g_api.fn.LWPA_MetricsContext_GetThroughputBreakdown_Begin(pMetricsContext, pThroughputName, pppCounterNames, pppSubThroughputNames);
}
LWPA_Status LWPW_MetricsContext_GetThroughputBreakdown_Begin(LWPW_MetricsContext_GetThroughputBreakdown_Begin_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetThroughputBreakdown_Begin(pParams);
}
LWPA_Status LWPA_MetricsContext_GetThroughputBreakdown_End(LWPA_MetricsContext* pMetricsContext)
{
    return g_api.fn.LWPA_MetricsContext_GetThroughputBreakdown_End(pMetricsContext);
}
LWPA_Status LWPW_MetricsContext_GetThroughputBreakdown_End(LWPW_MetricsContext_GetThroughputBreakdown_End_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetThroughputBreakdown_End(pParams);
}
LWPA_Status LWPA_MetricsContext_GetMetricProperties_Begin(LWPA_MetricsContext* pMetricsContext, const char* pMetricName, LWPA_MetricProperties* pMetricProperties)
{
    return g_api.fn.LWPA_MetricsContext_GetMetricProperties_Begin(pMetricsContext, pMetricName, pMetricProperties);
}
LWPA_Status LWPW_MetricsContext_GetMetricProperties_Begin(LWPW_MetricsContext_GetMetricProperties_Begin_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetMetricProperties_Begin(pParams);
}
LWPA_Status LWPA_MetricsContext_GetMetricProperties_End(LWPA_MetricsContext* pMetricsContext)
{
    return g_api.fn.LWPA_MetricsContext_GetMetricProperties_End(pMetricsContext);
}
LWPA_Status LWPW_MetricsContext_GetMetricProperties_End(LWPW_MetricsContext_GetMetricProperties_End_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetMetricProperties_End(pParams);
}
LWPA_Status LWPA_MetricsContext_SetCounterData(LWPA_MetricsContext* pMetricsContext, const uint8_t* pCounterDataImage, size_t rangeIndex, LWPA_Bool isolated)
{
    return g_api.fn.LWPA_MetricsContext_SetCounterData(pMetricsContext, pCounterDataImage, rangeIndex, isolated);
}
LWPA_Status LWPW_MetricsContext_SetCounterData(LWPW_MetricsContext_SetCounterData_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_SetCounterData(pParams);
}
LWPA_Status LWPA_MetricsContext_SetUserData(LWPA_MetricsContext* pMetricsContext, const LWPA_MetrilwserData* pMetrilwserData)
{
    return g_api.fn.LWPA_MetricsContext_SetUserData(pMetricsContext, pMetrilwserData);
}
LWPA_Status LWPW_MetricsContext_SetUserData(LWPW_MetricsContext_SetUserData_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_SetUserData(pParams);
}
LWPA_Status LWPA_MetricsContext_EvaluateToGpuValues(LWPA_MetricsContext* pMetricsContext, size_t numMetrics, const char* const* ppMetricNames, double* pMetricValues)
{
    return g_api.fn.LWPA_MetricsContext_EvaluateToGpuValues(pMetricsContext, numMetrics, ppMetricNames, pMetricValues);
}
LWPA_Status LWPW_MetricsContext_EvaluateToGpuValues(LWPW_MetricsContext_EvaluateToGpuValues_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_EvaluateToGpuValues(pParams);
}
LWPA_Status LWPW_MetricsContext_GetMetricSuffix_Begin(LWPW_MetricsContext_GetMetricSuffix_Begin_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetMetricSuffix_Begin(pParams);
}
LWPA_Status LWPW_MetricsContext_GetMetricSuffix_End(LWPW_MetricsContext_GetMetricSuffix_End_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetMetricSuffix_End(pParams);
}
LWPA_Status LWPW_MetricsContext_GetMetricBaseNames_Begin(LWPW_MetricsContext_GetMetricBaseNames_Begin_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetMetricBaseNames_Begin(pParams);
}
LWPA_Status LWPW_MetricsContext_GetMetricBaseNames_End(LWPW_MetricsContext_GetMetricBaseNames_End_Params* pParams)
{
    return g_api.fn.LWPW_MetricsContext_GetMetricBaseNames_End(pParams);
}
LWPA_Status LWPA_InitializeTarget(void)
{
    return g_api.fn.LWPA_InitializeTarget();
}
LWPA_Status LWPA_GetDeviceCount(size_t* pNumDevices)
{
    return g_api.fn.LWPA_GetDeviceCount(pNumDevices);
}
LWPA_Status LWPA_Device_GetNames(size_t deviceIndex, const char** ppDeviceName, const char** ppChipName)
{
    return g_api.fn.LWPA_Device_GetNames(deviceIndex, ppDeviceName, ppChipName);
}
LWPA_Status LWPA_CounterData_GetNumRanges(const uint8_t* pCounterDataImage, size_t* pNumRanges)
{
    return g_api.fn.LWPA_CounterData_GetNumRanges(pCounterDataImage, pNumRanges);
}
LWPA_Status LWPA_CounterData_GetRangeDescriptions(const uint8_t* pCounterDataImage, size_t rangeIndex, size_t numDescriptions, const char** ppDescriptions, size_t* pNumDescriptions)
{
    return g_api.fn.LWPA_CounterData_GetRangeDescriptions(pCounterDataImage, rangeIndex, numDescriptions, ppDescriptions, pNumDescriptions);
}
LWPA_Status LWPW_InitializeTarget(LWPW_InitializeTarget_Params* pParams)
{
    return g_api.fn.LWPW_InitializeTarget(pParams);
}
LWPA_Status LWPW_GetDeviceCount(LWPW_GetDeviceCount_Params* pParams)
{
    return g_api.fn.LWPW_GetDeviceCount(pParams);
}
LWPA_Status LWPW_Device_GetNames(LWPW_Device_GetNames_Params* pParams)
{
    return g_api.fn.LWPW_Device_GetNames(pParams);
}
LWPA_Status LWPW_Device_GetPciBusIds(LWPW_Device_GetPciBusIds_Params* pParams)
{
    return g_api.fn.LWPW_Device_GetPciBusIds(pParams);
}
LWPA_Status LWPW_Adapter_GetDeviceIndex(LWPW_Adapter_GetDeviceIndex_Params* pParams)
{
    return g_api.fn.LWPW_Adapter_GetDeviceIndex(pParams);
}
LWPA_Status LWPW_CounterData_GetNumRanges(LWPW_CounterData_GetNumRanges_Params* pParams)
{
    return g_api.fn.LWPW_CounterData_GetNumRanges(pParams);
}
LWPA_Status LWPW_Config_GetNumPasses(LWPW_Config_GetNumPasses_Params* pParams)
{
    return g_api.fn.LWPW_Config_GetNumPasses(pParams);
}
LWPA_Status LWPW_CounterData_GetRangeDescriptions(LWPW_CounterData_GetRangeDescriptions_Params* pParams)
{
    return g_api.fn.LWPW_CounterData_GetRangeDescriptions(pParams);
}
LWPA_Status LWPW_Profiler_CounterData_GetRangeDescriptions(LWPW_Profiler_CounterData_GetRangeDescriptions_Params* pParams)
{
    return g_api.fn.LWPW_Profiler_CounterData_GetRangeDescriptions(pParams);
}
LWPA_Status LWPW_PeriodicSampler_CounterData_GetDelimiters(LWPW_PeriodicSampler_CounterData_GetDelimiters_Params* pParams)
{
    return g_api.fn.LWPW_PeriodicSampler_CounterData_GetDelimiters(pParams);
}
LWPA_Status LWPW_PeriodicSampler_CounterData_GetSampleTime(LWPW_PeriodicSampler_CounterData_GetSampleTime_Params* pParams)
{
    return g_api.fn.LWPW_PeriodicSampler_CounterData_GetSampleTime(pParams);
}
LWPA_Status LWPW_PeriodicSampler_CounterData_TrimInPlace(LWPW_PeriodicSampler_CounterData_TrimInPlace_Params* pParams)
{
    return g_api.fn.LWPW_PeriodicSampler_CounterData_TrimInPlace(pParams);
}
LWPA_Status LWPW_LWDA_MetricsContext_Create(LWPW_LWDA_MetricsContext_Create_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_MetricsContext_Create(pParams);
}
LWPA_Status LWPW_LWDA_RawMetricsConfig_Create(LWPW_LWDA_RawMetricsConfig_Create_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_RawMetricsConfig_Create(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_CounterDataImage_CallwlateSize(LWPW_LWDA_Profiler_CounterDataImage_CallwlateSize_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_CounterDataImage_CallwlateSize(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_CounterDataImage_Initialize(LWPW_LWDA_Profiler_CounterDataImage_Initialize_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_CounterDataImage_Initialize(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_CounterDataImage_CallwlateScratchBufferSize(LWPW_LWDA_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_CounterDataImage_CallwlateScratchBufferSize(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_CounterDataImage_InitializeScratchBuffer(LWPW_LWDA_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_CounterDataImage_InitializeScratchBuffer(pParams);
}
LWPA_Status LWPW_LWDA_GetDeviceOrdinals(LWPW_LWDA_GetDeviceOrdinals_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_GetDeviceOrdinals(pParams);
}
LWPA_Status LWPW_LWDA_LoadDriver(LWPW_LWDA_LoadDriver_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_LoadDriver(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_CalcTraceBufferSize(LWPW_LWDA_Profiler_CalcTraceBufferSize_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_CalcTraceBufferSize(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_BeginSession(LWPW_LWDA_Profiler_BeginSession_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_BeginSession(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_EndSession(LWPW_LWDA_Profiler_EndSession_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_EndSession(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_SetConfig(LWPW_LWDA_Profiler_SetConfig_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_SetConfig(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_ClearConfig(LWPW_LWDA_Profiler_ClearConfig_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_ClearConfig(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_BeginPass(LWPW_LWDA_Profiler_BeginPass_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_BeginPass(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_EndPass(LWPW_LWDA_Profiler_EndPass_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_EndPass(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_DecodeCounters(LWPW_LWDA_Profiler_DecodeCounters_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_DecodeCounters(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_EnablePerLaunchProfiling(LWPW_LWDA_Profiler_EnablePerLaunchProfiling_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_EnablePerLaunchProfiling(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_DisablePerLaunchProfiling(LWPW_LWDA_Profiler_DisablePerLaunchProfiling_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_DisablePerLaunchProfiling(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_PushRange(LWPW_LWDA_Profiler_PushRange_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_PushRange(pParams);
}
LWPA_Status LWPW_LWDA_Profiler_PopRange(LWPW_LWDA_Profiler_PopRange_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_PopRange(pParams);
}
LWPA_Status LWPW_VK_MetricsContext_Create(LWPW_VK_MetricsContext_Create_Params* pParams)
{
    return g_api.fn.LWPW_VK_MetricsContext_Create(pParams);
}
LWPA_Status LWPW_VK_RawMetricsConfig_Create(LWPW_VK_RawMetricsConfig_Create_Params* pParams)
{
    return g_api.fn.LWPW_VK_RawMetricsConfig_Create(pParams);
}
LWPA_Status LWPW_VK_Profiler_CounterDataImage_CallwlateSize(LWPW_VK_Profiler_CounterDataImage_CallwlateSize_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_CounterDataImage_CallwlateSize(pParams);
}
LWPA_Status LWPW_VK_Profiler_CounterDataImage_Initialize(LWPW_VK_Profiler_CounterDataImage_Initialize_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_CounterDataImage_Initialize(pParams);
}
LWPA_Status LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize(LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_CounterDataImage_CallwlateScratchBufferSize(pParams);
}
LWPA_Status LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer(LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer(pParams);
}
LWPA_Status LWPW_VK_LoadDriver(LWPW_VK_LoadDriver_Params* pParams)
{
    return g_api.fn.LWPW_VK_LoadDriver(pParams);
}
LWPA_Status LWPW_VK_Device_GetDeviceIndex(LWPW_VK_Device_GetDeviceIndex_Params* pParams)
{
    return g_api.fn.LWPW_VK_Device_GetDeviceIndex(pParams);
}
LWPA_Status LWPW_VK_Profiler_GetRequiredInstanceExtensions(LWPW_VK_Profiler_GetRequiredInstanceExtensions_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_GetRequiredInstanceExtensions(pParams);
}
LWPA_Status LWPW_VK_Profiler_GetRequiredDeviceExtensions(LWPW_VK_Profiler_GetRequiredDeviceExtensions_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_GetRequiredDeviceExtensions(pParams);
}
LWPA_Status LWPW_VK_Profiler_CalcTraceBufferSize(LWPW_VK_Profiler_CalcTraceBufferSize_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_CalcTraceBufferSize(pParams);
}
LWPA_Status LWPW_VK_Profiler_Queue_BeginSession(LWPW_VK_Profiler_Queue_BeginSession_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_Queue_BeginSession(pParams);
}
LWPA_Status LWPW_VK_Profiler_Queue_EndSession(LWPW_VK_Profiler_Queue_EndSession_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_Queue_EndSession(pParams);
}
LWPA_Status LWPW_VK_Queue_ServicePendingGpuOperations(LWPW_VK_Queue_ServicePendingGpuOperations_Params* pParams)
{
    return g_api.fn.LWPW_VK_Queue_ServicePendingGpuOperations(pParams);
}
LWPA_Status LWPW_VK_Profiler_Queue_SetConfig(LWPW_VK_Profiler_Queue_SetConfig_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_Queue_SetConfig(pParams);
}
LWPA_Status LWPW_VK_Profiler_Queue_ClearConfig(LWPW_VK_Profiler_Queue_ClearConfig_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_Queue_ClearConfig(pParams);
}
LWPA_Status LWPW_VK_Profiler_Queue_BeginPass(LWPW_VK_Profiler_Queue_BeginPass_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_Queue_BeginPass(pParams);
}
LWPA_Status LWPW_VK_Profiler_Queue_EndPass(LWPW_VK_Profiler_Queue_EndPass_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_Queue_EndPass(pParams);
}
LWPA_Status LWPW_VK_Profiler_CommandBuffer_PushRange(LWPW_VK_Profiler_CommandBuffer_PushRange_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_CommandBuffer_PushRange(pParams);
}
LWPA_Status LWPW_VK_Profiler_CommandBuffer_PopRange(LWPW_VK_Profiler_CommandBuffer_PopRange_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_CommandBuffer_PopRange(pParams);
}
LWPA_Status LWPW_VK_Profiler_Queue_DecodeCounters(LWPW_VK_Profiler_Queue_DecodeCounters_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_Queue_DecodeCounters(pParams);
}
LWPA_Status LWPW_VK_Profiler_IsGpuSupported(LWPW_VK_Profiler_IsGpuSupported_Params* pParams)
{
    return g_api.fn.LWPW_VK_Profiler_IsGpuSupported(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_CallwlateMemoryOverhead(LWPW_VK_PeriodicSampler_CallwlateMemoryOverhead_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_CallwlateMemoryOverhead(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_Device_CallwlateMemoryOverhead(LWPW_VK_PeriodicSampler_Device_CallwlateMemoryOverhead_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_Device_CallwlateMemoryOverhead(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_Queue_BeginSession(LWPW_VK_PeriodicSampler_Queue_BeginSession_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_Queue_BeginSession(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_Queue_EndSession(LWPW_VK_PeriodicSampler_Queue_EndSession_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_Queue_EndSession(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_CommandBuffer_StartSampling(LWPW_VK_PeriodicSampler_CommandBuffer_StartSampling_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_CommandBuffer_StartSampling(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_CommandBuffer_StopSampling(LWPW_VK_PeriodicSampler_CommandBuffer_StopSampling_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_CommandBuffer_StopSampling(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_CommandBuffer_InsertDelimiter(LWPW_VK_PeriodicSampler_CommandBuffer_InsertDelimiter_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_CommandBuffer_InsertDelimiter(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_CommandBuffer_BeginFrame(LWPW_VK_PeriodicSampler_CommandBuffer_BeginFrame_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_CommandBuffer_BeginFrame(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_CommandBuffer_InsertTrigger(LWPW_VK_PeriodicSampler_CommandBuffer_InsertTrigger_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_CommandBuffer_InsertTrigger(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_Queue_GetLastError(LWPW_VK_PeriodicSampler_Queue_GetLastError_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_Queue_GetLastError(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_CounterDataImage_CallwlateSize(LWPW_VK_PeriodicSampler_CounterDataImage_CallwlateSize_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_CounterDataImage_CallwlateSize(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_CounterDataImage_Initialize(LWPW_VK_PeriodicSampler_CounterDataImage_Initialize_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_CounterDataImage_Initialize(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_Queue_DecodeCounters(LWPW_VK_PeriodicSampler_Queue_DecodeCounters_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_Queue_DecodeCounters(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_IsGpuSupported(LWPW_VK_PeriodicSampler_IsGpuSupported_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_IsGpuSupported(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_Queue_DiscardFrame(LWPW_VK_PeriodicSampler_Queue_DiscardFrame_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_Queue_DiscardFrame(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_Queue_CallwlateRecordBufferSize(LWPW_VK_PeriodicSampler_Queue_CallwlateRecordBufferSize_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_Queue_CallwlateRecordBufferSize(pParams);
}
LWPA_Status LWPW_VK_PeriodicSampler_Queue_SetConfig(LWPW_VK_PeriodicSampler_Queue_SetConfig_Params* pParams)
{
    return g_api.fn.LWPW_VK_PeriodicSampler_Queue_SetConfig(pParams);
}
LWPA_Status LWPW_OpenGL_MetricsContext_Create(LWPW_OpenGL_MetricsContext_Create_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_MetricsContext_Create(pParams);
}
LWPA_Status LWPW_OpenGL_RawMetricsConfig_Create(LWPW_OpenGL_RawMetricsConfig_Create_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_RawMetricsConfig_Create(pParams);
}
LWPA_Status LWPW_OpenGL_LoadDriver(LWPW_OpenGL_LoadDriver_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_LoadDriver(pParams);
}
LWPA_Status LWPW_OpenGL_GetLwrrentGraphicsContext(LWPW_OpenGL_GetLwrrentGraphicsContext_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_GetLwrrentGraphicsContext(pParams);
}
LWPA_Status LWPW_OpenGL_GraphicsContext_GetDeviceIndex(LWPW_OpenGL_GraphicsContext_GetDeviceIndex_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_GraphicsContext_GetDeviceIndex(pParams);
}
LWPA_Status LWPW_OpenGL_Profiler_IsGpuSupported(LWPW_OpenGL_Profiler_IsGpuSupported_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_Profiler_IsGpuSupported(pParams);
}
LWPA_Status LWPW_OpenGL_Profiler_CounterDataImage_CallwlateSize(LWPW_OpenGL_Profiler_CounterDataImage_CallwlateSize_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_Profiler_CounterDataImage_CallwlateSize(pParams);
}
LWPA_Status LWPW_OpenGL_Profiler_CounterDataImage_Initialize(LWPW_OpenGL_Profiler_CounterDataImage_Initialize_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_Profiler_CounterDataImage_Initialize(pParams);
}
LWPA_Status LWPW_OpenGL_Profiler_CounterDataImage_CallwlateScratchBufferSize(LWPW_OpenGL_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_Profiler_CounterDataImage_CallwlateScratchBufferSize(pParams);
}
LWPA_Status LWPW_OpenGL_Profiler_CounterDataImage_InitializeScratchBuffer(LWPW_OpenGL_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_Profiler_CounterDataImage_InitializeScratchBuffer(pParams);
}
LWPA_Status LWPW_OpenGL_Profiler_CalcTraceBufferSize(LWPW_OpenGL_Profiler_CalcTraceBufferSize_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_Profiler_CalcTraceBufferSize(pParams);
}
LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_BeginSession(LWPW_OpenGL_Profiler_GraphicsContext_BeginSession_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_BeginSession(pParams);
}
LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_EndSession(LWPW_OpenGL_Profiler_GraphicsContext_EndSession_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_EndSession(pParams);
}
LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_SetConfig(LWPW_OpenGL_Profiler_GraphicsContext_SetConfig_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_SetConfig(pParams);
}
LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_ClearConfig(LWPW_OpenGL_Profiler_GraphicsContext_ClearConfig_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_ClearConfig(pParams);
}
LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_BeginPass(LWPW_OpenGL_Profiler_GraphicsContext_BeginPass_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_BeginPass(pParams);
}
LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_EndPass(LWPW_OpenGL_Profiler_GraphicsContext_EndPass_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_EndPass(pParams);
}
LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_PushRange(LWPW_OpenGL_Profiler_GraphicsContext_PushRange_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_PushRange(pParams);
}
LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_PopRange(LWPW_OpenGL_Profiler_GraphicsContext_PopRange_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_PopRange(pParams);
}
LWPA_Status LWPW_OpenGL_Profiler_GraphicsContext_DecodeCounters(LWPW_OpenGL_Profiler_GraphicsContext_DecodeCounters_Params* pParams)
{
    return g_api.fn.LWPW_OpenGL_Profiler_GraphicsContext_DecodeCounters(pParams);
}
static void* LibOpen(char const* name)
{
    int flags = RTLD_NOW | RTLD_GLOBAL;
#ifdef RTLD_DEEPBIND
    flags |= RTLD_DEEPBIND;
#endif
    return dlopen(name, flags);
}

static LWPA_GenericFn LibSym(void* module, char const* name)
{
    return (LWPA_GenericFn)dlsym(module, name);
}

static void* LoadPerfworksLibrary(void)
{
    char const* const pLibName = "liblwperf_dcgm_host.so";

    if (g_api.numSearchPaths == 0)
    {
        /* Load from default paths */
        void* hPerfworks = LibOpen(pLibName);
        if (hPerfworks)
        {
            return hPerfworks;
        }
    }
    else
    {
        size_t pathIndex = 0;
        size_t libNameLength = 0;
        size_t libPathLength = 0;
        char* pLibFullName = 0;
        void* hPerfworks = 0;

        for (pathIndex = 0; pathIndex < g_api.numSearchPaths; ++pathIndex)
        {
            if (!g_api.ppSearchPaths[pathIndex])
            {
                continue;
            }

            libNameLength = strlen(pLibName);
            libPathLength = strlen(g_api.ppSearchPaths[pathIndex]);

            pLibFullName = (char*)malloc(libNameLength + libPathLength + 2);
            if (!pLibFullName)
            {
                continue;
            }

            strncpy(pLibFullName, g_api.ppSearchPaths[pathIndex], libPathLength);
            pLibFullName[libPathLength] = '/';

            strncpy(pLibFullName + libPathLength + 1, pLibName, libNameLength);
            pLibFullName[libPathLength + libNameLength + 1] = '\0';

            hPerfworks = LibOpen(pLibFullName);
            free(pLibFullName);
            if (hPerfworks)
            {
                return hPerfworks;
            }
        }
    }
    return 0;
}

/* Returns 0 on failure, 1 on success. */
static int InitPerfworks(void)
{
    if (!g_api.hModPerfworks)
    {
        g_api.hModPerfworks = LoadPerfworksLibrary();
        if (!g_api.hModPerfworks)
        {
            return 0;
        }
    }

    g_defaultStatus = LWPA_STATUS_FUNCTION_NOT_FOUND;
    g_api.perfworksGetProcAddress = (LWPA_GetProcAddress_Fn)LibSym(g_api.hModPerfworks, "LWPA_GetProcAddress");
    if (!g_api.perfworksGetProcAddress)
    {
        return 0;
    }
    g_api.perfworksGetProcAddress = (LWPA_GetProcAddress_Fn)g_api.perfworksGetProcAddress("1111");

    InitPerfworksProcs();
    return 1;
}
static LWPA_GenericFn GetPerfworksProc(char const* pName, LWPA_GenericFn pDefault)
{
    LWPA_GenericFn pProc = g_api.perfworksGetProcAddress(pName);
    if (pProc)
    {
        return pProc;
    }
    return pDefault;
}

static void FreeSearchPaths(void)
{
    if (g_api.ppSearchPaths)
    {
        size_t index;
        for (index = 0; index < g_api.numSearchPaths; ++index)
        {
            free(g_api.ppSearchPaths[index]);
        }
        free(g_api.ppSearchPaths);
        g_api.ppSearchPaths = NULL;
        g_api.numSearchPaths = 0;
    }
}

static LWPA_Status LWPW_SetLibraryLoadPaths_Default(LWPW_SetLibraryLoadPaths_Params* pParams)
{
    size_t index;

    /* free the old paths */
    FreeSearchPaths();

    if (pParams->numPaths == 0 || pParams->ppPaths == NULL)
    {
        return LWPA_STATUS_SUCCESS;
    }

    #ifdef _MSC_VER
    #pragma warning( push )
    #pragma warning( disable : 6385 )
    #endif

    g_api.numSearchPaths = pParams->numPaths;
    g_api.ppSearchPaths = (LWPW_User_PathCharType**)malloc(pParams->numPaths * sizeof(LWPW_User_PathCharType*));
    if (!g_api.ppSearchPaths)
    {
        return LWPA_STATUS_OUT_OF_MEMORY;
    }
    memset(g_api.ppSearchPaths, 0, pParams->numPaths * sizeof(LWPW_User_PathCharType*));

    for (index = 0; index < pParams->numPaths; ++index)
    {
        size_t len = strlen(pParams->ppPaths[index]) + 1;
        g_api.ppSearchPaths[index] = (LWPW_User_PathCharType*)malloc((len) * sizeof(LWPW_User_PathCharType));
        if (!g_api.ppSearchPaths[index])
        {
            return LWPA_STATUS_OUT_OF_MEMORY;
        }
#if defined(_WIN32)
        size_t numColwerted;
        mbstowcs_s(&numColwerted, g_api.ppSearchPaths[index], len, pParams->ppPaths[index], len);
#else
        strncpy(g_api.ppSearchPaths[index], pParams->ppPaths[index], len);
#endif
    }

    #ifdef _MSC_VER
    #pragma warning( pop )
    #endif

    return LWPA_STATUS_SUCCESS;
}

static LWPA_Status LWPW_SetLibraryLoadPathsW_Default(LWPW_SetLibraryLoadPathsW_Params* pParams)
{
    size_t index;

    /* free the old paths */
    FreeSearchPaths();

    if (pParams->numPaths == 0 || pParams->ppwPaths == NULL)
    {
        return LWPA_STATUS_SUCCESS;
    }

    #ifdef _MSC_VER
    #pragma warning( push )
    #pragma warning( disable : 6385 )
    #endif

    g_api.numSearchPaths = pParams->numPaths;
    g_api.ppSearchPaths = (LWPW_User_PathCharType**)malloc(pParams->numPaths * sizeof(LWPW_User_PathCharType*));
    if (!g_api.ppSearchPaths)
    {
        return LWPA_STATUS_OUT_OF_MEMORY;
    }
    memset(g_api.ppSearchPaths, 0, pParams->numPaths * sizeof(LWPW_User_PathCharType*));

    for (index = 0; index < pParams->numPaths; ++index)
    {
        /* callwlate the length of the dest */
#if defined(_WIN32)
        size_t len = wcslen(pParams->ppwPaths[index]) + 1;
#else
        size_t len = wcstombs(NULL, pParams->ppwPaths[index], 0) + 1;
#endif
        /* allocate the dest buffer */
        g_api.ppSearchPaths[index] = (LWPW_User_PathCharType*)malloc((len) * sizeof(LWPW_User_PathCharType));
        if (!g_api.ppSearchPaths[index])
        {
            return LWPA_STATUS_OUT_OF_MEMORY;
        }
        /* copy/colwert the source to dest */
#if defined(_WIN32)
        wcsncpy_s(g_api.ppSearchPaths[index], len, pParams->ppwPaths[index], len);
#else
        wcstombs(g_api.ppSearchPaths[index], pParams->ppwPaths[index], len);
#endif
    }

    #ifdef _MSC_VER
    #pragma warning( pop )
    #endif

    return LWPA_STATUS_SUCCESS;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
