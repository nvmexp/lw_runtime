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

#include <lwperf_target.h>
#include <lwperf_target_priv.h>
#include <lwperf_lwda_target.h>
#include <lwperf_lwda_target_priv.h>
#include <lwperf_dcgm_target.h>
#include <lwperf_dcgm_target_priv.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef LWPA_GenericFn (*LWPA_GetProcAddress_Fn)(const char* pFunctionName);
typedef LWPA_Status (*LWPW_SetLibraryLoadPaths_Fn)(LWPW_SetLibraryLoadPaths_Params* pParams);
typedef LWPA_Status (*LWPW_SetLibraryLoadPathsW_Fn)(LWPW_SetLibraryLoadPathsW_Params* pParams);
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
typedef LWPA_Status (*LWPW_Device_GetSmHierarchy_Fn)(LWPW_Device_GetSmHierarchy_Params* pParams);
typedef LWPA_Status (*LWPW_Device_GetVsmMappings_Fn)(LWPW_Device_GetVsmMappings_Params* pParams);
typedef LWPA_Status (*LWPW_Event_CreateEventStream_Fn)(LWPW_Event_CreateEventStream_Params* pParams);
typedef LWPA_Status (*LWPW_Event_DestroyEventStream_Fn)(LWPW_Event_DestroyEventStream_Params* pParams);
typedef LWPA_Status (*LWPW_Event_SetProducerMode_Fn)(LWPW_Event_SetProducerMode_Params* pParams);
typedef LWPA_Status (*LWPW_Event_QueryEvents_Fn)(LWPW_Event_QueryEvents_Params* pParams);
typedef LWPA_Status (*LWPW_Event_AcknowledgeEvents_Fn)(LWPW_Event_AcknowledgeEvents_Params* pParams);
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
typedef LWPA_Status (*LWPW_LWDA_Profiler_SetFunctionFilter_Fn)(LWPW_LWDA_Profiler_SetFunctionFilter_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_PcSampling_BeginSession_Fn)(LWPW_LWDA_PcSampling_BeginSession_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_PcSampling_EndSession_Fn)(LWPW_LWDA_PcSampling_EndSession_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_PcSampling_GetSampleStats_Fn)(LWPW_LWDA_PcSampling_GetSampleStats_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_PcSampling_GatherData_Fn)(LWPW_LWDA_PcSampling_GatherData_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_PcSampling_StartMeasuring_Fn)(LWPW_LWDA_PcSampling_StartMeasuring_Params* pParams);
typedef LWPA_Status (*LWPW_LWDA_PcSampling_StopMeasuring_Fn)(LWPW_LWDA_PcSampling_StopMeasuring_Params* pParams);
typedef LWPA_Status (*LWPW_DCGM_PeriodicSampler_BeginSession_Fn)(LWPW_DCGM_PeriodicSampler_BeginSession_Params* pParams);
typedef LWPA_Status (*LWPW_DCGM_PeriodicSampler_EndSession_Fn)(LWPW_DCGM_PeriodicSampler_EndSession_Params* pParams);
typedef LWPA_Status (*LWPW_DCGM_PeriodicSampler_SetConfig_Fn)(LWPW_DCGM_PeriodicSampler_SetConfig_Params* pParams);
typedef LWPA_Status (*LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling_Fn)(LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling_Params* pParams);
typedef LWPA_Status (*LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling_Fn)(LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling_Params* pParams);
typedef LWPA_Status (*LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep_Fn)(LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep_Params* pParams);
typedef LWPA_Status (*LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard_Fn)(LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard_Params* pParams);
typedef LWPA_Status (*LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize_Fn)(LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize_Params* pParams);
typedef LWPA_Status (*LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize_Fn)(LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize_Params* pParams);
typedef LWPA_Status (*LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics_Fn)(LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics_Params* pParams);
typedef LWPA_Status (*LWPW_DCGM_PeriodicSampler_DecodeCounters_Fn)(LWPW_DCGM_PeriodicSampler_DecodeCounters_Params* pParams);

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
        (void)pParams;                                                                                                                                                              return g_defaultStatus;
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
static LWPA_Status LWPW_Device_GetSmHierarchy_Default(LWPW_Device_GetSmHierarchy_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_Device_GetVsmMappings_Default(LWPW_Device_GetVsmMappings_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_Event_CreateEventStream_Default(LWPW_Event_CreateEventStream_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_Event_DestroyEventStream_Default(LWPW_Event_DestroyEventStream_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_Event_SetProducerMode_Default(LWPW_Event_SetProducerMode_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_Event_QueryEvents_Default(LWPW_Event_QueryEvents_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_Event_AcknowledgeEvents_Default(LWPW_Event_AcknowledgeEvents_Params* pParams)
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
static LWPA_Status LWPW_LWDA_Profiler_SetFunctionFilter_Default(LWPW_LWDA_Profiler_SetFunctionFilter_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_PcSampling_BeginSession_Default(LWPW_LWDA_PcSampling_BeginSession_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_PcSampling_EndSession_Default(LWPW_LWDA_PcSampling_EndSession_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_PcSampling_GetSampleStats_Default(LWPW_LWDA_PcSampling_GetSampleStats_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_PcSampling_GatherData_Default(LWPW_LWDA_PcSampling_GatherData_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_PcSampling_StartMeasuring_Default(LWPW_LWDA_PcSampling_StartMeasuring_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_LWDA_PcSampling_StopMeasuring_Default(LWPW_LWDA_PcSampling_StopMeasuring_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_DCGM_PeriodicSampler_BeginSession_Default(LWPW_DCGM_PeriodicSampler_BeginSession_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_DCGM_PeriodicSampler_EndSession_Default(LWPW_DCGM_PeriodicSampler_EndSession_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_DCGM_PeriodicSampler_SetConfig_Default(LWPW_DCGM_PeriodicSampler_SetConfig_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling_Default(LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling_Default(LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep_Default(LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard_Default(LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize_Default(LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize_Default(LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics_Default(LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
static LWPA_Status LWPW_DCGM_PeriodicSampler_DecodeCounters_Default(LWPW_DCGM_PeriodicSampler_DecodeCounters_Params* pParams)
{
    (void)pParams;
    return g_defaultStatus;
}
typedef struct PerfworksApi {
    LWPA_GetProcAddress_Fn                                       LWPA_GetProcAddress;
    LWPW_SetLibraryLoadPaths_Fn                                  LWPW_SetLibraryLoadPaths;
    LWPW_SetLibraryLoadPathsW_Fn                                 LWPW_SetLibraryLoadPathsW;
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
    LWPW_Device_GetSmHierarchy_Fn                                LWPW_Device_GetSmHierarchy;
    LWPW_Device_GetVsmMappings_Fn                                LWPW_Device_GetVsmMappings;
    LWPW_Event_CreateEventStream_Fn                              LWPW_Event_CreateEventStream;
    LWPW_Event_DestroyEventStream_Fn                             LWPW_Event_DestroyEventStream;
    LWPW_Event_SetProducerMode_Fn                                LWPW_Event_SetProducerMode;
    LWPW_Event_QueryEvents_Fn                                    LWPW_Event_QueryEvents;
    LWPW_Event_AcknowledgeEvents_Fn                              LWPW_Event_AcknowledgeEvents;
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
    LWPW_LWDA_Profiler_SetFunctionFilter_Fn                      LWPW_LWDA_Profiler_SetFunctionFilter;
    LWPW_LWDA_PcSampling_BeginSession_Fn                         LWPW_LWDA_PcSampling_BeginSession;
    LWPW_LWDA_PcSampling_EndSession_Fn                           LWPW_LWDA_PcSampling_EndSession;
    LWPW_LWDA_PcSampling_GetSampleStats_Fn                       LWPW_LWDA_PcSampling_GetSampleStats;
    LWPW_LWDA_PcSampling_GatherData_Fn                           LWPW_LWDA_PcSampling_GatherData;
    LWPW_LWDA_PcSampling_StartMeasuring_Fn                       LWPW_LWDA_PcSampling_StartMeasuring;
    LWPW_LWDA_PcSampling_StopMeasuring_Fn                        LWPW_LWDA_PcSampling_StopMeasuring;
    LWPW_DCGM_PeriodicSampler_BeginSession_Fn                    LWPW_DCGM_PeriodicSampler_BeginSession;
    LWPW_DCGM_PeriodicSampler_EndSession_Fn                      LWPW_DCGM_PeriodicSampler_EndSession;
    LWPW_DCGM_PeriodicSampler_SetConfig_Fn                       LWPW_DCGM_PeriodicSampler_SetConfig;
    LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling_Fn        LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling;
    LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling_Fn         LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling;
    LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep_Fn          LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep;
    LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard_Fn       LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard;
    LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize_Fn  LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize;
    LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize_Fn     LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize;
    LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics_Fn LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics;
    LWPW_DCGM_PeriodicSampler_DecodeCounters_Fn                  LWPW_DCGM_PeriodicSampler_DecodeCounters;
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
        , &LWPW_Device_GetSmHierarchy_Default
        , &LWPW_Device_GetVsmMappings_Default
        , &LWPW_Event_CreateEventStream_Default
        , &LWPW_Event_DestroyEventStream_Default
        , &LWPW_Event_SetProducerMode_Default
        , &LWPW_Event_QueryEvents_Default
        , &LWPW_Event_AcknowledgeEvents_Default
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
        , &LWPW_LWDA_Profiler_SetFunctionFilter_Default
        , &LWPW_LWDA_PcSampling_BeginSession_Default
        , &LWPW_LWDA_PcSampling_EndSession_Default
        , &LWPW_LWDA_PcSampling_GetSampleStats_Default
        , &LWPW_LWDA_PcSampling_GatherData_Default
        , &LWPW_LWDA_PcSampling_StartMeasuring_Default
        , &LWPW_LWDA_PcSampling_StopMeasuring_Default
        , &LWPW_DCGM_PeriodicSampler_BeginSession_Default
        , &LWPW_DCGM_PeriodicSampler_EndSession_Default
        , &LWPW_DCGM_PeriodicSampler_SetConfig_Default
        , &LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling_Default
        , &LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling_Default
        , &LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep_Default
        , &LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard_Default
        , &LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize_Default
        , &LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize_Default
        , &LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics_Default
        , &LWPW_DCGM_PeriodicSampler_DecodeCounters_Default
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
    g_api.fn.LWPW_Device_GetSmHierarchy = (LWPW_Device_GetSmHierarchy_Fn)GetPerfworksProc("LWPW_Device_GetSmHierarchy", (LWPA_GenericFn)g_api.fn.LWPW_Device_GetSmHierarchy);
    g_api.fn.LWPW_Device_GetVsmMappings = (LWPW_Device_GetVsmMappings_Fn)GetPerfworksProc("LWPW_Device_GetVsmMappings", (LWPA_GenericFn)g_api.fn.LWPW_Device_GetVsmMappings);
    g_api.fn.LWPW_Event_CreateEventStream = (LWPW_Event_CreateEventStream_Fn)GetPerfworksProc("LWPW_Event_CreateEventStream", (LWPA_GenericFn)g_api.fn.LWPW_Event_CreateEventStream);
    g_api.fn.LWPW_Event_DestroyEventStream = (LWPW_Event_DestroyEventStream_Fn)GetPerfworksProc("LWPW_Event_DestroyEventStream", (LWPA_GenericFn)g_api.fn.LWPW_Event_DestroyEventStream);
    g_api.fn.LWPW_Event_SetProducerMode = (LWPW_Event_SetProducerMode_Fn)GetPerfworksProc("LWPW_Event_SetProducerMode", (LWPA_GenericFn)g_api.fn.LWPW_Event_SetProducerMode);
    g_api.fn.LWPW_Event_QueryEvents = (LWPW_Event_QueryEvents_Fn)GetPerfworksProc("LWPW_Event_QueryEvents", (LWPA_GenericFn)g_api.fn.LWPW_Event_QueryEvents);
    g_api.fn.LWPW_Event_AcknowledgeEvents = (LWPW_Event_AcknowledgeEvents_Fn)GetPerfworksProc("LWPW_Event_AcknowledgeEvents", (LWPA_GenericFn)g_api.fn.LWPW_Event_AcknowledgeEvents);
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
    g_api.fn.LWPW_LWDA_Profiler_SetFunctionFilter = (LWPW_LWDA_Profiler_SetFunctionFilter_Fn)GetPerfworksProc("LWPW_LWDA_Profiler_SetFunctionFilter", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_Profiler_SetFunctionFilter);
    g_api.fn.LWPW_LWDA_PcSampling_BeginSession = (LWPW_LWDA_PcSampling_BeginSession_Fn)GetPerfworksProc("LWPW_LWDA_PcSampling_BeginSession", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_PcSampling_BeginSession);
    g_api.fn.LWPW_LWDA_PcSampling_EndSession = (LWPW_LWDA_PcSampling_EndSession_Fn)GetPerfworksProc("LWPW_LWDA_PcSampling_EndSession", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_PcSampling_EndSession);
    g_api.fn.LWPW_LWDA_PcSampling_GetSampleStats = (LWPW_LWDA_PcSampling_GetSampleStats_Fn)GetPerfworksProc("LWPW_LWDA_PcSampling_GetSampleStats", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_PcSampling_GetSampleStats);
    g_api.fn.LWPW_LWDA_PcSampling_GatherData = (LWPW_LWDA_PcSampling_GatherData_Fn)GetPerfworksProc("LWPW_LWDA_PcSampling_GatherData", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_PcSampling_GatherData);
    g_api.fn.LWPW_LWDA_PcSampling_StartMeasuring = (LWPW_LWDA_PcSampling_StartMeasuring_Fn)GetPerfworksProc("LWPW_LWDA_PcSampling_StartMeasuring", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_PcSampling_StartMeasuring);
    g_api.fn.LWPW_LWDA_PcSampling_StopMeasuring = (LWPW_LWDA_PcSampling_StopMeasuring_Fn)GetPerfworksProc("LWPW_LWDA_PcSampling_StopMeasuring", (LWPA_GenericFn)g_api.fn.LWPW_LWDA_PcSampling_StopMeasuring);
    g_api.fn.LWPW_DCGM_PeriodicSampler_BeginSession = (LWPW_DCGM_PeriodicSampler_BeginSession_Fn)GetPerfworksProc("LWPW_DCGM_PeriodicSampler_BeginSession", (LWPA_GenericFn)g_api.fn.LWPW_DCGM_PeriodicSampler_BeginSession);
    g_api.fn.LWPW_DCGM_PeriodicSampler_EndSession = (LWPW_DCGM_PeriodicSampler_EndSession_Fn)GetPerfworksProc("LWPW_DCGM_PeriodicSampler_EndSession", (LWPA_GenericFn)g_api.fn.LWPW_DCGM_PeriodicSampler_EndSession);
    g_api.fn.LWPW_DCGM_PeriodicSampler_SetConfig = (LWPW_DCGM_PeriodicSampler_SetConfig_Fn)GetPerfworksProc("LWPW_DCGM_PeriodicSampler_SetConfig", (LWPA_GenericFn)g_api.fn.LWPW_DCGM_PeriodicSampler_SetConfig);
    g_api.fn.LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling = (LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling_Fn)GetPerfworksProc("LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling", (LWPA_GenericFn)g_api.fn.LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling);
    g_api.fn.LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling = (LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling_Fn)GetPerfworksProc("LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling", (LWPA_GenericFn)g_api.fn.LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling);
    g_api.fn.LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep = (LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep_Fn)GetPerfworksProc("LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep", (LWPA_GenericFn)g_api.fn.LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep);
    g_api.fn.LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard = (LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard_Fn)GetPerfworksProc("LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard", (LWPA_GenericFn)g_api.fn.LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard);
    g_api.fn.LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize = (LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize_Fn)GetPerfworksProc("LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize", (LWPA_GenericFn)g_api.fn.LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize);
    g_api.fn.LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize = (LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize_Fn)GetPerfworksProc("LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize", (LWPA_GenericFn)g_api.fn.LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize);
    g_api.fn.LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics = (LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics_Fn)GetPerfworksProc("LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics", (LWPA_GenericFn)g_api.fn.LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics);
    g_api.fn.LWPW_DCGM_PeriodicSampler_DecodeCounters = (LWPW_DCGM_PeriodicSampler_DecodeCounters_Fn)GetPerfworksProc("LWPW_DCGM_PeriodicSampler_DecodeCounters", (LWPA_GenericFn)g_api.fn.LWPW_DCGM_PeriodicSampler_DecodeCounters);
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
    return g_api.fn.LWPW_Device_GetPciBusIds(pParams);                                                                                                                      }
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
LWPA_Status LWPW_Device_GetSmHierarchy(LWPW_Device_GetSmHierarchy_Params* pParams)
{
    return g_api.fn.LWPW_Device_GetSmHierarchy(pParams);
}
LWPA_Status LWPW_Device_GetVsmMappings(LWPW_Device_GetVsmMappings_Params* pParams)
{
    return g_api.fn.LWPW_Device_GetVsmMappings(pParams);
}
LWPA_Status LWPW_Event_CreateEventStream(LWPW_Event_CreateEventStream_Params* pParams)
{
    return g_api.fn.LWPW_Event_CreateEventStream(pParams);
}
LWPA_Status LWPW_Event_DestroyEventStream(LWPW_Event_DestroyEventStream_Params* pParams)
{
    return g_api.fn.LWPW_Event_DestroyEventStream(pParams);
}
LWPA_Status LWPW_Event_SetProducerMode(LWPW_Event_SetProducerMode_Params* pParams)
{
    return g_api.fn.LWPW_Event_SetProducerMode(pParams);
}
LWPA_Status LWPW_Event_QueryEvents(LWPW_Event_QueryEvents_Params* pParams)
{
    return g_api.fn.LWPW_Event_QueryEvents(pParams);
}
LWPA_Status LWPW_Event_AcknowledgeEvents(LWPW_Event_AcknowledgeEvents_Params* pParams)
{
    return g_api.fn.LWPW_Event_AcknowledgeEvents(pParams);
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
LWPA_Status LWPW_LWDA_Profiler_SetFunctionFilter(LWPW_LWDA_Profiler_SetFunctionFilter_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_Profiler_SetFunctionFilter(pParams);
}
LWPA_Status LWPW_LWDA_PcSampling_BeginSession(LWPW_LWDA_PcSampling_BeginSession_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_PcSampling_BeginSession(pParams);
}
LWPA_Status LWPW_LWDA_PcSampling_EndSession(LWPW_LWDA_PcSampling_EndSession_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_PcSampling_EndSession(pParams);
}
LWPA_Status LWPW_LWDA_PcSampling_GetSampleStats(LWPW_LWDA_PcSampling_GetSampleStats_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_PcSampling_GetSampleStats(pParams);
}
LWPA_Status LWPW_LWDA_PcSampling_GatherData(LWPW_LWDA_PcSampling_GatherData_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_PcSampling_GatherData(pParams);
}
LWPA_Status LWPW_LWDA_PcSampling_StartMeasuring(LWPW_LWDA_PcSampling_StartMeasuring_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_PcSampling_StartMeasuring(pParams);
}
LWPA_Status LWPW_LWDA_PcSampling_StopMeasuring(LWPW_LWDA_PcSampling_StopMeasuring_Params* pParams)
{
    return g_api.fn.LWPW_LWDA_PcSampling_StopMeasuring(pParams);
}
LWPA_Status LWPW_DCGM_PeriodicSampler_BeginSession(LWPW_DCGM_PeriodicSampler_BeginSession_Params* pParams)
{
    return g_api.fn.LWPW_DCGM_PeriodicSampler_BeginSession(pParams);
}
LWPA_Status LWPW_DCGM_PeriodicSampler_EndSession(LWPW_DCGM_PeriodicSampler_EndSession_Params* pParams)
{
    return g_api.fn.LWPW_DCGM_PeriodicSampler_EndSession(pParams);
}
LWPA_Status LWPW_DCGM_PeriodicSampler_SetConfig(LWPW_DCGM_PeriodicSampler_SetConfig_Params* pParams)
{
    return g_api.fn.LWPW_DCGM_PeriodicSampler_SetConfig(pParams);
}
LWPA_Status LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling(LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling_Params* pParams)
{
    return g_api.fn.LWPW_DCGM_PeriodicSampler_CPUTrigger_StartSampling(pParams);
}
LWPA_Status LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling(LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling_Params* pParams)
{
    return g_api.fn.LWPW_DCGM_PeriodicSampler_CPUTrigger_StopSampling(pParams);
}
LWPA_Status LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep(LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep_Params* pParams)
{
    return g_api.fn.LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerKeep(pParams);
}
LWPA_Status LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard(LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard_Params* pParams)
{
    return g_api.fn.LWPW_DCGM_PeriodicSampler_CPUTrigger_TriggerDiscard(pParams);
}
LWPA_Status LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize(LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize_Params* pParams)
{
    return g_api.fn.LWPW_DCGM_PeriodicSampler_CounterDataImage_CallwlateSize(pParams);
}
LWPA_Status LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize(LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize_Params* pParams)
{
    return g_api.fn.LWPW_DCGM_PeriodicSampler_CounterDataImage_Initialize(pParams);
}
LWPA_Status LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics(LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics_Params* pParams)
{
    return g_api.fn.LWPW_DCGM_PeriodicSampler_CounterDataImage_UnpackRawMetrics(pParams);
}
LWPA_Status LWPW_DCGM_PeriodicSampler_DecodeCounters(LWPW_DCGM_PeriodicSampler_DecodeCounters_Params* pParams)
{
    return g_api.fn.LWPW_DCGM_PeriodicSampler_DecodeCounters(pParams);
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
    char const* const pLibName = "liblwperf_target.so";

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
