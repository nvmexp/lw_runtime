/*
 * Copyright 2011-2020   LWPU Corporation.  All rights reserved.
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

#if !defined(_LWPTI_PROFILER_TARGET_H_)
#define _LWPTI_PROFILER_TARGET_H_

#include <lwca.h>
#include <lwpti_result.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) && defined(LWPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \defgroup LWPTI_PROFILER_API LWPTI Profiler API
 * Functions, types, and enums that implement the LWPTI Profiler API.
 * @{
 */
#ifndef LWPTI_PROFILER_STRUCT_SIZE
#define LWPTI_PROFILER_STRUCT_SIZE(type_, lastfield_)                     (offsetof(type_, lastfield_) + sizeof(((type_*)0)->lastfield_))
#endif

/**
 * \brief Profiler range attribute
 *
 * A metric enabled in the session's configuration is collected separately per unique range-stack in the pass.
 * This is an attribute to collect metrics around each kernel in a profiling session or in an user defined range.
 */
typedef enum
{
    /**
     * Invalid value
     */
    LWPTI_Range_ILWALID,
    /**
     * Ranges are auto defined around each kernel in a profiling session
     */
    LWPTI_AutoRange,
    /**
     * A range in which metric data to be collected is defined by the user
     */
    LWPTI_UserRange,
    /**
     * Range count
     */
    LWPTI_Range_COUNT,
} LWpti_ProfilerRange;

/**
 * \brief Profiler replay attribute
 *
 * For metrics which require multipass collection, a replay of the GPU kernel(s) is required.
 * This is an attribute which specify how the replay of the kernel(s) to be measured is done.
 */
typedef enum
{
    /**
     * Invalid Value
     */
    LWPTI_Replay_ILWALID,
    /**
     * Replay is done by LWPTI user around the process
     */
    LWPTI_ApplicationReplay,
    /**
     * Replay is done around kernel implicitly by LWPTI
     */
    LWPTI_KernelReplay,
    /**
     * Replay is done by LWPTI user within a process
     */
    LWPTI_UserReplay,
    /**
     * Replay count
     */
    LWPTI_Replay_COUNT,
} LWpti_ProfilerReplayMode;

/**
 * \brief Default parameter for lwptiProfilerInitialize
 */
typedef struct LWpti_Profiler_Initialize_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_Initialize_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

} LWpti_Profiler_Initialize_Params;
#define LWpti_Profiler_Initialize_Params_STRUCT_SIZE                  LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_Initialize_Params, pPriv)

/**
 * \brief Default parameter for lwptiProfilerDeInitialize
 */
typedef struct LWpti_Profiler_DeInitialize_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_DeInitialize_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

} LWpti_Profiler_DeInitialize_Params;
#define LWpti_Profiler_DeInitialize_Params_STRUCT_SIZE                  LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_DeInitialize_Params, pPriv)

/**
 * \brief Initializes the profiler interface
 *
 * Loads the required libraries in the process address space.
 * Sets up the hooks with the LWCA driver.
 */
LWptiResult LWPTIAPI lwptiProfilerInitialize(LWpti_Profiler_Initialize_Params *pParams);

/**
 * \brief DeInitializes the profiler interface
 */
LWptiResult LWPTIAPI lwptiProfilerDeInitialize(LWpti_Profiler_DeInitialize_Params *pParams);

/**
 * \brief Input parameter to define the counterDataImage
 */
typedef struct LWpti_Profiler_CounterDataImageOptions
{
    size_t structSize;                                          //!< [in] LWpti_Profiler_CounterDataImageOptions_Params_STRUCT_SIZE
    void* pPriv;                                                //!< [in] assign to NULL

    const uint8_t* pCounterDataPrefix;                          /**< [in] Address of CounterDataPrefix generated from LWPW_CounterDataBuilder_GetCounterDataPrefix().
                                                                    Must be align(8).*/
    size_t counterDataPrefixSize;                               //!< [in] Size of CounterDataPrefix generated from LWPW_CounterDataBuilder_GetCounterDataPrefix().
    uint32_t maxNumRanges;                                      //!< [in] Maximum number of ranges that can be profiled
    uint32_t maxNumRangeTreeNodes;                              //!< [in] Maximum number of RangeTree nodes; must be >= maxNumRanges
    uint32_t maxRangeNameLength;                                //!< [in] Maximum string length of each RangeName, including the trailing NULL character
} LWpti_Profiler_CounterDataImageOptions;
#define LWpti_Profiler_CounterDataImageOptions_STRUCT_SIZE                       LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_CounterDataImageOptions, maxRangeNameLength)

/**
 * \brief Params for lwptiProfilerCounterDataImageCallwlateSize
 */
typedef struct LWpti_Profiler_CounterDataImage_CallwlateSize_Params
{
    size_t structSize;                                          //!< [in] LWpti_Profiler_CounterDataImage_CallwlateSize_Params_STRUCT_SIZE
    void* pPriv;                                                //!< [in] assign to NULL

    size_t sizeofCounterDataImageOptions;                       //!< [in] LWpti_Profiler_CounterDataImageOptions_STRUCT_SIZE
    const LWpti_Profiler_CounterDataImageOptions* pOptions;     //!< [in] Pointer to LWpti_Profiler_CounterDataImageOptions
    size_t counterDataImageSize;                                //!< [out]
} LWpti_Profiler_CounterDataImage_CallwlateSize_Params;
#define LWpti_Profiler_CounterDataImage_CallwlateSize_Params_STRUCT_SIZE         LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_CounterDataImage_CallwlateSize_Params, counterDataImageSize)

/**
 * \brief Params for lwptiProfilerCounterDataImageInitialize
 */
typedef struct LWpti_Profiler_CounterDataImage_Initialize_Params
{
    size_t structSize;                                          //!< [in] LWpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE
    void* pPriv;                                                //!< [in] assign to NULL

    size_t sizeofCounterDataImageOptions;                       //!< [in] LWpti_Profiler_CounterDataImageOptions_STRUCT_SIZE
    const LWpti_Profiler_CounterDataImageOptions* pOptions;     //!< [in] Pointer to LWpti_Profiler_CounterDataImageOptions
    size_t counterDataImageSize;                                //!< [in] Size callwlated from lwptiProfilerCounterDataImageCallwlateSize
    uint8_t* pCounterDataImage;                                 //!< [in] The buffer to be initialized.
} LWpti_Profiler_CounterDataImage_Initialize_Params;
#define LWpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE            LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_CounterDataImage_Initialize_Params, pCounterDataImage)

/**
 * \brief A CounterData image allocates space for values for each counter for each range.
 *
 * User borne the resposibility of managing the counterDataImage allocations.
 * CounterDataPrefix contains meta data about the metrics that will be stored in counterDataImage.
 * Use these APIs to callwlate the allocation size and initialize counterData image.
 */
LWptiResult lwptiProfilerCounterDataImageCallwlateSize(LWpti_Profiler_CounterDataImage_CallwlateSize_Params* pParams);
LWptiResult lwptiProfilerCounterDataImageInitialize(LWpti_Profiler_CounterDataImage_Initialize_Params* pParams);

/**
 * \brief Params for lwptiProfilerCounterDataImageCallwlateScratchBufferSize
 */
typedef struct LWpti_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    size_t counterDataImageSize;                            //!< [in] size callwlated from lwptiProfilerCounterDataImageCallwlateSize
    uint8_t* pCounterDataImage;                             //!< [in]
    size_t counterDataScratchBufferSize;                    //!< [out]
} LWpti_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params;
#define LWpti_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params_STRUCT_SIZE    LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params, counterDataScratchBufferSize)

/**
 * \brief Params for lwptiProfilerCounterDataImageInitializeScratchBuffer
 */
typedef struct LWpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    size_t counterDataImageSize;                            //!< [in] size callwlated from lwptiProfilerCounterDataImageCallwlateSize
    uint8_t* pCounterDataImage;                             //!< [in]
    size_t counterDataScratchBufferSize;                    //!< [in] size callwlated using lwptiProfilerCounterDataImageCallwlateScratchBufferSize
    uint8_t* pCounterDataScratchBuffer;                     //!< [in] the scratch buffer to be initialized.
} LWpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params;
#define LWpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE       LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params, pCounterDataScratchBuffer)

/**
 * \brief A temporary storage for CounterData image needed for internal operations
 *
 * Use these APIs to callwlate the allocation size and initialize counterData image scratch buffer.
 */
LWptiResult lwptiProfilerCounterDataImageCallwlateScratchBufferSize(LWpti_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params* pParams);
LWptiResult lwptiProfilerCounterDataImageInitializeScratchBuffer(LWpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams);

/**
 * \brief Params for lwptiProfilerBeginSession
 */
typedef struct LWpti_Profiler_BeginSession_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_BeginSession_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    LWcontext ctx;                                          //!< [in] if NULL, the current LWcontext is used
    size_t counterDataImageSize;                            //!< [in] size callwlated from lwptiProfilerCounterDataImageCallwlateSize
    uint8_t* pCounterDataImage;                             //!< [in] address of CounterDataImage
    size_t counterDataScratchBufferSize;                    //!< [in] size callwlated from lwptiProfilerCounterDataImageInitializeScratchBuffer
    uint8_t* pCounterDataScratchBuffer;                     //!< [in] address of CounterDataImage scratch buffer
    uint8_t bDumpCounterDataInFile;                          //!< [in] [optional]
    const char* pCounterDataFilePath;                        //!< [in] [optional]
    LWpti_ProfilerRange range;                               //!< [in] LWpti_ProfilerRange
    LWpti_ProfilerReplayMode replayMode;                     //!< [in] LWpti_ProfilerReplayMode
    /* Replay options, required when replay is done by lwpti user */
    size_t maxRangesPerPass;                                //!< [in] Maximum number of ranges that can be recorded in a single pass.
    size_t maxLaunchesPerPass;                              //!< [in] Maximum number of kernel launches that can be recorded in a single pass; must be >= maxRangesPerPass.

} LWpti_Profiler_BeginSession_Params;
#define LWpti_Profiler_BeginSession_Params_STRUCT_SIZE                  LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_BeginSession_Params, maxLaunchesPerPass)
/**
 * \brief Params for lwptiProfilerEndSession
 */
typedef struct LWpti_Profiler_EndSession_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_EndSession_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    LWcontext ctx;                                          //!< [in] if NULL, the current LWcontext is used
} LWpti_Profiler_EndSession_Params;
#define LWpti_Profiler_EndSession_Params_STRUCT_SIZE                  LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_EndSession_Params, ctx)

/**
 * \brief Begin profiling session sets up the profiling on the device
 *
 * Although, it doesn't start the profiling but GPU resources needed for profiling are allocated.
 * Outside of a session, the GPU will return to its normal operating state.
 */
LWptiResult LWPTIAPI lwptiProfilerBeginSession(LWpti_Profiler_BeginSession_Params* pParams);
/**
 * \brief Ends profiling session
 *
 * Frees up the GPU resources acquired for profiling.
 * Outside of a session, the GPU will return to it's normal operating state.
 */
LWptiResult LWPTIAPI lwptiProfilerEndSession(LWpti_Profiler_EndSession_Params* pParams);

/**
 * \brief Params for lwptiProfilerSetConfig
 */
typedef struct LWpti_Profiler_SetConfig_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_SetConfig_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    LWcontext ctx;                                          //!< [in] if NULL, the current LWcontext is used
    const uint8_t* pConfig;                                 //!< [in] Config created by LWPW_RawMetricsConfig_GetConfigImage(). Must be align(8).
    size_t configSize;                                      //!< [in] size of config
    uint16_t minNestingLevel;                               //!< [in] the lowest nesting level to be profiled; must be >= 1
    uint16_t numNestingLevels;                              //!< [in] the number of nesting levels to profile; must be >= 1
    size_t passIndex;                                       //!< [in] Set this to zero for in-app replay; set this to the output of EndPass() for application replay
    uint16_t targetNestingLevel;                            //!< [in] Set this to minNestingLevel for in-app replay; set this to the output of EndPass() for application
} LWpti_Profiler_SetConfig_Params;

#define LWpti_Profiler_SetConfig_Params_STRUCT_SIZE                    LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_SetConfig_Params, targetNestingLevel)

/**
 * \brief Params for lwptiProfilerUnsetConfig
 */
typedef struct LWpti_Profiler_UnsetConfig_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_UnsetConfig_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    LWcontext ctx;                                          //!< [in] if NULL, the current LWcontext is used
} LWpti_Profiler_UnsetConfig_Params;
#define LWpti_Profiler_UnsetConfig_Params_STRUCT_SIZE                  LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_UnsetConfig_Params, ctx)

/**
 * \brief Set metrics configuration to be profiled
 *
 * Use these APIs to set the config to profile in a session. It can be used for advanced cases such as where multiple
 * configurations are collected into a single CounterData Image on the need basis, without restarting the session.
 */
LWptiResult LWPTIAPI lwptiProfilerSetConfig(LWpti_Profiler_SetConfig_Params* pParams);
/**
 * \brief Unset metrics configuration profiled
 *
 */
LWptiResult LWPTIAPI lwptiProfilerUnsetConfig(LWpti_Profiler_UnsetConfig_Params* pParams);

/**
 * \brief Params for lwptiProfilerBeginPass
 */
typedef struct LWpti_Profiler_BeginPass_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_BeginPass_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    LWcontext ctx;                                          //!< [in] if NULL, the current LWcontext is used
} LWpti_Profiler_BeginPass_Params;
#define LWpti_Profiler_BeginPass_Params_STRUCT_SIZE                  LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_BeginPass_Params, ctx)

/**
 * \brief Params for lwptiProfilerEndPass
 */
typedef struct LWpti_Profiler_EndPass_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_EndPass_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    LWcontext ctx;                                          //!< [in] if NULL, the current LWcontext is used
    uint16_t targetNestingLevel;                            //!  [out] The targetNestingLevel that will be collected by the *next* BeginPass.
    size_t passIndex;                                       //!< [out] The passIndex that will be collected by the *next* BeginPass
    uint8_t allPassesSubmitted;                             //!< [out] becomes true when the last pass has been queued to the GPU
} LWpti_Profiler_EndPass_Params;
#define LWpti_Profiler_EndPass_Params_STRUCT_SIZE                    LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_EndPass_Params, allPassesSubmitted)

/**
 * \brief Replay API: used for multipass collection.

 * These APIs are used if user chooses to replay by itself /ref LWPTI_UserReplay or /ref LWPTI_ApplicationReplay
 * for multipass collection of the metrics configurations.
 * It's a no-op in case of /ref LWPTI_KernelReplay.
 */
LWptiResult lwptiProfilerBeginPass(LWpti_Profiler_BeginPass_Params* pParams);

/**
 * \brief Replay API: used for multipass collection.

 * These APIs are used if user chooses to replay by itself /ref LWPTI_UserReplay or /ref LWPTI_ApplicationReplay
 * for multipass collection of the metrics configurations.
 * Its a no-op in case of /ref LWPTI_KernelReplay.
 * Returns information for next pass.
 */
LWptiResult lwptiProfilerEndPass(LWpti_Profiler_EndPass_Params* pParams);

/**
 * \brief Params for lwptiProfilerEnableProfiling
 */
typedef struct LWpti_Profiler_EnableProfiling_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_EnableProfiling_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    LWcontext ctx;                                          //!< [in] if NULL, the current LWcontext is used
} LWpti_Profiler_EnableProfiling_Params;
#define LWpti_Profiler_EnableProfiling_Params_STRUCT_SIZE                  LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_EnableProfiling_Params, ctx)

/**
 * \brief Params for lwptiProfilerDisableProfiling
 */
typedef struct LWpti_Profiler_DisableProfiling_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_DisableProfiling_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    LWcontext ctx;                                          //!< [in] if NULL, the current LWcontext is used
} LWpti_Profiler_DisableProfiling_Params;
#define LWpti_Profiler_DisableProfiling_Params_STRUCT_SIZE                  LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_DisableProfiling_Params, ctx)

/**
 * \brief Enables Profiling
 *
 * In /ref LWPTI_AutoRange, these APIs are used to enable/disable profiling for the kernels to be exelwted in
 * a profiling session.
 */
LWptiResult LWPTIAPI lwptiProfilerEnableProfiling(LWpti_Profiler_EnableProfiling_Params* pParams);

/**
 * \brief Disable Profiling
 *
 * In /ref LWPTI_AutoRange, these APIs are used to enable/disable profiling for the kernels to be exelwted in
 * a profiling session.
 */
LWptiResult LWPTIAPI lwptiProfilerDisableProfiling(LWpti_Profiler_DisableProfiling_Params* pParams);

/**
 * \brief Params for lwptiProfilerIsPassCollected
 */
typedef struct LWpti_Profiler_IsPassCollected_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_IsPassCollected_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    LWcontext ctx;                                          //!< [in] if NULL, the current LWcontext is used
    size_t numRangesDropped;                                //!< [out] number of ranges whose data was dropped in the processed pass
    size_t numTraceBytesDropped;                            //!< [out] number of bytes not written to TraceBuffer due to buffer full
    uint8_t onePassCollected;                               //!< [out] true if a pass was successfully decoded
    uint8_t allPassesCollected;                             //!< [out] becomes true when the last pass has been decoded
} LWpti_Profiler_IsPassCollected_Params;
#define LWpti_Profiler_IsPassCollected_Params_STRUCT_SIZE            LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_IsPassCollected_Params, allPassesCollected)

/**
 * \brief Asynchronous call to query if the submitted pass to GPU is collected
 *
 */
LWptiResult LWPTIAPI lwptiProfilerIsPassCollected(LWpti_Profiler_IsPassCollected_Params* pParams);

/**
 * \brief Params for lwptiProfilerFlushCounterData
 */
typedef struct LWpti_Profiler_FlushCounterData_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_FlushCounterData_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    LWcontext ctx;                                          //!< [in] if NULL, the current LWcontext is used
    size_t numRangesDropped;                                //!< [out] number of ranges whose data was dropped in the processed passes
    size_t numTraceBytesDropped;                            //!< [out] number of bytes not written to TraceBuffer due to buffer full
} LWpti_Profiler_FlushCounterData_Params;
#define LWpti_Profiler_FlushCounterData_Params_STRUCT_SIZE           LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_FlushCounterData_Params, numTraceBytesDropped)

/**
 * \brief Decode all the submitted passes
 *
 * Flush Counter data API to ensure every pass is decoded into the counterDataImage passed at beginSession.
 * This will cause the CPU/GPU sync to collect all the undecoded pass.
 */
LWptiResult LWPTIAPI lwptiProfilerFlushCounterData(LWpti_Profiler_FlushCounterData_Params* pParams);

typedef struct LWpti_Profiler_PushRange_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_PushRange_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    LWcontext ctx;                                          //!< [in] if NULL, the current LWcontext is used
    const char* pRangeName;                                 //!< [in] specifies the range for subsequent launches; must not be NULL
    size_t rangeNameLength;                                 //!< [in] assign to strlen(pRangeName) if known; if set to zero, the library will call strlen()
} LWpti_Profiler_PushRange_Params;
#define LWpti_Profiler_PushRange_Params_STRUCT_SIZE                  LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_PushRange_Params, rangeNameLength)

typedef struct LWpti_Profiler_PopRange_Params
{
    size_t structSize;                                      //!< [in] LWpti_Profiler_PopRange_Params_STRUCT_SIZE
    void* pPriv;                                            //!< [in] assign to NULL

    LWcontext ctx;                                          //!< [in] if NULL, the current LWcontext is used
} LWpti_Profiler_PopRange_Params;
#define LWpti_Profiler_PopRange_Params_STRUCT_SIZE                  LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_PopRange_Params, ctx)


/**
 * \brief Range API's : Push user range
 *
 * Counter data is collected per unique range-stack. Identified by a string label passsed by the user.
 * It's an invalid operation in case of /ref LWPTI_AutoRange.
 */
LWptiResult LWPTIAPI lwptiProfilerPushRange(LWpti_Profiler_PushRange_Params *pParams);

/**
 * \brief Range API's : Pop user range
 *
 * Counter data is collected per unique range-stack. Identified by a string label passsed by the user.
 * It's an invalid operation in case of /ref LWPTI_AutoRange.
 */
LWptiResult LWPTIAPI lwptiProfilerPopRange(LWpti_Profiler_PopRange_Params *pParams);

/**
 * \brief Params for lwptiProfilerGetCounterAvailability
 */
typedef struct LWpti_Profiler_GetCounterAvailability_Params
{
    
    size_t structSize;                                  //!< [in] LWpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE
    void* pPriv;                                        //!< [in] assign to NULL    
    LWcontext ctx;                                      //!< [in] if NULL, the current LWcontext is used
    size_t counterAvailabilityImageSize;                //!< [in/out] If `pCounterAvailabilityImage` is NULL, then the required size is returned in
                                                        //!< `counterAvailabilityImageSize`, otherwise `counterAvailabilityImageSize` should be set to the size of
                                                        //!< `pCounterAvailabilityImage`, and on return it would be overwritten with number of actual bytes copied
    uint8_t* pCounterAvailabilityImage;                 //!< [in] buffer receiving counter availability image, may be NULL
} LWpti_Profiler_GetCounterAvailability_Params;
#define LWpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE                  LWPTI_PROFILER_STRUCT_SIZE(LWpti_Profiler_GetCounterAvailability_Params, pCounterAvailabilityImage)

/**
 * \brief Query counter availibility 
 * 
 * Use this API to query counter availability information in a buffer which can be used to filter unavailable raw metrics on host.
 * Note: This API may fail, if any profiling or sampling session is active on the specified context or its device.
 */
LWptiResult LWPTIAPI lwptiProfilerGetCounterAvailability(LWpti_Profiler_GetCounterAvailability_Params *pParams);

/** @} */ /* END LWPTI_METRIC_API */
#if defined(__GNUC__) && defined(LWPTI_LIB)
    #pragma GCC visibility pop
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /*_LWPTI_PROFILER_TARGET_H_*/
