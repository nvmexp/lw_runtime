/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */
#pragma once

#include <exp/context/ErrorHandling.h>

#include <corelib/misc/String.h>
#include <corelib/system/ExelwtableModule.h>
#include <corelib/system/SystemError.h>
#include <prodlib/system/Logger.h>

#include <lwpti_profiler_target.h>
#include <lwpti_target.h>

#include <memory>

namespace prodlib {

class Lwpti
{
  public:
    OptixResult initialize( optix_exp::ErrorDetails& errDetails );
    bool available() const;

    LWptiResult profilerInitialize( LWpti_Profiler_Initialize_Params* pParams ) const;
    LWptiResult profilerDeInitialize( LWpti_Profiler_DeInitialize_Params* pParams ) const;
    LWptiResult deviceGetChipName( LWpti_Device_GetChipName_Params* pParams ) const;
    LWptiResult profilerGetCounterAvailability( LWpti_Profiler_GetCounterAvailability_Params* pParams ) const;
    LWptiResult profilerCounterDataImageCallwlateSize( LWpti_Profiler_CounterDataImage_CallwlateSize_Params* pParams ) const;
    LWptiResult profilerCounterDataImageInitialize( LWpti_Profiler_CounterDataImage_Initialize_Params* pParams ) const;
    LWptiResult profilerCounterDataImageCallwlateScratchBufferSize( LWpti_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params* pParams ) const;
    LWptiResult profilerCounterDataImageInitializeScratchBuffer( LWpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams ) const;
    LWptiResult profilerSetConfig( LWpti_Profiler_SetConfig_Params* pParams ) const;
    LWptiResult profilerUnsetConfig( LWpti_Profiler_UnsetConfig_Params* pParams ) const;
    LWptiResult profilerBeginSession( LWpti_Profiler_BeginSession_Params* pParams ) const;
    LWptiResult profilerEndSession( LWpti_Profiler_EndSession_Params* pParams ) const;
    LWptiResult profilerPushRange( LWpti_Profiler_PushRange_Params* pParams ) const;
    LWptiResult profilerPopRange( LWpti_Profiler_PopRange_Params* pParams ) const;
    LWptiResult profilerBeginPass( LWpti_Profiler_BeginPass_Params* pParams ) const;
    LWptiResult profilerEndPass( LWpti_Profiler_EndPass_Params* pParams ) const;
    LWptiResult profilerEnableProfiling( LWpti_Profiler_EnableProfiling_Params* pParams ) const;
    LWptiResult profilerDisableProfiling( LWpti_Profiler_DisableProfiling_Params* pParams ) const;
    LWptiResult profilerFlushCounterData( LWpti_Profiler_FlushCounterData_Params* pParams ) const;

  private:
    std::unique_ptr<corelib::ExelwtableModule> m_lib;

    bool m_available = false;

    using ProfilerInitialize_t             = LWptiResult( LWpti_Profiler_Initialize_Params* pParams );
    using ProfilerDeInitialize_t           = LWptiResult( LWpti_Profiler_DeInitialize_Params* pParams );
    using DeviceGetChipName_t              = LWptiResult( LWpti_Device_GetChipName_Params* pParams );
    using ProfilerGetCounterAvailability_t = LWptiResult( LWpti_Profiler_GetCounterAvailability_Params* pParams );
    using ProfilerCounterDataImageCallwlateSize_t = LWptiResult( LWpti_Profiler_CounterDataImage_CallwlateSize_Params* pParams );
    using ProfilerCounterDataImageInitialize_t = LWptiResult( LWpti_Profiler_CounterDataImage_Initialize_Params* pParams );
    using ProfilerCounterDataImageCallwlateScratchBufferSize_t =
        LWptiResult( LWpti_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params* pParams );
    using ProfilerCounterDataImageInitializeScratchBuffer_t = LWptiResult( LWpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams );
    using ProfilerSetConfig_t                               = LWptiResult( LWpti_Profiler_SetConfig_Params* pParams );
    using ProfilerUnsetConfig_t      = LWptiResult( LWpti_Profiler_UnsetConfig_Params* pParams );
    using ProfilerBeginSession_t     = LWptiResult( LWpti_Profiler_BeginSession_Params* pParams );
    using ProfilerEndSession_t       = LWptiResult( LWpti_Profiler_EndSession_Params* pParams );
    using ProfilerPushRange_t        = LWptiResult( LWpti_Profiler_PushRange_Params* pParams );
    using ProfilerPopRange_t         = LWptiResult( LWpti_Profiler_PopRange_Params* pParams );
    using ProfilerBeginPass_t        = LWptiResult( LWpti_Profiler_BeginPass_Params* pParams );
    using ProfilerEndPass_t          = LWptiResult( LWpti_Profiler_EndPass_Params* pParams );
    using ProfilerEnableProfiling_t  = LWptiResult( LWpti_Profiler_EnableProfiling_Params* pParams );
    using ProfilerDisableProfiling_t = LWptiResult( LWpti_Profiler_DisableProfiling_Params* pParams );
    using ProfilerFlushCounterData_t = LWptiResult( LWpti_Profiler_FlushCounterData_Params* pParams );

    ProfilerInitialize_t*                                 m_profilerInitialize                    = nullptr;
    ProfilerDeInitialize_t*                               m_profilerDeInitialize                  = nullptr;
    DeviceGetChipName_t*                                  m_deviceGetChipName                     = nullptr;
    ProfilerGetCounterAvailability_t*                     m_profilerGetCounterAvailability        = nullptr;
    ProfilerCounterDataImageCallwlateSize_t*              m_profilerCounterDataImageCallwlateSize = nullptr;
    ProfilerCounterDataImageInitialize_t*                 m_profilerCounterDataImageInitialize    = nullptr;
    ProfilerCounterDataImageCallwlateScratchBufferSize_t* m_profilerCounterDataImageCallwlateScratchBufferSize = nullptr;
    ProfilerCounterDataImageInitializeScratchBuffer_t*    m_profilerCounterDataImageInitializeScratchBuffer = nullptr;
    ProfilerSetConfig_t*                                  m_profilerSetConfig                               = nullptr;
    ProfilerUnsetConfig_t*                                m_profilerUnsetConfig                             = nullptr;
    ProfilerBeginSession_t*                               m_profilerBeginSession                            = nullptr;
    ProfilerEndSession_t*                                 m_profilerEndSession                              = nullptr;
    ProfilerPushRange_t*                                  m_profilerPushRange                               = nullptr;
    ProfilerPopRange_t*                                   m_profilerPopRange                                = nullptr;
    ProfilerBeginPass_t*                                  m_profilerBeginPass                               = nullptr;
    ProfilerEndPass_t*                                    m_profilerEndPass                                 = nullptr;
    ProfilerEnableProfiling_t*                            m_profilerEnableProfiling                         = nullptr;
    ProfilerDisableProfiling_t*                           m_profilerDisableProfiling                        = nullptr;
    ProfilerFlushCounterData_t*                           m_profilerFlushCounterData                        = nullptr;
};

}  // namespace prodlib
