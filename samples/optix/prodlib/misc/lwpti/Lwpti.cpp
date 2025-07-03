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
#include <prodlib/misc/lwpti/Lwpti.h>

#include <prodlib/system/Knobs.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Shlobj.h>
#include <comutil.h>
#include <windows.h>
#endif

namespace {

// clang-format off
Knob<std::string> k_lwptiDir( RT_DSTRING( "launch.lwptiDirectory" ), "", RT_DSTRING( "Directory where lwpti, lwperf_host, and lwperf_target dynamic libraries are located." ) );
Knob<std::string> k_lwptiDllName( RT_DSTRING( "launch.lwptiDllName" ), "lwpti64_2020.2.0.dll", RT_DSTRING( "The name of the LWPTI dynamic library. Only used on Windows (we always load liblwpti.so on Linux)." ) );
// clang-format on

}  // namespace


namespace prodlib {

bool Lwpti::available() const
{
    return m_available;
}

// Helper for casting function prototypes from an ExelwtableModule.
template <typename Fn>
void getFunction( Fn*& ptr, corelib::ExelwtableModule& module, const char* name )
{
    ptr = reinterpret_cast<Fn*>( module.getFunction( name ) );
}

OptixResult Lwpti::initialize( optix_exp::ErrorDetails& errDetails )
{
    std::string dllPathPrefix;
#if defined( _WIN32 )
    if( !k_lwptiDir.get().empty() )
        dllPathPrefix         = k_lwptiDir.get() + "\\";
    const std::string dllPath = dllPathPrefix + k_lwptiDllName.get();
#elif defined( __linux__ )
    if( !k_lwptiDir.get().empty() )
        dllPathPrefix         = k_lwptiDir.get() + "/";
    const std::string dllPath = "liblwpti.so";
#endif
    std::unique_ptr<corelib::ExelwtableModule> lwptiLib( new corelib::ExelwtableModule( dllPath.c_str() ) );

    if( !lwptiLib->init() )
        return errDetails.logDetails(
            OPTIX_ERROR_UNKNOWN, corelib::stringf( "LWPTI loading error, cannot open shared library at: %s\nERROR: %s",
                                                   dllPath.c_str(), corelib::getSystemErrorString().c_str() ) );

    getFunction( m_profilerInitialize, *lwptiLib, "lwptiProfilerInitialize" );
    getFunction( m_profilerDeInitialize, *lwptiLib, "lwptiProfilerDeInitialize" );
    getFunction( m_deviceGetChipName, *lwptiLib, "lwptiDeviceGetChipName" );
    getFunction( m_profilerGetCounterAvailability, *lwptiLib, "lwptiProfilerGetCounterAvailability" );
    getFunction( m_profilerCounterDataImageCallwlateSize, *lwptiLib, "lwptiProfilerCounterDataImageCallwlateSize" );
    getFunction( m_profilerCounterDataImageInitialize, *lwptiLib, "lwptiProfilerCounterDataImageInitialize" );
    getFunction( m_profilerCounterDataImageCallwlateScratchBufferSize, *lwptiLib,
                 "lwptiProfilerCounterDataImageCallwlateScratchBufferSize" );
    getFunction( m_profilerCounterDataImageInitializeScratchBuffer, *lwptiLib,
                 "lwptiProfilerCounterDataImageInitializeScratchBuffer" );
    getFunction( m_profilerSetConfig, *lwptiLib, "lwptiProfilerSetConfig" );
    getFunction( m_profilerUnsetConfig, *lwptiLib, "lwptiProfilerUnsetConfig" );
    getFunction( m_profilerBeginSession, *lwptiLib, "lwptiProfilerBeginSession" );
    getFunction( m_profilerEndSession, *lwptiLib, "lwptiProfilerEndSession" );
    getFunction( m_profilerPushRange, *lwptiLib, "lwptiProfilerPushRange" );
    getFunction( m_profilerPopRange, *lwptiLib, "lwptiProfilerPopRange" );
    getFunction( m_profilerBeginPass, *lwptiLib, "lwptiProfilerBeginPass" );
    getFunction( m_profilerEndPass, *lwptiLib, "lwptiProfilerEndPass" );
    getFunction( m_profilerEnableProfiling, *lwptiLib, "lwptiProfilerEnableProfiling" );
    getFunction( m_profilerDisableProfiling, *lwptiLib, "lwptiProfilerDisableProfiling" );
    getFunction( m_profilerFlushCounterData, *lwptiLib, "lwptiProfilerFlushCounterData" );

    // If we didn't find the initialization function, the library isn't available
    if( !m_profilerInitialize )
        return errDetails.logDetails( OPTIX_ERROR_UNKNOWN, "lwptiProfilerInitialize not found in LWPTI library\n" );

    m_lib       = std::move( lwptiLib );
    m_available = true;

    return OPTIX_SUCCESS;
}

LWptiResult Lwpti::profilerInitialize( LWpti_Profiler_Initialize_Params* pParams ) const
{
    if( !available() || !m_profilerInitialize )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerInitialize( pParams );
}

LWptiResult Lwpti::profilerDeInitialize( LWpti_Profiler_DeInitialize_Params* pParams ) const
{
    if( !available() || !m_profilerDeInitialize )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerDeInitialize( pParams );
}

LWptiResult Lwpti::deviceGetChipName( LWpti_Device_GetChipName_Params* pParams ) const
{
    if( !available() || !m_deviceGetChipName )
        return LWPTI_ERROR_UNKNOWN;
    return m_deviceGetChipName( pParams );
}

LWptiResult Lwpti::profilerGetCounterAvailability( LWpti_Profiler_GetCounterAvailability_Params* pParams ) const
{
    if( !available() || !m_profilerGetCounterAvailability )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerGetCounterAvailability( pParams );
}

LWptiResult Lwpti::profilerCounterDataImageCallwlateSize( LWpti_Profiler_CounterDataImage_CallwlateSize_Params* pParams ) const
{
    if( !available() || !m_profilerCounterDataImageCallwlateSize )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerCounterDataImageCallwlateSize( pParams );
}

LWptiResult Lwpti::profilerCounterDataImageInitialize( LWpti_Profiler_CounterDataImage_Initialize_Params* pParams ) const
{
    if( !available() || !m_profilerCounterDataImageInitialize )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerCounterDataImageInitialize( pParams );
}

LWptiResult Lwpti::profilerCounterDataImageCallwlateScratchBufferSize( LWpti_Profiler_CounterDataImage_CallwlateScratchBufferSize_Params* pParams ) const
{
    if( !available() || !m_profilerCounterDataImageCallwlateScratchBufferSize )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerCounterDataImageCallwlateScratchBufferSize( pParams );
}

LWptiResult Lwpti::profilerCounterDataImageInitializeScratchBuffer( LWpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* pParams ) const
{
    if( !available() || !m_profilerCounterDataImageInitializeScratchBuffer )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerCounterDataImageInitializeScratchBuffer( pParams );
}

LWptiResult Lwpti::profilerSetConfig( LWpti_Profiler_SetConfig_Params* pParams ) const
{
    if( !available() || !m_profilerSetConfig )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerSetConfig( pParams );
}

LWptiResult Lwpti::profilerUnsetConfig( LWpti_Profiler_UnsetConfig_Params* pParams ) const
{
    if( !available() || !m_profilerUnsetConfig )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerUnsetConfig( pParams );
}

LWptiResult Lwpti::profilerBeginSession( LWpti_Profiler_BeginSession_Params* pParams ) const
{
    if( !available() || !m_profilerBeginSession )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerBeginSession( pParams );
}

LWptiResult Lwpti::profilerEndSession( LWpti_Profiler_EndSession_Params* pParams ) const
{
    if( !available() || !m_profilerEndSession )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerEndSession( pParams );
}

LWptiResult Lwpti::profilerPushRange( LWpti_Profiler_PushRange_Params* pParams ) const
{
    if( !available() || !m_profilerPushRange )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerPushRange( pParams );
}

LWptiResult Lwpti::profilerPopRange( LWpti_Profiler_PopRange_Params* pParams ) const
{
    if( !available() || !m_profilerPopRange )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerPopRange( pParams );
}

LWptiResult Lwpti::profilerBeginPass( LWpti_Profiler_BeginPass_Params* pParams ) const
{
    if( !available() || !m_profilerBeginPass )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerBeginPass( pParams );
}

LWptiResult Lwpti::profilerEndPass( LWpti_Profiler_EndPass_Params* pParams ) const
{
    if( !available() || !m_profilerEndPass )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerEndPass( pParams );
}

LWptiResult Lwpti::profilerEnableProfiling( LWpti_Profiler_EnableProfiling_Params* pParams ) const
{
    if( !available() || !m_profilerEnableProfiling )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerEnableProfiling( pParams );
}

LWptiResult Lwpti::profilerDisableProfiling( LWpti_Profiler_DisableProfiling_Params* pParams ) const
{
    if( !available() || !m_profilerDisableProfiling )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerDisableProfiling( pParams );
}

LWptiResult Lwpti::profilerFlushCounterData( LWpti_Profiler_FlushCounterData_Params* pParams ) const
{
    if( !available() || !m_profilerFlushCounterData )
        return LWPTI_ERROR_UNKNOWN;
    return m_profilerFlushCounterData( pParams );
}
}
