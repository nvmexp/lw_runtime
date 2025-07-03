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
#include <prodlib/misc/lwpti/LwPerfTarget.h>

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
// clang-format on

}  // namespace

namespace prodlib {

bool LwPerfTarget::available() const
{
    return m_available;
}

// Helper for casting function prototypes from an ExelwtableModule.
template <typename Fn>
void getFunction( Fn*& ptr, corelib::ExelwtableModule& module, const char* name )
{
    ptr = reinterpret_cast<Fn*>( module.getFunction( name ) );
}

OptixResult LwPerfTarget::initialize( optix_exp::ErrorDetails& errDetails )
{
    std::string dllPathPrefix;
#if defined( _WIN32 )
    if( !k_lwptiDir.get().empty() )
        dllPathPrefix         = k_lwptiDir.get() + "\\";
    const std::string dllPath = dllPathPrefix + "lwperf_target.dll";
#elif defined( __linux__ )
    if( !k_lwptiDir.get().empty() )
        dllPathPrefix         = k_lwptiDir.get() + "/";
    const std::string dllPath = dllPathPrefix + "liblwperf_target.so";
#endif
    std::unique_ptr<corelib::ExelwtableModule> lwPerfTargetLib( new corelib::ExelwtableModule( dllPath.c_str() ) );

    if( !lwPerfTargetLib->init() )
        return errDetails.logDetails(
            OPTIX_ERROR_UNKNOWN,
            corelib::stringf( "lwperf_target loading error, cannot open shared library at %s\nERROR: %s\n",
                              dllPath.c_str(), corelib::getSystemErrorString().c_str() ) );

    getFunction( m_counterData_GetNumRanges, *lwPerfTargetLib, "LWPW_CounterData_GetNumRanges" );
    getFunction( m_profiler_CounterData_GetRangeDescriptions, *lwPerfTargetLib,
                 "LWPW_Profiler_CounterData_GetRangeDescriptions" );

    // If we didn't find the initialization function, the library isn't available
    if( !m_counterData_GetNumRanges )
        return errDetails.logDetails( OPTIX_ERROR_UNKNOWN,
                                      "LWPW_CounterData_GetNumRanges not found in lwPerfTarget library\n" );

    m_lib       = std::move( lwPerfTargetLib );
    m_available = true;

    return OPTIX_SUCCESS;
}

LWPA_Status LwPerfTarget::counterData_GetNumRanges( LWPW_CounterData_GetNumRanges_Params* pParams )
{
    if( !available() || !m_counterData_GetNumRanges )
        return LWPA_STATUS_ERROR;
    return m_counterData_GetNumRanges( pParams );
}

LWPA_Status LwPerfTarget::profiler_CounterData_GetRangeDescriptions( LWPW_Profiler_CounterData_GetRangeDescriptions_Params* pParams )
{
    if( !available() || !m_profiler_CounterData_GetRangeDescriptions )
        return LWPA_STATUS_ERROR;
    return m_profiler_CounterData_GetRangeDescriptions( pParams );
}
}
