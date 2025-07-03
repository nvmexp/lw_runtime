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
#include <prodlib/misc/lwpti/LwPerfHost.h>

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
}

namespace prodlib {

bool LwPerfHost::available() const
{
    return m_available;
}

// Helper for casting function prototypes from an ExelwtableModule.
template <typename Fn>
void getFunction( Fn*& ptr, corelib::ExelwtableModule& module, const char* name )
{
    ptr = reinterpret_cast<Fn*>( module.getFunction( name ) );
}

OptixResult LwPerfHost::initialize( optix_exp::ErrorDetails& errDetails )
{
    std::string dllPathPrefix;
#if defined( _WIN32 )
    if( !k_lwptiDir.get().empty() )
        dllPathPrefix         = k_lwptiDir.get() + "\\";
    const std::string dllPath = dllPathPrefix + "lwperf_host.dll";
#elif defined( __linux__ )
    if( !k_lwptiDir.get().empty() )
        dllPathPrefix         = k_lwptiDir.get() + "/";
    const std::string dllPath = dllPathPrefix + "liblwperf_host.so";
#endif
    std::unique_ptr<corelib::ExelwtableModule> lib( new corelib::ExelwtableModule( dllPath.c_str() ) );

    if( !lib->init() )
        return errDetails.logDetails(
            OPTIX_ERROR_UNKNOWN,
            corelib::stringf( "lwperf_host loading error, cannot open shared library at: %s\nERROR: %s\n",
                              dllPath.c_str(), corelib::getSystemErrorString().c_str() ) );

    getFunction( m_initializeHost, *lib, "LWPW_InitializeHost" );
    getFunction( m_metricsContext_Destroy, *lib, "LWPW_MetricsContext_Destroy" );
    getFunction( m_rawMetricsConfig_SetCounterAvailability, *lib, "LWPW_RawMetricsConfig_SetCounterAvailability" );
    getFunction( m_rawMetricsConfig_BeginPassGroup, *lib, "LWPW_RawMetricsConfig_BeginPassGroup" );
    getFunction( m_rawMetricsConfig_Create, *lib, "LWPA_RawMetricsConfig_Create" );
    getFunction( m_rawMetricsConfig_Destroy, *lib, "LWPW_RawMetricsConfig_Destroy" );
    getFunction( m_rawMetricsConfig_AddMetrics, *lib, "LWPW_RawMetricsConfig_AddMetrics" );
    getFunction( m_rawMetricsConfig_EndPassGroup, *lib, "LWPW_RawMetricsConfig_EndPassGroup" );
    getFunction( m_rawMetricsConfig_GenerateConfigImage, *lib, "LWPW_RawMetricsConfig_GenerateConfigImage" );
    getFunction( m_rawMetricsConfig_GetConfigImage, *lib, "LWPW_RawMetricsConfig_GetConfigImage" );
    getFunction( m_counterDataBuilder_Create, *lib, "LWPW_CounterDataBuilder_Create" );
    getFunction( m_counterDataBuilder_AddMetrics, *lib, "LWPW_CounterDataBuilder_AddMetrics" );
    getFunction( m_counterDataBuilder_GetCounterDataPrefix, *lib, "LWPW_CounterDataBuilder_GetCounterDataPrefix" );
    getFunction( m_counterDataBuilder_Destroy, *lib, "LWPW_CounterDataBuilder_Destroy" );
    getFunction( m_metricsContext_SetCounterData, *lib, "LWPW_MetricsContext_SetCounterData" );
    getFunction( m_metricsContext_EvaluateToGpuValues, *lib, "LWPW_MetricsContext_EvaluateToGpuValues" );
    getFunction( m_metricsContext_GetMetricProperties_Begin, *lib, "LWPW_MetricsContext_GetMetricProperties_Begin" );
    getFunction( m_metricsContext_GetMetricProperties_End, *lib, "LWPW_MetricsContext_GetMetricProperties_End" );
    getFunction( m_LWDA_MetricsContext_Create, *lib, "LWPW_LWDA_MetricsContext_Create" );

    // If we didn't find the initialization function, the library isn't available
    if( !m_initializeHost )
        return errDetails.logDetails( OPTIX_ERROR_UNKNOWN, "initializeHost not found in lwperf_host library\n" );

    m_lib       = std::move( lib );
    m_available = true;

    return OPTIX_SUCCESS;
}

LWPA_Status LwPerfHost::initializeHost( LWPW_InitializeHost_Params* pParams )
{
    if( !available() || !m_initializeHost )
        return LWPA_STATUS_ERROR;
    return m_initializeHost( pParams );
}

LWPA_Status LwPerfHost::metricsContext_Destroy( LWPW_MetricsContext_Destroy_Params* pParams )
{
    if( !available() || !m_metricsContext_Destroy )
        return LWPA_STATUS_ERROR;
    return m_metricsContext_Destroy( pParams );
}

LWPA_Status LwPerfHost::rawMetricsConfig_SetCounterAvailability( LWPW_RawMetricsConfig_SetCounterAvailability_Params* pParams )
{
    if( !available() || !m_rawMetricsConfig_SetCounterAvailability )
        return LWPA_STATUS_ERROR;
    return m_rawMetricsConfig_SetCounterAvailability( pParams );
}

LWPA_Status LwPerfHost::rawMetricsConfig_BeginPassGroup( LWPW_RawMetricsConfig_BeginPassGroup_Params* pParams )
{
    if( !available() || !m_rawMetricsConfig_BeginPassGroup )
        return LWPA_STATUS_ERROR;
    return m_rawMetricsConfig_BeginPassGroup( pParams );
}

LWPA_Status LwPerfHost::rawMetricsConfig_Create( const LWPA_RawMetricsConfigOptions* pMetricsConfigOptions,
                                                 LWPA_RawMetricsConfig**             ppRawMetricsConfig )
{
    if( !available() || !m_rawMetricsConfig_Create )
        return LWPA_STATUS_ERROR;
    return m_rawMetricsConfig_Create( pMetricsConfigOptions, ppRawMetricsConfig );
}

LWPA_Status LwPerfHost::rawMetricsConfig_Destroy( LWPW_RawMetricsConfig_Destroy_Params* pParams )
{
    if( !available() || !m_rawMetricsConfig_Destroy )
        return LWPA_STATUS_ERROR;
    return m_rawMetricsConfig_Destroy( pParams );
}

LWPA_Status LwPerfHost::rawMetricsConfig_AddMetrics( LWPW_RawMetricsConfig_AddMetrics_Params* pParams )
{
    if( !available() || !m_rawMetricsConfig_AddMetrics )
        return LWPA_STATUS_ERROR;
    return m_rawMetricsConfig_AddMetrics( pParams );
}

LWPA_Status LwPerfHost::rawMetricsConfig_EndPassGroup( LWPW_RawMetricsConfig_EndPassGroup_Params* pParams )
{
    if( !available() || !m_rawMetricsConfig_EndPassGroup )
        return LWPA_STATUS_ERROR;
    return m_rawMetricsConfig_EndPassGroup( pParams );
}

LWPA_Status LwPerfHost::rawMetricsConfig_GenerateConfigImage( LWPW_RawMetricsConfig_GenerateConfigImage_Params* pParams )
{
    if( !available() || !m_rawMetricsConfig_GenerateConfigImage )
        return LWPA_STATUS_ERROR;
    return m_rawMetricsConfig_GenerateConfigImage( pParams );
}

LWPA_Status LwPerfHost::rawMetricsConfig_GetConfigImage( LWPW_RawMetricsConfig_GetConfigImage_Params* pParams )
{
    if( !available() || !m_rawMetricsConfig_GetConfigImage )
        return LWPA_STATUS_ERROR;
    return m_rawMetricsConfig_GetConfigImage( pParams );
}

LWPA_Status LwPerfHost::counterDataBuilder_Create( LWPW_CounterDataBuilder_Create_Params* pParams )
{
    if( !available() || !m_counterDataBuilder_Create )
        return LWPA_STATUS_ERROR;
    return m_counterDataBuilder_Create( pParams );
}

LWPA_Status LwPerfHost::counterDataBuilder_AddMetrics( LWPW_CounterDataBuilder_AddMetrics_Params* pParams )
{
    if( !available() || !m_counterDataBuilder_AddMetrics )
        return LWPA_STATUS_ERROR;
    return m_counterDataBuilder_AddMetrics( pParams );
}

LWPA_Status LwPerfHost::counterDataBuilder_GetCounterDataPrefix( LWPW_CounterDataBuilder_GetCounterDataPrefix_Params* pParams )
{
    if( !available() || !m_counterDataBuilder_GetCounterDataPrefix )
        return LWPA_STATUS_ERROR;
    return m_counterDataBuilder_GetCounterDataPrefix( pParams );
}

LWPA_Status LwPerfHost::counterDataBuilder_Destroy( LWPW_CounterDataBuilder_Destroy_Params* pParams )
{
    if( !available() || !m_counterDataBuilder_Destroy )
        return LWPA_STATUS_ERROR;
    return m_counterDataBuilder_Destroy( pParams );
}

LWPA_Status LwPerfHost::metricsContext_SetCounterData( LWPW_MetricsContext_SetCounterData_Params* pParams )
{
    if( !available() || !m_metricsContext_SetCounterData )
        return LWPA_STATUS_ERROR;
    return m_metricsContext_SetCounterData( pParams );
}

LWPA_Status LwPerfHost::metricsContext_EvaluateToGpuValues( LWPW_MetricsContext_EvaluateToGpuValues_Params* pParams )
{
    if( !available() || !m_metricsContext_EvaluateToGpuValues )
        return LWPA_STATUS_ERROR;
    return m_metricsContext_EvaluateToGpuValues( pParams );
}

LWPA_Status LwPerfHost::metricsContext_GetMetricProperties_Begin( LWPW_MetricsContext_GetMetricProperties_Begin_Params* pParams )
{
    if( !available() || !m_metricsContext_GetMetricProperties_Begin )
        return LWPA_STATUS_ERROR;
    return m_metricsContext_GetMetricProperties_Begin( pParams );
}

LWPA_Status LwPerfHost::metricsContext_GetMetricProperties_End( LWPW_MetricsContext_GetMetricProperties_End_Params* pParams )
{
    if( !available() || !m_metricsContext_GetMetricProperties_End )
        return LWPA_STATUS_ERROR;
    return m_metricsContext_GetMetricProperties_End( pParams );
}

LWPA_Status LwPerfHost::LWDA_MetricsContext_Create( LWPW_LWDA_MetricsContext_Create_Params* pParams )
{
    if( !available() || !m_LWDA_MetricsContext_Create )
        return LWPA_STATUS_ERROR;
    return m_LWDA_MetricsContext_Create( pParams );
}
}
