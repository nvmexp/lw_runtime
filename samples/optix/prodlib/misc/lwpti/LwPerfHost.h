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

#include <lwperf_lwda_host.h>
#include <lwperf_host.h>

#include <memory>

namespace prodlib {

class LwPerfHost
{
  public:
    OptixResult initialize( optix_exp::ErrorDetails& errDetails );
    bool available() const;

    LWPA_Status initializeHost( LWPW_InitializeHost_Params* pParams );
    LWPA_Status metricsContext_Destroy( LWPW_MetricsContext_Destroy_Params* pParams );
    LWPA_Status rawMetricsConfig_SetCounterAvailability( LWPW_RawMetricsConfig_SetCounterAvailability_Params* pParams );
    LWPA_Status rawMetricsConfig_BeginPassGroup( LWPW_RawMetricsConfig_BeginPassGroup_Params* pParams );
    LWPA_Status rawMetricsConfig_Create( const LWPA_RawMetricsConfigOptions* pMetricsConfigOptions,
                                         LWPA_RawMetricsConfig**             ppRawMetricsConfig );
    LWPA_Status rawMetricsConfig_Destroy( LWPW_RawMetricsConfig_Destroy_Params* pParams );
    LWPA_Status rawMetricsConfig_AddMetrics( LWPW_RawMetricsConfig_AddMetrics_Params* pParams );
    LWPA_Status rawMetricsConfig_EndPassGroup( LWPW_RawMetricsConfig_EndPassGroup_Params* pParams );
    LWPA_Status rawMetricsConfig_GenerateConfigImage( LWPW_RawMetricsConfig_GenerateConfigImage_Params* pParams );
    LWPA_Status rawMetricsConfig_GetConfigImage( LWPW_RawMetricsConfig_GetConfigImage_Params* pParams );
    LWPA_Status counterDataBuilder_Create( LWPW_CounterDataBuilder_Create_Params* pParams );
    LWPA_Status counterDataBuilder_AddMetrics( LWPW_CounterDataBuilder_AddMetrics_Params* pParams );
    LWPA_Status counterDataBuilder_GetCounterDataPrefix( LWPW_CounterDataBuilder_GetCounterDataPrefix_Params* pParams );
    LWPA_Status counterDataBuilder_Destroy( LWPW_CounterDataBuilder_Destroy_Params* pParams );
    LWPA_Status metricsContext_SetCounterData( LWPW_MetricsContext_SetCounterData_Params* pParams );
    LWPA_Status metricsContext_EvaluateToGpuValues( LWPW_MetricsContext_EvaluateToGpuValues_Params* pParams );
    LWPA_Status metricsContext_GetMetricProperties_Begin( LWPW_MetricsContext_GetMetricProperties_Begin_Params* pParams );
    LWPA_Status metricsContext_GetMetricProperties_End( LWPW_MetricsContext_GetMetricProperties_End_Params* pParams );
    LWPA_Status LWDA_MetricsContext_Create( LWPW_LWDA_MetricsContext_Create_Params* pParams );

  private:
    std::unique_ptr<corelib::ExelwtableModule> m_lib;

    // DLL capabilities
    bool m_available = false;

    using InitializeHost_t                          = LWPA_Status( LWPW_InitializeHost_Params* pParams );
    using MetricsContext_Destroy_t                  = LWPA_Status( LWPW_MetricsContext_Destroy_Params* pParams );
    using RawMetricsConfig_SetCounterAvailability_t = LWPA_Status( LWPW_RawMetricsConfig_SetCounterAvailability_Params* pParams );
    using RawMetricsConfig_BeginPassGroup_t = LWPA_Status( LWPW_RawMetricsConfig_BeginPassGroup_Params* pParams );
    using RawMetricsConfig_Create_t         = LWPA_Status( const LWPA_RawMetricsConfigOptions* pMetricsConfigOptions,
                                                   LWPA_RawMetricsConfig**             ppRawMetricsConfig );
    using RawMetricsConfig_Destroy_t             = LWPA_Status( LWPW_RawMetricsConfig_Destroy_Params* pParams );
    using RawMetricsConfig_AddMetrics_t          = LWPA_Status( LWPW_RawMetricsConfig_AddMetrics_Params* pParams );
    using RawMetricsConfig_EndPassGroup_t        = LWPA_Status( LWPW_RawMetricsConfig_EndPassGroup_Params* pParams );
    using RawMetricsConfig_GenerateConfigImage_t = LWPA_Status( LWPW_RawMetricsConfig_GenerateConfigImage_Params* pParams );
    using RawMetricsConfig_GetConfigImage_t      = LWPA_Status( LWPW_RawMetricsConfig_GetConfigImage_Params* pParams );
    using CounterDataBuilder_Create_t            = LWPA_Status( LWPW_CounterDataBuilder_Create_Params* pParams );
    using CounterDataBuilder_AddMetrics_t        = LWPA_Status( LWPW_CounterDataBuilder_AddMetrics_Params* pParams );
    using CounterDataBuilder_GetCounterDataPrefix_t = LWPA_Status( LWPW_CounterDataBuilder_GetCounterDataPrefix_Params* pParams );
    using CounterDataBuilder_Destroy_t              = LWPA_Status( LWPW_CounterDataBuilder_Destroy_Params* pParams );
    using MetricsContext_SetCounterData_t           = LWPA_Status( LWPW_MetricsContext_SetCounterData_Params* pParams );
    using MetricsContext_EvaluateToGpuValues_t = LWPA_Status( LWPW_MetricsContext_EvaluateToGpuValues_Params* pParams );
    using MetricsContext_GetMetricProperties_Begin_t = LWPA_Status( LWPW_MetricsContext_GetMetricProperties_Begin_Params* pParams );
    using MetricsContext_GetMetricProperties_End_t = LWPA_Status( LWPW_MetricsContext_GetMetricProperties_End_Params* pParams );
    using LWDA_MetricsContext_Create_t             = LWPA_Status( LWPW_LWDA_MetricsContext_Create_Params* pParams );

    InitializeHost_t*                           m_initializeHost                           = nullptr;
    MetricsContext_Destroy_t*                   m_metricsContext_Destroy                   = nullptr;
    RawMetricsConfig_SetCounterAvailability_t*  m_rawMetricsConfig_SetCounterAvailability  = nullptr;
    RawMetricsConfig_BeginPassGroup_t*          m_rawMetricsConfig_BeginPassGroup          = nullptr;
    RawMetricsConfig_Create_t*                  m_rawMetricsConfig_Create                  = nullptr;
    RawMetricsConfig_Destroy_t*                 m_rawMetricsConfig_Destroy                 = nullptr;
    RawMetricsConfig_AddMetrics_t*              m_rawMetricsConfig_AddMetrics              = nullptr;
    RawMetricsConfig_EndPassGroup_t*            m_rawMetricsConfig_EndPassGroup            = nullptr;
    RawMetricsConfig_GenerateConfigImage_t*     m_rawMetricsConfig_GenerateConfigImage     = nullptr;
    RawMetricsConfig_GetConfigImage_t*          m_rawMetricsConfig_GetConfigImage          = nullptr;
    CounterDataBuilder_Create_t*                m_counterDataBuilder_Create                = nullptr;
    CounterDataBuilder_AddMetrics_t*            m_counterDataBuilder_AddMetrics            = nullptr;
    CounterDataBuilder_GetCounterDataPrefix_t*  m_counterDataBuilder_GetCounterDataPrefix  = nullptr;
    CounterDataBuilder_Destroy_t*               m_counterDataBuilder_Destroy               = nullptr;
    MetricsContext_SetCounterData_t*            m_metricsContext_SetCounterData            = nullptr;
    MetricsContext_EvaluateToGpuValues_t*       m_metricsContext_EvaluateToGpuValues       = nullptr;
    MetricsContext_GetMetricProperties_Begin_t* m_metricsContext_GetMetricProperties_Begin = nullptr;
    MetricsContext_GetMetricProperties_End_t*   m_metricsContext_GetMetricProperties_End   = nullptr;
    LWDA_MetricsContext_Create_t*               m_LWDA_MetricsContext_Create               = nullptr;
};
}
