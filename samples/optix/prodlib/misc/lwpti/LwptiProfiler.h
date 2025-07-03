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

#include <prodlib/misc/lwpti/Lwpti.h>
#include <prodlib/misc/lwpti/LwPerfHost.h>
#include <prodlib/misc/lwpti/LwPerfTarget.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>

namespace prodlib {

class LwptiMetricsJson
{
  public:
    bool open( const std::string& filePath );
    void addLaunch();
    void addMetricToLaunch( const std::string& metricName, double metricValue );
    void finishLaunch();
    void endDolwment();
    void flushIfFullBuffer();
    void flush();
    bool isOpen() const;

  private:
    std::ostringstream m_buffer;
    std::ofstream      m_outputFile;
    bool               m_firstLaunch = true;
    bool               m_firstMetric = true;
    bool               m_isOpen      = false;
};

class LwptiProfiler
{
  public:
    OptixResult initialize( LWcontext lwdaContext, const std::string& metricsToCollect, optix_exp::ErrorDetails& errDetails );
    OptixResult deinitialize( optix_exp::ErrorDetails& errDetails );

    bool openOutputFile( const std::string& filePath );
    void closeOutputFile();

    OptixResult beginProfile( optix_exp::ErrorDetails& errDetails );
    OptixResult endProfile( optix_exp::ErrorDetails& errDetails );

    bool isInitialized() const { return m_isInitialized; }

  private:
    // Dynamically loaded LWPTI library.
    Lwpti        m_lwpti;
    LwPerfHost   m_lwPerfHost;
    LwPerfTarget m_lwPerfTarget;

    std::vector<std::string> m_metricNames;
    std::string              m_chipName;
    std::vector<uint8_t>     m_counterAvailabilityImage;
    std::vector<uint8_t>     m_configImage;
    std::vector<uint8_t>     m_counterDataImagePrefix;
    std::vector<uint8_t>     m_counterDataImage;
    std::vector<uint8_t>     m_counterDataScratchBuffer;
    LwptiMetricsJson         m_outputFile;
    bool                     m_isInitialized = false;

    // These names must stay allocated until their corresponding raw metric requests are used.
    std::vector<std::string>           m_rawMetricNames;
    std::vector<LWPA_RawMetricRequest> m_rawMetricRequests;
    OptixResult getRawMetricRequests( LWPA_MetricsContext* pMetricsContext, optix_exp::ErrorDetails& errDetails );

    OptixResult createConfigImage( optix_exp::ErrorDetails& errDetails );
    OptixResult getCounterDataPrefixImage( optix_exp::ErrorDetails& errDetails );
    OptixResult createCounterDataImage( optix_exp::ErrorDetails& errDetails );
    OptixResult addMetricValuesToJson( optix_exp::ErrorDetails& errDetails );
};
}
