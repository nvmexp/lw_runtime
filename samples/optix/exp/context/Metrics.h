/*
 * Copyright (c) 2021 LWPU CORPORATION.  All rights reserved.
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

#include <optix_types.h>

#include <mutex>
#include <string>
#include <vector>

#include <exp/context/ErrorHandling.h>

#include <Util/JsonEscape.h>
#include <Util/SystemInfo.h>

#include <corelib/misc/String.h>
#include <corelib/system/Timer.h>

#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/misc/LwdaStopwatch.h>
#include <prodlib/system/Knobs.h>

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>

#include <stdint.h>
#include <stdio.h>

namespace optix_exp {

class DeviceContext;

// RAII for the metrics timer calls. Otherwise there is the danger of early returns inside the measured function call.
class StreamMetricTimer
{
  public:
    StreamMetricTimer( optix_exp::DeviceContext* deviceContext, LWstream stream, std::string metricName );
    ~StreamMetricTimer();

  private:
    optix_exp::DeviceContext* m_deviceContext;
    LWstream                  m_stream;
    std::string               m_metricName;
};

class Metrics
{
  public:
    enum ScopeType
    {
        OBJECT,
        ARRAY
    };

    Metrics( DeviceContext* context );
    ~Metrics();
    OptixResult destroy( ErrorDetails& errDetails );

    OptixResult init( std::string deviceName, std::string deviceId, std::string contextId, ErrorDetails& errDetails );
    bool        isEnabled() const;
    OptixResult logInt( const char* name, uint64_t value, ErrorDetails& errDetails );
    OptixResult logFloat( const char* name, double value, ErrorDetails& errDetails );
    OptixResult logString( const char* name, const std::string& value, ErrorDetails& errDetails );
    OptixResult flush( ErrorDetails& errorDetails );

    void startAsyncTimer( const std::string& name, lwdaStream_t stream );
    void stopAsyncTimerAndRecordMetric( const std::string& name, lwdaStream_t stream );

  private:
    struct Scope
    {
        std::string        name;
        Metrics::ScopeType type;
        int                numEntries;
    };

    OptixResult pushScope( const char* name, ErrorDetails& errDetails, Metrics::ScopeType type = Metrics::OBJECT );
    OptixResult popScope( ErrorDetails& errDetails );
    OptixResult append( const std::string& str, ErrorDetails& errDetails );
    OptixResult registerMetric( const char* name, ErrorDetails& errDetails );
    double      getTimestamp() const;
    OptixResult log( const char* name, const std::string& valueStr, ErrorDetails& errDetails );
    std::string sep( char c = ',' );

    DeviceContext*         m_deviceContext = nullptr;
    corelib::timerTick     m_start         = 0;
    std::atomic<uint64_t>  m_timestamp     = {0};
    FILE*                  m_file          = nullptr;
    bool                   m_enabled       = false;
    std::mutex             m_mutex;
    std::vector<char>      m_buffer;
    std::set<std::string>  m_registeredMetrics;
    std::vector<Scope>     m_scopeStack;
    prodlib::LwdaStopwatch m_lwdaStopwatch;
};

}  // end namespace optix_exp