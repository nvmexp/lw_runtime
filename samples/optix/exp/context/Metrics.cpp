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

#include <exp/context/DeviceContext.h>
#include <exp/context/Metrics.h>

#include <Util/JsonEscape.h>
#include <Util/SystemInfo.h>

#include <corelib/misc/String.h>
#include <corelib/system/Timer.h>
#include <corelib/system/System.h>

#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/system/Knobs.h>

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>

const int DEFAULT_BUFFER_SIZE = 32 * 1024;

namespace optix_exp {

StreamMetricTimer::StreamMetricTimer( optix_exp::DeviceContext* deviceContext, LWstream stream, std::string metricName )
    : m_deviceContext( deviceContext )
    , m_stream( stream )
    , m_metricName( metricName )
{
    m_deviceContext->startAsyncTimer( metricName, m_stream );
}

StreamMetricTimer::~StreamMetricTimer()
{
    m_deviceContext->stopAsyncTimerAndRecordMetric( m_metricName, m_stream );
}

struct MetricDescription
{
    const char* name;  // Name of metric. Use "<tag>" for embedded tags
    const char* description;
};

// Removes '<.*>' from the string
static std::string stripEmbeddedVal( const std::string& str )
{
    size_t angleStartPos = str.find( '<' );
    if( angleStartPos == str.npos )
        return str;

    size_t angleEndPos = str.find( '>' );
    if( angleEndPos == str.npos )
        return str;

    return str.substr( 0, angleStartPos ) + str.substr( angleEndPos + 1 );
}

static std::string doubleToSingleQuotes( const std::string& str )
{
    std::string ret = str;
    for( char& c : ret )
        if( c == '"' )
            c = '\'';
    return ret;
}

Metrics::Metrics( DeviceContext* deviceContext )
    : m_deviceContext( deviceContext )
{
}

Metrics::~Metrics()
{
}

OptixResult Metrics::destroy( ErrorDetails& errDetails )
{
    if( m_enabled )
    {

        // log all async metrics
        prodlib::ElapsedAsyncTimesMap elapsedTimes;
        m_lwdaStopwatch.getElapsedTimes( elapsedTimes );

        // We don't really care what stream a metrics came from, just need them to be unique. 
        int phonyStreamId = 0;
        for( const auto& streamPair : elapsedTimes )
        {
            for( auto& namePair : streamPair.second )
            {
                std::string name = namePair.first + "_strm:" + std::to_string( phonyStreamId );
                auto& timeVector = namePair.second;
                logFloat( name.c_str(), timeVector.front(), errDetails );

                // If more than one timepoint exists for a name, make the names distinct. 
                if( timeVector.size() > 1 )
                    for( int i = 1; i < timeVector.size(); ++i )
                        logFloat( ( name + "_" + std::to_string(i) ).c_str(), timeVector[i], errDetails );
            }
            ++phonyStreamId;
        }

        logFloat( "total_sec", corelib::getDeltaSeconds( m_start ), errDetails );

        popScope( errDetails );  // end metrics
        popScope( errDetails );
        flush( errDetails );
        if( fclose( m_file ) )
        {
            std::string errString = "Error closing metrics file - ";
            return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errString.append( strerror( errno ) ) );
        }
    }
    return OPTIX_SUCCESS;
}

OptixResult Metrics::init( std::string deviceName, std::string deviceId, std::string contextId, ErrorDetails& errDetails )
{
    {
        std::replace( deviceName.begin(), deviceName.end(), ' ', '_' );
        const std::string fileName = deviceName + "-" + deviceId + "-" + contextId + "-metrics.json";

        std::string metricsDir;
        corelib::getelw( "OPTIX_O7_METRICS_DIR", metricsDir );
        const std::string metricsPath = metricsDir + fileName;

        if( ( m_file = fopen( metricsPath.c_str(), "wt" ) ) == nullptr )
            return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, "Error opening metrics file." );

        m_buffer.reserve( DEFAULT_BUFFER_SIZE );
        append( "{", errDetails );
        m_scopeStack.push_back( {"", Metrics::OBJECT, 0} );

        m_enabled = true;
    }

    m_start = corelib::getTimerTick();

    pushScope( "device_info", errDetails );
    logString( "device_name", deviceName, errDetails );
	logString( "deviceId", deviceId, errDetails );
	logString( "contextId", contextId, errDetails );
    logInt( "abi_version", static_cast<uint64_t>( m_deviceContext->getAbiVersion() ), errDetails );
    logInt( "compute_capability", m_deviceContext->getComputeCapability(), errDetails );
    logString( "has_RTCore_unit", m_deviceContext->hasTTU() ? "true" : "false", errDetails );
    logString( "has_motion_RTCore_unit", m_deviceContext->hasMotionTTU() ? "true" : "false", errDetails );
    logInt( "max_threads_per_multiproc", m_deviceContext->getMaxThreadsPerMultiProcessor(), errDetails );
    logInt( "max_threads_per_block", m_deviceContext->getMaxThreadsPerBlock(), errDetails );
    logInt( "multiprocessor_count", m_deviceContext->getMultiProcessorCount(), errDetails );
    logInt( "architecture", m_deviceContext->getArchitecture(), errDetails );
    logInt( "architecture_impl", m_deviceContext->getArchitectureImplementation(), errDetails );
    logInt( "max_trace_relwrsion_depth", m_deviceContext->getRtcMaxTraceRelwrsionDepth(), errDetails );
    logInt( "sbt_header_size", m_deviceContext->getSbtHeaderSize(), errDetails );
    logInt( "max_scene_graph_depth", m_deviceContext->getMaxSceneGraphDepth(), errDetails );
    logInt( "max_prims_per_gas", m_deviceContext->getMaxPrimsPerGAS(), errDetails );
    logInt( "max_sbt_records_per_gas", m_deviceContext->getMaxSbtRecordsPerGAS(), errDetails );
    logInt( "max_instances_per_ias", m_deviceContext->getMaxInstancesPerIAS(), errDetails );
    logInt( "max_instance_id", m_deviceContext->getMaxInstanceId(), errDetails );
    logInt( "max_sbt_offset", m_deviceContext->getMaxSbtOffset(), errDetails );
    logInt( "callable_param_register_count", m_deviceContext->getCallableParamRegCount(), errDetails );
    popScope( errDetails );

    const optix::SystemInfo info{optix::getSystemInfo()};
    pushScope( "system_info", errDetails );
	logString( "contextId", contextId, errDetails );
    logString( "host_name", info.hostName, errDetails );
    logString( "platform", info.platform, errDetails );
    logString( "cpu_name", info.cpuName, errDetails );
    logInt( "num_cpu_cores", info.numCpuCores, errDetails );
    logString( "driver_version", info.driverVersion, errDetails );
    popScope( errDetails );

    std::string nondefaultKnobList;
    for( const std::string& nondefaultKnob : info.nondefaultKnobs )
    {
        if( corelib::stringBeginsWith( nondefaultKnob, "metrics.runJson" ) )
            continue;  // filter this out because the values will already be included as part of run_info
        nondefaultKnobList += std::string( ( !nondefaultKnobList.empty() ) ? "," : "" ) + "\""
                              + optix::escapeJsonString( doubleToSingleQuotes( nondefaultKnob ) ) + "\"";
    }

    pushScope( "run_info", errDetails );
	logString( "contextId", contextId, errDetails );
    logString( "start_time", info.timestamp, errDetails );
    logInt( "available_memory", info.availableMemory, errDetails );
    logString( "build_description", info.buildDescription, errDetails );
    log( "nondefault_knobs", "[" + nondefaultKnobList + "]", errDetails );
    popScope( errDetails );

    // Start metrics
    pushScope( "metrics", errDetails );

    return OPTIX_SUCCESS;
}

bool Metrics::isEnabled() const
{
    return m_enabled;
}

OptixResult Metrics::logInt( const char* name, uint64_t value, ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> guard( m_mutex );

    return log( name, corelib::to_string( value ), errDetails );
}

OptixResult Metrics::logFloat( const char* name, double value, ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> guard( m_mutex );

    return log( name, corelib::to_string( value ), errDetails );
}

OptixResult Metrics::logString( const char* name, const std::string& value, ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> guard( m_mutex );

    return log( name, "\"" + optix::escapeJsonString( value ) + "\"", errDetails );
}

OptixResult Metrics::pushScope( const char* name, ErrorDetails& errDetails, Metrics::ScopeType type )
{
    if( !m_enabled )
        return errDetails.logDetails( OPTIX_ERROR_UNKNOWN, "Metrics not initialized" );

    std::lock_guard<std::mutex> guard( m_mutex );

    Metrics::ScopeType lwrType       = m_scopeStack.back().type;
    std::string        scopeOpenChar = ( type == Metrics::ARRAY ) ? "[" : "{";
    std::string        field         = ( name ) ? name : "";
    if( name || lwrType == Metrics::OBJECT )
    {
        if( name )
            field = name;
        else
            field = corelib::stringf( "scope<%d>", m_scopeStack.back().numEntries );

        if( lwrType == Metrics::ARRAY )
        {
            // must be in an object to create a named scope
            append( sep() + "{", errDetails );
            m_scopeStack.back().numEntries++;
            m_scopeStack.push_back( {"<NAMED-WRAPPER>", Metrics::OBJECT, 0} );
        }

        append( sep() + "\"" + field + "\":" + scopeOpenChar, errDetails );
    }
    else
    {
        append( sep() + scopeOpenChar, errDetails );
    }

    m_scopeStack.back().numEntries++;
    m_scopeStack.push_back( {field, type, 0} );

    return OPTIX_SUCCESS;
}

OptixResult Metrics::popScope( ErrorDetails& errDetails )
{
    if( !m_enabled )
        return errDetails.logDetails( OPTIX_ERROR_UNKNOWN, "Metrics not initialized" );

    std::lock_guard<std::mutex> guard( m_mutex );

    std::string scopeCloseChar = ( m_scopeStack.back().type == Metrics::ARRAY ) ? "]" : "}";
    m_scopeStack.pop_back();
    append( sep( 0 ) + scopeCloseChar, errDetails );

    if( !m_scopeStack.empty() && m_scopeStack.back().name == "<NAMED-WRAPPER>" )
    {
        // Pop again
        m_scopeStack.pop_back();
        append( sep( 0 ) + "}", errDetails );
    }

    return OPTIX_SUCCESS;
}

OptixResult Metrics::flush( ErrorDetails& errDetails )
{
    if( !m_enabled )
        return errDetails.logDetails( OPTIX_ERROR_UNKNOWN, "Metrics not initialized." );

    std::lock_guard<std::mutex> guard( m_mutex );

    fwrite( m_buffer.data(), 1, m_buffer.size(), m_file );
    fflush( m_file );
    m_buffer.clear();

    return OPTIX_SUCCESS;
}

OptixResult Metrics::log( const char* name, const std::string& valueStr, ErrorDetails& errDetails )
{
    if( !m_enabled )
        return errDetails.logDetails( OPTIX_ERROR_UNKNOWN, "Metrics not initialized." );

    std::string nameValue = corelib::stringf( "\"%s\":%s", name, valueStr.c_str() );
    std::string entry;
    if( m_scopeStack.back().type == Metrics::ARRAY )  // entry needs to be it's own object
        entry = sep() + "{" + nameValue + "}";
    else
        entry = sep() + nameValue;

    registerMetric( name, errDetails );
    append( entry, errDetails );
    m_scopeStack.back().numEntries++;

    return OPTIX_SUCCESS;
}

std::string Metrics::sep( char c )
{
    std::string cStr;
    if( c && m_scopeStack.back().numEntries > 0 )
        cStr      = c;
    size_t indent = m_scopeStack.size();
    return cStr + "\n" + std::string( indent, '\t' );
}

OptixResult Metrics::registerMetric( const char* name, ErrorDetails& errDetails )
{
    m_registeredMetrics.insert( name );

    return OPTIX_SUCCESS;
}

OptixResult Metrics::append( const std::string& str, ErrorDetails& errDetails )
{
    m_buffer.insert( m_buffer.end(), str.begin(), str.end() );

    return OPTIX_SUCCESS;
}

double Metrics::getTimestamp() const
{
    return corelib::getDeltaMilliseconds( m_start );
}

void Metrics::startAsyncTimer( const std::string& name, lwdaStream_t stream )
{
    m_lwdaStopwatch.startAsync( name, stream );
}

void Metrics::stopAsyncTimerAndRecordMetric( const std::string& name, lwdaStream_t stream )
{
    m_lwdaStopwatch.stopAsync( name, stream );
}

}  // end namespace optix_exp