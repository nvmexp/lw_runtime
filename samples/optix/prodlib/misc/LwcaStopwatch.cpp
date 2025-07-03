//
//  Copyright (c) 2021 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include <prodlib/misc/LwdaStopwatch.h>

namespace prodlib {

static void LWDART_CB startCallback( void* ptr )
{
    LwdaStopwatchPayload* payload = reinterpret_cast<LwdaStopwatchPayload*>( ptr );
    payload->recordStartTime();
    delete payload;
}

static void LWDART_CB stopCallback( void* ptr )
{
    LwdaStopwatchPayload* payload = reinterpret_cast<LwdaStopwatchPayload*>( ptr );
    payload->recordStopTime();
    delete payload;
}

void LwdaStopwatchPayload::recordStartTime()
{
    m_stopwatch->insertStartTime( m_name, m_stream );
}

void LwdaStopwatchPayload::recordStopTime()
{
    TimePoint    startTime;
    Milliseconds elapsedMs;
    if( m_stopwatch->getStartTime( m_name, m_stream, startTime ) )
    {
        TimePoint stopTime = std::chrono::steady_clock::now();
        elapsedMs          = stopTime - startTime;
        m_stopwatch->deleteStartTime( m_name, m_stream );
    }
    m_stopwatch->recordElapsedMs( m_name, m_stream, elapsedMs.count() );
}

void LwdaStopwatch::startAsync( const std::string& name, lwdaStream_t stream )
{
    LwdaStopwatchPayload* payload = new LwdaStopwatchPayload( this, name, stream );
    lwdaLaunchHostFunc( stream, startCallback, reinterpret_cast<void*>( payload ) );
}

void LwdaStopwatch::stopAsync( const std::string& name, lwdaStream_t stream )
{
    LwdaStopwatchPayload* payload = new LwdaStopwatchPayload( this, name, stream );
    lwdaLaunchHostFunc( stream, stopCallback, reinterpret_cast<void*>( payload ) );
}

void LwdaStopwatch::insertStartTime( std::string name, lwdaStream_t stream )
{
    std::lock_guard<std::mutex> lock( m_startTimesMutex );
    m_startTimes[stream][name] = std::chrono::steady_clock::now();
}

bool LwdaStopwatch::getStartTime( std::string name, lwdaStream_t stream, TimePoint& startTime )
{
    std::lock_guard<std::mutex> lock( m_startTimesMutex );
    auto                        streamLookup = m_startTimes.find( stream );
    bool                        found        = streamLookup != m_startTimes.end();
    if( found )
    {
        auto nameLookup = streamLookup->second.find( name );
        found           = nameLookup != streamLookup->second.end();
        if( found )
            startTime = nameLookup->second;
    }
    return found;
}

void LwdaStopwatch::deleteStartTime( std::string name, lwdaStream_t stream )
{
    std::lock_guard<std::mutex> lock( m_startTimesMutex );
    auto                        streamLookup = m_startTimes.find( stream );
    if( streamLookup != m_startTimes.end() )
    {
        streamLookup->second.erase( name );
    }
}

void LwdaStopwatch::recordElapsedMs( std::string name, lwdaStream_t stream, double time )
{
    std::lock_guard<std::mutex> lock( m_elapsedTimesMutex );
    m_elapsedTimes[stream][name].push_back( time );
}

void LwdaStopwatch::getElapsedTimes( ElapsedAsyncTimesMap& elapsedTimes )
{
    std::lock_guard<std::mutex> lock( m_elapsedTimesMutex );
    elapsedTimes = m_elapsedTimes;
}

}  // namespace prodlib