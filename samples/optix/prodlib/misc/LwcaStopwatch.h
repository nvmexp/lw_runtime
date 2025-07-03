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

#pragma once

#include <lwda_runtime.h>

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace prodlib {

using Milliseconds         = std::chrono::duration<double, std::milli>;
using TimePoint            = std::chrono::time_point<std::chrono::steady_clock>;
using ElapsedAsyncTimesMap = std::unordered_map<lwdaStream_t, std::unordered_map<std::string, std::vector<double>>>;

class LwdaStopwatch
{
  public:
    LwdaStopwatch() {}
    void startAsync( const std::string& name, lwdaStream_t stream = 0 );
    void stopAsync( const std::string& name, lwdaStream_t stream = 0 );
    void getElapsedTimes( ElapsedAsyncTimesMap& elapsedTimes );

  private:
    void insertStartTime( std::string name, lwdaStream_t stream );
    bool getStartTime( std::string name, lwdaStream_t stream, TimePoint& startTime );
    void deleteStartTime( std::string name, lwdaStream_t stream );
    void recordElapsedMs( std::string name, lwdaStream_t stream, double time );

    std::unordered_map<lwdaStream_t, std::unordered_map<std::string, TimePoint>> m_startTimes;
    ElapsedAsyncTimesMap m_elapsedTimes;
    std::mutex           m_startTimesMutex;
    std::mutex           m_elapsedTimesMutex;

    friend class LwdaStopwatchPayload;
};

class LwdaStopwatchPayload
{
  public:
    LwdaStopwatchPayload( LwdaStopwatch* stopwatch, std::string name, lwdaStream_t stream )
        : m_stopwatch{stopwatch}
        , m_name{name}
        , m_stream{stream}
    {
    }
    void recordStartTime();
    void recordStopTime();

  private:
    LwdaStopwatch* m_stopwatch;
    std::string    m_name;
    lwdaStream_t   m_stream;
};

typedef std::shared_ptr<LwdaStopwatch> LwdaStopwatchHandle;

}  // namespace prodlib