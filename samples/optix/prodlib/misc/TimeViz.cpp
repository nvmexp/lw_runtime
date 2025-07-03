// Copyright LWPU Corporation 2015
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES


#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <prodlib/misc/TimeViz.h>

#include <corelib/system/Timer.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#ifdef ENABLE_LWTX
#include <lwtx3/lwToolsExt.h>
#endif

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>

using namespace prodlib;
using namespace corelib;


namespace {
// clang-format off
PublicKnob<std::string> k_file( RT_PUBLIC_DSTRING("timeviz.file"), "timeviz.json", RT_PUBLIC_DSTRING("Specifies the output file for timeviz profiling.") );
PublicKnob<size_t>      k_flushThreshold( RT_PUBLIC_DSTRING("timeviz.flushThreshold"), 100000, RT_PUBLIC_DSTRING("The number of timeviz entries that need to be buffered up before we flush to file.") );
// clang-format on
}

namespace prodlib {

// Helpers
static unsigned long long getThreadId();

#ifdef ENABLE_LWTX
class LwTxLib
{
  public:
    LwTxLib()
    {
        lwtxInitialize( 0 );

        // Perform a push and pop to make sure we can use LWTX.
        const int pushRes = ::lwtxRangePushA( "Initialize LWTX" );
        const int popRes  = ::lwtxRangePop();
        if( pushRes < 0 || popRes < 0 )
        {
            // Print out an error and exit here, since we'll probably swallow the exception if we throw here.
            std::cerr << "LWTX initialization failure. Must run inside of profiler if LWTX is enabled.\n";
            exit( -1 );
        }
    }

    static void lwtxRangePushA( const char* message )
    {
        const int res = ::lwtxRangePushA( message );
        if( res < 0 )
        {
            std::cerr << "LWTX RangePushA failure. Result=" << res << "\n";
            exit( -1 );
        }
    }

    static void lwtxRangePop( void )
    {
        const int res = ::lwtxRangePop();
        if( res < 0 )
        {
            std::cerr << "LWTX RangePopA failure. Result=" << res << "\n";
            exit( -1 );
        }
    }
};
#endif


struct TimeVizEvent
{
    long long          count     = -1;
    unsigned long long tid       = 0;
    double             timestamp = 0;
    std::string        name;
    char               phase = '-';
};

class TimeVizState
{
  public:
    TimeVizState() = default;

    ~TimeVizState()
    {
#ifndef ENABLE_LWTX
        if( k_timevizEnabled.get() )
        {
            flushToFile( true );

            if( fp )
            {
                fprintf( fp, "\n]\n" );
                fclose( fp );
            }
        }
#endif
    }

    void push( const TimeVizEvent& e )
    {
        std::lock_guard<std::mutex> guard( mutex );
        reserveEvents();
        events.push_back( e );
    }

    void pushCount( const char* name, double timestamp, unsigned long long tid, long long count )
    {
        std::lock_guard<std::mutex> guard( mutex );
        reserveEvents();

        TimeVizEvent e;
        e.name      = name;
        e.count     = count;
        e.phase     = 'C';
        e.tid       = tid;
        e.timestamp = timestamp;

        events.push_back( e );
    }

    void flush()
    {
        std::lock_guard<std::mutex> guard( mutex );
        flushToFile( false );
    }

    corelib::timerTick getStart() const { return start; }

#ifdef ENABLE_LWTX
    LwTxLib& getLwTx() { return lwTxFuncs; }
#endif

  private:
    void reserveEvents()
    {
        // Will only do anything if current capacity is smaller than the argument.
        // Add a little more than the threshold because we usually run over a little
        // before the next flush operation realizes we're full.
        events.reserve( static_cast<size_t>( static_cast<double>( k_flushThreshold.get() ) * 1.1 ) );
    }

    void flushToFile( bool force )
    {
        const bool needsFlush = force || events.size() >= k_flushThreshold.get();
        if( !needsFlush )
            return;

        if( !fp )
        {
            fp = fopen( k_file.get().c_str(), "w" );
            fprintf( fp, "[ {}" );
        }

        for( const auto& e : events )
        {
            writeJson( e.name.c_str(), "-", e.phase, e.timestamp, e.tid, e.count );
        }
        events.clear();
    }

    void writeJson( const char* name, const char* cat, char phase, double timestamp, unsigned long long tid, long long count )
    {
        // Trim potential leading "optix::", to avoid clutter in the output
        if( strncmp( name, "optix::", 7 ) == 0 )
            name += 7;

        if( count == -1 )
        {  // duration event (begin or end)
            fprintf( fp, "\n,{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"%c\",\"pid\":0,\"tid\":%llu,\"ts\":%.1f}", name,
                     cat, phase, tid, timestamp * 1000000.0 );
        }
        else
        {  // counter event
            fprintf( fp,
                     "\n,{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"%c\",\"pid\":0,\"tid\":%llu,\"ts\":%.1f,\"args\":{"
                     "\"v\":%"
                     "lld}}",
                     name, cat, phase, tid, timestamp * 1000000.0, count );
        }
    }

    FILE*                     fp    = nullptr;
    const timerTick           start = corelib::getTimerTick();
    std::vector<TimeVizEvent> events;
    std::mutex                mutex;

#ifdef ENABLE_LWTX
    LwTxLib lwTxFuncs;
#endif

} g_timeVizState;


void TimeViz_Scope::doCtor( const char* name )
{
#ifdef ENABLE_LWTX
    g_timeVizState.getLwTx().lwtxRangePushA( name );
#else
    // Bother with constructing a string only if tracing is enabled (i.e. if we get here)
    const size_t len = strlen( name );
    m_name           = new char[len + 1];
    strcpy( m_name, name );

    TimeVizEvent e;
    e.name              = name;
    e.phase             = 'B';
    e.tid               = getThreadId();
    const timerTick lwr = corelib::getTimerTick();  // get time as late as possible since we're timing what's follwing
    e.timestamp         = getDeltaSeconds( g_timeVizState.getStart(), lwr );
    g_timeVizState.push( e );
#endif
}

void TimeViz_Scope::doDtor()
{
#ifdef ENABLE_LWTX
    g_timeVizState.getLwTx().lwtxRangePop();
#else
    const timerTick lwr = corelib::getTimerTick();

    TimeVizEvent e;
    e.name      = m_name;
    e.phase     = 'E';
    e.tid       = getThreadId();
    e.timestamp = corelib::getDeltaSeconds( g_timeVizState.getStart(), lwr );
    g_timeVizState.push( e );

    // If tracing is enabled, then m_name is a valid pointer we need to delete
    delete[] m_name;
#endif
}

void TimeViz_doCount( const char* name, long long count )
{
#ifndef ENABLE_LWTX
    const timerTick lwr       = corelib::getTimerTick();
    const double    timestamp = getDeltaSeconds( g_timeVizState.getStart(), lwr );
    g_timeVizState.pushCount( name, timestamp, getThreadId(), count );
#endif
}

void TimeViz_doFlush()
{
#ifndef ENABLE_LWTX
    g_timeVizState.flush();
#endif
}

static unsigned long long getThreadId()
{
#ifdef _WIN32
    return (unsigned long long)GetLwrrentThreadId();
#else
    return (unsigned long long)pthread_self();
#endif
}

}  // namespace
