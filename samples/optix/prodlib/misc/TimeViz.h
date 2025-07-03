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

#pragma once

//
// TimeViz: dump profiling data in JSON format to be parsed and visualized by
// Chrome's built-in chrome://tracing facility.
//
// The trace format description can be found here:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#heading=h.nso4gcezn7n1
//

#include <corelib/misc/String.h>
#include <corelib/system/Preprocessor.h>  // for function name macros
#include <prodlib/system/Knobs.h>

#include <enable_lwtx.h>

namespace {
// clang-format off
// Define this knob in the header so we can inline the query
PublicKnob<bool> k_timevizEnabled( RT_PUBLIC_DSTRING("timeviz.enable"), false, RT_PUBLIC_DSTRING("Enables or disables dumping of profiling events. Type \"chrome://tracing\" in Chrome and load the dump file to view the data.") );
// clang-format on
}


#if defined( DEBUG ) || defined( DEVELOP )

// Use this at the top of a scope to add it to the timeline
#define TIMEVIZ_CONCAT_( x, y ) x##y
#define TIMEVIZ_CONCAT( x, y ) TIMEVIZ_CONCAT_( x, y )
#define TIMEVIZ_SCOPE( name ) prodlib::TimeViz_Scope TIMEVIZ_CONCAT( tvscope_, __LINE__ )( name )

// Shortlwt for TIMEVIZ_SCOPE that automatically uses the current function name as input
#define TIMEVIZ_FUNC TIMEVIZ_SCOPE( RTAPI_FUNC_NAME )

// Use this to record counters.
#define TIMEVIZ_COUNT( name, count ) prodlib::TimeViz_Count( name, count )

// Call this periodically to give the system a chance to write out the buffered data
#define TIMEVIZ_FLUSH prodlib::TimeViz_Flush()

#else

// The overhead of these is already very low when timeviz isn't enabled, but
// let's make it zero for public release builds.
#define TIMEVIZ_SCOPE( name )
#define TIMEVIZ_FUNC
#define TIMEVIZ_COUNT( name, count )
#define TIMEVIZ_FLUSH

#endif  // DEBUG || DEVELOP


//
// --- Don't use directly anything below this point ---
//

namespace prodlib {

class TimeViz_Scope
{
  public:
    TimeViz_Scope( const char* name )
    {
#ifndef ENABLE_LWTX
        if( k_timevizEnabled.get() )
#endif
            doCtor( name );
    }
    ~TimeViz_Scope()
    {
#ifndef ENABLE_LWTX
        if( m_name )
#endif
            doDtor();
    }

    TimeViz_Scope( const TimeViz_Scope& ) = delete;
    TimeViz_Scope& operator=( const TimeViz_Scope& ) = delete;

  private:
    void doCtor( const char* name );
    void doDtor();

    char* m_name = nullptr;
};

void TimeViz_doCount( const char* name, long long count );
void TimeViz_doFlush();

inline void TimeViz_Count( const char* name, long long count )
{
#ifndef ENABLE_LWTX
    if( k_timevizEnabled.get() )
#endif
        TimeViz_doCount( name, count );
}
inline void TimeViz_Flush()
{
    TIMEVIZ_FUNC;  // time ourselves, so perf hiclwps due to flushing are obvious in the trace
#ifndef ENABLE_LWTX
    if( k_timevizEnabled.get() )
#endif
        TimeViz_doFlush();
}
}
