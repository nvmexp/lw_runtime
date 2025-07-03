// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Util/ApiTime.h>

#include <corelib/system/Timer.h>

#include <cassert>

namespace optix {

// Tracking the effective API time, ie the time spend inside API calls.
class ApiTimeTracker
{
  public:
    // Reset m_timerTick to the current time for the starting API call.
    void enter() { m_timerTick = corelib::getTimerTick(); }
    // Add time of this finished API call to the aclwmulated m_apiTime.
    void exit() { m_apiTime += corelib::getDeltaSeconds( m_timerTick ); }

    // This member allows to retrieve the exact time spend inside API calls up to right now.
    // Note that m_apiTime holds the time spend in finished API calls, whereas the time spend
    // in the current API call would not be considered. This is what this function considers too.
    double getTime() const
    {
        // adding time from start of this API call up to right now
        return m_apiTime + corelib::getDeltaSeconds( m_timerTick );
    }

  private:
    corelib::timerTick m_timerTick;    // cache for each API call's starting time
    double             m_apiTime = 0;  // aclwmulated time spend inside API calls so far
};

// The global instance of the ApiTimeTracker. We are using a pointer here to have either
// lazy initialization or none at all if the API time tracking is disabled.
ApiTimeTracker* g_apiTimeTracker = nullptr;

// Return effective API time.
double getApiTime()
{
    return g_apiTimeTracker ? g_apiTimeTracker->getTime() : 0;
}

// Start API call timing.
void apiTimeEnter()
{
    static ApiTimeTracker g_timeTracker;
    if( !g_apiTimeTracker )
        g_apiTimeTracker = &g_timeTracker;
    g_apiTimeTracker->enter();
}

// Stop API call timing.
void apiTimeExit()
{
    // while this should not happen we guard against it anyway...
    if( !g_apiTimeTracker )
        return;

    g_apiTimeTracker->exit();
}
}
