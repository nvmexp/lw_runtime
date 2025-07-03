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

#pragma once

#include <corelib/misc/String.h>
#include <prodlib/system/Knobs.h>

namespace {
// clang-format off
// Define this knob in the header so we can inline the query. We are reusing the "metrics.enable" knob here,
// since lwrrently the only usage of measuring the API time is reporting the startup time (see Context.cpp).
// As soon as the effective API time should be used independent from the metrics a new knob should be used.
PublicKnob<bool> k_metricsEnable(RT_PUBLIC_DSTRING("metrics.enable"), false, RT_PUBLIC_DSTRING("Enable metrics output to metrics.json") );
// clang-format on
}

#if defined( DEBUG ) || defined( DEVELOP )
// Use this at the top of an API call to measure its API time. Please note that
// this support single-threaded OptiX API calls only.
#define API_TIME_SCOPE optix::ApiTimeScope apiTimeScope
#else
// The overhead of this is already very low when ApiTimeScope isn't enabled, but
// let's make it zero for public release builds.
#define API_TIME_SCOPE
#endif  // DEBUG || DEVELOP

namespace optix {

// Return the effective API time, which is the sum of the time spent in the API calls so far.
// In seconds.
double getApiTime();

//
// --- Don't use directly anything below this point ---
//

// Start the measurement of the time spend inside an API call.
void apiTimeEnter();
// Stop the measurement of the time spend inside an API call. Internally each API call's time will be aclwmulated
// such that it can be retrieved via optix::getApiTime().
void apiTimeExit();

// RAII helper to measure the time spend inside an API call.
class ApiTimeScope
{
  public:
    inline ApiTimeScope()
    {
        if( k_metricsEnable.get() )
            apiTimeEnter();
    }
    inline ~ApiTimeScope()
    {
        if( k_metricsEnable.get() )
            apiTimeExit();
    }
};
}
