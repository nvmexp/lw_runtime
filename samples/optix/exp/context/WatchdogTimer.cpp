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

#include <exp/context/WatchdogTimer.h>

#include <cfloat>

namespace optix_exp {

WatchdogTimer::WatchdogTimer()
    : m_lastKick( std::chrono::steady_clock::now() )
{
}

float WatchdogTimer::kick()
{
    using namespace std::chrono;

    steady_clock::time_point lwrTime = steady_clock::now();
    duration<float>          elapsed = duration_cast<duration<float>>( lwrTime - m_lastKick );

    m_lastKick = lwrTime;

    if( m_initialized )
    {
        return elapsed.count();
    }
    else
    {
        m_initialized = true;

        // If this is the first kick, then treat the elapsed time as infinite.
        // But rather than returning INF, return the largest floating-point value.
        return FLT_MAX;
    }
}

}  // namespace optix_exp
