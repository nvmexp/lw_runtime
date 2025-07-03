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

#include <optix_types.h>

#include <Device/MaxDevices.h>

#include <exp/context/ErrorHandling.h>
#include <exp/context/GpuWarmup.h>
#include <exp/context/WatchdogTimer.h>

#include <mutex>

namespace optix_exp {
class GpuWarmup;
}

namespace optix {

class Context;
class DeviceSet;

class WatchdogManager
{
  public:
    WatchdogManager( Context* context );
    ~WatchdogManager();

    float kick( uint32_t deviceIndex );
    OptixResult launchWarmup( const DeviceSet& devices, optix_exp::ErrorDetails& errDetails );

  private:
    Context*                 m_context = nullptr;
    optix_exp::GpuWarmup     m_gpuWarmup[OPTIX_MAX_DEVICES];
    optix_exp::WatchdogTimer m_watchdogs[OPTIX_MAX_DEVICES];
    std::mutex               m_mutex;
};

}  // namespace optix
