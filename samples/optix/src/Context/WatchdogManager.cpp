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

#include <Context/WatchdogManager.h>

#include <Context/Context.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>

#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#include <chrono>
#include <cstdint>
#include <ctime>

namespace {
// clang-format off
Knob<float> k_gpuWarmupThreshold( RT_DSTRING( "rtx.gpuWarmupThreshold" ), 2.0, RT_DSTRING( "Time in seconds to wait before relaunching the GPU warm-up kernel" ) );
}  // namespace

namespace optix {

WatchdogManager::WatchdogManager( Context* context )
    : m_context( context )
{
}

WatchdogManager::~WatchdogManager()
{
    for( int idx = 0; idx < OPTIX_MAX_DEVICES; ++idx )
    {
        optix_exp::ErrorDetails errDetails;
        if( OptixResult result = m_gpuWarmup[idx].destroy( errDetails ) )
        {
            lerr << errDetails.m_description << "\n";
        }
    }
}

float WatchdogManager::kick( uint32_t deviceIndex )
{
    // It shouldn't be possible for the index to be out of range, but in the off-chance that
    // it is, just return 0.0 so no warmup is triggered on the out-of-range device.
    return ( deviceIndex < OPTIX_MAX_DEVICES ) ? m_watchdogs[deviceIndex].kick() : 0.0f;
}

OptixResult WatchdogManager::launchWarmup( const DeviceSet& devices, optix_exp::ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_mutex );

    for( int allDeviceListIndex : devices )
    {
        LWDADevice* lwdaDevice = deviceCast<LWDADevice>( m_context->getDeviceManager()->allDevices()[allDeviceListIndex] );
        if( !lwdaDevice )
            continue;
        if( !( lwdaDevice->computeCapability().version() == 75 && lwdaDevice->supportsTTU() ) )
            continue;

        // Kicking the watchdog returns the elapsed time since the last kick and resets the internal timer.
        float kickThreshold = std::max( k_gpuWarmupThreshold.get(), 0.0f );
        float elapsed       = kick( lwdaDevice->allDeviceListIndex() );
        if( elapsed >= kickThreshold )
        {
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            lwdaDevice->makeLwrrent();
            optix_exp::GpuWarmup& gpuWarmup = m_gpuWarmup[allDeviceListIndex];
            if( OptixResult result = gpuWarmup.init( errDetails ) )
                return result;
            if( OptixResult result = gpuWarmup.launch( lwdaDevice->lwdaDevice().MULTIPROCESSOR_COUNT(), errDetails ) )
                return result;
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

            if( !prodlib::log::active( 10 ) )
                continue;

            double elapsed = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000.0;
            llog( 10 ) << "Launched warm-up kernel for device: " << allDeviceListIndex
                       << ", launch time: " << elapsed << " ms\n";
        }
    }
    return OPTIX_SUCCESS;
}

}  // namespace optix
