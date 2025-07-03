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

#include <LWCA/Event.h>
#include <LWCA/Memory.h>
#include <Context/Context.h>
#include <Context/RTCore.h>
#include <Device/DeviceManager.h>
#include <ExelwtionStrategy/RTX/RTXES.h>
#include <ExelwtionStrategy/RTX/RTXFrameTask.h>
#include <ExelwtionStrategy/RTX/RTXLaunchResources.h>
#include <ExelwtionStrategy/RTX/RTXPlan.h>

#include <prodlib/misc/TimeViz.h>
#include <prodlib/system/Logger.h>

#include <memory>
#include <utility>

using namespace optix;

RTXES::RTXES( Context* context )
    : ExelwtionStrategy( context )
{
    m_compiledProgramCache.reset( new CompiledProgramCache() );
}

RTXES::~RTXES()
{
    // Clean up resources
    preSetActiveDevices( m_context->getDeviceManager()->activeDevices() );
}

std::unique_ptr<Plan> RTXES::createPlan( unsigned int entry, int dimensionality, const DeviceSet& devices, int numLaunchDevices ) const
{
    std::unique_ptr<RTXPlan> plan( new RTXPlan( m_context, m_compiledProgramCache.get(), devices, numLaunchDevices ) );
    plan->createPlan( entry, dimensionality );
    return std::move( plan );
}

std::shared_ptr<LaunchResources> RTXES::acquireLaunchResources( const DeviceSet&           devices,
                                                                const FrameTask*           ft,
                                                                const optix::lwca::Stream& syncStream,
                                                                const unsigned int         width,
                                                                const unsigned int         height,
                                                                const unsigned int         depth )
{
    TIMEVIZ_SCOPE( "Acquire Launch Resources" );

    llog( 30 ) << "Acquiring Launch Resources\n";

    // Create shared pointer so memory allocation is contiguous, then cast to the right type for use in the function.
    std::shared_ptr<LaunchResources> resptr( new RTXLaunchResources( this, devices, syncStream ) );
    RTXLaunchResources*              res   = static_cast<RTXLaunchResources*>( resptr.get() );
    const RTXFrameTask*              rtxft = static_cast<const RTXFrameTask*>( ft );

    std::vector<unsigned int> streamIndices;
    streamIndices.resize( devices.count() );

    res->m_t0.resize( devices.count() );
    res->m_t1.resize( devices.count() );

    // Make sure access to launch resources book-keeping is multi-thread safe
    std::unique_lock<std::mutex> guard( m_launchResourcesMutex );

    for( int allDeviceListIndex : devices )
    {
        // Make the device current
        LWDADevice* device = deviceCast<LWDADevice>( m_context->getDeviceManager()->allDevices()[allDeviceListIndex] );
        RT_ASSERT_MSG( device != nullptr, "Non-lwca device encountered while acquiring launch resources" );
        device->makeLwrrent();

        int index = devices.getArrayPosition( allDeviceListIndex );

        streamIndices[index] = m_streamIndexCounters[allDeviceListIndex]++;

        // Reuse available events or create new ones if needed.
        auto& eventQueue = m_eventQueue[allDeviceListIndex];
        if( !eventQueue.empty() )
        {
            res->m_t0[index] = eventQueue.front();
            eventQueue.pop_front();
        }
        else
        {
            res->m_t0[index] = lwca::Event::create();
        }

        if( !eventQueue.empty() )
        {
            res->m_t1[index] = eventQueue.front();
            eventQueue.pop_front();
        }
        else
        {
            res->m_t1[index] = lwca::Event::create();
        }

        const RtcPipeline pipeline = rtxft->getRtcPipeline( allDeviceListIndex );
        RT_ASSERT_MSG( pipeline != nullptr,
                       "Trying to allocate launch resources, but there is no pipeline in the frame task for this "
                       "device" );

        // Reuse or allocate frame status buffer (fixed size so no checks needed)
        auto& frameStatusQueue = m_frameStatusQueue[allDeviceListIndex];
        if( !frameStatusQueue.empty() )
        {
            res->m_deviceFrameStatus[index] = frameStatusQueue.front();
            frameStatusQueue.pop_front();
        }
        else
        {
            // Need to temporarily release the mutex around every lwca::memAlloc and lwca::memFree because they perform
            // implicit device syncs. releaseLaunchResources gets called from a host function enqueued on a LWCA stream,
            // so any lwca::memAlloc and lwca::memFree calls in this function will need to wait for that to finish
            // before continuing. Since releaseLaunchResources needs to acquire that mutex in order to finish,
            // temporarily release it here to avoid deadlock.
            guard.unlock();
            LWdeviceptr temp = lwca::memAlloc( sizeof( cort::FrameStatus ) );
            guard.lock();

            res->m_deviceFrameStatus[index] = temp;
        }

        // Reuse or allocate launch buffer. Find a buffer of sufficient size or create a new one,
        // and destroy any too small buffers encountered.
        size_t lbsize, lbalign;
        m_context->getRTCore()->pipelineGetLaunchBufferInfo( pipeline, (Rtlw64*)&lbsize, (Rtlw64*)&lbalign );
        auto& launchBufferQueue = m_launchBufferQueue[allDeviceListIndex];

        while( !launchBufferQueue.empty() && launchBufferQueue.front().second < lbsize )
        {
            LWdeviceptr temp = launchBufferQueue.front().first;
            launchBufferQueue.pop_front();
            guard.unlock();
            lwca::memFree( temp );
            guard.lock();
        }

        if( !launchBufferQueue.empty() )
        {
            res->m_launchBuffers[index] = launchBufferQueue.front();
            launchBufferQueue.pop_front();
        }
        else
        {
            guard.unlock();
            devptr_size_t temp = std::make_pair( lwca::memAlloc( lbsize ), lbsize );
            guard.lock();

            res->m_launchBuffers[index] = temp;
        }

        // Reuse or allocate scratch buffer. Find a buffer of sufficient size or create a new one,
        // and destroy any too small buffers encountered.
        size_t nbytesMin, sbsize, sbalign;
        m_context->getRTCore()->pipelineGetScratchBufferInfo3D( pipeline, width, height, depth, (Rtlw64*)&nbytesMin,
                                                                (Rtlw64*)&sbsize, (Rtlw64*)&sbalign );
        auto& scratchBufferQueue = m_scratchBufferQueue[allDeviceListIndex];

        if( sbsize > 0 )
        {
            while( !scratchBufferQueue.empty() && scratchBufferQueue.front().second < sbsize )
            {
                LWdeviceptr temp = scratchBufferQueue.front().first;
                scratchBufferQueue.pop_front();
                guard.unlock();
                lwca::memFree( temp );
                guard.lock();
            }

            if( !scratchBufferQueue.empty() )
            {
                res->m_scratchBuffers[index] = scratchBufferQueue.front();
                scratchBufferQueue.pop_front();
            }
            else
            {
                guard.unlock();
                devptr_size_t temp = std::make_pair( lwca::memAlloc( sbsize ), sbsize );
                guard.lock();

                res->m_scratchBuffers[index] = temp;
            }
        }

        llog( 30 ) << "  Device " << allDeviceListIndex << "\n";
        llog( 30 ) << "    FrameStatus: size " << sizeof( cort::FrameStatus ) << ", align " << 16 << "\n";
        llog( 30 ) << "    launchbuf: size " << lbsize << ", align " << lbalign << "\n";
        llog( 30 ) << "    scratchbuf: size " << sbsize << ", align " << sbalign << "\n";
    }

    res->m_streamIndices = std::move( streamIndices );

    llog( 30 ) << "Acquired Launch Resources: " << resptr.get() << "\n";

    return resptr;
}

void RTXES::releaseLaunchResources( LaunchResources* launchResources )
{
    TIMEVIZ_SCOPE( "Release Launch Resources" );

    RTXLaunchResources* res = static_cast<RTXLaunchResources*>( launchResources );

    // Make sure access to launch resources book-keeping is multi-thread safe
    std::lock_guard<std::mutex> guard( m_launchResourcesMutex );

    // Release resources back to the queues
    for( int allDeviceListIndex : res->m_devices )
    {
        int index = res->m_devices.getArrayPosition( allDeviceListIndex );

        auto& sbptr_size = res->m_scratchBuffers[index];
        // Scratch buffer can be empty, so we do not assert here.
        if( sbptr_size.first )
            m_scratchBufferQueue[allDeviceListIndex].push_back( sbptr_size );

        auto& lbptr_size = res->m_launchBuffers[index];
        RT_ASSERT( lbptr_size.first );
        m_launchBufferQueue[allDeviceListIndex].push_back( lbptr_size );

        m_frameStatusQueue[allDeviceListIndex].push_back( res->m_deviceFrameStatus[index] );

        auto& eventQueue = m_eventQueue[allDeviceListIndex];
        eventQueue.push_back( res->m_t0[index] );
        eventQueue.push_back( res->m_t1[index] );
    }
    res->m_deviceFrameStatus.clear();

    llog( 30 ) << "Released Launch Resources: " << res << "\n";
}

void RTXES::removeProgramsForDevices( const DeviceArray& devices )
{
    for( LWDADevice* const device : LWDADeviceArrayView( devices ) )
        m_compiledProgramCache->removeProgramsForDevice( device );
}

void RTXES::preSetActiveDevices( const DeviceArray& removedDevices )
{
    for( LWDADevice* const device : LWDADeviceArrayView( removedDevices ) )
    {
        // Skip if the device is not enabled.
        // This function will be called during ES destruction, with a list
        // of active devices. However, in certain situations a device
        // might be active but not yet enabled. If we try to operate on it
        // it will assert. Since the device is not enabled, there are no
        // resources associated with it, so it is safe to skip it here.
        if( !device->isEnabled() )
            continue;

        const unsigned int allDeviceIndex = device->allDeviceListIndex();

        device->makeLwrrent();

        // Clean up device event queue
        auto& eventQueue = m_eventQueue[allDeviceIndex];
        for( lwca::Event& event : eventQueue )
            event.destroy();
        eventQueue.clear();

        // Clean up buffer queues
        {
            auto& bufferQueue = m_frameStatusQueue[allDeviceIndex];
            for( LWdeviceptr dptr : bufferQueue )
                lwca::memFree( dptr );
            bufferQueue.clear();
        }
        {
            auto& bufferQueue = m_launchBufferQueue[allDeviceIndex];
            for( devptr_size_t& dptr : bufferQueue )
                lwca::memFree( dptr.first );
            bufferQueue.clear();
        }
        {
            auto& bufferQueue = m_scratchBufferQueue[allDeviceIndex];
            for( devptr_size_t& dptr : bufferQueue )
                lwca::memFree( dptr.first );
            bufferQueue.clear();
        }
    }
}
