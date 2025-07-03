// Copyright (c) 2018, LWPU CORPORATION.
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
#include <Context/Context.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <ExelwtionStrategy/RTX/RTXLaunchResources.h>
#include <ExelwtionStrategy/RTX/RTXWaitHandle.h>
#include <ExelwtionStrategy/RTX/RTXWaitHandle.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;

RTXWaitHandle::RTXWaitHandle( std::shared_ptr<LaunchResources> launchResources, Context* context )
    : WaitHandle( launchResources )
    , m_context( context )
{
}

RTXWaitHandle::~RTXWaitHandle()
{
    // Block before our memory is destroyed since there may be
    // outstanding memcopies in flight.
    try
    {
        if( m_state == State::InProgress )
            block();
        reset( DeviceSet() );  // clear status_per_device.
    }
    catch( ... )
    { /* swallow exceptions in destructor */
    }
}

void RTXWaitHandle::block()
{
    // This could happen if optix becomes multi-threaded and proper behavior needs to be
    // implemented in that case (subsequent calls need to be blocked until synchronization
    // is complete)
    RT_ASSERT( m_state != State::Blocking );

    // Waiting for a wait handle that is not in progress is a nop.
    if( m_state != State::InProgress )
        return;

    m_state = State::Blocking;

    RTXLaunchResources* lrs = dynamic_cast<RTXLaunchResources*>( m_launchResources.get() );
    RT_ASSERT( lrs );

    for( int allDeviceListIndex : m_launchDevices )
    {
        LWDADevice* device = deviceCast<LWDADevice>( m_context->getDeviceManager()->allDevices()[allDeviceListIndex] );
        device->makeLwrrent();
        int index = m_launchDevices.getArrayPosition( allDeviceListIndex );
        lrs->m_t1[index].synchronize();
    }
    m_state = State::Complete;
}

float RTXWaitHandle::getElapsedMilliseconds() const
{
    RT_ASSERT( m_state == State::Complete );

    RTXLaunchResources* lrs = dynamic_cast<RTXLaunchResources*>( m_launchResources.get() );
    RT_ASSERT( lrs );

    int allDeviceListIndex = *m_launchDevices.begin();
    int index              = m_launchDevices.getArrayPosition( allDeviceListIndex );
    return lwca::Event::elapsedTime( lrs->m_t0[index], lrs->m_t1[index] );
}


void RTXWaitHandle::reset( const DeviceSet& launchDevices )
{
    m_launchDevices = launchDevices;

    RT_ASSERT( m_state == State::Inactive || m_state == State::Complete );
    m_state = State::Inactive;

    // Clear all FrameStatus entries
    if( m_launchResources.get() != nullptr )
        m_launchResources->resetHostFrameStatus();
}

void RTXWaitHandle::syncStream( const lwca::Stream& stream )
{
    RT_ASSERT( m_state == State::InProgress );

    RTXLaunchResources* lrs = dynamic_cast<RTXLaunchResources*>( m_launchResources.get() );
    RT_ASSERT( lrs );

    for( lwca::Event& event : lrs->m_t1 )
        stream.waitEvent( event, 0 );
}

void optix::RTXWaitHandle::releaseResources()
{
    m_state = State::Complete;
    m_launchResources.reset();
}

void RTXWaitHandle::recordEvent( std::vector<optix::lwca::Event>& events, int allDeviceListIndex )
{
    RTXLaunchResources* lrs = dynamic_cast<RTXLaunchResources*>( m_launchResources.get() );
    RT_ASSERT( lrs );

    LWDADevice* device = deviceCast<LWDADevice>( m_context->getDeviceManager()->allDevices()[allDeviceListIndex] );
    device->makeLwrrent();
    int index = m_launchDevices.getArrayPosition( allDeviceListIndex );
    events[index].record( lrs->getStream( allDeviceListIndex ) );
}

void RTXWaitHandle::recordStart()
{
    RT_ASSERT( m_state == State::Inactive );
    m_state = State::Started;

    RTXLaunchResources* lrs = dynamic_cast<RTXLaunchResources*>( m_launchResources.get() );
    RT_ASSERT( lrs );

    recordEvent( lrs->m_t0, *m_launchDevices.begin() );
}

void RTXWaitHandle::recordStop()
{
    RT_ASSERT( m_state == State::Started );
    m_state = State::InProgress;

    RTXLaunchResources* lrs = dynamic_cast<RTXLaunchResources*>( m_launchResources.get() );
    RT_ASSERT( lrs );

    for( int allDeviceListIndex : m_launchDevices )
        recordEvent( lrs->m_t1, allDeviceListIndex );
}
