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

#include <Memory/DemandLoad/DeviceEventPool.h>

#include <Device/LWDADevice.h>
#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Util/ContainerAlgorithm.h>

#include <prodlib/exceptions/Assert.h>

#include <algorithm>

using namespace optix::demandLoad;

namespace optix {

void DeviceEventPool::init( LWDADevice* device, unsigned int eventCount )
{
    destroy();
    m_device = device;
    m_eventsUsed.resize( eventCount );
    m_events.resize( eventCount );
    m_device->makeLwrrent();
    for( lwca::Event& e : m_events )
    {
        e = lwca::Event::create( LW_EVENT_DISABLE_TIMING | LW_EVENT_BLOCKING_SYNC );
    }
}

void DeviceEventPool::destroy()
{
    for( lwca::Event& e : m_events )
    {
        e.destroy();
    }
    m_eventsUsed.clear();
    m_events.clear();
}

bool DeviceEventPool::isEventAvailable() const
{
    return algorithm::find( m_eventsUsed, false ) != m_eventsUsed.end();
}

unsigned int DeviceEventPool::recordEvent( lwca::Stream stream )
{
    // Record the event.  For robustness we continue even if recording the event failed.
    if( !isEventAvailable() )
    {
        LOG_MEDIUM_VERBOSE( "DeviceEventPool::recordEvent events exhausted\n" );
        m_device->makeLwrrent();
        m_events.push_back( lwca::Event::create( LW_EVENT_DISABLE_TIMING | LW_EVENT_BLOCKING_SYNC ) );
        m_eventsUsed.push_back( false );
    }
    RT_ASSERT_MSG( isEventAvailable(), "Events not available after creating one" );

    const std::vector<bool>::iterator firstAvailableEvent = algorithm::find( m_eventsUsed, false );
    const unsigned int eventIndex = static_cast<unsigned int>( std::distance( m_eventsUsed.begin(), firstAvailableEvent ) );
    m_eventsUsed[eventIndex]      = true;
    m_events[eventIndex].record( stream );
    return eventIndex;
}

void DeviceEventPool::waitForEvent( unsigned int id )
{
    m_events[id].synchronize();
    // We can now reuse the event
    m_eventsUsed[id] = false;
}

}  // namespace optix
