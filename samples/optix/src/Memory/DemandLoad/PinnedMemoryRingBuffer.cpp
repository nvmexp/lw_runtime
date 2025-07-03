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

#include <Memory/DemandLoad/PinnedMemoryRingBuffer.h>

#include <LWCA/Memory.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Exceptions/ExceptionHelpers.h>
#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Memory/DemandLoad/StagingPageAllocator.h>
#include <Util/ContainerAlgorithm.h>

#include <prodlib/exceptions/Assert.h>

#include <o6/optix.h>

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>

namespace optix {

PinnedMemoryRingBuffer::~PinnedMemoryRingBuffer()
{
    destroy();
}

void PinnedMemoryRingBuffer::init( DeviceManager* dm, size_t numBytes, unsigned int numEvents, unsigned int numRequests )
{
    LOG_MEDIUM_VERBOSE( "PinnedMemoryRingBuffer::init bufferSize " << numBytes << ", eventCount " << numEvents << '\n' );

    makeLWDADeviceLwrrent( dm );
    m_buffer     = static_cast<unsigned char*>( lwca::memHostAlloc( numBytes, LW_MEMHOSTREGISTER_PORTABLE ) );
    m_bufferSize = numBytes;
    m_numEvents  = numEvents;
    m_requests.resize( numRequests );
}

void PinnedMemoryRingBuffer::destroy()
{
    LOG_MEDIUM_VERBOSE( "PinnedMemoryRingBuffer::destroy\n" );

    if( m_buffer )
    {
        lwca::memFreeHost( m_buffer );
        m_buffer = nullptr;
    }
    for( DeviceEventPool& deviceEvents : m_deviceEvents )
    {
        deviceEvents.destroy();
    }
}

// Acquire an allocation of the specified size, returning the data via a result parameter. The
// caller must call the finalize() method with the data pointer when all host-side operations
// on the data are done.
StagingPageAllocation PinnedMemoryRingBuffer::acquireResource( size_t size )
{
    LOG_MEDIUM_VERBOSE( "PinnedMemoryRingBuffer::acquireResource size " << size << '\n' );

    std::unique_lock<std::mutex> lock( m_mutex );

    // If necessary, retire old allocations to make space.
    RT_ASSERT( size <= m_bufferSize );
    while( !isSpaceAvailable( size ) || !isRequestAvailable() )
    {
        // Wrap the tail around the end of the buffer if necessary.
        if( m_tail > m_head && m_head > 0 )
        {
            m_tail = 0;
            continue;
        }

        // Wait until all the head's events are recorded.
        m_resourceReleased.wait( lock, [this]() { return m_requests[m_requestHead].allEventsRecorded; } );

        if( !isSpaceAvailable( size ) || !isRequestAvailable() )
        {
            // Retire the oldest allocation.
            retireHead();
        }
    }

    // Allocate storage by advancing the tail.
    StagingPageAllocation allocation{};
    allocation.address = &m_buffer[m_tail];
    m_tail += size;
    RT_ASSERT( m_tail <= m_bufferSize );

    // Construct and enqueue the request.
    allocation.id                               = m_requestTail;
    allocation.size                             = size;
    m_requests[m_requestTail].begin             = static_cast<unsigned char*>( allocation.address ) - m_buffer;
    m_requests[m_requestTail].allEventsRecorded = false;
    algorithm::fill( m_requests[m_requestTail].eventsRecorded, false );
    m_requestTail = ( m_requestTail + 1 ) % m_requests.size();

    return allocation;
}

void PinnedMemoryRingBuffer::recordEvent( lwca::Stream stream, unsigned int allDeviceListIndex, const StagingPageAllocation& allocation )
{
    LOG_MEDIUM_VERBOSE( "PinnedMemoryRingBuffer::recordEvent device " << allDeviceListIndex << ", allocation "
                                                                      << allocation.id << '\n' );

    std::unique_lock<std::mutex> lock( m_mutex );
    RT_ASSERT_MSG( allocation.id < m_requests.size(), "Allocation id out of range" );
    RT_ASSERT_MSG( m_activeDevices.isSet( allDeviceListIndex ), "Device index not in the active device set" );

    Request& request                           = m_requests[allocation.id];
    request.eventIds[allDeviceListIndex]       = m_deviceEvents[allDeviceListIndex].recordEvent( stream );
    request.eventsRecorded[allDeviceListIndex] = true;
    LOG_MEDIUM_VERBOSE( "PinnedMemoryRingBuffer::recordEvent device "
                        << allDeviceListIndex << ", allocation " << allocation.id << ", event "
                        << request.eventIds[allDeviceListIndex] << " recorded\n" );
}

// Called when all host-side operations on an allocation are completed.  Records an event on the
// given stream that determines when the allocation can be reclaimed.
void PinnedMemoryRingBuffer::releaseResource( const StagingPageAllocation& allocation )
{
    LOG_MEDIUM_VERBOSE( "PinnedMemoryRingBuffer::releaseResource allocation " << allocation.id << '\n' );

    std::unique_lock<std::mutex> lock( m_mutex );

    RT_ASSERT_MSG( allocation.id < m_requests.size(), "Invalid allocation id" );

    Request& request = m_requests[allocation.id];

    // Notify any threads waiting in retireHead() that an event has been recorded.
    request.allEventsRecorded = true;

    m_resourceReleased.notify_all();
}

void PinnedMemoryRingBuffer::clear()
{
    LOG_MEDIUM_VERBOSE( "PinnedMemoryRingBuffer::clear\n" );

    std::unique_lock<std::mutex> lock( m_mutex );

    while( m_requestHead != m_requestTail )
    {
        retireHead();
    }
}

void PinnedMemoryRingBuffer::removeActiveDevices( const DeviceArray& allDevices, const DeviceSet& removedDevices )
{
    LOG_MEDIUM_VERBOSE( "PinnedMemoryRingBuffer::removeActiveDevices " << removedDevices.toString() << '\n' );

    m_activeDevices -= removedDevices;
    for( unsigned int allDeviceListIndex : removedDevices )
    {
        deviceCast<LWDADevice>( allDevices[allDeviceListIndex] )->makeLwrrent();
        m_deviceEvents[allDeviceListIndex].destroy();
    }
}

void PinnedMemoryRingBuffer::setActiveDevices( const DeviceArray& allDevices, const DeviceSet& devices )
{
    LOG_MEDIUM_VERBOSE( "PinnedMemoryRingBuffer::setActiveDevices " << devices.toString() << '\n' );

    m_activeDevices = devices;
    for( unsigned int allDeviceListIndex : devices )
    {
        m_deviceEvents[allDeviceListIndex].init( deviceCast<LWDADevice>( allDevices[allDeviceListIndex] ), m_numEvents );
    }
}

// Check whether an allocation of the specified size is possible.
bool PinnedMemoryRingBuffer::isSpaceAvailable( size_t size ) const
{
    if( m_tail >= m_head )
    {
        // There must be storage between the tail and the end of the buffer.
        return m_bufferSize - m_tail >= size;
    }

    // There must be storage between the tail and the head.
    return m_head - m_tail >= size;
}

bool PinnedMemoryRingBuffer::isRequestAvailable() const
{
    if( m_requestTail >= m_requestHead )
    {
        return m_requests.size() - m_requestTail >= 1;
    }

    return m_requestHead - m_requestTail >= 1;
}

// Wait until the head request has its associated event recorded, and then synchronize on the event.
void PinnedMemoryRingBuffer::retireHead()
{
    LOG_MEDIUM_VERBOSE( "PinnedMemoryRingBuffer::retireHead\n" );

    // Wait for the LWCA event.  Note that we continue to hold the mutex while waiting.
    // No other threads will be able to proceed anyway, since the ring buffer is full.
    Request& request = m_requests[m_requestHead];

    // Wait for all pending events on specific devices
    for( unsigned int allDeviceListIndex : m_activeDevices )
    {
        if( request.eventsRecorded[allDeviceListIndex] )
        {
            LOG_MEDIUM_VERBOSE( "PinnedMemoryRingBuffer::retireHead device "
                                << allDeviceListIndex << " event " << request.eventIds[allDeviceListIndex] << '\n' );
            m_deviceEvents[allDeviceListIndex].waitForEvent( request.eventIds[allDeviceListIndex] );
            request.eventsRecorded[allDeviceListIndex] = false;
        }
    }

    // Pop the request and update the head.
    m_requestHead = ( m_requestHead + 1 ) % m_requests.size();
    if( m_requestHead != m_requestTail )
    {
        m_head = m_requests[m_requestHead].begin;
    }
    else
    {
        m_head = 0;
        m_tail = 0;
    }
}

void PinnedMemoryRingBuffer::makeLWDADeviceLwrrent( DeviceManager* dm )
{
    for( LWDADevice* device : LWDADeviceArrayView( dm->activeDevices() ) )
    {
        device->makeLwrrent();
        break;
    }
}

}  // namespace optix
