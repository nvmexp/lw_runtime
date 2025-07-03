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

#include <LWCA/Event.h>
#include <LWCA/Stream.h>
#include <Device/Device.h>
#include <Device/DeviceSet.h>
#include <Memory/DemandLoad/DeviceEventPool.h>

#include <o6/optix.h>

#include <condition_variable>
#include <mutex>
#include <vector>

#include "StagingPageAllocator.h"

namespace optix {

class DeviceManager;

class PinnedMemoryRingBuffer
{
  public:
    PinnedMemoryRingBuffer() = default;
    ~PinnedMemoryRingBuffer();
    PinnedMemoryRingBuffer( const PinnedMemoryRingBuffer& other ) = delete;
    PinnedMemoryRingBuffer( PinnedMemoryRingBuffer&& other )      = delete;
    PinnedMemoryRingBuffer& operator=( const PinnedMemoryRingBuffer& other ) = delete;
    PinnedMemoryRingBuffer& operator=( PinnedMemoryRingBuffer&& other ) = delete;

    // ringSize: size in bytes of device memory used for the launch resources
    //
    // eventCount: number of outstanding requests.  Host will block if we exceed the number
    // of outstanding requests.
    void init( DeviceManager* dm, size_t numBytes, unsigned int numEvents, unsigned int numRequests );

    // Blocks for any outstanding requests.
    //
    // Deletes any GPU resources
    void destroy();

    // A demand load callback acquires a chunk of pinned memory, initiates asynchronous copies to
    // devices using the pinned memory as the source of the copy, then records an event on the
    // stream associated with the device that is the target of the asynchronous copy.  Releasing
    // the allocated resource tells the ring buffer that it is free to wait on the recorded events
    // when the request is retired.
    StagingPageAllocation acquireResource( size_t size );
    void recordEvent( lwca::Stream stream, unsigned int allDeviceListIndex, const StagingPageAllocation& allocation );
    void releaseResource( const StagingPageAllocation& allocation );

    // Waits for all pending copies to complete.
    void clear();

    // Manage the events created for active devices.
    void removeActiveDevices( const DeviceArray& allDevices, const DeviceSet& removedDevices );
    void setActiveDevices( const DeviceArray& allDevices, const DeviceSet& devices );

  private:
    bool isSpaceAvailable( size_t size ) const;
    bool isRequestAvailable() const;
    void retireHead();
    void makeLWDADeviceLwrrent( DeviceManager* dm );

    unsigned char* m_buffer     = nullptr;
    size_t         m_bufferSize = 0U;
    size_t         m_head       = 0;  // first byte of oldest allocation
    size_t         m_tail       = 0;  // first byte of next allocation

    struct Request
    {
        size_t       begin = 0;
        bool         eventsRecorded[16]{};
        unsigned int eventIds[16]{};
        bool         allEventsRecorded = false;
    };
    std::vector<Request> m_requests;
    unsigned int         m_requestHead = 0U;
    unsigned int         m_requestTail = 0U;

    unsigned int    m_numEvents = 0U;
    DeviceSet       m_activeDevices;
    DeviceEventPool m_deviceEvents[16]{};

    std::mutex              m_mutex;
    std::condition_variable m_resourceReleased;

    // Only one request can be outstanding at a time
    std::mutex m_requestLock;

    std::condition_variable m_requestReleased;
};

}  // namespace optix
