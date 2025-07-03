/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
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

#include <lwca.h>

#include <mutex>
#include <vector>

namespace optix_exp {

class ErrorDetails;
class LaunchResourceManager
{
  public:
    // ringSize: size in bytes of device memory used for the launch resources
    //
    // eventCount: number of oustanding requests.  Host will block if we exceed the number
    // of outstanding requests.
    OptixResult init( size_t ringSize, unsigned int eventCount, ErrorDetails& errDetails );

    // Blocks for any outstanding requests.
    //
    // Deletes any GPU resources
    OptixResult destroy( ErrorDetails& errDetails );

    // acquire and release represent a critical section.  Only one thread at a time can
    // acquire a resource.  If there was an error acquiring the resource, the resource
    // should not be released.  If the resource acquisition was successfull, the resource
    // must be released to avoid application deadlock regardless of intervening errors
    // from the thread holding the resource.
    OptixResult acquireResource( LWstream stream, size_t size, LWdeviceptr& resourcePtr, ErrorDetails& errDetails );
    OptixResult releaseResource( LWstream stream, size_t size, ErrorDetails& errDetails );

    // If any event has not oclwrred on the device by the time we issue a wait on it,
    // increment the corresponding counter.  Doing this check can cost 12-20 us, so the
    // frequency of checking is controlled by a knob.
    unsigned int m_ringPressureCount  = 0;
    unsigned int m_eventPressureCount = 0;

  private:
    // Next position in the ring buffer to put the allocations
    size_t m_ringHead = 0;
    // Next slot to put the Request.  Does not point to a valid Request.
    unsigned int m_eventHead = 0;
    // Next request to retire.  Points to a valid Request except when equal to m_eventHead
    // (which is never valid).
    unsigned int m_eventTail = 0;

    // Device allocation of the ring buffer.  Never resized.
    LWdeviceptr m_ring     = 0;
    size_t      m_ringSize = 0;

    struct Request
    {
        LWevent event = 0;
        size_t  begin = 0;
        size_t  end   = 0;
    };

    std::vector<Request> m_requests;

    // Only one request can be outstanding at a time
    std::mutex m_requestLock;

    OptixResult retireRange( LWstream stream, size_t rangeBegin, size_t rangeEnd, ErrorDetails& errDetails );
};

}  // end namespace optix_exp
