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


#include <exp/context/ErrorHandling.h>
#include <exp/context/LaunchResourceManager.h>
#include <exp/context/OptixResultOneShot.h>

#include <corelib/system/LwdaDriver.h>
#include <prodlib/system/Knobs.h>

namespace {
// clang-format off
Knob<unsigned int> k_launchResourcePressureCheckFrequency( RT_DSTRING( "o7.launchResources.pressureCheckFrequency" ), 100, RT_DSTRING( "Check launch resource pressure after N launches.  zero disables." ) );
// clang-format on
}  // namespace

namespace optix_exp {

//////////////////////////////////////////////////////////////////

OptixResult LaunchResourceManager::init( size_t ringSize, unsigned int eventCount, ErrorDetails& errDetails )
{
    m_ringSize = ringSize;
    if( LWresult lwResult = corelib::lwdaDriver().LwMemAlloc( &m_ring, m_ringSize ) )
        return errDetails.logDetails( lwResult, "Failed to allocate launch resources" );
    m_requests.resize( eventCount );
    for( Request& request : m_requests )
    {
        unsigned int Flags = LW_EVENT_DISABLE_TIMING;
        if( LWresult lwResult = corelib::lwdaDriver().LwEventCreate( &request.event, Flags ) )
        {
            destroy( errDetails );
            return errDetails.logDetails( lwResult, "Failed to allocate events for launch resources" );
        }
    }

    return OPTIX_SUCCESS;
}

OptixResult LaunchResourceManager::destroy( ErrorDetails& errDetails )
{
    LWresult lwResult = LWDA_SUCCESS;
    for( Request& request : m_requests )
    {
        if( !request.event )
            continue;
        // Synchronize the event, so that we know that we are done with the resource
        if( LWresult lwResultEvent = corelib::lwdaDriver().LwEventSynchronize( request.event ) )
            lwResult = lwResult ? lwResult : lwResultEvent;
        if( LWresult lwResultEvent = corelib::lwdaDriver().LwEventDestroy( request.event ) )
            lwResult  = lwResult ? lwResult : lwResultEvent;
        request.event = nullptr;
    }
    m_requests.clear();

    if( LWresult lwResultRing = corelib::lwdaDriver().LwMemFree( m_ring ) )
        lwResult = lwResult ? lwResult : lwResultRing;
    m_ring       = 0;

    if( lwResult )
        return errDetails.logDetails( lwResult, "Failed to destroy launch resources" );
    return OPTIX_SUCCESS;
}

/*

  The different allocations of the ring buffer.  If there isn't room, then allocation
  starts back at the beginning of the ring.  In the following diagram C overlaps A, and D
  overlaps A and B.  F overlaps C, D and E.

                +------+----------+
RING PASS 1     |  A   |    B     |
                +---+--+--+-------+
RING PASS 2     | C |  D  |   E   |
                +---+-----+-------+
RING PASS 3     |       F         |
                +-----------------+

  In the following diagram, exelwtion time is roughly represented by the X axis.  In this
  case B is launched after A, but it finishes before A.  E is also launched before C and D
  on the device, despite the host issuing the launch after C and D.

  R<N> : Record event for N
  W<N> : Wait on event RN

                +-----------+
Stream 1        |    A   |RA|
                +---------+-+
                          |
                 +----+   |
Stream 2         |B|RB|   |
                 +--+-+   v
                    |    +-------------+
Stream 3            |    |WA|   C   |RC|
                    |    +-----------+-+
                    |     |          |
                    +--------+       |
                    |     v  v       |
                    |    ++--+--------+
Stream 4            |    |WA|WB| D |RD|
                    |    +----------+-+
                    v               ||
                   ++----------+    ||
Stream 5           |WB|  E  |RE|    +---+
                   +---------+-+     |  |
                             +-------------+
                                     |  |  |
                                     v  v  v
                                    ++--+--+---------+
Stream 6                            |WC|WD|WE| F  |RF|
                                    +----------------+

  When a launch is requested, we check to see if there are any overlapping resources.
  This is done using Resource::begin and Resource::end starting at m_eventTail.  If we
  find an overlapping region we insert a wait on the event recorded at the end of launch
  (stored in Resource::event).  If a new request overlaps the end of a previous event we
  move m_eventTail to the next Resource.  This process is repeated until the new request
  is satisfied (note that F has three waits and D has two).

  retireRange inserts the wait events for all the overlapping requests and moves m_eventTail.
*/

OptixResult LaunchResourceManager::retireRange( LWstream stream, size_t rangeBegin, size_t rangeEnd, ErrorDetails& errDetails )
{
    unsigned int       tail          = m_eventTail;
    bool               pressureEvent = false;
    OptixResultOneShot result;
    while( tail != m_eventHead && m_requests[tail].begin < rangeEnd && m_requests[tail].end > rangeBegin )
    {
        LWevent             event           = m_requests[tail].event;
        static unsigned int eventQueryCount = k_launchResourcePressureCheckFrequency.get();
        if( eventQueryCount && --eventQueryCount == 0 )
        {
            eventQueryCount   = k_launchResourcePressureCheckFrequency.get();
            LWresult lwResult = corelib::lwdaDriver().LwEventQuery( event );
            if( lwResult == LWDA_ERROR_NOT_READY )
            {
                pressureEvent = true;
            }
            else if( lwResult )
            {
                result += errDetails.logDetails( lwResult, "Error querying event on outstanding request" );
            }
        }
        if( LWresult lwResult = corelib::lwdaDriver().LwStreamWaitEvent( stream, event, 0 ) )
        {
            result += errDetails.logDetails( lwResult, "Error recording wait on outstanding request event ( T = "
                                                           + std::to_string( tail ) + ", H = "
                                                           + std::to_string( m_eventHead ) + " )" );
        }
        tail = ( tail + 1 ) % m_requests.size();

        if( m_requests[m_eventTail].end <= rangeEnd )
            m_eventTail = tail;
    }
    if( pressureEvent )
        m_ringPressureCount++;
    return result;
}

OptixResult LaunchResourceManager::acquireResource( LWstream stream, size_t requestSize, LWdeviceptr& resourcePtr, ErrorDetails& errDetails )
{
    if( requestSize > m_ringSize )
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, "Launch resource request size ( " + std::to_string( requestSize )
                                                                      + " ) is larger than ring buffer size ( "
                                                                      + std::to_string( m_ringSize ) + " )" );
    if( requestSize % 128 != 0 )
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR,
                                      "requestSize for launch resource is not multiple of 128: " + std::to_string( requestSize ) );


    m_requestLock.lock();

    size_t requestEnd = m_ringHead + requestSize;
    if( requestEnd > m_ringSize )
    {
        // Doesn't fit in the end.  Retire everything to the end and start at the
        // beginning.  All resources must be retired in order (starting at m_eventTail).
        if( OptixResult result = retireRange( stream, m_ringHead, m_ringSize, errDetails ) )
        {
            m_requestLock.unlock();
            return result;
        }
        m_ringHead = 0;
        requestEnd = requestSize;
    }

    // At this point, there should be enough space to allocate the resource.  Starting at
    // m_ringHead retire the necessary requests to make space for the new request.
    if( OptixResult result = retireRange( stream, m_ringHead, requestEnd, errDetails ) )
    {
        m_requestLock.unlock();
        return result;
    }

    resourcePtr = m_ring + m_ringHead;
    return OPTIX_SUCCESS;
}

/*

  If the number of outstanding events exceeds the maximum number of Request objects, we
  will need to wait on the host for the event to free up.  Note that we only need to wait
  before we record the event, so we can launch the work in the meantime if the launch
  resources are available.

  In the following example there are only two Request slots (signified by R1 and R2).

  Stream 3 waits on A for launch resources, then can proceed to launch C.  The host must
  wait after launching C on stream 3 for A to finish (R1A) before recording the event in
  stream 3.  This is because the event C needs, R1, is in use by A.

  Stream 4 waits on A and B for launch resources, launches, then waits on the host host
  for B for its event (R2).

  Stream 5 waits for B for launch resources, but the host must block until C has finished
  (R1C) before reusing R1 and recording R1E.  In this case E can complete on the device
  before R1 is available to be reused and recorded.  The host thread launching E would
  block until C has finished.

  Stream 6 doesn't wait to launch F at all, but must wait on the host for D when the event
  (R2) is available.

  By setting the default number of outstanding events large enough, we hope to never hit
  this case in practice.  As of 3/20/2019 this number is 1000 (configured in knob
  o7.launchResources.numEvents).

              +------+-------+
RING PASS 1   |  A   |    B  |
              +-----++--+----++--+
RING PASS 2   |  C  | D | E   |F |
              +-----+---+-----+--+


          +------------+
Stream 1  |    A   |R1A|
          +---------+--+
                    |
           +-----+  +-----------+
Stream 2   |B|R2B|  |           |
           +--+--+  v           v
              |    ++-----------+------+
Stream 3      |    |WA |   C   |WHA|R1C|
              |    +----------------+--+
              |     |               |
              +---------+-------+   |
              |     |   |       |   |
              |     v   v       v   |
              |    ++------------------+
Stream 4      |    |WA |WB | D |WHB|R2D|
              |    +-----------------+-+
              |                     ||
              v                     v|
             ++--------+           ++------+
Stream 5     |WB |  E  |           |WHC|R1E|
             +---------+           +-------+
                                     |
                                     v
               +--------------+     +-------+
Stream 6       |      F       |     |WHD|R2F|
               +--------------+     +-------+


 */
OptixResult LaunchResourceManager::releaseResource( LWstream stream, size_t requestSize, ErrorDetails& errDetails )
{
    // We always want to unlock when exiting.  Use std::adopt_lock to take the mutex which
    // should already be in a locked state.
    std::lock_guard<std::mutex> unlockMe( m_requestLock, std::adopt_lock );

    OptixResultOneShot result;

    // Make sure we have a request slot
    const unsigned int nextSlot = ( m_eventHead + 1 ) % m_requests.size();
    if( nextSlot == m_eventTail )
    {
        LWevent             event           = m_requests[m_eventTail].event;
        static unsigned int eventQueryCount = k_launchResourcePressureCheckFrequency.get();
        if( eventQueryCount && --eventQueryCount == 0 )
        {
            eventQueryCount   = k_launchResourcePressureCheckFrequency.get();
            LWresult lwResult = corelib::lwdaDriver().LwEventQuery( event );
            if( lwResult == LWDA_ERROR_NOT_READY )
            {
                m_eventPressureCount++;
            }
            else if( lwResult )
            {
                result += errDetails.logDetails( lwResult, "Error querying event on outstanding request" );
            }
        }
        if( LWresult lwResult = corelib::lwdaDriver().LwEventSynchronize( event ) )
        {
            result += errDetails.logDetails( lwResult, "Error synchronizing on an outstanding request" );
        }
        m_eventTail = ( m_eventTail + 1 ) % m_requests.size();
    }

    Request& request = m_requests[m_eventHead];
    m_eventHead      = nextSlot;

    size_t requestEnd = m_ringHead + requestSize;
    request.begin     = m_ringHead;
    request.end       = requestEnd;
    m_ringHead        = requestEnd;

    // Record on user stream
    if( LWresult lwResult = corelib::lwdaDriver().LwEventRecord( request.event, stream ) )
    {
        result += errDetails.logDetails( lwResult, "Error recording resource event on user stream" );
    }

    return result;
}

}  // end namespace optix_exp
