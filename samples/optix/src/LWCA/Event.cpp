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

#include <LWCA/ErrorCheck.h>
#include <LWCA/Stream.h>
#include <corelib/system/LwdaDriver.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;
using namespace optix::lwca;
using namespace corelib;

Event::Event()
    : m_event( nullptr )
{
}

Event::Event( LWevent event )
    : m_event( event )
{
}

LWevent Event::get()
{
    return m_event;
}

const LWevent Event::get() const
{
    return m_event;
}

Event Event::create( unsigned int Flags, LWresult* returnResult )
{
    LWevent event = nullptr;
    CHECK( lwdaDriver().LwEventCreate( &event, Flags ) );
    return Event( event );
}

void Event::destroy( LWresult* returnResult )
{
    RT_ASSERT( m_event != nullptr );
    CHECK( lwdaDriver().LwEventDestroy( m_event ) );
    m_event = nullptr;
}

float Event::elapsedTime( const Event& start, const Event& end, LWresult* returnResult )
{
    RT_ASSERT( start.get() != nullptr );
    RT_ASSERT( end.get() != nullptr );
    float result = -1;
    CHECK( lwdaDriver().LwEventElapsedTime( &result, start.get(), end.get() ) );
    return result;
}

bool Event::query( LWresult* returnResult ) const
{
    RT_ASSERT( m_event != nullptr );
    LWresult err = lwdaDriver().LwEventQuery( m_event );
    if( err == LWDA_SUCCESS )
        return true;
    if( err == LWDA_ERROR_NOT_READY )
        return false;

    // A real error
    if( returnResult )
    {
        *returnResult = err;
        return false;
    }
    else
    {
        throw prodlib::LwdaError( RT_EXCEPTION_INFO, "lwEventQuery( m_event )", err );
    }
}

void Event::record( const Stream& stream, LWresult* returnResult )
{
    RT_ASSERT( stream.get() != nullptr );
    RT_ASSERT( m_event != nullptr );
    CHECK( lwdaDriver().LwEventRecord( m_event, stream.get() ) );
}

// Waits for an event to complete.
void Event::synchronize( LWresult* returnResult ) const
{
    RT_ASSERT( m_event != nullptr );
    CHECK( lwdaDriver().LwEventSynchronize( m_event ) );
}
