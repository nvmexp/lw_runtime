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
#include <Device/LWDADevice.h>
#include <ExelwtionStrategy/LWDAWaitHandle.h>
#include <ExelwtionStrategy/LaunchResources.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;
using namespace optix::lwca;

LWDAWaitHandle::LWDAWaitHandle( std::shared_ptr<LaunchResources> launchResources, LWDADevice* primary_device, unsigned int number_active_devices )
    : WaitHandle( launchResources )
    , m_primaryDevice( primary_device )
    , m_state( Inactive )
{
    m_primaryDevice->makeLwrrent();
    m_t0 = Event::create();
    m_t1 = Event::create();
}

LWDAWaitHandle::~LWDAWaitHandle()
{
    RT_ASSERT_NOTHROW( m_primaryDevice->isEnabled(),
                       "A LWDAWaitHandle is still alive when the device is already disabled." );

    // Block before our memory is destroyed since there may be
    // outstanding memcopies in flight.
    try
    {
        if( m_state == InProgress )
            block();
        reset();  // clear status_per_device.
        m_primaryDevice->makeLwrrent();
        m_t0.destroy();
        m_t1.destroy();
    }
    catch( ... )
    { /* swallow exceptions in destructor */
    }
}

void LWDAWaitHandle::reset()
{
    RT_ASSERT( m_state == Inactive || m_state == Complete );
    m_state = Inactive;
    m_launchResources->resetHostFrameStatus();
}

void LWDAWaitHandle::recordStart( const Stream& stream )
{
    RT_ASSERT( m_state == Inactive );
    m_state = Started;
    m_primaryDevice->makeLwrrent();
    m_t0.record( stream );
}

void LWDAWaitHandle::recordStart()
{
    recordStart( m_primaryDevice->primaryStream() );
}

void LWDAWaitHandle::recordStop( const Stream& stream )
{
    RT_ASSERT( m_state == Started );
    m_state = InProgress;
    m_primaryDevice->makeLwrrent();
    m_t1.record( stream );
}

void LWDAWaitHandle::recordStop()
{
    recordStop( m_primaryDevice->primaryStream() );
}

void LWDAWaitHandle::block()
{
    RT_ASSERT( m_state == InProgress );
    m_state = Blocking;
    m_primaryDevice->makeLwrrent();
    m_t1.synchronize();
    m_state = Complete;
}

float LWDAWaitHandle::getElapsedMilliseconds() const
{
    RT_ASSERT( m_state == Complete );
    return Event::elapsedTime( m_t0, m_t1 );
}

void LWDAWaitHandle::addDependencyFrom( lwca::Event& ev )
{
    // 'Wait' on event. In practice, just add dependency on it.
    m_primaryDevice->makeLwrrent();
    m_primaryDevice->primaryStream().waitEvent( ev, 0 );
}
