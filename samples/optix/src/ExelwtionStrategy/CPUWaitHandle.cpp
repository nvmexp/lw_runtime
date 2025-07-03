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

#include <ExelwtionStrategy/CPUWaitHandle.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;

CPUWaitHandle::CPUWaitHandle( std::shared_ptr<LaunchResources> launchResources )
    : WaitHandle( launchResources )
    , m_state( Inactive )
{
}

CPUWaitHandle::~CPUWaitHandle()
{
}

void CPUWaitHandle::recordStart()
{
    RT_ASSERT( m_state == Inactive );
    m_state = Started;
    m_t0    = corelib::getTimerTick();
}

void CPUWaitHandle::recordStop()
{
    RT_ASSERT( m_state == Started );
    m_state = InProgress;
    m_t1    = corelib::getTimerTick();
}

void CPUWaitHandle::block()
{
    RT_ASSERT( m_state == InProgress );
    m_state = Blocking;
    // CPU lwrrently synchronous.  Nothing to do (yet)
    m_state = Complete;
}

float CPUWaitHandle::getElapsedMilliseconds() const
{
    RT_ASSERT( m_state == Complete );
    return corelib::getDeltaSeconds( m_t0, m_t1 ) / 1000;
}
