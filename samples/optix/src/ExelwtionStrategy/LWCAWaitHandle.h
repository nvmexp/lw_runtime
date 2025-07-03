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

#pragma once

#include <LWCA/Event.h>
#include <ExelwtionStrategy/WaitHandle.h>
#include <memory>

namespace optix {

class LWDADevice;

class LWDAWaitHandle : public WaitHandle
{
  public:
    LWDAWaitHandle( std::shared_ptr<LaunchResources> launchResources, LWDADevice* primary_device, unsigned int number_active_devices );
    ~LWDAWaitHandle() override;
    virtual void reset();

    void  block() override;
    float getElapsedMilliseconds() const override;

    // Helper methods, use device primaryStream
    void recordStart();
    void recordStop();

    // Instruct the WaitHandle to wait for this event when the WaitHandle blocks.
    // This call does not block, it only adds the dependency to a list.
    void addDependencyFrom( lwca::Event& ev );

  protected:
    LWDADevice* m_primaryDevice = nullptr;
    lwca::Event m_t0;
    lwca::Event m_t1;

    enum State
    {
        Inactive,
        Started,
        InProgress,
        Blocking,
        Complete
    } m_state;

    void recordStart( const lwca::Stream& stream );
    void recordStop( const lwca::Stream& stream );

    LWDAWaitHandle& operator=( const LWDAWaitHandle& );
    LWDAWaitHandle( const LWDAWaitHandle& );
};
}
