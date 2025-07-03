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

#pragma once

#include <Device/DeviceSet.h>
#include <ExelwtionStrategy/WaitHandle.h>

#include <vector>

namespace optix {

class Context;
class DeviceManager;

class RTXWaitHandle : public WaitHandle
{
  public:
    RTXWaitHandle( std::shared_ptr<LaunchResources> launchResources, Context* context );
    virtual ~RTXWaitHandle();

    void  block() override;
    float getElapsedMilliseconds() const override;

    void reset( const DeviceSet& launchDevices );
    void recordStart();
    void recordStop();

    // Make sure that the provided stream can't continue until the work it represents has
    // finished. Must be called after recordStop();
    void syncStream( const lwca::Stream& stream );

    // Must only be called by the context.
    void releaseResources() override;

  private:
    void recordEvent( std::vector<lwca::Event>& events, int allDeviceListIndex );

    enum class State
    {
        Inactive,
        Started,
        InProgress,
        Blocking,
        Complete
    } m_state = State::Inactive;

    DeviceSet m_launchDevices;
    Context*  m_context = nullptr;
};
}
