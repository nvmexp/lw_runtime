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

#include <ExelwtionStrategy/LaunchResources.h>
#include <memory>

namespace cort {
struct FrameStatus;
}

namespace optix {

class WaitHandle
{
  public:
    WaitHandle( std::shared_ptr<LaunchResources> launchResources );
    virtual ~WaitHandle();

    // Disallow copying
    WaitHandle( const WaitHandle& ) = delete;
    WaitHandle& operator=( const WaitHandle& ) = delete;

    virtual void block() = 0;

    // NOTE: May only be called when rendering synchronously.
    virtual float getElapsedMilliseconds() const = 0;

    void               checkFrameStatus() const;
    cort::FrameStatus* getFrameStatusHostPtr( int activeDeviceListIndex );

    struct contextIndex_fn
    {
        int& operator()( std::shared_ptr<WaitHandle>& entry ) { return entry->m_contextListIndex; }
        int& operator()( WaitHandle* entry ) { return entry->m_contextListIndex; }
    };

    std::shared_ptr<LaunchResources>& getLaunchResources();

    // Releases the wait handle resources. Must only be called by the context.
    virtual void releaseResources();

  protected:
    std::shared_ptr<LaunchResources> m_launchResources;

    mutable int m_contextListIndex = -1;
};
}
