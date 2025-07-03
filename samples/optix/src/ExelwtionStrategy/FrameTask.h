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

#include <Context/ProfileMapping.h>
#include <Context/UpdateManager.h>

#include <memory>

namespace cort {
struct AabbRequest;
}

namespace optix {
class Context;
class LaunchResources;
class ProfileMapping;
class WaitHandle;

class FrameTask
{
  public:
    FrameTask& operator=( const FrameTask& ) = delete;
    FrameTask( const FrameTask& )            = delete;

    virtual ~FrameTask();

    virtual void activate()   = 0;
    virtual void deactivate() = 0;
    virtual void launch( std::shared_ptr<WaitHandle> waitHandle,
                         unsigned int                entry,
                         int                         dimensionality,
                         RTsize                      width,
                         RTsize                      height,
                         RTsize                      depth,
                         unsigned int                subframe_index,
                         const cort::AabbRequest&    aabbParams ) = 0;

    // Optionally add a mapping to decode profiling information
    void setProfileMapping( const std::shared_ptr<ProfileMapping>& newMapping );
    std::shared_ptr<ProfileMapping> getProfileMapping() const;

    virtual std::shared_ptr<WaitHandle> acquireWaitHandle( std::shared_ptr<LaunchResources>& launchResources ) = 0;

  protected:
    FrameTask( Context* );
    Context*                        m_context;
    std::shared_ptr<ProfileMapping> m_profileMapping;

    // If we are limiting the min and max launch indices, these will be set to useful
    // values.
    unsigned int minMaxLaunchIndex[6] = {0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF};
};
}
