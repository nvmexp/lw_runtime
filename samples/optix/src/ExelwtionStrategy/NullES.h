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

#include <ExelwtionStrategy/ExelwtionStrategy.h>
#include <ExelwtionStrategy/FrameTask.h>
#include <ExelwtionStrategy/LaunchResources.h>
#include <ExelwtionStrategy/Plan.h>
#include <ExelwtionStrategy/WaitHandle.h>

namespace optix {
class NullES : public ExelwtionStrategy
{
  public:
    ~NullES() override;
    std::unique_ptr<Plan> createPlan( unsigned int entry, int dimensionality, const DeviceSet& devices, int numLaunchDevices ) const override;

  protected:
    NullES( Context* context );

    // Only context can construct
    friend class Context;
};

class NullESPlan : public Plan
{
  public:
    NullESPlan( Context* context, const DeviceSet& devices );
    ~NullESPlan() override;

    std::string summaryString() const override;
    bool supportsLaunchConfiguration( unsigned int entry, int dimensionality, const DeviceSet& devices, int numLaunchDevices ) const override;
    bool isCompatibleWith( const Plan* otherPlan ) const override;
    void         compile() const override;
    virtual void update() const;

  private:
};

class NullESFrameTask : public FrameTask
{
  public:
    NullESFrameTask( Context* context );
    ~NullESFrameTask() override;

    void activate() override;
    void deactivate() override;
    void launch( std::shared_ptr<WaitHandle> waitHandle,
                 unsigned int                entry,
                 int                         dimensionality,
                 RTsize                      width,
                 RTsize                      height,
                 RTsize                      depth,
                 unsigned int                subframe_index,
                 const cort::AabbRequest&    aabbParams ) override;
    std::shared_ptr<WaitHandle> acquireWaitHandle( std::shared_ptr<LaunchResources>& launchResources );

  private:
};

class NullWaitHandle : public WaitHandle
{
  public:
    NullWaitHandle( std::shared_ptr<LaunchResources> launchResources );
    ~NullWaitHandle() override;
    void  block() override;
    float getElapsedMilliseconds() const override;

  private:
    NullWaitHandle& operator=( const NullWaitHandle& ) = delete;
    NullWaitHandle( const NullWaitHandle& )            = delete;
};
}
