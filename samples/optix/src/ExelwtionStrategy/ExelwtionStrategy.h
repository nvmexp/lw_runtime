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

#include <LWCA/Stream.h>

#include <memory>

namespace optix {
class Context;
class DeviceSet;
class FrameTask;
class LaunchResources;
class Plan;

// Base class for all exelwtion strategies.
//
// Every available exelwtion strategy inherits from this class. The subclass is the
// entry point for the creation of an exelwtion plan for the given strategy.
//
// Additionally, the exelwtion strategy is responsible for managing launch resources.
// Every launch of frame task must have an unique set of launch resources of its correct
// strategy, and this base class is responsible for the acquire, release and reuse of
// available resources.
class ExelwtionStrategy
{
  public:
    virtual std::unique_ptr<Plan> createPlan( unsigned int entry, int dimensionality, const DeviceSet& devices, int numLaunchDevices ) const = 0;
    virtual std::shared_ptr<LaunchResources> acquireLaunchResources( const DeviceSet&           devices,
                                                                     const FrameTask*           ft,
                                                                     const optix::lwca::Stream& syncStream,
                                                                     const unsigned int         width,
                                                                     const unsigned int         height,
                                                                     const unsigned int         depth );
    virtual void releaseLaunchResources( LaunchResources* launchResources );

    Context* getContext();

  protected:
    ExelwtionStrategy( Context* context );
    virtual ~ExelwtionStrategy();

    friend class Context;

    Context* m_context;

  private:
    ExelwtionStrategy& operator=( const ExelwtionStrategy& ) = delete;
    ExelwtionStrategy( const ExelwtionStrategy& )            = delete;
};
}
