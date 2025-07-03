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

#include <Device/LWDADevice.h>
#include <Device/Device.h>
#include <ExelwtionStrategy/ExelwtionStrategy.h>
#include <ExelwtionStrategy/LaunchResources.h>
#include <ExelwtionStrategy/Plan.h>
#include <ExelwtionStrategy/RTX/CompiledProgramCache.h>
#include <Memory/BulkMemoryPool.h>

#include <array>
#include <deque>
#include <memory>
#include <mutex>

namespace llvm {
class Module;
}

namespace optix {
class Context;
class FrameTask;

class RTXES : public ExelwtionStrategy
{
  public:
    ~RTXES() override;
    std::unique_ptr<Plan> createPlan( unsigned int entry, int dimensionality, const DeviceSet& devices, int numLaunchDevices ) const override;

    virtual std::shared_ptr<LaunchResources> acquireLaunchResources( const DeviceSet&           devices,
                                                                     const FrameTask*           ft,
                                                                     const optix::lwca::Stream& syncStream,
                                                                     const unsigned int         width,
                                                                     const unsigned int         height,
                                                                     const unsigned int         depth );
    virtual void releaseLaunchResources( LaunchResources* launchResources );

    void preSetActiveDevices( const DeviceArray& removedDevices );

    // Remove programs compiled for the specified devices from the cache.
    void removeProgramsForDevices( const DeviceArray& devices );

  protected:
    RTXES( Context* context );

    // Owner of the compiled programs.
    std::unique_ptr<CompiledProgramCache> m_compiledProgramCache;

    // Only context can construct
    friend class Context;

  private:
    // Used to decide which stream index to use for RTXLaunchResources instances.
    std::atomic_uint m_streamIndexCounters[OPTIX_MAX_DEVICES] = {};

    // Protects book-keeping needed to aquire and release launch resources.
    std::mutex m_launchResourcesMutex;

    // Stores events per device for reuse by the launch resources.
    std::array<std::deque<lwca::Event>, OPTIX_MAX_DEVICES> m_eventQueue;

    // Stores lwca memory buffers per device for reuse by the launch resources.
    typedef std::pair<LWdeviceptr, size_t>                   devptr_size_t;
    std::array<std::deque<LWdeviceptr>, OPTIX_MAX_DEVICES>   m_frameStatusQueue;
    std::array<std::deque<devptr_size_t>, OPTIX_MAX_DEVICES> m_launchBufferQueue;
    std::array<std::deque<devptr_size_t>, OPTIX_MAX_DEVICES> m_scratchBufferQueue;
};
}
