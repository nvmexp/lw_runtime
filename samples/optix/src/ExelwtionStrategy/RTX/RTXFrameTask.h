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

#include <Context/SBTManager.h>
#include <Control/ManagedLWDAModule.h>
#include <ExelwtionStrategy/FrameTask.h>
#include <ExelwtionStrategy/RTX/CompiledProgramCache.h>
#include <ExelwtionStrategy/RTX/RTXLaunchResources.h>
#include <ExelwtionStrategy/RTX/RTXRuntime.h>
#include <Util/DevicePtr.h>
#include <map>
#include <rtcore/interface/types.h>
#include <vector>


namespace optix {
class ConstantMemAllocations;
class LWDADevice;
class RTXWaitHandle;
class WaitHandle;
namespace lwca {
class Function;
class Event;
}

class RTXFrameTask : public FrameTask
{
  public:
    RTXFrameTask( Context* context, const DeviceSet& launchDevices, const std::vector<unsigned int>& validEntryPoints, const int traversableGraphDepth );

    ~RTXFrameTask() override;

    void activate() override;
    void deactivate() override;
    void launch( std::shared_ptr<WaitHandle> waiter,
                 unsigned int                entry,
                 int                         dimensionality,
                 RTsize                      width,
                 RTsize                      height,
                 RTsize                      depth,
                 unsigned int                subframe_index,
                 const cort::AabbRequest&    aabbParams ) override;

    std::shared_ptr<WaitHandle> acquireWaitHandle( std::shared_ptr<LaunchResources>& launchResources );

    void setDeviceInfo( LWDADevice*                   device,
                        RtcPipeline                   pipeline,
                        const ConstantMemAllocations& constMemAllocs,
                        bool                          isAabbLaunch,
                        const unsigned int            directCallableStackSizeFromTraversal,
                        const unsigned int            directCallableStackSizeFromState,
                        const unsigned int            continuationStackSize );

    void addCompiledModule( const CanonicalProgram* cp, SemanticType stype, SemanticType inheritedStype, Device* device, ModuleEntryRefPair& compilerOutput );

    const RtcPipeline getRtcPipeline( const unsigned int allDeviceListIndex ) const;

    void setTraversableGraphDepth( const unsigned int traversableGraphDepth );

  private:
    //------------------------------------------------------------------------
    // Caching used for packing the SBTRecord headers after compile
    //------------------------------------------------------------------------
    SBTManager::CompiledProgramMap m_perPlanCompiledProgramCache;

    std::vector<unsigned int> m_validEntryPoints;
    struct PerLaunchDevice
    {
        LWDADevice*                    device            = nullptr;
        RtcPipeline                    pipeline          = nullptr;
        LWdeviceptr                    launchbuf         = 0;
        LWdeviceptr                    scratchbuf        = 0;
        size_t                         scratchbytes      = 0;
        LWdeviceptr                    toolsOutputVA     = 0;
        size_t                         toolsOutputSize   = 0;
        std::unique_ptr<char[]>        toolsOutputBuffer = nullptr;
        RtcPipelineInfoProfileMetadata profileMetadata   = {};

        LWdeviceptr                   statusDevicePtr    = 0;
        LWdeviceptr                   const_Global       = 0;
        LWdeviceptr                   const_ObjectRecord = 0;
        LWdeviceptr                   const_BufferTable  = 0;
        LWdeviceptr                   const_ProgramTable = 0;
        LWdeviceptr                   const_TextureTable = 0;
        std::unique_ptr<cort::Global> globalStruct;

        unsigned int directCallableStackSizeFromTraversal = 0;
        unsigned int directCallableStackSizeFromState     = 0;
        unsigned int continuationStackSize                = 0;
        unsigned int traversableGraphDepth                = 0;

        // When limiting the active launch indices specify the min and max ranges.  This
        // should be sizeof(unsigned[6]).
        LWdeviceptr minMaxLaunchIndex = 0;
    };
    std::vector<PerLaunchDevice> m_perLaunchDevice;

    struct PerDeviceOffsets
    {
        const static size_t INVALID            = ~0;
        size_t              minMaxLaunchIndex  = PerDeviceOffsets::INVALID;
        size_t              const_Global       = PerDeviceOffsets::INVALID;
        size_t              const_ObjectRecord = PerDeviceOffsets::INVALID;
        size_t              const_BufferTable  = PerDeviceOffsets::INVALID;
        size_t              const_ProgramTable = PerDeviceOffsets::INVALID;
        size_t              const_TextureTable = PerDeviceOffsets::INVALID;
    };
    std::vector<PerDeviceOffsets> m_perDeviceOffsets;

    DeviceSet m_devices;

    std::map<LWcontext, lwca::Event> m_events;

    unsigned int m_traversableGraphDepth = 0;

    // Helper function
    void initializePadPointers( struct PerLaunchDevice& pad, const RTXLaunchResources& res, const unsigned int allDeviceListIndex );

    // Helper that creates an event that is compatible with the provided stream (created using the same lwca context)
    lwca::Event createEventForStream( optix::lwca::Stream stream, LWresult* returnResult = nullptr ) const;
};
}  // namespace optix
