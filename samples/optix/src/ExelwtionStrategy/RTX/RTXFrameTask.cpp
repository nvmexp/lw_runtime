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

#include <ExelwtionStrategy/RTX/RTXFrameTask.h>

#include <LWCA/ErrorCheck.h>
#include <LWCA/Event.h>
#include <LWCA/Memory.h>
#include <Context/BindingManager.h>
#include <Context/Context.h>
#include <Context/ProfileManager.h>
#include <Context/ProgramManager.h>
#include <Context/RTCore.h>
#include <Context/SBTManager.h>
#include <Context/TableManager.h>
#include <Context/WatchdogManager.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <ExelwtionStrategy/Common/ConstantMemAllocations.h>
#include <ExelwtionStrategy/FrameStatus.h>
#include <ExelwtionStrategy/RTX/RTXLaunchResources.h>
#include <ExelwtionStrategy/RTX/RTXPlan.h>
#include <ExelwtionStrategy/RTX/RTXWaitHandle.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <Memory/DemandLoad/PagingService.h>
#include <Memory/MemoryManager.h>
#include <corelib/misc/ProfileDump.h>
#include <corelib/system/LwdaDriver.h>
#include <corelib/system/LwdaDriver.h>
#include <prodlib/misc/TimeViz.h>
#include <prodlib/system/Knobs.h>

using namespace optix;
using namespace optix::lwca;

namespace {
// clang-format off
PublicKnob<bool>   k_enableRTcoreProfiling(      RT_PUBLIC_DSTRING( "rtx.enableRTcoreProfiling" ),       false, RT_PUBLIC_DSTRING( "Enables or disables RTcore profiling. If enabled the profiling data of each device will be written to disk." ) );
Knob<unsigned int> k_rtcoreProfilingDumpCounter( RT_DSTRING(        "rtx.dumpRTcoreProfilingInterval" ), 1,     RT_DSTRING(        "Dump the profiling data every N frames." ) );
Knob<std::string>  k_limitActiveLaunchIndices(   RT_DSTRING(        "launch.limitActiveIndices" ),       "",    RT_DSTRING(        "When specified limit which launch indices are active. Syntax: [minX, maxX], [minY, maxY]" ) );
Knob<int>          k_traversableGraphDepth(      RT_DSTRING(        "rtx.traversableGraphDepth" ),       0,     RT_DSTRING(        "Set max traversable graph depth for rtcore." ) );
Knob<std::string>  k_gatherLwptiMetrics(         RT_DSTRING(        "launch.gatherMetrics" ),            "",    RT_DSTRING(        "Comma separated list of LWPTI metrics to gather." ) );
Knob<bool>         k_gpuWarmupEnabled(           RT_DSTRING(        "rtx.gpuWarmupEnabled" ),            true,  RT_DSTRING(        "Enable the GPU warm-up kernel." ) );
// clang-format on
}

// Holds the wait handle so that it can be retrieved from the lwca callback function.
// This also makes sure that the wait handle will survive until the callback has been
// made, even if it is cleaned up everywhere else.
struct Callback_data
{
    optix::Context*             m_context;
    std::shared_ptr<WaitHandle> m_waitHandle;
};

// This jump function only goals are to get the callback, unpack the pointers,
// let the context handle the data and get rid of the cb_data pack.
// TODO: this struct should be part of the launch resources, then it does not get lost.
static void LWDA_CB waitHandleCallback( void* data )
{
    RT_ASSERT( data != nullptr );

    Callback_data* dp = reinterpret_cast<Callback_data*>( data );
    dp->m_context->launchFinished( dp->m_waitHandle );
    delete dp;
}


RTXFrameTask::RTXFrameTask( optix::Context*                  context,
                            const DeviceSet&                 launchDevices,
                            const std::vector<unsigned int>& validEntryPoints,
                            const int                        traversableGraphDepth )
    : FrameTask( context )
    , m_validEntryPoints( validEntryPoints )
    , m_perLaunchDevice( launchDevices.count() )
    , m_perDeviceOffsets( launchDevices.count() )
    , m_devices( launchDevices )
{
    setTraversableGraphDepth( traversableGraphDepth );
}

RTXFrameTask::~RTXFrameTask()
{
    for( const auto& pad : m_perLaunchDevice )
    {
        m_context->getRTCore()->pipelineDestroy( pad.pipeline );

        if( pad.toolsOutputVA != 0 )
            corelib::lwdaDriver().LwMemFree( pad.toolsOutputVA );
    }

    for( auto& iter : m_events )
        iter.second.destroy();
}

void RTXFrameTask::activate()
{
    m_context->getSBTManager()->updatePrograms( m_perPlanCompiledProgramCache );
}

void RTXFrameTask::deactivate()
{
}


void RTXFrameTask::launch( std::shared_ptr<WaitHandle> waitHandle,
                           unsigned int                entry,
                           int                         dimensionality,
                           RTsize                      width,
                           RTsize                      height,
                           RTsize                      depth,
                           unsigned int                subframe_index,
                           const cort::AabbRequest&    aabbRequest )
{
    TIMEVIZ_FUNC;

    std::shared_ptr<LaunchResources>& launchResources = waitHandle->getLaunchResources();

    // Cast to RTXLaunchResources
    const RTXLaunchResources* rtxres = dynamic_cast<const RTXLaunchResources*>( launchResources.get() );
    RT_ASSERT_MSG( rtxres != nullptr,
                   "RTXFrameTask could not launch, launch resources was not of type RTXLaunchResources." );

    // Cast to RTXWaitHandle
    RTXWaitHandle* waiter = dynamic_cast<RTXWaitHandle*>( waitHandle.get() );
    RT_ASSERT_MSG( waiter != nullptr, "RTXFrameTask could not launch, wait handle was not of type RTXWaitHandle." );

    const DeviceSet& launchDevices = launchResources->getDevices();

    ProfileManager* pm      = m_context->getProfileManager();
    TableManager*   tm      = m_context->getTableManager();
    MemoryManager*  mm      = m_context->getMemoryManager();
    PagingService*  pageMgr = m_context->getPagingManager();

    // The sync stream and event. If set the streams used for work need to wait on this stream before any work can be queued up.
    const lwca::Stream& syncStream = rtxres->getSyncStream();
    lwca::Event         syncEvent;
    if( syncStream.get() != nullptr )
    {
        LWcontext ctx = nullptr;
        corelib::lwdaDriver().LwStreamGetCtx( syncStream.get(), &ctx );
        RT_ASSERT_MSG( ctx != nullptr,
                       "RTXFrameTask could not launch. Failed to get the lwca context of the sync stream." );
        lwca::Event& ev = m_events[ctx];
        if( ev.get() == nullptr )
            ev    = createEventForStream( syncStream );
        syncEvent = ev;
    }

    // Update missing GeometryGroups in SBT and synchronize to devices.
    m_context->getSBTManager()->finalizeBeforeLaunch();

    // Initialize waiter set.
    waiter->reset( launchDevices );

    // Record the event before all devices have started. Use this for profiling.
    waiter->recordStart();

    // Skip launch if any of the dimensions is zero.
    if( width == 0 || height == 0 || depth == 0 )
    {
        waiter->recordStop();
        return;
    }

    if( syncEvent.get() != nullptr )
        syncEvent.record( syncStream );

    // 2nd. pass: data sync
    for( int allDeviceListIndex : launchDevices )
    {
        PerLaunchDevice& pad = m_perLaunchDevice[m_devices.getArrayPosition( allDeviceListIndex )];
        pad.device->makeLwrrent();
        Stream stream = rtxres->getStream( allDeviceListIndex );

        // Make sure that the stream waits for any work done on the primary stream
        pad.device->syncStreamToPrimaryStream( stream );

        // Make sure that the stream waits for the sync stream
        if( syncEvent.get() != nullptr )
            stream.waitEvent( syncEvent, 0 );

        // Process launch buffer and offsets from the pipeline to initialize pointers
        initializePadPointers( pad, *rtxres, allDeviceListIndex );

        RT_ASSERT_MSG( pad.statusDevicePtr, "RTXFrameTask status pointer not set. Task was not properly activated." );

#if 0
    // Assign bound textures from managed memory (Fermi and texheap)
    // TODO: subscribe to events and do this incrementally? (probably not worth it since we don't support Fermi any more)
    {
      const BitSet& bound = mm->getAssignedTexReferences( pad.device );
      for( BitSet::const_iterator iter = bound.begin(); iter != bound.end(); ++iter )
      {
        unsigned int unit = *iter;
        RT_ASSERT( unit < pad.texrefs.size() );

        lwca::TexRef texref = pad.texrefs[unit];
        mm->bindTexReference( pad.device, unit, texref );
      }

      // MM may have changed the current context. Ensure we are in the right one.
      pad.device->makeLwrrent();
    }
#endif

        // Get table pointers
        void* objectRecordPtr         = tm->getObjectRecordDevicePointer( pad.device );
        void* bufferHeaderPtr         = tm->getBufferHeaderDevicePointer( pad.device );
        void* textureHeaderPtr        = tm->getTextureHeaderDevicePointer( pad.device );
        void* programHeaderPtr        = tm->getProgramHeaderDevicePointer( pad.device );
        void* traversableHeaderPtr    = tm->getTraversableHeaderDevicePointer( pad.device );
        void* dynamicVariableTablePtr = tm->getDynamicVariableTableDevicePointer( pad.device );
        void* profileDataPtr = pm->getProfileDataDevicePointer( pad.device );  // can be null if the profiler is disabled

        const int3& printIndex = m_context->getPrintLaunchIndex();

        cort::Global& global             = *pad.globalStruct.get();
        global.objectRecords             = reinterpret_cast<char*>( objectRecordPtr );
        global.bufferTable               = reinterpret_cast<cort::Buffer*>( bufferHeaderPtr );
        global.textureTable              = reinterpret_cast<cort::TextureSampler*>( textureHeaderPtr );
        global.programTable              = reinterpret_cast<cort::ProgramHeader*>( programHeaderPtr );
        global.traversableTable          = reinterpret_cast<cort::TraversableHeader*>( traversableHeaderPtr );
        global.dynamicVariableTable      = reinterpret_cast<unsigned short*>( dynamicVariableTablePtr );
        global.pageUsageBits             = reinterpret_cast<cort::uint*>( pageMgr->getUsageBits( pad.device ) );
        global.pageResidenceBits         = reinterpret_cast<cort::uint*>( pageMgr->getResidenceBits( pad.device ) );
        global.pageTable                 = reinterpret_cast<cort::uint64*>( pageMgr->getPageTable( pad.device ) );
        global.tileArrays                = reinterpret_cast<cort::uint64*>( pageMgr->getTileArrays( pad.device ) );
        global.demandLoadMode            = static_cast<cort::PagingMode>( pageMgr->getLwrrentPagingMode() );
        global.profileData               = reinterpret_cast<cort::uint64*>( profileDataPtr );
        global.statusReturn              = reinterpret_cast<cort::FrameStatus*>( pad.statusDevicePtr );
        global.objectRecordsSize         = tm->getObjectRecordSize();
        global.numBuffers                = tm->getNumberOfBuffers( pad.device );
        global.numTextures               = tm->getNumberOfTextures( pad.device );
        global.numPrograms               = tm->getNumberOfPrograms( pad.device );
        global.numTraversables           = tm->getNumberOfTraversables( pad.device );
        global.launchDim                 = cort::uint3( width, height, depth );

        // NOTE: We colwert from int to unsigned int here.
        global.printIndex        = cort::uint3( printIndex.x, printIndex.y, printIndex.z );
        global.printEnabled      = m_context->getPrintEnabled();
        global.dimensionality    = dimensionality;
        global.entry             = entry;
        global.subframeIndex     = subframe_index;
        global.aabbRequest       = aabbRequest;
        global.activeDeviceIndex = launchDevices.getArrayPosition( allDeviceListIndex );
        global.activeDeviceCount = launchDevices.count();
        global.rayTypeCount      = m_context->getRayTypeCount();

        lwca::memcpyHtoDAsync( pad.const_Global, &global, sizeof( global ), stream );

        if( pad.const_ObjectRecord )
        {
            // TODO TableManager::getRecordSize loops over all objects to compute
            // actual size.  Consider caching this value to avoid the recomputation (either at
            // the planning stage or elsewhere).  Note you can't use the
            // ConstantMemAllocations::objectRecordSize, because this size may be larger than
            // the source pointer.
            lwca::memcpyDtoDAsync( pad.const_ObjectRecord, reinterpret_cast<LWdeviceptr>( objectRecordPtr ),
                                   global.objectRecordsSize, stream );
        }
        if( pad.const_BufferTable )
        {
            lwca::memcpyDtoDAsync( pad.const_BufferTable, reinterpret_cast<LWdeviceptr>( bufferHeaderPtr ),
                                   tm->getBufferHeaderTableSizeInBytes(), stream );
        }
        if( pad.const_ProgramTable )
        {
            lwca::memcpyDtoDAsync( pad.const_ProgramTable, reinterpret_cast<LWdeviceptr>( programHeaderPtr ),
                                   tm->getProgramHeaderTableSizeInBytes(), stream );
        }
        if( pad.const_TextureTable )
        {
            lwca::memcpyDtoDAsync( pad.const_TextureTable, reinterpret_cast<LWdeviceptr>( textureHeaderPtr ),
                                   tm->getTextureHeaderTableSizeInBytes(), stream );
        }
        if( pad.minMaxLaunchIndex && entry != m_context->getAabbEntry() )
        {
            lwca::memcpyHtoDAsync( pad.minMaxLaunchIndex, &minMaxLaunchIndex, sizeof( FrameTask::minMaxLaunchIndex ), stream );
        }
    }

    // Finalize outstanding transfers
    mm->finalizeTransfersBeforeLaunch();

    // SGP todo: subframe, lookup entry program

    // For Turing, we need to prime the TTU when returning from an idle state by opening a dummy TTU
    // transaction. Otherwise, the TTU can return false misses. This has been fixed in SM86+.
    // See: Bug 3147856, Turing HW Bug 2648362.
    if( k_gpuWarmupEnabled.get() )
    {
        optix_exp::ErrorDetails errDetails;
        if( m_context->getWatchdogManager()->launchWarmup( launchDevices, errDetails ) )
            throw prodlib::UnknownError( RT_EXCEPTION_INFO, errDetails.m_description );
    }

    if( dimensionality == 3 )
    {
        width = width * depth;
    }

    // First, we determine how much work can be split evenly across each GPU.
    // After that, we add the remaining work to each GPU.
    //
    // To determine the amount of work we can evenly divide across GPUs, we
    // divide the width of the launch by the number of devices * TILE_SIZE.
    // Note that we do this because we horizontally shift each tile based on its
    // vertical index and the device on which it will reside. This creates a
    // checkerboarded pattern that distributes work more evenly across GPUs.
    //
    // For example, if we have tile width 8, 3 GPUs, and a launch width of 28:
    //
    // Vertical shift region
    //     |       Leftover work
    //     |         |
    //     V         V
    // |-----------|---|
    // -----------------
    // | 8 | 8 | 8 | 4 |
    // -----------------
    //
    // We determine that each GPU can take a single tile, since we have a single
    // horizontal shift region:
    // GPU 0   GPU 1   GPU 2
    // -----   -----   -----
    // | 8 |   | 8 |   | 8 |
    // -----   -----   -----
    //
    // Then, we see that there are 4 pixels left, (or half of a tile). To determine
    // the amount of work to add to each launch, we take the min of TILE_SIZE and
    // the leftover work. We do this because we actually cover the area of a full
    // horizontal shift region when we add TILE_SIZE to each launch, so don't need to add
    // more than a single tile width to each launch to cover any leftover work.
    //
    // In our example, we end up with 3 12 pixel launches:
    // GPU 0       GPU 1       GPU 2
    // ---------   ---------   ---------
    // | 8 | 4 |   | 8 | 4 |   | 8 | 4 |
    // ---------   ---------   ---------
    //
    // If our example was slightly larger, say 45 pixels, it would look like this:
    // Horizontal shift region
    //     |       Leftover work
    //     |         |
    //     V         V
    // |-----------|-----------|
    // -------------------------
    // | 8 | 8 | 8 | 8 | 8 | 5 |
    // -------------------------
    //
    // After taking the min of the leftover work (which is 21 pixels), and our TILE_SIZE
    // (8 pixels), we would then add a single tile to each launch, like so:
    //
    // GPU 0       GPU 1       GPU 2
    // ---------   ---------   ---------
    // | 8 | 8 |   | 8 | 8 |   | 8 | 8 |
    // ---------   ---------   ---------
    //
    // At that point, our launches cover the entire scene (or a little bit more, due
    // to the horizontal shift). We might have overlaunched by a small amount, but a
    // bounds-check function will cause extra threads to exit before they perform
    // any work.
    //
    // The implementation of the tiling algorithm can be found in getTiledLaunchIndex
    // in RTXRuntime.cpp. The early-out function can be found in RTX_indexIsOutsideOfLaunch,
    // which is also in RTXRuntime.cpp.

    // Found in RTXRuntime.cpp in RTX_getLaunchIndex
    const RTsize TILE_WIDTH = 64;

    const RTsize leftoverWork    = width % ( TILE_WIDTH * launchDevices.count() );
    const RTsize evenlyDivisible = width - leftoverWork;
    const RTsize padding         = std::min( leftoverWork, TILE_WIDTH );
    const RTsize launchWidth     = evenlyDivisible / launchDevices.count() + padding;
    const RTsize launchDepth     = 1;

    static unsigned int launchCounter = 0;
    launchCounter++;

    for( int allDeviceListIndex : launchDevices )
    {
        PerLaunchDevice& pad = m_perLaunchDevice[m_devices.getArrayPosition( allDeviceListIndex )];

        if( pad.traversableGraphDepth != m_traversableGraphDepth )
        {
            // Update the pipeline stack size
            pad.traversableGraphDepth = m_traversableGraphDepth;
            m_context->getRTCore()->pipelineSetStackSize( pad.pipeline, pad.directCallableStackSizeFromTraversal,
                                                          pad.directCallableStackSizeFromState,
                                                          pad.continuationStackSize, pad.traversableGraphDepth );
        }

        int sbtEntrySize = m_context->getSBTManager()->getSBTRecordSize();

        RtcGpuVA raygenSbtVA =
            reinterpret_cast<RtcGpuVA>( m_context->getSBTManager()->getRaygenSBTRecordDevicePtr( pad.device, entry ) );
        RtcGpuVA exceptionSbtRecordVA =
            reinterpret_cast<RtcGpuVA>( m_context->getSBTManager()->getExceptionSBTRecordDevicePtr( pad.device, entry ) );
        RtcGpuVA firstMissSbtVA =
            reinterpret_cast<RtcGpuVA>( m_context->getSBTManager()->getMissSBTRecordDevicePtr( pad.device ) );
        RtcGpuVA firstInstanceSbtVA =
            reinterpret_cast<RtcGpuVA>( m_context->getSBTManager()->getInstancesSBTRecordDevicePtr( pad.device ) );
        RtcGpuVA firstCallableProgramSbtVA =
            reinterpret_cast<RtcGpuVA>( m_context->getSBTManager()->getCallableProgramSBTRecordDevicePtr( pad.device ) );

        if( pad.toolsOutputVA )
        {
            // Clear the profile data buffer before each kernel launch
            corelib::lwdaDriver().LwMemsetD8( pad.toolsOutputVA, 0, pad.profileMetadata.outputDataSizeInBytes );
        }

        pad.device->makeLwrrent();  // TODO(jbigler) WAR for rtcore not setting the context before doing some API calls.

        if( !k_gatherLwptiMetrics.get().empty() )
        {
            optix_exp::ErrorDetails lwptiErrDetails;
            if( OptixResult res = pad.device->getLwptiProfiler().beginProfile( lwptiErrDetails ) )
                throw prodlib::UnknownError( RT_EXCEPTION_INFO, lwptiErrDetails.m_description );
        }

        m_context->getRTCore()->launch3D( rtxres->getRtcCommandList( allDeviceListIndex ), pad.pipeline, pad.launchbuf,
                                          pad.scratchbuf, raygenSbtVA, exceptionSbtRecordVA, firstMissSbtVA,
                                          sbtEntrySize, 0, firstInstanceSbtVA, sbtEntrySize, 0,
                                          firstCallableProgramSbtVA, sbtEntrySize, 0, pad.toolsOutputVA,
                                          pad.toolsOutputSize, pad.scratchbytes, launchWidth, height, launchDepth );

        if( !k_gatherLwptiMetrics.get().empty() )
        {
            optix_exp::ErrorDetails lwptiErrDetails;
            if( OptixResult res = pad.device->getLwptiProfiler().endProfile( lwptiErrDetails ) )
                throw prodlib::UnknownError( RT_EXCEPTION_INFO, lwptiErrDetails.m_description );
        }

        // Dump profiling data if this feature is enabled
        if( k_enableRTcoreProfiling.get() && pad.toolsOutputVA && launchCounter >= k_rtcoreProfilingDumpCounter.get() )
        {
            // Add the device id to the dump file name since the profiling is per GPU
            std::stringstream profileDataFilename;
            profileDataFilename << "rtcore_profile_gpu" << pad.device->lwdaOrdinal() << ".txt";

            FILE* dumpfile = fopen( profileDataFilename.str().c_str(), "a+" );
            if( dumpfile )
            {
                RtcResult res;

                RtcPipelineInfoShaderInfo shaderInfo;
                m_context->getRTCore()->pipelineGetInfo( pad.pipeline, RTC_PIPELINE_INFO_TYPE_SHADER_INFO,
                                                         sizeof( RtcPipelineInfoShaderInfo ), &shaderInfo, &res );

                // Copy profiling data from gpu to host
                corelib::lwdaDriver().LwMemcpyDtoH( pad.toolsOutputBuffer.get(), pad.toolsOutputVA,
                                                    pad.profileMetadata.outputDataSizeInBytes );

                llog( 15 ) << "dumping profile data..." << std::endl;
                corelib::dumpRTcoreProfileData( dumpfile, &pad.profileMetadata, &shaderInfo,
                                                (const char*)pad.toolsOutputBuffer.get() );

                fclose( dumpfile );
            }
            else
            {
                lerr << "Error: Could not open file \"" << profileDataFilename.str()
                     << "\" to dump rtcore profiling data" << std::endl;
            }
        }
    }

    if( launchCounter >= k_rtcoreProfilingDumpCounter.get() )
        launchCounter = 0;


    waiter->recordStop();

    // Make sure work on the sync stream can't continue until the work queued up by the launch completes.
    if( syncStream.get() != nullptr )
        waiter->syncStream( syncStream );

    // Queue a callback to the context to do book-keeping when the launch is done, but not for
    // AABB launces.
    // We queue after the sync stream, because a callback will pause the stream. In order to
    // allow a possible interop process to move forward while the internal stream is pause,
    // we queue after the syncStream event is recorded, and not before
    if( !aabbRequest.aabbOutputPointer )
    {
        // Because the waiter->recordStop already synced all devices, we can trigger only in the first one.
        Callback_data* cbData = new Callback_data;
        cbData->m_waitHandle  = waitHandle;
        cbData->m_context     = m_context;
        corelib::lwdaDriver().LwLaunchHostFunc( rtxres->getStream( launchDevices[0] ).get(), waitHandleCallback, cbData );
    }
}

std::shared_ptr<WaitHandle> optix::RTXFrameTask::acquireWaitHandle( std::shared_ptr<LaunchResources>& launchResources )
{
    return std::shared_ptr<RTXWaitHandle>( new RTXWaitHandle( launchResources, m_context ) );
}

const RtcPipeline RTXFrameTask::getRtcPipeline( const unsigned int allDeviceListIndex ) const
{
    return m_perLaunchDevice[m_devices.getArrayPosition( allDeviceListIndex )].pipeline;
}

void RTXFrameTask::initializePadPointers( struct PerLaunchDevice& pad, const RTXLaunchResources& res, const unsigned int allDeviceListIndex )
{
    pad.statusDevicePtr = res.m_deviceFrameStatus[res.getDevices().getArrayPosition( allDeviceListIndex )];
    pad.launchbuf       = res.m_launchBuffers[res.getDevices().getArrayPosition( allDeviceListIndex )].first;
    pad.scratchbuf      = res.m_scratchBuffers[res.getDevices().getArrayPosition( allDeviceListIndex )].first;
    pad.scratchbytes    = res.m_scratchBuffers[res.getDevices().getArrayPosition( allDeviceListIndex )].second;

    PerDeviceOffsets pdo = m_perDeviceOffsets[m_devices.getArrayPosition( allDeviceListIndex )];

    pad.minMaxLaunchIndex = ( pdo.minMaxLaunchIndex == PerDeviceOffsets::INVALID ) ? 0 : pdo.minMaxLaunchIndex + pad.launchbuf;
    pad.const_Global = ( pdo.const_Global == PerDeviceOffsets::INVALID ) ? 0 : pdo.const_Global + pad.launchbuf;
    pad.const_ObjectRecord = ( pdo.const_ObjectRecord == PerDeviceOffsets::INVALID ) ? 0 : pdo.const_ObjectRecord + pad.launchbuf;
    pad.const_BufferTable = ( pdo.const_BufferTable == PerDeviceOffsets::INVALID ) ? 0 : pdo.const_BufferTable + pad.launchbuf;
    pad.const_ProgramTable = ( pdo.const_ProgramTable == PerDeviceOffsets::INVALID ) ? 0 : pdo.const_ProgramTable + pad.launchbuf;
    pad.const_TextureTable = ( pdo.const_TextureTable == PerDeviceOffsets::INVALID ) ? 0 : pdo.const_TextureTable + pad.launchbuf;
}

void RTXFrameTask::setTraversableGraphDepth( const unsigned int traversableGraphDepth )
{
    if( k_traversableGraphDepth.isDefault() )
        m_traversableGraphDepth = traversableGraphDepth;
    else
        m_traversableGraphDepth = k_traversableGraphDepth.get();
}

void RTXFrameTask::setDeviceInfo( LWDADevice*                   device,
                                  RtcPipeline                   pipeline,
                                  const ConstantMemAllocations& constMemAllocs,
                                  const bool                    isAabbLaunch,
                                  const unsigned int            directCallableStackSizeFromTraversal,
                                  const unsigned int            directCallableStackSizeFromState,
                                  const unsigned int            continuationStackSize )
{
    RT_ASSERT( m_devices.isSet( device ) );

    PerLaunchDevice& pad = m_perLaunchDevice[m_devices.getArrayPosition( device->allDeviceListIndex() )];
    pad.device           = device;
    pad.pipeline         = pipeline;
    pad.globalStruct.reset( new cort::Global() );
    pad.directCallableStackSizeFromTraversal = directCallableStackSizeFromTraversal;
    pad.directCallableStackSizeFromState     = directCallableStackSizeFromState;
    pad.continuationStackSize                = continuationStackSize;
    pad.traversableGraphDepth = ~0u;  // set as uninitialized. this triggers a call to pipelineSetStackSize on the next launch.

    bool enableProfiling = k_enableRTcoreProfiling.get();

    // Allocate device memory for rtcore profiling if enabled
    if( enableProfiling )
    {
        RtcResult result = RTC_SUCCESS;
        m_context->getRTCore()->pipelineGetInfo( pad.pipeline, RTC_PIPELINE_INFO_TYPE_PROFILING_METADATA,
                                                 sizeof( pad.profileMetadata ), &pad.profileMetadata, &result );
        if( result == RTC_SUCCESS )
        {
            llog( 10 ) << "profiling metadata size=" << pad.profileMetadata.outputDataSizeInBytes << std::endl;

            // Free the buffer if it has already been allocated
            if( pad.toolsOutputVA != 0 )
            {
                corelib::lwdaDriver().LwMemFree( pad.toolsOutputVA );
            }

            RT_ASSERT( pad.toolsOutputVA == 0 );

            // Allocate GPU memory for the profiling data
            corelib::lwdaDriver().LwMemAlloc( &pad.toolsOutputVA, pad.profileMetadata.outputDataSizeInBytes );
            corelib::lwdaDriver().LwMemsetD8( pad.toolsOutputVA, 0, pad.profileMetadata.outputDataSizeInBytes );
            pad.toolsOutputSize = pad.profileMetadata.outputDataSizeInBytes;

            // Allocate the same amount of host memory
            pad.toolsOutputBuffer = std::unique_ptr<char[]>{new char[pad.profileMetadata.outputDataSizeInBytes]};
        }
        else
        {
            lerr << "Error could not get rtcore profiling metadata info: " << result << std::endl;
        }
    }

    // Store offsets in the corresponding pdo. They will be used to construct the device pointer when the launchbuf pointer is available (from the resource)
    PerDeviceOffsets& pdo = m_perDeviceOffsets[m_devices.getArrayPosition( device->allDeviceListIndex() )];

    if( !k_limitActiveLaunchIndices.isDefault() && !isAabbLaunch )
    {
        Rtlw64 size   = 0;
        Rtlw64 offset = 0;
        m_context->getRTCore()->pipelineGetNamedConstantInfo( pad.pipeline, "const_MinMaxLaunchIndex", &offset, &size );
        pdo.minMaxLaunchIndex = offset;
        RT_ASSERT_MSG( size == sizeof( FrameTask::minMaxLaunchIndex ),
                       "Size of RTX::minMaxLaunchIndex does not match." );
    }

    Rtlw64 size   = 0;
    Rtlw64 offset = 0;
    m_context->getRTCore()->pipelineGetNamedConstantInfo( pad.pipeline, "const_Global", &offset, &size );
    pdo.const_Global = offset;
    llog( 10 ) << "const_Global size = " << size << ", offset = " << offset << "\n";
    RT_ASSERT_MSG( size == constMemAllocs.structGlobalSize, "size of const_Global does not match pipeline" );

    if( constMemAllocs.objectRecordSize )
    {
        m_context->getRTCore()->pipelineGetNamedConstantInfo( pad.pipeline, "const_ObjectRecord", &offset, &size );
        pdo.const_ObjectRecord = offset;
        RT_ASSERT_MSG( size >= constMemAllocs.objectRecordSize, "size of const_ObjectRecord is too small" );
    }
    else
    {
        pdo.const_ObjectRecord = PerDeviceOffsets::INVALID;
    }
    if( constMemAllocs.bufferTableSize )
    {
        m_context->getRTCore()->pipelineGetNamedConstantInfo( pad.pipeline, "const_BufferHeaderTable", &offset, &size );
        pdo.const_BufferTable = offset;
        RT_ASSERT_MSG( size >= constMemAllocs.bufferTableSize, "size of const_BufferHeaderTable is too small" );
    }
    else
    {
        pdo.const_BufferTable = PerDeviceOffsets::INVALID;
    }
    if( constMemAllocs.programTableSize )
    {
        m_context->getRTCore()->pipelineGetNamedConstantInfo( pad.pipeline, "const_ProgramHeaderTable", &offset, &size );
        pdo.const_ProgramTable = offset;
        RT_ASSERT_MSG( size >= constMemAllocs.programTableSize, "size of const_ProgramHeaderTable is too small" );
    }
    else
    {
        pdo.const_ProgramTable = PerDeviceOffsets::INVALID;
    }
    if( constMemAllocs.textureTableSize )
    {
        m_context->getRTCore()->pipelineGetNamedConstantInfo( pad.pipeline, "const_TextureHeaderTable", &offset, &size );
        pdo.const_TextureTable = offset;
        RT_ASSERT_MSG( size >= constMemAllocs.textureTableSize, "size of const_TextureHeaderTable is too small" );
    }
    else
    {
        pdo.const_TextureTable = PerDeviceOffsets::INVALID;
    }
}

// -----------------------------------------------------------------------------
void RTXFrameTask::addCompiledModule( const CanonicalProgram* cp,
                                      SemanticType            stype,
                                      SemanticType            inheritedStype,
                                      optix::Device*          device,
                                      ModuleEntryRefPair&     compilerOutput )
{
    SBTManager::CompiledProgramKey key      = {cp->getID(), stype, inheritedStype, device->allDeviceListIndex()};
    auto                           inserted = m_perPlanCompiledProgramCache.emplace( key, compilerOutput );
    if( !inserted.second )
        if( inserted.first->second != compilerOutput )
        {
            std::ostringstream o;
            o << "Key collision between two compiled programs in m_perPlanCompiledProgramCache";
            const SBTManager::CompiledProgramKey& cache = inserted.first->first;
            o << "\tkey: CP            : " << cache.cpID << "\n";
            o << "\t   : stype         : " << semanticTypeToString( cache.stype ) << "\n";
            o << "\t   : inheritedStype: " << semanticTypeToString( cache.inheritedStype ) << "\n";
            o << "\t   : devID         : " << cache.allDeviceListIndex << "\n";
            o << "\told: entry ref     : " << inserted.first->second.second << "\n";
            o << "\tnew: entry ref     : " << compilerOutput.second << "\n";
            o << "\told: ptr           : " << inserted.first->second.first << "\n";
            o << "\tnew: ptr           : " << compilerOutput.first << "\n";
            throw prodlib::AssertionFailure( RT_EXCEPTION_INFO, o.str() );
        }
}

optix::lwca::Event RTXFrameTask::createEventForStream( optix::lwca::Stream stream, LWresult* returnResult ) const
{
    RT_ASSERT( stream.get() != nullptr );

    LWcontext    syncContext    = nullptr;
    unsigned int syncContextVer = 0;
    CHECK( corelib::lwdaDriver().LwStreamGetCtx( stream.get(), &syncContext ) );
    CHECK( corelib::lwdaDriver().LwCtxGetApiVersion( syncContext, &syncContextVer ) );

    LWcontext    lwrrentContext    = nullptr;
    unsigned int lwrrentContextVer = 0;
    CHECK( corelib::lwdaDriver().LwCtxGetLwrrent( &lwrrentContext ) );
    CHECK( corelib::lwdaDriver().LwCtxGetApiVersion( lwrrentContext, &lwrrentContextVer ) );

    // Sanity check, contexts needs to be from the same version
    RT_ASSERT( syncContextVer == lwrrentContextVer );

    // If the context of the sync stream differs from the current context, then switch context temporarily
    if( syncContext != lwrrentContext )
        CHECK( corelib::lwdaDriver().LwCtxSetLwrrent( syncContext ) );

    LWresult result;
    Event    ev = Event::create( LW_EVENT_DEFAULT, &result );

    // Restore the previous context
    if( syncContext != lwrrentContext )
        CHECK( corelib::lwdaDriver().LwCtxSetLwrrent( lwrrentContext ) );

    if( result != LWDA_SUCCESS )
        throw prodlib::LwdaError( RT_EXCEPTION_INFO, "LwEventCreate", result );

    return ev;
}
