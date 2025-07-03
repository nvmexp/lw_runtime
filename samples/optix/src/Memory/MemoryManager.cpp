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

#include <Memory/MemoryManager.h>

#include <LWCA/Memory.h>
#include <Context/Context.h>
#include <Device/CPUDevice.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Memory/DemandLoad/PagingService.h>
#include <Memory/MResources.h>
#include <Memory/PolicyDetails.h>
#include <Memory/ResourceManager.h>
#include <Objects/Buffer.h>
#include <Util/ContainerAlgorithm.h>
#include <Util/Metrics.h>
#include <Util/UsageReport.h>
#include <prodlib/misc/TimeViz.h>

#include <corelib/misc/String.h>
#include <prodlib/exceptions/IlwalidOperation.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/MemoryAllocationFailed.h>
#include <prodlib/exceptions/UnknownError.h>
#include <prodlib/misc/BufferFormats.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>
#include <src/Exceptions/AlreadyMapped.h>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
using namespace prodlib;
using namespace corelib;

// clang-format off
namespace {
  Knob<bool>        k_forceMultiGPU( RT_DSTRING("mem.forceMultiGPU"), false, RT_DSTRING("Force memory manager to use multiGPU code paths"));
  Knob<bool>        k_validateBeforeLaunch( RT_DSTRING("mem.validateBeforeLaunch"), false, RT_DSTRING("Validate memory manager allocated and valid bits before each launch"));
  Knob<int>         k_mmll( RT_DSTRING("mem.logLevel"), 30, RT_DSTRING("Log level used for MemoryManager logging"));
  Knob<int>         k_stateDumpLogLevel( RT_DSTRING("mem.stateDumpLogLevel"), -1, RT_DSTRING("Log level for MemoryManager state dumps. If on default, will use the same as mem.logLevel. Setting it individually is useful to get the state dumps only, e.g. for diffing."));
  Knob<size_t>      k_limitDeviceMemoryInMB( RT_DSTRING("mem.limitDeviceMemoryInMB"), 0, RT_DSTRING("Artifically limits (prereserve) available memory"));
  Knob<bool>        k_trackMaxMemUsage( RT_DSTRING("mem.trackMaxMemUsage"), false, RT_DSTRING("Track the maximum memory usage on each device and print them on exit"));
  Knob<size_t>      k_highWaterReserveInMB( RT_DSTRING("mem.highWaterReserveInMB"), 1536, RT_DSTRING("Memory manager will attempt to leave this many MB free on devices, e.g. by using P2P"));
  Knob<int>         k_p2pBufferMinSizeInBytes( RT_DSTRING("mem.p2pBufferMinSizeInBytes"), 1024*1024, RT_DSTRING("Minimum size in bytes for a buffer to be considered for P2P access"));
  Knob<std::string> k_p2pMode( RT_DSTRING("mem.p2pMode"), "mixed", RT_DSTRING("How to use peer-to-peer if available. Possible values: gmemOnly|texOnly|gmemFirst|texFirst|mixed"));
  Knob<bool>        k_p2pBvhAllowed( RT_DSTRING("mem.p2pBvhAllowed"), true, RT_DSTRING("Allow to put BVHs in P2P"));
  Knob<int>         k_maxRetryFailsBeforeOOM( RT_DSTRING("mem.maxRetryFailsBeforeOOM"), 50, RT_DSTRING("Number of retries of the outer satisfyDeferredAllocations loop before triggering an out-of-memory failure"));
  Knob<int>         k_maxAcquireFailsBeforeRetry( RT_DSTRING("mem.maxAcquireFailsBeforeRetry"), 20, RT_DSTRING("Number of individual acquire failures before triggering an overall retry"));
}
// clang-format on


namespace optix {

/****************************************************************
 *
 * Constructor and destructor
 *
 ****************************************************************/

MemoryManager::MemoryManager( Context* context )
    : m_context( context )
    , m_deviceManager( context->getDeviceManager() )
    , m_updateManager( context->getUpdateManager() )
{
    for( int i = 0; i < OPTIX_MAX_DEVICES; ++i )
    {
        m_statsTotalMem[i]   = 0;
        m_statsMaxUsedMem[i] = 0;
    }
    trackMaxMemoryUsage();

    // Initialize device sets. Host device is always set, even if it is
    // not active, but the others are based on the lwrrently active
    // devices.
    const DeviceArray& activeDevices = m_deviceManager->activeDevices();
    const DeviceArray& allDevices    = m_deviceManager->allDevices();
    DeviceSet          allLwda;
    for( LWDADevice* device : LWDADeviceArrayView( allDevices ) )
    {
        allLwda.insert( device );
    }

    m_hostDevices       = DeviceSet( m_deviceManager->cpuDevice() );
    m_allDevices        = DeviceSet( allDevices );
    m_activeDevices     = DeviceSet( activeDevices );
    m_allLwdaDevices    = allLwda;
    m_activeLwdaDevices = m_activeDevices & m_allLwdaDevices;

    // Initialize device data
    for( Device* activeDevice : activeDevices )
        initializePerDeviceInfo( activeDevice );

    // Create the resource manager only after initializing everything
    // else. Because it can trigger allocations (texheap) we do this in
    // 2 steps.
    m_resourceManager.reset( new ResourceManager( this, m_context ) );
    m_resourceManager->initialize();
}

MemoryManager::~MemoryManager()
{
}

void MemoryManager::shutdown()
{
    printMaxMemoryUsage();

    // Shut down resource manager including texheap
    m_resourceManager->shutdown();

    // Ensure that memory manager allocations have all been released
    RT_ASSERT_MSG( m_masterList.empty(), "Memory manager destroyed with outstanding allocations" );
}

/****************************************************************
 *
 * Allocate MBuffers and MTextureSamplers
 *
 ****************************************************************/

MBufferHandle MemoryManager::allocateMBuffer( const BufferDimensions& size, MBufferPolicy policy, MBufferListener* listener )
{
    // Allocation is valid on all devices.
    DeviceSet allDevices = ~DeviceSet();
    return allocateMBuffer( size, policy, allDevices, listener );
}

MBufferHandle MemoryManager::allocateMBuffer( const BufferDimensions& size,
                                              MBufferPolicy           policy,
                                              const DeviceSet&        allowedDevices,
                                              MBufferListener*        listener )
{
    // Create a bare MBuffer object with the provided size that is valid
    // only on the specified device. It is a unique pointer until moved
    // into a shared pointer when finalized below.
    std::unique_ptr<MBuffer> buf( new MBuffer( this, size, policy, allowedDevices ) );

    // Add the requested callback if any.
    buf->addListener( listener );

    // Complete the initialization of the MBuffer object
    return finishAllocateMBuffer( buf );
}

MBufferHandle MemoryManager::allocateMBuffer( const GfxInteropResource& gfxResource,
                                              Device*                   gfxInteropDevice,
                                              MBufferPolicy             policy,
                                              MBufferListener*          listener )
{
    // Create a bare MBuffer object with the graphics resource as the
    // source and/or sink of the data on the interopDevice. Forge a
    // temporary size which will be updated below when the resource is
    // mapped. Buf is a unique pointer until moved into a shared pointer
    // when finalized below.
    BufferDimensions         tempSize;
    std::unique_ptr<MBuffer> buf( new MBuffer( this, tempSize, policy, ~DeviceSet() ) );

    // Set up resources for interop
    m_resourceManager->setupGfxInteropResource( buf->m_resources.get(), gfxResource, gfxInteropDevice );

    // Add the requested callback if any.
    buf->addListener( listener );

    // Verify that the requested policy is compatible with gfx interop.
    const PolicyDetails& details = getPolicyDetails( buf->getPolicy() );
    if( details.interopMode != PolicyDetails::DIRECT && details.interopMode != PolicyDetails::INDIRECT )
        throw MemoryAllocationFailed( RT_EXCEPTION_INFO,
                                      "Illegal graphics interop for buffer with policy " + toString( buf->getPolicy() ) );

    // Get buffer size from the graphics resource only for array.
    // This allows GL buffer object to be 0-sized at this point
    if( gfxResource.isArray() )
    {
        buf->m_dims = m_resourceManager->queryGfxInteropResourceSize( buf->m_resources.get() );
    }
    else
    {
        buf->m_dims.setFormat( RT_FORMAT_UNKNOWN, 1 );
    }

    // Complete the initialization of the MBuffer object
    MBufferHandle mbuf = finishAllocateMBuffer( buf );

    // Register the resource.
    registerGfxInteropResource( mbuf );

    return mbuf;
}

MTextureSamplerHandle MemoryManager::attachMTextureSampler( const MBufferHandle&     backing,
                                                            const TextureDescriptor& texDesc,
                                                            MTextureSamplerListener* listener )
{
    const PolicyDetails& policy = getPolicyDetails( backing->getPolicy() );
    RT_ASSERT_MSG( policy.allowsAttachedTextureSamplers,
                   "Texture sampler attached to buffer that does not allow texturing" );

    // Create the object and attach it to the base
    MTextureSampler* tex = new MTextureSampler( backing, texDesc );
    if( listener )
        tex->addListener( listener );
    backing->m_attachedTextureSamplers.insert( tex );

    // Add to lists
    m_masterTexList.addItem( tex );

    // Greedily attempt to reserve a texture unit for this allocation
    // on each active LWCA device.  If it fails, we will just wait and
    // evaluate all allocations before launch.
    reserveHWTextureIfAvailable( tex, m_activeDevices, policy );

    // Sync if necessary
    lazySync( tex );

    // Create the shared pointer to call our freeTextureSampler() when all references are released.
    return MTextureSamplerHandle( tex, [this]( MTextureSampler* ts ) { this->freeMTextureSampler( ts ); } );
}


/****************************************************************
 *
 * Manual synchronization
 *
 ****************************************************************/

void MemoryManager::manualSynchronize( const MBufferHandle& bufhdl )
{
    MBuffer* buf = bufhdl.get();
    if( log::active( k_mmll.get() ) )
        log( buf, "explicit sync" );

    // Execute the pre-access hook, which will handle interop.
    preAccess( buf );

    // Allocate if necessary
    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
    acquireResourcesOnDevicesOrFail( buf, m_activeDevices, policy );

    // Trigger the sync
    synchronizeMBuffer( buf, true );

    // Execute the post-access hook
    postAccess( buf );
}

void MemoryManager::manualSynchronize( const MTextureSamplerHandle& texHandle, unsigned int allDeviceListIndex )
{
    MTextureSampler* tex = texHandle.get();
    if( log::active( k_mmll.get() ) )
    {
        llog( k_mmll.get() ) << "explicit texture sync";
    }

    const DeviceSet allowedDevices( allDeviceListIndex );
    synchronizeTexToBacking( tex, allowedDevices );
    m_deferredTexSyncList.removeItem( tex );
}


/****************************************************************
 *
 * Update methods
 *
 ****************************************************************/

void MemoryManager::changeSize( const MBufferHandle& bufhdl, const BufferDimensions& newDims, bool preserveContents )
{
    // Setup and log
    MBuffer*             buf    = bufhdl.get();
    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );

    RT_ASSERT_MSG( !buf->m_mappedToHost, "Cannot change size while buffer is mapped" );
    if( log::active( k_mmll.get() ) )
        log( buf, "changeSize, from: " + buf->m_dims.toString() + " to: " + newDims.toString() );

    // Bail out if nothing has changed.
    if( newDims == buf->m_dims )
        return;

    // Compute sizes in bytes
    const size_t oldSize = buf->m_dims.getTotalSizeInBytes();
    const size_t newSize = newDims.getTotalSizeInBytes();

    updateValidSet( buf, policy );

    if( policy.isBackingStore )
    {
        // Content preservation for backing stores is handled indirectly by copying
        // the client buffers to the host where necessary. The backing store itself
        // thus doesn't need to preserve contents.
        RT_ASSERT_MSG( !preserveContents, "Content-preserving resize not supported for backing stores" );

        // "Back up" the contents of the client buffers on the host, so it's safe
        // to free the backing store on the active devices. Make sure to mark
        // *only* the host copies as valid, so that we re-sync correctly later.
        for( MBuffer* client : buf->m_backingClientsList )
        {
            const PolicyDetails clientPolicy = getPolicyDetails( client->getPolicy() );
            acquireResourcesOnDevicesOrFail( client, m_hostDevices, clientPolicy );
            synchronizeToDevices( client, m_hostDevices, clientPolicy );
            setValidSet( client, m_hostDevices, clientPolicy );
            lazyAllocateAndSync( client, clientPolicy );
        }

        // Free the backing resource and update the size
        releaseResourcesOnDevices( buf, ~DeviceSet(), policy );

        // Backing stores never have valid bits set.
        RT_ASSERT( buf->m_validSet == DeviceSet() );
    }
    else
    {
        // We can fulfill a potential request to preserve the content of the
        // overlapping region if the old size and new size are both 1D.  Otherwise,
        // all contents are destroyed. Also skip the copy if the overlap is zero or
        // if there are no valid copies.

        const size_t copySize = std::min( oldSize, newSize );
        if( preserveContents && copySize > 0 && !buf->m_validSet.empty() )
        {
            RT_ASSERT_MSG( buf->m_dims.dimensionality() == 1 && newDims.dimensionality() == 1,
                           "Can only preserve content during resize for 1D buffers" );

            RT_ASSERT_MSG( !policy.discardHostMemoryOnUnmap,
                           "Cannot preserve content of buffers with RT_BUFFER_DISCARD_HOST_MEMORY set" );

            // Replace the host data with the same policy but a different
            // size. Will update buf->m_size.
            BufferDimensions oldDims = buf->getDimensions();
            replaceHostData( buf, policy, policy, newDims, copySize );

            // Release resources on all other devices. Use the old
            // buffer dimensions for the release operation, then restore
            // the new dimensions.
            buf->m_dims = oldDims;
            releaseResourcesOnDevices( buf, ~m_hostDevices, policy );
            buf->m_dims = newDims;
        }
        else
        {
            // Free all resources, including frozen ones, but not externally owned
            // ones. For externally owned resources, we assume the user knows what
            // he's doing when triggering a resize.
            const DeviceSet external = m_resourceManager->getLwdaMallocExternalSet( buf->m_resources.get() );
            releaseResourcesOnDevices( buf, ~external, policy );
        }
    }

    // Update the size
    buf->m_dims = newDims;

    // Sync at next launch if necessary
    lazyAllocateAndSync( buf, policy );
}

void MemoryManager::changeSize( const MBufferHandle&    bufhdl,
                                const BufferDimensions& newDims,
                                std::function<void( LWDADevice*, LWdeviceptr, LWdeviceptr )> copyFunc )
{
    // Setup and log
    MBuffer*             buf     = bufhdl.get();
    BufferDimensions     oldDims = buf->m_dims;
    const PolicyDetails& policy  = getPolicyDetails( buf->getPolicy() );

    RT_ASSERT_MSG( !buf->m_mappedToHost, "Cannot change size while buffer is mapped" );
    if( log::active( k_mmll.get() ) )
        log( buf, "Changing size from: " + oldDims.toString() + " to: " + newDims.toString()
                      + " with custom copy function" );

    // Bail out if nothing has changed.
    if( newDims == oldDims )
        return;

    DeviceManager* dm           = m_context->getDeviceManager();
    DeviceSet      allocatedSet = buf->m_allocatedSet;
    for( int allDeviceIdx : allocatedSet )
    {
        LWDADevice* lwdaDevice = deviceCast<LWDADevice>( dm->allDevices()[allDeviceIdx] );
        RT_ASSERT_MSG( lwdaDevice, "MemoryManager::changeSize with custom copy function only works with LWCA devices" );

        // Save the old pointer
        MAccess oldAccess = buf->getAccess( lwdaDevice );
        RT_ASSERT( oldAccess.getKind() == MAccess::LINEAR );
        LWdeviceptr oldPtr = (LWdeviceptr)oldAccess.getLinearPtr();

        // Swap temporary resources with buffer resources
        // (otherwise acquireResourcesOnDevices will complain about a double allocation)
        MResources tmpResources( buf );
        m_resourceManager->moveDeviceResources( &tmpResources, buf->m_resources.get(), lwdaDevice );

        // Acquire the new resources
        DeviceSet lwdaDeviceSet = DeviceSet( lwdaDevice );
        buf->m_allocatedSet -= lwdaDeviceSet;
        buf->m_dims = newDims;
        acquireResourcesOnDevicesOrFail( buf, lwdaDeviceSet, policy );

        // Get the pointer from the new access
        MAccess newAccess = buf->getAccess( lwdaDevice );
        RT_ASSERT( newAccess.getKind() == MAccess::LINEAR );
        LWdeviceptr newPtr = (LWdeviceptr)newAccess.getLinearPtr();

        // Copy the acceleration structure
        copyFunc( lwdaDevice, oldPtr, newPtr );

        // Free the old block
        buf->m_dims = oldDims;
        buf->setAccess( lwdaDevice, oldAccess );
        m_resourceManager->releaseResourcesOnDevices( &tmpResources, lwdaDeviceSet, policy );

        // Ilwoke callbacks to update traversables
        buf->m_dims = newDims;
        buf->setAccess( lwdaDevice, newAccess );
    }
}

void MemoryManager::changePolicy( const MBufferHandle& bufhdl, MBufferPolicy toPolicy )
{
    // Changing the policy is a delicate operation. We must revisit all
    // assumptions made on the buffer and update allocations and other
    // data structures accordingly. Note that changing policy while the
    // buffer is mapped is legal.

    // Setup and log
    MBuffer* buf = bufhdl.get();
    if( log::active( k_mmll.get() ) )
        log( buf, "changePolicy, from: " + toString( buf->getPolicy() ) + " to: " + toString( toPolicy ) );

    const PolicyDetails& oldPolicy = getPolicyDetails( buf->getPolicy() );
    const PolicyDetails& newPolicy = getPolicyDetails( toPolicy );

    // Throw an error message if the policy's cpu allocation strategy has
    // changed. This behaves poorly (throws a cryptic error) if it oclwrs, and
    // usually only oclwrs when multiple GPUs are in use. This error message
    // suggests a reasonable fix at the user level (unmapping the buffer before
    // performing the operation that triggered the policy change).
    if( oldPolicy.cpuAllocation != newPolicy.cpuAllocation && buf->m_mappedToHost )
        throw AlreadyMapped( RT_EXCEPTION_INFO, "nodegraph edit performed while buffer was mapped" );

    changePolicy( buf, oldPolicy, newPolicy );
}

MemoryManager::AcquireStatus MemoryManager::acquireResourcesOnDevices( MBuffer* buf, const DeviceSet& onDevices, const PolicyDetails& policy )
{
    if( log::active( k_mmll.get() ) )
        log( buf, "acquireResourcesOnDevices, onDevices: " + onDevices.toString() );

    const DeviceSet allowed  = actualAllowedSet( buf, policy );
    DeviceSet       allNeeds = ( onDevices & allowed ) - buf->m_allocatedSet;
    AcquireStatus   status   = 0;

    //
    // Determine host allocations - malloc and zerocopy.
    //
    const DeviceSet hostNeeds = m_hostDevices & allNeeds;
    if( !hostNeeds.empty() )
    {
        DeviceSet requests = hostNeeds;

        switch( policy.cpuAllocation )
        {
            case PolicyDetails::CPU_ZEROCOPY:
            {
                m_resourceManager->acquireZeroCopyOnDevices( buf->m_resources.get(), requests, policy );
            }
            break;

            case PolicyDetails::CPU_MALLOC:
            {
                m_resourceManager->acquireHostMallocOnDevices( buf->m_resources.get(), requests, policy );
            }
            break;

            case PolicyDetails::CPU_PREFER_SINGLE:
            {
                m_resourceManager->acquireHostSingleCopyOnDevices( buf->m_resources.get(), requests, policy );
                if( !requests.empty() )
                    m_resourceManager->acquireZeroCopyOnDevices( buf->m_resources.get(), requests, policy );
            }
            break;

            case PolicyDetails::CPU_NONE:
            {
                RT_ASSERT_FAIL();  // device type / policy mismatch
            }
            break;
                // Default case intentionally omitted
        }

        const DeviceSet successful = hostNeeds - requests;
        allNeeds -= successful;
        buf->m_allocatedSet |= successful;
    }

    //
    // Determine LWCA allocations
    //
    const DeviceSet lwdaNeeds = m_allLwdaDevices & allNeeds;
    if( !lwdaNeeds.empty() )
    {
        DeviceSet requests = lwdaNeeds;

        switch( policy.lwdaAllocation )
        {
            case PolicyDetails::LWDA_GLOBAL:
            {
                if( buf->m_p2pRequested )
                    m_resourceManager->acquireLwdaMallocP2POnDevices( buf->m_resources.get(), requests, policy );
                else
                    m_resourceManager->acquireLwdaMallocOnDevices( buf->m_resources.get(), requests, policy );
            }
            break;

            case PolicyDetails::LWDA_ZEROCOPY:
            {
                m_resourceManager->acquireZeroCopyOnDevices( buf->m_resources.get(), requests, policy );
            }
            break;

            case PolicyDetails::LWDA_PREFER_ARRAY:
            {
                if( buf->m_p2pRequested )
                {
                    m_resourceManager->acquireLwdaArrayP2POnDevices( buf->m_resources.get(), requests );
                }
                else
                {
                    // Oversubscribed texture will trigger the fallback path.
                    m_resourceManager->acquireLwdaArrayOnDevices( buf->m_resources.get(), requests );
                    if( !requests.empty() )
                        m_resourceManager->acquireLwdaMallocOnDevices( buf->m_resources.get(), requests, policy );
                }
            }
            break;

            case PolicyDetails::LWDA_PREFER_TEX_HEAP:
            {
                m_resourceManager->acquireTexHeapOnDevices( buf->m_resources.get(), requests );
                if( !requests.empty() )
                {
                    status |= AcquireUsedTexHeapFallback;
                    if( buf->m_p2pRequested )
                        m_resourceManager->acquireLwdaMallocP2POnDevices( buf->m_resources.get(), requests, policy );
                    else
                        m_resourceManager->acquireLwdaMallocOnDevices( buf->m_resources.get(), requests, policy );
                }
            }
            break;

            case PolicyDetails::LWDA_PREFER_SINGLE:
            {
                m_resourceManager->acquireLwdaSingleCopyOnDevices( buf->m_resources.get(), requests, policy );
                if( !requests.empty() )
                    m_resourceManager->acquireZeroCopyOnDevices( buf->m_resources.get(), requests, policy );
            }
            break;

            case PolicyDetails::LWDA_PREFER_SPARSE:
            {
                if( m_context->getPagingManager()->getLwrrentPagingMode() == PagingMode::LWDA_SPARSE_HYBRID
                    || m_context->getPagingManager()->getLwrrentPagingMode() == PagingMode::LWDA_SPARSE_HARDWARE )
                    m_resourceManager->acquireLwdaSparseArrayOnDevices( buf->m_resources.get(), requests );
                else
                    m_resourceManager->acquireDemandLoadArrayOnDevices( buf->m_resources.get(), requests );
                break;
            }

            case PolicyDetails::LWDA_DEMAND_LOAD:
                m_resourceManager->acquireDemandLoadOnDevices( buf->m_resources.get(), requests );
                break;

            case PolicyDetails::LWDA_DEMAND_LOAD_TILE_ARRAY:
                m_resourceManager->acquireDemandLoadTileArrayOnDevices( buf->m_resources.get(), requests );
                break;

            case PolicyDetails::LWDA_SPARSE_BACKING:
                m_resourceManager->acquireDemandLoadTileArraySparseOnDevices( buf->m_resources.get(), requests );
                break;

            case PolicyDetails::LWDA_NONE:
            {
                RT_ASSERT_FAIL();  // device type / policy mismatch
            }
            break;
                // Default case intentionally omitted
        }

        const DeviceSet successful = lwdaNeeds - requests;
        allNeeds -= successful;
        buf->m_allocatedSet |= successful;
    }

    if( !allNeeds.empty() )
    {
        status |= AcquireFailed;
        llog( k_mmll.get() ) << " - failed on devices: " << allNeeds.toString() << '\n';
    }

    return status;
}

void MemoryManager::acquireResourcesOnDevicesOrFail( MBuffer* buf, const DeviceSet& onDevices, const PolicyDetails& policy )
{
    const AcquireStatus status = acquireResourcesOnDevices( buf, onDevices, policy );
    if( status & AcquireFailed )
        throw MemoryAllocationFailed( RT_EXCEPTION_INFO, "Memory allocation failed" );
}

void MemoryManager::releaseResourcesOnDevices( MBuffer* buf, const DeviceSet& onDevices, const PolicyDetails& policy )
{
    // This function releases the resources on the specified devices. Note that
    // lazyAllocateAndSync() must be called if the buffer is to be used
    // afterward.

    updateValidSet( buf, policy );

    const DeviceSet toClear = onDevices & buf->m_allocatedSet;
    m_resourceManager->releaseResourcesOnDevices( buf->m_resources.get(), toClear, policy );
    buf->m_validSet -= onDevices;
    buf->m_allocatedSet -= onDevices;
    buf->m_frozenSet -= onDevices;

    // Registered graphics interop should be unregistered manually
    // Note that presence of gfx interop device in allocated set means that gfx resource is mapped
    // Gfx resource registration in LWCA is a separate thing
    if( policy.interopMode != PolicyDetails::NOINTEROP )
    {
        RT_ASSERT_MSG( buf->getGfxInteropResource().isOGL(), "DX interop is not supported" );
        Device* interopDevice = m_resourceManager->getGfxInteropDevice( buf->m_resources.get() );
        if( onDevices.isSet( interopDevice ) )
        {
            m_resourceManager->freeGfxInteropResource( buf->m_resources.get() );
        }
    }
}

void MemoryManager::abandonResourcesOnDevices( MBuffer* buf, const DeviceSet& devices, const DeviceSet& potentialTargets )
{
    // Setup and log
    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
    if( log::active( k_mmll.get() ) )
        log( buf, "abandon on devices: " + devices.toString() );

    // Automatically ilwalidate non-active devices on each launch.
    updateValidSet( buf, policy );

    // Ensure zero-copy registrations have a valid device as registrar.
    DeviceSet lwdaDevicesLeftActive = ( potentialTargets - devices ) & m_allLwdaDevices;
    if( !lwdaDevicesLeftActive.empty() )
    {
        int         deviceIndex  = *lwdaDevicesLeftActive.begin();
        LWDADevice* first_device = deviceCast<LWDADevice>( m_deviceManager->allDevices()[deviceIndex] );
        m_resourceManager->updateRegistrations( buf->m_resources.get(), first_device );
    }

    DeviceSet newValid = buf->m_validSet - devices;
    if( newValid.empty() && !buf->m_validSet.empty() )
    {
        // Determine another device to store the buffer before we remove this one.
        DeviceSet targets = ( actualAllowedSet( buf, policy ) - devices ) & potentialTargets;

        if( policy.discardHostMemoryOnUnmap )
        {
            targets -= m_hostDevices;
        }

        // If you hit this assert, it's likely because of a buffer that's allowed
        // only on a single device, and that device got deactivated. Clients that
        // use buffers on restriced device sets must ensure they get freed before
        // the last allowed device is deactivated. This is so we can keep the
        // valuable check below in place.
        RT_ASSERT_MSG( !targets.empty(), "Buffer abandoned but no other devices are allowed to hold a copy" );

        // Find the first device on which we can successfully allocate.
        int target = -1;
        for( int device : targets )
        {
            const AcquireStatus status = acquireResourcesOnDevices( buf, DeviceSet( device ), policy );
            if( ( status & AcquireFailed ) == 0 )
            {
                target = device;
                break;
            }
        }

        // Couldn't find another device for the buffer.
        if( target == -1 )
        {
            throw MemoryAllocationFailed( RT_EXCEPTION_INFO, "Out of memory" );
        }

        // Trigger copies.
        synchronizeToDevices( buf, DeviceSet( target ), policy );
        lazyAllocateAndSync( buf, policy );
    }

    // Free resources.
    releaseResourcesOnDevices( buf, devices, policy );
}

void MemoryManager::updateTextureDescriptor( const MTextureSamplerHandle& sampler, const TextureDescriptor& texDesc )
{
    // Anything with a hw texture unit is now out of sync
    sampler->m_syncedToBacking -= sampler->m_hasHwTextureReserved;
    sampler->m_texDesc = texDesc;
    lazySync( sampler.get() );
}

/****************************************************************
 *
 * Manage host mappings.
 *
 ****************************************************************/

char* MemoryManager::mapToHost( const MBufferHandle& bufhdl, MapMode mode )
{
    MBuffer* buf = bufhdl.get();
    // Trivial mapping
    if( !buf )
        return nullptr;

    // Policy checks are only enforced for external mapToHost, since it is also
    // used inside of the memory manager to manage some transfers.
    {
        const PolicyDetails& policy  = getPolicyDetails( buf->getPolicy() );
        const bool           wantsRd = mode == MAP_READ || mode == MAP_READ_WRITE;
        const bool           wantsWr = mode == MAP_WRITE_DISCARD || mode == MAP_READ_WRITE;
        if( wantsRd && !policy.allowsHostReadAccess() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Illegal mapping of buffer with policy "
                                                       + toString( buf->getPolicy() ) + " to host for read" );
        if( wantsWr && !policy.allowsHostWriteAccess() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Illegal mapping of buffer with policy "
                                                       + toString( buf->getPolicy() ) + " to host for write" );
    }

    return mapToHostInternal( buf, mode );
}

char* MemoryManager::mapToHostInternal( MBuffer* buf, MapMode mode )
{
    // Setup and log
    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
    if( log::active( k_mmll.get() ) )
        log( buf, "mapping to host, allocated: " + buf->m_allocatedSet.toString() + ", valid: " + buf->m_validSet.toString() );

    // Error checks
    RT_ASSERT_MSG( !buf->m_mappedToHost, "Allocation already mapped to host" );
    RT_ASSERT_MSG( policy.cpuAllocation != PolicyDetails::CPU_NONE,
                   "Trying to map buffer to host, but allocation is not allowed" );

    // Ensure that the host copy is allocated
    acquireResourcesOnDevicesOrFail( buf, m_hostDevices, policy );

    // Execute the pre-access hook, which will handle interop.
    preAccess( buf );

    // Ensure that there is a valid copy on the host, triggering copies
    // if needed. It is possible that there are no valid copies.
    updateValidSet( buf, policy );
    if( !buf->m_validSet.empty() && ( mode == MAP_READ || mode == MAP_READ_WRITE ) )
        synchronizeToDevices( buf, m_hostDevices, policy );

    // Validate the host and ilwalidate other copies if appropriate.
    if( mode == MAP_WRITE_DISCARD || mode == MAP_READ_WRITE )
    {
        setValidSet( buf, m_hostDevices, policy );
        lazyAllocateAndSync( buf, policy );
    }

    buf->m_mappedToHost = true;

    return getMappedToHostPtrInternal( buf );
}

char* MemoryManager::getMappedToHostPtr( const MBufferHandle& buf )
{
    return getMappedToHostPtrInternal( buf.get() );
}

char* MemoryManager::getMappedToHostPtrInternal( MBuffer* buf )
{
    RT_ASSERT_MSG( buf->m_mappedToHost, "Not mapped to host" );

    // Retrieve the pointer from MAccess as a colwenience to the caller.
    CPUDevice*    cpuDevice = m_deviceManager->cpuDevice();
    const MAccess memAccess = buf->getAccess( cpuDevice );
    if( buf->m_dims.mipLevelCount() > 1 )
        RT_ASSERT_MSG( memAccess.getKind() == MAccess::MULTI_PITCHED_LINEAR, "Wrong access kind for MIP buffer" );

    // assume continuous memory for all MIP levels
    // host interop with user owned memory per level may break this assumption
    if( memAccess.getKind() == MAccess::MULTI_PITCHED_LINEAR )
        return memAccess.getPitchedLinear( 0 ).ptr;
    return memAccess.getLinearPtr();
}

void MemoryManager::unmapFromHost( const MBufferHandle& bufhdl )
{
    // Setup and log
    MBuffer* buf = bufhdl.get();
    if( !buf )
        return;

    unmapFromHostInternal( buf );
}

bool MemoryManager::isMappedToHost( const MBufferHandle& bufhdl )
{
    // Setup and log
    MBuffer* buf = bufhdl.get();
    RT_ASSERT( buf );

    return buf->m_mappedToHost;
}

void MemoryManager::unmapFromHostInternal( MBuffer* buf )
{
    RT_ASSERT_MSG( buf->m_mappedToHost, "Allocation unmapped from host but not mapped" );
    if( log::active( k_mmll.get() ) )
        log( buf, "unmapped from host" );

    // Execute the post-access hook
    postAccess( buf );

    // Unmark
    buf->m_mappedToHost = false;

    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
    if( policy.discardHostMemoryOnUnmap )
    {
        acquireResourcesOnDevices( buf, m_activeDevices, policy );
        synchronizeMBuffer( buf, true );
        releaseResourcesOnDevices( buf, m_hostDevices, policy );
        // Could be removed from the lazy alloc & sync lists here
    }
}


/****************************************************************
 *
 * Pin/unpin a buffer on a device.
 *
 ****************************************************************/

// This ensures that the buffer will not be fiddled with between calls to pin/unpin.
// The buffer must have an allocation on the device, throws otherwise.
// The result of pinning is that the buffer gets set to valid on the specified device, and invalid everywhere else.

char* MemoryManager::pinToDevice( const MBufferHandle& bufhdl, Device* device, MapMode mode )
{
    MBuffer*             buf = bufhdl.get();
    const DeviceSet      pinnedDevice( device );
    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );

    if( log::active( k_mmll.get() ) )
        log( buf, "pinning to device: " + pinnedDevice.toString() + ", allocated: " + buf->m_allocatedSet.toString()
                      + ", valid: " + buf->m_validSet.toString() );

    RT_ASSERT_MSG( buf->m_pinnedSet.empty(), "MBuffer is already pinned" );
    RT_ASSERT_MSG( deviceCast<const LWDADevice>( device ) != nullptr, "Trying to pin to non-LWCA device" );

    preAccess( buf );

    RT_ASSERT_MSG( !( pinnedDevice & buf->m_allocatedSet ).empty(),
                   "Trying to pin buffer to device without allocation" );

    // Ensure that there is a valid copy on the device, triggering copies
    // if needed. It is possible that there are no valid copies.
    updateValidSet( buf, policy );
    if( !buf->m_validSet.empty() && ( mode == MAP_READ || mode == MAP_READ_WRITE ) )
        synchronizeToDevices( buf, pinnedDevice, policy );

    if( mode == MAP_WRITE_DISCARD || mode == MAP_READ_WRITE )
    {
        setValidSet( buf, pinnedDevice, policy );
        lazyAllocateAndSync( buf, policy );
    }

    buf->m_pinnedSet |= pinnedDevice;

    return buf->getDevicePtr( device );
}

void MemoryManager::unpinFromDevice( const MBufferHandle& bufhdl, Device* device )
{
    MBuffer*        buf = bufhdl.get();
    const DeviceSet pinnedDevice( device );
    if( log::active( k_mmll.get() ) )
        log( buf, "unpinned from device: " + pinnedDevice.toString() );

    RT_ASSERT_MSG( buf->m_pinnedSet.isSet( device ), "MBuffer is not pinned on specified device" );
    RT_ASSERT_MSG( deviceCast<const LWDADevice>( device ) != nullptr, "Trying to unpin from non-LWCA device" );

    postAccess( buf );

    buf->m_pinnedSet -= pinnedDevice;
}

/****************************************************************
 *
 * Special interface for builders that write to device memory (like
 * texheap). Valid only after syncAllMemoryBeforeLaunch and until
 * releaseMemoryAfterLaunch.
 *
 ****************************************************************/
char* MemoryManager::getWritablePointerForBuild( const MBufferHandle& bufhdl, Device* device, bool exclusiveMode )
{
    MBuffer* buf = bufhdl.get();
    if( log::active( k_mmll.get() ) )
        log( buf, "get writable pointer for build, valid: " + buf->m_validSet.toString() );
    DeviceSet buildDevice( device );
    RT_ASSERT_MSG( m_activeDevices.overlaps( buildDevice ), "Build destination is not valid" );
    if( exclusiveMode )
        RT_ASSERT_MSG( buf->m_validSet.overlaps( buildDevice ),
                       "Exclusive mode only allows getting pointer for a single device" );

    // Ilwalidate other device copies and mark the buffer for syncing to other devices
    buf->m_validSetTimestamp = m_lwrrentLaunchTimeStamp;

    if( exclusiveMode )
    {
        const PolicyDetails& policy   = getPolicyDetails( buf->getPolicy() );
        buf->m_validSet               = DeviceSet( device );
        const DeviceSet allowedActive = actualAllowedSet( buf, policy ) & m_activeDevices;
        const DeviceSet needs         = allowedActive - buf->m_validSet;
        if( policy.launchRequiresValidCopy && !needs.empty() )
            m_deferredSyncList.addItem( buf );
    }
    else
    {
        buf->m_validSet |= DeviceSet( device );
    }

    // Find the pointer, which might be the texheap backing
    MAccess access = buf->getAccess( device );
    if( access.getKind() == MAccess::LINEAR )
    {
        return access.getLinear().ptr;
    }
    else if( access.getKind() == MAccess::TEX_REFERENCE )
    {
        RT_ASSERT_MSG( exclusiveMode, "Texture heap can only be used in exclusive device mode" );
        return m_resourceManager->getTexHeapBackingPointer( device, access );
    }
    else
    {
        throw UnknownError( RT_EXCEPTION_INFO, "Invalid build destination" );
    }
}

/****************************************************************
 *
 * Graphics interop
 *
 ****************************************************************/

void MemoryManager::registerGfxInteropResource( const MBufferHandle& bufhdl )
{
    // Setup and log
    MBuffer* buf = bufhdl.get();
    if( log::active( k_mmll.get() ) )
        log( buf, "registerGraphicsResource" );
    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );

    // Policy checks
    if( policy.interopMode != PolicyDetails::DIRECT && policy.interopMode != PolicyDetails::INDIRECT )
        throw MemoryAllocationFailed( RT_EXCEPTION_INFO,
                                      "Illegal graphics interop for buffer with policy " + toString( buf->getPolicy() ) );

    // Register the resource
    m_resourceManager->registerGfxInteropResource( buf->m_resources.get() );

    // Set up hooks so that this buffer will get mapped before launch.
    addLaunchHooks( buf, policy );
}

void MemoryManager::unregisterGfxInteropResource( const MBufferHandle& bufhdl )
{
    // Setup and log
    MBuffer* buf = bufhdl.get();
    if( log::active( k_mmll.get() ) )
        log( buf, "unregisterGraphicsResource" );
    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );

    // Policy checks
    if( policy.interopMode != PolicyDetails::DIRECT && policy.interopMode != PolicyDetails::INDIRECT )
        throw MemoryAllocationFailed( RT_EXCEPTION_INFO,
                                      "Illegal graphics interop for buffer with policy " + toString( buf->getPolicy() ) );

    // Delegate the registration to the pool
    m_resourceManager->unregisterGfxInteropResource( buf->m_resources.get() );

    // Release texture (if any)
    for( MTextureSampler* tex : buf->m_attachedTextureSamplers )
    {
        releaseResourcesOnDevices( tex, ~DeviceSet(), policy );
        m_deferredTexSyncList.removeItem( tex );
    }

    // Release resources on all devices
    releaseResourcesOnDevices( buf, ~DeviceSet(), policy );

    // Remove launch hooks
    m_prelaunchList.removeItem( buf );
    m_postlaunchList.removeItem( buf );
}


/****************************************************************
 *
 * LWCA interop
 *
 ****************************************************************/

void MemoryManager::setLwdaInteropPointer( const MBufferHandle& bufhdl, void* ptr, Device* device )
{
    MBuffer* buf = bufhdl.get();
    if( log::active( k_mmll.get() ) )
        log( buf, "setLwdaInteropPointer" );

    RT_ASSERT( ptr != nullptr );
    RT_ASSERT( deviceCast<LWDADevice>( device ) != nullptr );

    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
    RT_ASSERT( policy.lwdaAllocation != PolicyDetails::LWDA_ZEROCOPY );  // shouldn't happen because set ptr is disallowed
                                                                         // on output buffers

    // Free potential previous resources
    const DeviceSet onDevices( device );
    releaseResourcesOnDevices( buf, onDevices, policy );

    // Acquire new external lwca malloc resource
    DeviceSet requests = onDevices;
    m_resourceManager->acquireLwdaMallocExternalOnDevices( buf->m_resources.get(), requests, (char*)ptr );
    RT_ASSERT_MSG( requests.empty(), "external lwca malloc failed" );

    buf->m_allocatedSet |= onDevices;
    buf->m_frozenSet |= onDevices;
}

void* MemoryManager::getLwdaInteropPointer( const MBufferHandle& bufhdl, Device* device )
{
    MBuffer* buf = bufhdl.get();
    if( log::active( k_mmll.get() ) )
        log( buf, "getLwdaInteropPointer" );

    RT_ASSERT( deviceCast<LWDADevice>( device ) != nullptr );

    // Ensure that the resource is allocated
    DeviceSet            onDevices( device );
    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
    acquireResourcesOnDevicesOrFail( buf, onDevices, policy );
    buf->m_frozenSet |= onDevices;

    return buf->getAccess( device ).getLinearPtr();
}

void MemoryManager::markDirtyLwdaInterop( const MBufferHandle& bufhdl )
{
    MBuffer* buf = bufhdl.get();
    if( log::active( k_mmll.get() ) )
        log( buf, "markDirty" );

    // Set valid set to *only* the frozen set.
    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
    setValidSet( buf, buf->m_frozenSet, policy );
    lazyAllocateAndSync( buf, policy );
}

DeviceSet MemoryManager::getLwdaInteropDevices( const MBufferHandle& bufhdl ) const
{
    MBuffer* buf = bufhdl.get();
    if( log::active( k_mmll.get() ) )
        log( buf, "getLwdaInteropDevices" );

    return buf->m_frozenSet;
}

/****************************************************************
 *
 * Context interface
 *
 ****************************************************************/

void MemoryManager::enterFromAPI()
{
    RT_ASSERT_MSG( !m_launching, "Memory manager launch is already active" );
    RT_ASSERT_MSG( !m_enteredFromAPI, "Memory manager already entered from API" );
    dump( "enterFromAPI" );

    // Execute the prelaunch tasks (interop buffers)
    if( !m_prelaunchList.empty() )
    {
        TIMEVIZ_SCOPE( "preAccess" );
        preAccess( m_prelaunchList.getList() );
    }

    m_enteredFromAPI = true;
}

void MemoryManager::exitToAPI()
{
    RT_ASSERT_MSG( !m_launching, "Memory manager launch is still active" );
    RT_ASSERT_MSG( m_enteredFromAPI, "Memory manager is exiting to API without first entering from it" );

    m_enteredFromAPI = false;

    // Execute the postlaunch tasks (interop buffers)
    if( !m_postlaunchList.empty() )
    {
        TIMEVIZ_SCOPE( "postAccess" );
        postAccess( m_postlaunchList.getList() );
    }

    // At some point, we may want to have policies that initiate memory
    // transfer immediately after launch.
}

void MemoryManager::syncAllMemoryBeforeLaunch()
{
    // Logging, debugging, stats
    TIMEVIZ_FUNC;
    RT_ASSERT_MSG( !m_launching, "Memory manager launched while launch active" );
    reserveMemoryForDebug();
    dump( "pre-sync" );

    // If we detected an out-of-texture situation (Fermi only or bindless
    // disabled) then reassign the limited texture units.
    // TODO: probably no longer necessary since we no longer support Fermi
    reevaluateArrayAssignments();

    // Satisfy allocations of both regular buffers and backing stores (texheap).
    satisfyDeferredAllocations();

    // Synchronize buffers that were marked for lazy synchronize and clear the list
    synchronizeBuffers();

    // Synchronize textures that were marked for lazy allocate/synchronize and clear the list
    synchronizeTextures();

    // Optionally verify that all buffers have been allocated/synced as appropriate
    if( k_validateBeforeLaunch.get() )
        verifyBeforeLaunch();

    // Set the launching flag to disallow further allocations until the
    // manager is unlocked.
    m_launching = true;

    // Logging, debugging, stats
    trackMaxMemoryUsage();
    reportMemoryUsage();
    dump( "post-sync" );
}

void MemoryManager::finalizeTransfersBeforeLaunch()
{
    // no-op because we don't have truly async memcopies yet
}

void MemoryManager::releaseMemoryAfterLaunch()
{
    llog( k_mmll.get() ) << "--- release after launch #" << m_lwrrentLaunchTimeStamp
                         << "\n    postLaunch: " << m_postlaunchList.size() << "\n";
    RT_ASSERT_MSG( m_launching, "Memory manager released while launch not active" );

    // Reset the launching state and increment time
    m_launching = false;
    m_lwrrentLaunchTimeStamp++;
}

RTsize MemoryManager::getUsedHostMemory() const
{
    RTsize     sizeOnHost = 0;
    CPUDevice* cpuDevice  = m_deviceManager->cpuDevice();

    for( MBuffer* buf : m_masterList )
    {
        const MAccess access = buf->getAccess( cpuDevice );
        if( access.getKind() == MAccess::NONE )
            continue;

        const BufferDimensions& dim = buf->getDimensions();
        sizeOnHost += dim.getTotalSizeInBytes();
    }
    return sizeOnHost;
}

/****************************************************************
 *
 * ExelwtionStrategy interface
 *
 ****************************************************************/

// Update the texture descriptor properties without reallocating the texture
const BitSet& MemoryManager::getAssignedTexReferences( Device* device ) const
{
    unsigned int idx = device->allDeviceListIndex();
    return m_pdi[idx].assignedTextureUnits;
}

void MemoryManager::bindTexReference( Device* device, unsigned int texUnit, lwca::TexRef texRef )
{
    unsigned int deviceIndex = device->allDeviceListIndex();
    LWDADevice*  lwdaDevice  = deviceCast<LWDADevice>( device );
    RT_ASSERT( lwdaDevice != nullptr );
    RT_ASSERT( m_activeLwdaDevices.isSet( device ) );
    RT_ASSERT( texUnit < m_pdi[deviceIndex].assignedBoundTextures.size() );
    RT_ASSERT( m_pdi[deviceIndex].assignedTextureUnits.isSet( texUnit ) );
    RT_ASSERT( m_pdi[deviceIndex].assignedBoundTextures[texUnit] != nullptr );


    const MTextureSampler* tex = m_pdi[deviceIndex].assignedBoundTextures[texUnit];
    if( log::active( k_mmll.get() ) )
        log( tex->m_backing.get(), "binding texture reference to device" );

    const MAccess texAccess = tex->getAccess( deviceIndex );
    RT_ASSERT( texAccess.getKind() == MAccess::TEX_REFERENCE );

    // Attach storage to texref
    m_resourceManager->bindTexReference( tex->m_backing->m_resources.get(), lwdaDevice, tex->m_texDesc, texRef );
}


/****************************************************************
 *
 * DeviceManager interface
 *
 ****************************************************************/
void MemoryManager::setActiveDevices( const DeviceSet& devices )
{
    // Switching devices while buffers are mapped to host is not allowed.
    for( MBuffer* buf : m_masterList )
    {
        if( buf->m_mappedToHost )
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Cannot change active devices while buffers are mapped." );
    }

    DeviceSet addedDevices   = devices - m_activeDevices;
    DeviceSet removedDevices = m_activeDevices - devices;

    // Remove texture sampler resources
    for( MTextureSampler* tex : m_masterTexList )
    {
        const PolicyDetails& policy = getPolicyDetails( tex->m_backing->getPolicy() );
        releaseResourcesOnDevices( tex, removedDevices, policy );
    }

    // Remove memory resources, syncing to other devices if necessary
    for( MBuffer* buf : m_masterList )
    {
        abandonResourcesOnDevices( buf, removedDevices, devices );

        // Interop device may change
        const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
        if( policy.interopMode != PolicyDetails::NOINTEROP )
        {
            const GfxInteropResource& gfxResource = buf->getGfxInteropResource();
            RT_ASSERT_MSG( buf->getGfxInteropResource().isOGL(), "DX interop is not supported" );
            Device* newInteropDevice = m_deviceManager->glInteropDevice();
            Device* oldInteropDevice = m_resourceManager->getGfxInteropDevice( buf->m_resources.get() );

            if( oldInteropDevice != newInteropDevice )
            {
                m_resourceManager->freeGfxInteropResource( buf->m_resources.get() );
                m_resourceManager->setupGfxInteropResource( buf->m_resources.get(), gfxResource, newInteropDevice );
            }
        }
    }

    // Inform the ResourceManager about the removed devices so that it can clean any resources it owns, such as
    // bulk memory allocation pools
    m_resourceManager->removedDevices( removedDevices );

    // Update the device sets
    unsigned int oldVariant = computeLwrrentPolicyVariant();
    m_activeDevices         = DeviceSet( devices );
    m_activeLwdaDevices     = m_activeDevices & m_allLwdaDevices;

    // Reinitialize device information
    const DeviceArray& allDevices = m_deviceManager->allDevices();
    for( DeviceSet::position deviceIndex : removedDevices )
        resetPerDeviceInfo( allDevices[deviceIndex] );
    for( DeviceSet::position deviceIndex : addedDevices )
        initializePerDeviceInfo( allDevices[deviceIndex] );

    // Update policies if necessary
    updatePoliciesAfterVariantChange( oldVariant );

    // Reserve textures on new devices
    for( MTextureSampler* tex : m_masterTexList )
    {
        const PolicyDetails& policy = getPolicyDetails( tex->m_backing->getPolicy() );
        reserveHWTextureIfAvailable( tex, addedDevices, policy );
    }

    // Resync buffers and texture samplers
    lazyResyncAll();
}

/****************************************************************
 *
 * Allocation properties
 *
 ****************************************************************/

void MemoryManager::determineMixedAccess( const MTextureSamplerHandle& tex, bool& mixedAccess, bool& mixedPointer, MAccess& access )
{
    if( !tex )
    {
        mixedAccess  = false;
        mixedPointer = false;
        access       = MAccess::makeNone();
        return;
    }
    // Determine if the texture sampler uses the same or different
    // access kinds / pointers.
    for( DeviceSet::position deviceIndex : m_activeDevices )
    {
        const MAccess memAccess = tex->getAccess( deviceIndex );
        if( memAccess.getKind() == MAccess::NONE )
        {
            continue;
        }
        else if( access.getKind() != MAccess::NONE && access.getKind() != memAccess.getKind() )
        {
            mixedAccess = mixedPointer = true;
            return;
        }
        else if( access.getKind() != MAccess::NONE && access != memAccess )
        {
            mixedPointer = true;
        }
        else
        {
            access = memAccess;
        }
    }
}

void MemoryManager::determineMixedAccess( const MBufferHandle& buf, bool& mixedAccess, bool& mixedPointer, MAccess& access )
{
    if( !buf )
    {
        mixedAccess  = false;
        mixedPointer = false;
        access       = MAccess::makeNone();
        return;
    }
    // Determine if the buffer uses the same or different access kinds /
    // pointers.
    for( DeviceSet::position deviceIndex : m_activeDevices )
    {
        const MAccess memAccess = buf->getAccess( deviceIndex );
        if( memAccess.getKind() == MAccess::NONE )
        {
            continue;
        }
        else if( access.getKind() != MAccess::NONE && access.getKind() != memAccess.getKind() )
        {
            mixedAccess = mixedPointer = true;
            return;
        }
        else if( access.getKind() != MAccess::NONE && access != memAccess )
        {
            mixedPointer = true;
        }
        else
        {
            access = memAccess;
        }
    }
}


/****************************************************************
 *
 * Allocate and free
 *
 ****************************************************************/

MBufferHandle MemoryManager::finishAllocateMBuffer( std::unique_ptr<MBuffer>& buf )
{
    RT_ASSERT_MSG( !m_launching, "Illegal allocation during launch" );
    RT_ASSERT_MSG( !buf->m_allowedSet.empty(), "Allocation not allowed on any device" );

    // Assign serial number - used for debug logs.
    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
    buf->m_serialNumber         = m_nextSerialNumber++;
    if( log::active( k_mmll.get() ) )
        log( buf.get(), "finalizeAllocation, allocatedSet: " + buf->m_allocatedSet.toString() + " allowedSet: "
                            + buf->m_allowedSet.toString() );
    TIMEVIZ_COUNT( "MBuffers", m_masterList.size() );

    // Assign sync time which will be used to automatically ilwalidate
    // device-written data after launch.
    buf->m_validSetTimestamp = m_lwrrentLaunchTimeStamp;

    // Add to any required lists
    m_masterList.addItem( buf.get() );

    // Allocate and sync before launch
    lazyAllocateAndSync( buf.get(), policy );

    // Create the shared pointer that will notify us when all references are released.  Note
    // that we need to swallow exceptions, since this is called from a destructor which
    // could be called when unwinding the stack from another exception.
    return MBufferHandle( buf.release(), [this]( MBuffer* mb ) {
        try
        {
            this->freeMBuffer( mb );
        }
        catch( ... )
        {
            lerr << "Error in freeing MBuffer... swallowing it since it was called from a smart pointer destructor\n";
        }
    } );
}

void MemoryManager::freeMBuffer( MBuffer* buf )
{
    // Setup and log
    if( log::active( k_mmll.get() ) )
        log( buf, "freeMBuffer, allocatedSet: " + buf->m_allocatedSet.toString() + " activeSet: " + m_activeDevices.toString() );
    RT_ASSERT_MSG( !buf->m_mappedToHost, "Buffer freed while being mapped to host" );

    // Remove from lists
    m_deferredAllocList.removeItem( buf );
    m_deferredBackingAllocList.removeItem( buf );
    m_deferredSyncList.removeItem( buf );
    m_masterList.removeItem( buf );
    m_prelaunchList.removeItem( buf );
    m_postlaunchList.removeItem( buf );

    TIMEVIZ_COUNT( "MBuffers", m_masterList.size() );

    // Free resources
    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
    releaseResourcesOnDevices( buf, ~DeviceSet(), policy );
    delete buf;

    // Give backing stores a chance to shrink
    resizeBackingStores();
}

void MemoryManager::freeMTextureSampler( MTextureSampler* tex )
{
    const PolicyDetails& policy = getPolicyDetails( tex->m_backing->getPolicy() );

    // Remove from lists
    m_masterTexList.removeItem( tex );
    m_deferredTexSyncList.removeItem( tex );

    const size_t nerased = tex->m_backing->m_attachedTextureSamplers.erase( tex );
    RT_ASSERT( nerased == 1 );

    // Free resources
    releaseResourcesOnDevices( tex, ~DeviceSet(), policy );
    delete tex;
}


/****************************************************************
 *
 * Valid set management
 *
 ****************************************************************/

void MemoryManager::updateValidSet( MBuffer* buf, const PolicyDetails& policy )
{
    if( buf->m_validSetTimestamp >= m_lwrrentLaunchTimeStamp )
        return;

    // Update the valid set and update the timestamp
    if( policy.launchIlwalidatesOtherDevices )
        buf->m_validSet -= ~m_activeDevices;
    if( policy.allowsActiveDeviceWriteAccess() )
        buf->m_validSet |= ( buf->m_allowedSet & m_activeDevices );

    buf->m_validSetTimestamp = m_lwrrentLaunchTimeStamp;
}

void MemoryManager::setValidSet( MBuffer* buf, const DeviceSet& validSet, const PolicyDetails& policy )
{
    // Clear any pending lazy updates.
    updateValidSet( buf, policy );

    // Set the new valid set, ilwalidating any other copies.
    buf->m_validSet = validSet;

    // In the case of zero copy, if we validate any copy then all other
    // copies will also be valid.
    m_resourceManager->expandValidSet( buf->m_resources.get(), buf->m_validSet );
}


/****************************************************************
 *
 * High-level resource management
 *
 ****************************************************************/

void MemoryManager::changePolicy( MBuffer* buf, const PolicyDetails& oldPolicy, const PolicyDetails& newPolicy )
{
    updateValidSet( buf, oldPolicy );

    // Note that order is important here. We need to change first the policy in LWCA, then on the CPU,
    // because a LWCA change may trigger a copy to the host for synchronization, and it will otherwise
    // be a potential policy mismatch.
    //
    // TODO: It is possible that we can save one copy by going directly to the new CPU policy for certain cases.

    //
    // Update the LWCA allocation if necessary
    //
    const bool preserveContents = !buf->m_validSet.empty() && buf->m_frozenSet.empty();
    // BL: graphics interop here?

    if( oldPolicy.lwdaAllocation != newPolicy.lwdaAllocation )
    {
        // Release texture (if any)
        for( MTextureSampler* tex : buf->m_attachedTextureSamplers )
            releaseResourcesOnDevices( tex, ~DeviceSet(), oldPolicy );

        // Re-acquire texture with the new policy
        if( newPolicy.policyKind != MBufferPolicy::unused && newPolicy.allowsAttachedTextureSamplers )
        {
            // workaround for test_AllTextureModes, when we change texture buffer we get newPolicy == MBufferPolicy_unused
            // there is OP-707 task to figure this out
            RT_ASSERT( buf->m_attachedTextureSamplers.empty() );
            for( MTextureSampler* tex : buf->m_attachedTextureSamplers )
                reserveHWTextureIfAvailable( tex, m_activeDevices, newPolicy );
        }

        if( !preserveContents || !changePolicyLwdaAllocationWithinDevice( buf, oldPolicy, newPolicy ) )
        {
            // If necessary preserve data by syncing to host
            if( preserveContents )
            {
                // Ensure that the host copy is allocated
                acquireResourcesOnDevicesOrFail( buf, m_hostDevices, oldPolicy );

                // Sync to host
                synchronizeToDevices( buf, m_hostDevices, oldPolicy );
            }

            // Release resources on other devices
            DeviceSet otherDevices = ~m_hostDevices;
            releaseResourcesOnDevices( buf, otherDevices, oldPolicy );

            // Set the valid devices to only the host (if it was valid
            // already)
            setValidSet( buf, buf->m_validSet & m_hostDevices, oldPolicy );

            // Synchronize buffer and discard host memory if requested.
            if( preserveContents && newPolicy.discardHostMemoryOnUnmap )
            {
                acquireResourcesOnDevicesOrFail( buf, m_activeDevices, newPolicy );
                synchronizeMBuffer( buf, true );
                releaseResourcesOnDevices( buf, m_hostDevices, newPolicy );
            }
        }
    }

    //
    // Update the CPU allocation if necessary
    //
    if( oldPolicy.cpuAllocation != newPolicy.cpuAllocation )
    {
        RT_ASSERT_MSG( !buf->m_mappedToHost, "Trying to change CPU allocation policy while buffer is mapped" );
        if( preserveContents )
        {
            const size_t copySize = buf->m_dims.getTotalSizeInBytes();
            replaceHostData( buf, oldPolicy, newPolicy, buf->m_dims, copySize );
        }
        else
            releaseResourcesOnDevices( buf, m_hostDevices, oldPolicy );
    }

    // Replace the policy
    buf->m_policy = newPolicy.policyKind;

    // Update pre/postlaunch lists
    m_prelaunchList.removeItem( buf );
    m_postlaunchList.removeItem( buf );
    addLaunchHooks( buf, newPolicy );

    // Sync at next launch if needed
    lazyAllocateAndSync( buf, newPolicy );
    for( MTextureSampler* tex : buf->m_attachedTextureSamplers )
        lazySync( tex );
}

bool MemoryManager::changePolicyLwdaAllocationWithinDevice( MBuffer* buf, const PolicyDetails& oldPolicy, const PolicyDetails& newPolicy )
{
    if( buf->m_validSet.overlaps( m_hostDevices ) || newPolicy.lwdaAllocation == PolicyDetails::LWDA_NONE )
    {
        return false;
    }

    if( newPolicy.lwdaAllocation == PolicyDetails::LWDA_DEMAND_LOAD )
    {
        return true;
    }

    // Lwrrently we trigger a host to device copy if the host is in the valid devices regardless of there potentially
    // being a valid copy on the device. For DISCARD buffers, there should never be a valid host copy at this point,
    // but for others there will always be one (see assert below).
    // TODO: Enable this branch for all buffers if any non-host device is in the valid set. Would need to expand
    // ResourceManager::copyResources to support transition from LWDA_GLOBAL to LWDA_ZEROCOPY and back [OP-2380]

    RT_ASSERT( oldPolicy.discardHostMemoryOnUnmap );

    // Try to directly copy from device to device without going through the host.
    // This may produce a new high-water mark on the device, since we temporarily allocate
    // the memory twice.

    // TODO: This is done by reusing the resource copying mechanism, but this requires an extra copying operation. This
    // should be redesigned so the extra copying is removed, see Jira task OP-2596.

    // Create a temp buffer and acquire resources with the new policy
    MBuffer tmp( this, buf->getDimensions(), newPolicy.policyKind, m_activeLwdaDevices );
    if( acquireResourcesOnDevices( &tmp, m_activeLwdaDevices, newPolicy ) == AcquireStatusBits::AcquireFailed )
        return false;

    // Copy data from buf to temp (device to device)
    for( DeviceSet::position deviceIndex : m_activeLwdaDevices )
    {
        Device* toDevice = m_deviceManager->allDevices()[deviceIndex];
        if( !buf->m_validSet.overlaps( DeviceSet( deviceIndex ) ) )
        {
            // possible after device change -> buffer should be in deferred lists
            RT_ASSERT( m_deferredAllocList.itemIsInList( buf ) && m_deferredSyncList.itemIsInList( buf ) );
            continue;
        }
        m_resourceManager->copyResource( tmp.m_resources.get(), toDevice, buf->m_resources.get(), toDevice, buf->m_dims );
    }

    DeviceSet validSet = buf->m_validSet;

    // Release old buf resources
    releaseResourcesOnDevices( buf, m_activeDevices, oldPolicy );

    // Acquire new buf resources with the new policy
    if( acquireResourcesOnDevices( buf, m_activeLwdaDevices, newPolicy ) == AcquireStatusBits::AcquireFailed )
    {
        releaseResourcesOnDevices( &tmp, m_activeDevices, newPolicy );
        return false;
    }

    // Copy resources back from tmp to buf
    for( DeviceSet::position deviceIndex : m_activeLwdaDevices )
    {
        Device* toDevice = m_deviceManager->allDevices()[deviceIndex];
        if( !validSet.overlaps( DeviceSet( deviceIndex ) ) )
            continue;
        m_resourceManager->copyResource( buf->m_resources.get(), toDevice, tmp.m_resources.get(), toDevice, buf->m_dims );
    }

    // Release the tmp resources
    releaseResourcesOnDevices( &tmp, m_activeDevices, newPolicy );

    // And finally update the valid set for buf
    setValidSet( buf, validSet, newPolicy );

    return true;
}

void MemoryManager::lazyAllocateAndSync( MBuffer* buf, const PolicyDetails& policy )
{
    const DeviceSet allowedActive = actualAllowedSet( buf, policy ) & m_activeDevices;

    // Post to appropriate alloc list if there are any outstanding allocations
    const bool inAllocList =
        policy.isBackingStore ? m_deferredBackingAllocList.itemIsInList( buf ) : m_deferredAllocList.itemIsInList( buf );
    if( !inAllocList && policy.activeDeviceAccess != PolicyDetails::N )
    {
        const DeviceSet needs = allowedActive - buf->m_allocatedSet;
        if( !needs.empty() )
        {
            if( policy.isBackingStore )
                m_deferredBackingAllocList.addItem( buf );
            else
                m_deferredAllocList.addItem( buf );
        }
    }

    // If it is already in the deferred list, no need to sync
    if( !m_deferredSyncList.itemIsInList( buf ) )
    {
        // Post to sync list if there are non-valid copies and the policy requires them
        updateValidSet( buf, policy );
        const DeviceSet needs = allowedActive - buf->m_validSet;
        if( policy.launchRequiresValidCopy && !needs.empty() )
        {
            m_deferredSyncList.addItem( buf );
        }
    }
}

void MemoryManager::replaceHostData( MBuffer*                buf,
                                     const PolicyDetails&    oldPolicy,
                                     const PolicyDetails&    newPolicy,
                                     const BufferDimensions& newSize,
                                     size_t                  copyBytes )
{
    // Map the host copy, which will synchronize to the host if necessary
    char* oldPtr = mapToHostInternal( buf, MAP_READ );

    // Set the valid set to just the host devices
    setValidSet( buf, m_hostDevices, oldPolicy );

    // Move the resource temporarily to a placeholder
    Device*    cpuDevice = m_deviceManager->cpuDevice();
    MAccess    oldAccess = buf->getAccess( cpuDevice );
    MResources tmpResources( buf );
    m_resourceManager->moveHostResources( &tmpResources, buf->m_resources.get() );

    // Update the size
    BufferDimensions oldSize = buf->m_dims;
    buf->m_dims              = newSize;

    // Allocate new resources on the CPU. The access pointer will now
    // contain the new pointer.
    buf->m_allocatedSet -= m_hostDevices;
    acquireResourcesOnDevicesOrFail( buf, m_hostDevices, newPolicy );
    MAccess newAccess = buf->getAccess( cpuDevice );
    RT_ASSERT( ( buf->m_dims.mipLevelCount() == 1 && newAccess.getKind() == MAccess::LINEAR )
               || ( buf->m_dims.mipLevelCount() > 1 && newAccess.getKind() == MAccess::MULTI_PITCHED_LINEAR ) );
    char* newPtr = nullptr;
    if( newAccess.getKind() == MAccess::LINEAR )
        newPtr = newAccess.getLinearPtr();
    else
        newPtr = newAccess.getPitchedLinear( 0 ).ptr;
    RT_ASSERT( oldPtr != newPtr );

    // Copy the overlap region
    std::memcpy( newPtr, oldPtr, copyBytes );

    // Release the temporary resources. We need to put the oldAccess
    // back for the release to work correctly.  Temporarily put the size
    // back so that logs and accounting are correct.
    buf->setAccess( cpuDevice, oldAccess );
    buf->m_dims          = oldSize;
    DeviceSet allDevices = ~DeviceSet();
    m_resourceManager->releaseResourcesOnDevices( &tmpResources, allDevices, newPolicy );

    // Restore the access pointer and size
    buf->setAccess( cpuDevice, newAccess );
    buf->m_dims = newSize;

    // And unmap
    unmapFromHostInternal( buf );
    lazyAllocateAndSync( buf, newPolicy );
}

void MemoryManager::updatePoliciesAfterVariantChange( unsigned int oldVariant )
{
    const unsigned int newVariant = computeLwrrentPolicyVariant();
    if( oldVariant == newVariant )
        return;  // No change

    for( MBuffer* buf : m_masterList )
    {
        const PolicyDetails& oldPolicy = PolicyDetails::getPolicyDetails( buf->getPolicy(), oldVariant );
        const PolicyDetails& newPolicy = PolicyDetails::getPolicyDetails( buf->getPolicy(), newVariant );
        if( &oldPolicy != &newPolicy )
            changePolicy( buf, oldPolicy, newPolicy );
    }
}

void MemoryManager::lazyResyncAll()
{
    // Process all buffers that need to be allocated or synced
    for( MBuffer* buf : m_masterList )
    {
        lazyAllocateAndSync( buf, getPolicyDetails( buf->getPolicy() ) );
    }

    // Resync texture samplers
    for( MTextureSampler* tex : m_masterTexList )
    {
        lazySync( tex );
    }
}

void MemoryManager::reevaluateArrayAssignments()
{
    for( int devIdx : m_activeLwdaDevices )
    {
        if( m_pdi[devIdx].reevaluateArrayAssignments )
            reevaluateArrayAssignments( devIdx );
    }
}

void MemoryManager::satisfyDeferredAllocations()
{
    // Bail out if there's nothing to do.
    if( m_deferredAllocList.empty() && m_deferredBackingAllocList.empty() )
        return;

    TIMEVIZ_SCOPE( "satisfyDeferredAllocations" );
    // Keep track of whether we've tried defragging the texheap.
    bool texHeapDefragAttempted = false;

    for( int retryCnt = 0;; ++retryCnt )
    {
        llog( k_mmll.get() ) << "satisfyDeferredAllocations, retryCnt: " << retryCnt << "\n";
        if( retryCnt > k_maxRetryFailsBeforeOOM.get() )
        {
            throw MemoryAllocationFailed( RT_EXCEPTION_INFO, "Out of memory" );
        }

        //
        // Allocate regular buffers.
        //
        {
            AcquireStatus status = 0;
            {
                // Try to allocate all the buffers to "collect" as many failure reasons
                // as we can. However, stop if we get too many, since lwca malloc can
                // be kind of slow, and with lots of buffers this can get painful.
                int failCnt = 0;
                for( MBuffer* buf : m_deferredAllocList )
                {
                    const AcquireStatus result =
                        acquireResourcesOnDevices( buf, m_activeDevices, getPolicyDetails( buf->getPolicy() ) );
                    status |= result;
                    if( result & AcquireFailed )
                        ++failCnt;
                    if( failCnt > k_maxAcquireFailsBeforeRetry.get() )
                        break;
                }
            }

            // If we couldn't get texheap space but were expecting to (texheap is
            // on), then "defrag" the texheap by resetting it, or if that didn't help
            // then turn it off entirely.
            if( m_resourceManager->isTexHeapEnabled() && ( status & AcquireUsedTexHeapFallback ) != 0 )
            {
                if( texHeapDefragAttempted == false && m_resourceManager->isTexHeapEligibleForDefrag() )
                {
                    llog( k_mmll.get() ) << "TexHeap defrag\n";
                    resetTexHeapAllocations();
                    resizeBackingStores();
                    texHeapDefragAttempted = true;
                    continue;
                }
                else
                {
                    llog( k_mmll.get() ) << "TexHeap disable\n";
                    resetTexHeapAllocations();
                    resizeBackingStores();
                    m_resourceManager->setTexHeapEnabled( false );
                    continue;
                }
            }

            // If at least one allocation couldn't be satisfied, try to make space by
            // promoting some buffers to peer-to-peer.
            if( status & AcquireFailed )
            {
                if( selectBuffersForPeerToPeer( 10 ) )
                {
                    continue;
                }
            }

            // If none of the above was successful, give up.
            if( status & AcquireFailed )
            {
                throw MemoryAllocationFailed( RT_EXCEPTION_INFO, "Out of memory" );
            }

            m_deferredAllocList.clear();
        }

        //
        // Allocate backing stores (texheap). We only know the required size for
        // those after all regular buffers have been allocated, so make sure this
        // comes second.
        //
        resizeBackingStores();
        {
            AcquireStatus status = 0;
            for( MBuffer* buf : m_deferredBackingAllocList )
            {
                status |= acquireResourcesOnDevices( buf, m_activeDevices, getPolicyDetails( buf->getPolicy() ) );
            }

            // If at least one allocation couldn't be satisfied, try to make space by
            // promoting some buffers to peer-to-peer.
            if( status & AcquireFailed )
            {
                if( selectBuffersForPeerToPeer( 10 ) )
                {
                    continue;
                }
            }

            // If none of the above was successful, give up.
            if( status & AcquireFailed )
            {
                throw MemoryAllocationFailed( RT_EXCEPTION_INFO, "Out of memory" );
            }

            m_deferredBackingAllocList.clear();
        }

        //
        // Even if all allocations went through, try to stay under a mem watermark
        // on each device, so we can reasonably hope that subsequent kernel
        // launches won't OOM due to driver-side memory allocations (lmem).
        //
        if( selectBuffersForPeerToPeer( 0 ) )
        {
            continue;
        }

        // Made it all the way here without triggering a retry -- done.
        break;
    }

    RT_ASSERT( m_deferredAllocList.empty() && m_deferredBackingAllocList.empty() );
    llog( k_mmll.get() ) << "satisfyDeferredAllocations done\n";
}

void MemoryManager::synchronizeBuffers()
{
    if( !m_deferredSyncList.empty() )
    {
        TIMEVIZ_SCOPE( "buffer sync" );
        for( MBuffer* buf : m_deferredSyncList )
            synchronizeMBuffer( buf, false );
        m_deferredSyncList.clear();
    }
}

void MemoryManager::synchronizeTextures()
{
    if( !m_deferredTexSyncList.empty() )
    {
        TIMEVIZ_SCOPE( "tex sync to backing" );
        for( MTextureSampler* tex : m_deferredTexSyncList )
            synchronizeTexToBacking( tex, m_activeDevices );
        m_deferredTexSyncList.clear();
    }
}

void MemoryManager::resizeBackingStores()
{
    // Note: lwrrently our only backing store is the texheap. We could easily
    // make this more generic if needed, by asking the resource manager for a
    // list of backing store MBuffers and the sizes they need.

    size_t        requestedSize = 0;
    MBufferHandle backingBuf;
    m_resourceManager->getTexHeapSizeRequest( &backingBuf, &requestedSize );

    // May not exist (during shutdown)
    if( !backingBuf )
        return;

    // If resource manager requested a new size, perform a resize
    const size_t lwrrentSize = backingBuf->getDimensions().width();
    if( requestedSize != lwrrentSize )
    {
        llog( k_mmll.get() ) << "TexHeap backing store resize from: " << lwrrentSize
                             << " elements to: " << requestedSize << " elements\n";
        BufferDimensions newDimensions = backingBuf->getDimensions();
        newDimensions.setSize( requestedSize );
        changeSize( backingBuf, newDimensions );
        TIMEVIZ_COUNT( "Texheap Backing Bytes", newDimensions.getTotalSizeInBytes() );
    }
}

void MemoryManager::resetTexHeapAllocations()
{
    // Abandon all texture resources on LWCA devices and mark them for re-sync.
    // Since the texheap is all-or-nothing, this is required when switching it on
    // or off.

    for( MBuffer* buf : m_masterList )
    {
        const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
        if( policy.lwdaAllocation == PolicyDetails::LWDA_PREFER_TEX_HEAP )
        {
            abandonResourcesOnDevices( buf, m_activeLwdaDevices, m_hostDevices );
            lazyAllocateAndSync( buf, policy );
        }
    }
}

size_t MemoryManager::promoteBufferToPeerToPeer( MBuffer* buf, const PolicyDetails& policy, int leastFreeMemDevIdx )
{
    // Release textures and reserve them again, because texture objects /
    // references must be recreated with foreign LWCA arrays.
    for( MTextureSampler* tex : buf->m_attachedTextureSamplers )
    {
        releaseResourcesOnDevices( tex, m_activeLwdaDevices, policy );
        reserveHWTextureIfAvailable( tex, m_activeLwdaDevices, policy );
    }

    // Leave one device allocated per-island. Switch that device to a P2P allocation.
    DeviceSet devicesToPreserve;
    for( DeviceSet island : m_deviceManager->getLwlinkIslands() )
    {
        bool foundDevToPreserve = false;

        // Try to preserve allocations that aren't on the device with the least
        // amount of memory
        for( int islandDevIdx : island )
        {
            if( islandDevIdx == leastFreeMemDevIdx )
                continue;

            devicesToPreserve |= DeviceSet( islandDevIdx );
            foundDevToPreserve = true;
        }

        if( !foundDevToPreserve )
            devicesToPreserve |= DeviceSet( island[0] );
    }

    DeviceSet abandonDevices   = m_activeLwdaDevices - devicesToPreserve;
    DeviceSet potentialTargets = m_hostDevices | devicesToPreserve;

    // Free all LWCA resources in preparation for a P2P allocation.
    abandonResourcesOnDevices( buf, abandonDevices, potentialTargets );

    // Mark the buffer for allocation in P2P.
    buf->m_p2pRequested = true;

    // Sync at next launch.
    lazyAllocateAndSync( buf, policy );
    for( MTextureSampler* tex : buf->m_attachedTextureSamplers )
    {
        lazySync( tex );
    }

    // Return a simple estimate how much memory we'll save per device on average
    // by moving this buffer to P2P. This assumes we keep a single copy of the
    // buffer data per device island, as opposed to keeping a
    // full copy on each allowed LWCA device.
    const int    ncopies  = ( actualAllowedSet( buf, policy ) & m_activeLwdaDevices ).count();
    const size_t oldBytes = buf->getDimensions().getTotalSizeInBytes() * ncopies;
    const size_t newBytes = buf->getDimensions().getTotalSizeInBytes() * m_deviceManager->getLwlinkIslands().size();
    return ( oldBytes - newBytes ) / m_activeLwdaDevices.count();
}

bool MemoryManager::selectBuffersForPeerToPeer( double forcePercent )
{
    // This function decides which (if any) buffers to place in P2P in order to
    // save memory, and releases+marks the selected ones. The algorithm takes
    // into account the total and available memory on the active devices. In
    // order to keep the implementation simple, we pick the device with the
    // lwrrently lowest amount of memory available as a reference for these
    // computations.
    //
    // If 'forcePercent' is zero, the request isn't too urgent, i.e. there wasn't
    // necessarily a failed allocation that triggered it. The function will try
    // to arrange things so that we stay below a high-watermark of device mem
    // usage on all devices. The purpose of this is to leave a reserve space for
    // allocations that may happen later and are out of the control of the memory
    // manager, e.g. launches that cause the driver to allocate local memory.
    //
    // If 'forcePercent' is >0, it will try to free up 'forcePercent' of the
    // total device memory, or reach the watermark if that means more freed
    // space. This is usually in reaction to a failed memory allocation, to give
    // the retry a change to succeed.
    //
    // Returns true if at least one buffer has been transitioned to P2P.

    if( !m_deviceManager->anyActiveDevicesAreLwlinkEnabled() )
        return false;

    const bool allowGmem = k_p2pMode.get() != "texOnly";
    const bool allowTex  = k_p2pMode.get() != "gmemOnly";
    const bool gmemFirst = k_p2pMode.get() == "gmemFirst";
    const bool texFirst  = k_p2pMode.get() == "texFirst";

    if( !allowGmem && !allowTex )
        return false;

    // Get info on the device with the least amount of free memory.
    size_t      freeDeviceBytes = 0, totalDeviceBytes = 0;
    LWDADevice* leastFreeMemDevice = m_deviceManager->leastFreeMemLWDADevice( &freeDeviceBytes, &totalDeviceBytes );

    // Figure out how much memory we're aiming to free (roughly, on average) on each device.
    const intptr_t reserveBytes = k_highWaterReserveInMB.get() * 1024 * 1024;
    const intptr_t bytesToSave =
        std::max( intptr_t( totalDeviceBytes * forcePercent * 0.01 ), reserveBytes - intptr_t( freeDeviceBytes ) );

    llog( k_mmll.get() ) << "selectBuffersForPeerToPeer, forcePercent: " << forcePercent
                         << ", totalDeviceBytes: " << corelib::formatSize( totalDeviceBytes )
                         << ", freeDeviceBytes: " << corelib::formatSize( freeDeviceBytes )
                         << ", bytesToSave: " << corelib::formatSize( bytesToSave ) << "\n";

    // Bail out if there's nothing to do.
    if( bytesToSave == 0 )
        return false;

    //
    // Collect viable candidate buffers for P2P, sort them by size, and
    // incrementally move them to P2P starting with the largest. This heuristic
    // could be enhanced, e.g. by taking into account the actual free space on
    // each device (rather than just a single one and working with averages), or
    // by asking the binding manager about references and aim for efficient
    // specialization.
    //
    // If the dynamic collection/sorting of buffers here ever becomes a perf
    // issue then we can keep track of sorted lists incrementally.
    //

    // Filter by eligibility.
    std::vector<MBuffer*> candidates;
    std::copy_if( m_masterList.begin(), m_masterList.end(), std::back_inserter( candidates ), [&]( const MBuffer* buf ) {
        const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
        if( policy.allowsP2P == false
            // HACK to allow BVHs in P2P for experiments
            // TODO: BVHs in P2P actually work very well, so should implement this properly
            && !( buf->getPolicy() == MBufferPolicy::internal_preferTexheap && k_p2pBvhAllowed.get()
                  && !m_resourceManager->isTexHeapEnabled() ) )
        {
            return false;
        }
        if( ( buf->m_allowedSet & m_activeLwdaDevices ) != m_activeLwdaDevices )
            return false;
        if( policy.lwdaTextureKind == PolicyDetails::TEX_NONE && !allowGmem )
            return false;
        if( policy.lwdaTextureKind != PolicyDetails::TEX_NONE && !allowTex )
            return false;
        if( buf->m_p2pRequested )
            return false;
        if( buf->getDimensions().getTotalSizeInBytes() <= static_cast<unsigned int>( k_p2pBufferMinSizeInBytes.get() ) )
            return false;
        return true;
    } );

    llog( k_mmll.get() ) << "selectBuffersForPeerToPeer found " << candidates.size() << " candidates out of "
                         << m_masterList.size() << "\n";

    // Bail out if we didn't find any candidates.
    if( candidates.empty() )
        return false;

    // Sorty by size.
    algorithm::sort( candidates, []( const MBuffer* lhs, const MBuffer* rhs ) {
        return lhs->getDimensions().getTotalSizeInBytes() > rhs->getDimensions().getTotalSizeInBytes();
    } );

    // Mark buffers for p2p until we think we've saved enough space. Run two passes
    // to handle gmem-first and tex-first modes.
    intptr_t totalBytesSaved = 0;
    for( int pass = 0; pass < 2; ++pass )
    {
        for( MBuffer* buf : candidates )
        {
            const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
            const bool           isTex  = policy.lwdaTextureKind != PolicyDetails::TEX_NONE;

            // Skip if it's not our turn depending on p2p mode.
            if( pass == 0 )
            {
                if( ( gmemFirst && isTex ) || ( texFirst && !isTex ) )
                    continue;
            }
            else
            {
                if( ( gmemFirst && !isTex ) || ( texFirst && isTex ) )
                    continue;
            }

            // Release and mark the candidate as p2p and keep track of (average)
            // space saved.
            const size_t bytesSaved = promoteBufferToPeerToPeer( buf, policy, leastFreeMemDevice->allDeviceListIndex() );
            totalBytesSaved += bytesSaved;

            if( log::active( k_mmll.get() ) )
                log( buf, "promoted to P2P, avg bytes saved per device: " + corelib::formatSize( bytesSaved )
                              + ", target: " + corelib::formatSize( bytesToSave ) );

            // Check if we're done.
            if( totalBytesSaved >= bytesToSave )
                break;
        }
    }

    return true;
}


/****************************************************************
 *
 * Texture management
 *
 ****************************************************************/

void MemoryManager::releaseResourcesOnDevices( MTextureSampler* tex, const DeviceSet& devices, const PolicyDetails& policy )
{
    const DeviceSet toFree = devices & m_allDevices;

    // Free the resources, which are stored directly in MAccess.
    for( DeviceSet::position deviceIndex : toFree )
    {
        Device* device = m_deviceManager->allDevices()[deviceIndex];
        if( LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device ) )
        {
            const MAccess texAccess = tex->getAccess( deviceIndex );
            if( texAccess.getKind() == MAccess::TEX_OBJECT )
            {
                RT_ASSERT( lwdaDevice->supportsHWBindlessTexture() );
                lwdaDevice->makeLwrrent();
                lwca::TexObject texObject = texAccess.getTexObject().texObject;
                texObject.destroy();
            }
            else if( texAccess.getKind() == MAccess::DEMAND_TEX_OBJECT )
            {
                RT_ASSERT( lwdaDevice->supportsHWBindlessTexture() );
                lwdaDevice->makeLwrrent();
                lwca::TexObject texObject = texAccess.getDemandTexObject().texObject;
                texObject.destroy();
            }
            else if( texAccess.getKind() == MAccess::TEX_REFERENCE )
            {
                unassignTexReference( deviceIndex, texAccess.getTexReference().texUnit );

                /*
           // TODO: uncomment this once the rest of MM is more robust and we're actually testing SW tex
      } else if( texAccess.getKind() == MAccess::PITCH_LINEAR ) {
        // SW texture, nothing to do
      }
      */
            }
            else
            {
                RT_ASSERT( texAccess.getKind() == MAccess::NONE );
            }
        }

        tex->setAccess( device, MAccess::makeNone() );
    }

    // Release the texture reservations
    for( DeviceSet::position deviceIndex : toFree )
    {
        unreserveHWTexture( tex, deviceIndex, policy );
    }

    // It is no longer synced to backing
    tex->m_syncedToBacking -= toFree;
}

void MemoryManager::lazySync( MTextureSampler* tex )
{
    DeviceSet needsSync = m_activeDevices - tex->m_syncedToBacking;
    if( !needsSync.empty() )
        m_deferredTexSyncList.addItem( tex );
}

void MemoryManager::reserveHWTextureIfAvailable( MTextureSampler* tex, const DeviceSet& devices, const PolicyDetails& policy )
{
    if( policy.lwdaTextureKind == PolicyDetails::TEX_NONE )
        return;

    const DeviceSet toReserve = ( devices & m_activeLwdaDevices ) & tex->m_backing->m_allowedSet;
    llog( k_mmll.get() ) << " reserveHWTextureIfAvailable: toReserve: " << toReserve.toString()
                         << ", activeLwdaSet: " << m_activeLwdaDevices.toString()
                         << ", tex_allowedSet: " << tex->m_backing->m_allowedSet.toString() << '\n';

    for( DeviceSet::position deviceIndex : toReserve )
    {
        if( !reserveHWTexture( tex, deviceIndex, policy ) )
        {
            // Reservation failed.  Leave it alone but trigger a global
            // reallocation of texture resources at some point in the
            // future.
            PerDeviceInfo& pdi             = m_pdi[deviceIndex];
            pdi.reevaluateArrayAssignments = true;
        }
    }
}

bool MemoryManager::reserveHWTexture( MTextureSampler* tex, unsigned int deviceIndex, const PolicyDetails& policy )
{
    Device* texDevice = m_deviceManager->allDevices()[deviceIndex];
    if( tex->m_hasHwTextureReserved.isSet( texDevice ) )
        return true;

    const bool     reserveTexRef = policy.lwdaTextureKind == PolicyDetails::TEX_REF;
    PerDeviceInfo& pdi           = m_pdi[deviceIndex];

    if( pdi.useBindlessTexture && !reserveTexRef )
    {
        if( pdi.numAvailableBindlessTextures == 0 )
            return false;
        pdi.numAvailableBindlessTextures--;
    }
    else
    {
        if( pdi.numAvailableBoundTextures == 0 )
            return false;
        pdi.numAvailableBoundTextures--;
    }

    // Reservation succeeded on this device (bindless or bound)
    tex->m_hasHwTextureReserved.insert( texDevice );
    return true;
}

void MemoryManager::unreserveHWTexture( MTextureSampler* tex, unsigned int deviceIndex, const PolicyDetails& policy )
{
    Device*        texDevice     = m_deviceManager->allDevices()[deviceIndex];
    const bool     reserveTexRef = policy.lwdaTextureKind == PolicyDetails::TEX_REF;
    PerDeviceInfo& pdi           = m_pdi[deviceIndex];

    if( !tex->m_hasHwTextureReserved.isSet( texDevice ) )
        return;

    if( pdi.useBindlessTexture && !reserveTexRef )
    {
        pdi.numAvailableBindlessTextures++;
    }
    else
    {
        pdi.numAvailableBoundTextures++;
    }

    tex->m_hasHwTextureReserved.remove( texDevice );
}

unsigned int MemoryManager::assignTexReference( unsigned int deviceIndex, MTextureSampler* tex )
{
    PerDeviceInfo& pdi              = m_pdi[deviceIndex];
    unsigned int   unit             = pdi.assignedTextureUnits.findFirst( false );
    pdi.assignedBoundTextures[unit] = tex;
    pdi.assignedTextureUnits.set( unit );
    return unit;
}

void MemoryManager::unassignTexReference( unsigned int deviceIndex, unsigned int texUnit )
{
    PerDeviceInfo& pdi                 = m_pdi[deviceIndex];
    pdi.assignedBoundTextures[texUnit] = nullptr;
    pdi.assignedTextureUnits.clear( texUnit );
}

void MemoryManager::reevaluateArrayAssignments( unsigned int deviceIndex )
{
    PerDeviceInfo& pdi = m_pdi[deviceIndex];
    if( !pdi.reevaluateArrayAssignments )
        return;

    // TODO: this function is probably outdated, see if we can remove it

    pdi.reevaluateArrayAssignments = false;
}

void MemoryManager::synchronizeTexToBacking( MTextureSampler* tex, const DeviceSet& activeDevices )
{
    const PolicyDetails& policy = getPolicyDetails( tex->m_backing->getPolicy() );
    if( log::active( k_mmll.get() ) )
        log( tex->m_backing.get(), "texture sync" );

    // Synchronize texture samplers to backing
    DeviceSet syncTargets = activeDevices - tex->m_syncedToBacking;
    for( DeviceSet::position deviceIndex : syncTargets )
        synchronizeTexToBacking( tex, deviceIndex, policy );
    tex->m_syncedToBacking |= syncTargets;
}

void MemoryManager::synchronizeTexToBacking( MTextureSampler* tex, unsigned int deviceIndex, const PolicyDetails& policy )
{
    llog( k_mmll.get() ) << " - Synchronize texture sampler on device: " << deviceIndex << '\n';

    const MAccess memAccess = tex->m_backing->getAccess( deviceIndex );

    Device* device = m_deviceManager->allDevices()[deviceIndex];
    if( deviceCast<const CPUDevice>( device ) )
    {
        // CPU is always SW texture
        RT_ASSERT( memAccess.getKind() == MAccess::MULTI_PITCHED_LINEAR );
        tex->setAccess( device, memAccess );
    }
    else if( LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device ) )
    {
        bool usesHWTexture = tex->m_hasHwTextureReserved.isSet( device );
        if( usesHWTexture )
        {
            bool reserveTexRef = policy.lwdaTextureKind == PolicyDetails::TEX_REF;
            if( lwdaDevice->supportsHWBindlessTexture() && !reserveTexRef )
            {
                const MAccess oldAccess = tex->getAccess( deviceIndex );
                // Destroy any existing tex object, as it cannot be updated in place
                if( oldAccess.getKind() == MAccess::TEX_OBJECT )
                {
                    lwdaDevice->makeLwrrent();
                    lwca::TexObject texObject = oldAccess.getTexObject().texObject;
                    texObject.destroy();
                    tex->setAccess( device, MAccess::makeNone() );
                }
                else if( oldAccess.getKind() == MAccess::DEMAND_TEX_OBJECT )
                {
                    lwdaDevice->makeLwrrent();
                    lwca::TexObject texObject = oldAccess.getDemandTexObject().texObject;
                    texObject.destroy();
                    tex->setAccess( device, MAccess::makeNone() );
                }

                // Create the texture object
                lwca::TexObject texObject =
                    m_resourceManager->createTexObject( tex->m_backing->m_resources.get(), lwdaDevice, tex->m_texDesc );

                if( tex->isDemandLoad( device ) )
                {
                    unsigned int startPage   = 0U;
                    unsigned int numPages    = 0U;
                    unsigned int minMipLevel = std::numeric_limits<unsigned int>::max();
                    unsigned int maxMipLevel = 0U;
                    if( oldAccess.getKind() == MAccess::DEMAND_TEX_OBJECT )
                    {
                        DemandTexObjectAccess demandTex = oldAccess.getDemandTexObject();
                        startPage                       = demandTex.startPage;
                        numPages                        = demandTex.numPages;
                        minMipLevel                     = demandTex.minMipLevel;
                        maxMipLevel                     = demandTex.maxMipLevel;
                    }
                    else
                    {
                        const PagingMode mode = m_context->getPagingManager()->getLwrrentPagingMode();
                        if( mode == PagingMode::SOFTWARE_SPARSE || mode == PagingMode::WHOLE_MIPLEVEL )
                        {
                            // need to allocate pages for this tex object
                            startPage = tex->allocateVirtualPages( m_context->getPagingManager(), numPages );
                        }
                        else
                        {
                            // Switching from LWCA sparse texture to plain texture for single level texture that
                            // would fit in the sparse texture miptail.
                            const DemandLoadArrayAccess access =
                                tex->m_backing->getAccess( device->allDeviceListIndex() ).getDemandLoadArray();
                            startPage   = access.virtualPageBegin;
                            numPages    = access.numPages;
                            minMipLevel = access.minMipLevel;
                            maxMipLevel = minMipLevel;
                        }
                    }
                    MAccess texAccess = MAccess::makeDemandTexObject( texObject, 0, startPage, numPages, minMipLevel, maxMipLevel );
                    tex->setAccess( device, texAccess );
                }
                else
                {
                    // Assign the object to the accessor
                    MAccess texAccess = MAccess::makeTexObject( texObject, 0 );
                    tex->setAccess( device, texAccess );
                }
            }
            else
            {
                // Punt if texture sampler has changed.
                RT_ASSERT( tex->getAccess( deviceIndex ).getKind() == MAccess::NONE );

                // No bindless, assign a free unit (which should have already
                // been reserved but not assigned).
                int     unit      = assignTexReference( deviceIndex, tex );
                MAccess texAccess = MAccess::makeTexReference( unit, 0 );
                tex->setAccess( device, texAccess );
            }
        }
        else
        {
            // SW texture
            RT_ASSERT_MSG( memAccess.getKind() == MAccess::MULTI_PITCHED_LINEAR, "Unknown memory pointer for texture" );
            tex->setAccess( device, memAccess );
        }
    }
    else
    {
        throw MemoryAllocationFailed( RT_EXCEPTION_INFO, "Illegal device kind" );
    }
}

void MemoryManager::switchLwdaSparseArrayToLwdaArray( MBuffer* buffer, DeviceSet devices )
{
    m_resourceManager->switchLwdaSparseArrayToLwdaArray( buffer->m_resources.get(), devices );
}

LWDA_ARRAY_SPARSE_PROPERTIES MemoryManager::getSparseTexturePropertiesFromMBufferProperties( const MBuffer* buffer )
{
    return m_resourceManager->getSparseTexturePropertiesFromMBufferProperties( buffer );
}

void MemoryManager::reallocDemandLoadLwdaArray( const MBufferHandle& buffer, unsigned int allDeviceListIndex, int minLevel, int maxLevel )
{
    m_resourceManager->reallocDemandLoadLwdaArray( buffer->m_resources.get(), allDeviceListIndex, minLevel, maxLevel );

    // Update the buffer's MAccess, which records the minMipLevel.
    buffer->updateDemandLoadArrayAccess( allDeviceListIndex );

    for( MTextureSampler* tex : buffer->m_attachedTextureSamplers )
    {
        reserveHWTexture( tex, allDeviceListIndex, getPolicyDetails( buffer->getPolicy() ) );
        tex->m_syncedToBacking.clear();
        lazySync( tex );
    }
}

void MemoryManager::syncDemandLoadMipLevel( const Buffer* buffer, void* baseAddress, size_t byteCount, unsigned int allDeviceListIndex, int mipLevel )
{
    m_resourceManager->syncDemandLoadMipLevel( buffer->getMBuffer()->m_resources.get(), baseAddress, byteCount,
                                               allDeviceListIndex, mipLevel );
}

void MemoryManager::syncDemandLoadMipLevelAsync( lwca::Stream& stream,
                                                 const Buffer* buffer,
                                                 void*         baseAddress,
                                                 size_t        byteCount,
                                                 unsigned int  allDeviceListIndex,
                                                 int           mipLevel )
{
    m_resourceManager->syncDemandLoadMipLevelAsync( stream, buffer->getMBuffer()->m_resources.get(), baseAddress,
                                                    byteCount, allDeviceListIndex, mipLevel );
}

void MemoryManager::fillTile( const MBufferHandle& buffer, unsigned int allDeviceListIndex, unsigned int layer, const void* data )
{
    m_resourceManager->fillTile( buffer->m_resources.get(), allDeviceListIndex, layer, data );
}

void MemoryManager::fillTileAsync( lwca::Stream&        stream,
                                   const MBufferHandle& buffer,
                                   unsigned int         allDeviceListIndex,
                                   unsigned int         layer,
                                   const void*          data )
{
    m_resourceManager->fillTileAsync( stream, buffer->m_resources.get(), allDeviceListIndex, layer, data );
}

void MemoryManager::fillHardwareTileAsync( lwca::Stream&        stream,
                                           const MBufferHandle& arrayBuf,
                                           const MBufferHandle& backingStorageBuf,
                                           unsigned int         allDeviceListIndex,
                                           const RTmemoryblock& memBlock,
                                           int                  offset )
{
    m_resourceManager->fillHardwareTileAsync( stream, arrayBuf->m_resources.get(), backingStorageBuf->m_resources.get(),
                                              allDeviceListIndex, memBlock, offset );
}

void MemoryManager::bindHardwareMipTailAsync( lwca::Stream&        stream,
                                              const MBufferHandle& arrayBuf,
                                              const MBufferHandle& backingStorageBuf,
                                              unsigned int         allDeviceListIndex,
                                              int                  mipTailSizeInBytes,
                                              int                  offset )
{
    m_resourceManager->bindHardwareMipTailAsync( stream, arrayBuf->m_resources.get(), backingStorageBuf->m_resources.get(),
                                                 allDeviceListIndex, mipTailSizeInBytes, offset );
}

void MemoryManager::fillHardwareMipTail( const MBufferHandle& arrayBuf, unsigned int allDeviceListIndex, const RTmemoryblock& memBlock )
{
    m_resourceManager->fillHardwareMipTail( arrayBuf->m_resources.get(), allDeviceListIndex, memBlock );
}

void MemoryManager::fillHardwareMipTailAsync( lwca::Stream&        stream,
                                              const MBufferHandle& arrayBuf,
                                              unsigned int         allDeviceListIndex,
                                              const RTmemoryblock& memBlock )
{
    m_resourceManager->fillHardwareMipTailAsync( stream, arrayBuf->m_resources.get(), allDeviceListIndex, memBlock );
}

/****************************************************************
 *
 * Sync and data transfer
 *
 ****************************************************************/

void MemoryManager::synchronizeMBuffer( MBuffer* buf, bool manualSync )
{
    if( log::active( k_mmll.get() ) )
        log( buf, "sync - activeDevices: " + m_activeDevices.toString() + " allocatedSet: " + buf->m_allocatedSet.toString() );

    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );

    // If the buffer policy requires a sync, or if it is explicitly requested
    if( policy.launchRequiresValidCopy || manualSync )
    {
        DeviceSet targets = actualAllowedSet( buf, policy ) & m_activeDevices;

        // If this is a P2P buffer, only sync to devices that are in the P2P allocated set
        if( buf->m_p2pRequested )
        {
            for( int devIdx : m_activeDevices )
            {
                if( !buf->isAllocatedP2P( devIdx ) )
                    targets -= DeviceSet( devIdx );
            }
        }

        RT_ASSERT_MSG( !buf->m_mappedToHost, "Allocation is still mapped to host during synchronize" );
        RT_ASSERT_MSG( ( buf->m_allocatedSet & targets ) == targets,
                       "Trying to synchronize MBuffer to device without allocation" );
        synchronizeToDevices( buf, targets, policy );
    }
}

void MemoryManager::synchronizeToDevices( MBuffer* buf, const DeviceSet& toDevices, const PolicyDetails& policy )
{
    RT_ASSERT_MSG( ( buf->m_allocatedSet & toDevices ) == toDevices, "Trying to sync to device without allocation" );
    if( log::active( k_mmll.get() ) )
        log( buf, "synchronizeToDevices - toDevices: " + toDevices.toString() );

    updateValidSet( buf, policy );
    const DeviceSet needsCopy = toDevices - buf->m_validSet;

    if( needsCopy.empty() )
        return;

    if( !buf->m_validSet.empty() )
    {
        copyToDevices( buf, needsCopy, buf->m_validSet );
    }
    else
    {
        // There was no valid copy. This can happen e.g. if the user did not
        // fill (=map+unmap) an input buffer on the host.
    }

    // Update the valid set to include the copies. Doing this even if there
    // was no valid copy, because it doesn't hurt but may save copies later.
    setValidSet( buf, buf->m_validSet | needsCopy, policy );
}

void MemoryManager::copyToDevices( MBuffer* buf, const DeviceSet& dstSet, const DeviceSet& srcSet )
{
    RT_ASSERT( srcSet.count() > 0 );
    if( log::active( k_mmll.get() ) )
        log( buf, "copy to: " + dstSet.toString() + " from: " + srcSet.toString() );

    // TODO: This just naively picks the first valid device as the source. It
    // should be improved to (explicitly) prefer device-to-device copies if
    // available.
    Device* fromDevice = m_deviceManager->allDevices()[srcSet[0]];

    // Copy to each device
    for( DeviceSet::position deviceIndex : dstSet )
    {
        Device* toDevice = m_deviceManager->allDevices()[deviceIndex];
        m_resourceManager->copyResource( buf->m_resources.get(), toDevice, buf->m_resources.get(), fromDevice, buf->m_dims );
    }
}


/****************************************************************
 *
 * Device management
 *
 ****************************************************************/

void MemoryManager::initializePerDeviceInfo( Device* device )
{
    unsigned int idx                        = device->allDeviceListIndex();
    m_pdi[idx].reevaluateArrayAssignments   = false;
    m_pdi[idx].useBindlessTexture           = false;
    m_pdi[idx].numAvailableBoundTextures    = 0;
    m_pdi[idx].numAvailableBindlessTextures = 0;
    if( LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device ) )
    {
        unsigned int numBound                = lwdaDevice->maximumBoundTextures();
        m_pdi[idx].useBindlessTexture        = lwdaDevice->supportsHWBindlessTexture();
        m_pdi[idx].numAvailableBoundTextures = numBound;
        m_pdi[idx].assignedBoundTextures.resize( numBound, nullptr );
        m_pdi[idx].assignedTextureUnits.resize( numBound );
        m_pdi[idx].assignedTextureUnits.clearAll();
        m_pdi[idx].numAvailableBindlessTextures = lwdaDevice->maximumBindlessTextures();
    }
}

void MemoryManager::resetPerDeviceInfo( Device* device )
{
    // Re-initialize to clear constants
    initializePerDeviceInfo( device );

    // Free arrays
    unsigned int idx = device->allDeviceListIndex();
    m_pdi[idx].assignedBoundTextures.clear();
    m_pdi[idx].assignedTextureUnits.resize( 0 );
}

/****************************************************************
 *
 * Helper functions
 *
 ****************************************************************/

void MemoryManager::addLaunchHooks( MBuffer* buf, const PolicyDetails& policy )
{
    // Add the preAccessHooks
    if( policy.accessHook != PolicyDetails::NOHOOK )
        m_prelaunchList.addItem( buf );

    // And postAccessHooks
    if( policy.accessHook != PolicyDetails::NOHOOK )
        m_postlaunchList.addItem( buf );
}

void MemoryManager::removeLaunchHooks( MBuffer* buf )
{
    m_prelaunchList.removeItem( buf );
    m_postlaunchList.removeItem( buf );
}

void MemoryManager::preAccess( MBuffer* buf )
{
    // Don't perform preAccess if it has already been done as part of enterFromAPI
    if( m_enteredFromAPI )
        return;

    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
    switch( policy.accessHook )
    {
        case PolicyDetails::NOHOOK:
            break;
        case PolicyDetails::GFXINTEROP:
            preAccess_GFXINTEROP( buf, policy, nullptr );
            break;
        case PolicyDetails::LWDAINTEROP:
            preAccess_LWDAINTEROP( buf, policy );
            break;
            // Default case intentionally omitted
    }
}

void MemoryManager::preAccess( const std::vector<MBuffer*> buffers )
{
    GfxInteropResourceBatch gfxResourceBatch;

    for( MBuffer* buf : buffers )
    {
        const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
        switch( policy.accessHook )
        {
            case PolicyDetails::NOHOOK:
                break;
            case PolicyDetails::GFXINTEROP:
                preAccess_GFXINTEROP( buf, policy, &gfxResourceBatch );
                break;
            case PolicyDetails::LWDAINTEROP:
                preAccess_LWDAINTEROP( buf, policy );
                break;
                // Default case intentionally omitted
        }
    }

    m_resourceManager->mapGfxInteropResourceBatch( gfxResourceBatch );
}

void MemoryManager::postAccess( MBuffer* buf )
{
    // Don't perform postAccess if it will be done as part of exitToAPI
    if( m_enteredFromAPI )
        return;

    const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
    switch( policy.accessHook )
    {
        case PolicyDetails::NOHOOK:
            break;
        case PolicyDetails::GFXINTEROP:
            postAccess_GFXINTEROP( buf, policy, nullptr );
            break;
        case PolicyDetails::LWDAINTEROP:
            postAccess_LWDAINTEROP( buf );
            break;
            // Default case intentionally omitted
    }
}

void MemoryManager::postAccess( const std::vector<MBuffer*> buffers )
{
    GfxInteropResourceBatch gfxResourceBatch;

    for( MBuffer* buf : buffers )
    {
        const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
        switch( policy.accessHook )
        {
            case PolicyDetails::NOHOOK:
                break;
            case PolicyDetails::GFXINTEROP:
                postAccess_GFXINTEROP( buf, policy, &gfxResourceBatch );
                break;
            case PolicyDetails::LWDAINTEROP:
                postAccess_LWDAINTEROP( buf );
                break;
                // Default case intentionally omitted
        }
    }
    m_resourceManager->unmapGfxInteropResourceBatch( gfxResourceBatch );
}

void MemoryManager::preAccess_GFXINTEROP( MBuffer* buf, const PolicyDetails& policy, GfxInteropResourceBatch* gfxResourceBatch )
{
    if( log::active( k_mmll.get() ) )
        log( buf, "preAccess_GFXINTEROP" );

    // Ensure that we have the resource (host memory in the case of
    // foreign interop) and mark it as the only valid copy.
    Device* interopDevice = m_resourceManager->getGfxInteropDevice( buf->m_resources.get() );

    // Update buffer size from the graphics resource only for texture.
    // Interop buffer dimensions must be set by user
    if( m_resourceManager->doesGfxInteropResourceSizeNeedUpdate( buf->m_resources.get() ) )
    {
        BufferDimensions newDims = m_resourceManager->queryGfxInteropResourceSize( buf->m_resources.get() );
        if( buf->m_dims != newDims )
            buf->m_dims = newDims;
    }

    // If there is intermediate copy then allocate.
    if( !m_resourceManager->isGfxInteropResourceImmediate( buf->m_resources.get(), policy ) )
    {
        acquireResourcesOnDevicesOrFail( buf, interopDevice, policy );
    }

    DeviceSet validDeviceSet( interopDevice );
    setValidSet( buf, validDeviceSet, policy );

    // Map the graphics resource to the interop device - either directly
    // or through a copy.
    m_resourceManager->mapGfxInteropResource( buf->m_resources.get(), policy, gfxResourceBatch );
    buf->m_allocatedSet.insert( interopDevice );

    // Foreign and multi-GPU interop need sync.
    lazyAllocateAndSync( buf, policy );
}

void MemoryManager::postAccess_GFXINTEROP( MBuffer* buf, const PolicyDetails& policy, GfxInteropResourceBatch* gfxResourceBatch )
{
    // Process ilwalidations from recent launch (if any)
    updateValidSet( buf, policy );
    if( log::active( k_mmll.get() ) )
        log( buf, "postAccess_GFXINTEROP: validSet=" + buf->m_validSet.toString() );

    // Ensure that we (again) have a valid copy on the interop device.
    Device* interopDevice = m_resourceManager->getGfxInteropDevice( buf->m_resources.get() );
    synchronizeToDevices( buf, DeviceSet( interopDevice ), policy );

    // Unmap the resource, triggering copies if necessary
    m_resourceManager->unmapGfxInteropResource( buf->m_resources.get(), policy, gfxResourceBatch );

    // If it is immediate (no intermediate copy) then remove the interop resource from the allocated set.
    if( m_resourceManager->isGfxInteropResourceImmediate( buf->m_resources.get(), policy ) )
    {
        buf->m_allocatedSet.remove( interopDevice );
    }

    // After unmap, there are no valid copies.
    setValidSet( buf, DeviceSet(), policy );
}

void MemoryManager::preAccess_LWDAINTEROP( MBuffer* buf, const PolicyDetails& policy )
{
    if( log::active( k_mmll.get() ) )
        log( buf, "preAccess_LWDAINTEROP" );

    if( buf->m_frozenSet.empty() )
        return;  // no lwca interop

    if( buf->m_frozenSet.count() != 1 && buf->m_frozenSet.count() != m_activeDevices.count() )
        throw IlwalidOperation( RT_EXCEPTION_INFO,
                                "LWCA interop: must set/get the device pointer on either one or all active devices" );

    if( buf->m_frozenSet.count() > 1 )
    {
        // All pointers set/gotten, so we can assume they're all valid.
        setValidSet( buf, buf->m_frozenSet, policy );
    }
    else if( buf->m_frozenSet.count() == 1 )
    {
        // Single-pointer case. Auto-sync depending on copy-on-dirty flag and
        // whether the host copy is valid from a previous map.
        if( ( buf->m_validSet & m_hostDevices ).empty() )
        {
            if( policy.copyOnDirty == false )
                setValidSet( buf, buf->m_frozenSet, policy );
        }
        lazyAllocateAndSync( buf, policy );
    }
}

void MemoryManager::postAccess_LWDAINTEROP( MBuffer* buf )
{
    if( log::active( k_mmll.get() ) )
        log( buf, "postAccess_LWDAINTEROP" );
}

void MemoryManager::verifyBeforeLaunch()
{
    for( MBuffer* buf : m_masterList )
    {
        const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
        log( buf, "validate" );

        // Validate allocations on active devices
        if( policy.activeDeviceAccess != PolicyDetails::N )
        {
            DeviceSet unallocated = m_activeDevices - buf->m_allocatedSet;
            if( !unallocated.empty() )
            {
                log( buf, "Unallocated devices remain: " + unallocated.toString() );
                RT_ASSERT_FAIL_MSG( "memory manager validation failed" );
            }

            // Validate a valid MAccess
            for( DeviceSet::position deviceIndex : m_activeDevices )
            {
                const MAccess mAccess = buf->getAccess( deviceIndex );
                if( mAccess.getKind() == MAccess::NONE )
                    log( buf, "WARNING: Invalid MAccess found during validateBeforeLaunch" );
            }
        }

        if( policy.launchRequiresValidCopy )
        {
            DeviceSet unsynced = m_activeDevices - buf->m_validSet;
            if( !unsynced.empty() )
            {
                log( buf, "Unallocated devices remain: " + unsynced.toString() );
                RT_ASSERT_FAIL_MSG( "memory manager validation failed" );
            }
        }
    }
}

void MemoryManager::log( const MBuffer* buf, const std::string& desc ) const
{
    llog( k_mmll.get() ) << "M" << buf->m_serialNumber << ": " << desc << " (dims: " << buf->getDimensions().toString()
                         << ", size: " << buf->getDimensions().getTotalSizeInBytes()
                         << ", policy: " << toString( buf->getPolicy() ) << ")\n";
}

void MemoryManager::dump( const char* where ) const
{
    const int logLevel = k_stateDumpLogLevel.isDefault() ? k_mmll.get() : k_stateDumpLogLevel.get();
    if( !log::active( logLevel ) )
        return;

    llog( logLevel ) << "---- MemoryManager " << where << " #" << m_lwrrentLaunchTimeStamp << " ----\n";

    // Sort by policy first, then size
    auto pred = []( MBuffer* lhs, MBuffer* rhs ) {
        if( lhs->getPolicy() != rhs->getPolicy() )
            return (int)lhs->getPolicy() < (int)rhs->getPolicy();
        return lhs->m_dims.getTotalSizeInBytes() > rhs->m_dims.getTotalSizeInBytes();
    };

    // Make a copy of the lists and sort
    std::vector<MBuffer*> master = m_masterList.getList();
    std::vector<MBuffer*> dal    = m_deferredAllocList.getList();
    std::vector<MBuffer*> dbal   = m_deferredBackingAllocList.getList();
    std::vector<MBuffer*> dsl    = m_deferredSyncList.getList();

    algorithm::sort( master, pred );
    algorithm::sort( dal, pred );
    algorithm::sort( dbal, pred );
    algorithm::sort( dsl, pred );

    llog( logLevel ) << "masterList:\n";
    for( MBuffer* buf : master )
    {
        llog( logLevel ) << "  " << buf->stateString() << "\n";
    }

    llog( logLevel ) << "deferredAllocList:\n";
    for( MBuffer* buf : dal )
    {
        llog( logLevel ) << "  M" << buf->m_serialNumber << " (" << toString( buf->m_policy ) << ", "
                         << corelib::formatSize( buf->m_dims.getTotalSizeInBytes() ) << " bytes)\n";
    }

    llog( logLevel ) << "deferredBackingAllocList:\n";
    for( MBuffer* buf : dbal )
    {
        llog( logLevel ) << "  M" << buf->m_serialNumber << " (" << toString( buf->m_policy ) << ", "
                         << corelib::formatSize( buf->m_dims.getTotalSizeInBytes() ) << " bytes)\n";
    }

    llog( logLevel ) << "deferredSyncList:\n";
    for( MBuffer* buf : dsl )
    {
        llog( logLevel ) << "  M" << buf->m_serialNumber << " (" << toString( buf->m_policy ) << ", "
                         << corelib::formatSize( buf->m_dims.getTotalSizeInBytes() ) << " bytes)\n";
    }

    // Distribution summary
    {
        size_t bytesReg = 0, bytesRegP2P = 0, bytesTex = 0, bytesTexP2P = 0, bytesBacking = 0;
        int    countReg = 0, countRegP2P = 0, countTex = 0, countTexP2P = 0, countBacking = 0;

        for( MBuffer* buf : master )
        {
            const PolicyDetails& policy = getPolicyDetails( buf->getPolicy() );
            if( policy.isBackingStore )
            {
                bytesBacking += buf->m_dims.getTotalSizeInBytes();
                countBacking++;
            }
            else if( buf->m_p2pRequested )
            {
                if( buf->m_attachedTextureSamplers.empty() )
                {
                    bytesRegP2P += buf->m_dims.getTotalSizeInBytes();
                    countRegP2P++;
                }
                else
                {
                    bytesTexP2P += buf->m_dims.getTotalSizeInBytes();
                    countTexP2P++;
                }
            }
            else
            {
                if( buf->m_attachedTextureSamplers.empty() )
                {
                    bytesReg += buf->m_dims.getTotalSizeInBytes();
                    countReg++;
                }
                else
                {
                    bytesTex += buf->m_dims.getTotalSizeInBytes();
                    countTex++;
                }
            }
        }
        llog( logLevel ) << "MBuffer Distribution:\n";
        llog( logLevel ) << "    regular local     : " << corelib::formatSize( bytesReg ) << " bytes total, "
                         << countReg << " buffers\n";
        llog( logLevel ) << "    regular p2p       : " << corelib::formatSize( bytesRegP2P ) << " bytes total, "
                         << countRegP2P << " buffers\n";
        llog( logLevel ) << "    tex backing local : " << corelib::formatSize( bytesTex ) << " bytes total, "
                         << countTex << " buffers\n";
        llog( logLevel ) << "    tex backing p2p   : " << corelib::formatSize( bytesTexP2P ) << " bytes total, "
                         << countTexP2P << " buffers\n";
        llog( logLevel ) << "    other backing     : " << corelib::formatSize( bytesBacking ) << " bytes total, "
                         << countBacking << " buffers\n";
    }
}

static void printMemTableEntry( UsageReport& ur, const std::string& category, const size_t count, const size_t bytes )
{
    ureport2( ur, "MEM USAGE" ) << "| " << std::right << std::setw( 16 ) << category << " | " << std::right << std::setw( 6 )
                                << count << " | " << std::right << std::setw( 12 ) << std::fixed << std::setprecision( 1 )
                                << static_cast<float>( bytes ) / ( 1024.f * 1024.f ) << " |" << std::endl;
}

void MemoryManager::reportMemoryUsage()
{
    UsageReport& ur = m_context->getUsageReport();
    if( !ur.isActive( 2 ) )
        return;

    size_t buffer_count = 0, buffer_bytes = 0, gfx_interop_count = 0, gfx_interop_bytes = 0, lwda_interop_count = 0,
           lwda_interop_bytes = 0, texture_count = 0, texture_bytes = 0, internal_count = 0, internal_bytes = 0,
           p2p_buffer_count = 0, p2p_buffer_bytes = 0, p2p_texture_count = 0, p2p_texture_bytes = 0,
           demand_load_count = 0, demand_load_bytes = 0;
    bool raw_access_detected = false;

    const std::vector<MBuffer*>& mbuffs = getMasterList();
    for( MBuffer* mbuff : mbuffs )
    {
        const PolicyDetails& policy = getPolicyDetails( mbuff->getPolicy() );
        if( !policy.isBackingStore && mbuff->m_p2pRequested )
        {
            if( mbuff->m_attachedTextureSamplers.empty() )
            {
                p2p_buffer_bytes += mbuff->m_dims.getTotalSizeInBytes();
                p2p_buffer_count++;
            }
            else
            {
                p2p_texture_bytes += mbuff->m_dims.getTotalSizeInBytes();
                p2p_texture_count++;
            }
        }

        const size_t bytes = mbuff->getDimensions().getTotalSizeInBytes();
        switch( mbuff->getPolicy() )
        {
            case MBufferPolicy::readonly_raw:
            case MBufferPolicy::readonly_discard_hostmem_raw:
            case MBufferPolicy::readwrite_raw:
            case MBufferPolicy::writeonly_raw:
                raw_access_detected = true;
            // fall through

            case MBufferPolicy::readonly:
            case MBufferPolicy::readonly_discard_hostmem:
            case MBufferPolicy::readwrite:
            case MBufferPolicy::writeonly:
                ++buffer_count;
                buffer_bytes += bytes;
                break;

            case MBufferPolicy::readonly_lwdaInterop:
            case MBufferPolicy::readwrite_lwdaInterop:
            case MBufferPolicy::writeonly_lwdaInterop:
            case MBufferPolicy::readonly_lwdaInterop_copyOnDirty:
            case MBufferPolicy::readwrite_lwdaInterop_copyOnDirty:
            case MBufferPolicy::writeonly_lwdaInterop_copyOnDirty:
                ++lwda_interop_count;
                lwda_interop_bytes += bytes;
                break;

            case MBufferPolicy::readonly_sparse_backing:
                // TODO: Report something meaningful here.
                break;

            case MBufferPolicy::readonly_demandload:
            case MBufferPolicy::texture_readonly_demandload:
            case MBufferPolicy::tileArray_readOnly_demandLoad:
                ++demand_load_count;
                demand_load_bytes += bytes;
                break;

            case MBufferPolicy::gpuLocal:
                ++buffer_count;
                buffer_bytes += bytes;
                break;

            case MBufferPolicy::readonly_gfxInterop:
            case MBufferPolicy::readwrite_gfxInterop:
            case MBufferPolicy::writeonly_gfxInterop:
                ++gfx_interop_count;
                gfx_interop_bytes += bytes;
                break;

            case MBufferPolicy::texture_linear:
            case MBufferPolicy::texture_linear_discard_hostmem:
            case MBufferPolicy::texture_array:
            case MBufferPolicy::texture_array_discard_hostmem:
            case MBufferPolicy::texture_gfxInterop:
                ++texture_count;
                texture_bytes += bytes;
                break;

            case MBufferPolicy::internal_readonly:
            case MBufferPolicy::internal_readwrite:
            case MBufferPolicy::internal_writeonly:
            case MBufferPolicy::internal_readonly_manualSync:
            case MBufferPolicy::internal_readwrite_manualSync:
            case MBufferPolicy::internal_texheapBacking:
            case MBufferPolicy::internal_preferTexheap:
            case MBufferPolicy::internal_hostonly:
            case MBufferPolicy::internal_readonly_deviceonly:
                ++internal_count;
                internal_bytes += bytes;
                break;

            case MBufferPolicy::unused:
                break;
                // intentionally omit default case
        }
    }
    ureport2( ur, "MEM USAGE" ) << "Buffer GPU memory usage:" << std::endl;
    {
        ureport2( ur, "MEM USAGE" ) << "|         Category |  Count |  Total MByte |" << std::endl;
        printMemTableEntry( ur, "buffer", buffer_count, buffer_bytes );
        printMemTableEntry( ur, " > p2p", p2p_buffer_count, p2p_buffer_bytes );
        printMemTableEntry( ur, "texture", texture_count, texture_bytes );
        printMemTableEntry( ur, "  > p2p", p2p_texture_count, p2p_texture_bytes );
        printMemTableEntry( ur, "demand load", demand_load_count, demand_load_bytes );
        printMemTableEntry( ur, "gfx interop", gfx_interop_count, gfx_interop_bytes );
        printMemTableEntry( ur, "lwca interop", lwda_interop_count, lwda_interop_bytes );
        printMemTableEntry( ur, "optix internal", internal_count, internal_bytes );
    }
    ureport2( ur, "MEM USAGE" ) << "Buffer host memory usage: " << static_cast<float>( getUsedHostMemory() ) / ( 1024.f * 1024.f )
                                << " Mbytes" << std::endl;

    if( raw_access_detected )
    {
        ureport2( ur, "MEM USAGE" ) << "WARNING: Raw pointer buffer access detected." << std::endl;
    }
}


void MemoryManager::trackMaxMemoryUsage()
{
    if( !k_trackMaxMemUsage.get() && !Metrics::isEnabled() )
        return;

    // Query the used memory on each device and update the stats
    for( int dev : m_activeLwdaDevices )
    {
        LWDADevice* device = deviceCast<LWDADevice>( m_deviceManager->allDevices()[dev] );
        RT_ASSERT( device );

        size_t totalMem = 0, freeMem = 0;
        device->makeLwrrent();
        lwca::memGetInfo( &freeMem, &totalMem );

        m_statsMaxUsedMem[dev] = std::max( m_statsMaxUsedMem[dev], totalMem - freeMem );
        m_statsTotalMem[dev]   = totalMem;

        Metrics::logInt( stringf( "mem_used_dev<%d>", dev ).c_str(), totalMem - freeMem );
    }
}

void MemoryManager::printMaxMemoryUsage()
{
    if( !k_trackMaxMemUsage.get() )
        return;

    lprint << "Max device memory used:\n";
    for( Device* device : m_deviceManager->allDevices() )
    {
        const int idx = device->allDeviceListIndex();
        if( m_statsMaxUsedMem[idx] == 0 )
            continue;

        lprint << "  " << std::setw( 5 ) << m_statsMaxUsedMem[idx] / ( 1024 * 1024 ) << " / " << std::setw( 5 )
               << m_statsTotalMem[idx] / ( 1024 * 1024 ) << " MB"
               << "            " << device->deviceName() << "\n";
    }
}

void MemoryManager::reserveMemoryForDebug()
{
    if( !k_limitDeviceMemoryInMB.get() || m_memoryForDebugReserved )
    {
        return;
    }

    for( int allDeviceIndex : m_activeLwdaDevices )
    {
        LWDADevice* device = deviceCast<LWDADevice>( m_deviceManager->allDevices()[allDeviceIndex] );
        if( !device )
            continue;
        device->makeLwrrent();
        size_t limitMB = k_limitDeviceMemoryInMB.get();
        // preallocate limit size to free it at the end
        void* ptr = m_resourceManager->reallocateMemoryForDebug( nullptr, limitMB * 1024 * 1024 );

        // to make it faster we allocate memory in different scale
        // allocate gigabytes
        try
        {
            while( true )
            {
                m_resourceManager->reallocateMemoryForDebug( nullptr, 1024 * 1024 * 1024 );
            }
        }
        catch( ... )
        {
        }
        // allocate tens of megabytes
        try
        {
            while( true )
            {
                m_resourceManager->reallocateMemoryForDebug( nullptr, 1024 * 1024 * 10 );
            }
        }
        catch( ... )
        {
        }
        // allocate megabytes
        try
        {
            while( true )
            {
                m_resourceManager->reallocateMemoryForDebug( nullptr, 1024 * 1024 );
            }
        }
        catch( ... )
        {
        }

        // free preallocated limit
        m_resourceManager->reallocateMemoryForDebug( ptr, 0 );
        lprint << "Device " << device->deviceName()
               << " memory available: " << device->getAvailableMemory() / 1024 / 1024 << " MB" << std::endl;
    }

    m_memoryForDebugReserved = true;
}

unsigned int MemoryManager::computeLwrrentPolicyVariant() const
{
    // Note that if any method changes one of these assumptions, then
    // all buffers must be considered for reallocation: use
    // updatePoliciesAfterVariantChange.
    unsigned int variant = 0;
    // Add multiGPU and singleGPU variants.
    if( m_activeDevices.count() > 1 || k_forceMultiGPU.get() )
        variant |= PolicyDetails::MULTIGPU;
    else
        variant |= PolicyDetails::SINGLEGPU;
    return variant;
}

const PolicyDetails& MemoryManager::getPolicyDetails( MBufferPolicy policy ) const
{
    const unsigned int variant = computeLwrrentPolicyVariant();
    return PolicyDetails::getPolicyDetails( policy, variant );
}

DeviceSet MemoryManager::actualAllowedSet( MBuffer* buf, const PolicyDetails& policy ) const
{
    DeviceSet result = buf->m_allowedSet;

    // We usually want to allow mapping to the host unless CPU allocations are
    // explicitly forbidden by the policy.
    if( policy.cpuAllocation != PolicyDetails::CPU_NONE )
        result |= m_hostDevices;

    // Exclude CPU devices if the policy does not allow them.
    if( policy.cpuAllocation == PolicyDetails::CPU_NONE )
        result -= m_hostDevices;

    // Exclude LWCA devices if the policy doesn't allow them.
    if( policy.lwdaAllocation == PolicyDetails::LWDA_NONE )
        result -= m_allLwdaDevices;

    return result;
}

/****************************************************************
 *
 * Utilities used by other classes
 *
 ****************************************************************/

// Utility function used by MBuffer to get the device pointer from the
// device index
Device* MemoryManager::getDevice( unsigned int allDeviceIndex )
{
    return m_deviceManager->allDevices()[allDeviceIndex];
}

const std::vector<MBuffer*>& MemoryManager::getMasterList()
{
    return m_masterList.getList();
}

/****************************************************************
 *
 * Utilities used for testing
 *
 ****************************************************************/

bool MemoryManager::policyAllowsMapToHostForRead( const MBufferHandle& buf )
{
    const PolicyDetails& details = getPolicyDetails( buf->getPolicy() );
    return details.allowsHostReadAccess();
}

bool MemoryManager::policyAllowsMapToHostForWrite( const MBufferHandle& buf )
{
    const PolicyDetails& details = getPolicyDetails( buf->getPolicy() );
    return details.allowsHostWriteAccess();
}

}  // namespace optix
