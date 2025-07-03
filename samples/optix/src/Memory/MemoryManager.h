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

#include <LWCA/TexRef.h>
#include <Memory/MBuffer.h>
#include <Memory/MBufferPolicy.h>
#include <Memory/MTextureSampler.h>
#include <Memory/MapMode.h>
#include <Util/IndexedVector.h>

#include <Util/BitSet.h>

#include <lwca.h>
#include <lwda_etbl/texture.h>

#include <functional>
#include <vector>

namespace optix {

class Buffer;
class BufferDimensions;
class Context;
class CPUDevice;
class LWDADevice;
class Device;
class DeviceManager;
class ResourceManager;
class UpdateManager;
struct InteropResource;
struct PolicyDetails;
struct TextureDescriptor;
struct GfxInteropResourceBatch;

namespace lwca {
class Stream;
}  // namespace lwca

/**
 *
 * The memory manager is responsible for most allocations in an
 * optix program, including back store for buffers, texture
 * allocations, and backing store for internal tables.
 *
 * The primary design of the memory manager is to allow an
 * allocation to be treated as a single unit (MBuffer). The data in
 * an MBuffer may be migrated from one device to another and/or
 * duplicated, as long as one valid copy is always kept. This design
 * is similar to a distributed shared memory coherence mechanism.
 *
 * A MBuffer is allocated on a set of devices, called the
 * allowedSet. In most cases, the allowedSet will be the universe of
 * devices. However, some internal uses of buffers will allow an
 * allocation only on a specific set.
 *
 * Another foundational concept of the MemoryManager is that an
 * allocation can be honored in multiple ways - using device memory,
 * textures, 0-copy, paged memory or other resources. Furthermore,
 * the memory manager can change the low-level allocation at any
 * point outside of the launch window (see below). If the low-level
 * allocation is changed, the MemoryManager will issue a callback to
 * any attached MBufferListener that might be interested in the
 * change.
 *
 * During a launch (rtContextLaunch), the active device allocations
 * will be locked down so that Plan assumptions can be built around
 * memory allocations. At the beginning of this launch, all MBuffers
 * will be provisioned with necessary resources, and most MBuffers
 * will be synchronized to the active devices by transferring data
 * for any invalid copies. However, a few internal tables request a
 * "manualSync" policy so that they will not be copied implicitly
 * during syncAllMemoryBeforeLaunch. In these cases, the client must
 * call manualSynchronize after the table is populated and before
 * finalizeTransfersBeforeLaunch.
 *
 * Additional design principles:
 * - Support millions of MBuffers efficiently
 * - No O(N) work in launch or allocate, where N is the number of
 *   outstanding buffers.
 * - Limited use of abstraction so that low-level operations are
 *   always explicit and intentional.
 * - Driver-like structure with simple code paths and unfancy
 *   data structures.
 * - Memory Manager should be self-contained so that it can make all
 *   decisions about memory management. If you add something that
 *   calls a virtual function then something is probably wrong.
 * - Memory Manager should be idempotent, meaning that if a set of
 *   allocations are performed in any order then the system always
 *   arrives at an equivalent state. In cases where hysteresis is
 *   used to avoid resource thrash, it should be well dolwmented and
 *   still reversible.
 */

/*
   From Steve in OP-655 (TODO: work this into the function docs below):
      It should not be legal to change the size of a mapped buffer, but
      changing the policy should be allowed.  As we finish the design for
      LWCA interop, we need to add the frozen bits. It will be illegal to
      change the size for anything that is frozen. Mapping a buffer will
      freeze the pointer.  And if the pointer is frozen then it should not
      get reallocated into a different memory space.
*/

class MemoryManager
{
  public:
    // Disallow copying
    MemoryManager( const MemoryManager& ) = delete;
    MemoryManager& operator=( const MemoryManager& ) = delete;

    /****************************************************************
   * Allocate MBuffers and MTextureSamplers. Free is implicit,
   * oclwring when the MBufferHandle is release by the last shared
   * owner.
   ****************************************************************/

    // Normal allocation, which is valid on all devices including the
    // host.  Optionally add an event listener.
    MBufferHandle allocateMBuffer( const BufferDimensions& size, MBufferPolicy policy, MBufferListener* listener = nullptr );

    // Device-specific allocation - memory manager will allocate only
    // on the specified devices. If mapToHost is required then host must
    // be part of the allowedDevices set.  Optionally add an event
    // listener.
    MBufferHandle allocateMBuffer( const BufferDimensions& size,
                                   MBufferPolicy           policy,
                                   const DeviceSet&        allowedDevices,
                                   MBufferListener*        listener = nullptr );

    // Allocate a buffer for graphics interop where the resource is on
    // the specified device. Valid on all devices including the host.
    // Optionally add an event listener.
    MBufferHandle allocateMBuffer( const GfxInteropResource& resource,
                                   Device*                   gfxInteropDevice,
                                   MBufferPolicy             policy,
                                   MBufferListener*          listener = nullptr );


    // Allocate a texture sampler attached to the specified memory.
    // Optionally attach an event listener.
    MTextureSamplerHandle attachMTextureSampler( const MBufferHandle&     backing,
                                                 const TextureDescriptor& texDesc,
                                                 MTextureSamplerListener* listener = nullptr );


    /****************************************************************
   * Manual synchronization
   ****************************************************************/

    // Explicitly synchronize this memory object (ensure a valid allocation
    // on all allowed devices in the active set). Any required transfers may be
    // asynchronous.
    void manualSynchronize( const MBufferHandle& mem );

    void manualSynchronize( const MTextureSamplerHandle& texHandle, unsigned int allDeviceListIndex );


    /****************************************************************
   * Update methods. In general, these should be very limited since
   * changing state can void a lot of assumptions.
   ****************************************************************/

    // Resize the MBuffer. Preserving the contents of the overlapping region is
    // optionally possible for 1D buffers.
    void changeSize( const MBufferHandle& buffer, const BufferDimensions& newDims, bool preserveContents = false );
    void changeSize( const MBufferHandle&    buffer,
                     const BufferDimensions& newDims,
                     std::function<void( LWDADevice*, LWdeviceptr, LWdeviceptr )> copyFunc );

    // Change the policy. Preserves contents.
    void changePolicy( const MBufferHandle& buffer, MBufferPolicy newPolicy );

    // Update the texture descriptor properties without reallocating the texture
    void updateTextureDescriptor( const MTextureSamplerHandle& sampler, const TextureDescriptor& texDesc );


    /****************************************************************
   * Manage host mappings which allow the API or other CPU clients
   * to fill and read data.
   ****************************************************************/

    // Map the data to the host with the specified map mode. Depending
    // on the map mode, the data may no longer be valid on non-host
    // devices.  Only one host mapping is allowed to be open at a time
    // for each MBuffer.
    char* mapToHost( const MBufferHandle& mem, MapMode mode );

    // Release the host mapping, after which the host pointer returned
    // by mapToHost may no longer be valid.
    void unmapFromHost( const MBufferHandle& );

    // The host pointer is returned.
    char* getMappedToHostPtr( const MBufferHandle& );

    // Return true is the buffer handle is mapped to the host.
    bool isMappedToHost( const MBufferHandle& mem );

    /****************************************************************
   * Graphics interop
   ****************************************************************/

    // Register the graphics resource with LWCA, making the resource
    // immutable. This will update the size and format of the buffer.
    void registerGfxInteropResource( const MBufferHandle& buf );

    // Unregister the resource from LWCA, making it mutable
    // again. Further mappings or launches are illegal until the
    // resource is re-registered.
    void unregisterGfxInteropResource( const MBufferHandle& buf );


    /****************************************************************
   * Manage pointers for LWCA interop
   ****************************************************************/

    // Set the LWCA global memory pointer for the specified device,
    // marking it both as frozen (will not change) and external.
    void setLwdaInteropPointer( const MBufferHandle& buf, void* ptr, Device* device );

    // Get the LWCA global memory pointer for the specified device,
    // marking it as frozen (will not change). The returned pointer will
    // become invalid if buf changes size.
    void* getLwdaInteropPointer( const MBufferHandle& buf, Device* device );

    // Marks the devices dirty (valid) for which set/getLwdaInteropPointer has been called.
    void markDirtyLwdaInterop( const MBufferHandle& buf );

    // Get device set for which set/getLwdaInteropPointer has been called.
    DeviceSet getLwdaInteropDevices( const MBufferHandle& bufhdl ) const;

    /****************************************************************
   * Context interface - a three step launch sequence, which is
   * called ONLY by Context during a launch.  During the launch
   * window, no further allocations are allowed so that the Plan can
   * be chosen and compiled against the current memory allocations.
   ****************************************************************/

    // Create and destroy memory manager. Done only by the context.
    MemoryManager( Context* context );
    ~MemoryManager();

    // Called by the context to perform cleanup that might throw exceptions before destroying
    // the MemoryMananger.
    void shutdown();

    // Triggers OpenGL and LWCA interop functions that are required once
    // when OptiX is launched or a command list is exelwted.
    void enterFromAPI();

    // Finish interop functions before transferring control back to the
    // application.
    void exitToAPI();

    // Allocates storage for all outstanding MBuffers on each active
    // device, and begin synchronization of any incoherent MBuffers
    // (possibly asynchronously).  Allocation of MBuffer and
    // MTextureSampler are not allowed again until
    // releaseMemoryAfterLaunch().
    void syncAllMemoryBeforeLaunch();

    void switchLwdaSparseArrayToLwdaArray( MBuffer* buffer, DeviceSet devices );
    LWDA_ARRAY_SPARSE_PROPERTIES getSparseTexturePropertiesFromMBufferProperties( const MBuffer* buffer );
    void reallocDemandLoadLwdaArray( const MBufferHandle& buffer, unsigned int allDeviceListIndex, int minLevel, int maxLevel );
    void syncDemandLoadMipLevel( const Buffer* buffer, void* baseAddress, size_t byteCount, unsigned int allDeviceListIndex, int mipLevel );
    void syncDemandLoadMipLevelAsync( lwca::Stream& stream,
                                      const Buffer* buffer,
                                      void*         baseAddress,
                                      size_t        byteCount,
                                      unsigned int  allDeviceListIndex,
                                      int           mipLevel );
    void fillTile( const MBufferHandle& buffer, unsigned int allDeviceListIndex, unsigned int layer, const void* data );
    void fillTileAsync( lwca::Stream& stream, const MBufferHandle& buffer, unsigned int allDeviceListIndex, unsigned int layer, const void* data );
    void fillHardwareTileAsync( lwca::Stream&        stream,
                                const MBufferHandle& arrayBuf,
                                const MBufferHandle& backingStorageBuf,
                                unsigned int         allDeviceListIndex,
                                const RTmemoryblock& memBlock,
                                int                  offset );
    void bindHardwareMipTailAsync( lwca::Stream&        stream,
                                   const MBufferHandle& arrayBuf,
                                   const MBufferHandle& backingStorageBuf,
                                   unsigned int         allDeviceListIndex,
                                   int                  mipTailSizeInBytes,
                                   int                  offset );
    void fillHardwareMipTail( const MBufferHandle& arrayBuf, unsigned int allDeviceListIndex, const RTmemoryblock& memBlock );
    void fillHardwareMipTailAsync( lwca::Stream&        stream,
                                   const MBufferHandle& arrayBuf,
                                   unsigned int         allDeviceListIndex,
                                   const RTmemoryblock& memBlock );

    // Ensure that transfers of data will be complete before a kernel
    // is launched, needed in multi-GPU configurations to ensure that
    // copies of peer data will be visible before the kernel launch if
    // peer-to-peer is being used.
    void finalizeTransfersBeforeLaunch();

    // Re-enable further allocations and ilwalidate any copies for
    // data that may have been written during launch.
    void releaseMemoryAfterLaunch();


    RTsize getUsedHostMemory() const;

    /****************************************************************
   * ExelwtionStrategy interface - used for exelwtion strategies to
   * set up texture references before launch. This is used for bound
   * textures - Fermi and specialized textures such as texheap.
   ****************************************************************/
    const BitSet& getAssignedTexReferences( Device* device ) const;
    void bindTexReference( Device* device, unsigned int texUnit, lwca::TexRef texref );

    /****************************************************************
   * DeviceManager interface - when the DeviceManger initiates a
   * change in the active devices, this will update the memory
   * manager state to accommodate the new configuration.
   ****************************************************************/
    void setActiveDevices( const DeviceSet& devices );

    /****************************************************************
   * Utilities to determine properties of allocations, primarily for
   * use in specialization code.
   ****************************************************************/
    void determineMixedAccess( const MTextureSamplerHandle& tex, bool& mixedAccess, bool& mixedPointer, MAccess& access );
    void determineMixedAccess( const MBufferHandle& buf, bool& mixedAccess, bool& mixedPointer, MAccess& access );

    /****************************************************************
   * Functions to assist with testing
   ****************************************************************/
    bool policyAllowsMapToHostForRead( const MBufferHandle& buf );
    bool policyAllowsMapToHostForWrite( const MBufferHandle& buf );

    /****************************************************************
   * Pinning to a device
   ****************************************************************/
    char* pinToDevice( const MBufferHandle& bufhdl, Device* device, MapMode mode = MAP_WRITE_DISCARD );
    void unpinFromDevice( const MBufferHandle& bufhdl, Device* device );

    /****************************************************************
     * Special interface for builders that write to device memory
     * (like texheap). Valid only until releaseMemoryAfterLaunch.
     * exclusiveMode means only a single pointer for all devices will
     * be queried (data will be synchronized to other devices).
     ****************************************************************/
    char* getWritablePointerForBuild( const MBufferHandle& buf, Device* device, bool exclusiveMode );

  private:
    typedef unsigned int AcquireStatus;
    enum AcquireStatusBits
    {
        AcquireFailed              = 0x1,  // could not fulfill allocation on all requested devices
        AcquireUsedTexHeapFallback = 0x2   // had to take the fallback path for a texheap allocation
    };

    /****************************************************************
   * Member data
   ****************************************************************/
    Context*       m_context       = nullptr;
    DeviceManager* m_deviceManager = nullptr;
    UpdateManager* m_updateManager = nullptr;

    // True during the interval between syncAllMemoryBeforeLaunch and
    // releaseMemoryAfterLaunch.
    bool m_launching = false;

    // True during the interval between enterFromAPI and exitToAPI
    bool m_enteredFromAPI = false;

    // MBuffer serial number used only for debug logging
    size_t m_nextSerialNumber = 1;

    // Device sets
    DeviceSet m_hostDevices;
    DeviceSet m_allDevices;
    DeviceSet m_activeDevices;
    DeviceSet m_allLwdaDevices;
    DeviceSet m_activeLwdaDevices;

    // Launch "timestamp" mechanism, which is used to ilwalidate all
    // output buffers with O(1) cost. Timestamp is incremented on
    // each launch.
    size_t m_lwrrentLaunchTimeStamp = 1;

    // Keep track of some basic statistics
    size_t m_statsTotalMem[OPTIX_MAX_DEVICES];
    size_t m_statsMaxUsedMem[OPTIX_MAX_DEVICES];

    // Buffer and texture lists, built using a simple class that
    // allows us to insert & remove from multiple lists in constant
    // time.
    IndexedVector<MBuffer*, MBuffer::masterListIndex_fn>               m_masterList;
    IndexedVector<MBuffer*, MBuffer::deferredAllocListIndex_fn>        m_deferredAllocList;
    IndexedVector<MBuffer*, MBuffer::deferredBackingAllocListIndex_fn> m_deferredBackingAllocList;
    IndexedVector<MBuffer*, MBuffer::deferredSyncListIndex_fn>         m_deferredSyncList;
    IndexedVector<MBuffer*, MBuffer::prelaunchListIndex_fn>            m_prelaunchList;
    IndexedVector<MBuffer*, MBuffer::postlaunchListIndex_fn>           m_postlaunchList;

    IndexedVector<MTextureSampler*, MTextureSampler::masterListIndex_fn>       m_masterTexList;
    IndexedVector<MTextureSampler*, MTextureSampler::deferredSyncListIndex_fn> m_deferredTexSyncList;

    // Resource manager (it is used exclusively by the memory manager, so it
    // is created here instead of in Context.
    std::unique_ptr<ResourceManager> m_resourceManager;

    /****************************************************************
   * Core functions
   ****************************************************************/

    // Allocate & free
    MBufferHandle finishAllocateMBuffer( std::unique_ptr<MBuffer>& mb );
    void freeMBuffer( MBuffer* mem );
    void freeMTextureSampler( MTextureSampler* tex );

    // Mapping to host
    char* mapToHostInternal( MBuffer* buf, MapMode mode );
    void unmapFromHostInternal( MBuffer* buf );
    char* getMappedToHostPtrInternal( MBuffer* buf );

    // "Automatically" update the buffer's valid set if a launch has happened
    // between the current and the previous call to this function. This
    // implements a lazy valid set update mechanism in order to avoid an O(n)
    // operation after each launch. To avoid stale data, any code that accesses
    // a buffer's valid set must first call this function.
    void updateValidSet( MBuffer* buf, const PolicyDetails& policy );

    // Use this function to change the values of the validSet. When buf uses zero copy
    // making one of the zero-copy devices valid makes them all valid.
    void setValidSet( MBuffer* buf, const DeviceSet& devices, const PolicyDetails& policy );


    // Allocate resources on all eligible requested devices. If the policy has
    // a fallback memory space ("preferXX"), it will automatically fall back to
    // that memory space if the allocation fails in the primary space.
    AcquireStatus acquireResourcesOnDevices( MBuffer* buf, const DeviceSet& onDevices, const PolicyDetails& policy );
    void acquireResourcesOnDevicesOrFail( MBuffer* buf, const DeviceSet& onDevices, const PolicyDetails& policy );

    // High-level resource management
    void changePolicy( MBuffer* buf, const PolicyDetails& oldPolicy, const PolicyDetails& newPolicy );

    void releaseResourcesOnDevices( MBuffer* buf, const DeviceSet& devices, const PolicyDetails& policy );
    void abandonResourcesOnDevices( MBuffer*         buf,
                                    const DeviceSet& devices,
                                    const DeviceSet& potentialTargets );  // Moves resources to another device before
                                                                          // releasing
    void lazyAllocateAndSync( MBuffer* buf, const PolicyDetails& policy );
    size_t promoteBufferToPeerToPeer( MBuffer* buf, const PolicyDetails& policy, int leastFreeMemDevIdx );

    void replaceHostData( MBuffer* buf, const PolicyDetails& oldPolicy, const PolicyDetails& newPolicy, const BufferDimensions& newSize, size_t copyBytes );
    void updatePoliciesAfterVariantChange( unsigned int oldVariant );
    void lazyResyncAll();
    void reevaluateArrayAssignments();
    void satisfyDeferredAllocations();
    void synchronizeBuffers();
    void synchronizeTextures();
    void resizeBackingStores();
    void resetTexHeapAllocations();
    bool selectBuffersForPeerToPeer( double forcePercent );

    // Texture management
    void releaseResourcesOnDevices( MTextureSampler* tex, const DeviceSet& devices, const PolicyDetails& policy );
    void lazySync( MTextureSampler* tex );
    void reserveHWTextureIfAvailable( MTextureSampler* tex, const DeviceSet& devices, const PolicyDetails& policy );
    bool reserveHWTexture( MTextureSampler* tex, unsigned int deviceIndex, const PolicyDetails& policy );
    void unreserveHWTexture( MTextureSampler* tex, unsigned int deviceIndex, const PolicyDetails& policy );
    unsigned int assignTexReference( unsigned int deviceIndex, MTextureSampler* tex );
    void unassignTexReference( unsigned int deviceIndex, unsigned int texUnit );
    void reevaluateArrayAssignments( unsigned int deviceIndex );
    void synchronizeTexToBacking( MTextureSampler* tex, const DeviceSet& activeSet );
    void synchronizeTexToBacking( MTextureSampler* tex, unsigned int deviceIndex, const PolicyDetails& policy );

    //
    // Synchronization and data transfer
    //

    // Ensures that all active devices that need a valid copy of the MBuffer data has one, if it exists.
    void synchronizeMBuffer( MBuffer* buf, bool manualSync );

    // Copies valid data from the validSet, if not empty, to the specified devices.
    void synchronizeToDevices( MBuffer* buf, const DeviceSet& devices, const PolicyDetails& policy );

    // Copies to the devices in dstSet from devices in srcSet (which are
    // assumed to have the identical data).
    void copyToDevices( MBuffer* buf, const DeviceSet& dstSet, const DeviceSet& srcSet );

    bool changePolicyLwdaAllocationWithinDevice( MBuffer* buf, const PolicyDetails& oldPolicy, const PolicyDetails& newPolicy );

    /****************************************************************
   * Device management
   ****************************************************************/
    struct PerDeviceInfo
    {
        // Management of LWCA textures
        bool                          useBindlessTexture           = false;
        bool                          reevaluateArrayAssignments   = false;
        int                           numAvailableBoundTextures    = 0;
        int                           numAvailableBindlessTextures = 0;
        std::vector<MTextureSampler*> assignedBoundTextures;
        BitSet                        assignedTextureUnits;
    };
    PerDeviceInfo m_pdi[OPTIX_MAX_DEVICES];
    void initializePerDeviceInfo( Device* device );
    void resetPerDeviceInfo( Device* device );


    /****************************************************************
   * Helper functions
   ****************************************************************/

    // List management functions
    void addLaunchHooks( MBuffer* buf, const PolicyDetails& policy );
    void removeLaunchHooks( MBuffer* buf );

    // Perform any necessary setup (e.g. mapping for graphics interop)
    // before an access.
    void preAccess( MBuffer* buf );
    void preAccess( std::vector<MBuffer*> buffers );  // batched version
    void preAccess_GFXINTEROP( MBuffer* buf, const PolicyDetails& policy, GfxInteropResourceBatch* gfxResourceBatch );
    void preAccess_LWDAINTEROP( MBuffer* buf, const PolicyDetails& policy );

    // Perform any necessary cleanup (e.g. unmap after graphics interop)
    // after access.
    void postAccess( MBuffer* buf );
    void postAccess( std::vector<MBuffer*> buffers );  // batched version
    void postAccess_GFXINTEROP( MBuffer* buf, const PolicyDetails& policy, GfxInteropResourceBatch* gfxResourceBatch );
    void postAccess_LWDAINTEROP( MBuffer* buf );

    // Optional verify for debug
    void verifyBeforeLaunch();

    // Debug & stats utilities
    void log( const MBuffer* mem, const std::string& desc ) const;
    void dump( const char* where ) const;
    void reportMemoryUsage();
    void trackMaxMemoryUsage();
    void printMaxMemoryUsage();
    void reserveMemoryForDebug();
    bool m_memoryForDebugReserved = false;

    // Policy helpers
    const PolicyDetails& getPolicyDetails( MBufferPolicy policy ) const;
    unsigned int computeLwrrentPolicyVariant() const;
    DeviceSet actualAllowedSet( MBuffer* buf, const PolicyDetails& policy ) const;

    /****************************************************************
   * Utilities used only by MBuffer and ResourceManager
   ****************************************************************/
    friend class MBuffer;
    friend class ResourceManager;
    Device* getDevice( unsigned int allDeviceIndex );
    const std::vector<MBuffer*>& getMasterList();
};
}  // namespace optix
