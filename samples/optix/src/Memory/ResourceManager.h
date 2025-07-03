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

#include <AS/ASManager.h>
#include <LWCA/GraphicsResource.h>
#include <LWCA/TexRef.h>
#include <Memory/BulkMemoryPool.h>
#include <Memory/LightweightAllocator.h>
#include <Memory/MBuffer.h>
#include <Memory/MBufferListener.h>

#include <lwca.h>

#include <array>
#include <memory>
#include <mutex>
#include <vector>


namespace optix {

class Allocator;
class BufferDimensions;
class Context;
class LWDADevice;
class Device;
class DeviceManager;
class DeviceSet;
class MAccess;
class MemoryManager;
class MResources;
class PagingService;
struct GfxInteropResource;
struct GfxInteropResourceBatch;
struct PolicyDetails;
struct TextureDescriptor;

namespace lwca {
class Stream;
}  // namespace lwca

/*
 * Manager for low-level resources of different kinds. Used
 * exclusively by the MemoryManager.  Nothing else should include
 * this file.
 */
class ResourceManager : public MTextureSamplerListener
{
  public:
    static BufferDimensions actualDimsForDemandLoadTexture( const BufferDimensions& nominalDims,
                                                            unsigned int            minDemandLevel,
                                                            unsigned int            maxDemandLevel );

    // Disallow copying
    ResourceManager( const ResourceManager& ) = delete;
    ResourceManager& operator=( const ResourceManager& ) = delete;

    ~ResourceManager() override;
    ResourceManager( MemoryManager* mm, Context* context );

    // Finish initialization after MemoryManager is fully formed.
    void initialize();
    void shutdown();

    // Perform cleanup for resources owned by the ResourceManager for the removed devices, such as bulk allocation pools.
    void removedDevices( DeviceSet& removedDevices );

    // Acquire resources for the specific kind.  Will clear the bits in the
    // device set for which the allocation succeeded and leave the others
    // untouched.
    void acquireLwdaArrayOnDevices( MResources* res, DeviceSet& onDevices );
    void acquireLwdaSparseArrayOnDevices( MResources* res, DeviceSet& onDevices );
    void acquireLwdaMallocOnDevices( MResources* res, DeviceSet& onDevices, const PolicyDetails& policy );
    void acquireLwdaSingleCopyOnDevices( MResources* res, DeviceSet& onDevices, const PolicyDetails& policy );
    void acquireHostSingleCopyOnDevices( MResources* res, DeviceSet& onDevices, const PolicyDetails& policy );
    void acquireHostMallocOnDevices( MResources* res, DeviceSet& onDevices, const PolicyDetails& policy );
    void acquireTexHeapOnDevices( MResources* res, DeviceSet& onDevices );
    void acquireZeroCopyOnDevices( MResources* res, DeviceSet& onDevices, const PolicyDetails& policy );
    void acquireLwdaMallocP2POnDevices( MResources* res, DeviceSet& onDevices, const PolicyDetails& policy );
    void acquireLwdaArrayP2POnDevices( MResources* res, DeviceSet& onDevices );
    void acquireLwdaMallocExternalOnDevices( MResources* res, DeviceSet& onDevices, char* ptr );
    void acquireDemandLoadOnDevices( MResources* res, DeviceSet& onDevices );
    void acquireDemandLoadArrayOnDevices( MResources* res, DeviceSet& onDevices );
    void acquireDemandLoadTileArrayOnDevices( MResources* res, DeviceSet& onDevices );
    void acquireDemandLoadTileArraySparseOnDevices( MResources* res, DeviceSet& onDevices );

    void switchLwdaSparseArrayToLwdaArray( MResources* resources, DeviceSet devices );
    LWDA_ARRAY_SPARSE_PROPERTIES getSparseTexturePropertiesFromMBufferProperties( const MBuffer* buffer );
    void reallocDemandLoadLwdaArray( MResources* resources, unsigned int allDeviceListIndex, unsigned int minLevel, unsigned int maxLevel );
    void syncDemandLoadMipLevel( MResources* res, void* baseAddress, size_t byteCount, unsigned int allDeviceListIndex, int mipLevel );
    void syncDemandLoadMipLevelAsync( lwca::Stream& stream, MResources* res, void* baseAddress, size_t byteCount, unsigned int allDeviceListIndex, int mipLevel );
    void fillTile( MResources* resources, unsigned int allDeviceListIndex, unsigned int layer, const void* data );
    void fillTileAsync( lwca::Stream& stream, MResources* resources, unsigned int allDeviceListIndex, unsigned int layer, const void* data );
    void fillHardwareTileAsync( lwca::Stream&        stream,
                                MResources*          arrayResources,
                                MResources*          backingStoreResources,
                                unsigned int         allDeviceListIndex,
                                const RTmemoryblock& memBlock,
                                int                  offset );
    void bindHardwareMipTailAsync( lwca::Stream& stream,
                                   MResources*   arrayResources,
                                   MResources*   backingStoreResources,
                                   unsigned int  allDeviceListIndex,
                                   int           miptailSizeInBytes,
                                   int           offset );
    void fillHardwareMipTail( MResources* arrayResources, unsigned int allDeviceListIndex, const RTmemoryblock& memBlock );
    void fillHardwareMipTailAsync( lwca::Stream& stream, MResources* arrayResources, unsigned int allDeviceListIndex, const RTmemoryblock& memBlock );

    // Release resources associated with the specified devices
    void releaseResourcesOnDevices( MResources* res, const DeviceSet& onDevices, const PolicyDetails& policy );

    // Special functions used by the memory manager to move or copy
    // resources from one MResource to another
    void moveHostResources( MResources* dst, MResources* src );
    void moveDeviceResources( MResources* dst, MResources* src, LWDADevice* lwdaDevice );
    void copyResource( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );

    // Expand the valid set to include other devices that are also
    // valid. Used lwrrently for ZeroCopy.
    void expandValidSet( MResources* res, DeviceSet& validDevices );

    // If a device that holds a registration of ZeroCopy is disabled, we need
    // to move the memory registration to another valid device.
    void updateRegistrations( MResources* res, LWDADevice* newRegistrar );

    // Texture binding - used for both LwdaArray and pitched linear
    // resource kinds.  TexObjects are a resource that must be
    // released. A TexRef does not need to be released.
    lwca::TexObject createTexObject( const MResources* res, LWDADevice* lwdaDevice, const TextureDescriptor& texDesc );
    void bindTexReference( const MResources* res, LWDADevice* lwdaDevice, const TextureDescriptor& texDesc, lwca::TexRef texRef );

    // Functionality for graphics interop resources
    void mapGfxInteropResourceBatch( GfxInteropResourceBatch& gfxResourceBatch );
    void unmapGfxInteropResourceBatch( GfxInteropResourceBatch& gfxResourceBatch );
    void setupGfxInteropResource( MResources* res, const GfxInteropResource& resource, Device* interopDevice );
    Device* getGfxInteropDevice( MResources* res );
    void registerGfxInteropResource( MResources* res );
    void unregisterGfxInteropResource( MResources* res );
    void freeGfxInteropResource( MResources* res );
    void mapGfxInteropResource( MResources* res, const PolicyDetails& policy, GfxInteropResourceBatch* );
    void unmapGfxInteropResource( MResources* res, const PolicyDetails& policy, GfxInteropResourceBatch* );
    BufferDimensions queryGfxInteropResourceSize( MResources* res );
    bool doesGfxInteropResourceSizeNeedUpdate( MResources* res ) const;
    bool isGfxInteropResourceImmediate( MResources* res, const PolicyDetails& policy ) const;  // if there is no intermediate copy

    // TexHeap
    void getTexHeapSizeRequest( MBufferHandle* texHeapBacking, size_t* requestedSize );
    char* getTexHeapBackingPointer( Device* onDevice, const MAccess& access );
    bool isTexHeapEligibleForDefrag();
    void setTexHeapEnabled( bool enable );
    bool isTexHeapEnabled() const;

    // Return the set of devices for which we have an externally owned lwca malloc
    DeviceSet getLwdaMallocExternalSet( MResources* res ) const;

    // only for debugging
    void* reallocateMemoryForDebug( void* ptr, size_t size );

  private:
    // Demand load methods.
    LWDA_MEMCPY2D getSyncDemandLoadMipLevelCopyArgs( MResources*  res,
                                                     void*        baseAddress,
                                                     size_t       byteCount,
                                                     unsigned int allDeviceListIndex,
                                                     int          nominalMipLevel ) const;
    LWDA_MEMCPY3D getFillTileCopyArgs( MResources* resources, unsigned int allDeviceListIndex, unsigned int layer, const void* data ) const;
    LWarrayMapInfo getFillHardwareTileBindInfo( MResources*          arrayResources,
                                                MResources*          backingStoreResources,
                                                unsigned int         allDeviceListIndex,
                                                const RTmemoryblock& memBlock,
                                                int                  offset ) const;
    LWDA_MEMCPY2D getFillHardwareTileCopyArgs( MResources*          arrayResources,
                                               unsigned int         allDeviceListIndex,
                                               const RTmemoryblock& memBlock ) const;
    LWarrayMapInfo getBindHardwareMipTailBindInfo( MResources*  arrayResources,
                                                   MResources*  backingStoreResources,
                                                   unsigned int allDeviceListIndex,
                                                   int          miptailSizeInBytes,
                                                   int          offset ) const;
    LWDA_MEMCPY2D getFillHardwareMipTailCopyArgs( MResources*          arrayResources,
                                                  unsigned int         allDeviceListIndex,
                                                  const RTmemoryblock& memBlock ) const;

    // LwdaArray functions
    bool createLwdaArray( MResources* res, const BufferDimensions& dims, int onDevice );
    bool createSparseLwdaArray( MResources* res, const BufferDimensions& dims, int onDevice );
    void destroyLwdaArray( MResources* res, int onDevice );
    bool acquireLwdaArray( MResources* res, int onDevice, bool p2p );
    void releaseLwdaArray( MResources* res, int onDevice );
    void unbindLwdaSparseArray( MResources* res, int onDevice );
    void releaseLwdaSparseArray( MResources* res, int onDevice );
    void releaseLwdaArrayP2POnDevices( MResources* res, const DeviceSet& onDevices );

    bool createLwdaSparseBacking( MResources* res, const BufferDimensions& dims, int onDevice );

    // LwdaMalloc functions and resources
    bool acquireLwdaMalloc( MResources* res, int onDevice, const PolicyDetails& policy );
    void releaseLwdaMalloc( MResources* res, int onDevice );
    void releaseLwdaMallocP2POnDevices( MResources* res, const DeviceSet& onDevices );
    size_t getAllocationThreshold();
    void lazyInitializeBulkMemoryPools( unsigned int allDeviceIndex );
    LWdeviceptr bulkAllocate( int onDevice, size_t nbytes, size_t alignment, bool deviceIsHost );
    void bulkFree( int onDevice, LWdeviceptr ptr, size_t nbytes, size_t alignment, bool deviceIsHost );

    std::array<BulkMemoryPool, OPTIX_MAX_DEVICES> m_bulkMemoryPools_small;
    std::array<BulkMemoryPool, OPTIX_MAX_DEVICES> m_bulkMemoryPools_large;

    // Demand load functions and data
    void releaseDemandLoadOnDevices( MResources* res, const DeviceSet& onDevices );
    void releaseDemandLoadArray( MResources* res, int onDevice );
    void releaseDemandLoadTileArray( MResources* res, int onDevice );
    void releaseLwdaSparseBacking( MResources* res, int onDevice );

    // GfxInterop functions and data
    lwca::GraphicsResource registerGLResource( const GfxInteropResource& resource, const PolicyDetails& policy );

    // HostMalloc functions
    bool acquireHostMalloc( MResources* res, int onDevice, const PolicyDetails& policy );
    void releaseHostMalloc( MResources* res, int onDevice );

    // TexHeap functions and resources
    void releaseTexHeapOnDevices( MResources* res, const DeviceSet& onDevices );
    bool reserveTexHeap( MResources* res, const BufferDimensions& dims );
    // float   texHeapPercentFull() const;
    MAccess makeTexHeapAccess( const MAccess& backingAccess, unsigned int offset );
    std::unique_ptr<Allocator> m_texHeapAllocator;
    MBufferHandle              m_texHeapBacking;  // buffer containing the heap
    MTextureSamplerHandle      m_texHeapSampler;  // texture object
    bool   m_texHeapEnabled   = true;  // this can change dynamically, e.g. if there is no room left in texture heap
    size_t m_texHeapTotalSize = 0;     // Total size of tex heap if all allocations are fulfilled

    // ZeroCopy functions and resources
    void releaseZeroCopyOnDevices( MResources* res, const DeviceSet& onDevices );

    // Single copy functions
    LWdeviceptr allocateLwdaSingleCopy( MResources* res, const DeviceSet& possibleDevices, const PolicyDetails& policy, const size_t nbytes );
    void releaseLwdaSingleCopyOnDevices( MResources* res, const DeviceSet& onDevices, const PolicyDetails& policy );

    // Utility functions
    Device* getDevice( unsigned int index ) const;
    LWDADevice* getLWDADevice( unsigned int index ) const;
    bool shouldAllocateWritableBufferOnDevice() const;

    // This class is also a listener for access event changes,
    // lwrrently used only for the sampler associated with the texheap
    // parent buffer.
    void eventMTextureSamplerMAccessDidChange( const Device* device, const MAccess& oldMBA, const MAccess& newMBA ) override;

    // Copy functions for various cases. Can be found in ResourceCopying.cpp
    void copyMemcpy( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copyDtoD( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copyDtoD_th( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims );
    void copyArraytoArray( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copyDtoArray( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims );
    void copyArraytoD( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims );
    void copyDtoH( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copyArraytoH( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copyDtoH_th( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copyDtoH_zc( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copyHtoD( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copyHtoArray( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copyHtoD_th( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copyHtoD_zc( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copyTwoStage( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copyZeroCopy( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copyHtoP2P( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copyP2PtoH( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims );
    void copyP2PtoP2P( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims );
    void copyArrayP2PtoArrayP2P( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims );
    void copyHtoArrayP2P( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copyIlwalid( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& size );
    void copySCtoH( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims );
    void copyHtoSC( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims );
    void copySCtoSC( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims );

    // Low-level copy helpers
    void copyHtoD_common( char* dstPtr, Device* dstDevice, char* srcPtr, Device* srcDevice, const BufferDimensions& size );
    void copyDtoH_common( char* dstPtr, Device* dstDevice, char* srcPtr, Device* srcDevice, const BufferDimensions& size );

    // Helper that clears all bulk memory pools for the provided device index.
    void freeBulkMemoryPools( int allDeviceIndex );

    // Private data
    Context*       m_context       = nullptr;
    MemoryManager* m_memoryManager = nullptr;
    DeviceManager* m_deviceManager = nullptr;
};
}
