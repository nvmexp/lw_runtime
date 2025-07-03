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

#include <o6/optix.h>

#include <Device/DeviceSet.h>
#include <Device/MaxDevices.h>
#include <Memory/BufferDimensions.h>
#include <Memory/MAccess.h>
#include <Memory/MBufferPolicy.h>
#include <Memory/MResources.h>
#include <Util/IndexedVector.h>

#include <lwca.h>

#include <memory>
#include <set>
#include <vector>


namespace optix {

class Device;
class MBuffer;
class MBufferListener;
class MemoryManager;
class MTextureSampler;
class MResources;
struct GfxInteropResource;

using MBufferHandle         = std::shared_ptr<MBuffer>;
using MTextureSamplerHandle = std::shared_ptr<MTextureSampler>;

/***********************************************************************
 * An MBuffer object manages the allocations for OptiX buffers and
 * internal buffers.  The memory manager will keep per-device copies
 * in sync.
 ***********************************************************************/
class MBuffer
{
  public:
    MBuffer( const MBuffer& ) = delete;
    MBuffer& operator=( const MBuffer& ) = delete;

    ~MBuffer();

    const BufferDimensions& getDimensions() const;
    MBufferPolicy           getPolicy() const;
    MAccess getAccess( int allDeviceIndex ) const;
    MAccess getAccess( unsigned int allDeviceListIndex ) const
    {
        return getAccess( static_cast<int>( allDeviceListIndex ) );
    }
    MAccess getAccess( const Device* device ) const;
    const GfxInteropResource& getGfxInteropResource() const;

    void updateDemandLoadArrayAccess( int allDeviceListIndex );
    LWDA_ARRAY_SPARSE_PROPERTIES getSparseTextureProperties( int allDeviceListIndex ) const;
    LWDA_ARRAY_SPARSE_PROPERTIES getSparseTextureProperties() const;

    void addListener( MBufferListener* listener );
    void removeListener( MBufferListener* );

    // Returns true if the buffer is allocated on slow zero copy memory on the provided device.
    bool isZeroCopy( int allDeviceIndex ) const;

    unsigned int getDemandLoadStartPage() const;
    void switchLwdaSparseArrayToLwdaArray( DeviceSet devices );

private:
    MBufferPolicy    m_policy;  // Buffer management policy
    BufferDimensions m_dims;    // Size/shape of the memory region

    std::vector<MAccess> m_access;  // Device-specific pointers

    // Grows vector, but does not shrink
    void growAccess( int count );

    MemoryManager* m_memoryManager = nullptr;

    //
    // Private information used by MemoryManager and MResources
    //
    friend class MemoryManager;
    friend class ResourceManager;
    friend class MResources;

    using ListenerListType = std::vector<MBufferListener*>;
    ListenerListType m_listeners;
    bool             m_notifyingListeners = false;

    DeviceSet m_allocatedSet;  // Devices for which storage has been allocated, either internally or externally. The
                               // MAccess type for an allocated device should not be NONE.
    DeviceSet m_validSet;      // Devices with an up-to-date copy of the data (updated lazily after a launch)
    DeviceSet m_allowedSet;    // Devices for which storage can be allocated
    DeviceSet m_frozenSet;     // Devices for which a pointer has been released outside of the API, so the value is not
                               // allowed to change
    DeviceSet m_pinnedSet;     // Device for which the buffer is pinned. This set can only contain 0 or 1 devices

    bool   m_mappedToHost      = false;  // if any MIP level is mapped to host
    bool   m_p2pRequested      = false;  // whether the next allocation should go to peer-to-peer
    size_t m_validSetTimestamp = 0;      // when the valid set was last updated
    size_t m_serialNumber      = 0;      // Used for debug logs

    // Return true if the given devices contains a peer-to-peer allocation, false otherwise.
    bool isAllocatedP2P( int allDeviceIndex ) const;

    std::set<MTextureSampler*> m_attachedTextureSamplers;  // Texture samplers for which this MBuffers serves as backing
                                                           // store

    // Low-level resources
    std::unique_ptr<MResources> m_resources;

    // For each type of list, we need an index and a functor to
    // retrieve the index. -1 indicates that this item is not in the
    // specified list.
    int m_masterListIndex               = -1;
    int m_deferredAllocListIndex        = -1;
    int m_deferredBackingAllocListIndex = -1;
    int m_deferredSyncListIndex         = -1;
    int m_demandLoadListIndex           = -1;
    int m_prelaunchListIndex            = -1;
    int m_postlaunchListIndex           = -1;

    struct masterListIndex_fn
    {
        int& operator()( MBuffer* mb ) { return mb->m_masterListIndex; }
    };
    struct deferredAllocListIndex_fn
    {
        int& operator()( MBuffer* mb ) { return mb->m_deferredAllocListIndex; }
    };
    struct deferredBackingAllocListIndex_fn
    {
        int& operator()( MBuffer* mb ) { return mb->m_deferredBackingAllocListIndex; }
    };
    struct deferredSyncListIndex_fn
    {
        int& operator()( MBuffer* mb ) { return mb->m_deferredSyncListIndex; }
    };
    struct prelaunchListIndex_fn
    {
        int& operator()( MBuffer* mb ) { return mb->m_prelaunchListIndex; }
    };
    struct postlaunchListIndex_fn
    {
        int& operator()( MBuffer* mb ) { return mb->m_postlaunchListIndex; }
    };
    struct demandLoadListIndex_fn
    {
        int& operator()( MBuffer* mb ) { return mb->m_demandLoadListIndex; }
    };

    // Buffers may use another buffer as backing store, e.g. the texheap.
    void attachBackingStore( const MBufferHandle& backing );
    void          detachBackingStore();
    MBufferHandle m_backing;

    // If this buffer is a backing store, this list keeps track of
    // all the buffers that use it.
    int m_backingClientsListIndex = -1;
    struct backingClientsListIndex_fn
    {
        int& operator()( MBuffer* mb ) { return mb->m_backingClientsListIndex; }
    };
    IndexedVector<MBuffer*, MBuffer::backingClientsListIndex_fn> m_backingClientsList;

    //
    // Private methods used by MemoryManager
    //
    void setAccess( int allDeviceIndex, const MAccess& newAccess );
    void setAccess( const Device* device, const MAccess& newAccess );

    char* getDevicePtr( Device* device );

    MBuffer( MemoryManager* manager, const BufferDimensions& size, MBufferPolicy policy, const DeviceSet& allowedSet );
    MBuffer( MemoryManager*          manager,
             const BufferDimensions& size,
             MBufferPolicy           policy,
             const DeviceSet&        allowedSet,
             RTbuffercallback        callback,
             void*                   callbackData );
    BufferDimensions getNonZeroDimensions() const;

    std::string stateString() const;
};

}  // namespace optix
