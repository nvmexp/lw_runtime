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

#include <Device/DeviceSet.h>
#include <Device/MaxDevices.h>
#include <Memory/MAccess.h>
#include <Memory/MBuffer.h>
#include <Util/TextureDescriptor.h>

#include <corelib/misc/Concepts.h>

#include <vector>

namespace optix {

class Device;
class MTextureSamplerListener;
class PagingService;

class MTextureSampler : private corelib::NonCopyable
{
  public:
    ~MTextureSampler();

    MAccess getAccess( int allDeviceIndex ) const;
    MAccess getAccess( unsigned int allDeviceListIndex ) const
    {
        return getAccess( static_cast<int>( allDeviceListIndex ) );
    }
    MAccess getAccess( const Device* device ) const;
    bool isDemandLoad( const Device* device ) const;

    void addListener( MTextureSamplerListener* listener );
    void removeListener( MTextureSamplerListener* );

    unsigned int getDemandLoadMinMipLevel( int allDeviceListIndex ) const;

    // Allocate a range of page ids.  Returns the start page, along with the number of pages as a
    // result parameter.
    unsigned int allocateVirtualPages( PagingService* pagingMgr, unsigned int& numPages );
    void releaseVirtualPages( PagingService* pagingMgr );

  private:
    // Client-visible data
    std::vector<MAccess> m_access;

    // Grows vector, but does not shrink
    void growAccess( int count );

    //
    // Private information used by MemoryManager
    //
    friend class MemoryManager;
    typedef std::vector<MTextureSamplerListener*> ListenerListType;
    ListenerListType                              m_listeners;
    MBufferHandle                                 m_backing;
    TextureDescriptor                             m_texDesc;
    DeviceSet                                     m_syncedToBacking;
    DeviceSet                                     m_hasHwTextureReserved;

    // Info for DemandTexObject; the first virtual page and page count are identical across all devices
    std::shared_ptr<size_t> m_startPage;
    size_t                  m_numPages = 0;

    int m_masterListIndex       = -1;
    int m_deferredSyncListIndex = -1;
    struct masterListIndex_fn
    {
        int& operator()( MTextureSampler* mts ) { return mts->m_masterListIndex; }
    };
    struct deferredSyncListIndex_fn
    {
        int& operator()( MTextureSampler* mts ) { return mts->m_deferredSyncListIndex; }
    };

    void setAccess( const Device* device, const MAccess& newAccess );

    MTextureSampler( const MBufferHandle& backing, const TextureDescriptor& texDesc );
};
}
