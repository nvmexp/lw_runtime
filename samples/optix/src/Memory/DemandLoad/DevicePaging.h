//
//  Copyright (c) 2018 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//
#pragma once

#include <LWCA/Event.h>
#include <LWCA/Stream.h>
#include <Memory/DemandLoad/HardwareTileManager.h>
#include <Memory/DemandLoad/TileLocator.h>
#include <Memory/DemandLoad/TileManager.h>
#include <Memory/DemandLoad/optixPaging/optixPaging.h>
#include <Memory/MBufferDeviceVector.h>
#include <Memory/MBufferListener.h>

#include <vector>

namespace optix {

class Buffer;
class LWDADevice;
class MemoryManager;
struct StagingPageAllocation;
class StagingPageAllocator;
class TileManager;

class DevicePaging : public MBufferListener
{
  public:
    DevicePaging()           = default;
    ~DevicePaging() override = default;

    DevicePaging( const DevicePaging& ) = delete;
    DevicePaging( DevicePaging&& )      = delete;
    void operator=( const DevicePaging& ) = delete;
    void operator=( DevicePaging&& ) = delete;

    void activate( LWDADevice* device, MemoryManager* mm, unsigned int allDeviceListIndex, unsigned int numVirtualPages );
    void deactivate();

    bool                isInitialized() const { return m_initialized; }
    unsigned int*       getUsageBits() const { return m_pagingContext->usageBits; }
    unsigned int*       getResidenceBits() const { return m_pagingContext->residenceBits; }
    unsigned long long* getPageTable() const { return m_pagingContext->pageTable; }
    unsigned long long* getTileArrays() const { return m_tileManager.getDevTexObjects( m_allDeviceListIndex ); }
    int                 getAllDeviceListIndex() const { return m_allDeviceListIndex; }

    lwca::Stream getCopyStream() const { return m_copyStream; }
    void pullRequests( std::vector<unsigned int>& requestedPages );
    void pushMappings();
    void synchronizeTiles();
    void addPageMapping( PageMapping mapping );
    void copyPageMappingsToDevice();
    void copyRequestedPagesToHost( std::vector<unsigned int>& requestedPages, unsigned int numRequests );
    void reallocDemandLoadLwdaArray( const MBufferHandle& buffer, int minLevel, int maxLevel );
    LWdeviceptr copyPageToDevice( StagingPageAllocator* stagingPages, const StagingPageAllocation& allocation, const void* data, bool synchronous );
    TileLocator copyTileToDevice( StagingPageAllocator*        stagingPages,
                                  const StagingPageAllocation& allocation,
                                  const RTmemoryblock&         memoryBlock,
                                  const Buffer*                buffer );
    TileLocator copyTileToDeviceAsync( StagingPageAllocator*        stagingPages,
                                       const StagingPageAllocation& allocation,
                                       const RTmemoryblock&         memoryBlock,
                                       const Buffer*                buffer );

    void syncDemandLoadMipLevel( StagingPageAllocator*        stagingPages,
                                 const StagingPageAllocation& allocation,
                                 const Buffer*                buffer,
                                 void*                        baseAddress,
                                 size_t                       byteCount,
                                 int                          mipLevel );
    void syncDemandLoadMipLevelAsync( StagingPageAllocator*        stagingPages,
                                      const StagingPageAllocation& allocation,
                                      const Buffer*                buffer,
                                      void*                        baseAddress,
                                      size_t                       byteCount,
                                      int                          mipLevel );
    void bindTileToMemory( StagingPageAllocator*        stagingPages,
                           const StagingPageAllocation& allocation,
                           const Buffer*                buffer,
                           const RTmemoryblock&         memBlock );
    void bindTileToMemoryAsync( StagingPageAllocator*        stagingPages,
                                const StagingPageAllocation& allocation,
                                const Buffer*                buffer,
                                const RTmemoryblock&         memBlock );
    void bindMipTailToMemory( const Buffer* buffer, int mipTailSizeInBytes );
    void bindMipTailToMemoryAsync( const Buffer* buffer, int mipTailSizeInBytes );
    void fillHardwareMipTail( StagingPageAllocator*        stagingPages,
                              const StagingPageAllocation& allocation,
                              const Buffer*                buffer,
                              const RTmemoryblock&         memBlock );
    void fillHardwareMipTailAsync( StagingPageAllocator*        stagingPages,
                                   const StagingPageAllocation& allocation,
                                   const Buffer*                buffer,
                                   const RTmemoryblock&         memBlock );

    void eventMBufferMAccessDidChange( const MBuffer* buffer, const Device* device, const MAccess& oldMBA, const MAccess& newMBA ) override;

  private:
    LWdeviceptr allocatePagePoolDeviceAddress();

    LWDADevice*                             m_device;
    MemoryManager*                          m_mm                 = nullptr;
    unsigned int                            m_allDeviceListIndex = ~0U;
    bool                                    m_initialized        = false;
    bool                                    m_havePageTable      = false;
    bool                                    m_haveUsageBits      = false;
    OptixPagingContext*                     m_pagingContext      = nullptr;
    OptixPagingSizes                        m_pagingSizes        = {};
    MBufferDeviceVector<unsigned long long> m_pageTable;
    MBufferDeviceVector<unsigned int>       m_usageBits;
    MBufferDeviceVector<unsigned long long> m_pagePool;
    unsigned int                            m_pagePoolCount = 0U;
    lwca::Stream                            m_copyStream;
    lwca::Event                             m_copyEvent;
    MBufferDeviceVector<unsigned int>       m_devRequestedPages;
    unsigned int                            m_numRequestedPages = 0U;
    MBufferDeviceVector<unsigned int>       m_devNumPagesReturned;
    MBufferDeviceVector<PageMapping>        m_devStalePages;
    unsigned int                            m_numStalePages = 0U;
    MBufferDeviceVector<unsigned int>       m_devEvictablePages;
    unsigned int                            m_numEvictablePages = 0U;
    MBufferDeviceVector<PageMapping>        m_devFilledPages;
    int                                     m_filledPageCount = 0;
    std::vector<PageMapping>                m_hostFilledPages;
    MBufferDeviceVector<unsigned int>       m_devIlwalidatedPages;
    int                                     m_ilwalidatedPageCount = 0;
    TileManager                             m_tileManager;
    HardwareTileManager                     m_hardwareTileManager;
};

}  // namespace optix
