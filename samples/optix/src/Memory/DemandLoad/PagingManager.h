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

#include <Context/UpdateManager.h>
#include <Device/DeviceSet.h>
#include <Memory/DemandLoad/BufferPageHeap.h>
#include <Memory/DemandLoad/DevicePaging.h>
#include <Memory/DemandLoad/PageRequests.h>
#include <Memory/DemandLoad/PagingService.h>
#include <Memory/DemandLoad/StagingPageAllocator.h>
#include <Memory/DemandLoad/TileManager.h>
#include <Memory/MBufferDeviceVector.h>

#include <memory>
#include <string>
#include <vector>

#include <stdint.h>

namespace optix {

class BufferDimensions;
class Context;
class Device;

class PagingManager : public PagingService, public UpdateEventListenerNop
{
  public:
    PagingManager( Context* context, bool ilwokeCallbacksPerTileNotPerMipLevel, unsigned int numVirtualPages );
    ~PagingManager() override = default;

    void enable() override;

    void tearDown() override;

    void launchPrepare( const DeviceSet& devices ) override;
    void launchComplete() override;

    std::shared_ptr<size_t> reservePageTableEntries( size_t numPages ) override;
    void releasePageTableEntries( size_t startPage, size_t numPages ) override;

    void bufferMAccessDidChange( Buffer* buffer, const Device* device, const MAccess& oldMA, const MAccess& newMA ) override;
    void eventTextureSamplerMAccessDidChange( const TextureSampler* sampler,
                                              const Device*         device,
                                              const MAccess&        oldMA,
                                              const MAccess&        newMA ) override;


    unsigned int getSoftwareMipTailFirstLevel( const BufferDimensions& dims ) const override;
    uint3 getPageDimensions( const BufferDimensions& dims ) const override;
    size_t computeNumDemandBufferPages( const BufferDimensions& dims ) const override;
    size_t computeSoftwareNumDemandTexturePages( const BufferDimensions& dims ) const override;
    size_t computeNumPages( const BufferDimensions& dims, unsigned int tileWidth, unsigned int tileHeight, unsigned int mipTailFirstLevel ) const override;

    unsigned int* getUsageBits( const Device* device ) const override;
    unsigned int* getResidenceBits( const Device* device ) const override;
    unsigned long long* getPageTable( const Device* device ) const override;
    unsigned long long* getTileArrays( const Device* device ) const override;

    void preSetActiveDevices( const DeviceSet& removedDevices ) override;

    bool getMultiThreadedCallbacksEnabled() const override;
    void setMultiThreadedCallbacksEnabled( bool enabled ) override;

    void setLwdaSparseTexturesEnabled( bool enabled ) override;

    PagingMode getLwrrentPagingMode() const override;

    unsigned int getTileWidth() const override;
    unsigned int getTileHeight() const override;

    void forceSynchronousRequestsChanged() override;
    void useAsynchronousCopiesChanged() override;
    void stampMemoryBlocksChanged() override;

    // Callback ilwoked when LWCA sparse texture support changes.
    void eventActiveDevicesSupportLwdaSparseTexturesDidChange( bool supported ) override;

  private:
    Context*                              m_context;
    std::vector<DevicePaging>             m_deviceState;
    DeviceSet                             m_launchDevices;
    BufferPageHeap                        m_heap;
    std::unique_ptr<StagingPageAllocator> m_stagingPages;
    bool                                  m_isActive                             = false;
    bool                                  m_multiThreadedCallbacksEnabled        = false;
    bool                                  m_ilwokeCallbacksPerTileNotPerMipLevel = true;

    bool m_activeDevicesSupportLwdaSparse;
    bool m_lwdaSparseTexturesEnabled = false;

    std::vector<unsigned int>     m_rawRequests;
    std::unique_ptr<PageRequests> m_pageRequests;

    unsigned int m_numVirtualPages;
    int          m_copyMsec = 0;

    void activateDevices();

    bool getAsynchronousCopies() const;
    void resetStagingPageAllocator( Context* context );
    void createPageRequests();

    void pullRequests();
    void processRequests();
    void reportMetrics() const;

    void processPageRequests();
    void copyPageMappingsToDevices();
};

}  // namespace optix
