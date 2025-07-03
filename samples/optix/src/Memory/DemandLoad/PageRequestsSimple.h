//
//  Copyright (c) 2019 LWPU Corporation.  All rights reserved.
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

#include <Memory/DemandLoad/PageRequests.h>
#include <Memory/DemandLoad/RequestHandler/BufferRequestHandler.h>
#include <Memory/DemandLoad/RequestHandler/HardwareMipTailRequestHandler.h>
#include <Memory/DemandLoad/RequestHandler/HardwareTileRequestHandler.h>
#include <Memory/DemandLoad/RequestHandler/MipTailRequestHandler.h>
#include <Memory/DemandLoad/RequestHandler/TextureTileRequestHandler.h>

#include <vector>

namespace optix {

class ApiCapture;
class BufferPageHeap;
class DevicePaging;
class RequestHandler;
class StagingPageAllocator;
class ThreadPool;

/// PageRequestsSimple processes requests by ilwoking all the callbacks on multiple
/// thread first, then doing all copying from staging memory to device memory on the main
/// thread.  The copies are always synchronous because the corresponding StagingPageAllocator
/// is StagingPageAllocatorSimple which always uses plain host memory.  Asynchronous copies
/// require pinned memory as the source.
class PageRequestsSimple : public PageRequests
{
  public:
    /// The PagingManager calls the constructor with a map from pageId to Buffer.
    PageRequestsSimple( ThreadPool* threadPool, BufferPageHeap* bufferPages, bool usePerTileCallbacks, bool multiThreadedCallbacksEnabled );
    virtual ~PageRequestsSimple() = default;

    /// Add an array of requested page ids from the specified device.
    void addRequests( unsigned int* requestedPages, unsigned int numRequests, unsigned int allDeviceListIndex ) override;
    void synchronizeAllDevices( StagingPageAllocator* stagingPages, std::vector<DevicePaging>& deviceState, PagingMode lwrrPagingMode );

    void processRequests( PagingMode                 lwrrPagingMode,
                          StagingPageAllocator*      stagingPages,
                          ApiCapture&                apiCapture,
                          std::vector<DevicePaging>& deviceState ) override;

    /// Clear the page requests.
    void clear() override;

    /// Report metrics, e.g. number of tiles requested.
    void reportMetrics() override;

  private:
    struct PerDeviceRequest
    {
        unsigned int pageId;
        unsigned int allDeviceListIndex;
    };

    void createRequestHandlers( PagingMode lwrrPagingMode );
    void reallocateMipLevels( std::vector<DevicePaging>& deviceState );

    /// Fill memory blocks for each page request (sequentially), dispatching to RequestHandler::ilwoke().
    void fillMemoryBlocks( StagingPageAllocator* stagingPages, ApiCapture& apiCapture );

    /// Fill memory blocks for each page request (in parallel), dispatching to RequestHandler::ilwoke().
    void fillMemoryBlocksThreaded( StagingPageAllocator* stagingPages, ApiCapture& apiCapture );

    /// Synchronize page data for each page request to the specified device,
    /// dispatching to RequestHandler::synchronize().
    void synchronize( StagingPageAllocator* stagingPages, unsigned int allDeviceListIndex, DevicePaging& devicePaging );

    void releaseMemoryBlocks( StagingPageAllocator* stagingPages );

    ThreadPool*                                m_threadPool;
    BufferPageHeap*                            m_bufferPages;
    std::vector<PerDeviceRequest>              m_perDeviceRequests;
    std::vector<TextureTileRequestHandler>     m_textureTileRequests;
    std::vector<BufferRequestHandler>          m_bufferPageRequests;
    std::vector<MipTailRequestHandler>         m_mipTailRequests;
    std::vector<MipLevelRequestHandler>        m_mipLevelRequests;
    std::vector<HardwareTileRequestHandler>    m_hardwareTileRequests;
    std::vector<HardwareMipTailRequestHandler> m_hardwareMipTailRequests;
    bool                                       m_usePerTileCallbacks;
    bool                                       m_multiThreadedCallbacksEnabled;
    int                                        m_callbackMsec = 0;
    int                                        m_copyMsec     = 0;
};

}  // namespace optix
