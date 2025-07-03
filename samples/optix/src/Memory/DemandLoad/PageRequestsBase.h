//
//  Copyright (c) 2020 LWPU Corporation.  All rights reserved.
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
#include <Memory/DemandLoad/RequestHandler/MipLevelRequestHandler.h>
#include <Memory/DemandLoad/RequestHandler/MipTailRequestHandler.h>
#include <Memory/DemandLoad/RequestHandler/TextureTileRequestHandler.h>

#include <vector>

namespace optix {

class BufferPageHeap;
class DeviceManager;

/// PageRequestsBase handles everything in the PageRequests interface except processRequests.
class PageRequestsBase : public PageRequests
{
  public:
    /// The PagingManager calls the constructor with a map from pageId to Buffer.
    PageRequestsBase( DeviceManager* dm, BufferPageHeap* bufferPages, bool usePerTileCallbacks );
    ~PageRequestsBase() override = default;

    /// Add an array of requested page ids from the specified device.
    void addRequests( unsigned int* requestedPages, unsigned int numRequests, unsigned int allDeviceListIndex ) override;

    /// Clear the page requests.
    void clear() override;

    /// Report metrics, e.g. number of tiles requested.
    void reportMetrics() override;

  protected:
    struct PerDeviceRequest
    {
        unsigned int pageId;
        unsigned int allDeviceListIndex;
    };

    DeviceManager*                             m_dm;
    std::vector<PerDeviceRequest>              m_perDeviceRequests;
    std::vector<TextureTileRequestHandler>     m_textureTileRequests;
    std::vector<BufferRequestHandler>          m_bufferPageRequests;
    std::vector<MipTailRequestHandler>         m_mipTailRequests;
    std::vector<MipLevelRequestHandler>        m_mipLevelRequests;
    std::vector<HardwareTileRequestHandler>    m_hardwareTileRequests;
    std::vector<HardwareMipTailRequestHandler> m_hardwareMipTailRequests;

    bool            m_usePerTileCallbacks;
    BufferPageHeap* m_bufferPages;
    int             m_callbackMsec = 0;

    void createRequestHandlers( PagingMode lwrrPagingMode );
    void makeLwrrent( unsigned int allDeviceIndex ) const;
    void reallocateMipLevels( std::vector<DevicePaging>& deviceState );
};

}  // namespace optix
