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

#include <Memory/DemandLoad/PagingService.h>

#include <vector>

namespace optix {

class ApiCapture;
class DevicePaging;
class RequestHandler;
class StagingPageAllocator;

/// PageRequests stores and processes page requests in collaboration with RequestHandler.
class PageRequests
{
  public:
    virtual ~PageRequests() = default;

    /// Add an array of requested page ids from the specified device.
    virtual void addRequests( unsigned int* requestedPages, unsigned int numRequests, unsigned int allDeviceListIndex ) = 0;

    /// Ilwoke callback for each page request (sequentially), dispatching to RequestHandler::ilwoke().
    virtual void processRequests( PagingMode                 lwrrPagingMode,
                                  StagingPageAllocator*      stagingPages,
                                  ApiCapture&                apiCapture,
                                  std::vector<DevicePaging>& deviceState ) = 0;

    /// Clear the page requests.
    virtual void clear() = 0;

    /// Report metrics, e.g. number of tiles requested.
    virtual void reportMetrics() = 0;
};

}  // namespace optix
