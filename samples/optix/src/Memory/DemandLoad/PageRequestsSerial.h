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

#include <Memory/DemandLoad/PageRequestsBase.h>

namespace optix {

/// PageRequestsSerial processes requests on the main thread with synchronous or asynchronous
/// copies from host memory to device memory.
class PageRequestsSerial : public PageRequestsBase
{
  public:
    /// The PagingManager calls the constructor with a map from pageId to Buffer.
    PageRequestsSerial( DeviceManager* dm, BufferPageHeap* bufferPages, bool usePerTileCallbacks, bool synchronous );
    ~PageRequestsSerial() override = default;

    void processRequests( PagingMode                 lwrrPagingMode,
                          StagingPageAllocator*      stagingPages,
                          ApiCapture&                apiCapture,
                          std::vector<DevicePaging>& deviceState ) override;

  private:
    bool m_synchronous;
};

}  // namespace optix
