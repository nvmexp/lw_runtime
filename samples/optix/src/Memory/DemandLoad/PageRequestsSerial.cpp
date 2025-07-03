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

#include <Memory/DemandLoad/PageRequestsSerial.h>

#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Util/ElapsedTimeCapture.h>

#include <prodlib/exceptions/IlwalidOperation.h>

#include <chrono>

using namespace prodlib;

namespace optix {

namespace {

template <typename Request>
void processRequestImpl( DeviceManager*             dm,
                         std::vector<Request>&      requests,
                         StagingPageAllocator*      stagingPages,
                         std::vector<DevicePaging>& deviceState,
                         ApiCapture&                apiCapture,
                         bool                       synchronous )
{
    for( Request& request : requests )
    {
        request.processRequest( dm, stagingPages, deviceState, synchronous );
        if( request.caughtException() )
            throw IlwalidOperation( RT_EXCEPTION_INFO, request.getExceptionMessage() );
        request.captureTrace( apiCapture );
    }
}

}  // anonymous namespace

PageRequestsSerial::PageRequestsSerial( DeviceManager* dm, BufferPageHeap* bufferPages, bool usePerTileCallbacks, bool synchronous )
    : PageRequestsBase( dm, bufferPages, usePerTileCallbacks )
    , m_synchronous( synchronous )
{
}

void PageRequestsSerial::processRequests( PagingMode                 lwrrPagingMode,
                                          StagingPageAllocator*      stagingPages,
                                          ApiCapture&                apiCapture,
                                          std::vector<DevicePaging>& deviceState )
{
    ElapsedTimeCapture elapsed( m_callbackMsec );

    createRequestHandlers( lwrrPagingMode );
    reallocateMipLevels( deviceState );

    // Dispatch callbacks from the main thread.
    LOG_NORMAL( "PageRequestsSerial::processRequests(): "
                << m_bufferPageRequests.size() << " buffer, " << m_mipLevelRequests.size() << " mip level, "
                << m_textureTileRequests.size() << " texture tile, " << m_mipTailRequests.size() << " mip tail,"
                << m_hardwareTileRequests.size() << " HW tile, " << m_hardwareMipTailRequests.size()
                << " HW mip tail requests\n" );
    processRequestImpl( m_dm, m_bufferPageRequests, stagingPages, deviceState, apiCapture, m_synchronous );
    processRequestImpl( m_dm, m_mipLevelRequests, stagingPages, deviceState, apiCapture, m_synchronous );
    processRequestImpl( m_dm, m_textureTileRequests, stagingPages, deviceState, apiCapture, m_synchronous );
    processRequestImpl( m_dm, m_mipTailRequests, stagingPages, deviceState, apiCapture, m_synchronous );
    processRequestImpl( m_dm, m_hardwareTileRequests, stagingPages, deviceState, apiCapture, m_synchronous );
    processRequestImpl( m_dm, m_hardwareMipTailRequests, stagingPages, deviceState, apiCapture, m_synchronous );
}

}  // namespace optix
