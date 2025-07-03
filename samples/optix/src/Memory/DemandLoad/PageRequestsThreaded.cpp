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

#include <Memory/DemandLoad/PageRequestsThreaded.h>

#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Memory/DemandLoad/BufferPageHeap.h>
#include <Memory/DemandLoad/DevicePaging.h>
#include <Memory/DemandLoad/PageRequests.h>
#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Memory/DemandLoad/StagingPageAllocator.h>
#include <Objects/TextureSampler.h>
#include <ThreadPool/Job.h>
#include <ThreadPool/ThreadPool.h>
#include <Util/ContainerAlgorithm.h>
#include <Util/ElapsedTimeCapture.h>
#include <Util/Metrics.h>
#include <c-api/ApiCapture.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidOperation.h>
#include <prodlib/system/Logger.h>

#include <algorithm>
#include <iostream>

using namespace prodlib;
using namespace optix::demandLoad;

namespace optix {

// ---------- Threaded callback invocation ----------

namespace {

class IlwokeCallbacksJob : public FragmentedJob
{
  public:
    IlwokeCallbacksJob( DeviceManager*                dm,
                        std::vector<RequestHandler*>& pageRequests,
                        StagingPageAllocator*         stagingPages,
                        std::vector<DevicePaging>&    deviceState,
                        bool                          synchronous )
        : FragmentedJob( pageRequests.size() )
        , m_dm( dm )
        , m_requests( pageRequests )
        , m_stagingPages( stagingPages )
        , m_deviceState( deviceState )
        , m_synchronous( synchronous )
    {
    }

    void exelwteFragment( size_t index, size_t count ) noexcept override
    {
        m_requests[index]->processRequest( m_dm, m_stagingPages, m_deviceState, m_synchronous );
    }

  private:
    DeviceManager*                m_dm;
    std::vector<RequestHandler*>& m_requests;
    StagingPageAllocator*         m_stagingPages;
    std::vector<DevicePaging>&    m_deviceState;
    bool                          m_synchronous;
};

}  // namespace

PageRequestsThreaded::PageRequestsThreaded( DeviceManager* dm, ThreadPool* threadPool, BufferPageHeap* bufferPages, bool usePerTileCallbacks, bool synchronous )
    : PageRequestsBase( dm, bufferPages, usePerTileCallbacks )
    , m_threadPool( threadPool )
    , m_synchronous( synchronous )
{
    RT_ASSERT( dm != nullptr );
    RT_ASSERT( threadPool != nullptr );
    RT_ASSERT( bufferPages != nullptr );
}

void PageRequestsThreaded::processRequests( PagingMode                 lwrrPagingMode,
                                            StagingPageAllocator*      stagingPages,
                                            ApiCapture&                apiCapture,
                                            std::vector<DevicePaging>& deviceState )
{
    LOG_NORMAL( "PageRequestsThreaded::processRequests mode " << toString( lwrrPagingMode ) << '\n' );
    ElapsedTimeCapture elapsed( m_callbackMsec );

    createRequestHandlers( lwrrPagingMode );
    if( m_textureTileRequests.empty() && m_bufferPageRequests.empty() && m_mipTailRequests.empty()
        && m_mipLevelRequests.empty() && m_hardwareTileRequests.empty() && m_hardwareMipTailRequests.empty() )
    {
        return;
    }

    reallocateMipLevels( deviceState );

    // Gather request handlers in a single vector for maximum parallelism.
    static std::vector<RequestHandler*> requests;
    gatherRequests( requests );

    // Dispatch callbacks in parallel.
    const std::shared_ptr<IlwokeCallbacksJob> ilwokeCallbacksJob(
        std::make_shared<IlwokeCallbacksJob>( m_dm, requests, stagingPages, deviceState, m_synchronous ) );
    m_threadPool->submitJobAndWait( ilwokeCallbacksJob );

    const auto excepter =
        algorithm::find_if( requests, []( const RequestHandler* request ) { return request->caughtException(); } );
    if( excepter != requests.end() )
        throw IlwalidOperation( RT_EXCEPTION_INFO, ( *excepter )->getExceptionMessage() );

    // Record API call trace (sequentially).
    for( RequestHandler* request : requests )
    {
        request->captureTrace( apiCapture );
    }
}

void PageRequestsThreaded::gatherRequests( std::vector<RequestHandler*>& requests )
{
    requests.clear();
    requests.reserve( m_textureTileRequests.size() + m_bufferPageRequests.size() + m_mipTailRequests.size()
                      + m_mipLevelRequests.size() + m_hardwareTileRequests.size() + m_hardwareMipTailRequests.size() );
    for( RequestHandler& request : m_textureTileRequests )
        requests.push_back( &request );
    for( RequestHandler& request : m_bufferPageRequests )
        requests.push_back( &request );
    for( RequestHandler& request : m_mipTailRequests )
        requests.push_back( &request );
    for( RequestHandler& request : m_mipLevelRequests )
        requests.push_back( &request );
    for( RequestHandler& request : m_hardwareTileRequests )
        requests.push_back( &request );
    for( RequestHandler& request : m_hardwareMipTailRequests )
        requests.push_back( &request );
}

}  // namespace optix
