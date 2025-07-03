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

#include <Memory/DemandLoad/PageRequestsSimple.h>

#include <Memory/DemandLoad/BufferPageHeap.h>
#include <Memory/DemandLoad/DevicePaging.h>
#include <Memory/DemandLoad/PageRequests.h>
#include <Memory/DemandLoad/PagingService.h>
#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Memory/DemandLoad/StagingPageAllocatorSimple.h>
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

namespace optix {

PageRequestsSimple::PageRequestsSimple( ThreadPool* threadPool, BufferPageHeap* bufferPages, bool usePerTileCallbacks, bool multiThreadedCallbacksEnabled )
    : m_threadPool( threadPool )
    , m_bufferPages( bufferPages )
    , m_usePerTileCallbacks( usePerTileCallbacks )
    , m_multiThreadedCallbacksEnabled( multiThreadedCallbacksEnabled )
{
    RT_ASSERT( threadPool != nullptr );
    RT_ASSERT( bufferPages != nullptr );
}

void PageRequestsSimple::clear()
{
    m_perDeviceRequests.clear();
    m_textureTileRequests.clear();
    m_bufferPageRequests.clear();
    m_mipTailRequests.clear();
    m_mipLevelRequests.clear();
    m_hardwareTileRequests.clear();
    m_hardwareMipTailRequests.clear();
    m_callbackMsec = 0;
    m_copyMsec     = 0;
}

void PageRequestsSimple::addRequests( unsigned int* requestedPages, unsigned int numRequests, unsigned int allDeviceListIndex )
{
    // Pair the page requests with the device index.
    m_perDeviceRequests.reserve( m_perDeviceRequests.size() + numRequests );
    for( unsigned int i = 0; i < numRequests; ++i )
    {
        m_perDeviceRequests.push_back( PerDeviceRequest{requestedPages[i], allDeviceListIndex} );
    }
}

void PageRequestsSimple::processRequests( PagingMode                 lwrrPagingMode,
                                          StagingPageAllocator*      stagingPages,
                                          ApiCapture&                apiCapture,
                                          std::vector<DevicePaging>& deviceState )
{
    LOG_NORMAL( "PageRequestsSimple::processRequests mode " << toString( lwrrPagingMode ) << '\n' );

    // Nothing to do?
    if( m_perDeviceRequests.empty() )
    {
        return;
    }

    createRequestHandlers( lwrrPagingMode );
    reallocateMipLevels( deviceState );

    if( m_multiThreadedCallbacksEnabled )
    {
        fillMemoryBlocksThreaded( stagingPages, apiCapture );
    }
    else
    {
        fillMemoryBlocks( stagingPages, apiCapture );
    }

    synchronizeAllDevices( stagingPages, deviceState, lwrrPagingMode );

    releaseMemoryBlocks( stagingPages );
}

void PageRequestsSimple::synchronizeAllDevices( StagingPageAllocator* stagingPages, std::vector<DevicePaging>& deviceState, PagingMode lwrrPagingMode )
{
    LOG_NORMAL( "PageRequestsSimple::synchronizeAllDevices " << lwrrPagingMode << '\n' );
    ElapsedTimeCapture elapsed( m_copyMsec );

    for( DevicePaging& devicePaging : deviceState )
    {
        if( !devicePaging.isInitialized() )
            continue;

        synchronize( stagingPages, devicePaging.getAllDeviceListIndex(), devicePaging );
    }
}

void PageRequestsSimple::createRequestHandlers( PagingMode lwrrPagingMode )
{
    LOG_NORMAL( "PageRequestsSimple::createRequestHandlers mode " << toString( lwrrPagingMode ) << '\n' );
    RT_ASSERT( !m_perDeviceRequests.empty() );

    // Sort requests by page number, so that requests for the same page from different devices are contiguous.
    algorithm::sort( m_perDeviceRequests,
                     []( const PerDeviceRequest& a, const PerDeviceRequest& b ) { return a.pageId < b.pageId; } );

    // Gather requests for the same page from multiple devices into a single request that contains a DeviceSet.
    auto it = m_perDeviceRequests.cbegin();
    while( it != m_perDeviceRequests.cend() )
    {
        unsigned int pageId = it->pageId;

        // Gather all the devices requesting this page.  The requests are adjacent because the vector was sorted.
        DeviceSet devices;
        while( it != m_perDeviceRequests.cend() && it->pageId == pageId )
        {
            devices |= DeviceSet( it->allDeviceListIndex );
            ++it;
        }

        // Find the buffer associated with this page request via the BufferPageHeap.  It kept track
        // of the range of page table entries that was reserved when the buffer was allocated and
        // provides an ilwerse mapping.
        const BufferPageHeap::HeapEntry* heapEntry = m_bufferPages->find( pageId );
        RT_ASSERT_MSG( heapEntry != nullptr, "No demand buffer associated with requested page" );
        RT_ASSERT_MSG( ( ( heapEntry->isBuffer || heapEntry->isLwdaSparse ) && heapEntry->resource.buffer != nullptr )
                           || ( !( heapEntry->isBuffer || heapEntry->isLwdaSparse ) && !heapEntry->resource.samplers.empty() ),
                       "NULL resource associated with requested page" );
        const bool            isBufferEntry = heapEntry->isBuffer || heapEntry->isLwdaSparse;
        const TextureSampler* sampler       = isBufferEntry ? nullptr : heapEntry->resource.samplers.back();
        const Buffer*         buffer        = isBufferEntry ? heapEntry->resource.buffer : sampler->getBuffer();
        unsigned int          startPage     = heapEntry->startPage;

        // Construct RequestHandler, which records the requested page, the device set, and the buffer.
        switch( lwrrPagingMode )
        {
            case PagingMode::WHOLE_MIPLEVEL:
                // If per-tile callbacks are disabled, textures are loaded with whole-miplevel request
                if( !heapEntry->isBuffer )
                {
                    const unsigned int mipLevel = pageId - startPage;
                    m_mipLevelRequests.emplace_back( mipLevel, pageId, devices, buffer, startPage, true );
                }
                else
                    m_bufferPageRequests.emplace_back( pageId, devices, buffer, startPage );
                break;

            case PagingMode::SOFTWARE_SPARSE:
                if( RequestHandler::isSmallNonMipmappedTextureRequest( pageId, startPage, sampler ) )
                    m_mipLevelRequests.emplace_back( 0, pageId, devices, buffer, startPage, true );
                else if( RequestHandler::isMipTailRequest( pageId, startPage, buffer ) )
                    m_mipTailRequests.emplace_back( pageId, devices, buffer, startPage, true );
                else if( !heapEntry->isBuffer )
                    m_textureTileRequests.emplace_back( pageId, devices, buffer, sampler, startPage );
                else
                    m_bufferPageRequests.emplace_back( pageId, devices, buffer, startPage );
                break;

            case PagingMode::LWDA_SPARSE_HYBRID:
            case PagingMode::LWDA_SPARSE_HARDWARE:
                if( HardwareMipTailRequestHandler::isSmallNonMipmapped( pageId, startPage, buffer ) )
                {
                    // Switching from LWCA sparse array to LWCA array?
                    if( heapEntry->isLwdaSparse )
                    {
                        buffer->switchLwdaSparseArrayToLwdaArray( devices );
                        m_bufferPages->switchSparseTextureToTexture( startPage );
                    }
                    m_mipLevelRequests.emplace_back( 0, pageId, devices, buffer, startPage, true );
                }
                else if( HardwareMipTailRequestHandler::isMipTailRequest( pageId, startPage, buffer ) )
                    m_hardwareMipTailRequests.emplace_back( pageId, devices, buffer, startPage, true );
                else if( !heapEntry->isBuffer )
                    m_hardwareTileRequests.emplace_back( pageId, devices, buffer, startPage );
                else
                    m_bufferPageRequests.emplace_back( pageId, devices, buffer, startPage );
                break;

            case PagingMode::UNKNOWN:
                RT_ASSERT_FAIL_MSG( "Unknown paging mode." );
        }
    }
    m_perDeviceRequests.clear();
}

namespace {
template <typename Request>
void fillMemoryBlocksImpl( std::vector<Request>& requests, StagingPageAllocator* stagingPages, ApiCapture& apiCapture )
{
    for( Request& request : requests )
    {
        request.fillMemoryBlock( stagingPages );
        if( request.caughtException() )
            throw IlwalidOperation( RT_EXCEPTION_INFO, request.getExceptionMessage() );
        request.captureTrace( apiCapture );
    }
}

template <typename Request>
void synchronizeImpl( std::vector<Request>& requests,
                      StagingPageAllocator* stagingPages,
                      unsigned int          allDeviceListIndex,
                      DevicePaging&         devicePaging,
                      bool                  synchronous )
{
    for( Request& request : requests )
    {
        request.synchronize( stagingPages, allDeviceListIndex, devicePaging, synchronous );
    }
}

template <typename Request>
void releaseMemoryBlocksImpl( std::vector<Request>& requests, StagingPageAllocator* stagingPages )
{
    for( Request& request : requests )
    {
        request.releaseMemoryBlock( stagingPages );
    }
}

}  // anonymous namespace

// Dispatch callbacks from the main thread.
void PageRequestsSimple::fillMemoryBlocks( StagingPageAllocator* stagingPages, ApiCapture& apiCapture )
{
    LOG_NORMAL( "PageRequestsSimple::fillMemoryBlocks\n" );
    ElapsedTimeCapture elapsed( m_callbackMsec );

    fillMemoryBlocksImpl( m_bufferPageRequests, stagingPages, apiCapture );
    fillMemoryBlocksImpl( m_mipLevelRequests, stagingPages, apiCapture );
    fillMemoryBlocksImpl( m_textureTileRequests, stagingPages, apiCapture );
    fillMemoryBlocksImpl( m_mipTailRequests, stagingPages, apiCapture );
    fillMemoryBlocksImpl( m_hardwareTileRequests, stagingPages, apiCapture );
    fillMemoryBlocksImpl( m_hardwareMipTailRequests, stagingPages, apiCapture );
}

void PageRequestsSimple::synchronize( StagingPageAllocator* stagingPages, unsigned int allDeviceListIndex, DevicePaging& devicePaging )
{
    LOG_NORMAL( "PageRequestsSimple::synchronize( "
                << allDeviceListIndex << " ) " << m_bufferPageRequests.size() << " buffer, "
                << m_mipLevelRequests.size() << " mip level, " << m_textureTileRequests.size() << " texture tile, "
                << m_mipTailRequests.size() << " mip tail," << m_hardwareTileRequests.size() << " HW tile, "
                << m_hardwareMipTailRequests.size() << " HW mip tail requests\n" );
    synchronizeImpl( m_bufferPageRequests, stagingPages, allDeviceListIndex, devicePaging, true );
    synchronizeImpl( m_mipLevelRequests, stagingPages, allDeviceListIndex, devicePaging, true );
    synchronizeImpl( m_textureTileRequests, stagingPages, allDeviceListIndex, devicePaging, true );
    synchronizeImpl( m_mipTailRequests, stagingPages, allDeviceListIndex, devicePaging, true );
    synchronizeImpl( m_hardwareTileRequests, stagingPages, allDeviceListIndex, devicePaging, true );
    synchronizeImpl( m_hardwareMipTailRequests, stagingPages, allDeviceListIndex, devicePaging, true );
}

void PageRequestsSimple::releaseMemoryBlocks( StagingPageAllocator* stagingPages )
{
    LOG_NORMAL( "PageRequestsSimple::releaseMemoryBlocks() "
                << m_bufferPageRequests.size() << " buffer, " << m_mipLevelRequests.size() << " mip level, "
                << m_textureTileRequests.size() << " texture tile, " << m_mipTailRequests.size() << " mip tail,"
                << m_hardwareTileRequests.size() << " HW tile, " << m_hardwareMipTailRequests.size()
                << " HW mip tail requests\n" );
    releaseMemoryBlocksImpl( m_bufferPageRequests, stagingPages );
    releaseMemoryBlocksImpl( m_mipLevelRequests, stagingPages );
    releaseMemoryBlocksImpl( m_textureTileRequests, stagingPages );
    releaseMemoryBlocksImpl( m_mipTailRequests, stagingPages );
    releaseMemoryBlocksImpl( m_hardwareTileRequests, stagingPages );
    releaseMemoryBlocksImpl( m_hardwareMipTailRequests, stagingPages );
}

void PageRequestsSimple::reallocateMipLevels( std::vector<DevicePaging>& deviceState )
{
    LOG_NORMAL( "PageRequestsSimple::reallocateMipLevels\n" );
    for( DevicePaging& devicePaging : deviceState )
    {
        if( !devicePaging.isInitialized() )
            continue;

        const unsigned int allDeviceListIndex = devicePaging.getAllDeviceListIndex();

        // The page requests were initially sorted by pageId, so all the miplevel requests for a given
        // texture are adjacent.
        {
            auto it = m_mipLevelRequests.cbegin();
            while( it != m_mipLevelRequests.cend() )
            {
                if( !it->getDeviceSet().isSet( allDeviceListIndex ) )
                {
                    ++it;
                    continue;
                }
                const Buffer* buffer   = it->getBuffer();
                unsigned int  minLevel = it->getMipLevel();
                unsigned int  maxLevel = it->getMipLevel();
                for( ++it; it != m_mipLevelRequests.cend() && it->getBuffer() == buffer; ++it )
                {
                    if( it->getDeviceSet().isSet( allDeviceListIndex ) )
                    {
                        minLevel = std::min( minLevel, it->getMipLevel() );
                        maxLevel = std::max( maxLevel, it->getMipLevel() );
                    }
                }
                devicePaging.reallocDemandLoadLwdaArray( buffer->getMBuffer(), minLevel, maxLevel );
            }
        }
    }

    for( MipTailRequestHandler& request : m_mipTailRequests )
    {
        request.reallocMipTail( deviceState );
    }
}

// ---------- Threaded callback invocation ----------

class FillMemoryBlocksJob : public FragmentedJob
{
  public:
    FillMemoryBlocksJob( std::vector<RequestHandler*>& pageRequests, StagingPageAllocator* stagingPages )
        : FragmentedJob( pageRequests.size() )
        , m_requests( pageRequests )
        , m_stagingPages( stagingPages )
    {
    }

    void exelwteFragment( size_t index, size_t count ) noexcept override
    {
        m_requests[index]->fillMemoryBlock( m_stagingPages );
    }

  private:
    std::vector<RequestHandler*>& m_requests;
    StagingPageAllocator*         m_stagingPages;
};

void PageRequestsSimple::fillMemoryBlocksThreaded( StagingPageAllocator* stagingPages, ApiCapture& apiCapture )
{
    LOG_NORMAL( "PageRequestsSimple::fillMemoryBlocksThreaded\n" );
    ElapsedTimeCapture elapsed( m_callbackMsec );

    // Gather request handlers in a single vector for maximum parallelism.
    std::vector<RequestHandler*> requests;
    requests.reserve( m_bufferPageRequests.size() + m_textureTileRequests.size() + m_mipTailRequests.size()
                      + m_mipLevelRequests.size() + m_hardwareTileRequests.size() + m_hardwareMipTailRequests.size() );
    for( RequestHandler& request : m_bufferPageRequests )
        requests.push_back( &request );
    for( RequestHandler& request : m_textureTileRequests )
        requests.push_back( &request );
    for( RequestHandler& request : m_mipTailRequests )
        requests.push_back( &request );
    for( RequestHandler& request : m_mipLevelRequests )
        requests.push_back( &request );
    for( RequestHandler& request : m_hardwareTileRequests )
        requests.push_back( &request );
    for( RequestHandler& request : m_hardwareMipTailRequests )
        requests.push_back( &request );

    // Dispatch callbacks in parallel.
    const std::shared_ptr<FillMemoryBlocksJob> fillMemoryBlocksJob( std::make_shared<FillMemoryBlocksJob>( requests, stagingPages ) );
    m_threadPool->submitJobAndWait( fillMemoryBlocksJob );
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

void PageRequestsSimple::reportMetrics()
{
    LOG_NORMAL( "PageRequestsSimple::reportMetrics\n" );
    size_t              tilesRequested             = 0;
    size_t              tilesRequestedNonMipMapped = 0;
    size_t              bytesFilled                = 0;
    size_t              tilesFilled                = 0;
    std::vector<size_t> tilesFilledPerMipLevel;  // from coarsest to finest.

    for( const TextureTileRequestHandler& request : m_textureTileRequests )
    {
        ++tilesRequested;
        unsigned int numMipLevels = request.getBuffer()->getMipLevelCount();
        if( numMipLevels <= 1 )
            ++tilesRequestedNonMipMapped;

        if( request.isFilled() )
        {
            ++tilesFilled;

            const RTmemoryblock& block = request.getMemoryBlock();
            bytesFilled += block.width * block.height * request.getBuffer()->getElementSize();

            if( numMipLevels > 1 )
            {
                // Per-miplevel stats are recorded from coarsest to finest levels.
                unsigned int mipLevel = request.getMemoryBlock().mipLevel;
                unsigned int levelNum = numMipLevels - mipLevel;
                if( levelNum >= tilesFilledPerMipLevel.size() )
                    tilesFilledPerMipLevel.resize( levelNum + 1 );
                ++tilesFilledPerMipLevel[levelNum];
            }
            else
            {
                // Non-mipmapped texture.  Use the resolution to determine
                // what the corresponding miplevel would be if it were mipmapped.
                unsigned int width    = request.getBuffer()->getWidth();
                unsigned int height   = request.getBuffer()->getHeight();
                unsigned int levelNum = 1 + std::log2( std::max( width, height ) );
                if( levelNum >= tilesFilledPerMipLevel.size() )
                    tilesFilledPerMipLevel.resize( levelNum + 1 );
                ++tilesFilledPerMipLevel[levelNum];
            }
        }
    }

    for( const MipTailRequestHandler& request : m_mipTailRequests )
    {
        unsigned int numMipLevels = request.getNumMipLevels();
        tilesRequested += numMipLevels;
        if( request.isFilled() )
        {
            ++tilesFilled;

            const RTmemoryblock& block = request.getMemoryBlock();
            bytesFilled += block.width * block.height * request.getBuffer()->getElementSize();

            if( numMipLevels >= tilesFilledPerMipLevel.size() )
                tilesFilledPerMipLevel.resize( numMipLevels );
            for( unsigned int i = 0; i < numMipLevels; ++i )
            {
                ++tilesFilledPerMipLevel[i];
            }
        }
    }

    for( const MipLevelRequestHandler& request : m_mipLevelRequests )
    {
        if( request.isFilled() )
        {
            const RTmemoryblock& block = request.getMemoryBlock();
            bytesFilled += block.width * block.height * request.getBuffer()->getElementSize();
        }
    }

    Metrics::logInt( "demand_texture_callback_msec", m_callbackMsec );
    Metrics::logInt( "demand_texture_tiles_requested", tilesRequested );
    Metrics::logInt( "demand_texture_tiles_requested_non_mipmapped", tilesRequestedNonMipMapped );
    Metrics::logInt( "demand_texture_tiles_filled", tilesFilled );
    Metrics::logFloat( "demand_texture_megabytes_filled", bytesFilled / ( 1024.0 * 1024.0 ) );

    for( size_t i = 0; i < tilesFilledPerMipLevel.size(); ++i )
    {
        std::stringstream stream;
        stream << "demand_texture_tiles_filled_level<" << i << ">";
        Metrics::logInt( stream.str().c_str(), tilesFilledPerMipLevel[i] );
    }
}

}  // namespace optix
