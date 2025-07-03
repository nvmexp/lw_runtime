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

#include <Memory/DemandLoad/PageRequestsBase.h>

#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Memory/DemandLoad/BufferPageHeap.h>
#include <Memory/DemandLoad/DevicePaging.h>
#include <Memory/DemandLoad/PageRequests.h>
#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Memory/DemandLoad/StagingPageAllocator.h>
#include <Objects/TextureSampler.h>
#include <Util/ContainerAlgorithm.h>
#include <Util/Metrics.h>
#include <c-api/ApiCapture.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidOperation.h>

#include <algorithm>

using namespace prodlib;

namespace optix {

PageRequestsBase::PageRequestsBase( DeviceManager* dm, BufferPageHeap* bufferPages, bool usePerTileCallbacks )
    : m_dm( dm )
    , m_usePerTileCallbacks( usePerTileCallbacks )
    , m_bufferPages( bufferPages )
{
}

void PageRequestsBase::clear()
{
    m_perDeviceRequests.clear();
    m_textureTileRequests.clear();
    m_bufferPageRequests.clear();
    m_mipTailRequests.clear();
    m_mipLevelRequests.clear();
    m_hardwareTileRequests.clear();
    m_hardwareMipTailRequests.clear();
    m_callbackMsec = 0;
}

void PageRequestsBase::addRequests( unsigned int* requestedPages, unsigned int numRequests, unsigned int allDeviceListIndex )
{
    // Pair the page requests with the device index.
    m_perDeviceRequests.reserve( m_perDeviceRequests.size() + numRequests );
    for( unsigned int i = 0; i < numRequests; ++i )
    {
        m_perDeviceRequests.push_back( PerDeviceRequest{requestedPages[i], allDeviceListIndex} );
    }
}

void PageRequestsBase::createRequestHandlers( PagingMode lwrrPagingMode )
{
    if( m_perDeviceRequests.empty() )
        return;

    LOG_NORMAL( "PageRequestsBase::createRequestHandlers mode " << toString( lwrrPagingMode ) << '\n' );

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
        RT_ASSERT_MSG( ( ( heapEntry->isBuffer || heapEntry->isLwdaSparse ) && ( heapEntry->resource.buffer != nullptr ) )
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
                    m_mipLevelRequests.emplace_back( mipLevel, pageId, devices, buffer, startPage, false );
                }
                else
                    m_bufferPageRequests.emplace_back( pageId, devices, buffer, startPage );
                break;

            case PagingMode::SOFTWARE_SPARSE:
                if( RequestHandler::isSmallNonMipmappedTextureRequest( pageId, startPage, sampler ) )
                    m_mipLevelRequests.emplace_back( 0, pageId, devices, buffer, startPage, true );
                else if( RequestHandler::isMipTailRequest( pageId, startPage, buffer ) )
                    m_mipTailRequests.emplace_back( pageId, devices, buffer, startPage, false );
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
                    m_hardwareMipTailRequests.emplace_back( pageId, devices, buffer, startPage, false );
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

void PageRequestsBase::reallocateMipLevels( std::vector<DevicePaging>& deviceState )
{
    LOG_NORMAL( "PageRequestsBase::reallocateMipLevels\n" );

    for( DevicePaging& devicePaging : deviceState )
    {
        if( !devicePaging.isInitialized() )
            continue;

        const unsigned int allDeviceListIndex = devicePaging.getAllDeviceListIndex();
        makeLwrrent( allDeviceListIndex );

        // The page requests were initially sorted by pageId, so all the miplevel requests for a given
        // texture are adjacent.
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

void PageRequestsBase::reportMetrics()
{
    LOG_NORMAL( "PageRequestsBase::reportMetrics\n" );

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


void PageRequestsBase::makeLwrrent( unsigned int allDeviceIndex ) const
{
    Device*     device     = m_dm->allDevices()[allDeviceIndex];
    LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device );
    RT_ASSERT( lwdaDevice );
    lwdaDevice->makeLwrrent();
}

}  // namespace optix
