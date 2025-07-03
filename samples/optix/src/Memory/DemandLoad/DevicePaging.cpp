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
#include <Memory/DemandLoad/DevicePaging.h>

#include <LWCA/Context.h>
#include <LWCA/ErrorCheck.h>
#include <LWCA/Memory.h>
#include <Device/LWDADevice.h>
#include <Memory/DemandLoad/PagingService.h>
#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Memory/DemandLoad/StagingPageAllocator.h>
#include <Memory/DemandLoad/TileManager.h>
#include <Memory/DemandLoad/TilePool.h>
#include <Objects/Buffer.h>

#include <prodlib/system/Knobs.h>

#include <corelib/system/LwdaDriver.h>

#include <cstdint>

using namespace prodlib;
using namespace optix::lwca;
using namespace optix::demandLoad;

namespace {

const unsigned int MAX_REQUESTED_PAGES       = 1024 * 1024;
const unsigned int MAX_STALE_PAGES           = 16;
const unsigned int MAX_EVICTABLE_PAGES       = 16;
const unsigned int NUM_PAGES_IN_POOL         = 1024;
const unsigned int MAX_NUM_FILLED_PAGES      = 1024 * 1024;
const unsigned int MAX_NUM_ILWALIDATED_PAGES = 16;

}  // namespace

namespace optix {

void DevicePaging::activate( LWDADevice* device, MemoryManager* mm, unsigned int allDeviceListIndex, unsigned int numVirtualPages )
{
    if( m_initialized )
        return;

    LOG_MEDIUM_VERBOSE( "DevicePaging::activate device " << allDeviceListIndex << ", " << numVirtualPages
                                                         << " pages.\n" );

    m_device = device;
    m_device->makeLwrrent();
    m_mm                 = mm;
    m_allDeviceListIndex = allDeviceListIndex;
    m_copyStream         = lwca::Stream::create();
    m_copyEvent          = lwca::Event::create();

    OptixPagingOptions options{numVirtualPages, numVirtualPages};
    optixPagingCreate( &options, &m_pagingContext );
    optixPagingCallwlateSizes( options.initialVaSizeInPages, m_pagingSizes );

    const DeviceSet allowedDevices( static_cast<int>( allDeviceListIndex ) );
    m_pageTable = MBufferDeviceVector<unsigned long long>( mm, allowedDevices, m_pagingSizes.pageTableSizeInBytes
                                                                                   / sizeof( unsigned long long ) );
    m_pageTable.addListener( this );
    m_havePageTable = false;
    m_usageBits =
        MBufferDeviceVector<unsigned int>( mm, allowedDevices, m_pagingSizes.usageBitsSizeInBytes / sizeof( unsigned int ) );
    m_usageBits.addListener( this );
    m_haveUsageBits = false;

    m_pagePool = MBufferDeviceVector<unsigned long long>( mm, allowedDevices, NUM_PAGES_IN_POOL * PagingService::PAGE_SIZE_IN_BYTES
                                                                                  / sizeof( unsigned long long ) );
    m_pagePoolCount = 0;

    m_numRequestedPages   = MAX_REQUESTED_PAGES;  // TODO: real number here
    m_devRequestedPages   = MBufferDeviceVector<unsigned int>( mm, allowedDevices, m_numRequestedPages );
    m_numStalePages       = MAX_STALE_PAGES;  // TODO: real number here
    m_devStalePages       = MBufferDeviceVector<PageMapping>( mm, allowedDevices, m_numStalePages );
    m_numEvictablePages   = MAX_EVICTABLE_PAGES;  // TODO: real number here
    m_devEvictablePages   = MBufferDeviceVector<unsigned int>( mm, allowedDevices, m_numEvictablePages );
    m_devNumPagesReturned = MBufferDeviceVector<unsigned int>( mm, allowedDevices, 3 );
    m_devFilledPages =
        MBufferDeviceVector<PageMapping>( mm, allowedDevices, MAX_NUM_FILLED_PAGES, MBufferPolicy::internal_readonly );
    m_devIlwalidatedPages = MBufferDeviceVector<unsigned int>( mm, allowedDevices, MAX_NUM_ILWALIDATED_PAGES );

    m_tileManager.activate( m_mm, m_allDeviceListIndex );
    m_hardwareTileManager.activate( m_mm, allDeviceListIndex );

    m_initialized = true;
}

void DevicePaging::deactivate()
{
    if( !m_initialized )
        return;

    LOG_MEDIUM_VERBOSE( "DevicePaging::deactivate device " << m_allDeviceListIndex << '\n' );

    m_device->makeLwrrent();
    m_copyEvent.destroy();
    m_copyStream.destroy();
    m_pagePool.reset();
    m_pageTable.reset();
    m_havePageTable = false;
    m_usageBits.reset();
    m_haveUsageBits = false;
    m_devRequestedPages.reset();
    m_devNumPagesReturned.reset();
    m_devStalePages.reset();
    m_devEvictablePages.reset();
    m_devFilledPages.reset();
    m_devIlwalidatedPages.reset();
    m_tileManager.deactivate();
    m_hostFilledPages.clear();
    m_hostFilledPages.shrink_to_fit();

    // Reset these last as resetting MBuffers can cause us to receive events
    m_allDeviceListIndex = ~0U;
    m_device             = nullptr;
    m_mm                 = nullptr;
    m_initialized        = false;
}

// Launch paging system kernel to pull page requests from the specified device.
// Copy the requested pages from the device to the given host.
void DevicePaging::pullRequests( std::vector<unsigned int>& requestedPages )
{
    // TODO: make this asynchronous
    RT_ASSERT( m_initialized );
    m_device->makeLwrrent();
    optixPagingPullRequests( m_pagingContext, m_devRequestedPages.getDevicePtr( m_allDeviceListIndex ),
                             m_numRequestedPages, m_devStalePages.getDevicePtr( m_allDeviceListIndex ), m_numStalePages,
                             m_devEvictablePages.getDevicePtr( m_allDeviceListIndex ), m_numEvictablePages,
                             m_devNumPagesReturned.getDevicePtr( m_allDeviceListIndex ) );

    // Get the sizes of the requested, stale, and evictable page lists.
    unsigned int numReturned[3] = {0};
    m_devNumPagesReturned.copyToHost( m_mm, &numReturned[0], 3 );

    LOG_MEDIUM_VERBOSE( "DevicePaging::pullRequests " << numReturned[0] << " requested, " << numReturned[1]
                                                      << " stale, " << numReturned[2] << " evictable\n" );
    // Early exit if no pages requested.
    unsigned int numRequests = numReturned[0];
    if( numRequests == 0 )
        return;

    // Copy the requested page list from this device.
    copyRequestedPagesToHost( requestedPages, numRequests );
}

void DevicePaging::pushMappings()
{
    RT_ASSERT( m_initialized );

    LOG_MEDIUM_VERBOSE( "DevicePaging::pushMappings " << m_filledPageCount << " filled, " << m_ilwalidatedPageCount << " ilwalidated\n" );

    m_device->makeLwrrent();

    // synch against any pending memcpy operations in flight from previous pullRequests on previous launch
    m_copyStream.waitEvent( m_copyEvent, 0 );

    // We always need the event above, but we can skip the kernel launch if there's no work to do.
    if( m_filledPageCount != 0 || m_ilwalidatedPageCount != 0 )
        optixPagingPushMappings( m_pagingContext, m_devFilledPages.getDevicePtr( m_allDeviceListIndex ), m_filledPageCount,
                                 m_devIlwalidatedPages.getDevicePtr( m_allDeviceListIndex ), m_ilwalidatedPageCount );
}

void DevicePaging::synchronizeTiles()
{
    LOG_MEDIUM_VERBOSE( "DevicePaging::synchronizeTiles device " << m_allDeviceListIndex << '\n' );
    m_device->makeLwrrent();
    m_tileManager.synchronize();
}

static std::string toString( const PageMapping& item )
{
    return "{ " + std::to_string( item.id ) + ", " + std::to_string( item.page ) + " }";
}

static std::ostream& operator<<( std::ostream& stream, const PageMapping& item )
{
    return stream << toString( item );
}

void DevicePaging::addPageMapping( PageMapping mapping )
{
    LOG_VERBOSE( "DevicePaging::addPageMapping device " << m_allDeviceListIndex << ",  " << mapping << '\n' );
    m_hostFilledPages.push_back( mapping );
}

void DevicePaging::copyPageMappingsToDevice()
{
    m_filledPageCount = static_cast<int>( m_hostFilledPages.size() );
    LOG_MEDIUM_VERBOSE( "DevicePaging::copyPageMappingsToDevice() device " << m_allDeviceListIndex << ", "
                                                                           << m_filledPageCount << " filled pages\n" );

    if( m_filledPageCount != 0 )
    {
        if( isLogVerboseActive() )
        {
            std::string mappings;
            mappings.reserve( m_hostFilledPages.size() * 10 );
            for( const PageMapping& item : m_hostFilledPages )
            {
                if( !mappings.empty() )
                {
                    mappings += ", ";
                }
                mappings += toString( item );
            }
            LOG_VERBOSE( "DevicePaging::copyPageMappingsToDevice() device " << m_allDeviceListIndex << " mappings { "
                                                                            << mappings << " }\n" );
        }

        m_device->makeLwrrent();
        m_devFilledPages.copyToDevice( m_mm, m_hostFilledPages.data(), m_filledPageCount );
        m_devFilledPages.manualSynchronize( m_mm );
        m_hostFilledPages.clear();
        m_copyEvent.record( m_copyStream );
    }
}

void DevicePaging::copyRequestedPagesToHost( std::vector<unsigned int>& requestedPages, unsigned int numRequests )
{
    LOG_MEDIUM_VERBOSE( "DevicePaging::copyRequestedPagesToHost device " << m_allDeviceListIndex << ", " << numRequests
                                                                         << " requests.\n" );

    requestedPages.resize( numRequests );
    m_devRequestedPages.copyToHost( m_mm, requestedPages.data(), numRequests );
    if( isLogVerboseActive() )
    {
        std::string pages;
        pages.reserve( numRequests * 3 );
        for( unsigned int page : requestedPages )
        {
            if( !pages.empty() )
            {
                pages += ", ";
            }
            pages += std::to_string( page );
        }
        LOG_VERBOSE( "DevicePaging::copyRequestedPagesToHost device " << m_allDeviceListIndex << " pages { " << pages << " }\n" );
    }
}

LWdeviceptr DevicePaging::copyPageToDevice( StagingPageAllocator*        stagingPages,
                                            const StagingPageAllocation& allocation,
                                            const void*                  data,
                                            bool                         synchronous )
{
    LOG_MEDIUM_VERBOSE( "DevicePaging::copyPageToDevice device " << m_allDeviceListIndex << ", allocation "
                                                                 << allocation.id << '\n' );

    // Allocate page from page pool for memcpy destination.  TODO: handle page pool exhaustion.
    const LWdeviceptr devPage = allocatePagePoolDeviceAddress();

    // Copy from page in staging area to the device page.
    if( synchronous )
    {
        memcpyHtoD( devPage, data, PagingService::PAGE_SIZE_IN_BYTES );
    }
    else
    {
        memcpyHtoDAsync( devPage, data, PagingService::PAGE_SIZE_IN_BYTES, m_copyStream );
    }
    stagingPages->recordEvent( m_copyStream, m_allDeviceListIndex, allocation );

    return devPage;
}

TileLocator DevicePaging::copyTileToDevice( StagingPageAllocator*        stagingPages,
                                            const StagingPageAllocation& allocation,
                                            const RTmemoryblock&         memoryBlock,
                                            const Buffer*                buffer )
{
    RT_ASSERT( buffer->hasTextureAttached() && buffer->isDemandLoad() );

    LOG_MEDIUM_VERBOSE( "DevicePaging::copyTileToDevice device " << m_allDeviceListIndex << ", allocation "
                                                                 << allocation.id << '\n' );

    const TextureDescriptor& descriptor = buffer->getAttachedTextureSampler()->getDescriptor();
    TileLocator              locator =
        m_tileManager.allocateTileWithinPool( memoryBlock.width, memoryBlock.height, buffer->getFormat(), descriptor );

    // Fill tile.
    TileArray* tileArray = m_tileManager.getTileArray( locator.unpacked.tileArray );
    tileArray->fillTile( m_allDeviceListIndex, locator.unpacked.tileIndex, memoryBlock.baseAddress );
    stagingPages->recordEvent( m_copyStream, m_allDeviceListIndex, allocation );

    return locator;
}

TileLocator DevicePaging::copyTileToDeviceAsync( StagingPageAllocator*        stagingPages,
                                                 const StagingPageAllocation& allocation,
                                                 const RTmemoryblock&         memoryBlock,
                                                 const Buffer*                buffer )
{
    RT_ASSERT( buffer->hasTextureAttached() && buffer->isDemandLoad() );

    LOG_MEDIUM_VERBOSE( "DevicePaging::copyTileToDeviceAsync device " << m_allDeviceListIndex << ", allocation "
                                                                      << allocation.id << '\n' );

    const TextureDescriptor& descriptor = buffer->getAttachedTextureSampler()->getDescriptor();
    TileLocator              locator =
        m_tileManager.allocateTileWithinPool( memoryBlock.width, memoryBlock.height, buffer->getFormat(), descriptor );

    // Fill tile.
    TileArray* tileArray = m_tileManager.getTileArray( locator.unpacked.tileArray );
    tileArray->fillTileAsync( m_copyStream, m_allDeviceListIndex, locator.unpacked.tileIndex, memoryBlock.baseAddress );
    stagingPages->recordEvent( m_copyStream, m_allDeviceListIndex, allocation );

    return locator;
}

void DevicePaging::reallocDemandLoadLwdaArray( const MBufferHandle& buffer, int minLevel, int maxLevel )
{
    LOG_MEDIUM_VERBOSE( "DevicePaging::reallocDemandLoadLwdaArray device "
                        << m_allDeviceListIndex << ", minLevel " << minLevel << ", maxLevel " << maxLevel << '\n' );

    m_mm->reallocDemandLoadLwdaArray( buffer, m_allDeviceListIndex, minLevel, maxLevel );
}

void DevicePaging::syncDemandLoadMipLevel( StagingPageAllocator*        stagingPages,
                                           const StagingPageAllocation& allocation,
                                           const Buffer*                buffer,
                                           void*                        baseAddress,
                                           size_t                       byteCount,
                                           int                          mipLevel )
{
    LOG_MEDIUM_VERBOSE( "DevicePaging::syncDemandLoadMipLevel device " << m_allDeviceListIndex << ", mipLevel " << mipLevel << '\n' );

    m_mm->syncDemandLoadMipLevel( buffer, baseAddress, byteCount, m_allDeviceListIndex, mipLevel );
    stagingPages->recordEvent( m_copyStream, m_allDeviceListIndex, allocation );
}

void DevicePaging::syncDemandLoadMipLevelAsync( StagingPageAllocator*        stagingPages,
                                                const StagingPageAllocation& allocation,
                                                const Buffer*                buffer,
                                                void*                        baseAddress,
                                                size_t                       byteCount,
                                                int                          mipLevel )
{
    LOG_MEDIUM_VERBOSE( "DevicePaging::syncDemandLoadMipLevelAsync device " << m_allDeviceListIndex << ", mipLevel "
                                                                            << mipLevel << '\n' );

    m_mm->syncDemandLoadMipLevelAsync( m_copyStream, buffer, baseAddress, byteCount, m_allDeviceListIndex, mipLevel );
    stagingPages->recordEvent( m_copyStream, m_allDeviceListIndex, allocation );
}

LWdeviceptr DevicePaging::allocatePagePoolDeviceAddress()
{
    RT_ASSERT_MSG( m_pagePoolCount < NUM_PAGES_IN_POOL, "Page pool exhausted" );
    return reinterpret_cast<LWdeviceptr>( m_pagePool.getDevicePtr( m_allDeviceListIndex ) )
           + m_pagePoolCount++ * PagingService::PAGE_SIZE_IN_BYTES;
}

#define KIND_CASE( kind_ )                                                                                             \
    case MAccess::kind_:                                                                                               \
        return str << #kind_

static std::ostream& operator<<( std::ostream& str, MAccess::Kind kind )
{
    switch( kind )
    {
        KIND_CASE( LINEAR );
        KIND_CASE( MULTI_PITCHED_LINEAR );
        KIND_CASE( TEX_OBJECT );
        KIND_CASE( TEX_REFERENCE );
        KIND_CASE( LWDA_SPARSE );
        KIND_CASE( LWDA_SPARSE_BACKING );
        KIND_CASE( DEMAND_LOAD );
        KIND_CASE( DEMAND_LOAD_ARRAY );
        KIND_CASE( DEMAND_LOAD_TILE_ARRAY );
        KIND_CASE( DEMAND_TEX_OBJECT );
        KIND_CASE( NONE );
    }
    return str << "Unknown access kind " << static_cast<int>( kind );
}

#undef KIND_CASE

static std::ostream& operator<<( std::ostream& str, const MAccess& access )
{
    return str << access.getKind();
}

static std::ostream& operator<<( std::ostream& str, const OptixPagingSizes& sizes )
{
    return str << "{ " << sizes.pageTableSizeInBytes << ", " << sizes.usageBitsSizeInBytes << " }";
}

void DevicePaging::eventMBufferMAccessDidChange( const MBuffer* buffer, const Device* device, const MAccess& oldMBA, const MAccess& newMBA )
{
    // Do nothing if this DevicePaging instance has been deactivated.
    if( m_device == nullptr )
    {
        return;
    }

    RT_ASSERT( m_device == deviceCast<LWDADevice>( device ) );
    LOG_MEDIUM_VERBOSE( "DevicePaging::eventMBufferMAccessDidChange device " << m_allDeviceListIndex << ' ' << oldMBA
                                                                             << " => " << newMBA << '\n' );

    // We're only interested when the buffers first get allocated on the device.
    if( !( oldMBA.getKind() == MAccess::NONE && newMBA.getKind() == MAccess::LINEAR ) )
        return;

    if( buffer == m_pageTable )
    {
        m_pagingContext->pageTable = reinterpret_cast<unsigned long long*>( newMBA.getLinearPtr() );
        m_havePageTable            = true;
    }
    else if( buffer == m_usageBits )
    {
        m_pagingContext->usageBits = reinterpret_cast<unsigned int*>( newMBA.getLinearPtr() );
        m_haveUsageBits            = true;
    }
    else
    {
        throw AssertionFailure( RT_EXCEPTION_INFO, "Unknown buffer in callback "
                                                       + std::to_string( reinterpret_cast<std::intptr_t>( buffer ) ) );
    }

    if( m_havePageTable && m_haveUsageBits )
    {
        LOG_MEDIUM_VERBOSE( "DevicePaging::eventMBufferMAccessDidChange optixPagingSetup device "
                            << m_allDeviceListIndex << ", paging sizes " << m_pagingSizes << '\n' );
        m_device->makeLwrrent();
        optixPagingSetup( m_pagingContext, m_pagingSizes, 1 );
    }
}

void DevicePaging::bindTileToMemory( StagingPageAllocator*        stagingPages,
                                     const StagingPageAllocation& allocation,
                                     const Buffer*                buffer,
                                     const RTmemoryblock&         memBlock )
{
    bindTileToMemoryAsync( stagingPages, allocation, buffer, memBlock );
    LWresult* returnResult = nullptr;
    CHECK( corelib::lwdaDriver().LwCtxSynchronize() );
}

void DevicePaging::bindTileToMemoryAsync( StagingPageAllocator*        stagingPages,
                                          const StagingPageAllocation& allocation,
                                          const Buffer*                buffer,
                                          const RTmemoryblock&         memBlock )
{
    LOG_MEDIUM_VERBOSE( "DevicePaging::bindTileToMemoryAsync device " << m_allDeviceListIndex << ", allocation "
                                                                      << allocation.id << '\n' );

    m_hardwareTileManager.bindTileToMemoryAsync( m_copyStream, buffer, m_allDeviceListIndex, memBlock );
    stagingPages->recordEvent( m_copyStream, m_allDeviceListIndex, allocation );
}

void DevicePaging::bindMipTailToMemory( const Buffer* buffer, int mipTailSizeInBytes )
{
    bindMipTailToMemoryAsync( buffer, mipTailSizeInBytes );
    LWresult* returnResult = nullptr;
    CHECK( corelib::lwdaDriver().LwCtxSynchronize() );
}

void DevicePaging::bindMipTailToMemoryAsync( const Buffer* buffer, int mipTailSizeInBytes )
{
    LOG_MEDIUM_VERBOSE( "DevicePaging::bindMipTailToMemoryAsync device " << m_allDeviceListIndex << ", size "
                                                                         << mipTailSizeInBytes << '\n' );

    m_hardwareTileManager.bindMipTailToMemoryAsync( m_copyStream, buffer, m_allDeviceListIndex, mipTailSizeInBytes );
}

void DevicePaging::fillHardwareMipTail( StagingPageAllocator*        stagingPages,
                                        const StagingPageAllocation& allocation,
                                        const Buffer*                buffer,
                                        const RTmemoryblock&         memBlock )
{
    LOG_MEDIUM_VERBOSE( "DevicePaging::fillHardwareMipTail device " << m_allDeviceListIndex << ", allocation "
                                                                    << allocation.id << '\n' );

    m_hardwareTileManager.fillHardwareMipTail( buffer, m_allDeviceListIndex, memBlock );
    stagingPages->recordEvent( m_copyStream, m_allDeviceListIndex, allocation );
}

void DevicePaging::fillHardwareMipTailAsync( StagingPageAllocator*        stagingPages,
                                             const StagingPageAllocation& allocation,
                                             const Buffer*                buffer,
                                             const RTmemoryblock&         memBlock )
{
    LOG_MEDIUM_VERBOSE( "DevicePaging::fillHardwareMipTailAsync device " << m_allDeviceListIndex << ", allocation "
                                                                         << allocation.id << '\n' );

    m_hardwareTileManager.fillHardwareMipTailAsync( m_copyStream, buffer, m_allDeviceListIndex, memBlock );
    stagingPages->recordEvent( m_copyStream, m_allDeviceListIndex, allocation );
}

}  // namespace optix
