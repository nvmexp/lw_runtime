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

#include <Memory/DemandLoad/RequestHandler/RequestHandler.h>

#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Memory/DemandLoad/RequestHandler/BlockStamper.h>
#include <Memory/DemandLoad/StagingPageAllocator.h>
#include <Objects/Buffer.h>
#include <c-api/ApiCapture.h>

#include <prodlib/misc/RTFormatUtil.h>

#include <mutex>
#include <stdexcept>

using namespace prodlib;
using namespace optix::demandLoad;

namespace {

// clang-format off
Knob<bool> k_stampMemoryBlockDebugPattern( RT_DSTRING( "rtx.demandLoad.stampMemoryBlockDebugPattern" ), false, RT_DSTRING( "Stamp a debug pattern into the memory block before ilwoking the callback and look for it afterwards" ) );
// clang-format on

}  // namespace

namespace optix {

inline unsigned long long ptrToPrintable( void* ptr )
{
    return reinterpret_cast<unsigned long long>( ptr );
}

// NOTE: Don't change this formatting in order to keep OAC traces working.
static std::string toTraceCompatibleString( const RTmemoryblock& memoryBlock, const std::string& bufferFormat )
{
    return corelib::stringf(
        "fmt=%s baseaddr=0x%llx miplevel=%u x=%u y=%u z=%u w=%u h=%u d=%u rowpitch=%u planepitch=%u", bufferFormat.c_str(),
        ptrToPrintable( memoryBlock.baseAddress ), memoryBlock.mipLevel, memoryBlock.x, memoryBlock.y, memoryBlock.z,
        memoryBlock.width, memoryBlock.height, memoryBlock.depth, memoryBlock.rowPitch, memoryBlock.planePitch );
}

// Colwert a memory block to human readable form for logging.
std::string toString( const RTmemoryblock& memoryBlock )
{
    return "{ " + toTraceCompatibleString( memoryBlock, prodlib::toString( memoryBlock.format, 0U ) ) + " }";
}

// Colwert a memory block to a string for the OAC trace.
static std::string toTraceString( const RTmemoryblock& memoryBlock )
{
    return toTraceCompatibleString( memoryBlock, std::to_string( static_cast<int>( memoryBlock.format ) ) );
}

inline std::string toStringFromPtr( void* ptr )
{
    std::ostringstream str;
    str << "0x" << std::hex << ptrToPrintable( ptr );
    return str.str();
}

bool RequestHandler::s_stampMemoryBlocks{};

void RequestHandler::setStampMemoryBlocks( bool stampMemoryBlocks )
{
    s_stampMemoryBlocks = stampMemoryBlocks;
}

bool RequestHandler::isSmallNonMipmappedTextureRequest( unsigned pageId, unsigned startPage, const TextureSampler* sampler )
{
    const unsigned int pageIndex = pageId - startPage;
    const Buffer*      buffer    = sampler == nullptr ? nullptr : sampler->getBuffer();
    return sampler != nullptr && pageIndex == 0 && buffer->getMipLevelCount() == 1
           && buffer->getWidth() < sampler->getTileWidth() && buffer->getHeight() < sampler->getTileHeight();
}

// Page 0 of a demand-loaded texture represents the "mip tail", which is a complete texture
// containing all the miplevels that are tile-sized (or smaller).
bool RequestHandler::isMipTailRequest( unsigned int pageId, unsigned int startPage, const Buffer* buffer )
{
    // Exception: a pageId in a non-mipmapped texture is simply the (zero-based) tile offset.
    // Non-mipmapped textures do not have a mip tail.
    unsigned int pageIndex = pageId - startPage;
    return pageIndex == 0 && buffer->hasTextureAttached() && buffer->getMipLevelCount() > 1;
}

RequestHandler::RequestHandler( unsigned int pageId, DeviceSet devices, const Buffer* buffer, unsigned int startPage )
    : m_pageId( pageId )
    , m_devices( devices )
    , m_buffer( buffer )
    , m_startPage( startPage )
{
    RT_ASSERT_MSG( m_buffer != nullptr, "No demand buffer associated with requested page" );
    RT_ASSERT( m_buffer->isDemandLoad() );
}

static RTbuffer api_cast( const Buffer* ptr )
{
    return reinterpret_cast<RTbuffer>( const_cast<Buffer*>( ptr ) );
}

std::string RequestHandler::getCallbackTraceString( bool wasFilled, const Buffer* buffer, const RTmemoryblock& memoryBlock ) const
{
    return corelib::stringf( "  callback = filled=%d callback=0x%llx cbdata=0x%llx buffer=0x%llx %s\n",
                             wasFilled ? 1 : 0, ptrToPrintable( reinterpret_cast<void*>( buffer->getCallback() ) ),
                             ptrToPrintable( buffer->getCallbackData() ), ptrToPrintable( api_cast( buffer ) ),
                             toTraceString( memoryBlock ).c_str() );
}

void RequestHandler::copyDataToDevices( DeviceManager* dm, StagingPageAllocator* stagingPages, std::vector<DevicePaging>& devicePaging, bool synchronous )
{
    if( m_isFilled )
    {
        LOG_MEDIUM_VERBOSE( "RequestHandler::copyDataToDevices page " << m_pageId << '\n' );

        // Ensure that all data is synchronized to all devices for this request,
        // one request at a time.
        //
        // Synchronizing a request may require TileArrays and TilePools to be allocated
        // via TileManager::acquireTilePool.  TileManager::acquireTilePool manipulates
        // data structures in TileManager and may also create a TilePool.  TilePool::allocate
        // manipulates data structures in TilePool and may also create a TileArray (via TileManager).
        // Therefore, we need a mutex that isn't tied to a specific instance of TilePool, TileArray,
        // or RequestHandler, but spans the entire tile system.  This ensures that the page requests
        // being processed on multiple worker threads don't interfere with each other and the tile system
        // is consistent from the point of view of the worker threads.
        //
        static std::mutex            s_mutex;
        std::unique_lock<std::mutex> lock( s_mutex );

        for( unsigned int allDeviceListIndex : m_devices )
        {
            makeLwrrent( dm, allDeviceListIndex );
            synchronize( stagingPages, allDeviceListIndex, devicePaging[allDeviceListIndex], synchronous );
        }
    }
}

void RequestHandler::makeLwrrent( DeviceManager* dm, unsigned int allDeviceListIndex )
{
    deviceCast<LWDADevice>( dm->allDevices()[allDeviceListIndex] )->makeLwrrent();
}

void RequestHandler::fillMemoryBlock( StagingPageAllocator* stagingPages )
{
    // Create memory block and allocate staging memory for the callback.
    createMemoryBlock( stagingPages );

    m_isFilled = false;
    if( m_memoryBlock.baseAddress != nullptr )
    {
        fillMemoryBlockDebugPattern();
        ilwokeCallback( m_memoryBlock );
        logCallback( "RequestHandler" );
        checkMemoryBlockDebugPattern();
        stampMemoryBlock();
    }
}

bool RequestHandler::processRequest( DeviceManager* dm, StagingPageAllocator* stagingPages, std::vector<DevicePaging>& deviceState, bool synchronous )
{
    LOG_MEDIUM_VERBOSE( "RequestHandler::processRequest page " << m_pageId << '\n' );

    fillMemoryBlock( stagingPages );

    if( m_isFilled )
    {
        copyDataToDevices( dm, stagingPages, deviceState, synchronous );
    }

    releaseMemoryBlock( stagingPages );

    return m_isFilled;
}

void RequestHandler::captureTrace( ApiCapture& capture ) const
{
    if( !capture.capture_enabled() )
        return;

    capture.capture( getCallbackTraceString( m_isFilled, m_buffer, m_memoryBlock ) );

    if( m_isFilled )
    {
        // capture filled data
        const size_t size = m_memoryBlock.width * m_memoryBlock.height * m_buffer->getElementSize();
        capture.capture_buffer( size, m_memoryBlock.baseAddress, "filldata", "cbdata", nullptr );
    }
}

std::string RequestHandler::createExceptionMessage() const
{
    return corelib::stringf(
        "Demand load buffer callback at 0x%llx threw an exception when ilwoked with (0x%llx, 0x%llx, "
        "{ %d, 0x%llx, %u, %u, %u, %u, %u, %u, %u, %u, %u })",
        ptrToPrintable( reinterpret_cast<void*>( m_buffer->getCallback() ) ),
        ptrToPrintable( m_buffer->getCallbackData() ), ptrToPrintable( api_cast( m_buffer ) ),
        static_cast<int>( m_memoryBlock.format ), ptrToPrintable( m_memoryBlock.baseAddress ), m_memoryBlock.mipLevel,
        m_memoryBlock.x, m_memoryBlock.y, m_memoryBlock.z, m_memoryBlock.width, m_memoryBlock.height,
        m_memoryBlock.depth, m_memoryBlock.rowPitch, m_memoryBlock.planePitch );
}

void RequestHandler::ilwokeCallback( RTmemoryblock& block )
{
    const RTbuffercallback callbackFn = m_buffer->getCallback();
    m_caughtException                 = false;
    m_isFilled                        = false;
    try
    {
        m_isFilled = callbackFn( m_buffer->getCallbackData(), api_cast( m_buffer ), &block ) != 0;
    }
    catch( const std::exception& e )
    {
        m_caughtException  = true;
        m_exceptionMessage = e.what() + std::string( " thrown by callback" );
    }
    catch( ... )
    {
        m_caughtException  = true;
        m_exceptionMessage = "Unknown exception thrown by callback";
    }
}

void RequestHandler::releaseMemoryBlock( StagingPageAllocator* stagingPages )
{
    LOG_MEDIUM_VERBOSE( "RequestHandler::releaseMemoryBlock page " << m_pageId << '\n' );
    stagingPages->releasePage( m_allocation );
}

void RequestHandler::stampMemoryBlock( const RTmemoryblock& block, unsigned int id, unsigned int xOffset, unsigned int yOffset )
{
    if( m_isFilled && s_stampMemoryBlocks )
    {
        stampMemoryBlockWithId( block, id, xOffset, yOffset );
    }
}

// If the knob is on, then fill with a debug pattern that we will look for after the
// callback returns.  Otherwise in a debug or develop build, fill the memory block with
// red for the pixel format. To allow visual detection of skipped pixels during fill.
void RequestHandler::fillMemoryBlockDebugPattern( const RTmemoryblock& block )
{
    if( k_stampMemoryBlockDebugPattern.get() )
    {
        stampMemoryBlockWithDebugPattern( block );
    }
#if( defined( DEBUG ) || defined( DEVELOP ) )
    else
    {
        stampMemoryBlockWithRed( block );
    }
#endif
}

static std::string toByteOffsetString( const std::vector<unsigned int>& byteOffsets )
{
    std::string offsets;
    for( unsigned int offset : byteOffsets )
    {
        if( !offsets.empty() )
        {
            offsets += ", ";
        }
        offsets += std::to_string( offset );
    }
    return offsets;
}

static std::string toPixelCoordString( const std::vector<unsigned int>& pixelCoords )
{
    std::string coords;
    bool        firstCoord = true;
    for( unsigned int coord : pixelCoords )
    {
        if( !coords.empty() && firstCoord )
        {
            coords += ", ";
        }
        coords += std::to_string( coord );
        if( firstCoord )
        {
            coords += ',';
        }
        firstCoord = !firstCoord;
    }
    return coords;
}

void RequestHandler::checkMemoryBlockDebugPattern( const RTmemoryblock& block )
{
    if( m_isFilled && k_stampMemoryBlockDebugPattern.get() )
    {
        std::vector<unsigned int> byteOffsets;
        std::vector<unsigned int> pixelCoords;
        unsigned int              pixelCount = 0;
        const unsigned int byteCount = checkMemoryBlockForDebugPattern( block, byteOffsets, pixelCoords, pixelCount );
        if( byteCount > 0 && isLogActive() )
        {
            LOG_NORMAL( "RequestHandler::checkMemoryBlockDebugPattern page "
                        << m_pageId << ", allocId " << m_allocation.id << ", block " << toString( block ) << " found "
                        << byteCount << " debug bytes, " << pixelCount << " unfilled pixels\n" );
            LOG_NORMAL( "RequestHandler::checkMemoryBlockDebugPattern page " << m_pageId << ", allocId " << m_allocation.id << " byte offsets "
                                                                             << toByteOffsetString( byteOffsets ) << '\n' );
            LOG_NORMAL( "RequestHandler::checkMemoryBlockDebugPattern page " << m_pageId << ", allocId " << m_allocation.id << " pixel coords "
                                                                             << toPixelCoordString( pixelCoords ) << '\n' );
        }
    }
}

void RequestHandler::logCallback( const char* classLabel ) const
{
    logCallback( classLabel, m_memoryBlock );
}

void RequestHandler::logCallback( const char* classLabel, const RTmemoryblock& memoryBlock ) const
{
    LOG_MEDIUM_VERBOSE( classLabel << "::ilwokeCallback page " << m_pageId << " ("
                                   << toStringFromPtr( reinterpret_cast<void*>( m_buffer->getCallback() ) ) << ", "
                                   << toStringFromPtr( m_buffer->getCallbackData() ) << ", "
                                   << toStringFromPtr( api_cast( m_buffer ) ) << ", { " << toString( memoryBlock )
                                   << " } ) = " << std::boolalpha << m_isFilled << '\n' );
}

}  // namespace optix
