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

#include <Memory/DemandLoad/RequestHandler/TextureTileRequestHandler.h>

#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Memory/DemandLoad/RequestHandler/BlockStamper.h>
#include <Memory/DemandLoad/StagingPageAllocator.h>
#include <Memory/DemandLoad/TileIndexing.h>
#include <Objects/Buffer.h>
#include <c-api/ApiCapture.h>

using namespace optix::demandLoad;

namespace optix {

TextureTileRequestHandler::TextureTileRequestHandler( unsigned int          pageId,
                                                      DeviceSet             devices,
                                                      const Buffer*         buffer,
                                                      const TextureSampler* sampler,
                                                      unsigned int          startPage )
    : RequestHandler( pageId, devices, buffer, startPage )
    , m_sampler( sampler )
{
    RT_ASSERT_MSG( sampler != nullptr, "No texture sampler associated with requested page" );
    // For now we support only 2D demand-loaded textures, and tiles are always square.
    // (The tile size is controlled by a knob; it is not yet configurable via the OptiX API.)
    RT_ASSERT( m_buffer->getDimensionality() == 2 );

    m_borderColor.resize( m_buffer->getElementSize(), 0 );
}

void TextureTileRequestHandler::captureTrace( ApiCapture& capture ) const
{
    if( !capture.capture_enabled() )
        return;

    for( unsigned int i = 0; i < m_numReadRegions; ++i )
    {
        const RTmemoryblock& lwrrBlock = m_callbackMemoryBlocks[i];

        capture.capture( getCallbackTraceString( m_isFilled, m_buffer, lwrrBlock ) );

        if( m_isFilled )
        {
            // capture filled data
            const size_t size = lwrrBlock.width * lwrrBlock.height * m_buffer->getElementSize();
            capture.capture_buffer( size, lwrrBlock.baseAddress, "filldata", "cbdata", nullptr );
        }
    }
}

inline bool isPowerOfTwo( unsigned int value )
{
    return ( value & ( value - 1 ) ) == 0;
}

inline const char* transposeToCString( bool value )
{
    return value ? "Y" : "n";
}

static std::string toString( const ReadRegion& region )
{
    return corelib::stringf( "{ pos=%d,%d, off=%d,%d, %dx%d, xpose=%s,%s }", region.imageX(), region.imageY(),
                             region.destinationBufferXOffset(), region.destinationBufferYOffset(), region.width(),
                             region.height(), transposeToCString( region.transposeX() ),
                             transposeToCString( region.transposeY() ) );
}

static std::string toString( const ReadRegion* readRegions, unsigned int numReadRegions )
{
    std::string regions;
    for( unsigned int i = 0; i < numReadRegions; ++i )
    {
        if( !regions.empty() )
        {
            regions += ", ";
        }
        regions += toString( readRegions[i] );
    }
    return "{ " + regions + " }";
}

// Callwlate memory block for a page in a demand-loaded buffer or texture (other than the mip tail).
void TextureTileRequestHandler::createMemoryBlock( StagingPageAllocator* stagingPages )
{
    // Get texel coordinates for tile block.
    const unsigned int tileWidth  = m_sampler->getTileWidth();
    const unsigned int tileHeight = m_sampler->getTileHeight();
    unsigned int       mipLevel;
    unsigned int       pixelX;
    unsigned int       pixelY;
    TileIndexing( m_buffer->getWidth(), m_buffer->getHeight(), tileWidth, tileHeight )
        .unpackTileIndex( getPageIndex(), m_buffer->getMipTailFirstLevel(), mipLevel, pixelX, pixelY );

    // Allocate space large enough to hold the texture tile with gutters.
    const unsigned int gutterSize                  = m_sampler->getGutterWidth();
    const unsigned int gutteredTileWidth           = tileWidth + 2 * gutterSize;
    const unsigned int gutteredTileHeight          = tileHeight + 2 * gutterSize;
    const unsigned int gutteredTileRowPitchInBytes = gutteredTileWidth * m_buffer->getElementSize();
    const unsigned int gutteredTileSizeInBytes = gutteredTileWidth * gutteredTileHeight * m_buffer->getElementSize();
    m_allocation                               = stagingPages->acquirePage( gutteredTileSizeInBytes );

    // Block describing tile interior.
    // TODO: the clamp/wrap code is reading the x, y values and casting them back to an int.
    // Communicate these values as ints outside of the memory block.
    // The regionStartX/Y can be negative for tiles at the edge of the texture and they look "wrong" in the log.
    const int regionStartX = static_cast<int>( pixelX ) - static_cast<int>( gutterSize );
    const int regionStartY = static_cast<int>( pixelY ) - static_cast<int>( gutterSize );
    {
        RTmemoryblock block{};
        block.format      = m_buffer->getFormat();
        block.mipLevel    = mipLevel;
        block.x           = static_cast<unsigned int>( regionStartX );
        block.y           = static_cast<unsigned int>( regionStartY );
        block.width       = gutteredTileWidth;
        block.height      = gutteredTileHeight;
        block.depth       = 1;
        block.rowPitch    = gutteredTileRowPitchInBytes;
        block.baseAddress = m_allocation.address;
        m_memoryBlock     = block;
    }

    {
        const RTwrapmode wrapModeX   = m_sampler->getWrapMode( 0 );
        const RTwrapmode wrapModeY   = m_sampler->getWrapMode( 1 );
        const int        imageWidth  = static_cast<int>( m_buffer->getLevelWidth( mipLevel ) );
        const int        imageHeight = static_cast<int>( m_buffer->getLevelHeight( mipLevel ) );
        m_numReadRegions =
            determineWrappedReadRegions( wrapModeX, wrapModeY, regionStartX, regionStartY, static_cast<int>( gutteredTileWidth ),
                                         static_cast<int>( gutteredTileHeight ), imageWidth, imageHeight, m_regionsToRead );
    }

    // Blocks describing sub-regions (which might be wrapped) that are used to ilwoke client callbacks.
    for( unsigned int i = 0; i < m_numReadRegions; ++i )
    {
        const ReadRegion& region    = m_regionsToRead[i];
        RTmemoryblock&    lwrrBlock = m_callbackMemoryBlocks[i];

        lwrrBlock.x          = region.imageX();
        lwrrBlock.y          = region.imageY();
        lwrrBlock.z          = 0;
        lwrrBlock.width      = region.width();
        lwrrBlock.height     = region.height();
        lwrrBlock.depth      = 1;
        lwrrBlock.rowPitch   = gutteredTileRowPitchInBytes;
        lwrrBlock.planePitch = 0;
        lwrrBlock.mipLevel   = mipLevel;
        lwrrBlock.format     = m_buffer->getFormat();

        lwrrBlock.baseAddress = static_cast<uint8_t*>( m_allocation.address ) + region.destinationBufferYOffset() * gutteredTileRowPitchInBytes
                                + region.destinationBufferXOffset() * m_buffer->getElementSize();
    }

    LOG_MEDIUM_VERBOSE( "TextureTileRequestHandler::createMemoryBlock page "
                        << m_pageId << ", allocId " << m_allocation.id << ", tile block " << toString( m_memoryBlock )
                        << ", regions " << m_numReadRegions << ' ' << toString( m_regionsToRead, m_numReadRegions ) << '\n' );
    if( isLogVerboseActive() )
    {
        std::string blocks;
        for( unsigned int i = 0; i < m_numReadRegions; ++i )
        {
            if( !blocks.empty() )
            {
                blocks += ", ";
            }
            blocks += toString( m_callbackMemoryBlocks[i] );
        }
        LOG_VERBOSE( "TextureTileRequestHandler::createMemoryBlock page " << m_pageId << ", allocId " << m_allocation.id
                                                                          << ", callback blocks " << blocks << '\n' );
    }
}

void TextureTileRequestHandler::fillMemoryBlock( optix::StagingPageAllocator* stagingPages )
{
    LOG_MEDIUM_VERBOSE( "TextureTileRequestHandler::fillMemoryBlock page " << m_pageId << '\n' );

    createMemoryBlock( stagingPages );

    // The returned memory block has a null base address if a staging area could not be allocated.
    // For now, we simply stop filling page requests when that happens.
    m_isFilled = false;
    if( m_memoryBlock.baseAddress != nullptr )
    {
        // Ilwoke the callbacks on the regions to fill the image data.
        m_isFilled = true;
        for( unsigned int i = 0; m_isFilled && i < m_numReadRegions; ++i )
        {
            fillMemoryBlockDebugPattern( m_callbackMemoryBlocks[i] );
            ilwokeCallback( m_callbackMemoryBlocks[i] );
            logCallback( "TextureTileRequestHandler", m_callbackMemoryBlocks[i] );
            checkMemoryBlockDebugPattern( m_callbackMemoryBlocks[i] );
        }
    }

    if( m_isFilled )
    {
        // Fill in gutters from pixels already read
        const RTwrapmode wrapModeX   = m_sampler->getWrapMode( 0 );
        const RTwrapmode wrapModeY   = m_sampler->getWrapMode( 1 );
        const int        imageWidth  = m_buffer->getLevelWidth( m_memoryBlock.mipLevel );
        const int        imageHeight = m_buffer->getLevelHeight( m_memoryBlock.mipLevel );

        // Mirror modes
        const unsigned int elementSize = m_buffer->getElementSize();
        const unsigned int rowPitch    = elementSize * m_memoryBlock.width;
        if( wrapModeX == RT_WRAP_MIRROR || wrapModeY == RT_WRAP_MIRROR )
        {
            for( unsigned int i = 0; i < m_numReadRegions; ++i )
            {
                transposeRegion( m_regionsToRead[i], m_callbackMemoryBlocks[i], elementSize, rowPitch );
            }
        }

        // Clamping modes
        if( wrapModeX == RT_WRAP_CLAMP_TO_BORDER || wrapModeX == RT_WRAP_CLAMP_TO_EDGE )
        {
            void* srcColor = ( wrapModeX == RT_WRAP_CLAMP_TO_BORDER ) ? m_borderColor.data() : nullptr;
            clampRegionHorizontal( m_memoryBlock, elementSize, rowPitch, imageWidth, imageHeight, srcColor );
        }

        if( wrapModeY == RT_WRAP_CLAMP_TO_BORDER || wrapModeY == RT_WRAP_CLAMP_TO_EDGE )
        {
            void* srcColor = ( wrapModeY == RT_WRAP_CLAMP_TO_BORDER ) ? m_borderColor.data() : nullptr;
            clampRegiolwertical( m_memoryBlock, elementSize, rowPitch, imageWidth, imageHeight, srcColor );
        }

        // Nothing needs to be done for WRAP_REPEAT

        stampMemoryBlock( m_memoryBlock, m_allocation.id, m_sampler->getGutterWidth(), m_sampler->getGutterWidth() );
    }
}

void TextureTileRequestHandler::synchronize( StagingPageAllocator* stagingPages,
                                             unsigned int          allDeviceListIndex,
                                             DevicePaging&         devicePaging,
                                             bool                  synchronous ) const
{
    // Do nothing if this page wasn't requested on the specified device.
    if( !m_devices.isSet( static_cast<int>( allDeviceListIndex ) ) )
        return;

    // We might not have ilwoked the callback (if, for example, we ran out of staging memory).
    if( m_isFilled )
    {
        LOG_MEDIUM_VERBOSE( "TextureTileRequestHandler::synchronize page "
                            << m_pageId << ", device " << allDeviceListIndex << ", id " << m_allocation.id << '\n' );

        const TileLocator locator =
            synchronous ? devicePaging.copyTileToDevice( stagingPages, m_allocation, m_memoryBlock, m_buffer ) :
                          devicePaging.copyTileToDeviceAsync( stagingPages, m_allocation, m_memoryBlock, m_buffer );
        devicePaging.addPageMapping( PageMapping{m_pageId, static_cast<unsigned long long>( locator.packed )} );
    }
}
}  // namespace optix
