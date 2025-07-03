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

#include <Memory/DemandLoad/RequestHandler/HardwareTileRequestHandler.h>

#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Memory/DemandLoad/RequestHandler/BlockStamper.h>
#include <Memory/DemandLoad/StagingPageAllocator.h>
#include <Memory/DemandLoad/TileIndexing.h>
#include <Objects/Buffer.h>

using namespace prodlib;
using namespace optix::demandLoad;

namespace optix {

static RTbuffer api_cast( const Buffer* ptr )
{
    return reinterpret_cast<RTbuffer>( const_cast<Buffer*>( ptr ) );
}

HardwareTileRequestHandler::HardwareTileRequestHandler( unsigned int pageId, DeviceSet devices, const Buffer* buffer, unsigned int startPage )
    : RequestHandler( pageId, devices, buffer, startPage )
{
    // For now we support only 2D demand-loaded textures, and tiles are always square.
    // (The tile size is controlled by a knob; it is not yet configurable via the OptiX API.)
    RT_ASSERT( m_buffer->getDimensionality() == 2 );
}

static bool isPowerOfTwo( unsigned int value )
{
    return ( value & ( value - 1 ) ) == 0;
}

// Callwlate memory block for a page in a demand-loaded buffer or texture (other than the mip tail).
void HardwareTileRequestHandler::createMemoryBlock( StagingPageAllocator* stagingPages )
{
    LOG_MEDIUM_VERBOSE( "HardwareTileRequestHandler::createMemoryBlock page " << m_pageId << '\n' );
    const unsigned int tileIndex = getPageIndex();

    // Get the tile dimensions from the sparse texture properties, which are guaranteed to be the same on all devices
    // (m_devices[0] is the all-device-index of the first device that requested the tile).
    LWDA_ARRAY_SPARSE_PROPERTIES textureProps = m_buffer->getMBuffer()->getSparseTextureProperties( m_devices[0] );
    const unsigned int           tileWidth    = textureProps.tileExtent.width;
    const unsigned int           tileHeight   = textureProps.tileExtent.height;

    // We only support power of 2 tile sizes
    RT_ASSERT( isPowerOfTwo( tileWidth ) );
    RT_ASSERT( isPowerOfTwo( tileHeight ) );

    unsigned int mipLevel;
    unsigned int pixelX;
    unsigned int pixelY;
    TileIndexing( m_buffer->getWidth(), m_buffer->getHeight(), tileWidth, tileHeight )
        .unpackTileIndex( tileIndex, textureProps.miptailFirstLevel, mipLevel, pixelX, pixelY );

    // Sanity check and clamp miplevel.  See bug 3228650.
    const BufferDimensions dimensions = m_buffer->getDimensions();
#if defined( DEBUG ) || defined( DEVELOP )
    RT_ASSERT_MSG( mipLevel < dimensions.mipLevelCount(), "unpackTileLevel returned invalid miplevel" );
#endif
    m_memoryBlock.mipLevel = std::min( mipLevel, dimensions.mipLevelCount() - 1 );
    m_memoryBlock.x        = pixelX;
    m_memoryBlock.y        = pixelY;
    m_memoryBlock.z        = 0;

    // Make sure we don't request memory outside of the mip level.
    m_memoryBlock.width = clamp( tileWidth, 0U, static_cast<unsigned int>( dimensions.levelWidth( m_memoryBlock.mipLevel ) )
                                                    - m_memoryBlock.x );
    m_memoryBlock.height = clamp( tileHeight, 0U, static_cast<unsigned int>( dimensions.levelHeight( m_memoryBlock.mipLevel ) )
                                                      - m_memoryBlock.y );
    m_memoryBlock.depth      = 1;
    m_memoryBlock.rowPitch   = m_memoryBlock.width * m_buffer->getElementSize();  // in bytes.
    m_memoryBlock.planePitch = 0;
    m_memoryBlock.format     = m_buffer->getFormat();

    const unsigned int size = m_memoryBlock.width * m_memoryBlock.height * m_memoryBlock.depth * m_buffer->getElementSize();
    m_allocation            = stagingPages->acquirePage( size );
    m_memoryBlock.baseAddress = m_allocation.address;
}

void HardwareTileRequestHandler::synchronize( StagingPageAllocator* stagingPages,
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
        LOG_MEDIUM_VERBOSE( "HardwareTileRequestHandler::synchronize page " << m_pageId << ", device " << allDeviceListIndex << '\n' );

        if( synchronous )
        {
            devicePaging.bindTileToMemory( stagingPages, m_allocation, m_buffer, m_memoryBlock );
        }
        else
        {
            devicePaging.bindTileToMemoryAsync( stagingPages, m_allocation, m_buffer, m_memoryBlock );
        }
        devicePaging.addPageMapping( PageMapping{m_pageId, 1} );
    }
}

}  // namespace optix
