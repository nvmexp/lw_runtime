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

#include <Memory/DemandLoad/TileManager.h>

#include <Memory/DemandLoad/PagingServiceLogging.h>

#include <prodlib/exceptions/Assert.h>

#include <private/optix_6_enum_printers.h>

#include <mutex>

namespace {

const unsigned int MAX_NUM_TILE_ARRAYS = 1024;

}  // namespace

using namespace optix::demandLoad;

namespace optix {

void TileManager::activate( MemoryManager* mm, unsigned int allDeviceListIndex )
{
    LOG_MEDIUM_VERBOSE( "TileManager::activate device " << allDeviceListIndex << '\n' );

    m_mm                 = mm;
    m_allDeviceListIndex = allDeviceListIndex;
    const DeviceSet allowedDevices( static_cast<int>( allDeviceListIndex ) );
    m_devTexObjects = MBufferDeviceVector<unsigned long long>( mm, allowedDevices, MAX_NUM_TILE_ARRAYS );
}

void TileManager::deactivate()
{
    LOG_MEDIUM_VERBOSE( "TileManager::deactivate device " << m_allDeviceListIndex << '\n' );

    m_devTexObjects.reset();
}

// Get the TilePool for tiles with the specified dimensions, format, and texture options.
// Creates a new TilePool if necessary.
TilePool* TileManager::acquireTilePool( unsigned int tileWidth, unsigned int tileHeight, RTformat format, const TextureDescriptor& descriptor )
{
    LOG_MEDIUM_VERBOSE( "TileManager::acquireTilePool device " << m_allDeviceListIndex << ", " << tileWidth << 'x'
                                                               << tileHeight << ", " << format << '\n' );


    // Look for an existing pool with the specified options.
    TilePoolInfo info{tileWidth, tileHeight, format, descriptor};
    RT_ASSERT( m_poolInfo.size() == m_pools.size() );
    for( size_t i = 0; i < m_poolInfo.size(); ++i )
    {
        if( m_poolInfo[i] == info )
            return &m_pools[i];
    }
    // Not found.  Create a new pool.
    m_pools.emplace_back( m_mm, this, tileWidth, tileHeight, format, descriptor );
    m_poolInfo.push_back( info );
    return &m_pools.back();
}


/// Construct a tile array, allocating device memory.  Returns the TileArray index.
unsigned int TileManager::createTileArray( unsigned int             numTiles,
                                           unsigned int             tileWidth,
                                           unsigned int             tileHeight,
                                           RTformat                 format,
                                           const TextureDescriptor& descriptor ) noexcept
{
    LOG_MEDIUM_VERBOSE( "TileManager::createTileArray device " << m_allDeviceListIndex << ", numTiles " << numTiles << ", "
                                                               << tileWidth << 'x' << tileHeight << ", " << format << '\n' );

    m_tileArrays.emplace_back( m_mm, numTiles, tileWidth, tileHeight, format, descriptor );

    return static_cast<unsigned int>( m_tileArrays.size() - 1 );
}

void TileManager::synchronize()
{
    LOG_MEDIUM_VERBOSE( "TileManager::synchronize device " << m_allDeviceListIndex << '\n' );

    std::vector<LWtexObject> handles;
    handles.reserve( m_tileArrays.size() );
    for( TileArray& tileArray : m_tileArrays )
    {
        tileArray.synchronize( m_allDeviceListIndex );
        handles.push_back( tileArray.getTexObject() );
    }

    if( !handles.empty() )
    {
        m_devTexObjects.copyToDevice( m_mm, &handles[0], static_cast<unsigned int>( handles.size() ) );
    }
}

static std::string toString( TileLocator value )
{
    return "{ " + std::to_string( value.unpacked.tileArray ) + ", " + std::to_string( value.unpacked.tileIndex ) + " }";
}

TileLocator TileManager::allocateTileWithinPool( unsigned int tileWidth, unsigned int tileHeight, RTformat format, const TextureDescriptor& descriptor )
{
    LOG_MEDIUM_VERBOSE( "TileManager::allocateTileWithinPool device " << m_allDeviceListIndex << ", " << tileWidth
                                                                      << 'x' << tileHeight << ", " << format << '\n' );

    // TODO: Split responsibilities of TileManager and refactor into separate classes.
    // TilePool calls back into TileManager, because TileManager has two
    // responsibilities: manage the TilePools/TileArrays and create TileArrays.
    TilePool* tilePool = acquireTilePool( tileWidth, tileHeight, format, descriptor );

    // Allocate tile.
    const TileLocator result = tilePool->allocate();
    LOG_MEDIUM_VERBOSE( "TileManager::allocateTileWithinPool device " << m_allDeviceListIndex << ", result "
                                                                      << toString( result ) << '\n' );
    return result;
}

}  // namespace demandLoading
