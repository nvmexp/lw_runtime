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

#include <Memory/DemandLoad/TilePool.h>

#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Memory/DemandLoad/TileManager.h>

using namespace optix::demandLoad;

namespace optix {

TilePool::TilePool( MemoryManager* mm, TileManager* tileManager, unsigned int tileWidth, unsigned int tileHeight, RTformat format, const TextureDescriptor& descriptor )
    : m_mm( mm )
    , m_tileManager( tileManager )
    , m_tileWidth( tileWidth )
    , m_tileHeight( tileHeight )
    , m_format( format )
    , m_descriptor( descriptor )
{
    LOG_MEDIUM_VERBOSE( "TilePool::TilePool " << tileWidth << 'x' << tileHeight << ", " << format << '\n' );

    m_nextTile.unpacked.tileArray =
        m_tileManager->createTileArray( m_tileArraySize, m_tileWidth, m_tileHeight, m_format, m_descriptor );
    m_nextTile.unpacked.tileIndex = 0;
}

TileLocator TilePool::allocate()
{
    LOG_MEDIUM_VERBOSE( "TilePool::allocate\n" );

    // If possible, use the next available tile from the current TileArray.
    if( static_cast<unsigned int>( m_nextTile.unpacked.tileIndex ) + 1 < m_tileArraySize )
    {
        TileLocator result( m_nextTile );
        ++m_nextTile.unpacked.tileIndex;
        return result;
    }

    // Create a new tile array.
    m_nextTile.unpacked.tileArray =
        m_tileManager->createTileArray( m_tileArraySize, m_tileWidth, m_tileHeight, m_format, m_descriptor );
    m_nextTile.unpacked.tileIndex = 0;
    return m_nextTile;
}

}  // namespace optix
