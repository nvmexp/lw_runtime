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

#pragma once

#include <Memory/DemandLoad/TileArray.h>
#include <Memory/DemandLoad/TilePool.h>
#include <Memory/MBufferDeviceVector.h>
#include <Util/TextureDescriptor.h>

#include <prodlib/exceptions/Assert.h>

#include <cstring>
#include <vector>

namespace optix {

class MemoryManager;

/// The TileManager provides TilePools for tiles with various dimensions, formats, and texture options.
/// It also serves as a factory for TileArrays, which are allocated by TilePools.
class TileManager
{
  public:
    TileManager() = default;

    void activate( MemoryManager* mm, unsigned int allDeviceListIndex );
    void deactivate();

    /// Construct a tile array, allocating device memory.  Returns the TileArray index.
    unsigned int createTileArray( unsigned int             numTiles,
                                  unsigned int             tileWidth,
                                  unsigned int             tileHeight,
                                  RTformat                 format,
                                  const TextureDescriptor& descriptor ) noexcept;

    /// Get the tile array with the specified index.
    TileArray* getTileArray( unsigned int index )
    {
        RT_ASSERT( index < m_tileArrays.size() );
        return &m_tileArrays[index];
    }

    /// Synchronize samplers to the specified device if necessary.
    void synchronize();

    LWtexObject* getDevTexObjects( unsigned int allDeviceListIndex ) const
    {
        return m_devTexObjects.getDevicePtr( allDeviceListIndex );
    }

    //. Get the TilePool for tiles with the specified dimensions, format, and texture options.
    /// Creates a new TilePool if necessary.  Then create a allocate a tile within that pool
    /// and return the locator for that tile.
    TileLocator allocateTileWithinPool( unsigned int tileWidth, unsigned int tileHeight, RTformat format, const TextureDescriptor& descriptor );

  private:
    TilePool* acquireTilePool( unsigned int tileWidth, unsigned int tileHeight, RTformat format, const TextureDescriptor& descriptor );

    // TilePoolInfo keeps track of the dimensions, format, and texture options for each TilePool.
    struct TilePoolInfo
    {
        unsigned int      tileWidth;
        unsigned int      tileHeight;
        RTformat          format;
        TextureDescriptor descriptor;

        bool operator==( const TilePoolInfo& other ) const
        {
            return 0 == std::memcmp( this, &other, sizeof( TilePoolInfo ) );
        }
    };

    MemoryManager*                   m_mm                 = nullptr;
    unsigned int                     m_allDeviceListIndex = 0U;
    std::vector<TilePoolInfo>        m_poolInfo;
    std::vector<TilePool>            m_pools;
    std::vector<TileArray>           m_tileArrays;
    MBufferDeviceVector<LWtexObject> m_devTexObjects;
};

}  // namespace optix
