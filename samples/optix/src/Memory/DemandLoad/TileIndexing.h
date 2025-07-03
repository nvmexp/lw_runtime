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

#pragma once

#include <ExelwtionStrategy/CORTTypes.h>

#ifndef TILE_INDEXING_DEVICE
#include <algorithm>
#include <cmath>
#endif

namespace optix {

class TileIndexing
{
  public:
    TileIndexing( unsigned int textureWidth, unsigned int textureHeight, unsigned int tileWidth, unsigned int tileHeight )
        : m_textureWidth( textureWidth )
        , m_textureHeight( textureHeight )
        , m_tileWidth( tileWidth )
        , m_tileHeight( tileHeight )
        , m_ilwTileWidth( 1.0f / static_cast<float>( tileWidth ) )
        , m_ilwTileHeight( 1.0f / static_cast<float>( tileHeight ) )
    {
    }

    void callwlateLevelDims( unsigned int mipLevel, unsigned int& levelWidth, unsigned int& levelHeight ) const
    {
        levelWidth  = uimax( m_textureWidth >> mipLevel, 1U );
        levelHeight = uimax( m_textureHeight >> mipLevel, 1U );
    }

    unsigned int callwlateNumPages( unsigned int mipTailFirstLevel ) const
    {
        // We also include a single tile for the mip tail.
        return mipTailFirstLevel == 0 ? 1 : 1 + getNumTilesInMipLevels( 0, mipTailFirstLevel - 1 );
    }

    unsigned int callwlateTileIndex( unsigned int                   mipLevel,
                                     int                            pixelX,
                                     int                            pixelY,
                                     ::lwca::lwdaTextureAddressMode wrapModeX,
                                     ::lwca::lwdaTextureAddressMode wrapModeY,
                                     unsigned int                   levelWidth,
                                     unsigned int                   levelHeight,
                                     unsigned int                   mipTailFirstLevel ) const
    {
        unsigned int tileX = callwlateWrappedTileCoord( wrapModeX, pixelX, levelWidth, m_ilwTileWidth );
        unsigned int tileY = callwlateWrappedTileCoord( wrapModeY, pixelY, levelHeight, m_ilwTileHeight );

        return callwlateTileIndexFromTileCoords( mipLevel, tileX, tileY, levelWidth, mipTailFirstLevel );
    }

    /// The caller of callwlateTileRequests must provide an output array with this capacity.
    static const unsigned int MAX_TILES_CALLWLATED = 4;

    void callwlateTileRequests( unsigned int                   mipLevel,
                                float                          normX,
                                float                          normY,
                                unsigned int                   mipTailFirstLevel,
                                unsigned int                   anisotropy,
                                ::lwca::lwdaTextureAddressMode wrapModeX,
                                ::lwca::lwdaTextureAddressMode wrapModeY,
                                // The output array must have a capacity of at least MAX_TILES_CALLWLATED.
                                unsigned int* outTilesToRequest,
                                unsigned int& outNumTilesToRequest ) const
    {
        outNumTilesToRequest = 0;

        // If the requested miplevel is in the mip tail, return a request for tile index zero.
        if( mipLevel >= mipTailFirstLevel )
        {
            outTilesToRequest[outNumTilesToRequest] = 0;
            ++outNumTilesToRequest;
            return;
        }

        unsigned int levelWidth;
        unsigned int levelHeight;
        callwlateLevelDims( mipLevel, levelWidth, levelHeight );

        int pixelX         = static_cast<int>( normX * levelWidth );
        int pixelY         = static_cast<int>( normY * levelHeight );
        int halfAnisotropy = anisotropy >> 1;

        // Compute the x and y tile coordinates for left, right, top, bottom
        unsigned int xTileCoords[2];
        unsigned int yTileCoords[2];
        xTileCoords[0] = callwlateWrappedTileCoord( wrapModeX, pixelX - halfAnisotropy, levelWidth, m_ilwTileWidth );
        xTileCoords[1] = callwlateWrappedTileCoord( wrapModeX, pixelX + halfAnisotropy, levelWidth, m_ilwTileWidth );
        yTileCoords[0] = callwlateWrappedTileCoord( wrapModeY, pixelY - halfAnisotropy, levelHeight, m_ilwTileHeight );
        yTileCoords[1] = callwlateWrappedTileCoord( wrapModeY, pixelY + halfAnisotropy, levelHeight, m_ilwTileHeight );

        // Set the loop bounds to avoid duplicate values
        int xmax = ( xTileCoords[0] == xTileCoords[1] ) ? 1 : 2;
        int ymax = ( yTileCoords[0] == yTileCoords[1] ) ? 1 : 2;

        // Add each unique tileIndex to the request array
        for( int j = 0; j < ymax; ++j )
        {
            for( int i = 0; i < xmax; ++i )
            {
                outTilesToRequest[outNumTilesToRequest] =
                    callwlateTileIndexFromTileCoords( mipLevel, xTileCoords[i], yTileCoords[j], levelWidth, mipTailFirstLevel );
                ++outNumTilesToRequest;
            }
        }
    }

    // Callwlate the tiles covered by the given set of texels.
    void callwlateTileRequestsFromTexels( unsigned int                   mipLevel,
                                          unsigned int*                  texelXCoords,
                                          unsigned int*                  texelYCoords,
                                          unsigned int                   numTexelsCovered,
                                          unsigned int                   xTexelSize,
                                          unsigned int                   yTexelSize,
                                          unsigned int                   mipTailFirstLevel,
                                          ::lwca::lwdaTextureAddressMode wrapModeX,
                                          ::lwca::lwdaTextureAddressMode wrapModeY,
                                          unsigned int*                  outTilesToRequest,
                                          unsigned int&                  outNumTilesToRequest ) const
    {
        outNumTilesToRequest = 0;

        if( mipLevel >= mipTailFirstLevel )
        {
            outTilesToRequest[outNumTilesToRequest] = 0;
            ++outNumTilesToRequest;
            return;
        }

        unsigned int levelWidth;
        unsigned int levelHeight;
        callwlateLevelDims( mipLevel, levelWidth, levelHeight );

        // For each texel, get the tile IDs for each of the four corners of the texel and add them to a unique array.
        // This assumes that a texel will never be larger than a texture tile.
        unsigned int xTileCoords[2];
        unsigned int yTileCoords[2];
        for( int i = 0; i < numTexelsCovered; ++i )
        {
            xTileCoords[0] = callwlateWrappedTileCoord( wrapModeX, texelXCoords[i], levelWidth, m_ilwTileWidth );
            xTileCoords[1] = callwlateWrappedTileCoord( wrapModeX, texelXCoords[i] + (xTexelSize - 1), levelWidth, m_ilwTileWidth );
            yTileCoords[0] = callwlateWrappedTileCoord( wrapModeY, texelYCoords[i], levelHeight, m_ilwTileHeight );
            yTileCoords[1] = callwlateWrappedTileCoord( wrapModeY, texelYCoords[i] + (yTexelSize - 1), levelHeight, m_ilwTileHeight );

            for( int xIndex = 0; xIndex < 2; ++xIndex )
            {
                for( int yIndex = 0; yIndex < 2; ++yIndex )
                {
                    unsigned int lwrrTile = callwlateTileIndexFromTileCoords( mipLevel, xTileCoords[xIndex], yTileCoords[yIndex],
                                                                              levelWidth, mipTailFirstLevel );

                    bool isPresent = false;
                    for( int i = 0; i < outNumTilesToRequest; ++i )
                    {
                        if( outTilesToRequest[i] == lwrrTile )
                        {
                            isPresent = true;
                            break;
                        }
                    }

                    if( !isPresent )
                        outTilesToRequest[outNumTilesToRequest++] = lwrrTile;
                }
            }
        }
    }

    void unpackTileIndex( unsigned int tileIndex, unsigned int mipTailFirstLevel, unsigned int& outMipLevel, unsigned int& outX, unsigned int& outY ) const
    {
        outMipLevel                = 0;
        outX                       = 0;
        outY                       = 0;
        unsigned int totalNumTiles = 1;  // one tile index is allocated to the mip tail.
        for( int mipLevel = mipTailFirstLevel - 1; mipLevel >= 0; --mipLevel )
        {
            unsigned int numTiles = numTilesInLevel( mipLevel );
            if( totalNumTiles + numTiles > tileIndex )
            {
                unsigned int levelWidth;
                unsigned int levelHeight;
                callwlateLevelDims( mipLevel, levelWidth, levelHeight );

                // We're rounding up to accommodate partial tiles.
                unsigned int widthInTiles = ceilMult( levelWidth, m_tileWidth, m_ilwTileWidth );

                unsigned int indexInLevel = tileIndex >= totalNumTiles ? tileIndex - totalNumTiles : 0;
                unsigned int tileY        = indexInLevel / widthInTiles;
                unsigned int tileX        = indexInLevel % widthInTiles;

                outMipLevel = mipLevel;
                outX        = tileX * m_tileWidth;
                outY        = tileY * m_tileHeight;
                return;
            }
            totalNumTiles += numTiles;
        }
    }

    void textureToSoftwareTileCoord( unsigned int                   tileGutterWidth,
                                     unsigned int                   levelWidth,
                                     unsigned int                   levelHeight,
                                     float                          normX,
                                     float                          normY,
                                     ::lwca::lwdaTextureAddressMode wrapModeX,
                                     ::lwca::lwdaTextureAddressMode wrapModeY,
                                     float&                         outNormX,
                                     float&                         outNormY ) const
    {
        // Wrap the given coordinates if we're not using border mode. (We use
        // the tiles' gutters if we are using borders).
        if( wrapModeX != ::lwca::lwdaAddressModeBorder )
            normX = wrapNormCoord( normX, wrapModeX );
        if( wrapModeY != ::lwca::lwdaAddressModeBorder )
            normY = wrapNormCoord( normY, wrapModeY );

        // Colwert from texture to tile coordinates
        unsigned int widthInTiles  = ceilMult( levelWidth, m_tileWidth, m_ilwTileWidth );
        unsigned int heightInTiles = ceilMult( levelHeight, m_tileHeight, m_ilwTileHeight );

        // Scale coordinates for partial tiles
        float partialTileXScale = static_cast<float>( levelWidth ) / static_cast<float>( widthInTiles * m_tileWidth );
        float partialTileYScale = static_cast<float>( levelHeight ) / static_cast<float>( heightInTiles * m_tileHeight );

        // Compute tile texture coordinates (outNormX and outNormY)
        float xInTileCoords = normX * partialTileXScale * widthInTiles;
        float yInTileCoords = normY * partialTileYScale * heightInTiles;

        float nearestX = floorf( clamp( xInTileCoords, 0.0f, widthInTiles - 1.0f) );
        float nearestY = floorf( clamp( yInTileCoords, 0.0f, heightInTiles - 1.0f) );

        outNormX = xInTileCoords - nearestX;
        outNormY = yInTileCoords - nearestY;

        // Colwert to tile w/o gutter -> gutter
        unsigned int tileWidthWithGutter  = m_tileWidth + tileGutterWidth * 2;
        unsigned int tileHeightWithGutter = m_tileHeight + tileGutterWidth * 2;

        outNormX = ( outNormX * m_tileWidth + tileGutterWidth ) / tileWidthWithGutter;
        outNormY = ( outNormY * m_tileHeight + tileGutterWidth ) / tileHeightWithGutter;
    }

    void getScaleFactorForTileDerivatives( unsigned int levelWidth, unsigned int levelHeight, float& outScaleX, float& outScaleY ) const
    {
        outScaleX = static_cast<float>( levelWidth ) / m_tileWidth;
        outScaleY = static_cast<float>( levelHeight ) / m_tileHeight;
    }

    static unsigned int wrapPixelCoord(::lwca::lwdaTextureAddressMode wrapMode, int coord, int max )
    {
        if( wrapMode == ::lwca::lwdaAddressModeClamp || wrapMode == ::lwca::lwdaAddressModeBorder )
        {
            return coord < 0 ? 0 : ( coord >= max ? max - 1 : coord );
        }
        else  // wrap and mirror modes
        {
            // Compute (floored) quotient and remainder
            int q = ifloor( static_cast<float>( coord ) / static_cast<float>( max ) );
            int r = clampi( coord - q * max, 0, max - 1 );
            // In mirror mode, flip the coordinate (r) if the q is odd
            return ( wrapMode == ::lwca::lwdaAddressModeMirror && ( q & 1 ) ) ? ( max - 1 - r ) : r;
        }
    }

    static inline float wrapNormCoord( float x, ::lwca::lwdaTextureAddressMode addressMode )
    {
        // Wrap mode
        if( addressMode == ::lwca::lwdaAddressModeWrap )
            return x - floorf( x );

        // Clamp and border modes (border color handled by gutters)
        if( addressMode == ::lwca::lwdaAddressModeClamp )
            return clamp( x, 0.0f, FIRST_FLOAT_LESS_THAN_ONE );  // result must be < 1

        // Border mode
        if( addressMode == ::lwca::lwdaAddressModeBorder )
            return clamp( x, 0.0f, FIRST_FLOAT_LESS_THAN_ONE );  // result must be < 1

        // Mirror mode
        if( static_cast<int>( floorf( x ) ) & 0x1 )  // flip odd parity tiles
        {
            float y = ceilf( x ) - x;
            // When the coordinate is an odd integer, ceil(x) - x returns 0, but should return near 1
            return ( y <= 0.0f ) ? FIRST_FLOAT_LESS_THAN_ONE : y;
        }
        else
        {
            return x - floorf( x );
        }
    }

    static inline unsigned int getSoftwareMipTailFirstLevel( unsigned int textureWidth,
                                                             unsigned int textureHeight,
                                                             unsigned int tileWidth,
                                                             unsigned int tileHeight )
    {
        // The first level of the software mip tail is the first tile in which
        // we can store the entire mip level.
        unsigned int mipTailFirstLevel = 0;
        unsigned int lwrrLevelWidth    = textureWidth;
        unsigned int lwrrLevelHeight   = textureHeight;
        while( lwrrLevelWidth > tileWidth || lwrrLevelHeight > tileHeight )
        {
            lwrrLevelWidth  = lwrrLevelWidth >> 1;
            lwrrLevelHeight = lwrrLevelHeight >> 1;
            mipTailFirstLevel++;
        }
        return mipTailFirstLevel;
    }

    static inline bool isMipTailIndex( unsigned int pageIndex )
    {
        // Page 0 always contains the mip tail.
        return pageIndex == 0;
    }

  private:
    constexpr static const float FIRST_FLOAT_LESS_THAN_ONE = 0.999999940395355224609375f;

    unsigned int m_textureWidth;
    unsigned int m_textureHeight;
    unsigned int m_tileWidth;
    unsigned int m_tileHeight;
    float        m_ilwTileWidth;
    float        m_ilwTileHeight;

#ifdef TILE_INDEXING_DEVICE
    static int ifloor( float x ) { return (int)lwca::floorf( x ); }
    static float floorf( float x ) { return lwca::floorf( x ); }
    static float ceilf( float x ) { return lwca::ceilf( x ); }
    static float maxf( float x, float y ) { return lwca::maxf( x, y ); }
    static float minf( float x, float y ) { return lwca::minf( x, y ); }
    static unsigned int uimax( unsigned int a, unsigned int b ) { return ( a > b ) ? a : b; }
#else
    static int ifloor( float x ) { return static_cast<int>( std::floor( x ) ); }
    static float floorf( float x ) { return std::floor( x ); }
    static float ceilf( float x ) { return std::ceil( x ); }
    static float maxf( float x, float y ) { return std::max( x, y ); }
    static float minf( float x, float y ) { return std::min( x, y ); }
    static unsigned int uimax( unsigned int a, unsigned int b ) { return std::max( a, b ); }
#endif
    static inline float clamp( float f, float a, float b ) { return maxf( a, minf( f, b ) ); }
    static inline int clampi( int i, int a, int b ) { return ( i < a ) ? a : ( i > b ? b : i ); }

    static unsigned int ceilMult( unsigned int levelDim, unsigned int tileDim, float ilwTileDim )
    {
        // This should work as long as (levelDim + tileDim - 1) < 16M
        return static_cast<unsigned int>( static_cast<float>( levelDim + tileDim - 1 ) * ilwTileDim );
    }

    unsigned int numTilesInLevel( unsigned int mipLevel ) const
    {
        unsigned int levelWidth;
        unsigned int levelHeight;
        callwlateLevelDims( mipLevel, levelWidth, levelHeight );

        // We're rounding up to accommodate partial tiles.
        unsigned int widthInTiles  = ceilMult( levelWidth, m_tileWidth, m_ilwTileWidth );
        unsigned int heightInTiles = ceilMult( levelHeight, m_tileHeight, m_ilwTileHeight );

        return widthInTiles * heightInTiles;
    }

    // Get the number of tiles in the specified range of miplevels.
    unsigned int getNumTilesInMipLevels( unsigned int firstLevel, unsigned int lastLevel ) const
    {
        unsigned int numTiles = 0;
        for( unsigned int level = firstLevel; level <= lastLevel; ++level )
        {
            numTiles += numTilesInLevel( level );
        }
        return numTiles;
    }

    unsigned int callwlateTileIndexFromTileCoords( unsigned int mipLevel,
                                                   unsigned int tileX,
                                                   unsigned int tileY,
                                                   unsigned int levelWidth,
                                                   unsigned int mipTailFirstLevel ) const
    {
        // If the requested miplevel is in the mip tail, return a request for tile index zero.
        if( mipLevel >= mipTailFirstLevel )
            return 0;

        // We're rounding up to accommodate partial tiles.
        unsigned int widthInTiles     = ceilMult( levelWidth, m_tileWidth, m_ilwTileWidth );
        unsigned int indexWithinLevel = tileY * widthInTiles + tileX;

        // We also include a single tile for the mip tail.
        return indexWithinLevel + getNumTilesInMipLevels( mipLevel + 1, mipTailFirstLevel - 1 ) + 1;
    }

    // Use the reciprocal of tile size to avoid callwlating it again and again on multiple calls
    static unsigned int callwlateWrappedTileCoord(::lwca::lwdaTextureAddressMode wrapMode, int coord, unsigned int levelSize, float ilwTileSize )
    {
        return static_cast<unsigned int>( static_cast<float>( wrapPixelCoord( wrapMode, coord, levelSize ) ) * ilwTileSize );
    }
};

}  // namespace optix
