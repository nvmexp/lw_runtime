// Copyright (c) 2019, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS


#include <Memory/DemandLoad/TileIndexing.h>

#include <gtest/gtest.h>

using namespace optix;

static const unsigned int NUM_TILES_FOR_MIP_TAIL = 1;

namespace {
// This function approximates the values returned by the LWCA sparse textures API. It should only be called in test code.
unsigned int getLwdaMipTailFirstLevel( unsigned int textureWidth, unsigned int textureHeight, unsigned int tileWidth, unsigned int tileHeight )
{
    TileIndexing indexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int mipLevel = 0;
    for( ; true; ++mipLevel )
    {
        unsigned int levelWidth;
        unsigned int levelHeight;
        indexing.callwlateLevelDims( mipLevel, levelWidth, levelHeight );
        if( levelWidth < tileWidth || levelHeight < tileHeight )
            return mipLevel;
    }
    return mipLevel;
}
}

/*
 * callwlateLevelDims
 */

TEST( TestTileIndexing, callwlateLevelDims_Pow2_Square )
{
    unsigned int levelWidth;
    unsigned int levelHeight;
    TileIndexing( 512, 512, 32, 32 ).callwlateLevelDims( 1, levelWidth, levelHeight );
    EXPECT_EQ( 256, levelWidth );
    EXPECT_EQ( 256, levelHeight );
}

TEST( TestTileIndexing, callwlateLevelDims_NonPow2_NonSquare )
{
    unsigned int levelWidth;
    unsigned int levelHeight;
    TileIndexing( 511, 513, 32, 32 ).callwlateLevelDims( 1, levelWidth, levelHeight );
    EXPECT_EQ( 255, levelWidth );
    EXPECT_EQ( 256, levelHeight );
}

TEST( TestTileIndexing, callwlateLevelDims_CoarsestLevel_Square )
{
    unsigned int levelWidth;
    unsigned int levelHeight;
    TileIndexing( 16, 16, 32, 32 ).callwlateLevelDims( 4, levelWidth, levelHeight );
    EXPECT_EQ( 1, levelWidth );
    EXPECT_EQ( 1, levelHeight );
}

TEST( TestTileIndexing, callwlateLevelDims_CoarsestLevel_NonSquare )
{
    unsigned int levelWidth;
    unsigned int levelHeight;
    TileIndexing( 16, 32, 32, 32 ).callwlateLevelDims( 5, levelWidth, levelHeight );
    EXPECT_EQ( 1, levelWidth );
    EXPECT_EQ( 1, levelHeight );
}

TEST( TestTileIndexing, callwlateLevelDims_SecondCoarsestLevel_NonSquare )
{
    unsigned int levelWidth;
    unsigned int levelHeight;
    TileIndexing( 16, 32, 32, 32 ).callwlateLevelDims( 4, levelWidth, levelHeight );
    EXPECT_EQ( 1, levelWidth );
    EXPECT_EQ( 2, levelHeight );
}

TEST( TestTileIndexing, getLwdaMipTailFirstLevel )
{
    EXPECT_EQ( 5, getLwdaMipTailFirstLevel( 512, 512, 32, 32 ) );
    EXPECT_EQ( 4, getLwdaMipTailFirstLevel( 512, 512, 64, 32 ) );

    EXPECT_EQ( 4, getLwdaMipTailFirstLevel( 511, 511, 32, 32 ) );
    EXPECT_EQ( 5, getLwdaMipTailFirstLevel( 513, 513, 32, 32 ) );

    EXPECT_EQ( 3, getLwdaMipTailFirstLevel( 511, 511, 32, 64 ) );
    EXPECT_EQ( 4, getLwdaMipTailFirstLevel( 513, 513, 32, 64 ) );

    EXPECT_EQ( 4, getLwdaMipTailFirstLevel( 511, 257, 32, 32 ) );
    EXPECT_EQ( 3, getLwdaMipTailFirstLevel( 513, 255, 32, 32 ) );

    EXPECT_EQ( 3, getLwdaMipTailFirstLevel( 511, 257, 64, 32 ) );
    EXPECT_EQ( 3, getLwdaMipTailFirstLevel( 513, 255, 64, 32 ) );

    EXPECT_EQ( 2, getLwdaMipTailFirstLevel( 200, 140, 64, 64 ) );
}

/*
 * callwlateNumPages
 */

TEST( TestTileIndexing, callwlateNumPages_AllMipTail )
{
    unsigned int mipTailFirstLevel = 0;
    EXPECT_EQ( 1, TileIndexing( 16, 16, 32, 32 ).callwlateNumPages( mipTailFirstLevel ) );
}

TEST( TestTileIndexing, callwlateNumPages_Pow2Tex_SquareTex_SquareTile )
{
    unsigned int totalNumTiles = 16 * 16 + 8 * 8 + 4 * 4 + 2 * 2 + 1 + NUM_TILES_FOR_MIP_TAIL;
    TileIndexing tileIndexing( 512, 512, 32, 32 );
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( 512, 512, 32, 32 );
    EXPECT_EQ( totalNumTiles, tileIndexing.callwlateNumPages( mipTailFirstLevel ) );
}

TEST( TestTileIndexing, callwlateNumPages_Pow2Tex_SquareTex_NonSquareTile )
{
    unsigned int totalNumTiles = 8 * 16 + 4 * 8 + 2 * 4 + 1 * 2 + NUM_TILES_FOR_MIP_TAIL;
    TileIndexing tileIndexing( 512, 512, 64, 32 );
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( 512, 512, 64, 32 );
    EXPECT_EQ( totalNumTiles, tileIndexing.callwlateNumPages( mipTailFirstLevel ) );
}

TEST( TestTileIndexing, callwlateNumPages_Pow2Tex_NonSquareTex_SquareTile )
{
    unsigned int totalNumTiles = 16 * 8 + 8 * 4 + 4 * 2 + 2 * 1 + NUM_TILES_FOR_MIP_TAIL;
    TileIndexing tileIndexing( 512, 256, 32, 32 );
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( 512, 256, 32, 32 );
    EXPECT_EQ( totalNumTiles, tileIndexing.callwlateNumPages( mipTailFirstLevel ) );
}

TEST( TestTileIndexing, callwlateNumPages_Pow2Tex_NonSquareTex_NonSquareTile )
{
    unsigned int totalNumTiles = 8 * 8 + 4 * 4 + 2 * 2 + 1 * 1 + NUM_TILES_FOR_MIP_TAIL;
    TileIndexing tileIndexing( 512, 256, 64, 32 );
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( 512, 256, 64, 32 );
    EXPECT_EQ( totalNumTiles, tileIndexing.callwlateNumPages( mipTailFirstLevel ) );
}

TEST( TestTileIndexing, callwlateNumPages_UndersizeNonPow2Tex_SquareTex_SquareTile )
{
    unsigned int totalNumTiles = 16 * 16 + 8 * 8 + 4 * 4 + 2 * 2 + NUM_TILES_FOR_MIP_TAIL;
    TileIndexing tileIndexing( 511, 511, 32, 32 );
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( 511, 511, 32, 32 );
    EXPECT_EQ( totalNumTiles, tileIndexing.callwlateNumPages( mipTailFirstLevel ) );
}

TEST( TestTileIndexing, callwlateNumPages_OversizeNonPow2Tex_SquareTex_SquareTile )
{
    unsigned int totalNumTiles = 17 * 17 + 8 * 8 + 4 * 4 + 2 * 2 + 1 * 1 + NUM_TILES_FOR_MIP_TAIL;
    TileIndexing tileIndexing( 513, 513, 32, 32 );
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( 513, 513, 32, 32 );
    EXPECT_EQ( totalNumTiles, tileIndexing.callwlateNumPages( mipTailFirstLevel ) );
}

TEST( TestTileIndexing, callwlateNumPages_UndersizeNonPow2Tex_SquareTex_NonSquareTile )
{
    unsigned int totalNumTiles = 16 * 8 + 8 * 4 + 4 * 2 + NUM_TILES_FOR_MIP_TAIL;
    TileIndexing tileIndexing( 511, 511, 32, 64 );
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( 511, 511, 32, 64 );
    EXPECT_EQ( totalNumTiles, tileIndexing.callwlateNumPages( mipTailFirstLevel ) );
}

TEST( TestTileIndexing, callwlateNumPages_OversizeNonPow2Tex_SquareTex_NonSquareTile )
{
    unsigned int totalNumTiles = 17 * 9 + 8 * 4 + 4 * 2 + 2 * 1 + NUM_TILES_FOR_MIP_TAIL;
    TileIndexing tileIndexing( 513, 513, 32, 64 );
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( 513, 513, 32, 64 );
    EXPECT_EQ( totalNumTiles, tileIndexing.callwlateNumPages( mipTailFirstLevel ) );
}

TEST( TestTileIndexing, callwlateNumPages_MixedSizeNonPow2Tex_NonSquareTex_SquareTile )
{
    unsigned int totalNumTiles = 16 * 9 + 8 * 4 + 4 * 2 + 2 * 1 + NUM_TILES_FOR_MIP_TAIL;
    TileIndexing tileIndexing( 511, 257, 32, 32 );
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( 511, 257, 32, 32 );
    EXPECT_EQ( totalNumTiles, tileIndexing.callwlateNumPages( mipTailFirstLevel ) );
}

TEST( TestTileIndexing, callwlateNumPages_MixedSize2NonPow2Tex_NonSquareTex_SquareTile )
{
    unsigned int totalNumTiles = 17 * 8 + 8 * 4 + 4 * 2 + NUM_TILES_FOR_MIP_TAIL;
    TileIndexing tileIndexing( 513, 255, 32, 32 );
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( 513, 255, 32, 32 );
    EXPECT_EQ( totalNumTiles, tileIndexing.callwlateNumPages( mipTailFirstLevel ) );
}


TEST( TestTileIndexing, callwlateNumPages_MixedSizeNonPow2Tex_NonSquareTex_NonSquareTile )
{
    unsigned int totalNumTiles = 8 * 9 + 4 * 4 + 2 * 2 + NUM_TILES_FOR_MIP_TAIL;
    TileIndexing tileIndexing( 511, 257, 64, 32 );
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( 511, 257, 64, 32 );
    EXPECT_EQ( totalNumTiles, tileIndexing.callwlateNumPages( mipTailFirstLevel ) );
}

TEST( TestTileIndexing, callwlateNumPages_MixedSize2NonPow2Tex_NonSquareTex_NonSquareTile )
{
    unsigned int totalNumTiles = 9 * 8 + 4 * 4 + 2 * 2 + NUM_TILES_FOR_MIP_TAIL;
    TileIndexing tileIndexing( 513, 255, 64, 32 );
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( 513, 255, 64, 32 );
    EXPECT_EQ( totalNumTiles, tileIndexing.callwlateNumPages( mipTailFirstLevel ) );
}

/*
 * callwlateTileIndex
 */

TEST( TestTileIndexing, callwlateTileIndex )
{
    unsigned int mipLevel      = 1;
    unsigned int textureWidth  = 512;
    unsigned int textureHeight = 512;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int levelWidth;
    unsigned int levelHeight;
    tileIndexing.callwlateLevelDims( mipLevel, levelWidth, levelHeight );

    int                          pixelX    = 33;
    int                          pixelY    = 33;
    lwca::lwdaTextureAddressMode wrapModeX = lwca::lwdaAddressModeBorder;
    lwca::lwdaTextureAddressMode wrapModeY = lwca::lwdaAddressModeBorder;
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    EXPECT_EQ( 31, tileIndexing.callwlateTileIndex( mipLevel, pixelX, pixelY, wrapModeX, wrapModeY, levelWidth,
                                                    levelHeight, mipTailFirstLevel ) );
}

TEST( TestTileIndexing, callwlateTileIndexAdjacentToMipTail )
{
    unsigned int mipLevel      = 4;
    unsigned int textureWidth  = 512;
    unsigned int textureHeight = 512;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;

    unsigned int levelWidth;
    unsigned int levelHeight;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );
    tileIndexing.callwlateLevelDims( mipLevel, levelWidth, levelHeight );

    int                          pixelX    = 15;
    int                          pixelY    = 15;
    lwca::lwdaTextureAddressMode wrapModeX = lwca::lwdaAddressModeBorder;
    lwca::lwdaTextureAddressMode wrapModeY = lwca::lwdaAddressModeBorder;
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    EXPECT_EQ( 1, tileIndexing.callwlateTileIndex( mipLevel, pixelX, pixelY, wrapModeX, wrapModeY, levelWidth,
                                                   levelHeight, mipTailFirstLevel ) );
}

TEST( TestTileIndexing, callwlateTileIndexAtMipTail )
{
    unsigned int mipLevel      = 5;
    unsigned int textureWidth  = 512;
    unsigned int textureHeight = 512;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;

    unsigned int levelWidth;
    unsigned int levelHeight;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );
    tileIndexing.callwlateLevelDims( mipLevel, levelWidth, levelHeight );

    int                          pixelX    = 15;
    int                          pixelY    = 15;
    lwca::lwdaTextureAddressMode wrapModeX = lwca::lwdaAddressModeBorder;
    lwca::lwdaTextureAddressMode wrapModeY = lwca::lwdaAddressModeBorder;
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    EXPECT_EQ( 0, tileIndexing.callwlateTileIndex( mipLevel, pixelX, pixelY, wrapModeX, wrapModeY, levelWidth,
                                                   levelHeight, mipTailFirstLevel ) );
}

TEST( TestTileIndexing, callwlateTileIndexInMipTail )
{
    unsigned int mipLevel      = 7;
    unsigned int textureWidth  = 512;
    unsigned int textureHeight = 512;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;

    unsigned int levelWidth;
    unsigned int levelHeight;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );
    tileIndexing.callwlateLevelDims( mipLevel, levelWidth, levelHeight );

    int                          pixelX    = 3;
    int                          pixelY    = 3;
    lwca::lwdaTextureAddressMode wrapModeX = lwca::lwdaAddressModeBorder;
    lwca::lwdaTextureAddressMode wrapModeY = lwca::lwdaAddressModeBorder;
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    EXPECT_EQ( 0, tileIndexing.callwlateTileIndex( mipLevel, pixelX, pixelY, wrapModeX, wrapModeY, levelWidth,
                                                   levelHeight, mipTailFirstLevel ) );
}

/*
 * unpackTileIndex
 */

TEST( TestTileIndexing, unpackTileLevel1 )
{
    unsigned int mipLevel      = 1;
    unsigned int textureWidth  = 512;
    unsigned int textureHeight = 512;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;

    unsigned int levelWidth;
    unsigned int levelHeight;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );
    tileIndexing.callwlateLevelDims( mipLevel, levelWidth, levelHeight );

    int                          pixelX    = 33;
    int                          pixelY    = 33;
    lwca::lwdaTextureAddressMode wrapModeX = lwca::lwdaAddressModeClamp;
    lwca::lwdaTextureAddressMode wrapModeY = lwca::lwdaAddressModeClamp;
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int tileIndex = tileIndexing.callwlateTileIndex( mipLevel, pixelX, pixelY, wrapModeX, wrapModeY,
                                                              levelWidth, levelHeight, mipTailFirstLevel );

    unsigned int outMipLevel;
    unsigned int outX;
    unsigned int outY;
    tileIndexing.unpackTileIndex( tileIndex, mipTailFirstLevel, outMipLevel, outX, outY );

    EXPECT_EQ( mipLevel, outMipLevel );

    int nearestTileX = pixelX - ( pixelX % tileWidth );
    int nearestTileY = pixelY - ( pixelY % tileHeight );
    EXPECT_EQ( nearestTileX, outX );
    EXPECT_EQ( nearestTileY, outY );
}

TEST( TestTileIndexing, unpackTileLevelOnMipTailBoundary )
{
    unsigned int mipLevel      = 0;
    unsigned int textureWidth  = 200;
    unsigned int textureHeight = 140;
    unsigned int tileWidth     = 64;
    unsigned int tileHeight    = 64;

    unsigned int levelWidth;
    unsigned int levelHeight;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );
    tileIndexing.callwlateLevelDims( mipLevel, levelWidth, levelHeight );

    int                          pixelX    = 33;
    int                          pixelY    = 33;
    lwca::lwdaTextureAddressMode wrapModeX = lwca::lwdaAddressModeClamp;
    lwca::lwdaTextureAddressMode wrapModeY = lwca::lwdaAddressModeClamp;
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int tileIndex = tileIndexing.callwlateTileIndex( mipLevel, pixelX, pixelY, wrapModeX, wrapModeY,
                                                              levelWidth, levelHeight, mipTailFirstLevel );

    unsigned int outMipLevel;
    unsigned int outX;
    unsigned int outY;
    tileIndexing.unpackTileIndex( tileIndex, mipTailFirstLevel, outMipLevel, outX, outY );

    EXPECT_EQ( mipLevel, outMipLevel );
    int nearestTileX = pixelX - ( pixelX % tileWidth );
    int nearestTileY = pixelY - ( pixelY % tileHeight );
    EXPECT_EQ( nearestTileX, outX );
    EXPECT_EQ( nearestTileY, outY );
}

/*
 * callwlateTileRequests
 */

TEST( TestTileIndexing, callwlateTileRequests_NoAnisotropy )
{
    unsigned int tilesToRequest[TileIndexing::MAX_TILES_CALLWLATED];
    unsigned int numTilesToRequest;

    unsigned int mipLevel      = 1;
    float        normX         = .2f;
    float        normY         = .2f;
    unsigned int textureWidth  = 512;
    unsigned int textureHeight = 512;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int                 anisotropy = 0;
    lwca::lwdaTextureAddressMode wrapModeX  = lwca::lwdaAddressModeClamp;
    lwca::lwdaTextureAddressMode wrapModeY  = lwca::lwdaAddressModeClamp;

    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    tileIndexing.callwlateTileRequests( mipLevel, normX, normY, mipTailFirstLevel, anisotropy, wrapModeX, wrapModeY,
                                        tilesToRequest, numTilesToRequest );

    EXPECT_EQ( 1, numTilesToRequest );
    EXPECT_EQ( 31, tilesToRequest[0] );
}

TEST( TestTileIndexing, callwlateTileRequests_WithAnisotropy_OnLeftTileBorder )
{
    unsigned int tilesToRequest[TileIndexing::MAX_TILES_CALLWLATED];
    unsigned int numTilesToRequest;

    unsigned int mipLevel      = 1;
    float        normX         = .125f;                // left border of tile
    float        normY         = .125f + .125f / 2.f;  // mid tile
    unsigned int textureWidth  = 512;
    unsigned int textureHeight = 512;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int                 anisotropy = 16;
    lwca::lwdaTextureAddressMode wrapModeX  = lwca::lwdaAddressModeClamp;
    lwca::lwdaTextureAddressMode wrapModeY  = lwca::lwdaAddressModeClamp;

    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    tileIndexing.callwlateTileRequests( mipLevel, normX, normY, mipTailFirstLevel, anisotropy, wrapModeX, wrapModeY,
                                        tilesToRequest, numTilesToRequest );

    EXPECT_EQ( 2, numTilesToRequest );
    EXPECT_EQ( 30, tilesToRequest[0] );
    EXPECT_EQ( 31, tilesToRequest[1] );
}

TEST( TestTileIndexing, callwlateTileRequests_WithAnisotropy_OnTwoBorders )
{
    unsigned int tilesToRequest[TileIndexing::MAX_TILES_CALLWLATED];
    unsigned int numTilesToRequest;

    unsigned int mipLevel      = 1;
    float        normX         = .125f;  // left border of tile
    float        normY         = .125f;  // top border of tile
    unsigned int textureWidth  = 512;
    unsigned int textureHeight = 512;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int                 anisotropy = 16;
    lwca::lwdaTextureAddressMode wrapModeX  = lwca::lwdaAddressModeClamp;
    lwca::lwdaTextureAddressMode wrapModeY  = lwca::lwdaAddressModeClamp;

    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    tileIndexing.callwlateTileRequests( mipLevel, normX, normY, mipTailFirstLevel, anisotropy, wrapModeX, wrapModeY,
                                        tilesToRequest, numTilesToRequest );

    EXPECT_EQ( 4, numTilesToRequest );
    EXPECT_EQ( 22, tilesToRequest[0] );
    EXPECT_EQ( 22 + 1, tilesToRequest[1] );
    EXPECT_EQ( 22 + 8, tilesToRequest[2] );
    EXPECT_EQ( 22 + 8 + 1, tilesToRequest[3] );
}

TEST( TestTileIndexing, callwlateTileRequests_WithAnisotropy_MidTile )
{
    unsigned int tilesToRequest[TileIndexing::MAX_TILES_CALLWLATED];
    unsigned int numTilesToRequest;

    unsigned int mipLevel      = 1;
    float        normX         = .125f + .125f / 2.f;  // mid tile
    float        normY         = .125f + .125f / 2.f;  // mid tile
    unsigned int textureWidth  = 512;
    unsigned int textureHeight = 512;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int                 anisotropy = 16;
    lwca::lwdaTextureAddressMode wrapModeX  = lwca::lwdaAddressModeClamp;
    lwca::lwdaTextureAddressMode wrapModeY  = lwca::lwdaAddressModeClamp;

    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    tileIndexing.callwlateTileRequests( mipLevel, normX, normY, mipTailFirstLevel, anisotropy, wrapModeX, wrapModeY,
                                        tilesToRequest, numTilesToRequest );

    EXPECT_EQ( 1, numTilesToRequest );
    EXPECT_EQ( 31, tilesToRequest[0] );
}

TEST( TestTileIndexing, callwlateTileRequests_optixTextureModes_wrapMode1 )
{
    unsigned int tilesToRequest[TileIndexing::MAX_TILES_CALLWLATED];
    unsigned int numTilesToRequest;

    unsigned int mipLevel      = 0;
    float        normX         = 1.064453f;
    float        normY         = 1.064453f;
    unsigned int textureWidth  = 1024;
    unsigned int textureHeight = 1024;
    unsigned int tileWidth     = 64;
    unsigned int tileHeight    = 64;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int                 anisotropy = 16;
    lwca::lwdaTextureAddressMode wrapModeX  = lwca::lwdaAddressModeClamp;
    lwca::lwdaTextureAddressMode wrapModeY  = lwca::lwdaAddressModeClamp;

    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    tileIndexing.callwlateTileRequests( mipLevel, normX, normY, mipTailFirstLevel, anisotropy, wrapModeX, wrapModeY,
                                        tilesToRequest, numTilesToRequest );

    EXPECT_EQ( 1, numTilesToRequest );
    EXPECT_EQ( 16 * 16 - 1 + 8 * 8 + 4 * 4 + 2 * 2 + 1 + 1, tilesToRequest[0] );
}


/*
 * callwlateTileRequestsFromTexels
 */

TEST( TestTileIndexing, callwlateTileRequestsFromTexels )
{
    unsigned int tilesToRequest[TileIndexing::MAX_TILES_CALLWLATED];
    unsigned int numTilesToRequest;

    unsigned int textureWidth  = 1024;
    unsigned int textureHeight = 1024;
    unsigned int tileWidth     = 64;
    unsigned int tileHeight    = 64;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int       mipLevel           = 2;
    const unsigned int numTexels          = 1;
    unsigned int       xCoords[numTexels] = {50};
    unsigned int       yCoords[numTexels] = {50};
    const unsigned int texelWidth         = 2;
    const unsigned int texelHeight        = 2;
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );
    lwca::lwdaTextureAddressMode wrapModeX = lwca::lwdaAddressModeClamp;
    lwca::lwdaTextureAddressMode wrapModeY = lwca::lwdaAddressModeClamp;

    tileIndexing.callwlateTileRequestsFromTexels( mipLevel, xCoords, yCoords, numTexels, texelWidth, texelHeight,
                                                  mipTailFirstLevel, wrapModeX, wrapModeY, tilesToRequest, numTilesToRequest );
    EXPECT_EQ( 1, numTilesToRequest );
    EXPECT_EQ( 6, tilesToRequest[0] );
}

TEST( TestTileIndexing, callwlateTileRequestsFromTexels_edgeOfFinestMipLevel )
{
    unsigned int tilesToRequest[TileIndexing::MAX_TILES_CALLWLATED];
    unsigned int numTilesToRequest;

    unsigned int textureWidth  = 1024;
    unsigned int textureHeight = 1024;
    unsigned int tileWidth     = 64;
    unsigned int tileHeight    = 64;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int       mipLevel           = 0;
    const unsigned int numTexels          = 1;
    unsigned int       xCoords[numTexels] = {1022};
    unsigned int       yCoords[numTexels] = {1022};
    const unsigned int texelWidth         = 2;
    const unsigned int texelHeight        = 2;
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );
    lwca::lwdaTextureAddressMode wrapModeX = lwca::lwdaAddressModeClamp;
    lwca::lwdaTextureAddressMode wrapModeY = lwca::lwdaAddressModeClamp;

    tileIndexing.callwlateTileRequestsFromTexels( mipLevel, xCoords, yCoords, numTexels, texelWidth, texelHeight,
                                                  mipTailFirstLevel, wrapModeX, wrapModeY, tilesToRequest, numTilesToRequest );
    EXPECT_EQ( 1, numTilesToRequest );
    EXPECT_EQ( 341, tilesToRequest[0] );
}

TEST( TestTileIndexing, callwlateTileRequestsFromTexels_finestMipLevelInMipTail )
{
    unsigned int tilesToRequest[TileIndexing::MAX_TILES_CALLWLATED];
    unsigned int numTilesToRequest;

    unsigned int textureWidth  = 8;
    unsigned int textureHeight = 8;
    unsigned int tileWidth     = 128;
    unsigned int tileHeight    = 64;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int                 mipLevel           = 0;
    const unsigned int           numTexels          = 1;
    unsigned int                 xCoords[numTexels] = {6};
    unsigned int                 yCoords[numTexels] = {8};
    const unsigned int           texelWidth         = 2;
    const unsigned int           texelHeight        = 2;
    lwca::lwdaTextureAddressMode wrapModeX          = lwca::lwdaAddressModeClamp;
    lwca::lwdaTextureAddressMode wrapModeY          = lwca::lwdaAddressModeClamp;
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    tileIndexing.callwlateTileRequestsFromTexels( mipLevel, xCoords, yCoords, numTexels, texelWidth, texelHeight,
                                                  mipTailFirstLevel, wrapModeX, wrapModeY, tilesToRequest, numTilesToRequest );
    EXPECT_EQ( 1, numTilesToRequest );
    EXPECT_EQ( 0, tilesToRequest[0] );
}

TEST( TestTileIndexing, callwlateTileRequestsFromTexels_nonSquareTextureTile )
{
    unsigned int tilesToRequest[TileIndexing::MAX_TILES_CALLWLATED];
    unsigned int numTilesToRequest;

    unsigned int textureWidth  = 4096;
    unsigned int textureHeight = 4096;
    unsigned int tileWidth     = 128;
    unsigned int tileHeight    = 64;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int                 mipLevel           = 2;
    const unsigned int           numTexels          = 4;
    unsigned int                 xCoords[numTexels] = {802, 804, 802, 804};
    unsigned int                 yCoords[numTexels] = {964, 964, 966, 966};
    const unsigned int           texelWidth         = 2;
    const unsigned int           texelHeight        = 2;
    lwca::lwdaTextureAddressMode wrapModeX          = lwca::lwdaAddressModeClamp;
    lwca::lwdaTextureAddressMode wrapModeY          = lwca::lwdaAddressModeClamp;
    unsigned int mipTailFirstLevel = getLwdaMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    tileIndexing.callwlateTileRequestsFromTexels( mipLevel, xCoords, yCoords, numTexels, texelWidth, texelHeight,
                                                  mipTailFirstLevel, wrapModeX, wrapModeY, tilesToRequest, numTilesToRequest );
    EXPECT_EQ( 1, numTilesToRequest );
    EXPECT_EQ( 169, tilesToRequest[0] );
}

TEST( TestTileIndexing, callwlateTileRequestsFromTexels_requiresWrapping )
{
    unsigned int tilesToRequest[TileIndexing::MAX_TILES_CALLWLATED];
    unsigned int numTilesToRequest;

    unsigned int textureWidth  = 1024;
    unsigned int textureHeight = 1024;
    unsigned int tileWidth     = 64;
    unsigned int tileHeight    = 64;
    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int                 mipLevel           = 0;
    const unsigned int           numTexels          = 1;
    unsigned int                 xCoords[numTexels] = {1023};
    unsigned int                 yCoords[numTexels] = {1023};
    const unsigned int           texelWidth         = 2;
    const unsigned int           texelHeight        = 2;
    lwca::lwdaTextureAddressMode wrapModeX          = lwca::lwdaAddressModeWrap;
    lwca::lwdaTextureAddressMode wrapModeY          = lwca::lwdaAddressModeWrap;
    unsigned int                 mipTailFirstLevel  = 1;

    tileIndexing.callwlateTileRequestsFromTexels( mipLevel, xCoords, yCoords, numTexels, texelWidth, texelHeight,
                                                  mipTailFirstLevel, wrapModeX, wrapModeY, tilesToRequest, numTilesToRequest );
    EXPECT_EQ( 4, numTilesToRequest );
    EXPECT_EQ( 256, tilesToRequest[0] );
    EXPECT_EQ( 16, tilesToRequest[1] );
    EXPECT_EQ( 241, tilesToRequest[2] );
    EXPECT_EQ( 1, tilesToRequest[3] );
}

/*
 * wrapNormCoord
 */

TEST( TestTileIndexing, wrapNormCoord_addressModeWrap )
{
    const float EPS = 0.000001f;
    EXPECT_NEAR( 0.1f, TileIndexing::wrapNormCoord( 0.1f, lwca::lwdaAddressModeWrap ), EPS );
    EXPECT_NEAR( 0.5f, TileIndexing::wrapNormCoord( 1.5f, lwca::lwdaAddressModeWrap ), EPS );
    EXPECT_NEAR( 0.2f, TileIndexing::wrapNormCoord( -0.8f, lwca::lwdaAddressModeWrap ), EPS );
}

TEST( TestTileIndexing, wrapNormCoord_addressModeClamp )
{
    const float EPS = 0.000001f;
    EXPECT_NEAR( 0.1f, TileIndexing::wrapNormCoord( 0.1f, lwca::lwdaAddressModeClamp ), EPS );
    EXPECT_NEAR( 0.0f, TileIndexing::wrapNormCoord( -1.5f, lwca::lwdaAddressModeClamp ), EPS );
    EXPECT_NEAR( 1.0f, TileIndexing::wrapNormCoord( 10.2f, lwca::lwdaAddressModeClamp ), EPS );
    EXPECT_TRUE( TileIndexing::wrapNormCoord( 10.2f, lwca::lwdaAddressModeClamp ) < 1.0f );
}

TEST( TestTileIndexing, wrapNormCoord_addressModeBorder )
{
    const float EPS = 0.000001f;

    EXPECT_NEAR( 0.0f, TileIndexing::wrapNormCoord( -1.5f, lwca::lwdaAddressModeBorder ), EPS );
}

TEST( TestTileIndexing, wrapNormCoord_addressModeMirror )
{
    const float EPS = 0.000001f;

    EXPECT_NEAR( 0.25f, TileIndexing::wrapNormCoord( 1.75f, lwca::lwdaAddressModeMirror ), EPS );
}

/*
 * wrapPixelCoord
 */

TEST( TestTileIndexing, ClampOrWrapCoordinateClampInsideRegion )
{
    int                          coordinate = 5;
    int                          max        = 10;
    lwca::lwdaTextureAddressMode wrapMode   = lwca::lwdaAddressModeClamp;

    unsigned int result = TileIndexing::wrapPixelCoord( wrapMode, coordinate, max );

    ASSERT_EQ( 5, result );
}

TEST( TestTileIndexing, ClampOrWrapCoordinateClampOutsideRegionNegative )
{
    int                          coordinate = -5;
    int                          max        = 10;
    lwca::lwdaTextureAddressMode wrapMode   = lwca::lwdaAddressModeClamp;

    unsigned int result = TileIndexing::wrapPixelCoord( wrapMode, coordinate, max );

    ASSERT_EQ( 0, result );
}

TEST( TestTileIndexing, ClampOrWrapCoordinateClampOutsideRegionPositive )
{
    int                          coordinate = 15;
    int                          max        = 10;
    lwca::lwdaTextureAddressMode wrapMode   = lwca::lwdaAddressModeClamp;

    unsigned int result = TileIndexing::wrapPixelCoord( wrapMode, coordinate, max );

    ASSERT_EQ( 9, result );
}

TEST( TestTileIndexing, ClampOrWrapCoordinateClampOnPositiveBorder )
{
    int                          coordinate = 10;
    int                          max        = 10;
    lwca::lwdaTextureAddressMode wrapMode   = lwca::lwdaAddressModeClamp;

    unsigned int result = TileIndexing::wrapPixelCoord( wrapMode, coordinate, max );

    ASSERT_EQ( 9, result );
}

TEST( TestTileIndexing, ClampOrWrapCoordinateClampOnNegativeBorder )
{
    int                          coordinate = 0;
    int                          max        = 10;
    lwca::lwdaTextureAddressMode wrapMode   = lwca::lwdaAddressModeClamp;

    unsigned int result = TileIndexing::wrapPixelCoord( wrapMode, coordinate, max );

    ASSERT_EQ( 0, result );
}

TEST( TestTileIndexing, ClampOrWrapCoordinateWrapRepeatOutsideRegionPositive )
{
    int                          coordinate = 15;
    int                          max        = 10;
    lwca::lwdaTextureAddressMode wrapMode   = lwca::lwdaAddressModeWrap;

    unsigned int result = TileIndexing::wrapPixelCoord( wrapMode, coordinate, max );

    ASSERT_EQ( 5, result );
}

TEST( TestTileIndexing, ClampOrWrapCoordinateWrapRepeatOutsideRegionPositiveSeveralMultiples )
{
    int                          coordinate = 35;
    int                          max        = 10;
    lwca::lwdaTextureAddressMode wrapMode   = lwca::lwdaAddressModeWrap;

    unsigned int result = TileIndexing::wrapPixelCoord( wrapMode, coordinate, max );

    ASSERT_EQ( 5, result );
}

TEST( TestTileIndexing, ClampOrWrapCoordinateWrapRepeatOutsideRegionNegative )
{
    int                          coordinate = -7;
    int                          max        = 10;
    lwca::lwdaTextureAddressMode wrapMode   = lwca::lwdaAddressModeWrap;

    unsigned int result = TileIndexing::wrapPixelCoord( wrapMode, coordinate, max );

    ASSERT_EQ( 3, result );
}

TEST( TestTileIndexing, ClampOrWrapCoordinateWrapRepeatOutsideRegionNegativeSeveralMultiples )
{
    int                          coordinate = -24;
    int                          max        = 10;
    lwca::lwdaTextureAddressMode wrapMode   = lwca::lwdaAddressModeWrap;

    unsigned int result = TileIndexing::wrapPixelCoord( wrapMode, coordinate, max );

    ASSERT_EQ( 6, result );
}

TEST( TestTileIndexing, ClampOrWrapCoordinateMirrorInsideRegion )
{
    int                          coordinate = 5;
    int                          max        = 10;
    lwca::lwdaTextureAddressMode wrapMode   = lwca::lwdaAddressModeMirror;

    unsigned int result = TileIndexing::wrapPixelCoord( wrapMode, coordinate, max );

    ASSERT_EQ( 5, result );
}

TEST( TestTileIndexing, ClampOrWrapCoordinateMirrorOutsideRegionPositive )
{
    int                          coordinate = 13;
    int                          max        = 10;
    lwca::lwdaTextureAddressMode wrapMode   = lwca::lwdaAddressModeMirror;

    unsigned int result = TileIndexing::wrapPixelCoord( wrapMode, coordinate, max );

    ASSERT_EQ( 6, result );
}

TEST( TestTileIndexing, ClampOrWrapCoordinateMirrorOutsideRegionPositiveSeveralMultiples )
{
    int                          coordinate = 24;
    int                          max        = 10;
    lwca::lwdaTextureAddressMode wrapMode   = lwca::lwdaAddressModeMirror;

    unsigned int result = TileIndexing::wrapPixelCoord( wrapMode, coordinate, max );

    ASSERT_EQ( 4, result );
}

TEST( TestTileIndexing, ClampOrWrapCoordinateMirrorOutsideRegionNegative )
{
    int                          coordinate = -2;
    int                          max        = 10;
    lwca::lwdaTextureAddressMode wrapMode   = lwca::lwdaAddressModeMirror;

    unsigned int result = TileIndexing::wrapPixelCoord( wrapMode, coordinate, max );

    ASSERT_EQ( 1, result );
}

TEST( TestTileIndexing, ClampOrWrapCoordinateMirrorOutsideRegionNegativeSeveralMultiples )
{
    int                          coordinate = -22;
    int                          max        = 10;
    lwca::lwdaTextureAddressMode wrapMode   = lwca::lwdaAddressModeMirror;

    unsigned int result = TileIndexing::wrapPixelCoord( wrapMode, coordinate, max );

    ASSERT_EQ( 1, result );
}

/*
 * textureToSoftwareTileCoord
 */

TEST( TestTileIndexing, textureToSoftwareTileCoord_MiddleOfTileNoGutter )
{
    unsigned int textureWidth  = 128;
    unsigned int textureHeight = 128;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;

    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int                 tileGutterWidth = 0;
    unsigned int                 levelWidth      = 128;
    unsigned int                 levelHeight     = 128;
    float                        normX           = 0.3f;
    float                        normY           = 0.6f;
    lwca::lwdaTextureAddressMode wrapModeX       = lwca::lwdaAddressModeBorder;
    lwca::lwdaTextureAddressMode wrapModeY       = lwca::lwdaAddressModeBorder;

    float outNormX;
    float outNormY;
    tileIndexing.textureToSoftwareTileCoord( tileGutterWidth, levelWidth, levelHeight, normX, normY, wrapModeX,
                                             wrapModeY, outNormX, outNormY );

    EXPECT_FLOAT_EQ( 0.2f, outNormX );
    EXPECT_FLOAT_EQ( 0.4f, outNormY );
}

TEST( TestTileIndexing, textureToSoftwareTileCoord_OutsideOfTilePositiveNoGutter )
{
    unsigned int textureWidth  = 128;
    unsigned int textureHeight = 128;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;

    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int                 tileGutterWidth = 0;
    unsigned int                 levelWidth      = 128;
    unsigned int                 levelHeight     = 128;
    float                        normX           = 1.2f;
    float                        normY           = 1.3f;
    lwca::lwdaTextureAddressMode wrapModeX       = lwca::lwdaAddressModeBorder;
    lwca::lwdaTextureAddressMode wrapModeY       = lwca::lwdaAddressModeBorder;

    float outNormX;
    float outNormY;
    tileIndexing.textureToSoftwareTileCoord( tileGutterWidth, levelWidth, levelHeight, normX, normY, wrapModeX,
                                             wrapModeY, outNormX, outNormY );

    EXPECT_FLOAT_EQ( 1.8f, outNormX );
    EXPECT_FLOAT_EQ( 2.2f, outNormY );
}

TEST( TestTileIndexing, textureToSoftwareTileCoord_OutsideOfTileNegativeNoGutter )
{
    unsigned int textureWidth  = 128;
    unsigned int textureHeight = 128;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;

    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int                 tileGutterWidth = 0;
    unsigned int                 levelWidth      = 128;
    unsigned int                 levelHeight     = 128;
    float                        normX           = -0.2f;
    float                        normY           = -0.3f;
    lwca::lwdaTextureAddressMode wrapModeX       = lwca::lwdaAddressModeBorder;
    lwca::lwdaTextureAddressMode wrapModeY       = lwca::lwdaAddressModeBorder;

    float outNormX;
    float outNormY;
    tileIndexing.textureToSoftwareTileCoord( tileGutterWidth, levelWidth, levelHeight, normX, normY, wrapModeX,
                                             wrapModeY, outNormX, outNormY );

    EXPECT_FLOAT_EQ( -0.8f, outNormX );
    EXPECT_FLOAT_EQ( -1.2f, outNormY );
}

TEST( TestTileIndexing, textureToSoftwareTileCoord_MiddleOfTileWithGutter )
{
    unsigned int textureWidth  = 128;
    unsigned int textureHeight = 128;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;

    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int                 tileGutterWidth = 8;
    unsigned int                 levelWidth      = 128;
    unsigned int                 levelHeight     = 128;
    float                        normX           = 0.3f;
    float                        normY           = 0.6f;
    lwca::lwdaTextureAddressMode wrapModeX       = lwca::lwdaAddressModeBorder;
    lwca::lwdaTextureAddressMode wrapModeY       = lwca::lwdaAddressModeBorder;

    float outNormX;
    float outNormY;
    tileIndexing.textureToSoftwareTileCoord( tileGutterWidth, levelWidth, levelHeight, normX, normY, wrapModeX,
                                             wrapModeY, outNormX, outNormY );

    EXPECT_FLOAT_EQ( 0.3f, outNormX );
    EXPECT_FLOAT_EQ( 0.4333334f, outNormY );
}

TEST( TestTileIndexing, textureToSoftwareTileCoord_OutsideOfTilePositiveWithGutter )
{
    unsigned int textureWidth  = 128;
    unsigned int textureHeight = 128;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;

    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int                 tileGutterWidth = 8;
    unsigned int                 levelWidth      = 128;
    unsigned int                 levelHeight     = 128;
    float                        normX           = 1.2f;
    float                        normY           = 1.3f;
    lwca::lwdaTextureAddressMode wrapModeX       = lwca::lwdaAddressModeBorder;
    lwca::lwdaTextureAddressMode wrapModeY       = lwca::lwdaAddressModeBorder;

    float outNormX;
    float outNormY;
    tileIndexing.textureToSoftwareTileCoord( tileGutterWidth, levelWidth, levelHeight, normX, normY, wrapModeX,
                                             wrapModeY, outNormX, outNormY );

    EXPECT_FLOAT_EQ( 1.3666668f, outNormX );
    EXPECT_FLOAT_EQ( 1.633333f, outNormY );
}

TEST( TestTileIndexing, textureToSoftwareTileCoord_OutsideOfTileNegativeWithGutter )
{
    unsigned int textureWidth  = 128;
    unsigned int textureHeight = 128;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;

    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int                 tileGutterWidth = 8;
    unsigned int                 levelWidth      = 128;
    unsigned int                 levelHeight     = 128;
    float                        normX           = -0.2f;
    float                        normY           = -0.3f;
    lwca::lwdaTextureAddressMode wrapModeX       = lwca::lwdaAddressModeBorder;
    lwca::lwdaTextureAddressMode wrapModeY       = lwca::lwdaAddressModeBorder;

    float outNormX;
    float outNormY;
    tileIndexing.textureToSoftwareTileCoord( tileGutterWidth, levelWidth, levelHeight, normX, normY, wrapModeX,
                                             wrapModeY, outNormX, outNormY );

    EXPECT_FLOAT_EQ( -0.3666668f, outNormX );
    EXPECT_FLOAT_EQ( -0.63333338f, outNormY );
}

TEST( TestTileIndexing, textureToSoftwareTileCoord_MiddleOfTileNoGutterNonSquare )
{
    unsigned int textureWidth  = 128;
    unsigned int textureHeight = 100;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;

    TileIndexing tileIndexing( textureWidth, textureHeight, tileWidth, tileHeight );

    unsigned int tileGutterWidth = 0;
    unsigned int                 levelWidth      = 128;
    unsigned int                 levelHeight     = 100;
    float                        normX           = 0.3f;
    float                        normY           = 0.6f;
    lwca::lwdaTextureAddressMode wrapModeX       = lwca::lwdaAddressModeBorder;
    lwca::lwdaTextureAddressMode wrapModeY       = lwca::lwdaAddressModeBorder;

    float outNormX;
    float outNormY;
    tileIndexing.textureToSoftwareTileCoord( tileGutterWidth, levelWidth, levelHeight, normX, normY, wrapModeX,
                                             wrapModeY, outNormX, outNormY );

    EXPECT_FLOAT_EQ( 0.2f, outNormX );
    EXPECT_FLOAT_EQ( 0.875f, outNormY );
}

/*
 * getSoftwareMipTailFirstLevel
 */

TEST( TestTileIndexing, getSoftwareMipTailFirstLevel_SquareTextureSquareTextureTile )
{
    unsigned int textureWidth  = 128;
    unsigned int textureHeight = 128;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;

    unsigned int mipTailFirstLevel =
        TileIndexing::getSoftwareMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    EXPECT_EQ( 2, mipTailFirstLevel );
}

TEST( TestTileIndexing, getSoftwareMipTailFirstLevel_NonSquareTextureSquareTextureTile )
{
    unsigned int textureWidth  = 256;
    unsigned int textureHeight = 128;
    unsigned int tileWidth     = 32;
    unsigned int tileHeight    = 32;

    unsigned int mipTailFirstLevel =
        TileIndexing::getSoftwareMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    EXPECT_EQ( 3, mipTailFirstLevel );
}

TEST( TestTileIndexing, getSoftwareMipTailFirstLevel_SquareTextureNonSquareTextureTile )
{
    unsigned int textureWidth  = 128;
    unsigned int textureHeight = 128;
    unsigned int tileWidth     = 64;
    unsigned int tileHeight    = 32;

    unsigned int mipTailFirstLevel =
        TileIndexing::getSoftwareMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    EXPECT_EQ( 2, mipTailFirstLevel );
}

TEST( TestTileIndexing, getSoftwareMipTailFirstLevel_NonSquareTextureNonSquareTextureTile )
{
    unsigned int textureWidth  = 256;
    unsigned int textureHeight = 128;
    unsigned int tileWidth     = 64;
    unsigned int tileHeight    = 32;

    unsigned int mipTailFirstLevel =
        TileIndexing::getSoftwareMipTailFirstLevel( textureWidth, textureHeight, tileWidth, tileHeight );

    EXPECT_EQ( 2, mipTailFirstLevel );
}
