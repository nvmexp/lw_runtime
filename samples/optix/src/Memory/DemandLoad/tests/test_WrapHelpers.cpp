// Copyright (c) 2019, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS

#include <srcTests.h>

#include <Memory/DemandLoad/RequestHandler/WrapHelpers.h>

#include <internal/optix_declarations.h>

using namespace optix;

void printExpectedAndResultRegions( const std::vector<ReadRegion>& expectedRegions, const std::vector<ReadRegion>& resultRegions )
{
    printf( "Expected regions:\n" );
    for( const ReadRegion& region : expectedRegions )
        printf(
            "imageX = %d, imageY = %d, width = %d, height = %d, destinationBufferXOffset = %d, "
            "destinationBufferYOffset = %d, transposeX = %d, transposeY = %d\n",
            region.imageX(), region.imageY(), region.width(), region.height(), region.destinationBufferXOffset(),
            region.destinationBufferYOffset(), static_cast<int>( region.transposeX() ), static_cast<int>( region.transposeY() ) );

    printf( "Result regions:\n" );
    for( const ReadRegion& region : resultRegions )
        printf(
            "imageX = %d, imageY = %d, width = %d, height = %d, destinationBufferXOffset = %d, "
            "destinationBufferYOffset = %d, transposeX = %d, transposeY = %d\n",
            region.imageX(), region.imageY(), region.width(), region.height(), region.destinationBufferXOffset(),
            region.destinationBufferYOffset(), static_cast<int>( region.transposeX() ), static_cast<int>( region.transposeY() ) );
}

TEST( TestWrapping, ClampOrWrapCoordinateClampInsideRegion )
{
    int        coordinate = 5;
    int        max        = 10;
    RTwrapmode wrapMode   = RT_WRAP_CLAMP_TO_EDGE;

    int result = clampOrWrapCoordinate( wrapMode, coordinate, max );

    ASSERT_EQ( 5, result );
}

TEST( TestWrapping, ClampOrWrapCoordinateClampOutsideRegionNegative )
{
    int        coordinate = -5;
    int        max        = 10;
    RTwrapmode wrapMode   = RT_WRAP_CLAMP_TO_EDGE;

    int result = clampOrWrapCoordinate( wrapMode, coordinate, max );

    ASSERT_EQ( 0, result );
}

TEST( TestWrapping, ClampOrWrapCoordinateClampOutsideRegionPositive )
{
    int        coordinate = 15;
    int        max        = 10;
    RTwrapmode wrapMode   = RT_WRAP_CLAMP_TO_EDGE;

    int result = clampOrWrapCoordinate( wrapMode, coordinate, max );

    ASSERT_EQ( 10, result );
}

TEST( TestWrapping, ClampOrWrapCoordinateClampOnPositiveBorder )
{
    int        coordinate = 10;
    int        max        = 10;
    RTwrapmode wrapMode   = RT_WRAP_CLAMP_TO_EDGE;

    int result = clampOrWrapCoordinate( wrapMode, coordinate, max );

    ASSERT_EQ( 10, result );
}

TEST( TestWrapping, ClampOrWrapCoordinateClampOnNegativeBorder )
{
    int        coordinate = 0;
    int        max        = 10;
    RTwrapmode wrapMode   = RT_WRAP_CLAMP_TO_EDGE;

    int result = clampOrWrapCoordinate( wrapMode, coordinate, max );

    ASSERT_EQ( 0, result );
}

TEST( TestWrapping, ClampOrWrapCoordinateWrapRepeatOutsideRegionPositive )
{
    int        coordinate = 15;
    int        max        = 10;
    RTwrapmode wrapMode   = RT_WRAP_REPEAT;

    int result = clampOrWrapCoordinate( wrapMode, coordinate, max );

    ASSERT_EQ( 5, result );
}

TEST( TestWrapping, ClampOrWrapCoordinateWrapRepeatOutsideRegionPositiveSeveralMultiples )
{
    int        coordinate = 35;
    int        max        = 10;
    RTwrapmode wrapMode   = RT_WRAP_REPEAT;

    int result = clampOrWrapCoordinate( wrapMode, coordinate, max );

    ASSERT_EQ( 5, result );
}

TEST( TestWrapping, ClampOrWrapCoordinateWrapRepeatOutsideRegionNegative )
{
    int        coordinate = -7;
    int        max        = 10;
    RTwrapmode wrapMode   = RT_WRAP_REPEAT;

    int result = clampOrWrapCoordinate( wrapMode, coordinate, max );

    ASSERT_EQ( 3, result );
}

TEST( TestWrapping, ClampOrWrapCoordinateWrapRepeatOutsideRegionNegativeSeveralMultiples )
{
    int        coordinate = -24;
    int        max        = 10;
    RTwrapmode wrapMode   = RT_WRAP_REPEAT;

    int result = clampOrWrapCoordinate( wrapMode, coordinate, max );

    ASSERT_EQ( 6, result );
}

TEST( TestWrapping, ClampOrWrapCoordinateMirrorInsideRegion )
{
    int        coordinate = 5;
    int        max        = 10;
    RTwrapmode wrapMode   = RT_WRAP_MIRROR;

    int result = clampOrWrapCoordinate( wrapMode, coordinate, max );

    ASSERT_EQ( 5, result );
}

TEST( TestWrapping, ClampOrWrapCoordinateMirrorOutsideRegionPositive )
{
    int        coordinate = 13;
    int        max        = 10;
    RTwrapmode wrapMode   = RT_WRAP_MIRROR;

    int result = clampOrWrapCoordinate( wrapMode, coordinate, max );

    ASSERT_EQ( 7, result );
}

TEST( TestWrapping, ClampOrWrapCoordinateMirrorOutsideRegionPositiveSeveralMultiples )
{
    int        coordinate = 24;
    int        max        = 10;
    RTwrapmode wrapMode   = RT_WRAP_MIRROR;

    int result = clampOrWrapCoordinate( wrapMode, coordinate, max );

    ASSERT_EQ( 6, result );
}

TEST( TestWrapping, ClampOrWrapCoordinateMirrorOutsideRegionNegative )
{
    int        coordinate = -2;
    int        max        = 10;
    RTwrapmode wrapMode   = RT_WRAP_MIRROR;

    int result = clampOrWrapCoordinate( wrapMode, coordinate, max );

    ASSERT_EQ( 1, result );
}

TEST( TestWrapping, ClampOrWrapCoordinateMirrorOutsideRegionNegativeSeveralMultiples )
{
    int        coordinate = -22;
    int        max        = 10;
    RTwrapmode wrapMode   = RT_WRAP_MIRROR;

    int result = clampOrWrapCoordinate( wrapMode, coordinate, max );

    ASSERT_EQ( 1, result );
}

TEST( TestWrapping, DetermineReadStartAndSizeWhenRegionIsntTransformed )
{
    RTwrapmode wrapMode = RT_WRAP_CLAMP_TO_EDGE;
    int        start    = 0;
    int        end      = 10;
    int        max      = 10;

    int startResult = -1;
    int sizeResult  = -1;

    determineReadStartAndSize( wrapMode, start, end, max, &startResult, &sizeResult );

    ASSERT_EQ( 0, startResult );
    ASSERT_EQ( 10, sizeResult );
}

TEST( TestWrapping, DetermineReadStartAndSizeWhenRegionIsClampedInNegative )
{
    RTwrapmode wrapMode = RT_WRAP_CLAMP_TO_EDGE;
    int        start    = -3;
    int        end      = 10;
    int        max      = 10;

    int startResult = -1;
    int sizeResult  = -1;

    determineReadStartAndSize( wrapMode, start, end, max, &startResult, &sizeResult );

    ASSERT_EQ( 0, startResult );
    ASSERT_EQ( 10, sizeResult );
}

TEST( TestWrapping, DetermineReadStartAndSizeWhenRegionIsClampedInPositive )
{
    RTwrapmode wrapMode = RT_WRAP_CLAMP_TO_EDGE;
    int        start    = 0;
    int        end      = 15;
    int        max      = 10;

    int startResult = -1;
    int sizeResult  = -1;

    determineReadStartAndSize( wrapMode, start, end, max, &startResult, &sizeResult );

    ASSERT_EQ( 0, startResult );
    ASSERT_EQ( 10, sizeResult );
}

TEST( TestWrapping, DetermineReadStartAndSizeWhenRegionIsClampedInNegativeAndPositive )
{
    RTwrapmode wrapMode = RT_WRAP_CLAMP_TO_EDGE;
    int        start    = -3;
    int        end      = 12;
    int        max      = 10;

    int startResult = -1;
    int sizeResult  = -1;

    determineReadStartAndSize( wrapMode, start, end, max, &startResult, &sizeResult );

    ASSERT_EQ( 0, startResult );
    ASSERT_EQ( 10, sizeResult );
}

TEST( TestWrapping, DetermineReadStartAndSizeRegionWrapRepeatNegative )
{
    RTwrapmode wrapMode = RT_WRAP_REPEAT;
    int        start    = -3;
    int        end      = 0;
    int        max      = 10;

    int startResult = -1;
    int sizeResult  = -1;

    determineReadStartAndSize( wrapMode, start, end, max, &startResult, &sizeResult );

    ASSERT_EQ( 7, startResult );
    ASSERT_EQ( 3, sizeResult );
}

TEST( TestWrapping, DetermineReadStartAndSizeRegionWrapRepeatPositive )
{
    RTwrapmode wrapMode = RT_WRAP_REPEAT;
    int        start    = 10;
    int        end      = 13;
    int        max      = 10;

    int startResult = -1;
    int sizeResult  = -1;

    determineReadStartAndSize( wrapMode, start, end, max, &startResult, &sizeResult );

    ASSERT_EQ( 0, startResult );
    ASSERT_EQ( 3, sizeResult );
}

TEST( TestWrapping, DetermineReadStartAndSizeRegionWrapMirrorNegative )
{
    RTwrapmode wrapMode = RT_WRAP_MIRROR;
    int        start    = -3;
    int        end      = 0;
    int        max      = 10;

    int startResult = -1;
    int sizeResult  = -1;

    determineReadStartAndSize( wrapMode, start, end, max, &startResult, &sizeResult );

    ASSERT_EQ( 0, startResult );
    ASSERT_EQ( 3, sizeResult );
}

TEST( TestWrapping, DetermineReadStartAndSizeRegionWrapMirrorPositive )
{
    RTwrapmode wrapMode = RT_WRAP_MIRROR;
    int        start    = 10;
    int        end      = 13;
    int        max      = 10;

    int startResult = -1;
    int sizeResult  = -1;

    determineReadStartAndSize( wrapMode, start, end, max, &startResult, &sizeResult );

    ASSERT_EQ( 7, startResult );
    ASSERT_EQ( 3, sizeResult );
}

TEST( TestWrapping, DetermineReadStartAndSizeRegionWrapMirrorPositiveBorder )
{
    RTwrapmode wrapMode = RT_WRAP_MIRROR;
    int        start    = 10;
    int        end      = 11;
    int        max      = 10;

    int startResult = -1;
    int sizeResult  = -1;

    determineReadStartAndSize( wrapMode, start, end, max, &startResult, &sizeResult );

    ASSERT_EQ( 9, startResult );
    ASSERT_EQ( 1, sizeResult );
}

TEST( TestWrapping, DetermineReadRegionCoordsAllCoordsInRegion )
{
    int startCoord = 5;
    int regionSize = 4;
    int imageSize  = 10;

    std::vector<int> coordsToRead;
    determineReadRegionCoords( startCoord, regionSize, imageSize, coordsToRead );

    ASSERT_EQ( 2, coordsToRead.size() );
    ASSERT_EQ( 5, coordsToRead[0] );
    ASSERT_EQ( 9, coordsToRead[1] );
}

TEST( TestWrapping, DetermineReadRegionCoordsOutsideRegionInNegativeAndPositive )
{
    int startCoord = -3;
    int regionSize = 15;
    int imageSize  = 10;

    std::vector<int> coordsToRead;
    determineReadRegionCoords( startCoord, regionSize, imageSize, coordsToRead );

    ASSERT_EQ( 4, coordsToRead.size() );
    ASSERT_EQ( -3, coordsToRead[0] );
    ASSERT_EQ( 0, coordsToRead[1] );
    ASSERT_EQ( 10, coordsToRead[2] );
    ASSERT_EQ( 12, coordsToRead[3] );
}

TEST( TestWrapping, DetermineReadRegionCoordsOutsideRegionInNegative )
{
    int startCoord = -3;
    int regionSize = 13;
    int imageSize  = 10;

    std::vector<int> coordsToRead;
    determineReadRegionCoords( startCoord, regionSize, imageSize, coordsToRead );

    ASSERT_EQ( 3, coordsToRead.size() );
    ASSERT_EQ( -3, coordsToRead[0] );
    ASSERT_EQ( 0, coordsToRead[1] );
    ASSERT_EQ( 10, coordsToRead[2] );
}

TEST( TestWrapping, DetermineWrappedReadRegionsCoordsInsideImage )
{
    RTwrapmode wrapModeX   = RT_WRAP_CLAMP_TO_EDGE;
    RTwrapmode wrapModeY   = RT_WRAP_CLAMP_TO_EDGE;
    int        startX      = 3;
    int        startY      = 4;
    int        width       = 10;
    int        height      = 12;
    int        imageWidth  = 64;
    int        imageHeight = 32;

    std::vector<ReadRegion> wrappedRegions( MAX_READ_REGIONS );
    unsigned int numReadRegions = determineWrappedReadRegions( wrapModeX, wrapModeY, startX, startY, width, height,
                                                               imageWidth, imageHeight, wrappedRegions.data() );

    ASSERT_EQ( 1, numReadRegions );

    const ReadRegion& resultRegion = wrappedRegions[0];
    ASSERT_EQ( 3, resultRegion.imageX() );
    ASSERT_EQ( 10, resultRegion.width() );
    ASSERT_EQ( 4, resultRegion.imageY() );
    ASSERT_EQ( 12, resultRegion.height() );
    ASSERT_EQ( 0, resultRegion.destinationBufferXOffset() );
    ASSERT_EQ( 0, resultRegion.destinationBufferYOffset() );
}

TEST( TestWrapping, DetermineWrappedReadRegionsCoordsOutsideImageInXAndYWrapRepeat )
{
    RTwrapmode wrapModeX   = RT_WRAP_REPEAT;
    RTwrapmode wrapModeY   = RT_WRAP_REPEAT;
    int        startX      = -3;
    int        startY      = -4;
    int        width       = 69;
    int        height      = 71;
    int        imageWidth  = 64;
    int        imageHeight = 64;

    std::vector<ReadRegion> resultRegions( MAX_READ_REGIONS );
    unsigned int numReadRegions = determineWrappedReadRegions( wrapModeX, wrapModeY, startX, startY, width, height,
                                                               imageWidth, imageHeight, resultRegions.data() );

    ASSERT_EQ( 9, numReadRegions );
    resultRegions.resize( numReadRegions );

    std::vector<ReadRegion> expectedRegions;
    expectedRegions.emplace_back( 61, 3, 60, 4, 0, 0, false, false );
    expectedRegions.emplace_back( 0, 64, 60, 4, 3, 0, false, false );
    expectedRegions.emplace_back( 0, 2, 60, 4, 67, 0, false, false );
    expectedRegions.emplace_back( 61, 3, 0, 64, 0, 4, false, false );
    expectedRegions.emplace_back( 0, 64, 0, 64, 3, 4, false, false );
    expectedRegions.emplace_back( 0, 2, 0, 64, 67, 4, false, false );
    expectedRegions.emplace_back( 61, 3, 0, 3, 0, 68, false, false );
    expectedRegions.emplace_back( 0, 64, 0, 3, 3, 68, false, false );
    expectedRegions.emplace_back( 0, 2, 0, 3, 67, 68, false, false );

    std::sort( expectedRegions.begin(), expectedRegions.end() );
    std::sort( resultRegions.begin(), resultRegions.end() );

    for( int i = 0; i < numReadRegions; ++i )
        EXPECT_EQ( expectedRegions[i], resultRegions[i] );

    if( HasFailure() )
        printExpectedAndResultRegions( expectedRegions, resultRegions );
}

TEST( TestWrapping, DetermineWrappedReadRegionsCoordsOutsideImageInXAndYWrapMirror )
{
    RTwrapmode wrapModeX   = RT_WRAP_MIRROR;
    RTwrapmode wrapModeY   = RT_WRAP_MIRROR;
    int        startX      = -3;
    int        startY      = -4;
    int        width       = 69;
    int        height      = 71;
    int        imageWidth  = 64;
    int        imageHeight = 64;

    std::vector<ReadRegion> resultRegions( MAX_READ_REGIONS );
    unsigned int numReadRegions = determineWrappedReadRegions( wrapModeX, wrapModeY, startX, startY, width, height,
                                                               imageWidth, imageHeight, resultRegions.data() );

    ASSERT_EQ( 9, numReadRegions );
    resultRegions.resize( numReadRegions );

    std::vector<ReadRegion> expectedRegions;
    expectedRegions.emplace_back( 0, 3, 0, 4, 0, 0, true, true );
    expectedRegions.emplace_back( 0, 64, 0, 4, 3, 0, false, true );
    expectedRegions.emplace_back( 62, 2, 0, 4, 67, 0, true, true );
    expectedRegions.emplace_back( 0, 3, 0, 64, 0, 4, true, false );
    expectedRegions.emplace_back( 0, 64, 0, 64, 3, 4, false, false );
    expectedRegions.emplace_back( 62, 2, 0, 64, 67, 4, true, false );
    expectedRegions.emplace_back( 0, 3, 61, 3, 0, 68, true, true );
    expectedRegions.emplace_back( 0, 64, 61, 3, 3, 68, false, true );
    expectedRegions.emplace_back( 62, 2, 61, 3, 67, 68, true, true );

    std::sort( expectedRegions.begin(), expectedRegions.end() );
    std::sort( resultRegions.begin(), resultRegions.end() );

    for( int i = 0; i < numReadRegions; ++i )
        EXPECT_EQ( expectedRegions[i], resultRegions[i] );

    if( HasFailure() )
        printExpectedAndResultRegions( expectedRegions, resultRegions );
}

TEST( TestWrapping, DetermineWrappedReadRegionsCoordsOutsideImageInXAndYClamp )
{
    RTwrapmode wrapModeX   = RT_WRAP_CLAMP_TO_EDGE;
    RTwrapmode wrapModeY   = RT_WRAP_CLAMP_TO_EDGE;
    int        startX      = -3;
    int        startY      = -4;
    int        width       = 69;
    int        height      = 71;
    int        imageWidth  = 64;
    int        imageHeight = 64;

    std::vector<ReadRegion> resultRegions( MAX_READ_REGIONS );
    unsigned int numReadRegions = determineWrappedReadRegions( wrapModeX, wrapModeY, startX, startY, width, height,
                                                               imageWidth, imageHeight, resultRegions.data() );

    ReadRegion expectedRegion( 0, 64, 0, 64, 3, 4, false, false );

    ASSERT_EQ( 1, numReadRegions );
    ASSERT_TRUE( expectedRegion == resultRegions[0] );
}

TEST( TestWrapping, DetermineWrappedReadRegionsCoordsOutsideImageInXAndYMixed )
{
    RTwrapmode wrapModeX   = RT_WRAP_REPEAT;
    RTwrapmode wrapModeY   = RT_WRAP_CLAMP_TO_EDGE;
    int        startX      = -3;
    int        startY      = -4;
    int        width       = 32;
    int        height      = 32;
    int        imageWidth  = 64;
    int        imageHeight = 64;

    std::vector<ReadRegion> resultRegions( MAX_READ_REGIONS );
    unsigned int numReadRegions = determineWrappedReadRegions( wrapModeX, wrapModeY, startX, startY, width, height,
                                                               imageWidth, imageHeight, resultRegions.data() );
    ASSERT_EQ( 2, numReadRegions );
    resultRegions.resize( numReadRegions );

    std::vector<ReadRegion> expectedRegions;
    expectedRegions.emplace_back( 0, 29, 0, 28, 3, 4, false, false );
    expectedRegions.emplace_back( 61, 3, 0, 28, 0, 4, false, false );

    std::sort( expectedRegions.begin(), expectedRegions.begin() + numReadRegions );
    std::sort( resultRegions.begin(), resultRegions.begin() + numReadRegions );

    for( int i = 0; i < numReadRegions; ++i )
        EXPECT_EQ( expectedRegions[i], resultRegions[i] );

    if( HasFailure() )
        printExpectedAndResultRegions( expectedRegions, resultRegions );
}

TEST( TestWrapping, TransposeRegionXAndYOddWidthHeight )
{
    int region[3][3];
    for( int y = 0; y < 3; ++y )
        for( int x       = 0; x < 3; ++x )
            region[y][x] = y * 3 + x;

    transposeRegion( true, true, (unsigned char*)&region, 3, 3, sizeof( int ), 3 * sizeof( int ) );

    for( int y = 0; y < 3; ++y )
        for( int x = 0; x < 3; ++x )
            EXPECT_EQ( 8 - ( y * 3 + x ), region[y][x] );
}

TEST( TestWrapping, TransposeRegionXAndYEvenWidthHeight )
{
    int region[2][2];
    for( int y = 0; y < 2; ++y )
        for( int x       = 0; x < 2; ++x )
            region[y][x] = y * 2 + x;

    transposeRegion( true, true, (unsigned char*)&region, 2, 2, sizeof( int ), 2 * sizeof( int ) );

    for( int y = 0; y < 2; ++y )
        for( int x = 0; x < 2; ++x )
            EXPECT_EQ( 3 - ( y * 2 + x ), region[y][x] );
}

TEST( TestWrapping, TransposeRegionXOddWidthHeight )
{
    int region[3][3];
    for( int y = 0; y < 3; ++y )
        for( int x       = 0; x < 3; ++x )
            region[y][x] = y * 3 + x;

    transposeRegion( true, false, (unsigned char*)&region, 3, 3, sizeof( int ), 3 * sizeof( int ) );

    for( int y = 0; y < 3; ++y )
        for( int x = 0; x < 3; ++x )
            EXPECT_EQ( y * 3 + x, region[y][2 - x] );
}

TEST( TestWrapping, TransposeRegionYOddWidthHeight )
{
    int region[3][3];
    for( int y = 0; y < 3; ++y )
        for( int x       = 0; x < 3; ++x )
            region[y][x] = y * 3 + x;

    transposeRegion( false, true, (unsigned char*)&region, 3, 3, sizeof( int ), 3 * sizeof( int ) );

    for( int y = 0; y < 3; ++y )
        for( int x = 0; x < 3; ++x )
            EXPECT_EQ( ( 2 - y ) * 3 + x, region[y][x] );
}

TEST( TestWrapping, TransposeRegionXAndYWithRowPitch )
{
    int region[3][6];
    for( int y = 0; y < 3; ++y )
        for( int x       = 0; x < 3; ++x )
            region[y][x] = y * 3 + x;
    for( int y = 0; y < 3; ++y )
        for( int x       = 3; x < 6; ++x )
            region[y][x] = -1;

    transposeRegion( true, true, (unsigned char*)&region, 3, 3, sizeof( int ), 6 * sizeof( int ) );

    for( int y = 0; y < 3; ++y )
        for( int x = 0; x < 3; ++x )
            EXPECT_EQ( 8 - ( y * 3 + x ), region[y][x] );
    for( int y = 0; y < 3; ++y )
        for( int x = 3; x < 6; ++x )
            EXPECT_EQ( -1, region[y][x] );
}
