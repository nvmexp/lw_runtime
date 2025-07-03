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

// Collection of functions to help with wrapping.

#include <Memory/DemandLoad/RequestHandler/WrapHelpers.h>

#include <algorithm>
#include <cstring>

namespace optix {

ReadRegion::ReadRegion( int imageX, int width, int imageY, int height, int destinationBufferXOffset, int destinationBufferYOffset, bool transposeX, bool transposeY )
    : m_imageX( imageX )
    , m_destinationBufferXOffset( destinationBufferXOffset )
    , m_width( width )
    , m_imageY( imageY )
    , m_destinationBufferYOffset( destinationBufferYOffset )
    , m_height( height )
    , m_transposeX( transposeX )
    , m_transposeY( transposeY )
{
}

ReadRegion::ReadRegion()
    : m_imageX( -1 )
    , m_destinationBufferXOffset( -1 )
    , m_width( -1 )
    , m_imageY( -1 )
    , m_destinationBufferYOffset( -1 )
    , m_height( -1 )
    , m_transposeX( 0 )
    , m_transposeY( 0 )
{
}

bool ReadRegion::operator==( const ReadRegion& rhs ) const
{
    return m_imageX == rhs.m_imageX && m_destinationBufferXOffset == rhs.m_destinationBufferXOffset && m_width == rhs.m_width
           && m_imageY == rhs.m_imageY && m_destinationBufferYOffset == rhs.m_destinationBufferYOffset
           && m_height == rhs.m_height && m_transposeX == rhs.m_transposeX && m_transposeY == rhs.m_transposeY;
}

bool ReadRegion::operator<( const ReadRegion& other ) const
{
    if( m_imageX != other.m_imageX )
        return m_imageX < other.m_imageX;
    if( m_imageY != other.m_imageY )
        return m_imageY < other.m_imageY;
    if( m_destinationBufferXOffset != other.m_destinationBufferXOffset )
        return m_destinationBufferXOffset < other.m_destinationBufferXOffset;
    if( m_destinationBufferYOffset != other.m_destinationBufferYOffset )
        return m_destinationBufferYOffset < other.m_destinationBufferYOffset;
    if( m_width != other.m_width )
        return m_width < other.m_width;
    if( m_height != other.m_height )
        return m_height < other.m_height;
    if( m_transposeX != other.m_transposeX )
        return m_transposeX < other.m_transposeX;
    return m_transposeY < other.m_transposeY;
}

int clampOrWrapCoordinate( RTwrapmode wrapMode, int coord, int max )
{
    switch( wrapMode )
    {
        case RT_WRAP_CLAMP_TO_EDGE:
        case RT_WRAP_CLAMP_TO_BORDER:
            return std::min( max, std::max( coord, 0 ) );
        case RT_WRAP_REPEAT:
            if( coord >= max )
                return coord % max;
            if( coord < 0 )
                return max - ( ( -coord ) % max );
            return coord;
        case RT_WRAP_MIRROR:
            if( coord >= max )
                return max - ( coord % max );
            if( coord < 0 )
                // We need to subtract one here to account for the fact that
                // the positive region starts with 0, but the negative region
                // starts with -1.
                return ( ( coord * -1 ) % max ) - 1;
            return coord;

        default:
            return -1;
    }
}

void determineReadStartAndSize( RTwrapmode wrapMode, int start, int end, int max, int* startOut, int* sizeOut )
{
    int wrappedStart = clampOrWrapCoordinate( wrapMode, start, max );
    int size         = -1;
    if( wrapMode == RT_WRAP_CLAMP_TO_BORDER || wrapMode == RT_WRAP_CLAMP_TO_EDGE )
    {
        int wrappedEnd = clampOrWrapCoordinate( wrapMode, end, max );
        size           = wrappedEnd - wrappedStart;
    }
    else if( wrapMode == RT_WRAP_MIRROR )
    {
        int wrappedEnd = clampOrWrapCoordinate( wrapMode, end, max );
        wrappedStart   = std::min( wrappedStart, wrappedEnd );
        size           = end - start;
    }
    else
        size = end - start;

    *startOut = wrappedStart;
    *sizeOut  = size;
}

void determineReadRegionCoords( int regionStartCoord, int regionSize, int imageSize, std::vector<int>& regionReadCoords )
{
    regionReadCoords.push_back( regionStartCoord );
    if( regionStartCoord < 0 )
        regionReadCoords.push_back( 0 );
    if( regionStartCoord + regionSize > imageSize )
        regionReadCoords.push_back( imageSize );
    regionReadCoords.push_back( regionStartCoord + regionSize );
}

unsigned int determineWrappedReadRegions( RTwrapmode  wrapModeX,
                                          RTwrapmode  wrapModeY,
                                          int         regionStartX,
                                          int         regionStartY,
                                          int         regionWidth,
                                          int         regionHeight,
                                          int         imageWidth,
                                          int         imageHeight,
                                          ReadRegion* wrappedRegions )
{
    // Build two spans, each containing the x and y coordinates of the regions to read.
    std::vector<int> xCoordsToRead;
    determineReadRegionCoords( regionStartX, regionWidth, imageWidth, xCoordsToRead );

    std::vector<int> yCoordsToRead;
    determineReadRegionCoords( regionStartY, regionHeight, imageHeight, yCoordsToRead );

    unsigned int numReadRegions = 0;

    for( int yIdx = 0; yIdx < yCoordsToRead.size() - 1; ++yIdx )
    {
        int yStart = -1;
        int height = -1;
        determineReadStartAndSize( wrapModeY, yCoordsToRead[yIdx], yCoordsToRead[yIdx + 1], imageHeight, &yStart, &height );

        bool transposeY = wrapModeY == RT_WRAP_MIRROR && yStart != yCoordsToRead[yIdx];

        int yBufferOffset = yCoordsToRead[yIdx] - regionStartY;

        for( int xIdx = 0; xIdx < xCoordsToRead.size() - 1; ++xIdx )
        {
            int xStart = -1;
            int width  = -1;
            determineReadStartAndSize( wrapModeX, xCoordsToRead[xIdx], xCoordsToRead[xIdx + 1], imageWidth, &xStart, &width );

            bool transposeX = wrapModeX == RT_WRAP_MIRROR && xStart != xCoordsToRead[xIdx];

            int xBufferOffset = xCoordsToRead[xIdx] - regionStartX;

            if( width > 0 && height > 0 )
            {
                wrappedRegions[numReadRegions] =
                    ReadRegion{xStart, width, yStart, height, xBufferOffset, yBufferOffset, transposeX, transposeY};
                numReadRegions++;
            }
        }
    }

    return numReadRegions;
}

void clampRegionHorizontal( unsigned char* baseAddress, int x, int y, int width, int height, int xMax, int yMax, int elementSize, int rowPitch, void* sourceColor )
{
    for( int lwrrX = 0; lwrrX < width; ++lwrrX )
    {
        int textureX        = lwrrX + x;
        int clampedTextureX = std::min( xMax, std::max( 0, textureX ) );
        int destX           = clampedTextureX - x;

        if( clampedTextureX != textureX )
        {
            for( int lwrrY = 0; lwrrY < height; ++lwrrY )
            {
                int textureY        = lwrrY + y;
                int clampedTextureY = std::min( yMax, std::max( 0, textureY ) );
                int destY           = lwrrY;  // Don't clamp this coordinate

                unsigned char* lwrrDest = baseAddress + lwrrY * rowPitch + lwrrX * elementSize;
                void* lwrrSource = sourceColor ? sourceColor : (unsigned char*)baseAddress + destY * rowPitch + destX * elementSize;
                memcpy( lwrrDest, lwrrSource, elementSize );
            }
        }
    }
}

void clampRegiolwertical( unsigned char* baseAddress, int x, int y, int width, int height, int xMax, int yMax, int elementSize, int rowPitch, void* sourceColor )
{
    for( int lwrrY = 0; lwrrY < width; ++lwrrY )
    {
        int textureY        = lwrrY + y;
        int clampedTextureY = std::min( yMax, std::max( 0, textureY ) );
        int destY           = clampedTextureY - y;

        if( clampedTextureY != textureY )
        {
            for( int lwrrX = 0; lwrrX < width; ++lwrrX )
            {
                int textureX        = lwrrX + x;
                int clampedTextureX = std::min( yMax, std::max( 0, textureX ) );
                int destX           = lwrrX;  // Don't clamp this coordinate

                unsigned char* lwrrDest = baseAddress + lwrrY * rowPitch + lwrrX * elementSize;
                void* lwrrSource = sourceColor ? sourceColor : (unsigned char*)baseAddress + destY * rowPitch + destX * elementSize;
                memcpy( lwrrDest, lwrrSource, elementSize );
            }
        }
    }
}

// Transpose the specified region in the X and/or Y axes.
void transposeRegion( bool transposeX, bool transposeY, unsigned char* region, int width, int height, int elementSize, int rowPitch )
{
    // We should never have an element larger than a float4 (16 bytes).
    unsigned char tmp[16];

    if( transposeX && transposeY )
    {
        for( int y = 0; y < ( height + 1 ) / 2; ++y )
        {
            for( int x = 0; x < width; ++x )
            {
                if( y * width + x > width * height / 2 )
                    break;

                unsigned char* first  = region + y * rowPitch + x * elementSize;
                unsigned char* second = region + ( height - 1 - y ) * rowPitch + ( width - 1 - x ) * elementSize;

                memcpy( tmp, first, elementSize );
                memcpy( first, second, elementSize );
                memcpy( second, tmp, elementSize );
            }
        }
    }
    else if( transposeX )
    {
        for( int y = 0; y < height; ++y )
        {
            for( int x = 0; x < width / 2; ++x )
            {
                unsigned char* first  = region + y * rowPitch + x * elementSize;
                unsigned char* second = region + y * rowPitch + ( width - 1 - x ) * elementSize;

                memcpy( tmp, first, elementSize );
                memcpy( first, second, elementSize );
                memcpy( second, tmp, elementSize );
            }
        }
    }
    else if( transposeY )
    {
        for( int y = 0; y < height / 2; ++y )
        {
            for( int x = 0; x < width; ++x )
            {
                unsigned char* first  = region + y * rowPitch + x * elementSize;
                unsigned char* second = region + ( height - 1 - y ) * rowPitch + x * elementSize;

                memcpy( tmp, first, elementSize );
                memcpy( first, second, elementSize );
                memcpy( second, tmp, elementSize );
            }
        }
    }
}

}  // namespace optix
