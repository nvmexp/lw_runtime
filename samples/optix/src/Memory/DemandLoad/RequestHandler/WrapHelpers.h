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

#pragma once

#include <o6/optix.h>

#include <vector>

namespace optix {

// The maximum number of regions we might need (the main region, and 8 wrapped
// regions: one for each side, and one for each corner).
enum
{
    MAX_READ_REGIONS = 9
};

// Contains the information necessary to read a wrapped region into an output
// buffer.
class ReadRegion
{
  public:
    ReadRegion();
    ReadRegion( int imageX, int width, int imageY, int height, int destinationBufferXOffset, int destinationBufferYOffset, bool transposeX, bool transposeY );

    bool operator==( const ReadRegion& rhs ) const;
    bool operator<( const ReadRegion& other ) const;

    int  imageX() const { return m_imageX; }
    int  imageY() const { return m_imageY; }
    int  destinationBufferXOffset() const { return m_destinationBufferXOffset; }
    int  destinationBufferYOffset() const { return m_destinationBufferYOffset; }
    int  width() const { return m_width; }
    int  height() const { return m_height; }
    bool transposeX() const { return m_transposeX; }
    bool transposeY() const { return m_transposeY; }

  private:
    int m_imageX;
    int m_destinationBufferXOffset;
    int m_width;

    int m_imageY;
    int m_destinationBufferYOffset;
    int m_height;

    bool m_transposeX;
    bool m_transposeY;
};

// Clamp or wrap the given coordinate between 0 and max based on the specifed
// wrap mode.
int clampOrWrapCoordinate( RTwrapmode wrapMode, int coord, int max );

// Determine the starting coordinate and read size based on the specified start
// and end coordinates and maximum coordinate.
void determineReadStartAndSize( RTwrapmode wrapMode, int start, int end, int max, int* startOut, int* sizeOut );

// Based on the given start coordinate, read region size, and image size, break
// the given region into spans to be wrapped or clamped and read.
void determineReadRegionCoords( int regionStartCoord, int regionSize, int imageSize, std::vector<int>& regionReadCoords );

// Determine the set of wrapped (or clamped) sub-regions to read into the given
// region.
unsigned int determineWrappedReadRegions( RTwrapmode  wrapModeX,
                                          RTwrapmode  wrapModeY,
                                          int         regionStartX,
                                          int         regionStartY,
                                          int         regionWidth,
                                          int         regionHeight,
                                          int         imageWidth,
                                          int         imageHeight,
                                          ReadRegion* wrappedRegions );

// Perform clamping on the given region (both horizontal and vertical).
//
// Parameters:
//   baseAddress - the address of the image
//   x - the x coordinate of the start of the region, relative to the image (may be negative or greater than the width of the image)
//   y - the y coordinate of the start of the region
//   width - the width of the region
//   height - the height of the region
//   xMax - the maximum x coordinate of the image (i.e. the coordinate to clamp to)
//   yMax - the maximum y coordinate of the iamge
//   elementSize - the size, in bytes, of a single element of the image
//   rowPitch - the size, in bytes, of a single row of the region
//   sourceColor - if not null, the address of a color to use as a border. If null, the pixels at the image's edge are used

void clampRegionHorizontal( unsigned char* baseAddress, int x, int y, int width, int height, int xMax, int yMax, int elementSize, int rowPitch, void* sourceColor );

inline void clampRegionHorizontal( const RTmemoryblock& memoryBlock,
                                   unsigned int         elementSize,
                                   unsigned int         rowPitch,
                                   unsigned int         imageWidth,
                                   unsigned int         imageHeight,
                                   void*                srcColor )
{
    clampRegionHorizontal( (unsigned char*)memoryBlock.baseAddress, memoryBlock.x, memoryBlock.y, memoryBlock.width,
                           memoryBlock.height, imageWidth - 1, imageHeight - 1, elementSize, rowPitch, srcColor );
}

void clampRegiolwertical( unsigned char* baseAddress, int x, int y, int width, int height, int xMax, int yMax, int elementSize, int rowPitch, void* sourceColor );

inline void clampRegiolwertical( const RTmemoryblock& memoryBlock,
                                 unsigned int         elementSize,
                                 unsigned int         rowPitch,
                                 unsigned int         imageWidth,
                                 unsigned int         imageHeight,
                                 void*                srcColor )
{
    clampRegiolwertical( (unsigned char*)memoryBlock.baseAddress, memoryBlock.x, memoryBlock.y, memoryBlock.width,
                         memoryBlock.height, imageWidth - 1, imageHeight - 1, elementSize, rowPitch, srcColor );
}

// Transpose the specified region in the X and/or Y axes.
void transposeRegion( bool transposeX, bool transposeY, unsigned char* region, int width, int height, int elementSize, int rowPitch );

inline void transposeRegion( const ReadRegion& region, const RTmemoryblock& block, unsigned int elementSize, unsigned int rowPitch )
{
    transposeRegion( region.transposeX(), region.transposeY(), (unsigned char*)block.baseAddress, block.width,
                     block.height, elementSize, rowPitch );
}

}  // namespace optix
