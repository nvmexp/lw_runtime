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

#include <Memory/BufferDimensions.h>
#include <Memory/MBuffer.h>

#include <o6/optix.h>

#include <vector>

namespace optix {

class Buffer;
class MemoryManager;

namespace lwca {
class Stream;
}  // namespace lwca

class HardwareTileManager
{
  public:
    void activate( MemoryManager* mm, int allDeviceListIndex );
    void bindTileToMemoryAsync( lwca::Stream& stream, const Buffer* buffer, unsigned int allDeviceListIndex, const RTmemoryblock& memBlock );
    void bindMipTailToMemoryAsync( lwca::Stream& stream, const Buffer* buffer, unsigned int allDeviceListIndex, int miptailSize );
    void fillHardwareMipTail( const Buffer* buffer, unsigned int allDeviceListIndex, const RTmemoryblock& memBlock );
    void fillHardwareMipTailAsync( lwca::Stream& stream, const Buffer* buffer, unsigned int allDeviceListIndex, const RTmemoryblock& memBlock );

  private:
    // TODO: Use lwMemGetAllocationGranularity to determine optimal backing storage size
    static const unsigned int TILES_PER_ALLOCATION = 2048;
    static const unsigned int TILE_SIZE            = 65536;
    static const unsigned int BUFFER_SIZE          = TILES_PER_ALLOCATION * TILE_SIZE;

    BufferDimensions m_dims;

    MemoryManager* m_mm = nullptr;

    unsigned int               m_lwrrOffset = 0;
    int                        m_lwrrHandle = -1;
    std::vector<MBufferHandle> m_storage;

    void acquireNewChunk();

    unsigned int reserveTile( int size = TILE_SIZE );
};

}  // namespace optix
