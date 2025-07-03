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

#include <Memory/DemandLoad/HardwareTileManager.h>

#include <Memory/MemoryManager.h>
#include <Objects/Buffer.h>

namespace optix {

void HardwareTileManager::activate( MemoryManager* mm, int allDeviceListIndex )
{
    m_mm = mm;
    m_dims = BufferDimensions( RT_FORMAT_BYTE, 1, 1, BUFFER_SIZE, 1, 1, /* levels */ 1, /* lwbe */ false, /* layered */ false );
}

void HardwareTileManager::acquireNewChunk()
{
    m_storage.push_back( m_mm->allocateMBuffer( m_dims, MBufferPolicy::readonly_sparse_backing ) );
    m_lwrrHandle++;
    m_mm->manualSynchronize( m_storage[m_lwrrHandle] );
    m_lwrrOffset = 0;
}

unsigned int HardwareTileManager::reserveTile( int size )
{
    RT_ASSERT( size % TILE_SIZE == 0 );

    // If there isn't enough room in the buffer to allocate the given size, create a new buffer.
    if( m_lwrrOffset + size >= BUFFER_SIZE || m_lwrrHandle < 0 )
        acquireNewChunk();

    const unsigned int lwrrOffset = m_lwrrOffset;

    m_lwrrOffset += size;
    if( m_lwrrOffset >= BUFFER_SIZE )
        acquireNewChunk();

    return lwrrOffset;
}

void HardwareTileManager::bindTileToMemoryAsync( lwca::Stream&        stream,
                                                 const Buffer*        buffer,
                                                 unsigned int         allDeviceListIndex,
                                                 const RTmemoryblock& memBlock )
{
    const unsigned int offset = reserveTile();
    RT_ASSERT( offset < m_storage[m_lwrrHandle]->getDimensions().getTotalSizeInBytes() );

    m_mm->fillHardwareTileAsync( stream, buffer->getMBuffer(), m_storage[m_lwrrHandle], allDeviceListIndex, memBlock, offset );
}

void HardwareTileManager::bindMipTailToMemoryAsync( lwca::Stream& stream, const Buffer* buffer, unsigned int allDeviceListIndex, int mipTailSizeInBytes )
{
    const unsigned int offset = reserveTile( mipTailSizeInBytes );
    RT_ASSERT( offset < m_storage[0]->getDimensions().getTotalSizeInBytes() );

    m_mm->bindHardwareMipTailAsync( stream, buffer->getMBuffer(), m_storage[m_lwrrHandle], allDeviceListIndex,
                                    mipTailSizeInBytes, offset );
}

void HardwareTileManager::fillHardwareMipTail( const Buffer* buffer, unsigned int allDeviceListIndex, const RTmemoryblock& memBlock )
{
    const unsigned int offset = reserveTile();
    RT_ASSERT( offset < m_storage[0]->getDimensions().getTotalSizeInBytes() );

    m_mm->fillHardwareMipTail( buffer->getMBuffer(), allDeviceListIndex, memBlock );
}

void HardwareTileManager::fillHardwareMipTailAsync( lwca::Stream&        stream,
                                                    const Buffer*        buffer,
                                                    unsigned int         allDeviceListIndex,
                                                    const RTmemoryblock& memBlock )
{
    const unsigned int offset = reserveTile();
    RT_ASSERT( offset < m_storage[0]->getDimensions().getTotalSizeInBytes() );

    m_mm->fillHardwareMipTailAsync( stream, buffer->getMBuffer(), allDeviceListIndex, memBlock );
}

}  // namespace demandLoading
