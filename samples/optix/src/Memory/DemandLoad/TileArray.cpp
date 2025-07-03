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

#include <Memory/DemandLoad/TileArray.h>

#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Memory/MTextureSampler.h>
#include <Memory/MemoryManager.h>

#include <Util/MakeUnique.h>
#include <Util/TextureDescriptor.h>

#include <prodlib/misc/RTFormatUtil.h>

using namespace optix::demandLoad;

namespace optix {

TileArray::TileArray( MemoryManager* mm, unsigned int numTiles, unsigned int tileWidth, unsigned int tileHeight, RTformat format, const TextureDescriptor& texDesc )
    : m_mm( mm )
{
    LOG_MEDIUM_VERBOSE( "TileArray::TileArray numTiles " << numTiles << ' ' << tileWidth << 'x' << tileHeight << ", "
                                                         << format << '\n' );

    const BufferDimensions dims( format, prodlib::getElementSize( format ), /* dims */ 3, tileWidth, tileHeight,
                                 numTiles, /* levels */ 1, /* lwbe */ false, /* layered */ true );
    m_storage = mm->allocateMBuffer( dims, MBufferPolicy::tileArray_readOnly_demandLoad );
    m_sampler = mm->attachMTextureSampler( m_storage, texDesc );
    mm->manualSynchronize( m_storage );
}

void TileArray::fillTile( unsigned int allDeviceListIndex, unsigned int index, const void* data )
{
    LOG_MEDIUM_VERBOSE( "TileArray::fillTile device " << allDeviceListIndex << ", index " << index << '\n' );

    m_mm->fillTile( m_storage, allDeviceListIndex, index, data );
}

void TileArray::fillTileAsync( lwca::Stream& stream, unsigned int allDeviceListIndex, unsigned int index, const void* data )
{
    LOG_MEDIUM_VERBOSE( "TileArray::fillTileAsync device " << allDeviceListIndex << ", index " << index << '\n' );

    m_mm->fillTileAsync( stream, m_storage, allDeviceListIndex, index, data );
}

void TileArray::synchronize( unsigned int allDeviceListIndex )
{
    LOG_MEDIUM_VERBOSE( "TileArray::synchronize device " << allDeviceListIndex << '\n' );

    m_mm->manualSynchronize( m_sampler, allDeviceListIndex );
    const MAccess access = m_sampler->getAccess( allDeviceListIndex );
    m_texObject          = access.getTexObject().texObject.get();
}

}  // namespace optix
