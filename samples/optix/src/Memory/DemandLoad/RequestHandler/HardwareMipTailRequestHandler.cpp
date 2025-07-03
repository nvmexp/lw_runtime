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

#include <Memory/DemandLoad/RequestHandler/HardwareMipTailRequestHandler.h>

#include <LWCA/ErrorCheck.h>
#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Memory/DemandLoad/RequestHandler/BlockStamper.h>
#include <Memory/DemandLoad/StagingPageAllocator.h>
#include <Objects/Buffer.h>

#include <corelib/misc/String.h>

using namespace optix::demandLoad;

namespace optix {

static RTbuffer api_cast( const Buffer* ptr )
{
    return reinterpret_cast<RTbuffer>( const_cast<Buffer*>( ptr ) );
}

// Page 0 of a demand-loaded texture represents the "mip tail", which
// contains all the miplevels that can fit in a single tile.
bool HardwareMipTailRequestHandler::isMipTailRequest( unsigned int pageId, unsigned int startPage, const Buffer* buffer )
{
    unsigned int pageIndex = pageId - startPage;
    if( pageIndex > 0 || !buffer->hasTextureAttached() )
        return false;

    // Guard against non-mipmapped texture.
    LWDA_ARRAY_SPARSE_PROPERTIES textureProps = buffer->getMBuffer()->getSparseTextureProperties();
    return textureProps.miptailFirstLevel < buffer->getMipLevelCount();
}

bool HardwareMipTailRequestHandler::isSmallNonMipmapped( unsigned int pageId, unsigned int startPage, const Buffer* buffer )
{
    return buffer->getMipLevelCount() == 1 && isMipTailRequest( pageId, startPage, buffer );
}

HardwareMipTailRequestHandler::HardwareMipTailRequestHandler( unsigned int  pageId,
                                                              DeviceSet     devices,
                                                              const Buffer* buffer,
                                                              unsigned int  startPage,
                                                              bool          useSimpleAllocationStrategy )
    : RequestHandler( pageId, devices, buffer, startPage )
    , m_useSimpleAllocationStrategy( useSimpleAllocationStrategy )
{
    // Get sparse texture properties, which are guaranteed to be the same on all devices
    // (m_devices[0] is the all-device-index of the first device that requested the tile).
    LWDA_ARRAY_SPARSE_PROPERTIES textureProps = m_buffer->getMBuffer()->getSparseTextureProperties( m_devices[0] );
    m_mipTailLength                           = m_buffer->getMipLevelCount() - textureProps.miptailFirstLevel;
    m_mipTailSizeInBytes                      = textureProps.miptailSize;

    // We know that, at most, every mip level will fit in the mip tail.
    m_mipLevelMemoryBlocks.resize( m_mipTailLength );

    m_allocations.resize( m_mipTailLength );
    // If we're using the simple allocation strategy, we might need to manage our own allocations.
    if( m_useSimpleAllocationStrategy )
        m_stagingMemory.resize( m_mipTailLength );
}

void HardwareMipTailRequestHandler::createMemoryBlock( StagingPageAllocator* stagingPages )
{
    LOG_MEDIUM_VERBOSE( "HardwareMipTailRequestHandler::createMemoryBlock page " << m_pageId << '\n' );

    // For now we support only 2D demand-loaded textures, and tiles are always square.
    // (The tile size is controlled by a knob; it is not yet configurable via the OptiX API.)
    RT_ASSERT( m_buffer->getDimensionality() == 2 );

    // Fill in fields for block describing entire region.
    for( int i = 0; i < m_mipTailLength; ++i )
    {
        const unsigned int lwrrMipLevel = m_buffer->getMipLevelCount() - m_mipTailLength + i;
        RTmemoryblock&     lwrrBlock    = m_mipLevelMemoryBlocks[i];

        lwrrBlock.x          = 0;
        lwrrBlock.y          = 0;
        lwrrBlock.z          = 0;
        lwrrBlock.width      = m_buffer->getLevelWidth( lwrrMipLevel );
        lwrrBlock.height     = m_buffer->getLevelHeight( lwrrMipLevel );
        lwrrBlock.depth      = 1;
        lwrrBlock.rowPitch   = lwrrBlock.width * m_buffer->getElementSize();  // in bytes.
        lwrrBlock.planePitch = 0;
        lwrrBlock.format     = m_buffer->getFormat();
        lwrrBlock.mipLevel   = lwrrMipLevel;

        const unsigned int lwrrBlockSize = lwrrBlock.width * lwrrBlock.height * m_buffer->getElementSize();

        if( m_useSimpleAllocationStrategy )
        {
            if( lwrrBlockSize <= stagingPages->getPageSizeInBytes() )
            {
                m_allocations[i]      = stagingPages->acquirePage( lwrrBlockSize );
                lwrrBlock.baseAddress = m_allocations[i].address;
            }
            else
            {
                m_allocations[i] = StagingPageAllocation{};
                m_stagingMemory[i].resize( lwrrBlockSize );
                lwrrBlock.baseAddress = m_stagingMemory[i].data();
            }
        }
        else
        {
            m_allocations[i]      = stagingPages->acquirePage( lwrrBlockSize );
            lwrrBlock.baseAddress = m_allocations[i].address;
        }
    }
}

void HardwareMipTailRequestHandler::releaseMemoryBlock( StagingPageAllocator* stagingPages )
{
    LOG_MEDIUM_VERBOSE( "HardwareMipTailRequestHandler::releaseMemoryBlock page " << m_pageId << '\n' );

    for( int i = 0; i < m_mipTailLength; ++i )
    {
        if( m_allocations[i].address != nullptr )
            stagingPages->releasePage( m_allocations[i] );
    }

    if( m_useSimpleAllocationStrategy )
    {
        for( int i = 0; i < m_mipTailLength; ++i )
        {
            if( !m_stagingMemory[i].empty() )
                m_stagingMemory[i].clear();
        }
    }
}

// Ilwoke the callback for each miplevel in the mip tail.
void HardwareMipTailRequestHandler::fillMemoryBlock( StagingPageAllocator* stagingPages )
{
    createMemoryBlock( stagingPages );

    // We want to consider the mip tail to have been filled, only if all the levels
    // were successfully filled because we can't use a partially filled miptail.
    m_isFilled = true;
    for( unsigned int i = 0; m_isFilled && i < m_mipTailLength; ++i )
    {
        if( m_mipLevelMemoryBlocks[i].baseAddress != nullptr )
        {
            fillMemoryBlockDebugPattern( m_mipLevelMemoryBlocks[i] );
            ilwokeCallback( m_mipLevelMemoryBlocks[i] );
            logCallback( "HardwareMipTailRequestHandler", m_mipLevelMemoryBlocks[i] );
            checkMemoryBlockDebugPattern( m_mipLevelMemoryBlocks[i] );
            stampMemoryBlock( m_mipLevelMemoryBlocks[i], m_allocations[i].id );
        }
        else
        {
            m_isFilled = false;
        }
    }
}

void HardwareMipTailRequestHandler::synchronize( StagingPageAllocator* stagingPages,
                                                 unsigned int          allDeviceListIndex,
                                                 DevicePaging&         devicePaging,
                                                 bool                  synchronous ) const
{
    // Do nothing if this page wasn't requested on the specified device.
    if( !m_devices.isSet( static_cast<int>( allDeviceListIndex ) ) )
        return;

    // We might not have ilwoked the callback (if, for example, we ran out of staging memory).
    if( m_isFilled )
    {
        LOG_MEDIUM_VERBOSE( "HardwareMipTailRequestHandler::synchronize page " << m_pageId << ", device "
                                                                               << allDeviceListIndex << '\n' );

        if( synchronous )
        {
            devicePaging.bindMipTailToMemory( m_buffer, m_mipTailSizeInBytes );
        }
        else
        {
            devicePaging.bindMipTailToMemoryAsync( m_buffer, m_mipTailSizeInBytes );
        }
        for( unsigned int i = 0; i < m_mipTailLength; ++i )
        {
            if( synchronous )
            {
                devicePaging.fillHardwareMipTail( stagingPages, m_allocations[i], m_buffer, m_mipLevelMemoryBlocks[i] );
            }
            else
            {
                devicePaging.fillHardwareMipTailAsync( stagingPages, m_allocations[i], m_buffer, m_mipLevelMemoryBlocks[i] );
            }
        }
        devicePaging.addPageMapping( PageMapping{m_pageId, 1} );
    }
}

}  // namespace optix
