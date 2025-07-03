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

#include <Memory/DemandLoad/RequestHandler/MipLevelRequestHandler.h>

#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Memory/DemandLoad/StagingPageAllocator.h>
#include <Objects/Buffer.h>

namespace optix {

MipLevelRequestHandler::MipLevelRequestHandler( unsigned int          mipLevel,
                                                unsigned int          pageId,
                                                DeviceSet             devices,
                                                const Buffer*         buffer,
                                                unsigned int          startPage,
                                                bool                  useSimpleAllocationStrategy )
    : RequestHandler( pageId, devices, buffer, startPage )
    , m_mipLevel( mipLevel )
    , m_useSimpleAllocationStrategy( useSimpleAllocationStrategy )
{
}

// Callwlate memory block for a miplevel in the "mip tail".
void MipLevelRequestHandler::createMemoryBlock( StagingPageAllocator* stagingPages )
{
    LOG_MEDIUM_VERBOSE( "MipLevelRequestHandler::createMemoryBlock page " << m_pageId << '\n' );
    const BufferDimensions dimensions = m_buffer->getDimensions();
    RTmemoryblock          block;
    block.format     = m_buffer->getFormat();
    block.mipLevel   = m_mipLevel;
    block.x          = 0;
    block.y          = 0;
    block.z          = 0;
    block.width      = dimensions.levelWidth( block.mipLevel );
    block.height     = dimensions.levelHeight( block.mipLevel );
    block.depth      = 1;
    block.rowPitch   = block.width * dimensions.elementSize();
    block.planePitch = 0;

    // Allocate host-side staging memory for the callback to fill in.
    const unsigned int size = block.width * block.height * block.depth * dimensions.elementSize();
    // Get memory from the StagingPageAllocator.
    if( m_useSimpleAllocationStrategy )
    {
        if( size <= stagingPages->getPageSizeInBytes() )
        {
            // Get a page from the StagingPageAllocator.
            m_allocation      = stagingPages->acquirePage( size );
            block.baseAddress = m_allocation.address;
        }
        else
        {
            // More than one page required.  This oclwrs only when filling whole miplevels outside the
            // mip tail.  The allocated storage is owned by this request handler.
            m_allocation = StagingPageAllocation{};
            m_stagingMemory.resize( size );
            block.baseAddress = m_stagingMemory.data();
        }
    }
    else
    {
        m_allocation      = stagingPages->acquirePage( size );
        block.baseAddress = m_allocation.address;
    }

    m_memoryBlock = block;
}

void MipLevelRequestHandler::releaseMemoryBlock( StagingPageAllocator* stagingPages )
{
    LOG_MEDIUM_VERBOSE( "MipLevelRequestHandler::releaseMemoryBlock page " << m_pageId << '\n' );
    if( !m_stagingMemory.empty() )
    {
        // We didn't get the staging memory from the StagingPageAllocator.
        m_stagingMemory.clear();
    }
    else
    {
        RequestHandler::releaseMemoryBlock( stagingPages );
    }
}

void MipLevelRequestHandler::synchronize( StagingPageAllocator* stagingPages,
                                          unsigned int          allDeviceListIndex,
                                          DevicePaging&         devicePaging,
                                          bool                  synchronous ) const
{
    // Do nothing if this page wasn't requested on the specified device.
    if( !m_devices.isSet( allDeviceListIndex ) )
        return;

    // We might not have ilwoked the callback (if, for example, we ran out of staging memory).
    if( !m_isFilled )
        return;

    LOG_MEDIUM_VERBOSE( "MipLevelRequestHandler::synchronize device " << allDeviceListIndex << '\n' );

    unsigned int numBytes = m_memoryBlock.width * m_memoryBlock.height * m_memoryBlock.depth * m_buffer->getElementSize();
    if( synchronous )
    {
        devicePaging.syncDemandLoadMipLevel( stagingPages, m_allocation, m_buffer, m_memoryBlock.baseAddress, numBytes,
                                             static_cast<int>( m_memoryBlock.mipLevel ) );
    }
    else
    {
        devicePaging.syncDemandLoadMipLevelAsync( stagingPages, m_allocation, m_buffer, m_memoryBlock.baseAddress,
                                                  numBytes, static_cast<int>( m_memoryBlock.mipLevel ) );
    }

    // Push page mapping.  It's redundant (but harmless) to do this for each level in the mip tail.
    // It's necessary when whole miplevel callbacks are enabled, in which case each miplevel has a
    // different page id.
    devicePaging.addPageMapping( PageMapping{m_pageId, 1} );
}
}  // namespace optix
