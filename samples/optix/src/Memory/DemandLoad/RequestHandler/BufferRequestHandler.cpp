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

#include <Memory/DemandLoad/RequestHandler/BufferRequestHandler.h>

#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Memory/DemandLoad/StagingPageAllocator.h>
#include <Objects/Buffer.h>

#include <corelib/misc/String.h>

#include <cmath>
#include <ios>
#include <sstream>
#include <string>

using namespace prodlib;
using namespace optix::demandLoad;

namespace optix {

inline unsigned long long ptrToPrintable( void* ptr )
{
    return reinterpret_cast<unsigned long long>( ptr );
}

inline std::string toStringFromPtr( void* ptr )
{
    std::ostringstream str;
    str << "0x" << std::hex << ptrToPrintable( ptr );
    return str.str();
}

BufferRequestHandler::BufferRequestHandler( unsigned int pageId, DeviceSet devices, const Buffer* buffer, unsigned int startPage )
    : RequestHandler( pageId, devices, buffer, startPage )
{
    RT_ASSERT_MSG( m_buffer != nullptr, "No demand buffer associated with requested page" );
    RT_ASSERT( m_buffer->isDemandLoad() );
}

static unsigned int numPagesPerSide( size_t bufferDimension, unsigned int pageDimension )
{
    return static_cast<unsigned int>( std::ceil( float( bufferDimension ) / float( pageDimension ) ) );
}

static unsigned int blockDimension( bool isLastPage, size_t bufferDimension, unsigned int pageDimension )
{
    if( isLastPage )
    {
        const unsigned int remainder = static_cast<unsigned int>( bufferDimension ) % pageDimension;
        return remainder == 0 ? pageDimension : remainder;
    }
    return pageDimension;
}

// Callwlate memory block for a page in a demand-loaded buffer or texture (other than the mip tail).
void BufferRequestHandler::createMemoryBlock( StagingPageAllocator* stagingPages )
{
    LOG_MEDIUM_VERBOSE( "BufferPageRequestHandler::createMemoryBlock page " << m_pageId << '\n' );

    RTmemoryblock block;
    block.format = m_buffer->getFormat();

    const unsigned int pageWidth     = m_buffer->getPageWidth();
    const unsigned int pageHeight    = m_buffer->getPageHeight();
    const unsigned int pageDepth     = m_buffer->getPageDepth();
    const unsigned int dimensions    = pageDepth == 1 ? ( pageHeight == 1 ? 1 : 2 ) : 3;
    const unsigned int widthInPages  = numPagesPerSide( m_buffer->getWidth(), pageWidth );
    const unsigned int heightInPages = numPagesPerSide( m_buffer->getHeight(), pageHeight );
    const unsigned int depthInPages  = numPagesPerSide( m_buffer->getDepth(), pageDepth );
    const unsigned int pageIndex     = getPageIndex();
    const unsigned int pageX         = pageIndex % widthInPages;
    const unsigned int pageY         = dimensions > 1 ? pageIndex / widthInPages : 0;
    const unsigned int pageZ         = dimensions > 2 ? pageIndex / ( widthInPages * heightInPages ) : 0;
    const size_t       elementSize   = m_buffer->getElementSize();  // in bytes.
    const bool isLastPage = ( pageX == widthInPages - 1 ) && ( pageY == heightInPages - 1 ) && ( pageZ == depthInPages - 1 );

    block.mipLevel   = 0;
    block.x          = pageX * pageWidth;
    block.y          = pageY * pageHeight;
    block.z          = pageZ * pageDepth;
    block.width      = blockDimension( isLastPage, m_buffer->getWidth(), pageWidth );
    block.height     = blockDimension( isLastPage, m_buffer->getHeight(), pageHeight );
    block.depth      = blockDimension( isLastPage, m_buffer->getDepth(), pageDepth );
    block.rowPitch   = dimensions > 1 ? pageWidth * elementSize : 0;  // in bytes
    block.planePitch = dimensions > 2 ? pageWidth * pageHeight * elementSize : 0;

    // Allocate host-side staging memory for the callback to fill in.
    const unsigned int size = block.width * block.height * block.depth * m_buffer->getElementSize();
    m_allocation            = stagingPages->acquirePage( size );
    block.baseAddress       = m_allocation.address;
    m_memoryBlock           = block;
}

void BufferRequestHandler::synchronize( StagingPageAllocator* stagingPages,
                                        unsigned int          allDeviceListIndex,
                                        DevicePaging&         devicePaging,
                                        bool                  synchronous ) const
{
    // Do nothing if this page wasn't requested on the specified device.
    if( !m_devices.isSet( allDeviceListIndex ) )
        return;

    // We might not have ilwoked the callback (if, for example, we ran out of staging memory).
    if( m_isFilled )
    {
        LOG_MEDIUM_VERBOSE( "BufferPageRequestHandler::synchronize page " << m_pageId << ", device " << allDeviceListIndex << '\n' );

        // TODO: optimize copying of partially filled pages.
        const LWdeviceptr devPage =
            devicePaging.copyPageToDevice( stagingPages, m_allocation, m_memoryBlock.baseAddress, synchronous );
        devicePaging.addPageMapping( PageMapping{m_pageId, devPage} );
    }
}

}  // namespace optix
