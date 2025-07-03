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

#include <Memory/DemandLoad/RequestHandler/MipTailRequestHandler.h>

#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Objects/Buffer.h>
#include <c-api/ApiCapture.h>

#include <mutex>

using namespace optix::demandLoad;

namespace optix {

MipTailRequestHandler::MipTailRequestHandler( unsigned int pageId, DeviceSet devices, const Buffer* buffer, unsigned int startPage, bool useSimpleAllocationStrategy )
    : RequestHandler( pageId, devices, buffer, startPage )
{
    // Create a request for each miplevel in the mip tail.
    const unsigned int totalMipLevels = m_buffer->getMipLevelCount();
    unsigned int       mipTailLength  = m_buffer->getMipLevelCount() - m_buffer->getMipTailFirstLevel();
    m_mipLevelRequests.reserve( mipTailLength );

    // Create a page request handler for each miplevel in the mip tail.
    const unsigned int firstMipLevel = totalMipLevels - mipTailLength;
    for( unsigned int mipLevel = firstMipLevel; mipLevel < totalMipLevels; ++mipLevel )
    {
        m_mipLevelRequests.emplace_back( mipLevel, pageId, devices, buffer, startPage, useSimpleAllocationStrategy );
    }
}

void MipTailRequestHandler::fillMemoryBlock( StagingPageAllocator* stagingPages )
{
    // We want to consider the mip tail to have been filled, only if all the levels
    // were successfully filled because we can't use a partially filled miptail.
    m_isFilled = true;
    // Ilwoke the callback for each miplevel in the mip tail.
    for( MipLevelRequestHandler& mipLevelRequest : m_mipLevelRequests )
    {
        mipLevelRequest.fillMemoryBlock( stagingPages );
        if( mipLevelRequest.caughtException() )
        {
            m_isFilled         = false;
            m_caughtException  = true;
            m_exceptionMessage = mipLevelRequest.getExceptionMessage();
        }
        m_isFilled = m_isFilled && mipLevelRequest.isFilled();
        if( !m_isFilled )
        {
            break;
        }
    }
}

bool MipTailRequestHandler::processRequest( DeviceManager*             dm,
                                            StagingPageAllocator*      stagingPages,
                                            std::vector<DevicePaging>& deviceState,
                                            bool                       synchronous )
{
    LOG_MEDIUM_VERBOSE( "MipTailRequestHandler::processRequest page " << m_pageId << '\n' );

    // No memoryBlock needs to be created, since we are delegating to other request handlers.

    // We need to serialize access to reallocMipTail, since we do this by updating the MAccess
    // on the MBuffer associated with the SDK Buffer.  Otherwise, there's a possibility of two
    // worker threads attempting to update the MAccess conlwrrently and the MBuffer can only
    // be locked by one thread at a time.
    {
        static std::mutex           s_mutex;
        std::lock_guard<std::mutex> lock( s_mutex );
        reallocMipTail( deviceState );
    }

    // Because we have to ilwoke processRequest on the contained requests, we can't just call
    // ilwokeCallback here.

    // We want to consider the mip tail to have been filled, only if all the levels
    // were successfully filled because we can't use a partially filled miptail.
    m_isFilled = true;
    // Ilwoke the callback for each miplevel in the mip tail.
    for( MipLevelRequestHandler& mipLevelRequest : m_mipLevelRequests )
    {
        m_isFilled = mipLevelRequest.processRequest( dm, stagingPages, deviceState, synchronous );
        if( mipLevelRequest.caughtException() )
        {
            m_isFilled         = false;
            m_caughtException  = true;
            m_exceptionMessage = mipLevelRequest.getExceptionMessage();
        }
        if( !m_isFilled )
        {
            break;
        }
    }

    if( m_isFilled )
    {
        copyDataToDevices( dm, stagingPages, deviceState, synchronous );
    }

    // No memoryBlock needs to be released, since we are delegating to other request handlers.

    return m_isFilled;
}

void MipTailRequestHandler::captureTrace( ApiCapture& capture ) const
{
    if( !capture.capture_enabled() )
        return;

    // Capture trace for each of the callbacks for the miplevels in the mip tail.
    for( const MipLevelRequestHandler& mipLevelRequest : m_mipLevelRequests )
    {
        mipLevelRequest.captureTrace( capture );
    }
}

void MipTailRequestHandler::reallocMipTail( std::vector<DevicePaging>& deviceState )
{
    // Allocate backing storage.
    for( unsigned int allDeviceListIndex : m_devices )
    {
        deviceState[allDeviceListIndex].reallocDemandLoadLwdaArray( m_buffer->getMBuffer(), m_buffer->getMipTailFirstLevel(),
                                                                    m_buffer->getMipLevelCount() - 1 );
    }
}

void MipTailRequestHandler::synchronize( StagingPageAllocator* stagingPages,
                                         unsigned int          allDeviceListIndex,
                                         DevicePaging&         devicePaging,
                                         bool                  synchronous ) const
{
    // Do nothing if this page wasn't requested on the specified device.
    if( !m_devices.isSet( static_cast<int>( allDeviceListIndex ) ) )
        return;

    // We might not have ilwoked the callback (if, for example, we ran out of staging memory).
    if( !m_isFilled )
        return;

    LOG_MEDIUM_VERBOSE( "MipTailRequestHandler::synchronize page " << m_pageId << ", device " << allDeviceListIndex << '\n' );

    // Synchronize each miplevel.
    for( const MipLevelRequestHandler& mipLevelRequest : m_mipLevelRequests )
    {
        mipLevelRequest.synchronize( stagingPages, allDeviceListIndex, devicePaging, synchronous );
    }

    // We add an arbitrary mapping to mark the mip tail as resident.  The value of the page table
    // entry is unused, since the texture object for the mip tail is accessed via
    // TextureSampler::DeviceDependent (see ExelwtionStrategy/CORTTypes.h)
    devicePaging.addPageMapping( PageMapping{m_pageId, 1} );
}

void MipTailRequestHandler::createMemoryBlock( StagingPageAllocator* /*stagingPages*/ )
{
    // All work is done by delegating to MipLevelRequestHandlers.
}

void MipTailRequestHandler::releaseMemoryBlock( StagingPageAllocator* /*stagingPages*/ )
{
    // All work is done by delegating to MipLevelRequestHandlers.
}

}  // namespace optix
