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

#include <Memory/DemandLoad/StagingPageAllocatorSimple.h>

#include <Device/DeviceSet.h>
#include <Memory/DemandLoad/PagingServiceLogging.h>

#include <prodlib/exceptions/Assert.h>

namespace optix {

StagingPageAllocatorSimple::StagingPageAllocatorSimple( unsigned int maxNumPages, unsigned int pageSizeInBytes )
    : m_maxNumPages( maxNumPages )
    , m_pageSizeInBytes( pageSizeInBytes + 2U * getElectricFenceSize() )
{
}

void StagingPageAllocatorSimple::initializeDeferred()
{
    // Staging pages are lazily allocated.
    if( m_pageData.empty() )
        m_pageData.resize( m_maxNumPages * m_pageSizeInBytes );
}

void StagingPageAllocatorSimple::tearDown()
{
    clear();
}

StagingPageAllocation StagingPageAllocatorSimple::acquirePage( size_t numBytes )
{
    std::lock_guard<std::mutex> guard( m_mutex );

    // The page size is fixed; the size is provided for sanity checking.
    RT_ASSERT( numBytes > 0 && numBytes <= ( m_pageSizeInBytes - 2U * getElectricFenceSize() ) );

    // Take the next page.
    uint8_t* result = m_pageData.data() + m_numPages * m_pageSizeInBytes;
    ++m_numPages;

    // Return null if exhausted.
    StagingPageAllocation alloc{};
    alloc.id      = m_id++;
    alloc.address = m_numPages > m_maxNumPages ? nullptr : result + getElectricFenceSize();
    alloc.size    = numBytes;
    LOG_MEDIUM_VERBOSE( "StagingPageAllocatorSimple::acquirePage " << numBytes << " => " << alloc.id << '\n' );
    writeElectricFence( alloc );
    return alloc;
}

void StagingPageAllocatorSimple::clear()
{
    LOG_MEDIUM_VERBOSE( "StagingPageAllocatorSimple::clear\n" );
    std::lock_guard<std::mutex> guard( m_mutex );
    m_numPages = 0;
}

void StagingPageAllocatorSimple::recordEvent( lwca::Stream& stream, unsigned int allDeviceListIndex, const StagingPageAllocation& allocation )
{
    LOG_MEDIUM_VERBOSE( "StagingPageAllocatorSimple::recordEvent device " << allDeviceListIndex << ", allocation "
                                                                          << allocation.id << '\n' );
}

void StagingPageAllocatorSimple::releasePage( const StagingPageAllocation& alloc )
{
    LOG_MEDIUM_VERBOSE( "StagingPageAllocatorSimple::releasePage allocation " << alloc.id << '\n' );
    RT_ASSERT( alloc.id < m_id );
    RT_ASSERT_MSG( !checkElectricFenceModified( alloc ), "Buffer overrun/underrun detected" );
}

void StagingPageAllocatorSimple::removeActiveDevices( const DeviceSet& removedDevices )
{
    LOG_MEDIUM_VERBOSE( "StagingPageAllocatorSimple::removeActiveDevices " << removedDevices.toString() << '\n' );
}

void StagingPageAllocatorSimple::setActiveDevices( const DeviceSet& devices )
{
    LOG_MEDIUM_VERBOSE( "StagingPageAllocatorSimple::setActiveDevices " << devices.toString() << '\n' );
}

}  // namespace optix
