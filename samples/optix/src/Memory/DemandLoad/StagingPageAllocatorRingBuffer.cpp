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

#include <Memory/DemandLoad/StagingPageAllocatorRingBuffer.h>

#include <LWCA/Stream.h>
#include <Device/DeviceManager.h>
#include <Memory/DemandLoad/PagingServiceLogging.h>

#include <prodlib/exceptions/Assert.h>

namespace optix {

StagingPageAllocatorRingBuffer::StagingPageAllocatorRingBuffer( DeviceManager* dm, unsigned int maxNumPages, unsigned int pageSizeInBytes )
    : m_dm( dm )
    , m_maxNumPages( maxNumPages )
    , m_pageSizeInBytes( pageSizeInBytes + 2U * getElectricFenceSize() )
{
}

StagingPageAllocatorRingBuffer::~StagingPageAllocatorRingBuffer()
{
    StagingPageAllocatorRingBuffer::tearDown();
}

void StagingPageAllocatorRingBuffer::initializeDeferred()
{
    if( !m_ringBufferInitialized )
    {
        LOG_MEDIUM_VERBOSE( "StagingPageAllocatorRingBuffer::initializeDeferred maxNumPages "
                            << m_maxNumPages << ", pageSize " << m_pageSizeInBytes << '\n' );

        const size_t       ringSizeInBytes = m_maxNumPages * m_pageSizeInBytes;
        const unsigned int numEvents       = 1024;  // 1K events per device
        const unsigned int numRequests     = m_maxNumPages;
        m_ringBuffer.init( m_dm, ringSizeInBytes, numEvents, numRequests );
        m_ringBufferInitialized = true;
    }
}

void StagingPageAllocatorRingBuffer::tearDown()
{
    LOG_MEDIUM_VERBOSE( "StagingPageAllocatorRingBuffer::tearDown\n" );

    m_ringBuffer.destroy();
}

StagingPageAllocation StagingPageAllocatorRingBuffer::acquirePage( size_t numBytes )
{
    const StagingPageAllocation rawAllocation = m_ringBuffer.acquireResource( numBytes + 2U * getElectricFenceSize() );
    unsigned char* const userAllocationStart = static_cast<unsigned char*>( rawAllocation.address ) + getElectricFenceSize();
    const StagingPageAllocation allocation{rawAllocation.id, userAllocationStart, numBytes};
    LOG_MEDIUM_VERBOSE( "StagingPageAllocatorRingBuffer::acquirePage " << numBytes << " => " << allocation.id << '\n' );
    writeElectricFence( allocation );
    return allocation;
}

void StagingPageAllocatorRingBuffer::recordEvent( lwca::Stream& stream, unsigned int allDeviceListIndex, const StagingPageAllocation& allocation )
{
    LOG_MEDIUM_VERBOSE( "StagingPageAllocatorRingBuffer::recordEvent device " << allDeviceListIndex << ", allocation "
                                                                              << allocation.id << '\n' );

    m_ringBuffer.recordEvent( stream, allDeviceListIndex, allocation );
}

void StagingPageAllocatorRingBuffer::releasePage( const StagingPageAllocation& alloc )
{
    LOG_MEDIUM_VERBOSE( "StagingPageAllocatorRingBuffer::releasePage allocation " << alloc.id << '\n' );
    RT_ASSERT_MSG( !checkElectricFenceModified( alloc ), "Buffer overrun/underrun detected" );
    const StagingPageAllocation rawAlloc{alloc.id, static_cast<unsigned char*>( alloc.address ) - getElectricFenceSize(),
                                         alloc.size + 2U * getElectricFenceSize()};
    m_ringBuffer.releaseResource( rawAlloc );
}

void StagingPageAllocatorRingBuffer::clear()
{
    LOG_MEDIUM_VERBOSE( "StagingPageAllocatorRingBuffer::clear\n" );

    m_ringBuffer.clear();
}

void StagingPageAllocatorRingBuffer::removeActiveDevices( const DeviceSet& removedDevices )
{
    LOG_MEDIUM_VERBOSE( "StagingPageAllocatorRingBuffer::removeActiveDevices " << removedDevices.toString() << '\n' );

    m_ringBuffer.removeActiveDevices( m_dm->allDevices(), removedDevices );
}

void StagingPageAllocatorRingBuffer::setActiveDevices( const DeviceSet& devices )
{
    LOG_MEDIUM_VERBOSE( "StagingPageAllocatorRingBuffer::setActiveDevices " << devices.toString() << '\n' );

    m_ringBuffer.setActiveDevices( m_dm->allDevices(), devices );
}

}  // namespace optix
