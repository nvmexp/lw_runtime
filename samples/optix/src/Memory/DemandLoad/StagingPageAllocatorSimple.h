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

#pragma once

#include <Memory/DemandLoad/StagingPageAllocator.h>

#include <mutex>
#include <vector>

namespace optix {

class StagingPageAllocatorSimple : public StagingPageAllocator
{
  public:
    /// Construct allocator.  No memory is allocated until the first request.
    StagingPageAllocatorSimple( unsigned int maxNumPages, unsigned int pageSizeInBytes );
    ~StagingPageAllocatorSimple() override = default;

    /// Get the page size in bytes.
    unsigned int getPageSizeInBytes() const override { return m_pageSizeInBytes; }

    void initializeDeferred() override;
    void tearDown() override;

    StagingPageAllocation acquirePage( size_t numBytes ) override;

    /// Clear allocations.  Thread safe.  Does not reclaim memory.
    void clear() override;

    void recordEvent( lwca::Stream& stream, unsigned int allDeviceListIndex, const StagingPageAllocation& allocation ) override;
    void releasePage( const StagingPageAllocation& alloc ) override;
    void removeActiveDevices( const DeviceSet& removedDevices ) override;
    void setActiveDevices( const DeviceSet& devices ) override;

  private:
    std::mutex           m_mutex;
    std::vector<uint8_t> m_pageData;
    unsigned int         m_numPages        = 0;
    unsigned int         m_maxNumPages     = 0;
    unsigned int         m_pageSizeInBytes = 0;
    unsigned int         m_id              = 0;
};

}  // namespace optix
