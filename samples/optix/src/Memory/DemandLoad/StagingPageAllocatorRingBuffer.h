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

#include <Memory/DemandLoad/PinnedMemoryRingBuffer.h>
#include <Memory/DemandLoad/StagingPageAllocator.h>

#include <mutex>
#include <vector>

namespace optix {

namespace lwca {
class Stream;
}  // namespace lwca

class DeviceManager;
class DeviceSet;

class StagingPageAllocatorRingBuffer : public StagingPageAllocator
{
  public:
    /// Construct allocator.  No memory is allocated until initializeDeferred is called.
    StagingPageAllocatorRingBuffer( DeviceManager* dm, unsigned int maxNumPages, unsigned int pageSizeInBytes );
    ~StagingPageAllocatorRingBuffer() override;

    /// Get the page size in bytes.
    unsigned int getPageSizeInBytes() const override { return m_pageSizeInBytes; }

    /// Should only be called from the main thread.  Performs any deferred initialization
    /// by initializing the held PinnedMemoryRingBuffer.
    void initializeDeferred() override;

    /// The opposite of initializeDeferred.  Allows resources to be destroyed with
    /// exception handling outside the destructor.
    void tearDown() override;

    /// Allocate a page to accommodate the specified number of bytes.  Thread safe.  Lwrrently the page size is
    /// fixed; the size is provided for sanity checking.
    StagingPageAllocation acquirePage( size_t numBytes ) override;
    void recordEvent( lwca::Stream& stream, unsigned int allDeviceListIndex, const StagingPageAllocation& allocation ) override;
    void releasePage( const StagingPageAllocation& alloc ) override;

    /// Clear allocations.  Thread safe.  Does not reclaim memory.
    void clear() override;

    void removeActiveDevices( const DeviceSet& removedDevices ) override;
    void setActiveDevices( const DeviceSet& devices ) override;

  private:
    DeviceManager*         m_dm;
    unsigned int           m_maxNumPages;
    unsigned int           m_pageSizeInBytes;
    PinnedMemoryRingBuffer m_ringBuffer;
    bool                   m_ringBufferInitialized = false;
};

}  // namespace optix
