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

#include <memory>

namespace optix {

namespace lwca {
class Stream;
}  // namespace lwca

class DeviceManager;
class DeviceSet;

// The result of an allocation:
// - A request id which identifies this allocation.
// - The address of the allocation.
struct StagingPageAllocation
{
    unsigned int id;
    void*        address;
    size_t       size;
};

class StagingPageAllocator
{
  public:
    virtual ~StagingPageAllocator() = default;

    /// Get the page size in bytes.
    virtual unsigned int getPageSizeInBytes() const = 0;

    /// Should only be called from the main thread.  Performs any deferred initialization.
    virtual void initializeDeferred() = 0;

    /// The opposite of initializeDeferred.  Allows resources to be destroyed with
    /// exception handling outside the destructor.
    virtual void tearDown() = 0;

    /// Allocate a page to accommodate the specified number of bytes.  Thread safe.  Lwrrently the page size is
    /// fixed; the size is provided for sanity checking.
    virtual StagingPageAllocation acquirePage( size_t numBytes ) = 0;
    virtual void recordEvent( lwca::Stream& stream, unsigned int allDeviceListIndex, const StagingPageAllocation& allocation ) = 0;
    virtual void releasePage( const StagingPageAllocation& alloc ) = 0;

    /// Clear allocations.  Thread safe.  Does not reclaim memory.
    virtual void clear() = 0;

    virtual void removeActiveDevices( const DeviceSet& removedDevices ) = 0;
    virtual void setActiveDevices( const DeviceSet& devices )           = 0;
};

std::unique_ptr<StagingPageAllocator> createStagingPageAllocator( DeviceManager* dm,
                                                                  unsigned int   maxNumPages,
                                                                  unsigned int   pageSizeInBytes,
                                                                  bool           useRingBuffer );

// Returns the number of bytes for the electric fence.  The true allocation size in
// bytes is the requested size plus 2 times the electric fence size.  The allocation
// address should point to the user data after the electric fence bytes.
size_t getElectricFenceSize();

// Writes electric fence bytes before and after the user data area.
void writeElectricFence( const StagingPageAllocation& alloc );

// Returns true if the electric fence has been changed.
bool checkElectricFenceModified( const StagingPageAllocation& alloc );

}  // namespace optix
