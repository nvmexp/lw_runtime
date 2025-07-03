// Copyright (c) 2018, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

#include <Memory/LightweightAllocator.h>

#include <lwca.h>

#include <memory>
#include <mutex>
#include <vector>

namespace optix {

class Context;
class Device;
class DeviceManager;
class LWDADevice;

// The BulkMemoryPool is an extension to the LightweightAllocator, as it organizes multiple
// allocators into a single pool of GPU memory.
//
// Each entry into the pool is an allocator that allows the pool to grow on demand. It is
// expected that only a few entries exist per pool.
//
// Allocation requests to the Pool are all with the same preset alignment. This is an implementation
// detail to keep alignment requirements but keep tracked information to a minimum.
//
class BulkMemoryPool
{
  public:
    // Initialize the pool.
    // context        - the associated context
    // allDeviceIndex - GPU where the memory is allocated
    // alignment      - default alignment for all allocations in the pool
    void initialize( Context* context, int allDeviceIndex, size_t alignment );

    // Query the initialization status of the pool.
    bool isInitialized() { return ( m_context != nullptr ) && ( m_allDeviceIndex != ~0U ); }

    // Clear all entries in the pool. Frees all memory.
    void clear();

    // Allocate memory from the pool.
    // nbytes - Amount of bytes to allocate (will be aligned to the alignment of the pool)
    LWdeviceptr allocate( size_t nbytes, bool deviceIsHost );

    // Release memory back to the pool.
    // ptr    - Pointer to the allocated block
    // nbytes - Size of the allocated  block being released.
    void free( LWdeviceptr ptr, size_t nbytes, bool deviceIsHost );

  protected:
    // Utility functions

    // Return the associated device
    Device* getDevice( unsigned int allDeviceIndex ) const;

    // Return the associated device as a LWCA device
    LWDADevice* getLWDADevice( unsigned int allDeviceIndex ) const;

  private:
    // Represent each memory pool entry
    struct BulkMemoryPoolEntry
    {
        LWdeviceptr                           basePointer;   // Base address
        std::unique_ptr<LightweightAllocator> allocator;     // Allocator responsible of allocations in entry
        bool                                  deviceIsHost;  // Host or GPU memory
    };

    std::vector<BulkMemoryPoolEntry> m_entries;        // List of Pool entries
    std::mutex                       m_poolMutex;      // Mutex to ensure proper handling in multithreaded code
    size_t                           m_alignment = 0;  // Alignment associated with the Pool

    Context*     m_context        = nullptr;  // Associated OptiX context
    unsigned int m_allDeviceIndex = ~0U;      // Associated device
};
};
