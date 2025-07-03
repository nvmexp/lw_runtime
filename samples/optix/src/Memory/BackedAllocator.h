// Copyright (c) 2017, LWPU CORPORATION.
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

#include <Memory/MBuffer.h>

#include <memory>


namespace optix {

class Allocator;
class MemoryManager;
class ObjectManager;

class BackedAllocator
{
  public:
    typedef std::shared_ptr<size_t> Handle;

    BackedAllocator( size_t size, size_t alignment, MBufferPolicy policy, MemoryManager* memoryManager );
    virtual ~BackedAllocator();

    // Returns an offset for the allocation of the given size. Calling memory()
    // will return the underlying MBuffer from which the base pointer can be
    // retrieved.
    // If the backingUnmapped pointer is not nullptr, the pointed bool is set to
    // true if the backing storage has been unmapped and reallocated to honor the
    // allocation request. In this case the client code is responsible for
    // remapping the backing buffer.
    Handle alloc( size_t size, bool* backingUnmapped = nullptr );

    // Returns the total size in bytes of the underlying MBuffer
    size_t memorySize() const;

    // Returns one past the largest address that is in use
    size_t getUsedAddressRangeEnd() const;

    // Returns the managed memory object
    MBufferHandle memory();

  private:
    MBufferPolicy              m_allocationPolicy;
    MemoryManager*             m_memoryManager;
    std::unique_ptr<Allocator> m_allocator;
    MBufferHandle              m_memory;

    // Increase the size of the backing store using size as a guide. m_memory must
    // not be mapped.
    void growMemory( size_t size );
};

}  // namespace optix
