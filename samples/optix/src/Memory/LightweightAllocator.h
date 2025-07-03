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

#include <functional>
#include <map>
#include <memory>

//#define LWA_UNIT_TEST

namespace optix {

// The LightweightAllocator is used to allocate blocks from a fixed
// size address space with minimal tracking.
//
// Internally, the allocator maintains one linked list:
// 1) an ordered list of free blocks
//
// Blocks are defined by an offset and a size. If a block is
// allocated, it is not tracked.
//
// The alloc() function iterates through the list of free blocks to
// find one that is large enough. If the found block is too large, it
// is split. The block is destroyed if no space remains.
//
// When a block is freed, we check if its predecessor and/or successor is also
// free, in which case we merge the blocks.
//
// NOTE: The freePrev pointer is lwrrently not used since we only do forward
//       iteration in alloc().
class LightweightAllocator
{
  public:
    // size - specifies the total size to be allocated.
    // alignment - default alignment for allocations. Must be a power-of-two.
    LightweightAllocator( size_t poolSize );
    ~LightweightAllocator();

    // Allocate a block. The allocation will return ~0U if there is not
    // enough free space to make the allocation.
    // On success, returns the offset to the allocated memory.
    // On failure, return BAD_ADDR
    size_t alloc( size_t size, size_t alignment );

    // Free a block. The size must be correct to ensure correctness.
    void free( size_t ptr, size_t size, size_t alignment );

    // Returns the total amount of free space
    size_t freeSpace() const;

    // Returns the total size
    size_t size() const;

    // Returns true if the free space is the same as the size
    bool empty() const;

    // Returns true if the internal block structure is valid.
    bool validate();

    static const size_t BAD_ADDR = ~0;

  protected:
    size_t m_size           = 0;
    size_t m_free           = 0;
    size_t m_gteLargestFree = 0;  // This is >= (gte) size of largest free block

    std::map<size_t, size_t> m_beginMap;
    std::map<size_t, size_t> m_endMap;

    void addBlock( size_t begin, size_t end );
    void delBlock( size_t begin, size_t end );
    void changeBlock( size_t oldBegin, size_t oldEnd, size_t begin, size_t end );

#ifdef LWA_UNIT_TEST
    void test();
#endif  // LWA_UNIT_TEST
};


}  // namespace optix
