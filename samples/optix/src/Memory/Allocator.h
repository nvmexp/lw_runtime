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
#include <memory>

namespace optix {

// The Allocator is used to allocate blocks from a fixed size address space.
// alloc() returns a Handle which contains the offset of the allocation.
// The allocation is freed when the Handle is destroyed or reset.
//
// Internally, the allocator maintains two linked lists:
// 1) an ordered list of all existing blocks
// 2) an unordered list of free blocks
//
// Blocks are defined by an offset and a size. If a block is free, it is
// contained in both lists.
//
// The alloc() function iterates through the list of free blocks to find one
// that is large enough. If the found block is too large, it is split. The
// function returns a handle that calls free() when destroyed. This does not
// destroy the Block object, it only adds it to the free list.
//
// When a block is freed, we check if its predecessor and/or successor is also
// free, in which case we merge the blocks. This is the only situation where
// a Block object is actually deleted (other than tearing down the whole
// allocator).
//
// NOTE: The freePrev pointer is lwrrently not used since we only do forward
//       iteration in alloc().
class Allocator
{
  public:
    typedef std::shared_ptr<size_t> Handle;

    // size - specifies the total size to be allocated
    // alignment - default alignment for allocations. Must be a power-of-two.
    Allocator( size_t size, size_t alignment = 1 );
    ~Allocator();

    // Allocate a block. Returns a handle. The allocation will fail if there is not
    // enough free space to make the allocation. If the succeeded parameter is provided
    // then it will be set to false when the allocation fails. Otherwise an exception
    // is thrown. The Handle is reference counted. When the reference count falls to
    // 0 the block is freed.
    Handle alloc( size_t size, /*out*/ bool* succeeded = nullptr );

    // Set a custom callback that is to be called when a block is freed.
    typedef std::function<void( size_t size, size_t offset )> FreeBlockCallback;
    void setFreeBlockCallback( FreeBlockCallback callback );

    // Increase the size of the allocation pool. Must be larger than the current size.
    void expand( size_t newSize );

    // Returns the total amount of free space
    size_t freeSpace() const;

    // Returns the total size
    size_t size() const;

    // Returns one past the largest address that is in use
    size_t getUsedAddressRangeEnd() const;

    // Returns true if the free space is the same as the size
    bool empty() const;

    // Returns true if the internal block structure is valid.
    bool validate();

  protected:
    struct Block
    {
        size_t size     = 0;
        size_t offset   = 0;
        Block* prev     = nullptr;  // previous block in the block list
        Block* next     = nullptr;  // next     block in the block list
        Block* freePrev = nullptr;  // previous block in the free list
        Block* freeNext = nullptr;  // next     block in the free list

        Block( size_t size = 0, size_t offset = 0 );

        bool isFree() const;

        // Insert this block after the given block
        void insertAfter( Block* block );

        // Insert into the free list after the given block
        void freeInsertAfter( Block* block );

        // Unlink this block from the block list
        void unlink();

        // Unlink this block from the free list
        void freeUnlink();

        // Split this block and return the new block
        Block* split( size_t splitSize );

        // Merge with next block
        void mergeNext();
        // Merge with prev block
        void mergePrev();
    };

    size_t m_size      = 0;
    size_t m_alignment = 0;
    size_t m_free      = 0;

    // A list of all the blocks in order
    Block m_head;
    Block m_tail;

    // A list of all the free blocks (not in order)
    Block m_freeHead;
    Block m_freeTail;

    FreeBlockCallback m_callback;

    // Returns a block to the free list
    void free( Block* block );
};


}  // namespace optix
