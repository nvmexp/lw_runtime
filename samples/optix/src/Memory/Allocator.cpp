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

#include <Memory/Allocator.h>

#include <corelib/math/MathUtil.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/MemoryAllocationFailed.h>
#include <prodlib/math/Bits.h>

#include <utility>

using namespace optix;
using namespace prodlib;
using namespace corelib;


Allocator::Allocator( size_t size, size_t alignment )
    : m_size( size )
    , m_alignment( alignment )
    , m_free( size )
{
    RT_ASSERT( isPow2( m_alignment ) );

    m_head.next         = &m_tail;
    m_tail.prev         = &m_head;
    m_freeHead.freeNext = &m_freeTail;
    m_freeTail.freePrev = &m_freeHead;

    Block* freeBlock = new Block( m_size, 0 );
    freeBlock->insertAfter( &m_head );
    freeBlock->freeInsertAfter( &m_freeHead );
}

Allocator::~Allocator()
{
    Block* lwr = m_head.next;
    while( lwr != &m_tail )
    {
        Block* next = lwr->next;
        delete lwr;
        lwr = next;
    }
}

Allocator::Handle Allocator::alloc( size_t size, bool* succeeded )
{
    if( size == 0 )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot allocate 0 bytes" );

    size = align( size, m_alignment );

    // Find a free block
    Block* lwr = m_freeHead.freeNext;
    while( lwr && size > lwr->size )
        lwr = lwr->freeNext;
    if( lwr == nullptr )
    {
        if( succeeded )
        {
            *succeeded = false;
            return std::shared_ptr<size_t>();
        }
        else
            throw MemoryAllocationFailed( RT_EXCEPTION_INFO, "Insufficient free space" );
    }

    Block* block = nullptr;
    if( lwr->size == size )
    {
        lwr->freeUnlink();
        block = lwr;
    }
    else
    {
        block = lwr->split( size );
    }

    m_free -= size;

    if( succeeded )
        *succeeded = true;

    // Create a shared pointer. Make sure free() is called when it is destroyed.
    return Handle( &block->offset, [=]( size_t* ) { this->free( block ); } );
}

void Allocator::setFreeBlockCallback( FreeBlockCallback callback )
{
    m_callback = std::move( callback );
}

void Allocator::free( Block* block )
{
    if( m_callback )
        m_callback( block->offset, block->size );

    block->freeInsertAfter( &m_freeHead );
    m_free += block->size;

    // coalesce
    if( block->prev->isFree() )
    {
        // now we could merge either block->prev with block or block with block->prev
        // To preserve the ability to keep this block at the front of the free list,
        // ie right behind m_freeHead, we choose to merge block->prev into block.
        // This allows an Allocator client to free blocks in a way that the next allocation
        // happens at offset 0, eg as required for allocating the GlobalScope's object record.
        block->mergePrev();
    }
    if( block->next->isFree() )
        block->mergeNext();
}

void Allocator::expand( size_t newSize )
{
    if( newSize <= m_size )
        throw MemoryAllocationFailed( RT_EXCEPTION_INFO, "New size must be larger than previous size" );

    Block* newBlock = new Block( newSize - m_size, m_size );
    newBlock->insertAfter( m_tail.prev );
    free( newBlock );
    m_size = newSize;
}

size_t Allocator::freeSpace() const
{
    return m_free;
}

size_t Allocator::size() const
{
    return m_size;
}

size_t Allocator::getUsedAddressRangeEnd() const
{
    // We exploit the fact that the block list is ordered and contains all blocks,
    // and that we coalesce free neighbors. This means that if the last block is
    // free, the largest used address is always its offset, since there can be no
    // additional free block in front of it. If the last block is not free, the
    // last used address is the offset of the block plus its size.
    const Block* last = m_tail.prev;
    if( last->isFree() )
        return last->offset;
    else
        return last->offset + last->size;
}

bool Allocator::empty() const
{
    return m_size == m_free;
}

bool Allocator::validate()
{
    Block* lwr    = m_head.next;
    size_t offset = 0;
    while( lwr != &m_tail )
    {
        if( lwr->offset != offset )
            return false;
        offset += lwr->size;
        lwr = lwr->next;
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Block
////////////////////////////////////////////////////////////////////////////////

Allocator::Block::Block( size_t size, size_t offset )
    : size( size )
    , offset( offset )
    , prev( nullptr )
    , next( nullptr )
    , freePrev( nullptr )
    , freeNext( nullptr )
{
}

bool Allocator::Block::isFree() const
{
    return freePrev != nullptr || freeNext != nullptr;
}

void Allocator::Block::insertAfter( Block* block )
{
    this->next        = block->next;
    this->prev        = block;
    block->next->prev = this;
    block->next       = this;
}

void Allocator::Block::freeInsertAfter( Block* block )
{
    this->freeNext            = block->freeNext;
    this->freePrev            = block;
    block->freeNext->freePrev = this;
    block->freeNext           = this;
}

void Allocator::Block::unlink()
{
    prev->next = next;
    next->prev = prev;
    next       = nullptr;
    prev       = nullptr;
}

void Allocator::Block::freeUnlink()
{
    freePrev->freeNext = freeNext;
    freeNext->freePrev = freePrev;
    freePrev           = nullptr;
    freeNext           = nullptr;
}

Allocator::Block* Allocator::Block::split( size_t splitSize )
{
    Block* newBlock = new Block( splitSize, offset );
    newBlock->insertAfter( prev );
    size -= splitSize;
    offset += splitSize;
    return newBlock;
}

void Allocator::Block::mergeNext()
{
    Block* old = next;
    size += old->size;
    old->unlink();
    old->freeUnlink();
    delete old;
}

void Allocator::Block::mergePrev()
{
    Block* old = prev;
    size += old->size;
    offset = old->offset;
    old->unlink();
    old->freeUnlink();
    delete old;
}

#if 0
// This is a draft of a replacement for freeInsertAfter() if we want to
// maintain an ordered list of free blocks.
void Allocator::Block::freeInsert( Block* freeHead, Block* freeTail )
{
  // Determine the block after which to insert.
  // If the predecessor or successor block is already free, we can directly
  // insert there. Otherwise, do an O(n) search through the list.
  Block* insertAfter = nullptr;
  if( prev->isFree() )
  {
    insertAfter = prev;
  }
  else if( next->isFree() )
  {
    insertAfter = next->freePrev;
  }
  else
  {
    // If the next would be freeTail (offset 0), we choose the last one before it.
    Block* lwr = freeHead;
    while( offset > lwr->offset && lwr != freeTail )
      lwr = lwr->freeNext;
    insertAfter = lwr->freePrev;
  }
  RT_ASSERT( insertAfter );

  RT_ASSERT_MSG( insertAfter == freeHead || offset != insertAfter->offset,
                 "must never have two blocks with the same offset" );

  // Insert after the chosen block.
  this->freeNext = insertAfter->freeNext;
  this->freePrev = insertAfter;
  insertAfter->freeNext->freePrev = this;
  insertAfter->freeNext = this;
}
#endif
