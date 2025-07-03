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

#include <Memory/BackedAllocator.h>

#include <Memory/Allocator.h>
#include <Memory/BufferDimensions.h>
#include <Memory/MemoryManager.h>

#include <corelib/misc/String.h>
#include <prodlib/exceptions/MemoryAllocationFailed.h>
#include <prodlib/system/Knobs.h>

#include <algorithm>
#include <cstring>

using namespace optix;
using namespace prodlib;

namespace {
// clang-format off
  Knob<bool>  k_clearBuffersOnAlloc( RT_DSTRING("mem.clearBuffersOnAllocation"), false, RT_DSTRING( "Clears a buffer to mem.clearValue during allocation."));
  Knob<int>   k_clearValue( RT_DSTRING("mem.clearValue"), 0, RT_DSTRING( "Value (byte) to clear with in mem.clearBuffersOnAllocation."));
// clang-format on
}


BackedAllocator::BackedAllocator( size_t size, size_t alignment, MBufferPolicy policy, MemoryManager* memoryManager )
    : m_allocationPolicy( policy )
    , m_memoryManager( memoryManager )
    , m_memory( nullptr )
{
    m_allocator.reset( new Allocator( size, alignment ) );
    if( k_clearBuffersOnAlloc.get() )
    {
        m_allocator->setFreeBlockCallback( [this]( size_t offset, size_t size ) {
            const bool alreadyMapped = m_memoryManager->isMappedToHost( m_memory );
            if( !alreadyMapped )
                m_memoryManager->mapToHost( m_memory, MAP_READ_WRITE );
            memset( m_memoryManager->getMappedToHostPtr( m_memory ) + offset, k_clearValue.get(), size );
            if( !alreadyMapped )
                m_memoryManager->unmapFromHost( m_memory );
        } );
    }

    BufferDimensions BD;
    BD.setFormat( RT_FORMAT_BYTE, 1 );
    BD.setSize( size );
    m_memory = m_memoryManager->allocateMBuffer( BD, m_allocationPolicy );
}

BackedAllocator::~BackedAllocator()
{
}

BackedAllocator::Handle BackedAllocator::alloc( size_t size, bool* backingUnmapped )
{
    if( backingUnmapped != nullptr )
        *backingUnmapped = false;
    bool   succeeded;
    Handle handle = m_allocator->alloc( size, &succeeded );
    if( !succeeded )
    {
        if( m_memoryManager->isMappedToHost( m_memory ) )
        {
            m_memoryManager->unmapFromHost( m_memory );
            if( backingUnmapped != nullptr )
                *backingUnmapped = true;
        }
        growMemory( size );
        m_allocator->expand( m_memory->getDimensions().getTotalSizeInBytes() );
        handle = m_allocator->alloc( size, &succeeded );
    }

    return handle;
}

size_t BackedAllocator::memorySize() const
{
    return m_memory->getDimensions().getTotalSizeInBytes();
}

size_t BackedAllocator::getUsedAddressRangeEnd() const
{
    return m_allocator->getUsedAddressRangeEnd();
}

void BackedAllocator::growMemory( size_t size )
{
    BufferDimensions BD         = m_memory->getDimensions();
    const size_t     sizeNeeded = BD.width() + size;
    do
    {
        BD.setSize( 2 * BD.width() );
    } while( BD.width() < sizeNeeded );
    BD.setSize( std::max( 2 * BD.width(), BD.width() + size ) );
    m_memoryManager->changeSize( m_memory, BD, true );
}

MBufferHandle BackedAllocator::memory()
{
    return m_memory;
}
