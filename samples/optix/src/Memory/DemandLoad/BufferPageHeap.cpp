//
// Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
//
// LWPU Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software, related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from LWPU Corporation is strictly
// prohibited.
//
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED//AS IS*
// AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
// INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
// SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
// LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
// BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES
//
#include <Memory/DemandLoad/BufferPageHeap.h>

#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Util/ContainerAlgorithm.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/MemoryAllocationFailed.h>

#include <algorithm>

namespace optix {

unsigned int BufferPageHeap::allocate( unsigned int pageCount )
{
    RT_ASSERT( pageCount != 0 );
    unsigned int startPage = 0;
    if( !m_heap.empty() )
    {
        const auto& back           = m_heap.back();
        startPage                  = back.startPage + back.pageCount;
        const unsigned int endPage = startPage + pageCount;
        if( endPage > m_numPages )
        {
            const std::string msg = "Requested " + std::to_string( pageCount ) + " pages, only "
                                    + std::to_string( endPage - m_numPages ) + " pages available";
            LOG_NORMAL( "BufferPageHeap::allocate() MemoryAllocationFailed " << msg );
            throw prodlib::MemoryAllocationFailed( RT_EXCEPTION_INFO, msg );
        }
    }
    m_heap.emplace_back( startPage, pageCount );
    return startPage;
}

void BufferPageHeap::associateBuffer( const Buffer* buffer, unsigned int startPage )
{
    RT_ASSERT( buffer != nullptr );
    const auto entry =
        algorithm::find_if( m_heap, [startPage]( const HeapEntry& entry ) { return startPage == entry.startPage; } );
    RT_ASSERT( entry != m_heap.end() );
    RT_ASSERT( entry->resource.buffer == nullptr || entry->resource.buffer == buffer );
    RT_ASSERT( entry->resource.samplers.empty() );
    entry->resource.buffer = buffer;
    entry->isBuffer        = true;
    entry->isLwdaSparse    = false;
}

void BufferPageHeap::associateLwdaSparse( const Buffer* buffer, unsigned int startPage )
{
    RT_ASSERT( buffer != nullptr );
    const auto entry =
        algorithm::find_if( m_heap, [startPage]( const HeapEntry& entry ) { return startPage == entry.startPage; } );
    RT_ASSERT( entry != m_heap.end() );
    // Either there is no buffer associated, or this buffer is already associated.
    // It's OK for this buffer to be repeatedly associated with this start page.
    RT_ASSERT( entry->resource.buffer == nullptr || entry->resource.buffer == buffer );
    RT_ASSERT( entry->resource.samplers.empty() );
    entry->resource.buffer = buffer;
    entry->isBuffer        = false;
    entry->isLwdaSparse    = true;
}

void BufferPageHeap::switchSparseTextureToTexture( unsigned int startPage )
{
    const auto entry =
        algorithm::find_if( m_heap, [startPage]( const HeapEntry& entry ) { return startPage == entry.startPage; } );
    RT_ASSERT( entry != m_heap.end() );
    RT_ASSERT( entry->isLwdaSparse );
    RT_ASSERT( entry->resource.samplers.empty() );
    entry->isBuffer     = false;
    entry->isLwdaSparse = false;
    entry->resource.buffer = nullptr;
    entry->resource.samplers.clear();
}

void BufferPageHeap::associateSampler( const TextureSampler* sampler, unsigned int startPage )
{
    RT_ASSERT( sampler != nullptr );
    const auto entry =
        algorithm::find_if( m_heap, [startPage]( const HeapEntry& entry ) { return startPage == entry.startPage; } );
    RT_ASSERT( entry != m_heap.end() );
    RT_ASSERT( !entry->isBuffer && entry->resource.buffer == nullptr );
    // There is only one BufferPageHeap across all devices, so multiple devices may ask
    // that the sampler be associated with the start page.  Therefore it's not an error
    // if the sampler is already in the list.
    if( algorithm::find( entry->resource.samplers, sampler ) == entry->resource.samplers.end() )
        entry->resource.samplers.push_back( sampler );
    entry->isBuffer     = false;
    entry->isLwdaSparse = false;
}

void BufferPageHeap::freeBuffer( const Buffer* buffer, unsigned int startPage )
{
    RT_ASSERT( buffer != nullptr );
    const auto entry =
        algorithm::find_if( m_heap, [startPage]( const HeapEntry& entry ) { return startPage == entry.startPage; } );
    RT_ASSERT( entry->isBuffer && entry->resource.samplers.empty() );
    RT_ASSERT( entry != m_heap.end() );
    entry->resource.buffer = nullptr;
    // Note that we can't erase the heap entry because page table entries can't be reused.
}

void BufferPageHeap::freeSampler( const TextureSampler* sampler, unsigned int startPage )
{
    RT_ASSERT( sampler != nullptr );
    const auto entry =
        algorithm::find_if( m_heap, [startPage]( const HeapEntry& entry ) { return startPage == entry.startPage; } );
    RT_ASSERT( !entry->isBuffer && entry->resource.buffer == nullptr );
    RT_ASSERT( entry != m_heap.end() );
    // We only store the sampler once across all devices, but we'll free the sampler once for every device.
    entry->resource.samplers.erase( std::remove( entry->resource.samplers.begin(), entry->resource.samplers.end(), sampler ),
                                    entry->resource.samplers.end() );
    // Note that we can't remove the heap entry because page table entries can't be reused.
}

const BufferPageHeap::HeapEntry* BufferPageHeap::find( unsigned int page ) const
{
    // Pages are allocated in increasing order, so the array of mappings is sorted, allowing us
    // to use binary search to find the the given page id.  (However, this does not easily permit eviction.)
    const auto least = std::lower_bound( m_heap.cbegin(), m_heap.cend(), page, []( const HeapEntry& entry, unsigned int page ) {
        return page > ( entry.startPage + entry.pageCount - 1 );
    } );
    return least != m_heap.cend() ? &*least : nullptr;
}

}  // namespace optix
