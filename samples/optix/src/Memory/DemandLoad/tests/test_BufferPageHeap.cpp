// Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
//
// LWPU Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software, related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from LWPU Corporation is strictly
// prohibited.
//
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
// AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
// INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
// SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
// LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
// BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES

#include <srcTests.h>

#include <Memory/DemandLoad/BufferPageHeap.h>
#include <Objects/Buffer.h>
#include <Util/ContainerAlgorithm.h>

#include <prodlib/exceptions/MemoryAllocationFailed.h>

using namespace optix;
using namespace testing;

namespace {

const unsigned int NUM_PAGES = 1024U;

class TestBufferPageHeap : public Test
{
  public:
    TestBufferPageHeap()
        : m_heap( NUM_PAGES )
        , m_buffer( reinterpret_cast<const Buffer*>( 0xdeadbeefULL ) )
        , m_sampler( reinterpret_cast<const TextureSampler*>( 0xfacef00dULL ) )
        , m_queryPage( 20U )
    {
    }

    BufferPageHeap        m_heap;
    const Buffer*         m_buffer;
    const TextureSampler* m_sampler;
    unsigned int          m_queryPage;
};

}  // namespace

TEST_F( TestBufferPageHeap, find_buffer_for_page_in_range )
{
    const unsigned int pageCount = 24;
    const unsigned int startPage = m_heap.allocate( pageCount );
    m_heap.associateBuffer( m_buffer, startPage );

    const BufferPageHeap::HeapEntry* foundEntry = m_heap.find( m_queryPage );

    ASSERT_TRUE( foundEntry != nullptr );
    ASSERT_EQ( m_buffer, foundEntry->resource.buffer );
}

TEST_F( TestBufferPageHeap, returns_nullptr_for_buffer_page_not_in_heap )
{
    const unsigned int pageCount = 12;
    const unsigned int startPage = m_heap.allocate( pageCount );
    m_heap.associateBuffer( m_buffer, startPage );

    const BufferPageHeap::HeapEntry* foundEntry = m_heap.find( m_queryPage );

    ASSERT_EQ( nullptr, foundEntry );
}

TEST_F( TestBufferPageHeap, find_sampler_for_page_in_range )
{
    const unsigned int pageCount = 24;
    const unsigned int startPage = m_heap.allocate( pageCount );
    m_heap.associateSampler( m_sampler, startPage );

    const BufferPageHeap::HeapEntry* foundEntry = m_heap.find( m_queryPage );

    ASSERT_TRUE( foundEntry != nullptr );
    ASSERT_TRUE( algorithm::find( foundEntry->resource.samplers, m_sampler ) != foundEntry->resource.samplers.end() );
}

TEST_F( TestBufferPageHeap, returns_nullptr_for_sampler_page_not_in_heap )
{
    const unsigned int pageCount = 12;
    const unsigned int startPage = m_heap.allocate( pageCount );
    m_heap.associateSampler( m_sampler, startPage );

    const BufferPageHeap::HeapEntry* foundEntry = m_heap.find( m_queryPage );

    ASSERT_EQ( nullptr, foundEntry );
}

TEST_F( TestBufferPageHeap, allocates_pages_starting_at_zero )
{
    const unsigned int pageCount = 512;

    const unsigned int startPage = m_heap.allocate( pageCount );

    ASSERT_EQ( 0U, startPage );
}

TEST_F( TestBufferPageHeap, allocates_additional_pages_at_the_end )
{
    const unsigned int pageCount = 512;
    const unsigned int pageZero  = m_heap.allocate( pageCount );

    const unsigned int secondAllocation = m_heap.allocate( pageCount );

    ASSERT_EQ( 0U, pageZero );
    ASSERT_EQ( pageCount, secondAllocation );
}

// TODO: 2757537 Virtual Pages for a demand load buffer aren't released when the buffer is destroyed
TEST_F( TestBufferPageHeap, DISABLED_freeBuffer_allows_pages_to_be_reallocated )
{
    const unsigned int pageCount = 10;
    const unsigned int startPage = m_heap.allocate( pageCount );
    m_heap.associateBuffer( m_buffer, startPage );

    m_heap.freeBuffer( m_buffer, startPage );
    const unsigned int secondStartPage = m_heap.allocate( pageCount );

    ASSERT_EQ( startPage, secondStartPage );
}

TEST_F( TestBufferPageHeap, exhaustively_test_find )
{
    const unsigned int        count = 10;
    std::vector<unsigned int> startPages( count );
    for( unsigned int i = 1; i <= count; ++i )
    {
        const unsigned int numPages = i;
        startPages[i - 1]           = m_heap.allocate( numPages );
        m_heap.associateBuffer( reinterpret_cast<Buffer*>( i ), startPages[i - 1] );
    }

    for( unsigned int i = 1; i <= count; ++i )
    {
        unsigned int startPage = startPages[i - 1];
        unsigned int lastPage  = startPage + i - 1;
        Buffer*      buffer    = reinterpret_cast<Buffer*>( i );

        EXPECT_EQ( buffer, m_heap.find( startPage )->resource.buffer );
        EXPECT_EQ( buffer, m_heap.find( lastPage )->resource.buffer );
    }
}

TEST_F( TestBufferPageHeap, single_range )
{
    m_heap.allocate( 656 );

    const BufferPageHeap::HeapEntry* entry = m_heap.find( 198 );

    ASSERT_NE( nullptr, entry );
}

TEST_F( TestBufferPageHeap, page_exhaustion_throws_specific_error )
{
    m_heap.allocate( NUM_PAGES );

    EXPECT_THROW( m_heap.allocate( 1 ), prodlib::MemoryAllocationFailed );
}

TEST_F( TestBufferPageHeap, entry_marked_as_buffer_when_associated )
{
    const unsigned int count     = 1;
    const unsigned int firstPage = m_heap.allocate( count );
    const Buffer*      buffer    = reinterpret_cast<Buffer*>( 0xdeadbeef );
    m_heap.associateBuffer( buffer, firstPage );

    const BufferPageHeap::HeapEntry* entry = m_heap.find( firstPage );

    EXPECT_TRUE( entry->isBuffer );
    EXPECT_FALSE( entry->isLwdaSparse );
}

TEST_F( TestBufferPageHeap, entry_marked_as_sampler_when_associated )
{
    const unsigned int    count     = 1;
    const unsigned int    firstPage = m_heap.allocate( count );
    const TextureSampler* sampler   = reinterpret_cast<TextureSampler*>( 0xdeadbeef );
    m_heap.associateSampler( sampler, firstPage );

    const BufferPageHeap::HeapEntry* entry = m_heap.find( firstPage );

    EXPECT_FALSE( entry->isBuffer );
    EXPECT_FALSE( entry->isLwdaSparse );
}

TEST_F( TestBufferPageHeap, entry_marked_as_lwda_sparse_when_associated )
{
    const unsigned int count     = 1;
    const unsigned int firstPage = m_heap.allocate( count );
    const Buffer*      buffer    = reinterpret_cast<Buffer*>( 0xdeadbeef );
    m_heap.associateLwdaSparse( buffer, firstPage );

    const BufferPageHeap::HeapEntry* entry = m_heap.find( firstPage );

    EXPECT_FALSE( entry->isBuffer );
    EXPECT_TRUE( entry->isLwdaSparse );
}

TEST_F( TestBufferPageHeap, cannot_release_buffer_for_associated_sampler )
{
    const unsigned int    count     = 1;
    const unsigned int    firstPage = m_heap.allocate( count );
    const TextureSampler* sampler   = reinterpret_cast<TextureSampler*>( 0xdeadbeef );
    const Buffer*         buffer    = reinterpret_cast<Buffer*>( 0xdeadbeef );
    m_heap.associateSampler( sampler, firstPage );

    EXPECT_THROW( m_heap.freeBuffer( buffer, firstPage ), prodlib::AssertionFailure );
}

TEST_F( TestBufferPageHeap, cannot_release_sampler_for_associated_buffer )
{
    const unsigned int    count     = 1;
    const unsigned int    firstPage = m_heap.allocate( count );
    const TextureSampler* sampler   = reinterpret_cast<TextureSampler*>( 0xdeadbeef );
    const Buffer*         buffer    = reinterpret_cast<Buffer*>( 0xdeadbeef );
    m_heap.associateBuffer( buffer, firstPage );

    EXPECT_THROW( m_heap.freeSampler( sampler, firstPage ), prodlib::AssertionFailure );
}

TEST_F( TestBufferPageHeap, associate_multiple_samplers_with_single_start_page )
{
    const unsigned int    count     = 1;
    const unsigned int    firstPage = m_heap.allocate( count );
    const TextureSampler* sampler1  = reinterpret_cast<TextureSampler*>( 0xdeadbeef );
    const TextureSampler* sampler2  = reinterpret_cast<TextureSampler*>( 0xbaadf00d );

    m_heap.associateSampler( sampler1, firstPage );
    m_heap.associateSampler( sampler2, firstPage );
}

TEST_F( TestBufferPageHeap, associate_same_sampler_multiple_times_is_ok )
{
    const unsigned int    count     = 1;
    const unsigned int    firstPage = m_heap.allocate( count );
    const TextureSampler* sampler   = reinterpret_cast<TextureSampler*>( 0xdeadbeef );

    m_heap.associateSampler( sampler, firstPage );
    m_heap.associateSampler( sampler, firstPage );
}

TEST_F( TestBufferPageHeap, free_multiple_samplers_with_single_start_page )
{
    const unsigned int    count     = 1;
    const unsigned int    firstPage = m_heap.allocate( count );
    const TextureSampler* sampler1  = reinterpret_cast<TextureSampler*>( 0xdeadbeef );
    const TextureSampler* sampler2  = reinterpret_cast<TextureSampler*>( 0xbaadf00d );
    m_heap.associateSampler( sampler1, firstPage );
    m_heap.associateSampler( sampler2, firstPage );

    m_heap.freeSampler( sampler1, firstPage );
    m_heap.freeSampler( sampler2, firstPage );
}

TEST_F( TestBufferPageHeap, free_same_sampler_multiple_times_is_ok )
{
    const unsigned int    count     = 1;
    const unsigned int    firstPage = m_heap.allocate( count );
    const TextureSampler* sampler   = reinterpret_cast<TextureSampler*>( 0xdeadbeef );
    m_heap.associateSampler( sampler, firstPage );
    m_heap.associateSampler( sampler, firstPage );

    m_heap.freeSampler( sampler, firstPage );
    m_heap.freeSampler( sampler, firstPage );
}

TEST_F( TestBufferPageHeap, switching_from_lwda_sparse_to_plain_texture )
{
    const unsigned int count     = 1;
    const unsigned int firstPage = m_heap.allocate( count );
    const Buffer*      buffer    = reinterpret_cast<Buffer*>( 0xdeadbeef );
    m_heap.associateLwdaSparse( buffer, firstPage );

    m_heap.switchSparseTextureToTexture( firstPage );

    const BufferPageHeap::HeapEntry* entry = m_heap.find( firstPage );
    EXPECT_FALSE( entry->isBuffer );
    EXPECT_FALSE( entry->isLwdaSparse );
    EXPECT_EQ( entry->resource.buffer, nullptr );
    EXPECT_TRUE( entry->resource.samplers.empty() );
}
