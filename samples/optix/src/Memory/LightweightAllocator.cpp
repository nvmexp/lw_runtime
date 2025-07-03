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

#include <Memory/LightweightAllocator.h>

#include <corelib/math/MathUtil.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/math/Bits.h>

#include <string>
#include <utility>
#include <vector>

using namespace optix;
using namespace prodlib;
using namespace corelib;


LightweightAllocator::LightweightAllocator( size_t poolSize )
    : m_size( poolSize )
    , m_free( poolSize )
    , m_gteLargestFree( poolSize )
{
    addBlock( 0, poolSize );

#ifdef LWA_UNIT_TEST
    this->test();
#endif  // LWA_UNIT_TEST
}

LightweightAllocator::~LightweightAllocator()
{
}

size_t LightweightAllocator::alloc( size_t size, size_t alignment )
{
    RT_ASSERT( isPow2( alignment ) );

    if( size == 0 )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot allocate 0 bytes" );

    size = align( size, alignment );

    if( size > m_gteLargestFree )
        return BAD_ADDR;

    // Find a free block
    size_t freeBegin = 0;
    size_t freeEnd   = 0;
    size_t freeSize  = 0;

    size_t usedBegin = 0;
    size_t usedSize  = size;
    size_t usedEnd   = 0;

    size_t largestFreeSize = 0;
    bool   foundFreeBlock  = false;

    for( auto const& blockIt : m_beginMap )
    {
        freeBegin = blockIt.first;
        freeSize  = blockIt.second;
        freeEnd   = freeBegin + freeSize;

        if( freeSize > largestFreeSize )
            largestFreeSize = freeSize;

        usedBegin = align( freeBegin, alignment );
        usedEnd   = usedBegin + usedSize;

        if( usedEnd <= freeEnd )
        {
            foundFreeBlock = true;
            break;
        }
    }
    if( !foundFreeBlock )
    {
        m_gteLargestFree = largestFreeSize;
        return BAD_ADDR;
    }

    if( usedBegin != freeBegin )
    {
        // split block to make alignment work
        changeBlock( freeBegin, freeEnd, freeBegin, usedBegin );
        addBlock( usedBegin, freeEnd );
    }

    if( usedEnd == freeEnd )
    {
        delBlock( usedBegin, freeEnd );
    }
    else
    {
        changeBlock( usedBegin, freeEnd, usedEnd, freeEnd );
    }

    m_free -= size;
    RT_ASSERT( m_beginMap.size() == m_endMap.size() );
    return usedBegin;
}

void LightweightAllocator::free( size_t offset, size_t size, size_t alignment )
{
    size = align( size, alignment );
    m_free += size;

    size_t usedBegin = offset;
    size_t usedEnd   = offset + size;

    auto prevIt = m_endMap.find( usedBegin );
    auto nextIt = m_beginMap.find( usedEnd );

    size_t prevBegin = 0;
    size_t prevEnd   = 0;
    size_t prevSize  = 0;

    size_t nextBegin = 0;
    size_t nextEnd   = 0;
    size_t nextSize  = 0;

    if( prevIt != m_endMap.end() )
    {
        // Merge with previous block
        prevEnd   = prevIt->first;
        prevSize  = prevIt->second;
        prevBegin = prevEnd - prevSize;
        changeBlock( prevBegin, prevEnd, prevBegin, usedEnd );

        // Combine with next block if needed
        if( nextIt != m_beginMap.end() )
        {
            nextBegin = nextIt->first;
            nextSize  = nextIt->second;
            nextEnd   = nextBegin + nextSize;

            delBlock( nextBegin, nextEnd );
            changeBlock( prevBegin, usedEnd, prevBegin, nextEnd );
        }
    }
    else if( nextIt != m_beginMap.end() )
    {
        // Merge with next block
        nextBegin = nextIt->first;
        nextSize  = nextIt->second;
        nextEnd   = nextBegin + nextSize;

        changeBlock( nextBegin, nextEnd, usedBegin, nextEnd );
    }
    else
    {
        // Insert new block
        addBlock( usedBegin, usedEnd );
    }

    m_gteLargestFree = std::max( m_gteLargestFree, prevSize + size + nextSize );

    RT_ASSERT( m_beginMap.size() == m_endMap.size() );
}

size_t LightweightAllocator::freeSpace() const
{
    return m_free;
}

size_t LightweightAllocator::size() const
{
    return m_size;
}

bool LightweightAllocator::empty() const
{
    return m_size == m_free;
}

bool LightweightAllocator::validate()
{
    return true;
}

void LightweightAllocator::addBlock( size_t begin, size_t end )
{
    // Create a new block and add it to the maps
    size_t size       = end - begin;
    m_beginMap[begin] = size;
    m_endMap[end]     = size;
}

void LightweightAllocator::delBlock( size_t begin, size_t end )
{
    // Remove the block from the maps
    m_beginMap.erase( begin );
    m_endMap.erase( end );
}

void LightweightAllocator::changeBlock( size_t oldBegin, size_t oldEnd, size_t begin, size_t end )
{
    size_t newSize = end - begin;

    m_beginMap.erase( oldBegin );
    m_endMap.erase( oldEnd );

    m_beginMap[begin] = newSize;
    m_endMap[end]     = newSize;
}

#ifdef LWA_UNIT_TEST
void LightweightAllocator::test()
{
    static bool s_unitTestRan = false;

    if( !s_unitTestRan )
    {
        s_unitTestRan = true;

        // since we're using a random number generator, run this test a few times
        for( int retryCount = 10; retryCount; retryCount-- )
        {
            RT_ASSERT( m_beginMap.size() == 1 );
            RT_ASSERT( m_endMap.size() == 1 );

            size_t freeBefore = m_free;

            std::vector<std::tuple<size_t, size_t, size_t>> allocs;
            int sz;

            // fill randomly until first failure
            while( true )
            {
                const int alignment = 0x100 << ( rand() % 6 );
                sz                  = align( rand() % 10000 + 1, alignment );
                RT_ASSERT( sz > 0 );
                const int offset = this->alloc( sz, alignment );
                if( offset == BAD_ADDR )
                    break;
                allocs.push_back( std::tuple<size_t, size_t, size_t>( offset, sz, alignment ) );
            }

            // now fill forcefully with decreasing sizes until absolutely full
            const int alignment = 1;
            while( sz > 0 )
            {
                const int offset = this->alloc( sz, alignment );
                if( offset == BAD_ADDR )
                {
                    sz = sz / 2;
                }
                else
                {
                    allocs.push_back( std::tuple<size_t, size_t, size_t>( offset, sz, alignment ) );
                }
            }

            RT_ASSERT( m_free == 0 );

            for( const auto& blockIt : allocs )
            {
                const size_t offset    = std::get<0>( blockIt );
                const size_t sz        = std::get<1>( blockIt );
                const size_t alignment = std::get<2>( blockIt );
                this->free( offset, sz, alignment );
            }

            size_t freeAfter = m_free;

            RT_ASSERT( freeAfter == freeBefore );
            RT_ASSERT( m_beginMap.size() == 1 );
            RT_ASSERT( m_endMap.size() == 1 );
        }

        printf( "*** LightweightAllocator unit test PASSED\n" );
    }
}
#endif  // LWA_UNIT_TEST
