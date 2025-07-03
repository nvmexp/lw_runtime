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

#include <Util/BitSet.h>

#ifdef _WIN32
// In order to work around a VS9 bug, math.h needs to be included before intrin.h.
// Since this is the only place where this is done, we will pre-include math.h here.
// https://connect.microsoft.com/VisualStudio/feedback/ViewFeedback.aspx?FeedbackID=381422&wa=wsignin1.0
#include <intrin.h>
#include <math.h>
#endif

#include <Util/Memcpy.h>
#include <Util/Misc.h>
#include <prodlib/exceptions/Assert.h>

#include <algorithm>
#include <memory.h>

using namespace optix;
using namespace prodlib;


BitSet::BitSet()
    : m_size( 0 )
    , m_capacity( 0 )
    , m_data( nullptr )
    , m_owned( true )
{
}

BitSet::BitSet( index_t bitCount, bool setAllOne )
    : m_size( bitCount )
    , m_capacity( 0 )
    , m_data( nullptr )
{
    allocate();
    if( setAllOne )
        setAll();
    else
        clearAll();
}

BitSet::BitSet( element_t* data, index_t bitCount, index_t capacity )
    : m_size( bitCount )
    , m_capacity( capacity )
    , m_data( data )
    , m_owned( false )
{
    resize( bitCount );
    clearAll();
}

BitSet::BitSet( const BitSet& copy )
    : m_size( copy.m_size )
    , m_capacity( 0 )
    , m_data( nullptr )
{
    allocate();
    index_t nel = numElements();
    memcpy( m_data, copy.m_data, nel << element_t_size_power_of_2 );
}

BitSet::~BitSet()
{
    deallocate();
}

void BitSet::set( index_t idx )
{
    RT_ASSERT( idx < m_size );
    index_t   element = idx / bits_per_element;
    index_t   bit     = idx & bitmask;
    element_t mask    = element_t( 1 ) << bit;
    m_data[element] |= mask;
}

void BitSet::clear( index_t idx )
{
    RT_ASSERT( idx < m_size );
    index_t   element = idx / bits_per_element;
    index_t   bit     = idx & bitmask;
    element_t mask    = element_t( 1 ) << bit;
    m_data[element] &= ~mask;
}

void BitSet::clearAll()
{
    const index_t nel = numElements();
#if defined( SSE_41_AVAILABLE ) && !defined( FORCE_VANILLA_BITSET_IMPL )
    const size_t sizeInBytes  = nel << element_t_size_power_of_2;
    const size_t sseCondition = (size_t)m_data;  // Take the data address
    // If the address is aligned correctly, we can use SSE instructions
    if( ( sseCondition & 15 ) == 0 )
    {
        const size_t sizeAligned = sizeInBytes & ~( 16 - 1 );

        if( sizeAligned )
        {
            __m128i*             sse_dst = (__m128i*)( (char*)m_data );
            const __m128i* const dst_end = ( const __m128i* const )( (char*)m_data + sizeAligned );
            const __m128i        zero128 = _mm_setzero_si128();

            do
            {
                _mm_store_si128( sse_dst, zero128 );
                sse_dst++;
            } while( sse_dst != dst_end );
        }

        const size_t unalignedBytesToDo = sizeInBytes & 15;
        if( unalignedBytesToDo )
        {
            const size_t elementsDone = sizeAligned >> element_t_size_power_of_2;
            for( index_t i = elementsDone; i < nel; ++i )
            {
                m_data[i] = 0;
            }
        }
    }
    else
#endif
    {
        memset( m_data, 0, nel << element_t_size_power_of_2 );
    }
}

void BitSet::setAll()
{
    index_t nel = numElements();
    for( index_t i = 0; i < nel; ++i )
        m_data[i]  = ~0;
    // Last element needs to have only the valid bits set
    index_t bit = m_size & bitmask;
    if( bit )
    {
        element_t mask  = ( element_t( 1 ) << bit ) - 1;
        m_data[nel - 1] = mask;
    }
}

bool BitSet::isSet( index_t idx ) const
{
    RT_ASSERT( idx < m_size );
    index_t   element = idx / bits_per_element;
    index_t   bit     = idx & bitmask;
    element_t mask    = element_t( 1 ) << bit;
    return !!( m_data[element] & mask );
}

bool BitSet::allCleared() const
{
    index_t nel = numElements();
    for( index_t i = 0; i < nel; ++i )
    {
        if( m_data[i] )
            return false;
    }
    return true;
}

bool BitSet::rangeCleared( index_t begin, index_t end ) const
{
    RT_ASSERT( begin <= end );
    RT_ASSERT( end <= m_size );
    // only element boundaries supported atm
    RT_ASSERT( ( begin & bitmask ) == 0 );
    RT_ASSERT( ( end & bitmask ) == 0 );

    if( begin == end )
        return true;

    const index_t last      = end - 1;
    const index_t firstelem = begin / bits_per_element;
    const index_t lastelem  = last / bits_per_element;

    for( index_t i = firstelem; i < lastelem; ++i )
        if( m_data[i] )
            return false;

    return true;
}

bool BitSet::elementCleared( index_t idx ) const
{
    RT_ASSERT( idx < m_size );
    return m_data[idx / bits_per_element] == 0;
}

void BitSet::resize( index_t bitCount )
{
    if( numElements( bitCount ) > m_capacity )
    {
        if( m_owned )
        {
            // reallocate
            m_size     = bitCount;
            m_capacity = numElements();
            m_data     = (element_t*)alignedRealloc( m_data, m_capacity << element_t_size_power_of_2, 16 );
        }
        else
        {
            deallocate();
            m_size = bitCount;
            allocate();
        }
    }
    else
    {
        m_size = bitCount;
    }
}

BitSet::index_t BitSet::size() const
{
    return m_size;
}


BitSet& BitSet::operator=( const BitSet& rhs )
{
    if( &rhs == this )
        return *this;
    resize( rhs.m_size );

    const index_t nel = numElements();
#if defined( SSE_41_AVAILABLE ) && !defined( FORCE_VANILLA_BITSET_IMPL )
    const size_t sizeInBytes   = nel << element_t_size_power_of_2;
    const size_t pointersOrred = (size_t)m_data | (size_t)rhs.m_data;
    // if both pointers are aligned correctly, we can use SSE instructions
    if( ( pointersOrred & 15 ) == 0 )
    {
        const size_t sizeAligned = sizeInBytes & ~( 16 - 1 );

        if( sizeAligned )
        {
            __m128i*             sse_dst = (__m128i*)( (char*)m_data );
            const __m128i*       sse_src = (const __m128i*)( (char*)rhs.m_data );
            const __m128i* const src_end = ( const __m128i* const )( (char*)rhs.m_data + sizeAligned );

            do
            {
                _mm_store_si128( sse_dst, _mm_load_si128( sse_src ) );

                sse_src++;
                sse_dst++;
            } while( sse_src != src_end );
        }

        const size_t unalignedBytesToDo = sizeInBytes & 15;
        if( unalignedBytesToDo )
        {
            const size_t elementsDone = sizeAligned >> element_t_size_power_of_2;
            for( index_t i = elementsDone; i < nel; ++i )
            {
                m_data[i] = rhs.m_data[i];
            }
        }
    }
    else
#endif
    {
        memcpy( m_data, rhs.m_data, nel << element_t_size_power_of_2 );
    }

    return *this;
}

bool BitSet::operator!=( const BitSet& rhs ) const
{
    return !operator==( rhs );
}


bool BitSet::operator==( const BitSet& rhs ) const
{
    RT_ASSERT( m_size == rhs.m_size );
    const index_t nel = numElements();
#if defined( SSE_41_AVAILABLE ) && !defined( FORCE_VANILLA_BITSET_IMPL )
    const size_t sizeInBytes   = nel << element_t_size_power_of_2;
    const size_t pointersOrred = (size_t)m_data | (size_t)rhs.m_data;
    // if both pointers are aligned correctly, we can use SSE instructions
    if( ( pointersOrred & 15 ) == 0 )
    {
        const size_t sizeAligned = sizeInBytes & ~( 16 - 1 );

        if( sizeAligned )
        {
            const __m128i*       sse_dst = (__m128i*)( (char*)m_data );
            const __m128i*       sse_src = (const __m128i*)( (char*)rhs.m_data );
            const __m128i* const src_end = ( const __m128i* const )( (char*)rhs.m_data + sizeAligned );

            do
            {
                const __m128i val_sse = _mm_xor_si128( _mm_load_si128( sse_dst ), _mm_load_si128( sse_src ) );

                if( ( (unsigned long long*)&val_sse )[0] | ( (unsigned long long*)&val_sse )[1] )
                {
                    return false;
                }

                sse_src++;
                sse_dst++;
            } while( sse_src != src_end );
        }

        const size_t unalignedBytesToDo = sizeInBytes & 15;
        if( unalignedBytesToDo )
        {
            const size_t elementsDone = sizeAligned >> element_t_size_power_of_2;
            for( index_t i = elementsDone; i < nel; ++i )
            {
                if( m_data[i] != rhs.m_data[i] )
                    return false;
            }
        }
    }
    else
#endif
    {
        for( index_t i = 0; i < nel; ++i )
        {
            if( m_data[i] != rhs.m_data[i] )
                return false;
        }
    }

    return true;
}


BitSet& BitSet::operator|=( const BitSet& rhs )
{
    RT_ASSERT( m_size == rhs.m_size );
    const index_t nel = numElements();
#if defined( SSE_41_AVAILABLE ) && !defined( FORCE_VANILLA_BITSET_IMPL )
    const size_t sizeInBytes   = nel << element_t_size_power_of_2;
    const size_t pointersOrred = (size_t)m_data | (size_t)rhs.m_data;
    // if both pointers are aligned correctly, we can use SSE instructions
    if( ( pointersOrred & 15 ) == 0 )
    {
        const size_t sizeAligned = sizeInBytes & ~( 16 - 1 );

        if( sizeAligned )
        {
            __m128i*             sse_dst = (__m128i*)( (char*)m_data );
            const __m128i*       sse_src = (const __m128i*)( (char*)rhs.m_data );
            const __m128i* const src_end = ( const __m128i* const )( (char*)rhs.m_data + sizeAligned );

            do
            {
                _mm_store_si128( sse_dst, _mm_or_si128( _mm_load_si128( sse_dst ), _mm_load_si128( sse_src ) ) );
                sse_src++;
                sse_dst++;
            } while( sse_src != src_end );
        }

        const size_t unalignedBytesToDo = sizeInBytes & 15;
        if( unalignedBytesToDo )
        {
            const size_t elementsDone = sizeAligned >> element_t_size_power_of_2;
            for( index_t i = elementsDone; i < nel; ++i )
            {
                m_data[i] |= rhs.m_data[i];
            }
        }
    }
    else
#endif
    {
        for( index_t i = 0; i < nel; ++i )
        {
            m_data[i] |= rhs.m_data[i];
        }
    }

    return *this;
}


BitSet& BitSet::operator-=( const BitSet& rhs )
{
    RT_ASSERT( m_size == rhs.m_size );
    const index_t nel = numElements();
#if defined( SSE_41_AVAILABLE ) && !defined( FORCE_VANILLA_BITSET_IMPL )
    const size_t sizeInBytes   = nel << element_t_size_power_of_2;
    const size_t pointersOrred = (size_t)m_data | (size_t)rhs.m_data;
    // if both pointers are aligned correctly and SSE 4.1 is supported, we can use SSE instructions
    if( ( pointersOrred & 15 ) == 0 )
    {
        const size_t sizeAligned = sizeInBytes & ~( 16 - 1 );

        if( sizeAligned )
        {
            __m128i*             sse_dst = (__m128i*)( (char*)m_data );
            const __m128i*       sse_src = (const __m128i*)( (char*)rhs.m_data );
            const __m128i* const src_end = ( const __m128i* const )( (char*)rhs.m_data + sizeAligned );

            do
            {
                // NOTE that the "andnot" SSE intrinsic below actually returns "notand": ~A & B;
                // so we swap the parameter as needed.
                _mm_store_si128( sse_dst, _mm_andnot_si128( _mm_load_si128( sse_src ), _mm_load_si128( sse_dst ) ) );

                sse_src++;
                sse_dst++;
            } while( sse_src != src_end );
        }

        const size_t unalignedBytesToDo = sizeInBytes & 15;
        if( unalignedBytesToDo )
        {
            const size_t elementsDone = sizeAligned >> element_t_size_power_of_2;
            for( index_t i = elementsDone; i < nel; ++i )
            {
                m_data[i] &= ~rhs.m_data[i];
            }
        }
    }
    else
#endif
    {
        for( index_t i = 0; i < nel; ++i )
        {
            m_data[i] &= ~rhs.m_data[i];
        }
    }

    return *this;
}

BitSet& BitSet::operator&=( const BitSet& rhs )
{
    RT_ASSERT( m_size == rhs.m_size );
    const index_t nel = numElements();
#if defined( SSE_41_AVAILABLE ) && !defined( FORCE_VANILLA_BITSET_IMPL )
    const size_t sizeInBytes   = nel << element_t_size_power_of_2;
    const size_t pointersOrred = (size_t)m_data | (size_t)rhs.m_data;
    // if both pointers are aligned correctly, we can use SSE instructions
    if( ( pointersOrred & 15 ) == 0 )
    {
        const size_t sizeAligned = sizeInBytes & ~( 16 - 1 );

        if( sizeAligned )
        {
            __m128i*             sse_dst = (__m128i*)( (char*)m_data );
            const __m128i*       sse_src = (const __m128i*)( (char*)rhs.m_data );
            const __m128i* const src_end = ( const __m128i* const )( (char*)rhs.m_data + sizeAligned );

            do
            {
                _mm_store_si128( sse_dst, _mm_and_si128( _mm_load_si128( sse_src ), _mm_load_si128( sse_dst ) ) );

                sse_src++;
                sse_dst++;
            } while( sse_src != src_end );
        }

        const size_t unalignedBytesToDo = sizeInBytes & 15;
        if( unalignedBytesToDo )
        {
            const size_t elementsDone = sizeAligned >> element_t_size_power_of_2;
            for( index_t i = elementsDone; i < nel; ++i )
            {
                m_data[i] &= rhs.m_data[i];
            }
        }
    }
    else
#endif
    {
        for( index_t i = 0; i < nel; ++i )
        {
            m_data[i] &= rhs.m_data[i];
        }
    }

    return *this;
}

static int findFirstSet( BitSet::element_t elt )
{
    int idx = 0;
    if( ( elt & 0xffffffff ) == 0 )
    {
        elt >>= 32;
        idx += 32;
    }
    if( ( elt & 0xffff ) == 0 )
    {
        elt >>= 16;
        idx += 16;
    }
    if( ( elt & 0xff ) == 0 )
    {
        elt >>= 8;
        idx += 8;
    }
    if( ( elt & 0xf ) == 0 )
    {
        elt >>= 4;
        idx += 4;
    }
    if( ( elt & 0x3 ) == 0 )
    {
        elt >>= 2;
        idx += 2;
    }
    if( ( elt & 0x1 ) == 0 )
    {
        elt >>= 1;
        idx += 1;
    }
    return idx;
}


static int findFirstClear( BitSet::element_t elt )
{
    int idx = 0;
    if( ( elt & 0xffffffff ) == 0xffffffff )
    {
        elt >>= 32;
        idx += 32;
    }
    if( ( elt & 0xffff ) == 0xffff )
    {
        elt >>= 16;
        idx += 16;
    }
    if( ( elt & 0xff ) == 0xff )
    {
        elt >>= 8;
        idx += 8;
    }
    if( ( elt & 0xf ) == 0xf )
    {
        elt >>= 4;
        idx += 4;
    }
    if( ( elt & 0x3 ) == 0x3 )
    {
        elt >>= 2;
        idx += 2;
    }
    if( ( elt & 0x1 ) == 0x1 )
    {
        elt >>= 1;
        idx += 1;
    }
    return idx;
}


BitSet::index_t BitSet::findFirst( bool value ) const
{
    // First the first cleared (value==false) or set (value==true)
    const index_t nel = numElements();
    if( value )
    {
        for( index_t i = 0; i < nel; ++i )
        {
            if( m_data[i] != 0ull )
                return std::min( m_size, i * bits_per_element + findFirstSet( m_data[i] ) );
        }
    }
    else
    {
        for( index_t i = 0; i < nel; ++i )
        {
            if( m_data[i] != ~0ull )
                return std::min( m_size, i * bits_per_element + findFirstClear( m_data[i] ) );
        }
    }
    return m_size;
}

static int findLastSet( BitSet::element_t elt )
{
    int idx = 0;
    if( ( elt << 32 ) == 0 )
    {
        elt <<= 32;
        idx += 32;
    }
    if( ( elt << 16 ) == 0 )
    {
        elt <<= 16;
        idx += 16;
    }
    if( ( elt << 8 ) == 0 )
    {
        elt <<= 8;
        idx += 8;
    }
    if( ( elt << 4 ) == 0 )
    {
        elt >>= 4;
        idx += 4;
    }
    if( ( elt << 2 ) == 0 )
    {
        elt <<= 2;
        idx += 2;
    }
    if( ( elt << 1 ) == 0 )
    {
        elt <<= 1;
        idx += 1;
    }
    return idx;
}


static int findLastClear( BitSet::element_t elt )
{
    int idx = 0;
    if( ( elt << 32 ) == 0xffffffff )
    {
        elt <<= 32;
        idx += 32;
    }
    if( ( elt << 16 ) == 0xffff )
    {
        elt <<= 16;
        idx += 16;
    }
    if( ( elt << 8 ) == 0xff )
    {
        elt <<= 8;
        idx += 8;
    }
    if( ( elt << 4 ) == 0xf )
    {
        elt >>= 4;
        idx += 4;
    }
    if( ( elt << 2 ) == 0x3 )
    {
        elt <<= 2;
        idx += 2;
    }
    if( ( elt << 1 ) == 0x1 )
    {
        elt <<= 1;
        idx += 1;
    }
    return idx;
}


BitSet::index_t BitSet::findLast( bool value ) const
{
    // First the last cleared (value==false) or set (value==true)
    const index_t nel = numElements();
    if( value )
    {
        for( index_t i = nel - 1; i != ~0ull; --i )
        {
            if( m_data[i] != 0ull )
                return std::min( m_size, i * bits_per_element + findLastSet( m_data[i] ) );
        }
    }
    else
    {
        for( index_t i = nel - 1; i != ~0ull; --i )
        {
            if( m_data[i] != ~0ull )
                return std::min( m_size, i * bits_per_element + findLastClear( m_data[i] ) );
        }
    }
    return m_size;
}

BitSet::const_iterator BitSet::begin() const
{
    return const_iterator( this, findFirst( true ) );
}

BitSet::const_iterator BitSet::end() const
{
    return const_iterator( this, m_size );
}

BitSet::index_t BitSet::const_iterator::operator*() const
{
    return pos;
}

BitSet::const_iterator& BitSet::const_iterator::operator++()
{
    // Inefficient algorithm - reimplement in the unlikely event it
    // shows up in a profile
    pos++;
    while( pos < parent->size() && !parent->isSet( pos ) )
        pos++;
    return *this;
}

bool BitSet::const_iterator::operator!=( const const_iterator& b ) const
{
    return parent != b.parent || pos != b.pos;
}

bool BitSet::const_iterator::operator==( const const_iterator& b ) const
{
    return parent == b.parent && pos == b.pos;
}

BitSet::const_iterator::const_iterator( const BitSet* parent, index_t pos )
    : parent( parent )
    , pos( pos )
{
}

BitSet::index_t BitSet::numElements( BitSet::index_t bitCount )
{
    return ( bitCount + bitmask ) >> bits_per_element_power_of_2;
}

BitSet::index_t BitSet::numElements() const
{
    return numElements( m_size );
}

void BitSet::allocate()
{
    RT_ASSERT( m_data == nullptr );
    if( m_size )
    {
        m_capacity = numElements();
        m_data     = (element_t*)alignedMalloc( m_capacity << element_t_size_power_of_2, 16 );
    }
    m_owned = true;
}

void BitSet::deallocate()
{
    if( m_owned )
    {
        alignedFree( m_data );
        m_data     = nullptr;
        m_capacity = 0;
    }
    else
    {
        m_capacity = 0;
        m_data     = nullptr;
    }
}

// BitSetStack implementation

template <size_t Size>
BitSetStack<Size>::BitSetStack( size_t bitCount )
    : BitSet( (BitSet::element_t*)m_buf, bitCount, Size * 2 )
{
}

template <size_t Size>
BitSetStack<Size>::BitSetStack( const BitSet& copy )
    : BitSet( (BitSet::element_t*)m_buf, copy.size(), Size * 2 )
{
}

template <size_t   Size>
BitSetStack<Size>& BitSetStack<Size>::operator=( const BitSet& copy )
{
    BitSet::operator=( copy );
    return *this;
}

// Explicit instantiations
namespace optix {
template class BitSetStack<4>;
}
