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

#include <Util/Misc.h>
#include <corelib/system/Preprocessor.h>
#include <prodlib/exceptions/Assert.h>

#include <stddef.h>

#include <sse_support.h>
#if defined( SSE_41_AVAILABLE ) && !defined( FORCE_VANILLA_BITSET_IMPL )
#include <xmmintrin.h>
#endif


namespace optix {

class BitSet
{
  public:
    typedef size_t             index_t;
    typedef unsigned long long element_t;  // storage elements are sizeof(unsigned long long) = 8 bytes in size
    static const unsigned int  element_t_size_power_of_2 = 3;  // NOTE that this must match the above element_t!
    // (this is used to avoid divisions in the unaligned tail copies)

    // Create an empty bitset.
    BitSet();

    // Create a bitset with all bits cleared.
    explicit BitSet( index_t bitCount, bool setAllOne = false );

    // Create a bitset that doesn't own its data. Clears all bits.
    BitSet( element_t* data, index_t bitCount, index_t capacity );

    BitSet( const BitSet& copy );
    ~BitSet();

    void set( index_t idx );
    void clear( index_t idx );
    void clearAll();
    void setAll();
    bool isSet( index_t idx ) const;
    bool allCleared() const;
    bool rangeCleared( index_t begin, index_t end ) const;  // end is one past last bit to be checked
    bool elementCleared( index_t idx ) const;

    void resize( index_t bitCount );
    index_t size() const;

    BitSet& operator=( const BitSet& rhs );
    bool operator!=( const BitSet& rhs ) const;

    bool operator==( const BitSet& rhs ) const;
    BitSet& operator|=( const BitSet& rhs );
    BitSet& operator-=( const BitSet& rhs );
    BitSet& operator&=( const BitSet& rhs );

    index_t findFirst( bool value ) const;
    index_t findLast( bool value ) const;

    struct const_iterator
    {
      public:
        index_t operator*() const;
        const_iterator& operator++();
        bool operator!=( const const_iterator& b ) const;
        bool operator==( const const_iterator& b ) const;

      private:
        friend class BitSet;
        const_iterator( const BitSet* parent, index_t pos );
        const BitSet* parent;
        index_t       pos;
    };
    const_iterator begin() const;
    const_iterator end() const;

    static unsigned int bitsPerElement() { return bits_per_element; }

  private:
    static const unsigned int bits_per_element_power_of_2 = element_t_size_power_of_2 + 3;
    static const unsigned int bits_per_element            = 1 << bits_per_element_power_of_2;
    static const index_t      bitmask                     = bits_per_element - 1;
    static index_t numElements( BitSet::index_t bitCount );
    index_t numElements() const;

    index_t    m_size;      // in bits
    index_t    m_capacity;  // number of allocated elements
    element_t* m_data;
    bool       m_owned;

    void allocate();
    void deallocate();
};

// Stack based Bitset
template <size_t Size = 4>
class BitSetStack : public BitSet
{
  public:
    // Create a bitset with all bits cleared.
    explicit BitSetStack( size_t bitCount );
    BitSetStack( const BitSet& copy );
    BitSetStack& operator=( const BitSet& copy );

  private:
#if defined( SSE_41_AVAILABLE ) && !defined( FORCE_VANILLA_BITSET_IMPL )
    __m128 m_buf[Size];  // force alignment of 16 for SSE operations
#else
    BitSet::element_t m_buf[Size * 2];
#endif
};

}  // end namespace optix
