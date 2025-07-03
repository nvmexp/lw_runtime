// Copyright LWPU Corporation 2008
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

#include <prodlib/exceptions/Assert.h>


namespace prodlib {

// Return the 1-based index of the most significant bit that's set.
inline int mostSignificantBitSet( unsigned int x )
{
    int bit = 0;
    while( x )
    {
        x >>= 1;
        bit++;
    }
    return bit;
}

// Return the 1-based index of the least significant bit that's set.
inline int leastSignificantBitSet( unsigned int x )
{
    if( !x )
        return 0;
    int bit = 33;
    do
    {
        x <<= 1;
        bit--;
    } while( x );
    return bit;
}

// Return the number of bits set in a
inline unsigned int popCount( unsigned int a )
{
    unsigned int c;
    c = ( a & 0x55555555 ) + ( ( a >> 1 ) & 0x55555555 );
    c = ( c & 0x33333333 ) + ( ( c >> 2 ) & 0x33333333 );
    c = ( c & 0x0f0f0f0f ) + ( ( c >> 4 ) & 0x0f0f0f0f );
    c = ( c & 0x00ff00ff ) + ( ( c >> 8 ) & 0x00ff00ff );
    c = ( c & 0x0000ffff ) + ( c >> 16 );
    return c;
}

// Sets or unsets the bit specified by 'index' in 'bits'
template <typename BitT>
void setBit( BitT& bits, const int index, const bool enabled )
{
    RT_ASSERT( index < 64 );
    if( enabled )
        bits |= ( uint64_t( 1 ) << index );
    else
        bits &= ~( uint64_t( 1 ) << index );
}

// Returns whether the bit specified by 'index' is set in 'bits'
template <typename BitT>
bool isBitSet( const BitT bits, const int index )
{
    RT_ASSERT( index < 64 );
    return ( bits >> index ) & 1;
}

// Returns a copy of "bits" where the bits specified by flags are 0.
template <typename BitT, typename FlagT>
BitT maskOutFlags( BitT bits, FlagT flags )
{
    static_assert( sizeof( BitT ) >= sizeof( FlagT ), "Incompatible type sizes" );
    BitT flags_cast = static_cast<BitT>( flags );
    return ( bits & ~flags_cast );
}

// Returns a copy of "bits" where the bits not specified by flags are 0.
template <typename BitT, typename FlagT>
BitT onlyFlagBits( BitT bits, FlagT flags )
{
    static_assert( sizeof( BitT ) >= sizeof( FlagT ), "Incompatible type sizes" );
    BitT flags_cast = static_cast<BitT>( flags );
    return ( bits & flags_cast );
}

// Returns true if all the bits specified by flags are 1.
template <typename BitT, typename FlagT>
bool flagsOn( BitT bits, FlagT flags )
{
    static_assert( sizeof( BitT ) >= sizeof( FlagT ), "Incompatible type sizes" );
    BitT flags_cast = static_cast<BitT>( flags );
    return ( bits & flags_cast ) == flags_cast;
}

// Returns true if any of the bits specified by flags are 1.
template <typename BitT, typename FlagT>
bool anyflagsOn( BitT bits, FlagT flags )
{
    return onlyFlagBits( bits, flags ) != 0;
}

// alignment must be a non-zero power-of-two
template <typename BaseT, typename AlignmentT>
BaseT align( BaseT base, AlignmentT alignment )
{
    return ( base + alignment - 1 ) & ~( alignment - 1 );
}

//
template <typename BaseT, typename AlignmentT>
bool isAligned( BaseT base, AlignmentT alignment )
{
    return ( base & ( alignment - 1 ) ) == 0;
}

}  // end namespace prodlib
