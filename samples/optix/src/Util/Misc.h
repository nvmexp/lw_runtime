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

#include <prodlib/exceptions/Assert.h>
#include <prodlib/math/Bits.h>

#include <cstdlib>
#include <string>
#include <vector>


namespace optix {

// Counts the number of objects in a statically sized array
#define countof( theArray ) ( sizeof( theArray ) / sizeof( ( theArray )[0] ) )

// Clear the vector to zero size, and reserve at least s elements, but no more than 150% of s.
template <typename T>
inline void clearAndSetCapacity( std::vector<T>& v, size_t s )
{
    // Remove the elements, setting the size to zero.
    v.clear();

    if( v.capacity() < s )
    {
        // Not enough. Reserve more.
        v.reserve( s );
    }

    if( v.capacity() > s + s / 2 )
    {
        // Too much. Free it and re-reserve.
        std::vector<T>().swap( v );
        v.reserve( s );
    }
}

// Make sure a vector has the given minimum size.
template <typename T>
inline void minsize( std::vector<T>& v, size_t s )
{
    if( v.size() < s )
        v.resize( s );
}

inline const size_t alignedAddress( const size_t offset, const size_t alignment )
{
    return prodlib::align( offset, alignment );
}

inline const size_t getVariableAlignment( const size_t size )
{
    unsigned int alignment = 1;

    if( size >= 16 )
        alignment = 16;
    else if( size >= 8 )
        alignment = 8;
    else if( size >= 4 )
        alignment = 4;
    else if( size >= 2 )
        alignment = 2;

    return alignment;
}

inline void* alignedMalloc( size_t byte_size, size_t alignment )
{
#ifdef _WIN32
    return _aligned_malloc( byte_size, alignment );
#else
    void* mem_ptr;
    int   res = posix_memalign( &mem_ptr, alignment, byte_size );
    RT_ASSERT( res == 0 );
    return mem_ptr;
#endif
}

inline void* alignedRealloc( void* ptr, size_t byte_size, size_t alignment )
{
#ifdef _WIN32
    return _aligned_realloc( ptr, byte_size, alignment );
#else
    // TODO: realloc implementation for Linux
    void* mem_ptr;
    int   res = posix_memalign( &mem_ptr, alignment, byte_size );
    RT_ASSERT( res == 0 );
    return mem_ptr;
#endif
}

inline void alignedFree( void* mem_ptr )
{
#ifdef _WIN32
    _aligned_free( mem_ptr );
#else
    return free( mem_ptr );
#endif
}

}  // end namespace optix
