// Copyright LWPU Corporation 2017
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

#include <corelib/math/MathUtil.h>

namespace prodlib {

class BufferLayoutUtil
{
  public:
    BufferLayoutUtil( void* buffer = nullptr ) { resetBase( buffer ); }

    void resetBase( void* buffer ) { m_max = m_begin = m_end = (char*)buffer; }

    // Alignment assumed to be a power of 2
    void* append( size_t sizeInBytes, size_t pow2Alignment = 0 )
    {
        char* block = m_end;
        if( pow2Alignment > 0 )
        {
            size_t addr = ( size_t )( block );
            addr        = corelib::roundUpPow2( addr, pow2Alignment );
            block       = (char*)addr;
        }
        m_end = block + sizeInBytes;
        if( m_end > m_max )
            m_max = m_end;
        return block;
    }

    void* overlay( void* buffer, size_t sizeInBytes = 0, size_t pow2Alignment = 0 )
    {
        m_end = (char*)buffer;
        return append( sizeInBytes, pow2Alignment );
    }

    template <typename T>
    inline T* append( size_t count, size_t pow2Alignment = 0 )
    {
        return (T*)append( count * sizeof( T ), pow2Alignment );
    }

    template <typename T>
    inline T* overlay( void* buffer, size_t count, size_t pow2Alignment = 0 )
    {
        return (T*)overlay( buffer, count * sizeof( T ), pow2Alignment );
    }

    // returns a pointer to the highest address allocated
    char* maxPtr() { return m_max; }


    size_t appendOffset( size_t sizeInBytes, size_t pow2Alignment = 0 )
    {
        return (char*)append( sizeInBytes, pow2Alignment ) - m_begin;
    }

    void* overlayOffset( size_t offset, size_t sizeInBytes = 0, size_t pow2Alignment = 0 )
    {
        return (char*)overlay( m_begin + offset, sizeInBytes, pow2Alignment );
    }

    template <typename T>
    inline size_t appendOffset( size_t count, size_t pow2Alignment = 0 )
    {
        return (char*)append<T>( count, pow2Alignment ) - m_begin;
    }

    template <typename T>
    inline size_t overlayOffset( size_t offset, size_t count, size_t pow2Alignment = 0 )
    {
        return (char*)overlay<T>( offset, count, pow2Alignment ) - m_begin;
    }

    size_t maxOffset() { return m_max - m_begin; }

  private:
    char* m_begin;
    char* m_end;
    char* m_max;
};
}
