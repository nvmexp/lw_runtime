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

#include <Objects/Buffer.h>
#include <Objects/TextureSampler.h>
#include <lwca.h>

#include <o6/optix.h>
#include <vector_types.h>


namespace optix {

// clang-format off
// Set the buffer format for T.
template<typename T> inline void setBufferFormat      ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_USER); buffer->setElementSize( sizeof(T) ); }
template<> inline void setBufferFormat<float>         ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_FLOAT); }
template<> inline void setBufferFormat<float1>        ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_FLOAT); }
template<> inline void setBufferFormat<float2>        ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_FLOAT2); }
template<> inline void setBufferFormat<float3>        ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_FLOAT3); }
template<> inline void setBufferFormat<float4>        ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_FLOAT4); }
template<> inline void setBufferFormat<char>          ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_BYTE); }
template<> inline void setBufferFormat<char1>         ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_BYTE); }
template<> inline void setBufferFormat<char2>         ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_BYTE2); }
template<> inline void setBufferFormat<char3>         ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_BYTE3); }
template<> inline void setBufferFormat<char4>         ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_BYTE4); }
template<> inline void setBufferFormat<unsigned char> ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_UNSIGNED_BYTE); }
template<> inline void setBufferFormat<uchar1>        ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_UNSIGNED_BYTE); }
template<> inline void setBufferFormat<uchar2>        ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_UNSIGNED_BYTE2); }
template<> inline void setBufferFormat<uchar3>        ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_UNSIGNED_BYTE3); }
template<> inline void setBufferFormat<uchar4>        ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_UNSIGNED_BYTE4); }
template<> inline void setBufferFormat<short>         ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_SHORT); }
template<> inline void setBufferFormat<short1>        ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_SHORT); }
template<> inline void setBufferFormat<short2>        ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_SHORT2); }
template<> inline void setBufferFormat<short3>        ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_SHORT3); }
template<> inline void setBufferFormat<short4>        ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_SHORT4); }
template<> inline void setBufferFormat<unsigned short>( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_UNSIGNED_SHORT); }
template<> inline void setBufferFormat<ushort1>       ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_UNSIGNED_SHORT); }
template<> inline void setBufferFormat<ushort2>       ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_UNSIGNED_SHORT2); }
template<> inline void setBufferFormat<ushort3>       ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_UNSIGNED_SHORT3); }
template<> inline void setBufferFormat<ushort4>       ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_UNSIGNED_SHORT4); }
template<> inline void setBufferFormat<int>           ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_INT); }
template<> inline void setBufferFormat<int1>          ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_INT); }
template<> inline void setBufferFormat<int2>          ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_INT2); }
template<> inline void setBufferFormat<int3>          ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_INT3); }
template<> inline void setBufferFormat<int4>          ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_INT4); }
template<> inline void setBufferFormat<unsigned int>  ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_UNSIGNED_INT); }
template<> inline void setBufferFormat<uint1>         ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_UNSIGNED_INT); }
template<> inline void setBufferFormat<uint2>         ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_UNSIGNED_INT2); }
template<> inline void setBufferFormat<uint3>         ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_UNSIGNED_INT3); }
template<> inline void setBufferFormat<uint4>         ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_UNSIGNED_INT4); }
template<> inline void setBufferFormat<long long>(Buffer* buffer) { buffer->setFormat(RT_FORMAT_LONG_LONG); }
template<> inline void setBufferFormat<longlong1>(Buffer* buffer) { buffer->setFormat(RT_FORMAT_LONG_LONG); }
template<> inline void setBufferFormat<longlong2>(Buffer* buffer) { buffer->setFormat(RT_FORMAT_LONG_LONG2); }
template<> inline void setBufferFormat<longlong3>(Buffer* buffer) { buffer->setFormat(RT_FORMAT_LONG_LONG3); }
template<> inline void setBufferFormat<longlong4>(Buffer* buffer) { buffer->setFormat(RT_FORMAT_LONG_LONG4); }
template<> inline void setBufferFormat<unsigned long long >(Buffer* buffer) { buffer->setFormat(RT_FORMAT_UNSIGNED_LONG_LONG); }
template<> inline void setBufferFormat<ulonglong1>(Buffer* buffer) { buffer->setFormat(RT_FORMAT_UNSIGNED_LONG_LONG); }
template<> inline void setBufferFormat<ulonglong2>(Buffer* buffer) { buffer->setFormat(RT_FORMAT_UNSIGNED_LONG_LONG2); }
template<> inline void setBufferFormat<ulonglong3>(Buffer* buffer) { buffer->setFormat(RT_FORMAT_UNSIGNED_LONG_LONG3); }
template<> inline void setBufferFormat<ulonglong4>(Buffer* buffer) { buffer->setFormat(RT_FORMAT_UNSIGNED_LONG_LONG4); }
template<> inline void setBufferFormat<RTbuffer>      ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_BUFFER_ID); }
template<> inline void setBufferFormat<RTprogram>     ( Buffer* buffer ) { buffer->setFormat(RT_FORMAT_PROGRAM_ID); }

// Check if a buffer has the correct format for T.
template<typename T> inline bool checkBufferFormat      ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_USER && buffer->getElementSize() == sizeof(T); }
template<> inline bool checkBufferFormat<float>         ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_FLOAT; }
template<> inline bool checkBufferFormat<float1>        ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_FLOAT; }
template<> inline bool checkBufferFormat<float2>        ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_FLOAT2; }
template<> inline bool checkBufferFormat<float3>        ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_FLOAT3; }
template<> inline bool checkBufferFormat<float4>        ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_FLOAT4; }
template<> inline bool checkBufferFormat<char>          ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_BYTE; }
template<> inline bool checkBufferFormat<char1>         ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_BYTE; }
template<> inline bool checkBufferFormat<char2>         ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_BYTE2; }
template<> inline bool checkBufferFormat<char3>         ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_BYTE3; }
template<> inline bool checkBufferFormat<char4>         ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_BYTE4; }
template<> inline bool checkBufferFormat<unsigned char> ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_BYTE; }
template<> inline bool checkBufferFormat<uchar1>        ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_BYTE; }
template<> inline bool checkBufferFormat<uchar2>        ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_BYTE2; }
template<> inline bool checkBufferFormat<uchar3>        ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_BYTE3; }
template<> inline bool checkBufferFormat<uchar4>        ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_BYTE4; }
template<> inline bool checkBufferFormat<short>         ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_SHORT; }
template<> inline bool checkBufferFormat<short1>        ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_SHORT; }
template<> inline bool checkBufferFormat<short2>        ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_SHORT2; }
template<> inline bool checkBufferFormat<short3>        ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_SHORT3; }
template<> inline bool checkBufferFormat<short4>        ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_SHORT4; }
template<> inline bool checkBufferFormat<unsigned short>( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_SHORT; }
template<> inline bool checkBufferFormat<ushort1>       ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_SHORT; }
template<> inline bool checkBufferFormat<ushort2>       ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_SHORT2; }
template<> inline bool checkBufferFormat<ushort3>       ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_SHORT3; }
template<> inline bool checkBufferFormat<ushort4>       ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_SHORT4; }
template<> inline bool checkBufferFormat<int>           ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_INT; }
template<> inline bool checkBufferFormat<int1>          ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_INT; }
template<> inline bool checkBufferFormat<int2>          ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_INT2; }
template<> inline bool checkBufferFormat<int3>          ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_INT3; }
template<> inline bool checkBufferFormat<int4>          ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_INT4; }
template<> inline bool checkBufferFormat<unsigned int>  ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_INT; }
template<> inline bool checkBufferFormat<uint1>         ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_INT; }
template<> inline bool checkBufferFormat<uint2>         ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_INT2; }
template<> inline bool checkBufferFormat<uint3>         ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_INT3; }
template<> inline bool checkBufferFormat<uint4>         ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_INT4; }
template<> inline bool checkBufferFormat<long long>(const Buffer* buffer) { return buffer->getFormat() == RT_FORMAT_LONG_LONG; }
template<> inline bool checkBufferFormat<longlong1>(const Buffer* buffer) { return buffer->getFormat() == RT_FORMAT_LONG_LONG; }
template<> inline bool checkBufferFormat<longlong2>(const Buffer* buffer) { return buffer->getFormat() == RT_FORMAT_LONG_LONG2; }
template<> inline bool checkBufferFormat<longlong3>(const Buffer* buffer) { return buffer->getFormat() == RT_FORMAT_LONG_LONG3; }
template<> inline bool checkBufferFormat<longlong4>(const Buffer* buffer) { return buffer->getFormat() == RT_FORMAT_LONG_LONG4; }
template<> inline bool checkBufferFormat<unsigned long long>(const Buffer* buffer) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_LONG_LONG; }
template<> inline bool checkBufferFormat<ulonglong1>(const Buffer* buffer) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_LONG_LONG; }
template<> inline bool checkBufferFormat<ulonglong2>(const Buffer* buffer) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_LONG_LONG2; }
template<> inline bool checkBufferFormat<ulonglong3>(const Buffer* buffer) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_LONG_LONG3; }
template<> inline bool checkBufferFormat<ulonglong4>(const Buffer* buffer) { return buffer->getFormat() == RT_FORMAT_UNSIGNED_LONG_LONG4; }
template<> inline bool checkBufferFormat<RTbuffer>      ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_BUFFER_ID; }
template<> inline bool checkBufferFormat<RTprogram>     ( const Buffer* buffer ) { return buffer->getFormat() == RT_FORMAT_PROGRAM_ID; }
// clang-format on

// Helper to create a new T buffer with the given parameters.
template <typename T>
Buffer* createBuffer( Context* context, int buffer_dim = 1, unsigned int buffer_type = RT_BUFFER_INPUT )
{
    Buffer::checkBufferType( buffer_type, false );
    Buffer* buffer = new Buffer( context, buffer_type );
    setBufferFormat<T>( buffer );
    switch( buffer_dim )
    {
        case 1:
            buffer->setSize1D( 0 );
            break;
        case 2:
            buffer->setSize2D( 0, 0 );
            break;
        case 3:
            buffer->setSize3D( 0, 0, 0 );
            break;
        default:
            RT_ASSERT( !!!"invalid buffer dimension" );
    }
    return buffer;
}

// Wrap the mapping/unmapping of a buffer object.
// The template parameter is the element type of the buffer.
template <typename T>
class MappedBuffer
{
  public:
    // Set a new buffer size if one is specified, then map the buffer.
    MappedBuffer( Buffer* buffer, MapMode mode, size_t size_x )
        : m_buffer( buffer )
        , m_mapped( true )
    {
        RT_ASSERT( m_buffer );
        RT_ASSERT( checkBufferFormat<T>( m_buffer ) );

        if( m_buffer->getWidth() != size_x )
        {
            // If the buffer is being mapped read only, it doesn't make sense
            // to resize it at the same time, as that will cause an allocation.
            RT_ASSERT( mode != MAP_READ );

            m_buffer->setSize1D( size_x );
        }

        m_ptr = static_cast<T*>( m_buffer->map( mode ) );
    }

    // Automatically unmap the buffer on destruction.
    ~MappedBuffer() { unmap(); }

    // Return a pointer to the mapped buffer.
    T* ptr() const
    {
        RT_ASSERT( m_mapped );
        return m_ptr;
    }

    // Unmap the buffer.
    void unmap()
    {
        if( m_mapped )
        {
            m_buffer->unmap();
            m_mapped = false;
        }
    }

    Buffer* getBuffer() const { return m_buffer; }

  private:
    Buffer* m_buffer;
    bool    m_mapped;
    T*      m_ptr;
};

}  // namespace optix
