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

#include <FrontEnd/PTX/PTXStitch/std/stdTypes.h>

#include <optixu/optixu_aabb.h>
#include <optixu/optixu_math.h>

#include <map>
#include <string.h>
#include <string>
#include <vector>

namespace optix {

//
// Serializer.
//

class Serializer
{
  public:
    // Create a dummy serializer, used only for data size computation.
    Serializer()
        : m_size( 0 )
        , m_data( nullptr )
    {
    }

    // Create a serializer with a target pointer, for actual serialization.
    Serializer( void* data )
        : m_size( 0 )
        , m_data( (unsigned char*)data )
    {
    }

    // Add data to serializer.
    void write( char value ) { push( value ); }
    void write( signed char value ) { push( value ); }
    void write( unsigned char value ) { push( value ); }
    void write( Int16 value ) { push( value ); }
    void write( uInt16 value ) { push( value ); }
    void write( Int32 value ) { push( value ); }
    void write( uInt32 value ) { push( value ); }
    void write( Int64 value ) { push( value ); }
    void write( uInt64 value ) { push( value ); }
    void write( float value ) { push( value ); }
    void write( double value ) { push( value ); }

    // Get data size.
    size_t getSize() const { return m_size; }

    // Get data pointer.
    const void* getData() const { return m_data; }

    // Get current write pointer.
    void* getWriteLocation() const
    {
        if( !m_data )
            return nullptr;
        return &m_data[m_size];
    }

  private:
    // Actually push data into the serializer. This is private so there is
    // no temptation to push platform-dependent types (e.g. padded structs).
    template <typename T>
    void push( T value )
    {
        if( m_data )
        {
            memcpy( &m_data[m_size], &value, sizeof( T ) );
        }
        m_size += sizeof( T );
    }

    size_t         m_size;  // current size of serialized data
    unsigned char* m_data;  // serialized data target
};


//
// Deserializer.
//

class Deserializer
{
  public:
    // Create a deserializer. Will not make a copy of the specified data,
    // so the pointer has to stay valid while the deserializer is in use.
    Deserializer( const void* data )
        : m_data( data )
        , m_lwr( 0 )
    {
    }

    // Read data from deserializer.
    void read( char& value ) { pull( value ); }
    void read( signed char& value ) { pull( value ); }
    void read( unsigned char& value ) { pull( value ); }
    void read( Int16& value ) { pull( value ); }
    void read( uInt16& value ) { pull( value ); }
    void read( Int32& value ) { pull( value ); }
    void read( uInt32& value ) { pull( value ); }
    void read( Int64& value ) { pull( value ); }
    void read( uInt64& value ) { pull( value ); }
    void read( float& value ) { pull( value ); }
    void read( double& value ) { pull( value ); }

  private:
    // Actually pull data from the serializer. This is private so there is
    // no temptation to pull platform-dependent types (e.g. padded structs).
    template <typename T>
    void pull( T& value )
    {
        memcpy( &value, static_cast<const char* const>( m_data ) + m_lwr, sizeof( T ) );
        m_lwr += sizeof( T );
    }

    const void* m_data;  // data pointer
    size_t      m_lwr;   // current position for deserialization
};


//
// Serialization helpers.
//

// Basic serialize function for both native and non-native types.
// Non-native types are required to provide a 'serialize' member function.
template <typename T>
inline void serialize( Serializer& serializer, const T& val )
{
    val.serialize( serializer );
}
template <>
inline void serialize<char>( Serializer& serializer, const char& val )
{
    serializer.write( val );
}
template <>
inline void serialize<unsigned char>( Serializer& serializer, const unsigned char& val )
{
    serializer.write( val );
}
template <>
inline void serialize<signed char>( Serializer& serializer, const signed char& val )
{
    serializer.write( val );
}
template <>
inline void serialize<Int16>( Serializer& serializer, const Int16& val )
{
    serializer.write( val );
}
template <>
inline void serialize<uInt16>( Serializer& serializer, const uInt16& val )
{
    serializer.write( val );
}
template <>
inline void serialize<Int32>( Serializer& serializer, const Int32& val )
{
    serializer.write( val );
}
template <>
inline void serialize<uInt32>( Serializer& serializer, const uInt32& val )
{
    serializer.write( val );
}
template <>
inline void serialize<Int64>( Serializer& serializer, const Int64& val )
{
    serializer.write( val );
}
template <>
inline void serialize<uInt64>( Serializer& serializer, const uInt64& val )
{
    serializer.write( val );
}
template <>
inline void serialize<float>( Serializer& serializer, const float& val )
{
    serializer.write( val );
}
template <>
inline void serialize<double>( Serializer& serializer, const double& val )
{
    serializer.write( val );
}
template <>
inline void serialize<bool>( Serializer& serializer, const bool& val )
{
    serializer.write( (Int32)val );
}

// Serialization of size_t is always done using a 64-bit uint.
// We can't overload serialize with size_t again, so provide a separate function.
inline void serializeSizeT( Serializer& serializer, size_t val )
{
    serialize( serializer, static_cast<uInt64>( val ) );
}

inline void serialize( Serializer& serializer, const float1& val )
{
    serialize( serializer, val.x );
}

inline void serialize( Serializer& serializer, const float2& val )
{
    serialize( serializer, val.x );
    serialize( serializer, val.y );
}

inline void serialize( Serializer& serializer, const float3& val )
{
    serialize( serializer, val.x );
    serialize( serializer, val.y );
    serialize( serializer, val.z );
}

inline void serialize( Serializer& serializer, const float4& val )
{
    serialize( serializer, val.x );
    serialize( serializer, val.y );
    serialize( serializer, val.z );
    serialize( serializer, val.w );
}

template <typename T>
inline void serialize( Serializer& serializer, const std::vector<T>& val )
{
    serializeSizeT( serializer, val.size() );
    for( size_t i = 0; i < val.size(); ++i )
        serialize( serializer, val[i] );
}

template <typename T1, typename T2>
inline void serialize( Serializer& serializer, const std::map<T1, T2>& val )
{
    serializeSizeT( serializer, val.size() );
    for( typename std::map<T1, T2>::const_iterator it = val.begin(); it != val.end(); ++it )
    {
        serialize( serializer, it->first );
        serialize( serializer, it->second );
    }
}

inline void serialize( Serializer& serializer, const std::string& val )
{
    serializeSizeT( serializer, val.size() );
    for( char i : val )
        serialize( serializer, i );
}

inline void serialize( Serializer& serializer, const Aabb& val )
{
    serialize( serializer, val.m_min );
    serialize( serializer, val.m_max );
}


//
// Deserialization helpers.
//

// Basic deserialize function for both native and non-native types.
// Non-native types are required to provide a 'deserialize' member function.
template <typename T>
inline void deserialize( Deserializer& deserializer, T& val )
{
    val.deserialize( deserializer );
}
template <>
inline void deserialize<char>( Deserializer& deserializer, char& val )
{
    deserializer.read( val );
}
template <>
inline void deserialize<unsigned char>( Deserializer& deserializer, unsigned char& val )
{
    deserializer.read( val );
}
template <>
inline void deserialize<signed char>( Deserializer& deserializer, signed char& val )
{
    deserializer.read( val );
}
template <>
inline void deserialize<Int16>( Deserializer& deserializer, Int16& val )
{
    deserializer.read( val );
}
template <>
inline void deserialize<uInt16>( Deserializer& deserializer, uInt16& val )
{
    deserializer.read( val );
}
template <>
inline void deserialize<Int32>( Deserializer& deserializer, Int32& val )
{
    deserializer.read( val );
}
template <>
inline void deserialize<uInt32>( Deserializer& deserializer, uInt32& val )
{
    deserializer.read( val );
}
template <>
inline void deserialize<Int64>( Deserializer& deserializer, Int64& val )
{
    deserializer.read( val );
}
template <>
inline void deserialize<uInt64>( Deserializer& deserializer, uInt64& val )
{
    deserializer.read( val );
}
template <>
inline void deserialize<float>( Deserializer& deserializer, float& val )
{
    deserializer.read( val );
}
template <>
inline void deserialize<double>( Deserializer& deserializer, double& val )
{
    deserializer.read( val );
}
template <>
inline void deserialize<bool>( Deserializer& deserializer, bool& val )
{
    Int32 tmp;
    deserializer.read( tmp );
    val = tmp != 0;
}

// Serialization of size_t is always done using a 64-bit uint.
// We can't overload deserialize with size_t again, so provide a separate function.
inline void deserializeSizeT( Deserializer& deserializer, size_t& val )
{
    uInt64 temp;
    deserialize( deserializer, temp );
    val = static_cast<size_t>( temp );
}

inline void deserialize( Deserializer& deserializer, float1& val )
{
    deserialize( deserializer, val.x );
}

inline void deserialize( Deserializer& deserializer, float2& val )
{
    deserialize( deserializer, val.x );
    deserialize( deserializer, val.y );
}

inline void deserialize( Deserializer& deserializer, float3& val )
{
    deserialize( deserializer, val.x );
    deserialize( deserializer, val.y );
    deserialize( deserializer, val.z );
}

inline void deserialize( Deserializer& deserializer, float4& val )
{
    deserialize( deserializer, val.x );
    deserialize( deserializer, val.y );
    deserialize( deserializer, val.z );
    deserialize( deserializer, val.w );
}

template <typename T>
inline void deserialize( Deserializer& deserializer, std::vector<T>& val )
{
    size_t size;
    deserializeSizeT( deserializer, size );
    val.resize( size );
    for( size_t i = 0; i < size; ++i )
        deserialize( deserializer, val[i] );
}

template <typename T1, typename T2>
inline void deserialize( Deserializer& deserializer, std::map<T1, T2>& val )
{
    size_t size;
    deserializeSizeT( deserializer, size );
    val.clear();
    for( size_t i = 0; i < size; ++i )
    {
        T1 first;
        T2 second;
        deserialize( deserializer, first );
        deserialize( deserializer, second );
        val.insert( std::make_pair( first, second ) );
    }
}

inline void deserialize( Deserializer& deserializer, std::string& val )
{
    size_t size;
    deserializeSizeT( deserializer, size );
    val.resize( size );
    for( size_t i = 0; i < size; ++i )
        deserialize( deserializer, val[i] );
}

inline void deserialize( Deserializer& deserializer, Aabb& val )
{
    deserialize( deserializer, val.m_min );
    deserialize( deserializer, val.m_max );
}

}  // namespace optix
