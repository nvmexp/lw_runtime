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

#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace optix_exp {
class DeviceContextLogger;
}

namespace optix {
namespace lwca {
class ComputeCapability;
}

class PersistentStream
{
  public:
    virtual ~PersistentStream()                        = default;
    PersistentStream( const PersistentStream& stream ) = delete;
    PersistentStream& operator=( const PersistentStream& ) = delete;

    // Returns true if there has been an error reading or writing
    bool error() const;

    // Put the stream in an error state
    void setError();

    // Returns true if the stream is operating in the given mode.
    bool reading() const;
    bool writing() const;
    bool hashing() const;

    virtual void flush( optix_exp::DeviceContextLogger& ) {}

    // Returns the 32-character hexadecimal hash string if this is
    // a hashing stream. Returns an empty string otherwise.
    virtual std::string getDigestString() const;

    struct LabelHelper
    {
        LabelHelper( PersistentStream* stream )
            : stream( stream )
        {
        }
        ~LabelHelper() { stream->popLabel(); }
        PersistentStream* stream;
    };
    LabelHelper pushObject( const char* label, const char* classname )
    {
        pushLabel( label, classname );
        return LabelHelper( this );
    }

    enum Format
    {
        Opaque,
        None,
        String,
        Bool,
        Char,
        Int,
        UInt,
        Short,
        UShort,
        ULong,
        LongLong,
        ULongLong
    };

    // I/O functions
    virtual void pushLabel( const char* label, const char* classname );
    virtual void popLabel();
    virtual void readOrWriteObjectVersion( const unsigned int* version ) = 0;
    virtual void readOrWrite( char* data, size_t size, const char* label, Format format ) = 0;

  protected:
    enum Mode
    {
        Reading,
        Writing,
        Hashing
    };
    PersistentStream( Mode mode );
    bool m_error = false;

  private:
    Mode m_mode;
};

template <class T>
struct PersistentStreamFormat
{
};
template <>
struct PersistentStreamFormat<bool>
{
    static const PersistentStream::Format format = PersistentStream::Bool;
};
template <>
struct PersistentStreamFormat<char>
{
    static const PersistentStream::Format format = PersistentStream::Char;
};
template <>
struct PersistentStreamFormat<int>
{
    static const PersistentStream::Format format = PersistentStream::Int;
};
template <>
struct PersistentStreamFormat<unsigned int>
{
    static const PersistentStream::Format format = PersistentStream::UInt;
};
template <>
struct PersistentStreamFormat<short>
{
    static const PersistentStream::Format format = PersistentStream::Short;
};
template <>
struct PersistentStreamFormat<unsigned short>
{
    static const PersistentStream::Format format = PersistentStream::UShort;
};
template <>
struct PersistentStreamFormat<unsigned long>
{
    static const PersistentStream::Format format = PersistentStream::ULong;
};
template <>
struct PersistentStreamFormat<long long>
{
    static const PersistentStream::Format format = PersistentStream::LongLong;
};
template <>
struct PersistentStreamFormat<unsigned long long>
{
    static const PersistentStream::Format format = PersistentStream::ULongLong;
};
template <typename T>
struct PersistentStreamFormat<const T> : public PersistentStreamFormat<T>
{
};

// Useful, slightly dangerous function to strip const off of a
// pointer type. Useful for passing in objects to readOrWrite for
// writing and hashing.
template <class T>
T* deconst( const T* ptr )
{
    return const_cast<T*>( ptr );
}

// Read and write arithmetic types
template <class T>
void readOrWrite( PersistentStream* stream,
                  T*                value,
                  const char*       label,
                  typename std::enable_if<std::is_arithmetic<T>::value, void>::type* = nullptr )
{
    stream->readOrWrite( (char*)value, sizeof( T ), label, PersistentStreamFormat<T>::format );
}

// Read and write enums
template <class T>
void readOrWrite( PersistentStream* stream, T* value, const char* label, typename std::enable_if<std::is_enum<T>::value, void>::type* = nullptr )
{
    stream->readOrWrite( (char*)value, sizeof( T ), label, PersistentStream::Int );
}

// Read and write other optix types
void readOrWrite( PersistentStream* stream, lwca::ComputeCapability* value, const char* label );

// Read and write stl types
void readOrWrite( PersistentStream* stream, std::string* value, const char* label );
void readOrWrite( PersistentStream* stream, const std::string* value, const char* label );
void readOrWrite( PersistentStream* stream, std::vector<std::string>* value, const char* label );
void readOrWrite( PersistentStream* stream, std::vector<bool>* array, const char* label );
template <class T>
void readOrWrite( PersistentStream* stream,
                  std::vector<T>*   value,
                  const char*       label,
                  typename std::enable_if<std::is_arithmetic<T>::value, void>::type* = 0 )
{
    auto tmp = stream->pushObject( label, "vector" );
    if( stream->reading() )
    {
        size_t size;
        readOrWrite( stream, &size, "size" );
        if( stream->error() )
            return;
        value->resize( size );
    }
    else
    {
        size_t size = value->size();
        readOrWrite( stream, &size, "size" );
    }
    stream->readOrWrite( reinterpret_cast<char*>( value->data() ), sizeof( T ) * value->size(), "value", PersistentStreamFormat<T>::format );
}

template <class T, class Create>
void readOrWrite( PersistentStream* stream, std::vector<const T*>* array, Create createElement, const char* label )
{
    auto tmp = stream->pushObject( label, "vector" );
    if( stream->reading() )
    {
        // Read the size and check for errors
        size_t size = ~0ULL;
        readOrWrite( stream, &size, "size" );
        if( size == ~0ULL )
            stream->setError();
        if( stream->error() )
            return;

        // Resize the array
        array->resize( size );

        for( size_t i = 0; i < size; ++i )
        {
            // Allocate the element and then read it.
            ( *array )[i] = createElement();
            readOrWrite( stream, const_cast<T*>( ( *array )[i] ), "elt" );
        }
    }
    else
    {
        // Writing or hashing
        size_t size = array->size();
        readOrWrite( stream, &size, label );
        for( auto elt : *array )
            readOrWrite( stream, const_cast<T*>( elt ), "elt" );
    }
}

template <class T>
void readOrWrite( PersistentStream* stream, std::set<T>* array, const char* label )
{
    auto tmp = stream->pushObject( label, "set" );
    if( stream->reading() )
    {
        // Read the size and check for errors
        size_t size = ~0ULL;
        readOrWrite( stream, &size, "size" );
        if( size == ~0ULL )
            stream->setError();
        if( stream->error() )
            return;

        for( size_t i = 0; i < size; ++i )
        {
            // Allocate the element and then read it.
            T element;
            readOrWrite( stream, &element, "elt" );
            array->insert( element );
        }
    }
    else
    {
        // Writing or hashing
        size_t size = array->size();
        readOrWrite( stream, &size, label );
        for( auto elt : *array )
            readOrWrite( stream, &elt, "elt" );
    }
}

template <class T1, class T2>
void readOrWrite( PersistentStream* stream, std::pair<T1, T2>* pair, const char* label )
{
    auto tmp = stream->pushObject( label, "pair" );
    readOrWrite( stream, &pair->first, "first" );
    readOrWrite( stream, &pair->second, "second" );
}

template <class K, class V>
void readOrWrite( PersistentStream* stream, std::map<K, V>* map, const char* label )
{
    auto tmp = stream->pushObject( label, "map" );
    if( stream->reading() )
    {
        size_t size = ~0ULL;
        readOrWrite( stream, &size, "size" );
        if( size == ~0ULL )
            stream->setError();
        if( stream->error() )
            return;

        for( size_t i = 0; i < size; ++i )
        {
            typename std::map<K, V>::value_type item;
            readOrWrite( stream, &item, "elt" );
            map->emplace( item );
        }
    }
    else
    {
        size_t size = map->size();
        readOrWrite( stream, &size, "size" );
        for( auto elt : *map )
            readOrWrite( stream, &elt, "elt" );
    }
}
}
