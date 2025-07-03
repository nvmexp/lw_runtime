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

#include <LWCA/ComputeCapability.h>
#include <Util/PersistentStream.h>

using namespace optix;

bool PersistentStream::error() const
{
    return m_error;
}

void PersistentStream::setError()
{
    m_error = true;
}

bool PersistentStream::reading() const
{
    return m_mode == Reading;
}

bool PersistentStream::writing() const
{
    return m_mode == Writing;
}

bool PersistentStream::hashing() const
{
    return m_mode == Hashing;
}

std::string PersistentStream::getDigestString() const
{
    return "";
}

void PersistentStream::pushLabel( const char* label, const char* classname )
{
}

void PersistentStream::popLabel()
{
}

PersistentStream::PersistentStream( Mode mode )
{
    m_mode = mode;
}


void optix::readOrWrite( PersistentStream* stream, std::string* value, const char* label )
{
    if( stream->reading() )
    {
        int length = -1;
        readOrWrite( stream, &length, nullptr );
        if( length < 0 )
            stream->setError();
        if( stream->error() )
            return;
        value->resize( length );
    }
    else
    {
        int length = value->length();
        readOrWrite( stream, &length, nullptr );
    }
    stream->readOrWrite( &( *value )[0], value->length(), label, PersistentStream::String );
}

void optix::readOrWrite( PersistentStream* stream, const std::string* value, const char* label )
{
    readOrWrite( stream, deconst( value ), label );
}

void optix::readOrWrite( PersistentStream* stream, std::vector<std::string>* value, const char* label )
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
    size_t size = value->size();
    for( size_t i = 0; i < size; ++i )
    {
        readOrWrite( stream, &( *value )[i], "element" );
    }
}

void optix::readOrWrite( PersistentStream* stream, lwca::ComputeCapability* value, const char* label )
{
    if( stream->reading() )
    {
        int tmp = -999;
        readOrWrite( stream, &tmp, label );
        *value = lwca::ComputeCapability( tmp );
    }
    else
    {
        int tmp = value->version();
        readOrWrite( stream, &tmp, label );
    }
}

void optix::readOrWrite( PersistentStream* stream, std::vector<bool>* vector, const char* label )
{
    // vector<bool> is not necessarily contiguous, so write one
    // element at a time.  There may be more efficient ways to do this
    // if we need to write large vectors.
    auto tmp = stream->pushObject( label, "vector" );
    if( stream->reading() )
    {
        int n = -1;
        readOrWrite( stream, &n, "size" );
        if( n < 0 )
            stream->setError();
        if( stream->error() )
            return;
        vector->resize( n );
        for( int i = 0; i < n; ++i )
        {
            char b;
            readOrWrite( stream, &b, "elt" );
            ( *vector )[i] = !!b;
        }
    }
    else
    {
        int n = vector->size();
        readOrWrite( stream, &n, "size" );
        for( int i = 0; i < n; ++i )
        {
            char b = ( *vector )[i];
            readOrWrite( stream, &b, "elt" );
        }
    }
}
