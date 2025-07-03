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

#include <Util/MemoryStream.h>
#include <Util/optixUuid.h>

#include <cstring>

#include <prodlib/exceptions/Assert.h>

namespace optix {

MemoryWriter::MemoryWriter()
    : PersistentStream( Writing )
{
}

void MemoryWriter::readOrWriteObjectVersion( const unsigned int* version )
{
    optix::readOrWrite( this, &version[0], "version[0]" );
    optix::readOrWrite( this, &version[1], "version[1]" );
    optix::readOrWrite( this, &version[2], "version[2]" );
    optix::readOrWrite( this, &version[3], "version[3]" );
}

void MemoryWriter::readOrWrite( char* data, size_t size, const char* /*label*/, Format /*format*/ )
{
    if( m_error )
        return;

    size_t oldSize = m_buffer.size();
    m_buffer.resize( oldSize + size );
    memcpy( &m_buffer[0] + oldSize, data, size );
}

void readOrWrite( PersistentStream* stream, MemoryWriter* writer, const char* label )
{
    RT_ASSERT( !stream->reading() );

    auto tmp = stream->pushObject( label, "MemoryReader/Writer" );

    static const unsigned int* version = getOptixUUID();
    stream->readOrWriteObjectVersion( version );

    size_t bufferSize = writer->getBufferSize();
    readOrWrite( stream, &bufferSize, "bufferSize" );

    stream->readOrWrite( writer->getBuffer(), bufferSize, "buffer", PersistentStream::Opaque );
}

MemoryReader::MemoryReader()
    : PersistentStream( Reading )
{
}

void MemoryReader::readOrWriteObjectVersion( const unsigned int* expectedVersion )
{
    unsigned int version[4] = {0, 0, 0, 0};
    optix::readOrWrite( this, &version[0], "version[0]" );
    optix::readOrWrite( this, &version[1], "version[1]" );
    optix::readOrWrite( this, &version[2], "version[2]" );
    optix::readOrWrite( this, &version[3], "version[3]" );

    if( version[0] != expectedVersion[0] || version[1] != expectedVersion[1] || version[2] != expectedVersion[2]
        || version[3] != expectedVersion[3] )
        m_error = true;
}

void MemoryReader::readOrWrite( char* data, size_t size, const char* /*label*/, Format /*format*/ )
{
    if( m_error )
        return;

    if( size > m_buffer.size() - m_offset )
    {
        m_error = true;
        return;
    }

    memcpy( data, &m_buffer[m_offset], size );
    m_offset += size;
}

void MemoryReader::resize( size_t size )
{
    m_buffer.resize( size );
    m_offset = 0;
}

void readOrWrite( PersistentStream* stream, MemoryReader* reader, const char* label )
{
    RT_ASSERT( stream->reading() );

    auto tmp = stream->pushObject( label, "MemoryReader/Writer" );

    static const unsigned int* version = getOptixUUID();
    stream->readOrWriteObjectVersion( version );

    size_t bufferSize = 0;  // initialize in case the stream has an error
    readOrWrite( stream, &bufferSize, "bufferSize" );

    reader->resize( bufferSize );
    stream->readOrWrite( reader->getBuffer(), bufferSize, "buffer", PersistentStream::Opaque );
}
}
