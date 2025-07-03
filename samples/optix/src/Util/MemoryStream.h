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

#include <Util/PersistentStream.h>

#include <vector>

namespace optix {

/// A stream that serializes to its memory buffer.
///
/// Note that the underlying stream is \em not persistent despite the name of its base class (which
/// should probably be renamed to Stream).
class MemoryWriter : public PersistentStream
{
  public:
    MemoryWriter();

    /// Returns the buffer contents (to be used after serialization).
    char* getBuffer() { return m_buffer.data(); }
    /// Returns the buffer size (to be used after serialization).
    size_t getBufferSize() const { return m_buffer.size(); }

    void readOrWriteObjectVersion( const unsigned int* version ) override;
    void readOrWrite( char* data, size_t size, const char* label, Format format ) override;

  private:
    std::vector<char> m_buffer;
};

/// Serializes a MemoryWriter to the stream. Supports only writing and hashing.
void readOrWrite( PersistentStream* stream, MemoryWriter* writer, const char* label );

/// A stream that deserializes from its memory buffer.
///
/// Note that this underlying stream is \em not persistent despite the name of its base class (which
/// should probably be renamed to Stream).
struct MemoryReader : public PersistentStream
{
  public:
    MemoryReader();

    /// Returns the buffer contents (can be used before deserialization to manipulate the buffer).
    char* getBuffer() { return m_buffer.data(); }
    /// Returns the buffer size (can be used before deserialization to manipulate the buffer).
    size_t getBufferSize() const { return m_buffer.size(); }

    friend void readOrWrite( PersistentStream* stream, MemoryReader* buffer, const char* label );

    void readOrWriteObjectVersion( const unsigned int* expectedVersion ) override;
    void readOrWrite( char* data, size_t size, const char* label, Format format ) override;

  private:
    void resize( size_t size );  // used by readOrWrite()

    std::vector<char> m_buffer;
    size_t            m_offset = 0;
};

/// Deserializes a MemoryReader from the stream. Supports only reading.
void readOrWrite( PersistentStream* stream, MemoryReader* reader, const char* label );
}
