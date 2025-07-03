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

#include <iosfwd>

namespace optix {

/// A stream that prints to std::ostream
///
/// Useful for printing out a human readable, formatted copy of a streamed object to log for debugging
class PrintStream : public PersistentStream
{
  public:
    PrintStream( std::ostream& out );

    void pushLabel( const char* label, const char* classname );
    void popLabel();
    void readOrWriteObjectVersion( const unsigned int* version ) override;
    void readOrWrite( char* data, size_t size, const char* label, Format format ) override;

  private:
    std::ostream& m_out;
    std::string   m_scope_indent;
};

}  // namespace optix
