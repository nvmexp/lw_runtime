// Copyright LWPU Corporation 2019
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

#include <cstring>
#include <string>

#include <stddef.h>

namespace prodlib {

// a minimal implementation which should be replaced by std::string_view as soon as C++17 can be used
// by rtcore / OptiX
class StringView
{
  public:
    StringView() = default;

    StringView( const char* data, size_t len )
        : m_begin( data )
        , m_size( len )
    {
    }

    const char* data() const { return m_begin; }
    size_t      size() const { return m_size; }

  private:
    const char* m_begin = nullptr;
    size_t      m_size  = 0U;
};

inline StringView toStringView( const char* cStr )
{
    return {cStr, std::strlen( cStr )};
}

inline StringView toStringView( const std::string& str )
{
    return {str.c_str(), str.size()};
}

// Locates the first oclwrence of the null-terminated string \p needle in the string \p hayStack, but does not consider
// more than \p hayStackLen characters from \p haystack.
//
// The function behaves as if hayStack[hayStackLen] == '\0', but without ever accessing hayStack[hayStackLen] (or any
// other charachters beyond this one).
//
// Returns the location of the first oclwrence, or \c nullptr if such location exists. Returns \p hayStack if \p needle
// is the empty string (the empty string matches everywhere). Otherwise, returns \c nullptr if \p haystack is the
// empty string.
//
// This function is a variant of strstr() for non-null-terminated hay stacks. Its name is derived from strnstr() in the
// BSD C library.
const char* strNStr( const char* hayStack, const char* needle, size_t hayStackLen );

}  // namespace prodlib
