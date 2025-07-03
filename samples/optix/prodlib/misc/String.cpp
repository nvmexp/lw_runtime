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

// For memmem()
#ifndef _WIN32
#undef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <prodlib/misc/String.h>

#include <cstring>

namespace prodlib {

const char* strNStr( const char* hayStack, const char* needle, size_t hayStackLen )
{
#ifndef _WIN32

    // Use existing memmem() which should be highly optimized.

    size_t needleLen = strlen( needle );
    return static_cast<const char*>( memmem( hayStack, hayStackLen, needle, needleLen ) );

#else

    // Not sure which implementation is the most efficient one. For now, use a combination of memchr() and memcmp().

    size_t needleLen = strlen( needle );

    if( needleLen == 0 )
        return hayStack;

    while( hayStackLen >= needleLen )
    {
        const char* initialMatch = static_cast<const char*>( memchr( hayStack, *needle, hayStackLen ) );
        if( !initialMatch )
            return nullptr;

        hayStackLen -= initialMatch - hayStack;
        hayStack = initialMatch;
        if( hayStackLen < needleLen )
            return nullptr;

        if( memcmp( hayStack, needle, needleLen ) == 0 )
            return hayStack;

        --hayStackLen;
        ++hayStack;
    }

    return nullptr;
#endif
}

}  // namespace prodlib
