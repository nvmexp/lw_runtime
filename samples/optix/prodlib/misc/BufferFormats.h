// Copyright LWPU Corporation 2015
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

#include <algorithm>


namespace prodlib {

inline size_t getBufferTotalByteSize( size_t width, size_t height, size_t depth, size_t levels, size_t element_size, bool lwbeOrLayered )
{
    size_t size = 0;
    if( width * height * depth == 0 )
        return 0;

    const size_t one = 1ull;
    for( unsigned int level = 0; level < levels; ++level )
    {
        if( lwbeOrLayered )
            size += std::max( width >> level, one ) * std::max( height >> level, one ) * depth * element_size;
        else
            size += std::max( width >> level, one ) * std::max( height >> level, one ) * std::max( depth >> level, one ) * element_size;
    }
    return size;
}

inline size_t getBufferLevelByteSize( size_t width, size_t height, size_t depth, size_t level, size_t element_size, bool lwbeOrLayered )
{
    if( width * height * depth == 0 )
        return 0;

    const size_t one = 1ull;
    if( lwbeOrLayered )
        return std::max( width >> level, one ) * std::max( height >> level, one ) * depth * element_size;
    else
        return std::max( width >> level, one ) * std::max( height >> level, one ) * std::max( depth >> level, one ) * element_size;
}

inline size_t getBufferLevelWidth( size_t width, size_t level )
{
    return std::max( width >> level, static_cast<size_t>( 1ULL ) );
}

inline size_t getBufferLevelHeight( size_t height, size_t level )
{
    return std::max( height >> level, static_cast<size_t>( 1ULL ) );
}

inline size_t getBufferLevelDepth( size_t depth, size_t level, bool lwbeOrLayered )
{
    if( lwbeOrLayered )
        return depth;
    else
        return depth == 0 ? 0 : std::max( depth >> level, (size_t)1ull );
}

}  // end namespace prodlib
