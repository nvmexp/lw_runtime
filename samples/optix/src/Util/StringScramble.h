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

#include <string>


//
// Utility to hide a string from the exelwtable. We can't have string template
// parameters, so this makes use of the fact that char liteals can be up to 4 bytes.
//
// Usage:   std::string blah = hide_string<'Hell','o ','Opti','X'>();
//

namespace optix {

template <unsigned int I0>
inline std::string     scramble_string()
{
    // this is already all the magic:
    const unsigned int i0 = I0 ^ 0xFFFFFFFF;

    std::string ret;
    for( int i = 0; i < 4; ++i )
    {
        const unsigned int ii = ( i0 >> ( ( 3 - i ) * 8 ) ) & 0xFF;
        if( ii != 0xFF )  // don't push bytes that were all zero
            ret.push_back( static_cast<char>( ii ) );
    }
    return ret;
}
// Multi-argument versions
template <unsigned int I0, unsigned int I1>
inline std::string scramble_string()
{
    return scramble_string<I0>() + scramble_string<I1>();
}
template <unsigned int I0, unsigned int I1, unsigned int I2>
inline std::string scramble_string()
{
    return scramble_string<I0>() + scramble_string<I1>() + scramble_string<I2>();
}
template <unsigned int I0, unsigned int I1, unsigned int I2, unsigned int I3>
inline std::string scramble_string()
{
    return scramble_string<I0>() + scramble_string<I1>() + scramble_string<I2>() + scramble_string<I3>();
}
template <unsigned int I0, unsigned int I1, unsigned int I2, unsigned int I3, unsigned int I4>
inline std::string scramble_string()
{
    return scramble_string<I0>() + scramble_string<I1>() + scramble_string<I2>() + scramble_string<I3>() + scramble_string<I4>();
}
template <unsigned int I0, unsigned int I1, unsigned int I2, unsigned int I3, unsigned int I4, unsigned int I5>
inline std::string scramble_string()
{
    return scramble_string<I0>() + scramble_string<I1>() + scramble_string<I2>() + scramble_string<I3>()
           + scramble_string<I4>() + scramble_string<I5>();
}

// Ilwerse of scramble
inline std::string unscramble_string( const std::string& s )
{
    std::string ret;
    for( unsigned int ii : s )
    {
        ii ^= 0xFF;
        ret.push_back( static_cast<char>( ii ) );
    }
    return ret;
}

// Shortlwt for hiding a string by scrambling and unscrambling it
template <unsigned int I0>
inline std::string     hide_string()
{
    return unscramble_string( scramble_string<I0>() );
}
template <unsigned int I0, unsigned int I1>
inline std::string hide_string()
{
    return unscramble_string( scramble_string<I0, I1>() );
}
template <unsigned int I0, unsigned int I1, unsigned int I2>
inline std::string hide_string()
{
    return unscramble_string( scramble_string<I0, I1, I2>() );
}
template <unsigned int I0, unsigned int I1, unsigned int I2, unsigned int I3>
inline std::string hide_string()
{
    return unscramble_string( scramble_string<I0, I1, I2, I3>() );
}
template <unsigned int I0, unsigned int I1, unsigned int I2, unsigned int I3, unsigned int I4>
inline std::string hide_string()
{
    return unscramble_string( scramble_string<I0, I1, I2, I3, I4>() );
}
template <unsigned int I0, unsigned int I1, unsigned int I2, unsigned int I3, unsigned int I4, unsigned int I5>
inline std::string hide_string()
{
    return unscramble_string( scramble_string<I0, I1, I2, I3, I4, I5>() );
}

}  // end namespace optix
