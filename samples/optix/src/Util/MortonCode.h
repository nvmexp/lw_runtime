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


namespace optix {
// returns the lwrve index.
// x,y,z are the integer coordinates in 3D.
// nbits is the lwrve order.
template <typename Index_t>
Index_t HilbertIndex( Index_t x, Index_t y, Index_t z, int nbits )
{
    const int transform[8] = {0, 1, 7, 6, 3, 2, 4, 5};
    Index_t   s            = 0;
    for( int i = nbits - 1; i >= 0; i-- )
    {
        int xi = int( ( x ) >> i ) & 1;
        int yi = int( ( y ) >> i ) & 1;
        int zi = int( ( z ) >> i ) & 1;

        //change y and z
        if( xi == 0 && yi == 0 && zi == 0 )
        {
            Index_t temp = z;
            z            = y;
            y            = temp;
        }
        //change x and y
        else if( xi == 0 && yi == 0 && zi == 1 )
        {
            Index_t temp = x;
            x            = y;
            y            = temp;
        }
        //change x and y
        else if( xi == 1 && yi == 0 && zi == 1 )
        {
            Index_t temp = x;
            x            = y;
            y            = temp;
        }
        // complement z and x
        else if( xi == 1 && yi == 0 && zi == 0 )
        {
            x = ( x ) ^ ( -1 );
            z = ( z ) ^ ( -1 );
        }
        // complement z and x
        else if( xi == 1 && yi == 1 && zi == 0 )
        {
            x = ( x ) ^ ( -1 );
            z = ( z ) ^ ( -1 );
        }
        //change x and y and complement them
        else if( xi == 1 && yi == 1 && zi == 1 )
        {
            Index_t temp = ( x ) ^ ( -1 );
            x            = ( y ) ^ ( -1 );
            y            = temp;
        }
        //change x and y and complement them
        else if( xi == 0 && yi == 1 && zi == 1 )
        {
            Index_t temp = ( x ) ^ ( -1 );
            x            = ( y ) ^ ( -1 );
            y            = temp;
        }
        //xi==0, yi==1, zi==0
        //change z and y and complement them
        else
        {
            Index_t temp = ( z ) ^ ( -1 );
            x            = ( y ) ^ ( -1 );
            y            = temp;
        }
        int index = ( xi << 2 ) + ( yi << 1 ) + zi;
        s         = ( s << 3 ) + transform[index];
    }
    return s;
}

// P is inside the Box
template <typename Index_t>
Index_t hilbert_code( float3 P, float3 alpha, float3 beta )
{
    const int KeyMax  = sizeof( Index_t ) > sizeof( int ) ? 1048575 : 1023;
    const int LoopMax = sizeof( Index_t ) > sizeof( int ) ? 20 : 10;

    Index_t keyx = clamp( int( alpha.x * ( P.x - beta.x ) ), 0, KeyMax );
    Index_t keyy = clamp( int( alpha.y * ( P.y - beta.y ) ), 0, KeyMax );
    Index_t keyz = clamp( int( alpha.z * ( P.z - beta.z ) ), 0, KeyMax );

    return HilbertIndex( keyx, keyy, keyz, LoopMax );
}

template <typename Index_t>
Index_t morton_code( float3 P, float3 alpha, float3 beta )
{
    const int KeyMax  = sizeof( Index_t ) > sizeof( int ) ? 1048575 : 1023;
    const int LoopMax = sizeof( Index_t ) > sizeof( int ) ? 40 : 20;

    Index_t keyx = clamp( int( alpha.x * ( P.x - beta.x ) ), 0, KeyMax );
    Index_t keyy = clamp( int( alpha.y * ( P.y - beta.y ) ), 0, KeyMax );
    Index_t keyz = clamp( int( alpha.z * ( P.z - beta.z ) ), 0, KeyMax );

    Index_t codex = 0;
    Index_t codey = 0;
    Index_t codez = 0;

    for( int i = 0, andbit = 1; i < LoopMax; i += 2, andbit <<= 1 )
    {
        codex |= ( keyx & andbit ) << i;
        codey |= ( keyy & andbit ) << i;
        codez |= ( keyz & andbit ) << i;
    }

    return ( codez << 2 ) | ( codey << 1 ) | codex;
}
}  // namespace optix
