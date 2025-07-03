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

#include <optixu/optixu_matrix.h>

namespace optix {

int findRealRoots4( const float coeffs[5], /*out*/ float roots[4] );

// Stores coefficients of a linear function of px,py,pz.
// See PBRT-v3 section 2.9.
struct SrtDerivativeTerm
{
    float kx, ky, kz, kc;
    SrtDerivativeTerm()
        : kx( 0 )
        , ky( 0 )
        , kz( 0 )
        , kc( 0 )
    {
    }
    SrtDerivativeTerm( float x, float y, float z, float c )
        : kx( x )
        , ky( y )
        , kz( z )
        , kc( c )
    {
    }
    float eval( float3 p ) const { return kx * p.x + ky * p.y + kz * p.z + kc; }
};

void makeSrtDerivativeTerms( const float*      matrix0,
                             const float*      matrix1,
                             float4            quat0,
                             float4            quat1,
                             SrtDerivativeTerm c0[3],
                             SrtDerivativeTerm c1[3],
                             SrtDerivativeTerm c2[3],
                             SrtDerivativeTerm c3[3],
                             SrtDerivativeTerm c4[3] );

}  // namespace optix
