/*
 * Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <vector_types.h>

namespace optix_exp {
namespace motion {

// Find the roots of a degree (up to) 4 polynomial.
// Returns the number of real roots.
// Only real ones are collected and stored, complex roots are omitted.
M_DEVICE_HOST __noinline__ static int findRealRoots4( const float coeffs[5], /*out*/ float roots[4] );

// Stores coefficients of a linear function of px,py,pz.
// See PBRT-v3 section 2.9.
struct SRNumeratorDerivativeTerm
{
    float         kx, ky, kz, kc;
    M_DEVICE_HOST SRNumeratorDerivativeTerm()
        : kx( 0 )
        , ky( 0 )
        , kz( 0 )
        , kc( 0 )
    {
    }
    M_DEVICE_HOST SRNumeratorDerivativeTerm( float x, float y, float z, float c )
        : kx( x )
        , ky( y )
        , kz( z )
        , kc( c )
    {
    }
    M_DEVICE_HOST float eval( const float3& p ) const { return kx * p.x + ky * p.y + kz * p.z + kc; }
};

// The derivate of the SR interpolation of a fixed point is a rational polynomial with degree 4 in the numerator and denominator.
// The coefficients of the numerator depend on the point to transform, which is not yet known here.
// The coefficients of the denominator only depend on the quaternions of the two SRTs (due to a normalization of the interpolated quaternion).
//
// Here, we compute the coefficients of the numerator of the derivative of the given SR applied to a point.
// Note that these are still 'factored' in the sense that the actual coefficients per axis are computed by applying a dot product with the point (see SrtDerivativeTerm).
//
// The numerator and denominator terms will be used for computing the extrema of the SR (wrt. to a constant or linear function) evaluated for a given input point.
M_DEVICE_HOST __noinline__ static void makeSRDerivativeTerms( const SRTData&            key0,
                                                              const SRTData&            key1,
                                                              SRNumeratorDerivativeTerm c0[3],
                                                              SRNumeratorDerivativeTerm c1[3],
                                                              SRNumeratorDerivativeTerm c2[3],
                                                              SRNumeratorDerivativeTerm c3[3],
                                                              SRNumeratorDerivativeTerm c4[3],
                                                              float                     denominatorCoeffs[5] );


}  // namespace motion
}  // namespace optix_exp

#include "srTransformHelper.hpp"
