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

#include "jama/jama_eig.h"
#include "jama/tnt_array1d.h"
#include "jama/tnt_array2d.h"

namespace optix_exp {
namespace motion {

// Find roots of a polynomial of degree 4 or lower.
// This is done by using the companion matrix code from jama, see below.
//
// (We compared this to computing the polynomial roots directly using
//  Ferrari's method, that implementation was less precise.)

M_DEVICE_HOST static int findRealRoots4( const float coeffs[5],  // coefficients of the polynomial
                                         float       roots[4] )        // out: real roots, up to 4
{
    const float abs_eps       = 3.e-6f;
    const float abs_coeffs[5] = {fabsf( coeffs[0] ), fabsf( coeffs[1] ), fabsf( coeffs[2] ), fabsf( coeffs[3] ),
                                 fabsf( coeffs[4] )};

    float max_coeff = abs_coeffs[0];
    for( int i = 1; i < 5; ++i )
        max_coeff = ::max( max_coeff, abs_coeffs[i] );

    if( max_coeff < abs_eps )
        return 0;  // All coeffs are nearly zero.

    // Remove coefficients that have very small magnitudes.
    int         n   = 4;
    const float eps = 1.e-5f;
    while( abs_coeffs[n] < abs_eps || abs_coeffs[n] / max_coeff < eps )
    {
        n--;
        if( n == 0 )
        {  // constant polynomial
            return 0;
        }
    }

    float normalized_coeffs[4];
    for( int i = 0; i < n; ++i )
    {
        normalized_coeffs[i] = -1.0f * coeffs[i] / coeffs[n];
    }

    // Handle linear and quadratic polynomials directly.
    if( n == 1 )
    {
        roots[0] = normalized_coeffs[0];
        return 1;
    }
    else if( n == 2 )
    {
        float p_half = -normalized_coeffs[1] * 0.5f;
        float q      = -normalized_coeffs[0];
        float rad    = p_half * p_half - q;
        if( rad >= 0.f )
        {
            if( rad == 0.f )
            {
                roots[0] = -p_half;
                return 1;
            }
            else
            {
                float s  = sqrtf( rad );
                roots[0] = -p_half + s;
                roots[1] = -p_half - s;
                return 2;
            }
        }
        else
        {
            return 0;
        }
    }

    // Build companion matrix
    TNT::Array2D<float, 4, 4> C( n, n, 0.0f );
    for( int i = 1; i < n; ++i )
        C[i][i - 1] = 1.0f;
    for( int i = 0; i < n; ++i )
    {
        C[i][n - 1] = normalized_coeffs[i];
    }

    // Solve for eigelwalues
    JAMA::Eigelwalue<float, 4> eigs( C );
    int count = 0;
    for( int i = 0; i < n; ++i )
    {
        if( fabs( eigs.getImagEigelwalue( i ) ) < eps )
        {
            roots[count++] = eigs.getRealEigelwalues( i );
        }
    }

    return count;
};

M_DEVICE_HOST static void makeSRDerivativeTerms( const SRTData&            key0,
                                                 const SRTData&            key1,
                                                 SRNumeratorDerivativeTerm c0[3],
                                                 SRNumeratorDerivativeTerm c1[3],
                                                 SRNumeratorDerivativeTerm c2[3],
                                                 SRNumeratorDerivativeTerm c3[3],
                                                 SRNumeratorDerivativeTerm c4[3],
                                                 float                     denominatorCoeffs[5] )
{
    // End goal: bound a polynomial c(t): c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4
    // To evaluate c0, c1, ..., we will need a point 'p', which we do not yet have.
    // The SrtDerivativeTerm holds everything else needed to evaluate c before knowing 'p'.
    //
    // There is a separate polynomial per axis.

    // we lwrrently only support scale matrices of the form
    // | sx a  b  pvx |
    // | 0  sy c  pvy |
    // | 0  0  sz pvz |
    // but the generated code below assumes a full matrix.
    // TODO: re-generate code with the prior assumption about the sparsity of the matrix.
    // WAR: Explicitly write zeros to the variables and leave the optimization of the expressions below to the compiler.

    // M0
    const float m000 = key0.sx;
    const float m001 = key0.a;
    const float m002 = key0.b;
    const float m003 = key0.pvx;
    const float m010 = 0;
    const float m011 = key0.sy;
    const float m012 = key0.c;
    const float m013 = key0.pvy;
    const float m020 = 0;
    const float m021 = 0;
    const float m022 = key0.sz;
    const float m023 = key0.pvz;

    // M1
    const float m100 = key1.sx;
    const float m101 = key1.a;
    const float m102 = key1.b;
    const float m103 = key1.pvx;
    const float m110 = 0;
    const float m111 = key1.sy;
    const float m112 = key1.c;
    const float m113 = key1.pvy;
    const float m120 = 0;
    const float m121 = 0;
    const float m122 = key1.sz;
    const float m123 = key1.pvz;

    // Q0
    const float q00 = key0.qx;
    const float q01 = key0.qy;
    const float q02 = key0.qz;
    const float q03 = key0.qw;

    // Q1
    const float q10 = key1.qx;
    const float q11 = key1.qy;
    const float q12 = key1.qz;
    const float q13 = key1.qw;

    // Precompute some powers
    const float q00_2 = q00 * q00;
    const float q01_2 = q01 * q01;
    const float q02_2 = q02 * q02;
    const float q03_2 = q03 * q03;
    const float q00_3 = q00 * q00_2;
    const float q01_3 = q01 * q01_2;
    const float q02_3 = q02 * q02_2;
    const float q03_3 = q03 * q03_2;
    const float q00_4 = q00 * q00_3;
    const float q01_4 = q01 * q01_3;
    const float q02_4 = q02 * q02_3;
    const float q03_4 = q03 * q03_3;

    const float q10_2 = q10 * q10;
    const float q11_2 = q11 * q11;
    const float q12_2 = q12 * q12;
    const float q13_2 = q13 * q13;
    const float q10_3 = q10 * q10_2;
    const float q11_3 = q11 * q11_2;
    const float q12_3 = q12 * q12_2;
    const float q13_3 = q13 * q13_2;

    // Precompute some squares.
    const float q001_2   = ( q00_2 + q01_2 ) * ( q00_2 + q01_2 );
    const float q012_2   = ( q01_2 + q02_2 ) * ( q01_2 + q02_2 );
    const float q013_2   = ( q01_2 + q03_2 ) * ( q01_2 + q03_2 );
    const float q023_2   = ( q02_2 + q03_2 ) * ( q02_2 + q03_2 );
    const float q00_10_2 = ( q00 - q10 ) * ( q00 - q10 );
    const float q01_11_2 = ( q01 - q11 ) * ( q01 - q11 );
    const float q02_12_2 = ( q02 - q12 ) * ( q02 - q12 );
    const float q03_13_2 = ( q03 - q13 ) * ( q03 - q13 );

    const float a = q00_2 + q01_2 + q02_2 + q03_2;
    const float b = q00_10_2 + q01_11_2 + q02_12_2 + q03_13_2;

    denominatorCoeffs[0] = a * a;
    denominatorCoeffs[1] = -4 * a * ( a - q00 * q10 - q01 * q11 - q02 * q12 - q03 * q13 );
    denominatorCoeffs[2] =
        2
        * ( 3 * q00_4 + 3 * q01_4 + 3 * q02_4 + 6 * q02_2 * q03_2 + 3 * q03_4 - 6 * q00_3 * q10 + q02_2 * q10_2
            + q03_2 * q10_2 - 6 * q01_3 * q11 + q02_2 * q11_2 + q03_2 * q11_2 - 6 * q02_3 * q12 - 6 * q02 * q03_2 * q12
            + 3 * q02_2 * q12_2 + q03_2 * q12_2 - 6 * q03 * ( q02_2 + q03_2 ) * q13 + 4 * q02 * q03 * q12 * q13
            + ( q02_2 + 3 * q03_2 ) * q13_2 + 2 * q01 * q11 * ( -3 * q02_2 - 3 * q03_2 + 2 * q02 * q12 + 2 * q03 * q13 )
            + 2 * q00 * q10 * ( -3 * q01_2 - 3 * q02_2 - 3 * q03_2 + 2 * q01 * q11 + 2 * q02 * q12 + 2 * q03 * q13 )
            + q00_2 * ( 6 * q01_2 + 6 * q02_2 + 6 * q03_2 + 3 * q10_2 - 6 * q01 * q11 + q11_2 - 6 * q02 * q12 + q12_2 - 6 * q03 * q13 + q13_2 )
            + q01_2 * ( 6 * q02_2 + 6 * q03_2 + q10_2 + 3 * q11_2 - 6 * q02 * q12 + q12_2 - 6 * q03 * q13 + q13_2 ) );
    denominatorCoeffs[3] = -4 * b * ( a - q00 * q10 - q01 * q11 - q02 * q12 - q03 * q13 );
    denominatorCoeffs[4] = b * b;

    // Reformatted version of one of the 'coeffs.txt' files generated from Mathematica.

    // X
    c0[0] = SRNumeratorDerivativeTerm(
        m100 * ( q00_2 - q01_2 - q02_2 + q03_2 ) * ( a )
            + m000
                  * ( -q00_4 + q012_2 + 4 * q00 * ( q01_2 + q02_2 ) * q10 - 2 * q00_2 * ( q03_2 + 2 * q01 * q11 + 2 * q02 * q12 )
                      + q03 * ( -( q03 * ( q03_2 + 4 * q01 * q11 + 4 * q02 * q12 ) ) + 4 * ( q01_2 + q02_2 ) * q13 ) )
            - 2
                  * ( m020 * q00_3 * q02 - m120 * q00_3 * q02 + m020 * q00 * q01_2 * q02 - m120 * q00 * q01_2 * q02
                      + m020 * q00 * q02_3 - m120 * q00 * q02_3 + m020 * q00_2 * q01 * q03 - m120 * q00_2 * q01 * q03
                      + m020 * q01_3 * q03 - m120 * q01_3 * q03 + m020 * q01 * q02_2 * q03 - m120 * q01 * q02_2 * q03
                      + m020 * q00 * q02 * q03_2 - m120 * q00 * q02 * q03_2 + m020 * q01 * q03_3 - m120 * q01 * q03_3
                      - m110 * ( q00 * q01 - q02 * q03 ) * ( a ) + m020 * q00_2 * q02 * q10 - m020 * q01_2 * q02 * q10
                      - m020 * q02_3 * q10 + 2 * m020 * q00 * q01 * q03 * q10 - m020 * q02 * q03_2 * q10
                      + 2 * m020 * q00 * q01 * q02 * q11 - m020 * q00_2 * q03 * q11 + m020 * q01_2 * q03 * q11
                      - m020 * q02_2 * q03 * q11 - m020 * q03_3 * q11 - m020 * q00_3 * q12 - m020 * q00 * q01_2 * q12
                      + m020 * q00 * q02_2 * q12 + 2 * m020 * q01 * q02 * q03 * q12 - m020 * q00 * q03_2 * q12
                      - m020 * ( q01 * ( q00_2 + q01_2 + q02_2 ) - 2 * q00 * q02 * q03 - q01 * q03_2 ) * q13
                      + m010
                            * ( -( q01_3 * q10 ) + q00_3 * ( q01 - q11 ) - q01 * ( ( q02_2 + q03_2 ) * q10 + 2 * q02 * q03 * q11 )
                                - q03 * ( q02_3 + q02 * q03_2 + q02_2 * q12 - q03_2 * q12 )
                                + q02 * ( q02 - q03 ) * ( q02 + q03 ) * q13 + q01_2 * ( -( q02 * q03 ) + q03 * q12 + q02 * q13 )
                                + q00_2 * ( q01 * q10 + q03 * q12 + q02 * ( -q03 + q13 ) )
                                + q00
                                      * ( q01_3 - 2 * q02 * q03 * q10 + q01_2 * q11 - ( q02_2 + q03_2 ) * q11
                                          + q01 * ( q02_2 + q03_2 + 2 * q02 * q12 + 2 * q03 * q13 ) ) ) ),
        m101 * ( q00_2 - q01_2 - q02_2 + q03_2 ) * ( a )
            + m001
                  * ( -q00_4 + q012_2 + 4 * q00 * ( q01_2 + q02_2 ) * q10 - 2 * q00_2 * ( q03_2 + 2 * q01 * q11 + 2 * q02 * q12 )
                      + q03 * ( -( q03 * ( q03_2 + 4 * q01 * q11 + 4 * q02 * q12 ) ) + 4 * ( q01_2 + q02_2 ) * q13 ) )
            - 2
                  * ( m021 * q00_3 * q02 - m121 * q00_3 * q02 + m021 * q00 * q01_2 * q02 - m121 * q00 * q01_2 * q02
                      + m021 * q00 * q02_3 - m121 * q00 * q02_3 + m021 * q00_2 * q01 * q03 - m121 * q00_2 * q01 * q03
                      + m021 * q01_3 * q03 - m121 * q01_3 * q03 + m021 * q01 * q02_2 * q03 - m121 * q01 * q02_2 * q03
                      + m021 * q00 * q02 * q03_2 - m121 * q00 * q02 * q03_2 + m021 * q01 * q03_3 - m121 * q01 * q03_3
                      - m111 * ( q00 * q01 - q02 * q03 ) * ( a ) + m021 * q00_2 * q02 * q10 - m021 * q01_2 * q02 * q10
                      - m021 * q02_3 * q10 + 2 * m021 * q00 * q01 * q03 * q10 - m021 * q02 * q03_2 * q10
                      + 2 * m021 * q00 * q01 * q02 * q11 - m021 * q00_2 * q03 * q11 + m021 * q01_2 * q03 * q11
                      - m021 * q02_2 * q03 * q11 - m021 * q03_3 * q11 - m021 * q00_3 * q12 - m021 * q00 * q01_2 * q12
                      + m021 * q00 * q02_2 * q12 + 2 * m021 * q01 * q02 * q03 * q12 - m021 * q00 * q03_2 * q12
                      - m021 * ( q01 * ( q00_2 + q01_2 + q02_2 ) - 2 * q00 * q02 * q03 - q01 * q03_2 ) * q13
                      + m011
                            * ( -( q01_3 * q10 ) + q00_3 * ( q01 - q11 ) - q01 * ( ( q02_2 + q03_2 ) * q10 + 2 * q02 * q03 * q11 )
                                - q03 * ( q02_3 + q02 * q03_2 + q02_2 * q12 - q03_2 * q12 )
                                + q02 * ( q02 - q03 ) * ( q02 + q03 ) * q13 + q01_2 * ( -( q02 * q03 ) + q03 * q12 + q02 * q13 )
                                + q00_2 * ( q01 * q10 + q03 * q12 + q02 * ( -q03 + q13 ) )
                                + q00 * ( q01_3 - 2 * q02 * q03 * q10 + q01_2 * q11 - ( q02_2 + q03_2 ) * q11 + q01 * ( q02_2 + q03_2 + 2 * q02 * q12 + 2 * q03 * q13 ) ) ) ),
        m102 * ( q00_2 - q01_2 - q02_2 + q03_2 ) * ( a )
            + m002
                  * ( -q00_4 + q012_2 + 4 * q00 * ( q01_2 + q02_2 ) * q10 - 2 * q00_2 * ( q03_2 + 2 * q01 * q11 + 2 * q02 * q12 )
                      + q03 * ( -( q03 * ( q03_2 + 4 * q01 * q11 + 4 * q02 * q12 ) ) + 4 * ( q01_2 + q02_2 ) * q13 ) )
            - 2
                  * ( m022 * q00_3 * q02 - m122 * q00_3 * q02 + m022 * q00 * q01_2 * q02 - m122 * q00 * q01_2 * q02
                      + m022 * q00 * q02_3 - m122 * q00 * q02_3 + m022 * q00_2 * q01 * q03 - m122 * q00_2 * q01 * q03
                      + m022 * q01_3 * q03 - m122 * q01_3 * q03 + m022 * q01 * q02_2 * q03 - m122 * q01 * q02_2 * q03
                      + m022 * q00 * q02 * q03_2 - m122 * q00 * q02 * q03_2 + m022 * q01 * q03_3 - m122 * q01 * q03_3
                      - m112 * ( q00 * q01 - q02 * q03 ) * ( a ) + m022 * q00_2 * q02 * q10 - m022 * q01_2 * q02 * q10
                      - m022 * q02_3 * q10 + 2 * m022 * q00 * q01 * q03 * q10 - m022 * q02 * q03_2 * q10
                      + 2 * m022 * q00 * q01 * q02 * q11 - m022 * q00_2 * q03 * q11 + m022 * q01_2 * q03 * q11
                      - m022 * q02_2 * q03 * q11 - m022 * q03_3 * q11 - m022 * q00_3 * q12 - m022 * q00 * q01_2 * q12
                      + m022 * q00 * q02_2 * q12 + 2 * m022 * q01 * q02 * q03 * q12 - m022 * q00 * q03_2 * q12
                      - m022 * ( q01 * ( q00_2 + q01_2 + q02_2 ) - 2 * q00 * q02 * q03 - q01 * q03_2 ) * q13
                      + m012
                            * ( -( q01_3 * q10 ) + q00_3 * ( q01 - q11 ) - q01 * ( ( q02_2 + q03_2 ) * q10 + 2 * q02 * q03 * q11 )
                                - q03 * ( q02_3 + q02 * q03_2 + q02_2 * q12 - q03_2 * q12 )
                                + q02 * ( q02 - q03 ) * ( q02 + q03 ) * q13 + q01_2 * ( -( q02 * q03 ) + q03 * q12 + q02 * q13 )
                                + q00_2 * ( q01 * q10 + q03 * q12 + q02 * ( -q03 + q13 ) )
                                + q00 * ( q01_3 - 2 * q02 * q03 * q10 + q01_2 * q11 - ( q02_2 + q03_2 ) * q11 + q01 * ( q02_2 + q03_2 + 2 * q02 * q12 + 2 * q03 * q13 ) ) ) ),
        -2 * m013 * q00_3 * q01 + 2 * m113 * q00_3 * q01 - 2 * m013 * q00 * q01_3 + 2 * m113 * q00 * q01_3
            - 2 * m023 * q00_3 * q02 + 2 * m123 * q00_3 * q02 - 2 * m023 * q00 * q01_2 * q02 + 2 * m123 * q00 * q01_2 * q02
            - 2 * m013 * q00 * q01 * q02_2 + 2 * m113 * q00 * q01 * q02_2 - 2 * m023 * q00 * q02_3 + 2 * m123 * q00 * q02_3
            - 2 * m023 * q00_2 * q01 * q03 + 2 * m123 * q00_2 * q01 * q03 - 2 * m023 * q01_3 * q03 + 2 * m123 * q01_3 * q03
            + 2 * m013 * q00_2 * q02 * q03 - 2 * m113 * q00_2 * q02 * q03 + 2 * m013 * q01_2 * q02 * q03
            - 2 * m113 * q01_2 * q02 * q03 - 2 * m023 * q01 * q02_2 * q03 + 2 * m123 * q01 * q02_2 * q03 + 2 * m013 * q02_3 * q03
            - 2 * m113 * q02_3 * q03 - 2 * m013 * q00 * q01 * q03_2 + 2 * m113 * q00 * q01 * q03_2 - 2 * m023 * q00 * q02 * q03_2
            + 2 * m123 * q00 * q02 * q03_2 - 2 * m023 * q01 * q03_3 + 2 * m123 * q01 * q03_3 + 2 * m013 * q02 * q03_3
            - 2 * m113 * q02 * q03_3 + m103 * ( q00_2 - q01_2 - q02_2 + q03_2 ) * (a)-2 * m013 * q00_2 * q01 * q10
            + 2 * m013 * q01_3 * q10 - 2 * m023 * q00_2 * q02 * q10 + 2 * m023 * q01_2 * q02 * q10 + 2 * m013 * q01 * q02_2 * q10
            + 2 * m023 * q02_3 * q10 - 4 * m023 * q00 * q01 * q03 * q10 + 4 * m013 * q00 * q02 * q03 * q10
            + 2 * m013 * q01 * q03_2 * q10 + 2 * m023 * q02 * q03_2 * q10 + 2 * m013 * q00_3 * q11
            - 2 * m013 * q00 * q01_2 * q11 - 4 * m023 * q00 * q01 * q02 * q11 + 2 * m013 * q00 * q02_2 * q11
            + 2 * m023 * q00_2 * q03 * q11 - 2 * m023 * q01_2 * q03 * q11 + 4 * m013 * q01 * q02 * q03 * q11
            + 2 * m023 * q02_2 * q03 * q11 + 2 * m013 * q00 * q03_2 * q11 + 2 * m023 * q03_3 * q11 + 2 * m023 * q00_3 * q12
            + 2 * m023 * q00 * q01_2 * q12 - 4 * m013 * q00 * q01 * q02 * q12 - 2 * m023 * q00 * q02_2 * q12
            - 2 * m013 * q00_2 * q03 * q12 - 2 * m013 * q01_2 * q03 * q12 - 4 * m023 * q01 * q02 * q03 * q12
            + 2 * m013 * q02_2 * q03 * q12 + 2 * m023 * q00 * q03_2 * q12 - 2 * m013 * q03_3 * q12
            + 2 * m023 * q00_2 * q01 * q13 + 2 * m023 * q01_3 * q13 - 2 * m013 * q00_2 * q02 * q13 - 2 * m013 * q01_2 * q02 * q13
            + 2 * m023 * q01 * q02_2 * q13 - 2 * m013 * q02_3 * q13 - 4 * m013 * q00 * q01 * q03 * q13
            - 4 * m023 * q00 * q02 * q03 * q13 - 2 * m023 * q01 * q03_2 * q13 + 2 * m013 * q02 * q03_2 * q13
            + m003
                  * ( -q00_4 + q012_2 + 4 * q00 * ( q01_2 + q02_2 ) * q10 - 2 * q00_2 * ( q03_2 + 2 * q01 * q11 + 2 * q02 * q12 )
                      + q03 * ( -( q03 * ( q03_2 + 4 * q01 * q11 + 4 * q02 * q12 ) ) + 4 * ( q01_2 + q02_2 ) * q13 ) ) );
    c1[0] = SRNumeratorDerivativeTerm(
        4
            * ( 2 * m010 * q00_3 * q01 - 2 * m110 * q00_3 * q01 + 2 * m010 * q00 * q01_3 - 2 * m110 * q00 * q01_3
                + 2 * m020 * q00_3 * q02 - 2 * m120 * q00_3 * q02 + 2 * m020 * q00 * q01_2 * q02 - 2 * m120 * q00 * q01_2 * q02
                + 2 * m010 * q00 * q01 * q02_2 - 2 * m110 * q00 * q01 * q02_2 + 2 * m020 * q00 * q02_3 - 2 * m120 * q00 * q02_3
                + 2 * m020 * q00_2 * q01 * q03 - 2 * m120 * q00_2 * q01 * q03 + 2 * m020 * q01_3 * q03 - 2 * m120 * q01_3 * q03
                - 2 * m010 * q00_2 * q02 * q03 + 2 * m110 * q00_2 * q02 * q03 - 2 * m010 * q01_2 * q02 * q03
                + 2 * m110 * q01_2 * q02 * q03 + 2 * m020 * q01 * q02_2 * q03 - 2 * m120 * q01 * q02_2 * q03
                - 2 * m010 * q02_3 * q03 + 2 * m110 * q02_3 * q03 + 2 * m010 * q00 * q01 * q03_2 - 2 * m110 * q00 * q01 * q03_2
                + 2 * m020 * q00 * q02 * q03_2 - 2 * m120 * q00 * q02 * q03_2 + 2 * m020 * q01 * q03_3 - 2 * m120 * q01 * q03_3
                - 2 * m010 * q02 * q03_3 + 2 * m110 * q02 * q03_3 + m110 * q00_2 * q01 * q10 - 2 * m010 * q01_3 * q10
                + m110 * q01_3 * q10 + m120 * q00_2 * q02 * q10 - 2 * m020 * q01_2 * q02 * q10 + m120 * q01_2 * q02 * q10
                - 2 * m010 * q01 * q02_2 * q10 + m110 * q01 * q02_2 * q10 - 2 * m020 * q02_3 * q10 + m120 * q02_3 * q10
                + 2 * m020 * q00 * q01 * q03 * q10 - 2 * m010 * q00 * q02 * q03 * q10 - 2 * m010 * q01 * q03_2 * q10
                + m110 * q01 * q03_2 * q10 - 2 * m020 * q02 * q03_2 * q10 + m120 * q02 * q03_2 * q10 - m010 * q00 * q01 * q10_2
                - m020 * q00 * q02 * q10_2 - m020 * q01 * q03 * q10_2 + m010 * q02 * q03 * q10_2 - 2 * m010 * q00_3 * q11
                + m110 * q00_3 * q11 + m110 * q00 * q01_2 * q11 + 2 * m020 * q00 * q01 * q02 * q11 - 2 * m010 * q00 * q02_2 * q11
                + m110 * q00 * q02_2 * q11 - 2 * m020 * q00_2 * q03 * q11 + m120 * q00_2 * q03 * q11
                + m120 * q01_2 * q03 * q11 - 2 * m010 * q01 * q02 * q03 * q11 - 2 * m020 * q02_2 * q03 * q11
                + m120 * q02_2 * q03 * q11 - 2 * m010 * q00 * q03_2 * q11 + m110 * q00 * q03_2 * q11 - 2 * m020 * q03_3 * q11
                + m120 * q03_3 * q11 + m010 * q00_2 * q10 * q11 + m010 * q01_2 * q10 * q11 + m010 * q02_2 * q10 * q11
                + m010 * q03_2 * q10 * q11 - m010 * q00 * q01 * q11_2 - m020 * q00 * q02 * q11_2 - m020 * q01 * q03 * q11_2
                + m010 * q02 * q03 * q11_2 - 2 * m020 * q00_3 * q12 + m120 * q00_3 * q12 - 2 * m020 * q00 * q01_2 * q12
                + m120 * q00 * q01_2 * q12 + 2 * m010 * q00 * q01 * q02 * q12 + m120 * q00 * q02_2 * q12
                + 2 * m010 * q00_2 * q03 * q12 - m110 * q00_2 * q03 * q12 + 2 * m010 * q01_2 * q03 * q12
                - m110 * q01_2 * q03 * q12 + 2 * m020 * q01 * q02 * q03 * q12 - m110 * q02_2 * q03 * q12
                - 2 * m020 * q00 * q03_2 * q12 + m120 * q00 * q03_2 * q12 + 2 * m010 * q03_3 * q12 - m110 * q03_3 * q12
                + m020 * q00_2 * q10 * q12 + m020 * q01_2 * q10 * q12 + m020 * q02_2 * q10 * q12 + m020 * q03_2 * q10 * q12
                - m010 * q00 * q01 * q12_2 - m020 * q00 * q02 * q12_2 - m020 * q01 * q03 * q12_2 + m010 * q02 * q03 * q12_2
                - m100 * ( a ) * ( q00_2 - q00 * q10 + q01 * ( -q01 + q11 ) + q02 * ( -q02 + q12 ) + q03 * ( q03 - q13 ) )
                + ( 2 * m010 * q00_2 * q02 - m110 * q00_2 * q02 + 2 * m010 * q01_2 * q02 - m110 * q01_2 * q02
                    + 2 * m010 * q02_3 - m110 * q02_3 + 2 * m010 * q00 * q01 * q03 - m110 * q02 * q03_2 + m120 * q01 * ( a )
                    + m020 * ( -2 * q01 * ( q00_2 + q01_2 + q02_2 ) + 2 * q00 * q02 * q03 + (a)*q11 ) - m010 * (a)*q12 )
                      * q13
                - ( m010 * q00 * q01 + m020 * q00 * q02 + m020 * q01 * q03 - m010 * q02 * q03 ) * q13_2
                + m000
                      * ( q00_4 - q012_2 + q03_4 - q00_3 * q10 - q00 * ( 3 * ( q01_2 + q02_2 ) + q03_2 ) * q10
                          + q03_2 * ( 3 * q01 * q11 - q11_2 + ( 3 * q02 - q12 ) * q12 ) - 3 * ( q01_2 + q02_2 ) * q03 * q13
                          - q03_3 * q13 + q00_2 * ( 2 * q03_2 + 3 * q01 * q11 - q11_2 + 3 * q02 * q12 - q12_2 - q03 * q13 )
                          + ( q01_2 + q02_2 ) * ( q10_2 + q01 * q11 + q02 * q12 + q13_2 ) ) ),
        4
            * ( 2 * m011 * q00_3 * q01 - 2 * m111 * q00_3 * q01 + 2 * m011 * q00 * q01_3 - 2 * m111 * q00 * q01_3
                + 2 * m021 * q00_3 * q02 - 2 * m121 * q00_3 * q02 + 2 * m021 * q00 * q01_2 * q02 - 2 * m121 * q00 * q01_2 * q02
                + 2 * m011 * q00 * q01 * q02_2 - 2 * m111 * q00 * q01 * q02_2 + 2 * m021 * q00 * q02_3 - 2 * m121 * q00 * q02_3
                + 2 * m021 * q00_2 * q01 * q03 - 2 * m121 * q00_2 * q01 * q03 + 2 * m021 * q01_3 * q03 - 2 * m121 * q01_3 * q03
                - 2 * m011 * q00_2 * q02 * q03 + 2 * m111 * q00_2 * q02 * q03 - 2 * m011 * q01_2 * q02 * q03
                + 2 * m111 * q01_2 * q02 * q03 + 2 * m021 * q01 * q02_2 * q03 - 2 * m121 * q01 * q02_2 * q03
                - 2 * m011 * q02_3 * q03 + 2 * m111 * q02_3 * q03 + 2 * m011 * q00 * q01 * q03_2 - 2 * m111 * q00 * q01 * q03_2
                + 2 * m021 * q00 * q02 * q03_2 - 2 * m121 * q00 * q02 * q03_2 + 2 * m021 * q01 * q03_3 - 2 * m121 * q01 * q03_3
                - 2 * m011 * q02 * q03_3 + 2 * m111 * q02 * q03_3 + m111 * q00_2 * q01 * q10 - 2 * m011 * q01_3 * q10
                + m111 * q01_3 * q10 + m121 * q00_2 * q02 * q10 - 2 * m021 * q01_2 * q02 * q10 + m121 * q01_2 * q02 * q10
                - 2 * m011 * q01 * q02_2 * q10 + m111 * q01 * q02_2 * q10 - 2 * m021 * q02_3 * q10 + m121 * q02_3 * q10
                + 2 * m021 * q00 * q01 * q03 * q10 - 2 * m011 * q00 * q02 * q03 * q10 - 2 * m011 * q01 * q03_2 * q10
                + m111 * q01 * q03_2 * q10 - 2 * m021 * q02 * q03_2 * q10 + m121 * q02 * q03_2 * q10 - m011 * q00 * q01 * q10_2
                - m021 * q00 * q02 * q10_2 - m021 * q01 * q03 * q10_2 + m011 * q02 * q03 * q10_2 - 2 * m011 * q00_3 * q11
                + m111 * q00_3 * q11 + m111 * q00 * q01_2 * q11 + 2 * m021 * q00 * q01 * q02 * q11 - 2 * m011 * q00 * q02_2 * q11
                + m111 * q00 * q02_2 * q11 - 2 * m021 * q00_2 * q03 * q11 + m121 * q00_2 * q03 * q11
                + m121 * q01_2 * q03 * q11 - 2 * m011 * q01 * q02 * q03 * q11 - 2 * m021 * q02_2 * q03 * q11
                + m121 * q02_2 * q03 * q11 - 2 * m011 * q00 * q03_2 * q11 + m111 * q00 * q03_2 * q11 - 2 * m021 * q03_3 * q11
                + m121 * q03_3 * q11 + m011 * q00_2 * q10 * q11 + m011 * q01_2 * q10 * q11 + m011 * q02_2 * q10 * q11
                + m011 * q03_2 * q10 * q11 - m011 * q00 * q01 * q11_2 - m021 * q00 * q02 * q11_2 - m021 * q01 * q03 * q11_2
                + m011 * q02 * q03 * q11_2 - 2 * m021 * q00_3 * q12 + m121 * q00_3 * q12 - 2 * m021 * q00 * q01_2 * q12
                + m121 * q00 * q01_2 * q12 + 2 * m011 * q00 * q01 * q02 * q12 + m121 * q00 * q02_2 * q12
                + 2 * m011 * q00_2 * q03 * q12 - m111 * q00_2 * q03 * q12 + 2 * m011 * q01_2 * q03 * q12
                - m111 * q01_2 * q03 * q12 + 2 * m021 * q01 * q02 * q03 * q12 - m111 * q02_2 * q03 * q12
                - 2 * m021 * q00 * q03_2 * q12 + m121 * q00 * q03_2 * q12 + 2 * m011 * q03_3 * q12 - m111 * q03_3 * q12
                + m021 * q00_2 * q10 * q12 + m021 * q01_2 * q10 * q12 + m021 * q02_2 * q10 * q12 + m021 * q03_2 * q10 * q12
                - m011 * q00 * q01 * q12_2 - m021 * q00 * q02 * q12_2 - m021 * q01 * q03 * q12_2 + m011 * q02 * q03 * q12_2
                - m101 * ( a ) * ( q00_2 - q00 * q10 + q01 * ( -q01 + q11 ) + q02 * ( -q02 + q12 ) + q03 * ( q03 - q13 ) )
                + ( 2 * m011 * q00_2 * q02 - m111 * q00_2 * q02 + 2 * m011 * q01_2 * q02 - m111 * q01_2 * q02
                    + 2 * m011 * q02_3 - m111 * q02_3 + 2 * m011 * q00 * q01 * q03 - m111 * q02 * q03_2 + m121 * q01 * ( a )
                    + m021 * ( -2 * q01 * ( q00_2 + q01_2 + q02_2 ) + 2 * q00 * q02 * q03 + (a)*q11 ) - m011 * (a)*q12 )
                      * q13
                - ( m011 * q00 * q01 + m021 * q00 * q02 + m021 * q01 * q03 - m011 * q02 * q03 ) * q13_2
                + m001
                      * ( q00_4 - q012_2 + q03_4 - q00_3 * q10 - q00 * ( 3 * ( q01_2 + q02_2 ) + q03_2 ) * q10
                          + q03_2 * ( 3 * q01 * q11 - q11_2 + ( 3 * q02 - q12 ) * q12 ) - 3 * ( q01_2 + q02_2 ) * q03 * q13
                          - q03_3 * q13 + q00_2 * ( 2 * q03_2 + 3 * q01 * q11 - q11_2 + 3 * q02 * q12 - q12_2 - q03 * q13 )
                          + ( q01_2 + q02_2 ) * ( q10_2 + q01 * q11 + q02 * q12 + q13_2 ) ) ),
        4
            * ( 2 * m012 * q00_3 * q01 - 2 * m112 * q00_3 * q01 + 2 * m012 * q00 * q01_3 - 2 * m112 * q00 * q01_3
                + 2 * m022 * q00_3 * q02 - 2 * m122 * q00_3 * q02 + 2 * m022 * q00 * q01_2 * q02 - 2 * m122 * q00 * q01_2 * q02
                + 2 * m012 * q00 * q01 * q02_2 - 2 * m112 * q00 * q01 * q02_2 + 2 * m022 * q00 * q02_3 - 2 * m122 * q00 * q02_3
                + 2 * m022 * q00_2 * q01 * q03 - 2 * m122 * q00_2 * q01 * q03 + 2 * m022 * q01_3 * q03 - 2 * m122 * q01_3 * q03
                - 2 * m012 * q00_2 * q02 * q03 + 2 * m112 * q00_2 * q02 * q03 - 2 * m012 * q01_2 * q02 * q03
                + 2 * m112 * q01_2 * q02 * q03 + 2 * m022 * q01 * q02_2 * q03 - 2 * m122 * q01 * q02_2 * q03
                - 2 * m012 * q02_3 * q03 + 2 * m112 * q02_3 * q03 + 2 * m012 * q00 * q01 * q03_2 - 2 * m112 * q00 * q01 * q03_2
                + 2 * m022 * q00 * q02 * q03_2 - 2 * m122 * q00 * q02 * q03_2 + 2 * m022 * q01 * q03_3 - 2 * m122 * q01 * q03_3
                - 2 * m012 * q02 * q03_3 + 2 * m112 * q02 * q03_3 + m112 * q00_2 * q01 * q10 - 2 * m012 * q01_3 * q10
                + m112 * q01_3 * q10 + m122 * q00_2 * q02 * q10 - 2 * m022 * q01_2 * q02 * q10 + m122 * q01_2 * q02 * q10
                - 2 * m012 * q01 * q02_2 * q10 + m112 * q01 * q02_2 * q10 - 2 * m022 * q02_3 * q10 + m122 * q02_3 * q10
                + 2 * m022 * q00 * q01 * q03 * q10 - 2 * m012 * q00 * q02 * q03 * q10 - 2 * m012 * q01 * q03_2 * q10
                + m112 * q01 * q03_2 * q10 - 2 * m022 * q02 * q03_2 * q10 + m122 * q02 * q03_2 * q10 - m012 * q00 * q01 * q10_2
                - m022 * q00 * q02 * q10_2 - m022 * q01 * q03 * q10_2 + m012 * q02 * q03 * q10_2 - 2 * m012 * q00_3 * q11
                + m112 * q00_3 * q11 + m112 * q00 * q01_2 * q11 + 2 * m022 * q00 * q01 * q02 * q11 - 2 * m012 * q00 * q02_2 * q11
                + m112 * q00 * q02_2 * q11 - 2 * m022 * q00_2 * q03 * q11 + m122 * q00_2 * q03 * q11
                + m122 * q01_2 * q03 * q11 - 2 * m012 * q01 * q02 * q03 * q11 - 2 * m022 * q02_2 * q03 * q11
                + m122 * q02_2 * q03 * q11 - 2 * m012 * q00 * q03_2 * q11 + m112 * q00 * q03_2 * q11 - 2 * m022 * q03_3 * q11
                + m122 * q03_3 * q11 + m012 * q00_2 * q10 * q11 + m012 * q01_2 * q10 * q11 + m012 * q02_2 * q10 * q11
                + m012 * q03_2 * q10 * q11 - m012 * q00 * q01 * q11_2 - m022 * q00 * q02 * q11_2 - m022 * q01 * q03 * q11_2
                + m012 * q02 * q03 * q11_2 - 2 * m022 * q00_3 * q12 + m122 * q00_3 * q12 - 2 * m022 * q00 * q01_2 * q12
                + m122 * q00 * q01_2 * q12 + 2 * m012 * q00 * q01 * q02 * q12 + m122 * q00 * q02_2 * q12
                + 2 * m012 * q00_2 * q03 * q12 - m112 * q00_2 * q03 * q12 + 2 * m012 * q01_2 * q03 * q12
                - m112 * q01_2 * q03 * q12 + 2 * m022 * q01 * q02 * q03 * q12 - m112 * q02_2 * q03 * q12
                - 2 * m022 * q00 * q03_2 * q12 + m122 * q00 * q03_2 * q12 + 2 * m012 * q03_3 * q12 - m112 * q03_3 * q12
                + m022 * q00_2 * q10 * q12 + m022 * q01_2 * q10 * q12 + m022 * q02_2 * q10 * q12 + m022 * q03_2 * q10 * q12
                - m012 * q00 * q01 * q12_2 - m022 * q00 * q02 * q12_2 - m022 * q01 * q03 * q12_2 + m012 * q02 * q03 * q12_2
                - m102 * ( a ) * ( q00_2 - q00 * q10 + q01 * ( -q01 + q11 ) + q02 * ( -q02 + q12 ) + q03 * ( q03 - q13 ) )
                + ( 2 * m012 * q00_2 * q02 - m112 * q00_2 * q02 + 2 * m012 * q01_2 * q02 - m112 * q01_2 * q02
                    + 2 * m012 * q02_3 - m112 * q02_3 + 2 * m012 * q00 * q01 * q03 - m112 * q02 * q03_2 + m122 * q01 * ( a )
                    + m022 * ( -2 * q01 * ( q00_2 + q01_2 + q02_2 ) + 2 * q00 * q02 * q03 + (a)*q11 ) - m012 * (a)*q12 )
                      * q13
                - ( m012 * q00 * q01 + m022 * q00 * q02 + m022 * q01 * q03 - m012 * q02 * q03 ) * q13_2
                + m002
                      * ( q00_4 - q012_2 + q03_4 - q00_3 * q10 - q00 * ( 3 * ( q01_2 + q02_2 ) + q03_2 ) * q10
                          + q03_2 * ( 3 * q01 * q11 - q11_2 + ( 3 * q02 - q12 ) * q12 ) - 3 * ( q01_2 + q02_2 ) * q03 * q13
                          - q03_3 * q13 + q00_2 * ( 2 * q03_2 + 3 * q01 * q11 - q11_2 + 3 * q02 * q12 - q12_2 - q03 * q13 )
                          + ( q01_2 + q02_2 ) * ( q10_2 + q01 * q11 + q02 * q12 + q13_2 ) ) ),
        4
            * ( 2 * m013 * q00_3 * q01 - 2 * m113 * q00_3 * q01 + 2 * m013 * q00 * q01_3 - 2 * m113 * q00 * q01_3
                + 2 * m023 * q00_3 * q02 - 2 * m123 * q00_3 * q02 + 2 * m023 * q00 * q01_2 * q02 - 2 * m123 * q00 * q01_2 * q02
                + 2 * m013 * q00 * q01 * q02_2 - 2 * m113 * q00 * q01 * q02_2 + 2 * m023 * q00 * q02_3 - 2 * m123 * q00 * q02_3
                + 2 * m023 * q00_2 * q01 * q03 - 2 * m123 * q00_2 * q01 * q03 + 2 * m023 * q01_3 * q03 - 2 * m123 * q01_3 * q03
                - 2 * m013 * q00_2 * q02 * q03 + 2 * m113 * q00_2 * q02 * q03 - 2 * m013 * q01_2 * q02 * q03
                + 2 * m113 * q01_2 * q02 * q03 + 2 * m023 * q01 * q02_2 * q03 - 2 * m123 * q01 * q02_2 * q03
                - 2 * m013 * q02_3 * q03 + 2 * m113 * q02_3 * q03 + 2 * m013 * q00 * q01 * q03_2 - 2 * m113 * q00 * q01 * q03_2
                + 2 * m023 * q00 * q02 * q03_2 - 2 * m123 * q00 * q02 * q03_2 + 2 * m023 * q01 * q03_3 - 2 * m123 * q01 * q03_3
                - 2 * m013 * q02 * q03_3 + 2 * m113 * q02 * q03_3 + m113 * q00_2 * q01 * q10 - 2 * m013 * q01_3 * q10
                + m113 * q01_3 * q10 + m123 * q00_2 * q02 * q10 - 2 * m023 * q01_2 * q02 * q10 + m123 * q01_2 * q02 * q10
                - 2 * m013 * q01 * q02_2 * q10 + m113 * q01 * q02_2 * q10 - 2 * m023 * q02_3 * q10 + m123 * q02_3 * q10
                + 2 * m023 * q00 * q01 * q03 * q10 - 2 * m013 * q00 * q02 * q03 * q10 - 2 * m013 * q01 * q03_2 * q10
                + m113 * q01 * q03_2 * q10 - 2 * m023 * q02 * q03_2 * q10 + m123 * q02 * q03_2 * q10 - m013 * q00 * q01 * q10_2
                - m023 * q00 * q02 * q10_2 - m023 * q01 * q03 * q10_2 + m013 * q02 * q03 * q10_2 - 2 * m013 * q00_3 * q11
                + m113 * q00_3 * q11 + m113 * q00 * q01_2 * q11 + 2 * m023 * q00 * q01 * q02 * q11 - 2 * m013 * q00 * q02_2 * q11
                + m113 * q00 * q02_2 * q11 - 2 * m023 * q00_2 * q03 * q11 + m123 * q00_2 * q03 * q11
                + m123 * q01_2 * q03 * q11 - 2 * m013 * q01 * q02 * q03 * q11 - 2 * m023 * q02_2 * q03 * q11
                + m123 * q02_2 * q03 * q11 - 2 * m013 * q00 * q03_2 * q11 + m113 * q00 * q03_2 * q11 - 2 * m023 * q03_3 * q11
                + m123 * q03_3 * q11 + m013 * q00_2 * q10 * q11 + m013 * q01_2 * q10 * q11 + m013 * q02_2 * q10 * q11
                + m013 * q03_2 * q10 * q11 - m013 * q00 * q01 * q11_2 - m023 * q00 * q02 * q11_2 - m023 * q01 * q03 * q11_2
                + m013 * q02 * q03 * q11_2 - 2 * m023 * q00_3 * q12 + m123 * q00_3 * q12 - 2 * m023 * q00 * q01_2 * q12
                + m123 * q00 * q01_2 * q12 + 2 * m013 * q00 * q01 * q02 * q12 + m123 * q00 * q02_2 * q12
                + 2 * m013 * q00_2 * q03 * q12 - m113 * q00_2 * q03 * q12 + 2 * m013 * q01_2 * q03 * q12
                - m113 * q01_2 * q03 * q12 + 2 * m023 * q01 * q02 * q03 * q12 - m113 * q02_2 * q03 * q12
                - 2 * m023 * q00 * q03_2 * q12 + m123 * q00 * q03_2 * q12 + 2 * m013 * q03_3 * q12 - m113 * q03_3 * q12
                + m023 * q00_2 * q10 * q12 + m023 * q01_2 * q10 * q12 + m023 * q02_2 * q10 * q12 + m023 * q03_2 * q10 * q12
                - m013 * q00 * q01 * q12_2 - m023 * q00 * q02 * q12_2 - m023 * q01 * q03 * q12_2 + m013 * q02 * q03 * q12_2
                - m103 * ( a ) * ( q00_2 - q00 * q10 + q01 * ( -q01 + q11 ) + q02 * ( -q02 + q12 ) + q03 * ( q03 - q13 ) )
                - 2 * m023 * q00_2 * q01 * q13 + m123 * q00_2 * q01 * q13 - 2 * m023 * q01_3 * q13 + m123 * q01_3 * q13
                + 2 * m013 * q00_2 * q02 * q13 - m113 * q00_2 * q02 * q13 + 2 * m013 * q01_2 * q02 * q13
                - m113 * q01_2 * q02 * q13 - 2 * m023 * q01 * q02_2 * q13 + m123 * q01 * q02_2 * q13
                + 2 * m013 * q02_3 * q13 - m113 * q02_3 * q13 + 2 * m013 * q00 * q01 * q03 * q13
                + 2 * m023 * q00 * q02 * q03 * q13 + m123 * q01 * q03_2 * q13 - m113 * q02 * q03_2 * q13
                + m023 * q00_2 * q11 * q13 + m023 * q01_2 * q11 * q13 + m023 * q02_2 * q11 * q13 + m023 * q03_2 * q11 * q13
                - m013 * q00_2 * q12 * q13 - m013 * q01_2 * q12 * q13 - m013 * q02_2 * q12 * q13 - m013 * q03_2 * q12 * q13
                - m013 * q00 * q01 * q13_2 - m023 * q00 * q02 * q13_2 - m023 * q01 * q03 * q13_2 + m013 * q02 * q03 * q13_2
                + m003
                      * ( q00_4 - q012_2 + q03_4 - q00_3 * q10 - q00 * ( 3 * ( q01_2 + q02_2 ) + q03_2 ) * q10
                          + q03_2 * ( 3 * q01 * q11 - q11_2 + ( 3 * q02 - q12 ) * q12 ) - 3 * ( q01_2 + q02_2 ) * q03 * q13
                          - q03_3 * q13 + q00_2 * ( 2 * q03_2 + 3 * q01 * q11 - q11_2 + 3 * q02 * q12 - q12_2 - q03 * q13 )
                          + ( q01_2 + q02_2 ) * ( q10_2 + q01 * q11 + q02 * q12 + q13_2 ) ) ) );
    c2[0] = SRNumeratorDerivativeTerm(
        -2
            * ( 6 * m010 * q00_3 * q01 - 6 * m110 * q00_3 * q01 + 6 * m010 * q00 * q01_3 - 6 * m110 * q00 * q01_3
                + 6 * m020 * q00_3 * q02 - 6 * m120 * q00_3 * q02 + 6 * m020 * q00 * q01_2 * q02 - 6 * m120 * q00 * q01_2 * q02
                + 6 * m010 * q00 * q01 * q02_2 - 6 * m110 * q00 * q01 * q02_2 + 6 * m020 * q00 * q02_3 - 6 * m120 * q00 * q02_3
                + 6 * m020 * q00_2 * q01 * q03 - 6 * m120 * q00_2 * q01 * q03 + 6 * m020 * q01_3 * q03 - 6 * m120 * q01_3 * q03
                - 6 * m010 * q00_2 * q02 * q03 + 6 * m110 * q00_2 * q02 * q03 - 6 * m010 * q01_2 * q02 * q03
                + 6 * m110 * q01_2 * q02 * q03 + 6 * m020 * q01 * q02_2 * q03 - 6 * m120 * q01 * q02_2 * q03
                - 6 * m010 * q02_3 * q03 + 6 * m110 * q02_3 * q03 + 6 * m010 * q00 * q01 * q03_2 - 6 * m110 * q00 * q01 * q03_2
                + 6 * m020 * q00 * q02 * q03_2 - 6 * m120 * q00 * q02 * q03_2 + 6 * m020 * q01 * q03_3
                - 6 * m120 * q01 * q03_3 - 6 * m010 * q02 * q03_3 + 6 * m110 * q02 * q03_3 - 6 * m010 * q00_2 * q01 * q10
                + 7 * m110 * q00_2 * q01 * q10 - 6 * m010 * q01_3 * q10 + 5 * m110 * q01_3 * q10 - 6 * m020 * q00_2 * q02 * q10
                + 7 * m120 * q00_2 * q02 * q10 - 6 * m020 * q01_2 * q02 * q10 + 5 * m120 * q01_2 * q02 * q10
                - 6 * m010 * q01 * q02_2 * q10 + 5 * m110 * q01 * q02_2 * q10 - 6 * m020 * q02_3 * q10 + 5 * m120 * q02_3 * q10
                + 2 * m120 * q00 * q01 * q03 * q10 - 2 * m110 * q00 * q02 * q03 * q10 - 6 * m010 * q01 * q03_2 * q10
                + 5 * m110 * q01 * q03_2 * q10 - 6 * m020 * q02 * q03_2 * q10 + 5 * m120 * q02 * q03_2 * q10
                - m010 * q00 * q01 * q10_2 - m110 * q00 * q01 * q10_2 - m020 * q00 * q02 * q10_2 - m120 * q00 * q02 * q10_2
                - 3 * m020 * q01 * q03 * q10_2 + m120 * q01 * q03 * q10_2 + 3 * m010 * q02 * q03 * q10_2 - m110 * q02 * q03 * q10_2
                + m010 * q01 * q10_3 + m020 * q02 * q10_3 - 6 * m010 * q00_3 * q11 + 5 * m110 * q00_3 * q11
                - 6 * m010 * q00 * q01_2 * q11 + 7 * m110 * q00 * q01_2 * q11 + 2 * m120 * q00 * q01 * q02 * q11
                - 6 * m010 * q00 * q02_2 * q11 + 5 * m110 * q00 * q02_2 * q11 - 6 * m020 * q00_2 * q03 * q11
                + 5 * m120 * q00_2 * q03 * q11 - 6 * m020 * q01_2 * q03 * q11 + 7 * m120 * q01_2 * q03 * q11
                - 2 * m110 * q01 * q02 * q03 * q11 - 6 * m020 * q02_2 * q03 * q11 + 5 * m120 * q02_2 * q03 * q11
                - 6 * m010 * q00 * q03_2 * q11 + 5 * m110 * q00 * q03_2 * q11 - 6 * m020 * q03_3 * q11 + 5 * m120 * q03_3 * q11
                + 7 * m010 * q00_2 * q10 * q11 - 5 * m110 * q00_2 * q10 * q11 + 7 * m010 * q01_2 * q10 * q11
                - 5 * m110 * q01_2 * q10 * q11 + 2 * m020 * q01 * q02 * q10 * q11 - 2 * m120 * q01 * q02 * q10 * q11
                + 5 * m010 * q02_2 * q10 * q11 - 3 * m110 * q02_2 * q10 * q11 + 2 * m020 * q00 * q03 * q10 * q11
                - 2 * m120 * q00 * q03 * q10 * q11 + 5 * m010 * q03_2 * q10 * q11 - 3 * m110 * q03_2 * q10 * q11
                - m010 * q00 * q10_2 * q11 + m020 * q03 * q10_2 * q11 - m010 * q00 * q01 * q11_2 - m110 * q00 * q01 * q11_2
                - 3 * m020 * q00 * q02 * q11_2 + m120 * q00 * q02 * q11_2 - m020 * q01 * q03 * q11_2 - m120 * q01 * q03 * q11_2
                + 3 * m010 * q02 * q03 * q11_2 - m110 * q02 * q03 * q11_2 - m010 * q01 * q10 * q11_2 + m020 * q02 * q10 * q11_2
                + m010 * q00 * q11_3 + m020 * q03 * q11_3 - 6 * m020 * q00_3 * q12 + 5 * m120 * q00_3 * q12
                - 6 * m020 * q00 * q01_2 * q12 + 5 * m120 * q00 * q01_2 * q12 + 2 * m110 * q00 * q01 * q02 * q12
                - 6 * m020 * q00 * q02_2 * q12 + 7 * m120 * q00 * q02_2 * q12 + 6 * m010 * q00_2 * q03 * q12
                - 5 * m110 * q00_2 * q03 * q12 + 6 * m010 * q01_2 * q03 * q12 - 5 * m110 * q01_2 * q03 * q12
                + 2 * m120 * q01 * q02 * q03 * q12 + 6 * m010 * q02_2 * q03 * q12 - 7 * m110 * q02_2 * q03 * q12
                - 6 * m020 * q00 * q03_2 * q12 + 5 * m120 * q00 * q03_2 * q12 + 6 * m010 * q03_3 * q12 - 5 * m110 * q03_3 * q12
                + 7 * m020 * q00_2 * q10 * q12 - 5 * m120 * q00_2 * q10 * q12 + 5 * m020 * q01_2 * q10 * q12
                - 3 * m120 * q01_2 * q10 * q12 + 2 * m010 * q01 * q02 * q10 * q12 - 2 * m110 * q01 * q02 * q10 * q12
                + 7 * m020 * q02_2 * q10 * q12 - 5 * m120 * q02_2 * q10 * q12 - 2 * m010 * q00 * q03 * q10 * q12
                + 2 * m110 * q00 * q03 * q10 * q12 + 5 * m020 * q03_2 * q10 * q12 - 3 * m120 * q03_2 * q10 * q12
                - m020 * q00 * q10_2 * q12 - m010 * q03 * q10_2 * q12 + 2 * m020 * q00 * q01 * q11 * q12
                - 2 * m120 * q00 * q01 * q11 * q12 + 2 * m010 * q00 * q02 * q11 * q12 - 2 * m110 * q00 * q02 * q11 * q12
                - 2 * m010 * q01 * q03 * q11 * q12 + 2 * m110 * q01 * q03 * q11 * q12 + 2 * m020 * q02 * q03 * q11 * q12
                - 2 * m120 * q02 * q03 * q11 * q12 - 2 * m020 * q01 * q10 * q11 * q12 - 2 * m010 * q02 * q10 * q11 * q12
                + m020 * q00 * q11_2 * q12 - m010 * q03 * q11_2 * q12 - 3 * m010 * q00 * q01 * q12_2 + m110 * q00 * q01 * q12_2
                - m020 * q00 * q02 * q12_2 - m120 * q00 * q02 * q12_2 - 3 * m020 * q01 * q03 * q12_2 + m120 * q01 * q03 * q12_2
                + m010 * q02 * q03 * q12_2 + m110 * q02 * q03 * q12_2 + m010 * q01 * q10 * q12_2 - m020 * q02 * q10 * q12_2
                + m010 * q00 * q11 * q12_2 + m020 * q03 * q11 * q12_2 + m020 * q00 * q12_3 - m010 * q03 * q12_3
                + ( 6 * m010 * q00_2 * q02 - 5 * m110 * q00_2 * q02 + 6 * m010 * q01_2 * q02 - 5 * m110 * q01_2 * q02
                    + 6 * m010 * q02_3 - 5 * m110 * q02_3 + 2 * m110 * q00 * q01 * q03 + 6 * m010 * q02 * q03_2
                    - 7 * m110 * q02 * q03_2 - 2 * m010 * q00 * q02 * q10 + 2 * m110 * q00 * q02 * q10 + 2 * m010 * q01 * q03 * q10
                    - 2 * m110 * q01 * q03 * q10 - m010 * q02 * q10_2 - 2 * m010 * q01 * q02 * q11 + 2 * m110 * q01 * q02 * q11
                    + 2 * m010 * q00 * q03 * q11 - 2 * m110 * q00 * q03 * q11 - 2 * m010 * q03 * q10 * q11 - m010 * q02 * q11_2
                    + ( m110 * ( 3 * q00_2 + 3 * q01_2 + 5 * ( q02_2 + q03_2 ) )
                        - m010 * ( 5 * q00_2 + 7 * ( q02_2 + q03_2 ) - 2 * q00 * q10 + q01 * ( 5 * q01 - 2 * q11 ) ) )
                          * q12
                    + m010 * q02 * q12_2
                    + m120
                          * ( 5 * q01_3 - 2 * q02 * q03 * q10 + q00_2 * ( 5 * q01 - 3 * q11 ) - 5 * q01_2 * q11
                              - 3 * q02_2 * q11 - 5 * q03_2 * q11 + q01 * ( 5 * q02_2 + 7 * q03_2 - 2 * q02 * q12 )
                              + 2 * q00 * ( q02 * q03 - q01 * q10 - q03 * q12 ) )
                    + m020
                          * ( -6 * q01_3 + 2 * q02 * q03 * q10 + 7 * q01_2 * q11 + 5 * q02_2 * q11 + 7 * q03_2 * q11
                              + q00_2 * ( -6 * q01 + 5 * q11 ) - 2 * ( q03 * q10 + q02 * q11 ) * q12
                              + 2 * q00 * ( q01 * q10 - q10 * q11 + q03 * q12 )
                              + q01 * ( -6 * q02_2 - 6 * q03_2 + q10_2 - q11_2 + 2 * q02 * q12 + q12_2 ) ) )
                      * q13
                + ( m110 * q00 * q01 - 3 * m020 * q00 * q02 + m120 * q00 * q02 - m020 * q01 * q03 - m120 * q01 * q03
                    + m110 * q02 * q03 + m020 * q02 * q10 - m020 * q03 * q11 + m020 * q00 * q12
                    + m010 * ( q01 * q10 + q00 * ( -3 * q01 + q11 ) + q03 * ( q02 + q12 ) ) )
                      * q13_2
                + ( m020 * q01 - m010 * q02 ) * q13_3
                + m000
                      * ( 3 * q00_4 - 3 * q012_2 + 3 * q03_4 - 6 * q00_3 * q10 - 2 * q01 * q10_2 * q11 - 3 * q01_2 * q11_2
                          - q02_2 * q11_2 + 2 * ( q01_2 + q02_2 ) * ( 2 * q10_2 + 3 * q01 * q11 ) + 6 * q01_2 * q02 * q12
                          + 6 * q02_3 * q12 - 2 * q02 * q10_2 * q12 - 4 * q01 * q02 * q11 * q12 - q01_2 * q12_2
                          - 3 * q02_2 * q12_2 - 6 * q03_3 * q13 + 2 * q03 * ( -3 * q01_2 - 3 * q02_2 + q11_2 + q12_2 ) * q13
                          + 2 * ( 2 * q01_2 - q01 * q11 + q02 * ( 2 * q02 - q12 ) ) * q13_2
                          + 2 * q00 * q10 * ( -3 * q01_2 - 3 * q02_2 - 3 * q03_2 + q11_2 + q12_2 + 2 * q03 * q13 )
                          + q00_2 * ( 6 * q03_2 + 3 * q10_2 + 6 * q01 * q11 - 4 * q11_2 + 6 * q02 * q12 - 4 * q12_2 - 6 * q03 * q13 + q13_2 )
                          + q03_2 * ( q10_2 + 6 * q01 * q11 - 4 * q11_2 + 6 * q02 * q12 - 4 * q12_2 + 3 * q13_2 ) )
                + m100
                      * ( -3 * q00_4 + 3 * q012_2 - 3 * q03_4 + 6 * q00_3 * q10 + 3 * q01_2 * q11_2 + q02_2 * q11_2
                          - 2 * ( q01_2 + q02_2 ) * ( q10_2 + 3 * q01 * q11 ) - 6 * q01_2 * q02 * q12 - 6 * q02_3 * q12
                          + 4 * q01 * q02 * q11 * q12 + q01_2 * q12_2 + 3 * q02_2 * q12_2
                          + 2 * q00 * q10 * ( 2 * q01_2 + 2 * q02_2 + q03 * ( 3 * q03 - 2 * q13 ) )
                          + 4 * ( q01_2 + q02_2 ) * q03 * q13 + 6 * q03_3 * q13 - 2 * ( q01_2 + q02_2 ) * q13_2
                          - q00_2 * ( 6 * q03_2 + 3 * q10_2 + 4 * q01 * q11 - 2 * ( q11_2 + q12 * ( -2 * q02 + q12 ) ) - 6 * q03 * q13 + q13_2 )
                          - q03_2 * ( q10_2 + 4 * q01 * q11 - 2 * ( q11_2 - 2 * q02 * q12 + q12_2 ) + 3 * q13_2 ) ) ),
        -2
            * ( 6 * m011 * q00_3 * q01 - 6 * m111 * q00_3 * q01 + 6 * m011 * q00 * q01_3 - 6 * m111 * q00 * q01_3
                + 6 * m021 * q00_3 * q02 - 6 * m121 * q00_3 * q02 + 6 * m021 * q00 * q01_2 * q02 - 6 * m121 * q00 * q01_2 * q02
                + 6 * m011 * q00 * q01 * q02_2 - 6 * m111 * q00 * q01 * q02_2 + 6 * m021 * q00 * q02_3 - 6 * m121 * q00 * q02_3
                + 6 * m021 * q00_2 * q01 * q03 - 6 * m121 * q00_2 * q01 * q03 + 6 * m021 * q01_3 * q03 - 6 * m121 * q01_3 * q03
                - 6 * m011 * q00_2 * q02 * q03 + 6 * m111 * q00_2 * q02 * q03 - 6 * m011 * q01_2 * q02 * q03
                + 6 * m111 * q01_2 * q02 * q03 + 6 * m021 * q01 * q02_2 * q03 - 6 * m121 * q01 * q02_2 * q03
                - 6 * m011 * q02_3 * q03 + 6 * m111 * q02_3 * q03 + 6 * m011 * q00 * q01 * q03_2 - 6 * m111 * q00 * q01 * q03_2
                + 6 * m021 * q00 * q02 * q03_2 - 6 * m121 * q00 * q02 * q03_2 + 6 * m021 * q01 * q03_3
                - 6 * m121 * q01 * q03_3 - 6 * m011 * q02 * q03_3 + 6 * m111 * q02 * q03_3 - 6 * m011 * q00_2 * q01 * q10
                + 7 * m111 * q00_2 * q01 * q10 - 6 * m011 * q01_3 * q10 + 5 * m111 * q01_3 * q10 - 6 * m021 * q00_2 * q02 * q10
                + 7 * m121 * q00_2 * q02 * q10 - 6 * m021 * q01_2 * q02 * q10 + 5 * m121 * q01_2 * q02 * q10
                - 6 * m011 * q01 * q02_2 * q10 + 5 * m111 * q01 * q02_2 * q10 - 6 * m021 * q02_3 * q10 + 5 * m121 * q02_3 * q10
                + 2 * m121 * q00 * q01 * q03 * q10 - 2 * m111 * q00 * q02 * q03 * q10 - 6 * m011 * q01 * q03_2 * q10
                + 5 * m111 * q01 * q03_2 * q10 - 6 * m021 * q02 * q03_2 * q10 + 5 * m121 * q02 * q03_2 * q10
                - m011 * q00 * q01 * q10_2 - m111 * q00 * q01 * q10_2 - m021 * q00 * q02 * q10_2 - m121 * q00 * q02 * q10_2
                - 3 * m021 * q01 * q03 * q10_2 + m121 * q01 * q03 * q10_2 + 3 * m011 * q02 * q03 * q10_2 - m111 * q02 * q03 * q10_2
                + m011 * q01 * q10_3 + m021 * q02 * q10_3 - 6 * m011 * q00_3 * q11 + 5 * m111 * q00_3 * q11
                - 6 * m011 * q00 * q01_2 * q11 + 7 * m111 * q00 * q01_2 * q11 + 2 * m121 * q00 * q01 * q02 * q11
                - 6 * m011 * q00 * q02_2 * q11 + 5 * m111 * q00 * q02_2 * q11 - 6 * m021 * q00_2 * q03 * q11
                + 5 * m121 * q00_2 * q03 * q11 - 6 * m021 * q01_2 * q03 * q11 + 7 * m121 * q01_2 * q03 * q11
                - 2 * m111 * q01 * q02 * q03 * q11 - 6 * m021 * q02_2 * q03 * q11 + 5 * m121 * q02_2 * q03 * q11
                - 6 * m011 * q00 * q03_2 * q11 + 5 * m111 * q00 * q03_2 * q11 - 6 * m021 * q03_3 * q11 + 5 * m121 * q03_3 * q11
                + 7 * m011 * q00_2 * q10 * q11 - 5 * m111 * q00_2 * q10 * q11 + 7 * m011 * q01_2 * q10 * q11
                - 5 * m111 * q01_2 * q10 * q11 + 2 * m021 * q01 * q02 * q10 * q11 - 2 * m121 * q01 * q02 * q10 * q11
                + 5 * m011 * q02_2 * q10 * q11 - 3 * m111 * q02_2 * q10 * q11 + 2 * m021 * q00 * q03 * q10 * q11
                - 2 * m121 * q00 * q03 * q10 * q11 + 5 * m011 * q03_2 * q10 * q11 - 3 * m111 * q03_2 * q10 * q11
                - m011 * q00 * q10_2 * q11 + m021 * q03 * q10_2 * q11 - m011 * q00 * q01 * q11_2 - m111 * q00 * q01 * q11_2
                - 3 * m021 * q00 * q02 * q11_2 + m121 * q00 * q02 * q11_2 - m021 * q01 * q03 * q11_2 - m121 * q01 * q03 * q11_2
                + 3 * m011 * q02 * q03 * q11_2 - m111 * q02 * q03 * q11_2 - m011 * q01 * q10 * q11_2 + m021 * q02 * q10 * q11_2
                + m011 * q00 * q11_3 + m021 * q03 * q11_3 - 6 * m021 * q00_3 * q12 + 5 * m121 * q00_3 * q12
                - 6 * m021 * q00 * q01_2 * q12 + 5 * m121 * q00 * q01_2 * q12 + 2 * m111 * q00 * q01 * q02 * q12
                - 6 * m021 * q00 * q02_2 * q12 + 7 * m121 * q00 * q02_2 * q12 + 6 * m011 * q00_2 * q03 * q12
                - 5 * m111 * q00_2 * q03 * q12 + 6 * m011 * q01_2 * q03 * q12 - 5 * m111 * q01_2 * q03 * q12
                + 2 * m121 * q01 * q02 * q03 * q12 + 6 * m011 * q02_2 * q03 * q12 - 7 * m111 * q02_2 * q03 * q12
                - 6 * m021 * q00 * q03_2 * q12 + 5 * m121 * q00 * q03_2 * q12 + 6 * m011 * q03_3 * q12 - 5 * m111 * q03_3 * q12
                + 7 * m021 * q00_2 * q10 * q12 - 5 * m121 * q00_2 * q10 * q12 + 5 * m021 * q01_2 * q10 * q12
                - 3 * m121 * q01_2 * q10 * q12 + 2 * m011 * q01 * q02 * q10 * q12 - 2 * m111 * q01 * q02 * q10 * q12
                + 7 * m021 * q02_2 * q10 * q12 - 5 * m121 * q02_2 * q10 * q12 - 2 * m011 * q00 * q03 * q10 * q12
                + 2 * m111 * q00 * q03 * q10 * q12 + 5 * m021 * q03_2 * q10 * q12 - 3 * m121 * q03_2 * q10 * q12
                - m021 * q00 * q10_2 * q12 - m011 * q03 * q10_2 * q12 + 2 * m021 * q00 * q01 * q11 * q12
                - 2 * m121 * q00 * q01 * q11 * q12 + 2 * m011 * q00 * q02 * q11 * q12 - 2 * m111 * q00 * q02 * q11 * q12
                - 2 * m011 * q01 * q03 * q11 * q12 + 2 * m111 * q01 * q03 * q11 * q12 + 2 * m021 * q02 * q03 * q11 * q12
                - 2 * m121 * q02 * q03 * q11 * q12 - 2 * m021 * q01 * q10 * q11 * q12 - 2 * m011 * q02 * q10 * q11 * q12
                + m021 * q00 * q11_2 * q12 - m011 * q03 * q11_2 * q12 - 3 * m011 * q00 * q01 * q12_2 + m111 * q00 * q01 * q12_2
                - m021 * q00 * q02 * q12_2 - m121 * q00 * q02 * q12_2 - 3 * m021 * q01 * q03 * q12_2 + m121 * q01 * q03 * q12_2
                + m011 * q02 * q03 * q12_2 + m111 * q02 * q03 * q12_2 + m011 * q01 * q10 * q12_2 - m021 * q02 * q10 * q12_2
                + m011 * q00 * q11 * q12_2 + m021 * q03 * q11 * q12_2 + m021 * q00 * q12_3 - m011 * q03 * q12_3
                + ( 6 * m011 * q00_2 * q02 - 5 * m111 * q00_2 * q02 + 6 * m011 * q01_2 * q02 - 5 * m111 * q01_2 * q02
                    + 6 * m011 * q02_3 - 5 * m111 * q02_3 + 2 * m111 * q00 * q01 * q03 + 6 * m011 * q02 * q03_2
                    - 7 * m111 * q02 * q03_2 - 2 * m011 * q00 * q02 * q10 + 2 * m111 * q00 * q02 * q10 + 2 * m011 * q01 * q03 * q10
                    - 2 * m111 * q01 * q03 * q10 - m011 * q02 * q10_2 - 2 * m011 * q01 * q02 * q11 + 2 * m111 * q01 * q02 * q11
                    + 2 * m011 * q00 * q03 * q11 - 2 * m111 * q00 * q03 * q11 - 2 * m011 * q03 * q10 * q11 - m011 * q02 * q11_2
                    + ( m111 * ( 3 * q00_2 + 3 * q01_2 + 5 * ( q02_2 + q03_2 ) )
                        - m011 * ( 5 * q00_2 + 7 * ( q02_2 + q03_2 ) - 2 * q00 * q10 + q01 * ( 5 * q01 - 2 * q11 ) ) )
                          * q12
                    + m011 * q02 * q12_2
                    + m121
                          * ( 5 * q01_3 - 2 * q02 * q03 * q10 + q00_2 * ( 5 * q01 - 3 * q11 ) - 5 * q01_2 * q11
                              - 3 * q02_2 * q11 - 5 * q03_2 * q11 + q01 * ( 5 * q02_2 + 7 * q03_2 - 2 * q02 * q12 )
                              + 2 * q00 * ( q02 * q03 - q01 * q10 - q03 * q12 ) )
                    + m021
                          * ( -6 * q01_3 + 2 * q02 * q03 * q10 + 7 * q01_2 * q11 + 5 * q02_2 * q11 + 7 * q03_2 * q11
                              + q00_2 * ( -6 * q01 + 5 * q11 ) - 2 * ( q03 * q10 + q02 * q11 ) * q12
                              + 2 * q00 * ( q01 * q10 - q10 * q11 + q03 * q12 )
                              + q01 * ( -6 * q02_2 - 6 * q03_2 + q10_2 - q11_2 + 2 * q02 * q12 + q12_2 ) ) )
                      * q13
                + ( m111 * q00 * q01 - 3 * m021 * q00 * q02 + m121 * q00 * q02 - m021 * q01 * q03 - m121 * q01 * q03
                    + m111 * q02 * q03 + m021 * q02 * q10 - m021 * q03 * q11 + m021 * q00 * q12
                    + m011 * ( q01 * q10 + q00 * ( -3 * q01 + q11 ) + q03 * ( q02 + q12 ) ) )
                      * q13_2
                + ( m021 * q01 - m011 * q02 ) * q13_3
                + m001
                      * ( 3 * q00_4 - 3 * q012_2 + 3 * q03_4 - 6 * q00_3 * q10 - 2 * q01 * q10_2 * q11 - 3 * q01_2 * q11_2
                          - q02_2 * q11_2 + 2 * ( q01_2 + q02_2 ) * ( 2 * q10_2 + 3 * q01 * q11 ) + 6 * q01_2 * q02 * q12
                          + 6 * q02_3 * q12 - 2 * q02 * q10_2 * q12 - 4 * q01 * q02 * q11 * q12 - q01_2 * q12_2
                          - 3 * q02_2 * q12_2 - 6 * q03_3 * q13 + 2 * q03 * ( -3 * q01_2 - 3 * q02_2 + q11_2 + q12_2 ) * q13
                          + 2 * ( 2 * q01_2 - q01 * q11 + q02 * ( 2 * q02 - q12 ) ) * q13_2
                          + 2 * q00 * q10 * ( -3 * q01_2 - 3 * q02_2 - 3 * q03_2 + q11_2 + q12_2 + 2 * q03 * q13 )
                          + q00_2 * ( 6 * q03_2 + 3 * q10_2 + 6 * q01 * q11 - 4 * q11_2 + 6 * q02 * q12 - 4 * q12_2 - 6 * q03 * q13 + q13_2 )
                          + q03_2 * ( q10_2 + 6 * q01 * q11 - 4 * q11_2 + 6 * q02 * q12 - 4 * q12_2 + 3 * q13_2 ) )
                + m101
                      * ( -3 * q00_4 + 3 * q012_2 - 3 * q03_4 + 6 * q00_3 * q10 + 3 * q01_2 * q11_2 + q02_2 * q11_2
                          - 2 * ( q01_2 + q02_2 ) * ( q10_2 + 3 * q01 * q11 ) - 6 * q01_2 * q02 * q12 - 6 * q02_3 * q12
                          + 4 * q01 * q02 * q11 * q12 + q01_2 * q12_2 + 3 * q02_2 * q12_2
                          + 2 * q00 * q10 * ( 2 * q01_2 + 2 * q02_2 + q03 * ( 3 * q03 - 2 * q13 ) )
                          + 4 * ( q01_2 + q02_2 ) * q03 * q13 + 6 * q03_3 * q13 - 2 * ( q01_2 + q02_2 ) * q13_2
                          - q00_2 * ( 6 * q03_2 + 3 * q10_2 + 4 * q01 * q11 - 2 * ( q11_2 + q12 * ( -2 * q02 + q12 ) ) - 6 * q03 * q13 + q13_2 )
                          - q03_2 * ( q10_2 + 4 * q01 * q11 - 2 * ( q11_2 - 2 * q02 * q12 + q12_2 ) + 3 * q13_2 ) ) ),
        -2
            * ( 6 * m012 * q00_3 * q01 - 6 * m112 * q00_3 * q01 + 6 * m012 * q00 * q01_3 - 6 * m112 * q00 * q01_3
                + 6 * m022 * q00_3 * q02 - 6 * m122 * q00_3 * q02 + 6 * m022 * q00 * q01_2 * q02 - 6 * m122 * q00 * q01_2 * q02
                + 6 * m012 * q00 * q01 * q02_2 - 6 * m112 * q00 * q01 * q02_2 + 6 * m022 * q00 * q02_3 - 6 * m122 * q00 * q02_3
                + 6 * m022 * q00_2 * q01 * q03 - 6 * m122 * q00_2 * q01 * q03 + 6 * m022 * q01_3 * q03 - 6 * m122 * q01_3 * q03
                - 6 * m012 * q00_2 * q02 * q03 + 6 * m112 * q00_2 * q02 * q03 - 6 * m012 * q01_2 * q02 * q03
                + 6 * m112 * q01_2 * q02 * q03 + 6 * m022 * q01 * q02_2 * q03 - 6 * m122 * q01 * q02_2 * q03
                - 6 * m012 * q02_3 * q03 + 6 * m112 * q02_3 * q03 + 6 * m012 * q00 * q01 * q03_2 - 6 * m112 * q00 * q01 * q03_2
                + 6 * m022 * q00 * q02 * q03_2 - 6 * m122 * q00 * q02 * q03_2 + 6 * m022 * q01 * q03_3
                - 6 * m122 * q01 * q03_3 - 6 * m012 * q02 * q03_3 + 6 * m112 * q02 * q03_3 - 6 * m012 * q00_2 * q01 * q10
                + 7 * m112 * q00_2 * q01 * q10 - 6 * m012 * q01_3 * q10 + 5 * m112 * q01_3 * q10 - 6 * m022 * q00_2 * q02 * q10
                + 7 * m122 * q00_2 * q02 * q10 - 6 * m022 * q01_2 * q02 * q10 + 5 * m122 * q01_2 * q02 * q10
                - 6 * m012 * q01 * q02_2 * q10 + 5 * m112 * q01 * q02_2 * q10 - 6 * m022 * q02_3 * q10 + 5 * m122 * q02_3 * q10
                + 2 * m122 * q00 * q01 * q03 * q10 - 2 * m112 * q00 * q02 * q03 * q10 - 6 * m012 * q01 * q03_2 * q10
                + 5 * m112 * q01 * q03_2 * q10 - 6 * m022 * q02 * q03_2 * q10 + 5 * m122 * q02 * q03_2 * q10
                - m012 * q00 * q01 * q10_2 - m112 * q00 * q01 * q10_2 - m022 * q00 * q02 * q10_2 - m122 * q00 * q02 * q10_2
                - 3 * m022 * q01 * q03 * q10_2 + m122 * q01 * q03 * q10_2 + 3 * m012 * q02 * q03 * q10_2 - m112 * q02 * q03 * q10_2
                + m012 * q01 * q10_3 + m022 * q02 * q10_3 - 6 * m012 * q00_3 * q11 + 5 * m112 * q00_3 * q11
                - 6 * m012 * q00 * q01_2 * q11 + 7 * m112 * q00 * q01_2 * q11 + 2 * m122 * q00 * q01 * q02 * q11
                - 6 * m012 * q00 * q02_2 * q11 + 5 * m112 * q00 * q02_2 * q11 - 6 * m022 * q00_2 * q03 * q11
                + 5 * m122 * q00_2 * q03 * q11 - 6 * m022 * q01_2 * q03 * q11 + 7 * m122 * q01_2 * q03 * q11
                - 2 * m112 * q01 * q02 * q03 * q11 - 6 * m022 * q02_2 * q03 * q11 + 5 * m122 * q02_2 * q03 * q11
                - 6 * m012 * q00 * q03_2 * q11 + 5 * m112 * q00 * q03_2 * q11 - 6 * m022 * q03_3 * q11 + 5 * m122 * q03_3 * q11
                + 7 * m012 * q00_2 * q10 * q11 - 5 * m112 * q00_2 * q10 * q11 + 7 * m012 * q01_2 * q10 * q11
                - 5 * m112 * q01_2 * q10 * q11 + 2 * m022 * q01 * q02 * q10 * q11 - 2 * m122 * q01 * q02 * q10 * q11
                + 5 * m012 * q02_2 * q10 * q11 - 3 * m112 * q02_2 * q10 * q11 + 2 * m022 * q00 * q03 * q10 * q11
                - 2 * m122 * q00 * q03 * q10 * q11 + 5 * m012 * q03_2 * q10 * q11 - 3 * m112 * q03_2 * q10 * q11
                - m012 * q00 * q10_2 * q11 + m022 * q03 * q10_2 * q11 - m012 * q00 * q01 * q11_2 - m112 * q00 * q01 * q11_2
                - 3 * m022 * q00 * q02 * q11_2 + m122 * q00 * q02 * q11_2 - m022 * q01 * q03 * q11_2 - m122 * q01 * q03 * q11_2
                + 3 * m012 * q02 * q03 * q11_2 - m112 * q02 * q03 * q11_2 - m012 * q01 * q10 * q11_2 + m022 * q02 * q10 * q11_2
                + m012 * q00 * q11_3 + m022 * q03 * q11_3 - 6 * m022 * q00_3 * q12 + 5 * m122 * q00_3 * q12
                - 6 * m022 * q00 * q01_2 * q12 + 5 * m122 * q00 * q01_2 * q12 + 2 * m112 * q00 * q01 * q02 * q12
                - 6 * m022 * q00 * q02_2 * q12 + 7 * m122 * q00 * q02_2 * q12 + 6 * m012 * q00_2 * q03 * q12
                - 5 * m112 * q00_2 * q03 * q12 + 6 * m012 * q01_2 * q03 * q12 - 5 * m112 * q01_2 * q03 * q12
                + 2 * m122 * q01 * q02 * q03 * q12 + 6 * m012 * q02_2 * q03 * q12 - 7 * m112 * q02_2 * q03 * q12
                - 6 * m022 * q00 * q03_2 * q12 + 5 * m122 * q00 * q03_2 * q12 + 6 * m012 * q03_3 * q12 - 5 * m112 * q03_3 * q12
                + 7 * m022 * q00_2 * q10 * q12 - 5 * m122 * q00_2 * q10 * q12 + 5 * m022 * q01_2 * q10 * q12
                - 3 * m122 * q01_2 * q10 * q12 + 2 * m012 * q01 * q02 * q10 * q12 - 2 * m112 * q01 * q02 * q10 * q12
                + 7 * m022 * q02_2 * q10 * q12 - 5 * m122 * q02_2 * q10 * q12 - 2 * m012 * q00 * q03 * q10 * q12
                + 2 * m112 * q00 * q03 * q10 * q12 + 5 * m022 * q03_2 * q10 * q12 - 3 * m122 * q03_2 * q10 * q12
                - m022 * q00 * q10_2 * q12 - m012 * q03 * q10_2 * q12 + 2 * m022 * q00 * q01 * q11 * q12
                - 2 * m122 * q00 * q01 * q11 * q12 + 2 * m012 * q00 * q02 * q11 * q12 - 2 * m112 * q00 * q02 * q11 * q12
                - 2 * m012 * q01 * q03 * q11 * q12 + 2 * m112 * q01 * q03 * q11 * q12 + 2 * m022 * q02 * q03 * q11 * q12
                - 2 * m122 * q02 * q03 * q11 * q12 - 2 * m022 * q01 * q10 * q11 * q12 - 2 * m012 * q02 * q10 * q11 * q12
                + m022 * q00 * q11_2 * q12 - m012 * q03 * q11_2 * q12 - 3 * m012 * q00 * q01 * q12_2 + m112 * q00 * q01 * q12_2
                - m022 * q00 * q02 * q12_2 - m122 * q00 * q02 * q12_2 - 3 * m022 * q01 * q03 * q12_2 + m122 * q01 * q03 * q12_2
                + m012 * q02 * q03 * q12_2 + m112 * q02 * q03 * q12_2 + m012 * q01 * q10 * q12_2 - m022 * q02 * q10 * q12_2
                + m012 * q00 * q11 * q12_2 + m022 * q03 * q11 * q12_2 + m022 * q00 * q12_3 - m012 * q03 * q12_3
                + ( 6 * m012 * q00_2 * q02 - 5 * m112 * q00_2 * q02 + 6 * m012 * q01_2 * q02 - 5 * m112 * q01_2 * q02
                    + 6 * m012 * q02_3 - 5 * m112 * q02_3 + 2 * m112 * q00 * q01 * q03 + 6 * m012 * q02 * q03_2
                    - 7 * m112 * q02 * q03_2 - 2 * m012 * q00 * q02 * q10 + 2 * m112 * q00 * q02 * q10 + 2 * m012 * q01 * q03 * q10
                    - 2 * m112 * q01 * q03 * q10 - m012 * q02 * q10_2 - 2 * m012 * q01 * q02 * q11 + 2 * m112 * q01 * q02 * q11
                    + 2 * m012 * q00 * q03 * q11 - 2 * m112 * q00 * q03 * q11 - 2 * m012 * q03 * q10 * q11 - m012 * q02 * q11_2
                    + ( m112 * ( 3 * q00_2 + 3 * q01_2 + 5 * ( q02_2 + q03_2 ) )
                        - m012 * ( 5 * q00_2 + 7 * ( q02_2 + q03_2 ) - 2 * q00 * q10 + q01 * ( 5 * q01 - 2 * q11 ) ) )
                          * q12
                    + m012 * q02 * q12_2
                    + m122
                          * ( 5 * q01_3 - 2 * q02 * q03 * q10 + q00_2 * ( 5 * q01 - 3 * q11 ) - 5 * q01_2 * q11
                              - 3 * q02_2 * q11 - 5 * q03_2 * q11 + q01 * ( 5 * q02_2 + 7 * q03_2 - 2 * q02 * q12 )
                              + 2 * q00 * ( q02 * q03 - q01 * q10 - q03 * q12 ) )
                    + m022
                          * ( -6 * q01_3 + 2 * q02 * q03 * q10 + 7 * q01_2 * q11 + 5 * q02_2 * q11 + 7 * q03_2 * q11
                              + q00_2 * ( -6 * q01 + 5 * q11 ) - 2 * ( q03 * q10 + q02 * q11 ) * q12
                              + 2 * q00 * ( q01 * q10 - q10 * q11 + q03 * q12 )
                              + q01 * ( -6 * q02_2 - 6 * q03_2 + q10_2 - q11_2 + 2 * q02 * q12 + q12_2 ) ) )
                      * q13
                + ( m112 * q00 * q01 - 3 * m022 * q00 * q02 + m122 * q00 * q02 - m022 * q01 * q03 - m122 * q01 * q03
                    + m112 * q02 * q03 + m022 * q02 * q10 - m022 * q03 * q11 + m022 * q00 * q12
                    + m012 * ( q01 * q10 + q00 * ( -3 * q01 + q11 ) + q03 * ( q02 + q12 ) ) )
                      * q13_2
                + ( m022 * q01 - m012 * q02 ) * q13_3
                + m002
                      * ( 3 * q00_4 - 3 * q012_2 + 3 * q03_4 - 6 * q00_3 * q10 - 2 * q01 * q10_2 * q11 - 3 * q01_2 * q11_2
                          - q02_2 * q11_2 + 2 * ( q01_2 + q02_2 ) * ( 2 * q10_2 + 3 * q01 * q11 ) + 6 * q01_2 * q02 * q12
                          + 6 * q02_3 * q12 - 2 * q02 * q10_2 * q12 - 4 * q01 * q02 * q11 * q12 - q01_2 * q12_2
                          - 3 * q02_2 * q12_2 - 6 * q03_3 * q13 + 2 * q03 * ( -3 * q01_2 - 3 * q02_2 + q11_2 + q12_2 ) * q13
                          + 2 * ( 2 * q01_2 - q01 * q11 + q02 * ( 2 * q02 - q12 ) ) * q13_2
                          + 2 * q00 * q10 * ( -3 * q01_2 - 3 * q02_2 - 3 * q03_2 + q11_2 + q12_2 + 2 * q03 * q13 )
                          + q00_2 * ( 6 * q03_2 + 3 * q10_2 + 6 * q01 * q11 - 4 * q11_2 + 6 * q02 * q12 - 4 * q12_2 - 6 * q03 * q13 + q13_2 )
                          + q03_2 * ( q10_2 + 6 * q01 * q11 - 4 * q11_2 + 6 * q02 * q12 - 4 * q12_2 + 3 * q13_2 ) )
                + m102
                      * ( -3 * q00_4 + 3 * q012_2 - 3 * q03_4 + 6 * q00_3 * q10 + 3 * q01_2 * q11_2 + q02_2 * q11_2
                          - 2 * ( q01_2 + q02_2 ) * ( q10_2 + 3 * q01 * q11 ) - 6 * q01_2 * q02 * q12 - 6 * q02_3 * q12
                          + 4 * q01 * q02 * q11 * q12 + q01_2 * q12_2 + 3 * q02_2 * q12_2
                          + 2 * q00 * q10 * ( 2 * q01_2 + 2 * q02_2 + q03 * ( 3 * q03 - 2 * q13 ) )
                          + 4 * ( q01_2 + q02_2 ) * q03 * q13 + 6 * q03_3 * q13 - 2 * ( q01_2 + q02_2 ) * q13_2
                          - q00_2 * ( 6 * q03_2 + 3 * q10_2 + 4 * q01 * q11 - 2 * ( q11_2 + q12 * ( -2 * q02 + q12 ) ) - 6 * q03 * q13 + q13_2 )
                          - q03_2 * ( q10_2 + 4 * q01 * q11 - 2 * ( q11_2 - 2 * q02 * q12 + q12_2 ) + 3 * q13_2 ) ) ),
        -2
            * ( 6 * m013 * q00_3 * q01 - 6 * m113 * q00_3 * q01 + 6 * m013 * q00 * q01_3 - 6 * m113 * q00 * q01_3
                + 6 * m023 * q00_3 * q02 - 6 * m123 * q00_3 * q02 + 6 * m023 * q00 * q01_2 * q02 - 6 * m123 * q00 * q01_2 * q02
                + 6 * m013 * q00 * q01 * q02_2 - 6 * m113 * q00 * q01 * q02_2 + 6 * m023 * q00 * q02_3 - 6 * m123 * q00 * q02_3
                + 6 * m023 * q00_2 * q01 * q03 - 6 * m123 * q00_2 * q01 * q03 + 6 * m023 * q01_3 * q03 - 6 * m123 * q01_3 * q03
                - 6 * m013 * q00_2 * q02 * q03 + 6 * m113 * q00_2 * q02 * q03 - 6 * m013 * q01_2 * q02 * q03
                + 6 * m113 * q01_2 * q02 * q03 + 6 * m023 * q01 * q02_2 * q03 - 6 * m123 * q01 * q02_2 * q03
                - 6 * m013 * q02_3 * q03 + 6 * m113 * q02_3 * q03 + 6 * m013 * q00 * q01 * q03_2 - 6 * m113 * q00 * q01 * q03_2
                + 6 * m023 * q00 * q02 * q03_2 - 6 * m123 * q00 * q02 * q03_2 + 6 * m023 * q01 * q03_3 - 6 * m123 * q01 * q03_3
                - 6 * m013 * q02 * q03_3 + 6 * m113 * q02 * q03_3 - 6 * m013 * q00_2 * q01 * q10 + 7 * m113 * q00_2 * q01 * q10
                - 6 * m013 * q01_3 * q10 + 5 * m113 * q01_3 * q10 - 6 * m023 * q00_2 * q02 * q10 + 7 * m123 * q00_2 * q02 * q10
                - 6 * m023 * q01_2 * q02 * q10 + 5 * m123 * q01_2 * q02 * q10 - 6 * m013 * q01 * q02_2 * q10
                + 5 * m113 * q01 * q02_2 * q10 - 6 * m023 * q02_3 * q10 + 5 * m123 * q02_3 * q10 + 2 * m123 * q00 * q01 * q03 * q10
                - 2 * m113 * q00 * q02 * q03 * q10 - 6 * m013 * q01 * q03_2 * q10 + 5 * m113 * q01 * q03_2 * q10
                - 6 * m023 * q02 * q03_2 * q10 + 5 * m123 * q02 * q03_2 * q10 - m013 * q00 * q01 * q10_2
                - m113 * q00 * q01 * q10_2 - m023 * q00 * q02 * q10_2 - m123 * q00 * q02 * q10_2 - 3 * m023 * q01 * q03 * q10_2
                + m123 * q01 * q03 * q10_2 + 3 * m013 * q02 * q03 * q10_2 - m113 * q02 * q03 * q10_2 + m013 * q01 * q10_3
                + m023 * q02 * q10_3 - 6 * m013 * q00_3 * q11 + 5 * m113 * q00_3 * q11 - 6 * m013 * q00 * q01_2 * q11
                + 7 * m113 * q00 * q01_2 * q11 + 2 * m123 * q00 * q01 * q02 * q11 - 6 * m013 * q00 * q02_2 * q11
                + 5 * m113 * q00 * q02_2 * q11 - 6 * m023 * q00_2 * q03 * q11 + 5 * m123 * q00_2 * q03 * q11
                - 6 * m023 * q01_2 * q03 * q11 + 7 * m123 * q01_2 * q03 * q11 - 2 * m113 * q01 * q02 * q03 * q11
                - 6 * m023 * q02_2 * q03 * q11 + 5 * m123 * q02_2 * q03 * q11 - 6 * m013 * q00 * q03_2 * q11
                + 5 * m113 * q00 * q03_2 * q11 - 6 * m023 * q03_3 * q11 + 5 * m123 * q03_3 * q11 + 7 * m013 * q00_2 * q10 * q11
                - 5 * m113 * q00_2 * q10 * q11 + 7 * m013 * q01_2 * q10 * q11 - 5 * m113 * q01_2 * q10 * q11
                + 2 * m023 * q01 * q02 * q10 * q11 - 2 * m123 * q01 * q02 * q10 * q11 + 5 * m013 * q02_2 * q10 * q11
                - 3 * m113 * q02_2 * q10 * q11 + 2 * m023 * q00 * q03 * q10 * q11 - 2 * m123 * q00 * q03 * q10 * q11
                + 5 * m013 * q03_2 * q10 * q11 - 3 * m113 * q03_2 * q10 * q11 - m013 * q00 * q10_2 * q11
                + m023 * q03 * q10_2 * q11 - m013 * q00 * q01 * q11_2 - m113 * q00 * q01 * q11_2 - 3 * m023 * q00 * q02 * q11_2
                + m123 * q00 * q02 * q11_2 - m023 * q01 * q03 * q11_2 - m123 * q01 * q03 * q11_2 + 3 * m013 * q02 * q03 * q11_2
                - m113 * q02 * q03 * q11_2 - m013 * q01 * q10 * q11_2 + m023 * q02 * q10 * q11_2 + m013 * q00 * q11_3
                + m023 * q03 * q11_3 - 6 * m023 * q00_3 * q12 + 5 * m123 * q00_3 * q12 - 6 * m023 * q00 * q01_2 * q12
                + 5 * m123 * q00 * q01_2 * q12 + 2 * m113 * q00 * q01 * q02 * q12 - 6 * m023 * q00 * q02_2 * q12
                + 7 * m123 * q00 * q02_2 * q12 + 6 * m013 * q00_2 * q03 * q12 - 5 * m113 * q00_2 * q03 * q12
                + 6 * m013 * q01_2 * q03 * q12 - 5 * m113 * q01_2 * q03 * q12 + 2 * m123 * q01 * q02 * q03 * q12
                + 6 * m013 * q02_2 * q03 * q12 - 7 * m113 * q02_2 * q03 * q12 - 6 * m023 * q00 * q03_2 * q12
                + 5 * m123 * q00 * q03_2 * q12 + 6 * m013 * q03_3 * q12 - 5 * m113 * q03_3 * q12 + 7 * m023 * q00_2 * q10 * q12
                - 5 * m123 * q00_2 * q10 * q12 + 5 * m023 * q01_2 * q10 * q12 - 3 * m123 * q01_2 * q10 * q12
                + 2 * m013 * q01 * q02 * q10 * q12 - 2 * m113 * q01 * q02 * q10 * q12 + 7 * m023 * q02_2 * q10 * q12
                - 5 * m123 * q02_2 * q10 * q12 - 2 * m013 * q00 * q03 * q10 * q12 + 2 * m113 * q00 * q03 * q10 * q12
                + 5 * m023 * q03_2 * q10 * q12 - 3 * m123 * q03_2 * q10 * q12 - m023 * q00 * q10_2 * q12
                - m013 * q03 * q10_2 * q12 + 2 * m023 * q00 * q01 * q11 * q12 - 2 * m123 * q00 * q01 * q11 * q12
                + 2 * m013 * q00 * q02 * q11 * q12 - 2 * m113 * q00 * q02 * q11 * q12 - 2 * m013 * q01 * q03 * q11 * q12
                + 2 * m113 * q01 * q03 * q11 * q12 + 2 * m023 * q02 * q03 * q11 * q12 - 2 * m123 * q02 * q03 * q11 * q12
                - 2 * m023 * q01 * q10 * q11 * q12 - 2 * m013 * q02 * q10 * q11 * q12 + m023 * q00 * q11_2 * q12
                - m013 * q03 * q11_2 * q12 - 3 * m013 * q00 * q01 * q12_2 + m113 * q00 * q01 * q12_2 - m023 * q00 * q02 * q12_2
                - m123 * q00 * q02 * q12_2 - 3 * m023 * q01 * q03 * q12_2 + m123 * q01 * q03 * q12_2 + m013 * q02 * q03 * q12_2
                + m113 * q02 * q03 * q12_2 + m013 * q01 * q10 * q12_2 - m023 * q02 * q10 * q12_2 + m013 * q00 * q11 * q12_2
                + m023 * q03 * q11 * q12_2 + m023 * q00 * q12_3 - m013 * q03 * q12_3 - 6 * m023 * q00_2 * q01 * q13
                + 5 * m123 * q00_2 * q01 * q13 - 6 * m023 * q01_3 * q13 + 5 * m123 * q01_3 * q13 + 6 * m013 * q00_2 * q02 * q13
                - 5 * m113 * q00_2 * q02 * q13 + 6 * m013 * q01_2 * q02 * q13 - 5 * m113 * q01_2 * q02 * q13
                - 6 * m023 * q01 * q02_2 * q13 + 5 * m123 * q01 * q02_2 * q13 + 6 * m013 * q02_3 * q13 - 5 * m113 * q02_3 * q13
                + 2 * m113 * q00 * q01 * q03 * q13 + 2 * m123 * q00 * q02 * q03 * q13 - 6 * m023 * q01 * q03_2 * q13
                + 7 * m123 * q01 * q03_2 * q13 + 6 * m013 * q02 * q03_2 * q13 - 7 * m113 * q02 * q03_2 * q13
                + 2 * m023 * q00 * q01 * q10 * q13 - 2 * m123 * q00 * q01 * q10 * q13 - 2 * m013 * q00 * q02 * q10 * q13
                + 2 * m113 * q00 * q02 * q10 * q13 + 2 * m013 * q01 * q03 * q10 * q13 - 2 * m113 * q01 * q03 * q10 * q13
                + 2 * m023 * q02 * q03 * q10 * q13 - 2 * m123 * q02 * q03 * q10 * q13 + m023 * q01 * q10_2 * q13
                - m013 * q02 * q10_2 * q13 + 5 * m023 * q00_2 * q11 * q13 - 3 * m123 * q00_2 * q11 * q13
                + 7 * m023 * q01_2 * q11 * q13 - 5 * m123 * q01_2 * q11 * q13 - 2 * m013 * q01 * q02 * q11 * q13
                + 2 * m113 * q01 * q02 * q11 * q13 + 5 * m023 * q02_2 * q11 * q13 - 3 * m123 * q02_2 * q11 * q13
                + 2 * m013 * q00 * q03 * q11 * q13 - 2 * m113 * q00 * q03 * q11 * q13 + 7 * m023 * q03_2 * q11 * q13
                - 5 * m123 * q03_2 * q11 * q13 - 2 * m023 * q00 * q10 * q11 * q13 - 2 * m013 * q03 * q10 * q11 * q13
                - m023 * q01 * q11_2 * q13 - m013 * q02 * q11_2 * q13 - 5 * m013 * q00_2 * q12 * q13
                + 3 * m113 * q00_2 * q12 * q13 - 5 * m013 * q01_2 * q12 * q13 + 3 * m113 * q01_2 * q12 * q13
                + 2 * m023 * q01 * q02 * q12 * q13 - 2 * m123 * q01 * q02 * q12 * q13 - 7 * m013 * q02_2 * q12 * q13
                + 5 * m113 * q02_2 * q12 * q13 + 2 * m023 * q00 * q03 * q12 * q13 - 2 * m123 * q00 * q03 * q12 * q13
                - 7 * m013 * q03_2 * q12 * q13 + 5 * m113 * q03_2 * q12 * q13 + 2 * m013 * q00 * q10 * q12 * q13
                - 2 * m023 * q03 * q10 * q12 * q13 + 2 * m013 * q01 * q11 * q12 * q13 - 2 * m023 * q02 * q11 * q12 * q13
                + m023 * q01 * q12_2 * q13 + m013 * q02 * q12_2 * q13 - 3 * m013 * q00 * q01 * q13_2
                + m113 * q00 * q01 * q13_2 - 3 * m023 * q00 * q02 * q13_2 + m123 * q00 * q02 * q13_2
                - m023 * q01 * q03 * q13_2 - m123 * q01 * q03 * q13_2 + m013 * q02 * q03 * q13_2 + m113 * q02 * q03 * q13_2
                + m013 * q01 * q10 * q13_2 + m023 * q02 * q10 * q13_2 + m013 * q00 * q11 * q13_2 - m023 * q03 * q11 * q13_2
                + m023 * q00 * q12 * q13_2 + m013 * q03 * q12 * q13_2 + m023 * q01 * q13_3 - m013 * q02 * q13_3
                + m003
                      * ( 3 * q00_4 - 3 * q012_2 + 3 * q03_4 - 6 * q00_3 * q10 - 2 * q01 * q10_2 * q11 - 3 * q01_2 * q11_2
                          - q02_2 * q11_2 + 2 * ( q01_2 + q02_2 ) * ( 2 * q10_2 + 3 * q01 * q11 ) + 6 * q01_2 * q02 * q12
                          + 6 * q02_3 * q12 - 2 * q02 * q10_2 * q12 - 4 * q01 * q02 * q11 * q12 - q01_2 * q12_2
                          - 3 * q02_2 * q12_2 - 6 * q03_3 * q13 + 2 * q03 * ( -3 * q01_2 - 3 * q02_2 + q11_2 + q12_2 ) * q13
                          + 2 * ( 2 * q01_2 - q01 * q11 + q02 * ( 2 * q02 - q12 ) ) * q13_2
                          + 2 * q00 * q10 * ( -3 * q01_2 - 3 * q02_2 - 3 * q03_2 + q11_2 + q12_2 + 2 * q03 * q13 )
                          + q00_2 * ( 6 * q03_2 + 3 * q10_2 + 6 * q01 * q11 - 4 * q11_2 + 6 * q02 * q12 - 4 * q12_2 - 6 * q03 * q13 + q13_2 )
                          + q03_2 * ( q10_2 + 6 * q01 * q11 - 4 * q11_2 + 6 * q02 * q12 - 4 * q12_2 + 3 * q13_2 ) )
                + m103
                      * ( -3 * q00_4 + 3 * q012_2 - 3 * q03_4 + 6 * q00_3 * q10 + 3 * q01_2 * q11_2 + q02_2 * q11_2
                          - 2 * ( q01_2 + q02_2 ) * ( q10_2 + 3 * q01 * q11 ) - 6 * q01_2 * q02 * q12 - 6 * q02_3 * q12
                          + 4 * q01 * q02 * q11 * q12 + q01_2 * q12_2 + 3 * q02_2 * q12_2
                          + 2 * q00 * q10 * ( 2 * q01_2 + 2 * q02_2 + q03 * ( 3 * q03 - 2 * q13 ) )
                          + 4 * ( q01_2 + q02_2 ) * q03 * q13 + 6 * q03_3 * q13 - 2 * ( q01_2 + q02_2 ) * q13_2
                          - q00_2 * ( 6 * q03_2 + 3 * q10_2 + 4 * q01 * q11 - 2 * ( q11_2 + q12 * ( -2 * q02 + q12 ) ) - 6 * q03 * q13 + q13_2 )
                          - q03_2 * ( q10_2 + 4 * q01 * q11 - 2 * ( q11_2 - 2 * q02 * q12 + q12_2 ) + 3 * q13_2 ) ) ) );
    c3[0] = SRNumeratorDerivativeTerm(
        4 * ( a - q00 * q10 - q01 * q11 - q02 * q12 - q03 * q13 )
            * ( 2
                    * ( ( m020 - m120 ) * ( ( q00 - q10 ) * ( q02 - q12 ) + ( q01 - q11 ) * ( q03 - q13 ) )
                        + m010 * ( ( q00 - q10 ) * ( q01 - q11 ) - ( q02 - q12 ) * ( q03 - q13 ) )
                        + m110 * ( -( ( q00 - q10 ) * ( q01 - q11 ) ) + ( q02 - q12 ) * ( q03 - q13 ) ) )
                + m000 * ( q00_2 - 2 * q00 * q10 + q10_2 - q01_11_2 - q02_12_2 + q03_13_2 )
                + m100 * ( -q00_10_2 + q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ),
        4 * ( a - q00 * q10 - q01 * q11 - q02 * q12 - q03 * q13 )
            * ( 2
                    * ( ( m021 - m121 ) * ( ( q00 - q10 ) * ( q02 - q12 ) + ( q01 - q11 ) * ( q03 - q13 ) )
                        + m011 * ( ( q00 - q10 ) * ( q01 - q11 ) - ( q02 - q12 ) * ( q03 - q13 ) )
                        + m111 * ( -( ( q00 - q10 ) * ( q01 - q11 ) ) + ( q02 - q12 ) * ( q03 - q13 ) ) )
                + m001 * ( q00_2 - 2 * q00 * q10 + q10_2 - q01_11_2 - q02_12_2 + q03_13_2 )
                + m101 * ( -q00_10_2 + q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ),
        4 * ( a - q00 * q10 - q01 * q11 - q02 * q12 - q03 * q13 )
            * ( 2
                    * ( ( m022 - m122 ) * ( ( q00 - q10 ) * ( q02 - q12 ) + ( q01 - q11 ) * ( q03 - q13 ) )
                        + m012 * ( ( q00 - q10 ) * ( q01 - q11 ) - ( q02 - q12 ) * ( q03 - q13 ) )
                        + m112 * ( -( ( q00 - q10 ) * ( q01 - q11 ) ) + ( q02 - q12 ) * ( q03 - q13 ) ) )
                + m002 * ( q00_2 - 2 * q00 * q10 + q10_2 - q01_11_2 - q02_12_2 + q03_13_2 )
                + m102 * ( -q00_10_2 + q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ),
        4 * ( a - q00 * q10 - q01 * q11 - q02 * q12 - q03 * q13 )
            * ( 2 * m013 * q00 * q01 - 2 * m113 * q00 * q01 + 2 * m023 * q00 * q02 - 2 * m123 * q00 * q02 + 2 * m023 * q01 * q03
                - 2 * m123 * q01 * q03 - 2 * m013 * q02 * q03 + 2 * m113 * q02 * q03 - 2 * m013 * q01 * q10
                + 2 * m113 * q01 * q10 - 2 * m023 * q02 * q10 + 2 * m123 * q02 * q10 - 2 * m013 * q00 * q11
                + 2 * m113 * q00 * q11 - 2 * m023 * q03 * q11 + 2 * m123 * q03 * q11 + 2 * m013 * q10 * q11 - 2 * m113 * q10 * q11
                - 2 * m023 * q00 * q12 + 2 * m123 * q00 * q12 + 2 * m013 * q03 * q12 - 2 * m113 * q03 * q12 + 2 * m023 * q10 * q12
                - 2 * m123 * q10 * q12 + m003 * ( q00_2 - 2 * q00 * q10 + q10_2 - q01_11_2 - q02_12_2 + q03_13_2 )
                - 2 * m023 * q01 * q13 + 2 * m123 * q01 * q13 + 2 * m013 * q02 * q13 - 2 * m113 * q02 * q13
                + 2 * m023 * q11 * q13 - 2 * m123 * q11 * q13 - 2 * m013 * q12 * q13 + 2 * m113 * q12 * q13
                + m103 * ( -q00_10_2 + q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ) );
    c4[0] = SRNumeratorDerivativeTerm(
        -( ( b )
           * ( 2
                   * ( ( m020 - m120 ) * ( ( q00 - q10 ) * ( q02 - q12 ) + ( q01 - q11 ) * ( q03 - q13 ) )
                       + m010 * ( ( q00 - q10 ) * ( q01 - q11 ) - ( q02 - q12 ) * ( q03 - q13 ) )
                       + m110 * ( -( ( q00 - q10 ) * ( q01 - q11 ) ) + ( q02 - q12 ) * ( q03 - q13 ) ) )
               + m000 * ( q00_2 - 2 * q00 * q10 + q10_2 - q01_11_2 - q02_12_2 + q03_13_2 )
               + m100 * ( -q00_10_2 + q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ) ),
        -( ( b )
           * ( 2
                   * ( ( m021 - m121 ) * ( ( q00 - q10 ) * ( q02 - q12 ) + ( q01 - q11 ) * ( q03 - q13 ) )
                       + m011 * ( ( q00 - q10 ) * ( q01 - q11 ) - ( q02 - q12 ) * ( q03 - q13 ) )
                       + m111 * ( -( ( q00 - q10 ) * ( q01 - q11 ) ) + ( q02 - q12 ) * ( q03 - q13 ) ) )
               + m001 * ( q00_2 - 2 * q00 * q10 + q10_2 - q01_11_2 - q02_12_2 + q03_13_2 )
               + m101 * ( -q00_10_2 + q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ) ),
        -( ( b )
           * ( 2
                   * ( ( m022 - m122 ) * ( ( q00 - q10 ) * ( q02 - q12 ) + ( q01 - q11 ) * ( q03 - q13 ) )
                       + m012 * ( ( q00 - q10 ) * ( q01 - q11 ) - ( q02 - q12 ) * ( q03 - q13 ) )
                       + m112 * ( -( ( q00 - q10 ) * ( q01 - q11 ) ) + ( q02 - q12 ) * ( q03 - q13 ) ) )
               + m002 * ( q00_2 - 2 * q00 * q10 + q10_2 - q01_11_2 - q02_12_2 + q03_13_2 )
               + m102 * ( -q00_10_2 + q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ) ),
        -( ( b )
           * ( 2 * m013 * q00 * q01 - 2 * m113 * q00 * q01 + 2 * m023 * q00 * q02 - 2 * m123 * q00 * q02 + 2 * m023 * q01 * q03
               - 2 * m123 * q01 * q03 - 2 * m013 * q02 * q03 + 2 * m113 * q02 * q03 - 2 * m013 * q01 * q10
               + 2 * m113 * q01 * q10 - 2 * m023 * q02 * q10 + 2 * m123 * q02 * q10 - 2 * m013 * q00 * q11
               + 2 * m113 * q00 * q11 - 2 * m023 * q03 * q11 + 2 * m123 * q03 * q11 + 2 * m013 * q10 * q11 - 2 * m113 * q10 * q11
               - 2 * m023 * q00 * q12 + 2 * m123 * q00 * q12 + 2 * m013 * q03 * q12 - 2 * m113 * q03 * q12 + 2 * m023 * q10 * q12
               - 2 * m123 * q10 * q12 + m003 * ( q00_2 - 2 * q00 * q10 + q10_2 - q01_11_2 - q02_12_2 + q03_13_2 )
               - 2 * m023 * q01 * q13 + 2 * m123 * q01 * q13 + 2 * m013 * q02 * q13 - 2 * m113 * q02 * q13
               + 2 * m023 * q11 * q13 - 2 * m123 * q11 * q13 - 2 * m013 * q12 * q13 + 2 * m113 * q12 * q13
               + m103 * ( -q00_10_2 + q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ) ) );

    // Y
    c0[1] = SRNumeratorDerivativeTerm(
        -( m110 * ( q00_2 - q01_2 + q02_2 - q03_2 ) * ( a ) )
            + m010
                  * ( q00_4 - q013_2 - 4 * q00 * ( q01_2 + q03_2 ) * q10 + 2 * q00_2 * ( q02_2 + 2 * q01 * q11 + 2 * q03 * q13 )
                      + q02 * ( q02_3 - 4 * ( q01_2 + q03_2 ) * q12 + 4 * q02 * ( q01 * q11 + q03 * q13 ) ) )
            - 2
                  * ( m020 * q00_2 * q01 * q02 - m120 * q00_2 * q01 * q02 + m020 * q01_3 * q02 - m120 * q01_3 * q02
                      + m020 * q01 * q02_3 - m120 * q01 * q02_3 - m020 * q00_3 * q03 + m120 * q00_3 * q03
                      - m020 * q00 * q01_2 * q03 + m120 * q00 * q01_2 * q03 - m020 * q00 * q02_2 * q03 + m120 * q00 * q02_2 * q03
                      + m020 * q01 * q02 * q03_2 - m120 * q01 * q02 * q03_2 - m020 * q00 * q03_3 + m120 * q00 * q03_3
                      - m100 * ( q00 * q01 + q02 * q03 ) * ( a ) + 2 * m020 * q00 * q01 * q02 * q10 - m020 * q00_2 * q03 * q10
                      + m020 * q01_2 * q03 * q10 + m020 * q02_2 * q03 * q10 + m020 * q03_3 * q10 - m020 * q00_2 * q02 * q11
                      + m020 * q01_2 * q02 * q11 - m020 * q02_3 * q11 - 2 * m020 * q00 * q01 * q03 * q11 - m020 * q02 * q03_2 * q11
                      - m020 * q00_2 * q01 * q12 - m020 * q01_3 * q12 + m020 * q01 * q02_2 * q12 - 2 * m020 * q00 * q02 * q03 * q12
                      - m020 * q01 * q03_2 * q12 + m020 * ( q00_3 + 2 * q01 * q02 * q03 + q00 * ( q01_2 + q02_2 - q03_2 ) ) * q13
                      + m000
                            * ( q02_3 * q03 + q02 * q03_3 - q01_3 * q10 - q01 * ( q02_2 + q03_2 ) * q10
                                + q00_3 * ( q01 - q11 ) + 2 * q01 * q02 * q03 * q11 + q02_2 * q03 * q12 - q03_3 * q12
                                + q00_2 * ( q01 * q10 - q03 * q12 + q02 * ( q03 - q13 ) ) - q02_3 * q13
                                + q02 * q03_2 * q13 + q01_2 * ( q02 * q03 - q03 * q12 - q02 * q13 )
                                + q00
                                      * ( q01_3 + 2 * q02 * q03 * q10 + q01_2 * q11 - ( q02_2 + q03_2 ) * q11
                                          + q01 * ( q02_2 + q03_2 + 2 * q02 * q12 + 2 * q03 * q13 ) ) ) ),
        -( m111 * ( q00_2 - q01_2 + q02_2 - q03_2 ) * ( a ) )
            + m011
                  * ( q00_4 - q013_2 - 4 * q00 * ( q01_2 + q03_2 ) * q10 + 2 * q00_2 * ( q02_2 + 2 * q01 * q11 + 2 * q03 * q13 )
                      + q02 * ( q02_3 - 4 * ( q01_2 + q03_2 ) * q12 + 4 * q02 * ( q01 * q11 + q03 * q13 ) ) )
            - 2
                  * ( m021 * q00_2 * q01 * q02 - m121 * q00_2 * q01 * q02 + m021 * q01_3 * q02 - m121 * q01_3 * q02
                      + m021 * q01 * q02_3 - m121 * q01 * q02_3 - m021 * q00_3 * q03 + m121 * q00_3 * q03
                      - m021 * q00 * q01_2 * q03 + m121 * q00 * q01_2 * q03 - m021 * q00 * q02_2 * q03 + m121 * q00 * q02_2 * q03
                      + m021 * q01 * q02 * q03_2 - m121 * q01 * q02 * q03_2 - m021 * q00 * q03_3 + m121 * q00 * q03_3
                      - m101 * ( q00 * q01 + q02 * q03 ) * ( a ) + 2 * m021 * q00 * q01 * q02 * q10 - m021 * q00_2 * q03 * q10
                      + m021 * q01_2 * q03 * q10 + m021 * q02_2 * q03 * q10 + m021 * q03_3 * q10 - m021 * q00_2 * q02 * q11
                      + m021 * q01_2 * q02 * q11 - m021 * q02_3 * q11 - 2 * m021 * q00 * q01 * q03 * q11 - m021 * q02 * q03_2 * q11
                      - m021 * q00_2 * q01 * q12 - m021 * q01_3 * q12 + m021 * q01 * q02_2 * q12 - 2 * m021 * q00 * q02 * q03 * q12
                      - m021 * q01 * q03_2 * q12 + m021 * ( q00_3 + 2 * q01 * q02 * q03 + q00 * ( q01_2 + q02_2 - q03_2 ) ) * q13
                      + m001
                            * ( q02_3 * q03 + q02 * q03_3 - q01_3 * q10 - q01 * ( q02_2 + q03_2 ) * q10
                                + q00_3 * ( q01 - q11 ) + 2 * q01 * q02 * q03 * q11 + q02_2 * q03 * q12 - q03_3 * q12
                                + q00_2 * ( q01 * q10 - q03 * q12 + q02 * ( q03 - q13 ) ) - q02_3 * q13
                                + q02 * q03_2 * q13 + q01_2 * ( q02 * q03 - q03 * q12 - q02 * q13 )
                                + q00 * ( q01_3 + 2 * q02 * q03 * q10 + q01_2 * q11 - ( q02_2 + q03_2 ) * q11 + q01 * ( q02_2 + q03_2 + 2 * q02 * q12 + 2 * q03 * q13 ) ) ) ),
        -( m112 * ( q00_2 - q01_2 + q02_2 - q03_2 ) * ( a ) )
            + m012
                  * ( q00_4 - q013_2 - 4 * q00 * ( q01_2 + q03_2 ) * q10 + 2 * q00_2 * ( q02_2 + 2 * q01 * q11 + 2 * q03 * q13 )
                      + q02 * ( q02_3 - 4 * ( q01_2 + q03_2 ) * q12 + 4 * q02 * ( q01 * q11 + q03 * q13 ) ) )
            - 2
                  * ( m022 * q00_2 * q01 * q02 - m122 * q00_2 * q01 * q02 + m022 * q01_3 * q02 - m122 * q01_3 * q02
                      + m022 * q01 * q02_3 - m122 * q01 * q02_3 - m022 * q00_3 * q03 + m122 * q00_3 * q03
                      - m022 * q00 * q01_2 * q03 + m122 * q00 * q01_2 * q03 - m022 * q00 * q02_2 * q03 + m122 * q00 * q02_2 * q03
                      + m022 * q01 * q02 * q03_2 - m122 * q01 * q02 * q03_2 - m022 * q00 * q03_3 + m122 * q00 * q03_3
                      - m102 * ( q00 * q01 + q02 * q03 ) * ( a ) + 2 * m022 * q00 * q01 * q02 * q10 - m022 * q00_2 * q03 * q10
                      + m022 * q01_2 * q03 * q10 + m022 * q02_2 * q03 * q10 + m022 * q03_3 * q10 - m022 * q00_2 * q02 * q11
                      + m022 * q01_2 * q02 * q11 - m022 * q02_3 * q11 - 2 * m022 * q00 * q01 * q03 * q11 - m022 * q02 * q03_2 * q11
                      - m022 * q00_2 * q01 * q12 - m022 * q01_3 * q12 + m022 * q01 * q02_2 * q12 - 2 * m022 * q00 * q02 * q03 * q12
                      - m022 * q01 * q03_2 * q12 + m022 * ( q00_3 + 2 * q01 * q02 * q03 + q00 * ( q01_2 + q02_2 - q03_2 ) ) * q13
                      + m002
                            * ( q02_3 * q03 + q02 * q03_3 - q01_3 * q10 - q01 * ( q02_2 + q03_2 ) * q10
                                + q00_3 * ( q01 - q11 ) + 2 * q01 * q02 * q03 * q11 + q02_2 * q03 * q12 - q03_3 * q12
                                + q00_2 * ( q01 * q10 - q03 * q12 + q02 * ( q03 - q13 ) ) - q02_3 * q13
                                + q02 * q03_2 * q13 + q01_2 * ( q02 * q03 - q03 * q12 - q02 * q13 )
                                + q00 * ( q01_3 + 2 * q02 * q03 * q10 + q01_2 * q11 - ( q02_2 + q03_2 ) * q11 + q01 * ( q02_2 + q03_2 + 2 * q02 * q12 + 2 * q03 * q13 ) ) ) ),
        -2 * m003 * q00_3 * q01 + 2 * m103 * q00_3 * q01 - 2 * m003 * q00 * q01_3 + 2 * m103 * q00 * q01_3
            - 2 * m023 * q00_2 * q01 * q02 + 2 * m123 * q00_2 * q01 * q02 - 2 * m023 * q01_3 * q02
            + 2 * m123 * q01_3 * q02 - 2 * m003 * q00 * q01 * q02_2 + 2 * m103 * q00 * q01 * q02_2 - 2 * m023 * q01 * q02_3
            + 2 * m123 * q01 * q02_3 + 2 * m023 * q00_3 * q03 - 2 * m123 * q00_3 * q03 + 2 * m023 * q00 * q01_2 * q03
            - 2 * m123 * q00 * q01_2 * q03 - 2 * m003 * q00_2 * q02 * q03 + 2 * m103 * q00_2 * q02 * q03
            - 2 * m003 * q01_2 * q02 * q03 + 2 * m103 * q01_2 * q02 * q03 + 2 * m023 * q00 * q02_2 * q03
            - 2 * m123 * q00 * q02_2 * q03 - 2 * m003 * q02_3 * q03 + 2 * m103 * q02_3 * q03 - 2 * m003 * q00 * q01 * q03_2
            + 2 * m103 * q00 * q01 * q03_2 - 2 * m023 * q01 * q02 * q03_2 + 2 * m123 * q01 * q02 * q03_2
            + 2 * m023 * q00 * q03_3 - 2 * m123 * q00 * q03_3 - 2 * m003 * q02 * q03_3 + 2 * m103 * q02 * q03_3
            - m113 * ( q00_2 - q01_2 + q02_2 - q03_2 ) * (a)-2 * m003 * q00_2 * q01 * q10 + 2 * m003 * q01_3 * q10
            - 4 * m023 * q00 * q01 * q02 * q10 + 2 * m003 * q01 * q02_2 * q10 + 2 * m023 * q00_2 * q03 * q10
            - 2 * m023 * q01_2 * q03 * q10 - 4 * m003 * q00 * q02 * q03 * q10 - 2 * m023 * q02_2 * q03 * q10
            + 2 * m003 * q01 * q03_2 * q10 - 2 * m023 * q03_3 * q10 + 2 * m003 * q00_3 * q11 - 2 * m003 * q00 * q01_2 * q11
            + 2 * m023 * q00_2 * q02 * q11 - 2 * m023 * q01_2 * q02 * q11 + 2 * m003 * q00 * q02_2 * q11
            + 2 * m023 * q02_3 * q11 + 4 * m023 * q00 * q01 * q03 * q11 - 4 * m003 * q01 * q02 * q03 * q11
            + 2 * m003 * q00 * q03_2 * q11 + 2 * m023 * q02 * q03_2 * q11 + 2 * m023 * q00_2 * q01 * q12
            + 2 * m023 * q01_3 * q12 - 4 * m003 * q00 * q01 * q02 * q12 - 2 * m023 * q01 * q02_2 * q12
            + 2 * m003 * q00_2 * q03 * q12 + 2 * m003 * q01_2 * q03 * q12 + 4 * m023 * q00 * q02 * q03 * q12
            - 2 * m003 * q02_2 * q03 * q12 + 2 * m023 * q01 * q03_2 * q12 + 2 * m003 * q03_3 * q12 - 2 * m023 * q00_3 * q13
            - 2 * m023 * q00 * q01_2 * q13 + 2 * m003 * q00_2 * q02 * q13 + 2 * m003 * q01_2 * q02 * q13
            - 2 * m023 * q00 * q02_2 * q13 + 2 * m003 * q02_3 * q13 - 4 * m003 * q00 * q01 * q03 * q13
            - 4 * m023 * q01 * q02 * q03 * q13 + 2 * m023 * q00 * q03_2 * q13 - 2 * m003 * q02 * q03_2 * q13
            + m013
                  * ( q00_4 - q013_2 - 4 * q00 * ( q01_2 + q03_2 ) * q10 + 2 * q00_2 * ( q02_2 + 2 * q01 * q11 + 2 * q03 * q13 )
                      + q02 * ( q02_3 - 4 * ( q01_2 + q03_2 ) * q12 + 4 * q02 * ( q01 * q11 + q03 * q13 ) ) ) );
    c1[1] = SRNumeratorDerivativeTerm(
        4
            * ( 2 * m000 * q00_3 * q01 - 2 * m100 * q00_3 * q01 + 2 * m000 * q00 * q01_3 - 2 * m100 * q00 * q01_3
                + 2 * m020 * q00_2 * q01 * q02 - 2 * m120 * q00_2 * q01 * q02 + 2 * m020 * q01_3 * q02 - 2 * m120 * q01_3 * q02
                + 2 * m000 * q00 * q01 * q02_2 - 2 * m100 * q00 * q01 * q02_2 + 2 * m020 * q01 * q02_3 - 2 * m120 * q01 * q02_3
                - 2 * m020 * q00_3 * q03 + 2 * m120 * q00_3 * q03 - 2 * m020 * q00 * q01_2 * q03 + 2 * m120 * q00 * q01_2 * q03
                + 2 * m000 * q00_2 * q02 * q03 - 2 * m100 * q00_2 * q02 * q03 + 2 * m000 * q01_2 * q02 * q03
                - 2 * m100 * q01_2 * q02 * q03 - 2 * m020 * q00 * q02_2 * q03 + 2 * m120 * q00 * q02_2 * q03
                + 2 * m000 * q02_3 * q03 - 2 * m100 * q02_3 * q03 + 2 * m000 * q00 * q01 * q03_2 - 2 * m100 * q00 * q01 * q03_2
                + 2 * m020 * q01 * q02 * q03_2 - 2 * m120 * q01 * q02 * q03_2 - 2 * m020 * q00 * q03_3 + 2 * m120 * q00 * q03_3
                + 2 * m000 * q02 * q03_3 - 2 * m100 * q02 * q03_3 + m100 * q00_2 * q01 * q10 - 2 * m000 * q01_3 * q10
                + m100 * q01_3 * q10 + 2 * m020 * q00 * q01 * q02 * q10 - 2 * m000 * q01 * q02_2 * q10
                + m100 * q01 * q02_2 * q10 - m120 * q00_2 * q03 * q10 + 2 * m020 * q01_2 * q03 * q10 - m120 * q01_2 * q03 * q10
                + 2 * m000 * q00 * q02 * q03 * q10 + 2 * m020 * q02_2 * q03 * q10 - m120 * q02_2 * q03 * q10
                - 2 * m000 * q01 * q03_2 * q10 + m100 * q01 * q03_2 * q10 + 2 * m020 * q03_3 * q10 - m120 * q03_3 * q10
                - m000 * q00 * q01 * q10_2 - m020 * q01 * q02 * q10_2 + m020 * q00 * q03 * q10_2 - m000 * q02 * q03 * q10_2
                - 2 * m000 * q00_3 * q11 + m100 * q00_3 * q11 + m100 * q00 * q01_2 * q11 - 2 * m020 * q00_2 * q02 * q11
                + m120 * q00_2 * q02 * q11 + m120 * q01_2 * q02 * q11 - 2 * m000 * q00 * q02_2 * q11
                + m100 * q00 * q02_2 * q11 - 2 * m020 * q02_3 * q11 + m120 * q02_3 * q11 - 2 * m020 * q00 * q01 * q03 * q11
                + 2 * m000 * q01 * q02 * q03 * q11 - 2 * m000 * q00 * q03_2 * q11 + m100 * q00 * q03_2 * q11
                - 2 * m020 * q02 * q03_2 * q11 + m120 * q02 * q03_2 * q11 + m000 * q00_2 * q10 * q11 + m000 * q01_2 * q10 * q11
                + m000 * q02_2 * q10 * q11 + m000 * q03_2 * q10 * q11 - m000 * q00 * q01 * q11_2 - m020 * q01 * q02 * q11_2
                + m020 * q00 * q03 * q11_2 - m000 * q02 * q03 * q11_2 - 2 * m020 * q00_2 * q01 * q12 + m120 * q00_2 * q01 * q12
                - 2 * m020 * q01_3 * q12 + m120 * q01_3 * q12 + 2 * m000 * q00 * q01 * q02 * q12 + m120 * q01 * q02_2 * q12
                - 2 * m000 * q00_2 * q03 * q12 + m100 * q00_2 * q03 * q12 - 2 * m000 * q01_2 * q03 * q12
                + m100 * q01_2 * q03 * q12 - 2 * m020 * q00 * q02 * q03 * q12 + m100 * q02_2 * q03 * q12
                - 2 * m020 * q01 * q03_2 * q12 + m120 * q01 * q03_2 * q12 - 2 * m000 * q03_3 * q12 + m100 * q03_3 * q12
                + m020 * q00_2 * q11 * q12 + m020 * q01_2 * q11 * q12 + m020 * q02_2 * q11 * q12 + m020 * q03_2 * q11 * q12
                - m000 * q00 * q01 * q12_2 - m020 * q01 * q02 * q12_2 + m020 * q00 * q03 * q12_2 - m000 * q02 * q03 * q12_2
                + ( -2 * m000 * q00_2 * q02 + m100 * q00_2 * q02 - 2 * m000 * q01_2 * q02 + m100 * q01_2 * q02
                    - 2 * m000 * q02_3 + m100 * q02_3 + 2 * m000 * q00 * q01 * q03 + m100 * q02 * q03_2
                    + 2 * m020 * ( q00_3 + q00 * ( q01_2 + q02_2 ) + q01 * q02 * q03 ) - m120 * q00 * (a)-m020 * (a)*q10
                    + m000 * (a)*q12 )
                      * q13
                - ( m000 * q00 * q01 + m020 * q01 * q02 - m020 * q00 * q03 + m000 * q02 * q03 ) * q13_2
                + m110 * ( a ) * ( q00_2 + q02_2 - q00 * q10 + q01 * ( -q01 + q11 ) - q02 * q12 + q03 * ( -q03 + q13 ) )
                + m010
                      * ( -q00_4 - q02_4 + q013_2 + q00_3 * q10 + q00 * ( 3 * q01_2 + q02_2 + 3 * q03_2 ) * q10 + q02_3 * q12
                          + 3 * q02 * ( q01_2 + q03_2 ) * q12 - ( q01_2 + q03_2 ) * ( q10_2 + q01 * q11 + q12_2 + q03 * q13 )
                          + q00_2 * ( -3 * q01 * q11 + q11_2 + q02 * ( -2 * q02 + q12 ) - 3 * q03 * q13 + q13_2 )
                          + q02_2 * ( -3 * q01 * q11 + q11_2 + q13 * ( -3 * q03 + q13 ) ) ) ),
        4
            * ( 2 * m001 * q00_3 * q01 - 2 * m101 * q00_3 * q01 + 2 * m001 * q00 * q01_3 - 2 * m101 * q00 * q01_3
                + 2 * m021 * q00_2 * q01 * q02 - 2 * m121 * q00_2 * q01 * q02 + 2 * m021 * q01_3 * q02 - 2 * m121 * q01_3 * q02
                + 2 * m001 * q00 * q01 * q02_2 - 2 * m101 * q00 * q01 * q02_2 + 2 * m021 * q01 * q02_3 - 2 * m121 * q01 * q02_3
                - 2 * m021 * q00_3 * q03 + 2 * m121 * q00_3 * q03 - 2 * m021 * q00 * q01_2 * q03 + 2 * m121 * q00 * q01_2 * q03
                + 2 * m001 * q00_2 * q02 * q03 - 2 * m101 * q00_2 * q02 * q03 + 2 * m001 * q01_2 * q02 * q03
                - 2 * m101 * q01_2 * q02 * q03 - 2 * m021 * q00 * q02_2 * q03 + 2 * m121 * q00 * q02_2 * q03
                + 2 * m001 * q02_3 * q03 - 2 * m101 * q02_3 * q03 + 2 * m001 * q00 * q01 * q03_2 - 2 * m101 * q00 * q01 * q03_2
                + 2 * m021 * q01 * q02 * q03_2 - 2 * m121 * q01 * q02 * q03_2 - 2 * m021 * q00 * q03_3 + 2 * m121 * q00 * q03_3
                + 2 * m001 * q02 * q03_3 - 2 * m101 * q02 * q03_3 + m101 * q00_2 * q01 * q10 - 2 * m001 * q01_3 * q10
                + m101 * q01_3 * q10 + 2 * m021 * q00 * q01 * q02 * q10 - 2 * m001 * q01 * q02_2 * q10
                + m101 * q01 * q02_2 * q10 - m121 * q00_2 * q03 * q10 + 2 * m021 * q01_2 * q03 * q10 - m121 * q01_2 * q03 * q10
                + 2 * m001 * q00 * q02 * q03 * q10 + 2 * m021 * q02_2 * q03 * q10 - m121 * q02_2 * q03 * q10
                - 2 * m001 * q01 * q03_2 * q10 + m101 * q01 * q03_2 * q10 + 2 * m021 * q03_3 * q10 - m121 * q03_3 * q10
                - m001 * q00 * q01 * q10_2 - m021 * q01 * q02 * q10_2 + m021 * q00 * q03 * q10_2 - m001 * q02 * q03 * q10_2
                - 2 * m001 * q00_3 * q11 + m101 * q00_3 * q11 + m101 * q00 * q01_2 * q11 - 2 * m021 * q00_2 * q02 * q11
                + m121 * q00_2 * q02 * q11 + m121 * q01_2 * q02 * q11 - 2 * m001 * q00 * q02_2 * q11
                + m101 * q00 * q02_2 * q11 - 2 * m021 * q02_3 * q11 + m121 * q02_3 * q11 - 2 * m021 * q00 * q01 * q03 * q11
                + 2 * m001 * q01 * q02 * q03 * q11 - 2 * m001 * q00 * q03_2 * q11 + m101 * q00 * q03_2 * q11
                - 2 * m021 * q02 * q03_2 * q11 + m121 * q02 * q03_2 * q11 + m001 * q00_2 * q10 * q11 + m001 * q01_2 * q10 * q11
                + m001 * q02_2 * q10 * q11 + m001 * q03_2 * q10 * q11 - m001 * q00 * q01 * q11_2 - m021 * q01 * q02 * q11_2
                + m021 * q00 * q03 * q11_2 - m001 * q02 * q03 * q11_2 - 2 * m021 * q00_2 * q01 * q12 + m121 * q00_2 * q01 * q12
                - 2 * m021 * q01_3 * q12 + m121 * q01_3 * q12 + 2 * m001 * q00 * q01 * q02 * q12 + m121 * q01 * q02_2 * q12
                - 2 * m001 * q00_2 * q03 * q12 + m101 * q00_2 * q03 * q12 - 2 * m001 * q01_2 * q03 * q12
                + m101 * q01_2 * q03 * q12 - 2 * m021 * q00 * q02 * q03 * q12 + m101 * q02_2 * q03 * q12
                - 2 * m021 * q01 * q03_2 * q12 + m121 * q01 * q03_2 * q12 - 2 * m001 * q03_3 * q12 + m101 * q03_3 * q12
                + m021 * q00_2 * q11 * q12 + m021 * q01_2 * q11 * q12 + m021 * q02_2 * q11 * q12 + m021 * q03_2 * q11 * q12
                - m001 * q00 * q01 * q12_2 - m021 * q01 * q02 * q12_2 + m021 * q00 * q03 * q12_2 - m001 * q02 * q03 * q12_2
                + ( -2 * m001 * q00_2 * q02 + m101 * q00_2 * q02 - 2 * m001 * q01_2 * q02 + m101 * q01_2 * q02
                    - 2 * m001 * q02_3 + m101 * q02_3 + 2 * m001 * q00 * q01 * q03 + m101 * q02 * q03_2
                    + 2 * m021 * ( q00_3 + q00 * ( q01_2 + q02_2 ) + q01 * q02 * q03 ) - m121 * q00 * (a)-m021 * (a)*q10
                    + m001 * (a)*q12 )
                      * q13
                - ( m001 * q00 * q01 + m021 * q01 * q02 - m021 * q00 * q03 + m001 * q02 * q03 ) * q13_2
                + m111 * ( a ) * ( q00_2 + q02_2 - q00 * q10 + q01 * ( -q01 + q11 ) - q02 * q12 + q03 * ( -q03 + q13 ) )
                + m011
                      * ( -q00_4 - q02_4 + q013_2 + q00_3 * q10 + q00 * ( 3 * q01_2 + q02_2 + 3 * q03_2 ) * q10 + q02_3 * q12
                          + 3 * q02 * ( q01_2 + q03_2 ) * q12 - ( q01_2 + q03_2 ) * ( q10_2 + q01 * q11 + q12_2 + q03 * q13 )
                          + q00_2 * ( -3 * q01 * q11 + q11_2 + q02 * ( -2 * q02 + q12 ) - 3 * q03 * q13 + q13_2 )
                          + q02_2 * ( -3 * q01 * q11 + q11_2 + q13 * ( -3 * q03 + q13 ) ) ) ),
        4
            * ( 2 * m002 * q00_3 * q01 - 2 * m102 * q00_3 * q01 + 2 * m002 * q00 * q01_3 - 2 * m102 * q00 * q01_3
                + 2 * m022 * q00_2 * q01 * q02 - 2 * m122 * q00_2 * q01 * q02 + 2 * m022 * q01_3 * q02 - 2 * m122 * q01_3 * q02
                + 2 * m002 * q00 * q01 * q02_2 - 2 * m102 * q00 * q01 * q02_2 + 2 * m022 * q01 * q02_3 - 2 * m122 * q01 * q02_3
                - 2 * m022 * q00_3 * q03 + 2 * m122 * q00_3 * q03 - 2 * m022 * q00 * q01_2 * q03 + 2 * m122 * q00 * q01_2 * q03
                + 2 * m002 * q00_2 * q02 * q03 - 2 * m102 * q00_2 * q02 * q03 + 2 * m002 * q01_2 * q02 * q03
                - 2 * m102 * q01_2 * q02 * q03 - 2 * m022 * q00 * q02_2 * q03 + 2 * m122 * q00 * q02_2 * q03
                + 2 * m002 * q02_3 * q03 - 2 * m102 * q02_3 * q03 + 2 * m002 * q00 * q01 * q03_2 - 2 * m102 * q00 * q01 * q03_2
                + 2 * m022 * q01 * q02 * q03_2 - 2 * m122 * q01 * q02 * q03_2 - 2 * m022 * q00 * q03_3 + 2 * m122 * q00 * q03_3
                + 2 * m002 * q02 * q03_3 - 2 * m102 * q02 * q03_3 + m102 * q00_2 * q01 * q10 - 2 * m002 * q01_3 * q10
                + m102 * q01_3 * q10 + 2 * m022 * q00 * q01 * q02 * q10 - 2 * m002 * q01 * q02_2 * q10
                + m102 * q01 * q02_2 * q10 - m122 * q00_2 * q03 * q10 + 2 * m022 * q01_2 * q03 * q10 - m122 * q01_2 * q03 * q10
                + 2 * m002 * q00 * q02 * q03 * q10 + 2 * m022 * q02_2 * q03 * q10 - m122 * q02_2 * q03 * q10
                - 2 * m002 * q01 * q03_2 * q10 + m102 * q01 * q03_2 * q10 + 2 * m022 * q03_3 * q10 - m122 * q03_3 * q10
                - m002 * q00 * q01 * q10_2 - m022 * q01 * q02 * q10_2 + m022 * q00 * q03 * q10_2 - m002 * q02 * q03 * q10_2
                - 2 * m002 * q00_3 * q11 + m102 * q00_3 * q11 + m102 * q00 * q01_2 * q11 - 2 * m022 * q00_2 * q02 * q11
                + m122 * q00_2 * q02 * q11 + m122 * q01_2 * q02 * q11 - 2 * m002 * q00 * q02_2 * q11
                + m102 * q00 * q02_2 * q11 - 2 * m022 * q02_3 * q11 + m122 * q02_3 * q11 - 2 * m022 * q00 * q01 * q03 * q11
                + 2 * m002 * q01 * q02 * q03 * q11 - 2 * m002 * q00 * q03_2 * q11 + m102 * q00 * q03_2 * q11
                - 2 * m022 * q02 * q03_2 * q11 + m122 * q02 * q03_2 * q11 + m002 * q00_2 * q10 * q11 + m002 * q01_2 * q10 * q11
                + m002 * q02_2 * q10 * q11 + m002 * q03_2 * q10 * q11 - m002 * q00 * q01 * q11_2 - m022 * q01 * q02 * q11_2
                + m022 * q00 * q03 * q11_2 - m002 * q02 * q03 * q11_2 - 2 * m022 * q00_2 * q01 * q12 + m122 * q00_2 * q01 * q12
                - 2 * m022 * q01_3 * q12 + m122 * q01_3 * q12 + 2 * m002 * q00 * q01 * q02 * q12 + m122 * q01 * q02_2 * q12
                - 2 * m002 * q00_2 * q03 * q12 + m102 * q00_2 * q03 * q12 - 2 * m002 * q01_2 * q03 * q12
                + m102 * q01_2 * q03 * q12 - 2 * m022 * q00 * q02 * q03 * q12 + m102 * q02_2 * q03 * q12
                - 2 * m022 * q01 * q03_2 * q12 + m122 * q01 * q03_2 * q12 - 2 * m002 * q03_3 * q12 + m102 * q03_3 * q12
                + m022 * q00_2 * q11 * q12 + m022 * q01_2 * q11 * q12 + m022 * q02_2 * q11 * q12 + m022 * q03_2 * q11 * q12
                - m002 * q00 * q01 * q12_2 - m022 * q01 * q02 * q12_2 + m022 * q00 * q03 * q12_2 - m002 * q02 * q03 * q12_2
                + ( -2 * m002 * q00_2 * q02 + m102 * q00_2 * q02 - 2 * m002 * q01_2 * q02 + m102 * q01_2 * q02
                    - 2 * m002 * q02_3 + m102 * q02_3 + 2 * m002 * q00 * q01 * q03 + m102 * q02 * q03_2
                    + 2 * m022 * ( q00_3 + q00 * ( q01_2 + q02_2 ) + q01 * q02 * q03 ) - m122 * q00 * (a)-m022 * (a)*q10
                    + m002 * (a)*q12 )
                      * q13
                - ( m002 * q00 * q01 + m022 * q01 * q02 - m022 * q00 * q03 + m002 * q02 * q03 ) * q13_2
                + m112 * ( a ) * ( q00_2 + q02_2 - q00 * q10 + q01 * ( -q01 + q11 ) - q02 * q12 + q03 * ( -q03 + q13 ) )
                + m012
                      * ( -q00_4 - q02_4 + q013_2 + q00_3 * q10 + q00 * ( 3 * q01_2 + q02_2 + 3 * q03_2 ) * q10 + q02_3 * q12
                          + 3 * q02 * ( q01_2 + q03_2 ) * q12 - ( q01_2 + q03_2 ) * ( q10_2 + q01 * q11 + q12_2 + q03 * q13 )
                          + q00_2 * ( -3 * q01 * q11 + q11_2 + q02 * ( -2 * q02 + q12 ) - 3 * q03 * q13 + q13_2 )
                          + q02_2 * ( -3 * q01 * q11 + q11_2 + q13 * ( -3 * q03 + q13 ) ) ) ),
        4
            * ( 2 * m003 * q00_3 * q01 - 2 * m103 * q00_3 * q01 + 2 * m003 * q00 * q01_3 - 2 * m103 * q00 * q01_3
                + 2 * m023 * q00_2 * q01 * q02 - 2 * m123 * q00_2 * q01 * q02 + 2 * m023 * q01_3 * q02 - 2 * m123 * q01_3 * q02
                + 2 * m003 * q00 * q01 * q02_2 - 2 * m103 * q00 * q01 * q02_2 + 2 * m023 * q01 * q02_3 - 2 * m123 * q01 * q02_3
                - 2 * m023 * q00_3 * q03 + 2 * m123 * q00_3 * q03 - 2 * m023 * q00 * q01_2 * q03 + 2 * m123 * q00 * q01_2 * q03
                + 2 * m003 * q00_2 * q02 * q03 - 2 * m103 * q00_2 * q02 * q03 + 2 * m003 * q01_2 * q02 * q03
                - 2 * m103 * q01_2 * q02 * q03 - 2 * m023 * q00 * q02_2 * q03 + 2 * m123 * q00 * q02_2 * q03
                + 2 * m003 * q02_3 * q03 - 2 * m103 * q02_3 * q03 + 2 * m003 * q00 * q01 * q03_2 - 2 * m103 * q00 * q01 * q03_2
                + 2 * m023 * q01 * q02 * q03_2 - 2 * m123 * q01 * q02 * q03_2 - 2 * m023 * q00 * q03_3 + 2 * m123 * q00 * q03_3
                + 2 * m003 * q02 * q03_3 - 2 * m103 * q02 * q03_3 + m103 * q00_2 * q01 * q10 - 2 * m003 * q01_3 * q10
                + m103 * q01_3 * q10 + 2 * m023 * q00 * q01 * q02 * q10 - 2 * m003 * q01 * q02_2 * q10
                + m103 * q01 * q02_2 * q10 - m123 * q00_2 * q03 * q10 + 2 * m023 * q01_2 * q03 * q10 - m123 * q01_2 * q03 * q10
                + 2 * m003 * q00 * q02 * q03 * q10 + 2 * m023 * q02_2 * q03 * q10 - m123 * q02_2 * q03 * q10
                - 2 * m003 * q01 * q03_2 * q10 + m103 * q01 * q03_2 * q10 + 2 * m023 * q03_3 * q10 - m123 * q03_3 * q10
                - m003 * q00 * q01 * q10_2 - m023 * q01 * q02 * q10_2 + m023 * q00 * q03 * q10_2 - m003 * q02 * q03 * q10_2
                - 2 * m003 * q00_3 * q11 + m103 * q00_3 * q11 + m103 * q00 * q01_2 * q11 - 2 * m023 * q00_2 * q02 * q11
                + m123 * q00_2 * q02 * q11 + m123 * q01_2 * q02 * q11 - 2 * m003 * q00 * q02_2 * q11
                + m103 * q00 * q02_2 * q11 - 2 * m023 * q02_3 * q11 + m123 * q02_3 * q11 - 2 * m023 * q00 * q01 * q03 * q11
                + 2 * m003 * q01 * q02 * q03 * q11 - 2 * m003 * q00 * q03_2 * q11 + m103 * q00 * q03_2 * q11
                - 2 * m023 * q02 * q03_2 * q11 + m123 * q02 * q03_2 * q11 + m003 * q00_2 * q10 * q11 + m003 * q01_2 * q10 * q11
                + m003 * q02_2 * q10 * q11 + m003 * q03_2 * q10 * q11 - m003 * q00 * q01 * q11_2 - m023 * q01 * q02 * q11_2
                + m023 * q00 * q03 * q11_2 - m003 * q02 * q03 * q11_2 - 2 * m023 * q00_2 * q01 * q12 + m123 * q00_2 * q01 * q12
                - 2 * m023 * q01_3 * q12 + m123 * q01_3 * q12 + 2 * m003 * q00 * q01 * q02 * q12 + m123 * q01 * q02_2 * q12
                - 2 * m003 * q00_2 * q03 * q12 + m103 * q00_2 * q03 * q12 - 2 * m003 * q01_2 * q03 * q12 + m103 * q01_2 * q03 * q12
                - 2 * m023 * q00 * q02 * q03 * q12 + m103 * q02_2 * q03 * q12 - 2 * m023 * q01 * q03_2 * q12
                + m123 * q01 * q03_2 * q12 - 2 * m003 * q03_3 * q12 + m103 * q03_3 * q12 + m023 * q00_2 * q11 * q12
                + m023 * q01_2 * q11 * q12 + m023 * q02_2 * q11 * q12 + m023 * q03_2 * q11 * q12 - m003 * q00 * q01 * q12_2
                - m023 * q01 * q02 * q12_2 + m023 * q00 * q03 * q12_2 - m003 * q02 * q03 * q12_2 + 2 * m023 * q00_3 * q13
                - m123 * q00_3 * q13 + 2 * m023 * q00 * q01_2 * q13 - m123 * q00 * q01_2 * q13 - 2 * m003 * q00_2 * q02 * q13
                + m103 * q00_2 * q02 * q13 - 2 * m003 * q01_2 * q02 * q13 + m103 * q01_2 * q02 * q13 + 2 * m023 * q00 * q02_2 * q13
                - m123 * q00 * q02_2 * q13 - 2 * m003 * q02_3 * q13 + m103 * q02_3 * q13 + 2 * m003 * q00 * q01 * q03 * q13
                + 2 * m023 * q01 * q02 * q03 * q13 - m123 * q00 * q03_2 * q13 + m103 * q02 * q03_2 * q13
                - m023 * q00_2 * q10 * q13 - m023 * q01_2 * q10 * q13 - m023 * q02_2 * q10 * q13 - m023 * q03_2 * q10 * q13
                + m003 * q00_2 * q12 * q13 + m003 * q01_2 * q12 * q13 + m003 * q02_2 * q12 * q13 + m003 * q03_2 * q12 * q13
                - m003 * q00 * q01 * q13_2 - m023 * q01 * q02 * q13_2 + m023 * q00 * q03 * q13_2 - m003 * q02 * q03 * q13_2
                + m113 * ( a ) * ( q00_2 + q02_2 - q00 * q10 + q01 * ( -q01 + q11 ) - q02 * q12 + q03 * ( -q03 + q13 ) )
                + m013
                      * ( -q00_4 - q02_4 + q013_2 + q00_3 * q10 + q00 * ( 3 * q01_2 + q02_2 + 3 * q03_2 ) * q10 + q02_3 * q12
                          + 3 * q02 * ( q01_2 + q03_2 ) * q12 - ( q01_2 + q03_2 ) * ( q10_2 + q01 * q11 + q12_2 + q03 * q13 )
                          + q00_2 * ( -3 * q01 * q11 + q11_2 + q02 * ( -2 * q02 + q12 ) - 3 * q03 * q13 + q13_2 )
                          + q02_2 * ( -3 * q01 * q11 + q11_2 + q13 * ( -3 * q03 + q13 ) ) ) ) );
    c2[1] = SRNumeratorDerivativeTerm(
        -2
            * ( 6 * m000 * q00_3 * q01 - 6 * m100 * q00_3 * q01 + 6 * m000 * q00 * q01_3 - 6 * m100 * q00 * q01_3
                + 6 * m020 * q00_2 * q01 * q02 - 6 * m120 * q00_2 * q01 * q02 + 6 * m020 * q01_3 * q02 - 6 * m120 * q01_3 * q02
                + 6 * m000 * q00 * q01 * q02_2 - 6 * m100 * q00 * q01 * q02_2 + 6 * m020 * q01 * q02_3 - 6 * m120 * q01 * q02_3
                - 6 * m020 * q00_3 * q03 + 6 * m120 * q00_3 * q03 - 6 * m020 * q00 * q01_2 * q03 + 6 * m120 * q00 * q01_2 * q03
                + 6 * m000 * q00_2 * q02 * q03 - 6 * m100 * q00_2 * q02 * q03 + 6 * m000 * q01_2 * q02 * q03
                - 6 * m100 * q01_2 * q02 * q03 - 6 * m020 * q00 * q02_2 * q03 + 6 * m120 * q00 * q02_2 * q03
                + 6 * m000 * q02_3 * q03 - 6 * m100 * q02_3 * q03 + 6 * m000 * q00 * q01 * q03_2 - 6 * m100 * q00 * q01 * q03_2
                + 6 * m020 * q01 * q02 * q03_2 - 6 * m120 * q01 * q02 * q03_2 - 6 * m020 * q00 * q03_3 + 6 * m120 * q00 * q03_3
                + 6 * m000 * q02 * q03_3 - 6 * m100 * q02 * q03_3 - 6 * m000 * q00_2 * q01 * q10 + 7 * m100 * q00_2 * q01 * q10
                - 6 * m000 * q01_3 * q10 + 5 * m100 * q01_3 * q10 + 2 * m120 * q00 * q01 * q02 * q10 - 6 * m000 * q01 * q02_2 * q10
                + 5 * m100 * q01 * q02_2 * q10 + 6 * m020 * q00_2 * q03 * q10 - 7 * m120 * q00_2 * q03 * q10
                + 6 * m020 * q01_2 * q03 * q10 - 5 * m120 * q01_2 * q03 * q10 + 2 * m100 * q00 * q02 * q03 * q10
                + 6 * m020 * q02_2 * q03 * q10 - 5 * m120 * q02_2 * q03 * q10 - 6 * m000 * q01 * q03_2 * q10
                + 5 * m100 * q01 * q03_2 * q10 + 6 * m020 * q03_3 * q10 - 5 * m120 * q03_3 * q10 - m000 * q00 * q01 * q10_2
                - m100 * q00 * q01 * q10_2 - 3 * m020 * q01 * q02 * q10_2 + m120 * q01 * q02 * q10_2 + m020 * q00 * q03 * q10_2
                + m120 * q00 * q03 * q10_2 - 3 * m000 * q02 * q03 * q10_2 + m100 * q02 * q03 * q10_2 + m000 * q01 * q10_3
                - m020 * q03 * q10_3 - 6 * m000 * q00_3 * q11 + 5 * m100 * q00_3 * q11 - 6 * m000 * q00 * q01_2 * q11
                + 7 * m100 * q00 * q01_2 * q11 - 6 * m020 * q00_2 * q02 * q11 + 5 * m120 * q00_2 * q02 * q11
                - 6 * m020 * q01_2 * q02 * q11 + 7 * m120 * q01_2 * q02 * q11 - 6 * m000 * q00 * q02_2 * q11
                + 5 * m100 * q00 * q02_2 * q11 - 6 * m020 * q02_3 * q11 + 5 * m120 * q02_3 * q11
                - 2 * m120 * q00 * q01 * q03 * q11 + 2 * m100 * q01 * q02 * q03 * q11 - 6 * m000 * q00 * q03_2 * q11
                + 5 * m100 * q00 * q03_2 * q11 - 6 * m020 * q02 * q03_2 * q11 + 5 * m120 * q02 * q03_2 * q11
                + 7 * m000 * q00_2 * q10 * q11 - 5 * m100 * q00_2 * q10 * q11 + 7 * m000 * q01_2 * q10 * q11
                - 5 * m100 * q01_2 * q10 * q11 + 2 * m020 * q00 * q02 * q10 * q11 - 2 * m120 * q00 * q02 * q10 * q11
                + 5 * m000 * q02_2 * q10 * q11 - 3 * m100 * q02_2 * q10 * q11 - 2 * m020 * q01 * q03 * q10 * q11
                + 2 * m120 * q01 * q03 * q10 * q11 + 5 * m000 * q03_2 * q10 * q11 - 3 * m100 * q03_2 * q10 * q11
                - m000 * q00 * q10_2 * q11 + m020 * q02 * q10_2 * q11 - m000 * q00 * q01 * q11_2 - m100 * q00 * q01 * q11_2
                - m020 * q01 * q02 * q11_2 - m120 * q01 * q02 * q11_2 + 3 * m020 * q00 * q03 * q11_2 - m120 * q00 * q03 * q11_2
                - 3 * m000 * q02 * q03 * q11_2 + m100 * q02 * q03 * q11_2 - m000 * q01 * q10 * q11_2 - m020 * q03 * q10 * q11_2
                + m000 * q00 * q11_3 + m020 * q02 * q11_3 - 6 * m020 * q00_2 * q01 * q12 + 5 * m120 * q00_2 * q01 * q12
                - 6 * m020 * q01_3 * q12 + 5 * m120 * q01_3 * q12 + 2 * m100 * q00 * q01 * q02 * q12
                - 6 * m020 * q01 * q02_2 * q12 + 7 * m120 * q01 * q02_2 * q12 - 6 * m000 * q00_2 * q03 * q12
                + 5 * m100 * q00_2 * q03 * q12 - 6 * m000 * q01_2 * q03 * q12 + 5 * m100 * q01_2 * q03 * q12
                - 2 * m120 * q00 * q02 * q03 * q12 - 6 * m000 * q02_2 * q03 * q12 + 7 * m100 * q02_2 * q03 * q12
                - 6 * m020 * q01 * q03_2 * q12 + 5 * m120 * q01 * q03_2 * q12 - 6 * m000 * q03_3 * q12 + 5 * m100 * q03_3 * q12
                + 2 * m020 * q00 * q01 * q10 * q12 - 2 * m120 * q00 * q01 * q10 * q12 + 2 * m000 * q01 * q02 * q10 * q12
                - 2 * m100 * q01 * q02 * q10 * q12 + 2 * m000 * q00 * q03 * q10 * q12 - 2 * m100 * q00 * q03 * q10 * q12
                - 2 * m020 * q02 * q03 * q10 * q12 + 2 * m120 * q02 * q03 * q10 * q12 + m020 * q01 * q10_2 * q12
                + m000 * q03 * q10_2 * q12 + 5 * m020 * q00_2 * q11 * q12 - 3 * m120 * q00_2 * q11 * q12
                + 7 * m020 * q01_2 * q11 * q12 - 5 * m120 * q01_2 * q11 * q12 + 2 * m000 * q00 * q02 * q11 * q12
                - 2 * m100 * q00 * q02 * q11 * q12 + 7 * m020 * q02_2 * q11 * q12 - 5 * m120 * q02_2 * q11 * q12
                + 2 * m000 * q01 * q03 * q11 * q12 - 2 * m100 * q01 * q03 * q11 * q12 + 5 * m020 * q03_2 * q11 * q12
                - 3 * m120 * q03_2 * q11 * q12 - 2 * m020 * q00 * q10 * q11 * q12 - 2 * m000 * q02 * q10 * q11 * q12
                - m020 * q01 * q11_2 * q12 + m000 * q03 * q11_2 * q12 - 3 * m000 * q00 * q01 * q12_2 + m100 * q00 * q01 * q12_2
                - m020 * q01 * q02 * q12_2 - m120 * q01 * q02 * q12_2 + 3 * m020 * q00 * q03 * q12_2 - m120 * q00 * q03 * q12_2
                - m000 * q02 * q03 * q12_2 - m100 * q02 * q03 * q12_2 + m000 * q01 * q10 * q12_2 - m020 * q03 * q10 * q12_2
                + m000 * q00 * q11 * q12_2 - m020 * q02 * q11 * q12_2 + m020 * q01 * q12_3 + m000 * q03 * q12_3
                + ( -6 * m000 * q00_2 * q02 + 5 * m100 * q00_2 * q02 - 6 * m000 * q01_2 * q02 + 5 * m100 * q01_2 * q02
                    - 6 * m000 * q02_3 + 5 * m100 * q02_3 + 2 * m100 * q00 * q01 * q03 - 6 * m000 * q02 * q03_2
                    + 7 * m100 * q02 * q03_2 + 2 * m000 * q00 * q02 * q10 - 2 * m100 * q00 * q02 * q10 + 2 * m000 * q01 * q03 * q10
                    - 2 * m100 * q01 * q03 * q10 + m000 * q02 * q10_2 + 2 * m000 * q01 * q02 * q11 - 2 * m100 * q01 * q02 * q11
                    + 2 * m000 * q00 * q03 * q11 - 2 * m100 * q00 * q03 * q11 - 2 * m000 * q03 * q10 * q11 + m000 * q02 * q11_2
                    + ( -( m100 * ( 3 * q00_2 + 3 * q01_2 + 5 * ( q02_2 + q03_2 ) ) )
                        + m000 * ( 5 * q00_2 + 7 * ( q02_2 + q03_2 ) - 2 * q00 * q10 + q01 * ( 5 * q01 - 2 * q11 ) ) )
                          * q12
                    - m000 * q02 * q12_2
                    + m120
                          * ( -5 * q00_3 + 5 * q00_2 * q10 + 3 * q01_2 * q10 + 3 * q02_2 * q10 + 5 * q03_2 * q10
                              - 2 * q02 * q03 * q11 + 2 * q01 * q03 * ( q02 - q12 )
                              + q00 * ( -5 * q01_2 - 5 * q02_2 - 7 * q03_2 + 2 * q01 * q11 + 2 * q02 * q12 ) )
                    + m020
                          * ( 6 * q00_3 - 7 * q00_2 * q10 - 5 * q01_2 * q10 - 5 * q02_2 * q10 - 7 * q03_2 * q10
                              + 2 * q02 * q03 * q11 + 2 * q01 * q10 * q11 + 2 * ( q01 * q03 + q02 * q10 - q03 * q11 ) * q12
                              + q00 * ( 6 * q01_2 + 6 * q02_2 + 6 * q03_2 + q10_2 - 2 * q01 * q11 - q11_2 - 2 * q02 * q12 - q12_2 ) ) )
                      * q13
                + ( m100 * q00 * q01 - 3 * m020 * q01 * q02 + m120 * q01 * q02 + m020 * q00 * q03 + m120 * q00 * q03
                    - m100 * q02 * q03 + m020 * q03 * q10 + m020 * q02 * q11 + m020 * q01 * q12
                    + m000 * ( q01 * q10 + q00 * ( -3 * q01 + q11 ) - q03 * ( q02 + q12 ) ) )
                      * q13_2
                + ( -( m020 * q00 ) + m000 * q02 ) * q13_3
                + m010
                      * ( -3 * q00_4 + 3 * q01_4 - 3 * q02_4 + 3 * q03_4 + 6 * q00_3 * q10 - 4 * q03_2 * q10_2
                          - 6 * q01_3 * q11 + q03_2 * q11_2 + 6 * q02 * q03_2 * q12 - 2 * q02 * q11_2 * q12
                          - 3 * q02_2 * q12_2 - 4 * q03_2 * q12_2 + q02_2 * ( -q10_2 + 4 * q11_2 + 6 * q02 * q12 )
                          + 2 * q03 * ( -3 * q02_2 - 3 * q03_2 + q10_2 + q12_2 ) * q13 + ( 4 * q02_2 + 3 * q03_2 - 2 * q02 * q12 ) * q13_2
                          + 2 * q01 * q11 * ( -3 * q02_2 - 3 * q03_2 + q10_2 + q12_2 + 2 * q03 * q13 )
                          - q00_2 * ( 6 * q02_2 + 3 * q10_2 + 6 * q01 * q11 - 4 * q11_2 - 6 * q02 * q12 + q12_2 + 6 * q03 * q13 - 4 * q13_2 )
                          + 2 * q00 * q10 * ( 3 * q01_2 + 3 * q02_2 + 3 * q03_2 - q11_2 - 2 * q02 * q12 - q13_2 )
                          + q01_2 * ( 6 * q03_2 - 4 * q10_2 + 3 * q11_2 + 6 * q02 * q12 - 4 * q12_2 - 6 * q03 * q13 + q13_2 ) )
                + m110
                      * ( 3 * q00_4 + 3 * q02_4 - 3 * q013_2 - 6 * q00_3 * q10 + 2 * q03_2 * q10_2 + 6 * q01 * q03_2 * q11
                          - 3 * q01_2 * q11_2 - q03_2 * q11_2 + 2 * q01_2 * ( q10_2 + 3 * q01 * q11 ) - 6 * q02_3 * q12
                          - 4 * q02 * ( q01_2 + q03_2 ) * q12 + 2 * q01_2 * q12_2 + 2 * q03_2 * q12_2
                          - 2 * q00 * q10 * ( 2 * q01_2 + 3 * q02_2 + 2 * q03_2 - 2 * q02 * q12 )
                          + 2 * q03 * ( 3 * ( q01_2 + q03_2 ) - 2 * q01 * q11 ) * q13 - ( q01_2 + 3 * q03_2 ) * q13_2
                          + q02_2 * ( q10_2 + 4 * q01 * q11 - 2 * q11_2 + 3 * q12_2 + 4 * q03 * q13 - 2 * q13_2 )
                          + q00_2 * ( 6 * q02_2 + 3 * q10_2 + 4 * q01 * q11 - 6 * q02 * q12 + q12_2 - 2 * ( q11_2 + q13 * ( -2 * q03 + q13 ) ) ) ) ),
        -2
            * ( 6 * m001 * q00_3 * q01 - 6 * m101 * q00_3 * q01 + 6 * m001 * q00 * q01_3 - 6 * m101 * q00 * q01_3
                + 6 * m021 * q00_2 * q01 * q02 - 6 * m121 * q00_2 * q01 * q02 + 6 * m021 * q01_3 * q02 - 6 * m121 * q01_3 * q02
                + 6 * m001 * q00 * q01 * q02_2 - 6 * m101 * q00 * q01 * q02_2 + 6 * m021 * q01 * q02_3 - 6 * m121 * q01 * q02_3
                - 6 * m021 * q00_3 * q03 + 6 * m121 * q00_3 * q03 - 6 * m021 * q00 * q01_2 * q03 + 6 * m121 * q00 * q01_2 * q03
                + 6 * m001 * q00_2 * q02 * q03 - 6 * m101 * q00_2 * q02 * q03 + 6 * m001 * q01_2 * q02 * q03
                - 6 * m101 * q01_2 * q02 * q03 - 6 * m021 * q00 * q02_2 * q03 + 6 * m121 * q00 * q02_2 * q03
                + 6 * m001 * q02_3 * q03 - 6 * m101 * q02_3 * q03 + 6 * m001 * q00 * q01 * q03_2 - 6 * m101 * q00 * q01 * q03_2
                + 6 * m021 * q01 * q02 * q03_2 - 6 * m121 * q01 * q02 * q03_2 - 6 * m021 * q00 * q03_3 + 6 * m121 * q00 * q03_3
                + 6 * m001 * q02 * q03_3 - 6 * m101 * q02 * q03_3 - 6 * m001 * q00_2 * q01 * q10 + 7 * m101 * q00_2 * q01 * q10
                - 6 * m001 * q01_3 * q10 + 5 * m101 * q01_3 * q10 + 2 * m121 * q00 * q01 * q02 * q10 - 6 * m001 * q01 * q02_2 * q10
                + 5 * m101 * q01 * q02_2 * q10 + 6 * m021 * q00_2 * q03 * q10 - 7 * m121 * q00_2 * q03 * q10
                + 6 * m021 * q01_2 * q03 * q10 - 5 * m121 * q01_2 * q03 * q10 + 2 * m101 * q00 * q02 * q03 * q10
                + 6 * m021 * q02_2 * q03 * q10 - 5 * m121 * q02_2 * q03 * q10 - 6 * m001 * q01 * q03_2 * q10
                + 5 * m101 * q01 * q03_2 * q10 + 6 * m021 * q03_3 * q10 - 5 * m121 * q03_3 * q10 - m001 * q00 * q01 * q10_2
                - m101 * q00 * q01 * q10_2 - 3 * m021 * q01 * q02 * q10_2 + m121 * q01 * q02 * q10_2 + m021 * q00 * q03 * q10_2
                + m121 * q00 * q03 * q10_2 - 3 * m001 * q02 * q03 * q10_2 + m101 * q02 * q03 * q10_2 + m001 * q01 * q10_3
                - m021 * q03 * q10_3 - 6 * m001 * q00_3 * q11 + 5 * m101 * q00_3 * q11 - 6 * m001 * q00 * q01_2 * q11
                + 7 * m101 * q00 * q01_2 * q11 - 6 * m021 * q00_2 * q02 * q11 + 5 * m121 * q00_2 * q02 * q11
                - 6 * m021 * q01_2 * q02 * q11 + 7 * m121 * q01_2 * q02 * q11 - 6 * m001 * q00 * q02_2 * q11
                + 5 * m101 * q00 * q02_2 * q11 - 6 * m021 * q02_3 * q11 + 5 * m121 * q02_3 * q11
                - 2 * m121 * q00 * q01 * q03 * q11 + 2 * m101 * q01 * q02 * q03 * q11 - 6 * m001 * q00 * q03_2 * q11
                + 5 * m101 * q00 * q03_2 * q11 - 6 * m021 * q02 * q03_2 * q11 + 5 * m121 * q02 * q03_2 * q11
                + 7 * m001 * q00_2 * q10 * q11 - 5 * m101 * q00_2 * q10 * q11 + 7 * m001 * q01_2 * q10 * q11
                - 5 * m101 * q01_2 * q10 * q11 + 2 * m021 * q00 * q02 * q10 * q11 - 2 * m121 * q00 * q02 * q10 * q11
                + 5 * m001 * q02_2 * q10 * q11 - 3 * m101 * q02_2 * q10 * q11 - 2 * m021 * q01 * q03 * q10 * q11
                + 2 * m121 * q01 * q03 * q10 * q11 + 5 * m001 * q03_2 * q10 * q11 - 3 * m101 * q03_2 * q10 * q11
                - m001 * q00 * q10_2 * q11 + m021 * q02 * q10_2 * q11 - m001 * q00 * q01 * q11_2 - m101 * q00 * q01 * q11_2
                - m021 * q01 * q02 * q11_2 - m121 * q01 * q02 * q11_2 + 3 * m021 * q00 * q03 * q11_2 - m121 * q00 * q03 * q11_2
                - 3 * m001 * q02 * q03 * q11_2 + m101 * q02 * q03 * q11_2 - m001 * q01 * q10 * q11_2 - m021 * q03 * q10 * q11_2
                + m001 * q00 * q11_3 + m021 * q02 * q11_3 - 6 * m021 * q00_2 * q01 * q12 + 5 * m121 * q00_2 * q01 * q12
                - 6 * m021 * q01_3 * q12 + 5 * m121 * q01_3 * q12 + 2 * m101 * q00 * q01 * q02 * q12
                - 6 * m021 * q01 * q02_2 * q12 + 7 * m121 * q01 * q02_2 * q12 - 6 * m001 * q00_2 * q03 * q12
                + 5 * m101 * q00_2 * q03 * q12 - 6 * m001 * q01_2 * q03 * q12 + 5 * m101 * q01_2 * q03 * q12
                - 2 * m121 * q00 * q02 * q03 * q12 - 6 * m001 * q02_2 * q03 * q12 + 7 * m101 * q02_2 * q03 * q12
                - 6 * m021 * q01 * q03_2 * q12 + 5 * m121 * q01 * q03_2 * q12 - 6 * m001 * q03_3 * q12 + 5 * m101 * q03_3 * q12
                + 2 * m021 * q00 * q01 * q10 * q12 - 2 * m121 * q00 * q01 * q10 * q12 + 2 * m001 * q01 * q02 * q10 * q12
                - 2 * m101 * q01 * q02 * q10 * q12 + 2 * m001 * q00 * q03 * q10 * q12 - 2 * m101 * q00 * q03 * q10 * q12
                - 2 * m021 * q02 * q03 * q10 * q12 + 2 * m121 * q02 * q03 * q10 * q12 + m021 * q01 * q10_2 * q12
                + m001 * q03 * q10_2 * q12 + 5 * m021 * q00_2 * q11 * q12 - 3 * m121 * q00_2 * q11 * q12
                + 7 * m021 * q01_2 * q11 * q12 - 5 * m121 * q01_2 * q11 * q12 + 2 * m001 * q00 * q02 * q11 * q12
                - 2 * m101 * q00 * q02 * q11 * q12 + 7 * m021 * q02_2 * q11 * q12 - 5 * m121 * q02_2 * q11 * q12
                + 2 * m001 * q01 * q03 * q11 * q12 - 2 * m101 * q01 * q03 * q11 * q12 + 5 * m021 * q03_2 * q11 * q12
                - 3 * m121 * q03_2 * q11 * q12 - 2 * m021 * q00 * q10 * q11 * q12 - 2 * m001 * q02 * q10 * q11 * q12
                - m021 * q01 * q11_2 * q12 + m001 * q03 * q11_2 * q12 - 3 * m001 * q00 * q01 * q12_2 + m101 * q00 * q01 * q12_2
                - m021 * q01 * q02 * q12_2 - m121 * q01 * q02 * q12_2 + 3 * m021 * q00 * q03 * q12_2 - m121 * q00 * q03 * q12_2
                - m001 * q02 * q03 * q12_2 - m101 * q02 * q03 * q12_2 + m001 * q01 * q10 * q12_2 - m021 * q03 * q10 * q12_2
                + m001 * q00 * q11 * q12_2 - m021 * q02 * q11 * q12_2 + m021 * q01 * q12_3 + m001 * q03 * q12_3
                + ( -6 * m001 * q00_2 * q02 + 5 * m101 * q00_2 * q02 - 6 * m001 * q01_2 * q02 + 5 * m101 * q01_2 * q02
                    - 6 * m001 * q02_3 + 5 * m101 * q02_3 + 2 * m101 * q00 * q01 * q03 - 6 * m001 * q02 * q03_2
                    + 7 * m101 * q02 * q03_2 + 2 * m001 * q00 * q02 * q10 - 2 * m101 * q00 * q02 * q10 + 2 * m001 * q01 * q03 * q10
                    - 2 * m101 * q01 * q03 * q10 + m001 * q02 * q10_2 + 2 * m001 * q01 * q02 * q11 - 2 * m101 * q01 * q02 * q11
                    + 2 * m001 * q00 * q03 * q11 - 2 * m101 * q00 * q03 * q11 - 2 * m001 * q03 * q10 * q11 + m001 * q02 * q11_2
                    + ( -( m101 * ( 3 * q00_2 + 3 * q01_2 + 5 * ( q02_2 + q03_2 ) ) )
                        + m001 * ( 5 * q00_2 + 7 * ( q02_2 + q03_2 ) - 2 * q00 * q10 + q01 * ( 5 * q01 - 2 * q11 ) ) )
                          * q12
                    - m001 * q02 * q12_2
                    + m121
                          * ( -5 * q00_3 + 5 * q00_2 * q10 + 3 * q01_2 * q10 + 3 * q02_2 * q10 + 5 * q03_2 * q10
                              - 2 * q02 * q03 * q11 + 2 * q01 * q03 * ( q02 - q12 )
                              + q00 * ( -5 * q01_2 - 5 * q02_2 - 7 * q03_2 + 2 * q01 * q11 + 2 * q02 * q12 ) )
                    + m021
                          * ( 6 * q00_3 - 7 * q00_2 * q10 - 5 * q01_2 * q10 - 5 * q02_2 * q10 - 7 * q03_2 * q10
                              + 2 * q02 * q03 * q11 + 2 * q01 * q10 * q11 + 2 * ( q01 * q03 + q02 * q10 - q03 * q11 ) * q12
                              + q00 * ( 6 * q01_2 + 6 * q02_2 + 6 * q03_2 + q10_2 - 2 * q01 * q11 - q11_2 - 2 * q02 * q12 - q12_2 ) ) )
                      * q13
                + ( m101 * q00 * q01 - 3 * m021 * q01 * q02 + m121 * q01 * q02 + m021 * q00 * q03 + m121 * q00 * q03
                    - m101 * q02 * q03 + m021 * q03 * q10 + m021 * q02 * q11 + m021 * q01 * q12
                    + m001 * ( q01 * q10 + q00 * ( -3 * q01 + q11 ) - q03 * ( q02 + q12 ) ) )
                      * q13_2
                + ( -( m021 * q00 ) + m001 * q02 ) * q13_3
                + m011
                      * ( -3 * q00_4 + 3 * q01_4 - 3 * q02_4 + 3 * q03_4 + 6 * q00_3 * q10 - 4 * q03_2 * q10_2
                          - 6 * q01_3 * q11 + q03_2 * q11_2 + 6 * q02 * q03_2 * q12 - 2 * q02 * q11_2 * q12
                          - 3 * q02_2 * q12_2 - 4 * q03_2 * q12_2 + q02_2 * ( -q10_2 + 4 * q11_2 + 6 * q02 * q12 )
                          + 2 * q03 * ( -3 * q02_2 - 3 * q03_2 + q10_2 + q12_2 ) * q13 + ( 4 * q02_2 + 3 * q03_2 - 2 * q02 * q12 ) * q13_2
                          + 2 * q01 * q11 * ( -3 * q02_2 - 3 * q03_2 + q10_2 + q12_2 + 2 * q03 * q13 )
                          - q00_2 * ( 6 * q02_2 + 3 * q10_2 + 6 * q01 * q11 - 4 * q11_2 - 6 * q02 * q12 + q12_2 + 6 * q03 * q13 - 4 * q13_2 )
                          + 2 * q00 * q10 * ( 3 * q01_2 + 3 * q02_2 + 3 * q03_2 - q11_2 - 2 * q02 * q12 - q13_2 )
                          + q01_2 * ( 6 * q03_2 - 4 * q10_2 + 3 * q11_2 + 6 * q02 * q12 - 4 * q12_2 - 6 * q03 * q13 + q13_2 ) )
                + m111
                      * ( 3 * q00_4 + 3 * q02_4 - 3 * q013_2 - 6 * q00_3 * q10 + 2 * q03_2 * q10_2 + 6 * q01 * q03_2 * q11
                          - 3 * q01_2 * q11_2 - q03_2 * q11_2 + 2 * q01_2 * ( q10_2 + 3 * q01 * q11 ) - 6 * q02_3 * q12
                          - 4 * q02 * ( q01_2 + q03_2 ) * q12 + 2 * q01_2 * q12_2 + 2 * q03_2 * q12_2
                          - 2 * q00 * q10 * ( 2 * q01_2 + 3 * q02_2 + 2 * q03_2 - 2 * q02 * q12 )
                          + 2 * q03 * ( 3 * ( q01_2 + q03_2 ) - 2 * q01 * q11 ) * q13 - ( q01_2 + 3 * q03_2 ) * q13_2
                          + q02_2 * ( q10_2 + 4 * q01 * q11 - 2 * q11_2 + 3 * q12_2 + 4 * q03 * q13 - 2 * q13_2 )
                          + q00_2 * ( 6 * q02_2 + 3 * q10_2 + 4 * q01 * q11 - 6 * q02 * q12 + q12_2 - 2 * ( q11_2 + q13 * ( -2 * q03 + q13 ) ) ) ) ),
        -2
            * ( 6 * m002 * q00_3 * q01 - 6 * m102 * q00_3 * q01 + 6 * m002 * q00 * q01_3 - 6 * m102 * q00 * q01_3
                + 6 * m022 * q00_2 * q01 * q02 - 6 * m122 * q00_2 * q01 * q02 + 6 * m022 * q01_3 * q02 - 6 * m122 * q01_3 * q02
                + 6 * m002 * q00 * q01 * q02_2 - 6 * m102 * q00 * q01 * q02_2 + 6 * m022 * q01 * q02_3 - 6 * m122 * q01 * q02_3
                - 6 * m022 * q00_3 * q03 + 6 * m122 * q00_3 * q03 - 6 * m022 * q00 * q01_2 * q03 + 6 * m122 * q00 * q01_2 * q03
                + 6 * m002 * q00_2 * q02 * q03 - 6 * m102 * q00_2 * q02 * q03 + 6 * m002 * q01_2 * q02 * q03
                - 6 * m102 * q01_2 * q02 * q03 - 6 * m022 * q00 * q02_2 * q03 + 6 * m122 * q00 * q02_2 * q03
                + 6 * m002 * q02_3 * q03 - 6 * m102 * q02_3 * q03 + 6 * m002 * q00 * q01 * q03_2 - 6 * m102 * q00 * q01 * q03_2
                + 6 * m022 * q01 * q02 * q03_2 - 6 * m122 * q01 * q02 * q03_2 - 6 * m022 * q00 * q03_3 + 6 * m122 * q00 * q03_3
                + 6 * m002 * q02 * q03_3 - 6 * m102 * q02 * q03_3 - 6 * m002 * q00_2 * q01 * q10 + 7 * m102 * q00_2 * q01 * q10
                - 6 * m002 * q01_3 * q10 + 5 * m102 * q01_3 * q10 + 2 * m122 * q00 * q01 * q02 * q10 - 6 * m002 * q01 * q02_2 * q10
                + 5 * m102 * q01 * q02_2 * q10 + 6 * m022 * q00_2 * q03 * q10 - 7 * m122 * q00_2 * q03 * q10
                + 6 * m022 * q01_2 * q03 * q10 - 5 * m122 * q01_2 * q03 * q10 + 2 * m102 * q00 * q02 * q03 * q10
                + 6 * m022 * q02_2 * q03 * q10 - 5 * m122 * q02_2 * q03 * q10 - 6 * m002 * q01 * q03_2 * q10
                + 5 * m102 * q01 * q03_2 * q10 + 6 * m022 * q03_3 * q10 - 5 * m122 * q03_3 * q10 - m002 * q00 * q01 * q10_2
                - m102 * q00 * q01 * q10_2 - 3 * m022 * q01 * q02 * q10_2 + m122 * q01 * q02 * q10_2 + m022 * q00 * q03 * q10_2
                + m122 * q00 * q03 * q10_2 - 3 * m002 * q02 * q03 * q10_2 + m102 * q02 * q03 * q10_2 + m002 * q01 * q10_3
                - m022 * q03 * q10_3 - 6 * m002 * q00_3 * q11 + 5 * m102 * q00_3 * q11 - 6 * m002 * q00 * q01_2 * q11
                + 7 * m102 * q00 * q01_2 * q11 - 6 * m022 * q00_2 * q02 * q11 + 5 * m122 * q00_2 * q02 * q11
                - 6 * m022 * q01_2 * q02 * q11 + 7 * m122 * q01_2 * q02 * q11 - 6 * m002 * q00 * q02_2 * q11
                + 5 * m102 * q00 * q02_2 * q11 - 6 * m022 * q02_3 * q11 + 5 * m122 * q02_3 * q11
                - 2 * m122 * q00 * q01 * q03 * q11 + 2 * m102 * q01 * q02 * q03 * q11 - 6 * m002 * q00 * q03_2 * q11
                + 5 * m102 * q00 * q03_2 * q11 - 6 * m022 * q02 * q03_2 * q11 + 5 * m122 * q02 * q03_2 * q11
                + 7 * m002 * q00_2 * q10 * q11 - 5 * m102 * q00_2 * q10 * q11 + 7 * m002 * q01_2 * q10 * q11
                - 5 * m102 * q01_2 * q10 * q11 + 2 * m022 * q00 * q02 * q10 * q11 - 2 * m122 * q00 * q02 * q10 * q11
                + 5 * m002 * q02_2 * q10 * q11 - 3 * m102 * q02_2 * q10 * q11 - 2 * m022 * q01 * q03 * q10 * q11
                + 2 * m122 * q01 * q03 * q10 * q11 + 5 * m002 * q03_2 * q10 * q11 - 3 * m102 * q03_2 * q10 * q11
                - m002 * q00 * q10_2 * q11 + m022 * q02 * q10_2 * q11 - m002 * q00 * q01 * q11_2 - m102 * q00 * q01 * q11_2
                - m022 * q01 * q02 * q11_2 - m122 * q01 * q02 * q11_2 + 3 * m022 * q00 * q03 * q11_2 - m122 * q00 * q03 * q11_2
                - 3 * m002 * q02 * q03 * q11_2 + m102 * q02 * q03 * q11_2 - m002 * q01 * q10 * q11_2 - m022 * q03 * q10 * q11_2
                + m002 * q00 * q11_3 + m022 * q02 * q11_3 - 6 * m022 * q00_2 * q01 * q12 + 5 * m122 * q00_2 * q01 * q12
                - 6 * m022 * q01_3 * q12 + 5 * m122 * q01_3 * q12 + 2 * m102 * q00 * q01 * q02 * q12
                - 6 * m022 * q01 * q02_2 * q12 + 7 * m122 * q01 * q02_2 * q12 - 6 * m002 * q00_2 * q03 * q12
                + 5 * m102 * q00_2 * q03 * q12 - 6 * m002 * q01_2 * q03 * q12 + 5 * m102 * q01_2 * q03 * q12
                - 2 * m122 * q00 * q02 * q03 * q12 - 6 * m002 * q02_2 * q03 * q12 + 7 * m102 * q02_2 * q03 * q12
                - 6 * m022 * q01 * q03_2 * q12 + 5 * m122 * q01 * q03_2 * q12 - 6 * m002 * q03_3 * q12 + 5 * m102 * q03_3 * q12
                + 2 * m022 * q00 * q01 * q10 * q12 - 2 * m122 * q00 * q01 * q10 * q12 + 2 * m002 * q01 * q02 * q10 * q12
                - 2 * m102 * q01 * q02 * q10 * q12 + 2 * m002 * q00 * q03 * q10 * q12 - 2 * m102 * q00 * q03 * q10 * q12
                - 2 * m022 * q02 * q03 * q10 * q12 + 2 * m122 * q02 * q03 * q10 * q12 + m022 * q01 * q10_2 * q12
                + m002 * q03 * q10_2 * q12 + 5 * m022 * q00_2 * q11 * q12 - 3 * m122 * q00_2 * q11 * q12
                + 7 * m022 * q01_2 * q11 * q12 - 5 * m122 * q01_2 * q11 * q12 + 2 * m002 * q00 * q02 * q11 * q12
                - 2 * m102 * q00 * q02 * q11 * q12 + 7 * m022 * q02_2 * q11 * q12 - 5 * m122 * q02_2 * q11 * q12
                + 2 * m002 * q01 * q03 * q11 * q12 - 2 * m102 * q01 * q03 * q11 * q12 + 5 * m022 * q03_2 * q11 * q12
                - 3 * m122 * q03_2 * q11 * q12 - 2 * m022 * q00 * q10 * q11 * q12 - 2 * m002 * q02 * q10 * q11 * q12
                - m022 * q01 * q11_2 * q12 + m002 * q03 * q11_2 * q12 - 3 * m002 * q00 * q01 * q12_2 + m102 * q00 * q01 * q12_2
                - m022 * q01 * q02 * q12_2 - m122 * q01 * q02 * q12_2 + 3 * m022 * q00 * q03 * q12_2 - m122 * q00 * q03 * q12_2
                - m002 * q02 * q03 * q12_2 - m102 * q02 * q03 * q12_2 + m002 * q01 * q10 * q12_2 - m022 * q03 * q10 * q12_2
                + m002 * q00 * q11 * q12_2 - m022 * q02 * q11 * q12_2 + m022 * q01 * q12_3 + m002 * q03 * q12_3
                + ( -6 * m002 * q00_2 * q02 + 5 * m102 * q00_2 * q02 - 6 * m002 * q01_2 * q02 + 5 * m102 * q01_2 * q02
                    - 6 * m002 * q02_3 + 5 * m102 * q02_3 + 2 * m102 * q00 * q01 * q03 - 6 * m002 * q02 * q03_2
                    + 7 * m102 * q02 * q03_2 + 2 * m002 * q00 * q02 * q10 - 2 * m102 * q00 * q02 * q10 + 2 * m002 * q01 * q03 * q10
                    - 2 * m102 * q01 * q03 * q10 + m002 * q02 * q10_2 + 2 * m002 * q01 * q02 * q11 - 2 * m102 * q01 * q02 * q11
                    + 2 * m002 * q00 * q03 * q11 - 2 * m102 * q00 * q03 * q11 - 2 * m002 * q03 * q10 * q11 + m002 * q02 * q11_2
                    + ( -( m102 * ( 3 * q00_2 + 3 * q01_2 + 5 * ( q02_2 + q03_2 ) ) )
                        + m002 * ( 5 * q00_2 + 7 * ( q02_2 + q03_2 ) - 2 * q00 * q10 + q01 * ( 5 * q01 - 2 * q11 ) ) )
                          * q12
                    - m002 * q02 * q12_2
                    + m122
                          * ( -5 * q00_3 + 5 * q00_2 * q10 + 3 * q01_2 * q10 + 3 * q02_2 * q10 + 5 * q03_2 * q10
                              - 2 * q02 * q03 * q11 + 2 * q01 * q03 * ( q02 - q12 )
                              + q00 * ( -5 * q01_2 - 5 * q02_2 - 7 * q03_2 + 2 * q01 * q11 + 2 * q02 * q12 ) )
                    + m022
                          * ( 6 * q00_3 - 7 * q00_2 * q10 - 5 * q01_2 * q10 - 5 * q02_2 * q10 - 7 * q03_2 * q10
                              + 2 * q02 * q03 * q11 + 2 * q01 * q10 * q11 + 2 * ( q01 * q03 + q02 * q10 - q03 * q11 ) * q12
                              + q00 * ( 6 * q01_2 + 6 * q02_2 + 6 * q03_2 + q10_2 - 2 * q01 * q11 - q11_2 - 2 * q02 * q12 - q12_2 ) ) )
                      * q13
                + ( m102 * q00 * q01 - 3 * m022 * q01 * q02 + m122 * q01 * q02 + m022 * q00 * q03 + m122 * q00 * q03
                    - m102 * q02 * q03 + m022 * q03 * q10 + m022 * q02 * q11 + m022 * q01 * q12
                    + m002 * ( q01 * q10 + q00 * ( -3 * q01 + q11 ) - q03 * ( q02 + q12 ) ) )
                      * q13_2
                + ( -( m022 * q00 ) + m002 * q02 ) * q13_3
                + m012
                      * ( -3 * q00_4 + 3 * q01_4 - 3 * q02_4 + 3 * q03_4 + 6 * q00_3 * q10 - 4 * q03_2 * q10_2
                          - 6 * q01_3 * q11 + q03_2 * q11_2 + 6 * q02 * q03_2 * q12 - 2 * q02 * q11_2 * q12
                          - 3 * q02_2 * q12_2 - 4 * q03_2 * q12_2 + q02_2 * ( -q10_2 + 4 * q11_2 + 6 * q02 * q12 )
                          + 2 * q03 * ( -3 * q02_2 - 3 * q03_2 + q10_2 + q12_2 ) * q13 + ( 4 * q02_2 + 3 * q03_2 - 2 * q02 * q12 ) * q13_2
                          + 2 * q01 * q11 * ( -3 * q02_2 - 3 * q03_2 + q10_2 + q12_2 + 2 * q03 * q13 )
                          - q00_2 * ( 6 * q02_2 + 3 * q10_2 + 6 * q01 * q11 - 4 * q11_2 - 6 * q02 * q12 + q12_2 + 6 * q03 * q13 - 4 * q13_2 )
                          + 2 * q00 * q10 * ( 3 * q01_2 + 3 * q02_2 + 3 * q03_2 - q11_2 - 2 * q02 * q12 - q13_2 )
                          + q01_2 * ( 6 * q03_2 - 4 * q10_2 + 3 * q11_2 + 6 * q02 * q12 - 4 * q12_2 - 6 * q03 * q13 + q13_2 ) )
                + m112
                      * ( 3 * q00_4 + 3 * q02_4 - 3 * q013_2 - 6 * q00_3 * q10 + 2 * q03_2 * q10_2 + 6 * q01 * q03_2 * q11
                          - 3 * q01_2 * q11_2 - q03_2 * q11_2 + 2 * q01_2 * ( q10_2 + 3 * q01 * q11 ) - 6 * q02_3 * q12
                          - 4 * q02 * ( q01_2 + q03_2 ) * q12 + 2 * q01_2 * q12_2 + 2 * q03_2 * q12_2
                          - 2 * q00 * q10 * ( 2 * q01_2 + 3 * q02_2 + 2 * q03_2 - 2 * q02 * q12 )
                          + 2 * q03 * ( 3 * ( q01_2 + q03_2 ) - 2 * q01 * q11 ) * q13 - ( q01_2 + 3 * q03_2 ) * q13_2
                          + q02_2 * ( q10_2 + 4 * q01 * q11 - 2 * q11_2 + 3 * q12_2 + 4 * q03 * q13 - 2 * q13_2 )
                          + q00_2 * ( 6 * q02_2 + 3 * q10_2 + 4 * q01 * q11 - 6 * q02 * q12 + q12_2 - 2 * ( q11_2 + q13 * ( -2 * q03 + q13 ) ) ) ) ),
        -2
            * ( 6 * m003 * q00_3 * q01 - 6 * m103 * q00_3 * q01 + 6 * m003 * q00 * q01_3 - 6 * m103 * q00 * q01_3
                + 6 * m023 * q00_2 * q01 * q02 - 6 * m123 * q00_2 * q01 * q02 + 6 * m023 * q01_3 * q02 - 6 * m123 * q01_3 * q02
                + 6 * m003 * q00 * q01 * q02_2 - 6 * m103 * q00 * q01 * q02_2 + 6 * m023 * q01 * q02_3 - 6 * m123 * q01 * q02_3
                - 6 * m023 * q00_3 * q03 + 6 * m123 * q00_3 * q03 - 6 * m023 * q00 * q01_2 * q03 + 6 * m123 * q00 * q01_2 * q03
                + 6 * m003 * q00_2 * q02 * q03 - 6 * m103 * q00_2 * q02 * q03 + 6 * m003 * q01_2 * q02 * q03
                - 6 * m103 * q01_2 * q02 * q03 - 6 * m023 * q00 * q02_2 * q03 + 6 * m123 * q00 * q02_2 * q03
                + 6 * m003 * q02_3 * q03 - 6 * m103 * q02_3 * q03 + 6 * m003 * q00 * q01 * q03_2 - 6 * m103 * q00 * q01 * q03_2
                + 6 * m023 * q01 * q02 * q03_2 - 6 * m123 * q01 * q02 * q03_2 - 6 * m023 * q00 * q03_3 + 6 * m123 * q00 * q03_3
                + 6 * m003 * q02 * q03_3 - 6 * m103 * q02 * q03_3 - 6 * m003 * q00_2 * q01 * q10 + 7 * m103 * q00_2 * q01 * q10
                - 6 * m003 * q01_3 * q10 + 5 * m103 * q01_3 * q10 + 2 * m123 * q00 * q01 * q02 * q10 - 6 * m003 * q01 * q02_2 * q10
                + 5 * m103 * q01 * q02_2 * q10 + 6 * m023 * q00_2 * q03 * q10 - 7 * m123 * q00_2 * q03 * q10
                + 6 * m023 * q01_2 * q03 * q10 - 5 * m123 * q01_2 * q03 * q10 + 2 * m103 * q00 * q02 * q03 * q10
                + 6 * m023 * q02_2 * q03 * q10 - 5 * m123 * q02_2 * q03 * q10 - 6 * m003 * q01 * q03_2 * q10
                + 5 * m103 * q01 * q03_2 * q10 + 6 * m023 * q03_3 * q10 - 5 * m123 * q03_3 * q10 - m003 * q00 * q01 * q10_2
                - m103 * q00 * q01 * q10_2 - 3 * m023 * q01 * q02 * q10_2 + m123 * q01 * q02 * q10_2 + m023 * q00 * q03 * q10_2
                + m123 * q00 * q03 * q10_2 - 3 * m003 * q02 * q03 * q10_2 + m103 * q02 * q03 * q10_2 + m003 * q01 * q10_3
                - m023 * q03 * q10_3 - 6 * m003 * q00_3 * q11 + 5 * m103 * q00_3 * q11 - 6 * m003 * q00 * q01_2 * q11
                + 7 * m103 * q00 * q01_2 * q11 - 6 * m023 * q00_2 * q02 * q11 + 5 * m123 * q00_2 * q02 * q11
                - 6 * m023 * q01_2 * q02 * q11 + 7 * m123 * q01_2 * q02 * q11 - 6 * m003 * q00 * q02_2 * q11
                + 5 * m103 * q00 * q02_2 * q11 - 6 * m023 * q02_3 * q11 + 5 * m123 * q02_3 * q11
                - 2 * m123 * q00 * q01 * q03 * q11 + 2 * m103 * q01 * q02 * q03 * q11 - 6 * m003 * q00 * q03_2 * q11
                + 5 * m103 * q00 * q03_2 * q11 - 6 * m023 * q02 * q03_2 * q11 + 5 * m123 * q02 * q03_2 * q11
                + 7 * m003 * q00_2 * q10 * q11 - 5 * m103 * q00_2 * q10 * q11 + 7 * m003 * q01_2 * q10 * q11
                - 5 * m103 * q01_2 * q10 * q11 + 2 * m023 * q00 * q02 * q10 * q11 - 2 * m123 * q00 * q02 * q10 * q11
                + 5 * m003 * q02_2 * q10 * q11 - 3 * m103 * q02_2 * q10 * q11 - 2 * m023 * q01 * q03 * q10 * q11
                + 2 * m123 * q01 * q03 * q10 * q11 + 5 * m003 * q03_2 * q10 * q11 - 3 * m103 * q03_2 * q10 * q11
                - m003 * q00 * q10_2 * q11 + m023 * q02 * q10_2 * q11 - m003 * q00 * q01 * q11_2 - m103 * q00 * q01 * q11_2
                - m023 * q01 * q02 * q11_2 - m123 * q01 * q02 * q11_2 + 3 * m023 * q00 * q03 * q11_2 - m123 * q00 * q03 * q11_2
                - 3 * m003 * q02 * q03 * q11_2 + m103 * q02 * q03 * q11_2 - m003 * q01 * q10 * q11_2 - m023 * q03 * q10 * q11_2
                + m003 * q00 * q11_3 + m023 * q02 * q11_3 - 6 * m023 * q00_2 * q01 * q12 + 5 * m123 * q00_2 * q01 * q12
                - 6 * m023 * q01_3 * q12 + 5 * m123 * q01_3 * q12 + 2 * m103 * q00 * q01 * q02 * q12
                - 6 * m023 * q01 * q02_2 * q12 + 7 * m123 * q01 * q02_2 * q12 - 6 * m003 * q00_2 * q03 * q12
                + 5 * m103 * q00_2 * q03 * q12 - 6 * m003 * q01_2 * q03 * q12 + 5 * m103 * q01_2 * q03 * q12
                - 2 * m123 * q00 * q02 * q03 * q12 - 6 * m003 * q02_2 * q03 * q12 + 7 * m103 * q02_2 * q03 * q12
                - 6 * m023 * q01 * q03_2 * q12 + 5 * m123 * q01 * q03_2 * q12 - 6 * m003 * q03_3 * q12 + 5 * m103 * q03_3 * q12
                + 2 * m023 * q00 * q01 * q10 * q12 - 2 * m123 * q00 * q01 * q10 * q12 + 2 * m003 * q01 * q02 * q10 * q12
                - 2 * m103 * q01 * q02 * q10 * q12 + 2 * m003 * q00 * q03 * q10 * q12 - 2 * m103 * q00 * q03 * q10 * q12
                - 2 * m023 * q02 * q03 * q10 * q12 + 2 * m123 * q02 * q03 * q10 * q12 + m023 * q01 * q10_2 * q12
                + m003 * q03 * q10_2 * q12 + 5 * m023 * q00_2 * q11 * q12 - 3 * m123 * q00_2 * q11 * q12
                + 7 * m023 * q01_2 * q11 * q12 - 5 * m123 * q01_2 * q11 * q12 + 2 * m003 * q00 * q02 * q11 * q12
                - 2 * m103 * q00 * q02 * q11 * q12 + 7 * m023 * q02_2 * q11 * q12 - 5 * m123 * q02_2 * q11 * q12
                + 2 * m003 * q01 * q03 * q11 * q12 - 2 * m103 * q01 * q03 * q11 * q12 + 5 * m023 * q03_2 * q11 * q12
                - 3 * m123 * q03_2 * q11 * q12 - 2 * m023 * q00 * q10 * q11 * q12 - 2 * m003 * q02 * q10 * q11 * q12
                - m023 * q01 * q11_2 * q12 + m003 * q03 * q11_2 * q12 - 3 * m003 * q00 * q01 * q12_2 + m103 * q00 * q01 * q12_2
                - m023 * q01 * q02 * q12_2 - m123 * q01 * q02 * q12_2 + 3 * m023 * q00 * q03 * q12_2 - m123 * q00 * q03 * q12_2
                - m003 * q02 * q03 * q12_2 - m103 * q02 * q03 * q12_2 + m003 * q01 * q10 * q12_2 - m023 * q03 * q10 * q12_2
                + m003 * q00 * q11 * q12_2 - m023 * q02 * q11 * q12_2 + m023 * q01 * q12_3 + m003 * q03 * q12_3
                + 6 * m023 * q00_3 * q13 - 5 * m123 * q00_3 * q13 + 6 * m023 * q00 * q01_2 * q13 - 5 * m123 * q00 * q01_2 * q13
                - 6 * m003 * q00_2 * q02 * q13 + 5 * m103 * q00_2 * q02 * q13 - 6 * m003 * q01_2 * q02 * q13
                + 5 * m103 * q01_2 * q02 * q13 + 6 * m023 * q00 * q02_2 * q13 - 5 * m123 * q00 * q02_2 * q13
                - 6 * m003 * q02_3 * q13 + 5 * m103 * q02_3 * q13 + 2 * m103 * q00 * q01 * q03 * q13
                + 2 * m123 * q01 * q02 * q03 * q13 + 6 * m023 * q00 * q03_2 * q13 - 7 * m123 * q00 * q03_2 * q13
                - 6 * m003 * q02 * q03_2 * q13 + 7 * m103 * q02 * q03_2 * q13 - 7 * m023 * q00_2 * q10 * q13
                + 5 * m123 * q00_2 * q10 * q13 - 5 * m023 * q01_2 * q10 * q13 + 3 * m123 * q01_2 * q10 * q13
                + 2 * m003 * q00 * q02 * q10 * q13 - 2 * m103 * q00 * q02 * q10 * q13 - 5 * m023 * q02_2 * q10 * q13
                + 3 * m123 * q02_2 * q10 * q13 + 2 * m003 * q01 * q03 * q10 * q13 - 2 * m103 * q01 * q03 * q10 * q13
                - 7 * m023 * q03_2 * q10 * q13 + 5 * m123 * q03_2 * q10 * q13 + m023 * q00 * q10_2 * q13 + m003 * q02 * q10_2 * q13
                - 2 * m023 * q00 * q01 * q11 * q13 + 2 * m123 * q00 * q01 * q11 * q13 + 2 * m003 * q01 * q02 * q11 * q13
                - 2 * m103 * q01 * q02 * q11 * q13 + 2 * m003 * q00 * q03 * q11 * q13 - 2 * m103 * q00 * q03 * q11 * q13
                + 2 * m023 * q02 * q03 * q11 * q13 - 2 * m123 * q02 * q03 * q11 * q13 + 2 * m023 * q01 * q10 * q11 * q13
                - 2 * m003 * q03 * q10 * q11 * q13 - m023 * q00 * q11_2 * q13 + m003 * q02 * q11_2 * q13
                + 5 * m003 * q00_2 * q12 * q13 - 3 * m103 * q00_2 * q12 * q13 + 5 * m003 * q01_2 * q12 * q13
                - 3 * m103 * q01_2 * q12 * q13 - 2 * m023 * q00 * q02 * q12 * q13 + 2 * m123 * q00 * q02 * q12 * q13
                + 7 * m003 * q02_2 * q12 * q13 - 5 * m103 * q02_2 * q12 * q13 + 2 * m023 * q01 * q03 * q12 * q13
                - 2 * m123 * q01 * q03 * q12 * q13 + 7 * m003 * q03_2 * q12 * q13 - 5 * m103 * q03_2 * q12 * q13
                - 2 * m003 * q00 * q10 * q12 * q13 + 2 * m023 * q02 * q10 * q12 * q13 - 2 * m003 * q01 * q11 * q12 * q13
                - 2 * m023 * q03 * q11 * q12 * q13 - m023 * q00 * q12_2 * q13 - m003 * q02 * q12_2 * q13
                - 3 * m003 * q00 * q01 * q13_2 + m103 * q00 * q01 * q13_2 - 3 * m023 * q01 * q02 * q13_2 + m123 * q01 * q02 * q13_2
                + m023 * q00 * q03 * q13_2 + m123 * q00 * q03 * q13_2 - m003 * q02 * q03 * q13_2 - m103 * q02 * q03 * q13_2
                + m003 * q01 * q10 * q13_2 + m023 * q03 * q10 * q13_2 + m003 * q00 * q11 * q13_2 + m023 * q02 * q11 * q13_2
                + m023 * q01 * q12 * q13_2 - m003 * q03 * q12 * q13_2 - m023 * q00 * q13_3 + m003 * q02 * q13_3
                + m013
                      * ( -3 * q00_4 + 3 * q01_4 - 3 * q02_4 + 3 * q03_4 + 6 * q00_3 * q10 - 4 * q03_2 * q10_2
                          - 6 * q01_3 * q11 + q03_2 * q11_2 + 6 * q02 * q03_2 * q12 - 2 * q02 * q11_2 * q12
                          - 3 * q02_2 * q12_2 - 4 * q03_2 * q12_2 + q02_2 * ( -q10_2 + 4 * q11_2 + 6 * q02 * q12 )
                          + 2 * q03 * ( -3 * q02_2 - 3 * q03_2 + q10_2 + q12_2 ) * q13 + ( 4 * q02_2 + 3 * q03_2 - 2 * q02 * q12 ) * q13_2
                          + 2 * q01 * q11 * ( -3 * q02_2 - 3 * q03_2 + q10_2 + q12_2 + 2 * q03 * q13 )
                          - q00_2 * ( 6 * q02_2 + 3 * q10_2 + 6 * q01 * q11 - 4 * q11_2 - 6 * q02 * q12 + q12_2 + 6 * q03 * q13 - 4 * q13_2 )
                          + 2 * q00 * q10 * ( 3 * q01_2 + 3 * q02_2 + 3 * q03_2 - q11_2 - 2 * q02 * q12 - q13_2 )
                          + q01_2 * ( 6 * q03_2 - 4 * q10_2 + 3 * q11_2 + 6 * q02 * q12 - 4 * q12_2 - 6 * q03 * q13 + q13_2 ) )
                + m113
                      * ( 3 * q00_4 + 3 * q02_4 - 3 * q013_2 - 6 * q00_3 * q10 + 2 * q03_2 * q10_2 + 6 * q01 * q03_2 * q11
                          - 3 * q01_2 * q11_2 - q03_2 * q11_2 + 2 * q01_2 * ( q10_2 + 3 * q01 * q11 ) - 6 * q02_3 * q12
                          - 4 * q02 * ( q01_2 + q03_2 ) * q12 + 2 * q01_2 * q12_2 + 2 * q03_2 * q12_2
                          - 2 * q00 * q10 * ( 2 * q01_2 + 3 * q02_2 + 2 * q03_2 - 2 * q02 * q12 )
                          + 2 * q03 * ( 3 * ( q01_2 + q03_2 ) - 2 * q01 * q11 ) * q13 - ( q01_2 + 3 * q03_2 ) * q13_2
                          + q02_2 * ( q10_2 + 4 * q01 * q11 - 2 * q11_2 + 3 * q12_2 + 4 * q03 * q13 - 2 * q13_2 )
                          + q00_2
                                * ( 6 * q02_2 + 3 * q10_2 + 4 * q01 * q11 - 6 * q02 * q12 + q12_2
                                    - 2 * ( q11_2 + q13 * ( -2 * q03 + q13 ) ) ) ) ) );
    c3[1] = SRNumeratorDerivativeTerm(
        -4 * ( a - q00 * q10 - q01 * q11 - q02 * q12 - q03 * q13 )
            * ( -2
                    * ( ( m020 - m120 ) * ( ( q01 - q11 ) * ( q02 - q12 ) - ( q00 - q10 ) * ( q03 - q13 ) )
                        + m100 * ( -( ( q00 - q10 ) * ( q01 - q11 ) ) - ( q02 - q12 ) * ( q03 - q13 ) )
                        + m000 * ( ( q00 - q10 ) * ( q01 - q11 ) + ( q02 - q12 ) * ( q03 - q13 ) ) )
                + m110 * ( q01_2 - q00_10_2 - 2 * q01 * q11 + q11_2 - q02_12_2 + q03_13_2 )
                + m010 * ( q00_10_2 - q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ),
        -4 * ( a - q00 * q10 - q01 * q11 - q02 * q12 - q03 * q13 )
            * ( -2
                    * ( ( m021 - m121 ) * ( ( q01 - q11 ) * ( q02 - q12 ) - ( q00 - q10 ) * ( q03 - q13 ) )
                        + m101 * ( -( ( q00 - q10 ) * ( q01 - q11 ) ) - ( q02 - q12 ) * ( q03 - q13 ) )
                        + m001 * ( ( q00 - q10 ) * ( q01 - q11 ) + ( q02 - q12 ) * ( q03 - q13 ) ) )
                + m111 * ( q01_2 - q00_10_2 - 2 * q01 * q11 + q11_2 - q02_12_2 + q03_13_2 )
                + m011 * ( q00_10_2 - q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ),
        -4 * ( a - q00 * q10 - q01 * q11 - q02 * q12 - q03 * q13 )
            * ( -2
                    * ( ( m022 - m122 ) * ( ( q01 - q11 ) * ( q02 - q12 ) - ( q00 - q10 ) * ( q03 - q13 ) )
                        + m102 * ( -( ( q00 - q10 ) * ( q01 - q11 ) ) - ( q02 - q12 ) * ( q03 - q13 ) )
                        + m002 * ( ( q00 - q10 ) * ( q01 - q11 ) + ( q02 - q12 ) * ( q03 - q13 ) ) )
                + m112 * ( q01_2 - q00_10_2 - 2 * q01 * q11 + q11_2 - q02_12_2 + q03_13_2 )
                + m012 * ( q00_10_2 - q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ),
        -4 * ( a - q00 * q10 - q01 * q11 - q02 * q12 - q03 * q13 )
            * ( -2 * m003 * q00 * q01 + 2 * m103 * q00 * q01 - 2 * m023 * q01 * q02 + 2 * m123 * q01 * q02 + 2 * m023 * q00 * q03
                - 2 * m123 * q00 * q03 - 2 * m003 * q02 * q03 + 2 * m103 * q02 * q03 + 2 * m003 * q01 * q10
                - 2 * m103 * q01 * q10 - 2 * m023 * q03 * q10 + 2 * m123 * q03 * q10 + 2 * m003 * q00 * q11
                - 2 * m103 * q00 * q11 + 2 * m023 * q02 * q11 - 2 * m123 * q02 * q11 - 2 * m003 * q10 * q11 + 2 * m103 * q10 * q11
                + 2 * m023 * q01 * q12 - 2 * m123 * q01 * q12 + 2 * m003 * q03 * q12 - 2 * m103 * q03 * q12 - 2 * m023 * q11 * q12
                + 2 * m123 * q11 * q12 + m113 * ( q01_2 - q00_10_2 - 2 * q01 * q11 + q11_2 - q02_12_2 + q03_13_2 )
                - 2 * m023 * q00 * q13 + 2 * m123 * q00 * q13 + 2 * m003 * q02 * q13 - 2 * m103 * q02 * q13
                + 2 * m023 * q10 * q13 - 2 * m123 * q10 * q13 - 2 * m003 * q12 * q13 + 2 * m103 * q12 * q13
                + m013 * ( q00_10_2 - q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ) );
    c4[1] = SRNumeratorDerivativeTerm(
        ( b )
            * ( -2
                    * ( ( m020 - m120 ) * ( ( q01 - q11 ) * ( q02 - q12 ) - ( q00 - q10 ) * ( q03 - q13 ) )
                        + m100 * ( -( ( q00 - q10 ) * ( q01 - q11 ) ) - ( q02 - q12 ) * ( q03 - q13 ) )
                        + m000 * ( ( q00 - q10 ) * ( q01 - q11 ) + ( q02 - q12 ) * ( q03 - q13 ) ) )
                + m110 * ( q01_2 - q00_10_2 - 2 * q01 * q11 + q11_2 - q02_12_2 + q03_13_2 )
                + m010 * ( q00_10_2 - q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ),
        ( b )
            * ( -2
                    * ( ( m021 - m121 ) * ( ( q01 - q11 ) * ( q02 - q12 ) - ( q00 - q10 ) * ( q03 - q13 ) )
                        + m101 * ( -( ( q00 - q10 ) * ( q01 - q11 ) ) - ( q02 - q12 ) * ( q03 - q13 ) )
                        + m001 * ( ( q00 - q10 ) * ( q01 - q11 ) + ( q02 - q12 ) * ( q03 - q13 ) ) )
                + m111 * ( q01_2 - q00_10_2 - 2 * q01 * q11 + q11_2 - q02_12_2 + q03_13_2 )
                + m011 * ( q00_10_2 - q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ),
        ( b )
            * ( -2
                    * ( ( m022 - m122 ) * ( ( q01 - q11 ) * ( q02 - q12 ) - ( q00 - q10 ) * ( q03 - q13 ) )
                        + m102 * ( -( ( q00 - q10 ) * ( q01 - q11 ) ) - ( q02 - q12 ) * ( q03 - q13 ) )
                        + m002 * ( ( q00 - q10 ) * ( q01 - q11 ) + ( q02 - q12 ) * ( q03 - q13 ) ) )
                + m112 * ( q01_2 - q00_10_2 - 2 * q01 * q11 + q11_2 - q02_12_2 + q03_13_2 )
                + m012 * ( q00_10_2 - q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ),
        ( b )
            * ( -2 * m003 * q00 * q01 + 2 * m103 * q00 * q01 - 2 * m023 * q01 * q02 + 2 * m123 * q01 * q02 + 2 * m023 * q00 * q03
                - 2 * m123 * q00 * q03 - 2 * m003 * q02 * q03 + 2 * m103 * q02 * q03 + 2 * m003 * q01 * q10
                - 2 * m103 * q01 * q10 - 2 * m023 * q03 * q10 + 2 * m123 * q03 * q10 + 2 * m003 * q00 * q11
                - 2 * m103 * q00 * q11 + 2 * m023 * q02 * q11 - 2 * m123 * q02 * q11 - 2 * m003 * q10 * q11 + 2 * m103 * q10 * q11
                + 2 * m023 * q01 * q12 - 2 * m123 * q01 * q12 + 2 * m003 * q03 * q12 - 2 * m103 * q03 * q12 - 2 * m023 * q11 * q12
                + 2 * m123 * q11 * q12 + m113 * ( q01_2 - q00_10_2 - 2 * q01 * q11 + q11_2 - q02_12_2 + q03_13_2 )
                - 2 * m023 * q00 * q13 + 2 * m123 * q00 * q13 + 2 * m003 * q02 * q13 - 2 * m103 * q02 * q13
                + 2 * m023 * q10 * q13 - 2 * m123 * q10 * q13 - 2 * m003 * q12 * q13 + 2 * m103 * q12 * q13
                + m013 * ( q00_10_2 - q01_11_2 + ( q02 + q03 - q12 - q13 ) * ( q02 - q03 - q12 + q13 ) ) ) );

    // Z
    c0[2] = SRNumeratorDerivativeTerm(
        m120 * ( -q001_2 + q023_2 )
            + m020
                  * ( q00_4 + q01_4 - q023_2 - 4 * q00 * ( q02_2 + q03_2 ) * q10 - 4 * q01 * ( q02_2 + q03_2 ) * q11
                      + 4 * q01_2 * ( q02 * q12 + q03 * q13 ) + 2 * q00_2 * ( q01_2 + 2 * q02 * q12 + 2 * q03 * q13 ) )
            - 2
                  * ( m010 * q00_2 * q01 * q02 - m110 * q00_2 * q01 * q02 + m010 * q01_3 * q02 - m110 * q01_3 * q02
                      + m010 * q01 * q02_3 - m110 * q01 * q02_3 + m010 * q00_3 * q03 - m110 * q00_3 * q03
                      + m010 * q00 * q01_2 * q03 - m110 * q00 * q01_2 * q03 + m010 * q00 * q02_2 * q03 - m110 * q00 * q02_2 * q03
                      + m010 * q01 * q02 * q03_2 - m110 * q01 * q02 * q03_2 + m010 * q00 * q03_3 - m110 * q00 * q03_3
                      - m100 * ( q00 * q02 - q01 * q03 ) * ( a ) + 2 * m010 * q00 * q01 * q02 * q10 + m010 * q00_2 * q03 * q10
                      - m010 * q01_2 * q03 * q10 - m010 * q02_2 * q03 * q10 - m010 * q03_3 * q10 - m010 * q00_2 * q02 * q11
                      + m010 * q01_2 * q02 * q11 - m010 * q02_3 * q11 + 2 * m010 * q00 * q01 * q03 * q11 - m010 * q02 * q03_2 * q11
                      - m010 * q00_2 * q01 * q12 - m010 * q01_3 * q12 + m010 * q01 * q02_2 * q12 + 2 * m010 * q00 * q02 * q03 * q12
                      - m010 * q01 * q03_2 * q12 - m010 * ( q00_3 - 2 * q01 * q02 * q03 + q00 * ( q01_2 + q02_2 - q03_2 ) ) * q13
                      + m000
                            * ( -( ( q02_2 + q03_2 ) * ( q02 * q10 - q03 * q11 ) ) - q01_2 * ( q02 * q10 + q03 * q11 )
                                + q00_3 * ( q02 - q12 ) + q01_3 * ( -q03 + q13 ) + q00_2 * ( q02 * q10 + q03 * q11 + q01 * ( -q03 + q13 ) )
                                - q01 * ( 2 * q02 * q03 * q12 + q02_2 * ( q03 - q13 ) + q03_2 * ( q03 + q13 ) )
                                + q00
                                      * ( q02_3 + q01 * ( -2 * q03 * q10 + 2 * q02 * q11 ) + q01_2 * ( q02 - q12 )
                                          + q02_2 * q12 - q03_2 * q12 + q02 * q03 * ( q03 + 2 * q13 ) ) ) ),
        m121 * ( -q001_2 + q023_2 )
            + m021
                  * ( q00_4 + q01_4 - q023_2 - 4 * q00 * ( q02_2 + q03_2 ) * q10 - 4 * q01 * ( q02_2 + q03_2 ) * q11
                      + 4 * q01_2 * ( q02 * q12 + q03 * q13 ) + 2 * q00_2 * ( q01_2 + 2 * q02 * q12 + 2 * q03 * q13 ) )
            - 2
                  * ( m011 * q00_2 * q01 * q02 - m111 * q00_2 * q01 * q02 + m011 * q01_3 * q02 - m111 * q01_3 * q02
                      + m011 * q01 * q02_3 - m111 * q01 * q02_3 + m011 * q00_3 * q03 - m111 * q00_3 * q03
                      + m011 * q00 * q01_2 * q03 - m111 * q00 * q01_2 * q03 + m011 * q00 * q02_2 * q03 - m111 * q00 * q02_2 * q03
                      + m011 * q01 * q02 * q03_2 - m111 * q01 * q02 * q03_2 + m011 * q00 * q03_3 - m111 * q00 * q03_3
                      - m101 * ( q00 * q02 - q01 * q03 ) * ( a ) + 2 * m011 * q00 * q01 * q02 * q10 + m011 * q00_2 * q03 * q10
                      - m011 * q01_2 * q03 * q10 - m011 * q02_2 * q03 * q10 - m011 * q03_3 * q10 - m011 * q00_2 * q02 * q11
                      + m011 * q01_2 * q02 * q11 - m011 * q02_3 * q11 + 2 * m011 * q00 * q01 * q03 * q11 - m011 * q02 * q03_2 * q11
                      - m011 * q00_2 * q01 * q12 - m011 * q01_3 * q12 + m011 * q01 * q02_2 * q12 + 2 * m011 * q00 * q02 * q03 * q12
                      - m011 * q01 * q03_2 * q12 - m011 * ( q00_3 - 2 * q01 * q02 * q03 + q00 * ( q01_2 + q02_2 - q03_2 ) ) * q13
                      + m001
                            * ( -( ( q02_2 + q03_2 ) * ( q02 * q10 - q03 * q11 ) ) - q01_2 * ( q02 * q10 + q03 * q11 )
                                + q00_3 * ( q02 - q12 ) + q01_3 * ( -q03 + q13 ) + q00_2 * ( q02 * q10 + q03 * q11 + q01 * ( -q03 + q13 ) )
                                - q01 * ( 2 * q02 * q03 * q12 + q02_2 * ( q03 - q13 ) + q03_2 * ( q03 + q13 ) )
                                + q00 * ( q02_3 + q01 * ( -2 * q03 * q10 + 2 * q02 * q11 ) + q01_2 * ( q02 - q12 ) + q02_2 * q12 - q03_2 * q12 + q02 * q03 * ( q03 + 2 * q13 ) ) ) ),
        m122 * ( -q001_2 + q023_2 )
            + m022
                  * ( q00_4 + q01_4 - q023_2 - 4 * q00 * ( q02_2 + q03_2 ) * q10 - 4 * q01 * ( q02_2 + q03_2 ) * q11
                      + 4 * q01_2 * ( q02 * q12 + q03 * q13 ) + 2 * q00_2 * ( q01_2 + 2 * q02 * q12 + 2 * q03 * q13 ) )
            - 2
                  * ( m012 * q00_2 * q01 * q02 - m112 * q00_2 * q01 * q02 + m012 * q01_3 * q02 - m112 * q01_3 * q02
                      + m012 * q01 * q02_3 - m112 * q01 * q02_3 + m012 * q00_3 * q03 - m112 * q00_3 * q03
                      + m012 * q00 * q01_2 * q03 - m112 * q00 * q01_2 * q03 + m012 * q00 * q02_2 * q03 - m112 * q00 * q02_2 * q03
                      + m012 * q01 * q02 * q03_2 - m112 * q01 * q02 * q03_2 + m012 * q00 * q03_3 - m112 * q00 * q03_3
                      - m102 * ( q00 * q02 - q01 * q03 ) * ( a ) + 2 * m012 * q00 * q01 * q02 * q10 + m012 * q00_2 * q03 * q10
                      - m012 * q01_2 * q03 * q10 - m012 * q02_2 * q03 * q10 - m012 * q03_3 * q10 - m012 * q00_2 * q02 * q11
                      + m012 * q01_2 * q02 * q11 - m012 * q02_3 * q11 + 2 * m012 * q00 * q01 * q03 * q11 - m012 * q02 * q03_2 * q11
                      - m012 * q00_2 * q01 * q12 - m012 * q01_3 * q12 + m012 * q01 * q02_2 * q12 + 2 * m012 * q00 * q02 * q03 * q12
                      - m012 * q01 * q03_2 * q12 - m012 * ( q00_3 - 2 * q01 * q02 * q03 + q00 * ( q01_2 + q02_2 - q03_2 ) ) * q13
                      + m002
                            * ( -( ( q02_2 + q03_2 ) * ( q02 * q10 - q03 * q11 ) ) - q01_2 * ( q02 * q10 + q03 * q11 )
                                + q00_3 * ( q02 - q12 ) + q01_3 * ( -q03 + q13 ) + q00_2 * ( q02 * q10 + q03 * q11 + q01 * ( -q03 + q13 ) )
                                - q01 * ( 2 * q02 * q03 * q12 + q02_2 * ( q03 - q13 ) + q03_2 * ( q03 + q13 ) )
                                + q00 * ( q02_3 + q01 * ( -2 * q03 * q10 + 2 * q02 * q11 ) + q01_2 * ( q02 - q12 ) + q02_2 * q12 - q03_2 * q12 + q02 * q03 * ( q03 + 2 * q13 ) ) ) ),
        -2 * m003 * q00_3 * q02 + 2 * m103 * q00_3 * q02 - 2 * m013 * q00_2 * q01 * q02 + 2 * m113 * q00_2 * q01 * q02
            - 2 * m003 * q00 * q01_2 * q02 + 2 * m103 * q00 * q01_2 * q02 - 2 * m013 * q01_3 * q02
            + 2 * m113 * q01_3 * q02 - 2 * m003 * q00 * q02_3 + 2 * m103 * q00 * q02_3 - 2 * m013 * q01 * q02_3
            + 2 * m113 * q01 * q02_3 - 2 * m013 * q00_3 * q03 + 2 * m113 * q00_3 * q03 + 2 * m003 * q00_2 * q01 * q03
            - 2 * m103 * q00_2 * q01 * q03 - 2 * m013 * q00 * q01_2 * q03 + 2 * m113 * q00 * q01_2 * q03
            + 2 * m003 * q01_3 * q03 - 2 * m103 * q01_3 * q03 - 2 * m013 * q00 * q02_2 * q03 + 2 * m113 * q00 * q02_2 * q03
            + 2 * m003 * q01 * q02_2 * q03 - 2 * m103 * q01 * q02_2 * q03 - 2 * m003 * q00 * q02 * q03_2
            + 2 * m103 * q00 * q02 * q03_2 - 2 * m013 * q01 * q02 * q03_2 + 2 * m113 * q01 * q02 * q03_2
            - 2 * m013 * q00 * q03_3 + 2 * m113 * q00 * q03_3 + 2 * m003 * q01 * q03_3 - 2 * m103 * q01 * q03_3
            - m123 * ( q00_2 + q01_2 - q02_2 - q03_2 ) * (a)-2 * m003 * q00_2 * q02 * q10 - 4 * m013 * q00 * q01 * q02 * q10
            + 2 * m003 * q01_2 * q02 * q10 + 2 * m003 * q02_3 * q10 - 2 * m013 * q00_2 * q03 * q10
            + 4 * m003 * q00 * q01 * q03 * q10 + 2 * m013 * q01_2 * q03 * q10 + 2 * m013 * q02_2 * q03 * q10
            + 2 * m003 * q02 * q03_2 * q10 + 2 * m013 * q03_3 * q10 + 2 * m013 * q00_2 * q02 * q11
            - 4 * m003 * q00 * q01 * q02 * q11 - 2 * m013 * q01_2 * q02 * q11 + 2 * m013 * q02_3 * q11
            - 2 * m003 * q00_2 * q03 * q11 - 4 * m013 * q00 * q01 * q03 * q11 + 2 * m003 * q01_2 * q03 * q11
            - 2 * m003 * q02_2 * q03 * q11 + 2 * m013 * q02 * q03_2 * q11 - 2 * m003 * q03_3 * q11
            + 2 * m003 * q00_3 * q12 + 2 * m013 * q00_2 * q01 * q12 + 2 * m003 * q00 * q01_2 * q12 + 2 * m013 * q01_3 * q12
            - 2 * m003 * q00 * q02_2 * q12 - 2 * m013 * q01 * q02_2 * q12 - 4 * m013 * q00 * q02 * q03 * q12
            + 4 * m003 * q01 * q02 * q03 * q12 + 2 * m003 * q00 * q03_2 * q12 + 2 * m013 * q01 * q03_2 * q12
            + 2 * m013 * q00_3 * q13 - 2 * m003 * q00_2 * q01 * q13 + 2 * m013 * q00 * q01_2 * q13 - 2 * m003 * q01_3 * q13
            + 2 * m013 * q00 * q02_2 * q13 - 2 * m003 * q01 * q02_2 * q13 - 4 * m003 * q00 * q02 * q03 * q13
            - 4 * m013 * q01 * q02 * q03 * q13 - 2 * m013 * q00 * q03_2 * q13 + 2 * m003 * q01 * q03_2 * q13
            + m023
                  * ( q00_4 - q023_2 - 4 * q00 * ( q02_2 + q03_2 ) * q10 + 2 * q00_2 * ( q01_2 + 2 * q02 * q12 + 2 * q03 * q13 )
                      + q01 * ( q01_3 - 4 * ( q02_2 + q03_2 ) * q11 + 4 * q01 * ( q02 * q12 + q03 * q13 ) ) ) );
    c1[2] = SRNumeratorDerivativeTerm(
        4
            * ( 2 * m000 * q00_3 * q02 - 2 * m100 * q00_3 * q02 + 2 * m010 * q00_2 * q01 * q02 - 2 * m110 * q00_2 * q01 * q02
                + 2 * m000 * q00 * q01_2 * q02 - 2 * m100 * q00 * q01_2 * q02 + 2 * m010 * q01_3 * q02 - 2 * m110 * q01_3 * q02
                + 2 * m000 * q00 * q02_3 - 2 * m100 * q00 * q02_3 + 2 * m010 * q01 * q02_3 - 2 * m110 * q01 * q02_3
                + 2 * m010 * q00_3 * q03 - 2 * m110 * q00_3 * q03 - 2 * m000 * q00_2 * q01 * q03 + 2 * m100 * q00_2 * q01 * q03
                + 2 * m010 * q00 * q01_2 * q03 - 2 * m110 * q00 * q01_2 * q03 - 2 * m000 * q01_3 * q03 + 2 * m100 * q01_3 * q03
                + 2 * m010 * q00 * q02_2 * q03 - 2 * m110 * q00 * q02_2 * q03 - 2 * m000 * q01 * q02_2 * q03
                + 2 * m100 * q01 * q02_2 * q03 + 2 * m000 * q00 * q02 * q03_2 - 2 * m100 * q00 * q02 * q03_2
                + 2 * m010 * q01 * q02 * q03_2 - 2 * m110 * q01 * q02 * q03_2 + 2 * m010 * q00 * q03_3 - 2 * m110 * q00 * q03_3
                - 2 * m000 * q01 * q03_3 + 2 * m100 * q01 * q03_3 + m100 * q00_2 * q02 * q10 + 2 * m010 * q00 * q01 * q02 * q10
                - 2 * m000 * q01_2 * q02 * q10 + m100 * q01_2 * q02 * q10 - 2 * m000 * q02_3 * q10 + m100 * q02_3 * q10
                + m110 * q00_2 * q03 * q10 - 2 * m000 * q00 * q01 * q03 * q10 - 2 * m010 * q01_2 * q03 * q10
                + m110 * q01_2 * q03 * q10 - 2 * m010 * q02_2 * q03 * q10 + m110 * q02_2 * q03 * q10
                - 2 * m000 * q02 * q03_2 * q10 + m100 * q02 * q03_2 * q10 - 2 * m010 * q03_3 * q10 + m110 * q03_3 * q10
                - m000 * q00 * q02 * q10_2 - m010 * q01 * q02 * q10_2 - m010 * q00 * q03 * q10_2 + m000 * q01 * q03 * q10_2
                - 2 * m010 * q00_2 * q02 * q11 + m110 * q00_2 * q02 * q11 + 2 * m000 * q00 * q01 * q02 * q11
                + m110 * q01_2 * q02 * q11 - 2 * m010 * q02_3 * q11 + m110 * q02_3 * q11 + 2 * m000 * q00_2 * q03 * q11
                - m100 * q00_2 * q03 * q11 + 2 * m010 * q00 * q01 * q03 * q11 - m100 * q01_2 * q03 * q11
                + 2 * m000 * q02_2 * q03 * q11 - m100 * q02_2 * q03 * q11 - 2 * m010 * q02 * q03_2 * q11
                + m110 * q02 * q03_2 * q11 + 2 * m000 * q03_3 * q11 - m100 * q03_3 * q11 - m000 * q00 * q02 * q11_2
                - m010 * q01 * q02 * q11_2 - m010 * q00 * q03 * q11_2 + m000 * q01 * q03 * q11_2 - 2 * m000 * q00_3 * q12
                + m100 * q00_3 * q12 - 2 * m010 * q00_2 * q01 * q12 + m110 * q00_2 * q01 * q12 - 2 * m000 * q00 * q01_2 * q12
                + m100 * q00 * q01_2 * q12 - 2 * m010 * q01_3 * q12 + m110 * q01_3 * q12 + m100 * q00 * q02_2 * q12
                + m110 * q01 * q02_2 * q12 + 2 * m010 * q00 * q02 * q03 * q12 - 2 * m000 * q01 * q02 * q03 * q12
                - 2 * m000 * q00 * q03_2 * q12 + m100 * q00 * q03_2 * q12 - 2 * m010 * q01 * q03_2 * q12 + m110 * q01 * q03_2 * q12
                + m000 * q00_2 * q10 * q12 + m000 * q01_2 * q10 * q12 + m000 * q02_2 * q10 * q12 + m000 * q03_2 * q10 * q12
                + m010 * q00_2 * q11 * q12 + m010 * q01_2 * q11 * q12 + m010 * q02_2 * q11 * q12 + m010 * q03_2 * q11 * q12
                - m000 * q00 * q02 * q12_2 - m010 * q01 * q02 * q12_2 - m010 * q00 * q03 * q12_2 + m000 * q01 * q03 * q12_2
                + ( 2 * m000 * q00_2 * q01 - m100 * q00_2 * q01 + 2 * m000 * q01_3 - m100 * q01_3 + 2 * m000 * q01 * q02_2
                    - m100 * q01 * q02_2 + 2 * m000 * q00 * q02 * q03 - m100 * q01 * q03_2 + m110 * q00 * ( a )
                    + m010 * ( -2 * q00 * ( q00_2 + q01_2 + q02_2 ) + 2 * q01 * q02 * q03 + (a)*q10 ) - m000 * (a)*q11 )
                      * q13
                - ( m000 * q00 * q02 + m010 * q01 * q02 + m010 * q00 * q03 - m000 * q01 * q03 ) * q13_2
                + m120 * ( a ) * ( q00_2 + q01_2 - q00 * q10 - q01 * q11 + q02 * ( -q02 + q12 ) + q03 * ( -q03 + q13 ) )
                + m020
                      * ( -q00_4 - q01_4 + q023_2 + q00_3 * q10 + q00 * ( q01_2 + 3 * ( q02_2 + q03_2 ) ) * q10 + q01_3 * q11
                          + 3 * q01 * ( q02_2 + q03_2 ) * q11 - ( q02_2 + q03_2 ) * ( q10_2 + q11_2 + q02 * q12 + q03 * q13 )
                          + q00_2 * ( q01 * ( -2 * q01 + q11 ) - 3 * q02 * q12 + q12_2 - 3 * q03 * q13 + q13_2 )
                          + q01_2 * ( -3 * q02 * q12 + q12_2 + q13 * ( -3 * q03 + q13 ) ) ) ),
        4
            * ( 2 * m001 * q00_3 * q02 - 2 * m101 * q00_3 * q02 + 2 * m011 * q00_2 * q01 * q02 - 2 * m111 * q00_2 * q01 * q02
                + 2 * m001 * q00 * q01_2 * q02 - 2 * m101 * q00 * q01_2 * q02 + 2 * m011 * q01_3 * q02 - 2 * m111 * q01_3 * q02
                + 2 * m001 * q00 * q02_3 - 2 * m101 * q00 * q02_3 + 2 * m011 * q01 * q02_3 - 2 * m111 * q01 * q02_3
                + 2 * m011 * q00_3 * q03 - 2 * m111 * q00_3 * q03 - 2 * m001 * q00_2 * q01 * q03 + 2 * m101 * q00_2 * q01 * q03
                + 2 * m011 * q00 * q01_2 * q03 - 2 * m111 * q00 * q01_2 * q03 - 2 * m001 * q01_3 * q03 + 2 * m101 * q01_3 * q03
                + 2 * m011 * q00 * q02_2 * q03 - 2 * m111 * q00 * q02_2 * q03 - 2 * m001 * q01 * q02_2 * q03
                + 2 * m101 * q01 * q02_2 * q03 + 2 * m001 * q00 * q02 * q03_2 - 2 * m101 * q00 * q02 * q03_2
                + 2 * m011 * q01 * q02 * q03_2 - 2 * m111 * q01 * q02 * q03_2 + 2 * m011 * q00 * q03_3 - 2 * m111 * q00 * q03_3
                - 2 * m001 * q01 * q03_3 + 2 * m101 * q01 * q03_3 + m101 * q00_2 * q02 * q10 + 2 * m011 * q00 * q01 * q02 * q10
                - 2 * m001 * q01_2 * q02 * q10 + m101 * q01_2 * q02 * q10 - 2 * m001 * q02_3 * q10 + m101 * q02_3 * q10
                + m111 * q00_2 * q03 * q10 - 2 * m001 * q00 * q01 * q03 * q10 - 2 * m011 * q01_2 * q03 * q10
                + m111 * q01_2 * q03 * q10 - 2 * m011 * q02_2 * q03 * q10 + m111 * q02_2 * q03 * q10
                - 2 * m001 * q02 * q03_2 * q10 + m101 * q02 * q03_2 * q10 - 2 * m011 * q03_3 * q10 + m111 * q03_3 * q10
                - m001 * q00 * q02 * q10_2 - m011 * q01 * q02 * q10_2 - m011 * q00 * q03 * q10_2 + m001 * q01 * q03 * q10_2
                - 2 * m011 * q00_2 * q02 * q11 + m111 * q00_2 * q02 * q11 + 2 * m001 * q00 * q01 * q02 * q11
                + m111 * q01_2 * q02 * q11 - 2 * m011 * q02_3 * q11 + m111 * q02_3 * q11 + 2 * m001 * q00_2 * q03 * q11
                - m101 * q00_2 * q03 * q11 + 2 * m011 * q00 * q01 * q03 * q11 - m101 * q01_2 * q03 * q11
                + 2 * m001 * q02_2 * q03 * q11 - m101 * q02_2 * q03 * q11 - 2 * m011 * q02 * q03_2 * q11
                + m111 * q02 * q03_2 * q11 + 2 * m001 * q03_3 * q11 - m101 * q03_3 * q11 - m001 * q00 * q02 * q11_2
                - m011 * q01 * q02 * q11_2 - m011 * q00 * q03 * q11_2 + m001 * q01 * q03 * q11_2 - 2 * m001 * q00_3 * q12
                + m101 * q00_3 * q12 - 2 * m011 * q00_2 * q01 * q12 + m111 * q00_2 * q01 * q12 - 2 * m001 * q00 * q01_2 * q12
                + m101 * q00 * q01_2 * q12 - 2 * m011 * q01_3 * q12 + m111 * q01_3 * q12 + m101 * q00 * q02_2 * q12
                + m111 * q01 * q02_2 * q12 + 2 * m011 * q00 * q02 * q03 * q12 - 2 * m001 * q01 * q02 * q03 * q12
                - 2 * m001 * q00 * q03_2 * q12 + m101 * q00 * q03_2 * q12 - 2 * m011 * q01 * q03_2 * q12 + m111 * q01 * q03_2 * q12
                + m001 * q00_2 * q10 * q12 + m001 * q01_2 * q10 * q12 + m001 * q02_2 * q10 * q12 + m001 * q03_2 * q10 * q12
                + m011 * q00_2 * q11 * q12 + m011 * q01_2 * q11 * q12 + m011 * q02_2 * q11 * q12 + m011 * q03_2 * q11 * q12
                - m001 * q00 * q02 * q12_2 - m011 * q01 * q02 * q12_2 - m011 * q00 * q03 * q12_2 + m001 * q01 * q03 * q12_2
                + ( 2 * m001 * q00_2 * q01 - m101 * q00_2 * q01 + 2 * m001 * q01_3 - m101 * q01_3 + 2 * m001 * q01 * q02_2
                    - m101 * q01 * q02_2 + 2 * m001 * q00 * q02 * q03 - m101 * q01 * q03_2 + m111 * q00 * ( a )
                    + m011 * ( -2 * q00 * ( q00_2 + q01_2 + q02_2 ) + 2 * q01 * q02 * q03 + (a)*q10 ) - m001 * (a)*q11 )
                      * q13
                - ( m001 * q00 * q02 + m011 * q01 * q02 + m011 * q00 * q03 - m001 * q01 * q03 ) * q13_2
                + m121 * ( a ) * ( q00_2 + q01_2 - q00 * q10 - q01 * q11 + q02 * ( -q02 + q12 ) + q03 * ( -q03 + q13 ) )
                + m021
                      * ( -q00_4 - q01_4 + q023_2 + q00_3 * q10 + q00 * ( q01_2 + 3 * ( q02_2 + q03_2 ) ) * q10 + q01_3 * q11
                          + 3 * q01 * ( q02_2 + q03_2 ) * q11 - ( q02_2 + q03_2 ) * ( q10_2 + q11_2 + q02 * q12 + q03 * q13 )
                          + q00_2 * ( q01 * ( -2 * q01 + q11 ) - 3 * q02 * q12 + q12_2 - 3 * q03 * q13 + q13_2 )
                          + q01_2 * ( -3 * q02 * q12 + q12_2 + q13 * ( -3 * q03 + q13 ) ) ) ),
        4
            * ( 2 * m002 * q00_3 * q02 - 2 * m102 * q00_3 * q02 + 2 * m012 * q00_2 * q01 * q02 - 2 * m112 * q00_2 * q01 * q02
                + 2 * m002 * q00 * q01_2 * q02 - 2 * m102 * q00 * q01_2 * q02 + 2 * m012 * q01_3 * q02 - 2 * m112 * q01_3 * q02
                + 2 * m002 * q00 * q02_3 - 2 * m102 * q00 * q02_3 + 2 * m012 * q01 * q02_3 - 2 * m112 * q01 * q02_3
                + 2 * m012 * q00_3 * q03 - 2 * m112 * q00_3 * q03 - 2 * m002 * q00_2 * q01 * q03 + 2 * m102 * q00_2 * q01 * q03
                + 2 * m012 * q00 * q01_2 * q03 - 2 * m112 * q00 * q01_2 * q03 - 2 * m002 * q01_3 * q03 + 2 * m102 * q01_3 * q03
                + 2 * m012 * q00 * q02_2 * q03 - 2 * m112 * q00 * q02_2 * q03 - 2 * m002 * q01 * q02_2 * q03
                + 2 * m102 * q01 * q02_2 * q03 + 2 * m002 * q00 * q02 * q03_2 - 2 * m102 * q00 * q02 * q03_2
                + 2 * m012 * q01 * q02 * q03_2 - 2 * m112 * q01 * q02 * q03_2 + 2 * m012 * q00 * q03_3 - 2 * m112 * q00 * q03_3
                - 2 * m002 * q01 * q03_3 + 2 * m102 * q01 * q03_3 + m102 * q00_2 * q02 * q10 + 2 * m012 * q00 * q01 * q02 * q10
                - 2 * m002 * q01_2 * q02 * q10 + m102 * q01_2 * q02 * q10 - 2 * m002 * q02_3 * q10 + m102 * q02_3 * q10
                + m112 * q00_2 * q03 * q10 - 2 * m002 * q00 * q01 * q03 * q10 - 2 * m012 * q01_2 * q03 * q10
                + m112 * q01_2 * q03 * q10 - 2 * m012 * q02_2 * q03 * q10 + m112 * q02_2 * q03 * q10
                - 2 * m002 * q02 * q03_2 * q10 + m102 * q02 * q03_2 * q10 - 2 * m012 * q03_3 * q10 + m112 * q03_3 * q10
                - m002 * q00 * q02 * q10_2 - m012 * q01 * q02 * q10_2 - m012 * q00 * q03 * q10_2 + m002 * q01 * q03 * q10_2
                - 2 * m012 * q00_2 * q02 * q11 + m112 * q00_2 * q02 * q11 + 2 * m002 * q00 * q01 * q02 * q11
                + m112 * q01_2 * q02 * q11 - 2 * m012 * q02_3 * q11 + m112 * q02_3 * q11 + 2 * m002 * q00_2 * q03 * q11
                - m102 * q00_2 * q03 * q11 + 2 * m012 * q00 * q01 * q03 * q11 - m102 * q01_2 * q03 * q11
                + 2 * m002 * q02_2 * q03 * q11 - m102 * q02_2 * q03 * q11 - 2 * m012 * q02 * q03_2 * q11
                + m112 * q02 * q03_2 * q11 + 2 * m002 * q03_3 * q11 - m102 * q03_3 * q11 - m002 * q00 * q02 * q11_2
                - m012 * q01 * q02 * q11_2 - m012 * q00 * q03 * q11_2 + m002 * q01 * q03 * q11_2 - 2 * m002 * q00_3 * q12
                + m102 * q00_3 * q12 - 2 * m012 * q00_2 * q01 * q12 + m112 * q00_2 * q01 * q12 - 2 * m002 * q00 * q01_2 * q12
                + m102 * q00 * q01_2 * q12 - 2 * m012 * q01_3 * q12 + m112 * q01_3 * q12 + m102 * q00 * q02_2 * q12
                + m112 * q01 * q02_2 * q12 + 2 * m012 * q00 * q02 * q03 * q12 - 2 * m002 * q01 * q02 * q03 * q12
                - 2 * m002 * q00 * q03_2 * q12 + m102 * q00 * q03_2 * q12 - 2 * m012 * q01 * q03_2 * q12 + m112 * q01 * q03_2 * q12
                + m002 * q00_2 * q10 * q12 + m002 * q01_2 * q10 * q12 + m002 * q02_2 * q10 * q12 + m002 * q03_2 * q10 * q12
                + m012 * q00_2 * q11 * q12 + m012 * q01_2 * q11 * q12 + m012 * q02_2 * q11 * q12 + m012 * q03_2 * q11 * q12
                - m002 * q00 * q02 * q12_2 - m012 * q01 * q02 * q12_2 - m012 * q00 * q03 * q12_2 + m002 * q01 * q03 * q12_2
                + ( 2 * m002 * q00_2 * q01 - m102 * q00_2 * q01 + 2 * m002 * q01_3 - m102 * q01_3 + 2 * m002 * q01 * q02_2
                    - m102 * q01 * q02_2 + 2 * m002 * q00 * q02 * q03 - m102 * q01 * q03_2 + m112 * q00 * ( a )
                    + m012 * ( -2 * q00 * ( q00_2 + q01_2 + q02_2 ) + 2 * q01 * q02 * q03 + (a)*q10 ) - m002 * (a)*q11 )
                      * q13
                - ( m002 * q00 * q02 + m012 * q01 * q02 + m012 * q00 * q03 - m002 * q01 * q03 ) * q13_2
                + m122 * ( a ) * ( q00_2 + q01_2 - q00 * q10 - q01 * q11 + q02 * ( -q02 + q12 ) + q03 * ( -q03 + q13 ) )
                + m022
                      * ( -q00_4 - q01_4 + q023_2 + q00_3 * q10 + q00 * ( q01_2 + 3 * ( q02_2 + q03_2 ) ) * q10 + q01_3 * q11
                          + 3 * q01 * ( q02_2 + q03_2 ) * q11 - ( q02_2 + q03_2 ) * ( q10_2 + q11_2 + q02 * q12 + q03 * q13 )
                          + q00_2 * ( q01 * ( -2 * q01 + q11 ) - 3 * q02 * q12 + q12_2 - 3 * q03 * q13 + q13_2 )
                          + q01_2 * ( -3 * q02 * q12 + q12_2 + q13 * ( -3 * q03 + q13 ) ) ) ),
        4
            * ( 2 * m003 * q00_3 * q02 - 2 * m103 * q00_3 * q02 + 2 * m013 * q00_2 * q01 * q02 - 2 * m113 * q00_2 * q01 * q02
                + 2 * m003 * q00 * q01_2 * q02 - 2 * m103 * q00 * q01_2 * q02 + 2 * m013 * q01_3 * q02 - 2 * m113 * q01_3 * q02
                + 2 * m003 * q00 * q02_3 - 2 * m103 * q00 * q02_3 + 2 * m013 * q01 * q02_3 - 2 * m113 * q01 * q02_3
                + 2 * m013 * q00_3 * q03 - 2 * m113 * q00_3 * q03 - 2 * m003 * q00_2 * q01 * q03 + 2 * m103 * q00_2 * q01 * q03
                + 2 * m013 * q00 * q01_2 * q03 - 2 * m113 * q00 * q01_2 * q03 - 2 * m003 * q01_3 * q03 + 2 * m103 * q01_3 * q03
                + 2 * m013 * q00 * q02_2 * q03 - 2 * m113 * q00 * q02_2 * q03 - 2 * m003 * q01 * q02_2 * q03
                + 2 * m103 * q01 * q02_2 * q03 + 2 * m003 * q00 * q02 * q03_2 - 2 * m103 * q00 * q02 * q03_2
                + 2 * m013 * q01 * q02 * q03_2 - 2 * m113 * q01 * q02 * q03_2 + 2 * m013 * q00 * q03_3 - 2 * m113 * q00 * q03_3
                - 2 * m003 * q01 * q03_3 + 2 * m103 * q01 * q03_3 + m103 * q00_2 * q02 * q10 + 2 * m013 * q00 * q01 * q02 * q10
                - 2 * m003 * q01_2 * q02 * q10 + m103 * q01_2 * q02 * q10 - 2 * m003 * q02_3 * q10 + m103 * q02_3 * q10
                + m113 * q00_2 * q03 * q10 - 2 * m003 * q00 * q01 * q03 * q10 - 2 * m013 * q01_2 * q03 * q10
                + m113 * q01_2 * q03 * q10 - 2 * m013 * q02_2 * q03 * q10 + m113 * q02_2 * q03 * q10 - 2 * m003 * q02 * q03_2 * q10
                + m103 * q02 * q03_2 * q10 - 2 * m013 * q03_3 * q10 + m113 * q03_3 * q10 - m003 * q00 * q02 * q10_2
                - m013 * q01 * q02 * q10_2 - m013 * q00 * q03 * q10_2 + m003 * q01 * q03 * q10_2 - 2 * m013 * q00_2 * q02 * q11
                + m113 * q00_2 * q02 * q11 + 2 * m003 * q00 * q01 * q02 * q11 + m113 * q01_2 * q02 * q11
                - 2 * m013 * q02_3 * q11 + m113 * q02_3 * q11 + 2 * m003 * q00_2 * q03 * q11 - m103 * q00_2 * q03 * q11
                + 2 * m013 * q00 * q01 * q03 * q11 - m103 * q01_2 * q03 * q11 + 2 * m003 * q02_2 * q03 * q11
                - m103 * q02_2 * q03 * q11 - 2 * m013 * q02 * q03_2 * q11 + m113 * q02 * q03_2 * q11 + 2 * m003 * q03_3 * q11
                - m103 * q03_3 * q11 - m003 * q00 * q02 * q11_2 - m013 * q01 * q02 * q11_2 - m013 * q00 * q03 * q11_2
                + m003 * q01 * q03 * q11_2 - 2 * m003 * q00_3 * q12 + m103 * q00_3 * q12 - 2 * m013 * q00_2 * q01 * q12
                + m113 * q00_2 * q01 * q12 - 2 * m003 * q00 * q01_2 * q12 + m103 * q00 * q01_2 * q12 - 2 * m013 * q01_3 * q12
                + m113 * q01_3 * q12 + m103 * q00 * q02_2 * q12 + m113 * q01 * q02_2 * q12 + 2 * m013 * q00 * q02 * q03 * q12
                - 2 * m003 * q01 * q02 * q03 * q12 - 2 * m003 * q00 * q03_2 * q12 + m103 * q00 * q03_2 * q12
                - 2 * m013 * q01 * q03_2 * q12 + m113 * q01 * q03_2 * q12 + m003 * q00_2 * q10 * q12 + m003 * q01_2 * q10 * q12
                + m003 * q02_2 * q10 * q12 + m003 * q03_2 * q10 * q12 + m013 * q00_2 * q11 * q12 + m013 * q01_2 * q11 * q12
                + m013 * q02_2 * q11 * q12 + m013 * q03_2 * q11 * q12 - m003 * q00 * q02 * q12_2 - m013 * q01 * q02 * q12_2
                - m013 * q00 * q03 * q12_2 + m003 * q01 * q03 * q12_2 - 2 * m013 * q00_3 * q13 + m113 * q00_3 * q13
                + 2 * m003 * q00_2 * q01 * q13 - m103 * q00_2 * q01 * q13 - 2 * m013 * q00 * q01_2 * q13 + m113 * q00 * q01_2 * q13
                + 2 * m003 * q01_3 * q13 - m103 * q01_3 * q13 - 2 * m013 * q00 * q02_2 * q13 + m113 * q00 * q02_2 * q13
                + 2 * m003 * q01 * q02_2 * q13 - m103 * q01 * q02_2 * q13 + 2 * m003 * q00 * q02 * q03 * q13
                + 2 * m013 * q01 * q02 * q03 * q13 + m113 * q00 * q03_2 * q13 - m103 * q01 * q03_2 * q13
                + m013 * q00_2 * q10 * q13 + m013 * q01_2 * q10 * q13 + m013 * q02_2 * q10 * q13 + m013 * q03_2 * q10 * q13
                - m003 * q00_2 * q11 * q13 - m003 * q01_2 * q11 * q13 - m003 * q02_2 * q11 * q13 - m003 * q03_2 * q11 * q13
                - m003 * q00 * q02 * q13_2 - m013 * q01 * q02 * q13_2 - m013 * q00 * q03 * q13_2 + m003 * q01 * q03 * q13_2
                + m123 * ( a ) * ( q00_2 + q01_2 - q00 * q10 - q01 * q11 + q02 * ( -q02 + q12 ) + q03 * ( -q03 + q13 ) )
                + m023
                      * ( -q00_4 - q01_4 + q023_2 + q00_3 * q10 + q00 * ( q01_2 + 3 * ( q02_2 + q03_2 ) ) * q10 + q01_3 * q11
                          + 3 * q01 * ( q02_2 + q03_2 ) * q11 - ( q02_2 + q03_2 ) * ( q10_2 + q11_2 + q02 * q12 + q03 * q13 )
                          + q00_2 * ( q01 * ( -2 * q01 + q11 ) - 3 * q02 * q12 + q12_2 - 3 * q03 * q13 + q13_2 )
                          + q01_2 * ( -3 * q02 * q12 + q12_2 + q13 * ( -3 * q03 + q13 ) ) ) ) );
    c2[2] = SRNumeratorDerivativeTerm(
        -2
            * ( 6 * m000 * q00_3 * q02 - 6 * m100 * q00_3 * q02 + 6 * m010 * q00_2 * q01 * q02 - 6 * m110 * q00_2 * q01 * q02
                + 6 * m000 * q00 * q01_2 * q02 - 6 * m100 * q00 * q01_2 * q02 + 6 * m010 * q01_3 * q02 - 6 * m110 * q01_3 * q02
                + 6 * m000 * q00 * q02_3 - 6 * m100 * q00 * q02_3 + 6 * m010 * q01 * q02_3 - 6 * m110 * q01 * q02_3
                + 6 * m010 * q00_3 * q03 - 6 * m110 * q00_3 * q03 - 6 * m000 * q00_2 * q01 * q03 + 6 * m100 * q00_2 * q01 * q03
                + 6 * m010 * q00 * q01_2 * q03 - 6 * m110 * q00 * q01_2 * q03 - 6 * m000 * q01_3 * q03 + 6 * m100 * q01_3 * q03
                + 6 * m010 * q00 * q02_2 * q03 - 6 * m110 * q00 * q02_2 * q03 - 6 * m000 * q01 * q02_2 * q03
                + 6 * m100 * q01 * q02_2 * q03 + 6 * m000 * q00 * q02 * q03_2 - 6 * m100 * q00 * q02 * q03_2
                + 6 * m010 * q01 * q02 * q03_2 - 6 * m110 * q01 * q02 * q03_2 + 6 * m010 * q00 * q03_3
                - 6 * m110 * q00 * q03_3 - 6 * m000 * q01 * q03_3 + 6 * m100 * q01 * q03_3 - 6 * m000 * q00_2 * q02 * q10
                + 7 * m100 * q00_2 * q02 * q10 + 2 * m110 * q00 * q01 * q02 * q10 - 6 * m000 * q01_2 * q02 * q10
                + 5 * m100 * q01_2 * q02 * q10 - 6 * m000 * q02_3 * q10 + 5 * m100 * q02_3 * q10 - 6 * m010 * q00_2 * q03 * q10
                + 7 * m110 * q00_2 * q03 * q10 - 2 * m100 * q00 * q01 * q03 * q10 - 6 * m010 * q01_2 * q03 * q10
                + 5 * m110 * q01_2 * q03 * q10 - 6 * m010 * q02_2 * q03 * q10 + 5 * m110 * q02_2 * q03 * q10
                - 6 * m000 * q02 * q03_2 * q10 + 5 * m100 * q02 * q03_2 * q10 - 6 * m010 * q03_3 * q10
                + 5 * m110 * q03_3 * q10 - m000 * q00 * q02 * q10_2 - m100 * q00 * q02 * q10_2 - 3 * m010 * q01 * q02 * q10_2
                + m110 * q01 * q02 * q10_2 - m010 * q00 * q03 * q10_2 - m110 * q00 * q03 * q10_2 + 3 * m000 * q01 * q03 * q10_2
                - m100 * q01 * q03 * q10_2 + m000 * q02 * q10_3 + m010 * q03 * q10_3 - 6 * m010 * q00_2 * q02 * q11
                + 5 * m110 * q00_2 * q02 * q11 + 2 * m100 * q00 * q01 * q02 * q11 - 6 * m010 * q01_2 * q02 * q11
                + 7 * m110 * q01_2 * q02 * q11 - 6 * m010 * q02_3 * q11 + 5 * m110 * q02_3 * q11 + 6 * m000 * q00_2 * q03 * q11
                - 5 * m100 * q00_2 * q03 * q11 + 2 * m110 * q00 * q01 * q03 * q11 + 6 * m000 * q01_2 * q03 * q11
                - 7 * m100 * q01_2 * q03 * q11 + 6 * m000 * q02_2 * q03 * q11 - 5 * m100 * q02_2 * q03 * q11
                - 6 * m010 * q02 * q03_2 * q11 + 5 * m110 * q02 * q03_2 * q11 + 6 * m000 * q03_3 * q11 - 5 * m100 * q03_3 * q11
                + 2 * m010 * q00 * q02 * q10 * q11 - 2 * m110 * q00 * q02 * q10 * q11 + 2 * m000 * q01 * q02 * q10 * q11
                - 2 * m100 * q01 * q02 * q10 * q11 - 2 * m000 * q00 * q03 * q10 * q11 + 2 * m100 * q00 * q03 * q10 * q11
                + 2 * m010 * q01 * q03 * q10 * q11 - 2 * m110 * q01 * q03 * q10 * q11 + m010 * q02 * q10_2 * q11
                - m000 * q03 * q10_2 * q11 - 3 * m000 * q00 * q02 * q11_2 + m100 * q00 * q02 * q11_2 - m010 * q01 * q02 * q11_2
                - m110 * q01 * q02 * q11_2 - 3 * m010 * q00 * q03 * q11_2 + m110 * q00 * q03 * q11_2 + m000 * q01 * q03 * q11_2
                + m100 * q01 * q03 * q11_2 + m000 * q02 * q10 * q11_2 + m010 * q03 * q10 * q11_2 + m010 * q02 * q11_3
                - m000 * q03 * q11_3 - 6 * m000 * q00_3 * q12 + 5 * m100 * q00_3 * q12 - 6 * m010 * q00_2 * q01 * q12
                + 5 * m110 * q00_2 * q01 * q12 - 6 * m000 * q00 * q01_2 * q12 + 5 * m100 * q00 * q01_2 * q12
                - 6 * m010 * q01_3 * q12 + 5 * m110 * q01_3 * q12 - 6 * m000 * q00 * q02_2 * q12 + 7 * m100 * q00 * q02_2 * q12
                - 6 * m010 * q01 * q02_2 * q12 + 7 * m110 * q01 * q02_2 * q12 + 2 * m110 * q00 * q02 * q03 * q12
                - 2 * m100 * q01 * q02 * q03 * q12 - 6 * m000 * q00 * q03_2 * q12 + 5 * m100 * q00 * q03_2 * q12
                - 6 * m010 * q01 * q03_2 * q12 + 5 * m110 * q01 * q03_2 * q12 + 7 * m000 * q00_2 * q10 * q12
                - 5 * m100 * q00_2 * q10 * q12 + 2 * m010 * q00 * q01 * q10 * q12 - 2 * m110 * q00 * q01 * q10 * q12
                + 5 * m000 * q01_2 * q10 * q12 - 3 * m100 * q01_2 * q10 * q12 + 7 * m000 * q02_2 * q10 * q12
                - 5 * m100 * q02_2 * q10 * q12 + 2 * m010 * q02 * q03 * q10 * q12 - 2 * m110 * q02 * q03 * q10 * q12
                + 5 * m000 * q03_2 * q10 * q12 - 3 * m100 * q03_2 * q10 * q12 - m000 * q00 * q10_2 * q12
                + m010 * q01 * q10_2 * q12 + 5 * m010 * q00_2 * q11 * q12 - 3 * m110 * q00_2 * q11 * q12
                + 2 * m000 * q00 * q01 * q11 * q12 - 2 * m100 * q00 * q01 * q11 * q12 + 7 * m010 * q01_2 * q11 * q12
                - 5 * m110 * q01_2 * q11 * q12 + 7 * m010 * q02_2 * q11 * q12 - 5 * m110 * q02_2 * q11 * q12
                - 2 * m000 * q02 * q03 * q11 * q12 + 2 * m100 * q02 * q03 * q11 * q12 + 5 * m010 * q03_2 * q11 * q12
                - 3 * m110 * q03_2 * q11 * q12 - 2 * m010 * q00 * q10 * q11 * q12 - 2 * m000 * q01 * q10 * q11 * q12
                + m000 * q00 * q11_2 * q12 - m010 * q01 * q11_2 * q12 - m000 * q00 * q02 * q12_2 - m100 * q00 * q02 * q12_2
                - m010 * q01 * q02 * q12_2 - m110 * q01 * q02 * q12_2 - 3 * m010 * q00 * q03 * q12_2 + m110 * q00 * q03 * q12_2
                + 3 * m000 * q01 * q03 * q12_2 - m100 * q01 * q03 * q12_2 - m000 * q02 * q10 * q12_2 + m010 * q03 * q10 * q12_2
                - m010 * q02 * q11 * q12_2 - m000 * q03 * q11 * q12_2 + m000 * q00 * q12_3 + m010 * q01 * q12_3
                + ( 6 * m000 * q00_2 * q01 - 5 * m100 * q00_2 * q01 + 6 * m000 * q01_3 - 5 * m100 * q01_3 + 6 * m000 * q01 * q02_2
                    - 5 * m100 * q01 * q02_2 + 2 * m100 * q00 * q02 * q03 + 6 * m000 * q01 * q03_2 - 7 * m100 * q01 * q03_2
                    - 2 * m000 * q00 * q01 * q10 + 2 * m100 * q00 * q01 * q10 + 2 * m000 * q02 * q03 * q10
                    - 2 * m100 * q02 * q03 * q10 - m000 * q01 * q10_2 - 5 * m000 * q00_2 * q11 + 3 * m100 * q00_2 * q11
                    - 7 * m000 * q01_2 * q11 + 5 * m100 * q01_2 * q11 - 5 * m000 * q02_2 * q11 + 3 * m100 * q02_2 * q11
                    - 7 * m000 * q03_2 * q11 + 5 * m100 * q03_2 * q11 + 2 * m000 * q00 * q10 * q11 + m000 * q01 * q11_2
                    + 2 * ( m100 * ( q01 * q02 - q00 * q03 ) + m000 * ( -( q01 * q02 ) + q00 * q03 - q03 * q10 + q02 * q11 ) ) * q12
                    - m000 * q01 * q12_2
                    + m110
                          * ( 5 * q00_3 - 5 * q00_2 * q10 - 3 * q01_2 * q10 - 3 * q02_2 * q10 - 5 * q03_2 * q10
                              - 2 * q02 * q03 * q11 + 2 * q01 * q03 * ( q02 - q12 )
                              + q00 * ( 5 * q01_2 + 5 * q02_2 + 7 * q03_2 - 2 * q01 * q11 - 2 * q02 * q12 ) )
                    + m010
                          * ( -6 * q00_3 + 7 * q00_2 * q10 + 5 * q01_2 * q10 + 5 * q02_2 * q10 + 7 * q03_2 * q10
                              + 2 * q02 * q03 * q11 - 2 * q01 * q10 * q11 - 2 * ( -( q01 * q03 ) + q02 * q10 + q03 * q11 ) * q12
                              + q00 * ( -6 * q01_2 - 6 * q02_2 - 6 * q03_2 - q10_2 + 2 * q01 * q11 + q11_2 + 2 * q02 * q12 + q12_2 ) ) )
                      * q13
                + ( m100 * q00 * q02 - 3 * m010 * q01 * q02 + m110 * q01 * q02 - m010 * q00 * q03 - m110 * q00 * q03
                    + m100 * q01 * q03 - m010 * q03 * q10 + m010 * q02 * q11 + m010 * q01 * q12
                    + m000 * ( q02 * q10 + q03 * ( q01 + q11 ) + q00 * ( -3 * q02 + q12 ) ) )
                      * q13_2
                + ( m010 * q00 - m000 * q01 ) * q13_3
                + m020
                      * ( -3 * q00_4 - 3 * q01_4 + 3 * q023_2 + 6 * q00_3 * q10 - 4 * q02_2 * q10_2 - 4 * q03_2 * q10_2
                          + 6 * q01_3 * q11 - 4 * q02_2 * q11_2 - 4 * q03_2 * q11_2 - 6 * q02_3 * q12 - 6 * q02 * q03_2 * q12
                          + 2 * q02 * q10_2 * q12 + 2 * q02 * q11_2 * q12 + 3 * q02_2 * q12_2 + q03_2 * q12_2
                          + 2 * q03 * ( -3 * q02_2 - 3 * q03_2 + q10_2 + q11_2 + 2 * q02 * q12 ) * q13 + ( q02_2 + 3 * q03_2 ) * q13_2
                          - q00_2 * ( 6 * q01_2 + 3 * q10_2 - 6 * q01 * q11 + q11_2 + 6 * q02 * q12 - 4 * q12_2 + 6 * q03 * q13 - 4 * q13_2 )
                          - q01_2 * ( q10_2 + 3 * q11_2 + 6 * q02 * q12 - 4 * q12_2 + 6 * q03 * q13 - 4 * q13_2 )
                          + 2 * q01 * q11 * ( 3 * q02_2 + 3 * q03_2 - q12_2 - q13_2 )
                          + 2 * q00 * q10 * ( 3 * q01_2 + 3 * q02_2 + 3 * q03_2 - 2 * q01 * q11 - q12_2 - q13_2 ) )
                + m120
                      * ( 3 * q00_4 + 3 * q01_4 - 3 * q023_2 - 6 * q00_3 * q10 + 2 * q02_2 * q10_2 + 2 * q03_2 * q10_2
                          - 6 * q01_3 * q11 - 4 * q01 * ( q02_2 + q03_2 ) * q11 + 2 * q02_2 * q11_2 + 2 * q03_2 * q11_2
                          - 2 * q00 * q10 * ( 3 * q01_2 + 2 * ( q02_2 + q03_2 ) - 2 * q01 * q11 ) + 6 * q02_3 * q12
                          + 6 * q02 * q03_2 * q12 - 3 * q02_2 * q12_2 - q03_2 * q12_2
                          + 2 * q03 * ( 3 * ( q02_2 + q03_2 ) - 2 * q02 * q12 ) * q13 - ( q02_2 + 3 * q03_2 ) * q13_2
                          + q00_2 * ( 6 * q01_2 + 3 * q10_2 - 6 * q01 * q11 + q11_2 + 4 * q02 * q12 - 2 * ( q12_2 + q13 * ( -2 * q03 + q13 ) ) )
                          + q01_2 * ( q10_2 + 3 * q11_2 + 4 * q02 * q12 - 2 * ( q12_2 + q13 * ( -2 * q03 + q13 ) ) ) ) ),
        -2
            * ( 6 * m001 * q00_3 * q02 - 6 * m101 * q00_3 * q02 + 6 * m011 * q00_2 * q01 * q02 - 6 * m111 * q00_2 * q01 * q02
                + 6 * m001 * q00 * q01_2 * q02 - 6 * m101 * q00 * q01_2 * q02 + 6 * m011 * q01_3 * q02 - 6 * m111 * q01_3 * q02
                + 6 * m001 * q00 * q02_3 - 6 * m101 * q00 * q02_3 + 6 * m011 * q01 * q02_3 - 6 * m111 * q01 * q02_3
                + 6 * m011 * q00_3 * q03 - 6 * m111 * q00_3 * q03 - 6 * m001 * q00_2 * q01 * q03 + 6 * m101 * q00_2 * q01 * q03
                + 6 * m011 * q00 * q01_2 * q03 - 6 * m111 * q00 * q01_2 * q03 - 6 * m001 * q01_3 * q03 + 6 * m101 * q01_3 * q03
                + 6 * m011 * q00 * q02_2 * q03 - 6 * m111 * q00 * q02_2 * q03 - 6 * m001 * q01 * q02_2 * q03
                + 6 * m101 * q01 * q02_2 * q03 + 6 * m001 * q00 * q02 * q03_2 - 6 * m101 * q00 * q02 * q03_2
                + 6 * m011 * q01 * q02 * q03_2 - 6 * m111 * q01 * q02 * q03_2 + 6 * m011 * q00 * q03_3
                - 6 * m111 * q00 * q03_3 - 6 * m001 * q01 * q03_3 + 6 * m101 * q01 * q03_3 - 6 * m001 * q00_2 * q02 * q10
                + 7 * m101 * q00_2 * q02 * q10 + 2 * m111 * q00 * q01 * q02 * q10 - 6 * m001 * q01_2 * q02 * q10
                + 5 * m101 * q01_2 * q02 * q10 - 6 * m001 * q02_3 * q10 + 5 * m101 * q02_3 * q10 - 6 * m011 * q00_2 * q03 * q10
                + 7 * m111 * q00_2 * q03 * q10 - 2 * m101 * q00 * q01 * q03 * q10 - 6 * m011 * q01_2 * q03 * q10
                + 5 * m111 * q01_2 * q03 * q10 - 6 * m011 * q02_2 * q03 * q10 + 5 * m111 * q02_2 * q03 * q10
                - 6 * m001 * q02 * q03_2 * q10 + 5 * m101 * q02 * q03_2 * q10 - 6 * m011 * q03_3 * q10
                + 5 * m111 * q03_3 * q10 - m001 * q00 * q02 * q10_2 - m101 * q00 * q02 * q10_2 - 3 * m011 * q01 * q02 * q10_2
                + m111 * q01 * q02 * q10_2 - m011 * q00 * q03 * q10_2 - m111 * q00 * q03 * q10_2 + 3 * m001 * q01 * q03 * q10_2
                - m101 * q01 * q03 * q10_2 + m001 * q02 * q10_3 + m011 * q03 * q10_3 - 6 * m011 * q00_2 * q02 * q11
                + 5 * m111 * q00_2 * q02 * q11 + 2 * m101 * q00 * q01 * q02 * q11 - 6 * m011 * q01_2 * q02 * q11
                + 7 * m111 * q01_2 * q02 * q11 - 6 * m011 * q02_3 * q11 + 5 * m111 * q02_3 * q11 + 6 * m001 * q00_2 * q03 * q11
                - 5 * m101 * q00_2 * q03 * q11 + 2 * m111 * q00 * q01 * q03 * q11 + 6 * m001 * q01_2 * q03 * q11
                - 7 * m101 * q01_2 * q03 * q11 + 6 * m001 * q02_2 * q03 * q11 - 5 * m101 * q02_2 * q03 * q11
                - 6 * m011 * q02 * q03_2 * q11 + 5 * m111 * q02 * q03_2 * q11 + 6 * m001 * q03_3 * q11 - 5 * m101 * q03_3 * q11
                + 2 * m011 * q00 * q02 * q10 * q11 - 2 * m111 * q00 * q02 * q10 * q11 + 2 * m001 * q01 * q02 * q10 * q11
                - 2 * m101 * q01 * q02 * q10 * q11 - 2 * m001 * q00 * q03 * q10 * q11 + 2 * m101 * q00 * q03 * q10 * q11
                + 2 * m011 * q01 * q03 * q10 * q11 - 2 * m111 * q01 * q03 * q10 * q11 + m011 * q02 * q10_2 * q11
                - m001 * q03 * q10_2 * q11 - 3 * m001 * q00 * q02 * q11_2 + m101 * q00 * q02 * q11_2 - m011 * q01 * q02 * q11_2
                - m111 * q01 * q02 * q11_2 - 3 * m011 * q00 * q03 * q11_2 + m111 * q00 * q03 * q11_2 + m001 * q01 * q03 * q11_2
                + m101 * q01 * q03 * q11_2 + m001 * q02 * q10 * q11_2 + m011 * q03 * q10 * q11_2 + m011 * q02 * q11_3
                - m001 * q03 * q11_3 - 6 * m001 * q00_3 * q12 + 5 * m101 * q00_3 * q12 - 6 * m011 * q00_2 * q01 * q12
                + 5 * m111 * q00_2 * q01 * q12 - 6 * m001 * q00 * q01_2 * q12 + 5 * m101 * q00 * q01_2 * q12
                - 6 * m011 * q01_3 * q12 + 5 * m111 * q01_3 * q12 - 6 * m001 * q00 * q02_2 * q12 + 7 * m101 * q00 * q02_2 * q12
                - 6 * m011 * q01 * q02_2 * q12 + 7 * m111 * q01 * q02_2 * q12 + 2 * m111 * q00 * q02 * q03 * q12
                - 2 * m101 * q01 * q02 * q03 * q12 - 6 * m001 * q00 * q03_2 * q12 + 5 * m101 * q00 * q03_2 * q12
                - 6 * m011 * q01 * q03_2 * q12 + 5 * m111 * q01 * q03_2 * q12 + 7 * m001 * q00_2 * q10 * q12
                - 5 * m101 * q00_2 * q10 * q12 + 2 * m011 * q00 * q01 * q10 * q12 - 2 * m111 * q00 * q01 * q10 * q12
                + 5 * m001 * q01_2 * q10 * q12 - 3 * m101 * q01_2 * q10 * q12 + 7 * m001 * q02_2 * q10 * q12
                - 5 * m101 * q02_2 * q10 * q12 + 2 * m011 * q02 * q03 * q10 * q12 - 2 * m111 * q02 * q03 * q10 * q12
                + 5 * m001 * q03_2 * q10 * q12 - 3 * m101 * q03_2 * q10 * q12 - m001 * q00 * q10_2 * q12
                + m011 * q01 * q10_2 * q12 + 5 * m011 * q00_2 * q11 * q12 - 3 * m111 * q00_2 * q11 * q12
                + 2 * m001 * q00 * q01 * q11 * q12 - 2 * m101 * q00 * q01 * q11 * q12 + 7 * m011 * q01_2 * q11 * q12
                - 5 * m111 * q01_2 * q11 * q12 + 7 * m011 * q02_2 * q11 * q12 - 5 * m111 * q02_2 * q11 * q12
                - 2 * m001 * q02 * q03 * q11 * q12 + 2 * m101 * q02 * q03 * q11 * q12 + 5 * m011 * q03_2 * q11 * q12
                - 3 * m111 * q03_2 * q11 * q12 - 2 * m011 * q00 * q10 * q11 * q12 - 2 * m001 * q01 * q10 * q11 * q12
                + m001 * q00 * q11_2 * q12 - m011 * q01 * q11_2 * q12 - m001 * q00 * q02 * q12_2 - m101 * q00 * q02 * q12_2
                - m011 * q01 * q02 * q12_2 - m111 * q01 * q02 * q12_2 - 3 * m011 * q00 * q03 * q12_2 + m111 * q00 * q03 * q12_2
                + 3 * m001 * q01 * q03 * q12_2 - m101 * q01 * q03 * q12_2 - m001 * q02 * q10 * q12_2 + m011 * q03 * q10 * q12_2
                - m011 * q02 * q11 * q12_2 - m001 * q03 * q11 * q12_2 + m001 * q00 * q12_3 + m011 * q01 * q12_3
                + ( 6 * m001 * q00_2 * q01 - 5 * m101 * q00_2 * q01 + 6 * m001 * q01_3 - 5 * m101 * q01_3 + 6 * m001 * q01 * q02_2
                    - 5 * m101 * q01 * q02_2 + 2 * m101 * q00 * q02 * q03 + 6 * m001 * q01 * q03_2 - 7 * m101 * q01 * q03_2
                    - 2 * m001 * q00 * q01 * q10 + 2 * m101 * q00 * q01 * q10 + 2 * m001 * q02 * q03 * q10
                    - 2 * m101 * q02 * q03 * q10 - m001 * q01 * q10_2 - 5 * m001 * q00_2 * q11 + 3 * m101 * q00_2 * q11
                    - 7 * m001 * q01_2 * q11 + 5 * m101 * q01_2 * q11 - 5 * m001 * q02_2 * q11 + 3 * m101 * q02_2 * q11
                    - 7 * m001 * q03_2 * q11 + 5 * m101 * q03_2 * q11 + 2 * m001 * q00 * q10 * q11 + m001 * q01 * q11_2
                    + 2 * ( m101 * ( q01 * q02 - q00 * q03 ) + m001 * ( -( q01 * q02 ) + q00 * q03 - q03 * q10 + q02 * q11 ) ) * q12
                    - m001 * q01 * q12_2
                    + m111
                          * ( 5 * q00_3 - 5 * q00_2 * q10 - 3 * q01_2 * q10 - 3 * q02_2 * q10 - 5 * q03_2 * q10
                              - 2 * q02 * q03 * q11 + 2 * q01 * q03 * ( q02 - q12 )
                              + q00 * ( 5 * q01_2 + 5 * q02_2 + 7 * q03_2 - 2 * q01 * q11 - 2 * q02 * q12 ) )
                    + m011
                          * ( -6 * q00_3 + 7 * q00_2 * q10 + 5 * q01_2 * q10 + 5 * q02_2 * q10 + 7 * q03_2 * q10
                              + 2 * q02 * q03 * q11 - 2 * q01 * q10 * q11 - 2 * ( -( q01 * q03 ) + q02 * q10 + q03 * q11 ) * q12
                              + q00 * ( -6 * q01_2 - 6 * q02_2 - 6 * q03_2 - q10_2 + 2 * q01 * q11 + q11_2 + 2 * q02 * q12 + q12_2 ) ) )
                      * q13
                + ( m101 * q00 * q02 - 3 * m011 * q01 * q02 + m111 * q01 * q02 - m011 * q00 * q03 - m111 * q00 * q03
                    + m101 * q01 * q03 - m011 * q03 * q10 + m011 * q02 * q11 + m011 * q01 * q12
                    + m001 * ( q02 * q10 + q03 * ( q01 + q11 ) + q00 * ( -3 * q02 + q12 ) ) )
                      * q13_2
                + ( m011 * q00 - m001 * q01 ) * q13_3
                + m021
                      * ( -3 * q00_4 - 3 * q01_4 + 3 * q023_2 + 6 * q00_3 * q10 - 4 * q02_2 * q10_2 - 4 * q03_2 * q10_2
                          + 6 * q01_3 * q11 - 4 * q02_2 * q11_2 - 4 * q03_2 * q11_2 - 6 * q02_3 * q12 - 6 * q02 * q03_2 * q12
                          + 2 * q02 * q10_2 * q12 + 2 * q02 * q11_2 * q12 + 3 * q02_2 * q12_2 + q03_2 * q12_2
                          + 2 * q03 * ( -3 * q02_2 - 3 * q03_2 + q10_2 + q11_2 + 2 * q02 * q12 ) * q13 + ( q02_2 + 3 * q03_2 ) * q13_2
                          - q00_2 * ( 6 * q01_2 + 3 * q10_2 - 6 * q01 * q11 + q11_2 + 6 * q02 * q12 - 4 * q12_2 + 6 * q03 * q13 - 4 * q13_2 )
                          - q01_2 * ( q10_2 + 3 * q11_2 + 6 * q02 * q12 - 4 * q12_2 + 6 * q03 * q13 - 4 * q13_2 )
                          + 2 * q01 * q11 * ( 3 * q02_2 + 3 * q03_2 - q12_2 - q13_2 )
                          + 2 * q00 * q10 * ( 3 * q01_2 + 3 * q02_2 + 3 * q03_2 - 2 * q01 * q11 - q12_2 - q13_2 ) )
                + m121
                      * ( 3 * q00_4 + 3 * q01_4 - 3 * q023_2 - 6 * q00_3 * q10 + 2 * q02_2 * q10_2 + 2 * q03_2 * q10_2
                          - 6 * q01_3 * q11 - 4 * q01 * ( q02_2 + q03_2 ) * q11 + 2 * q02_2 * q11_2 + 2 * q03_2 * q11_2
                          - 2 * q00 * q10 * ( 3 * q01_2 + 2 * ( q02_2 + q03_2 ) - 2 * q01 * q11 ) + 6 * q02_3 * q12
                          + 6 * q02 * q03_2 * q12 - 3 * q02_2 * q12_2 - q03_2 * q12_2
                          + 2 * q03 * ( 3 * ( q02_2 + q03_2 ) - 2 * q02 * q12 ) * q13 - ( q02_2 + 3 * q03_2 ) * q13_2
                          + q00_2 * ( 6 * q01_2 + 3 * q10_2 - 6 * q01 * q11 + q11_2 + 4 * q02 * q12 - 2 * ( q12_2 + q13 * ( -2 * q03 + q13 ) ) )
                          + q01_2 * ( q10_2 + 3 * q11_2 + 4 * q02 * q12 - 2 * ( q12_2 + q13 * ( -2 * q03 + q13 ) ) ) ) ),
        -2
            * ( 6 * m002 * q00_3 * q02 - 6 * m102 * q00_3 * q02 + 6 * m012 * q00_2 * q01 * q02 - 6 * m112 * q00_2 * q01 * q02
                + 6 * m002 * q00 * q01_2 * q02 - 6 * m102 * q00 * q01_2 * q02 + 6 * m012 * q01_3 * q02 - 6 * m112 * q01_3 * q02
                + 6 * m002 * q00 * q02_3 - 6 * m102 * q00 * q02_3 + 6 * m012 * q01 * q02_3 - 6 * m112 * q01 * q02_3
                + 6 * m012 * q00_3 * q03 - 6 * m112 * q00_3 * q03 - 6 * m002 * q00_2 * q01 * q03 + 6 * m102 * q00_2 * q01 * q03
                + 6 * m012 * q00 * q01_2 * q03 - 6 * m112 * q00 * q01_2 * q03 - 6 * m002 * q01_3 * q03 + 6 * m102 * q01_3 * q03
                + 6 * m012 * q00 * q02_2 * q03 - 6 * m112 * q00 * q02_2 * q03 - 6 * m002 * q01 * q02_2 * q03
                + 6 * m102 * q01 * q02_2 * q03 + 6 * m002 * q00 * q02 * q03_2 - 6 * m102 * q00 * q02 * q03_2
                + 6 * m012 * q01 * q02 * q03_2 - 6 * m112 * q01 * q02 * q03_2 + 6 * m012 * q00 * q03_3
                - 6 * m112 * q00 * q03_3 - 6 * m002 * q01 * q03_3 + 6 * m102 * q01 * q03_3 - 6 * m002 * q00_2 * q02 * q10
                + 7 * m102 * q00_2 * q02 * q10 + 2 * m112 * q00 * q01 * q02 * q10 - 6 * m002 * q01_2 * q02 * q10
                + 5 * m102 * q01_2 * q02 * q10 - 6 * m002 * q02_3 * q10 + 5 * m102 * q02_3 * q10 - 6 * m012 * q00_2 * q03 * q10
                + 7 * m112 * q00_2 * q03 * q10 - 2 * m102 * q00 * q01 * q03 * q10 - 6 * m012 * q01_2 * q03 * q10
                + 5 * m112 * q01_2 * q03 * q10 - 6 * m012 * q02_2 * q03 * q10 + 5 * m112 * q02_2 * q03 * q10
                - 6 * m002 * q02 * q03_2 * q10 + 5 * m102 * q02 * q03_2 * q10 - 6 * m012 * q03_3 * q10
                + 5 * m112 * q03_3 * q10 - m002 * q00 * q02 * q10_2 - m102 * q00 * q02 * q10_2 - 3 * m012 * q01 * q02 * q10_2
                + m112 * q01 * q02 * q10_2 - m012 * q00 * q03 * q10_2 - m112 * q00 * q03 * q10_2 + 3 * m002 * q01 * q03 * q10_2
                - m102 * q01 * q03 * q10_2 + m002 * q02 * q10_3 + m012 * q03 * q10_3 - 6 * m012 * q00_2 * q02 * q11
                + 5 * m112 * q00_2 * q02 * q11 + 2 * m102 * q00 * q01 * q02 * q11 - 6 * m012 * q01_2 * q02 * q11
                + 7 * m112 * q01_2 * q02 * q11 - 6 * m012 * q02_3 * q11 + 5 * m112 * q02_3 * q11 + 6 * m002 * q00_2 * q03 * q11
                - 5 * m102 * q00_2 * q03 * q11 + 2 * m112 * q00 * q01 * q03 * q11 + 6 * m002 * q01_2 * q03 * q11
                - 7 * m102 * q01_2 * q03 * q11 + 6 * m002 * q02_2 * q03 * q11 - 5 * m102 * q02_2 * q03 * q11
                - 6 * m012 * q02 * q03_2 * q11 + 5 * m112 * q02 * q03_2 * q11 + 6 * m002 * q03_3 * q11 - 5 * m102 * q03_3 * q11
                + 2 * m012 * q00 * q02 * q10 * q11 - 2 * m112 * q00 * q02 * q10 * q11 + 2 * m002 * q01 * q02 * q10 * q11
                - 2 * m102 * q01 * q02 * q10 * q11 - 2 * m002 * q00 * q03 * q10 * q11 + 2 * m102 * q00 * q03 * q10 * q11
                + 2 * m012 * q01 * q03 * q10 * q11 - 2 * m112 * q01 * q03 * q10 * q11 + m012 * q02 * q10_2 * q11
                - m002 * q03 * q10_2 * q11 - 3 * m002 * q00 * q02 * q11_2 + m102 * q00 * q02 * q11_2 - m012 * q01 * q02 * q11_2
                - m112 * q01 * q02 * q11_2 - 3 * m012 * q00 * q03 * q11_2 + m112 * q00 * q03 * q11_2 + m002 * q01 * q03 * q11_2
                + m102 * q01 * q03 * q11_2 + m002 * q02 * q10 * q11_2 + m012 * q03 * q10 * q11_2 + m012 * q02 * q11_3
                - m002 * q03 * q11_3 - 6 * m002 * q00_3 * q12 + 5 * m102 * q00_3 * q12 - 6 * m012 * q00_2 * q01 * q12
                + 5 * m112 * q00_2 * q01 * q12 - 6 * m002 * q00 * q01_2 * q12 + 5 * m102 * q00 * q01_2 * q12
                - 6 * m012 * q01_3 * q12 + 5 * m112 * q01_3 * q12 - 6 * m002 * q00 * q02_2 * q12 + 7 * m102 * q00 * q02_2 * q12
                - 6 * m012 * q01 * q02_2 * q12 + 7 * m112 * q01 * q02_2 * q12 + 2 * m112 * q00 * q02 * q03 * q12
                - 2 * m102 * q01 * q02 * q03 * q12 - 6 * m002 * q00 * q03_2 * q12 + 5 * m102 * q00 * q03_2 * q12
                - 6 * m012 * q01 * q03_2 * q12 + 5 * m112 * q01 * q03_2 * q12 + 7 * m002 * q00_2 * q10 * q12
                - 5 * m102 * q00_2 * q10 * q12 + 2 * m012 * q00 * q01 * q10 * q12 - 2 * m112 * q00 * q01 * q10 * q12
                + 5 * m002 * q01_2 * q10 * q12 - 3 * m102 * q01_2 * q10 * q12 + 7 * m002 * q02_2 * q10 * q12
                - 5 * m102 * q02_2 * q10 * q12 + 2 * m012 * q02 * q03 * q10 * q12 - 2 * m112 * q02 * q03 * q10 * q12
                + 5 * m002 * q03_2 * q10 * q12 - 3 * m102 * q03_2 * q10 * q12 - m002 * q00 * q10_2 * q12
                + m012 * q01 * q10_2 * q12 + 5 * m012 * q00_2 * q11 * q12 - 3 * m112 * q00_2 * q11 * q12
                + 2 * m002 * q00 * q01 * q11 * q12 - 2 * m102 * q00 * q01 * q11 * q12 + 7 * m012 * q01_2 * q11 * q12
                - 5 * m112 * q01_2 * q11 * q12 + 7 * m012 * q02_2 * q11 * q12 - 5 * m112 * q02_2 * q11 * q12
                - 2 * m002 * q02 * q03 * q11 * q12 + 2 * m102 * q02 * q03 * q11 * q12 + 5 * m012 * q03_2 * q11 * q12
                - 3 * m112 * q03_2 * q11 * q12 - 2 * m012 * q00 * q10 * q11 * q12 - 2 * m002 * q01 * q10 * q11 * q12
                + m002 * q00 * q11_2 * q12 - m012 * q01 * q11_2 * q12 - m002 * q00 * q02 * q12_2 - m102 * q00 * q02 * q12_2
                - m012 * q01 * q02 * q12_2 - m112 * q01 * q02 * q12_2 - 3 * m012 * q00 * q03 * q12_2 + m112 * q00 * q03 * q12_2
                + 3 * m002 * q01 * q03 * q12_2 - m102 * q01 * q03 * q12_2 - m002 * q02 * q10 * q12_2 + m012 * q03 * q10 * q12_2
                - m012 * q02 * q11 * q12_2 - m002 * q03 * q11 * q12_2 + m002 * q00 * q12_3 + m012 * q01 * q12_3
                + ( 6 * m002 * q00_2 * q01 - 5 * m102 * q00_2 * q01 + 6 * m002 * q01_3 - 5 * m102 * q01_3 + 6 * m002 * q01 * q02_2
                    - 5 * m102 * q01 * q02_2 + 2 * m102 * q00 * q02 * q03 + 6 * m002 * q01 * q03_2 - 7 * m102 * q01 * q03_2
                    - 2 * m002 * q00 * q01 * q10 + 2 * m102 * q00 * q01 * q10 + 2 * m002 * q02 * q03 * q10
                    - 2 * m102 * q02 * q03 * q10 - m002 * q01 * q10_2 - 5 * m002 * q00_2 * q11 + 3 * m102 * q00_2 * q11
                    - 7 * m002 * q01_2 * q11 + 5 * m102 * q01_2 * q11 - 5 * m002 * q02_2 * q11 + 3 * m102 * q02_2 * q11
                    - 7 * m002 * q03_2 * q11 + 5 * m102 * q03_2 * q11 + 2 * m002 * q00 * q10 * q11 + m002 * q01 * q11_2
                    + 2 * ( m102 * ( q01 * q02 - q00 * q03 ) + m002 * ( -( q01 * q02 ) + q00 * q03 - q03 * q10 + q02 * q11 ) ) * q12
                    - m002 * q01 * q12_2
                    + m112
                          * ( 5 * q00_3 - 5 * q00_2 * q10 - 3 * q01_2 * q10 - 3 * q02_2 * q10 - 5 * q03_2 * q10
                              - 2 * q02 * q03 * q11 + 2 * q01 * q03 * ( q02 - q12 )
                              + q00 * ( 5 * q01_2 + 5 * q02_2 + 7 * q03_2 - 2 * q01 * q11 - 2 * q02 * q12 ) )
                    + m012
                          * ( -6 * q00_3 + 7 * q00_2 * q10 + 5 * q01_2 * q10 + 5 * q02_2 * q10 + 7 * q03_2 * q10
                              + 2 * q02 * q03 * q11 - 2 * q01 * q10 * q11 - 2 * ( -( q01 * q03 ) + q02 * q10 + q03 * q11 ) * q12
                              + q00 * ( -6 * q01_2 - 6 * q02_2 - 6 * q03_2 - q10_2 + 2 * q01 * q11 + q11_2 + 2 * q02 * q12 + q12_2 ) ) )
                      * q13
                + ( m102 * q00 * q02 - 3 * m012 * q01 * q02 + m112 * q01 * q02 - m012 * q00 * q03 - m112 * q00 * q03
                    + m102 * q01 * q03 - m012 * q03 * q10 + m012 * q02 * q11 + m012 * q01 * q12
                    + m002 * ( q02 * q10 + q03 * ( q01 + q11 ) + q00 * ( -3 * q02 + q12 ) ) )
                      * q13_2
                + ( m012 * q00 - m002 * q01 ) * q13_3
                + m022
                      * ( -3 * q00_4 - 3 * q01_4 + 3 * q023_2 + 6 * q00_3 * q10 - 4 * q02_2 * q10_2 - 4 * q03_2 * q10_2
                          + 6 * q01_3 * q11 - 4 * q02_2 * q11_2 - 4 * q03_2 * q11_2 - 6 * q02_3 * q12 - 6 * q02 * q03_2 * q12
                          + 2 * q02 * q10_2 * q12 + 2 * q02 * q11_2 * q12 + 3 * q02_2 * q12_2 + q03_2 * q12_2
                          + 2 * q03 * ( -3 * q02_2 - 3 * q03_2 + q10_2 + q11_2 + 2 * q02 * q12 ) * q13 + ( q02_2 + 3 * q03_2 ) * q13_2
                          - q00_2 * ( 6 * q01_2 + 3 * q10_2 - 6 * q01 * q11 + q11_2 + 6 * q02 * q12 - 4 * q12_2 + 6 * q03 * q13 - 4 * q13_2 )
                          - q01_2 * ( q10_2 + 3 * q11_2 + 6 * q02 * q12 - 4 * q12_2 + 6 * q03 * q13 - 4 * q13_2 )
                          + 2 * q01 * q11 * ( 3 * q02_2 + 3 * q03_2 - q12_2 - q13_2 )
                          + 2 * q00 * q10 * ( 3 * q01_2 + 3 * q02_2 + 3 * q03_2 - 2 * q01 * q11 - q12_2 - q13_2 ) )
                + m122
                      * ( 3 * q00_4 + 3 * q01_4 - 3 * q023_2 - 6 * q00_3 * q10 + 2 * q02_2 * q10_2 + 2 * q03_2 * q10_2
                          - 6 * q01_3 * q11 - 4 * q01 * ( q02_2 + q03_2 ) * q11 + 2 * q02_2 * q11_2 + 2 * q03_2 * q11_2
                          - 2 * q00 * q10 * ( 3 * q01_2 + 2 * ( q02_2 + q03_2 ) - 2 * q01 * q11 ) + 6 * q02_3 * q12
                          + 6 * q02 * q03_2 * q12 - 3 * q02_2 * q12_2 - q03_2 * q12_2
                          + 2 * q03 * ( 3 * ( q02_2 + q03_2 ) - 2 * q02 * q12 ) * q13 - ( q02_2 + 3 * q03_2 ) * q13_2
                          + q00_2 * ( 6 * q01_2 + 3 * q10_2 - 6 * q01 * q11 + q11_2 + 4 * q02 * q12 - 2 * ( q12_2 + q13 * ( -2 * q03 + q13 ) ) )
                          + q01_2 * ( q10_2 + 3 * q11_2 + 4 * q02 * q12 - 2 * ( q12_2 + q13 * ( -2 * q03 + q13 ) ) ) ) ),
        -2
            * ( 6 * m003 * q00_3 * q02 - 6 * m103 * q00_3 * q02 + 6 * m013 * q00_2 * q01 * q02 - 6 * m113 * q00_2 * q01 * q02
                + 6 * m003 * q00 * q01_2 * q02 - 6 * m103 * q00 * q01_2 * q02 + 6 * m013 * q01_3 * q02 - 6 * m113 * q01_3 * q02
                + 6 * m003 * q00 * q02_3 - 6 * m103 * q00 * q02_3 + 6 * m013 * q01 * q02_3 - 6 * m113 * q01 * q02_3
                + 6 * m013 * q00_3 * q03 - 6 * m113 * q00_3 * q03 - 6 * m003 * q00_2 * q01 * q03 + 6 * m103 * q00_2 * q01 * q03
                + 6 * m013 * q00 * q01_2 * q03 - 6 * m113 * q00 * q01_2 * q03 - 6 * m003 * q01_3 * q03 + 6 * m103 * q01_3 * q03
                + 6 * m013 * q00 * q02_2 * q03 - 6 * m113 * q00 * q02_2 * q03 - 6 * m003 * q01 * q02_2 * q03
                + 6 * m103 * q01 * q02_2 * q03 + 6 * m003 * q00 * q02 * q03_2 - 6 * m103 * q00 * q02 * q03_2
                + 6 * m013 * q01 * q02 * q03_2 - 6 * m113 * q01 * q02 * q03_2 + 6 * m013 * q00 * q03_3
                - 6 * m113 * q00 * q03_3 - 6 * m003 * q01 * q03_3 + 6 * m103 * q01 * q03_3 - 6 * m003 * q00_2 * q02 * q10
                + 7 * m103 * q00_2 * q02 * q10 + 2 * m113 * q00 * q01 * q02 * q10 - 6 * m003 * q01_2 * q02 * q10
                + 5 * m103 * q01_2 * q02 * q10 - 6 * m003 * q02_3 * q10 + 5 * m103 * q02_3 * q10 - 6 * m013 * q00_2 * q03 * q10
                + 7 * m113 * q00_2 * q03 * q10 - 2 * m103 * q00 * q01 * q03 * q10 - 6 * m013 * q01_2 * q03 * q10
                + 5 * m113 * q01_2 * q03 * q10 - 6 * m013 * q02_2 * q03 * q10 + 5 * m113 * q02_2 * q03 * q10
                - 6 * m003 * q02 * q03_2 * q10 + 5 * m103 * q02 * q03_2 * q10 - 6 * m013 * q03_3 * q10
                + 5 * m113 * q03_3 * q10 - m003 * q00 * q02 * q10_2 - m103 * q00 * q02 * q10_2 - 3 * m013 * q01 * q02 * q10_2
                + m113 * q01 * q02 * q10_2 - m013 * q00 * q03 * q10_2 - m113 * q00 * q03 * q10_2 + 3 * m003 * q01 * q03 * q10_2
                - m103 * q01 * q03 * q10_2 + m003 * q02 * q10_3 + m013 * q03 * q10_3 - 6 * m013 * q00_2 * q02 * q11
                + 5 * m113 * q00_2 * q02 * q11 + 2 * m103 * q00 * q01 * q02 * q11 - 6 * m013 * q01_2 * q02 * q11
                + 7 * m113 * q01_2 * q02 * q11 - 6 * m013 * q02_3 * q11 + 5 * m113 * q02_3 * q11 + 6 * m003 * q00_2 * q03 * q11
                - 5 * m103 * q00_2 * q03 * q11 + 2 * m113 * q00 * q01 * q03 * q11 + 6 * m003 * q01_2 * q03 * q11
                - 7 * m103 * q01_2 * q03 * q11 + 6 * m003 * q02_2 * q03 * q11 - 5 * m103 * q02_2 * q03 * q11
                - 6 * m013 * q02 * q03_2 * q11 + 5 * m113 * q02 * q03_2 * q11 + 6 * m003 * q03_3 * q11 - 5 * m103 * q03_3 * q11
                + 2 * m013 * q00 * q02 * q10 * q11 - 2 * m113 * q00 * q02 * q10 * q11 + 2 * m003 * q01 * q02 * q10 * q11
                - 2 * m103 * q01 * q02 * q10 * q11 - 2 * m003 * q00 * q03 * q10 * q11 + 2 * m103 * q00 * q03 * q10 * q11
                + 2 * m013 * q01 * q03 * q10 * q11 - 2 * m113 * q01 * q03 * q10 * q11 + m013 * q02 * q10_2 * q11
                - m003 * q03 * q10_2 * q11 - 3 * m003 * q00 * q02 * q11_2 + m103 * q00 * q02 * q11_2 - m013 * q01 * q02 * q11_2
                - m113 * q01 * q02 * q11_2 - 3 * m013 * q00 * q03 * q11_2 + m113 * q00 * q03 * q11_2 + m003 * q01 * q03 * q11_2
                + m103 * q01 * q03 * q11_2 + m003 * q02 * q10 * q11_2 + m013 * q03 * q10 * q11_2 + m013 * q02 * q11_3
                - m003 * q03 * q11_3 - 6 * m003 * q00_3 * q12 + 5 * m103 * q00_3 * q12 - 6 * m013 * q00_2 * q01 * q12
                + 5 * m113 * q00_2 * q01 * q12 - 6 * m003 * q00 * q01_2 * q12 + 5 * m103 * q00 * q01_2 * q12
                - 6 * m013 * q01_3 * q12 + 5 * m113 * q01_3 * q12 - 6 * m003 * q00 * q02_2 * q12 + 7 * m103 * q00 * q02_2 * q12
                - 6 * m013 * q01 * q02_2 * q12 + 7 * m113 * q01 * q02_2 * q12 + 2 * m113 * q00 * q02 * q03 * q12
                - 2 * m103 * q01 * q02 * q03 * q12 - 6 * m003 * q00 * q03_2 * q12 + 5 * m103 * q00 * q03_2 * q12
                - 6 * m013 * q01 * q03_2 * q12 + 5 * m113 * q01 * q03_2 * q12 + 7 * m003 * q00_2 * q10 * q12
                - 5 * m103 * q00_2 * q10 * q12 + 2 * m013 * q00 * q01 * q10 * q12 - 2 * m113 * q00 * q01 * q10 * q12
                + 5 * m003 * q01_2 * q10 * q12 - 3 * m103 * q01_2 * q10 * q12 + 7 * m003 * q02_2 * q10 * q12
                - 5 * m103 * q02_2 * q10 * q12 + 2 * m013 * q02 * q03 * q10 * q12 - 2 * m113 * q02 * q03 * q10 * q12
                + 5 * m003 * q03_2 * q10 * q12 - 3 * m103 * q03_2 * q10 * q12 - m003 * q00 * q10_2 * q12
                + m013 * q01 * q10_2 * q12 + 5 * m013 * q00_2 * q11 * q12 - 3 * m113 * q00_2 * q11 * q12
                + 2 * m003 * q00 * q01 * q11 * q12 - 2 * m103 * q00 * q01 * q11 * q12 + 7 * m013 * q01_2 * q11 * q12
                - 5 * m113 * q01_2 * q11 * q12 + 7 * m013 * q02_2 * q11 * q12 - 5 * m113 * q02_2 * q11 * q12
                - 2 * m003 * q02 * q03 * q11 * q12 + 2 * m103 * q02 * q03 * q11 * q12 + 5 * m013 * q03_2 * q11 * q12
                - 3 * m113 * q03_2 * q11 * q12 - 2 * m013 * q00 * q10 * q11 * q12 - 2 * m003 * q01 * q10 * q11 * q12
                + m003 * q00 * q11_2 * q12 - m013 * q01 * q11_2 * q12 - m003 * q00 * q02 * q12_2 - m103 * q00 * q02 * q12_2
                - m013 * q01 * q02 * q12_2 - m113 * q01 * q02 * q12_2 - 3 * m013 * q00 * q03 * q12_2 + m113 * q00 * q03 * q12_2
                + 3 * m003 * q01 * q03 * q12_2 - m103 * q01 * q03 * q12_2 - m003 * q02 * q10 * q12_2 + m013 * q03 * q10 * q12_2
                - m013 * q02 * q11 * q12_2 - m003 * q03 * q11 * q12_2 + m003 * q00 * q12_3 + m013 * q01 * q12_3
                - 6 * m013 * q00_3 * q13 + 5 * m113 * q00_3 * q13 + 6 * m003 * q00_2 * q01 * q13 - 5 * m103 * q00_2 * q01 * q13
                - 6 * m013 * q00 * q01_2 * q13 + 5 * m113 * q00 * q01_2 * q13 + 6 * m003 * q01_3 * q13 - 5 * m103 * q01_3 * q13
                - 6 * m013 * q00 * q02_2 * q13 + 5 * m113 * q00 * q02_2 * q13 + 6 * m003 * q01 * q02_2 * q13
                - 5 * m103 * q01 * q02_2 * q13 + 2 * m103 * q00 * q02 * q03 * q13 + 2 * m113 * q01 * q02 * q03 * q13
                - 6 * m013 * q00 * q03_2 * q13 + 7 * m113 * q00 * q03_2 * q13 + 6 * m003 * q01 * q03_2 * q13
                - 7 * m103 * q01 * q03_2 * q13 + 7 * m013 * q00_2 * q10 * q13 - 5 * m113 * q00_2 * q10 * q13
                - 2 * m003 * q00 * q01 * q10 * q13 + 2 * m103 * q00 * q01 * q10 * q13 + 5 * m013 * q01_2 * q10 * q13
                - 3 * m113 * q01_2 * q10 * q13 + 5 * m013 * q02_2 * q10 * q13 - 3 * m113 * q02_2 * q10 * q13
                + 2 * m003 * q02 * q03 * q10 * q13 - 2 * m103 * q02 * q03 * q10 * q13 + 7 * m013 * q03_2 * q10 * q13
                - 5 * m113 * q03_2 * q10 * q13 - m013 * q00 * q10_2 * q13 - m003 * q01 * q10_2 * q13
                - 5 * m003 * q00_2 * q11 * q13 + 3 * m103 * q00_2 * q11 * q13 + 2 * m013 * q00 * q01 * q11 * q13
                - 2 * m113 * q00 * q01 * q11 * q13 - 7 * m003 * q01_2 * q11 * q13 + 5 * m103 * q01_2 * q11 * q13
                - 5 * m003 * q02_2 * q11 * q13 + 3 * m103 * q02_2 * q11 * q13 + 2 * m013 * q02 * q03 * q11 * q13
                - 2 * m113 * q02 * q03 * q11 * q13 - 7 * m003 * q03_2 * q11 * q13 + 5 * m103 * q03_2 * q11 * q13
                + 2 * m003 * q00 * q10 * q11 * q13 - 2 * m013 * q01 * q10 * q11 * q13 + m013 * q00 * q11_2 * q13
                + m003 * q01 * q11_2 * q13 + 2 * m013 * q00 * q02 * q12 * q13 - 2 * m113 * q00 * q02 * q12 * q13
                - 2 * m003 * q01 * q02 * q12 * q13 + 2 * m103 * q01 * q02 * q12 * q13 + 2 * m003 * q00 * q03 * q12 * q13
                - 2 * m103 * q00 * q03 * q12 * q13 + 2 * m013 * q01 * q03 * q12 * q13 - 2 * m113 * q01 * q03 * q12 * q13
                - 2 * m013 * q02 * q10 * q12 * q13 - 2 * m003 * q03 * q10 * q12 * q13 + 2 * m003 * q02 * q11 * q12 * q13
                - 2 * m013 * q03 * q11 * q12 * q13 + m013 * q00 * q12_2 * q13 - m003 * q01 * q12_2 * q13
                - 3 * m003 * q00 * q02 * q13_2 + m103 * q00 * q02 * q13_2 - 3 * m013 * q01 * q02 * q13_2 + m113 * q01 * q02 * q13_2
                - m013 * q00 * q03 * q13_2 - m113 * q00 * q03 * q13_2 + m003 * q01 * q03 * q13_2 + m103 * q01 * q03 * q13_2
                + m003 * q02 * q10 * q13_2 - m013 * q03 * q10 * q13_2 + m013 * q02 * q11 * q13_2 + m003 * q03 * q11 * q13_2
                + m003 * q00 * q12 * q13_2 + m013 * q01 * q12 * q13_2 + m013 * q00 * q13_3 - m003 * q01 * q13_3
                + m023
                      * ( -3 * q00_4 - 3 * q01_4 + 3 * q023_2 + 6 * q00_3 * q10 - 4 * q02_2 * q10_2 - 4 * q03_2 * q10_2
                          + 6 * q01_3 * q11 - 4 * q02_2 * q11_2 - 4 * q03_2 * q11_2 - 6 * q02_3 * q12 - 6 * q02 * q03_2 * q12
                          + 2 * q02 * q10_2 * q12 + 2 * q02 * q11_2 * q12 + 3 * q02_2 * q12_2 + q03_2 * q12_2
                          + 2 * q03 * ( -3 * q02_2 - 3 * q03_2 + q10_2 + q11_2 + 2 * q02 * q12 ) * q13 + ( q02_2 + 3 * q03_2 ) * q13_2
                          - q00_2 * ( 6 * q01_2 + 3 * q10_2 - 6 * q01 * q11 + q11_2 + 6 * q02 * q12 - 4 * q12_2 + 6 * q03 * q13 - 4 * q13_2 )
                          - q01_2 * ( q10_2 + 3 * q11_2 + 6 * q02 * q12 - 4 * q12_2 + 6 * q03 * q13 - 4 * q13_2 )
                          + 2 * q01 * q11 * ( 3 * q02_2 + 3 * q03_2 - q12_2 - q13_2 )
                          + 2 * q00 * q10 * ( 3 * q01_2 + 3 * q02_2 + 3 * q03_2 - 2 * q01 * q11 - q12_2 - q13_2 ) )
                + m123
                      * ( 3 * q00_4 + 3 * q01_4 - 3 * q023_2 - 6 * q00_3 * q10 + 2 * q02_2 * q10_2 + 2 * q03_2 * q10_2
                          - 6 * q01_3 * q11 - 4 * q01 * ( q02_2 + q03_2 ) * q11 + 2 * q02_2 * q11_2 + 2 * q03_2 * q11_2
                          - 2 * q00 * q10 * ( 3 * q01_2 + 2 * ( q02_2 + q03_2 ) - 2 * q01 * q11 ) + 6 * q02_3 * q12
                          + 6 * q02 * q03_2 * q12 - 3 * q02_2 * q12_2 - q03_2 * q12_2
                          + 2 * q03 * ( 3 * ( q02_2 + q03_2 ) - 2 * q02 * q12 ) * q13 - ( q02_2 + 3 * q03_2 ) * q13_2
                          + q00_2 * ( 6 * q01_2 + 3 * q10_2 - 6 * q01 * q11 + q11_2 + 4 * q02 * q12 - 2 * ( q12_2 + q13 * ( -2 * q03 + q13 ) ) )
                          + q01_2 * ( q10_2 + 3 * q11_2 + 4 * q02 * q12 - 2 * ( q12_2 + q13 * ( -2 * q03 + q13 ) ) ) ) ) );
    c3[2] = SRNumeratorDerivativeTerm(
        -4
            * ( -2
                    * ( ( m010 - m110 ) * ( ( q01 - q11 ) * ( q02 - q12 ) + ( q00 - q10 ) * ( q03 - q13 ) )
                        + m000 * ( ( q00 - q10 ) * ( q02 - q12 ) - ( q01 - q11 ) * ( q03 - q13 ) )
                        + m100 * ( -( ( q00 - q10 ) * ( q02 - q12 ) ) + ( q01 - q11 ) * ( q03 - q13 ) ) )
                + m020 * ( q00_10_2 + q01_11_2 - q02_12_2 - q03_13_2 ) + m120 * ( -q00_10_2 - q01_11_2 + q02_12_2 + q03_13_2 ) )
            * ( a - q00 * q10 - q01 * q11 - q02 * q12 - q03 * q13 ),
        -4
            * ( -2
                    * ( ( m011 - m111 ) * ( ( q01 - q11 ) * ( q02 - q12 ) + ( q00 - q10 ) * ( q03 - q13 ) )
                        + m001 * ( ( q00 - q10 ) * ( q02 - q12 ) - ( q01 - q11 ) * ( q03 - q13 ) )
                        + m101 * ( -( ( q00 - q10 ) * ( q02 - q12 ) ) + ( q01 - q11 ) * ( q03 - q13 ) ) )
                + m021 * ( q00_10_2 + q01_11_2 - q02_12_2 - q03_13_2 ) + m121 * ( -q00_10_2 - q01_11_2 + q02_12_2 + q03_13_2 ) )
            * ( a - q00 * q10 - q01 * q11 - q02 * q12 - q03 * q13 ),
        -4
            * ( -2
                    * ( ( m012 - m112 ) * ( ( q01 - q11 ) * ( q02 - q12 ) + ( q00 - q10 ) * ( q03 - q13 ) )
                        + m002 * ( ( q00 - q10 ) * ( q02 - q12 ) - ( q01 - q11 ) * ( q03 - q13 ) )
                        + m102 * ( -( ( q00 - q10 ) * ( q02 - q12 ) ) + ( q01 - q11 ) * ( q03 - q13 ) ) )
                + m022 * ( q00_10_2 + q01_11_2 - q02_12_2 - q03_13_2 ) + m122 * ( -q00_10_2 - q01_11_2 + q02_12_2 + q03_13_2 ) )
            * ( a - q00 * q10 - q01 * q11 - q02 * q12 - q03 * q13 ),
        -4 * ( a - q00 * q10 - q01 * q11 - q02 * q12 - q03 * q13 )
            * ( -2 * m003 * q00 * q02 + 2 * m103 * q00 * q02 - 2 * m013 * q01 * q02 + 2 * m113 * q01 * q02
                - 2 * m013 * q00 * q03 + 2 * m113 * q00 * q03 + 2 * m003 * q01 * q03 - 2 * m103 * q01 * q03
                + 2 * m003 * q02 * q10 - 2 * m103 * q02 * q10 + 2 * m013 * q03 * q10 - 2 * m113 * q03 * q10
                + 2 * m013 * q02 * q11 - 2 * m113 * q02 * q11 - 2 * m003 * q03 * q11 + 2 * m103 * q03 * q11
                + 2 * m003 * q00 * q12 - 2 * m103 * q00 * q12 + 2 * m013 * q01 * q12 - 2 * m113 * q01 * q12
                - 2 * m003 * q10 * q12 + 2 * m103 * q10 * q12 - 2 * m013 * q11 * q12 + 2 * m113 * q11 * q12
                + m023 * ( q00_10_2 + q01_11_2 - q02_12_2 - q03_13_2 ) + m123 * ( -q00_10_2 - q01_11_2 + q02_12_2 + q03_13_2 )
                + 2 * m013 * q00 * q13 - 2 * m113 * q00 * q13 - 2 * m003 * q01 * q13 + 2 * m103 * q01 * q13
                - 2 * m013 * q10 * q13 + 2 * m113 * q10 * q13 + 2 * m003 * q11 * q13 - 2 * m103 * q11 * q13 ) );
    c4[2] = SRNumeratorDerivativeTerm(
        ( -2
              * ( ( m010 - m110 ) * ( ( q01 - q11 ) * ( q02 - q12 ) + ( q00 - q10 ) * ( q03 - q13 ) )
                  + m000 * ( ( q00 - q10 ) * ( q02 - q12 ) - ( q01 - q11 ) * ( q03 - q13 ) )
                  + m100 * ( -( ( q00 - q10 ) * ( q02 - q12 ) ) + ( q01 - q11 ) * ( q03 - q13 ) ) )
          + m020 * ( q00_10_2 + q01_11_2 - q02_12_2 - q03_13_2 ) + m120 * ( -q00_10_2 - q01_11_2 + q02_12_2 + q03_13_2 ) )
            * ( b ),
        ( -2
              * ( ( m011 - m111 ) * ( ( q01 - q11 ) * ( q02 - q12 ) + ( q00 - q10 ) * ( q03 - q13 ) )
                  + m001 * ( ( q00 - q10 ) * ( q02 - q12 ) - ( q01 - q11 ) * ( q03 - q13 ) )
                  + m101 * ( -( ( q00 - q10 ) * ( q02 - q12 ) ) + ( q01 - q11 ) * ( q03 - q13 ) ) )
          + m021 * ( q00_10_2 + q01_11_2 - q02_12_2 - q03_13_2 ) + m121 * ( -q00_10_2 - q01_11_2 + q02_12_2 + q03_13_2 ) )
            * ( b ),
        ( -2
              * ( ( m012 - m112 ) * ( ( q01 - q11 ) * ( q02 - q12 ) + ( q00 - q10 ) * ( q03 - q13 ) )
                  + m002 * ( ( q00 - q10 ) * ( q02 - q12 ) - ( q01 - q11 ) * ( q03 - q13 ) )
                  + m102 * ( -( ( q00 - q10 ) * ( q02 - q12 ) ) + ( q01 - q11 ) * ( q03 - q13 ) ) )
          + m022 * ( q00_10_2 + q01_11_2 - q02_12_2 - q03_13_2 ) + m122 * ( -q00_10_2 - q01_11_2 + q02_12_2 + q03_13_2 ) )
            * ( b ),
        ( b )
            * ( -2 * m003 * q00 * q02 + 2 * m103 * q00 * q02 - 2 * m013 * q01 * q02 + 2 * m113 * q01 * q02
                - 2 * m013 * q00 * q03 + 2 * m113 * q00 * q03 + 2 * m003 * q01 * q03 - 2 * m103 * q01 * q03
                + 2 * m003 * q02 * q10 - 2 * m103 * q02 * q10 + 2 * m013 * q03 * q10 - 2 * m113 * q03 * q10
                + 2 * m013 * q02 * q11 - 2 * m113 * q02 * q11 - 2 * m003 * q03 * q11 + 2 * m103 * q03 * q11
                + 2 * m003 * q00 * q12 - 2 * m103 * q00 * q12 + 2 * m013 * q01 * q12 - 2 * m113 * q01 * q12
                - 2 * m003 * q10 * q12 + 2 * m103 * q10 * q12 - 2 * m013 * q11 * q12 + 2 * m113 * q11 * q12
                + m023 * ( q00_10_2 + q01_11_2 - q02_12_2 - q03_13_2 ) + m123 * ( -q00_10_2 - q01_11_2 + q02_12_2 + q03_13_2 )
                + 2 * m013 * q00 * q13 - 2 * m113 * q00 * q13 - 2 * m003 * q01 * q13 + 2 * m103 * q01 * q13
                - 2 * m013 * q10 * q13 + 2 * m113 * q10 * q13 + 2 * m003 * q11 * q13 - 2 * m103 * q11 * q13 ) );
}

}  // namespace motion
}  // namespace optix_exp
