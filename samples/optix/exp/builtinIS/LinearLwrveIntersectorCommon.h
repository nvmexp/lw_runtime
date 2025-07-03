/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#pragma once

#include "BuiltinISCommon.h"

namespace optix_exp {

__device__ __forceinline__ float flipsign( float v, bool flag )
{
    return ( flag ) ? -v : v;
}

static __device__ __forceinline__ Intersection intersect_linear( float3 rayOrig,
                                                                 float3 D,  // rayDir
                                                                 float3 P0,
                                                                 float3 P1,
                                                                 float  r0,
                                                                 float  r1 )  // input lwrve
{
    /* Naming colwentions:
       single upper case names are vector( float3 )
       single lower case names are scalar
       double upper case names are scalar dot product AD = A.D */

    // Use the midpoint to anchor everything - this gives the best precision balance
    const float3 Pm = 0.5f * ( P0 + P1 );
    const float  rm = 0.5f * ( r0 + r1 );

    // a is our first conic coefficient for the t^2 term
    const float a     = dot3( D, D );
    const float rcp_a = 1.f / a;

    // precision fix -- push the ray to be as close as possible to the lwrve.
    const float dpa = dot3( D, Pm - rayOrig ) * rcp_a;
    rayOrig         = rayOrig + dpa * D;
    const float3 O  = rayOrig - Pm;

    const float3 A  = P1 - P0;  // lwrve axis
    const float  rd = r1 - r0;  // radius delta

    // precompute some sub-expressions we need
    const float AD = dot3( A, D );
    const float AA = dot3( A, A );
    const float DO = dot3( D, O );
    const float AO = dot3( A, O );
    const float OO = dot3( O, O );

    // Compute the conic coefficients.
    // lwrve: C = Pm + u A, r = rm + u rd
    //
    // We'll set:
    // Pm = 1/2 (P0 + P1)
    // rm = 1/2 (r0 + r1)
    // O = rayOrig - Pm
    //
    // ray: P = rayOrig + t rayDir
    //
    // sphere: |P-C| = r
    //
    // So our cone equation is: | ray(t) - lwrve(u) | = radius(u)
    //
    // Expanding:
    // | rayOrig + t V - Pm - u A |^2 = (rm + u rd)^2
    // Gives us:
    // ttV.V - 2tuA.V + uu(A.A - rd rd) - 2t(-O.V) - 2u (A.O + rd rm) + O.O - rm rm = 0
    //
    // Our canonical conic equation is at^2 - 2btu + lw^2 - 2dt - 2eu + f = 0
    // The single letter variables a through f below are these coefficients
    // a is already up top
    const float b = AD;
    const float c = AA - rd * rd;
    const float d = -DO;
    const float e = AO + rm * rd;
    const float f = OO - rm * rm;

    // Now solve our conic for u(t):
    // att - 2but + lwu - 2dt - 2eu + f = 0
    // Group the u terms and look for a quadratic in u: Au u^2 -2 Bu u + Lw
    // + lwu - 2u(bt + e) + (att - 2dt + f) = 0
    //
    // Au = c
    // Bu = bt + e
    // Lw = att - 2dt + f
    //
    // u(t) = [ Bu +/- sqrt(BuBu - AuLw) ] / Au
    //
    // solve for discrim = 0 - we are looking for the extremal values of u,
    // which happen to coincide with the t values we're looking for!
    // so, Bu Bu - Au Lw = 0
    // tt(bb - ca) -2t(-be -cd) + (ee - cf) = 0
    //
    // At = bb-ca
    // Bt = -be-cd
    // Ct = ee-cf
    // t = [ Bt +/- sqrt(BtBt-AtCt) ] / At

    const float At = b * b - a * c;
    const float Bt = b * e + c * d;  // Using -Bt makes it prettier
    const float Ct = e * e - c * f;

    const float BtBt = Bt * Bt;
    const float AtCt = At * Ct;

    const bool badLwrve = ( r0 <= 0.f && r1 <= 0.f ); // optional: filter bad data
    // Infinite cone MISS
    if( BtBt < AtCt || badLwrve )
        // no intersection
        return Intersection();

    const float sqdiscrim = sqrtf( BtBt - AtCt );
    const float rcpAt     = 1.f / At;
    const float t0        = ( -Bt + sqdiscrim ) * rcpAt;  // t0 == cone hit point

    // Callwlate u (parametric length along lwrve)
    // u is in range [-.5, .5] when on the mid-section between endcaps
    const float rcp_c = 1.f / c;
    float       u0    = ( b * t0 + e ) * rcp_c;

    const bool radiusIsNegative = u0 * rd < -rm;

    // Mid-section cone HIT
    if( fabsf( u0 ) <= 0.5f && !radiusIsNegative )
    {
        return Intersection( t0 + dpa, u0 + 0.5f );
    }

    // Determine which end-cap to test
    bool useP0 = ( u0 < 0.f ) ^ radiusIsNegative;

    // What follows is a simple sphere intersection with the appropriate end-cap sphere,
    // substituting values we already have anywhere possible.

    // const float endu  = (useP0) ? -0.5f : 0.5f; // endcap u
    // const float endr  = (useP0) ? r0 : r1;      // endcap r
    const float Bs = d + flipsign( b, useP0 ) * 0.5f;  // Bs = dot(E, D) ==> dot(O,D) - dot(A,D) * endu, where E = O - A * endu;
    const float BsBs = Bs * Bs;
    const float Cs   = f + 0.25f * c - flipsign( e, useP0 );  // dot(E, E) - endr*endr;
    const float AsCs = a * Cs;

    // end-cap MISS
    if( BsBs < AsCs )
        // no intersection
        return Intersection();

    const float sqSdiscrim = sqrtf( BsBs - AsCs );
    const float min_endt   = ( Bs - sqSdiscrim ) * rcp_a;

    // end-cap HIT
    return Intersection( min_endt + dpa, ( useP0 ) ? 0.f : 1.f );
}

}  // namespace optix_exp
