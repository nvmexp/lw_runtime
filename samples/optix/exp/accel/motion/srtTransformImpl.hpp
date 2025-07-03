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

#include "motionCommon.hpp"
#include "matrixTransformImpl.hpp"
#include "srTransformHelper.h"
#include <cmath>

// force to output a single AABB per SRT interval instead of an AABB at the begin and an AABB at the end of the interval
//#define FORCE_STATIC_OUTPUT_AABB

namespace optix_exp {
namespace motion {

template<bool applyTranslation = true>
static M_DEVICE_HOST void srtToMatrix( const SRTData& srt, float* m )
{
    const float4 q = make_float4( srt.qx, srt.qy, srt.qz, srt.qw );

    // normalize
    const float  ilw_sql = 1.f / ( srt.qx * srt.qx + srt.qy * srt.qy + srt.qz * srt.qz + srt.qw * srt.qw );
    const float4 nq      = make_float4( q.x * ilw_sql, q.y * ilw_sql, q.z * ilw_sql, q.w * ilw_sql );

    const float sqw = q.w * nq.w;
    const float sqx = q.x * nq.x;
    const float sqy = q.y * nq.y;
    const float sqz = q.z * nq.z;

    const float xy = q.x * nq.y;
    const float zw = q.z * nq.w;
    const float xz = q.x * nq.z;
    const float yw = q.y * nq.w;
    const float yz = q.y * nq.z;
    const float xw = q.x * nq.w;

    m[0] = ( sqx - sqy - sqz + sqw );
    m[1] = 2.0f * ( xy - zw );
    m[2] = 2.0f * ( xz + yw );

    m[4] = 2.0f * ( xy + zw );
    m[5] = ( -sqx + sqy - sqz + sqw );
    m[6] = 2.0f * ( yz - xw );

    m[8]  = 2.0f * ( xz - yw );
    m[9]  = 2.0f * ( yz + xw );
    m[10] = ( -sqx - sqy + sqz + sqw );

    m[3]  = m[0] * srt.pvx + m[1] * srt.pvy + m[2] * srt.pvz;
    m[7]  = m[4] * srt.pvx + m[5] * srt.pvy + m[6] * srt.pvz;
    m[11] = m[8] * srt.pvx + m[9] * srt.pvy + m[10] * srt.pvz;

    m[2]  = m[0] * srt.b + m[1] * srt.c + m[2] * srt.sz;
    m[6]  = m[4] * srt.b + m[5] * srt.c + m[6] * srt.sz;
    m[10] = m[8] * srt.b + m[9] * srt.c + m[10] * srt.sz;

    m[1] = m[0] * srt.a + m[1] * srt.sy;
    m[5] = m[4] * srt.a + m[5] * srt.sy;
    m[9] = m[8] * srt.a + m[9] * srt.sy;

    m[0] = m[0] * srt.sx;
    m[4] = m[4] * srt.sx;
    m[8] = m[8] * srt.sx;

    if( applyTranslation )
    {
        m[3]  += srt.tx;
        m[7]  += srt.ty;
        m[11] += srt.tz;
    }
}

namespace {

    M_DEVICE_HOST __inline__ SRTData lerpNonNormalizedQuaternion( const SRTData& key0, const SRTData& key1, float t )
    {
        SRTData r;
        r.sx  = optix_exp::motion::lerp( key0.sx , key1.sx , t );
        r.a   = optix_exp::motion::lerp( key0.a  , key1.a  , t );
        r.b   = optix_exp::motion::lerp( key0.b  , key1.b  , t );
        r.pvx = optix_exp::motion::lerp( key0.pvx, key1.pvx, t );
        r.sy  = optix_exp::motion::lerp( key0.sy , key1.sy , t );
        r.c   = optix_exp::motion::lerp( key0.c  , key1.c  , t );
        r.pvy = optix_exp::motion::lerp( key0.pvy, key1.pvy, t );
        r.sz  = optix_exp::motion::lerp( key0.sz , key1.sz , t );
        r.pvz = optix_exp::motion::lerp( key0.pvz, key1.pvz, t );
        r.qx  = optix_exp::motion::lerp( key0.qx , key1.qx , t );
        r.qy  = optix_exp::motion::lerp( key0.qy , key1.qy , t );
        r.qz  = optix_exp::motion::lerp( key0.qz , key1.qz , t );
        r.qw  = optix_exp::motion::lerp( key0.qw , key1.qw , t );
        r.tx  = optix_exp::motion::lerp( key0.tx , key1.tx , t );
        r.ty  = optix_exp::motion::lerp( key0.ty , key1.ty , t );
        r.tz  = optix_exp::motion::lerp( key0.tz , key1.tz , t );

        return r;
    }

    struct Quaternion
    {
        /** quaternion x, y, z, w */
        // cannot use float4, to avoid alignment issues (causing padding) when used as part of SRT data
        float m_q[4];

        M_DEVICE_HOST __inline__ float3 rotate( const float3& v ) const;
        M_DEVICE_HOST __inline__ float3 rotateNonNormalized( const float3& v ) const;
        M_DEVICE_HOST __inline__ void normalize();
        M_DEVICE_HOST __inline__ float* data() { return m_q; }
        M_DEVICE_HOST __inline__ const float* data() const { return m_q; }

        M_DEVICE_HOST __inline__ static Quaternion nlerp( const Quaternion& q0, const Quaternion& q1, float t );
        M_DEVICE_HOST __inline__ static Quaternion nlerp( const float q0[4], const float q1[4], float t );

    };

    M_DEVICE_HOST void Quaternion::normalize()
    {
        float ilwLen = 1.0f / sqrtf( m_q[0] * m_q[0] + m_q[1] * m_q[1] + m_q[2] * m_q[2] + m_q[3] * m_q[3] );
        m_q[0] *= ilwLen;
        m_q[1] *= ilwLen;
        m_q[2] *= ilwLen;
        m_q[3] *= ilwLen;
    }

    M_DEVICE_HOST Quaternion Quaternion::nlerp( const Quaternion& quat0, const Quaternion& quat1, float t )
    {
        Quaternion q = {lerp( quat0.m_q[0], quat1.m_q[0], t ), lerp( quat0.m_q[1], quat1.m_q[1], t ),
                        lerp( quat0.m_q[2], quat1.m_q[2], t ), lerp( quat0.m_q[3], quat1.m_q[3], t )};
        q.normalize();
        return q;
    }

    M_DEVICE_HOST Quaternion Quaternion::nlerp( const float q0[4], const float q1[4], float t )
    {
        Quaternion q ={ lerp( q0[0], q1[0], t ), lerp( q0[1], q1[1], t ),
                        lerp( q0[2], q1[2], t ), lerp( q0[3], q1[3], t )};
        q.normalize();
        return q;
    }

    M_DEVICE_HOST float3 Quaternion::rotate( const float3& v ) const
    {
        const float3 u = make_float3( m_q[0], m_q[1], m_q[2] );
        const float s = m_q[3];

        const float udotv = u.x * v.x + u.y * v.y + u.z * v.z;
        const float udotu = u.x * u.x + u.y * u.y + u.z * u.z;
        const float3 cross = make_float3( u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x );

        const float3 vprime =
            (2.f * udotv) * u
            + (s*s - udotu) * v
            + (2.f * s) * cross;

        return vprime;
    }

    // this variant avoids a sqrt since the normalization factor ends up squared in the result (i.e., as a squared coefficient)
    M_DEVICE_HOST float3 Quaternion::rotateNonNormalized( const float3& v ) const
    {
        const float3 u = make_float3( m_q[0], m_q[1], m_q[2] );
        const float s = m_q[3];

        const float udotv = u.x * v.x + u.y * v.y + u.z * v.z;
        const float udotu = u.x * u.x + u.y * u.y + u.z * u.z;
        const float3 cross = make_float3( u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x );

        const float ilwSqrLen = 1.0f / ( m_q[0] * m_q[0] + m_q[1] * m_q[1] + m_q[2] * m_q[2] + m_q[3] * m_q[3] );
        const float3 vprime =
            ilwSqrLen * (2.f * udotv) * u
            + ilwSqrLen * (s*s - udotu) * v
            + ilwSqrLen * (2.f * s) * cross;

        return vprime;
    }

    M_DEVICE_HOST __inline__ void applyTranslationfromSRT( const SRTData& key, Aabb& inoutAabb )
    {
        inoutAabb.min.x += key.tx;
        inoutAabb.min.y += key.ty;
        inoutAabb.min.z += key.tz;
        inoutAabb.max.x += key.tx;
        inoutAabb.max.y += key.ty;
        inoutAabb.max.z += key.tz;
    }

    M_DEVICE_HOST __inline__ void extractScaleFromSRT( const SRTData& srt, float* m )
    {
        m[0]  = srt.sx;
        m[1]  = srt.a;
        m[2]  = srt.b;
        m[3]  = srt.pvx;
        m[4]  = 0;
        m[5]  = srt.sy;
        m[6]  = srt.c;
        m[7]  = srt.pvy;
        m[8]  = 0;
        m[9]  = 0;
        m[10] = srt.sz;
        m[11] = srt.pvz;
    }

    struct SRTQuatData
    {
        float      s[9];
        Quaternion q;
        float      t[3];
    };
    static_assert( sizeof( SRTQuatData ) == sizeof( SRTData ), "sizeof( SRTQuatData ) != sizeof( SRTData" );


    M_DEVICE_HOST __inline__ float3 evalSRNonNormalized( const SRTData& srt, const float3& p )
    {
        float3 v;
        v.x = p.x * srt.sx + p.y * srt.a + p.z * srt.b + srt.pvx;
        v.y = p.y * srt.sy + p.z * srt.c + srt.pvy;
        v.z = p.z * srt.sz + srt.pvz;

        const float3 u = make_float3( srt.qx, srt.qy, srt.qz );
        const float s  = srt.qw;

        const float udotv = u.x * v.x + u.y * v.y + u.z * v.z;
        const float udotu = u.x * u.x + u.y * u.y + u.z * u.z;
        const float3 cross = make_float3( u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x );

        const float ilwSqrLen = 1.0f / ( udotu + s*s );
        const float3 vprime =
            ilwSqrLen * (2.f * udotv) * u
            + ilwSqrLen * (s*s - udotu) * v
            + ilwSqrLen * (2.f * s) * cross;

        return vprime;
    }

    M_DEVICE_HOST __inline__ float3 eval_sr( const SRTData& srt, const float3& p )
    {
        float3 r;
        r.x = p.x * srt.sx + p.y * srt.a + p.z * srt.b + srt.pvx;
        r.y = p.y * srt.sy + p.z * srt.c + srt.pvy;
        r.z = p.z * srt.sz + srt.pvz;

        const SRTQuatData& data = reinterpret_cast<const SRTQuatData&>(srt);
        return data.q.rotate( r );
    }


    M_DEVICE_HOST void includeSRCornerPath( Aabb&             diffAabb,
                                            const Aabb&       refAabb0,
                                            const Aabb&       refAabb1,
                                            const float3&     p,
                                            const SRTData&    key0,
                                            const SRTData&    key1,
                                            SRNumeratorDerivativeTerm c0[],
                                            SRNumeratorDerivativeTerm c1[],
                                            SRNumeratorDerivativeTerm c2[],
                                            SRNumeratorDerivativeTerm c3[],
                                            SRNumeratorDerivativeTerm c4[],
                                            const float denomCoeffs[5])
    {
        // Specific input point p (e.g., one corner of an aabb) to be transformed

#ifdef FORCE_STATIC_OUTPUT_AABB

#ifdef __LWDACC__
#pragma unroll 1
#endif
        for( int k = 0; k < 3; ++k )
        {
            // Evaluate coefficients for axis k with given input point.
            // 5: coefficients of polynomial c(t)
            float coeffs[5];
            coeffs[0] = c0[k].eval( p );
            coeffs[1] = c1[k].eval( p );
            coeffs[2] = c2[k].eval( p );
            coeffs[3] = c3[k].eval( p );
            coeffs[4] = c4[k].eval( p );
            float roots[4];

            // root finding for current axis
            const int numroots = findRealRoots4( coeffs, roots );
            //min / max
#ifdef __LWDACC__
#pragma unroll 1
#endif
            for( int j = 0; j < 2; ++j )
            {
                const int idx = j*3 + k;
                // Evaluate at roots that are inside segment time interval
                for( int i = 0; i < numroots; ++i )
                {
                    if( roots[i] > 0 && roots[i] < 1 )
                    {
                        // evaluate RS*p at this root and update min and max for this axis
                        float3 pt   = evalSRNonNormalized( lerpNonNormalizedQuaternion( key0, key1, roots[i] ), p );
                        float  diff = ( &pt.x )[k] - refAabb0.minMax[idx];
                        diffAabb.minMax[idx] = j == 0 ? ::min( diffAabb.minMax[idx], diff ) : ::max( diffAabb.minMax[idx], diff );
                    }
                }
            }
        }

#else // FORCE_STATIC_OUTPUT_AABB

#ifdef __LWDACC__
#pragma unroll 1
#endif
        for( int k = 0; k < 3; ++k )
        {
            float coeffs[5];
            coeffs[0] = c0[k].eval( p );
            coeffs[1] = c1[k].eval( p );
            coeffs[2] = c2[k].eval( p );
            coeffs[3] = c3[k].eval( p );
            coeffs[4] = c4[k].eval( p );
            float roots[4];

            //min / max
#ifdef __LWDACC__
#pragma unroll 1
#endif
            for( int j = 0; j < 2; ++j )
            {
                const int idx = j*3 + k;
                float grad = refAabb1.minMax[idx] - refAabb0.minMax[idx];
                // given a reference linear function ('gradient', g(t) here), compute the coefficients of the derivate of f(p,t) - g(t)
                // to be more precise, the derivate f'(p,t) is a rational polynomial in t with degree 4 in the nominator and denominator like:
                // f'(p,t)          = ( c4 t^4 + c3 t^3 + c2 t^2 + c1 t + c0 ) / ( d4 t^4 + d3 t^3 + d2 t^2 + d1 t + d0 )
                // (f(p,t) - g(t))' = ( c4 t^4 + c3 t^3 + c2 t^2 + c1 t + c0 ) / ( d4 t^4 + d3 t^3 + d2 t^2 + d1 t + d0 ) - g'(t)
                // if g(t) is constant, g'(t) is zero (the case used in FORCE_STATIC_OUTPUT_AABB)
                // if g(t) is linear, g'(t) is a constant
                //
                // We only need to solve for the roots of the derivative (setting 0 = (f(p,t) - g(t))'), hence, we simply multiply by ( d4 t^4 + d3 t^3 + d2 t^2 + d1 t + d0 )
                // and end up with:
                // g'(t) = e
                // (f(p,t) - g(t))' = (c4 - d4*e) t^4 + (c3 - d3*e) t^3 + (c2 - d2*e) t^2 + (c1 - d1*e) t + c0 - d0*e = 0
                float coeffsGrad[5];
                coeffsGrad[0] = coeffs[0] - grad * denomCoeffs[0];
                coeffsGrad[1] = coeffs[1] - grad * denomCoeffs[1];
                coeffsGrad[2] = coeffs[2] - grad * denomCoeffs[2];
                coeffsGrad[3] = coeffs[3] - grad * denomCoeffs[3];
                coeffsGrad[4] = coeffs[4] - grad * denomCoeffs[4];
                const int numroots = findRealRoots4( coeffsGrad, roots );
                // Evaluate at roots that are inside segment time interval
                for( int i = 0; i < numroots; ++i )
                {
                    if( roots[i] > 0 && roots[i] < 1 )
                    {
                        // evaluate RS*p at this root and update min and max for this axis
                        float3 pt   = evalSRNonNormalized( lerpNonNormalizedQuaternion( key0, key1, roots[i] ), p );
                        float  ref  = lerp( refAabb0.minMax[idx], refAabb1.minMax[idx], roots[i] );
                        float  diff = ( &pt.x )[k] - ref;
                        diffAabb.minMax[idx] = j == 0 ? ::min( diffAabb.minMax[idx], diff ) : ::max( diffAabb.minMax[idx], diff );
                    }
                }
            }
        }

#endif // FORCE_STATIC_OUTPUT_AABB
    }

    // static input aabb
    // NO translation
    // interpolated srt: key0 -> key1
    // transform aabb by interpolated SR
    // result: aabb of interpolated srt applied on input aabb within interval [t0, t1]
    M_DEVICE_HOST void transformSR( const Aabb& aabb, const SRTData& key0, const SRTData& key1, Aabb& outAabb0, Aabb& outAabb1 )
    {
        // this function should only apply the scale and rotation, but not the translation!
        // this happens outside of this function
        float key0_matrix[12], key1_matrix[12];
        srtToMatrix<false>( key0, key0_matrix );
        srtToMatrix<false>( key1, key1_matrix );
        outAabb0 = transform( aabb, key0_matrix );
        outAabb1 = transform( aabb, key1_matrix );
#ifdef COMP
#error need to define COMP
#endif
#define COMP( comp ) ( key1.comp == key0.comp )
        // if no rotation, use matrix computation!
        // check delta of rotation
        if( COMP( qx ) && COMP( qy ) && COMP( qz ) && COMP( qw ) )
            return;
#undef COMP

        // outAabb0 and outAabb1 are our baseline aabbs now
        // we assume a linear interpolation between those two and see if the interpolated SRT has extrema bigger than the linear interpolation

        SRNumeratorDerivativeTerm c0[3], c1[3], c2[3], c3[3], c4[3];
        float denomCoeffs[5];
        makeSRDerivativeTerms( key0, key1, c0, c1, c2, c3, c4, denomCoeffs );

        // 8 corner points of axis-aligned bounding box
        float3 v[8];
        extractCornersFromAabb( v, aabb );

        //////////////////////////////////////////////////////////////////////////
        Aabb differenceAabb ={}; // init to zero

#ifdef __LWDACC__
#pragma unroll 1
#endif
        for( float3& vert : v )
        {
            includeSRCornerPath( differenceAabb, outAabb0, outAabb1, vert, key0, key1, c0, c1, c2, c3, c4, denomCoeffs );
        }

        // Pad bounds to account for error in root finding.
        differenceAabb.min = differenceAabb.min - 0.00001f * ( differenceAabb.max - differenceAabb.min );
        differenceAabb.max = differenceAabb.max + 0.00001f * ( differenceAabb.max - differenceAabb.min );

        // min is <=0, max is >=0
        outAabb0.min += differenceAabb.min;
        outAabb0.max += differenceAabb.max;

        outAabb1.min += differenceAabb.min;
        outAabb1.max += differenceAabb.max;

#ifdef FORCE_STATIC_OUTPUT_AABB
        outAabb1 = outAabb0;
#endif
        //////////////////////////////////////////////////////////////////////////
    }

    // transformSR transforms moving input aabbs, the interpolation between aabbb0 and aabb1,
    // with an SR transform specified by key0 and key1. T is handled separately outside this function.
    //
    // interpolated srt: key0 at time 0 -> key1 at time 1
    // interpolated input aabb: aabb0 is at time 0, aabb1 is at time 1
    // result: aabb of interpolated srt applied on input aabb within time interval [0, 1]
    M_DEVICE_HOST void transformSR( const Aabb& aabb0, const Aabb& aabb1, const SRTData& key0, const SRTData& key1, Aabb& outAabb0, Aabb& outAabb1 )
    {
#ifdef COMP
#error need to define COMP
#endif
#define COMP( comp ) ( key1.comp == key0.comp )
        // if no rotation, use matrix computation!
        // check delta of rotation
        if( COMP( qx ) && COMP( qy ) && COMP( qz ) && COMP( qw ) )
        {
            float key0_matrix[12], key1_matrix[12];
            // this function should only apply the scale and rotation, but not the translation!
            // this happens outside of this function
            srtToMatrix<false>( key0, key0_matrix );
            srtToMatrix<false>( key1, key1_matrix );
            transform( aabb0, aabb1, key0_matrix, key1_matrix, outAabb0, outAabb1 );
        }
        // check if we can bake the movement of the input aabbs into the transforms
        // this is possible if the delta of the transforms has no scale/shear
        // if not, right now, we can only compute the aabb for a static input, hence, the union of aabb0 and aabb1
        else if( COMP( sx ) && COMP( a ) && COMP( b ) && COMP( sy ) && COMP( c ) && COMP( sz ) )
#undef COMP
        {
            auto bakeAabbIntoTransform = []( const SRTData& key, const Aabb& aabb ) -> SRTData
            {
                float3 scaleAabb = ( aabb.max - aabb.min ) * 0.5f;
                float3 centerAabb = center( aabb );

                SRTData key_prime = key;
                key_prime.sx *= scaleAabb.x;
                key_prime.a  *= scaleAabb.y;
                key_prime.sy *= scaleAabb.y;
                key_prime.b  *= scaleAabb.z;
                key_prime.c  *= scaleAabb.z;
                key_prime.sz *= scaleAabb.z;
                key_prime.pvx += key.sx * centerAabb.x + key.a * centerAabb.y + key.b * centerAabb.z;
                key_prime.pvy += key.sy * centerAabb.y + key.c * centerAabb.z;
                key_prime.pvz += key.sz * centerAabb.z;

                return key_prime;
            };

            // alternative to using the unit aabb would be using aabb0, an interpolated box or aabb1 as reference.
            // e.g., when using aabb0, only key1 needs to be adjusted
            // however caution is required when computing the scale since the aabbs may be degenerate (length of one axis == 0)
            Aabb unitAabb = {-1, -1, -1, 1, 1, 1};
            transformSR( unitAabb, bakeAabbIntoTransform( key0, aabb0 ), bakeAabbIntoTransform( key1, aabb1 ), outAabb0, outAabb1 );
        }
        else
        {
            // Applying S on a moving input gives quadratic behavior in time t.
            // However, the SRT algorithm is lwrrently (8th of April 2020) only designed to handle linear input, i.e.,
            // applying an SRT on a static input AABB or an SRT with a no scale/shear component on a linearly moving AABB (the if case above).
            // We can linearize the problem by applying the matrix transform (that of S) on the moving input AABB and express resulting moving AABB as a delta in S from key0 to key1.
            float scale0[12], scale1[12];
            extractScaleFromSRT( key0, scale0 );
            extractScaleFromSRT( key1, scale1 );

            Aabb aabb0_prime, aabb1_prime;
            transform( aabb0, aabb1, scale0, scale1, aabb0_prime, aabb1_prime );

            // we only keep the R component (the quaternion), S is completely replaced by the AABB
            auto replaceSbyAABB = []( const SRTData& key, const Aabb& aabb ) -> SRTData
            {
                float3 scaleAabb = ( urb( aabb ) - llf( aabb ) ) * 0.5f;
                float3 centerAabb = center( aabb );

                return { scaleAabb.x, 0, 0, centerAabb.x, scaleAabb.y, 0, centerAabb.y, scaleAabb.z, centerAabb.z, key.qx, key.qy, key.qz, key.qw, 0, 0, 0 };
            };

            Aabb unitAabb = {-1, -1, -1, 1, 1, 1};
            transformSR( unitAabb, replaceSbyAABB( key0, aabb0_prime ), replaceSbyAABB( key1, aabb1_prime ), outAabb0, outAabb1 );
        }
    }
} // namespace

M_DEVICE_HOST __inline__ SRTData lerp( const SRTData& key0, const SRTData& key1, float t )
{
    const SRTQuatData& k0 = reinterpret_cast<const SRTQuatData&>(key0);
    const SRTQuatData& k1 = reinterpret_cast<const SRTQuatData&>(key1);
    SRTQuatData r;
    r.s[0] = lerp( k0.s[0], k1.s[0], t);
    r.s[1] = lerp( k0.s[1], k1.s[1], t);
    r.s[2] = lerp( k0.s[2], k1.s[2], t);
    r.s[3] = lerp( k0.s[3], k1.s[3], t);
    r.s[4] = lerp( k0.s[4], k1.s[4], t);
    r.s[5] = lerp( k0.s[5], k1.s[5], t);
    r.s[6] = lerp( k0.s[6], k1.s[6], t);
    r.s[7] = lerp( k0.s[7], k1.s[7], t);
    r.s[8] = lerp( k0.s[8], k1.s[8], t);
    r.q = Quaternion::nlerp( {key0.qx, key0.qy, key0.qz, key0.qw}, {key1.qx, key1.qy, key1.qz, key1.qw}, t );
    r.t[0] = lerp( k0.t[0], k1.t[0], t);
    r.t[1] = lerp( k0.t[1], k1.t[1], t);
    r.t[2] = lerp( k0.t[2], k1.t[2], t);

    return *reinterpret_cast<SRTData*>(&r);
}

M_DEVICE_HOST __inline__ Aabb transform( const Aabb& aabb, const SRTData& key )
{
    float m[12];
    srtToMatrix( key, m );

    return transform( aabb, m );
}

M_DEVICE_HOST __inline__ void transform( const Aabb& aabb, const SRTData& key0, const SRTData& key1, Aabb& outAabb0, Aabb& outAabb1 )
{
    transformSR( aabb, key0, key1, outAabb0, outAabb1 );
    // factored translation
    applyTranslationfromSRT( key0, outAabb0 );
    applyTranslationfromSRT( key1, outAabb1 );
}

M_DEVICE_HOST __inline__ void transform( const Aabb& aabb0, const Aabb& aabb1, const SRTData& key0, const SRTData& key1, Aabb& outAabb0, Aabb& outAabb1 )
{
    if( aabb0.min == aabb1.min && aabb0.max == aabb1.max )
        transformSR( aabb0, key0, key1, outAabb0, outAabb1 );
    else
        transformSR( aabb0, aabb1, key0, key1, outAabb0, outAabb1 );
    // factored translation
    applyTranslationfromSRT( key0, outAabb0 );
    applyTranslationfromSRT( key1, outAabb1 );
}

M_DEVICE_HOST __inline__ void transformInOut0Out1( const Aabb& aabb0, const Aabb& aabb1, const SRTData& key0, const SRTData& key1, Aabb& inOutAabb0, Aabb& outAabb1 )
{
    // transformation of the motion aabb at time 0 and time 1
    // inOutAabb0 is assumed to include transform( aabb0, key0 )
    //inOutAabb0 = transform( aabb0, key0 );
    outAabb1 = transform( aabb1, key1 );

    //TODO: transform should make use of inoutAabbs
    Aabb outAabb0;
    transform( aabb0, aabb1, key0, key1, outAabb0, outAabb1 );
    include( inOutAabb0, outAabb0 );
}

M_DEVICE_HOST __inline__ void transformInOut0InOut1( const Aabb& aabb0, const Aabb& aabb1, const SRTData& key0, const SRTData& key1, Aabb& inOutAabb0, Aabb& inOutAabb1 )
{
    // transformation of the motion aabb at time 0 and time 1
    // inOutAabb0 is assumed to include transform( aabb0, key0 )
    // inOutAabb1 is assumed to include transform( aabb1, key1 )
    //inOutAabb0 = transform( aabb0, key0 );
    //inOutAabb1 = transform( aabb1, key1 );

    //TODO: transform should make use of inoutAabbs
    Aabb outAabb0, outAabb1;
    transform( aabb0, aabb1, key0, key1, outAabb0, outAabb1 );
    include( inOutAabb0, outAabb0 );
    include( inOutAabb1, outAabb1 );
}

}  // namespace motion
}  // namespace optix_exp
