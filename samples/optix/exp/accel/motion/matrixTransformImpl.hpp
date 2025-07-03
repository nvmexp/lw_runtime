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

namespace optix_exp {
namespace motion {

M_DEVICE_HOST __inline__ Matrix3x4 lerp( const Matrix3x4& a, const Matrix3x4& b, float t )
{
    Matrix3x4 r = {{lerp( a.m[0], b.m[0], t ), lerp( a.m[1], b.m[1], t ), lerp( a.m[2], b.m[2], t ),
                    lerp( a.m[3], b.m[3], t ), lerp( a.m[4], b.m[4], t ), lerp( a.m[5], b.m[5], t ),
                    lerp( a.m[6], b.m[6], t ), lerp( a.m[7], b.m[7], t ), lerp( a.m[8], b.m[8], t ),
                    lerp( a.m[9], b.m[9], t ), lerp( a.m[10], b.m[10], t ), lerp( a.m[11], b.m[11], t )}};
    return r;
}

M_DEVICE_HOST __inline__ Aabb transform( const Aabb& aabb, const float* m )
{
    // Column vectors
    //x ={ m[0], m[4], m[8] }
    //y ={ m[1], m[5], m[9] }
    //z ={ m[2], m[6], m[10] }
    // 3,7,11 translation

    // no need to initialize, will be overwritten completely
    Aabb result;

    const float loxx = m[0] * aabb.min.x;
    const float hixx = m[0] * aabb.max.x;

    const float loyx = m[1] * aabb.min.y;
    const float hiyx = m[1] * aabb.max.y;

    const float lozx = m[2] * aabb.min.z;
    const float hizx = m[2] * aabb.max.z;

    result.min.x = ::min( loxx, hixx ) + ::min( loyx, hiyx ) + ::min( lozx, hizx ) + m[3];
    result.max.x = ::max( loxx, hixx ) + ::max( loyx, hiyx ) + ::max( lozx, hizx ) + m[3];

    const float loxy = m[4] * aabb.min.x;
    const float hixy = m[4] * aabb.max.x;

    const float loyy = m[5] * aabb.min.y;
    const float hiyy = m[5] * aabb.max.y;

    const float lozy = m[6] * aabb.min.z;
    const float hizy = m[6] * aabb.max.z;

    result.min.y = ::min( loxy, hixy ) + ::min( loyy, hiyy ) + ::min( lozy, hizy ) + m[7];
    result.max.y = ::max( loxy, hixy ) + ::max( loyy, hiyy ) + ::max( lozy, hizy ) + m[7];

    const float loxz = m[8] * aabb.min.x;
    const float hixz = m[8] * aabb.max.x;

    const float loyz = m[9] * aabb.min.y;
    const float hiyz = m[9] * aabb.max.y;

    const float lozz = m[10] * aabb.min.z;
    const float hizz = m[10] * aabb.max.z;

    result.min.z = ::min( loxz, hixz ) + ::min( loyz, hiyz ) + ::min( lozz, hizz ) + m[11];
    result.max.z = ::max( loxz, hixz ) + ::max( loyz, hiyz ) + ::max( lozz, hizz ) + m[11];

    return result;
}

M_DEVICE_HOST __inline__ Aabb transform( const Aabb& aabb, const Matrix3x4& m )
{
    return transform( aabb, m.m );
}

// Given an aabb, extract its 8 corner points.
M_DEVICE_HOST __inline__ void extractCornersFromAabb( float3* v, const Aabb& aabb )
{
    v[0] = make_float3( aabb.min.x, aabb.min.y, aabb.min.z );
    v[1] = make_float3( aabb.min.x, aabb.min.y, aabb.max.z );
    v[2] = make_float3( aabb.min.x, aabb.max.y, aabb.min.z );
    v[3] = make_float3( aabb.min.x, aabb.max.y, aabb.max.z );
    v[4] = make_float3( aabb.max.x, aabb.min.y, aabb.min.z );
    v[5] = make_float3( aabb.max.x, aabb.min.y, aabb.max.z );
    v[6] = make_float3( aabb.max.x, aabb.max.y, aabb.min.z );
    v[7] = make_float3( aabb.max.x, aabb.max.y, aabb.max.z );
}

M_DEVICE_HOST __inline__ void adjust_difference_aabb( Aabb& difference_aabb, float t, float val, unsigned int dim, const Aabb& transf_aabb0, const Aabb& transf_aabb1 )
{
    float interp_min =
        *( &transf_aabb0.min.x + dim ) + ( *( &transf_aabb1.min.x + dim ) - *( &transf_aabb0.min.x + dim ) ) * t;
    float interp_max =
        *( &transf_aabb0.max.x + dim ) + ( *( &transf_aabb1.max.x + dim ) - *( &transf_aabb0.max.x + dim ) ) * t;
    float diff = val - interp_min;
    if( diff < *( &difference_aabb.min.x + dim ) )
        *( &difference_aabb.min.x + dim ) = diff;
    diff = val - interp_max;
    if( diff > *( &difference_aabb.max.x + dim ) )
        *( &difference_aabb.max.x + dim ) = diff;
}

// Enlarge aabbs by differences given in difference_aabbs.

M_DEVICE_HOST __inline__ void add_difference( Aabb& aabb0, Aabb& aabb1, const Aabb& difference_aabb )
{
    aabb0.min += difference_aabb.min;
    aabb0.max += difference_aabb.max;
    aabb1.min += difference_aabb.min;
    aabb1.max += difference_aabb.max;
}

// Given two aabbs at time 0 and time 1 and its linear interpolation for t between 0 and 1,
// apply the transformation given by the keys to it. (transformation on moving input)
// It's not sufficient to apply key0 on aabb0 and key1 on aabb1, since the per-vertex function is potentially quadratic (if aabb0 vs. aabb1 and key0 vs. key1 differ).
// We take an initial guess for a derivative for the linear output function outAabb0 to outAabb1 (initial guess is passed in, could be key0*aabb0 and key1*aabb1).
// Below, we compute the minimum/maximum of the quadratic function minus the initial guess.
// We do this by computing the derivative of the quadratic function minus the initial guess and compute the root of the this linear function.
// The difference at the root (time t) is used to enlarge the initial guess such that it will fit (min/max resp.) the values at the root.
// TODO: check if we really need to do this for all corner points or simply for aabb min/max.
static M_DEVICE_HOST __noinline__ Aabb transformComputeAdjustment( const Aabb& aabb0, const Aabb& aabb1, const float* key0, const float* key1, const Aabb& outAabb0, const Aabb& outAabb1 )
{
    // v0 vertices of aabb0, v1 vertices from aabb1
    float3 v0[8], v1[8];
    extractCornersFromAabb( v0, aabb0 );
    extractCornersFromAabb( v1, aabb1 );

    float3 d_v[8] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2], v1[3] - v0[3],
                     v1[4] - v0[4], v1[5] - v0[5], v1[6] - v0[6], v1[7] - v0[7]};

    float d_k[12] = {key1[0] - key0[0], key1[1] - key0[1], key1[2] - key0[2],   key1[3] - key0[3],
                     key1[4] - key0[4], key1[5] - key0[5], key1[6] - key0[6],   key1[7] - key0[7],
                     key1[8] - key0[8], key1[9] - key0[9], key1[10] - key0[10], key1[11] - key0[11]};

    // difference_aabb will contain the necessary enlargement of transf_aabb0 and transf_aabb1,
    // such that the linear interpolation between these two will contain all (non-linear) motion between 0 and 1
    Aabb difference_aabb ={}; // init to zero

    // For all 8 corners check whether its quadratic motion path exceeds the interpolation, separately for x, y, z.
    // This is done by computing the extrema of the difference of the motion path
    // and the interpolation of the end points of the motion path. The extrema is checked whether its inside
    // the interpolated transformed aabbs, otherwise the transformed aabbs at time 0 and 1 are extended
    // such that the interpolation contain the intermediate motion. This gives tighter aabbs than just using the union of
    // the input aabbs as replacement for the input aabbs at 0 and 1.

    // point i
    for( unsigned int i = 0; i < 8; ++i )
    {
        // dimension X, Y, or Z
        for( unsigned int dim = 0; dim < 3; ++dim )
        {
            const unsigned int off = dim * 4;

            float denom = d_v[i].x * d_k[off + 0] + d_v[i].y * d_k[off + 1] + d_v[i].z * d_k[off + 2];
            if( denom != 0.f )
            {
                float num = v0[i].x * d_k[off + 0] + d_v[i].x * key0[off + 0] +
                            v0[i].y * d_k[off + 1] + d_v[i].y * key0[off + 1] +
                            v0[i].z * d_k[off + 2] + d_v[i].z * key0[off + 2];

                float t = -0.5f * ( num - ( *( &outAabb1.min.x + dim ) - *( &outAabb0.min.x + dim ) ) ) / denom;
                if( t > 0.f && t < 1.f )
                {
                    float val = ( v0[i].x + d_v[i].x * t ) * ( key0[off + 0] + d_k[off + 0] * t ) +
                                ( v0[i].y + d_v[i].y * t ) * ( key0[off + 1] + d_k[off + 1] * t ) +
                                ( v0[i].z + d_v[i].z * t ) * ( key0[off + 2] + d_k[off + 2] * t ) +
                                key0[off + 3] + d_k[off + 3] * t;
                    adjust_difference_aabb( difference_aabb, t, val, dim, outAabb0, outAabb1 );
                }

                t = -0.5f * ( num - ( *( &outAabb1.max.x + dim ) - *( &outAabb0.max.x + dim ) ) ) / denom;
                if( t > 0.f && t < 1.f )
                {
                    float val = ( v0[i].x + d_v[i].x * t ) * ( key0[off + 0] + d_k[off + 0] * t ) +
                                ( v0[i].y + d_v[i].y * t ) * ( key0[off + 1] + d_k[off + 1] * t ) +
                                ( v0[i].z + d_v[i].z * t ) * ( key0[off + 2] + d_k[off + 2] * t ) +
                                key0[off + 3] + d_k[off + 3] * t;
                    adjust_difference_aabb( difference_aabb, t, val, dim, outAabb0, outAabb1 );
                }
            }
        }
    }

    return difference_aabb;
}

static M_DEVICE_HOST __inline__ void transform( const Aabb& aabb0, const Aabb& aabb1, const float* key0, const float* key1, Aabb& outAabb0, Aabb& outAabb1 )
{
    // transformation of the motion aabb at time 0 and time 1
    outAabb0 = transform( aabb0, key0 );
    outAabb1 = transform( aabb1, key1 );

    Aabb adjustment = transformComputeAdjustment( aabb0, aabb1, key0, key1, outAabb0, outAabb1 );

    add_difference( outAabb0, outAabb1, adjustment );
}

static M_DEVICE_HOST __inline__ void transformInOut0Out1( const Aabb& aabb0, const Aabb& aabb1, const float* key0, const float* key1, Aabb& inOutAabb0, Aabb& outAabb1 )
{
    // transformation of the motion aabb at time 0 and time 1
    // inOutAabb0 is assumed to include transform( aabb0, key0 )
    //inOutAabb0 = transform( aabb0, key0 );
    outAabb1 = transform( aabb1, key1 );

    Aabb adjustment = transformComputeAdjustment( aabb0, aabb1, key0, key1, inOutAabb0, outAabb1 );

    add_difference( inOutAabb0, outAabb1, adjustment );
}

static M_DEVICE_HOST __inline__ void transformInOut0InOut1( const Aabb& aabb0, const Aabb& aabb1, const float* key0, const float* key1, Aabb& inOutAabb0, Aabb& inOutAabb1 )
{
    // transformation of the motion aabb at time 0 and time 1
    // inOutAabb0 is assumed to include transform( aabb0, key0 )
    // inOutAabb1 is assumed to include transform( aabb1, key1 )
    //inOutAabb0 = transform( aabb0, key0 );
    //inOutAabb1 = transform( aabb1, key1 );

    Aabb adjustment = transformComputeAdjustment( aabb0, aabb1, key0, key1, inOutAabb0, inOutAabb1 );

    add_difference( inOutAabb0, inOutAabb1, adjustment );
}

// Matrix3x4 of the above
static M_DEVICE_HOST __inline__ void transform( const Aabb& aabb0, const Aabb& aabb1, const Matrix3x4& key0, const Matrix3x4& key1, Aabb& outAabb0, Aabb& outAabb1 )
{
    return transform( aabb0, aabb1, key0.m, key1.m, outAabb0, outAabb1 );
}
static M_DEVICE_HOST __inline__ void transformInOut0Out1( const Aabb& aabb0, const Aabb& aabb1, const Matrix3x4& key0, const Matrix3x4& key1, Aabb& inOutAabb0, Aabb& outAabb1 )
{
    transformInOut0Out1( aabb0, aabb1, key0.m, key1.m, inOutAabb0, outAabb1 );
}
static M_DEVICE_HOST __inline__ void transformInOut0InOut1( const Aabb& aabb0, const Aabb& aabb1, const Matrix3x4& key0, const Matrix3x4& key1, Aabb& inOutAabb0, Aabb& inOutAabb1 )
{
    transformInOut0InOut1( aabb0, aabb1, key0.m, key1.m, inOutAabb0, inOutAabb1 );
}

}  // namespace motion
}  // namespace optix_exp
