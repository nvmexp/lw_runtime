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

#include "motionTypes.h"
#include <vector_functions.h>
#include <cfloat>
#ifndef __LWDACC__
#include <algorithm>
#endif

namespace optix_exp {
namespace motion {

#ifndef __LWDACC__
using std::min;
using std::max;
#endif


#ifndef __LWDACC__
inline float3 fminf( const float3& a, const float3& b )
{
    return make_float3( min( a.x, b.x ), min( a.y, b.y ), min( a.z, b.z ) );
}
inline float3 fmaxf( const float3& a, const float3& b )
{
    return make_float3( max( a.x, b.x ), max( a.y, b.y ), max( a.z, b.z ) );
}
#endif
M_DEVICE_HOST __inline__ bool operator==( const float3& a, const float3& b )
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
M_DEVICE_HOST __inline__ float3 fminf( const float3& a, float b )
{
    return make_float3( min( a.x, b ), min( a.y, b ), min( a.z, b ) );
}
M_DEVICE_HOST __inline__ float3 fminf( float b, const float3& a )
{
    return make_float3( min( a.x, b ), min( a.y, b ), min( a.z, b ) );
}
M_DEVICE_HOST __inline__ float3 fmaxf( const float3& a, float b )
{
    return make_float3( max( a.x, b ), max( a.y, b ), max( a.z, b ) );
}
M_DEVICE_HOST __inline__ float3 fmaxf( float b, const float3& a )
{
    return make_float3( max( a.x, b ), max( a.y, b ), max( a.z, b ) );
}
M_DEVICE_HOST __inline__ float3 operator-( const float3& a, const float3& b )
{
    return make_float3( a.x - b.x, a.y - b.y, a.z - b.z );
}
M_DEVICE_HOST __inline__ float3 operator+( const float3& a, const float3& b )
{
    return make_float3( a.x + b.x, a.y + b.y, a.z + b.z );
}
M_DEVICE_HOST __inline__ void operator+=( float3& a, const float3& b )
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
M_DEVICE_HOST __inline__ float3 operator*( const float3& a, const float3& b )
{
    return make_float3( a.x * b.x, a.y * b.y, a.z * b.z );
}
M_DEVICE_HOST __inline__ float3 operator/( const float3& a, const float3& b )
{
    return make_float3( a.x / b.x, a.y / b.y, a.z / b.z );
}
M_DEVICE_HOST __inline__ float3 operator/( const float3& a, float b )
{
    return make_float3( a.x / b, a.y / b, a.z / b );
}
M_DEVICE_HOST __inline__ float4 operator+( const float4& a, const float4& b )
{
    return make_float4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}
M_DEVICE_HOST __inline__ float4 operator-( const float4& a, const float4& b )
{
    return make_float4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

M_DEVICE_HOST __inline__ float3 operator*( const float3& a, const float s )
{
    return make_float3( a.x * s, a.y * s, a.z * s );
}
M_DEVICE_HOST __inline__ float3 operator*( const float s, const float3& a )
{
    return make_float3( a.x * s, a.y * s, a.z * s );
}

M_DEVICE_HOST __inline__ float4 operator*( const float4& a, const float s )
{
    return make_float4( a.x * s, a.y * s, a.z * s, a.w * s );
}
M_DEVICE_HOST __inline__ float4 operator*( const float s, const float4& a )
{
    return make_float4( a.x * s, a.y * s, a.z * s, a.w * s );
}

M_DEVICE_HOST __inline__ float fract( float a )
{
    return a - truncf( a ); // or use modff?
}

M_DEVICE_HOST __inline__ float lerp( float a, float b, float t )
{
    return a + t * ( b - a );
}
M_DEVICE_HOST __inline__ float3 lerp( const float3& a, const float3& b, float t )
{
    return a + t * ( b - a );
}
M_DEVICE_HOST __inline__ float4 lerp( const float4& a, const float4& b, float t )
{
    return a + t * ( b - a );
}
M_DEVICE_HOST __inline__ Aabb lerp( const Aabb& a, const Aabb& b, float t )
{
    return { lerp( a.min, b.min, t ), lerp( a.max, b.max, t ) };
}

M_DEVICE_HOST __inline__ void include( Aabb& a, const Aabb& b )
{
    a.min.x = min( a.min.x, b.min.x );
    a.min.y = min( a.min.y, b.min.y );
    a.min.z = min( a.min.z, b.min.z );
    a.max.x = max( a.max.x, b.max.x );
    a.max.y = max( a.max.y, b.max.y );
    a.max.z = max( a.max.z, b.max.z );
}

M_DEVICE_HOST __inline__ void include( Aabb& a, const float3& b )
{
    a.min.x = min( a.min.x, b.x );
    a.min.y = min( a.min.y, b.y );
    a.min.z = min( a.min.z, b.z );
    a.max.x = max( a.max.x, b.x );
    a.max.y = max( a.max.y, b.y );
    a.max.z = max( a.max.z, b.z );
}

M_DEVICE_HOST __inline__ float3 llf( const Aabb& a )
{
    return a.min;
}
M_DEVICE_HOST __inline__ float3 urb( const Aabb& a )
{
    return a.max;
}
M_DEVICE_HOST __inline__ float3 center( const Aabb& a )
{
    return ( a.min + a.max ) * 0.5f;
}

M_DEVICE_HOST __inline__ void ilwalidate( Aabb& aabb )
{
    aabb.min = make_float3( FLT_MAX, FLT_MAX, FLT_MAX );
    aabb.max = make_float3( -FLT_MAX, -FLT_MAX, -FLT_MAX );
}

M_DEVICE_HOST __inline__  bool valid( const Aabb& aabb )
{
    return aabb.min.x <= aabb.max.x && aabb.min.y <= aabb.max.y && aabb.min.z <= aabb.max.z;
}

M_DEVICE_HOST __inline__ bool contains( const Aabb& a, const Aabb& b )
{
    return a.min.x <= b.min.x && a.min.y <= b.min.y && a.min.z <= b.min.z && a.max.x >= b.max.x && a.max.y >= b.max.y && a.max.z >= b.max.z;
}


M_DEVICE_HOST __inline__ void resampleSimpleUniform( const Aabb& inAabb, float localt, Aabb& inoutAabb0, Aabb& inoutAabb1 )
{
    // Very simple, but robust resampling scheme, do NOT change the derivative, just bump aabb sizes
    // This scheme works for any aabb transformation function where inAabb is the extrema of the function within the time range from inoutAabb0 to inoutAabb1
    // Calling this function for multiple extrema also works and yields a correct (conservative) end result.
    // We could also favor pumping inoutAabb0 over inoutAabb1 in case localt < 0.5 and the other way around if localt > 0.5, however, this would not have the aforementioned guarantee.
#define MOTION_TRANSFORM_RESAMPLE_MINMAX(min) \
        { \
            float3 adjustment = f##min##f( 0.0f, inAabb.min - lerp( inoutAabb0.min, inoutAabb1.min, localt ) ); \
            inoutAabb0.min += adjustment; \
            inoutAabb1.min += adjustment; \
        }
    MOTION_TRANSFORM_RESAMPLE_MINMAX( min );
    MOTION_TRANSFORM_RESAMPLE_MINMAX( max );
#undef MOTION_TRANSFORM_RESAMPLE_MINMAX
};

M_DEVICE_HOST __inline__ void resampleLocalOptima( const Aabb& inAabb, float localt, Aabb& inoutAabb0, Aabb& inoutAabb1 )
{
    // simple resampling scheme, change the derivative based on the pivot (localt)

    if( localt < 0.5f )
    {
        inoutAabb0.min += fminf( 0.0f, ( inAabb.min - lerp( inoutAabb0.min, inoutAabb1.min, localt ) ) / ( 1.0f - localt ) );
        inoutAabb0.max += fmaxf( 0.0f, ( inAabb.max - lerp( inoutAabb0.max, inoutAabb1.max, localt ) ) / ( 1.0f - localt ) );
    }
    else if( localt == 0.5f )
    {
        return resampleSimpleUniform( inAabb, 0.5f, inoutAabb0, inoutAabb1 );
    }
    else
    {
        inoutAabb1.min += fminf( 0.0f, ( inAabb.min - lerp( inoutAabb0.min, inoutAabb1.min, localt ) ) / ( localt ) );
        inoutAabb1.max += fmaxf( 0.0f, ( inAabb.max - lerp( inoutAabb0.max, inoutAabb1.max, localt ) ) / ( localt ) );
    }
};

}
}
