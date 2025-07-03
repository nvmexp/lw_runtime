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

#define M_DEVICE_HOST __device__

#include <vector_types.h>
#include <cmath>

namespace optix_exp {
namespace motion {

union Aabb {
    struct {
        float3 min;
        float3 max;
    };
    float minMax[6];
};

//////////////////////////////////////////////////////////////////////////
struct AABB {
    float3 lo;  // Min corner.
    float3 hi;  // Max corner.
};

struct PrimitiveAABB {
    union {
        struct {
            float   lox;            // Min corner of the AABB.
            float   loy;
            float   loz;
            float   hix;            // Max corner of the AABB.

            float   hiy;
            float   hiz;
            int     primitiveIdx;   // Index of the primitive that this AABB corresponds to.
            int     pad;
        };
        struct {
            float4 f4[2];
        };
    };
};
//////////////////////////////////////////////////////////////////////////

struct Matrix3x4
{
    float m[12];
};

struct SRTData {
    float sx, a, b, pvx, sy, c, pvy, sz, pvz, qx, qy, qz, qw, tx, ty, tz;
};

M_DEVICE_HOST __inline__ Aabb lerp( const Aabb& a, const Aabb& b, float t );
M_DEVICE_HOST __inline__ float fract( float a );

struct AabbArray {
    enum {
        capacity_ = 5 // potential values: 2,3,5,9,17, ... such that we double the number of intervals: x_{i+1} = 2 x_i - 1
    };
    unsigned int size; // potential values: 2,3,5,9,17, ... max capacity_
    Aabb aabbs[capacity_];

    M_DEVICE_HOST __inline__ Aabb interpolate( unsigned int first, float localt ) const
    {
        const float interval    = 1.f / size;
        const float interpolant = ( localt - ( first * interval ) ) / interval;
        return lerp( aabbs[first], aabbs[first + 1], interpolant );
    }
};

struct MotionOptions {
    /// If numKeys > 1, motion is enabled. timeBegin,
    /// timeEnd and flags are all ignored when motion is disabled.
    unsigned short numKeys;

    /// Combinations of #OptixMotionFlags
    unsigned short flags;

    /// Point in time where motion starts.
    float timeBegin;

    /// Point in time where motion ends.
    float timeEnd;

    M_DEVICE_HOST __inline__ float intervalSize() const
    {
        return ( timeEnd - timeBegin ) / ( numKeys - 1 );
    }

    M_DEVICE_HOST __inline__ bool timesAlign( const MotionOptions& other ) const
    {
        return timeBegin == other.timeBegin && timeEnd == other.timeEnd;
    }
    M_DEVICE_HOST __inline__ bool triviallyAligns( const MotionOptions& other ) const
    {
        return numKeys == other.numKeys && timesAlign( other );
    }

    M_DEVICE_HOST __inline__ float keyAtTNonclamped( float t ) const
    {
        return ( t - timeBegin ) / intervalSize();
    }
    M_DEVICE_HOST __inline__ float keyAtTNonclamped( float t, float interval ) const
    {
        return ( t - timeBegin ) / interval;
    }

    M_DEVICE_HOST __inline__ float keyAtT( float t ) const
    {
        return fmaxf( 0.0f, fminf( (float)(numKeys - 1), keyAtTNonclamped( t ) ) );
    }
    M_DEVICE_HOST __inline__ float keyAtT( float t, float interval ) const
    {
        return fmaxf( 0.0f, fminf( (float)(numKeys - 1), keyAtTNonclamped( t, interval ) ) );
    }

    M_DEVICE_HOST __inline__ void keyAtT( float t, unsigned int& key, float& localt ) const
    {
        float fkey = fmaxf( 0.0f, fminf( (float)(numKeys - 1), keyAtTNonclamped( t ) ) );
        localt = fract( fkey );
        key = (unsigned int)fkey;
    }
    M_DEVICE_HOST __inline__ void keyAtT( float t, unsigned int& key, float& localt, float interval ) const
    {
        float fkey = fmaxf( 0.0f, fminf( (float)(numKeys - 1), keyAtTNonclamped( t, interval ) ) );
        localt = fract( fkey );
        key = (unsigned int)fkey;
    }

    M_DEVICE_HOST __inline__ unsigned int keyAtOrAfterT( float t ) const
    {
        return (unsigned int)ceilf( keyAtT( t ) );
    }
    M_DEVICE_HOST __inline__ unsigned int keyAtOrAfterT( float t, float interval ) const
    {
        return (unsigned int)ceilf( keyAtT( t, interval ) );
    }

    M_DEVICE_HOST __inline__ unsigned int keyAtOrBeforeT( float t ) const
    {
        return (unsigned int)floorf( keyAtT( t ) );
    }
    M_DEVICE_HOST __inline__ unsigned int keyAtOrBeforeT( float t, float interval ) const
    {
        return (unsigned int)floorf( keyAtT( t, interval ) );
    }

    M_DEVICE_HOST __inline__ float timeAtKey( unsigned int key ) const
    {
        return timeBegin + key * intervalSize();
    }
    M_DEVICE_HOST __inline__ float timeAtKey( unsigned int key, float interval ) const
    {
        return timeBegin + key * interval;
    }
};


}
}
