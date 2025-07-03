/*
 * Copyright (c) 2016, LWPU CORPORATION. All rights reserved.
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

#include <stdint.h>

#include <lwda_runtime.h>
#include <vector_functions.h>
#include <vector_types.h>

namespace optix_exp {

// float3 operators
__forceinline__ __host__ __device__ float3 operator+( const float3& a, const float3& b )
{
    return ::make_float3( a.x + b.x, a.y + b.y, a.z + b.z );
}
__forceinline__ __host__ __device__ float3 operator-( const float3& a, const float3& b )
{
    return ::make_float3( a.x - b.x, a.y - b.y, a.z - b.z );
}
__forceinline__ __host__ __device__ float3 operator-( const float3& a, const float b )
{
    return ::make_float3( a.x - b, a.y - b, a.z - b );
}
__forceinline__ __host__ __device__ float3 operator-( const float a, const float3& b )
{
    return ::make_float3( a - b.x, a - b.y, a - b.z );
}
__forceinline__ __host__ __device__ float3 operator-( const float3& a )
{
    return ::make_float3( -a.x, -a.y, -a.z );
}
__forceinline__ __host__ __device__ float3 operator*( const float s, const float3& a )
{
    return ::make_float3( a.x * s, a.y * s, a.z * s );
}

__forceinline__ __host__ __device__ float3 operator*( const float3& a, const float s )
{
    return ::make_float3( a.x * s, a.y * s, a.z * s );
}

// float4 operators
__forceinline__ __host__ __device__ float4 operator+( const float4& a, const float4& b )
{
    return ::make_float4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}
__forceinline__ __host__ __device__ void operator+=( float4& a, const float4& b )
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
__forceinline__ __host__ __device__ float4 operator-( const float4& a )
{
    return ::make_float4( -a.x, -a.y, -a.z, -a.w );
}
__forceinline__ __host__ __device__ float4 operator-( const float4& a, const float4& b )
{
    return ::make_float4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}
__forceinline__ __host__ __device__ void operator-=( float4& a, const float4& b )
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
__forceinline__ __host__ __device__ float4 operator*( const float4& a, const float4& s )
{
    return ::make_float4( a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w );
}
__forceinline__ __host__ __device__ float4 operator*( const float4& a, const float s )
{
    return ::make_float4( a.x * s, a.y * s, a.z * s, a.w * s );
}
__forceinline__ __host__ __device__ float4 operator*( const float s, const float4& a )
{
    return ::make_float4( a.x * s, a.y * s, a.z * s, a.w * s );
}
__forceinline__ __host__ __device__ void operator*=( float4& a, const float4& s )
{
    a.x *= s.x;
    a.y *= s.y;
    a.z *= s.z;
    a.w *= s.w;
}
__forceinline__ __host__ __device__ void operator*=( float4& a, const float s )
{
    a.x *= s;
    a.y *= s;
    a.z *= s;
    a.w *= s;
}

static __forceinline__ __device__ float3 make_float3( const float4& v )
{
    return ::make_float3( v.x, v.y, v.z );
}
static __forceinline__ __device__ float3 make_float3( const float x, const float y, const float z )
{
    return ::make_float3( x, y, z );
}

// colwert Bspline segment to Bezier

// Colwert quadratic B-spline control points for this segment into cubic Bezier control points.
static __device__ __inline__ void colwSegBsplineBez2( float4* p, const float4* q )
{
    // B-spline to bezier colwersion. NB 4-channel lerp, position+radius
    p[0] = ( q[0] + q[1] ) * 0.5f;
    p[1] = q[1];
    p[2] = ( q[1] + q[2] ) * 0.5f;
}

// colwert Bspline segment to Bezier
// Colwert cubic B-spline control points for this segment into cubic Bezier control points.
__forceinline__ __host__ __device__ void colwSegBsplineBez3( float4* p, const float4* q )
{
    // bez = Matrix([[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]])
    // bsp = Matrix([[1, 4, 1, 0], [-3, 0, 3, 0], [3, -6, 3, 0], [-1, 3, -3, 1]])/6
    // bsp2bez = bez.ilw("LU") * bsp = Matrix([[1/6, 2/3, 1/6, 0], [0, 2/3, 1/3, 0], [0, 1/3, 2/3, 0], [0, 1/6, 2/3, 1/6]])

    // (For not losing precision, please don't rearrange computation.)
    const float rcp6 = 1.f / 6.f;
    p[0]             = q[0] * rcp6 + q[1] * ( 4.f * rcp6 ) + q[2] * rcp6;
    p[1]             = q[1] * ( 4.f * rcp6 ) + q[2] * ( 2.f * rcp6 );
    p[2]             = q[1] * ( 2.f * rcp6 ) + q[2] * ( 4.f * rcp6 );
    p[3]             = q[1] * rcp6 + q[2] * ( 4.f * rcp6 ) + q[3] * rcp6;
}

// colwert CatmullRom segment to cubic Bezier
// Catrom-to-Bezier = Matrix([[0, 1, 0, 0], [-1/6, 1, 1/6, 0], [0, 1/6, 1, -1/6], [0, 0, 1, 0]])
__forceinline__ __host__ __device__ void colwSegCatromBez3( float4* p, const float4* q )
{
    // s=0.5
    // Catrom = Matrix([[0, 1, 0, 0], [-s, 0, s, 0], [2*s, s-3, 3-2*s, -s], [-s, 2-s, s-2, s]])
    // Bezier     = Matrix([[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]])
    // Catrom2Bezier = Bezier.ilw("LU") * Catrom = Matrix([[0, 1, 0, 0], [-1/6, 1, 1/6, 0], [0, 1/6, 1, -1/6], [0, 0, 1, 0]])
    p[0] = q[1];
    p[1] = q[0] * ( -1.f / 6.f ) + q[1] + q[2] * ( 1.f / 6.f );
    p[2] = q[1] * ( 1.f / 6.f ) + q[2] + q[3] * ( -1.f / 6.f );
    p[3] = q[2];
}

__forceinline__ __host__ __device__ float dot3( const float3& aValue0, const float3& aValue1 )
{
    return aValue0.x * aValue1.x + aValue0.y * aValue1.y + aValue0.z * aValue1.z;
}

__forceinline__ __host__ __device__ float3 cross3( const float3& a, const float3& b )
{
    return ::make_float3( a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x );
}

__forceinline__ __host__ __device__ float length3( const float3& v )
{
    return sqrtf( dot3( v, v ) );
}

static __forceinline__ __host__ __device__ float dot4( const float4& aValue0, const float4& aValue1 )
{
    return aValue0.x * aValue1.x + aValue0.y * aValue1.y + aValue0.z * aValue1.z + aValue0.w * aValue1.w;
}

__forceinline__ __host__ __device__ float length4( const float4& r )
{
    return sqrtf( dot4( r, r ) );
}

static __device__ __inline__ float3 normalize3( const float3& n )
{
    return n * ( 1.f / sqrtf( dot3( n, n ) ) );
}

__forceinline__ __host__ __device__ float3 shortestVectorToLine( const float3& point, const float3& pointOnLine, const float3& direction )
{
    float3 dp = point - pointOnLine;
    return dot3( direction, dp ) / dot3( direction, direction ) * direction - dp;
}

__forceinline__ __host__ __device__ float clamp( const float f, const float a, const float b )
{
    return fmaxf( a, fminf( f, b ) );
}


// cubic polynom solver
// (used by computation of approximate inflection points of cubic bsplines)

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

static __device__ __forceinline__ float lwbicroot( float x )
{
    return copysignf( exp2f( log2f( fabsf( x ) ) * ( 1.f / 3.f ) ), x );
}

static __device__ __forceinline__ float Newton_lwbic_step( float& root, float d, float c, float b, float a )
{
    float f0 = d + root * ( c + root * ( b + a * root ) );
    //for (int it = 0; it < 3; it++) {
    // if (fabsf(f0) < 1e-9f) return f0;
    float f1    = c + root * ( 2 * b + 3 * a * root );
    float rootn = root - f0 / f1;
    float fn    = d + rootn * ( c + rootn * ( b + a * rootn ) );
    if( fabsf( fn ) < fabsf( f0 ) )
    {
        root = rootn;
    }
    f0 = fn;
    //}
    return f0;
}

// returns a single "most distinct" root
static __device__ __forceinline__ float solvelwbic1( float d, float c, float b, bool ilwerse )
{
    // equation is d + c*v + b*v*v + v*v*v
    float p, p3, q, det, v, m, n;
    float bc = b * c;
    float b2 = b * b;
    float b3 = b * b2;
    p        = c - b2 / 3.;
    p3       = p * p * p;
    q        = d + ( float( 2. / 27.0 ) ) * b3 - bc / 3.;
    det      = q * q + p3 * float( 4. / 27. );

    if( det >= 0 )
    {

        float z = sqrtf( det );

        {
            float c2 = c * c;
            float c3 = c * c2;
            float at, ct, dt;  // values with tildes
            bool  direct = b3 * d >= c3;
            if( direct )
            {
                float sig1 = p / 3.;
                float sig2 = d - bc / 9.;
                at         = 1.f;
                ct         = sig1;
                dt         = float( 2. / 3.0 ) * b * sig1 - sig2;
            }
            else
            {
                float sig2 = d - bc / 9;
                float sig3 = b * d / 3 - c2 / 9;
                at         = d;
                ct         = sig3;
                dt         = d * sig2 - float( 2 / 3.0 ) * c * sig3;
            }
            float t0 = copysignf( at, dt ) * z;
            float t1 = t0 + dt;
            float p1 = lwbicroot( t1 / 2 );
            float q1 = ( t0 == t1 ) ? p1 : ct / p1;
            float xt = ( ct <= 0 ) ? p1 - q1 : dt / ( p1 * p1 + q1 * q1 + ct );
            v        = direct ? xt - b / 3 : -d / ( xt + c / 3 );
        }

        return v;
    }

    v = -sqrtf( -float( 27.0 / 4 ) / p3 );
    v *= q;
    v = v <= -1 ? M_PIf / 3 : v >= 1 ? 0 : acos( v ) / 3;

    m = cos( v );
    n = sin( v ) * sqrtf( float( 3.0 ) );
    p = sqrtf( -p / 3 );

    b /= 3;
    float v1 = ( +m + m ) * p - b;
    float v2 = ( -n - m ) * p - b;
    float v3 = ( +n - m ) * p - b;

    //   i  d
    // < 2  1
    // > 1  2
    const float close = float( 0.01 );
    if( ( fabsf( v1 - v3 ) < ( ilwerse ? close * v1 * v3 : close ) ) || ( fabsf( v2 - v3 ) < ( ilwerse ? close * v2 * v3 : close ) ) )
        v = v3;
    else
        v = ilwerse ^ ( fabsf( v1 ) < fabsf( v2 ) ) ? v1 : v2;

    return v;
}

static __device__ __forceinline__ int solvelwbic_i( const float d0,
                                                    const float c0,
                                                    const float b0,
                                                    const float a0,
                                                    float       roots[3],
                                                    float       t0  = 0,
                                                    float       t1  = 1,
                                                    float       eps = 1e-7f )
{

    // quadratic polynom for "other" roots in (u - u0) (c2 + b2 u + a2 u^2)
    float c2, b2, a2;  // c2 + b2 u + a2 u^2
    float u0;          // a distinct root
    int   nroots = 0;  // yet

    if( fabsf( a0 ) < eps )
    {
        if( fabsf( b0 ) < eps )
        {
            // ~linear equation
            u0 = -d0 / c0;
        }
        else
        {
            c2 = d0;
            b2 = c0;
            a2 = b0;
            goto quadratic;
        }
    }
    else if( fabsf( d0 ) < eps )
    {
        u0 = 0;
    }
    else
    {
        // We do not want to shift if dnew will be smaller since it is dangerous for (u^2 - 1) (u - other_root)
        float h    = 0.5;
        float dnew = d0 + ( ( h / 2 ) * b0 - ( h / 4 ) * a0 - h * c0 );
        h          = fabsf( dnew ) > fabsf( d0 ) ? h : 0;
        float d1 = d0, c1 = c0, b1 = b0, a1 = a0;
        if( h )
        {
            d1 = dnew;
            c1 += ( h + h / 2 ) * a0 - b0;
            b1 -= ( 3 * h ) * a0;
        }
        bool ilwerse;

        // We want a  smaller (conservative) interval that contains all roots.
        // (Could also go with L2 metric.)
        float bc             = fmaxf( fabsf( c1 ), fabsf( b1 ) );
        float direct_Cauchy  = fmaxf( fabsf( d1 ), bc );  // /a1
        float ilwerse_Cauchy = fmaxf( fabsf( a1 ), bc );  // /d1
        ilwerse              = direct_Cauchy * fabsf( d1 ) > ilwerse_Cauchy * fabsf( a1 );

        // Colwert to c2 + b2 u + a2 u^2 + u^3
        if( ilwerse )
        {
            c2 = a1 / d1;
            b2 = b1 / d1;
            a2 = c1 / d1;  // u := 1/(u - h)
        }
        else
        {
            c2 = d1 / a1;
            b2 = c1 / a1;
            a2 = b1 / a1;  // u := u - h
        }

        u0 = solvelwbic1( c2, b2, a2, ilwerse );
        if( ilwerse )
            u0 = 1 / u0;
        u0 -= h;
    }

    // using the original polynom from now on
    // and split it into (u - u0) (c2 + b2 u + a2 u^2)

    Newton_lwbic_step( u0, d0, c0, b0, a0 );

    if( !clamp || ( t0 <= u0 ) == ( u0 <= t1 ) )
        roots[nroots++] = u0;
    /*
      {d0, c0, b0, a0} = CoefficientList[scale (u - u0) (u - u1) (u - u2) , u];
      a2 = a0; b2 = b0 + a2*u0; c2 = c0 + b2*u0;
      Solve[c2 + b2 u + a2 u^2 == 0, u]
    */
    a2 = a0;
    b2 = b0 + a2 * u0;
    c2 = c0 + b2 * u0;

quadratic:

    // c2 + b2 u + a2 u^2
    float det = b2 * b2 - 4 * a2 * c2;
    if( det < 0 )
        return nroots;
    det = sqrtf( det );
    float u1, u2;
    u1 = ( -b2 - ( b2 < 0 ? -det : det ) ) / 2;  // numerically "stable" root
    u2 = c2 / u1;                                // Vieta's formula for u1*u2
    u1 /= a2;

    Newton_lwbic_step( u1, d0, c0, b0, a0 );
    Newton_lwbic_step( u2, d0, c0, b0, a0 );

    if( !clamp || ( t0 <= u1 ) == ( u1 <= t1 ) )
        roots[nroots++] = u1;
    if( !clamp || ( t0 <= u2 ) == ( u2 <= t1 ) )
        roots[nroots++] = u2;

    return nroots;
}

struct TightBounds
{
    static __device__ __forceinline__ float3 unit_3D_circle_extent( const float3& n )
    {
        float l2 = dot3( n, n );
        if( l2 == 0.f )
            return make_float3( 1.f, 1.f, 1.f );  // 0 at double or triple points
        return make_float3( sqrtf( fmaxf( 0.f, 1.f - n.x * n.x / l2 ) ), sqrtf( fmaxf( 0.f, 1.f - n.y * n.y / l2 ) ),
                            sqrtf( fmaxf( 0.f, 1.f - n.z * n.z / l2 ) ) );
    }

    static __device__ __forceinline__ OptixAabb circleBound( const float3& p, const float3& v, float r )
    {
        // use oriented circle(cap)
        float3 cr = r * unit_3D_circle_extent( v );

        float minx = __fadd_rd( p.x, -cr.x );
        float miny = __fadd_rd( p.y, -cr.y );
        float minz = __fadd_rd( p.z, -cr.z );
        float maxx = __fadd_ru( p.x, cr.x );
        float maxy = __fadd_ru( p.y, cr.y );
        float maxz = __fadd_ru( p.z, cr.z );

        OptixAabb aabb = { minx, miny, minz, maxx, maxy, maxz };
        return aabb;
    }
};


struct LwrvePolynom3 : public TightBounds
{
    __device__ __forceinline__ LwrvePolynom3() {}

    __device__ __forceinline__ LwrvePolynom3( const float4* q, const bool isCatmullRom )
    {
        if( isCatmullRom )
            initializeFromCatmullRom( q );
        else
            initializeFromBSpline( q );
    }

    __device__ __forceinline__ void initializeFromBSpline( const float4* q )
    {
        p[0] = 1.f / 6.f * q[0] + ( 4.f / 6.0f ) * q[1] + 1.f / 6.f * q[2];
        p[1] = 1.f / 2.f * q[2] - 1.f / 2.f * q[0];
        p[2] = 1.f / 2.f * q[0] - q[1] + 1.f / 2.f * q[2];
        p[3] = 1.f / 2.f * q[1] - 1.f / 2.f * q[2] + 1.f / 6.f * q[3] - 1.f / 6.f * q[0];
    }

    __device__ __forceinline__ void initializeFromCatmullRom( const float4* q )
    {
        // Matrix to evaluate a Catmull-Rom lwrve with tension = 0.5
        // Matrix([[0.,1.,0.,0.],[-s, 0., s, 0.], [2.*s, s-3., 3.-2.*s, -s], [-s, 2.-s, s-2., s]])
        // = Matrix([[0, 1, 0, 0], [-1/2, 0, 1/2, 0], [1, -5/2, 2, -1/2], [-1/2, 3/2, -3/2, 1/2]])
        p[0] = q[1];
        p[1] = q[0] * -0.5f + q[2] * 0.5f;
        p[2] = q[0] + q[1] * -2.5f + q[2] * 2.f + q[3] * -0.5;
        p[3] = q[0] * -0.5f + q[1] * 1.5f + q[2] * -1.5f + q[3] * 0.5f;
    }

    __device__ __forceinline__ float3 position3( float t ) const
    {
        return make_float3( p[0].x + t * p[1].x + t * t * p[2].x + t * t * t * p[3].x,
                            p[0].y + t * p[1].y + t * t * p[2].y + t * t * t * p[3].y,
                            p[0].z + t * p[1].z + t * t * p[2].z + t * t * t * p[3].z );
    }

    __device__ __forceinline__ float3 tangent3( float t ) const
    {
        return make_float3( p[1].x + 2 * t * p[2].x + 3 * t * t * p[3].x, p[1].y + 2 * t * p[2].y + 3 * t * t * p[3].y,
                            p[1].z + 2 * t * p[2].z + 3 * t * t * p[3].z );
    }

    __device__ __forceinline__ float radius( float t ) const
    {
        return p[0].w + t * p[1].w + t * t * p[2].w + t * t * t * p[3].w;
    }

    __device__ __forceinline__ OptixAabb circleBound( float t, float r ) const
    {
        float3 p = position3( t );
        float3 v = tangent3( t );
        return TightBounds::circleBound( p, v, r );
    }

    static __device__ __forceinline__ void findExtremums( float q1, float q2, float q3, float& tx1, float& tx2 )
    {
        // We solve D[q0 + q1 t + q2 t^2 + q3 t^3, t] == 0
        tx1 = 0;
        tx2 = 1;  // default values that do not harm us in a degenerate case
        if( fabsf( q3 ) < 1.e-6f )
        {
            // If it is a quadratic lwrve in this dimension, we'll need only one t to check.
            if( q1 == 0.f || q2 == 0.f )
                return;
            tx1 = -0.5f * q1 / q2;
            tx1 -= 1.5f * q3 * tx1 * tx1 / ( 3.f * q3 * tx1 + q2 );  // Newton for q3 != 0
        }
        else
        {
            // check 2 extrema of a cubic
            float den = -3.f * q3;
            float det = q2 * q2 + den * q1;
            if( det < 0.f )
                return;  //  no extrema
            det = sqrtf( det );
            tx1 = ( q2 - det ) / den;
            tx2 = ( q2 + det ) / den;
        }
    }

    __device__ __forceinline__ void findExtremums( float4& tx1, float4& tx2 ) const
    {
        findExtremums( p[1].x, p[2].x, p[3].x, tx1.x, tx2.x );
        findExtremums( p[1].y, p[2].y, p[3].y, tx1.y, tx2.y );
        findExtremums( p[1].z, p[2].z, p[3].z, tx1.z, tx2.z );
        findExtremums( p[1].w, p[2].w, p[3].w, tx1.w, tx2.w );
    }

    __device__ __forceinline__ void includeExtremums( OptixAabb& bb, float t0, float t1, float tx1, float tx2, float max_r ) const
    {
        if( ( t0 < tx1 ) == ( tx1 < t1 ) )
        {
            OptixAabb cb = circleBound( tx1, max_r );
            bb.minX      = fminf( bb.minX, cb.minX );
            bb.minY      = fminf( bb.minY, cb.minY );
            bb.minZ      = fminf( bb.minZ, cb.minZ );
            bb.maxX      = fmaxf( bb.maxX, cb.maxX );
            bb.maxY      = fmaxf( bb.maxY, cb.maxY );
            bb.maxZ      = fmaxf( bb.maxZ, cb.maxZ );
        }
        if( ( t0 < tx2 ) == ( tx2 < t1 ) )
        {
            OptixAabb cb = circleBound( tx2, max_r );
            bb.minX      = fminf( bb.minX, cb.minX );
            bb.minY      = fminf( bb.minY, cb.minY );
            bb.minZ      = fminf( bb.minZ, cb.minZ );
            bb.maxX      = fmaxf( bb.maxX, cb.maxX );
            bb.maxY      = fmaxf( bb.maxY, cb.maxY );
            bb.maxZ      = fmaxf( bb.maxZ, cb.maxZ );
        }
    }

    __device__ __forceinline__ OptixAabb paddedBounds( float t0, float t1, const float4& tx1, const float4& tx2 ) const
    {
        float max_r = fmaxf( radius( t0 ), radius( t1 ) );
        if( ( t0 < tx1.w ) == ( tx1.w < t1 ) )
            max_r = fmaxf( radius( tx1.w ), max_r );
        if( ( t0 < tx2.w ) == ( tx2.w < t1 ) )
            max_r = fmaxf( radius( tx2.w ), max_r );
        OptixAabb bb_t0 = circleBound( t0, max_r );
        OptixAabb bb_t1 = circleBound( t1, max_r );
        OptixAabb bb;
        bb.minX = fminf( bb_t0.minX, bb_t1.minX );
        bb.minY = fminf( bb_t0.minY, bb_t1.minY );
        bb.minZ = fminf( bb_t0.minZ, bb_t1.minZ );
        bb.maxX = fmaxf( bb_t0.maxX, bb_t1.maxX );
        bb.maxY = fmaxf( bb_t0.maxY, bb_t1.maxY );
        bb.maxZ = fmaxf( bb_t0.maxZ, bb_t1.maxZ );
        includeExtremums( bb, t0, t1, tx1.x, tx2.x, max_r );
        includeExtremums( bb, t0, t1, tx1.y, tx2.y, max_r );
        includeExtremums( bb, t0, t1, tx1.z, tx2.z, max_r );
        return bb;
    }

    float4 p[4];
};

struct LwrvePolynom2 : public TightBounds
{
    __device__ __forceinline__ LwrvePolynom2( const float4* q ) { initializeFromBSpline( q ); }

    __device__ __forceinline__ void initializeFromBSpline( const float4* q )
    {
        p[0] = 0.5f * q[0] + 0.5f * q[1];
        p[1] = q[1] - q[0];
        p[2] = 0.5f * q[0] - q[1] + 0.5f * q[2];
    }

    __device__ __forceinline__ float3 position3( float t ) const
    {
        return make_float3( p[0].x + t * p[1].x + t * t * p[2].x, p[0].y + t * p[1].y + t * t * p[2].y,
                            p[0].z + t * p[1].z + t * t * p[2].z );
    }

    __device__ __forceinline__ float3 tangent3( float t ) const
    {
        return make_float3( p[1].x + 2.f * t * p[2].x, p[1].y + 2.f * t * p[2].y, p[1].z + 2.f * t * p[2].z );
    }

    __device__ __forceinline__ float radius( float t ) const { return p[0].w + t * p[1].w + t * t * p[2].w; }

    __device__ __forceinline__ OptixAabb circleBound( float t, float r ) const
    {
        float3 p = position3( t );
        float3 v = tangent3( t );
        return TightBounds::circleBound( p, v, r );
    }

    __device__ __forceinline__ static void findExtremums( float q1, float q2, float& tx1 )
    {
        // We solve D[q0 + q1 t + q2 t^2, t] == 0
        tx1 = ( q2 == 0 ) ? 0 : -0.5f * q1 / q2;
    }

    __device__ __forceinline__ void findExtremums( float4& tx1 ) const
    {
        findExtremums( p[1].x, p[2].x, tx1.x );
        findExtremums( p[1].y, p[2].y, tx1.y );
        findExtremums( p[1].z, p[2].z, tx1.z );
        findExtremums( p[1].w, p[2].w, tx1.w );
    }

    __device__ __forceinline__ void includeExtremums( OptixAabb& bb, float t0, float t1, float tx1, float max_r ) const
    {
        if( ( t0 < tx1 ) == ( tx1 < t1 ) )
        {
            OptixAabb cb = circleBound( tx1, max_r );
            bb.minX      = fminf( bb.minX, cb.minX );
            bb.minY      = fminf( bb.minY, cb.minY );
            bb.minZ      = fminf( bb.minZ, cb.minZ );
            bb.maxX      = fmaxf( bb.maxX, cb.maxX );
            bb.maxY      = fmaxf( bb.maxY, cb.maxY );
            bb.maxZ      = fmaxf( bb.maxZ, cb.maxZ );
        }
    }

    __device__ __forceinline__ OptixAabb paddedBounds( float t0, float t1, const float4& tx1 ) const
    {
        float max_r = fmaxf( radius( t0 ), radius( t1 ) );
        if( ( t0 < tx1.w ) == ( tx1.w < t1 ) )
            max_r = fmaxf( radius( tx1.w ), max_r );
        OptixAabb bb = circleBound( t0, max_r );
        OptixAabb cb = circleBound( t1, max_r );
        bb.minX      = fminf( bb.minX, cb.minX );
        bb.minY      = fminf( bb.minY, cb.minY );
        bb.minZ      = fminf( bb.minZ, cb.minZ );
        bb.maxX      = fmaxf( bb.maxX, cb.maxX );
        bb.maxY      = fmaxf( bb.maxY, cb.maxY );
        bb.maxZ      = fmaxf( bb.maxZ, cb.maxZ );
        includeExtremums( bb, t0, t1, tx1.x, max_r );
        includeExtremums( bb, t0, t1, tx1.y, max_r );
        includeExtremums( bb, t0, t1, tx1.z, max_r );
        return bb;
    }

    float4 p[3];
};

__device__ __forceinline__ float2 decodeSegmentRangeWithInflection( int rangeCode, unsigned char inflection )
{
    float2 u;
    float  infl = inflection / 256.f;

    switch( rangeCode )
    {
        case 2:
            u.x = 0.f;
            u.y = infl;
            break;
        case 3:
            u.x = infl;
            u.y = 1.f;
            break;
        case 4:
            u.x = 0.f;
            u.y = infl * 0.5f;
            break;
        case 5:
            u.x = infl * 0.5f;
            u.y = infl;
            break;
        case 6:
            u.x = infl;
            u.y = infl + 0.5f * ( 1.f - infl );
            break;
        case 7:
            u.x = infl + 0.5f * ( 1.f - infl );
            u.y = 1.f;
            break;
        case 8:
            u.x = 0.f;
            u.y = infl * 0.25f;
            break;
        case 9:
            u.x = infl * 0.25f;
            u.y = infl * 0.5f;
            break;
        case 10:
            u.x = infl * 0.5f;
            u.y = infl * 0.75f;
            break;
        case 11:
            u.x = infl * 0.75f;
            u.y = infl;
            break;
        case 12:
            u.x = infl;
            u.y = infl + ( 1.f - infl ) * 0.25;
            break;
        case 13:
            u.x = infl + ( 1.f - infl ) * 0.25;
            u.y = infl + ( 1.f - infl ) * 0.5;
            break;
        case 14:
            u.x = infl + ( 1.f - infl ) * 0.5;
            u.y = infl + ( 1.f - infl ) * 0.75;
            break;
        case 15:
            u.x = infl + ( 1.f - infl ) * 0.75;
            u.y = 1.f;
            break;
        default:  // rangeCode <= 1
            u.x = 0.f;
            u.y = 1.f;
            break;
    }
    return u;
}

}  // namespace optix_exp
