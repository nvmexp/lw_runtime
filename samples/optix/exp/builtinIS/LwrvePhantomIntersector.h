/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
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

#include <private/optix_7_device_private.h>

#include "BuiltinISCompileTimeConstants.h"
#include "BuiltinISHelpers.h"

#define FIX_TRIPLE_KNOTS

#define RAY_TYPE_OCCLUSION 1

// The performance of "regular" models does not improve much if these values are reduced.
// Having bigger MAXITERATIONS (100) helps with wilder models.
#define DT_ACLWRACY 1.0e-5f  // we'll abort iterations when tspan < 10*DT_ACLWRACY
#define MAXITERATIONS 100    // or next iteration >= this (iterations go faster for !valid intersections)

#define CHECK_INTERNAL_SPLITTING 0
#define SKIP_INTERNAL_SPLITTING 1

namespace optix_exp {

enum LwrveBasis
{
    BSPLINE        = 0,
    CATMULLROM     = 1
};

//------------------------------------------------------------------------------
// Lwrves with different layouts; all defined by BSpline points.
//------------------------------------------------------------------------------

// Storing -- and using -- {p0, 6*(p1 - p0), 3*(p2 - p1), 6*(p3 - p2)},
// where {p0, p1, p2, p3} are Bezier control points.

struct Lwrve3ScaledDifferentialBezier
{
    __device__ __forceinline__ Lwrve3ScaledDifferentialBezier() {}
    __device__ __forceinline__ Lwrve3ScaledDifferentialBezier( const float4* q, LwrveBasis basis )
    {
        if( basis == CATMULLROM )
            initializeFromCatmullRom( q );
        else
            initializeFromBSpline( q );
    }
    __device__ __forceinline__ Lwrve3ScaledDifferentialBezier( const float4* q, const float3& offset, LwrveBasis basis )
    {
        if( basis == CATMULLROM )
            initializeFromCatmullRom( q, offset );
        else
            initializeFromBSpline( q, offset );
    }

    // Bspline-to-Bezier = Matrix([[1/6, 2/3, 1/6, 0], [0, 2/3, 1/3, 0], [0, 1/3, 2/3, 0], [0, 1/6, 2/3, 1/6]])
    // Bezier-to-Lwrve3ScaledDifferentialBezier = Matrix([[1, 0, 0, 0], [-6, 6, 0, 0], [0, -3, 3, 0], [0, 0, -6, 6]])
    // Multiply to get:
    // bspline-to-Lwrve3ScaledDifferentialBezier = Matrix([[1/6, 2/3, 1/6, 0], [-1, 0, 1, 0], [0, -1, 1, 0], [0, -1, 0, 1]])

    __device__ __forceinline__ void initializeFromBSpline( const float4* q )
    {
        p[0] = q[0] * (float)( 1. / 6. ) + q[1] * (float)( 4. / 6. ) + q[2] * (float)( 1. / 6. );
        p[1] = q[2] - q[0];
        p[2] = q[2] - q[1];
        p[3] = q[3] - q[1];
    }

    __device__ __forceinline__ void initializeFromBSpline( const float4* q, const float3& origin )
    {
        // Ensure connecting segments meet at the same point, when the lwrve is far away from origin.
        float3 q0 = make_float3( q[0] ) - origin;
        float3 q1 = make_float3( q[1] ) - origin;
        float3 q2 = make_float3( q[2] ) - origin;

        p[0] = make_float4( q0.x * (float)( 1. / 6. ) + q1.x * (float)( 4. / 6. ) + q2.x * (float)( 1. / 6. ),
                            q0.y * (float)( 1. / 6. ) + q1.y * (float)( 4. / 6. ) + q2.y * (float)( 1. / 6. ),
                            q0.z * (float)( 1. / 6. ) + q1.z * (float)( 4. / 6. ) + q2.z * (float)( 1. / 6. ),
                            q[0].w * (float)( 1. / 6. ) + q[1].w * (float)( 4. / 6. ) + q[2].w * (float)( 1. / 6. ) );
        p[1] = q[2] - q[0];
        p[2] = q[2] - q[1];
        p[3] = q[3] - q[1];
    }

    __device__ __forceinline__ void initializeFromCatmullRom( const float4* q )
    {
        // CatmullRom-to-Bezier = Matrix([[0, 1, 0, 0], [-1/6, 1, 1/6, 0], [0, 1/6, 1, -1/6], [0, 0, 1, 0]])
        // Bezier-to-Lwrve3ScaledDifferentialBezier = Matrix([[1, 0, 0, 0], [-6, 6, 0, 0], [0, -3, 3, 0], [0, 0, -6, 6]])
        // Multiply to get:
        // CatmullRom-to-Lwrve3ScaledDifferentialBezier = Matrix([[0, 1, 0, 0], [-1, 0, 1, 0], [0.5, -2.5, 2.5, -0.5], [0, -1, 0, 1]])
        p[0] = q[1];
        p[1] = q[2] - q[0];
        p[2] = q[0] * (float)(1./2.) + q[1] * (float)(-5./2.) + q[2] * (float)(5./2.) + q[3] * (float)(-1./2.);
        p[3] = q[3] - q[1];
    }

    __device__ __forceinline__ void initializeFromCatmullRom( const float4* q, const float3& origin )
    {
        // CatmullRom-to-Bezier = Matrix([[0, 1, 0, 0], [-1/6, 1, 1/6, 0], [0, 1/6, 1, -1/6], [0, 0, 1, 0]])
        // Bezier-to-Lwrve3ScaledDifferentialBezier = Matrix([[1, 0, 0, 0], [-6, 6, 0, 0], [0, -3, 3, 0], [0, 0, -6, 6]])
        // Multiply to get:
        // CatmullRom-to-Lwrve3ScaledDifferentialBezier = Matrix([[0, 1, 0, 0], [-1, 0, 1, 0], [1/2, -5/2, 5/2, -1/2], [0, -1, 0, 1]])

        // Ensure connecting segments meet at the same point, when the lwrve is far away from origin.
        float4 q0 = make_float4( q[0].x - origin.x, q[0].y - origin.y, q[0].z - origin.z, q[0].w );
        float4 q1 = make_float4( q[1].x - origin.x, q[1].y - origin.y, q[1].z - origin.z, q[1].w );
        float4 q2 = make_float4( q[2].x - origin.x, q[2].y - origin.y, q[2].z - origin.z, q[2].w );
        float4 q3 = make_float4( q[3].x - origin.x, q[3].y - origin.y, q[3].z - origin.z, q[3].w );

        p[0] = q1;
        p[1] = q2 - q0;
        p[2] = q[0] * (float)(1./2.) + q[1] * (float)(-5./2.) + q[2] * (float)(5./2.) + q[3] * (float)(-1./2.);
        p[3] = q3 - q1;
    }

    __device__ __forceinline__ float4 lwrvatureNumeratorDerivativePolynom() const
    {
        // Bezier form, this one, and a colwersion
        // bc  = {{q0x, q0y, q0z}, {q1x, q1y, q1z}, {q2x, q2y, q2z}, {q3x, q3y, q3z}}
        // dc  = {{p0x, p0y, p0z}, {p1x, p1y, p1z}, {p2x, p2y, p2z}, {p3x, p3y, p3z}}
        // sub = Thread[Flatten[bc] -> Flatten[dlwrve2Bezier[dc]]];

        float3 x12[3] = {
            // Transpose[CoefficientList[Cross[lwrve'[t, bc], lwrve''[t, bc]], t]]/.sub
            make_float3( -( p[1].z * p[2].y ) + p[1].y * p[2].z, p[1].z * p[2].x - p[1].x * p[2].z,
                         -( p[1].y * p[2].x ) + p[1].x * p[2].y ),
            make_float3( 0.5f * ( 4 * p[1].z * p[2].y - 4 * p[1].y * p[2].z - p[1].z * p[3].y + p[1].y * p[3].z ),
                         0.5f * ( -4 * p[1].z * p[2].x + 4 * p[1].x * p[2].z + p[1].z * p[3].x - p[1].x * p[3].z ),
                         0.5f * ( 4 * p[1].y * p[2].x - 4 * p[1].x * p[2].y - p[1].y * p[3].x + p[1].x * p[3].y ) ),
            make_float3( -( p[2].z * p[3].y ) + 0.5f * ( p[1].z * ( -2 * p[2].y + p[3].y ) )
                             + p[1].y * ( p[2].z - 0.5f * p[3].z ) + p[2].y * p[3].z,
                         p[1].z * ( p[2].x - 0.5f * p[3].x ) + p[2].z * p[3].x - p[2].x * p[3].z
                             + 0.5f * ( p[1].x * ( -2 * p[2].z + p[3].z ) ),
                         -( p[2].y * p[3].x ) + 0.5f * ( p[1].y * ( -2 * p[2].x + p[3].x ) )
                             + p[1].x * ( p[2].y - 0.5f * p[3].y ) + p[2].x * p[3].y ) };

        float3 x13[3] = {
            // Transpose[CoefficientList[Cross[lwrve'[t, bc], lwrve'''[t, bc]], t]]/.sub
            make_float3( 0.5f * ( 4 * p[1].z * p[2].y - 4 * p[1].y * p[2].z - p[1].z * p[3].y + p[1].y * p[3].z ),
                         0.5f * ( -4 * p[1].z * p[2].x + 4 * p[1].x * p[2].z + p[1].z * p[3].x - p[1].x * p[3].z ),
                         0.5f * ( 4 * p[1].y * p[2].x - 4 * p[1].x * p[2].y - p[1].y * p[3].x + p[1].x * p[3].y ) ),
            make_float3( 2 * p[1].y * p[2].z - 2 * p[2].z * p[3].y + p[1].z * ( -2 * p[2].y + p[3].y ) - p[1].y * p[3].z
                             + 2 * p[2].y * p[3].z,
                         2 * p[1].z * p[2].x - 2 * p[1].x * p[2].z - p[1].z * p[3].x + 2 * p[2].z * p[3].x
                             + p[1].x * p[3].z - 2 * p[2].x * p[3].z,
                         2 * p[1].x * p[2].y - 2 * p[2].y * p[3].x + p[1].y * ( -2 * p[2].x + p[3].x ) - p[1].x * p[3].y
                             + 2 * p[2].x * p[3].y ) };

        float q0 = dot3( x12[0], x13[0] );
        float q1 = dot3( x12[1], x13[0] ) + dot3( x12[0], x13[1] );
        float q2 = dot3( x12[2], x13[0] ) + dot3( x12[1], x13[1] );
        float q3 = dot3( x12[2], x13[1] );

        return make_float4( q0, q1, q2, q3 );
    }

    __device__ __forceinline__ float curvature( float t ) const
    {
        // |lwrve' X lwrve''|/|lwrve'|^3
        float3 d1  = velocity3( t );
        float3 d2  = acceleration3( t );
        float  den = length3( d1 );
        if( den == 0.f )
            return LWRVE_FLT_MAX;
        return length3( cross3( d1, d2 ) ) / ( den * den * den );
    }

    __device__ __forceinline__ float approximateInflection() const
    {

        // No need to search for the potential inflections if the lwrve is essentially straight
        // and its width is not varying too match.
        float rmin         = min_radius();
        float nonlinearity = length4( p[2] - 0.5f * p[1] ) + length4( p[1] - 4 * p[2] + p[3] );
        if( nonlinearity < fmaxf( 1e-5f, rmin ) )
            return -1.f;

        float4 kp = lwrvatureNumeratorDerivativePolynom();
        float  roots[3];
        int    nroots = solvelwbic_i( kp.x, kp.y, kp.z, kp.w, roots );
        if( nroots == 0 )
            return -1.f;

        float tmin = -1.f, kmin = LWRVE_FLT_MAX;
        for( int i = 0; i < nroots; i++ )
        {
            float t = roots[i];
            float k = curvature( t );
            if( k < kmin /*&& 1/k < 50*ebb*/ )
            {
                float delta = 0.01f;
                float km    = curvature( t - delta );
                float kp    = curvature( t + delta );
                if( k < km && k < kp )
                {
                    kmin = k;
                    tmin = t;
                    continue;
                }
                if( k > km && k > kp )
                {
                    continue;
                }
                // approximate curvature is ambiguous; do a real thing
                if( km < k )
                {
                    k     = km;
                    delta = -delta;
                }
                else
                {
                    k = kp;
                }
                float tn = t + delta;
                for( int j = 0; j < 10; j++ )
                {
                    tn += delta;
                    if( !( ( 0 <= tn ) == ( tn <= 1 ) ) )
                        break;
                    km = curvature( tn );
                    if( km > k )
                    {
                        tmin = tn - delta;
                        break;
                    }
                    k = km;
                    delta *= 1.5f;
                }
            }
        }

        if( tmin > 0 )
        {
            // Is it really a zigzag?
            float3 pm  = position3( tmin );
            float3 vm  = velocity3( tmin );
            float3 p0l = shortestVectorToLine( position3( 0.f ), pm, vm );
            float3 p1l = shortestVectorToLine( position3( 1.f ), pm, vm );
            float  dd  = dot3( p0l, p1l );
            float  l0  = dot3( p0l, p0l );
            float  l1  = dot3( p1l, p1l );
            float  r0  = radius( 0.f );
            float  r1  = radius( 1.f );
            if( ( dd <= 0.f ) && ( l0 > r0 * r0 ) && ( l1 > r1 * r1 ) )
                return tmin;  // /\/
        }

        return -1;
    }

    __device__ __forceinline__ static float3 terms( float u )
    {
        float uu = u * u;
        float u3 = ( 1.f / 6.f ) * uu * u;
        return make_float3( u3 + 0.5f * ( u - uu ), uu - 4.f * u3, u3 );
    }

    __device__ __forceinline__ void moveTo( const float3& origin )
    {
        // non-differential component
        p[0].x -= origin.x;
        p[0].y -= origin.y;
        p[0].z -= origin.z;
    }

    __device__ __forceinline__ float4 operator*( const float3& n ) const
    {
        return make_float4( dot3( make_float3( p[0] ), n ), dot3( make_float3( p[1] ), n ),
                            dot3( make_float3( p[2] ), n ), dot3( make_float3( p[3] ), n ) );
    }
    __device__ __forceinline__ float4 operator*( const float2& n ) const
    {
        // xy-plane version (orthogonal to ray = {0,0,1})
        return make_float4( p[0].x * n.x + p[0].y * n.y, p[1].x * n.x + p[1].y * n.y, p[2].x * n.x + p[2].y * n.y,
                            p[3].x * n.x + p[3].y * n.y );
    }

    __device__ __forceinline__ float4 x() const { return make_float4( p[0].x, p[1].x, p[2].x, p[3].x ); }
    __device__ __forceinline__ float4 y() const { return make_float4( p[0].y, p[1].y, p[2].y, p[3].y ); }
    __device__ __forceinline__ float4 z() const { return make_float4( p[0].z, p[1].z, p[2].z, p[3].z ); }
    __device__ __forceinline__ float4 w() const { return make_float4( p[0].w, p[1].w, p[2].w, p[3].w ); }

    __device__ __forceinline__ static float position( const float4& px, const float3& uterms )
    {
        return px.x + px.y * uterms.x + px.z * uterms.y + px.w * uterms.z;
    }

    __device__ __forceinline__ static float position( const float4& px, float u ) { return position( px, terms( u ) ); }

    __device__ __forceinline__ float radius( const float3& uterms ) const
    {
        return p[0].w + p[1].w * uterms.x + p[2].w * uterms.y + p[3].w * uterms.z;
    }

    __device__ __forceinline__ float min_radius() const
    {
        // conservative minimum on [0, 1] interval (min of Bezier CP values)
        float v1 = p[0].w + p[1].w / 6.f;
        float v2 = v1 + p[2].w / 3.f;
        float v3 = v2 + p[3].w / 6.f;
        return fminf( fminf( p[0].w, v1 ), fminf( v2, v3 ) );
    }

    __device__ __forceinline__ void transformyz( const float4& px, const float3& axisy, const float3& axisz )
    {
        float3 v;
        v      = make_float3( p[0] );
        p[0].x = px.x;
        p[0].y = dot3( v, axisy );
        p[0].z = dot3( v, axisz );
        v      = make_float3( p[1] );
        p[1].x = px.y;
        p[1].y = dot3( v, axisy );
        p[1].z = dot3( v, axisz );
        v      = make_float3( p[2] );
        p[2].x = px.z;
        p[2].y = dot3( v, axisy );
        p[2].z = dot3( v, axisz );
        v      = make_float3( p[3] );
        p[3].x = px.w;
        p[3].y = dot3( v, axisy );
        p[3].z = dot3( v, axisz );
    }

    __device__ __forceinline__ float3 position3( float u ) const
    {
        float3 q = terms( u );
        return make_float3( p[0] ) + q.x * make_float3( p[1] ) + q.y * make_float3( p[2] ) + q.z * make_float3( p[3] );
    }
    __device__ __forceinline__ float4 position4( float u ) const
    {
        float3 q = terms( u );
        return p[0] + q.x * p[1] + q.y * p[2] + q.z * p[3];
    }

    __device__ __forceinline__ float radius( float u ) const
    {
        // return position(u).w;
        return p[0].w
               + u * ( p[1].w * 0.5f + u * ( ( p[2].w - p[1].w * 0.5f ) + u * ( p[1].w - 4.f * p[2].w + p[3].w ) * ( 1.f / 6.f ) ) );
    }

    __device__ __forceinline__ float max_radius() const
    {
        // conservative maximum on [0, 1] interval (max of Bezier control point values)
        float v1 = p[0].w + p[1].w / 6.f;
        float v2 = v1 + p[2].w / 3.f;
        float v3 = v2 + p[3].w / 6.f;
        return fmaxf( fmaxf( p[0].w, v1 ), fmaxf( v2, v3 ) );
    }
    __device__ __forceinline__ float max_radius( float u1, float u2 ) const
    {
        // exact maximum on [u1, u2] interval
        if( p[1].w == 0.f && p[2].w == 0.f && p[3].w == 0.f )
            return p[0].w;  // a quick bypass for constant width
        // a + 2 b u - c u^2
        float a    = p[1].w;
        float b    = 2.f * p[2].w - p[1].w;
        float c    = 4.f * p[2].w - p[1].w - p[3].w;
        float rmax = fmaxf( radius( u1 ), radius( u2 ) );
        if( fabsf( c ) < 1.e-5f )
        {
            float root1 = clamp( -0.5f * a / b, u1, u2 );
            return fmaxf( rmax, radius( root1 ) );
        }
        else
        {
            float det   = b * b + a * c;
            det         = det <= 0.0f ? 0.0f : sqrtf( det );
            float root1 = clamp( ( b + det ) / c, u1, u2 );
            float root2 = clamp( ( b - det ) / c, u1, u2 );
            return fmaxf( rmax, fmaxf( radius( root1 ), radius( root2 ) ) );
        }
    }

    __device__ __forceinline__ bool regular( float bl ) const
    {
        // dot(velocity3(0), velocity3(1)) > 0
        float v0v1 = p[1].x * p[3].x + p[1].y * p[3].y + p[1].z * p[3].z;
        return (bool)( v0v1 > 0.f && v0v1 < bl );
    }

    __device__ __forceinline__ float3 velocity3( float u ) const
    {
        // return make_float3(velocity4(u));
        float v = 1.f - u;
        return 0.5f * v * v * make_float3( p[1] ) + 2.f * v * u * make_float3( p[2] ) + 0.5f * u * u * make_float3( p[3] );
    }

    __device__ __forceinline__ float4 velocity4( float u ) const
    {
        float v = 1.f - u;
        return 0.5f * v * v * p[1] + 2.f * v * u * p[2] + 0.5f * u * u * p[3];
    }

    __device__ __forceinline__ float4 acceleration4( float u ) const
    {
        return 2.f * p[2] - p[1] + ( p[1] - 4.f * p[2] + p[3] ) * u;
    }

    __device__ __forceinline__ float3 acceleration3( float u ) const { return make_float3( acceleration4( u ) ); }

    __device__ __forceinline__ static bool hasRoots( const float4& px, float t0, float t1 )
    {
        // f' = a + 2 b t - c t^2
        float a  = px.y;
        float b  = 2.f * px.z - a;
        float c  = a + 2.f * b - px.w;
        float f0 = a + t0 * ( 2.f * b - c * t0 );
        float f1 = a + t1 * ( 2.f * b - c * t1 );
        if( f0 * f1 < 0.f )
            return true;
        float t2 = b / c;  // find f'' == 0
        if( t2 < t0 || t2 > t1 )
            return false;
        float f2 = a + t2 * b;
        return f0 * f2 < 0.f;
    }

    __device__ __forceinline__ static void slabRoots( const float4& px, float& root1, float& root2 )
    {
        // a + 2 b t - c t^2
        float a = px.y;
        float b = 2.f * px.z - a;
        float c = a + 2.f * b - px.w;

        float det = b * b + a * c;
        if( det <= 0.f )
        {
            root1 = root2 = -1.f;
            return;
        }

        // choose more robust expression (could also use b*det)
        det   = ( a == 0.f ) ? -b : ( ( c == 0.f ) ? b : copysignf( sqrtf( det ), a ) );
        root1 = ( b * det <= 0.f ) ? ( ( b - det ) / c ) : ( -a / ( b + det ) );
        root2 = -a / ( root1 * c );  // Vieta
    }

    __device__ __forceinline__ static float minRoot( const float4& px )
    {
        // f' = a + 2 b t - c t^2
        float a = px.y;
        float b = 2.f * px.z - a;
        float c = 2.f * px.z + b - px.w;

        float det = b * b + a * c;
        if( det <= 0.f )
        {
            return -1.f;
        }

        float root1, root2;
        det   = ( a == 0.f ) ? -b : ( ( c == 0.f ) ? b : copysignf( sqrtf( det ), a ) );
        root1 = ( b * det <= 0.f ) ? ( ( b - det ) / c ) : ( -a / ( b + det ) );
        root2 = -a / ( root1 * c );  // Vieta
        // we choose the root, for which f'' > 0 (i.e. f has a minimum)
        return ( det > 0.f ) ? root1 : root2;
    }

    float4 p[4];
};


// (basis is not used, but needed to support uniform interface with cubic lwrves)
struct Lwrve2DifferentialBezier
{
    __device__ __forceinline__ Lwrve2DifferentialBezier() {}
    __device__ __forceinline__ Lwrve2DifferentialBezier( const float4* q, LwrveBasis basis )
    {
        initializeFromBSpline( q );
    }
    __device__ __forceinline__ Lwrve2DifferentialBezier( const float4* q, const float3& origin, LwrveBasis basis )
    {
        initializeFromBSpline( q, origin );
    }

    __device__ __forceinline__ void initializeFromBSpline( const float4* q )
    {
        p[0] = q[0] * ( 1.f / 2.f ) + q[1] * ( 1.f / 2.f );
        p[1] = q[1] - q[0];
        p[2] = q[0] * ( 1.f / 2.f ) + q[2] * ( 1.f / 2.f ) - q[1];
    }

    __device__ __forceinline__ void initializeFromBSpline( const float4* q, const float3& origin )
    {
        float3 q0 = make_float3( q[0] ) - origin;
        float3 q1 = make_float3( q[1] ) - origin;
        p[0]      = 0.5f * make_float4( q0.x + q1.x, q0.y + q1.y, q0.z + q1.z, q[0].w + q[1].w );
        p[1]      = q[1] - q[0];
        p[2]      = 0.5f * ( q[2] - q[1] - p[1] );
    }

    __device__ __forceinline__ static float2 terms( float u ) { return make_float2( u, u * u ); }

    __device__ __forceinline__ void moveTo( const float3& origin )
    {
        // non-differential component
        p[0].x -= origin.x;
        p[0].y -= origin.y;
        p[0].z -= origin.z;
    }

    __device__ __forceinline__ float3 operator*( const float3& n ) const
    {
        return make_float3( dot3( make_float3( p[0] ), n ), dot3( make_float3( p[1] ), n ), dot3( make_float3( p[2] ), n ) );
    }
    __device__ __forceinline__ float3 operator*( const float2& n ) const
    {
        // xy-plane version (orthogonal to ray = {0,0,1})
        return make_float3( p[0].x * n.x + p[0].y * n.y, p[1].x * n.x + p[1].y * n.y, p[2].x * n.x + p[2].y * n.y );
    }

    __device__ __forceinline__ float3 x() const { return make_float3( p[0].x, p[1].x, p[2].x ); }
    __device__ __forceinline__ float3 y() const { return make_float3( p[0].y, p[1].y, p[2].y ); }
    __device__ __forceinline__ float3 z() const { return make_float3( p[0].z, p[1].z, p[2].z ); }
    __device__ __forceinline__ float3 w() const { return make_float3( p[0].w, p[1].w, p[2].w ); }

    __device__ __forceinline__ static float position( const float3& px, const float2& uterms )
    {
        return px.x + px.y * uterms.x + px.z * uterms.y;
    }

    __device__ __forceinline__ static float position( const float3& px, float u ) { return position( px, terms( u ) ); }

    __device__ __forceinline__ float radius( const float2& uterms ) const
    {
        return p[0].w + p[1].w * uterms.x + p[2].w * uterms.y;
    }

    __device__ __forceinline__ void transformyz( const float3& px, const float3& axisy, const float3& axisz )
    {
        float3 v;
        v      = make_float3( p[0] );
        p[0].x = px.x;
        p[0].y = dot3( v, axisy );
        p[0].z = dot3( v, axisz );
        v      = make_float3( p[1] );
        p[1].x = px.y;
        p[1].y = dot3( v, axisy );
        p[1].z = dot3( v, axisz );
        v      = make_float3( p[2] );
        p[2].x = px.z;
        p[2].y = dot3( v, axisy );
        p[2].z = dot3( v, axisz );
    }

    __device__ __forceinline__ float3 position3( float u ) const
    {
        return make_float3( p[0] ) + u * make_float3( p[1] ) + u * u * make_float3( p[2] );
    }

    __device__ __forceinline__ float4 position4( float u ) const { return p[0] + u * p[1] + u * u * p[2]; }

    __device__ __forceinline__ float radius( float u ) const { return p[0].w + u * ( p[1].w + u * p[2].w ); }

    __device__ __forceinline__ float min_radius( float u1 = 0.f, float u2 = 1.f ) const
    {
        float root1 = clamp( -0.5f * p[1].w / p[2].w, u1, u2 );
        return fminf( fminf( radius( u1 ), radius( u2 ) ), radius( root1 ) );
    }

    __device__ __forceinline__ float max_radius( float u1 = 0.f, float u2 = 1.f ) const
    {
        if( p[1].w == 0.f && p[2].w == 0.f )
            return p[0].w;  // a quick bypass for constant width
        float root1 = clamp( -0.5f * p[1].w / p[2].w, u1, u2 );
        return fmaxf( fmaxf( radius( u1 ), radius( u2 ) ), radius( root1 ) );
    }

    __device__ __forceinline__ bool regular( float bl ) const
    {
        // dot(velocity3(0), velocity3(1)) > 0
        float v0v1 =
            p[1].x * ( p[1].x + 2.f * p[2].x ) + p[1].y * ( p[1].y + 2.f * p[2].y ) + p[1].z * ( p[1].z + 2.f * p[2].z );
        return (bool)( v0v1 > 0.f && v0v1 < bl );
    }

    __device__ __forceinline__ float3 velocity3( float u ) const
    {
        return make_float3( p[1] ) + 2.f * u * make_float3( p[2] );
    }

    __device__ __forceinline__ float4 velocity4( float u ) const { return p[1] + 2.f * u * p[2]; }

    __device__ __forceinline__ float3 acceleration3( float u ) const { return 2.f * make_float3( p[2] ); }

    __device__ __forceinline__ float4 acceleration4( float u ) const { return 2.f * p[2]; }

    __device__ __forceinline__ static bool hasRoots( const float3& px, float t0, float t1 )
    {
        // linear
        float a  = px.y;
        float b  = 2.f * px.z;
        float f0 = a + b * t0;
        float f1 = a + b * t1;
        return f0 * f1 < 0.f;
    }

    __device__ __forceinline__ static void slabRoots( const float3& px, float& root1, float& root2 )
    {
        // linear
        float a = px.y;
        float b = px.z;
        root1   = -1.f;  // will not be used
        root2   = b ? -0.5f * a / b : 1.f;
    }

    __device__ __forceinline__ static float minRoot( const float3& px )
    {
        // a single one
        float a = px.y;
        float b = px.z;
        return b ? -0.5f * a / b : 1.f;
    }

    float4 p[3];
};

__device__ __forceinline__ float intersectCylinder( const float3& c0, const float3& cd, float r )
{
    float c    = cd.x * cd.x + cd.y * cd.y;
    float cdz2 = cd.z * cd.z;
    float ddd  = c + cdz2;  // dot(cd, cd)

    // solve a - 2 b s + c s^2 for ray ^ cylinder
    float dp  = c0.x * c0.x + c0.y * c0.y;
    float cdd = c0.x * cd.x + c0.y * cd.y;
    float cxd = c0.x * cd.y - c0.y * cd.x;
    float r2  = r * r;

    float b = -cd.z * cdd;
    float a = cxd * cxd + dp * cdz2 - ddd * r2;

    float det = b * b - a * c;
    det       = sqrtf( fmaxf( 0.f, det ) );
    float s0  = ( b - det ) / c;
    return ( s0 * cd.z - cdd ) / ddd;  // dt
}

// Finding intersection(s) with Ray({0,0,0}, {0,0,1}).

template <typename LwrveType>
struct PhantomIntersector : public LwrveType
{
    __device__ __forceinline__ PhantomIntersector( const float4* q, LwrveBasis basis )
        : LwrveType( q, basis )
    {
    }

    __device__ __forceinline__ PhantomIntersector( const float4* q, const float3& origin, LwrveBasis basis )
        : LwrveType( q, origin, basis )
    {
    }

    __device__ __forceinline__ bool intersection( float u )
    {
        // cone is defined by base center c0, radius r, axis cd, and slant dr
        float ua = u;
    redux:
        cd      = LwrveType::velocity4( ua );
        float c = cd.x * cd.x + cd.y * cd.y;
        cdz2    = cd.z * cd.z;
        ddd     = c + cdz2;  // dot(cd, cd)
#if defined( FIX_TRIPLE_KNOTS )
        if( ddd < 1.e-14f )
        {  // ~ FLT_EPSILON^2
            // could happen at endpoints with the merged control points
            // (for which only cubic coefficient is non-zero)
            float hedge = 0.1f;
            ua += hedge;
            goto redux;
        }
#endif  // FIX_TRIPLE_KNOTS

        c0 = LwrveType::position4( u );

        // solve a - 2 b s + c s^2 for ray ^ cone
        dp        = c0.x * c0.x + c0.y * c0.y;  // all possible combinations
        float cdd = c0.x * cd.x + c0.y * cd.y;  // of x*y terms
        float cxd = c0.x * cd.y - c0.y * cd.x;  // cross(cd,c0).z
        float dr  = cd.w;

        sp = cd.z ? cdd / cd.z : 0.f;  // will add c0.z latter

        if( dr * cd.z > 0.f )
        {
            float3 ca     = LwrveType::acceleration3( u );
            float  adjust = ( c0.x * ca.x + c0.y * ca.y - sp * ca.z ) / ddd;
            if( adjust < -1.05f || adjust > -0.95f )
                dr /= fabsf( 1.f + adjust );
        }

        float r   = c0.w;
        float drr = r * dr;
        float r2  = r * r;

        float b  = cd.z * ( drr - cdd );
        float a0 = 2.f * drr * cdd + cxd * cxd + dp * cdz2;
        float a  = a0 - ddd * r2;

        float det = b * b - a * c;
        det       = sqrtf( fmaxf( 0.f, det ) );
        if( det * b > 0.f )
        {
            s0 = a / ( b + det );
        }
        else
        {
            s0 = ( b - det ) / c;
        }
        // We will add c0.z to s0 and sp later if needed
        dt = ( s0 * cd.z - cdd ) / ddd;  // wrt lwrve.u
        dc = s0 * s0 + dp;               // |(ray ^ cone)  - c0|^2
        dp += sp * sp;                   // |(ray ^ plane) - c0|^2

        return det > 0.f;
    }

    __device__ __forceinline__ float distanceToCone() const
    {
        return s0 + c0.z;  // from {0,0}
    }

    __device__ __forceinline__ float distanceToPlane() const
    {
        return sp + c0.z;  // from {0,0}
    }

    // lwll by *this * slab_normal - this->w(); then iterate with phantom
    template <int skip_internal_splitting>
    __device__ __forceinline__ void intersect( const float3& rayOrigin,
                                               const float3& rayDirection,
                                               int           raytype,
                                               float         raytmin,
                                               float         raytmax,
                                               float&        t,  // lwrve
                                               float&        s,  // ray
                                               float         t0,
                                               float         t1 )
    {
        // Intersect the lwrve defined by bspline_points on [lwrve.t0, lwrve.t1];
        // the actual intersection may be found outside the interval (but inside [0,1]).
        // Both dt_aclwracy and r2_adjustment are used for stopping the iterations.
        // When |(ray ^ cone(t))  - lwrve(t)|^2 < r2_adjustment * r^2, we ascertain that we're
        // within DT_ACLWRACY of the true root since we use the final dt value to further improve
        // ray.s without finding phantom.intersection() again and the colwergence rate is 2 orders of magnitude on average.

        // LwrveType dbg; for (int i = 0; i < sizeof(LwrveType::p) / sizeof(*LwrveType::p); i++) dbg.p[i] = LwrveType::p[i];

        // If there is no intersection, this value is returned.
        s = t = LWRVE_FLT_MAX;

        //LwrveType::moveTo( rayOrigin ); // is needed if not done in constructor

        // RCC is defined by the ray and the lwrve
        float        tm          = ( t0 + t1 ) / 2.f;
        float3       pm          = LwrveType::position3( tm );
        const float3 base        = LwrveType::velocity3( tm );  // alternatives: p1 - p0 or LwrveType::between3(t0, t1)
        float3       rd          = rayDirection;                // may not have unit length
        float3       slab_normal = cross3( rd, base );

        float irlen       = dot3( rd, rd );
        float base_length = dot3( base, base );
        float slab_length = dot3( slab_normal, slab_normal );
        bool  ray_II_base = slab_length < 1.e-3f * base_length;  // 3% cases
        if( ray_II_base )
        {  // ||
            // We will change the separation plane to
            slab_normal = cross3( rd, slab_normal );
            slab_length *= irlen;
            // slab_normal = p0 - dot(p0, rd) * rd; slab_length = dot(slab_normal, slab_normal);
        }

        float v0    = dot3( pm, slab_normal );
        slab_length = sqrtf( slab_length );
        float rsign = copysignf( 1.f / slab_length, v0 );
        slab_normal.x *= rsign;
        slab_normal.y *= rsign;
        slab_normal.z *= rsign;
        auto  pxc = *this * slab_normal;
        auto  pxw = pxc - this->w();
        float px0 = LwrveType::position( pxw, t0 );
        float px1 = LwrveType::position( pxw, t1 );
        if( px0 >= 0.f && px1 >= 0.f )
        {
            // We can ascertain the separation only if the offset surface projection
            // to the slab_normal is positive, i.e. separated from the ray = axis z.
            // It is similar to using planes of the oriented BB for the lwrve.
            float xroot = LwrveType::minRoot( pxw );
            if( xroot < t0 || xroot > t1 || LwrveType::position( pxw, xroot ) >= 0.f )
            {
                return;
            }
        }
        irlen = 1.0f / sqrtf( irlen );
        rd.x *= irlen;
        rd.y *= irlen;
        rd.z *= irlen;

        float3 axisy = cross3( rd, slab_normal );
        LwrveType::transformyz( pxc, axisy, rd );

        // Lwlling by y or z makes it slower, even though such cases exist.

        // If early_exit is true, we'll accept the first found intersection, otherwise we'd look at the other end as well.
        // Check the monotonicity of the lwrve for the two orthogonal directions.
        float rmax       = LwrveType::max_radius();  // conservative maximum on [0, 1] interval
        bool  hasa       = LwrveType::hasRoots( *this * make_float2( 2.f, -1.f ), t0, t1 );  //|  \/  |
        bool  hasb       = LwrveType::hasRoots( *this * make_float2( 1.f, 2.f ), t0, t1 );   //|  /\  |
        float rmin       = LwrveType::min_radius();
        bool  early_exit = LwrveType::regular( 10.f * base_length ) && rmax < 2.f * rmin && !( hasa || hasb );

        int                dropped    = 2;
        float              tha        = t1 - t0;
        const unsigned int num_splits = __float2uint_rn( 1.f / tha );
        if( num_splits <= 2 && !early_exit && !skip_internal_splitting )
        {
            // -split 1 and -split 2 will end up here;
            // let's see if we can shrink [t0, t1] interval:
            // tic/tac/toe == true if the corresponding subinterval can be dropped
            bool  tic = false;                          // |X| | |
            bool  tac = false;                          // | |X| |
            bool  toe = false;                          // | | |X|
            auto  pyc = this->y();                      // use this axis for ray/lwrve separation
            float root1, root2;                         // need 2 roots since we don't know if we'll use min or max
            LwrveType::slabRoots( pyc, root1, root2 );  // cubic distance extrema
            // 3 splits are defined by 4 points on [t0, t1] interval: | | | |
            // we use asymmetric processing to minimize # of instructions
            // (see toe for the original style)
            tha *= ( 1.f / 3.f );
            float t2     = t0 + tha;
            float y0     = LwrveType::position( pyc, t0 );  // Y | | |
            float y1     = LwrveType::position( pyc, t2 );  // | Y | |
            float f1     = ( t0 < root1 && root1 < t1 ) ? LwrveType::position( pyc, root1 ) : LWRVE_FLT_MAX;
            float f2     = ( t0 < root2 && root2 < t1 ) ? LwrveType::position( pyc, root2 ) : LWRVE_FLT_MAX;
            float rs     = copysignf( rmax, y1 );                // for the first 2 (of 3) parts
            bool  ys     = y1 < rs;                              // true if y1 < -rmax; false if y1 > rmax
            bool  tictac =                                       // use distance(lwrve(roots), ray) on |X|X| |
                ( f1 == LWRVE_FLT_MAX || ( f1 < rs ) == ys ) &&  // true if roots are outside [t0, t1] or
                ( f2 == LWRVE_FLT_MAX || ( f2 < rs ) == ys );    // "coordinated" with y1
            tic = tictac && ( fmaxf( y0, y1 ) < -rmax || fminf( y0, y1 ) > rmax );
            t2  = t2 + tha;
            y0  = LwrveType::position( pyc, t2 );  // | | Y |
            tac = tictac && ( fmaxf( y0, y1 ) < -rmax || fminf( y0, y1 ) > rmax );
            y1  = LwrveType::position( pyc, t1 );  // | | | Y
            if( fmaxf( y0, y1 ) < -rmax || fminf( y0, y1 ) > rmax )
            {
                rs  = copysignf( rmax, y1 );
                ys  = y1 < rs;
                toe = ( f1 == LWRVE_FLT_MAX || ( f1 < rs ) == ys ) && ( f2 == LWRVE_FLT_MAX || ( f2 < rs ) == ys );
            }

            dropped = tic + tac + toe;
            if( dropped == 3 )
                return;
            t0 += ( tic ? (float)( 1 + tac ) * tha : 0.f );
            t1 -= ( toe ? (float)( 1 + tac ) * tha : 0.f );
        }

        bool lwrve_caps_off = optix_ext::optixPrivateGetCompileTimeConstant( BUILTIN_IS_COMPILE_TIME_CONSTANT_LWRVE_CAPSOFF );

        float dmz    = dot3( rd, base );
        float dz     = copysignf( 1.0f, dmz );
        float tnext  = dz > 0.f ? t0 : t1;
        float tstart = tnext;
        pm        = make_float3( fabsf( v0 ) / slab_length, dot3( pm, axisy ), 0.f );    // == LwrveType::position3(tm);
        float3 dm = make_float3( dot3( slab_normal, base ), dot3( axisy, base ), dmz );  // == LwrveType::velocity3(tm);
        float  vl2 = dm.x * dm.x + dm.y * dm.y;  // if (ray_II_base) y = 0; else x = 0;
        float  vl3 = vl2 + dmz * dmz;
        // Explore a possibility of an early_exit( and guess tnext value in such a case ).
        // We'll do it only if interval's length is less than 1 and lwrve's thickness < its length.
        if( dropped && slab_length > rmax )
        {
            if( early_exit || raytype == RAY_TYPE_OCCLUSION )
            {
                // lwrve is flat-ish, let's predict tnext
                tnext = tm + intersectCylinder( pm, dm, rmax );
                tnext = __saturatef( tnext );

                // Check endpoints if ray || lwrve (more or less)
                if( !lwrve_caps_off && ( raytype != RAY_TYPE_OCCLUSION && vl3 > 4.f * vl2 ) )
                {
                    float3 cd = LwrveType::velocity3( tstart );
                    if( tstart == ( cd.z < 0.f ) )
                    {
                        float4 c0  = LwrveType::position4( tstart );
                        float  cdd = c0.x * cd.x + c0.y * cd.y;
                        float  sp  = cdd / cd.z;
                        float  s0  = c0.z + sp;
                        if( s0 > raytmin )
                        {
                            float dp = c0.x * c0.x + c0.y * c0.y + sp * sp;
                            float r2 = c0.w * c0.w;
                            if( dp <= r2 )
                            {
                                t = tstart;
                                s = s0;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            early_exit = false;
        }

        early_exit &= vl3 > this->p[1].w * this->p[1].w;

        // For wildly swinging lwrves, we'll increase # of the internal steps even more

        if( hasa && hasb )
            tha *= 0.5f;

        // signed step: it will be used to adjust lwrve's steps
        float ths = tha * dz;
        // For 2 splits, we'll do 6 effective splits anyway, no reason to reduce the step even further
        const int np = sizeof( LwrveType::p ) / sizeof( *LwrveType::p );  // 4 for cubic and 3 for quadratic lwrves
        if( num_splits != 2 || ( skip_internal_splitting && np == 4 ) )
            ths *= 0.25f * (float)( dropped + 2 );
        tha *= 0.66667f;  // will be used to clamp dt (by 2/3 interval in most cases)
                          // (on purpose the value is slightly larger than 2/3)

        const float r2_adjustment = 1.f + 25.f * (float)( 1 + raytype ) * DT_ACLWRACY;  // * r2

        // we need RAY_TYPE_OCCLUSION == 1;
        // jobs will be decremented once we reach "other" endpoint;
        // we stop stepping inside the interval (i.e. setting tnext = tstart + ths) if jobs <= 2
        int jobs = early_exit ? RAY_TYPE_OCCLUSION : 3;
        int dbgi = 0;  // true # of iterations: unless it is used, it will not change the code

        int exit_at = early_exit ? ray_II_base : -1;

    restart:

        int iteration = 0;

        // anycloser will be true if |ray ^ plane - lwrve| < r at one of the iterations.
        // It will be used for the forced rcx determination when the current interval is small.
        bool anycloser = false;
        // interval [tpos, tneg] brackets the root; these values will be used only if dtpos*dtneg < 0.
        float told1 = 0, dtold1 = 0.f;
        float told2, dtold2     = 0.f;

        while( true )
        {
            dbgi++;
            // valid = !phantom
            bool  valid    = intersection( tnext );
            float r2       = radius2();
            bool  endpoint = tnext == 0.f || tnext == 1.f;
            float d2 = endpoint ? dp : fminf( dp, dc );  // need dp for the endpoints (which corresponds to the plane hit)
            bool  closer = d2 < ( 1.f + 50.f * DT_ACLWRACY ) * r2;
            float raytp  = distanceToPlane();
            float raytc  = distanceToCone();

            if( raytype == RAY_TYPE_OCCLUSION )
            {
                if( dp < ( 1.f + DT_ACLWRACY ) * r2 && raytp > 0.f )
                {  // do not use dc here
                    // We already know that there is a hit and accuracy is not important.
                    if( raytmin < raytp )
                    {
                        s = raytp * irlen;
                        return;
                    }
                    goto next;
                }
                if( !( raytc > 0.f ) && iteration > 1 )
                {
                    goto next;
                }
            }

            {
                float dtlwrr = dt;
                float tlwrr  = tnext;

                if( dtold1 * dtlwrr < 0.f )
                {
                    tnext  = ( dtlwrr * told1 - dtold1 * tlwrr ) / ( dtlwrr - dtold1 );
                    told2  = told1;
                    dtold2 = dtold1;
                }
                else if( dtold2 * dtlwrr < 0.f )
                {
                    tnext = ( dtlwrr * told2 - dtold2 * tlwrr ) / ( dtlwrr - dtold2 );
                }
                else if( fabsf( dtold1 ) > 2.f * fabsf( dtlwrr ) )
                {
                    tnext       = ( dtlwrr * told1 - dtold1 * tlwrr ) / ( dtlwrr - dtold1 );
                    float dtmax = fminf( tha, 2.f * fabsf( dtlwrr ) );
                    tnext       = clamp( tnext, tlwrr - dtmax, tlwrr + dtmax );
                    tnext       = __saturatef( tnext );  // for segment affinity, use clamp(tnext,t0,t1)
                }
                else
                {
                    // Clamping helps for wild lwrves, but not for the plain-vanilla models
                    // since it increases # of iterations if we start at the "wrong" buttend.
                    dtlwrr = clamp( dtlwrr, -tha, tha );
                    tnext  = tlwrr + dtlwrr;
                    told2  = told1;
                    dtold2 = dtold1;
                    tnext  = clamp( tnext, tlwrr - tha, tlwrr + tha );
                    tnext  = __saturatef( tnext );  // for segment affinity, use clamp(tnext,t0,t1)
                }

                anycloser |= ( closer && valid );

                bool good = false;
                if( anycloser )
                {
                    if( iteration == exit_at && fabsf( dtlwrr ) < 0.05f && 0.02f < tnext && tnext < 0.98f )
                    {
                        good = dc < 1.01f * r2_adjustment * r2;
                    }
                    else
                    {
                        // the scaling between segt and rayt is defined by
                        // the orthogonality factor = cd.cd/cd.z^2
                        float factor = fminf( valid ? 12.5f : 100.0f, ddd / cdz2 );  // keep it
                        // it is !phantom rcx and we're close to the offset surface
                        good = ( ( 1.f + factor * fabsf( dtlwrr ) ) * dc < 1.01f * r2_adjustment * r2 )
                               && ( dc > ( 0.95f + 0.04f * rmin / rmax ) * r2 );
                    }
                }

                // rcx is proclaimed if ray.t > 0 and we either
                // 1. have a valid intersection strictly inside ]0, 1[ interval or
                // 2. hit a buttend
                // For !early_exit, we will continue searching for a closer one.

                {
                    if( !lwrve_caps_off )
                    {
                        int tint = float_as_int( tnext * good );
                        good     = ( 0 != tint ) == ( tint != 0x3f800000 );  // ]0,1[
                    }
                    if( good )
                    {
                        closer = raytc < s;
                        if( ( raytmin < raytc ) == closer )
                        {
                            // rcx inside ]0,1[; we will use the improved values:
                            // rayt was computed for the current tnext, however, it will be projected to
                            // the predicted one using surface_normal.
                            s = raytc;
                            t = tnext;  // the predicted one
                        }
                    }
                    else if( ( tlwrr == static_cast<float>( cd.z < 0 ) || cd.z == 0 ) && closer )
                    {
                        if( !lwrve_caps_off )
                        {
                            // It is a buttend hit, have to use s = ray ^ plane
                            // (for the connected lwrves, it will be occluded by the adjacent lwrve).
                            // If we had rcx before, which is very close, we'll prefer a buttend hit.
                            if( raytmin < raytp && raytp < s || fabsf( t - tlwrr ) < 50.f * DT_ACLWRACY )
                            {
                                // The adjustment is to treat rim hits as inside surface hits
                                // (it helps avoiding fireflies in some situations).
                                if( cd.z && dp < 0.9999f * r2 )
                                {
                                    s = raytp;
                                    t = tlwrr;
                                }
                                else
                                {
                                    s = raytc;
                                    t = tlwrr ? 1.f - 1.e-7f : 1.e-7f;
                                }
                            }
                        }
                    }
                    else if( tnext == static_cast<float>( cd.z < 0.f ) && tnext != tlwrr )
                    {
                        iteration++;  // let's give tnext a chance to prove
                        continue;     // that it is an endpoint hit
                    }
                    else if( tnext != tlwrr )
                    {
                        if( iteration > 3 )
                        {
                            if( !valid && fabsf( dtlwrr ) < 5.0e-7f )
                                goto next;  // unlikely to find the real rcx with such a small step
                        }
                        // First  term: we'll declare 'no intersection' (for the current segment) if either
                        //              the interval is very small or we are stuck (at either buttend).
                        // Second term: for AO rays, we will try moving faster.

                        iteration += static_cast<int>( (float)MAXITERATIONS * DT_ACLWRACY
                                                       / ( 10.f * DT_ACLWRACY + fabsf( tnext - tlwrr ) ) )
                                     + ( 5 + 2 * ( 5 * raytype - valid - anycloser ) );
                        if( iteration < ( raytc < 0.f ? MAXITERATIONS / 3 : MAXITERATIONS ) )
                        {
                            // If we cannot bracket the root after so many iterations,
                            // move faster, especially for the phantom intersections.
                            if( iteration > 12 )
                            {
                                if( !valid )
                                    iteration += MAXITERATIONS / 8;
                            }
                            told1  = tlwrr;
                            dtold1 = dtlwrr;
                            continue;  // to the next iteration
                        }
                    }
                }
            }

        next:
            if( jobs <= 2 )
                break;

            // Restart iterations using tnext and
            // make sure we test "other" endpoint only once
            // (0.0001f is used to hedge against numerical errors in tnext for multiple inner steps).

            tnext = tstart + ths;
            jobs -= static_cast<int>( tnext < t0 + 0.0001f || tnext > t1 - 0.0001f );
            tstart = tnext;
            tnext  = __saturatef( tnext );
            goto restart;
        }

        if( s != LWRVE_FLT_MAX )
        {
            s = s * irlen;
            if( lwrve_caps_off )
            {
                if( t >= 1.f )
                    t = 1.f - 1.e-7f;
                else if( t <= 0.f )
                    t = 1.e-7f;
            }
        }

        return;
    }

    __device__ __forceinline__ float radius2() const { return c0.w * c0.w; }

    float4 c0;    // lwrve(t)   in RCC (base center + radius)
    float4 cd;    // tangent(t) in RCC (cone's axis + radius')
    float  s0;    // ray.s - c0.z for ray ^ cone(t)
    float  dt;    // dt to the intersection from t
    float  dp;    // |(ray ^ plane(t)) - lwrve(t)|^2
    float  dc;    // |(ray ^ cone(t))  - lwrve(t)|^2
    float  sp;    // ray.s - c0.z for ray ^ plane(t)
    float  ddd;   // dot(cd.cd)
    float  cdz2;  // cd.z^2
};

}  // namespace optix_exp
