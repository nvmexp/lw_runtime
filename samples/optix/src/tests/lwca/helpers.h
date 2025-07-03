
/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
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

#include <optixu/optixu_math.h>

// Colwert a float3 in [0,1)^3 to a uchar4 in [0,255]^4 -- 4th channel is set to 255
#ifdef __LWDACC__
__device__ __inline__ uchar4 make_color( const float3& c )
{
    return make_uchar4( static_cast<unsigned char>( __saturatef( c.z ) * 255.99f ), /* B */
                        static_cast<unsigned char>( __saturatef( c.y ) * 255.99f ), /* G */
                        static_cast<unsigned char>( __saturatef( c.x ) * 255.99f ), /* R */
                        255u );                                                     /* A */
}
#endif

// Sample Phong lobe relative to U, V, W frame
__host__ __device__ __inline__ float3 sample_phong_lobe( float2 sample, float exponent, float3 U, float3 V, float3 W )
{
    const float power = expf( logf( sample.y ) / ( exponent + 1.0f ) );
    const float phi   = sample.x * 2.0f * (float)M_PIf;
    const float scale = sqrtf( 1.0f - power * power );

    const float x = cosf( phi ) * scale;
    const float y = sinf( phi ) * scale;
    const float z = power;

    return x * U + y * V + z * W;
}

// Create ONB from normal.  Resulting W is parallel to normal
__host__ __device__ __inline__ void create_onb( const float3& n, float3& U, float3& V, float3& W )
{
    W = normalize( n );
    U = cross( W, make_float3( 0.0f, 1.0f, 0.0f ) );

    if( abs( U.x ) < 0.001f && abs( U.y ) < 0.001f && abs( U.z ) < 0.001f )
        U = cross( W, make_float3( 1.0f, 0.0f, 0.0f ) );

    U = normalize( U );
    V = cross( W, U );
}

// Create ONB from normalized vector
__device__ __inline__ void create_onb( const float3& n, float3& U, float3& V )
{
    U = cross( n, make_float3( 0.0f, 1.0f, 0.0f ) );

    if( dot( U, U ) < 1e-3f )
        U = cross( n, make_float3( 1.0f, 0.0f, 0.0f ) );

    U = normalize( U );
    V = cross( n, U );
}

// Compute the origin ray differential for transfer
__host__ __device__ __inline__ float3 differential_transfer_origin( float3 dPdx, float3 dDdx, float t, float3 direction, float3 normal )
{
    float dtdx = -dot( ( dPdx + t * dDdx ), normal ) / dot( direction, normal );
    return ( dPdx + t * dDdx ) + dtdx * direction;
}

// Compute the direction ray differential for a pinhole camera
__host__ __device__ __inline__ float3 differential_generation_direction( float3 d, float3 basis )
{
    float dd = dot( d, d );
    return ( dd * basis - dot( d, basis ) * d ) / ( dd * sqrtf( dd ) );
}

// Compute the direction ray differential for reflection
__host__ __device__ __inline__ float3 differential_reflect_direction( float3 dPdx, float3 dDdx, float3 dNdP, float3 D, float3 N )
{
    float3 dNdx  = dNdP * dPdx;
    float  dDNdx = dot( dDdx, N ) + dot( D, dNdx );
    return dDdx - 2 * ( dot( D, N ) * dNdx + dDNdx * N );
}

// Compute the direction ray differential for refraction
__host__ __device__ __inline__ float3 differential_refract_direction( float3 dPdx, float3 dDdx, float3 dNdP, float3 D, float3 N, float ior, float3 T )
{
    float eta;
    if( dot( D, N ) > 0.f )
    {
        eta = ior;
        N   = -N;
    }
    else
    {
        eta = 1.f / ior;
    }

    float3 dNdx  = dNdP * dPdx;
    float  mu    = eta * dot( D, N ) - dot( T, N );
    float  TN    = -sqrtf( 1 - eta * eta * ( 1 - dot( D, N ) * dot( D, N ) ) );
    float  dDNdx = dot( dDdx, N ) + dot( D, dNdx );
    float  dmudx = ( eta - ( eta * eta * dot( D, N ) ) / TN ) * dDNdx;
    return eta * dDdx - ( mu * dNdx + dmudx * N );
}
