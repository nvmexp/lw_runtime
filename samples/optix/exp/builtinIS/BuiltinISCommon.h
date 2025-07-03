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

#include <exp/accel/ExtendedAccelHeader.h>

#include "BuiltinISHelpers.h"

#define LWRVE_FLT_MAX  3.402823e+38F  // no <float.h> in LWCA

__device__ __forceinline__ uint4 __ldg_uint4( const uint4* ptr )
{
    uint4 ret;
    asm volatile( "ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                  : "=r"( ret.x ), "=r"( ret.y ), "=r"( ret.z ), "=r"( ret.w )
                  : "l"( ptr ) );
    return ret;
}
__device__ __forceinline__ uint2 __ldg_uint2( const uint2* ptr )
{
    uint2 ret;
    asm volatile( "ld.global.v2.u32 {%0,%1}, [%2];" : "=r"( ret.x ), "=r"( ret.y ) : "l"( ptr ) );
    return ret;
}

template <class T>
__device__ __forceinline__ T loadCachedAlign16( const T* ptr )
{
    T v;
    for( int ofs = 0; ofs < sizeof( T ); ofs += 16 )
        *(uint4*)( (char*)&v + ofs ) = __ldg_uint4( (uint4*)( (char*)ptr + ofs ) );
    return v;
}
template <class T>
__device__ __forceinline__ T loadCachedAlign8( const T* ptr )
{
    T v;
    for( int ofs = 0; ofs < sizeof( T ); ofs += 8 )
        *(uint2*)( (char*)&v + ofs ) = __ldg_uint2( (uint2*)( (char*)ptr + ofs ) );
    return v;
}


namespace optix_exp {


__device__ __forceinline__ const ExtendedAccelHeader* optixGetExtendedAccelHeaderFromHandle( OptixTraversableHandle gasHandle )
{
    size_t ptrGAS;
    asm( "call (%0), _optix_get_gas_ptr, (%1);" : "=l"( ptrGAS ) : "l"( gasHandle ) : );
    return (const ExtendedAccelHeader*)( (ptrGAS)-EXTENDED_ACCEL_HEADER_SIZE );
}


__device__ __forceinline__ unsigned long long optixGetPrimitiveVA()
{
    unsigned long long primVA;
    asm( "call (%0), _optix_read_prim_va, ();" : "=l"( primVA ) : );
    return primVA;
}


__device__ __forceinline__ float optixGetGASKeyTime()
{
    float time;
    asm( "call (%0), _optix_read_key_time, ();" : "=f"( time ) : );
    return time;
}


__device__ __forceinline__ bool hasExtendedHeader( size_t gasHandle )
{
    // we check the 7th bit of the traversable to see if this is an extended header.
    // this is internal rtcore knowledge which is not part of the rtcore interface.
    // TODO: add an rtcore intrinsic to query this from the handle
    return ( ( gasHandle & 0x40 ) != 0 );
}


__device__ __forceinline__ float2 decodeSegmentRange( LwrveSegmentData data )
{
    float u0, u1;
    if( data.uniform )
    {
        const float rcp = 1.f / (float)( data.un + 1 );
        u0              = ( data.u0 * rcp );
        u1              = ( ( data.u0 + 1 ) * rcp );
    }
    else
    {
        u0 = ( data.u0 / 256.f );
        u1 = ( ( (int)( data.u0 + data.un ) + 1 ) / 256.f );
    }
    return make_float2( u0, u1 );
}

//------------------------------------------------------------------------------

/*
* Intersection information for lwrve intersectors: Ray distance t, parametric position u.
*
*/

struct Intersection
{
    float t;  // ray parameter
    float u;  // lwrve parameter

    __device__ __forceinline__ Intersection()
        : t( LWRVE_FLT_MAX )
    {
    }  // no intersection
    __device__ __forceinline__ Intersection( const float t, const float u )
        : t( t )
        , u( u )
    {
    }
};

}  // namespace optix_exp
