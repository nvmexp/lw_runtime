// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

#include <KernelUtils/Utility.h>

/*
   Use 3-way video min/max instructions. Only available on >=SM_20 devices.
*/
#define TRAV_USE_VMINMAX

//
// Helper functions
//

inline __device__ float raySpanMin( float3 t0, float3 t1, float t )
{
#if defined( TRAV_USE_VMINMAX )
    return fvmaxmaxf( fminf( t0.x, t1.x ), fminf( t0.y, t1.y ), fvminmaxf( t0.z, t1.z, t ) );  // 4 ops
#else
    return fmaxf( fmaxf( fminf( t0, t1 ) ), t );  // 6 ops
#endif
}

inline __device__ float raySpanMax( float3 t0, float3 t1, float t )
{
#if defined( TRAV_USE_VMINMAX )
    return fvminminf( fmaxf( t0.x, t1.x ), fmaxf( t0.y, t1.y ), fvmaxminf( t0.z, t1.z, t ) );  // 4 ops
#else
    return fminf( fminf( fmaxf( t0, t1 ) ), t );  // 6 ops
#endif
}
