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

// prevent clang from including msvc141 stddef.h, it causes compile errors
#define __LWDACC_RTC__
typedef unsigned long long size_t;

typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;

#define __OPTIX_INCLUDE_INTERNAL_HEADERS__
#include <include/optix_7_types.h>
#undef __OPTIX_INCLUDE_INTERNAL_HEADERS__

#include <exp/accel/ExtendedAccelHeader.h>
#include <exp/pipeline/O7Runtime.h>

typedef float float4  __attribute__( ( ext_vector_type(  4 ) ) );
typedef float float8  __attribute__( ( ext_vector_type(  8 ) ) );
typedef float float12 __attribute__( ( ext_vector_type( 12 ) ) );
typedef float float16 __attribute__( ( ext_vector_type( 16 ) ) );

#define __forceinline__ __attribute__( ( always_inline ) )

extern "C" __forceinline__ float16 RUNTIME_FETCH_LWRVE4_VERTEX_DATA( const unsigned long long ptrGAS,
                                                                     unsigned int             primIdx,
                                                                     const unsigned int       inputIdx,
                                                                     const unsigned int       motionStepCount,
                                                                     const float              motionTimeBegin,
                                                                     const float              motionTimeEnd,
                                                                     const float              time )
{
    const unsigned long long              ehPtr = ptrGAS - EXTENDED_ACCEL_HEADER_SIZE;
    const optix_exp::ExtendedAccelHeader* eh    = (const optix_exp::ExtendedAccelHeader*)ehPtr;

    const unsigned long long indexOffset     = ( (unsigned long long)eh->indexOffset ) << 4;
    const unsigned long long primIndexOffset = ( (unsigned long long)eh->primIndexOffset ) << 4;

    const float4*       vertices = reinterpret_cast<float4*>( ptrGAS + eh->dataOffset );
    const unsigned int* indices  = reinterpret_cast<unsigned int*>( ptrGAS + eh->dataOffset + indexOffset );

    const unsigned int* index_offset_ptr = reinterpret_cast<unsigned int*>( ptrGAS + eh->dataOffset + primIndexOffset );

    float4 v0, v1, v2, v3;

    if( motionStepCount > 1 )
    {
        float local_time = 0.f;
        int   keyIdx     = 0;
        if( motionTimeBegin < motionTimeEnd )
        {
            float motionIntervalCount = (float)motionStepCount - 1.f;
            local_time = ( time - motionTimeBegin ) * motionIntervalCount / ( motionTimeEnd - motionTimeBegin );
            local_time = local_time < 0.f ? 0.f : ( local_time > motionIntervalCount ? motionIntervalCount : local_time );
            keyIdx     = (int)local_time;
            local_time -= (float)keyIdx;
        }

        unsigned int firstVertexIdx = indices[primIdx + index_offset_ptr[inputIdx]] * motionStepCount;

        const float4* q_0 = &vertices[firstVertexIdx + keyIdx];
        const float4* q_1 = &vertices[firstVertexIdx + motionStepCount + keyIdx];
        const float4* q_2 = &vertices[firstVertexIdx + 2 * motionStepCount + keyIdx];
        const float4* q_3 = &vertices[firstVertexIdx + 3 * motionStepCount + keyIdx];

        v0.x = q_0[0].x + local_time * ( q_0[1].x - q_0[0].x );
        v0.y = q_0[0].y + local_time * ( q_0[1].y - q_0[0].y );
        v0.z = q_0[0].z + local_time * ( q_0[1].z - q_0[0].z );
        v0.w = q_0[0].w + local_time * ( q_0[1].w - q_0[0].w );

        v1.x = q_1[0].x + local_time * ( q_1[1].x - q_1[0].x );
        v1.y = q_1[0].y + local_time * ( q_1[1].y - q_1[0].y );
        v1.z = q_1[0].z + local_time * ( q_1[1].z - q_1[0].z );
        v1.w = q_1[0].w + local_time * ( q_1[1].w - q_1[0].w );

        v2.x = q_2[0].x + local_time * ( q_2[1].x - q_2[0].x );
        v2.y = q_2[0].y + local_time * ( q_2[1].y - q_2[0].y );
        v2.z = q_2[0].z + local_time * ( q_2[1].z - q_2[0].z );
        v2.w = q_2[0].w + local_time * ( q_2[1].w - q_2[0].w );

        v3.x = q_3[0].x + local_time * ( q_3[1].x - q_3[0].x );
        v3.y = q_3[0].y + local_time * ( q_3[1].y - q_3[0].y );
        v3.z = q_3[0].z + local_time * ( q_3[1].z - q_3[0].z );
        v3.w = q_3[0].w + local_time * ( q_3[1].w - q_3[0].w );
    }
    else
    {
        unsigned int firstVertexIdx = indices[primIdx + index_offset_ptr[inputIdx]];

        const float4* q = &vertices[firstVertexIdx];

        v0 = q[0];
        v1 = q[1];
        v2 = q[2];
        v3 = q[3];
    }

    float16 data;

    // clang-format off
    data[ 0+0] = v0.x; data[ 0+1] = v0.y; data[ 0+2] = v0.z; data[ 0+3] = v0.w;
    data[ 4+0] = v1.x; data[ 4+1] = v1.y; data[ 4+2] = v1.z; data[ 4+3] = v1.w;
    data[ 8+0] = v2.x; data[ 8+1] = v2.y; data[ 8+2] = v2.z; data[ 8+3] = v2.w;
    data[12+0] = v3.x; data[12+1] = v3.y; data[12+2] = v3.z; data[12+3] = v3.w;
    // clang-format on

    return data;
}

extern "C" __forceinline__ float12 RUNTIME_FETCH_LWRVE3_VERTEX_DATA( const unsigned long long ptrGAS,
                                                                     unsigned int             primIdx,
                                                                     const unsigned int       inputIdx,
                                                                     const unsigned int       motionStepCount,
                                                                     const float              motionTimeBegin,
                                                                     const float              motionTimeEnd,
                                                                     const float              time )
{
    const unsigned long long              ehPtr = ptrGAS - EXTENDED_ACCEL_HEADER_SIZE;
    const optix_exp::ExtendedAccelHeader* eh    = (const optix_exp::ExtendedAccelHeader*)ehPtr;

    const float4*       vertices = reinterpret_cast<float4*>( ptrGAS + eh->dataOffset );
    const unsigned int* indices =
        reinterpret_cast<unsigned int*>( ptrGAS + eh->dataOffset + ( ( (unsigned long long)eh->indexOffset ) << 4 ) );

    const unsigned int* index_offset_ptr =
        reinterpret_cast<unsigned int*>( ptrGAS + eh->dataOffset + ( ( (unsigned long long)eh->primIndexOffset ) << 4 ) );

    float4 v0, v1, v2;

    if( motionStepCount > 1 )
    {
        float local_time = 0.f;
        int   keyIdx     = 0;
        if( motionTimeBegin < motionTimeEnd )
        {
            float motionIntervalCount = (float)motionStepCount - 1.f;
            local_time = ( time - motionTimeBegin ) * motionIntervalCount / ( motionTimeEnd - motionTimeBegin );
            local_time = local_time < 0.f ? 0.f : ( local_time > motionIntervalCount ? motionIntervalCount : local_time );
            keyIdx     = (int)local_time;
            local_time -= (float)keyIdx;
        }

        unsigned int firstVertexIdx = indices[primIdx + index_offset_ptr[inputIdx]] * motionStepCount;

        const float4* q_0 = &vertices[firstVertexIdx + keyIdx];
        const float4* q_1 = &vertices[firstVertexIdx + motionStepCount + keyIdx];
        const float4* q_2 = &vertices[firstVertexIdx + 2 * motionStepCount + keyIdx];

        v0.x = q_0[0].x + local_time * ( q_0[1].x - q_0[0].x );
        v0.y = q_0[0].y + local_time * ( q_0[1].y - q_0[0].y );
        v0.z = q_0[0].z + local_time * ( q_0[1].z - q_0[0].z );
        v0.w = q_0[0].w + local_time * ( q_0[1].w - q_0[0].w );

        v1.x = q_1[0].x + local_time * ( q_1[1].x - q_1[0].x );
        v1.y = q_1[0].y + local_time * ( q_1[1].y - q_1[0].y );
        v1.z = q_1[0].z + local_time * ( q_1[1].z - q_1[0].z );
        v1.w = q_1[0].w + local_time * ( q_1[1].w - q_1[0].w );

        v2.x = q_2[0].x + local_time * ( q_2[1].x - q_2[0].x );
        v2.y = q_2[0].y + local_time * ( q_2[1].y - q_2[0].y );
        v2.z = q_2[0].z + local_time * ( q_2[1].z - q_2[0].z );
        v2.w = q_2[0].w + local_time * ( q_2[1].w - q_2[0].w );
    }
    else
    {
        unsigned int firstVertexIdx = indices[primIdx + index_offset_ptr[inputIdx]];

        const float4* q = &vertices[firstVertexIdx];

        v0 = q[0];
        v1 = q[1];
        v2 = q[2];
    }

    float12 data;

    // clang-format off
    data[ 0+0] = v0.x; data[ 0+1] = v0.y; data[ 0+2] = v0.z; data[ 0+3] = v0.w;
    data[ 4+0] = v1.x; data[ 4+1] = v1.y; data[ 4+2] = v1.z; data[ 4+3] = v1.w;
    data[ 8+0] = v2.x; data[ 8+1] = v2.y; data[ 8+2] = v2.z; data[ 8+3] = v2.w;
    // clang-format on

    return data;
}

extern "C" __forceinline__ float8 RUNTIME_FETCH_LWRVE2_VERTEX_DATA( const unsigned long long ptrGAS,
                                                                    unsigned int             primIdx,
                                                                    const unsigned int       inputIdx,
                                                                    const unsigned int       motionStepCount,
                                                                    const float              motionTimeBegin,
                                                                    const float              motionTimeEnd,
                                                                    const float              time )

{
    const unsigned long long              ehPtr = ptrGAS - EXTENDED_ACCEL_HEADER_SIZE;
    const optix_exp::ExtendedAccelHeader* eh    = (const optix_exp::ExtendedAccelHeader*)ehPtr;

    const float4*       vertices = reinterpret_cast<float4*>( ptrGAS + eh->dataOffset );
    const unsigned int* indices =
        reinterpret_cast<unsigned int*>( ptrGAS + eh->dataOffset + ( ( (unsigned long long)eh->indexOffset ) << 4 ) );

    const unsigned int* index_offset_ptr =
        reinterpret_cast<unsigned int*>( ptrGAS + eh->dataOffset + ( ( (unsigned long long)eh->primIndexOffset ) << 4 ) );

    float4 v0, v1;

    if( motionStepCount > 1 )
    {
        float local_time = 0.f;
        int   keyIdx     = 0;
        if( motionTimeBegin < motionTimeEnd )
        {
            float motionIntervalCount = (float)motionStepCount - 1.f;
            local_time = ( time - motionTimeBegin ) * motionIntervalCount / ( motionTimeEnd - motionTimeBegin );
            local_time = local_time < 0.f ? 0.f : ( local_time > motionIntervalCount ? motionIntervalCount : local_time );
            keyIdx     = (int)local_time;
            local_time -= (float)keyIdx;
        }

        unsigned int firstVertexIdx = indices[primIdx + index_offset_ptr[inputIdx]] * motionStepCount;

        const float4* q_0 = &vertices[firstVertexIdx + keyIdx];
        const float4* q_1 = &vertices[firstVertexIdx + motionStepCount + keyIdx];

        v0.x = q_0[0].x + local_time * ( q_0[1].x - q_0[0].x );
        v0.y = q_0[0].y + local_time * ( q_0[1].y - q_0[0].y );
        v0.z = q_0[0].z + local_time * ( q_0[1].z - q_0[0].z );
        v0.w = q_0[0].w + local_time * ( q_0[1].w - q_0[0].w );

        v1.x = q_1[0].x + local_time * ( q_1[1].x - q_1[0].x );
        v1.y = q_1[0].y + local_time * ( q_1[1].y - q_1[0].y );
        v1.z = q_1[0].z + local_time * ( q_1[1].z - q_1[0].z );
        v1.w = q_1[0].w + local_time * ( q_1[1].w - q_1[0].w );
    }
    else
    {
        unsigned int firstVertexIdx = indices[primIdx + index_offset_ptr[inputIdx]];

        const float4* q = &vertices[firstVertexIdx];

        v0 = q[0];
        v1 = q[1];
    }

    float8 data;

    // clang-format off
    data[ 0+0] = v0.x; data[ 0+1] = v0.y; data[ 0+2] = v0.z; data[ 0+3] = v0.w;
    data[ 4+0] = v1.x; data[ 4+1] = v1.y; data[ 4+2] = v1.z; data[ 4+3] = v1.w;
    // clang-format on

    return data;
}

extern "C" __forceinline__ float4 RUNTIME_FETCH_SPHERE_DATA( const unsigned long long ptrGAS,
                                                             unsigned int             primIdx,
                                                             const unsigned int       inputIdx,
                                                             const unsigned int       motionStepCount,
                                                             const float              motionTimeBegin,
                                                             const float              motionTimeEnd,
                                                             const float              time )
{
    const unsigned long long              ehPtr = ptrGAS - EXTENDED_ACCEL_HEADER_SIZE;
    const optix_exp::ExtendedAccelHeader* eh    = (const optix_exp::ExtendedAccelHeader*)ehPtr;

    const float4*       vertices = reinterpret_cast<float4*>( ptrGAS + eh->dataOffset );
    const unsigned int* index_offset_ptr =
        reinterpret_cast<unsigned int*>( ptrGAS + eh->dataOffset + ( ( (unsigned long long)eh->primIndexOffset ) << 4 ) );

    float4 v;

    if( motionStepCount > 1 )
    {
        float local_time = 0.f;
        int   keyIdx     = 0;
        if( motionTimeBegin < motionTimeEnd )
        {
            float motionIntervalCount = (float)motionStepCount - 1.f;
            local_time = ( time - motionTimeBegin ) * motionIntervalCount / ( motionTimeEnd - motionTimeBegin );
            local_time = local_time < 0.f ? 0.f : ( local_time > motionIntervalCount ? motionIntervalCount : local_time );
            keyIdx     = (int)local_time;
            local_time -= (float)keyIdx;
        }

        unsigned int firstVertexIdx = ( primIdx + index_offset_ptr[inputIdx] ) * motionStepCount;

        const float4* q = &vertices[firstVertexIdx + keyIdx];

        v.x = q[0].x + local_time * ( q[1].x - q[0].x );
        v.y = q[0].y + local_time * ( q[1].y - q[0].y );
        v.z = q[0].z + local_time * ( q[1].z - q[0].z );
        v.w = q[0].w + local_time * ( q[1].w - q[0].w );
    }
    else
    {
        unsigned int firstVertexIdx = primIdx + index_offset_ptr[inputIdx];

        v = vertices[firstVertexIdx];
    }

    return v;
}

/* 
     We could use bitmagic to colwert hitkind to primitive types like :

        unsigned int primitiveType;

        if( hitkind < 0x80 )
            primitiveType = OPTIX_PRIMITIVE_TYPE_LWSTOM;
        else
            primitiveType = ((hitkind - 0x80) >> 1) + OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;

        bool isSupported = ( ( 1u << ( primitiveType - OPTIX_PRIMITIVE_TYPE_LWSTOM ) ) & supportedFlags ) != 0;

    We expect switching afterwards on the builtin type as in:

    extern "C" __global__ void __closesthit__radiance()
    {
        const OptixPrimitiveType primType = optixGetPrimitiveType( optixGetHitKind() );

        if( primType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE )
            optixSetPayload_0( 0xDEAD );
        else if( primType == OPTIX_PRIMITIVE_TYPE_ROUND_LWBIC_BSPLINE )
            optixSetPayload_0( 0xBEAF );
    }

    Unfortunately llvm and ptxas are not very good at reasoning through the bit magic and produce suboptimal sass.
*/

extern "C" __forceinline__ uint32_t RUNTIME_BUILTIN_TYPE_FROM_HIT_KIND( unsigned int hitkind )
{
    // ignore face bit
    unsigned int signedHitKind = ( hitkind & ( ~OPTIX_HIT_KIND_BACKFACE_MASK ) );
    // explicitly check supported types
    switch( signedHitKind )
    {
        case OPTIX_HIT_KIND_LWRVES_LINEAR_HIT:
            return OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
        case OPTIX_HIT_KIND_LWRVES_QUADRATIC_BSPLINE_HIT:
            return OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
        case OPTIX_HIT_KIND_LWRVES_LWBIC_BSPLINE_HIT:
            return OPTIX_PRIMITIVE_TYPE_ROUND_LWBIC_BSPLINE;
        case OPTIX_HIT_KIND_LWRVES_CATMULLROM_HIT:
            return OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM;
        case OPTIX_HIT_KIND_SPHERE:
            return OPTIX_PRIMITIVE_TYPE_SPHERE;
        case OPTIX_HIT_KIND_TRIANGLE:
            return OPTIX_PRIMITIVE_TYPE_TRIANGLE;
    };

    // fall through
    return OPTIX_PRIMITIVE_TYPE_LWSTOM;
}

extern "C" __forceinline__ bool RUNTIME_IS_BUILTIN_TYPE_SUPPORTED( unsigned int builtinType, unsigned int supportedFlags )
{
    // explicitly check supported types
    switch( builtinType )
    {
        case OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR:
            return ( supportedFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR ) != 0;
        case OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE:
            return ( supportedFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE ) != 0;
        case OPTIX_PRIMITIVE_TYPE_ROUND_LWBIC_BSPLINE:
            return ( supportedFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LWBIC_BSPLINE ) != 0;
        case OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM:
            return ( supportedFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM ) != 0;
        case OPTIX_PRIMITIVE_TYPE_SPHERE:
            return ( supportedFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE ) != 0;
        case OPTIX_PRIMITIVE_TYPE_TRIANGLE:
            return ( supportedFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE ) != 0;
    };

    // fall through
    return ( supportedFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_LWSTOM ) != 0;
}

extern "C" __forceinline__ bool RUNTIME_IS_EXCEPTION_LINE_INFO_AVAILABLE( int exceptionCode )
{
    switch( exceptionCode )
    {
        case OPTIX_EXCEPTION_CODE_STACK_OVERFLOW:
        case OPTIX_EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED:
        case OPTIX_EXCEPTION_CODE_TRAVERSAL_DEPTH_EXCEEDED:
        case OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_TRAVERSABLE:
        case OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_MISS_SBT:
        case OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_HIT_SBT:
        case OPTIX_EXCEPTION_CODE_UNSUPPORTED_DATA_ACCESS:
        case OPTIX_EXCEPTION_CODE_PAYLOAD_TYPE_MISMATCH:
            return false;
        case OPTIX_EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH:
        case OPTIX_EXCEPTION_CODE_UNSUPPORTED_PRIMITIVE_TYPE:
        case OPTIX_EXCEPTION_CODE_ILWALID_RAY:
            return true;
    }
    if( exceptionCode >= 0 )
    {
        // user exception
        return true;
    }
    return false;
}
