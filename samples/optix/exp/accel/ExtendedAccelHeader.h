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

#define EXTENDED_ACCEL_HEADER_SIZE 128

namespace optix_exp {

// proxy_size_t will end up being the same thing as size_t, but without including a header that clang can't find.
using proxy_size_t = decltype( sizeof( char ) );

struct ExtendedAccelHeader
{
    proxy_size_t dataOffset;           // offset in bytes to vertex data, relative to traversable handle
    proxy_size_t dataCompactedOffset;  // offset in bytes to vertex data after accel compaction, relative to traversable handle

    // ----------------------- following data is overwritten by copyExtendedAccelHeader ----------------------- //

    proxy_size_t dataSize;  // total size of lwrve/sphere data in bytes

    // auxiliary lwrve data
    unsigned int normalOffset;    // offset in 16 bytes to normal data, relative to data, 0 if no normals
    unsigned int numBuildInputs;  // number of build inputs

    // intersection lwrve data (16 byte aligned for efficient loading in the lwrve/sphere intersectors)
    unsigned int dataOffset32;         // offset in vertices(16 byte) to vertex data, relative to traversable handle
    unsigned int indexOffset;          // offset in 16 bytes to lwrve indices, relative to data

    unsigned int sbtMappingOffset;     // offset in 16 bytes to the mapping of sbt indices to build input indices

    unsigned int primIndexOffset;      // offset in 16 bytes to the list of primitive indices offsets, relative to data

    OptixPrimitiveType primitiveType;  // the primitive type of this builtin primitive

    bool lowMem;                       // indicating low memory version without storage of LwrveSegmentData/SphereIntersectorData in the primLwstomVABuffer,
                                       // used for exception tests in builtin intersectors which check the compatibility of build type and intersector
};

// Representation of sub-primitive (segment) in intersector.
struct LwrveSegmentData
{
    // switches between the uniform and fixed point range encoding.
    unsigned long long uniform  : 1;
    unsigned long long reserved : 2;

    // relative address to first lwrve vertex. bits [3,39]
    // this needs to start at bit 3 for make sure the segment data is a decorated pointer.
    unsigned long long vertexOffset : 37;

    // number of motion keys.
    unsigned long long numKeysMinusOne : 8;

    // lwrve range encoding.
    //   uniform     [u0/(un+1),(u0+1)/(un+1))]
    //   fixed point [u0/256,(u0+un+1)/256]
    unsigned long long u0 : 8;
    unsigned long long un : 8;

#if defined(__LWDACC__)
    __device__ __forceinline__ unsigned long long getVertexOffsetInBytes() const
    {
        return vertexOffset << 3;
    }

    __device__ __forceinline__ void encodeUniformSegmentData( unsigned int numSplits, unsigned int intervalId, int c0_idx, int numKeys )
    {
        uniform = 1;
        numKeysMinusOne = (unsigned char)(numKeys - 1);
        // colwert to [u0/(un+1),(u0+1)/(un+1))] encoding
        u0 = (unsigned char)intervalId;
        un = (unsigned char)(numSplits - 1);
        vertexOffset = (c0_idx * numKeys * sizeof(float4)) >> 3;
    }

    __device__ __forceinline__ void encodeSegmentData( float2 range, int c0_idx, int numKeys )
    {
        int ix = (int)ceilf(range.x * 256.f);
        int iy = (int)ceilf(range.y * 256.f);

        uniform = 0;
        numKeysMinusOne = (unsigned char)(numKeys - 1);
        // colwert to [u0/256,(u0+un+1)/256] encoding
        u0 = (unsigned char)ix;
        un = (unsigned char)(iy - ix - 1);
        vertexOffset = (c0_idx * numKeys * sizeof(float4)) >> 3;
    }

    __device__ __forceinline__ float2 decodeSegmentRange() const
    {
        float  fu0, fu1;
        if( uniform )
        {
            const float rcp = 1.f / (float)(un + 1);
            fu0 = (u0 * rcp);
            fu1 = ((u0 + 1) * rcp);
        }
        else
        {
            fu0 = (u0 / 256.f);
            fu1 = (((int)(u0 + un) + 1) / 256.f);
        }
        return make_float2(fu0, fu1);
    }
#endif
};

// Representation of sphere in intersector.
struct SphereIntersectorData
{
    unsigned long long reserved_0 : 3;

    // relative address to first sphere vertex. bits [3,39]
    // this needs to start at bit 3 for make sure the sphere data is a decorated pointer.
    unsigned long long vertexOffset : 37;

    // bits [40,63] unused
    unsigned long long reserved_1 : 24;

#if defined( __LWDACC__ )
    __device__ __forceinline__ unsigned long long getVertexOffsetInBytes() const { return vertexOffset << 3; }

    __device__ __forceinline__ void encodeSphereData( int c0_idx, int numKeys )
    {
        vertexOffset = ( c0_idx * numKeys * sizeof( float4 ) ) >> 3;
    }
#endif
};

// single bit to encode backface or frontface hit
#define OPTIX_HIT_KIND_BACKFACE_MASK 0x1


#define OPTIX_HIT_KIND_TRIANGLE 0xfe  // (backface 0xff)

// hit kinds are signed 8bit. All hitkinds <= OPTIX_HIT_KIND_LWRVES are lwrves
#define OPTIX_HIT_KIND_LWRVES -121                        //  0x87
#define OPTIX_HIT_KIND_LWRVES_QUADRATIC_BSPLINE_HIT 0x80  // (backface 0x81)
#define OPTIX_HIT_KIND_LWRVES_LWBIC_BSPLINE_HIT 0x82      // (backface 0x83)
#define OPTIX_HIT_KIND_LWRVES_LINEAR_HIT 0x84             // (backface 0x85)
#define OPTIX_HIT_KIND_LWRVES_CATMULLROM_HIT 0x86         // (backface 0x87)
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_RIBBONS
#define OPTIX_HIT_KIND_RIBBONS 0x88                       // (backface 0x89)
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_RIBBONS
#define OPTIX_HIT_KIND_SPHERE 0x8A                        // (backface 0x8B)

}  // namespace optix_exp
