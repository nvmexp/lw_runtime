// Copyright LWPU Corporation 2015
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
#include <prodlib/bvhtools/src/common/TypesInternal.hpp>
#include <prodlib/bvhtools/src/common/Intrinsics.hpp>

#define APM_MAX_RESOLUTION  8

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// Apex point map (APM) for an entire model, generalizing the traditional
// model-space AABB. Output of ApexPointMapConstructor, input of various
// other building blocks, especially the instance node generator.
//
// The m_dots array stores precallwlated min/max extents of the input
// primitives against a predetermined set of directions. The slab lookup
// function getSlab() returns a conservative estimate of
// min(dot(dir, input)) and max(dot(dir, input)) for an arbitrary
// direction vector, where the min/max are taken over all input
// primitives. The lookup takes a constant time regardless of the number
// of input primitives or the APM resolution.
//
// Resolution of the APM determines how accurate the queried bounds are
// for query directions other than the object-space coordinate axes.
// Resolution of 1 corresponds to storing only the object-space AABB, and
// the resulting bounding planes are the same as would be obtained from
// its corners.
//
// The first three float2s in m_dots are always min/max x, y, z extents of
// the input primitives in object space, thus corresponding to the model-
// space AABB.
//
// This is an optimized variant of the method described in paper "Apex
// Point Maps for Constant-Time Bounding Plane Approximation" published
// at Proceedings of EGSR 2015. The main differences to the paper are that
// the lookup is based on octahedral mapping instead of lwbical mapping,
// and the stored data contains precallwlated dot products instead of 3D
// apex points.

struct ApexPointMap
{
                    int             m_resolution;
                    uint2           m_dots[3];          // Note: The actual size varies depending on resolution; see getNumDirections().

    static INLINE   int             getNumDirections    (int resolution)    { return 2 * resolution * resolution + 1; }
    static INLINE   size_t          getNumBytes         (int resolution)    { return sizeof(ApexPointMap) + (getNumDirections(resolution) - 3) * sizeof(uint2); }

    INLINE          AABB            getAABB             (void) const;
    INLINE          float2          getSlab             (float3 dir) const; // Implementation in "ApexPointMapLookup.hpp".

    // Helpers.

    static INLINE   unsigned int    encodeFloat         (float val)         { unsigned int bits = __float_as_int(val); bits ^= ((int)bits >> 31) | 0x80000000u; return bits; }
    static INLINE   float           decodeFloat         (unsigned int bits) { bits ^= ~((int)bits >> 31) | 0x80000000u; return __int_as_float(bits); }
    static INLINE   float2          decodeDots          (uint2 in, int flip);
    static INLINE   float3          decodeDirection     (unsigned int in, float rcpResolution);
};

//------------------------------------------------------------------------

INLINE AABB ApexPointMap::getAABB(void) const
{
    ApexPointMap apm = loadCachedAlign16(this);
    return AABB(
        make_float3(decodeFloat(apm.m_dots[0].x), decodeFloat(apm.m_dots[1].x), decodeFloat(apm.m_dots[2].x)),
        make_float3(decodeFloat(apm.m_dots[0].y), decodeFloat(apm.m_dots[1].y), decodeFloat(apm.m_dots[2].y)));
}

//------------------------------------------------------------------------

INLINE float2 ApexPointMap::decodeDots(uint2 in, int flip)
{
    unsigned int m  = (unsigned int)(flip >> 31);       // SAR
    unsigned int ia = (in.x & ~m) | (~in.y & m);        // LOP3
    unsigned int ib = (in.y & ~m) | (~in.x & m);        // LOP3
    float        a  = ApexPointMap::decodeFloat(ia);    // SAR + LOP3
    float        b  = ApexPointMap::decodeFloat(ib);    // SAR + LOP3

    return make_float2(a, b);
}

//------------------------------------------------------------------------

INLINE float3 ApexPointMap::decodeDirection(unsigned int in, float rcpResolution)
{
    float3 dir;
    dir.x = (float)(((int)in) << 16 >> 16) * rcpResolution;
    dir.y = (float)(((int)in) >> 16) * rcpResolution;
    dir.z = 1.f - fabsf(dir.x) - fabsf(dir.y);
    return dir;
}

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
