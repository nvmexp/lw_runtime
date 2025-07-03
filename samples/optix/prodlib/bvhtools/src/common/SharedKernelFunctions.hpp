
// Copyright LWPU Corporation 2014
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
#include "TypesInternal.hpp"
#include <stdio.h>

namespace prodlib
{
namespace bvhtools
{
  //------------------------------------------------------------------------

  // Implementation of delta(i, j), i.e. length of the longest common prefix between two Morton codes.
#ifdef __LWDACC__
  static INLINE int longestCommonPrefix(int i, int j, unsigned long long ki, unsigned long long kj)
  {
    int a = findLeadingOne(__double2hiint(__longlong_as_double(ki)) ^ __double2hiint(__longlong_as_double(kj)));
    int b = findLeadingOne(__double2loint(__longlong_as_double(ki)) ^ __double2loint(__longlong_as_double(kj)));
    int c = findLeadingOne(i ^ j);
    return slct(31 - a, slct(63 - b, 95 - c, b), a);
  }
#else
  static inline int longestCommonPrefix(int i, int j, unsigned long long ki, unsigned long long kj)
  {
    unsigned long long diff = ki ^ kj;
    unsigned int a = (unsigned int)(diff >> 32);
    if (a != 0)
      return 31 - findLeadingOne(a);

    unsigned int b = (unsigned int)diff;
    if (b != 0)
      return 63 - findLeadingOne(b);

    return 95 - findLeadingOne(i ^ j);
  }
#endif

  //------------------------------------------------------------------------
  // BL: This needs a comment
  // s - split position
  // d - direction
  // j - other end of the interval (may be less than i)
  static INLINE void computeInterval( const int& i, const int& numPrims, const unsigned long long* mortonCodes, int& s, int& d, int& j )
  {
    // Choose direction.

    unsigned long long code = mortonCodes[i];
    int prefix_prev = (i == 0) ? -1 : longestCommonPrefix(i, i - 1, code, mortonCodes[i - 1]);
    int prefix_next = longestCommonPrefix(i, i + 1, code, mortonCodes[i + 1]);

    d = (prefix_next > prefix_prev) ? 1 : -1;
    int prefix_min = min(prefix_prev, prefix_next);

    // Find upper bound for length.

    int lmax = 128 >> 2;
    unsigned int probe;
    do
    {
      lmax <<= 2;
      probe = i + lmax * d;
    }
    while (probe < (unsigned int)numPrims && longestCommonPrefix(i, probe, code, mortonCodes[probe]) > prefix_min);

    // Determine length.

    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1)
    {
      probe = i + (l + t) * d;
      //if (probe < (unsigned int)numPrims & longestCommonPrefix(i, probe, code, mortonCodeTex(probe)) > prefix_min)
      if (probe < (unsigned int)numPrims && longestCommonPrefix(i, probe, code, mortonCodes[probe]) > prefix_min)
        l += t;
    }
    j = i + l * d;
    int prefix_node = longestCommonPrefix(i, j, code, mortonCodes[j]);

    // Find split position.

    s = 0;
    int t = l;
    do
    {
      t = (t + 1) >> 1;
      probe = i + (s + t) * d;
      //if (probe < (unsigned int)numPrims & longestCommonPrefix(i, probe, code, mortonCodeTex(probe)) > prefix_node)
      if (probe < (unsigned int)numPrims && longestCommonPrefix(i, probe, code, mortonCodes[probe]) > prefix_node)
        s += t;
    }
    while (t > 1);
  }

  //------------------------------------------------------------------------

  static INLINE int computeNodeDepth( const int* const __restrict nodeParents, int idx )
  {
    //
    // Walk to the root and compute depth L
    //

    int depth = -1;
    do
    {
        depth++;
        if (idx < 0)
            idx = ~idx;
        idx = LDG_OR_GLOBAL( &nodeParents[idx] );
    }
    while (idx != INT_MAX);

    return depth;
  }

  //------------------------------------------------------------------------

  static INLINE float det4x3_affine(const float* const __restrict m ) // assume a last row with 0,0,0,1
  {
    float d = m[0] * m[5] * m[10]- 
              m[0] * m[9] * m[ 6]- 
              m[4] * m[1] * m[10]+ 
              m[4] * m[9] * m[ 2]+ 
              m[8] * m[1] * m[ 6]- 
              m[8] * m[5] * m[ 2];

    return d;
  }

  //------------------------------------------------------------------------
  // Assume a last row with 0,0,0,1
  // BL: This can probably be heavily optimized using the properties of affine
  // matrices
  static INLINE void computeIlwerse4x3_affine( float* const outIlwMatrice, const float* const __restrict m )
  {
    const float d = 1.0f / det4x3_affine( m );

    // let the compiler optimize this!
    outIlwMatrice[ 0] = d * (m[ 5] * (m[10] * 1.0f  - 0.0f  * m[11]) + m[ 9] * (0.0f  * m[ 7] - m[ 6] * 1.0f)  + 0.0f  * (m[ 6] * m[11] - m[10] * m[ 7]));
    outIlwMatrice[ 4] = d * (m[ 6] * (m[ 8] * 1.0f  - 0.0f  * m[11]) + m[10] * (0.0f  * m[ 7] - m[ 4] * 1.0f)  + 0.0f  * (m[ 4] * m[11] - m[ 8] * m[ 7]));
    outIlwMatrice[ 8] = d * (m[ 7] * (m[ 8] * 0.0f  - 0.0f  * m[ 9]) + m[11] * (0.0f  * m[ 5] - m[ 4] * 0.0f)  + 1.0f  * (m[ 4] * m[ 9] - m[ 8] * m[ 5]));
    outIlwMatrice[ 1] = d * (m[ 9] * (m[ 2] * 1.0f  - 0.0f  * m[ 3]) + 0.0f  * (m[10] * m[ 3] - m[ 2] * m[11]) + m[ 1] * (0.0f  * m[11] - m[10] *  1.0f));
    outIlwMatrice[ 5] = d * (m[10] * (m[ 0] * 1.0f  - 0.0f  * m[ 3]) + 0.0f  * (m[ 8] * m[ 3] - m[ 0] * m[11]) + m[ 2] * (0.0f  * m[11] - m[ 8] *  1.0f));
    outIlwMatrice[ 9] = d * (m[11] * (m[ 0] * 0.0f  - 0.0f  * m[ 1]) + 1.0f  * (m[ 8] * m[ 1] - m[ 0] * m[ 9]) + m[ 3] * (0.0f  * m[ 9] - m[ 8] *  0.0f));
    outIlwMatrice[ 2] = d * (0.0f  * (m[ 2] * m[ 7] - m[ 6] * m[ 3]) + m[ 1] * (m[ 6] * 1.0f  - 0.0f  * m[ 7]) + m[ 5] * (0.0f  * m[ 3] - m[ 2] *  1.0f));
    outIlwMatrice[ 6] = d * (0.0f  * (m[ 0] * m[ 7] - m[ 4] * m[ 3]) + m[ 2] * (m[ 4] * 1.0f  - 0.0f  * m[ 7]) + m[ 6] * (0.0f  * m[ 3] - m[ 0] *  1.0f));
    outIlwMatrice[10] = d * (1.0f  * (m[ 0] * m[ 5] - m[ 4] * m[ 1]) + m[ 3] * (m[ 4] * 0.0f  - 0.0f  * m[ 5]) + m[ 7] * (0.0f  * m[ 1] - m[ 0] *  0.0f));
    outIlwMatrice[ 3] = d * (m[ 1] * (m[10] * m[ 7] - m[ 6] * m[11]) + m[ 5] * (m[ 2] * m[11] - m[10] * m[ 3]) + m[ 9] * (m[ 6] * m[ 3] - m[ 2] * m[ 7]));
    outIlwMatrice[ 7] = d * (m[ 2] * (m[ 8] * m[ 7] - m[ 4] * m[11]) + m[ 6] * (m[ 0] * m[11] - m[ 8] * m[ 3]) + m[10] * (m[ 4] * m[ 3] - m[ 0] * m[ 7]));
    outIlwMatrice[11] = d * (m[ 3] * (m[ 8] * m[ 5] - m[ 4] * m[ 9]) + m[ 7] * (m[ 0] * m[ 9] - m[ 8] * m[ 1]) + m[11] * (m[ 4] * m[ 1] - m[ 0] * m[ 5]));
  }

  //------------------------------------------------------------------------
  
  static INLINE void transformRay( float3& orig, float3& dir, const float4* const __restrict im)
  {
    float3 o, d;
  
    const float4 m0 = im[0];
    o.x = m0.x * orig.x + m0.y * orig.y + m0.z * orig.z + m0.w;
    d.x = m0.x * dir.x  + m0.y * dir.y  + m0.z * dir.z;
  
    const float4 m1 = im[1];
    o.y = m1.x * orig.x + m1.y * orig.y + m1.z * orig.z + m1.w;
    d.y = m1.x * dir.x  + m1.y * dir.y  + m1.z * dir.z;    
  
    const float4 m2 = im[2];
    o.z = m2.x * orig.x + m2.y * orig.y + m2.z * orig.z + m2.w;
    d.z = m2.x * dir.x  + m2.y * dir.y  + m2.z * dir.z;
  
    orig = o;
    dir  = d;
  }
  
  //------------------------------------------------------------------------
  
  static INLINE float4* getTransformPtr( const float* const __restrict basePtr, const int which, const int stride )
  {
    return (float4*)(((char*)basePtr) + which * stride);
  }

  //------------------------------------------------------------------------------

  template <class T> static INLINE T* make_ptr(void* const __restrict base, const size_t offset) { return (T*)((char*)base + offset); }
  template <class T> static INLINE const T* make_ptr(const void* const __restrict base, const size_t offset) { return (T*)((const char*)base + offset); }

  //----------------------------------------------------------------------------

  static INLINE void print(const BvhNode& n, int idx = -1)
  {
    printf(
      "%4d: [%8.2g,%8.2g,%8.2g][%8.2g,%8.2g,%8.2g] %5d %5d || "
           "[%8.2g,%8.2g,%8.2g][%8.2g,%8.2g,%8.2g] %5d %5d\n",
      idx, n.c0lox, n.c0loy, n.c0loz, n.c0hix, n.c0hiy, n.c0hiz, n.c0idx, n.c0num,
      n.c1lox, n.c1loy, n.c1loz, n.c1hix, n.c1hiy, n.c1hiz, n.c1idx, n.c1num
    );
  }

  //----------------------------------------------------------------------------

  static INLINE void print(const PrimitiveAABB& bb, const int idx = -1)
  {
    printf("%4d: [%8.2g,%8.2g,%8.2g][%8.2g,%8.2g,%8.2g] %5d %5d\n",
      idx, bb.lox, bb.loy, bb.loz, bb.hix, bb.hiy, bb.hiz, bb.primitiveIdx, bb.pad);
  }

  //------------------------------------------------------------------------------

  static INLINE void print(const BvhHeader* header)
  {
    printf(
      "BvhHeader            : %p\n"
      "  size               : %lu\n"
      "  instanceDataOffset : %lu\n"
      "  flags              : %u\n"
      "  numEntities        : %u\n"
      "  nodesOffset        : %lu\n"
      "  remapOffset        : %lu\n"
      "  apmOffset          : %lu\n"
      "  trianglesOffset    : %lu\n"
      "  numNodes           : %u\n"
      "  numRemaps          : %u\n"
      "  numTriangles       : %u\n"
      "  numInstances       : %u\n",
      header,
      (unsigned long)header->size,
      (unsigned long)header->instanceDataOffset,
      header->flags,
      header->numEntities,
      (unsigned long)header->nodesOffset,
      (unsigned long)header->remapOffset,
      (unsigned long)header->apmOffset,
      (unsigned long)header->trianglesOffset,
      header->numNodes,
      header->numRemaps,
      header->numTriangles,
      header->numInstances
    );
  }

  //------------------------------------------------------------------------------

  static INLINE void print(const InstanceDesc& inst)
  {
    const float* t = inst.transform;
    printf(
      "InstanceDesc\n"
      "  transform         : %8.2g %8.2g %8.2g %8.2g\n"
      "                      %8.2g %8.2g %8.2g %8.2g\n"
      "                      %8.2g %8.2g %8.2g %8.2g\n"
      "  instanceId        : %u\n"
      "  flags             : %08x\n"
      "  bvh               : %p\n",
      t[0], t[1], t[2], t[3],
      t[4], t[5], t[6], t[7],
      t[8], t[9], t[10], t[11],
      inst.instanceId,
      inst.instanceOffsetAndFlags >> 24,
      inst.bvh
    );
  }

  //------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
