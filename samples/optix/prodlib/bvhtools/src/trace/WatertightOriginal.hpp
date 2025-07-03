/*
 *  Copyright (c) 2016, LWPU Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of LWPU Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __WATERTIGHTORIGINAL_HPP__
#define __WATERTIGHTORIGINAL_HPP__

#include <prodlib/bvhtools/src/common/Intrinsics.hpp>

namespace WatertightOriginal
{
//------------------------------------------------------------------------
// Adapted from original LWCA version by Tero Karras:
// https://wiki.lwpu.com/lwresearch/index.php/Ray_Tracing_100x/Watertight_Ray_Triangle_Intersection

static inline float3 swizzle(const float3& v, const int axis)
{
  if (axis == 0) return make_float3(v.y, v.z, v.x);
  if (axis == 1) return make_float3(v.z, v.x, v.y);
  return v;
}

static inline void setupRay(const float3& orig, const float3& dir, int& axis, float3& translate, float3& shear)
{
    // Choose main axis.

    const float ax = fabsf(dir.x);
    const float ay = fabsf(dir.y);
    const float az = fabsf(dir.z);
    axis = (ax >= ay && ax >= az) ? 0 : (ay >= az) ? 1 : 2;

    // Swizzle origin and direction.

    translate = swizzle(orig, axis);
    shear = swizzle(dir, axis);

    // Output ray.

    shear.z  = 1.0f / shear.z;
    shear.x *= -shear.z;
    shear.y *= -shear.z;
}

static inline bool intersectTriangle(
    const int axis, const float3& translate, const float3& shear,
    const float3& v0, const float3& v1, const float3& v2, const float tmin, 
    const float tmax, float * const out_t, float * const out_u, float * const out_v)
{
  // Swizzle the components of triangle vertices.

  float ax = (axis == 0) ? v0.y : (axis == 1) ? v0.z : v0.x;
  float ay = (axis == 0) ? v0.z : (axis == 1) ? v0.x : v0.y;
  float az = (axis == 0) ? v0.x : (axis == 1) ? v0.y : v0.z;
  float bx = (axis == 0) ? v1.y : (axis == 1) ? v1.z : v1.x;
  float by = (axis == 0) ? v1.z : (axis == 1) ? v1.x : v1.y;
  float bz = (axis == 0) ? v1.x : (axis == 1) ? v1.y : v1.z;
  float cx = (axis == 0) ? v2.y : (axis == 1) ? v2.z : v2.x;
  float cy = (axis == 0) ? v2.z : (axis == 1) ? v2.x : v2.y;
  float cz = (axis == 0) ? v2.x : (axis == 1) ? v2.y : v2.z;

  // Translate, shear, and scale the vertices.

  az -= translate.z;
  bz -= translate.z;
  cz -= translate.z;
  ax += shear.x*az - translate.x;
  ay += shear.y*az - translate.y;
  bx += shear.x*bz - translate.x;
  by += shear.y*bz - translate.y;
  cx += shear.x*cz - translate.x;
  cy += shear.y*cz - translate.y;

  // Compute scaled barycentric coordinates.
  // Note: No FMA allowed here!

  float u = cx*by - cy*bx;
  float v = ax*cy - ay*cx;
  float w = bx*ay - by*ax;

  // Uncertain result => fall back to double precision.

  if ( (u == 0.0f) || (v == 0.0f) || (w == 0.0f) )
  {
    u = (float)((double)cx * (double)by - (double)cy * (double)bx);
    v = (float)((double)ax * (double)cy - (double)ay * (double)cx);
    w = (float)((double)bx * (double)ay - (double)by * (double)ax);
  }

  // Outside the triangle => miss.

  const float denom = u + v + w;
  if ((( (u < 0.0f) || (v < 0.0f) || (w < 0.0f) ) && 
       ( (u > 0.0f) || (v > 0.0f) || (w > 0.0f) )) || (denom == 0.0f) )
  {
    return false;
  }

  // Compute intersection t-value.

  const float rcpDet = 1.0f / denom;
  const float num = u*az + v*bz + w*cz;
  const float t = num*(shear.z*rcpDet);

  // Outside the current t-range => miss.

  if ( (t <= tmin) || (t >= tmax) ) {
    return false;
  }

  // Otherwise => hit.
  *out_t = t;
  *out_u = u*rcpDet;
  *out_v = v*rcpDet;
  return true;
}

//------------------------------------------------------------------------
} // namespace WatertightOriginal
#endif // __WATERTIGHTORIGINAL_HPP__
