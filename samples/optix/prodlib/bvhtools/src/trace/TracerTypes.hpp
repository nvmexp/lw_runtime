/*
 *  Copyright (c) 2012, LWPU Corporation
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

#ifndef __TRACERTYPES_HPP__
#define __TRACERTYPES_HPP__

#ifdef __cplusplus
namespace prodlib
{
namespace bvhtools
{
#endif

//------------------------------------------------------------------------
// Shared definitions for TracerLwda and TracerISPC.
//------------------------------------------------------------------------

enum RayFormat
{
  RAY_ORIGIN_DIRECTION           = 0, // float3:origin float3:direction
  RAY_ORIGIN_TMIN_DIRECTION_TMAX = 1, // float3:origin, float:tmin, float3:direction, float:tmax
  RAY_ORIGIN_MASK_DIRECTION_TMAX = 2  // float3:origin, int:mask, float3:direction, float:tmax
};

//------------------------------------------------------------------------

enum HitFormat
{  
  HIT_BITMASK                    = 0, // one bit per ray 0=miss, 1=hit
  HIT_T                          = 1, // float:ray distance (t < 0 for miss)
  HIT_T_TRIID                    = 2, // float:ray distance, int:triangle id
  HIT_T_TRIID_U_V                = 3, // float:ray distance, int:triangle id, float2:barycentric coordinates u,v (w=1-u-v)
  HIT_T_TRIID_INSTID             = 4, // float:ray distance (t < 0 for miss), int:triangle id, int:instance id*/
  HIT_T_TRIID_INSTID_U_V         = 5  // float:ray distance (t < 0 for miss), int:triangle id, int:instance id, float2:barycentric coordinates u,v (w=1-u-v) */
};

//------------------------------------------------------------------------

enum IntersectionType
{
  INTERSECT_WOOP            = 0,
  INTERSECT_INDEXED_LIST    = 1,
  INTERSECT_VERTEX_LIST     = 2
};

//------------------------------------------------------------------------

#ifdef __cplusplus
} // namespace bvhtools
} // namespace prodlib
#endif

#endif // __TRACERTYPES_HPP__
