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

#ifndef __TRACERISPC_HPP__
#define __TRACERISPC_HPP__

#include "Tracer.hpp"

#include <vector>

//------------------------------------------------------------------------

struct TracerParamsMesh
{
  const float4*   nodes;
  const float4*   triangles;
  const int*     indices;
  const float*    vertices;
  const int*      remap;
  int             indexStride;
  int             vertexStride;
  int             rootNode;
  int             numEntities;
};

//------------------------------------------------------------------------

struct TracerParamsGroup
{
  const float4*   nodes;
  const int*      remap;
  const float*    ilwMatrices;
  const int*      modelId;
  int             rootNode;
  int             numEntities;
  int             matrixStride;
};

//------------------------------------------------------------------------

// WARNING: this all needs to be compatible with the KernelParameters struct
struct tracerIspcParams
{
  const void*                  rays;
  void*                        hits;
  TracerParamsGroup*           group;
  TracerParamsMesh*            meshes;
  int                          numRays;
  int                          rayFormat;
  int                          hitFormat;
  int                          anyhit;
  int                          watertight;
  int                          isecMode;
  int                          useMasking;
  int                          endMarker;          // Use &endMarker instead of sizeof() to avoid uploading too much data.
};

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

class TracerISPC : public Tracer
{
public:
                        TracerISPC          (void);
    virtual             ~TracerISPC         (void);

    virtual const char* getBackendName      (void) const    { return m_backendName; }
    virtual Layout      getBVHLayout        (void) const    { return m_bvhLayout; }    

    virtual void        traceFromHostMem(const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyHit, const bool watertight, const void* const stream = 0);

private:
                        TracerISPC          (const TracerISPC&); // forbidden
    TracerISPC&         operator=           (const TracerISPC&); // forbidden

    void launch1LevelBvh(const TracerDataMesh& tdm, const void* rays, int rayFormat, void* hits, int hitFormat, int numRays, bool anyHit, bool watertight);
    void launch2LevelBvh(const TracerDataGroup& tdg, const void* rays, int rayFormat, void* hits, int hitFormat, int numRays, bool anyHit, bool watertight);

private:
    const char*         m_backendName;
    Layout              m_bvhLayout;

    TracerParamsGroup             m_group;
    std::vector<TracerParamsMesh> m_meshes;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
#endif // __TRACERISPC_HPP__
