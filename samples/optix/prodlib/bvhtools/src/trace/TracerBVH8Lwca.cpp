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

#include "TracerBVH8Lwda.hpp"
#include <prodlib/bvhtools/src/common/Utils.hpp>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/Assert.h>
#include <string.h>
#include <vector>

using namespace prodlib::bvhtools;
using namespace prodlib;

//------------------------------------------------------------------------

TracerBVH8Lwda::TracerBVH8Lwda(void)
:   m_backendName   (NULL),
    m_launchFunc    (NULL),
    m_warpCounters  (0),
    m_meshData_d    (NULL),
    m_streamCount   (0),
    m_numMeshes     (1)
{
    m_config.useInstancing   = false;
    m_config.useExternalTris = false;
    m_config.useWatertight   = false;
    m_config.anyhit          = false;
}

//------------------------------------------------------------------------

TracerBVH8Lwda::~TracerBVH8Lwda(void)
{
    m_lwda.deviceFree(m_warpCounters);
    m_lwda.deviceFree(m_meshData_d);
}

//------------------------------------------------------------------------

void TracerBVH8Lwda::init(int streamCount)
{
    // Choose implementation.

    if (m_lwda.getSMArch() >= 30)
    {
        m_backendName = "GPU-accelerated BVH8 (optimized for sm_52)";
        m_launchFunc = launchBVH8Trace52;
    }
    else
    {
        throw IlwalidValue(RT_EXCEPTION_INFO, "Unsupported SM version!");
    }

    // Allocate mesh data.

    m_lwda.deviceFree(m_meshData_d);
    m_meshData_d = (TracerParamsMesh*)m_lwda.deviceAlloc(m_numMeshes * sizeof(TracerParamsMesh));
    m_meshData_h.resize(m_numMeshes);

    // Allocate warp counters.

    m_lwda.deviceFree(m_warpCounters);
    m_warpCounters = (int*)m_lwda.deviceAlloc(streamCount*sizeof(int));
    m_streamCount = streamCount;      
}

//------------------------------------------------------------------------

Tracer::Layout TracerBVH8Lwda::getBVHLayout(void) const // obsolete boilerplate
{
    Layout layout;
    layout.storeOnGPU       = true;
    layout.arrayAlign       = ((LwdaUtils&)m_lwda).getTextureAlign();
    layout.reorderTriangles = true;
    layout.optixNodes       = false;
    return layout;
}

//------------------------------------------------------------------------

void TracerBVH8Lwda::traceFromDeviceMem(const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyhit, const bool watertight, const void* const stream, const int streamIdx)
{
    if (!bvh.inDeviceMem)
        throw IlwalidValue(RT_EXCEPTION_INFO, "TracerBVH8Lwda requires BVH stored in device memory");

    configure(bvh, rayFormat, watertight, anyhit);
    launch(bvh, rays, rayFormat, hits, hitFormat, numRays, stream, streamIdx);
}

//------------------------------------------------------------------------

void TracerBVH8Lwda::configure(const TracerData& data, int rayFormat, bool watertight, bool anyhit)
{
    const TracerDataGroupBVH8* tdg = dynamic_cast<const TracerDataGroupBVH8*>(&data);
    int numMeshes = (tdg) ? (int)tdg->meshes.size() : 1;

    const TracerDataMeshBVH8* tdm = NULL;
    if (!tdg)
    {
        tdm = dynamic_cast<const TracerDataMeshBVH8*>(&data);
        RT_ASSERT(tdm);
    }
    else if (numMeshes)
    {
        tdm = &tdg->bvh8Meshes[0];
        for (int i = 1; i < numMeshes && !tdg->bvh8Meshes[i].numBVH8Triangles; i++)
            tdm = &tdg->bvh8Meshes[i]; // prefer a mesh that actually has some triangles
    }

    TracerBVH8LwdaConfig c;
    c.useInstancing   = (tdg != NULL);
    c.useExternalTris = (tdm && tdm->numBVH8Triangles == 0);
    c.useWatertight   = watertight;
    c.anyhit          = anyhit;

    if (memcmp(&m_config, &c, sizeof(c)) != 0 || numMeshes != m_numMeshes)
    {
        m_config = c;
        m_numMeshes = numMeshes;
        init(m_streamCount);
    }
}

//------------------------------------------------------------------------

void TracerBVH8Lwda::launch(const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const void* const stream, const int streamIdx)
{
    TracerBVH8LwdaParams params;
    setupKernelParams(bvh, rays, rayFormat, hits, hitFormat, numRays, stream, streamIdx, params);
    m_lwda.clearDeviceBuffer(&m_warpCounters[streamIdx], 0, sizeof(int), (lwdaStream_t)stream);
    if (!m_launchFunc(m_lwda.getSMArch(), m_lwda.getNumSMs(), (lwdaStream_t)stream, params, m_config))
        throw LwdaRuntimeError(RT_EXCEPTION_INFO, "launchTrace()", lwdaErrorUnknown);
}

//------------------------------------------------------------------------

void TracerBVH8Lwda::setupKernelParams(const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const void* const stream, const int streamIdx, TracerBVH8LwdaParams& params)
{
    if (dynamic_cast<const TracerDataMeshBVH8*>(&bvh))
        copyMesh(params.mesh, dynamic_cast<const TracerDataMeshBVH8&>(bvh));
    else
    {
        const TracerDataGroupBVH8& tdg = dynamic_cast<const TracerDataGroupBVH8&>(bvh);
        params.group.nodes        = (float4*)tdg.bvh8Nodes;
        params.group.remap        = tdg.remap;
        params.group.ilwMatrices  = tdg.group.ilwMatrices;
        params.group.modelId      = tdg.group.modelIds;
        params.group.rootNode     = 0;
        params.group.numEntities  = tdg.group.numInstances;
        params.group.matrixStride = tdg.group.matrixStride;

        for (int i = 0; i < m_numMeshes; i++)
            copyMesh(m_meshData_h[i], tdg.bvh8Meshes[i]);

        // TODO [tkarras]: What if the user tries to execute multiple conlwrrent queries on several different RTPmodels?
        m_lwda.memcpyHtoDAsync(m_meshData_d, m_meshData_h.data(), m_numMeshes * sizeof(TracerParamsMesh), (lwdaStream_t)stream);
    }

    params.rays         = rays;
    params.hits         = hits;
    params.meshes       = m_meshData_d;
    params.warpCounter  = &m_warpCounters[streamIdx];
    params.numRays      = numRays;
    params.rayFormat    = rayFormat;
    params.hitFormat    = hitFormat;
}

//------------------------------------------------------------------------

void prodlib::bvhtools::copyMesh(TracerParamsMesh& dst, const TracerDataMeshBVH8& src)
{
    dst.nodes        = (float4*)src.bvh8Nodes;
    dst.triangles    = (float4*)src.bvh8Triangles;
    dst.indices      = (int3*)src.mesh.indices;
    dst.vertices     = src.mesh.vertices;
    dst.remap        = src.remap;
    dst.indexStride  = src.mesh.indexStride;
    dst.vertexStride = src.mesh.vertexStride;
    dst.rootNode     = 0;
    dst.numNodes     = src.numBVH8Nodes;
    dst.numRemaps    = src.numRemaps;
}

//------------------------------------------------------------------------
