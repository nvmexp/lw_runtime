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

#include "TracerLwda.hpp"
#include "src/common/Utils.hpp"
#include <string.h>

#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/Assert.h>

#include <vector>

using namespace prodlib::bvhtools;
using namespace prodlib;

// copyMesh is in separate build unit to cut linker dependence chain
// between TTU traversal and BVH2 traversal
namespace prodlib
{
namespace bvhtools
{
  void copyMesh( TracerParamsMesh& dst, const TracerDataMesh& src );
} // namespace bvhtools
} // namespace prodlib

//------------------------------------------------------------------------

TracerLwda::TracerLwda(void)
:   m_warpsPerBlock         (0),
    m_usePersistentThreads  (false),
    m_launchFunc            (NULL),
    m_numPersistentThreads  (0),
    m_warpCounters          (0),
    m_kernelData_d          (0),
    m_streamCount           (0),
    m_numMeshes             (-1)
{
    strcpy(m_backendName, "GPU-accelerated BVH2");

    m_bvhLayout.storeOnGPU        = true;
    m_bvhLayout.arrayAlign        = 4096;
    m_bvhLayout.reorderTriangles  = true;
    m_bvhLayout.optixNodes        = false;

    m_config.twoLevel      = false;
    m_config.useWoop       = false;
    m_config.useFloat4     = false;
    m_config.useTex        = false;
    m_config.useMasking    = false;
    m_config.useWatertight = false;
}

//------------------------------------------------------------------------

TracerLwda::~TracerLwda(void)
{
    m_lwda.deviceFree(m_warpCounters);
    m_lwda.deviceFree(m_kernelData_d);
}

//------------------------------------------------------------------------

struct SMConfig
{
  int   smVersion;
  int   warpsPerBlock;
  bool  usePersistentThreads;
  const char* backendName;
  TraceLaunchFunc launchFunc;
};

static SMConfig SM52CONFIG = { 52, 4, true , " (optimized for sm_52)", launchTrace52 };
static SMConfig SM35CONFIG = { 35, 4, true , " (optimized for sm_35)", launchTrace35 };
static SMConfig SM30CONFIG = { 30, 4, true , " (optimized for sm_30)", launchTrace30 };

static SMConfig getSmConfig( int smVersion )
{
  if (smVersion >= 50) {
    return SM52CONFIG;
  } else if ( smVersion >= 35 ) {
    return SM35CONFIG;
  } else if( smVersion >= 30 ) {
    return SM30CONFIG;
  } else {
    throw IlwalidValue( RT_EXCEPTION_INFO, "Unsupported SM version!" );
  }
}

//------------------------------------------------------------------------

void TracerLwda::init(int streamCount)
{
    // Choose implementation.

    m_bvhLayout.arrayAlign = m_lwda.getTextureAlign();

    SMConfig smConfig = getSmConfig( m_lwda.getSMArch() );
    m_warpsPerBlock = smConfig.warpsPerBlock; 
    m_usePersistentThreads = smConfig.usePersistentThreads;    
    m_launchFunc = smConfig.launchFunc;

    strcpy(m_backendName, "GPU-accelerated BVH2");
    strcat( m_backendName, smConfig.backendName );

    // We store the group and mesh data in one block of data.

    int newKernelDataSize = sizeof(TracerParamsGroup) + m_lwda.getTextureAlign() + m_numMeshes*sizeof(TracerParamsMesh);
    m_lwda.deviceFree( m_kernelData_d );
    m_kernelData_d = (char*)m_lwda.deviceAlloc( newKernelDataSize );
    m_kernelData_h.resize( newKernelDataSize );

    // Initialize persistent threads.

    if (m_usePersistentThreads)
    {
      // Query the number of threads per SM.

      int maxThreadsPerSM = m_lwda.getMaxThreadsPerSM();
      
      // Determine how many threads to launch.

      m_numPersistentThreads = m_lwda.getNumSMs() * maxThreadsPerSM;

      // Allocate warp counters
      m_lwda.deviceFree( m_warpCounters );
      m_warpCounters = (int*)m_lwda.deviceAlloc( streamCount*sizeof(int) );
      m_streamCount = streamCount;      
    }
}

//------------------------------------------------------------------------

void TracerLwda::traceFromDeviceMem(const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyhit, const bool watertight, const void* const stream, const int streamIdx)
{
  if( !bvh.inDeviceMem )
    throw IlwalidValue( RT_EXCEPTION_INFO, "Can't trace BVH stored on host" );

  configure( bvh, rayFormat, watertight );

  launch( bvh, rays, rayFormat, hits, hitFormat, numRays, anyhit, stream, streamIdx );
}

//------------------------------------------------------------------------

void TracerLwda::configure( const TracerData& data, int rayFormat, bool watertight )
{
  const TracerDataGroup* tdg = dynamic_cast<const TracerDataGroup*>( &data );
  const int numMeshes = tdg ? static_cast<int>(tdg->meshes.size()) : 1;

  const TracerDataMesh *tdm = (tdg && numMeshes > 0) ? &tdg->meshes[0] : dynamic_cast<const TracerDataMesh*>( &data );
  const int vertexStride = tdm ? tdm->mesh.vertexStride : 0;
 
  const size_t ntexels = (data.numNodes * sizeof(BvhNode)) / sizeof(float4);
  
  TracerLwdaConfig c;
  c.twoLevel   = (tdg != NULL);
  c.useWoop    = tdm ? (tdm->woopTriangles != NULL) : false;
  c.useFloat4  = (!c.useWoop && vertexStride == 16);
  c.useTex     = (!c.twoLevel && m_lwda.getSMArch() < 35 && ntexels < (size_t)m_lwda.getMaxTextureSize1D());
  c.useMasking = (rayFormat == RAY_ORIGIN_MASK_DIRECTION_TMAX);
  c.useWatertight = watertight;
  if( memcmp(&m_config, &c, sizeof(c)) != 0 || numMeshes != m_numMeshes)
  {
    m_config = c;
    m_numMeshes = numMeshes;
    init( m_streamCount );
  }
}

//------------------------------------------------------------------------

void TracerLwda::launch( const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyhit, const void* const stream, const int streamIdx )
{
  // Initialize kernel parameters.
  TracerLwdaParams params;
  setupKernelParams( bvh, rays, rayFormat, hits, hitFormat, numRays, anyhit, stream, streamIdx, params);

  int numThreads = numRays;
  if (m_usePersistentThreads)
  {
    numThreads = m_numPersistentThreads;
    m_lwda.clearDeviceBuffer(&m_warpCounters[streamIdx], 0, sizeof(int), (lwdaStream_t)stream);
  }

  dim3 blockDim(32, m_warpsPerBlock);
  dim3 gridDim = m_lwda.calcGridDim(numThreads, blockDim);
  if (!m_launchFunc(gridDim, blockDim, (lwdaStream_t)stream, params, m_config))
    throw LwdaRuntimeError(RT_EXCEPTION_INFO, "launchTrace()", lwdaErrorUnknown);
}

//------------------------------------------------------------------------
void TracerLwda::setupKernelParams( const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyhit, const void* const stream, const int streamIdx, TracerLwdaParams& params )
{
  // Initialize kernel parameters.
  char* hostTracerParams = m_kernelData_h.data();

  size_t align = m_lwda.getTextureAlign();
  TracerParamsGroup* groupParams = (TracerParamsGroup*)hostTracerParams;
  TracerParamsMesh*   meshParams = (TracerParamsMesh*)(((uintptr_t)groupParams + sizeof(TracerParamsGroup) + align - 1) & ~(align - 1));
  ptrdiff_t         meshesOffset = (char*)meshParams - (char*)groupParams;

  bool isTriangleMesh = dynamic_cast<const TracerDataMesh*>( &bvh ) != 0 ? true : false;

  if( isTriangleMesh )
  {
    const TracerDataMesh& mesh = dynamic_cast<const TracerDataMesh&>(bvh);
    copyMesh( *meshParams, mesh );
  }
  else
  {
    const TracerDataGroup& tdg = dynamic_cast<const TracerDataGroup&>(bvh);

    groupParams->nodes        = (float4*)tdg.nodes;
    groupParams->remap        = tdg.remap;
    groupParams->ilwMatrices  = tdg.group.ilwMatrices;
    groupParams->modelId      = tdg.group.modelIds;
    groupParams->rootNode     = (tdg.nodes) ? 0 : EntrypointSentinel;
    groupParams->numEntities  = tdg.group.numInstances;
    groupParams->matrixStride = tdg.group.matrixStride;

    for( size_t i = 0; i < tdg.meshes.size(); i++ )
    {
      const TracerDataMesh& mesh = dynamic_cast<const TracerDataMesh&>(tdg.meshes[i]);
      copyMesh( meshParams[i], mesh );
    }
  }
  // TODO [tkarras]: What if the user tries to execute multiple conlwrrent queries on several different RTPmodels?
  m_lwda.memcpyHtoDAsync( m_kernelData_d, hostTracerParams, m_kernelData_h.size(), (lwdaStream_t)stream);

  params.rays         = rays;
  params.hits         = hits;
  params.warpCounter  = &m_warpCounters[streamIdx];
  params.numRays      = numRays;
  params.rayFormat    = rayFormat;
  params.hitFormat    = hitFormat;
  params.anyhit       = (anyhit) ? 1 : 0;
  params.group        = (TracerParamsGroup*)m_kernelData_d;
  params.meshes       = (TracerParamsMesh*)(m_kernelData_d + meshesOffset);

  if (m_numMeshes)
      params.firstMesh = meshParams[0];
}
