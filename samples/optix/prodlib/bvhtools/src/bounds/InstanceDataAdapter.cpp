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

#include "InstanceDataAdapter.hpp"
#include "InstanceDataAdapterKernels.hpp"
#include "ApexPointMapLookup.hpp"
#include <prodlib/bvhtools/src/common/SharedKernelFunctions.hpp>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

void InstanceDataAdapter::configure(const Config& cfg)
{
    RT_ASSERT(cfg.numInstances >= 0);
    if (cfg.inInstanceDescs.getNumElems() == 0)
    {
      RT_ASSERT(cfg.matrixStride % 4 == 0 && cfg.matrixStride >= 48);
      RT_ASSERT(cfg.inApexPointMaps.getNumElems() > 0 || cfg.numInstances == 0);
      RT_ASSERT(cfg.inTransforms.getNumElems() >= (size_t)(cfg.numInstances * (cfg.matrixStride / 4)));
      RT_ASSERT(cfg.inInstanceIds.getNumElems() >= (size_t)cfg.numInstances);    
    }

    // Set config and resize outputs.

    m_cfg = cfg;
    if( m_cfg.computeAabbs )
      m_cfg.outWorldSpaceAabbs.setNumElems(m_cfg.numInstances);
    if (cfg.inInstanceDescs.getNumElems() > 0)
      m_cfg.outBvhInstanceData.setNumElems(m_cfg.numInstances);
    else
      m_cfg.outIlwMatrices.setNumElems(m_cfg.numInstances * (m_cfg.matrixStride / 4));
}

//------------------------------------------------------------------------

void InstanceDataAdapter::execute(void)
{
    if (m_cfg.useLwda)
    {
        m_cfg.lwdaUtils->beginTimer(getName());
        execDevice();
        m_cfg.lwdaUtils->endTimer();
    }
    else
    {
        execHost();
    }
}

//------------------------------------------------------------------------

void InstanceDataAdapter::execDevice(void)
{
    // Launch InstanceDataAdapterExec.
    {
        InstanceDataAdapterExecParams p = {};
        p.outBvhInstanceData = m_cfg.outBvhInstanceData.writeDiscardLWDA();
        p.outWorldSpaceAabbs = m_cfg.outWorldSpaceAabbs.writeDiscardLWDA();
        p.outIlwMatrices     = m_cfg.outIlwMatrices.writeDiscardLWDA();
        p.inInstanceDesc     = m_cfg.inInstanceDescs.readLWDA();
        p.inApexPointMaps    = m_cfg.inApexPointMaps.readLWDA();
        p.inTransforms       = m_cfg.inTransforms.readLWDA();
        p.inInstanceIds      = m_cfg.inInstanceIds.readLWDA();
        p.numInstances       = m_cfg.numInstances;
        p.matrixStride       = m_cfg.matrixStride;

        LAUNCH(*m_cfg.lwdaUtils, InstanceDataAdapterExec, INSTANCE_DATA_ADAPTER_EXEC_WARPS_PER_BLOCK,
            m_cfg.numInstances, p);
    }
}

//------------------------------------------------------------------------

void InstanceDataAdapter::execHost(void)
{
    m_cfg.outBvhInstanceData  .writeDiscardHost();
    m_cfg.outWorldSpaceAabbs  .writeDiscardHost();
    m_cfg.outIlwMatrices      .writeDiscardHost();
    m_cfg.inInstanceDescs     .readHost();
    m_cfg.inApexPointMaps     .readHost();
    m_cfg.inTransforms        .readHost();
    m_cfg.inInstanceIds       .readHost();

    Config& p = m_cfg;

    for (int idx = 0; idx < m_cfg.numInstances; idx++)
    {
        if(p.inInstanceDescs.getNumElems() > 0)
        {
          const InstanceDesc& inInst = p.inInstanceDescs[idx];

          BufferRef<const BvhHeader> bvhHeaderBuf;
          bvhHeaderBuf.assignExternal((const BvhHeader*)inInst.bvh, 1, m_cfg.lwdaUtils ? MemorySpace_LWDA : MemorySpace_Host);
          const BvhHeader* bvhHeader = bvhHeaderBuf.materialize(m_cfg.lwdaUtils).readHost();

          BvhInstanceData outInst;
          copyTransform(outInst.transform, inInst.transform);
          computeIlwerse4x3_affine(outInst.ilwTransform, inInst.transform);
          outInst.bvh = inInst.bvh;
          outInst.instanceId = inInst.instanceId;
          outInst.instanceOffset = inInst.instanceOffsetAndFlags & 0xffffff;
          outInst.flags = inInst.instanceOffsetAndFlags >> 24;
          p.outBvhInstanceData[idx] = outInst;

          if(p.outWorldSpaceAabbs.getNumElems() > 0)
          {
            BufferRef<const ApexPointMap> apmBuf;
            apmBuf.assignExternal((const ApexPointMap*)make_ptr<ApexPointMap>(inInst.bvh, bvhHeader->apmOffset), 1, m_cfg.lwdaUtils ? MemorySpace_LWDA : MemorySpace_Host);
            const ApexPointMap* apm = apmBuf.materialize(m_cfg.lwdaUtils).readHost();

            PrimitiveAABB primAabb;
            computeAabb(
              inInst.transform,
              apm,
              idx,
              &primAabb
            );
            p.outWorldSpaceAabbs[idx] = primAabb;
          }
        }
        else
        {
          float transform[12];
          copyTransform(transform, &p.inTransforms[idx * (p.matrixStride / 4)]);
          computeIlwerse4x3_affine(&p.outIlwMatrices[idx * (p.matrixStride / 4)], transform);

          PrimitiveAABB primAabb;
          int instanceId = p.inInstanceIds[idx];
          computeAabb(
            transform,
            p.inApexPointMaps[instanceId],
            idx,
            &primAabb);
          p.outWorldSpaceAabbs[idx] = primAabb;
        }
    }
}
