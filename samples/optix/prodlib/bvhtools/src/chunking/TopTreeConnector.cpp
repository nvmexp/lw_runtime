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

#include "TopTreeConnector.hpp"
#include "TopTreeConnectorKernels.hpp"
#include "../common/SharedKernelFunctions.hpp"
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>

using namespace prodlib::bvhtools; 

void TopTreeConnector::configure(const Config& cfg)
{
  RT_ASSERT(cfg.maxNodes >= 0);
  RT_ASSERT(cfg.ioNodes.getNumElems() >= (size_t)cfg.maxNodes);
  RT_ASSERT(cfg.inNodeRange.getNumElems() == 1);
  RT_ASSERT(cfg.inRemap.getNumElems() >= (size_t)cfg.maxNodes + 1);
  RT_ASSERT(cfg.inTrimmedAabbs.getNumElems() >= (size_t)cfg.maxNodes + 1);

  m_cfg = cfg;
  m_cfg.outRoot.setNumElems(1);
}

void TopTreeConnector::execute(void)
{
  if (m_cfg.lwca)
  {
    m_cfg.lwca->beginTimer(getName());
    execDevice();
    m_cfg.lwca->endTimer();
  }
  else
  {
    execHost();
  }
}

void TopTreeConnector::execDevice(void)
{
  {
    TopTreeConnectorParams p = {};
    p.outRoot           = m_cfg.outRoot.writeDiscardLWDA();
    p.ioNodes           = m_cfg.ioNodes.readWriteLWDA();
    p.inRemap           = m_cfg.inRemap.readLWDA();
    p.inTrimmedAabbs    = m_cfg.inTrimmedAabbs.readLWDA();
    p.nodeRange         = m_cfg.inNodeRange.readLWDA();

    LAUNCH(*m_cfg.lwca, TopTreeConnector, TOP_TREE_CONNECTOR_WARPS_PER_BLOCK, m_cfg.maxNodes, p);
  }
}

void TopTreeConnector::execHost(void)
{
  m_cfg.outRoot         .writeDiscardHost();
  m_cfg.ioNodes         .readWriteHost();
  m_cfg.inNodeRange     .readHost();
  m_cfg.inRemap         .readHost();
  m_cfg.inTrimmedAabbs  .readHost();

  int numTopNodes = m_cfg.inNodeRange->span();
  BufferRef<BvhNode> topNodes = m_cfg.ioNodes.getSubrange(m_cfg.inNodeRange->start, numTopNodes);
  for( int idx = 0; idx < numTopNodes; idx++ ) 
  {
    if( topNodes[idx].c1num == BVHNODE_NOT_IN_USE ) 
      continue;
    
    BvhHalfNode* halfNodes = (BvhHalfNode*)&topNodes[idx];
    connectTopTreeLeaf( m_cfg.inTrimmedAabbs.getLwrPtr(), m_cfg.inRemap.getLwrPtr(), halfNodes[0] );
    connectTopTreeLeaf( m_cfg.inTrimmedAabbs.getLwrPtr(), m_cfg.inRemap.getLwrPtr(), halfNodes[1] );
  }

  // Move the root
  *m_cfg.outRoot = topNodes[0];
  topNodes[0].c1num = BVHNODE_NOT_IN_USE;
}
