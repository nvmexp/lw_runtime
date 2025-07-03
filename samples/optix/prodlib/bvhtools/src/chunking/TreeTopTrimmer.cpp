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

#include "TreeTopTrimmer.hpp"
#include "TreeTopTrimmerKernels.hpp"
#include "../common/SharedKernelFunctions.hpp"
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>

using namespace prodlib::bvhtools;

void TreeTopTrimmer::configure(const Config& cfg)
{
  RT_ASSERT(cfg.maxNodes >= 0);
  RT_ASSERT(cfg.ioTrimmed.getNumElems() >= 2);
  RT_ASSERT(cfg.ioTrimmedRange.getNumElems() == 1);
  RT_ASSERT(cfg.ioNodes.getNumElems() >= (size_t)cfg.maxNodes);
  RT_ASSERT(cfg.inNodeRange.getNumElems() == 1);
  RT_ASSERT(cfg.inLwtoffLevel.getNumElems() == 1);
  RT_ASSERT(cfg.inNodeParents.getNumElems() >= (size_t)cfg.maxNodes);

  m_cfg = cfg;
}

void TreeTopTrimmer::execute(void)
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

void TreeTopTrimmer::execDevice(void)
{
  {
    TrimTopmostNodesParams p = {};
    p.outTrimmed        = m_cfg.ioTrimmed.writeLWDA();
    p.ioTrimmedRange    = m_cfg.ioTrimmedRange.readWriteLWDA();
    p.ioNodes           = m_cfg.ioNodes.readWriteLWDA();
    p.lwtoffLevel       = m_cfg.inLwtoffLevel.readLWDA();
    p.nodeParents       = m_cfg.inNodeParents.readLWDA();
    p.nodeRange         = m_cfg.inNodeRange.readLWDA();

    LAUNCH(*m_cfg.lwca, TrimTopmostNodes, TRIM_TOPMOST_NODES_WARPS_PER_BLOCK, m_cfg.maxNodes, p);
  }
}

void TreeTopTrimmer::execHost(void)
{
  m_cfg.ioTrimmed       .writeHost();
  m_cfg.ioTrimmedRange  .readWriteHost();
  m_cfg.ioNodes         .readWriteHost();
  m_cfg.inLwtoffLevel   .readHost();
  m_cfg.inNodeRange     .readHost();
  m_cfg.inNodeParents   .readHost();

  int nodeOfs = m_cfg.inNodeRange->start;
  int numNodes = m_cfg.inNodeRange->span();
  BufferRef<BvhNode> nodes = m_cfg.ioNodes.getSubrange(nodeOfs, numNodes);

  for( int idx=0; idx < numNodes; idx++ ) 
  {
    if( nodes[idx].c1num == BVHNODE_NOT_IN_USE ) 
      continue;
    
    int L = computeNodeDepth( m_cfg.inNodeParents.getLwrPtr(), idx );
    if( L <= *m_cfg.inLwtoffLevel ) 
    {
      BvhNode& n = nodes[idx];
      BvhHalfNode* hn = (BvhHalfNode*)&n;

      // Store exposed treelet roots immediately above cutoff level,
      // and any leaves above the cutoff level.

      if ((L == *m_cfg.inLwtoffLevel || n.c0num > 0) && isHalfNodeValid(hn[0]))
      {
        int pos = m_cfg.ioTrimmedRange->end++;
        storeTrimmedHalfNode( m_cfg.ioTrimmed.getLwrPtr(), hn[0], pos );
      }

      if ((L == *m_cfg.inLwtoffLevel || n.c1num > 0) && isHalfNodeValid(hn[1]))
      {
        int pos = m_cfg.ioTrimmedRange->end++;
        storeTrimmedHalfNode( m_cfg.ioTrimmed.getLwrPtr(), hn[1], pos );
      }

      nodes[idx].c1num = BVHNODE_NOT_IN_USE;
    }
  }
}
