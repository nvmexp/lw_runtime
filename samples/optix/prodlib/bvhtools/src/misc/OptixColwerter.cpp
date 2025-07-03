// Copyright LWPU Corporation 2013
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "OptixColwerter.hpp"
#include "OptixColwerterKernels.hpp"
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>
#include <corelib/math/MathUtil.h>

//#include <lwda_runtime.h>
#include <iostream>

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

void OptixColwerter::configure(const Config& cfg)
{
    RT_ASSERT(cfg.maxNodes >= 0);  
    RT_ASSERT(cfg.ioNodes.getNumElems() >= static_cast<size_t>( cfg.maxNodes ));
    RT_ASSERT(cfg.inNumNodes.getNumElems() == 1);
    RT_ASSERT(cfg.inApexPointMap.getNumBytes() >= sizeof(ApexPointMap));

    m_cfg = cfg;
    cfg.outRemap.setNumElems(m_cfg.inRemap.getNumElems());

    m_leafSize      .assignNew(m_cfg.maxNodes * 2);
    m_leafPos       .assignNew(m_cfg.maxNodes * 2);
    m_blockCount    .assignNew(1);
    m_scanTemp      .assignNew(m_cfg.lwca ? exclusiveScanTempSize(m_cfg.maxNodes * 2) : 0);

    m_cfg.tempBuffer
        .aggregate(m_leafSize)
        .aggregate(m_leafPos)
        .aggregate(m_blockCount)
        .aggregate(m_scanTemp)
      ;
}

//------------------------------------------------------------------------

void OptixColwerter::execute(void)
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

    m_cfg.tempBuffer.markAsUninitialized();
}

//------------------------------------------------------------------------

void OptixColwerter::execDevice(void)
{
    // Launch GatherLeafSizesKernel
    {
      GatherLeafSizesKernelParams p = {};
      p.outLeafSize  = m_leafSize.writeDiscardLWDA();
      p.nodes        = m_cfg.ioNodes.getSubrange(1).readLWDA();
      p.numNodes     = m_cfg.inNumNodes.readLWDA();

      LAUNCH(*m_cfg.lwca, GatherLeafSizesKernel, COLWERTER_WARPS_PER_BLOCK, m_cfg.maxNodes, p);
    }
    
    // Scan leafSize to get leafPos
    exclusiveScan( m_scanTemp.writeDiscardLWDA(), m_scanTemp.getNumBytes(), 
      m_leafSize.reinterpret<unsigned>().readWriteLWDA(), m_leafPos.reinterpret<unsigned>().writeDiscardLWDA(),
      2*(m_cfg.maxNodes), m_cfg.lwca->getDefaultStream());

    // Clear blockCount
    m_blockCount.clearLWDA(0);

    // Compute configuration
    int numThreads = m_cfg.maxNodes;
    int segmentSize = 0;
    int segmentCount = 0;
    if( m_cfg.shiftNodes )
    {
      numThreads = 32 * COLWERTER_WARPS_PER_BLOCK * m_cfg.lwca->getNumSMs();
      int warpIterations = corelib::idivCeil(m_cfg.maxNodes,numThreads);
      segmentSize = 32 * warpIterations;
      segmentCount = corelib::idivCeil( m_cfg.maxNodes, segmentSize );
    }

    // Launch OptixColwerterKernel
    {
      OptixColwerterKernelParams p = {};
      p.ioNodes         = m_cfg.ioNodes.readWriteLWDA();
      p.outRemap        = m_cfg.outRemap.writeDiscardLWDA();
      p.ioBlockCounter  = m_blockCount.readWriteLWDA();
      p.inRemap         = m_cfg.inRemap.readLWDA();
      p.numNodes        = m_cfg.inNumNodes.readLWDA();
      p.inApexPointMap  = m_cfg.inApexPointMap.readLWDA();
      p.inLeafPos       = m_leafPos.readLWDA();
      p.maxNodes        = m_cfg.maxNodes;
      p.scale           = getScale();
      p.segmentSize     = segmentSize;
      p.segmentCount    = segmentCount;

      LAUNCH(*m_cfg.lwca, OptixColwerterKernel, COLWERTER_WARPS_PER_BLOCK, numThreads, p, m_cfg.shiftNodes);
    }
}

//------------------------------------------------------------------------

int OptixColwerter::getScale()
{
  return m_cfg.bake ? 4 : 2;
}

//------------------------------------------------------------------------
// Do an exclusive scan on the sizes of the nodes and use these as the
// new leaf positions. This will ensure that if a node has two primitive
// lists, then they will end up next to each other in remap.

static void computeLeafPositions( int numNodes, BufferRef<const BvhNode> nodes, BufferRef<int> leafPos )
{
  int sum=0;
  for( int i=0; i < numNodes; i++ )
  {
    int c0num, c1num;
    getChildLeafSizes( nodes[i], &c0num, &c1num );
    leafPos[2*i+0] = sum;   sum += c0num;
    leafPos[2*i+1] = sum;   sum += c1num;
  }
}

//------------------------------------------------------------------------

void OptixColwerter::execHost(void)
{
  m_cfg.ioNodes         .readWriteHost();
  m_cfg.inNumNodes      .readHost();
  m_cfg.outRemap        .writeDiscardHost();
  m_cfg.inRemap         .readHost();
  m_cfg.inApexPointMap  .readHost();
  m_leafPos             .writeDiscardHost();

  if (m_cfg.shiftNodes)
    execHostWithShift();
  else
    execHostWithoutShift();
}

//------------------------------------------------------------------------
// TODO [tkarras]: Remove. This is no longer used in any configuration.

void OptixColwerter::execHostWithShift(void)
{
  int scale = getScale();
  RT_ASSERT( *m_cfg.inNumNodes <= m_cfg.maxNodes );
  computeLeafPositions( *m_cfg.inNumNodes, m_cfg.ioNodes, m_leafPos );

  // colwert nodes and shift up to create space for dummy root
  for (int i = *m_cfg.inNumNodes - 1; i >= 0; i--)
  {
    BvhNode& node = m_cfg.ioNodes[i];
    colwertNode( node, m_cfg.outRemap.getLwrPtr(), m_cfg.inRemap.getLwrPtr(), &m_leafPos[2*i], scale, 1 );
    m_cfg.ioNodes[i+1] = m_cfg.ioNodes[i];
  }

  writeDummyRoot( m_cfg.ioNodes[0], m_cfg.inApexPointMap.getLwrPtr(), scale );
}

//------------------------------------------------------------------------

void OptixColwerter::execHostWithoutShift(void)
{
  int scale = getScale();
  RT_ASSERT( *m_cfg.inNumNodes <= m_cfg.maxNodes );
  computeLeafPositions( *m_cfg.inNumNodes - 1, m_cfg.ioNodes.getSubrange(1), m_leafPos );

  for (int i = 0; i < *m_cfg.inNumNodes - 1; i++)
  {
    BvhNode& node = m_cfg.ioNodes[i+1];
    colwertNode( node, m_cfg.outRemap.getLwrPtr(), m_cfg.inRemap.getLwrPtr(), &m_leafPos[2*i], scale, 0 );
  }

  // Zero out the unused part of the buffer for easy detection of non-nodes
  memset(&m_cfg.ioNodes[*m_cfg.inNumNodes], 0, (m_cfg.maxNodes - *m_cfg.inNumNodes) * sizeof(BvhNode));

  writeDummyRoot( m_cfg.ioNodes[0], m_cfg.inApexPointMap.getLwrPtr(), scale );
}
