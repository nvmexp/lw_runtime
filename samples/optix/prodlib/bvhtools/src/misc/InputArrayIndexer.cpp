// Copyright LWPU Corporation 2017
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "InputArrayIndexer.hpp"
#include <prodlib/bvhtools/src/common/SharedKernelFunctions.hpp>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/bvhtools/src/common/Utils.hpp>

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

void InputArrayIndexer::configure(const Config& cfg)
{
  m_cfg = cfg;

  m_numBlocks = (m_cfg.inBuffers->numPrimitives + INPUT_ARRAY_INDEXER_BLOCK_SIZE - 1) / INPUT_ARRAY_INDEXER_BLOCK_SIZE;
  m_cfg.outArrayBaseGlobalIndex.setNumElems(m_cfg.inBuffers->numArrays + 1);
  m_cfg.outArrayTransitionBits.setNumElems(m_numBlocks * (INPUT_ARRAY_INDEXER_BLOCK_SIZE / 32));
  m_cfg.outBlockStartArrayIndex.setNumElems(m_numBlocks);

  if (m_cfg.inBuffers->needsGeometryIndexRemap())
      m_cfg.outGeometryIndexArray.setNumElems(m_cfg.inBuffers->numArrays);
}

//------------------------------------------------------------------------

void InputArrayIndexer::execute(void)
{
   
    {
        int numArrays = m_cfg.inBuffers->numArrays;

        // Prefix sum over the input array lengths

        // TODO: Is it worth moving this to device?
        std::vector<unsigned int> arrayBaseGlobalIndex(numArrays + 1);
        unsigned int sumPrims = 0;
        for( int i = 0; i < numArrays; i++ )
        {
            arrayBaseGlobalIndex[i] = sumPrims;
            sumPrims += m_cfg.inBuffers->numPrimsArray[i];
        }
        arrayBaseGlobalIndex[numArrays] = sumPrims;

        MemorySpace memSpace = m_cfg.lwca ? MemorySpace_LWDA : MemorySpace_Host;
        memcpyInlineWAR((char *) m_cfg.outArrayBaseGlobalIndex.writeDiscard(memSpace), (char *) arrayBaseGlobalIndex.data(), sizeof(unsigned int) * arrayBaseGlobalIndex.size(), m_cfg.lwca);
        if (m_cfg.inBuffers->needsGeometryIndexRemap())
            memcpyInlineWAR((char *) m_cfg.outGeometryIndexArray.writeDiscard(memSpace), (char *) m_cfg.inBuffers->geometryIndexArray.readHost(), sizeof(int) * m_cfg.inBuffers->numArrays, m_cfg.lwca);
    }

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

//------------------------------------------------------------------------

void InputArrayIndexer::execDevice(void)
{

    // Initialize bit blocks to zero before starting

    m_cfg.outArrayTransitionBits.clearLWDA(0);

    // Launch InputArrayIndexer.
    {
        InputArrayIndexerExecParams p = {};

        p.numInputs               = m_cfg.inBuffers->numArrays;
        p.numBlocks               = m_numBlocks;
        p.outArrayBaseGlobalIndex  = m_cfg.outArrayBaseGlobalIndex.readLWDA();
        p.outArrayTransitionBits  = m_cfg.outArrayTransitionBits.writeLWDA();
        p.outBlockStartArrayIndex = m_cfg.outBlockStartArrayIndex.writeDiscardLWDA();

        LAUNCH(*m_cfg.lwca, InputArrayIndexerExec, INPUT_ARRAY_INDEXER_EXEC_WARPS_PER_BLOCK, max( p.numBlocks, p.numInputs), p);
    }
}

//------------------------------------------------------------------------

void InputArrayIndexer::execHost(void)
{
    int numInputs = m_cfg.inBuffers->numArrays;
    const unsigned int *arrayBaseGlobalIndex = m_cfg.outArrayBaseGlobalIndex.readHost();
    unsigned int *arrayTransitionBits = m_cfg.outArrayTransitionBits.writeDiscardHost();
    int *blockIndex = m_cfg.outBlockStartArrayIndex.writeDiscardHost();

    // Initialize bit blocks to zero

    memset(arrayTransitionBits, 0, m_cfg.outArrayTransitionBits.getNumBytes());

    // Mark transitions in bit blocks

    for (int inputIdx = 1; inputIdx < numInputs + 1; inputIdx++)
    {
        int loc = arrayBaseGlobalIndex[inputIdx] - 1;
        int block = loc / 32;
        arrayTransitionBits[block] |= 1 << (loc & 31);
    }

    // Binary search to assign block indices
    // Note: This is not the fastest algorithm to use on the host side
    // and could be optimized in the future.

    int nBlocks = (int) m_cfg.outBlockStartArrayIndex.getNumElems();
    for (int b = 0; b < nBlocks; b++)
    {
        int offset = INPUT_ARRAY_INDEXER_BLOCK_SIZE * b;
        int start = 0;
        int end = numInputs;

        int index = 0;
        for (;;)
        {
            index = (end + start) >> 1;
            if (offset < (int) arrayBaseGlobalIndex[index])
                end = index;
            else if (offset >= (int) arrayBaseGlobalIndex[index + 1])
                start = index + 1;
            else
                break;
        }

        blockIndex[b] = index;
    }

}

//------------------------------------------------------------------------
