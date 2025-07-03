// Copyright LWPU Corporation 2016
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "BVH8Constructor.hpp"
#include <prodlib/bvhtools/src/common/SharedKernelFunctions.hpp>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

void BVH8Constructor::configure(const Config& cfg)
{
    // Check for errors.

    RT_ASSERT(cfg.maxPrims >= 0);
    RT_ASSERT(cfg.maxNodes >= 0);
    RT_ASSERT(cfg.inNumNodes.getNumElems() == 1);
    RT_ASSERT(cfg.inBinaryNodes.getNumElems() >= 2);

    if (cfg.maxBranchingFactor < 2 || cfg.maxBranchingFactor > 8)
        throw IlwalidValue(RT_EXCEPTION_INFO, "maxBranchingFactor must be between 2 and 8!", cfg.maxBranchingFactor);

    if (cfg.maxLeafSize < 1 || cfg.maxLeafSize > BVH8CONSTRUCTOR_EXEC_MAX_LEAF_SIZE)
        throw IlwalidValue(RT_EXCEPTION_INFO, "maxLeafSize must be between 1 and BVH8CONSTRUCTOR_EXEC_MAX_LEAF_SIZE!", cfg.maxLeafSize);

    // Set config and resize outputs.

    m_cfg = cfg;
    m_cfg.outNodes.setNumElems(m_cfg.maxNodes);
    m_cfg.outNodeAux.setNumElems(m_cfg.maxNodes);
    m_cfg.outRemap.setNumElems(m_cfg.maxPrims);

    // Layout temp buffers.

    m_nodeCounter.assignNew(1);
    m_cfg.tempBuffer.aggregate(m_nodeCounter);
}

//------------------------------------------------------------------------

void BVH8Constructor::execute(void)
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

void BVH8Constructor::execDevice(void)
{
    RT_ASSERT((1 << BVH8CONSTRUCTOR_EXEC_MAX_DEPTH) == 32);
    RT_ASSERT(BVH8CONSTRUCTOR_EXEC_MAX_DEPTH == PLANNER_EXEC_MAX_DEPTH);

    // Clear buffers.

    m_cfg.outNodeAux.clearLWDA((unsigned char)-1);
    m_nodeCounter.clearLWDA(0);

    // Launch BVH8ConstructorExec.
    {
        BVH8ConstructorExecParams p = {};
        p.outNodes              = m_cfg.outNodes.writeDiscardLWDA();
        p.outNodeAux            = m_cfg.outNodeAux.readWriteLWDA();
        p.outRemap              = m_cfg.outRemap.writeDiscardLWDA();
        p.nodeCounter           = m_nodeCounter.readWriteLWDA();
        p.inNumNodes            = m_cfg.inNumNodes.readLWDA();
        p.inBinaryNodes         = m_cfg.inBinaryNodes.readLWDA();
        p.maxBranchingFactor    = m_cfg.maxBranchingFactor;

        LAUNCH(*m_cfg.lwca, BVH8ConstructorExec, BVH8CONSTRUCTOR_EXEC_WARPS_PER_BLOCK,
            m_cfg.maxNodes * 32, p);
    }
}

//------------------------------------------------------------------------

void BVH8Constructor::execHost(void)
{
    m_cfg.outNodes      .writeDiscardHost();
    m_cfg.outNodeAux    .writeDiscardHost();
    m_cfg.outRemap      .writeDiscardHost();
    m_cfg.inNumNodes    .readHost();
    m_cfg.inBinaryNodes .readHost();

    RT_ASSERT(*m_cfg.inNumNodes <= m_cfg.maxNodes);

    // Clear aux data.

    memset(m_cfg.outNodeAux.getLwrPtr(), -1, m_cfg.outNodeAux.getNumBytes());

    // Initialize aux data for root node.

    BVH8NodeAux& rootAux    = m_cfg.outNodeAux[0];
    rootAux.numChildNodes   = 0;
    rootAux.parentNodeIdx   = -1;
    rootAux.rootPairIdx     = 0;
    rootAux.firstRemapIdx   = 0;

    // Process nodes in top-down order.

    std::vector<int> binaryNodeStack;
    int nodeIdxOut = 1;

    for (int nodeIdx = 0; nodeIdx < nodeIdxOut; nodeIdx++)
    {
        BVH8Node&       node        = m_cfg.outNodes[nodeIdx];
        BVH8NodeAux&    aux         = m_cfg.outNodeAux[nodeIdx];
        int             remapIdx    = aux.firstRemapIdx;

        // Small subtree => ignore ownerSubtreePrims and expand the node all the way.

        int subtreePrims = (m_cfg.inBinaryNodes[aux.rootPairIdx + 0].afterPlan.subtreePrims + m_cfg.inBinaryNodes[aux.rootPairIdx + 1].afterPlan.subtreePrims) & INT_MAX;
        int internalNodeTag = (subtreePrims > m_cfg.maxBranchingFactor) ? subtreePrims : 0;

        // Identify the children of this node by traversing the binary tree.

        bool childIsLeaf[8];
        int childBinaryNodeIdx[8];
        int numChildren = 0;

        binaryNodeStack.clear();
        for (int c = 1; c >= 0; c--)
            binaryNodeStack.push_back(aux.rootPairIdx + c);

        while (binaryNodeStack.size())
        {
            int binaryNodeIdx = binaryNodeStack.back();
            binaryNodeStack.pop_back();
            const PlannerBinaryNode& binaryNode = m_cfg.inBinaryNodes[binaryNodeIdx];

            // Internal binary node of the 8-wide node?

            if (binaryNode.afterPlan.ownerSubtreePrims >= internalNodeTag && binaryNode.childPairIdx >= 0)
            {
                for (int c = 1; c >= 0; c--)
                    binaryNodeStack.push_back(binaryNode.childPairIdx + c);
            }

            // Child of the 8-wide node?

            else if (binaryNode.childPairIdx != INT_MIN)
            {
                RT_ASSERT(numChildren < m_cfg.maxBranchingFactor);
                childIsLeaf[numChildren] = (binaryNode.afterPlan.subtreePrims < 0);
                childBinaryNodeIdx[numChildren] = binaryNodeIdx;
                numChildren++;
            }
        }

        // Initialize node header.

        memset(&node, 0, sizeof(BVH8Node));
        node.header.firstChildIdx = nodeIdxOut;
        node.header.firstRemapIdx = remapIdx;
        for (int i = 0; i < 8; i++)
            node.header.meta[i].setEmpty();

        // Process each leaf child.

        for (int childIdx = 0; childIdx < numChildren; childIdx++)
        {
            if (!childIsLeaf[childIdx])
                continue;

            // Initialize child data.

            int numLeafPrims = m_cfg.inBinaryNodes[childBinaryNodeIdx[childIdx]].afterPlan.subtreePrims & INT_MAX;
            node.header.meta[childIdx].setLeaf(remapIdx - node.header.firstRemapIdx, numLeafPrims);

            // Traverse BVH subtree to identify individual primitives.

            binaryNodeStack.clear();
            binaryNodeStack.push_back(childBinaryNodeIdx[childIdx]);

            while (binaryNodeStack.size())
            {
                int childPairIdx = m_cfg.inBinaryNodes[binaryNodeStack.back()].childPairIdx;
                binaryNodeStack.pop_back();

                if (childPairIdx < 0)
                {
                    m_cfg.outRemap[remapIdx] = ~childPairIdx;
                    remapIdx++;
                }
                else
                {
                    for (int c = 1; c >= 0; c--)
                        binaryNodeStack.push_back(childPairIdx + c);
                }
            }
        }

        // Process each child node.

        for (int childIdx = 0; childIdx < numChildren; childIdx++)
        {
            if (childIsLeaf[childIdx])
                continue;

            // Initialize aux header for the child node.

            BVH8NodeAux& childAux   = m_cfg.outNodeAux[nodeIdxOut];
            childAux.numChildNodes  = 0; // incremented when processing the child
            childAux.parentNodeIdx  = nodeIdx;
            childAux.rootPairIdx    = m_cfg.inBinaryNodes[childBinaryNodeIdx[childIdx]].childPairIdx;
            childAux.firstRemapIdx  = remapIdx;

            // Mark as a inner node and increment node/primitive index.

            node.header.meta[childIdx].setInner(childIdx);
            nodeIdxOut++;
            remapIdx += m_cfg.inBinaryNodes[childBinaryNodeIdx[childIdx]].afterPlan.subtreePrims & INT_MAX;
            aux.numChildNodes++;
        }
    }

    RT_ASSERT(nodeIdxOut == *m_cfg.inNumNodes);
}

//------------------------------------------------------------------------
