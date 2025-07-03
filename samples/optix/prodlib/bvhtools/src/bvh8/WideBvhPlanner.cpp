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

#include "WideBvhPlanner.hpp"
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>
#include <algorithm>

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

void WideBvhPlanner::configure(const Config& cfg)
{
    // Check for errors.

    RT_ASSERT(cfg.ioBinaryNodes.getNumElems() >= 2 && cfg.ioBinaryNodes.getNumElems() % 2 == 0);
    RT_ASSERT(cfg.inPrimRange.getNumElems() == 1);
    RT_ASSERT(cfg.inApexPointMap.getNumBytes() >= sizeof(ApexPointMap));

    if (cfg.maxBranchingFactor < 2 || cfg.maxBranchingFactor > PLANNER_SQUEEZE_MAX_BFACTOR)
        throw IlwalidValue(RT_EXCEPTION_INFO, "maxBranchingFactor must be between 2 and PLANNER_SQUEEZE_MAX_BFACTOR!", cfg.maxBranchingFactor);

    if (cfg.maxLeafSize < 1 || cfg.maxLeafSize > PLANNER_SQUEEZE_MAX_LEAF_SIZE)
        throw IlwalidValue(RT_EXCEPTION_INFO, "maxLeafSize must be between 1 and PLANNER_SQUEEZE_MAX_LEAF_SIZE!", cfg.maxLeafSize);

    if (!(cfg.minAreaPerRoot >= 0.0f && cfg.minAreaPerRoot <= 1.0f))
        throw IlwalidValue(RT_EXCEPTION_INFO, "minAreaRoot must be between 0.0 and 1.0!", cfg.minAreaPerRoot);

    if (!(cfg.sahNodeCost >= 0.0f))
        throw IlwalidValue(RT_EXCEPTION_INFO, "sahNodeCost must be non-negative!", cfg.sahNodeCost);

    if (!(cfg.sahPrimCost >= 0.0f))
        throw IlwalidValue(RT_EXCEPTION_INFO, "sahPrimCost must be non-negative!", cfg.sahPrimCost);

    if (!(cfg.sahRootCost >= 0.0f))
        throw IlwalidValue(RT_EXCEPTION_INFO, "sahRootCost must be non-negative!", cfg.sahRootCost);

    if (!(cfg.sahTopLevelCost >= 0.0f))
        throw IlwalidValue(RT_EXCEPTION_INFO, "sahTopLevelCost must be non-negative!", cfg.sahTopLevelCost);

    // Set config and resize outputs.

    m_cfg = cfg;
    m_maxWideNodes = callwlateMaxWideNodes((int)m_cfg.ioBinaryNodes.getNumElems() / 2 + 1);

    m_cfg.outNumWideNodes   .setNumElems(1);
    m_cfg.outNumRoots       .setNumElems((m_cfg.allowMultipleRoots) ? 1 : 0);
    m_cfg.outSubtreeRoots   .setNumElems((m_cfg.allowMultipleRoots) ? m_cfg.ioBinaryNodes.getNumElems() : 0);

    // Layout temp buffers.

    m_workCounter   .assignNew(1);
    m_taskCounter   .assignNew(1);
    m_primCounter   .assignNew(1);
    m_tempA         .assignNew(m_cfg.ioBinaryNodes.getNumElems() / 2);
    m_tempB         .assignNew(m_cfg.ioBinaryNodes.getNumElems() / 2);
    m_squeezeTasks  .assignNew(m_maxWideNodes);

    m_cfg.tempBufferA
        .aggregate(m_tempA)
        .aggregate(m_workCounter);

    m_cfg.tempBufferB
        .overlay(m_tempB)
        .overlay(aggregate(m_squeezeTasks, m_taskCounter, m_primCounter));
}

//------------------------------------------------------------------------

void WideBvhPlanner::execute(void)
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

    m_cfg.tempBufferA.markAsUninitialized();
    m_cfg.tempBufferB.markAsUninitialized();
}

//------------------------------------------------------------------------

void WideBvhPlanner::execDevice(void)
{
    RT_ASSERT((1 << PLANNER_EXEC_MAX_DEPTH) <= 32);
    RT_ASSERT(PLANNER_SQUEEZE_MAX_BFACTOR * 2 - 2 <= 32);

    // Clear buffers for PlannerExec.

    m_cfg.outSubtreeRoots.clearLWDA(0);
    m_workCounter.clearLWDA(0);
    m_tempA.clearLWDA(0);

    // Launch PlannerExec.
    {
        PlannerExecParams p     = {};
        p.outNumWideNodes       = m_cfg.outNumWideNodes.writeDiscardLWDA();
        p.outNumRoots           = m_cfg.outNumRoots.writeDiscardLWDA();
        p.outSubtreeRoots       = m_cfg.outSubtreeRoots.readWriteLWDA();
        p.ioBinaryNodes         = m_cfg.ioBinaryNodes.readWriteLWDA();
        p.workCounter           = m_workCounter.readWriteLWDA();
        p.tempA                 = m_tempA.readWriteLWDA();
        p.tempB                 = m_tempB.writeDiscardLWDA();
        p.inPrimRange           = m_cfg.inPrimRange.readLWDA();
        p.inApexPointMap        = m_cfg.inApexPointMap.readLWDA();
        p.maxBranchingFactor    = m_cfg.maxBranchingFactor;
        p.maxLeafSize           = m_cfg.maxLeafSize;
        p.minAreaPerRoot        = m_cfg.minAreaPerRoot;
        p.sahNodeCost           = m_cfg.sahNodeCost;
        p.sahPrimCost           = m_cfg.sahPrimCost;
        p.sahRootCost           = m_cfg.sahRootCost;
        p.sahTopLevelCost       = m_cfg.sahTopLevelCost;

        LAUNCH(*m_cfg.lwca, PlannerExec, PLANNER_EXEC_WARPS_PER_BLOCK,
            min(m_cfg.lwca->getMaxThreads(), (int)m_cfg.ioBinaryNodes.getNumElems()), p);
    }

    // Squeezing enabled?

    if (m_cfg.squeezeMode != SqueezeMode_NoSqueeze)
    {
        // Clear buffers for PlannerSqueeze.

        m_squeezeTasks.clearLWDA((unsigned char)-1);
        m_workCounter.clear(0);
        m_primCounter.clearLWDA((unsigned char)-1);

        // Launch PlannerSqueeze.
        {
            PlannerSqueezeParams p  = {};
            p.ioNumWideNodes        = m_cfg.outNumWideNodes.readWriteLWDA();
            p.ioNumRoots            = m_cfg.outNumRoots.readWriteLWDA();
            p.ioBinaryNodes         = m_cfg.ioBinaryNodes.readWriteLWDA();
            p.squeezeTasks          = m_squeezeTasks.readWriteLWDA();
            p.workCounter           = m_workCounter.readWriteLWDA();
            p.taskCounter           = m_taskCounter.writeDiscardLWDA();
            p.primCounter           = m_primCounter.readWriteLWDA();
            p.inPrimRange           = m_cfg.inPrimRange.readLWDA();
            p.inTempA               = m_tempA.readLWDA();
            p.maxBranchingFactor    = m_cfg.maxBranchingFactor;
            p.maxLeafSize           = m_cfg.maxLeafSize;
            p.maxWideNodes          = m_maxWideNodes;

            LAUNCH(*m_cfg.lwca, PlannerSqueeze, PLANNER_SQUEEZE_WARPS_PER_BLOCK,
                min(m_cfg.lwca->getMaxThreads(), m_maxWideNodes * 32), p);
        }
    }
}

//------------------------------------------------------------------------

void WideBvhPlanner::execHost(void)
{
    m_cfg.outNumWideNodes   .writeDiscardHost();
    m_cfg.outNumRoots       .writeDiscardHost();
    m_cfg.outSubtreeRoots   .writeDiscardHost();
    m_cfg.ioBinaryNodes     .readWriteHost();
    m_cfg.inApexPointMap    .readHost();
    m_cfg.inPrimRange       .readHost();
    m_tempA                 .writeDiscardHost();
    m_squeezeTasks          .writeDiscardHost();

    initialPlanningPass();
    squeezingPass();
}

//------------------------------------------------------------------------

void WideBvhPlanner::initialPlanningPass(void)
{
    // List leaves of the binary tree.

    int numPrims = m_cfg.inPrimRange->span();
    int maxBinaryNodes = (int)m_cfg.ioBinaryNodes.getNumElems();
    std::vector<int> nodeOrder;
    nodeOrder.reserve(maxBinaryNodes);

    if (numPrims <= 2) // 2 primitives or less => nodes 0 and 1 are always valid leaves
    {
        nodeOrder.push_back(0);
        nodeOrder.push_back(1);
    }
    else // otherwise => check each node if it's marked as a valid leaf
    {
        for (int i = 0; i < numPrims * 2; i++)
        {
            int childPairIdx = m_cfg.ioBinaryNodes[i].childPairIdx;
            if (childPairIdx < 0 && childPairIdx != INT_MIN)
                nodeOrder.push_back(i);
        }
    }

    // List internal nodes of the binary tree in bottom-up order.
    {
        std::vector<unsigned char> nodeVisited(maxBinaryNodes + 1, 0);
        for (int i = 0; i < (int)nodeOrder.size(); i++)
        {
            int nodeIdx = nodeOrder[i];
            if (nodeIdx == -1)
                continue;

            int parentIdx = m_cfg.ioBinaryNodes[nodeIdx].beforePlan.parentIdx;
            unsigned char& parentVisited = nodeVisited[parentIdx + 1];
            parentVisited++;

            RT_ASSERT(parentVisited == 1 || parentVisited == 2);
            if (parentVisited == 2)
                nodeOrder.push_back(parentIdx);
        }
    }

    // Initialize temporary arrays.

    std::vector<int>    subtreeWideNodes        (maxBinaryNodes, 0);
    std::vector<float>  subtreeCostRoots        (maxBinaryNodes, 0.0f);
    std::vector<int>    subtreeWideNodesRoots   (maxBinaryNodes, 0);

    memset(m_cfg.outSubtreeRoots.getLwrPtr(), 0, m_cfg.outSubtreeRoots.getNumBytes());

    // Callwlate normalized model AABB area.

    AABB modelAABB = m_cfg.inApexPointMap->getAABB();
    float sizeMaxRcp = modelAABB.getSizeMaxRcp();
    float modelHalfArea = modelAABB.getHalfArea() * (sizeMaxRcp * sizeMaxRcp);

    // Process binary nodes in bottom-up order.

    for (int orderIdx = 0; orderIdx < (int)nodeOrder.size(); orderIdx++)
    {
        // Fetch node and determine surface area.

        PlannerBinaryNode node = {};
        float halfArea = modelHalfArea;

        int nodeIdx = nodeOrder[orderIdx];
        bool nodeIsBVHRoot = (nodeIdx == -1);
        if (!nodeIsBVHRoot)
        {
            node = m_cfg.ioBinaryNodes[nodeIdx];
            halfArea = node.beforePlan.halfArea;
        }

        // Callwlate aggregates.

        if (node.childPairIdx < 0)
        {
            node.afterPlan.subtreePrims = 1; // overwrites parentIdx
            node.beforePlan.primitiveCost = halfArea * m_cfg.sahPrimCost;
        }
        else
        {
            node.afterPlan.subtreePrims = 0; // overwrites parentIdx
            node.beforePlan.primitiveCost = 0.0f;
            int squeezeTag = 0;

            for (int c = 0; c < 2; c++)
            {
                const PlannerBinaryNode& child = m_cfg.ioBinaryNodes[node.childPairIdx + c];
                node.afterPlan.subtreePrims += child.afterPlan.subtreePrims & INT_MAX;
                node.beforePlan.primitiveCost += child.beforePlan.primitiveCost;
                int childTag = (child.childPairIdx < 0) ? 1 : m_tempA[child.childPairIdx >> 1].squeezeTag;
                squeezeTag += (childTag == -1) ? m_cfg.maxLeafSize : childTag;
            }

            // Squeezed wide node has become large enough => mark as complete.

            if (squeezeTag * 2 >= m_cfg.maxBranchingFactor * m_cfg.maxLeafSize + 2)
                squeezeTag = -1;
            m_tempA[node.childPairIdx >> 1].squeezeTag = squeezeTag;
        }

        // Large subtree => optimize wide node shape.
        // Small subtree => must become a single wide node whose children contain one primitive each.

        int totalWideNodes;
        if (node.afterPlan.subtreePrims > m_cfg.maxBranchingFactor)
            chooseWideNodeShape(node, m_cfg.ioBinaryNodes, totalWideNodes, subtreeWideNodes);
        else
        {
            node.afterPlan.subtreeCost = node.beforePlan.primitiveCost; // overwrites halfArea
            totalWideNodes = 1;
            if (node.childPairIdx >= 0)
                for (int c = 0; c < 2; c++)
                    m_cfg.ioBinaryNodes[node.childPairIdx + c].afterPlan.ownerSubtreePrims = 0; // not owned by anyone yet, overwrites primitiveCost
        }

        // Callwlate SAH cost assuming that we create a wide node.

        node.afterPlan.subtreeCost += halfArea * m_cfg.sahNodeCost;

        // Node is large enough => decide whether to create one root at the node itself, or multiple roots at its ancestors.

        if (nodeIsBVHRoot || (m_cfg.allowMultipleRoots && halfArea >= modelHalfArea * m_cfg.minAreaPerRoot && node.childPairIdx >= 0))
        {
            // One root?

            float tmpCost = node.afterPlan.subtreeCost + halfArea * m_cfg.sahRootCost;
            int tmpRoots = 1;
            int totalWideNodesRoots = totalWideNodes;

            // Multiple roots?

            if (m_cfg.allowMultipleRoots && m_cfg.outSubtreeRoots[node.childPairIdx + 0] && m_cfg.outSubtreeRoots[node.childPairIdx + 1])
            {
                float costSplit = subtreeCostRoots[node.childPairIdx + 0] + subtreeCostRoots[node.childPairIdx + 1] + halfArea * m_cfg.sahTopLevelCost;
                if (costSplit < tmpCost)
                {
                    tmpCost = costSplit;
                    tmpRoots = m_cfg.outSubtreeRoots[node.childPairIdx + 0] + m_cfg.outSubtreeRoots[node.childPairIdx + 1];
                    totalWideNodesRoots = subtreeWideNodesRoots[node.childPairIdx + 0] + subtreeWideNodesRoots[node.childPairIdx + 1];
                }
            }

            // Store results.

            if (nodeIsBVHRoot)
            {
                *m_cfg.outNumWideNodes = totalWideNodesRoots;
                if (m_cfg.allowMultipleRoots)
                    *m_cfg.outNumRoots = tmpRoots;
            }
            else
            {
                subtreeCostRoots[nodeIdx] = tmpCost;
                subtreeWideNodesRoots[nodeIdx] = totalWideNodesRoots;
                if (m_cfg.allowMultipleRoots)
                    m_cfg.outSubtreeRoots[nodeIdx] = tmpRoots;
            }
        }

        // Decide whether to collapse the subtree into a leaf node, and store results.

        if (!nodeIsBVHRoot)
        {
            float leafCost = halfArea * (float)node.afterPlan.subtreePrims * m_cfg.sahPrimCost;
            if (node.childPairIdx < 0 || (node.afterPlan.subtreePrims <= m_cfg.maxLeafSize && leafCost <= node.afterPlan.subtreeCost))
            {
                node.afterPlan.subtreePrims |= 1 << 31;
                node.afterPlan.subtreeCost = leafCost;
                totalWideNodes = 0;
            }

            m_cfg.ioBinaryNodes[nodeIdx] = node;
            subtreeWideNodes[nodeIdx] = totalWideNodes;
            if (node.childPairIdx >= 0)
                m_tempA[node.childPairIdx >> 1].subtreeWideNodes = totalWideNodes; // for the squeezing pass
        }
    }
}

//------------------------------------------------------------------------

void WideBvhPlanner::chooseWideNodeShape(
    PlannerBinaryNode&                  root,               // in/out
    const BufferRef<PlannerBinaryNode>& binaryNodes,        // in/out
    int&                                totalWideNodes,     // out
    const std::vector<int>&             subtreeWideNodes)
{
    // Initialize the set of internal nodes for this wide node to the union of the
    // internal nodes of the hypothetical wide nodes rooted by the immediate child nodes.

    m_innerNodes.clear();
    m_innerDepth.clear();

    for (int subtreeIdx = 0; subtreeIdx < 2; subtreeIdx++)
    {
        int subtreeRootIdx = root.childPairIdx + subtreeIdx;
        if (binaryNodes[subtreeRootIdx].childPairIdx >= 0)
        {
            int internalNodeTag = binaryNodes[subtreeRootIdx].afterPlan.subtreePrims & INT_MAX;
            if (internalNodeTag <= m_cfg.maxBranchingFactor)
                internalNodeTag = 0; // extend all the way

            m_innerNodes.push_back(subtreeRootIdx);
            m_innerDepth.push_back(1);

            for (int i = (int)m_innerNodes.size() - 1; i < (int)m_innerNodes.size(); i++)
            for (int c = 0; c < 2; c++)
            {
                int childIdx = binaryNodes[m_innerNodes[i]].childPairIdx + c;
                if (binaryNodes[childIdx].childPairIdx >= 0 && binaryNodes[childIdx].afterPlan.ownerSubtreePrims >= internalNodeTag && m_innerDepth[i] < PLANNER_EXEC_MAX_DEPTH - 1)
                {
                    m_innerNodes.push_back(childIdx);
                    m_innerDepth.push_back(m_innerDepth[i] + 1);
                }
            }
        }
    }

    // Callwlate total cost of the wide node children and record deltas for internal nodes.

    totalWideNodes = subtreeWideNodes[root.childPairIdx + 0] + subtreeWideNodes[root.childPairIdx + 1] + 1;
    root.afterPlan.subtreeCost = binaryNodes[root.childPairIdx + 0].afterPlan.subtreeCost + binaryNodes[root.childPairIdx + 1].afterPlan.subtreeCost; // overwrites halfArea
    m_deltaWideNodes.clear();
    m_deltaCost.clear();

    for (int i = 0; i < (int)m_innerNodes.size(); i++)
    {
        int nodeIdx = m_innerNodes[i];
        int pairIdx = binaryNodes[nodeIdx].childPairIdx;
        m_deltaWideNodes.push_back(subtreeWideNodes[nodeIdx] - subtreeWideNodes[pairIdx + 0] - subtreeWideNodes[pairIdx + 1]);
        m_deltaCost.push_back(binaryNodes[nodeIdx].afterPlan.subtreeCost - binaryNodes[pairIdx + 0].afterPlan.subtreeCost - binaryNodes[pairIdx + 1].afterPlan.subtreeCost);
        totalWideNodes -= m_deltaWideNodes.back();
        root.afterPlan.subtreeCost -= m_deltaCost.back();
    }

    // Remove internal nodes until the branching factor is small enough.
    // Note: Reinterpreting the floats as unsigned integers ensures that
    // the comparison stays robust even if the input contains NaNs/Infs.

    while ((int)m_innerNodes.size() > m_cfg.maxBranchingFactor - 2)
    {
        unsigned int bestDeltaCostU32 = ~0u;
        int bestInner = -1;

        for (int i = 0; i < (int)m_innerNodes.size(); i++)
        {
            int pair = binaryNodes[m_innerNodes[i]].childPairIdx;
            unsigned int deltaCostU32 = __float_as_int(fmaxf(m_deltaCost[i], 0.0f));

            if (deltaCostU32 <= bestDeltaCostU32 &&
                std::find(m_innerNodes.begin(), m_innerNodes.end(), pair + 0) == m_innerNodes.end() &&
                std::find(m_innerNodes.begin(), m_innerNodes.end(), pair + 1) == m_innerNodes.end())
            {
                bestDeltaCostU32 = deltaCostU32;
                bestInner = i;
            }
        }

        RT_ASSERT(bestInner != -1);
        m_innerNodes.erase(m_innerNodes.begin() + bestInner);
        totalWideNodes += m_deltaWideNodes[bestInner];
        root.afterPlan.subtreeCost += m_deltaCost[bestInner];
        m_deltaWideNodes.erase(m_deltaWideNodes.begin() + bestInner);
        m_deltaCost.erase(m_deltaCost.begin() + bestInner);
    }

    // Tag the final internal nodes.

    for (int c = 0; c < 2; c++)
        binaryNodes[root.childPairIdx + c].afterPlan.ownerSubtreePrims = 0; // immediate child => not owned by anyone yet, overwrites primitiveCost

    for (int i = 0; i < (int)m_innerNodes.size(); i++)
        binaryNodes[m_innerNodes[i]].afterPlan.ownerSubtreePrims = root.afterPlan.subtreePrims & INT_MAX; // overwrites primitiveCost
}

//------------------------------------------------------------------------

void WideBvhPlanner::squeezingPass(void)
{
    // Budget not exceeded => nothing to do.

    if (*m_cfg.outNumWideNodes <= m_maxWideNodes)
        return;

    // Setup squeeze task for the root.

    m_squeezeTasks[0].subtreeRootPair = 0;
    m_squeezeTasks[0].wideNodeBudget = m_maxWideNodes;
    int numSqueezeTasks = 1;

    *m_cfg.outNumWideNodes = 0; // start counting from scratch
    if (m_cfg.allowMultipleRoots)
        *m_cfg.outNumRoots = 1; // do not generate multiple roots when squeezing

    // Initialize temporary arrays for squeeze tasks.

    std::vector<int> stack;             // nodePairIdx
    std::vector<int> inners;            // nodePairIdx
    std::vector<int> prims;             // ~primitiveIdx
    std::vector<int> childPairIdx;      // nodePairIdx or ~primitiveIdx
    std::vector<int> childSubtreePrims; // Same semantics as PlannerBinaryNode.afterPlan.subtreePrims.
    std::vector<int> childBudget;       // Current budget for each child.
    std::vector<int> childBudgetOrig;   // Original budget before squeezing.
    std::vector<int> budgetSum;         // Temporary prefix sum for budget distribution.

    // Execute squeeze tasks.

    for (int taskIdx = 0; taskIdx < numSqueezeTasks; taskIdx++)
    {
        const PlannerSqueezeTask& task = m_squeezeTasks[taskIdx];
        (*m_cfg.outNumWideNodes)++;

        // Callwlate total size of the subtree.

        int totalSubtreePrims = m_cfg.ioBinaryNodes[task.subtreeRootPair + 0].afterPlan.subtreePrims;
        totalSubtreePrims += m_cfg.ioBinaryNodes[task.subtreeRootPair + 1].afterPlan.subtreePrims;
        totalSubtreePrims &= INT_MAX;

        // Fits trivially within one wide node => nothing to do.

        if (totalSubtreePrims <= m_cfg.maxBranchingFactor)
            continue;

        // Clear temporary arrays.

        stack.clear();
        inners.clear();
        prims.clear();
        childPairIdx.clear();
        childSubtreePrims.clear();

        // Traverse subtree to find the original planned children of this wide node.
        // The logic here is the same as in XxxConstructor.

        stack.push_back(task.subtreeRootPair);
        while (stack.size())
        {
            int pairIdx = stack.back();
            stack.pop_back();

            for (int c = 0; c < 2; c++)
            {
                const PlannerBinaryNode& node = m_cfg.ioBinaryNodes[pairIdx + c];
                if (node.childPairIdx >= 0 && node.afterPlan.subtreePrims >= 0)
                {
                    if (node.afterPlan.ownerSubtreePrims >= totalSubtreePrims)
                        stack.push_back(node.childPairIdx);
                    else
                    {
                        childPairIdx.push_back(node.childPairIdx);
                        childSubtreePrims.push_back(node.afterPlan.subtreePrims);
                    }
                }
            }
        }

        // Callwlate budget for the children.

        childBudget.resize((int)childPairIdx.size());
        childBudgetOrig.resize((int)childPairIdx.size());
        for (int i = 0; i < (int)childPairIdx.size(); i++)
        {
            childBudgetOrig[i] = m_tempA[childPairIdx[i] >> 1].subtreeWideNodes;
            childBudget[i] = min(childBudgetOrig[i], minWideNodesAfterSqueeze(childSubtreePrims[i], m_cfg.maxBranchingFactor, m_cfg.maxLeafSize));
        }

        // Total budget exceeded => need to squeeze the wide node.

        int totalBudget = 0;
        for (int i = 0; i < (int)childPairIdx.size(); i++)
            totalBudget += childBudget[i];
        bool needToSqueeze = (totalBudget + 1 > task.wideNodeBudget);

        if (needToSqueeze)
        {
            // Find new children and primitives for the wide node.

            childPairIdx.clear();
            childSubtreePrims.clear();
            stack.push_back(task.subtreeRootPair);

            while (stack.size())
            {
                int pairIdx = stack.back();
                stack.pop_back();
                inners.push_back(pairIdx);

                for (int c = 0; c < 2; c++)
                {
                    const PlannerBinaryNode& node = m_cfg.ioBinaryNodes[pairIdx + c];
                    if (node.childPairIdx < 0)
                    {
                        prims.push_back(node.childPairIdx);
                    }
                    else if (m_tempA[node.childPairIdx >> 1].squeezeTag == -1)
                    {
                        childPairIdx.push_back(node.childPairIdx);
                        childSubtreePrims.push_back(node.afterPlan.subtreePrims & INT_MAX);
                    }
                    else
                    {
                        stack.push_back(node.childPairIdx);
                    }
                }
            }

            // Recallwlate budget for the children.

            childBudget.resize((int)childPairIdx.size());
            childBudgetOrig.resize((int)childPairIdx.size());
            for (int i = 0; i < (int)childPairIdx.size(); i++)
            {
                childBudgetOrig[i] = m_tempA[childPairIdx[i] >> 1].subtreeWideNodes;
                childBudget[i] = min(childBudgetOrig[i], minWideNodesAfterSqueeze(childSubtreePrims[i], m_cfg.maxBranchingFactor, m_cfg.maxLeafSize));
            }
        }

        // Distribute excess budget among the children, proportional to their original budget.

        totalBudget = 0;
        for (int i = 0; i < (int)childPairIdx.size(); i++)
            totalBudget += childBudget[i];
        int excessBudget = task.wideNodeBudget - totalBudget - 1;

        if (excessBudget < 0)
        {
            for (int i = 0; i < (int)childPairIdx.size(); i++)
                childBudget[i] = 0; // no budget left => maximal squeeze
        }
        else
        {
            budgetSum.resize((int)childPairIdx.size());
            for (int i = 0; i < (int)childPairIdx.size(); i++)
                budgetSum[i] = ((i) ? budgetSum[i - 1] : 0) + childBudgetOrig[i] - childBudget[i];
            int totalBudgetSum = max(budgetSum[(int)childPairIdx.size() - 1], 1);

            for (int i = 0; i < (int)childPairIdx.size(); i++)
            {
                long long tmpA = (long long)excessBudget * ((i) ? budgetSum[i - 1] : 0) / totalBudgetSum;
                long long tmpB = (long long)excessBudget * budgetSum[i] / totalBudgetSum;
                childBudget[i] += (int)tmpB - (int)tmpA;
            }
        }

        // Create tasks for child subtrees that still require squeezing.

        for (int i = 0; i < (int)childPairIdx.size(); i++)
        {
            if (childBudget[i] >= childBudgetOrig[i] || childSubtreePrims[i] <= m_cfg.maxBranchingFactor)
                *m_cfg.outNumWideNodes += childBudgetOrig[i];
            else
            {
                m_squeezeTasks[numSqueezeTasks].subtreeRootPair = childPairIdx[i];
                m_squeezeTasks[numSqueezeTasks].wideNodeBudget = childBudget[i];
                numSqueezeTasks++;
            }
        }

        // Did not squeeze => task done.

        if (!needToSqueeze)
            continue;

        // Distribute primitives into leaves.

        int numLeaves = min((int)prims.size(), m_cfg.maxBranchingFactor - (int)childPairIdx.size());
        for (int leafIdx = 0; leafIdx < numLeaves; leafIdx++)
        {
            int primBase = leafIdx * (int)prims.size() / numLeaves;
            int leafSize = (leafIdx + 1) * (int)prims.size() / numLeaves - primBase;

            // Create a subtree to contain the primitives.

            for (int i = 1; i < leafSize; i++)
            {
                int innerPairIdx = inners.back();
                inners.pop_back();

                PlannerBinaryNode& nodeA = m_cfg.ioBinaryNodes[innerPairIdx + 0];
                nodeA.childPairIdx = prims[primBase];
                nodeA.afterPlan.subtreePrims = i + (1 << 31); // leaf
                nodeA.afterPlan.ownerSubtreePrims = 0; // not inner

                PlannerBinaryNode& nodeB = m_cfg.ioBinaryNodes[innerPairIdx + 1];
                nodeB.childPairIdx = prims[primBase + i];
                nodeB.afterPlan.subtreePrims = 1 + (1 << 31); // leaf
                nodeB.afterPlan.ownerSubtreePrims = 0; // not inner

                prims[primBase] = innerPairIdx;
            }

            // Append to the list of children.

            childPairIdx.push_back(prims[primBase]);
            childSubtreePrims.push_back(leafSize + (1 << 31));
        }

        // Create a balanced binary tree out of the remaining internal nodes.

        for (int i = 0; i < (int)childPairIdx.size() * 2 - 2; i++)
        {
            PlannerBinaryNode& node = m_cfg.ioBinaryNodes[inners[i >> 1] + (i & 1)];
            int childIdx = i - ((int)childPairIdx.size() - 2);
            if (childIdx < 0)
            {
                node.childPairIdx = inners[i + 1];
                node.afterPlan.subtreePrims = INT_MAX / 2; // not leaf, not trivially small
                node.afterPlan.ownerSubtreePrims = INT_MAX; // inner
            }
            else
            {
                node.childPairIdx = childPairIdx[childIdx];
                node.afterPlan.subtreePrims = childSubtreePrims[childIdx];
                node.afterPlan.ownerSubtreePrims = 0; // not inner
            }
        }
    }

    // Must not exceed budget after squeezing.

    RT_ASSERT(*m_cfg.outNumWideNodes <= m_maxWideNodes);
}

//------------------------------------------------------------------------

int WideBvhPlanner::callwlateMaxWideNodes(int maxPrims) const
{
    // All primitives fit trivially within a single wide node => the planning phase is guaranteed to do this.

    if (maxPrims <= m_cfg.maxBranchingFactor)
        return 1;

    // Callwlate the minimum number of wide nodes that the squeeze pass is guaranteed to achieve.
    // SqueezeMode_MaxSqueeze => use the result directly as maxWideNodes.

    int lowerBound = minWideNodesAfterSqueeze(maxPrims, m_cfg.maxBranchingFactor, m_cfg.maxLeafSize);
    if (m_cfg.squeezeMode == SqueezeMode_MaxSqueeze)
        return lowerBound;

    // Callwlate the maximum number of wide nodes that the planning pass is guaranteed to never exceed.
    // SqueezeMode_NoSqueeze => use the result directly as maxWideNodes.

    int upperBound = max(maxPrims - 1, 1);
    if (m_cfg.squeezeMode == SqueezeMode_NoSqueeze)
        return upperBound;

    // The following formulas represent an empirical model of the expected maximum number
    // of wide nodes as a function of the relevant config parameters. The model is based on
    // a tens of thousands of measurements over several test scenes and parameter combinations.
    // For any parameter combination, the goal is to yield the lowest possible value for
    // maxWideNodes that is not exceeded by any of the measurements.
    //
    // Source code for performing the measurements and fitting the model can be found here:
    // //research/ttu/common/primetest/src/WideBvhSqueezeTest.cpp
    // //research/ttu/common/primetest/fit/fit.py

    RT_ASSERT(m_cfg.squeezeMode == SqueezeMode_Default);

    // The only relevant effect of sahPrimCost and sahNodeCost appears to be that they
    // effectively induce an upper bound for maxLeafSize.

    float relPrimCost = fminf(fmaxf(m_cfg.sahPrimCost / fmaxf(m_cfg.sahNodeCost, 1.0e-8f), 0.0f), 2.0f);
    float inducedMaxLeafSize = expf(relPrimCost * -7.19344f) * 191.891f + 1.21792f;
    float L = fmaxf(fminf((float)m_cfg.maxLeafSize, inducedMaxLeafSize), 1.0f);

    // It turns out we can aclwrately model the expected ratio of wide nodes to primitives
    // as a second order rational of maxBranchingFactor and maxLeafSize.

    float B = fminf(fmaxf((float)m_cfg.maxBranchingFactor, 1.0f), 16.0f);
    float num = B * (B * 0.0102961f + L * 0.050588f + -0.101521f) + L * (L * -0.0221323f + 0.432227f);
    float den = B * (B * 0.0285739f + L * 0.700374f + -0.696792f) + L * (L * -0.0203872f + -0.757004f) + 1.0f;
    float expectedMaxWideNodesPerPrim = num / den;

    // The value of maxWideNodes behaves linearly with respect to the number of primitives,
    // but it should not exceed the lower/upper bounds. The constant term (16.0f) is necessary
    // to guarantee conservativeness even with very small primitive counts.

    int expectedMaxWideNodes = (int)((float)maxPrims * expectedMaxWideNodesPerPrim + 16.0f);
    return min(max(expectedMaxWideNodes, lowerBound), upperBound);
}

//------------------------------------------------------------------------
