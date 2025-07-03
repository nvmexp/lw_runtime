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

#include "AacBuilder.hpp"
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>
#include <algorithm>

using namespace prodlib::bvhtools;
using namespace prodlib;

//------------------------------------------------------------------------

void AacBuilder::configure(const Config& cfg)
{
    // Check for errors.

    RT_ASSERT(cfg.maxPrims >= 0);
    RT_ASSERT(cfg.inPrimOrder.getNumElems() >= (size_t)cfg.maxPrims);
    RT_ASSERT(cfg.inMortonCodes.getNumElems() >= (size_t)cfg.maxPrims);
    RT_ASSERT(cfg.inPrimRange.getNumElems() == 1);
    RT_ASSERT(cfg.inModel.isValid());
    RT_ASSERT(cfg.inApexPointMap.getNumBytes() >= sizeof(ApexPointMap));

    if (cfg.bytesPerMortonCode != 4 && cfg.bytesPerMortonCode != 8)
        throw IlwalidValue(RT_EXCEPTION_INFO, "bytesPerMortonCode must be 4 or 8!", cfg.bytesPerMortonCode);

    if (cfg.aacMaxClusterSize != 4 && cfg.aacMaxClusterSize != 8 && cfg.aacMaxClusterSize != 16)
        throw IlwalidValue(RT_EXCEPTION_INFO, "aacMaxClusterSize must be 4, 8, or 16!", cfg.aacMaxClusterSize);

    if (!(cfg.aacPenaltyTermCoef >= 0.0f))
        throw IlwalidValue(RT_EXCEPTION_INFO, "aacPenaltyTermCoef must be non-negative!", cfg.aacPenaltyTermCoef);

    // Set config and resize outputs.

    m_cfg = cfg;
    m_cfg.outNodes.setNumElems(std::max(cfg.maxPrims * 2, 2));

    // Layout temp buffers.

    m_workCounter   .assignNew(1);
    m_clusterIndices.assignNew(m_cfg.maxPrims);
    m_clusterEntries = m_cfg.outNodes.reinterpret<AacClusterEntry>(); // overlay on top of outNodes
    m_nodePrimRanges.assignNew(m_cfg.maxPrims);

    RT_ASSERT(sizeof(AacClusterEntry) == sizeof(PlannerBinaryNode) * 2);
    RT_ASSERT(m_clusterEntries.getNumElems() >= (size_t)cfg.maxPrims);

    m_cfg.tempBuffer
        .aggregate(m_workCounter)
        .aggregate(m_clusterIndices)
        .aggregate(m_nodePrimRanges);
}

//------------------------------------------------------------------------

void AacBuilder::execute(void)
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

void AacBuilder::execDevice(void)
{
    // Clear buffers.

    m_clusterIndices.clearLWDA((unsigned char)-1);
    m_nodePrimRanges.clearLWDA((unsigned char)-1);
    m_workCounter.clearLWDA(0);

    // Launch AacExec.
    {
        AacExecParams p     = {};
        p.outNodes          = m_cfg.outNodes.writeDiscardLWDA();
        p.clusterIndices    = m_clusterIndices.writeDiscardLWDA();
        p.clusterEntries    = m_clusterEntries.writeDiscardLWDA();
        p.nodePrimRanges    = m_nodePrimRanges.readWriteLWDA();
        p.workCounter       = m_workCounter.readWriteLWDA();
        p.inPrimOrder       = m_cfg.inPrimOrder.readLWDA();
        p.inMortonCodes     = m_cfg.inMortonCodes.readLWDA();
        p.inModel           = ModelPointers(m_cfg.inModel, MemorySpace_LWDA);
        p.inApexPointMap    = m_cfg.inApexPointMap.readLWDA();
        p.inPrimRange       = m_cfg.inPrimRange.readLWDA();
        p.maxPrims          = m_cfg.maxPrims;
        p.penaltyTermCoef   = m_cfg.aacPenaltyTermCoef;

        LAUNCH(*m_cfg.lwca, AacExec, AAC_EXEC_WARPS_PER_BLOCK,
            min(m_cfg.lwca->getMaxThreads(), max(m_cfg.maxPrims, 1)),
            p, m_cfg.bytesPerMortonCode, m_cfg.aacMaxClusterSize);
    }
}

//------------------------------------------------------------------------

void AacBuilder::execHost(void)
{
    m_cfg.outNodes          .writeDiscardHost();
    m_cfg.inPrimOrder       .readHost();
    m_cfg.inMortonCodes     .readHost();
    ModelPointers inModel   (m_cfg.inModel, MemorySpace_Host);
    m_cfg.inApexPointMap    .readHost();
    m_cfg.inPrimRange       .readHost();
    m_clusterIndices        .writeDiscardHost();
    m_clusterEntries        .writeDiscardHost();
    m_nodePrimRanges        .writeDiscardHost();

    // Less than 2 primitives => special case.

    RT_ASSERT(m_cfg.inPrimRange->span() <= m_cfg.maxPrims);
    if (m_cfg.inPrimRange->span() < 2)
    {
        for (int i = 0; i < 2; i++)
        {
            PlannerBinaryNode& n       = m_cfg.outNodes[i];
            n.childPairIdx             = INT_MIN;
            n.beforePlan.halfArea      = 0.0f;
            n.beforePlan.parentIdx     = -1;
            n.beforePlan.primitiveCost = 0.0f;

            if (i < m_cfg.inPrimRange->span())
                n.childPairIdx = ~m_cfg.inPrimOrder[i + m_cfg.inPrimRange->start];
        }
        return;
    }

    // Initialize.

    AABB    modelAABB       = m_cfg.inApexPointMap->getAABB();
    float3  modelCenter     = modelAABB.getCenter();
    float   modelSizeMaxRcp = modelAABB.getSizeMaxRcp();

    memset(m_nodePrimRanges.getLwrPtr(), -1, m_nodePrimRanges.getNumBytes());

    // Process each primitive.

    for (int sortedPrimIdx = 0; sortedPrimIdx < m_cfg.inPrimRange->span(); sortedPrimIdx++)
    {
        // Initialize leaf cluster entry for the primitive.
        {
            int primitiveIdx = m_cfg.inPrimOrder[sortedPrimIdx + m_cfg.inPrimRange->start];
            AABB primAABB = inModel.loadPrimitiveAABB(primitiveIdx);

            m_clusterIndices[sortedPrimIdx] = sortedPrimIdx;
            AacClusterEntry& e = m_clusterEntries[sortedPrimIdx];
            e.childPairIdx  = ~primitiveIdx;
            e.subtreePrims  = 1;
            e.aabb.lo       = AABB::transformRelative(primAABB.lo, modelCenter, modelSizeMaxRcp);
            e.aabb.hi       = AABB::transformRelative(primAABB.hi, modelCenter, modelSizeMaxRcp);

            // Clean up NaN/Inf to avoid issues with the CPU implementation of evaluateMergeCost() not being strictly symmetric.

            const float maxFloat = LWDART_MAX_NORMAL_F;
            e.aabb.lo.x = std::min(std::max(e.aabb.lo.x, -maxFloat), +maxFloat);
            e.aabb.lo.y = std::min(std::max(e.aabb.lo.y, -maxFloat), +maxFloat);
            e.aabb.lo.z = std::min(std::max(e.aabb.lo.z, -maxFloat), +maxFloat);
            e.aabb.hi.x = std::min(std::max(e.aabb.hi.x, -maxFloat), +maxFloat);
            e.aabb.hi.y = std::min(std::max(e.aabb.hi.y, -maxFloat), +maxFloat);
            e.aabb.hi.z = std::min(std::max(e.aabb.hi.z, -maxFloat), +maxFloat);
        }

        // Walk up the nodes of an implicit LBVH.

        int2 nodePrimRange = make_int2(sortedPrimIdx, sortedPrimIdx + 1); // node covers range [x,y[
        int mergeSplitIdx = -1;
        for (;;)
        {
            // Compare Morton codes against their neighbors at each end of current range.

            unsigned long long diffLo, diffHi;
            if (m_cfg.bytesPerMortonCode == 4)
            {
                BufferRef<const unsigned int> morton = m_cfg.inMortonCodes.reinterpret<const unsigned int>().getSubrange(m_cfg.inPrimRange->start);
                diffLo = (nodePrimRange.x > 0)                         ? (morton[nodePrimRange.x] ^ morton[nodePrimRange.x - 1]) : ~1ull;
                diffHi = (nodePrimRange.y < m_cfg.inPrimRange->span()) ? (morton[nodePrimRange.y] ^ morton[nodePrimRange.y - 1]) : ~0ull;
            }
            else
            {
                BufferRef<const unsigned long long> morton = m_cfg.inMortonCodes.reinterpret<const unsigned long long>().getSubrange(m_cfg.inPrimRange->start);
                diffLo = (nodePrimRange.x > 0)                         ? (morton[nodePrimRange.x] ^ morton[nodePrimRange.x - 1]) : ~1ull;
                diffHi = (nodePrimRange.y < m_cfg.inPrimRange->span()) ? (morton[nodePrimRange.y] ^ morton[nodePrimRange.y - 1]) : ~0ull;
            }

            // All Morton codes are identical => tie-break using leaf indices.

            if (diffLo == diffHi)
            {
                diffLo = nodePrimRange.x ^ (nodePrimRange.x - 1);
                diffHi = nodePrimRange.y ^ (nodePrimRange.y - 1);
            }

            // The parent node inherits nodePrimRange.x from its left child and nodePrimRange.y from its right child.
            // Swap the components temporarily so that nodePrimRange.x will come from the current node and nodePrimRange.y from its sibling.

            if (diffLo < diffHi)
                nodePrimRange = make_int2(nodePrimRange.y, nodePrimRange.x);

            // Determine mergeSplitIdx and (swapped) primRange of the parent node.
            // The sibling node is not ready yet => leave this branch for now.

            mergeSplitIdx = nodePrimRange.y;
            nodePrimRange.y = m_nodePrimRanges[mergeSplitIdx];
            m_nodePrimRanges[mergeSplitIdx] = nodePrimRange.x;
            if (nodePrimRange.y == -1)
                break;

            // Undo the swapping.

            if (diffLo < diffHi)
                nodePrimRange = make_int2(nodePrimRange.y, nodePrimRange.x);

            // Determine cluster size limit.

            int clusterLimit = m_cfg.aacMaxClusterSize;
            if (nodePrimRange.x == 0 && nodePrimRange.y == m_cfg.inPrimRange->span())
                clusterLimit = 1;

            // Cluster size exceeds limit => merge entries.

            if (nodePrimRange.y - nodePrimRange.x > clusterLimit)
                mergeClusters(nodePrimRange.x, mergeSplitIdx, nodePrimRange.y, clusterLimit);
        }
    }
}

//------------------------------------------------------------------------
// Left subtree covers [leafA, leafB[, right subtree covers [leafB, leafC[

void AacBuilder::mergeClusters(int leafA, int leafB, int leafC, int clusterLimit)
{
    // Collect cluster indices from the left and right LBVH subtree.

    m_tmpIndices.clear();

    int numAB = min(leafB - leafA, m_cfg.aacMaxClusterSize);
    for (int i = 0; i < numAB; i++)
        m_tmpIndices.push_back(leafA + i);

    int numBC = min(leafC - leafB, m_cfg.aacMaxClusterSize);
    for (int i = 0; i < numBC; i++)
        m_tmpIndices.push_back(leafB + i);

    // Fetch the corresponding cluster entries.

    m_tmpEntries.clear();
    for (int i = 0; i < (int)m_tmpIndices.size(); i++)
        m_tmpEntries.push_back(m_clusterEntries[m_tmpIndices[i]]);

    // Perform merges until the cluster is small enough.

    while ((int)m_tmpEntries.size() > clusterLimit)
    {
        // For each cluster entry, find the best pair to merge with.
        // Note: Reinterpreting the floats as unsigned integers ensures that
        // the comparison stays robust even if the input contains NaNs/Infs.

        m_tmpPairs.resize(m_tmpEntries.size());
        for (int i = 0; i < (int)m_tmpEntries.size(); i++)
        {
            unsigned int bestCostU32 = ~0u;
            m_tmpPairs[i] = -1;
            for (int j = 0; j < (int)m_tmpEntries.size(); j++)
            {
                if (j != i)
                {
                    float cost = evaluateMergeCost(m_tmpEntries[i], m_tmpEntries[j]);
                    unsigned int costU32 = __float_as_int(fmaxf(cost, 0.0f));
                    bestCostU32 = std::min(bestCostU32, costU32);
                    if (costU32 == bestCostU32)
                        m_tmpPairs[i] = j;
                }
            }
        }

        // Merge entries with their pairs if the bonds are mutual.

        while ((int)m_tmpEntries.size() > clusterLimit)
        {
            // Find a mutual bond.

            int2 pair = make_int2(-1, -1);
            for (int i = 0; i < (int)m_tmpEntries.size(); i++)
            {
                int j = m_tmpPairs[i];
                if (j != -1 && m_tmpPairs[j] == i)
                {
                    pair = make_int2(i, j);
                    break;
                }
            }

            if (pair.x == -1)
                break;

            // Conditionally swap the chosen entries to improve any-hit ray perf.

            if (m_tmpEntries[pair.x].aabb.getHalfArea() < m_tmpEntries[pair.y].aabb.getHalfArea())
                pair = make_int2(pair.y, pair.x);

            // Allocate a pair of output nodes.
            // The nodes are overlaid on top of the cluster entries, so we need to be very careful about indexing.
            // In practice, we grab last unused cluster indices, and repurpose each cluster index to represent two nodes.

            int outPairIdx = m_tmpIndices.back() * 2;
            m_tmpIndices.pop_back();

            if (m_tmpEntries.size() == 2)
                outPairIdx = 0; // root

            // Write out BVH nodes.

            for (int c = 0; c < 2; c++)
            {
                int outNodeIdx = outPairIdx + c;
                const AacClusterEntry& entry = m_tmpEntries[(c == 0) ? pair.x : pair.y];

                PlannerBinaryNode& n       = m_cfg.outNodes[outNodeIdx];
                n.childPairIdx             = entry.childPairIdx;
                n.beforePlan.parentIdx     = -1; // filled by the parent
                n.beforePlan.halfArea      = entry.aabb.getHalfArea();
                n.beforePlan.primitiveCost = 0.0f; // don't care

                // Update parent pointers.

                if (n.childPairIdx >= 0)
                {
                    m_cfg.outNodes[n.childPairIdx + 0].beforePlan.parentIdx = outNodeIdx;
                    m_cfg.outNodes[n.childPairIdx + 1].beforePlan.parentIdx = outNodeIdx;
                }
            }

            // Merge entries.

            m_tmpEntries[pair.x].childPairIdx = outPairIdx;
            m_tmpEntries[pair.x].subtreePrims += m_tmpEntries[pair.y].subtreePrims;
            m_tmpEntries[pair.x].aabb         = AABB(m_tmpEntries[pair.x].aabb, m_tmpEntries[pair.y].aabb);
            m_tmpEntries.erase(m_tmpEntries.begin() + pair.y);

            // Update pairings to reflect the change.

            m_tmpPairs.erase(m_tmpPairs.begin() + pair.y);
            for (int i = 0; i < (int)m_tmpEntries.size(); i++)
            {
                int& j = m_tmpPairs[i];
                j = (j == pair.y) ? -1 : (j < pair.y) ? j : j - 1;
            }
        }
    }

    // BVH root => nodes 2 and 3 are always unused.

    if (clusterLimit == 1)
    {
        m_cfg.outNodes[2].childPairIdx = INT_MIN;
        m_cfg.outNodes[3].childPairIdx = INT_MIN;
    }

    // Otherwise => write out the cluster entries.

    else
    {
        for (int i = 0; i < (int)m_tmpEntries.size(); i++)
        {
            m_clusterIndices[leafA + i] = m_tmpIndices[i];
            m_clusterEntries[m_tmpIndices[i]] = m_tmpEntries[i];
        }
    }
}

//------------------------------------------------------------------------

float AacBuilder::evaluateMergeCost(const AacClusterEntry& a, const AacClusterEntry& b)
{
    // Walter's original cost function.

    float cost = AABB(a.aabb, b.aabb).getHalfArea();

    // Penalty term.

    float penalty = 0.0f;
    for (int c = 0; c < 3; c++)
    {
        float        v0 = chooseComponent(a.aabb.lo, c) + chooseComponent(a.aabb.hi, c);
        float        v1 = chooseComponent(b.aabb.lo, c) + chooseComponent(b.aabb.hi, c);
        unsigned int b0 = (unsigned int)(v0 * (float)(1ull << 31));
        unsigned int b1 = (unsigned int)(v1 * (float)(1ull << 31));
        int          sh = findLeadingOne(b0 ^ b1);
        unsigned int s0 = (unsigned int)(((unsigned long long)b0 << 32) >> sh);
        unsigned int s1 = (unsigned int)(((unsigned long long)b1 << 32) >> sh);
        unsigned int dd = std::min((unsigned int)abs((int)s0), (unsigned int)abs((int)s1));
        penalty        += (float)dd * __int_as_float((32 << 23) + (sh << 24));
    }
    cost += penalty * m_cfg.aacPenaltyTermCoef;
    return cost;
}

//------------------------------------------------------------------------
