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

#include "TrBvhBuilder.hpp"
#include <prodlib/bvhtools/src/common/SharedKernelFunctions.hpp>
#include <prodlib/bvhtools/src/common/Utils.hpp>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/Assert.h>
#include <vector>

using namespace prodlib::bvhtools;
using namespace prodlib;

// filters out NaNs in aabbs, since fmaxf( nan, 0 ) == 0
static __device__ __forceinline__ float     aabbClampedHalfArea(float sizeX, float sizeY, float sizeZ) { return fmaxf(sizeX * sizeY + sizeY * sizeZ + sizeZ * sizeX, 0.0f); }

//------------------------------------------------------------------------

void TrbvhBuilder::configure(const Config& cfg)
{
    // Check for errors.

    RT_ASSERT(cfg.maxPrims >= 0);
    int maxNodes = max(cfg.maxPrims - 1, 1);
    int maxRemaps = max(cfg.maxPrims, 1);

    RT_ASSERT(cfg.ioNodes.getNumElems() >= (size_t)maxNodes);
    RT_ASSERT(cfg.ioNumNodes.getNumElems() == 1);
    RT_ASSERT(cfg.ioRemap.getNumElems() >= (size_t)maxRemaps);
    RT_ASSERT(cfg.ioNumRemaps.getNumElems() == 1);
    RT_ASSERT(cfg.ioMortonCodes.getNumElems() >= (size_t)cfg.maxPrims);
    RT_ASSERT(cfg.inPrimOrder.getNumElems() >= (size_t)cfg.maxPrims);
    RT_ASSERT(cfg.inPrimRange.getNumElems() <= 1);

    if (!(cfg.sahNodeCost >= 0.0f))
      throw IlwalidValue( RT_EXCEPTION_INFO, "sahNodeCost must be non-negative!", cfg.sahNodeCost );

    if (!(cfg.sahPrimCost >= 0.0f))
      throw IlwalidValue( RT_EXCEPTION_INFO, "sahPrimCost must be non-negative!", cfg.sahPrimCost );

    if (cfg.maxLeafSize < 1)
      throw IlwalidValue( RT_EXCEPTION_INFO, "maxLeafSize must be at least one!", cfg.maxLeafSize );

    if (cfg.optTreeletSize < 5 || cfg.optTreeletSize > 8)
      throw IlwalidValue( RT_EXCEPTION_INFO, "optTreeletSize must be between 5 and 8!", cfg.optTreeletSize );

    if (cfg.optRounds < 0)
      throw IlwalidValue( RT_EXCEPTION_INFO, "optRounds must be non-negative!", cfg.optRounds );

    if (cfg.optGamma < 1)
      throw IlwalidValue( RT_EXCEPTION_INFO, "optGamma must be at least one!", cfg.optGamma );

    // Set config.

    m_cfg = cfg;
    m_cfg.optGamma = max(m_cfg.optGamma, m_cfg.optTreeletSize);
    m_cfg.maxLeafSize = min(m_cfg.maxLeafSize, min(TRBVH_COLLAPSE_MAX_LEAF_SIZE, (int)RLLEPACK_LEN_MAX));

    // Resize output buffers.

    if (m_cfg.outNodeParents.getNumElems() < (size_t)maxNodes)
      m_cfg.outNodeParents.setNumElems(maxNodes);
    m_cfg.outNodeRange.setNumElems(1);

    // Layout temp buffers.

    m_lwrRound          .assignNew(1);
    m_workCounter       .assignNew(1);
    m_dummyNodeCosts    .assignNew((m_cfg.maxPrims == 0) ? 1 : 0);

    m_cfg.tempBuffer
        .aggregate(m_lwrRound)
        .aggregate(m_workCounter)
        .aggregate(m_dummyNodeCosts);
}

//------------------------------------------------------------------------

void TrbvhBuilder::execute(void)
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
    m_cfg.ioMortonCodes.markAsUninitialized();
}

//------------------------------------------------------------------------

void TrbvhBuilder::execDevice(void)
{
    BufferRef<TrbvhNode> trbvhNodes = m_cfg.ioNodes.reinterpret<TrbvhNode>();
    BufferRef<TrbvhNodeCosts> nodeCosts = (m_cfg.maxPrims == 0) ? m_dummyNodeCosts : m_cfg.ioMortonCodes.reinterpret<TrbvhNodeCosts>();

    // Clear buffers.

    m_lwrRound.clearLWDA(0);
    m_workCounter.clearLWDA(0);

    // Launch TrbvhRadixTree.
    {
        TrbvhRadixTreeParams p = {};
        p.nodes             = trbvhNodes.writeLWDA();
        p.nodeVisited       = m_cfg.ioRemap.writeLWDA();
        p.nodeParents       = m_cfg.outNodeParents.writeDiscardLWDA();
        p.outNodeRange      = m_cfg.outNodeRange.writeDiscardLWDA();
        p.mortonCodes       = m_cfg.ioMortonCodes.readLWDA();
        p.primRange         = m_cfg.inPrimRange.readLWDA();
        p.nodeRangeStart    = m_cfg.ioNumNodes.readLWDA();
        p.nodeVisitedOfs    = m_cfg.ioNumRemaps.readLWDA();
        p.maxPrims          = m_cfg.maxPrims;

        LAUNCH(*m_cfg.lwca, TrbvhRadixTree, TRBVH_RADIXTREE_WARPS_PER_BLOCK,
            max(m_cfg.maxPrims - 1, 1), p);
    }

    // Launch TrbvhFit.
    {
        TrbvhFitParams p    = {};
        p.nodes             = trbvhNodes.readWriteLWDA();
        p.nodeVisited       = m_cfg.ioRemap.readWriteLWDA();
        p.nodeCosts         = nodeCosts.writeDiscardLWDA();
        p.outNodeRangeEnd   = m_cfg.ioNumNodes.writeDiscardLWDA();
        p.inModel           = ModelPointers(m_cfg.inModel, MemorySpace_LWDA);
        p.sortOrder         = m_cfg.inPrimOrder.readLWDA();
        p.nodeParents       = m_cfg.outNodeParents.readLWDA();
        p.primRange         = m_cfg.inPrimRange.readLWDA();
        p.nodeRange         = m_cfg.outNodeRange.readLWDA();
        p.nodeVisitedOfs    = m_cfg.ioNumRemaps.readLWDA();
        p.maxPrims          = m_cfg.maxPrims;
        p.sahNodeCost       = m_cfg.sahNodeCost;
        p.sahPrimCost       = m_cfg.sahPrimCost;
        p.maxLeafSize       = (float)m_cfg.maxLeafSize;

        LAUNCH(*m_cfg.lwca, TrbvhFit, TRBVH_FIT_WARPS_PER_BLOCK,
            max((m_cfg.maxPrims - 1) * 2, 1), p);
    }

    // Launch TrbvhOpt.
    {
        TrbvhOptParams p    = {};
        p.nodes             = trbvhNodes.readWriteLWDA();
        p.nodeVisited       = m_cfg.ioRemap.readWriteLWDA();
        p.nodeParents       = m_cfg.outNodeParents.readWriteLWDA();
        p.nodeCosts         = nodeCosts.readWriteLWDA();
        p.lwrRound          = m_lwrRound.readWriteLWDA();
        p.workCounter       = m_workCounter.readWriteLWDA();
        p.nodeRange         = m_cfg.outNodeRange.readLWDA();
        p.nodeVisitedOfs    = m_cfg.ioNumRemaps.readLWDA();
        p.maxPrims          = m_cfg.maxPrims;
        p.sahNodeCost       = m_cfg.sahNodeCost;
        p.sahPrimCost       = m_cfg.sahPrimCost;
        p.maxLeafSize       = (float)m_cfg.maxLeafSize;

        for (unsigned i = 0; i < sizeof(p.gamma) / sizeof(p.gamma[0]); i++)
        {
            p.gamma[i] = m_cfg.optGamma;
            if (m_cfg.optAdaptiveGamma)
                p.gamma[i] <<= i;
        }

        for (int i = 0; i < m_cfg.optRounds; i++) 
            LAUNCH(*m_cfg.lwca, TrbvhOpt, TRBVH_OPT_WARPS_PER_BLOCK,
                min(m_cfg.lwca->getMaxThreads(), max(m_cfg.maxPrims - 1, 1)),
                p, m_cfg.optTreeletSize);
    }

    // Launch TrbvhCollapse.
    {
        TrbvhCollapseParams p = {};
        p.nodes         = trbvhNodes.readWriteLWDA();
        p.remap         = m_cfg.ioRemap.writeLWDA();
        p.remapSize     = m_cfg.ioNumRemaps.readWriteLWDA();
        p.nodeParents   = m_cfg.outNodeParents.readLWDA();
        p.nodeCosts     = nodeCosts.readLWDA();
        p.primRange     = m_cfg.inPrimRange.readLWDA();
        p.nodeRange     = m_cfg.outNodeRange.readLWDA();
        p.maxPrims      = m_cfg.maxPrims;
        p.maxLeafSize   = (float)m_cfg.maxLeafSize;
        p.listLenEnc    = m_cfg.listLenEnc;

        LAUNCH(*m_cfg.lwca, TrbvhCollapse, TRBVH_COLLAPSE_WARPS_PER_BLOCK,
            max(m_cfg.maxPrims * 2, 1), p);
    }
}

//------------------------------------------------------------------------

static void computeBinaryRadixTree(
    ModelPointers                       inModel,
    BufferRef<const unsigned long long> mortonCodes,
    BufferRef<const int>                sortOrder,
    const Range&                        primRange,
    BufferRef<int>                      nodeParents,
    BufferRef<BvhNode>                  ioNodes)
{
    nodeParents[0] = INT_MAX;

    // Less than 2 primitives => setup a special root node.

    if (primRange.span() < 2)
    {
        // Left child.

        BvhNode n;
        if (primRange.span())
        {
            const PrimitiveAABB pa = inModel.loadPrimitiveAABB(sortOrder[primRange.start]);
            n.c0idx = ~pa.primitiveIdx;
            n.c0lox = pa.lox, n.c0loy = pa.loy, n.c0loz = pa.loz; // true AABB
            n.c0hix = pa.hix, n.c0hiy = pa.hiy, n.c0hiz = pa.hiz;
        }
        else
        {
            n.c0idx = ~0;
            n.c0lox = n.c0loy = n.c0loz = LWDART_NAN_F; // unhittable AABB
            n.c0hix = n.c0hiy = n.c0hiz = LWDART_NAN_F;
        }

        // Right child.

        n.c1idx = n.c0idx;
        n.c1lox = n.c1loy = n.c1loz = LWDART_NAN_F; // unhittable AABB
        n.c1hix = n.c1hiy = n.c1hiz = LWDART_NAN_F;

        // Output.

        ioNodes[0] = n;
        return;
    }

    // Create each node.

    for (int i = 0; i < primRange.span() - 1; i++)
    {
        int s, j, d;
        computeInterval(i, primRange.span(), mortonCodes.getLwrPtr() + primRange.start, s, d, j);

        // Determine children.
        BvhNode& n = ioNodes[i];
        n.c0idx = i + s * d + min(d, 0);
        n.c1idx = n.c0idx + 1;

        // Internal node => output parent pointer.
        // Leaf node => remap index and fetch AABB.
        if (n.c0idx > min(i, j)) {
            nodeParents[n.c0idx] = i;
        }
        else
        {
            const PrimitiveAABB pa = inModel.loadPrimitiveAABB(sortOrder[n.c0idx + primRange.start]);
            n.c0idx = ~pa.primitiveIdx;
            n.c0lox = pa.lox, n.c0hix = fmaxf(pa.hix, pa.lox);
            n.c0loy = pa.loy, n.c0hiy = fmaxf(pa.hiy, pa.loy);
            n.c0loz = pa.loz, n.c0hiz = fmaxf(pa.hiz, pa.loz);
        }

        if (n.c1idx < max(i, j)) {
            nodeParents[n.c1idx] = ~i; // complement to indicate right child
        }
        else
        {
            const PrimitiveAABB& pa = inModel.loadPrimitiveAABB(sortOrder[n.c1idx + primRange.start]);
            n.c1idx = ~pa.primitiveIdx;
            n.c1lox = pa.lox, n.c1hix = fmaxf(pa.hix, pa.lox);
            n.c1loy = pa.loy, n.c1hiy = fmaxf(pa.hiy, pa.loy);
            n.c1loz = pa.loz, n.c1hiz = fmaxf(pa.hiz, pa.loz);
        }
    }
}

//------------------------------------------------------------------------

static inline void topologicalSort(std::vector<int>& nodeOrder, BufferRef<const BvhNode> nodes)
{
    nodeOrder.clear();
    nodeOrder.push_back(0);
    for (int i = 0; i < (int)nodeOrder.size(); i++)
    {
        const BvhNode& n = nodes[nodeOrder[i]];
        if (n.c0idx >= 0)
            nodeOrder.push_back(n.c0idx);
        if (n.c1idx >= 0)
            nodeOrder.push_back(n.c1idx);
    }
}

//------------------------------------------------------------------------

static void computeAabbsAndCosts(
    std::vector<int>&   nodeOrder,
    BufferRef<BvhNode>  ioNodes,
    BufferRef<int>      nodeParents,
    std::vector<int>&   nodeSubtreeSize,
    std::vector<float>& nodeSubtreeCost,
    float               sahPrimCost,
    float               sahNodeCost,
    int                 maxLeafSize)
{
    for (int orderIdx = (int)nodeOrder.size() - 1; orderIdx >= 0; orderIdx--) // bottom-up order
    {
        // Determine AABB.

        int nodeIdx = nodeOrder[orderIdx];
        BvhNode& n = ioNodes[nodeIdx];
        float3 lo = make_float3(fminf(n.c0lox, n.c1lox), fminf(n.c0loy, n.c1loy), fminf(n.c0loz, n.c1loz));
        float3 hi = make_float3(fmaxf(n.c0hix, n.c1hix), fmaxf(n.c0hiy, n.c1hiy), fmaxf(n.c0hiz, n.c1hiz));

        // Store AABB to parent.

        if (nodeIdx != 0)
        {
            int parentIdx = nodeParents[nodeIdx];
            if (parentIdx >= 0)
            {
                BvhNode& p = ioNodes[parentIdx];
                p.c0lox = lo.x, p.c0loy = lo.y, p.c0loz = lo.z;
                p.c0hix = hi.x, p.c0hiy = hi.y, p.c0hiz = hi.z;
            }
            else
            {
                BvhNode& p = ioNodes[~parentIdx];
                p.c1lox = lo.x, p.c1loy = lo.y, p.c1loz = lo.z;
                p.c1hix = hi.x, p.c1hiy = hi.y, p.c1hiz = hi.z;
            }
        }

        // Callwlate sum of child costs.

        int subtreeSize = 0;
        float subtreeCost = 0.0f;

        if (n.c0idx < 0)
        {
            subtreeSize++;
            subtreeCost += aabbClampedHalfArea(n.c0hix - n.c0lox, n.c0hiy - n.c0loy, n.c0hiz - n.c0loz) * sahPrimCost;
        }
        else
        {
            subtreeSize += nodeSubtreeSize[n.c0idx];
            subtreeCost += nodeSubtreeCost[n.c0idx];
        }

        if (n.c1idx < 0)
        {
            subtreeSize++;
            subtreeCost += aabbClampedHalfArea(n.c1hix - n.c1lox, n.c1hiy - n.c1loy, n.c1hiz - n.c1loz) * sahPrimCost;
        }
        else
        {
            subtreeSize += nodeSubtreeSize[n.c1idx];
            subtreeCost += nodeSubtreeCost[n.c1idx];
        }

        // Callwlate final SAH cost for this node.

        float area = aabbClampedHalfArea(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);
        subtreeCost = setLSB(subtreeCost + area * sahNodeCost);

        if (subtreeSize <= maxLeafSize)
        {
            float leafCost = area * (float)subtreeSize * sahPrimCost;
            subtreeCost = fminf(subtreeCost, clearLSB(leafCost));
        }

        // Store results.

        nodeSubtreeSize[nodeIdx] = subtreeSize;
        nodeSubtreeCost[nodeIdx] = subtreeCost;
    }
}

//-----------------------------------------------------------------------

static void restructureTreelet(
    const TrbvhBuilder::Config& config,
    int                         treeletRoot,
    const BufferRef<BvhNode>&   nodes,
    const BufferRef<int>&       nodeParents,
    std::vector<int>&           nodeSubtreeSize,
    std::vector<float>&         nodeSubtreeCost)
{
    int N = config.optTreeletSize;
    int numSets = 1 << N;
    const int MAX_N = 8;
    const int MAX_SETS = 1 << MAX_N;

    // Allocate storage for the nodes of the treelet.
    // Indices 0..N-1 represent treelet leaves,
    // indices N..N*2-2 represent treelet internal nodes.
    // Treelet root is the first internal node, stored at index N.

    int     nodeIdx     [MAX_N * 2 - 1];
    float3  nodeLo      [MAX_N * 2 - 1];
    float3  nodeHi      [MAX_N * 2 - 1];
    float   nodeArea    [MAX_N * 2 - 1];
    int     nodeSet     [MAX_N * 2 - 1];
    int     nodeChild   [MAX_N * 2 - 1][2];

    // Form the treelet.

    nodeIdx[0] = treeletRoot;
    nodeArea[0] = 0.0f; // irrelevant

    for (int size = 1; size < N; size++)
    {
        // Find the largest treelet leaf that can be expanded.
        // Note: Reinterpreting the floats as unsigned integers ensures that
        // the comparison stays robust even if the input contains NaNs/Infs.

        unsigned int largestAreaU32 = 0;
        int largestIdx = -1;
        for (int i = 0; i < size; i++)
        {
            unsigned int nodeAreaU32 = __float_as_int(nodeArea[i]);
            if (nodeAreaU32 >= largestAreaU32 && nodeIdx[i] >= 0)
            {
                largestAreaU32 = nodeAreaU32;
                largestIdx = i;
            }
        }

        // Use the leaf as treelet internal node.

        int largestNode = nodeIdx[largestIdx];
        nodeIdx[N + size - 1] = largestNode;
        const BvhNode& n = nodes[largestNode];

        // Use the children as new treelet leaves.

        nodeIdx[largestIdx] = n.c0idx;
        nodeLo[largestIdx] = make_float3(n.c0lox, n.c0loy, n.c0loz);
        nodeHi[largestIdx] = make_float3(n.c0hix, n.c0hiy, n.c0hiz);
        nodeArea[largestIdx] = aabbClampedHalfArea(n.c0hix - n.c0lox, n.c0hiy - n.c0loy, n.c0hiz - n.c0loz);

        nodeIdx[size] = n.c1idx;
        nodeLo[size] = make_float3(n.c1lox, n.c1loy, n.c1loz);
        nodeHi[size] = make_float3(n.c1hix, n.c1hiy, n.c1hiz);
        nodeArea[size] = aabbClampedHalfArea(n.c1hix - n.c1lox, n.c1hiy - n.c1loy, n.c1hiz - n.c1loz);
    }

    // Allocate storage for each possible subset of the treelet leaf nodes.

    float   setArea         [MAX_SETS];
    float   setSubtreeCost  [MAX_SETS];
    int     setSubtreeSize  [MAX_SETS];
    int     setPartition    [MAX_SETS];

    // Callwlate surface area for each subset of leaves.
    {
        float3 tmpLo[MAX_N + 1];
        float3 tmpHi[MAX_N + 1];
        tmpLo[MAX_N] = make_float3(+LWDART_MAX_NORMAL_F, +LWDART_MAX_NORMAL_F, +LWDART_MAX_NORMAL_F);
        tmpHi[MAX_N] = make_float3(-LWDART_MAX_NORMAL_F, -LWDART_MAX_NORMAL_F, -LWDART_MAX_NORMAL_F);

        for (int set = 2; set < numSets; set += 2)
        {
            int leaf = findLeadingOne(set & -set);
            int parentSet = ( set & (set - 1) ) | MAX_SETS;
            int parentLeaf = findLeadingOne(parentSet & -parentSet);

            float3 lo0 = min(tmpLo[parentLeaf], nodeLo[leaf]);
            float3 hi0 = max(tmpHi[parentLeaf], nodeHi[leaf]);
            float3 lo1 = min(lo0, nodeLo[0]);
            float3 hi1 = max(hi0, nodeHi[0]);
            setArea[set + 0] = aabbClampedHalfArea(hi0.x - lo0.x, hi0.y - lo0.y, hi0.z - lo0.z);
            setArea[set + 1] = aabbClampedHalfArea(hi1.x - lo1.x, hi1.y - lo1.y, hi1.z - lo1.z);
            tmpLo[leaf] = lo0;
            tmpHi[leaf] = hi0;
        }
    }

    // Initialize costs for subsets of size 1.

    for (int i = 0; i < N; i++)
    {
        int node = nodeIdx[i];
        int set = 1 << i;

        if (node < 0) // BVH leaf
        {
            setSubtreeCost[set] = fmaxf(nodeArea[i] * config.sahPrimCost, 0.0f);
            setSubtreeSize[set] = 1;
        }
        else // BVH internal node
        {
            setSubtreeCost[set] = fmaxf(nodeSubtreeCost[node], 0.0f);
            setSubtreeSize[set] = nodeSubtreeSize[node];
        }
    }

    // Find optimal partitioning for each subset of size 2..N.

    for (int set = 3; set < numSets; set++)
    {
        // Smaller than 2 => skip.

        int delta = set & (set - 1);
        if (delta == 0)
            continue;

        // Try each way of partitioning the leaves into two subsets.

        float bestChildCost = LWDART_MAX_NORMAL_F;
        int bestPartition = 0;

        int partition = -delta & set;
        do
        {
            float childCost = setSubtreeCost[partition] + setSubtreeCost[set - partition];
            if (childCost <= bestChildCost)
            {
                bestChildCost = childCost;
                bestPartition = partition;
            }
            partition = (partition - delta) & set;
        }
        while (partition != 0);

        // Callwlate costs for this subset.

        setPartition[set] = bestPartition;
        setSubtreeSize[set] = setSubtreeSize[bestPartition] + setSubtreeSize[set - bestPartition];

        setSubtreeCost[set] = setLSB(setArea[set] * config.sahNodeCost + bestChildCost);
        if (setSubtreeSize[set] <= config.maxLeafSize)
        {
            float leafCost = clearLSB(setArea[set] * (float)setSubtreeSize[set] * config.sahPrimCost);
            setSubtreeCost[set] = fminf(setSubtreeCost[set], leafCost);
        }
    }

    // SAH cost not improved, or failed due to NaNs/Infs => skip the output part of the algorithm.

    if (setSubtreeCost[numSets - 1] >= nodeSubtreeCost[treeletRoot] ||
        (unsigned int)__float_as_int(setSubtreeCost[numSets - 1]) >= 0x7F7FFFFEu) // NaN, Inf, < 0.0f, or >= clearLSB(LWDART_MAX_NORMAL_F)?
    {
        return;
    }

    // Reconstruct the optimal treelet.

    nodeSet[N] = numSets - 1;

    for (int i = N, j = N + 1; i < j; i++)
    {
        // Which subsets do the children of this node correpond to?

        int set = nodeSet[i];
        int childSet[2];
        childSet[0] = setPartition[set];
        childSet[1] = set - childSet[0];

        // Assign a node to represent each child.

        for (int k = 0; k < 2; k++)
        {
            if ((childSet[k] & (childSet[k] - 1)) == 0) // treelet leaf
                nodeChild[i][k] = findLeadingOne(childSet[k]);
            else
            {
                nodeChild[i][k] = j;
                nodeSet[j] = childSet[k];
                j++;
            }
        }
    }

    // Update the internal nodes.

    for (int i = N * 2 - 2; i >= N; i--) // bottom-up order
    {
        int idx = nodeIdx[i];
        int set = nodeSet[i];
        int c0 = nodeChild[i][0];
        int c1 = nodeChild[i][1];
        BvhNode& n = nodes[idx];

        nodeLo[i] = min(nodeLo[c0], nodeLo[c1]);
        nodeHi[i] = max(nodeHi[c0], nodeHi[c1]);
        nodeSubtreeCost[idx] = setSubtreeCost[set];
        nodeSubtreeSize[idx] = setSubtreeSize[set];

        n.c0idx = nodeIdx[c0];
        n.c0lox = nodeLo[c0].x, n.c0loy = nodeLo[c0].y, n.c0loz = nodeLo[c0].z;
        n.c0hix = nodeHi[c0].x, n.c0hiy = nodeHi[c0].y, n.c0hiz = nodeHi[c0].z;
        if (n.c0idx >= 0)
            nodeParents[n.c0idx] = idx;

        n.c1idx = nodeIdx[c1];
        n.c1lox = nodeLo[c1].x, n.c1loy = nodeLo[c1].y, n.c1loz = nodeLo[c1].z;
        n.c1hix = nodeHi[c1].x, n.c1hiy = nodeHi[c1].y, n.c1hiz = nodeHi[c1].z;
        if (n.c1idx >= 0)
            nodeParents[n.c1idx] = ~idx;
    }
}

//------------------------------------------------------------------------

static void optimizeNodeTopology(
    TrbvhBuilder::Config&   config,
    BufferRef<BvhNode>      ioNodes,
    std::vector<int>&       nodeOrder,
    BufferRef<int>          nodeParents,
    std::vector<int>&       nodeSubtreeSize,
    std::vector<float>&     nodeSubtreeCost)
{
    for (int round = 0; round < config.optRounds; round++)
    {
        // Choose gamma.

        int gamma = config.optGamma;
        if (config.optAdaptiveGamma)
            gamma <<= min(round, 3);

        // Enumerate treelet roots.

        topologicalSort(nodeOrder, ioNodes);
        for (int orderIdx = (int)nodeOrder.size() - 1; orderIdx >= 0; orderIdx--) // bottom-up order
        {
            int nodeIdx = nodeOrder[orderIdx];
            if (nodeSubtreeSize[nodeIdx] >= gamma)
                restructureTreelet(config, nodeIdx, ioNodes, nodeParents, nodeSubtreeSize, nodeSubtreeCost);
        }
    }
}

//------------------------------------------------------------------------

void collapseSubtreesIntoLeaves(
    TrbvhBuilder::Config&       config,
    int                         numPrims,
    std::vector<int>&           nodeOrder,
    BufferRef<BvhNode>          ioNodes,
    std::vector<float>&         nodeSubtreeCost,
    BufferRef<int>              ioRemap,
    BufferRef<int>              ioNumRemaps,
    int                         nodeRangeStart)
{
    // Less than 2 primitives => special case to match the logic in computeBinaryRadixTree().

    if (numPrims < 2)
    {
        int remapVal = ~ioNodes[0].c0idx;
        if (config.listLenEnc == RLLE_COMPLEMENT_LAST)
            remapVal ^= -1;
        else if (config.listLenEnc == RLLE_PACK_IN_FIRST)
            remapVal |= numPrims << RLLEPACK_LEN_SHIFT;

        int remapIdx = *ioNumRemaps;
        *ioNumRemaps += numPrims;
        ioRemap[remapIdx] = remapVal;

        ioNodes[0].c0idx = ~remapIdx;
        ioNodes[0].c1idx = ~remapIdx;
        ioNodes[0].c0num = numPrims;
        ioNodes[0].c1num = 0;
        return;
    }

    // Process nodes in top-down order.

    std::vector<int> leafPrimitives;
    std::vector<int> collapseStack;
    leafPrimitives.reserve(config.maxLeafSize);
    collapseStack.reserve(config.maxLeafSize);

    topologicalSort(nodeOrder, ioNodes);
    for (int orderIdx = 0; orderIdx < static_cast<int>( nodeOrder.size() ); orderIdx++)
    {
        // Parent was already collapsed => skip.

        int parentIdx = nodeOrder[orderIdx];
        BvhNode& parent = ioNodes[parentIdx];
        if (parentIdx != 0 && getLSB(nodeSubtreeCost[parentIdx]) == 0)
            continue;

        // Consider each child.

        for (int edgeDir = 0; edgeDir < 2; edgeDir++)
        {
            int childIdx = (edgeDir == 0) ? parent.c0idx : parent.c1idx;
            int leafSize = BVHNODE_CHILD_IS_INTERNAL_NODE;

            // Child is a leaf, or wants to be a leaf => collapse.

            if (childIdx < 0 || getLSB(nodeSubtreeCost[childIdx]) == 0)
            {
                // Traverse subtree and collect primitives.

                leafPrimitives.clear();
                collapseStack.push_back(childIdx);
                while (collapseStack.size())
                {
                    int nodeIdx = collapseStack.back();
                    collapseStack.pop_back();

                    if (nodeIdx < 0)
                        leafPrimitives.push_back(~nodeIdx);
                    else
                    {
                        nodeSubtreeCost[nodeIdx] = 0.0f; // mark as collapsed
                        ioNodes[nodeIdx].c1num = BVHNODE_NOT_IN_USE;
                        collapseStack.push_back(ioNodes[nodeIdx].c0idx);
                        collapseStack.push_back(ioNodes[nodeIdx].c1idx);
                    }
                }

                // Remove duplicates.

                for (int i = (int)leafPrimitives.size() - 1; i > 0; i--)
                {
                    int ti = leafPrimitives[i];
                    for (int j = 0; j < i; j++)
                    {
                        if (ti == leafPrimitives[j])
                        {
                            leafPrimitives[i] = leafPrimitives.back();
                            leafPrimitives.pop_back();
                            break;
                        }
                    }
                }

                // Encode list length

                leafSize = (int)leafPrimitives.size();
                if (config.listLenEnc == RLLE_COMPLEMENT_LAST)
                    leafPrimitives.back() ^= -1;
                else if(config.listLenEnc == RLLE_PACK_IN_FIRST)
                    leafPrimitives[0] |= (leafSize << RLLEPACK_LEN_SHIFT);

                // Output remap entries.

                childIdx = ~(*ioNumRemaps);
                for (int i = 0; i < leafSize; i++) 
                    ioRemap[(*ioNumRemaps)++] = leafPrimitives[i];               
            }

            // Child is an internal node => express the index as absolute wrt. the entire node buffer.

            if (childIdx >= 0)
                childIdx += nodeRangeStart;

            // Update child reference.

            if (edgeDir == 0)
                parent.c0idx = childIdx, parent.c0num = leafSize;
            else
                parent.c1idx = childIdx, parent.c1num = leafSize;
        }
    }
}

//------------------------------------------------------------------------

void TrbvhBuilder::execHost(void)
{
    m_cfg.outNodeParents    .writeDiscardHost();
    m_cfg.outNodeRange      .writeDiscardHost();
    m_cfg.ioNodes           .writeHost();
    m_cfg.ioNumNodes        .readWriteHost();
    m_cfg.ioRemap           .writeHost();
    m_cfg.ioNumRemaps       .readWriteHost();
    m_cfg.ioMortonCodes     .readHost();
    m_cfg.inPrimOrder       .readHost();
    m_cfg.inPrimRange       .readHost();
    ModelPointers inModel   (m_cfg.inModel, MemorySpace_Host);

    // Callwlate output node range.

    Range primRange = (m_cfg.inPrimRange.getNumElems()) ? *m_cfg.inPrimRange : Range(0, m_cfg.maxPrims);
    RT_ASSERT(primRange.span() >= 0 && primRange.span() <= m_cfg.maxPrims);

    int nodeRangeStart = *m_cfg.ioNumNodes;
    int numNodes = max(primRange.span() - 1, 1);
    *m_cfg.outNodeRange = Range(nodeRangeStart, nodeRangeStart + numNodes);
    *m_cfg.ioNumNodes += numNodes;

    // Build binary radix tree.

    BufferRef<BvhNode> nodes = m_cfg.ioNodes.getSubrange(nodeRangeStart, numNodes);
    computeBinaryRadixTree( inModel, m_cfg.ioMortonCodes, m_cfg.inPrimOrder, primRange, m_cfg.outNodeParents, nodes );
    
    std::vector<int> nodeOrder;
    nodeOrder.reserve(numNodes);
    topologicalSort(nodeOrder, nodes);

    // Optimize topology.
    
    std::vector<int> nodeSubtreeSize(numNodes, 0);
    std::vector<float> nodeSubtreeCost(numNodes, 0);
    computeAabbsAndCosts( nodeOrder, nodes, m_cfg.outNodeParents, nodeSubtreeSize, nodeSubtreeCost, m_cfg.sahPrimCost, m_cfg.sahNodeCost, m_cfg.maxLeafSize );    
    optimizeNodeTopology( m_cfg, nodes, nodeOrder, m_cfg.outNodeParents, nodeSubtreeSize, nodeSubtreeCost );

    // Collapse.
    
    collapseSubtreesIntoLeaves( m_cfg, primRange.span(), nodeOrder, nodes, nodeSubtreeCost, m_cfg.ioRemap, m_cfg.ioNumRemaps, nodeRangeStart );
}

//------------------------------------------------------------------------
