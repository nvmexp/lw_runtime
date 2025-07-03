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

#include "BVH8Fitter.hpp"
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>
#include <algorithm>

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

void BVH8Fitter::configure(const Config& cfg)
{
    RT_ASSERT(cfg.maxPrims >= 0);
    RT_ASSERT(cfg.ioRemap.getNumElems() >= (size_t)cfg.maxPrims);
    RT_ASSERT(cfg.inNumNodes.getNumElems() == 1);
    RT_ASSERT(cfg.inModel.isValid());
    RT_ASSERT(!m_cfg.colwertTriangles || !m_cfg.inModel.isAABBs());

    if (cfg.maxLeafSize < 1 || cfg.maxLeafSize > BVH8FITTER_NODES_MAX_LEAF_SIZE)
        throw IlwalidValue(RT_EXCEPTION_INFO, "maxLeafSize must be between 1 and BVH8FITTER_NODES_MAX_LEAF_SIZE!", cfg.maxLeafSize);

    if (cfg.numReorderRounds < 1)
        throw IlwalidValue(RT_EXCEPTION_INFO, "numReorderRounds must be at least 1!", cfg.numReorderRounds);

    // Set config and resize outputs.

    m_cfg = cfg;
    m_cfg.outTriangles.setNumElems((m_cfg.colwertTriangles) ? m_cfg.maxPrims : 0);

    // Layout temp buffers.

    m_workCounter.assignNew(1);
    m_cfg.tempBuffer.aggregate(m_workCounter);
}

//------------------------------------------------------------------------

void BVH8Fitter::execute(void)
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

void BVH8Fitter::execDevice(void)
{
    RT_ASSERT((BVH8FITTER_NODES_GROUP_SIZE & (BVH8FITTER_NODES_GROUP_SIZE - 1)) == 0);
    RT_ASSERT(32 % BVH8FITTER_NODES_GROUP_SIZE == 0);
    bool colwertDirectly = (m_cfg.colwertTriangles && !m_cfg.translateIndices && !m_cfg.removeDuplicates && m_cfg.supportRefit && !m_cfg.inModel.splitAABBs.getNumElems());
    ModelBuffers modelWithoutSplitAABBs = ModelBuffers(m_cfg.inModel, EmptyBuf);

    // Clear work counter.

    m_workCounter.clearLWDA(0);

    // Launch BVH8FitterNodes.
    {
        BVH8FitterNodesParams p = {};
        p.ioNodes           = m_cfg.ioNodes.readWriteLWDA();
        p.ioNodeAux         = m_cfg.ioNodeAux.readWriteLWDA();
        p.ioRemap           = m_cfg.ioRemap.readWriteLWDA();
        p.outTriangles      = (colwertDirectly) ? m_cfg.outTriangles.writeLWDA() : NULL;
        p.workCounter       = m_workCounter.readWriteLWDA();
        p.inNumNodes        = m_cfg.inNumNodes.readLWDA();
        p.inModel           = ModelPointers(m_cfg.inModel, MemorySpace_LWDA);
        p.translateIndices  = (m_cfg.translateIndices) ? 1 : 0;
        p.removeDuplicates  = (m_cfg.removeDuplicates) ? 1 : 0;
        p.numReorderRounds  = (m_cfg.reorderChildSlots) ? m_cfg.numReorderRounds : 0;
        p.supportRefit      = (m_cfg.supportRefit) ? 1 : 0;
    
        LAUNCH(*m_cfg.lwca, BVH8FitterNodes, BVH8FITTER_NODES_WARPS_PER_BLOCK,
            min(m_cfg.lwca->getMaxThreads(), (int)m_cfg.ioNodes.getNumElems() * BVH8FITTER_NODES_GROUP_SIZE), p);
    }

    // Launch BVH8FitterTriangles if requested.

    if (m_cfg.colwertTriangles && !colwertDirectly)
    {
        BVH8FitterTrianglesParams p = {};
        p.outTriangles  = m_cfg.outTriangles.writeLWDA();
        p.inRemap       = m_cfg.ioRemap.readLWDA();
        p.inModel       = ModelPointers(modelWithoutSplitAABBs, MemorySpace_LWDA);
        p.maxPrims      = m_cfg.maxPrims;
    
        LAUNCH(*m_cfg.lwca, BVH8FitterTriangles, BVH8FITTER_TRIANGLES_WARPS_PER_BLOCK,
            m_cfg.maxPrims, p);
    }
}

//------------------------------------------------------------------------

void BVH8Fitter::execHost(void)
{
    m_cfg.ioNodes       .readWriteHost();
    m_cfg.ioNodeAux     .readWriteHost();
    m_cfg.ioRemap       .readWriteHost();
    m_cfg.inNumNodes    .readHost();
    ModelPointers inModel(m_cfg.inModel, MemorySpace_Host);

    // Process nodes in bottom-up order.

    for (int nodeIdx = *m_cfg.inNumNodes - 1; nodeIdx >= 0; nodeIdx--)
    {
        BVH8Node& node = m_cfg.ioNodes[nodeIdx];
        BVH8NodeAux& aux = m_cfg.ioNodeAux[nodeIdx];

        // Callwlate child AABBs, finalize remap, and populate leaf references.

        BVH8Meta childMeta[8];
        AABB childAABB[8];
        int numChildNodes = 0;

        for (int lane = 0; lane < 8; lane++)
        {
            childMeta[lane] = node.header.meta[lane];
            childAABB[lane] = AABB(+FLT_MAX, -FLT_MAX);

            // Inner node => look up AABB.

            if (childMeta[lane].isInner())
            {
                int childNodeIdx = node.header.firstChildIdx + numChildNodes;
                const BVH8Node& childNode = m_cfg.ioNodes[childNodeIdx];
                childAABB[lane].lo.x = childNode.header.pos[0]; // min corner comes from the finalized child node
                childAABB[lane].lo.y = childNode.header.pos[1];
                childAABB[lane].lo.z = childNode.header.pos[2];
                childAABB[lane].hi = m_cfg.ioNodeAux[childNodeIdx].aabbHi; // max corner comes from the temporary array
                numChildNodes++;
            }

            // Leaf node => process each primitive.

            else if (childMeta[lane].isLeaf())
            {
                int remapOfs        = childMeta[lane].getLeafRemapOfs();
                int numLeafPrims    = childMeta[lane].getLeafNumPrims();
                int remapBase       = node.header.firstRemapIdx + remapOfs;
                int numUniquePrims  = 0;
                childAABB[lane]     = AABB(+FLT_MAX, -FLT_MAX);

                for (int i = 0; i < numLeafPrims; i++)
                {
                    // Fetch primitive AABB.

                    int primitiveIdx = m_cfg.ioRemap[remapBase + i];
                    PrimitiveAABB primAABB = inModel.loadPrimitiveAABB(primitiveIdx);
                    childAABB[lane] = AABB(childAABB[lane], primAABB);

                    // Translate primitive index if requested.

                    if (m_cfg.translateIndices)
                        primitiveIdx = primAABB.primitiveIdx;

                    // Detect duplicates if requested.

                    bool duplicate = false;
                    if (m_cfg.removeDuplicates)
                        for (int j = 0; j < numUniquePrims && !duplicate; j++)
                            if (m_cfg.ioRemap[remapBase + j] == primitiveIdx)
                                duplicate = true;

                    // Not a duplicate => output updated remap entry.

                    if (!duplicate)
                    {
                        m_cfg.ioRemap[remapBase + numUniquePrims] = primitiveIdx;
                        numUniquePrims++;
                    }
                }

                // Update leaf data.

                if (m_cfg.removeDuplicates)
                    childMeta[lane].setLeaf(remapOfs, numUniquePrims);
            }
        }

        // Compute node AABB as the union of all child AABBs.

        AABB nodeAABB = childAABB[0];
        for (int lane = 1; lane < 8; lane++)
            nodeAABB = AABB(nodeAABB, childAABB[lane]);

        // Optimize child slot assignment if requested.

        int childSlot[8]; // [lane] => slot
        for (int i = 0; i < 8; i++)
            childSlot[i] = i;

        if (m_cfg.reorderChildSlots)
        {
            // Initialize assignment score matrix.
            // score[lane][slot] indicates how desirable it is to place the given child in the given slot.

            float score[8][8];
            for (int lane = 0; lane < 8; lane++)
            {
                // Callwlate child centroid relative to the parent.

                float cx = 0.0f, cy = 0.0f, cz = 0.0f;
                if (childMeta[lane].isInner())
                {
                    cx = childAABB[lane].lo.x + childAABB[lane].hi.x - nodeAABB.lo.x - nodeAABB.hi.x;
                    cy = childAABB[lane].lo.y + childAABB[lane].hi.y - nodeAABB.lo.y - nodeAABB.hi.y;
                    cz = childAABB[lane].lo.z + childAABB[lane].hi.z - nodeAABB.lo.z - nodeAABB.hi.z;
                }

                // Callwlate assignment score as a dot product with the
                // diagonal vector corresponding to the given child slot.

                for (int slot = 0; slot < 8; slot++)
                {
                    score[lane][slot] =
                        (((slot & 1) == 0) ? cx : -cx) +
                        (((slot & 2) == 0) ? cy : -cy) +
                        (((slot & 4) == 0) ? cz : -cz);
                }
            }

            // Optimize child slots for the given number of rounds using a greedy algorithm that
            // exchanges slots between a pair of lanes whenever this increases the overall score.

            for (int round = 0; round < m_cfg.numReorderRounds; round++)
            for (int mask = 1; mask < 8; mask++)
            for (int lane = 0; lane < 8; lane++)
            {
                int pair = lane ^ mask;
                float laneDiff = score[lane][childSlot[pair]] - score[lane][childSlot[lane]];
                float pairDiff = score[pair][childSlot[lane]] - score[pair][childSlot[pair]];
                if (laneDiff + pairDiff > 0.0f)
                    std::swap(childSlot[lane], childSlot[pair]);
            }
        }

        // Store min corner in node header and determine scale, rounding up to next power of two.
        // (zero/small => 1, FLT_MAX/Inf/NaN => 248)

        node.header.pos[0] = nodeAABB.lo.x;
        node.header.pos[1] = nodeAABB.lo.y;
        node.header.pos[2] = nodeAABB.lo.z;

        int magic = 0x0000FFFF - (7 << 23); // yields quantized AABB range 0..255; change to "0x007FFFFF - (7 << 23)" for 0..128
        node.header.scale[0] = (unsigned char)std::min(std::max((__float_as_int(__fsub_ru(nodeAABB.hi.x, nodeAABB.lo.x)) + magic) >> 23, 1), 248);
        node.header.scale[1] = (unsigned char)std::min(std::max((__float_as_int(__fsub_ru(nodeAABB.hi.y, nodeAABB.lo.y)) + magic) >> 23, 1), 248);
        node.header.scale[2] = (unsigned char)std::min(std::max((__float_as_int(__fsub_ru(nodeAABB.hi.z, nodeAABB.lo.z)) + magic) >> 23, 1), 248);

        // Update node meta and innerMask.

        node.header.innerMask = 0u;
        for (int lane = 0; lane < 8; lane++)
        {
            int slot = childSlot[lane];
            if (childMeta[lane].isInner())
            {
                node.header.meta[slot].setInner(slot);
                node.header.innerMask |= 1u << slot;
            }
            else
            {
                node.header.meta[slot] = childMeta[lane];
            }
        }

        // Store max corner in aux data, unless this is the root node.

        if (aux.parentNodeIdx != -1)
            aux.aabbHi = nodeAABB.hi;
        RT_ASSERT(aux.numChildNodes == numChildNodes);

        // Callwlate multipliers for child AABB planes (1.0f / exp2(scl)).

        float xmul = __int_as_float((254 - node.header.scale[0]) << 23);
        float ymul = __int_as_float((254 - node.header.scale[1]) << 23);
        float zmul = __int_as_float((254 - node.header.scale[2]) << 23);

        // Fill in quantized child AABB data.

        for (int lane = 0; lane < 8; lane++)
        {
            if (childMeta[lane].isEmpty())
                continue;

            const AABB& cb = childAABB[lane];
            int slot = childSlot[lane];
            node.lox[slot] = (unsigned char)std::min(std::max((int)floorf(__fsub_rd(cb.lo.x, nodeAABB.lo.x) * xmul), 0x00), 0xFF);
            node.loy[slot] = (unsigned char)std::min(std::max((int)floorf(__fsub_rd(cb.lo.y, nodeAABB.lo.y) * ymul), 0x00), 0xFF);
            node.loz[slot] = (unsigned char)std::min(std::max((int)floorf(__fsub_rd(cb.lo.z, nodeAABB.lo.z) * zmul), 0x00), 0xFF);
            node.hix[slot] = (unsigned char)std::min(std::max((int)ceilf (__fsub_ru(cb.hi.x, nodeAABB.lo.x) * xmul), 0x00), 0xFF);
            node.hiy[slot] = (unsigned char)std::min(std::max((int)ceilf (__fsub_ru(cb.hi.y, nodeAABB.lo.y) * ymul), 0x00), 0xFF);
            node.hiz[slot] = (unsigned char)std::min(std::max((int)ceilf (__fsub_ru(cb.hi.z, nodeAABB.lo.z) * zmul), 0x00), 0xFF);
        }

        // Permute child nodes in memory to match the new slot assignment.

        if (m_cfg.reorderChildSlots)
        {
            BVH8Node childNodes[8] = {};
            BVH8NodeAux childAux[8] = {};

            int oldIdx = node.header.firstChildIdx;
            for (int i = 0; i < 8; i++)
            {
                if (childMeta[i].isInner())
                {
                    childNodes[i] = m_cfg.ioNodes[oldIdx];
                    childAux[i] = m_cfg.ioNodeAux[oldIdx];
                    oldIdx++;
                }
            }

            for (int i = 0; i < 8; i++)
            {
                int newIdx = node.header.firstChildIdx + __popc(node.header.innerMask & ((1 << childSlot[i]) - 1));
                if (childMeta[i].isInner())
                {
                    m_cfg.ioNodes[newIdx] = childNodes[i];
                    m_cfg.ioNodeAux[newIdx] = childAux[i];
                }
            }
        }

        // Reset parentNodeIdx for subsequent refits if requested.

        if (m_cfg.supportRefit)
        {
            for (int i = 0; i < aux.numChildNodes; i++)
            {
                // Always reset for the direct child.

                int child = node.header.firstChildIdx + i;
                m_cfg.ioNodeAux[child].parentNodeIdx = nodeIdx;

                // Also reset for the grandchildren if nodes were reordered.

                if (m_cfg.reorderChildSlots)
                {
                    for (int j = 0; j < m_cfg.ioNodeAux[child].numChildNodes; j++)
                    {
                        int grandchild = m_cfg.ioNodes[child].header.firstChildIdx + j;
                        m_cfg.ioNodeAux[grandchild].parentNodeIdx = child;
                    }
                }
            }
        }
    }

    // Colwert triangles if requested.

    if (m_cfg.colwertTriangles)
    {
        m_cfg.outTriangles.writeHost();
        for (int remapIdx = 0; remapIdx < m_cfg.maxPrims; remapIdx++)
        {
            BVH8Triangle tri = {};
            tri.userTriangleID = m_cfg.ioRemap[remapIdx];

            if (tri.userTriangleID >= 0 && tri.userTriangleID < inModel.numPrimitives)
            {
                float3 v0, v1, v2;
                inModel.loadVertexPositions(v0, v1, v2, tri.userTriangleID);
                tri.primBits = m_cfg.outTriangles[remapIdx].primBits;
                tri.v0x = v0.x, tri.v0y = v0.y, tri.v0z = v0.z;
                tri.v1x = v1.x, tri.v1y = v1.y, tri.v1z = v1.z;
                tri.v2x = v2.x, tri.v2y = v2.y, tri.v2z = v2.z;
            }
            m_cfg.outTriangles[remapIdx] = tri;
        }
    }
}

//------------------------------------------------------------------------
