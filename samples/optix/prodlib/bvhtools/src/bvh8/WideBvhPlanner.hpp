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

#pragma once
#include <prodlib/bvhtools/src/BuildingBlock.hpp>
#include "WideBvhPlannerKernels.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// WideBvhPlanner finds a near-optimal way to group the nodes of a binary
// BVH together to form a set of wide BVH nodes that minimize the total
// SAH cost.
//
// The input binary BVH is in a specific format with one primitive per
// leaf, dolwmented in WideBvhPlannerKernels.hpp. The main output of
// WideBvhPlanner is that it augments the nodes with additional grouping
// information that enables subsequent building blocks to construct the
// actual wide BVH afterwards. WideBvhPlanner itself does not output
// any BVH nodes.
//
// WideBvhPlanner provides a strict upper bound for the maximum number of
// wide BVH nodes. The upper bound can be queried through getMaxWideNodes(),
// and it is mainly useful for allocating memory for the subsequent building
// blocks. By default, the upper bound about 1.5x higher than the actual
// number of wide nodes in typical cases. The exact behavior can be
// controlled through the squeezeMode parameter.
//
// The upper bound is enforced through a two-step process. We first perform
// an initial planning pass that aims to minimize the SAH cost ignoring the
// upper bound completely. We then perform an additional squeezing pass to
// adjust the planned wide nodes so that they are guaranteed to satisfy the
// upper bound. Our expectation is that the initial planning pass will
// already satisfy the upper bound in 99.9% of the cases, so the squeezing
// pass will be a no-op that exelwtes very quickly.
//
// The basic problem of colwerting binary BVHs to wide BVHs is dislwssed
// on slides 16--29 of the following presentation, and the WideBvhPlanner
// algorithm is detailed on slides 43--60:
//
// https://p4viewer.lwpu.com/get///research/research/tkarras/rt-moonshot/docs/cbvh-builders/flexcbvh-part2-2015-06-03.pptx
//
// In addition to grouping binary BVH nodes, WideBvhPlanner also supports
// the concept of instance splitting as described on slides 20--28 of the
// following presentation:
//
// https://p4viewer.lwpu.com/get///research/research/tkarras/rt-moonshot/docs/instancing-2014-06-19/instancing-2014-06-19.pptx
//
// Instance splitting can be enabled/disabled through the allowMultipleRoots
// parameter. When disabled, the root of the wide BVH corresponds strictly
// to the root of the binary BVH and the corresponding output buffers
// (outNumRoots, outSubtreeRoots) are left empty. When enabled, the wide BVH
// may end up with several roots that each correspond to a different subtree
// of the binary BVH. The roots can be located by traversing the outSubtreeRoots
// array that tells how many roots there are within any given binary BVH subtree.
//
// Strictly speaking, WideBvhPlanner attempts to minimize the following SAH
// cost formula:
//
//   sumOverWideNodes(surfaceArea[i] * sahNodeCost)
// + sumOverWideLeaves(surfaceArea[i] * numPrimitives[i] * sahPrimCost)
// + sumOverWideRoots(surfaceArea[i] * sahRootCost)
// + sumOverBinaryNodesAboveWideRoots(surfaceArea[i] * sahTopLevelCost)
//
// The term 'wide BVH node' generally refers to an internal node of the wide BVH.
// Leaves are not considered as 'wide BVH nodes'. A 'wide BVH root' is always
// considered to be a 'wide BVH node'.

class WideBvhPlanner : public BuildingBlock
{
public:
    // SqueezeMode controls the way WideBvhPlanner callwlates the value of getMaxWideNodes().
    // Each choice provides a different tradeoff between memory usage, BVH quality, and code coverage.
    enum SqueezeMode
    {
        SqueezeMode_Default = 0,        // Empirical formula that is suitable for most practical use cases.
                                        // Avoids the squeeze pass with ~99.9% probability while providing a reasonable upper bound that is conservative by ~1.5x.
        SqueezeMode_MaxSqueeze,         // Always squeeze the wide BVH as much as possible. Provides a very tight upper bound, but also a very low BVH quality.
                                        // This mode is mainly useful for unit tests, where we want to check that the squeeze pass works as intended.
        SqueezeMode_NoSqueeze,          // Never run the squeeze pass. Provides the best possible BVH quality, but also makes the upper bound extremely conservative (>5x).
                                        // This mode is mainly useful for research experiments where we wish to ignore the effect of the squeeze pass.
    };

    struct Config
    {
        LwdaUtils*  lwca;               // Non-NULL => execute on LWCA.
        int         maxBranchingFactor; // Maximum number of children per wide BVH node.
        int         maxLeafSize;        // Maximum number of primitives per wide BVH leaf node.
        SqueezeMode squeezeMode;        // How to callwlate the value of getMaxWideNodes()?

        float       sahNodeCost;        // Cost of testing the ray against the children of one wide BVH node.
        float       sahPrimCost;        // Cost of testing the ray against one primitive.
        float       sahRootCost;        // Cost of entering one wide BVH root.

        bool        allowMultipleRoots; // Allow creating multiple wide BVH roots?
        float       minAreaPerRoot;     // Minimum surface area per wide BVH root, relative to model AABB area.
        float       sahTopLevelCost;    // Cost of deferring one binary node to the top-level BVH.

                                                            // Size                                 Description
        BufferRef<int>                  outNumWideNodes;    // = 1                                  Number of planned wide BVH nodes.
        BufferRef<int>                  outNumRoots;        // = 0 or 1                             Number of planned wide BVH roots.
        BufferRef<int>                  outSubtreeRoots;    // = 0 or ioBinaryNodes.getNumElems()   Number of wide BVH roots within each binary BVH subtree, assuming that none of its ancestors is a root.
        BufferRef<>                     tempBufferA;        // = ~4*ioBinaryNodes.getNumElems()     First temporary buffer.
        BufferRef<>                     tempBufferB;        // = ~4*ioBinaryNodes.getNumElems()     Second temporary buffer. Split in two to facilitate buffer layout in top-level builder classes.

        BufferRef<PlannerBinaryNode>    ioBinaryNodes;      // >= 2                                 Input: Binary BVH nodes. Output: Nodes annotated with grouping info about the planned wide BVH nodes.
        BufferRef<const Range>          inPrimRange;        // 1                                    Range of input primitives represented by ioBinaryNodes.
        BufferRef<const ApexPointMap>   inApexPointMap;     // <varies>                             Produced by ApexPointMapConstructor.

        Config(void)
        {
            lwca                = NULL;
            maxBranchingFactor  = 12;
            maxLeafSize         = 16;
            squeezeMode         = SqueezeMode_Default;
            sahNodeCost         = 8.0f;
            sahPrimCost         = 1.0f;
            sahRootCost         = 32.0f;
            allowMultipleRoots  = false;
            minAreaPerRoot      = 0.01f;    // Avoid excessive number of roots in pathological cases.
            sahTopLevelCost     = 4.0f;
        }
    };

public:
                            WideBvhPlanner              (void) {}
    virtual                 ~WideBvhPlanner             (void) {}

    virtual const char*     getName                     (void) const { return "WideBvhPlanner"; }
    void                    configure                   (const Config& config);
    int                     getMaxWideNodes             (void) const { return m_maxWideNodes; }
    void                    execute                     (void);

private:
    void                    execDevice                  (void);
    void                    execHost                    (void);

    void                    initialPlanningPass         (void);
    void                    chooseWideNodeShape         (PlannerBinaryNode&                     root,               // in/out
                                                         const BufferRef<PlannerBinaryNode>&    binaryNodes,        // in/out
                                                         int&                                   totalWideNodes,     // out
                                                         const std::vector<int>&                subtreeWideNodes);

    void                    squeezingPass               (void);
    int                     callwlateMaxWideNodes       (int maxPrims) const;

private:
                            WideBvhPlanner              (const WideBvhPlanner&); // forbidden
    WideBvhPlanner&         operator=                   (const WideBvhPlanner&); // forbidden

private:
    Config                  m_cfg;
    int                     m_maxWideNodes;

    // Temp buffers.

    BufferRef<int>          m_workCounter;
    BufferRef<int>          m_taskCounter;
    BufferRef<int>          m_primCounter;
    BufferRef<PlannerTempA> m_tempA;
    BufferRef<PlannerTempB> m_tempB;
    BufferRef<PlannerSqueezeTask> m_squeezeTasks;

    // Temps for host-side chooseWideNodeShape().

    std::vector<int>        m_innerNodes;
    std::vector<int>        m_innerDepth;
    std::vector<int>        m_deltaWideNodes;
    std::vector<float>      m_deltaCost;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
