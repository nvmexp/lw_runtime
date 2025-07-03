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
#include "AacBuilderKernels.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// AacBuilder constructs a medium-quality binary BVH from the given set
// of AABBs and sorted Morton codes. It combines several recently published
// methods as well as some LWPU secret sauce to construct the entire
// hierarchy in bottom-up fashion using a single kernel launch.
//
// See slides 61--99 of the following presentation for a high-level overview:
//
// https://p4viewer.lwpu.com/get///research/research/tkarras/rt-moonshot/docs/cbvh-builders/flexcbvh-part2-2015-06-03.pptx
//
// Algorithm:
//
// - We define an implicit LBVH hierarchy based on the input Morton codes,
//   and traverse its nodes in a bottom-up fashion by each thread using
//   a variant of the following method:
//
//      Fast and Simple Agglomerative LBVH Construction
//      Ciprian Apetrei, Computer Graphics and Visual Computing 2014
//      http://diglib.eg.org/handle/10.2312/cgvc.20141206.041-044
//
// - We create the output nodes using approximate agglomerative clustering
//   (AAC), as described in the following paper. For each LBVH node, we keep
//   around up to aacMaxClusterSize unmerged output nodes.
//
//      Efficient BVH construction via approximate agglomerative clustering
//      Gu, He, Fatahalian, Blelloch, High Performance Graphics 2013
//      http://dl.acm.org/citation.cfm?id=2492054
//
// - We implement node merging in a parallel fashion by switching to
//   warp-wide processing at each LBVH node. Several pairs of nodes are
//   identified and merged simultaneously as described in the following
//   LWPU patent:
//
//      Agglomerative treelet restructuring for bounding volume hierarchies
//      Aila, Karras, US 20140365529 A1
//      http://www.google.com/patents/US20140365529
//
// - The AAC merge cost function includes a penalty term (see the p4viewer link
//   above) that greatly improves the resulting BVH quality on regular input
//   geometry. This is a partilwlarly important optimization when triangle
//   splitting is enabled.
//
// Input:
//
// - AacBuilder is intended to be exelwted right after MortonSorter.
//   inModel, and inApexPointMap should be the same for both blocks.
//   inPrimOrder, inMortonCodes, and inPrimRange should be grabbed directly
//   from the corresponding MortonSorter outputs.
//
// Output:
//
// - At the moment, AacBuilder is only capable of producing the node
//   hierarchy in the format expected by WideBvhPlanner with one primitive
//   per leaf. See WideBvhPlannerKernels.hpp for details.
//
// - In the future, other output formats could be added relatively easily.
//   AacBuilder itself does not need to read the output node array;
//   it only writes out the data.

class AacBuilder : public BuildingBlock
{
public:
    struct Config
    {
        LwdaUtils*  lwca;                       // Non-NULL => execute on LWCA.
        int         maxPrims;                   // Upper bound for the number of input primitives.

        int         bytesPerMortonCode;         // Size of the Morton codes in bytes. Must be 4 or 8.
        int         aacMaxClusterSize;          // Max. unmerged subtrees to keep around per binary node. Must be 4, 8, or 16.
        float       aacPenaltyTermCoef;         // Coefficient for the merge cost penalty term.

                                                            // Size                     Description
        BufferRef<PlannerBinaryNode>    outNodes;           // = max(maxPrims*2,2)      Output nodes for WideBvhPlanner. The children of the root are always at indices 0 and 1.
        BufferRef<>                     tempBuffer;         // = ~8*maxPrims            Temporary buffer.

        BufferRef<const int>            inPrimOrder;        // >= maxPrims              Indices of the PrimitiveAABBs in Morton order.
        BufferRef<const unsigned char>  inMortonCodes;      // >= maxPrims              Morton codes corresponding to inPrimOrder. Must be non-descending.
        BufferRef<const Range>          inPrimRange;        // 1                        Range of input primitives in inPrimOrder.
        ModelBuffers                    inModel;            // <varies>                 Input model.
        BufferRef<const ApexPointMap>   inApexPointMap;     // <varies>                 Produced by ApexPointMapConstructor. Must enclose all input primitives.

        Config(void)
        {
            lwca                = NULL;
            maxPrims            = 0;
            bytesPerMortonCode  = sizeof(unsigned long long);
            aacMaxClusterSize   = 16;   // Aim for maximum quality by default.
            aacPenaltyTermCoef  = 1.0f; // The exact value does not matter much. 1.0f is as good a choice as any.
        }
    };

public:
                                AacBuilder          (void) {}
    virtual                     ~AacBuilder         (void) {}

    virtual const char*         getName             (void) const { return "AacBuilder"; }
    void                        configure           (const Config& config);
    void                        execute             (void);

private:
    void                        execDevice          (void);
    void                        execHost            (void);
    void                        mergeClusters       (int leafA, int leafB, int leafC, int clusterLimit);
    float                       evaluateMergeCost   (const AacClusterEntry& a, const AacClusterEntry& b);

private:
                                AacBuilder          (const AacBuilder&); // forbidden
    AacBuilder&                 operator=           (const AacBuilder&); // forbidden

private:
    Config                      m_cfg;

    // Temp buffers.

    BufferRef<int>              m_workCounter;
    BufferRef<int>              m_clusterIndices;
    BufferRef<AacClusterEntry>  m_clusterEntries;
    BufferRef<int>              m_nodePrimRanges;

    // Temps for host-side mergeClusters().

    std::vector<int>                m_tmpIndices;
    std::vector<AacClusterEntry>    m_tmpEntries;
    std::vector<int>                m_tmpPairs;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
