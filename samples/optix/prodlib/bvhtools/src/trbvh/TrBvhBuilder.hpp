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

#pragma once
#include "TrBvhBuilderKernels.hpp"
#include <prodlib/bvhtools/src/BuildingBlock.hpp>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// Building block implementing the BVH construction method described in
// the following paper:
//
//   "Fast Parallel Construction of High-Quality Bounding Volume Hierarchies"
//   Tero Karras, Timo Aila
//   High Performance Graphics 2013
//
// Builds a BVH from input primitive AABBs.
//
// Note: If numPrims=1, TrbvhBuilder creates a single root node that
// references the same primitive twice, first with the true AABB and
// then with an unhittable AABB. If numPrims=0, the root node
// references a dummy primitive in the same way. In this case, ioRemap is
// initialized for the dummy primitive as if it was a real one, but
// ioNumRemaps is set to 0.

class TrbvhBuilder : public BuildingBlock
{
public:
    struct Config
    {
        LwdaUtils*              lwca;                   // Non-NULL => execute on LWCA.
        int                     maxPrims;               // Upper bound for the number of input primitives.
        RemapListLenEncoding    listLenEnc;             // How the primitive list length is encoded in remap

        float                   sahNodeCost;            // Cost of two ray-box tests (C_i).
        float                   sahPrimCost;            // Cost of one ray-primitive test (C_t).
        int                     maxLeafSize;            // Maximum number of primitives in a leaf.
        int                     optTreeletSize;         // Number of treelet leaves. Between 5 and 8.
        int                     optRounds;              // Number of optimization rounds.
        int                     optGamma;               // Minimum subtree size for forming a treelet.
        bool                    optAdaptiveGamma;       // Double the value of gamma after each round?

                                                            // Size                     Description
        BufferRef<int>                  outNodeParents;     // = max(maxPrims-1,1)      Parent pointer for each node, complemented if the node is the right child. The indexing is relative to outNodeRange.start.
        BufferRef<Range>                outNodeRange;       // = 1                      Range of output nodes produced. Starts at the previous value of ioNumNodes.
        BufferRef<>                     tempBuffer;         // = ~0                     Temporary buffer.

        BufferRef<BvhNode>              ioNodes;            // >= max(maxPrims-1,1)     Output node array. Not all nodes will be used and will be marked as unused.
        BufferRef<int>                  ioNumNodes;         // 1                        Cumulative number of nodes in the ioNodes buffer. Must be cleared to zero from the outside.
        BufferRef<int>                  ioRemap;            // >= max(maxPrims,1)       Primitive remapping table. One entry per resulting primitive reference.
        BufferRef<int>                  ioNumRemaps;        // 1                        Cumulative number of entries in ioRemap. Must be cleared to zero from the outside.
        BufferRef<unsigned long long>   ioMortonCodes;      // >= maxPrims              Sorted Morton codes corresponding. Replaced with garbage by execute().
        BufferRef<const int>            inPrimOrder;        // >= maxPrims              Indices of the input primitives in Morton order.
        BufferRef<const Range>          inPrimRange;        // 0 or 1                   Range of input primitives in inPrimOrder. EmptyBuf => (0, maxPrims).
        ModelBuffers                    inModel;            // <varies>                 Input model.
        
        Config(void)
        {
            lwca                    = NULL;
            maxPrims                = 0;
            listLenEnc              = RLLE_NONE;

            sahNodeCost             = 1.0f;     // Original paper: 1.2f
            sahPrimCost             = 1.8f;     // Original paper: 1.0f
            maxLeafSize             = 7;        // Original paper: 8

            optTreeletSize          = 7;
            optRounds               = 3;
            optGamma                = 7;
            optAdaptiveGamma        = true;
        }
    };

public:
                            TrbvhBuilder        (void) {}
    virtual                 ~TrbvhBuilder       (void) {}

    virtual const char*     getName             (void) const { return "TrbvhBuilder"; }
    void                    configure           (const Config& cfg);
    void                    execute             (void);

private:
    void                    execDevice          (void);
    void                    execHost            (void);

private:
                            TrbvhBuilder        (const TrbvhBuilder&); // forbidden
    TrbvhBuilder&           operator=           (const TrbvhBuilder&); // forbidden

private:
    Config                  m_cfg;

    // Temp buffers.

    BufferRef<int>          m_lwrRound;
    BufferRef<int>          m_workCounter;
    BufferRef<TrbvhNodeCosts> m_dummyNodeCosts; // Only used when maxPrims = 0.
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
