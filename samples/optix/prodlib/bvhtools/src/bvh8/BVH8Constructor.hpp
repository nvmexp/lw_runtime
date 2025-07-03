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

#pragma once
#include <prodlib/bvhtools/src/BuildingBlock.hpp>
#include "BVH8ConstructorKernels.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// BVH8Constructor constructs an array of BVH8Nodes based on the output
// of WideBvhPlanner. Only a subset of the BVH8Node fields are
// initialized by BVH8Constructor; the remaining fields are intended to be
// filled in by BVH8Fitter:
//
// - BVH8NodeHeader::firstChildIdx
// - BVH8NodeHeader::firstRemapIdx
// - BVH8NodeHeader::meta
//
// The root node is always placed at index 0 in the outNodes array.
// Note that in case of vertex animation, BVH8Constructor needs to be
// exelwted only once during the initial BVH build. The output node array
// can then be reused over several animation frames by re-exelwting
// BVH8Fitter and to refit the AABBs and triangles to the updated vertex data.

class BVH8Constructor : public BuildingBlock
{
public:
    struct Config
    {
        LwdaUtils*  lwca;                   // Non-NULL => execute on LWCA.
        int         maxPrims;               // Upper bound for the number of input primitives.
        int         maxNodes;               // Upper bound for the number of nodes. Must not be lower than the value returned by WideBvhPlanner::getMaxWideNodes().
        int         maxBranchingFactor;     // Maximum number of children per node. Must match the value used by WideBvhPlanner.
        int         maxLeafSize;            // Maximum number of primitives per leaf node. Must match the value used by WideBvhPlanner.

                                                            // Size             Description
        BufferRef<BVH8Node>                 outNodes;       // = maxNodes       Output nodes. Special allocation policy; see above.
        BufferRef<BVH8NodeAux>              outNodeAux;     // = maxNodes       Auxiliary per-node data for BVH8Fitter. Special allocation policy; see above.
        BufferRef<int>                      outRemap;       // = maxPrims       Primitive index for each referenced primitive.
        BufferRef<>                         tempBuffer;     // = ~4             Temporary buffer.

        BufferRef<const int>                inNumNodes;     // 1                From WideBvhPlanner: Total number of nodes.
        BufferRef<const PlannerBinaryNode>  inBinaryNodes;  // >= 2             From WideBvhPlanner: Binary BVH nodes annotated with grouping info about the planned nodes.

        Config(void)
        {
            lwca                = NULL;
            maxPrims            = 0;
            maxBranchingFactor  = 8;
            maxLeafSize         = 3;
        }
    };

public:
                            BVH8Constructor         (void) {}
    virtual                 ~BVH8Constructor        (void) {}

    virtual const char*     getName                 (void) const { return "BVH8Constructor"; }
    void                    configure               (const Config& config);
    void                    execute                 (void);

private:
    void                    execDevice              (void);
    void                    execHost                (void);

private:
                            BVH8Constructor         (const BVH8Constructor&); // forbidden
    BVH8Constructor&        operator=               (const BVH8Constructor&); // forbidden

private:
    Config                  m_cfg;

    // Temp buffers.

    BufferRef<int>          m_nodeCounter;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
