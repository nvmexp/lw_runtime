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
#include "BVH8FitterKernels.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// BVH8Fitter updates the AABB information of an existing array of
// BVH8Nodes to match the given set of primitives. It can also perform
// various optional transformations on the node data.
//
// In addition to the nodes themselves, BVH8Fitter requires the nodeAux
// and remap buffers in the same format as they were produced by
// BVH8Constructor. Although BVH8Fitter modifies the nodeAux buffer in
// the process, it is guaranteed that the fitting operation can be
// exelwted several times using the same buffers. This makes it possible,
// for example, to refit the same BVH for different frames under vertex
// animation.
//
// Optional data transformations:
//
// - Translate item indices by replacing remap[i] with
//   inModel.loadPrimitiveAABB(remap[i]).primitiveIdx.
//
//      This is useful with triangle splitting, where a single primitive
//      of the input model can turn into several primitives. After
//      BVH8Constructor, some entries of remap may point to the split
//      primitives, but we will ultimately want the entries to point to
//      the original primitives of the input model.
//
//      Note: Index translation destroys the original contents of the
//      remap array, making it impossible to re-run BVH8Fitter again on
//      the same data.
//
// - Remove duplicate items within each leaf.
//
//      Useful in conjunction with the index translation described above.
//      With triangle splitting, several AABBs originating from a single
//      model primitive may end up within the same leaf node. After index
//      translation, these AABBs will show up as duplicate entries in the
//      remap array that are not useful as far as the travesal is concerned.
//
//      Note: Duplicate removal destroys the original contents of the
//      remap array, making it impossible to re-run BVH8Fitter again on
//      the same data.
//
// - Reorder the child slots of each node to enable efficient octant-based
//   traversal order.
//
// - Populate an array of BVH8Triangles for efficient access by the
//   traversal kernel.

class BVH8Fitter : public BuildingBlock
{
public:
    struct Config
    {
        LwdaUtils*  lwca;               // Non-NULL => execute on LWCA.
        int         maxPrims;           // Upper bound for the number of input primitives.
        int         maxLeafSize;        // Maximum number of primitives per leaf node. Must match the value used by WideBvhPlanner.
        int         numReorderRounds;   // Number of optimization rounds to perform when reorderChildSlots=True. 3 is sufficient in practice, 2 is not enough.

        bool        translateIndices;   // True => replace indices in ioRemap with their corresponding PrimitiveAABB::primitiveIdx.
        bool        removeDuplicates;   // True => remove duplicates in ioRemap within each leaf, compact remaining entries within each node, and mark unused slots with -1.
        bool        reorderChildSlots;  // True => optimize child slot assignment to enable efficient octant-based traversal order.
        bool        colwertTriangles;   // True => populate outTriangles.
        bool        supportRefit;       // True => enable refitting the same BVH for new AABB data later on.

                                                        // Size                 Description
        BufferRef<BVH8Node>             ioNodes;        // >= maxNodes          Nodes to modify.
        BufferRef<BVH8NodeAux>          ioNodeAux;      // >= maxNodes          Auxiliary per-node data from BVH8NodeConstructor.
        BufferRef<int>                  ioRemap;        // >= maxPrims          Input: User index for each referenced primitive. Output: Depends on the flags.
        BufferRef<BVH8Triangle>         outTriangles;   // = maxPrims or 0      Optional output: Scene triangles colwerted to BVH8Triangle and reordered according to ioRemap.
        BufferRef<>                     tempBuffer;     // = ~4                 Temporary buffer.

        BufferRef<const int>            inNumNodes;     // 1                    Total number of nodes.
        ModelBuffers                    inModel;        // <varies>             Input model.

        Config(void)
        {
            lwca                = NULL;
            maxPrims            = 0;
            maxLeafSize         = 3;
            numReorderRounds    = 3;
            translateIndices    = false;
            removeDuplicates    = false;
            reorderChildSlots   = true;
            colwertTriangles    = false;
            supportRefit        = false;
        }
    };

public:
                        BVH8Fitter          (void) {}
    virtual             ~BVH8Fitter         (void) {}

    virtual const char* getName             (void) const { return "BVH8Fitter"; }
    void                configure           (const Config& config);
    void                execute             (void);

private:
    void                execDevice          (void);
    void                execHost            (void);

private:
                        BVH8Fitter          (const BVH8Fitter&); // forbidden
    BVH8Fitter&         operator=           (const BVH8Fitter&); // forbidden

private:
    Config              m_cfg;

    // Temp buffers.

    BufferRef<int>      m_workCounter;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
