// Copyright LWPU Corporation 2014
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
#include "../BuildingBlock.hpp"
#include "../common/TypesInternal.hpp"

namespace prodlib
{
namespace bvhtools
{
  //------------------------------------------------------------------------
  // Trims off nodes at or above a given cutoff level and marks them as
  // unused. Any trimmed leaves and exposed treelet roots are stored in another array
  // as PrimitiveAABBs from which the top level tree can be built. Also writes
  // the prim range for the top-level tree.

  class TreeTopTrimmer : public BuildingBlock
  {
  public:
    struct Config
    {
      LwdaUtils*    lwca;       // Non-NULL => execute on LWCA.
      int           maxNodes;   // Max possible size of nodeRange.

                                                    // Size                 Description
      BufferRef<PrimitiveAABB>  ioTrimmed;          // >= 2^(lwtoffLevel+1) Where to store trimmed node AABBs.
      BufferRef<Range>          ioTrimmedRange;     // 1                    Range of trimmed node AABBs.
      BufferRef<BvhNode>        ioNodes;            // >= maxNodes          Node array. Trimmed nodes marked unused.
      BufferRef<const Range>    inNodeRange;        // 1                    Node range for the tree to be trimmed.
      BufferRef<const int>      inLwtoffLevel;      // 1                    Level at or below which nodes are trimmed.
      BufferRef<const int>      inNodeParents;      // >= maxNodes          Parent pointer for each node, complemented if the node is the right child. The indexing is relative to inNodeRange.start.

      Config(void)
      {
        lwca = NULL;
        maxNodes = 0;
      }
    };

  public:
                        TreeTopTrimmer               (void) {}
    virtual             ~TreeTopTrimmer              (void) {}

    virtual const char* getName                      (void) const { return "TreeTopTrimmer"; }
    void                configure                    (const Config& cfg);
    void                execute                      (void);

  private:
    void                execDevice                   (void);
    void                execHost                     (void);

  private:
    TreeTopTrimmer                                   (const TreeTopTrimmer&); // forbidden
    TreeTopTrimmer&    operator=                     (const TreeTopTrimmer&); // forbidden

  private:
    Config              m_cfg;
  };

  //------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
