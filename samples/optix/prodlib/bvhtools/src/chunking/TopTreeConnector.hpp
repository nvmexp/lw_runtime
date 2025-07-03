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
  // TreeTopTrimmer trims off the top-level nodes in each chunk and stores
  // the leaves and exposed treelet roots as PrimitiveAABBs, packing the links
  // back to the original data in an unused field of PrimitiveABBB. After building
  // the top-level tree from these PrimitiveAABBs the leaves of the tree, this
  // building block reconnected the links back to the original data. The the 
  // root of the top-level tree is also written to a designated location.
  
  class TopTreeConnector : public BuildingBlock
  {
  public:
    struct Config
    {
      LwdaUtils*    lwca;       // Non-NULL => execute on LWCA.
      int           maxNodes;   // Max possible size of nodeRange.

                                                        // Size             Description
      BufferRef<BvhNode>                outRoot;        // = 1              Location for the resulting root node.
      BufferRef<BvhNode>                ioNodes;        // >= maxNodes      Node array.
      BufferRef<const Range>            inNodeRange;    // 1                Node range for the top-level tree. 
      BufferRef<const int>              inRemap;        // >= maxNodes+1    Maps from leaf index in BVH to original leaf index.
      BufferRef<const PrimitiveAABB>    inTrimmedAabbs; // >= maxNodes+1    Aabbs used for the top-level build.

      Config(void)
      {
        lwca = NULL;
        maxNodes = 0;
      }
    };

  public:
                        TopTreeConnector    (void) {}
    virtual             ~TopTreeConnector   (void) {}

    virtual const char* getName             (void) const { return "TopTreeConnector"; }
    void                configure           (const Config& cfg);
    void                execute             (void);

  private:
    void                execDevice          (void);
    void                execHost            (void);

  private:
                        TopTreeConnector    (const TopTreeConnector&); // forbidden
    TopTreeConnector&   operator=           (const TopTreeConnector&); // forbidden

  private:
    Config              m_cfg;
  };

  //------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
