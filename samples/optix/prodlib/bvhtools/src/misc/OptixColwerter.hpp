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
#include <prodlib/bvhtools/src/BuildingBlock.hpp>
#include <prodlib/bvhtools/src/common/TypesInternal.hpp>
#include <prodlib/bvhtools/src/bounds/ApexPointMap.hpp>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// Prepares BVH for consumption by OptiX.
//  - colwerts prodlib::bvhtools::BvhNode to optix::BvhNode (half nodes). 
//  - inserts a dummy root node at position 0.
//  - swaps idx/num for internal nodes.
//  - premultiplies internal node child indices by 2 or 4.
//  - colwerts leaf idx/num to begin/end.
//  In addition, OptixColwerter reorders the remapping table so that
//  the primitive lists of sibling nodes are adjacent and ordered (left 
//  before right).

class OptixColwerter : public BuildingBlock
{
public:
    struct Config
    {
        LwdaUtils*  lwca;           // Non-NULL => execute on LWCA.
        int         maxNodes;       // Maximum possible value of inNumNodes
        bool        bake;           // Bake in the extra *2 used by OptiX traversers (for float4 accesses) // TODO [tkarras]: This is always true; remove.
        bool        shiftNodes;     // If true, nodes are shifted up one position to make room for the root at index 0. Otherwise the node is presumed to be free. // TODO [tkarras]: This is always true; remove.

                                                        // Size             Description
        BufferRef<int>                  outRemap;       // = maxNumRemaps   Primitive remapping table. Siblings have adjacent primitive lists.
        BufferRef<>                     tempBuffer;     // = ~8*maxNodes    Temporary buffer

        BufferRef<BvhNode>              ioNodes;        // >= maxNodes      Output node array. May contain unused nodes. Root is at index 0/1 if shiftNodes=true/false. Adds a dummy root.
        BufferRef<const int>            inNumNodes;     // 1                Pointer to an integer that tells the number of nodes.
        BufferRef<const int>            inRemap;        // maxNumRemaps     Primitive remapping table. One entry per resulting primitive reference.
        BufferRef<const ApexPointMap>   inApexPointMap; // 1                Produced by ApexPointMapConstructor.

        Config(void)
        {
            lwca        = NULL;
            maxNodes    = 0;
            bake        = true;
            shiftNodes  = true;
        }
    };

public:
                        OptixColwerter      (void) {}
    virtual             ~OptixColwerter     (void) {}

    virtual const char* getName             (void) const { return "OptixColwerter"; }
    void                configure           (const Config& cfg);
    void                execute             (void);

private:
    void                execDevice          (void);
    void                execHost            (void);
    void                execHostWithShift   (void);
    void                execHostWithoutShift(void);

    
    int                 getScale(); // Scale factor for interior node child indices

private:
                        OptixColwerter      (const OptixColwerter&); // forbidden
    OptixColwerter&     operator=           (const OptixColwerter&); // forbidden

private:
    Config              m_cfg;

    // Temp buffers.

    BufferRef<int>      m_leafSize;
    BufferRef<int>      m_leafPos;
    BufferRef<int>      m_blockCount;
    BufferRef<char>     m_scanTemp;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
