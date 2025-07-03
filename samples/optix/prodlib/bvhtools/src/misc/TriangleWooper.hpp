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
#include "../BuildingBlock.hpp"
#include "../common/TypesInternal.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// Building block that:
// - Initializes per-triangle data for Sven Woop's intersection test.
// - Sets the LSB of WoopTriangle.t.w for complemented remap entries.
// - Optionally un-complements the remap entries.
//
// Note: If inNumRemaps=0, TriangleWooper still outputs one dummy
// triangle at outWoop[0]. This triangle cannot be hit during traversal,
// so it can safely referenced by the root node of an empty BVH.

class TriangleWooper : public BuildingBlock
{
public:
    struct Config
    {
        LwdaUtils*              lwca;               // Non-NULL => execute on LWCA.
        bool                    uncomplementRemaps; // True => un-complement remap entries.

                                                    // Size             Description
        BufferRef<WoopTriangle> outWoop;            // = maxNumRemaps   Output Woop data for each entry in the remapping table.

        BufferRef<int>          ioRemap;            // maxNumRemaps     Triangle remapping table. One entry per triangle reference.
        BufferRef<const int>    inNumRemaps;        // 1                Pointer to an integer that tells the size of the remapping table.
        ModelBuffers            inModel;            // <varies>         Input model.

        Config(void)
        {
            lwca                = NULL;
            uncomplementRemaps  = false;
        }
    };

public:
                        TriangleWooper      (void) {}
    virtual             ~TriangleWooper     (void) {}

    virtual const char* getName             (void) const { return "TriangleWooper"; }
    void                configure           (const Config& config);
    void                execute             (void);

private:
    void                execDevice          (void);
    void                execHost            (void);

private:
                        TriangleWooper      (const TriangleWooper&); // forbidden
    TriangleWooper&     operator=           (const TriangleWooper&); // forbidden

private:
    Config              m_cfg;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
