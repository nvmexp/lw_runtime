// Copyright LWPU Corporation 2017
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
// Building block that refits an existing set of bvh nodes using new motion bounding boxes.
//

class MotionRefitter : public BuildingBlock
{
public:
    struct Config
    {
        LwdaUtils*              lwca;               // Non-NULL => execute on LWCA.

        int                     numAabbs;
        int                     motionSteps;
        unsigned int            maxNodes;
                                                // Size                    Description
        BufferRef<BvhHeader>     ioBvh;          // =1                     BVH header needed for refit
        BufferRef<BvhNode>       ioAllNodes;     // =maxNodes*motionSteps  BVH nodes as a BufferRef; colwenient for copying
        BufferRef<InputAABBs>    inAabbs;        // =numAabbs*motionSteps  AABBs for all motion steps
        BufferRef<int>           ioNodeParents;  // =maxNodes              parent indices

        Config(void)
        {
            lwca                = NULL;
        }
    };

public:
                        MotionRefitter      (void) {}
    virtual             ~MotionRefitter     (void) {}

    virtual const char* getName             (void) const { return "MotionRefitter"; }
    void                configure           (const Config& config);
    void                execute             (void);

private:
    void                execDevice          (void);
    void                execHost            (void);

private:
                        MotionRefitter      (const MotionRefitter&); // forbidden
    MotionRefitter&     operator=           (const MotionRefitter&); // forbidden

private:
    Config              m_cfg;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
