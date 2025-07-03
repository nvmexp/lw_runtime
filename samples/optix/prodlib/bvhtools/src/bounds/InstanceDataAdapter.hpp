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
#include <prodlib/bvhtools/include/Types.hpp>
#include "ApexPointMap.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// InstanceDataAdapter takes the data associated with instances and produces
// PrimitiveAABBs for builders to consume, as well as assembling side-data
// required during BVH traversal. Specifically, it takes a set of object-to-world 
// space transformation matrices and associated object-space apex point maps,
// and produces a set of world-to-object space transformation matrices
// along with the world-space AABBs for the objects. The world-space AABBs are
// constructed by querying the object-space apex point maps in directions
// that correspond to world-space coordinate axes after the transformation.

class InstanceDataAdapter : public BuildingBlock
{
public:
    struct Config
    {
        bool          useLwda;        // Perform build operation on device
        LwdaUtils*    lwdaUtils;      // Must be non-null if useLwda is True
        int           numInstances;   // Number of geometry instances.
        int           matrixStride;   // Number of bytes between conselwtive matrices in outIlwMatrices and inTransforms. Must be divisible by 4, and at least 48.
        bool          computeAabbs;   // FIXME: We shouldn't need this, but we lwrrently don't have a way to determine if a BufferRef is pointing to EmptyBuf.
        
                                                                    // Size                                Description
        BufferRef<BvhInstanceData>              outBvhInstanceData; // = numInstances                      BvhInstanceData containing ilwerse transform. Only used when inInstanceData != NULL                                                    
        BufferRef<PrimitiveAABB>                outWorldSpaceAabbs; // = numInstances                      Conservative per-instance world-space AABBs.
        BufferRef<float>                        outIlwMatrices;     // = numInstances*(matrixStride/4)     Ilwerted per-instance transformation matrices (world-to-object, 3x4, row-major).

        BufferRef<const InstanceDesc>           inInstanceDescs;    // = numInstances                      InstanceDescs containing object-to-world transform. All other inputs ignored if this is non-NULL                                     
        BufferRef<const ApexPointMap* const>    inApexPointMaps;    // <varies>                            Per-object ApexPointMaps in object-space.
        BufferRef<const float>                  inTransforms;       // >= numInstances*(matrixStride/4)    Per-instance transformation matrices (object-to-world, 3x4, row-major).
        BufferRef<const int>                    inInstanceIds;      // >= numInstances                     Per-instance IDs to place in outWorldSpaceAabbs.
        
        Config(void)
        {
            useLwda         = false;
            lwdaUtils       = NULL;
            numInstances    = 0;
            matrixStride    = 0;
            computeAabbs    = true;
        }
    };

public:
                            InstanceDataAdapter     (void) {}
    virtual                 ~InstanceDataAdapter    (void) {}

    virtual const char*     getName                 (void) const { return "InstanceDataAdapter"; }
    void                    configure               (const Config& cfg);
    void                    execute                 (void);

private:
    void                    execDevice              (void);
    void                    execHost                (void);

private:
                            InstanceDataAdapter     (const InstanceDataAdapter&); // forbidden
    InstanceDataAdapter&    operator=               (const InstanceDataAdapter&); // forbidden

private:
    Config                  m_cfg;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
