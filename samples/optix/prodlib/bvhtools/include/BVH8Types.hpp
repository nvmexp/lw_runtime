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
#include "Types.hpp"
#include <prodlib/bvhtools/src/common/Intrinsics.hpp> // for INLINE

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// Type definitions for 8-wide BVHs.
// For further details, please refer to:
// https://p4viewer.lwpu.com/get///research/research/hylitie/docs/Efficient-RT%202016-07-09.pptx 

//------------------------------------------------------------------------
// Meta struct contains child node pointers, child node type (inner/leaf) 
// and number of primitives in leaf child nodes.
// Values in bits 5-7:
//      Empty child               -> 0b00000000
//      1 primitive or inner node -> 0b00100000
//      2 primitives              -> 0b01100000
//      3 primitives              -> 0b11100000
// Values in bits 0-4:
//      inner node                -> child slot index + 24
//      leaf node                 -> Index of first triangle (of the leaf) relative to firstRemapIndex

struct BVH8Meta
{
    unsigned char value;

    INLINE void setInner            (int childSlot) { value = (unsigned char)(childSlot + 0x38u); }
    INLINE void setLeaf             (int remapOfs, int numPrims) { value = (unsigned char)(remapOfs + (0xE0602000u >> (numPrims << 3))); }
    INLINE void setEmpty            (void)          { value = 0x00u; }

    INLINE bool isInner             (void) const    { return (value >= 0x38u && value < 0x40u); }
    INLINE bool isLeaf              (void) const    { return (value != 0x00u && (value < 0x38u || value >= 0x40u)); }
    INLINE bool isEmpty             (void) const    { return (value == 0x00u); }

    INLINE int  getInnerChildSlot   (void) const    { return value - 0x38u; }
    INLINE int  getLeafRemapOfs     (void) const    { return value & 0x1Fu; }
    INLINE int  getLeafNumPrims     (void) const    { return __popc(value >> 5); }
};

//------------------------------------------------------------------------
// BVH8NodeHeader contains type and indexing information of child nodes.
// Child slots store child node index offsets from base index of corresponding 
// child node type (inner/leaf). Inner and leaf/primitive remap child nodes are 
// stored continuously and compactly in separate arrays, in same order as child slots.
// Child slots are sorted in Z-order for ordered traversal:
// inner, leaf and empty children can end up in any slot.
//      BVH8Node* childi = nodes[firstChildIdx + getOffset(meta[i])];

// Child node bounding boxes are quantized to 8-bit in coordinate system
// given by BVH8NodeHeader. Uncompressed boxes can be obtained by
// box.lo.x = header.pos[0] + lox * header.scale[0] etc.

struct BVH8NodeHeader // 32 bytes
{
    union {
        struct {
            float           pos[3];
        };
        float3              pos3;
    };
    union {
        struct {
            unsigned char   scale[3];
            unsigned char   innerMask;      // Bitmask of filled inner node children.
        };
        uint32_t            scale_innerMask;
    };
    int             firstChildIdx;  // Index of first child node in subtree.
    int             firstRemapIdx;  // Index of first primitive remap.
    union {
        struct {
            BVH8Meta        meta[8];        // Index offsets and child types for each child slot.
        };
        struct {
            uint32_t        meta4[2];
        };
    };
};

//------------------------------------------------------------------------

struct BVH8Node // 80 bytes
{
    BVH8NodeHeader      header;
    
    // Quantized child bounding boxes for each child slot.
    union {
        struct {
            unsigned char       lox[8];
            unsigned char       loy[8];
            unsigned char       loz[8];
            unsigned char       hix[8];
            unsigned char       hiy[8];
            unsigned char       hiz[8];
        };
        struct {
            unsigned int        lox4[2];
            unsigned int        loy4[2];
            unsigned int        loz4[2];
            unsigned int        hix4[2];
            unsigned int        hiy4[2];
            unsigned int        hiz4[2];
        };
    };
};

//------------------------------------------------------------------------

struct BVH8Triangle // 48 bytes
{
    float v0x, v1x, v2x;
    int userTriangleID;     // Original index of the corresponding triangle specified by the user.

    float v0y, v1y, v2y;
    unsigned int mask;      // User-specified mask. This field is not set by the builder; it's only used internally by TracerBVH8LwdaKernels.

    float v0z, v1z, v2z;
    unsigned int primBits;
};

//------------------------------------------------------------------------
// Additional data buffers needed to trace rays against BVH8, compared to
// the ones already present in TracerDataXxx.

struct TracerDataBVH8
{
    int             numBVH8Nodes;       // Non-zero even if the scene is empty.
    BVH8Node*       bvh8Nodes;

    int             numBVH8Triangles;   // Depends on the caller-triangles-flag: true => zero, false => equal to numRemaps.
    BVH8Triangle*   bvh8Triangles;

    //int           numRemaps;          // Already present in TracerData.
    //int*          remap;              // Already present in TracerData.

    TracerDataBVH8(void)
    {
        numBVH8Nodes        = 0;
        bvh8Nodes           = NULL;
        numBVH8Triangles    = 0;
        bvh8Triangles       = NULL;
    }

    virtual ~TracerDataBVH8() {} // make TracerDataBVH8 a polymorphic type
};

//------------------------------------------------------------------------
// BVH8-specific counterparts of TracerDataMesh and TracerDataGroup.

#ifndef __LWDACC__

struct TracerDataMeshBVH8 : public TracerDataMesh,  public TracerDataBVH8
{
};

struct TracerDataGroupBVH8 : public TracerDataGroup, public TracerDataBVH8
{
  std::vector<TracerDataMeshBVH8> bvh8Meshes;
};

#endif

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
