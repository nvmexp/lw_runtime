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
#include <prodlib/bvhtools/src/common/TypesInternal.hpp>
#include <prodlib/bvhtools/src/BuildingBlock.hpp>

#include <prodlib/bvhtools/src/misc/InputArrayIndexer.hpp>
#include <prodlib/bvhtools/src/bounds/InstanceDataAdapter.hpp>
#include <prodlib/bvhtools/src/bounds/AabbAdapter.hpp>
#include <prodlib/bvhtools/src/bounds/TriangleAdapter.hpp>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// The InputAdapter is responsible for taking the input data found in
// InputBuffer and colwerting it to a format that subsequent BuildingBlocks
// can consume. The InputAdapter also computes which format the primBits-
// buffer should use.
//

class InputAdapter : public BuildingBlock
{
public:
    struct Config
    {
        bool                useLwda;                // Perform build operation on device
        LwdaUtils*          lwdaUtils;              // Must be non-null if useLwda is True
        const InputBuffers* inBuffers;              // Input model buffers

        bool                usePrimBits;            // Output primBits
        bool                refitOnly;              // True if is a refit
        int                 motionSteps;            // For aabb inputs only: number of copies of each aabb

        PrimBitsFormat      allowedPrimBitsFormats; // Flags specifying which primBits formats to consider

                                                                      // Size                                                         Description
        BufferRef<unsigned int>             outArrayBaseGlobalIndex;  // ~ numArrays+1                                                Global primitive index at the beginning of each input array
        BufferRef<unsigned int>             outArrayTransitionBits;   // ~ ceil(numPrims/32)                                          Global primitive index -> Transitions in the input array
        BufferRef<int>                      outBlockStartArrayIndex;  // ~ ceil(numPrims/128)                                         Input array index at start of 128-bit block
        BufferRef<int>                      outGeometryIndexArray;    // ~ numArrays, IF there are empty input arrays                 Only needed if there were empty input geometries
        BufferRef<const unsigned int>       outOpaqueFlags;           // ~ ceil(numArrays/32)                                         Geometry opaque flags, 1 bit per item
        BufferRef<PrimitiveAABB>            outAabbs;                 // ~ 16  bytes * maxSplitPrims, IF m_enableSplits == true       Flattened PrimitiveAABB array
        BufferRef<>                         outTmpPrimBits;           // ~ 4 or 8 bytes * maxSplitPrims                               Temporary primBits buffer. The final primBits will be written in either TTUTriangleCompressor, GatherPrimBits, or to outIndirectPrimBits.
        BufferRef<>                         outIndirectPrimBits;      // ~ 8 bytes * maxSplitPrims                                    If primBits format is INDIRECT_32_TO_64, the primBits are directly written to its final buffer location.
        BufferRef<BvhInstanceData>          outBvhInstanceData;       // ~ 112 bytes * numInputPrims, IF m_inputType == IT_INSTANCE   Instance data buffer
        BufferRef<InputAABBs>               outAabbArray;             // ~ 16 bytes * numInputs, IF m_inputType == IT_AABB            Device side AABB arrays
        BufferRef<InputTrianglesDesc>       outTriangleDescArray;     // ~ 16 bytes * numInputs, IF m_inputType == IT_TRI             Device side triangle descriptor arrays
        BufferRef<InputTrianglesPointers>   outTrianglePtrArray;      // ~ 24 bytes * numInputs, IF m_inputType == IT_TRI             Device side triangle pointer arrays
        BufferRef<float3>                   outCoalescedVertices;     // ~ 12 bytes * numPrims                                        Flattened vertex array

        Config(void)
        {
            useLwda             = false;
            lwdaUtils           = NULL;
            inBuffers           = NULL;
            outIndirectPrimBits = EmptyBuf;
            motionSteps         = 1;
        }
    };

public:
                            InputAdapter            (void) {}
    virtual                 ~InputAdapter           (void) {}

    virtual const char*     getName                 (void) const { return "InputAdapter"; }
    void                    configure               (const Config& config);
    ModelBuffers            getModel                (void) const { return m_model; }
    bool                    isCoalesced             (void) const { return m_coalesced; }
    int                     getPrimBitsSizeInBytes  (void) const { return m_primBitsFormat == PRIMBITS_NONE ? 0 : ( m_primBitsFormat == PRIMBITS_DIRECT_32 || m_primBitsFormat == PRIMBITS_LEGACY_DIRECT_32 ? 4 : 8 ); }
    PrimBitsFormat          getPrimBitsFormat       (void) const { return m_primBitsFormat; }
    int                     getNumPrimitiveIndexBits(void) const { return m_numPrimitiveIndexBits; }
    int                     getPrimBitsFlags        (void) const { return m_primBitsFlags; }
    void                    execute                 (void);

private:
                            InputAdapter            (const InputAdapter&); // forbidden
    InputAdapter&           operator=               (const InputAdapter&); // forbidden

private:
    Config                  m_cfg;
    ModelBuffers            m_model;
    bool                    m_coalesced;
    PrimBitsFormat          m_primBitsFormat;
    int                     m_numPrimitiveIndexBits;
    int                     m_primBitsFlags;

    InputArrayIndexer       m_inputArrayIndexer;
    InstanceDataAdapter     m_instanceDataGen;
    AabbAdapter             m_aabbAdapter;
    TriangleAdapter         m_triangleAdapter;

};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
