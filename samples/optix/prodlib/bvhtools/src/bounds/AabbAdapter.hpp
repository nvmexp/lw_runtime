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
#include <prodlib/bvhtools/src/BuildingBlock.hpp>
#include <prodlib/bvhtools/include/Types.hpp>
#include <prodlib/bvhtools/src/common/TypesInternal.hpp>

namespace prodlib
{
namespace bvhtools
{
struct AABB;

//------------------------------------------------------------------------
// AabbAdapter takes pure AABBs and colwerts them into PrimitiveAABBs 
// for consumption by other BuildingBlocks. Multiple arrays of AABBs will
// be coalesced in to a single buffer, and the global index of each AABB
// is used as its primitive index.
//

class AabbAdapter : public BuildingBlock
{
public:
    struct Config
    {
        bool                useLwda;                    // Perform build operation on device
        LwdaUtils*          lwdaUtils;                  // Must be non-null if useLwda is True
        const InputBuffers* inBuffers;                  // Input model buffers
        int                 primitiveIndexBits;         // Number of bits allocated for primitive index (4 byte primBits format only)
        PrimBitsFormat      primBitsFormat;             // Which primBits format to encode to
        bool                computePrimBits;            // True if outPrimBits should be populated
        int                 motionSteps;

                                                        // Size                 Description
        BufferRef<PrimitiveAABB>      outPrimAabbs;     // = numAabbs           Flattened PrimitiveAABB array
        BufferRef<>                   outPrimBits;      // = numAabbs           Output primBits buffer. Should be NULL if computePrimBits is False

        BufferRef<InputAABBs>         outAabbArray;     // = numInputs          Device side aabb descriptor arrays

        InputArrayIndexBuffers        inArrayIndexing;  // = varies             Global primitive index -> Input array index, geometry index and local primitive index

        Config(void)
        {
            useLwda               = false;
            lwdaUtils             = NULL;
            inBuffers             = NULL;
            outAabbArray          = EmptyBuf;
            computePrimBits       = false;
            motionSteps           = 1;
        }
    };

public:
                            AabbAdapter             (void) {}
    virtual                 ~AabbAdapter            (void) {}

    virtual const char*     getName                 (void) const { return "AabbAdapter"; }
    void                    configure               (const Config& cfg);
    void                    execute                 (void);

private:
    void                    execDevice              (void);
    void                    execHost                (void);

private:
                            AabbAdapter             (const AabbAdapter&); // forbidden
    AabbAdapter&            operator=               (const AabbAdapter&); // forbidden

private:
    Config                  m_cfg;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
