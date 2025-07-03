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
#include <prodlib/bvhtools/src/common/TypesInternal.hpp>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// For each i in [0,numRemaps) : 
//    outPrimBits[i] = primBits[remap[i]]    - if primBits not null and remap[i] < numPrims
//    outPrimBits[i] = remap[i]              - if primBits is null
//

class GatherPrimBits : public BuildingBlock
{
public:
    struct Config
    {
        LwdaUtils*            lwca;               // Non-NULL => execute on LWCA.
        int                   numRemaps;          // Number of entries in inRemap
        int                   numPrims;           // Number of primitives
        uint64_t              flags;              // Flags that will be or'd together with every entry in the primBits buffer
        PrimBitsFormat        primBitsFormat;     // Which primBits format to encode to

        bool                  useBufferOverlay;   // True = primitive bits are placed within an existing buffer (in BVH8Triangle buffer for example)
        int                   inPrimBitsStride;   // Stride in bytes between primBits inputs
        int                   inPrimBitsOffset;   // Byte offset to first primBits intput
        int                   outPrimBitsStride;  // Stride in bytes between primBits outputs
        int                   outPrimBitsOffset;  // Byte offset to first primBits output

                                                  // Size             Description
        BufferRef<>           outPrimBitsRaw;     // = maxPrims       Output primitive index for each referenced primitive.
        BufferRef<>           inPrimBitsRaw;      // = maxPrims       Input primitive index for each referenced primitive.
        BufferRef<const int>  inRemap;            // = maxPrims       Remap buffer.

        Config(void)
        {
            lwca                  = NULL;
            flags                 = 0;
            useBufferOverlay      = false;
            inPrimBitsOffset      = 0;
            inPrimBitsStride      = sizeof( int );
            outPrimBitsOffset     = 0;
            outPrimBitsStride     = sizeof( int );
            primBitsFormat        = PRIMBITS_NONE;
        }
    };

public:
                            GatherPrimBits          (void) {}
    virtual                 ~GatherPrimBits         (void) {}

    virtual const char*     getName                 (void) const { return "GatherPrimBits"; }
    void                    configure               (const Config& config);
    void                    execute                 (void);

private:
    void                    execDevice              (void);
    void                    execHost                (void);

private:
                            GatherPrimBits          (const GatherPrimBits&); // forbidden
    GatherPrimBits&         operator=               (const GatherPrimBits&); // forbidden

private:
    Config                  m_cfg;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
