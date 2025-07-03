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
#include <prodlib/bvhtools/include/Types.hpp>

namespace prodlib
{
namespace bvhtools
{

//------------------------------------------------------------------------
// TriangleAdapter takes multiple vertex and index buffer inputs with any
// of the formats listed in InputTrianglesDesc and colwerts them to a
// single flattened and transformed float3 buffer.
//

class TriangleAdapter : public BuildingBlock
{
public:
    struct Config
    {
        bool                useLwda;                              // Perform build operation on device
        LwdaUtils*          lwdaUtils;                            // Must be non-null if useLwda is True
        const InputBuffers* inBuffers;                            // Input model buffers
        int                 primitiveIndexBits;                   // Number of bits allocated for primitive index (4 byte primBits format only)
        bool                computePrimBits;                      // True if outPrimBits should be populated
        PrimBitsFormat      primBitsFormat;                       // Which primBits format to encode to
        bool                refitOnly;                            // True if is a refit

                                                                  // Size                 Description
        BufferRef<float3>                 outVertices;            // = numPrims           Flattened vertex array
        BufferRef<>                       outPrimBits;            // = numPrims           Output primBits buffer. Should be NULL if computePrimBits is False

        BufferRef<InputTrianglesDesc>     outTrianglesDescArray;  // = numInputs          Device side triangle descriptor arrays
        BufferRef<InputTrianglesPointers> outTrianglesPtrArray;   // = numInputs          Device side triangle pointer array

        InputArrayIndexBuffers            inArrayIndexing;        // = varies             Global primitive index -> Input array index, geometry index and local primitive index

        Config(void)
        {
            useLwda               = false;
            lwdaUtils             = NULL;
            inBuffers               = NULL;
            outTrianglesDescArray   = EmptyBuf;
            outTrianglesPtrArray    = EmptyBuf;
            computePrimBits         = false;
        }
    };

public:
                            TriangleAdapter         (void) {}
    virtual                 ~TriangleAdapter        (void) {}

    virtual const char*     getName                 (void) const { return "TriangleAdapter"; }
    void                    configure               (const Config& cfg);
    void                    execute                 (void);

private:
    void                    execDevice              (void);
    void                    execHost                (void);

private:
                            TriangleAdapter         (const TriangleAdapter&); // forbidden
    TriangleAdapter&        operator=               (const TriangleAdapter&); // forbidden

private:
    Config                  m_cfg;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
