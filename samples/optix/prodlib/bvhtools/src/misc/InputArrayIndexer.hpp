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
#include "InputArrayIndexerKernels.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// The output buffers can be used to map a global primitive index to an
// input array index and local primitive index within that array.
//
// This building block computes a buffer which uses ~1 bit per primitive to
// reference to the input arrays. The coalesced array consists of N bit blocks
// that encode the transitions between input arrays and 32-bit index per block
// that encodes the input array index at the start of the block.
// The coalescing data structure is built using the following algorithm:
//  - Do a prefix sum on input array lengths into arrayBaseGlobalIndex
//  - Initialize arrayTransitionBits to zero.
//  - The input array boundaries in arrayTransitionBits are tagged with an
//    atomicOr().
//  - Binary search arrayBaseGlobalIndex to find assigned bit block for each
//    thread and set the corresponding input array index in
//    BlockStartArrayIndex.
// At runtime the coalescing data structure is used by looking up the input
// array index and bit block corresponding to the global primitive index
// (primitive index in the conceptually coalesced array). popc() on all the
// bits leading up the global primitive index is added to input array index,
// while globalIndex - arrayBaseGlobalIndex[inputIndex] will give the local
// primitive offset in the local array.
// This approach doesn't work if there are empty arrays among the input.
// To solve that problem, an additional remapping buffer, outGeometryIndexArray,
// is used to colwert an array index to a geometry index, at the cost of an
// extra indirection.
//

class InputArrayIndexer : public BuildingBlock
{
public:
    struct Config
    {
        LwdaUtils*              lwca;                     // Non-NULL => execute on LWCA.
        const InputBuffers*     inBuffers;

                                                          // Size                 Description
        BufferRef<unsigned int> outArrayBaseGlobalIndex;  // = numArrays+1        Prefix sum over input array lengths
        BufferRef<unsigned int> outArrayTransitionBits;   // ~ ceil(numPrims/32)  Global primitive index -> Transitions in the input array
        BufferRef<int>          outBlockStartArrayIndex;  // ~ ceil(numPrims/128) Input array index at start of 128-bit block
        BufferRef<int>          outGeometryIndexArray;    // = numArrays          Only needed if there were empty input geometries

        Config(void)
        {
            lwca      = NULL;
            inBuffers = NULL;
        }
    };

public:
                            InputArrayIndexer       (void) {}
    virtual                 ~InputArrayIndexer      (void) {}

    virtual const char*     getName                 (void) const { return "InputArrayIndexer"; }
    void                    configure               (const Config& config);
    void                    execute                 (void);

private:
    void                    execDevice              (void);
    void                    execHost                (void);

private:
                            InputArrayIndexer       (const InputArrayIndexer&); // forbidden
    InputArrayIndexer&      operator=               (const InputArrayIndexer&); // forbidden

private:
    Config                  m_cfg;
    int                     m_numBlocks;

};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
