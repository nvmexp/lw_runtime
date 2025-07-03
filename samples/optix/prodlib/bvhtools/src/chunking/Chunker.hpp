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

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// The Chunker partitions the list of primitives into chunks that will fit within
// the given size. Descending from the root of the radix tree built on the sorted
// morton codes of the primitives, the first nodes containing no more 
// than chunkPrims primitives become the roots of each chunk.

class Chunker : public BuildingBlock
{
public:
    struct Config
    {
        LwdaUtils*                      lwca;                   // Non-NULL => execute on LWCA.
        int                             numPrims;               // Number of input primitives.
        int                             maxChunkPrims;          // Maximum number of primitives in a chunk
        int                             preferredTopTreePrims;  // Preferred total number of primitives in the top level tree. See computeLwtoffLevel() for details.

                                                                // Size                 Description
        BufferRef<Range>                outPrimRanges;          // = <maxChunks>        Range of primitives for each chunk in inPrimMorton.
        BufferRef<int>                  outNumChunks;           // = 1                  Number of chunks generated
        BufferRef<int>                  outLwtoffLevel;         // = 1                  Cutoff level at or below which chunks are trimmed. Chosen based on preferredTopTreePrims.
        BufferRef<PrimitiveAABB>        allocTrimmedAABBs;      // = <maxTopTreePrims>  Resized to the maximum number of primitives in the top level tree. The data is left uninitialized.
        BufferRef<unsigned long long>   inPrimMorton;           // >= numPrims          Sorted Morton codes of the input primitives.

        Config(void)
        {
            lwca                    = NULL;
            numPrims                = 0;
            maxChunkPrims           = 0;
            preferredTopTreePrims   = 0;
        }
    };

public:
                        Chunker             (void) {}
    virtual             ~Chunker            (void) {}

    virtual const char* getName             (void) const { return "Chunker"; }
    void                configure           (const Config& cfg);
    void                execute             (void);

private:
    void                execDevice          (void);
    void                execHost            (void);

private:
                        Chunker             (const Chunker&); // forbidden
    Chunker&            operator=           (const Chunker&); // forbidden

private:
    Config              m_cfg;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
