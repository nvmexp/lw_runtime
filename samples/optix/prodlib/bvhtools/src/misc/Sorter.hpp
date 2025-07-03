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

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// Sorts the given set of key/value pairs in ascending order.
//
// Input:
// - The keys and values are specified in separate buffers.
// - The number of key/value pairs of the buffers is fixed at configure().
// - Keys can be 32-bit or 64-bit unsigned ints.
// - Values must be 32-bit ints/floats/structs.
//
// Output:
// - The input buffers are left untouched
// - The sort is stable; so it will not reorder identical keys.

class Sorter : public BuildingBlock
{
public:
    struct Config
    {
        LwdaUtils*  lwca;           // Non-NULL => execute on LWCA.
        int         numItems;       // Total number of key/value pairs.
        size_t      bytesPerKey;    // Key size in bytes. Must be 4 or 8.
        size_t      bytesPerValue;  // Value size in bytes. Must be 4.

                                    // Size                                                 Description
        BufferRef<> outKeys;        // >= numItems * bytesPerKey                            Output key array.
        BufferRef<> outValues;      // >= numItems * bytesPerValue                          Output value array.
        BufferRef<> inKeys;         // >= numItems * bytesPerKey                            Input key array.
        BufferRef<> ilwalues;       // >= numItems * bytesPerValue                          Input value array.
        BufferRef<> tempBuffer;     // = ~numItems * (bytesPerMortonCode + byterPerValue)   Temporary buffer.

        Config(void)
        {
            lwca            = NULL;
            numItems        = 0;
            bytesPerKey     = sizeof(unsigned long long);
            bytesPerValue   = sizeof(int);
        }
    };

public:
                        Sorter              (void) {}
    virtual             ~Sorter             (void) {}

    virtual const char* getName             (void) const { return "Sorter"; }
    void                configure           (const Config& cfg);
    void                execute             (void);

private:
    void                execDevice          (void);
    void                execHost            (void);

private:
                        Sorter              (const Sorter&); // forbidden
    Sorter&             operator=           (const Sorter&); // forbidden

private:
    Config              m_cfg;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
