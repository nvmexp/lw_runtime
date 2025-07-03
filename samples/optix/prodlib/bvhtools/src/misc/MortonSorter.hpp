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
#include <prodlib/bvhtools/src/common/TypesInternal.hpp>
#include <prodlib/bvhtools/src/misc/Sorter.hpp>
#include <prodlib/bvhtools/src/bounds/ApexPointMap.hpp>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// Callwlates Morton codes for the given input primitives and sorts them
// in ascending order.
//
// Input:
//
// - The input primitives are specified through inModel as defined in
//   ModelPointers::loadPrimitiveAABB().
//
// - The set of input primitives can be filtered using inPrimRange and
//   inPrimOrder as follows:
//
//      for (int i = inPrimRange->start; i < inPrimRange->end; i++)
//          addInputPrimitiveIdx(inPrimOrder[i]);
//
// - Either/both inPrimRange and inPrimOrder can be left out by setting
//   them to EmptyBuf. The default interpretation is:
//   
//      inPrimRange    = Range(0, cfg.maxPrims)
//      inPrimOrder[i] = i
//
// - The maximum number of input primitives is specified at configure(),
//   but the actual number can vary at runtime.
//
// Output:
//
// - The sorted Morton codes and their corresponding input primitive indices
//   are placed in outMortonCodes and outPrimOrder.
//
// - The range of valid output is (0, inPrimRange->span()).
//   This range is written into outPrimRange for colwenience.
//
// - More than inPrimRange->span() elements may be written to the
//   output buffers. Any extraneous data must be ignored by the caller.

class MortonSorter : public BuildingBlock
{
public:
    struct Config
    {
        LwdaUtils*                      lwca;               // Non-NULL => execute on LWCA.
        int                             maxPrims;           // Upper bound for the number of input primitives, i.e., inPrimRange->span().
        int                             bytesPerMortonCode; // Size of the Morton codes in bytes. Must be 4 or 8.

                                                            // Size                                 Description
        BufferRef<>                     outMortonCodes;     // = maxPrims*bytesPerMortonCode        Sorted Morton codes.
        BufferRef<int>                  outPrimOrder;       // = maxPrims                           Primitive indices reordered according to the Morton codes.
        BufferRef<Range>                outPrimRange;       // = 1                                  Range of entries in outMortonCodes: (0, inPrimRange.span()).
        BufferRef<>                     tempBuffer;         // = ~(bytesPerMortonCode+4)*2*maxPrims Temporary buffer.

        BufferRef<const int>            inPrimOrder;        // 0 or >= maxPrims                     Input primitive indices. Can be the same buffer as outPrimOrder.
        ModelBuffers                    inModel;            // <varies>                             Input model.
        BufferRef<const ApexPointMap>   inApexPointMap;     // <varies>                             Produced by ApexPointMapConstructor. Must enclose all input primitives.

        Config(void)
        {
            lwca                = NULL;
            maxPrims            = 0;
            bytesPerMortonCode  = sizeof(unsigned long long);
        }
    };

public:
                            MortonSorter        (void) {}
    virtual                 ~MortonSorter       (void) {}

    virtual const char*     getName             (void) const { return "MortonSorter"; }
    void                    configure           (const Config& config);
    void                    execute             (BufferRef<const Range> inPrimRange); // inPrimRange = range of elements in inPrimOrder. EmptyBuf => (0, maxPrims).

private:
    void                    execDevice          (void);
    void                    execHost            (void);

private:
                            MortonSorter        (const MortonSorter&); // forbidden
    MortonSorter&           operator=           (const MortonSorter&); // forbidden

private:
    Sorter                  m_sorter;
    Config                  m_cfg;
    BufferRef<const Range>  m_inPrimRange;
    BufferRef<>             m_tmpMortonCodes;
    BufferRef<int>          m_tmpPrimOrder;
    BufferRef<>             m_sorterTemp;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
