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
#include "ApexPointMap.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// ApexPointMapConstructor constructs an ApexPointMap object for a given
// set of input primitives. See ApexPointMap.hpp for description of the
// data. Different kernel variants are employed for different types of
// input primitives (unindexed vertices, indexed vertices, AABBs).

class ApexPointMapConstructor : public BuildingBlock
{
public:
    struct Config
    {
        LwdaUtils*              lwca;               // Non-NULL => execute on LWCA.
        int                     apmResolution;      // Between 1 and 8. Resolution 1 corresponds to an AABB.

                                                    // Size             Description
        BufferRef<ApexPointMap> outApexPointMap;    // = <depends>      Resulting apex point map. Number of bytes depends on the resolution.
        BufferRef<>             tempBuffer;         // = ~4             Temporary buffer.
        ModelBuffers            inModel;            // <varies>         Input model.

        Config(void)
        {
            lwca            = NULL;
            apmResolution   = 3;
        }
    };

public:
                                ApexPointMapConstructor (void) {}
    virtual                     ~ApexPointMapConstructor(void) {}

    virtual const char*         getName                 (void) const { return "ApexPointMapConstructor"; }
    void                        configure               (const Config& cfg);
    void                        execute                 (void);

    static void                 printLUTSource          (void);

private:
    void                        execDevice              (void);
    void                        execHost                (void);

private:
                                ApexPointMapConstructor (const ApexPointMapConstructor&); // forbidden
    ApexPointMapConstructor&    operator=               (const ApexPointMapConstructor&); // forbidden

private:
    Config                      m_cfg;

    // Temp buffers.

    BufferRef<int>              m_workCounter;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
