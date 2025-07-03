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
#include "common/BufferRef.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// Base class for all "building blocks" in the BVH builder library.
// A building block is an independent, well-defined, stateless operation
// that reads some input data and writes some output data.
//
// Notes:
//
// - Creating and destroying building blocks must be a light-weight
//   operation.
//
// - Each building block must contain at least a CPU implementation.
//   It may also contain a GPU implementation.
//
// - A building block must not perform GPU memory allocations itself.
//
// - The input/output data arrays are passed in as BufferRef members
//   of the Config struct.
//
// - The configure() method of each building block should assert the
//   sizes of the input buffers and explicitly resize the output buffers.
//
// - If a building block needs temporary storage, it should accept
//   an explicit tempBuffer as a part of the Config. The configure()
//   method can then aggregate/overlay several internal buffers within
//   the tempBuffer.

class BuildingBlock
{
public:
//  struct Config;

public:
                        BuildingBlock   (void);
    virtual             ~BuildingBlock  (void);

    virtual const char* getName         (void) const = 0;
//  void                configure       (const Config& cfg);
//  void                execute         (void);

private:
                        BuildingBlock   (const BuildingBlock&); // forbidden
    BuildingBlock&      operator=       (const BuildingBlock&); // forbidden
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
