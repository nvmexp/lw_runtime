// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO FUNCTION SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

#include <lwca.h>

typedef size_t( LWDA_CB* LWoclwpancyB2DSize )( int blockSize );

namespace optix {
namespace lwca {

class Function;

class Oclwpancy
{
  public:
    // Returns oclwpancy of a function.
    static int maxActiveBlocksPerMultiprocessor( const Function&, int blockSize, size_t dynamicSMemSize, LWresult* returnResult = nullptr );

    // Suggest a launch configuration with reasonable oclwpancy.
    static void maxPotentialBlockSize( int* minGridSize,
                                       int* blockSize,
                                       const Function&,
                                       LWoclwpancyB2DSize blockSizeToDynamicSMemSize,
                                       size_t             dynamicSMemSize,
                                       int                blockSizeLimit,
                                       unsigned int       flags        = LW_OCLWPANCY_DEFAULT,
                                       LWresult*          returnResult = nullptr );


  private:
    Oclwpancy() = delete;
};

}  // namespace lwca
}  // namespace optix
