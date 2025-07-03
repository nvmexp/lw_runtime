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

#include <LWCA/Oclwpancy.h>

#include <LWCA/ErrorCheck.h>
#include <LWCA/Function.h>
#include <corelib/system/LwdaDriver.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix::lwca;
using namespace corelib;

int Oclwpancy::maxActiveBlocksPerMultiprocessor( const Function& func, int blockSize, size_t dynamicSMemSize, LWresult* returnResult )
{
    RT_ASSERT( func.get() != nullptr );
    int result = 0;
    CHECK( lwdaDriver().LwOclwpancyMaxActiveBlocksPerMultiprocessor( &result, func.get(), blockSize, dynamicSMemSize ) );
    return result;
}

void Oclwpancy::maxPotentialBlockSize( int*               minGridSize,
                                       int*               blockSize,
                                       const Function&    func,
                                       LWoclwpancyB2DSize blockSizeToDynamicSMemSize,
                                       size_t             dynamicSMemSize,
                                       int                blockSizeLimit,
                                       unsigned int       flags,
                                       LWresult*          returnResult )

{
    RT_ASSERT( func.get() != nullptr );
    CHECK( lwdaDriver().LwOclwpancyMaxPotentialBlockSizeWithFlags( minGridSize, blockSize, func.get(), blockSizeToDynamicSMemSize,
                                                                   dynamicSMemSize, blockSizeLimit, flags ) );
}
