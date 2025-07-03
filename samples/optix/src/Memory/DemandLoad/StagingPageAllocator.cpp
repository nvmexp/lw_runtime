//
//  Copyright (c) 2019 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include <Memory/DemandLoad/StagingPageAllocator.h>

#include <Memory/DemandLoad/StagingPageAllocatorRingBuffer.h>
#include <Memory/DemandLoad/StagingPageAllocatorSimple.h>
#include <Util/ContainerAlgorithm.h>
#include <Util/MakeUnique.h>

#include <prodlib/system/Knobs.h>

#include <cstdint>

// Use at least 16-bytes for the fence so that the pointer to user data is 16-byte aligned
// to allow error-free casting to float4* and maintain the necessary alignment..
static unsigned char s_electricFencePattern[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCD, 0xCD, 0xCD, 0xCD,
                                                 0xFA, 0xCE, 0xF0, 0x0D, 0xDC, 0xDC, 0xDC, 0xDC};

static const size_t s_electricFencePatternSize = sizeof( s_electricFencePattern );

namespace {

// clang-format off
Knob<size_t> k_electricFenceSize( RT_DSTRING( "rtx.demandLoad.electricFenceSize" ), s_electricFencePatternSize, RT_DSTRING( "Number of bytes to fill with a pattern before and after the user data area to check for underruns/overruns" ) );
// clang-format on

}  // namespace

namespace optix {

std::unique_ptr<StagingPageAllocator> createStagingPageAllocator( DeviceManager* dm,
                                                                  unsigned int   maxNumPages,
                                                                  unsigned int   pageSizeInBytes,
                                                                  bool           useAsyncCopies )
{
    if( useAsyncCopies )
        return makeUnique<StagingPageAllocatorRingBuffer>( dm, maxNumPages, pageSizeInBytes );

    return makeUnique<StagingPageAllocatorSimple>( maxNumPages, pageSizeInBytes );
}

size_t getElectricFenceSize()
{
    // Can't have an electric fence smaller than the pattern.
    return std::max( k_electricFenceSize.get(), s_electricFencePatternSize );
}

static void writeElectricFence( unsigned char* fenceStart )
{
    const size_t electricFenceSize = getElectricFenceSize();
    const size_t numPatterns       = electricFenceSize / s_electricFencePatternSize;
    // Copy repetitions of whole pattern
    for( size_t offset = 0; offset < numPatterns * s_electricFencePatternSize; offset += s_electricFencePatternSize )
    {
        algorithm::copy( s_electricFencePattern, &fenceStart[offset] );
    }
    // Copy partial remaining pattern
    std::copy( &s_electricFencePattern[0], &s_electricFencePattern[electricFenceSize % s_electricFencePatternSize],
               &fenceStart[numPatterns * s_electricFencePatternSize] );
}

void writeElectricFence( const StagingPageAllocation& alloc )
{
    if( alloc.address == nullptr )
        return;

    RT_ASSERT_MSG( reinterpret_cast<std::intptr_t>( alloc.address ) % 16 == 0,
                   "User data pointer is not 16-byte aligned" );
    unsigned char* address = static_cast<unsigned char*>( alloc.address );
    writeElectricFence( address - getElectricFenceSize() );
    writeElectricFence( address + alloc.size );
}

static bool checkElectricFenceModified( const unsigned char* fenceStart )
{
    const size_t electricFenceSize = getElectricFenceSize();
    const size_t numPatterns       = electricFenceSize / s_electricFencePatternSize;
    // Check repetitions of whole pattern
    for( size_t offset = 0; offset < numPatterns * s_electricFencePatternSize; offset += s_electricFencePatternSize )
    {
        if( !algorithm::equal( s_electricFencePattern, &fenceStart[offset] ) )
        {
            return true;
        }
    }

    // Check partial remaining pattern
    return !std::equal( &s_electricFencePattern[0], &s_electricFencePattern[electricFenceSize % s_electricFencePatternSize],
                        &fenceStart[numPatterns * s_electricFencePatternSize] );
}

bool checkElectricFenceModified( const StagingPageAllocation& alloc )
{
    if( alloc.address == nullptr )
        return false;

    const unsigned char* address = static_cast<const unsigned char*>( alloc.address );
    return checkElectricFenceModified( address - getElectricFenceSize() ) || checkElectricFenceModified( address + alloc.size );
}

}  // namespace optix
