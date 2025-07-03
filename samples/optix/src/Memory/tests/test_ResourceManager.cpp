//
//  Copyright (c) 2018 LWPU Corporation.  All rights reserved.
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
#include <srcTests.h>

#include <Memory/BufferDimensions.h>
#include <Memory/ResourceManager.h>

using namespace optix;

// These tests are disabled due to this bug:
// TODO: 2757565 ResourceManager::actualDimsForDemandLoadTexture doesn't compute the right dimensions
TEST( actualDimsForDemandLoadTexture, identity_single_level )
{
    BufferDimensions nominalDims;
    nominalDims.setSize( 1024, 1024 );
    nominalDims.setMipLevelCount( 1 );

    const BufferDimensions actualDims = ResourceManager::actualDimsForDemandLoadTexture( nominalDims, 0, 0 );

    ASSERT_EQ( 1, actualDims.mipLevelCount() );
    ASSERT_EQ( 1024, actualDims.levelWidth( 0 ) );
    ASSERT_EQ( 1024, actualDims.levelHeight( 0 ) );
}

TEST( actualDimsForDemandLoadTexture, identity_mip_pyramid )
{
    BufferDimensions nominalDims;
    nominalDims.setSize( 4096, 4096 );
    nominalDims.setMipLevelCount( 10 );

    const BufferDimensions actualDims = ResourceManager::actualDimsForDemandLoadTexture( nominalDims, 0, 9 );

    ASSERT_EQ( 10, actualDims.mipLevelCount() );
    ASSERT_EQ( 4096, actualDims.levelWidth( 0 ) );
    ASSERT_EQ( 4096, actualDims.levelHeight( 0 ) );
}

TEST( actualDimsForDemandLoadTexture, truncates_finer_levels )
{
    // Level 0: 4096
    // 1: 2048
    // 2: 1024
    // 3: 512
    // 4: 256
    // 5: 128
    // 6: 64
    // 7: 32
    // 8: 16
    // 9: 8
    BufferDimensions nominalDims;
    nominalDims.setSize( 4096, 4096 );
    nominalDims.setMipLevelCount( 10 );

    const BufferDimensions actualDims = ResourceManager::actualDimsForDemandLoadTexture( nominalDims, 5, 9 );

    ASSERT_EQ( 5, actualDims.mipLevelCount() );
    ASSERT_EQ( 4096 >> 5, actualDims.levelWidth( 0 ) );
    ASSERT_EQ( 4096 >> 5, actualDims.levelHeight( 0 ) );
}

TEST( actualDimsForDemandLoadTexture, truncates_coarsest_levels )
{
    // Level 0: 4096
    // 1: 2048
    // 2: 1024
    // 3: 512
    // 4: 256
    // 5: 128
    // 6: 64
    // 7: 32
    // 8: 16
    // 9: 8
    BufferDimensions nominalDims;
    nominalDims.setSize( 4096, 4096 );
    nominalDims.setMipLevelCount( 10 );

    const BufferDimensions actualDims = ResourceManager::actualDimsForDemandLoadTexture( nominalDims, 0, 5 );

    ASSERT_EQ( 6, actualDims.mipLevelCount() );
    ASSERT_EQ( 4096, actualDims.levelWidth( 0 ) );
    ASSERT_EQ( 4096, actualDims.levelHeight( 0 ) );
    ASSERT_EQ( 4096 >> 5, actualDims.levelWidth( 5 ) );
    ASSERT_EQ( 4096 >> 5, actualDims.levelWidth( 5 ) );
}


TEST( actualDimsForDemandLoadTexture, truncates_both_ends )
{
    // Level 0: 4096
    // 1: 2048
    // 2: 1024
    // 3: 512
    // 4: 256
    // 5: 128
    // 6: 64
    // 7: 32
    // 8: 16
    // 9: 8
    BufferDimensions nominalDims;
    nominalDims.setSize( 4096, 4096 );
    nominalDims.setMipLevelCount( 10 );

    const BufferDimensions actualDims = ResourceManager::actualDimsForDemandLoadTexture( nominalDims, 5, 7 );

    ASSERT_EQ( 3, actualDims.mipLevelCount() );
    ASSERT_EQ( 4096 >> 5, actualDims.levelWidth( 0 ) );
    ASSERT_EQ( 4096 >> 5, actualDims.levelHeight( 0 ) );
    ASSERT_EQ( 4096 >> 7, actualDims.levelWidth( 2 ) );
    ASSERT_EQ( 4096 >> 7, actualDims.levelWidth( 2 ) );
}

TEST( actualDimsForDemandLoadTexture, rectangular_power_of_two )
{
    // Level 0: 4096x512
    // 1: 2048x256
    // 2: 1024x128
    // 3: 512x64
    // 4: 256x32
    // 5: 128x16
    // 6: 64x8
    // 7: 32x4
    // 8: 16x2
    // 9: 8x1
    BufferDimensions nominalDims;
    nominalDims.setSize( 4096, 512 );
    nominalDims.setMipLevelCount( 10 );

    const BufferDimensions actualDims = ResourceManager::actualDimsForDemandLoadTexture( nominalDims, 5, 7 );

    ASSERT_EQ( 3, actualDims.mipLevelCount() );
    ASSERT_EQ( 4096 >> 5, actualDims.levelWidth( 0 ) );
    ASSERT_EQ( 512 >> 5, actualDims.levelHeight( 0 ) );
    ASSERT_EQ( 4096 >> 7, actualDims.levelWidth( 2 ) );
    ASSERT_EQ( 512 >> 7, actualDims.levelHeight( 2 ) );
}

TEST( actualDimsForDemandLoadTexture, rectangular_non_power_of_two )
{
    // 0: 511 x 2049
    // 1: 255 x 1024
    // 2: 127 x 512
    // 3: 63 x 256
    // 4: 31 x 128
    // 5: 15 x 64
    // 6: 7 x 32
    // 7: 3 x 16
    // 8: 1 x 8
    // 9: 1 x 4
    BufferDimensions nominalDims;
    nominalDims.setSize( 511, 2049 );
    nominalDims.setMipLevelCount( 10 );

    const BufferDimensions actualDims = ResourceManager::actualDimsForDemandLoadTexture( nominalDims, 5, 7 );

    ASSERT_EQ( 3, actualDims.mipLevelCount() );
    ASSERT_EQ( 15, actualDims.levelWidth( 0 ) );
    ASSERT_EQ( 64, actualDims.levelHeight( 0 ) );
    ASSERT_EQ( 3, actualDims.levelWidth( 2 ) );
    ASSERT_EQ( 16, actualDims.levelHeight( 2 ) );
}
