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
//
//

#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include <exptest/coverageFileWriter.h>
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

#include <lwda_runtime_api.h>

#include <optix_function_table_definition.h>

#include <exptest/testTools.h>

int main( int argc, char** argv )
{
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    const char* fileName = OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE_FILENAME;  // defined in sanity.lwmk
    // in case that the configured fileName should quickly be changed - not to be confused with --gtest_filter etc
    if( argc > 1 && argv[1][0] != '-' )
        fileName = argv[1];
    const char*             testName = "sanity";
    CoverageFileWriterGuard guard( fileName, testName, true );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    const int      DENOISER_PIXEL_FORMAT_HALF_COMPATIBILE_SM = 5;
    int            device;
    lwdaDeviceProp deviceProp;
    lwdaGetDevice( &device );
    lwdaGetDeviceProperties( &deviceProp, device );
    int smMajor = deviceProp.major;

    testing::InitGoogleTest( &argc, argv );
    if( smMajor < DENOISER_PIXEL_FORMAT_HALF_COMPATIBILE_SM )
        testing::GTEST_FLAG( filter ) = "-O7_API_SM50*";
    return RUN_ALL_TESTS();
}
