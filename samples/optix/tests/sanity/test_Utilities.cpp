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

#include <optix.h>
#include <optix_stubs.h>

#include <exptest/loggingHelpers.h>
#include <exptest/testTools.h>

#include <gmock/gmock.h>

#include "CommonAsserts.h"

using namespace testing;

//---------------------------------------------------------------------------
// 	optixInit
//---------------------------------------------------------------------------

struct O7_API_optixInit : Test
{
    void SetUp() override { exptest::lwdaInitialize(); }
    void TearDown() override {}
};


TEST_F( O7_API_optixInit, RunWithSuccess )
{
    ASSERT_OPTIX_SUCCESS( optixInit() );
}


//---------------------------------------------------------------------------
// 	optixInitWithHandle
//---------------------------------------------------------------------------

struct O7_API_optixInitWithHandle : O7_API_optixInit
{
};


TEST_F( O7_API_optixInitWithHandle, RunWithSuccess )
{
    void* handle;
    ASSERT_OPTIX_SUCCESS( optixInitWithHandle( &handle ) );
}


TEST_F( O7_API_optixInitWithHandle, RunWithNullptr )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixInitWithHandle( nullptr ) );
}
