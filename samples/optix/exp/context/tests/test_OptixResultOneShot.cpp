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

#include <srcTests.h>

#include <optix_types.h>

#include <exp/context/OptixResultOneShot.h>

using namespace testing;
using namespace optix_exp;

TEST( TestOptixResultOneShot, constructs_as_success )
{
    const OptixResultOneShot result;

    ASSERT_EQ( OPTIX_SUCCESS, result );
}

TEST( TestOptixResultOneShot, remembers_failure_from_construct )
{
    OptixResultOneShot result( OPTIX_ERROR_UNKNOWN );

    result += OPTIX_SUCCESS;

    ASSERT_EQ( OPTIX_ERROR_UNKNOWN, result );
}

TEST( TestOptixResultOneShot, remembers_failure_from_assignment )
{
    OptixResultOneShot result;

    result += OPTIX_ERROR_UNKNOWN;
    result += OPTIX_SUCCESS;

    ASSERT_EQ( OPTIX_ERROR_UNKNOWN, result );
}

TEST( TestOptixResultOneShot, remembers_first_failure )
{
    OptixResultOneShot result;

    result += OPTIX_ERROR_ILWALID_VALUE;
    result += OPTIX_ERROR_UNKNOWN;

    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, result );
}
