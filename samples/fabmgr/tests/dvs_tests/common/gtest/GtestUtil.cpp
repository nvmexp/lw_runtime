/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#include "GtestUtil.h"
#include "lwos.h"
#include "lwerror.h"

::testing::AssertionResult LwosFailureHelper(
    const char* expr,
    const char* expected,
    LwU32 lwosStatus)
{
    std::stringstream ss;
    ss << std::hex << lwosStatus;

    return ::testing::AssertionFailure()
        << "Expected: " << expr << " " << expected << ".\n"
        << "  Actual: " << ss.str() << " - " << lwstatusToString(lwosStatus) << "\n";
}

::testing::AssertionResult IsLwosStatusSuccess(const char* expr, LwU32 lwosStatus)
{
    if (lwosStatus == LW_OK)
    {
        return ::testing::AssertionSuccess();
    }
    return LwosFailureHelper(expr, "succeeds", lwosStatus);
}

::testing::AssertionResult IsLwosStatusFailure(const char* expr, LwU32 lwosStatus)
{
    if (lwosStatus != LW_OK)
    {
        return ::testing::AssertionSuccess();
    }
    return LwosFailureHelper(expr, "fails", lwosStatus);
}
