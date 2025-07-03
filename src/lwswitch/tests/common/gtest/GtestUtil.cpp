/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
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
