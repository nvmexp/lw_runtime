/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


#ifndef _gtestutil_h_
#define _gtestutil_h_

#include "lwtypes.h"
#include "gtest/gtest.h"

// for some reason lwRmApi.h doesn't know what PUCHAR is so I had to put this here
#ifndef PUCHAR
    typedef unsigned char *PUCHAR;
#endif

const char *LwosStatusToStr(LwU32 lwosStatus);

///////////////////////////////////////////////////////////////////////////////
// Macros that test for LW_STATUS failure and success
//
//    * {ASSERT|EXPECT}_LWOS_STATUS_{SUCCEEDED|FAILED}(expr)
//
// When expr unexpectedly fails or succeeds, Google Test prints the
// expected result and the actual result with both a human-readable
// string representation of the error, if available, as well as the
// hex result code.
# define EXPECT_LWOS_STATUS_SUCCEEDED(expr) \
    EXPECT_PRED_FORMAT1(IsLwosStatusSuccess, (expr))

# define ASSERT_LWOS_STATUS_SUCCEEDED(expr) \
    ASSERT_PRED_FORMAT1(IsLwosStatusSuccess, (expr))

# define EXPECT_LWOS_STATUS_FAILED(expr) \
    EXPECT_PRED_FORMAT1(IsLwosStatusFailure, (expr))

# define ASSERT_LWOS_STATUS_FAILED(expr) \
    ASSERT_PRED_FORMAT1(IsLwosStatusFailure, (expr))


::testing::AssertionResult IsLwosStatusSuccess(const char* expr, LwU32 lwosStatus);

::testing::AssertionResult IsLwosStatusFailure(const char* expr, LwU32 lwosStatus);

#endif // _gtestutil_h_
