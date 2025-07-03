/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_TEST_INTEGRATION_H
#define INCLUDED_LWSCIBUF_TEST_INTEGRATION_H

#include <stdio.h>
#include <stdlib.h>

#if defined(LW_TEGRA_MIRROR_INCLUDES) || defined(__x86_64__)
#include "mobile_common.h"
#else
#include "lwrm_sync.h"
#endif

#include "lwscibuf.h"
#include "lwscibuf_internal.h"
#include "lwscilog.h"

#if BACKEND_RESMAN == 0
#include "lwscibuf_test_integration_tegra.h"
#else
#include "lwscibuf_test_integration_x86.h"
#endif

// This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0

#include "gtest/gtest.h"

#define PRINTF(...)                                                            \
    do {                                                                       \
        printf("[          ] ");                                               \
        printf(__VA_ARGS__);                                                   \
    } while (0)

// C++ stream interface
class TestCout : public std::stringstream
{
public:
    ~TestCout() override
    {
        PRINTF("%s\n", str().c_str());
    }
};

#define TEST_COUT TestCout()

// Negative test print macro
// Usage:
// {
//      NEGATIVE_TEST();
//      /* Some negative test */
// }
class NegativeTest
{
public:
    NegativeTest()
    {
        LWSCI_ERR_STR("LWSCIBUF NEGATIVE TEST START: (IGNORE ERRORS FROM HERE)");
    }

    ~NegativeTest()
    {
        LWSCI_ERR_STR("LWSCIBUF NEGATIVE TEST ENDED: (EXPECT NO MORE ERRORS)");
    }
};

#define NEGATIVE_TEST() NegativeTest negativeTest##__LINE__

LwU64 GetMemorySize(LwSciBufRmHandle rmhandle);
LwU32 GetLwRmAccessFlags(LwSciBufAttrValAccessPerm perm);

static inline LwU64 GetPageSize()
{
    return sysconf(_SC_PAGESIZE);
}
bool CheckBufferAccessFlags(LwSciBufObj bufObj, LwSciBufRmHandle rmHandle);
bool CompareRmHandlesAccessPermissions(LwSciBufRmHandle rmHandle1,
                                               LwSciBufRmHandle rmHandle2);
bool isRMHandleFree(LwSciBufRmHandle rmHandle);

#define TESTERR_CHECK(errvar, failure_log, retval)                             \
    do {                                                                       \
        if (errvar != LwSciError_Success) {                                    \
            printf("\nTEST FAILED: " failure_log "\n");                        \
            return retval;                                                     \
        }                                                                      \
    } while (0)

#endif
