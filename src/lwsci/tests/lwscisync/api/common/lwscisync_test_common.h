/*
 * Copyright (c) 2019-2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#ifndef INCLUDED_LWSCISYNC_TEST_COMMON_H
#define INCLUDED_LWSCISYNC_TEST_COMMON_H

// This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0
#include <gtest/gtest.h>

#include <cinttypes>
#include <lwscilog.h>
#include <lwscisync.h>
#include <lwscisync_internal.h>
#include <sstream>
#include <string>
#include <time.h>

#include "ipc_wrapper_old.h"
#include "test_info.h"

#define LWSCISYNC_TEST_STANDARD_SUBMIT_SIZE (1024)
#define LWSCISYNC_TEST_UNREASONABLE_SUBMIT_SIZE (0x100000)

/* Declare new tests with this macro to make sure each test case has Jama ID */
#define LWSCISYNC_DECLARE_TEST(testSuite, testName, JamaID)                    \
    class testSuite##JamaID : public LwSciSyncBaseTest<JamaID>                 \
    {                                                                          \
    };                                                                         \
    TEST_F(testSuite##JamaID, testName)

#define ATTR_NAME(key) LwSciSyncAttrKeyToString((key))
#define INTERNAL_ATTR_NAME(key) LwSciSyncInternalAttrKeyToString((key))

#define PRINTF(...)                                                            \
    do {                                                                       \
        printf("[          ] ");                                               \
        printf(__VA_ARGS__);                                                   \
    } while (0)

#define TEST_COUT TestCout()

#define WHERE()                                                                \
    do {                                                                       \
        TEST_COUT << __FILE__ << ":" << __LINE__;                              \
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

enum class LwSciSyncTestStatus {
    Success,
    Failure
};

template <int64_t JamaID>
class LwSciSyncBaseTest : public ::testing::Test
{
public:
    TestInfo* info;

    LwSciSyncBaseTest() : info(TestInfo::get())
    {
    }

    void TearDown() override
    {
        if (!HasFailure()) {
            TEST_COUT << JamaID << " OK";
        } else {
            TEST_COUT << JamaID << " FAILED";
        }
    }
};

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TestResourcesRec* TestResources;

/* structures created by init necessary for functioning */
struct StreamResources {
    IpcWrapperOld* upstreamIpcs;
    size_t upstreamSize;
    IpcWrapperOld downstreamIpc;
    LwSciSyncObj syncObj;
    LwSciSyncCpuWaitContext waitContext;
};

/* configuration of the process/thread */
struct ThreadConf {
    LwSciError (*fillAttrList)(LwSciSyncAttrList list);
    LwSciError (*stream)(struct ThreadConf* conf,
                         struct StreamResources* resources);
    const char* downstream;
    const char** upstream;
    size_t upstreamSize;
    LwSciSyncAccessPerm objExportPerm;
    LwSciSyncAccessPerm objImportPerm;
    TestInfo* info;
};

static inline void printTimestampDiff(struct timespec* begin,
                                      struct timespec* end)
{
    time_t diff;

    clock_gettime(CLOCK_REALTIME, end);
    diff = end->tv_sec - begin->tv_sec;
    printf("%ld s has passed since the start\n", diff);
}

static inline void NegativeTestPrint(void)
{
    LWSCI_ERR_STR("NEGATIVE CASE: EXPECTED: ");
}

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
        LWSCI_ERR_STR(
            "LWSCISYNC NEGATIVE TEST START: (IGNORE ERRORS FROM HERE)");
    }

    ~NegativeTest()
    {
        LWSCI_ERR_STR("LWSCISYNC NEGATIVE TEST ENDED: (EXPECT NO MORE ERRORS)");
    }
};

#define NEGATIVE_TEST() NegativeTest negativeTest##__LINE__

static const char* LwSciSyncAttrKeyToString(LwSciSyncAttrKey key)
{
    switch (key) {
    case LwSciSyncAttrKey_NeedCpuAccess:
        return "LwSciSyncAttrKey_NeedCpuAccess";
        break;
    case LwSciSyncAttrKey_RequiredPerm:
        return "LwSciSyncAttrKey_RequiredPerm";
        break;
    case LwSciSyncAttrKey_ActualPerm:
        return "LwSciSyncAttrKey_ActualPerm";
        break;
    case LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports:
        return "LwSciSyncAttrKey_WaiterContextInsensitiveFenceExports";
        break;
    case LwSciSyncAttrKey_WaiterRequireTimestamps:
        return "LwSciSyncAttrKey_WaiterRequireTimestamps";
        break;
    case LwSciSyncAttrKey_RequireDeterministicFences:
        return "LwSciSyncAttrKey_RequireDeterministicFences";
        break;
    default:
        return "Unknown LwSciSyncAttrKey";
    }
}

static const char*
LwSciSyncInternalAttrKeyToString(LwSciSyncInternalAttrKey key)
{
    switch (key) {
    case LwSciSyncInternalAttrKey_SignalerPrimitiveInfo:
        return "LwSciSyncInternalAttrKey_SignalerPrimitiveInfo";
        break;
    case LwSciSyncInternalAttrKey_WaiterPrimitiveInfo:
        return "LwSciSyncInternalAttrKey_WaiterPrimitiveInfo";
        break;
    case LwSciSyncInternalAttrKey_SignalerPrimitiveCount:
        return "LwSciSyncInternalAttrKey_SignalerPrimitiveCount";
        break;
    case LwSciSyncInternalAttrKey_GpuId:
        return "LwSciSyncInternalAttrKey_GpuId";
        break;
    case LwSciSyncInternalAttrKey_SignalerTimestampInfo:
        return "LwSciSyncInternalAttrKey_SignalerTimestampInfo";
        break;
    default:
        return "Unknown LwSciSyncInternalAttrKey";
    }
}

#ifdef __cplusplus
}
#endif

#endif
