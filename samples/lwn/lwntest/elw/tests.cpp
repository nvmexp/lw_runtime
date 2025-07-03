/*
 * Copyright (c) 1999 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "ogtest.h"
#include "elw.h"

#define TEST_DESC_I(name, path) \
    TEST_TABLE_ENTRY(name, path)


// External function prototypes

#define BEGIN_TEST_GROUP(x)
#define END_TEST_GROUP(x)

#define TEST_TABLE_ENTRY(testname, pathMask)                                                        \
CPP_PREFIX void ogtest_##testname##_GetDescription(char *str, const char *testName);                \
CPP_PREFIX int ogtest_##testname##_IsSupported(const char *testName);                               \
CPP_PREFIX void ogtest_##testname##_InitGraphics(float lwScale, const char *testName, int path);    \
CPP_PREFIX void ogtest_##testname##_DoGraphics(float lwScale, const char *testName, int path);      \
CPP_PREFIX void ogtest_##testname##_ExitGraphics(const char *testName, int path);

#include "tests.h"

#undef BEGIN_TEST_GROUP
#undef END_TEST_GROUP
#undef TEST_TABLE_ENTRY

// Test group arrays

#define BEGIN_TEST_GROUP(_groupName)            \
OGTEST lwog_ ## _groupName ## _TestList[] = {

#define END_TEST_GROUP(_groupName)                  \
};                                                  \
TestGroup lwog_ ## _groupName ## _TestGroup = {     \
    lwog_ ## _groupName ## _TestList,               \
    (sizeof(lwog_ ## _groupName ## _TestList) /     \
     sizeof(lwog_ ## _groupName ## _TestList[0]))   \
};

#define TEST_TABLE_ENTRY(testname, pathMask)            \
    {                                                   \
        ogtest_##testname##_GetDescription,             \
        ogtest_##testname##_IsSupported,                \
        ogtest_##testname##_InitGraphics,               \
        ogtest_##testname##_DoGraphics,                 \
        ogtest_##testname##_ExitGraphics,               \
        #testname,                                      \
        pathMask,                                       \
        0,         /*md5ValidMask*/                     \
        {{'\0'}},  /*md5*/                              \
    },

#include "tests.h"


