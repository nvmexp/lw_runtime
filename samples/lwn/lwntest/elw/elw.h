/*
 * Copyright (c) 1999 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __ELW_H
#define __ELW_H

#define MAX_MD5_GOLDS       4 // Number of possible MD5 matches per test.

#define LWOG_TEST_PATH_MASK_NONE        0x00000000
#define LWOG_TEST_PATH_MASK_BASE        0x00000001
#define LWOG_TEST_PATH_MASK_EXCLUDE     0x00100000
#define LWOG_TEST_PATH_MASK_DEBUG_TEST  0x00200000
#define LWOG_TEST_PATH_MASK_INCLUDE     0x04000000
#define LWOG_TEST_PATH_MASK_RUNTIMESUPP 0x10000000
#define LWOG_TEST_PATH_MASK_SKIP        0x20000000

typedef struct _OGTEST {
    void (*getDescription)(char *str, const char *testName);
    int (*isSupportedFunc)(const char *testName);
    void (*initFunc)(float, const char *testName, int path);
    void (*doFunc)(float, const char *testName, int path);
    void (*exitFunc)(const char *testName, int path);
    const char *name;
    unsigned int pathMask;
    unsigned int  md5ValidMask;
    unsigned char md5[MAX_MD5_GOLDS][16];

#define TEST_BASE_NAME(testEntry)                       \
    ((testEntry)->name)
#define TEST_PROF_NAME(testEntry)                       \
    ((testEntry)->name)
#define TEST_PATH_NAME(testEntry, path)                 \
    ((testEntry)->name)
#define TEST_FULL_NAME(testEntry, path)                 \
    ((testEntry)->name)
} OGTEST;

typedef struct _TestGroup {
    OGTEST          *tests;
    unsigned int    nTests;
} TestGroup;

extern int TestCount;

#endif
