/*
 * Copyright (c) 2019-2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <algorithm>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <lwscicommon_libc.h>

//This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0
#include "gtest/gtest.h"

static int CompareUint64(const void* elem1, const void* elem2) {
    const uint64_t a = *(const uint64_t*)elem1;
    const uint64_t b = *(const uint64_t*)elem2;

    if (a > b) {
        return 1;
    }
    if (a < b) {
        return -1;
    }
    return 0;
}

static int CompareUint32(const void* elem1, const void* elem2) {
    const uint32_t a = *(const uint32_t*)elem1;
    const uint32_t b = *(const uint32_t*)elem2;

    if (a > b) {
        return 1;
    }
    if (a < b) {
        return -1;
    }
    return 0;
}

TEST(TestPlatformUtilities, ValidateSystemMemory) {
    uint64_t* ptr1 = NULL;
    uint64_t* ptr2 = NULL;

    // Allocate memory to one object using LwSciCommonCalloc
    ptr1 = (uint64_t*) LwSciCommonCalloc(1, sizeof(uint64_t));

    // Allocate memory to another object using calloc
    ptr2 = (uint64_t*) calloc(1, sizeof(uint64_t));

    *ptr1 = 24;
    *ptr2 = 50;

    // Free memory of first object
    LwSciCommonFree(ptr1);

    /* If the memory is not allocated by LwSciCommon we may abort due to panic()
     * or due to seg fault since we access pointer before start of memory to
     * check for header.
     *
     * ASSERT_DEATH checks if 1st statement exited abnormally else it marks as failure.
     * LwSciCommonFree(ptr2),exit(0) : this statement should ideally try to exit program with
     * abort(), in abnormal condition (abort() fails) it will try to exit program with exit(0),
     * ASSERT_DEATH() will detect this abort() failure and report us.
     * ".*" is regular expression matches which tells to match everything in string.
     */
    // Free memory of second object and verify it failed
    ASSERT_DEATH((LwSciCommonFree(ptr2),exit(0)), ".*");
}

TEST(TestPlatformUtilities, TestLibC) {
    uint64_t* obj = NULL;
    uint64_t* objCopy = NULL;

    // Allocate 2 uint64_t's
    obj = (uint64_t*) LwSciCommonCalloc(1, sizeof(uint64_t));
    objCopy = (uint64_t*) LwSciCommonCalloc(1, sizeof(uint64_t));
    ASSERT_TRUE(obj != NULL);
    ASSERT_TRUE(objCopy != NULL);
    ASSERT_TRUE(obj != objCopy);

    // Store some value to the first object
    *obj = 42;

    // Make a copy
    LwSciCommonMemcpyS(obj, sizeof(uint64_t), objCopy, sizeof(uint64_t));

    // Copy should be equal
    ASSERT_TRUE(LwSciCommonMemcmp(obj, objCopy, sizeof(uint64_t)) == 0);
    ASSERT_EQ(*obj, *objCopy);

    // Free memory
    LwSciCommonFree(obj);
    LwSciCommonFree(objCopy);
}

TEST(TestPlatformUtilities, TestMemCpyS) {
    /*
     * We reserve a memory block of 30 elements
     * |__________SSSSSSSSSS__________|
     *
     * Fill up data from 10 to 20 with integers, this will be src memory for
     *  LwSciCommonMemcpyS.
     */

    {
    /* Test 1: Success case
     *  Copy 10 elements from 10-19 to 0-9
     * |__________SSSSSSSSSS__________|
     * |DDDDDDDDDD____________________|
     */
        uint64_t memoryBlock[30] = {0};
        for (int i=10; i<20; i++) {
            memoryBlock[i] = i-10 + 1;
        }

        LwSciCommonMemcpyS(&memoryBlock[0], 10*sizeof(uint64_t),
                            &memoryBlock[10], 10*sizeof(uint64_t));

        int ret = memcmp(&memoryBlock[0], &memoryBlock[10],
                            10*sizeof(uint64_t));
        ASSERT_EQ(0, ret) << "Failed to copy data";
    }
    {
    /* Test 2: Success case
     *  Copy 10 elements from 10-19 to 20-30
     * |__________SSSSSSSSSS__________|
     * |____________________DDDDDDDDDD|
     */
        uint64_t memoryBlock[30] = {0};
        for (int i=10; i<20; i++) {
            memoryBlock[i] = i-10 + 1;
        }

        LwSciCommonMemcpyS(&memoryBlock[20], 10*sizeof(uint64_t),
                            &memoryBlock[10], 10*sizeof(uint64_t));

        int ret = memcmp(&memoryBlock[20], &memoryBlock[10],
                            10*sizeof(uint64_t));
        ASSERT_EQ(0, ret) << "Failed to copy data";
    }
    {
    /* Test 3: Success case
     *  Copy 10 elements from 10-19 to 0-19
     * |__________SSSSSSSSSS__________|
     * |DDDDDDDDDDDDDDDDDDDD__________|
     */
        uint64_t memoryBlock[30] = {0};
        for (int i=10; i<20; i++) {
            memoryBlock[i] = i-10 + 1;
        }

        LwSciCommonMemcpyS(&memoryBlock[0], 20*sizeof(uint64_t),
                            &memoryBlock[10], 10*sizeof(uint64_t));
        int ret = memcmp(&memoryBlock[0], &memoryBlock[10],
                            10*sizeof(uint64_t));
        ASSERT_EQ(0, ret) << "Failed to copy data";
    }
    {
    /* Test 4: Overlap case
     *  Copy 10 elements from 10-19 to 5-14
     * |__________SSSSSSSSSS__________|
     * |_____DDDDDDDDDD_______________|
     */
        uint64_t memoryBlock[30] = {0};
        for (int i=10; i<20; i++) {
            memoryBlock[i] = i-10 + 1;
        }

        ASSERT_EXIT((LwSciCommonMemcpyS(
                            &memoryBlock[5], 10*sizeof(uint64_t),
                            &memoryBlock[10], 10*sizeof(uint64_t)),exit(0)),
                    ::testing::KilledBySignal(SIGABRT), ".*");
    }
    {
    /* Test 5: Overlap case
     *  Copy 10 elements from 10-19 to 5-29
     * |__________SSSSSSSSSS__________|
     * |_____DDDDDDDDDDDDDDDDDDDDDDDDD|
     */
        uint64_t memoryBlock[30] = {0};
        for (int i=10; i<20; i++) {
            memoryBlock[i] = i-10 + 1;
        }

        ASSERT_EXIT((LwSciCommonMemcpyS(
                            &memoryBlock[5], 25*sizeof(uint64_t),
                            &memoryBlock[10], 10*sizeof(uint64_t)),exit(0)),
                    ::testing::KilledBySignal(SIGABRT), ".*");
    }
    {
    /* Test 6: Overlap case
     *  Copy 10 elements from 10-19 to 15-24
     * |__________SSSSSSSSSS__________|
     * |_______________DDDDDDDDDD_____|
     */
        uint64_t memoryBlock[30] = {0};
        for (int i=10; i<20; i++) {
            memoryBlock[i] = i-10 + 1;
        }

        ASSERT_EXIT((LwSciCommonMemcpyS(
                            &memoryBlock[15], 10*sizeof(uint64_t),
                            &memoryBlock[10], 10*sizeof(uint64_t)),exit(0)),
                    ::testing::KilledBySignal(SIGABRT), ".*");
    }
    {
    /* Test 6: Overlap case
     *  Copy 10 elements from 10-19 to 15-24
     * |__________SSSSSSSSSS__________|
     * |__________DDDDDDDDDDDDDDDDDDDD|
     */
        uint64_t memoryBlock[30] = {0};
        for (int i=10; i<20; i++) {
            memoryBlock[i] = i-10 + 1;
        }

        ASSERT_EXIT((LwSciCommonMemcpyS(
                            &memoryBlock[10], 20*sizeof(uint64_t),
                            &memoryBlock[10], 10*sizeof(uint64_t)),exit(0)),
                    ::testing::KilledBySignal(SIGABRT), ".*");
    }
    {
    /* Test 7: destSize less than srcSize
     *  Copy 10 elements from 10-19 to 0-4
     * |__________SSSSSSSSSS__________|
     * |DDDDD_________________________|
     */
        uint64_t memoryBlock[30] = {0};
        for (int i=10; i<20; i++) {
            memoryBlock[i] = i-10 + 1;
        }

        ASSERT_EXIT((LwSciCommonMemcpyS(
                            &memoryBlock[0], 5*sizeof(uint64_t),
                            &memoryBlock[10], 10*sizeof(uint64_t)),exit(0)),
                    ::testing::KilledBySignal(SIGABRT), ".*");
    }
}

TEST(TestPlatformUtilities, TestCallocZeroSize) {
    uint64_t* obj = NULL;

    // Allocate a zero length array
    obj = (uint64_t*)LwSciCommonCalloc(0U, sizeof(uint64_t));
    ASSERT_EQ(nullptr, obj);

    // Allocate an array with 0 size elements
    obj = (uint64_t*)LwSciCommonCalloc(16U, 0U);
    ASSERT_EQ(nullptr, obj);
}

TEST(TestPlatformUtilities, TestSortArrayIlwalidComparator) {
    LwSciError err = LwSciError_Success;
    uint64_t arr[] = { 0U };

    ASSERT_EXIT((LwSciCommonSort(arr, 1U, sizeof(arr[0]), NULL),
                    exit(0)), ::testing::KilledBySignal(SIGABRT), ".*");
}

TEST(TestPlatformUtilities, TestSortArrayIlwalidBaseAddress) {
    LwSciError err = LwSciError_Success;
    uint64_t arr[] = { 0U };

    ASSERT_EXIT((LwSciCommonSort(NULL, 1U, sizeof(arr[0]), CompareUint64),
                    exit(0)), ::testing::KilledBySignal(SIGABRT), ".*");
}

TEST(TestPlatformUtilities, TestSortArrayIlwalidArrayLen) {
    LwSciError err = LwSciError_Success;
    uint64_t arr[] = { 0U };

    ASSERT_EXIT((LwSciCommonSort(arr, 0U, sizeof(arr[0]), CompareUint64),
                    exit(0)), ::testing::KilledBySignal(SIGABRT), ".*");
}

TEST(TestPlatformUtilities, TestSortArrayIlwalidArrayElementSize) {
    LwSciError err = LwSciError_Success;
    uint64_t arr[] = { 0U };

    ASSERT_EXIT((LwSciCommonSort(arr, 1U, 0U, CompareUint64),
                    exit(0)), ::testing::KilledBySignal(SIGABRT), ".*");
}

TEST(TestPlatformUtilities, TestSortArraySingleElementUint64) {
    LwSciError err = LwSciError_Success;
    uint64_t arr[] = { 0U };
    size_t len = sizeof(arr) / sizeof(arr[0]);

    LwSciCommonSort(arr, 1U, sizeof(arr[0]), CompareUint64);

    ASSERT_EQ(err, LwSciError_Success);
    ASSERT_TRUE(std::is_sorted(arr, arr + len));
}

TEST(TestPlatformUtilities, TestSortArraySingleElementUint32) {
    LwSciError err = LwSciError_Success;
    uint32_t arr[] = { 0U };
    size_t len = sizeof(arr) / sizeof(arr[0]);

    LwSciCommonSort(arr, len, sizeof(arr[0]), CompareUint32);

    ASSERT_EQ(err, LwSciError_Success);
    ASSERT_TRUE(std::is_sorted(arr, arr + len));
}

TEST(TestPlatformUtilities, TestSortArrayOddNumElementsUint64) {
    LwSciError err = LwSciError_Success;
    uint64_t arr[] = { 0U, 2U, 19U, 1U, 5U };
    size_t len = sizeof(arr) / sizeof(arr[0]);

    LwSciCommonSort(arr, len, sizeof(arr[0]), CompareUint32);

    ASSERT_EQ(err, LwSciError_Success);
    ASSERT_TRUE(std::is_sorted(arr, arr + len));
}

TEST(TestPlatformUtilities, TestSortArrayEvenNumElementsUint64) {
    LwSciError err = LwSciError_Success;
    uint64_t arr[] = { 12U, 2U, 9U, 1U, 5U, 21U };
    size_t len = sizeof(arr) / sizeof(arr[0]);

    LwSciCommonSort(arr, len, sizeof(arr[0]), CompareUint32);

    ASSERT_EQ(err, LwSciError_Success);
    ASSERT_TRUE(std::is_sorted(arr, arr + len));
}
