/*
 * Copyright (c) 2019-2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <stdio.h>
#include <lwscilist.h>
#include "lwscicommon_objref.h"
//This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0
#include "gtest/gtest.h"

typedef struct TestReferenceData {
    LwSciObj referenceHeader;
    int count;
} testData;

static void cleanupRef(LwSciRef* refptr)
{
}

static void cleanupObj(LwSciObj* objptr)
{
}

class TestObjectReference : public ::testing::Test {

public:

    void SetUp() override {
        ASSERT_EQ(LwSciCommonAllocObjWithRef(sizeof(testData), sizeof(LwSciRef),
        &obj, &ref), LwSciError_Success);
    }

    void TearDown() override {
        LwSciCommonFreeObjAndRef(ref, NULL, NULL);
        LwSciCommonFreeObjAndRef(ref, NULL, NULL);
        LwSciCommonFreeObjAndRef(dupRef, cleanupObj, cleanupRef);
    }

    LwSciRef* ref;
    LwSciRef* dupRef;
    LwSciObj* obj;
};

TEST_F(TestObjectReference, ObjectReference) {
    ASSERT_EQ(LwSciCommonIncrAllRefCounts(ref), LwSciError_Success);

    LwSciObj* dataParam;
    LwSciCommonGetObjFromRef(ref, &dataParam);
    testData* data = lw_container_of(dataParam, testData, referenceHeader);
    // Modify data
    data->count = 111;
    // Drop reference
    data = NULL;
    // Get referenced object
    LwSciCommonGetObjFromRef(ref, &dataParam);
    data = lw_container_of(dataParam, testData, referenceHeader);
    // Check data has been modified
    ASSERT_EQ(data->count, 111);

    LwSciCommonRefLock(ref);
    LwSciCommonRefUnlock(ref);

    LwSciCommonObjLock(ref);
    LwSciCommonObjUnlock(ref);

    ASSERT_EQ(LwSciCommonDuplicateRef(ref, &dupRef), LwSciError_Success);
}
