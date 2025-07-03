/*
 * Copyright (c) 2019-2021 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include "lwscibuf_test_integration.h"

TEST(TestLwSciBufVersionCompatibility, CheckVersionCompatibility)
{
    LwSciError err = LwSciError_Success;
    bool isCompatible = false;

    // Check with valid major & minor versions
    // API major number = loaded library major number
    // API minor number = loaded library minor number
    ASSERT_EQ(LwSciBufCheckVersionCompatibility(LwSciBufMajorVersion,
            LwSciBufMinorVersion, &isCompatible), LwSciError_Success)
        << "LwSciBufCheckVersionCompatibility returned error";
    ASSERT_EQ(isCompatible, true)
        << "Expected Compatibility to be true, but returned false";

    // Check with valid major & minor versions
    // API major number = loaded library major number
    // API minor number < loaded library minor number
    ASSERT_EQ(LwSciBufCheckVersionCompatibility(LwSciBufMajorVersion,
            LwSciBufMinorVersion - 1U, &isCompatible), LwSciError_Success)
        << "LwSciBufCheckVersionCompatibility returned err";
    ASSERT_EQ(isCompatible, true)
        << "Expected Compatibility to be true, but returned false";

    // Check with valid major & invalid minor versions
    // API major number = loaded library major number
    // API minor number > loaded library minor number
    ASSERT_EQ(LwSciBufCheckVersionCompatibility(LwSciBufMajorVersion,
            LwSciBufMinorVersion + 1U, &isCompatible), LwSciError_Success)
        << "LwSciBufCheckVersionCompatibility returned err";
    ASSERT_EQ(isCompatible, false)
        << "Expected Compatibility to be false, but returned true";

    // Check with invalid major version
    // API major number < loaded library major number
    // API minor number = loaded library minor number
    ASSERT_EQ(LwSciBufCheckVersionCompatibility(LwSciBufMajorVersion - 1U,
            LwSciBufMinorVersion, &isCompatible), LwSciError_Success)
        << "LwSciBufCheckVersionCompatibility returned err";
    ASSERT_EQ(isCompatible, false)
        << "Expected Compatibility to be false, but returned true";

    // Check with invalid major version
    // API major number > loaded library major number
    // API minor number = loaded library minor number
    ASSERT_EQ(LwSciBufCheckVersionCompatibility(LwSciBufMajorVersion + 1U,
            LwSciBufMinorVersion, &isCompatible), LwSciError_Success)
        << "LwSciBufCheckVersionCompatibility returned err";
    ASSERT_EQ(isCompatible, false)
        << "Expected Compatibility to be false, but returned true";
}
