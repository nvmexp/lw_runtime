/*
 * Copyright (c) 2019-2021 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <lwscisync_test_common.h>
#include <lwscisync_test_signaler.h>
#include <lwscisync_test_waiter.h>
#include <lwscisync.h>
#include <umd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <cinttypes>

/** @jama{12764137} Version compatibility check
 */
LWSCISYNC_DECLARE_TEST(TestVersionCompatibility, CheckVersionCompatibility, 12764137)
{
    LwSciError err;
    bool isCompatible = false;

    // Check with valid major & minor versions
    err = LwSciSyncCheckVersionCompatibility(LwSciSyncMajorVersion,
            LwSciSyncMinorVersion, &isCompatible);
    ASSERT_EQ(err, LwSciError_Success);
    ASSERT_EQ(isCompatible, true);

    // Check with valid major & invalid minor versions
    if (LwSciSyncMinorVersion) {
        err = LwSciSyncCheckVersionCompatibility(LwSciSyncMajorVersion,
                LwSciSyncMinorVersion - 1U, &isCompatible);
        ASSERT_EQ(err, LwSciError_Success);
        ASSERT_EQ(isCompatible, true);
    }

    err = LwSciSyncCheckVersionCompatibility(LwSciSyncMajorVersion,
            LwSciSyncMinorVersion + 1U, &isCompatible);
    ASSERT_EQ(err, LwSciError_Success);
    ASSERT_EQ(isCompatible, false);

    // Check with invalid major version
    if (LwSciSyncMajorVersion) {
        err = LwSciSyncCheckVersionCompatibility(LwSciSyncMajorVersion - 1U,
                LwSciSyncMinorVersion, &isCompatible);
        ASSERT_EQ(err, LwSciError_Success);
        ASSERT_EQ(isCompatible, false);
    }

    err = LwSciSyncCheckVersionCompatibility(LwSciSyncMajorVersion + 1U,
            LwSciSyncMinorVersion, &isCompatible);
    ASSERT_EQ(err, LwSciError_Success);
    ASSERT_EQ(isCompatible, false);
}
