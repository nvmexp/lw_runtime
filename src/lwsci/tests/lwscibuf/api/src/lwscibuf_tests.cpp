/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "ipc_wrapper.h"
// This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0
#include "lwscibuf_test_platform.h"
#include "gtest/gtest.h"

void parseArgs (
    int argc,
    char *argv[])
{
    /* Do nothing */
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    parseArgs(argc, argv);

    /* Automatically initialize S3 logging APIs after fork() */
    pthread_atfork(NULL, NULL, &TestLwSciBufPlatformForkHandler);

    int result = RUN_ALL_TESTS();

    if (result == 0) {
        printf("PASSED\n");
    }
    return result;
}
