/*
 * Copyright (c) 2019-2020 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "lwscisync_test_common.h"

static void showUsage(void)
{
    printf("\nUsage: test_lwscisync_api [options]\n");
    printf("\ntest_lwscisync_api options:\n");
    printf("-submit-size <n>                             "
            "Makes #n submits. Default: "
            "0x400, Max: 0x%x.\n", LWSCISYNC_TEST_UNREASONABLE_SUBMIT_SIZE);
    printf("-reportProgress <n>                          "
            "Print additional data on tests' progress. "
            "Default: 0.\n");
    printf("-verbose <n>                              "
            "Print additional data provided by tests. "
            "Default: 0.\n");
    printf("\nsubtest details:\n");
}

static void printTooLargeSubmitSize(void)
{
    printf("test_lwscisync_api has the option of running\n");
    printf("with -submit-size over 0x100000 experimentally or for debug\n");
    printf("purposes but it is not intended to be used in QA testing\n");
    printf("If you are running this test as part of regular testing,\n");
    printf("then you are doing something wrong\n");
    printf("Litany of failure for scripts: FAIL ERROR FAILURE BREAK\n");
    printf("LOSS FIASCO\n");
}

int main(int argc, char* argv[])
{
    unsigned int param;

    ::testing::InitGoogleTest(&argc, argv);

    /* parse args */
    while (argc > 1)
    {
        --argc; ++argv;
        const char *arg = *argv;

        if (strcmp(arg, "-submit-size") == 0 && argc > 1)
        {
            --argc;
            ++argv;
            param = strtoul(*argv, NULL, 0);
            if (param == 0) {
                printf("invalid submit-size %d\n", param);
                return EXIT_FAILURE;
            }
           TestInfo::get()->submitSize = param;
        } else if (strcmp(arg, "-info") == 0) {
            showUsage();
            TestInfo::get()->showDescription = true;
        } else if (strcmp(arg, "-reportProgress") == 0 && argc > 1) {
            --argc;
            ++argv;
            param = strtoul(*argv, NULL, 0);
            if (param > 1) {
                printf("invalid reportProgress %d\n", param);
                return EXIT_FAILURE;
            }
            TestInfo::get()->reportProgress = param;
        } else {
            showUsage();
            TestInfo::get()->showDescription = true;
        }
    }

    /* show parameters */
    printf("submit-size:\t%u\n", TestInfo::get()->submitSize);
    printf("reportProgress:\t%u\n", TestInfo::get()->reportProgress);

    /* support all submitSizes but warn if large */
    if (TestInfo::get()->submitSize > LWSCISYNC_TEST_UNREASONABLE_SUBMIT_SIZE) {
        printTooLargeSubmitSize();
    }

    int result = RUN_ALL_TESTS();

    if (result == 0) {
        printf("PASSED\n");
    }

    /* print the warn message once again to be easily visible after the test */
    if (TestInfo::get()->submitSize > LWSCISYNC_TEST_UNREASONABLE_SUBMIT_SIZE) {
        printTooLargeSubmitSize();
    }

    return result;
}
