/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*
 * @brief main file to launch the test cases
 */

#include <stdio.h>
#include "rmutapis.h"
#include "utility.h"

LwSuite* suite_fifoServiceTop_GF100();

void RunAllTests(unsigned int bVerboseSummary)
{
    LwSuite  *suite = UTAPI_NEW_SUITE(NULL, NULL);
    LwString *output = UTAPI_NewString();

    // Test for fbAllocGF_100()
    UTAPI_AddSuite(suite, suite_fifoServiceTop_GF100());

    // Now suites are ready to run.
    UTAPI_ExelwteSuite(suite, bVerboseSummary);

    // test reprots will be available here
    UTAPI_SuiteDetailInfo(suite, output);
    printf("%s\n", output->buffer);

    UTAPI_DestroySuite(suite);
}

int main(int argc, char *argv[])
{
    switch (argc)
    {
    case 1:// default case non-verbose summary
        RunAllTests(0);
        break;
    case 2:
        if (strcmp(argv[1],"-verbose") == 0 || strcmp(argv[1],"-Verbose") == 0)
            RunAllTests(1);
        else
        {
            printf("\n Only -verbose option is allowed");
            return 1;
        }
        break;
    default:
        printf("\n Only -verbose option is allowed");
        return 1;
    }
    closeLogFile();
    return 0;
}
