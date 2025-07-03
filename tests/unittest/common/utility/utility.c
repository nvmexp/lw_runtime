/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2009 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file   utility.c
 * @brief  Logging related functions
 */

#include <stdio.h>
#include "utility.h"

static FILE *pUtLogFile;

static char *logName = "unitTestAssert.log";

/*!
 * @brief opens the file for logging
 */
void initLogFile()
{
    if((pUtLogFile = fopen(logName, "w")) == NULL)
    {
        printf("can not open the Log file \n");
        return;
    }
}

/*!
 * @brief writes the log into the file
 *
 * @param[in]      file    name of teh file in which assertion failed
 * @param[in]      line    line on which assertion failed
 *
 */
void unitPrintf(char *file, int line)
{
    if (!pUtLogFile)
            initLogFile(); // for logging asserts

    if (pUtLogFile)
    {
        fprintf(pUtLogFile,
        "Error::Assertion Failed at File:%s Line:%d\n", file, line);
    }
}

/*!
 * @brief closes the file after logging
 */
void closeLogFile()
{
    if (pUtLogFile)
    {
        fclose(pUtLogFile);
    }
}

int checkIfAssertLogFile()
{
    if (pUtLogFile)
        return 1;

    return 0;
}
