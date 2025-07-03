/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2015 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 *
 * File: commandline.c
 *
 * Previous locations:
 *     //sw/apps/gpu/drivers/resman/commandline/commandline.c
 *     //sw/dev/gpu_drv/chips_a/apps/lwml/common/commandline/commandline.c
 *     //sw/dev/gpu_drv/chips_a/drivers/common/shared/commandline/commandline.c
 *
 * This file, along with its accompanying commandline.h, provides a very small,
 * simple command line parser.
 *
 * It is also supposed to be:
 *
 *      -- Easy to use: the API is small and minimalistic, but you only need to
 *      make three API calls in order to get some use out of it:
 *
 *      1) Initialize, via cmdline_init().
 *
 *      2) Check for an argument, via cmdline_exists().
 *
 *      3) Check for an argument and get it's value if it does exist, via
 *         cmdline_getStringVal() or cmdline_getIntegerVal().
 *
 *      -- Easy to build: Just add commandline.c to your project's source file
 *         list, and the commandline directory to your build systems'
 *         INCLUDE path.
 *
 *     -- Policy-free: the arguments, including any leading dashes or slashes
 *        or whatever you like, are entirely specified by the programmer. The
 *        only real policy is that all arguments and values must be separated
 *        by spaces (or whatever your command shell uses, in order to construct
 *        the argv[] array).  This rules out the unix-getopt-style of typing:
 *
 *              -abcdf filename.txt -gh  (NOT ALLOWED)
 *
 *        ...when what you really meant was:
 *
 *              -a -b -c -d -f filename.txt -g -h
 *
 *      -- Easy to understand and maintain. There is no parsing code here.
 *         The arguments, as parsed by whatever command shell is running, are
 *         already in the argv[] array. Furthermore, argv[] is not modified.
 *
 *      -- Portable: this is pure C code, and will run on any OS. It depends
 *         only on a small part of the C library.
 *
 *      -- Unenlwmbered. Written from scratch at LWPU, in September, 2010. We
 *         are therefore free to use this code internally, externally,
 *         wherever we like.
 *
 * For an example, please see:
 *     //sw/dev/gpu_drv/chips_a/drivers/common/shared/utils/commandline/commandline_unit_test.c
 *
 * Additional examples:
 *     //sw/dev/gpu_drv/chips_a/apps/lwDebugDump/lwdd_main.c
 */

#if defined(_WIN32) || defined(_WIN64)
#define strtoull _strtoui64
#endif

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "commandline.h"

struct internal_cmd_line
{
    int               argc;
    char **           argv;
    const struct all_args * allArgs;
    int               nElements;
};

int cmdline_init(int argc, char * argv[],
                 const struct all_args * allArgs,
                 int numberElementsInAllArgs,
                 void ** pCmdLine)
{
    struct internal_cmd_line * pcl = (struct internal_cmd_line*) malloc(sizeof(struct internal_cmd_line));

    if (pcl)
    {
        memset(pcl, 0, sizeof(struct internal_cmd_line));
        pcl->argc      = argc;
        pcl->argv      = argv;
        pcl->allArgs   = allArgs;
        pcl->nElements = numberElementsInAllArgs;

        *pCmdLine = pcl;
        return 0; // Success
    }

    return -1;
}

static int internal_cmdline_exists(const void * pCmdLine, int argName)
{
    const struct internal_cmd_line * pcl = (struct internal_cmd_line *)pCmdLine;
    int argIndex = 0;
    char * arg = NULL;

    if (pcl->argc < 2 || argName >= pcl->nElements)
    {
        return 0;
    }

    for (argIndex = 1; argIndex < pcl->argc; ++argIndex)
    {
        arg = pcl->argv[argIndex];

        if (0 == strcmp(pcl->allArgs[argName].shortForm, arg))
        {
            return argIndex; // Found short form of the argument
        }
        if (0 == strcmp(pcl->allArgs[argName].longForm, arg))
        {
            return argIndex; // Found long form of the argument
        }
    }
    return 0;
}

enum cmdline_check_options
cmdline_checkOptions(const void * pCmdLine,
                     const char ** firstUnknownOptionName,
                     const char ** optionWithMissingValue)
{
    const struct internal_cmd_line * pcl = (struct internal_cmd_line *)pCmdLine;
    char * arg           = NULL;
    char * prevArg       = NULL;
    int knownOption      = 0;
    int argIndex         = 0;
    int found            = 0;
    int mayHaveValue     = 0;
    int prevMayHaveValue = 0;

    for (argIndex = 1; argIndex < pcl->argc; ++argIndex)
    {
        prevArg   = arg;
        arg = pcl->argv[argIndex];
        found = 0;

        prevMayHaveValue = mayHaveValue;
        mayHaveValue     = 0;

        for (knownOption = 0; knownOption < pcl->nElements; ++knownOption)
        {
            if (0 == strcmp(pcl->allArgs[knownOption].shortForm, arg) ||
                0 == strcmp(pcl->allArgs[knownOption].longForm, arg))
            {
                found = 1;
                mayHaveValue = pcl->allArgs[knownOption].mayHaveAFollowingValue;
                break; // Found a short or long form of the argument
            }
        }

        if (!found && !prevMayHaveValue)
        {
            if (firstUnknownOptionName)
            {
                *firstUnknownOptionName = arg;
                *optionWithMissingValue = NULL;
            }
            return CMDLINE_CHECK_OPTIONS_UNKNOWN_OPTION_FOUND;
        }
        else if (found && (prevMayHaveValue == CMDLINE_OPTION_VALUE_REQUIRED))
        {
            // Should not have found a new option, when we were expecting
            // an option value that is marked as "required":
            *firstUnknownOptionName = NULL;
            *optionWithMissingValue = prevArg;
            return CMDLINE_CHECK_OPTIONS_MISSING_REQUIRED_VALUE;
        }
        else if (found && (mayHaveValue == CMDLINE_OPTION_VALUE_REQUIRED) &&
                 argIndex == pcl->argc - 1)
        {
            // Should not have hit the LAST option, if it has
            // an option value that is marked as "required":
            *firstUnknownOptionName = NULL;
            *optionWithMissingValue = arg;
            return CMDLINE_CHECK_OPTIONS_MISSING_REQUIRED_VALUE;
        }
    }
    return CMDLINE_CHECK_OPTIONS_SUCCESS;
}

int cmdline_exists(const void * pCmdLine, int argName)
{
    return !!internal_cmdline_exists(pCmdLine, argName);
}

int cmdline_getStringVal(const void * pCmdLine, int argName, const char ** val)
{
    const struct internal_cmd_line * pcl = (struct internal_cmd_line *)pCmdLine;
    int index = internal_cmdline_exists(pCmdLine, argName);

    if ((index > 0) &&                  // If it was found
        ((index + 1) < pcl->argc) &&    // ...and there is a following arg
        (((pcl->argv[index + 1])[0] != '-') || // ...that doesn't start with a dash
         ((pcl->argv[index + 1])[1] == '\0'))) // ...or the next arg is just a dash (for stdout/stdin)
    {
        *val = pcl->argv[index + 1];
        return 1; // TRUE: value exists and is not another argument
    }

    return 0; // FALSE: value was not found
}

int cmdline_getIntegerVal(const void * pCmdLine, int argName,
                          unsigned long long * val)
{
    const char * strVal = NULL;

    int result = cmdline_getStringVal(pCmdLine, argName, &strVal);
    if (result)
    {
        *val = strtoull(strVal, 0, 0);
        return 1; // TRUE: value exists and is not another argument
    }

    return 0; // FALSE: value was not found
}

int cmdline_printOptionsSummary(const void * pCmdLine, int useLongHelpStrings)
{
    const struct internal_cmd_line * pcl = (struct internal_cmd_line *)pCmdLine;
    int optIndex = 0;

    while (optIndex < pcl->nElements)
    {
        const char * pHelpStr = NULL;
        if (useLongHelpStrings)
        {
            pHelpStr = pcl->allArgs[optIndex].longHelpString;
            printf("\n");
        }
        else
        {
            pHelpStr = pcl->allArgs[optIndex].shortHelpString;
        }

        printf("    [%s | %s]: %s\n",
               pcl->allArgs[optIndex].shortForm,
               pcl->allArgs[optIndex].longForm,
               pHelpStr);
        ++optIndex;
    }
    return 0;
}

int cmdline_destroy(void * pCmdLine)
{
    struct internal_cmd_line * pcl = (struct internal_cmd_line *)pCmdLine;
    free(pcl);
    return 0;
}

