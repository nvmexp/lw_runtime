/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2015 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 *
 * File: commandline.h
 *
 * Previous locations:
 *     //sw/apps/gpu/drivers/resman/commandline/commandline.h
 *     //sw/dev/gpu_drv/chips_a/apps/lwml/common/commandline/commandline.h
 *     //sw/dev/gpu_drv/chips_a/drivers/common/shared/inc/commandline.h
 *
 * This file, along with its accompanying commandline.c, provides a very small,
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

#ifndef __LW_APP_LIBS_COMMANDLINE_H
#define __LW_APP_LIBS_COMMANDLINE_H

#if defined(__cplusplus)
extern "C" {
#endif

enum cmdline_option_value_existence
{
    CMDLINE_OPTION_NO_VALUE_ALLOWED,
    CMDLINE_OPTION_VALUE_OPTIONAL,
    CMDLINE_OPTION_VALUE_REQUIRED,
};

struct all_args
{
   int    internalName;
   const char * shortForm;
   const char * longForm;
   const char * shortHelpString;
   const char * longHelpString;
   int    mayHaveAFollowingValue;
};

int cmdline_init(int argc, char * argv[],
                 const struct all_args * allArgs,
                 int numberElementsInAllArgs,
                 void ** pCmdLine);

enum cmdline_check_options
{
    CMDLINE_CHECK_OPTIONS_SUCCESS,
    CMDLINE_CHECK_OPTIONS_UNKNOWN_OPTION_FOUND,
    CMDLINE_CHECK_OPTIONS_MISSING_REQUIRED_VALUE,
};

enum cmdline_check_options
cmdline_checkOptions(const void * pCmdLine,
                     const char ** firstUnknownOptionName,
                     const char ** optionWithMissingValue);

int cmdline_exists(const void * pCmdLine, int argName);

int cmdline_getStringVal(const void * pCmdLine, int argName, const char ** val);
int cmdline_getIntegerVal(const void * pCmdLine, int argName,
                          unsigned long long * val);

int cmdline_printOptionsSummary(const void * pCmdLine, int useLongHelpStrings);
int cmdline_destroy(void * pCmdLine);

#if defined(__cplusplus)
}
#endif

#endif
