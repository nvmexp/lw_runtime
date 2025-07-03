#!/bin/python

# _LWRM_COPYRIGHTBEGIN
#
# Copyright 2019 by LWPU Corporation. All rights reserved. All
# information contained herein is proprietary and confidential to LWPU
# Corporation. Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# _LWRM_COPYRIGHTEND

"""
This is a very simple script for generating a CPP header file that is built
into the testListHelper binary.  The script simply generates a header that
defines a couple of variables: the suite name list and the platform-specific
binary suffix. These variables are passed in from the makefile and is based on
build-time variables.  These variables will also depend on the target platform
(i.e. ".exe" is suffix for Windows binaries, but Android uses "_srt").  The
resulting header is included in the exelwtable "testListHelper" so the
parameters can be queried at run-time by scripts.

Inputs:
    1. Binary suffix (string)
    2. Suite list with each suite name on a newline (string)

Example:
    python gen_helper.py .exe "suite_test_name_1 suite_test_name_2 suite_test_name_3"

Generates:

    const char * g_testlist_srt[] = {
        srt_test_name_1,
        srt_test_name_2,
        srt_test_name_3,
    };

    const char * g_bin_suffix = ".exe";
"""

import sys


def print_usage():
    """Print out the usage of this script to the console"""
    print(sys.argv[0] + " <binary suffix> <space-separated"
                        " string of suites (single argument!! Use double"
                        " quotes)> <helper runnner python file>")


def exit_program_from_error(msg):
    """Print an error message and exit the program"""
    sys.exit(sys.argv[0] + ": Error: " + msg)


def generate_cpp_header():
    """ Parse the input arguments and create a combined test list dictionary
        containing all suites and subtests along with optional conditions for
        skipping exelwtion of each.

        This function assums argv[1] is the file name of the skip list, argv[2]
        is the binary suffix string.

        Return value:


        A dictionary of two key/value pairs:
          "TestListData" : The JSON string of the test list data,

          "QueryTemplate" : The input template python file which the JSON
          string of test list data will be prepended to.
        """

    total_args = len(sys.argv)

    if total_args != 3:
        print_usage()
        exit_program_from_error("Exiting... Invalid number of arguments.")

    bin_suffix = sys.argv[1]

    suite_list = sys.argv[2]

    print ("const char * g_testlist_str[] = {")
    for suite_name in suite_list.split():
        print("\"" + suite_name + "\",")
    print("};")
    print("")
    print ("const char * g_bin_suffix = \"" + bin_suffix + "\";\n")

generate_cpp_header()
