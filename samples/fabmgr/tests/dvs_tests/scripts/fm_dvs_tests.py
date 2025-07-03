#! /usr/bin/elw python

# Copyright 2018-2021 LWPU Corporation.  All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.

# The purpose of having DVS call this script rather than calling
# run_tests_fm.py directly is to allow us to easily change the arguments that
# are passed to run_tests_fm.py in case the script changes its
# arguments/functionality, or if the relative locations of the inputs ever
# change
#
# The arguments that this script takes from DVS are:
#   * perCl or nightly
#   * debug or release
#
# The arguments that this script will pass to run_tests_fm.py are:
# --bin-loc
#    In the current FM package, the test binaries are in the script's
#    parent directory on Linux. These tests are lwrrently not supported on
#    Windows.
#
# --helper-bin-name testListHelper
#    When GVS runs this script on mobile, it needs to use a precompiled
#    helper binary to deternmine what tests to run do to limitations in the
#    GVS testing infrastructure. This argument may eventually be removed if
#    another solution is found, or we have the script behave differently
#    when run on desktop.
#
# --whitelist <path to whitelist>
#    This optional argument was created for per CL dvs runs to limit the
#    test suites/fixtures/tests to only those that are listed in the
#    json whitelist found in the location above. The SKIP_TESTS
#    blacklist is still applied on top of this whitelist. See
#    'run_tests_fm.py --help' for more details on how this feature works.
#
# --summary-log <filepath>
#    This argument produces the summarylog.txt file that DVS looks for to
#    parse the test results
#
# --dvs
#    Enables certain features that are disabled by default to maintain
#    compatibility with GVS.
#
# > fm_dvs_tests.log
#    The gtest standard output is saved in this log file, which DVS will
#    search for and upload along with the test results.
#
#
#    NOTE: run_tests_fm.py has a dependency on Python 2.7 due to Android/GVS
#          restrictions.

import os
import sys
import platform
import subprocess

def main():

    # get lowercase args and OS type
    args = sys.argv
    for i, a in enumerate(args):
        args[i] = a.lower()

    OS = platform.system()

    if ((OS == 'Linux') and ('percl' in args) and ('release' in args)):
        with open("fm_dvs_tests.log", "w") as log:
            log.write("Exelwting tests for: Linux Per Changelist Release\n\n")
        os.system('python2.7 run_tests_fm.py --bin-loc ../ --whitelist ./fm_dvs_config/dvs_per_cl_whitelist.json --dvs --summary-log summarylog.txt >> fm_dvs_tests.log 2>&1')

    elif ((OS == 'Linux') and ('nightly' in args) and ('release' in args)):
        with open("fm_dvs_tests.log", "w") as log:
            log.write("Exelwting tests for: Linux Nightly Release\n\n")
        os.system('python2.7 run_tests_fm.py --bin-loc ../ --dvs --summary-log summarylog.txt >> fm_dvs_tests.log 2>&1')

    elif ((OS == 'Linux') and ('percl' in args) and ('debug' in args)):
        with open("fm_dvs_tests.log", "w") as log:
            log.write("Exelwting tests for: Linux Per Changelist Debug\n\n")
        os.system('python2.7 run_tests_fm.py --bin-loc ../ --ignore-blacklist --whitelist ./fm_dvs_config/dvs_debug_whitelist.json --dvs --summary-log summarylog.txt >> fm_dvs_tests.log 2>&1')

    elif ((OS == 'Linux') and ('nightly' in args) and ('debug' in args)):
        with open("fm_dvs_tests.log", "w") as log:
            log.write("Exelwting tests for: Linux Nightly Debug\n\n")
        os.system('python2.7 run_tests_fm.py --bin-loc ../ --ignore-blacklist --whitelist ./fm_dvs_config/dvs_debug_whitelist.json --dvs --summary-log summarylog.txt >> fm_dvs_tests.log 2>&1')

    elif ((OS == 'Windows') and ('percl' in args) and ('release' in args)):
        with open("fm_dvs_tests.log", "w") as log:
            log.write("Exelwting tests for: Windows Per Changelist Release\r\n\r\n")
        os.system(r'python run_tests_fm.py --bin-loc .\ --whitelist dvs_per_cl_whitelist.json --dvs --summary-log summarylog.txt >> fm_dvs_tests.log 2>&1')

    elif ((OS == 'Windows') and ('nightly' in args) and ('release' in args)):
        with open("fm_dvs_tests.log", "w") as log:
            log.write("Exelwting tests for: Windows Nightly Release\r\n\r\n")
        os.system(r'python run_tests_fm.py --bin-loc .\ --dvs --summary-log summarylog.txt >> fm_dvs_tests.log 2>&1')

    elif ((OS == 'Windows') and ('debug' in args)):
        sys.exit("WINDOWS DEBUG FM RUNS NOT SUPPORTED")

    else:
        sys.exit("invalid/missing arguments. USAGE: fm_dvs_tests.py <debug/release> <percl/nightly>")


    # See if Python 3 is available on the machine
    with open("fm_dvs_tests.log", "a") as log:
        log.write(os.linesep*3 + "Current Python version: " + sys.version + os.linesep*3)
        log.write("Available Python 3 version: ")

    if OS == 'Linux':
        subprocess.call("python3 --version >> fm_dvs_tests.log 2>&1", shell=True)

    elif OS == 'Windows':
        subprocess.call("py -3 --version >> fm_dvs_tests.log 2>&1", shell=True)

if __name__ == "__main__":
    ret = main()
    sys.exit(ret)
