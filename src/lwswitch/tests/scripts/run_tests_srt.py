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
This script will execute or print out the commands that run all SRT test
suites.  The script by default will execute all the tests on the system; with
the --cmds-only option, the script will only print out which commands it would
use to execute the SRT tests on the system and not actually run them.

The script requires access to a prebuilt binary test helper (built inside the
SRT tree as part of the normal SRT test suite build process) to provide some
platform-specific information and build-time generated information.  The helper
binary is expected to be in the same location as the SRT test binaries
(somewhere in PATH or via explicit "--bin-loc" argument location).

The script's optional argument, "--helper-bin-name", provides the name of
the prebuilt helper binary. Prior to any SRT test list exelwtion or command
output, the script will execute the helper binary to 1) obtain the list of SRT
exelwtables, 2) obtain the extensions for each SRT exelwtable, and 3) determine
the type of GPU being used on the test system. This argument is set to
testListHelper on Linux and testListHelper.exe on Windows if not provided.

"--bin-loc" optional argument can be used to prefix a path to each SRT binary
exelwtion; otherwise without the argument, the script will try to execute SRT
tests without any prefix (i.e. typically they would need to be in the system's
PATH to be exelwtable this way).

"--whitelist" optional argument can be used to specify a path to a JSON
whitelist, similar in format to the SKIP_TESTS list below. Only the suites,
fixtures, and tests in this whitelist will be exelwted. The SKIP_TESTS
blacklist will still be applied on top of the whitelist, and suites that were
not compiled (not in test_list) will be skipped. Otherwise without the argument,
the script will run all tests that are not blacklisted on the current platform.
This argument is lwrrently not supported on Android/GVS until support for the
json standard Python library is added (LWBUG #2461139).

"--ignore-blacklist" - Optional argument; ignore contents of SKIP_TESTS. Acts
as if nothing was blacklisted.

"--cmds-only" - Optional argument; only print the commands used for exelwtion,
don't actually execute any of the tests (--helper-bin-name binary is still
exelwted)

"--dvs" - Optional argument; produce summarylog.txt file with score formatted
for parsing by DVS.

"--gsp" - Optional argument; disable tests with DISABLE_GSP constraint set in
SKIP_TESTS.

For details on each input to the script, see the function print_usage().

Example:

Assuming all SRT binaries are located in /vendor/bin and that is part of the
system's PATH, a user could execute all SRT tests (according to skiplist in
"SKIP_TESTS" dictionary in this script) via the simple command:

    python run_tests_srt.py --helper-bin-name "testListHelper.exe"

Or to just print out the commands one-by-one:

    python run_tests_srt.py --helper-bin-name "testListHelper.exe" --cmds_only

Note - ".exe" is required on the --helper-bin-name argument if on Windows.  If
on Linux, it it may simply be "--helper-bin-name testListHelper", or on
Android, could be "--helper-bin-name testListHelper_srt".
"""

import sys
import os
import re
import subprocess
import platform

# The following SKIP_TESTS is a maintained skiplist, along with the condition
# for skipping them.  SKIP_TESTS is a List of dictionaries for each suite that
# has suite-level, fixture-level, or subtest-level skips (all optional).
#
# Each suite dictionary in the list has the form:
#  {
#    "suiteName"  : "<SRT suite name>",
#    "constraint" : "<constraint reason for skipping>",
#    "fixtures"   : <List of fixture dictionaries to conditionally skip>
#  }
#
# Each "fixtures" dictionary is of a similar form:
#  {
#    "fixtureName" : "<SRT suite fixture name>",
#    "constraint"  : "<constraint reason for skipping>",
#    "subtests"    : "<List of subtest dictionaries to unconditionally skip>
#  }
#
# Each "subtests" dictionary is of a similar form:
#  {
#    "subtestName" : "<fixture subtest name>",
#    "constraint"  : "<constraint reason for skipping>",
#  }
#
# Valid values for "constraint" in all cases:
#   DISABLE                 - Unconditionally disable on all platforms.
#       DISABLE_DESKTOP     - Disable on all desktop GPU systems. (DVS)
#           DISABLE_LINUX   - Disable on Linux systems            (DVS)
#               DISABLE_GSP - Disable on GSP RM test              (DVS)
#           DISABLE_WINDOWS - Disable on Windows systems          (DVS)
#       DISABLE_MOBILE      - Disable on all mobile GPU systems.  (GVS)
#
# You can see the heirarchy of keywords above. Multiple keywords can be listed
# and the strictest set of kewords will be applied. Entries such as:
#       "constraint" : "DISABLE_MOBILE, DISABLE_DESKTOP, DISABLE_WINDOWS"
# are allowed, but it is best to be as concise as possible and use the
# funtionally equivalent entry of:
#       "constraint" : "DISABLE"
# unless there are different reasons/bugs tracking the disablement of the
# test(s) on each platform.
#
# All constraints can have an optional "(...reason for disable...)" at the end
# in order to take notes of bug numbers or provide additional justification.  The
# script will ignore the trailing parenthesis.  For example, the following is
# a valid constraint for disabling on mobile platforms with some info about why:
#   DISABLE_MOBILE(See bug 1234567)
#
# Note that "constraint" is optional in each case.  If "constraint" appears in
# the top-level suite dictionary, that entire suite is skipped.  If
# "constraint" appears in a "fixtures" dictionary, only that fixture is
# skipped.  If a subtest has "constraint", then only that subtest is skipped
# (i.e. rest of fixture/suite is exelwted according to other constraints).
#
# For any suite/test/fixture that doesn't have any entry in this skiplist, or
# the constraint dictates it should NOT be skipped on this platform, then it
# will be automatically exelwted if it is whitelisted or no whitelist is used.
#
# Entries in the whitelist/blacklist can also have an "arguments" field at the
# suite level. The string provided in this field will be passed to the
# exelwtable when it is called. The arguments strings from both the whitelist
# and blacklist are combined if both provide one. Argument strings from the
# blacklist are not used if '--ignore-blacklist' is set.


SKIP_TESTS = [
    {
        "suiteName" : "exampleSuite",
        "fixtures" : [
            {
                "fixtureName" : "ExampleGroup/AllocatorExample",
                "subtests" :
                [
                    {
                        "subtestName" : "subtestName*",
                        "constraint" : "DISABLE_MOBILE"
                    }
                ]
            }
        ]
    }
]

def validate_skip_tests():
    valid = True
    for suite in SKIP_TESTS:
        valid = valid and validate_suite(suite)

    return valid

def validate_subtest(subtest):
    valid = True
    valid_subtest_keys = {"subtestName", "constraint"}
    keys = set(subtest.keys())
    unknown_keys = keys - valid_subtest_keys
    if unknown_keys:
        print("Invalid subtest keys: {}".format(unknown_keys))
        valid = False

    if "contraint" in keys:
        if not is_constraint_valid(subtest["contraint"]):
            print("Constraint is invalid: {}".format(subtest["contraint"]))
            valid = False

    return valid

def validate_fixture(fixture):
    valid = True
    valid_fixture_keys = {"fixtureName", "constraint", "subtests"}
    keys = set(fixture.keys())
    unknown_keys = keys - valid_fixture_keys
    if unknown_keys:
        print("Invalid fixture keys: {}".format(unknown_keys))
        valid = False

    if "contraint" in keys:
        if not is_constraint_valid(fixture["contraint"]):
            print("Constraint is invalid: {}".format(fixture["contraint"]))
            valid = False

    if "subtests" in keys:
        for subtest in fixture["subtests"]:
            valid = valid and validate_subtest(subtest)

    return valid

def validate_suite(suite):
    valid = True
    valid_suite_keys = {"suiteName", "constraint", "fixtures", "arguments"}
    keys = set(suite.keys())
    unknown_keys = keys - valid_suite_keys
    if unknown_keys:
        print("Invalid suite keys: {}".format(unknown_keys))
        valid = False

    if "contraint" in keys:
        if not is_constraint_valid(suite["contraint"]):
            print("Constraint is invalid: {}".format(suite["contraint"]))
            valid = False

    if "fixtures" in keys:
        for fixture in suite["fixtures"]:
            valid = valid and validate_fixture(fixture)

    return valid


def is_constraint_valid(constraint):
    # Remove comment portion of constraint
    constraint = constraint.split("(")[0]
    valid_values = ["DISABLE", "DISABLE_DESKTOP", "DISABLE_LINUX", "DISABLE_GSP", "DISABLE_WINDOWS", "DISABLE_MOBILE"]
    return constraint in valid_values

def print_usage():
    """Print out the usage of this script to the console"""

    print("\n" + sys.argv[0] +
          " [--helper-bin-name <filename of srt_testlist_helper "
          "binary>] [--bin-loc <optional custom path to binary files>] "
          "[--whitelist <path to optional custom whitelist>] "
          "[--ignore-blacklist] [--cmds_only] [--dvs] "
          "[--summary-log <filename>] [--gsp] \n\n")

    print("  --helper-bin-name <arg> - Required argument; full file name\n "
          "\t(including file extension) of the srt_testlist_helper binary.  The\n"
          "\t[--bin_loc] argument, if specified, will be prepended to this for lookup\n "
          "\tfor exelwtion.\n")

    print("  --bin-loc <path> - Optional argument; custom path to binaries to\n"
          "\tbe prependend for exelwtion.\n")

    print("  --whitelist <path> - Optional argument; feed custom json whitelist\n"
          "\tof tests/suites to run. If a suite is included with no fixtures, the\n"
          "\tentire suite is run, otherwise only the listed fixtures will be run.\n"
          "\tSimilarly, if a fixure is listed with no subtests, all tests in the\n"
          "\tfixture are run, otherwise only the listed tests will be run. Suites\n"
          "\tnot included in this file will not be run. The SKIP_TESTS blacklist\n"
          "\tis still applied on top of this whitelist. In the absence of this\n"
          "\targument, all exelwtables returned by testListHelper will be\n"
          "\tdirectly fed to the blacklist.\n")

    print("  --ignore-blacklist - Optional argument; ignore contents of\n"
          "\tSKIP_TESTS. Acts as if nothing was blacklisted\n")

    print("  --cmds-only - Optional argument; only print the commands used for\n"
          "\texelwtion, don't actually execute any of the tests (--helper_bin_name\n"
          "\tbinary is still exelwted)\n")

    print("  --dvs - Optional argument; produce summarylog.txt file with score\n"
          "\tformatted for parsing by DVS\n")

    print("  --summary-log <filename> - Optional argument; produce <filename> with \n"
          "\tscore formatted for parsing by DVS (same as --dvs with option to name file\n")

    print("  --gsp - Optional argument; disable tests with DISABLE_GSP constraint\n"
          "\tset in SKIP_TESTS\n")



def exit_program_from_error(msg):
    """Print an error message and exit the program"""
    sys.exit(sys.argv[0] + ": Error: " + msg)

def merge_suites_with_skiplist(binloc, helper_bin):
    """ Merges the binary names retrieved from the testListHelper reported test
        with the skip tests.

        Returns a merged dictionary with the skiptests and the
        binaries
    """
    test_list = exelwte_single_test(binloc + helper_bin + " --testlist").split()

    ret_tests = []

    for test_name in test_list:
        # Check if it's already in the skiplist
        found_skip_dict = None

        for skip_test_dict in SKIP_TESTS:
            skip_name = skip_test_dict.get("suiteName")
            if skip_name == test_name:
                # It is in the skip list - so just use that dict
                found_skip_dict = skip_test_dict
                break

        if found_skip_dict is not None:
            ret_tests.append(found_skip_dict)
        else:
            ret_tests.append({"suiteName" : test_name})

    return ret_tests

def parse_arguments():
    """Parse the input command line arguments.

    Return value -- A dictionary containing all the enabled suites, fixtures,
    and sub-tests along with their conditions of exelwtion
    """

    # IMPORTANT NOTE: This script does not use 'argparse' to parse command-line
    # arguments because some mobile systems' Python distribution do not have
    # the 'argparse' module available - so we resort to a simple manual method.

    # set reasonable defaults
    bin_loc = ""
    helper_bin_name = "testListHelper"
    if platform.system() == 'Windows':
        helper_bin_name += '.exe'

    # flag to specify only commands should be output; don't actually execute
    # them.
    do_commands_only = False

    valid_args = ["--help", "-h", "--bin_loc", "--bin-loc", "--cmds_only", "--cmds-only",
                  "--helper_bin_name", "--helper-bin-name", "--summary-log",
                  "--dvs", "--whitelist", "--ignore-blacklist", "--gsp", "--validate"]
    total_args = len(sys.argv)

    skip_next_arg = False
    called_from_dvs = False
    ignore_blacklist = False
    run_on_gsp_rm = False
    summary_log_name = None
    whitelist = None
    whitelist_path = None

    for ndx, arg in enumerate(sys.argv):
        if skip_next_arg is True or ndx == 0:
            skip_next_arg = False
            continue

        if arg not in valid_args:
            print_usage()
            exit_program_from_error("Invalid argument: " + arg)

        elif arg == "--help" or arg == "-h":
            print_usage()
            sys.exit()

        elif arg == "--validate":
            if validate_skip_tests():
                print("SKIP_TEST configuration is valid")
            sys.exit()

        elif arg == "--bin_loc" or arg == "--bin-loc":
            if ndx == (total_args - 1) or sys.argv[ndx+1].startswith("--"):
                exit_program_from_error("--bin-loc must have argument supplied.")
            bin_loc = sys.argv[ndx+1]
            skip_next_arg = True

        elif arg == "--helper_bin_name" or arg == "--helper-bin-name":
            if ndx == (total_args - 1) or sys.argv[ndx+1].startswith("--"):
                exit_program_from_error("--helper-bin-name must have argument supplied.")
            helper_bin_name = sys.argv[ndx+1]
            skip_next_arg = True

        elif arg == "--whitelist":
            if ndx == (total_args - 1) or sys.argv[ndx+1].startswith("--"):
                exit_program_from_error("--whitelist must have argument supplied.")
            whitelist_path = sys.argv[ndx+1]
            skip_next_arg = True

        elif arg == "--cmds_only" or arg == "--cmds-only":
            do_commands_only = True

        elif arg == "--dvs":
            called_from_dvs = True

        elif arg == "--summary-log":
            if ndx == (total_args - 1) or sys.argv[ndx+1].startswith("--"):
                exit_program_from_error("--summary-log must have argument supplied.")
            summary_log_name = sys.argv[ndx+1]
            skip_next_arg = True

        elif arg == "--ignore-blacklist":
            ignore_blacklist = True

        elif arg == "--gsp":
            if platform.system() != "Linux":
                exit_program_from_error("'--gsp': GSP RM is only supported on Linux.")
            run_on_gsp_rm = True

    # Make sure testListHelper can be found in either location specified by
    # bin_loc (DVS), or in environment PATH (GVS)
    helper_found = False
    if os.path.exists(bin_loc + helper_bin_name):
        helper_found = True
    else:
        for path in os.elwiron['PATH'].split(os.pathsep):
            if os.path.exists(os.path.join(path, helper_bin_name)):
                helper_found = True
#    if not helper_found:
#        exit_program_from_error("Could not find " + bin_loc + helper_bin_name +
#                                ". Check --bin-loc and --helper-bin-name")

    json_data = merge_suites_with_skiplist(bin_loc, helper_bin_name)

    # The standard json library is not supported on Android Python, which is why
    # it is imported within this conditional. Therefore, providing a custom
    # whitelist will be unsupported on GVS until support is added. LWBug 2461139
    # has been filed to track this RFE.
    if whitelist_path is not None:
        import json
        with open(whitelist_path) as whitelist_file:
            whitelist_data = whitelist_file.read()
        whitelist = json.loads(whitelist_data)

    return {"JsonData": json_data,
            "Whitelist" : whitelist,
            "BinLoc" : bin_loc,
            "DoCommandsOnly" : do_commands_only,
            "HelperBinName" : helper_bin_name,
            "CalledFromDvs" : called_from_dvs,
            "SummaryLogName" : summary_log_name,
            "IgnoreBlacklist" : ignore_blacklist,
            "RunOnGspRm" : run_on_gsp_rm}


def should_disable(constraint_name, gpu_type):
    """Whether the constraint indicates that the test should be disabled given
    the input gpu_type.

    Return value:

    True - If the constraint should disable the test and the script was not
           called with the --ignore-blacklist argument

    False - If the constraint should not disable the test or the script was
            called with the --ignore-blacklist argument
    """

    if constraint_name is None or ARG_DICT["IgnoreBlacklist"]:
        return False

    if "igpu" not in gpu_type and "dgpu" not in gpu_type:
        exit_program_from_error("Invalid GPU type retrieved from helper binary: " +
                                " : " + gpu_type)

    # Colwert to lowercase, remove optional comments, split, and strip spaces
    constraint_string = constraint_name.lower()
    constraint_string = re.sub(r'(\(.*?\))', '', constraint_string)
    constraint_list = constraint_string.split(',')
    constraint_list = [c.strip() for c in constraint_list]

    # This logic checks whether the constraint should be disabled given the
    # GPU and platform.
    if "disable" in constraint_list:
        return True
    elif gpu_type == "igpu":
        return bool("disable_mobile" in constraint_list)
    elif gpu_type == "dgpu":
        if "disable_desktop" in constraint_list:
            return True
        elif ARG_DICT["RunOnGspRm"] and "disable_gsp" in constraint_list:
            return True
        elif platform.system() == "Linux":
            return bool("disable_linux" in constraint_list)
        elif platform.system() == "Windows":
            return bool("disable_windows" in constraint_list)

    return False

def get_positive_gtest_filter(fixture_dict):
    """Prints the list of fixtures/tests to include within a suite if not all
       of them are to be enabled"""

    fixture_name = fixture_dict.get("fixtureName")
    string_to_return = ""

    subtest_list = fixture_dict.get("subtests")
    if subtest_list is None:
        # enable entire fixture
        string_to_return += fixture_name + ".*:"
    else:
        # enable specific tests in fixture
        for subtest_dict in subtest_list:
            subtest_name = subtest_dict.get("subtestName")
            string_to_return += fixture_name + "." + subtest_name + ":"

    return string_to_return

def get_negative_gtest_filter(fixture_dict, gpu_name):
    """Prints the list of fixtures/tests to exclude if the entire suite is not
       to be disabled"""

    fixture_constraint = fixture_dict.get("constraint")
    fixture_name = fixture_dict.get("fixtureName")

    do_disable = False

    if fixture_constraint is not None:
        do_disable = should_disable(fixture_constraint, gpu_name)
    else:
        do_disable = False

    string_to_return = ""

    if do_disable:
        # Disable the entire fixture
        string_to_return += fixture_name + ".*:"
    else:
        # Parse the subtests
        subtest_list = fixture_dict.get("subtests")
        if subtest_list is not None:
            for subtest_dict in subtest_list:
                subtest_name = subtest_dict.get("subtestName")
                subtest_constraint = subtest_dict["constraint"]
                if should_disable(subtest_constraint, gpu_name):
                    string_to_return += fixture_name + "." + subtest_name + ":"

    return string_to_return


def get_command_string_suite(blacklist_suite_dict, whitelist_suite_dict,
                             gpu_name, bin_suffix):
    """This function builds up a string to execute the suite.  It does this by
    traversing the constraints, fixture constraints, and subtest constraints to
    build up a a string like so:

        srttest.exe --gtest_filter=-FixtureName.Subtest:FixtureName2.*

    which will execute the "srttest.exe" and exclude the subtest
    FixtureName.Subtest, and exclude the whole fixture FixtureName2 from
    exelwtion, but run all other fixtures/tests.

    If a whitelist was provided via --whitelist on the commandline, and only
    certain fixtures/tests were whitelisted in the suite, then a positive filter
    would be pre-pended to filter, like so:

        srttest.exe --gtest_filter=FixtureName3.*:FixtureName2.*:
                                     -FixtureName.Subtest:FixtureName2.*

    In the above example, only tests in FixtureName3 will run, because
    FixtureName was not listed in the positive filter and the negative filter on
    FixtureName2 takes precedence over the positive one. (Blacklist applied on
    top of whitelist)

    If there are no constraints at all, it will just be the suite binary name:

        srttest.exe --gtest_filter=-

    which will execute all tests.

    If the arguments field was set at the suite level in either/both the
    whitelist and/or blacklist, the arguments will be passed in here as well:

        srttest.exe --arg-from-whitelist --arg-from-blacklist --gtest_filter=-

    """

    suite_bin_name = blacklist_suite_dict.get("suiteName") + bin_suffix
    suite_constraint = blacklist_suite_dict.get("constraint")

    string_to_print = None

    # This function would not have been called if suite was not in whitelist,
    # so only need to check blacklist at suite level
    if not should_disable(suite_constraint, gpu_name):
        gtest_filter = " --gtest_filter="

        # form "+" filter (if needed)
        if whitelist_suite_dict is not None:
            fixture_list = whitelist_suite_dict.get("fixtures")
            if fixture_list is not None:
                for fixture_dict in fixture_list:
                    gtest_filter += get_positive_gtest_filter(fixture_dict)

        # form "-" filter (if needed)
        gtest_filter += "-"
        fixture_list = blacklist_suite_dict.get("fixtures")
        if fixture_list is not None:
            for fixture_dict in fixture_list:
                gtest_filter += get_negative_gtest_filter(fixture_dict, gpu_name)

        # get additional args from whitelist and blacklist (if any)
        args = ""

        if not ARG_DICT["IgnoreBlacklist"]:
            if blacklist_suite_dict.get("arguments"):
                args += " " + blacklist_suite_dict.get("arguments")

        if whitelist_suite_dict is not None:
            if whitelist_suite_dict.get("arguments"):
                args += " " + whitelist_suite_dict.get("arguments")

        string_to_print = suite_bin_name + args + gtest_filter

    # No else -if the entire suite is disabled don't run it at all.
    return string_to_print


def exelwte_commands(cmd_list, bin_prefix, called_from_dvs):
    """Exelwtes the commands under the current working directory.  If bin_prefix
       is provided, all test commands will be prefixed with this

       The function returns info about all the tests (see below).  The parsing
       logic for generating this info relies on the fact that the output of the
       SRT tests are like the following:

           [==========] 3 tests from 1 test case ran. (1 ms total)
           [  PASSED  ] 2 tests.
           [  FAILED  ] 1 test, listed below:
           [  FAILED  ] ClientTest.FailingTest

       Returns:
         A dictionary with overall test info.  The dictionary contents are:
           fail_count - A numeric value indicating how many subtests failed (or
             suite commands that did not complete cleanly)

           pass_count - A numeric value indicating how many subtests passed.

           failed_suite_list - A string containing the command used to launch a
             suite with failed tests.

           incomplete_suite_list - A string containing the command used to
             launch a suite which did not compelte cleanly (early exit, crash,
             etc).  Each instance of this counts as a failure.

    """
    pass_regex = re.compile(r"\[\s*PASSED\s*\] (\d+) test")
    fail_regex = re.compile(r"\[\s*FAILED\s*\] (\d+) test")

    # Counter to count the pass results + the failed results as well as overall
    # suite failures.
    out_results = {"fail_count":0, "pass_count":0, \
                   "failed_suite_list":[], "incomplete_suite_list":[]}

    # Initialise Hulk Helper settings
    if called_from_dvs:
        import hulk_helper
        hulk_helper.init_hulk_helper()
        # Replace this with names of test which need hulk license.
        tests_need_hulk = ["eccInjection", "errorContainment"]
        hulk_license_loaded = False
        hulk_helper.pre_test_system_setup()

    for cmd in cmd_list:
        cmd_with_prefix = bin_prefix + cmd

        # Load Hulk License for tests which need them.
        if called_from_dvs:
            test_name = cmd.split()[0]
            if test_name in tests_need_hulk:
                hulk_license_loaded = hulk_helper.load_hulk_license_if_available(test_name, 0)

        # Execute the tests.
        print "Exelwting command: \"" + cmd_with_prefix + "\""
        cmd_output_str = exelwte_single_test(cmd_with_prefix)
        print cmd_output_str

        # Unload Hulk License, if it was loaded.
        if called_from_dvs and hulk_license_loaded:
            hulk_helper.unload_lw_kernel_module()
            hulk_helper.load_lw_kernel_module()
            hulk_license_loaded = False

        test_fail_count = 0
        test_pass_count = 0

        for line in cmd_output_str.splitlines():
            line = line.strip()

            # Note: Results may have both PASSED and FAILED in the same output.
            match = pass_regex.match(line)
            if match:
                test_pass_count += int(match.group(1))

            match = fail_regex.match(line)
            if match:
                test_fail_count += int(match.group(1))

        if test_fail_count:
            # We had at least one failure in the suite.
            out_results["failed_suite_list"].append(cmd)
        elif "PASSED" not in cmd_output_str:
            # Check for the string PASSED to make sure the test exited cleanly.
            # We can't just check that the pass counter is 0 because, in the
            # case of tests filtered using the gtest filters, it may still say
            # "[  PASSED  ] 0 tests" if no tests were run; so instead check to
            # make sure the message was in there at all.
            out_results["incomplete_suite_list"].append(cmd)

            # add 1 to the failed test so this doesn't get reported as 100% pass.
            test_fail_count += 1

        out_results["fail_count"] += test_fail_count
        out_results["pass_count"] += test_pass_count

    return out_results

def exelwte_single_test(cmd_str):
    """Calls into the subprocess module to execute the cmd_str

       The return value is the stdout of the exelwtable process
    """

    # The return value is the stdout of the exeutabel

    cmd_list = cmd_str.split()

    proc = subprocess.Popen(cmd_list,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    outs = proc.communicate()

    proc.wait()

    return outs[0]

def get_gpu_names(args_dict):
    """Queries the helper binary for whether the GPU(s) is "igpu" or "dgpu"
    """
    helper_bin = args_dict["BinLoc"] + args_dict["HelperBinName"]
    gpu_name = exelwte_single_test(helper_bin + " --gpuid")

    # Extract the actual GPU part
    if "igpu" in gpu_name:
        gpu_name = "igpu"
    elif "dgpu" in gpu_name:
        gpu_name = "dgpu"
    else:
        exit_program_from_error("Invalid GPU string returned from " +
                                helper_bin + " \"" + gpu_name + " \"")

    return gpu_name

def get_commands(args_dict):
    """Main entry point to print out the commands to execute"""

    json_data = args_dict["JsonData"]
    whitelist = args_dict["Whitelist"]
    # Execute testListHelper to get the name of the GPU type For now, only grab
    # the first line. We may or may not need to figure out how to support
    # multi-gpu in the future.
    gpu_name = get_gpu_names(args_dict).splitlines()[0]

    ret_cmds = []

    # execute the helper binary to get the binary suffix used on this platform
    helper_bin = args_dict["BinLoc"] + args_dict["HelperBinName"]
    bin_suffix = exelwte_single_test(helper_bin + " --binsuffix").strip()

    # Traverse the JSON file and print the commands to execute each suite.
    for blacklist_suite_dict in json_data:
        # if a whitelist was provided, look for the suite in the whitelist
        whitelist_suite_dict = None
        if whitelist is not None:
            for suite_dict in whitelist:
                if blacklist_suite_dict["suiteName"] == suite_dict["suiteName"]:
                    whitelist_suite_dict = suite_dict

        # only run suite if no whitelist was provided, or the suite exists in
        # the whitelist
        if whitelist is None or whitelist_suite_dict is not None:
            suite_command_string = get_command_string_suite(blacklist_suite_dict,
                                                            whitelist_suite_dict,
                                                            gpu_name, bin_suffix)
            if suite_command_string is not None:
                ret_cmds.append(suite_command_string)

    return ret_cmds

def print_results(test_info):
    """Display the score information to allow DVS/GVS to parse them correctly."""

    result = ""

    # output the individual test suites which failed or are incomplete
    if test_info["incomplete_suite_list"]:
        incomplete_list = test_info["incomplete_suite_list"]
        result += str(len(incomplete_list)) + " SRT suite commands failed due " + \
            "to not cleanly exiting ( no FAILED or PASSED string ):\n"
        for test_incomplete in incomplete_list:
            result += "    " + test_incomplete +"\n"
        result += "\n"

    if test_info["failed_suite_list"]:
        fail_list = test_info["failed_suite_list"]
        result += str(len(fail_list)) + " SRT suite commands contained failed tests:\n"
        for test_failed in fail_list:
            result += "    " + test_failed + "\n"
        result += "\n"

    # output the overall score
    score = 0.0
    total = test_info["pass_count"] + test_info["fail_count"]
    if total != 0:
        score = (float(test_info["pass_count"]) / float(total)) * 100.0
        result += "RESULT\n"
        result += "Passes   : " + str(test_info["pass_count"]) + "\n"
        result += "Failures : " + str(test_info["fail_count"]) + "\n"
        result += "SANITY SCORE: %.2f" % score
        result += "\n"

    # write result to stout
    print result

    # write same info into summarylog.txt for DVS
    if ARG_DICT["SummaryLogName"] is not None:
        with open(ARG_DICT["SummaryLogName"], "w") as summary:
            summary.write(result)

# MAIN ENTRY POINT FOR SCRIPT:
if not validate_skip_tests():
    print("SKIP_TEST configuration is invalid")
    sys.exit(1)

# Parse the arguments and print out the commands to stdout.
ARG_DICT = parse_arguments()
CMDS = get_commands(ARG_DICT)

if ARG_DICT["DoCommandsOnly"] is True:
    print "\n".join(CMDS)
else:
    RESULTS = exelwte_commands(CMDS, ARG_DICT["BinLoc"], ARG_DICT["CalledFromDvs"])
    print_results(RESULTS)
