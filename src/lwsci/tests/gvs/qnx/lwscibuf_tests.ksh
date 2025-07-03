#!/bin/ksh

# Copyright (c) 2019-2022 LWPU Corporation.  All rights reserved.
#
# LWPU Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU Corporation is strictly prohibited.

. common.ksh

local PROJ=`cat /dev/lwsku/project`

# List of tests
#
# Note: When a process is spawned using iolauncher, we cannot rely on $? to
# determine the exit code of the spawned program, since iolauncher does not
# wait for the program to terminate.
#
# Instead, we add --gtest_output=xml to force a file to be created and rely on
# GTest. When writing an XML file, GTest interallly calls
# XmlUnitTestResultPrinter::OnTestIterationEnd, which will open() and write
# the file when a single iteration completes. Since we're only doing 1
# iteration, this is sufficient and we don't need to implement OnTestProgramEnd.
local XML_FILE="/tmp/lwscibuf.xml"
rm -f "${XML_FILE}"
TEST_NAME[0]="iolauncher -U 1100:1100,10100,10150,10170,3000,2000 -A nonroot,allow,fork -T test_lwsciipc -T test_lwscibuf test_lwscibuf_api --gtest_output=xml:${XML_FILE}"

#Following list is for enabling Tests for corresponding boards
#Test Index  0
set -A B3550 1
set -A P3710 1
set -A P3663 1

Tests_exelwtion()
{
	tests_exelwtion_list=$@
	test_index=0
	for execute in ${tests_exelwtion_list[@]}; do
		if [ $execute -eq 1 ]; then
			run_test ${TEST_NAME[$test_index]}
		fi
		test_index=$((test_index + 1))
	done
}

if [ ${PROJ} -eq 63550 ]; then
	Tests_exelwtion ${B3550[@]}
elif [ ${PROJ} -eq 1313428047 ]; then
	Tests_exelwtion ${P3710[@]}
elif [ ${PROJ} -eq 63663 ]; then
	Tests_exelwtion ${P3663[@]}
fi

echo "Project id is ${PROJ}"

# Block until the test is complete
#
# Note: We need to block here because the VRL harness will race against the
# test application once this script exits. Since we're using iolauncher to run
# the test application, it is possible that this script exits while the test
# application is still running, at which point the VRL harness will pull the
# logs and preemptively declare a failure before the test application prints
# the pass marker.
waitfor "${XML_FILE}" 50

exit 0
