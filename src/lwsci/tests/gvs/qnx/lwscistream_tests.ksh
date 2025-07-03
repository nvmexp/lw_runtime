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
local PLATFORM=`cat /dev/lwsys/tegra_platform`
local CHIP_ID=`cat /dev/lwsys/tegra_chip_id`

local FAIL=0

# List of tests
# Note:
# Add "--wait" to make iolauncher wait for the test to terminate.
TEST_NAME[0]="iolauncher -U 1100:1100,10100,10140,10150,10170,2400,3000,2000,2230,3420,10160 \
-A nonroot,allow,fork -A nonroot,allow,wait \
-A nonroot,allow,public_channel -T test_lwsciipc -T test_lwscistream \
--wait test_lwscistream_api"


#Following list is for enabling Tests for corresponding boards
#Test Index  0
set -A B3550 1
set -A T23X_SIM 1
set -A P3710 1
set -A P3663 1

Tests_exelwtion()
{
	tests_exelwtion_list=$@
	test_index=0
	for execute in ${tests_exelwtion_list[@]};do
		if [ $execute -eq 1 ];then
		run_test ${TEST_NAME[$test_index]}
		FAIL=$((($FAIL)|($?)))
	fi
	test_index=$((test_index + 1))
	done
}

if [ $PROJ -eq 63550 ];then
	Tests_exelwtion ${B3550[@]}
elif [[ $PLATFORM = "vdk" && $CHIP_ID -eq 0x23 ]]; then
	Tests_exelwtion ${T23X_SIM[@]}
elif [ ${PROJ} -eq 1313428047 ]; then
	Tests_exelwtion ${P3710[@]}
elif [ ${PROJ} -eq 63663 ]; then
	Tests_exelwtion ${P3663[@]}
fi

if [[ $FAIL -eq 1 ]]; then
	print "FAILED"
	echo "Project id is ${PROJ}"
	exit 1
else
	print "PASSED"
	exit 0
fi
