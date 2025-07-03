#!/bin/ksh
##########################################################################################
# Copyright (c) 2020-2021, LWPU CORPORATION. All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.
##########################################################################################

# use this if /lwidia_overlay/tail doesn't work properly
#export PATH=/tmp:$PATH

# prerequisite:
#
# - add debug channels and endpoints to DT/PCT
# - install tools
#   "awk" "pidin" "ps" "rm" "cut" "slay" "slog2info" "tail" "touch"
# - install qnx/src/tests/scripts/common.ksh to DUT
#
. common.ksh

# awk : parsing output message
# tail : get last line
# cut : cut part of output message
# grep : search string
# rm : remove file or node
# sleep : waiting time
# touch : create file
check_tools_accessible "awk" "grep" "pidin" "ps" "rm" "cut" "slay" "sleep" "slog2info" "tail" "touch"

# not ready for standard build
#local PROJ=`cat /dev/lwsku/project`
local PROJ=`cat /dev/lwdt/model|cut -b 1-5`
local FAIL=0

local TOTAL=0
local PASSCNT=0
local SLEEP_SEC=0.2 # 200ms
local SLEEP_LAUNCH=0.5 # 500ms
local SLEEP_OVERRIDE="0"
local ID=0 # test id count
local LWSCIIPC_GID=2000 # lwsciipc group id

# define unit name
local LWSCIIPC_INIT=lwsciipc_init
local LWSCIIPC_RESMGR=io-lwsciipc

local LWSCIIPC_INIT_SGIDS=""
local IO_LWSCIIPC_SGIDS=""
local TOOL_ETC_SGIDS=""
local TOOL_WR_SGIDS=""
local TOOL_RD_SGIDS=""
local CFGSHM_SGIDS=""
local CHASHM_SGIDS=""

local SUB_TEST[0]
local SUB_TEST[1]
local SUB_TEST[2]
local SUB_TEST[3]
local SUB_TEST[4]

# [command-line usage]
# lwsciipc_int_tests.ksh {req|inst} {unit_dir} {skipinit|std}
#		req : requirement-based test
#		inst : instrumenting test (remove some tests)
#		unit_dir : location of test libs/apps
#		skipinit : skip relaunching init of daemon/resmgr
#
# ./lwsciipc_int_tests.ksh inst /tmp
# ./lwsciipc_int_tests.ksh req /tmp
#
if [ "$1" != "inst" ] && [ "$1" != "req" ]; then
print "Choose inst or req for the first parameter"
fi
local TEST_TYPE=$1

# unit process directory
if [ "$2" != "" ]; then
local UNIT_DIR=$2
else
# unit process directory (w/o instrumenting)
local UNIT_DIR=/proc/boot
fi

# skip relaunch exelwtables
if [ "$3" == "skipinit" ]; then
local SKIP_INIT=1
else
local SKIP_INIT=0
fi

# standard build
# TODO: authentication shall be enabled even in standard build
if [ "$3" == "std" ]; then
local SAFETY=0
else
local SAFETY=1
fi

#-----------------------------------------------------
# Support functions
#-----------------------------------------------------

# [Joshua 09/13/21]
# for file permission update of rel-33
update_tmp_libs()
{
	chgrp 45047 /tmp/liblwsciipc.so
	chgrp 45046 /tmp/liblwscievent.so
	chmod 040 /tmp/liblwsciipc.so
	chmod 040 /tmp/liblwscievent.so
}

# kill any existing test processes
#
cleanup_test()
{
	local count=5

	while [[ $count -gt 0 ]]; do
		slay -f test_lwsciipc_read
		slay -f test_lwsciipc_readm
		slay -f test_lwsciipc_write
		slay -f test_lwsciipc_perf
		slay -f test_lwsciipc_resmgr
		((count-=1))
	done
	sleep $SLEEP_SEC
}

# relaunch drivers
#
init_test()
{
	if [ $SKIP_INIT -eq 0 ]; then
		slay -f lwsciipc_init
		slay -f io-lwsciipc
		/tmp/rm -f /dev/shmem/LwSciIpcChannel*
		/tmp/rm -f /dev/shmem/LwSciIpcConfig

		iolauncher --wait -U 0:2000$LWSCIIPC_INIT_SGIDS $UNIT_DIR/lwsciipc_init -U 2000:2000 -u 2200 -v
		iolauncher --wait -U 2000:2000$IO_LWSCIIPC_SGIDS -Anonroot,allow,pathspace -Anonroot,allow,public_channel $UNIT_DIR/io-lwsciipc -v 2000 -t 5
		sleep $SLEEP_LAUNCH
	fi
}

# print commandline and run it
# return result for positive test
#
run_lwsciipc_cmd()
{
	print $@
	$@
	return $?
}

# print commandline and run it
# return ilwerted result for negative test
#
run_lwsciipc_neg_cmd()
{
	print $@
	$@
	if [ $? -eq 1 ];then
		return 0
	else
		return 1
	fi
}

# print commandline and run it
# return result based on output message for positive test
# when process is spawned by iolauncher, $? can not be used
# PASSED : return 0
# others : return 1
# delay can be overriden by SLEEP_OVERRIDE
#
run_lwsciipc_test()
{
	local delay=$SLEEP_SEC

	if [ "$SLEEP_OVERRIDE" != "0" ]; then
		delay=$SLEEP_OVERRIDE
	fi

	run_lwsciipc_cmd $@
	sleep $delay
	print;

	local cmd=`$@`
	local out=`print $cmd|tail -1|awk '{print $NF}'`
	if [ "$out" == "PASSED" ]; then
		sleep $delay
		return 0
	else
		sleep $delay
		return 1
	fi
}

# print commandline and run it
# return ilwerted result based on output message for negative test
# when process is spawned by iolauncher, $? can not be used
# FAILED : return 0
# others : return 1
# delay can be overriden by SLEEP_OVERRIDE
#
run_lwsciipc_neg_test()
{
	local delay=$SLEEP_SEC

	if [ "$SLEEP_OVERRIDE" != "0" ]; then
		delay=$SLEEP_OVERRIDE
	fi

	run_lwsciipc_cmd $@
	sleep $delay
	print;

	local cmd=`$@`
	local out=`print $cmd|tail -1|awk '{print $NF}'`
	if [ "$out" == "FAILED" ]; then
		sleep $delay
		return 0
	else
		sleep $delay
		return 1
	fi
}

# print external two commandlines and run them
# SUB_TEST[0] : 1st commandline
# SUB_TEST[1] : 2nd commandline
# return result based on output message for positive test
# 1st commandline is run in background
# when process is spawned by iolauncher, $? can not be used
# PASSED on both commands : return 0
# others : return 1
# delay can be overriden by 1st argument ($1)
#
run_lwsciipc_test2()
{
	local ret=0
	local cmd
	local out
	local delay=$SLEEP_SEC

	if [ -n "$1" ]; then
		delay=$1
	fi

	print ${SUB_TEST[0]}
	${SUB_TEST[0]}
	print ${SUB_TEST[1]}
	${SUB_TEST[1]}
	sleep $delay
	print;

	# clear existing flag file
	rm -f /tmp/run_lwsciipc_test2.fail

	{
		local cmd1=`${SUB_TEST[0]}`
		local out1=`print $cmd1|tail -1|awk '{print $NF}'`
		if [ "$out1" != "PASSED" ]; then
			touch /tmp/run_lwsciipc_test2.fail
		fi
	}&

	cmd=`${SUB_TEST[1]}`
	out=`print $cmd|tail -1|awk '{print $NF}'`
	if [ "$out" != "PASSED" ]; then
		ret=$(($ret|1))
	fi

	sleep $delay

	# check test result of other BG tasks
	if [[ -e "/tmp/run_lwsciipc_test2.fail" ]]; then
		ret=$(($ret|1))
	fi
	rm -f /tmp/run_lwsciipc_test2.fail

	return $ret
}

# print external four commandlines and run them
# SUB_TEST[0] : 1st commandline
# SUB_TEST[1] : 2nd commandline
# SUB_TEST[2] : 3rd commandline
# return result based on output message for positive test
# 1st commandline is run in background
# when process is spawned by iolauncher, $? can not be used
# PASSED on all commands : return 0
# others : return 1
# delay can be overriden by 1st and 2nd argument ($1, $2)
# $1 is delay for btw processes
# $2 is final delay after last process
#
run_lwsciipc_test3()
{
	local ret=0
	local cmd
	local out
	local delay=$SLEEP_SEC
	local delay2=$SLEEP_SEC

	if [ -n "$1" ]; then
		delay=$1
	fi
	if [ -n "$2" ]; then
		delay2=$2
	fi

	print ${SUB_TEST[0]}
	${SUB_TEST[0]}
	sleep $delay
	print ${SUB_TEST[1]}
	${SUB_TEST[1]}
	sleep $delay
	print ${SUB_TEST[2]}
	${SUB_TEST[2]}
	sleep $delay2
	print;

	# clear existing flag file
	rm -f /tmp/run_lwsciipc_test3.fail

	{
		local cmd1=`${SUB_TEST[0]}`
		local out1=`print $cmd1|tail -1|awk '{print $NF}'`
		if [ "$out1" != "PASSED" ]; then
			touch /tmp/run_lwsciipc_test3.fail
		fi
	}&
	sleep $delay

	{
		local cmd2=`${SUB_TEST[1]}`
		local out2=`print $cmd2|tail -1|awk '{print $NF}'`
		if [ "$out2" != "PASSED" ]; then
			touch /tmp/run_lwsciipc_test3.fail
		fi
	}&
	sleep $delay

	cmd=`${SUB_TEST[2]}`
	out=`print $cmd|tail -1|awk '{print $NF}'`
	if [ "$out" != "PASSED" ]; then
		ret=$(($ret|1))
	fi

	sleep $delay2

	# check test result of other BG tasks
	if [[ -e "/tmp/run_lwsciipc_test3.fail" ]]; then
		ret=$(($ret|1))
	fi
	rm -f /tmp/run_lwsciipc_test3.fail

	return $ret
}

# print external four commandlines and run them
# SUB_TEST[0] : 1st commandline
# SUB_TEST[1] : 2nd commandline
# SUB_TEST[2] : 3rd commandline
# SUB_TEST[3] : 4th commandline
# return result based on output message for positive test
# 1st commandline is run in background
# when process is spawned by iolauncher, $? can not be used
# PASSED on all commands : return 0
# others : return 1
# delay can be overriden by 1st and 2nd argument ($1, $2)
# $1 is delay for btw processes
# $2 is final delay after last process
#
run_lwsciipc_test4()
{
	local ret=0
	local cmd
	local out
	local delay=$SLEEP_SEC
	local delay2=$SLEEP_SEC

	if [ -n "$1" ]; then
		delay=$1
	fi
	if [ -n "$2" ]; then
		delay2=$2
	fi

	print ${SUB_TEST[0]}
	${SUB_TEST[0]}
	sleep $delay
	print ${SUB_TEST[1]}
	${SUB_TEST[1]}
	sleep $delay
	print ${SUB_TEST[2]}
	${SUB_TEST[2]}
	sleep $delay
	print ${SUB_TEST[3]}
	${SUB_TEST[3]}
	sleep $delay2
	print;

	# clear existing flag file
	rm -f /tmp/run_lwsciipc_test4.fail

	{
		local cmd1=`${SUB_TEST[0]}`
		local out1=`print $cmd1|tail -1|awk '{print $NF}'`
		if [ "$out1" != "PASSED" ]; then
			touch /tmp/run_lwsciipc_test4.fail
		fi
	}&
	sleep $delay

	{
		local cmd2=`${SUB_TEST[1]}`
		local out2=`print $cmd2|tail -1|awk '{print $NF}'`
		if [ "$out2" != "PASSED" ]; then
			touch /tmp/run_lwsciipc_test4.fail
		fi
	}&
	sleep $delay

	{
		local cmd3=`${SUB_TEST[2]}`
		local out3=`print $cmd3|tail -1|awk '{print $NF}'`
		if [ "$out3" != "PASSED" ]; then
			touch /tmp/run_lwsciipc_test4.fail
		fi
	}&
	sleep $delay

	cmd=`${SUB_TEST[3]}`
	out=`print $cmd|tail -1|awk '{print $NF}'`
	if [ "$out" != "PASSED" ]; then
		ret=$(($ret|1))
	fi

	sleep $delay2

	# check test result of other BG tasks
	if [[ -e "/tmp/run_lwsciipc_test4.fail" ]]; then
		ret=$(($ret|1))
	fi
	rm -f /tmp/run_lwsciipc_test4.fail

	return $ret
}


# print external four commandlines and run them
# SUB_TEST[0] : 1st commandline
# SUB_TEST[1] : 2nd commandline
# SUB_TEST[2] : 3rd commandline
# SUB_TEST[3] : 4th commandline
# SUB_TEST[4] : 5th commandline
# return result based on output message for positive test
# 1st commandline is run in background
# when process is spawned by iolauncher, $? can not be used
# PASSED on all commands : return 0
# others : return 1
# delay can be overriden by 1st and 2nd argument ($1, $2)
# $1 is delay for btw processes
# $2 is final delay after last process
#
run_lwsciipc_test5()
{
	local ret=0
	local cmd
	local out
	local delay=$SLEEP_SEC
	local delay2=$SLEEP_SEC

	if [ -n "$1" ]; then
		delay=$1
	fi
	if [ -n "$2" ]; then
		delay2=$2
	fi

	print ${SUB_TEST[0]}
	${SUB_TEST[0]}
	sleep $delay
	print ${SUB_TEST[1]}
	${SUB_TEST[1]}
	sleep $delay
	print ${SUB_TEST[2]}
	${SUB_TEST[2]}
	sleep $delay
	print ${SUB_TEST[3]}
	${SUB_TEST[3]}
	sleep $delay
	print ${SUB_TEST[4]}
	${SUB_TEST[4]}
	sleep $delay2
	print;

	# clear existing flag file
	rm -f /tmp/run_lwsciipc_test5.fail

	{
		local cmd1=`${SUB_TEST[0]}`
		local out1=`print $cmd1|tail -1|awk '{print $NF}'`
		if [ "$out1" != "PASSED" ]; then
			touch /tmp/run_lwsciipc_test5.fail
		fi
	}&
	sleep $delay

	{
		local cmd2=`${SUB_TEST[1]}`
		local out2=`print $cmd2|tail -1|awk '{print $NF}'`
		if [ "$out2" != "PASSED" ]; then
			touch /tmp/run_lwsciipc_test5.fail
		fi
	}&
	sleep $delay

	{
		local cmd3=`${SUB_TEST[2]}`
		local out3=`print $cmd3|tail -1|awk '{print $NF}'`
		if [ "$out3" != "PASSED" ]; then
			touch /tmp/run_lwsciipc_test5.fail
		fi
	}&
	sleep $delay

	{
		local cmd3=`${SUB_TEST[3]}`
		local out3=`print $cmd3|tail -1|awk '{print $NF}'`
		if [ "$out3" != "PASSED" ]; then
			touch /tmp/run_lwsciipc_test5.fail
		fi
	}&
	sleep $delay

	cmd=`${SUB_TEST[4]}`
	out=`print $cmd|tail -1|awk '{print $NF}'`
	if [ "$out" != "PASSED" ]; then
		ret=$(($ret|1))
	fi

	sleep $delay2

	# check test result of other BG tasks
	if [[ -e "/tmp/run_lwsciipc_test5.fail" ]]; then
		ret=$(($ret|1))
	fi
	rm -f /tmp/run_lwsciipc_test5.fail

	return $ret
}

#-----------------------------------------------------
# Test functions
#-----------------------------------------------------

test_configure_endpoint()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_func_configure_endpoint"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_func_configure_endpoint"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] check config blob"
	LOCAL_CMD="ls -al /dev/shmem/LwSciIpcConfig"
	run_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [POSITIVE] read config blob"
	run_lwsciipc_test test_lwsciipc_cfgblob -e itc_test_0
	ret=$((($ret)|($?)))

	run_lwsciipc_test test_lwsciipc_cfgblob -e itc_test_1
	ret=$((($ret)|($?)))

	run_lwsciipc_test test_lwsciipc_cfgblob -e ipc_test_0
	ret=$((($ret)|($?)))

	run_lwsciipc_test test_lwsciipc_cfgblob -e ipc_test_1
	ret=$((($ret)|($?)))

	run_lwsciipc_test test_lwsciipc_cfgblob -e loopback_tx
	ret=$((($ret)|($?)))

	run_lwsciipc_test test_lwsciipc_cfgblob -e loopback_rx
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	return $ret
}

test_inter_thread_comm()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_func_inter_thread_comm"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_func_inter_thread_comm QNXBSP_LWSCIIPC_ITS_REQ_selwrity_message_ordering"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] inter-thread communication"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_perf -s itc_test_0 -r itc_test_1"
	run_lwsciipc_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	return $ret
}

test_inter_process_comm()
{
	local ret=0

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_func_inter_process_comm"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_func_inter_process_comm QNXBSP_LWSCIIPC_ITS_REQ_selwrity_message_ordering"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] inter-process communication"
	SUB_TEST[0]="iolauncher -U 1000:1000,2000$TOOL_RD_SGIDS -T test_lwsciipc test_lwsciipc_read -c ipc_test_0 -b -M"
	SUB_TEST[1]="iolauncher -U 1001:1001,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_1 -b -M"
	run_lwsciipc_test2
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	return $ret
}

test_inter_vm_comm()
{
	local ret=0

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_func_inter_vm_comm"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_func_inter_vm_comm QNXBSP_LWSCIIPC_ITS_REQ_selwrity_message_ordering"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] inter-VM communication (single guest VM)"
	SUB_TEST[0]="iolauncher -U 1000:1000,2000$TOOL_RD_SGIDS -T test_lwsciipc test_lwsciipc_read -c loopback_rx -b -M"
	SUB_TEST[1]="iolauncher -U 1001:1001,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c loopback_tx -b -M"
	run_lwsciipc_test2
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	return $ret
}

test_open_endpoint_name()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_func_open_endpoint_name"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_func_endpoint_name"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [NEGATIVE] access unknown endpoint"
	LOCAL_CMD="test_lwsciipc_read -c unknown"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	print "\t[SETUP] open endpoint"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_RD_SGIDS -T test_lwsciipc test_lwsciipc_read -c ipc_test_0"
	run_lwsciipc_cmd $LOCAL_CMD

	((ID+=1))
	print "\t[$ID] [NEGATIVE] open endpoint again"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_RD_SGIDS -T test_lwsciipc test_lwsciipc_read -c ipc_test_0"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	return $ret
}

test_api_sanity()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_api_sanity"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_func_nonblock_access QNXBSP_LWSCIIPC_ITS_REQ_safety_report_api_error"
	print "\t-------------------------------------------------------------------------"

	#SLEEP_OVERRIDE="0.5"
	# colossus remote terminal delay
	SLEEP_OVERRIDE="4"

	((ID+=1))
	print "\t[$ID] [POSITIVE] inter-thread usecase"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_unit -s itc_test_0 -r itc_test_1 -w 5"
	run_lwsciipc_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	((ID+=1))
	print "\t[$ID] [POSITIVE] inter-VM usecase (single guest VM)"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_unit -s loopback_tx -r loopback_rx -w 5"
	run_lwsciipc_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	SLEEP_OVERRIDE=0

	# test tear down
	cleanup_test

	return $ret
}

test_recover_connection()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_func_recover_connection"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_func_recover_connection"
	print "\t-------------------------------------------------------------------------"

	SLEEP_OVERRIDE="0.5"

	((ID+=1))
	print "\t[$ID] [POSTIVE] inter-thread usecase"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_perf -s itc_test_0 -r itc_test_1 -c"
	run_lwsciipc_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	SLEEP_OVERRIDE="0.5"

	((ID+=1))
	print "\t[$ID] [POSITIVE] inter-process usecase"
	SUB_TEST[0]="iolauncher -U 1000:1000,2000$TOOL_RD_SGIDS -T test_lwsciipc test_lwsciipc_read -c ipc_test_0 -m -l 300"
	SUB_TEST[1]="iolauncher -U 1001:1001,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_1 -l 100"
	SUB_TEST[2]="iolauncher -U 1002:1002,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_1 -l 100"
	SUB_TEST[3]="iolauncher -U 1003:1003,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_1 -l 100"
	run_lwsciipc_test4 .5 1
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	((ID+=1))
	print "\t[$ID] [POSITIVE] inter-VM usecase (single guest VM)"
	SUB_TEST[0]="iolauncher -U 1000:1000,2000$TOOL_RD_SGIDS -T test_lwsciipc test_lwsciipc_read -c loopback_rx -m -l 300"
	SUB_TEST[1]="iolauncher -U 1001:1001,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c loopback_tx -l 100"
	SUB_TEST[2]="iolauncher -U 1002:1002,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c loopback_tx -l 100"
	SUB_TEST[3]="iolauncher -U 1003:1003,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c loopback_tx -l 100"
	run_lwsciipc_test4 .5 1
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	return $ret
}

test_lwscievent_api()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_lwscievent_api"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_lwscievent_event_handling"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] inter-thread usecase"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_perf -s itc_test_0 -r itc_test_1 -E U"
	run_lwsciipc_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

    # [NOTE] it doesn't work with code enumeration
	if [ $TEST_TYPE == "req" ]; then
		((ID+=1))
		print "\t[$ID] [POSITIVE] inter-process usecase"
		SUB_TEST[0]="iolauncher -U 1000:1000,2000$TOOL_RD_SGIDS -T test_lwsciipc test_lwsciipc_read -c ipc_test_0 -b -M -E U"
		SUB_TEST[1]="iolauncher -U 1001:1001,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_1 -b -M -E U"
		run_lwsciipc_test2
		ret=$((($ret)|($?)))

		# test tear down
		cleanup_test
	fi

	((ID+=1))
	print "\t[$ID] [POSITIVE] inter-VM usecase (single guest VM)"
	SUB_TEST[0]="iolauncher -U 1000:1000,2000$TOOL_RD_SGIDS -T test_lwsciipc test_lwsciipc_read -c loopback_rx -b -M -E U"
	SUB_TEST[1]="iolauncher -U 1001:1001,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c loopback_tx -b -M -E U"
	run_lwsciipc_test2
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	return $ret
}

test_get_and_auth_token()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_get_and_auth_token"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_get_and_auth_token"
	print "\t-------------------------------------------------------------------------"

	print "\t[SETUP] launch test resource manager"
	LOCAL_CMD="test_lwsciipc_resmgr -u 2200:2000 &"
	run_lwsciipc_cmd $LOCAL_CMD

	SLEEP_OVERRIDE="0.5"
	((ID+=1))
	print "\t[$ID] [POSITIVE] inter-process usecase"
	LOCAL_CMD="test_lwsciipc_lwmap -e ipc_test_0 -u 1000:2000,21000 -a 10002:10002"
	run_lwsciipc_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	SLEEP_OVERRIDE="0.5"
	((ID+=1))
	print "\t[$ID] [POSITIVE] inter-VM usecase (single guest VM)"
	LOCAL_CMD="test_lwsciipc_lwmap -e ivc_test -u 1000:2000,23000 -a 200ff:200ff"
	run_lwsciipc_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	if [ $SAFETY -eq 1 ];then
		SLEEP_OVERRIDE="0.5"
		((ID+=1))
		print "\t[$ID] [NEGATIVE] inter-process usecase (send incorrect auth token)"
		LOCAL_CMD="test_lwsciipc_lwmap -e ipc_test_0 -u 1000:2000,21000 -a 10002:10002 -n"
		run_lwsciipc_neg_test $LOCAL_CMD
		ret=$((($ret)|($?)))

		SLEEP_OVERRIDE="0.5"
		((ID+=1))
		print "\t[$ID] [NEGATIVE] inter-VM usecase (send incorrect auth token)"
		LOCAL_CMD="test_lwsciipc_lwmap -e ivc_test -u 1000:2000,23000 -a 200ff:200ff -n"
		run_lwsciipc_neg_test $LOCAL_CMD
		ret=$((($ret)|($?)))
	fi

	# test tear down
	cleanup_test

	return $ret
}

test_selwrity_endpoint()
{
	local ret=0
	local LOCAL_CMD
	local err

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_selwrity_endpoint"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_selwrity_endpoint"
	print "\t-------------------------------------------------------------------------"

# LwSciIpcOpenEndpoint: fail (34) - IlwalidState // Busy 514
	((ID+=1))
	print "\t[$ID] [NEGATIVE] inter-thread usecase: w/o tag"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS test_lwsciipc_perf -s itc_test_0 -r itc_test_1"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

# ChannelCreate_r: fail (33) - NotPermitted
	((ID+=1))
	print "\t[$ID] [NEGATIVE] inter-process usecase: w/o tag"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS test_lwsciipc_write -c ipc_test_0"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

# ChannelCreate_r: fail (33) - NotPermitted
	((ID+=1))
	print "\t[$ID] [NEGATIVE] inter-VM usecase (single guest VM): w/o tag"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS test_lwsciipc_write -c loopback_tx"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

# "ldd:FATAL: Could not load library liblwsciipc.so" can not be checked -> changed
# LwSciIpcInit: fail (33) - NotPermitted
	((ID+=1))
	print "\t[$ID] [NEGATIVE] inter-thread usecase: incorrect SGID"
	LOCAL_CMD="iolauncher -U 1000:1000,2001$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_perf -s itc_test_0 -r itc_test_1"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

# "ldd:FATAL: Could not load library liblwsciipc.so" can not be checked -> changed
# LwSciIpcInit: fail (33) - NotPermitted
	((ID+=1))
	print "\t[$ID] [NEGATIVE] inter-process usecase: incorrect SGID"
	LOCAL_CMD="iolauncher -U 1000:1000,2001$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_0"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

# "ldd:FATAL: Could not load library liblwsciipc.so" can not be checked -> changed
# LwSciIpcInit: fail (33) - NotPermitted
	((ID+=1))
	print "\t[$ID] [NEGATIVE] inter-VM usecase (single guest VM): incorrect SGID"
	LOCAL_CMD="iolauncher -U 1000:1000,2001$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c loopback_tx"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

# LwSciIpcOpenEndpoint: fail (34) - IlwalidState // 33 NotPermitted
	((ID+=1))
	print "\t[$ID] [NEGATIVE] inter-process usecase: incorrect security policy"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T lwbpmpivc_safety test_lwsciipc_write -c ipc_test_0"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

# LwSciIpcOpenEndpoint: fail (34) - IlwalidState // 33 NotPermitted
	((ID+=1))
	print "\t[$ID] [NEGATIVE] inter-VM usecase (single guest VM): incorrect security policy"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T lwbpmpivc_safety test_lwsciipc_write -c loopback_tx"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	return $ret
}

test_selwrity_unique_session_id()
{
	local ret=0
	local LOCAL_CMD
	local idx1
	local idx2
	local idx3
	local idx4
	local idx5

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_selwrity_unique_session_id"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_selwrity_unique_session_id"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] unique session id"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_readm -o -c itc_test_0:itc_test_1:ipc_test_0:ipc_test_1:ivc_test -v"
	run_lwsciipc_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	idx1=`$LOCAL_CMD|grep Handle|awk -F": " '{print $3}'|awk 'NR==1'|cut -b 3-5`
	idx2=`$LOCAL_CMD|grep Handle|awk -F": " '{print $3}'|awk 'NR==2'|cut -b 3-5`
	idx3=`$LOCAL_CMD|grep Handle|awk -F": " '{print $3}'|awk 'NR==3'|cut -b 3-5`
	idx4=`$LOCAL_CMD|grep Handle|awk -F": " '{print $3}'|awk 'NR==4'|cut -b 3-5`
	idx5=`$LOCAL_CMD|grep Handle|awk -F": " '{print $3}'|awk 'NR==5'|cut -b 3-5`

	# check if session id is unique
	if [[ $idx1 -eq 1 && $idx2 -eq 2 && $idx3 -eq 3 &&
	$idx4 -eq 4 && $idx5 -eq 5 ]]; then
		ret=$((($ret)|0))
	else
		ret=$((($ret)|1))
	fi

	# test tear down
	cleanup_test

	return $ret
}

test_selwrity_resouce_access()
{
	local ret=0
	local LOCAL_CMD
	local addr
	local size
	local irq

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_selwrity_resource_access"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_selwrity_resource_access"
	print "\t-------------------------------------------------------------------------"

	print "\t[SETUP] read lwbpmpivc_0 endpoint info"
	LOCAL_CMD="test_lwsciipc_cfgblob -e lwbpmpivc_0"
	run_lwsciipc_cmd $LOCAL_CMD
	addr=`$LOCAL_CMD|grep phy|awk -F",|:" '{print $2}'|cut -b 3-12`
	size=`$LOCAL_CMD|grep phy|awk -F",|:" '{print $4}'|cut -b 3-12`
	irq=`$LOCAL_CMD|grep phy|awk -F",|:" '{print $6}'`

	((ID+=1))
	print "\t[$ID] [POSITIVE] access unauthorized resources"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_unit -n $addr,$size,$irq"
	run_lwsciipc_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	return $ret
}

test_selwrity_spoofing_resmgr()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_selwrity_spoofing_resmgr"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_selwrity_spoofing_resmgr"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [NEGATIVE] lwsciipc_init"
	pidin -f abUVApJB -p lwsciipc_init
	LOCAL_CMD="iolauncher --wait -U 0:2000$LWSCIIPC_INIT_SGIDS $UNIT_DIR/lwsciipc_init -U 2000:2000 -u 2200"
	run_lwsciipc_neg_cmd $LOCAL_CMD
	ret=$((($ret)|($?)))

#	slog2info -b lwsciipc_init | tail
#	print ""

	((ID+=1))
	print "\t[$ID] [NEGATIVE] io-lwsciipc"
	pidin -f abUVApJB -p io-lwsciipc
	LOCAL_CMD="iolauncher --wait -U 2000:2000$IO_LWSCIIPC_SGIDS -Anonroot,allow,pathspace -Anonroot,allow,public_channel --wait $UNIT_DIR/io-lwsciipc -v 1000 -t 5"
	run_lwsciipc_neg_cmd $LOCAL_CMD
	ret=$((($ret)|($?)))

#	slog2info -b io-lwsciipc | tail
#	print ""

	return $ret
}

#--------------------------------------------------------
# [Joshua 09/13/21]
# added SGID permission
# 40029 : /proc/boot dir access
# 40002 : /proc/boot/libc.so.4
# 40006 : /proc/boot/libslog2.so.1
# 45068 : /proc/boot/liblw_qnx_overrides.so
# 45031 : /proc/boot/liblwivc.so
# 45037 : /proc/boot/liblwos_s3_safety.so
# 45047 : /proc/boot/liblwsciipc.so
#--------------------------------------------------------
test_selwrity_access_info()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_selwrity_access_info"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_selwrity_access_info"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [NEGATIVE] prohibit access from non-root process"
	LOCAL_CMD="iolauncher -U 2000:2000$TOOL_ETC_SGIDS test_lwsciipc_cfgblob -e itc_test_0"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	LOCAL_CMD="iolauncher -U 2000:2000$TOOL_ETC_SGIDS test_lwsciipc_cfgblob -e ipc_test_0"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	LOCAL_CMD="iolauncher -U 2000:2000$TOOL_ETC_SGIDS test_lwsciipc_cfgblob -e loopback_tx"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	return $ret
}

test_selwrity_drop_root_privileges()
{
	local ret=0
	local LOCAL_CMD
	local uid
	local gid

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_selwrity_drop_root_privileges"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_selwrity_drop_root_privileges"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] drop privilege"
	LOCAL_CMD="pidin -f UVn -p $LWSCIIPC_INIT"
	run_lwsciipc_cmd $LOCAL_CMD
	uid=`$LOCAL_CMD|grep $LWSCIIPC_INIT|awk '{print $1}'`
	gid=`$LOCAL_CMD|grep $LWSCIIPC_INIT|awk '{print $2}'`
	if [[ $uid -eq $LWSCIIPC_GID && $gid -eq $LWSCIIPC_GID ]]; then
		ret=$((($ret)|0))
	else
		ret=$((($ret)|1))
	fi

	LOCAL_CMD="pidin -f UVn -p $LWSCIIPC_RESMGR"
	run_lwsciipc_cmd $LOCAL_CMD
	uid=`$LOCAL_CMD|grep $LWSCIIPC_RESMGR|awk '{print $1}'`
	gid=`$LOCAL_CMD|grep $LWSCIIPC_RESMGR|awk '{print $2}'`
	if [[ $uid -eq $LWSCIIPC_GID && $gid -eq $LWSCIIPC_GID ]]; then
		ret=$((($ret)|0))
	else
		ret=$((($ret)|1))
	fi

	return $ret
}

test_safety_create_shared_resources()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_safety_create_shared_resources"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_safety_create_shared_resources"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] configuration blob"
	LOCAL_CMD="ls /dev/shmem/LwSciIpcConfig"
	run_lwsciipc_cmd $LOCAL_CMD
	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [POSITIVE] intra-VM channel data"
	LOCAL_CMD="ls /dev/shmem/LwSciIpcChannel*"
	run_lwsciipc_cmd $LOCAL_CMD
	ret=$((($ret)|($?)))

	return $ret
}

#--------------------------------------------------------
# [Joshua 09/13/21]
# added SGID permission
# 40029 : /proc/boot dir access
# 40002 : /proc/boot/libc.so.4
# 45068 : /proc/boot/liblw_qnx_overrides.so
#--------------------------------------------------------
test_safety_preserve_shared_memory_nodes()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_safety_preserve_shared_memory_nodes"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_safety_preserve_shared_memory_nodes"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [NEGATIVE] preserve shared memory"
	LOCAL_CMD="iolauncher -U 2000:2000$CFGSHM_SGIDS --wait rm -f /dev/shmem/LwSciIpcConfig"
	run_lwsciipc_neg_cmd $LOCAL_CMD
	ret=$((($ret)|($?)))

	LOCAL_CMD="iolauncher -U 2000:2000$CHASHM_SGIDS --wait rm -f /dev/shmem/LwSciIpcChannel*"
	run_lwsciipc_neg_cmd $LOCAL_CMD
	ret=$((($ret)|($?)))

	return $ret
}

# no impact on other tests ?
# verified
test_safety_detect_missing_configblob()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_safety_detect_missing_configblob"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_safety_detect_missing_configblob"
	print "\t-------------------------------------------------------------------------"

	print "\t[SETUP] remove config blob"
	slay -f lwsciipc_init
	sleep $SLEEP_SEC
	rm -f /dev/shmem/LwSciIpcConfig
	print;

	((ID+=1))
	print "\t[$ID] [NEGATIVE] detect missing resources#1"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_0"
	run_lwsciipc_neg_cmd $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	((ID+=1))
	print "\t[$ID] [NEGATIVE] detect missing resources#2"
	LOCAL_CMD="test_lwsciipc_write -c ipc_test_0"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	# test tear down
	rm -f /dev/shmem/LwSciIpcConfig
	rm -f /dev/shmem/LwSciIpcChannel*
	$UNIT_DIR/lwsciipc_init -U 2000:2000 -u 2200
	sleep $SLEEP_SEC

	return $ret
}

# no impact on other tests ?
# verified
test_safety_detect_missing_intra_vm_resources()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_safety_detect_missing_intra_vm_resources"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_safety_detect_missing_intra_vm_resources"
	print "\t-------------------------------------------------------------------------"

	print "\t[SETUP] remove intra-VM channel memory"
	slay -f lwsciipc_init
	sleep $SLEEP_SEC
	rm -f /dev/shmem/LwSciIpcChannel*
	print;

	((ID+=1))
	print "\t[$ID] [NEGATIVE] detect missing resources"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_0"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	# test tear down
	rm -f /dev/shmem/LwSciIpcConfig
	rm -f /dev/shmem/LwSciIpcChannel*
	$UNIT_DIR/lwsciipc_init -U 2000:2000 -u 2200

	return $ret
}

# impact on VSCD operation due to endpoint mutex
# verified
test_safety_detect_missing_lwsciipc_resmgr_node()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_safety_detect_missing_lwsciipc_resmgr_node"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_safety_detect_missing_lwsciipc_resmgr_node"
	print "\t-------------------------------------------------------------------------"

	print "\t[SETUP] kill io-lwsciipc"
	slay -f io-lwsciipc
	sleep $SLEEP_SEC
	ls -al /dev/lwsciipc*
	print;

	((ID+=1))
	print "\t[$ID] [NEGATIVE] detect missing resources"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_0"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	# test tear down
	iolauncher -U 2000:2000$IO_LWSCIIPC_SGIDS -Anonroot,allow,pathspace -Anonroot,allow,public_channel $UNIT_DIR/io-lwsciipc -v 1000 -t 10
	sleep $SLEEP_LAUNCH

	return $ret
}

# verified
test_safety_report_internal_process_init_error()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_safety_report_internal_process_init_error"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_safety_report_internal_process_init_error"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [NEGATIVE] lwsciipc_init"
	LOCAL_CMD="iolauncher --wait $UNIT_DIR/lwsciipc_init -u 0"
	run_lwsciipc_neg_cmd $LOCAL_CMD
	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [NEGATIVE] io-lwsciipc"
	LOCAL_CMD="iolauncher --wait $UNIT_DIR/io-lwsciipc"
	run_lwsciipc_neg_cmd $LOCAL_CMD
	ret=$((($ret)|($?)))

	return $ret
}

# verified
test_conlwrrent_comm()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_func_conlwrrent_comm"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_func_conlwrrent_comm"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] multiple endpoints in different backend type"
	SUB_TEST[0]="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_perf -s itc_test_0 -r itc_test_1 -l 500000"
	SUB_TEST[1]="iolauncher -U 1001:1001,2000$TOOL_RD_SGIDS -T test_lwsciipc test_lwsciipc_read  -c ipc_test_0 -p -l 500000"
	SUB_TEST[2]="iolauncher -U 1002:1002,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_1 -p -l 500000"
	SUB_TEST[3]="iolauncher -U 1003:1003,2000$TOOL_RD_SGIDS -T test_lwsciipc test_lwsciipc_read  -c loopback_rx -p -l 200000"
	SUB_TEST[4]="iolauncher -U 1004:1004,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c loopback_tx -p -l 200000"
	# 3sec for colossus remote delay
	run_lwsciipc_test5 0 3
	ret=$((($ret)|($?)))
	print;

	# test tear down
	cleanup_test

	((ID+=1))
	print "\t[$ID] [POSITIVE] multiple endpoints in same backend type"
	SUB_TEST[0]="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_readm -c ipc_test_a_0:ipc_test_b_0:ipc_test_c_0 -l 30000"
	SUB_TEST[1]="iolauncher -U 1001:1001,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_a_1 -l 10000"
	SUB_TEST[2]="iolauncher -U 1002:1002,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_b_1 -l 10000"
	SUB_TEST[3]="iolauncher -U 1003:1003,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_c_1 -l 10000"
	# 1sec for colossus remote delay
	run_lwsciipc_test4 0 1
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	return $ret
}

# doesn't work in statement+branch test
test_lwscievent_local_event()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_lwscievent_local_event"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_lwscievent_local_event"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] lwscievent single local event"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_readm -E U -L 1 -l 0,50"
	run_lwsciipc_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test


	return $ret
}

test_lwscievent_wait_multiple_native_events()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_lwscievent_wait_multiple_native_events"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_lwscievent_wait_multiple_events"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] lwscievent wait for multiple native events"
	SUB_TEST[0]="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_readm -c ipc_test_a_0:ipc_test_b_0:ipc_test_c_0 -l 384 -E U"
	SUB_TEST[1]="iolauncher -U 1001:1001,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_a_1 -t 0 -w 2 -l 128 -E U"
	SUB_TEST[2]="iolauncher -U 1002:1002,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_b_1 -t 10 -w 2 -l 128 -E U"
	SUB_TEST[3]="iolauncher -U 1003:1003,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_c_1 -t 20 -w 2 -l 128 -E U"
	run_lwsciipc_test4 0 2
	ret=$((($ret)|($?)))
	print;

	# test tear down
	cleanup_test

	return $ret
}

test_lwscievent_wait_multiple_local_events()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_lwscievent_wait_multiple_local_events"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_lwscievent_wait_multiple_events"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] lwscievent multiple local events"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_readm -E U -L 5 -l 0,10"
	run_lwsciipc_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	return $ret
}

test_lwscievent_wait_multiple_mixed_events()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_lwscievent_wait_multiple_mixed_events"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_lwscievent_wait_multiple_events"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] lwscievent wait for multiple native and local events"
	SUB_TEST[0]="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_readm -c ipc_test_a_0:ipc_test_b_0:ipc_test_c_0 -E U -L 3 -l 384,384"
	SUB_TEST[1]="iolauncher -U 1001:1001,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_a_1 -t 0 -w 2 -l 128 -E U"
	SUB_TEST[2]="iolauncher -U 1002:1002,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_b_1 -t 10 -w 2 -l 128 -E U"
	SUB_TEST[3]="iolauncher -U 1003:1003,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_write -c ipc_test_c_1 -t 20 -w 2 -l 128 -E U"
	run_lwsciipc_test4 0 2
	ret=$((($ret)|($?)))
	print;

	# test tear down
	cleanup_test

	return $ret
}

#--------------------------------------------------------
# [Joshua 09/13/21]
# msg receiving timeout doesn't work with VC instrumenting
# timeout doesn't work with code enumeration
#--------------------------------------------------------
test_lwscievent_wait_multiple_events_timeout()
{
	local ret=0
	local LOCAL_CMD

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_lwscievent_wait_multiple_events_timeout"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_lwscievent_wait_multiple_events"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [NEGATIVE] lwscievent wait for multiple events timeout"
	LOCAL_CMD="iolauncher -U 1000:1000,2000$TOOL_WR_SGIDS -T test_lwsciipc test_lwsciipc_readm -c ipc_test_a_0:ipc_test_b_0:ipc_test_c_0 -E 1 -L 3"
	run_lwsciipc_neg_test $LOCAL_CMD
	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	return $ret
}

test_stack_overflow()
{
	local ret=0
	local LOCAL_CMD
	local pid
	local tcnt

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_REQ_func_stack_overflow"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_REQ_func_stack_overflow"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] handles SIGSEGV signal in LwSciIpc unit processes"

	LOCAL_CMD="pidin -p lwsciipc_init"
	run_lwsciipc_cmd $LOCAL_CMD
	LOCAL_CMD="pidin -p io-lwsciipc"
	run_lwsciipc_cmd $LOCAL_CMD
	echo "\n"

	pid=`pidin -P lwsciipc_init | tail -1 | awk '{print $1}'`
	tcnt=`pidin -P lwsciipc_init | tail -1 | awk '{print $2}'`
	echo "lwsciipc_init pid:$pid thread cnt:$tcnt"
	LOCAL_CMD="slay -s SIGSEGV -T $tcnt $pid"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	pid=`pidin -P io-lwsciipc | tail -1 | awk '{print $1}'`
	tcnt=`pidin -P io-lwsciipc | tail -1 | awk '{print $2}'`
	echo "io-lwsciipc pid:$pid thread cnt:$tcnt"
	LOCAL_CMD="slay -s SIGSEGV -T $tcnt $pid"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	echo "\n"
	LOCAL_CMD="pidin -p lwsciipc_init"
	run_lwsciipc_cmd $LOCAL_CMD
	LOCAL_CMD="pidin -p io-lwsciipc"
	run_lwsciipc_cmd $LOCAL_CMD

	# test tear down
	cleanup_test

	# relaunch daemon and resource managers
	init_test

	return $ret
}

test_cmdline_lwsciipc_init_positive()
{
	local ret=0
	local LOCAL_CMD
	local pid
	local tcnt

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_CMD_lwsciipc_init_positive"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_CMD_lwsciipc_init"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] -U option"

	slay -f lwsciipc_init
	rm -f /dev/shmem/LwSciIpcConfig
	rm -f /dev/shmem/LwSciIpcCha*

	LOCAL_CMD="$UNIT_DIR/lwsciipc_init -U 2000:2000"
	run_lwsciipc_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [POSITIVE] -u option"

	slay -f lwsciipc_init
	rm -f /dev/shmem/LwSciIpcConfig
	rm -f /dev/shmem/LwSciIpcCha*

	LOCAL_CMD="$UNIT_DIR/lwsciipc_init -U 2000:2000 -u 1"
	run_lwsciipc_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [POSITIVE] -g option"

	slay -f lwsciipc_init
	rm -f /dev/shmem/LwSciIpcConfig
	rm -f /dev/shmem/LwSciIpcCha*

	LOCAL_CMD="$UNIT_DIR/lwsciipc_init -U 2000:2000 -g 1"
	run_lwsciipc_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [BOUNDARY] -u option with 64 uids"

	slay -f lwsciipc_init
	rm -f /dev/shmem/LwSciIpcConfig
	rm -f /dev/shmem/LwSciIpcCha*

	LOCAL_CMD="$UNIT_DIR/lwsciipc_init -U 2000:2000 -u 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64"
	run_lwsciipc_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [BOUNDARY] -g option with 64 gids"

	slay -f lwsciipc_init
	rm -f /dev/shmem/LwSciIpcConfig
	rm -f /dev/shmem/LwSciIpcCha*

	LOCAL_CMD="$UNIT_DIR/lwsciipc_init -U 2000:2000 -g 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64"
	run_lwsciipc_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))


	# test tear down
	cleanup_test

	# relaunch daemon and resource managers
	init_test

	return $ret
}

test_cmdline_lwsciipc_init_negative()
{
	local ret=0
	local LOCAL_CMD
	local pid
	local tcnt

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_CMD_lwsciipc_init_negative"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_CMD_lwsciipc_init"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [NEGATIVE] with -h option"

	LOCAL_CMD="$UNIT_DIR/lwsciipc_init -h"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [NEGATIVE] without -U option"

	LOCAL_CMD="$UNIT_DIR/lwsciipc_init"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [NEGATIVE] -U 0:0 option"

	LOCAL_CMD="$UNIT_DIR/lwsciipc_init -U 0:0"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [NEGATIVE] -u option with duplicated uids"

	LOCAL_CMD="$UNIT_DIR/lwsciipc_init -U 2000:2000 -u 1,1"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [NEGATIVE] -g option with duplicated gids"

	LOCAL_CMD="$UNIT_DIR/lwsciipc_init -U 2000:2000 -g 1,1"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [BOUNDARY] -u option with 65 uids (> max 64 uids)"

	LOCAL_CMD="$UNIT_DIR/lwsciipc_init -U 2000:2000 -u 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [BOUNDARY] -g option with 65 uids (> max 64 uids)"

	LOCAL_CMD="$UNIT_DIR/lwsciipc_init -U 2000:2000 -g 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))


	# test tear down
	cleanup_test

	# relaunch daemon and resource managers
	init_test

	return $ret
}


test_cmdline_io_lwsciipc_positive()
{
	local ret=0
	local LOCAL_CMD
	local pid
	local tcnt

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_CMD_io_lwsciipc_positive"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_CMD_io_lwsciipc"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [POSITIVE] -v and -t option"

	slay -f io-lwsciipc

	LOCAL_CMD="$UNIT_DIR/io-lwsciipc -v 1000 -t 10"
	run_lwsciipc_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [BOUNDARY] -t 1 option (0 < thread <= 50)"

	slay -f io-lwsciipc

	LOCAL_CMD="$UNIT_DIR/io-lwsciipc -v 1000 -t 1"
	run_lwsciipc_cmd $LOCAL_CMD

	((ID+=1))
	print "\t[$ID] [BOUNDARY] -t 50 option (0 < thread <= 50)"

	slay -f io-lwsciipc

	LOCAL_CMD="$UNIT_DIR/io-lwsciipc -v 1000 -t 50"
	run_lwsciipc_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [BOUNDARY] -v 1 option (0 < VUID <= 2000)"

	slay -f io-lwsciipc

	LOCAL_CMD="$UNIT_DIR/io-lwsciipc -v 1 -t 5"
	run_lwsciipc_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [BOUNDARY] -v 2000 option (0 < VUID <= 2000)"

	slay -f io-lwsciipc

	LOCAL_CMD="$UNIT_DIR/io-lwsciipc -v 2000 -t 5"
	run_lwsciipc_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	# relaunch daemon and resource managers
	init_test

	return $ret
}

test_cmdline_io_lwsciipc_negative()
{
	local ret=0
	local LOCAL_CMD
	local pid
	local tcnt

	print "\t-------------------------------------------------------------------------"
	print "\tTest Case ID: QNXBSP_LWSCIIPC_IT_CMD_io_lwsciipc_negative"
	print "\tUpstream ID: QNXBSP_LWSCIIPC_ITS_CMD_io_lwsciipc"
	print "\t-------------------------------------------------------------------------"

	((ID+=1))
	print "\t[$ID] [NEGATIVE] without option"

	LOCAL_CMD="io-lwsciipc"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [NEGATIVE] -v option only"

	LOCAL_CMD="$UNIT_DIR/io-lwsciipc -v 1000"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [NEGATIVE] -t option only"

	LOCAL_CMD="$UNIT_DIR/io-lwsciipc -t 10"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [BOUNDARY] -t 0 option (0 < thread <= 50)"

	LOCAL_CMD="$UNIT_DIR/io-lwsciipc -v 1000 -t 0"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [BOUNDARY] -t 51 option (0 < thread <= 50)"

	LOCAL_CMD="$UNIT_DIR/io-lwsciipc -v 1000 -t 51"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [BOUNDARY] -v 0 option (0 < VUID <= 2000)"

	LOCAL_CMD="$UNIT_DIR/io-lwsciipc -v 0 -t 5"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	((ID+=1))
	print "\t[$ID] [BOUNDARY] -v 2001 option (0 < VUID <= 2000)"

	LOCAL_CMD="$UNIT_DIR/io-lwsciipc -v 2001 -t 5"
	run_lwsciipc_neg_cmd $LOCAL_CMD

	ret=$((($ret)|($?)))

	# test tear down
	cleanup_test

	# relaunch daemon and resource managers
	init_test

	return $ret
}

# List of tests
TEST_NAME[0]=test_configure_endpoint
TEST_NAME[1]=test_inter_thread_comm
TEST_NAME[2]=test_inter_process_comm
TEST_NAME[3]=test_inter_vm_comm
TEST_NAME[4]=test_open_endpoint_name
TEST_NAME[5]=test_api_sanity
TEST_NAME[6]=test_recover_connection
TEST_NAME[7]=test_lwscievent_api
TEST_NAME[8]=test_get_and_auth_token
TEST_NAME[9]=test_selwrity_endpoint
TEST_NAME[10]=test_selwrity_unique_session_id
TEST_NAME[11]=test_selwrity_resouce_access
TEST_NAME[12]=test_selwrity_spoofing_resmgr
TEST_NAME[13]=test_selwrity_access_info
TEST_NAME[14]=test_selwrity_drop_root_privileges
TEST_NAME[15]=test_safety_create_shared_resources
TEST_NAME[16]=test_safety_preserve_shared_memory_nodes
TEST_NAME[17]=test_safety_detect_missing_configblob
TEST_NAME[18]=test_safety_detect_missing_intra_vm_resources
TEST_NAME[19]=test_safety_detect_missing_lwsciipc_resmgr_node
TEST_NAME[20]=test_safety_report_internal_process_init_error
TEST_NAME[21]=test_conlwrrent_comm
TEST_NAME[22]=test_lwscievent_local_event
TEST_NAME[23]=test_lwscievent_wait_multiple_native_events
TEST_NAME[24]=test_lwscievent_wait_multiple_local_events
TEST_NAME[25]=test_lwscievent_wait_multiple_mixed_events
TEST_NAME[26]=test_lwscievent_wait_multiple_events_timeout
TEST_NAME[27]=test_stack_overflow
TEST_NAME[28]=test_cmdline_lwsciipc_init_positive
TEST_NAME[29]=test_cmdline_lwsciipc_init_negative
TEST_NAME[30]=test_cmdline_io_lwsciipc_positive
TEST_NAME[31]=test_cmdline_io_lwsciipc_negative

#Following list is for enabling Tests for corresponding boards
if [ $TEST_TYPE == "req" ]; then
	print "Requirement-based test"
	# Integration test
	#Test Index  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
	set -A B3550 1 1 1 1 1 1 1 1 1 1 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
	set -A B3663 1 1 1 1 1 1 1 1 1 1 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
	#set -A B3550 0 0 0 0 0 0 0 0 0 0 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
else
	print "Instrumenting test"
	# Function & Function Call coverage test - remove destructive tests
	#Test Index  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
	set -A B3550 1 1 1 1 1 1 1 1 1 1 1  1  1  1  0  1  1  0  0  0  0  1  1  0  1  1  1  0  0  1  0  1
	set -A B3663 1 1 1 1 1 1 1 1 1 1 1  1  1  1  0  1  1  0  0  0  0  1  1  0  1  1  1  0  0  1  0  1
	#set -A B3550 0 0 0 0 0 0 0 0 0 0 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
fi

Tests_exelwtion()
{
	tests_exelwtion_list=$@
	test_index=0

	# only for local integration test
	update_tmp_libs

	cleanup_test
	init_test
	for execute in ${tests_exelwtion_list[@]};do
		if [ $execute -eq 1 ];then
			print "[T#$test_index]${TEST_NAME[$test_index]}+++"
			${TEST_NAME[$test_index]}
			if [ $? -eq 1 ];then
				print "[T#$test_index]${TEST_NAME[$test_index]}---FAILED\n"
				FAIL=$((FAIL + 1))
			else
				PASSCNT=$((PASSCNT + 1))
				print "[T#$test_index]${TEST_NAME[$test_index]}---PASSED\n"
			fi
			TOTAL=$((TOTAL + 1))
		fi
		test_index=$((test_index + 1))
	done
	cleanup_test
}

print "[LWSCIIPC INTEGRATION TEST]"
print;
if [ $PROJ -eq 63550 ];then
	Tests_exelwtion ${B3550[@]}
fi
if [ "$PROJ" == "p3663" ];then
	Tests_exelwtion ${B3663[@]}
fi
print "Statistics [$PASSCNT/$TOTAL]"

if [[ $FAIL -gt 0 ]];then
	print "FAILED"
	print "Project id is ${PROJ}"
	exit 1
else
	print "PASSED"
	exit 0
fi

