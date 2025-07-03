#!/bin/ksh
################################################################################
# Copyright (c) 2021, LWPU CORPORATION. All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.
################################################################################

local LOOP_COUNT=$1
local AVAIL_MEM_START=0
local AVAIL_MEM_END=0
local TOOL_DIR
local LWSCIIPC_TOP_START
local LWSCIIPC_HOGS_RES
local LWSCIIPC_PIDIN_RES
local LWSCIIPC_MEM_INFO
local LWSCIIPC_TOP_END
local count
# test tool directory
TOOL_DIR=/tmp

#----------------------------------
# required tools:
# top, hogs, pidin, awk, grep
#----------------------------------

LwSciIpc_test()
{
	echo ""
	echo "LwSciIpc Test Iteration : $count/$LOOP_COUNT"
	echo ""

	# Native LwSciIpc Test
	$TOOL_DIR/test_lwsciipc_read -c ipc_test_0 -y 5 -l 1500 &
	$TOOL_DIR/test_lwsciipc_write -c ipc_test_1 -y 5 -l 1500 &
	$TOOL_DIR/test_lwsciipc_read -c loopback_rx -y 5 -l 1500 &
	$TOOL_DIR/test_lwsciipc_write -c loopback_tx -y 5 -l 1500 &

	# LwSciEventService Test
	$TOOL_DIR/test_lwsciipc_readm -c ipc_test_a_0:ipc_test_b_0:ipc_test_c_0 -E U -L 3 -l 4500,384 &
	$TOOL_DIR/test_lwsciipc_write -c ipc_test_a_1 -t  0 -w 5 -l 1500 -E U &
	$TOOL_DIR/test_lwsciipc_write -c ipc_test_b_1 -t 10 -w 5 -l 1500 -E U &
	$TOOL_DIR/test_lwsciipc_write -c ipc_test_c_1 -t 20 -w 5 -l 1500 -E U &
}

Print_status()
{
	# total measure time : 7sec
	LWSCIIPC_TOP_START=`top -bd -i 1 | grep -E "CPU|Mem|Average|Thread|lwsciipc|ivc"`
	LWSCIIPC_HOGS_RES=`hogs -i 1 | grep -E "MEMORY|lwsciipc|lwivc"`
	LWSCIIPC_PIDIN_RES=`pidin -F "%a %b %50N %p %Q %J %c %d %m" -M " %50: @%> %; %< %= %@"`
	LWSCIIPC_MEM_INFO=`echo "$LWSCIIPC_PIDIN_RES" | grep -E "stack|io-lwsciipc|devv-lwivc|lwsciipc_init|test_lwsciipc"`

	echo " "
	echo "$@"
	echo " "
	echo "- System Available Memory :: top -bd -i 1 "
	echo "$LWSCIIPC_TOP_START"
	echo " "
	echo "- Process usage for hogging :: hogs -i 1 "
	echo "$LWSCIIPC_HOGS_RES"
	echo " "
	echo "- Test Process Memory :: pidin -F \"%a %b %50N %p %Q %J %c %d %m\" -M \" %50: @%> %; %< %= %@\""
	echo "$LWSCIIPC_MEM_INFO"
	echo " "
}

echo "[LWSCIIPC STRESS TEST]"
echo;

# clear existing test programs
slay -f test_lwsciipc*

date
Print_status "===== BEFORE TEST"
echo "$LWSCIIPC_PIDIN_RES" > pid_mem_info.before

i=0
while((i < $LOOP_COUNT)); do

	count=$((i+1))

	LwSciIpc_test

	sleep 1

	# spend 7sec
	Print_status "===== RUNNING STATUS"
	echo "$LWSCIIPC_PIDIN_RES" > pid_mem_info

	sleep 4

	i=$((i+1))

	date
done

sleep 5

Print_status "===== AFTER TEST"
echo "$LWSCIIPC_PIDIN_RES" > pid_mem_info.after
date
