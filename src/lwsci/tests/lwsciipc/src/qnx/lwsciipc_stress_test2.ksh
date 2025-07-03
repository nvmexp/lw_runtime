#!/bin/ksh
##########################################################################################
# Copyright (c) 2021, LWPU CORPORATION. All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.
##########################################################################################
#--------------------------------------------------
# LWSCIIPC STRESS TEST SCRIPT
# Parallel Exelwtion of APIs
#--------------------------------------------------
# required tools : top, grep, sleep, slay, rm
#                  test_lwsciipc_read
#                  test_lwsciipc_write
#--------------------------------------------------

alias BEGINCOMMENT="if [ ]; then"
alias ENDCOMMENT="fi"

#--------------
# 1hour  : 6
# 2hour  : 12
# 4hour  : 24
# 8hour  : 48
# 12hour : 72
#--------------
local LOOP_COUNT=$1
# test tool directory
TOOL_DIR=/tmp

# logging period (sec) - 0: no log
#local LOG=10
local LOG=60
#local LOG=0
#-------------------------------
# testing period (sec) - XAVIOR
#-------------------------------
# VITER: 27000 with 1ms delay (10min)
# PITER: 79000 with 1ms delay (10min)
#-------------------------------
local VITER=27000
local PITER=79000
# TX delay (ms)
local DELAY=1

print_status()
{
	LWSCIIPC_TOP_START=`$TOOL_DIR/top -bd -i 1 | $TOOL_DIR/grep -E "CPU|Mem|Average|Thread|lwsciipc|ivc"`
	echo " "
	echo "$@"
	echo "- System Status using top"
	echo "$LWSCIIPC_TOP_START"
	echo " "
}

print_status_all()
{
    LWSCIIPC_TOP_START=`$TOOL_DIR/top -bd -i 1 | $TOOL_DIR/grep -E "CPU|Mem|Average|Thread|lwsciipc|ivc"`
    LWSCIIPC_HOGS_RES=`hogs -i 1 | grep -E "MEMORY|lwsciipc|lwivc"`
    LWSCIIPC_PIDIN_RES=`pidin -F "%a %b %50N %p %Q %J %c %d %m" -M " %50: @%> %; %< %= %@"`
    LWSCIIPC_MEM_INFO=`echo "$LWSCIIPC_PIDIN_RES" | grep -E "stack|io-lwsciipc|devv-lwivc|lwsciipc_init|test_lwsciipc"`
    LWSCIIPC_TMP_USAGE=`$TOOL_DIR/du -k /dev/shmem`

    echo " "
    echo "$@"
    echo "- System Status using top"
    echo "$LWSCIIPC_TOP_START"
    echo " "
    echo "- Process usage for hogging :: hogs -i 1 "
    echo "$LWSCIIPC_HOGS_RES"
    echo " "
    echo "- Test Process Memory :: LwSciIpc only"
    echo "$LWSCIIPC_MEM_INFO"
    echo " "
    echo "- /tmp usage (RAM)"
    echo "$LWSCIIPC_TMP_USAGE"
    echo " "
}

echo "[LWSCIIPC STRESS TEST - PARALLEL EXEC]\n"
echo "CMD ITERATION: $ITER"
echo "LOOP: $LOOP_COUNT"

# clear existing test programs
$TOOL_DIR/slay -f test_lwsciipc_read
$TOOL_DIR/slay -f test_lwsciipc_write
$TOOL_DIR/rm -f /tmp/test_lwsciipc_*.*

print_status_all "===== BEFORE TEST"

count=0
while (($count < $LOOP_COUNT)); do
count=$((count+1))

echo ""
echo "LOOP START: $count/$LOOP_COUNT"
echo "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
echo ""

#BEGINCOMMENT
# inter-VM test (2 processes)
 $TOOL_DIR/test_lwsciipc_read -X ivcrx:0001:0030:$LOG -l $VITER -M -Z &
$TOOL_DIR/test_lwsciipc_write -X ivctx:0001:0030:$LOG -t 5 -w $DELAY -l $VITER -M -A &
#ENDCOMMENT

#BEGINCOMMENT
# inter-process test (2 processes)
 $TOOL_DIR/test_lwsciipc_read -X ipcrx:0001:0050:$LOG -l $PITER -M -Z &
$TOOL_DIR/test_lwsciipc_write -X ipctx:0001:0050:$LOG -t 5 -w $DELAY -l $PITER -M &
#ENDCOMMENT

$TOOL_DIR/sleep 300 # 5min

print_status "===== DURING TEST"

$TOOL_DIR/sleep 300 # 5min

echo ""
echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
echo "LOOP END: $count/$LOOP_COUNT"
echo ""
done

print_status_all "===== AFTER TEST (0sec)"

# clear existing test programs
$TOOL_DIR/slay -f test_lwsciipc_read
$TOOL_DIR/slay -f test_lwsciipc_write
$TOOL_DIR/rm -f test_lwsciipc_*.*

$TOOL_DIR/sleep 10 # 10sec
print_status_all "===== AFTER TEST (10sec)"
