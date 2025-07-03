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
#--------------------------------------------------
# required tools : top, grep, sleep, slay, rm
#                  test_lwsciipc_read
#                  test_lwsciipc_write
#                  test_lwsciipc_perf
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
#local LOG=60
local LOG=0
#-------------------------------
# testing period (sec) - XAVIOR
#-------------------------------
# SEC=30 - 4m54s (1ms delay)
# SEC=30 - 1m15s (0 delay)
# SEC=60 - 9m58s (2ms delay)
# SEC=60 - 9m31s (1ms delay)
# SEC=60 - 1m37s (0 delay)
#-------------------------------
local SEC=60
local ITER=$((SEC * 1000 / 2))
# TX delay (ms)
local DELAY=1

print_status()
{
	LWSCIIPC_TOP_START=`top -bd -i 1 | grep -E "CPU|Mem|Average|Thread|lwsciipc|ivc"`
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

echo "[LWSCIIPC STRESS TEST]\n"
echo "CMD ITERATION: $ITER"
echo "LOOP: $LOOP_COUNT"

# clear existing test programs
slay -f test_lwsciipc*
rm -f /tmp/test_lwsciipc_*.*

print_status_all "===== BEFORE TEST"

count=0
while (($count < $LOOP_COUNT)); do
count=$((count+1))

echo ""
echo "LOOP START: $count/$LOOP_COUNT"
echo "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
echo ""

# TOTAL 75 processes

#BEGINCOMMENT
# inter-VM test (2 processes)
 $TOOL_DIR/test_lwsciipc_read -S ivcrx:0001:0030:$LOG -l $ITER -Z &
$TOOL_DIR/test_lwsciipc_write -S ivctx:0001:0030:$LOG -t 5 -w $DELAY -l $ITER -A &
#ENDCOMMENT

#BEGINCOMMENT
# inter-process test (36 processes)
 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0001:0050:$LOG -l $ITER -Z &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0001:0050:$LOG -t 5 -w $DELAY -l $ITER &
 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0051:0100:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0051:0100:$LOG -t 5 -w $DELAY -l $ITER &
 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0101:0150:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0101:0150:$LOG -t 5 -w $DELAY -l $ITER &
 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0151:0200:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0151:0200:$LOG -t 5 -w $DELAY -l $ITER &

 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0201:0250:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0201:0250:$LOG -t 5 -w $DELAY -l $ITER &
 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0251:0300:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0251:0300:$LOG -t 5 -w $DELAY -l $ITER &
 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0301:0350:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0301:0350:$LOG -t 5 -w $DELAY -l $ITER &
 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0351:0400:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0351:0400:$LOG -t 5 -w $DELAY -l $ITER &

 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0401:0450:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0401:0450:$LOG -t 5 -w $DELAY -l $ITER &
 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0451:0500:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0451:0500:$LOG -t 5 -w $DELAY -l $ITER &
 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0501:0550:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0501:0550:$LOG -t 5 -w $DELAY -l $ITER &
 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0551:0600:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0551:0600:$LOG -t 5 -w $DELAY -l $ITER &

 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0601:0650:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0601:0650:$LOG -t 5 -w $DELAY -l $ITER &
 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0651:0700:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0651:0700:$LOG -t 5 -w $DELAY -l $ITER &
 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0701:0750:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0701:0750:$LOG -t 5 -w $DELAY -l $ITER &
 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0751:0800:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0751:0800:$LOG -t 5 -w $DELAY -l $ITER &

 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0801:0850:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0801:0850:$LOG -t 5 -w $DELAY -l $ITER &
 $TOOL_DIR/test_lwsciipc_read -S ipcrx:0851:0900:$LOG -l $ITER &
$TOOL_DIR/test_lwsciipc_write -S ipctx:0851:0900:$LOG -t 5 -w $DELAY -l $ITER &
#ENDCOMMENT

#BEGINCOMMENT
# inter-thread test (36 processes)
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0001:0025:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0026:0050:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0051:0075:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0076:0100:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0101:0125:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0126:0150:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0151:0175:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0176:0200:$LOG -t 5 -w $DELAY -l $ITER &

$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0201:0225:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0226:0250:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0251:0275:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0276:0300:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0301:0325:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0326:0350:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0351:0375:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0376:0400:$LOG -t 5 -w $DELAY -l $ITER &

$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0401:0425:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0426:0450:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0451:0475:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0476:0500:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0501:0525:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0526:0550:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0551:0575:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0576:0600:$LOG -t 5 -w $DELAY -l $ITER &

$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0601:0625:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0626:0650:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0651:0675:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0676:0700:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0701:0725:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0726:0750:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0751:0775:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0776:0800:$LOG -t 5 -w $DELAY -l $ITER &

$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0801:0825:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0826:0850:$LOG -t 5 -w $DELAY -l $ITER &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0851:0875:$LOG -t 5 -w $DELAY -l $ITER -A -Z &
$TOOL_DIR/test_lwsciipc_perf -S itctx:itcrx:0876:0900:$LOG -t 5 -w $DELAY -l $ITER &
#ENDCOMMENT
sleep 180 # 2min

print_status "===== DURING TEST(1)"

sleep 60 # 1min

print_status "===== DURING TEST(2)"

sleep 420 # 7min

echo ""
echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
echo "LOOP END: $count/$LOOP_COUNT"
echo ""
done

print_status_all "===== AFTER TEST (0sec)"

# clear existing test programs
$TOOL_DIR/slay -f test_lwsciipc_read
$TOOL_DIR/slay -f test_lwsciipc_write
$TOOL_DIR/slay -f test_lwsciipc_perf
$TOOL_DIR/rm -f test_lwsciipc_*.*

$TOOL_DIR/sleep 10 # 10sec
print_status_all "===== AFTER TEST (10sec)"
