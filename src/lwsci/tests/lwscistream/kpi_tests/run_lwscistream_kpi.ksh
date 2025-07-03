#!/bin/ksh

# Copyright (c) 2021-2022 LWPU CORPORATION.  All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited
#
# LwSciStream kpi perf test

# Define KPI test and target
#  KPI requirements can be found at
#  https://lwpu.jamacloud.com/perspective.req#/containers/20052906?projectId=22182
set -A KPITest
set -A KPITargetAvg
set -A KPITargetMax
# KPILog
# 0: Single process
# 1: Output on the producer side
# 2: Output on the consumer side
set -A KPILog

KPITest[0]="ProducerCreate"
KPITargetAvg[0]=19
KPITargetMax[0]=118
KPILog[0]=0

KPITest[1]="ConsumerCreate"
KPITargetAvg[1]=22
KPITargetMax[1]=127
KPILog[1]=0

KPITest[2]="PoolCreate"
KPITargetAvg[2]=16
KPITargetMax[2]=82
KPILog[2]=0

KPITest[3]="FifoQueueCreate"
KPITargetAvg[3]=16
KPITargetMax[3]=81
KPILog[3]=0

KPITest[4]="IpcSrcCreate"
KPITargetAvg[4]=1903
KPITargetMax[4]=3331
KPILog[4]=1

KPITest[5]="IpcDstCreate"
KPITargetAvg[5]=1819
KPITargetMax[5]=3190
KPILog[5]=2

KPITest[6]="ConnectProd2IpcSrc"
KPITargetAvg[6]=19
KPITargetMax[6]=229
KPILog[6]=1

KPITest[7]="ConnectIpcDst2Cons"
KPITargetAvg[7]=19
KPITargetMax[7]=229
KPILog[7]=2

KPITest[8]="SyncAttr"
KPITargetAvg[8]=204
KPITargetMax[8]=653
KPILog[8]=2

KPITest[9]="SyncObj"
KPITargetAvg[9]=313
KPITargetMax[9]=742
KPILog[9]=2

KPITest[10]="ConsumerElements"
KPITargetAvg[10]=615
KPITargetMax[10]=1367
KPILog[10]=1

KPITest[11]="ReconciledElements"
KPITargetAvg[11]=762
KPITargetMax[11]=1729
KPILog[11]=2

KPITest[12]="PacketCreate"
KPITargetAvg[12]=136
KPITargetMax[12]=433
KPILog[12]=2

KPITest[13]="PacketStatus"
KPITargetAvg[13]=111
KPITargetMax[13]=513
KPILog[13]=1

KPITest[14]="ProducerGet"
KPITargetAvg[14]=8
KPITargetMax[14]=50
KPILog[14]=1

KPITest[15]="ConsumerAcquire"
KPITargetAvg[15]=9
KPITargetMax[15]=62
KPILog[15]=2

KPITest[16]="ProducerPresent"
KPITargetAvg[16]=116
KPITargetMax[16]=450
KPILog[16]=2

KPITest[17]="ConsumerRelease"
KPITargetAvg[17]=116
KPITargetMax[17]=405
KPILog[17]=1

# Test usage
usage() {
    echo "Usage: $0
        [-n <int>] Run each test for n times.
            Default: 100."
        1>&2; exit;
}

# Run each test for N times
N=100

# Test options
OPTIND=1
while getopts ":n::" o; do
    case "${o}" in
        n)
            N=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

echo "Each test will be run $N times."
echo "The result should be within 5% of the target."

# Test result
PASSED=0

# Run all kpi test
t=0
while [ $t -lt ${#KPITest[*]} ]; do
    test=${KPITest[$t]}
    log=${KPILog[$t]}

    echo "-------------------------------------------------------"
    echo "KPI ${test}: "

    output=${test}".txt"
    rm -rf ${output}

    # Run each test N times
    i=0
    while [ $i -lt $N ]; do
        # check return
        if [ $log -eq 0 ]; then
            ./test_lwscistream_kpi -t ${test} >> ${output}
        elif [ $log -eq 1 ]; then
            ./test_lwscistream_kpi -t ${test} -p >> ${output} &
            pid1=$!
            ./test_lwscistream_kpi -t ${test} -c &
            pid2=$!
            wait $pid1 2>/dev/null
            wait $pid2 2>/dev/null
        else
            ./test_lwscistream_kpi -t ${test} -p &
            pid1=$!
            ./test_lwscistream_kpi -t ${test} -c >> ${output} &
            pid2=$!
            wait $pid1 2>/dev/null
            wait $pid2 2>/dev/null
        fi

        # remove files in /tmp/
        rm /tmp/test_lwscistream_kpi.*

        i=$(( $i + 1))
    done

    # Post process
    min=1000000.0
    max=0.0
    sum=0.0
    numMax=0
    # The KPI shall be within 5% of the KPI target.
    avgTarget=$(echo "${KPITargetAvg[$t]} * 1.05" | bc)
    maxTarget=$(echo "${KPITargetMax[$t]} * 1.05" | bc)

    while read line; do
        #echo ${line}
        sum=$(echo "$sum+$line" | bc)
        if (( $(echo "$line > $max" |bc) )); then
            max=$line
        fi
        if (( $(echo "$line < $min" |bc) )); then
            min=$line
        fi
        if (( $(echo "$line > $maxTarget" |bc) )); then
            ((numMax++))
        fi
    done < ${output}
    avg=$(echo "scale=4;$sum/$N" | bc)

    # Test result
    echo "* 99.99% Target (us): ${KPITargetMax[$t]}"
    echo "* Average Target (us): ${KPITargetAvg[$t]}"
    echo "Min (us): $min"
    echo "Max (us): $max"
    echo "Avg (us): $avg"
    echo "Number of tests not within 5% of the 99.99% Target: $numMax"

    # 99.99% of the test should be within 5% of the 99.99% Target.
    # If the number of outliers is less than 0.01% * N, it is passed.
    if (( $(echo "$avg <= $avgTarget" |bc) && $(echo "$numMax < ($N * 0.0001 + 1)" |bc) )); then
        ((PASSED++))
        echo "Passed"
    else
        echo "Failed"
    fi

    t=$(( $t + 1))
done

echo "======================================================"
echo "PASSED $PASSED/$t"

unset KPITest
unset KPITargetAvg
unset KPITargetMax
unset KPILog