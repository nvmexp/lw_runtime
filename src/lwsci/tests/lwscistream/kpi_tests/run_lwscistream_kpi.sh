#!/bin/bash

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
declare -A KPITargetAvg
KPITargetAvg["ProducerCreate"]=19
KPITargetAvg["ConsumerCreate"]=22
KPITargetAvg["PoolCreate"]=16
KPITargetAvg["FifoQueueCreate"]=16
KPITargetAvg["IpcSrcCreate"]=1903
KPITargetAvg["IpcDstCreate"]=1819
KPITargetAvg["ConnectProd2IpcSrc"]=19
KPITargetAvg["ConnectIpcDst2Cons"]=19
KPITargetAvg["SyncAttr"]=204
KPITargetAvg["SyncObj"]=313
KPITargetAvg["ConsumerElements"]=615
KPITargetAvg["ReconciledElements"]=762
KPITargetAvg["PacketCreate"]=136
KPITargetAvg["PacketStatus"]=111
KPITargetAvg["ProducerGet"]=8
KPITargetAvg["ConsumerAcquire"]=9
KPITargetAvg["ProducerPresent"]=116
KPITargetAvg["ConsumerRelease"]=116

declare -A KPITargetMax
KPITargetMax["ProducerCreate"]=118
KPITargetMax["ConsumerCreate"]=127
KPITargetMax["PoolCreate"]=82
KPITargetMax["FifoQueueCreate"]=81
KPITargetMax["IpcSrcCreate"]=3331
KPITargetMax["IpcDstCreate"]=3190
KPITargetMax["ConnectProd2IpcSrc"]=229
KPITargetMax["ConnectIpcDst2Cons"]=229
KPITargetMax["SyncAttr"]=653
KPITargetMax["SyncObj"]=742
KPITargetMax["ConsumerElements"]=1367
KPITargetMax["ReconciledElements"]=1729
KPITargetMax["PacketCreate"]=433
KPITargetMax["PacketStatus"]=513
KPITargetMax["ProducerGet"]=50
KPITargetMax["ConsumerAcquire"]=62
KPITargetMax["ProducerPresent"]=450
KPITargetMax["ConsumerRelease"]=405

# KPILog
# 0: Single process
# 1: Output on the producer side
# 2: Output on the consumer side
declare -A KPILog
KPILog["ProducerCreate"]=0
KPILog["ConsumerCreate"]=0
KPILog["PoolCreate"]=0
KPILog["FifoQueueCreate"]=0
KPILog["IpcSrcCreate"]=1
KPILog["IpcDstCreate"]=2
KPILog["ConnectProd2IpcSrc"]=1
KPILog["ConnectIpcDst2Cons"]=2
KPILog["SyncAttr"]=2
KPILog["SyncObj"]=2
KPILog["ConsumerElements"]=1
KPILog["ReconciledElements"]=2
KPILog["PacketCreate"]=2
KPILog["PacketStatus"]=1
KPILog["ProducerGet"]=1
KPILog["ConsumerAcquire"]=2
KPILog["ProducerPresent"]=2
KPILog["ConsumerRelease"]=1

# Test usage
usage() {
    echo "Usage: $0
        [-n <int>] Run each test for n times.
            Default: 100."
        1>&2; exit 1;
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
CNT=0
PASSED=0

# Run all kpi test
for t in "${!KPITargetAvg[@]}"; do
    echo "-------------------------------------------------------"
    echo "KPI ${t}: "

    output=${t}".txt"
    rm -rf ${output}

    log=${KPILog[$t]}

    # Run each test N times
    for ((i=1;i<=N;i++)); do
        # check return
        if [[ $log -eq 0 ]]; then
            ./test_lwscistream_kpi -t ${t} >> ${output} &
            pid1=$!
            wait $pid1 2>/dev/null
        elif [[ $log -eq 1 ]]; then
            ./test_lwscistream_kpi -t ${t} -p >> ${output} &
            pid1=$!
            ./test_lwscistream_kpi -t ${t} -c &
            pid2=$!
            wait $pid1 2>/dev/null
            wait $pid2 2>/dev/null
        else
            ./test_lwscistream_kpi -t ${t} -p &
            pid1=$!
            ./test_lwscistream_kpi -t ${t} -c >> ${output} &
            pid2=$!
            wait $pid1 2>/dev/null
            wait $pid2 2>/dev/null
        fi
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
        sum=$(echo "$sum+$line" | bc -l)
        if (( $(echo "$line > $max" |bc -l) )); then
            max=$line
        fi
        if (( $(echo "$line < $min" |bc -l) )); then
            min=$line
        fi
        if (( $(echo "$line > $maxTarget" |bc -l) )); then
            ((numMax++))
        fi
    done < ${output}
    avg=$(echo "scale=4;$sum/$N" | bc -l)

    # Test result
    echo "* 99.99% Target (us): ${KPITargetMax[$t]}"
    echo "* Average Target (us): ${KPITargetAvg[$t]}"
    echo "Min (us): $min"
    echo "Max (us): $max"
    echo "Avg (us): $avg"
    echo "Number of tests not within 5% of the 99.99% Target: $numMax"

    # 99.99% of the test should be within 5% of the 99.99% Target.
    # If the number of outliers is less than 0.01% * N, it is passed.
    if (( $(echo "$avg <= $avgTarget && $numMax < ($N * 0.0001 + 1)" |bc -l) )); then
        ((PASSED++))
        echo "Passed"
    else
        echo "Failed"
    fi
    ((CNT++))
done

echo "======================================================"
echo "PASSED $PASSED/$CNT"

unset KPILog
unset KPITargetMax
unset KPITargetAvg