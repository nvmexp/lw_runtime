#!/bin/bash

# Copyright (c) 2021 LWPU CORPORATION.  All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited
#
# LwSciSync kpi perf test

# Define KPI test and target
#  KPI reqruiements can be found at
#  https://lwpu.jamacloud.com/perspective.req#/containers/20672025?projectId=22182
declare -A KPITargetAvg
KPITargetAvg["ModuleOpen"]=267
KPITargetAvg["AttrListCreate"]=16
KPITargetAvg["AttrListSetAttrs_Signaler"]=4
KPITargetAvg["AttrListSetAttrs_Waiter"]=4
KPITargetAvg["AttrListSetInternalAttrs_Signaler"]=2
KPITargetAvg["AttrListSetInternalAttrs_Waiter"]=2
KPITargetAvg["AttrListReconcile"]=43
KPITargetAvg["ObjAlloc"]=105
KPITargetAvg["ObjDup"]=7
KPITargetAvg["ObjGetPrimitiveType"]=8
KPITargetAvg["ObjGetNumPrimitives"]=8
KPITargetAvg["ObjRef"]=4
KPITargetAvg["FenceExtract"]=2
KPITargetAvg["FenceUpdate"]=3
KPITargetAvg["FenceDup"]=3

declare -A KPITargetMax
KPITargetMax["ModuleOpen"]=792
KPITargetMax["AttrListCreate"]=97
KPITargetMax["AttrListSetAttrs_Signaler"]=47
KPITargetMax["AttrListSetAttrs_Waiter"]=47
KPITargetMax["AttrListSetInternalAttrs_Signaler"]=46
KPITargetMax["AttrListSetInternalAttrs_Waiter"]=46
KPITargetMax["AttrListReconcile"]=117
KPITargetMax["ObjAlloc"]=472
KPITargetMax["ObjDup"]=58
KPITargetMax["ObjGetPrimitiveType"]=57
KPITargetMax["ObjGetNumPrimitives"]=50
KPITargetMax["ObjRef"]=42
KPITargetMax["FenceExtract"]=36
KPITargetMax["FenceUpdate"]=44
KPITargetMax["FenceDup"]=48


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

    # Run each test N times
    for ((i=1;i<=N;i++)); do
        # check return
        ./test_lwscisync_kpi -t ${t} >> ${output}
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

unset KPITargetMax
unset KPITargetAvg