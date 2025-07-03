#!/bin/bash

# Copyright (c) 2021 LWPU CORPORATION.  All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited
#
# LwSciBuf kpi perf test

# Define KPI test and target
#  KPI requirements can be found at
#  https://lwpu.jamacloud.com/perspective.req#/containers/20672016?projectId=22182
declare -A KPITargetAvg
KPITargetAvg["ModuleOpen"]=2471
KPITargetAvg["AttrListCreate"]=19
KPITargetAvg["AttrListSetAttrs_Camera"]=92
KPITargetAvg["AttrListSetAttrs_ISP"]=92
KPITargetAvg["AttrListSetAttrs_Display"]=92
KPITargetAvg["AttrListSetInternalAttrs_Camera"]=9
KPITargetAvg["AttrListSetInternalAttrs_ISP"]=9
KPITargetAvg["AttrListSetInternalAttrs_Display"]=9
KPITargetAvg["AttrListReconcile_Camera"]=821
KPITargetAvg["AttrListReconcile_Isp_Display"]=821
KPITargetAvg["ObjAlloc"]=1655
KPITargetAvg["ObjRef"]=2

declare -A KPITargetMax
KPITargetMax["ModuleOpen"]=4302
KPITargetMax["AttrListCreate"]=108
KPITargetMax["AttrListSetAttrs_Camera"]=222
KPITargetMax["AttrListSetAttrs_ISP"]=222
KPITargetMax["AttrListSetAttrs_Display"]=222
KPITargetMax["AttrListSetInternalAttrs_Camera"]=67
KPITargetMax["AttrListSetInternalAttrs_ISP"]=67
KPITargetMax["AttrListSetInternalAttrs_Display"]=67
KPITargetMax["AttrListReconcile_Camera"]=1488
KPITargetMax["AttrListReconcile_Isp_Display"]=1488
KPITargetMax["ObjAlloc"]=10700
KPITargetMax["ObjRef"]=31


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
        ./test_lwscibuf_kpi -t ${t} >> ${output}
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