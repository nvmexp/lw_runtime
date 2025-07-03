#!/bin/ksh

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
set -A KPITest
set -A KPITargetAvg
set -A KPITargetMax
KPITest[0]="ModuleOpen"
KPITargetAvg[0]=2471
KPITargetMax[0]=4302
KPITest[1]="AttrListCreate"
KPITargetAvg[1]=19
KPITargetMax[1]=108
KPITest[2]="AttrListSetAttrs_Camera"
KPITargetAvg[2]=92
KPITargetMax[2]=222
KPITest[3]="AttrListSetAttrs_ISP"
KPITargetAvg[3]=92
KPITargetMax[3]=222
KPITest[4]="AttrListSetAttrs_Display"
KPITargetAvg[4]=92
KPITargetMax[4]=222
KPITest[5]="AttrListSetInternalAttrs_Camera"
KPITargetAvg[5]=9
KPITargetMax[5]=67
KPITest[6]="AttrListSetInternalAttrs_ISP"
KPITargetAvg[6]=9
KPITargetMax[6]=67
KPITest[7]="AttrListSetInternalAttrs_Display"
KPITargetAvg[7]=9
KPITargetMax[7]=67
KPITest[8]="AttrListReconcile_Camera"
KPITargetAvg[8]=821
KPITargetMax[8]=1488
KPITest[9]="AttrListReconcile_Isp_Display"
KPITargetAvg[9]=821
KPITargetMax[9]=1488
KPITest[10]="ObjAlloc"
KPITargetAvg[10]=1655
KPITargetMax[10]=10700
KPITest[11]="ObjRef"
KPITargetAvg[11]=2
KPITargetMax[11]=31

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

    echo "-------------------------------------------------------"
    echo "KPI ${test}: "

    output=${test}".txt"
    rm -rf ${output}

    # Run each test N times
    i=0
    while [ $i -lt $N ]; do
        # check return
        ./test_lwscibuf_kpi -t ${test} >> ${output}

        # remove files in /tmp/
        rm /tmp/test_lwscibuf_kpi.*

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
    if (( $(echo "$avg <= $avgTarget && $numMax < ($N * 0.0001 + 1)" |bc) )); then
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