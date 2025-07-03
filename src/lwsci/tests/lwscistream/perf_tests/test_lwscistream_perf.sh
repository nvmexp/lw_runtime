#!/bin/bash

###############################################################################
#  test_lwscistream_perf.sh
#
#  Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
#
#  LWPU CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from LWPU CORPORATION is strictly prohibited.
#
#  Shell script to run some examples of test_lwscistream_perf
###############################################################################

# Examples
$(dmesg -C)

index=$((index+1))
echo "Test $index: Measure latency for intra-proc unicast stream"
./test_lwscistream_perf -l
wait $!

index=$((index+1))
echo "Test $index: Measure latency for intra-proc multicast stream with two consumers:"
./test_lwscistream_perf -n 2 -l
wait $!

index=$((index+1))
echo "Test $index: Measure latency for inter-proc unicast stream:"
./test_lwscistream_perf -p -l &
./test_lwscistream_perf -c 0 -l
wait $!

index=$((index+1))
echo "Test $index: Measure latency for inter-proc multicast stream with two consumers:"
./test_lwscistream_perf -p -n 2 -l &
./test_lwscistream_perf -c 0 -l &
./test_lwscistream_perf -c 1 -l
wait $!

index=$((index+1))
echo "Test $index: Collect latency data for inter-proc unicast stream for 10 frames:"
./test_lwscistream_perf -p -f 10 -l &
./test_lwscistream_perf -c 0 -l -v
wait $!

index=$((index+1))
echo "Test $index: Test whether the average latency is within 5% of 110 us and"
echo "the 99.99% latency is within 5% of 520 us for 1,000 frames:"
./test_lwscistream_perf -p -f 1000 -l &
./test_lwscistream_perf -c 0 -l -a 110 -m 520
wait $!

index=$((index+1))
echo "Test $index: Run the test with producer-present rate at 100Hz without latency"
echo "measurement (no buffer read/write) or sync obj signal/wait:"
# This command can be used to measure the cpu/memory/bandwith utilization
./test_lwscistream_perf -p -s 0 -r 100 &
./test_lwscistream_perf -c 0 -s 0
wait $!

# Test to measure packet delivery latency between two SoCs.
# Run producer on one SoC with PCIe root port
# ./test_lwscistream_perf -P 0 lwscic2c_pcie_s0_c5_1 -l
# Run consumer on another SoC with PCIe end port
# ./test_lwscistream_perf -C 0 lwscic2c_pcie_s0_c6_1 -l

#==============================================================================
# Test complete
#==============================================================================
echo "================================"