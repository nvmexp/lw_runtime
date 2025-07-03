#!/bin/ksh
##########################################################################################
# Copyright (c) 2019-2020, LWPU CORPORATION. All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.
##########################################################################################

echo "[LWSCIIPC PERFORMANCE TEST]"
echo;

echo "[LATENCY] - INTER-PROC"
echo "---------------------------------------------------"
echo "iolauncher -U 1000:1000,2000 -T test_lwsciipc ./test_lwsciipc_read -c ipc_test_0 -y -l 10000"
echo "iolauncher -U 1001:1001,2000 -T test_lwsciipc ./test_lwsciipc_write -c ipc_test_1 -y -l 10000"
iolauncher -U 1000:1000,2000 -T test_lwsciipc ./test_lwsciipc_read -c ipc_test_0 -y -l 10000
iolauncher -U 1001:1001,2000 -T test_lwsciipc ./test_lwsciipc_write -c ipc_test_1 -y -l 10000
sleep 25;echo;echo;

echo "[LATENCY] - INTER-VM"
echo "---------------------------------------------------"
echo "iolauncher -U 1000:1000,2000 -T test_lwsciipc ./test_lwsciipc_read -c loopback_tx -y -l 10000"
echo "iolauncher -U 1000:1000,2000 -T test_lwsciipc ./test_lwsciipc_write -c loopback_rx -y -l 10000"
iolauncher -U 1000:1000,2000 -T test_lwsciipc ./test_lwsciipc_read -c loopback_tx -y -l 10000
iolauncher -U 1000:1000,2000 -T test_lwsciipc ./test_lwsciipc_write -c loopback_rx -y -l 10000
sleep 25;echo;echo;

echo "[THROUGHPUT] - INTER-THREAD (UNI)"
echo "---------------------------------------------------"
echo "iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_perf -s itc_test_0 -r itc_test_1 -l 1000000"
iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_perf -s itc_test_0 -r itc_test_1 -l 1000000
sleep 3;echo;echo;

echo "[THROUGHPUT] - INTER-PROC (UNI)"
echo "---------------------------------------------------"
echo "iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_read -c ipc_test_0 -p -l 1000000"
echo "iolauncher -U 1001:1001,2000 -T test_lwsciipc /tmp/test_lwsciipc_write -c ipc_test_1 -p -l 1000000"
iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_read -c ipc_test_0 -p -l 1000000
iolauncher -U 1001:1001,2000 -T test_lwsciipc /tmp/test_lwsciipc_write -c ipc_test_1 -p -l 1000000
sleep 3;echo;echo;

echo "[THROUGHPUT] - INTER-VM (UNI)"
echo "---------------------------------------------------"
echo "iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_read -c loopback_rx -p -l 100000"
echo "iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_write -c loopback_tx -p -l 100000"
iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_read -c loopback_rx -p -l 100000
iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_write -c loopback_tx -p -l 100000
sleep 3;echo;echo;

echo "[THROUGHPUT] - INTER-THREAD (BI)"
echo "---------------------------------------------------"
echo "iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_perf -s itc_test_0 -r itc_test_1 -b -l 1000000"
iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_perf -s itc_test_0 -r itc_test_1 -b -l 1000000
sleep 3;echo;echo;

echo "[THROUGHPUT] - INTER-PROC (BI)"
echo "---------------------------------------------------"
echo "iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_read -c ipc_test_0 -b -p -l 1000000"
echo "iolauncher -U 1001:1001,2000 -T test_lwsciipc /tmp/test_lwsciipc_write -c ipc_test_1 -b -p -l 1000000"
iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_read -c ipc_test_0 -b -p -l 1000000
iolauncher -U 1001:1001,2000 -T test_lwsciipc /tmp/test_lwsciipc_write -c ipc_test_1 -b -p -l 1000000
sleep 3;echo;echo;

echo "[THROUGHPUT] - INTER-VM (BI)"
echo "---------------------------------------------------"
echo "iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_read -c loopback_rx -b -p -l 100000"
echo "iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_write -c loopback_tx -b -p -l 100000"
iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_read -c loopback_rx -b -p -l 100000
iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_write -c loopback_tx -b -p -l 100000
sleep 3;echo;echo;

echo "[TEST COMPLETED]"
echo; echo;
