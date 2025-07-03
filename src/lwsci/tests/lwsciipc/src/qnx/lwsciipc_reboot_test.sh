#!/bin/bash
# $TEGRA_TOP shall be defined by build elwironmental setup
# TEST ITERATION
COUNT=1000
# set board IP address
BD_IP=10.19.11.105
# set virtual console of GOS 0
ttyGOS0=/dev/pts/9
# set virtual console of safety VM
ttySAFETY=/dev/pts/12
logRaw=./raw_output.log
FLAVOR="debug-safety" # debug safety build
stty -F $ttyGOS0 115200
stty -F $ttySAFETY 115200
i=1
while ((i <= $COUNT))
do
		timestamp=`date`
        echo "$i/$COUNT......$timestamp"
        echo -e "\n\n:::BEGIN::: COUNT $i/$COUNT...... :::\n\n" | ts >> $logRaw
        # reset system via update VM console
        echo "echo 1 > /dev/lwsyseventclient/trigger_sys_reboot" > $ttySAFETY
        sleep 100s

		echo -e "\n\n" > $ttyGOS0
		echo "echo REBOOT COMPLETED: $i/$COUNT" > $ttyGOS0
		echo -e "\n\n" > $ttyGOS0

		sshpass -p "root" scp $TEGRA_TOP/out/embedded-qnx-t186ref-$FLAVOR/lwpu/gpu/drv/drivers/lwsci/tests/lwsciipc-qnx_64/test_lwsciipc_* root@$BD_IP:/tmp
		sleep 1s

		# test inter-thread backend
		sshpass -p "root" ssh root@$BD_IP /proc/boot/iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_perf -s itc_test_0 -r itc_test_1 | ts >> $logRaw
		sleep 1s

		# test inter-process backend
		sshpass -p "root" ssh root@$BD_IP /proc/boot/iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_read -c ipc_test_0 -b -M | ts >> $logRaw &
		sshpass -p "root" ssh root@$BD_IP /proc/boot/iolauncher -U 1001:1001,2000 -T test_lwsciipc /tmp/test_lwsciipc_write -c ipc_test_1 -b -M | ts >> $logRaw
		sleep 1s

		# test inter-VM backend
		sshpass -p "root" ssh root@$BD_IP /proc/boot/iolauncher -U 1000:1000,2000 -T test_lwsciipc /tmp/test_lwsciipc_read -c loopback_rx -b -M | ts >> $logRaw &
		sshpass -p "root" ssh root@$BD_IP /proc/boot/iolauncher -U 1001:1001,2000 -T test_lwsciipc /tmp/test_lwsciipc_write -c loopback_tx -b -M | ts >> $logRaw
		sleep 2s

        echo -e "\n\n:::END::: COUNT $i/$COUNT...... :::\n\n" | ts >> $logRaw
        ((i++))
done

