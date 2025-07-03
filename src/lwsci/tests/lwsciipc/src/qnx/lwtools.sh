#!/bin/bash
#-----------------------------------------------------------------------------
# Copyright (c) 2021, LWPU CORPORATION.  All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.
#-----------------------------------------------------------------------------
#
# This is script for host machine.
# copy tools of QNX system to target
# usage: lwtools.sh {low two bytes ip addr of DUT}
#        i.e. lwtools.sh 11.210

if [ "$1" == "" ]; then
echo "usage: $0 {target IP}"
exit 1
fi

echo "IF YOU WANT TO USE PUSHED LIBRARIES IN /tmp FOLDER, SET ELW"
echo "export LD_LIBRARY_PATH=/tmp:\${LD_LIBRARY_PATH}"
echo "SET PATH TO USE PUSHED TOOLS"
echo "export PATH=/tmp:\${PATH}"

P4QNX7TOOLS="$P4ROOT/sw/tools/embedded/qnx/qnx710-ga1/target/qnx7/aarch64le"
P4QNX7ETC="$P4ROOT/sw/tools/embedded/qnx/qnx710-ga1/target/qnx7/etc"
IFSROOT="$TEGRA_TOP/qnx/src/bsp/armv8/src/hardware/startup/boards/lwpu-t18x/vcm31t186"

IP=$1

echo "TARGET B'D IP: $IP"

# common script for integration test
sshpass -p "root" scp $TEGRA_TOP/qnx/src/tests/scripts/common.ksh root@$IP:/tmp
# integration test (FULL)
sshpass -p "root" scp $TEGRA_TOP/gpu/drv/drivers/lwsci/tests/lwsciipc/src/qnx/*.ksh root@$IP:/tmp
# integration test for GVS
sshpass -p "root" scp $TEGRA_TOP/gpu/drv/drivers/lwsci/tests/gvs/qnx/lwsciipc_tests.ksh root@$IP:/tmp

# tools (usr/bin)
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/awk root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/cut root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/date root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/find root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/grep root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/ldd root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/scp root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/sleep root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/tail root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/tee root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/time root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/top root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/touch root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/which root@$IP:/tmp

# tools (bin)
sshpass -p "root" scp $P4QNX7TOOLS/bin/cat root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/bin/chgrp root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/bin/chmod root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/bin/du root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/bin/getfacl root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/bin/ls root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/bin/pidin root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/bin/ps root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/bin/rm root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/bin/slay root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/bin/slog2info root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/bin/setfacl root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/bin/uname root@$IP:/tmp

sshpass -p "root" scp $P4QNX7TOOLS/bin/vi root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/lib/libnlwrsesw.so.1 root@$IP:/tmp

# ssh
# sshd -f /tmp/sshd_config -h /tmp/ssh_host_rsa_key -R
sshpass -p "root" scp $P4QNX7TOOLS/usr/sbin/sshd root@$IP:/tmp
sshpass -p "root" scp $P4QNX7ETC/ssh/ssh_known_hosts root@$IP:/tmp
sshpass -p "root" scp $P4QNX7ETC/ssh/ssh_config root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/libexec/ssh-keysign root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/ssh-agent root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/ssh root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/ssh-keyscan root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/ssh-add root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/bin/ssh-keygen root@$IP:/tmp
sshpass -p "root" scp $P4QNX7TOOLS/usr/lib/libcrypto1_1.so.2.1 root@$IP:/tmp

sshpass -p "root" scp $IFSROOT/sshd_config root@$IP:/tmp
sshpass -p "root" scp $IFSROOT/ssh_host_rsa_key root@$IP:/tmp
sshpass -p "root" scp $HOME/bin/runsshd.ksh root@$IP:/tmp

# nfs ; required for vectorcast
sshpass -p "root" scp $P4QNX7TOOLS/usr/sbin/fs-nfs3 root@$IP:/tmp

