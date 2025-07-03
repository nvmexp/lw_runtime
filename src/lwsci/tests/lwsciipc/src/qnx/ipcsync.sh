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
# copy library and exelwtables to target
# set up build environment before using it
# usage: ipcsync.sh {build flavor} {low two bytes ip addr of DUT}
#        i.e. ipcsync.sh 11.210

if [ "$1" == "" ] || [ "$2" == "" ]; then
echo "usage: $0 {debug-safety|release-safety|debug-none|release-none} {target IP}"
exit 1
fi

FLAVOR=$1

echo "PLZ MAKE SURE YOUR CURRENT BUILD: $FLAVOR"
echo "IF YOU WANT TO USE PUSHED LIBRARIES IN /tmp FOLDER, SET ELW"
echo "export LD_LIBRARY_PATH=/tmp:\${LD_LIBRARY_PATH}"

IP=$2

echo "TARGET B'D IP: $IP"

#sshpass -p "root" scp $TEGRA_TOP/out/embedded-qnx-t186ref-$FLAVOR/lwpu/qnx/src/tools/iolauncher-qnx_64/iolauncher root@$IP:/tmp
sshpass -p "root" scp $TEGRA_TOP/out/embedded-qnx-t186ref-$FLAVOR/lwpu/gpu/drv/drivers/lwsci/tests/lwsciipc-qnx_64/test_lwsciipc_* root@$IP:/tmp

#scp ./out/embedded-qnx-t186ref-$FLAVOR/lwpu/lwscic2c/libs/c2c-qnx_64/liblwscic2c.so root@$IP:/tmp
sshpass -p "root" scp $TEGRA_TOP/out/embedded-qnx-t186ref-$FLAVOR/lwpu/gpu/drv/drivers/lwsci/lwscievent-qnx_64/liblwscievent.so root@$IP:/tmp
sshpass -p "root" scp $TEGRA_TOP/out/embedded-qnx-t186ref-$FLAVOR/lwpu/gpu/drv/drivers/lwsci/lwsciipc-qnx_64/liblwsciipc.so root@$IP:/tmp
sshpass -p "root" scp $TEGRA_TOP/out/embedded-qnx-t186ref-$FLAVOR/lwpu/qnx/src/tools/lwsciipc_init-qnx_64/lwsciipc_init root@$IP:/tmp
sshpass -p "root" scp $TEGRA_TOP/out/embedded-qnx-t186ref-$FLAVOR/lwpu/qnx/src/resmgrs/lwsciipc-qnx_64/io-lwsciipc root@$IP:/tmp
sshpass -p "root" scp $TEGRA_TOP/out/embedded-qnx-t186ref-$FLAVOR/lwpu/qnx/src/resmgrs/lwivc-qnx_64/devv-lwivc root@$IP:/tmp

#scp ./out/embedded-qnx-t186ref-$FLAVOR/lwpu/ivclib/ivc-qnx_64/liblwivc.so root@$IP:/tmp
#scp ./out/embedded-qnx-t186ref-$FLAVOR/lwpu/gpu/drv/drivers/lwsci/tests/lwscisync/api-qnx_64/QNX_aarch64_debug/drivers/lwsci/tests/lwscisync/api/test_lwscisync_api root@$IP:/tmp
#scp ./out/embedded-qnx-t186ref-$FLAVOR/lwpu/gpu/drv/drivers/lwsci/lwscibuf-qnx_64/QNX_aarch64_debug/drivers/lwsci/lwscibuf/liblwscibuf.so root@$IP:/tmp
