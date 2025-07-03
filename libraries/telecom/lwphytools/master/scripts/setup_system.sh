#!/bin/bash

function usage {
        echo "./setup_system.sh <NIC Eth interface> <NIC Bus ID>"
}

IFFACE=$1
BUSID=$2

#### MUST BE SUDOER
if [ "$EUID" -ne 0 ]
  then echo "Please run as root/sudoer"
  exit
fi

[[ -z $IFFACE ]] && { echo "ERROR: IFFACE param not defined"; usage; exit 1; }
[[ -z $BUSID ]] && { echo "ERROR: BUSID param not defined"; usage; exit 1; }
if [[ -z "${RTE_SDK}" ]]; then
    echo "RTE_SDK environment variable is not set"
    usage
    exit 1
fi

#### BUILD AND INSTALL GDRDRV
pushd $PWD &>/dev/null
pushd $RTE_SDK/kernel/linux/gdrdrv/x86_64 &>/dev/null
make clean && make
./insmod.sh
popd &>/dev/null

if [[ -n "$(grep -e "^gdrdrv" /proc/modules)" ]]; then
        echo "Module gdrdrv loaded"
else
        echo "ERROR: Module gdrdrv NOT loaded ( drivers/gdrdrv )"
        exit 1
fi

if [[ -n "$(grep -e "^lw_peer_mem" /proc/modules)" ]]; then
        echo "Module lw_peer_mem present"
else
        echo "ERROR: Module lw_peer_mem NOT loaded"
        exit 1
fi

#### CONFIGURE ETHERNET INTERFACE
ifconfig $IFFACE up
# Please note: in case of dual-port CX5 you need to disable traffic control 
# on all the port (Ethernet interfaces)
ethtool -A $IFFACE rx off tx off
ifconfig $IFFACE mtu 9978

### LWCA DRIVER PERSISTENT MODE
lwpu-smi -pm 1

