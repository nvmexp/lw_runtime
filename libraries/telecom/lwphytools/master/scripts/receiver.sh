#!/bin/bash

BUSID=$1
JSON_CONFIG=$2
NUMA_NODE=$3
CORE_LIST=$4

function usage {
        echo "./receiver.sh <NIC Bus ID> <JSON config file path> <numa node> <cpu core list>"
}

#### MUST BE SUDOER
if [ "$EUID" -ne 0 ]
  then echo "Please run as root/sudoer"
  exit
fi

[[ -z $BUSID ]] && { echo "ERROR: Missing network card bus ID"; usage; exit 1; }
[[ -z $JSON_CONFIG ]] && { echo "ERROR: Missing JSON config file path"; usage; exit 1; }
[[ -z $NUMA_NODE ]] && { NUMA_NODE=0; }
[[ -z $CORE_LIST ]] && { CORE_LIST="0-10"; }

if [[ -n "$(grep -e "^lw_peer_mem" /proc/modules)" ]]; then
        echo "Module lw_peer_mem loaded"
else
        echo "ERROR: Module lw_peer_mem NOT loaded ( https://github.com/Mellanox/lw_peer_memory )"
        exit 1
fi

if [[ -n "$(grep -e "^gdrdrv" /proc/modules)" ]]; then
        echo "Module gdrdrv loaded"
else
        echo "ERROR: Module gdrdrv NOT loaded ( drivers/gdrdrv )"
        exit 1
fi

NIC_WHITELIST="$BUSID,txq_max_inline_len=0"
numactl --membind=${NUMA_NODE} ./lwPHYTools_receiver -l ${CORE_LIST} -n 8 -w ${NIC_WHITELIST} --file-prefix receiver_dpdk --base-virtaddr=0x7f0000000000 -- --json $JSON_CONFIG
#numactl --membind=${NUMA_NODE} ./lwPHYTools_receiver -l ${CORE_LIST} -n 8 -w ${NIC_WHITELIST} --file-prefix receiver_dpdk --base-virtaddr=0x7f0000000000 -- --json $JSON_CONFIG 2>&1 | tee receiver_output_$(date "+%m%d%H%M").txt
