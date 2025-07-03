#!/bin/bash

BUSID=$1
JSON_CONFIG=$2
NUMA_NODE=$3
CORE_LIST=$4

function usage {
        echo "./generator.sh <NIC Bus ID> <JSON config file path> <numa node> <cpu core list>"
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

NIC_WHITELIST="$BUSID,txq_max_inline_len=0"
numactl --membind=${NUMA_NODE} ./lwPHYTools_generator --base-virtaddr 0x7f0000000000 -l ${CORE_LIST} -n 8 -w ${NIC_WHITELIST} --file-prefix generator_dpdk -- --json $JSON_CONFIG
