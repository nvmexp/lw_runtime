#!/bin/bash
set -e 
set -x

#bottom only
./topology_gen_LR.py -c luna_2_vm.csv --no-partitions -t  luna_2_vm.topo.txt -b  luna_2_vm.topo.bin -n luna_2_vm
../fabricTool/_out/Linux_amd64_debug/fabrictool -b luna_2_vm.topo.bin -o luna_2_vm.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b luna_2_vm.topo.bin > luna_2_vm.topo_table.txt


set +x
echo "all topologies generated successfully"

