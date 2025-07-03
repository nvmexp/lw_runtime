#!/bin/bash
set -e 
set -x
TOOL_PATH=.
#top loopout
$TOOL_PATH/topology_gen_LR.py -c delta_ampere_both_loopback_e4593.csv --no-partitions -e -t  delta_ampere_top_loopback_e4593.topo.txt -b delta_ampere_top_loopback_e4593.topo.bin -n delta_ampere_top_loopback_e4593 --path-lens 6 --gpu-ids 8,9,10,11,12,13,14,15 --switch-ids 24,25,26,27,28,29
$TOOL_PATH/fabrictool -b delta_ampere_top_loopback_e4593.topo.bin -o delta_ampere_top_loopback_e4593.topo_hex.txt
$TOOL_PATH/fabrictool -b delta_ampere_top_loopback_e4593.topo.bin > delta_ampere_top_loopback_e4593.topo_table.txt

#bottom loopout
$TOOL_PATH/topology_gen_LR.py -c delta_ampere_both_loopback_e4593.csv --no-partitions -e -t  delta_ampere_bottom_loopback_e4593.topo.txt -b  delta_ampere_bottom_loopback_e4593.topo.bin -n delta_ampere_bottom_loopback_e4593 --path-lens 6 --gpu-ids 0,1,2,3,4,5,6,7 --switch-ids 8,9,10,11,12,13
$TOOL_PATH/fabrictool -b delta_ampere_bottom_loopback_e4593.topo.bin -o delta_ampere_bottom_loopback_e4593.topo_hex.txt
$TOOL_PATH/fabrictool -b delta_ampere_bottom_loopback_e4593.topo.bin > delta_ampere_bottom_loopback_e4593.topo_table.txt

#both loopout
$TOOL_PATH/topology_gen_LR.py -c delta_ampere_both_loopback_e4593.csv --no-partitions -e -t  delta_ampere_both_loopback_e4593.topo.txt -b  delta_ampere_both_loopback_e4593.topo.bin -n delta_ampere_both_loopback_e4593 --path-lens 6
$TOOL_PATH/fabrictool -b delta_ampere_both_loopback_e4593.topo.bin -o delta_ampere_both_loopback_e4593.topo_hex.txt
$TOOL_PATH/fabrictool -b delta_ampere_both_loopback_e4593.topo.bin > delta_ampere_both_loopback_e4593.topo_table.txt

set +x
echo "all topologies generated successfully"

