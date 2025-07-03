#!/bin/bash
set -e 
set -x
TOOL_PATH=.


#bottom loopout
$TOOL_PATH/topology_gen_LR.py -c delta_ampere_optical_loopout.csv --no-partitions -e -t  delta_ampere_optical_loopout.topo.txt -b  delta_ampere_optical_loopout.topo.bin -n delta_ampere_optical_loopout --path-lens 6 --gpu-ids 0,1,2,3,4,5,6,7 -v > delta_ampere_optical_loopout.paths.txt
$TOOL_PATH/fabrictool -b delta_ampere_optical_loopout.topo.bin -o delta_ampere_optical_loopout.topo_hex.txt
$TOOL_PATH/fabrictool -b delta_ampere_optical_loopout.topo.bin > delta_ampere_optical_loopout.topo_table.txt

set +x
echo "all topologies generated successfully"

