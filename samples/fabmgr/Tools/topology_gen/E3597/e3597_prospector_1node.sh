#!/bin/bash
set -e 
set -x
TOOL_PATH= 

if test -f "../fabrictool"
then
    TOOL_PATH=../
else
    echo "fabrictool not found"
    exit 1
fi

#
# Script to generate FM Topology files for 1 Prospector + 3 Wolf (E3597) Bringup
#
$TOOL_PATH/topology_gen_WOLF_LR.py --match-target-phy-id --match-gpu-port-num --no-match-even --switch-nodes 1,2,3 --path-lens 8 -v -c e3597_prospector_1node.csv -t e3597_prospector_1node.topo.txt -b e3597_prospector_1node.topo.bin -n e3597_prospector_1node > e3597_prospector_1node.paths.txt 
$TOOL_PATH/fabrictool -b e3597_prospector_1node.topo.bin -o e3597_prospector_1node.topo_hex.txt
$TOOL_PATH/fabrictool -b e3597_prospector_1node.topo.bin  > e3597_prospector_1node.topo_table.txt

set +x
echo "all 1 Prospector + 3 Wolf (E3597) topology files generated successfully"

