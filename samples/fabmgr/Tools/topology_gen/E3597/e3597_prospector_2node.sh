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
# Script to generate FM Topology files for 2 Prospector + 3 Wolf (E3597) Bringup 
# No RLAN or spray
#
#$TOOL_PATH/topology_gen_LR.py --match-target-phy-id --match-gpu-port-num --no-match-even --switch-nodes 2,3,4 --path-lens 4,8 -v -c e3597_prospector_2node.csv -t e3597_prospector_2node.topo.txt -b e3597_prospector_2node.topo.bin -n e3597_prospector_2node > e3597_prospector_2node.paths.txt

#
# Script to generate FM Topology files for 2 Prospector + 3 Wolf (E3597) Bringup 
# Use RLAN and spray
#
$TOOL_PATH/topology_gen_LR.py --match-target-phy-id --match-gpu-port-num --no-match-even --set-rlan --spray --switch-nodes 2,3,4 --path-lens 4,8 -v -c e3597_prospector_2node.csv -t e3597_prospector_2node.topo.txt -b e3597_prospector_2node.topo.bin -n e3597_prospector_2node > e3597_prospector_2node.paths.txt

$TOOL_PATH/fabrictool -b e3597_prospector_2node.topo.bin -o e3597_prospector_2node.topo_hex.txt
$TOOL_PATH/fabrictool -b e3597_prospector_2node.topo.bin  > e3597_prospector_2node.topo_table.txt

set +x
echo "all 2 Prospector + 3 Wolf (E3597) topology files generated successfully"

