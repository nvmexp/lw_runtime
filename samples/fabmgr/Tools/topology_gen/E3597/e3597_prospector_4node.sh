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
# Script to generate FM Topology files for 4 Prospector + 6 Wolf (E3597) Bringup
# Use RLAN and spray
#
$TOOL_PATH/topology_gen_LR.py --match-target-phy-id --match-gpu-port-num --no-match-even --set-rlan --spray --switch-nodes 4,5,6,7,8,9 --path-lens 4,8 -v -c e3597_prospector_4node.csv -t e3597_prospector_4node.topo.txt -b e3597_prospector_4node.topo.bin -n e3597_prospector_4node > e3597_prospector_4node.paths.txt 

$TOOL_PATH/fabrictool -b e3597_prospector_4node.topo.bin -o e3597_prospector_4node.topo_hex.txt
$TOOL_PATH/fabrictool -b e3597_prospector_4node.topo.bin  > e3597_prospector_4node.topo_table.txt

set +x
echo "all 4 Prospector + 6 Wolf (E3597) topology files generated successfully"

