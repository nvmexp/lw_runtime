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
# Script to generate FM Topology files for E4840 Back2Back Multinode Bringup
#
# With Spray and Rlan features
#$TOOL_PATH/topology_gen_LS.py --match-target-phy-id --match-gpu-port-num --path-lens 4,6 --spray --set-rlan -v -c e4840_b2b.csv -t e4840_b2b.topo.txt -b e4840_b2b.topo.bin -n e4840_b2b > e4840_b2b.paths.txt

# Without Spray and Rlan features
$TOOL_PATH/topology_gen_LS.py --match-target-phy-id --match-gpu-port-num --unique-access-port --no-match-even --path-lens 4,6 -v -c e4840_b2b.csv -t e4840_b2b.topo.txt -b e4840_b2b.topo.bin -n e4840_b2b > e4840_b2b.paths.txt
$TOOL_PATH/fabrictool -b e4840_b2b.topo.bin -o e4840_b2b.topo_hex.txt
$TOOL_PATH/fabrictool -b e4840_b2b.topo.bin  > e4840_b2b.topo_table.txt

set +x
echo "all E4840 B2B topology files generated successfully"
