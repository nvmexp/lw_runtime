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
# Script to generate FM Topology files for E3597 (WOLF) Back2Back Bringup
#
$TOOL_PATH/topology_gen_LR.py -v -c e3597_lwos_b2b.csv -t e3597_lwos_b2b.topo.txt -b e3597_lwos_b2b.topo.bin -n e3597_lwos_b2b > e3597_lwos_b2b.paths.txt
$TOOL_PATH/fabrictool -b e3597_lwos_b2b.topo.bin -o e3597_lwos_b2b.topo_hex.txt
$TOOL_PATH/fabrictool -b e3597_lwos_b2b.topo.bin  > e3597_lwos_b2b.topo_table.txt

set +x
echo "all E3597 (Wolf) LWOS B2B topology files generated successfully"

