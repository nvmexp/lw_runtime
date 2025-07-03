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
# Script to generate FM Topology files for E3597 (WOLF) MODS Production Testing
#
$TOOL_PATH/topology_gen_LR.py -v -c e3597_mods_loopout.csv -t e3597_mods_loopout.topo.txt -b e3597_mods_loopout.topo.bin -n e3597_mods_loopout > e3597_mods_loopout.paths.txt
$TOOL_PATH/fabrictool -b e3597_mods_loopout.topo.bin -o e3597_mods_loopout.topo_hex.txt
$TOOL_PATH/fabrictool -b e3597_mods_loopout.topo.bin  > e3597_mods_loopout.topo_table.txt

set +x
echo "all E3597 MODS Loopout topology files generated successfully"

