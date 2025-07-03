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
$TOOL_PATH/topology_gen_LS.py -v -c vulcan.csv -t vulcan.topo.txt -b vulcan.topo.bin -n vulcan > vulcan.topo.paths.txt
$TOOL_PATH/fabrictool -b vulcan.topo.bin -o  vulcan.topo_hex.txt
$TOOL_PATH/fabrictool -b vulcan.topo.bin > vulcan.topo_table.txt

set +x
echo "all Vulcan topology files generated successfully"
