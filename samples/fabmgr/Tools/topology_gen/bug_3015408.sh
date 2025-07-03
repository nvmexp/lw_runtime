#!/bin/bash
set -e 
set -x
TOOL_PATH=.


#top only WOLB
$TOOL_PATH/topology_gen_LR.py -c delta_ampere_top.csv --no-partitions -t  delta_ampere_top.topo.txt -b delta_ampere_top.topo.bin -n delta_ampere_top
$TOOL_PATH/fabrictool -b delta_ampere_top.topo.bin -o delta_ampere_top.topo_hex.txt
$TOOL_PATH/fabrictool -b delta_ampere_top.topo.bin > delta_ampere_top.topo_table.txt

#bottom only WOLB
$TOOL_PATH/topology_gen_LR.py -c delta_ampere_bottom.csv --no-partitions -t  delta_ampere_bottom.topo.txt -b  delta_ampere_bottom.topo.bin -n delta_ampere_bottom
$TOOL_PATH/fabrictool -b delta_ampere_bottom.topo.bin -o delta_ampere_bottom.topo_hex.txt
$TOOL_PATH/fabrictool -b delta_ampere_bottom.topo.bin > delta_ampere_bottom.topo_table.txt

#both boards
$TOOL_PATH/topology_gen_LR.py -c delta_ampere_dgx2.csv --no-partitions -t  delta_ampere_dgx2.topo.txt -b delta_ampere_dgx2.topo.bin -n delta_ampere_dgx2
$TOOL_PATH/fabrictool -b delta_ampere_dgx2.topo.bin -o  delta_ampere_dgx2.topo_hex.txt
$TOOL_PATH/fabrictool -b delta_ampere_dgx2.topo.bin > delta_ampere_dgx2.topo_table.txt

#top loopback
$TOOL_PATH/topology_gen_LR.py -c delta_ampere_top_loopback.csv --no-partitions -e -t  delta_ampere_top_loopback.topo.txt -b delta_ampere_top_loopback.topo.bin -n delta_ampere_top_loopback
$TOOL_PATH/fabrictool -b delta_ampere_top_loopback.topo.bin -o delta_ampere_top_loopback.topo_hex.txt
$TOOL_PATH/fabrictool -b delta_ampere_top_loopback.topo.bin > delta_ampere_top_loopback.topo_table.txt

#bottom loopback
$TOOL_PATH/topology_gen_LR.py -c delta_ampere_bottom_loopback.csv --no-partitions -e -t  delta_ampere_bottom_loopback.topo.txt -b  delta_ampere_bottom_loopback.topo.bin -n delta_ampere_bottom_loopback
$TOOL_PATH/fabrictool -b delta_ampere_bottom_loopback.topo.bin -o delta_ampere_bottom_loopback.topo_hex.txt
$TOOL_PATH/fabrictool -b delta_ampere_bottom_loopback.topo.bin > delta_ampere_bottom_loopback.topo_table.txt

#both loopback
$TOOL_PATH/topology_gen_LR.py -c delta_ampere_both_loopback.csv --no-partitions -e -t  delta_ampere_both_loopback.topo.txt -b  delta_ampere_both_loopback.topo.bin -n delta_ampere_both_loopback
$TOOL_PATH/fabrictool -b delta_ampere_both_loopback.topo.bin -o delta_ampere_both_loopback.topo_hex.txt
$TOOL_PATH/fabrictool -b delta_ampere_both_loopback.topo.bin > delta_ampere_both_loopback.topo_table.txt

#both no trunk
$TOOL_PATH/topology_gen_LR.py -c delta_ampere_both_no_trunk.csv --no-partitions -t  delta_ampere_both_no_trunk.topo.txt -b  delta_ampere_both_no_trunk.topo.bin -n delta_ampere_both_no_trunk
$TOOL_PATH/fabrictool -b delta_ampere_both_no_trunk.topo.bin -o delta_ampere_both_no_trunk.topo_hex.txt
$TOOL_PATH/fabrictool -b delta_ampere_both_no_trunk.topo.bin > delta_ampere_both_no_trunk.topo_table.txt
#top loopout
$TOOL_PATH/topology_gen_LR.py -c delta_ampere_both_loopout.csv --no-partitions -e -t  delta_ampere_top_loopout.topo.txt -b delta_ampere_top_loopout.topo.bin -n delta_ampere_top_loopout --path-lens 6 --gpu-ids 8,9,10,11,12,13,14,15
$TOOL_PATH/fabrictool -b delta_ampere_top_loopout.topo.bin -o delta_ampere_top_loopout.topo_hex.txt
$TOOL_PATH/fabrictool -b delta_ampere_top_loopout.topo.bin > delta_ampere_top_loopout.topo_table.txt

#bottom loopout
$TOOL_PATH/topology_gen_LR.py -c delta_ampere_both_loopout.csv --no-partitions -e -t  delta_ampere_bottom_loopout.topo.txt -b  delta_ampere_bottom_loopout.topo.bin -n delta_ampere_bottom_loopout --path-lens 6 --gpu-ids 0,1,2,3,4,5,6,7
$TOOL_PATH/fabrictool -b delta_ampere_bottom_loopout.topo.bin -o delta_ampere_bottom_loopout.topo_hex.txt
$TOOL_PATH/fabrictool -b delta_ampere_bottom_loopout.topo.bin > delta_ampere_bottom_loopout.topo_table.txt

#both loopout
$TOOL_PATH/topology_gen_LR.py -c delta_ampere_both_loopout.csv --no-partitions -e -t  delta_ampere_both_loopout.topo.txt -b  delta_ampere_both_loopout.topo.bin -n delta_ampere_both_loopout --path-lens 6
$TOOL_PATH/fabrictool -b delta_ampere_both_loopout.topo.bin -o delta_ampere_both_loopout.topo_hex.txt
$TOOL_PATH/fabrictool -b delta_ampere_both_loopout.topo.bin > delta_ampere_both_loopout.topo_table.txt

set +x
echo "all topologies generated successfully"

