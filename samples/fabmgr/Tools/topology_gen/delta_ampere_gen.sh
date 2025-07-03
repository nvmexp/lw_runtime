#!/bin/bash
set -e 
set -x

FABRICTOOL=

if test -f "$FABRICTOOL"
then
    FABRICTOOL=$FABRICTOOL
elif test -f "./fabrictool"
then
    FABRICTOOL=./fabrictool
elif test -f "~/fabrictool"
then
    FABRICTOOL=~/fabrictool
else
    echo "fabrictool not found"
    exit 1
fi


#top only
./topology_gen_LR.py -c delta_ampere_top.csv --no-partitions -t  delta_ampere_top.topo.txt -b delta_ampere_top.topo.bin -n delta_ampere_top
$FABRICTOOL -b delta_ampere_top.topo.bin -o delta_ampere_top.topo_hex.txt
$FABRICTOOL -b delta_ampere_top.topo.bin > delta_ampere_top.topo_table.txt
#bottom only
./topology_gen_LR.py -c delta_ampere_bottom.csv --no-partitions -t  delta_ampere_bottom.topo.txt -b  delta_ampere_bottom.topo.bin -n delta_ampere_bottom
$FABRICTOOL -b delta_ampere_bottom.topo.bin -o delta_ampere_bottom.topo_hex.txt
$FABRICTOOL -b delta_ampere_bottom.topo.bin > delta_ampere_bottom.topo_table.txt

#both boards
./topology_gen_LR.py -c delta_ampere_dgx2.csv --partition-file partitions_file_dgx2_delta.txt  -t  delta_ampere_dgx2.topo.txt -b delta_ampere_dgx2.topo.bin -n delta_ampere_dgx2
$FABRICTOOL -b delta_ampere_dgx2.topo.bin -o  delta_ampere_dgx2.topo_hex.txt
$FABRICTOOL -b delta_ampere_dgx2.topo.bin > delta_ampere_dgx2.topo_table.txt

#top loopback
./topology_gen_LR.py -c delta_ampere_top_loopback.csv --no-partitions -e -t  delta_ampere_top_loopback.topo.txt -b delta_ampere_top_loopback.topo.bin -n delta_ampere_top_loopback -port-map-info prospector_osfp_port_mapping.txt
$FABRICTOOL -b delta_ampere_top_loopback.topo.bin -o delta_ampere_top_loopback.topo_hex.txt
$FABRICTOOL -b delta_ampere_top_loopback.topo.bin > delta_ampere_top_loopback.topo_table.txt

#bottom loopback
./topology_gen_LR.py -c delta_ampere_bottom_loopback.csv --no-partitions -e -t  delta_ampere_bottom_loopback.topo.txt -b  delta_ampere_bottom_loopback.topo.bin -n delta_ampere_bottom_loopback -port-map-info prospector_osfp_port_mapping.txt
$FABRICTOOL -b delta_ampere_bottom_loopback.topo.bin -o delta_ampere_bottom_loopback.topo_hex.txt
$FABRICTOOL -b delta_ampere_bottom_loopback.topo.bin > delta_ampere_bottom_loopback.topo_table.txt

#both loopback
./topology_gen_LR.py -c delta_ampere_both_loopback.csv --no-partitions -e -t  delta_ampere_both_loopback.topo.txt -b  delta_ampere_both_loopback.topo.bin -n delta_ampere_both_loopback -port-map-info prospector_osfp_port_mapping.txt
$FABRICTOOL -b delta_ampere_both_loopback.topo.bin -o delta_ampere_both_loopback.topo_hex.txt
$FABRICTOOL -b delta_ampere_both_loopback.topo.bin > delta_ampere_both_loopback.topo_table.txt

#both no trunk
./topology_gen_LR.py -c delta_ampere_both_no_trunk.csv --no-partitions -t  delta_ampere_both_no_trunk.topo.txt -b  delta_ampere_both_no_trunk.topo.bin -n delta_ampere_both_no_trunk
$FABRICTOOL -b delta_ampere_both_no_trunk.topo.bin -o delta_ampere_both_no_trunk.topo_hex.txt
$FABRICTOOL -b delta_ampere_both_no_trunk.topo.bin > delta_ampere_both_no_trunk.topo_table.txt
set +x
echo "all topologies generated successfully"

