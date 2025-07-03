#!/bin/bash
set -x 
set -e

#Delta lumiere
./topology_gen_LR.py --no-partitions -c se_two_node_optical.csv -t se_two_node_optical.topo.txt -n se_two_node_optical -b se_two_node_optical.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b se_two_node_optical.topo.bin -o se_two_node_optical.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b se_two_node_optical.topo.bin > se_two_node_optical.topo_table.txt

#sed -i 's/192.168.254.1000/10.150.30.80/' GTC_ring_dgx2.topo.txt
#sed -i 's/192.168.254.1001/10.150.30.81/' GTC_ring_dgx2.topo.txt

set +x
set +e
