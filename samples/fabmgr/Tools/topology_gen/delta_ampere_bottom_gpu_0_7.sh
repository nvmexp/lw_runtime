#!/bin/bash
set -e 
set -x

#top only
#./topology_gen_LR.py -c delta_ampere_top.csv --no-partitions -t  delta_ampere_top.topo.txt -b delta_ampere_top.topo.bin -n delta_ampere_top
#../fabricTool/_out/Linux_amd64_debug/fabrictool -b delta_ampere_top.topo.bin -o delta_ampere_top.topo_hex.txt
#../fabricTool/_out/Linux_amd64_debug/fabrictool -b delta_ampere_top.topo.bin > delta_ampere_top.topo_table.txt
#bottom only
./topology_gen_LR.py -c delta_ampere_bottom.csv --no-partitions -t  delta_ampere_bottom_gpu_0_7.topo.txt -b  delta_ampere_bottom_gpu_0_7.topo.bin -n delta_ampere_bottom_gpu_0_7 --gpu-ids 0,7
../fabricTool/_out/Linux_amd64_debug/fabrictool -b delta_ampere_bottom_gpu_0_7.topo.bin -o delta_ampere_bottom_gpu_0_7.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b delta_ampere_bottom_gpu_0_7.topo.bin > delta_ampere_bottom_gpu_0_7.topo_table.txt

#both boards
#./topology_gen_LR.py -c delta_ampere_dgx2.csv --no-partitions -t  delta_ampere_dgx2.topo.txt -b delta_ampere_dgx2.topo.bin -n delta_ampere_dgx2
#../fabricTool/_out/Linux_amd64_debug/fabrictool -b delta_ampere_dgx2.topo.bin -o  delta_ampere_dgx2.topo_hex.txt
#../fabricTool/_out/Linux_amd64_debug/fabrictool -b delta_ampere_dgx2.topo.bin > delta_ampere_dgx2.topo_table.txt

#top loopback
#./topology_gen_LR.py -c delta_ampere_top_loopback.csv --no-partitions -e -t  delta_ampere_top_loopback.topo.txt -b delta_ampere_top_loopback.topo.bin -n delta_ampere_top_loopback
#../fabricTool/_out/Linux_amd64_debug/fabrictool -b delta_ampere_top_loopback.topo.bin -o delta_ampere_top_loopback.topo_hex.txt
#../fabricTool/_out/Linux_amd64_debug/fabrictool -b delta_ampere_top_loopback.topo.bin > delta_ampere_top_loopback.topo_table.txt

#bottom loopback
./topology_gen_LR.py -c delta_ampere_bottom_loopback.csv --no-partitions -e -t  delta_ampere_bottom_gpu_0_7_loopback.topo.txt -b  delta_ampere_bottom_gpu_0_7_loopback.topo.bin -n delta_ampere_bottom_gpu_0_7_loopback --gpu-ids 0,7
../fabricTool/_out/Linux_amd64_debug/fabrictool -b delta_ampere_bottom_gpu_0_7_loopback.topo.bin -o delta_ampere_bottom_gpu_0_7_loopback.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b delta_ampere_bottom_gpu_0_7_loopback.topo.bin > delta_ampere_bottom_gpu_0_7_loopback.topo_table.txt

set +x
echo "all topologies generated successfully"

