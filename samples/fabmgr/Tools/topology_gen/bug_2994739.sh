#!/bin/bash
set -e 
set -x

#top only WOLB
./topology_gen_LR.py -c delta_ampere_top.csv --no-partitions -t  delta_ampere_top.topo.txt -b delta_ampere_top.topo.bin -n delta_ampere_top
./fabrictool -b delta_ampere_top.topo.bin -o delta_ampere_top.topo_hex.txt
./fabrictool -b delta_ampere_top.topo.bin > delta_ampere_top.topo_table.txt

#top only WOLB 8_15
./topology_gen_LR.py -c delta_ampere_top.csv --no-partitions -t  delta_ampere_top_gpu_8_15.topo.txt -b delta_ampere_top_gpu_8_15.topo.bin -n delta_ampere_top_gpu_8_15 --gpu-ids 8,15
./fabrictool -b delta_ampere_top_gpu_8_15.topo.bin -o delta_ampere_top_gpu_8_15.topo_hex.txt
./fabrictool -b delta_ampere_top_gpu_8_15.topo.bin > delta_ampere_top_gpu_8_15.topo_table.txt

#bottom only WOLB
./topology_gen_LR.py -c delta_ampere_bottom.csv --no-partitions -t  delta_ampere_bottom.topo.txt -b  delta_ampere_bottom.topo.bin -n delta_ampere_bottom
./fabrictool -b delta_ampere_bottom.topo.bin -o delta_ampere_bottom.topo_hex.txt
./fabrictool -b delta_ampere_bottom.topo.bin > delta_ampere_bottom.topo_table.txt

#bottom only WOLB 0_7
./topology_gen_LR.py -c delta_ampere_bottom.csv --no-partitions -t  delta_ampere_bottom_gpu_0_7.topo.txt -b  delta_ampere_bottom_gpu_0_7.topo.bin -n delta_ampere_bottom_gpu_0_7 --gpu-ids 0,7
./fabrictool -b delta_ampere_bottom_gpu_0_7.topo.bin -o delta_ampere_bottom_gpu_0_7.topo_hex.txt
./fabrictool -b delta_ampere_bottom_gpu_0_7.topo.bin > delta_ampere_bottom_gpu_0_7.topo_table.txt

#both boards
./topology_gen_LR.py -c delta_ampere_dgx2.csv --no-partitions -t  delta_ampere_dgx2.topo.txt -b delta_ampere_dgx2.topo.bin -n delta_ampere_dgx2
./fabrictool -b delta_ampere_dgx2.topo.bin -o  delta_ampere_dgx2.topo_hex.txt
./fabrictool -b delta_ampere_dgx2.topo.bin > delta_ampere_dgx2.topo_table.txt

#top loopback
./topology_gen_LR.py -c delta_ampere_top_loopback.csv --no-partitions -e -t  delta_ampere_top_loopback.topo.txt -b delta_ampere_top_loopback.topo.bin -n delta_ampere_top_loopback
./fabrictool -b delta_ampere_top_loopback.topo.bin -o delta_ampere_top_loopback.topo_hex.txt
./fabrictool -b delta_ampere_top_loopback.topo.bin > delta_ampere_top_loopback.topo_table.txt

#top loopback 8_15
./topology_gen_LR.py -c delta_ampere_top_loopback.csv --no-partitions -e -t  delta_ampere_top_gpu_8_15_loopback.topo.txt -b delta_ampere_top_gpu_8_15_loopback.topo.bin -n delta_ampere_top_gpu_8_15_loopback --gpu-ids 8,15
./fabrictool -b delta_ampere_top_gpu_8_15_loopback.topo.bin -o delta_ampere_top_gpu_8_15_loopback.topo_hex.txt
./fabrictool -b delta_ampere_top_gpu_8_15_loopback.topo.bin > delta_ampere_top_gpu_8_15_loopback.topo_table.txt

#bottom loopback
./topology_gen_LR.py -c delta_ampere_bottom_loopback.csv --no-partitions -e -t  delta_ampere_bottom_loopback.topo.txt -b  delta_ampere_bottom_loopback.topo.bin -n delta_ampere_bottom_loopback
./fabrictool -b delta_ampere_bottom_loopback.topo.bin -o delta_ampere_bottom_loopback.topo_hex.txt
./fabrictool -b delta_ampere_bottom_loopback.topo.bin > delta_ampere_bottom_loopback.topo_table.txt

#bottom loopback 0_7
./topology_gen_LR.py -c delta_ampere_bottom_loopback.csv --no-partitions -e -t  delta_ampere_bottom_gpu_0_7_loopback.topo.txt -b  delta_ampere_bottom_gpu_0_7_loopback.topo.bin -n delta_ampere_bottom_gpu_0_7_loopback --gpu-ids 0,7
./fabrictool -b delta_ampere_bottom_gpu_0_7_loopback.topo.bin -o delta_ampere_bottom_gpu_0_7_loopback.topo_hex.txt
./fabrictool -b delta_ampere_bottom_gpu_0_7_loopback.topo.bin > delta_ampere_bottom_gpu_0_7_loopback.topo_table.txt

#both loopback
./topology_gen_LR.py -c delta_ampere_both_loopback.csv --no-partitions -e -t  delta_ampere_both_loopback.topo.txt -b  delta_ampere_both_loopback.topo.bin -n delta_ampere_both_loopback
./fabrictool -b delta_ampere_both_loopback.topo.bin -o delta_ampere_both_loopback.topo_hex.txt
./fabrictool -b delta_ampere_both_loopback.topo.bin > delta_ampere_both_loopback.topo_table.txt

#both loopback 0_7_8_15
./topology_gen_LR.py -c delta_ampere_both_loopback.csv --no-partitions -e -t  delta_ampere_both_gpu_0_7_8_15_loopback.topo.txt -b  delta_ampere_both_gpu_0_7_8_15_loopback.topo.bin -n delta_ampere_both_gpu_0_7_8_15_loopback --gpu-ids 0,7,8,15
./fabrictool -b delta_ampere_both_gpu_0_7_8_15_loopback.topo.bin -o delta_ampere_both_gpu_0_7_8_15_loopback.topo_hex.txt
./fabrictool -b delta_ampere_both_gpu_0_7_8_15_loopback.topo.bin > delta_ampere_both_gpu_0_7_8_15_loopback.topo_table.txt

#both no trunk
./topology_gen_LR.py -c delta_ampere_both_no_trunk.csv --no-partitions -t  delta_ampere_both_no_trunk.topo.txt -b  delta_ampere_both_no_trunk.topo.bin -n delta_ampere_both_no_trunk
./fabrictool -b delta_ampere_both_no_trunk.topo.bin -o delta_ampere_both_no_trunk.topo_hex.txt
./fabrictool -b delta_ampere_both_no_trunk.topo.bin > delta_ampere_both_no_trunk.topo_table.txt

#both no trunk 0_7_8_15
./topology_gen_LR.py -c delta_ampere_both_no_trunk.csv --no-partitions -t  delta_ampere_both_gpu_0_7_8_15_no_trunk.topo.txt -b  delta_ampere_both_gpu_0_7_8_15_no_trunk.topo.bin -n delta_ampere_both_gpu_0_7_8_15_no_trunk --gpu-ids 0,7,8,15
./fabrictool -b delta_ampere_both_gpu_0_7_8_15_no_trunk.topo.bin -o delta_ampere_both_gpu_0_7_8_15_no_trunk.topo_hex.txt
./fabrictool -b delta_ampere_both_gpu_0_7_8_15_no_trunk.topo.bin > delta_ampere_both_gpu_0_7_8_15_no_trunk.topo_table.txt


set +x
echo "all topologies generated successfully"

