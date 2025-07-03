#!/bin/bash
set -e
set -x
./topology_gen_LR.py  --no-partitions --no-phy-id --no-match-even --match-gpu-port-num -c E4700_PG506_PG506_rev.csv -t E4700_PG506_PG506_rev_gpu-2-3.topo.txt -b E4700_PG506_PG506_rev_gpu-2-3.topo.bin --gpu-ids 1,2,3
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_gpu-2-3.topo.bin -o E4700_PG506_PG506_rev_gpu-2-3.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_gpu-2-3.topo.bin > E4700_PG506_PG506_rev_gpu-2-3.topo_table.txt

./topology_gen_LR.py  --no-partitions --no-phy-id --no-match-even --match-gpu-port-num -c E4700_PG506_PG506_rev.csv -t E4700_PG506_PG506_rev.topo.txt -b E4700_PG506_PG506_rev.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev.topo.bin -o E4700_PG506_PG506_rev.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev.topo.bin > E4700_PG506_PG506_rev.topo_table.txt


./topology_gen_LR.py  --set-rlan --spray --no-partitions --no-phy-id --no-match-even -c E4700_PG506_PG506_rev.csv -t E4700_PG506_PG506_rev_spray.topo.txt -b E4700_PG506_PG506_rev_spray.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_spray.topo.bin -o E4700_PG506_PG506_rev_spray.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_spray.topo.bin > E4700_PG506_PG506_rev_spray.topo_table.txt

#./topology_gen_LR.py  --set-rlan --no-partitions --no-phy-id --no-match-even --match-gpu-port-num -c E4700_PG506_PG506_rev.csv -t E4700_PG506_PG506_rev_rlan.topo.txt -b E4700_PG506_PG506_rev_rlan.topo.bin
./topology_gen_LR.py  --set-rlan --no-partitions --no-phy-id --no-match-even -c E4700_PG506_PG506_rev.csv -t E4700_PG506_PG506_rev_rlan.topo.txt -b E4700_PG506_PG506_rev_rlan.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_rlan.topo.bin -o E4700_PG506_PG506_rev_rlan.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_rlan.topo.bin > E4700_PG506_PG506_rev_rlan.topo_table.txt

scp E4700_PG506_PG506_rev_gpu-2-3.topo.bin E4700_PG506_PG506_rev.topo.bin E4700_PG506_PG506_rev_spray.topo.bin E4700_PG506_PG506_rev_rlan.topo.bin root@172.16.154.227:

set +x
echo "Topology generation successful"
