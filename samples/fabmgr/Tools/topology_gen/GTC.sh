#!/bin/bash
set -x 
set -e
#DGX-2
./topology_gen_LR.py --no-partitions -c delta_ampere_dgx2.csv -t delta_ampere_dgx2.topo.txt -n delta_ampere_dgx2 --no-partitions -b delta_ampere_dgx2.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b delta_ampere_dgx2.topo.bin -o delta_ampere_dgx2.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b delta_ampere_dgx2.topo.bin > delta_ampere_dgx2.topo_table.txt

#Delta lumiere
./topology_gen_LR.py --no-partitions -c delta_ampere_dgx2_lumiere.csv -t delta_ampere_dgx2_lumiere.topo.txt -n delta_ampere_dgx2_lumiere -b delta_ampere_dgx2_lumiere.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b delta_ampere_dgx2_lumiere.topo.bin -o delta_ampere_dgx2_lumiere.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b delta_ampere_dgx2_lumiere.topo.bin > delta_ampere_dgx2_lumiere.topo_table.txt

#fictional GTC ring with only one switch plane
#./topology_gen_LR.py -c fictional_GTC_ring.csv -r 1 -s -t fictional_GTC_ring.topo.txt --match-target-phy-id --set-rlan
./topology_gen_LR.py --no-partitions -c fictional_GTC_ring.csv -r 1 -s -t fictional_GTC_ring.topo.txt --match-target-phy-id -n fictional_GTC_ring -b fictional_GTC_ring.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b fictional_GTC_ring.topo.bin -o fictional_GTC_ring.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b fictional_GTC_ring.topo.bin > fictional_GTC_ring.topo_table.txt

#GTC DGX-2 ring full
./topology_gen_LR.py --no-partitions -c GTC_ring_dgx2.csv -r 1 -s -t GTC_ring_dgx2.topo.txt --match-target-phy-id -n GTC_ring_dgx2 -b GTC_ring_dgx2.topo.bin
#sed -i 's/192.168.254.1000/10.150.30.80/' GTC_ring_dgx2.topo.txt
#sed -i 's/192.168.254.1001/10.150.30.81/' GTC_ring_dgx2.topo.txt
#sed -i 's/192.168.254.1002/10.150.30.82/' GTC_ring_dgx2.topo.txt
#sed -i 's/192.168.254.1003/10.150.30.83/' GTC_ring_dgx2.topo.txt

../fabricTool/_out/Linux_amd64_debug/fabrictool -b GTC_ring_dgx2.topo.bin -o GTC_ring_dgx2.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b GTC_ring_dgx2.topo.bin > GTC_ring_dgx2.topo_table.txt

#GTC Luna ring full
./topology_gen_LR.py --no-partitions -c GTC_ring_luna.csv -r 1 -s -t GTC_ring_luna.topo.txt --match-target-phy-id -n GTC_ring_luna -b GTC_ring_luna.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b GTC_ring_luna.topo.bin -o GTC_ring_luna.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b GTC_ring_luna.topo.bin > GTC_ring_luna.topo_table.txt

#son of lumiere (E4700)
./topology_gen_LR.py  --unique-access-port --no-partitions --no-phy-id --no-match-even --match-gpu-port-num -c E4700_PG506_PG506_rev_2_multinode.csv -t E4700_PG506_PG506_rev_2_multinode.topo.txt -b E4700_PG506_PG506_rev_2_multinode.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_2_multinode.topo.bin -o E4700_PG506_PG506_rev_2_multinode.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_2_multinode.topo.bin > E4700_PG506_PG506_rev_2_multinode.topo_table.txt

set +x
set +e
