#!/bin/bash
set -e
set -x
./topology_gen_LR.py  --unique-access-port --no-partitions --no-phy-id --no-match-even --match-gpu-port-num -c E4700_PG506_PG506_rev_2.csv -t E4700_PG506_PG506_rev_2.topo.txt -b E4700_PG506_PG506_rev_2.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_2.topo.bin -o E4700_PG506_PG506_rev_2.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_2.topo.bin > E4700_PG506_PG506_rev_2.topo_table.txt


./topology_gen_LR.py  --set-rlan --spray --no-partitions --no-phy-id --no-match-even -c E4700_PG506_PG506_rev_2.csv -t E4700_PG506_PG506_rev_2_spray.topo.txt -b E4700_PG506_PG506_rev_2_spray.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_2_spray.topo.bin -o E4700_PG506_PG506_rev_2_spray.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_2_spray.topo.bin > E4700_PG506_PG506_rev_2_spray.topo_table.txt

#./topology_gen_LR.py  --set-rlan --no-partitions --no-phy-id --no-match-even --match-gpu-port-num -c E4700_PG506_PG506_rev_2.csv -t E4700_PG506_PG506_rev_2_rlan.topo.txt -b E4700_PG506_PG506_rev_2_rlan.topo.bin
./topology_gen_LR.py  --set-rlan --spray --no-partitions --no-phy-id --no-match-even -c E4700_PG506_PG506_rev_2.csv -t E4700_PG506_PG506_rev_2_rlan_spray.topo.txt -b E4700_PG506_PG506_rev_2_rlan_spray.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_2_rlan_spray.topo.bin -o E4700_PG506_PG506_rev_2_rlan_spray.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_2_rlan_spray.topo.bin > E4700_PG506_PG506_rev_2_rlan_spray.topo_table.txt

./topology_gen_LR.py  --set-rlan --no-partitions --no-phy-id --no-match-even --match-gpu-port-num -c E4700_PG506_PG506_rev_2.csv -t E4700_PG506_PG506_rev_2_rlan.topo.txt -b E4700_PG506_PG506_rev_2_rlan.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_2_rlan.topo.bin -o E4700_PG506_PG506_rev_2_rlan.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_rev_2_rlan.topo.bin > E4700_PG506_PG506_rev_2_rlan.topo_table.txt

if [[ $1 == "scp" ]]; then
    echo "Copying files to target"
    scp E4700_PG506_PG506_rev_2.topo.bin E4700_PG506_PG506_rev_2_spray.topo.bin E4700_PG506_PG506_rev_2_rlan_spray.topo.bin E4700_PG506_PG506_rev_2_rlan.topo.bin root@$2:
else
    echo "Not copying files to target"
fi

set +x
echo "Topology generation successful"
