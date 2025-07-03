#!/bin/bash
set -e
./topology_gen_LR.py  --no-partitions --no-phy-id --no-match-even --loopout-from-trunk -c E4700_PG506_E4702.csv -t E4700_PG506_E4702.topo.txt -b E4700_PG506_E4702.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_E4702.topo.bin -o E4700_PG506_E4702.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_E4702.topo.bin  > E4700_PG506_E4702.topo_table.txt


./topology_gen_LR.py  --no-partitions --no-phy-id --no-match-even --loopout-from-trunk -c E4700_E4702_PG506.csv -t E4700_E4702_PG506.topo.txt -b E4700_E4702_PG506.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_E4702_PG506.topo.bin -o E4700_E4702_PG506.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_E4702_PG506.topo.bin  > E4700_E4702_PG506.topo_table.txt

./topology_gen_LR.py  --no-partitions --no-phy-id --no-match-even --match-gpu-port-num -c E4700_PG506_PG506.csv -t E4700_PG506_PG506.topo.txt -b E4700_PG506_PG506.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506.topo.bin -o E4700_PG506_PG506.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506.topo.bin > E4700_PG506_PG506.topo_table.txt


./topology_gen_LR.py  --set-rlan --spray --no-partitions --no-phy-id --no-match-even -c E4700_PG506_PG506_rev.csv -t E4700_PG506_PG506_spray.topo.txt -b E4700_PG506_PG506_spray.topo.bin
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_spray.topo.bin -o E4700_PG506_PG506_spray.topo_hex.txt
../fabricTool/_out/Linux_amd64_debug/fabrictool -b  E4700_PG506_PG506_spray.topo.bin > E4700_PG506_PG506_spray.topo_table.txt

cp E4700_PG506_E4702.topo_hex.txt E4700_PG506_E4702.topo_table.txt E4700_E4702_PG506.topo_hex.txt E4700_E4702_PG506.topo_table.txt E4700_PG506_PG506.topo_hex.txt E4700_PG506_PG506.topo_table.txt ~/e4700_topo/


