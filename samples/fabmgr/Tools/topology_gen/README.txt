This tool uses the hardware links specified via a CSV file to generate a graph. 
It finds the shortest paths between GPUs and uses these to set up the correct routes.

It doesn’t yet handle snake topologies. Small scale snake topologies can be specified 
manually by providing a path through the graph via the “-p” option, and providing a paths 
file(see below for details). 

The tool requires topology_pb.py which is generated from the topology protobuf file 

Usage/Help
================================================================================================
#Display help
./topology_gen_LR.py -h

optional arguments:
  -h, --help            show this help message and exit
  -c CSV_FILE, --csv CSV_FILE
                        Input CSV file for hardware connections
  -v, --verbose         Verbose info messages
  -t TEXT_FILE, --text TEXT_FILE
                        Print text file for topology
  -n TOPOLOGY_NAME, --topology-name TOPOLOGY_NAME
                        Print text file for topology
  -b BINARY_FILE, --binary BINARY_FILE
                        Print binary file for topology
  -s, --spray           Spray over trunk links
  -r RING_DATELINE, --ring RING_DATELINE
                        Specify dateline for ring
  -l, --loopback        Access port loopback to GPU
  --loopback-from-trunk
                        Trunk port loopback to GPU
  --loopout-from-trunk  Trunk port loopout to GPU
  --stdout              Write topology file text to stdout
  -p PATHS_FILE, --paths PATHS_FILE
                        Input file specifying paths to use instead of
                        callwlating paths
  -m, --match-gpu-port-num
                        Src port of route should always be same as the dest
                        port
  --unique-access-port  Connect each trunk port to exactly one access port
                        intead of the reverse which is the default
  --no-match-even       Inter-connect all switch ports when building graph,
                        not just even-even, odd-odd
  --match-target-phy-id
                        Set Target ID to same value as physical ID
  --no-phy-id           Do not set switch Physical ID in topology file
  -e, --external-loopback
                        The paths need to go though external loopback port
  --set-rlan            Set rlan to "n" for nth GPU port connected to switch
  --use-gpu-indexes     Use gpuIndexes instead of Phy IDs
  -j JSON_OUTPUT_FILE, --json-output JSON_OUTPUT_FILE
                        Print JSON file for topology
  --no-partitions       Don't include partition information
  --gpu-ids GPU_ID_LIST
                        List of GPU physical ids to filter in
  --switch-ids SWITCH_ID_LIST
                        List of switch physical ids to filter in
  --partition-file PARTITIONS_FILE
                        File specifying the partitions
  --path-lens PATH_LENS
                        only paths of specified lengths will be choosen


CSV file
================================================================================================
* Each line provides one bi-directional hardware link. 
* nodeId, phyId, linkIndex, devType: These columns provide one end of the link
* nodeIdFar, phyIdFar, linkIndexFar, devTypeFar: these columns provide the other end of the link

nodeId, nodeIdFar: For a single node system these field should be set to zero
phyId, phyIdFar: These provides the physical ID of the switch/GPU
linkIndex, linkIndexFar: These provides the port/link number of the link
devType, devTypeFar: these provide the device type 0=switch 1=GPU

Paths File(optional)
================================================================================================
This provides a dictionary mapping a source, destination pairs to a lists of paths.

{(src1, dest1): [[path1], [path2]], (src2, dest2): [[path3]]}

src, dest: A tuple containing (Node, Physical ID, Link Index, device Type)
Path: A list of lists. Each list specifies a path from source to destination. the first entry is always the 
      source and the last entry is always a destination. Paths are uni-directional only. Each hop in the
	path is a tuple containing (Node, Physical ID, Link Index, device Type)

Files
================================================================================================
topology_gen_LR.py: tool to generate a topology protobuf for Ampere/Limerock
topology_gen.py: tool to generate a topology protobuf for Willow/Volta

topology_pb2.py: protobuf file (not -included). auto-generated from topology.proto.precomp

emulator_phy.csv: A CSV file describing the link in a emulator topology with one GPU and on Switch
emulator_paths_phy.txt:  An example paths file that specfies a path through the above emulator topology. 
			When this s provided only this path is used and paths are no longer callwlated over the graph

fictional_phy.csv: A fictional topology with 4 Ampere GPUs (phy Ids = 10,11,12,13) and four Limerock Switches (phy Ids = 3,4,5,6)
fictional.JPG: A hand drawn graph of the above fictional topology

delta_interposer_dgx2.csv: DGX2 populated with two delta interposer boards  
delta_interposer_bottom.csv: delta interposer board for bottom tray of DGX2  
delta_interposer_bottom_loopback.csv: delta interposer board for bottom tray of DGX2 with external loopback for trunk ports 
delta_interposer_top.csv: delta interposer board for top tray of DGX2
delta_interposer_top_loopback.csv: delta interposer board for top tray of DGX2 with external loopback for trunk ports

dgx2.csv: DGX-2 with top and bottom boards and Phy IDs from hardware docs
dgx2_lwl.csv: DGX-2 with top and bottom boards. generated by lwlink-train app hence no phy ids for GPUs

delta_ampere_bottom.csv: delta ampere board for bottom tray of DGX2
delta_ampere_top.csv: delta ampere board for top tray of DGX2
delta_ampere_dgx2.csv: DGX2 populated with two delta ampere boards

e4700_config1_ref.csv: reference CSV for dirrent possible combinations of GPU(PG506)/EXAMAX/E4702 on E4700
E4700_E4702_PG506.csv: E4702 in slot 0 , PG506 in slot 1 on E4700
E4700_PG506_E4702.csv: PG506 in slot 0 , E4702 in slot 1 on E4700
E4700_PG506_PG506.csv: PG506 in slot 0 and slot 1 on E4700

delta_ampere_both_loopout.csv: meant for loopout topologies required for bug 3015408


Ring topologies
====================================================================================================
GTC_ring_dgx2.csv, GTC_ring_luna.csv and fictional_GTC_ring.csv are three examples of ring topologies. See GTC.sh "DGX-2 ring full"
for an example of the argements to be given to construct a ring.

A few features of ring topos are:
1. In these topologies one node should be chosen as the dateline using the "-r <nodeid>" arguement. 
2. A ring is lwrently created as a spray topology by default. An access link sprays to all trunk links. 
3. When traffic crosses a switch on a node on an intermediate hop, it goes from an incoming trunk 
   port and is sprayed to outgoing trunk ports on all possible shortest paths.
4. On a dateline node for traffic that goes from trunk port to trunk port the VC number is switched from 0->1.
5. on destination node the VC number is always forced to 0. So traffic going from trunk to access port gets forced to VC 0
6. rings are bidirectional. The responses may/may not take the exact same path as the request.
7. from access ports traffic is sprayed on all shortest path. On a 4-node ring (A <-> B <-> C <-> D <-> A)This can lead to twice the number of
   paths to a node one hop away than to an adjacent node as paths A->B->C are of same length as paths A->D->C.
8. For Ampere/LR even numbered ports are connected to even numbered ports only and odd numbered ports to odd numbered ports only. 
   This is so that we don't require RLANs

Path lengths usage
=================================================================================================
There are some topologies in which only paths of certain lengths are desired. For example in loopout topologies in bug_3015408.sh
the correct paths are only of length 6. Some ports can still be connected via longer paths but this ends up connecting even and 
odd numbered ports. There might be a better solution to this issue but i haven't found one yet.
 

This option is also useful in some experimental setups

Python modules needed
================================================================================================
Some addintional python modules may be needed:
pip install pandas
pip install networkx
pip install protobuf

topology_pb2.py: thisis generated when Fabric manager is compiled and is not checked in to this directory

Examples:
================================================================================================
#Generate a topology for the emulator config with one GOU and one Switch. Hardcoded path provided
./topology_gen_LR.py -c emulator_phy.csv -t emulator_topo.txt -p emulator_paths_phy.txt -b emulator_topo.bin

#Generate a topology for a fictional config
./topology_gen_LR.py -c fictional_phy.csv -t fictional_topo.txt -b fictional_topo.bin -l


