#!/usr/bin/python
import sys
import pandas as pd
import networkx as nx
import topology_pb2
import copy
import datetime
from time import gmtime, strftime
import argparse
from google.protobuf import text_format
import ast


#For each hardware  The following valuses need to be provided
#GPU arch specific
memoryPerGpu = (64 * 1024 * 1024 * 1024)
memoryPerGpuReal = (64 *1024 * 1024 * 1024)
portsPerGpu = 12
fabricVersion = 0x100
#TODO: remove this hardcoding
flaIndexStart =  0x10

usePhyId = False
isSnake = False
#TODO:only partial support for spray. Doesn't work yet
isSpray = False

def getNodeId(portName):
    return portName[0]

def getPhyId(portName):
    return portName[1]

def getPortId(portName):
    return portName[2]

def getBusIdFromBdf(bdf):
    return int(bdf.split(":")[1])

def isGpuPort(portName):
    return portName[3] == 1

def isSwitchPort(portName):
    return portName[3] == 0

def getAllGpuPorts(allLinks):
    return [_ for _ in allLinks.iterkeys() if isGpuPort(_)]

def getAllSwitchPorts(allLinks):
    return [_ for _ in allLinks.iterkeys() if isSwitchPort(_)]

def getConnectedPort(allLinks, port):
    return allLinks[port]

def gpuBaseAddress(nodeId, phyId, gpuPhyIds):
    gpuIndex = gpuPhyIds.index((nodeId, phyId))
    return gpuIndex * memoryPerGpu

def gpuFlaBaseAddress(nodeId, phyId, gpuPhyIds):
    gpuIndex = gpuPhyIds.index((nodeId, phyId)) + flaIndexStart
    return gpuIndex * memoryPerGpu

def gpuTargetId(nodeId, phyId, gpuPhyIds):
    return gpuPhyIds.index((nodeId, phyId))

def print_verbose(s = "\n"):
    if args.verbose:
        print s
    return 

#check if a trunk port should be connected to an access port
def isAccessPortConnected(portsPerSwitch, switchNodeId, switchPhyId, trunkPortId, accessPortId):
    if portsPerSwitch[(switchNodeId, switchPhyId)][0].index(accessPortId) == portsPerSwitch[(switchNodeId, switchPhyId)][1].index(trunkPortId):
        return True
    else:
        return False

def isAccessPort(portsPerSwitch, p):
    if getPortId(p) in portsPerSwitch[(getNodeId(p), getPhyId(p))][0]:
        return True
    else:
        return False

def getAllLinks(df):
    allLinks = dict()
    for index, row in df.iterrows():
        #TODO:When the CSV file is from lwlink-train tool it has no way of providing the Phy Ids for GPUs.
        #So we use BusIds instead. This is a temporary solution
        phyId = row["phyId"]
        if phyId == -1:
            phyId = getBusIdFromBdf(row["(d::b:d.f)"])

        phyIdFar = row["phyIdFar"]
        if phyIdFar == -1:
            phyId = getBusIdFromBdf(row["(d::b:d.f)Far"])

        allLinks[(row["nodeId"] , phyId, row["linkIndex"], row["devType"])] = (row["nodeIdFar"], phyIdFar, row["linkIndexFar"], row["devTypeFar"])
        allLinks[(row["nodeIdFar"], phyIdFar, row["linkIndexFar"], row["devTypeFar"])] = (row["nodeId"] , phyId, row["linkIndex"], row["devType"])
    return allLinks

def getAllGpuPhyIds(gpuPorts):
    gpuPhyIds = list(set([ (getNodeId(_), getPhyId(_)) for _ in gpuPorts]))
    gpuPhyIds.sort()
    return gpuPhyIds

def getAllSwitchPhyIds(switchPorts):
    switchPhyIds = list(set([(getNodeId(_), getPhyId(_)) for _ in switchPorts]))
    switchPhyIds.sort()
    return switchPhyIds

def getAllNodeIds(allLinks):
    nodeIds = list(set([getNodeId(_) for _ in allLinks]))
    nodeIds.sort()
    return nodeIds

def getPortsPerSwitch(allLinks, switchPorts):
    portsPerSwitch = dict()

    for i in switchPorts:
        accessPorts = []
        trunkPorts = []
        if (getNodeId(i), getPhyId(i)) in portsPerSwitch:
            accessPorts = portsPerSwitch[(getNodeId(i), getPhyId(i))][0]
            trunkPorts = portsPerSwitch[(getNodeId(i), getPhyId(i))][1]

        if isSwitchPort(getConnectedPort(allLinks, i)):
            #i am trunk port do something
            trunkPorts.append(getPortId(i))
            trunkPorts.sort()
        else:
            #i am access port do something   
            accessPorts.append(getPortId(i))
            accessPorts.sort()
        portsPerSwitch[(getNodeId(i), getPhyId(i))] = (accessPorts, trunkPorts)
    return portsPerSwitch

def buildGraph(allLinks, switchPorts, portsPerSwitch):
    G = nx.DiGraph()
    for l, r in allLinks.iteritems():
        G.add_edge(l, r)
    if isSnake == False:
        print_verbose("Building graph: Interconnecting ports within a switch")
        for i in xrange(len(switchPorts)):
            for j in xrange(len(switchPorts)):
                if switchPorts[i] == switchPorts[j]:
                    continue
                fromPortId = getPortId(switchPorts[i])
                toPortId = getPortId(switchPorts[j])
                #note that even ports are connected to even ports only and odd ports to odd ports only
                #TODO: In case of spray we might need to set RLANs and not bother about even/odd
                if (fromPortId % 2) != (toPortId % 2):
                    continue
                if getNodeId(switchPorts[i]) == getNodeId(switchPorts[j]) and getPhyId(switchPorts[i]) == getPhyId(switchPorts[j]):
                    #Check if switchPorts[i] is an access port. Connect an access port to all other ports(uni-directional)
                    if isAccessPort(portsPerSwitch, switchPorts[i]):
                        G.add_edge(switchPorts[i], switchPorts[j])
                    #If we get here it means switchPorts[i] is a trunk port. connect a trunk port to exactly one access port
                    elif isAccessPort(portsPerSwitch, switchPorts[j]):
                        #a trunk port is assigned to a gpu..this remains same accross all switchPorts.
                        if isSpray or isAccessPortConnected(portsPerSwitch, getNodeId(switchPorts[i]), getPhyId(switchPorts[i]), fromPortId, toPortId):
                            G.add_edge(switchPorts[i], switchPorts[j])
    else:
        #TODO: Doesn't work yet.
        #snake ports by adding uni-derctional links between conselwtive ports within
        #a switch that are not already directly connected
        for nodeId, phyId in portsPerSwitch.iterkeys():
            thisSwitchPorts = portsPerSwitch[(nodeId, phyId)][0] + portsPerSwitch[(nodeId, phyId)][1]
            thisSwitchPorts.sort()
            for i in xrange(len(thisSwitchPorts)):
                l = (nodeId, phyId, thisSwitchPorts[i])
                r = (nodeId, phyId, thisSwitchPorts[(i + 1) % len(thisSwitchPorts)])
                if allLinks[l] != r:
                    G.add_edge((nodeId, phyId, thisSwitchPorts[i]) , 
                                (nodeId, phyId, thisSwitchPorts[(i + 1) % len(thisSwitchPorts)]))
                    print "adding uni-directional link" , l, r
    return G

def findRoutes(G, gpuPorts):
    #get paths from file instead of callwlating from graph
    if args.paths_file:
        with open(args.paths_file, 'r') as f:
            return ast.literal_eval(f.read())
    
    #callwlate paths from graph
    allPaths = dict()
    found_path=0
    for l in gpuPorts:
        for r in gpuPorts:
            if getNodeId(l) != getNodeId(r) or getPhyId(l) != getPhyId(r):
                if nx.has_path(G, l, r):
                    if isSpray == True:
                        shortest_paths =  [_ for _ in nx.all_shortest_paths(G, l, r)]
                        allPaths[(l, r)] = shortest_paths
                        found_path = found_path + 1

                    else:
                        shortest_path =  nx.shortest_path(G, l, r)
                        allPaths[(l, r)] = [shortest_path]
                        found_path = found_path + 1

                 
            elif l == r:
                if args.loopback:
                    nbrs = [_ for _ in G.neighbors(l)]
                    loop = [l, nbrs[0], nbrs[0], l]
                    allPaths[(l, r)] = [loop]
                    found_path = found_path + 1
                if isSnake:
                    pass
    return allPaths

def buildTopologyInfo(allPaths, nodeIds, gpuPhyIds, switchPhyIds, portsPerSwitch, switchPorts):
    def buildRmapPolicyInfo(topologyInfo, gpuPhyIds, dest, path, hopNum, isFlaOnly):
        gpaIndex = gpuTargetId(getNodeId(dest), getPhyId(dest), gpuPhyIds)
        #GPA entry
        if not isFlaOnly:
            rmapEntry = dict()
            rmapEntry["targetId"] = gpaIndex
            rmapEntry["index"] = gpaIndex
            rmapEntry["address"] = gpuBaseAddress(getNodeId(dest), getPhyId(dest), gpuPhyIds)
            topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["rmapEntries"][gpaIndex] = rmapEntry

        #FLA entry
        rmapEntry = dict()
        flaIndex = gpaIndex + flaIndexStart
        rmapEntry["targetId"] = gpaIndex
        rmapEntry["index"] = flaIndex
        rmapEntry["address"] = gpuFlaBaseAddress(getNodeId(dest), getPhyId(dest), gpuPhyIds)
        topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["rmapEntries"][flaIndex] = rmapEntry

    def buildRidRouteInfo(topologyInfo, gpuPhyIds, dest, path, hopNum):
        #for forward direction of path and return path
        ridEntry = dict()
        gpaIndex = gpuTargetId(getNodeId(dest), getPhyId(dest), gpuPhyIds)
        ridEntry["index"] = gpaIndex
        ridEntry["port"] = getPortId( path[ hopNum  + 1])
        topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["ridEntries"][gpaIndex] = ridEntry

    def buildRlanRouteInfo(topologyInfo, gpuPhyIds, dest, path, hopNum):
        rlanEntry = dict()
        gpaIndex = gpuTargetId(getNodeId(dest), getPhyId(dest), gpuPhyIds)
        rlanEntry["index"] = gpaIndex
        #this is used to check if we are on even or or port hence if we should use rlan 0 or 1 
        #Note it doesn't matter here if setting hopNum or hopNum + 1 as both will either be even or odd
        rlanEntry["port"] = getPortId( path[ hopNum  + 1])
        topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["rlanEntries"][gpaIndex] = rlanEntry

    topologyInfo = dict()
    for nodeId in nodeIds:
        topologyInfo[nodeId]  = dict()
        topologyInfo[nodeId]["version"] = 0x100
        topologyInfo[nodeId]["gpuPhyIds"] = [p for n, p in gpuPhyIds if n == nodeId]
        topologyInfo[nodeId]["gpuPhyIds"].sort()
        topologyInfo[nodeId]["switchPhyIds"] = [p for n, p in switchPhyIds if n == nodeId]
        topologyInfo[nodeId]["switchPhyIds"].sort()

    for nodeId, switchPhyId in switchPhyIds:
        topologyInfo[nodeId][switchPhyId] = dict()

    for nodeId, switchPhyId, portId, _ in switchPorts:
        topologyInfo[nodeId][switchPhyId][portId] = dict()
        topologyInfo[nodeId][switchPhyId][portId]["rmapEntries"] = dict()
        topologyInfo[nodeId][switchPhyId][portId]["ridEntries"] = dict()
        topologyInfo[nodeId][switchPhyId][portId]["rlanEntries"] = dict()

    for (src, dest), paths in allPaths.iteritems():
        #print "Walking a path, len=", len(paths[0]), "path=", paths[0]
        if len(paths) != 1:
            print paths
            exit(0)

        for path in paths:
            #isSpray
            for hopNum in xrange(1, len(path) - 1, 2):
                #TODO: Amepere/Limerock modification go into these two functions
                if isAccessPort(portsPerSwitch, path[hopNum]):
                    if src == dest and args.loopback:
                        buildRmapPolicyInfo(topologyInfo, gpuPhyIds, dest, path, hopNum, True)
                    else:
                        #print (src, "--->", dest, path)
                        buildRmapPolicyInfo(topologyInfo, gpuPhyIds, dest, path, hopNum, False)
                buildRidRouteInfo(topologyInfo, gpuPhyIds, dest, path, hopNum)
                buildRlanRouteInfo(topologyInfo, gpuPhyIds, dest, path, hopNum)
    return topologyInfo

def buildTopologyProtobuf(topologyInfo, nodeIds, allLinks, portsPerSwitch, gpuPhyIds):
    def setPartitionInfo(fabricNode, isBaremetal, gpuCount, switchCount, lwLinkIntraTrunkConnCount, lwLinkInterTrunkConnCount):
        partitionInfo = None
        if isBaremetal:
            partitionInfo = fabricNode.partitionInfo.bareMetalInfo.add()
        else:
            partitionInfo = fabricNode.partitionInfo.ptVirtualInfo.add()
        partitionInfo.metaData.gpuCount = gpuCount
        partitionInfo.metaData.switchCount = switchCount
        partitionInfo.metaData.lwLinkIntraTrunkConnCount = lwLinkIntraTrunkConnCount
        partitionInfo.metaData.lwLinkInterTrunkConnCount = lwLinkInterTrunkConnCount
    def setGpuinfo(topologyInfo, fabricNode, nodeId, gpuPhyIds):
        for gpuPhyId in topologyInfo[nodeId]["gpuPhyIds"]:
            gpu = fabricNode.gpu.add()
            gpu.version = fabricVersion
            gpuIndex = topologyInfo[nodeId]["gpuPhyIds"].index(gpuPhyId)
            gpu.ECID = "N" + str(nodeId) + "_G" + str(gpuIndex)
            gpu.fabricAddrBase = gpuBaseAddress(nodeId, gpuPhyId, gpuPhyIds)
            gpu.fabricAddrRange = memoryPerGpuReal
            gpu.GPABase = gpuBaseAddress(nodeId, gpuPhyId, gpuPhyIds)
            gpu.targetId = gpuTargetId(nodeId, gpuPhyId, gpuPhyIds)
            gpu.GPARange = memoryPerGpuReal
            gpu.FLABase = gpuFlaBaseAddress(nodeId, gpuPhyId, gpuPhyIds)
            gpu.FLARange = memoryPerGpuReal
            gpu.physicalId = gpuPhyId

    def setAccessPort(topologyInfo, gpuPhyIds, lwswitch, portsPerSwitch, nodeId, switchPhyId, portId, val):
        access = lwswitch.access.add()
        access.version = fabricVersion
        access.connectType = topology_pb2.ACCESS_GPU_CONNECT
        access.localPortNum = portId
        access.farNodeID, access.farPeerID, access.farPortNum, _ = getConnectedPort(allLinks, (nodeId, switchPhyId, portId, 0))

        access.config.version = fabricVersion
        access.config.type = topology_pb2.ACCESS_PORT_GPU
        access.config.RequesterLinkID = gpuTargetId(access.farNodeID, access.farPeerID, gpuPhyIds)
        access.config.phyMode = topology_pb2.DC_COUPLED
        access.config.RlanID = 0
        access.config.maxTargetID = topology_pb2.MAX_512_TARGET_ID

        #iterate over all rmap entries for this port
        inorder = [(index, rmapEntry) for index, rmapEntry in val["rmapEntries"].iteritems()]
        inorder.sort()
        for index, rmapEntry in inorder:
            rmapRte = access.rmapPolicyTable.add()
            rmapRte.targetId = rmapEntry["targetId"]
            rmapRte.index = index
            rmapRte.version = fabricVersion
            rmapRte.entryValid = 1
            rmapRte.address = rmapEntry["address"]
            rmapRte.reqContextChk = 0x0
            rmapRte.reqContextMask = 0x0
            rmapRte.reqContextRep = 0x0
            rmapRte.addressOffset = 0x0
            rmapRte.addressBase = 0x0
            rmapRte.addressLimit = 0x0
            rmapRte.routingFunction = 0x1
            rmapRte.irlSelect = 0x0
            rmapRte.p2rSwizEnable = 0x0
            rmapRte.mult2 = 0x0
            rmapRte.planeSelect = 0x0

        #iterate over all rid entries for this port
        inorder = [(index, ridEntry) for index, ridEntry in val["ridEntries"].iteritems()]
        inorder.sort()
        for index, ridEntry in inorder:
            ridRte = access.ridRouteTable.add()
            ridRte.index = index
            ridRte.version = fabricVersion
            ridRte.valid = 1
            #we use fixed routing mode in which rmod[3:0] gives the entry to use in the portList. 
            #setting rmod[6]=0 means don't use the RLAN table entry we still have to set the corresponding RLAN table entry to valid
            ridRte.rMod = 0x0
            port = ridRte.portList.add()
            port.portIndex = ridEntry["port"]
            port.vcMap = 0
        
        #iterate over all rlan entries for this port
        inorder = [(index, rlanEntry) for index, rlanEntry in val["rlanEntries"].iteritems()]
        inorder.sort()
        for index, rlanEntry in inorder:
            rlanRte = access.rlanRouteTable.add()
            rlanRte.index = index
            rlanRte.version = fabricVersion
            rlanRte.valid = 1
        return

    def setTrunkPort(topologyInfo, allLinks, lwswitch, portsPerSwitch, nodeId, switchPhyId, portId, val):
        trunk = lwswitch.trunk.add() 
        trunk.version = fabricVersion
        trunk.connectType = topology_pb2.TRUNK_SWITCH_CONNECT
        trunk.localPortNum = portId
        trunk.farNodeID, trunk.farSwitchID, trunk.farPortNum, _ = getConnectedPort(allLinks, (nodeId, switchPhyId, portId, 0))

        trunk.config.version = fabricVersion
        trunk.config.type = topology_pb2.TRUNK_PORT_SWITCH
        trunk.config.phyMode = topology_pb2.DC_COUPLED
        trunk.config.RlanID = 0
        trunk.config.maxTargetID = topology_pb2.MAX_512_TARGET_ID

        #iterate over all rid entries for this port
        inorder = [(index, ridEntry) for index, ridEntry in val["ridEntries"].iteritems()]
        inorder.sort()
        for index, ridEntry in inorder:
            ridRte = trunk.ridRouteTable.add()
            ridRte.index = index
            ridRte.version = fabricVersion
            ridRte.valid = 1
            #we use fixed routing mode in which rmod[3:0] gives the entry to use in the portList. 
            #setting rmod[6]=0 means don't use the RLAN table entry we still have to set the corresponding RLAN table entry to valid
            ridRte.rMod = 0x0
            port = ridRte.portList.add()
            port.portIndex = ridEntry["port"]
            port.vcMap = 0
 
        #iterate over all rlan entries for this port
        inorder = [(index, rlanEntry) for index, rlanEntry in val["rlanEntries"].iteritems()]
        inorder.sort()
        for index, rlanEntry in inorder:
            rlanRte = trunk.rlanRouteTable.add()
            rlanRte.index = index
            rlanRte.version = fabricVersion
            rlanRte.valid = 1

        return

    def setSwitchInfo(topologyInfo, gpuPhyIds, fabricNode, portsPerSwitch, nodeId):
        for switchPhyId in topologyInfo[nodeId]["switchPhyIds"]:
            lwswitch = fabricNode.lwswitch.add()
            lwswitch.version = fabricVersion
            switchIndex = topologyInfo[nodeId]["switchPhyIds"].index(switchPhyId)
            lwswitch.ECID = "N" + str(nodeId) + "_S" + str(switchIndex)
            lwswitch.physicalId = switchPhyId
            #iterate over all ports in this switch
            for portId, val in topologyInfo[nodeId][switchPhyId].iteritems():
                #Amepere/Limerock modification go into these two functions( setAccessPort & setTrunkPort)
                if isAccessPort(portsPerSwitch, (nodeId, switchPhyId, portId)):
                    setAccessPort(topologyInfo, gpuPhyIds, lwswitch, portsPerSwitch, nodeId, switchPhyId, portId, val)
                else:
                    setTrunkPort(topologyInfo, allLinks, lwswitch, portsPerSwitch, nodeId, switchPhyId, portId, val)


    fabric = topology_pb2.fabric()
    fabric.version = fabricVersion

    first = True
    for nodeId in nodeIds:
        fabricNode = fabric.fabricNode.add()
        fabricNode.version = fabricVersion
        #set IP address
        if len(nodeIds) > 1:
            if first == True:
                #fabricNode.IPAddress = "192.168.122.245"
                fabricNode.IPAddress = "192.168.254.85"
                first = False
            else:
                #fabricNode.IPAddress = "192.168.122.40"
                fabricNode.IPAddress = "192.168.254.119"
                
        setGpuinfo(topologyInfo, fabricNode, nodeId, gpuPhyIds)

        setSwitchInfo(topologyInfo, gpuPhyIds, fabricNode, portsPerSwitch, nodeId)

        #Add all valid partitions
        setPartitionInfo(fabricNode, True, 0x10, 0xc, 0x30, 0x0)
        setPartitionInfo(fabricNode, True, 0x8, 0x6, 0x0, 0x0)

        setPartitionInfo(fabricNode, False, 0x10, 0xc, 0x30, 0x0)
        setPartitionInfo(fabricNode, False, 0x8, 0x6, 0x0, 0x0)
        setPartitionInfo(fabricNode, False, 0x4, 0x3, 0x0, 0x0)
        setPartitionInfo(fabricNode, False, 0x2, 0x1, 0x0, 0x0)
        setPartitionInfo(fabricNode, False, 0x1, 0x0, 0x0, 0x0)
    return fabric

def main():
    #read in the topoloy CSV file
    print_verbose("Reading data: Reading CSV file describing link between all ports") 
    df = pd.read_csv(args.csv_file, delim_whitespace=True)

    print_verbose("Pre-processing: Creating a dictionary of all hardware links") 
    allLinks = getAllLinks(df)

    print_verbose("Pre-processing: Creating a list of all switch ports") 
    switchPorts = getAllSwitchPorts(allLinks)

    print_verbose("Pre-processing: Creating a list of all gpu ports")
    gpuPorts =  getAllGpuPorts(allLinks)

    print_verbose("Pre-processing: Creating a sorted list of (nodeID, phyID) tuples for GPUs")
    gpuPhyIds = getAllGpuPhyIds(gpuPorts)

    print_verbose("Pre-processing: Creating a sorted list of (nodeID, phyID) tuples for switches")
    switchPhyIds = getAllSwitchPhyIds(switchPorts)

    print_verbose("Pre-processing: Creating a sorted list of nodeIDs")
    nodeIds = getAllNodeIds(allLinks)

    print_verbose("Pre-processing: Creating dict of two per-switch-port lists 1. access ports 2. trunk ports. Trunk ports have link to access port at corresponding index") 
    portsPerSwitch = getPortsPerSwitch(allLinks, switchPorts)

    print_verbose("Building graph: Adding bi-directional edges in graph for all links")
    G = buildGraph(allLinks, switchPorts, portsPerSwitch)

    print_verbose("Finding routes: Finding shortest paths between all pairs of GPU ports")
    allPaths = findRoutes(G, gpuPorts)

    print_verbose("Building topology info: Creating a data structure that has all the data to be used when creating the protobuf message later")
    topologyInfo = buildTopologyInfo(allPaths, nodeIds, gpuPhyIds, switchPhyIds, portsPerSwitch, switchPorts)

    print_verbose("Builing topology: Creating the protobuf message")
    fabric = buildTopologyProtobuf(topologyInfo, nodeIds, allLinks, portsPerSwitch, gpuPhyIds)

    if args.text_file != None:
        with open (args.text_file, "w") as f:
            f.write(text_format.MessageToString(fabric))
            f.write('name: "LR_CONFIG"')
            f.write('time: {}'.format(strftime("%a %b %d %H:%M:%S %Y", gmtime())))
    if args.write_to_stdout:
        print (fabric)
        print ('name: "LR_CONFIG"')
        print ('time: {}'.format(strftime("%a %b %d %H:%M:%S %Y", gmtime())))

    if args.binary_file != None:
        with open (args.binary_file, "w") as f:
            f.write(fabric.SerializeToString())


if __name__== "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', "--csv",  action='store',
                        dest='csv_file', required=True,
                        help='Input CSV file for hardware connections')
    parser.add_argument('-v', "--verbose",  action='store_true', default=False,
                        dest='verbose',
                        help='Verbose info messages')
    parser.add_argument('-t', "--text",  action='store',
                        dest='text_file',
                        help='Print text file for topology')
    parser.add_argument('-b', "--binary",  action='store',
                        dest='binary_file',
                        help='Print binary file for topology')
    parser.add_argument('-l', "--loopback",  action='store_true', default=False,
                        dest='loopback',
                        help='Access port loopback to GPU')
    parser.add_argument('--stdout',  action='store_true', default=False,
                        dest='write_to_stdout',
                        help='Write topology file text to stdout')
    parser.add_argument('-p', "--paths",  action='store',
                    dest='paths_file',
                    help='Input file specifying paths to use instead of callwlating paths')

    args = parser.parse_args()

    main()
    exit (0)
