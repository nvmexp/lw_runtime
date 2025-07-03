#!/usr/bin/python
import pandas as pd
import networkx as nx
import topology_pb2
import copy
import datetime
from time import gmtime, strftime
import argparse
from google.protobuf import text_format
import ast


#for each hardware  The following valuses need to be provided
#GPU arch specific
memoryPerGpu = (64 * 1024 * 1024 * 1024)
reqEntrySize = (16 *1024 * 1024 * 1024)
memoryPerGpuReal = (32 *1024 * 1024 * 1024)
portsPerGpu = 6
portsPerSwitch = 18
fabricVersion = 0x100

#callwlated values
reqEntriesPerGpu = (memoryPerGpu / reqEntrySize)

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

def generateRlid(nodeId, gpuPhyId, gpuPortId, gpuPhyIds):
    return gpuPhyIds.index((nodeId, gpuPhyId)) * portsPerGpu + gpuPortId

def print_verbose(s = "\n"):
    if args.verbose:
        print s
    return 

def setValidPorts(route, ports):
    route.vcModeValid7_0 = 0
    route.vcModeValid15_8 = 0
    route.vcModeValid17_16 = 0
    for port in ports:
        if port >= 0 and port < 8:
            route.vcModeValid7_0 = route.vcModeValid7_0 | (1 << (port * 4))
        elif port >= 8 and port < 16:
            route.vcModeValid15_8 |= (1 << ((port - 8) * 4))
        elif port == 16 or port == 17:
            route.vcModeValid17_16 |= (1 << ((port - 16) * 4))
        else:
            print "Invalid port number {}".format(port)
            exit(0)

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
    switchPhyIds = list(set([ (getNodeId(_), getPhyId(_)) for _ in switchPorts]))
    switchPhyIds.sort()
    return switchPhyIds

def getAllNodeIds(allLinks):
    nodeIds = list(set([getNodeId(_) for _ in allLinks]))
    nodeIds.sort()
    return nodeIds

def getPortsPerSwitch(allLinks, switchPorts):
    portsPerSwitch = dict()

    for i in switchPorts:
        if (getNodeId(i), getPhyId(i)) in portsPerSwitch:
            accessPorts = portsPerSwitch[(getNodeId(i), getPhyId(i))][0]
            trunkPorts = portsPerSwitch[(getNodeId(i), getPhyId(i))][1]
        else:
            accessPorts = []
            trunkPorts = []

        if isSwitchPort(getConnectedPort(allLinks, i)):
            #i am trunk port
            trunkPorts.append(getPortId(i))
        else:
            #i am access port 
            accessPorts.append(getPortId(i))
        portsPerSwitch[(getNodeId(i), getPhyId(i))] = (accessPorts, trunkPorts)
    return portsPerSwitch

def buildGraph(allLinks, switchPorts, portsPerSwitch):
    G = nx.DiGraph()
    for l, r in allLinks.iteritems():
        G.add_edge(l, r)

    print_verbose("Building graph: Interconnecting ports within a switch")
    for i in xrange(len(switchPorts)):
        for j in xrange(len(switchPorts)):
            if switchPorts[i] == switchPorts[j]:
                continue
            fromPortId = getPortId(switchPorts[i])
            toPortId = getPortId(switchPorts[j])
            if getNodeId(switchPorts[i]) == getNodeId(switchPorts[j]) and getPhyId(switchPorts[i]) == getPhyId(switchPorts[j]):
                #Check if switchPorts[i] is an access port. Connect an access port to all other ports(uni-directional)
                if isAccessPort(portsPerSwitch, switchPorts[i]):
                    G.add_edge(switchPorts[i], switchPorts[j])
                #If we get here it means switchPorts[i] is a trunk port. connect a trunk port to exactly one access port
                elif isAccessPort(portsPerSwitch, switchPorts[j]):
                    #a trunk port is assigned to a gpu..this remains same accross all switchPorts.
                    if args.spray or isAccessPortConnected(portsPerSwitch, getNodeId(switchPorts[i]), getPhyId(switchPorts[i]), fromPortId, toPortId):
                        G.add_edge(switchPorts[i], switchPorts[j])
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
            if l != r:
                if nx.has_path(G, l, r):
                    shortest_paths =  [_ for _ in nx.all_shortest_paths(G, l, r)]
                    allPaths[(l, r)] = shortest_paths
                    found_path = found_path + 1
    return allPaths

def buildTopologyInfo(allPaths, nodeIds, gpuPhyIds, switchPhyIds, portsPerSwitch, switchPorts):
    def buildRequestInfo(topologyInfo, gpuPhyIds, dest, portsPerSwitch, path, hopNum):
        #entryNum = if each entry 16 GB is size and a GPU can have 64GB mem then entryNum = 0..3
        for entryNum in range(reqEntriesPerGpu):
            index = ((gpuBaseAddress(getNodeId(dest), getPhyId(dest), gpuPhyIds) + (entryNum * reqEntrySize)) / reqEntrySize)
            if index > 64:
                print " 1 index=", index, "address=", reqEntry["address"], "destNodeId=", destNodeId, "destPhyId=", destPhyId
                exit(0)

            if index in topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["requestEntries"]:
                reqEntry = topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["requestEntries"][index]
            else:
                reqEntry = dict()
                reqEntry["port"] = []

            if isAccessPort(portsPerSwitch, path[hopNum + 1]):
                reqEntry["address"] = entryNum * reqEntrySize
            else:
                reqEntry["address"] = gpuBaseAddress(getNodeId(dest), getPhyId(dest), gpuPhyIds) + (entryNum * reqEntrySize)

            reqEntry["index"] = index
            reqEntry["port"].append(getPortId( path[ hopNum + 1]))
            topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["requestEntries"][index] = reqEntry

    def buildResponseInfo(topologyInfo, gpuPhyIds, dest, path, hopNum):
        rlidDest = generateRlid(getNodeId(dest), getPhyId(dest), getPortId(dest), gpuPhyIds)
        if rlidDest in topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["responseEntries"]:
            rspEntry = topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["responseEntries"] [rlidDest]
        else:
            rspEntry = dict()
            rspEntry["port"] = []
            rspEntry["index"] = rlidDest

        rspEntry["port"].append( getPortId( path[ hopNum  + 1]))
        topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["responseEntries"] [rlidDest] = rspEntry


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

    for nodeId, switchPhyId, portId,_ in switchPorts:
        topologyInfo[nodeId][switchPhyId][portId] = dict()
        topologyInfo[nodeId][switchPhyId][portId]["requestEntries"] = dict()
        topologyInfo[nodeId][switchPhyId][portId]["responseEntries"] = dict()

    for (src, dest), paths in allPaths.iteritems():
        for path in paths:
            for hopNum in xrange(1, len(path) - 2, 2):
                buildRequestInfo(topologyInfo, gpuPhyIds, dest, portsPerSwitch, path, hopNum)
                buildResponseInfo(topologyInfo, gpuPhyIds, dest, path, hopNum)
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
            gpu.physicalId = gpuPhyId

    def setAccessPort(topologyInfo, gpuPhyIds, lwswitch, portsPerSwitch, nodeId, switchPhyId, portId, val):
        access = lwswitch.access.add()
        access.version = fabricVersion
        access.connectType = topology_pb2.ACCESS_GPU_CONNECT
        access.localPortNum = portId
        access.farNodeID, access.farPeerID, access.farPortNum, _ = getConnectedPort(allLinks, (nodeId, switchPhyId, portId, 0))

        access.config.version = fabricVersion
        access.config.type = topology_pb2.ACCESS_PORT_GPU
        access.config.RequesterLinkID = generateRlid(access.farNodeID, access.farPeerID ,access.farPortNum, gpuPhyIds)
        access.config.phyMode = topology_pb2.DC_COUPLED

        #iterate over all request entries for this port
        inorder = [(index, reqEntry) for index, reqEntry in val["requestEntries"].iteritems()]
        inorder.sort()
        for index, reqEntry in inorder:
            reqRte = access.reqRte.add()
            reqRte.version = fabricVersion
            reqRte.routePolicy = 0
            reqRte.entryValid = 1
            reqRte.index = index
            reqRte.address = reqEntry["address"]
            setValidPorts(reqRte, reqEntry["port"])

        #iterate over all response entries for this port
        inorder = [(index, rspEntry) for index, rspEntry in val["responseEntries"].iteritems()]
        inorder.sort()
        for index, rspEntry in inorder:
            rspRte = access.rspRte.add()
            rspRte.version = fabricVersion
            rspRte.routePolicy = 0
            rspRte.entryValid = 1
            rspRte.index = index
            setValidPorts(rspRte, rspEntry["port"])

    def setTrunkPort(topologyInfo, allLinks, lwswitch, portsPerSwitch, nodeId, switchPhyId, portId, val):
        trunk = lwswitch.trunk.add() 
        trunk.version = fabricVersion
        trunk.connectType = topology_pb2.TRUNK_SWITCH_CONNECT
        trunk.localPortNum = portId
        trunk.farNodeID, trunk.farSwitchID, trunk.farPortNum, _ = getConnectedPort(allLinks, (nodeId, switchPhyId, portId, 0))

        trunk.config.version = fabricVersion
        trunk.config.type = topology_pb2.TRUNK_PORT_SWITCH
        trunk.config.phyMode = topology_pb2.DC_COUPLED

        #iterate over all request entries for this port
        inorder = [(index, reqEntry) for index, reqEntry in val["requestEntries"].iteritems()]
        inorder.sort()
        for index, reqEntry in inorder:
            reqRte = trunk.reqRte.add()
            reqRte.version = fabricVersion
            reqRte.routePolicy = 0
            reqRte.entryValid = 1
            reqRte.index = index
            reqRte.address = reqEntry["address"]
            setValidPorts(reqRte, reqEntry["port"])

        #iterate over all response entries for this port
        inorder = [(index, rspEntry) for index, rspEntry in val["responseEntries"].iteritems()]
        inorder.sort()
        for index, rspEntry in inorder:
            rspRte = trunk.rspRte.add()
            rspRte.version = fabricVersion
            rspRte.routePolicy = 0
            rspRte.entryValid = 1
            rspRte.index = index
            setValidPorts(rspRte, rspEntry["port"])

    def setSwitchInfo(topologyInfo, gpuPhyIds, fabricNode, portsPerSwitch, nodeId):
        for switchPhyId in topologyInfo[nodeId]["switchPhyIds"]:
            lwswitch = fabricNode.lwswitch.add()
            lwswitch.version = fabricVersion
            switchIndex = topologyInfo[nodeId]["switchPhyIds"].index(switchPhyId)
            lwswitch.ECID = "N" + str(nodeId) + "_S" + str(switchIndex)
            lwswitch.physicalId = switchPhyId
            #iterate over all ports in this switch
            for portId, val in topologyInfo[nodeId][switchPhyId].iteritems():
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

        #Add all valid partitions . TODO: hard-coded for now. Neew to get input as a file and add command line option
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
    #print "switchPorts=", switchPorts

    print_verbose("Pre-processing: Creating a list of all gpu ports")
    gpuPorts =  getAllGpuPorts(allLinks)
    #print "gpuPorts=", gpuPorts

    print_verbose("Pre-processing: Creating a sorted list of (nodeID, phyID) tuples for GPUs")
    gpuPhyIds = getAllGpuPhyIds(gpuPorts)
    #print "gpuPhyIds=", gpuPhyIds

    print_verbose("Pre-processing: Creating a sorted list of (nodeID, phyID) tuples for switches")
    switchPhyIds = getAllSwitchPhyIds(switchPorts)
    #print "switchPhyIds=", switchPhyIds

    print_verbose("Pre-processing: Creating a sorted list of nodeIDs")
    nodeIds = getAllNodeIds(allLinks)
    #print "nodeIds=", nodeIds

    print_verbose("Pre-processing: Creating dict of two per-switch-port lists 1. access ports 2. trunk ports. Trunk ports have link to access port at corresponding index") 
    portsPerSwitch = getPortsPerSwitch(allLinks, switchPorts)
    #print "portsPerSwitch=", portsPerSwitch

    print_verbose("Building graph: Adding bi-directional edges in graph for all links")
    G = buildGraph(allLinks, switchPorts, portsPerSwitch)

    print_verbose("Finding routes: Finding shortest paths between all pairs of GPU ports")
    allPaths = findRoutes(G, gpuPorts)
    #print "allPaths=", allPaths
    #print "len(allPaths)=", len(allPaths)

    print_verbose("Building topology info: Creating a data structure that has all the data to be used when creating the protobuf message later")
    topologyInfo = buildTopologyInfo(allPaths, nodeIds, gpuPhyIds, switchPhyIds, portsPerSwitch, switchPorts)

    print_verbose("Builing topology: Creating the protobuf message")
    fabric = buildTopologyProtobuf(topologyInfo, nodeIds, allLinks, portsPerSwitch, gpuPhyIds)

    if args.text_file != None:
        with open (args.text_file, "w") as f:
            f.write(text_format.MessageToString(fabric))
            f.write('name: "DGX2_CONFIG"')
            f.write('time: {}'.format(strftime("%a %b %d %H:%M:%S %Y", gmtime())))
    if args.write_to_stdout:
        print (fabric)
        print ('name: "DGX2_CONFIG"')
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
    parser.add_argument('-s', "--spray",  action='store_true', default=False,
                        dest='spray',
                        help='Spray over trunk links')
    parser.add_argument('--stdout',  action='store_true', default=False,
                        dest='write_to_stdout',
                        help='Write topology file text to stdout')
    parser.add_argument('-p', "--paths",  action='store',
                        dest='paths_file',
                        help='Input file specifying paths to use instead of callwlating paths')

    args = parser.parse_args()

    main()
    exit (0)
