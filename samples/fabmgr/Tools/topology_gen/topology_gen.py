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
from google.protobuf.json_format import MessageToJson
import json


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

def setValidPorts(route, ports, even=True, trunkPort=False, isFlipNode=False, portsPerSwitch=None, nodeId=None, switchPhyId=None):
    route.vcModeValid7_0 = 0
    route.vcModeValid15_8 = 0
    route.vcModeValid17_16 = 0
    validAndVc = 1
    ports = set(ports)
    num_ports = len(ports)
    num_ports_set = 0
    for port in ports:
        #for dedicating sparate paths to req/rsp 
        if (num_ports > 1) and ((even==True and ((port % 2) != 0)) or (even==False and ((port%2)==0))) :
            #continue
            pass
        num_ports_set += 1
        if args.ring_dateline != None and trunkPort == True:
            if isAccessPort(portsPerSwitch, (nodeId ,switchPhyId , port)):
                validAndVc = 5
            elif isFlipNode:
                #print "Flip"
                validAndVc = 3
            else:
                #print "Keep"
                validAndVc = 1
                
        if port >= 0 and port < 8:
            route.vcModeValid7_0 = route.vcModeValid7_0 | (validAndVc << (port * 4))
        elif port >= 8 and port < 16:
            route.vcModeValid15_8 |= (validAndVc << ((port - 8) * 4))
        elif port == 16 or port == 17:
            route.vcModeValid17_16 |= (validAndVc << ((port - 16) * 4))
        else:
            print "Invalid port number {}".format(port)
            exit(0)

def numTrunkPorts(portsPerSwitch, switchNodeId, switchPhyId):
    return len(portsPerSwitch[(switchNodeId, switchPhyId)][1])

def numAccessPorts(portsPerSwitch, switchNodeId, switchPhyId):
    return len(portsPerSwitch[(switchNodeId, switchPhyId)][0])

#check if a trunk port should be connected to an access port
"""
def isAccessPortConnected(portsPerSwitch, switchNodeId, switchPhyId, trunkPortId, accessPortId):
    accessPortIndex = portsPerSwitch[(switchNodeId, switchPhyId)][0].index(accessPortId)
    trunkPortIndex = portsPerSwitch[(switchNodeId, switchPhyId)][1].index(trunkPortId)
    if (accessPortIndex % numTrunkPorts(portsPerSwitch, switchNodeId, switchPhyId)) == trunkPortIndex:
        return True
    else:
        return False
"""

def isTrunkPortConnected(portsPerSwitch, switchNodeId, switchPhyId, accessPortId, trunkPortId):
    accessPortIndex = portsPerSwitch[(switchNodeId, switchPhyId)][0].index(accessPortId)
    trunkPortIndex = portsPerSwitch[(switchNodeId, switchPhyId)][1].index(trunkPortId)
    if (accessPortIndex % numTrunkPorts(portsPerSwitch, switchNodeId, switchPhyId)) == trunkPortIndex:
        return True
    else:
        return False


def isTrunkPort(portsPerSwitch, p):
    if getPortId(p) in portsPerSwitch[(getNodeId(p), getPhyId(p))][1]:
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
        #node Id default to 0 if not present in data
        nodeId = 0
        nodeIdFar =0 
        if 'nodeId' in df.columns:
            nodeId = row['nodeId']
            nodeIdFar = row['nodeIdFar']

        #TODO:When the CSV file is from lwlink-train tool it has no way of providing the Phy Ids for GPUs. 
        #So we use BusIds instead. This is a temporary solution 
        phyId = row["phyId"]
        if phyId == -1:
            phyId = getBusIdFromBdf(row["(d::b:d.f)"])

        phyIdFar = row["phyIdFar"]
        if phyIdFar == -1:
            phyIdFar = getBusIdFromBdf(row["(d::b:d.f)Far"])
        allLinks[(nodeId , phyId, row["linkIndex"], row["devType"])] = (nodeIdFar, phyIdFar, row["linkIndexFar"], row["devTypeFar"])
        allLinks[(nodeIdFar, phyIdFar, row["linkIndexFar"], row["devTypeFar"])] = (nodeId , phyId, row["linkIndex"], row["devType"])
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
            """
            if getNodeId(switchPorts[i]) == getNodeId(switchPorts[j]) and getPhyId(switchPorts[i]) == getPhyId(switchPorts[j]):
                #Check if switchPorts[i] is an access port. Connect an access port to all other ports(uni-directional)
                if isAccessPort(portsPerSwitch, switchPorts[i]):
                    G.add_edge(switchPorts[i], switchPorts[j])
                #If we get here it means switchPorts[i] is a trunk port. connect a trunk port to exactly one access port
                elif isAccessPort(portsPerSwitch, switchPorts[j]):
                    #a trunk port is assigned to a gpu..this remains same accross all switchPorts.
                    if args.spray or isAccessPortConnected(portsPerSwitch, getNodeId(switchPorts[i]), getPhyId(switchPorts[i]), fromPortId, toPortId):
                        G.add_edge(switchPorts[i], switchPorts[j])
            """
            if getNodeId(switchPorts[i]) == getNodeId(switchPorts[j]) and getPhyId(switchPorts[i]) == getPhyId(switchPorts[j]):
                #Check if switchPorts[i] is a trunk port. Connect a trunk port to all other access ports(uni-directional)
                if isTrunkPort(portsPerSwitch, switchPorts[i]) and ((args.spray and args.ring_dateline != None)or isAccessPort(portsPerSwitch, switchPorts[j])):
                    G.add_edge(switchPorts[i], switchPorts[j])
                elif isAccessPort(portsPerSwitch, switchPorts[i]):
                #If we get here it means switchPorts[i] is an access port. connect an access port to exactly one trunk port
                    if isTrunkPort(portsPerSwitch, switchPorts[j]) and (args.spray or isTrunkPortConnected(portsPerSwitch, getNodeId(switchPorts[i]), getPhyId(switchPorts[i]), fromPortId, toPortId)):
                    #a trunk port is assigned to a gpu..this remains same accross all switchPorts.
                        G.add_edge(switchPorts[i], switchPorts[j])
                    elif isAccessPort(portsPerSwitch, switchPorts[j]) and args.external_loopback == False:
                        #an access port is connected to other access ports within a switch only if we are not doing external loopback
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
            if l != r and (args.match_port == False or getPortId(l) == getPortId(r)):
                if nx.has_path(G, l, r):
                    shortest_paths =  [_ for _ in nx.all_shortest_paths(G, l, r)]
                    allPaths[(l, r)] = shortest_paths
                    found_path = found_path + 1
    if args.verbose:
        for path, val in allPaths.iteritems():
            print "path", 
            print "{:0>8d}".format((path[0][1] * 1000000 + path[0][2] * 10000 + path[1][1] * 100 + path[1][2])), 
            print "[ (", path[0][0], path[0][1], path[0][2], ") to ", "(", path[1][0], path[1][1], path[1][2], ") ] = ",  
            for n in val:
                for l in n:
                    #print "(", l[0], l[1], l[2], ")", 
                    print "(", l[0], ")", 
                print "]", len(n)
    return allPaths

def buildTopologyInfo(allPaths, nodeIds, gpuPhyIds, switchPhyIds, portsPerSwitch, switchPorts):
    def buildRequestInfo(topologyInfo, gpuPhyIds, dest, portsPerSwitch, path, hopNum):
        #entryNum = if each entry 16 GB is size and a GPU can have 64GB mem then entryNum = 0..3
        for entryNum in range(reqEntriesPerGpu):
            index = ((gpuBaseAddress(getNodeId(dest), getPhyId(dest), gpuPhyIds) + (entryNum * reqEntrySize)) / reqEntrySize)

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
                if args.external_loopback == True and isSwitchPort(path[hopNum + 1]):
                    buildRequestInfo(topologyInfo, gpuPhyIds, dest, portsPerSwitch, path, hopNum + 1)
                    buildResponseInfo(topologyInfo, gpuPhyIds, dest, path, hopNum + 1)

    return topologyInfo

def buildTopologyProtobuf(topologyInfo, nodeIds, allLinks, portsPerSwitch, gpuPhyIds, allPaths):
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
            setValidPorts(reqRte, reqEntry["port"], even=True)

        #iterate over all response entries for this port
        inorder = [(index, rspEntry) for index, rspEntry in val["responseEntries"].iteritems()]
        inorder.sort()
        for index, rspEntry in inorder:
            rspRte = access.rspRte.add()
            rspRte.version = fabricVersion
            rspRte.routePolicy = 0
            rspRte.entryValid = 1
            rspRte.index = index
            setValidPorts(rspRte, rspEntry["port"], even=False)

    def setTrunkPort(topologyInfo, allLinks, lwswitch, portsPerSwitch, nodeId, switchPhyId, portId, val):
        trunk = lwswitch.trunk.add() 
        trunk.version = fabricVersion
        trunk.connectType = topology_pb2.TRUNK_SWITCH_CONNECT
        trunk.localPortNum = portId
        trunk.farNodeID, trunk.farSwitchID, trunk.farPortNum, _ = getConnectedPort(allLinks, (nodeId, switchPhyId, portId, 0))

        trunk.config.version = fabricVersion
        trunk.config.type = topology_pb2.TRUNK_PORT_SWITCH
        trunk.config.phyMode = topology_pb2.DC_COUPLED
        if args.ring_dateline != None:
            trunk.config.enableVCSet1 = 1

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
            if nodeId == int(args.ring_dateline):
                #print "dateline=", nodeId
                setValidPorts(reqRte, reqEntry["port"], even=True, trunkPort = True, isFlipNode=True, portsPerSwitch = portsPerSwitch, nodeId = nodeId, switchPhyId = switchPhyId)
            else:
                setValidPorts(reqRte, reqEntry["port"], even=True, trunkPort = True, isFlipNode=False, portsPerSwitch = portsPerSwitch, nodeId = nodeId, switchPhyId = switchPhyId)


            #setValidPorts(reqRte, reqEntry["port"])

        #iterate over all response entries for this port
        inorder = [(index, rspEntry) for index, rspEntry in val["responseEntries"].iteritems()]
        inorder.sort()
        for index, rspEntry in inorder:
            rspRte = trunk.rspRte.add()
            rspRte.version = fabricVersion
            rspRte.routePolicy = 0
            rspRte.entryValid = 1
            rspRte.index = index
            if nodeId == int(args.ring_dateline):
                setValidPorts(rspRte, rspEntry["port"], even=False, trunkPort = True, isFlipNode=True, portsPerSwitch = portsPerSwitch, nodeId = nodeId, switchPhyId = switchPhyId )
            else:
                setValidPorts(rspRte, rspEntry["port"], even=False, trunkPort = True, isFlipNode=False, portsPerSwitch = portsPerSwitch, nodeId = nodeId, switchPhyId = switchPhyId )


            #setValidPorts(rspRte, rspEntry["port"])


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
            fabricNode.IPAddress = "192.168.254."
            node_ip = int(nodeId) + 1000
            fabricNode.IPAddress =  fabricNode.IPAddress + str(node_ip)
            fabricNode.nodeId = int(nodeId)
                
        setGpuinfo(topologyInfo, fabricNode, nodeId, gpuPhyIds)

        setSwitchInfo(topologyInfo, gpuPhyIds, fabricNode, portsPerSwitch, nodeId)

        #Add all valid partitions . TODO: hard-coded for now. Neew to get input as a file and add command line option
        #setPartitionInfo(fabricNode, True, 0x10, 0xc, 0x30, 0x0)
        #setPartitionInfo(fabricNode, True, 0x8, 0x6, 0x0, 0x0)
        #setPartitionInfo(fabricNode, True, 0x8, 0x6, 0x0, 0x30)
        #setPartitionInfo(fabricNode, True, 0x4, 0x3, 0x0, 0x0)

        #setPartitionInfo(fabricNode, False, 0x10, 0xc, 0x30, 0x0)
        #setPartitionInfo(fabricNode, False, 0x8, 0x6, 0x0, 0x0)
        #setPartitionInfo(fabricNode, False, 0x4, 0x3, 0x0, 0x0)
        #setPartitionInfo(fabricNode, False, 0x2, 0x1, 0x0, 0x0)
        #setPartitionInfo(fabricNode, False, 0x1, 0x0, 0x0, 0x0)
        makePartitions(allPaths, fabricNode)

    if args.topology_name != None:
        fabric.name = args.topology_name
    else:
        fabric.name = "DGX2_CONFIG"

    fabric.time = strftime("%a %b %d %H:%M:%S %Y", gmtime())
    return fabric

def main():
    #read in the topoloy CSV file
    print_verbose("Reading data: Reading CSV file describing link between all ports") 
    df = pd.read_csv(args.csv_file, delim_whitespace=True)

    print_verbose("Pre-processing: Creating a dictionary of all hardware links") 
    allLinks = getAllLinks(df)
    #print allLinks

    print_verbose("Pre-processing: Creating a list of all switch ports") 
    switchPorts = getAllSwitchPorts(allLinks)
    #print "len(switchPorts)=", len(switchPorts),  "switchPorts=", switchPorts

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
    #for path, val in allPaths.iteritems():
        #print path, val
    #print "allPaths=", allPaths
    #print "len(allPaths)=", len(allPaths)

    print_verbose("Building topology info: Creating a data structure that has all the data to be used when creating the protobuf message later")
    topologyInfo = buildTopologyInfo(allPaths, nodeIds, gpuPhyIds, switchPhyIds, portsPerSwitch, switchPorts)

    print_verbose("Builing topology: Creating the protobuf message")
    fabric = buildTopologyProtobuf(topologyInfo, nodeIds, allLinks, portsPerSwitch, gpuPhyIds, allPaths)

    if args.text_file != None:
        with open (args.text_file, "w") as f:
            f.write(text_format.MessageToString(fabric))

    if args.json_output_file != None:
        with open (args.json_output_file, "w") as f:
            #f.write(MessageToJson(fabric))
            json.dump(topologyInfo, f)

    if args.write_to_stdout:
        print (fabric)

    if args.binary_file != None:
        with open (args.binary_file, "w") as f:
            f.write(fabric.SerializeToString())
    
    #makePartitions(allPaths)


#the entire path has to be in partition
def isPathInPartition(path, gpuList, switchList):
    for n in range(0, len(path)):
        if isSwitchPort( path[n] ) and (getPhyId( path[n] ) not in switchList):
            return False
        if isGpuPort( path[n]) and ( getPhyId( path[n] ) not in gpuList):
            return False
    return True

def getBitMask(validBits):
    bitMask = 0
    for bit in validBits:
        bitMask |= (1 << bit)
    return bitMask

PART_TYPE_BAREMETAL = 0
PART_TYPE_VIRTUAL = 1
PART_TYPE_SHARED = 2

PART_ID = 0
PART_TYPE = 1
GPU_LIST = 2
SWITCH_LIST = 3


def makePartitions(allPaths, fabricNode):
    partition_info = {}
    if args.partitions_file != None:
        with open(args.partitions_file, 'r') as f:
            partition_info = ast.literal_eval(f.read())
    else:
        return

    for parts in partition_info:
        if parts[PART_TYPE] == PART_TYPE_BAREMETAL:
            partition = fabricNode.partitionInfo.bareMetalInfo.add()
        elif parts[PART_TYPE] == PART_TYPE_VIRTUAL:
            partition = fabricNode.partitionInfo.ptVirtualInfo.add()
        else:
            partition = fabricNode.partitionInfo.sharedLWSwitchInfo.add()

        #print "-------------------------------------------------------"
        #print "gpu list = ", parts[GPU_LIST]
        #print "switch list = ", parts[SWITCH_LIST]
        #print "gpuCount = ", len(parts[GPU_LIST])
        #print "switchCount = ", len(parts[SWITCH_LIST])
        partition.metaData.gpuCount = len(parts[GPU_LIST])
        partition.metaData.switchCount = len(parts[SWITCH_LIST])

        if len(parts[SWITCH_LIST]) == 0:
            partition.metaData.lwLinkIntraTrunkConnCount = 0
            partition.metaData.lwLinkInterTrunkConnCount = 0
            if parts[PART_TYPE] == PART_TYPE_SHARED:
                for gpu in parts[GPU_LIST]:
                    gpuInfo = partition.gpuInfo.add()
                    gpuInfo.physicalId = gpu
                    gpuInfo.numEnabledLinks = 0
                    gpuInfo.enabledLinkMask = 0

            continue

        #print "partition type = ", parts[PART_TYPE]
        if parts[PART_TYPE] == PART_TYPE_SHARED:
            #print "partition id = ", parts[PART_ID]
            partition.partitionId = parts[PART_ID]

        intraTrunkConns = set()
        interTrunkConns = set()
        partPaths = dict()

        for key, val in allPaths.iteritems():
            #doesn't handle multiple paths yet
            if isPathInPartition(val[0], parts[GPU_LIST], parts[SWITCH_LIST]):
                partPaths[key] = val

        validGpuPorts = dict()
        for g in parts[GPU_LIST]:
            validGpuPorts[g] = set()

        validSwitchPorts = dict()
        for s in parts[SWITCH_LIST]:
            validSwitchPorts[s] = set()

        for _ , paths in partPaths.iteritems():
            #print paths
            #continue
            for path in paths:
                for hop in range(0, len(path)):
                    phyId = getPhyId( path[hop] )
                    if isGpuPort( path[hop]):
                        val = validGpuPorts[ phyId ]
                        val.add(getPortId( path[hop]))
                        validGpuPorts[ phyId ] = val
                    else:
                        val = validSwitchPorts[ phyId ]
                        val.add(getPortId( path[hop]))
                        validSwitchPorts[ phyId ] = val



                for hop in range(0, len(path) - 1):
                    if getNodeId( path[ hop ] ) == getNodeId( path[ hop + 1 ] ):
                        if isSwitchPort (path[ hop ]) and isSwitchPort (path[ hop + 1]) and (getPhyId(path[ hop ]) != getPhyId(path[ hop + 1])):
                            intraTrunkConns.add( (path[ hop ], path[ hop + 1]) )
                    else:
                        #print "multi-node not fully handles yet"
                        exit(1)
                        if isSwitchPort (path[ hop ]) and isSwitchPort (path[ hop + 1]):
                            interTrunkConns.add( (path[ hop ], path[ hop + 1]) )

        lwLinkIntraTrunkConnCount = len (intraTrunkConns) / 2
        lwLinkInterTrunkConnCount = len (interTrunkConns) / 2
        #print "lwLinkIntraTrunkConnCount = ", lwLinkIntraTrunkConnCount
        #print "lwLinkInterTrunkConnCount = ", lwLinkInterTrunkConnCount
        partition.metaData.lwLinkIntraTrunkConnCount = lwLinkIntraTrunkConnCount
        partition.metaData.lwLinkInterTrunkConnCount = lwLinkInterTrunkConnCount


        if parts[PART_TYPE] == PART_TYPE_SHARED:
            for gpu in parts[GPU_LIST]:
                #print "gpu id =", gpu, "numEnabledLinks = ", len(validGpuPorts[gpu]) ,
                #print "bitmask = ", hex(getBitMask(validGpuPorts[gpu]))
                #print "validGpuPorts = ", validGpuPorts[gpu]
                gpuInfo = partition.gpuInfo.add()
                gpuInfo.physicalId = gpu
                gpuInfo.numEnabledLinks = len(validGpuPorts[gpu])
                gpuInfo.enabledLinkMask = getBitMask(validGpuPorts[gpu])


            for switch in parts[SWITCH_LIST]:
                #print "switch id =", switch, "validSwitchPorts = ", "numEnabledLinks = ", len(validSwitchPorts[switch]) , 
                #print "bitmask = ", hex(getBitMask(validSwitchPorts[switch]))
                switchInfo = partition.switchInfo.add()
                switchInfo.physicalId = switch
                switchInfo.numEnabledLinks = len(validSwitchPorts[switch])
                switchInfo.enabledLinkMask = getBitMask(validSwitchPorts[switch])
                #validSwitchPorts[switch]



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
    parser.add_argument('-n', "--topology-name",  action='store',
                        dest='topology_name',
                        help='Print text file for topology')
    parser.add_argument('-b', "--binary",  action='store',
                        dest='binary_file',
                        help='Print binary file for topology')
    parser.add_argument('-s', "--spray",  action='store_true', default=False,
                        dest='spray',
                        help='Spray over trunk links')
    parser.add_argument('-r', "--ring",  action='store', default=False,
                        dest='ring_dateline',
                        help='Specify dateline for ring')
    parser.add_argument('--stdout',  action='store_true', default=False,
                        dest='write_to_stdout',
                        help='Write topology file text to stdout')
    parser.add_argument('-p', "--paths",  action='store',
                        dest='paths_file',
                        help='Input file specifying paths to use instead of callwlating paths')
    parser.add_argument('-m', "--match-ports",  action='store_true', default=False,
                        dest='match_port',
                        help='Src port of route should always be same as the dest port')
    parser.add_argument('-e', "--external-loopback",  action='store_true', default=False,
                        dest='external_loopback',
                        help='The paths need to go though external loopback port')
    parser.add_argument('-j', "--json-output",  action='store',
                        dest='json_output_file',
                        help='Print JSON file for topology')
    parser.add_argument("--partition-file", action='store',
                        dest='partitions_file',
                        help='File specifying the partitions')

    args = parser.parse_args()

    main()
    exit (0)
