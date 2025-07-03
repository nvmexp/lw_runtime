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
from google.protobuf.json_format import MessageToJson
import time
import multiprocessing as mp

#For each hardware  The following valuses need to be provided
#GPU arch specific

#Laguna supports a max of 2048 endpoints
DEFAULT_MODE_MAX_ENDPOINTS = 2048

memoryPerGpu = (512 * 1024 * 1024 * 1024)
memoryPerGpuReal = (512 *1024 * 1024 * 1024)
portsPerGpu = 18
fabricVersion = 0x100
flaIndexStart =  DEFAULT_MODE_MAX_ENDPOINTS
rmapEntryMaxSize = (512 *1024 * 1024 * 1024)

#First 64 entries are reserved for SPA
lagunaFirstFlaRemapSlot = 64

#Alternate entries are reserved for EGM
hopperEgmAddressRangeMultiplier = 2

#Laguna GPA Base Address
lagunaGpaBaseAddress = (lagunaFirstFlaRemapSlot + DEFAULT_MODE_MAX_ENDPOINTS * hopperEgmAddressRangeMultiplier) * rmapEntryMaxSize

usePhyId = False
isSnake = False
ring_dateline_switches = None
switchNodeList = None

# There is no need to implicitly connect a trunk port to another trunk port within the same switch
# except in a ring topology where packets can be routed from one trunk port to another within a switch.
enable_inter_trunk_spray = False
enable_trunk_to_all_access = True

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
    return [_ for _ in iter(allLinks.keys()) if isGpuPort(_)]

def getAllSwitchPorts(allLinks):
    return [_ for _ in iter(allLinks.keys()) if isSwitchPort(_)]

def getConnectedPort(allLinks, port):
    return allLinks[port]

def gpuBaseAddress(nodeId, phyId, gpuPhyIds):
    if args.match_target_phy_id == True:
        return lagunaGpaBaseAddress + (phyId * memoryPerGpu * hopperEgmAddressRangeMultiplier)
    else:
        return lagunaGpaBaseAddress + (gpuPhyIds.index((nodeId, phyId)) * memoryPerGpu * hopperEgmAddressRangeMultiplier)

    #gpuIndex = gpuPhyIds.index((nodeId, phyId))
    #return gpuIndex * memoryPerGpu

def gpuFlaBaseAddress(nodeId, phyId, gpuPhyIds, targetId):
    ##gpuIndex = gpuPhyIds.index((nodeId, phyId)) + flaIndexStart
    #gpuIndex = gpuPhyIds.index((nodeId, phyId))
    gpuIndex = targetId
    return (flaIndexStart + (gpuIndex * hopperEgmAddressRangeMultiplier * (memoryPerGpu/rmapEntryMaxSize))) * rmapEntryMaxSize

def gpuTargetId(nodeId, phyId, gpuPhyIds):
    if args.match_target_phy_id == True:
        return phyId
    else:
        return gpuPhyIds.index((nodeId, phyId))

def print_verbose(s = "\n"):
    if args.verbose:
        print(s)
    return 

def numTrunkPorts(portsPerSwitch, switchNodeId, switchPhyId):
    return len(portsPerSwitch[(switchNodeId, switchPhyId)][1])

def numAccessPorts(portsPerSwitch, switchNodeId, switchPhyId):
    return len(portsPerSwitch[(switchNodeId, switchPhyId)][0])

#check if a trunk port should be connected to an access port
def isAccessPortConnected(portsPerSwitch, switchNodeId, switchPhyId, trunkPortId, accessPortId):
    accessPortIndex = portsPerSwitch[(switchNodeId, switchPhyId)][0].index(accessPortId)
    trunkPortIndex = portsPerSwitch[(switchNodeId, switchPhyId)][1].index(trunkPortId)
    if (accessPortIndex % numTrunkPorts(portsPerSwitch, switchNodeId, switchPhyId)) == trunkPortIndex:
        return True
    else:
        return False

def getConnectedTrunkPort(portsPerSwitch, n):
    switchNodeId = getNodeId(n)
    switchPhyId = getPhyId(n)
    accessPortId = getPortId(n)
    accessPortIndex = portsPerSwitch[(switchNodeId, switchPhyId)][0].index(accessPortId)
    trunkPortId = portsPerSwitch[(switchNodeId, switchPhyId)][1][accessPortIndex]
    return (switchNodeId, switchPhyId, trunkPortId, 0)

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

# Get list of Nodes associated with Lwlink Rack switches
def getAllSwitchOnlyNodes():
    global switchNodeList
    global enable_trunk_to_all_access
    if args.switch_node_list != None:
        switchNodeList = []
        # for wolf based system without spray, disable spraying from trunk port to all access ports 
        if args.spray == False:
            enable_trunk_to_all_access = False
        for id in args.switch_node_list.split(","):
            switchNodeList.append(int(id))
    print_verbose ("switchNodeList = " + str(switchNodeList))


# Check whether a given node is Lwlink Rack Switch or not
def isLwlinkRackSwitch(nodeID):
    if switchNodeList != None and nodeID in switchNodeList:
        return True
    else:
        return False

def getAllLinks(df):
    allLinks = dict()
    gpuList = None

    if args.gpu_id_list != None:
        gpuList = []
        for id in args.gpu_id_list.split(","):
            gpuList.append(int(id))
    print_verbose("gpu list = " +  str(gpuList))

    switchList = None
    if args.switch_id_list != None:
        switchList = []
        for id in args.switch_id_list.split(","):
            switchList.append(int(id))
    print_verbose ("switch list = " + str(switchList))

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

        if gpuList != None:
            if row["devType"] == 1 and phyId not in gpuList:
                continue
            if row["devTypeFar"] == 1 and phyIdFar not in gpuList:
                continue

        if switchList != None:
            if row["devType"] == 0 and phyId not in switchList:
                continue
            if row["devTypeFar"] == 0 and phyIdFar not in switchList:
                continue

        allLinks[(nodeId , phyId, row["linkIndex"], row["devType"])] = (nodeIdFar, phyIdFar, row["linkIndexFar"], row["devTypeFar"])
        allLinks[(nodeIdFar, phyIdFar, row["linkIndexFar"], row["devTypeFar"])] = (nodeId , phyId, row["linkIndex"], row["devType"])
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
    for l, r in iter(allLinks.items()):
        G.add_edge(l, r)
    if isSnake == False:
        print_verbose("Building graph: Interconnecting ports within a switch")
        for i in range(len(switchPorts)):
            for j in range(len(switchPorts)):
                if switchPorts[i] == switchPorts[j]:
                    continue
                fromPortId = getPortId(switchPorts[i])
                toPortId = getPortId(switchPorts[j])
                #note that even ports are connected to even ports only and odd ports to odd ports only
                #In case of spray we might need to set RLANs and not bother about even/odd
                if args.match_even and (fromPortId % 2) != (toPortId % 2):
                    continue

                # Only interconnect if it is two ports on the same switch
                if getNodeId(switchPorts[i]) != getNodeId(switchPorts[j]) or getPhyId(switchPorts[i]) != getPhyId(switchPorts[j]):
                    continue

                #TODO if true doesn't handle external loopback correctly
                if args.unique_access_port:
                    #Check if switchPorts[i] is an access port. Connect an access port to all other ports(uni-directional)
                    if isAccessPort(portsPerSwitch, switchPorts[i]) and (isTrunkPort(portsPerSwitch, switchPorts[j]) or args.external_loopback == False or args.spray):
                        G.add_edge(switchPorts[i], switchPorts[j])
                    #If we get here it means switchPorts[i] is a trunk port. connect a trunk port to exactly one access port
                    elif isTrunkPort(portsPerSwitch, switchPorts[i]):
                        #a trunk port is assigned to a gpu..this remains same accross all switchPorts.
                        if isAccessPort(portsPerSwitch, switchPorts[j]) and (args.spray or isAccessPortConnected(portsPerSwitch, getNodeId(switchPorts[i]), getPhyId(switchPorts[i]), fromPortId, toPortId)):
                            G.add_edge(switchPorts[i], switchPorts[j])
                        elif isTrunkPort(portsPerSwitch, switchPorts[j]) and args.spray and (args.ring_dateline != None):
                            G.add_edge(switchPorts[i], switchPorts[j])
                        #elif isTrunkPort(portsPerSwitch, switchPorts[j]) and args.external_loopback == False:
                            #G.add_edge(switchPorts[i], switchPorts[j])
                        
                else:
                    #Check if switchPorts[i] is a trunk port. Connect a trunk port to all other access ports(uni-directional)
                    if isTrunkPort(portsPerSwitch, switchPorts[i]):
                            if isLwlinkRackSwitch( getNodeId(switchPorts[j])):
                                # on a switch a trunk port is connect to all other trunk ports
                                G.add_edge(switchPorts[i], switchPorts[j])
                                #print("adding edge between two trunk ports on node", switchPorts[i], switchPorts[j], getNodeId(switchPorts[j]))
                            else:
                                if isAccessPort(portsPerSwitch, switchPorts[j]) and (args.spray or enable_trunk_to_all_access or isTrunkPortConnected(portsPerSwitch, getNodeId(switchPorts[i]), getPhyId(switchPorts[i]), toPortId, fromPortId)):  
                                    # on a non-switch node a trunk port is connected to all access ports unless enable_trunk_to_all_access is set to False
                                    #print("adding trunk to access edge",  switchPorts[i], switchPorts[j])
                                    G.add_edge(switchPorts[i], switchPorts[j])
                                elif (args.spray and args.ring_dateline != None):
                                    # on a non-switch node a trunk port is connected to another trunk port only for ring topologies 
                                    G.add_edge(switchPorts[i], switchPorts[j])
                    elif isAccessPort(portsPerSwitch, switchPorts[i]):
                    #If we get here it means switchPorts[i] is an access port. connect an access port to exactly one trunk port
                        if isTrunkPort(portsPerSwitch, switchPorts[j]) and (args.spray or isTrunkPortConnected(portsPerSwitch, getNodeId(switchPorts[i]), getPhyId(switchPorts[i]), fromPortId, toPortId)):
                        #a trunk port is assigned to a gpu..this remains same accross all switchPorts.
                            G.add_edge(switchPorts[i], switchPorts[j])
                            #print("adding access to trunk edge",  switchPorts[i], switchPorts[j])
                        elif isAccessPort(portsPerSwitch, switchPorts[j]) and args.external_loopback == False:
                            #an access port is connected to other access ports within a switch only if we are not doing external loopback
                            G.add_edge(switchPorts[i], switchPorts[j])

    else:
        #TODO: Doesn't work yet.
        #snake ports by adding uni-derctional links between conselwtive ports within
        #a switch that are not already directly connected
        for nodeId, phyId in iter(portsPerSwitch.keys()):
            thisSwitchPorts = portsPerSwitch[(nodeId, phyId)][0] + portsPerSwitch[(nodeId, phyId)][1]
            thisSwitchPorts.sort()
            for i in range(len(thisSwitchPorts)):
                l = (nodeId, phyId, thisSwitchPorts[i])
                r = (nodeId, phyId, thisSwitchPorts[(i + 1) % len(thisSwitchPorts)])
                if allLinks[l] != r:
                    G.add_edge((nodeId, phyId, thisSwitchPorts[i]) , 
                                (nodeId, phyId, thisSwitchPorts[(i + 1) % len(thisSwitchPorts)]))
                    print ("adding uni-directional link" , l, r)
    return G

def addPaths(allPaths, l, r, paths, pathLenList):
    pathsToAdd = []
    for path in paths:
        if pathLenList == None or len(path) in pathLenList:
            pathsToAdd.append(path)
    if len(pathsToAdd) > 0:
        allPaths[(l, r)] = pathsToAdd

sentinel = None
def worker(G, inqueue, outqueue):
    result = []
    count = 0
    
    for l,r in iter(inqueue.get, sentinel):
        count += 1
        mylist = []
        try:
            iterator = nx.all_shortest_paths(G, l, r)
            mylist = [_ for _ in iterator]
        except:
            pass
        if mylist:
            outqueue.put(((l, r), mylist))
    outqueue.put(sentinel)

def findAllShortestPathsParallel(G, gpuPorts):
    allShortestPaths = dict()
    inqueue = mp.Queue()
    outqueue = mp.Queue()
    
    # create a queue of all input pairs
    for l in gpuPorts:
        for r in gpuPorts:
            if (getNodeId(l) != getNodeId(r) or getPhyId(l) != getPhyId(r)) and (args.match_gpu_port_num == False or getPortId(l) == getPortId(r)):
                inqueue.put((l , r))

    procs = [mp.Process(target = worker, args = (G, inqueue, outqueue))
             for i in range(mp.cpu_count())]
    for proc in procs:
        inqueue.put(sentinel)
    for proc in procs:
        proc.daemon = True
        proc.start()
    for proc in procs:
        for key, value in iter(outqueue.get, sentinel):
            allShortestPaths[key] = value
    for proc in procs:
        proc.join()
    return allShortestPaths

def findAllShortestPaths(G, gpuPorts):
    allShortestPaths = dict()
    for l in gpuPorts:
        for r in gpuPorts:
            if (getNodeId(l) != getNodeId(r) or getPhyId(l) != getPhyId(r)) and (args.match_gpu_port_num == False or getPortId(l) == getPortId(r)):
                mylist = []
                try:
                    iterator = nx.all_shortest_paths(G, l, r)
                    mylist = [_ for _ in iterator]
                except:
                    pass
                if mylist:
                    allShortestPaths[(l, r)] = mylist
    return allShortestPaths

def findRoutes(G, gpuPorts, allLinks, portsPerSwitch):
    #get paths from file instead of callwlating from graph
    if args.paths_file:
        with open(args.paths_file, 'r') as f:
            return ast.literal_eval(f.read())
    
    pathLenList = None

    if args.path_lens != None:
        pathLenList = []
        for id in args.path_lens.split(","):
            pathLenList.append(int(id))

    print_verbose("path len list = " +  str(pathLenList))
    #callwlate paths from graph
    allPaths = dict()
    found_path=0
    allShortestPaths = findAllShortestPathsParallel(G, gpuPorts)
    #allShortestPaths = findAllShortestPaths(G, gpuPorts)
    print_verbose("size of allShortestPaths=" + str(len(allShortestPaths)) )    
    for l in gpuPorts:
        for r in gpuPorts:
            if (getNodeId(l) != getNodeId(r) or getPhyId(l) != getPhyId(r)) and (args.match_gpu_port_num == False or getPortId(l) == getPortId(r)):
                if (l, r) in allShortestPaths:
                    shortest_paths =  allShortestPaths[(l, r)]
                    paths_to_insert = []
                    l_switch = allLinks[l]
                    r_switch = allLinks[r]
                    if args.external_loopback and (getNodeId(l_switch) == getNodeId(r_switch) and getPhyId(l_switch) == getPhyId(r_switch)):
                        for pathnum in xrange(0, len(shortest_paths)):
                            # second hop in path is potentially one end of a loopout/loopback link
                            loopback_trunk_port = shortest_paths[pathnum][2]
                            # only insert path in node if second hop in path is a loopout/loopback
                            if ((getNodeId(loopback_trunk_port) == getNodeId(allLinks[loopback_trunk_port])) and (getPhyId(loopback_trunk_port) == getPhyId(allLinks[loopback_trunk_port]))):
                                shortest_paths[pathnum].insert(3, allLinks[loopback_trunk_port])
                                paths_to_insert.append(shortest_paths[pathnum])
                    else:
                        paths_to_insert = shortest_paths
                    addPaths(allPaths, l , r, paths_to_insert, pathLenList)
                    found_path = found_path + len(shortest_paths)
            elif l == r:
                if args.loopback:
                    accessPort = [_ for _ in G.neighbors(l)][0]
                    loop = [l, accessPort, accessPort, l]
                    addPaths(allPaths, l , r, [loop], pathLenList)
                    found_path = found_path + 1
                if args.loopback_from_trunk:
                    accessPort = [_ for _ in G.neighbors(l)][0]
                    trunkPorts = []
                    if args.spray == True:
                        trunkPorts = portsPerSwitch[ accessPort ][1]
                    else:
                        trunkPorts = portsPerSwitch[(getNodeId( accessPort ), getPhyId(accessPort))][1]
                    
                    loopPaths = [] 
                    for trunkPort in trunkPorts:
                        loopPaths.append([l, accessPort, trunkPort, trunkPort, accessPort, r])
                        found_path = found_path + 1
                    addPaths(allPaths, l , r, loopPaths, pathLenList)

                if args.loopout_from_trunk:
                    accessPort = [_ for _ in G.neighbors(l)][0]
                    trunkPorts = []
                    if args.spray == True:
                        trunkPorts = [ (getNodeId(accessPort), getPhyId(accessPort), _ , 0) for _ in portsPerSwitch[(getNodeId(accessPort), getPhyId(accessPort))][1]]
                    else:
                        trunkPorts = [getConnectedTrunkPort(portsPerSwitch, accessPort) ]
                    loopPaths = [] 
                    for trunkPort in trunkPorts:
                        loopPaths.append([l, accessPort, trunkPort, allLinks[trunkPort], accessPort, r])
                        found_path = found_path + 1
                    addPaths(allPaths, l , r, loopPaths, pathLenList)

                if isSnake:
                    pass
    print ("Paths found between {} endpoints".format(len(allPaths)))
    if args.verbose:
        path_count = 0
        print_verbose ("printing " + str(len(allPaths)) + " paths")
        for path, val in iter(allPaths.items()):
            path_str = "path "
            path_str += "{:0>8d} ".format((path[0][1] * 1000000 + path[0][2] * 10000 + path[1][1] * 100 + path[1][2]))
            path_str += "[ ( {} {} {} ) to ( {} {} {} ) ] = {} ".format(path[0][0], path[0][1], path[0][2], path[1][0], path[1][1], path[1][2], len(val))
            for n in val:
                path_str += "["
                for l in n:
                    path_str += " ( {} {} {} ) ".format(l[0], l[1], l[2])
                path_str += "]"
            print_verbose(path_str)
            print_verbose("")
            path_count += len(val)
        print ("Total paths = {}".format(path_count))
    return allPaths

def readLwlinkToOsfpPortMapping():
    osfpPortMapping = {}
    switchPhyIdToSlotIdMapping = {}
    if args.port_map_info:
        with open(args.port_map_info, 'r') as f:
            print_verbose ("reading given OSPF port mapping file - " + args.port_map_info)
            mapping_info = ast.literal_eval(f.read())
            osfpPortMapping = mapping_info['lwlink_to_osfp_port_map']
            switchPhyIdToSlotIdMapping = mapping_info['switch_phy_id_slot_id_map']

    return osfpPortMapping, switchPhyIdToSlotIdMapping

def buildTopologyInfo(allPaths, nodeIds, gpuPhyIds, switchPhyIds, portsPerSwitch, switchPorts):
    def buildRmapPolicyInfo(topologyInfo, gpuPhyIds, dest, path, hopNum, isFlaOnly):
        gpaIndex = gpuTargetId(getNodeId(dest), getPhyId(dest), gpuPhyIds)

        #GPA entry
        if not isFlaOnly:
            extBRmapEntry = dict()
            extBRmapEntry["targetId"] = gpaIndex
            extBRmapEntry["index"] = gpaIndex * 2 
            extBRmapEntry["address"] = 0
            topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["extBRmapEntries"][gpaIndex *2] = extBRmapEntry

        #FLA entry
        rmapEntry = dict()
        flaIndex = (gpaIndex * 2) + flaIndexStart
        rmapEntry["targetId"] = gpaIndex
        rmapEntry["index"] = flaIndex
        rmapEntry["address"] = gpuFlaBaseAddress(getNodeId(dest), getPhyId(dest), gpuPhyIds, gpaIndex)
        topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["rmapEntries"][flaIndex] = rmapEntry

    def buildRidRouteInfo(topologyInfo, gpuPhyIds, dest, path, hopNum):
        #for forward direction of path and return path
        gpaIndex = gpuTargetId(getNodeId(dest), getPhyId(dest), gpuPhyIds)
        #print("topologyInfo=", topologyInfo)
        #print("path=", path)
        ridEntry = dict()
        #if ridEntry already exists add to it, otherwise construct a new ridEntry to attach at this index
        if gpaIndex in topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["ridEntries"]:
            ridEntry = topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["ridEntries"][gpaIndex]
        else:
            ridEntry["port"] = []
            ridEntry["index"] = gpaIndex
        #this port's RID entries should point to the port which is the next hop on this path
        ridEntry["port"].append(getPortId( path[ hopNum  + 1]))
        #multiple paths can go hopNum--->(hopNum + 1). Only one RID entry is needed 
        ridEntry["port"] = list(set(ridEntry["port"]))
        topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["ridEntries"][gpaIndex] = ridEntry

    def buildRlanRouteInfo(topologyInfo, gpuPhyIds, dest, path, hopNum):
        gpaIndex = gpuTargetId(getNodeId(dest), getPhyId(dest), gpuPhyIds)
        rlanEntry = dict()
        #if rlanEntry already exists add to it, otherwise construct a new rlanEntry to attach at this index
        if gpaIndex in topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["rlanEntries"]:
            rlanEntry = topologyInfo[getNodeId(path[ hopNum ])] [getPhyId(path[ hopNum ])] [getPortId(path[ hopNum ])] ["rlanEntries"][gpaIndex]
        else:
            rlanEntry["index"] = gpaIndex
        #this is used to check if we are on even or odd port hence if we should use rlan 0 or 1 
        #Note it doesn't matter here if setting hopNum or hopNum + 1 as both will either be even or odd
        #rlanEntry["port"].append(getPortId( path[ hopNum  + 1]))
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
        topologyInfo[nodeId][switchPhyId][portId]["extBRmapEntries"] = dict()
        topologyInfo[nodeId][switchPhyId][portId]["ridEntries"] = dict()
        topologyInfo[nodeId][switchPhyId][portId]["rlanEntries"] = dict()

    for (src, dest), paths in iter(allPaths.items()):
        #print "Walking a path, len=", len(paths[0]), "path=", paths[0]
        for path in paths:
            # In a N hop path the 1st/last are GPU ports. Starting from the second hop every alternate port is an ingress port.
            # we just set ingress port entries when walking  a path
            for hopNum in range(1, len(path) - 2, 2):
                #RMAP table is only valid on access ports
                if isAccessPort(portsPerSwitch, path[hopNum]):
                    #Check if this is a loopback route
                    if src == dest and args.loopback:
                        buildRmapPolicyInfo(topologyInfo, gpuPhyIds, dest, path, hopNum, True)
                    else:
                        #print (src, "--->", dest, path)
                        buildRmapPolicyInfo(topologyInfo, gpuPhyIds, dest, path, hopNum, False)
                buildRidRouteInfo(topologyInfo, gpuPhyIds, dest, path, hopNum)
                buildRlanRouteInfo(topologyInfo, gpuPhyIds, dest, path, hopNum)
    return topologyInfo

def buildTopologyProtobuf(topologyInfo, nodeIds, allLinks, portsPerSwitch, gpuPhyIds, allPaths, lwlinkToOsfpPortMapping, switchPhyIdToSlotIdMapping):
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
            ecidStr = "N" + str(nodeId) + "_G" + str(gpuIndex)
            gpu.ECID = ecidStr.encode('ASCII')
            gpu.fabricAddrBase = gpuBaseAddress(nodeId, gpuPhyId, gpuPhyIds)
            gpu.fabricAddrRange = memoryPerGpuReal
            gpu.GPABase = gpuBaseAddress(nodeId, gpuPhyId, gpuPhyIds)
            gpu.targetId = gpuTargetId(nodeId, gpuPhyId, gpuPhyIds)
            gpu.GPARange = memoryPerGpuReal
            gpu.FLABase = int(gpuFlaBaseAddress(nodeId, gpuPhyId, gpuPhyIds, gpu.targetId))
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

        if args.set_rlan == True:
            access.config.RlanID = access.farPortNum
        else:
            access.config.RlanID = 0

        access.config.maxTargetID = topology_pb2.MAX_512_TARGET_ID

        # GPA: iterate over all rmap entries for this port
        inorder = [(index, rmapEntry) for index, rmapEntry in iter(val["extBRmapEntries"].items())]
        inorder.sort()
        for index, rmapEntry in inorder:
            rmapRte = access.extBRmapPolicyTable.add()
            rmapRte.targetId = rmapEntry["targetId"]
            rmapRte.index = int(index)
            rmapRte.version = fabricVersion
            rmapRte.entryValid = 1
            rmapRte.address = int(rmapEntry["address"])
            rmapRte.reqContextChk = 0x0
            rmapRte.reqContextMask = 0x0
            rmapRte.reqContextRep = 0x0
            rmapRte.addressOffset = 0x0
            rmapRte.addressBase = 0x0
            rmapRte.addressLimit = 0x0
            #TODO limit addresses per entry
            #rmapRte.addressBase = rmapEntry["address"]
            #rmapRte.addressLimit = rmapEntryMaxSize
            rmapRte.remapFlags = 0x1
            rmapRte.irlSelect = 0x0
            rmapRte.p2rSwizEnable = 0x0
            rmapRte.mult2 = 0x0
            rmapRte.planeSelect = 0x0

        # FLA: iterate over all rmap entries for this port
        inorder = [(index, rmapEntry) for index, rmapEntry in iter(val["rmapEntries"].items())]
        inorder.sort()
        for index, rmapEntry in inorder:
            rmapRte = access.rmapPolicyTable.add()
            rmapRte.targetId = rmapEntry["targetId"]
            rmapRte.index = int(index)
            rmapRte.version = fabricVersion
            rmapRte.entryValid = 1
            rmapRte.address = int(rmapEntry["address"])
            rmapRte.reqContextChk = 0x0
            rmapRte.reqContextMask = 0x0
            rmapRte.reqContextRep = 0x0
            rmapRte.addressOffset = 0x0
            rmapRte.addressBase = 0x0
            rmapRte.addressLimit = 0x0
            #TODO limit addresses per entry
            #rmapRte.addressBase = rmapEntry["address"]
            #rmapRte.addressLimit = rmapEntryMaxSize
            rmapRte.remapFlags = 0x1
            rmapRte.irlSelect = 0x0
            rmapRte.p2rSwizEnable = 0x0
            rmapRte.mult2 = 0x0
            rmapRte.planeSelect = 0x0

        #iterate over all rid entries for this port
        inorderRid = [(index, ridEntry) for index, ridEntry in iter(val["ridEntries"].items())]
        inorderRid.sort()
        portNum = dict()
        pointsToTrunk = False
        for index, ridEntry in inorderRid:
            ridRte = access.ridRouteTable.add()
            ridRte.index = index
            ridRte.version = fabricVersion
            ridRte.valid = 1
            #setting rmod[6]=0 means don't use the RLAN table entry we still have to set the corresponding RLAN table entry to valid

            #if len(ridEntry["port"]) > 0:
                #print "ridEntry ports", index, len(ridEntry["port"]), ridEntry["port"]
            pointsToTrunk = False
            for p in ridEntry["port"]:
                port = ridRte.portList.add()
                port.portIndex = p
                if isTrunkPort(portsPerSwitch, (nodeId, switchPhyId, p, 0) ):
                    pointsToTrunk =True
                port.vcMap = 2 #force to 0
            portNum[index] = len(ridEntry["port"])

            if args.set_rlan == True and pointsToTrunk == False:
                ridRte.rMod = ( 0x1 << 6)
            else: 
                ridRte.rMod = 0x0
            #if args.set_rlan == True:
            #TODO
            if False:
                for p1 in range(portNum[index], 16):
                    port = ridRte.portList.add()
                    #TODO set as 63
                    #port.portIndex = ridEntry["port"][0]
                    port.portIndex = 63
                    port.vcMap = 2  #force to 0
            #point to an invalid port
            #TODO uncomment these three lines
            #port = ridRte.portList.add()
            #port.portIndex = 63
            #port.vcMap = 0
        #print "portList=", ridRte.portList
        
        #iterate over all rlan entries for this port
        inorderRlan = [(index, rlanEntry) for index, rlanEntry in iter(val["rlanEntries"].items())]
        inorderRlan.sort()
        #print "inorderRlan=", inorderRlan
        for index, rlanEntry in inorderRlan:
            rlanRte = access.rlanRouteTable.add()
            rlanRte.index = index
            rlanRte.version = fabricVersion
            rlanRte.valid = 1
            if args.set_rlan == True:
                #TODO hard-coded good vlan range as 0-12
                offset = inorderRlan.index((index, rlanEntry))
                #print inorderRid[offset][1]["port"], "------", len(inorderRid[offset][1]["port"])
                rlanToPortMap = dict()
                for pt in inorderRid[offset][1]["port"]:
                    #The path to each target ID is via either trunk links or access links but not both at the same time
                    #Just check the first port in the list of ports for the corresponding RID and determine access or trunk
                    if isAccessPort(portsPerSwitch, (nodeId, switchPhyId, pt, 0) ):
                        rlan = getPortId(getConnectedPort(allLinks, (nodeId, switchPhyId, pt, 0)))
                        rlanToPortMap[rlan] = pt
                        inorderRlan[offset][1]["pointToTrunk"] = False
                    else:
                        inorderRlan[offset][1]["pointToTrunk"] = True

                #print "rlanToPortMap=", rlanToPortMap
                for rlNum in range(0,portsPerGpu):
                    groupEntry = rlanRte.groupList.add()
                    if rlNum in rlanToPortMap:
                        groupEntry.groupSelect = inorderRid[offset][1]["port"].index( rlanToPortMap[rlNum] )
                        groupEntry.groupSize = 1
                    else:
                        #A path to an access port cannot be a spray as we need to get response back to the requesting port
                        #Note: RLAN is only used for responses not for requests
                        #Hence the RLAN points to exactly one port in the  
                        if inorderRlan[offset][1]["pointToTrunk"] == False:
                            groupEntry.groupSelect = portNum[index]
                            groupEntry.groupSize = 1
                        else:
                            groupEntry.groupSelect = 0
                            groupEntry.groupSize = portNum[index]
                #bad group 1 for all other vlans
                #print "portNum[index]=", portNum[index]
                #for bad_group in range(portNum[index],16):
                #TODO
                #for bad_group in range(12,16):
                """
                for bad_group in range(12,12):
                    groupEntry = rlanRte.groupList.add()
                    groupEntry.groupSelect = portNum[index]
                    #TODO fix this
                    #groupEntry.groupSelect =  0
                    groupEntry.groupSize = 0
                """

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
        if args.ring_dateline != None:
            trunk.config.enableVCSet1 = 1

        trunk.config.RlanID = 0
        trunk.config.maxTargetID = topology_pb2.MAX_512_TARGET_ID

        #iterate over all rid entries for this port
        inorderRid = [(index, ridEntry) for index, ridEntry in iter(val["ridEntries"].items())]
        inorderRid.sort()
        portNum = dict()
        for index, ridEntry in inorderRid:
            ridRte = trunk.ridRouteTable.add()
            ridRte.index = index
            ridRte.version = fabricVersion
            ridRte.valid = 1
            #we use fixed routing mode in which rmod[3:0] gives the entry to use in the portList. 
            #setting rmod[6]=0 means don't use the RLAN table entry we still have to set the corresponding RLAN table entry to valid
             #setting rmod[6]=0 means don't use the RLAN table entry we still have to set the corresponding RLAN table entry to valid
            if args.set_rlan == True:
                ridRte.rMod = ( 0x1 << 6)
            else: 
                ridRte.rMod = 0x0
            #if len(ridEntry["port"]) > 0:
                #print "ridEntry ports", index, len(ridEntry["port"]), ridEntry["port"]
            for p in ridEntry["port"]:
                port = ridRte.portList.add()
                port.portIndex = p
                #vcMap for ring
                if args.ring_dateline != None and (nodeId == int(args.ring_dateline)):
                    if ring_dateline_switches == None or (ring_dateline_switches != None and switch in ring_dateline_switches):
                        port.vcMap = 1  #flip
                    else:
                        port.vcMap = 0  #flip
                else:
                    port.vcMap = 0  #keep
                if isAccessPort(portsPerSwitch, (nodeId, switchPhyId, p,0) ):
                    port.vcMap = 2  #force to 0
            portNum[index] = len(ridEntry["port"])
            #point to an invalid port
            #TODO uncomment these three lines
            #port = ridRte.portList.add()
            #port.portIndex = 63
            #port.vcMap = 0

        #iterate over all rlan entries for this port
        inorderRlan = [(index, rlanEntry) for index, rlanEntry in iter(val["rlanEntries"].items())]
        inorderRlan.sort()
        for index, rlanEntry in inorderRlan:
            rlanRte = trunk.rlanRouteTable.add()
            rlanRte.index = index
            rlanRte.version = fabricVersion
            rlanRte.valid = 1
            #handle setting RLAN correctly
            if args.set_rlan == True:
                #TODO hard-coded good vlan range as 0-12
                offset = inorderRlan.index((index, rlanEntry))
                #print inorderRid[offset][1]["port"], "------", len(inorderRid[offset][1]["port"])
                rlanToPortMap = dict()
                for pt in inorderRid[offset][1]["port"]:
                    if isAccessPort(portsPerSwitch, (nodeId, switchPhyId, pt, 0) ):
                        rlan = getPortId(getConnectedPort(allLinks, (nodeId, switchPhyId, pt, 0)))
                        rlanToPortMap[rlan] = pt

                for rlNum in range(0, portsPerGpu):
                    groupEntry = rlanRte.groupList.add()
                    #TODO fix the group select to point to the right port
                    #TODO handle trunk to trunk connections
                    if rlNum in rlanToPortMap:
                        groupEntry.groupSelect = inorderRid[offset][1]["port"].index( rlanToPortMap[rlNum] )
                        groupEntry.groupSize = 1
                    else:
                        if len(rlanToPortMap) == 0:
                            # on wolf or another pure switsch system we won't have any access ports so rlanToPortMap would be empty
                            groupEntry.groupSelect = 0
                            groupEntry.groupSize = portNum[index]
                        else:
                            groupEntry.groupSelect = portNum[index]
                            groupEntry.groupSize = 1

                    #print "rlan = ", rlNum, "switchPort=", rlanToPortMap[rlNum]
                #bad group 1 for all other vlans
                #for bad_group in range(portNum[index],16):
                #for bad_group in range(12,16):
                    #groupEntry = rlanRte.groupList.add()
                    #groupEntry.groupSelect = portNum[index]
                    #groupEntry.groupSize = 1
        return

    def setSwitchInfo(topologyInfo, gpuPhyIds, fabricNode, portsPerSwitch, nodeId):
        for switchPhyId in topologyInfo[nodeId]["switchPhyIds"]:
            lwswitch = fabricNode.lwswitch.add()
            lwswitch.version = fabricVersion
            switchIndex = topologyInfo[nodeId]["switchPhyIds"].index(switchPhyId)
            ecidStr = "N" + str(nodeId) + "_S" + str(switchIndex)
            lwswitch.ECID = ecidStr.encode('ASCII')
            if args.set_phy_id:
                lwswitch.physicalId = switchPhyId
                if switchPhyId in switchPhyIdToSlotIdMapping:
                    lwswitch.slotId = switchPhyIdToSlotIdMapping[switchPhyId]
            #iterate over all ports in this switch
            for portId, val in iter(topologyInfo[nodeId][switchPhyId].items()):
                #Hopper/Laguna modification go into these two functions( setAccessPort & setTrunkPort)
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
            node_ip = int(nodeId) + 1000
            node_ip_str = "192.168.254." + str(node_ip)
            fabricNode.IPAddress =  node_ip_str.encode('ASCII')
                
        setGpuinfo(topologyInfo, fabricNode, nodeId, gpuPhyIds)

        setSwitchInfo(topologyInfo, gpuPhyIds, fabricNode, portsPerSwitch, nodeId)

        #Add all valid partitions
        #if args.enable_partitions:
            #setPartitionInfo(fabricNode, True, 0x10, 0xc, 0x30, 0x0)
            #setPartitionInfo(fabricNode, True, 0x8, 0x6, 0x0, 0x0)
            #setPartitionInfo(fabricNode, False, 0x10, 0xc, 0x30, 0x0)
            #setPartitionInfo(fabricNode, False, 0x8, 0x6, 0x0, 0x0)
            #setPartitionInfo(fabricNode, False, 0x4, 0x3, 0x0, 0x0)
            #setPartitionInfo(fabricNode, False, 0x2, 0x1, 0x0, 0x0)
            #setPartitionInfo(fabricNode, False, 0x1, 0x0, 0x0, 0x0)
        makePartitions(allPaths, fabricNode)

    if args.topology_name != None:
        fabric.name = args.topology_name
    else:
        fabric.name = "LS_CONFIG"
    fabric.time = strftime("%a %b %d %H:%M:%S %Y", gmtime())
    fabric.arch = topology_pb2.LWSWITCH_ARCH_TYPE_LS10

    for linkIdx, externalPortNum in lwlinkToOsfpPortMapping.items():
        print_verbose("index number" + str(linkIdx))
        print_verbose("port number" + str(externalPortNum))
        portMap = fabric.portMap.add()
        portMap.linkIndex = linkIdx
        portMap.externalPortNum = externalPortNum

    return fabric

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

        if parts[PART_TYPE] == PART_TYPE_SHARED:
            partition.partitionId = parts[PART_ID]

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
            #print "partition id = ", parts[PART_ID]

        intraTrunkConns = set()
        interTrunkConns = set()
        partPaths = dict()

        for key, val in iteritems(allPaths):
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

def getAllGpuPhyIndexes(gpuPhyIds):
        gpuPhyIdToIndex = dict()
        for n,p in gpuPhyIds:
            gpuPhyIdToIndex[(n,p)] = (n , gpuPhyIds.index((n,p)) )
        return gpuPhyIdToIndex

def fixAllLinks(allLinks, gpuPhyIdToIndex):
    tmpAllLinks = dict()
    for l, r in allLinks.iteritems():
        n1, p1, l1, t1 = l
        n2, p2, l2, t2 = r
        if t1 == 1:
            _, p1 = gpuPhyIdToIndex[(n1, p1)]
        if t2 == 1:
            _, p2 = gpuPhyIdToIndex[(n2, p2)]

        tmpAllLinks[(n1, p1, l1, t1)] = (n2, p2, l2, t2)
    return tmpAllLinks

def main():
    #read in the topoloy CSV file
    print_verbose("Reading data: Reading CSV file describing link between all ports") 
    #df = pd.read_csv(args.csv_file, delim_whitespace=True)
    df = pd.read_csv(args.csv_file, sep=r'\,|\t', engine='python')

    print_verbose("Pre-processing: Creating a dictionary of all hardware links") 
    allLinks = getAllLinks(df)

    if args.ring_dateline_switches != None:
        ring_dateline_switches = []
        for id in args.ring_dateline_switches.split(","):
            ring_dateline_switches.append(int(id))

    getAllSwitchOnlyNodes()
    
    print_verbose("Pre-processing: Creating a list of all switch ports") 
    switchPorts = getAllSwitchPorts(allLinks)

    print_verbose("Pre-processing: Creating a list of all gpu ports")
    gpuPorts =  getAllGpuPorts(allLinks)

    print_verbose("Pre-processing: Creating a sorted list of (nodeID, phyID) tuples for GPUs")
    gpuPhyIds = getAllGpuPhyIds(gpuPorts)

    numGpus = len(gpuPhyIds)
    #Laguna supports a max of 2048 endpoints without using the 1K or 2k modes
    if numGpus > DEFAULT_MODE_MAX_ENDPOINTS:
        print ("max GPUs supported = ", DEFAULT_MODE_MAX_ENDPOINTS)
        exit (1)

    global flaIndexStart
    flaIndexStart = lagunaFirstFlaRemapSlot

    if args.use_gpu_indexes == True:
        gpuPhyIdToIndex = getAllGpuPhyIndexes(gpuPhyIds)
        allLinks = fixAllLinks(allLinks, gpuPhyIdToIndex)

        gpuPorts =  getAllGpuPorts(allLinks)
        gpuPhyIds = getAllGpuPhyIds(gpuPorts)

    print_verbose("Pre-processing: Creating a sorted list of (nodeID, phyID) tuples for switches")
    switchPhyIds = getAllSwitchPhyIds(switchPorts)

    print_verbose("Pre-processing: Creating a sorted list of nodeIDs")
    nodeIds = getAllNodeIds(allLinks)

    print_verbose("Pre-processing: Creating dict of two per-switch-port lists 1. access ports 2. trunk ports. Trunk ports have link to access port at corresponding index") 
    portsPerSwitch = getPortsPerSwitch(allLinks, switchPorts)

    print_verbose("Building graph: Adding bi-directional edges in graph for all links")
    G = buildGraph(allLinks, switchPorts, portsPerSwitch)
    print_verbose("buildGraph done " + str(time.time()))

    print_verbose("Finding routes: Finding shortest paths between all pairs of GPU ports")
    allPaths = findRoutes(G, gpuPorts, allLinks, portsPerSwitch)
    print_verbose("findRoutes done " + str(time.time()))

    print_verbose("Finding mapping: Finding mapping info between link index and external port nums")
    lwlinkToOsfpPortMapping, switchPhyIdToSlotIdMapping = readLwlinkToOsfpPortMapping()

    print_verbose("Building topology info: Creating a data structure that has all the data to be used when creating the protobuf message later")
    topologyInfo = buildTopologyInfo(allPaths, nodeIds, gpuPhyIds, switchPhyIds, portsPerSwitch, switchPorts)
    print_verbose("buildTopologyInfo done " + str(time.time()))

    print_verbose("Builing topology: Creating the protobuf message")
    fabric = buildTopologyProtobuf(topologyInfo, nodeIds, allLinks, portsPerSwitch, gpuPhyIds, allPaths, lwlinkToOsfpPortMapping, switchPhyIdToSlotIdMapping)
    print_verbose("buildTopologyProtobuf done " + str(time.time()))

    if args.text_file != None:
        with open (args.text_file, "w") as f:
            f.write(text_format.MessageToString(fabric))

    if args.json_output_file != None:
        with open (args.json_output_file, "w") as f:
            f.write(MessageToJson(fabric))

    if args.write_to_stdout:
        print (fabric)

    if args.binary_file != None:
        with open (args.binary_file, "wb") as f:
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
    parser.add_argument('-n', "--topology-name",  action='store',
                        dest='topology_name',
                        help='Print text file for topology')
    parser.add_argument('-b', "--binary",  action='store',
                        dest='binary_file',
                        help='Print binary file for topology')
    parser.add_argument('-s', "--spray",  action='store_true', default=False,
                        dest='spray',
                        help='Spray over trunk links')
    #parser.add_argument('-r', "--ring",  action='store', default=None,
    parser.add_argument('-r', "--ring",  action='store',
                        dest='ring_dateline',
                        help='Specify dateline for ring')
    parser.add_argument("--ring-dateline-switches",  action='store',
                        dest='ring_dateline_switches',
                        help='List of switches within the dateline node. Default all if this option is not specified')
    parser.add_argument('-l', "--loopback",  action='store_true', default=False,
                        dest='loopback',
                        help='Access port loopback to GPU')
    parser.add_argument("--loopback-from-trunk",  action='store_true', default=False,
                        dest='loopback_from_trunk',
                        help='Trunk port loopback to GPU')
    parser.add_argument("--loopout-from-trunk",  action='store_true', default=False,
                        dest='loopout_from_trunk',
                        help='Trunk port loopout to GPU')
    parser.add_argument('--stdout',  action='store_true', default=False,
                        dest='write_to_stdout',
                        help='Write topology file text to stdout')
    parser.add_argument('-p', "--paths",  action='store',
                    dest='paths_file',
                    help='Input file specifying paths to use instead of callwlating paths')
    parser.add_argument('-m', "--match-gpu-port-num",  action='store_true', default=False,
                        dest='match_gpu_port_num',
                        help='Src port of route should always be same as the dest port')
    parser.add_argument("--unique-access-port",  action='store_true', default=False,
                        dest='unique_access_port',
                        help='Connect each trunk port to exactly one access port intead of the reverse which is the default')
    parser.add_argument("--no-match-even",  action='store_false', default=True,
                        dest='match_even',
                        help='Inter-connect all switch ports when building graph, not just even-even, odd-odd')
    parser.add_argument("--match-target-phy-id",  action='store_false', default=True,
                        dest='match_target_phy_id',
                        help='Set Target ID to same value as physical ID')
    parser.add_argument("--no-phy-id",  action='store_false', default=True,
                        dest='set_phy_id',
                        help='Do not set switch Physical ID in topology file')
    parser.add_argument('-e', "--external-loopback",  action='store_true', default=False,
                        dest='external_loopback',
                        help='The paths need to go though external loopback port')
    parser.add_argument("--set-rlan",  action='store_true', default=False,
                        dest='set_rlan',
                        help='Set rlan to "n" for nth GPU port connected to switch')
    parser.add_argument("--use-gpu-indexes",  action='store_true', default=False,
                        dest='use_gpu_indexes',
                        help='Use gpuIndexes instead of Phy IDs ')
    parser.add_argument('-j', "--json-output",  action='store',
                        dest='json_output_file',
                        help='Print JSON file for topology')
    parser.add_argument("--no-partitions",  action='store_false', default=True,
                        dest='enable_partitions',
                        help="Don't include partition information")
    parser.add_argument("--gpu-ids",  action='store',
                    dest='gpu_id_list',
                    help='List of GPU physical ids to filter in')
    parser.add_argument("--switch-ids",  action='store',
                    dest='switch_id_list',
                    help='List of switch physical ids to filter in')
    parser.add_argument("--partition-file", action='store',
                        dest='partitions_file',
                        help='File specifying the partitions')
    parser.add_argument("--path-lens", action='store',
                        dest='path_lens',
                        help='only paths of specified lengths will be choosen')
    parser.add_argument("--port-map-info",  action='store',
                        dest='port_map_info',
                        help='Input file specifying link index to external port num info')
    parser.add_argument("--switch-nodes", action='store',
                        dest='switch_node_list',
                        help='specify the list of Node IDs associated with LWLink Rack Switches')

    args = parser.parse_args()
    
    # For ring topologies packets need to traverse a switch and hence trunk ports need to be inter-cconnected 
    if args.ring_dateline != None:
        enable_inter_trunk_spray = True

    main()
    exit (0)
