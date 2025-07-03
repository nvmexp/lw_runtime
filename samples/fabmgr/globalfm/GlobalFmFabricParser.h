/*
 *  Copyright 2018-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#pragma once

#include "FMErrorCodesInternal.h"
#include "FMCommonTypes.h"
#include "topology.pb.h"
#include "fabricmanager.pb.h"
#include <set>


typedef struct NodeKeyType
{
    uint32_t nodeId;

    bool operator<(const NodeKeyType &k) const {
        return (nodeId < k.nodeId);
    }
    bool operator==(const NodeKeyType &k) const {
        return (nodeId == k.nodeId);
    }
} NodeKeyType;

typedef struct NodeConfig
{
    uint32_t     nodeId;
    std::string *IPAddress;
    nodeSystemPartitionInfo partitionInfo;
} NodeConfig;

typedef struct SwitchKeyType
{
    uint32_t nodeId;
    uint32_t physicalId;

    bool operator<(const SwitchKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) );
    }
} SwitchKeyType;

typedef struct GpuKeyType
{
    uint32_t nodeId;
    uint32_t physicalId;

    bool operator<(const GpuKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) );
    }
} GpuKeyType;

typedef struct PortKeyType
{
    uint32_t  nodeId;
    uint32_t  portIndex;
    uint32_t physicalId;

    bool operator<(const PortKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex <  k.portIndex) ) );
    }
} PortKeyType;

typedef struct PartitionKeyType
{
    uint32_t nodeId;
    uint32_t partitionId;

    bool operator<(const PartitionKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (partitionId <  k.partitionId) ) );
    }
} PartitionKeyType;

typedef struct ReqTableKeyType
{
    uint32_t  nodeId;
    uint32_t  portIndex;
    uint32_t  physicalId;
    uint32_t  index;

    bool operator<(const ReqTableKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex <  k.portIndex) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex ==  k.portIndex) && (index < k.index) ) );
    }
} ReqTableKeyType;

typedef struct RespTableKeyType
{
    uint32_t  nodeId;
    uint32_t  portIndex;
    uint32_t  physicalId;
    uint32_t  index;

    bool operator<(const RespTableKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex <  k.portIndex) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex ==  k.portIndex) && (index <  k.index) ) );
    }
} RespTableKeyType;

// new routing tables introduced with LimeRock 

typedef struct RmapTableKeyType
{
    uint32_t  nodeId;
    uint32_t  portIndex;
    uint32_t  physicalId;
    uint32_t  index;

    bool operator<(const RmapTableKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex <  k.portIndex) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex ==  k.portIndex) && (index < k.index) ) );
    }
} RmapTableKeyType;

typedef struct RidTableKeyType
{
    uint32_t  nodeId;
    uint32_t  portIndex;
    uint32_t  physicalId;
    uint32_t  index;

    bool operator<(const RidTableKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex <  k.portIndex) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex ==  k.portIndex) && (index <  k.index) ) );
    }
} RidTableKeyType;

typedef struct RlanTableKeyType
{
    uint32_t  nodeId;
    uint32_t  portIndex;
    uint32_t  physicalId;
    uint32_t  index;

    bool operator<(const RlanTableKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex <  k.portIndex) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex ==  k.portIndex) && (index <  k.index) ) );
    }
} RlanTableKeyType;

typedef struct GangedLinkTableKeyType
{
    uint32_t  nodeId;
    uint32_t  portIndex;
    uint32_t  physicalId;
    uint32_t  index;

    bool operator<(const GangedLinkTableKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex <  k.portIndex) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex ==  k.portIndex) && (index <  k.index) ) );
    }
} GangedLinkTableKeyType;


typedef struct MulticastGroupKeyType
{
    uint32_t partitionId;
    uint32_t groupId;

    bool operator<(const MulticastGroupKeyType &k) const {
        return ( (partitionId < k.partitionId) ||
             ( (partitionId ==  k.partitionId) && (groupId <  k.groupId) ) );
    }
} MulticastGroupKeyType;

typedef struct TopologyLWLinkConnEndPoint
{
    uint32_t nodeId;
    uint32_t lwswitchOrGpuId;
    uint32_t portIndex;

    bool operator==(const TopologyLWLinkConnEndPoint& rhs)
    {
        if ( (nodeId == rhs.nodeId) &&
             (lwswitchOrGpuId == rhs.lwswitchOrGpuId) &&
             (portIndex == rhs.portIndex) ) {
            return true;
        } else {
            return false;
        }
    }
} TopologyLWLinkConnEndPoint;

typedef struct TopologyLWLinkConn
{
    TopologyLWLinkConnEndPoint localEnd;
    TopologyLWLinkConnEndPoint farEnd;
    uint8_t  connType; //ACCESS_XX vs TRUNK
} TopologyLWLinkConn;

// list of connections for a node
typedef std::list<TopologyLWLinkConn> TopologyLWLinkConnList;
// map of connections for all the nodes
typedef std::map<uint32, TopologyLWLinkConnList> TopologyLWLinkConnMap;

// parse and populate each LWSwitch trunk port assignment/mask information
typedef std::map<SwitchKeyType, uint64_t> TopologySwitchTrunkPortMaskInfo;

// maps for GPU reachability
// GPU specified by uint32_t targetId can be reached via switch ports specified by uint64_t port mask
typedef std::map<uint32_t, uint64_t> PortsToReachGpu;
typedef std::map<SwitchKeyType, PortsToReachGpu> PortsToReachGpuMap;

typedef std::set<uint32_t> GpuTargetIds;
// Port specified by uint32_t port number can reach a set of GPUs.
typedef std::map<uint32_t, GpuTargetIds> GpusReachableFromPort;
typedef std::map<SwitchKeyType, GpusReachableFromPort> GpusReachableFromPortMap;

class FMFabricParser
{
public:
    typedef enum PruneFabricType {
        PRUNE_SWITCH = 0,
        PRUNE_GPU,
        PRUNE_PARTITION
    } PruneFabricType;

    FMFabricParser( const char *fabricPartitionFileName );
    virtual ~FMFabricParser();

    FMIntReturn_t parseFabricTopology( const char *topoFile );
#ifdef DEBUG
    FMIntReturn_t parseFabricTopologyConf( const char *topoConfFile, lwSwitchArchType arch );
#endif
    void fabricParserCleanup();
    bool isSwtichGpioPresent();

    void disableGpu( GpuKeyType &gpu, lwSwitchArchType arch );
    void disableGpus( std::set<GpuKeyType> &gpus, lwSwitchArchType arch );

    void disableSwitch( SwitchKeyType &key );
    void disableSwitches( std::set<SwitchKeyType> &switches );

    void disablePartition( PartitionKeyType &key );
    void disablePartitions( std::set<PartitionKeyType> &partitions );

    void modifyFabric(PruneFabricType pruneType, lwSwitchArchType arch);
    int getNumDisabledGpus();
    int getNumDisabledSwitches();
    int getNumDisabledPartitions();
    uint32_t getNumGpusInPartition(PartitionKeyType key);
    FMIntReturn_t getSwitchTrunkLinkMask(uint32_t nodeId, uint32_t physicalId, uint64 &trunkLinkMask);

    uint32_t getNodeIndex(uint32_t nodeId);
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    uint32_t getMaxNumNodes();
#endif

    lwSwitchArchType getFabricArch();

    accessPort *getAccessPortInfo( uint32_t nodeId, uint32_t physicalId, uint32_t portNum );
    accessPort *getAccessPortInfoFromCopy( uint32_t nodeId, uint32_t physicalId, uint32_t portNum );
    trunkPort *getTrunkPortInfo( uint32_t nodeId, uint32_t physicalId, uint32_t portNum );
    trunkPort *getTrunkPortInfoFromCopy( uint32_t nodeId, uint32_t physicalId, uint32_t portNum );
    bool getSwitchPortConfig( uint32_t nodeId, uint32_t physicalId, uint32_t portNum, switchPortConfig &portCfg );
    bool isAccessPort( uint32_t nodeId, uint32_t physicalId, uint32_t portNum );
    bool isTrunkPort( uint32_t nodeId, uint32_t physicalId, uint32_t portNum );

    const char *getPlatformId();

    void getConnectedSwitches( SwitchKeyType &key, std::map <SwitchKeyType, uint64_t> &connectedSwitches );
    void getConnectedGpus( SwitchKeyType &key, std::map <GpuKeyType, uint64_t> &connectedGpus );
    void getSharedLWSwitchPartitionsWithSwitch( SwitchKeyType &key, std::set<PartitionKeyType> &partitions );
    void getSharedLWSwitchPartitionsWithGpu( GpuKeyType &key, std::set<PartitionKeyType> &partitions );
    void getSwitchPortMaskToGpu( GpuKeyType &gpuKey, std::map <SwitchKeyType, uint64_t> &connectedSwitches );
    void getSharedLWSwitchPartitionsWithTrunkLinks( uint32_t nodeId, std::set<PartitionKeyType> &partitions );
    void removeSwitchFromSharedLWSwitchPartitions( SwitchKeyType &switchKey, std::map<uint32_t, int> &modifiedPartitions );

    uint32_t getNumConfiguredSharedLwswitchPartition( uint32_t nodeId );
    sharedLWSwitchPartitionInfo *getSharedLwswitchPartitionCfg( uint32_t nodeId, uint32_t partitionId );

    bool isTrunkConnectedToSwitch(PortKeyType &trunkPortKey, SwitchKeyType &switchKey, uint32_t &portNum);

    bool getGpuTargetIdFromKey( GpuKeyType key, uint32_t &targetId );
    bool getGpuKeyFromTargetId( uint32_t targetId, GpuKeyType &key );
    void ridEntryToDstPortMask(ridRouteEntry *ridEntry, uint64_t &dstPortMask);

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    void updateFabricNodeAddressInfo(std::map<uint32_t, std::string> nodeToIpAddrMap, std::set<uint32_t> &degradedNodes);

    std::map<uint32_t, uint32_t> mOsfpPortMap;
    void getLwlinkToOsfpPortMappingInfo();
    uint32_t getOsfpPortNumForLinkIndex(uint32_t linkIndex);
    bool getSlotId(uint32_t physicalId, uint32_t &slotId);
    bool isSlotIdProvided();
#endif

    std::map <NodeKeyType,      NodeConfig *>               NodeCfg;
    std::map <SwitchKeyType,    lwswitch::switchInfo *>     lwswitchCfg;
    std::map <PortKeyType,      lwswitch::switchPortInfo *> portInfo;
    std::map <ReqTableKeyType,  ingressRequestTable  *>     reqEntry;
    std::map <RespTableKeyType, ingressResponseTable *>     respEntry;

    std::map <RmapTableKeyType,   rmapPolicyEntry *>        rmapEntry;
    std::map <RidTableKeyType,    ridRouteEntry *>          ridEntry;
    std::map <RlanTableKeyType,   rlanRouteEntry *>         rlanEntry;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    std::map <RmapTableKeyType,   rmapPolicyEntry *> rmapExtAEntry; // extended range A rmap table
    std::map <RmapTableKeyType,   rmapPolicyEntry *> rmapExtBEntry; // extended range B rmap table
    std::map <RmapTableKeyType,   rmapPolicyEntry *> rmapMcEntry;   // multicast rmap table
#endif

    std::map <GangedLinkTableKeyType, int32_t *>            gangedLinkEntry;
    std::map <GpuKeyType,       lwswitch::gpuInfo * >       gpuCfg;
    std::map <PartitionKeyType, sharedLWSwitchPartitionInfo *> sharedLwswitchPartitionCfg;

    TopologyLWLinkConnMap    lwLinkConnMap;

    // GPU reachability maps derived from the unicast route
    PortsToReachGpuMap portsToReachGpuMap;
    GpusReachableFromPortMap gpusReachableFromPortMap;

private:
    fabric               *mpFabric;
    fabric               *mpFabricCopy;
    nodeSystemPartitionInfo *mpPartitionInfo;

    std::set <PortKeyType> mLoopbackPorts;
    std::set <PartitionKeyType> mDisabledPartitions;
    std::set <uint32_t> mDisabledGpuEndpointIds;
    std::set <SwitchKeyType> mDisabledSwitches;
    std::set <uint32_t> mIlwalidIngressReqEntries;
    std::set <uint32_t> mIlwalidIngressRespEntries;
    TopologySwitchTrunkPortMaskInfo mSwitchTrunkPortMaskInfo;
    std::map <uint32_t, GpuKeyType> mGpuTargetIdTbl;

// routing tables
    std::set <uint32_t> mIlwalidIngressRmapEntries;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    std::set <uint32_t> mIlwalidIngressRmapExtAEntries;
    std::set <uint32_t> mIlwalidIngressRmapExtBEntries;
    std::set <uint32_t> mIlwalidIngressRmapMcEntries;
#endif

    std::set <uint32_t> mIlwalidIngressRidEntries;
    std::set <uint32_t> mIlwalidIngressRlanEntries;

    const char *mFabricPartitionFileName;

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    std::map <uint32_t, uint32_t> mSwitchPhyiscalIdToSlotIdMapping; 
#endif

    FMIntReturn_t parseOneNode( const node &node, int nodeIndex );
    FMIntReturn_t parseOneLwswitch( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId );
    FMIntReturn_t parseAccessPorts( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId );
    FMIntReturn_t parseTrunkPorts( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId );
    FMIntReturn_t parseOnePort(const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId,
                      int portIndex, int isAccess);
    FMIntReturn_t parseOneGpu( const GPU &gpu, uint32_t nodeId, int gpuIndex );

    FMIntReturn_t parseOneIngressReqEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                           const ingressRequestTable &cfg);
    FMIntReturn_t parseOneIngressRespEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                            const ingressResponseTable &cfg);
    FMIntReturn_t parseOneGangedLinkEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                           const gangedLinkTable &cfg);

    FMIntReturn_t parseIngressReqTable( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId );
    FMIntReturn_t parseIngressRespTable( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId );

// new routing tables introduced with LimeRock 

    FMIntReturn_t parseOneIngressRmapEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                            RemapTable remapTable, const rmapPolicyEntry &cfg);
    FMIntReturn_t parseOneIngressRidEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                            const ridRouteEntry &cfg);
    FMIntReturn_t parseOneIngressRlanEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                            const rlanRouteEntry &cfg);

    FMIntReturn_t parseIngressRmapTable( const lwSwitch &lwSwitch, RemapTable remapTable,
                                         uint32_t nodeId, uint32_t physicalId );
    FMIntReturn_t parseIngressRidTable( const lwSwitch &lwSwitch, uint32_t nodeId, uint32_t physicalId );
    FMIntReturn_t parseIngressRlanTable( const lwSwitch &lwSwitch, uint32_t nodeId, uint32_t physicalId );

    FMIntReturn_t parseGangedLinkTable( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId );
    FMIntReturn_t parseLWLinkConnections( const node &node, uint32_t nodeId );
    FMIntReturn_t parseSharedLwswitchPartitions(uint32_t nodeId, nodeSystemPartitionInfo &partitionInfo);

#ifdef DEBUG
    FMIntReturn_t parseDisableGPUConf( std::vector<std::string> &gpus, lwSwitchArchType arch );
    FMIntReturn_t parseDisableSwitchConf( std::vector<std::string> &switches, lwSwitchArchType arch );
    FMIntReturn_t parseLoopbackPorts( std::vector<std::string> &ports );
    FMIntReturn_t parseDisablePartitionConf( std::vector<std::string> &partitions );
#endif

    void modifyOneIngressReqEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                   ingressRequestTable *entry );
    void modifyOneIngressRespEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                    ingressResponseTable *entry );
    void modifyRoutingTable(lwSwitchArchType arch);

// new routing tables introduced with LimeRock 

    void modifyOneIngressRmapEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                    RemapTable remapTable, rmapPolicyEntry *entry, lwSwitchArchType arch );
    void modifyOneIngressRidEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                    ridRouteEntry *entry );
    void modifyOneIngressRlanEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                    rlanRouteEntry *entry );

    void insertIlwalidateIngressRmapEntry(RemapTable remapTable, uint32_t index);

    void pruneOneAccessPort( uint32_t nodeId, uint32_t physicalId,
                             lwSwitch *pLwswitch, uint32_t localportNum );
    void pruneAccessPorts();

    void pruneOneGpu( uint32_t gpuEndpointID );
    void pruneGpus();

    void pruneOneTrunkPort( uint32_t nodeId, uint32_t physicalId,
                            lwSwitch *pLwswitch, uint32_t localportNum );
    void pruneTrunkPorts();

    void pruneOneSwitch( SwitchKeyType &key );
    void pruneSwitches();

    void pruneOneSharedLWSwitchPartition( uint32_t nodeId, uint32_t partitionId );
    void pruneSharedLWSwitchPartitions();

    int checkLWLinkConnExists( TopologyLWLinkConnList &connList,
                               TopologyLWLinkConn newConn );

    void copyTrunkPortConnInfo( trunkPort &trunkPort, uint32_t nodeId,
                                uint32_t physicalId,int portIndex, 
                                TopologyLWLinkConnList &connList );

    void copyAccessPortConnInfo( accessPort &accessPort, uint32_t nodeId,
                                 uint32_t physicalId,int portIndex, 
                                 TopologyLWLinkConnList &connList );

    void updateSwitchTrunkPortMaskInfo(TopologyLWLinkConn &connInfo);

    void removeOneSwitchFromSharedPartition( SwitchKeyType &key,  sharedLWSwitchPartitionInfo *partInfo );

    FMIntReturn_t parseLwstomSharedFabricPartition( void );
    FMIntReturn_t validateLwstomSharedFabricPartition( void );

    void  constructPortsToReachGpuMap( uint32_t nodeId, uint32_t physicalId,
                                       uint32_t localPortNum, const ridRouteEntry &cfg );
    void  consructGpusReachableFromPortMap( uint32_t nodeId, uint32_t physicalId,
                                            uint32_t localPortNum, const ridRouteEntry &cfg );

    void  constructPortsToReachGpuMap( uint32_t nodeId, uint32_t physicalId,
                                       uint32_t localPortNum, const accessPort &access );
    void  consructGpusReachableFromPortMap( uint32_t nodeId, uint32_t physicalId,
                                            uint32_t localPortNum, const accessPort &access );

};

