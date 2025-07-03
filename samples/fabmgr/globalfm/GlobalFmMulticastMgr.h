/*
 *  Copyright 2020-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#pragma once

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

/*****************************************************************************/
/*  Implement all the globalFM side Multicast related interfaces/methods.    */
/*****************************************************************************/

#include "FMCommonTypes.h"
#include "lw_fm_types.h"
#include "fabricmanagerHA.pb.h"
#include "GlobalFmFabricParser.h"

typedef struct {
    uint32_t portNum;
    uint32_t vcHop;
} MulticastPort;

typedef std::vector<MulticastPort> MulticastPorts;

typedef struct {
    MulticastPorts  ports;
    bool     replicaValid;
    uint32_t replicaOffset; // if replicaValid is true, the offset of replica in the portList
} MulticastPortList;

typedef std::vector<MulticastPortList> MulticastSprays; // sprays to multiple mcast port list

typedef struct {
    uint32_t        portNum;
    bool            extendedTable;
    uint32_t        index;                 // index of mcid table if extendedTable is false
                                           // index of extended mcid table if extendedTable is true
    MulticastSprays sprays;                // sprays to multiple egress port lists
    bool            extendedTablePtrValid; // true if the this entry is extended by extendedTablePtr
    uint32_t        extendedTablePtr;
} PortMcIdTableEntry;

typedef struct {
    uint32_t             portNum;
    PortMcIdTableEntry   mcIdEntry;         // port McId table keyed by portNum
    PortMcIdTableEntry   extendedMcIdEntry; // port extended McId table keyed by portNum
} PortMulticastInfo;

typedef std::map<PortKeyType, PortMulticastInfo>    PortMulticastTable;   // a map of multicast tables on all ilwolved ports
typedef std::map<SwitchKeyType, PortMulticastTable> SwitchMulticastTable; // a map of multicast tables on all ilwolved switches
typedef std::map<uint32_t, SwitchMulticastTable>    NodeMulticastTable;   // a map of multicast tables on all ilwolved nodes

typedef struct {
    uint32_t partitionId;
    uint32_t mcId;                        // multicast table index, the same as groupId
    uint32_t parentMcId;                  // this group is a subgroup of a parent group
                                          // if the is group does not have a parent group, parentMcId is the same as mcId
    bool     reflectiveMode;              // true if reflective mode is enabled
    bool     excludeSelf;                 // true the requesting GPU should be excluded in the multicast tree
    bool     noDynRsp;                    // no dynamic alt selection on MC responses
    std::set<GpuKeyType> gpuList;         // the GPUs that are participating in this group
    GpuKeyType           primaryReplica;  // the GPU that is assigned as primaryReplica
    NodeMulticastTable   multicastTable;  // multicast tables on all switches ilwolved in the multicast tree
} MulticastGroupInfo;

typedef std::map<MulticastGroupKeyType, MulticastGroupInfo *> MulticastGroupTable;

class GlobalFmMulticastMgr
{
    friend class GlobalFMCommandServer;

public:
    GlobalFmMulticastMgr(GlobalFabricManager *pGfm);

    ~GlobalFmMulticastMgr();

    FMIntReturn_t allocateMulticastGroup(uint32_t partitionId, uint32_t &groupId);
    FMIntReturn_t freeMulticastGroup(uint32_t partitionId, uint32_t groupId);
    void freeMulticastGroups(uint32_t partitionId);
    void getAvailableMulticastGroups(uint32_t partitionId, std::list<uint32_t> &groupIds);

    FMIntReturn_t setMulticastGroup(uint32_t partitionId,
                                    uint32_t groupId,
                                    bool reflectiveMode,
                                    bool excludeSelf,
                                    std::set<GpuKeyType> &gpus);

    FMIntReturn_t setMulticastGroup(uint32_t partitionId,
                                    uint32_t groupId,
                                    bool reflectiveMode,
                                    bool excludeSelf,
                                    std::set<GpuKeyType> &gpus,
                                    GpuKeyType primaryReplica);

    FMIntReturn_t getMulticastGroupBaseAddress(uint32_t groupId, uint64_t &multicastAddrBase);

    FMIntReturn_t getMulticastGroupBaseAddrAndRange(uint32_t groupId, uint64_t &multicastAddrBase,
                                                    uint64_t &multicastAddrRange);

    void dumpMulticastGroup( uint32_t partitionId, uint32_t groupId, std::stringstream &outStr );

    uint32_t getMaxNumMulitcastGroups(void) { return mMaxNumMcGroups; };

    void handleMessage(lwswitch::fmMessage *pFmMessage);

private:
    GlobalFabricManager *mpGfm;
    LWOSCriticalSection mLock;

    uint32_t mMaxNumMcGroups;
    bool *pMcGroupAllocated;

    bool mEnableSpray;
    bool mNoDynRsp;

    MulticastGroupTable mGroupTable;

    void clearPortMcIdEntry ( PortMcIdTableEntry &entry );
    void clearMulticastGroup( MulticastGroupInfo *groupInfo );
    FMIntReturn_t constructMulticastTree(  MulticastGroupInfo *groupInfo );
    FMIntReturn_t configMulticastRoutes(MulticastGroupInfo *groupInfo, bool freeGroup);
    void handleMulticastConfigError(MulticastGroupInfo *groupInfo);

    MulticastGroupInfo *getFreeMulticastGroup(uint32_t partitionId);
    void freeMulticastGroupInfo(MulticastGroupInfo *groupInfo);

    void dumpMcIdEntry(PortMcIdTableEntry &entry, std::stringstream &outStr);
    void dumpMulticastGroupInfo(MulticastGroupInfo *groupInfo, std::stringstream &outStr);

    void getAllSwitchActiveLinkMask( uint32_t partitionId, std::map<SwitchKeyType, uint64_t> &switchActiveLinkMask );
    bool sourcePortCheck(MulticastGroupInfo *groupInfo, PortKeyType srcPortKey, bool isSrcPortAccess,
                         GpuTargetIds &srcGpuSet, GpuTargetIds &dstGpuSet);

    bool getDstPorts(MulticastGroupInfo *groupInfo, SwitchKeyType switchKey, PortKeyType srcPortKey,
                     uint64_t switchActiveLinkMask,GpuTargetIds &dstGpuSet, uint64_t &primaryReplicaDstPorts,
                     std::set<uint64_t> &dstPortGroups);

    void setPortMulticastTbl(MulticastGroupInfo *groupInfo, SwitchKeyType switchKey, uint32_t srcPortNum,
                             uint64_t &primaryReplicaDstPorts, std::set<uint64_t> &dstPortGroups);

    void portMaskTolist(uint64_t portMask, std::vector<uint32_t> &portList);

    bool areGPUsReachableOnPort(MulticastGroupInfo *groupInfo, PortKeyType portKey);

    void getDstPortMaskToGpu(PortKeyType srcPortKey, uint32_t targetId, uint64_t &dstPortMask);

    MulticastGroupInfo *getMulticastGroup(uint32_t partitionId, uint32_t groupId);
    MulticastGroupTable::iterator getMulticastGroupIterator(uint32_t partitionId, uint32_t groupId);

    FMIntReturn_t sendGroupCreateRspMsg(uint64_t mcHandle, uint32_t  createNodeId,
                                        lwswitch::configStatus rspCode);
    FMIntReturn_t sendGroupBindRspMsg(uint64_t mcHandle, uint32 createNodeId, uint32_t bindNodeId,
                                      lwswitch::configStatus rspCode);
    FMIntReturn_t sendGroupSetupCompleteReqMsg(uint64_t mcHandle, uint32_t createNodeId,
                                               MulticastGroupInfo *groupInfo,
                                               lwswitch::configStatus errCode);
    FMIntReturn_t sendGroupReleaseRspMsg(uint64_t mcHandle, uint32 createNodeId, uint32_t releaseNodeId,
                                         lwswitch::configStatus rspCode);
    FMIntReturn_t sendGroupReleaseCompleteReqMsg(uint64_t mcHandle, uint32_t createNodeId,
                                                 MulticastGroupInfo *groupInfo,
                                                 lwswitch::configStatus errCode);

    void handleGroupCreateReqMsg(lwswitch::fmMessage *pFmMessage);
    void handleGroupBindReqMsg(lwswitch::fmMessage *pFmMessage);
    void handleGroupSetupCompleteAckMsg(lwswitch::fmMessage *pFmMessage);
    void handleGroupReleaseReqMsg(lwswitch::fmMessage *pFmMessage);
    void handleGroupReleaseCompleteAckMsg(lwswitch::fmMessage *pFmMessage);
};
#endif
